#!/usr/bin/env python3
import argparse
import csv
import math
import os
import random
import shutil
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Callable, DefaultDict, Optional, Sequence

import pandas as pd
import torch
from PIL import Image
from tabulate import tabulate
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm

from leaffliction.Augmentation import (
    apply_crop,
    apply_distortion,
    apply_flip,
    apply_rotation,
    apply_shear,
)
from leaffliction.distribution import IMG_EXTENSIONS

print_lock = Lock()
DEFAULT_IMAGE_SIZE = (256, 256)
DEFAULT_NORMALIZE_MEAN = (0.485, 0.456, 0.406)
DEFAULT_NORMALIZE_STD = (0.229, 0.224, 0.225)

def thread_safe_print(msg: str) -> None:
    with print_lock:
        print(msg)


def build_default_transforms() -> T.Compose:
    return T.Compose([
        T.Resize(DEFAULT_IMAGE_SIZE),
        T.ToTensor(),
        T.Normalize(DEFAULT_NORMALIZE_MEAN, DEFAULT_NORMALIZE_STD),
    ])


def _process_path(p: Path, exts: set[str], base_dir: Path) -> Optional[dict[str, str]]:
    if p.is_file() and p.suffix.lower() in exts:
        parts = p.parts
        split = None
        for part in parts:
            if part in ['train', 'eval', 'test']:
                split = part
                break
        
        group_stem = p.stem.split("_g")[0] if "_g" in p.stem else p.stem.split("_")[0]
        resolved_path = p.resolve()
        try:
            relative_path = resolved_path.relative_to(base_dir)
            stored_path = str(relative_path)
        except ValueError:
            stored_path = str(resolved_path)
        
        return {
            "path": stored_path,
            "name": p.name,
            "class": p.parent.name,
            "stem": p.stem,
            "group": group_stem,
            "split": split if split else "unknown",
        }
    return None


def get_augmentation_transforms() -> list[tuple[str, Callable[[Image.Image], Image.Image]]]:
    return [
        ("flip", apply_flip),
        ("rotate", apply_rotation),
        ("crop", apply_crop),
        ("shear", apply_shear),
        ("distortion", apply_distortion)
    ]


def process_single_image_augmentation(args: tuple[str, str, str, int]) -> Optional[tuple[str, list[str]]]:
    src_path, output_dir, group_id, num_transforms = args
    
    try:
        with Image.open(src_path) as img:
            original_output = os.path.join(output_dir, f"{group_id}_original.jpg")
            img.save(original_output)
            
            transforms = get_augmentation_transforms()
            transform_paths = []
            
            for tname, tfunc in transforms[:num_transforms]:
                timg = tfunc(img)
                transform_output = os.path.join(
                    output_dir, f"{group_id}_{tname}.jpg"
                )
                timg.save(transform_output)
                transform_paths.append(transform_output)
            
            return (original_output, transform_paths)
    except Exception as e:
        thread_safe_print(f"Warning: failed to process {src_path}: {e}")
        return None




def build_csv_from_directory(root_dir: str | Path, output_csv: str | Path, max_workers: int = 8) -> pd.DataFrame:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif"}
    root_path = Path(root_dir)
    base_dir = root_path.resolve().parent
    paths = list(root_path.rglob("*"))
    rows = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for result in executor.map(lambda p: _process_path(p, exts, base_dir), paths):
            if result:
                rows.append(result)
    
    if not rows:
        raise ValueError(f"No images found in {root_dir}")
    
    df = pd.DataFrame.from_records(rows)
    df.to_csv(output_csv, index=False)
    print(f"CSV saved to {output_csv}")
    return df




def collect_images_by_class(root_path: str) -> dict[str, list[str]]:
    class_images: DefaultDict[str, list[str]] = defaultdict(list)
    
    if not os.path.isdir(root_path):
        print(f"Error: '{root_path}' is not a directory.")
        return class_images
    
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.lower().endswith(IMG_EXTENSIONS):
                file_path = os.path.join(root, file)
                class_name = os.path.basename(root)
                class_images[class_name].append(file_path)
    
    return class_images

def augment_class(class_name: str, image_paths: Sequence[str], target_total_images: int,
                   output_dir: str, transforms_per_image: int = 5, max_workers: int = 4) -> list[tuple[str, list[str]]]:
    class_output_dir = os.path.join(output_dir, class_name)
    os.makedirs(class_output_dir, exist_ok=True)
    
    originals = list(image_paths)
    random.shuffle(originals)
    num_originals = len(originals)
    
    images_per_group = 1 + transforms_per_image
    num_groups_needed = math.ceil(target_total_images / images_per_group)
    
    if num_groups_needed <= num_originals:
        selected_originals = originals[:num_groups_needed]
    else:
        selected_originals = []
        for i in range(num_groups_needed):
            selected_originals.append(originals[i % num_originals])
    
    full_groups = target_total_images // images_per_group
    remainder = target_total_images % images_per_group
    
    thread_safe_print(
        f"  Class '{class_name}': Creating {num_groups_needed} groups "
        f"({target_total_images} images)"
    )
    
    jobs = []
    for idx, src_path in enumerate(selected_originals):
        base_name = os.path.splitext(os.path.basename(src_path))[0]
        group_id = f"{base_name}_g{idx}"
        
        if idx < full_groups:
            num_transforms = transforms_per_image
        elif idx == full_groups:
            num_transforms = max(0, remainder - 1)
        else:
            break
        
        jobs.append((src_path, class_output_dir, group_id, num_transforms))
    
    image_groups = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_single_image_augmentation, job)
            for job in jobs
        ]
        
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                image_groups.append(result)
    
    total_images = sum(1 + len(tpaths) for _, tpaths in image_groups)
    thread_safe_print(
        f"    Created {len(image_groups)} groups ({total_images} images)"
    )
    
    return image_groups


def split_dataset(class_groups: dict[str, list[tuple[str, list[str]]]], output_base_dir: str,
                   train_ratio: float = 0.7, eval_ratio: float = 0.15, test_ratio: float = 0.15) -> dict[str, int]:
    train_dir = os.path.join(output_base_dir, "train")
    eval_dir = os.path.join(output_base_dir, "eval")
    test_dir = os.path.join(output_base_dir, "test")
    
    stats = {"train": 0, "eval": 0, "test": 0}
    
    for class_name, groups in class_groups.items():
        random.shuffle(groups)
        
        total_groups = len(groups)
        train_count = int(total_groups * train_ratio)
        eval_count = int(total_groups * eval_ratio)
        
        train_groups = groups[:train_count]
        eval_groups = groups[train_count:train_count + eval_count]
        test_groups = groups[train_count + eval_count:]
        
        for split_dir, split_groups, split_name in [
            (train_dir, train_groups, "train"),
            (eval_dir, eval_groups, "eval"),
            (test_dir, test_groups, "test")
        ]:
            class_split_dir = os.path.join(split_dir, class_name)
            os.makedirs(class_split_dir, exist_ok=True)
            
            for original_path, transform_paths in split_groups:
                try:
                    shutil.copy2(original_path, class_split_dir)
                    stats[split_name] += 1
                except Exception as e:
                    print(f"Warning copying original {original_path}: {e}")
                
                for transform_path in transform_paths:
                    try:
                        shutil.copy2(transform_path, class_split_dir)
                        stats[split_name] += 1
                    except Exception as e:
                        print(f"Warning copying transform {transform_path}: {e}")
    
    return stats

class ImageMLP(nn.Module):
    def __init__(self, in_shape: tuple[int, int, int], num_classes: int, p_drop: float = 0.2) -> None:
        super().__init__()
        self.in_shape = in_shape
        self.num_classes = num_classes
        self.p_drop = p_drop
        c, h, w = in_shape
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.MaxPool2d(2, stride=2),

            nn.Flatten(),
            nn.Linear(8*8*256, 200),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(200, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CSVDataset(Dataset):
    def __init__(
        self,
        mask: str,
        csv_path: str | Path,
        root: str | Path = ".",
        transforms: Optional[Any] = None,
        label_map: Optional[dict[str, int]] = None,
    ) -> None:
        self.root = Path(root)
        self.rows = []
        all_classes: set[str] = set()

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                all_classes.add(r["class"])
                if r["split"].lower() == mask.lower():
                    self.rows.append({"file": r["path"], "label": r["class"]})

        if not self.rows:
            raise ValueError(f"No rows found for split='{mask}' in {csv_path}")
        
        if label_map is None:
            classes = sorted(all_classes)
            self.class_to_idx = {c: i for i, c in enumerate(classes)}
        else:
            self.class_to_idx = dict(label_map)
            unknown_labels = sorted({r["label"] for r in self.rows} - set(self.class_to_idx))
            if unknown_labels:
                raise ValueError(
                    f"Found labels not present in label_map for split='{mask}': {unknown_labels}"
                )
        
        if transforms is None:
            self.transforms = build_default_transforms()
        else:
            self.transforms = transforms

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict[str, Any]:
        row = self.rows[i]
        path = Path(row["file"])
        if not path.is_absolute():
            path = self.root / path
        path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(f"Missing image: {path}")

        with Image.open(path) as im:
            im = im.convert("RGB")
        x = self.transforms(im)
        y = torch.tensor(self.class_to_idx[row["label"]], dtype=torch.long)
        return {"image": x, "y": y, "label": row["label"], "path": str(path)}


def calculate_metrics(
    model: nn.Module, dataloader: DataLoader, device: torch.device | str = "cpu"
) -> tuple[float, float, float]:
    _, accuracy, sensitivity, specificity = compute_loss_and_metrics(
        model, dataloader, device
    )
    return accuracy, sensitivity, specificity


def compute_loss(model: nn.Module, dataloader: DataLoader, device: torch.device | str = "cpu") -> float:
    loss, _, _, _ = compute_loss_and_metrics(model, dataloader, device)
    return loss


def compute_loss_and_metrics(
    model: nn.Module, dataloader: DataLoader, device: torch.device | str = "cpu"
) -> tuple[float, float, float, float]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    criterion = nn.CrossEntropyLoss()
    all_true = []
    all_pred = []

    with torch.no_grad():
        for batch in dataloader:
            x = batch["image"].to(device)
            y = batch["y"].to(device)
            logits = model(x)
            batch_loss = criterion(logits, y)
            total_loss += batch_loss.item() * x.size(0)
            total_samples += x.size(0)

            y_pred = torch.argmax(logits, dim=1)
            all_true.append(y)
            all_pred.append(y_pred)

    if total_samples == 0 or not all_true:
        return float("nan"), 0.0, 0.0, 0.0

    loss = total_loss / total_samples
    y_true = torch.cat(all_true).to(torch.long)
    y_pred = torch.cat(all_pred).to(torch.long)

    n_classes = int(max(y_true.max().item(), y_pred.max().item()) + 1)
    flat_index = y_true * n_classes + y_pred
    confusion = torch.bincount(
        flat_index,
        minlength=n_classes * n_classes
    ).reshape(n_classes, n_classes).to(torch.float32)

    total = confusion.sum().item()
    accuracy = confusion.diag().sum().item() / max(1.0, total)

    tp = confusion.diag()
    fn = confusion.sum(dim=1) - tp
    fp = confusion.sum(dim=0) - tp
    tn = confusion.sum() - (tp + fn + fp)

    sensitivity = (tp / torch.clamp(tp + fn, min=1.0)).mean().item()
    specificity = (tn / torch.clamp(tn + fp, min=1.0)).mean().item()
    return loss, accuracy, sensitivity, specificity


def write_metrics_history(
    metrics_history: list[dict[str, int | float]], metrics_csv: str | Path
) -> None:
    metrics_path = Path(metrics_csv)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics_history).to_csv(metrics_path, index=False)


def evaluate_model(
    model: nn.Module,
    eval_loader: DataLoader,
    device: torch.device | str = "cpu",
) -> dict[str, float]:
    e_loss, e_acc, e_sens, e_spec = compute_loss_and_metrics(
        model, eval_loader, device
    )
    
    headers = ["Metric", "Evaluation (eval split)"]
    table = [
        ["Loss", f"{e_loss:.4f}"],
        ["Accuracy", f"{e_acc:.4f}"],
        ["Sensitivity (macro avg)", f"{e_sens:.4f}"],
        ["Specificity (macro avg)", f"{e_spec:.4f}"],
    ]
    print(tabulate(table, headers=headers, tablefmt="fancy_grid"))

    return {
        "eval_accuracy": e_acc,
        "eval_loss": e_loss,
        "eval_sensitivity_macro": e_sens,
        "eval_specificity_macro": e_spec,
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    device: torch.device | str,
    epochs: int = 10,
    lr: float = 1e-3,
    save_every: int = 2,
    checkpoint_dir: str = "./checkpoints",
    metrics_csv: str | Path = "training_metrics.csv",
    target_accuracy: float = 0.92,
) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    class_to_idx = dict(getattr(train_loader.dataset, "class_to_idx", {}))
    if not class_to_idx:
        raise ValueError(
            "Training dataset must expose class_to_idx for portable checkpoint metadata."
        )

    metadata = {
        "class_to_idx": class_to_idx,
        "idx_to_class": {idx: class_name for class_name, idx in class_to_idx.items()},
        "model": {
            "name": type(model).__name__,
            "in_shape": getattr(model, "in_shape", None),
            "num_classes": getattr(model, "num_classes", len(class_to_idx)),
            "p_drop": getattr(model, "p_drop", None),
        },
        "preprocess": {
            "image_size": list(DEFAULT_IMAGE_SIZE),
            "normalize_mean": list(DEFAULT_NORMALIZE_MEAN),
            "normalize_std": list(DEFAULT_NORMALIZE_STD),
        },
    }
    metrics_history: list[dict[str, int | float]] = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total_samples = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            x = batch["image"].to(device, non_blocking=True)
            y = batch["y"].to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()

            running_loss += loss.item() * x.size(0)
            total_samples += x.size(0)

        avg_train_loss = running_loss / max(1, total_samples)
        print(f"Epoch {epoch+1} train loss: {avg_train_loss:.4f}")
        
        eval_metrics = evaluate_model(model, eval_loader, device)
        acc = eval_metrics["eval_accuracy"]
        print(f"Epoch {epoch+1} eval accuracy: {acc:.4f}")
        metrics_history.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            **eval_metrics,
        })
        write_metrics_history(metrics_history, metrics_csv)

        if (epoch + 1) % save_every == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"model_epoch{epoch+1}.pt")
            torch.save(model, ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")
            artifact_path = os.path.join(
                checkpoint_dir, f"model_epoch{epoch+1}.artifact.pt"
            )
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "epoch": epoch + 1,
                    "eval_accuracy": acc,
                    "metadata": metadata,
                },
                artifact_path
            )
            print(f"Portable checkpoint saved: {artifact_path}")

        if acc > target_accuracy:
            ckpt_path = os.path.join(checkpoint_dir, "model_best.pt")
            torch.save(model, ckpt_path)
            print(f"Early stop at epoch {epoch+1}, best model: {ckpt_path}")
            artifact_path = os.path.join(checkpoint_dir, "model_best.artifact.pt")
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "epoch": epoch + 1,
                    "eval_accuracy": acc,
                    "metadata": metadata,
                },
                artifact_path
            )
            print(f"Portable best checkpoint saved: {artifact_path}")
            break


def create_zip_archive(
    augmented_dir: str,
    checkpoint_dir: str,
    output_zip: str,
    metrics_csv: str | Path | None = None,
    dataset_csv: str | Path | None = None,
) -> None:
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(augmented_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, os.path.dirname(augmented_dir))
                zipf.write(file_path, arcname)
        
        for root, dirs, files in os.walk(checkpoint_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, os.path.dirname(checkpoint_dir))
                zipf.write(file_path, arcname)

        if metrics_csv is not None and os.path.isfile(metrics_csv):
            zipf.write(metrics_csv, os.path.basename(metrics_csv))

        if dataset_csv is not None and os.path.isfile(dataset_csv):
            zipf.write(dataset_csv, os.path.basename(dataset_csv))
    
    print(f"Archive created: {output_zip}")
