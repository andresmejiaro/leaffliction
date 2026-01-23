#!/usr/bin/env python3
import argparse
import os
import shutil
import zipfile
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torch import nn
from PIL import Image
import csv
from pathlib import Path
from tqdm import tqdm
from tabulate import tabulate
from shared_utils import (
    collect_images_by_class,
    augment_class,
    split_dataset,
    build_csv_from_directory
)
from typing import Any, Optional


class ImageMLP(nn.Module):
    def __init__(self, in_shape: tuple[int, int, int], num_classes: int, p_drop: float = 0.2) -> None:
        super().__init__()
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

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r["split"].lower() == mask.lower():
                    self.rows.append({"file": r["path"], "label": r["class"]})

        if not self.rows:
            raise ValueError(f"No rows found for split='{mask}' in {csv_path}")
        
        if label_map is None:
            classes = sorted({r["label"] for r in self.rows})
            self.class_to_idx = {c: i for i, c in enumerate(classes)}
        else:
            self.class_to_idx = dict(label_map)
        
        if transforms is None:
            self.transforms = T.Compose([
                T.Resize((256, 256)),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        else:
            self.transforms = transforms

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict[str, Any]:
        row = self.rows[i]
        path = Path(row["file"]).resolve()
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
    model.eval()
    all_true = []
    all_pred = []

    with torch.no_grad():
        for batch in dataloader:
            x = batch["image"].to(device)
            y_true = batch["y"].to(device)
            logits = model(x)
            y_pred = torch.argmax(logits, dim=1)
            all_true.append(y_true)
            all_pred.append(y_pred)

    if not all_true:
        return 0.0, 0.0, 0.0

    y_true = torch.cat(all_true)
    y_pred = torch.cat(all_pred)

    tp = ((y_true == 1) & (y_pred == 1)).sum().item()
    tn = ((y_true == 0) & (y_pred == 0)).sum().item()
    fp = ((y_true == 0) & (y_pred == 1)).sum().item()
    fn = ((y_true == 1) & (y_pred == 0)).sum().item()

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / max(1, total)
    sensitivity = tp / max(1, (tp + fn))
    specificity = tn / max(1, (tn + fp))

    return accuracy, sensitivity, specificity


def compute_loss(model: nn.Module, dataloader: DataLoader, device: torch.device | str = "cpu") -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in dataloader:
            x = batch["image"].to(device)
            y = batch["y"].to(device)
            logits = model(x)
            batch_loss = criterion(logits, y)
            total_loss += batch_loss.item() * x.size(0)
            total_samples += x.size(0)

    return total_loss / total_samples if total_samples > 0 else float("nan")


def evaluate_model(
    model: nn.Module,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    device: torch.device | str = "cpu",
) -> float:
    t_loss = compute_loss(model, train_loader, device)
    e_loss = compute_loss(model, eval_loader, device)
    t_acc, t_sens, t_spec = calculate_metrics(model, train_loader, device)
    e_acc, e_sens, e_spec = calculate_metrics(model, eval_loader, device)
    
    headers = ["Metric", "Train", "Eval"]
    table = [
        ["Loss", f"{t_loss:.4f}", f"{e_loss:.4f}"],
        ["Accuracy", f"{t_acc:.4f}", f"{e_acc:.4f}"],
        ["Sensitivity", f"{t_sens:.4f}", f"{e_sens:.4f}"],
        ["Specificity", f"{t_spec:.4f}", f"{e_spec:.4f}"],
    ]
    print(tabulate(table, headers=headers, tablefmt="fancy_grid"))
    
    return e_acc


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    device: torch.device | str,
    epochs: int = 10,
    lr: float = 1e-3,
    save_every: int = 2,
    checkpoint_dir: str = "./checkpoints",
    target_accuracy: float = 0.99,
) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

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
        
        acc = evaluate_model(model, train_loader, eval_loader, device)
        print(f"Epoch {epoch+1} eval accuracy: {acc:.4f}")

        if (epoch + 1) % save_every == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"model_epoch{epoch+1}.pt")
            torch.save(model, ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")

        if acc > target_accuracy:
            ckpt_path = os.path.join(checkpoint_dir, "model_best.pt")
            torch.save(model, ckpt_path)
            print(f"Early stop at epoch {epoch+1}, best model: {ckpt_path}")
            break


def create_zip_archive(augmented_dir: str, checkpoint_dir: str, output_zip: str) -> None:
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
    
    print(f"Archive created: {output_zip}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train disease classification model"
    )
    parser.add_argument(
        "directory",
        type=str,
        help="Root directory containing images organized by class"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: 0.001)"
    )
    
    args = parser.parse_args()
    
    augmented_dir = "augmented_directory"
    checkpoint_dir = "checkpoints"
    csv_path = "dataset.csv"
    output_zip = "trained_model.zip"
    
    if os.path.exists(augmented_dir):
        shutil.rmtree(augmented_dir)
    os.makedirs(augmented_dir, exist_ok=True)
    
    print("Step 1: Collecting images...")
    class_images = collect_images_by_class(args.directory)
    if not class_images:
        print("No images found.")
        return
    
    print("Step 2: Augmenting images...")
    transforms_per_image = 5
    max_class_size = max(len(images) for images in class_images.values())
    target_images = max_class_size * (1 + transforms_per_image)
    
    temp_dir = os.path.join(augmented_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    class_groups = {}
    for class_name, image_paths in class_images.items():
        groups = augment_class(
            class_name, image_paths, target_images,
            temp_dir, transforms_per_image=transforms_per_image
        )
        class_groups[class_name] = groups
    
    print("Step 3: Splitting dataset...")
    split_dataset(class_groups, augmented_dir)
    shutil.rmtree(temp_dir)
    
    print("Step 4: Building CSV...")
    build_csv_from_directory(augmented_dir, csv_path)
    
    print("Step 5: Training model...")
    training_data = CSVDataset("train", csv_path, root=".")
    evaluation_data = CSVDataset("eval", csv_path, root=".")
    
    use_pin = torch.cuda.is_available()
    train_loader = DataLoader(
        training_data, batch_size=16, shuffle=True,
        num_workers=2, pin_memory=use_pin
    )
    eval_loader = DataLoader(
        evaluation_data, batch_size=16, shuffle=False,
        num_workers=2, pin_memory=use_pin
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImageMLP(
        in_shape=(3, 256, 256),
        num_classes=len(training_data.class_to_idx)
    ).to(device)
    
    train_model(
        model, train_loader, eval_loader, device,
        epochs=args.epochs, lr=args.lr, checkpoint_dir=checkpoint_dir
    )
    
    print("Step 6: Creating archive...")
    create_zip_archive(augmented_dir, checkpoint_dir, output_zip)
    
    print("Training complete!")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
