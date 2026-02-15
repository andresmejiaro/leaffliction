from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from leaffliction.predict import (
    build_inference_transform,
    load_model_bundle,
    resolve_model_path,
)

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif")


class DirectoryDataset(Dataset):
    def __init__(self, samples: list[tuple[str, str]], transform: Any) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_path, label = self.samples[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
        image_tensor = self.transform(image)
        return {"image": image_tensor, "label": label, "path": image_path}


def collect_labeled_images(directory: str | Path) -> list[tuple[str, str]]:
    root_dir = Path(directory)
    if not root_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {root_dir}")

    samples: list[tuple[str, str]] = []
    for image_path in root_dir.rglob("*"):
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() not in IMG_EXTENSIONS:
            continue
        class_name = image_path.parent.name
        samples.append((str(image_path.resolve()), class_name))

    if not samples:
        raise ValueError(f"No images found in directory: {root_dir}")
    return samples


def _resolve_idx_to_class(
    class_names: list[str] | None,
    idx_to_class: dict[int, str],
) -> dict[int, str]:
    if class_names:
        return {idx: class_name for idx, class_name in enumerate(class_names)}
    return idx_to_class


def evaluate_accuracy_on_directory(
    directory: str | Path,
    model_path: str = "checkpoints/model_best.artifact.pt",
    class_names: list[str] | None = None,
    batch_size: int = 32,
    num_workers: int = 2,
    requested_device: str | None = None,
) -> dict[str, Any]:
    samples = collect_labeled_images(directory)
    resolved_model_path = resolve_model_path(model_path)
    model, device, idx_to_class, metadata = load_model_bundle(
        resolved_model_path,
        requested_device=requested_device,
    )
    idx_to_class = _resolve_idx_to_class(class_names, idx_to_class)
    if not idx_to_class:
        raise ValueError(
            "Class index mapping unavailable. Use an artifact checkpoint or pass --classes."
        )

    transform, _, _ = build_inference_transform(metadata)
    dataset = DirectoryDataset(samples, transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    total = 0
    correct = 0
    per_class_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"correct": 0, "total": 0}
    )

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            true_labels = batch["label"]
            logits = model(images)
            predicted_indices = torch.argmax(logits, dim=1).tolist()

            for predicted_index, true_label in zip(predicted_indices, true_labels):
                predicted_label = idx_to_class.get(predicted_index, f"Class_{predicted_index}")
                per_class_counts[true_label]["total"] += 1
                total += 1
                if predicted_label == true_label:
                    per_class_counts[true_label]["correct"] += 1
                    correct += 1

    per_class = []
    for class_name in sorted(per_class_counts):
        class_correct = per_class_counts[class_name]["correct"]
        class_total = per_class_counts[class_name]["total"]
        class_accuracy = class_correct / class_total if class_total > 0 else 0.0
        per_class.append(
            {
                "class_name": class_name,
                "correct": class_correct,
                "total": class_total,
                "accuracy": class_accuracy,
            }
        )

    accuracy = correct / total if total > 0 else 0.0
    return {
        "directory": str(Path(directory)),
        "model_path": resolved_model_path,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "per_class": per_class,
    }
