import random
from typing import Any

import numpy as np
import plotly.express as px
import torch
from PIL import Image
from torch.utils.data import DataLoader

from leaffliction.predict import build_inference_transform, load_model_bundle, resolve_model_path
from leaffliction.train import CSVDataset


def create_eval_loader(
    csv_path: str,
    split: str,
    transform: Any,
    label_map: dict[str, int] | None,
    batch_size: int = 16,
    num_workers: int = 2,
) -> DataLoader:
    evaluation_data = CSVDataset(
        split,
        csv_path,
        root=".",
        transforms=transform,
        label_map=label_map,
    )
    return DataLoader(
        evaluation_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def visualize_random_sample(
    model: torch.nn.Module,
    dataloader: DataLoader,
    idx_to_class: dict[int, str],
    device: torch.device | str = "cpu",
) -> None:
    torch_device = torch.device(device)
    model.to(torch_device).eval()

    batch = next(iter(dataloader))
    x = batch["image"].to(torch_device)
    labels = batch["label"]
    paths = batch["path"]
    idx = random.randrange(x.size(0))

    with torch.no_grad():
        logits = model(x[idx:idx + 1])
        probabilities = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, pred_idx].item()

    true_label = labels[idx]
    pred_label = idx_to_class.get(pred_idx, f"Class_{pred_idx}")
    with Image.open(paths[idx]) as image:
        image_np = np.array(image.convert("RGB"))

    title = f"True: {true_label} | Pred: {pred_label} ({confidence:.2%})"
    fig = px.imshow(image_np, title=title)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.show()


def run_viewer(
    model_path: str = "checkpoints/model_best.artifact.pt",
    csv_path: str = "dataset.csv",
    split: str = "eval",
    batch_size: int = 16,
    num_workers: int = 2,
    requested_device: str | None = None,
) -> None:
    resolved_model_path = resolve_model_path(model_path)
    model, device, idx_to_class, metadata = load_model_bundle(
        resolved_model_path,
        requested_device=requested_device,
    )
    transform, _, _ = build_inference_transform(metadata)

    class_to_idx_raw = metadata.get("class_to_idx", {}) if metadata else {}
    label_map = None
    if isinstance(class_to_idx_raw, dict) and class_to_idx_raw:
        label_map = {str(class_name): int(index) for class_name, index in class_to_idx_raw.items()}

    dataloader = create_eval_loader(
        csv_path=csv_path,
        split=split,
        transform=transform,
        label_map=label_map,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    if not idx_to_class:
        idx_to_class = {index: class_name for class_name, index in dataloader.dataset.class_to_idx.items()}

    visualize_random_sample(
        model=model,
        dataloader=dataloader,
        idx_to_class=idx_to_class,
        device=device,
    )
