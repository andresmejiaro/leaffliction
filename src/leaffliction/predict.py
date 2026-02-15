from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import torch
from PIL import Image
from plotly.subplots import make_subplots
from torch import nn
from torchvision import transforms as T

from leaffliction.train import (
    DEFAULT_IMAGE_SIZE,
    DEFAULT_NORMALIZE_MEAN,
    DEFAULT_NORMALIZE_STD,
    ImageMLP,
)


def resolve_device(requested_device: str | None = None) -> torch.device:
    if requested_device:
        return torch.device(requested_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_image_shape(value: Any) -> tuple[int, int, int]:
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return tuple(int(item) for item in value)
    return (3, DEFAULT_IMAGE_SIZE[0], DEFAULT_IMAGE_SIZE[1])


def _to_normalize_tuple(
    value: Any,
    fallback: tuple[float, float, float],
) -> tuple[float, float, float]:
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return tuple(float(item) for item in value)
    return fallback


def build_inference_transform(
    metadata: dict[str, Any] | None = None,
) -> tuple[T.Compose, tuple[float, float, float], tuple[float, float, float]]:
    preprocess = {}
    if metadata is not None:
        preprocess = metadata.get("preprocess", {})

    image_size_raw = preprocess.get("image_size", DEFAULT_IMAGE_SIZE)
    if isinstance(image_size_raw, (list, tuple)) and len(image_size_raw) == 2:
        image_size = (int(image_size_raw[0]), int(image_size_raw[1]))
    else:
        image_size = DEFAULT_IMAGE_SIZE

    mean = _to_normalize_tuple(preprocess.get("normalize_mean"), DEFAULT_NORMALIZE_MEAN)
    std = _to_normalize_tuple(preprocess.get("normalize_std"), DEFAULT_NORMALIZE_STD)

    transform = T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    return transform, mean, std


def _extract_label_mappings(metadata: dict[str, Any]) -> tuple[dict[str, int], dict[int, str]]:
    class_to_idx_raw = metadata.get("class_to_idx", {})
    if not isinstance(class_to_idx_raw, dict):
        return {}, {}
    class_to_idx = {str(class_name): int(index) for class_name, index in class_to_idx_raw.items()}
    idx_to_class = {index: class_name for class_name, index in class_to_idx.items()}
    return class_to_idx, idx_to_class


def _load_model_from_artifact(
    checkpoint: dict[str, Any],
) -> tuple[nn.Module, dict[str, Any], dict[int, str]]:
    metadata = checkpoint.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    model_meta = metadata.get("model", {})
    if not isinstance(model_meta, dict):
        model_meta = {}

    model_name = model_meta.get("name", "ImageMLP")
    if model_name != "ImageMLP":
        raise ValueError(f"Unsupported model architecture in artifact: {model_name}")

    class_to_idx, idx_to_class = _extract_label_mappings(metadata)
    num_classes = int(model_meta.get("num_classes", len(class_to_idx)))
    if num_classes <= 0:
        raise ValueError("Invalid number of classes in artifact metadata.")

    p_drop_raw = model_meta.get("p_drop", 0.2)
    p_drop = 0.2 if p_drop_raw is None else float(p_drop_raw)
    in_shape = _to_image_shape(model_meta.get("in_shape"))

    model = ImageMLP(in_shape=in_shape, num_classes=num_classes, p_drop=p_drop)
    model.load_state_dict(checkpoint["state_dict"])
    return model, metadata, idx_to_class


def load_model_bundle(
    model_path: str,
    requested_device: str | None = None,
) -> tuple[nn.Module, torch.device, dict[int, str], dict[str, Any]]:
    device = resolve_device(requested_device)
    checkpoint = torch.load(model_path, map_location=device)

    metadata: dict[str, Any] = {}
    idx_to_class: dict[int, str] = {}
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model, metadata, idx_to_class = _load_model_from_artifact(checkpoint)
    elif isinstance(checkpoint, nn.Module):
        model = checkpoint
    else:
        raise ValueError("Unsupported checkpoint format. Use a .pt or .artifact.pt checkpoint.")

    model.to(device).eval()
    return model, device, idx_to_class, metadata


def load_and_preprocess_image(
    image_path: str,
    transform: T.Compose,
) -> tuple[Image.Image, torch.Tensor]:
    with Image.open(image_path) as image:
        original_image = image.convert("RGB").copy()
    image_tensor = transform(original_image).unsqueeze(0)
    return original_image, image_tensor


def predict_disease(
    model: nn.Module,
    img_tensor: torch.Tensor,
    device: torch.device,
    class_names: list[str] | None = None,
    idx_to_class: dict[int, str] | None = None,
) -> tuple[str, float]:
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        logits = model(img_tensor)
        probabilities = torch.softmax(logits, dim=1)
        predicted_index = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_index].item()

    if class_names and predicted_index < len(class_names):
        disease_name = class_names[predicted_index]
    elif idx_to_class and predicted_index in idx_to_class:
        disease_name = idx_to_class[predicted_index]
    else:
        disease_name = f"Class_{predicted_index}"

    return disease_name, confidence


def _denormalize_tensor_image(
    transformed_img_tensor: torch.Tensor,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> np.ndarray:
    mean_tensor = torch.tensor(mean).view(3, 1, 1)
    std_tensor = torch.tensor(std).view(3, 1, 1)
    transformed_image = transformed_img_tensor.squeeze(0).cpu() * std_tensor + mean_tensor
    transformed_image = transformed_image.clamp(0, 1).permute(1, 2, 0).numpy()
    return (transformed_image * 255).astype(np.uint8)


def visualize_prediction(
    original_img: Image.Image,
    transformed_img_tensor: torch.Tensor,
    disease_name: str,
    confidence: float,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> None:
    transformed_np = _denormalize_tensor_image(transformed_img_tensor, mean, std)
    original_np = np.array(original_img)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Original Image", "Transformed Image"],
    )
    fig.add_trace(go.Image(z=original_np), row=1, col=1)
    fig.add_trace(go.Image(z=transformed_np), row=1, col=2)
    fig.update_xaxes(visible=False, row=1, col=1)
    fig.update_yaxes(visible=False, row=1, col=1)
    fig.update_xaxes(visible=False, row=1, col=2)
    fig.update_yaxes(visible=False, row=1, col=2)
    fig.update_layout(
        title=f"Prediction: {disease_name} (Confidence: {confidence:.2%})",
        margin=dict(l=0, r=0, t=60, b=0),
        height=550,
        width=1100,
    )
    fig.show()


def resolve_model_path(model_path: str) -> str:
    path = Path(model_path)
    if path.is_file():
        return str(path)

    artifact_suffix = ".artifact.pt"
    if model_path.endswith(artifact_suffix):
        fallback = model_path.replace(artifact_suffix, ".pt")
        if Path(fallback).is_file():
            return fallback
    raise FileNotFoundError(f"Model checkpoint not found: {model_path}")


def run_prediction(
    image_path: str,
    model_path: str = "checkpoints/model_best.artifact.pt",
    class_names: list[str] | None = None,
    requested_device: str | None = None,
    show: bool = True,
) -> tuple[str, float]:
    resolved_model_path = resolve_model_path(model_path)
    model, device, idx_to_class, metadata = load_model_bundle(
        resolved_model_path,
        requested_device=requested_device,
    )
    transform, mean, std = build_inference_transform(metadata)
    original_img, img_tensor = load_and_preprocess_image(image_path, transform)
    disease_name, confidence = predict_disease(
        model,
        img_tensor,
        device,
        class_names=class_names,
        idx_to_class=idx_to_class,
    )

    if show:
        visualize_prediction(
            original_img,
            img_tensor,
            disease_name,
            confidence,
            mean,
            std,
        )
    return disease_name, confidence
