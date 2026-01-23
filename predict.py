#!/usr/bin/env python3
import argparse
import torch
from torch import nn
from PIL import Image
from torchvision import transforms as T
import matplotlib.pyplot as plt
import sys

def load_model(model_path: str) ->tuple[nn.Module, torch.device]:
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = torch.load(model_path, map_location=device)
    except Exception as e:
        print(f"Something wrong opening the model in {model_path} Error: {e}")
        sys.exit(1)
    model.eval()
    return model, device


def load_and_preprocess_image(image_path: str)-> tuple[Image.Image, torch.Tensor]:
    try:
        with Image.open(image_path) as img:    
            img = img.convert("RGB").copy()
    except Exception as e:
        print(f"Something wrong opening the image in {image_path} Error: {e}")
        sys.exit(1)

    
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    img_tensor = transform(img).unsqueeze(0)
    return img, img_tensor


def predict_disease(model: nn.Module, img_tensor: torch.Tensor, device: torch.device, class_names: list[str]|None = None)-> tuple[str,float]:
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():
        logits = model(img_tensor)
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class].item()
    
    if class_names and predicted_class < len(class_names):
        disease_name = class_names[predicted_class]
    else:
        disease_name = f"Class_{predicted_class}"
    
    return disease_name, confidence


def visualize_prediction(original_img: Image.Image, transformed_img_tensor: torch.Tensor, disease_name:str,
                          confidence: float):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    transformed_img = transformed_img_tensor.squeeze(0) * std + mean
    transformed_img = transformed_img.clamp(0, 1)
    transformed_img = transformed_img.permute(1, 2, 0).cpu().numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].imshow(original_img)
    axes[0].set_title("Original Image", fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(transformed_img)
    axes[1].set_title("Transformed Image", fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    fig.suptitle(
        f"Prediction: {disease_name} (Confidence: {confidence:.2%})",
        fontsize=16,
        fontweight='bold'
    )
    
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Predict plant disease from leaf image"
    )
    parser.add_argument(
        "image",
        type=str,
        help="Path to the image file"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="checkpoints/model_best.pt",
        help="Path to trained model (default: checkpoints/model_best.pt)"
    )
    parser.add_argument(
        "-c", "--classes",
        type=str,
        nargs="+",
        help="List of class names in order"
    )
    
    args = parser.parse_args()
    
    print(f"Loading model from: {args.model}")
    model, device = load_model(args.model)
    
    print(f"Loading image: {args.image}")
    original_img, img_tensor = load_and_preprocess_image(args.image)
    
    print("Making prediction...")
    disease_name, confidence = predict_disease(
        model, img_tensor, device, args.classes
    )
    
    print(f"\nPrediction: {disease_name}")
    print(f"Confidence: {confidence:.2%}")
    
    visualize_prediction(original_img, img_tensor, disease_name, confidence)


if __name__ == "__main__":
    main()
