import argparse
import os
import shutil
from pathlib import Path
from typing import Any, Optional

import torch
from PIL import Image
from tabulate import tabulate
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from leaffliction.train import (
    CSVDataset,
    ImageMLP,
    augment_class,
    build_csv_from_directory,
    collect_images_by_class,
    create_zip_archive,
    split_dataset,
    train_model,
)



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
    metrics_csv = "training_metrics.csv"
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
        epochs=args.epochs,
        lr=args.lr,
        checkpoint_dir=checkpoint_dir,
        metrics_csv=metrics_csv,
    )
    
    print("Step 6: Creating archive...")
    create_zip_archive(
        augmented_dir,
        checkpoint_dir,
        output_zip,
        metrics_csv=metrics_csv,
        dataset_csv=csv_path,
    )
    
    print("Training complete!")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
