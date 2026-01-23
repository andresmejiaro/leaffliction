#!/usr/bin/env python3
import argparse
import os
from collections import defaultdict
from typing import DefaultDict
import plotly.express as px
import pandas as pd


IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif")


def collect_images_by_class(root_path: str)-> DefaultDict[str, list[str]]:
    class_images: DefaultDict[str, list[str]] = defaultdict(list)
    
    if not os.path.isdir(root_path):
        print(f"Error: '{root_path}' is not a directory.")
        return class_images
    
    for root, _, files in os.walk(root_path):
        for file in files:
            if file.lower().endswith(IMG_EXTENSIONS):
                file_path = os.path.join(root, file)
                class_name = os.path.basename(root)
                class_images[class_name].append(file_path)
    
    return class_images


def print_statistics(class_images: DefaultDict[str, list[str]]):
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    
    print("\nImages per Class:")
    for class_name, images in sorted(class_images.items()):
        print(f"  {class_name}: {len(images)} images")
    print(f"  TOTAL: {sum(len(imgs) for imgs in class_images.values())}")
    print("="*60 + "\n")


def visualize_distribution(class_images: DefaultDict[str, list[str]], directory_name: str):
    data = []
    for class_name, images in class_images.items():
        data.append({"Class": class_name, "Count": len(images)})
    
    df = pd.DataFrame(data)
    
    print("Generating bar chart...")
    fig1 = px.bar(
        df,
        x="Class",
        y="Count",
        text="Count",
        color="Class",
        color_discrete_sequence=px.colors.qualitative.Set3,
        title=f"Image Distribution for {directory_name}"
    )
    fig1.update_layout(
        xaxis_title="Class",
        yaxis_title="Count",
        showlegend=False
    )
    fig1.show()
    
    print("Generating pie chart...")
    fig2 = px.pie(
        df,
        names="Class",
        values="Count",
        color="Class",
        color_discrete_sequence=px.colors.qualitative.Set3,
        title=f"Class Distribution for {directory_name}"
    )
    fig2.show()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and visualize image distribution by class"
    )
    parser.add_argument(
        "directory",
        type=str,
        help="Directory containing images organized by class"
    )
    
    args = parser.parse_args()
    
    directory_name = os.path.basename(os.path.normpath(args.directory))
    
    print(f"Scanning directory: {args.directory}")
    class_images = collect_images_by_class(args.directory)
    
    if not class_images:
        print("No images found.")
        return
    
    print(f"Found {len(class_images)} classes")
    print_statistics(class_images)
    visualize_distribution(class_images, directory_name)


if __name__ == "__main__":
    main()
