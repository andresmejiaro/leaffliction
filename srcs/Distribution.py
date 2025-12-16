#!/usr/bin/env python3
import argparse
import os
import multiprocessing
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import polars as pl
import plotly.express as px
import pandas as pd

def _process_path(p: Path, exts: set) -> dict | None:
    if p.is_file() and p.suffix.lower() in exts:
        parts = p.parts
        split = None
        for i, part in enumerate(parts):
            if part in ['train', 'eval', 'test']:
                split = part
                break
        
        return {
            "path": str(p.resolve()),
            "name": p.name,
            "class": p.parent.name,
            "stem": p.stem,
            "group": p.stem.split("_g")[0] if "_g" in p.stem else p.stem.split("_")[0],
            "split": split if split else "unknown",
        }
    return None

def list_images(root: str, max_workers: int = 8) -> pl.DataFrame:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif"}
    paths = list(Path(root).rglob("*"))
    rows = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for result in executor.map(lambda p: _process_path(p, exts), paths):
            if result:
                rows.append(result)
    
    if not rows:
        raise ValueError(f"No images found in {root}")
    
    return pl.from_records(rows)

def train_test_val(df: pl.DataFrame) -> pl.DataFrame:
    TRAIN_RATIO = 0.7
    EVAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    group_info = (
        df.group_by(["class", "group"])
        .agg(pl.len().alias("image_count"))
    )
    
    group_shuffled = group_info.sample(fraction=1, shuffle=True, seed=42)
    group_with_cumsum = (
        group_shuffled
        .sort(["class", "group"])
        .with_columns([
            pl.col("image_count").cum_sum().over("class").alias("cumsum_images"),
            pl.col("image_count").sum().over("class").alias("total_images"),
        ])
    )
    group_with_pos = group_with_cumsum.with_columns(
        (pl.col("cumsum_images") / pl.col("total_images")).alias("position")
    )
    group_with_split = group_with_pos.with_columns(
        pl.when(pl.col("position") <= TRAIN_RATIO)
        .then(pl.lit("train"))
        .when(pl.col("position") <= TRAIN_RATIO + EVAL_RATIO)
        .then(pl.lit("eval"))
        .otherwise(pl.lit("test"))
        .alias("split")
    )
    
    group_splits = group_with_split.select(["class", "group", "split"])
    result = df.join(group_splits, on=["class", "group"], how="left")
    
    return result

def build_dataset_csv(root: str, out_csv: str, max_workers: int = 8) -> None:
    print(f"Scanning images in {root}...")
    df = list_images(root, max_workers=max_workers)
    print(f"   Found {len(df)} images")
    
    if "split" in df.columns:
        unknown_splits = df.filter(pl.col("split") == "unknown")
        if len(unknown_splits) > 0:
            print(f"   Warning: {len(unknown_splits)} images have unknown split")
    
    df.write_csv(out_csv)
    print(f"Dataset CSV saved to {out_csv}")

def visualize_dataset(csv_path: str) -> None:
    print(f"\nLoading dataset from {csv_path}...")
    df = pl.read_csv(csv_path)
    pdf = df.to_pandas()
    
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    
    print("\nImages per Class:")
    class_counts = pdf['class'].value_counts().sort_index()
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count}")
    print(f"  TOTAL: {len(pdf)}")
    
    print("\nImages per Split:")
    split_counts = pdf['split'].value_counts()
    for split_name in ['train', 'eval', 'test']:
        if split_name in split_counts.index:
            count = split_counts[split_name]
            percentage = (count / len(pdf)) * 100
            print(f"  {split_name}: {count} ({percentage:.1f}%)")
    
    print("\nImages per Class per Split:")
    for split_name in ['train', 'eval', 'test']:
        split_df = pdf[pdf['split'] == split_name]
        if len(split_df) > 0:
            print(f"\n  {split_name.upper()}:")
            split_class_counts = split_df['class'].value_counts().sort_index()
            for class_name, count in split_class_counts.items():
                class_total = len(pdf[pdf['class'] == class_name])
                percentage = (count / class_total) * 100
                print(f"    {class_name}: {count} ({percentage:.1f}%)")
    
    print("\nVerifying group integrity...")
    groups_per_split = pdf.groupby('group')['split'].nunique()
    leaked_groups = groups_per_split[groups_per_split > 1]
    if len(leaked_groups) > 0:
        print(f"  WARNING: {len(leaked_groups)} groups appear in multiple splits!")
        print(f"  First few leaked groups: {leaked_groups.head().index.tolist()}")
    else:
        print(f"  OK: All {len(groups_per_split)} groups are contained in single splits")
    
    print("="*60 + "\n")
    
    print("Generating histogram by class...")
    fig1 = px.histogram(
        pdf, 
        x="class", 
        text_auto=True, 
        color="class",
        color_discrete_sequence=px.colors.qualitative.Set3,
        title="Image Distribution by Class"
    )
    fig1.update_layout(
        xaxis_title="Class",
        yaxis_title="Count",
        showlegend=False
    )
    fig1.show()
    
    print("Generating pie chart by class...")
    fig2 = px.pie(
        pdf, 
        names="class", 
        color="class",
        color_discrete_sequence=px.colors.qualitative.Set3,
        title="Class Distribution"
    )
    fig2.show()
    
    print("Generating split distribution...")
    fig3 = px.histogram(
        pdf,
        x="split",
        color="class",
        text_auto=True,
        color_discrete_sequence=px.colors.qualitative.Set3,
        title="Image Distribution by Split and Class",
        category_orders={"split": ["train", "eval", "test"]}
    )
    fig3.update_layout(
        xaxis_title="Split",
        yaxis_title="Count"
    )
    fig3.show()
    
    if (pdf["split"] != "unknown").any():
        print("Generating pie chart by split...")
        fig4 = px.pie(
            pdf,
            names="split",
            title="Train/Eval/Test Split Distribution",
            category_orders={"split": ["train", "eval", "test"]},
            color_discrete_sequence=["#4CAF50", "#FFC107", "#F44336"]
        )
        fig4.show()

def main():
    parser = argparse.ArgumentParser(
        description="Scan images, build dataset CSV, and visualize class distribution."
    )
    parser.add_argument(
        "directory",
        type=str,
        nargs="?",
        help="Root directory containing images (required unless --graph_only is used)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="dataset.csv",
        help="Output CSV file name (default: dataset.csv)"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Number of threads to use (default: max available)"
    )
    parser.add_argument(
        "--build_only",
        action="store_true",
        help="Only build dataset CSV, do not generate graphs"
    )
    parser.add_argument(
        "--graph_only",
        action="store_true",
        help="Only generate graphs from existing CSV (skip building)"
    )
    
    args = parser.parse_args()
    
    if args.build_only and args.graph_only:
        parser.error("Options --build_only and --graph_only cannot be used together.")
    
    if not args.graph_only and not args.directory:
        parser.error("directory argument is required unless --graph_only is used")
    
    csv_path = args.output
    
    if args.graph_only:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"[ERROR] CSV file '{csv_path}' not found. Run build first.")
        visualize_dataset(csv_path)
        return
    build_dataset_csv(args.directory, csv_path, max_workers=args.threads)
    if not args.build_only:
        visualize_dataset(csv_path)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"an error {e} has occured")