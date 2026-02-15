import argparse
import sys

from tabulate import tabulate

from leaffliction.accuracy import evaluate_accuracy_on_directory


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate model accuracy on images from a directory"
    )
    parser.add_argument(
        "directory",
        type=str,
        help="Directory containing images organized by class folders",
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="checkpoints/model_best.artifact.pt",
        help="Path to checkpoint (default: checkpoints/model_best.artifact.pt)",
    )
    parser.add_argument(
        "-c", "--classes",
        type=str,
        nargs="+",
        help="Optional class names override in index order",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation (default: 32)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of data loader workers (default: 2)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device override (example: cpu, cuda)",
    )

    args = parser.parse_args()

    try:
        report = evaluate_accuracy_on_directory(
            directory=args.directory,
            model_path=args.model,
            class_names=args.classes,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            requested_device=args.device,
        )
    except (FileNotFoundError, PermissionError, ValueError, OSError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"Directory: {report['directory']}")
    print(f"Model: {report['model_path']}")
    print(f"Correct: {report['correct']} / {report['total']}")
    print(f"Accuracy: {report['accuracy']:.2%}")

    per_class_rows = [
        [
            row["class_name"],
            row["correct"],
            row["total"],
            f"{row['accuracy']:.2%}",
        ]
        for row in report["per_class"]
    ]
    if per_class_rows:
        print(tabulate(
            per_class_rows,
            headers=["Class", "Correct", "Total", "Accuracy"],
            tablefmt="fancy_grid",
        ))


if __name__ == "__main__":
    main()
