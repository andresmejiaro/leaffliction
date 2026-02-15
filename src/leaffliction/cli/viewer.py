import argparse
import sys

from leaffliction.viewer import run_viewer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize random validation sample with model prediction"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="checkpoints/model_best.artifact.pt",
        help="Path to checkpoint (default: checkpoints/model_best.artifact.pt)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="dataset.csv",
        help="Path to dataset CSV (default: dataset.csv)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="eval",
        help="Dataset split to sample from (default: eval)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for data loading (default: 16)",
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
        run_viewer(
            model_path=args.model,
            csv_path=args.csv,
            split=args.split,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            requested_device=args.device,
        )
    except (FileNotFoundError, PermissionError, ValueError, OSError) as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
