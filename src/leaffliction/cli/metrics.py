import argparse
import sys

from leaffliction.metrics import plot_metrics_from_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot training metrics from CSV in panel format"
    )
    parser.add_argument(
        "metrics_csv",
        nargs="?",
        default="training_metrics.csv",
        help="Path to metrics CSV (default: training_metrics.csv)",
    )

    args = parser.parse_args()

    try:
        plot_metrics_from_csv(args.metrics_csv)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
