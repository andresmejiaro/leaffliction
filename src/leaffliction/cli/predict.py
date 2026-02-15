import argparse
import sys

from leaffliction.predict import run_prediction


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict plant disease from leaf image"
    )
    parser.add_argument(
        "image",
        type=str,
        help="Path to the image file",
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
        "--device",
        type=str,
        default=None,
        help="Torch device override (example: cpu, cuda)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Skip opening the visualization window",
    )

    args = parser.parse_args()

    try:
        disease_name, confidence = run_prediction(
            image_path=args.image,
            model_path=args.model,
            class_names=args.classes,
            requested_device=args.device,
            show=not args.no_show,
        )
    except (FileNotFoundError, PermissionError, ValueError, OSError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"Prediction: {disease_name}")
    print(f"Confidence: {confidence:.2%}")


if __name__ == "__main__":
    main()
