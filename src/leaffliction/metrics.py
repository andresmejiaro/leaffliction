from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

REQUIRED_METRIC_COLUMNS = (
    "epoch",
    "train_loss",
    "eval_loss",
    "eval_accuracy",
    "eval_sensitivity_macro",
    "eval_specificity_macro",
)


def load_metrics_csv(metrics_csv: str | Path) -> pd.DataFrame:
    metrics_path = Path(metrics_csv)
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Metrics CSV not found: {metrics_path}")

    metrics_df = pd.read_csv(metrics_path)
    missing_columns = sorted(
        column for column in REQUIRED_METRIC_COLUMNS if column not in metrics_df.columns
    )
    if missing_columns:
        raise ValueError(
            f"Metrics CSV missing required columns: {', '.join(missing_columns)}"
        )

    return metrics_df.sort_values("epoch")


def plot_metrics_panels(metrics_df: pd.DataFrame) -> None:
    if metrics_df.empty:
        raise ValueError("Metrics CSV is empty.")

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Loss (Train vs Eval)",
            "Eval Accuracy",
            "Eval Sensitivity (macro avg)",
            "Eval Specificity (macro avg)",
        ],
    )

    epochs = metrics_df["epoch"]
    fig.add_trace(
        go.Scatter(x=epochs, y=metrics_df["train_loss"], name="Train Loss", mode="lines+markers"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=metrics_df["eval_loss"], name="Eval Loss", mode="lines+markers"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=metrics_df["eval_accuracy"],
            name="Eval Accuracy",
            mode="lines+markers",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=metrics_df["eval_sensitivity_macro"],
            name="Eval Sensitivity (macro avg)",
            mode="lines+markers",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=metrics_df["eval_specificity_macro"],
            name="Eval Specificity (macro avg)",
            mode="lines+markers",
        ),
        row=2,
        col=2,
    )

    for row, col in ((1, 2), (2, 1), (2, 2)):
        fig.update_yaxes(range=[0, 1], row=row, col=col)

    for row, col in ((1, 1), (1, 2), (2, 1), (2, 2)):
        fig.update_xaxes(title_text="Epoch", row=row, col=col)

    fig.update_layout(
        title="Training Metrics",
        height=900,
        width=1300,
        legend=dict(orientation="h", yanchor="bottom", y=-0.05, xanchor="center", x=0.5),
    )
    fig.show()


def plot_metrics_from_csv(metrics_csv: str | Path) -> None:
    metrics_df = load_metrics_csv(metrics_csv)
    plot_metrics_panels(metrics_df)
