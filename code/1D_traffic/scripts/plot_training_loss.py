#!/usr/bin/env python3
"""Plot training and validation loss from a saved GNN checkpoint.

The training script stores its complete epoch history in ``checkpoint.pt``.
This script plots the normalized mean squared error for the training and
validation datasets and marks the epoch with the lowest validation loss.

Example:

    python scripts/plot_training_loss.py \
        --checkpoint runs/variable_networks/sageconv/300ep/checkpoint.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

try:
    import torch
except ModuleNotFoundError as exc:
    raise SystemExit(
        "This script requires PyTorch to read checkpoint.pt."
    ) from exc


def load_history(checkpoint_path: Path) -> tuple[
    Dict[str, object],
    List[Mapping[str, object]],
]:
    """Load and validate the training history in a trusted checkpoint."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {checkpoint_path}"
        )
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("The checkpoint must contain a dictionary.")

    history = checkpoint.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError(
            "The checkpoint does not contain a non-empty training history."
        )
    required = {
        "epoch",
        "train_normalized_mse",
        "validation_normalized_mse",
    }
    for index, record in enumerate(history):
        if not isinstance(record, Mapping):
            raise ValueError(
                f"History record {index} is not a dictionary."
            )
        missing = required.difference(record)
        if missing:
            raise ValueError(
                f"History record {index} is missing fields: "
                f"{sorted(missing)}"
            )
    return checkpoint, history


def plot_loss_history(
    checkpoint: Dict[str, object],
    history: Sequence[Mapping[str, object]],
    output_path: Path,
    *,
    yscale: str,
    title: str,
    dpi: int,
) -> None:
    """Create and save the train/validation loss figure."""
    epochs = np.asarray(
        [record["epoch"] for record in history],
        dtype=np.int64,
    )
    train_loss = np.asarray(
        [record["train_normalized_mse"] for record in history],
        dtype=np.float64,
    )
    validation_loss = np.asarray(
        [record["validation_normalized_mse"] for record in history],
        dtype=np.float64,
    )
    if not (
        np.all(np.isfinite(train_loss))
        and np.all(np.isfinite(validation_loss))
    ):
        raise ValueError("Training history contains non-finite losses.")
    if yscale == "log" and (
        np.any(train_loss <= 0.0) or np.any(validation_loss <= 0.0)
    ):
        raise ValueError("A logarithmic axis requires positive losses.")

    best_index = int(np.argmin(validation_loss))
    best_epoch = int(epochs[best_index])
    best_validation = float(validation_loss[best_index])
    saved_best_epoch = checkpoint.get("best_epoch")
    if saved_best_epoch is not None:
        saved_best_epoch = int(saved_best_epoch)
        matching = np.flatnonzero(epochs == saved_best_epoch)
        if matching.size:
            best_index = int(matching[0])
            best_epoch = saved_best_epoch
            best_validation = float(validation_loss[best_index])

    figure, axis = plt.subplots(
        figsize=(8.5, 5.2),
        constrained_layout=True,
    )
    axis.plot(
        epochs,
        train_loss,
        color="tab:blue",
        linewidth=1.6,
        label="Train",
    )
    axis.plot(
        epochs,
        validation_loss,
        color="tab:orange",
        linewidth=1.6,
        label="Validation",
    )
    axis.axvline(
        best_epoch,
        color="0.35",
        linestyle="--",
        linewidth=1.1,
        label=f"Best validation epoch ({best_epoch})",
    )
    axis.scatter(
        [best_epoch],
        [best_validation],
        color="tab:orange",
        edgecolor="white",
        linewidth=0.8,
        s=45,
        zorder=3,
    )
    axis.set_title(title)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Normalized MSE loss")
    axis.set_yscale(yscale)
    axis.grid(True, alpha=0.28)
    axis.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot training and validation normalized MSE versus epoch "
            "from checkpoint.pt."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Checkpoint created by scripts/train.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output PNG path; defaults to loss_vs_epoch.png beside the "
            "checkpoint."
        ),
    )
    parser.add_argument(
        "--yscale",
        choices=("log", "linear"),
        default="log",
        help="Loss-axis scale; default: log.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional figure title; defaults to the run directory name.",
    )
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Load a checkpoint and plot its loss history."""
    args = parse_args(argv)
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive.")

    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else checkpoint_path.parent / "loss_vs_epoch.png"
    )
    title = (
        args.title
        if args.title is not None
        else f"Training history: {checkpoint_path.parent.name}"
    )
    checkpoint, history = load_history(checkpoint_path)
    plot_loss_history(
        checkpoint,
        history,
        output_path,
        yscale=args.yscale,
        title=title,
        dpi=args.dpi,
    )

    validation_loss = np.asarray(
        [
            record["validation_normalized_mse"]
            for record in history
        ],
        dtype=np.float64,
    )
    best_index = int(np.argmin(validation_loss))
    best_epoch = int(history[best_index]["epoch"])
    print(
        f"best validation epoch: {best_epoch} "
        f"(nMSE={validation_loss[best_index]:.6e})"
    )
    print(f"saved plot: {output_path}")


if __name__ == "__main__":
    main()
