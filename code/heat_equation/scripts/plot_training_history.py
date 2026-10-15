#!/usr/bin/env python3
"""Plot training/validation loss curves from `history.json`.

Command-line usage:
  python scripts/plot_training_history.py \
      --history runs/no_physics_input/multiple_datasets/1000ep/history.json

  python scripts/plot_training_history.py \
      --run-dir runs/no_physics_input/multiple_datasets/1000ep \
      --out runs/no_physics_input/multiple_datasets/1000ep/loss_curves.png \
      --no-show
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot train/val loss curves from a training history.json file."
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=None,
        help="Path to history.json.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Run directory containing history.json (alternative to --history).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path. Default: <history_dir>/loss_curves.png",
    )
    parser.add_argument(
        "--train-key",
        type=str,
        default="train_loss",
        help="History key for training loss (default: train_loss).",
    )
    parser.add_argument(
        "--val-key",
        type=str,
        default="val_loss",
        help="History key for validation loss (default: val_loss).",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Training/Validation Loss",
        help="Plot title.",
    )
    parser.add_argument(
        "--logy",
        action="store_true",
        help="Use logarithmic y-axis.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display the plot window.",
    )
    return parser.parse_args()


def _resolve_history_path(args: argparse.Namespace) -> Path:
    if args.history is not None and args.run_dir is not None:
        raise ValueError("Use either --history or --run-dir, not both.")
    if args.history is None and args.run_dir is None:
        raise ValueError("Provide one of --history or --run-dir.")
    if args.history is not None:
        return args.history
    return Path(args.run_dir) / "history.json"


def _load_history(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    history = payload.get("history", None)
    if not isinstance(history, list):
        raise ValueError(f"File does not contain a list under key 'history': {path}")
    if len(history) == 0:
        raise ValueError(f"History is empty: {path}")
    return history


def _extract_curves(
    history: List[Dict[str, Any]],
    train_key: str,
    val_key: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    epochs: List[float] = []
    train_vals: List[float] = []
    val_vals: List[float] = []

    for i, row in enumerate(history):
        if not isinstance(row, dict):
            continue
        epoch = float(row.get("epoch", i + 1))
        train_v = row.get(train_key, np.nan)
        val_v = row.get(val_key, np.nan)
        epochs.append(epoch)
        train_vals.append(float(train_v) if train_v is not None else np.nan)
        val_vals.append(float(val_v) if val_v is not None else np.nan)

    if len(epochs) == 0:
        raise ValueError("No valid epoch rows found in history.")
    return (
        np.asarray(epochs, dtype=np.float64),
        np.asarray(train_vals, dtype=np.float64),
        np.asarray(val_vals, dtype=np.float64),
    )


def main() -> None:
    args = parse_args()
    history_path = _resolve_history_path(args)
    if not history_path.exists():
        raise FileNotFoundError(f"history.json not found: {history_path}")

    out_path = args.out or (history_path.parent / "loss_curves.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    history = _load_history(history_path)
    epochs, train_loss, val_loss = _extract_curves(
        history=history,
        train_key=args.train_key,
        val_key=args.val_key,
    )

    import matplotlib

    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
    ax.plot(epochs, train_loss, label=args.train_key, color="tab:blue", lw=2.0)
    ax.plot(epochs, val_loss, label=args.val_key, color="tab:orange", lw=2.0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(args.title)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    if args.logy:
        ax.set_yscale("log")
    fig.savefig(out_path, dpi=170)
    print(f"Saved loss curve plot: {out_path}")

    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
