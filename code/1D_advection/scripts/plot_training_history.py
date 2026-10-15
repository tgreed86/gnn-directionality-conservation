#!/usr/bin/env python3
"""Plot training/validation curves from `history.json`.

Command-line usage:
  python scripts/plot_training_history.py \
      --history runs/no_physics_input/multiple_datasets/1000ep/history.json

  python scripts/plot_training_history.py \
      --run-dir runs/no_physics_input/multiple_datasets/1000ep \
      --out runs/no_physics_input/multiple_datasets/1000ep/loss_curves.png \
      --no-show

  python scripts/plot_training_history.py \
      --run-dir runs/cell_averaged/no_physics_input/fluxgraphnet/example \
      --metric both \
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
        description="Plot train/val curves from a training history.json file."
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
        "--metric",
        choices=("loss", "rollout_state_rel_l2", "both"),
        default="loss",
        help=(
            "Curves to plot: loss (default), rollout_state_rel_l2, or both "
            "as two panels in the same image."
        ),
    )
    parser.add_argument(
        "--train-key",
        type=str,
        default=None,
        help="Override the selected training-history key.",
    )
    parser.add_argument(
        "--val-key",
        type=str,
        default=None,
        help="Override the selected validation-history key.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Plot title. A metric-specific title is used by default.",
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
    parser.add_argument(
        "--skip-first-n",
        type=int,
        default=0,
        help=(
            "Skip the first N history rows/epochs before plotting. "
            "Default: 0 (plot all)."
        ),
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


def _require_finite_curve(values: np.ndarray, *, key: str) -> None:
    if not np.any(np.isfinite(values)):
        raise ValueError(
            f"History key '{key}' is missing or contains no finite values."
        )


def main() -> None:
    args = parse_args()
    if int(args.skip_first_n) < 0:
        raise ValueError(
            f"--skip-first-n must be >= 0, got {args.skip_first_n}."
        )
    history_path = _resolve_history_path(args)
    if not history_path.exists():
        raise FileNotFoundError(f"history.json not found: {history_path}")

    out_path = args.out or (history_path.parent / "loss_curves.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    history = _load_history(history_path)
    if args.metric == "loss":
        metric_specs = [
            (
                args.train_key or "train_loss",
                args.val_key or "val_loss",
                "Loss",
                "Training/Validation Loss",
            )
        ]
    elif args.metric == "rollout_state_rel_l2":
        metric_specs = [
            (
                args.train_key or "train_rollout_state_rel_l2",
                args.val_key or "val_rollout_state_rel_l2",
                "State relative L2",
                "Training/Validation Rollout State Relative L2",
            )
        ]
    else:
        if args.train_key is not None or args.val_key is not None:
            raise ValueError(
                "--train-key/--val-key cannot be combined with --metric both."
            )
        metric_specs = [
            (
                "train_loss",
                "val_loss",
                "Loss",
                "Training/Validation Loss",
            ),
            (
                "train_rollout_state_rel_l2",
                "val_rollout_state_rel_l2",
                "State relative L2",
                "Training/Validation Rollout State Relative L2",
            ),
        ]

    curves = []
    for train_key, val_key, ylabel, panel_title in metric_specs:
        epochs, train_values, val_values = _extract_curves(
            history=history,
            train_key=train_key,
            val_key=val_key,
        )
        _require_finite_curve(train_values, key=train_key)
        _require_finite_curve(val_values, key=val_key)
        curves.append(
            (
                epochs,
                train_values,
                val_values,
                train_key,
                val_key,
                ylabel,
                panel_title,
            )
        )

    skip_n = int(args.skip_first_n)
    if skip_n > 0:
        if skip_n >= curves[0][0].size:
            raise ValueError(
                "--skip-first-n removes all points: "
                f"skip_n={skip_n}, available={curves[0][0].size}."
            )
        curves = [
            (
                epochs[skip_n:],
                train_values[skip_n:],
                val_values[skip_n:],
                train_key,
                val_key,
                ylabel,
                panel_title,
            )
            for (
                epochs,
                train_values,
                val_values,
                train_key,
                val_key,
                ylabel,
                panel_title,
            ) in curves
        ]

    import matplotlib

    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        nrows=len(curves),
        ncols=1,
        figsize=(8.5, 5.0 if len(curves) == 1 else 8.0),
        constrained_layout=True,
        squeeze=False,
    )
    for ax, curve in zip(axes[:, 0], curves):
        (
            epochs,
            train_values,
            val_values,
            train_key,
            val_key,
            ylabel,
            panel_title,
        ) = curve
        ax.plot(epochs, train_values, label=train_key, color="tab:blue", lw=2.0)
        ax.plot(epochs, val_values, label=val_key, color="tab:orange", lw=2.0)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(panel_title if len(curves) > 1 or args.title is None else args.title)
        ax.grid(alpha=0.3)
        ax.legend(loc="best")
        if args.logy:
            ax.set_yscale("log")
    if len(curves) > 1 and args.title is not None:
        fig.suptitle(args.title)
    fig.savefig(out_path, dpi=170)
    print(f"Saved training-history plot: {out_path}")

    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
