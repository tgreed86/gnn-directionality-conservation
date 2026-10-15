#!/usr/bin/env python3
"""Lightweight diagnostic plots for rollout evaluation outputs.

Expected input is the `.npz` produced by `scripts/rollout_gnn_heat.py`.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze rollout diagnostics NPZ.")
    parser.add_argument(
        "--rollout-npz",
        type=Path,
        required=True,
        help="Path to rollout_eval.npz",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for diagnostic plots and summary JSON.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=1.0,
        help="Time-step value used for plotting x-axis.",
    )
    parser.add_argument(
        "--steps",
        type=str,
        default="",
        help=(
            "Comma-separated rollout steps for overlay plots, e.g. '0,10,25,50,100'. "
            "If omitted, steps are selected automatically."
        ),
    )
    parser.add_argument(
        "--num-step-plots",
        type=int,
        default=6,
        help="Number of auto-selected steps when --steps is omitted.",
    )
    parser.add_argument(
        "--nx",
        type=int,
        default=None,
        help="Optional grid size in x for reshaping representative trajectories.",
    )
    parser.add_argument(
        "--ny",
        type=int,
        default=None,
        help="Optional grid size in y for reshaping representative trajectories.",
    )
    parser.add_argument(
        "--spectrum-logy",
        action="store_true",
        help="Use log scale on y-axis for spectrum overlay plots.",
    )
    return parser.parse_args()


def _load_npz(path: Path) -> Dict[str, Any]:
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def _safe_last(x: np.ndarray) -> float | None:
    if x.size == 0:
        return None
    val = float(x.reshape(-1)[-1])
    if not np.isfinite(val):
        return None
    return val


def _parse_steps(text: str, *, max_step: int) -> List[int]:
    if max_step < 0:
        return []
    raw = text.strip()
    if raw == "":
        return []
    parts = [p.strip() for p in raw.split(",") if p.strip() != ""]
    out: List[int] = []
    for p in parts:
        try:
            k = int(p)
        except ValueError as exc:
            raise ValueError(
                f"Invalid step '{p}' in --steps='{text}'. Use comma-separated integers."
            ) from exc
        if k < 0 or k > max_step:
            raise ValueError(
                f"Step {k} is out of range [0, {max_step}] for this rollout."
            )
        out.append(k)
    return list(dict.fromkeys(out))


def _auto_steps(*, max_step: int, num_steps: int) -> List[int]:
    if max_step < 0:
        return []
    if num_steps <= 0:
        raise ValueError(f"--num-step-plots must be positive, got {num_steps}.")
    n = min(int(num_steps), int(max_step + 1))
    arr = np.linspace(0, max_step, num=n, dtype=np.int64)
    arr = np.unique(arr)
    return arr.tolist()


def _infer_grid_shape(
    *,
    n_nodes: int,
    nx_arg: int | None,
    ny_arg: int | None,
) -> tuple[int, int, str]:
    if n_nodes <= 0:
        raise ValueError(f"n_nodes must be positive, got {n_nodes}.")

    if nx_arg is not None and nx_arg <= 0:
        raise ValueError(f"--nx must be positive, got {nx_arg}.")
    if ny_arg is not None and ny_arg <= 0:
        raise ValueError(f"--ny must be positive, got {ny_arg}.")

    if nx_arg is not None and ny_arg is not None:
        nx = int(nx_arg)
        ny = int(ny_arg)
        if nx * ny != n_nodes:
            raise ValueError(
                f"Provided --nx*--ny={nx*ny} does not match n_nodes={n_nodes}."
            )
        return nx, ny, "provided"

    if nx_arg is not None:
        nx = int(nx_arg)
        if n_nodes % nx != 0:
            raise ValueError(
                f"n_nodes={n_nodes} is not divisible by --nx={nx}."
            )
        ny = int(n_nodes // nx)
        return nx, ny, "nx-provided"

    if ny_arg is not None:
        ny = int(ny_arg)
        if n_nodes % ny != 0:
            raise ValueError(
                f"n_nodes={n_nodes} is not divisible by --ny={ny}."
            )
        nx = int(n_nodes // ny)
        return nx, ny, "ny-provided"

    sq = int(round(math.sqrt(float(n_nodes))))
    if sq * sq == n_nodes:
        return sq, sq, "inferred-square"

    # Fallback: choose factor pair closest to square.
    best_a = 1
    best_b = n_nodes
    best_gap = abs(best_b - best_a)
    lim = int(math.floor(math.sqrt(float(n_nodes))))
    for a in range(1, lim + 1):
        if n_nodes % a != 0:
            continue
        b = n_nodes // a
        gap = abs(b - a)
        if gap < best_gap:
            best_a, best_b, best_gap = a, b, gap
    nx = int(max(best_a, best_b))
    ny = int(min(best_a, best_b))
    return nx, ny, "inferred-factor"


def _radial_spectrum_2d(field_2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return radially averaged 2D FFT magnitude vs |k|-bin."""
    if field_2d.ndim != 2:
        raise ValueError(f"field_2d must be 2D, got shape {field_2d.shape}.")

    nx, ny = field_2d.shape
    fft2 = np.fft.fft2(field_2d)
    amp = np.abs(fft2)

    kx = np.fft.fftfreq(nx) * float(nx)
    ky = np.fft.fftfreq(ny) * float(ny)
    kxx, kyy = np.meshgrid(kx, ky, indexing="ij")
    kr = np.sqrt(kxx * kxx + kyy * kyy)
    kbin = np.rint(kr).astype(np.int64)

    weights = amp.reshape(-1).astype(np.float64)
    bins = kbin.reshape(-1)
    sums = np.bincount(bins, weights=weights)
    counts = np.bincount(bins)
    valid = counts > 0
    k_vals = np.arange(sums.shape[0], dtype=np.float64)[valid]
    radial_mean = sums[valid] / counts[valid]
    return k_vals, radial_mean


def _plot_state_and_spectrum_overlays_2d(
    *,
    pred_traj_2d: np.ndarray,
    true_traj_2d: np.ndarray,
    steps: List[int],
    dt: float,
    sample_id: int | None,
    out_state_path: Path,
    out_spec_path: Path,
    dpi: int,
    spectrum_logy: bool,
) -> None:
    import matplotlib.pyplot as plt

    if len(steps) == 0:
        return

    n = len(steps)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    x_idx = np.arange(pred_traj_2d.shape[1], dtype=np.int64)
    y_mid = pred_traj_2d.shape[2] // 2

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.4 * ncols, 3.2 * nrows), constrained_layout=True
    )
    axes_arr = np.atleast_1d(axes).reshape(-1)
    for j, k in enumerate(steps):
        ax = axes_arr[j]
        line_true = true_traj_2d[k, :, y_mid]
        line_pred = pred_traj_2d[k, :, y_mid]
        ax.plot(x_idx, line_true, color="tab:green", lw=1.6, label="true")
        ax.plot(x_idx, line_pred, color="tab:red", lw=1.4, ls="--", label="pred")
        ax.set_title(f"Step {k}")
        ax.set_xlabel("x-index")
        ax.set_ylabel(f"state @ y={y_mid}")
        ax.grid(alpha=0.3)
        if j == 0:
            ax.legend(loc="best")
    for j in range(n, len(axes_arr)):
        axes_arr[j].axis("off")
    sample_text = "sample_id=unknown" if sample_id is None else f"sample_id={sample_id}"
    fig.suptitle(
        f"State Overlay at Selected Rollout Steps (Centerline) | {sample_text}",
        fontsize=12,
    )
    fig.savefig(out_state_path, dpi=dpi)
    plt.close(fig)

    fig2, axes2 = plt.subplots(
        nrows, ncols, figsize=(4.4 * ncols, 3.2 * nrows), constrained_layout=True
    )
    axes2_arr = np.atleast_1d(axes2).reshape(-1)
    for j, k in enumerate(steps):
        ax = axes2_arr[j]
        k_true, spec_true = _radial_spectrum_2d(true_traj_2d[k])
        k_pred, spec_pred = _radial_spectrum_2d(pred_traj_2d[k])
        k_common = int(min(k_true.shape[0], k_pred.shape[0]))
        kk = np.arange(k_common, dtype=np.int64)
        ax.plot(kk, spec_true[:k_common], color="tab:green", lw=1.5, label="|FFT2(true)| radial mean")
        ax.plot(kk, spec_pred[:k_common], color="tab:red", lw=1.3, ls="--", label="|FFT2(pred)| radial mean")
        ax.set_title(f"Step {k}")
        ax.set_xlabel("radial mode |k| (binned)")
        ax.set_ylabel("magnitude")
        if spectrum_logy:
            ax.set_yscale("log")
        ax.grid(alpha=0.3)
        if j == 0:
            ax.legend(loc="best")
    for j in range(n, len(axes2_arr)):
        axes2_arr[j].axis("off")
    fig2.suptitle(
        f"Radial Spectrum Overlay at Selected Rollout Steps | {sample_text}",
        fontsize=12,
    )
    fig2.savefig(out_spec_path, dpi=dpi)
    plt.close(fig2)


def main() -> None:
    args = _parse_args()
    payload = _load_npz(args.rollout_npz)

    out_dir = args.out_dir or (args.rollout_npz.parent / "rollout_diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)

    mean_rmse = np.asarray(payload.get("mean_rmse", np.array([])), dtype=np.float64)
    mean_rel_l2 = np.asarray(payload.get("mean_rel_l2", np.array([])), dtype=np.float64)
    raw_horizon = payload.get("horizon", None)
    if raw_horizon is None:
        horizon = int(mean_rmse.size)
    else:
        horizon = int(np.asarray(raw_horizon).reshape(-1)[0])

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if mean_rmse.size > 0 and mean_rel_l2.size > 0:
        steps = np.arange(1, mean_rmse.size + 1, dtype=np.float64)
        times = steps * float(args.dt)
        fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0), constrained_layout=True)
        axes[0].plot(times, mean_rmse, color="tab:blue", lw=2.0)
        axes[1].plot(times, mean_rel_l2, color="tab:orange", lw=2.0)
        axes[0].set_title("Mean RMSE")
        axes[1].set_title("Mean rel-L2")
        for ax in axes:
            ax.set_xlabel("time")
            ax.grid(alpha=0.3)
        fig.savefig(out_dir / "mean_error_curves.png", dpi=170)
        plt.close(fig)

    # Representative sample diagnostics when rollout trajectories are present.
    sample_pred_traj = np.asarray(payload.get("sample_pred_traj", np.array([])), dtype=np.float64)
    sample_true_traj = np.asarray(payload.get("sample_true_traj", np.array([])), dtype=np.float64)
    sample_traj_id_raw = payload.get("sample_traj_id", None)
    sample_traj_id: int | None = None
    if sample_traj_id_raw is not None:
        sample_traj_id_val = int(np.asarray(sample_traj_id_raw).reshape(-1)[0])
        if sample_traj_id_val >= 0:
            sample_traj_id = sample_traj_id_val
    overlay_status = "not_available"
    overlay_steps: List[int] = []
    nx_used: int | None = None
    ny_used: int | None = None
    shape_source: str | None = None
    if (
        sample_pred_traj.ndim == 2
        and sample_true_traj.ndim == 2
        and sample_pred_traj.shape == sample_true_traj.shape
        and sample_pred_traj.shape[0] >= 1
        and sample_pred_traj.shape[1] >= 4
    ):
        n_frames = int(sample_pred_traj.shape[0])
        n_nodes = int(sample_pred_traj.shape[1])
        max_step = n_frames - 1
        try:
            nx_used, ny_used, shape_source = _infer_grid_shape(
                n_nodes=n_nodes,
                nx_arg=args.nx,
                ny_arg=args.ny,
            )
            # Convert flattened states back to [frame, nx, ny].
            pred_traj_2d = sample_pred_traj.reshape(n_frames, nx_used, ny_used)
            true_traj_2d = sample_true_traj.reshape(n_frames, nx_used, ny_used)

            parsed_steps = _parse_steps(args.steps, max_step=max_step)
            overlay_steps = (
                parsed_steps
                if len(parsed_steps) > 0
                else _auto_steps(max_step=max_step, num_steps=int(args.num_step_plots))
            )
            _plot_state_and_spectrum_overlays_2d(
                pred_traj_2d=pred_traj_2d,
                true_traj_2d=true_traj_2d,
                steps=overlay_steps,
                dt=float(args.dt),
                sample_id=sample_traj_id,
                out_state_path=out_dir / "state_overlay_steps.png",
                out_spec_path=out_dir / "spectrum_overlay_steps.png",
                dpi=170,
                spectrum_logy=bool(args.spectrum_logy),
            )
            overlay_status = "ok"
        except Exception as exc:
            overlay_status = f"failed: {type(exc).__name__}: {exc}"

    summary = {
        "rollout_npz": str(args.rollout_npz),
        "horizon": int(horizon),
        "final_mean_rmse": _safe_last(mean_rmse),
        "final_mean_rel_l2": _safe_last(mean_rel_l2),
        "num_samples": int(np.asarray(payload.get("sample_ids", np.array([]))).size),
        "overlay_status": overlay_status,
        "overlay_steps": overlay_steps,
        "overlay_sample_id": sample_traj_id,
        "overlay_nx": nx_used,
        "overlay_ny": ny_used,
        "overlay_shape_source": shape_source,
    }

    with open(out_dir / "diagnostics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Diagnostics complete.")
    print(f"  out_dir: {out_dir}")
    print(f"  horizon: {summary['horizon']}")
    print(f"  final_mean_rmse: {summary['final_mean_rmse']}")
    print(f"  final_mean_rel_l2: {summary['final_mean_rel_l2']}")
    print(f"  overlay_status: {summary['overlay_status']}")


if __name__ == "__main__":
    main()
