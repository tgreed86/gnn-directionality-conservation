#!/usr/bin/env python3
"""Autoregressive rollout evaluation for trained heat-equation GNN."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from heat_data import (
    exact_heat_spectral_step,
    exact_heat_spectral_step_1d,
    load_heat_dataset,
)
from gnn_heat import (
    build_model_from_config,
    predict_type_from_target_mode,
    resolve_device,
    save_json,
)
from input_features import (
    apply_state_minmax_minus1_1,
    append_position_channels,
    count_input_feature_channels,
    resolve_input_feature_cfg,
)


def _replace_nonfinite_with_nan(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float64).copy()
    out[~np.isfinite(out)] = np.nan
    return out


def _nanmean_over_samples(arr: np.ndarray, horizon: int) -> np.ndarray:
    if arr.shape[0] == 0:
        return np.zeros((horizon,), dtype=np.float64)
    out = np.full((horizon,), np.nan, dtype=np.float64)
    for j in range(horizon):
        col = arr[:, j]
        finite = col[np.isfinite(col)]
        if finite.size > 0:
            out[j] = float(finite.mean())
    return out


def _parse_sample_plot_steps(raw: str | None, *, horizon: int) -> List[int]:
    """Parse representative-sample plot steps; step 0 is the initial state."""
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}.")
    if raw is None or str(raw).strip() == "":
        return [0, int(horizon)]

    steps: List[int] = []
    for token in str(raw).split(","):
        key = token.strip().lower()
        if key == "":
            continue
        if key in {"last", "final", "horizon"}:
            step = int(horizon)
        else:
            step = int(key)
        if step < 0 or step > int(horizon):
            raise ValueError(
                "sample plot steps must satisfy 0 <= step <= horizon. "
                f"Got step={step} with horizon={horizon}."
            )
        if step not in steps:
            steps.append(step)

    if len(steps) == 0:
        raise ValueError("--sample-plot-steps did not contain any valid steps.")
    return steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run autoregressive rollout evaluation.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to checkpoint_best.pt or checkpoint_last.pt from training.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Optional dataset path override. Defaults to checkpoint config path.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Optional rollout horizon override.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Optional number of test samples override.",
    )
    parser.add_argument(
        "--allow-full-dataset-fallback",
        action="store_true",
        help=(
            "Allow fallback to using all dataset samples when checkpoint "
            "split_indices.test is missing. Disabled by default for safety."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output npz path for rollout curves.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help=(
            "Directory for rollout visualizations. "
            "Defaults to '<checkpoint_dir>/rollout_plots'."
        ),
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable visualization generation.",
    )
    parser.add_argument(
        "--delta-error-mode",
        type=str,
        default="percent",
        choices=("percent", "absolute"),
        help=(
            "Error scaling used in the optional representative-sample GIF. "
            "'percent' plots 100*|pred-true|/max(|true| over final frame, eps) (default), "
            "'absolute' plots |pred-true|."
        ),
    )
    parser.add_argument(
        "--delta-percent-eps",
        type=float,
        default=1e-8,
        help=(
            "Denominator floor eps for global percent error mode: "
            "100*|pred-true|/max(max(|true| over final frame), eps)."
        ),
    )
    parser.add_argument(
        "--make-sample-gif",
        action="store_true",
        help=(
            "Also write a GIF that shows the representative sample's "
            "Exact / Predicted / Error panels over all rollout steps."
        ),
    )
    parser.add_argument(
        "--gif-out",
        type=Path,
        default=None,
        help=(
            "Optional output path for representative-sample GIF. "
            "Default: <plots_dir>/rollout_sample_evolution.gif"
        ),
    )
    parser.add_argument(
        "--gif-fps",
        type=int,
        default=8,
        help="Frames per second for representative-sample GIF.",
    )
    parser.add_argument(
        "--sample-plot-steps",
        type=str,
        default=None,
        help=(
            "Comma-separated rollout step indices to show in "
            "rollout_sample_final_state.png. Step 0 is the initial condition; "
            "use 'last' for the rollout horizon. Default: 0,last."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=("auto", "cpu", "cuda", "mps"),
        help=(
            "Optional device override for rollout inference. "
            "If omitted, uses the device stored in the checkpoint config."
        ),
    )
    return parser.parse_args()


def _rollout_one(
    *,
    model: torch.nn.Module,
    x0: torch.Tensor,
    edge_index: torch.Tensor,
    pos_nodes: torch.Tensor,
    nu: float,
    dt: float,
    Lx: float,
    Ly: float,
    nx: int,
    ny: int,
    spatial_dim: int,
    horizon: int,
    device: torch.device,
    input_cfg,
) -> Dict[str, np.ndarray]:
    model.eval()
    x_pred = x0.clone().detach()
    x_true = x0.clone().detach()

    rmse = np.zeros((horizon,), dtype=np.float64)
    rel_l2 = np.zeros((horizon,), dtype=np.float64)
    pred_traj = np.zeros((horizon + 1, int(x0.shape[0])), dtype=np.float64)
    true_traj = np.zeros((horizon + 1, int(x0.shape[0])), dtype=np.float64)
    pred_traj[0] = x_pred.squeeze(-1).cpu().numpy().astype(np.float64)
    true_traj[0] = x_true.squeeze(-1).cpu().numpy().astype(np.float64)

    with torch.no_grad():
        for k in range(horizon):
            x_in = x_pred.to(device=device, dtype=torch.float32)
            ei = edge_index.to(device=device)
            x_model = append_position_channels(
                x_state=x_in,
                pos_nodes=pos_nodes,
                input_cfg=input_cfg,
                batch_size=None,
            )
            x_model = apply_state_minmax_minus1_1(
                x_model=x_model,
                input_cfg=input_cfg,
            )
            y = model.predict_state(x_model, ei, dt=dt)
            x_pred = y.detach().cpu()

            # Exact one-step state via spectral heat evolution.
            if spatial_dim == 1:
                x_true_np = x_true.squeeze(-1).cpu().numpy().reshape(nx)
                x_true_next = exact_heat_spectral_step_1d(
                    x_true_np,
                    nu=nu,
                    dt=dt,
                    Lx=Lx,
                ).reshape(nx)
            elif spatial_dim == 2:
                x_true_np = x_true.squeeze(-1).cpu().numpy().reshape(nx, ny)
                x_true_next = exact_heat_spectral_step(
                    x_true_np,
                    nu=nu,
                    dt=dt,
                    Lx=Lx,
                    Ly=Ly,
                ).reshape(nx * ny)
            else:
                raise ValueError(f"Unsupported spatial_dim={spatial_dim}.")
            x_true = torch.from_numpy(x_true_next[:, None]).to(dtype=torch.float32)
            pred_traj[k + 1] = x_pred.squeeze(-1).cpu().numpy().astype(np.float64)
            true_traj[k + 1] = x_true.squeeze(-1).cpu().numpy().astype(np.float64)

            err = x_pred - x_true
            rmse[k] = math.sqrt(float(torch.mean(err * err).item()))
            rel_l2[k] = float(torch.norm(err).item() / max(torch.norm(x_true).item(), 1e-12))

    return {
        "rmse": rmse,
        "rel_l2": rel_l2,
        "pred_traj": pred_traj,
        "true_traj": true_traj,
    }


def _write_rollout_plots(
    *,
    plots_dir: Path,
    horizon: int,
    sample_ids: List[int],
    rmse_arr: np.ndarray,
    rel_l2_arr: np.ndarray,
    mean_rmse: np.ndarray,
    mean_rel_l2: np.ndarray,
    dt: float,
    nx: int,
    ny: int,
    spatial_dim: int,
    delta_error_mode: str,
    delta_percent_eps: float,
    make_sample_gif: bool,
    gif_out: Path | None,
    gif_fps: int,
    sample_plot_steps: List[int],
    sample_traj_id: int | None,
    sample_pred_traj: np.ndarray | None,
    sample_true_traj: np.ndarray | None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir.mkdir(parents=True, exist_ok=True)

    steps = np.arange(1, horizon + 1, dtype=np.int64)
    times = steps.astype(np.float64) * float(dt)

    # 1) Mean error curves + per-sample faint lines.
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    ax0, ax1 = axes
    if rmse_arr.shape[0] > 0:
        for i in range(rmse_arr.shape[0]):
            ax0.plot(times, rmse_arr[i], color="tab:blue", alpha=0.15, lw=0.9)
            ax1.plot(times, rel_l2_arr[i], color="tab:orange", alpha=0.15, lw=0.9)
    ax0.plot(times, mean_rmse, color="tab:blue", lw=2.2, label="mean RMSE")
    ax1.plot(times, mean_rel_l2, color="tab:orange", lw=2.2, label="mean rel-L2")
    ax0.set_title("Rollout RMSE vs Time")
    ax1.set_title("Rollout Relative L2 vs Time")
    ax0.set_xlabel("time")
    ax1.set_xlabel("time")
    ax0.set_ylabel("RMSE")
    ax1.set_ylabel("rel-L2")
    ax0.grid(alpha=0.3)
    ax1.grid(alpha=0.3)
    ax0.legend(loc="upper left")
    ax1.legend(loc="upper left")
    fig.suptitle(
        f"Heat Rollout Error Curves | samples={len(sample_ids)} | horizon={horizon}",
        fontsize=11,
    )
    fig.savefig(plots_dir / "rollout_error_curves.png", dpi=160)
    plt.close(fig)

    # 2) Final-step error distribution.
    if rmse_arr.shape[0] > 0:
        fig2, axes2 = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
        ax20, ax21 = axes2
        final_rmse = rmse_arr[:, -1]
        final_rel = rel_l2_arr[:, -1]
        final_rmse_finite = final_rmse[np.isfinite(final_rmse)]
        final_rel_finite = final_rel[np.isfinite(final_rel)]
        if final_rmse_finite.size > 0:
            ax20.hist(
                final_rmse_finite,
                bins=min(20, max(5, len(final_rmse_finite))),
                color="tab:blue",
                alpha=0.75,
            )
        if final_rel_finite.size > 0:
            ax21.hist(
                final_rel_finite,
                bins=min(20, max(5, len(final_rel_finite))),
                color="tab:orange",
                alpha=0.75,
            )
        ax20.set_title("Final-Step RMSE Distribution")
        ax21.set_title("Final-Step rel-L2 Distribution")
        ax20.set_xlabel("RMSE at final step")
        ax21.set_xlabel("rel-L2 at final step")
        ax20.set_ylabel("count")
        ax21.set_ylabel("count")
        ax20.grid(alpha=0.25)
        ax21.grid(alpha=0.25)
        fig2.savefig(plots_dir / "rollout_final_error_hist.png", dpi=160)
        plt.close(fig2)

    # 3) Representative sample final-state comparison.
    if (
        sample_traj_id is not None
        and sample_pred_traj is not None
        and sample_true_traj is not None
    ):
        if spatial_dim == 1:
            x_idx = np.arange(nx, dtype=np.int64)
            pred_traj_1d = _replace_nonfinite_with_nan(
                sample_pred_traj.reshape(sample_pred_traj.shape[0], nx)
            )
            true_traj_1d = _replace_nonfinite_with_nan(
                sample_true_traj.reshape(sample_true_traj.shape[0], nx)
            )
            n_frames = int(pred_traj_1d.shape[0])
            if n_frames != int(true_traj_1d.shape[0]):
                raise ValueError(
                    "Representative predicted/true trajectories have different frame counts: "
                    f"{n_frames} vs {int(true_traj_1d.shape[0])}."
                )
            bad_steps = [s for s in sample_plot_steps if s < 0 or s >= n_frames]
            if bad_steps:
                raise ValueError(
                    "sample_plot_steps contains steps outside the representative "
                    f"trajectory frame range [0, {n_frames - 1}]: {bad_steps}."
                )

            finite_parts: List[np.ndarray] = []
            for step in sample_plot_steps:
                true_finite = true_traj_1d[step][np.isfinite(true_traj_1d[step])]
                pred_finite = pred_traj_1d[step][np.isfinite(pred_traj_1d[step])]
                if true_finite.size > 0:
                    finite_parts.append(true_finite)
                if pred_finite.size > 0:
                    finite_parts.append(pred_finite)
            finite_state = np.concatenate(
                finite_parts if finite_parts else [np.array([], dtype=np.float64)]
            )
            if finite_state.size == 0:
                state_ylim = (-1.0, 1.0)
            else:
                ymin = float(np.min(finite_state))
                ymax = float(np.max(finite_state))
                if ymax <= ymin:
                    pad = max(1e-6, abs(ymin) * 1e-6)
                else:
                    pad = 0.05 * (ymax - ymin)
                state_ylim = (ymin - pad, ymax + pad)

            n_steps = len(sample_plot_steps)
            ncols = min(2, n_steps) if n_steps <= 4 else min(3, n_steps)
            nrows = int(math.ceil(n_steps / ncols))
            fig3, axes3 = plt.subplots(
                nrows,
                ncols,
                figsize=(5.4 * ncols, 3.7 * nrows),
                constrained_layout=True,
            )
            axes_flat = np.atleast_1d(axes3).reshape(-1)
            for ax, step in zip(axes_flat, sample_plot_steps):
                ax.plot(
                    x_idx,
                    true_traj_1d[step],
                    color="tab:blue",
                    lw=1.9,
                    label="Exact",
                )
                ax.plot(
                    x_idx,
                    pred_traj_1d[step],
                    color="tab:orange",
                    lw=1.7,
                    linestyle="--",
                    label="Predicted",
                )
                #ax.set_title(f"step={step}, t={step * float(dt):.3e}")
                ax.set_title(f"Time Step={step}", fontsize=15)
                ax.set_xlabel("Position", fontsize=12)
                ax.set_ylabel("Amplitude", fontsize=12)
                ax.set_ylim(*state_ylim)
                ax.grid(alpha=0.3)
                ax.legend(loc="best")
            for ax in axes_flat[n_steps:]:
                ax.set_visible(False)

            fig3.suptitle(
                f"1D Heat Equation Rollout",
                #f"1D Heat Equation Rollout Sample | sample_id={sample_traj_id}",
                fontsize=20,
            )
            fig3.savefig(plots_dir / "rollout_sample_final_state.png", dpi=170)
            plt.close(fig3)

            if make_sample_gif:
                from matplotlib import animation as mplanimation

                gif_path = gif_out if gif_out is not None else (plots_dir / "rollout_sample_evolution.gif")
                gif_path = Path(gif_path)
                gif_path.parent.mkdir(parents=True, exist_ok=True)

                mode_key = str(delta_error_mode).strip().lower()
                eps = max(float(delta_percent_eps), 0.0)
                true_last_abs_max = float(np.nanmax(np.abs(true_traj_1d[-1])))
                denom_global = max(true_last_abs_max, eps, 1e-12)

                def _err_frame(pred_frame: np.ndarray, true_frame: np.ndarray) -> np.ndarray:
                    if mode_key == "absolute":
                        return np.abs(pred_frame - true_frame)
                    if mode_key == "percent":
                        return 100.0 * np.abs(pred_frame - true_frame) / denom_global
                    raise ValueError(
                        f"Unsupported delta_error_mode='{delta_error_mode}'. "
                        "Use one of: percent, absolute."
                    )

                err_final = _err_frame(pred_traj_1d[-1], true_traj_1d[-1])
                finite_err_final = err_final[np.isfinite(err_final)]
                if finite_err_final.size == 0:
                    err_ylim = (0.0, 1.0)
                else:
                    err_ymax = float(np.max(finite_err_final))
                    err_ylim = (0.0, max(err_ymax, 1e-12) * 1.05)

                fig4, axes4 = plt.subplots(1, 3, figsize=(13.0, 4.2), constrained_layout=True)
                line0, = axes4[0].plot(x_idx, true_traj_1d[0], color="tab:blue", lw=1.8)
                line1, = axes4[1].plot(x_idx, pred_traj_1d[0], color="tab:orange", lw=1.8)
                line2, = axes4[2].plot(
                    x_idx,
                    _err_frame(pred_traj_1d[0], true_traj_1d[0]),
                    color="tab:red",
                    lw=1.8,
                )
                axes4[0].set_title("Exact State")
                axes4[1].set_title("Predicted State")
                axes4[2].set_title(err_title)
                for ax in axes4[:2]:
                    ax.set_xlabel("x-index")
                    ax.set_ylabel("state")
                    ax.set_ylim(*state_ylim)
                    ax.grid(alpha=0.3)
                axes4[2].set_xlabel("x-index")
                axes4[2].set_ylabel(err_ylabel)
                axes4[2].set_ylim(*err_ylim)
                axes4[2].grid(alpha=0.3)

                supt = fig4.suptitle(
                    f"Representative 1D Heat Rollout | sample_id={sample_traj_id} | step=0/{n_frames - 1}",
                    fontsize=11,
                )

                def _update(frame_idx: int):
                    line0.set_ydata(true_traj_1d[frame_idx])
                    line1.set_ydata(pred_traj_1d[frame_idx])
                    line2.set_ydata(_err_frame(pred_traj_1d[frame_idx], true_traj_1d[frame_idx]))
                    supt.set_text(
                        "Representative 1D Heat Rollout "
                        f"| sample_id={sample_traj_id} | step={frame_idx}/{n_frames - 1}"
                    )
                    return (line0, line1, line2, supt)

                ani = mplanimation.FuncAnimation(
                    fig4,
                    _update,
                    frames=n_frames,
                    interval=max(1, int(round(1000.0 / max(1, int(gif_fps))))),
                    blit=False,
                    repeat=True,
                )
                writer = mplanimation.PillowWriter(fps=max(1, int(gif_fps)))
                ani.save(str(gif_path), writer=writer, dpi=150)
                plt.close(fig4)
            return

        true_initial = _replace_nonfinite_with_nan(sample_true_traj[0].reshape(nx, ny))
        pred_last = _replace_nonfinite_with_nan(sample_pred_traj[-1].reshape(nx, ny))
        true_last = _replace_nonfinite_with_nan(sample_true_traj[-1].reshape(nx, ny))
        if str(delta_error_mode).strip().lower() == "absolute":
            err_last = np.abs(pred_last - true_last)
            err_title = "|Error| Final State (Absolute)"
            err_cbar_label = "absolute error"
        elif str(delta_error_mode).strip().lower() == "percent":
            eps = max(float(delta_percent_eps), 0.0)
            true_abs_max = float(np.nanmax(np.abs(true_last)))
            denom = max(true_abs_max, eps)
            err_last = 100.0 * np.abs(pred_last - true_last) / denom
            err_title = "|Error| Final State (%) [global scale]"
            err_cbar_label = "percent error (%) [global]"
        else:
            raise ValueError(
                f"Unsupported delta_error_mode='{delta_error_mode}'. "
                "Use one of: percent, absolute."
            )

        finite_ref = true_last[np.isfinite(true_last)]
        if finite_ref.size == 0:
            vmin, vmax = -1.0, 1.0
        else:
            vmin = float(np.min(finite_ref))
            vmax = float(np.max(finite_ref))
            if vmax <= vmin:
                eps = max(1e-6, abs(vmin) * 1e-6)
                vmin -= eps
                vmax += eps
        finite_initial = true_initial[np.isfinite(true_initial)]
        if finite_initial.size == 0:
            vmin_initial, vmax_initial = -1.0, 1.0
        else:
            vmin_initial = float(np.min(finite_initial))
            vmax_initial = float(np.max(finite_initial))
            if vmax_initial <= vmin_initial:
                eps = max(1e-6, abs(vmin_initial) * 1e-6)
                vmin_initial -= eps
                vmax_initial += eps

        fig3, axes3 = plt.subplots(2, 2, figsize=(9.6, 8.4), constrained_layout=True)
        ax_initial = axes3[0, 0]
        ax_true_final = axes3[0, 1]
        ax_pred_final = axes3[1, 0]
        ax_error = axes3[1, 1]

        im_initial = ax_initial.imshow(
            true_initial,
            origin="lower",
            cmap="viridis",
            vmin=vmin_initial,
            vmax=vmax_initial,
        )
        im_true_final = ax_true_final.imshow(
            true_last,
            origin="lower",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        im_pred_final = ax_pred_final.imshow(
            pred_last,
            origin="lower",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        im_error = ax_error.imshow(err_last, origin="lower", cmap="magma")

        ax_initial.set_title("Exact Initial State")
        ax_true_final.set_title("Exact Final State")
        ax_pred_final.set_title("Predicted Final State")
        ax_error.set_title(err_title)
        for ax in axes3.reshape(-1):
            ax.set_xlabel("y-index")
            ax.set_ylabel("x-index")

        cbar_initial = fig3.colorbar(im_initial, ax=ax_initial, fraction=0.046, pad=0.04)
        cbar_true_final = fig3.colorbar(im_true_final, ax=ax_true_final, fraction=0.046, pad=0.04)
        cbar_pred_final = fig3.colorbar(im_pred_final, ax=ax_pred_final, fraction=0.046, pad=0.04)
        cbar_initial.set_label("state")
        cbar_true_final.set_label("state")
        cbar_pred_final.set_label("state")
        error_cbar = fig3.colorbar(im_error, ax=ax_error, fraction=0.046, pad=0.04)
        error_cbar.set_label(err_cbar_label)
        fig3.suptitle(
            f"Representative Heat Rollout Sample | sample_id={sample_traj_id}",
            fontsize=11,
        )
        fig3.savefig(plots_dir / "rollout_sample_final_state.png", dpi=170)
        plt.close(fig3)

        # 4) Representative sample animation over all rollout steps.
        if make_sample_gif:
            from matplotlib import animation as mplanimation

            gif_path = gif_out if gif_out is not None else (plots_dir / "rollout_sample_evolution.gif")
            gif_path = Path(gif_path)
            gif_path.parent.mkdir(parents=True, exist_ok=True)

            pred_traj_2d = _replace_nonfinite_with_nan(
                sample_pred_traj.reshape(sample_pred_traj.shape[0], nx, ny)
            )
            true_traj_2d = _replace_nonfinite_with_nan(
                sample_true_traj.reshape(sample_true_traj.shape[0], nx, ny)
            )
            n_frames = int(pred_traj_2d.shape[0])
            if n_frames != int(true_traj_2d.shape[0]):
                raise ValueError(
                    "Representative predicted/true trajectories have different frame counts: "
                    f"{n_frames} vs {int(true_traj_2d.shape[0])}."
                )

            mode_key = str(delta_error_mode).strip().lower()
            eps = max(float(delta_percent_eps), 0.0)
            true_last_abs_max = float(np.nanmax(np.abs(true_traj_2d[-1])))
            denom_global = max(true_last_abs_max, eps, 1e-12)

            def _err_frame(pred_frame: np.ndarray, true_frame: np.ndarray) -> np.ndarray:
                if mode_key == "absolute":
                    return np.abs(pred_frame - true_frame)
                if mode_key == "percent":
                    return 100.0 * np.abs(pred_frame - true_frame) / denom_global
                raise ValueError(
                    f"Unsupported delta_error_mode='{delta_error_mode}'. "
                    "Use one of: percent, absolute."
                )

            err_final = _err_frame(pred_traj_2d[-1], true_traj_2d[-1])
            finite_err_final = err_final[np.isfinite(err_final)]
            if finite_err_final.size == 0:
                err_vmax = 1.0
            else:
                err_vmax = float(np.max(finite_err_final))
                if err_vmax <= 0.0:
                    err_vmax = 1e-12

            fig4, axes4 = plt.subplots(1, 3, figsize=(13.0, 4.2), constrained_layout=True)
            im0 = axes4[0].imshow(
                true_traj_2d[0],
                origin="lower",
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
            )
            im1 = axes4[1].imshow(
                pred_traj_2d[0],
                origin="lower",
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
            )
            err0 = _err_frame(pred_traj_2d[0], true_traj_2d[0])
            im2 = axes4[2].imshow(
                err0,
                origin="lower",
                cmap="magma",
                vmin=0.0,
                vmax=err_vmax,
            )

            axes4[0].set_title("Exact State")
            axes4[1].set_title("Predicted State")
            axes4[2].set_title(err_title)
            for ax in axes4:
                ax.set_xlabel("y-index")
                ax.set_ylabel("x-index")

            cbar0 = fig4.colorbar(im0, ax=axes4[0], fraction=0.046, pad=0.04)
            cbar1 = fig4.colorbar(im1, ax=axes4[1], fraction=0.046, pad=0.04)
            cbar2 = fig4.colorbar(im2, ax=axes4[2], fraction=0.046, pad=0.04)
            del cbar0, cbar1
            cbar2.set_label(err_cbar_label)

            supt = fig4.suptitle(
                f"Representative Heat Rollout | sample_id={sample_traj_id} | step=0/{n_frames - 1}",
                fontsize=11,
            )

            def _update(frame_idx: int):
                im0.set_data(true_traj_2d[frame_idx])
                im1.set_data(pred_traj_2d[frame_idx])
                im2.set_data(_err_frame(pred_traj_2d[frame_idx], true_traj_2d[frame_idx]))
                supt.set_text(
                    "Representative Heat Rollout "
                    f"| sample_id={sample_traj_id} | step={frame_idx}/{n_frames - 1}"
                )
                return (im0, im1, im2, supt)

            ani = mplanimation.FuncAnimation(
                fig4,
                _update,
                frames=n_frames,
                interval=max(1, int(round(1000.0 / max(1, int(gif_fps))))),
                blit=False,
                repeat=True,
            )
            writer = mplanimation.PillowWriter(fps=max(1, int(gif_fps)))
            ani.save(str(gif_path), writer=writer, dpi=150)
            plt.close(fig4)


def main() -> None:
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    data_cfg = dict(cfg.get("data", {}))
    rollout_cfg = dict(cfg.get("rollout_eval", {}))
    target_mode = str(data_cfg.get("target_mode", "state")).strip().lower()

    cfg_model = dict(cfg.get("model", {}))
    cfg_model["predict_type"] = predict_type_from_target_mode(target_mode)
    cfg["model"] = cfg_model

    dataset_path = args.dataset or Path(data_cfg.get("dataset_path", "data/single_steps/heat_train.npz"))
    horizon = int(
        args.horizon
        if args.horizon is not None
        else rollout_cfg.get("horizon", 100)
    )
    sample_plot_steps = _parse_sample_plot_steps(
        args.sample_plot_steps,
        horizon=horizon,
    )
    num_samples = int(
        args.num_samples
        if args.num_samples is not None
        else rollout_cfg.get("num_test_samples", 16)
    )
    if num_samples <= 0:
        raise ValueError(f"--num-samples must be positive, got {num_samples}.")

    dset = load_heat_dataset(dataset_path, dtype=torch.float32)
    metadata = dset.metadata
    config_spatial_dim_raw = data_cfg.get("spatial-dim", data_cfg.get("spatial_dim", None))
    config_spatial_dim = (
        None if config_spatial_dim_raw is None else int(config_spatial_dim_raw)
    )
    spatial_dim = int(metadata.get("spatial_dim", config_spatial_dim or 2))
    if spatial_dim not in {1, 2}:
        raise ValueError(
            f"Dataset metadata spatial_dim must be 1 or 2, got {spatial_dim}."
        )
    if config_spatial_dim is not None and spatial_dim != config_spatial_dim:
        raise ValueError(
            "Checkpoint config data.spatial-dim does not match dataset metadata for "
            f"{dataset_path}: config has {config_spatial_dim}, dataset has "
            f"{spatial_dim}."
        )
    nx = int(metadata["nx"])
    ny = int(metadata["ny"])
    nu = float(metadata["nu"])
    dt = float(metadata["dt"])
    Lx = float(metadata["Lx"])
    Ly = float(metadata["Ly"])
    input_cfg = resolve_input_feature_cfg(cfg)
    if input_cfg.normalize_state_to_minus1_1:
        if input_cfg.state_min is None or input_cfg.state_max is None:
            raise ValueError(
                "input_features.normalize_state_to_minus1_1 is enabled, but "
                "state_min/state_max are missing from checkpoint config. "
                "Retrain with normalization enabled to persist these stats."
            )
        print(
            "[INFO] State normalization enabled: "
            f"range=[-1,1], state_min={float(input_cfg.state_min):.6e}, "
            f"state_max={float(input_cfg.state_max):.6e}, "
            f"clip={bool(input_cfg.normalize_clip)}",
            flush=True,
        )
    else:
        print("[INFO] State normalization disabled.", flush=True)

    split = ckpt.get("split_indices", {})
    test_idx_raw = split.get("test", None) if isinstance(split, dict) else None
    selection_mode = "test_split"
    if test_idx_raw is None:
        if not bool(args.allow_full_dataset_fallback):
            raise ValueError(
                "checkpoint split_indices.test is missing. "
                "Refusing to fall back to full dataset by default. "
                "If this is intentional, rerun with --allow-full-dataset-fallback."
            )
        selection_mode = "full_dataset_fallback"
        test_idx_list = list(range(len(dset)))
    else:
        try:
            test_idx_list = [int(i) for i in list(test_idx_raw)]
        except Exception as exc:
            raise ValueError(
                "checkpoint split_indices.test could not be parsed as a list "
                f"of integer indices: {type(exc).__name__}: {exc}"
            ) from exc
        if len(test_idx_list) == 0:
            raise ValueError(
                "checkpoint split_indices.test is present but empty. "
                "Cannot run rollout evaluation on an empty test split."
            )

    bad_ids = [i for i in test_idx_list if (i < 0 or i >= len(dset))]
    if bad_ids:
        preview = bad_ids[:10]
        raise ValueError(
            "checkpoint split_indices.test contains out-of-range dataset indices. "
            f"Dataset size={len(dset)}, bad ids (first up to 10)={preview}."
        )

    sample_ids = list(test_idx_list[: min(num_samples, len(test_idx_list))])
    if len(sample_ids) == 0:
        raise ValueError(
            "No sample ids selected for rollout evaluation. "
            "Check --num-samples and checkpoint split_indices.test."
        )

    device_pref = str(args.device if args.device is not None else cfg.get("device", "auto"))
    device = resolve_device(device_pref)
    in_channels = 1 + count_input_feature_channels(input_cfg)
    model = build_model_from_config(cfg, in_channels=in_channels).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    edge_index = dset.edge_index.detach().cpu().long()
    pos_nodes = dset.pos.detach().cpu()

    rmse_all: List[np.ndarray] = []
    rel_l2_all: List[np.ndarray] = []
    sample_traj_id: int | None = None
    sample_pred_traj: np.ndarray | None = None
    sample_true_traj: np.ndarray | None = None
    for sid in sample_ids:
        x0 = dset.x[int(sid)].detach().cpu()
        curves = _rollout_one(
            model=model,
            x0=x0,
            edge_index=edge_index,
            pos_nodes=pos_nodes,
            nu=nu,
            dt=dt,
            Lx=Lx,
            Ly=Ly,
            nx=nx,
            ny=ny,
            spatial_dim=spatial_dim,
            horizon=horizon,
            device=device,
            input_cfg=input_cfg,
        )
        rmse_all.append(curves["rmse"])
        rel_l2_all.append(curves["rel_l2"])
        if sample_traj_id is None:
            sample_traj_id = int(sid)
            sample_pred_traj = curves["pred_traj"]
            sample_true_traj = curves["true_traj"]

    rmse_arr_raw = np.stack(rmse_all, axis=0) if rmse_all else np.zeros((0, horizon))
    rel_l2_arr_raw = np.stack(rel_l2_all, axis=0) if rel_l2_all else np.zeros((0, horizon))
    rmse_arr = _replace_nonfinite_with_nan(rmse_arr_raw)
    rel_l2_arr = _replace_nonfinite_with_nan(rel_l2_arr_raw)

    mean_rmse = _nanmean_over_samples(rmse_arr, horizon)
    mean_rel_l2 = _nanmean_over_samples(rel_l2_arr, horizon)
    nonfinite_rmse_count = int(np.count_nonzero(~np.isfinite(rmse_arr_raw)))
    nonfinite_rel_count = int(np.count_nonzero(~np.isfinite(rel_l2_arr_raw)))

    out_npz = args.out
    if out_npz is None:
        out_npz = args.checkpoint.parent / "rollout_eval.npz"
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        sample_ids=np.array(sample_ids, dtype=np.int64),
        rmse_by_sample=rmse_arr,
        rel_l2_by_sample=rel_l2_arr,
        mean_rmse=mean_rmse,
        mean_rel_l2=mean_rel_l2,
        horizon=np.int64(horizon),
        sample_traj_id=np.int64(-1 if sample_traj_id is None else sample_traj_id),
        sample_pred_traj=np.array([]) if sample_pred_traj is None else sample_pred_traj,
        sample_true_traj=np.array([]) if sample_true_traj is None else sample_true_traj,
        spatial_dim=np.int64(spatial_dim),
        nx=np.int64(nx),
        ny=np.int64(ny),
        sample_plot_steps=np.array(sample_plot_steps, dtype=np.int64),
    )

    summary = {
        "checkpoint": str(args.checkpoint),
        "dataset": str(dataset_path),
        "sample_selection_mode": selection_mode,
        "allow_full_dataset_fallback": bool(args.allow_full_dataset_fallback),
        "device_requested": device_pref,
        "device_resolved": str(device),
        "num_samples": int(len(sample_ids)),
        "horizon": int(horizon),
        "spatial_dim": int(spatial_dim),
        "nx": int(nx),
        "ny": int(ny),
        "model_in_channels": int(in_channels),
        "input_features": {
            "include_pos": bool(input_cfg.include_pos),
            "normalize_state_to_minus1_1": bool(
                input_cfg.normalize_state_to_minus1_1
            ),
            "normalize_clip": bool(input_cfg.normalize_clip),
            "state_min": (
                None if input_cfg.state_min is None else float(input_cfg.state_min)
            ),
            "state_max": (
                None if input_cfg.state_max is None else float(input_cfg.state_max)
            ),
        },
        "final_mean_rmse": (
            float(mean_rmse[-1]) if (mean_rmse.size > 0 and np.isfinite(mean_rmse[-1])) else None
        ),
        "final_mean_rel_l2": (
            float(mean_rel_l2[-1]) if (mean_rel_l2.size > 0 and np.isfinite(mean_rel_l2[-1])) else None
        ),
        "nonfinite_rmse_count": nonfinite_rmse_count,
        "nonfinite_rel_l2_count": nonfinite_rel_count,
        "delta_error_mode": str(args.delta_error_mode),
        "delta_percent_eps": float(args.delta_percent_eps),
        "sample_gif_enabled": bool(args.make_sample_gif),
        "sample_gif_fps": int(args.gif_fps),
        "sample_gif_out": (
            None
            if args.gif_out is None
            else str(args.gif_out)
        ),
        "sample_plot_steps": [int(s) for s in sample_plot_steps],
        "out_npz": str(out_npz),
    }

    plots_dir = args.plots_dir
    if plots_dir is None:
        plots_dir = args.checkpoint.parent / "rollout_plots"
    if not args.no_plots:
        _write_rollout_plots(
            plots_dir=plots_dir,
            horizon=horizon,
            sample_ids=sample_ids,
            rmse_arr=rmse_arr,
            rel_l2_arr=rel_l2_arr,
            mean_rmse=mean_rmse,
            mean_rel_l2=mean_rel_l2,
            dt=dt,
            nx=nx,
            ny=ny,
            spatial_dim=spatial_dim,
            delta_error_mode=str(args.delta_error_mode),
            delta_percent_eps=float(args.delta_percent_eps),
            make_sample_gif=bool(args.make_sample_gif),
            gif_out=args.gif_out,
            gif_fps=int(args.gif_fps),
            sample_plot_steps=sample_plot_steps,
            sample_traj_id=sample_traj_id,
            sample_pred_traj=sample_pred_traj,
            sample_true_traj=sample_true_traj,
        )
        summary["plots_dir"] = str(plots_dir)

    save_json(summary, args.checkpoint.parent / "rollout_eval_summary.json")

    print("Rollout evaluation complete.")
    print(f"  samples: {len(sample_ids)}")
    print(f"  horizon: {horizon}")
    print(f"  spatial_dim: {spatial_dim}")
    print(f"  sample_plot_steps: {sample_plot_steps}")
    print(f"  device: {device}")
    print(f"  out: {out_npz}")
    if not args.no_plots:
        print(f"  plots: {plots_dir}")
    if mean_rmse.size > 0 and np.isfinite(mean_rmse[-1]) and np.isfinite(mean_rel_l2[-1]):
        print(f"  final mean RMSE: {mean_rmse[-1]:.6e}")
        print(f"  final mean rel-L2: {mean_rel_l2[-1]:.6e}")
    else:
        print("  final mean metrics: non-finite (rollout divergence likely).")
    if (nonfinite_rmse_count > 0) or (nonfinite_rel_count > 0):
        print(
            f"  warning: non-finite values detected "
            f"(rmse={nonfinite_rmse_count}, rel_l2={nonfinite_rel_count})."
        )


if __name__ == "__main__":
    main()
