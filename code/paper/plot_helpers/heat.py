# Rendering-only extraction; heat snapshot axis label corrected to Node Index
# to match the author's 10-13-26 manuscript. Numerical calculations unchanged.
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Sequence, Mapping, Optional, Iterable
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import ScalarFormatter


# Source: code/heat_equation/scripts/rollout_gnn_heat.py (_replace_nonfinite_with_nan)
def _replace_nonfinite_with_nan(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float64).copy()
    out[~np.isfinite(out)] = np.nan
    return out


# Source: code/heat_equation/scripts/rollout_gnn_heat.py (_write_rollout_plots)
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
                ax.set_xlabel("Node Index", fontsize=12)
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
