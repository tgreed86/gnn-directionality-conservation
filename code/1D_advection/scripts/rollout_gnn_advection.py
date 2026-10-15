#!/usr/bin/env python3
"""Autoregressive rollout evaluation for trained advection GNN.

In addition to writing rollout metrics (`rollout_eval.npz`), this script now
also writes visualization PNGs by default.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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

from advection_data import exact_advect_spectral_shift, load_advection_dataset
from gnn_advection import (
    build_model_from_config,
    predict_type_from_target_mode,
    resolve_device,
    resolve_state_residual_mode,
    save_json,
    state_residual_from_mode,
)
from physics_inputs import (
    ADVECTION_SCHEME_UPWIND,
    ADVECTION_SCHEMES,
    PhysicsInputConfig,
    build_physics_augmented_inputs_1d,
    compute_advdiff_terms_abs_1d,
    count_physics_input_channels,
    resolve_physics_input_cfg,
)
from input_features import (
    InputFeatureConfig,
    apply_state_minmax_minus1_1,
    append_position_channel,
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
        "--device",
        type=str,
        default=None,
        choices=("auto", "cpu", "cuda", "mps"),
        help=(
            "Optional device override for rollout inference. "
            "If omitted, uses the device stored in the checkpoint config."
        ),
    )
    parser.add_argument(
        "--state-residual-mode",
        type=str,
        default=None,
        help=(
            "Override model.state_residual_mode from the checkpoint config. "
            "Use 'raw' for current behavior or 'legacy'/'model_input' for the "
            "older normalized-input residual behavior."
        ),
    )
    return parser.parse_args()


@dataclass(frozen=True)
class GodunovCorrectionConfig:
    enabled: bool = False
    advection_scheme: str = ADVECTION_SCHEME_UPWIND
    c_override: float | None = None


def _resolve_godunov_correction_cfg(cfg: Dict) -> GodunovCorrectionConfig:
    train_cfg_raw = cfg.get("train", {})
    model_cfg_raw = cfg.get("model", {})
    train_cfg = train_cfg_raw if isinstance(train_cfg_raw, dict) else {}
    model_cfg = model_cfg_raw if isinstance(model_cfg_raw, dict) else {}
    raw = train_cfg.get("godunov_correction", model_cfg.get("godunov_correction", {}))

    if isinstance(raw, bool):
        enabled = bool(raw)
        scheme = ADVECTION_SCHEME_UPWIND
        c_override = None
    elif isinstance(raw, dict):
        enabled = bool(raw.get("enabled", False))
        scheme = str(
            raw.get("advection_scheme", raw.get("scheme", ADVECTION_SCHEME_UPWIND))
        ).strip().lower()
        c_raw = raw.get("c_override", raw.get("c", None))
        c_override = None if c_raw is None else float(c_raw)
    else:
        raise ValueError(
            "godunov_correction must be either a boolean or object, got "
            f"{type(raw).__name__}."
        )

    if scheme not in ADVECTION_SCHEMES:
        raise ValueError(
            "godunov_correction.advection_scheme must be one of "
            f"{sorted(ADVECTION_SCHEMES)}, got '{scheme}'."
        )
    return GodunovCorrectionConfig(
        enabled=enabled,
        advection_scheme=scheme,
        c_override=c_override,
    )


def _target_from_states(
    *,
    x_now: torch.Tensor,
    state_next: torch.Tensor,
    dt: float,
    target_mode: str,
) -> torch.Tensor:
    if target_mode == "state":
        return state_next
    if target_mode == "delta":
        return state_next - x_now
    if target_mode == "rate":
        return (state_next - x_now) / float(dt)
    raise RuntimeError(f"Unexpected target_mode='{target_mode}'.")


def _state_from_target(
    *,
    x_now: torch.Tensor,
    target: torch.Tensor,
    dt: float,
    target_mode: str,
) -> torch.Tensor:
    if target_mode == "state":
        return target
    if target_mode == "delta":
        return x_now + target
    if target_mode == "rate":
        return x_now + float(dt) * target
    raise RuntimeError(f"Unexpected target_mode='{target_mode}'.")


def _godunov_next_state(
    *,
    x_now: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    dt: float,
    dx: float,
    c_from_data: float,
    godunov_cfg: GodunovCorrectionConfig,
) -> torch.Tensor:
    c_eff = (
        float(godunov_cfg.c_override)
        if godunov_cfg.c_override is not None
        else float(c_from_data)
    )
    r_adv_abs, _ = compute_advdiff_terms_abs_1d(
        x_abs=x_now,
        edge_index=edge_index,
        edge_attr=edge_attr,
        c=c_eff,
        dx=float(dx),
        nu=0.0,
        advection_scheme=godunov_cfg.advection_scheme,
        compute_adv=True,
        compute_diff=False,
    )
    assert r_adv_abs is not None
    return x_now + float(dt) * r_adv_abs


def _rollout_one(
    *,
    model: torch.nn.Module,
    x0: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    pos_nodes: torch.Tensor,
    c: float,
    dt: float,
    dx: float,
    L: float,
    horizon: int,
    device: torch.device,
    physics_cfg: PhysicsInputConfig,
    input_cfg: InputFeatureConfig,
    target_mode: str,
    state_residual_mode: str,
    godunov_cfg: GodunovCorrectionConfig,
    true_traj_prefix: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    model.eval()
    x_pred = x0.clone().detach()
    x_true = x0.clone().detach()
    prefix: np.ndarray | None = None

    rmse = np.zeros((horizon,), dtype=np.float64)
    rel_l2 = np.zeros((horizon,), dtype=np.float64)
    pred_traj = np.zeros((horizon + 1, int(x0.shape[0])), dtype=np.float64)
    true_traj = np.zeros((horizon + 1, int(x0.shape[0])), dtype=np.float64)
    pred_traj[0] = x_pred.squeeze(-1).cpu().numpy().astype(np.float64)
    if true_traj_prefix is not None:
        prefix = np.asarray(true_traj_prefix, dtype=np.float64)
        if prefix.ndim != 2 or prefix.shape[1] != int(x0.shape[0]):
            raise ValueError(
                "true_traj_prefix must have shape [T, N], got "
                f"{tuple(prefix.shape)} for N={int(x0.shape[0])}."
            )
        if prefix.shape[0] < 1:
            raise ValueError("true_traj_prefix must contain at least one state.")
        true_traj[0] = prefix[0]
        x_true = torch.from_numpy(prefix[0][:, None]).to(dtype=torch.float32)
    else:
        true_traj[0] = x_true.squeeze(-1).cpu().numpy().astype(np.float64)

    with torch.no_grad():
        for k in range(horizon):
            # Predicted one-step state.
            x_in = x_pred.to(device=device, dtype=torch.float32)
            ei = edge_index.to(device=device)
            ea = edge_attr.to(device=device, dtype=x_in.dtype)
            x_model = build_physics_augmented_inputs_1d(
                x_state=x_in,
                edge_index=ei,
                edge_attr=ea,
                dt=dt,
                dx=dx,
                c_from_data=c,
                physics_cfg=physics_cfg,
                step_k=k,
            )
            x_model = append_position_channel(
                x_state=x_model,
                pos_nodes=pos_nodes,
                input_cfg=input_cfg,
                batch_size=None,
            )
            x_model = apply_state_minmax_minus1_1(
                x_model=x_model,
                input_cfg=input_cfg,
            )
            if godunov_cfg.enabled:
                if target_mode == "state":
                    raise ValueError(
                        "godunov_correction currently requires target_mode in "
                        "{delta, rate}; state-mode correction is not supported."
                    )
                god_state = _godunov_next_state(
                    x_now=x_in,
                    edge_index=ei,
                    edge_attr=ea,
                    dt=dt,
                    dx=dx,
                    c_from_data=c,
                    godunov_cfg=godunov_cfg,
                )
                god_target = _target_from_states(
                    x_now=x_in,
                    state_next=god_state,
                    dt=dt,
                    target_mode=target_mode,
                )
                pred_corr = model(x_model, ei, edge_attr=ea, dt=dt)
                pred_target = god_target + pred_corr
                y = _state_from_target(
                    x_now=x_in,
                    target=pred_target,
                    dt=dt,
                    target_mode=target_mode,
                )
            else:
                y = model.predict_state(
                    x_model,
                    ei,
                    edge_attr=ea,
                    dt=dt,
                    state_residual=state_residual_from_mode(
                        x_raw=x_in,
                        state_residual_mode=state_residual_mode,
                    ),
                )
            x_pred = y.detach().cpu()

            # Exact one-step GT state: prefer dataset rollout trajectory prefix.
            if prefix is not None and (k + 1) < int(prefix.shape[0]):
                x_true_next_np = prefix[k + 1]
            else:
                x_true_np = x_true.squeeze(-1).cpu().numpy()
                x_true_next_np = exact_advect_spectral_shift(
                    x_true_np, c=c, dt=dt, L=L
                )
            x_true = torch.from_numpy(x_true_next_np[:, None]).to(dtype=torch.float32)
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
    pos: np.ndarray,
    dt: float,
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
        f"Rollout Error Curves | samples={len(sample_ids)} | horizon={horizon}",
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
        else:
            ax20.text(
                0.5,
                0.5,
                "No finite RMSE values at final step",
                ha="center",
                va="center",
                transform=ax20.transAxes,
            )
        if final_rel_finite.size > 0:
            ax21.hist(
                final_rel_finite,
                bins=min(20, max(5, len(final_rel_finite))),
                color="tab:orange",
                alpha=0.75,
            )
        else:
            ax21.text(
                0.5,
                0.5,
                "No finite rel-L2 values at final step",
                ha="center",
                va="center",
                transform=ax21.transAxes,
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

    # 3) Space-time true/pred/error for one representative sample.
    if (
        sample_traj_id is not None
        and sample_pred_traj is not None
        and sample_true_traj is not None
    ):
        sample_pred_plot = _replace_nonfinite_with_nan(sample_pred_traj)
        sample_true_plot = _replace_nonfinite_with_nan(sample_true_traj)
        abs_err = np.abs(sample_pred_plot - sample_true_plot)
        t_axis = np.arange(sample_pred_traj.shape[0], dtype=np.float64) * float(dt)
        extent = [float(pos.min()), float(pos.max()), float(t_axis.max()), float(t_axis.min())]

        # Use GT-based scaling across panels for consistent visual comparison.
        gt_finite = sample_true_plot[np.isfinite(sample_true_plot)]
        if gt_finite.size == 0:
            gt_min, gt_max = -1.0, 1.0
        else:
            gt_min = float(np.min(gt_finite))
            gt_max = float(np.max(gt_finite))
            if not np.isfinite(gt_min) or not np.isfinite(gt_max):
                gt_min, gt_max = -1.0, 1.0
        if gt_max <= gt_min:
            eps = max(1e-6, abs(gt_min) * 1e-6)
            gt_min -= eps
            gt_max += eps
        gt_abs = max(abs(gt_min), abs(gt_max), 1e-6)

        fig3, axes3 = plt.subplots(1, 3, figsize=(14.5, 4.6), constrained_layout=True)
        im0 = axes3[0].imshow(
            sample_true_plot,
            aspect="auto",
            extent=extent,
            cmap="viridis",
            vmin=gt_min,
            vmax=gt_max,
        )
        im1 = axes3[1].imshow(
            sample_pred_plot,
            aspect="auto",
            extent=extent,
            cmap="viridis",
            vmin=gt_min,
            vmax=gt_max,
        )
        im2 = axes3[2].imshow(
            abs_err,
            aspect="auto",
            extent=extent,
            cmap="magma",
            vmin=0.0,
            vmax=gt_abs,
        )

        axes3[0].set_title("Exact State")
        axes3[1].set_title("Predicted State")
        axes3[2].set_title("|Error| (GT-Scaled)")
        for ax in axes3:
            ax.set_xlabel("Spatial coordinate x_i")
            ax.set_ylabel("time")
        fig3.colorbar(im0, ax=axes3[0], fraction=0.046, pad=0.04)
        fig3.colorbar(im1, ax=axes3[1], fraction=0.046, pad=0.04)
        fig3.colorbar(im2, ax=axes3[2], fraction=0.046, pad=0.04)
        fig3.suptitle(
            f"Representative Rollout Sample | sample_id={sample_traj_id}",
            fontsize=11,
        )
        fig3.savefig(plots_dir / "rollout_sample_spacetime.png", dpi=170)
        plt.close(fig3)


def main() -> None:
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    data_cfg = dict(cfg.get("data", {}))
    rollout_cfg = dict(cfg.get("rollout_eval", {}))
    target_mode = str(data_cfg.get("target_mode", "state")).strip().lower()
    godunov_cfg = _resolve_godunov_correction_cfg(cfg)
    if godunov_cfg.enabled and (target_mode == "state"):
        raise ValueError(
            "godunov_correction is enabled but data.target_mode='state'. "
            "Use target_mode='delta' or 'rate' for correction learning."
        )

    # Keep model output contract tied to training target_mode.
    cfg_model = dict(cfg.get("model", {}))
    if args.state_residual_mode is not None:
        cfg_model["state_residual_mode"] = str(args.state_residual_mode)
    cfg_model["predict_type"] = predict_type_from_target_mode(target_mode)
    cfg["model"] = cfg_model
    state_residual_mode = resolve_state_residual_mode(cfg)
    cfg_model["state_residual_mode"] = state_residual_mode
    cfg["model"] = cfg_model

    dataset_path = args.dataset or Path(data_cfg.get("dataset_path", "data/advection_train.npz"))
    horizon = int(
        args.horizon
        if args.horizon is not None
        else rollout_cfg.get("horizon", 100)
    )
    num_samples = int(
        args.num_samples
        if args.num_samples is not None
        else rollout_cfg.get("num_test_samples", 16)
    )

    dset = load_advection_dataset(dataset_path, dtype=torch.float32)
    metadata = dset.metadata
    c = float(metadata["c"])
    dt = float(metadata["dt"])
    dx = float(metadata["dx"])
    L = float(metadata["L"])
    pos = dset.pos.squeeze(-1).detach().cpu().numpy()
    physics_cfg = resolve_physics_input_cfg(cfg)
    input_cfg = resolve_input_feature_cfg(cfg)
    print(
        "[INFO] Physics-input channels: "
        f"enabled={physics_cfg.enabled}, "
        f"backend={physics_cfg.backend}, "
        f"include_adv={physics_cfg.include_adv}, "
        f"include_diff={physics_cfg.include_diff}",
        flush=True,
    )
    print(
        "[INFO] Godunov correction: "
        f"enabled={godunov_cfg.enabled}, "
        f"advection_scheme={godunov_cfg.advection_scheme}, "
        f"c_override={godunov_cfg.c_override}",
        flush=True,
    )
    print(
        "[INFO] State residual mode: "
        f"{state_residual_mode}",
        flush=True,
    )
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
    if (
        physics_cfg.enabled
        and physics_cfg.include_adv
        and physics_cfg.normalize_adv_to_minus1_1
    ):
        if physics_cfg.adv_min is None or physics_cfg.adv_max is None:
            raise ValueError(
                "physics_inputs.normalize_adv_to_minus1_1 is enabled, but "
                "adv_min/adv_max are missing from checkpoint config. "
                "Retrain with advection normalization enabled to persist these stats."
            )
        print(
            "[INFO] Advection normalization enabled: "
            f"range=[-1,1], adv_min={float(physics_cfg.adv_min):.6e}, "
            f"adv_max={float(physics_cfg.adv_max):.6e}, "
            f"clip={bool(physics_cfg.normalize_adv_clip)}",
            flush=True,
        )

    # Prefer split from checkpoint; fall back to first K samples.
    split = ckpt.get("split_indices", {})
    test_idx = split.get("test", list(range(len(dset))))
    sample_ids = list(test_idx[: min(num_samples, len(test_idx))])

    device_pref = str(args.device if args.device is not None else cfg.get("device", "auto"))
    device = resolve_device(device_pref)
    in_channels = 1 + count_physics_input_channels(physics_cfg) + count_input_feature_channels(input_cfg)
    model = build_model_from_config(
        cfg,
        in_channels=in_channels,
        edge_attr_channels=int(dset.edge_attr.shape[-1]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    edge_index = dset.edge_index.detach().cpu().long()
    edge_attr = dset.edge_attr.detach().cpu()
    pos_nodes = dset.pos.detach().cpu()

    rmse_all: List[np.ndarray] = []
    rel_l2_all: List[np.ndarray] = []
    sample_traj_id: int | None = None
    sample_pred_traj: np.ndarray | None = None
    sample_true_traj: np.ndarray | None = None
    pair_trajectory_id = dset.pair_trajectory_id
    pair_time_index = dset.pair_time_index
    rollout_states = dset.rollout_states
    gt_rollout_prefix_samples = 0
    gt_spectral_fallback_steps = 0
    for sid in sample_ids:
        x0 = dset.x[int(sid)].detach().cpu()
        true_traj_prefix = None
        if (
            rollout_states is not None
            and pair_trajectory_id is not None
            and pair_time_index is not None
            and 0 <= int(sid) < int(pair_trajectory_id.shape[0])
        ):
            traj_id = int(pair_trajectory_id[int(sid)].item())
            t0 = int(pair_time_index[int(sid)].item())
            if 0 <= traj_id < int(rollout_states.shape[0]) and 0 <= t0 < int(
                rollout_states.shape[1]
            ):
                true_traj_prefix = (
                    rollout_states[traj_id, t0:, :, 0].detach().cpu().numpy().astype(np.float64)
                )
                gt_rollout_prefix_samples += 1
                gt_spectral_fallback_steps += max(
                    0, int(horizon) - int(true_traj_prefix.shape[0] - 1)
                )
        curves = _rollout_one(
            model=model,
            x0=x0,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos_nodes=pos_nodes,
            c=c,
            dt=dt,
            dx=dx,
            L=L,
            horizon=horizon,
            device=device,
            physics_cfg=physics_cfg,
            input_cfg=input_cfg,
            target_mode=target_mode,
            state_residual_mode=state_residual_mode,
            godunov_cfg=godunov_cfg,
            true_traj_prefix=true_traj_prefix,
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
    )

    summary = {
        "checkpoint": str(args.checkpoint),
        "dataset": str(dataset_path),
        "device_requested": device_pref,
        "device_resolved": str(device),
        "num_samples": int(len(sample_ids)),
        "horizon": int(horizon),
        "model_in_channels": int(in_channels),
        "state_residual_mode": str(state_residual_mode),
        "gt_source": {
            "rollout_states_available": bool(
                rollout_states is not None
                and pair_trajectory_id is not None
                and pair_time_index is not None
            ),
            "samples_using_rollout_prefix": int(gt_rollout_prefix_samples),
            "spectral_fallback_steps_total": int(gt_spectral_fallback_steps),
        },
        "physics_inputs": {
            "enabled": bool(physics_cfg.enabled),
            "include_adv": bool(physics_cfg.include_adv),
            "include_diff": bool(physics_cfg.include_diff),
            "input_form": str(physics_cfg.input_form),
            "backend": str(physics_cfg.backend),
            "advection_scheme": str(physics_cfg.advection_scheme),
            "mls_ridge_eps": float(physics_cfg.mls_ridge_eps),
            "nu": float(physics_cfg.nu),
            "c_override": (
                None if physics_cfg.c_override is None else float(physics_cfg.c_override)
            ),
            "adv_all_steps": bool(physics_cfg.adv_all_steps),
            "input_weighted": bool(physics_cfg.input_weighted),
            "adv_weight": float(physics_cfg.adv_weight),
            "diff_weight": float(physics_cfg.diff_weight),
            "detach_inputs": bool(physics_cfg.detach_inputs),
            "normalize_adv_to_minus1_1": bool(
                physics_cfg.normalize_adv_to_minus1_1
            ),
            "adv_min": (
                None if physics_cfg.adv_min is None else float(physics_cfg.adv_min)
            ),
            "adv_max": (
                None if physics_cfg.adv_max is None else float(physics_cfg.adv_max)
            ),
            "normalize_adv_clip": bool(physics_cfg.normalize_adv_clip),
        },
        "godunov_correction": {
            "enabled": bool(godunov_cfg.enabled),
            "advection_scheme": str(godunov_cfg.advection_scheme),
            "c_override": (
                None if godunov_cfg.c_override is None else float(godunov_cfg.c_override)
            ),
        },
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
            pos=pos,
            dt=dt,
            sample_traj_id=sample_traj_id,
            sample_pred_traj=sample_pred_traj,
            sample_true_traj=sample_true_traj,
        )
        summary["plots_dir"] = str(plots_dir)

    save_json(summary, args.checkpoint.parent / "rollout_eval_summary.json")

    print("Rollout evaluation complete.")
    print(f"  samples: {len(sample_ids)}")
    print(f"  horizon: {horizon}")
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
