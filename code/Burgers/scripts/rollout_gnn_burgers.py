#!/usr/bin/env python3
"""Autoregressive rollout evaluation for trained 1D Burgers GNN."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from burgers_data import filter_burgers_dataset_by_regime, load_burgers_dataset
from gnn_burgers import build_model_from_config, predict_type_from_target_mode, resolve_device, save_json
from input_features import (
    apply_u_minmax_minus1_1,
    count_input_feature_channels,
    resolve_input_feature_cfg,
    select_feature_channels,
    u_column_index,
)
from physics_inputs import (
    PhysicsInputConfig,
    build_physics_augmented_inputs_1d,
    count_physics_input_channels,
    resolve_physics_input_cfg,
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
        "--data-dir",
        type=Path,
        default=None,
        help="Optional dataset directory override (defaults to checkpoint config).",
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
        "--allow-full-test-fallback",
        action="store_true",
        help=(
            "Allow fallback to all valid test start indices when checkpoint "
            "split_indices.test_effective is missing."
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
        help="Directory for rollout visualizations.",
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
    return parser.parse_args()


def _rollout_one(
    *,
    model: torch.nn.Module,
    x_base_full: torch.Tensor,
    rollout_states: torch.Tensor,
    traj_local_id: int,
    t0: int,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    dt: float,
    dx: float,
    horizon: int,
    device: torch.device,
    input_cfg,
    physics_cfg: PhysicsInputConfig,
) -> Dict[str, np.ndarray]:
    model.eval()

    n_nodes = int(x_base_full.shape[0])
    window = int(rollout_states.shape[1] - 1)
    max_h = int(min(horizon, max(0, window - t0)))

    x_pred = rollout_states[traj_local_id, t0].clone().detach()

    rmse = np.full((horizon,), np.nan, dtype=np.float64)
    rel_l2 = np.full((horizon,), np.nan, dtype=np.float64)

    pred_traj = np.full((max_h + 1, n_nodes), np.nan, dtype=np.float64)
    true_traj = np.full((max_h + 1, n_nodes), np.nan, dtype=np.float64)

    pred_traj[0] = x_pred.squeeze(-1).cpu().numpy().astype(np.float64)
    true_traj[0] = rollout_states[traj_local_id, t0].squeeze(-1).cpu().numpy().astype(np.float64)

    with torch.no_grad():
        for k in range(max_h):
            true_next = rollout_states[traj_local_id, t0 + k + 1].clone().detach()

            x_full_step = x_base_full.clone()
            x_full_step[:, 3:4] = x_pred

            x_model = select_feature_channels(x_full=x_full_step, input_cfg=input_cfg)
            x_in = x_model.to(device=device, dtype=torch.float32)
            ei = edge_index.to(device=device)
            ea = edge_attr.to(device=device, dtype=x_in.dtype)
            x_in = build_physics_augmented_inputs_1d(
                x_base=x_in,
                x_state=x_pred.to(device=device, dtype=x_in.dtype),
                edge_index=ei,
                edge_attr=ea,
                dt=dt,
                dx=dx,
                physics_cfg=physics_cfg,
                step_k=k,
            )
            x_in = apply_u_minmax_minus1_1(x_model=x_in, input_cfg=input_cfg)

            pred_next = model.predict_state(
                x_in,
                ei,
                edge_attr=ea,
                dt=dt,
                state_override=x_pred.to(device=device, dtype=x_in.dtype),
            ).detach().cpu()

            pred_traj[k + 1] = pred_next.squeeze(-1).numpy().astype(np.float64)
            true_traj[k + 1] = true_next.squeeze(-1).numpy().astype(np.float64)

            err = pred_next - true_next
            rmse[k] = math.sqrt(float(torch.mean(err * err).item()))
            rel_l2[k] = float(
                torch.norm(err).item() / max(torch.norm(true_next).item(), 1e-12)
            )

            x_pred = pred_next

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
    sample_pair_id: int | None,
    sample_pred_traj: np.ndarray | None,
    sample_true_traj: np.ndarray | None,
    x_nodes: np.ndarray,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir.mkdir(parents=True, exist_ok=True)

    steps = np.arange(1, horizon + 1, dtype=np.int64)
    times = steps.astype(np.float64) * float(dt)

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
        f"Burgers Rollout Error Curves | pairs={len(sample_ids)} | horizon={horizon}",
        fontsize=11,
    )
    fig.savefig(plots_dir / "rollout_error_curves.png", dpi=160)
    plt.close(fig)

    if (
        sample_pair_id is not None
        and sample_pred_traj is not None
        and sample_true_traj is not None
        and sample_pred_traj.shape[0] >= 2
    ):
        n_frames = int(sample_pred_traj.shape[0])
        frame_ids = sorted(set([0, max(0, n_frames // 2), n_frames - 1]))

        fig2, axes2 = plt.subplots(
            len(frame_ids),
            1,
            figsize=(9.5, 2.7 * len(frame_ids)),
            constrained_layout=True,
            sharex=True,
        )
        if not isinstance(axes2, np.ndarray):
            axes2 = np.array([axes2])

        for i, fi in enumerate(frame_ids):
            ax = axes2[i]
            u_true = sample_true_traj[fi]
            u_pred = sample_pred_traj[fi]
            ax.plot(x_nodes, u_true, lw=2.0, color="tab:blue", label="true")
            ax.plot(x_nodes, u_pred, lw=1.8, color="tab:orange", ls="--", label="pred")
            ax.plot(
                x_nodes,
                np.abs(u_pred - u_true),
                lw=1.2,
                color="tab:red",
                alpha=0.8,
                label="abs error",
            )
            ax.set_ylabel("u")
            ax.grid(alpha=0.25)
            ax.set_title(f"step={fi}")
            if i == 0:
                ax.legend(loc="best")

        axes2[-1].set_xlabel("x")
        fig2.suptitle(
            f"Representative Burgers Rollout | pair_index={sample_pair_id}",
            fontsize=11,
        )
        fig2.savefig(plots_dir / "rollout_sample_profiles.png", dpi=170)
        plt.close(fig2)


def _default_test_effective_indices(dset, horizon: int) -> List[int]:
    window = int(dset.rollout_states.shape[1] - 1)
    max_start_t = int(window - horizon)
    if max_start_t < 0:
        max_start_t = 0

    t = dset.pair_time_index.detach().cpu().numpy()
    keep = np.nonzero(t <= max_start_t)[0]
    return keep.astype(np.int64).tolist()


def _safe_index_list(raw: object) -> List[int] | None:
    if raw is None:
        return None
    try:
        return [int(i) for i in list(raw)]
    except Exception:
        return None


def main() -> None:
    args = parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]

    data_cfg = dict(cfg.get("data", {}))
    rollout_cfg = dict(cfg.get("rollout_eval", {}))
    target_mode = str(data_cfg.get("target_mode", "delta")).strip().lower()
    regime_filter = str(data_cfg.get("regime_filter", "all")).strip().lower()
    source_format = str(data_cfg.get("source_format", "auto")).strip().lower()

    cfg_model = dict(cfg.get("model", {}))
    cfg_model["predict_type"] = predict_type_from_target_mode(target_mode)
    cfg["model"] = cfg_model

    data_dir = Path(data_cfg.get("data_dir", "data/burgers_1d_dataset_v1/burgers_data"))
    if "wenoa_pt_path" in data_cfg:
        data_dir = Path(data_cfg["wenoa_pt_path"])
    if args.data_dir is not None:
        data_dir = args.data_dir

    source_options: Dict[str, object] = {}
    if "wenoa_split_fractions" in data_cfg:
        source_options["split_fractions"] = data_cfg["wenoa_split_fractions"]
    if "wenoa_split_seed" in data_cfg:
        source_options["split_seed"] = int(data_cfg["wenoa_split_seed"])
    if "wenoa_nu" in data_cfg:
        source_options["nu_value"] = float(data_cfg["wenoa_nu"])
    if "wenoa_split_paths" in data_cfg:
        source_options["split_paths"] = data_cfg["wenoa_split_paths"]

    horizon = int(
        args.horizon if args.horizon is not None else rollout_cfg.get("horizon", 100)
    )
    num_samples = int(
        args.num_samples
        if args.num_samples is not None
        else rollout_cfg.get("num_test_samples", 16)
    )
    if horizon < 1:
        raise ValueError(f"--horizon must be >= 1, got {horizon}.")
    if num_samples <= 0:
        raise ValueError(f"--num-samples must be positive, got {num_samples}.")

    dset = load_burgers_dataset(
        data_dir,
        split="test",
        include_rollout_states=True,
        dtype=torch.float32,
        source_format=source_format,
        source_options=source_options,
    )
    dset = filter_burgers_dataset_by_regime(dset, regime_filter)
    assert dset.rollout_states is not None
    source_format_resolved = str(dset.metadata.get("source_format", source_format)).strip().lower()

    print(
        "[INFO] Dataset source: "
        f"source_format={source_format_resolved} | path={data_dir}",
        flush=True,
    )
    print(
        "[INFO] Regime filter: "
        f"{regime_filter} | pairs={len(dset)} | "
        f"trajectories={int(dset.trajectory_ids.shape[0])}",
        flush=True,
    )

    dt = float(dset.metadata["dt"])
    dx = float(dset.metadata["dx"])
    input_cfg = resolve_input_feature_cfg(cfg, source_format=source_format_resolved)
    physics_cfg = resolve_physics_input_cfg(cfg)
    if cfg_model.get("state_channel", None) is None:
        u_col = u_column_index(input_cfg)
        if u_col is None:
            raise ValueError(
                "input_features.feature_columns must include 'u' for rollout evaluation."
            )
        cfg_model["state_channel"] = int(u_col)
        cfg["model"] = cfg_model
    if input_cfg.normalize_u_to_minus1_1:
        if input_cfg.u_min is None or input_cfg.u_max is None:
            raise ValueError(
                "input_features.normalize_u_to_minus1_1 is enabled, but "
                "u_min/u_max are missing from checkpoint config."
            )
        print(
            "[INFO] u normalization enabled: "
            f"range=[-1,1], u_min={float(input_cfg.u_min):.6e}, "
            f"u_max={float(input_cfg.u_max):.6e}, "
            f"clip={bool(input_cfg.normalize_clip)}",
            flush=True,
        )
    else:
        print("[INFO] u normalization disabled.", flush=True)
    print(
        "[INFO] Physics inputs: "
        f"enabled={physics_cfg.enabled}, "
        f"include_adv={physics_cfg.include_adv}, "
        f"input_form={physics_cfg.input_form}, "
        f"advection_scheme={physics_cfg.advection_scheme}, "
        f"normalize_adv_to_minus1_1={physics_cfg.normalize_adv_to_minus1_1}, "
        f"normalize_adv_percentile_low={physics_cfg.normalize_adv_percentile_low}, "
        f"normalize_adv_percentile_high={physics_cfg.normalize_adv_percentile_high}",
        flush=True,
    )

    split = ckpt.get("split_indices", {})
    test_effective_raw = (
        split.get("test_effective", None) if isinstance(split, dict) else None
    )

    test_effective = _safe_index_list(test_effective_raw)
    selection_mode = "checkpoint_test_effective"
    if test_effective is None or len(test_effective) == 0:
        if not bool(args.allow_full_test_fallback):
            raise ValueError(
                "checkpoint split_indices.test_effective is missing/empty. "
                "Rerun with --allow-full-test-fallback to use all valid test starts."
            )
        selection_mode = "full_valid_test_fallback"
        test_effective = _default_test_effective_indices(dset, horizon=horizon)

    if len(test_effective) == 0:
        raise ValueError("No rollout starts available for requested horizon.")

    bad_ids = [i for i in test_effective if (i < 0 or i >= len(dset))]
    if bad_ids:
        raise ValueError(
            "Invalid pair indices in selected test set. "
            f"Dataset size={len(dset)}, bad ids (first 10)={bad_ids[:10]}"
        )

    sample_ids = list(test_effective[: min(num_samples, len(test_effective))])

    device_pref = str(args.device if args.device is not None else cfg.get("device", "auto"))
    device = resolve_device(device_pref)

    in_channels = (
        count_input_feature_channels(input_cfg)
        + count_physics_input_channels(physics_cfg)
    )
    model = build_model_from_config(
        cfg,
        in_channels=in_channels,
        edge_attr_channels=int(dset.edge_attr.shape[-1]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    edge_index = dset.edge_index.detach().cpu().long()
    edge_attr = dset.edge_attr.detach().cpu()

    rmse_all: List[np.ndarray] = []
    rel_l2_all: List[np.ndarray] = []
    n_eval = int(len(sample_ids))
    n_nodes = int(dset.x_full.shape[1])
    pred_traj_by_sample = np.full((n_eval, horizon + 1, n_nodes), np.nan, dtype=np.float64)
    true_traj_by_sample = np.full((n_eval, horizon + 1, n_nodes), np.nan, dtype=np.float64)
    valid_steps_by_sample = np.zeros((n_eval,), dtype=np.int64)
    sample_time_index = np.zeros((n_eval,), dtype=np.int64)
    sample_local_traj_id = np.zeros((n_eval,), dtype=np.int64)
    sample_global_traj_id = np.zeros((n_eval,), dtype=np.int64)

    sample_pair_id: int | None = None
    sample_pred_traj: np.ndarray | None = None
    sample_true_traj: np.ndarray | None = None

    for si, pid in enumerate(sample_ids):
        local_tid = int(dset.pair_trajectory_id[pid].item())
        t0 = int(dset.pair_time_index[pid].item())
        global_tid = int(dset.trajectory_ids[local_tid].item())

        curves = _rollout_one(
            model=model,
            x_base_full=dset.x_full[pid].detach().cpu(),
            rollout_states=dset.rollout_states.detach().cpu(),
            traj_local_id=local_tid,
            t0=t0,
            edge_index=edge_index,
            edge_attr=edge_attr,
            dt=dt,
            dx=dx,
            horizon=horizon,
            device=device,
            input_cfg=input_cfg,
            physics_cfg=physics_cfg,
        )

        rmse_all.append(curves["rmse"])
        rel_l2_all.append(curves["rel_l2"])
        n_states = int(curves["pred_traj"].shape[0])
        valid_steps = max(0, n_states - 1)
        pred_traj_by_sample[si, :n_states, :] = curves["pred_traj"]
        true_traj_by_sample[si, :n_states, :] = curves["true_traj"]
        valid_steps_by_sample[si] = int(valid_steps)
        sample_time_index[si] = int(t0)
        sample_local_traj_id[si] = int(local_tid)
        sample_global_traj_id[si] = int(global_tid)

        if sample_pair_id is None:
            sample_pair_id = int(pid)
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
        x_nodes=dset.pos.detach().cpu().numpy().reshape(-1),
        sample_time_index=sample_time_index,
        sample_local_traj_id=sample_local_traj_id,
        sample_global_traj_id=sample_global_traj_id,
        sample_valid_steps=valid_steps_by_sample,
        sample_pred_traj_by_sample=pred_traj_by_sample,
        sample_true_traj_by_sample=true_traj_by_sample,
        sample_pair_id=np.int64(-1 if sample_pair_id is None else sample_pair_id),
        sample_pred_traj=np.array([]) if sample_pred_traj is None else sample_pred_traj,
        sample_true_traj=np.array([]) if sample_true_traj is None else sample_true_traj,
    )

    summary = {
        "checkpoint": str(args.checkpoint),
        "data_dir": str(data_dir),
        "regime_filter": regime_filter,
        "sample_selection_mode": selection_mode,
        "allow_full_test_fallback": bool(args.allow_full_test_fallback),
        "device_requested": device_pref,
        "device_resolved": str(device),
        "num_samples": int(len(sample_ids)),
        "horizon": int(horizon),
        "model_in_channels": int(in_channels),
        "input_features": {
            "feature_columns": list(input_cfg.feature_columns),
            "use_pos": bool(input_cfg.use_pos),
            "normalize_u_to_minus1_1": bool(input_cfg.normalize_u_to_minus1_1),
            "normalize_clip": bool(input_cfg.normalize_clip),
            "u_min": None if input_cfg.u_min is None else float(input_cfg.u_min),
            "u_max": None if input_cfg.u_max is None else float(input_cfg.u_max),
        },
        "physics_inputs": {
            "enabled": bool(physics_cfg.enabled),
            "include_adv": bool(physics_cfg.include_adv),
            "input_form": str(physics_cfg.input_form),
            "advection_scheme": str(physics_cfg.advection_scheme),
            "adv_all_steps": bool(physics_cfg.adv_all_steps),
            "input_weighted": bool(physics_cfg.input_weighted),
            "adv_weight": float(physics_cfg.adv_weight),
            "normalize_adv_to_minus1_1": bool(physics_cfg.normalize_adv_to_minus1_1),
            "normalize_adv_percentile_low": float(physics_cfg.normalize_adv_percentile_low),
            "normalize_adv_percentile_high": float(physics_cfg.normalize_adv_percentile_high),
            "normalize_adv_clip": bool(physics_cfg.normalize_adv_clip),
            "adv_min": None if physics_cfg.adv_min is None else float(physics_cfg.adv_min),
            "adv_max": None if physics_cfg.adv_max is None else float(physics_cfg.adv_max),
            "detach_inputs": bool(physics_cfg.detach_inputs),
        },
        "final_mean_rmse": (
            float(mean_rmse[-1])
            if (mean_rmse.size > 0 and np.isfinite(mean_rmse[-1]))
            else None
        ),
        "final_mean_rel_l2": (
            float(mean_rel_l2[-1])
            if (mean_rel_l2.size > 0 and np.isfinite(mean_rel_l2[-1]))
            else None
        ),
        "nonfinite_rmse_count": nonfinite_rmse_count,
        "nonfinite_rel_l2_count": nonfinite_rel_count,
        "out_npz": str(out_npz),
    }

    plots_dir = args.plots_dir
    if plots_dir is None:
        plots_dir = args.checkpoint.parent / "rollout_plots"

    if not args.no_plots:
        x_nodes = dset.pos.detach().cpu().numpy().reshape(-1)
        _write_rollout_plots(
            plots_dir=plots_dir,
            horizon=horizon,
            sample_ids=sample_ids,
            rmse_arr=rmse_arr,
            rel_l2_arr=rel_l2_arr,
            mean_rmse=mean_rmse,
            mean_rel_l2=mean_rel_l2,
            dt=dt,
            sample_pair_id=sample_pair_id,
            sample_pred_traj=sample_pred_traj,
            sample_true_traj=sample_true_traj,
            x_nodes=x_nodes,
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
