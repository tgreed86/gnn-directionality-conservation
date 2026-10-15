#!/usr/bin/env python
"""Extract main-results metrics from saved rollout evaluation artifacts.

This script writes the canonical CSV/JSON used by scripts/make_results_table.py.
It computes conservation error as a per-trajectory absolute final summed-field
drift and then reports the mean and sample standard deviation over trajectories:

    |sum_i u_hat_i(T) - sum_i u_i(0)|

Run from the paper workspace root:

    python scripts/make_results_metrics.py
"""

from __future__ import print_function

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


DEFAULT_BASE_DIR = Path("/Users/trevorreed/UVA_postdoc_stuff/gparc/toy_model")
DEFAULT_OUTPUT_JSON = Path("drafts/6-25-26/results_table_metrics.json")
DEFAULT_OUTPUT_CSV = Path("drafts/6-25-26/results_table_metrics.csv")
HIGH_GRADIENT_RELATIVE_THRESHOLD = 0.2


PROJECT_DIR_BY_DATASET = {
    "Inviscid Burgers": "Burgers",
    "Linear Advection": "1D_advection",
    "LWR Traffic": "1D_traffic",
    "Chemotactic Cell Migration": "chemotactic_cell_migration",
}

CHEMOTAXIS_SSIM_WINDOW_SIZE = 11
CHEMOTAXIS_SSIM_SIGMA_CELLS = 1.5
CHEMOTAXIS_SSIM_K1 = 0.01
CHEMOTAXIS_SSIM_K2 = 0.03


RUNS = [
    (
        "Inviscid Burgers",
        "FluxGraphNet + advection",
        "Burgers/runs/with_shocks/fluxgraphnet_w_advection/300_timesteps/"
        "300_timesteps_dt-1p5e-3_cfl-0p2_lr-1e-4_200ep_delta",
    ),
    (
        "Inviscid Burgers",
        "MeshGraphNet + advection",
        "Burgers/runs/with_shocks/meshgraphnet_w_advection/no-dropout/"
        "300_timesteps_dt-1p5e-3_cfl-0p2_lr-1e-4_200ep_delta",
    ),
    (
        "Inviscid Burgers",
        "FluxGraphNet (no advection)",
        "Burgers/runs/with_shocks/fluxgraphnet/300_timesteps/"
        "300_timesteps_dt-1p5e-3_cfl-0p2_lr-1e-4_200ep_delta",
    ),
    (
        "Inviscid Burgers",
        "MeshGraphNet (no advection)",
        "Burgers/runs/with_shocks/vanilla_meshgraphnet/no-dropout/"
        "cfl-0p2_dt-1p5e-3/lr-1e-4_200ep_delta",
    ),
    (
        "Inviscid Burgers",
        "SAGEConv + advection",
        "Burgers/runs/with_shocks/sageconv_w_advection/no-dropout/"
        "cfl-0p2_dt-1p5e-3/lr-1e-4_200ep_delta",
    ),
    (
        "Inviscid Burgers",
        "SAGEConv (no advection)",
        "Burgers/runs/with_shocks/vanilla_sageconv/no-dropout/"
        "cfl-0p2_dt-1p5e-3/"
        "300_timesteps_dt-1p5e-3_cfl-0p2_lr-1e-4_500ep_delta",
    ),
    (
        "Inviscid Burgers",
        "MeshGraphNet-abs (no advection)",
        "Burgers/runs/with_shocks/meshgraphnet-abs/300-timesteps_no_dropout/"
        "300_timesteps_dt-1p5e-3_cfl-0p2_lr-1e-4_500ep_delta",
    ),
    # Linear Advection runs
    (
        "Linear Advection",
        "FluxGraphNet + advection",
        "1D_advection/runs/cell_averaged/with_physics/fluxgraphnet/"
        "200ep_lr-sched_adv-norm_rel-l2-early-stop_tv-loss_10ts_seed-1234",
    ),
    (
        "Linear Advection",
        "MeshGraphNet + advection",
        "1D_advection/runs/cell_averaged/with_physics/meshgraphnet/"
        "200ep_lr-sched_adv-norm_rel-l2-early-stop_tv-loss_10ts_seed-1234",
    ),
    (
        "Linear Advection",
        "FluxGraphNet (no advection)",
        "1D_advection/runs/cell_averaged/no_physics_input/fluxgraphnet/"
        "400ep_lr-sched_rel-l2-early-stop_tv-loss_10ts_seed-1234",
    ),
    (
        "Linear Advection",
        "MeshGraphNet (no advection)",
        "1D_advection/runs/cell_averaged/no_physics_input/meshgraphnet/"
        "400ep_lr-sched_rel-l2-early-stop_tv-loss_10ts_seed-1234",
    ),
    (
        "Linear Advection",
        "SAGEConv + advection",
        "1D_advection/runs/cell_averaged/with_physics/sageconv/"
        "200ep_lr-sched_adv-norm_rel-l2-early-stop_tv-loss_10ts_seed-1234",
    ),
    (
        "Linear Advection",
        "SAGEConv (no advection)",
        "1D_advection/runs/cell_averaged/no_physics_input/sageconv/"
        "200ep_lr-sched_rel-l2-early-stop_tv-loss_10ts_seed-1234",
    ),
    (
        "Linear Advection",
        "MeshGraphNet-abs (no advection)",
        "1D_advection/runs/cell_averaged/no_physics_input/meshgraphnet-abs/"
        "400ep_lr-sched_rel-l2-early-stop_tv-loss_10ts_seed-1234",
    ),
    # LWR traffic runs
    (
        "LWR Traffic",
        "FluxGraphNet + advection",
        "1D_traffic/runs/structured_directional/fluxgraphnet/flux-loss-weight/with_advection/"
        "1000ep_lr-1e-4_10p0-flux-loss-weight_pred-rate",
    ),
    (
        "LWR Traffic",
        "MeshGraphNet + advection",
        "1D_traffic/runs/structured_directional/meshgraphnet/with_advection/"
        "1000ep_lr-1e-4_pred-rate",
    ),
    (
        "LWR Traffic",
        "FluxGraphNet (no advection)",
        "1D_traffic/runs/structured_directional/fluxgraphnet/flux-loss-weight/"
        "1000ep_lr-1e-4_pred-rate_flux-loss-10p0",
    ),
    (
        "LWR Traffic",
        "MeshGraphNet (no advection)",
        "1D_traffic/runs/structured_directional/meshgraphnet/"
        "1000ep_lr-1e-4_pred-rate",
    ),
    (
        "LWR Traffic",
        "SAGEConv + advection",
        "1D_traffic/runs/structured_directional/sageconv/with_advection/"
        "1000ep_lr-1e-4_pred-rate",
    ),
    (
        "LWR Traffic",
        "SAGEConv (no advection)",
        "1D_traffic/runs/structured_directional/sageconv/"
        "1000ep_lr-1e-4_pred-rate",
    ),
    # Chemotactic Cell Migration runs
        (
        "Chemotactic Cell Migration",
        "FluxGraphNet + advection",
        "chemotactic_cell_migration/runs/17x17-unique-meshes_constant-chi-0p1/fluxgraphnet/with_advection/"
        "unique_meshes_constant-chi_0p1_500ep_lr-5e-5_5-ts_abs-pos",
    ),
    (
        "Chemotactic Cell Migration",
        "MeshGraphNet + advection",
        "chemotactic_cell_migration/runs/17x17-unique-meshes_constant-chi-0p1/meshgraphnet/with_advection/"
        "17x17-unique-meshes_chi-0p1_500ep_lr-5e-5_5-ts_abs-pos",
    ),
    (
        "Chemotactic Cell Migration",
        "FluxGraphNet (no advection)",
        "chemotactic_cell_migration/runs/17x17-unique-meshes_constant-chi-0p1/fluxgraphnet/no_physics/"
        "17x17-unique-meshes_chi-0p1_500ep_lr-5e-5_5-ts_abs-pos",
    ),
    (
        "Chemotactic Cell Migration",
        "MeshGraphNet (no advection)",
        "chemotactic_cell_migration/runs/17x17-unique-meshes_constant-chi-0p1/meshgraphnet/no_physics/"
        "17x17-unique-meshes_chi-0p1_500ep_lr-5e-5_5-ts_abs-pos",
    ),
    (
        "Chemotactic Cell Migration",
        "SAGEConv + advection",
        "chemotactic_cell_migration/runs/17x17-unique-meshes_constant-chi-0p1/sageconv/with_advection/"
        "17x17-unique-meshes_chi-0p1_500ep_lr-5e-5_5-ts_abs-pos",
    ),
    (
        "Chemotactic Cell Migration",
        "SAGEConv (no advection)",
        "chemotactic_cell_migration/runs/17x17-unique-meshes_constant-chi-0p1/sageconv/no_physics/"
        "17x17-unique-meshes_chi-0p1_500ep_lr-5e-5_5-ts_abs-pos",
    ),
]


COLUMNS = [
    "dataset",
    "method",
    "run_dir",
    "eval_npz",
    "summary_json",
    "num_samples_metrics",
    "horizon_steps",
    "dt",
    "final_time",
    "final_rel_l2_mean",
    "final_rel_l2_std",
    "final_ssim_mean",
    "final_ssim_std",
    "final_high_gradient_rel_l2_mean",
    "final_high_gradient_rel_l2_std",
    "final_high_gradient_rel_l2_n",
    "high_gradient_relative_threshold",
    "final_conservation_error_mean",
    "final_conservation_error_std",
    "final_conservation_error_n",
    "final_conservation_error_source",
    "final_conservation_signed_drift_mean",
    "final_conservation_signed_drift_std",
    "max_conservation_error_over_time_mean",
    "max_conservation_error_over_time_std",
    "notes",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract final-time table metrics from saved NPZ/JSON artifacts."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help="Base directory prepended to the configured run paths.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help="Output JSON path.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Output CSV path.",
    )
    return parser.parse_args()


def finite_float(value):
    if value is None:
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    return x


def mean_std(values):
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None, None, 0
    mean = float(np.mean(finite))
    std = float(np.std(finite, ddof=1)) if finite.size > 1 else None
    return mean, std, int(finite.size)


def load_json(path):
    if not path.exists():
        return {}
    with path.open("r") as fh:
        return json.load(fh)


def is_placeholder_run_path(relative_run_path):
    text = str(relative_run_path or "").strip()
    return not text or text.upper().startswith("TODO")


def get_summary_dataset_path(summary):
    for key in ("data_dir", "dataset", "dataset_path", "data"):
        if key not in summary:
            continue
        value = summary[key]
        if isinstance(value, dict):
            for nested_key in ("data_dir", "dataset", "dataset_path", "path"):
                if nested_key in value:
                    return value[nested_key]
            continue
        return value
    return None


def resolve_dataset_path(base_dir, dataset_name, summary):
    raw_path = get_summary_dataset_path(summary)
    if raw_path is None:
        return None

    path = Path(str(raw_path))
    if path.is_absolute():
        return path

    project_dir = PROJECT_DIR_BY_DATASET.get(dataset_name)
    project_root = base_dir / project_dir if project_dir is not None else base_dir
    return project_root / path


def load_rollout_true_trajs(base_dir, dataset_name, summary, sample_ids, horizon):
    dataset_path = resolve_dataset_path(base_dir, dataset_name, summary)
    if dataset_path is None or not dataset_path.exists() or dataset_path.suffix != ".npz":
        return None, dataset_path

    data = np.load(dataset_path, allow_pickle=False)
    if "rollout_states" not in data.files:
        return None, dataset_path

    states = np.asarray(data["rollout_states"], dtype=np.float64)
    if states.ndim == 4:
        states = states[..., 0]
    if states.ndim != 3:
        return None, dataset_path

    ids = np.asarray(sample_ids, dtype=np.int64).reshape(-1)
    if ids.size == 0 or np.any(ids < 0) or np.any(ids >= states.shape[0]):
        return None, dataset_path

    max_horizon = min(int(horizon), states.shape[1] - 1)
    return states[ids, : max_horizon + 1, :], dataset_path


def conservation_from_trajs(pred, true):
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    if pred.ndim != 3 or true.ndim != 3:
        raise ValueError("Expected rank-3 trajectory arrays [sample, time, node].")
    if pred.shape[0] != true.shape[0] or pred.shape[2] != true.shape[2]:
        raise ValueError(
            "Prediction/reference trajectory shapes are incompatible: "
            "pred={0}, true={1}".format(pred.shape, true.shape)
        )

    horizon = min(pred.shape[1] - 1, true.shape[1] - 1)
    initial_true_sum = true[:, 0, :].sum(axis=1)
    signed_final = pred[:, horizon, :].sum(axis=1) - initial_true_sum
    abs_final = np.abs(signed_final)
    abs_over_time = np.abs(pred[:, : horizon + 1, :].sum(axis=2) - initial_true_sum[:, None])
    max_abs = np.nanmax(abs_over_time, axis=1)
    return signed_final, abs_final, max_abs


def prediction_reference_trajs(base_dir, dataset_name, summary, npz):
    if "sample_gnn_traj_by_sample" in npz.files and "sample_true_traj_by_sample" in npz.files:
        return (
            np.asarray(npz["sample_gnn_traj_by_sample"], dtype=np.float64),
            np.asarray(npz["sample_true_traj_by_sample"], dtype=np.float64),
            "saved_all_true_and_predicted_trajectories",
            "",
        )

    if "all_gnn_trajs" in npz.files and np.asarray(npz["all_gnn_trajs"]).size > 0:
        pred = np.asarray(npz["all_gnn_trajs"], dtype=np.float64)
        sample_id_key = "all_sample_ids" if "all_sample_ids" in npz.files else "sample_ids"
        sample_ids = np.asarray(npz[sample_id_key], dtype=np.int64).reshape(-1)
        horizon = int(np.asarray(npz["horizon"]).item()) if "horizon" in npz.files else pred.shape[1] - 1
        true, dataset_path = load_rollout_true_trajs(
            base_dir,
            dataset_name,
            summary,
            sample_ids,
            horizon,
        )
        if true is not None:
            note = "true initial states loaded from {0}".format(dataset_path)
            return (
                pred,
                true,
                "saved_all_predicted_trajectories_plus_dataset_initial_states",
                note,
            )

    if (
        "sample_gnn_traj" in npz.files
        and "sample_true_traj" in npz.files
        and np.asarray(npz["sample_gnn_traj"]).size > 0
    ):
        pred = np.asarray(npz["sample_gnn_traj"], dtype=np.float64)[None, :, :]
        true = np.asarray(npz["sample_true_traj"], dtype=np.float64)[None, :, :]
        sample_id = int(np.asarray(npz["sample_traj_id"]).item()) if "sample_traj_id" in npz.files else None
        note = "only one saved trajectory available"
        if sample_id is not None:
            note += " (sample_traj_id={0})".format(sample_id)
        return pred, true, "single_saved_trajectory", note

    return None, None, "unavailable", "no saved trajectories available"


def high_gradient_final_rel_l2(pred, true, relative_threshold=HIGH_GRADIENT_RELATIVE_THRESHOLD):
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    if pred.ndim != 3 or true.ndim != 3:
        return np.asarray([], dtype=np.float64)
    if pred.shape[0] != true.shape[0] or pred.shape[2] != true.shape[2]:
        return np.asarray([], dtype=np.float64)

    horizon = min(pred.shape[1] - 1, true.shape[1] - 1)
    pred_final = pred[:, horizon, :]
    true_final = true[:, horizon, :]
    values = []
    for pred_row, true_row in zip(pred_final, true_final):
        left_jump = np.abs(true_row - np.roll(true_row, 1))
        right_jump = np.abs(np.roll(true_row, -1) - true_row)
        local_jump = np.maximum(left_jump, right_jump)
        max_jump = float(np.nanmax(local_jump)) if local_jump.size else 0.0
        if not math.isfinite(max_jump) or max_jump <= 0.0:
            values.append(np.nan)
            continue
        mask = local_jump >= float(relative_threshold) * max_jump
        if not np.any(mask):
            values.append(np.nan)
            continue
        numerator = np.sum((pred_row[mask] - true_row[mask]) ** 2)
        denominator = np.sum(true_row[mask] ** 2) + 1.0e-12
        values.append(float(np.sqrt(numerator / denominator)))
    return np.asarray(values, dtype=np.float64)


def get_conservation(base_dir, dataset_name, summary, npz):
    pred, true, source, note = prediction_reference_trajs(base_dir, dataset_name, summary, npz)
    if pred is not None and true is not None:
        signed, abs_final, max_abs = conservation_from_trajs(pred, true)
        if source == "single_saved_trajectory":
            note = note.replace("only one saved trajectory available", "only one saved trajectory available for conservation error")
        return signed, abs_final, max_abs, source, note

    empty = np.asarray([], dtype=np.float64)
    return empty, empty, empty, "unavailable", "no saved trajectories available for conservation error"


def resolve_run_path(base_dir, relative_run_path):
    path = Path(str(relative_run_path))
    if path.is_absolute():
        return path
    return base_dir / path


def resolve_eval_artifact(base_dir, dataset, relative_run_path):
    path = resolve_run_path(base_dir, relative_run_path)
    if path.suffix == ".npz":
        eval_npz = path
        run_dir = path.parent
        summary_json = run_dir / "summary.json"
        if not summary_json.exists():
            summary_json = run_dir / "gnn_vs_godunov_summary.json"
        return run_dir, eval_npz, summary_json

    run_dir = path
    if not run_dir.exists():
        raise FileNotFoundError(str(run_dir))

    if dataset == "LWR Traffic":
        candidates = [
            run_dir / "lwr_rollout_metrics_all_tests.npz",
            run_dir / "rollout_plots" / "lwr_rollout_metrics_all_tests.npz",
            run_dir / "lwr_rollout_metrics.npz",
            run_dir / "rollout_plots" / "lwr_rollout_metrics.npz",
        ]
        for candidate in candidates:
            if candidate.exists():
                summary_json = run_dir / "rollout_plots" / "summary.json"
                if not summary_json.exists():
                    summary_json = run_dir / "summary.json"
                return run_dir, candidate, summary_json

    if dataset == "Chemotactic Cell Migration":
        candidates = [
            run_dir / "test_rollout_predictions.npz",
            run_dir / "rollout_plots" / "test_rollout_predictions.npz",
        ]
        for candidate in candidates:
            if candidate.exists():
                summary_json = candidate.parent / "summary_metrics.json"
                if not summary_json.exists():
                    summary_json = run_dir / "summary_metrics.json"
                return run_dir, candidate, summary_json

    eval_npz = run_dir / "gnn_vs_godunov_eval_local_ssim.npz"
    summary_json = run_dir / "gnn_vs_godunov_summary.json"
    return run_dir, eval_npz, summary_json


def scalar_from_npz(npz, key, default=None):
    if key not in npz.files:
        return default
    value = np.asarray(npz[key])
    if value.shape == ():
        return value.item()
    return default


def lwr_summary_value(npz, metric_name, stat_name, fallback=None):
    preferred = "{0}_split_topology_balanced_{1}".format(metric_name, stat_name)
    if preferred in npz.files:
        return finite_float(np.asarray(npz[preferred]).item())
    raw = "{0}_{1}".format(metric_name, stat_name)
    if raw in npz.files:
        return finite_float(np.asarray(npz[raw]).item())
    return fallback


def extract_lwr_metrics_row(run_dir, eval_npz, summary_json, dataset, method):
    if not eval_npz.exists():
        raise FileNotFoundError(str(eval_npz))

    npz = np.load(eval_npz, allow_pickle=False)
    rel_l2 = np.asarray(npz["rel_l2_gnn_by_sample"], dtype=np.float64)
    ssim = np.asarray(npz["ssim_gnn_by_sample"], dtype=np.float64)
    num_samples = int(rel_l2.shape[0])
    horizon = int(scalar_from_npz(npz, "horizon", rel_l2.shape[1] - 1))

    rel_mean_raw, rel_std_raw, rel_n = mean_std(rel_l2[:, -1])
    ssim_mean_raw, ssim_std_raw, ssim_n = mean_std(ssim[:, -1])
    rel_mean = lwr_summary_value(npz, "final_rel_l2", "mean", fallback=rel_mean_raw)
    rel_std = lwr_summary_value(npz, "final_rel_l2", "std", fallback=rel_std_raw)
    ssim_mean = lwr_summary_value(npz, "final_ssim", "mean", fallback=ssim_mean_raw)
    ssim_std = lwr_summary_value(npz, "final_ssim", "std", fallback=ssim_std_raw)

    if "final_high_gradient_rel_l2_by_sample" in npz.files:
        high_grad_values = np.asarray(
            npz["final_high_gradient_rel_l2_by_sample"], dtype=np.float64
        )
    else:
        high_grad_values = np.asarray([], dtype=np.float64)
    high_grad_mean_raw, high_grad_std_raw, high_grad_n = mean_std(high_grad_values)
    high_grad_mean = lwr_summary_value(
        npz,
        "final_high_gradient_rel_l2",
        "mean",
        fallback=high_grad_mean_raw,
    )
    high_grad_std = lwr_summary_value(
        npz,
        "final_high_gradient_rel_l2",
        "std",
        fallback=high_grad_std_raw,
    )

    if "final_conservation_error_by_sample" in npz.files:
        abs_drift = np.asarray(npz["final_conservation_error_by_sample"], dtype=np.float64)
    else:
        abs_drift = np.asarray([], dtype=np.float64)
    if "final_conservation_signed_drift_by_sample" in npz.files:
        signed_drift = np.asarray(
            npz["final_conservation_signed_drift_by_sample"], dtype=np.float64
        )
    else:
        signed_drift = np.asarray([], dtype=np.float64)
    if "max_conservation_error_over_time_by_sample" in npz.files:
        max_abs_drift = np.asarray(
            npz["max_conservation_error_over_time_by_sample"], dtype=np.float64
        )
    else:
        max_abs_drift = np.asarray([], dtype=np.float64)

    cons_mean_raw, cons_std_raw, cons_n = mean_std(abs_drift)
    signed_mean, signed_std, _ = mean_std(signed_drift)
    max_cons_mean_raw, max_cons_std_raw, _ = mean_std(max_abs_drift)
    cons_mean = lwr_summary_value(
        npz,
        "final_conservation_error",
        "mean",
        fallback=cons_mean_raw,
    )
    cons_std = lwr_summary_value(
        npz,
        "final_conservation_error",
        "std",
        fallback=cons_std_raw,
    )
    max_cons_mean = lwr_summary_value(
        npz,
        "max_conservation_error_over_time",
        "mean",
        fallback=max_cons_mean_raw,
    )
    max_cons_std = lwr_summary_value(
        npz,
        "max_conservation_error_over_time",
        "std",
        fallback=max_cons_std_raw,
    )

    high_gradient_threshold = finite_float(
        scalar_from_npz(npz, "high_gradient_relative_threshold", HIGH_GRADIENT_RELATIVE_THRESHOLD)
    )
    dt = None
    final_time = None
    if "dt_by_sample" in npz.files:
        dt_values = np.asarray(npz["dt_by_sample"], dtype=np.float64)
        finite_dt = dt_values[np.isfinite(dt_values)]
        if finite_dt.size and np.allclose(finite_dt, finite_dt[0]):
            dt = float(finite_dt[0])
            final_time = dt * float(horizon)

    notes = [
        "LWR metrics read from precomputed rollout metrics NPZ",
        "means/stds use split-topology-balanced values when available",
    ]
    if "selected_splits" in npz.files:
        splits = [str(item) for item in np.asarray(npz["selected_splits"]).tolist()]
        notes.append("splits={0}".format(",".join(splits)))
    if dt is None:
        notes.append("dt varies across traffic topologies")

    return {
        "dataset": dataset,
        "method": method,
        "run_dir": str(run_dir),
        "eval_npz": str(eval_npz),
        "summary_json": str(summary_json),
        "num_samples_metrics": num_samples,
        "horizon_steps": horizon,
        "dt": dt,
        "final_time": final_time,
        "final_rel_l2_mean": rel_mean,
        "final_rel_l2_std": rel_std,
        "final_ssim_mean": ssim_mean,
        "final_ssim_std": ssim_std,
        "final_high_gradient_rel_l2_mean": high_grad_mean,
        "final_high_gradient_rel_l2_std": high_grad_std,
        "final_high_gradient_rel_l2_n": high_grad_n,
        "high_gradient_relative_threshold": high_gradient_threshold,
        "final_conservation_error_mean": cons_mean,
        "final_conservation_error_std": cons_std,
        "final_conservation_error_n": cons_n,
        "final_conservation_error_source": "lwr_rollout_metrics_npz",
        "final_conservation_signed_drift_mean": signed_mean,
        "final_conservation_signed_drift_std": signed_std,
        "max_conservation_error_over_time_mean": max_cons_mean,
        "max_conservation_error_over_time_std": max_cons_std,
        "notes": "; ".join(notes),
    }


def chemotaxis_density_array(values, name, path):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 4 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 3:
        raise ValueError(
            "{0} in {1} must have shape [sample, time, node, 1] or "
            "[sample, time, node].".format(name, path)
        )
    return array


def chemotaxis_trajectory_path(dataset_path, trajectory_id):
    if dataset_path.is_dir():
        candidates = [
            dataset_path / "trajectory_{0:05d}.npz".format(trajectory_id),
            dataset_path / "trajectory_{0}.npz".format(trajectory_id),
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            "No source archive found for chemotaxis trajectory {0} in {1}.".format(
                trajectory_id,
                dataset_path,
            )
        )
    raise ValueError(
        "Chemotaxis dataset path must be a directory of trajectory NPZ files: {0}".format(
            dataset_path
        )
    )


def load_chemotaxis_geometry(dataset_path, trajectory_ids, num_nodes):
    geometries = []
    for trajectory_id in trajectory_ids:
        source = chemotaxis_trajectory_path(dataset_path, int(trajectory_id))
        with np.load(source, allow_pickle=False) as archive:
            required = ("cell_centers", "cell_areas", "undirected_edge_index")
            missing = [name for name in required if name not in archive.files]
            if missing:
                raise KeyError(
                    "{0} is missing geometry array(s): {1}".format(
                        source,
                        ", ".join(missing),
                    )
                )
            centers = np.asarray(archive["cell_centers"], dtype=np.float64).reshape(
                -1, 2
            )
            areas = np.asarray(archive["cell_areas"], dtype=np.float64).reshape(-1)
            edges = np.asarray(
                archive["undirected_edge_index"], dtype=np.int64
            ).reshape(2, -1)

        if centers.shape != (num_nodes, 2) or areas.shape != (num_nodes,):
            raise ValueError(
                "Chemotaxis geometry in {0} does not match {1} rollout nodes.".format(
                    source,
                    num_nodes,
                )
            )
        if not np.all(np.isfinite(centers)):
            raise ValueError("Chemotaxis cell centers must be finite in {0}.".format(source))
        if not np.all(np.isfinite(areas)) or np.any(areas <= 0.0):
            raise ValueError(
                "Chemotaxis cell areas must be positive and finite in {0}.".format(
                    source
                )
            )
        if edges.size and (np.any(edges < 0) or np.any(edges >= num_nodes)):
            raise ValueError("Chemotaxis edge indices are out of bounds in {0}.".format(source))
        geometries.append((centers, areas, edges))
    return geometries


def chemotaxis_local_ssim_final(
    truth,
    prediction,
    centers,
    areas,
    reference_spacing,
    data_range,
):
    radius = (CHEMOTAXIS_SSIM_WINDOW_SIZE // 2) * float(reference_spacing)
    sigma = CHEMOTAXIS_SSIM_SIGMA_CELLS * float(reference_spacing)
    delta = centers[:, None, :] - centers[None, :, :]
    distance_squared = np.sum(delta * delta, axis=2)
    tolerance = np.finfo(np.float64).eps * max(1.0, radius * radius) * 8.0
    mask = distance_squared <= radius * radius + tolerance
    weights = areas[None, :] * np.exp(-0.5 * distance_squared / (sigma * sigma))
    weights = np.where(mask, weights, 0.0)
    totals = np.sum(weights, axis=1)
    if np.any(~np.isfinite(totals)) or np.any(totals <= 0.0):
        raise FloatingPointError("Chemotaxis local SSIM window has invalid weights.")
    weights /= totals[:, None]

    truth_mean = weights.dot(truth)
    pred_mean = weights.dot(prediction)
    truth_var = np.maximum(weights.dot(truth * truth) - truth_mean * truth_mean, 0.0)
    pred_var = np.maximum(
        weights.dot(prediction * prediction) - pred_mean * pred_mean,
        0.0,
    )
    covariance = weights.dot(truth * prediction) - truth_mean * pred_mean
    c1 = (CHEMOTAXIS_SSIM_K1 * data_range) ** 2
    c2 = (CHEMOTAXIS_SSIM_K2 * data_range) ** 2
    numerator = (2.0 * truth_mean * pred_mean + c1) * (2.0 * covariance + c2)
    denominator = (
        (truth_mean * truth_mean + pred_mean * pred_mean + c1)
        * (truth_var + pred_var + c2)
    )
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        local_ssim = numerator / denominator
    if not np.all(np.isfinite(local_ssim)):
        raise FloatingPointError("Chemotaxis local SSIM produced non-finite values.")
    local_ssim = np.clip(local_ssim, -1.0, 1.0)
    return float(np.sum(areas * local_ssim) / np.sum(areas))


def chemotaxis_high_gradient_final_rel_l2(
    truth,
    prediction,
    areas,
    edges,
    relative_threshold=HIGH_GRADIENT_RELATIVE_THRESHOLD,
):
    local_jump = np.zeros(truth.shape[0], dtype=np.float64)
    if edges.shape[1] == 0:
        return np.nan
    jump = np.abs(truth[edges[0]] - truth[edges[1]])
    np.maximum.at(local_jump, edges[0], jump)
    np.maximum.at(local_jump, edges[1], jump)
    max_jump = float(np.max(local_jump))
    if not math.isfinite(max_jump) or max_jump <= 0.0:
        return np.nan
    selected = local_jump >= float(relative_threshold) * max_jump
    numerator = np.sum(areas[selected] * (prediction[selected] - truth[selected]) ** 2)
    denominator = np.sum(areas[selected] * truth[selected] ** 2)
    denominator = max(float(denominator), np.finfo(np.float64).eps)
    return float(np.sqrt(numerator / denominator))


def extract_chemotaxis_metrics_row(
    base_dir,
    run_dir,
    eval_npz,
    summary_json,
    dataset,
    method,
):
    summary = load_json(summary_json)
    dataset_path = resolve_dataset_path(base_dir, dataset, summary)
    if dataset_path is None or not dataset_path.exists():
        raise FileNotFoundError(
            "Could not resolve the chemotaxis source dataset from {0}.".format(
                summary_json
            )
        )

    with np.load(eval_npz, allow_pickle=False) as archive:
        required = ("trajectory_ids", "truth", "prediction", "time", "mass_error")
        missing = [name for name in required if name not in archive.files]
        if missing:
            raise KeyError(
                "{0} is missing array(s): {1}".format(eval_npz, ", ".join(missing))
            )
        trajectory_ids = np.asarray(archive["trajectory_ids"], dtype=np.int64).reshape(-1)
        truth = chemotaxis_density_array(archive["truth"], "truth", eval_npz).copy()
        prediction = chemotaxis_density_array(
            archive["prediction"], "prediction", eval_npz
        ).copy()
        time = np.asarray(archive["time"], dtype=np.float64).reshape(-1).copy()
        mass_error = np.asarray(archive["mass_error"], dtype=np.float64).copy()

    if prediction.shape != truth.shape:
        raise ValueError("Chemotaxis truth and prediction shapes differ in {0}.".format(eval_npz))
    num_samples, num_times, num_nodes = truth.shape
    if trajectory_ids.shape != (num_samples,):
        raise ValueError("Chemotaxis trajectory IDs do not match rollout rows in {0}.".format(eval_npz))
    if time.shape != (num_times,) or mass_error.shape != (num_samples, num_times):
        raise ValueError("Chemotaxis time or mass-error shape is incompatible in {0}.".format(eval_npz))
    if not all(np.all(np.isfinite(values)) for values in (truth, prediction, time, mass_error)):
        raise ValueError("Chemotaxis rollout artifact contains non-finite values: {0}".format(eval_npz))

    geometries = load_chemotaxis_geometry(dataset_path, trajectory_ids, num_nodes)
    all_areas = np.concatenate([areas for _, areas, _ in geometries])
    reference_spacing = float(np.sqrt(np.median(all_areas)))
    data_range = float(np.max(truth) - np.min(truth))
    if not math.isfinite(data_range) or data_range <= 0.0:
        raise ValueError("Chemotaxis reference density range must be positive.")

    final_truth = truth[:, -1, :]
    final_prediction = prediction[:, -1, :]
    rel_l2_values = []
    ssim_values = []
    high_gradient_values = []
    for sample_index, (centers, areas, edges) in enumerate(geometries):
        truth_row = final_truth[sample_index]
        pred_row = final_prediction[sample_index]
        numerator = np.sum(areas * (pred_row - truth_row) ** 2)
        denominator = max(
            float(np.sum(areas * truth_row**2)),
            np.finfo(np.float64).eps,
        )
        rel_l2_values.append(float(np.sqrt(numerator / denominator)))
        ssim_values.append(
            chemotaxis_local_ssim_final(
                truth_row,
                pred_row,
                centers,
                areas,
                reference_spacing,
                data_range,
            )
        )
        high_gradient_values.append(
            chemotaxis_high_gradient_final_rel_l2(
                truth_row,
                pred_row,
                areas,
                edges,
            )
        )

    rel_mean, rel_std, rel_n = mean_std(rel_l2_values)
    ssim_mean, ssim_std, ssim_n = mean_std(ssim_values)
    high_grad_mean, high_grad_std, high_grad_n = mean_std(high_gradient_values)
    signed_final = mass_error[:, -1]
    abs_final = np.abs(signed_final)
    max_abs = np.max(np.abs(mass_error), axis=1)
    cons_mean, cons_std, cons_n = mean_std(abs_final)
    signed_mean, signed_std, _ = mean_std(signed_final)
    max_cons_mean, max_cons_std, _ = mean_std(max_abs)

    dt = None
    if time.size > 1:
        time_steps = np.diff(time)
        if np.allclose(time_steps, time_steps[0], rtol=1.0e-8, atol=1.0e-12):
            dt = float(time_steps[0])
    final_time = float(time[-1] - time[0]) if time.size else None
    notes = (
        "chemotaxis metrics computed from saved test rollouts; relative L2 and "
        "high-gradient relative L2 are cell-area weighted; SSIM uses local 2D "
        "Gaussian mesh windows with truncated no-flux boundaries; conservation "
        "error uses saved cell-area-weighted mass error; dataset={0}; "
        "SSIM data range={1:.12g}; SSIM reference spacing={2:.12g}"
    ).format(dataset_path, data_range, reference_spacing)

    if rel_n != num_samples or ssim_n != num_samples or cons_n != num_samples:
        raise ValueError("Chemotaxis metrics were not available for every test trajectory.")

    return {
        "dataset": dataset,
        "method": method,
        "run_dir": str(run_dir),
        "eval_npz": str(eval_npz),
        "summary_json": str(summary_json),
        "num_samples_metrics": num_samples,
        "horizon_steps": num_times - 1,
        "dt": dt,
        "final_time": final_time,
        "final_rel_l2_mean": rel_mean,
        "final_rel_l2_std": rel_std,
        "final_ssim_mean": ssim_mean,
        "final_ssim_std": ssim_std,
        "final_high_gradient_rel_l2_mean": high_grad_mean,
        "final_high_gradient_rel_l2_std": high_grad_std,
        "final_high_gradient_rel_l2_n": high_grad_n,
        "high_gradient_relative_threshold": HIGH_GRADIENT_RELATIVE_THRESHOLD,
        "final_conservation_error_mean": cons_mean,
        "final_conservation_error_std": cons_std,
        "final_conservation_error_n": cons_n,
        "final_conservation_error_source": "saved_cell_area_weighted_mass_error",
        "final_conservation_signed_drift_mean": signed_mean,
        "final_conservation_signed_drift_std": signed_std,
        "max_conservation_error_over_time_mean": max_cons_mean,
        "max_conservation_error_over_time_std": max_cons_std,
        "notes": notes,
    }


def extract_row(base_dir, dataset, method, relative_run_path):
    run_dir, eval_npz, summary_json = resolve_eval_artifact(
        base_dir,
        dataset,
        relative_run_path,
    )
    if dataset == "LWR Traffic" and eval_npz.name.startswith("lwr_rollout_metrics"):
        return extract_lwr_metrics_row(run_dir, eval_npz, summary_json, dataset, method)
    if dataset == "Chemotactic Cell Migration":
        return extract_chemotaxis_metrics_row(
            base_dir,
            run_dir,
            eval_npz,
            summary_json,
            dataset,
            method,
        )

    if not run_dir.exists():
        raise FileNotFoundError(str(run_dir))
    if not eval_npz.exists():
        raise FileNotFoundError(str(eval_npz))

    summary = load_json(summary_json)
    npz = np.load(eval_npz, allow_pickle=False)

    rel_l2 = np.asarray(npz["rel_l2_gnn_by_sample"], dtype=np.float64)
    ssim = np.asarray(npz["ssim_gnn_by_sample"], dtype=np.float64)
    num_samples = int(rel_l2.shape[0])
    horizon = int(np.asarray(npz["horizon"]).item()) if "horizon" in npz.files else int(rel_l2.shape[1])

    rel_mean, rel_std, rel_n = mean_std(rel_l2[:, -1])
    ssim_mean, ssim_std, ssim_n = mean_std(ssim[:, -1])

    signed_drift, abs_drift, max_abs_drift, cons_source, cons_note = get_conservation(
        base_dir,
        dataset,
        summary,
        npz,
    )
    cons_mean, cons_std, cons_n = mean_std(abs_drift)
    signed_mean, signed_std, _ = mean_std(signed_drift)
    max_cons_mean, max_cons_std, _ = mean_std(max_abs_drift)
    metric_pred, metric_true, _, high_grad_note = prediction_reference_trajs(
        base_dir,
        dataset,
        summary,
        npz,
    )
    high_grad_values = (
        high_gradient_final_rel_l2(metric_pred, metric_true)
        if metric_pred is not None and metric_true is not None
        else np.asarray([], dtype=np.float64)
    )
    high_grad_mean, high_grad_std, high_grad_n = mean_std(high_grad_values)

    dt = finite_float(summary.get("dt"))
    final_time = None if dt is None else dt * float(horizon)

    notes = []
    if rel_n != num_samples:
        notes.append("rel_l2 finite n={0} differs from metric row count n={1}".format(rel_n, num_samples))
    if ssim_n != num_samples:
        notes.append("ssim finite n={0} differs from metric row count n={1}".format(ssim_n, num_samples))
    if cons_note:
        notes.append(cons_note)
    if high_grad_note and high_grad_note != cons_note:
        notes.append(high_grad_note)
    if cons_n != num_samples:
        notes.append(
            "conservation error computed for n={0}; accuracy metrics use n={1}".format(
                cons_n,
                num_samples,
            )
        )

    return {
        "dataset": dataset,
        "method": method,
        "run_dir": str(run_dir),
        "eval_npz": str(eval_npz),
        "summary_json": str(summary_json),
        "num_samples_metrics": num_samples,
        "horizon_steps": horizon,
        "dt": dt,
        "final_time": final_time,
        "final_rel_l2_mean": rel_mean,
        "final_rel_l2_std": rel_std,
        "final_ssim_mean": ssim_mean,
        "final_ssim_std": ssim_std,
        "final_high_gradient_rel_l2_mean": high_grad_mean,
        "final_high_gradient_rel_l2_std": high_grad_std,
        "final_high_gradient_rel_l2_n": high_grad_n,
        "high_gradient_relative_threshold": HIGH_GRADIENT_RELATIVE_THRESHOLD,
        "final_conservation_error_mean": cons_mean,
        "final_conservation_error_std": cons_std,
        "final_conservation_error_n": cons_n,
        "final_conservation_error_source": cons_source,
        "final_conservation_signed_drift_mean": signed_mean,
        "final_conservation_signed_drift_std": signed_std,
        "max_conservation_error_over_time_mean": max_cons_mean,
        "max_conservation_error_over_time_std": max_cons_std,
        "notes": "; ".join(notes),
    }


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in COLUMNS})


def write_json(path, base_dir, rows, skipped_runs=None, training_columns=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_script": str(Path(__file__).resolve()),
        "base_dir": str(base_dir),
        "training_columns": training_columns or [],
        "metric_definitions": {
            "final_rel_l2": (
                "Mean and sample standard deviation of relative L2 error at "
                "the final rollout step. Chemotaxis uses finite-volume cell-area "
                "weighting; the one-dimensional benchmarks use their saved "
                "discrete relative-L2 values."
            ),
            "final_ssim": (
                "Mean and sample standard deviation of SSIM at the final rollout "
                "step. Chemotaxis uses area-weighted local two-dimensional mesh "
                "SSIM with truncated, renormalized no-flux boundary windows."
            ),
            "final_conservation_error": (
                "For each trajectory, compute the absolute final conserved-quantity "
                "error and report its mean and sample standard deviation. The "
                "one-dimensional benchmarks use summed-field drift; chemotaxis "
                "uses the saved cell-area-weighted total-mass error."
            ),
            "final_high_gradient_rel_l2": (
                "At the final rollout step, compute relative L2 error only on "
                "nodes adjacent to a large reference jump. The local jump is "
                "computed from left/right neighbors in one dimension and from "
                "face-adjacent cells on the chemotaxis mesh. Chemotaxis also uses "
                "cell-area weighting. The inclusion threshold is 0.2 times the "
                "maximum local jump for that trajectory."
            ),
            "final_conservation_signed_drift": (
                "Mean and sample standard deviation of the signed final "
                "conserved-quantity error. This is saved only for diagnostics "
                "and is not intended for the main table because positive and "
                "negative errors can cancel."
            ),
            "max_conservation_error_over_time": (
                "Per-trajectory maximum absolute conserved-quantity error over "
                "rollout time, then mean/std over available trajectories."
            ),
            "std": "Sample standard deviation with ddof=1 where at least two values are available; otherwise null.",
        },
        "skipped_runs": skipped_runs or [],
        "rows": rows,
    }
    with path.open("w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False)


def main():
    from make_model_hyperparameters import COLUMNS as TRAINING_COLUMNS
    from make_model_hyperparameters import build_row as build_training_row

    args = parse_args()
    rows = []
    skipped_runs = []
    for run in RUNS:
        if len(run) < 3:
            raise ValueError("Each RUNS entry must have at least dataset, method, and path.")
        dataset, method = run[0], run[1]
        relative_run_path = Path(str(run[2]))
        for part in run[3:]:
            relative_run_path = relative_run_path / str(part)
        if is_placeholder_run_path(relative_run_path):
            skipped_runs.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "run_dir": str(relative_run_path),
                    "reason": "placeholder run path",
                }
            )
            continue
        row = extract_row(args.base_dir, dataset, method, relative_run_path)
        flat_training, rich_training = build_training_row(
            args.base_dir,
            dataset,
            method,
            relative_run_path,
        )
        for key, value in flat_training.items():
            row.setdefault(key, value)
        row["training"] = rich_training
        rows.append(row)

    write_json(
        args.output_json,
        args.base_dir,
        rows,
        skipped_runs=skipped_runs,
        training_columns=TRAINING_COLUMNS,
    )
    write_csv(args.output_csv, rows)
    print("Wrote {0}".format(args.output_json))
    print("Wrote {0}".format(args.output_csv))
    if skipped_runs:
        print("Skipped {0} placeholder run path(s).".format(len(skipped_runs)))


if __name__ == "__main__":
    main()
