#!/usr/bin/env python
"""Build a table-ready LWR traffic metrics NPZ from rollout files.

The LWR traffic runs save per-topology rollout artifacts under

    rollout_plots/<split>/topology_XXXX/rollout_predictions.npz

rather than the gnn_vs_godunov_eval_local_ssim.npz files used by the
one-dimensional PDE experiments. This script reads those rollout artifacts,
computes per-trajectory metrics over all selected test samples, and writes one
compressed NPZ that can be used as the canonical result artifact for the paper
table.

Default usage from the paper workspace root:

    python scripts/make_lwr_rollout_metrics_npz.py \
        /path/to/model/run/or/rollout_plots \
        --output /path/to/lwr_rollout_metrics_all_tests.npz

If --splits is omitted, all discovered split directories whose names start
with "test" are included. For the paper experiments this usually means
test_in_size, test_large, and test_parameter_ood.
"""

from __future__ import print_function

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


DEFAULT_BASE_DIR = Path("/Users/trevorreed/UVA_postdoc_stuff/gparc/toy_model")
DEFAULT_STANDARD_TEST_SPLITS = ("test_in_size", "test_large", "test_parameter_ood")
HIGH_GRADIENT_RELATIVE_THRESHOLD = 0.2
SSIM_K1 = 0.01
SSIM_K2 = 0.03
SSIM_WINDOW_SIZE = 11
SSIM_SIGMA_CELLS = 1.5


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a metrics NPZ from LWR traffic rollout_predictions.npz files."
    )
    parser.add_argument(
        "rollout_dir",
        type=Path,
        help=(
            "Model run directory, rollout_plots directory, summary.json, or "
            "checkpoint.pt."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output NPZ path. Defaults to "
            "<rollout_plots>/lwr_rollout_metrics_all_tests.npz."
        ),
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing topology_XXXX.npz files. If omitted, the "
            "path is inferred from rollout_plots/summary.json."
        ),
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help="Base toy_model directory used to resolve relative dataset paths.",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=None,
        help=(
            "Split names to include. Omit to include all discovered test* "
            "splits, with standard paper splits first."
        ),
    )
    parser.add_argument(
        "--allow-missing-splits",
        action="store_true",
        help="Skip requested splits that are not present instead of failing.",
    )
    parser.add_argument(
        "--high-gradient-relative-threshold",
        type=float,
        default=HIGH_GRADIENT_RELATIVE_THRESHOLD,
        help=(
            "Final-time high-gradient mask threshold as a fraction of the "
            "maximum roadwise jump for a trajectory."
        ),
    )
    parser.add_argument(
        "--ssim-window-size",
        type=int,
        default=SSIM_WINDOW_SIZE,
        help="Odd number of reference cells in the local SSIM window.",
    )
    parser.add_argument(
        "--ssim-sigma-cells",
        type=float,
        default=SSIM_SIGMA_CELLS,
        help="Gaussian local-SSIM sigma in reference-cell units.",
    )
    parser.add_argument(
        "--ssim-reference-spacing",
        type=float,
        default=None,
        help=(
            "Physical reference spacing for local SSIM. If omitted, use the "
            "median cell length across selected topology metadata."
        ),
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


def mean_std(values, weights=None):
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = np.isfinite(arr)
    if weights is None:
        vals = arr[finite]
        if vals.size == 0:
            return np.nan, np.nan, 0
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if vals.size > 1 else np.nan
        return mean, std, int(vals.size)

    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    finite = finite & np.isfinite(w) & (w > 0.0)
    vals = arr[finite]
    weights_finite = w[finite]
    if vals.size == 0:
        return np.nan, np.nan, 0
    weights_finite = weights_finite / np.sum(weights_finite)
    mean = float(np.sum(weights_finite * vals))
    variance = float(np.sum(weights_finite * (vals - mean) ** 2))
    return mean, math.sqrt(max(variance, 0.0)), int(vals.size)


def resolve_rollout_root(path):
    path = path.expanduser()
    if path.is_file():
        if path.name == "summary.json":
            candidate = path.parent
        elif path.name == "checkpoint.pt":
            candidate = path.parent / "rollout_plots"
        else:
            raise ValueError(
                "Input file must be summary.json or checkpoint.pt, got {0}".format(path)
            )
    elif path.is_dir():
        candidate = path if (path / "summary.json").is_file() else path / "rollout_plots"
    else:
        raise FileNotFoundError(str(path))

    if not candidate.is_dir():
        raise FileNotFoundError("Rollout directory does not exist: {0}".format(candidate))
    if not any(candidate.glob("*/topology_*/rollout_predictions.npz")):
        raise FileNotFoundError(
            "No rollout_predictions.npz files found under {0}".format(candidate)
        )
    return candidate.resolve()


def load_json(path):
    with path.open("r") as fh:
        return json.load(fh)


def resolve_dataset_dir(rollout_root, explicit_dataset_dir, base_dir):
    if explicit_dataset_dir is not None:
        path = explicit_dataset_dir.expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError("Dataset directory does not exist: {0}".format(path))
        return path

    summary_path = rollout_root / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            "{0} is required to infer --dataset-dir.".format(summary_path)
        )
    summary = load_json(summary_path)
    if "data" not in summary:
        raise KeyError("{0} does not contain a 'data' entry.".format(summary_path))

    raw = Path(str(summary["data"])).expanduser()
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend(
            [
                Path.cwd() / raw,
                base_dir / raw,
                base_dir / "1D_traffic" / raw,
            ]
        )
        candidates.extend(parent / raw for parent in rollout_root.parents)

    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not resolve dataset directory {0} from {1}.".format(
            raw, summary_path
        )
    )


def discover_rollout_files(rollout_root):
    discovered = {}
    for path in sorted(rollout_root.glob("*/topology_*/rollout_predictions.npz")):
        split = path.parent.parent.name
        topology_name = path.parent.name
        prefix = "topology_"
        if topology_name.startswith(prefix):
            topology_id = int(topology_name[len(prefix) :])
        else:
            topology_id = int(topology_name)
        key = (split, topology_id)
        if key in discovered:
            raise ValueError("Duplicate rollout artifact for {0}: {1}".format(key, path))
        discovered[key] = path
    return discovered


def select_splits(discovered, requested_splits, allow_missing=False):
    available = sorted({split for split, _ in discovered})
    if requested_splits:
        missing = [split for split in requested_splits if split not in available]
        if missing and not allow_missing:
            raise ValueError(
                "Requested split(s) not found: {0}. Available splits: {1}".format(
                    ", ".join(missing), ", ".join(available)
                )
            )
        return [split for split in requested_splits if split in available]

    test_splits = [split for split in available if split.startswith("test")]
    standard = [split for split in DEFAULT_STANDARD_TEST_SPLITS if split in test_splits]
    extras = [split for split in test_splits if split not in standard]
    selected = standard + extras
    return selected if selected else available


def topology_path(dataset_dir, topology_id):
    direct = dataset_dir / "topology_{0:04d}.npz".format(int(topology_id))
    if direct.exists():
        return direct
    matches = sorted(dataset_dir.glob("**/topology_{0:04d}.npz".format(int(topology_id))))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        "Could not find topology_{0:04d}.npz under {1}".format(
            int(topology_id), dataset_dir
        )
    )


def load_topology_metadata(path, expected_topology_id):
    required = ("cell_lengths", "cell_road_id", "cell_road_coordinate", "rho_max")
    with np.load(path, allow_pickle=False) as archive:
        missing = [name for name in required if name not in archive]
        if missing:
            raise KeyError(
                "{0} is missing topology metric metadata: {1}".format(
                    path, ", ".join(missing)
                )
            )
        topology_id = (
            int(np.asarray(archive["topology_id"]).item())
            if "topology_id" in archive
            else int(expected_topology_id)
        )
        cell_lengths = np.asarray(archive["cell_lengths"], dtype=np.float64).reshape(-1)
        cell_road_id = np.asarray(archive["cell_road_id"], dtype=np.int64).reshape(-1)
        cell_road_coordinate = np.asarray(
            archive["cell_road_coordinate"], dtype=np.float64
        ).reshape(-1)
        rho_max = float(np.asarray(archive["rho_max"]).item())

    if topology_id != int(expected_topology_id):
        raise ValueError(
            "Topology ID in {0} is {1}; expected {2}.".format(
                path, topology_id, expected_topology_id
            )
        )
    node_count = cell_lengths.size
    if node_count == 0:
        raise ValueError("{0} has no cells.".format(path))
    for name, values in (
        ("cell_road_id", cell_road_id),
        ("cell_road_coordinate", cell_road_coordinate),
    ):
        if values.shape != (node_count,):
            raise ValueError("{0} shape does not match cell_lengths in {1}.".format(name, path))
    if not np.all(np.isfinite(cell_lengths)) or np.any(cell_lengths <= 0.0):
        raise ValueError("cell_lengths in {0} must be positive and finite.".format(path))
    if not np.isfinite(rho_max) or rho_max <= 0.0:
        raise ValueError("rho_max in {0} must be positive and finite.".format(path))

    return {
        "topology_id": int(topology_id),
        "cell_lengths": cell_lengths,
        "cell_road_id": cell_road_id,
        "cell_road_coordinate": cell_road_coordinate,
        "rho_max": rho_max,
        "source": str(path),
    }


def load_rollout(path):
    with np.load(path, allow_pickle=False) as archive:
        required = (
            "topology_id",
            "trajectory_ids",
            "ground_truth",
            "prediction",
            "dt",
        )
        missing = [name for name in required if name not in archive]
        if missing:
            raise KeyError("{0} is missing {1}".format(path, ", ".join(missing)))
        data = {
            "topology_id": int(np.asarray(archive["topology_id"]).item()),
            "trajectory_ids": np.asarray(archive["trajectory_ids"], dtype=np.int64).copy(),
            "ground_truth": np.asarray(archive["ground_truth"], dtype=np.float64).copy(),
            "prediction": np.asarray(archive["prediction"], dtype=np.float64).copy(),
            "dt": float(np.asarray(archive["dt"]).item()),
            "trajectory_pair_id": (
                np.asarray(archive["trajectory_pair_id"], dtype=np.int64).copy()
                if "trajectory_pair_id" in archive
                else None
            ),
            "flow_direction": (
                np.asarray(archive["trajectory_flow_direction"], dtype=np.float64).copy()
                if "trajectory_flow_direction" in archive
                else None
            ),
            "mass_error": (
                np.asarray(archive["mass_error"], dtype=np.float64).copy()
                if "mass_error" in archive
                else None
            ),
            "relative_mass_error": (
                np.asarray(archive["relative_mass_error_by_time"], dtype=np.float64).copy()
                if "relative_mass_error_by_time" in archive
                else None
            ),
        }

    truth = data["ground_truth"]
    pred = data["prediction"]
    if truth.ndim != 3:
        raise ValueError("ground_truth in {0} must be [trajectory,time,node].".format(path))
    if pred.shape != truth.shape:
        raise ValueError("prediction shape does not match ground_truth in {0}.".format(path))
    if data["trajectory_ids"].shape != (truth.shape[0],):
        raise ValueError("trajectory_ids shape mismatch in {0}.".format(path))
    if data["trajectory_pair_id"] is None:
        data["trajectory_pair_id"] = np.full(truth.shape[0], -1, dtype=np.int64)
    if data["flow_direction"] is None:
        data["flow_direction"] = np.full(truth.shape[0], np.nan, dtype=np.float64)
    for key in ("trajectory_pair_id", "flow_direction"):
        if data[key].shape != (truth.shape[0],):
            raise ValueError("{0} shape mismatch in {1}.".format(key, path))
    if data["mass_error"] is not None and data["mass_error"].shape != truth.shape[:2]:
        raise ValueError("mass_error shape mismatch in {0}.".format(path))
    if data["relative_mass_error"] is not None and data["relative_mass_error"].shape != truth.shape[:2]:
        raise ValueError("relative_mass_error_by_time shape mismatch in {0}.".format(path))
    if not np.isfinite(data["dt"]) or data["dt"] <= 0.0:
        raise ValueError("dt in {0} must be positive and finite.".format(path))
    return data


def cell_length_weighted_relative_l2_by_time(truth, prediction, cell_lengths):
    lengths = np.asarray(cell_lengths, dtype=np.float64).reshape(-1)
    if truth.ndim != 2 or prediction.shape != truth.shape:
        raise ValueError("truth and prediction must be [time,node].")
    if lengths.shape != (truth.shape[1],):
        raise ValueError("cell_lengths must match node dimension.")
    numerator = np.sum((prediction - truth) ** 2 * lengths[None, :], axis=1)
    denominator = np.sum(truth**2 * lengths[None, :], axis=1)
    return np.sqrt(numerator / np.maximum(denominator, np.finfo(np.float64).eps))


def gaussian_road_window_weights(coordinates, cell_lengths, reference_spacing, window_size, sigma_cells):
    coordinates = np.asarray(coordinates, dtype=np.float64).reshape(-1)
    cell_lengths = np.asarray(cell_lengths, dtype=np.float64).reshape(-1)
    radius = (int(window_size) // 2) * float(reference_spacing)
    sigma = float(sigma_cells) * float(reference_spacing)
    distance = np.abs(coordinates[:, None] - coordinates[None, :])
    weights = np.exp(-0.5 * (distance / sigma) ** 2) * cell_lengths[None, :]
    weights[distance > radius + np.finfo(np.float64).eps] = 0.0
    row_sums = np.sum(weights, axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise RuntimeError("A local SSIM window has no samples.")
    return weights / row_sums


def roadwise_local_ssim_by_time(
    truth,
    prediction,
    metadata,
    reference_spacing,
    window_size,
    sigma_cells,
):
    if truth.ndim != 2 or prediction.shape != truth.shape:
        raise ValueError("truth and prediction must be [time,node].")
    cell_lengths = metadata["cell_lengths"]
    cell_road_id = metadata["cell_road_id"]
    cell_road_coordinate = metadata["cell_road_coordinate"]
    if cell_lengths.shape != (truth.shape[1],):
        raise ValueError("metadata cell count does not match rollout node count.")

    c1 = (SSIM_K1 * metadata["rho_max"]) ** 2
    c2 = (SSIM_K2 * metadata["rho_max"]) ** 2
    weighted_score = np.zeros(truth.shape[0], dtype=np.float64)
    total_length = float(np.sum(cell_lengths))

    for road_id in np.unique(cell_road_id):
        indices = np.flatnonzero(cell_road_id == road_id)
        order = np.argsort(cell_road_coordinate[indices], kind="stable")
        indices = indices[order]
        coordinates = cell_road_coordinate[indices]
        windows = gaussian_road_window_weights(
            coordinates,
            cell_lengths=cell_lengths[indices],
            reference_spacing=reference_spacing,
            window_size=window_size,
            sigma_cells=sigma_cells,
        )
        true_road = truth[:, indices]
        pred_road = prediction[:, indices]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            true_mean = true_road @ windows.T
            pred_mean = pred_road @ windows.T
            true_var = true_road**2 @ windows.T - true_mean**2
            pred_var = pred_road**2 @ windows.T - pred_mean**2
            cov = (true_road * pred_road) @ windows.T - true_mean * pred_mean
            true_var = np.maximum(true_var, 0.0)
            pred_var = np.maximum(pred_var, 0.0)
            numerator = (2.0 * true_mean * pred_mean + c1) * (2.0 * cov + c2)
            denominator = (true_mean**2 + pred_mean**2 + c1) * (true_var + pred_var + c2)
            local_ssim = numerator / denominator
        local_ssim = np.clip(local_ssim, -1.0, 1.0)
        weighted_score += np.sum(local_ssim * cell_lengths[indices][None, :], axis=1)

    return weighted_score / total_length


def high_gradient_final_rel_l2(truth_final, pred_final, metadata, relative_threshold):
    cell_lengths = metadata["cell_lengths"]
    cell_road_id = metadata["cell_road_id"]
    cell_road_coordinate = metadata["cell_road_coordinate"]
    local_jump = np.zeros_like(truth_final, dtype=np.float64)

    for road_id in np.unique(cell_road_id):
        indices = np.flatnonzero(cell_road_id == road_id)
        order = np.argsort(cell_road_coordinate[indices], kind="stable")
        indices = indices[order]
        values = truth_final[indices]
        if values.size <= 1:
            continue
        jumps = np.abs(np.diff(values))
        local_jump[indices[:-1]] = np.maximum(local_jump[indices[:-1]], jumps)
        local_jump[indices[1:]] = np.maximum(local_jump[indices[1:]], jumps)

    max_jump = float(np.nanmax(local_jump)) if local_jump.size else 0.0
    if not math.isfinite(max_jump) or max_jump <= 0.0:
        return np.nan
    mask = local_jump >= float(relative_threshold) * max_jump
    if not np.any(mask):
        return np.nan
    numerator = np.sum((pred_final[mask] - truth_final[mask]) ** 2 * cell_lengths[mask])
    denominator = np.sum(truth_final[mask] ** 2 * cell_lengths[mask])
    return float(np.sqrt(numerator / max(denominator, np.finfo(np.float64).eps)))


def compute_mass_error(truth, prediction, metadata):
    lengths = metadata["cell_lengths"]
    initial_true_mass = np.sum(truth[:, 0, :] * lengths[None, :], axis=1)
    pred_mass = np.sum(prediction * lengths[None, None, :], axis=2)
    signed = pred_mass - initial_true_mass[:, None]
    denom = np.maximum(np.abs(initial_true_mass[:, None]), np.finfo(np.float64).eps)
    return signed, signed / denom


def sample_weights(split_values, topology_values):
    split_values = np.asarray(split_values)
    topology_values = np.asarray(topology_values)
    n = split_values.size
    equal = np.full(n, 1.0 / n if n else np.nan, dtype=np.float64)
    balanced = np.zeros(n, dtype=np.float64)

    unique_splits = list(dict.fromkeys(split_values.tolist()))
    for split in unique_splits:
        split_mask = split_values == split
        split_topologies = list(dict.fromkeys(topology_values[split_mask].tolist()))
        for topology in split_topologies:
            topo_mask = split_mask & (topology_values == topology)
            count = int(np.sum(topo_mask))
            if count:
                balanced[topo_mask] = 1.0 / (len(unique_splits) * len(split_topologies) * count)
    return equal, balanced


def metric_summary(prefix, values, equal_weights, balanced_weights):
    equal_mean, equal_std, equal_n = mean_std(values, weights=equal_weights)
    balanced_mean, balanced_std, balanced_n = mean_std(values, weights=balanced_weights)
    raw_mean, raw_std, raw_n = mean_std(values)
    return {
        prefix + "_mean": raw_mean,
        prefix + "_std": raw_std,
        prefix + "_n": raw_n,
        prefix + "_equal_weight_mean": equal_mean,
        prefix + "_equal_weight_std": equal_std,
        prefix + "_equal_weight_n": equal_n,
        prefix + "_split_topology_balanced_mean": balanced_mean,
        prefix + "_split_topology_balanced_std": balanced_std,
        prefix + "_split_topology_balanced_n": balanced_n,
    }


def main():
    args = parse_args()
    if args.ssim_window_size < 3 or args.ssim_window_size % 2 == 0:
        raise ValueError("--ssim-window-size must be an odd integer >= 3.")
    if args.ssim_sigma_cells <= 0.0 or not np.isfinite(args.ssim_sigma_cells):
        raise ValueError("--ssim-sigma-cells must be positive and finite.")
    if (
        args.high_gradient_relative_threshold < 0.0
        or not np.isfinite(args.high_gradient_relative_threshold)
    ):
        raise ValueError("--high-gradient-relative-threshold must be finite and >= 0.")

    rollout_root = resolve_rollout_root(args.rollout_dir)
    dataset_dir = resolve_dataset_dir(rollout_root, args.dataset_dir, args.base_dir)
    output = args.output
    if output is None:
        output = rollout_root / "lwr_rollout_metrics_all_tests.npz"

    discovered = discover_rollout_files(rollout_root)
    selected_splits = select_splits(
        discovered,
        args.splits,
        allow_missing=args.allow_missing_splits,
    )
    selected_items = [
        (split, topology_id, path)
        for (split, topology_id), path in sorted(discovered.items())
        if split in selected_splits
    ]
    if not selected_items:
        raise ValueError("No rollout files selected.")

    topology_ids_needed = sorted({topology_id for _, topology_id, _ in selected_items})
    metadata_by_topology = {
        topology_id: load_topology_metadata(topology_path(dataset_dir, topology_id), topology_id)
        for topology_id in topology_ids_needed
    }
    if args.ssim_reference_spacing is not None:
        reference_spacing = float(args.ssim_reference_spacing)
        reference_spacing_source = "command_line"
    else:
        reference_spacing = float(
            np.median(
                np.concatenate(
                    [metadata_by_topology[topology_id]["cell_lengths"] for topology_id in topology_ids_needed]
                )
            )
        )
        reference_spacing_source = "median_selected_topology_cell_length"

    rows = []
    rel_l2_curves = []
    ssim_curves = []
    high_grad_values = []
    mass_error_curves = []
    relative_mass_error_curves = []
    time_lengths = []

    for split, topology_id, path in selected_items:
        rollout = load_rollout(path)
        metadata = metadata_by_topology[topology_id]
        truth_all = rollout["ground_truth"]
        pred_all = rollout["prediction"]
        if metadata["cell_lengths"].shape != (truth_all.shape[2],):
            raise ValueError(
                "Topology {0} metadata has {1} cells but rollout has {2} nodes in {3}.".format(
                    topology_id,
                    metadata["cell_lengths"].size,
                    truth_all.shape[2],
                    path,
                )
            )
        if rollout["mass_error"] is None or rollout["relative_mass_error"] is None:
            mass_error_all, relative_mass_error_all = compute_mass_error(
                truth_all, pred_all, metadata
            )
        else:
            mass_error_all = rollout["mass_error"]
            relative_mass_error_all = rollout["relative_mass_error"]

        for row_index, trajectory_id in enumerate(rollout["trajectory_ids"]):
            truth = truth_all[row_index]
            pred = pred_all[row_index]
            rel_l2 = cell_length_weighted_relative_l2_by_time(
                truth,
                pred,
                metadata["cell_lengths"],
            )
            ssim = roadwise_local_ssim_by_time(
                truth,
                pred,
                metadata,
                reference_spacing=reference_spacing,
                window_size=args.ssim_window_size,
                sigma_cells=args.ssim_sigma_cells,
            )
            high_grad = high_gradient_final_rel_l2(
                truth[-1],
                pred[-1],
                metadata,
                args.high_gradient_relative_threshold,
            )
            rel_l2_curves.append(rel_l2)
            ssim_curves.append(ssim)
            high_grad_values.append(high_grad)
            mass_error_curves.append(mass_error_all[row_index])
            relative_mass_error_curves.append(relative_mass_error_all[row_index])
            time_lengths.append(rel_l2.size)
            rows.append(
                {
                    "split": split,
                    "topology_id": int(topology_id),
                    "trajectory_id": int(trajectory_id),
                    "trajectory_pair_id": int(rollout["trajectory_pair_id"][row_index]),
                    "flow_direction": float(rollout["flow_direction"][row_index]),
                    "dt": float(rollout["dt"]),
                    "num_nodes": int(truth.shape[1]),
                    "rollout_file": str(path),
                    "topology_metadata": metadata["source"],
                }
            )

    common_num_times = min(time_lengths)
    if common_num_times < 2:
        raise ValueError("Selected rollouts must contain at least one forecast step.")
    rel_l2_by_sample = np.stack([curve[:common_num_times] for curve in rel_l2_curves], axis=0)
    ssim_by_sample = np.stack([curve[:common_num_times] for curve in ssim_curves], axis=0)
    mass_error_by_sample = np.stack(
        [curve[:common_num_times] for curve in mass_error_curves], axis=0
    )
    relative_mass_error_by_sample = np.stack(
        [curve[:common_num_times] for curve in relative_mass_error_curves], axis=0
    )

    split_by_sample = np.asarray([row["split"] for row in rows])
    topology_id_by_sample = np.asarray([row["topology_id"] for row in rows], dtype=np.int64)
    trajectory_id_by_sample = np.asarray([row["trajectory_id"] for row in rows], dtype=np.int64)
    trajectory_pair_id_by_sample = np.asarray(
        [row["trajectory_pair_id"] for row in rows], dtype=np.int64
    )
    flow_direction_by_sample = np.asarray([row["flow_direction"] for row in rows], dtype=np.float64)
    dt_by_sample = np.asarray([row["dt"] for row in rows], dtype=np.float64)
    num_nodes_by_sample = np.asarray([row["num_nodes"] for row in rows], dtype=np.int64)
    final_time_by_sample = dt_by_sample * float(common_num_times - 1)
    high_grad_by_sample = np.asarray(high_grad_values, dtype=np.float64)
    final_mass_error_by_sample = mass_error_by_sample[:, -1]
    final_conservation_error_by_sample = np.abs(final_mass_error_by_sample)
    max_conservation_error_over_time_by_sample = np.nanmax(
        np.abs(mass_error_by_sample), axis=1
    )
    final_relative_mass_error_by_sample = relative_mass_error_by_sample[:, -1]
    max_relative_mass_error_over_time_by_sample = np.nanmax(
        np.abs(relative_mass_error_by_sample), axis=1
    )

    equal_weights, balanced_weights = sample_weights(split_by_sample, topology_id_by_sample)

    summary = {}
    for prefix, values in (
        ("final_rel_l2", rel_l2_by_sample[:, -1]),
        ("final_ssim", ssim_by_sample[:, -1]),
        ("final_high_gradient_rel_l2", high_grad_by_sample),
        ("final_conservation_error", final_conservation_error_by_sample),
        ("final_conservation_signed_drift", final_mass_error_by_sample),
        ("max_conservation_error_over_time", max_conservation_error_over_time_by_sample),
        ("final_relative_mass_error", np.abs(final_relative_mass_error_by_sample)),
        ("max_relative_mass_error_over_time", max_relative_mass_error_over_time_by_sample),
    ):
        summary.update(metric_summary(prefix, values, equal_weights, balanced_weights))

    metric_definitions = {
        "rel_l2_gnn_by_sample": "Cell-length-weighted relative L2 error at each saved rollout step.",
        "ssim_gnn_by_sample": "Roadwise local Gaussian SSIM at each saved rollout step.",
        "final_high_gradient_rel_l2_by_sample": (
            "Final-time cell-length-weighted relative L2 error on cells adjacent "
            "to roadwise density jumps above the configured relative threshold."
        ),
        "final_conservation_error_by_sample": (
            "Absolute final finite-volume mass drift from the rollout artifact "
            "mass_error array, or recomputed from predicted density and cell lengths."
        ),
        "sample_weight_split_topology_balanced": (
            "Weights giving equal mass to each selected split, then equal mass "
            "to each topology within a split, then equal mass to trajectories "
            "within a topology."
        ),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        generated_at_utc=np.asarray(datetime.now(timezone.utc).isoformat()),
        source_script=np.asarray(str(Path(__file__).resolve())),
        rollout_root=np.asarray(str(rollout_root)),
        dataset_dir=np.asarray(str(dataset_dir)),
        selected_splits=np.asarray(selected_splits),
        rollout_files_by_sample=np.asarray([row["rollout_file"] for row in rows]),
        topology_metadata_by_sample=np.asarray([row["topology_metadata"] for row in rows]),
        split_by_sample=split_by_sample,
        topology_id_by_sample=topology_id_by_sample,
        trajectory_id_by_sample=trajectory_id_by_sample,
        trajectory_pair_id_by_sample=trajectory_pair_id_by_sample,
        flow_direction_by_sample=flow_direction_by_sample,
        dt_by_sample=dt_by_sample,
        final_time_by_sample=final_time_by_sample,
        num_nodes_by_sample=num_nodes_by_sample,
        num_samples_metrics=np.asarray(rel_l2_by_sample.shape[0], dtype=np.int64),
        horizon=np.asarray(common_num_times - 1, dtype=np.int64),
        horizon_steps=np.asarray(common_num_times - 1, dtype=np.int64),
        common_num_times=np.asarray(common_num_times, dtype=np.int64),
        original_num_times_by_sample=np.asarray(time_lengths, dtype=np.int64),
        rel_l2_gnn_by_sample=rel_l2_by_sample,
        ssim_gnn_by_sample=ssim_by_sample,
        final_high_gradient_rel_l2_by_sample=high_grad_by_sample,
        high_gradient_relative_threshold=np.asarray(
            args.high_gradient_relative_threshold, dtype=np.float64
        ),
        mass_error_by_sample=mass_error_by_sample,
        relative_mass_error_by_sample=relative_mass_error_by_sample,
        final_conservation_error_by_sample=final_conservation_error_by_sample,
        final_conservation_signed_drift_by_sample=final_mass_error_by_sample,
        max_conservation_error_over_time_by_sample=max_conservation_error_over_time_by_sample,
        final_relative_mass_error_by_sample=final_relative_mass_error_by_sample,
        max_relative_mass_error_over_time_by_sample=max_relative_mass_error_over_time_by_sample,
        sample_weight_equal=equal_weights,
        sample_weight_split_topology_balanced=balanced_weights,
        ssim_reference_spacing=np.asarray(reference_spacing, dtype=np.float64),
        ssim_reference_spacing_source=np.asarray(reference_spacing_source),
        ssim_window_size=np.asarray(args.ssim_window_size, dtype=np.int64),
        ssim_sigma_cells=np.asarray(args.ssim_sigma_cells, dtype=np.float64),
        metric_definitions_json=np.asarray(json.dumps(metric_definitions, sort_keys=True)),
        summary_json=np.asarray(json.dumps(summary, sort_keys=True)),
        **{key: np.asarray(value) for key, value in summary.items()}
    )

    print("Wrote {0}".format(output))
    print("Included {0} samples from {1} split(s).".format(len(rows), len(selected_splits)))
    print("Splits: {0}".format(", ".join(selected_splits)))


if __name__ == "__main__":
    main()
