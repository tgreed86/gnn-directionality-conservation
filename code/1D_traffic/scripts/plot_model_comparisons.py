#!/usr/bin/env python3
"""Overlay rollout diagnostics from two or more trained GNN models.

This script compares the ``rollout_predictions.npz`` artifacts written by
``plot_variable_sageconv_rollouts.py``.  It does not rerun model inference, so
all requested runs must already have rollout plots for the same dataset split,
topologies, trajectories, and rollout interval.

Each ``--model`` argument has the form ``LABEL=PATH``.  ``PATH`` may point to
a training-run directory, its ``rollout_plots`` directory, its
``rollout_plots/summary.json`` file, or its ``checkpoint.pt`` file.  Example:

    python scripts/plot_model_comparisons.py \
        --model "MeshGraphNet=runs/structured_directional/meshgraphnet/500ep" \
        --model "FluxGraphNet (flux 10)=runs/structured_directional/fluxgraphnet/flux-loss-weight/1000ep_lr-1e-4_pred-rate_flux-loss-10p0" \
        --splits test_large \
        --output-dir runs/structured_directional/model_comparison

The output contains aggregate comparison figures for each split, including a
Figure-6-style relative-L2/SSIM panel, overlaid per-trajectory diagnostics, a
CSV table, and a JSON summary.  Relative L2 is weighted by finite-volume cell
length.  SSIM is evaluated locally along each physical road with a fixed
density range and a Gaussian window expressed in physical-distance units;
windows never cross road junctions.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REQUIRED_ARRAYS = (
    "topology_id",
    "trajectory_ids",
    "ground_truth",
    "prediction",
    "relative_mass_error_by_time",
    "outside_bounds_by_time",
    "dt",
)

SSIM_WINDOW_SIZE = 11
SSIM_SIGMA_CELLS = 1.5
SSIM_K1 = 0.01
SSIM_K2 = 0.03


@dataclass(frozen=True)
class ModelSpec:
    """User-facing model label and resolved rollout artifact directory."""

    label: str
    rollout_root: Path


@dataclass(frozen=True)
class RolloutData:
    """Saved predictions and diagnostics for one topology."""

    topology_id: int
    trajectory_ids: np.ndarray
    flow_directions: np.ndarray
    ground_truth: np.ndarray
    prediction: np.ndarray
    relative_mass_error: np.ndarray
    outside_bounds: np.ndarray
    dt: float
    source: Path


@dataclass(frozen=True)
class TopologyMetricMetadata:
    """Finite-volume geometry needed by the fidelity metrics."""

    topology_id: int
    cell_lengths: np.ndarray
    cell_road_id: np.ndarray
    cell_road_coordinate: np.ndarray
    rho_max: float
    source: Path


@dataclass(frozen=True)
class TrajectoryMetrics:
    """Time-resolved and scalar metrics for one model trajectory."""

    topology_id: int
    trajectory_id: int
    mae_by_time: np.ndarray
    rmse_by_time: np.ndarray
    bias_by_time: np.ndarray
    relative_mass_error_by_time: np.ndarray
    outside_bounds_by_time: np.ndarray
    weighted_relative_l2_by_time: np.ndarray
    local_ssim_by_time: np.ndarray
    rollout_mae: float
    rollout_rmse: float
    final_step_mae: float
    final_step_rmse: float
    mean_signed_error: float
    relative_l2_error: float
    weighted_relative_l2_error: float
    mean_rollout_ssim: float
    final_step_ssim: float
    r_squared: float
    pearson_correlation: float
    max_relative_mass_error: float
    outside_bounds_fraction: float


def _format_axes(axis: plt.Axes) -> None:
    """Apply the plot style used by the existing rollout diagnostics."""
    axis.grid(True, alpha=0.28)
    axis.set_axisbelow(True)


def resolve_rollout_root(path: Path) -> Path:
    """Resolve a run, checkpoint, summary, or rollout directory."""
    path = path.expanduser()
    if path.is_file():
        if path.name == "summary.json":
            candidate = path.parent
        elif path.name == "checkpoint.pt":
            candidate = path.parent / "rollout_plots"
        else:
            raise ValueError(
                f"Model path {path} is a file, but it is neither "
                "summary.json nor checkpoint.pt."
            )
    elif path.is_dir():
        candidate = (
            path
            if (path / "summary.json").is_file()
            else path / "rollout_plots"
        )
    else:
        raise FileNotFoundError(f"Model path does not exist: {path}")

    if not candidate.is_dir():
        raise FileNotFoundError(
            f"Rollout directory does not exist: {candidate}. Run "
            "plot_variable_sageconv_rollouts.py for this checkpoint first."
        )
    if not any(candidate.glob("*/topology_*/rollout_predictions.npz")):
        raise FileNotFoundError(
            f"No rollout_predictions.npz files were found under {candidate}."
        )
    return candidate.resolve()


def resolve_dataset_path(raw_path: object, *, rollout_root: Path) -> Path:
    """Resolve a dataset path recorded relative to an earlier working tree."""
    path = Path(str(raw_path)).expanduser()
    candidates = [path] if path.is_absolute() else [Path.cwd() / path]
    if not path.is_absolute():
        candidates.extend(parent / path for parent in rollout_root.parents)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Could not resolve dataset directory {path!s} recorded in "
        f"{rollout_root / 'summary.json'}. Use --data to specify it."
    )


def infer_dataset_root(
    specs: Sequence[ModelSpec], *, explicit_path: Optional[Path]
) -> Path:
    """Return the common variable-topology dataset used by every model."""
    if explicit_path is not None:
        path = explicit_path.expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {path}")
        return path

    roots: Dict[str, Path] = {}
    for spec in specs:
        summary_path = spec.rollout_root / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(
                f"{summary_path} is required to infer topology metadata. "
                "Use --data to specify the dataset directory."
            )
        with summary_path.open("r", encoding="utf-8") as stream:
            summary = json.load(stream)
        if not isinstance(summary, dict) or "data" not in summary:
            raise KeyError(
                f"{summary_path} does not contain a 'data' dataset path. "
                "Use --data to specify it."
            )
        roots[spec.label] = resolve_dataset_path(
            summary["data"], rollout_root=spec.rollout_root
        )

    unique_roots = {str(path) for path in roots.values()}
    if len(unique_roots) != 1:
        details = "; ".join(
            f"{label}: {path}" for label, path in roots.items()
        )
        raise ValueError(
            "Compared models reference different datasets. Use a common "
            f"dataset or an explicit --data path ({details})."
        )
    return next(iter(roots.values()))


def load_dataset_manifest(dataset_root: Path) -> Dict[str, object]:
    """Load and validate a variable-topology dataset manifest."""
    manifest_path = dataset_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Variable-topology manifest does not exist: {manifest_path}"
        )
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict):
        raise ValueError(f"{manifest_path} must contain a JSON object.")
    if manifest.get("format") != "variable_topology_lwr_v1":
        raise ValueError(
            f"{manifest_path} is not a variable_topology_lwr_v1 dataset."
        )
    records = manifest.get("topologies")
    if not isinstance(records, list):
        raise ValueError(f"{manifest_path} has no topology record list.")
    return manifest


def topology_paths_from_manifest(dataset_root: Path) -> Dict[int, Path]:
    """Index topology archives declared by a variable-topology manifest."""
    manifest_path = dataset_root / "manifest.json"
    manifest = load_dataset_manifest(dataset_root)
    records = manifest["topologies"]
    assert isinstance(records, list)

    paths: Dict[int, Path] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"Invalid topology record in {manifest_path}.")
        topology_id = int(record["topology_id"])
        path = dataset_root / str(record["file"])
        if topology_id in paths:
            raise ValueError(
                f"Duplicate topology ID {topology_id} in {manifest_path}."
            )
        if not path.is_file():
            raise FileNotFoundError(
                f"Topology {topology_id} archive does not exist: {path}"
            )
        paths[topology_id] = path
    return paths


def resolve_ssim_reference_spacing(
    dataset_root: Path,
    *,
    explicit_spacing: Optional[float],
    topology_paths: Mapping[int, Path],
) -> Tuple[float, str]:
    """Resolve a reproducible physical scale for local SSIM windows.

    An explicit CLI value takes precedence. Structured datasets record their
    nominal cells-per-unit value, which gives a dataset-defined spacing. For
    other variable-topology datasets, the fallback is the median cell length
    over training topologies, rather than over a user-selected test subset.
    """
    if explicit_spacing is not None:
        spacing = float(explicit_spacing)
        if not np.isfinite(spacing) or spacing <= 0.0:
            raise ValueError(
                "Explicit SSIM reference spacing must be positive and finite."
            )
        return spacing, "command_line"

    manifest = load_dataset_manifest(dataset_root)
    generation_config = manifest.get("generation_config")
    if isinstance(generation_config, dict):
        cells_per_unit = generation_config.get("cells_per_unit")
        if cells_per_unit is not None:
            cells_per_unit = float(cells_per_unit)
            if not np.isfinite(cells_per_unit) or cells_per_unit <= 0.0:
                raise ValueError(
                    "manifest generation_config.cells_per_unit must be "
                    "positive and finite."
                )
            return (
                1.0 / cells_per_unit,
                "manifest_generation_config_cells_per_unit",
            )

    records = manifest["topologies"]
    assert isinstance(records, list)
    training_ids = [
        int(record["topology_id"])
        for record in records
        if isinstance(record, dict) and record.get("split") == "train"
    ]
    fallback_ids = training_ids or sorted(topology_paths)
    cell_lengths: List[np.ndarray] = []
    for topology_id in fallback_ids:
        if topology_id not in topology_paths:
            raise KeyError(
                f"Manifest has no archive path for topology {topology_id}."
            )
        path = topology_paths[topology_id]
        with np.load(path, allow_pickle=False) as archive:
            if "cell_lengths" not in archive:
                raise KeyError(f"{path} is missing cell_lengths.")
            lengths = np.asarray(
                archive["cell_lengths"], dtype=np.float64
            ).reshape(-1)
        if (
            lengths.size == 0
            or not np.all(np.isfinite(lengths))
            or np.any(lengths <= 0.0)
        ):
            raise ValueError(
                f"cell_lengths in {path} must be positive and finite."
            )
        cell_lengths.append(lengths)
    if not cell_lengths:
        raise ValueError("No topology cell lengths are available for SSIM.")
    return (
        float(np.median(np.concatenate(cell_lengths))),
        (
            "median_training_topology_cell_length"
            if training_ids
            else "median_dataset_topology_cell_length"
        ),
    )


def load_topology_metric_metadata(
    path: Path, *, expected_topology_id: int
) -> TopologyMetricMetadata:
    """Load and validate geometry used by weighted L2 and local SSIM."""
    required = (
        "cell_lengths",
        "cell_road_id",
        "cell_road_coordinate",
        "rho_max",
    )
    with np.load(path, allow_pickle=False) as archive:
        missing = [name for name in required if name not in archive]
        if missing:
            raise KeyError(
                f"{path} is missing metric metadata: {', '.join(missing)}"
            )
        topology_id = (
            int(np.asarray(archive["topology_id"]).item())
            if "topology_id" in archive
            else expected_topology_id
        )
        cell_lengths = np.asarray(
            archive["cell_lengths"], dtype=np.float64
        ).reshape(-1).copy()
        cell_road_id = np.asarray(
            archive["cell_road_id"], dtype=np.int64
        ).reshape(-1).copy()
        cell_road_coordinate = np.asarray(
            archive["cell_road_coordinate"], dtype=np.float64
        ).reshape(-1).copy()
        rho_max = float(np.asarray(archive["rho_max"]).item())

    if topology_id != expected_topology_id:
        raise ValueError(
            f"Topology ID in {path} is {topology_id}; expected "
            f"{expected_topology_id}."
        )
    node_count = cell_lengths.size
    for name, values in (
        ("cell_road_id", cell_road_id),
        ("cell_road_coordinate", cell_road_coordinate),
    ):
        if values.shape != (node_count,):
            raise ValueError(
                f"{name} in {path} must have shape ({node_count},)."
            )
    if node_count == 0:
        raise ValueError(f"{path} contains no finite-volume cells.")
    if not np.all(np.isfinite(cell_lengths)) or np.any(cell_lengths <= 0.0):
        raise ValueError(f"cell_lengths in {path} must be positive and finite.")
    if not np.all(np.isfinite(cell_road_coordinate)):
        raise ValueError(
            f"cell_road_coordinate in {path} must be finite."
        )
    if np.any(cell_road_id < 0):
        raise ValueError(f"cell_road_id in {path} must be nonnegative.")
    if not np.isfinite(rho_max) or rho_max <= 0.0:
        raise ValueError(f"rho_max in {path} must be positive and finite.")
    return TopologyMetricMetadata(
        topology_id=topology_id,
        cell_lengths=cell_lengths,
        cell_road_id=cell_road_id,
        cell_road_coordinate=cell_road_coordinate,
        rho_max=rho_max,
        source=path,
    )


def parse_model_spec(value: str) -> ModelSpec:
    """Parse ``LABEL=PATH`` into a validated model specification."""
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "--model must have the form LABEL=PATH."
        )
    label, raw_path = value.split("=", 1)
    label = label.strip()
    raw_path = raw_path.strip()
    if not label:
        raise argparse.ArgumentTypeError("Model label cannot be empty.")
    if not raw_path:
        raise argparse.ArgumentTypeError("Model path cannot be empty.")
    try:
        rollout_root = resolve_rollout_root(Path(raw_path))
    except (FileNotFoundError, ValueError) as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return ModelSpec(label=label, rollout_root=rollout_root)


def discover_rollout_files(root: Path) -> Dict[Tuple[str, int], Path]:
    """Index saved rollout files by split and topology ID."""
    discovered: Dict[Tuple[str, int], Path] = {}
    for path in sorted(root.glob("*/topology_*/rollout_predictions.npz")):
        split = path.parent.parent.name
        topology_name = path.parent.name
        try:
            prefix = "topology_"
            topology_id = int(
                topology_name[len(prefix) :]
                if topology_name.startswith(prefix)
                else topology_name
            )
        except ValueError as exc:
            raise ValueError(
                f"Could not parse topology ID from {path.parent}."
            ) from exc
        key = (split, topology_id)
        if key in discovered:
            raise ValueError(f"Duplicate rollout artifact for {key}: {path}")
        discovered[key] = path
    return discovered


def load_rollout(path: Path) -> RolloutData:
    """Load and validate one saved topology rollout."""
    with np.load(path, allow_pickle=False) as archive:
        missing = [name for name in REQUIRED_ARRAYS if name not in archive]
        if missing:
            raise KeyError(
                f"{path} is missing required array(s): {', '.join(missing)}"
            )
        topology_id = int(np.asarray(archive["topology_id"]).item())
        trajectory_ids = np.asarray(
            archive["trajectory_ids"], dtype=np.int64
        ).copy()
        truth = np.asarray(archive["ground_truth"], dtype=np.float64).copy()
        prediction = np.asarray(
            archive["prediction"], dtype=np.float64
        ).copy()
        relative_mass_error = np.asarray(
            archive["relative_mass_error_by_time"], dtype=np.float64
        ).copy()
        outside_bounds = np.asarray(
            archive["outside_bounds_by_time"], dtype=np.float64
        ).copy()
        if "trajectory_flow_direction" in archive:
            flow_directions = np.asarray(
                archive["trajectory_flow_direction"], dtype=np.float64
            ).copy()
        else:
            flow_directions = np.full(trajectory_ids.shape, np.nan)
        dt = float(np.asarray(archive["dt"]).item())

    if truth.ndim != 3:
        raise ValueError(
            f"ground_truth in {path} must have shape "
            "[trajectory, time, node]."
        )
    if prediction.shape != truth.shape:
        raise ValueError(
            f"prediction shape {prediction.shape} does not match "
            f"ground_truth shape {truth.shape} in {path}."
        )
    expected_time_shape = truth.shape[:2]
    for name, values in (
        ("relative_mass_error_by_time", relative_mass_error),
        ("outside_bounds_by_time", outside_bounds),
    ):
        if values.shape != expected_time_shape:
            raise ValueError(
                f"{name} in {path} has shape {values.shape}; expected "
                f"{expected_time_shape}."
            )
    if trajectory_ids.shape != (truth.shape[0],):
        raise ValueError(
            f"trajectory_ids in {path} must have shape ({truth.shape[0]},)."
        )
    if np.unique(trajectory_ids).size != trajectory_ids.size:
        raise ValueError(f"trajectory_ids in {path} contain duplicates.")
    if flow_directions.shape != trajectory_ids.shape:
        raise ValueError(
            f"trajectory_flow_direction in {path} must match trajectory_ids."
        )
    if truth.shape[1] < 2:
        raise ValueError(f"{path} must contain at least one rollout step.")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt in {path} must be positive and finite.")
    return RolloutData(
        topology_id=topology_id,
        trajectory_ids=trajectory_ids,
        flow_directions=flow_directions,
        ground_truth=truth,
        prediction=prediction,
        relative_mass_error=relative_mass_error,
        outside_bounds=outside_bounds,
        dt=dt,
        source=path,
    )


def trajectory_index(data: RolloutData, trajectory_id: int) -> int:
    """Return the stored row corresponding to a trajectory ID."""
    matches = np.flatnonzero(data.trajectory_ids == trajectory_id)
    if matches.size != 1:
        raise ValueError(
            f"Trajectory {trajectory_id} was not found exactly once in "
            f"{data.source}."
        )
    return int(matches[0])


def cell_length_weighted_relative_l2_by_time(
    truth: np.ndarray,
    prediction: np.ndarray,
    *,
    cell_lengths: np.ndarray,
) -> np.ndarray:
    """Compute the finite-volume relative L2 error at every saved state."""
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    lengths = np.asarray(cell_lengths, dtype=np.float64).reshape(-1)
    if truth.ndim != 2 or prediction.shape != truth.shape:
        raise ValueError("truth and prediction must have shape [time, node].")
    if lengths.shape != (truth.shape[1],):
        raise ValueError("cell_lengths must match the node dimension.")
    squared_error = (prediction - truth) ** 2
    numerator = np.sum(squared_error * lengths[None, :], axis=1)
    denominator = np.sum(truth**2 * lengths[None, :], axis=1)
    epsilon = np.finfo(np.float64).eps
    return np.sqrt(numerator / np.maximum(denominator, epsilon))


def gaussian_road_window_weights(
    coordinates: np.ndarray,
    *,
    cell_lengths: np.ndarray,
    reference_spacing: float,
    window_size: int,
    sigma_cells: float,
) -> np.ndarray:
    """Build finite-volume Gaussian quadrature weights for local SSIM."""
    coordinates = np.asarray(coordinates, dtype=np.float64).reshape(-1)
    cell_lengths = np.asarray(cell_lengths, dtype=np.float64).reshape(-1)
    if coordinates.size == 0:
        raise ValueError("A road must contain at least one cell.")
    if cell_lengths.shape != coordinates.shape:
        raise ValueError("cell_lengths must match the road coordinates.")
    if (
        not np.all(np.isfinite(cell_lengths))
        or np.any(cell_lengths <= 0.0)
    ):
        raise ValueError("cell_lengths must be positive and finite.")
    if not np.isfinite(reference_spacing) or reference_spacing <= 0.0:
        raise ValueError("reference_spacing must be positive and finite.")
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("window_size must be an odd integer of at least 3.")
    if not np.isfinite(sigma_cells) or sigma_cells <= 0.0:
        raise ValueError("sigma_cells must be positive and finite.")

    radius = (window_size // 2) * reference_spacing
    sigma = sigma_cells * reference_spacing
    distance = np.abs(coordinates[:, None] - coordinates[None, :])
    weights = (
        np.exp(-0.5 * (distance / sigma) ** 2)
        * cell_lengths[None, :]
    )
    weights[distance > radius + np.finfo(np.float64).eps] = 0.0
    row_sums = np.sum(weights, axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise RuntimeError("A Gaussian SSIM window has no samples.")
    return weights / row_sums


def roadwise_local_ssim_by_time(
    truth: np.ndarray,
    prediction: np.ndarray,
    *,
    metadata: TopologyMetricMetadata,
    reference_spacing: float,
    window_size: int,
    sigma_cells: float,
    k1: float = SSIM_K1,
    k2: float = SSIM_K2,
) -> np.ndarray:
    """Compute road-local SSIM and length-weight its spatial average.

    The Gaussian moments are evaluated in physical distance on each road
    independently.  This is the local-window SSIM construction adapted to a
    nonuniform finite-volume road mesh: no window can mix cells from different
    roads, and the fixed signal range is ``[0, rho_max]``.
    """
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if truth.ndim != 2 or prediction.shape != truth.shape:
        raise ValueError("truth and prediction must have shape [time, node].")
    if metadata.cell_lengths.shape != (truth.shape[1],):
        raise ValueError(
            f"Topology {metadata.topology_id} metadata has "
            f"{metadata.cell_lengths.size} cells but rollout has "
            f"{truth.shape[1]} nodes."
        )

    c1 = (float(k1) * metadata.rho_max) ** 2
    c2 = (float(k2) * metadata.rho_max) ** 2
    weighted_score = np.zeros(truth.shape[0], dtype=np.float64)
    total_length = float(np.sum(metadata.cell_lengths))

    for road_id in np.unique(metadata.cell_road_id):
        indices = np.flatnonzero(metadata.cell_road_id == road_id)
        order = np.argsort(
            metadata.cell_road_coordinate[indices], kind="stable"
        )
        indices = indices[order]
        coordinates = metadata.cell_road_coordinate[indices]
        if coordinates.size > 1 and np.any(np.diff(coordinates) <= 0.0):
            raise ValueError(
                f"Road {int(road_id)} coordinates in {metadata.source} "
                "must be strictly increasing."
            )
        windows = gaussian_road_window_weights(
            coordinates,
            cell_lengths=metadata.cell_lengths[indices],
            reference_spacing=reference_spacing,
            window_size=window_size,
            sigma_cells=sigma_cells,
        )
        true_road = truth[:, indices]
        predicted_road = prediction[:, indices]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            true_mean = true_road @ windows.T
            predicted_mean = predicted_road @ windows.T
            true_variance = (
                true_road**2 @ windows.T - true_mean**2
            )
            predicted_variance = (
                predicted_road**2 @ windows.T - predicted_mean**2
            )
            covariance = (
                (true_road * predicted_road) @ windows.T
                - true_mean * predicted_mean
            )
            true_variance = np.maximum(true_variance, 0.0)
            predicted_variance = np.maximum(predicted_variance, 0.0)
            numerator = (
                (2.0 * true_mean * predicted_mean + c1)
                * (2.0 * covariance + c2)
            )
            denominator = (
                (true_mean**2 + predicted_mean**2 + c1)
                * (true_variance + predicted_variance + c2)
            )
            local_ssim = numerator / denominator
        local_ssim = np.clip(local_ssim, -1.0, 1.0)
        weighted_score += np.sum(
            local_ssim * metadata.cell_lengths[indices][None, :],
            axis=1,
        )
    return weighted_score / total_length


def compute_metrics(
    data: RolloutData,
    trajectory_id: int,
    *,
    num_times: int,
    metadata: TopologyMetricMetadata,
    ssim_reference_spacing: float,
    ssim_window_size: int,
    ssim_sigma_cells: float,
) -> TrajectoryMetrics:
    """Compute diagnostics over a common comparison interval."""
    index = trajectory_index(data, trajectory_id)
    truth = data.ground_truth[index, :num_times]
    prediction = data.prediction[index, :num_times]
    error = prediction - truth
    future_error = error[1:]
    future_truth = truth[1:]
    error_squared_sum = float(np.sum(future_error**2))
    epsilon = np.finfo(np.float64).eps
    truth_squared_sum = max(float(np.sum(future_truth**2)), epsilon)
    truth_centered = future_truth - float(np.mean(future_truth))
    truth_variation = max(float(np.sum(truth_centered**2)), epsilon)
    predicted_values = prediction[1:].reshape(-1)
    true_values = future_truth.reshape(-1)
    correlation = (
        float(np.corrcoef(true_values, predicted_values)[0, 1])
        if (
            float(np.std(true_values)) > epsilon
            and float(np.std(predicted_values)) > epsilon
        )
        else float("nan")
    )
    mae_by_time = np.mean(np.abs(error), axis=1)
    rmse_by_time = np.sqrt(np.mean(error**2, axis=1))
    weighted_relative_l2_by_time = (
        cell_length_weighted_relative_l2_by_time(
            truth,
            prediction,
            cell_lengths=metadata.cell_lengths,
        )
    )
    local_ssim_by_time = roadwise_local_ssim_by_time(
        truth,
        prediction,
        metadata=metadata,
        reference_spacing=ssim_reference_spacing,
        window_size=ssim_window_size,
        sigma_cells=ssim_sigma_cells,
    )
    weighted_future_squared_error = float(
        np.sum(future_error**2 * metadata.cell_lengths[None, :])
    )
    weighted_future_truth_squared = max(
        float(np.sum(future_truth**2 * metadata.cell_lengths[None, :])),
        epsilon,
    )
    relative_mass_error = data.relative_mass_error[index, :num_times]
    outside_bounds = data.outside_bounds[index, :num_times]
    return TrajectoryMetrics(
        topology_id=data.topology_id,
        trajectory_id=trajectory_id,
        mae_by_time=mae_by_time,
        rmse_by_time=rmse_by_time,
        bias_by_time=np.mean(error, axis=1),
        relative_mass_error_by_time=relative_mass_error,
        outside_bounds_by_time=outside_bounds,
        weighted_relative_l2_by_time=weighted_relative_l2_by_time,
        local_ssim_by_time=local_ssim_by_time,
        rollout_mae=float(np.mean(np.abs(future_error))),
        rollout_rmse=float(np.sqrt(np.mean(future_error**2))),
        final_step_mae=float(mae_by_time[-1]),
        final_step_rmse=float(rmse_by_time[-1]),
        mean_signed_error=float(np.mean(future_error)),
        relative_l2_error=float(
            np.sqrt(error_squared_sum / truth_squared_sum)
        ),
        weighted_relative_l2_error=float(
            np.sqrt(
                weighted_future_squared_error
                / weighted_future_truth_squared
            )
        ),
        mean_rollout_ssim=float(np.mean(local_ssim_by_time[1:])),
        final_step_ssim=float(local_ssim_by_time[-1]),
        r_squared=float(1.0 - error_squared_sum / truth_variation),
        pearson_correlation=correlation,
        max_relative_mass_error=float(
            np.max(np.abs(relative_mass_error))
        ),
        outside_bounds_fraction=float(np.mean(outside_bounds)),
    )


def model_styles(labels: Sequence[str]) -> Dict[str, Dict[str, object]]:
    """Assign a stable color and line style to every model."""
    color_map = plt.get_cmap("tab10")
    line_styles = ("-", "--", "-.", ":")
    return {
        label: {
            "color": color_map(index % 10),
            "linestyle": line_styles[(index // 10) % len(line_styles)],
            "linewidth": 1.8,
        }
        for index, label in enumerate(labels)
    }


def plot_trajectory_diagnostics(
    *,
    split: str,
    topology_id: int,
    trajectory_id: int,
    flow_direction: float,
    metrics_by_model: Mapping[str, TrajectoryMetrics],
    styles: Mapping[str, Mapping[str, object]],
    output_path: Path,
    dpi: int,
) -> None:
    """Plot overlaid diagnostics for one trajectory."""
    first_metrics = next(iter(metrics_by_model.values()))
    steps = np.arange(first_metrics.rmse_by_time.size)
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(15.5, 8.6),
        constrained_layout=True,
    )
    ax_mae, ax_rmse, ax_bias, ax_mass, ax_bounds, ax_scores = axes.ravel()

    panels = (
        (ax_mae, "mae_by_time", "Mean absolute error", "Node MAE"),
        (
            ax_rmse,
            "rmse_by_time",
            "Root mean squared error",
            "Node RMSE",
        ),
        (
            ax_bias,
            "bias_by_time",
            "Signed prediction bias",
            r"Mean $\hat\rho-\rho$",
        ),
        (
            ax_mass,
            "relative_mass_error_by_time",
            "Relative mass error",
            r"$(\hat M-M)/|M_0|$",
        ),
        (
            ax_bounds,
            "outside_bounds_by_time",
            "Physical-bound violations",
            "Fraction of nodes",
        ),
    )
    for axis, attribute, title, ylabel in panels:
        for label, metrics in metrics_by_model.items():
            axis.plot(
                steps,
                getattr(metrics, attribute),
                label=label,
                **styles[label],
            )
        axis.set_title(title)
        axis.set_xlabel("Rollout step")
        axis.set_ylabel(ylabel)
        axis.legend(fontsize=8)
        _format_axes(axis)
    ax_bias.axhline(0.0, color="0.25", linestyle="--", linewidth=1.0)
    ax_mass.axhline(0.0, color="0.25", linestyle="--", linewidth=1.0)

    maximum_bounds = max(
        float(np.max(metrics.outside_bounds_by_time))
        for metrics in metrics_by_model.values()
    )
    ax_bounds.set_ylim(-0.01, max(0.05, 1.05 * maximum_bounds))

    ax_scores.axis("off")
    ax_scores.set_title("Scalar scores")
    columns = (
        "Rollout\nRMSE",
        "Final\nRMSE",
        "Weighted\nrel. $L_2$",
        "Final\nSSIM",
        "Max rel.\nmass err.",
        "Outside\nbounds",
    )
    cell_text = []
    labels = list(metrics_by_model)
    for label in labels:
        metrics = metrics_by_model[label]
        cell_text.append(
            [
                f"{metrics.rollout_rmse:.4g}",
                f"{metrics.final_step_rmse:.4g}",
                f"{metrics.weighted_relative_l2_error:.4g}",
                f"{metrics.final_step_ssim:.4g}",
                f"{metrics.max_relative_mass_error:.2e}",
                f"{metrics.outside_bounds_fraction:.2%}",
            ]
        )
    table = ax_scores.table(
        cellText=cell_text,
        rowLabels=labels,
        colLabels=columns,
        cellLoc="center",
        rowLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.45)
    for row_index, label in enumerate(labels, start=1):
        table[(row_index, -1)].get_text().set_color(
            styles[label]["color"]
        )

    readable_split = split.replace("_", " ")
    direction_suffix = ""
    if np.isfinite(flow_direction):
        direction_suffix = (
            ", canonical direction"
            if flow_direction >= 0.0
            else ", reversed direction"
        )
    figure.suptitle(
        "Model-comparison diagnostics: "
        f"{readable_split}, topology {topology_id:04d}, "
        f"trajectory {trajectory_id:04d}{direction_suffix}",
        fontsize=14,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def stack_metric(
    metrics: Sequence[TrajectoryMetrics], attribute: str
) -> np.ndarray:
    """Stack one time-resolved metric over trajectories."""
    return np.stack(
        [np.asarray(getattr(item, attribute)) for item in metrics], axis=0
    )


def group_metrics_by_topology(
    metrics: Sequence[TrajectoryMetrics],
) -> Dict[int, List[TrajectoryMetrics]]:
    """Group trajectory metrics without weighting large topology groups more."""
    grouped: Dict[int, List[TrajectoryMetrics]] = {}
    for item in metrics:
        grouped.setdefault(item.topology_id, []).append(item)
    if not grouped:
        raise ValueError("At least one trajectory metric is required.")
    return grouped


def topology_balanced_mean_curve(
    metrics: Sequence[TrajectoryMetrics], attribute: str
) -> np.ndarray:
    """Average trajectories within topology, then average topologies equally."""
    grouped = group_metrics_by_topology(metrics)
    topology_means = [
        np.mean(stack_metric(values, attribute), axis=0)
        for values in grouped.values()
    ]
    return np.mean(np.stack(topology_means, axis=0), axis=0)


def topology_balanced_mean_scalar(
    metrics: Sequence[TrajectoryMetrics], attribute: str
) -> float:
    """Average a scalar over trajectories, then equally over topologies."""
    grouped = group_metrics_by_topology(metrics)
    topology_means = [
        float(np.mean([float(getattr(item, attribute)) for item in values]))
        for values in grouped.values()
    ]
    return float(np.mean(topology_means))


def plot_split_summary(
    *,
    split: str,
    metrics_by_model: Mapping[str, Sequence[TrajectoryMetrics]],
    styles: Mapping[str, Mapping[str, object]],
    output_path: Path,
    dpi: int,
    show_iqr: bool,
) -> None:
    """Plot aggregate overlaid diagnostics for a dataset split."""
    first_metrics = next(iter(metrics_by_model.values()))[0]
    steps = np.arange(first_metrics.rmse_by_time.size)
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(15.5, 8.6),
        constrained_layout=True,
    )
    ax_rmse, ax_mae, ax_bias, ax_mass, ax_bounds, ax_ecdf = axes.ravel()
    panels = (
        (ax_rmse, "rmse_by_time", "Median node RMSE", "Node RMSE"),
        (ax_mae, "mae_by_time", "Median node MAE", "Node MAE"),
        (
            ax_bias,
            "bias_by_time",
            "Median absolute prediction bias",
            r"$|\mathrm{mean}(\hat\rho-\rho)|$",
        ),
        (
            ax_mass,
            "relative_mass_error_by_time",
            "Median absolute relative mass error",
            r"$|(\hat M-M)/M_0|$",
        ),
        (
            ax_bounds,
            "outside_bounds_by_time",
            "Median physical-bound violations",
            "Fraction of nodes",
        ),
    )
    absolute_attributes = {
        "bias_by_time",
        "relative_mass_error_by_time",
    }
    for axis, attribute, title, ylabel in panels:
        for label, all_metrics in metrics_by_model.items():
            values = stack_metric(all_metrics, attribute)
            if attribute in absolute_attributes:
                values = np.abs(values)
            median = np.median(values, axis=0)
            axis.plot(steps, median, label=label, **styles[label])
            if show_iqr:
                lower, upper = np.quantile(values, (0.25, 0.75), axis=0)
                axis.fill_between(
                    steps,
                    lower,
                    upper,
                    color=styles[label]["color"],
                    alpha=0.10,
                    linewidth=0.0,
                )
        axis.set_title(title)
        axis.set_xlabel("Rollout step")
        axis.set_ylabel(ylabel)
        axis.legend(fontsize=8)
        _format_axes(axis)

    for label, all_metrics in metrics_by_model.items():
        values = np.sort(
            np.asarray([item.rollout_rmse for item in all_metrics])
        )
        cumulative = np.arange(1, values.size + 1) / values.size
        ax_ecdf.step(
            values,
            cumulative,
            where="post",
            label=label,
            **styles[label],
        )
    ax_ecdf.set_title("Trajectory rollout-RMSE distribution")
    ax_ecdf.set_xlabel("Rollout RMSE")
    ax_ecdf.set_ylabel("Empirical cumulative fraction")
    ax_ecdf.set_ylim(0.0, 1.02)
    ax_ecdf.legend(fontsize=8)
    _format_axes(ax_ecdf)

    figure.suptitle(
        f"Model comparison: {split.replace('_', ' ')}", fontsize=14
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def plot_fidelity_summary(
    *,
    split: str,
    metrics_by_model: Mapping[str, Sequence[TrajectoryMetrics]],
    styles: Mapping[str, Mapping[str, object]],
    output_path: Path,
    dpi: int,
) -> None:
    """Plot topology-balanced fidelity curves versus rollout step."""
    first_metrics = next(iter(metrics_by_model.values()))[0]
    steps = np.arange(first_metrics.weighted_relative_l2_by_time.size)
    figure, (ax_l2, ax_ssim) = plt.subplots(
        1,
        2,
        figsize=(12.4, 4.6),
    )
    figure.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.16,
        top=0.78,
        wspace=0.24,
    )

    mean_ssim_curves: List[np.ndarray] = []
    for label, all_metrics in metrics_by_model.items():
        mean_relative_l2 = topology_balanced_mean_curve(
            all_metrics, "weighted_relative_l2_by_time"
        )
        ax_l2.plot(
            steps[1:],
            mean_relative_l2[1:],
            label=label,
            **styles[label],
        )
        mean_ssim = topology_balanced_mean_curve(
            all_metrics, "local_ssim_by_time"
        )[1:]
        mean_ssim_curves.append(mean_ssim)
        ax_ssim.plot(
            steps[1:],
            mean_ssim,
            label=label,
            **styles[label],
        )

    ax_l2.set_title(r"Cell-length-weighted relative $L_2$ error")
    ax_l2.set_xlabel("Rollout step")
    ax_l2.set_ylabel(r"Relative $L_2$")
    ax_l2.legend(fontsize=8)
    _format_axes(ax_l2)

    ax_ssim.set_title("Mean roadwise local spatial SSIM")
    ax_ssim.set_xlabel("Rollout step")
    ax_ssim.set_ylabel("SSIM")
    ax_ssim.legend(fontsize=8)
    _format_axes(ax_ssim)
    all_ssim = np.concatenate(mean_ssim_curves)
    if np.all(np.isfinite(all_ssim)):
        lower = max(-1.0, float(np.min(all_ssim)) - 0.03)
        ax_ssim.set_ylim(lower, 1.005)

    figure.suptitle(
        f"Traffic rollout fidelity: {split.replace('_', ' ')}",
        fontsize=13,
        y=0.96,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def validate_common_rollouts(
    rollouts: Mapping[str, RolloutData],
    *,
    rollout_steps: Optional[int],
) -> int:
    """Validate shared truth and return the number of compared time states."""
    reference_label = next(iter(rollouts))
    reference = rollouts[reference_label]
    available_times = min(
        data.ground_truth.shape[1] for data in rollouts.values()
    )
    if rollout_steps is None:
        num_times = available_times
    else:
        num_times = rollout_steps + 1
        if num_times > available_times:
            details = ", ".join(
                f"{label}={data.ground_truth.shape[1] - 1}"
                for label, data in rollouts.items()
            )
            raise ValueError(
                f"--rollout-steps={rollout_steps} exceeds an available "
                f"rollout ({details})."
            )
    for label, data in rollouts.items():
        if data.topology_id != reference.topology_id:
            raise ValueError("Compared rollout files have different topology IDs.")
        if not np.isclose(data.dt, reference.dt, rtol=1.0e-8, atol=1.0e-12):
            raise ValueError(
                f"Time step differs for {reference_label} and {label} on "
                f"topology {data.topology_id}."
            )
        if data.ground_truth.shape[2] != reference.ground_truth.shape[2]:
            raise ValueError(
                f"Node count differs for {reference_label} and {label} on "
                f"topology {data.topology_id}."
            )
    return num_times


def common_values(
    values_by_label: Mapping[str, Iterable[int]],
    *,
    requested: Optional[Sequence[int]],
    description: str,
) -> List[int]:
    """Select requested IDs or the intersection shared by every model."""
    sets = {
        label: set(int(value) for value in values)
        for label, values in values_by_label.items()
    }
    common = set.intersection(*sets.values())
    if requested is not None:
        selected = list(dict.fromkeys(int(value) for value in requested))
        missing = {
            label: [value for value in selected if value not in available]
            for label, available in sets.items()
        }
        missing = {label: values for label, values in missing.items() if values}
        if missing:
            details = "; ".join(
                f"{label}: {values}" for label, values in missing.items()
            )
            raise ValueError(
                f"Requested {description} are not available in every model "
                f"({details})."
            )
        return selected
    if not common:
        raise ValueError(f"Models share no {description}.")
    if any(available != common for available in sets.values()):
        print(
            f"warning: comparing only common {description}: "
            + ", ".join(str(value) for value in sorted(common))
        )
    return sorted(common)


def write_metrics_csv(
    path: Path, rows: Sequence[Mapping[str, object]]
) -> None:
    """Write one scalar-metric row per model and trajectory."""
    if not rows:
        raise ValueError("No comparison metric rows were generated.")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def add_boolean_optional_argument(
    parser: argparse.ArgumentParser,
    name: str,
    *,
    default: bool,
    help_text: str,
) -> None:
    """Add portable ``--NAME`` and ``--no-NAME`` Boolean flags."""
    destination = name.replace("-", "_")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        f"--{name}",
        dest=destination,
        action="store_true",
        help=help_text,
    )
    group.add_argument(
        f"--no-{name}",
        dest=destination,
        action="store_false",
        help=f"Do not {help_text[0].lower() + help_text[1:]}",
    )
    parser.set_defaults(**{destination: default})


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Overlay rollout diagnostics from two or more trained-model "
            "rollout directories."
        )
    )
    parser.add_argument(
        "--model",
        action="append",
        type=parse_model_spec,
        required=True,
        metavar="LABEL=PATH",
        help=(
            "Model label and run/rollout path. Repeat once per model."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory in which comparison figures and metrics are saved.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help=(
            "Variable-topology dataset directory. By default it is inferred "
            "from every rollout_plots/summary.json file."
        ),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Splits to compare. Defaults to those shared by every model.",
    )
    parser.add_argument(
        "--topologies",
        nargs="+",
        type=int,
        default=None,
        help="Optional topology IDs to compare within each selected split.",
    )
    parser.add_argument(
        "--trajectories",
        nargs="+",
        type=int,
        default=None,
        help="Optional trajectory IDs; defaults to all shared trajectories.",
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=None,
        help=(
            "Number of autoregressive steps to compare. Defaults to the "
            "shortest saved rollout."
        ),
    )
    add_boolean_optional_argument(
        parser,
        "trajectory-plots",
        default=True,
        help_text="Write overlaid diagnostics for each selected trajectory.",
    )
    add_boolean_optional_argument(
        parser,
        "iqr",
        default=True,
        help_text="Show interquartile bands in split-summary figures.",
    )
    parser.add_argument(
        "--ssim-window-size",
        type=int,
        default=SSIM_WINDOW_SIZE,
        help=(
            "Odd nominal Gaussian-window width in reference mesh cells for "
            "roadwise local SSIM (default: 11)."
        ),
    )
    parser.add_argument(
        "--ssim-sigma-cells",
        type=float,
        default=SSIM_SIGMA_CELLS,
        help=(
            "Gaussian SSIM sigma in reference mesh-cell units "
            "(default: 1.5)."
        ),
    )
    parser.add_argument(
        "--ssim-reference-spacing",
        type=float,
        default=None,
        help=(
            "Physical mesh spacing used to scale the SSIM window. Defaults "
            "to 1/cells_per_unit when recorded by the dataset; otherwise "
            "uses the median training-topology cell length. Set this "
            "explicitly for a frozen paper metric."
        ),
    )
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Generate all requested model-comparison outputs."""
    args = parse_args(argv)
    specs: List[ModelSpec] = args.model
    if len(specs) < 2:
        raise ValueError("Specify at least two --model arguments.")
    labels = [spec.label for spec in specs]
    if len(set(labels)) != len(labels):
        raise ValueError("Every --model label must be unique.")
    if args.rollout_steps is not None and args.rollout_steps <= 0:
        raise ValueError("--rollout-steps must be positive.")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive.")
    if args.ssim_window_size < 3 or args.ssim_window_size % 2 == 0:
        raise ValueError("--ssim-window-size must be an odd integer >= 3.")
    if (
        not np.isfinite(args.ssim_sigma_cells)
        or args.ssim_sigma_cells <= 0.0
    ):
        raise ValueError("--ssim-sigma-cells must be positive and finite.")
    if args.ssim_reference_spacing is not None and (
        not np.isfinite(args.ssim_reference_spacing)
        or args.ssim_reference_spacing <= 0.0
    ):
        raise ValueError(
            "--ssim-reference-spacing must be positive and finite."
        )

    files_by_model = {
        spec.label: discover_rollout_files(spec.rollout_root)
        for spec in specs
    }
    split_sets = {
        label: {split for split, _ in files}
        for label, files in files_by_model.items()
    }
    common_splits = set.intersection(*split_sets.values())
    if args.splits is None:
        selected_splits = sorted(common_splits)
        if any(values != common_splits for values in split_sets.values()):
            print(
                "warning: comparing only splits shared by every model: "
                + ", ".join(selected_splits)
            )
    else:
        selected_splits = list(dict.fromkeys(args.splits))
        missing_splits = {
            label: [
                split for split in selected_splits if split not in available
            ]
            for label, available in split_sets.items()
        }
        missing_splits = {
            label: values
            for label, values in missing_splits.items()
            if values
        }
        if missing_splits:
            details = "; ".join(
                f"{label}: {values}"
                for label, values in missing_splits.items()
            )
            raise ValueError(
                "Requested splits are not available in every model "
                f"({details})."
            )
    if not selected_splits:
        raise ValueError("The models have no rollout splits in common.")

    topology_ids_by_split: Dict[str, List[int]] = {}
    for split in selected_splits:
        topologies_by_label = {
            label: [
                topology_id
                for candidate_split, topology_id in files
                if candidate_split == split
            ]
            for label, files in files_by_model.items()
        }
        topology_ids_by_split[split] = common_values(
            topologies_by_label,
            requested=args.topologies,
            description=f"topologies for {split}",
        )

    dataset_root = infer_dataset_root(specs, explicit_path=args.data)
    topology_paths = topology_paths_from_manifest(dataset_root)
    selected_topology_ids = sorted(
        {
            topology_id
            for values in topology_ids_by_split.values()
            for topology_id in values
        }
    )
    missing_metadata = [
        topology_id
        for topology_id in selected_topology_ids
        if topology_id not in topology_paths
    ]
    if missing_metadata:
        raise KeyError(
            f"Dataset {dataset_root} has no metadata for topology IDs "
            f"{missing_metadata}."
        )
    metadata_by_topology = {
        topology_id: load_topology_metric_metadata(
            topology_paths[topology_id],
            expected_topology_id=topology_id,
        )
        for topology_id in selected_topology_ids
    }
    (
        ssim_reference_spacing,
        ssim_reference_spacing_source,
    ) = resolve_ssim_reference_spacing(
        dataset_root,
        explicit_spacing=args.ssim_reference_spacing,
        topology_paths=topology_paths,
    )
    ssim_radius = (
        args.ssim_window_size // 2
    ) * ssim_reference_spacing
    ssim_sigma = args.ssim_sigma_cells * ssim_reference_spacing
    print(f"metric dataset: {dataset_root}")
    print(
        "roadwise local SSIM: "
        f"data_range=rho_max, window_size={args.ssim_window_size}, "
        f"reference_spacing={ssim_reference_spacing:.6g}, "
        f"reference_spacing_source={ssim_reference_spacing_source}, "
        f"physical_radius={ssim_radius:.6g}, "
        f"physical_sigma={ssim_sigma:.6g}"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    styles = model_styles(labels)
    rows: List[Dict[str, object]] = []
    json_splits: Dict[str, object] = {}

    for split in selected_splits:
        topology_ids = topology_ids_by_split[split]
        split_metrics: Dict[str, List[TrajectoryMetrics]] = {
            label: [] for label in labels
        }
        compared_steps: List[int] = []
        dt_by_topology: Dict[str, float] = {}
        physical_horizon_by_topology: Dict[str, float] = {}
        compared_trajectories = 0

        for topology_id in topology_ids:
            rollouts = {
                label: load_rollout(
                    files_by_model[label][(split, topology_id)]
                )
                for label in labels
            }
            num_times = validate_common_rollouts(
                rollouts, rollout_steps=args.rollout_steps
            )
            compared_steps.append(num_times - 1)
            reference = rollouts[labels[0]]
            topology_key = str(topology_id)
            dt_by_topology[topology_key] = reference.dt
            physical_horizon_by_topology[topology_key] = (
                (num_times - 1) * reference.dt
            )
            trajectory_ids = common_values(
                {
                    label: data.trajectory_ids
                    for label, data in rollouts.items()
                },
                requested=args.trajectories,
                description=(
                    f"trajectories for {split}/topology_{topology_id:04d}"
                ),
            )
            for trajectory_id in trajectory_ids:
                reference_index = trajectory_index(reference, trajectory_id)
                reference_truth = reference.ground_truth[
                    reference_index, :num_times
                ]
                metrics_by_model: Dict[str, TrajectoryMetrics] = {}
                for label in labels:
                    data = rollouts[label]
                    index = trajectory_index(data, trajectory_id)
                    truth = data.ground_truth[index, :num_times]
                    if not np.allclose(
                        truth,
                        reference_truth,
                        rtol=1.0e-6,
                        atol=1.0e-7,
                    ):
                        maximum_difference = float(
                            np.max(np.abs(truth - reference_truth))
                        )
                        raise ValueError(
                            "Ground truth differs between "
                            f"{labels[0]} and {label} for {split}, "
                            f"topology {topology_id}, trajectory "
                            f"{trajectory_id} (max difference "
                            f"{maximum_difference:.3e})."
                        )
                    metrics = compute_metrics(
                        data,
                        trajectory_id,
                        num_times=num_times,
                        metadata=metadata_by_topology[topology_id],
                        ssim_reference_spacing=ssim_reference_spacing,
                        ssim_window_size=args.ssim_window_size,
                        ssim_sigma_cells=args.ssim_sigma_cells,
                    )
                    metrics_by_model[label] = metrics
                    split_metrics[label].append(metrics)
                    rows.append(
                        {
                            "model_label": label,
                            "split": split,
                            "topology_id": topology_id,
                            "trajectory_id": trajectory_id,
                            "rollout_steps": num_times - 1,
                            "rollout_mae": metrics.rollout_mae,
                            "rollout_rmse": metrics.rollout_rmse,
                            "final_step_mae": metrics.final_step_mae,
                            "final_step_rmse": metrics.final_step_rmse,
                            "mean_signed_error": metrics.mean_signed_error,
                            "relative_l2_error": metrics.relative_l2_error,
                            "weighted_relative_l2_error": (
                                metrics.weighted_relative_l2_error
                            ),
                            "mean_rollout_ssim": metrics.mean_rollout_ssim,
                            "final_step_ssim": metrics.final_step_ssim,
                            "r_squared": metrics.r_squared,
                            "pearson_correlation": (
                                metrics.pearson_correlation
                            ),
                            "max_relative_mass_error": (
                                metrics.max_relative_mass_error
                            ),
                            "outside_bounds_fraction": (
                                metrics.outside_bounds_fraction
                            ),
                        }
                    )
                if args.trajectory_plots:
                    direction = reference.flow_directions[reference_index]
                    output_path = (
                        args.output_dir
                        / split
                        / f"topology_{topology_id:04d}"
                        / (
                            f"trajectory_{trajectory_id:04d}"
                            "_comparison_diagnostics.png"
                        )
                    )
                    plot_trajectory_diagnostics(
                        split=split,
                        topology_id=topology_id,
                        trajectory_id=trajectory_id,
                        flow_direction=float(direction),
                        metrics_by_model=metrics_by_model,
                        styles=styles,
                        output_path=output_path,
                        dpi=args.dpi,
                    )
                compared_trajectories += 1

        summary_path = args.output_dir / split / "comparison_summary.png"
        plot_split_summary(
            split=split,
            metrics_by_model=split_metrics,
            styles=styles,
            output_path=summary_path,
            dpi=args.dpi,
            show_iqr=args.iqr,
        )
        fidelity_path = (
            args.output_dir
            / split
            / "relative_l2_ssim_comparison.png"
        )
        plot_fidelity_summary(
            split=split,
            metrics_by_model=split_metrics,
            styles=styles,
            output_path=fidelity_path,
            dpi=args.dpi,
        )
        model_summary = {}
        for label, all_metrics in split_metrics.items():
            model_summary[label] = {
                "mean_rollout_mae": topology_balanced_mean_scalar(
                    all_metrics, "rollout_mae"
                ),
                "mean_rollout_rmse": topology_balanced_mean_scalar(
                    all_metrics, "rollout_rmse"
                ),
                "median_rollout_rmse": float(
                    np.median([item.rollout_rmse for item in all_metrics])
                ),
                "mean_final_step_rmse": topology_balanced_mean_scalar(
                    all_metrics, "final_step_rmse"
                ),
                "mean_relative_l2_error": topology_balanced_mean_scalar(
                    all_metrics, "relative_l2_error"
                ),
                "mean_weighted_relative_l2_error": (
                    topology_balanced_mean_scalar(
                        all_metrics, "weighted_relative_l2_error"
                    )
                ),
                "mean_rollout_ssim": topology_balanced_mean_scalar(
                    all_metrics, "mean_rollout_ssim"
                ),
                "mean_final_step_ssim": topology_balanced_mean_scalar(
                    all_metrics, "final_step_ssim"
                ),
                "mean_max_relative_mass_error": (
                    topology_balanced_mean_scalar(
                        all_metrics, "max_relative_mass_error"
                    )
                ),
                "mean_outside_bounds_fraction": (
                    topology_balanced_mean_scalar(
                        all_metrics, "outside_bounds_fraction"
                    )
                ),
            }
        json_splits[split] = {
            "topology_ids": topology_ids,
            "num_trajectories": compared_trajectories,
            "rollout_steps_by_topology": compared_steps,
            "dt_by_topology": dt_by_topology,
            "physical_horizon_by_topology": (
                physical_horizon_by_topology
            ),
            "models": model_summary,
        }
        print(
            f"[{split}] wrote {summary_path} using "
            f"{len(topology_ids)} topologies and "
            f"{compared_trajectories} trajectories"
        )
        print(f"[{split}] wrote {fidelity_path}")

    csv_path = args.output_dir / "comparison_metrics.csv"
    write_metrics_csv(csv_path, rows)
    summary = {
        "models": {
            spec.label: str(spec.rollout_root) for spec in specs
        },
        "dataset": str(dataset_root),
        "fidelity_metrics": {
            "relative_l2": "cell_length_weighted",
            "split_aggregation": {
                "trajectory_aggregation": "equal_within_topology",
                "topology_aggregation": "equal_across_topologies",
            },
            "horizontal_axis": "rollout_step",
            "physical_time_alignment": "not_applied",
            "ssim": {
                "type": "roadwise_local_gaussian",
                "data_range": "rho_max",
                "window_size_reference_cells": args.ssim_window_size,
                "sigma_reference_cells": args.ssim_sigma_cells,
                "reference_spacing": ssim_reference_spacing,
                "reference_spacing_source": (
                    ssim_reference_spacing_source
                ),
                "physical_window_radius": ssim_radius,
                "physical_sigma": ssim_sigma,
                "local_moment_weights": (
                    "gaussian_physical_distance_times_neighbor_cell_length"
                ),
                "spatial_aggregation": "center_cell_length_weighted",
                "junction_handling": "windows_do_not_cross_roads",
                "endpoint_handling": (
                    "truncate_at_road_endpoint_and_renormalize"
                ),
                "covariance": "population",
                "k1": SSIM_K1,
                "k2": SSIM_K2,
                "time_evaluation": "independent_at_each_rollout_step",
                "trajectory_aggregation": "equal_within_topology",
                "topology_aggregation": "equal_across_topologies",
            },
        },
        "splits": json_splits,
    }
    json_path = args.output_dir / "comparison_summary.json"
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(f"wrote metrics: {csv_path}")
    print(f"wrote summary: {json_path}")


if __name__ == "__main__":
    main()
