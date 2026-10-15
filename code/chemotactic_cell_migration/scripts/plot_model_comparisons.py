#!/usr/bin/env python3
"""Compare held-out chemotaxis rollouts from two or more trained GNNs.

The script consumes the ``test_rollout_predictions.npz`` files written by
``plot_gnn_rollouts.py``.  It does not rerun inference, so rollout artifacts
must already exist for every requested model.  Models are compared only on
the trajectory IDs and rollout interval shared by every artifact, and their
stored ground truth is checked before paired metrics are calculated.

Each ``--model`` argument has the form ``LABEL=PATH``.  ``PATH`` may point to
a run directory, its ``rollout_plots`` directory, ``checkpoint.pt``,
``summary_metrics.json``, or ``test_rollout_predictions.npz``.  For example:

    python scripts/plot_model_comparisons.py \
        --model "SAGEConv=runs/sageconv/predict_delta/100ep" \
        --model "MeshGraphNet=runs/meshgraphnet/predict_delta/200ep" \
        --model "FluxGraphNet=runs/fluxgraphnet/predict_delta/200ep" \
        --output-dir runs/model_comparison

Outputs include aggregate rollout and scalar-score figures, an optional
training-history comparison, optional per-trajectory diagnostics, a CSV with
one row per model/trajectory pair, and a machine-readable JSON summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

_mpl_cache = Path(tempfile.gettempdir()) / "chemotaxis-matplotlib-cache"
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))
os.environ.setdefault("XDG_CACHE_HOME", str(_mpl_cache))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from chemotaxis_archive import load_archive_fields


REQUIRED_ARRAYS = (
    "trajectory_ids",
    "truth",
    "prediction",
    "time",
    "mass_error",
)

SSIM_MODE_OFF = "off"
SSIM_MODE_LOCAL_MESH = "local-mesh"
SSIM_MODES = (SSIM_MODE_OFF, SSIM_MODE_LOCAL_MESH)
SSIM_WINDOW_SIZE = 11
SSIM_SIGMA_CELLS = 1.5
SSIM_K1 = 0.01
SSIM_K2 = 0.03

SOURCE_DATASET_FIELDS = (
    "cell_centers",
    "cell_areas",
    "undirected_edge_index",
    "shared_face_lengths",
    "shared_face_midpoints",
    "shared_face_normals",
    "center_normal_distances",
    "face_drift_speed",
    "drift_velocity",
    "chemo_source_count",
    "chemo_source_centers",
    "chemo_source_amplitudes",
    "chemo_source_sigmas",
    "trajectory_chi",
    "chi",
    "diffusion",
    "dt",
    "CFL",
    "boundary",
    "reference_num_cells",
    "density_physical_range",
    "density_physical_min",
    "density_physical_max",
)


@dataclass(frozen=True)
class ModelSpec:
    """A user label and the artifacts resolved from its supplied path."""

    label: str
    prediction_path: Path
    run_root: Path
    history_path: Optional[Path]


@dataclass(frozen=True)
class RolloutData:
    """Saved held-out rollouts for one trained model."""

    trajectory_ids: np.ndarray
    truth: np.ndarray
    prediction: np.ndarray
    persistence: np.ndarray
    time: np.ndarray
    mass_error: np.ndarray
    source: Path
    inference_timing: Optional[Dict[str, object]] = None


@dataclass(frozen=True)
class TrajectoryMetrics:
    """Time-resolved and scalar diagnostics for one trajectory."""

    mae_by_time: np.ndarray
    rmse_by_time: np.ndarray
    bias_by_time: np.ndarray
    mass_error_by_time: np.ndarray
    negative_fraction_by_time: np.ndarray
    persistence_mae_by_time: np.ndarray
    persistence_rmse_by_time: np.ndarray
    area_weighted_relative_l2_by_time: Optional[np.ndarray]
    persistence_area_weighted_relative_l2_by_time: Optional[np.ndarray]
    local_ssim_by_time: Optional[np.ndarray]
    persistence_local_ssim_by_time: Optional[np.ndarray]
    rollout_mae: float
    rollout_rmse: float
    persistence_rollout_rmse: float
    persistence_improvement_fraction: float
    final_step_mae: float
    final_step_rmse: float
    mean_signed_error: float
    relative_l2_error: float
    area_weighted_relative_l2_error: Optional[float]
    persistence_area_weighted_relative_l2_error: Optional[float]
    mean_rollout_ssim: Optional[float]
    final_step_ssim: Optional[float]
    r_squared: float
    pearson_correlation: float
    mean_absolute_mass_error: float
    max_absolute_mass_error: float
    negative_density_fraction: float
    max_negative_fraction: float


@dataclass(frozen=True)
class MeshGeometry:
    """Static finite-volume geometry for one rollout trajectory."""

    trajectory_id: int
    centers: np.ndarray
    cell_areas: np.ndarray


@dataclass(frozen=True)
class LocalMeshWindow:
    """Row-compressed local Gaussian quadrature weights."""

    indptr: np.ndarray
    indices: np.ndarray
    weights: np.ndarray
    cell_areas: np.ndarray

    @property
    def num_nodes(self) -> int:
        return int(self.cell_areas.size)

    def apply(self, values: np.ndarray) -> np.ndarray:
        """Apply every normalized local window to ``[time, node]`` values."""
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != self.num_nodes:
            raise ValueError(
                "Local-window values must have shape [time, node] matching "
                f"the geometry; got {array.shape} for {self.num_nodes} nodes."
            )
        gathered = array[:, self.indices] * self.weights[None, :]
        return np.add.reduceat(gathered, self.indptr[:-1], axis=1)


def _format_axis(axis: plt.Axes) -> None:
    axis.grid(True, alpha=0.27)
    axis.set_axisbelow(True)


def _prediction_from_directory(path: Path) -> Tuple[Path, Path]:
    """Return prediction file and run root for a directory argument."""
    direct = path / "test_rollout_predictions.npz"
    nested = path / "rollout_plots" / "test_rollout_predictions.npz"
    if direct.is_file():
        run_root = path.parent if path.name == "rollout_plots" else path
        return direct, run_root
    if nested.is_file():
        return nested, path
    raise FileNotFoundError(
        f"No test_rollout_predictions.npz was found in {path} or its "
        "rollout_plots directory. Run plot_gnn_rollouts.py first."
    )


def resolve_model_path(path: Path) -> Tuple[Path, Path, Optional[Path]]:
    """Resolve a run/checkpoint/summary/prediction path to saved artifacts."""
    path = path.expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Model path does not exist: {path}")
    if path.is_dir():
        prediction_path, run_root = _prediction_from_directory(path)
    elif path.name == "checkpoint.pt" or path.name == "history.json":
        run_root = path.parent
        prediction_path, _ = _prediction_from_directory(run_root)
    elif path.name in (
        "summary_metrics.json",
        "test_rollout_metrics.csv",
        "test_rollout_predictions.npz",
    ):
        rollout_root = path.parent
        prediction_path, run_root = _prediction_from_directory(rollout_root)
    else:
        raise ValueError(
            f"Unsupported model file {path}. Supply a run directory, rollout "
            "directory, checkpoint.pt, history.json, summary_metrics.json, "
            "test_rollout_metrics.csv, or test_rollout_predictions.npz."
        )
    history_path = run_root / "history.json"
    return (
        prediction_path.resolve(),
        run_root.resolve(),
        history_path.resolve() if history_path.is_file() else None,
    )


def parse_model_spec(value: str) -> ModelSpec:
    """Parse and validate one ``LABEL=PATH`` command-line value."""
    if "=" not in value:
        raise argparse.ArgumentTypeError("--model must have the form LABEL=PATH.")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    raw_path = raw_path.strip()
    if not label:
        raise argparse.ArgumentTypeError("Model label cannot be empty.")
    if not raw_path:
        raise argparse.ArgumentTypeError("Model path cannot be empty.")
    try:
        prediction_path, run_root, history_path = resolve_model_path(Path(raw_path))
    except (FileNotFoundError, ValueError) as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return ModelSpec(
        label=label,
        prediction_path=prediction_path,
        run_root=run_root,
        history_path=history_path,
    )


def _summary_dataset_path(spec: ModelSpec) -> Optional[Path]:
    """Return the source dataset recorded beside a rollout, when available."""
    summary_path = spec.prediction_path.parent / "summary_metrics.json"
    if not summary_path.is_file():
        return None
    with summary_path.open("r", encoding="utf-8") as stream:
        summary = json.load(stream)
    if not isinstance(summary, Mapping):
        raise ValueError(f"Expected a JSON object in {summary_path}.")
    value = summary.get("data")
    if value is None or not str(value).strip():
        return None
    return Path(str(value)).expanduser().resolve()


def infer_dataset_path(
    specs: Sequence[ModelSpec], *, explicit_path: Optional[Path]
) -> Tuple[Path, str]:
    """Resolve one common source dataset for all compared rollout artifacts."""
    inferred = {
        spec.label: path
        for spec in specs
        for path in [_summary_dataset_path(spec)]
        if path is not None
    }
    unique_inferred = set(inferred.values())
    if len(unique_inferred) > 1:
        details = "; ".join(
            f"{label}={path}" for label, path in sorted(inferred.items())
        )
        raise ValueError(
            "Compared rollout summaries refer to different datasets: " + details
        )

    if explicit_path is not None:
        dataset_path = explicit_path.expanduser().resolve()
        mismatches = {
            label: path
            for label, path in inferred.items()
            if path != dataset_path
        }
        if mismatches:
            details = "; ".join(
                f"{label}={path}" for label, path in sorted(mismatches.items())
            )
            raise ValueError(
                f"--data resolves to {dataset_path}, but rollout summaries "
                f"record different datasets ({details})."
            )
        source = "command_line"
    elif len(unique_inferred) == 1 and len(inferred) == len(specs):
        dataset_path = next(iter(unique_inferred))
        source = "rollout_summary_metrics"
    else:
        missing = [spec.label for spec in specs if spec.label not in inferred]
        raise ValueError(
            "Cannot infer one source dataset for every rollout artifact. "
            "Pass --data explicitly; missing summary metadata for: "
            + ", ".join(missing)
        )

    if not (dataset_path.is_file() or dataset_path.is_dir()):
        raise FileNotFoundError(f"Source dataset does not exist: {dataset_path}")
    return dataset_path, source


@lru_cache(maxsize=4)
def _load_source_dataset_cached(dataset_path: Path) -> Dict[str, np.ndarray]:
    """Load one plotting dataset, merging trajectory shards when needed."""
    return load_archive_fields(dataset_path, SOURCE_DATASET_FIELDS)


def load_source_dataset(dataset_path: Path) -> Dict[str, np.ndarray]:
    """Return an in-memory view of a monolithic or sharded source dataset."""
    return _load_source_dataset_cached(Path(dataset_path).expanduser().resolve())


def load_mesh_geometries(
    dataset_path: Path,
    trajectory_ids: Sequence[int],
    *,
    expected_num_nodes: int,
) -> Dict[int, MeshGeometry]:
    """Load and validate the trajectory-specific mesh centers and cell areas."""
    archive = load_source_dataset(dataset_path)
    missing = [
        name for name in ("cell_centers", "cell_areas") if name not in archive
    ]
    if missing:
        raise KeyError(
            f"{dataset_path} is missing geometry array(s): " + ", ".join(missing)
        )
    centers = np.asarray(archive["cell_centers"], dtype=np.float64)
    areas = np.asarray(archive["cell_areas"], dtype=np.float64)

    if centers.ndim != 3 or centers.shape[2] != 2:
        raise ValueError(
            f"cell_centers in {dataset_path} must have shape "
            "[trajectory, node, 2]."
        )
    if areas.ndim != 2 or areas.shape != centers.shape[:2]:
        raise ValueError(
            f"cell_areas in {dataset_path} must have shape [trajectory, node] "
            "matching cell_centers."
        )
    if centers.shape[1] != expected_num_nodes:
        raise ValueError(
            f"Dataset geometry has {centers.shape[1]} nodes but the rollout "
            f"artifacts have {expected_num_nodes}."
        )

    selected: Dict[int, MeshGeometry] = {}
    for raw_id in trajectory_ids:
        trajectory_id = int(raw_id)
        if trajectory_id < 0 or trajectory_id >= centers.shape[0]:
            raise ValueError(
                f"Trajectory {trajectory_id} does not exist in {dataset_path}; "
                f"valid IDs are 0 through {centers.shape[0] - 1}."
            )
        trajectory_centers = centers[trajectory_id].copy()
        trajectory_areas = areas[trajectory_id].copy()
        if not np.all(np.isfinite(trajectory_centers)):
            raise ValueError(
                f"cell_centers for trajectory {trajectory_id} contain "
                "non-finite values."
            )
        if (
            not np.all(np.isfinite(trajectory_areas))
            or np.any(trajectory_areas <= 0.0)
        ):
            raise ValueError(
                f"cell_areas for trajectory {trajectory_id} must be positive "
                "and finite."
            )
        selected[trajectory_id] = MeshGeometry(
            trajectory_id=trajectory_id,
            centers=trajectory_centers,
            cell_areas=trajectory_areas,
        )
    return selected


def resolve_ssim_reference_spacing(
    geometries: Mapping[int, MeshGeometry],
    *,
    explicit_spacing: Optional[float],
) -> Tuple[float, str]:
    """Resolve one physical reference spacing shared by the benchmark."""
    if explicit_spacing is not None:
        spacing = float(explicit_spacing)
        if not np.isfinite(spacing) or spacing <= 0.0:
            raise ValueError(
                "Explicit SSIM reference spacing must be positive and finite."
            )
        return spacing, "command_line"
    if not geometries:
        raise ValueError("At least one mesh is required to infer SSIM spacing.")
    all_areas = np.concatenate(
        [geometry.cell_areas for geometry in geometries.values()]
    )
    spacing = float(np.sqrt(np.median(all_areas)))
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("Could not infer a positive finite SSIM spacing.")
    return spacing, "sqrt_median_selected_cell_area"


def build_local_mesh_window(
    centers: np.ndarray,
    cell_areas: np.ndarray,
    *,
    reference_spacing: float,
    window_size: int = SSIM_WINDOW_SIZE,
    sigma_cells: float = SSIM_SIGMA_CELLS,
) -> LocalMeshWindow:
    """Build truncated, renormalized Gaussian windows on an irregular mesh."""
    points = np.asarray(centers, dtype=np.float64)
    areas = np.asarray(cell_areas, dtype=np.float64).reshape(-1)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("centers must have shape [node, 2].")
    if points.shape[0] == 0 or areas.shape != (points.shape[0],):
        raise ValueError("cell_areas must have shape [node] for a nonempty mesh.")
    if not np.all(np.isfinite(points)):
        raise ValueError("centers must be finite.")
    if not np.all(np.isfinite(areas)) or np.any(areas <= 0.0):
        raise ValueError("cell_areas must be positive and finite.")
    if not np.isfinite(reference_spacing) or reference_spacing <= 0.0:
        raise ValueError("reference_spacing must be positive and finite.")
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("window_size must be an odd integer of at least 3.")
    if not np.isfinite(sigma_cells) or sigma_cells <= 0.0:
        raise ValueError("sigma_cells must be positive and finite.")

    radius = (window_size // 2) * float(reference_spacing)
    sigma = float(sigma_cells) * float(reference_spacing)
    delta = points[:, None, :] - points[None, :, :]
    distance_squared = np.sum(delta * delta, axis=2)
    tolerance = np.finfo(np.float64).eps * max(1.0, radius * radius) * 8.0
    neighbor_mask = distance_squared <= radius * radius + tolerance
    if not np.all(np.diag(neighbor_mask)):
        raise RuntimeError("Every local SSIM window must contain its center cell.")

    indptr = np.zeros(points.shape[0] + 1, dtype=np.int64)
    row_indices: List[np.ndarray] = []
    row_weights: List[np.ndarray] = []
    for node in range(points.shape[0]):
        indices = np.flatnonzero(neighbor_mask[node])
        unnormalized = areas[indices] * np.exp(
            -0.5 * distance_squared[node, indices] / (sigma * sigma)
        )
        total_weight = float(np.sum(unnormalized))
        if not np.isfinite(total_weight) or total_weight <= 0.0:
            raise RuntimeError(
                f"Local SSIM window {node} has nonpositive total weight."
            )
        row_indices.append(indices.astype(np.int64, copy=False))
        row_weights.append(unnormalized / total_weight)
        indptr[node + 1] = indptr[node] + indices.size

    return LocalMeshWindow(
        indptr=indptr,
        indices=np.concatenate(row_indices),
        weights=np.concatenate(row_weights),
        cell_areas=areas.copy(),
    )


def area_weighted_relative_l2_by_time(
    truth: np.ndarray,
    prediction: np.ndarray,
    *,
    cell_areas: np.ndarray,
) -> np.ndarray:
    """Return the finite-volume relative L2 error independently at each time."""
    reference = np.asarray(truth, dtype=np.float64)
    candidate = np.asarray(prediction, dtype=np.float64)
    areas = np.asarray(cell_areas, dtype=np.float64).reshape(-1)
    if reference.ndim != 2 or candidate.shape != reference.shape:
        raise ValueError("truth and prediction must have shape [time, node].")
    if areas.shape != (reference.shape[1],):
        raise ValueError("cell_areas must match the rollout node dimension.")
    if not np.all(np.isfinite(areas)) or np.any(areas <= 0.0):
        raise ValueError("cell_areas must be positive and finite.")
    numerator = np.sum((candidate - reference) ** 2 * areas[None, :], axis=1)
    denominator = np.sum(reference**2 * areas[None, :], axis=1)
    denominator = np.maximum(denominator, np.finfo(np.float64).eps)
    return np.sqrt(numerator / denominator)


def local_mesh_ssim_map_by_time(
    truth: np.ndarray,
    prediction: np.ndarray,
    *,
    window: LocalMeshWindow,
    data_range: float,
    k1: float = SSIM_K1,
    k2: float = SSIM_K2,
) -> np.ndarray:
    """Return local SSIM at every rollout time and center cell."""
    reference = np.asarray(truth, dtype=np.float64)
    candidate = np.asarray(prediction, dtype=np.float64)
    if reference.ndim != 2 or candidate.shape != reference.shape:
        raise ValueError("truth and prediction must have shape [time, node].")
    if reference.shape[1] != window.num_nodes:
        raise ValueError("The local window and rollout node counts differ.")
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(candidate)):
        raise ValueError("SSIM fields must be finite.")
    signal_range = float(data_range)
    if not np.isfinite(signal_range) or signal_range <= 0.0:
        raise ValueError("SSIM data_range must be positive and finite.")
    if not np.isfinite(k1) or not np.isfinite(k2) or k1 <= 0.0 or k2 <= 0.0:
        raise ValueError("SSIM k1 and k2 must be positive and finite.")

    reference_mean = window.apply(reference)
    candidate_mean = window.apply(candidate)
    reference_variance = np.maximum(
        window.apply(reference**2) - reference_mean**2, 0.0
    )
    candidate_variance = np.maximum(
        window.apply(candidate**2) - candidate_mean**2, 0.0
    )
    covariance = (
        window.apply(reference * candidate) - reference_mean * candidate_mean
    )
    c1 = (float(k1) * signal_range) ** 2
    c2 = (float(k2) * signal_range) ** 2
    numerator = (
        (2.0 * reference_mean * candidate_mean + c1)
        * (2.0 * covariance + c2)
    )
    denominator = (
        (reference_mean**2 + candidate_mean**2 + c1)
        * (reference_variance + candidate_variance + c2)
    )
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        local_ssim = numerator / denominator
    if not np.all(np.isfinite(local_ssim)):
        raise FloatingPointError("Local mesh SSIM produced a non-finite value.")
    return np.clip(local_ssim, -1.0, 1.0)


def local_mesh_ssim_by_time(
    truth: np.ndarray,
    prediction: np.ndarray,
    *,
    window: LocalMeshWindow,
    data_range: float,
    k1: float = SSIM_K1,
    k2: float = SSIM_K2,
) -> np.ndarray:
    """Area-average the local mesh SSIM map independently at every time."""
    local_ssim = local_mesh_ssim_map_by_time(
        truth,
        prediction,
        window=window,
        data_range=data_range,
        k1=k1,
        k2=k2,
    )
    return (
        np.sum(local_ssim * window.cell_areas[None, :], axis=1)
        / np.sum(window.cell_areas)
    )


def _density_array(values: np.ndarray, name: str, path: Path) -> np.ndarray:
    """Convert a saved density array to [trajectory, time, node]."""
    result = np.asarray(values, dtype=np.float64)
    if result.ndim == 4 and result.shape[-1] == 1:
        result = result[..., 0]
    if result.ndim != 3:
        raise ValueError(
            f"{name} in {path} must have shape [trajectory, time, node, 1] "
            "or [trajectory, time, node]."
        )
    return result.copy()


def load_rollout(path: Path) -> RolloutData:
    """Load and validate one chemotaxis rollout-prediction artifact."""
    with np.load(path, allow_pickle=False) as archive:
        missing = [name for name in REQUIRED_ARRAYS if name not in archive]
        if missing:
            raise KeyError(f"{path} is missing array(s): {', '.join(missing)}")
        trajectory_ids = np.asarray(archive["trajectory_ids"], dtype=np.int64).copy()
        truth = _density_array(archive["truth"], "truth", path)
        prediction = _density_array(archive["prediction"], "prediction", path)
        if "persistence" in archive:
            persistence = _density_array(archive["persistence"], "persistence", path)
        else:
            persistence = np.broadcast_to(truth[:, :1], truth.shape).copy()
        time = np.asarray(archive["time"], dtype=np.float64).copy()
        mass_error = np.asarray(archive["mass_error"], dtype=np.float64).copy()
        timing_names = (
            "gnn_inference_seconds",
            "gnn_inference_num_trajectories",
            "gnn_inference_num_output_intervals",
            "gnn_inference_batch_size",
            "gnn_inference_device",
            "gnn_inference_timing_scope",
        )
        present_timing_names = [name for name in timing_names if name in archive]
        if present_timing_names and len(present_timing_names) != len(timing_names):
            missing_timing_names = sorted(set(timing_names) - set(present_timing_names))
            raise KeyError(
                f"{path} has incomplete GNN inference timing metadata; missing: "
                + ", ".join(missing_timing_names)
            )
        inference_timing = (
            {
                "total_seconds": float(
                    np.asarray(archive["gnn_inference_seconds"]).item()
                ),
                "num_trajectories": int(
                    np.asarray(archive["gnn_inference_num_trajectories"]).item()
                ),
                "num_output_intervals": int(
                    np.asarray(
                        archive["gnn_inference_num_output_intervals"]
                    ).item()
                ),
                "batch_size": int(
                    np.asarray(archive["gnn_inference_batch_size"]).item()
                ),
                "device": str(np.asarray(archive["gnn_inference_device"]).item()),
                "scope": str(
                    np.asarray(archive["gnn_inference_timing_scope"]).item()
                ),
            }
            if present_timing_names
            else None
        )

    if prediction.shape != truth.shape or persistence.shape != truth.shape:
        raise ValueError(
            f"truth, prediction, and persistence shapes differ in {path}: "
            f"{truth.shape}, {prediction.shape}, {persistence.shape}."
        )
    expected_rows = truth.shape[0]
    expected_times = truth.shape[1]
    if trajectory_ids.shape != (expected_rows,):
        raise ValueError(
            f"trajectory_ids in {path} must have shape ({expected_rows},)."
        )
    if np.unique(trajectory_ids).size != trajectory_ids.size:
        raise ValueError(f"trajectory_ids in {path} contain duplicates.")
    if time.shape != (expected_times,) or mass_error.shape != (
        expected_rows,
        expected_times,
    ):
        raise ValueError(
            f"time or mass_error has an incompatible shape in {path}."
        )
    if expected_times < 2:
        raise ValueError(f"{path} must contain at least one future rollout state.")
    arrays = (truth, prediction, persistence, time, mass_error)
    if not all(np.all(np.isfinite(values)) for values in arrays):
        raise ValueError(f"{path} contains non-finite rollout values.")
    if np.any(np.diff(time) <= 0.0):
        raise ValueError(f"time in {path} must be strictly increasing.")
    if inference_timing is not None:
        if (
            not np.isfinite(inference_timing["total_seconds"])
            or inference_timing["total_seconds"] < 0.0
            or inference_timing["num_trajectories"] <= 0
            or inference_timing["num_output_intervals"] <= 0
            or inference_timing["batch_size"] <= 0
        ):
            raise ValueError(f"{path} contains invalid GNN inference timing metadata.")
    return RolloutData(
        trajectory_ids=trajectory_ids,
        truth=truth,
        prediction=prediction,
        persistence=persistence,
        time=time,
        mass_error=mass_error,
        source=path,
        inference_timing=inference_timing,
    )


def trajectory_index(data: RolloutData, trajectory_id: int) -> int:
    matches = np.flatnonzero(data.trajectory_ids == trajectory_id)
    if matches.size != 1:
        raise ValueError(
            f"Trajectory {trajectory_id} was not found exactly once in {data.source}."
        )
    return int(matches[0])


def common_trajectory_ids(
    rollouts: Mapping[str, RolloutData],
    requested: Optional[Sequence[int]],
) -> List[int]:
    """Select requested IDs or the intersection available to every model."""
    available = {
        label: set(int(value) for value in data.trajectory_ids)
        for label, data in rollouts.items()
    }
    common = set.intersection(*available.values())
    if requested is None:
        if not common:
            raise ValueError("The model artifacts share no test trajectories.")
        if any(values != common for values in available.values()):
            print(
                "warning: comparing only common trajectory IDs: "
                + ", ".join(map(str, sorted(common)))
            )
        return sorted(common)
    selected = list(dict.fromkeys(int(value) for value in requested))
    missing = {
        label: [value for value in selected if value not in values]
        for label, values in available.items()
    }
    missing = {label: values for label, values in missing.items() if values}
    if missing:
        details = "; ".join(f"{label}: {values}" for label, values in missing.items())
        raise ValueError(
            f"Requested trajectories are not available in every model ({details})."
        )
    return selected


def validate_common_rollouts(
    rollouts: Mapping[str, RolloutData],
    trajectory_ids: Sequence[int],
    rollout_steps: Optional[int],
) -> int:
    """Validate common times and ground truth; return compared time count."""
    available_times = min(data.truth.shape[1] for data in rollouts.values())
    if rollout_steps is None:
        num_times = available_times
    else:
        num_times = rollout_steps + 1
        if num_times > available_times:
            details = ", ".join(
                f"{label}={data.truth.shape[1] - 1}"
                for label, data in rollouts.items()
            )
            raise ValueError(
                f"--rollout-steps={rollout_steps} exceeds an available "
                f"rollout ({details})."
            )
    reference_label = next(iter(rollouts))
    reference = rollouts[reference_label]
    reference_time = reference.time[:num_times]
    for label, data in rollouts.items():
        if data.truth.shape[2] != reference.truth.shape[2]:
            raise ValueError(
                f"Node counts differ between {reference_label} and {label}; "
                "these rollouts cannot be paired."
            )
        if not np.allclose(
            data.time[:num_times], reference_time, rtol=1.0e-8, atol=1.0e-12
        ):
            raise ValueError(
                f"Saved rollout times differ between {reference_label} and {label}."
            )
        for trajectory_id in trajectory_ids:
            reference_truth = reference.truth[
                trajectory_index(reference, trajectory_id), :num_times
            ]
            candidate_truth = data.truth[
                trajectory_index(data, trajectory_id), :num_times
            ]
            if not np.allclose(
                candidate_truth, reference_truth, rtol=1.0e-6, atol=1.0e-7
            ):
                maximum_difference = float(
                    np.max(np.abs(candidate_truth - reference_truth))
                )
                raise ValueError(
                    "Ground truth differs between "
                    f"{reference_label} and {label} for trajectory "
                    f"{trajectory_id} (max difference {maximum_difference:.3e}). "
                    "Evaluate the models on the same dataset and test split before "
                    "making a paired comparison."
                )
    return num_times


def _declared_physical_density_range(dataset_path: Path) -> Optional[Tuple[float, str]]:
    """Read an explicitly declared physical density range, if one is present."""
    archive = load_source_dataset(dataset_path)
    if "density_physical_range" in archive:
        bounds = np.asarray(
            archive["density_physical_range"], dtype=np.float64
        ).reshape(-1)
        if bounds.shape != (2,):
            raise ValueError(
                "density_physical_range must contain exactly [minimum, maximum]."
            )
        lower, upper = map(float, bounds)
        source = "dataset_density_physical_range"
    elif "density_physical_min" in archive and "density_physical_max" in archive:
        lower = float(np.asarray(archive["density_physical_min"]).item())
        upper = float(np.asarray(archive["density_physical_max"]).item())
        source = "dataset_density_physical_min_max"
    else:
        return None
    signal_range = upper - lower
    if (
        not np.isfinite(lower)
        or not np.isfinite(upper)
        or signal_range <= 0.0
    ):
        raise ValueError(
            f"The declared physical density range in {dataset_path} is invalid."
        )
    return float(signal_range), source


def resolve_ssim_data_range(
    rollouts: Mapping[str, RolloutData],
    trajectory_ids: Sequence[int],
    *,
    num_times: int,
    explicit_range: Optional[float],
    dataset_path: Optional[Path] = None,
) -> Tuple[float, str]:
    """Resolve one fixed prediction-independent SSIM signal range."""
    if explicit_range is not None:
        signal_range = float(explicit_range)
        if not np.isfinite(signal_range) or signal_range <= 0.0:
            raise ValueError("Explicit SSIM data range must be positive and finite.")
        return signal_range, "command_line"

    if dataset_path is not None:
        declared = _declared_physical_density_range(dataset_path)
        if declared is not None:
            return declared

    if not rollouts:
        raise ValueError("At least one rollout is required to infer SSIM data range.")
    reference = next(iter(rollouts.values()))
    selected_truth = np.concatenate(
        [
            reference.truth[
                trajectory_index(reference, int(trajectory_id)), :num_times
            ].reshape(-1)
            for trajectory_id in trajectory_ids
        ]
    )
    finite = selected_truth[np.isfinite(selected_truth)]
    if finite.size == 0:
        raise ValueError("Selected reference trajectories contain no finite values.")
    signal_range = float(np.max(finite) - np.min(finite))
    if not np.isfinite(signal_range) or signal_range <= 0.0:
        raise ValueError(
            "Selected reference trajectories have zero or invalid density range; "
            "pass --ssim-data-range explicitly."
        )
    return signal_range, "selected_reference_truth_max_minus_min"


def fidelity_metadata(
    *,
    data_range: float,
    data_range_source: str,
    window_size: int,
    sigma_cells: float,
    reference_spacing: float,
    reference_spacing_source: str,
) -> Dict[str, object]:
    """Describe the exact mesh-fidelity definitions used for saved artifacts."""
    return {
        "ssim_mode": SSIM_MODE_LOCAL_MESH,
        "ssim_data_range": float(data_range),
        "ssim_data_range_source": data_range_source,
        "ssim_window_size_reference_cells": int(window_size),
        "ssim_sigma_reference_cells": float(sigma_cells),
        "ssim_reference_spacing": float(reference_spacing),
        "ssim_reference_spacing_source": reference_spacing_source,
        "ssim_physical_window_radius": float(
            (window_size // 2) * reference_spacing
        ),
        "ssim_physical_sigma": float(sigma_cells * reference_spacing),
        "ssim_kernel": "gaussian_area_weighted",
        "ssim_boundary": "truncated_renormalized_no_flux_domain",
        "ssim_spatial_aggregation": "cell_area_weighted",
        "ssim_covariance": "population",
        "ssim_k1": SSIM_K1,
        "ssim_k2": SSIM_K2,
        "relative_l2": "cell_area_weighted",
    }


def compute_metrics(
    data: RolloutData,
    trajectory_id: int,
    *,
    num_times: int,
    local_window: Optional[LocalMeshWindow] = None,
    ssim_data_range: Optional[float] = None,
) -> TrajectoryMetrics:
    """Compute all comparison metrics on a common rollout interval."""
    index = trajectory_index(data, trajectory_id)
    truth = data.truth[index, :num_times]
    prediction = data.prediction[index, :num_times]
    persistence = data.persistence[index, :num_times]
    error = prediction - truth
    persistence_error = persistence - truth
    future_error = error[1:]
    future_truth = truth[1:]
    future_persistence_error = persistence_error[1:]
    epsilon = np.finfo(np.float64).eps
    error_squared_sum = float(np.sum(future_error**2))
    truth_squared_sum = max(float(np.sum(future_truth**2)), epsilon)
    centered_truth = future_truth - float(np.mean(future_truth))
    truth_variation = max(float(np.sum(centered_truth**2)), epsilon)
    true_values = future_truth.reshape(-1)
    predicted_values = prediction[1:].reshape(-1)
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
    persistence_mae = np.mean(np.abs(persistence_error), axis=1)
    persistence_rmse = np.sqrt(np.mean(persistence_error**2, axis=1))
    rollout_rmse = float(np.sqrt(np.mean(future_error**2)))
    persistence_rollout_rmse = float(
        np.sqrt(np.mean(future_persistence_error**2))
    )
    improvement = (
        1.0 - rollout_rmse / persistence_rollout_rmse
        if persistence_rollout_rmse > epsilon
        else float("nan")
    )
    mass_error = data.mass_error[index, :num_times]
    negative_fraction = np.mean(prediction < 0.0, axis=1)
    if (local_window is None) != (ssim_data_range is None):
        raise ValueError(
            "local_window and ssim_data_range must either both be supplied "
            "or both be omitted."
        )
    if local_window is None:
        weighted_relative_l2 = None
        persistence_weighted_relative_l2 = None
        local_ssim = None
        persistence_local_ssim = None
        weighted_rollout_relative_l2 = None
        persistence_weighted_rollout_relative_l2 = None
        mean_rollout_ssim = None
        final_step_ssim = None
    else:
        assert ssim_data_range is not None
        areas = local_window.cell_areas
        weighted_relative_l2 = area_weighted_relative_l2_by_time(
            truth, prediction, cell_areas=areas
        )
        persistence_weighted_relative_l2 = area_weighted_relative_l2_by_time(
            truth, persistence, cell_areas=areas
        )
        local_ssim = local_mesh_ssim_by_time(
            truth,
            prediction,
            window=local_window,
            data_range=ssim_data_range,
        )
        persistence_local_ssim = local_mesh_ssim_by_time(
            truth,
            persistence,
            window=local_window,
            data_range=ssim_data_range,
        )
        weighted_error_squared = float(
            np.sum(future_error**2 * areas[None, :])
        )
        weighted_truth_squared = max(
            float(np.sum(future_truth**2 * areas[None, :])), epsilon
        )
        weighted_rollout_relative_l2 = float(
            np.sqrt(weighted_error_squared / weighted_truth_squared)
        )
        persistence_weighted_error_squared = float(
            np.sum(future_persistence_error**2 * areas[None, :])
        )
        persistence_weighted_rollout_relative_l2 = float(
            np.sqrt(
                persistence_weighted_error_squared / weighted_truth_squared
            )
        )
        mean_rollout_ssim = float(np.mean(local_ssim[1:]))
        final_step_ssim = float(local_ssim[-1])
    return TrajectoryMetrics(
        mae_by_time=mae_by_time,
        rmse_by_time=rmse_by_time,
        bias_by_time=np.mean(error, axis=1),
        mass_error_by_time=mass_error,
        negative_fraction_by_time=negative_fraction,
        persistence_mae_by_time=persistence_mae,
        persistence_rmse_by_time=persistence_rmse,
        area_weighted_relative_l2_by_time=weighted_relative_l2,
        persistence_area_weighted_relative_l2_by_time=(
            persistence_weighted_relative_l2
        ),
        local_ssim_by_time=local_ssim,
        persistence_local_ssim_by_time=persistence_local_ssim,
        rollout_mae=float(np.mean(np.abs(future_error))),
        rollout_rmse=rollout_rmse,
        persistence_rollout_rmse=persistence_rollout_rmse,
        persistence_improvement_fraction=improvement,
        final_step_mae=float(mae_by_time[-1]),
        final_step_rmse=float(rmse_by_time[-1]),
        mean_signed_error=float(np.mean(future_error)),
        relative_l2_error=float(np.sqrt(error_squared_sum / truth_squared_sum)),
        area_weighted_relative_l2_error=weighted_rollout_relative_l2,
        persistence_area_weighted_relative_l2_error=(
            persistence_weighted_rollout_relative_l2
        ),
        mean_rollout_ssim=mean_rollout_ssim,
        final_step_ssim=final_step_ssim,
        r_squared=float(1.0 - error_squared_sum / truth_variation),
        pearson_correlation=correlation,
        mean_absolute_mass_error=float(np.mean(np.abs(mass_error))),
        max_absolute_mass_error=float(np.max(np.abs(mass_error))),
        negative_density_fraction=float(np.mean(negative_fraction)),
        max_negative_fraction=float(np.max(negative_fraction)),
    )


def model_styles(labels: Sequence[str]) -> Dict[str, Dict[str, object]]:
    """Assign stable line styles to model labels."""
    colors = plt.get_cmap("tab10")
    line_styles = ("-", "--", "-.", ":")
    return {
        label: {
            "color": colors(index % 10),
            "linestyle": line_styles[(index // 10) % len(line_styles)],
            "linewidth": 1.8,
        }
        for index, label in enumerate(labels)
    }


def stack_metric(
    metrics: Sequence[TrajectoryMetrics], attribute: str
) -> np.ndarray:
    return np.stack(
        [np.asarray(getattr(item, attribute)) for item in metrics], axis=0
    )


def save_figure(
    figure: plt.Figure,
    base_path: Path,
    *,
    formats: Sequence[str],
    dpi: int,
) -> List[Path]:
    """Save a figure in every requested format and return written paths."""
    base_path.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for extension in formats:
        output_path = base_path.with_suffix(f".{extension}")
        figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
        written.append(output_path)
    plt.close(figure)
    return written


def plot_rollout_summary(
    *,
    metrics_by_model: Mapping[str, Sequence[TrajectoryMetrics]],
    styles: Mapping[str, Mapping[str, object]],
    times: np.ndarray,
    output_base: Path,
    formats: Sequence[str],
    dpi: int,
    show_iqr: bool,
) -> List[Path]:
    """Plot aggregate time histories and trajectory-level distributions."""
    figure, axes = plt.subplots(
        2, 3, figsize=(15.4, 8.7), constrained_layout=True
    )
    ax_rmse, ax_mae, ax_mass, ax_negative, ax_ecdf, ax_skill = axes.ravel()
    panels = (
        (ax_rmse, "rmse_by_time", "Median density RMSE", "Node RMSE", False),
        (ax_mae, "mae_by_time", "Median density MAE", "Node MAE", False),
        (
            ax_mass,
            "mass_error_by_time",
            "Median absolute total-mass error",
            r"$|\hat M-M|$",
            True,
        ),
        (
            ax_negative,
            "negative_fraction_by_time",
            "Median negative-density fraction",
            "Fraction of nodes",
            False,
        ),
    )
    for axis, attribute, title, ylabel, absolute in panels:
        for label, all_metrics in metrics_by_model.items():
            values = stack_metric(all_metrics, attribute)
            if absolute:
                values = np.abs(values)
            median = np.median(values, axis=0)
            axis.plot(times, median, label=label, **styles[label])
            if show_iqr:
                lower, upper = np.quantile(values, (0.25, 0.75), axis=0)
                axis.fill_between(
                    times,
                    lower,
                    upper,
                    color=styles[label]["color"],
                    alpha=0.11,
                    linewidth=0.0,
                )
        axis.set_title(title)
        axis.set_xlabel("Time")
        axis.set_ylabel(ylabel)
        axis.legend(fontsize=8)
        _format_axis(axis)

    first_metrics = next(iter(metrics_by_model.values()))
    persistence_rmse = stack_metric(first_metrics, "persistence_rmse_by_time")
    persistence_mae = stack_metric(first_metrics, "persistence_mae_by_time")
    ax_rmse.plot(
        times,
        np.median(persistence_rmse, axis=0),
        color="0.35",
        linestyle=":",
        linewidth=2.0,
        label="Persistence",
    )
    ax_mae.plot(
        times,
        np.median(persistence_mae, axis=0),
        color="0.35",
        linestyle=":",
        linewidth=2.0,
        label="Persistence",
    )
    ax_rmse.legend(fontsize=8)
    ax_mae.legend(fontsize=8)

    for label, all_metrics in metrics_by_model.items():
        values = np.sort(np.asarray([item.rollout_rmse for item in all_metrics]))
        cumulative = np.arange(1, values.size + 1) / values.size
        ax_ecdf.step(
            values, cumulative, where="post", label=label, **styles[label]
        )
    ax_ecdf.set_title("Trajectory rollout-RMSE distribution")
    ax_ecdf.set_xlabel("Rollout RMSE")
    ax_ecdf.set_ylabel("Empirical cumulative fraction")
    ax_ecdf.set_ylim(0.0, 1.02)
    ax_ecdf.legend(fontsize=8)
    _format_axis(ax_ecdf)

    labels = list(metrics_by_model)
    skill_values = [
        np.asarray(
            [item.persistence_improvement_fraction for item in metrics_by_model[label]],
            dtype=np.float64,
        )
        for label in labels
    ]
    box = ax_skill.boxplot(
        skill_values,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.4},
    )
    for patch, label in zip(box["boxes"], labels):
        patch.set_facecolor(styles[label]["color"])
        patch.set_alpha(0.48)
    for index, values in enumerate(skill_values, start=1):
        offsets = np.linspace(-0.10, 0.10, values.size)
        ax_skill.scatter(
            index + offsets,
            values,
            s=15,
            alpha=0.48,
            color=styles[labels[index - 1]]["color"],
            edgecolors="none",
        )
    ax_skill.axhline(0.0, color="0.3", linestyle="--", linewidth=1.0)
    ax_skill.set_xticks(np.arange(1, len(labels) + 1), labels)
    ax_skill.set_title("Improvement over persistence")
    ax_skill.set_ylabel(r"$1-\mathrm{RMSE}_{model}/\mathrm{RMSE}_{persist}$")
    ax_skill.tick_params(axis="x", rotation=18)
    _format_axis(ax_skill)

    figure.suptitle(
        f"Held-out chemotaxis model comparison "
        f"({len(first_metrics)} shared trajectories)",
        fontsize=14,
    )
    return save_figure(
        figure, output_base, formats=formats, dpi=dpi
    )


def plot_fidelity_summary(
    *,
    metrics_by_model: Mapping[str, Sequence[TrajectoryMetrics]],
    styles: Mapping[str, Mapping[str, object]],
    times: np.ndarray,
    output_base: Path,
    formats: Sequence[str],
    dpi: int,
    show_iqr: bool,
    show_persistence: bool,
) -> List[Path]:
    """Plot the raw area-weighted relative-L2 and local-mesh SSIM curves."""
    figure, axes = plt.subplots(
        1, 2, figsize=(15.2, 4.9), constrained_layout=True
    )
    specifications = (
        (
            axes[0],
            "area_weighted_relative_l2_by_time",
            r"Area-weighted relative $L_2$",
        ),
        (axes[1], "local_ssim_by_time", "Local SSIM"),
    )
    for axis, attribute, ylabel in specifications:
        for label, all_metrics in metrics_by_model.items():
            values = stack_metric(all_metrics, attribute)
            mean = np.mean(values, axis=0)
            axis.plot(times, mean, label=label, **styles[label])
            if show_iqr:
                lower, upper = np.quantile(values, (0.25, 0.75), axis=0)
                axis.fill_between(
                    times,
                    lower,
                    upper,
                    color=styles[label]["color"],
                    alpha=0.11,
                    linewidth=0.0,
                )

        if show_persistence:
            first_metrics = next(iter(metrics_by_model.values()))
            persistence_attribute = (
                "persistence_area_weighted_relative_l2_by_time"
                if attribute == "area_weighted_relative_l2_by_time"
                else "persistence_local_ssim_by_time"
            )
            persistence_values = stack_metric(
                first_metrics, persistence_attribute
            )
            persistence_mean = np.mean(persistence_values, axis=0)
            axis.plot(
                times,
                persistence_mean,
                color="0.35",
                linestyle=":",
                linewidth=2.0,
                label="Persistence",
            )
            if show_iqr:
                lower, upper = np.quantile(
                    persistence_values, (0.25, 0.75), axis=0
                )
                axis.fill_between(
                    times,
                    lower,
                    upper,
                    color="0.35",
                    alpha=0.08,
                    linewidth=0.0,
                )
        axis.set_xlabel("Time")
        axis.set_ylabel(ylabel)
        axis.legend(fontsize=8)
        _format_axis(axis)
    axes[1].set_ylim(top=1.0)
    figure.suptitle("Held-out chemotaxis spatial fidelity", fontsize=14)
    return save_figure(figure, output_base, formats=formats, dpi=dpi)


def _bar_panel(
    axis: plt.Axes,
    labels: Sequence[str],
    values: Sequence[float],
    styles: Mapping[str, Mapping[str, object]],
    *,
    title: str,
    ylabel: str,
    percent: bool = False,
) -> None:
    colors = [styles[label]["color"] for label in labels]
    bars = axis.bar(np.arange(len(labels)), values, color=colors, alpha=0.78)
    axis.set_xticks(np.arange(len(labels)), labels, rotation=20, ha="right")
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    scale = max([abs(float(value)) for value in values] + [1.0e-12])
    for bar, value in zip(bars, values):
        text = f"{value:.1%}" if percent else f"{value:.3g}"
        offset = 0.025 * scale
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + (offset if value >= 0.0 else -offset),
            text,
            ha="center",
            va="bottom" if value >= 0.0 else "top",
            fontsize=8,
        )
    _format_axis(axis)


def plot_scalar_scores(
    *,
    metrics_by_model: Mapping[str, Sequence[TrajectoryMetrics]],
    styles: Mapping[str, Mapping[str, object]],
    output_base: Path,
    formats: Sequence[str],
    dpi: int,
) -> List[Path]:
    """Plot aggregate scalar metrics on separate, unit-compatible axes."""
    labels = list(metrics_by_model)
    figure, axes = plt.subplots(
        2, 3, figsize=(15.3, 8.6), constrained_layout=True
    )
    specifications = (
        (
            "Median rollout RMSE",
            "Node RMSE",
            lambda values: np.median([item.rollout_rmse for item in values]),
            False,
        ),
        (
            "Mean final-step RMSE",
            "Node RMSE",
            lambda values: np.mean([item.final_step_rmse for item in values]),
            False,
        ),
        (
            "Mean relative L2 error",
            "Relative L2",
            lambda values: np.mean([item.relative_l2_error for item in values]),
            False,
        ),
        (
            "Fraction beating persistence",
            "Fraction of trajectories",
            lambda values: np.mean(
                [item.rollout_rmse < item.persistence_rollout_rmse for item in values]
            ),
            True,
        ),
        (
            "Mean maximum mass error",
            r"$\max_t |\hat M-M|$",
            lambda values: np.mean(
                [item.max_absolute_mass_error for item in values]
            ),
            False,
        ),
        (
            "Mean negative-density fraction",
            "Fraction of nodes and times",
            lambda values: np.mean(
                [item.negative_density_fraction for item in values]
            ),
            True,
        ),
    )
    for axis, (title, ylabel, reducer, percent) in zip(axes.ravel(), specifications):
        values = [float(reducer(metrics_by_model[label])) for label in labels]
        _bar_panel(
            axis,
            labels,
            values,
            styles,
            title=title,
            ylabel=ylabel,
            percent=percent,
        )
    figure.suptitle(
        "Held-out rollout scores (lower is better except persistence fraction)",
        fontsize=14,
    )
    return save_figure(figure, output_base, formats=formats, dpi=dpi)


def plot_trajectory_diagnostics(
    *,
    trajectory_id: int,
    metrics_by_model: Mapping[str, TrajectoryMetrics],
    styles: Mapping[str, Mapping[str, object]],
    times: np.ndarray,
    output_base: Path,
    formats: Sequence[str],
    dpi: int,
) -> List[Path]:
    """Plot overlaid time diagnostics and scalar scores for one trajectory."""
    figure, axes = plt.subplots(
        2, 3, figsize=(15.4, 8.6), constrained_layout=True
    )
    ax_rmse, ax_mae, ax_bias, ax_mass, ax_negative, ax_table = axes.ravel()
    panels = (
        (ax_rmse, "rmse_by_time", "Density RMSE", "Node RMSE"),
        (ax_mae, "mae_by_time", "Density MAE", "Node MAE"),
        (ax_bias, "bias_by_time", "Signed density bias", r"Mean $\hat n-n$"),
        (
            ax_mass,
            "mass_error_by_time",
            "Total-mass error",
            r"$\hat M-M$",
        ),
        (
            ax_negative,
            "negative_fraction_by_time",
            "Negative-density fraction",
            "Fraction of nodes",
        ),
    )
    for axis, attribute, title, ylabel in panels:
        for label, metrics in metrics_by_model.items():
            axis.plot(times, getattr(metrics, attribute), label=label, **styles[label])
        axis.set_title(title)
        axis.set_xlabel("Time")
        axis.set_ylabel(ylabel)
        axis.legend(fontsize=8)
        _format_axis(axis)
    ax_bias.axhline(0.0, color="0.3", linestyle="--", linewidth=1.0)
    ax_mass.axhline(0.0, color="0.3", linestyle="--", linewidth=1.0)

    ax_table.axis("off")
    ax_table.set_title("Scalar scores")
    labels = list(metrics_by_model)
    columns = [
        "Rollout\nRMSE",
        "Final\nRMSE",
        "Relative\nL2",
        "$R^2$",
        "Persistence\nimprovement",
        "Max mass\nerror",
    ]
    include_fidelity = all(
        metrics.mean_rollout_ssim is not None
        for metrics in metrics_by_model.values()
    )
    if include_fidelity:
        columns.extend(("Area rel.\n$L_2$", "Mean local\nSSIM"))
    cell_text = []
    for label in labels:
        metrics = metrics_by_model[label]
        row = [
            f"{metrics.rollout_rmse:.4g}",
            f"{metrics.final_step_rmse:.4g}",
            f"{metrics.relative_l2_error:.4g}",
            f"{metrics.r_squared:.4g}",
            f"{metrics.persistence_improvement_fraction:.1%}",
            f"{metrics.max_absolute_mass_error:.2e}",
        ]
        if include_fidelity:
            assert metrics.area_weighted_relative_l2_error is not None
            assert metrics.mean_rollout_ssim is not None
            row.extend(
                (
                    f"{metrics.area_weighted_relative_l2_error:.4g}",
                    f"{metrics.mean_rollout_ssim:.4g}",
                )
            )
        cell_text.append(row)
    table = ax_table.table(
        cellText=cell_text,
        rowLabels=labels,
        colLabels=tuple(columns),
        cellLoc="center",
        rowLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.45)
    for row_index, label in enumerate(labels, start=1):
        table[(row_index, -1)].get_text().set_color(styles[label]["color"])

    figure.suptitle(
        f"Model-comparison diagnostics: held-out trajectory {trajectory_id}",
        fontsize=14,
    )
    return save_figure(figure, output_base, formats=formats, dpi=dpi)


def load_history(path: Path) -> List[Mapping[str, object]]:
    """Load a training history without requiring PyTorch."""
    with path.open("r", encoding="utf-8") as stream:
        history = json.load(stream)
    if not isinstance(history, list) or not history:
        raise ValueError(f"Training history is missing or empty: {path}")
    for index, record in enumerate(history):
        if not isinstance(record, Mapping) or "epoch" not in record:
            raise ValueError(f"Invalid history record {index} in {path}.")
    return history


def _history_series(
    history: Sequence[Mapping[str, object]], name: str
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if not all(name in record for record in history):
        return None
    epochs = np.asarray([record["epoch"] for record in history], dtype=np.float64)
    values = np.asarray([record[name] for record in history], dtype=np.float64)
    if not np.all(np.isfinite(epochs)) or not np.all(np.isfinite(values)):
        return None
    return epochs, values


def _set_history_scale(axis: plt.Axes, requested: str, values: List[np.ndarray]) -> None:
    if requested == "log" and values and all(np.all(value > 0.0) for value in values):
        axis.set_yscale("log")


def plot_training_histories(
    *,
    histories: Mapping[str, Sequence[Mapping[str, object]]],
    styles: Mapping[str, Mapping[str, object]],
    output_base: Path,
    formats: Sequence[str],
    dpi: int,
    yscale: str,
) -> List[Path]:
    """Compare saved training/validation diagnostics when histories exist."""
    figure, axes = plt.subplots(
        2, 2, figsize=(14.8, 8.5), constrained_layout=True
    )
    specifications = (
        (
            axes[0, 0],
            "train_normalized_mse",
            "validation_normalized_mse",
            "Normalized node-objective MSE",
            "Normalized MSE",
        ),
        (
            axes[0, 1],
            None,
            "validation_next_state_rmse",
            "Validation physical-state error",
            "Cell-density RMSE",
        ),
        (
            axes[1, 0],
            None,
            "validation_mass_mae",
            "Validation mass error",
            "Mass MAE",
        ),
        (
            axes[1, 1],
            "train_flux_normalized_mse",
            "validation_flux_normalized_mse",
            "Direct flux-supervision error",
            "Normalized flux MSE",
        ),
    )
    for axis, train_field, validation_field, title, ylabel in specifications:
        plotted_values: List[np.ndarray] = []
        plotted = False
        for label, history in histories.items():
            style = styles[label]
            validation = _history_series(history, validation_field)
            if validation is not None:
                epochs, values = validation
                axis.plot(epochs, values, label=label, **style)
                plotted_values.append(values)
                plotted = True
            if train_field is not None:
                train = _history_series(history, train_field)
                if train is not None:
                    epochs, values = train
                    axis.plot(
                        epochs,
                        values,
                        color=style["color"],
                        linestyle=":",
                        linewidth=1.2,
                        alpha=0.58,
                    )
                    plotted_values.append(values)
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        _set_history_scale(axis, yscale, plotted_values)
        _format_axis(axis)
        if plotted:
            axis.legend(fontsize=8)
        else:
            axis.text(
                0.5,
                0.5,
                "Metric not present in selected histories",
                transform=axis.transAxes,
                ha="center",
                va="center",
                color="0.4",
            )
    axes[0, 0].text(
        0.98,
        0.98,
        "solid: validation\ndotted: training",
        transform=axes[0, 0].transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="0.35",
    )
    axes[1, 1].text(
        0.98,
        0.98,
        "solid: validation\ndotted: training",
        transform=axes[1, 1].transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="0.35",
    )
    figure.suptitle("Training-history comparison", fontsize=14)
    return save_figure(figure, output_base, formats=formats, dpi=dpi)


def metric_row(
    label: str,
    trajectory_id: int,
    rollout_steps: int,
    metrics: TrajectoryMetrics,
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "model_label": label,
        "trajectory_id": trajectory_id,
        "rollout_steps": rollout_steps,
        "rollout_mae": metrics.rollout_mae,
        "rollout_rmse": metrics.rollout_rmse,
        "persistence_rollout_rmse": metrics.persistence_rollout_rmse,
        "persistence_improvement_fraction": (
            metrics.persistence_improvement_fraction
        ),
        "final_step_mae": metrics.final_step_mae,
        "final_step_rmse": metrics.final_step_rmse,
        "mean_signed_error": metrics.mean_signed_error,
        "relative_l2_error": metrics.relative_l2_error,
        "r_squared": metrics.r_squared,
        "pearson_correlation": metrics.pearson_correlation,
        "mean_absolute_mass_error": metrics.mean_absolute_mass_error,
        "max_absolute_mass_error": metrics.max_absolute_mass_error,
        "negative_density_fraction": metrics.negative_density_fraction,
        "max_negative_fraction": metrics.max_negative_fraction,
    }
    if metrics.local_ssim_by_time is not None:
        row.update(
            {
                "area_weighted_relative_l2_error": (
                    metrics.area_weighted_relative_l2_error
                ),
                "persistence_area_weighted_relative_l2_error": (
                    metrics.persistence_area_weighted_relative_l2_error
                ),
                "mean_rollout_ssim": metrics.mean_rollout_ssim,
                "final_step_ssim": metrics.final_step_ssim,
            }
        )
    return row


def summarize_model(metrics: Sequence[TrajectoryMetrics]) -> Dict[str, float]:
    """Reduce per-trajectory metrics to comparison summary values."""
    summary = {
        "mean_rollout_mae": float(np.mean([item.rollout_mae for item in metrics])),
        "mean_rollout_rmse": float(
            np.mean([item.rollout_rmse for item in metrics])
        ),
        "median_rollout_rmse": float(
            np.median([item.rollout_rmse for item in metrics])
        ),
        "mean_persistence_rollout_rmse": float(
            np.mean([item.persistence_rollout_rmse for item in metrics])
        ),
        "fraction_beating_persistence": float(
            np.mean(
                [item.rollout_rmse < item.persistence_rollout_rmse for item in metrics]
            )
        ),
        "mean_final_step_rmse": float(
            np.mean([item.final_step_rmse for item in metrics])
        ),
        "mean_relative_l2_error": float(
            np.mean([item.relative_l2_error for item in metrics])
        ),
        "mean_max_absolute_mass_error": float(
            np.mean([item.max_absolute_mass_error for item in metrics])
        ),
        "maximum_absolute_mass_error": float(
            np.max([item.max_absolute_mass_error for item in metrics])
        ),
        "mean_negative_density_fraction": float(
            np.mean([item.negative_density_fraction for item in metrics])
        ),
    }
    if all(item.local_ssim_by_time is not None for item in metrics):
        summary.update(
            {
                "mean_area_weighted_relative_l2_error": float(
                    np.mean(
                        [
                            item.area_weighted_relative_l2_error
                            for item in metrics
                        ]
                    )
                ),
                "mean_rollout_ssim": float(
                    np.mean([item.mean_rollout_ssim for item in metrics])
                ),
                "mean_final_step_ssim": float(
                    np.mean([item.final_step_ssim for item in metrics])
                ),
            }
        )
    return summary


def write_fidelity_npz(
    path: Path,
    *,
    labels: Sequence[str],
    trajectory_ids: Sequence[int],
    times: np.ndarray,
    metrics_by_model: Mapping[str, Sequence[TrajectoryMetrics]],
    metadata: Mapping[str, object],
) -> None:
    """Save unsmoothed per-time fidelity values and their full definition."""
    relative_l2 = np.stack(
        [
            stack_metric(
                metrics_by_model[label], "area_weighted_relative_l2_by_time"
            )
            for label in labels
        ],
        axis=0,
    )
    local_ssim = np.stack(
        [
            stack_metric(metrics_by_model[label], "local_ssim_by_time")
            for label in labels
        ],
        axis=0,
    )
    first_metrics = metrics_by_model[labels[0]]
    persistence_relative_l2 = stack_metric(
        first_metrics, "persistence_area_weighted_relative_l2_by_time"
    )
    persistence_ssim = stack_metric(
        first_metrics, "persistence_local_ssim_by_time"
    )
    arrays: Dict[str, np.ndarray] = {
        "model_labels": np.asarray(labels),
        "trajectory_ids": np.asarray(trajectory_ids, dtype=np.int64),
        "time": np.asarray(times, dtype=np.float64),
        "area_weighted_relative_l2_by_time": relative_l2,
        "local_ssim_by_time": local_ssim,
        "persistence_area_weighted_relative_l2_by_time": (
            persistence_relative_l2
        ),
        "persistence_local_ssim_by_time": persistence_ssim,
    }
    for key, value in metadata.items():
        arrays[key] = np.asarray(value)
    np.savez_compressed(path, **arrays)


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError("No comparison metric rows were generated.")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _add_boolean_optional_argument(
    parser: argparse.ArgumentParser,
    name: str,
    *,
    default: bool,
    help: str,
) -> None:
    """Add ``--flag``/``--no-flag`` on Python versions before argparse 3.9."""
    destination = name.lstrip("-").replace("-", "_")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(name, dest=destination, action="store_true", help=help)
    group.add_argument(
        "--no-" + name.lstrip("-"),
        dest=destination,
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(**{destination: default})


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare saved held-out rollout artifacts from two or more "
            "chemotaxis GNN models."
        )
    )
    parser.add_argument(
        "--model",
        action="append",
        type=parse_model_spec,
        required=True,
        metavar="LABEL=PATH",
        help="Model label and run/artifact path. Repeat once per model.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory in which comparison figures and tables are saved.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help=(
            "Source chemotaxis dataset. For local-mesh SSIM this defaults "
            "to the common data path in the rollout summary files."
        ),
    )
    parser.add_argument(
        "--ssim-mode",
        choices=SSIM_MODES,
        default=SSIM_MODE_OFF,
        help=(
            "Enable geometry-aware local mesh SSIM and area-weighted "
            "relative-L2 outputs (default: off)."
        ),
    )
    parser.add_argument("--ssim-data-range", type=float, default=None)
    parser.add_argument(
        "--ssim-window-size", type=int, default=SSIM_WINDOW_SIZE
    )
    parser.add_argument(
        "--ssim-window-sigma", type=float, default=SSIM_SIGMA_CELLS
    )
    parser.add_argument("--ssim-reference-spacing", type=float, default=None)
    parser.add_argument(
        "--trajectories",
        nargs="+",
        type=int,
        default=None,
        help="Trajectory IDs to compare; defaults to all IDs shared by every model.",
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=None,
        help="Number of steps to compare; defaults to the shortest saved rollout.",
    )
    _add_boolean_optional_argument(
        parser,
        "--trajectory-plots",
        default=True,
        help="Write overlaid diagnostics for each selected trajectory.",
    )
    _add_boolean_optional_argument(
        parser,
        "--training-history",
        default=True,
        help="Compare history.json files when they are available.",
    )
    _add_boolean_optional_argument(
        parser,
        "--iqr",
        default=True,
        help="Show interquartile bands in aggregate time-history panels.",
    )
    _add_boolean_optional_argument(
        parser,
        "--persistence",
        default=True,
        help="Include persistence in the local-mesh fidelity figure.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "pdf", "svg"),
        default=("png",),
        help="Figure format(s) to write (default: png).",
    )
    parser.add_argument(
        "--history-yscale",
        choices=("log", "linear"),
        default="log",
    )
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
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
    formats = list(dict.fromkeys(args.formats))

    rollouts = {
        spec.label: load_rollout(spec.prediction_path) for spec in specs
    }
    trajectory_ids = common_trajectory_ids(rollouts, args.trajectories)
    num_times = validate_common_rollouts(
        rollouts, trajectory_ids, args.rollout_steps
    )
    reference = rollouts[labels[0]]
    times = reference.time[:num_times]
    dataset_path: Optional[Path] = None
    dataset_path_source: Optional[str] = None
    fidelity_info: Optional[Dict[str, object]] = None
    local_windows: Dict[int, LocalMeshWindow] = {}
    ssim_data_range: Optional[float] = None
    if args.ssim_mode == SSIM_MODE_LOCAL_MESH:
        if args.ssim_window_size < 3 or args.ssim_window_size % 2 == 0:
            raise ValueError("--ssim-window-size must be an odd integer >= 3.")
        if (
            not np.isfinite(args.ssim_window_sigma)
            or args.ssim_window_sigma <= 0.0
        ):
            raise ValueError("--ssim-window-sigma must be positive and finite.")
        dataset_path, dataset_path_source = infer_dataset_path(
            specs, explicit_path=args.data
        )
        geometries = load_mesh_geometries(
            dataset_path,
            trajectory_ids,
            expected_num_nodes=reference.truth.shape[2],
        )
        reference_spacing, reference_spacing_source = (
            resolve_ssim_reference_spacing(
                geometries,
                explicit_spacing=args.ssim_reference_spacing,
            )
        )
        ssim_data_range, data_range_source = resolve_ssim_data_range(
            rollouts,
            trajectory_ids,
            num_times=num_times,
            explicit_range=args.ssim_data_range,
            dataset_path=dataset_path,
        )
        local_windows = {
            trajectory_id: build_local_mesh_window(
                geometry.centers,
                geometry.cell_areas,
                reference_spacing=reference_spacing,
                window_size=args.ssim_window_size,
                sigma_cells=args.ssim_window_sigma,
            )
            for trajectory_id, geometry in geometries.items()
        }
        fidelity_info = fidelity_metadata(
            data_range=ssim_data_range,
            data_range_source=data_range_source,
            window_size=args.ssim_window_size,
            sigma_cells=args.ssim_window_sigma,
            reference_spacing=reference_spacing,
            reference_spacing_source=reference_spacing_source,
        )
        fidelity_info.update(
            {
                "dataset": str(dataset_path),
                "dataset_path_source": dataset_path_source,
            }
        )
        print(
            "local-mesh SSIM: "
            f"data_range={ssim_data_range:.12g} ({data_range_source}), "
            f"reference_spacing={reference_spacing:.12g} "
            f"({reference_spacing_source}), window={args.ssim_window_size}, "
            f"sigma_cells={args.ssim_window_sigma:g}"
        )
    styles = model_styles(labels)
    metrics_by_model: Dict[str, List[TrajectoryMetrics]] = {
        label: [] for label in labels
    }
    metrics_by_trajectory: Dict[int, Dict[str, TrajectoryMetrics]] = {}
    rows: List[Dict[str, object]] = []
    for trajectory_id in trajectory_ids:
        by_model = {}
        for label in labels:
            metrics = compute_metrics(
                rollouts[label],
                trajectory_id,
                num_times=num_times,
                local_window=local_windows.get(trajectory_id),
                ssim_data_range=ssim_data_range,
            )
            metrics_by_model[label].append(metrics)
            by_model[label] = metrics
            rows.append(metric_row(label, trajectory_id, num_times - 1, metrics))
        metrics_by_trajectory[trajectory_id] = by_model

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    written_figures: List[Path] = []
    written_figures.extend(
        plot_rollout_summary(
            metrics_by_model=metrics_by_model,
            styles=styles,
            times=times,
            output_base=output_dir / "rollout_comparison",
            formats=formats,
            dpi=args.dpi,
            show_iqr=args.iqr,
        )
    )
    written_figures.extend(
        plot_scalar_scores(
            metrics_by_model=metrics_by_model,
            styles=styles,
            output_base=output_dir / "scalar_score_comparison",
            formats=formats,
            dpi=args.dpi,
        )
    )
    fidelity_npz_path: Optional[Path] = None
    if fidelity_info is not None:
        written_figures.extend(
            plot_fidelity_summary(
                metrics_by_model=metrics_by_model,
                styles=styles,
                times=times,
                output_base=output_dir / "fidelity_comparison",
                formats=formats,
                dpi=args.dpi,
                show_iqr=args.iqr,
                show_persistence=args.persistence,
            )
        )
        fidelity_npz_path = output_dir / "comparison_fidelity_metrics.npz"
        write_fidelity_npz(
            fidelity_npz_path,
            labels=labels,
            trajectory_ids=trajectory_ids,
            times=times,
            metrics_by_model=metrics_by_model,
            metadata=fidelity_info,
        )
    if args.trajectory_plots:
        for trajectory_id in trajectory_ids:
            written_figures.extend(
                plot_trajectory_diagnostics(
                    trajectory_id=trajectory_id,
                    metrics_by_model=metrics_by_trajectory[trajectory_id],
                    styles=styles,
                    times=times,
                    output_base=(
                        output_dir
                        / "trajectories"
                        / f"trajectory_{trajectory_id:03d}_comparison"
                    ),
                    formats=formats,
                    dpi=args.dpi,
                )
            )

    histories = {}
    if args.training_history:
        for spec in specs:
            if spec.history_path is None:
                print(f"warning: no history.json found for {spec.label}")
                continue
            histories[spec.label] = load_history(spec.history_path)
        if histories:
            written_figures.extend(
                plot_training_histories(
                    histories=histories,
                    styles=styles,
                    output_base=output_dir / "training_history_comparison",
                    formats=formats,
                    dpi=args.dpi,
                    yscale=args.history_yscale,
                )
            )

    csv_path = output_dir / "comparison_metrics.csv"
    write_csv(csv_path, rows)
    model_summary = {
        label: summarize_model(metrics_by_model[label]) for label in labels
    }
    ranking = sorted(
        labels, key=lambda label: model_summary[label]["mean_rollout_rmse"]
    )
    summary = {
        "models": {
            spec.label: {
                "run_root": str(spec.run_root),
                "rollout_predictions": str(spec.prediction_path),
                "history": str(spec.history_path) if spec.history_path else None,
            }
            for spec in specs
        },
        "trajectory_ids": trajectory_ids,
        "num_trajectories": len(trajectory_ids),
        "rollout_steps": num_times - 1,
        "time_start": float(times[0]),
        "time_end": float(times[-1]),
        "ranking_by_mean_rollout_rmse": ranking,
        "metrics": model_summary,
    }
    if fidelity_info is not None:
        summary["fidelity"] = fidelity_info
        summary["fidelity"]["raw_metrics"] = str(fidelity_npz_path)
    json_path = output_dir / "comparison_summary.json"
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")

    print(
        f"compared {len(labels)} models on {len(trajectory_ids)} shared "
        f"trajectories and {num_times - 1} rollout steps"
    )
    print("ranking by mean rollout RMSE:")
    for rank, label in enumerate(ranking, start=1):
        metrics = model_summary[label]
        print(
            f"  {rank}. {label}: mean RMSE={metrics['mean_rollout_rmse']:.6e}, "
            f"median RMSE={metrics['median_rollout_rmse']:.6e}, "
            "fraction beating persistence="
            f"{metrics['fraction_beating_persistence']:.1%}"
        )
    print(f"wrote metrics: {csv_path}")
    print(f"wrote summary: {json_path}")
    if fidelity_npz_path is not None:
        print(f"wrote raw fidelity metrics: {fidelity_npz_path}")
    print("wrote figures:")
    for path in written_figures:
        print(f"  {path}")


if __name__ == "__main__":
    main()
