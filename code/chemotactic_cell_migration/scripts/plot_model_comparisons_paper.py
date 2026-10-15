#!/usr/bin/env python3
"""Create Figure-6-style rollout-error plots for chemotaxis models.

The metric calculations come from ``plot_model_comparisons.py`` so the
general comparison and paper plotting entry points use exactly the same
cell-area-weighted relative L2 and local two-dimensional mesh SSIM.

For a Figure-6-style two-panel comparison::

    python scripts/plot_model_comparisons_paper.py \
        --model "FluxGraphNet=path/to/fluxgraphnet/run" \
        --model "MeshGraphNet=path/to/meshgraphnet/run" \
        --model "SAGEConv=path/to/sageconv/run" \
        --baseline-schemes fv-upwind fv-muscl \
        --baseline-step-mode frame-step \
        --error-curves-metrics rel_l2 SSIM \
        --output-dir output/chemotaxis_figure6

The default nine-step centered moving average matches the Figure 6 caption.
Smoothing is display-only; the saved NPZ always contains raw curves.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

_mpl_cache = Path(tempfile.gettempdir()) / "chemotaxis-matplotlib-cache"
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))
os.environ.setdefault("XDG_CACHE_HOME", str(_mpl_cache))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import is_color_like
import numpy as np

from generate_chemotactic_cell_migration_data import (
    advance_drift_diffusion_ssprk2,
    face_drift_velocity,
    least_squares_inverse_moments,
    stable_explicit_timestep,
)

from plot_model_comparisons import (
    SSIM_K1,
    SSIM_K2,
    SSIM_SIGMA_CELLS,
    SSIM_WINDOW_SIZE,
    LocalMeshWindow,
    ModelSpec,
    RolloutData,
    TrajectoryMetrics,
    build_local_mesh_window,
    common_trajectory_ids,
    compute_metrics,
    fidelity_metadata,
    infer_dataset_path,
    load_mesh_geometries,
    load_rollout,
    load_source_dataset,
    parse_model_spec,
    resolve_ssim_data_range,
    resolve_ssim_reference_spacing,
    save_figure,
    validate_common_rollouts,
)


ERROR_CURVE_METRIC_RMSE = "rmse"
ERROR_CURVE_METRIC_REL_L2 = "rel_l2"
ERROR_CURVE_METRIC_L1 = "l1"
ERROR_CURVE_METRIC_SSIM = "ssim"
ERROR_CURVE_METRICS = (
    ERROR_CURVE_METRIC_RMSE,
    ERROR_CURVE_METRIC_REL_L2,
    ERROR_CURVE_METRIC_L1,
    ERROR_CURVE_METRIC_SSIM,
)
ERROR_CURVE_METRIC_ALIASES = {
    "rmse": ERROR_CURVE_METRIC_RMSE,
    "rootmeansquareerror": ERROR_CURVE_METRIC_RMSE,
    "relativel2": ERROR_CURVE_METRIC_REL_L2,
    "rell2": ERROR_CURVE_METRIC_REL_L2,
    "l2rel": ERROR_CURVE_METRIC_REL_L2,
    "l1": ERROR_CURVE_METRIC_L1,
    "mae": ERROR_CURVE_METRIC_L1,
    "meanabsoluteerror": ERROR_CURVE_METRIC_L1,
    "ssim": ERROR_CURVE_METRIC_SSIM,
    "localssim": ERROR_CURVE_METRIC_SSIM,
}
ERROR_CURVE_METRIC_DISPLAY = {
    ERROR_CURVE_METRIC_RMSE: "RMSE",
    ERROR_CURVE_METRIC_REL_L2: r"rel-$L_2$",
    ERROR_CURVE_METRIC_L1: "L1",
    ERROR_CURVE_METRIC_SSIM: "Local SSIM",
}
ERROR_CURVE_METRIC_ATTRIBUTE = {
    ERROR_CURVE_METRIC_RMSE: "rmse_by_time",
    ERROR_CURVE_METRIC_REL_L2: "area_weighted_relative_l2_by_time",
    ERROR_CURVE_METRIC_L1: "mae_by_time",
    ERROR_CURVE_METRIC_SSIM: "local_ssim_by_time",
}
PERSISTENCE_METRIC_ATTRIBUTE = {
    ERROR_CURVE_METRIC_RMSE: "persistence_rmse_by_time",
    ERROR_CURVE_METRIC_REL_L2: (
        "persistence_area_weighted_relative_l2_by_time"
    ),
    ERROR_CURVE_METRIC_L1: "persistence_mae_by_time",
    ERROR_CURVE_METRIC_SSIM: "persistence_local_ssim_by_time",
}

FIGURE6_FAMILY_STYLES = {
    "fluxgraphnet": {"color": "tab:blue", "linestyle": "-"},
    "fvupwind": {"color": "tab:green", "linestyle": "--"},
    "fvmuscl": {"color": "black", "linestyle": "--"},
    "meshgraphnet": {"color": "magenta", "linestyle": "-"},
    "sageconv": {"color": "tab:orange", "linestyle": "-"},
    "persistence": {"color": "0.35", "linestyle": ":"},
}
FALLBACK_COLORS = (
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:olive",
    "tab:cyan",
    "black",
)

BASELINE_SCHEME_FV_UPWIND = "fv-upwind"
BASELINE_SCHEME_FV_MUSCL = "fv-muscl"
BASELINE_SCHEMES = (BASELINE_SCHEME_FV_UPWIND, BASELINE_SCHEME_FV_MUSCL)
BASELINE_STEP_MODE_CFL_SUBCYCLE = "cfl-subcycle"
BASELINE_STEP_MODE_FRAME_STEP = "frame-step"
BASELINE_STEP_MODES = (
    BASELINE_STEP_MODE_CFL_SUBCYCLE,
    BASELINE_STEP_MODE_FRAME_STEP,
)
BASELINE_LABELS = {
    BASELINE_SCHEME_FV_UPWIND: "FV-Upwind",
    BASELINE_SCHEME_FV_MUSCL: "FV-MUSCL",
}


@dataclass(frozen=True)
class CoarseFiniteVolumeInputs:
    """Dataset fields needed to evolve coarse numerical baselines."""

    trajectory_ids: np.ndarray
    areas: np.ndarray
    undirected_edges: np.ndarray
    shared_face_lengths: np.ndarray
    centers: np.ndarray
    face_midpoints: np.ndarray
    face_normals: np.ndarray
    center_normal_distances: np.ndarray
    drift_speed: np.ndarray
    drift_speed_source: str
    diffusion: float
    output_dt: float
    cfl: float
    stable_dt: float
    step_mode: str
    cfl_required_substeps_per_frame: int
    substeps_per_frame: int
    solver_dt: float
    num_dataset_trajectories: int
    reference_num_cells: Optional[int]


def _normalized_token(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def resolve_error_curve_metrics(
    raw_metrics: Optional[Sequence[str]],
) -> List[str]:
    """Normalize metric aliases while preserving the requested panel order."""
    raw_values = ["rel_l2", "ssim"] if raw_metrics is None else raw_metrics
    selected: List[str] = []
    saw_all = False
    for raw_item in raw_values:
        for part in str(raw_item).split(","):
            token = part.strip()
            if not token:
                continue
            normalized = _normalized_token(token)
            if normalized == "all":
                if selected:
                    raise ValueError(
                        "--error-curves-metrics all cannot be combined with "
                        "specific metrics."
                    )
                saw_all = True
                continue
            if saw_all:
                raise ValueError(
                    "--error-curves-metrics all cannot be combined with "
                    "specific metrics."
                )
            metric = ERROR_CURVE_METRIC_ALIASES.get(normalized)
            if metric is None:
                raise ValueError(
                    f"Unknown error-curve metric {part!r}; expected one of "
                    "rmse, rel_l2, l1, ssim, or all."
                )
            if metric not in selected:
                selected.append(metric)
    if saw_all:
        return list(ERROR_CURVE_METRICS)
    if not selected:
        raise ValueError("At least one error-curve metric must be requested.")
    return selected


def centered_moving_average(curve: np.ndarray, window: int) -> np.ndarray:
    """Return a NaN-aware centered average with truncated endpoint windows."""
    values = np.asarray(curve, dtype=np.float64).reshape(-1)
    if window <= 0 or window % 2 == 0:
        raise ValueError("The smoothing window must be a positive odd integer.")
    if window == 1 or values.size == 0:
        return values.copy()
    radius = window // 2
    smoothed = np.full(values.shape, np.nan, dtype=np.float64)
    for index in range(values.size):
        local = values[
            max(0, index - radius) : min(values.size, index + radius + 1)
        ]
        finite = local[np.isfinite(local)]
        if finite.size:
            smoothed[index] = float(np.mean(finite))
    return smoothed


def smooth_curves(curves: np.ndarray, window: int) -> np.ndarray:
    """Apply the centered moving average to a curve matrix."""
    values = np.asarray(curves, dtype=np.float64)
    if values.ndim == 1:
        return centered_moving_average(values, window)
    if values.ndim == 2:
        return np.stack(
            [centered_moving_average(row, window) for row in values], axis=0
        )
    raise ValueError(f"Expected one- or two-dimensional curves, got {values.shape}.")


def model_family(label: str) -> Optional[str]:
    """Recognize model families carrying fixed colors in Figure 6."""
    key = _normalized_token(label)
    if "fluxgraphnet" in key:
        return "fluxgraphnet"
    if "fvupwind" in key or "godunov" in key:
        return "fvupwind"
    if "fvmuscl" in key:
        return "fvmuscl"
    if "meshgraphnet" in key:
        return "meshgraphnet"
    if "sageconv" in key:
        return "sageconv"
    if "persistence" in key:
        return "persistence"
    return None


def parse_curve_color_overrides(
    raw_values: Optional[Sequence[str]],
) -> Dict[str, str]:
    """Parse optional LABEL=COLOR settings."""
    overrides: Dict[str, str] = {}
    for raw_item in raw_values or ():
        for part in str(raw_item).split(","):
            token = part.strip()
            if not token:
                continue
            if "=" not in token:
                raise ValueError(
                    f"Invalid --curve-color value {token!r}; expected LABEL=COLOR."
                )
            label, color = token.split("=", 1)
            key = _normalized_token(label)
            color = color.strip()
            if not key or not color or not is_color_like(color):
                raise ValueError(
                    f"Invalid --curve-color value {token!r}; {color!r} is not "
                    "a Matplotlib color."
                )
            overrides[key] = color
    return overrides


def figure6_model_styles(
    labels: Sequence[str],
    *,
    color_overrides: Optional[Mapping[str, str]] = None,
) -> Dict[str, Dict[str, object]]:
    """Assign semantic Figure 6 colors and stable fallbacks."""
    overrides = dict(color_overrides or {})
    used_families = set()
    fallback_index = 0
    styles: Dict[str, Dict[str, object]] = {}
    for label in labels:
        family = model_family(label)
        if family is not None and family not in used_families:
            style = dict(FIGURE6_FAMILY_STYLES[family])
            used_families.add(family)
        else:
            style = {
                "color": FALLBACK_COLORS[fallback_index % len(FALLBACK_COLORS)],
                "linestyle": "-",
            }
            fallback_index += 1
        override = overrides.get(_normalized_token(label))
        if override is None and family is not None:
            override = overrides.get(family)
        if override is not None:
            style["color"] = override
        style["linewidth"] = 2.0
        styles[label] = style
    return styles


def load_coarse_finite_volume_inputs(
    dataset_path: Path,
    trajectory_ids: Sequence[int],
    *,
    times: np.ndarray,
    explicit_cfl: Optional[float],
    step_mode: str = BASELINE_STEP_MODE_CFL_SUBCYCLE,
) -> CoarseFiniteVolumeInputs:
    """Load and validate the geometry and physics used by coarse baselines."""
    required = (
        "cell_areas",
        "undirected_edge_index",
        "shared_face_lengths",
        "cell_centers",
        "shared_face_midpoints",
        "shared_face_normals",
        "center_normal_distances",
        "diffusion",
        "dt",
    )
    archive = load_source_dataset(dataset_path)
    missing = [name for name in required if name not in archive]
    if missing:
        raise KeyError(
            f"{dataset_path} is missing baseline solver array(s): "
            + ", ".join(missing)
        )
    full_areas = np.asarray(archive["cell_areas"], dtype=np.float64)
    full_edges = np.asarray(archive["undirected_edge_index"], dtype=np.int64)
    full_lengths = np.asarray(archive["shared_face_lengths"], dtype=np.float64)
    full_centers = np.asarray(archive["cell_centers"], dtype=np.float64)
    full_midpoints = np.asarray(
        archive["shared_face_midpoints"], dtype=np.float64
    )
    full_normals = np.asarray(archive["shared_face_normals"], dtype=np.float64)
    full_normal_distances = np.asarray(
        archive["center_normal_distances"], dtype=np.float64
    )
    if "face_drift_speed" in archive:
        full_drift_speed = np.asarray(archive["face_drift_speed"], dtype=np.float64)
        if full_drift_speed.ndim == 3 and full_drift_speed.shape[-1] == 1:
            full_drift_speed = full_drift_speed[..., 0]
        drift_speed_source = "stored_face_drift_speed"
    else:
        source_keys = (
            "chemo_source_count",
            "chemo_source_centers",
            "chemo_source_amplitudes",
            "chemo_source_sigmas",
        )
        chi_key = (
            "trajectory_chi"
            if "trajectory_chi" in archive
            else "chi" if "chi" in archive else None
        )
        if chi_key is not None and all(key in archive for key in source_keys):
            chemo_metadata = {
                key: np.asarray(archive[key]).copy() for key in source_keys
            }
            full_drift_speed = face_drift_velocity(
                face_midpoints=full_midpoints,
                face_normals=full_normals,
                chemo_metadata=chemo_metadata,
                chi=np.asarray(archive[chi_key], dtype=np.float64),
            )
            drift_speed_source = (
                "reconstructed_exactly_from_gaussian_sources_and_" + chi_key
            )
        elif "drift_velocity" in archive:
            velocity = np.asarray(archive["drift_velocity"], dtype=np.float64)
            if velocity.ndim != 3 or velocity.shape[2] != 2:
                raise ValueError(
                    "drift_velocity must have shape [trajectory, node, 2]."
                )
            trajectory = np.arange(full_edges.shape[0])[:, None]
            source_velocity = velocity[trajectory, full_edges[:, 0]]
            target_velocity = velocity[trajectory, full_edges[:, 1]]
            full_drift_speed = np.sum(
                0.5 * (source_velocity + target_velocity) * full_normals,
                axis=2,
            )
            drift_speed_source = "interpolated_cell_center_drift_velocity"
        else:
            raise KeyError(
                f"{dataset_path} has no face_drift_speed and lacks the "
                "Gaussian-source or cell-velocity fields needed to reconstruct it."
            )
    diffusion = float(np.asarray(archive["diffusion"]).item())
    output_dt = float(np.asarray(archive["dt"]).item())
    if explicit_cfl is None:
        if "CFL" not in archive:
            raise KeyError(
                f"{dataset_path} has no CFL metadata; pass --baseline-cfl."
            )
        cfl = float(np.asarray(archive["CFL"]).item())
    else:
        cfl = float(explicit_cfl)
    boundary = (
        str(np.asarray(archive["boundary"]).item())
        if "boundary" in archive
        else "no_flux"
    )
    reference_num_cells = (
        int(np.asarray(archive["reference_num_cells"]).item())
        if "reference_num_cells" in archive
        else None
    )

    if boundary != "no_flux":
        raise ValueError(
            f"Coarse chemotaxis baselines require boundary='no_flux', got "
            f"{boundary!r}."
        )
    if not np.isfinite(diffusion) or diffusion < 0.0:
        raise ValueError("Dataset diffusion must be nonnegative and finite.")
    if not np.isfinite(output_dt) or output_dt <= 0.0:
        raise ValueError("Dataset dt must be positive and finite.")
    if not np.isfinite(cfl) or not (0.0 < cfl <= 1.0):
        raise ValueError("Baseline CFL must be finite and in (0, 1].")
    if step_mode not in BASELINE_STEP_MODES:
        raise ValueError(
            f"Unknown baseline step mode {step_mode!r}; expected one of "
            f"{BASELINE_STEP_MODES}."
        )

    if full_areas.ndim != 2:
        raise ValueError("cell_areas must have shape [trajectory, node].")
    num_trajectories, num_nodes = full_areas.shape
    expected_shapes = {
        "cell_centers": (num_trajectories, num_nodes, 2),
        "undirected_edge_index_prefix": (num_trajectories, 2),
    }
    if full_centers.shape != expected_shapes["cell_centers"]:
        raise ValueError(
            "cell_centers must have shape [trajectory, node, 2] matching areas."
        )
    if full_edges.ndim != 3 or full_edges.shape[:2] != expected_shapes[
        "undirected_edge_index_prefix"
    ]:
        raise ValueError(
            "undirected_edge_index must have shape [trajectory, 2, face]."
        )
    num_faces = full_edges.shape[2]
    if full_lengths.shape != (num_trajectories, num_faces):
        raise ValueError("shared_face_lengths must have shape [trajectory, face].")
    if full_midpoints.shape != (num_trajectories, num_faces, 2):
        raise ValueError(
            "shared_face_midpoints must have shape [trajectory, face, 2]."
        )
    if full_normals.shape != (num_trajectories, num_faces, 2):
        raise ValueError(
            "shared_face_normals must have shape [trajectory, face, 2]."
        )
    if full_normal_distances.shape != (num_trajectories, num_faces):
        raise ValueError(
            "center_normal_distances must have shape [trajectory, face]."
        )
    if full_drift_speed.shape != (num_trajectories, num_faces):
        raise ValueError("face_drift_speed must have shape [trajectory, face, 1].")
    finite_arrays = (
        full_areas,
        full_centers,
        full_lengths,
        full_midpoints,
        full_normals,
        full_normal_distances,
        full_drift_speed,
    )
    if not all(np.all(np.isfinite(array)) for array in finite_arrays):
        raise ValueError("Coarse baseline geometry and physics must be finite.")
    if (
        np.any(full_areas <= 0.0)
        or np.any(full_lengths <= 0.0)
        or np.any(full_normal_distances <= 0.0)
    ):
        raise ValueError(
            "Cell areas, face lengths, and center-normal distances must be positive."
        )
    if np.any(full_edges < 0) or np.any(full_edges >= num_nodes):
        raise ValueError("undirected_edge_index contains an invalid node index.")

    selected_ids = np.asarray(trajectory_ids, dtype=np.int64)
    if selected_ids.ndim != 1 or selected_ids.size == 0:
        raise ValueError("At least one baseline trajectory is required.")
    if np.any(selected_ids < 0) or np.any(selected_ids >= num_trajectories):
        raise ValueError("A selected baseline trajectory does not exist in the dataset.")
    rollout_times = np.asarray(times, dtype=np.float64)
    if rollout_times.ndim != 1 or rollout_times.size < 2:
        raise ValueError("Baseline rollout times must contain at least two entries.")
    time_steps = np.diff(rollout_times)
    if not np.allclose(time_steps, output_dt, rtol=1.0e-8, atol=1.0e-12):
        raise ValueError(
            "Rollout times do not match the source dataset's stored-frame dt."
        )

    stable_dt, _ = stable_explicit_timestep(
        areas=full_areas,
        undirected_edges=full_edges,
        shared_face_lengths=full_lengths,
        center_normal_distances=full_normal_distances,
        drift_speed=full_drift_speed,
        diffusion=diffusion,
        cfl=cfl,
    )
    cfl_required_substeps_per_frame = max(
        1, int(np.ceil(output_dt / stable_dt - 1.0e-12))
    )
    if step_mode == BASELINE_STEP_MODE_FRAME_STEP:
        substeps_per_frame = 1
    else:
        substeps_per_frame = cfl_required_substeps_per_frame
    solver_dt = output_dt / substeps_per_frame
    if (
        step_mode == BASELINE_STEP_MODE_CFL_SUBCYCLE
        and solver_dt > stable_dt * (1.0 + 1.0e-12)
    ):
        raise RuntimeError("Resolved baseline timestep exceeds its stability limit.")
    return CoarseFiniteVolumeInputs(
        trajectory_ids=selected_ids.copy(),
        areas=full_areas[selected_ids].copy(),
        undirected_edges=full_edges[selected_ids].copy(),
        shared_face_lengths=full_lengths[selected_ids].copy(),
        centers=full_centers[selected_ids].copy(),
        face_midpoints=full_midpoints[selected_ids].copy(),
        face_normals=full_normals[selected_ids].copy(),
        center_normal_distances=full_normal_distances[selected_ids].copy(),
        drift_speed=full_drift_speed[selected_ids].copy(),
        drift_speed_source=drift_speed_source,
        diffusion=diffusion,
        output_dt=output_dt,
        cfl=cfl,
        stable_dt=float(stable_dt),
        step_mode=step_mode,
        cfl_required_substeps_per_frame=cfl_required_substeps_per_frame,
        substeps_per_frame=substeps_per_frame,
        solver_dt=float(solver_dt),
        num_dataset_trajectories=num_trajectories,
        reference_num_cells=reference_num_cells,
    )


def _mass_change_from_face_flux(
    face_mass_flux: np.ndarray,
    *,
    undirected_edges: np.ndarray,
    num_nodes: int,
) -> np.ndarray:
    """Apply oriented interior-face fluxes with equal and opposite signs."""
    num_trajectories = face_mass_flux.shape[0]
    offsets = np.arange(num_trajectories, dtype=np.int64)[:, None] * num_nodes
    cell_i = offsets + undirected_edges[:, 0]
    cell_j = offsets + undirected_edges[:, 1]
    mass_change = np.zeros(num_trajectories * num_nodes, dtype=np.float64)
    np.add.at(mass_change, cell_i.ravel(), -face_mass_flux.ravel())
    np.add.at(mass_change, cell_j.ravel(), face_mass_flux.ravel())
    return mass_change.reshape(num_trajectories, num_nodes)


def fv_upwind_euler_step(
    density: np.ndarray,
    *,
    dt: float,
    inputs: CoarseFiniteVolumeInputs,
) -> np.ndarray:
    """Take one monotone first-order drift-diffusion Euler step."""
    num_trajectories, num_nodes = density.shape
    cell_i = inputs.undirected_edges[:, 0]
    cell_j = inputs.undirected_edges[:, 1]
    sample = np.arange(num_trajectories, dtype=np.int64)[:, None]
    density_i = density[sample, cell_i]
    density_j = density[sample, cell_j]
    upwind_density = np.where(
        inputs.drift_speed >= 0.0, density_i, density_j
    )
    normal_gradient = (
        density_j - density_i
    ) / inputs.center_normal_distances
    face_mass_flux = inputs.shared_face_lengths * (
        inputs.drift_speed * upwind_density
        - inputs.diffusion * normal_gradient
    )
    mass_change = _mass_change_from_face_flux(
        face_mass_flux,
        undirected_edges=inputs.undirected_edges,
        num_nodes=num_nodes,
    )
    result = density + float(dt) * mass_change / inputs.areas
    if np.min(result) < -1.0e-12:
        raise RuntimeError(
            "FV-Upwind Euler stage became negative; reduce --baseline-cfl."
        )
    return result


def advance_fv_upwind_ssprk2(
    density: np.ndarray,
    *,
    dt: float,
    inputs: CoarseFiniteVolumeInputs,
) -> np.ndarray:
    """Advance the first-order spatial baseline with SSP-RK2."""
    stage_one = fv_upwind_euler_step(density, dt=dt, inputs=inputs)
    stage_two_euler = fv_upwind_euler_step(
        stage_one, dt=dt, inputs=inputs
    )
    result = 0.5 * density + 0.5 * stage_two_euler
    if np.min(result) < -1.0e-10:
        raise RuntimeError(
            "FV-Upwind SSP-RK2 state became negative; reduce --baseline-cfl."
        )
    return result


def generate_coarse_baseline_prediction(
    initial_density: np.ndarray,
    *,
    num_times: int,
    inputs: CoarseFiniteVolumeInputs,
    scheme: str,
) -> np.ndarray:
    """Generate a non-leaking coarse numerical rollout from time zero only."""
    if scheme not in BASELINE_SCHEMES:
        raise ValueError(f"Unknown baseline scheme {scheme!r}.")
    initial = np.asarray(initial_density, dtype=np.float64)
    if initial.shape != inputs.areas.shape:
        raise ValueError(
            "Baseline initial density must have shape [trajectory, node] "
            "matching the selected coarse meshes."
        )
    if not np.all(np.isfinite(initial)):
        raise ValueError("Baseline initial density must be finite.")
    if np.min(initial) < -1.0e-12:
        raise ValueError("Baseline initial density must be nonnegative.")
    if num_times < 2:
        raise ValueError("A numerical baseline requires at least one future time.")

    inverse_moments = (
        least_squares_inverse_moments(
            centers=inputs.centers,
            undirected_edges=inputs.undirected_edges,
        )
        if scheme == BASELINE_SCHEME_FV_MUSCL
        else None
    )
    predictions = np.empty(
        (initial.shape[0], num_times, initial.shape[1]), dtype=np.float64
    )
    predictions[:, 0] = initial
    state = initial.copy()
    muscl_kwargs = {
        "areas": inputs.areas,
        "undirected_edges": inputs.undirected_edges,
        "shared_face_lengths": inputs.shared_face_lengths,
        "centers": inputs.centers,
        "face_midpoints": inputs.face_midpoints,
        "face_normals": inputs.face_normals,
        "center_normal_distances": inputs.center_normal_distances,
        "drift_speed": inputs.drift_speed,
        "diffusion": inputs.diffusion,
        "inverse_moments": inverse_moments,
    }
    for frame in range(1, num_times):
        try:
            for _ in range(inputs.substeps_per_frame):
                if scheme == BASELINE_SCHEME_FV_UPWIND:
                    state = advance_fv_upwind_ssprk2(
                        state, dt=inputs.solver_dt, inputs=inputs
                    )
                else:
                    state, _ = advance_drift_diffusion_ssprk2(
                        state, dt=inputs.solver_dt, **muscl_kwargs
                    )
        except RuntimeError as error:
            if inputs.step_mode == BASELINE_STEP_MODE_FRAME_STEP:
                if inputs.solver_dt > inputs.stable_dt * (1.0 + 1.0e-12):
                    guidance = (
                        "The frame timestep exceeds the configured CFL "
                        "stability limit; use --baseline-step-mode "
                        "cfl-subcycle for the stability-controlled calculation."
                    )
                else:
                    guidance = (
                        "Use --baseline-step-mode cfl-subcycle to retry with "
                        "stability-controlled timesteps."
                    )
            else:
                guidance = "Try reducing --baseline-cfl."
            raise RuntimeError(
                f"{BASELINE_LABELS[scheme]} failed at output interval {frame} "
                f"using --baseline-step-mode {inputs.step_mode} "
                f"(dt={inputs.solver_dt:.12g}, CFL stability limit="
                f"{inputs.stable_dt:.12g}). {guidance}"
            ) from error
        predictions[:, frame] = state
    if not np.all(np.isfinite(predictions)):
        raise FloatingPointError(
            f"{BASELINE_LABELS[scheme]} produced non-finite predictions."
        )
    return predictions


def build_numerical_baseline_rollouts(
    dataset_path: Path,
    trajectory_ids: Sequence[int],
    *,
    reference: RolloutData,
    num_times: int,
    schemes: Sequence[str],
    explicit_cfl: Optional[float],
    step_mode: str = BASELINE_STEP_MODE_CFL_SUBCYCLE,
) -> Tuple[Dict[str, RolloutData], Dict[str, Dict[str, object]]]:
    """Generate requested coarse FV baselines and reproducibility metadata."""
    selected_schemes = list(dict.fromkeys(str(value) for value in schemes))
    if not selected_schemes:
        return {}, {}
    unknown = [value for value in selected_schemes if value not in BASELINE_SCHEMES]
    if unknown:
        raise ValueError("Unknown baseline scheme(s): " + ", ".join(unknown))
    inputs = load_coarse_finite_volume_inputs(
        dataset_path,
        trajectory_ids,
        times=reference.time[:num_times],
        explicit_cfl=explicit_cfl,
        step_mode=step_mode,
    )
    truth = np.stack(
        [
            reference.truth[
                int(np.flatnonzero(reference.trajectory_ids == trajectory_id)[0]),
                :num_times,
            ]
            for trajectory_id in trajectory_ids
        ],
        axis=0,
    )
    initial = truth[:, 0]
    persistence = np.broadcast_to(truth[:, :1], truth.shape).copy()
    truth_mass = np.sum(truth * inputs.areas[:, None, :], axis=2)
    baseline_rollouts: Dict[str, RolloutData] = {}
    baseline_metadata: Dict[str, Dict[str, object]] = {}
    timestep_to_stability_limit_ratio = inputs.solver_dt / inputs.stable_dt
    effective_cfl = inputs.cfl * timestep_to_stability_limit_ratio
    stability_limit_satisfied = bool(
        inputs.solver_dt <= inputs.stable_dt * (1.0 + 1.0e-12)
    )
    if (
        inputs.step_mode == BASELINE_STEP_MODE_FRAME_STEP
        and not stability_limit_satisfied
    ):
        print(
            "WARNING: --baseline-step-mode frame-step uses one numerical "
            f"update per stored interval (dt={inputs.solver_dt:.12g}), which "
            f"is {timestep_to_stability_limit_ratio:.6g} times the configured "
            f"CFL stability limit ({inputs.stable_dt:.12g}); effective "
            f"CFL={effective_cfl:.6g}. The baseline may become unstable or "
            "fail its positivity checks.",
            flush=True,
        )
    for scheme in selected_schemes:
        label = BASELINE_LABELS[scheme]
        calculation_start = time.perf_counter()
        prediction = generate_coarse_baseline_prediction(
            initial,
            num_times=num_times,
            inputs=inputs,
            scheme=scheme,
        )
        calculation_seconds = time.perf_counter() - calculation_start
        seconds_per_trajectory = calculation_seconds / initial.shape[0]
        seconds_per_interval_per_trajectory = calculation_seconds / (
            initial.shape[0] * (num_times - 1)
        )
        print(
            f"Numerical timing: {label}: {calculation_seconds:.6f} s "
            f"for {initial.shape[0]} batched trajectories; "
            f"{seconds_per_trajectory:.6f} s/trajectory; "
            f"{1.0e3 * seconds_per_interval_per_trajectory:.6f} "
            "ms/output interval/trajectory; "
            f"{inputs.substeps_per_frame} substeps/output interval; "
            f"step mode={inputs.step_mode}",
            flush=True,
        )
        prediction_mass = np.sum(
            prediction * inputs.areas[:, None, :], axis=2
        )
        mass_error = prediction_mass - truth_mass
        mass_drift = prediction_mass - prediction_mass[:, :1]
        baseline_rollouts[label] = RolloutData(
            trajectory_ids=np.asarray(trajectory_ids, dtype=np.int64),
            truth=truth.copy(),
            prediction=prediction,
            persistence=persistence.copy(),
            time=reference.time[:num_times].copy(),
            mass_error=mass_error,
            source=dataset_path,
        )
        if scheme == BASELINE_SCHEME_FV_UPWIND:
            spatial_scheme = (
                "first_order_upwind_godunov_equivalent_drift_"
                "two_point_normal_diffusion"
            )
        else:
            spatial_scheme = (
                "barth_jespersen_muscl_face_normal_nonorthogonal_diffusion_"
                "positivity_flux_limited"
            )
        baseline_metadata[label] = {
            "scheme": scheme,
            "spatial_scheme": spatial_scheme,
            "time_integrator": "SSP_RK2",
            "boundary": "no_flux",
            "coarse_num_cells": int(inputs.areas.shape[1]),
            "reference_num_cells": inputs.reference_num_cells,
            "dataset_trajectories_used_for_stability_limit": int(
                inputs.num_dataset_trajectories
            ),
            "cfl": float(inputs.cfl),
            "effective_cfl": float(effective_cfl),
            "stable_timestep_limit": float(inputs.stable_dt),
            "solver_timestep": float(inputs.solver_dt),
            "output_timestep": float(inputs.output_dt),
            "step_mode": inputs.step_mode,
            "cfl_required_substeps_per_output_frame": int(
                inputs.cfl_required_substeps_per_frame
            ),
            "substeps_per_output_frame": int(inputs.substeps_per_frame),
            "timestep_to_stability_limit_ratio": float(
                timestep_to_stability_limit_ratio
            ),
            "stability_limit_satisfied": stability_limit_satisfied,
            "unsafe_frame_step": bool(
                inputs.step_mode == BASELINE_STEP_MODE_FRAME_STEP
                and not stability_limit_satisfied
            ),
            "face_drift_speed_source": inputs.drift_speed_source,
            "initial_condition": "held_out_reference_state_at_time_zero",
            "future_reference_states_used": False,
            "stored_reference_flux_labels_used": False,
            "calculation_seconds": float(calculation_seconds),
            "seconds_per_trajectory": float(seconds_per_trajectory),
            "seconds_per_output_interval_per_trajectory": float(
                seconds_per_interval_per_trajectory
            ),
            "maximum_absolute_total_mass_drift": float(
                np.max(np.abs(mass_drift))
            ),
            "maximum_absolute_mass_error_against_reference": float(
                np.max(np.abs(mass_error))
            ),
        }
    return baseline_rollouts, baseline_metadata


def report_gnn_inference_timings(
    specs: Sequence[ModelSpec],
    rollouts: Mapping[str, RolloutData],
    *,
    trajectory_ids: Sequence[int],
    num_times: int,
) -> Dict[str, Optional[Dict[str, object]]]:
    """Print inference timings recorded when learned rollouts were generated."""
    reported: Dict[str, Optional[Dict[str, object]]] = {}
    for spec in specs:
        timing = rollouts[spec.label].inference_timing
        if timing is None:
            print(
                f"GNN inference timing: {spec.label}: unavailable in "
                f"{spec.prediction_path.name}; rerun plot_gnn_rollouts.py to "
                "record it",
                flush=True,
            )
            reported[spec.label] = None
            continue
        total_seconds = float(timing["total_seconds"])
        num_trajectories = int(timing["num_trajectories"])
        num_intervals = int(timing["num_output_intervals"])
        seconds_per_trajectory = total_seconds / num_trajectories
        seconds_per_interval_per_trajectory = total_seconds / (
            num_trajectories * num_intervals
        )
        comparison_scope_matches = (
            num_trajectories == len(trajectory_ids)
            and num_intervals == num_times - 1
        )
        scope_note = (
            ""
            if comparison_scope_matches
            else (
                f"; current comparison uses {len(trajectory_ids)} trajectories "
                f"and {num_times - 1} output intervals"
            )
        )
        print(
            f"GNN inference timing: {spec.label}: {total_seconds:.6f} s "
            f"for {num_trajectories} trajectories; "
            f"{seconds_per_trajectory:.6f} s/trajectory; "
            f"{1.0e3 * seconds_per_interval_per_trajectory:.6f} "
            "ms/output interval/trajectory; "
            f"device={timing['device']}; batch_size={timing['batch_size']}"
            f"{scope_note}",
            flush=True,
        )
        reported[spec.label] = {
            **timing,
            "seconds_per_trajectory": float(seconds_per_trajectory),
            "seconds_per_output_interval_per_trajectory": float(
                seconds_per_interval_per_trajectory
            ),
            "comparison_scope_matches": comparison_scope_matches,
        }
    return reported


def _metric_matrix(
    metrics: Sequence[TrajectoryMetrics], attribute: str
) -> np.ndarray:
    curves = []
    for item in metrics:
        value = getattr(item, attribute)
        if value is None:
            raise ValueError(f"Metric {attribute} was not computed.")
        curves.append(np.asarray(value, dtype=np.float64))
    return np.stack(curves, axis=0)


def plot_figure6_error_curves(
    *,
    metrics_by_model: Mapping[str, Sequence[TrajectoryMetrics]],
    times: np.ndarray,
    selected_metrics: Sequence[str],
    styles: Mapping[str, Mapping[str, object]],
    output_base: Path,
    formats: Sequence[str],
    dpi: int,
    smooth_window: int,
    show_trajectories: bool,
    spread: str,
    log_y: bool,
    title: str,
    show_persistence: bool,
) -> List[Path]:
    """Plot selectable metrics using the visual language of paper Figure 6."""
    if not selected_metrics or not metrics_by_model:
        raise ValueError("At least one metric and one model are required.")
    panel_count = len(selected_metrics)
    if panel_count == 1:
        n_rows, n_columns, figure_size = 1, 1, (8.0, 4.8)
    elif panel_count == 2:
        n_rows, n_columns, figure_size = 1, 2, (15.5, 4.8)
    else:
        n_rows, n_columns, figure_size = 2, 2, (15.5, 8.3)
    figure, axes_object = plt.subplots(
        n_rows,
        n_columns,
        figsize=figure_size,
        constrained_layout=True,
        squeeze=False,
    )
    axes = list(np.asarray(axes_object, dtype=object).reshape(-1))
    for unused_axis in axes[panel_count:]:
        unused_axis.set_visible(False)

    labels = list(metrics_by_model)
    legend_axis = (
        axes[list(selected_metrics).index(ERROR_CURVE_METRIC_SSIM)]
        if ERROR_CURVE_METRIC_SSIM in selected_metrics
        else axes[panel_count - 1]
    )
    plot_times = np.asarray(times, dtype=np.float64)[1:]
    for panel_index, metric in enumerate(selected_metrics):
        axis = axes[panel_index]
        plotted_means: List[np.ndarray] = []
        for label in labels:
            raw_curves = _metric_matrix(
                metrics_by_model[label], ERROR_CURVE_METRIC_ATTRIBUTE[metric]
            )[:, 1:]
            display_curves = smooth_curves(raw_curves, smooth_window)
            mean_curve = centered_moving_average(
                np.mean(raw_curves, axis=0), smooth_window
            )
            plotted_means.append(mean_curve)
            if show_trajectories:
                for curve in display_curves:
                    axis.plot(
                        plot_times,
                        curve,
                        color=styles[label]["color"],
                        linestyle=styles[label]["linestyle"],
                        linewidth=0.75,
                        alpha=0.16,
                    )
            if spread == "iqr":
                lower, upper = np.quantile(display_curves, (0.25, 0.75), axis=0)
                axis.fill_between(
                    plot_times,
                    lower,
                    upper,
                    color=styles[label]["color"],
                    alpha=0.12,
                    linewidth=0.0,
                )
            elif spread == "std":
                standard_deviation = np.std(display_curves, axis=0)
                axis.fill_between(
                    plot_times,
                    mean_curve - standard_deviation,
                    mean_curve + standard_deviation,
                    color=styles[label]["color"],
                    alpha=0.12,
                    linewidth=0.0,
                )
            axis.plot(plot_times, mean_curve, label=label, **styles[label])

        if show_persistence:
            first_metrics = next(iter(metrics_by_model.values()))
            raw_persistence = _metric_matrix(
                first_metrics, PERSISTENCE_METRIC_ATTRIBUTE[metric]
            )[:, 1:]
            persistence_mean = centered_moving_average(
                np.mean(raw_persistence, axis=0), smooth_window
            )
            plotted_means.append(persistence_mean)
            persistence_style = FIGURE6_FAMILY_STYLES["persistence"]
            axis.plot(
                plot_times,
                persistence_mean,
                label="Persistence",
                linewidth=2.0,
                **persistence_style,
            )

        if n_rows == 1 or panel_index >= (n_rows - 1) * n_columns:
            axis.set_xlabel("Time", fontsize=20)
        axis.set_ylabel(ERROR_CURVE_METRIC_DISPLAY[metric], fontsize=20)
        axis.tick_params(axis="both", labelsize=16)
        axis.grid(alpha=0.3)
        if log_y and metric != ERROR_CURVE_METRIC_SSIM:
            finite_positive = np.concatenate(plotted_means)
            finite_positive = finite_positive[
                np.isfinite(finite_positive) & (finite_positive > 0.0)
            ]
            if finite_positive.size:
                axis.set_yscale("log")
        if metric == ERROR_CURVE_METRIC_SSIM:
            finite_ssim = np.concatenate(plotted_means)
            finite_ssim = finite_ssim[np.isfinite(finite_ssim)]
            if finite_ssim.size:
                minimum = max(-1.0, float(np.min(finite_ssim)))
                span = max(1.0 - minimum, 1.0e-3)
                axis.set_ylim(max(-1.0, minimum - 0.05 * span), 1.0)

    legend_axis.legend(
        loc="lower left" if ERROR_CURVE_METRIC_SSIM in selected_metrics else "best",
        fontsize=16,
    )
    figure.suptitle(title, fontsize=20)
    return save_figure(figure, output_base, formats=formats, dpi=dpi)


def write_raw_metrics(
    path: Path,
    *,
    labels: Sequence[str],
    trajectory_ids: Sequence[int],
    times: np.ndarray,
    selected_metrics: Sequence[str],
    metrics_by_model: Mapping[str, Sequence[TrajectoryMetrics]],
    metadata: Mapping[str, object],
    baseline_rollouts: Mapping[str, RolloutData],
) -> None:
    """Save unsmoothed metric curves and generated numerical trajectories."""
    values = np.stack(
        [
            np.stack(
                [
                    _metric_matrix(
                        metrics_by_model[label],
                        ERROR_CURVE_METRIC_ATTRIBUTE[metric],
                    )
                    for metric in selected_metrics
                ],
                axis=1,
            )
            for label in labels
        ],
        axis=0,
    )
    first_metrics = metrics_by_model[labels[0]]
    persistence_values = np.stack(
        [
            _metric_matrix(
                first_metrics, PERSISTENCE_METRIC_ATTRIBUTE[metric]
            )
            for metric in selected_metrics
        ],
        axis=1,
    )
    arrays: Dict[str, np.ndarray] = {
        "model_labels": np.asarray(labels),
        "trajectory_ids": np.asarray(trajectory_ids, dtype=np.int64),
        "metric_names": np.asarray(selected_metrics),
        "time": np.asarray(times, dtype=np.float64),
        "values": values,
        "persistence_values": persistence_values,
    }
    for key, value in metadata.items():
        arrays[key] = np.asarray(value)
    if baseline_rollouts:
        baseline_labels = list(baseline_rollouts)
        first_baseline = baseline_rollouts[baseline_labels[0]]
        arrays.update(
            {
                "numerical_baseline_labels": np.asarray(baseline_labels),
                "numerical_baseline_truth": first_baseline.truth,
                "numerical_baseline_predictions": np.stack(
                    [
                        baseline_rollouts[label].prediction
                        for label in baseline_labels
                    ],
                    axis=0,
                ),
                "numerical_baseline_mass_error": np.stack(
                    [
                        baseline_rollouts[label].mass_error
                        for label in baseline_labels
                    ],
                    axis=0,
                ),
            }
        )
    np.savez_compressed(path, **arrays)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot chemotaxis rollout-error curves in the style of Figure 6 "
            "of 7_6_26_scirep_v2.pdf."
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Source dataset; inferred from rollout summaries by default.",
    )
    parser.add_argument("--trajectories", nargs="+", type=int, default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument(
        "--baseline-schemes",
        nargs="+",
        choices=BASELINE_SCHEMES,
        default=(),
        help=(
            "Coarse finite-volume baselines to calculate from the held-out "
            "initial states. Omit to disable numerical baselines."
        ),
    )
    parser.add_argument(
        "--baseline-cfl",
        type=float,
        default=None,
        help=(
            "CFL used for baseline subcycling and frame-step stability "
            "diagnostics; defaults to the dataset CFL."
        ),
    )
    parser.add_argument(
        "--baseline-step-mode",
        choices=BASELINE_STEP_MODES,
        default=BASELINE_STEP_MODE_CFL_SUBCYCLE,
        help=(
            "How numerical baseline timesteps are selected. 'cfl-subcycle' "
            "(default) takes as many stable substeps as each stored/GNN "
            "interval requires. 'frame-step' takes exactly one SSP-RK2 step "
            "per stored/GNN interval, even if that violates the configured "
            "CFL stability limit."
        ),
    )
    parser.add_argument(
        "--error-curves-metrics",
        nargs="+",
        default=None,
        metavar="METRIC",
        help=(
            "Panels to plot in order. Accepts space- or comma-separated "
            "values from {rmse, rel_l2, l1, ssim}, or all. Default: rel_l2 ssim."
        ),
    )
    parser.add_argument(
        "--error-curves-smooth-window",
        type=int,
        default=9,
        help=(
            "Positive odd centered moving-average width. Default 9 matches "
            "Figure 6; use 1 to disable smoothing."
        ),
    )
    parser.add_argument("--error-curves-log-y", action="store_true")
    parser.add_argument(
        "--error-curves-show-trajectories",
        action="store_true",
        help="Show faint individual trajectory curves behind the means.",
    )
    parser.add_argument(
        "--error-curves-spread",
        choices=("none", "iqr", "std"),
        default="none",
        help="Optional spread band around each model mean (default: none).",
    )
    parser.add_argument(
        "--persistence",
        "--include-persistence",
        action="store_true",
        help="Include the persistence baseline in every panel.",
    )
    parser.add_argument(
        "--curve-color",
        nargs="+",
        default=None,
        metavar="LABEL=COLOR",
        help="Override a displayed label or model-family color.",
    )
    parser.add_argument(
        "--figure-title",
        default="2D Chemotactic Cell Migration: Rollout Accuracy",
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
        "--formats",
        nargs="+",
        choices=("png", "pdf", "svg"),
        default=("png",),
    )
    parser.add_argument("--dpi", type=int, default=170)
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    specs: List[ModelSpec] = args.model
    comparison_count = len(specs) + len(set(args.baseline_schemes))
    if comparison_count < 2:
        raise ValueError(
            "Specify at least two curves using --model and/or "
            "--baseline-schemes."
        )
    labels = [spec.label for spec in specs]
    if len(set(labels)) != len(labels):
        raise ValueError("Every --model label must be unique.")
    if args.rollout_steps is not None and args.rollout_steps <= 0:
        raise ValueError("--rollout-steps must be positive.")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive.")
    if (
        args.error_curves_smooth_window <= 0
        or args.error_curves_smooth_window % 2 == 0
    ):
        raise ValueError(
            "--error-curves-smooth-window must be a positive odd integer."
        )
    if args.ssim_window_size < 3 or args.ssim_window_size % 2 == 0:
        raise ValueError("--ssim-window-size must be an odd integer >= 3.")
    if (
        not np.isfinite(args.ssim_window_sigma)
        or args.ssim_window_sigma <= 0.0
    ):
        raise ValueError("--ssim-window-sigma must be positive and finite.")
    for value, name in (
        (args.ssim_data_range, "--ssim-data-range"),
        (args.ssim_reference_spacing, "--ssim-reference-spacing"),
    ):
        if value is not None and (not np.isfinite(value) or value <= 0.0):
            raise ValueError(f"{name} must be positive and finite.")
    if args.baseline_cfl is not None and (
        not np.isfinite(args.baseline_cfl)
        or not (0.0 < args.baseline_cfl <= 1.0)
    ):
        raise ValueError("--baseline-cfl must be finite and in (0, 1].")


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    validate_args(args)
    selected_metrics = resolve_error_curve_metrics(args.error_curves_metrics)
    color_overrides = parse_curve_color_overrides(args.curve_color)
    specs: List[ModelSpec] = args.model
    learned_labels = [spec.label for spec in specs]
    labels = list(learned_labels)
    rollouts: Dict[str, RolloutData] = {
        spec.label: load_rollout(spec.prediction_path) for spec in specs
    }
    trajectory_ids = common_trajectory_ids(rollouts, args.trajectories)
    num_times = validate_common_rollouts(
        rollouts, trajectory_ids, args.rollout_steps
    )
    reference = rollouts[learned_labels[0]]
    times = reference.time[:num_times]
    dataset_path, dataset_path_source = infer_dataset_path(
        specs, explicit_path=args.data
    )
    geometries = load_mesh_geometries(
        dataset_path,
        trajectory_ids,
        expected_num_nodes=reference.truth.shape[2],
    )
    reference_spacing, reference_spacing_source = resolve_ssim_reference_spacing(
        geometries, explicit_spacing=args.ssim_reference_spacing
    )
    data_range, data_range_source = resolve_ssim_data_range(
        rollouts,
        trajectory_ids,
        num_times=num_times,
        explicit_range=args.ssim_data_range,
        dataset_path=dataset_path,
    )
    gnn_inference_timings = report_gnn_inference_timings(
        specs,
        rollouts,
        trajectory_ids=trajectory_ids,
        num_times=num_times,
    )
    baseline_rollouts, baseline_metadata = build_numerical_baseline_rollouts(
        dataset_path,
        trajectory_ids,
        reference=reference,
        num_times=num_times,
        schemes=args.baseline_schemes,
        explicit_cfl=args.baseline_cfl,
        step_mode=args.baseline_step_mode,
    )
    conflicts = sorted(set(labels).intersection(baseline_rollouts))
    if conflicts:
        raise ValueError(
            "Model labels conflict with requested numerical baselines: "
            + ", ".join(conflicts)
        )
    rollouts.update(baseline_rollouts)
    labels.extend(baseline_rollouts)
    validate_common_rollouts(rollouts, trajectory_ids, num_times - 1)
    windows: Dict[int, LocalMeshWindow] = {
        trajectory_id: build_local_mesh_window(
            geometry.centers,
            geometry.cell_areas,
            reference_spacing=reference_spacing,
            window_size=args.ssim_window_size,
            sigma_cells=args.ssim_window_sigma,
        )
        for trajectory_id, geometry in geometries.items()
    }
    metrics_by_model: Dict[str, List[TrajectoryMetrics]] = {
        label: [] for label in labels
    }
    for trajectory_id in trajectory_ids:
        for label in labels:
            metrics_by_model[label].append(
                compute_metrics(
                    rollouts[label],
                    trajectory_id,
                    num_times=num_times,
                    local_window=windows[trajectory_id],
                    ssim_data_range=data_range,
                )
            )

    metadata = fidelity_metadata(
        data_range=data_range,
        data_range_source=data_range_source,
        window_size=args.ssim_window_size,
        sigma_cells=args.ssim_window_sigma,
        reference_spacing=reference_spacing,
        reference_spacing_source=reference_spacing_source,
    )
    metadata.update(
        {
            "dataset": str(dataset_path),
            "dataset_path_source": dataset_path_source,
            "baseline_step_mode": args.baseline_step_mode,
        }
    )
    styles = figure6_model_styles(labels, color_overrides=color_overrides)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    formats = list(dict.fromkeys(args.formats))
    figure_paths = plot_figure6_error_curves(
        metrics_by_model=metrics_by_model,
        times=times,
        selected_metrics=selected_metrics,
        styles=styles,
        output_base=output_dir / "error_curves_compare",
        formats=formats,
        dpi=args.dpi,
        smooth_window=args.error_curves_smooth_window,
        show_trajectories=args.error_curves_show_trajectories,
        spread=args.error_curves_spread,
        log_y=args.error_curves_log_y,
        title=args.figure_title,
        show_persistence=args.persistence,
    )
    raw_path = output_dir / "error_curves_metrics.npz"
    source_type_by_label = np.asarray(
        [
            "learned_model" if label in learned_labels else "numerical_baseline"
            for label in labels
        ]
    )
    baseline_scheme_by_label = np.asarray(
        [
            str(baseline_metadata.get(label, {}).get("scheme", ""))
            for label in labels
        ]
    )
    raw_metadata = dict(metadata)
    raw_metadata.update(
        {
            "source_type_by_label": source_type_by_label,
            "baseline_scheme_by_label": baseline_scheme_by_label,
            "numerical_baseline_metadata_json": json.dumps(
                baseline_metadata, sort_keys=True
            ),
            "gnn_inference_timing_json": json.dumps(
                gnn_inference_timings, sort_keys=True
            ),
        }
    )
    write_raw_metrics(
        raw_path,
        labels=labels,
        trajectory_ids=trajectory_ids,
        times=times,
        selected_metrics=selected_metrics,
        metrics_by_model=metrics_by_model,
        metadata=raw_metadata,
        baseline_rollouts=baseline_rollouts,
    )
    final_unsmoothed_means = {
        label: {
            metric: float(
                np.mean(
                    _metric_matrix(
                        metrics_by_model[label],
                        ERROR_CURVE_METRIC_ATTRIBUTE[metric],
                    ),
                    axis=0,
                )[-1]
            )
            for metric in selected_metrics
        }
        for label in labels
    }
    summary = {
        "source_figure": "Figure 6 of 7_6_26_scirep_v2.pdf",
        "models": {
            spec.label: str(spec.prediction_path) for spec in specs
        },
        "numerical_baselines": baseline_metadata,
        "baseline_step_mode": args.baseline_step_mode,
        "gnn_inference_timings": gnn_inference_timings,
        "curve_sources": {
            label: (
                "learned_model"
                if label in learned_labels
                else "numerical_baseline"
            )
            for label in labels
        },
        "trajectory_ids": trajectory_ids,
        "num_trajectories": len(trajectory_ids),
        "rollout_steps": num_times - 1,
        "time_start": float(times[0]),
        "time_end": float(times[-1]),
        "error_curves_metrics": selected_metrics,
        "error_curves_smooth_window": args.error_curves_smooth_window,
        "smoothing": "display_only_centered_moving_average",
        "error_curves_show_trajectories": args.error_curves_show_trajectories,
        "error_curves_spread": args.error_curves_spread,
        "include_persistence": args.persistence,
        "figures": [str(path) for path in figure_paths],
        "raw_metrics": str(raw_path),
        "final_unsmoothed_means": final_unsmoothed_means,
        "fidelity": metadata,
        "ssim_constants": {"k1": SSIM_K1, "k2": SSIM_K2},
    }
    summary_path = output_dir / "error_curves_summary.json"
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")

    print(
        "local-mesh SSIM: "
        f"data_range={data_range:.12g} ({data_range_source}), "
        f"reference_spacing={reference_spacing:.12g} "
        f"({reference_spacing_source}), window={args.ssim_window_size}, "
        f"sigma_cells={args.ssim_window_sigma:g}"
    )
    print(
        f"plotted {len(learned_labels)} learned models and "
        f"{len(baseline_rollouts)} numerical baselines on "
        f"{len(trajectory_ids)} trajectories "
        f"with metrics {selected_metrics}"
    )
    for label, details in baseline_metadata.items():
        print(
            f"  {label}: dt={details['solver_timestep']:.12g}, "
            f"step mode={details['step_mode']}, "
            f"substeps/frame={details['substeps_per_output_frame']}, "
            f"effective CFL={details['effective_cfl']:.6g}, "
            "max mass drift="
            f"{details['maximum_absolute_total_mass_drift']:.3e}"
        )
    for path in figure_paths:
        print(f"wrote figure: {path}")
    print(f"wrote raw metrics: {raw_path}")
    print(f"wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
