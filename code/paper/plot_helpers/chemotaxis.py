# Rendering-only extraction; numerical/rendering function bodies unchanged.
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


# Source: code/chemotactic_cell_migration/scripts/plot_gnn_rollouts.py (CHEMO_CONTOUR_METHOD_AUTO)
CHEMO_CONTOUR_METHOD_AUTO = "auto"


# Source: code/chemotactic_cell_migration/scripts/plot_gnn_rollouts.py (triangulation)
def triangulation(data: Dict[str, np.ndarray], trajectory: int) -> mtri.Triangulation:
    vertices = data["mesh_vertices"][trajectory]
    triangles = data["triangles"][trajectory]
    return mtri.Triangulation(vertices[:, 0], vertices[:, 1], triangles)


# Source: code/chemotactic_cell_migration/scripts/plot_gnn_rollouts.py (ChemoContourField)
@dataclass(frozen=True)
class ChemoContourField:
    """Static chemoattractant field prepared for one contour renderer."""

    method: str
    values: np.ndarray
    x: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None
    triangulation: Optional[mtri.Triangulation] = None


# Source: code/chemotactic_cell_migration/scripts/plot_gnn_rollouts.py (CHEMO_CONTOUR_METHOD_ANALYTIC)
CHEMO_CONTOUR_METHOD_ANALYTIC = "analytic-grid"


# Source: code/chemotactic_cell_migration/scripts/plot_gnn_rollouts.py (CHEMO_CONTOUR_METHOD_TRIANGULATION)
CHEMO_CONTOUR_METHOD_TRIANGULATION = "cell-triangulation"


# Source: code/chemotactic_cell_migration/scripts/plot_gnn_rollouts.py (CHEMO_CONTOUR_METHODS)
CHEMO_CONTOUR_METHODS = (
    CHEMO_CONTOUR_METHOD_AUTO,
    CHEMO_CONTOUR_METHOD_ANALYTIC,
    CHEMO_CONTOUR_METHOD_TRIANGULATION,
)


# Source: code/chemotactic_cell_migration/scripts/plot_gnn_rollouts.py (resolve_chemo_contour_method)
def resolve_chemo_contour_method(
    data: Dict[str, np.ndarray], requested_method: str
) -> str:
    """Select analytical contours when their source metadata is available."""
    if requested_method not in CHEMO_CONTOUR_METHODS:
        raise ValueError(
            "chemo contour method must be one of "
            + ", ".join(CHEMO_CONTOUR_METHODS)
        )
    analytic_keys = (
        "chemo_source_count",
        "chemo_source_centers",
        "chemo_source_amplitudes",
        "chemo_source_sigmas",
    )
    missing = [key for key in analytic_keys if key not in data]
    if requested_method == CHEMO_CONTOUR_METHOD_AUTO:
        return (
            CHEMO_CONTOUR_METHOD_ANALYTIC
            if not missing
            else CHEMO_CONTOUR_METHOD_TRIANGULATION
        )
    if requested_method == CHEMO_CONTOUR_METHOD_ANALYTIC and missing:
        raise KeyError(
            "Analytical chemoattractant contours require dataset fields: "
            + ", ".join(missing)
        )
    return requested_method


# Source: code/chemotactic_cell_migration/scripts/generate_chemotactic_cell_migration_data.py (gaussian_mixture_value_and_gradient)
def gaussian_mixture_value_and_gradient(
    *,
    points: np.ndarray,
    centers: np.ndarray,
    amplitudes: np.ndarray,
    sigmas: np.ndarray,
    baseline: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate an isotropic Gaussian mixture and its analytic gradient."""
    points = np.asarray(points, dtype=np.float64)
    delta = points[..., None, :] - centers
    sigma2 = np.square(sigmas)
    exponent = -0.5 * np.sum(np.square(delta), axis=-1) / sigma2
    components = amplitudes * np.exp(exponent)
    value = baseline + np.sum(components, axis=-1)
    gradient = np.sum(-components[..., None] * delta / sigma2[..., None], axis=-2)
    return value, gradient


# Source: code/chemotactic_cell_migration/scripts/plot_gnn_rollouts.py (cell_center_triangulation)
def cell_center_triangulation(
    data: Dict[str, np.ndarray], trajectory: int
) -> mtri.Triangulation:
    """Triangulate cell centers for contouring cell-averaged fields."""
    centers = np.asarray(data["pos"][trajectory], dtype=np.float64)
    return mtri.Triangulation(centers[:, 0], centers[:, 1])


# Source: code/chemotactic_cell_migration/scripts/plot_gnn_rollouts.py (prepare_chemo_contour_field)
def prepare_chemo_contour_field(
    data: Dict[str, np.ndarray],
    trajectory: int,
    mesh: mtri.Triangulation,
    *,
    method: str,
    resolution: int,
) -> ChemoContourField:
    """Prepare either an exact Gaussian grid or the legacy cell triangulation."""
    resolved_method = resolve_chemo_contour_method(data, method)
    if resolved_method == CHEMO_CONTOUR_METHOD_TRIANGULATION:
        values = np.asarray(
            data["chemoattractant"][trajectory], dtype=np.float64
        )
        if values.ndim == 2 and values.shape[1] == 1:
            values = values[:, 0]
        if values.ndim != 1:
            raise ValueError(
                "chemoattractant must have shape [trajectory, cell, 1] "
                "or [trajectory, cell]."
            )
        return ChemoContourField(
            method=resolved_method,
            values=values,
            triangulation=cell_center_triangulation(data, trajectory),
        )

    if resolution < 2:
        raise ValueError("Analytical contour resolution must be at least 2.")
    count = int(np.asarray(data["chemo_source_count"])[trajectory])
    centers = np.asarray(
        data["chemo_source_centers"][trajectory], dtype=np.float64
    )
    amplitudes = np.asarray(
        data["chemo_source_amplitudes"][trajectory], dtype=np.float64
    )
    sigmas = np.asarray(
        data["chemo_source_sigmas"][trajectory], dtype=np.float64
    )
    if count <= 0 or count > centers.shape[0]:
        raise ValueError("chemo_source_count is inconsistent with source arrays.")
    centers = centers[:count]
    amplitudes = amplitudes[:count]
    sigmas = sigmas[:count]
    if (
        centers.shape != (count, 2)
        or amplitudes.shape != (count,)
        or sigmas.shape != (count,)
        or not np.all(np.isfinite(centers))
        or not np.all(np.isfinite(amplitudes))
        or not np.all(np.isfinite(sigmas))
        or np.any(sigmas <= 0.0)
    ):
        raise ValueError("Invalid Gaussian chemoattractant source metadata.")
    baseline_array = np.asarray(data.get("chemo_baseline", 0.0), dtype=np.float64)
    if baseline_array.size != 1 or not np.all(np.isfinite(baseline_array)):
        raise ValueError("chemo_baseline must be a finite scalar.")
    x = np.linspace(float(np.min(mesh.x)), float(np.max(mesh.x)), resolution)
    y = np.linspace(float(np.min(mesh.y)), float(np.max(mesh.y)), resolution)
    grid_x, grid_y = np.meshgrid(x, y)
    points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    values, _ = gaussian_mixture_value_and_gradient(
        points=points,
        centers=centers,
        amplitudes=amplitudes,
        sigmas=sigmas,
        baseline=float(baseline_array.item()),
    )
    return ChemoContourField(
        method=resolved_method,
        values=values.reshape(grid_x.shape),
        x=x,
        y=y,
    )


# Source: code/chemotactic_cell_migration/scripts/plot_gnn_rollouts.py (_style_spatial_axis)
def _style_spatial_axis(
    axis: plt.Axes, mesh: mtri.Triangulation
) -> None:
    """Use the exact mesh domain without Matplotlib's autoscale padding."""
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(float(np.min(mesh.x)), float(np.max(mesh.x)))
    axis.set_ylim(float(np.min(mesh.y)), float(np.max(mesh.y)))
    axis.set_xticks([])
    axis.set_yticks([])


# Source: code/chemotactic_cell_migration/scripts/plot_gnn_rollouts.py (overlay_chemoattractant_contours)
def overlay_chemoattractant_contours(
    axis: plt.Axes,
    field: ChemoContourField,
    *,
    num_levels: int,
    alpha: float,
) -> None:
    """Overlay high-contrast contours of the static chemoattractant field."""
    lower = float(np.min(field.values))
    upper = float(np.max(field.values))
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        return
    levels = np.linspace(lower, upper, num_levels + 2, dtype=np.float64)[1:-1]
    if field.method == CHEMO_CONTOUR_METHOD_ANALYTIC:
        if field.x is None or field.y is None:
            raise RuntimeError("Analytical contour field has no grid coordinates.")

        def draw_contours(**kwargs: object) -> None:
            axis.contour(field.x, field.y, field.values, levels=levels, **kwargs)

    else:
        if field.triangulation is None:
            raise RuntimeError("Triangulated contour field has no triangulation.")

        def draw_contours(**kwargs: object) -> None:
            axis.tricontour(
                field.triangulation,
                field.values,
                levels=levels,
                **kwargs,
            )

    # A dark underlay keeps the white contours visible on both light and dark
    # regions of the density colormap.
    draw_contours(
        colors="black",
        linewidths=1.5,
        alpha=0.45 * alpha,
        zorder=3,
    )
    draw_contours(
        colors="white",
        linewidths=0.75,
        alpha=alpha,
        zorder=4,
    )


# Source: code/chemotactic_cell_migration/scripts/plot_gnn_rollouts.py (plot_spatial_comparison)
def plot_spatial_comparison(
    data: Dict[str, np.ndarray],
    truth: np.ndarray,
    prediction: np.ndarray,
    trajectory: int,
    output_path: Path,
    *,
    snapshot_steps: Sequence[int],
    dt: float,
    model_label: str,
    dpi: int,
    cmap: str,
    chemo_contours: bool,
    chemo_contour_levels: int,
    chemo_contour_alpha: float,
    chemo_contour_method: str = CHEMO_CONTOUR_METHOD_AUTO,
    chemo_contour_resolution: int = 300,
    spatial_time_axis: str = "rows",
) -> None:
    """Plot truth, prediction, error, and optional chemoattractant contours."""
    if spatial_time_axis not in ("rows", "columns"):
        raise ValueError("spatial_time_axis must be 'rows' or 'columns'.")
    mesh = triangulation(data, trajectory)
    contour_field = None
    if chemo_contours:
        contour_field = prepare_chemo_contour_field(
            data,
            trajectory,
            mesh,
            method=chemo_contour_method,
            resolution=chemo_contour_resolution,
        )
    error = prediction - truth
    steps = np.asarray(snapshot_steps, dtype=np.int64)
    density_min = float(min(np.min(truth[steps]), np.min(prediction[steps])))
    density_max = float(max(np.max(truth[steps]), np.max(prediction[steps])))
    error_limit = max(float(np.max(np.abs(error[steps]))), 1.0e-10)
    if spatial_time_axis == "rows":
        num_rows, num_columns = len(steps), 3
        figure_size = (11.5, 3.15 * len(steps))
    else:
        num_rows, num_columns = 3, len(steps)
        figure_size = (max(3.15 * len(steps), 7.0), 9.45)
    figure, axes = plt.subplots(
        num_rows,
        num_columns,
        figsize=figure_size,
        squeeze=False,
        constrained_layout=True,
    )
    density_artist = None
    error_artist = None
    for step_index, step in enumerate(steps):
        for field_index, (values, color_map, lower, upper) in enumerate(
            (
                (truth[step], cmap, density_min, density_max),
                (prediction[step], cmap, density_min, density_max),
                (error[step], "coolwarm", -error_limit, error_limit),
            )
        ):
            axis = (
                axes[step_index, field_index]
                if spatial_time_axis == "rows"
                else axes[field_index, step_index]
            )
            artist = axis.tripcolor(
                mesh,
                facecolors=values,
                shading="flat",
                cmap=color_map,
                vmin=lower,
                vmax=upper,
                edgecolors="none",
            )
            if (
                field_index < 2
                and contour_field is not None
            ):
                overlay_chemoattractant_contours(
                    axis,
                    contour_field,
                    num_levels=chemo_contour_levels,
                    alpha=chemo_contour_alpha,
                )
            _style_spatial_axis(axis, mesh)
            if field_index < 2:
                density_artist = artist
            else:
                error_artist = artist
    if spatial_time_axis == "rows":
        for row, step in enumerate(steps):
            axes[row, 0].set_ylabel(
                f"Step {step}\nt={step * dt:.4g}", fontsize=18
            )
        axes[0, 0].set_title("Ground Truth", fontsize=20)
        #axes[0, 1].set_title(f"{model_label} prediction", fontsize=20)
        axes[0, 1].set_title("GNN Prediction", fontsize=20)
        axes[0, 2].set_title("Prediction − Truth", fontsize=20)
        density_axes = list(axes[:, :2].ravel())
        error_axes = list(axes[:, 2])
    else:
        for column, step in enumerate(steps):
            axes[0, column].set_title(
                f"Step {step}\nt={step * dt:.4g}", fontsize=16
            )
        axes[0, 0].set_ylabel("Ground Truth", fontsize=16)
        axes[1, 0].set_ylabel("GNN Prediction", fontsize=16)
        axes[2, 0].set_ylabel("Prediction − Truth", fontsize=16)
        density_axes = list(axes[:2, :].ravel())
        error_axes = list(axes[2, :])
    if density_artist is not None:
        colorbar = figure.colorbar(
            density_artist, ax=density_axes, shrink=0.86, pad=0.015
        )
        colorbar.set_label("Cell density", fontsize=18)
        colorbar.ax.tick_params(labelsize=14)
    if error_artist is not None:
        colorbar = figure.colorbar(
            error_artist, ax=error_axes, shrink=0.86, pad=0.015
        )
        colorbar.set_label("Density error", fontsize=18)
        colorbar.ax.tick_params(labelsize=14)
    #title = f"Held-out trajectory {trajectory}: Autoregressive Rollout"
    title = f"Chemotactic Cell Migration: Autoregressive Rollout"
    #if chemo_contours:
    #    title += "\nWhite contours show chemoattractant concentration c"
    figure.suptitle(title, fontsize=25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)

