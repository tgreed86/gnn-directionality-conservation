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


# Source: code/1D_traffic/scripts/plot_network_lwr_traffic_data.py (_road_indices)
def _road_indices(data: Dict[str, np.ndarray], road: int) -> np.ndarray:
    """Return globally ordered cell indices for one road."""
    offsets = np.asarray(data["road_cell_offsets"], dtype=np.int64)
    return np.arange(offsets[road], offsets[road + 1], dtype=np.int64)


# Source: code/1D_traffic/scripts/plot_network_lwr_traffic_data.py (_road_coordinates)
def _road_coordinates(
    data: Dict[str, np.ndarray], road: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Return schematic coordinates including both junction endpoints."""
    indices = _road_indices(data, road)
    cell_positions = np.asarray(data["pos"], dtype=np.float64)[indices]
    junction_positions = np.asarray(
        data["junction_positions"], dtype=np.float64
    )
    start_id = int(data["road_start_junction"][road])
    end_id = int(data["road_end_junction"][road])
    coordinates = np.vstack(
        [
            junction_positions[start_id],
            cell_positions,
            junction_positions[end_id],
        ]
    )
    return indices, coordinates


# Source: code/1D_traffic/scripts/plot_network_lwr_traffic_data.py (_road_segments)
def _road_segments(
    data: Dict[str, np.ndarray],
    road: int,
    density: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create colored schematic segments and their density values."""
    indices, coordinates = _road_coordinates(data, road)
    road_density = density[indices]
    segments = np.stack([coordinates[:-1], coordinates[1:]], axis=1)
    segment_density = np.concatenate(
        [
            road_density[:1],
            0.5 * (road_density[:-1] + road_density[1:]),
            road_density[-1:],
        ]
    )
    return segments, segment_density


# Source: code/1D_traffic/scripts/plot_network_lwr_traffic_data.py (_add_direction_arrow)
def _add_direction_arrow(
    ax: plt.Axes,
    coordinates: np.ndarray,
    *,
    color: str = "0.15",
) -> None:
    """Add a small arrow in the physical traffic direction."""
    middle = max(0, coordinates.shape[0] // 2 - 1)
    start = coordinates[middle]
    end = coordinates[middle + 1]
    delta = end - start
    arrow_start = start + 0.10 * delta
    arrow_end = start + 0.90 * delta
    patch = FancyArrowPatch(
        arrow_start,
        arrow_end,
        arrowstyle="-|>",
        mutation_scale=9,
        linewidth=0.9,
        color=color,
        zorder=5,
    )
    ax.add_patch(patch)


# Source: code/1D_traffic/scripts/plot_network_lwr_traffic_data.py (_draw_density_network)
def _draw_density_network(
    ax: plt.Axes,
    data: Dict[str, np.ndarray],
    density: np.ndarray,
    *,
    norm: Normalize,
    cmap: str,
    flow_direction: float = 1.0,
) -> LineCollection:
    """Draw one density state on the embedded network."""
    road_names = np.asarray(data["road_names"])
    collection_for_colorbar: Optional[LineCollection] = None
    for road in range(road_names.size):
        segments, segment_density = _road_segments(
            data, road, density
        )
        underlay = LineCollection(
            segments,
            colors="0.22",
            linewidths=8.0,
            capstyle="round",
            zorder=1,
        )
        ax.add_collection(underlay)
        collection = LineCollection(
            segments,
            cmap=cmap,
            norm=norm,
            linewidths=5.5,
            capstyle="round",
            zorder=2,
        )
        collection.set_array(segment_density)
        ax.add_collection(collection)
        collection_for_colorbar = collection
        _, coordinates = _road_coordinates(data, road)
        arrow_coordinates = (
            coordinates
            if flow_direction >= 0.0
            else coordinates[::-1]
        )
        _add_direction_arrow(ax, arrow_coordinates, color="white")

    junction_positions = np.asarray(
        data["junction_positions"], dtype=np.float64
    )
    ax.scatter(
        junction_positions[:, 0],
        junction_positions[:, 1],
        s=28,
        facecolor="white",
        edgecolor="0.12",
        linewidth=0.8,
        zorder=7,
    )
    all_positions = np.asarray(data["pos"], dtype=np.float64)
    padding = 0.10 * max(
        float(np.ptp(all_positions[:, 0])),
        float(np.ptp(all_positions[:, 1])),
    )
    ax.set_xlim(
        float(np.min(all_positions[:, 0]) - padding),
        float(np.max(all_positions[:, 0]) + padding),
    )
    ax.set_ylim(
        float(np.min(all_positions[:, 1]) - padding),
        float(np.max(all_positions[:, 1]) + padding),
    )
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    if collection_for_colorbar is None:
        raise RuntimeError("The network contains no roads to draw.")
    return collection_for_colorbar


# Source: code/1D_traffic/scripts/plot_sageconv_rollouts.py (plot_network_comparison)
def plot_network_comparison(
    data: Dict[str, np.ndarray],
    truth: np.ndarray,
    prediction: np.ndarray,
    trajectory_id: int,
    output_path: Path,
    *,
    snapshot_steps: Sequence[int],
    dt: float,
    rho_max: float,
    dpi: int,
    cmap: str,
    flow_direction: float = 1.0,
    time_axis: str = "rows",
) -> None:
    """Plot ground truth, prediction, and signed error on the road network."""
    if time_axis not in ("rows", "columns"):
        raise ValueError(
            f"time_axis must be 'rows' or 'columns', got {time_axis!r}."
        )
    error = prediction - truth
    error_limit = max(
        float(np.max(np.abs(error[np.asarray(snapshot_steps)]))),
        1.0e-8,
    )
    density_norm = Normalize(vmin=0.0, vmax=rho_max)
    error_norm = Normalize(vmin=-error_limit, vmax=error_limit)

    time_in_rows = time_axis == "rows"
    if time_in_rows:
        num_rows = len(snapshot_steps)
        num_columns = 3
        figure_size = (10.8, 3.0 * len(snapshot_steps))
    else:
        num_rows = 3
        num_columns = len(snapshot_steps)
        figure_size = (3.6 * len(snapshot_steps), 9.0)
    figure, axes = plt.subplots(
        num_rows,
        num_columns,
        figsize=figure_size,
        constrained_layout=True,
        squeeze=False,
    )
    density_collection = None
    error_collection = None
    for snapshot_index, step in enumerate(snapshot_steps):
        if time_in_rows:
            truth_axis = axes[snapshot_index, 0]
            prediction_axis = axes[snapshot_index, 1]
            error_axis = axes[snapshot_index, 2]
        else:
            truth_axis = axes[0, snapshot_index]
            prediction_axis = axes[1, snapshot_index]
            error_axis = axes[2, snapshot_index]
        density_collection = _draw_density_network(
            truth_axis,
            data,
            truth[step],
            norm=density_norm,
            cmap=cmap,
            flow_direction=flow_direction,
        )
        _draw_density_network(
            prediction_axis,
            data,
            prediction[step],
            norm=density_norm,
            cmap=cmap,
            flow_direction=flow_direction,
        )
        error_collection = _draw_density_network(
            error_axis,
            data,
            error[step],
            norm=error_norm,
            cmap="coolwarm",
            flow_direction=flow_direction,
        )
        snapshot_label = f"step {step}\nt={step * dt:.4g}"
        if time_in_rows:
            truth_axis.text(
                -0.04,
                0.5,
                snapshot_label,
                transform=truth_axis.transAxes,
                ha="right",
                va="center",
                fontsize=18,
            )
        else:
            truth_axis.set_title(snapshot_label, fontsize=18)

    if time_in_rows:
        axes[0, 0].set_title("Ground Truth", fontsize=20)
        axes[0, 1].set_title("GNN Prediction", fontsize=20)
        axes[0, 2].set_title("Signed Error", fontsize=20)
        density_axes = list(axes[:, :2].ravel())
        error_axes = list(axes[:, 2])
    else:
        row_labels = ("Ground Truth", "GNN Prediction", "Signed Error")
        for row, row_label in enumerate(row_labels):
            axes[row, 0].text(
                -0.04,
                0.5,
                row_label,
                transform=axes[row, 0].transAxes,
                ha="right",
                va="center",
                fontsize=18,
            )
        density_axes = list(axes[:2, :].ravel())
        error_axes = list(axes[2, :].ravel())

    if density_collection is not None:
        density_colorbar = figure.colorbar(
            density_collection,
            ax=density_axes,
            pad=0.012,
            shrink=0.82,
        )
        density_colorbar.set_label(r"Density $\rho$", fontsize=20)
        density_colorbar.ax.tick_params(labelsize=14)
    if error_collection is not None:
        error_colorbar = figure.colorbar(
            error_collection,
            ax=error_axes,
            pad=0.012,
            shrink=0.82,
        )
        error_colorbar.set_label(r"$\hat\rho-\rho$", fontsize=20)
        error_colorbar.ax.tick_params(labelsize=14)
    figure.suptitle(
        #f"Held-out trajectory {trajectory_id}: network rollout",
        "LWR Traffic Flow",
        fontsize=25,
    )
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)

