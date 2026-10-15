#!/usr/bin/env python3
"""Plot one network-topology archive before model training.

The script supports archives produced by both the fixed network generator and
the manifest-based variable-network generators, including structured paired
direction-reversal trajectories.

For each selected trajectory, the script creates:

* density snapshots drawn on the embedded road network;
* one space-time density diagram per road;
* compact intersection-throughput and road-mass heatmaps;
* busiest junction movements and total-conservation diagnostics; and
* matched canonical/reversed density RMSE when pair metadata is available.

It also creates one dataset-level summary of the network, congestion-block
sampling, and conservation errors.

Example:

    python scripts/plot_network_lwr_traffic_data.py \
        data/lwr_network.npz \
        --trajectories 0 1 2 \
        --output-dir plots/lwr_network
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import ScalarFormatter


ROAD_COLORS = ("tab:blue", "tab:orange", "tab:green", "tab:red")


def _load_dataset(path: Path) -> Dict[str, np.ndarray]:
    """Load and validate the network fields used by the figures."""
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}

    required = (
        "rollout_states",
        "rollout_transfer_fluxes",
        "pos",
        "cell_lengths",
        "cell_road_id",
        "cell_road_coordinate",
        "road_names",
        "road_lengths",
        "road_cell_offsets",
        "road_start_junction",
        "road_end_junction",
        "junction_names",
        "junction_types",
        "junction_positions",
        "transfer_index",
        "transfer_type",
        "transfer_junction_id",
        "transfer_from_road",
        "transfer_to_road",
        "total_mass",
        "road_mass",
        "dt",
    )
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"Dataset is missing required fields: {missing}")

    rollout = np.asarray(data["rollout_states"])
    if rollout.ndim == 4 and rollout.shape[-1] == 1:
        rollout = rollout[..., 0]
    if rollout.ndim != 3:
        raise ValueError(
            "rollout_states must have shape [trajectory, time, cell] or "
            f"[trajectory, time, cell, 1], got {rollout.shape}."
        )
    data["rollout_states"] = rollout.astype(np.float64, copy=False)

    transfer_fluxes = np.asarray(data["rollout_transfer_fluxes"])
    if transfer_fluxes.ndim == 4 and transfer_fluxes.shape[-1] == 1:
        transfer_fluxes = transfer_fluxes[..., 0]
    if transfer_fluxes.ndim != 3:
        raise ValueError(
            "rollout_transfer_fluxes must have shape "
            "[trajectory, transition, transfer] or that shape plus a trailing "
            f"singleton channel, got {transfer_fluxes.shape}."
        )
    data["rollout_transfer_fluxes"] = transfer_fluxes.astype(
        np.float64, copy=False
    )

    num_trajectories, num_times, num_cells = rollout.shape
    num_transfers = transfer_fluxes.shape[2]
    num_roads = np.asarray(data["road_names"]).size
    if num_trajectories == 0:
        raise ValueError("The dataset contains no trajectories.")
    if transfer_fluxes.shape[:2] != (num_trajectories, num_times - 1):
        raise ValueError(
            "Transfer-flux trajectory and time dimensions do not match the "
            "state rollout."
        )
    if np.asarray(data["pos"]).shape != (num_cells, 2):
        raise ValueError("pos must have shape [num_cells, 2].")
    if np.asarray(data["cell_lengths"]).shape != (num_cells,):
        raise ValueError("cell_lengths must match the state cell dimension.")
    if np.asarray(data["transfer_index"]).shape != (2, num_transfers):
        raise ValueError("transfer_index must have shape [2, num_transfers].")
    if np.asarray(data["road_mass"]).shape != (
        num_trajectories,
        num_times,
        num_roads,
    ):
        raise ValueError(
            "road_mass must have shape [trajectory, time, road]."
        )
    if np.asarray(data["total_mass"]).shape != (num_trajectories, num_times):
        raise ValueError("total_mass must have shape [trajectory, time].")
    return data


def _scalar(data: Dict[str, np.ndarray], key: str, default: float) -> float:
    """Return a scalar archive field with a fallback."""
    if key not in data:
        return float(default)
    return float(np.asarray(data[key]).item())


def _trajectory_direction(
    data: Dict[str, np.ndarray],
    trajectory: int,
) -> float:
    """Return a trajectory's flow direction, defaulting to canonical."""
    if "trajectory_flow_direction" not in data:
        return 1.0
    directions = np.asarray(
        data["trajectory_flow_direction"],
        dtype=np.float64,
    )
    return float(directions[trajectory])


def _trajectory_description(
    data: Dict[str, np.ndarray],
    trajectory: int,
) -> str:
    """Return a readable trajectory, pair, and direction description."""
    description = f"trajectory {trajectory}"
    if "trajectory_pair_id" in data:
        pair_ids = np.asarray(data["trajectory_pair_id"], dtype=np.int64)
        description += f", pair {int(pair_ids[trajectory])}"
    if "trajectory_flow_direction" in data:
        direction = _trajectory_direction(data, trajectory)
        description += (
            ", canonical direction"
            if direction >= 0.0
            else ", reversed direction"
        )
    return description


def _direction_partner(
    data: Dict[str, np.ndarray],
    trajectory: int,
) -> Optional[int]:
    """Return the opposite-direction member of a matched pair, if present."""
    if (
        "trajectory_pair_id" not in data
        or "trajectory_flow_direction" not in data
    ):
        return None
    pair_ids = np.asarray(data["trajectory_pair_id"], dtype=np.int64)
    directions = np.asarray(
        data["trajectory_flow_direction"],
        dtype=np.float64,
    )
    candidates = np.flatnonzero(
        (pair_ids == pair_ids[trajectory])
        & (directions * directions[trajectory] < 0.0)
    )
    if candidates.size == 0:
        return None
    return int(candidates[0])


def _time_edges(num_times: int, dt: float) -> np.ndarray:
    """Construct plotting edges centered around saved times."""
    times = np.arange(num_times, dtype=np.float64) * dt
    if num_times == 1:
        return np.array([-0.5 * dt, 0.5 * dt], dtype=np.float64)
    edges = np.empty(num_times + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (times[:-1] + times[1:])
    edges[0] = times[0] - 0.5 * dt
    edges[-1] = times[-1] + 0.5 * dt
    return edges


def _default_snapshot_steps(num_times: int) -> List[int]:
    """Choose six distinct saved states spanning the rollout."""
    raw = np.linspace(0, num_times - 1, min(6, num_times))
    return sorted(set(int(round(value)) for value in raw))


def _validate_steps(steps: Sequence[int], num_times: int) -> List[int]:
    """Validate and deduplicate requested saved-state indices."""
    unique = sorted(set(int(step) for step in steps))
    invalid = [step for step in unique if step < 0 or step >= num_times]
    if invalid:
        raise IndexError(
            f"Snapshot steps {invalid} are outside the valid range "
            f"0 to {num_times - 1}."
        )
    return unique


def _road_indices(data: Dict[str, np.ndarray], road: int) -> np.ndarray:
    """Return globally ordered cell indices for one road."""
    offsets = np.asarray(data["road_cell_offsets"], dtype=np.int64)
    return np.arange(offsets[road], offsets[road + 1], dtype=np.int64)


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


def _draw_network_geometry(
    ax: plt.Axes,
    data: Dict[str, np.ndarray],
    *,
    annotate: bool,
) -> None:
    """Draw neutral road geometry, direction arrows, and junctions."""
    road_names = np.asarray(data["road_names"])
    for road in range(road_names.size):
        _, coordinates = _road_coordinates(data, road)
        ax.plot(
            coordinates[:, 0],
            coordinates[:, 1],
            color=ROAD_COLORS[road % len(ROAD_COLORS)],
            linewidth=3.0,
            alpha=0.85,
            solid_capstyle="round",
        )
        _add_direction_arrow(ax, coordinates)
        if annotate:
            midpoint = coordinates[coordinates.shape[0] // 2]
            ax.annotate(
                str(road_names[road]).replace("_", " "),
                midpoint,
                xytext=(4, 5),
                textcoords="offset points",
                fontsize=8,
            )

    junction_positions = np.asarray(
        data["junction_positions"], dtype=np.float64
    )
    ax.scatter(
        junction_positions[:, 0],
        junction_positions[:, 1],
        s=35,
        color="0.1",
        zorder=7,
    )
    if annotate:
        for position, name, kind in zip(
            junction_positions,
            data["junction_names"],
            data["junction_types"],
        ):
            readable_kind = str(kind).replace("_", " ")
            ax.annotate(
                f"{name} ({readable_kind})",
                position,
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")


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


def _format_axes(ax: plt.Axes) -> None:
    """Apply restrained formatting to standard Cartesian axes."""
    ax.grid(True, color="0.86", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_dataset_summary(
    data: Dict[str, np.ndarray],
    output_path: Path,
    *,
    dpi: int,
) -> None:
    """Plot topology, initial-condition sampling, and conservation statistics."""
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(10.2, 7.6),
        constrained_layout=True,
    )
    ax_network, ax_placements, ax_density, ax_mass = axes.ravel()

    road_names = np.asarray(data["road_names"])
    _draw_network_geometry(
        ax_network,
        data,
        annotate=road_names.size <= 10,
    )
    network_title = (
        "Closed road network (canonical orientation)"
        if "trajectory_flow_direction" in data
        else "Closed directed road network"
    )
    ax_network.set_title(network_title)

    if "initial_block_road_id" in data:
        block_roads = np.asarray(data["initial_block_road_id"], dtype=np.int64)
        valid = block_roads >= 0
        placement_count = np.bincount(
            block_roads[valid], minlength=road_names.size
        )
    else:
        placement_count = np.zeros(road_names.size, dtype=np.int64)
    positions = np.arange(road_names.size)
    ax_placements.bar(
        positions,
        placement_count,
        color=[
            ROAD_COLORS[index % len(ROAD_COLORS)]
            for index in range(road_names.size)
        ],
        alpha=0.82,
    )
    ax_placements.set_xticks(positions)
    road_labels = (
        [str(name).replace("_", "\n") for name in road_names]
        if road_names.size <= 10
        else [f"R{index:03d}" for index in positions]
    )
    ax_placements.set_xticklabels(
        road_labels,
        fontsize=7,
        rotation=90 if road_names.size > 10 else 0,
    )
    ax_placements.set_title("Initial congestion-block placement")
    ax_placements.set_ylabel("Block count")
    _format_axes(ax_placements)

    baseline_key = (
        "initial_baseline_density"
        if "initial_baseline_density" in data
        else "initial_road_baseline_density"
    )
    if baseline_key in data:
        baseline = np.asarray(
            data[baseline_key], dtype=np.float64
        ).reshape(-1)
        ax_density.hist(
            baseline,
            bins="auto",
            density=False,
            color="tab:blue",
            alpha=0.72,
            label="Background",
        )
    block_density_key = (
        "initial_block_high_density"
        if "initial_block_high_density" in data
        else "initial_block_density"
    )
    if block_density_key in data:
        high = np.asarray(
            data[block_density_key], dtype=np.float64
        )
        high = high[np.isfinite(high)]
        ax_density.hist(
            high,
            bins="auto",
            density=False,
            color="tab:red",
            alpha=0.58,
            label="Congestion block",
        )
    ax_density.set_title("Sampled initial densities")
    ax_density.set_xlabel(r"Density $\rho$")
    ax_density.set_ylabel("Count")
    handles, labels = ax_density.get_legend_handles_labels()
    if handles:
        ax_density.legend(handles, labels)
    _format_axes(ax_density)

    total_mass = np.asarray(data["total_mass"], dtype=np.float64)
    max_drift = np.max(
        np.abs(total_mass - total_mass[:, :1]), axis=1
    )
    positive_drift = np.maximum(max_drift, np.finfo(np.float64).tiny)
    ax_mass.hist(
        np.log10(positive_drift),
        bins="auto",
        color="tab:purple",
        alpha=0.78,
        edgecolor="white",
        linewidth=0.5,
    )
    ax_mass.set_title("Network-wide conservation")
    ax_mass.set_xlabel(
        r"$\log_{10}\!\left(\max_t|M(t)-M(0)|\right)$"
    )
    ax_mass.set_ylabel("Trajectory count")
    _format_axes(ax_mass)

    summary_title = (
        "Structured direction-reversal LWR dataset summary"
        if "trajectory_flow_direction" in data
        else "Network LWR dataset summary"
    )
    figure.suptitle(summary_title, fontsize=14)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def plot_network_snapshots(
    data: Dict[str, np.ndarray],
    trajectory: int,
    output_path: Path,
    *,
    snapshot_steps: Optional[Sequence[int]],
    dpi: int,
    cmap: str,
) -> None:
    """Plot selected density states on the embedded road network."""
    rollout = data["rollout_states"]
    num_trajectories, num_times, _ = rollout.shape
    if trajectory < 0 or trajectory >= num_trajectories:
        raise IndexError(
            f"Trajectory {trajectory} is outside the valid range "
            f"0 to {num_trajectories - 1}."
        )
    if snapshot_steps is None:
        steps = _default_snapshot_steps(num_times)
    else:
        steps = _validate_steps(snapshot_steps, num_times)

    columns = 3
    rows = int(np.ceil(len(steps) / columns))
    figure, axes_array = plt.subplots(
        rows,
        columns,
        figsize=(10.5, 3.25 * rows),
        constrained_layout=True,
        squeeze=False,
    )
    axes = axes_array.ravel()
    density_min = _scalar(data, "density_min", float(np.min(rollout)))
    density_max = _scalar(data, "density_max", float(np.max(rollout)))
    flow_direction = _trajectory_direction(data, trajectory)
    norm = Normalize(vmin=density_min, vmax=density_max)
    dt = _scalar(data, "dt", 1.0)
    color_collection: Optional[LineCollection] = None

    for axis, step in zip(axes, steps):
        color_collection = _draw_density_network(
            axis,
            data,
            rollout[trajectory, step],
            norm=norm,
            cmap=cmap,
            flow_direction=flow_direction,
        )
        axis.set_title(f"step {step},  $t={step * dt:.4g}$")
    for axis in axes[len(steps) :]:
        axis.set_visible(False)

    if color_collection is not None:
        colorbar = figure.colorbar(
            color_collection,
            ax=list(axes[: len(steps)]),
            pad=0.015,
            shrink=0.90,
        )
        colorbar.set_label(r"Traffic density $\rho$")
    figure.suptitle(
        f"Network LWR {_trajectory_description(data, trajectory)}: "
        "density snapshots",
        fontsize=14,
    )
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def plot_road_spacetime(
    data: Dict[str, np.ndarray],
    trajectory: int,
    output_path: Path,
    *,
    dpi: int,
    cmap: str,
) -> None:
    """Plot one aligned space-time density diagram for every road."""
    rollout = data["rollout_states"]
    num_trajectories, num_times, _ = rollout.shape
    if trajectory < 0 or trajectory >= num_trajectories:
        raise IndexError(
            f"Trajectory {trajectory} is outside the valid range "
            f"0 to {num_trajectories - 1}."
        )

    road_names = np.asarray(data["road_names"])
    dt = _scalar(data, "dt", 1.0)
    time_edges = _time_edges(num_times, dt)
    density_min = _scalar(data, "density_min", float(np.min(rollout)))
    density_max = _scalar(data, "density_max", float(np.max(rollout)))
    flow_direction = _trajectory_direction(data, trajectory)

    columns = 2
    rows = int(np.ceil(road_names.size / columns))
    figure, axes_array = plt.subplots(
        rows,
        columns,
        figsize=(10.0, 3.45 * rows),
        constrained_layout=True,
        squeeze=False,
        sharey=True,
    )
    axes = axes_array.ravel()
    mesh = None

    for road, axis in enumerate(axes[: road_names.size]):
        indices = _road_indices(data, road)
        coordinates = np.asarray(
            data["cell_road_coordinate"], dtype=np.float64
        )[indices]
        lengths = np.asarray(data["cell_lengths"], dtype=np.float64)[indices]
        spatial_edges = np.concatenate(
            [
                coordinates - 0.5 * lengths,
                np.array([coordinates[-1] + 0.5 * lengths[-1]]),
            ]
        )
        mesh = axis.pcolormesh(
            spatial_edges,
            time_edges,
            rollout[trajectory][:, indices],
            shading="flat",
            cmap=cmap,
            vmin=density_min,
            vmax=density_max,
            rasterized=True,
        )
        axis.set_title(str(road_names[road]).replace("_", " "))
        axis.set_xlabel("Canonical road coordinate")
        road_length = float(data["road_lengths"][road])
        axis.set_xlim(
            (0.0, road_length)
            if flow_direction >= 0.0
            else (road_length, 0.0)
        )
        axis.set_ylim(time_edges[-1], max(0.0, time_edges[0]))
        _format_axes(axis)
    for axis in axes[road_names.size :]:
        axis.set_visible(False)
    for axis in axes[::columns]:
        axis.set_ylabel("Time")

    if mesh is not None:
        colorbar = figure.colorbar(
            mesh,
            ax=list(axes[: road_names.size]),
            pad=0.015,
            shrink=0.92,
        )
        colorbar.set_label(r"Traffic density $\rho$")
    figure.suptitle(
        f"Network LWR {_trajectory_description(data, trajectory)}: "
        "roadwise evolution",
        fontsize=14,
    )
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _junction_flow_label(
    data: Dict[str, np.ndarray],
    transfer: int,
    *,
    flow_direction: float,
) -> str:
    """Create a compact junction and road-ID movement label."""
    junction = int(data["transfer_junction_id"][transfer])
    source_road = int(data["transfer_from_road"][transfer])
    target_road = int(data["transfer_to_road"][transfer])
    if flow_direction < 0.0:
        source_road, target_road = target_road, source_road
    return f"J{junction:03d}: R{source_road:03d} \u2192 R{target_road:03d}"


def plot_trajectory_diagnostics(
    data: Dict[str, np.ndarray],
    trajectory: int,
    output_path: Path,
    *,
    dpi: int,
) -> None:
    """Plot compact flow, road-mass, conservation, and pair diagnostics."""
    rollout = data["rollout_states"]
    num_trajectories, num_times, _ = rollout.shape
    if trajectory < 0 or trajectory >= num_trajectories:
        raise IndexError(
            f"Trajectory {trajectory} is outside the valid range "
            f"0 to {num_trajectories - 1}."
        )

    dt = _scalar(data, "dt", 1.0)
    state_times = np.arange(num_times, dtype=np.float64) * dt
    flux_times = np.arange(num_times - 1, dtype=np.float64) * dt
    flow_direction = _trajectory_direction(data, trajectory)
    fluxes = (
        data["rollout_transfer_fluxes"][trajectory] * flow_direction
    )
    road_mass = np.asarray(data["road_mass"][trajectory], dtype=np.float64)
    total_mass = np.asarray(data["total_mass"][trajectory], dtype=np.float64)
    road_names = np.asarray(data["road_names"])
    transfer_type = np.asarray(data["transfer_type"], dtype=np.int64)
    transfer_junction = np.asarray(
        data["transfer_junction_id"], dtype=np.int64
    )
    partner = _direction_partner(data, trajectory)

    if partner is None:
        figure = plt.figure(figsize=(12.0, 8.2), constrained_layout=True)
        grid = figure.add_gridspec(2, 2)
    else:
        figure = plt.figure(figsize=(12.0, 10.8), constrained_layout=True)
        grid = figure.add_gridspec(
            3,
            2,
            height_ratios=(1.0, 1.0, 0.72),
        )
    ax_throughput = figure.add_subplot(grid[0, 0])
    ax_flows = figure.add_subplot(grid[0, 1])
    ax_road_mass = figure.add_subplot(grid[1, 0])
    ax_total_mass = figure.add_subplot(grid[1, 1])
    ax_pair = (
        figure.add_subplot(grid[2, :]) if partner is not None else None
    )

    junction_transfers = np.flatnonzero(transfer_type == 1)
    junction_ids = np.unique(transfer_junction[junction_transfers])
    junction_throughput = np.zeros(
        (junction_ids.size, fluxes.shape[0]),
        dtype=np.float64,
    )
    for row, junction in enumerate(junction_ids):
        transfers = junction_transfers[
            transfer_junction[junction_transfers] == junction
        ]
        junction_throughput[row] = np.sum(
            fluxes[:, transfers],
            axis=1,
        )
    flux_edges = np.arange(fluxes.shape[0] + 1, dtype=np.float64) * dt
    junction_edges = np.arange(junction_ids.size + 1, dtype=np.float64)
    throughput_mesh = ax_throughput.pcolormesh(
        flux_edges,
        junction_edges,
        junction_throughput,
        shading="flat",
        cmap="magma",
        rasterized=True,
    )
    junction_stride = max(1, int(np.ceil(junction_ids.size / 20)))
    junction_tick_rows = np.arange(0, junction_ids.size, junction_stride)
    ax_throughput.set_yticks(junction_tick_rows + 0.5)
    ax_throughput.set_yticklabels(
        [f"J{junction_ids[row]:03d}" for row in junction_tick_rows],
        fontsize=8,
    )
    ax_throughput.set_title("Total throughput by intersection")
    ax_throughput.set_xlabel("Time")
    ax_throughput.set_ylabel("Intersection")
    figure.colorbar(
        throughput_mesh,
        ax=ax_throughput,
        pad=0.02,
        label="Total vehicle flow rate",
    )

    busy_count = min(6, junction_transfers.size)
    movement_activity = np.mean(
        np.abs(fluxes[:, junction_transfers]),
        axis=0,
    )
    busy_local = np.argsort(movement_activity)[-busy_count:][::-1]
    busy_transfers = junction_transfers[busy_local]
    for transfer in busy_transfers:
        ax_flows.plot(
            flux_times,
            fluxes[:, transfer],
            linewidth=1.35,
            label=_junction_flow_label(
                data,
                int(transfer),
                flow_direction=flow_direction,
            ),
        )
    ax_flows.set_title(f"{busy_count} busiest junction movements")
    ax_flows.set_xlabel("Time")
    ax_flows.set_ylabel("Vehicle flow rate")
    if busy_count:
        ax_flows.legend(fontsize=8)
    _format_axes(ax_flows)

    state_edges = _time_edges(num_times, dt)
    road_edges = np.arange(road_names.size + 1, dtype=np.float64)
    road_mass_mesh = ax_road_mass.pcolormesh(
        state_edges,
        road_edges,
        road_mass.T,
        shading="flat",
        cmap="viridis",
        rasterized=True,
    )
    road_stride = max(1, int(np.ceil(road_names.size / 24)))
    road_tick_rows = np.arange(0, road_names.size, road_stride)
    ax_road_mass.set_yticks(road_tick_rows + 0.5)
    ax_road_mass.set_yticklabels(
        [f"R{road:03d}" for road in road_tick_rows],
        fontsize=8,
    )
    ax_road_mass.set_title("Vehicle amount by road")
    ax_road_mass.set_xlabel("Time")
    ax_road_mass.set_ylabel("Road")
    figure.colorbar(
        road_mass_mesh,
        ax=ax_road_mass,
        pad=0.02,
        label=r"$\sum_{i\in r}\rho_i\Delta x_i$",
    )

    mass_drift = total_mass - total_mass[0]
    ax_total_mass.plot(
        state_times,
        mass_drift,
        color="tab:purple",
        linewidth=1.35,
    )
    ax_total_mass.axhline(0.0, color="0.30", linewidth=0.8)
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 2))
    ax_total_mass.yaxis.set_major_formatter(formatter)
    ax_total_mass.set_title("Network-wide conservation")
    ax_total_mass.set_xlabel("Time")
    ax_total_mass.set_ylabel(r"$M(t)-M(0)$")
    _format_axes(ax_total_mass)

    if ax_pair is not None and partner is not None:
        pair_error = rollout[trajectory] - rollout[partner]
        pair_rmse = np.sqrt(np.mean(pair_error**2, axis=1))
        ax_pair.plot(
            state_times,
            pair_rmse,
            color="tab:blue",
            linewidth=1.6,
        )
        ax_pair.axhline(0.0, color="0.30", linewidth=0.8)
        ax_pair.set_title(
            "Matched direction-pair divergence: "
            f"trajectory {trajectory} versus {partner}"
        )
        ax_pair.set_xlabel("Time")
        ax_pair.set_ylabel("Node-density RMSE")
        ax_pair.text(
            0.99,
            0.06,
            f"final RMSE = {pair_rmse[-1]:.4g}",
            transform=ax_pair.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
        )
        _format_axes(ax_pair)

    figure.suptitle(
        f"Network LWR {_trajectory_description(data, trajectory)}: "
        "dataset diagnostics",
        fontsize=14,
    )
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create figures for a network LWR .npz dataset."
    )
    parser.add_argument("dataset", type=Path, help="Input network LWR dataset.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for generated figures. Defaults to "
            "plots/<dataset filename without extension>."
        ),
    )
    parser.add_argument(
        "--trajectories",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Trajectory indices to plot. By default, plots trajectory 0 "
            "and its reversed partner when direction pairs are present."
        ),
    )
    parser.add_argument(
        "--snapshot-steps",
        nargs="+",
        type=int,
        default=None,
        help="Saved-state indices used in every network snapshot figure.",
    )
    parser.add_argument(
        "--cmap",
        default="viridis",
        help="Matplotlib colormap used for density (default: viridis).",
    )
    parser.add_argument(
        "--format",
        choices=("png", "pdf", "svg"),
        default="png",
        help="Output figure format (default: png).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Raster output resolution (default: 200).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if not args.dataset.is_file():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive.")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path("plots") / args.dataset.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    data = _load_dataset(args.dataset)
    trajectories = args.trajectories
    if trajectories is None:
        trajectories = (
            [0, 1]
            if (
                "trajectory_flow_direction" in data
                and data["rollout_states"].shape[0] >= 2
            )
            else [0]
        )

    suffix = args.format
    written: List[Path] = []
    summary_path = output_dir / f"dataset_summary.{suffix}"
    plot_dataset_summary(data, summary_path, dpi=args.dpi)
    written.append(summary_path)

    for trajectory in trajectories:
        snapshots_path = (
            output_dir / f"trajectory_{trajectory:04d}_network_snapshots.{suffix}"
        )
        spacetime_path = (
            output_dir / f"trajectory_{trajectory:04d}_road_spacetime.{suffix}"
        )
        diagnostics_path = (
            output_dir / f"trajectory_{trajectory:04d}_diagnostics.{suffix}"
        )
        plot_network_snapshots(
            data,
            trajectory,
            snapshots_path,
            snapshot_steps=args.snapshot_steps,
            dpi=args.dpi,
            cmap=args.cmap,
        )
        plot_road_spacetime(
            data,
            trajectory,
            spacetime_path,
            dpi=args.dpi,
            cmap=args.cmap,
        )
        plot_trajectory_diagnostics(
            data,
            trajectory,
            diagnostics_path,
            dpi=args.dpi,
        )
        written.extend([snapshots_path, spacetime_path, diagnostics_path])

    print(f"Generated {len(written)} figure(s):")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
