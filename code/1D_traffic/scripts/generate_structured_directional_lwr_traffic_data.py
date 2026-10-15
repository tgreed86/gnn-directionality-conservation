#!/usr/bin/env python3
"""Generate closed structured LWR networks with paired flow directions.

Each topology begins as a rectangular street grid containing a protected
directed Hamiltonian cycle. Additional grid streets make the initial layout
checkerboard-like. Complexity is introduced by:

* removing nonessential grid roads;
* subdividing roads with new intersections;
* adding diagonal roads between neighboring grid intersections; and
* adding cell-center intersections with four connecting roads.

The protected cycle is never removed, so every generated road network remains
closed and strongly connected. Every initial condition is simulated twice on
the same topology: once in the canonical road direction and once with every
road direction reversed. The paired trajectories share the same initial
density and speed magnitude. A trajectory-level ``flow_direction`` node
feature (+1 or -1) makes the two dynamics identifiable to a learning model.

The output uses the ``variable_topology_lwr_v1`` manifest contract consumed by
``scripts/train.py`` and ``scripts/plot_variable_sageconv_rollouts.py``.

Example:

    python scripts/generate_structured_directional_lwr_traffic_data.py \
        --out-dir data/lwr_structured_directional \
        --window 256 \
        --seed 1234
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from generate_network_lwr_traffic_data import (
    lwr_demand,
    lwr_godunov_flux,
    lwr_supply,
)


EDGE_ATTR_COLUMNS = (
    "traffic_direction,dx_position,dy_position,euclidean_distance,"
    "physical_center_distance,source_cell_length,target_cell_length,"
    "is_junction,turning_fraction,merge_priority"
)
NODE_CONDITIONING_COLUMNS = (
    "v_max,turning_fraction,has_turning_fraction,"
    "merge_priority,has_merge_priority,flow_direction"
)
SPLIT_NAMES = (
    "train",
    "validation",
    "test_in_size",
    "test_large",
    "test_parameter_ood",
)


@dataclass
class RoadSpec:
    """One canonically oriented geometric road."""

    start: int
    end: int
    control: np.ndarray
    kind: str
    protected: bool


@dataclass
class StructuredNetwork:
    """Finite-volume representation of one edited structured grid."""

    grid_rows: int
    grid_cols: int
    junction_positions: np.ndarray
    junction_types: np.ndarray
    road_start_junction: np.ndarray
    road_end_junction: np.ndarray
    road_kinds: np.ndarray
    road_lengths: np.ndarray
    road_num_cells: np.ndarray
    road_cell_offsets: np.ndarray
    cell_lengths: np.ndarray
    cell_road_id: np.ndarray
    cell_local_index: np.ndarray
    cell_road_coordinate: np.ndarray
    cell_positions: np.ndarray
    transfer_index: np.ndarray
    transfer_type: np.ndarray
    transfer_junction_id: np.ndarray
    transfer_from_road: np.ndarray
    transfer_to_road: np.ndarray
    transfer_center_distance: np.ndarray
    internal_transfer_ids: List[np.ndarray]
    movement_transfer_ids: List[np.ndarray]
    incoming_roads: List[np.ndarray]
    outgoing_roads: List[np.ndarray]
    incidence: np.ndarray
    edge_index: np.ndarray
    edge_attr_base: np.ndarray
    edge_transfer_id: np.ndarray
    edge_is_physical_direction: np.ndarray
    num_removed_roads: int
    num_subdivided_roads: int
    num_diagonal_roads: int
    num_center_intersections: int


def quadratic_bezier(
    start: np.ndarray,
    control: np.ndarray,
    end: np.ndarray,
    parameter: np.ndarray,
) -> np.ndarray:
    """Evaluate a quadratic Bézier curve."""
    t = parameter[:, None]
    return (
        (1.0 - t) ** 2 * start
        + 2.0 * (1.0 - t) * t * control
        + t**2 * end
    )


def bezier_length(
    start: np.ndarray,
    control: np.ndarray,
    end: np.ndarray,
) -> float:
    """Approximate a quadratic Bézier arc length."""
    parameter = np.linspace(0.0, 1.0, 65)
    points = quadratic_bezier(start, control, end, parameter)
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def grid_node(row: int, col: int, cols: int) -> int:
    """Return the flattened junction index for a grid coordinate."""
    return row * cols + col


def is_strongly_connected(
    num_junctions: int,
    road_start: np.ndarray,
    road_end: np.ndarray,
) -> bool:
    """Return whether the canonical directed junction graph is connected."""
    forward: List[List[int]] = [[] for _ in range(num_junctions)]
    reverse: List[List[int]] = [[] for _ in range(num_junctions)]
    for start, end in zip(road_start, road_end):
        forward[int(start)].append(int(end))
        reverse[int(end)].append(int(start))

    def reachable(adjacency: Sequence[Sequence[int]]) -> np.ndarray:
        visited = np.zeros(num_junctions, dtype=np.bool_)
        visited[0] = True
        stack = [0]
        while stack:
            junction = stack.pop()
            for neighbor in adjacency[junction]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
        return visited

    return bool(
        np.all(reachable(forward))
        and np.all(reachable(reverse))
    )


def straight_road(
    start: int,
    end: int,
    positions: Sequence[np.ndarray],
    *,
    kind: str,
    protected: bool,
) -> RoadSpec:
    """Construct a straight road specification."""
    control = 0.5 * (
        np.asarray(positions[start], dtype=np.float64)
        + np.asarray(positions[end], dtype=np.float64)
    )
    return RoadSpec(start, end, control, kind, protected)


def add_base_grid(
    rows: int,
    cols: int,
    spacing: float,
    rng: np.random.Generator,
) -> Tuple[List[np.ndarray], List[RoadSpec]]:
    """Create a checkerboard grid containing a protected directed cycle."""
    positions = [
        np.array((col * spacing, row * spacing), dtype=np.float64)
        for row in range(rows)
        for col in range(cols)
    ]
    roads: List[RoadSpec] = []
    existing_pairs = set()

    def add(
        start: int,
        end: int,
        *,
        kind: str,
        protected: bool,
        control: Optional[np.ndarray] = None,
    ) -> None:
        pair = tuple(sorted((start, end)))
        if pair in existing_pairs:
            return
        existing_pairs.add(pair)
        if control is None:
            roads.append(
                straight_road(
                    start,
                    end,
                    positions,
                    kind=kind,
                    protected=protected,
                )
            )
        else:
            roads.append(
                RoadSpec(start, end, control, kind, protected)
            )

    path: List[int] = []
    for row in range(rows):
        columns = range(cols) if row % 2 == 0 else range(cols - 1, -1, -1)
        path.extend(grid_node(row, col, cols) for col in columns)
    for start, end in zip(path[:-1], path[1:]):
        add(start, end, kind="protected_grid", protected=True)

    closure_start = path[-1]
    closure_end = path[0]
    start_position = positions[closure_start]
    end_position = positions[closure_end]
    side = -1.0 if start_position[0] <= 0.5 * (cols - 1) * spacing else 1.0
    control = 0.5 * (start_position + end_position)
    control[0] = (
        -1.5 * spacing
        if side < 0.0
        else (cols - 1) * spacing + 1.5 * spacing
    )
    add(
        closure_start,
        closure_end,
        kind="closure",
        protected=True,
        control=control,
    )

    for row in range(rows - 1):
        for col in range(cols):
            first = grid_node(row, col, cols)
            second = grid_node(row + 1, col, cols)
            pair = tuple(sorted((first, second)))
            if pair in existing_pairs:
                continue
            if rng.random() < 0.5:
                first, second = second, first
            add(first, second, kind="grid", protected=False)
    return positions, roads


def remove_grid_roads(
    roads: List[RoadSpec],
    fraction: float,
    rng: np.random.Generator,
) -> Tuple[List[RoadSpec], int]:
    """Remove a fraction of nonprotected checkerboard roads."""
    candidates = [
        index
        for index, road in enumerate(roads)
        if not road.protected and road.kind == "grid"
    ]
    count = min(
        len(candidates),
        int(round(fraction * len(candidates))),
    )
    if count == 0:
        return roads, 0
    removed = set(
        int(index)
        for index in rng.choice(candidates, size=count, replace=False)
    )
    return [
        road for index, road in enumerate(roads) if index not in removed
    ], count


def add_diagonal_roads(
    positions: List[np.ndarray],
    roads: List[RoadSpec],
    *,
    rows: int,
    cols: int,
    fraction: float,
    rng: np.random.Generator,
) -> int:
    """Add one diagonal to a selected fraction of grid cells."""
    cells = [
        (row, col)
        for row in range(rows - 1)
        for col in range(cols - 1)
    ]
    count = min(len(cells), int(round(fraction * len(cells))))
    if count == 0:
        return 0
    selected = rng.choice(len(cells), size=count, replace=False)
    existing_pairs = {
        tuple(sorted((road.start, road.end))) for road in roads
    }
    added = 0
    for index in selected:
        row, col = cells[int(index)]
        if rng.random() < 0.5:
            first = grid_node(row, col, cols)
            second = grid_node(row + 1, col + 1, cols)
        else:
            first = grid_node(row, col + 1, cols)
            second = grid_node(row + 1, col, cols)
        pair = tuple(sorted((first, second)))
        if pair in existing_pairs:
            continue
        if rng.random() < 0.5:
            first, second = second, first
        roads.append(
            straight_road(
                first,
                second,
                positions,
                kind="diagonal",
                protected=False,
            )
        )
        existing_pairs.add(pair)
        added += 1
    return added


def add_center_intersections(
    positions: List[np.ndarray],
    roads: List[RoadSpec],
    *,
    rows: int,
    cols: int,
    fraction: float,
    rng: np.random.Generator,
) -> int:
    """Add four-road center intersections to selected checkerboard cells."""
    cells = [
        (row, col)
        for row in range(rows - 1)
        for col in range(cols - 1)
    ]
    count = min(len(cells), int(round(fraction * len(cells))))
    if count == 0:
        return 0
    selected = rng.choice(len(cells), size=count, replace=False)
    for index in selected:
        row, col = cells[int(index)]
        corners = (
            grid_node(row, col, cols),
            grid_node(row, col + 1, cols),
            grid_node(row + 1, col + 1, cols),
            grid_node(row + 1, col, cols),
        )
        center_position = np.mean(
            np.stack([positions[corner] for corner in corners]),
            axis=0,
        )
        center = len(positions)
        positions.append(center_position)
        phase = int(rng.integers(0, 2))
        for local_index, corner in enumerate(corners):
            if (local_index + phase) % 2 == 0:
                start, end = corner, center
            else:
                start, end = center, corner
            roads.append(
                straight_road(
                    start,
                    end,
                    positions,
                    kind="center_spoke",
                    protected=False,
                )
            )
    return count


def subdivide_roads(
    positions: List[np.ndarray],
    roads: List[RoadSpec],
    fraction: float,
    rng: np.random.Generator,
) -> Tuple[List[RoadSpec], int]:
    """Replace selected roads by two roads meeting at a new intersection."""
    count = min(len(roads), int(round(fraction * len(roads))))
    if count == 0:
        return roads, 0
    selected = set(
        int(index)
        for index in rng.choice(len(roads), size=count, replace=False)
    )
    result: List[RoadSpec] = []
    for index, road in enumerate(roads):
        if index not in selected:
            result.append(road)
            continue
        start = positions[road.start]
        end = positions[road.end]
        first_control = 0.5 * (start + road.control)
        midpoint = 0.25 * start + 0.5 * road.control + 0.25 * end
        second_control = 0.5 * (road.control + end)
        middle = len(positions)
        positions.append(midpoint)
        result.extend(
            (
                RoadSpec(
                    road.start,
                    middle,
                    first_control,
                    f"subdivided_{road.kind}",
                    road.protected,
                ),
                RoadSpec(
                    middle,
                    road.end,
                    second_control,
                    f"subdivided_{road.kind}",
                    road.protected,
                ),
            )
        )
    return result, count


def build_structured_network(
    *,
    rows: int,
    cols: int,
    spacing: float,
    cells_per_unit: float,
    minimum_cells_per_road: int,
    remove_fraction: float,
    subdivide_fraction: float,
    diagonal_fraction: float,
    center_fraction: float,
    rng: np.random.Generator,
) -> StructuredNetwork:
    """Build one edited checkerboard topology and its finite-volume graph."""
    positions, roads = add_base_grid(rows, cols, spacing, rng)
    roads, removed = remove_grid_roads(roads, remove_fraction, rng)
    diagonals = add_diagonal_roads(
        positions,
        roads,
        rows=rows,
        cols=cols,
        fraction=diagonal_fraction,
        rng=rng,
    )
    centers = add_center_intersections(
        positions,
        roads,
        rows=rows,
        cols=cols,
        fraction=center_fraction,
        rng=rng,
    )
    roads, subdivisions = subdivide_roads(
        positions,
        roads,
        subdivide_fraction,
        rng,
    )

    junction_positions = np.asarray(positions, dtype=np.float64)
    road_start = np.asarray([road.start for road in roads], dtype=np.int64)
    road_end = np.asarray([road.end for road in roads], dtype=np.int64)
    road_kinds = np.asarray([road.kind for road in roads], dtype="U40")
    num_junctions = junction_positions.shape[0]
    incoming_roads = [
        np.flatnonzero(road_end == junction).astype(np.int64)
        for junction in range(num_junctions)
    ]
    outgoing_roads = [
        np.flatnonzero(road_start == junction).astype(np.int64)
        for junction in range(num_junctions)
    ]
    for junction, (incoming, outgoing) in enumerate(
        zip(incoming_roads, outgoing_roads)
    ):
        if incoming.size == 0 or outgoing.size == 0:
            raise RuntimeError(
                f"Junction {junction} is not closed in both directions."
            )
    if not is_strongly_connected(num_junctions, road_start, road_end):
        raise RuntimeError(
            "The protected road cycle failed to preserve strong "
            "connectivity."
        )
    junction_types = np.asarray(
        [
            f"{incoming.size}_in_{outgoing.size}_out"
            for incoming, outgoing in zip(incoming_roads, outgoing_roads)
        ],
        dtype="U32",
    )

    road_lengths = np.asarray(
        [
            bezier_length(
                junction_positions[road.start],
                road.control,
                junction_positions[road.end],
            )
            for road in roads
        ],
        dtype=np.float64,
    )
    road_num_cells = np.maximum(
        minimum_cells_per_road,
        np.rint(road_lengths * cells_per_unit).astype(np.int64),
    )
    road_offsets = np.concatenate(
        (
            np.array([0], dtype=np.int64),
            np.cumsum(road_num_cells, dtype=np.int64),
        )
    )
    num_cells = int(road_offsets[-1])
    cell_lengths = np.empty(num_cells, dtype=np.float64)
    cell_road_id = np.empty(num_cells, dtype=np.int64)
    cell_local_index = np.empty(num_cells, dtype=np.int64)
    cell_road_coordinate = np.empty(num_cells, dtype=np.float64)
    cell_positions = np.empty((num_cells, 2), dtype=np.float64)
    for road_id, road in enumerate(roads):
        start = int(road_offsets[road_id])
        end = int(road_offsets[road_id + 1])
        count = int(road_num_cells[road_id])
        length = float(road_lengths[road_id])
        dx = length / count
        coordinate = (np.arange(count, dtype=np.float64) + 0.5) * dx
        parameter = coordinate / length
        cell_lengths[start:end] = dx
        cell_road_id[start:end] = road_id
        cell_local_index[start:end] = np.arange(count, dtype=np.int64)
        cell_road_coordinate[start:end] = coordinate
        cell_positions[start:end] = quadratic_bezier(
            junction_positions[road.start],
            road.control,
            junction_positions[road.end],
            parameter,
        )

    transfer_sources: List[int] = []
    transfer_targets: List[int] = []
    transfer_types: List[int] = []
    transfer_junctions: List[int] = []
    transfer_from_roads: List[int] = []
    transfer_to_roads: List[int] = []
    internal_transfer_ids: List[np.ndarray] = []
    for road in range(len(roads)):
        start = int(road_offsets[road])
        end = int(road_offsets[road + 1])
        ids: List[int] = []
        for source in range(start, end - 1):
            ids.append(len(transfer_sources))
            transfer_sources.append(source)
            transfer_targets.append(source + 1)
            transfer_types.append(0)
            transfer_junctions.append(-1)
            transfer_from_roads.append(road)
            transfer_to_roads.append(road)
        internal_transfer_ids.append(np.asarray(ids, dtype=np.int64))

    movement_transfer_ids: List[np.ndarray] = []
    for junction in range(num_junctions):
        incoming = incoming_roads[junction]
        outgoing = outgoing_roads[junction]
        ids = np.empty((incoming.size, outgoing.size), dtype=np.int64)
        for incoming_index, incoming_road in enumerate(incoming):
            for outgoing_index, outgoing_road in enumerate(outgoing):
                transfer_id = len(transfer_sources)
                ids[incoming_index, outgoing_index] = transfer_id
                transfer_sources.append(
                    int(road_offsets[incoming_road + 1] - 1)
                )
                transfer_targets.append(int(road_offsets[outgoing_road]))
                transfer_types.append(1)
                transfer_junctions.append(junction)
                transfer_from_roads.append(int(incoming_road))
                transfer_to_roads.append(int(outgoing_road))
        movement_transfer_ids.append(ids)

    transfer_index = np.stack(
        (
            np.asarray(transfer_sources, dtype=np.int64),
            np.asarray(transfer_targets, dtype=np.int64),
        )
    )
    transfer_type = np.asarray(transfer_types, dtype=np.int64)
    transfer_junction_id = np.asarray(
        transfer_junctions,
        dtype=np.int64,
    )
    transfer_from_road = np.asarray(
        transfer_from_roads,
        dtype=np.int64,
    )
    transfer_to_road = np.asarray(transfer_to_roads, dtype=np.int64)
    transfer_distance = 0.5 * (
        cell_lengths[transfer_index[0]]
        + cell_lengths[transfer_index[1]]
    )
    num_transfers = transfer_index.shape[1]
    incidence = np.zeros((num_transfers, num_cells), dtype=np.float64)
    incidence[np.arange(num_transfers), transfer_index[0]] = -1.0
    incidence[np.arange(num_transfers), transfer_index[1]] = 1.0

    edge_sources = np.repeat(transfer_index[0], 2)
    edge_targets = np.repeat(transfer_index[1], 2)
    edge_sources[1::2] = transfer_index[1]
    edge_targets[1::2] = transfer_index[0]
    edge_index = np.stack((edge_sources, edge_targets))
    traffic_direction = np.tile(
        np.array((1.0, -1.0), dtype=np.float64),
        num_transfers,
    )
    displacement = cell_positions[edge_targets] - cell_positions[edge_sources]
    edge_attr_base = np.column_stack(
        (
            traffic_direction,
            displacement[:, 0],
            displacement[:, 1],
            np.linalg.norm(displacement, axis=1),
            np.repeat(transfer_distance, 2),
            cell_lengths[edge_sources],
            cell_lengths[edge_targets],
            np.repeat(transfer_type.astype(np.float64), 2),
            np.zeros(edge_sources.size, dtype=np.float64),
            np.zeros(edge_sources.size, dtype=np.float64),
        )
    )
    return StructuredNetwork(
        grid_rows=rows,
        grid_cols=cols,
        junction_positions=junction_positions,
        junction_types=junction_types,
        road_start_junction=road_start,
        road_end_junction=road_end,
        road_kinds=road_kinds,
        road_lengths=road_lengths,
        road_num_cells=road_num_cells,
        road_cell_offsets=road_offsets,
        cell_lengths=cell_lengths,
        cell_road_id=cell_road_id,
        cell_local_index=cell_local_index,
        cell_road_coordinate=cell_road_coordinate,
        cell_positions=cell_positions,
        transfer_index=transfer_index,
        transfer_type=transfer_type,
        transfer_junction_id=transfer_junction_id,
        transfer_from_road=transfer_from_road,
        transfer_to_road=transfer_to_road,
        transfer_center_distance=transfer_distance,
        internal_transfer_ids=internal_transfer_ids,
        movement_transfer_ids=movement_transfer_ids,
        incoming_roads=incoming_roads,
        outgoing_roads=outgoing_roads,
        incidence=incidence,
        edge_index=edge_index,
        edge_attr_base=edge_attr_base,
        edge_transfer_id=np.repeat(
            np.arange(num_transfers, dtype=np.int64),
            2,
        ),
        edge_is_physical_direction=np.tile(
            np.array((True, False)),
            num_transfers,
        ),
        num_removed_roads=removed,
        num_subdivided_roads=subdivisions,
        num_diagonal_roads=diagonals,
        num_center_intersections=centers,
    )


def sample_union(
    low_interval: Tuple[float, float],
    high_interval: Tuple[float, float],
    shape: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample values from a two-interval union."""
    choose_high = rng.random(shape) >= 0.5
    low = rng.uniform(*low_interval, size=shape)
    high = rng.uniform(*high_interval, size=shape)
    return np.where(choose_high, high, low)


def generate_paired_initial_conditions(
    *,
    network: StructuredNetwork,
    num_trajectories: int,
    num_blocks_min: int,
    num_blocks_max: int,
    road_baseline_jitter: float,
    rho_max: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Create identical initial conditions for forward/reverse pairs."""
    if num_trajectories % 2 != 0:
        raise ValueError("Direction-paired trajectory count must be even.")
    num_pairs = num_trajectories // 2
    num_roads = network.road_lengths.size
    base_states = np.empty(
        (num_pairs, network.cell_lengths.size),
        dtype=np.float64,
    )
    backgrounds = np.empty((num_pairs, num_roads), dtype=np.float64)
    block_count = np.empty(num_pairs, dtype=np.int64)
    block_road = np.full(
        (num_pairs, num_blocks_max),
        -1,
        dtype=np.int64,
    )
    block_start = np.full(
        (num_pairs, num_blocks_max),
        np.nan,
        dtype=np.float64,
    )
    block_width = np.full_like(block_start, np.nan)
    block_density = np.full_like(block_start, np.nan)

    for pair in range(num_pairs):
        central = float(rng.uniform(0.04, 0.32) * rho_max)
        road_background = np.clip(
            central
            + rng.uniform(
                -road_baseline_jitter,
                road_baseline_jitter,
                size=num_roads,
            )
            * rho_max,
            0.02 * rho_max,
            0.45 * rho_max,
        )
        backgrounds[pair] = road_background
        for road in range(num_roads):
            start = int(network.road_cell_offsets[road])
            end = int(network.road_cell_offsets[road + 1])
            base_states[pair, start:end] = road_background[road]

        count = int(rng.integers(num_blocks_min, num_blocks_max + 1))
        block_count[pair] = count
        for block in range(count):
            road = int(rng.integers(0, num_roads))
            road_length = float(network.road_lengths[road])
            width = float(road_length * rng.uniform(0.08, 0.38))
            start_position = float(
                rng.uniform(0.0, max(road_length - width, 0.0))
            )
            end_position = start_position + width
            high_density = float(rng.uniform(0.58, 0.96) * rho_max)
            global_start = int(network.road_cell_offsets[road])
            global_end = int(network.road_cell_offsets[road + 1])
            coordinates = network.cell_road_coordinate[
                global_start:global_end
            ]
            mask = (
                (coordinates >= start_position)
                & (coordinates <= end_position)
            )
            base_states[pair, global_start:global_end][mask] = high_density
            block_road[pair, block] = road
            block_start[pair, block] = start_position
            block_width[pair, block] = width
            block_density[pair, block] = high_density

    rollout_initial = np.repeat(base_states, 2, axis=0)
    directions = np.tile(
        np.array((1.0, -1.0), dtype=np.float64),
        num_pairs,
    )
    metadata = {
        "trajectory_pair_id": np.repeat(
            np.arange(num_pairs, dtype=np.int64),
            2,
        ),
        "initial_road_baseline_density": np.repeat(
            backgrounds,
            2,
            axis=0,
        ),
        "initial_block_count": np.repeat(block_count, 2),
        "initial_block_road_id": np.repeat(block_road, 2, axis=0),
        "initial_block_start": np.repeat(block_start, 2, axis=0),
        "initial_block_width": np.repeat(block_width, 2, axis=0),
        "initial_block_density": np.repeat(block_density, 2, axis=0),
    }
    return rollout_initial, directions, metadata


def allocate_equal_turning(
    demand: np.ndarray,
    supply: np.ndarray,
) -> np.ndarray:
    """Allocate fixed equal turning flows subject to outgoing supplies."""
    if demand.ndim != 1 or supply.ndim != 1:
        raise ValueError("Junction demand and supply must be vectors.")
    if demand.size == 0 or supply.size == 0:
        raise ValueError("Closed junctions require incoming and outgoing roads.")
    desired = np.repeat(
        (demand / supply.size)[:, None],
        supply.size,
        axis=1,
    )
    incoming_to_output = np.sum(desired, axis=0)
    scale = np.minimum(
        1.0,
        np.divide(
            supply,
            incoming_to_output,
            out=np.ones_like(supply),
            where=incoming_to_output > 0.0,
        ),
    )
    return desired * scale[None, :]


def compute_transfer_fluxes(
    state: np.ndarray,
    *,
    network: StructuredNetwork,
    v_max: np.ndarray,
    direction: np.ndarray,
    rho_max: float,
) -> np.ndarray:
    """Compute signed canonical transfer fluxes for both flow directions."""
    batch_size = state.shape[0]
    fluxes = np.zeros(
        (batch_size, network.transfer_index.shape[1]),
        dtype=np.float64,
    )
    forward = direction > 0.0
    for road, transfer_ids in enumerate(network.internal_transfer_ids):
        start = int(network.road_cell_offsets[road])
        end = int(network.road_cell_offsets[road + 1])
        left = state[:, start : end - 1]
        right = state[:, start + 1 : end]
        forward_flux = lwr_godunov_flux(
            left,
            right,
            v_max=v_max[:, None],
            rho_max=rho_max,
        )
        reverse_flux = lwr_godunov_flux(
            right,
            left,
            v_max=v_max[:, None],
            rho_max=rho_max,
        )
        fluxes[:, transfer_ids] = np.where(
            forward[:, None],
            forward_flux,
            -reverse_flux,
        )

    for junction, movement_ids in enumerate(
        network.movement_transfer_ids
    ):
        base_incoming = network.incoming_roads[junction]
        base_outgoing = network.outgoing_roads[junction]
        incoming_end_cells = (
            network.road_cell_offsets[base_incoming + 1] - 1
        )
        outgoing_start_cells = network.road_cell_offsets[base_outgoing]
        for trajectory in range(batch_size):
            speed = float(v_max[trajectory])
            if direction[trajectory] > 0.0:
                demand = lwr_demand(
                    state[trajectory, incoming_end_cells],
                    v_max=speed,
                    rho_max=rho_max,
                )
                supply = lwr_supply(
                    state[trajectory, outgoing_start_cells],
                    v_max=speed,
                    rho_max=rho_max,
                )
                movement = allocate_equal_turning(demand, supply)
                fluxes[trajectory, movement_ids.ravel()] = (
                    movement.ravel()
                )
            else:
                demand = lwr_demand(
                    state[trajectory, outgoing_start_cells],
                    v_max=speed,
                    rho_max=rho_max,
                )
                supply = lwr_supply(
                    state[trajectory, incoming_end_cells],
                    v_max=speed,
                    rho_max=rho_max,
                )
                reverse_movement = allocate_equal_turning(demand, supply)
                fluxes[trajectory, movement_ids.ravel()] = (
                    -reverse_movement.T.ravel()
                )
    return fluxes


def generate_rollouts(
    initial_state: np.ndarray,
    *,
    network: StructuredNetwork,
    window: int,
    dt: float,
    v_max: np.ndarray,
    direction: np.ndarray,
    rho_max: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate conservative paired-direction finite-volume rollouts."""
    num_trajectories, num_cells = initial_state.shape
    rollout = np.empty(
        (num_trajectories, window + 1, num_cells),
        dtype=np.float64,
    )
    flux_history = np.empty(
        (
            num_trajectories,
            window,
            network.transfer_index.shape[1],
        ),
        dtype=np.float64,
    )
    rollout[:, 0] = initial_state
    for step in range(window):
        flux = compute_transfer_fluxes(
            rollout[:, step],
            network=network,
            v_max=v_max,
            direction=direction,
            rho_max=rho_max,
        )
        amount = rollout[:, step] * network.cell_lengths[None, :]
        amount = amount + dt * (flux @ network.incidence)
        rollout[:, step + 1] = amount / network.cell_lengths[None, :]
        flux_history[:, step] = flux
    return rollout, flux_history


def road_mass(
    rollout: np.ndarray,
    network: StructuredNetwork,
) -> np.ndarray:
    """Return vehicle amount on each road."""
    result = np.empty(
        (
            rollout.shape[0],
            rollout.shape[1],
            network.road_lengths.size,
        ),
        dtype=np.float64,
    )
    for road in range(network.road_lengths.size):
        start = int(network.road_cell_offsets[road])
        end = int(network.road_cell_offsets[road + 1])
        result[:, :, road] = np.sum(
            rollout[:, :, start:end]
            * network.cell_lengths[None, None, start:end],
            axis=2,
        )
    return result


def topology_signature(network: StructuredNetwork) -> str:
    """Hash the structured geometry and canonical road directions."""
    digest = hashlib.sha256()
    digest.update(
        np.asarray(
            (network.grid_rows, network.grid_cols),
            dtype=np.int64,
        ).tobytes()
    )
    digest.update(
        np.round(network.junction_positions, decimals=10).tobytes()
    )
    digest.update(network.road_start_junction.tobytes())
    digest.update(network.road_end_junction.tobytes())
    digest.update(network.road_kinds.tobytes())
    return digest.hexdigest()


def save_topology(
    path: Path,
    *,
    topology_id: int,
    topology_seed: int,
    split: str,
    parameter_regime: str,
    network: StructuredNetwork,
    rollout: np.ndarray,
    flux_history: np.ndarray,
    v_max: np.ndarray,
    direction: np.ndarray,
    initial_metadata: Dict[str, np.ndarray],
    dt: float,
    rho_max: float,
    cfl: float,
    signature: str,
) -> None:
    """Validate and write one topology archive."""
    total_mass = np.sum(
        rollout * network.cell_lengths[None, None, :],
        axis=2,
    )
    mass_error = total_mass - total_mass[:, :1]
    maximum_mass_error = float(np.max(np.abs(mass_error)))
    density_min = float(np.min(rollout))
    density_max = float(np.max(rollout))
    tolerance = 5.0e-11
    if density_min < -tolerance or density_max > rho_max + tolerance:
        raise RuntimeError(
            f"Topology {topology_id} density range "
            f"[{density_min}, {density_max}] is outside [0, {rho_max}]."
        )
    if maximum_mass_error > tolerance:
        raise RuntimeError(
            f"Topology {topology_id} mass error {maximum_mass_error} "
            f"exceeds {tolerance}."
        )

    num_trajectories = rollout.shape[0]
    num_nodes = rollout.shape[2]
    node_conditioning = np.zeros(
        (num_trajectories, num_nodes, 6),
        dtype=np.float32,
    )
    node_conditioning[..., 0] = v_max[:, None]
    node_conditioning[..., 5] = direction[:, None]
    trajectory_edge_attr = np.repeat(
        network.edge_attr_base[None, :, :],
        num_trajectories,
        axis=0,
    )
    trajectory_edge_attr[..., 0] *= direction[:, None]
    road_names = np.asarray(
        [
            f"{kind}_road_{road:03d}"
            for road, kind in enumerate(network.road_kinds)
        ],
        dtype="U64",
    )
    junction_names = np.asarray(
        [
            f"intersection_{junction:03d}"
            for junction in range(network.junction_positions.shape[0])
        ],
        dtype="U32",
    )
    empty_parameters = np.empty(
        (num_trajectories, 0),
        dtype=np.float64,
    )
    np.savez_compressed(
        path,
        rollout_states=rollout[..., None].astype(np.float32),
        rollout_transfer_fluxes=flux_history[..., None].astype(np.float32),
        trajectory_edge_attr=trajectory_edge_attr.astype(np.float32),
        trajectory_node_conditioning=node_conditioning,
        trajectory_v_max=v_max.astype(np.float64),
        trajectory_flow_direction=direction.astype(np.float64),
        trajectory_diverge_split=empty_parameters,
        trajectory_merge_priority=empty_parameters,
        total_mass=total_mass.astype(np.float64),
        road_mass=road_mass(rollout, network),
        pos=network.cell_positions.astype(np.float32),
        edge_index=network.edge_index.astype(np.int64),
        edge_attr=network.edge_attr_base.astype(np.float32),
        edge_transfer_id=network.edge_transfer_id.astype(np.int64),
        edge_is_physical_direction=(
            network.edge_is_physical_direction.astype(np.bool_)
        ),
        transfer_index=network.transfer_index.astype(np.int64),
        transfer_type=network.transfer_type.astype(np.int64),
        transfer_junction_id=network.transfer_junction_id.astype(np.int64),
        transfer_from_road=network.transfer_from_road.astype(np.int64),
        transfer_to_road=network.transfer_to_road.astype(np.int64),
        transfer_center_distance=(
            network.transfer_center_distance.astype(np.float64)
        ),
        cell_lengths=network.cell_lengths.astype(np.float64),
        cell_road_id=network.cell_road_id.astype(np.int64),
        cell_local_index=network.cell_local_index.astype(np.int64),
        cell_road_coordinate=(
            network.cell_road_coordinate.astype(np.float64)
        ),
        road_names=road_names,
        road_kinds=network.road_kinds,
        road_lengths=network.road_lengths.astype(np.float64),
        road_num_cells=network.road_num_cells.astype(np.int64),
        road_cell_offsets=network.road_cell_offsets.astype(np.int64),
        road_start_junction=(
            network.road_start_junction.astype(np.int64)
        ),
        road_end_junction=network.road_end_junction.astype(np.int64),
        junction_names=junction_names,
        junction_types=network.junction_types,
        junction_positions=network.junction_positions.astype(np.float64),
        junction_diverge_slot=np.full(
            network.junction_positions.shape[0],
            -1,
            dtype=np.int64,
        ),
        junction_merge_slot=np.full(
            network.junction_positions.shape[0],
            -1,
            dtype=np.int64,
        ),
        topology_id=np.array(topology_id, dtype=np.int64),
        topology_seed=np.array(topology_seed, dtype=np.int64),
        topology_signature=np.array(signature),
        split=np.array(split),
        parameter_regime=np.array(parameter_regime),
        grid_rows=np.array(network.grid_rows, dtype=np.int64),
        grid_cols=np.array(network.grid_cols, dtype=np.int64),
        num_removed_roads=np.array(
            network.num_removed_roads,
            dtype=np.int64,
        ),
        num_subdivided_roads=np.array(
            network.num_subdivided_roads,
            dtype=np.int64,
        ),
        num_diagonal_roads=np.array(
            network.num_diagonal_roads,
            dtype=np.int64,
        ),
        num_center_intersections=np.array(
            network.num_center_intersections,
            dtype=np.int64,
        ),
        dt=np.array(dt, dtype=np.float64),
        rho_max=np.array(rho_max, dtype=np.float64),
        rho_crit=np.array(0.5 * rho_max, dtype=np.float64),
        CFL=np.array(cfl, dtype=np.float64),
        num_trajectories=np.array(num_trajectories, dtype=np.int64),
        window=np.array(rollout.shape[1] - 1, dtype=np.int64),
        num_cells=np.array(num_nodes, dtype=np.int64),
        num_roads=np.array(network.road_lengths.size, dtype=np.int64),
        num_junctions=np.array(
            network.junction_positions.shape[0],
            dtype=np.int64,
        ),
        num_diverges=np.array(0, dtype=np.int64),
        num_merges=np.array(0, dtype=np.int64),
        num_transfers=np.array(
            network.transfer_index.shape[1],
            dtype=np.int64,
        ),
        max_abs_mass_error=np.array(
            maximum_mass_error,
            dtype=np.float64,
        ),
        density_min_observed=np.array(density_min, dtype=np.float64),
        density_max_observed=np.array(density_max, dtype=np.float64),
        target_type=np.array("increment"),
        state_channels=np.array(1, dtype=np.int64),
        state_representation=np.array("finite_volume_cell_average"),
        equation=np.array(
            "rho_t + (s*v_max*rho*(1-rho/rho_max))_x = 0"
        ),
        junction_solver=np.array(
            "equal_turning_demand_supply_multiway"
        ),
        edge_attr_columns=np.array(EDGE_ATTR_COLUMNS),
        node_conditioning_columns=np.array(NODE_CONDITIONING_COLUMNS),
        **initial_metadata,
    )


def edit_rates(
    args: argparse.Namespace,
    regime: str,
    rng: np.random.Generator,
) -> Tuple[float, float, float, float]:
    """Sample topology edit fractions for one split regime."""
    maxima = np.asarray(
        (
            args.remove_road_fraction_max,
            args.subdivide_road_fraction_max,
            args.diagonal_cell_fraction_max,
            args.center_intersection_fraction_max,
        ),
        dtype=np.float64,
    )
    if regime == "in_distribution":
        values = rng.uniform(0.0, maxima)
    else:
        lower = maxima
        upper = np.minimum(
            np.asarray((0.75, 0.50, 0.75, 0.50)),
            np.maximum(1.5 * maxima, maxima + 0.05),
        )
        values = rng.uniform(lower, upper)
    return tuple(float(value) for value in values)


def generate_one_topology(
    *,
    output_path: Path,
    topology_id: int,
    topology_seed: int,
    split: str,
    parameter_regime: str,
    rows: int,
    cols: int,
    num_trajectories: int,
    args: argparse.Namespace,
) -> Dict[str, object]:
    """Generate, simulate, validate, and save one structured topology."""
    rng = np.random.default_rng(topology_seed)
    (
        remove_fraction,
        subdivide_fraction,
        diagonal_fraction,
        center_fraction,
    ) = edit_rates(args, parameter_regime, rng)
    network = build_structured_network(
        rows=rows,
        cols=cols,
        spacing=args.grid_spacing,
        cells_per_unit=args.cells_per_unit,
        minimum_cells_per_road=args.minimum_cells_per_road,
        remove_fraction=remove_fraction,
        subdivide_fraction=subdivide_fraction,
        diagonal_fraction=diagonal_fraction,
        center_fraction=center_fraction,
        rng=rng,
    )
    initial_state, direction, initial_metadata = (
        generate_paired_initial_conditions(
            network=network,
            num_trajectories=num_trajectories,
            num_blocks_min=args.num_blocks_min,
            num_blocks_max=args.num_blocks_max,
            road_baseline_jitter=args.road_baseline_jitter,
            rho_max=args.rho_max,
            rng=rng,
        )
    )
    num_pairs = num_trajectories // 2
    if parameter_regime == "in_distribution":
        pair_v_max = rng.uniform(0.8, 1.2, size=num_pairs)
    else:
        pair_v_max = sample_union(
            (0.55, 0.72),
            (1.28, 1.50),
            num_pairs,
            rng,
        )
    v_max = np.repeat(pair_v_max, 2)
    dt = float(
        args.CFL * np.min(network.cell_lengths) / np.max(v_max)
    )
    rollout, flux_history = generate_rollouts(
        initial_state,
        network=network,
        window=args.window,
        dt=dt,
        v_max=v_max,
        direction=direction,
        rho_max=args.rho_max,
    )
    signature = topology_signature(network)
    save_topology(
        output_path,
        topology_id=topology_id,
        topology_seed=topology_seed,
        split=split,
        parameter_regime=parameter_regime,
        network=network,
        rollout=rollout,
        flux_history=flux_history,
        v_max=v_max,
        direction=direction,
        initial_metadata=initial_metadata,
        dt=dt,
        rho_max=args.rho_max,
        cfl=args.CFL,
        signature=signature,
    )
    total_mass = np.sum(
        rollout * network.cell_lengths[None, None, :],
        axis=2,
    )
    return {
        "topology_id": topology_id,
        "file": output_path.name,
        "split": split,
        "parameter_regime": parameter_regime,
        "topology_seed": topology_seed,
        "topology_signature": signature,
        "grid_rows": rows,
        "grid_cols": cols,
        "num_trajectories": num_trajectories,
        "num_direction_pairs": num_pairs,
        "num_roads": int(network.road_lengths.size),
        "num_junctions": int(network.junction_positions.shape[0]),
        "num_cells": int(network.cell_lengths.size),
        "num_edges": int(network.edge_index.shape[1]),
        "num_removed_roads": network.num_removed_roads,
        "num_subdivided_roads": network.num_subdivided_roads,
        "num_diagonal_roads": network.num_diagonal_roads,
        "num_center_intersections": network.num_center_intersections,
        "dt": dt,
        "physical_horizon": float(args.window * dt),
        "v_max_min_observed": float(np.min(v_max)),
        "v_max_max_observed": float(np.max(v_max)),
        "max_abs_mass_error": float(
            np.max(np.abs(total_mass - total_mass[:, :1]))
        ),
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse generator command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate closed edited-checkerboard LWR networks with paired "
            "forward and reverse trajectories."
        )
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--train-topologies", type=int, default=16)
    parser.add_argument("--validation-topologies", type=int, default=4)
    parser.add_argument("--test-in-size-topologies", type=int, default=4)
    parser.add_argument("--test-large-topologies", type=int, default=4)
    parser.add_argument("--test-parameter-ood-topologies", type=int, default=4)
    parser.add_argument(
        "--train-trajectories-per-topology",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--eval-trajectories-per-topology",
        type=int,
        default=4,
    )
    parser.add_argument("--train-grid-rows-min", type=int, default=3)
    parser.add_argument("--train-grid-rows-max", type=int, default=4)
    parser.add_argument("--train-grid-cols-min", type=int, default=3)
    parser.add_argument("--train-grid-cols-max", type=int, default=4)
    parser.add_argument("--large-grid-rows-min", type=int, default=5)
    parser.add_argument("--large-grid-rows-max", type=int, default=6)
    parser.add_argument("--large-grid-cols-min", type=int, default=5)
    parser.add_argument("--large-grid-cols-max", type=int, default=6)
    parser.add_argument("--grid-spacing", type=float, default=1.0)
    parser.add_argument("--cells-per-unit", type=float, default=8.0)
    parser.add_argument("--minimum-cells-per-road", type=int, default=3)
    parser.add_argument(
        "--remove-road-fraction-max",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--subdivide-road-fraction-max",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--diagonal-cell-fraction-max",
        type=float,
        default=0.30,
    )
    parser.add_argument(
        "--center-intersection-fraction-max",
        type=float,
        default=0.15,
    )
    parser.add_argument("--num-blocks-min", type=int, default=3)
    parser.add_argument("--num-blocks-max", type=int, default=10)
    parser.add_argument("--road-baseline-jitter", type=float, default=0.12)
    parser.add_argument("--rho-max", type=float, default=1.0)
    parser.add_argument("--CFL", type=float, default=0.40)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    """Validate generator parameters."""
    positive_counts = (
        "window",
        "train_topologies",
        "validation_topologies",
        "test_in_size_topologies",
        "test_large_topologies",
        "test_parameter_ood_topologies",
        "train_trajectories_per_topology",
        "eval_trajectories_per_topology",
        "minimum_cells_per_road",
    )
    for name in positive_counts:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    for name in (
        "train_trajectories_per_topology",
        "eval_trajectories_per_topology",
    ):
        if getattr(args, name) % 2 != 0:
            raise ValueError(
                f"--{name.replace('_', '-')} must be even so every "
                "initial condition has both directions."
            )
    ranges = (
        ("train grid rows", args.train_grid_rows_min, args.train_grid_rows_max),
        ("train grid cols", args.train_grid_cols_min, args.train_grid_cols_max),
        ("large grid rows", args.large_grid_rows_min, args.large_grid_rows_max),
        ("large grid cols", args.large_grid_cols_min, args.large_grid_cols_max),
    )
    for label, minimum, maximum in ranges:
        if minimum < 2 or maximum < minimum:
            raise ValueError(f"Invalid {label} range.")
    if (
        args.large_grid_rows_min <= args.train_grid_rows_max
        and args.large_grid_cols_min <= args.train_grid_cols_max
    ):
        raise ValueError(
            "The large-grid split must exceed the training grid in at "
            "least one dimension."
        )
    if args.grid_spacing <= 0.0 or args.cells_per_unit <= 0.0:
        raise ValueError("Grid spacing and cells per unit must be positive.")
    fraction_names = (
        "remove_road_fraction_max",
        "subdivide_road_fraction_max",
        "diagonal_cell_fraction_max",
        "center_intersection_fraction_max",
    )
    for name in fraction_names:
        value = getattr(args, name)
        if not (0.0 <= value <= 1.0):
            raise ValueError(
                f"--{name.replace('_', '-')} must be in [0, 1]."
            )
    if not (1 <= args.num_blocks_min <= args.num_blocks_max):
        raise ValueError("Invalid congestion-block count range.")
    if not (0.0 <= args.road_baseline_jitter <= 0.4):
        raise ValueError("--road-baseline-jitter must be in [0, 0.4].")
    if args.rho_max <= 0.0:
        raise ValueError("--rho-max must be positive.")
    if not (0.0 < args.CFL <= 0.5):
        raise ValueError("--CFL must be in (0, 0.5].")


def json_config(args: argparse.Namespace) -> Dict[str, object]:
    """Convert generator arguments to JSON-compatible values."""
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Generate all topology-disjoint splits and their manifest."""
    args = parse_args(argv)
    validate_args(args)
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {args.out_dir}. "
            "Choose a new directory to avoid mixing datasets."
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    split_specs = (
        (
            "train",
            args.train_topologies,
            args.train_grid_rows_min,
            args.train_grid_rows_max,
            args.train_grid_cols_min,
            args.train_grid_cols_max,
            args.train_trajectories_per_topology,
            "in_distribution",
        ),
        (
            "validation",
            args.validation_topologies,
            args.train_grid_rows_min,
            args.train_grid_rows_max,
            args.train_grid_cols_min,
            args.train_grid_cols_max,
            args.eval_trajectories_per_topology,
            "in_distribution",
        ),
        (
            "test_in_size",
            args.test_in_size_topologies,
            args.train_grid_rows_min,
            args.train_grid_rows_max,
            args.train_grid_cols_min,
            args.train_grid_cols_max,
            args.eval_trajectories_per_topology,
            "in_distribution",
        ),
        (
            "test_large",
            args.test_large_topologies,
            args.large_grid_rows_min,
            args.large_grid_rows_max,
            args.large_grid_cols_min,
            args.large_grid_cols_max,
            args.eval_trajectories_per_topology,
            "in_distribution",
        ),
        (
            "test_parameter_ood",
            args.test_parameter_ood_topologies,
            args.train_grid_rows_min,
            args.train_grid_rows_max,
            args.train_grid_cols_min,
            args.train_grid_cols_max,
            args.eval_trajectories_per_topology,
            "parameter_and_structure_ood",
        ),
    )
    master_rng = np.random.default_rng(args.seed)
    records: List[Dict[str, object]] = []
    signatures = set()
    topology_id = 0
    for (
        split,
        count,
        row_min,
        row_max,
        col_min,
        col_max,
        num_trajectories,
        regime,
    ) in split_specs:
        for _ in range(count):
            output_path = (
                args.out_dir / f"topology_{topology_id:04d}.npz"
            )
            for attempt in range(200):
                topology_seed = int(
                    master_rng.integers(0, np.iinfo(np.int64).max)
                )
                rows = int(master_rng.integers(row_min, row_max + 1))
                cols = int(master_rng.integers(col_min, col_max + 1))
                record = generate_one_topology(
                    output_path=output_path,
                    topology_id=topology_id,
                    topology_seed=topology_seed,
                    split=split,
                    parameter_regime=regime,
                    rows=rows,
                    cols=cols,
                    num_trajectories=num_trajectories,
                    args=args,
                )
                signature = str(record["topology_signature"])
                if signature not in signatures:
                    signatures.add(signature)
                    break
            else:
                raise RuntimeError(
                    "Could not generate a unique topology after "
                    f"{attempt + 1} attempts for split {split}."
                )
            records.append(record)
            print(
                f"[{split}] topology {topology_id:04d}: "
                f"grid={rows}x{cols}, "
                f"roads={record['num_roads']}, "
                f"junctions={record['num_junctions']}, "
                f"cells={record['num_cells']}, "
                f"direction_pairs={record['num_direction_pairs']}, "
                f"mass_error={record['max_abs_mass_error']:.3e}"
            )
            topology_id += 1

    manifest = {
        "format": "variable_topology_lwr_v1",
        "dataset_family": "structured_direction_reversal_lwr_v1",
        "description": (
            "Closed edited-checkerboard LWR networks with paired canonical "
            "and globally reversed traffic-flow trajectories."
        ),
        "generation_config": json_config(args),
        "edge_attr_columns": EDGE_ATTR_COLUMNS,
        "node_conditioning_columns": NODE_CONDITIONING_COLUMNS,
        "trajectory_pairing": {
            "pairing": "adjacent_trajectories",
            "even_trajectory_direction": 1,
            "odd_trajectory_direction": -1,
            "shared_within_pair": [
                "initial_density",
                "v_max",
                "topology",
            ],
        },
        "parameter_ranges": {
            "in_distribution": {
                "v_max": [0.8, 1.2],
                "flow_direction": [-1, 1],
            },
            "parameter_and_structure_ood": {
                "v_max": [[0.55, 0.72], [1.28, 1.50]],
                "flow_direction": [-1, 1],
                "topology_edit_rates": "above training maxima",
            },
        },
        "splits": {
            split: [
                int(record["topology_id"])
                for record in records
                if record["split"] == split
            ]
            for split in SPLIT_NAMES
        },
        "topologies": records,
    }
    manifest_path = args.out_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(f"wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
