#!/usr/bin/env python3
"""Generate finite-volume LWR traffic data on a directed road network.

The network is a small, closed merge-diverge system embedded in two dimensions:

                         upper road
                      /--------------\
    return road ---> diverge          merge ---> return road
                      \\--------------/
                         lower road

The outgoing return road reconnects to the incoming return road through a
one-to-one junction.  The resulting network has no external sources or sinks.
It contains one diverge, one merge, and one one-to-one junction, while retaining
an exactly conserved network-wide vehicle count.

Every road is a one-dimensional LWR finite-volume domain,

    rho_t + f(rho)_x = 0,
    f(rho) = v_max * rho * (1 - rho / rho_max).

Internal road interfaces use the Godunov flux.  Junctions use the corresponding
LWR demand-supply formulation:

* one-to-one: flow is min(upstream demand, downstream supply);
* diverge: a fixed turning fraction sends flow to each outgoing road;
* merge: fixed priorities allocate limited downstream supply.

The finite-volume cells are graph nodes.  ``transfer_index`` stores each
physical vehicle-transfer direction once, while ``edge_index`` stores both
directions for message passing.  The conservative update is applied to cell
vehicle amounts, rho_i * cell_length_i, so irregular cell lengths are handled
correctly.

The output follows the existing one-step-pair convention (``x``, ``y``,
``u_next``, and ``rollout_states``) and adds:

* road, junction, cell, and physical-transfer topology arrays;
* ``rollout_transfer_fluxes`` at every saved transition;
* per-road and network-wide vehicle-count histories;
* controlled congestion-block initial-condition metadata.
* optional trajectory-specific ``v_max``, diverge split, and merge priority;
* optional road-specific backgrounds and multiple blocks on the same road.

Example:

    python scripts/generate_network_lwr_traffic_data.py \
        --num-samples 100 \
        --window 128 \
        --cells-per-road 24 \
        --CFL 0.45 \
        --out data/lwr_network.npz

A more heterogeneous conditioned dataset can be generated with:

    python scripts/generate_network_lwr_traffic_data.py \
        --num-samples 300 \
        --window 128 \
        --v-max-min 0.7 --v-max-max 1.3 \
        --branch-split-min 0.25 --branch-split-max 0.75 \
        --upper-merge-priority-min 0.25 \
        --upper-merge-priority-max 0.75 \
        --num-blocks-max 8 \
        --allow-multiple-blocks-per-road \
        --road-baseline-jitter 0.15 \
        --out data/lwr_network_complex.npz
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


TARGET_TYPES = ("increment", "rate")


@dataclass
class Junction:
    """Runtime description of one conservative road junction."""

    junction_id: int
    name: str
    kind: str
    position: np.ndarray
    incoming_roads: Tuple[int, ...]
    outgoing_roads: Tuple[int, ...]
    turning: np.ndarray
    merge_priorities: np.ndarray


@dataclass
class RoadNetwork:
    """Fixed finite-volume and graph representation of the road network."""

    junctions: List[Junction]
    junction_names: np.ndarray
    junction_types: np.ndarray
    junction_positions: np.ndarray
    road_names: np.ndarray
    road_lengths: np.ndarray
    road_num_cells: np.ndarray
    road_cell_offsets: np.ndarray
    road_start_junction: np.ndarray
    road_end_junction: np.ndarray
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
    transfer_turning_fraction: np.ndarray
    transfer_merge_priority: np.ndarray
    transfer_center_distance: np.ndarray
    internal_transfer_ids: List[np.ndarray]
    movement_transfer_ids: Dict[Tuple[int, int], int]
    incidence: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray
    edge_transfer_id: np.ndarray
    edge_is_physical_direction: np.ndarray
    turning_matrix: np.ndarray
    merge_priority_matrix: np.ndarray


def lwr_flux(rho: np.ndarray, *, v_max: float, rho_max: float) -> np.ndarray:
    """Evaluate the quadratic LWR flux."""
    return v_max * rho * (1.0 - rho / rho_max)


def lwr_demand(rho: np.ndarray, *, v_max: float, rho_max: float) -> np.ndarray:
    """Maximum flow that an upstream LWR state can send."""
    rho_arr = np.asarray(rho, dtype=np.float64)
    rho_critical = 0.5 * rho_max
    flux_capacity = 0.25 * v_max * rho_max
    return np.where(
        rho_arr <= rho_critical,
        lwr_flux(rho_arr, v_max=v_max, rho_max=rho_max),
        flux_capacity,
    )


def lwr_supply(rho: np.ndarray, *, v_max: float, rho_max: float) -> np.ndarray:
    """Maximum flow that a downstream LWR state can receive."""
    rho_arr = np.asarray(rho, dtype=np.float64)
    rho_critical = 0.5 * rho_max
    flux_capacity = 0.25 * v_max * rho_max
    return np.where(
        rho_arr <= rho_critical,
        flux_capacity,
        lwr_flux(rho_arr, v_max=v_max, rho_max=rho_max),
    )


def lwr_godunov_flux(
    rho_left: np.ndarray,
    rho_right: np.ndarray,
    *,
    v_max: float,
    rho_max: float,
) -> np.ndarray:
    """Godunov flux for LWR, expressed as upstream demand versus supply."""
    return np.minimum(
        lwr_demand(rho_left, v_max=v_max, rho_max=rho_max),
        lwr_supply(rho_right, v_max=v_max, rho_max=rho_max),
    )


def _quadratic_bezier(
    start: np.ndarray,
    control: np.ndarray,
    end: np.ndarray,
    parameter: np.ndarray,
) -> np.ndarray:
    """Evaluate a quadratic Bezier curve for display coordinates."""
    t = np.asarray(parameter, dtype=np.float64)[:, None]
    return (1.0 - t) ** 2 * start + 2.0 * (1.0 - t) * t * control + t**2 * end


def build_merge_diverge_network(
    *,
    cells_per_road: int,
    branch_split: float,
    upper_merge_priority: float,
) -> RoadNetwork:
    """Create the fixed closed network and all graph/FV indexing arrays."""
    if cells_per_road < 3:
        raise ValueError("cells_per_road must be at least 3.")
    if not (0.0 < branch_split < 1.0):
        raise ValueError("branch_split must be strictly between 0 and 1.")
    if not (0.0 < upper_merge_priority < 1.0):
        raise ValueError(
            "upper_merge_priority must be strictly between 0 and 1."
        )

    # Junctions: 0 is the diverge, 1 is the merge, and 2 closes the network.
    junction_positions = np.array(
        [
            [-1.0, 0.0],
            [1.0, 0.0],
            [0.0, -1.25],
        ],
        dtype=np.float64,
    )
    junction_names = np.array(["diverge", "merge", "return"], dtype="U32")
    junction_types = np.array(["diverge", "merge", "one_to_one"], dtype="U32")

    # Roads: 0 enters the diverge; 1 and 2 are its branches; 3 leaves the merge.
    road_names = np.array(
        ["incoming_return", "upper_branch", "lower_branch", "outgoing_return"],
        dtype="U32",
    )
    road_start_junction = np.array([2, 0, 0, 1], dtype=np.int64)
    road_end_junction = np.array([0, 1, 1, 2], dtype=np.int64)
    road_lengths = np.array([0.90, 1.20, 1.45, 1.00], dtype=np.float64)
    road_num_cells = np.full(road_lengths.size, cells_per_road, dtype=np.int64)
    road_cell_offsets = np.concatenate(
        [
            np.array([0], dtype=np.int64),
            np.cumsum(road_num_cells, dtype=np.int64),
        ]
    )
    num_roads = int(road_lengths.size)
    num_cells = int(road_cell_offsets[-1])

    road_control_points = np.array(
        [
            [-0.78, -0.78],
            [0.00, 0.72],
            [0.00, -0.46],
            [0.78, -0.82],
        ],
        dtype=np.float64,
    )

    cell_lengths = np.empty(num_cells, dtype=np.float64)
    cell_road_id = np.empty(num_cells, dtype=np.int64)
    cell_local_index = np.empty(num_cells, dtype=np.int64)
    cell_road_coordinate = np.empty(num_cells, dtype=np.float64)
    cell_positions = np.empty((num_cells, 2), dtype=np.float64)

    for road in range(num_roads):
        start = int(road_cell_offsets[road])
        end = int(road_cell_offsets[road + 1])
        count = int(road_num_cells[road])
        length = float(road_lengths[road])
        dx = length / count
        local_centers = (np.arange(count, dtype=np.float64) + 0.5) * dx
        parameter = local_centers / length

        start_position = junction_positions[road_start_junction[road]]
        end_position = junction_positions[road_end_junction[road]]
        positions = _quadratic_bezier(
            start_position,
            road_control_points[road],
            end_position,
            parameter,
        )

        cell_lengths[start:end] = dx
        cell_road_id[start:end] = road
        cell_local_index[start:end] = np.arange(count, dtype=np.int64)
        cell_road_coordinate[start:end] = local_centers
        cell_positions[start:end] = positions

    junctions = [
        Junction(
            junction_id=0,
            name="diverge",
            kind="diverge",
            position=junction_positions[0],
            incoming_roads=(0,),
            outgoing_roads=(1, 2),
            turning=np.array([[branch_split, 1.0 - branch_split]]),
            merge_priorities=np.array([1.0]),
        ),
        Junction(
            junction_id=1,
            name="merge",
            kind="merge",
            position=junction_positions[1],
            incoming_roads=(1, 2),
            outgoing_roads=(3,),
            turning=np.array([[1.0], [1.0]]),
            merge_priorities=np.array(
                [upper_merge_priority, 1.0 - upper_merge_priority]
            ),
        ),
        Junction(
            junction_id=2,
            name="return",
            kind="one_to_one",
            position=junction_positions[2],
            incoming_roads=(3,),
            outgoing_roads=(0,),
            turning=np.array([[1.0]]),
            merge_priorities=np.array([1.0]),
        ),
    ]

    sources: List[int] = []
    targets: List[int] = []
    transfer_type: List[int] = []
    transfer_junction_id: List[int] = []
    transfer_from_road: List[int] = []
    transfer_to_road: List[int] = []
    transfer_turning_fraction: List[float] = []
    transfer_merge_priority: List[float] = []
    internal_transfer_ids: List[np.ndarray] = []
    movement_transfer_ids: Dict[Tuple[int, int], int] = {}

    for road in range(num_roads):
        start = int(road_cell_offsets[road])
        end = int(road_cell_offsets[road + 1])
        road_transfer_ids: List[int] = []
        for source in range(start, end - 1):
            transfer_id = len(sources)
            sources.append(source)
            targets.append(source + 1)
            transfer_type.append(0)
            transfer_junction_id.append(-1)
            transfer_from_road.append(road)
            transfer_to_road.append(road)
            transfer_turning_fraction.append(1.0)
            transfer_merge_priority.append(1.0)
            road_transfer_ids.append(transfer_id)
        internal_transfer_ids.append(
            np.asarray(road_transfer_ids, dtype=np.int64)
        )

    turning_matrix = np.zeros((num_roads, num_roads), dtype=np.float64)
    merge_priority_matrix = np.full(
        (len(junctions), num_roads), np.nan, dtype=np.float64
    )
    for junction in junctions:
        for incoming_index, incoming_road in enumerate(junction.incoming_roads):
            merge_priority_matrix[
                junction.junction_id, incoming_road
            ] = junction.merge_priorities[incoming_index]
            for outgoing_index, outgoing_road in enumerate(
                junction.outgoing_roads
            ):
                turning_fraction = float(
                    junction.turning[incoming_index, outgoing_index]
                )
                if turning_fraction <= 0.0:
                    continue
                transfer_id = len(sources)
                sources.append(int(road_cell_offsets[incoming_road + 1] - 1))
                targets.append(int(road_cell_offsets[outgoing_road]))
                transfer_type.append(1)
                transfer_junction_id.append(junction.junction_id)
                transfer_from_road.append(incoming_road)
                transfer_to_road.append(outgoing_road)
                transfer_turning_fraction.append(turning_fraction)
                transfer_merge_priority.append(
                    float(junction.merge_priorities[incoming_index])
                )
                movement_transfer_ids[(incoming_road, outgoing_road)] = transfer_id
                turning_matrix[incoming_road, outgoing_road] = turning_fraction

    transfer_index = np.stack(
        [
            np.asarray(sources, dtype=np.int64),
            np.asarray(targets, dtype=np.int64),
        ],
        axis=0,
    )
    transfer_type_array = np.asarray(transfer_type, dtype=np.int64)
    transfer_junction_array = np.asarray(transfer_junction_id, dtype=np.int64)
    transfer_from_road_array = np.asarray(transfer_from_road, dtype=np.int64)
    transfer_to_road_array = np.asarray(transfer_to_road, dtype=np.int64)
    transfer_turning_array = np.asarray(
        transfer_turning_fraction, dtype=np.float64
    )
    transfer_priority_array = np.asarray(
        transfer_merge_priority, dtype=np.float64
    )
    transfer_center_distance = 0.5 * (
        cell_lengths[transfer_index[0]] + cell_lengths[transfer_index[1]]
    )

    num_transfers = transfer_index.shape[1]
    incidence = np.zeros((num_transfers, num_cells), dtype=np.float64)
    incidence[np.arange(num_transfers), transfer_index[0]] = -1.0
    incidence[np.arange(num_transfers), transfer_index[1]] = 1.0

    # Message-passing edges include both the physical traffic direction and its
    # reverse, allowing a GNN to inspect upstream and downstream cell states.
    edge_sources = np.repeat(transfer_index[0], 2)
    edge_targets = np.repeat(transfer_index[1], 2)
    edge_sources[1::2] = transfer_index[1]
    edge_targets[1::2] = transfer_index[0]
    edge_index = np.stack([edge_sources, edge_targets], axis=0)

    traffic_direction = np.tile(
        np.array([1.0, -1.0], dtype=np.float64), num_transfers
    )
    displacement = cell_positions[edge_targets] - cell_positions[edge_sources]
    euclidean_distance = np.linalg.norm(displacement, axis=1)
    physical_center_distance = np.repeat(transfer_center_distance, 2)
    source_length = cell_lengths[edge_sources]
    target_length = cell_lengths[edge_targets]
    is_junction = np.repeat(transfer_type_array.astype(np.float64), 2)
    edge_turning = np.repeat(transfer_turning_array, 2)
    edge_priority = np.repeat(transfer_priority_array, 2)
    edge_attr = np.column_stack(
        [
            traffic_direction,
            displacement[:, 0],
            displacement[:, 1],
            euclidean_distance,
            physical_center_distance,
            source_length,
            target_length,
            is_junction,
            edge_turning,
            edge_priority,
        ]
    )
    edge_transfer_id = np.repeat(
        np.arange(num_transfers, dtype=np.int64), 2
    )
    edge_is_physical_direction = np.tile(
        np.array([True, False], dtype=np.bool_), num_transfers
    )

    return RoadNetwork(
        junctions=junctions,
        junction_names=junction_names,
        junction_types=junction_types,
        junction_positions=junction_positions,
        road_names=road_names,
        road_lengths=road_lengths,
        road_num_cells=road_num_cells,
        road_cell_offsets=road_cell_offsets,
        road_start_junction=road_start_junction,
        road_end_junction=road_end_junction,
        cell_lengths=cell_lengths,
        cell_road_id=cell_road_id,
        cell_local_index=cell_local_index,
        cell_road_coordinate=cell_road_coordinate,
        cell_positions=cell_positions,
        transfer_index=transfer_index,
        transfer_type=transfer_type_array,
        transfer_junction_id=transfer_junction_array,
        transfer_from_road=transfer_from_road_array,
        transfer_to_road=transfer_to_road_array,
        transfer_turning_fraction=transfer_turning_array,
        transfer_merge_priority=transfer_priority_array,
        transfer_center_distance=transfer_center_distance,
        internal_transfer_ids=internal_transfer_ids,
        movement_transfer_ids=movement_transfer_ids,
        incidence=incidence,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_transfer_id=edge_transfer_id,
        edge_is_physical_direction=edge_is_physical_direction,
        turning_matrix=turning_matrix,
        merge_priority_matrix=merge_priority_matrix,
    )


def allocate_merge_flows(
    demands: np.ndarray,
    supply: np.ndarray,
    priorities: np.ndarray,
) -> np.ndarray:
    """Allocate one outgoing road's supply among competing incoming roads.

    Priority shares are used when every incoming road is demand-constrained by
    the available supply.  If one road cannot use its priority share, the unused
    capacity is redistributed among the remaining roads.
    """
    demand_array = np.asarray(demands, dtype=np.float64)
    supply_array = np.asarray(supply, dtype=np.float64)
    priority_array = np.asarray(priorities, dtype=np.float64)
    if demand_array.ndim != 2:
        raise ValueError("demands must have shape [batch, incoming_road].")
    if supply_array.shape != (demand_array.shape[0],):
        raise ValueError("supply must have shape [batch].")
    if priority_array.shape != (demand_array.shape[1],):
        raise ValueError("priorities must match the incoming-road dimension.")
    if np.any(priority_array <= 0.0):
        raise ValueError("All merge priorities must be positive.")

    flows = np.zeros_like(demand_array)
    for sample in range(demand_array.shape[0]):
        demand = demand_array[sample]
        available = max(float(supply_array[sample]), 0.0)
        if float(np.sum(demand)) <= available:
            flows[sample] = demand
            continue

        active = np.ones(demand.size, dtype=np.bool_)
        while np.any(active) and available > 0.0:
            active_indices = np.flatnonzero(active)
            active_priorities = priority_array[active_indices]
            shares = (
                available
                * active_priorities
                / float(np.sum(active_priorities))
            )
            limited = demand[active_indices] <= shares + 1.0e-15
            if not np.any(limited):
                flows[sample, active_indices] = shares
                available = 0.0
                break

            limited_indices = active_indices[limited]
            flows[sample, limited_indices] = demand[limited_indices]
            available -= float(np.sum(demand[limited_indices]))
            active[limited_indices] = False

    return flows


def compute_transfer_fluxes(
    rho: np.ndarray,
    *,
    network: RoadNetwork,
    v_max: float,
    rho_max: float,
) -> np.ndarray:
    """Compute every internal-road and junction movement flux."""
    state = np.asarray(rho, dtype=np.float64)
    if state.ndim != 2 or state.shape[1] != network.cell_lengths.size:
        raise ValueError(
            "rho must have shape [batch, num_cells], got "
            f"{state.shape}."
        )
    fluxes = np.zeros(
        (state.shape[0], network.transfer_index.shape[1]), dtype=np.float64
    )

    for road, transfer_ids in enumerate(network.internal_transfer_ids):
        start = int(network.road_cell_offsets[road])
        end = int(network.road_cell_offsets[road + 1])
        fluxes[:, transfer_ids] = lwr_godunov_flux(
            state[:, start : end - 1],
            state[:, start + 1 : end],
            v_max=v_max,
            rho_max=rho_max,
        )

    for junction in network.junctions:
        incoming_cells = np.array(
            [
                int(network.road_cell_offsets[road + 1] - 1)
                for road in junction.incoming_roads
            ],
            dtype=np.int64,
        )
        outgoing_cells = np.array(
            [
                int(network.road_cell_offsets[road])
                for road in junction.outgoing_roads
            ],
            dtype=np.int64,
        )
        demands = lwr_demand(
            state[:, incoming_cells], v_max=v_max, rho_max=rho_max
        )
        supplies = lwr_supply(
            state[:, outgoing_cells], v_max=v_max, rho_max=rho_max
        )

        if junction.kind == "one_to_one":
            movement_flows = np.minimum(demands[:, :1], supplies[:, :1])
        elif junction.kind == "diverge":
            turning = junction.turning[0]
            receiving_limits = supplies / turning[None, :]
            total_outflow = np.minimum(
                demands[:, 0], np.min(receiving_limits, axis=1)
            )
            movement_flows = total_outflow[:, None] * turning[None, :]
        elif junction.kind == "merge":
            incoming_flows = allocate_merge_flows(
                demands,
                supplies[:, 0],
                junction.merge_priorities,
            )
            movement_flows = incoming_flows[:, :, None]
        else:
            raise ValueError(f"Unsupported junction type: {junction.kind}")

        for incoming_index, incoming_road in enumerate(junction.incoming_roads):
            for outgoing_index, outgoing_road in enumerate(
                junction.outgoing_roads
            ):
                transfer_id = network.movement_transfer_ids[
                    (incoming_road, outgoing_road)
                ]
                if junction.kind == "merge":
                    value = movement_flows[:, incoming_index, outgoing_index]
                else:
                    value = movement_flows[:, outgoing_index]
                fluxes[:, transfer_id] = value

    return fluxes


def advance_network_one_step(
    rho: np.ndarray,
    *,
    network: RoadNetwork,
    dt: float,
    v_max: float,
    rho_max: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Advance one conservative network finite-volume time step."""
    state = np.asarray(rho, dtype=np.float64)
    transfer_fluxes = compute_transfer_fluxes(
        state,
        network=network,
        v_max=v_max,
        rho_max=rho_max,
    )
    vehicle_amount = state * network.cell_lengths[None, :]
    amount_change = dt * (transfer_fluxes @ network.incidence)
    next_state = (
        vehicle_amount + amount_change
    ) / network.cell_lengths[None, :]
    return next_state, transfer_fluxes


def generate_initial_conditions(
    *,
    num_samples: int,
    network: RoadNetwork,
    density_min: float,
    density_max: float,
    min_jump_fraction: float,
    num_blocks_min: int,
    num_blocks_max: int,
    block_width_fraction_min: float,
    block_width_fraction_max: float,
    road_baseline_jitter: float,
    allow_multiple_blocks_per_road: bool,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Generate exact cell averages of randomized congestion blocks."""
    num_roads = network.road_lengths.size
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    if not (0.0 <= density_min < density_max):
        raise ValueError("Require 0 <= density_min < density_max.")
    if not (0.0 <= min_jump_fraction < 1.0):
        raise ValueError("min_jump_fraction must be in [0, 1).")
    if not (1 <= num_blocks_min <= num_blocks_max):
        raise ValueError("Require 1 <= num_blocks_min <= num_blocks_max.")
    if not allow_multiple_blocks_per_road and num_blocks_max > num_roads:
        raise ValueError(
            "num_blocks_max cannot exceed the number of roads unless "
            "allow_multiple_blocks_per_road is enabled."
        )
    if not (0.0 <= road_baseline_jitter <= 1.0):
        raise ValueError("road_baseline_jitter must be in [0, 1].")
    if not (
        0.0
        < block_width_fraction_min
        <= block_width_fraction_max
        < 1.0
    ):
        raise ValueError(
            "Require 0 < block_width_fraction_min <= "
            "block_width_fraction_max < 1."
        )

    num_cells = network.cell_lengths.size
    states = np.empty((num_samples, num_cells), dtype=np.float64)
    baseline_density = np.empty(num_samples, dtype=np.float64)
    road_baseline_density = np.empty(
        (num_samples, num_roads), dtype=np.float64
    )
    block_count = np.empty(num_samples, dtype=np.int64)
    block_road_id = np.full(
        (num_samples, num_blocks_max), -1, dtype=np.int64
    )
    block_start = np.full(
        (num_samples, num_blocks_max), np.nan, dtype=np.float64
    )
    block_width = np.full(
        (num_samples, num_blocks_max), np.nan, dtype=np.float64
    )
    block_high_density = np.full(
        (num_samples, num_blocks_max), np.nan, dtype=np.float64
    )

    density_span = density_max - density_min
    minimum_jump = min_jump_fraction * density_span
    midpoint = 0.5 * (density_min + density_max)

    for sample in range(num_samples):
        low_density = float(
            rng.uniform(density_min, midpoint - 0.5 * minimum_jump)
        )
        baseline_density[sample] = low_density
        maximum_background = midpoint - 0.5 * minimum_jump
        if road_baseline_jitter == 0.0:
            road_backgrounds = np.full(num_roads, low_density)
        else:
            road_backgrounds = np.clip(
                low_density
                + rng.uniform(-1.0, 1.0, size=num_roads)
                * road_baseline_jitter
                * density_span,
                density_min,
                maximum_background,
            )
        road_baseline_density[sample] = road_backgrounds
        for road in range(num_roads):
            global_start = int(network.road_cell_offsets[road])
            global_end = int(network.road_cell_offsets[road + 1])
            states[sample, global_start:global_end] = road_backgrounds[road]

        count = int(rng.integers(num_blocks_min, num_blocks_max + 1))
        selected_roads = rng.choice(
            num_roads,
            size=count,
            replace=allow_multiple_blocks_per_road,
        )
        block_count[sample] = count

        for block_index, road in enumerate(selected_roads):
            road = int(road)
            road_length = float(network.road_lengths[road])
            width = float(
                road_length
                * rng.uniform(
                    block_width_fraction_min,
                    block_width_fraction_max,
                )
            )
            start_position = float(rng.uniform(0.0, road_length - width))
            end_position = start_position + width
            high_density = float(
                rng.uniform(midpoint + 0.5 * minimum_jump, density_max)
            )

            global_start = int(network.road_cell_offsets[road])
            global_end = int(network.road_cell_offsets[road + 1])
            count_on_road = int(network.road_num_cells[road])
            local_edges = np.linspace(
                0.0, road_length, count_on_road + 1, dtype=np.float64
            )
            overlap = np.maximum(
                0.0,
                np.minimum(local_edges[1:], end_position)
                - np.maximum(local_edges[:-1], start_position),
            )
            fraction = overlap / np.diff(local_edges)
            road_background = float(road_backgrounds[road])
            block_profile = (
                road_background
                + (high_density - road_background) * fraction
            )
            states[sample, global_start:global_end] = np.maximum(
                states[sample, global_start:global_end],
                block_profile,
            )

            block_road_id[sample, block_index] = road
            block_start[sample, block_index] = start_position
            block_width[sample, block_index] = width
            block_high_density[sample, block_index] = high_density

    metadata = {
        "initial_baseline_density": baseline_density,
        "initial_road_baseline_density": road_baseline_density,
        "initial_block_count": block_count,
        "initial_block_road_id": block_road_id,
        "initial_block_start": block_start,
        "initial_block_width": block_width,
        "initial_block_high_density": block_high_density,
    }
    return states, metadata


def generate_rollouts(
    *,
    rho0: np.ndarray,
    network: RoadNetwork,
    dt: float,
    window: int,
    v_max: float,
    rho_max: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate state and physical-transfer-flux trajectories."""
    if window <= 0:
        raise ValueError("window must be positive.")
    num_samples, num_cells = rho0.shape
    num_transfers = network.transfer_index.shape[1]
    rollout = np.empty(
        (num_samples, window + 1, num_cells), dtype=np.float64
    )
    transfer_fluxes = np.empty(
        (num_samples, window, num_transfers), dtype=np.float64
    )
    rollout[:, 0] = rho0
    for step in range(window):
        rollout[:, step + 1], transfer_fluxes[:, step] = (
            advance_network_one_step(
                rollout[:, step],
                network=network,
                dt=dt,
                v_max=v_max,
                rho_max=rho_max,
            )
        )
    return rollout, transfer_fluxes


def generate_parameterized_rollouts(
    *,
    rho0: np.ndarray,
    cells_per_road: int,
    dt: float,
    window: int,
    v_max: np.ndarray,
    rho_max: float,
    branch_split: np.ndarray,
    upper_merge_priority: np.ndarray,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Generate trajectories whose physical parameters vary by sample."""
    num_samples = rho0.shape[0]
    parameter_arrays = (v_max, branch_split, upper_merge_priority)
    if any(np.asarray(value).shape != (num_samples,) for value in parameter_arrays):
        raise ValueError(
            "Every trajectory parameter must have shape [num_samples]."
        )

    rollout: Optional[np.ndarray] = None
    transfer_fluxes: Optional[np.ndarray] = None
    trajectory_edge_attr: Optional[np.ndarray] = None
    trajectory_turning_matrix: Optional[np.ndarray] = None
    trajectory_merge_priority_matrix: Optional[np.ndarray] = None
    for sample in range(num_samples):
        sample_network = build_merge_diverge_network(
            cells_per_road=cells_per_road,
            branch_split=float(branch_split[sample]),
            upper_merge_priority=float(upper_merge_priority[sample]),
        )
        sample_rollout, sample_fluxes = generate_rollouts(
            rho0=rho0[sample : sample + 1],
            network=sample_network,
            dt=dt,
            window=window,
            v_max=float(v_max[sample]),
            rho_max=rho_max,
        )
        if rollout is None:
            rollout = np.empty(
                (num_samples,) + sample_rollout.shape[1:],
                dtype=np.float64,
            )
            transfer_fluxes = np.empty(
                (num_samples,) + sample_fluxes.shape[1:],
                dtype=np.float64,
            )
            trajectory_edge_attr = np.empty(
                (num_samples,) + sample_network.edge_attr.shape,
                dtype=np.float64,
            )
            trajectory_turning_matrix = np.empty(
                (num_samples,) + sample_network.turning_matrix.shape,
                dtype=np.float64,
            )
            trajectory_merge_priority_matrix = np.empty(
                (num_samples,) + sample_network.merge_priority_matrix.shape,
                dtype=np.float64,
            )
        rollout[sample] = sample_rollout[0]
        transfer_fluxes[sample] = sample_fluxes[0]
        trajectory_edge_attr[sample] = sample_network.edge_attr
        trajectory_turning_matrix[sample] = sample_network.turning_matrix
        trajectory_merge_priority_matrix[sample] = (
            sample_network.merge_priority_matrix
        )

    if (
        rollout is None
        or transfer_fluxes is None
        or trajectory_edge_attr is None
        or trajectory_turning_matrix is None
        or trajectory_merge_priority_matrix is None
    ):
        raise RuntimeError("No parameterized rollouts were generated.")
    return (
        rollout,
        transfer_fluxes,
        trajectory_edge_attr,
        trajectory_turning_matrix,
        trajectory_merge_priority_matrix,
    )


def save_dataset(
    *,
    out_path: Path,
    rollout: np.ndarray,
    transfer_fluxes: np.ndarray,
    network: RoadNetwork,
    initial_metadata: Dict[str, np.ndarray],
    metadata: Dict[str, object],
    trajectory_parameters: Dict[str, np.ndarray],
    target_type: str,
    dt: float,
) -> None:
    """Flatten one-step pairs and save the complete network archive."""
    if target_type not in TARGET_TYPES:
        raise ValueError(f"target_type must be one of {TARGET_TYPES}.")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    num_samples, num_time, num_cells = rollout.shape
    window = num_time - 1
    num_pairs = num_samples * window
    rho_now = rollout[:, :-1].reshape(num_pairs, num_cells)
    rho_next = rollout[:, 1:].reshape(num_pairs, num_cells)
    if target_type == "increment":
        target = rho_next - rho_now
    else:
        target = (rho_next - rho_now) / dt

    pair_trajectory_id = np.repeat(
        np.arange(num_samples, dtype=np.int64), window
    )
    pair_time_index = np.tile(
        np.arange(window, dtype=np.int64), num_samples
    )
    saved_trajectory_parameters: Dict[str, np.ndarray] = {}
    pair_parameters: Dict[str, np.ndarray] = {}
    for name, values in trajectory_parameters.items():
        parameter = np.asarray(values)
        if parameter.shape[0] != num_samples:
            raise ValueError(
                f"Trajectory parameter {name!r} has shape {parameter.shape}; "
                f"expected first dimension {num_samples}."
            )
        saved_trajectory_parameters[f"trajectory_{name}"] = parameter
        if parameter.ndim == 1:
            pair_parameters[f"pair_{name}"] = np.repeat(parameter, window)
    total_mass = np.sum(
        rollout * network.cell_lengths[None, None, :], axis=2
    )
    road_mass = np.empty(
        (num_samples, num_time, network.road_lengths.size), dtype=np.float64
    )
    for road in range(network.road_lengths.size):
        start = int(network.road_cell_offsets[road])
        end = int(network.road_cell_offsets[road + 1])
        road_mass[:, :, road] = np.sum(
            rollout[:, :, start:end]
            * network.cell_lengths[None, None, start:end],
            axis=2,
        )

    np.savez_compressed(
        out_path,
        x=rho_now[..., None].astype(np.float32),
        y=target[..., None].astype(np.float32),
        u_next=rho_next[..., None].astype(np.float32),
        pos=network.cell_positions.astype(np.float32),
        edge_index=network.edge_index.astype(np.int64),
        edge_attr=network.edge_attr.astype(np.float32),
        edge_transfer_id=network.edge_transfer_id.astype(np.int64),
        edge_is_physical_direction=(
            network.edge_is_physical_direction.astype(np.bool_)
        ),
        pair_trajectory_id=pair_trajectory_id,
        pair_time_index=pair_time_index,
        rollout_states=rollout[..., None].astype(np.float32),
        rollout_transfer_fluxes=transfer_fluxes[..., None].astype(np.float32),
        total_mass=total_mass.astype(np.float64),
        road_mass=road_mass.astype(np.float64),
        cell_lengths=network.cell_lengths.astype(np.float64),
        cell_widths=network.cell_lengths.astype(np.float64),
        cell_road_id=network.cell_road_id.astype(np.int64),
        cell_local_index=network.cell_local_index.astype(np.int64),
        cell_road_coordinate=network.cell_road_coordinate.astype(np.float64),
        road_names=network.road_names,
        road_lengths=network.road_lengths.astype(np.float64),
        road_num_cells=network.road_num_cells.astype(np.int64),
        road_cell_offsets=network.road_cell_offsets.astype(np.int64),
        road_start_junction=network.road_start_junction.astype(np.int64),
        road_end_junction=network.road_end_junction.astype(np.int64),
        junction_names=network.junction_names,
        junction_types=network.junction_types,
        junction_positions=network.junction_positions.astype(np.float64),
        turning_matrix=network.turning_matrix.astype(np.float64),
        merge_priority_matrix=network.merge_priority_matrix.astype(np.float64),
        transfer_index=network.transfer_index.astype(np.int64),
        transfer_type=network.transfer_type.astype(np.int64),
        transfer_junction_id=network.transfer_junction_id.astype(np.int64),
        transfer_from_road=network.transfer_from_road.astype(np.int64),
        transfer_to_road=network.transfer_to_road.astype(np.int64),
        transfer_turning_fraction=(
            network.transfer_turning_fraction.astype(np.float64)
        ),
        transfer_merge_priority=(
            network.transfer_merge_priority.astype(np.float64)
        ),
        transfer_center_distance=(
            network.transfer_center_distance.astype(np.float64)
        ),
        **saved_trajectory_parameters,
        **pair_parameters,
        **initial_metadata,
        **_metadata_to_npz(metadata),
    )


def _metadata_to_npz(metadata: Dict[str, object]) -> Dict[str, np.ndarray]:
    """Convert scalar metadata to arrays accepted by ``np.savez``."""
    converted: Dict[str, np.ndarray] = {}
    for key, value in metadata.items():
        if isinstance(value, str):
            converted[key] = np.array(value)
        elif isinstance(value, bool):
            converted[key] = np.array(value, dtype=np.bool_)
        elif isinstance(value, int):
            converted[key] = np.array(value, dtype=np.int64)
        elif isinstance(value, float):
            converted[key] = np.array(value, dtype=np.float64)
        else:
            converted[key] = np.asarray(value)
    return converted


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate cell-average LWR data on a closed directed road network "
            "with a merge, diverge, and irregular road lengths."
        )
    )
    parser.add_argument(
        "--num-samples", type=int, default=100, help="Number of trajectories."
    )
    parser.add_argument(
        "--window",
        type=int,
        default=128,
        help="Number of one-step transitions per trajectory.",
    )
    parser.add_argument(
        "--cells-per-road",
        type=int,
        default=24,
        help="Finite-volume cells on each of the four roads.",
    )
    parser.add_argument(
        "--v-max", type=float, default=1.0, help="Free-flow vehicle speed."
    )
    parser.add_argument(
        "--v-max-min",
        type=float,
        default=None,
        help="Minimum trajectory-specific v_max; requires --v-max-max.",
    )
    parser.add_argument(
        "--v-max-max",
        type=float,
        default=None,
        help="Maximum trajectory-specific v_max; requires --v-max-min.",
    )
    parser.add_argument(
        "--rho-max", type=float, default=1.0, help="Jam density."
    )
    parser.add_argument(
        "--CFL", type=float, default=0.45, help="Finite-volume CFL number."
    )
    parser.add_argument(
        "--branch-split",
        type=float,
        default=0.55,
        help="Fraction of diverging traffic sent to the upper branch.",
    )
    parser.add_argument(
        "--branch-split-min",
        type=float,
        default=None,
        help="Minimum trajectory-specific diverge split.",
    )
    parser.add_argument(
        "--branch-split-max",
        type=float,
        default=None,
        help="Maximum trajectory-specific diverge split.",
    )
    parser.add_argument(
        "--upper-merge-priority",
        type=float,
        default=0.55,
        help="Upper branch's priority share when downstream supply is limited.",
    )
    parser.add_argument(
        "--upper-merge-priority-min",
        type=float,
        default=None,
        help="Minimum trajectory-specific upper merge priority.",
    )
    parser.add_argument(
        "--upper-merge-priority-max",
        type=float,
        default=None,
        help="Maximum trajectory-specific upper merge priority.",
    )
    parser.add_argument(
        "--density-min",
        type=float,
        default=0.02,
        help="Minimum sampled background density.",
    )
    parser.add_argument(
        "--density-max",
        type=float,
        default=0.98,
        help="Maximum sampled congestion-block density.",
    )
    parser.add_argument(
        "--min-jump-fraction",
        type=float,
        default=0.20,
        help=(
            "Minimum block-to-background density jump as a fraction of the "
            "sampled density range."
        ),
    )
    parser.add_argument(
        "--num-blocks-min",
        type=int,
        default=1,
        help="Minimum congestion blocks per trajectory.",
    )
    parser.add_argument(
        "--num-blocks-max",
        type=int,
        default=3,
        help="Maximum congestion blocks per trajectory.",
    )
    parser.add_argument(
        "--block-width-fraction-min",
        type=float,
        default=0.15,
        help="Minimum block width as a fraction of its road length.",
    )
    parser.add_argument(
        "--block-width-fraction-max",
        type=float,
        default=0.45,
        help="Maximum block width as a fraction of its road length.",
    )
    parser.add_argument(
        "--road-baseline-jitter",
        type=float,
        default=0.0,
        help=(
            "Maximum road-specific background perturbation as a fraction of "
            "the density range."
        ),
    )
    parser.add_argument(
        "--allow-multiple-blocks-per-road",
        action="store_true",
        help="Sample congestion-block roads with replacement.",
    )
    parser.add_argument(
        "--target-type",
        choices=TARGET_TYPES,
        default="increment",
        help="Whether y contains the state increment or time rate.",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed.")
    parser.add_argument("--out", type=Path, required=True, help="Output .npz path.")
    return parser.parse_args(argv)


def resolve_parameter_range(
    *,
    center: float,
    lower: Optional[float],
    upper: Optional[float],
    name: str,
    minimum: float,
    maximum: Optional[float],
) -> Tuple[float, float]:
    """Resolve an optional sampling interval around a legacy fixed value."""
    if (lower is None) != (upper is None):
        raise ValueError(f"{name}-min and {name}-max must be supplied together.")
    if lower is None or upper is None:
        lower = center
        upper = center
    if lower < minimum or upper < lower:
        raise ValueError(
            f"Require {minimum} <= {name}-min <= {name}-max."
        )
    if maximum is not None and upper > maximum:
        raise ValueError(f"{name}-max must not exceed {maximum}.")
    return float(lower), float(upper)


def sample_parameter(
    rng: np.random.Generator,
    *,
    lower: float,
    upper: float,
    size: int,
) -> np.ndarray:
    """Sample a parameter without consuming RNG state for fixed values."""
    if lower == upper:
        return np.full(size, lower, dtype=np.float64)
    return rng.uniform(lower, upper, size=size)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.v_max <= 0.0:
        raise ValueError("v_max must be positive.")
    if args.rho_max <= 0.0:
        raise ValueError("rho_max must be positive.")
    if not (0.0 < args.CFL <= 1.0):
        raise ValueError("CFL must be in (0, 1].")
    if args.density_max > args.rho_max:
        raise ValueError("density_max must not exceed rho_max.")

    v_max_min, v_max_max = resolve_parameter_range(
        center=args.v_max,
        lower=args.v_max_min,
        upper=args.v_max_max,
        name="v-max",
        minimum=np.nextafter(0.0, 1.0),
        maximum=None,
    )
    branch_split_min, branch_split_max = resolve_parameter_range(
        center=args.branch_split,
        lower=args.branch_split_min,
        upper=args.branch_split_max,
        name="branch-split",
        minimum=np.nextafter(0.0, 1.0),
        maximum=np.nextafter(1.0, 0.0),
    )
    priority_min, priority_max = resolve_parameter_range(
        center=args.upper_merge_priority,
        lower=args.upper_merge_priority_min,
        upper=args.upper_merge_priority_max,
        name="upper-merge-priority",
        minimum=np.nextafter(0.0, 1.0),
        maximum=np.nextafter(1.0, 0.0),
    )
    network = build_merge_diverge_network(
        cells_per_road=args.cells_per_road,
        branch_split=args.branch_split,
        upper_merge_priority=args.upper_merge_priority,
    )
    rng = np.random.default_rng(args.seed)
    trajectory_v_max = sample_parameter(
        rng,
        lower=v_max_min,
        upper=v_max_max,
        size=args.num_samples,
    )
    trajectory_branch_split = sample_parameter(
        rng,
        lower=branch_split_min,
        upper=branch_split_max,
        size=args.num_samples,
    )
    trajectory_upper_merge_priority = sample_parameter(
        rng,
        lower=priority_min,
        upper=priority_max,
        size=args.num_samples,
    )
    dt = float(
        args.CFL
        * np.min(network.cell_lengths)
        / np.max(trajectory_v_max)
    )
    rho0, initial_metadata = generate_initial_conditions(
        num_samples=args.num_samples,
        network=network,
        density_min=args.density_min,
        density_max=args.density_max,
        min_jump_fraction=args.min_jump_fraction,
        num_blocks_min=args.num_blocks_min,
        num_blocks_max=args.num_blocks_max,
        block_width_fraction_min=args.block_width_fraction_min,
        block_width_fraction_max=args.block_width_fraction_max,
        road_baseline_jitter=args.road_baseline_jitter,
        allow_multiple_blocks_per_road=args.allow_multiple_blocks_per_road,
        rng=rng,
    )
    (
        rollout,
        transfer_fluxes,
        trajectory_edge_attr,
        trajectory_turning_matrix,
        trajectory_merge_priority_matrix,
    ) = generate_parameterized_rollouts(
        rho0=rho0,
        cells_per_road=args.cells_per_road,
        dt=dt,
        window=args.window,
        v_max=trajectory_v_max,
        rho_max=args.rho_max,
        branch_split=trajectory_branch_split,
        upper_merge_priority=trajectory_upper_merge_priority,
    )

    total_mass = np.sum(
        rollout * network.cell_lengths[None, None, :], axis=2
    )
    mass_error = total_mass - total_mass[:, :1]
    max_abs_mass_error = float(np.max(np.abs(mass_error)))
    density_min_observed = float(np.min(rollout))
    density_max_observed = float(np.max(rollout))
    tolerance = 1.0e-12
    if density_min_observed < -tolerance:
        raise RuntimeError(
            f"Generated a negative density: {density_min_observed}."
        )
    if density_max_observed > args.rho_max + tolerance:
        raise RuntimeError(
            "Generated a density above rho_max: "
            f"{density_max_observed} > {args.rho_max}."
        )

    metadata: Dict[str, object] = {
        "model": "network_LWR",
        "network_name": "closed_merge_diverge",
        "equation": "rho_t + f(rho)_x = 0 on each directed road",
        "flux": "v_max*rho*(1-rho/rho_max)",
        "scheme": "godunov_demand_supply",
        "boundary": "closed_network",
        "state_representation": "finite_volume_cell_average",
        "position_representation": "schematic_2d_road_embedding",
        "junction_solver": "demand_supply_turning_and_priority",
        "network_conserved_quantity": "sum(rho_i*cell_length_i)",
        "v_max": float(args.v_max),
        "v_max_min": v_max_min,
        "v_max_max": v_max_max,
        "rho_max": float(args.rho_max),
        "rho_crit": float(0.5 * args.rho_max),
        "flux_max": float(0.25 * args.v_max * args.rho_max),
        "flux_max_min": float(0.25 * v_max_min * args.rho_max),
        "flux_max_max": float(0.25 * v_max_max * args.rho_max),
        "CFL": float(args.CFL),
        "dt": dt,
        "num_samples": int(args.num_samples),
        "window": int(args.window),
        "num_pairs": int(args.num_samples * args.window),
        "num_cells": int(network.cell_lengths.size),
        "num_roads": int(network.road_lengths.size),
        "num_junctions": int(len(network.junctions)),
        "num_transfers": int(network.transfer_index.shape[1]),
        "cells_per_road": int(args.cells_per_road),
        "cell_length_min": float(np.min(network.cell_lengths)),
        "cell_length_max": float(np.max(network.cell_lengths)),
        "branch_split": float(args.branch_split),
        "branch_split_min": branch_split_min,
        "branch_split_max": branch_split_max,
        "upper_merge_priority": float(args.upper_merge_priority),
        "upper_merge_priority_min": priority_min,
        "upper_merge_priority_max": priority_max,
        "density_min": float(args.density_min),
        "density_max": float(args.density_max),
        "density_min_observed": density_min_observed,
        "density_max_observed": density_max_observed,
        "min_jump_fraction": float(args.min_jump_fraction),
        "num_blocks_min": int(args.num_blocks_min),
        "num_blocks_max": int(args.num_blocks_max),
        "block_width_fraction_min": float(args.block_width_fraction_min),
        "block_width_fraction_max": float(args.block_width_fraction_max),
        "road_baseline_jitter": float(args.road_baseline_jitter),
        "allow_multiple_blocks_per_road": bool(
            args.allow_multiple_blocks_per_road
        ),
        "trajectory_parameter_variation": bool(
            v_max_min != v_max_max
            or branch_split_min != branch_split_max
            or priority_min != priority_max
        ),
        "edge_attr_is_trajectory_invariant": bool(
            branch_split_min == branch_split_max
            and priority_min == priority_max
        ),
        "conditioning_fields": (
            "v_max,branch_split,upper_merge_priority"
        ),
        "target_type": str(args.target_type),
        "seed": int(args.seed),
        "max_abs_mass_error": max_abs_mass_error,
        "edge_attr_columns": (
            "traffic_direction,dx_position,dy_position,euclidean_distance,"
            "physical_center_distance,source_cell_length,target_cell_length,"
            "is_junction,turning_fraction,merge_priority"
        ),
        "transfer_type_labels": "0=internal_road,1=junction_movement",
    }
    save_dataset(
        out_path=args.out,
        rollout=rollout,
        transfer_fluxes=transfer_fluxes,
        network=network,
        initial_metadata=initial_metadata,
        metadata=metadata,
        trajectory_parameters={
            "v_max": trajectory_v_max,
            "branch_split": trajectory_branch_split,
            "upper_merge_priority": trajectory_upper_merge_priority,
            "edge_attr": trajectory_edge_attr.astype(np.float32),
            "turning_matrix": trajectory_turning_matrix,
            "merge_priority_matrix": trajectory_merge_priority_matrix,
        },
        target_type=args.target_type,
        dt=dt,
    )

    print("Generated network LWR traffic dataset")
    print(f"  out: {args.out}")
    print(f"  trajectories: {args.num_samples}")
    print(f"  rollout window: {args.window}")
    print(f"  cells: {network.cell_lengths.size}")
    print(f"  roads: {network.road_lengths.size}")
    print(f"  junctions: {len(network.junctions)}")
    print(f"  physical transfers: {network.transfer_index.shape[1]}")
    print(f"  message-passing edges: {network.edge_index.shape[1]}")
    print(f"  dt: {dt:.8e}")
    print(
        "  v_max range: "
        f"[{trajectory_v_max.min():.6g}, {trajectory_v_max.max():.6g}]"
    )
    print(
        "  branch-split range: "
        f"[{trajectory_branch_split.min():.6g}, "
        f"{trajectory_branch_split.max():.6g}]"
    )
    print(
        "  upper-merge-priority range: "
        f"[{trajectory_upper_merge_priority.min():.6g}, "
        f"{trajectory_upper_merge_priority.max():.6g}]"
    )
    print(
        "  cell-length range: "
        f"[{np.min(network.cell_lengths):.8e}, "
        f"{np.max(network.cell_lengths):.8e}]"
    )
    print(
        "  observed density range: "
        f"[{density_min_observed:.6g}, {density_max_observed:.6g}]"
    )
    print(f"  max |mass(t)-mass(0)|: {max_abs_mass_error:.8e}")


if __name__ == "__main__":
    main()
