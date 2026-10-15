#!/usr/bin/env python3
"""Evaluate and visualize autoregressive GNN test rollouts.

This script loads a checkpoint created by ``train.py``, recovers
the exact held-out test trajectory IDs stored in that checkpoint, and rolls the
model forward autoregressively from each test initial condition.  It saves:

* predictions and ground truth in a compressed ``.npz`` archive;
* a CSV table of per-trajectory metrics;
* a test-set summary figure;
* network snapshots, per-road space-time diagrams, and diagnostic plots for
  representative test trajectories (best, median, and worst rollout RMSE).

Use ``--all-test`` to create individual plots for every test trajectory, or
``--trajectories`` to select particular held-out trajectory IDs.

Example:

    python scripts/plot_sageconv_rollouts.py \
        --checkpoint outputs/sageconv_lwr/checkpoint.pt

By default, figures and evaluation artifacts are written to
``<checkpoint directory>/rollout_plots``.  Pass ``--output-dir`` to override
that location.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

try:
    import torch
except ModuleNotFoundError as exc:
    raise SystemExit(
        "This script requires PyTorch and PyTorch Geometric. Install the "
        "dependencies required by scripts/train.py first."
    ) from exc

from plot_network_lwr_traffic_data import (
    _draw_density_network,
    _format_axes,
    _road_indices,
    _time_edges,
)
from physics_inputs import (
    PhysicsInputConfig,
    build_lwr_advection_metadata,
    build_physics_augmented_inputs_1d,
    count_physics_input_channels,
    resolve_physics_input_cfg,
)
from train import (
    Normalization,
    apply_mass_projection,
    choose_device,
    decode_prediction,
    load_dataset,
    model_from_checkpoint,
    predict_normalized_target,
)


def load_checkpoint(path: Path) -> Dict[str, object]:
    """Load a trusted training checkpoint on the CPU."""
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    try:
        checkpoint = torch.load(
            path, map_location="cpu", weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("The checkpoint must contain a dictionary.")

    required = (
        "model_state_dict",
        "model_kwargs",
        "normalization",
        "edge_index",
        "static_features",
        "split_trajectory_ids",
        "target_type",
        "dt",
    )
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise KeyError(f"Checkpoint is missing required fields: {missing}")
    return checkpoint


def load_rollout_states(
    data: Dict[str, np.ndarray],
) -> np.ndarray:
    """Return rollout states shaped [trajectory, time, node]."""
    if "rollout_states" not in data:
        raise KeyError("Dataset is missing rollout_states.")
    rollout = np.asarray(data["rollout_states"], dtype=np.float32)
    if rollout.ndim == 4 and rollout.shape[-1] == 1:
        rollout = rollout[..., 0]
    if rollout.ndim != 3:
        raise ValueError(
            "rollout_states must have shape [trajectory, time, node] or that "
            f"shape plus a singleton channel, got {rollout.shape}."
        )
    return rollout


def make_batched_edge_index(
    edge_index: torch.Tensor,
    *,
    batch_size: int,
    num_nodes: int,
    device: torch.device,
) -> torch.Tensor:
    """Repeat a shared graph as one disjoint batched graph."""
    edge_index = edge_index.to(device=device, dtype=torch.long)
    offsets = (
        torch.arange(batch_size, device=device, dtype=torch.long)
        * num_nodes
    )
    batched = edge_index.unsqueeze(0) + offsets[:, None, None]
    return batched.permute(1, 0, 2).reshape(2, -1)


@torch.inference_mode()
def predict_rollouts(
    model: torch.nn.Module,
    initial_states: np.ndarray,
    *,
    num_steps: int,
    edge_index: torch.Tensor,
    edge_attributes: np.ndarray,
    cell_lengths: np.ndarray,
    static_features: torch.Tensor,
    trajectory_conditioning: np.ndarray,
    trajectory_v_max: np.ndarray,
    normalization: Normalization,
    physics_cfg: PhysicsInputConfig,
    target_type: str,
    mass_projection: bool,
    dt: float,
    rho_max: float,
    device: torch.device,
) -> np.ndarray:
    """Autoregressively predict every supplied trajectory in parallel."""
    if initial_states.ndim != 2:
        raise ValueError("initial_states must have shape [trajectory, node].")
    batch_size, num_nodes = initial_states.shape
    if static_features.shape[0] != num_nodes:
        raise ValueError("Checkpoint static features do not match the dataset.")
    if trajectory_conditioning.shape[0] != batch_size:
        raise ValueError(
            "Trajectory conditioning does not match the rollout batch."
        )
    trajectory_v_max = np.asarray(trajectory_v_max, dtype=np.float32)
    if trajectory_v_max.shape != (batch_size,):
        raise ValueError("trajectory_v_max must have shape [trajectory].")
    if np.any(trajectory_v_max <= 0.0):
        raise ValueError("trajectory_v_max values must be positive.")
    if rho_max <= 0.0:
        raise ValueError("rho_max must be positive.")
    edge_attributes = np.asarray(edge_attributes, dtype=np.float32)
    if edge_attributes.ndim == 2:
        edge_attributes = np.broadcast_to(
            edge_attributes[None, ...],
            (batch_size,) + edge_attributes.shape,
        )
    if (
        edge_attributes.ndim != 3
        or edge_attributes.shape[0] != batch_size
        or edge_attributes.shape[1] != edge_index.shape[1]
    ):
        raise ValueError(
            "edge_attributes must have shape [trajectory, edge, channel]."
        )
    cell_lengths = np.asarray(cell_lengths, dtype=np.float32)
    if cell_lengths.shape != (num_nodes,):
        raise ValueError("cell_lengths must have shape [node].")
    if np.any(cell_lengths <= 0.0):
        raise ValueError("cell_lengths values must be positive.")

    # Using lists also works in environments whose PyTorch NumPy bridge is
    # unavailable because of a local binary-version mismatch.
    current = torch.tensor(
        initial_states.tolist(), dtype=torch.float32, device=device
    ).unsqueeze(-1)
    static = static_features.to(
        device=device, dtype=torch.float32
    ).unsqueeze(0)
    static = static.expand(batch_size, -1, -1)
    conditioning = torch.tensor(
        trajectory_conditioning.tolist(),
        dtype=torch.float32,
        device=device,
    ).unsqueeze(1)
    conditioning = conditioning.expand(batch_size, num_nodes, -1)
    batched_edges = make_batched_edge_index(
        edge_index,
        batch_size=batch_size,
        num_nodes=num_nodes,
        device=device,
    )
    edge_attr_channels = int(edge_attributes.shape[-1])
    if edge_attr_channels == 0:
        batched_edge_attr = torch.empty(
            (batch_size * edge_index.shape[1], 0),
            dtype=torch.float32,
            device=device,
        )
    else:
        batched_edge_attr = torch.tensor(
            edge_attributes.tolist(),
            dtype=torch.float32,
            device=device,
        ).reshape(-1, edge_attr_channels)
    batched_cell_lengths = torch.tensor(
        cell_lengths.tolist(),
        dtype=torch.float32,
        device=device,
    ).view(1, num_nodes, 1)
    batched_cell_lengths = batched_cell_lengths.expand(
        batch_size, -1, -1
    ).reshape(-1, 1)
    v_max_nodes = torch.tensor(
        trajectory_v_max.tolist(), dtype=torch.float32, device=device
    ).view(batch_size, 1, 1)
    v_max_nodes = v_max_nodes.expand(batch_size, num_nodes, 1).reshape(
        -1, 1
    )
    rho_max_nodes = torch.full_like(batched_cell_lengths, float(rho_max))
    dt_nodes = torch.full_like(batched_cell_lengths, float(dt))
    graph_index = torch.arange(
        batch_size,
        device=device,
        dtype=torch.long,
    ).repeat_interleave(num_nodes)
    advection_metadata = None
    if physics_cfg.enabled and physics_cfg.include_adv:
        advection_metadata = build_lwr_advection_metadata(
            batched_edges, batched_edge_attr
        )

    predicted_states = [current]
    for step in range(num_steps):
        normalized_density = (
            current - normalization.density_mean
        ) / normalization.density_std
        base_features = torch.cat(
            (normalized_density, static, conditioning), dim=-1
        ).reshape(batch_size * num_nodes, -1)
        features = build_physics_augmented_inputs_1d(
            x_base=base_features,
            x_state=current.reshape(batch_size * num_nodes, 1),
            edge_index=batched_edges,
            edge_attr=batched_edge_attr,
            cell_length=batched_cell_lengths,
            dt_node=dt_nodes,
            v_max=v_max_nodes,
            rho_max=rho_max_nodes,
            physics_cfg=physics_cfg,
            step_k=step,
            metadata=advection_metadata,
            validate_inputs=advection_metadata is None,
        )
        normalized_prediction = predict_normalized_target(
            model,
            features,
            batched_edges,
            edge_attr=batched_edge_attr,
            cell_length=batched_cell_lengths,
        )
        normalized_prediction = apply_mass_projection(
            normalized_prediction,
            normalization=normalization,
            cell_length=batched_cell_lengths,
            graph_index=graph_index,
            num_graphs=batch_size,
            enabled=mass_projection,
        ).reshape(batch_size, num_nodes, 1)
        _, current = decode_prediction(
            normalized_prediction,
            current=current,
            normalization=normalization,
            target_type=target_type,
            dt=dt,
        )
        predicted_states.append(current)

    prediction = torch.stack(predicted_states, dim=1).squeeze(-1)
    return np.asarray(prediction.cpu().tolist(), dtype=np.float32)


def load_trajectory_conditioning(
    checkpoint: Dict[str, object],
    data: Dict[str, np.ndarray],
    trajectory_ids: np.ndarray,
) -> np.ndarray:
    """Load and normalize parameters used to condition a trained model."""
    keys = list(checkpoint.get("conditioning_keys", []))
    if not keys:
        return np.empty((trajectory_ids.size, 0), dtype=np.float32)

    columns = []
    for pair_key in keys:
        if not str(pair_key).startswith("pair_"):
            raise ValueError(f"Invalid conditioning key: {pair_key!r}.")
        trajectory_key = "trajectory_" + str(pair_key)[len("pair_") :]
        if trajectory_key not in data:
            raise KeyError(
                f"Dataset is missing conditioning field {trajectory_key!r}."
            )
        values = np.asarray(data[trajectory_key], dtype=np.float32)
        columns.append(values[trajectory_ids])
    raw = np.column_stack(columns).astype(np.float32, copy=False)
    mean = np.asarray(
        checkpoint.get("conditioning_mean"), dtype=np.float32
    )
    std = np.asarray(
        checkpoint.get("conditioning_std"), dtype=np.float32
    )
    if mean.shape != (len(keys),) or std.shape != (len(keys),):
        raise ValueError("Checkpoint conditioning statistics are invalid.")
    return ((raw - mean) / std).astype(np.float32)


def compute_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    *,
    cell_lengths: np.ndarray,
    rho_max: float,
) -> Dict[str, np.ndarray]:
    """Compute time-resolved model, persistence, mass, and bound metrics."""
    if truth.shape != prediction.shape:
        raise ValueError("Truth and prediction rollout shapes do not match.")
    persistence = np.broadcast_to(
        truth[:, :1], truth.shape
    ).astype(np.float32, copy=False)
    model_error = prediction - truth
    persistence_error = persistence - truth
    model_rmse_by_time = np.sqrt(np.mean(model_error**2, axis=2))
    persistence_rmse_by_time = np.sqrt(
        np.mean(persistence_error**2, axis=2)
    )

    lengths = np.asarray(cell_lengths, dtype=np.float64)
    truth_mass = np.sum(truth * lengths[None, None, :], axis=2)
    prediction_mass = np.sum(
        prediction * lengths[None, None, :], axis=2
    )
    mass_error = prediction_mass - truth_mass
    outside_bounds = (prediction < 0.0) | (prediction > rho_max)

    return {
        "persistence": persistence,
        "model_error": model_error,
        "model_rmse_by_time": model_rmse_by_time,
        "persistence_rmse_by_time": persistence_rmse_by_time,
        "model_rollout_rmse": np.sqrt(
            np.mean(model_error[:, 1:] ** 2, axis=(1, 2))
        ),
        "persistence_rollout_rmse": np.sqrt(
            np.mean(persistence_error[:, 1:] ** 2, axis=(1, 2))
        ),
        "truth_mass": truth_mass,
        "prediction_mass": prediction_mass,
        "mass_error": mass_error,
        "outside_bounds_fraction": np.mean(
            outside_bounds[:, 1:], axis=(1, 2)
        ),
    }


def representative_trajectories(
    trajectory_ids: np.ndarray,
    rollout_rmse: np.ndarray,
) -> List[int]:
    """Choose distinct best, median, and worst test trajectories."""
    order = np.argsort(rollout_rmse)
    positions = (0, len(order) // 2, len(order) - 1)
    return list(
        dict.fromkeys(int(trajectory_ids[order[position]]) for position in positions)
    )


def validate_requested_trajectories(
    requested: Sequence[int],
    test_trajectory_ids: np.ndarray,
) -> List[int]:
    """Require every requested trajectory to belong to the checkpoint test set."""
    test_set = set(int(value) for value in test_trajectory_ids)
    unique = list(dict.fromkeys(int(value) for value in requested))
    invalid = [value for value in unique if value not in test_set]
    if invalid:
        raise ValueError(
            f"Trajectory IDs {invalid} are not in the checkpoint test split."
        )
    return unique


def default_snapshot_steps(num_times: int) -> List[int]:
    """Choose five distinct saved states spanning a rollout."""
    values = np.linspace(0, num_times - 1, min(5, num_times))
    return sorted(set(int(round(value)) for value in values))


def validate_snapshot_steps(
    steps: Sequence[int],
    num_times: int,
) -> List[int]:
    """Validate requested saved-state indices."""
    unique = sorted(set(int(step) for step in steps))
    invalid = [step for step in unique if step < 0 or step >= num_times]
    if invalid:
        raise ValueError(
            f"Snapshot steps {invalid} are outside [0, {num_times - 1}]."
        )
    return unique


def plot_test_summary(
    test_trajectory_ids: np.ndarray,
    metrics: Dict[str, np.ndarray],
    output_path: Path,
    *,
    dt: float,
    selected_trajectory_ids: Sequence[int],
    dpi: int,
) -> None:
    """Plot aggregate accuracy, persistence, and conservation diagnostics."""
    model_time = metrics["model_rmse_by_time"]
    persistence_time = metrics["persistence_rmse_by_time"]
    state_times = np.arange(model_time.shape[1], dtype=np.float64) * dt

    figure, axes = plt.subplots(
        2, 2, figsize=(11.0, 8.0), constrained_layout=True
    )
    ax_time, ax_baseline, ax_mass, ax_rank = axes.ravel()

    for values, color, label in (
        (model_time, "tab:blue", "SAGEConv"),
        (persistence_time, "0.45", "Persistence"),
    ):
        median = np.median(values, axis=0)
        lower, upper = np.quantile(values, (0.25, 0.75), axis=0)
        ax_time.plot(state_times, median, color=color, label=label)
        ax_time.fill_between(
            state_times, lower, upper, color=color, alpha=0.18
        )
    ax_time.set_title("Test rollout error")
    ax_time.set_xlabel("Time")
    ax_time.set_ylabel("Node RMSE")
    ax_time.legend()
    _format_axes(ax_time)

    model_rmse = metrics["model_rollout_rmse"]
    persistence_rmse = metrics["persistence_rollout_rmse"]
    limit = 1.05 * max(
        float(np.max(model_rmse)),
        float(np.max(persistence_rmse)),
        1.0e-8,
    )
    ax_baseline.scatter(
        persistence_rmse, model_rmse, color="tab:blue", alpha=0.78
    )
    ax_baseline.plot([0.0, limit], [0.0, limit], "--", color="0.4")
    ax_baseline.set_xlim(0.0, limit)
    ax_baseline.set_ylim(0.0, limit)
    ax_baseline.set_aspect("equal", adjustable="box")
    ax_baseline.set_title("Model versus persistence")
    ax_baseline.set_xlabel("Persistence rollout RMSE")
    ax_baseline.set_ylabel("SAGEConv rollout RMSE")
    _format_axes(ax_baseline)

    absolute_mass_error = np.abs(metrics["mass_error"])
    ax_mass.plot(
        state_times,
        np.median(absolute_mass_error, axis=0),
        color="tab:purple",
        label="Median",
    )
    ax_mass.fill_between(
        state_times,
        np.quantile(absolute_mass_error, 0.25, axis=0),
        np.quantile(absolute_mass_error, 0.75, axis=0),
        color="tab:purple",
        alpha=0.18,
        label="Interquartile range",
    )
    ax_mass.plot(
        state_times,
        np.max(absolute_mass_error, axis=0),
        color="tab:red",
        linewidth=1.0,
        label="Maximum",
    )
    ax_mass.set_title("Predicted total-mass error")
    ax_mass.set_xlabel("Time")
    ax_mass.set_ylabel(r"$|\hat M(t)-M(t)|$")
    ax_mass.legend(fontsize=8)
    _format_axes(ax_mass)

    order = np.argsort(model_rmse)
    ordered_ids = test_trajectory_ids[order]
    selected = set(int(value) for value in selected_trajectory_ids)
    colors = [
        "tab:orange" if int(value) in selected else "tab:blue"
        for value in ordered_ids
    ]
    ax_rank.bar(
        np.arange(order.size),
        model_rmse[order],
        color=colors,
        alpha=0.82,
    )
    ax_rank.set_title("Test trajectories ranked by rollout RMSE")
    ax_rank.set_xlabel("Rank")
    ax_rank.set_ylabel("Rollout RMSE")
    _format_axes(ax_rank)

    figure.suptitle("SAGEConv held-out autoregressive rollouts", fontsize=14)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


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


def road_spatial_edges(
    data: Dict[str, np.ndarray],
    road: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return global road indices and finite-volume cell edges."""
    indices = _road_indices(data, road)
    coordinates = np.asarray(
        data["cell_road_coordinate"], dtype=np.float64
    )[indices]
    lengths = np.asarray(data["cell_lengths"], dtype=np.float64)[indices]
    edges = np.concatenate(
        (
            coordinates - 0.5 * lengths,
            np.array([coordinates[-1] + 0.5 * lengths[-1]]),
        )
    )
    return indices, edges


def _trajectory_transfer_fluxes(
    data: Dict[str, np.ndarray],
    trajectory_index: Optional[int],
) -> Optional[np.ndarray]:
    """Return scalar ground-truth transfer fluxes for one trajectory."""
    if trajectory_index is None or "rollout_transfer_fluxes" not in data:
        return None
    fluxes = np.asarray(data["rollout_transfer_fluxes"], dtype=np.float64)
    if fluxes.ndim == 4 and fluxes.shape[-1] == 1:
        fluxes = fluxes[..., 0]
    if fluxes.ndim != 3 or not 0 <= trajectory_index < fluxes.shape[0]:
        return None
    return fluxes[trajectory_index]


def busiest_road_indices(
    data: Dict[str, np.ndarray],
    truth: np.ndarray,
    *,
    trajectory_index: Optional[int],
    maximum: int,
) -> np.ndarray:
    """Rank roads by ground-truth traffic flux, with density fallback."""
    road_names = np.asarray(data["road_names"])
    num_roads = int(road_names.size)
    if maximum <= 0:
        raise ValueError("maximum must be positive.")
    if num_roads <= maximum:
        return np.arange(num_roads, dtype=np.int64)

    scores = np.empty(num_roads, dtype=np.float64)
    fluxes = _trajectory_transfer_fluxes(data, trajectory_index)
    can_use_flux = (
        fluxes is not None
        and "transfer_type" in data
        and "transfer_from_road" in data
    )
    if can_use_flux:
        fluxes = fluxes[: max(truth.shape[0] - 1, 1)]
        transfer_type = np.asarray(data["transfer_type"], dtype=np.int64)
        from_road = np.asarray(data["transfer_from_road"], dtype=np.int64)
        for road in range(num_roads):
            transfers = np.flatnonzero(
                (transfer_type == 0) & (from_road == road)
            )
            scores[road] = (
                float(np.mean(np.abs(fluxes[:, transfers])))
                if transfers.size
                else 0.0
            )
    else:
        for road in range(num_roads):
            indices = _road_indices(data, road)
            scores[road] = float(np.mean(truth[:, indices]))
    order = np.argsort(-scores, kind="stable")
    return order[:maximum].astype(np.int64, copy=False)


def plot_spacetime_comparison(
    data: Dict[str, np.ndarray],
    truth: np.ndarray,
    prediction: np.ndarray,
    trajectory_id: int,
    output_path: Path,
    *,
    dt: float,
    rho_max: float,
    dpi: int,
    cmap: str,
    trajectory_index: Optional[int] = None,
    max_roads: int = 6,
) -> None:
    """Plot the busiest roads' truth, prediction, and space-time error."""
    road_names = np.asarray(data["road_names"])
    time_edges = _time_edges(truth.shape[0], dt)
    error = prediction - truth
    selected_roads = busiest_road_indices(
        data,
        truth,
        trajectory_index=trajectory_index,
        maximum=max_roads,
    )
    selected_nodes = np.concatenate(
        [_road_indices(data, int(road)) for road in selected_roads]
    )
    error_limit = max(
        float(np.max(np.abs(error[:, selected_nodes]))),
        1.0e-8,
    )

    figure, axes = plt.subplots(
        selected_roads.size,
        3,
        figsize=(11.2, 2.65 * selected_roads.size),
        constrained_layout=True,
        squeeze=False,
        sharey=True,
    )
    density_mesh = None
    error_mesh = None
    for row, road_value in enumerate(selected_roads):
        road = int(road_value)
        indices, spatial_edges = road_spatial_edges(data, road)
        panels = (
            (truth[:, indices], cmap, 0.0, rho_max),
            (prediction[:, indices], cmap, 0.0, rho_max),
            (error[:, indices], "coolwarm", -error_limit, error_limit),
        )
        for column, (values, color_map, lower, upper) in enumerate(panels):
            mesh = axes[row, column].pcolormesh(
                spatial_edges,
                time_edges,
                values,
                shading="flat",
                cmap=color_map,
                vmin=lower,
                vmax=upper,
                rasterized=True,
            )
            if column < 2:
                density_mesh = mesh
            else:
                error_mesh = mesh
            axes[row, column].set_xlim(
                0.0, float(data["road_lengths"][road])
            )
            axes[row, column].set_ylim(
                time_edges[-1], max(0.0, time_edges[0])
            )
            axes[row, column].set_xlabel("Distance along road")
            _format_axes(axes[row, column])
        axes[row, 0].set_ylabel(
            str(road_names[road]).replace("_", " ") + "\nTime"
        )
    axes[0, 0].set_title("Ground truth")
    axes[0, 1].set_title("Autoregressive prediction")
    axes[0, 2].set_title("Signed error")

    if density_mesh is not None:
        colorbar = figure.colorbar(
            density_mesh,
            ax=list(axes[:, :2].ravel()),
            pad=0.012,
            shrink=0.88,
        )
        colorbar.set_label(r"Density $\rho$")
    if error_mesh is not None:
        colorbar = figure.colorbar(
            error_mesh,
            ax=list(axes[:, 2]),
            pad=0.012,
            shrink=0.88,
        )
        colorbar.set_label(r"$\hat\rho-\rho$")
    figure.suptitle(
        f"Held-out trajectory {trajectory_id}: "
        f"{selected_roads.size} busiest roads by GT flux",
        fontsize=14,
    )
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def junction_cell_indices(data: Dict[str, np.ndarray]) -> np.ndarray:
    """Return the cells immediately adjacent to physical junction transfers."""
    transfer_type = np.asarray(data["transfer_type"], dtype=np.int64)
    transfer_index = np.asarray(data["transfer_index"], dtype=np.int64)
    junction_transfers = np.flatnonzero(transfer_type == 1)
    return np.unique(transfer_index[:, junction_transfers])


def busiest_junction_groups(
    data: Dict[str, np.ndarray],
    truth: np.ndarray,
    *,
    trajectory_index: Optional[int],
    maximum: int,
) -> List[Tuple[int, np.ndarray]]:
    """Return busiest junction IDs and their unique adjacent cells."""
    if maximum <= 0:
        raise ValueError("maximum must be positive.")
    transfer_type = np.asarray(data["transfer_type"], dtype=np.int64)
    transfer_index = np.asarray(data["transfer_index"], dtype=np.int64)
    junction_mask = transfer_type == 1
    if "transfer_junction_id" not in data:
        cells = junction_cell_indices(data)
        scores = np.mean(truth[:, cells], axis=0)
        selected = cells[np.argsort(-scores, kind="stable")[:maximum]]
        return [(int(node), np.array([node], dtype=np.int64)) for node in selected]

    transfer_junction = np.asarray(
        data["transfer_junction_id"], dtype=np.int64
    )
    junction_ids = np.unique(transfer_junction[junction_mask])
    fluxes = _trajectory_transfer_fluxes(data, trajectory_index)
    if fluxes is not None:
        fluxes = fluxes[: max(truth.shape[0] - 1, 1)]
    ranked: List[Tuple[float, int, np.ndarray]] = []
    for junction in junction_ids:
        transfers = np.flatnonzero(
            junction_mask & (transfer_junction == junction)
        )
        cells = np.unique(transfer_index[:, transfers])
        if fluxes is not None:
            score = float(np.mean(np.sum(np.abs(fluxes[:, transfers]), axis=1)))
        else:
            score = float(np.mean(truth[:, cells]))
        ranked.append((score, int(junction), cells))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [(junction, cells) for _, junction, cells in ranked[:maximum]]


def junction_label(data: Dict[str, np.ndarray], junction: int) -> str:
    """Create a compact junction label, falling back to a numeric ID."""
    if "junction_names" in data:
        names = np.asarray(data["junction_names"])
        if 0 <= junction < names.size:
            return str(names[junction]).replace("_", " ")
    return f"junction {junction:03d}"


def plot_trajectory_diagnostics(
    data: Dict[str, np.ndarray],
    truth: np.ndarray,
    prediction: np.ndarray,
    trajectory_id: int,
    output_path: Path,
    *,
    dt: float,
    rho_max: float,
    cell_lengths: np.ndarray,
    dpi: int,
    trajectory_index: Optional[int] = None,
    max_junctions: int = 6,
) -> None:
    """Plot rollout error, conservation, physical bounds, and node traces."""
    times = np.arange(truth.shape[0], dtype=np.float64) * dt
    persistence = np.broadcast_to(truth[:1], truth.shape)
    model_rmse = np.sqrt(np.mean((prediction - truth) ** 2, axis=1))
    persistence_rmse = np.sqrt(
        np.mean((persistence - truth) ** 2, axis=1)
    )
    truth_mass = np.sum(truth * cell_lengths[None, :], axis=1)
    prediction_mass = np.sum(
        prediction * cell_lengths[None, :], axis=1
    )

    figure, axes = plt.subplots(
        2, 2, figsize=(11.0, 8.0), constrained_layout=True
    )
    ax_error, ax_mass, ax_bounds, ax_nodes = axes.ravel()

    ax_error.plot(times, model_rmse, label="SAGEConv", color="tab:blue")
    ax_error.plot(
        times, persistence_rmse, label="Persistence", color="0.45"
    )
    ax_error.set_title("Autoregressive rollout error")
    ax_error.set_xlabel("Time")
    ax_error.set_ylabel("Node RMSE")
    ax_error.legend()
    _format_axes(ax_error)

    ax_mass.plot(
        times,
        truth_mass - truth_mass[0],
        label="Ground truth",
        color="0.35",
    )
    ax_mass.plot(
        times,
        prediction_mass - prediction_mass[0],
        label="Prediction",
        color="tab:purple",
    )
    ax_mass.axhline(0.0, color="0.2", linewidth=0.8)
    ax_mass.set_title("Network-wide mass drift")
    ax_mass.set_xlabel("Time")
    ax_mass.set_ylabel(r"$M(t)-M(0)$")
    ax_mass.legend()
    _format_axes(ax_mass)

    ax_bounds.plot(
        times,
        np.min(truth, axis=1),
        color="0.45",
        linestyle="-",
        label="GT minimum",
    )
    ax_bounds.plot(
        times,
        np.max(truth, axis=1),
        color="0.45",
        linestyle="--",
        label="GT maximum",
    )
    ax_bounds.plot(
        times,
        np.min(prediction, axis=1),
        color="tab:blue",
        linestyle="-",
        label="Prediction minimum",
    )
    ax_bounds.plot(
        times,
        np.max(prediction, axis=1),
        color="tab:blue",
        linestyle="--",
        label="Prediction maximum",
    )
    ax_bounds.axhline(0.0, color="tab:red", linewidth=0.8)
    ax_bounds.axhline(rho_max, color="tab:red", linewidth=0.8)
    ax_bounds.set_title("Physical density bounds")
    ax_bounds.set_xlabel("Time")
    ax_bounds.set_ylabel(r"Density $\rho$")
    ax_bounds.legend(fontsize=8, ncol=2)
    _format_axes(ax_bounds)

    junction_groups = busiest_junction_groups(
        data,
        truth,
        trajectory_index=trajectory_index,
        maximum=max_junctions,
    )
    colors = plt.get_cmap("tab10")(
        np.linspace(0.0, 1.0, max(len(junction_groups), 1))
    )
    for color, (junction, nodes) in zip(colors, junction_groups):
        label = junction_label(data, junction)
        ax_nodes.plot(
            times,
            np.mean(truth[:, nodes], axis=1),
            color=color,
            linewidth=1.1,
            label=label,
        )
        ax_nodes.plot(
            times,
            np.mean(prediction[:, nodes], axis=1),
            color=color,
            linewidth=1.0,
            linestyle="--",
        )
    ax_nodes.set_title(
        f"{len(junction_groups)} busiest junctions by GT flux: "
        "mean adjacent density\n"
        "(solid GT, dashed prediction)"
    )
    ax_nodes.set_xlabel("Time")
    ax_nodes.set_ylabel(r"Density $\rho$")
    ax_nodes.legend(fontsize=7, ncol=2)
    _format_axes(ax_nodes)

    figure.suptitle(
        f"Held-out trajectory {trajectory_id}: diagnostics", fontsize=14
    )
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def write_metrics_csv(
    path: Path,
    trajectory_ids: np.ndarray,
    metrics: Dict[str, np.ndarray],
) -> None:
    """Write one summary row per test trajectory."""
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "trajectory_id",
                "model_rollout_rmse",
                "persistence_rollout_rmse",
                "rmse_improvement_fraction",
                "final_step_rmse",
                "max_abs_mass_error",
                "outside_bounds_fraction",
            ),
        )
        writer.writeheader()
        for index, trajectory_id in enumerate(trajectory_ids):
            model_rmse = float(metrics["model_rollout_rmse"][index])
            persistence_rmse = float(
                metrics["persistence_rollout_rmse"][index]
            )
            improvement = (
                1.0 - model_rmse / persistence_rmse
                if persistence_rmse > 0.0
                else float("nan")
            )
            writer.writerow(
                {
                    "trajectory_id": int(trajectory_id),
                    "model_rollout_rmse": model_rmse,
                    "persistence_rollout_rmse": persistence_rmse,
                    "rmse_improvement_fraction": improvement,
                    "final_step_rmse": float(
                        metrics["model_rmse_by_time"][index, -1]
                    ),
                    "max_abs_mass_error": float(
                        np.max(np.abs(metrics["mass_error"][index]))
                    ),
                    "outside_bounds_fraction": float(
                        metrics["outside_bounds_fraction"][index]
                    ),
                }
            )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create autoregressive rollout predictions and plots for the "
            "trajectory-level test split stored in a SAGEConv checkpoint."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/sageconv_lwr/checkpoint.pt"),
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Dataset override. Defaults to the path stored in the checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output override. Defaults to CHECKPOINT_DIR/rollout_plots.",
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=None,
        help="Number of autoregressive steps; defaults to the full rollout.",
    )
    parser.add_argument(
        "--mass-projection",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Override the checkpoint's mass-projection setting. By default "
            "the checkpoint setting is used."
        ),
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--trajectories",
        nargs="+",
        type=int,
        help="Specific held-out trajectory IDs to plot.",
    )
    selection.add_argument(
        "--all-test",
        action="store_true",
        help="Create individual figures for every held-out trajectory.",
    )
    parser.add_argument(
        "--snapshot-steps",
        nargs="+",
        type=int,
        default=None,
        help="Saved-state indices for the network comparison figures.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--cmap", default="viridis")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.rollout_steps is not None and args.rollout_steps <= 0:
        raise ValueError("--rollout-steps must be positive.")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive.")

    checkpoint = load_checkpoint(args.checkpoint)
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else args.checkpoint.parent / "rollout_plots"
    )
    data_path = (
        args.data
        if args.data is not None
        else Path(str(checkpoint.get("data_path", "data/lwr_network.npz")))
    )
    data = load_dataset(data_path)
    full_rollout = load_rollout_states(data)
    num_trajectories, num_times, num_nodes = full_rollout.shape
    num_steps = (
        num_times - 1
        if args.rollout_steps is None
        else args.rollout_steps
    )
    if num_steps >= num_times:
        raise ValueError(
            f"--rollout-steps must not exceed {num_times - 1}."
        )

    split_ids = checkpoint["split_trajectory_ids"]
    if not isinstance(split_ids, dict) or "test" not in split_ids:
        raise KeyError("Checkpoint does not contain test trajectory IDs.")
    test_trajectory_ids = np.asarray(split_ids["test"], dtype=np.int64)
    if test_trajectory_ids.size == 0:
        raise ValueError("The checkpoint test split is empty.")
    if (
        int(test_trajectory_ids.min()) < 0
        or int(test_trajectory_ids.max()) >= num_trajectories
    ):
        raise ValueError("Checkpoint test IDs do not match the dataset.")

    model = model_from_checkpoint(checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    device = choose_device(args.device)
    model = model.to(device)
    model.eval()

    normalization = Normalization(**dict(checkpoint["normalization"]))
    physics_cfg = resolve_physics_input_cfg(checkpoint)
    edge_index = checkpoint["edge_index"]
    static_features = checkpoint["static_features"]
    if not isinstance(edge_index, torch.Tensor):
        edge_index = torch.tensor(edge_index, dtype=torch.long)
    if not isinstance(static_features, torch.Tensor):
        static_features = torch.tensor(
            static_features, dtype=torch.float32
        )
    if static_features.shape[0] != num_nodes:
        raise ValueError(
            "Checkpoint node features do not match the dataset node count."
        )
    target_type = str(checkpoint["target_type"])
    checkpoint_mass_projection = checkpoint.get("mass_projection", False)
    if not isinstance(checkpoint_mass_projection, bool):
        raise ValueError("Checkpoint mass_projection must be boolean.")
    mass_projection = (
        checkpoint_mass_projection
        if args.mass_projection is None
        else args.mass_projection
    )
    dt = float(checkpoint["dt"])
    data_dt = float(np.asarray(data["dt"]).item())
    if not np.isclose(dt, data_dt):
        raise ValueError(
            f"Checkpoint dt ({dt}) does not match dataset dt ({data_dt})."
        )

    truth = full_rollout[
        test_trajectory_ids, : num_steps + 1
    ].astype(np.float32, copy=False)
    trajectory_conditioning = load_trajectory_conditioning(
        checkpoint,
        data,
        test_trajectory_ids,
    )
    if "trajectory_v_max" in data:
        trajectory_v_max = np.asarray(
            data["trajectory_v_max"], dtype=np.float32
        )[test_trajectory_ids]
    else:
        trajectory_v_max = np.full(
            test_trajectory_ids.size,
            float(np.asarray(data.get("v_max", 1.0)).item()),
            dtype=np.float32,
        )
    if "trajectory_edge_attr" in data:
        trajectory_edge_attributes = np.asarray(
            data["trajectory_edge_attr"], dtype=np.float32
        )[test_trajectory_ids]
    elif "edge_attr" in data:
        trajectory_edge_attributes = np.asarray(
            data["edge_attr"], dtype=np.float32
        )
    else:
        trajectory_edge_attributes = np.empty(
            (edge_index.shape[1], 0), dtype=np.float32
        )
    cell_lengths = np.asarray(data["cell_lengths"], dtype=np.float32)
    rho_max = float(np.asarray(data.get("rho_max", 1.0)).item())
    expected_features = int(checkpoint["model_kwargs"]["in_channels"])
    actual_features = (
        1
        + int(static_features.shape[1])
        + int(trajectory_conditioning.shape[1])
        + count_physics_input_channels(physics_cfg)
    )
    if actual_features != expected_features:
        raise ValueError(
            f"Rollout inputs have {actual_features} features; checkpoint "
            f"expects {expected_features}."
        )
    prediction = predict_rollouts(
        model,
        truth[:, 0],
        num_steps=num_steps,
        edge_index=edge_index,
        edge_attributes=trajectory_edge_attributes,
        cell_lengths=cell_lengths,
        static_features=static_features,
        trajectory_conditioning=trajectory_conditioning,
        trajectory_v_max=trajectory_v_max,
        normalization=normalization,
        physics_cfg=physics_cfg,
        target_type=target_type,
        mass_projection=mass_projection,
        dt=dt,
        rho_max=rho_max,
        device=device,
    )
    cell_lengths = cell_lengths.astype(np.float64, copy=False)
    metrics = compute_metrics(
        truth,
        prediction,
        cell_lengths=cell_lengths,
        rho_max=rho_max,
    )

    if args.all_test:
        selected_trajectory_ids = [
            int(value) for value in test_trajectory_ids
        ]
    elif args.trajectories is not None:
        selected_trajectory_ids = validate_requested_trajectories(
            args.trajectories, test_trajectory_ids
        )
    else:
        selected_trajectory_ids = representative_trajectories(
            test_trajectory_ids, metrics["model_rollout_rmse"]
        )
    trajectory_to_local = {
        int(trajectory_id): index
        for index, trajectory_id in enumerate(test_trajectory_ids)
    }
    snapshot_steps = (
        default_snapshot_steps(num_steps + 1)
        if args.snapshot_steps is None
        else validate_snapshot_steps(
            args.snapshot_steps, num_steps + 1
        )
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "test_rollout_predictions.npz"
    np.savez_compressed(
        prediction_path,
        trajectory_ids=test_trajectory_ids,
        ground_truth=truth,
        prediction=prediction,
        persistence=metrics["persistence"],
        model_rmse_by_time=metrics["model_rmse_by_time"],
        persistence_rmse_by_time=metrics["persistence_rmse_by_time"],
        mass_error=metrics["mass_error"],
        dt=np.array(dt, dtype=np.float64),
    )
    metrics_path = output_dir / "test_rollout_metrics.csv"
    write_metrics_csv(metrics_path, test_trajectory_ids, metrics)
    summary_path = output_dir / "test_rollout_summary.png"
    plot_test_summary(
        test_trajectory_ids,
        metrics,
        summary_path,
        dt=dt,
        selected_trajectory_ids=selected_trajectory_ids,
        dpi=args.dpi,
    )

    written_paths = [prediction_path, metrics_path, summary_path]
    for trajectory_id in selected_trajectory_ids:
        local = trajectory_to_local[trajectory_id]
        prefix = output_dir / f"trajectory_{trajectory_id:04d}"
        network_path = Path(str(prefix) + "_network.png")
        spacetime_path = Path(str(prefix) + "_spacetime.png")
        diagnostics_path = Path(str(prefix) + "_diagnostics.png")
        plot_network_comparison(
            data,
            truth[local],
            prediction[local],
            trajectory_id,
            network_path,
            snapshot_steps=snapshot_steps,
            dt=dt,
            rho_max=rho_max,
            dpi=args.dpi,
            cmap=args.cmap,
        )
        plot_spacetime_comparison(
            data,
            truth[local],
            prediction[local],
            trajectory_id,
            spacetime_path,
            dt=dt,
            rho_max=rho_max,
            dpi=args.dpi,
            cmap=args.cmap,
            trajectory_index=int(trajectory_id),
        )
        plot_trajectory_diagnostics(
            data,
            truth[local],
            prediction[local],
            trajectory_id,
            diagnostics_path,
            dt=dt,
            rho_max=rho_max,
            cell_lengths=cell_lengths,
            dpi=args.dpi,
            trajectory_index=int(trajectory_id),
        )
        written_paths.extend(
            (network_path, spacetime_path, diagnostics_path)
        )

    mean_model_rmse = float(np.mean(metrics["model_rollout_rmse"]))
    mean_persistence_rmse = float(
        np.mean(metrics["persistence_rollout_rmse"])
    )
    summary = {
        "checkpoint": str(args.checkpoint),
        "data": str(data_path),
        "device": str(device),
        "num_test_trajectories": int(test_trajectory_ids.size),
        "rollout_steps": int(num_steps),
        "mean_model_rollout_rmse": mean_model_rmse,
        "mean_persistence_rollout_rmse": mean_persistence_rmse,
        "mean_rmse_improvement_fraction": (
            1.0 - mean_model_rmse / mean_persistence_rmse
            if mean_persistence_rmse > 0.0
            else float("nan")
        ),
        "selected_trajectory_ids": selected_trajectory_ids,
    }
    summary_json_path = output_dir / "summary.json"
    with summary_json_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    written_paths.append(summary_json_path)

    print(json.dumps(summary, indent=2, sort_keys=True))
    print("wrote:")
    for path in written_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
