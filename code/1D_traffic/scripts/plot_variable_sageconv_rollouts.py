#!/usr/bin/env python3
"""Evaluate variable-topology GNN checkpoints autoregressively.

This script is the variable-network counterpart to
``plot_sageconv_rollouts.py``. It reads the topology splits stored in the
dataset manifest, reconstructs the exact normalized features used in training,
and rolls a trained model forward from every test initial condition.

By default, all three test splits are evaluated:

* ``test_in_size``;
* ``test_large``;
* ``test_parameter_ood``.

Outputs are written to ``<checkpoint directory>/rollout_plots`` unless
``--output-dir`` is supplied. Because topology files have different node
counts, predictions are saved in one archive per topology. Aggregate metrics
and summary plots are also produced for each split. Every trajectory gets a
``trajectory_XXXX_comparison_diagnostics.png`` containing single-model curves
and scalar scores that can be compared directly across trained models.

Example:

    python scripts/plot_variable_sageconv_rollouts.py \
        --checkpoint runs/variable_networks/sageconv_200ep/checkpoint.pt \
        --device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

try:
    import torch
except ModuleNotFoundError as exc:
    raise SystemExit(
        "This script requires PyTorch and PyTorch Geometric. Install the "
        "dependencies required by train.py first."
    ) from exc

from plot_network_lwr_traffic_data import _format_axes
from plot_sageconv_rollouts import (
    compute_metrics,
    default_snapshot_steps,
    make_batched_edge_index,
    plot_network_comparison,
    plot_spacetime_comparison,
    plot_trajectory_diagnostics,
    validate_snapshot_steps,
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
    VariableTopology,
    apply_mass_projection,
    canonical_target_type,
    choose_device,
    load_variable_topology_dataset,
    model_from_checkpoint,
    predict_normalized_target,
    prepare_variable_features,
)


TEST_SPLITS = (
    "test_in_size",
    "test_large",
    "test_parameter_ood",
)


@dataclass
class TopologyRolloutResult:
    """Predictions, metrics, and plotting data for one topology."""

    split: str
    topology: VariableTopology
    plot_data: Dict[str, np.ndarray]
    truth: np.ndarray
    prediction: np.ndarray
    metrics: Dict[str, np.ndarray]


def load_checkpoint(path: Path) -> Dict[str, object]:
    """Load and validate a trusted variable-topology checkpoint."""
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("The checkpoint must contain a dictionary.")
    required = (
        "model_state_dict",
        "model_kwargs",
        "normalization",
        "feature_normalization",
        "split_topology_ids",
        "dataset_format",
        "target_type",
        "data_path",
    )
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise KeyError(f"Checkpoint is missing required fields: {missing}")
    if checkpoint["dataset_format"] != "variable_topology_lwr_v1":
        raise ValueError(
            "This script requires a variable-topology checkpoint; got "
            f"{checkpoint['dataset_format']!r}."
        )
    checkpoint["target_type"] = canonical_target_type(
        checkpoint["target_type"]
    )
    return checkpoint


def load_plot_data(path: Path) -> Dict[str, np.ndarray]:
    """Load one topology archive for visualization."""
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    required = (
        "rollout_states",
        "pos",
        "cell_lengths",
        "cell_road_id",
        "cell_local_index",
        "cell_road_coordinate",
        "road_names",
        "road_lengths",
        "junction_names",
        "junction_types",
        "junction_positions",
        "transfer_index",
        "transfer_type",
        "rho_max",
        "dt",
    )
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"{path} is missing plotting fields: {missing}")
    rollout = np.asarray(data["rollout_states"], dtype=np.float32)
    if rollout.ndim == 4 and rollout.shape[-1] == 1:
        rollout = rollout[..., 0]
    if rollout.ndim != 3:
        raise ValueError(
            f"{path}: rollout_states must have shape "
            "[trajectory, time, node]."
        )
    data["rollout_states"] = rollout
    return data


def validate_checkpoint_splits(
    checkpoint: Dict[str, object],
    topologies_by_split: Dict[str, List[VariableTopology]],
) -> None:
    """Ensure checkpoint topology IDs match the selected dataset."""
    checkpoint_splits = checkpoint["split_topology_ids"]
    if not isinstance(checkpoint_splits, dict):
        raise ValueError("Checkpoint split_topology_ids must be a dictionary.")
    for split, topologies in topologies_by_split.items():
        if split not in checkpoint_splits:
            raise KeyError(f"Checkpoint is missing split {split!r}.")
        expected = [topology.topology_id for topology in topologies]
        actual = [int(value) for value in checkpoint_splits[split]]
        if expected != actual:
            raise ValueError(
                f"Checkpoint topology IDs for {split!r} do not match "
                "the dataset manifest."
            )


@torch.inference_mode()
def predict_topology_rollouts(
    model: torch.nn.Module,
    topology: VariableTopology,
    *,
    num_steps: int,
    normalization: Normalization,
    physics_cfg: PhysicsInputConfig,
    target_type: str,
    mass_projection: bool,
    device: torch.device,
) -> np.ndarray:
    """Predict all trajectories for one topology in parallel."""
    target_type = canonical_target_type(target_type)
    if topology.static_features is None:
        raise RuntimeError("Topology static features have not been prepared.")
    if topology.node_conditioning is None:
        raise RuntimeError("Topology conditioning has not been prepared.")
    initial = topology.rollout_states[:, 0]
    batch_size, num_nodes, channels = initial.shape
    if channels != 1:
        raise ValueError("Expected one scalar density channel.")

    current = torch.from_numpy(initial).to(
        device=device,
        dtype=torch.float32,
    )
    static = topology.static_features.to(
        device=device,
        dtype=torch.float32,
    )
    static = static.unsqueeze(0).expand(batch_size, -1, -1)
    conditioning = topology.node_conditioning.to(
        device=device,
        dtype=torch.float32,
    )
    edge_index = make_batched_edge_index(
        topology.edge_index,
        batch_size=batch_size,
        num_nodes=num_nodes,
        device=device,
    )
    if topology.trajectory_edge_attr is None:
        edge_attributes = topology.edge_attr.unsqueeze(0).expand(
            batch_size, -1, -1
        )
    else:
        edge_attributes = torch.from_numpy(
            topology.trajectory_edge_attr
        )
    edge_attr_channels = int(topology.edge_attr.shape[1])
    if edge_attr_channels == 0:
        edge_attributes = torch.empty(
            (batch_size * topology.edge_index.shape[1], 0),
            device=device,
            dtype=torch.float32,
        )
    else:
        edge_attributes = edge_attributes.to(
            device=device, dtype=torch.float32
        ).reshape(-1, edge_attr_channels)
    cell_lengths = torch.from_numpy(
        topology.cell_lengths_array
    ).to(device=device, dtype=torch.float32)
    cell_lengths = cell_lengths.view(1, num_nodes, 1).expand(
        batch_size, -1, -1
    ).reshape(-1, 1)
    v_max_nodes = torch.from_numpy(
        topology.raw_node_conditioning[..., 0:1]
    ).to(device=device, dtype=torch.float32).reshape(-1, 1)
    rho_max_nodes = torch.full_like(cell_lengths, topology.rho_max)
    dt_nodes = torch.full_like(cell_lengths, topology.dt)
    graph_index = torch.arange(
        batch_size,
        device=device,
        dtype=torch.long,
    ).repeat_interleave(num_nodes)
    advection_metadata = None
    if physics_cfg.enabled and physics_cfg.include_adv:
        advection_metadata = build_lwr_advection_metadata(
            edge_index, edge_attributes
        )

    predicted_states = [current]
    for step in range(num_steps):
        normalized_density = (
            current - normalization.density_mean
        ) / normalization.density_std
        base_features = torch.cat(
            (normalized_density, static, conditioning),
            dim=-1,
        ).reshape(batch_size * num_nodes, -1)
        features = build_physics_augmented_inputs_1d(
            x_base=base_features,
            x_state=current.reshape(batch_size * num_nodes, 1),
            edge_index=edge_index,
            edge_attr=edge_attributes,
            cell_length=cell_lengths,
            dt_node=dt_nodes,
            v_max=v_max_nodes,
            rho_max=rho_max_nodes,
            physics_cfg=physics_cfg,
            step_k=step,
            metadata=advection_metadata,
            validate_inputs=advection_metadata is None,
        )
        normalized_target = predict_normalized_target(
            model,
            features,
            edge_index,
            edge_attr=edge_attributes,
            cell_length=cell_lengths,
        )
        normalized_target = apply_mass_projection(
            normalized_target,
            normalization=normalization,
            cell_length=cell_lengths,
            graph_index=graph_index,
            num_graphs=batch_size,
            enabled=mass_projection,
        ).reshape(batch_size, num_nodes, 1)
        raw_target = (
            normalized_target * normalization.target_std
            + normalization.target_mean
        )
        if target_type == "rate":
            current = current + topology.dt * raw_target
        else:
            current = current + raw_target
        predicted_states.append(current)

    prediction = torch.stack(predicted_states, dim=1)
    return prediction.squeeze(-1).cpu().numpy().astype(
        np.float32,
        copy=False,
    )


def representative_keys(
    results: Sequence[TopologyRolloutResult],
) -> List[Tuple[int, int]]:
    """Choose distinct best, median, and worst trajectory keys."""
    candidates: List[Tuple[float, int, int]] = []
    for result in results:
        rollout_rmse = result.metrics["model_rollout_rmse"]
        for trajectory_id, value in enumerate(rollout_rmse):
            candidates.append(
                (
                    float(value),
                    result.topology.topology_id,
                    trajectory_id,
                )
            )
    candidates.sort()
    positions = (0, len(candidates) // 2, len(candidates) - 1)
    return list(
        dict.fromkeys(
            (
                candidates[position][1],
                candidates[position][2],
            )
            for position in positions
        )
    )


def compute_comparison_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    base_metrics: Dict[str, np.ndarray],
    *,
    rho_max: float,
) -> Dict[str, np.ndarray]:
    """Compute model-comparison metrics, excluding the shared initial state."""
    error = prediction - truth
    persistence_error = base_metrics["persistence"] - truth
    future_error = error[:, 1:]
    future_persistence_error = persistence_error[:, 1:]
    future_truth = truth[:, 1:]

    error_squared_sum = np.sum(
        future_error.astype(np.float64) ** 2,
        axis=(1, 2),
    )
    persistence_squared_sum = np.sum(
        future_persistence_error.astype(np.float64) ** 2,
        axis=(1, 2),
    )
    truth_squared_sum = np.sum(
        future_truth.astype(np.float64) ** 2,
        axis=(1, 2),
    )
    truth_centered = future_truth - np.mean(
        future_truth,
        axis=(1, 2),
        keepdims=True,
    )
    truth_variation = np.sum(
        truth_centered.astype(np.float64) ** 2,
        axis=(1, 2),
    )
    epsilon = np.finfo(np.float64).eps
    relative_l2 = np.sqrt(
        error_squared_sum / np.maximum(truth_squared_sum, epsilon)
    )
    persistence_relative_l2 = np.sqrt(
        persistence_squared_sum
        / np.maximum(truth_squared_sum, epsilon)
    )
    r_squared = 1.0 - (
        error_squared_sum / np.maximum(truth_variation, epsilon)
    )

    correlation = np.empty(truth.shape[0], dtype=np.float64)
    for trajectory in range(truth.shape[0]):
        true_values = future_truth[trajectory].reshape(-1)
        predicted_values = prediction[trajectory, 1:].reshape(-1)
        if (
            float(np.std(true_values)) <= epsilon
            or float(np.std(predicted_values)) <= epsilon
        ):
            correlation[trajectory] = np.nan
        else:
            correlation[trajectory] = np.corrcoef(
                true_values,
                predicted_values,
            )[0, 1]

    initial_mass = np.maximum(
        np.abs(base_metrics["truth_mass"][:, 0]),
        epsilon,
    )
    max_relative_mass_error = np.max(
        np.abs(base_metrics["mass_error"]),
        axis=1,
    ) / initial_mass
    relative_mass_error_by_time = (
        base_metrics["mass_error"] / initial_mass[:, None]
    )
    outside_bounds_by_time = np.mean(
        (prediction < 0.0)
        | (prediction > rho_max),
        axis=2,
    )
    return {
        "model_mae_by_time": np.mean(np.abs(error), axis=2),
        "persistence_mae_by_time": np.mean(
            np.abs(persistence_error),
            axis=2,
        ),
        "model_bias_by_time": np.mean(error, axis=2),
        "model_rollout_mae": np.mean(
            np.abs(future_error),
            axis=(1, 2),
        ),
        "persistence_rollout_mae": np.mean(
            np.abs(future_persistence_error),
            axis=(1, 2),
        ),
        "mean_signed_error": np.mean(
            future_error,
            axis=(1, 2),
        ),
        "relative_l2_error": relative_l2,
        "persistence_relative_l2_error": persistence_relative_l2,
        "r_squared": r_squared,
        "pearson_correlation": correlation,
        "max_relative_mass_error": max_relative_mass_error,
        "relative_mass_error_by_time": relative_mass_error_by_time,
        "outside_bounds_by_time": outside_bounds_by_time,
    }


def concatenate_metric(
    results: Sequence[TopologyRolloutResult],
    name: str,
) -> np.ndarray:
    """Concatenate a trajectory-indexed metric across topologies."""
    return np.concatenate(
        [result.metrics[name] for result in results],
        axis=0,
    )


def split_overall_scores(
    results: Sequence[TopologyRolloutResult],
) -> Dict[str, float]:
    """Compute globally weighted scalar scores for one test split."""
    truth = np.concatenate(
        [result.truth[:, 1:].reshape(-1) for result in results]
    ).astype(np.float64, copy=False)
    prediction = np.concatenate(
        [result.prediction[:, 1:].reshape(-1) for result in results]
    ).astype(np.float64, copy=False)
    persistence = np.concatenate(
        [
            result.metrics["persistence"][:, 1:].reshape(-1)
            for result in results
        ]
    ).astype(np.float64, copy=False)
    error = prediction - truth
    persistence_error = persistence - truth
    epsilon = np.finfo(np.float64).eps
    truth_centered = truth - float(np.mean(truth))
    prediction_std = float(np.std(prediction))
    truth_std = float(np.std(truth))
    correlation = (
        float(np.corrcoef(truth, prediction)[0, 1])
        if prediction_std > epsilon and truth_std > epsilon
        else float("nan")
    )
    return {
        "overall_mae": float(np.mean(np.abs(error))),
        "overall_rmse": float(np.sqrt(np.mean(error**2))),
        "persistence_mae": float(
            np.mean(np.abs(persistence_error))
        ),
        "persistence_rmse": float(
            np.sqrt(np.mean(persistence_error**2))
        ),
        "final_step_mae": float(
            np.mean(
                concatenate_metric(
                    results,
                    "model_mae_by_time",
                )[:, -1]
            )
        ),
        "mean_signed_error": float(np.mean(error)),
        "relative_l2_error": float(
            np.sqrt(
                np.sum(error**2)
                / max(float(np.sum(truth**2)), epsilon)
            )
        ),
        "r_squared": float(
            1.0
            - np.sum(error**2)
            / max(float(np.sum(truth_centered**2)), epsilon)
        ),
        "pearson_correlation": correlation,
        "mean_max_relative_mass_error": float(
            np.mean(
                concatenate_metric(
                    results,
                    "max_relative_mass_error",
                )
            )
        ),
        "mean_outside_bounds_fraction": float(
            np.mean(
                concatenate_metric(
                    results,
                    "outside_bounds_fraction",
                )
            )
        ),
    }


def plot_trajectory_comparison_diagnostics(
    result: TopologyRolloutResult,
    trajectory_id: int,
    output_path: Path,
    *,
    model_label: str,
    dpi: int,
) -> None:
    """Plot single-model metrics for one fixed rollout trajectory."""
    metrics = result.metrics
    steps = np.arange(result.truth.shape[1])
    model_mae = metrics["model_mae_by_time"][trajectory_id]
    model_rmse = metrics["model_rmse_by_time"][trajectory_id]
    model_bias = metrics["model_bias_by_time"][trajectory_id]
    relative_mass_error = metrics["relative_mass_error_by_time"][
        trajectory_id
    ]
    outside_bounds = metrics["outside_bounds_by_time"][trajectory_id]

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(14.0, 8.2),
        constrained_layout=True,
    )
    (
        ax_mae,
        ax_rmse,
        ax_bias,
        ax_mass,
        ax_bounds,
        ax_scores,
    ) = axes.ravel()

    for axis, values, title, ylabel, color in (
        (
            ax_mae,
            model_mae,
            "Mean absolute error",
            "Node MAE",
            "tab:blue",
        ),
        (
            ax_rmse,
            model_rmse,
            "Root mean squared error",
            "Node RMSE",
            "tab:green",
        ),
        (
            ax_bias,
            model_bias,
            "Signed prediction bias",
            r"Mean $\hat\rho-\rho$",
            "tab:orange",
        ),
        (
            ax_mass,
            relative_mass_error,
            "Relative mass error",
            r"$(\hat M-M)/|M_0|$",
            "tab:red",
        ),
        (
            ax_bounds,
            outside_bounds,
            "Physical-bound violations",
            "Fraction of nodes",
            "tab:purple",
        ),
    ):
        axis.plot(
            steps,
            values,
            color=color,
            label=model_label,
        )
        axis.set_title(title)
        axis.set_xlabel("Rollout step")
        axis.set_ylabel(ylabel)
        axis.legend()
        _format_axes(axis)

    ax_bias.axhline(0.0, color="0.25", linestyle="--", linewidth=1.0)
    ax_mass.axhline(0.0, color="0.25", linestyle="--", linewidth=1.0)
    ax_bounds.set_ylim(
        -0.01,
        max(0.05, 1.05 * float(np.max(outside_bounds))),
    )

    score_text = (
        f"Model: {model_label}\n\n"
        "overall rollout MAE\n"
        f"  {metrics['model_rollout_mae'][trajectory_id]:.6g}\n\n"
        "overall rollout RMSE\n"
        f"  {metrics['model_rollout_rmse'][trajectory_id]:.6g}\n\n"
        "final-step MAE\n"
        f"  {model_mae[-1]:.6g}\n\n"
        r"relative $L_2$ error"
        "\n"
        f"  {metrics['relative_l2_error'][trajectory_id]:.6g}\n\n"
        r"$R^2$"
        "\n"
        f"  {metrics['r_squared'][trajectory_id]:.6g}\n\n"
        "Pearson correlation\n"
        f"  {metrics['pearson_correlation'][trajectory_id]:.6g}\n\n"
        "mean signed error\n"
        f"  {metrics['mean_signed_error'][trajectory_id]:.6g}\n\n"
        "max relative mass error\n"
        f"  {metrics['max_relative_mass_error'][trajectory_id]:.2%}\n\n"
        "outside-bounds fraction\n"
        f"  {metrics['outside_bounds_fraction'][trajectory_id]:.2%}"
    )
    ax_scores.axis("off")
    ax_scores.set_title("Scalar scores")
    ax_scores.text(
        0.05,
        0.95,
        score_text,
        transform=ax_scores.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "edgecolor": "0.7",
            "alpha": 0.95,
        },
    )

    readable_split = result.split.replace("_", " ")
    trajectory_directions = result.plot_data.get(
        "trajectory_flow_direction"
    )
    direction_suffix = ""
    if trajectory_directions is not None:
        direction_suffix = (
            ", canonical direction"
            if float(trajectory_directions[trajectory_id]) >= 0.0
            else ", reversed direction"
        )
    figure.suptitle(
        "Model-comparison diagnostics: "
        f"{readable_split}, topology "
        f"{result.topology.topology_id:04d}, trajectory "
        f"{trajectory_id:04d}{direction_suffix}",
        fontsize=14,
    )
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def plot_split_summary(
    split: str,
    results: Sequence[TopologyRolloutResult],
    output_path: Path,
    *,
    selected_keys: Sequence[Tuple[int, int]],
    dpi: int,
) -> None:
    """Plot aggregate rollout diagnostics for one variable-network split."""
    model_time = concatenate_metric(results, "model_rmse_by_time")
    persistence_time = concatenate_metric(
        results,
        "persistence_rmse_by_time",
    )
    mass_error = concatenate_metric(results, "mass_error")
    model_rmse = concatenate_metric(results, "model_rollout_rmse")
    persistence_rmse = concatenate_metric(
        results,
        "persistence_rollout_rmse",
    )
    steps = np.arange(model_time.shape[1])

    labels: List[str] = []
    keys: List[Tuple[int, int]] = []
    for result in results:
        for trajectory_id in range(result.truth.shape[0]):
            labels.append(
                f"T{result.topology.topology_id:04d}/"
                f"r{trajectory_id:02d}"
            )
            keys.append((result.topology.topology_id, trajectory_id))

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(11.0, 8.0),
        constrained_layout=True,
    )
    ax_time, ax_baseline, ax_mass, ax_rank = axes.ravel()

    for values, color, label in (
        (model_time, "tab:blue", "SAGEConv"),
        (persistence_time, "0.45", "Persistence"),
    ):
        median = np.median(values, axis=0)
        lower, upper = np.quantile(values, (0.25, 0.75), axis=0)
        ax_time.plot(steps, median, color=color, label=label)
        ax_time.fill_between(
            steps,
            lower,
            upper,
            color=color,
            alpha=0.18,
        )
    ax_time.set_title("Autoregressive rollout error")
    ax_time.set_xlabel("Rollout step")
    ax_time.set_ylabel("Node RMSE")
    ax_time.legend()
    _format_axes(ax_time)

    limit = 1.05 * max(
        float(np.max(model_rmse)),
        float(np.max(persistence_rmse)),
        1.0e-8,
    )
    ax_baseline.scatter(
        persistence_rmse,
        model_rmse,
        color="tab:blue",
        alpha=0.78,
    )
    ax_baseline.plot([0.0, limit], [0.0, limit], "--", color="0.4")
    ax_baseline.set_xlim(0.0, limit)
    ax_baseline.set_ylim(0.0, limit)
    ax_baseline.set_aspect("equal", adjustable="box")
    ax_baseline.set_title("Model versus persistence")
    ax_baseline.set_xlabel("Persistence rollout RMSE")
    ax_baseline.set_ylabel("SAGEConv rollout RMSE")
    _format_axes(ax_baseline)

    absolute_mass_error = np.abs(mass_error)
    ax_mass.plot(
        steps,
        np.median(absolute_mass_error, axis=0),
        color="tab:purple",
        label="Median",
    )
    ax_mass.fill_between(
        steps,
        np.quantile(absolute_mass_error, 0.25, axis=0),
        np.quantile(absolute_mass_error, 0.75, axis=0),
        color="tab:purple",
        alpha=0.18,
        label="Interquartile range",
    )
    ax_mass.plot(
        steps,
        np.max(absolute_mass_error, axis=0),
        color="tab:red",
        linewidth=1.0,
        label="Maximum",
    )
    ax_mass.set_title("Predicted total-mass error")
    ax_mass.set_xlabel("Rollout step")
    ax_mass.set_ylabel(r"$|\hat M-M|$")
    ax_mass.legend(fontsize=8)
    _format_axes(ax_mass)

    order = np.argsort(model_rmse)
    selected = set(selected_keys)
    colors = [
        "tab:orange" if keys[index] in selected else "tab:blue"
        for index in order
    ]
    ax_rank.bar(
        np.arange(order.size),
        model_rmse[order],
        color=colors,
        alpha=0.82,
    )
    ax_rank.set_title("Trajectories ranked by rollout RMSE")
    ax_rank.set_xlabel("Rank")
    ax_rank.set_ylabel("Rollout RMSE")
    if order.size <= 30:
        ax_rank.set_xticks(np.arange(order.size))
        ax_rank.set_xticklabels(
            [labels[index] for index in order],
            rotation=75,
            fontsize=6,
        )
    _format_axes(ax_rank)

    readable_split = split.replace("_", " ")
    figure.suptitle(
        f"SAGEConv variable-network rollouts: {readable_split}",
        fontsize=14,
    )
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def metric_rows(
    results: Sequence[TopologyRolloutResult],
    *,
    model_label: str,
) -> List[Dict[str, object]]:
    """Create one serializable metric row per trajectory."""
    rows: List[Dict[str, object]] = []
    for result in results:
        metrics = result.metrics
        num_roads = int(
            np.asarray(result.plot_data["road_lengths"]).size
        )
        pair_ids = result.plot_data.get("trajectory_pair_id")
        flow_directions = result.plot_data.get(
            "trajectory_flow_direction"
        )
        for trajectory_id in range(result.truth.shape[0]):
            model_rmse = float(
                metrics["model_rollout_rmse"][trajectory_id]
            )
            persistence_rmse = float(
                metrics["persistence_rollout_rmse"][trajectory_id]
            )
            improvement = (
                1.0 - model_rmse / persistence_rmse
                if persistence_rmse > 0.0
                else float("nan")
            )
            rows.append(
                {
                    "model_label": model_label,
                    "split": result.split,
                    "topology_id": result.topology.topology_id,
                    "trajectory_id": trajectory_id,
                    "trajectory_pair_id": (
                        int(pair_ids[trajectory_id])
                        if pair_ids is not None
                        else -1
                    ),
                    "flow_direction": (
                        float(flow_directions[trajectory_id])
                        if flow_directions is not None
                        else float("nan")
                    ),
                    "num_roads": num_roads,
                    "num_nodes": result.topology.num_nodes,
                    "dt": result.topology.dt,
                    "rollout_steps": result.truth.shape[1] - 1,
                    "model_rollout_mae": float(
                        metrics["model_rollout_mae"][trajectory_id]
                    ),
                    "model_rollout_rmse": model_rmse,
                    "persistence_rollout_mae": float(
                        metrics["persistence_rollout_mae"][
                            trajectory_id
                        ]
                    ),
                    "persistence_rollout_rmse": persistence_rmse,
                    "rmse_improvement_fraction": improvement,
                    "final_step_mae": float(
                        metrics["model_mae_by_time"][
                            trajectory_id,
                            -1,
                        ]
                    ),
                    "final_step_rmse": float(
                        metrics["model_rmse_by_time"][
                            trajectory_id,
                            -1,
                        ]
                    ),
                    "max_abs_mass_error": float(
                        np.max(
                            np.abs(
                                metrics["mass_error"][trajectory_id]
                            )
                        )
                    ),
                    "max_relative_mass_error": float(
                        metrics["max_relative_mass_error"][
                            trajectory_id
                        ]
                    ),
                    "mean_signed_error": float(
                        metrics["mean_signed_error"][trajectory_id]
                    ),
                    "relative_l2_error": float(
                        metrics["relative_l2_error"][trajectory_id]
                    ),
                    "r_squared": float(
                        metrics["r_squared"][trajectory_id]
                    ),
                    "pearson_correlation": float(
                        metrics["pearson_correlation"][trajectory_id]
                    ),
                    "outside_bounds_fraction": float(
                        metrics["outside_bounds_fraction"][trajectory_id]
                    ),
                }
            )
    return rows


def write_metrics_csv(
    path: Path,
    rows: Sequence[Dict[str, object]],
) -> None:
    """Write aggregate trajectory metrics for all selected test splits."""
    if not rows:
        raise ValueError("No rollout metric rows were produced.")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_split(
    split: str,
    results: Sequence[TopologyRolloutResult],
    selected_keys: Sequence[Tuple[int, int]],
) -> Dict[str, object]:
    """Build the JSON summary for one split."""
    model_rmse = concatenate_metric(results, "model_rollout_rmse")
    persistence_rmse = concatenate_metric(
        results,
        "persistence_rollout_rmse",
    )
    outside = concatenate_metric(results, "outside_bounds_fraction")
    mass_error = concatenate_metric(results, "mass_error")
    mean_model = float(np.mean(model_rmse))
    mean_persistence = float(np.mean(persistence_rmse))
    summary = {
        "split": split,
        "topology_ids": [
            result.topology.topology_id for result in results
        ],
        "num_topologies": len(results),
        "num_trajectories": int(model_rmse.size),
        "rollout_steps": int(results[0].truth.shape[1] - 1),
        "mean_model_rollout_rmse": mean_model,
        "median_model_rollout_rmse": float(np.median(model_rmse)),
        "mean_persistence_rollout_rmse": mean_persistence,
        "mean_rmse_improvement_fraction": (
            1.0 - mean_model / mean_persistence
            if mean_persistence > 0.0
            else float("nan")
        ),
        "maximum_absolute_mass_error": float(
            np.max(np.abs(mass_error))
        ),
        "mean_outside_bounds_fraction": float(np.mean(outside)),
        "selected_representative_trajectories": [
            {
                "topology_id": topology_id,
                "trajectory_id": trajectory_id,
            }
            for topology_id, trajectory_id in selected_keys
        ],
    }
    summary.update(split_overall_scores(results))
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description=(
            "Create autoregressive rollout predictions and plots for a "
            "variable-topology SAGEConv checkpoint."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Dataset directory override; defaults to checkpoint data_path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to CHECKPOINT_DIR/rollout_plots.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=TEST_SPLITS,
        default=list(TEST_SPLITS),
        help="Test splits to evaluate; defaults to all three.",
    )
    parser.add_argument(
        "--topologies",
        nargs="+",
        type=int,
        default=None,
        help="Optional topology IDs to evaluate within the selected splits.",
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=None,
        help="Autoregressive steps; defaults to each full stored rollout.",
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
    parser.add_argument(
        "--all-trajectories",
        action="store_true",
        help="Create individual figures for every evaluated trajectory.",
    )
    parser.add_argument(
        "--snapshot-steps",
        nargs="+",
        type=int,
        default=None,
        help="Saved-state indices for network comparison figures.",
    )
    parser.add_argument(
        "--network-time-axis",
        choices=("rows", "columns"),
        default="rows",
        help=(
            "Arrange snapshot times as rows (default) or columns in "
            "trajectory network figures."
        ),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument(
        "--model-label",
        default=None,
        help=(
            "Label used in per-trajectory comparison diagnostics; "
            "defaults to the checkpoint directory name."
        ),
    )
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--cmap", default="viridis")
    parser.add_argument(
        "--max-spacetime-roads",
        type=int,
        default=6,
        help="Maximum busiest roads shown in each spacetime figure.",
    )
    parser.add_argument(
        "--max-diagnostic-junctions",
        type=int,
        default=6,
        help="Maximum busiest junctions shown in each diagnostics figure.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run variable-topology rollout evaluation and plotting."""
    args = parse_args(argv)
    if args.rollout_steps is not None and args.rollout_steps <= 0:
        raise ValueError("--rollout-steps must be positive.")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive.")
    if args.max_spacetime_roads <= 0:
        raise ValueError("--max-spacetime-roads must be positive.")
    if args.max_diagnostic_junctions <= 0:
        raise ValueError("--max-diagnostic-junctions must be positive.")
    selected_splits = list(dict.fromkeys(args.splits))

    checkpoint = load_checkpoint(args.checkpoint)
    data_path = (
        args.data
        if args.data is not None
        else Path(str(checkpoint["data_path"]))
    )
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else args.checkpoint.parent / "rollout_plots"
    )
    model_label = (
        args.model_label
        if args.model_label is not None
        else args.checkpoint.parent.name
    )
    topologies_by_split, _ = load_variable_topology_dataset(data_path)
    validate_checkpoint_splits(checkpoint, topologies_by_split)

    if args.topologies is not None:
        requested_ids = set(int(value) for value in args.topologies)
        available_ids = {
            topology.topology_id
            for split in selected_splits
            for topology in topologies_by_split[split]
        }
        invalid = sorted(requested_ids - available_ids)
        if invalid:
            raise ValueError(
                f"Topology IDs {invalid} are not in the selected splits."
            )
        for split in selected_splits:
            topologies_by_split[split] = [
                topology
                for topology in topologies_by_split[split]
                if topology.topology_id in requested_ids
            ]
        empty_splits = [
            split
            for split in selected_splits
            if not topologies_by_split[split]
        ]
        if empty_splits:
            raise ValueError(
                "Topology filtering left split(s) empty: "
                + ", ".join(empty_splits)
            )

    feature_normalization = dict(checkpoint["feature_normalization"])
    selected_topologies = {
        split: topologies_by_split[split] for split in selected_splits
    }
    prepare_variable_features(
        selected_topologies,
        feature_normalization,
    )

    model = model_from_checkpoint(checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    device = choose_device(args.device)
    model = model.to(device)
    model.eval()
    normalization = Normalization(**dict(checkpoint["normalization"]))
    physics_cfg = resolve_physics_input_cfg(checkpoint)
    target_type = canonical_target_type(checkpoint["target_type"])
    checkpoint_mass_projection = checkpoint.get("mass_projection", False)
    if not isinstance(checkpoint_mass_projection, bool):
        raise ValueError("Checkpoint mass_projection must be boolean.")
    mass_projection = (
        checkpoint_mass_projection
        if args.mass_projection is None
        else args.mass_projection
    )
    expected_features = int(checkpoint["model_kwargs"]["in_channels"])

    output_dir.mkdir(parents=True, exist_ok=True)
    results_by_split: Dict[str, List[TopologyRolloutResult]] = {}
    written_paths: List[Path] = []
    for split in selected_splits:
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        legacy_comparison_path = split_dir / "comparison_diagnostics.png"
        if legacy_comparison_path.is_file():
            legacy_comparison_path.unlink()
        results_by_split[split] = []
        for topology in selected_topologies[split]:
            if topology.static_features is None:
                raise RuntimeError("Static topology features are unavailable.")
            if topology.node_conditioning is None:
                raise RuntimeError("Node conditioning is unavailable.")
            feature_count = (
                1
                + int(topology.static_features.shape[1])
                + int(topology.node_conditioning.shape[2])
                + count_physics_input_channels(physics_cfg)
            )
            if feature_count != expected_features:
                raise ValueError(
                    f"Topology {topology.topology_id} has {feature_count} "
                    f"features; checkpoint expects {expected_features}."
                )
            available_steps = topology.window
            num_steps = (
                available_steps
                if args.rollout_steps is None
                else args.rollout_steps
            )
            if num_steps > available_steps:
                raise ValueError(
                    f"--rollout-steps={num_steps} exceeds topology "
                    f"{topology.topology_id}'s {available_steps} steps."
                )

            print(
                f"[{split}] topology {topology.topology_id:04d}: "
                f"trajectories={topology.num_trajectories}, "
                f"nodes={topology.num_nodes}, steps={num_steps}"
            )
            plot_data = load_plot_data(topology.path)
            truth = np.asarray(
                plot_data["rollout_states"][:, : num_steps + 1],
                dtype=np.float32,
            )
            prediction = predict_topology_rollouts(
                model,
                topology,
                num_steps=num_steps,
                normalization=normalization,
                physics_cfg=physics_cfg,
                target_type=target_type,
                mass_projection=mass_projection,
                device=device,
            )
            cell_lengths = np.asarray(
                plot_data["cell_lengths"],
                dtype=np.float64,
            )
            rho_max = float(np.asarray(plot_data["rho_max"]).item())
            metrics = compute_metrics(
                truth,
                prediction,
                cell_lengths=cell_lengths,
                rho_max=rho_max,
            )
            metrics.update(
                compute_comparison_metrics(
                    truth,
                    prediction,
                    metrics,
                    rho_max=rho_max,
                )
            )
            result = TopologyRolloutResult(
                split=split,
                topology=topology,
                plot_data=plot_data,
                truth=truth,
                prediction=prediction,
                metrics=metrics,
            )
            results_by_split[split].append(result)

            topology_dir = split_dir / (
                f"topology_{topology.topology_id:04d}"
            )
            topology_dir.mkdir(parents=True, exist_ok=True)
            prediction_path = topology_dir / "rollout_predictions.npz"
            np.savez_compressed(
                prediction_path,
                topology_id=np.array(
                    topology.topology_id,
                    dtype=np.int64,
                ),
                model_label=np.array(model_label),
                trajectory_ids=np.arange(
                    topology.num_trajectories,
                    dtype=np.int64,
                ),
                trajectory_pair_id=np.asarray(
                    plot_data.get(
                        "trajectory_pair_id",
                        np.full(
                            topology.num_trajectories,
                            -1,
                            dtype=np.int64,
                        ),
                    ),
                    dtype=np.int64,
                ),
                trajectory_flow_direction=np.asarray(
                    plot_data.get(
                        "trajectory_flow_direction",
                        np.full(
                            topology.num_trajectories,
                            np.nan,
                            dtype=np.float64,
                        ),
                    ),
                    dtype=np.float64,
                ),
                ground_truth=truth,
                prediction=prediction,
                persistence=metrics["persistence"],
                model_rmse_by_time=metrics["model_rmse_by_time"],
                persistence_rmse_by_time=(
                    metrics["persistence_rmse_by_time"]
                ),
                model_mae_by_time=metrics["model_mae_by_time"],
                persistence_mae_by_time=(
                    metrics["persistence_mae_by_time"]
                ),
                model_bias_by_time=metrics["model_bias_by_time"],
                model_rollout_mae=metrics["model_rollout_mae"],
                persistence_rollout_mae=(
                    metrics["persistence_rollout_mae"]
                ),
                relative_l2_error=metrics["relative_l2_error"],
                persistence_relative_l2_error=(
                    metrics["persistence_relative_l2_error"]
                ),
                r_squared=metrics["r_squared"],
                pearson_correlation=metrics[
                    "pearson_correlation"
                ],
                mass_error=metrics["mass_error"],
                max_relative_mass_error=(
                    metrics["max_relative_mass_error"]
                ),
                relative_mass_error_by_time=(
                    metrics["relative_mass_error_by_time"]
                ),
                outside_bounds_by_time=(
                    metrics["outside_bounds_by_time"]
                ),
                dt=np.array(topology.dt, dtype=np.float64),
            )
            written_paths.append(prediction_path)
            for trajectory_id in range(topology.num_trajectories):
                comparison_path = topology_dir / (
                    f"trajectory_{trajectory_id:04d}"
                    "_comparison_diagnostics.png"
                )
                plot_trajectory_comparison_diagnostics(
                    result,
                    trajectory_id,
                    comparison_path,
                    model_label=model_label,
                    dpi=args.dpi,
                )
                written_paths.append(comparison_path)

    all_rows: List[Dict[str, object]] = []
    split_summaries: Dict[str, Dict[str, object]] = {}
    snapshot_steps_by_split: Dict[str, List[int]] = {}
    for split, results in results_by_split.items():
        all_rows.extend(
            metric_rows(
                results,
                model_label=model_label,
            )
        )
        selected_keys = (
            [
                (result.topology.topology_id, trajectory_id)
                for result in results
                for trajectory_id in range(result.truth.shape[0])
            ]
            if args.all_trajectories
            else representative_keys(results)
        )
        num_times = results[0].truth.shape[1]
        snapshot_steps = (
            default_snapshot_steps(num_times)
            if args.snapshot_steps is None
            else validate_snapshot_steps(
                args.snapshot_steps,
                num_times,
            )
        )
        snapshot_steps_by_split[split] = snapshot_steps
        summary_plot_path = output_dir / split / "summary.png"
        plot_split_summary(
            split,
            results,
            summary_plot_path,
            selected_keys=selected_keys,
            dpi=args.dpi,
        )
        written_paths.append(summary_plot_path)

        result_by_id = {
            result.topology.topology_id: result for result in results
        }
        for topology_id, trajectory_id in selected_keys:
            result = result_by_id[topology_id]
            topology_dir = (
                output_dir
                / split
                / f"topology_{topology_id:04d}"
            )
            prefix = topology_dir / (
                f"trajectory_{trajectory_id:04d}"
            )
            network_path = Path(str(prefix) + "_network.png")
            spacetime_path = Path(str(prefix) + "_spacetime.png")
            diagnostics_path = Path(str(prefix) + "_diagnostics.png")
            trajectory_directions = result.plot_data.get(
                "trajectory_flow_direction"
            )
            flow_direction = (
                float(trajectory_directions[trajectory_id])
                if trajectory_directions is not None
                else 1.0
            )
            direction_label = (
                ", canonical direction"
                if flow_direction >= 0.0
                else ", reversed direction"
            )
            trajectory_label = (
                f"topology {topology_id:04d}, trajectory "
                f"{trajectory_id:04d}{direction_label}"
            )
            rho_max = float(
                np.asarray(result.plot_data["rho_max"]).item()
            )
            plot_network_comparison(
                result.plot_data,
                result.truth[trajectory_id],
                result.prediction[trajectory_id],
                trajectory_label,
                network_path,
                snapshot_steps=snapshot_steps,
                dt=result.topology.dt,
                rho_max=rho_max,
                dpi=args.dpi,
                cmap=args.cmap,
                flow_direction=flow_direction,
                time_axis=args.network_time_axis,
            )
            plot_spacetime_comparison(
                result.plot_data,
                result.truth[trajectory_id],
                result.prediction[trajectory_id],
                trajectory_label,
                spacetime_path,
                dt=result.topology.dt,
                rho_max=rho_max,
                dpi=args.dpi,
                cmap=args.cmap,
                trajectory_index=trajectory_id,
                max_roads=args.max_spacetime_roads,
            )
            plot_trajectory_diagnostics(
                result.plot_data,
                result.truth[trajectory_id],
                result.prediction[trajectory_id],
                trajectory_label,
                diagnostics_path,
                dt=result.topology.dt,
                rho_max=rho_max,
                cell_lengths=np.asarray(
                    result.plot_data["cell_lengths"],
                    dtype=np.float64,
                ),
                dpi=args.dpi,
                trajectory_index=trajectory_id,
                max_junctions=args.max_diagnostic_junctions,
            )
            written_paths.extend(
                (
                    network_path,
                    spacetime_path,
                    diagnostics_path,
                )
            )

        split_summaries[split] = summarize_split(
            split,
            results,
            selected_keys,
        )

    metrics_path = output_dir / "rollout_metrics.csv"
    write_metrics_csv(metrics_path, all_rows)
    written_paths.append(metrics_path)
    summary = {
        "checkpoint": str(args.checkpoint),
        "model_label": model_label,
        "data": str(data_path),
        "device": str(device),
        "splits": split_summaries,
        "snapshot_steps": snapshot_steps_by_split,
        "network_time_axis": args.network_time_axis,
    }
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    written_paths.append(summary_path)

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {len(written_paths)} files under {output_dir}")


if __name__ == "__main__":
    main()
