#!/usr/bin/env python3
"""Train a GNN surrogate for 1D periodic advection.

Typical usage from repo root (with `amr-gnn-pt27` activated):

    python scripts/train_gnn_advection.py --config config_gnn_advection.json

To enable multi-step rollout training, set `train.rollout_steps` in the config
to an integer > 1 (requires rollout dataset fields).
Optional: set `train.rollout_random_length` to sample rollout lengths during
training between configurable min/max bounds.
Set `train.early_stopping_metric` to `rollout_state_rel_l2` to choose the best
checkpoint (and apply early-stopping patience) using mean state-space relative
L2 across validation samples and rollout steps. The default remains `val_loss`.

When trajectory metadata is present in the dataset, train/val/test splits are
always computed by trajectory id (then expanded to pair indices).
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass, replace
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from advection_data import load_advection_dataset
from gnn_advection import (
    TARGET_MODE_DELTA,
    EpochMetrics,
    TARGET_MODE_RATE,
    TARGET_MODE_STATE,
    TARGET_MODES,
    build_batched_edge_index,
    build_model_from_config,
    make_target,
    predict_type_from_target_mode,
    repeat_edge_attr,
    resolve_device,
    resolve_state_residual_mode,
    save_json,
    set_seed,
    split_indices,
    state_residual_from_mode,
)
from physics_inputs import (
    ADVECTION_SCHEME_UPWIND,
    ADVECTION_SCHEMES,
    PhysicsInputConfig,
    build_physics_augmented_inputs_1d,
    compute_advdiff_terms_abs_1d,
    count_physics_input_channels,
    fit_adv_input_minmax_1d,
    resolve_physics_input_cfg,
)
from input_features import (
    InputFeatureConfig,
    apply_state_minmax_minus1_1,
    append_position_channel,
    count_input_feature_channels,
    fit_state_minmax,
    resolve_input_feature_cfg,
)


class IndexDataset(Dataset):
    """Simple dataset returning precomputed sample indices."""

    def __init__(self, indices: np.ndarray) -> None:
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, idx: int) -> int:
        return int(self.indices[idx])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GNN for advection data.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config_gnn_advection.json"),
        help="Path to JSON config.",
    )
    return parser.parse_args()


def _parse_lr_schedule(
    train_cfg: Mapping[str, Any],
    *,
    epochs: int,
) -> Tuple[bool, Dict[int, float]]:
    """Parse optional epoch-based learning-rate schedule from config.

    Supported config shape:
      train.lr_schedule: [{"epoch": 200, "lr": 5e-5}, {"epoch": 400, "lr": 1e-5}]

    A single object is also accepted:
      train.lr_schedule: {"epoch": 200, "lr": 5e-5}

    New object form with boolean toggle:
      train.lr_schedule: {
        "enabled": true,
        "entries": [{"epoch": 200, "lr": 5e-5}]
      }
    """
    raw = train_cfg.get("lr_schedule", None)
    if raw is None:
        return False, {}

    entries: List[Mapping[str, Any]]
    enabled = True
    if isinstance(raw, Mapping):
        enabled = bool(raw.get("enabled", True))
        if ("epoch" in raw) or ("lr" in raw):
            # Backward-compatible single entry object.
            entries = [raw]
        else:
            entries_raw = raw.get("entries", raw.get("milestones", []))
            if isinstance(entries_raw, Mapping):
                entries = [entries_raw]
            elif isinstance(entries_raw, list):
                entries = []
                for item in entries_raw:
                    if not isinstance(item, Mapping):
                        raise ValueError(
                            "train.lr_schedule entries must be objects with keys "
                            "'epoch' and 'lr'."
                        )
                    entries.append(item)
            else:
                raise ValueError(
                    "train.lr_schedule.entries must be an object or list of objects."
                )
    elif isinstance(raw, list):
        entries = []
        for item in raw:
            if not isinstance(item, Mapping):
                raise ValueError(
                    "train.lr_schedule entries must be objects with keys "
                    "'epoch' and 'lr'."
                )
            entries.append(item)
    else:
        raise ValueError(
            "train.lr_schedule must be either an object "
            "{epoch: <int>, lr: <float>} or a list of such objects."
        )

    if not enabled:
        return False, {}

    schedule: Dict[int, float] = {}
    for entry in entries:
        if "epoch" not in entry or "lr" not in entry:
            raise ValueError(
                "Each train.lr_schedule entry must include keys 'epoch' and 'lr'."
            )
        ep = int(entry["epoch"])
        lr_val = float(entry["lr"])
        if ep < 1:
            raise ValueError(f"train.lr_schedule epoch must be >= 1, got {ep}.")
        if lr_val <= 0.0:
            raise ValueError(f"train.lr_schedule lr must be > 0, got {lr_val}.")
        if ep > int(epochs):
            # Keep strictness low here: silently ignore out-of-range milestone.
            continue
        if ep in schedule:
            raise ValueError(
                f"Duplicate train.lr_schedule epoch {ep}. "
                "Provide at most one LR value per epoch."
            )
        schedule[ep] = lr_val

    return True, schedule


def _batch_metrics(pred: torch.Tensor, target: torch.Tensor) -> Tuple[float, float]:
    err = pred - target
    mse = float(torch.mean(err * err).item())
    rmse = math.sqrt(max(mse, 0.0))
    denom = float(torch.norm(target).item())
    rel_l2 = float(torch.norm(err).item() / max(denom, 1e-12))
    return rmse, rel_l2


EARLY_STOPPING_METRIC_VAL_LOSS = "val_loss"
EARLY_STOPPING_METRIC_ROLLOUT_STATE_REL_L2 = "rollout_state_rel_l2"
EARLY_STOPPING_METRICS = {
    EARLY_STOPPING_METRIC_VAL_LOSS,
    EARLY_STOPPING_METRIC_ROLLOUT_STATE_REL_L2,
}


def _resolve_early_stopping_metric(
    train_cfg: Mapping[str, Any],
    *,
    rollout_mode: bool,
) -> str:
    """Resolve the validation metric used for checkpointing and patience."""
    metric = str(
        train_cfg.get(
            "early_stopping_metric",
            EARLY_STOPPING_METRIC_VAL_LOSS,
        )
    ).strip().lower()
    if metric not in EARLY_STOPPING_METRICS:
        raise ValueError(
            "train.early_stopping_metric must be one of "
            f"{sorted(EARLY_STOPPING_METRICS)}, got '{metric}'."
        )
    if (
        metric == EARLY_STOPPING_METRIC_ROLLOUT_STATE_REL_L2
        and not rollout_mode
    ):
        raise ValueError(
            "train.early_stopping_metric='rollout_state_rel_l2' requires "
            "rollout training (train.rollout_steps > 1 or randomized rollout "
            "length enabled)."
        )
    return metric


def _mean_sample_rel_l2(
    pred_flat: torch.Tensor,
    target_flat: torch.Tensor,
    *,
    batch_size: int,
    num_nodes: int,
    eps: float = 1e-12,
) -> float:
    """Return mean state relative L2, giving every sample equal weight."""
    if pred_flat.shape != target_flat.shape:
        raise ValueError(
            "pred_flat and target_flat must have identical shapes, got "
            f"{tuple(pred_flat.shape)} and {tuple(target_flat.shape)}."
        )
    expected_rows = int(batch_size) * int(num_nodes)
    if pred_flat.ndim < 2 or int(pred_flat.shape[0]) != expected_rows:
        raise ValueError(
            "Expected flattened state with first dimension batch_size*num_nodes="
            f"{expected_rows}, got shape {tuple(pred_flat.shape)}."
        )
    pred_by_sample = pred_flat.reshape(int(batch_size), -1)
    target_by_sample = target_flat.reshape(int(batch_size), -1)
    error_norm = torch.linalg.vector_norm(pred_by_sample - target_by_sample, dim=1)
    target_norm = torch.linalg.vector_norm(target_by_sample, dim=1)
    relative_l2 = error_norm / torch.clamp(target_norm, min=float(eps))
    return float(torch.mean(relative_l2).item())


def _checkpoint_metric_value(metrics: EpochMetrics, metric: str) -> float:
    """Extract the configured best-checkpoint/early-stopping metric."""
    if metric == EARLY_STOPPING_METRIC_VAL_LOSS:
        return float(metrics.loss)
    if metric == EARLY_STOPPING_METRIC_ROLLOUT_STATE_REL_L2:
        if metrics.rollout_state_rel_l2 is None:
            raise ValueError(
                "rollout_state_rel_l2 was selected, but rollout state metrics "
                "were not computed."
            )
        return float(metrics.rollout_state_rel_l2)
    raise ValueError(f"Unsupported early-stopping metric '{metric}'.")


@dataclass(frozen=True)
class GradientWeightedLossConfig:
    """Controls optional gradient-magnitude weighting of MSE loss."""

    enabled: bool = False
    strength: float = 1.0
    power: float = 1.0
    eps: float = 1e-12
    normalize_mean: bool = True
    clip_max: float | None = None
    smooth_radius: int = 0
    peak_spread_sigma: float = 0.0
    peak_spread_mode: str = "gaussian"


@dataclass(frozen=True)
class RelativeMSELossConfig:
    """Controls optional per-sample scale-normalized MSE component."""

    enabled: bool = False
    alpha: float = 0.0
    eps: float = 1e-12


@dataclass(frozen=True)
class RandomizedRolloutLengthConfig:
    """Controls optional randomized rollout length during training."""

    enabled: bool = False
    min_steps: int = 1
    max_steps: int | None = None


def _resolve_rollout_random_length_cfg(
    cfg: Mapping[str, Any],
) -> RandomizedRolloutLengthConfig:
    """Resolve optional randomized rollout-length training configuration.

    Config location:
      train.rollout_random_length

    Backward-compatible alias:
      train.random_rollout_length
    """
    train_cfg_raw = cfg.get("train", {})
    train_cfg = train_cfg_raw if isinstance(train_cfg_raw, Mapping) else {}
    raw = train_cfg.get(
        "rollout_random_length",
        train_cfg.get("random_rollout_length", {}),
    )

    if isinstance(raw, bool):
        enabled = bool(raw)
        min_steps = 1
        max_steps = None
    elif isinstance(raw, Mapping):
        enabled = bool(raw.get("enabled", False))
        min_steps = int(raw.get("min_steps", raw.get("min_rollout_steps", 1)))
        max_raw = raw.get("max_steps", raw.get("max_rollout_steps", None))
        max_steps = None if max_raw is None else int(max_raw)
    else:
        raise ValueError(
            "train.rollout_random_length must be either a boolean or object, got "
            f"{type(raw).__name__}."
        )

    if min_steps < 1:
        raise ValueError(
            f"train.rollout_random_length.min_steps must be >= 1, got {min_steps}."
        )
    if max_steps is not None and max_steps < min_steps:
        raise ValueError(
            "train.rollout_random_length.max_steps must be >= min_steps, got "
            f"{max_steps} < {min_steps}."
        )

    return RandomizedRolloutLengthConfig(
        enabled=enabled,
        min_steps=min_steps,
        max_steps=max_steps,
    )


def _resolve_gradient_weighted_loss_cfg(
    cfg: Mapping[str, Any],
) -> GradientWeightedLossConfig:
    """Resolve optional shock-focused loss weighting configuration.

    Preferred location:
      train.gradient_weighted_loss

    Backward-compatible alias:
      train.shock_weighted_loss
    """
    train_cfg_raw = cfg.get("train", {})
    train_cfg = train_cfg_raw if isinstance(train_cfg_raw, Mapping) else {}
    raw = train_cfg.get(
        "gradient_weighted_loss",
        train_cfg.get("shock_weighted_loss", {}),
    )

    if isinstance(raw, bool):
        enabled = bool(raw)
        strength = 1.0
        power = 1.0
        eps = 1e-12
        normalize_mean = True
        clip_max = None
        smooth_radius = 0
        peak_spread_sigma = 0.0
        peak_spread_mode = "gaussian"
    elif isinstance(raw, Mapping):
        enabled = bool(raw.get("enabled", False))
        strength = float(raw.get("strength", raw.get("alpha", 1.0)))
        power = float(raw.get("power", 1.0))
        eps = float(raw.get("eps", 1e-12))
        normalize_mean = bool(raw.get("normalize_mean", True))
        clip_raw = raw.get("clip_max", None)
        clip_max = None if clip_raw is None else float(clip_raw)
        smooth_radius = int(raw.get("smooth_radius", 0))
        peak_spread_sigma = float(
            raw.get(
                "peak_spread_sigma",
                raw.get("peak_spread_radius", raw.get("spread_radius", 0)),
            )
        )
        peak_spread_mode = str(
            raw.get("peak_spread_mode", raw.get("spread_mode", "gaussian"))
        ).strip().lower()
    else:
        raise ValueError(
            "train.gradient_weighted_loss must be either a boolean or an object, got "
            f"{type(raw).__name__}."
        )

    if strength < 0.0:
        raise ValueError(
            f"gradient_weighted_loss.strength must be >= 0, got {strength}."
        )
    if power <= 0.0:
        raise ValueError(
            f"gradient_weighted_loss.power must be > 0, got {power}."
        )
    if eps <= 0.0:
        raise ValueError(f"gradient_weighted_loss.eps must be > 0, got {eps}.")
    if clip_max is not None and clip_max <= 0.0:
        raise ValueError(
            f"gradient_weighted_loss.clip_max must be > 0 when provided, got {clip_max}."
        )
    if smooth_radius < 0:
        raise ValueError(
            f"gradient_weighted_loss.smooth_radius must be >= 0, got {smooth_radius}."
        )
    if peak_spread_sigma < 0.0:
        raise ValueError(
            "gradient_weighted_loss.peak_spread_sigma must be >= 0, got "
            f"{peak_spread_sigma}."
        )
    if peak_spread_mode not in {"gaussian", "max"}:
        raise ValueError(
            "gradient_weighted_loss.peak_spread_mode must be one of "
            "['gaussian', 'max'], got "
            f"'{peak_spread_mode}'."
        )

    return GradientWeightedLossConfig(
        enabled=enabled,
        strength=strength,
        power=power,
        eps=eps,
        normalize_mean=normalize_mean,
        clip_max=clip_max,
        smooth_radius=smooth_radius,
        peak_spread_sigma=peak_spread_sigma,
        peak_spread_mode=peak_spread_mode,
    )


def _resolve_relative_mse_loss_cfg(
    cfg: Mapping[str, Any],
) -> RelativeMSELossConfig:
    """Resolve optional per-sample scale-normalized MSE configuration.

    Preferred location:
      train.relative_mse

    Backward-compatible aliases:
      train.relative_mse_loss
      train.per_sample_scale_loss
    """
    train_cfg_raw = cfg.get("train", {})
    train_cfg = train_cfg_raw if isinstance(train_cfg_raw, Mapping) else {}
    raw = train_cfg.get(
        "relative_mse",
        train_cfg.get(
            "relative_mse_loss",
            train_cfg.get("per_sample_scale_loss", {}),
        ),
    )

    if isinstance(raw, bool):
        enabled = bool(raw)
        alpha = 0.5
        eps = 1e-12
    elif isinstance(raw, Mapping):
        enabled = bool(raw.get("enabled", False))
        alpha = float(raw.get("alpha", 0.5))
        eps = float(raw.get("eps", 1e-12))
    else:
        raise ValueError(
            "train.relative_mse must be either a boolean or an object, got "
            f"{type(raw).__name__}."
        )

    if alpha < 0.0 or alpha > 1.0:
        raise ValueError(
            f"train.relative_mse.alpha must be in [0,1], got {alpha}."
        )
    if eps <= 0.0:
        raise ValueError(f"train.relative_mse.eps must be > 0, got {eps}.")

    return RelativeMSELossConfig(
        enabled=enabled,
        alpha=alpha,
        eps=eps,
    )


def _periodic_mean_filter_1d(values: torch.Tensor, radius: int) -> torch.Tensor:
    """Apply periodic moving-average smoothing across the node dimension.

    Args:
      values: Tensor [B, N, C]
      radius: Window radius r => kernel size (2r+1)
    """
    if radius <= 0:
        return values
    if values.ndim != 3:
        raise ValueError(f"values must be [B,N,C], got {tuple(values.shape)}.")

    kernel_size = int(2 * radius + 1)
    v = values.permute(0, 2, 1)  # [B,C,N]
    v_pad = F.pad(v, (radius, radius), mode="circular")
    k = torch.full(
        (int(v.shape[1]), 1, kernel_size),
        fill_value=1.0 / float(kernel_size),
        dtype=v.dtype,
        device=v.device,
    )
    out = F.conv1d(v_pad, weight=k, stride=1, padding=0, groups=int(v.shape[1]))
    return out.permute(0, 2, 1)


def _periodic_max_spread_1d(values: torch.Tensor, radius: int) -> torch.Tensor:
    """Spread local peaks periodically via 1D max pooling across nodes."""
    if radius <= 0:
        return values
    if values.ndim != 3:
        raise ValueError(f"values must be [B,N,C], got {tuple(values.shape)}.")

    kernel_size = int(2 * radius + 1)
    v = values.permute(0, 2, 1)  # [B,C,N]
    v_pad = F.pad(v, (radius, radius), mode="circular")
    out = F.max_pool1d(v_pad, kernel_size=kernel_size, stride=1, padding=0)
    return out.permute(0, 2, 1)


def _periodic_gaussian_filter_1d(
    values: torch.Tensor,
    *,
    sigma: float,
    truncate: float = 4.0,
) -> torch.Tensor:
    """Apply periodic Gaussian smoothing across nodes.

    Args:
      values: Tensor [B, N, C]
      sigma: Gaussian std-dev in units of node spacing.
      truncate: Kernel support radius in sigma units.
    """
    if sigma <= 0.0:
        return values
    if values.ndim != 3:
        raise ValueError(f"values must be [B,N,C], got {tuple(values.shape)}.")
    if truncate <= 0.0:
        raise ValueError(f"truncate must be > 0, got {truncate}.")

    radius = max(1, int(math.ceil(float(truncate) * float(sigma))))
    kernel_size = int(2 * radius + 1)

    x = torch.arange(
        -radius,
        radius + 1,
        device=values.device,
        dtype=values.dtype,
    )
    k1 = torch.exp(-0.5 * (x / float(sigma)) ** 2)
    k1 = k1 / torch.clamp(torch.sum(k1), min=1e-12)

    v = values.permute(0, 2, 1)  # [B,C,N]
    v_pad = F.pad(v, (radius, radius), mode="circular")
    k = k1.view(1, 1, kernel_size).repeat(int(v.shape[1]), 1, 1)
    out = F.conv1d(v_pad, weight=k, stride=1, padding=0, groups=int(v.shape[1]))
    return out.permute(0, 2, 1)


def _gradient_weights_from_true_state(
    *,
    state: torch.Tensor,
    loss_cfg: GradientWeightedLossConfig,
) -> torch.Tensor:
    """Compute node-wise gradient weights from true state tensor [B,N,F]."""
    grad_center = 0.5 * (
        torch.roll(state, shifts=-1, dims=1) - torch.roll(state, shifts=1, dims=1)
    )
    grad_mag = torch.mean(torch.abs(grad_center), dim=-1, keepdim=True)  # [B,N,1]

    if loss_cfg.smooth_radius > 0:
        grad_mag = _periodic_mean_filter_1d(grad_mag, radius=int(loss_cfg.smooth_radius))
    if loss_cfg.normalize_mean:
        mean_mag = torch.mean(grad_mag)
        grad_mag = grad_mag / torch.clamp(mean_mag, min=float(loss_cfg.eps))

    weights = 1.0 + float(loss_cfg.strength) * torch.pow(
        grad_mag + float(loss_cfg.eps),
        float(loss_cfg.power),
    )
    if loss_cfg.peak_spread_sigma > 0.0:
        weight_excess = torch.clamp(weights - 1.0, min=0.0)
        if loss_cfg.peak_spread_mode == "gaussian":
            weight_excess = _periodic_gaussian_filter_1d(
                weight_excess,
                sigma=float(loss_cfg.peak_spread_sigma),
            )
        else:
            spread_radius = max(1, int(round(float(loss_cfg.peak_spread_sigma))))
            weight_excess = _periodic_max_spread_1d(
                weight_excess,
                radius=spread_radius,
            )
        weights = 1.0 + weight_excess
    if loss_cfg.clip_max is not None:
        weights = torch.clamp(weights, max=float(loss_cfg.clip_max))
    return weights


def _gradient_weighted_mse_loss(
    *,
    pred_target_flat: torch.Tensor,
    true_target_flat: torch.Tensor,
    true_state_flat: torch.Tensor,
    batch_size: int,
    num_nodes: int,
    loss_cfg: GradientWeightedLossConfig,
) -> torch.Tensor:
    """Compute MSE, optionally weighted by periodic state-gradient magnitude."""
    sq_err = (pred_target_flat - true_target_flat) ** 2
    if not loss_cfg.enabled:
        return torch.mean(sq_err)

    if pred_target_flat.ndim != 2 or true_target_flat.ndim != 2:
        raise ValueError(
            "Expected flat tensors [B*N,F] for pred_target_flat/true_target_flat, got "
            f"{tuple(pred_target_flat.shape)} and {tuple(true_target_flat.shape)}."
        )
    if true_state_flat.ndim != 2:
        raise ValueError(
            f"true_state_flat must have shape [B*N,F], got {tuple(true_state_flat.shape)}."
        )
    if pred_target_flat.shape != true_target_flat.shape:
        raise ValueError(
            "pred/target flat shape mismatch: "
            f"{tuple(pred_target_flat.shape)} vs {tuple(true_target_flat.shape)}."
        )
    if true_state_flat.shape[0] != pred_target_flat.shape[0]:
        raise ValueError(
            "true_state_flat row count must match pred/target rows, got "
            f"{true_state_flat.shape[0]} vs {pred_target_flat.shape[0]}."
        )
    if (batch_size * num_nodes) != int(pred_target_flat.shape[0]):
        raise ValueError(
            "batch_size * num_nodes mismatch with flattened rows: "
            f"{batch_size}*{num_nodes} vs {pred_target_flat.shape[0]}."
        )

    state = true_state_flat.reshape(batch_size, num_nodes, -1)
    weights = _gradient_weights_from_true_state(
        state=state,
        loss_cfg=loss_cfg,
    )

    weights_flat = weights.reshape(batch_size * num_nodes, 1)
    weighted_sq = sq_err * weights_flat
    denom = torch.clamp(
        torch.sum(weights_flat) * float(sq_err.shape[1]),
        min=float(loss_cfg.eps),
    )
    return torch.sum(weighted_sq) / denom


def _relative_mse_component_loss(
    *,
    pred_target_flat: torch.Tensor,
    true_target_flat: torch.Tensor,
    batch_size: int,
    num_nodes: int,
    eps: float,
) -> torch.Tensor:
    """Compute per-sample scale-normalized MSE and average over batch.

    For sample b, this is:
      mse_b / max(mean(y_b^2), eps)
    where y_b is the full target tensor for that sample.
    """
    if pred_target_flat.ndim != 2 or true_target_flat.ndim != 2:
        raise ValueError(
            "Expected flat tensors [B*N,F] for pred_target_flat/true_target_flat, got "
            f"{tuple(pred_target_flat.shape)} and {tuple(true_target_flat.shape)}."
        )
    if pred_target_flat.shape != true_target_flat.shape:
        raise ValueError(
            "pred/target flat shape mismatch: "
            f"{tuple(pred_target_flat.shape)} vs {tuple(true_target_flat.shape)}."
        )
    if (batch_size * num_nodes) != int(pred_target_flat.shape[0]):
        raise ValueError(
            "batch_size * num_nodes mismatch with flattened rows: "
            f"{batch_size}*{num_nodes} vs {pred_target_flat.shape[0]}."
        )

    pred = pred_target_flat.reshape(batch_size, num_nodes, -1)
    true = true_target_flat.reshape(batch_size, num_nodes, -1)
    sq_err = (pred - true) ** 2
    mse_per_sample = torch.mean(sq_err, dim=(1, 2))
    scale_sq_per_sample = torch.mean(true * true, dim=(1, 2))
    rel_mse_per_sample = mse_per_sample / torch.clamp(
        scale_sq_per_sample,
        min=float(eps),
    )
    return torch.mean(rel_mse_per_sample)


def _data_fidelity_loss(
    *,
    pred_target_flat: torch.Tensor,
    true_target_flat: torch.Tensor,
    true_state_flat: torch.Tensor,
    batch_size: int,
    num_nodes: int,
    grad_loss_cfg: GradientWeightedLossConfig,
    relative_mse_cfg: RelativeMSELossConfig,
) -> torch.Tensor:
    """Compute blended data loss with optional relative-MSE component."""
    abs_loss = _gradient_weighted_mse_loss(
        pred_target_flat=pred_target_flat,
        true_target_flat=true_target_flat,
        true_state_flat=true_state_flat,
        batch_size=batch_size,
        num_nodes=num_nodes,
        loss_cfg=grad_loss_cfg,
    )
    if not relative_mse_cfg.enabled:
        return abs_loss

    rel_loss = _relative_mse_component_loss(
        pred_target_flat=pred_target_flat,
        true_target_flat=true_target_flat,
        batch_size=batch_size,
        num_nodes=num_nodes,
        eps=float(relative_mse_cfg.eps),
    )
    alpha = float(relative_mse_cfg.alpha)
    return (1.0 - alpha) * abs_loss + alpha * rel_loss


def _sanity_check_gradient_weighted_loss_impl() -> None:
    """Quick internal consistency checks for weighted-loss implementation."""
    x = torch.tensor(
        [[[0.0], [1.0], [0.0], [0.0]]],
        dtype=torch.float32,
    )  # [B=1,N=4,F=1]
    pred = torch.tensor(
        [[[0.0], [0.8], [0.2], [0.0]]],
        dtype=torch.float32,
    )
    true = torch.tensor(
        [[[0.0], [1.0], [0.0], [0.0]]],
        dtype=torch.float32,
    )
    pred_flat = pred.reshape(-1, 1)
    true_flat = true.reshape(-1, 1)
    x_flat = x.reshape(-1, 1)

    mse_ref = torch.mean((pred_flat - true_flat) ** 2)

    cfg_off = GradientWeightedLossConfig(enabled=False)
    loss_off = _gradient_weighted_mse_loss(
        pred_target_flat=pred_flat,
        true_target_flat=true_flat,
        true_state_flat=x_flat,
        batch_size=1,
        num_nodes=4,
        loss_cfg=cfg_off,
    )
    if not torch.allclose(loss_off, mse_ref, rtol=1e-6, atol=1e-8):
        raise RuntimeError(
            "Gradient-weighted loss sanity check failed (enabled=False mismatch)."
        )

    cfg_uniform = GradientWeightedLossConfig(
        enabled=True,
        strength=0.0,
        power=1.0,
        eps=1e-12,
        normalize_mean=True,
        clip_max=None,
        smooth_radius=0,
        peak_spread_sigma=0.0,
        peak_spread_mode="gaussian",
    )
    loss_uniform = _gradient_weighted_mse_loss(
        pred_target_flat=pred_flat,
        true_target_flat=true_flat,
        true_state_flat=x_flat,
        batch_size=1,
        num_nodes=4,
        loss_cfg=cfg_uniform,
    )
    if not torch.allclose(loss_uniform, mse_ref, rtol=1e-6, atol=1e-8):
        raise RuntimeError(
            "Gradient-weighted loss sanity check failed (strength=0 mismatch)."
        )


def _sanity_check_relative_mse_loss_impl() -> None:
    """Quick checks for relative-MSE scaling and blend behavior."""
    pred = torch.tensor(
        [
            [[2.0], [0.0]],
            [[20.0], [0.0]],
        ],
        dtype=torch.float32,
    )  # [B=2,N=2,F=1]
    true = torch.tensor(
        [
            [[1.0], [0.0]],
            [[10.0], [0.0]],
        ],
        dtype=torch.float32,
    )
    pred_flat = pred.reshape(-1, 1)
    true_flat = true.reshape(-1, 1)

    rel_loss = _relative_mse_component_loss(
        pred_target_flat=pred_flat,
        true_target_flat=true_flat,
        batch_size=2,
        num_nodes=2,
        eps=1e-12,
    )
    rel_expected = torch.tensor(1.0, dtype=torch.float32)
    if not torch.allclose(rel_loss, rel_expected, rtol=1e-6, atol=1e-8):
        raise RuntimeError(
            "Relative-MSE sanity check failed (scale normalization mismatch)."
        )

    x_state = true_flat.clone()
    grad_cfg = GradientWeightedLossConfig(enabled=False)
    rel_cfg = RelativeMSELossConfig(enabled=True, alpha=0.25, eps=1e-12)
    blended = _data_fidelity_loss(
        pred_target_flat=pred_flat,
        true_target_flat=true_flat,
        true_state_flat=x_state,
        batch_size=2,
        num_nodes=2,
        grad_loss_cfg=grad_cfg,
        relative_mse_cfg=rel_cfg,
    )
    mse_abs = torch.mean((pred_flat - true_flat) ** 2)
    expected_blended = (1.0 - float(rel_cfg.alpha)) * mse_abs + float(rel_cfg.alpha) * rel_expected
    if not torch.allclose(blended, expected_blended, rtol=1e-6, atol=1e-8):
        raise RuntimeError(
            "Relative-MSE sanity check failed (blend behavior mismatch)."
        )


@dataclass(frozen=True)
class MassConservationLossConfig:
    """Controls optional mass-conservation penalty on predicted next states."""

    enabled: bool = False
    lambda_mass: float = 0.0


TV_LOSS_MODE_EXCESS = "excess"
TV_LOSS_MODE_MATCH = "match"
TV_LOSS_MODES = {TV_LOSS_MODE_EXCESS, TV_LOSS_MODE_MATCH}


@dataclass(frozen=True)
class TVLossConfig:
    """Controls optional TV-based anti-oscillation regularization."""

    enabled: bool = False
    lambda_tv: float = 0.0
    mode: str = TV_LOSS_MODE_EXCESS


def _resolve_tv_loss_cfg(
    cfg: Mapping[str, Any],
) -> TVLossConfig:
    """Resolve optional total-variation regularization configuration.

    Config location:
      train.tv_loss

    Backward-compatible aliases:
      train.total_variation_loss
      train.tv_regularization
    """
    train_cfg_raw = cfg.get("train", {})
    train_cfg = train_cfg_raw if isinstance(train_cfg_raw, Mapping) else {}
    raw = train_cfg.get(
        "tv_loss",
        train_cfg.get("total_variation_loss", train_cfg.get("tv_regularization", {})),
    )

    if isinstance(raw, bool):
        enabled = bool(raw)
        lambda_tv = 1e-4
        mode = TV_LOSS_MODE_EXCESS
    elif isinstance(raw, Mapping):
        enabled = bool(raw.get("enabled", False))
        lambda_tv = float(raw.get("lambda_tv", raw.get("weight", 1e-4)))
        mode = str(raw.get("mode", TV_LOSS_MODE_EXCESS)).strip().lower()
    else:
        raise ValueError(
            "train.tv_loss must be either a boolean or object, got "
            f"{type(raw).__name__}."
        )

    if lambda_tv < 0.0:
        raise ValueError(f"train.tv_loss.lambda_tv must be >= 0, got {lambda_tv}.")
    if mode not in TV_LOSS_MODES:
        raise ValueError(
            "train.tv_loss.mode must be one of "
            f"{sorted(TV_LOSS_MODES)}, got '{mode}'."
        )

    return TVLossConfig(
        enabled=enabled,
        lambda_tv=lambda_tv,
        mode=mode,
    )


def _resolve_mass_conservation_loss_cfg(
    cfg: Mapping[str, Any],
) -> MassConservationLossConfig:
    """Resolve optional mass-conservation loss configuration.

    Config location:
      train.mass_conservation_loss
    """
    train_cfg_raw = cfg.get("train", {})
    train_cfg = train_cfg_raw if isinstance(train_cfg_raw, Mapping) else {}
    raw = train_cfg.get("mass_conservation_loss", {})

    if isinstance(raw, bool):
        enabled = bool(raw)
        lambda_mass = 1e-3
    elif isinstance(raw, Mapping):
        enabled = bool(raw.get("enabled", False))
        lambda_mass = float(raw.get("lambda_mass", raw.get("weight", 1e-3)))
    else:
        raise ValueError(
            "train.mass_conservation_loss must be either a boolean or object, got "
            f"{type(raw).__name__}."
        )

    if lambda_mass < 0.0:
        raise ValueError(
            "train.mass_conservation_loss.lambda_mass must be >= 0, got "
            f"{lambda_mass}."
        )

    return MassConservationLossConfig(
        enabled=enabled,
        lambda_mass=lambda_mass,
    )


def _mass_conservation_penalty(
    *,
    pred_state_flat: torch.Tensor,
    ref_state_flat: torch.Tensor,
    batch_size: int,
    num_nodes: int,
) -> torch.Tensor:
    """Penalty for drift in per-feature spatial mean between ref and prediction.

    For uniform 1D grids this is proportional to mass drift.
    """
    if pred_state_flat.ndim != 2 or ref_state_flat.ndim != 2:
        raise ValueError(
            "pred_state_flat/ref_state_flat must be [B*N,F], got "
            f"{tuple(pred_state_flat.shape)} and {tuple(ref_state_flat.shape)}."
        )
    if pred_state_flat.shape != ref_state_flat.shape:
        raise ValueError(
            "pred/ref state shape mismatch: "
            f"{tuple(pred_state_flat.shape)} vs {tuple(ref_state_flat.shape)}."
        )
    if (batch_size * num_nodes) != int(pred_state_flat.shape[0]):
        raise ValueError(
            "batch_size * num_nodes mismatch with flattened rows: "
            f"{batch_size}*{num_nodes} vs {pred_state_flat.shape[0]}."
        )

    pred = pred_state_flat.reshape(batch_size, num_nodes, -1)
    ref = ref_state_flat.reshape(batch_size, num_nodes, -1)
    mean_pred = torch.mean(pred, dim=1)  # [B,F]
    mean_ref = torch.mean(ref, dim=1)    # [B,F]
    drift = mean_pred - mean_ref
    return torch.mean(drift * drift)


def _total_variation_periodic_1d(
    *,
    state_flat: torch.Tensor,
    batch_size: int,
    num_nodes: int,
) -> torch.Tensor:
    """Periodic 1D total variation per sample/feature from flattened states.

    Returns:
      tv: Tensor [B, F]
    """
    if state_flat.ndim != 2:
        raise ValueError(
            f"state_flat must have shape [B*N,F], got {tuple(state_flat.shape)}."
        )
    if (batch_size * num_nodes) != int(state_flat.shape[0]):
        raise ValueError(
            "batch_size * num_nodes mismatch with flattened rows: "
            f"{batch_size}*{num_nodes} vs {state_flat.shape[0]}."
        )
    state = state_flat.reshape(batch_size, num_nodes, -1)
    diff = torch.roll(state, shifts=-1, dims=1) - state
    return torch.sum(torch.abs(diff), dim=1)


def _tv_regularization_penalty(
    *,
    pred_state_flat: torch.Tensor,
    ref_state_flat: torch.Tensor,
    batch_size: int,
    num_nodes: int,
    tv_loss_cfg: TVLossConfig,
) -> torch.Tensor:
    """TV regularization penalty to discourage spurious spatial oscillations.

    Modes:
      - `excess`: penalize only TV overshoot beyond reference TV.
      - `match`: penalize signed TV mismatch to reference TV.
    """
    if pred_state_flat.shape != ref_state_flat.shape:
        raise ValueError(
            "pred/ref state shape mismatch for TV penalty: "
            f"{tuple(pred_state_flat.shape)} vs {tuple(ref_state_flat.shape)}."
        )
    tv_pred = _total_variation_periodic_1d(
        state_flat=pred_state_flat,
        batch_size=batch_size,
        num_nodes=num_nodes,
    )
    tv_ref = _total_variation_periodic_1d(
        state_flat=ref_state_flat,
        batch_size=batch_size,
        num_nodes=num_nodes,
    )
    if tv_loss_cfg.mode == TV_LOSS_MODE_EXCESS:
        mismatch = torch.relu(tv_pred - tv_ref)
    elif tv_loss_cfg.mode == TV_LOSS_MODE_MATCH:
        mismatch = tv_pred - tv_ref
    else:
        raise RuntimeError(f"Unexpected tv_loss mode '{tv_loss_cfg.mode}'.")
    return torch.mean(mismatch * mismatch)


@dataclass(frozen=True)
class GodunovCorrectionConfig:
    """Controls optional residual learning around one-step Godunov baseline."""

    enabled: bool = False
    advection_scheme: str = ADVECTION_SCHEME_UPWIND
    c_override: float | None = None


def _resolve_godunov_correction_cfg(cfg: Mapping[str, Any]) -> GodunovCorrectionConfig:
    """Resolve optional correction-to-Godunov configuration.

    Preferred location:
      train.godunov_correction

    Backward-compatible fallback:
      model.godunov_correction
    """
    train_cfg_raw = cfg.get("train", {})
    model_cfg_raw = cfg.get("model", {})
    train_cfg = train_cfg_raw if isinstance(train_cfg_raw, Mapping) else {}
    model_cfg = model_cfg_raw if isinstance(model_cfg_raw, Mapping) else {}

    raw = train_cfg.get("godunov_correction", model_cfg.get("godunov_correction", {}))
    if isinstance(raw, bool):
        enabled = bool(raw)
        scheme = ADVECTION_SCHEME_UPWIND
        c_override = None
    elif isinstance(raw, Mapping):
        enabled = bool(raw.get("enabled", False))
        scheme = str(
            raw.get(
                "advection_scheme",
                raw.get("scheme", ADVECTION_SCHEME_UPWIND),
            )
        ).strip().lower()
        c_raw = raw.get("c_override", raw.get("c", None))
        c_override = None if c_raw is None else float(c_raw)
    else:
        raise ValueError(
            "godunov_correction must be either a boolean or an object, got "
            f"{type(raw).__name__}."
        )

    if scheme not in ADVECTION_SCHEMES:
        raise ValueError(
            "godunov_correction.advection_scheme must be one of "
            f"{sorted(ADVECTION_SCHEMES)}, got '{scheme}'."
        )

    return GodunovCorrectionConfig(
        enabled=enabled,
        advection_scheme=scheme,
        c_override=c_override,
    )


def _state_from_target(
    *,
    x_now_flat: torch.Tensor,
    target_flat: torch.Tensor,
    dt: float,
    target_mode: str,
) -> torch.Tensor:
    """Inverse map of target definition: return state_{n+1} from target tensor."""
    if target_mode == TARGET_MODE_STATE:
        return target_flat
    if target_mode == TARGET_MODE_DELTA:
        return x_now_flat + target_flat
    if target_mode == TARGET_MODE_RATE:
        return x_now_flat + float(dt) * target_flat
    raise RuntimeError(f"Unexpected target_mode='{target_mode}'.")


def _godunov_next_state_flat(
    *,
    x_now_flat: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    dt: float,
    dx: float,
    c_from_data: float,
    godunov_cfg: GodunovCorrectionConfig,
) -> torch.Tensor:
    """Compute one-step FV/DEC advection baseline state_{n+1}."""
    c_eff = (
        float(godunov_cfg.c_override)
        if godunov_cfg.c_override is not None
        else float(c_from_data)
    )
    r_adv_abs, _ = compute_advdiff_terms_abs_1d(
        x_abs=x_now_flat,
        edge_index=edge_index,
        edge_attr=edge_attr,
        c=c_eff,
        dx=float(dx),
        nu=0.0,
        advection_scheme=godunov_cfg.advection_scheme,
        compute_adv=True,
        compute_diff=False,
    )
    assert r_adv_abs is not None
    return x_now_flat + float(dt) * r_adv_abs


def _run_split(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    x_all: torch.Tensor,
    u_next_all: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    pos_nodes: torch.Tensor,
    dt: float,
    dx: float,
    c_from_data: float,
    physics_cfg: PhysicsInputConfig,
    input_cfg: InputFeatureConfig,
    godunov_cfg: GodunovCorrectionConfig,
    grad_loss_cfg: GradientWeightedLossConfig,
    relative_mse_cfg: RelativeMSELossConfig,
    mass_loss_cfg: MassConservationLossConfig,
    tv_loss_cfg: TVLossConfig,
    target_mode: str,
    state_residual_mode: str,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    grad_clip: float,
) -> EpochMetrics:
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_rmse = 0.0
    total_rel_l2 = 0.0
    total_count = 0

    for batch_indices in loader:
        idx = batch_indices.long()
        x_b = x_all[idx].to(device=device)
        u_next_b = u_next_all[idx].to(device=device)
        target_b = make_target(x_b, u_next_b, dt=dt, target_mode=target_mode)

        bsz, n_nodes, n_feat = x_b.shape
        x_flat = x_b.reshape(bsz * n_nodes, n_feat)
        true_state_flat = u_next_b.reshape(bsz * n_nodes, n_feat)
        target_flat = target_b.reshape(bsz * n_nodes, n_feat)
        ei_batched = build_batched_edge_index(
            edge_index=edge_index, num_nodes=n_nodes, batch_size=bsz, device=device
        )
        ea_batched = repeat_edge_attr(
            edge_attr=edge_attr,
            batch_size=bsz,
            device=device,
            dtype=x_flat.dtype,
        )

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        x_model_flat = build_physics_augmented_inputs_1d(
            x_state=x_flat,
            edge_index=ei_batched,
            edge_attr=ea_batched,
            dt=dt,
            dx=dx,
            c_from_data=c_from_data,
            physics_cfg=physics_cfg,
            step_k=0,
        )
        x_model_flat = append_position_channel(
            x_state=x_model_flat,
            pos_nodes=pos_nodes,
            input_cfg=input_cfg,
            batch_size=bsz,
        )
        x_model_flat = apply_state_minmax_minus1_1(
            x_model=x_model_flat,
            input_cfg=input_cfg,
        )

        if godunov_cfg.enabled:
            # Model predicts correction in target space around one-step Godunov baseline.
            if target_mode == TARGET_MODE_STATE:
                raise ValueError(
                    "godunov_correction currently requires target_mode in "
                    "{delta, rate}; state-mode correction is not supported."
                )
            god_state_flat = _godunov_next_state_flat(
                x_now_flat=x_flat,
                edge_index=ei_batched,
                edge_attr=ea_batched,
                dt=dt,
                dx=dx,
                c_from_data=c_from_data,
                godunov_cfg=godunov_cfg,
            )
            god_target_flat = _target_from_states(
                x_now_flat=x_flat,
                state_next_flat=god_state_flat,
                dt=dt,
                target_mode=target_mode,
            )
            pred_corr_flat = model(
                x_model_flat,
                ei_batched,
                edge_attr=ea_batched,
                dt=dt,
            )
            pred_target_flat = god_target_flat + pred_corr_flat
            pred_state_flat = _state_from_target(
                x_now_flat=x_flat,
                target_flat=pred_target_flat,
                dt=dt,
                target_mode=target_mode,
            )
        else:
            pred_state_flat = model.predict_state(
                x_model_flat,
                ei_batched,
                edge_attr=ea_batched,
                dt=dt,
                state_residual=state_residual_from_mode(
                    x_raw=x_flat,
                    state_residual_mode=state_residual_mode,
                ),
            )
            pred_target_flat = _target_from_states(
                x_now_flat=x_flat,
                state_next_flat=pred_state_flat,
                dt=dt,
                target_mode=target_mode,
            )

        loss_data = _data_fidelity_loss(
            pred_target_flat=pred_target_flat,
            true_target_flat=target_flat,
            true_state_flat=true_state_flat,
            batch_size=bsz,
            num_nodes=n_nodes,
            grad_loss_cfg=grad_loss_cfg,
            relative_mse_cfg=relative_mse_cfg,
        )
        loss = loss_data
        if mass_loss_cfg.enabled:
            mass_loss = _mass_conservation_penalty(
                pred_state_flat=pred_state_flat,
                ref_state_flat=x_flat,
                batch_size=bsz,
                num_nodes=n_nodes,
            )
            loss = loss + float(mass_loss_cfg.lambda_mass) * mass_loss
        if tv_loss_cfg.enabled:
            tv_loss = _tv_regularization_penalty(
                pred_state_flat=pred_state_flat,
                ref_state_flat=true_state_flat,
                batch_size=bsz,
                num_nodes=n_nodes,
                tv_loss_cfg=tv_loss_cfg,
            )
            loss = loss + float(tv_loss_cfg.lambda_tv) * tv_loss

        if is_train:
            loss.backward()
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

        batch_n = int(bsz)
        rmse, rel_l2 = _batch_metrics(pred_target_flat.detach(), target_flat.detach())
        total_loss += float(loss.item()) * batch_n
        total_rmse += rmse * batch_n
        total_rel_l2 += rel_l2 * batch_n
        total_count += batch_n

    if total_count == 0:
        return EpochMetrics(loss=float("nan"), rmse=float("nan"), rel_l2=float("nan"))
    return EpochMetrics(
        loss=total_loss / total_count,
        rmse=total_rmse / total_count,
        rel_l2=total_rel_l2 / total_count,
    )


def _target_from_states(
    *,
    x_now_flat: torch.Tensor,
    state_next_flat: torch.Tensor,
    dt: float,
    target_mode: str,
) -> torch.Tensor:
    if target_mode == TARGET_MODE_STATE:
        return state_next_flat
    if target_mode == TARGET_MODE_DELTA:
        return state_next_flat - x_now_flat
    if target_mode == TARGET_MODE_RATE:
        return (state_next_flat - x_now_flat) / float(dt)
    raise RuntimeError(f"Unexpected target_mode='{target_mode}'.")


def _select_rollout_start_indices(
    *,
    split_indices_np: np.ndarray,
    pair_time_index_all: torch.Tensor,
    window: int,
    rollout_steps: int,
) -> np.ndarray:
    if rollout_steps < 1:
        raise ValueError(f"rollout_steps must be >= 1, got {rollout_steps}.")
    if window < rollout_steps:
        return np.zeros((0,), dtype=np.int64)
    max_start_t = int(window - rollout_steps)
    split_idx_t = torch.as_tensor(split_indices_np, dtype=torch.long)
    split_t = pair_time_index_all[split_idx_t].detach().cpu().numpy()
    keep = split_t <= max_start_t
    return np.asarray(split_indices_np[keep], dtype=np.int64)


def _finite_dynamic_range(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or not np.isfinite(arr).all():
        return float("inf")
    return float(np.max(arr) - np.min(arr))


def _find_constant_trajectory_ids(
    *,
    x_all: torch.Tensor,
    u_next_all: torch.Tensor,
    pair_trajectory_id: torch.Tensor | None,
    rollout_states: torch.Tensor | None,
    atol: float,
) -> Tuple[np.ndarray, Dict[int, float]]:
    """Return trajectory ids whose state is constant over all available samples."""
    if pair_trajectory_id is None:
        return np.zeros((0,), dtype=np.int64), {}
    if atol < 0.0:
        raise ValueError(f"data.constant_trajectory_tol must be >= 0, got {atol}.")

    pair_traj_np = pair_trajectory_id.detach().cpu().numpy().astype(np.int64)
    unique_traj = np.unique(pair_traj_np)
    x_np = x_all.detach().cpu().numpy()
    u_next_np = u_next_all.detach().cpu().numpy()
    rollout_np = (
        None
        if rollout_states is None
        else rollout_states.detach().cpu().numpy()
    )

    constant_ids: List[int] = []
    constant_ranges: Dict[int, float] = {}
    for traj_id in unique_traj.tolist():
        tid = int(traj_id)
        if rollout_np is not None and 0 <= tid < int(rollout_np.shape[0]):
            values = rollout_np[tid]
        else:
            pair_mask = pair_traj_np == tid
            values = np.concatenate(
                [x_np[pair_mask].reshape(-1), u_next_np[pair_mask].reshape(-1)]
            )
        dynamic_range = _finite_dynamic_range(values)
        if dynamic_range <= float(atol):
            constant_ids.append(tid)
            constant_ranges[tid] = dynamic_range

    return np.asarray(constant_ids, dtype=np.int64), constant_ranges


def _split_pair_indices_by_trajectory(
    *,
    num_pairs: int,
    pair_trajectory_id: torch.Tensor | None,
    val_fraction: float,
    test_fraction: float,
    seed: int,
    exclude_trajectory_ids: np.ndarray | None = None,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    """Split pair indices by trajectory ids when available.

    If `pair_trajectory_id` is present, train/val/test are split over unique
    trajectory ids and then expanded to pair indices. Otherwise split directly
    over pair indices.
    """
    if pair_trajectory_id is None:
        train_idx, val_idx, test_idx = split_indices(
            num_pairs,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            seed=seed,
        )
        return train_idx, val_idx, test_idx, None, None, None

    pair_traj_np = pair_trajectory_id.detach().cpu().numpy().astype(np.int64)
    if pair_traj_np.shape[0] != int(num_pairs):
        raise ValueError(
            "pair_trajectory_id length mismatch: "
            f"expected {num_pairs}, got {pair_traj_np.shape[0]}."
        )

    unique_traj = np.unique(pair_traj_np)
    train_ti, val_ti, test_ti = split_indices(
        int(unique_traj.shape[0]),
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )
    train_traj = unique_traj[train_ti]
    val_traj = unique_traj[val_ti]
    test_traj = unique_traj[test_ti]

    if exclude_trajectory_ids is not None and exclude_trajectory_ids.size > 0:
        excluded = np.asarray(exclude_trajectory_ids, dtype=np.int64)
        train_traj = np.setdiff1d(train_traj, excluded, assume_unique=False)
        val_traj = np.setdiff1d(val_traj, excluded, assume_unique=False)
        test_traj = np.setdiff1d(test_traj, excluded, assume_unique=False)

    train_idx = np.nonzero(np.isin(pair_traj_np, train_traj))[0].astype(np.int64)
    val_idx = np.nonzero(np.isin(pair_traj_np, val_traj))[0].astype(np.int64)
    test_idx = np.nonzero(np.isin(pair_traj_np, test_traj))[0].astype(np.int64)

    return train_idx, val_idx, test_idx, train_traj, val_traj, test_traj


def _run_split_rollout(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    rollout_states: torch.Tensor,
    pair_trajectory_id_all: torch.Tensor,
    pair_time_index_all: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    pos_nodes: torch.Tensor,
    dt: float,
    dx: float,
    c_from_data: float,
    physics_cfg: PhysicsInputConfig,
    input_cfg: InputFeatureConfig,
    godunov_cfg: GodunovCorrectionConfig,
    grad_loss_cfg: GradientWeightedLossConfig,
    relative_mse_cfg: RelativeMSELossConfig,
    mass_loss_cfg: MassConservationLossConfig,
    tv_loss_cfg: TVLossConfig,
    target_mode: str,
    state_residual_mode: str,
    rollout_steps: int,
    randomized_rollout_cfg: RandomizedRolloutLengthConfig,
    randomize_rollout_length: bool,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    grad_clip: float,
) -> EpochMetrics:
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_rmse = 0.0
    total_rel_l2 = 0.0
    total_rollout_state_rel_l2 = 0.0
    total_count = 0

    for batch_indices in loader:
        idx = batch_indices.long()
        traj_idx = pair_trajectory_id_all[idx]
        t0_idx = pair_time_index_all[idx]

        x_curr = rollout_states[traj_idx, t0_idx].to(device=device)
        bsz, n_nodes, n_feat = x_curr.shape
        ei_batched = build_batched_edge_index(
            edge_index=edge_index, num_nodes=n_nodes, batch_size=bsz, device=device
        )
        ea_batched = repeat_edge_attr(
            edge_attr=edge_attr,
            batch_size=bsz,
            device=device,
            dtype=x_curr.dtype,
        )

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        loss_accum = torch.zeros((), dtype=x_curr.dtype, device=device)
        rmse_accum = 0.0
        rel_accum = 0.0
        rollout_state_rel_accum = 0.0

        with torch.set_grad_enabled(is_train):
            steps_this_batch = int(rollout_steps)
            if randomize_rollout_length and randomized_rollout_cfg.enabled:
                max_t = int(rollout_states.shape[1] - 1)
                max_available = int(torch.min(max_t - t0_idx).item())
                lower = int(randomized_rollout_cfg.min_steps)
                cfg_upper = (
                    max_available
                    if randomized_rollout_cfg.max_steps is None
                    else int(randomized_rollout_cfg.max_steps)
                )
                upper = min(cfg_upper, max_available)
                if upper < lower:
                    raise ValueError(
                        "Invalid randomized rollout bounds for this batch: "
                        f"min_steps={lower}, max_available={max_available}, "
                        f"max_steps={randomized_rollout_cfg.max_steps}."
                    )
                steps_this_batch = int(np.random.randint(lower, upper + 1))

            for step in range(1, steps_this_batch + 1):
                true_next = rollout_states[traj_idx, t0_idx + step].to(device=device)

                x_flat = x_curr.reshape(bsz * n_nodes, n_feat)
                true_next_flat = true_next.reshape(bsz * n_nodes, n_feat)
                x_model_flat = build_physics_augmented_inputs_1d(
                    x_state=x_flat,
                    edge_index=ei_batched,
                    edge_attr=ea_batched,
                    dt=dt,
                    dx=dx,
                    c_from_data=c_from_data,
                    physics_cfg=physics_cfg,
                    step_k=step - 1,
                )
                x_model_flat = append_position_channel(
                    x_state=x_model_flat,
                    pos_nodes=pos_nodes,
                    input_cfg=input_cfg,
                    batch_size=bsz,
                )
                x_model_flat = apply_state_minmax_minus1_1(
                    x_model=x_model_flat,
                    input_cfg=input_cfg,
                )
                if godunov_cfg.enabled:
                    if target_mode == TARGET_MODE_STATE:
                        raise ValueError(
                            "godunov_correction currently requires target_mode in "
                            "{delta, rate}; state-mode correction is not supported."
                        )
                    god_state_flat = _godunov_next_state_flat(
                        x_now_flat=x_flat,
                        edge_index=ei_batched,
                        edge_attr=ea_batched,
                        dt=dt,
                        dx=dx,
                        c_from_data=c_from_data,
                        godunov_cfg=godunov_cfg,
                    )
                    god_target_flat = _target_from_states(
                        x_now_flat=x_flat,
                        state_next_flat=god_state_flat,
                        dt=dt,
                        target_mode=target_mode,
                    )
                    pred_corr_flat = model(
                        x_model_flat,
                        ei_batched,
                        edge_attr=ea_batched,
                        dt=dt,
                    )
                    pred_target_flat = god_target_flat + pred_corr_flat
                    pred_state_flat = _state_from_target(
                        x_now_flat=x_flat,
                        target_flat=pred_target_flat,
                        dt=dt,
                        target_mode=target_mode,
                    )
                else:
                    pred_state_flat = model.predict_state(
                        x_model_flat,
                        ei_batched,
                        edge_attr=ea_batched,
                        dt=dt,
                        state_residual=state_residual_from_mode(
                            x_raw=x_flat,
                            state_residual_mode=state_residual_mode,
                        ),
                    )
                    pred_target_flat = _target_from_states(
                        x_now_flat=x_flat,
                        state_next_flat=pred_state_flat,
                        dt=dt,
                        target_mode=target_mode,
                    )
                pred_state = pred_state_flat.reshape(bsz, n_nodes, n_feat)

                true_target_flat = _target_from_states(
                    x_now_flat=x_flat,
                    state_next_flat=true_next_flat,
                    dt=dt,
                    target_mode=target_mode,
                )

                step_loss_data = _data_fidelity_loss(
                    pred_target_flat=pred_target_flat,
                    true_target_flat=true_target_flat,
                    true_state_flat=true_next_flat,
                    batch_size=bsz,
                    num_nodes=n_nodes,
                    grad_loss_cfg=grad_loss_cfg,
                    relative_mse_cfg=relative_mse_cfg,
                )
                step_loss = step_loss_data
                if mass_loss_cfg.enabled:
                    mass_loss_step = _mass_conservation_penalty(
                        pred_state_flat=pred_state_flat,
                        ref_state_flat=x_flat,
                        batch_size=bsz,
                        num_nodes=n_nodes,
                    )
                    step_loss = step_loss + float(mass_loss_cfg.lambda_mass) * mass_loss_step
                if tv_loss_cfg.enabled:
                    tv_loss_step = _tv_regularization_penalty(
                        pred_state_flat=pred_state_flat,
                        ref_state_flat=true_next_flat,
                        batch_size=bsz,
                        num_nodes=n_nodes,
                        tv_loss_cfg=tv_loss_cfg,
                    )
                    step_loss = step_loss + float(tv_loss_cfg.lambda_tv) * tv_loss_step
                loss_accum = loss_accum + step_loss

                rmse_step, rel_step = _batch_metrics(
                    pred_target_flat.detach(), true_target_flat.detach()
                )
                rmse_accum += rmse_step
                rel_accum += rel_step
                rollout_state_rel_accum += _mean_sample_rel_l2(
                    pred_state_flat.detach(),
                    true_next_flat.detach(),
                    batch_size=bsz,
                    num_nodes=n_nodes,
                )

                x_curr = pred_state

            loss = loss_accum / float(steps_this_batch)

            if is_train:
                loss.backward()
                if grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()

        batch_n = int(bsz)
        total_loss += float(loss.item()) * batch_n
        total_rmse += (rmse_accum / float(steps_this_batch)) * batch_n
        total_rel_l2 += (rel_accum / float(steps_this_batch)) * batch_n
        total_rollout_state_rel_l2 += (
            rollout_state_rel_accum / float(steps_this_batch)
        ) * batch_n
        total_count += batch_n

    if total_count == 0:
        return EpochMetrics(loss=float("nan"), rmse=float("nan"), rel_l2=float("nan"))
    return EpochMetrics(
        loss=total_loss / total_count,
        rmse=total_rmse / total_count,
        rel_l2=total_rel_l2 / total_count,
        rollout_state_rel_l2=total_rollout_state_rel_l2 / total_count,
    )


def main() -> None:
    args = parse_args()
    cfg = {}
    with open(args.config, "r", encoding="utf-8") as f:
        import json

        cfg = json.load(f)
    cfg_input = deepcopy(cfg)

    seed = int(cfg.get("seed", 1234))
    set_seed(seed)

    device = resolve_device(str(cfg.get("device", "auto")))

    data_cfg = dict(cfg.get("data", {}))
    dataset_path = Path(data_cfg.get("dataset_path", "data/advection_train.npz"))
    target_mode = str(data_cfg.get("target_mode", TARGET_MODE_STATE)).strip().lower()
    if target_mode not in TARGET_MODES:
        raise ValueError(
            f"data.target_mode must be one of {sorted(TARGET_MODES)}, got '{target_mode}'."
        )
    resolved_predict_type = predict_type_from_target_mode(target_mode)

    cfg_model = dict(cfg.get("model", {}))
    old_predict_type = cfg_model.get("predict_type", None)
    if old_predict_type is not None:
        old_key = str(old_predict_type).strip().lower()
        if old_key != resolved_predict_type:
            print(
                "[INFO] Ignoring model.predict_type from config and using "
                f"data.target_mode='{target_mode}' -> predict_type='{resolved_predict_type}'.",
                flush=True,
            )
    cfg_model["predict_type"] = resolved_predict_type
    state_residual_mode = resolve_state_residual_mode(cfg)
    cfg_model["state_residual_mode"] = state_residual_mode
    cfg["model"] = cfg_model

    batch_size = int(data_cfg.get("batch_size", 32))
    num_workers = int(data_cfg.get("num_workers", 0))
    val_fraction = float(data_cfg.get("val_fraction", 0.1))
    test_fraction = float(data_cfg.get("test_fraction", 0.1))
    constant_trajectory_tol = float(data_cfg.get("constant_trajectory_tol", 0.0))
    split_cfg = data_cfg.get("split", {})
    if isinstance(split_cfg, dict):
        split_unit_cfg = split_cfg.get("unit", None)
        if split_unit_cfg is not None:
            key = str(split_unit_cfg).strip().lower()
            if key != "trajectory":
                raise ValueError(
                    "data.split.unit is for documentation only and must be "
                    f"'trajectory' when provided, got '{split_unit_cfg}'."
                )
        train_fraction_cfg = split_cfg.get("train_fraction", None)
        if train_fraction_cfg is not None:
            expected_train_fraction = 1.0 - val_fraction - test_fraction
            if abs(float(train_fraction_cfg) - expected_train_fraction) > 1e-12:
                raise ValueError(
                    "data.split.train_fraction must satisfy "
                    "train_fraction = 1 - val_fraction - test_fraction. "
                    f"Got train_fraction={train_fraction_cfg}, "
                    f"val_fraction={val_fraction}, test_fraction={test_fraction}."
                )

    dset = load_advection_dataset(dataset_path, dtype=torch.float32)
    x_all = dset.x.detach().cpu()
    u_next_all = dset.u_next.detach().cpu()
    edge_index = dset.edge_index.detach().cpu()
    edge_attr = dset.edge_attr.detach().cpu()
    pos_nodes = dset.pos.detach().cpu()
    n_samples = int(len(dset))
    n_nodes = int(dset.x.shape[1])
    dt = float(dset.metadata["dt"])
    dx = float(dset.metadata["dx"])
    c_from_data = float(dset.metadata["c"])
    physics_cfg = resolve_physics_input_cfg(cfg)
    input_cfg = resolve_input_feature_cfg(cfg)
    godunov_cfg = _resolve_godunov_correction_cfg(cfg)
    grad_loss_cfg = _resolve_gradient_weighted_loss_cfg(cfg)
    relative_mse_cfg = _resolve_relative_mse_loss_cfg(cfg)
    mass_loss_cfg = _resolve_mass_conservation_loss_cfg(cfg)
    tv_loss_cfg = _resolve_tv_loss_cfg(cfg)
    _sanity_check_gradient_weighted_loss_impl()
    _sanity_check_relative_mse_loss_impl()
    if godunov_cfg.enabled and (target_mode == TARGET_MODE_STATE):
        raise ValueError(
            "godunov_correction is enabled but data.target_mode='state'. "
            "Use target_mode='delta' or 'rate' for correction learning."
        )
    raw_input_features_cfg = cfg.get("input_features", {})
    cfg_input_features = (
        dict(raw_input_features_cfg)
        if isinstance(raw_input_features_cfg, dict)
        else {}
    )
    raw_physics_inputs_cfg = cfg.get("physics_inputs", {})
    cfg_physics_inputs = (
        dict(raw_physics_inputs_cfg)
        if isinstance(raw_physics_inputs_cfg, dict)
        else {}
    )
    train_cfg = dict(cfg.get("train", {}))
    rollout_steps_cfg = int(train_cfg.get("rollout_steps", 1))
    rollout_random_cfg = _resolve_rollout_random_length_cfg(cfg)
    if (not rollout_random_cfg.enabled) and rollout_steps_cfg < 1:
        raise ValueError(
            f"train.rollout_steps must be >= 1 when randomized rollout is disabled, got {rollout_steps_cfg}."
        )

    pair_trajectory_id_all = dset.pair_trajectory_id
    pair_time_index_all = dset.pair_time_index
    rollout_states_all = dset.rollout_states
    (
        constant_trajectory_ids,
        constant_trajectory_ranges,
    ) = _find_constant_trajectory_ids(
        x_all=x_all,
        u_next_all=u_next_all,
        pair_trajectory_id=pair_trajectory_id_all,
        rollout_states=rollout_states_all,
        atol=constant_trajectory_tol,
    )
    if pair_trajectory_id_all is not None:
        if constant_trajectory_ids.size > 0:
            ids_text = ", ".join(str(int(i)) for i in constant_trajectory_ids.tolist())
            print(
                "[WARN] Ignoring constant trajectories before training/evaluation: "
                f"ids=[{ids_text}], tol={constant_trajectory_tol:.3e}.",
                flush=True,
            )
        else:
            print(
                "[INFO] Constant trajectory guard: no constant trajectories detected.",
                flush=True,
            )
    (
        train_idx,
        val_idx,
        test_idx,
        train_traj_ids,
        val_traj_ids,
        test_traj_ids,
    ) = _split_pair_indices_by_trajectory(
        num_pairs=n_samples,
        pair_trajectory_id=pair_trajectory_id_all,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
        exclude_trajectory_ids=constant_trajectory_ids,
    )
    if (
        len(train_idx) == 0
        or (val_fraction > 0.0 and len(val_idx) == 0)
        or (test_fraction > 0.0 and len(test_idx) == 0)
    ):
        raise ValueError(
            "No usable samples remain in at least one requested split after "
            "constant trajectory filtering. Adjust split fractions or regenerate "
            "the dataset without constant trajectories."
        )
    split_unit = "trajectory" if pair_trajectory_id_all is not None else "pair"
    num_trajectories = (
        int(torch.unique(pair_trajectory_id_all).numel())
        if pair_trajectory_id_all is not None
        else None
    )

    rollout_mode = bool(rollout_random_cfg.enabled) or (rollout_steps_cfg > 1)
    early_stopping_metric = _resolve_early_stopping_metric(
        train_cfg,
        rollout_mode=rollout_mode,
    )
    train_cfg["early_stopping_metric"] = early_stopping_metric
    rollout_steps_train_max = int(rollout_steps_cfg if rollout_steps_cfg > 0 else 1)
    rollout_steps_eval_fixed = int(rollout_steps_cfg if rollout_steps_cfg > 0 else 1)
    train_required_steps = int(1)
    val_required_steps = int(1)
    train_effective_idx = train_idx
    val_effective_idx = val_idx
    test_effective_idx = test_idx

    if split_unit == "trajectory":
        assert train_traj_ids is not None
        assert val_traj_ids is not None
        assert test_traj_ids is not None
        print(
            "[INFO] Data split by trajectory ids: "
            f"num_trajectories={num_trajectories}, "
            f"train/val/test trajectories="
            f"{len(train_traj_ids)}/{len(val_traj_ids)}/{len(test_traj_ids)}.",
            flush=True,
        )
    else:
        print(
            "[INFO] Data split by pair indices (no trajectory metadata).",
            flush=True,
        )

    data_cfg["constant_trajectory_tol"] = float(constant_trajectory_tol)
    data_cfg["ignored_constant_trajectory_ids"] = (
        None
        if pair_trajectory_id_all is None
        else constant_trajectory_ids.astype(np.int64).tolist()
    )
    cfg["data"] = data_cfg

    if input_cfg.normalize_state_to_minus1_1:
        train_idx_t = torch.as_tensor(train_idx, dtype=torch.long)
        state_min, state_max = fit_state_minmax(
            x_state=x_all,
            sample_indices=train_idx_t,
        )
        input_cfg = InputFeatureConfig(
            include_pos=bool(input_cfg.include_pos),
            normalize_state_to_minus1_1=True,
            state_min=float(state_min),
            state_max=float(state_max),
            normalize_clip=bool(input_cfg.normalize_clip),
        )
        print(
            "[INFO] State normalization enabled: "
            f"range=[-1,1], state_min={state_min:.6e}, state_max={state_max:.6e}, "
            f"clip={input_cfg.normalize_clip}",
            flush=True,
        )
    else:
        print("[INFO] State normalization disabled.", flush=True)

    if (
        physics_cfg.enabled
        and physics_cfg.include_adv
        and physics_cfg.normalize_adv_to_minus1_1
    ):
        if physics_cfg.adv_min is None or physics_cfg.adv_max is None:
            train_idx_t = torch.as_tensor(train_idx, dtype=torch.long)
            adv_min, adv_max = fit_adv_input_minmax_1d(
                x_state=x_all,
                edge_index=edge_index,
                edge_attr=edge_attr,
                dt=dt,
                dx=dx,
                c_from_data=c_from_data,
                physics_cfg=physics_cfg,
                sample_indices=train_idx_t,
            )
            physics_cfg = replace(
                physics_cfg,
                adv_min=float(adv_min),
                adv_max=float(adv_max),
            )
            print(
                "[INFO] Advection normalization enabled: "
                f"range=[-1,1], adv_min={float(adv_min):.6e}, "
                f"adv_max={float(adv_max):.6e}, "
                f"clip={bool(physics_cfg.normalize_adv_clip)}",
                flush=True,
            )
        else:
            print(
                "[INFO] Advection normalization enabled: using provided stats "
                f"adv_min={float(physics_cfg.adv_min):.6e}, "
                f"adv_max={float(physics_cfg.adv_max):.6e}, "
                f"clip={bool(physics_cfg.normalize_adv_clip)}",
                flush=True,
            )
    elif physics_cfg.normalize_adv_to_minus1_1:
        print(
            "[INFO] Advection normalization requested but advection channel is not "
            "active; skipping advection normalization.",
            flush=True,
        )

    cfg_input_features["include_pos"] = bool(input_cfg.include_pos)
    cfg_input_features["normalize_state_to_minus1_1"] = bool(
        input_cfg.normalize_state_to_minus1_1
    )
    cfg_input_features["normalize_clip"] = bool(input_cfg.normalize_clip)
    cfg_input_features["state_min"] = (
        None if input_cfg.state_min is None else float(input_cfg.state_min)
    )
    cfg_input_features["state_max"] = (
        None if input_cfg.state_max is None else float(input_cfg.state_max)
    )
    cfg["input_features"] = cfg_input_features
    cfg_physics_inputs["enabled"] = bool(physics_cfg.enabled)
    cfg_physics_inputs["include_adv"] = bool(physics_cfg.include_adv)
    cfg_physics_inputs["include_diff"] = bool(physics_cfg.include_diff)
    cfg_physics_inputs["input_form"] = str(physics_cfg.input_form)
    cfg_physics_inputs["backend"] = str(physics_cfg.backend)
    cfg_physics_inputs["advection_scheme"] = str(physics_cfg.advection_scheme)
    cfg_physics_inputs["mls_ridge_eps"] = float(physics_cfg.mls_ridge_eps)
    cfg_physics_inputs["nu"] = float(physics_cfg.nu)
    cfg_physics_inputs["c"] = (
        None if physics_cfg.c_override is None else float(physics_cfg.c_override)
    )
    cfg_physics_inputs["adv_all_steps"] = bool(physics_cfg.adv_all_steps)
    cfg_physics_inputs["input_weighted"] = bool(physics_cfg.input_weighted)
    cfg_physics_inputs["adv_weight"] = float(physics_cfg.adv_weight)
    cfg_physics_inputs["diff_weight"] = float(physics_cfg.diff_weight)
    cfg_physics_inputs["detach_inputs"] = bool(physics_cfg.detach_inputs)
    cfg_physics_inputs["normalize_adv_to_minus1_1"] = bool(
        physics_cfg.normalize_adv_to_minus1_1
    )
    cfg_physics_inputs["adv_min"] = (
        None if physics_cfg.adv_min is None else float(physics_cfg.adv_min)
    )
    cfg_physics_inputs["adv_max"] = (
        None if physics_cfg.adv_max is None else float(physics_cfg.adv_max)
    )
    cfg_physics_inputs["normalize_adv_clip"] = bool(physics_cfg.normalize_adv_clip)
    cfg["physics_inputs"] = cfg_physics_inputs

    if rollout_mode:
        if (
            rollout_states_all is None
            or pair_trajectory_id_all is None
            or pair_time_index_all is None
        ):
            raise ValueError(
                "Rollout training requires rollout dataset metadata "
                "(rollout_states, pair_trajectory_id, pair_time_index)."
            )
        window = int(
            dset.metadata.get("window", int(rollout_states_all.shape[1] - 1))
        )
        if rollout_random_cfg.enabled:
            max_steps_eff = (
                int(window)
                if rollout_random_cfg.max_steps is None
                else int(rollout_random_cfg.max_steps)
            )
            if max_steps_eff > window:
                raise ValueError(
                    "train.rollout_random_length.max_steps exceeds dataset window: "
                    f"{max_steps_eff} > {window}."
                )
            if rollout_random_cfg.min_steps > max_steps_eff:
                raise ValueError(
                    "train.rollout_random_length.min_steps must be <= max_steps, got "
                    f"{rollout_random_cfg.min_steps} > {max_steps_eff}."
                )
            rollout_steps_train_max = int(max_steps_eff)
            rollout_steps_eval_fixed = int(max_steps_eff)
            train_required_steps = int(rollout_random_cfg.min_steps)
            val_required_steps = int(rollout_random_cfg.min_steps)
        else:
            if rollout_steps_cfg > window:
                raise ValueError(
                    f"train.rollout_steps={rollout_steps_cfg} exceeds dataset window={window}."
                )
            rollout_steps_train_max = int(rollout_steps_cfg)
            rollout_steps_eval_fixed = int(rollout_steps_cfg)
            train_required_steps = int(rollout_steps_cfg)
            val_required_steps = int(rollout_steps_cfg)

        train_effective_idx = _select_rollout_start_indices(
            split_indices_np=train_idx,
            pair_time_index_all=pair_time_index_all,
            window=window,
            rollout_steps=train_required_steps,
        )
        val_effective_idx = _select_rollout_start_indices(
            split_indices_np=val_idx,
            pair_time_index_all=pair_time_index_all,
            window=window,
            rollout_steps=val_required_steps,
        )
        test_effective_idx = _select_rollout_start_indices(
            split_indices_np=test_idx,
            pair_time_index_all=pair_time_index_all,
            window=window,
            rollout_steps=rollout_steps_eval_fixed,
        )
        if (
            len(train_effective_idx) == 0
            or len(val_effective_idx) == 0
            or len(test_effective_idx) == 0
        ):
            raise ValueError(
                "No valid rollout start samples after filtering split indices. "
                "Try smaller train.rollout_steps or adjust split fractions/dataset size."
            )
        print(
            "[INFO] Rollout training enabled: "
            f"rollout_steps_cfg={rollout_steps_cfg}, "
            f"randomized_lengths={rollout_random_cfg.enabled}, "
            f"train_max_steps={rollout_steps_train_max}, "
            f"val_scheme={'randomized' if rollout_random_cfg.enabled else 'fixed'}, "
            f"min_steps_for_starts(train/val)={train_required_steps}/{val_required_steps}, "
            f"test_fixed_steps={rollout_steps_eval_fixed}, "
            f"effective starts (train/val/test)="
            f"{len(train_effective_idx)}/{len(val_effective_idx)}/{len(test_effective_idx)}.",
            flush=True,
        )

    split_indices_payload = {
        "train": train_idx.tolist(),
        "val": val_idx.tolist(),
        "test": test_idx.tolist(),
        "train_effective": train_effective_idx.tolist(),
        "val_effective": val_effective_idx.tolist(),
        "test_effective": test_effective_idx.tolist(),
        "train_trajectory_ids": (
            None if train_traj_ids is None else train_traj_ids.tolist()
        ),
        "val_trajectory_ids": (
            None if val_traj_ids is None else val_traj_ids.tolist()
        ),
        "test_trajectory_ids": (
            None if test_traj_ids is None else test_traj_ids.tolist()
        ),
        "ignored_constant_trajectory_ids": (
            None
            if pair_trajectory_id_all is None
            else constant_trajectory_ids.astype(np.int64).tolist()
        ),
        "constant_trajectory_tol": float(constant_trajectory_tol),
        "constant_trajectory_ranges": (
            None
            if pair_trajectory_id_all is None
            else {str(k): float(v) for k, v in constant_trajectory_ranges.items()}
        ),
    }

    train_cfg["rollout_random_length"] = {
        "enabled": bool(rollout_random_cfg.enabled),
        "min_steps": int(rollout_random_cfg.min_steps),
        "max_steps": int(
            rollout_steps_train_max
            if rollout_random_cfg.enabled
            else (
                rollout_steps_cfg
                if rollout_random_cfg.max_steps is None
                else int(rollout_random_cfg.max_steps)
            )
        ),
    }
    train_cfg["tv_loss"] = {
        "enabled": bool(tv_loss_cfg.enabled),
        "lambda_tv": float(tv_loss_cfg.lambda_tv),
        "mode": str(tv_loss_cfg.mode),
    }
    train_cfg["relative_mse"] = {
        "enabled": bool(relative_mse_cfg.enabled),
        "alpha": float(relative_mse_cfg.alpha),
        "eps": float(relative_mse_cfg.eps),
    }
    cfg["train"] = train_cfg

    train_loader = DataLoader(
        IndexDataset(train_effective_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        IndexDataset(val_effective_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        IndexDataset(test_effective_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    in_channels = (
        int(x_all.shape[-1])
        + count_physics_input_channels(physics_cfg)
        + count_input_feature_channels(input_cfg)
    )
    print(
        "[INFO] Physics-input channels: "
        f"enabled={physics_cfg.enabled}, "
        f"backend={physics_cfg.backend}, "
        f"include_adv={physics_cfg.include_adv}, "
        f"include_diff={physics_cfg.include_diff}, "
        f"total_model_in_channels={in_channels}",
        flush=True,
    )
    print(
        "[INFO] Extra input channels: "
        f"include_pos={input_cfg.include_pos}, "
        f"normalize_state_to_minus1_1={input_cfg.normalize_state_to_minus1_1}, "
        f"total_model_in_channels={in_channels}",
        flush=True,
    )
    print(
        "[INFO] Godunov correction: "
        f"enabled={godunov_cfg.enabled}, "
        f"advection_scheme={godunov_cfg.advection_scheme}, "
        f"c_override={godunov_cfg.c_override}",
        flush=True,
    )
    print(
        "[INFO] State residual mode: "
        f"{state_residual_mode}",
        flush=True,
    )
    print(
        "[INFO] Gradient-weighted loss: "
        f"enabled={grad_loss_cfg.enabled}, "
        f"strength={grad_loss_cfg.strength:.3f}, "
        f"power={grad_loss_cfg.power:.3f}, "
        f"normalize_mean={grad_loss_cfg.normalize_mean}, "
        f"clip_max={grad_loss_cfg.clip_max}, "
        f"smooth_radius={grad_loss_cfg.smooth_radius}, "
        f"peak_spread_sigma={grad_loss_cfg.peak_spread_sigma}, "
        f"peak_spread_mode={grad_loss_cfg.peak_spread_mode}",
        flush=True,
    )
    print(
        "[INFO] Mass-conservation loss: "
        f"enabled={mass_loss_cfg.enabled}, "
        f"lambda_mass={mass_loss_cfg.lambda_mass:.3e}",
        flush=True,
    )
    print(
        "[INFO] Relative-MSE loss: "
        f"enabled={relative_mse_cfg.enabled}, "
        f"alpha={relative_mse_cfg.alpha:.3f}, "
        f"eps={relative_mse_cfg.eps:.3e}",
        flush=True,
    )
    print(
        "[INFO] TV loss: "
        f"enabled={tv_loss_cfg.enabled}, "
        f"lambda_tv={tv_loss_cfg.lambda_tv:.3e}, "
        f"mode={tv_loss_cfg.mode}",
        flush=True,
    )
    print(
        "[INFO] Randomized rollout length: "
        f"enabled={rollout_random_cfg.enabled}, "
        f"min_steps={rollout_random_cfg.min_steps}, "
        f"max_steps={train_cfg['rollout_random_length']['max_steps']}",
        flush=True,
    )
    edge_attr_channels = int(edge_attr.shape[-1])
    model = build_model_from_config(
        cfg,
        in_channels=in_channels,
        edge_attr_channels=edge_attr_channels,
    ).to(device)
    epochs = int(train_cfg.get("epochs", 200))
    lr = float(train_cfg.get("lr", 1e-3))
    lr_initial = float(lr)
    weight_decay = float(train_cfg.get("weight_decay", 1e-6))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    raw_patience = train_cfg.get("early_stopping_patience", 30)
    patience: int | None
    if raw_patience is None:
        patience = None
    elif isinstance(raw_patience, str):
        key = raw_patience.strip().lower()
        if key in {"none", "null", "off", "disable", "disabled"}:
            patience = None
        else:
            patience = int(raw_patience)
            if patience <= 0:
                patience = None
    else:
        patience = int(raw_patience)
        if patience <= 0:
            patience = None
    log_every = int(train_cfg.get("log_every", 10))
    save_dir = Path(train_cfg.get("save_dir", "runs/advection_sage_state"))
    save_dir.mkdir(parents=True, exist_ok=True)
    config_input_path = save_dir / "config_input.json"
    config_resolved_path = save_dir / "config_resolved.json"
    save_json(cfg_input, config_input_path)
    save_json(cfg, config_resolved_path)
    print(
        "[INFO] Saved run configs: "
        f"input={config_input_path}, resolved={config_resolved_path}",
        flush=True,
    )
    lr_schedule_enabled, lr_schedule = _parse_lr_schedule(train_cfg, epochs=epochs)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if not lr_schedule_enabled:
        print("[INFO] LR schedule disabled.", flush=True)
    elif len(lr_schedule) == 0:
        print("[INFO] LR schedule enabled but no entries provided.", flush=True)
    else:
        schedule_str = ", ".join(
            [f"epoch {ep}: {val:.6e}" for ep, val in sorted(lr_schedule.items())]
        )
        print(f"[INFO] LR schedule: {schedule_str}", flush=True)

    if patience is None:
        print("[INFO] Early stopping disabled.", flush=True)
    print(
        "[INFO] Best-checkpoint/early-stopping metric: "
        f"{early_stopping_metric} (lower is better).",
        flush=True,
    )
    history: List[Dict[str, float]] = []
    best_metric_value = float("inf")
    best_checkpoint_val_loss = float("inf")
    best_epoch = -1
    stale = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        if lr_schedule_enabled and (epoch in lr_schedule):
            lr_new = float(lr_schedule[epoch])
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_new
            print(
                f"[INFO] LR schedule kick-in at epoch {epoch}: lr={lr_new:.6e}",
                flush=True,
            )

        current_lr = float(optimizer.param_groups[0]["lr"])
        epoch_t0 = time.time()
        if rollout_mode:
            assert rollout_states_all is not None
            assert pair_trajectory_id_all is not None
            assert pair_time_index_all is not None
            train_m = _run_split_rollout(
                model=model,
                loader=train_loader,
                rollout_states=rollout_states_all,
                pair_trajectory_id_all=pair_trajectory_id_all,
                pair_time_index_all=pair_time_index_all,
                edge_index=edge_index,
                edge_attr=edge_attr,
                pos_nodes=pos_nodes,
                dt=dt,
                dx=dx,
                c_from_data=c_from_data,
                physics_cfg=physics_cfg,
                input_cfg=input_cfg,
                godunov_cfg=godunov_cfg,
                grad_loss_cfg=grad_loss_cfg,
                relative_mse_cfg=relative_mse_cfg,
                mass_loss_cfg=mass_loss_cfg,
                tv_loss_cfg=tv_loss_cfg,
                target_mode=target_mode,
                state_residual_mode=state_residual_mode,
                rollout_steps=rollout_steps_train_max,
                randomized_rollout_cfg=rollout_random_cfg,
                randomize_rollout_length=bool(rollout_random_cfg.enabled),
                optimizer=optimizer,
                device=device,
                grad_clip=grad_clip,
            )
            val_m = _run_split_rollout(
                model=model,
                loader=val_loader,
                rollout_states=rollout_states_all,
                pair_trajectory_id_all=pair_trajectory_id_all,
                pair_time_index_all=pair_time_index_all,
                edge_index=edge_index,
                edge_attr=edge_attr,
                pos_nodes=pos_nodes,
                dt=dt,
                dx=dx,
                c_from_data=c_from_data,
                physics_cfg=physics_cfg,
                input_cfg=input_cfg,
                godunov_cfg=godunov_cfg,
                grad_loss_cfg=grad_loss_cfg,
                relative_mse_cfg=relative_mse_cfg,
                mass_loss_cfg=mass_loss_cfg,
                tv_loss_cfg=tv_loss_cfg,
                target_mode=target_mode,
                state_residual_mode=state_residual_mode,
                rollout_steps=rollout_steps_train_max,
                randomized_rollout_cfg=rollout_random_cfg,
                randomize_rollout_length=bool(rollout_random_cfg.enabled),
                optimizer=None,
                device=device,
                grad_clip=grad_clip,
            )
        else:
            train_m = _run_split(
                model=model,
                loader=train_loader,
                x_all=x_all,
                u_next_all=u_next_all,
                edge_index=edge_index,
                edge_attr=edge_attr,
                pos_nodes=pos_nodes,
                dt=dt,
                dx=dx,
                c_from_data=c_from_data,
                physics_cfg=physics_cfg,
                input_cfg=input_cfg,
                godunov_cfg=godunov_cfg,
                grad_loss_cfg=grad_loss_cfg,
                relative_mse_cfg=relative_mse_cfg,
                mass_loss_cfg=mass_loss_cfg,
                tv_loss_cfg=tv_loss_cfg,
                target_mode=target_mode,
                state_residual_mode=state_residual_mode,
                optimizer=optimizer,
                device=device,
                grad_clip=grad_clip,
            )
            val_m = _run_split(
                model=model,
                loader=val_loader,
                x_all=x_all,
                u_next_all=u_next_all,
                edge_index=edge_index,
                edge_attr=edge_attr,
                pos_nodes=pos_nodes,
                dt=dt,
                dx=dx,
                c_from_data=c_from_data,
                physics_cfg=physics_cfg,
                input_cfg=input_cfg,
                godunov_cfg=godunov_cfg,
                grad_loss_cfg=grad_loss_cfg,
                relative_mse_cfg=relative_mse_cfg,
                mass_loss_cfg=mass_loss_cfg,
                tv_loss_cfg=tv_loss_cfg,
                target_mode=target_mode,
                state_residual_mode=state_residual_mode,
                optimizer=None,
                device=device,
                grad_clip=grad_clip,
            )

        epoch_time_s = time.time() - epoch_t0

        checkpoint_metric_value = _checkpoint_metric_value(
            val_m,
            early_stopping_metric,
        )
        row = {
            "epoch": float(epoch),
            "lr": current_lr,
            "epoch_time_s": epoch_time_s,
            "train_loss": train_m.loss,
            "train_rmse": train_m.rmse,
            "train_rel_l2": train_m.rel_l2,
            "val_loss": val_m.loss,
            "val_rmse": val_m.rmse,
            "val_rel_l2": val_m.rel_l2,
            "checkpoint_metric_value": checkpoint_metric_value,
        }
        if train_m.rollout_state_rel_l2 is not None:
            row["train_rollout_state_rel_l2"] = train_m.rollout_state_rel_l2
        if val_m.rollout_state_rel_l2 is not None:
            row["val_rollout_state_rel_l2"] = val_m.rollout_state_rel_l2
        history.append(row)

        improved = checkpoint_metric_value < best_metric_value
        if improved:
            best_metric_value = checkpoint_metric_value
            best_checkpoint_val_loss = val_m.loss
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": cfg,
                    "metadata": dict(dset.metadata),
                    "split_indices": split_indices_payload,
                    "epoch": epoch,
                    "checkpoint_val_loss": best_checkpoint_val_loss,
                    # Backward-compatible name; when another selection metric is
                    # used, this is the validation loss at the selected epoch.
                    "best_val_loss": best_checkpoint_val_loss,
                    "early_stopping_metric": early_stopping_metric,
                    "best_early_stopping_metric_value": best_metric_value,
                    "val_rollout_state_rel_l2": val_m.rollout_state_rel_l2,
                },
                save_dir / "checkpoint_best.pt",
            )
        else:
            stale += 1

        if (epoch == 1) or (epoch % log_every == 0) or improved:
            log_line = (
                f"[epoch {epoch:04d}] "
                f"lr={current_lr:.6e} "
                f"train_loss={train_m.loss:.6e} val_loss={val_m.loss:.6e} "
                f"train_rmse={train_m.rmse:.6e} val_rmse={val_m.rmse:.6e}"
            )
            if (
                train_m.rollout_state_rel_l2 is not None
                and val_m.rollout_state_rel_l2 is not None
            ):
                log_line += (
                    " "
                    f"train_rollout_state_rel_l2="
                    f"{train_m.rollout_state_rel_l2:.6e} "
                    f"val_rollout_state_rel_l2="
                    f"{val_m.rollout_state_rel_l2:.6e}"
                )
            log_line += f" epoch_time={epoch_time_s:.2f}s"
            print(log_line, flush=True)

        if patience is not None and stale >= patience:
            print(
                f"Early stopping at epoch {epoch} (no {early_stopping_metric} "
                f"improvement for {patience} epochs).",
                flush=True,
            )
            break

    # Load best model for final test metrics.
    best_ckpt = torch.load(save_dir / "checkpoint_best.pt", map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    if rollout_mode:
        assert rollout_states_all is not None
        assert pair_trajectory_id_all is not None
        assert pair_time_index_all is not None
        test_m = _run_split_rollout(
            model=model,
            loader=test_loader,
            rollout_states=rollout_states_all,
            pair_trajectory_id_all=pair_trajectory_id_all,
            pair_time_index_all=pair_time_index_all,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos_nodes=pos_nodes,
            dt=dt,
            dx=dx,
            c_from_data=c_from_data,
            physics_cfg=physics_cfg,
            input_cfg=input_cfg,
            godunov_cfg=godunov_cfg,
            grad_loss_cfg=grad_loss_cfg,
            relative_mse_cfg=relative_mse_cfg,
            mass_loss_cfg=mass_loss_cfg,
            tv_loss_cfg=tv_loss_cfg,
            target_mode=target_mode,
            state_residual_mode=state_residual_mode,
            rollout_steps=rollout_steps_eval_fixed,
            randomized_rollout_cfg=rollout_random_cfg,
            randomize_rollout_length=False,
            optimizer=None,
            device=device,
            grad_clip=grad_clip,
        )
    else:
        test_m = _run_split(
            model=model,
            loader=test_loader,
            x_all=x_all,
            u_next_all=u_next_all,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos_nodes=pos_nodes,
            dt=dt,
            dx=dx,
            c_from_data=c_from_data,
            physics_cfg=physics_cfg,
            input_cfg=input_cfg,
            godunov_cfg=godunov_cfg,
            grad_loss_cfg=grad_loss_cfg,
            relative_mse_cfg=relative_mse_cfg,
            mass_loss_cfg=mass_loss_cfg,
            tv_loss_cfg=tv_loss_cfg,
            target_mode=target_mode,
            state_residual_mode=state_residual_mode,
            optimizer=None,
            device=device,
            grad_clip=grad_clip,
        )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": cfg,
            "metadata": dict(dset.metadata),
            "split_indices": split_indices_payload,
            "epoch": len(history),
            "best_epoch": best_epoch,
            "checkpoint_val_loss": best_checkpoint_val_loss,
            "best_val_loss": best_checkpoint_val_loss,
            "early_stopping_metric": early_stopping_metric,
            "best_early_stopping_metric_value": best_metric_value,
            "test_metrics": test_m.to_dict(),
        },
        save_dir / "checkpoint_last.pt",
    )

    summary: Dict[str, Any] = {
        "run_seconds": float(time.time() - t0),
        "device": str(device),
        "config_path": str(args.config),
        "config_input_path": str(config_input_path),
        "config_resolved_path": str(config_resolved_path),
        "dataset_path": str(dataset_path),
        "target_mode": target_mode,
        "state_residual_mode": str(state_residual_mode),
        "lr_initial": float(lr_initial),
        "lr_final": float(optimizer.param_groups[0]["lr"]),
        "lr_schedule_enabled": bool(lr_schedule_enabled),
        "lr_schedule": [
            {"epoch": int(ep), "lr": float(val)}
            for ep, val in sorted(lr_schedule.items())
        ],
        "split_unit": split_unit,
        "training_mode": "rollout" if rollout_mode else "one_step",
        "rollout_steps": int(rollout_steps_eval_fixed),
        "rollout_steps_config": int(rollout_steps_cfg),
        "rollout_steps_train_max": int(rollout_steps_train_max),
        "rollout_steps_val_mode": (
            "randomized" if rollout_random_cfg.enabled else "fixed"
        ),
        "rollout_random_length": {
            "enabled": bool(rollout_random_cfg.enabled),
            "min_steps": int(rollout_random_cfg.min_steps),
            "max_steps": int(train_cfg["rollout_random_length"]["max_steps"]),
        },
        "num_samples": n_samples,
        "num_pairs": int(n_samples),
        "num_trajectories": (
            None if num_trajectories is None else int(num_trajectories)
        ),
        "num_constant_trajectories_ignored": int(constant_trajectory_ids.size),
        "ignored_constant_trajectory_ids": (
            None
            if pair_trajectory_id_all is None
            else constant_trajectory_ids.astype(np.int64).tolist()
        ),
        "constant_trajectory_tol": float(constant_trajectory_tol),
        "num_nodes": n_nodes,
        "model_in_channels": int(in_channels),
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "test_size": int(len(test_idx)),
        "train_effective_size": int(len(train_effective_idx)),
        "val_effective_size": int(len(val_effective_idx)),
        "test_effective_size": int(len(test_effective_idx)),
        "physics_inputs": {
            "enabled": bool(physics_cfg.enabled),
            "include_adv": bool(physics_cfg.include_adv),
            "include_diff": bool(physics_cfg.include_diff),
            "input_form": str(physics_cfg.input_form),
            "backend": str(physics_cfg.backend),
            "advection_scheme": str(physics_cfg.advection_scheme),
            "mls_ridge_eps": float(physics_cfg.mls_ridge_eps),
            "nu": float(physics_cfg.nu),
            "c_override": (
                None if physics_cfg.c_override is None else float(physics_cfg.c_override)
            ),
            "adv_all_steps": bool(physics_cfg.adv_all_steps),
            "input_weighted": bool(physics_cfg.input_weighted),
            "adv_weight": float(physics_cfg.adv_weight),
            "diff_weight": float(physics_cfg.diff_weight),
            "detach_inputs": bool(physics_cfg.detach_inputs),
            "normalize_adv_to_minus1_1": bool(
                physics_cfg.normalize_adv_to_minus1_1
            ),
            "adv_min": (
                None if physics_cfg.adv_min is None else float(physics_cfg.adv_min)
            ),
            "adv_max": (
                None if physics_cfg.adv_max is None else float(physics_cfg.adv_max)
            ),
            "normalize_adv_clip": bool(physics_cfg.normalize_adv_clip),
        },
        "godunov_correction": {
            "enabled": bool(godunov_cfg.enabled),
            "advection_scheme": str(godunov_cfg.advection_scheme),
            "c_override": (
                None if godunov_cfg.c_override is None else float(godunov_cfg.c_override)
            ),
        },
        "gradient_weighted_loss": {
            "enabled": bool(grad_loss_cfg.enabled),
            "strength": float(grad_loss_cfg.strength),
            "power": float(grad_loss_cfg.power),
            "eps": float(grad_loss_cfg.eps),
            "normalize_mean": bool(grad_loss_cfg.normalize_mean),
            "clip_max": (
                None if grad_loss_cfg.clip_max is None else float(grad_loss_cfg.clip_max)
            ),
            "smooth_radius": int(grad_loss_cfg.smooth_radius),
            "peak_spread_sigma": float(grad_loss_cfg.peak_spread_sigma),
            "peak_spread_mode": str(grad_loss_cfg.peak_spread_mode),
        },
        "relative_mse": {
            "enabled": bool(relative_mse_cfg.enabled),
            "alpha": float(relative_mse_cfg.alpha),
            "eps": float(relative_mse_cfg.eps),
        },
        "mass_conservation_loss": {
            "enabled": bool(mass_loss_cfg.enabled),
            "lambda_mass": float(mass_loss_cfg.lambda_mass),
        },
        "tv_loss": {
            "enabled": bool(tv_loss_cfg.enabled),
            "lambda_tv": float(tv_loss_cfg.lambda_tv),
            "mode": str(tv_loss_cfg.mode),
        },
        "input_features": {
            "include_pos": bool(input_cfg.include_pos),
            "normalize_state_to_minus1_1": bool(
                input_cfg.normalize_state_to_minus1_1
            ),
            "normalize_clip": bool(input_cfg.normalize_clip),
            "state_min": (
                None if input_cfg.state_min is None else float(input_cfg.state_min)
            ),
            "state_max": (
                None if input_cfg.state_max is None else float(input_cfg.state_max)
            ),
        },
        "best_epoch": int(best_epoch),
        "checkpoint_val_loss": float(best_checkpoint_val_loss),
        "best_val_loss": float(best_checkpoint_val_loss),
        "early_stopping_metric": early_stopping_metric,
        "best_early_stopping_metric_value": float(best_metric_value),
        "test_metrics": test_m.to_dict(),
    }
    save_json(summary, save_dir / "summary.json")
    save_json({"history": history}, save_dir / "history.json")
    save_json(split_indices_payload, save_dir / "split_indices.json")

    print("Training complete.")
    print(f"  save_dir: {save_dir}")
    print(
        f"  training_mode: {'rollout' if rollout_mode else 'one_step'} "
        f"(rollout_steps_eval={rollout_steps_eval_fixed}, "
        f"rollout_steps_train_max={rollout_steps_train_max})"
    )
    print(f"  best_epoch: {best_epoch}")
    print(f"  early_stopping_metric: {early_stopping_metric}")
    print(f"  best_early_stopping_metric_value: {best_metric_value:.6e}")
    print(f"  best_checkpoint_val_loss: {best_checkpoint_val_loss:.6e}")
    print(f"  test_loss: {test_m.loss:.6e}")
    print(f"  test_rmse: {test_m.rmse:.6e}")
    print(f"  test_rel_l2: {test_m.rel_l2:.6e}")


if __name__ == "__main__":
    main()
