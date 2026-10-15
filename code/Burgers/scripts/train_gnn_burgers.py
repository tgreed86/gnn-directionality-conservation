#!/usr/bin/env python3
"""Train a SAGEConv GNN surrogate for the provided 1D Burgers dataset.

Typical usage from this project root:

    python scripts/train_gnn_burgers.py --config config_gnn_burgers.json

The loader expects the supplied split files:
  - train_samples.npz
  - val_samples.npz
  - test_samples.npz
and uses the split exactly as provided.

Alternatively, set `data.source_format="wenoa_pt"` and point
`data.wenoa_pt_path` (or `data.data_dir`) to a WENO-A `.pt` trajectory file.
The loader will convert it to GNN tensors in memory and create deterministic
train/val/test trajectory splits.

If `train.rollout_steps > 1`, training switches to autoregressive rollout loss
using per-trajectory states from `trajectories/traj_*.npz`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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

from burgers_data import filter_burgers_dataset_by_regime, load_burgers_dataset
from gnn_burgers import (
    TARGET_MODE_DELTA,
    TARGET_MODE_RATE,
    TARGET_MODE_STATE,
    TARGET_MODES,
    EpochMetrics,
    build_batched_edge_index,
    build_model_from_config,
    make_target,
    predict_type_from_target_mode,
    resolve_device,
    save_json,
    set_seed,
)
from input_features import (
    InputFeatureConfig,
    apply_u_minmax_minus1_1,
    count_input_feature_channels,
    fit_u_minmax,
    resolve_input_feature_cfg,
    select_feature_channels,
    u_column_index,
)
from physics_inputs import (
    PhysicsInputConfig,
    build_physics_augmented_inputs_1d,
    compute_adv_input_channel_1d,
    count_physics_input_channels,
    resolve_physics_input_cfg,
)


class IndexDataset(Dataset):
    """Simple dataset returning precomputed sample indices."""

    def __init__(self, indices: np.ndarray) -> None:
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, idx: int) -> int:
        return int(self.indices[idx])


@dataclass(frozen=True)
class GradientWeightedLossConfig:
    """Optional shock-focused weighting for data MSE loss."""

    enabled: bool = False
    strength: float = 1.0
    power: float = 1.0
    eps: float = 1e-12
    normalize_mean: bool = True
    clip_max: float | None = None
    smooth_radius: int = 0
    peak_spread_sigma: float = 0.0
    peak_spread_mode: str = "gaussian"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SAGE GNN for provided 1D Burgers dataset."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config_gnn_burgers.json"),
        help="Path to JSON config.",
    )
    return parser.parse_args()


def _parse_lr_schedule(
    train_cfg: Mapping[str, Any],
    *,
    epochs: int,
) -> Tuple[bool, Dict[int, float]]:
    """Parse optional epoch-based learning-rate schedule from config."""
    raw = train_cfg.get("lr_schedule", None)
    if raw is None:
        return False, {}

    entries: List[Mapping[str, Any]]
    enabled = True
    if isinstance(raw, Mapping):
        enabled = bool(raw.get("enabled", True))
        if ("epoch" in raw) or ("lr" in raw):
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
            "gradient_weighted_loss.clip_max must be > 0 when provided, got "
            f"{clip_max}."
        )
    if smooth_radius < 0:
        raise ValueError(
            "gradient_weighted_loss.smooth_radius must be >= 0, got "
            f"{smooth_radius}."
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


def _periodic_mean_filter_1d(values: torch.Tensor, radius: int) -> torch.Tensor:
    """Apply periodic moving-average smoothing across the node dimension."""
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
    """Apply periodic Gaussian smoothing across nodes."""
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


def _run_split(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    x_full_all: torch.Tensor,
    u_next_all: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    dt: float,
    dx: float,
    input_cfg: InputFeatureConfig,
    physics_cfg: PhysicsInputConfig,
    grad_loss_cfg: GradientWeightedLossConfig,
    target_mode: str,
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
        x_full_b = x_full_all[idx].to(device=device)
        u_next_b = u_next_all[idx].to(device=device)
        x_state_b = x_full_b[..., 3:4]
        target_b = make_target(x_state_b, u_next_b, dt=dt, target_mode=target_mode)

        bsz, n_nodes, _ = x_full_b.shape
        x_model_b = select_feature_channels(x_full=x_full_b, input_cfg=input_cfg)
        model_feat = int(x_model_b.shape[-1])

        x_model_flat = x_model_b.reshape(bsz * n_nodes, model_feat)
        x_state_flat = x_state_b.reshape(bsz * n_nodes, 1)
        true_state_flat = u_next_b.reshape(bsz * n_nodes, 1)
        target_flat = target_b.reshape(bsz * n_nodes, 1)

        ei_batched = build_batched_edge_index(
            edge_index=edge_index,
            num_nodes=n_nodes,
            batch_size=bsz,
            device=device,
        )
        ea_batched = (
            edge_attr.to(device=device, dtype=x_state_flat.dtype)
            .repeat(bsz, 1)
        )

        x_model_flat = build_physics_augmented_inputs_1d(
            x_base=x_model_flat,
            x_state=x_state_flat,
            edge_index=ei_batched,
            edge_attr=ea_batched,
            dt=dt,
            dx=dx,
            physics_cfg=physics_cfg,
            step_k=0,
        )
        x_model_flat = apply_u_minmax_minus1_1(x_model=x_model_flat, input_cfg=input_cfg)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        pred_state_flat = model.predict_state(
            x_model_flat,
            ei_batched,
            edge_attr=ea_batched,
            dt=dt,
            state_override=x_state_flat,
        )

        if target_mode == TARGET_MODE_STATE:
            pred_target_flat = pred_state_flat
        elif target_mode == TARGET_MODE_DELTA:
            pred_target_flat = pred_state_flat - x_state_flat
        elif target_mode == TARGET_MODE_RATE:
            pred_target_flat = (pred_state_flat - x_state_flat) / float(dt)
        else:
            raise RuntimeError(f"Unexpected target_mode='{target_mode}'.")

        loss = _gradient_weighted_mse_loss(
            pred_target_flat=pred_target_flat,
            true_target_flat=target_flat,
            true_state_flat=true_state_flat,
            batch_size=bsz,
            num_nodes=n_nodes,
            loss_cfg=grad_loss_cfg,
        )

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
    pair_time_index_all: torch.Tensor,
    window: int,
    rollout_steps: int,
) -> np.ndarray:
    if rollout_steps < 1:
        raise ValueError(f"rollout_steps must be >= 1, got {rollout_steps}.")
    if window < rollout_steps:
        return np.zeros((0,), dtype=np.int64)
    max_start_t = int(window - rollout_steps)
    t = pair_time_index_all.detach().cpu().numpy()
    keep = t <= max_start_t
    return np.nonzero(keep)[0].astype(np.int64)


def _run_split_rollout(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    x_full_all: torch.Tensor,
    rollout_states: torch.Tensor,
    pair_trajectory_id_all: torch.Tensor,
    pair_time_index_all: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    dt: float,
    dx: float,
    input_cfg: InputFeatureConfig,
    physics_cfg: PhysicsInputConfig,
    grad_loss_cfg: GradientWeightedLossConfig,
    target_mode: str,
    rollout_steps: int,
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
        traj_idx = pair_trajectory_id_all[idx]
        t0_idx = pair_time_index_all[idx]

        x_base = x_full_all[idx].to(device=device)
        x_curr_state = rollout_states[traj_idx, t0_idx].to(device=device)

        bsz, n_nodes, _ = x_base.shape
        ei_batched = build_batched_edge_index(
            edge_index=edge_index,
            num_nodes=n_nodes,
            batch_size=bsz,
            device=device,
        )
        ea_batched = (
            edge_attr.to(device=device, dtype=x_curr_state.dtype)
            .repeat(bsz, 1)
        )

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        loss_accum = torch.zeros((), dtype=x_base.dtype, device=device)
        rmse_accum = 0.0
        rel_accum = 0.0

        with torch.set_grad_enabled(is_train):
            for step in range(1, rollout_steps + 1):
                true_next = rollout_states[traj_idx, t0_idx + step].to(device=device)

                x_full_step = x_base.clone()
                x_full_step[..., 3:4] = x_curr_state

                x_model_step = select_feature_channels(x_full=x_full_step, input_cfg=input_cfg)
                feat_dim = int(x_model_step.shape[-1])
                x_model_flat = x_model_step.reshape(bsz * n_nodes, feat_dim)

                x_state_flat = x_curr_state.reshape(bsz * n_nodes, 1)
                x_model_flat = build_physics_augmented_inputs_1d(
                    x_base=x_model_flat,
                    x_state=x_state_flat,
                    edge_index=ei_batched,
                    edge_attr=ea_batched,
                    dt=dt,
                    dx=dx,
                    physics_cfg=physics_cfg,
                    step_k=step - 1,
                )
                x_model_flat = apply_u_minmax_minus1_1(
                    x_model=x_model_flat,
                    input_cfg=input_cfg,
                )

                true_next_flat = true_next.reshape(bsz * n_nodes, 1)

                pred_state_flat = model.predict_state(
                    x_model_flat,
                    ei_batched,
                    edge_attr=ea_batched,
                    dt=dt,
                    state_override=x_state_flat,
                )
                pred_state = pred_state_flat.reshape(bsz, n_nodes, 1)

                pred_target_flat = _target_from_states(
                    x_now_flat=x_state_flat,
                    state_next_flat=pred_state_flat,
                    dt=dt,
                    target_mode=target_mode,
                )
                true_target_flat = _target_from_states(
                    x_now_flat=x_state_flat,
                    state_next_flat=true_next_flat,
                    dt=dt,
                    target_mode=target_mode,
                )

                step_loss = _gradient_weighted_mse_loss(
                    pred_target_flat=pred_target_flat,
                    true_target_flat=true_target_flat,
                    true_state_flat=true_next_flat,
                    batch_size=bsz,
                    num_nodes=n_nodes,
                    loss_cfg=grad_loss_cfg,
                )
                loss_accum = loss_accum + step_loss

                rmse_step, rel_step = _batch_metrics(
                    pred_target_flat.detach(), true_target_flat.detach()
                )
                rmse_accum += rmse_step
                rel_accum += rel_step

                x_curr_state = pred_state

            loss = loss_accum / float(rollout_steps)

            if is_train:
                loss.backward()
                if grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()

        batch_n = int(bsz)
        total_loss += float(loss.item()) * batch_n
        total_rmse += (rmse_accum / float(rollout_steps)) * batch_n
        total_rel_l2 += (rel_accum / float(rollout_steps)) * batch_n
        total_count += batch_n

    if total_count == 0:
        return EpochMetrics(loss=float("nan"), rmse=float("nan"), rel_l2=float("nan"))
    return EpochMetrics(
        loss=total_loss / total_count,
        rmse=total_rmse / total_count,
        rel_l2=total_rel_l2 / total_count,
    )


def _materialize_input_cfg(
    *,
    cfg: Dict[str, Any],
    x_train_full: torch.Tensor,
    source_format: str,
) -> InputFeatureConfig:
    input_cfg = resolve_input_feature_cfg(cfg, source_format=source_format)
    raw = cfg.get("input_features", {})
    cfg_input = dict(raw) if isinstance(raw, dict) else {}

    if input_cfg.normalize_u_to_minus1_1:
        u_min, u_max = fit_u_minmax(x_full=x_train_full, sample_indices=None)
        input_cfg = InputFeatureConfig(
            feature_columns=tuple(input_cfg.feature_columns),
            use_pos=bool(input_cfg.use_pos),
            normalize_u_to_minus1_1=True,
            u_min=float(u_min),
            u_max=float(u_max),
            normalize_clip=bool(input_cfg.normalize_clip),
        )
        print(
            "[INFO] u normalization enabled: "
            f"range=[-1,1], u_min={u_min:.6e}, u_max={u_max:.6e}, "
            f"clip={input_cfg.normalize_clip}",
            flush=True,
        )
    else:
        print("[INFO] u normalization disabled.", flush=True)

    cfg_input["feature_columns"] = list(input_cfg.feature_columns)
    cfg_input["use_pos"] = bool(input_cfg.use_pos)
    cfg_input["normalize_u_to_minus1_1"] = bool(input_cfg.normalize_u_to_minus1_1)
    cfg_input["normalize_clip"] = bool(input_cfg.normalize_clip)
    cfg_input["u_min"] = None if input_cfg.u_min is None else float(input_cfg.u_min)
    cfg_input["u_max"] = None if input_cfg.u_max is None else float(input_cfg.u_max)
    cfg["input_features"] = cfg_input

    return input_cfg


def _fit_adv_minmax_from_train_pairs(
    *,
    x_train_full: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    dt: float,
    dx: float,
    physics_cfg: PhysicsInputConfig,
    batch_size: int = 128,
) -> tuple[float, float]:
    """Fit advection-channel normalization range from training pairs only.

    Uses configured train-percentile bounds (low/high) to reduce sensitivity
    to rare extremes. Set low/high to 0/100 to recover pure min/max behavior.
    """
    if x_train_full.ndim != 3 or x_train_full.shape[-1] < 4:
        raise ValueError(
            "x_train_full must have shape [S,N,F>=4] so u channel can be read, got "
            f"{tuple(x_train_full.shape)}."
        )
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}.")

    n_pairs = int(x_train_full.shape[0])
    n_nodes = int(x_train_full.shape[1])
    all_vals: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, n_pairs, batch_size):
            end = min(start + batch_size, n_pairs)
            x_state_b = x_train_full[start:end, :, 3:4]
            bsz = int(x_state_b.shape[0])

            x_state_flat = x_state_b.reshape(bsz * n_nodes, 1)
            ei_batched = build_batched_edge_index(
                edge_index=edge_index,
                num_nodes=n_nodes,
                batch_size=bsz,
                device=x_state_flat.device,
            )
            ea_batched = edge_attr.to(device=x_state_flat.device, dtype=x_state_flat.dtype).repeat(
                bsz, 1
            )
            adv_u = compute_adv_input_channel_1d(
                x_state=x_state_flat,
                edge_index=ei_batched,
                edge_attr=ea_batched,
                dt=float(dt),
                dx=float(dx),
                physics_cfg=physics_cfg,
                step_k=0,
            )
            if not torch.isfinite(adv_u).all():
                raise ValueError(
                    "Encountered non-finite advection values while fitting normalization stats."
                )
            all_vals.append(
                adv_u.detach().cpu().numpy().astype(np.float64, copy=False).reshape(-1)
            )

    if len(all_vals) == 0:
        raise ValueError(
            "Failed to fit advection normalization stats: no values were collected."
        )
    vals = np.concatenate(all_vals, axis=0)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise ValueError(
            "Failed to fit advection normalization stats: all collected values were non-finite."
        )

    p_low = float(physics_cfg.normalize_adv_percentile_low)
    p_high = float(physics_cfg.normalize_adv_percentile_high)
    adv_min = float(np.percentile(vals, p_low))
    adv_max = float(np.percentile(vals, p_high))
    if not np.isfinite(adv_min) or not np.isfinite(adv_max):
        raise ValueError(
            "Failed to fit finite advection percentile range, got "
            f"adv_min={adv_min}, adv_max={adv_max}."
        )
    if not (adv_max > adv_min):
        raise ValueError(
            "Advection percentile range is degenerate: "
            f"adv_min={adv_min}, adv_max={adv_max}, "
            f"percentiles=({p_low}, {p_high})."
        )
    return adv_min, adv_max


def _materialize_physics_cfg(
    *,
    cfg: Dict[str, Any],
    x_train_full: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    dt: float,
    dx: float,
) -> PhysicsInputConfig:
    physics_cfg = resolve_physics_input_cfg(cfg)
    raw = cfg.get("physics_inputs", {})
    cfg_physics = dict(raw) if isinstance(raw, dict) else {}
    cfg_physics["enabled"] = bool(physics_cfg.enabled)
    cfg_physics["include_adv"] = bool(physics_cfg.include_adv)
    cfg_physics["input_form"] = str(physics_cfg.input_form)
    cfg_physics["advection_scheme"] = str(physics_cfg.advection_scheme)
    cfg_physics["adv_all_steps"] = bool(physics_cfg.adv_all_steps)
    cfg_physics["input_weighted"] = bool(physics_cfg.input_weighted)
    cfg_physics["adv_weight"] = float(physics_cfg.adv_weight)
    cfg_physics["normalize_adv_to_minus1_1"] = bool(physics_cfg.normalize_adv_to_minus1_1)
    cfg_physics["normalize_adv_percentile_low"] = float(
        physics_cfg.normalize_adv_percentile_low
    )
    cfg_physics["normalize_adv_percentile_high"] = float(
        physics_cfg.normalize_adv_percentile_high
    )
    cfg_physics["normalize_adv_clip"] = bool(physics_cfg.normalize_adv_clip)
    if physics_cfg.normalize_adv_to_minus1_1 and physics_cfg.enabled and physics_cfg.include_adv:
        adv_min = physics_cfg.adv_min
        adv_max = physics_cfg.adv_max
        if adv_min is None or adv_max is None:
            adv_min, adv_max = _fit_adv_minmax_from_train_pairs(
                x_train_full=x_train_full,
                edge_index=edge_index,
                edge_attr=edge_attr,
                dt=float(dt),
                dx=float(dx),
                physics_cfg=physics_cfg,
            )
        cfg_physics["adv_min"] = float(adv_min)
        cfg_physics["adv_max"] = float(adv_max)
        print(
            "[INFO] advection normalization enabled: "
            f"range=[-1,1], fit_percentiles=({float(physics_cfg.normalize_adv_percentile_low):.3f}, "
            f"{float(physics_cfg.normalize_adv_percentile_high):.3f}), "
            f"adv_min={float(adv_min):.6e}, adv_max={float(adv_max):.6e}, "
            f"clip={bool(physics_cfg.normalize_adv_clip)}",
            flush=True,
        )
    else:
        cfg_physics["adv_min"] = (
            None if physics_cfg.adv_min is None else float(physics_cfg.adv_min)
        )
        cfg_physics["adv_max"] = (
            None if physics_cfg.adv_max is None else float(physics_cfg.adv_max)
        )
    cfg_physics["detach_inputs"] = bool(physics_cfg.detach_inputs)
    cfg["physics_inputs"] = cfg_physics
    return resolve_physics_input_cfg(cfg)


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        import json

        cfg: Dict[str, Any] = json.load(f)

    seed = int(cfg.get("seed", 1234))
    set_seed(seed)

    device = resolve_device(str(cfg.get("device", "auto")))

    data_cfg = dict(cfg.get("data", {}))
    data_dir = Path(data_cfg.get("data_dir", "data/burgers_1d_dataset_v1/burgers_data"))
    source_format = str(data_cfg.get("source_format", "auto")).strip().lower()
    if "wenoa_pt_path" in data_cfg:
        data_dir = Path(data_cfg["wenoa_pt_path"])

    source_options: Dict[str, Any] = {}
    if "wenoa_split_fractions" in data_cfg:
        source_options["split_fractions"] = data_cfg["wenoa_split_fractions"]
    if "wenoa_split_seed" in data_cfg:
        source_options["split_seed"] = int(data_cfg["wenoa_split_seed"])
    if "wenoa_nu" in data_cfg:
        source_options["nu_value"] = float(data_cfg["wenoa_nu"])
    if "wenoa_split_paths" in data_cfg:
        source_options["split_paths"] = data_cfg["wenoa_split_paths"]

    data_cfg["data_dir"] = str(data_dir)
    data_cfg["source_format"] = source_format
    cfg["data"] = data_cfg

    print(
        "[INFO] Dataset source: "
        f"source_format={source_format} | path={data_dir}",
        flush=True,
    )
    regime_filter = str(data_cfg.get("regime_filter", "all")).strip().lower()

    target_mode = str(data_cfg.get("target_mode", TARGET_MODE_DELTA)).strip().lower()
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
    cfg["model"] = cfg_model

    batch_size = int(data_cfg.get("batch_size", 32))
    num_workers = int(data_cfg.get("num_workers", 0))

    train_cfg = dict(cfg.get("train", {}))
    grad_loss_cfg = _resolve_gradient_weighted_loss_cfg(cfg)
    train_cfg["gradient_weighted_loss"] = {
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
    }
    cfg["train"] = train_cfg
    rollout_steps = int(train_cfg.get("rollout_steps", 1))
    if rollout_steps < 1:
        raise ValueError(f"train.rollout_steps must be >= 1, got {rollout_steps}.")
    rollout_mode = rollout_steps > 1

    dset_train = load_burgers_dataset(
        data_dir,
        split="train",
        include_rollout_states=rollout_mode,
        dtype=torch.float32,
        source_format=source_format,
        source_options=source_options,
    )
    dset_val = load_burgers_dataset(
        data_dir,
        split="val",
        include_rollout_states=rollout_mode,
        dtype=torch.float32,
        source_format=source_format,
        source_options=source_options,
    )
    dset_test = load_burgers_dataset(
        data_dir,
        split="test",
        include_rollout_states=rollout_mode,
        dtype=torch.float32,
        source_format=source_format,
        source_options=source_options,
    )
    dset_train = filter_burgers_dataset_by_regime(dset_train, regime_filter)
    dset_val = filter_burgers_dataset_by_regime(dset_val, regime_filter)
    dset_test = filter_burgers_dataset_by_regime(dset_test, regime_filter)

    print(
        "[INFO] Regime filter: "
        f"{regime_filter} | "
        f"pairs(train/val/test)={len(dset_train)}/{len(dset_val)}/{len(dset_test)} | "
        f"trajectories(train/val/test)="
        f"{int(dset_train.trajectory_ids.shape[0])}/"
        f"{int(dset_val.trajectory_ids.shape[0])}/"
        f"{int(dset_test.trajectory_ids.shape[0])}",
        flush=True,
    )

    # Shared grid graph for all splits.
    edge_index = dset_train.edge_index.detach().cpu()
    edge_attr = dset_train.edge_attr.detach().cpu()

    x_train_full = dset_train.x_full.detach().cpu()
    x_val_full = dset_val.x_full.detach().cpu()
    x_test_full = dset_test.x_full.detach().cpu()

    u_next_train = dset_train.u_next.detach().cpu()
    u_next_val = dset_val.u_next.detach().cpu()
    u_next_test = dset_test.u_next.detach().cpu()

    dt = float(dset_train.metadata["dt"])
    dx = float(dset_train.metadata["dx"])
    n_nodes = int(dset_train.x_full.shape[1])
    source_format_resolved = str(dset_train.metadata.get("source_format", source_format)).strip().lower()

    input_cfg = _materialize_input_cfg(
        cfg=cfg,
        x_train_full=x_train_full,
        source_format=source_format_resolved,
    )
    physics_cfg = _materialize_physics_cfg(
        cfg=cfg,
        x_train_full=x_train_full,
        edge_index=edge_index,
        edge_attr=edge_attr,
        dt=dt,
        dx=dx,
    )
    state_col = u_column_index(input_cfg)
    if state_col is None:
        raise ValueError(
            "input_features.feature_columns must include 'u' so model can predict "
            "state evolution."
        )

    cfg_model["state_channel"] = int(state_col)
    cfg["model"] = cfg_model

    train_idx = np.arange(len(dset_train), dtype=np.int64)
    val_idx = np.arange(len(dset_val), dtype=np.int64)
    test_idx = np.arange(len(dset_test), dtype=np.int64)

    train_effective_idx = train_idx
    val_effective_idx = val_idx
    test_effective_idx = test_idx

    if rollout_mode:
        for name, dset in (("train", dset_train), ("val", dset_val), ("test", dset_test)):
            if dset.rollout_states is None:
                raise ValueError(
                    f"rollout training requested but rollout_states missing for split '{name}'."
                )

        window_train = int(dset_train.rollout_states.shape[1] - 1)
        window_val = int(dset_val.rollout_states.shape[1] - 1)
        window_test = int(dset_test.rollout_states.shape[1] - 1)

        if rollout_steps > window_train or rollout_steps > window_val or rollout_steps > window_test:
            raise ValueError(
                f"train.rollout_steps={rollout_steps} exceeds available window in at least "
                f"one split (train/val/test windows={window_train}/{window_val}/{window_test})."
            )

        train_effective_idx = _select_rollout_start_indices(
            pair_time_index_all=dset_train.pair_time_index,
            window=window_train,
            rollout_steps=rollout_steps,
        )
        val_effective_idx = _select_rollout_start_indices(
            pair_time_index_all=dset_val.pair_time_index,
            window=window_val,
            rollout_steps=rollout_steps,
        )
        test_effective_idx = _select_rollout_start_indices(
            pair_time_index_all=dset_test.pair_time_index,
            window=window_test,
            rollout_steps=rollout_steps,
        )

        if (
            len(train_effective_idx) == 0
            or len(val_effective_idx) == 0
            or len(test_effective_idx) == 0
        ):
            raise ValueError(
                "No valid rollout start samples after filtering by rollout_steps. "
                "Try a smaller train.rollout_steps."
            )

        print(
            "[INFO] Rollout training enabled: "
            f"rollout_steps={rollout_steps}, "
            f"effective starts (train/val/test)="
            f"{len(train_effective_idx)}/{len(val_effective_idx)}/{len(test_effective_idx)}.",
            flush=True,
        )

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
        count_input_feature_channels(input_cfg)
        + count_physics_input_channels(physics_cfg)
    )
    print(
        "[INFO] Input channels: "
        f"feature_columns={list(input_cfg.feature_columns)}, "
        f"use_pos={input_cfg.use_pos}, "
        f"normalize_u_to_minus1_1={input_cfg.normalize_u_to_minus1_1}, "
        f"state_channel={state_col}, "
        f"total_model_in_channels={in_channels}",
        flush=True,
    )
    print(
        "[INFO] Physics-input channels: "
        f"enabled={physics_cfg.enabled}, "
        f"include_adv={physics_cfg.include_adv}, "
        f"input_form={physics_cfg.input_form}, "
        f"advection_scheme={physics_cfg.advection_scheme}, "
        f"normalize_adv_to_minus1_1={physics_cfg.normalize_adv_to_minus1_1}, "
        f"normalize_adv_percentile_low={physics_cfg.normalize_adv_percentile_low}, "
        f"normalize_adv_percentile_high={physics_cfg.normalize_adv_percentile_high}, "
        f"total_model_in_channels={in_channels}",
        flush=True,
    )
    print(
        "[INFO] Gradient-weighted loss: "
        f"enabled={grad_loss_cfg.enabled}, "
        f"strength={grad_loss_cfg.strength:.3e}, "
        f"power={grad_loss_cfg.power:.3e}, "
        f"smooth_radius={grad_loss_cfg.smooth_radius}, "
        f"peak_spread_sigma={grad_loss_cfg.peak_spread_sigma:.3e}, "
        f"clip_max={grad_loss_cfg.clip_max}",
        flush=True,
    )

    model = build_model_from_config(
        cfg,
        in_channels=in_channels,
        edge_attr_channels=int(edge_attr.shape[-1]),
    ).to(device)

    epochs = int(train_cfg.get("epochs", 200))
    val_every = int(train_cfg.get("val_every", 1))
    if val_every < 1:
        raise ValueError(f"train.val_every must be >= 1, got {val_every}.")
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
    save_dir = Path(train_cfg.get("save_dir", "runs/burgers_sage"))
    save_dir.mkdir(parents=True, exist_ok=True)

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
        "[INFO] Validation cadence: "
        f"every {val_every} epoch(s), with final-epoch validation enabled.",
        flush=True,
    )

    history: List[Dict[str, float]] = []
    best_val = float("inf")
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
        should_validate = (epoch % val_every == 0) or (epoch == epochs)

        if rollout_mode:
            assert dset_train.rollout_states is not None
            if should_validate:
                assert dset_val.rollout_states is not None

            train_m = _run_split_rollout(
                model=model,
                loader=train_loader,
                x_full_all=x_train_full,
                rollout_states=dset_train.rollout_states,
                pair_trajectory_id_all=dset_train.pair_trajectory_id,
                pair_time_index_all=dset_train.pair_time_index,
                edge_index=edge_index,
                edge_attr=edge_attr,
                dt=dt,
                dx=dx,
                input_cfg=input_cfg,
                physics_cfg=physics_cfg,
                grad_loss_cfg=grad_loss_cfg,
                target_mode=target_mode,
                rollout_steps=rollout_steps,
                optimizer=optimizer,
                device=device,
                grad_clip=grad_clip,
            )
            if should_validate:
                val_m = _run_split_rollout(
                    model=model,
                    loader=val_loader,
                    x_full_all=x_val_full,
                    rollout_states=dset_val.rollout_states,
                    pair_trajectory_id_all=dset_val.pair_trajectory_id,
                    pair_time_index_all=dset_val.pair_time_index,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    dt=dt,
                    dx=dx,
                    input_cfg=input_cfg,
                    physics_cfg=physics_cfg,
                    grad_loss_cfg=grad_loss_cfg,
                    target_mode=target_mode,
                    rollout_steps=rollout_steps,
                    optimizer=None,
                    device=device,
                    grad_clip=grad_clip,
                )
            else:
                val_m = None
        else:
            train_m = _run_split(
                model=model,
                loader=train_loader,
                x_full_all=x_train_full,
                u_next_all=u_next_train,
                edge_index=edge_index,
                edge_attr=edge_attr,
                dt=dt,
                dx=dx,
                input_cfg=input_cfg,
                physics_cfg=physics_cfg,
                grad_loss_cfg=grad_loss_cfg,
                target_mode=target_mode,
                optimizer=optimizer,
                device=device,
                grad_clip=grad_clip,
            )
            if should_validate:
                val_m = _run_split(
                    model=model,
                    loader=val_loader,
                    x_full_all=x_val_full,
                    u_next_all=u_next_val,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    dt=dt,
                    dx=dx,
                    input_cfg=input_cfg,
                    physics_cfg=physics_cfg,
                    grad_loss_cfg=grad_loss_cfg,
                    target_mode=target_mode,
                    optimizer=None,
                    device=device,
                    grad_clip=grad_clip,
                )
            else:
                val_m = None

        epoch_time_s = time.time() - epoch_t0

        row = {
            "epoch": float(epoch),
            "lr": current_lr,
            "epoch_time_s": epoch_time_s,
            "train_loss": train_m.loss,
            "train_rmse": train_m.rmse,
            "train_rel_l2": train_m.rel_l2,
            "val_loss": (float("nan") if val_m is None else val_m.loss),
            "val_rmse": (float("nan") if val_m is None else val_m.rmse),
            "val_rel_l2": (float("nan") if val_m is None else val_m.rel_l2),
        }
        history.append(row)

        improved = False
        if val_m is not None:
            improved = val_m.loss < best_val
            if improved:
                best_val = val_m.loss
                best_epoch = epoch
                stale = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": cfg,
                        "metadata": {
                            "train": dict(dset_train.metadata),
                            "val": dict(dset_val.metadata),
                            "test": dict(dset_test.metadata),
                        },
                        "split_indices": {
                            "train": train_idx.tolist(),
                            "val": val_idx.tolist(),
                            "test": test_idx.tolist(),
                            "train_effective": train_effective_idx.tolist(),
                            "val_effective": val_effective_idx.tolist(),
                            "test_effective": test_effective_idx.tolist(),
                        },
                        "epoch": epoch,
                        "best_val_loss": best_val,
                    },
                    save_dir / "checkpoint_best.pt",
                )
            else:
                stale += 1

        if (epoch == 1) or (epoch % log_every == 0) or improved:
            val_loss_str = "skipped" if val_m is None else f"{val_m.loss:.6e}"
            val_rmse_str = "skipped" if val_m is None else f"{val_m.rmse:.6e}"
            print(
                f"[epoch {epoch:04d}] "
                f"lr={current_lr:.6e} "
                f"train_loss={train_m.loss:.6e} val_loss={val_loss_str} "
                f"train_rmse={train_m.rmse:.6e} val_rmse={val_rmse_str} "
                f"epoch_time={epoch_time_s:.2f}s",
                flush=True,
            )

        if val_m is not None and patience is not None and stale >= patience:
            print(
                f"Early stopping at epoch {epoch} "
                f"(no val improvement for {patience} epochs).",
                flush=True,
            )
            break

    if best_epoch < 0:
        raise RuntimeError(
            "No validation epoch was run, so checkpoint_best.pt was never created. "
            "Ensure epochs >= 1 and train.val_every is valid."
        )

    best_ckpt = torch.load(save_dir / "checkpoint_best.pt", map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])

    if rollout_mode:
        assert dset_test.rollout_states is not None
        test_m = _run_split_rollout(
            model=model,
            loader=test_loader,
            x_full_all=x_test_full,
            rollout_states=dset_test.rollout_states,
            pair_trajectory_id_all=dset_test.pair_trajectory_id,
            pair_time_index_all=dset_test.pair_time_index,
            edge_index=edge_index,
            edge_attr=edge_attr,
            dt=dt,
            dx=dx,
            input_cfg=input_cfg,
            physics_cfg=physics_cfg,
            grad_loss_cfg=grad_loss_cfg,
            target_mode=target_mode,
            rollout_steps=rollout_steps,
            optimizer=None,
            device=device,
            grad_clip=grad_clip,
        )
    else:
        test_m = _run_split(
            model=model,
            loader=test_loader,
            x_full_all=x_test_full,
            u_next_all=u_next_test,
            edge_index=edge_index,
            edge_attr=edge_attr,
            dt=dt,
            dx=dx,
            input_cfg=input_cfg,
            physics_cfg=physics_cfg,
            grad_loss_cfg=grad_loss_cfg,
            target_mode=target_mode,
            optimizer=None,
            device=device,
            grad_clip=grad_clip,
        )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": cfg,
            "metadata": {
                "train": dict(dset_train.metadata),
                "val": dict(dset_val.metadata),
                "test": dict(dset_test.metadata),
            },
            "split_indices": {
                "train": train_idx.tolist(),
                "val": val_idx.tolist(),
                "test": test_idx.tolist(),
                "train_effective": train_effective_idx.tolist(),
                "val_effective": val_effective_idx.tolist(),
                "test_effective": test_effective_idx.tolist(),
            },
            "epoch": len(history),
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
            "test_metrics": test_m.to_dict(),
        },
        save_dir / "checkpoint_last.pt",
    )

    summary: Dict[str, Any] = {
        "run_seconds": float(time.time() - t0),
        "device": str(device),
        "data_dir": str(data_dir),
        "regime_filter": regime_filter,
        "target_mode": target_mode,
        "val_every": int(val_every),
        "lr_initial": float(lr_initial),
        "lr_final": float(optimizer.param_groups[0]["lr"]),
        "lr_schedule_enabled": bool(lr_schedule_enabled),
        "lr_schedule": [
            {"epoch": int(ep), "lr": float(val)}
            for ep, val in sorted(lr_schedule.items())
        ],
        "training_mode": "rollout" if rollout_mode else "one_step",
        "rollout_steps": int(rollout_steps),
        "num_nodes": n_nodes,
        "model_in_channels": int(in_channels),
        "state_channel": int(state_col),
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "test_size": int(len(test_idx)),
        "train_effective_size": int(len(train_effective_idx)),
        "val_effective_size": int(len(val_effective_idx)),
        "test_effective_size": int(len(test_effective_idx)),
        "input_features": {
            "feature_columns": list(input_cfg.feature_columns),
            "use_pos": bool(input_cfg.use_pos),
            "normalize_u_to_minus1_1": bool(input_cfg.normalize_u_to_minus1_1),
            "normalize_clip": bool(input_cfg.normalize_clip),
            "u_min": None if input_cfg.u_min is None else float(input_cfg.u_min),
            "u_max": None if input_cfg.u_max is None else float(input_cfg.u_max),
        },
        "physics_inputs": {
            "enabled": bool(physics_cfg.enabled),
            "include_adv": bool(physics_cfg.include_adv),
            "input_form": str(physics_cfg.input_form),
            "advection_scheme": str(physics_cfg.advection_scheme),
            "adv_all_steps": bool(physics_cfg.adv_all_steps),
            "input_weighted": bool(physics_cfg.input_weighted),
            "adv_weight": float(physics_cfg.adv_weight),
            "normalize_adv_to_minus1_1": bool(physics_cfg.normalize_adv_to_minus1_1),
            "normalize_adv_percentile_low": float(physics_cfg.normalize_adv_percentile_low),
            "normalize_adv_percentile_high": float(physics_cfg.normalize_adv_percentile_high),
            "normalize_adv_clip": bool(physics_cfg.normalize_adv_clip),
            "adv_min": None if physics_cfg.adv_min is None else float(physics_cfg.adv_min),
            "adv_max": None if physics_cfg.adv_max is None else float(physics_cfg.adv_max),
            "detach_inputs": bool(physics_cfg.detach_inputs),
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
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "test_metrics": test_m.to_dict(),
    }

    save_json(summary, save_dir / "summary.json")
    save_json({"history": history}, save_dir / "history.json")
    save_json(
        {
            "train": train_idx.tolist(),
            "val": val_idx.tolist(),
            "test": test_idx.tolist(),
            "train_effective": train_effective_idx.tolist(),
            "val_effective": val_effective_idx.tolist(),
            "test_effective": test_effective_idx.tolist(),
        },
        save_dir / "split_indices.json",
    )

    print("Training complete.")
    print(f"  save_dir: {save_dir}")
    print(
        f"  training_mode: {'rollout' if rollout_mode else 'one_step'} "
        f"(rollout_steps={rollout_steps})"
    )
    print(f"  best_epoch: {best_epoch}")
    print(f"  best_val_loss: {best_val:.6e}")
    print(f"  test_loss: {test_m.loss:.6e}")
    print(f"  test_rmse: {test_m.rmse:.6e}")
    print(f"  test_rel_l2: {test_m.rel_l2:.6e}")


if __name__ == "__main__":
    main()
