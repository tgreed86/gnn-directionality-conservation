"""Input feature assembly helpers for 1D advection models.

This module keeps non-physics feature options (for example node position
channels) in one place so training and rollout use identical input construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch


@dataclass(frozen=True)
class InputFeatureConfig:
    """Runtime controls for optional non-physics input channels."""

    include_pos: bool = False
    normalize_state_to_minus1_1: bool = False
    state_min: float | None = None
    state_max: float | None = None
    normalize_clip: bool = False


def resolve_input_feature_cfg(cfg: Mapping[str, Any]) -> InputFeatureConfig:
    """Resolve optional input-feature flags from config.

    Preferred key:
      input_features.include_pos

    Backward-compatible fallback:
      data.include_pos
    """
    data_cfg = cfg.get("data", {})
    include_pos_default = (
        bool(data_cfg.get("include_pos", False))
        if isinstance(data_cfg, Mapping)
        else False
    )

    input_cfg = cfg.get("input_features", {})
    if isinstance(input_cfg, Mapping):
        include_pos = bool(input_cfg.get("include_pos", include_pos_default))
        normalize_state = bool(
            input_cfg.get(
                "normalize_state_to_minus1_1",
                input_cfg.get("normalize_to_minus1_1", False),
            )
        )
        raw_state_min = input_cfg.get("state_min", None)
        raw_state_max = input_cfg.get("state_max", None)
        normalize_clip = bool(input_cfg.get("normalize_clip", False))
    else:
        include_pos = include_pos_default
        normalize_state = False
        raw_state_min = None
        raw_state_max = None
        normalize_clip = False

    state_min = None if raw_state_min is None else float(raw_state_min)
    state_max = None if raw_state_max is None else float(raw_state_max)

    return InputFeatureConfig(
        include_pos=include_pos,
        normalize_state_to_minus1_1=normalize_state,
        state_min=state_min,
        state_max=state_max,
        normalize_clip=normalize_clip,
    )


def count_input_feature_channels(input_cfg: InputFeatureConfig) -> int:
    """Return number of extra per-node channels from non-physics features."""
    return int(bool(input_cfg.include_pos))


def append_position_channel(
    *,
    x_state: torch.Tensor,
    pos_nodes: torch.Tensor,
    input_cfg: InputFeatureConfig,
    batch_size: int | None = None,
) -> torch.Tensor:
    """Append node-position channel to flattened node features when enabled.

    Args:
        x_state: Flattened node feature tensor [N_total, F].
        pos_nodes: Per-node positions [N, 1].
        input_cfg: Input feature configuration.
        batch_size: Number of independent graphs in `x_state` when batched.
            If omitted, `x_state` is assumed to contain exactly one graph.

    Returns:
        Tensor with optional position channel appended on feature dimension.
    """
    if x_state.ndim != 2:
        raise ValueError(f"x_state must have shape [N_total, F], got {tuple(x_state.shape)}.")
    if not input_cfg.include_pos:
        return x_state
    if pos_nodes.ndim != 2 or pos_nodes.shape[1] != 1:
        raise ValueError(
            f"pos_nodes must have shape [N,1], got {tuple(pos_nodes.shape)}."
        )

    pos_nodes_local = pos_nodes.to(device=x_state.device, dtype=x_state.dtype)
    n_nodes = int(pos_nodes_local.shape[0])

    if batch_size is None:
        if int(x_state.shape[0]) != n_nodes:
            raise ValueError(
                "x_state and pos_nodes size mismatch for single-graph inputs: "
                f"x_state has {x_state.shape[0]} rows but pos_nodes has {n_nodes}."
            )
        pos_flat = pos_nodes_local
    else:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")
        expected = int(batch_size) * n_nodes
        if int(x_state.shape[0]) != expected:
            raise ValueError(
                "x_state and pos_nodes size mismatch for batched inputs: "
                f"x_state has {x_state.shape[0]} rows, expected {expected} "
                f"from batch_size={batch_size} and n_nodes={n_nodes}."
            )
        pos_flat = (
            pos_nodes_local.unsqueeze(0)
            .expand(int(batch_size), n_nodes, 1)
            .reshape(expected, 1)
        )

    return torch.cat([x_state, pos_flat], dim=1)


def fit_state_minmax(
    *,
    x_state: torch.Tensor,
    sample_indices: torch.Tensor | None = None,
) -> tuple[float, float]:
    """Compute scalar min/max for state channel from dataset tensors.

    Args:
        x_state: Tensor with shape [S, N, 1] or [S, N, F].
        sample_indices: Optional sample-index subset over first axis.
    """
    if x_state.ndim != 3:
        raise ValueError(f"x_state must have shape [S,N,F], got {tuple(x_state.shape)}.")
    if x_state.shape[-1] < 1:
        raise ValueError("x_state must have at least one feature channel.")

    if sample_indices is None:
        x_sel = x_state[..., 0]
    else:
        idx = sample_indices.to(device=x_state.device, dtype=torch.long)
        x_sel = x_state[idx, :, 0]

    x_min = float(torch.min(x_sel).item())
    x_max = float(torch.max(x_sel).item())
    return x_min, x_max


def apply_state_minmax_minus1_1(
    *,
    x_model: torch.Tensor,
    input_cfg: InputFeatureConfig,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Apply fixed min-max scaling to state channel: x -> 2*(x-min)/(max-min)-1."""
    if x_model.ndim != 2:
        raise ValueError(f"x_model must have shape [N,F], got {tuple(x_model.shape)}.")
    if not input_cfg.normalize_state_to_minus1_1:
        return x_model
    if input_cfg.state_min is None or input_cfg.state_max is None:
        raise ValueError(
            "State normalization is enabled but state_min/state_max are missing."
        )

    mn = float(input_cfg.state_min)
    mx = float(input_cfg.state_max)
    denom = mx - mn
    x0 = x_model[:, :1]
    if abs(denom) <= float(eps):
        x0_scaled = torch.zeros_like(x0)
    else:
        x0_scaled = (2.0 * (x0 - mn) / denom) - 1.0

    if input_cfg.normalize_clip:
        x0_scaled = torch.clamp(x0_scaled, min=-1.0, max=1.0)

    if x_model.shape[1] == 1:
        return x0_scaled
    return torch.cat([x0_scaled, x_model[:, 1:]], dim=1)
