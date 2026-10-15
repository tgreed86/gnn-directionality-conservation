"""Input-feature helpers for 1D Burgers models.

The raw dataset stores five node features per sample:
  [x, sin(2*pi*x), cos(2*pi*x), u, nu]

This module keeps feature-column selection and optional u-normalization in one
place so training and rollout use identical construction.

For `source_format="wenoa_pt"`, feature selection is simplified to physical
state `u` plus optional position `x` controlled by `input_features.use_pos`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch


FEATURE_COLUMNS = ("x", "sin", "cos", "u", "nu")
FEATURE_TO_INDEX = {name: i for i, name in enumerate(FEATURE_COLUMNS)}


@dataclass(frozen=True)
class InputFeatureConfig:
    """Runtime controls for node input features."""

    feature_columns: tuple[str, ...] = FEATURE_COLUMNS
    use_pos: bool = True
    normalize_u_to_minus1_1: bool = False
    u_min: float | None = None
    u_max: float | None = None
    normalize_clip: bool = False


def _normalize_feature_name(name: str) -> str:
    key = str(name).strip().lower()
    aliases = {
        "sin(2*pi*x)": "sin",
        "sin(2pix)": "sin",
        "cos(2*pi*x)": "cos",
        "cos(2pix)": "cos",
    }
    return aliases.get(key, key)


def _resolve_feature_columns(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return FEATURE_COLUMNS
    if isinstance(raw, str):
        items = [raw]
    elif isinstance(raw, Sequence):
        items = [str(v) for v in raw]
    else:
        raise ValueError(
            "input_features.feature_columns must be a string or list of strings."
        )

    cols = tuple(_normalize_feature_name(v) for v in items)
    if len(cols) == 0:
        raise ValueError("input_features.feature_columns cannot be empty.")

    bad = [c for c in cols if c not in FEATURE_TO_INDEX]
    if bad:
        raise ValueError(
            "Unknown feature column(s): "
            f"{bad}. Valid columns are {list(FEATURE_COLUMNS)}."
        )

    if len(set(cols)) != len(cols):
        raise ValueError(
            "input_features.feature_columns contains duplicates. "
            f"Got: {list(cols)}"
        )
    return cols


def resolve_input_feature_cfg(
    cfg: Mapping[str, Any],
    *,
    source_format: str | None = None,
) -> InputFeatureConfig:
    """Resolve input-feature config from a run config dict."""
    input_cfg = cfg.get("input_features", {})
    if not isinstance(input_cfg, Mapping):
        input_cfg = {}

    raw_use_pos = input_cfg.get("use_pos", None)
    use_pos = True if raw_use_pos is None else bool(raw_use_pos)

    source_key = None if source_format is None else str(source_format).strip().lower()
    if source_key == "wenoa_pt":
        # For WENO-A trajectories, always use the physical field u and optionally
        # prepend position x as an additional context feature.
        cols = ("x", "u") if use_pos else ("u",)
    else:
        cols = _resolve_feature_columns(input_cfg.get("feature_columns", None))
        # Optional toggle for non-WENO-A datasets. If unspecified, preserve current
        # behavior driven solely by feature_columns.
        if raw_use_pos is not None:
            if use_pos:
                if "x" not in cols:
                    cols = ("x",) + tuple(c for c in cols if c != "x")
            else:
                cols = tuple(c for c in cols if c != "x")
                if len(cols) == 0:
                    raise ValueError(
                        "input_features.use_pos=false removed all selected features. "
                        "Choose at least one non-position feature."
                    )
    normalize_u = bool(
        input_cfg.get(
            "normalize_u_to_minus1_1",
            input_cfg.get("normalize_state_to_minus1_1", False),
        )
    )
    raw_u_min = input_cfg.get("u_min", input_cfg.get("state_min", None))
    raw_u_max = input_cfg.get("u_max", input_cfg.get("state_max", None))
    normalize_clip = bool(input_cfg.get("normalize_clip", False))

    return InputFeatureConfig(
        feature_columns=cols,
        use_pos=use_pos,
        normalize_u_to_minus1_1=normalize_u,
        u_min=None if raw_u_min is None else float(raw_u_min),
        u_max=None if raw_u_max is None else float(raw_u_max),
        normalize_clip=normalize_clip,
    )


def count_input_feature_channels(input_cfg: InputFeatureConfig) -> int:
    """Return model input-channel count after feature selection."""
    return int(len(input_cfg.feature_columns))


def resolve_feature_indices(input_cfg: InputFeatureConfig) -> list[int]:
    """Return selected raw-dataset feature indices."""
    return [FEATURE_TO_INDEX[c] for c in input_cfg.feature_columns]


def u_column_index(input_cfg: InputFeatureConfig) -> int | None:
    """Return index of `u` within selected model channels, if present."""
    for i, c in enumerate(input_cfg.feature_columns):
        if c == "u":
            return i
    return None


def select_feature_channels(
    *,
    x_full: torch.Tensor,
    input_cfg: InputFeatureConfig,
) -> torch.Tensor:
    """Select configured feature columns from the last dimension.

    Supports shapes [..., 5], returning [..., F_selected].
    """
    if x_full.shape[-1] != len(FEATURE_COLUMNS):
        raise ValueError(
            f"Expected x_full last dimension {len(FEATURE_COLUMNS)}, got {x_full.shape[-1]}."
        )
    idx = torch.as_tensor(resolve_feature_indices(input_cfg), device=x_full.device)
    return x_full.index_select(dim=-1, index=idx)


def fit_u_minmax(
    *,
    x_full: torch.Tensor,
    sample_indices: torch.Tensor | None = None,
) -> tuple[float, float]:
    """Compute scalar min/max for the raw `u` channel from dataset tensors.

    Args:
        x_full: Tensor shape [S, N, 5].
        sample_indices: Optional subset over first axis.
    """
    if x_full.ndim != 3 or x_full.shape[-1] != len(FEATURE_COLUMNS):
        raise ValueError(
            "x_full must have shape [S,N,5], got "
            f"{tuple(x_full.shape)}."
        )

    u_idx = FEATURE_TO_INDEX["u"]
    if sample_indices is None:
        x_sel = x_full[..., u_idx]
    else:
        idx = sample_indices.to(device=x_full.device, dtype=torch.long)
        x_sel = x_full[idx, :, u_idx]

    u_min = float(torch.min(x_sel).item())
    u_max = float(torch.max(x_sel).item())
    return u_min, u_max


def apply_u_minmax_minus1_1(
    *,
    x_model: torch.Tensor,
    input_cfg: InputFeatureConfig,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Apply fixed min-max scaling to selected `u` channel, if enabled."""
    if x_model.ndim != 2:
        raise ValueError(f"x_model must have shape [N,F], got {tuple(x_model.shape)}.")
    if not input_cfg.normalize_u_to_minus1_1:
        return x_model
    if input_cfg.u_min is None or input_cfg.u_max is None:
        raise ValueError(
            "u normalization is enabled but u_min/u_max are missing."
        )

    u_col = u_column_index(input_cfg)
    if u_col is None:
        raise ValueError(
            "u normalization is enabled but selected feature_columns do not include 'u'."
        )

    mn = float(input_cfg.u_min)
    mx = float(input_cfg.u_max)
    denom = mx - mn

    x_out = x_model.clone()
    u = x_out[:, u_col : u_col + 1]
    if abs(denom) <= float(eps):
        u_scaled = torch.zeros_like(u)
    else:
        u_scaled = (2.0 * (u - mn) / denom) - 1.0

    if input_cfg.normalize_clip:
        u_scaled = torch.clamp(u_scaled, min=-1.0, max=1.0)

    x_out[:, u_col : u_col + 1] = u_scaled
    return x_out
