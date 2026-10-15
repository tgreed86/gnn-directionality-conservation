"""GNN utilities for 1D advection surrogate training.

This module intentionally mirrors key patterns from the larger Physics-Aware
codebase in a simplified form:
- configurable activation helper
- stacked message-passing blocks (GraphSAGE, MeshGraphNet, flux-balance variants)
- optional skip and layer norm
- config-friendly model factory

The model is designed for one-step learning on periodic ring graphs where each
node carries a scalar state value.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import json
import random
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


TARGET_MODE_STATE = "state"
TARGET_MODE_DELTA = "delta"
TARGET_MODE_RATE = "rate"
TARGET_MODES = {TARGET_MODE_STATE, TARGET_MODE_DELTA, TARGET_MODE_RATE}

PREDICT_TYPE_STATE = "state"
PREDICT_TYPE_DELTA = "delta"
PREDICT_TYPE_RATE = "rate"
PREDICT_TYPES = {PREDICT_TYPE_STATE, PREDICT_TYPE_DELTA, PREDICT_TYPE_RATE}

STATE_RESIDUAL_MODE_RAW = "raw"
STATE_RESIDUAL_MODE_MODEL_INPUT = "model_input"
STATE_RESIDUAL_MODES = {STATE_RESIDUAL_MODE_RAW, STATE_RESIDUAL_MODE_MODEL_INPUT}
_STATE_RESIDUAL_MODE_ALIASES = {
    STATE_RESIDUAL_MODE_RAW: STATE_RESIDUAL_MODE_RAW,
    "raw_state": STATE_RESIDUAL_MODE_RAW,
    "physical": STATE_RESIDUAL_MODE_RAW,
    "physical_state": STATE_RESIDUAL_MODE_RAW,
    "current": STATE_RESIDUAL_MODE_RAW,
    STATE_RESIDUAL_MODE_MODEL_INPUT: STATE_RESIDUAL_MODE_MODEL_INPUT,
    "model": STATE_RESIDUAL_MODE_MODEL_INPUT,
    "input": STATE_RESIDUAL_MODE_MODEL_INPUT,
    "input_state": STATE_RESIDUAL_MODE_MODEL_INPUT,
    "normalized": STATE_RESIDUAL_MODE_MODEL_INPUT,
    "normalized_state": STATE_RESIDUAL_MODE_MODEL_INPUT,
    "legacy": STATE_RESIDUAL_MODE_MODEL_INPUT,
    "none": STATE_RESIDUAL_MODE_MODEL_INPUT,
}


def predict_type_from_target_mode(target_mode: str) -> str:
    """Resolve model predict_type from training target_mode.

    For this advection setup, mapping is one-to-one:
      state -> state
      delta -> delta
      rate  -> rate
    """
    key = str(target_mode).strip().lower()
    if key not in TARGET_MODES:
        raise ValueError(
            f"target_mode must be one of {sorted(TARGET_MODES)}, got '{target_mode}'."
        )
    return key


def normalize_state_residual_mode(mode: Any) -> str:
    """Normalize the state base used by predict_state reconstruction.

    Modes:
      raw: add predicted delta/rate update to the raw physical state supplied by
        callers.
      model_input: add predicted delta/rate update to x[:, :1], matching the
        legacy behavior before raw-state overrides were threaded through scripts.
    """
    key = str(mode).strip().lower().replace("-", "_")
    if key not in _STATE_RESIDUAL_MODE_ALIASES:
        raise ValueError(
            "state_residual_mode must be one of "
            f"{sorted(STATE_RESIDUAL_MODES)} "
            "(aliases include: legacy, normalized, raw_state), got "
            f"'{mode}'."
        )
    return _STATE_RESIDUAL_MODE_ALIASES[key]


def resolve_state_residual_mode(cfg: Mapping[str, Any] | str | None) -> str:
    """Resolve model.state_residual_mode from a run config.

    Missing config defaults to current behavior, "raw".
    """
    if cfg is None:
        return STATE_RESIDUAL_MODE_RAW
    if isinstance(cfg, str):
        return normalize_state_residual_mode(cfg)
    model_cfg_raw = cfg.get("model", {})
    model_cfg = model_cfg_raw if isinstance(model_cfg_raw, Mapping) else {}
    raw = model_cfg.get(
        "state_residual_mode",
        cfg.get("state_residual_mode", STATE_RESIDUAL_MODE_RAW),
    )
    return normalize_state_residual_mode(raw)


def state_residual_from_mode(
    *,
    x_raw: torch.Tensor,
    state_residual_mode: str,
) -> torch.Tensor | None:
    """Return the predict_state override for a configured residual-base mode."""
    mode = normalize_state_residual_mode(state_residual_mode)
    if mode == STATE_RESIDUAL_MODE_RAW:
        return x_raw
    if mode == STATE_RESIDUAL_MODE_MODEL_INPUT:
        return None
    raise RuntimeError(f"Unexpected state_residual_mode='{state_residual_mode}'.")


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_pref: str) -> torch.device:
    """Resolve device preference string to a torch.device."""
    key = str(device_pref).strip().lower()
    if key == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if key == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested device='cuda' but CUDA is not available.")
        return torch.device("cuda")
    if key == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Requested device='mps' but MPS is not available.")
        return torch.device("mps")
    if key == "cpu":
        return torch.device("cpu")
    return torch.device(device_pref)


def load_json(path: str | Path) -> Dict[str, Any]:
    """Load a JSON file into a dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Mapping[str, Any], path: str | Path) -> None:
    """Save mapping as pretty JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dict(data), f, indent=2, sort_keys=False)


def split_indices(
    num_samples: int,
    *,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create reproducible train/val/test index splits."""
    if num_samples <= 2:
        raise ValueError(f"Need at least 3 samples to split, got {num_samples}.")
    if not (0.0 <= val_fraction < 1.0):
        raise ValueError(f"val_fraction must be in [0,1), got {val_fraction}.")
    if not (0.0 <= test_fraction < 1.0):
        raise ValueError(f"test_fraction must be in [0,1), got {test_fraction}.")
    if (val_fraction + test_fraction) >= 1.0:
        raise ValueError(
            "val_fraction + test_fraction must be < 1.0, got "
            f"{val_fraction + test_fraction}."
        )

    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_samples)

    n_test = int(round(num_samples * test_fraction))
    n_val = int(round(num_samples * val_fraction))
    n_test = min(n_test, num_samples - 2)
    n_val = min(n_val, num_samples - n_test - 1)

    test_idx = np.sort(perm[:n_test])
    val_idx = np.sort(perm[n_test : n_test + n_val])
    train_idx = np.sort(perm[n_test + n_val :])
    if len(train_idx) == 0:
        raise ValueError("Empty training split; adjust val/test fractions.")
    return train_idx, val_idx, test_idx


def build_batched_edge_index(
    edge_index: torch.Tensor,
    *,
    num_nodes: int,
    batch_size: int,
    device: torch.device | None = None,
) -> torch.LongTensor:
    """Repeat one graph edge_index across a batch of independent graphs."""
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must have shape [2,E], got {tuple(edge_index.shape)}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    if num_nodes <= 0:
        raise ValueError(f"num_nodes must be positive, got {num_nodes}.")

    ei = edge_index.to(device=device, dtype=torch.long)
    e = ei.shape[1]
    offsets = (
        torch.arange(batch_size, device=ei.device, dtype=torch.long)
        .repeat_interleave(e)
        * int(num_nodes)
    )
    batched = ei.repeat(1, batch_size)
    batched = batched + offsets.unsqueeze(0)
    return batched


def repeat_edge_attr(
    edge_attr: torch.Tensor | None,
    *,
    batch_size: int,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor | None:
    """Repeat edge attributes across batched independent graphs."""
    if edge_attr is None:
        return None
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    out = edge_attr.to(device=device, dtype=dtype)
    return out.repeat(batch_size, 1)


def make_target(
    x_now: torch.Tensor,
    u_next: torch.Tensor,
    *,
    dt: float,
    target_mode: str,
) -> torch.Tensor:
    """Create supervision tensor for chosen target mode."""
    if target_mode not in TARGET_MODES:
        raise ValueError(
            f"target_mode must be one of {sorted(TARGET_MODES)}, got '{target_mode}'."
        )
    if target_mode == TARGET_MODE_STATE:
        return u_next
    if target_mode == TARGET_MODE_DELTA:
        return u_next - x_now
    return (u_next - x_now) / float(dt)


def _make_activation(
    name: str,
    *,
    negative_slope: float = 0.01,
    elu_alpha: float = 1.0,
) -> nn.Module:
    key = str(name).strip().lower()
    if key in {"relu"}:
        return nn.ReLU()
    if key in {"leaky_relu", "leakyrelu", "lrelu", "leaky"}:
        return nn.LeakyReLU(negative_slope=float(negative_slope))
    if key in {"silu", "swish"}:
        return nn.SiLU()
    if key in {"gelu"}:
        return nn.GELU()
    if key in {"elu"}:
        return nn.ELU(alpha=float(elu_alpha))
    if key in {"none", "identity", "linear"}:
        return nn.Identity()
    raise ValueError(
        f"Unsupported activation '{name}'. "
        "Use one of: relu, leaky_relu, silu, gelu, elu, identity."
    )


def _make_mlp(
    *,
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    activation: str,
    activation_negative_slope: float,
    activation_elu_alpha: float,
    dropout: float,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        _make_activation(
            activation,
            negative_slope=float(activation_negative_slope),
            elu_alpha=float(activation_elu_alpha),
        ),
        nn.Dropout(p=float(dropout)),
        nn.Linear(hidden_dim, out_dim),
    )


def _resolve_state_reference(
    *,
    x: torch.Tensor,
    state_residual: torch.Tensor | None,
) -> torch.Tensor:
    """Resolve base state used to reconstruct absolute next-state predictions.

    By default we use `x[:, :1]` (the model input state channel). When
    `state_residual` is supplied, it overrides that base and allows callers to
    compose outputs against an external/raw state reference.
    """
    state_input = x[:, :1]
    if state_residual is None:
        return state_input
    if state_residual.ndim != 2 or int(state_residual.shape[1]) != 1:
        raise ValueError(
            "state_residual must have shape [N,1], got "
            f"{tuple(state_residual.shape)}."
        )
    if int(state_residual.shape[0]) != int(state_input.shape[0]):
        raise ValueError(
            "state_residual row count mismatch: "
            f"state_residual has {state_residual.shape[0]} rows, expected "
            f"{state_input.shape[0]}."
        )
    return state_residual.to(device=state_input.device, dtype=state_input.dtype)


class AdvectionSAGEModel(nn.Module):
    """Simplified GraphSAGE model for 1D advection state/rate prediction."""

    def __init__(
        self,
        *,
        in_channels: int = 1,
        hidden: int = 64,
        layers: int = 2,
        dropout: float = 0.0,
        activation: str = "relu",
        activation_negative_slope: float = 0.01,
        activation_elu_alpha: float = 1.0,
        use_skip: bool = True,
        use_layernorm: bool = False,
        layernorm_eps: float = 1e-6,
        predict_type: str = PREDICT_TYPE_STATE,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be > 0, got {in_channels}.")
        if hidden <= 0:
            raise ValueError(f"hidden must be > 0, got {hidden}.")
        if layers < 1:
            raise ValueError(f"layers must be >= 1, got {layers}.")
        if predict_type not in PREDICT_TYPES:
            raise ValueError(
                f"predict_type must be one of {sorted(PREDICT_TYPES)}, got '{predict_type}'."
            )

        self.in_channels = int(in_channels)
        self.hidden = int(hidden)
        self.layers = int(layers)
        self.dropout = float(dropout)
        self.predict_type = str(predict_type).strip().lower()
        self.use_skip = bool(use_skip)
        self.use_layernorm = bool(use_layernorm)
        self.block_activation = _make_activation(
            activation,
            negative_slope=float(activation_negative_slope),
            elu_alpha=float(activation_elu_alpha),
        )

        dims = [self.in_channels] + [self.hidden] * self.layers
        self.convs = nn.ModuleList()
        self.skip_proj = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(len(dims) - 1):
            in_ch = int(dims[i])
            out_ch = int(dims[i + 1])
            self.convs.append(SAGEConv(in_ch, out_ch))
            if self.use_skip:
                if in_ch == out_ch:
                    self.skip_proj.append(nn.Identity())
                else:
                    self.skip_proj.append(nn.Linear(in_ch, out_ch, bias=False))
            else:
                self.skip_proj.append(nn.Identity())
            if self.use_layernorm:
                self.norms.append(nn.LayerNorm(out_ch, eps=float(layernorm_eps)))
            else:
                self.norms.append(nn.Identity())

        self.head = nn.Sequential(
            nn.Linear(self.hidden, self.hidden),
            _make_activation(
                activation,
                negative_slope=float(activation_negative_slope),
                elu_alpha=float(activation_elu_alpha),
            ),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        edge_attr: torch.Tensor | None = None,
        dt: float | None = None,
    ) -> torch.Tensor:
        """Run forward pass.

        Args:
            x: Node states [N, 1].
            edge_index: Edge list [2, E].
            edge_attr: Optional edge features [E, D]. Ignored by SAGE model.
            dt: Required only when predict_type='rate' to map rate -> state.

        Returns:
            Predicted field [N, 1].
            For predict_type='state', this is u_next.
            For predict_type='delta', this is delta.
            For predict_type='rate', this is rate.
        """
        state = x[:, :1]
        h = x
        for li, conv in enumerate(self.convs):
            h_res = h
            h = conv(h, edge_index)
            if self.use_skip:
                h = h + self.skip_proj[li](h_res)
            h = self.norms[li](h)
            h = self.block_activation(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        out = self.head(h)
        if self.predict_type == PREDICT_TYPE_STATE:
            # Residual-state formulation: predict delta and add to current state.
            return state + out
        if self.predict_type == PREDICT_TYPE_DELTA:
            return out
        if self.predict_type == PREDICT_TYPE_RATE:
            return out
        raise RuntimeError(f"Unexpected predict_type='{self.predict_type}'.")

    def predict_state(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        edge_attr: torch.Tensor | None = None,
        dt: float,
        state_residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Always return predicted next state u_{n+1} regardless of predict_type."""
        state_input = x[:, :1]
        state_base = _resolve_state_reference(x=x, state_residual=state_residual)
        y = self.forward(x, edge_index, edge_attr=edge_attr, dt=dt)
        if self.predict_type == PREDICT_TYPE_STATE:
            return y + (state_base - state_input)
        if self.predict_type == PREDICT_TYPE_DELTA:
            return state_base + y
        if self.predict_type == PREDICT_TYPE_RATE:
            return state_base + float(dt) * y
        raise RuntimeError(f"Unexpected predict_type='{self.predict_type}'.")


class MeshGraphNetProcessorBlock(nn.Module):
    """One edge-update + node-update processor block."""

    def __init__(
        self,
        *,
        hidden: int,
        dropout: float,
        activation: str,
        activation_negative_slope: float,
        activation_elu_alpha: float,
        use_layernorm: bool,
        layernorm_eps: float,
    ) -> None:
        super().__init__()
        self.edge_mlp = _make_mlp(
            in_dim=3 * int(hidden),
            hidden_dim=int(hidden),
            out_dim=int(hidden),
            activation=activation,
            activation_negative_slope=float(activation_negative_slope),
            activation_elu_alpha=float(activation_elu_alpha),
            dropout=float(dropout),
        )
        self.node_mlp = _make_mlp(
            in_dim=2 * int(hidden),
            hidden_dim=int(hidden),
            out_dim=int(hidden),
            activation=activation,
            activation_negative_slope=float(activation_negative_slope),
            activation_elu_alpha=float(activation_elu_alpha),
            dropout=float(dropout),
        )
        if use_layernorm:
            self.edge_norm = nn.LayerNorm(int(hidden), eps=float(layernorm_eps))
            self.node_norm = nn.LayerNorm(int(hidden), eps=float(layernorm_eps))
        else:
            self.edge_norm = nn.Identity()
            self.node_norm = nn.Identity()

    def forward(
        self,
        node_latent: torch.Tensor,
        edge_index: torch.Tensor,
        edge_latent: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        src = edge_index[0].long()
        dst = edge_index[1].long()

        edge_inputs = torch.cat(
            [node_latent[src], node_latent[dst], edge_latent],
            dim=-1,
        )
        edge_latent = edge_latent + self.edge_mlp(edge_inputs)
        edge_latent = self.edge_norm(edge_latent)

        agg = torch.zeros_like(node_latent)
        agg.index_add_(0, dst, edge_latent)
        node_inputs = torch.cat([node_latent, agg], dim=-1)
        node_latent = node_latent + self.node_mlp(node_inputs)
        node_latent = self.node_norm(node_latent)
        return node_latent, edge_latent


class AdvectionMeshGraphNetModel(nn.Module):
    """MeshGraphNet-style model that explicitly uses edge attributes.

    ``use_absolute_edge_attr=True`` removes the signs from geometric edge
    features before edge encoding. This provides a controlled comparison for
    measuring the contribution of signed edge directionality.
    """

    def __init__(
        self,
        *,
        in_channels: int = 1,
        edge_attr_channels: int = 1,
        hidden: int = 64,
        layers: int = 2,
        dropout: float = 0.0,
        activation: str = "relu",
        activation_negative_slope: float = 0.01,
        activation_elu_alpha: float = 1.0,
        use_layernorm: bool = False,
        layernorm_eps: float = 1e-6,
        predict_type: str = PREDICT_TYPE_STATE,
        use_absolute_edge_attr: bool = False,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be > 0, got {in_channels}.")
        if edge_attr_channels <= 0:
            raise ValueError(
                f"edge_attr_channels must be > 0, got {edge_attr_channels}."
            )
        if hidden <= 0:
            raise ValueError(f"hidden must be > 0, got {hidden}.")
        if layers < 1:
            raise ValueError(f"layers must be >= 1, got {layers}.")
        if predict_type not in PREDICT_TYPES:
            raise ValueError(
                f"predict_type must be one of {sorted(PREDICT_TYPES)}, got '{predict_type}'."
            )

        self.in_channels = int(in_channels)
        self.edge_attr_channels = int(edge_attr_channels)
        self.hidden = int(hidden)
        self.layers = int(layers)
        self.dropout = float(dropout)
        self.predict_type = str(predict_type).strip().lower()
        self.use_absolute_edge_attr = bool(use_absolute_edge_attr)
        self.block_activation = _make_activation(
            activation,
            negative_slope=float(activation_negative_slope),
            elu_alpha=float(activation_elu_alpha),
        )

        self.node_encoder = _make_mlp(
            in_dim=self.in_channels,
            hidden_dim=self.hidden,
            out_dim=self.hidden,
            activation=activation,
            activation_negative_slope=float(activation_negative_slope),
            activation_elu_alpha=float(activation_elu_alpha),
            dropout=self.dropout,
        )
        self.edge_encoder = _make_mlp(
            in_dim=self.edge_attr_channels,
            hidden_dim=self.hidden,
            out_dim=self.hidden,
            activation=activation,
            activation_negative_slope=float(activation_negative_slope),
            activation_elu_alpha=float(activation_elu_alpha),
            dropout=self.dropout,
        )
        self.processor = nn.ModuleList(
            [
                MeshGraphNetProcessorBlock(
                    hidden=self.hidden,
                    dropout=self.dropout,
                    activation=activation,
                    activation_negative_slope=float(activation_negative_slope),
                    activation_elu_alpha=float(activation_elu_alpha),
                    use_layernorm=bool(use_layernorm),
                    layernorm_eps=float(layernorm_eps),
                )
                for _ in range(self.layers)
            ]
        )
        self.head = nn.Sequential(
            nn.Linear(self.hidden, self.hidden),
            _make_activation(
                activation,
                negative_slope=float(activation_negative_slope),
                elu_alpha=float(activation_elu_alpha),
            ),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        edge_attr: torch.Tensor | None = None,
        dt: float | None = None,
    ) -> torch.Tensor:
        state = x[:, :1]
        if edge_attr is None:
            raise ValueError(
                "AdvectionMeshGraphNetModel requires edge_attr in forward/predict_state."
            )
        if edge_attr.ndim != 2:
            raise ValueError(
                f"edge_attr must be rank-2 [E,D], got shape={tuple(edge_attr.shape)}."
            )
        if edge_attr.shape[1] != self.edge_attr_channels:
            raise ValueError(
                "edge_attr channel mismatch: "
                f"expected {self.edge_attr_channels}, got {edge_attr.shape[1]}."
            )
        if edge_attr.shape[0] != edge_index.shape[1]:
            raise ValueError(
                "edge_attr edge count mismatch: "
                f"edge_attr has {edge_attr.shape[0]} rows, edge_index has {edge_index.shape[1]} edges."
            )

        node_latent = self.node_encoder(x)
        edge_encoder_input = (
            torch.abs(edge_attr) if self.use_absolute_edge_attr else edge_attr
        )
        edge_latent = self.edge_encoder(edge_encoder_input)
        for block in self.processor:
            node_latent, edge_latent = block(node_latent, edge_index, edge_latent)
            node_latent = self.block_activation(node_latent)
            node_latent = F.dropout(
                node_latent,
                p=self.dropout,
                training=self.training,
            )

        out = self.head(node_latent)
        if self.predict_type == PREDICT_TYPE_STATE:
            return state + out
        if self.predict_type == PREDICT_TYPE_DELTA:
            return out
        if self.predict_type == PREDICT_TYPE_RATE:
            return out
        raise RuntimeError(f"Unexpected predict_type='{self.predict_type}'.")

    def predict_state(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        edge_attr: torch.Tensor | None = None,
        dt: float,
        state_residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        state_input = x[:, :1]
        state_base = _resolve_state_reference(x=x, state_residual=state_residual)
        y = self.forward(x, edge_index, edge_attr=edge_attr, dt=dt)
        if self.predict_type == PREDICT_TYPE_STATE:
            return y + (state_base - state_input)
        if self.predict_type == PREDICT_TYPE_DELTA:
            return state_base + y
        if self.predict_type == PREDICT_TYPE_RATE:
            return state_base + float(dt) * y
        raise RuntimeError(f"Unexpected predict_type='{self.predict_type}'.")


class _ConservativeFluxDecoderMixin:
    """Shared conservative edge-flux decoder for latent node processors."""

    def _init_conservative_flux_decoder(
        self,
        *,
        edge_attr_channels: int,
        hidden: int,
        activation: str,
        activation_negative_slope: float,
        activation_elu_alpha: float,
        dropout: float,
        use_source: bool,
        signed_edge_channel: int,
    ) -> None:
        if edge_attr_channels <= 0:
            raise ValueError(
                f"edge_attr_channels must be > 0, got {edge_attr_channels}."
            )
        if signed_edge_channel < 0 or signed_edge_channel >= int(edge_attr_channels):
            raise ValueError(
                "signed_edge_channel must index an edge_attr column, got "
                f"signed_edge_channel={signed_edge_channel} with "
                f"edge_attr_channels={edge_attr_channels}."
            )

        self.edge_attr_channels = int(edge_attr_channels)
        self.use_source = bool(use_source)
        self.signed_edge_channel = int(signed_edge_channel)
        self._flux_pair_cache: dict[
            tuple[int, int, str],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None,
        ] = {}

        # Symmetric pair features + signed orientation => anti-symmetric edge flux.
        self.flux_head = _make_mlp(
            in_dim=(2 * int(hidden)) + self.edge_attr_channels,
            hidden_dim=int(hidden),
            out_dim=1,
            activation=activation,
            activation_negative_slope=float(activation_negative_slope),
            activation_elu_alpha=float(activation_elu_alpha),
            dropout=float(dropout),
        )
        if self.use_source:
            self.source_head = _make_mlp(
                in_dim=int(hidden),
                hidden_dim=int(hidden),
                out_dim=1,
                activation=activation,
                activation_negative_slope=float(activation_negative_slope),
                activation_elu_alpha=float(activation_elu_alpha),
                dropout=float(dropout),
            )
        else:
            self.source_head = None

    def _validate_flux_edge_attr(
        self,
        *,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None,
        model_name: str,
    ) -> torch.Tensor:
        if edge_attr is None:
            raise ValueError(
                f"{model_name} requires edge_attr in forward/predict_state."
            )
        if edge_attr.ndim != 2:
            raise ValueError(
                f"edge_attr must be rank-2 [E,D], got shape={tuple(edge_attr.shape)}."
            )
        if edge_attr.shape[1] != self.edge_attr_channels:
            raise ValueError(
                "edge_attr channel mismatch: "
                f"expected {self.edge_attr_channels}, got {edge_attr.shape[1]}."
            )
        if edge_attr.shape[0] != edge_index.shape[1]:
            raise ValueError(
                "edge_attr edge count mismatch: "
                f"edge_attr has {edge_attr.shape[0]} rows, "
                f"edge_index has {edge_index.shape[1]} edges."
            )
        return edge_attr

    def _get_cached_ring_flux_pairing(
        self,
        *,
        edge_index: torch.Tensor,
        signed: torch.Tensor,
        num_nodes: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        num_edges = int(edge_index.shape[1])
        cache_key = (num_nodes, num_edges, str(edge_index.device))
        cached = self._flux_pair_cache.get(cache_key, None)
        if cached is not None or cache_key in self._flux_pair_cache:
            return cached

        if num_edges != (2 * num_nodes):
            self._flux_pair_cache[cache_key] = None
            return None

        src = edge_index[0].long()
        dst = edge_index[1].long()
        src_pos = src[0::2]
        src_neg = src[1::2]
        dst_pos = dst[0::2]
        signed_flat = signed.squeeze(1)
        signed_pos = signed_flat[0::2]
        signed_neg = signed_flat[1::2]
        reverse_idx = (2 * dst_pos) + 1

        layout_ok = torch.all(src_pos == src_neg)
        if num_nodes > 0:
            layout_ok = layout_ok & (src_pos[0] == 0)
        if num_nodes > 1:
            layout_ok = layout_ok & torch.all(src_pos[1:] == (src_pos[:-1] + 1))
        layout_ok = layout_ok & torch.all(signed_pos > 0)
        layout_ok = layout_ok & torch.all(signed_neg < 0)
        layout_ok = layout_ok & torch.all((dst_pos >= 0) & (dst_pos < num_nodes))
        layout_ok = layout_ok & torch.all(src[reverse_idx] == dst_pos)
        layout_ok = layout_ok & torch.all(dst[reverse_idx] == src_pos)
        layout_ok = layout_ok & torch.isclose(
            signed_flat[reverse_idx],
            -signed_pos,
            atol=1e-6,
            rtol=0.0,
        ).all()

        if not bool(layout_ok.item()):
            self._flux_pair_cache[cache_key] = None
            return None

        rep_idx = torch.arange(0, num_edges, 2, device=edge_index.device, dtype=torch.long)
        cached = (rep_idx, src_pos, dst_pos, signed_pos[:, None])
        self._flux_pair_cache[cache_key] = cached
        return cached

    def _compute_conservative_delta(
        self,
        *,
        node_latent: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        src = edge_index[0].long()
        dst = edge_index[1].long()
        h_src = node_latent[src]
        h_dst = node_latent[dst]
        pair_mean = 0.5 * (h_src + h_dst)
        pair_absdiff = torch.abs(h_src - h_dst)
        edge_abs = torch.abs(edge_attr)
        flux_inputs = torch.cat([pair_mean, pair_absdiff, edge_abs], dim=-1)
        signed = edge_attr[:, self.signed_edge_channel : self.signed_edge_channel + 1]
        signed = signed.to(device=node_latent.device, dtype=node_latent.dtype)

        num_nodes = int(node_latent.shape[0])
        ring_pairing = self._get_cached_ring_flux_pairing(
            edge_index=edge_index,
            signed=signed,
            num_nodes=num_nodes,
        )
        if ring_pairing is not None:
            rep_idx, src_pos, dst_pos, signed_pos = ring_pairing
            flux_mag = self.flux_head(flux_inputs[rep_idx])  # [E_u,1]
            paired_flux = signed_pos.to(
                device=flux_mag.device,
                dtype=flux_mag.dtype,
            ) * flux_mag
            delta = torch.zeros(
                (node_latent.shape[0], 1),
                device=node_latent.device,
                dtype=node_latent.dtype,
            )
            delta.index_add_(0, src_pos, -paired_flux)
            delta.index_add_(0, dst_pos, paired_flux)
            return delta

        pair_lo = torch.minimum(src, dst)
        pair_hi = torch.maximum(src, dst)
        pair_key = pair_lo * int(num_nodes) + pair_hi
        unique_keys, inverse, counts = torch.unique(
            pair_key,
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        sort_perm = torch.argsort(pair_key)
        group_start = torch.cumsum(counts, dim=0) - counts
        rep_idx = sort_perm[group_start]

        # Deduplicate only true reverse-edge pairs; all other edge groups keep the
        # original directed-edge behavior to avoid changing semantics on graphs
        # that are not fully bidirected.
        signed_sum = torch.zeros(
            (int(unique_keys.shape[0]),),
            device=node_latent.device,
            dtype=node_latent.dtype,
        )
        signed_sum.index_add_(0, inverse, signed.squeeze(1))
        paired_groups = (
            (counts == 2)
            & (pair_lo[rep_idx] != pair_hi[rep_idx])
            & (torch.abs(signed_sum) <= 1e-6)
        )
        paired_edge_mask = paired_groups[inverse]

        delta = torch.zeros(
            (node_latent.shape[0], 1),
            device=node_latent.device,
            dtype=node_latent.dtype,
        )

        if torch.any(paired_groups):
            rep_idx_paired = rep_idx[paired_groups]
            flux_mag_paired = self.flux_head(flux_inputs[rep_idx_paired])  # [E_u,1]
            signed_paired = signed[rep_idx_paired].to(
                device=flux_mag_paired.device,
                dtype=flux_mag_paired.dtype,
            )
            paired_flux = signed_paired * flux_mag_paired
            delta.index_add_(0, src[rep_idx_paired], -paired_flux)
            delta.index_add_(0, dst[rep_idx_paired], paired_flux)

        if torch.any(~paired_edge_mask):
            fallback_idx = (~paired_edge_mask).nonzero(as_tuple=False).squeeze(1)
            flux_mag_fallback = self.flux_head(flux_inputs[fallback_idx])  # [E_f,1]
            signed_fallback = signed[fallback_idx].to(
                device=flux_mag_fallback.device,
                dtype=flux_mag_fallback.dtype,
            )
            fallback_flux = signed_fallback * flux_mag_fallback
            delta.index_add_(0, src[fallback_idx], -fallback_flux)

        return delta

    def _decode_flux_prediction(
        self,
        *,
        state: torch.Tensor,
        node_latent: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        dt: float | None,
        model_name: str,
    ) -> torch.Tensor:
        delta_transport = self._compute_conservative_delta(
            node_latent=node_latent,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )
        if self.source_head is not None:
            delta = delta_transport + self.source_head(node_latent)
        else:
            delta = delta_transport

        if self.predict_type == PREDICT_TYPE_STATE:
            return state + delta
        if self.predict_type == PREDICT_TYPE_DELTA:
            return delta
        if self.predict_type == PREDICT_TYPE_RATE:
            if dt is None:
                raise ValueError(
                    f"dt is required when predict_type='rate' for {model_name}."
                )
            return delta / float(dt)
        raise RuntimeError(f"Unexpected predict_type='{self.predict_type}'.")

    def predict_state(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        edge_attr: torch.Tensor | None = None,
        dt: float,
        state_residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        state_input = x[:, :1]
        state_base = _resolve_state_reference(x=x, state_residual=state_residual)
        y = self.forward(x, edge_index, edge_attr=edge_attr, dt=dt)
        if self.predict_type == PREDICT_TYPE_STATE:
            return y + (state_base - state_input)
        if self.predict_type == PREDICT_TYPE_DELTA:
            return state_base + y
        if self.predict_type == PREDICT_TYPE_RATE:
            return state_base + float(dt) * y
        raise RuntimeError(f"Unexpected predict_type='{self.predict_type}'.")


class AdvectionSAGEFluxModel(_ConservativeFluxDecoderMixin, nn.Module):
    """GraphSAGE processor with a conservative edge-flux decoder.

    GraphSAGE builds node latents while the shared FluxGraphNet decoder predicts
    anti-symmetric edge fluxes and accumulates their divergence at each node.
    Without a learned source term, the resulting update conserves total mass on
    bidirected graphs.
    """

    def __init__(
        self,
        *,
        in_channels: int = 1,
        edge_attr_channels: int = 1,
        hidden: int = 64,
        layers: int = 2,
        dropout: float = 0.0,
        activation: str = "relu",
        activation_negative_slope: float = 0.01,
        activation_elu_alpha: float = 1.0,
        use_skip: bool = True,
        use_layernorm: bool = False,
        layernorm_eps: float = 1e-6,
        predict_type: str = PREDICT_TYPE_STATE,
        use_source: bool = False,
        signed_edge_channel: int = 0,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be > 0, got {in_channels}.")
        if hidden <= 0:
            raise ValueError(f"hidden must be > 0, got {hidden}.")
        if layers < 1:
            raise ValueError(f"layers must be >= 1, got {layers}.")
        if predict_type not in PREDICT_TYPES:
            raise ValueError(
                f"predict_type must be one of {sorted(PREDICT_TYPES)}, got '{predict_type}'."
            )

        self.in_channels = int(in_channels)
        self.hidden = int(hidden)
        self.layers = int(layers)
        self.dropout = float(dropout)
        self.predict_type = str(predict_type).strip().lower()
        self.use_skip = bool(use_skip)
        self.use_layernorm = bool(use_layernorm)
        self.block_activation = _make_activation(
            activation,
            negative_slope=float(activation_negative_slope),
            elu_alpha=float(activation_elu_alpha),
        )

        dims = [self.in_channels] + [self.hidden] * self.layers
        self.convs = nn.ModuleList()
        self.skip_proj = nn.ModuleList()
        self.norms = nn.ModuleList()
        for in_ch, out_ch in zip(dims[:-1], dims[1:]):
            self.convs.append(SAGEConv(int(in_ch), int(out_ch)))
            if self.use_skip and in_ch != out_ch:
                self.skip_proj.append(nn.Linear(int(in_ch), int(out_ch), bias=False))
            else:
                self.skip_proj.append(nn.Identity())
            if self.use_layernorm:
                self.norms.append(nn.LayerNorm(int(out_ch), eps=float(layernorm_eps)))
            else:
                self.norms.append(nn.Identity())

        self._init_conservative_flux_decoder(
            edge_attr_channels=int(edge_attr_channels),
            hidden=self.hidden,
            activation=activation,
            activation_negative_slope=float(activation_negative_slope),
            activation_elu_alpha=float(activation_elu_alpha),
            dropout=self.dropout,
            use_source=bool(use_source),
            signed_edge_channel=int(signed_edge_channel),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        edge_attr: torch.Tensor | None = None,
        dt: float | None = None,
    ) -> torch.Tensor:
        edge_attr = self._validate_flux_edge_attr(
            edge_index=edge_index,
            edge_attr=edge_attr,
            model_name=self.__class__.__name__,
        )
        node_latent = x
        for li, conv in enumerate(self.convs):
            node_residual = node_latent
            node_latent = conv(node_latent, edge_index)
            if self.use_skip:
                node_latent = node_latent + self.skip_proj[li](node_residual)
            node_latent = self.norms[li](node_latent)
            node_latent = self.block_activation(node_latent)
            node_latent = F.dropout(
                node_latent,
                p=self.dropout,
                training=self.training,
            )

        return self._decode_flux_prediction(
            state=x[:, :1],
            node_latent=node_latent,
            edge_index=edge_index,
            edge_attr=edge_attr,
            dt=dt,
            model_name=self.__class__.__name__,
        )


class AdvectionFluxGraphNetModel(_ConservativeFluxDecoderMixin, nn.Module):
    """MeshGraphNet processor with a conservative edge-flux decoder.

    This model predicts edge-wise transported quantity on directed edges and
    updates node states through a divergence-like accumulation. When
    ``use_source=False``, the transport contribution is conservative over
    bidirected graphs.
    """

    def __init__(
        self,
        *,
        in_channels: int = 1,
        edge_attr_channels: int = 1,
        hidden: int = 64,
        layers: int = 2,
        dropout: float = 0.0,
        activation: str = "relu",
        activation_negative_slope: float = 0.01,
        activation_elu_alpha: float = 1.0,
        use_layernorm: bool = False,
        layernorm_eps: float = 1e-6,
        predict_type: str = PREDICT_TYPE_STATE,
        use_source: bool = False,
        signed_edge_channel: int = 0,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be > 0, got {in_channels}.")
        if hidden <= 0:
            raise ValueError(f"hidden must be > 0, got {hidden}.")
        if layers < 1:
            raise ValueError(f"layers must be >= 1, got {layers}.")
        if predict_type not in PREDICT_TYPES:
            raise ValueError(
                f"predict_type must be one of {sorted(PREDICT_TYPES)}, got '{predict_type}'."
            )

        self.in_channels = int(in_channels)
        self.hidden = int(hidden)
        self.layers = int(layers)
        self.dropout = float(dropout)
        self.predict_type = str(predict_type).strip().lower()
        self.block_activation = _make_activation(
            activation,
            negative_slope=float(activation_negative_slope),
            elu_alpha=float(activation_elu_alpha),
        )

        self.node_encoder = _make_mlp(
            in_dim=self.in_channels,
            hidden_dim=self.hidden,
            out_dim=self.hidden,
            activation=activation,
            activation_negative_slope=float(activation_negative_slope),
            activation_elu_alpha=float(activation_elu_alpha),
            dropout=self.dropout,
        )
        self.edge_encoder = _make_mlp(
            in_dim=int(edge_attr_channels),
            hidden_dim=self.hidden,
            out_dim=self.hidden,
            activation=activation,
            activation_negative_slope=float(activation_negative_slope),
            activation_elu_alpha=float(activation_elu_alpha),
            dropout=self.dropout,
        )
        self.processor = nn.ModuleList(
            [
                MeshGraphNetProcessorBlock(
                    hidden=self.hidden,
                    dropout=self.dropout,
                    activation=activation,
                    activation_negative_slope=float(activation_negative_slope),
                    activation_elu_alpha=float(activation_elu_alpha),
                    use_layernorm=bool(use_layernorm),
                    layernorm_eps=float(layernorm_eps),
                )
                for _ in range(self.layers)
            ]
        )
        self._init_conservative_flux_decoder(
            edge_attr_channels=int(edge_attr_channels),
            hidden=self.hidden,
            activation=activation,
            activation_negative_slope=float(activation_negative_slope),
            activation_elu_alpha=float(activation_elu_alpha),
            dropout=self.dropout,
            use_source=bool(use_source),
            signed_edge_channel=int(signed_edge_channel),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        edge_attr: torch.Tensor | None = None,
        dt: float | None = None,
    ) -> torch.Tensor:
        edge_attr = self._validate_flux_edge_attr(
            edge_index=edge_index,
            edge_attr=edge_attr,
            model_name=self.__class__.__name__,
        )

        node_latent = self.node_encoder(x)
        edge_latent = self.edge_encoder(edge_attr)
        for block in self.processor:
            node_latent, edge_latent = block(node_latent, edge_index, edge_latent)
            node_latent = self.block_activation(node_latent)
            node_latent = F.dropout(
                node_latent,
                p=self.dropout,
                training=self.training,
            )

        return self._decode_flux_prediction(
            state=x[:, :1],
            node_latent=node_latent,
            edge_index=edge_index,
            edge_attr=edge_attr,
            dt=dt,
            model_name=self.__class__.__name__,
        )


def build_model_from_config(
    cfg: Mapping[str, Any],
    *,
    in_channels: int = 1,
    edge_attr_channels: int | None = None,
) -> nn.Module:
    """Factory with a config shape similar to larger project style."""
    model_cfg = dict(cfg.get("model", cfg))
    model_type = model_cfg.get("type", None)
    name = str(model_cfg.get("name", "AdvectionSAGEModel"))
    sage_names = {"advectionsagemodel", "featurenet", "sageconv"}
    sage_flux_names = {
        "advectionsagefluxmodel",
        "sage-flux",
        "sage_flux",
        "sageflux",
    }
    mesh_names = {"advectionmeshgraphnetmodel", "meshgraphnet", "mgn"}
    abs_mesh_names = {
        "absolute-meshgraphnet",
        "meshgraphnet-abs",
        "meshgraphnet_abs",
        "meshgraphnetabs",
        "mgn-abs",
    }
    flux_names = {"advectionfluxgraphnetmodel", "fluxgraphnet", "fluxgnn", "flux"}

    if model_type is None:
        family_key = name.strip().lower()
    else:
        type_key = str(model_type).strip().lower()
        if type_key in sage_names:
            family_key = "sageconv"
        elif type_key in sage_flux_names:
            family_key = "sage-flux"
        elif type_key in abs_mesh_names:
            family_key = "meshgraphnet-abs"
        elif type_key in mesh_names:
            family_key = "meshgraphnet"
        elif type_key in flux_names:
            family_key = "fluxgraphnet"
        else:
            raise ValueError(
                f"Unsupported model.type '{model_type}'. "
                "Use one of: sageconv, sage-flux, meshgraphnet, "
                "meshgraphnet-abs, fluxgraphnet."
            )

    kwargs = dict(model_cfg)
    kwargs.pop("name", None)
    kwargs.pop("type", None)
    kwargs.pop("state_residual_mode", None)
    # Allow annotated JSON configs: ignore comment/helper keys.
    kwargs = {k: v for k, v in kwargs.items() if not str(k).startswith("_")}
    kwargs["in_channels"] = int(in_channels)

    def _filter_constructor_kwargs(
        ctor: type[nn.Module], values: Mapping[str, Any]
    ) -> Dict[str, Any]:
        sig = inspect.signature(ctor.__init__)
        valid = {k for k in sig.parameters.keys() if k != "self"}
        unknown = sorted(set(values.keys()) - valid)
        if unknown:
            print(
                "[WARN] Ignoring unknown model config keys for "
                f"{ctor.__name__}: {unknown}",
                flush=True,
            )
        return {k: values[k] for k in values.keys() if k in valid}

    if family_key in sage_names:
        kwargs.pop("edge_attr_channels", None)
        kwargs = _filter_constructor_kwargs(AdvectionSAGEModel, kwargs)
        return AdvectionSAGEModel(**kwargs)

    if family_key in sage_flux_names:
        cfg_edge_attr_channels = kwargs.get("edge_attr_channels", None)
        resolved_edge_attr_channels = cfg_edge_attr_channels
        if resolved_edge_attr_channels is None:
            resolved_edge_attr_channels = edge_attr_channels
        if resolved_edge_attr_channels is None:
            raise ValueError(
                "SAGE-Flux model requires edge_attr_channels. "
                "Provide model.edge_attr_channels in config or pass edge_attr_channels "
                "to build_model_from_config()."
            )
        if (
            edge_attr_channels is not None
            and cfg_edge_attr_channels is not None
            and int(cfg_edge_attr_channels) != int(edge_attr_channels)
        ):
            raise ValueError(
                "Conflicting edge_attr_channels between config and data: "
                f"config={cfg_edge_attr_channels}, data={edge_attr_channels}."
            )
        kwargs["edge_attr_channels"] = int(resolved_edge_attr_channels)
        kwargs = _filter_constructor_kwargs(AdvectionSAGEFluxModel, kwargs)
        return AdvectionSAGEFluxModel(**kwargs)

    if family_key in mesh_names or family_key in abs_mesh_names:
        # `use_skip` is a GraphSAGE-only option; ignore it for MeshGraphNet.
        kwargs.pop("use_skip", None)
        if family_key in abs_mesh_names:
            kwargs["use_absolute_edge_attr"] = True
        cfg_edge_attr_channels = kwargs.get("edge_attr_channels", None)
        resolved_edge_attr_channels = cfg_edge_attr_channels
        if resolved_edge_attr_channels is None:
            resolved_edge_attr_channels = edge_attr_channels
        if resolved_edge_attr_channels is None:
            raise ValueError(
                "MeshGraphNet model requires edge_attr_channels. "
                "Provide model.edge_attr_channels in config or pass edge_attr_channels "
                "to build_model_from_config()."
            )
        if (
            edge_attr_channels is not None
            and cfg_edge_attr_channels is not None
            and int(cfg_edge_attr_channels) != int(edge_attr_channels)
        ):
            raise ValueError(
                "Conflicting edge_attr_channels between config and data: "
                f"config={cfg_edge_attr_channels}, data={edge_attr_channels}."
            )
        kwargs["edge_attr_channels"] = int(resolved_edge_attr_channels)
        kwargs = _filter_constructor_kwargs(AdvectionMeshGraphNetModel, kwargs)
        return AdvectionMeshGraphNetModel(**kwargs)

    if family_key in flux_names:
        # `use_skip` is a GraphSAGE-only option; ignore it for flux graph model.
        kwargs.pop("use_skip", None)
        cfg_edge_attr_channels = kwargs.get("edge_attr_channels", None)
        resolved_edge_attr_channels = cfg_edge_attr_channels
        if resolved_edge_attr_channels is None:
            resolved_edge_attr_channels = edge_attr_channels
        if resolved_edge_attr_channels is None:
            raise ValueError(
                "FluxGraphNet model requires edge_attr_channels. "
                "Provide model.edge_attr_channels in config or pass edge_attr_channels "
                "to build_model_from_config()."
            )
        if (
            edge_attr_channels is not None
            and cfg_edge_attr_channels is not None
            and int(cfg_edge_attr_channels) != int(edge_attr_channels)
        ):
            raise ValueError(
                "Conflicting edge_attr_channels between config and data: "
                f"config={cfg_edge_attr_channels}, data={edge_attr_channels}."
            )
        kwargs["edge_attr_channels"] = int(resolved_edge_attr_channels)
        kwargs = _filter_constructor_kwargs(AdvectionFluxGraphNetModel, kwargs)
        return AdvectionFluxGraphNetModel(**kwargs)

    raise ValueError(
        f"Unsupported model selector (name='{name}', type={model_type}). "
        "Use model.name in {AdvectionSAGEModel, SAGEConv, SAGE-Flux, "
        "AdvectionSAGEFluxModel, MeshGraphNet, MeshGraphNet-Abs, "
        "AdvectionMeshGraphNetModel, FluxGraphNet, AdvectionFluxGraphNetModel} "
        "or model.type in {sageconv, sage-flux, meshgraphnet, "
        "meshgraphnet-abs, fluxgraphnet}."
    )


@dataclass
class EpochMetrics:
    """Scalar metrics for one epoch split."""

    loss: float
    rmse: float
    rel_l2: float
    rollout_state_rel_l2: float | None = None

    def to_dict(self) -> Dict[str, float]:
        metrics = {"loss": self.loss, "rmse": self.rmse, "rel_l2": self.rel_l2}
        if self.rollout_state_rel_l2 is not None:
            metrics["rollout_state_rel_l2"] = self.rollout_state_rel_l2
        return metrics
