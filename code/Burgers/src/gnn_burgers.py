"""GNN utilities for 1D Burgers surrogate training."""

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


def predict_type_from_target_mode(target_mode: str) -> str:
    """Resolve model predict_type from training target_mode."""
    key = str(target_mode).strip().lower()
    if key not in TARGET_MODES:
        raise ValueError(
            f"target_mode must be one of {sorted(TARGET_MODES)}, got '{target_mode}'."
        )
    return key


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
        nn.Linear(int(in_dim), int(hidden_dim)),
        _make_activation(
            activation,
            negative_slope=float(activation_negative_slope),
            elu_alpha=float(activation_elu_alpha),
        ),
        nn.Dropout(p=float(dropout)),
        nn.Linear(int(hidden_dim), int(out_dim)),
    )


class BurgersSAGEModel(nn.Module):
    """GraphSAGE model for 1D Burgers state/rate prediction."""

    def __init__(
        self,
        *,
        in_channels: int = 1,
        state_channel: int = 0,
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
        if state_channel < 0 or state_channel >= in_channels:
            raise ValueError(
                "state_channel must satisfy 0 <= state_channel < in_channels, got "
                f"state_channel={state_channel}, in_channels={in_channels}."
            )
        if hidden <= 0:
            raise ValueError(f"hidden must be > 0, got {hidden}.")
        if layers < 1:
            raise ValueError(f"layers must be >= 1, got {layers}.")
        if not (0.0 <= float(dropout) < 1.0):
            raise ValueError(f"dropout must satisfy 0 <= dropout < 1, got {dropout}.")
        if predict_type not in PREDICT_TYPES:
            raise ValueError(
                f"predict_type must be one of {sorted(PREDICT_TYPES)}, got '{predict_type}'."
            )

        self.in_channels = int(in_channels)
        self.state_channel = int(state_channel)
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
        """Run forward pass and return model-space output."""
        del edge_attr
        del dt

        state = x[:, self.state_channel : self.state_channel + 1]
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
        state_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Always return predicted next state u_{n+1} regardless of predict_type.

        Args:
            state_override:
                Optional current physical state with shape [N,1]. When provided,
                this value is used as the integration base for delta/rate modes
                (and to offset state mode) so feature-space normalization of `u`
                does not leak into state updates.
        """
        state_in = x[:, self.state_channel : self.state_channel + 1]
        if state_override is None:
            state_base = state_in
        else:
            if state_override.ndim != 2 or state_override.shape != state_in.shape:
                raise ValueError(
                    "state_override must have shape matching state channel "
                    f"{tuple(state_in.shape)}, got {tuple(state_override.shape)}."
                )
            state_base = state_override.to(device=x.device, dtype=x.dtype)
        y = self.forward(x, edge_index, edge_attr=edge_attr, dt=dt)
        if self.predict_type == PREDICT_TYPE_STATE:
            if state_override is None:
                return y
            return y + (state_base - state_in)
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


class BurgersMeshGraphNetModel(nn.Module):
    """MeshGraphNet-style model that explicitly uses edge attributes."""

    def __init__(
        self,
        *,
        in_channels: int = 1,
        state_channel: int = 0,
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
        if state_channel < 0 or state_channel >= in_channels:
            raise ValueError(
                "state_channel must satisfy 0 <= state_channel < in_channels, got "
                f"state_channel={state_channel}, in_channels={in_channels}."
            )
        if edge_attr_channels <= 0:
            raise ValueError(
                f"edge_attr_channels must be > 0, got {edge_attr_channels}."
            )
        if hidden <= 0:
            raise ValueError(f"hidden must be > 0, got {hidden}.")
        if layers < 1:
            raise ValueError(f"layers must be >= 1, got {layers}.")
        if not (0.0 <= float(dropout) < 1.0):
            raise ValueError(f"dropout must satisfy 0 <= dropout < 1, got {dropout}.")
        if predict_type not in PREDICT_TYPES:
            raise ValueError(
                f"predict_type must be one of {sorted(PREDICT_TYPES)}, got '{predict_type}'."
            )

        self.in_channels = int(in_channels)
        self.state_channel = int(state_channel)
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
        del dt

        state = x[:, self.state_channel : self.state_channel + 1]
        if edge_attr is None:
            raise ValueError(
                "BurgersMeshGraphNetModel requires edge_attr in forward/predict_state."
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
        state_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        state_in = x[:, self.state_channel : self.state_channel + 1]
        if state_override is None:
            state_base = state_in
        else:
            if state_override.ndim != 2 or state_override.shape != state_in.shape:
                raise ValueError(
                    "state_override must have shape matching state channel "
                    f"{tuple(state_in.shape)}, got {tuple(state_override.shape)}."
                )
            state_base = state_override.to(device=x.device, dtype=x.dtype)
        y = self.forward(x, edge_index, edge_attr=edge_attr, dt=dt)
        if self.predict_type == PREDICT_TYPE_STATE:
            if state_override is None:
                return y
            return y + (state_base - state_in)
        if self.predict_type == PREDICT_TYPE_DELTA:
            return state_base + y
        if self.predict_type == PREDICT_TYPE_RATE:
            return state_base + float(dt) * y
        raise RuntimeError(f"Unexpected predict_type='{self.predict_type}'.")


class BurgersFluxGraphNetModel(nn.Module):
    """Flux-balance graph model with conservative transport update.

    This model predicts edge-wise transported quantity on directed edges and
    updates node states through a divergence-like accumulation:

      delta_i = - sum_{j in N(i)} F_{i->j}
      u_next = u_now + delta

    when `use_source=False`, the transport contribution is conservative over
    bidirected graphs.
    """

    def __init__(
        self,
        *,
        in_channels: int = 1,
        state_channel: int = 0,
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
        if state_channel < 0 or state_channel >= in_channels:
            raise ValueError(
                "state_channel must satisfy 0 <= state_channel < in_channels, got "
                f"state_channel={state_channel}, in_channels={in_channels}."
            )
        if edge_attr_channels <= 0:
            raise ValueError(
                f"edge_attr_channels must be > 0, got {edge_attr_channels}."
            )
        if hidden <= 0:
            raise ValueError(f"hidden must be > 0, got {hidden}.")
        if layers < 1:
            raise ValueError(f"layers must be >= 1, got {layers}.")
        if not (0.0 <= float(dropout) < 1.0):
            raise ValueError(f"dropout must satisfy 0 <= dropout < 1, got {dropout}.")
        if predict_type not in PREDICT_TYPES:
            raise ValueError(
                f"predict_type must be one of {sorted(PREDICT_TYPES)}, got '{predict_type}'."
            )
        if signed_edge_channel < 0 or signed_edge_channel >= int(edge_attr_channels):
            raise ValueError(
                "signed_edge_channel must index an edge_attr column, got "
                f"signed_edge_channel={signed_edge_channel} with "
                f"edge_attr_channels={edge_attr_channels}."
            )

        self.in_channels = int(in_channels)
        self.state_channel = int(state_channel)
        self.edge_attr_channels = int(edge_attr_channels)
        self.hidden = int(hidden)
        self.layers = int(layers)
        self.dropout = float(dropout)
        self.predict_type = str(predict_type).strip().lower()
        self.use_source = bool(use_source)
        self.signed_edge_channel = int(signed_edge_channel)
        self._flux_pair_cache: dict[
            tuple[int, int, str],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None,
        ] = {}
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

        # Symmetric pair features + signed orientation => anti-symmetric edge flux.
        self.flux_head = _make_mlp(
            in_dim=(2 * self.hidden) + self.edge_attr_channels,
            hidden_dim=self.hidden,
            out_dim=1,
            activation=activation,
            activation_negative_slope=float(activation_negative_slope),
            activation_elu_alpha=float(activation_elu_alpha),
            dropout=self.dropout,
        )
        if self.use_source:
            self.source_head = _make_mlp(
                in_dim=self.hidden,
                hidden_dim=self.hidden,
                out_dim=1,
                activation=activation,
                activation_negative_slope=float(activation_negative_slope),
                activation_elu_alpha=float(activation_elu_alpha),
                dropout=self.dropout,
            )
        else:
            self.source_head = None

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

        rep_idx = torch.arange(
            0, num_edges, 2, device=edge_index.device, dtype=torch.long
        )
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

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        edge_attr: torch.Tensor | None = None,
        dt: float | None = None,
    ) -> torch.Tensor:
        state = x[:, self.state_channel : self.state_channel + 1]
        if edge_attr is None:
            raise ValueError(
                "BurgersFluxGraphNetModel requires edge_attr in forward/predict_state."
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
        edge_latent = self.edge_encoder(edge_attr)
        for block in self.processor:
            node_latent, edge_latent = block(node_latent, edge_index, edge_latent)
            node_latent = self.block_activation(node_latent)
            node_latent = F.dropout(
                node_latent,
                p=self.dropout,
                training=self.training,
            )

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
                    "dt is required when predict_type='rate' for BurgersFluxGraphNetModel."
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
        state_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        state_in = x[:, self.state_channel : self.state_channel + 1]
        if state_override is None:
            state_base = state_in
        else:
            if state_override.ndim != 2 or state_override.shape != state_in.shape:
                raise ValueError(
                    "state_override must have shape matching state channel "
                    f"{tuple(state_in.shape)}, got {tuple(state_override.shape)}."
                )
            state_base = state_override.to(device=x.device, dtype=x.dtype)
        y = self.forward(x, edge_index, edge_attr=edge_attr, dt=dt)
        if self.predict_type == PREDICT_TYPE_STATE:
            if state_override is None:
                return y
            return y + (state_base - state_in)
        if self.predict_type == PREDICT_TYPE_DELTA:
            return state_base + y
        if self.predict_type == PREDICT_TYPE_RATE:
            return state_base + float(dt) * y
        raise RuntimeError(f"Unexpected predict_type='{self.predict_type}'.")


def build_model_from_config(
    cfg: Mapping[str, Any],
    *,
    in_channels: int = 1,
    edge_attr_channels: int | None = None,
) -> nn.Module:
    """Factory with config selectors aligned with the 1D advection project."""
    model_cfg = dict(cfg.get("model", cfg))
    model_type = model_cfg.get("type", None)
    name = str(model_cfg.get("name", "BurgersSAGEModel"))
    sage_names = {"burgerssagemodel", "featurenet", "sageconv"}
    mesh_names = {"burgersmeshgraphnetmodel", "meshgraphnet", "mgn"}
    abs_mesh_names = {
        "absolute-meshgraphnet",
        "meshgraphnet-abs",
        "meshgraphnet_abs",
        "meshgraphnetabs",
        "mgn-abs",
    }
    flux_names = {"burgersfluxgraphnetmodel", "fluxgraphnet", "fluxgnn", "flux"}

    if model_type is None:
        family_key = name.strip().lower()
    else:
        type_key = str(model_type).strip().lower()
        if type_key in sage_names:
            family_key = "sageconv"
        elif type_key in abs_mesh_names:
            family_key = "meshgraphnet-abs"
        elif type_key in mesh_names:
            family_key = "meshgraphnet"
        elif type_key in flux_names:
            family_key = "fluxgraphnet"
        else:
            raise ValueError(
                f"Unsupported model.type '{model_type}'. "
                "Use one of: sageconv, meshgraphnet, meshgraphnet-abs, fluxgraphnet."
            )

    kwargs = dict(model_cfg)
    kwargs.pop("name", None)
    kwargs.pop("type", None)
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
        kwargs = _filter_constructor_kwargs(BurgersSAGEModel, kwargs)
        return BurgersSAGEModel(**kwargs)

    if family_key in mesh_names or family_key in abs_mesh_names:
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
        kwargs = _filter_constructor_kwargs(BurgersMeshGraphNetModel, kwargs)
        return BurgersMeshGraphNetModel(**kwargs)

    if family_key in flux_names:
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
        kwargs = _filter_constructor_kwargs(BurgersFluxGraphNetModel, kwargs)
        return BurgersFluxGraphNetModel(**kwargs)

    raise ValueError(
        f"Unsupported model selector (name='{name}', type={model_type}). "
        "Use model.name in {BurgersSAGEModel, SAGEConv, MeshGraphNet, MeshGraphNet-Abs, "
        "BurgersMeshGraphNetModel, FluxGraphNet, BurgersFluxGraphNetModel} "
        "or model.type in {sageconv, meshgraphnet, meshgraphnet-abs, fluxgraphnet}."
    )


@dataclass
class EpochMetrics:
    """Scalar metrics for one epoch split."""

    loss: float
    rmse: float
    rel_l2: float

    def to_dict(self) -> Dict[str, float]:
        return {"loss": self.loss, "rmse": self.rmse, "rel_l2": self.rel_l2}
