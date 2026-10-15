"""SAGEConv-based GNN utilities for heat-equation surrogate training."""

from __future__ import annotations

from dataclasses import dataclass
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


class HeatSAGEModel(nn.Module):
    """GraphSAGE model for heat-equation state/rate prediction."""

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
        dt: float | None = None,
    ) -> torch.Tensor:
        """Run forward pass and return model-space output."""
        del dt  # kept for a consistent call signature with predict_state users

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
        dt: float,
    ) -> torch.Tensor:
        """Always return predicted next state u_{n+1} regardless of predict_type."""
        state = x[:, :1]
        y = self.forward(x, edge_index, dt=dt)
        if self.predict_type == PREDICT_TYPE_STATE:
            return y
        if self.predict_type == PREDICT_TYPE_DELTA:
            return state + y
        if self.predict_type == PREDICT_TYPE_RATE:
            return state + float(dt) * y
        raise RuntimeError(f"Unexpected predict_type='{self.predict_type}'.")


def build_model_from_config(cfg: Mapping[str, Any], *, in_channels: int = 1) -> nn.Module:
    """Factory with a config shape similar to existing projects."""
    model_cfg = dict(cfg.get("model", cfg))
    name = str(model_cfg.get("name", "HeatSAGEModel"))
    if name not in {"HeatSAGEModel", "FeatureNet", "SAGEConv"}:
        raise ValueError(
            f"Unsupported model name '{name}'. Use 'HeatSAGEModel'."
        )
    kwargs = dict(model_cfg)
    kwargs.pop("name", None)
    kwargs["in_channels"] = int(in_channels)
    return HeatSAGEModel(**kwargs)


@dataclass
class EpochMetrics:
    """Scalar metrics for one epoch split."""

    loss: float
    rmse: float
    rel_l2: float

    def to_dict(self) -> Dict[str, float]:
        return {"loss": self.loss, "rmse": self.rmse, "rel_l2": self.rel_l2}
