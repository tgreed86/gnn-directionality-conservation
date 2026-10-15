"""Physics-operator input features for 1D Burgers GNN training.

This mirrors the 1D_advection project pattern, adapted for Burgers:
1. Compute an absolute advection operator term on the current state:
     r_adv_abs ~ -d/dx(0.5 * u^2)
2. Convert to model-input units ("rate" or "delta" form).
3. Concatenate the operator channel onto base node features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch


INPUT_FORM_RATE = "rate"
INPUT_FORM_DELTA = "delta"
INPUT_FORMS = {INPUT_FORM_RATE, INPUT_FORM_DELTA}

ADVECTION_SCHEME_UPWIND = "upwind"
ADVECTION_SCHEME_CENTRAL = "central"
ADVECTION_SCHEMES = {ADVECTION_SCHEME_UPWIND, ADVECTION_SCHEME_CENTRAL}


@dataclass(frozen=True)
class PhysicsInputConfig:
    """Runtime controls for advection-derived node inputs."""

    enabled: bool = False
    include_adv: bool = True
    input_form: str = INPUT_FORM_RATE
    advection_scheme: str = ADVECTION_SCHEME_UPWIND
    adv_all_steps: bool = True
    input_weighted: bool = False
    adv_weight: float = 1.0
    normalize_adv_to_minus1_1: bool = False
    adv_min: float | None = None
    adv_max: float | None = None
    normalize_adv_percentile_low: float = 1.0
    normalize_adv_percentile_high: float = 99.0
    normalize_adv_clip: bool = False
    detach_inputs: bool = True


def resolve_physics_input_cfg(cfg: Mapping[str, Any]) -> PhysicsInputConfig:
    """Resolve physics-input settings from config with backward fallbacks."""
    physics_raw = cfg.get("physics_inputs", {})
    loss_raw = cfg.get("loss", {})

    physics = physics_raw if isinstance(physics_raw, Mapping) else {}
    loss = loss_raw if isinstance(loss_raw, Mapping) else {}

    enabled = bool(
        physics.get(
            "enabled",
            bool(loss.get("parc", False) or loss.get("parc_inputs", False)),
        )
    )
    include_adv = bool(physics.get("include_adv", loss.get("parc_include_adv", True)))

    input_form = str(
        physics.get("input_form", loss.get("parc_input_form", INPUT_FORM_RATE))
    ).strip().lower()
    if input_form not in INPUT_FORMS:
        raise ValueError(
            f"physics_inputs.input_form must be one of {sorted(INPUT_FORMS)}, got '{input_form}'."
        )

    advection_scheme = str(
        physics.get(
            "advection_scheme",
            loss.get("advection_scheme", ADVECTION_SCHEME_UPWIND),
        )
    ).strip().lower()
    if advection_scheme not in ADVECTION_SCHEMES:
        raise ValueError(
            "physics_inputs.advection_scheme must be one of "
            f"{sorted(ADVECTION_SCHEMES)}, got '{advection_scheme}'."
        )

    normalize_adv_to_minus1_1 = bool(
        physics.get("normalize_adv_to_minus1_1", physics.get("normalize_adv", False))
    )
    adv_min_raw = physics.get("adv_min", None)
    adv_max_raw = physics.get("adv_max", None)
    normalize_adv_percentile_low = float(
        physics.get(
            "normalize_adv_percentile_low",
            physics.get("adv_norm_percentile_low", 1.0),
        )
    )
    normalize_adv_percentile_high = float(
        physics.get(
            "normalize_adv_percentile_high",
            physics.get("adv_norm_percentile_high", 99.0),
        )
    )
    if not (0.0 <= normalize_adv_percentile_low < normalize_adv_percentile_high <= 100.0):
        raise ValueError(
            "physics_inputs.normalize_adv_percentile_low/high must satisfy "
            f"0 <= low < high <= 100, got low={normalize_adv_percentile_low}, "
            f"high={normalize_adv_percentile_high}."
        )
    normalize_adv_clip = bool(physics.get("normalize_adv_clip", False))

    return PhysicsInputConfig(
        enabled=enabled,
        include_adv=include_adv,
        input_form=input_form,
        advection_scheme=advection_scheme,
        adv_all_steps=bool(physics.get("adv_all_steps", loss.get("adv_all_steps", True))),
        input_weighted=bool(
            physics.get("input_weighted", loss.get("parc_input_weighted", False))
        ),
        adv_weight=float(physics.get("adv_weight", loss.get("adv_weight", 1.0))),
        normalize_adv_to_minus1_1=normalize_adv_to_minus1_1,
        adv_min=None if adv_min_raw is None else float(adv_min_raw),
        adv_max=None if adv_max_raw is None else float(adv_max_raw),
        normalize_adv_percentile_low=normalize_adv_percentile_low,
        normalize_adv_percentile_high=normalize_adv_percentile_high,
        normalize_adv_clip=normalize_adv_clip,
        detach_inputs=bool(
            physics.get("detach_inputs", loss.get("parc_detach_inputs", True))
        ),
    )


def count_physics_input_channels(physics_cfg: PhysicsInputConfig) -> int:
    """Return number of extra per-node channels appended by physics inputs."""
    if not physics_cfg.enabled:
        return 0
    return int(bool(physics_cfg.include_adv))


def _physics_to_model_units_1d(
    r_abs: torch.Tensor,
    *,
    dt_phys: float,
    input_form: str,
) -> torch.Tensor:
    if input_form == INPUT_FORM_RATE:
        return r_abs
    if input_form == INPUT_FORM_DELTA:
        return float(dt_phys) * r_abs
    raise RuntimeError(f"Unexpected input_form='{input_form}'.")


def _apply_minmax_minus1_1(
    *,
    x: torch.Tensor,
    x_min: float,
    x_max: float,
    clip: bool,
    eps: float = 1e-12,
) -> torch.Tensor:
    denom = float(x_max) - float(x_min)
    if abs(denom) <= float(eps):
        y = torch.zeros_like(x)
    else:
        y = (2.0 * (x - float(x_min)) / denom) - 1.0
    if clip:
        y = torch.clamp(y, min=-1.0, max=1.0)
    return y


def compute_burgers_advection_term_abs_1d(
    *,
    x_state: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    dx: float,
    advection_scheme: str,
) -> torch.Tensor:
    """Compute conservative Burgers advection term in absolute units.

    Approximates:
      r_adv = - d/dx (0.5 * u^2)
    on the periodic 1D ring graph.
    """
    if x_state.ndim != 2:
        raise ValueError(f"x_state must have shape [N,F], got {tuple(x_state.shape)}.")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(
            f"edge_index must have shape [2,E], got {tuple(edge_index.shape)}."
        )
    if edge_attr.ndim != 2 or edge_attr.shape[0] != edge_index.shape[1]:
        raise ValueError(
            "edge_attr must have shape [E,D] with E matching edge_index. "
            f"Got edge_attr={tuple(edge_attr.shape)}, edge_index={tuple(edge_index.shape)}."
        )
    if edge_attr.shape[1] < 1:
        raise ValueError(
            "edge_attr must have at least one channel containing signed displacement."
        )
    if dx <= 0.0:
        raise ValueError(f"dx must be positive, got {dx}.")
    if advection_scheme not in ADVECTION_SCHEMES:
        raise ValueError(
            "advection_scheme must be one of "
            f"{sorted(ADVECTION_SCHEMES)}, got '{advection_scheme}'."
        )

    src = edge_index[0].long()
    dst = edge_index[1].long()
    signed = edge_attr[:, 0].to(device=x_state.device, dtype=x_state.dtype)

    u_src = x_state[src]
    u_dst = x_state[dst]

    if advection_scheme == ADVECTION_SCHEME_CENTRAL:
        # Central flux: 0.5 * (F(u_i) + F(u_j)) with F(u)=0.5*u^2.
        flux_face = 0.25 * (u_src * u_src + u_dst * u_dst)
    else:
        # Upwind flux for scalar Burgers using Roe speed a=(u_i+u_j)/2.
        # Upwind choice is based on oriented normal speed n_ij * a_face.
        wave_speed = 0.5 * (u_src + u_dst)
        oriented_speed = signed.unsqueeze(1) * wave_speed
        take_src = oriented_speed >= 0.0
        u_upwind = torch.where(take_src, u_src, u_dst)
        flux_face = 0.5 * (u_upwind * u_upwind)

    oriented_flux = signed.unsqueeze(1) * flux_face
    n_nodes, n_feat = x_state.shape
    div_num = torch.zeros((n_nodes, n_feat), device=x_state.device, dtype=x_state.dtype)
    div_num.index_add_(0, src, oriented_flux)
    div = div_num / float(dx)
    return -div


def compute_adv_input_channel_1d(
    *,
    x_state: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    dt: float,
    dx: float,
    physics_cfg: PhysicsInputConfig,
    step_k: int,
) -> torch.Tensor:
    """Compute advection-derived input channel before optional normalization."""
    need_adv_eval = bool(physics_cfg.adv_all_steps) or (int(step_k) == 0)
    if need_adv_eval:
        adv_abs = compute_burgers_advection_term_abs_1d(
            x_state=x_state,
            edge_index=edge_index,
            edge_attr=edge_attr,
            dx=float(dx),
            advection_scheme=physics_cfg.advection_scheme,
        )
    else:
        adv_abs = torch.zeros_like(x_state)

    if physics_cfg.input_weighted:
        adv_abs = float(physics_cfg.adv_weight) * adv_abs

    return _physics_to_model_units_1d(
        adv_abs,
        dt_phys=float(dt),
        input_form=physics_cfg.input_form,
    )


def build_physics_augmented_inputs_1d(
    *,
    x_base: torch.Tensor,
    x_state: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    dt: float,
    dx: float,
    physics_cfg: PhysicsInputConfig,
    step_k: int,
) -> torch.Tensor:
    """Append Burgers advection channel to base node features when enabled."""
    if x_base.ndim != 2:
        raise ValueError(f"x_base must have shape [N,F], got {tuple(x_base.shape)}.")
    if x_state.ndim != 2:
        raise ValueError(f"x_state must have shape [N,1], got {tuple(x_state.shape)}.")
    if x_base.shape[0] != x_state.shape[0]:
        raise ValueError(
            "x_base and x_state must have matching row counts, got "
            f"{x_base.shape[0]} and {x_state.shape[0]}."
        )
    if not physics_cfg.enabled:
        return x_base
    if not physics_cfg.include_adv:
        return x_base

    adv_u = compute_adv_input_channel_1d(
        x_state=x_state,
        edge_index=edge_index,
        edge_attr=edge_attr,
        dt=float(dt),
        dx=float(dx),
        physics_cfg=physics_cfg,
        step_k=step_k,
    )
    if physics_cfg.normalize_adv_to_minus1_1:
        if physics_cfg.adv_min is None or physics_cfg.adv_max is None:
            raise ValueError(
                "physics_inputs.normalize_adv_to_minus1_1 is enabled but adv_min/adv_max are missing."
            )
        adv_u = _apply_minmax_minus1_1(
            x=adv_u,
            x_min=float(physics_cfg.adv_min),
            x_max=float(physics_cfg.adv_max),
            clip=bool(physics_cfg.normalize_adv_clip),
        )
    if physics_cfg.detach_inputs:
        adv_u = adv_u.detach()

    return torch.cat([x_base, adv_u], dim=1)
