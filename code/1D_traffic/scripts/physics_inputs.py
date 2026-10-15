"""Physics-derived input features for LWR traffic GNNs.

The traffic generators evolve cell density with a conservative finite-volume
update.  This module exposes the corresponding graph-local advection rate as
an optional node feature so training and autoregressive rollout can construct
the same model input from the current (possibly predicted) state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor


INPUT_FORM_RATE = "rate"
INPUT_FORM_DELTA = "delta"
INPUT_FORMS = {INPUT_FORM_RATE, INPUT_FORM_DELTA}

ADVECTION_SCHEME_GODUNOV = "godunov"
ADVECTION_SCHEME_RUSANOV = "rusanov"
ADVECTION_SCHEME_CENTRAL = "central"
ADVECTION_SCHEMES = {
    ADVECTION_SCHEME_GODUNOV,
    ADVECTION_SCHEME_RUSANOV,
    ADVECTION_SCHEME_CENTRAL,
}


@dataclass(frozen=True)
class PhysicsInputConfig:
    """Runtime controls for the optional LWR advection input channel."""

    enabled: bool = False
    include_adv: bool = True
    input_form: str = INPUT_FORM_RATE
    advection_scheme: str = ADVECTION_SCHEME_GODUNOV
    adv_all_steps: bool = True
    input_weighted: bool = False
    adv_weight: float = 1.0
    normalize_adv_to_minus1_1: bool = False
    adv_min: float | None = None
    adv_max: float | None = None
    normalize_adv_clip: bool = False
    detach_inputs: bool = True


@dataclass(frozen=True)
class LWRAdvectionMetadata:
    """Prevalidated, positively oriented physical traffic interfaces."""

    edge_index: Tensor
    junction_attr: Tensor


def resolve_physics_input_cfg(cfg: Mapping[str, Any]) -> PhysicsInputConfig:
    """Resolve ``physics_inputs`` using the conventions of sibling projects."""
    raw = cfg.get("physics_inputs", {})
    physics = raw if isinstance(raw, Mapping) else {}

    input_form = str(physics.get("input_form", INPUT_FORM_RATE)).strip().lower()
    if input_form not in INPUT_FORMS:
        raise ValueError(
            "physics_inputs.input_form must be one of "
            f"{sorted(INPUT_FORMS)}, got {input_form!r}."
        )

    scheme = str(
        physics.get("advection_scheme", ADVECTION_SCHEME_GODUNOV)
    ).strip().lower()
    # The upwind option used by 1D_advection and Burgers corresponds to the
    # Godunov flux for the nonlinear, concave LWR flux function.
    if scheme == "upwind":
        scheme = ADVECTION_SCHEME_GODUNOV
    if scheme not in ADVECTION_SCHEMES:
        raise ValueError(
            "physics_inputs.advection_scheme must be one of "
            f"{sorted(ADVECTION_SCHEMES)} (or 'upwind'), got {scheme!r}."
        )

    adv_min_raw = physics.get("adv_min")
    adv_max_raw = physics.get("adv_max")
    adv_min = None if adv_min_raw is None else float(adv_min_raw)
    adv_max = None if adv_max_raw is None else float(adv_max_raw)
    if (adv_min is None) != (adv_max is None):
        raise ValueError(
            "physics_inputs.adv_min and adv_max must either both be supplied "
            "or both be null."
        )
    if adv_min is not None and adv_max is not None and adv_max <= adv_min:
        raise ValueError("physics_inputs.adv_max must be greater than adv_min.")

    return PhysicsInputConfig(
        enabled=bool(physics.get("enabled", False)),
        include_adv=bool(physics.get("include_adv", True)),
        input_form=input_form,
        advection_scheme=scheme,
        adv_all_steps=bool(physics.get("adv_all_steps", True)),
        input_weighted=bool(physics.get("input_weighted", False)),
        adv_weight=float(physics.get("adv_weight", 1.0)),
        normalize_adv_to_minus1_1=bool(
            physics.get(
                "normalize_adv_to_minus1_1",
                physics.get("normalize_adv", False),
            )
        ),
        adv_min=adv_min,
        adv_max=adv_max,
        normalize_adv_clip=bool(physics.get("normalize_adv_clip", False)),
        detach_inputs=bool(physics.get("detach_inputs", True)),
    )


def count_physics_input_channels(physics_cfg: PhysicsInputConfig) -> int:
    """Return the number of physics-derived per-node channels."""
    return int(physics_cfg.enabled and physics_cfg.include_adv)


def _as_node_column(
    value: Tensor,
    *,
    num_nodes: int,
    name: str,
    reference: Tensor,
) -> Tensor:
    """Broadcast a scalar or per-node tensor over leading state batches."""
    if reference.ndim < 2 or reference.shape[-1] != 1:
        raise ValueError(
            f"reference must end in [node, 1], got {tuple(reference.shape)}."
        )
    target_shape = (*reference.shape[:-2], num_nodes, 1)
    result = value.to(device=reference.device, dtype=reference.dtype)
    if result.ndim == 0:
        return result.reshape((1,) * len(target_shape)).expand(target_shape)
    if result.ndim == 1:
        if result.numel() == 1:
            return result.reshape((1,) * len(target_shape)).expand(
                target_shape
            )
        if result.numel() == num_nodes:
            result = result.reshape(num_nodes, 1)
    if tuple(result.shape) == target_shape:
        return result
    if result.shape == (1, 1):
        return result.reshape((1,) * len(target_shape)).expand(target_shape)
    if result.shape == (num_nodes, 1):
        view_shape = (1,) * (len(target_shape) - 2) + (num_nodes, 1)
        return result.reshape(view_shape).expand(target_shape)
    if result.ndim == len(target_shape):
        broadcastable = all(
            actual in (1, expected)
            for actual, expected in zip(result.shape, target_shape)
        )
        if broadcastable:
            return result.expand(target_shape)
    raise ValueError(
        f"{name} must be scalar or broadcastable to {target_shape}, got "
        f"{tuple(result.shape)}."
    )


def _lwr_flux(rho: Tensor, *, v_max: Tensor, rho_max: Tensor) -> Tensor:
    return v_max * rho * (1.0 - rho / rho_max)


def _lwr_demand(rho: Tensor, *, v_max: Tensor, rho_max: Tensor) -> Tensor:
    critical = 0.5 * rho_max
    capacity = 0.25 * v_max * rho_max
    return torch.where(
        rho <= critical,
        _lwr_flux(rho, v_max=v_max, rho_max=rho_max),
        capacity,
    )


def _lwr_supply(rho: Tensor, *, v_max: Tensor, rho_max: Tensor) -> Tensor:
    critical = 0.5 * rho_max
    capacity = 0.25 * v_max * rho_max
    return torch.where(
        rho <= critical,
        capacity,
        _lwr_flux(rho, v_max=v_max, rho_max=rho_max),
    )


def _physical_interface_rows(edge_index: Tensor, edge_attr: Tensor) -> Tensor:
    """Select the positive-flow row from each adjacent reverse-edge pair."""
    num_edges = int(edge_index.shape[1])
    if num_edges % 2 != 0:
        raise ValueError("LWR advection inputs require an even edge count.")
    if num_edges == 0:
        return torch.empty(0, dtype=torch.long, device=edge_index.device)

    paired_sources = edge_index[0].reshape(-1, 2)
    paired_targets = edge_index[1].reshape(-1, 2)
    if not bool(
        torch.all(
            (paired_sources[:, 0] == paired_targets[:, 1])
            & (paired_targets[:, 0] == paired_sources[:, 1])
        )
    ):
        raise ValueError(
            "LWR advection inputs require adjacent forward/reverse edge pairs."
        )

    direction = edge_attr[:, 0].reshape(-1, 2)
    positive = direction > 0.0
    negative = direction < 0.0
    if not bool(torch.all((positive.sum(dim=1) == 1) & (negative.sum(dim=1) == 1))):
        raise ValueError(
            "Every LWR interface must have one positive and one negative "
            "traffic-direction edge."
        )
    pair_rows = torch.arange(
        direction.shape[0], device=edge_index.device, dtype=torch.long
    )
    return 2 * pair_rows + positive.to(torch.long).argmax(dim=1)


def build_lwr_advection_metadata(
    edge_index: Tensor,
    edge_attr: Tensor,
) -> LWRAdvectionMetadata:
    """Validate a graph once and retain one positive-flow edge per interface."""
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(
            f"edge_index must have shape [2, edge], got {tuple(edge_index.shape)}."
        )
    if edge_attr.ndim != 2 or edge_attr.shape[0] != edge_index.shape[1]:
        raise ValueError(
            "edge_attr must have shape [edge, channel] with the same edge "
            "count as edge_index."
        )
    if edge_attr.shape[1] < 1:
        raise ValueError("edge_attr channel 0 must contain traffic direction.")
    physical_rows = _physical_interface_rows(edge_index, edge_attr)
    if edge_attr.shape[1] >= 10:
        junction_attr = edge_attr[physical_rows, 7:10].contiguous()
    else:
        junction_attr = edge_attr.new_empty((physical_rows.numel(), 0))
    return LWRAdvectionMetadata(
        edge_index=edge_index[:, physical_rows].contiguous(),
        junction_attr=junction_attr,
    )


def _compute_lwr_advection_from_metadata(
    *,
    x_state: Tensor,
    metadata: LWRAdvectionMetadata,
    lengths: Tensor,
    speeds: Tensor,
    jam_density: Tensor,
    scheme: str,
) -> Tensor:
    """Evaluate LWR advection using prevalidated physical interfaces."""
    sources = metadata.edge_index[0].to(device=x_state.device)
    targets = metadata.edge_index[1].to(device=x_state.device)
    rho_left = x_state[..., sources, :]
    rho_right = x_state[..., targets, :]
    speed_left = speeds[..., sources, :]
    speed_right = speeds[..., targets, :]
    rho_max_left = jam_density[..., sources, :]
    rho_max_right = jam_density[..., targets, :]

    demand = _lwr_demand(rho_left, v_max=speed_left, rho_max=rho_max_left)
    supply = _lwr_supply(rho_right, v_max=speed_right, rho_max=rho_max_right)
    if scheme == ADVECTION_SCHEME_GODUNOV:
        interface_flux = torch.minimum(demand, supply)
    else:
        flux_left = _lwr_flux(rho_left, v_max=speed_left, rho_max=rho_max_left)
        flux_right = _lwr_flux(
            rho_right, v_max=speed_right, rho_max=rho_max_right
        )
        interface_flux = 0.5 * (flux_left + flux_right)
        if scheme == ADVECTION_SCHEME_RUSANOV:
            wave_left = torch.abs(
                speed_left * (1.0 - 2.0 * rho_left / rho_max_left)
            )
            wave_right = torch.abs(
                speed_right * (1.0 - 2.0 * rho_right / rho_max_right)
            )
            wave_speed = torch.maximum(wave_left, wave_right)
            interface_flux = interface_flux - 0.5 * wave_speed * (
                rho_right - rho_left
            )

    junction_attr = metadata.junction_attr.to(device=x_state.device)
    if junction_attr.shape[1] == 3 and junction_attr.shape[0] > 0:
        is_junction = junction_attr[:, 0:1] > 0.5
        turning = junction_attr[:, 1:2].to(
            device=x_state.device, dtype=x_state.dtype
        )
        priority = junction_attr[:, 2:3].to(
            device=x_state.device, dtype=x_state.dtype
        )
        has_turning = is_junction & (turning > 0.0) & (turning < 1.0)
        has_priority = is_junction & (priority > 0.0) & (priority < 1.0)
        if scheme == ADVECTION_SCHEME_GODUNOV:
            turning_limited = torch.minimum(turning * demand, supply)
            priority_limited = torch.minimum(demand, priority * supply)
            interface_flux = torch.where(
                has_turning, turning_limited, interface_flux
            )
            interface_flux = torch.where(
                has_priority,
                torch.minimum(interface_flux, priority_limited),
                interface_flux,
            )
        else:
            scale = torch.ones_like(interface_flux)
            scale = torch.where(has_turning, turning, scale)
            scale = torch.where(has_priority, scale * priority, scale)
            interface_flux = interface_flux * scale

    amount_rate = torch.zeros_like(x_state)
    node_dimension = x_state.ndim - 2
    amount_rate.index_add_(node_dimension, sources, -interface_flux)
    amount_rate.index_add_(node_dimension, targets, interface_flux)
    return amount_rate / lengths


def compute_lwr_advection_term_abs_1d(
    *,
    x_state: Tensor,
    edge_index: Tensor,
    edge_attr: Tensor,
    cell_length: Tensor,
    v_max: Tensor,
    rho_max: Tensor,
    advection_scheme: str = ADVECTION_SCHEME_GODUNOV,
    metadata: LWRAdvectionMetadata | None = None,
    validate_inputs: bool = True,
) -> Tensor:
    """Compute the conservative LWR density rate on a bidirected graph.

    One physical flux is evaluated per adjacent reverse-edge pair.  The result
    is accumulated as amount gained/lost at each endpoint and divided by that
    node's cell length.  Junction turning fractions and merge priorities are
    used as local demand/supply caps when present in the standard ten-channel
    traffic edge attributes.
    """
    if x_state.ndim < 2 or x_state.shape[-1] != 1:
        raise ValueError(
            "x_state must end in [node, 1], got "
            f"{tuple(x_state.shape)}."
        )
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(
            f"edge_index must have shape [2, edge], got {tuple(edge_index.shape)}."
        )
    if edge_attr.ndim != 2 or edge_attr.shape[0] != edge_index.shape[1]:
        raise ValueError(
            "edge_attr must have shape [edge, channel] with the same edge "
            "count as edge_index."
        )
    if edge_attr.shape[1] < 1:
        raise ValueError("edge_attr channel 0 must contain traffic direction.")
    scheme = str(advection_scheme).strip().lower()
    if scheme == "upwind":
        scheme = ADVECTION_SCHEME_GODUNOV
    if scheme not in ADVECTION_SCHEMES:
        raise ValueError(
            f"advection_scheme must be one of {sorted(ADVECTION_SCHEMES)}."
        )

    num_nodes = int(x_state.shape[-2])
    lengths = _as_node_column(
        cell_length,
        num_nodes=num_nodes,
        name="cell_length",
        reference=x_state,
    )
    speeds = _as_node_column(
        v_max,
        num_nodes=num_nodes,
        name="v_max",
        reference=x_state,
    )
    jam_density = _as_node_column(
        rho_max,
        num_nodes=num_nodes,
        name="rho_max",
        reference=x_state,
    )
    if metadata is None:
        metadata = build_lwr_advection_metadata(edge_index, edge_attr)
    elif (
        metadata.edge_index.ndim != 2
        or metadata.edge_index.shape[0] != 2
        or metadata.junction_attr.ndim != 2
        or metadata.junction_attr.shape[0] != metadata.edge_index.shape[1]
        or metadata.junction_attr.shape[1] not in (0, 3)
    ):
        raise ValueError("Invalid precomputed LWR advection metadata shapes.")
    if validate_inputs:
        if metadata.edge_index.numel() and (
            int(torch.min(metadata.edge_index)) < 0
            or int(torch.max(metadata.edge_index)) >= num_nodes
        ):
            raise ValueError("edge_index contains an invalid node index.")
        if bool(torch.any(lengths <= 0.0)):
            raise ValueError("cell_length values must be positive.")
        if bool(torch.any(speeds <= 0.0)):
            raise ValueError("v_max values must be positive.")
        if bool(torch.any(jam_density <= 0.0)):
            raise ValueError("rho_max values must be positive.")

    return _compute_lwr_advection_from_metadata(
        x_state=x_state,
        metadata=metadata,
        lengths=lengths,
        speeds=speeds,
        jam_density=jam_density,
        scheme=scheme,
    )


def compute_adv_input_channel_1d(
    *,
    x_state: Tensor,
    edge_index: Tensor,
    edge_attr: Tensor,
    cell_length: Tensor,
    dt_node: Tensor,
    v_max: Tensor,
    rho_max: Tensor,
    physics_cfg: PhysicsInputConfig,
    step_k: int,
    metadata: LWRAdvectionMetadata | None = None,
    precomputed_advection: Tensor | None = None,
    validate_inputs: bool = True,
) -> Tensor:
    """Compute the configured advection channel before normalization."""
    should_compute = physics_cfg.adv_all_steps or int(step_k) == 0
    if should_compute and precomputed_advection is not None:
        advection = precomputed_advection.to(
            device=x_state.device, dtype=x_state.dtype
        )
        if advection.shape != x_state.shape:
            raise ValueError(
                "precomputed_advection must match x_state, got "
                f"{tuple(advection.shape)} and {tuple(x_state.shape)}."
            )
        return advection
    if should_compute:
        advection = compute_lwr_advection_term_abs_1d(
            x_state=x_state,
            edge_index=edge_index,
            edge_attr=edge_attr,
            cell_length=cell_length,
            v_max=v_max,
            rho_max=rho_max,
            advection_scheme=physics_cfg.advection_scheme,
            metadata=metadata,
            validate_inputs=validate_inputs,
        )
    else:
        advection = torch.zeros_like(x_state)
    if physics_cfg.input_weighted:
        advection = float(physics_cfg.adv_weight) * advection
    if physics_cfg.input_form == INPUT_FORM_DELTA:
        dt = _as_node_column(
            dt_node,
            num_nodes=int(x_state.shape[-2]),
            name="dt_node",
            reference=x_state,
        )
        advection = dt * advection
    return advection


def _normalize_adv_to_minus1_1(
    advection: Tensor,
    *,
    physics_cfg: PhysicsInputConfig,
    eps: float = 1.0e-12,
) -> Tensor:
    if physics_cfg.adv_min is None or physics_cfg.adv_max is None:
        raise ValueError(
            "Advection normalization is enabled but adv_min/adv_max are missing."
        )
    denominator = physics_cfg.adv_max - physics_cfg.adv_min
    if abs(denominator) <= eps:
        normalized = torch.zeros_like(advection)
    else:
        normalized = 2.0 * (advection - physics_cfg.adv_min) / denominator - 1.0
    if physics_cfg.normalize_adv_clip:
        normalized = torch.clamp(normalized, min=-1.0, max=1.0)
    return normalized


def build_physics_augmented_inputs_1d(
    *,
    x_base: Tensor,
    x_state: Tensor,
    edge_index: Tensor,
    edge_attr: Tensor,
    cell_length: Tensor,
    dt_node: Tensor,
    v_max: Tensor,
    rho_max: Tensor,
    physics_cfg: PhysicsInputConfig,
    step_k: int,
    metadata: LWRAdvectionMetadata | None = None,
    precomputed_advection: Tensor | None = None,
    validate_inputs: bool = True,
) -> Tensor:
    """Append the configured LWR advection channel to base node features."""
    if x_base.ndim != 2:
        raise ValueError(
            f"x_base must have shape [node, feature], got {tuple(x_base.shape)}."
        )
    if x_state.ndim != 2 or x_state.shape[1] != 1:
        raise ValueError(
            f"x_state must have shape [node, 1], got {tuple(x_state.shape)}."
        )
    if x_base.shape[0] != x_state.shape[0]:
        raise ValueError("x_base and x_state must have matching node counts.")
    if not physics_cfg.enabled or not physics_cfg.include_adv:
        return x_base

    computed = physics_cfg.adv_all_steps or int(step_k) == 0
    advection = compute_adv_input_channel_1d(
        x_state=x_state,
        edge_index=edge_index,
        edge_attr=edge_attr,
        cell_length=cell_length,
        dt_node=dt_node,
        v_max=v_max,
        rho_max=rho_max,
        physics_cfg=physics_cfg,
        step_k=step_k,
        metadata=metadata,
        precomputed_advection=precomputed_advection,
        validate_inputs=validate_inputs,
    )
    # Preserve a neutral zero feature on gated-off steps, including when the
    # fitted normalization interval is asymmetric.
    if physics_cfg.normalize_adv_to_minus1_1 and computed:
        advection = _normalize_adv_to_minus1_1(advection, physics_cfg=physics_cfg)
    if physics_cfg.detach_inputs:
        advection = advection.detach()
    return torch.cat((x_base, advection), dim=-1)
