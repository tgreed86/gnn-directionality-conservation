"""Physics-operator input features for 1D periodic advection GNN training.

This mirrors the Physics-Aware PARC pattern in simplified 1D form:
1. Compute absolute operator terms on current state:
     r_adv_abs  ~ -div(c * u)
     r_diff_abs ~ nu * Laplacian(u)
2. Convert those terms to model-input units ("rate" or "delta" form).
3. Concatenate operator channels onto the base state feature.

The discretization follows the same DEC-style split used in Physics-Aware:
- Advection uses a face-flux divergence with upwind/central face values.
- Diffusion uses graph-Laplacian accumulation with tau/area scaling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from mls import compute_mls_advection_term_abs_1d

INPUT_FORM_RATE = "rate"
INPUT_FORM_DELTA = "delta"
INPUT_FORMS = {INPUT_FORM_RATE, INPUT_FORM_DELTA}

ADVECTION_SCHEME_UPWIND = "upwind"
ADVECTION_SCHEME_CENTRAL = "central"
ADVECTION_SCHEMES = {ADVECTION_SCHEME_UPWIND, ADVECTION_SCHEME_CENTRAL}

PHYSICS_BACKEND_FV = "fv"
PHYSICS_BACKEND_MLS = "mls"
PHYSICS_BACKENDS = {PHYSICS_BACKEND_FV, PHYSICS_BACKEND_MLS}


@dataclass(frozen=True)
class PhysicsInputConfig:
    """Runtime controls for operator-derived node inputs."""

    enabled: bool = False
    include_adv: bool = True
    include_diff: bool = True
    input_form: str = INPUT_FORM_RATE
    backend: str = PHYSICS_BACKEND_FV
    advection_scheme: str = ADVECTION_SCHEME_UPWIND
    mls_ridge_eps: float = 1e-12
    nu: float = 0.0
    c_override: float | None = None
    adv_all_steps: bool = True
    input_weighted: bool = False
    adv_weight: float = 1.0
    diff_weight: float = 1.0
    detach_inputs: bool = True
    normalize_adv_to_minus1_1: bool = False
    adv_min: float | None = None
    adv_max: float | None = None
    normalize_adv_clip: bool = False


def resolve_physics_input_cfg(cfg: Mapping[str, Any]) -> PhysicsInputConfig:
    """Resolve physics-input settings from config with Physics-Aware-style fallbacks."""
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
    include_diff = bool(
        physics.get("include_diff", loss.get("parc_include_diff", True))
    )

    input_form = str(
        physics.get("input_form", loss.get("parc_input_form", INPUT_FORM_RATE))
    ).strip().lower()
    if input_form not in INPUT_FORMS:
        raise ValueError(
            f"physics_inputs.input_form must be one of {sorted(INPUT_FORMS)}, got '{input_form}'."
        )

    backend_raw = str(
        physics.get(
            "backend",
            physics.get("physics_backend", loss.get("physics_backend", PHYSICS_BACKEND_FV)),
        )
    ).strip().lower()
    if backend_raw in {
        PHYSICS_BACKEND_FV,
        "finite_volume",
        "finite-volume",
        "dec",
        "upwind",
    }:
        backend = PHYSICS_BACKEND_FV
    elif backend_raw in {
        PHYSICS_BACKEND_MLS,
        "moving_least_squares",
        "moving-least-squares",
    }:
        backend = PHYSICS_BACKEND_MLS
    else:
        raise ValueError(
            "physics_inputs.backend must be one of "
            f"{sorted(PHYSICS_BACKENDS)} (aliases: dec, finite_volume), got '{backend_raw}'."
        )

    advection_scheme = str(
        physics.get("advection_scheme", loss.get("advection_scheme", ADVECTION_SCHEME_UPWIND))
    ).strip().lower()
    if advection_scheme not in ADVECTION_SCHEMES:
        raise ValueError(
            "physics_inputs.advection_scheme must be one of "
            f"{sorted(ADVECTION_SCHEMES)}, got '{advection_scheme}'."
        )

    c_override_raw = physics.get("c", None)
    c_override = None if c_override_raw is None else float(c_override_raw)
    normalize_adv_to_minus1_1 = bool(
        physics.get(
            "normalize_adv_to_minus1_1",
            physics.get("normalize_adv", False),
        )
    )
    adv_min_raw = physics.get("adv_min", None)
    adv_max_raw = physics.get("adv_max", None)
    adv_min = None if adv_min_raw is None else float(adv_min_raw)
    adv_max = None if adv_max_raw is None else float(adv_max_raw)
    normalize_adv_clip = bool(physics.get("normalize_adv_clip", False))

    return PhysicsInputConfig(
        enabled=enabled,
        include_adv=include_adv,
        include_diff=include_diff,
        input_form=input_form,
        backend=backend,
        advection_scheme=advection_scheme,
        mls_ridge_eps=float(physics.get("mls_ridge_eps", 1e-12)),
        nu=float(physics.get("nu", loss.get("nu", 0.0))),
        c_override=c_override,
        adv_all_steps=bool(physics.get("adv_all_steps", loss.get("adv_all_steps", True))),
        input_weighted=bool(
            physics.get("input_weighted", loss.get("parc_input_weighted", False))
        ),
        adv_weight=float(physics.get("adv_weight", loss.get("adv_weight", 1.0))),
        diff_weight=float(physics.get("diff_weight", loss.get("diff_weight", 1.0))),
        detach_inputs=bool(
            physics.get("detach_inputs", loss.get("parc_detach_inputs", True))
        ),
        normalize_adv_to_minus1_1=normalize_adv_to_minus1_1,
        adv_min=adv_min,
        adv_max=adv_max,
        normalize_adv_clip=normalize_adv_clip,
    )


def count_physics_input_channels(physics_cfg: PhysicsInputConfig) -> int:
    """Return number of extra per-node channels appended by physics inputs."""
    if not physics_cfg.enabled:
        return 0
    return int(bool(physics_cfg.include_adv)) + int(bool(physics_cfg.include_diff))


def _physics_to_model_units_1d(
    r_abs: torch.Tensor,
    *,
    dt_phys: float,
    input_form: str,
) -> torch.Tensor:
    """Convert absolute operator rate to chosen input form."""
    if input_form == INPUT_FORM_RATE:
        return r_abs
    if input_form == INPUT_FORM_DELTA:
        return float(dt_phys) * r_abs
    raise RuntimeError(f"Unexpected input_form='{input_form}'.")


def _normalize_adv_to_minus1_1(
    adv_u: torch.Tensor,
    *,
    physics_cfg: PhysicsInputConfig,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Optionally normalize advection channel to [-1, 1] using stored stats."""
    if not physics_cfg.normalize_adv_to_minus1_1:
        return adv_u
    if physics_cfg.adv_min is None or physics_cfg.adv_max is None:
        raise ValueError(
            "physics_inputs.normalize_adv_to_minus1_1 is enabled, but adv_min/adv_max "
            "are missing."
        )
    mn = float(physics_cfg.adv_min)
    mx = float(physics_cfg.adv_max)
    denom = mx - mn
    if abs(denom) <= float(eps):
        out = torch.zeros_like(adv_u)
    else:
        out = (2.0 * (adv_u - mn) / denom) - 1.0
    if physics_cfg.normalize_adv_clip:
        out = torch.clamp(out, min=-1.0, max=1.0)
    return out


def fit_adv_input_minmax_1d(
    *,
    x_state: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    dt: float,
    dx: float,
    c_from_data: float,
    physics_cfg: PhysicsInputConfig,
    sample_indices: torch.Tensor | None = None,
) -> tuple[float, float]:
    """Fit scalar min/max stats for the advection input channel.

    Stats are computed in model-input units (after optional `adv_weight` and
    `input_form` conversion), using per-sample states from `x_state`.
    """
    if x_state.ndim != 3:
        raise ValueError(f"x_state must have shape [S,N,F], got {tuple(x_state.shape)}.")
    if x_state.shape[-1] < 1:
        raise ValueError("x_state must have at least one feature channel.")
    if not physics_cfg.enabled or not physics_cfg.include_adv:
        raise ValueError(
            "fit_adv_input_minmax_1d requires physics_inputs.enabled=true and "
            "physics_inputs.include_adv=true."
        )

    if sample_indices is None:
        x_sel = x_state
    else:
        idx = sample_indices.to(device=x_state.device, dtype=torch.long)
        x_sel = x_state[idx]

    if int(x_sel.shape[0]) < 1:
        raise ValueError("No samples selected for advection min/max fitting.")

    c_eff = (
        float(physics_cfg.c_override)
        if physics_cfg.c_override is not None
        else float(c_from_data)
    )

    adv_min = float("inf")
    adv_max = float("-inf")
    saw_finite = False

    for s in range(int(x_sel.shape[0])):
        x_abs = x_sel[s]
        if physics_cfg.backend == PHYSICS_BACKEND_MLS:
            r_adv_abs = compute_mls_advection_term_abs_1d(
                x_abs=x_abs,
                edge_index=edge_index,
                edge_attr=edge_attr,
                c=c_eff,
                dx=float(dx),
                ridge_eps=float(physics_cfg.mls_ridge_eps),
            )
        else:
            r_adv_abs, _ = compute_advdiff_terms_abs_1d(
                x_abs=x_abs,
                edge_index=edge_index,
                edge_attr=edge_attr,
                c=c_eff,
                dx=float(dx),
                nu=float(physics_cfg.nu),
                advection_scheme=physics_cfg.advection_scheme,
                compute_adv=True,
                compute_diff=False,
            )
            assert r_adv_abs is not None

        adv_abs = r_adv_abs
        if physics_cfg.input_weighted:
            adv_abs = float(physics_cfg.adv_weight) * adv_abs
        adv_u = _physics_to_model_units_1d(
            adv_abs,
            dt_phys=float(dt),
            input_form=physics_cfg.input_form,
        )
        finite = adv_u[torch.isfinite(adv_u)]
        if finite.numel() == 0:
            continue
        saw_finite = True
        local_min = float(torch.min(finite).item())
        local_max = float(torch.max(finite).item())
        if local_min < adv_min:
            adv_min = local_min
        if local_max > adv_max:
            adv_max = local_max

    if not saw_finite:
        raise ValueError("Advection min/max fitting found no finite values.")
    return adv_min, adv_max


def compute_advdiff_terms_abs_1d(
    *,
    x_abs: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    c: float,
    dx: float,
    nu: float,
    advection_scheme: str,
    compute_adv: bool = True,
    compute_diff: bool = True,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Compute DEC-style advection and diffusion terms in absolute units.

    Advection:
      div_i = sum_{j in N(i)} [ (u_n * phi_face)_ij * face_len ] / area_i
      r_adv = -div

    Diffusion:
      lap_i = sum_{j in N(i)} [ tau_ij * (phi_j - phi_i) ] / area_i
      r_diff = nu * lap

    For the 1D uniform ring we take:
      - face_len = 1
      - area_i = dx
      - tau_ij = 1 / dx
      - edge normal sign n_ij from edge_attr[:,0] in {-1,+1}
    """
    if x_abs.ndim != 2:
        raise ValueError(f"x_abs must have shape [N,F], got {tuple(x_abs.shape)}.")
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

    n_nodes, n_feat = x_abs.shape
    src = edge_index[0].long()
    dst = edge_index[1].long()
    signed = edge_attr[:, 0].to(device=x_abs.device, dtype=x_abs.dtype)

    if advection_scheme not in ADVECTION_SCHEMES:
        raise ValueError(
            f"advection_scheme must be one of {sorted(ADVECTION_SCHEMES)}, got '{advection_scheme}'."
        )

    r_adv = None
    if compute_adv:
        phi_src = x_abs[src]  # [E,F]
        phi_dst = x_abs[dst]  # [E,F]
        un = float(c) * signed  # [E]

        if advection_scheme == ADVECTION_SCHEME_CENTRAL:
            phi_face = 0.5 * (phi_src + phi_dst)
        else:
            take_src = (un >= 0.0).unsqueeze(1)
            phi_face = torch.where(take_src, phi_src, phi_dst)

        # face_len=1 for this 1D control-volume form.
        flux = un.unsqueeze(1) * phi_face  # [E,F]
        div_num = torch.zeros((n_nodes, n_feat), device=x_abs.device, dtype=x_abs.dtype)
        div_num.index_add_(0, src, flux)
        div = div_num / float(dx)
        r_adv = -div

    r_diff = None
    if compute_diff:
        tau = 1.0 / float(dx)
        diff_term = tau * (x_abs[dst] - x_abs[src])  # [E,F]
        lap_num = torch.zeros((n_nodes, n_feat), device=x_abs.device, dtype=x_abs.dtype)
        lap_num.index_add_(0, src, diff_term)
        lap = lap_num / float(dx)
        r_diff = float(nu) * lap

    return r_adv, r_diff


def build_physics_augmented_inputs_1d(
    *,
    x_state: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    dt: float,
    dx: float,
    c_from_data: float,
    physics_cfg: PhysicsInputConfig,
    step_k: int,
) -> torch.Tensor:
    """Build model input as [state, optional adv term, optional diff term]."""
    if x_state.ndim != 2:
        raise ValueError(
            f"x_state must have shape [N,1] (or [N,F]), got {tuple(x_state.shape)}."
        )
    if not physics_cfg.enabled:
        return x_state

    use_adv_channel = bool(physics_cfg.include_adv)
    use_diff_channel = bool(physics_cfg.include_diff)
    if (not use_adv_channel) and (not use_diff_channel):
        return x_state

    need_adv_eval = use_adv_channel and (
        bool(physics_cfg.adv_all_steps) or (int(step_k) == 0)
    )
    need_diff_eval = use_diff_channel

    c_eff = (
        float(physics_cfg.c_override)
        if physics_cfg.c_override is not None
        else float(c_from_data)
    )

    r_adv_abs: torch.Tensor | None = None
    if need_adv_eval:
        if physics_cfg.backend == PHYSICS_BACKEND_MLS:
            r_adv_abs = compute_mls_advection_term_abs_1d(
                x_abs=x_state,
                edge_index=edge_index,
                edge_attr=edge_attr,
                c=c_eff,
                dx=float(dx),
                ridge_eps=float(physics_cfg.mls_ridge_eps),
            )
        else:
            r_adv_abs, _ = compute_advdiff_terms_abs_1d(
                x_abs=x_state,
                edge_index=edge_index,
                edge_attr=edge_attr,
                c=c_eff,
                dx=float(dx),
                nu=float(physics_cfg.nu),
                advection_scheme=physics_cfg.advection_scheme,
                compute_adv=True,
                compute_diff=False,
            )

    r_diff_abs: torch.Tensor | None = None
    if need_diff_eval:
        _, r_diff_abs = compute_advdiff_terms_abs_1d(
            x_abs=x_state,
            edge_index=edge_index,
            edge_attr=edge_attr,
            c=c_eff,
            dx=float(dx),
            nu=float(physics_cfg.nu),
            advection_scheme=physics_cfg.advection_scheme,
            compute_adv=False,
            compute_diff=True,
        )

    extras: list[torch.Tensor] = []
    zeros = torch.zeros_like(x_state)

    if use_adv_channel:
        adv_computed = r_adv_abs is not None
        adv_abs = r_adv_abs if adv_computed else zeros
        if physics_cfg.input_weighted:
            adv_abs = float(physics_cfg.adv_weight) * adv_abs
        adv_u = _physics_to_model_units_1d(
            adv_abs, dt_phys=float(dt), input_form=physics_cfg.input_form
        )
        if adv_computed:
            adv_u = _normalize_adv_to_minus1_1(adv_u, physics_cfg=physics_cfg)
        extras.append(adv_u.detach() if physics_cfg.detach_inputs else adv_u)

    if use_diff_channel:
        diff_abs = r_diff_abs if r_diff_abs is not None else zeros
        if physics_cfg.input_weighted:
            diff_abs = float(physics_cfg.diff_weight) * diff_abs
        diff_u = _physics_to_model_units_1d(
            diff_abs, dt_phys=float(dt), input_form=physics_cfg.input_form
        )
        extras.append(diff_u.detach() if physics_cfg.detach_inputs else diff_u)

    return torch.cat([x_state] + extras, dim=1)
