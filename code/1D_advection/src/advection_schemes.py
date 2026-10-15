"""Finite-volume rollout schemes for 1D periodic linear advection."""

from __future__ import annotations

import numpy as np

BASELINE_SCHEME_GODUNOV = "godunov"
BASELINE_SCHEME_MUSCL = "muscl"
BASELINE_SCHEME_WENO3 = "weno3"
BASELINE_SCHEMES = (
    BASELINE_SCHEME_GODUNOV,
    BASELINE_SCHEME_MUSCL,
    BASELINE_SCHEME_WENO3,
)

BASELINE_SCHEME_DEFAULT_LABELS = {
    BASELINE_SCHEME_GODUNOV: "Godunov",
    BASELINE_SCHEME_MUSCL: "MUSCL",
    BASELINE_SCHEME_WENO3: "WENO-3",
}

_BASELINE_SCHEME_ALIASES = {
    "upwind": BASELINE_SCHEME_GODUNOV,
    "godunov-upwind": BASELINE_SCHEME_GODUNOV,
    "godunov_upwind": BASELINE_SCHEME_GODUNOV,
    "muscl2": BASELINE_SCHEME_MUSCL,
    "muscl-2": BASELINE_SCHEME_MUSCL,
    "muscl_2": BASELINE_SCHEME_MUSCL,
    "weno-3": BASELINE_SCHEME_WENO3,
    "weno_3": BASELINE_SCHEME_WENO3,
}


def normalize_baseline_scheme(scheme: str) -> str:
    """Return the canonical baseline scheme name."""
    s = str(scheme).strip().lower()
    if s in BASELINE_SCHEMES:
        return s
    if s in _BASELINE_SCHEME_ALIASES:
        return _BASELINE_SCHEME_ALIASES[s]
    allowed = sorted(set(BASELINE_SCHEMES).union(_BASELINE_SCHEME_ALIASES))
    raise ValueError(f"baseline scheme must be one of {allowed}, got '{scheme}'.")


def _as_1d_float_array(u: np.ndarray) -> np.ndarray:
    arr = np.asarray(u, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"u must be a 1D array, got shape {tuple(arr.shape)}.")
    if arr.size < 1:
        raise ValueError("u must contain at least one cell.")
    return arr


def _validate_step_inputs(u: np.ndarray, *, dx: float) -> np.ndarray:
    arr = _as_1d_float_array(u)
    if float(dx) <= 0.0:
        raise ValueError(f"dx must be positive, got {dx}.")
    return arr


def godunov_upwind_step(u: np.ndarray, *, c: float, dt: float, dx: float) -> np.ndarray:
    """One first-order Godunov/upwind step on a periodic grid."""
    arr = _validate_step_inputs(u, dx=dx)
    nu = float(c) * float(dt) / float(dx)
    if float(c) >= 0.0:
        return arr - nu * (arr - np.roll(arr, 1))
    return arr - nu * (np.roll(arr, -1) - arr)


def _minmod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    same_sign = (a * b) > 0.0
    limited = np.minimum(np.abs(a), np.abs(b))
    return np.where(same_sign, np.sign(a) * limited, 0.0)


def _fv_rhs_from_face_flux(face_flux: np.ndarray, *, dx: float) -> np.ndarray:
    return -(face_flux - np.roll(face_flux, 1)) / float(dx)


def _muscl_rhs(u: np.ndarray, *, c: float, dx: float) -> np.ndarray:
    du_left = u - np.roll(u, 1)
    du_right = np.roll(u, -1) - u
    slope = _minmod(du_left, du_right)
    if float(c) >= 0.0:
        face_state = u + 0.5 * slope
    else:
        face_state = np.roll(u, -1) - 0.5 * np.roll(slope, -1)
    return _fv_rhs_from_face_flux(float(c) * face_state, dx=dx)


def muscl_step(u: np.ndarray, *, c: float, dt: float, dx: float) -> np.ndarray:
    """One MUSCL-minmod step with SSP RK2 time integration."""
    arr = _validate_step_inputs(u, dx=dx)
    if arr.size < 3:
        raise ValueError("MUSCL requires at least 3 periodic cells.")
    k1 = _muscl_rhs(arr, c=float(c), dx=float(dx))
    u1 = arr + float(dt) * k1
    k2 = _muscl_rhs(u1, c=float(c), dx=float(dx))
    return 0.5 * arr + 0.5 * (u1 + float(dt) * k2)


def _weno3_weights(
    beta0: np.ndarray,
    beta1: np.ndarray,
    *,
    d0: float = 1.0 / 3.0,
    d1: float = 2.0 / 3.0,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    alpha0 = float(d0) / (float(eps) + beta0) ** 2
    alpha1 = float(d1) / (float(eps) + beta1) ** 2
    denom = alpha0 + alpha1
    return alpha0 / denom, alpha1 / denom


def _weno3_rhs(u: np.ndarray, *, c: float, dx: float) -> np.ndarray:
    if float(c) >= 0.0:
        u_im1 = np.roll(u, 1)
        u_ip1 = np.roll(u, -1)
        q0 = 1.5 * u - 0.5 * u_im1
        q1 = 0.5 * u + 0.5 * u_ip1
        beta0 = (u - u_im1) ** 2
        beta1 = (u_ip1 - u) ** 2
        w0, w1 = _weno3_weights(beta0, beta1)
        face_state = w0 * q0 + w1 * q1
    else:
        u_ip1 = np.roll(u, -1)
        u_ip2 = np.roll(u, -2)
        q0 = 1.5 * u_ip1 - 0.5 * u_ip2
        q1 = 0.5 * u + 0.5 * u_ip1
        beta0 = (u_ip2 - u_ip1) ** 2
        beta1 = (u_ip1 - u) ** 2
        w0, w1 = _weno3_weights(beta0, beta1)
        face_state = w0 * q0 + w1 * q1
    return _fv_rhs_from_face_flux(float(c) * face_state, dx=dx)


def weno3_step(u: np.ndarray, *, c: float, dt: float, dx: float) -> np.ndarray:
    """One WENO-3 step with SSP RK3 time integration."""
    arr = _validate_step_inputs(u, dx=dx)
    if arr.size < 3:
        raise ValueError("WENO-3 requires at least 3 periodic cells.")
    k1 = _weno3_rhs(arr, c=float(c), dx=float(dx))
    u1 = arr + float(dt) * k1
    k2 = _weno3_rhs(u1, c=float(c), dx=float(dx))
    u2 = 0.75 * arr + 0.25 * (u1 + float(dt) * k2)
    k3 = _weno3_rhs(u2, c=float(c), dx=float(dx))
    return (1.0 / 3.0) * arr + (2.0 / 3.0) * (u2 + float(dt) * k3)


def finite_volume_step(
    u: np.ndarray,
    *,
    c: float,
    dt: float,
    dx: float,
    scheme: str,
) -> np.ndarray:
    """Advance one step with the requested periodic finite-volume scheme."""
    resolved = normalize_baseline_scheme(scheme)
    if resolved == BASELINE_SCHEME_GODUNOV:
        return godunov_upwind_step(u, c=c, dt=dt, dx=dx)
    if resolved == BASELINE_SCHEME_MUSCL:
        return muscl_step(u, c=c, dt=dt, dx=dx)
    if resolved == BASELINE_SCHEME_WENO3:
        return weno3_step(u, c=c, dt=dt, dx=dx)
    raise RuntimeError(f"Unexpected baseline scheme '{resolved}'.")


def rollout_finite_volume(
    *,
    x0: np.ndarray,
    c: float,
    dt: float,
    dx: float,
    horizon: int,
    scheme: str,
) -> np.ndarray:
    """Roll out a periodic finite-volume advection baseline."""
    if int(horizon) < 0:
        raise ValueError(f"horizon must be >= 0, got {horizon}.")
    resolved = normalize_baseline_scheme(scheme)
    x_init = _as_1d_float_array(x0)
    traj = np.empty((int(horizon) + 1, x_init.shape[0]), dtype=np.float64)
    traj[0] = x_init
    for k in range(int(horizon)):
        traj[k + 1] = finite_volume_step(
            traj[k],
            c=float(c),
            dt=float(dt),
            dx=float(dx),
            scheme=resolved,
        )
    return traj
