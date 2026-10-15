"""Training-data generator for 1D periodic linear advection.

This module builds supervised pairs for the PDE

    u_t + c u_x = 0

on a periodic domain x in [0, L). By default labels are generated with exact
spectral advection shift. For discontinuous shock-like ICs, an analytic
characteristic-shift option is also available.
Given a state u^n sampled on a uniform grid, the next state can be computed via

1. U_k = FFT(u^n)
2. U_k <- U_k * exp(-i * 2*pi*f_k * c * dt)
3. u^{n+1} = IFFT(U_k)

where f_k are FFT frequencies in cycles per unit length.
No finite-difference time stepping is used for label generation.

Command-line generation entrypoint:
    python scripts/generate_advection_data.py --help
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

TARGET_TYPE_NEXT = "next"
TARGET_TYPE_RATE = "rate"
TARGET_TYPES = {TARGET_TYPE_NEXT, TARGET_TYPE_RATE}

IC_TYPE_SMOOTH = "smooth"
IC_TYPE_SHOCK = "shock"
IC_TYPE_MIXED = "mixed"
IC_TYPES = {IC_TYPE_SMOOTH, IC_TYPE_SHOCK, IC_TYPE_MIXED}

LABEL_METHOD_SPECTRAL = "spectral"
LABEL_METHOD_ANALYTIC = "analytic"
LABEL_METHODS = {LABEL_METHOD_SPECTRAL, LABEL_METHOD_ANALYTIC}

STATE_REPRESENTATION_POINT = "point"
STATE_REPRESENTATION_CELL_AVERAGE = "cell_average"
STATE_REPRESENTATIONS = {
    STATE_REPRESENTATION_POINT,
    STATE_REPRESENTATION_CELL_AVERAGE,
}


def compute_dx_dt(N: int, L: float, c: float, CFL: float) -> Tuple[float, float]:
    """Compute grid spacing and time step from CFL.

    Args:
        N: Number of grid nodes.
        L: Domain length.
        c: Advection speed.
        CFL: Courant number.

    Returns:
        (dx, dt)

    Raises:
        ValueError: If parameters are invalid or c is zero.
    """
    if N <= 1:
        raise ValueError(f"N must be > 1, got {N}.")
    if L <= 0.0:
        raise ValueError(f"L must be positive, got {L}.")
    if CFL <= 0.0:
        raise ValueError(f"CFL must be positive, got {CFL}.")
    if np.isclose(c, 0.0):
        raise ValueError(
            "Advection speed c must be non-zero to compute dt = CFL * dx / |c|."
        )

    dx = float(L) / float(N)
    dt = float(CFL) * dx / abs(float(c))
    return dx, dt


def build_periodic_ring_graph(
    N: int,
    *,
    dx: float = 1.0,
    dtype: torch.dtype = torch.float32,
    include_physical_displacement: bool = False,
    include_distance: bool = False,
) -> Tuple[torch.LongTensor, torch.Tensor]:
    """Build directed nearest-neighbor periodic ring graph.

    Edges:
      i -> (i + 1) mod N
      i -> (i - 1) mod N

    The first edge attribute channel is signed displacement dx_ij / dx, taking
    values +1 (right neighbor) and -1 (left neighbor).

    Args:
        N: Number of nodes.
        dx: Physical grid spacing.
        dtype: Torch dtype for edge_attr.
        include_physical_displacement: If True, append dx_ij.
        include_distance: If True, append |dx_ij|.

    Returns:
        edge_index: LongTensor [2, 2N]
        edge_attr: Tensor [2N, D], D >= 1
    """
    if N <= 1:
        raise ValueError(f"N must be > 1, got {N}.")
    if dx <= 0.0:
        raise ValueError(f"dx must be positive, got {dx}.")

    sources = np.repeat(np.arange(N, dtype=np.int64), 2)
    targets = np.empty(2 * N, dtype=np.int64)

    # Order per node: right edge, left edge.
    targets[0::2] = (np.arange(N) + 1) % N
    targets[1::2] = (np.arange(N) - 1) % N

    edge_index = torch.tensor(
        np.stack([sources, targets], axis=0), dtype=torch.long
    )

    signed = np.empty((2 * N,), dtype=np.float64)
    signed[0::2] = 1.0
    signed[1::2] = -1.0

    attrs = [signed[:, None]]
    if include_physical_displacement:
        physical = signed * dx
        attrs.append(physical[:, None])
    if include_distance:
        dist = np.abs(signed * dx)
        attrs.append(dist[:, None])

    edge_attr = torch.tensor(np.concatenate(attrs, axis=1), dtype=dtype)
    return edge_index, edge_attr


def exact_advect_spectral_shift(
    u: np.ndarray,
    *,
    c: float,
    dt: float,
    L: float,
) -> np.ndarray:
    """Advance a periodic signal by exact advection shift using FFT.

    Computes u(x, t + dt) = u(x - c*dt, t) exactly in spectral space on the
    discrete periodic grid (up to floating-point precision).

    Args:
        u: Array with final axis as spatial nodes (..., N).
        c: Advection speed.
        dt: Time step.
        L: Domain length.

    Returns:
        Shifted array with same shape as u.
    """
    u_arr = np.asarray(u, dtype=np.float64)
    N = u_arr.shape[-1]
    if N <= 1:
        raise ValueError("u must have final axis size > 1.")
    if L <= 0.0:
        raise ValueError(f"L must be positive, got {L}.")

    dx = L / N
    freqs = np.fft.fftfreq(N, d=dx)  # cycles / unit length
    phase = np.exp(-1j * 2.0 * np.pi * freqs * c * dt)

    u_hat = np.fft.fft(u_arr, axis=-1)
    shifted = np.fft.ifft(u_hat * phase, axis=-1).real
    return shifted


def generate_random_periodic_ic(
    *,
    num_samples: int,
    N: int,
    L: float,
    m_max: int,
    amplitude: float,
    spectral_decay: float,
    rng: np.random.Generator,
    normalize_rms: Optional[float] = None,
    state_representation: str = STATE_REPRESENTATION_POINT,
) -> np.ndarray:
    """Generate smooth random periodic initial conditions.

    Signals are generated as:

        u(x) = sum_{m=1}^{m_max} a_m sin(2*pi*m*x/L) + b_m cos(2*pi*m*x/L)

    with coefficients scaled by amplitude / m^spectral_decay.

    Args:
        num_samples: Number of independent samples.
        N: Number of grid points.
        L: Domain length.
        m_max: Highest Fourier mode included.
        amplitude: Overall coefficient scale.
        spectral_decay: Exponent p in 1 / m^p.
        rng: NumPy random generator.
        normalize_rms: If provided, each sample is scaled to this RMS.
        state_representation: If "point", return point samples at x_i. If
            "cell_average", return exact averages over cells centered at x_i.

    Returns:
        Array of shape [num_samples, N].
    """
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}.")
    if N <= 1:
        raise ValueError(f"N must be > 1, got {N}.")
    if L <= 0.0:
        raise ValueError(f"L must be positive, got {L}.")
    if m_max < 1:
        raise ValueError(f"m_max must be >= 1, got {m_max}.")
    if amplitude < 0.0:
        raise ValueError(f"amplitude must be non-negative, got {amplitude}.")
    if spectral_decay < 0.0:
        raise ValueError(
            f"spectral_decay must be non-negative, got {spectral_decay}."
        )
    if normalize_rms is not None and normalize_rms <= 0.0:
        raise ValueError(
            f"normalize_rms must be positive when provided, got {normalize_rms}."
        )
    if state_representation not in STATE_REPRESENTATIONS:
        raise ValueError(
            "state_representation must be one of "
            f"{sorted(STATE_REPRESENTATIONS)}, got '{state_representation}'."
        )

    x = np.arange(N, dtype=np.float64) * (L / N)
    m = np.arange(1, m_max + 1, dtype=np.float64)

    phase = (2.0 * np.pi / L) * np.outer(m, x)  # [m_max, N]
    sin_basis = np.sin(phase)
    cos_basis = np.cos(phase)
    if state_representation == STATE_REPRESENTATION_CELL_AVERAGE:
        # Average each Fourier mode over [x_i - dx/2, x_i + dx/2).  NumPy sinc
        # is normalized as sin(pi z)/(pi z), so z=m/N gives sin(pi m/N)/(pi m/N).
        avg_factor = np.sinc(m / float(N))
        sin_basis = avg_factor[:, None] * sin_basis
        cos_basis = avg_factor[:, None] * cos_basis

    coeff_scale = amplitude / np.power(m, spectral_decay)

    a = rng.standard_normal(size=(num_samples, m_max)) * coeff_scale[None, :]
    b = rng.standard_normal(size=(num_samples, m_max)) * coeff_scale[None, :]

    u = a @ sin_basis + b @ cos_basis

    if normalize_rms is not None:
        rms = np.sqrt(np.mean(u**2, axis=1, keepdims=True))
        safe_rms = np.maximum(rms, 1e-14)
        u = u * (normalize_rms / safe_rms)

    return u


def evaluate_piecewise_constant_periodic(
    x: np.ndarray,
    *,
    boundaries: np.ndarray,
    values: np.ndarray,
    L: float,
    phase: float = 0.0,
) -> np.ndarray:
    """Evaluate a periodic piecewise-constant function on query points.

    Representation:
      - boundaries: sorted interior breakpoints in (0, L), length J-1
      - values: segment values on intervals
          [0, b0), [b0, b1), ..., [b_{J-2}, L), length J
      - periodicity is enforced by evaluating at (x + phase) mod L

    Args:
        x: Query points.
        boundaries: Sorted interior breakpoints.
        values: Segment values with length len(boundaries)+1.
        L: Domain length.
        phase: Optional global phase shift.

    Returns:
        Evaluated values with same shape as x.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    b = np.asarray(boundaries, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    if L <= 0.0:
        raise ValueError(f"L must be positive, got {L}.")
    if b.ndim != 1:
        raise ValueError("boundaries must be 1D.")
    if v.ndim != 1:
        raise ValueError("values must be 1D.")
    if v.size != b.size + 1:
        raise ValueError(
            "values length must equal len(boundaries)+1, got "
            f"{v.size} vs {b.size + 1}."
        )
    if b.size > 0:
        if np.any(b <= 0.0) or np.any(b >= L):
            raise ValueError("boundaries must lie strictly inside (0, L).")
        if np.any(np.diff(b) <= 0.0):
            raise ValueError("boundaries must be strictly increasing.")

    x_mod = np.mod(x_arr + float(phase), float(L))
    seg = np.searchsorted(b, x_mod, side="right")
    return v[seg]


def _piecewise_constant_periodic_antiderivative(
    x: np.ndarray,
    *,
    boundaries: np.ndarray,
    values: np.ndarray,
    L: float,
) -> np.ndarray:
    """Periodic antiderivative for the piecewise-constant representation."""
    x_arr = np.asarray(x, dtype=np.float64)
    b = np.asarray(boundaries, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    breaks = np.concatenate(
        [
            np.array([0.0], dtype=np.float64),
            b,
            np.array([float(L)], dtype=np.float64),
        ]
    )
    widths = np.diff(breaks)
    cumulative = np.concatenate(
        [
            np.array([0.0], dtype=np.float64),
            np.cumsum(widths * v, dtype=np.float64),
        ]
    )
    domain_integral = float(cumulative[-1])

    periods = np.floor(x_arr / float(L))
    x_mod = x_arr - periods * float(L)
    seg = np.searchsorted(b, x_mod, side="right")
    local = cumulative[seg] + (x_mod - breaks[seg]) * v[seg]
    return periods * domain_integral + local


def cell_average_piecewise_constant_periodic(
    x: np.ndarray,
    *,
    dx: float,
    boundaries: np.ndarray,
    values: np.ndarray,
    L: float,
    phase: float = 0.0,
) -> np.ndarray:
    """Average a periodic piecewise-constant function over centered cells.

    Cell `i` is [x_i - dx/2, x_i + dx/2). The same phase convention as
    `evaluate_piecewise_constant_periodic` is used: the represented function is
    evaluated at `(x + phase) mod L`.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    b = np.asarray(boundaries, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    if L <= 0.0:
        raise ValueError(f"L must be positive, got {L}.")
    if dx <= 0.0:
        raise ValueError(f"dx must be positive, got {dx}.")
    if dx > L:
        raise ValueError(
            "dx must be <= L for periodic cell averages, got "
            f"dx={dx}, L={L}."
        )
    if b.ndim != 1:
        raise ValueError("boundaries must be 1D.")
    if v.ndim != 1:
        raise ValueError("values must be 1D.")
    if v.size != b.size + 1:
        raise ValueError(
            "values length must equal len(boundaries)+1, got "
            f"{v.size} vs {b.size + 1}."
        )
    if b.size > 0:
        if np.any(b <= 0.0) or np.any(b >= L):
            raise ValueError("boundaries must lie strictly inside (0, L).")
        if np.any(np.diff(b) <= 0.0):
            raise ValueError("boundaries must be strictly increasing.")

    half_dx = 0.5 * float(dx)
    lo = x_arr - half_dx + float(phase)
    hi = x_arr + half_dx + float(phase)
    integral = _piecewise_constant_periodic_antiderivative(
        hi,
        boundaries=b,
        values=v,
        L=float(L),
    ) - _piecewise_constant_periodic_antiderivative(
        lo,
        boundaries=b,
        values=v,
        L=float(L),
    )
    return integral / float(dx)


def generate_random_shock_ic(
    *,
    num_samples: int,
    N: int,
    L: float,
    amplitude: float,
    shock_num_jumps_min: int,
    shock_num_jumps_max: int,
    rng: np.random.Generator,
    normalize_rms: Optional[float] = None,
    state_representation: str = STATE_REPRESENTATION_POINT,
) -> Tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray, float]]]:
    """Generate discontinuous periodic initial conditions (shock-like ICs).

    Each sample is piecewise constant with random jump locations and random
    segment values. For periodic signals, `shock_num_jumps_*` is interpreted as
    the number of jumps around the full ring, so it must be >= 2.

    Returns:
        u0: Array [num_samples, N].
        params: List of per-sample tuples `(boundaries, values, phase)` used by
          the analytic characteristic shift generator.
    """
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}.")
    if N <= 1:
        raise ValueError(f"N must be > 1, got {N}.")
    if L <= 0.0:
        raise ValueError(f"L must be positive, got {L}.")
    if amplitude < 0.0:
        raise ValueError(f"amplitude must be non-negative, got {amplitude}.")
    if shock_num_jumps_min < 2:
        raise ValueError(
            "shock_num_jumps_min must be >= 2 for periodic piecewise-constant ICs."
        )
    if shock_num_jumps_max < shock_num_jumps_min:
        raise ValueError(
            "shock_num_jumps_max must be >= shock_num_jumps_min, got "
            f"{shock_num_jumps_max} < {shock_num_jumps_min}."
        )
    if shock_num_jumps_max > N:
        raise ValueError(
            "shock_num_jumps_max must be <= N to avoid degenerate tiny segments, got "
            f"{shock_num_jumps_max} for N={N}."
        )
    if normalize_rms is not None and normalize_rms <= 0.0:
        raise ValueError(
            f"normalize_rms must be positive when provided, got {normalize_rms}."
        )
    if state_representation not in STATE_REPRESENTATIONS:
        raise ValueError(
            "state_representation must be one of "
            f"{sorted(STATE_REPRESENTATIONS)}, got '{state_representation}'."
        )

    dx = float(L) / float(N)
    x = np.arange(N, dtype=np.float64) * dx
    u = np.empty((num_samples, N), dtype=np.float64)
    params: list[tuple[np.ndarray, np.ndarray, float]] = []
    tiny = max(1e-12, abs(float(amplitude)) * 1e-10)

    for s in range(num_samples):
        num_jumps = int(rng.integers(shock_num_jumps_min, shock_num_jumps_max + 1))
        n_interior = num_jumps - 1
        if n_interior > 0:
            boundaries = np.sort(rng.uniform(0.0, L, size=n_interior))
        else:
            boundaries = np.zeros((0,), dtype=np.float64)

        values = rng.standard_normal(size=num_jumps) * float(amplitude)
        # Ensure nontrivial jumps, including the periodic wrap jump at x=0.
        for j in range(1, values.size):
            if abs(values[j] - values[j - 1]) < tiny:
                values[j] += tiny
        if abs(values[0] - values[-1]) < tiny:
            values[-1] += tiny

        phase = float(rng.uniform(0.0, L))
        if state_representation == STATE_REPRESENTATION_CELL_AVERAGE:
            u[s] = cell_average_piecewise_constant_periodic(
                x,
                dx=dx,
                boundaries=boundaries,
                values=values,
                L=L,
                phase=phase,
            )
        else:
            u[s] = evaluate_piecewise_constant_periodic(
                x, boundaries=boundaries, values=values, L=L, phase=phase
            )
        params.append((boundaries, values, phase))

    if normalize_rms is not None:
        rms = np.sqrt(np.mean(u**2, axis=1, keepdims=True))
        safe_rms = np.maximum(rms, 1e-14)
        scales = float(normalize_rms) / safe_rms
        u = u * scales
        for s in range(num_samples):
            b, v, ph = params[s]
            params[s] = (b, v * float(scales[s, 0]), ph)

    return u, params


@dataclass
class AdvectionMetadata:
    """Metadata persisted with generated dataset arrays."""

    N: int
    L: float
    c: float
    CFL: float
    dt: float
    dx: float
    m_max: int
    amplitude: float
    spectral_decay: float
    target_type: str
    seed: int
    num_samples: int
    window: int
    num_pairs: int
    normalize_rms: float
    ic_type: str = IC_TYPE_SMOOTH
    label_method: str = LABEL_METHOD_SPECTRAL
    shock_fraction: float = 0.0
    shock_num_jumps_min: int = 2
    shock_num_jumps_max: int = 2
    state_representation: str = STATE_REPRESENTATION_POINT

    def to_dict(self) -> Dict[str, Any]:
        return {
            "N": self.N,
            "L": self.L,
            "c": self.c,
            "CFL": self.CFL,
            "dt": self.dt,
            "dx": self.dx,
            "m_max": self.m_max,
            "amplitude": self.amplitude,
            "spectral_decay": self.spectral_decay,
            "target_type": self.target_type,
            "seed": self.seed,
            "num_samples": self.num_samples,
            "window": self.window,
            "num_pairs": self.num_pairs,
            "normalize_rms": self.normalize_rms,
            "ic_type": self.ic_type,
            "label_method": self.label_method,
            "shock_fraction": self.shock_fraction,
            "shock_num_jumps_min": self.shock_num_jumps_min,
            "shock_num_jumps_max": self.shock_num_jumps_max,
            "state_representation": self.state_representation,
        }


class _BaseAdvectionDataset(Dataset):
    """Base dataset wrapper returning dictionary samples on a fixed ring graph."""

    def __init__(
        self,
        *,
        x: np.ndarray | torch.Tensor,
        y: np.ndarray | torch.Tensor,
        u_next: np.ndarray | torch.Tensor,
        pos: np.ndarray | torch.Tensor,
        edge_index: np.ndarray | torch.Tensor,
        edge_attr: np.ndarray | torch.Tensor,
        pair_trajectory_id: np.ndarray | torch.Tensor | None = None,
        pair_time_index: np.ndarray | torch.Tensor | None = None,
        rollout_states: np.ndarray | torch.Tensor | None = None,
        metadata: Mapping[str, Any],
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.x = torch.as_tensor(x, dtype=dtype)
        self.y = torch.as_tensor(y, dtype=dtype)
        self.u_next = torch.as_tensor(u_next, dtype=dtype)
        self.pos = torch.as_tensor(pos, dtype=dtype)
        self.edge_index = torch.as_tensor(edge_index, dtype=torch.long)
        self.edge_attr = torch.as_tensor(edge_attr, dtype=dtype)
        self.pair_trajectory_id = (
            None
            if pair_trajectory_id is None
            else torch.as_tensor(pair_trajectory_id, dtype=torch.long)
        )
        self.pair_time_index = (
            None
            if pair_time_index is None
            else torch.as_tensor(pair_time_index, dtype=torch.long)
        )
        self.rollout_states = (
            None if rollout_states is None else torch.as_tensor(rollout_states, dtype=dtype)
        )
        self.metadata = dict(metadata)
        self.dtype = dtype

        if self.x.ndim != 3 or self.x.shape[-1] != 1:
            raise ValueError(f"x must have shape [S, N, 1], got {tuple(self.x.shape)}.")
        if self.y.shape != self.x.shape:
            raise ValueError(
                f"y must have same shape as x, got y={tuple(self.y.shape)} "
                f"and x={tuple(self.x.shape)}."
            )
        if self.u_next.shape != self.x.shape:
            raise ValueError(
                f"u_next must have same shape as x, got u_next={tuple(self.u_next.shape)} "
                f"and x={tuple(self.x.shape)}."
            )
        if self.pos.shape != (self.x.shape[1], 1):
            raise ValueError(
                f"pos must have shape [N, 1], got {tuple(self.pos.shape)} for N={self.x.shape[1]}."
            )
        if self.edge_index.shape != (2, 2 * self.x.shape[1]):
            raise ValueError(
                f"edge_index must have shape [2, 2N], got {tuple(self.edge_index.shape)}."
            )
        if self.edge_attr.shape[0] != 2 * self.x.shape[1]:
            raise ValueError(
                f"edge_attr first dimension must be 2N, got {self.edge_attr.shape[0]}."
            )
        if self.pair_trajectory_id is not None:
            if self.pair_trajectory_id.shape != (self.x.shape[0],):
                raise ValueError(
                    "pair_trajectory_id must have shape [S], got "
                    f"{tuple(self.pair_trajectory_id.shape)} for S={self.x.shape[0]}."
                )
        if self.pair_time_index is not None:
            if self.pair_time_index.shape != (self.x.shape[0],):
                raise ValueError(
                    "pair_time_index must have shape [S], got "
                    f"{tuple(self.pair_time_index.shape)} for S={self.x.shape[0]}."
                )
        if self.rollout_states is not None:
            if self.rollout_states.ndim != 4 or self.rollout_states.shape[-1] != 1:
                raise ValueError(
                    "rollout_states must have shape [num_samples, window+1, N, 1], got "
                    f"{tuple(self.rollout_states.shape)}."
                )
            if self.rollout_states.shape[2] != self.x.shape[1]:
                raise ValueError(
                    "rollout_states N dimension must match pair N dimension, got "
                    f"{self.rollout_states.shape[2]} vs {self.x.shape[1]}."
                )

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample_id = (
            int(self.pair_trajectory_id[idx].item())
            if self.pair_trajectory_id is not None
            else int(idx)
        )
        time_index = (
            int(self.pair_time_index[idx].item())
            if self.pair_time_index is not None
            else 0
        )
        return {
            "x": self.x[idx],
            "y": self.y[idx],
            "u_next": self.u_next[idx],
            "pos": self.pos,
            "edge_index": self.edge_index,
            "edge_attr": self.edge_attr,
            "dt": float(self.metadata["dt"]),
            "dx": float(self.metadata["dx"]),
            "c": float(self.metadata["c"]),
            "L": float(self.metadata["L"]),
            "sample_id": sample_id,
            "pair_index": int(idx),
            "time_index": time_index,
        }


class PeriodicAdvectionDataset(_BaseAdvectionDataset):
    """Generate supervised data for 1D periodic linear advection.

    A rollout trajectory of length `window+1` is generated per initial condition,
    then flattened into adjacent single-step pairs for training.

    IC options:
      - ic_type='smooth': random Fourier sums (current baseline)
      - ic_type='shock': discontinuous piecewise-constant profiles
      - ic_type='mixed': random mixture of smooth + shock trajectories

    State representation options:
      - state_representation='point': u_i is a point sample at x_i
      - state_representation='cell_average': u_i is the exact average over
        [x_i - dx/2, x_i + dx/2)

    Label options:
      - label_method='spectral': FFT phase shift (default)
      - label_method='analytic': characteristic-shift labels for shock ICs
        and spectral shift fallback for smooth ICs.

    Dataset item outputs dictionaries containing:
      x:       [N, 1]  (u^n)
      y:       [N, 1]  (u^{n+1} or rate, depending on target_type)
      u_next:  [N, 1]  (always exact next state)
      pos:     [N, 1]
      edge_index: [2, 2N]
      edge_attr:  [2N, D]
      dt, dx, c, L, sample_id, pair_index, time_index
    """

    def __init__(
        self,
        *,
        N: int = 128,
        L: float = 1.0,
        c: float = 1.0,
        CFL: float = 0.25,
        num_samples: int = 1000,
        m_max: int = 16,
        amplitude: float = 1.0,
        spectral_decay: float = 1.0,
        ic_type: str = IC_TYPE_SMOOTH,
        label_method: str = LABEL_METHOD_SPECTRAL,
        shock_fraction: float = 0.5,
        shock_num_jumps_min: int = 2,
        shock_num_jumps_max: int = 4,
        target_type: str = TARGET_TYPE_RATE,
        state_representation: str = STATE_REPRESENTATION_POINT,
        window: int = 1,
        seed: int = 1234,
        normalize_rms: Optional[float] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if target_type not in TARGET_TYPES:
            raise ValueError(
                f"target_type must be one of {sorted(TARGET_TYPES)}, got '{target_type}'."
            )
        if ic_type not in IC_TYPES:
            raise ValueError(
                f"ic_type must be one of {sorted(IC_TYPES)}, got '{ic_type}'."
            )
        if label_method not in LABEL_METHODS:
            raise ValueError(
                "label_method must be one of "
                f"{sorted(LABEL_METHODS)}, got '{label_method}'."
            )
        if state_representation not in STATE_REPRESENTATIONS:
            raise ValueError(
                "state_representation must be one of "
                f"{sorted(STATE_REPRESENTATIONS)}, got '{state_representation}'."
            )
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}.")
        if not (0.0 <= shock_fraction <= 1.0):
            raise ValueError(
                f"shock_fraction must be in [0,1], got {shock_fraction}."
            )

        dx, dt = compute_dx_dt(N=N, L=L, c=c, CFL=CFL)
        x_grid = np.arange(N, dtype=np.float64) * dx

        rng = np.random.default_rng(seed)
        u0 = np.empty((num_samples, N), dtype=np.float64)
        shock_params: list[tuple[np.ndarray, np.ndarray, float] | None] = [
            None
        ] * num_samples

        if ic_type == IC_TYPE_SMOOTH:
            u0 = generate_random_periodic_ic(
                num_samples=num_samples,
                N=N,
                L=L,
                m_max=m_max,
                amplitude=amplitude,
                spectral_decay=spectral_decay,
                rng=rng,
                normalize_rms=normalize_rms,
                state_representation=state_representation,
            )
        elif ic_type == IC_TYPE_SHOCK:
            u0, params = generate_random_shock_ic(
                num_samples=num_samples,
                N=N,
                L=L,
                amplitude=amplitude,
                shock_num_jumps_min=shock_num_jumps_min,
                shock_num_jumps_max=shock_num_jumps_max,
                rng=rng,
                normalize_rms=normalize_rms,
                state_representation=state_representation,
            )
            shock_params = [p for p in params]
        else:
            # Mixed bank of ICs for broader training distribution.
            shock_mask = rng.random(num_samples) < float(shock_fraction)
            shock_idx = np.nonzero(shock_mask)[0]
            smooth_idx = np.nonzero(~shock_mask)[0]

            if smooth_idx.size > 0:
                u0_smooth = generate_random_periodic_ic(
                    num_samples=int(smooth_idx.size),
                    N=N,
                    L=L,
                    m_max=m_max,
                    amplitude=amplitude,
                    spectral_decay=spectral_decay,
                    rng=rng,
                    normalize_rms=normalize_rms,
                    state_representation=state_representation,
                )
                u0[smooth_idx] = u0_smooth

            if shock_idx.size > 0:
                u0_shock, params = generate_random_shock_ic(
                    num_samples=int(shock_idx.size),
                    N=N,
                    L=L,
                    amplitude=amplitude,
                    shock_num_jumps_min=shock_num_jumps_min,
                    shock_num_jumps_max=shock_num_jumps_max,
                    rng=rng,
                    normalize_rms=normalize_rms,
                    state_representation=state_representation,
                )
                u0[shock_idx] = u0_shock
                for local_i, global_i in enumerate(shock_idx.tolist()):
                    shock_params[global_i] = params[local_i]

        # Build rollout trajectories per IC: [num_samples, window+1, N].
        rollout = np.empty((num_samples, window + 1, N), dtype=np.float64)
        if label_method == LABEL_METHOD_SPECTRAL:
            rollout[:, 0, :] = u0
            for t in range(window):
                rollout[:, t + 1, :] = exact_advect_spectral_shift(
                    rollout[:, t, :], c=c, dt=dt, L=L
                )
        else:
            for s in range(num_samples):
                if shock_params[s] is None:
                    rollout[s, 0, :] = u0[s]
                    for t in range(window):
                        rollout[s, t + 1, :] = exact_advect_spectral_shift(
                            rollout[s, t, :], c=c, dt=dt, L=L
                        )
                else:
                    boundaries, values, phase = shock_params[s]
                    for t in range(window + 1):
                        x_q = x_grid - float(c) * float(dt) * float(t)
                        if state_representation == STATE_REPRESENTATION_CELL_AVERAGE:
                            rollout[s, t, :] = (
                                cell_average_piecewise_constant_periodic(
                                    x_q,
                                    dx=dx,
                                    boundaries=boundaries,
                                    values=values,
                                    L=L,
                                    phase=phase,
                                )
                            )
                        else:
                            rollout[s, t, :] = evaluate_piecewise_constant_periodic(
                                x_q,
                                boundaries=boundaries,
                                values=values,
                                L=L,
                                phase=phase,
                            )

        # Flatten adjacent pairs for single-step supervision.
        u_now_pairs = rollout[:, :-1, :]   # [num_samples, window, N]
        u_next_pairs = rollout[:, 1:, :]   # [num_samples, window, N]
        num_pairs = int(num_samples * window)
        u_now = u_now_pairs.reshape(num_pairs, N)
        u_next = u_next_pairs.reshape(num_pairs, N)
        pair_trajectory_id = np.repeat(np.arange(num_samples, dtype=np.int64), window)
        pair_time_index = np.tile(np.arange(window, dtype=np.int64), num_samples)

        if target_type == TARGET_TYPE_NEXT:
            y = u_next
        else:
            y = (u_next - u_now) / dt

        x = u_now[..., None]
        y = y[..., None]
        u_next = u_next[..., None]
        rollout_states = rollout[..., None]

        pos = (np.arange(N, dtype=np.float64) * dx)[:, None]
        edge_index, edge_attr = build_periodic_ring_graph(N=N, dx=dx, dtype=dtype)
        shock_fraction_meta = (
            1.0
            if ic_type == IC_TYPE_SHOCK
            else (float(shock_fraction) if ic_type == IC_TYPE_MIXED else 0.0)
        )

        metadata = AdvectionMetadata(
            N=N,
            L=float(L),
            c=float(c),
            CFL=float(CFL),
            dt=float(dt),
            dx=float(dx),
            m_max=int(m_max),
            amplitude=float(amplitude),
            spectral_decay=float(spectral_decay),
            target_type=target_type,
            seed=int(seed),
            num_samples=int(num_samples),
            window=int(window),
            num_pairs=int(num_pairs),
            normalize_rms=float(np.nan if normalize_rms is None else normalize_rms),
            ic_type=str(ic_type),
            label_method=str(label_method),
            shock_fraction=shock_fraction_meta,
            shock_num_jumps_min=int(shock_num_jumps_min),
            shock_num_jumps_max=int(shock_num_jumps_max),
            state_representation=str(state_representation),
        ).to_dict()

        super().__init__(
            x=x,
            y=y,
            u_next=u_next,
            pos=pos,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pair_trajectory_id=pair_trajectory_id,
            pair_time_index=pair_time_index,
            rollout_states=rollout_states,
            metadata=metadata,
            dtype=dtype,
        )


class LoadedAdvectionDataset(_BaseAdvectionDataset):
    """Dataset loaded from saved `.npz` arrays and metadata."""


def save_advection_dataset(dataset: _BaseAdvectionDataset, out_path: str | Path) -> None:
    """Save arrays and metadata to `.npz`.

    Stored keys include:
      x, y, u_next, pos, edge_index, edge_attr
      pair_trajectory_id, pair_time_index, rollout_states (when available)
      N, L, c, CFL, dt, dx, m_max, amplitude, spectral_decay,
      target_type, seed, num_samples, window, num_pairs, normalize_rms,
      ic_type, label_method, shock_fraction, shock_num_jumps_min,
      shock_num_jumps_max, state_representation
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    meta = dataset.metadata
    np.savez_compressed(
        out_path,
        x=dataset.x.cpu().numpy(),
        y=dataset.y.cpu().numpy(),
        u_next=dataset.u_next.cpu().numpy(),
        pos=dataset.pos.cpu().numpy(),
        edge_index=dataset.edge_index.cpu().numpy(),
        edge_attr=dataset.edge_attr.cpu().numpy(),
        pair_trajectory_id=(
            np.array([], dtype=np.int64)
            if dataset.pair_trajectory_id is None
            else dataset.pair_trajectory_id.cpu().numpy()
        ),
        pair_time_index=(
            np.array([], dtype=np.int64)
            if dataset.pair_time_index is None
            else dataset.pair_time_index.cpu().numpy()
        ),
        rollout_states=(
            np.array([], dtype=np.float32)
            if dataset.rollout_states is None
            else dataset.rollout_states.cpu().numpy()
        ),
        N=np.int64(meta["N"]),
        L=np.float64(meta["L"]),
        c=np.float64(meta["c"]),
        CFL=np.float64(meta["CFL"]),
        dt=np.float64(meta["dt"]),
        dx=np.float64(meta["dx"]),
        m_max=np.int64(meta["m_max"]),
        amplitude=np.float64(meta["amplitude"]),
        spectral_decay=np.float64(meta["spectral_decay"]),
        target_type=np.array(str(meta["target_type"])),
        seed=np.int64(meta["seed"]),
        num_samples=np.int64(meta["num_samples"]),
        window=np.int64(meta.get("window", 1)),
        num_pairs=np.int64(meta.get("num_pairs", int(dataset.x.shape[0]))),
        normalize_rms=np.float64(meta.get("normalize_rms", np.nan)),
        ic_type=np.array(str(meta.get("ic_type", IC_TYPE_SMOOTH))),
        label_method=np.array(str(meta.get("label_method", LABEL_METHOD_SPECTRAL))),
        shock_fraction=np.float64(meta.get("shock_fraction", 0.0)),
        shock_num_jumps_min=np.int64(meta.get("shock_num_jumps_min", 2)),
        shock_num_jumps_max=np.int64(meta.get("shock_num_jumps_max", 2)),
        state_representation=np.array(
            str(meta.get("state_representation", STATE_REPRESENTATION_POINT))
        ),
    )


def load_advection_dataset(
    path: str | Path, *, dtype: torch.dtype = torch.float32
) -> LoadedAdvectionDataset:
    """Load dataset arrays + metadata from `.npz`."""
    path = Path(path)
    with np.load(path) as data:
        window = int(data["window"].item()) if "window" in data.files else 1
        num_pairs = (
            int(data["num_pairs"].item())
            if "num_pairs" in data.files
            else int(data["x"].shape[0])
        )
        metadata = {
            "N": int(data["N"].item()),
            "L": float(data["L"].item()),
            "c": float(data["c"].item()),
            "CFL": float(data["CFL"].item()),
            "dt": float(data["dt"].item()),
            "dx": float(data["dx"].item()),
            "m_max": int(data["m_max"].item()),
            "amplitude": float(data["amplitude"].item()),
            "spectral_decay": float(data["spectral_decay"].item()),
            "target_type": str(data["target_type"].item()),
            "seed": int(data["seed"].item()),
            "num_samples": int(data["num_samples"].item()),
            "window": int(window),
            "num_pairs": int(num_pairs),
            "normalize_rms": float(data["normalize_rms"].item()),
            "ic_type": (
                str(data["ic_type"].item())
                if "ic_type" in data.files
                else IC_TYPE_SMOOTH
            ),
            "label_method": (
                str(data["label_method"].item())
                if "label_method" in data.files
                else LABEL_METHOD_SPECTRAL
            ),
            "shock_fraction": (
                float(data["shock_fraction"].item())
                if "shock_fraction" in data.files
                else 0.0
            ),
            "shock_num_jumps_min": (
                int(data["shock_num_jumps_min"].item())
                if "shock_num_jumps_min" in data.files
                else 2
            ),
            "shock_num_jumps_max": (
                int(data["shock_num_jumps_max"].item())
                if "shock_num_jumps_max" in data.files
                else 2
            ),
            "state_representation": (
                str(data["state_representation"].item())
                if "state_representation" in data.files
                else STATE_REPRESENTATION_POINT
            ),
        }

        pair_trajectory_id = None
        if "pair_trajectory_id" in data.files and data["pair_trajectory_id"].size > 0:
            pair_trajectory_id = data["pair_trajectory_id"]

        pair_time_index = None
        if "pair_time_index" in data.files and data["pair_time_index"].size > 0:
            pair_time_index = data["pair_time_index"]

        rollout_states = None
        if "rollout_states" in data.files and data["rollout_states"].size > 0:
            rollout_states = data["rollout_states"]

        dataset = LoadedAdvectionDataset(
            x=data["x"],
            y=data["y"],
            u_next=data["u_next"],
            pos=data["pos"],
            edge_index=data["edge_index"],
            edge_attr=data["edge_attr"],
            pair_trajectory_id=pair_trajectory_id,
            pair_time_index=pair_time_index,
            rollout_states=rollout_states,
            metadata=metadata,
            dtype=dtype,
        )
    return dataset
