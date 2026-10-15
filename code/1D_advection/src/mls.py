"""Moving-least-squares helpers for 1D physics input features.

This module currently exposes a 1D MLS estimate for the advection term:

    r_adv = -c * d(u)/dx

computed nodewise from immediate graph neighbors.
"""

from __future__ import annotations

import torch


def compute_mls_advection_term_abs_1d(
    *,
    x_abs: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    c: float,
    dx: float,
    ridge_eps: float = 1e-12,
) -> torch.Tensor:
    """Estimate 1D advection term with a local weighted least-squares slope.

    For each node i and feature channel, we fit:
      u_j - u_i ~ g_i * (x_j - x_i)
    in weighted least squares over graph neighbors j. The slope estimate is:
      g_i = sum_j w_ij * r_ij * (u_j - u_i) / sum_j w_ij * r_ij^2
    where r_ij = x_j - x_i.

    The returned advection rate is:
      r_adv_i = -c * g_i

    Notes:
    - This uses existing edge connectivity only (no stencil extension).
    - If `edge_attr` includes a physical displacement channel in column 1,
      it is used as r_ij. Otherwise r_ij is reconstructed from signed
      orientation (edge_attr[:,0]) and scalar grid spacing `dx`.
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
    if ridge_eps <= 0.0:
        raise ValueError(f"ridge_eps must be positive, got {ridge_eps}.")

    n_nodes, n_feat = x_abs.shape
    src = edge_index[0].long()
    dst = edge_index[1].long()

    edge_attr_local = edge_attr.to(device=x_abs.device, dtype=x_abs.dtype)
    if edge_attr_local.shape[1] >= 2:
        r_ij = edge_attr_local[:, 1]
    else:
        signed = edge_attr_local[:, 0]
        r_ij = signed * float(dx)

    # Gaussian weights: w_ij = exp(-(r_ij / h)^2) with h=dx.
    # On a uniform ring with immediate neighbors this becomes a constant
    # weight, but keeping the weighting form preserves MLS structure.
    h = max(float(dx), 1e-12)
    w_ij = torch.exp(-torch.square(torch.abs(r_ij) / h))

    du_ij = x_abs[dst] - x_abs[src]  # [E,F]
    wr = (w_ij * r_ij).unsqueeze(1)  # [E,1]
    wr2 = (w_ij * r_ij * r_ij).unsqueeze(1)  # [E,1]

    num = torch.zeros((n_nodes, n_feat), device=x_abs.device, dtype=x_abs.dtype)
    den = torch.zeros((n_nodes, 1), device=x_abs.device, dtype=x_abs.dtype)
    num.index_add_(0, src, wr * du_ij)
    den.index_add_(0, src, wr2)

    den_safe = den.clamp_min(float(ridge_eps))
    slope = num / den_safe

    has_support = den > float(ridge_eps)
    slope = torch.where(has_support, slope, torch.zeros_like(slope))
    return (-float(c)) * slope
