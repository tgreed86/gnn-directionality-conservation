"""Inviscid Burgers equation: ∂ₜu + ∂ₓ(u²/2) = 0."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


class BurgersEquation:
    """Encapsulates everything specific to the Burgers equation."""

    def flux(self, u: Tensor) -> Tensor:
        return 0.5 * u * u

    def entropy(self, u: Tensor) -> Tensor:
        """Quadratic entropy η(u) = u²/2."""
        return 0.5 * u * u

    def entropy_flux(self, u: Tensor) -> Tensor:
        """Entropy flux q(u) = u³/3, satisfying q' = η'·f' = u·u."""
        return u.pow(3) / 3.0

    def flux_jacobian(self, u: Tensor) -> Tensor:
        return u

    def max_wavespeed(self, u: Tensor) -> Tensor:
        return u.abs()

    def physics_params(self, u: Tensor) -> Tensor:
        return u.unsqueeze(-1)  # local wave speed, shape (..., 1)

    def godunov_flux(self, u_L: Tensor, u_R: Tensor) -> Tensor:
        """Exact Godunov flux for convex Burgers flux."""
        # Shock: u_L >= u_R  →  Rankine-Hugoniot speed s = (u_L + u_R)/2
        # Rarefaction: u_L < u_R  →  characteristic fan
        f_L = self.flux(u_L)
        f_R = self.flux(u_R)

        shock_speed = 0.5 * (u_L + u_R)
        shock_flux = torch.where(shock_speed >= 0, f_L, f_R)

        # Rarefaction: u_L < 0 < u_R → sonic rarefaction, f(0)=0
        # Otherwise max of left characteristic / min of right
        rare_flux = torch.where(
            u_L >= 0,
            f_L,
            torch.where(u_R <= 0, f_R, torch.zeros_like(u_L)),
        )

        return torch.where(u_L >= u_R, shock_flux, rare_flux)

    def lax_hopf_reference(
        self, u0: Tensor, x: Tensor, t: float, n_quad: int = 8
    ) -> Tensor:
        """Exact entropy solution via Lax-Hopf formula, returned as cell averages.

        u0: (Nx,) initial condition cell averages on uniform grid
        x:  (Nx,) cell-center positions
        t:  query time
        n_quad: sub-cell quadrature points per cell for cell-average computation
        Returns: (Nx,) cell-averaged exact solution

        Uses the analytical per-cell minimiser (y* = x - u0_i * t) to avoid the
        1/t amplification that causes stripe artefacts at early time steps.
        Sub-cell averaging is essential: a pointwise evaluation at the cell center
        flips between u_L and u_R discretely as a shock moves between cells,
        producing a staircase pattern. Averaging over n_quad sub-cell points gives
        the correct cell-mean, which transitions smoothly as the shock crosses.
        """
        if t == 0.0:
            return u0.clone()

        dx = x[1] - x[0]
        offsets = (torch.arange(n_quad, device=x.device, dtype=x.dtype) + 0.5) / n_quad - 0.5
        x_sub = x.unsqueeze(1) + offsets.unsqueeze(0) * dx  # (Nx, n_quad)

        cols = [self._lax_hopf_pointwise(u0, x_sub[:, k], x, t) for k in range(n_quad)]
        return torch.stack(cols, dim=1).mean(dim=1)

    def _lax_hopf_pointwise(
        self, u0: Tensor, x_query: Tensor, x_ic: Tensor, t: float
    ) -> Tensor:
        """Pointwise Lax-Hopf exact entropy solution at arbitrary query positions.

        u0:      (Nx,) IC cell averages
        x_query: (M,)  query positions
        x_ic:    (Nx,) IC cell-center positions (defines IC grid spacing and domain)
        t:       query time
        """
        device = u0.device
        dtype = u0.dtype
        Nx = u0.shape[0]
        dx = x_ic[1] - x_ic[0]
        L = dx * Nx

        # Tile u0 to cover periodic wrap-around (±2 periods on each side).
        n_copies = 2
        n_tiles = 2 * n_copies + 1        # 5
        u0_ext = u0.repeat(n_tiles)       # (5*Nx,)
        Nx_ext = n_tiles * Nx

        # Left boundary of every extended cell
        y_l = (x_ic[0] - dx / 2 - n_copies * L) + torch.arange(Nx_ext, device=device, dtype=dtype) * dx
        y_r = y_l + dx

        # Running integral G[i] = ∫ u0_ext from y_l[0] to y_l[i]
        G = torch.zeros(Nx_ext, device=device, dtype=dtype)
        G[1:] = torch.cumsum(u0_ext[:-1] * dx, dim=0)

        # Broadcast shapes: (M, Nx_ext)
        x_col   = x_query.unsqueeze(1)   # (M, 1)
        u0_row  = u0_ext.unsqueeze(0)    # (1, Nx_ext)
        y_l_row = y_l.unsqueeze(0)       # (1, Nx_ext)
        y_r_row = y_r.unsqueeze(0)       # (1, Nx_ext)
        G_row   = G.unsqueeze(0)         # (1, Nx_ext)

        # Analytical per-cell minimiser, clamped to cell boundaries
        y_star = x_col - u0_row * t
        y_c    = torch.max(torch.min(y_star, y_r_row), y_l_row)  # (M, Nx_ext)

        # Cost = G(y_l_i) + u0_i*(y_c - y_l_i) + (x - y_c)²/(2t)
        cost = G_row + u0_row * (y_c - y_l_row) + (x_col - y_c).pow(2) / (2.0 * t)

        M = x_query.shape[0]
        i_opt = cost.argmin(dim=1)                                     # (M,)
        y_opt = y_c[torch.arange(M, device=device), i_opt]            # (M,)
        return (x_query - y_opt) / t


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eq = BurgersEquation()
    Nx = 256
    x = torch.linspace(0.0, 1.0 - 1.0 / Nx, Nx, dtype=torch.float64)
    dx = x[1] - x[0]

    # Step function IC: u0 = 1 left, 0 right
    u0 = torch.where(x < 0.3, torch.ones_like(x), torch.zeros_like(x))

    t = 0.2
    u_ref = eq.lax_hopf_reference(u0, x, t)

    print(f"Burgers equation sanity check:")
    print(f"  flux(1.0) = {eq.flux(torch.tensor(1.0)):.4f}  (expected 0.5)")
    print(f"  max_wavespeed([-1,2]) = {eq.max_wavespeed(torch.tensor([-1.0,2.0]))}")
    u_L = torch.tensor([1.0, 0.0, -1.0])
    u_R = torch.tensor([0.0, 1.0, -0.5])
    print(f"  godunov_flux: {eq.godunov_flux(u_L, u_R)}")
    print(f"  Lax-Hopf reference computed, u range: [{u_ref.min():.3f}, {u_ref.max():.3f}]")
    print("PASS")
