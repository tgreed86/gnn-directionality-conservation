"""Randomized Riemann initial condition for Burgers' equation."""
from __future__ import annotations

import torch
from torch import Tensor


class RiemannIC:
    """u₀(x) = u_L for x < L/2, u_R for x ≥ L/2, with u_L and u_R drawn uniformly."""

    def __init__(
        self,
        u_left_min: float = -1.5,
        u_left_max: float = 1.5,
        u_right_min: float = -1.5,
        u_right_max: float = 1.5,
    ):
        self.u_left_min = u_left_min
        self.u_left_max = u_left_max
        self.u_right_min = u_right_min
        self.u_right_max = u_right_max

    def sample(self, Nx: int, batch_size: int, device: torch.device | str = "cpu") -> Tensor:
        """Returns (B, Nx) initial conditions."""
        u_L = (
            torch.rand(batch_size, device=device, dtype=torch.float64)
            * (self.u_left_max - self.u_left_min)
            + self.u_left_min
        )
        u_R = (
            torch.rand(batch_size, device=device, dtype=torch.float64)
            * (self.u_right_max - self.u_right_min)
            + self.u_right_min
        )
        u0 = torch.empty(batch_size, Nx, device=device, dtype=torch.float64)
        u0[:, : Nx // 2] = u_L.unsqueeze(1)
        u0[:, Nx // 2 :] = u_R.unsqueeze(1)
        return u0
