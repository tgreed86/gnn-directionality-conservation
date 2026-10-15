"""Randomized top-hat (rectangular pulse) initial condition generator."""
from __future__ import annotations

import torch
from torch import Tensor


class TopHatIC:
    """
    Generates u₀(x) = Σ aₖ · 𝟙[xₖ - wₖ/2, xₖ + wₖ/2](x) on [0, domain_length].

    Produces solutions with multiple interacting shocks and rarefactions.
    """

    def __init__(
        self,
        n_min: int = 2,
        n_max: int = 6,
        w_min: float = 0.05,
        w_max: float = 0.25,
        a_min: float = -1.5,
        a_max: float = 1.5,
        domain_length: float = 1.0,
    ):
        self.n_min = n_min
        self.n_max = n_max
        self.w_min = w_min
        self.w_max = w_max
        self.a_min = a_min
        self.a_max = a_max
        self.L = domain_length

    def sample(
        self,
        Nx: int,
        batch_size: int,
        device: torch.device | str = "cpu",
        seed: int | None = None,
    ) -> Tensor:
        """Returns (B, Nx) initial conditions on a uniform grid.

        When seed is given, uses an isolated CPU generator so the result is
        identical regardless of global torch random state or target device.
        """
        if seed is not None:
            gen = torch.Generator()
            gen.manual_seed(seed)
            work_device: torch.device | str = "cpu"
        else:
            gen = None
            work_device = device

        x = torch.linspace(0.0, self.L - self.L / Nx, Nx, device=work_device, dtype=torch.float64)
        u0 = torch.zeros(batch_size, Nx, device=work_device, dtype=torch.float64)

        for b in range(batch_size):
            n_hats = torch.randint(self.n_min, self.n_max + 1, (1,), generator=gen).item()
            centers = torch.rand(n_hats, device=work_device, dtype=torch.float64, generator=gen) * self.L
            widths = (
                torch.rand(n_hats, device=work_device, dtype=torch.float64, generator=gen)
                * (self.w_max - self.w_min)
                + self.w_min
            )
            amps = (
                torch.rand(n_hats, device=work_device, dtype=torch.float64, generator=gen)
                * (self.a_max - self.a_min)
                + self.a_min
            )

            # Vectorized: (n_hats, Nx) indicator
            x_row = x.unsqueeze(0)          # (1, Nx)
            c_col = centers.unsqueeze(1)    # (n_hats, 1)
            w_col = widths.unsqueeze(1)     # (n_hats, 1)
            a_col = amps.unsqueeze(1)       # (n_hats, 1)

            inside = (x_row >= c_col - 0.5 * w_col) & (x_row < c_col + 0.5 * w_col)
            u0[b] = (a_col * inside.to(torch.float64)).sum(0)

        return u0.to(device)


if __name__ == "__main__":
    ic = TopHatIC()
    u0 = ic.sample(Nx=256, batch_size=4, device="cpu")
    print(f"TopHatIC sample shape: {u0.shape}")
    print(f"Value range: [{u0.min():.3f}, {u0.max():.3f}]")
    print("PASS")
