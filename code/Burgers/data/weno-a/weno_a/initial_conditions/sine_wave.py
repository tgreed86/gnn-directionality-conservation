"""Randomized superposition-of-sines initial condition generator."""
from __future__ import annotations

import torch
from torch import Tensor


class SineWaveIC:
    """
    Generates u₀(x) = Σ Aₖ · sin(2π nₖ x + φₖ) on [0, domain_length].

    Integer frequencies nₖ guarantee periodicity on the domain.
    Produces smooth ICs that develop shocks at later times — a harder test
    for smooth-region accuracy than top-hat ICs.
    """

    def __init__(
        self,
        n_min: int = 1,
        n_max: int = 3,
        freq_max: int = 4,
        a_min: float = -1.5,
        a_max: float = 1.5,
        domain_length: float = 1.0,
    ):
        self.n_min = n_min
        self.n_max = n_max
        self.freq_max = freq_max
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
            n_components = torch.randint(self.n_min, self.n_max + 1, (1,), generator=gen).item()
            freqs = torch.randint(1, self.freq_max + 1, (n_components,), device=work_device, generator=gen)
            amps = (
                torch.rand(n_components, device=work_device, dtype=torch.float64, generator=gen)
                * (self.a_max - self.a_min)
                + self.a_min
            )
            phases = torch.rand(n_components, device=work_device, dtype=torch.float64, generator=gen) * 2.0 * torch.pi

            for k in range(n_components):
                u0[b] += amps[k] * torch.sin(
                    2.0 * torch.pi * freqs[k].double() * x / self.L + phases[k]
                )

        return u0.to(device)


if __name__ == "__main__":
    ic = SineWaveIC()
    u0 = ic.sample(Nx=256, batch_size=4, device="cpu")
    print(f"SineWaveIC sample shape: {u0.shape}")
    print(f"Value range: [{u0.min():.3f}, {u0.max():.3f}]")
    print("PASS")
