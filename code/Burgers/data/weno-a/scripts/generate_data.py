"""Generate offline Lax-Hopf reference trajectories for supervised training.

For each sampled IC, computes the exact entropy solution at n_steps time steps
using the Lax-Hopf formula and stores the resulting trajectories to disk.
The saved file is consumed by LaxHopfDataset during supervised training.

Usage:
    python scripts/generate_data.py \\
        --config configs/burgers_default.yaml \\
        --output data/lax_hopf_train.pt \\
        --n-samples 5000 \\
        --n-steps 20 \\
        --dt 0.0015 \\
        --device cuda

    python scripts/generate_data.py \\
        --config configs/burgers_default.yaml \\
        --output data/lax_hopf_train.pt \\
        --n-samples 5000 \\
        --n-steps 20 \\
        --cfl 0.4 \\
        --device cuda

    python scripts/generate_data.py \\
        --config configs/burgers_default.yaml \\
        --output data/lax_hopf_train.pt \\
        --n-samples 5000 \\
        --n-steps 20 \\
        --dt 0.0015 \\
        --cfl 0.4 \\
        --device cuda

    python scripts/generate_data.py \\
        --config configs/burgers_default.yaml \\
        --output data/lax_hopf_train.pt \\
        --n-samples 5000 \\
        --n-steps 20 \\
        --dt 0.0015 \\
        --cfl-range 0.1 0.4 \\
        --device cuda
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from weno_a.equations.burgers import BurgersEquation
from weno_a.initial_conditions.riemann import RiemannIC
from weno_a.initial_conditions.top_hat import TopHatIC

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def generate_dataset(
    n_samples: int,
    n_steps: int,
    Nx: int,
    dx: float,
    dt: float | None,
    cfl: float | None,
    cfl_range: tuple[float, float] | None,
    ic_sampler: TopHatIC,
    equation: BurgersEquation,
    n_quad: int = 8,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Generate n_samples Lax-Hopf reference trajectories at fixed dx.

    Returns a dict with:
        trajectories : (N, n_steps+1, Nx) float32 — cell-averaged exact solutions
        dts          : (N,) float32 — fixed dt, or per-sample dt from CFL-only mode
        cfls         : (N,) float32 — actual CFL value for each stored sample
        max_speeds   : (N,) float32 — max wave speed for each stored sample
        dx           : float — uniform cell width
        meta         : dict — provenance information
    """
    if cfl_range is not None and dt is None:
        raise ValueError("cfl_range mode requires a fixed dt.")
    n_cfl_modes = int(cfl is not None) + int(cfl_range is not None)
    if n_cfl_modes > 1:
        raise ValueError(
            "Provide at most one of cfl or cfl_range to generate_dataset."
        )
    if dt is None and cfl is None:
        raise ValueError("Provide dt, cfl, or cfl_range with fixed dt.")

    x = (torch.arange(Nx, device=device, dtype=torch.float64) + 0.5) * dx

    all_trajectories: list[torch.Tensor] = []
    all_dts: list[float] = []
    all_cfls: list[float] = []
    all_max_speeds: list[float] = []
    skipped = 0

    pbar = tqdm(range(n_samples), desc="Generating Lax-Hopf trajectories")
    for _ in pbar:
        u0 = ic_sampler.sample(Nx=Nx, batch_size=1, device=device).squeeze(0)  # (Nx,)

        base_max_speed = float(equation.max_wavespeed(u0).max().item())
        if base_max_speed < 1e-14:
            skipped += 1
            continue

        if cfl_range is not None:
            assert dt is not None
            cfl_min, cfl_max = cfl_range
            r = float(torch.rand((), device=device).item())
            sample_cfl = float(cfl_min) + r * (float(cfl_max) - float(cfl_min))
            sample_dt = float(dt)
            max_speed = sample_cfl * float(dx) / sample_dt
            u0 = u0 * (max_speed / base_max_speed)
        elif dt is not None and cfl is not None:
            sample_cfl = float(cfl)
            sample_dt = float(dt)
            max_speed = sample_cfl * float(dx) / sample_dt
            u0 = u0 * (max_speed / base_max_speed)
        elif dt is not None:
            max_speed = base_max_speed
            sample_dt = float(dt)
            sample_cfl = max_speed * sample_dt / float(dx)
        else:
            assert cfl is not None
            sample_cfl = float(cfl)
            max_speed = base_max_speed
            sample_dt = sample_cfl * float(dx) / max_speed

        if not math.isfinite(sample_dt) or sample_dt <= 0.0:
            raise ValueError(f"Computed non-positive/non-finite dt={sample_dt}.")
        if not math.isfinite(sample_cfl) or sample_cfl <= 0.0:
            raise ValueError(f"Computed non-positive/non-finite CFL={sample_cfl}.")
        if not math.isfinite(max_speed) or max_speed <= 0.0:
            raise ValueError(f"Computed non-positive/non-finite max_speed={max_speed}.")

        traj = [u0.float()]
        u0_d = u0.double()
        for n in range(1, n_steps + 1):
            u_t = equation.lax_hopf_reference(u0_d, x, n * sample_dt, n_quad=n_quad)
            traj.append(u_t.float())

        all_trajectories.append(torch.stack(traj, dim=0))  # (n_steps+1, Nx)
        all_dts.append(sample_dt)
        all_cfls.append(sample_cfl)
        all_max_speeds.append(max_speed)
        pbar.set_postfix(
            {
                "stored": len(all_trajectories),
                "skipped": skipped,
                "dt": f"{sample_dt:.2e}",
                "cfl": f"{sample_cfl:.2e}",
                "max|u|": f"{max_speed:.2e}",
            }
        )

    if not all_trajectories:
        raise RuntimeError("All samples were skipped (zero wavespeed). Check IC config.")

    n_stored = len(all_trajectories)
    dts = torch.tensor(all_dts, dtype=torch.float32)
    cfls = torch.tensor(all_cfls, dtype=torch.float32)
    max_speeds = torch.tensor(all_max_speeds, dtype=torch.float32)
    dt_min = float(min(all_dts))
    dt_max = float(max(all_dts))
    dt_mean = float(sum(all_dts) / len(all_dts))
    cfl_min_actual = float(min(all_cfls))
    cfl_max_actual = float(max(all_cfls))
    cfl_mean_actual = float(sum(all_cfls) / len(all_cfls))
    max_speed_min = float(min(all_max_speeds))
    max_speed_max = float(max(all_max_speeds))
    max_speed_mean = float(sum(all_max_speeds) / len(all_max_speeds))
    if dt is not None and cfl_range is not None:
        time_step_mode = "random_cfl_fixed_dt"
    elif dt is not None and cfl is not None:
        time_step_mode = "fixed_cfl_fixed_dt"
    elif dt is not None:
        time_step_mode = "fixed_dt"
    elif cfl is not None:
        time_step_mode = "fixed_cfl"
    else:
        raise RuntimeError("Unexpected time-step mode.")

    return {
        "trajectories": torch.stack(all_trajectories, dim=0),  # (N, n_steps+1, Nx)
        "dts": dts,
        "cfls": cfls,
        "max_speeds": max_speeds,
        "dx": float(dx),
        "meta": {
            "Nx": Nx,
            "n_steps": n_steps,
            "n_quad": n_quad,
            "domain_length": Nx * dx,
            "dx": dx,
            "dt": float(dt) if dt is not None else None,
            "cfl": float(cfl) if cfl is not None else None,
            "cfl_range": (
                None
                if cfl_range is None
                else [float(cfl_range[0]), float(cfl_range[1])]
            ),
            "time_step_mode": time_step_mode,
            "dt_min": dt_min,
            "dt_max": dt_max,
            "dt_mean": dt_mean,
            "cfl_min": cfl_min_actual,
            "cfl_max": cfl_max_actual,
            "cfl_mean": cfl_mean_actual,
            "max_speed_min": max_speed_min,
            "max_speed_max": max_speed_max,
            "max_speed_mean": max_speed_mean,
            "n_samples_requested": n_samples,
            "n_samples_stored": n_stored,
        },
    }


def _positive_float(value: float | int | str, *, name: str) -> float:
    out = float(value)
    if not math.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} must be finite and > 0, got {value}.")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Lax-Hopf supervised training data")
    parser.add_argument("--config", default="configs/burgers_default.yaml")
    parser.add_argument(
        "--output", default=None,
        help="Output path for the dataset (.pt file); overrides training_data.data_path in config",
    )
    parser.add_argument("--n-samples", type=int, default=None, help="Number of ICs to generate (overrides training_data.n_samples in config)")
    parser.add_argument(
        "--n-steps", type=int, default=None,
        help="Time steps per trajectory (default: n_rollout_max from config)",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help=(
            "Fixed time step; overrides training_data.dt from the config. "
            "May be combined with --cfl or --cfl-range to rescale IC amplitudes."
        ),
    )
    time_group = parser.add_mutually_exclusive_group()
    time_group.add_argument(
        "--cfl",
        type=float,
        default=None,
        help=(
            "Fixed CFL number. With --dt, keeps dt fixed and rescales each IC so "
            "max_wavespeed(u0) = cfl * dx / dt. Without --dt, computes per-sample "
            "dt as cfl * dx / max_wavespeed(u0). Mutually exclusive with --cfl-range."
        ),
    )
    time_group.add_argument(
        "--cfl-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help=(
            "Random CFL range. Samples one CFL uniformly from [MIN, MAX] per "
            "trajectory, keeps dt fixed, and rescales the IC so "
            "max_wavespeed(u0) = cfl * dx / dt. Mutually exclusive with --cfl."
        ),
    )
    parser.add_argument(
        "--n-quad", type=int, default=8,
        help="Sub-cell quadrature points per cell for Lax-Hopf cell averaging",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device(args.device)

    td_cfg = config.get("training_data", {})
    Nx: int = td_cfg.get("Nx", 256)
    dx_raw = td_cfg.get("dx")
    if dx_raw is None:
        raise ValueError("training_data.dx must be set in the config")
    dx = _positive_float(dx_raw, name="training_data.dx")

    cfl = _positive_float(args.cfl, name="--cfl") if args.cfl is not None else None
    cfl_range: tuple[float, float] | None = None
    if args.cfl_range is not None:
        cfl_min = _positive_float(args.cfl_range[0], name="--cfl-range MIN")
        cfl_max = _positive_float(args.cfl_range[1], name="--cfl-range MAX")
        if cfl_max < cfl_min:
            raise ValueError(
                f"--cfl-range MAX must be >= MIN, got MIN={cfl_min}, MAX={cfl_max}."
            )
        cfl_range = (cfl_min, cfl_max)

    if args.dt is not None:
        dt = _positive_float(args.dt, name="--dt")
    elif cfl is None:
        dt_raw = td_cfg.get("dt")
        if dt_raw is None:
            if cfl_range is not None:
                raise ValueError(
                    "--cfl-range requires a fixed dt. Pass --dt or set training_data.dt "
                    "in the config."
                )
            raise ValueError(
                "Set training_data.dt in the config, or pass --dt, --cfl, or --cfl-range "
                "with a fixed dt."
            )
        dt = _positive_float(dt_raw, name="training_data.dt")
    else:
        dt = None

    n_steps: int = args.n_steps or td_cfg.get("Nt", 20)
    n_samples: int = args.n_samples or td_cfg.get("n_samples", 5000)

    ic_cfg = td_cfg.get("initial_conditions", {})
    if ic_cfg.get("type") == "riemann":
        ic_sampler = RiemannIC(
            u_left_min=ic_cfg.get("u_left_min", -1.5),
            u_left_max=ic_cfg.get("u_left_max", 1.5),
            u_right_min=ic_cfg.get("u_right_min", -1.5),
            u_right_max=ic_cfg.get("u_right_max", 1.5),
        )
    else:
        ic_sampler = TopHatIC(
            n_min=ic_cfg.get("n_hats_min", 2),
            n_max=ic_cfg.get("n_hats_max", 6),
            w_min=ic_cfg.get("width_min", 0.05),
            w_max=ic_cfg.get("width_max", 0.25),
            a_min=ic_cfg.get("amplitude_min", -1.5),
            a_max=ic_cfg.get("amplitude_max", 1.5),
            domain_length=Nx * dx,
        )
    equation = BurgersEquation()

    ic_type = ic_cfg.get("type", "top_hat")
    if cfl_range is not None:
        time_desc = (
            f"dt={dt}, cfl_range=[{cfl_range[0]}, {cfl_range[1]}] "
            "(per-sample cfl ~ Uniform range; IC scaled so max|u0| = cfl*dx/dt)"
        )
    elif dt is not None and cfl is not None:
        time_desc = (
            f"dt={dt}, cfl={cfl} "
            "(IC scaled so max|u0| = cfl*dx/dt)"
        )
    elif dt is not None:
        time_desc = f"dt={dt}"
    else:
        time_desc = f"cfl={cfl} (per-sample dt = cfl*dx/max|u0|)"
    logger.info(
        f"Generating {n_samples} samples | "
        f"Nx={Nx}, Nt={n_steps}, dx={dx}, {time_desc}, ic={ic_type}, device={device}"
    )

    data = generate_dataset(
        n_samples=n_samples,
        n_steps=n_steps,
        Nx=Nx,
        dx=dx,
        dt=dt,
        cfl=cfl,
        cfl_range=cfl_range,
        ic_sampler=ic_sampler,
        equation=equation,
        n_quad=args.n_quad,
        device=device,
    )

    output_path = Path(args.output or td_cfg.get("data_path") or "data/lax_hopf_train.pt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, output_path)

    traj = data["trajectories"]
    size_mb = traj.numel() * 4 / 1e6  # float32
    logger.info(f"Saved to {output_path}")
    logger.info(f"  shape : {tuple(traj.shape)}  ({size_mb:.1f} MB)")
    logger.info(f"  dx    : {data['dx']:.6f}")
    logger.info(
        "  dt    : "
        f"[{data['dts'].min().item():.6f}, {data['dts'].max().item():.6f}] "
        f"(mean {data['dts'].mean().item():.6f})"
    )
    logger.info(
        "  cfl   : "
        f"[{data['cfls'].min().item():.6f}, {data['cfls'].max().item():.6f}] "
        f"(mean {data['cfls'].mean().item():.6f})"
    )
    logger.info(
        "  max|u|: "
        f"[{data['max_speeds'].min().item():.6f}, {data['max_speeds'].max().item():.6f}] "
        f"(mean {data['max_speeds'].mean().item():.6f})"
    )
    if dt is None:
        logger.warning(
            "  note  : CFL-based generation stores per-sample dt values; downstream code that "
            "expects one fixed dt per dataset may need a fixed --dt dataset instead."
        )
    logger.info(
        f"  stored: {data['meta']['n_samples_stored']} / "
        f"{data['meta']['n_samples_requested']}"
    )


if __name__ == "__main__":
    main()
