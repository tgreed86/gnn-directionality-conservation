#!/usr/bin/env python3
"""CLI for generating periodic heat-equation training data.

Quick start (from repo root):

    python scripts/generate_heat_data.py \
        --spatial-dim 2 \
        --nx 32 \
        --ny 32 \
        --num-samples 100 \
        --window 50 \
        --m-max-x 8 \
        --m-max-y 8 \
        --CFL 0.2 \
        --nu 0.01 \
        --target-type rate \
        --out data/single_steps/heat_train.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from heat_data import PeriodicHeatDataset, save_heat_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate supervised u^n -> (u^{n+1} or rate) data for the periodic "
            "heat equation using exact FFT spectral diffusion labels."
        )
    )
    parser.add_argument(
        "--spatial-dim",
        "--dim",
        type=int,
        default=2,
        choices=(1, 2),
        help="Spatial dimension for the heat equation. Use 1 for u_t=nu*u_xx.",
    )
    parser.add_argument("--nx", type=int, default=32, help="Number of x-direction nodes.")
    parser.add_argument(
        "--ny",
        type=int,
        default=32,
        help="Number of y-direction nodes. Ignored when --spatial-dim 1.",
    )
    parser.add_argument("--Lx", type=float, default=1.0, help="Domain length in x.")
    parser.add_argument(
        "--Ly",
        type=float,
        default=1.0,
        help="Domain length in y. Ignored when --spatial-dim 1.",
    )
    parser.add_argument("--nu", type=float, default=0.01, help="Diffusivity.")
    parser.add_argument(
        "--CFL",
        type=float,
        default=0.2,
        help="Diffusion CFL-like number for choosing dt.",
    )
    parser.add_argument(
        "--num-samples", type=int, default=1000, help="Number of IC samples."
    )
    parser.add_argument(
        "--window",
        type=int,
        default=1,
        help=(
            "Number of one-step transitions per trajectory. "
            "Total flattened training pairs = num_samples * window."
        ),
    )
    parser.add_argument("--m-max-x", type=int, default=8, help="Max Fourier mode in x.")
    parser.add_argument(
        "--m-max-y",
        type=int,
        default=8,
        help="Max Fourier mode in y. Ignored when --spatial-dim 1.",
    )
    parser.add_argument(
        "--amplitude", type=float, default=1.0, help="IC coefficient amplitude scale."
    )
    parser.add_argument(
        "--spectral-decay",
        type=float,
        default=1.0,
        help="Spectral decay exponent p in 1/(m_x^2+m_y^2)^(p/2).",
    )
    parser.add_argument(
        "--normalize-rms",
        type=float,
        default=None,
        help="Optional per-sample RMS normalization target.",
    )
    parser.add_argument(
        "--target-type",
        type=str,
        default="rate",
        choices=("rate", "next"),
        help="Supervision target type.",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed.")
    parser.add_argument("--out", type=Path, required=True, help="Output .npz file path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = PeriodicHeatDataset(
        spatial_dim=args.spatial_dim,
        nx=args.nx,
        ny=args.ny,
        Lx=args.Lx,
        Ly=args.Ly,
        nu=args.nu,
        CFL=args.CFL,
        num_samples=args.num_samples,
        window=args.window,
        m_max_x=args.m_max_x,
        m_max_y=args.m_max_y,
        amplitude=args.amplitude,
        spectral_decay=args.spectral_decay,
        target_type=args.target_type,
        seed=args.seed,
        normalize_rms=args.normalize_rms,
        dtype=torch.float32,
    )
    save_heat_dataset(dataset, args.out)

    print("Generated heat-equation dataset")
    print(f"  out: {args.out}")
    print(f"  trajectory samples: {dataset.metadata['num_samples']}")
    print(f"  rollout window: {dataset.metadata.get('window', 1)}")
    print(f"  total training pairs: {len(dataset)}")
    print(f"  spatial_dim: {dataset.metadata['spatial_dim']}")
    print(f"  nx, ny: {dataset.metadata['nx']}, {dataset.metadata['ny']}")
    print(f"  N: {dataset.metadata['N']}")
    print(f"  Lx, Ly: {dataset.metadata['Lx']}, {dataset.metadata['Ly']}")
    print(f"  nu: {dataset.metadata['nu']}")
    print(f"  CFL: {dataset.metadata['CFL']}")
    print(f"  dx: {dataset.metadata['dx']:.8e}")
    print(f"  dy: {dataset.metadata['dy']:.8e}")
    print(f"  dt: {dataset.metadata['dt']:.8e}")
    print(f"  m_max_x: {dataset.metadata['m_max_x']}")
    print(f"  m_max_y: {dataset.metadata['m_max_y']}")
    print(f"  amplitude: {dataset.metadata['amplitude']}")
    print(f"  spectral_decay: {dataset.metadata['spectral_decay']}")
    print(f"  target_type: {dataset.metadata['target_type']}")
    print(f"  seed: {dataset.metadata['seed']}")
    print(f"  x shape: {tuple(dataset.x.shape)}")
    print(f"  y shape: {tuple(dataset.y.shape)}")
    if dataset.rollout_states is not None:
        print(f"  rollout_states shape: {tuple(dataset.rollout_states.shape)}")
    print(f"  edge_index shape: {tuple(dataset.edge_index.shape)}")
    print(f"  edge_attr shape: {tuple(dataset.edge_attr.shape)}")


if __name__ == "__main__":
    main()
