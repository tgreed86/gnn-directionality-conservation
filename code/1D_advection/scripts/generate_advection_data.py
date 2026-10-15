#!/usr/bin/env python3
"""CLI for generating 1D periodic advection training data.

Quick start (from an activated `amr-gnn-pt27` shell, at repo root):

    python scripts/generate_advection_data.py \
        --N 128 \
        --num-samples 50 \
        --window 500 \
        --m-max 16 \
        --CFL 0.25 \
        --c 1.0 \
        --target-type rate \
        --out data/advection_train.npz

Shock-like discontinuous ICs with analytic characteristic labels:

    python scripts/generate_advection_data.py \
        --N 128 \
        --num-samples 50 \
        --window 128 \
        --CFL 1.0 \
        --ic-type shock \
        --label-method analytic \
        --shock-num-jumps-min 2 \
        --shock-num-jumps-max 4 \
        --target-type next \
        --out data/rollout/advection_shock_train.npz

This writes a compressed `.npz` file containing:
- flattened single-step training pairs:
  `x`, `y`, `u_next` with shape [num_samples * window, N, 1]
- rollout indexing arrays:
  `pair_trajectory_id`, `pair_time_index`
- full rollout trajectories:
  `rollout_states` with shape [num_samples, window+1, N, 1]
- graph data (`edge_index`, `edge_attr`) and metadata fields.
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

from advection_data import PeriodicAdvectionDataset, save_advection_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate supervised u^n -> (u^{n+1} or rate) data for 1D periodic "
            "linear advection using exact spectral or analytic shift labels."
        )
    )
    parser.add_argument("--N", type=int, default=128, help="Number of nodes.")
    parser.add_argument("--L", type=float, default=1.0, help="Domain length.")
    parser.add_argument("--c", type=float, default=1.0, help="Advection speed.")
    parser.add_argument("--CFL", type=float, default=0.25, help="CFL number.")
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
    parser.add_argument("--m-max", type=int, default=16, help="Max Fourier mode.")
    parser.add_argument(
        "--amplitude", type=float, default=1.0, help="IC coefficient amplitude scale."
    )
    parser.add_argument(
        "--spectral-decay",
        type=float,
        default=1.0,
        help="Spectral decay exponent p in 1/m^p.",
    )
    parser.add_argument(
        "--ic-type",
        type=str,
        default="smooth",
        choices=("smooth", "shock", "mixed"),
        help="Initial-condition family.",
    )
    parser.add_argument(
        "--label-method",
        type=str,
        default="spectral",
        choices=("spectral", "analytic"),
        help=(
            "Label-generation method. "
            "Use 'analytic' to advect shock ICs by characteristic shift."
        ),
    )
    parser.add_argument(
        "--shock-fraction",
        type=float,
        default=0.5,
        help="For ic-type=mixed: fraction of trajectories sampled as shock ICs.",
    )
    parser.add_argument(
        "--shock-num-jumps-min",
        type=int,
        default=2,
        help="Minimum number of jumps for shock ICs.",
    )
    parser.add_argument(
        "--shock-num-jumps-max",
        type=int,
        default=4,
        help="Maximum number of jumps for shock ICs.",
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
    parser.add_argument(
        "--state-representation",
        type=str,
        default="point",
        choices=("point", "cell_average"),
        help=(
            "Stored state semantics. Use 'point' for point samples at x_i or "
            "'cell_average' for exact averages over cells centered at x_i."
        ),
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed.")
    parser.add_argument(
        "--out", type=Path, required=True, help="Output .npz file path."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = PeriodicAdvectionDataset(
        N=args.N,
        L=args.L,
        c=args.c,
        CFL=args.CFL,
        num_samples=args.num_samples,
        window=args.window,
        m_max=args.m_max,
        amplitude=args.amplitude,
        spectral_decay=args.spectral_decay,
        ic_type=args.ic_type,
        label_method=args.label_method,
        shock_fraction=args.shock_fraction,
        shock_num_jumps_min=args.shock_num_jumps_min,
        shock_num_jumps_max=args.shock_num_jumps_max,
        target_type=args.target_type,
        state_representation=args.state_representation,
        seed=args.seed,
        normalize_rms=args.normalize_rms,
        dtype=torch.float32,
    )
    save_advection_dataset(dataset, args.out)

    print("Generated advection dataset")
    print(f"  out: {args.out}")
    print(f"  trajectory samples: {dataset.metadata['num_samples']}")
    print(f"  rollout window: {dataset.metadata.get('window', 1)}")
    print(f"  total training pairs: {len(dataset)}")
    print(f"  N: {dataset.metadata['N']}")
    print(f"  L: {dataset.metadata['L']}")
    print(f"  c: {dataset.metadata['c']}")
    print(f"  CFL: {dataset.metadata['CFL']}")
    print(f"  dx: {dataset.metadata['dx']:.8e}")
    print(f"  dt: {dataset.metadata['dt']:.8e}")
    print(f"  m_max: {dataset.metadata['m_max']}")
    print(f"  amplitude: {dataset.metadata['amplitude']}")
    print(f"  spectral_decay: {dataset.metadata['spectral_decay']}")
    print(f"  ic_type: {dataset.metadata.get('ic_type', 'smooth')}")
    print(f"  label_method: {dataset.metadata.get('label_method', 'spectral')}")
    print(f"  shock_fraction: {dataset.metadata.get('shock_fraction', 0.0)}")
    print(
        "  shock_num_jumps: "
        f"{dataset.metadata.get('shock_num_jumps_min', 2)}-"
        f"{dataset.metadata.get('shock_num_jumps_max', 2)}"
    )
    print(f"  target_type: {dataset.metadata['target_type']}")
    print(f"  state_representation: {dataset.metadata['state_representation']}")
    print(f"  seed: {dataset.metadata['seed']}")
    print(f"  x shape: {tuple(dataset.x.shape)}")
    print(f"  y shape: {tuple(dataset.y.shape)}")
    if dataset.rollout_states is not None:
        print(f"  rollout_states shape: {tuple(dataset.rollout_states.shape)}")
    print(f"  edge_index shape: {tuple(dataset.edge_index.shape)}")
    print(f"  edge_attr shape: {tuple(dataset.edge_attr.shape)}")


if __name__ == "__main__":
    main()
