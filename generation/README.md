# Reconstructed dataset-generation commands

Run from the `paper_release/` root, in an environment with the dependencies in `manifests/dependencies.json`. These commands target `regenerated/`, leaving the archived `datasets/` intact. They are reconstructed recipes, not recovered shell history. Do not point their output arguments at the archived datasets.

Heat and advection: all regenerated float32 rollout states matched. Traffic: a complete generator replay was performed; see `manifests/generator_replay_checks.json`. Chemotaxis: representative replay checks are documented there. Burgers: the original RNG seed/state and IC configuration are not recorded, so this recipe makes a new sample bank and cannot recreate the archived bank exactly. Use the archived Burgers dataset to reproduce the paper.

The four nominal chemotaxis substeps determine frame spacing; the reference solver increases this to 13 for stability. Do not replace `--substeps-per-frame 4` with 13. Omit `--shared-mesh`, `--shared-chemoattractant`, and `--normalize-rms` to retain the stored choices. A `--solver-dt` override equal to the saved nominal value is indistinguishable from automatic selection in the archived metadata.

## heat_equation

```sh
python code/heat_equation/scripts/generate_heat_data.py \
  --spatial-dim 1 \
  --nx 64 \
  --Lx 1 \
  --nu 0.01 \
  --CFL 0.2 \
  --num-samples 100 \
  --window 130 \
  --m-max-x 8 \
  --amplitude 1 \
  --spectral-decay 1 \
  --target-type rate \
  --seed 1234 \
  --out regenerated/heat_equation/100_samples_heat_train.npz
```

## 1D_advection

```sh
python code/1D_advection/scripts/generate_advection_data.py \
  --N 384 \
  --L 3 \
  --c 1 \
  --CFL 0.3 \
  --num-samples 100 \
  --window 128 \
  --m-max 16 \
  --amplitude 1 \
  --spectral-decay 1 \
  --ic-type shock \
  --label-method analytic \
  --shock-num-jumps-min 6 \
  --shock-num-jumps-max 12 \
  --target-type next \
  --state-representation cell_average \
  --seed 1234 \
  --out regenerated/1D_advection/advection_shock_N384_L3_cfl-0p3_cellavg.npz
```

## Burgers

```sh
python code/Burgers/data/weno-a/scripts/generate_data.py \
  --config generation/burgers_reconstructed.yaml \
  --output regenerated/Burgers/100-samples_cfl-0p2_dt-1p5e-3_dataset.pt \
  --n-samples 100 \
  --n-steps 300 \
  --dt 0.0015 \
  --cfl 0.2 \
  --n-quad 8 \
  --device cpu
```

## 1D_traffic

```sh
python code/1D_traffic/scripts/generate_structured_directional_lwr_traffic_data.py \
  --CFL 0.4 \
  --cells-per-unit 8.0 \
  --center-intersection-fraction-max 0.15 \
  --diagonal-cell-fraction-max 0.3 \
  --eval-trajectories-per-topology 4 \
  --grid-spacing 1.0 \
  --large-grid-cols-max 6 \
  --large-grid-cols-min 5 \
  --large-grid-rows-max 6 \
  --large-grid-rows-min 5 \
  --minimum-cells-per-road 3 \
  --num-blocks-max 10 \
  --num-blocks-min 3 \
  --out-dir regenerated/1D_traffic/lwr_structured_directional \
  --remove-road-fraction-max 0.25 \
  --rho-max 1.0 \
  --road-baseline-jitter 0.12 \
  --seed 1234 \
  --subdivide-road-fraction-max 0.15 \
  --test-in-size-topologies 4 \
  --test-large-topologies 4 \
  --test-parameter-ood-topologies 4 \
  --train-grid-cols-max 4 \
  --train-grid-cols-min 3 \
  --train-grid-rows-max 4 \
  --train-grid-rows-min 3 \
  --train-topologies 16 \
  --train-trajectories-per-topology 8 \
  --validation-topologies 4 \
  --window 256
```

## chemotactic_cell_migration

```sh
python code/chemotactic_cell_migration/scripts/generate_chemotactic_cell_migration_data.py \
  --nx 17 \
  --ny 17 \
  --lx 1 \
  --ly 1 \
  --mesh-mode delaunay \
  --mesh-jitter 0.3 \
  --num-samples 100 \
  --window 128 \
  --substeps-per-frame 4 \
  --reference-refinement-levels 1 \
  --reference-cfl 0.8 \
  --diffusion 0.01 \
  --chi 0.1 \
  --CFL 0.45 \
  --chemo-sources-min 1 \
  --chemo-sources-max 3 \
  --chemo-amplitude-min 0.5 \
  --chemo-amplitude-max 1.5 \
  --chemo-sigma-min 0.1 \
  --chemo-sigma-max 0.22 \
  --chemo-baseline 0 \
  --source-margin-fraction 0.08 \
  --density-blobs-min 1 \
  --density-blobs-max 3 \
  --density-amplitude-min 0.5 \
  --density-amplitude-max 1.5 \
  --density-sigma-min 0.06 \
  --density-sigma-max 0.14 \
  --density-background-min 0.01 \
  --density-background-max 0.05 \
  --density-margin-fraction 0.08 \
  --primary-separation 0.25 \
  --target-type increment \
  --one-trajectory-at-a-time \
  --seed 1234 \
  --out regenerated/chemotactic_cell_migration/unique_meshes_17x17
```

## Burgers visualization bank (length 3)

This is separate from the length-1 bank used for training and the checked metric
archives. The user identified it as the source for the Burgers space-time figures.
Its metadata records 10 trajectories, 768 cells, 300 transitions, dx=1/256,
dt=0.0015, CFL=0.2 and Q=8. The copied current `burgers_wider_domain.yaml` sets
6–18 hats. This command is reconstructed; it will not recover the identical
sample bank without the original RNG state. Use the archived file for the paper.

```sh
python code/Burgers/data/weno-a/scripts/generate_data.py \
  --config code/Burgers/data/weno-a/configs/burgers_wider_domain.yaml \
  --output regenerated/Burgers/10-samples_L-3_cfl-0p2_dt-1p5e-3_dataset.pt \
  --n-samples 10 --n-steps 300 --dt 0.0015 --cfl 0.2 --n-quad 8 --device cpu
```
