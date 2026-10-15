# Model inventory

Snapshot: 2026-10-12. Manuscript: `drafts/10-12-26_APLML/`.

The first 26 entries follow `RUNS` exactly. The final heat-equation control was supplied separately by the user; its selection is recorded in `provenance/additional_models.json`.

| Run | Dataset | Method | Best epoch | State-dict elements |
|---|---|---|---:|---:|
| [Burgers/fluxgraphnet_with_advection](../models/Burgers/fluxgraphnet_with_advection/config_checkpoint.json) | Inviscid Burgers | FluxGraphNet + advection | 180 | 412,417 |
| [Burgers/meshgraphnet_with_advection](../models/Burgers/meshgraphnet_with_advection/config_checkpoint.json) | Inviscid Burgers | MeshGraphNet + advection | 190 | 395,905 |
| [Burgers/fluxgraphnet_no_advection](../models/Burgers/fluxgraphnet_no_advection/config_checkpoint.json) | Inviscid Burgers | FluxGraphNet (no advection) | 160 | 412,289 |
| [Burgers/meshgraphnet_no_advection](../models/Burgers/meshgraphnet_no_advection/config_checkpoint.json) | Inviscid Burgers | MeshGraphNet (no advection) | 160 | 395,777 |
| [Burgers/graphsage_with_advection](../models/Burgers/graphsage_with_advection/config_checkpoint.json) | Inviscid Burgers | SAGEConv + advection | 155 | 83,073 |
| [Burgers/graphsage_no_advection](../models/Burgers/graphsage_no_advection/config_checkpoint.json) | Inviscid Burgers | SAGEConv (no advection) | 380 | 82,817 |
| [Burgers/meshgraphnet-abs_no_advection](../models/Burgers/meshgraphnet-abs_no_advection/config_checkpoint.json) | Inviscid Burgers | MeshGraphNet-abs (no advection) | 485 | 395,777 |
| [1D_advection/fluxgraphnet_with_advection](../models/1D_advection/fluxgraphnet_with_advection/config_checkpoint.json) | Linear Advection | FluxGraphNet + advection | 124 | 412,417 |
| [1D_advection/meshgraphnet_with_advection](../models/1D_advection/meshgraphnet_with_advection/config_checkpoint.json) | Linear Advection | MeshGraphNet + advection | 98 | 395,905 |
| [1D_advection/fluxgraphnet_no_advection](../models/1D_advection/fluxgraphnet_no_advection/config_checkpoint.json) | Linear Advection | FluxGraphNet (no advection) | 238 | 412,289 |
| [1D_advection/meshgraphnet_no_advection](../models/1D_advection/meshgraphnet_no_advection/config_checkpoint.json) | Linear Advection | MeshGraphNet (no advection) | 296 | 395,777 |
| [1D_advection/graphsage_with_advection](../models/1D_advection/graphsage_with_advection/config_checkpoint.json) | Linear Advection | SAGEConv + advection | 62 | 83,073 |
| [1D_advection/graphsage_no_advection](../models/1D_advection/graphsage_no_advection/config_checkpoint.json) | Linear Advection | SAGEConv (no advection) | 12 | 82,817 |
| [1D_advection/meshgraphnet-abs_no_advection](../models/1D_advection/meshgraphnet-abs_no_advection/config_checkpoint.json) | Linear Advection | MeshGraphNet-abs (no advection) | 16 | 395,777 |
| [1D_traffic/fluxgraphnet_with_advection](../models/1D_traffic/fluxgraphnet_with_advection/config_checkpoint.json) | LWR Traffic | FluxGraphNet + advection | 867 | 546,689 |
| [1D_traffic/meshgraphnet_with_advection](../models/1D_traffic/meshgraphnet_with_advection/config_checkpoint.json) | LWR Traffic | MeshGraphNet + advection | 656 | 518,401 |
| [1D_traffic/fluxgraphnet_no_advection](../models/1D_traffic/fluxgraphnet_no_advection/config_checkpoint.json) | LWR Traffic | FluxGraphNet (no advection) | 971 | 546,561 |
| [1D_traffic/meshgraphnet_no_advection](../models/1D_traffic/meshgraphnet_no_advection/config_checkpoint.json) | LWR Traffic | MeshGraphNet (no advection) | 989 | 517,889 |
| [1D_traffic/graphsage_with_advection](../models/1D_traffic/graphsage_with_advection/config_checkpoint.json) | LWR Traffic | SAGEConv + advection | 454 | 102,785 |
| [1D_traffic/graphsage_no_advection](../models/1D_traffic/graphsage_no_advection/config_checkpoint.json) | LWR Traffic | SAGEConv (no advection) | 971 | 102,529 |
| [chemotactic_cell_migration/fluxgraphnet_with_advection](../models/chemotactic_cell_migration/fluxgraphnet_with_advection/config_checkpoint.json) | Chemotactic Cell Migration | FluxGraphNet + advection | 76 | 546,307 |
| [chemotactic_cell_migration/meshgraphnet_with_advection](../models/chemotactic_cell_migration/meshgraphnet_with_advection/config_checkpoint.json) | Chemotactic Cell Migration | MeshGraphNet + advection | 127 | 513,537 |
| [chemotactic_cell_migration/fluxgraphnet_no_advection](../models/chemotactic_cell_migration/fluxgraphnet_no_advection/config_checkpoint.json) | Chemotactic Cell Migration | FluxGraphNet (no advection) | 44 | 546,179 |
| [chemotactic_cell_migration/meshgraphnet_no_advection](../models/chemotactic_cell_migration/meshgraphnet_no_advection/config_checkpoint.json) | Chemotactic Cell Migration | MeshGraphNet (no advection) | 72 | 513,409 |
| [chemotactic_cell_migration/graphsage_with_advection](../models/chemotactic_cell_migration/graphsage_with_advection/config_checkpoint.json) | Chemotactic Cell Migration | SAGEConv + advection | 86 | 101,249 |
| [chemotactic_cell_migration/graphsage_no_advection](../models/chemotactic_cell_migration/graphsage_no_advection/config_checkpoint.json) | Chemotactic Cell Migration | SAGEConv (no advection) | 130 | 100,993 |
| [heat_equation/graphsage_1d](../models/heat_equation/graphsage_1d/config_checkpoint.json) | Heat Equation | SAGEConv | 78 | 82,817 |

State-dict element counts include any persistent buffers; they are not asserted to be trainable-parameter counts.

## Model implementations

- Burgers: `code/Burgers/src/gnn_burgers.py`.
- Linear advection: `code/1D_advection/src/gnn_advection.py`.
- Traffic: `code/1D_traffic/scripts/train.py`.
- Chemotaxis: `code/chemotactic_cell_migration/scripts/train.py`.
- Heat equation: `code/heat_equation/src/gnn_heat.py` (HeatSAGEModel).

The one-dimensional MeshGraphNet-abs variants are selected through their checkpoint configuration; they are not separate checkpoint classes.

## Checks and limits

- All 27 source run directories and selected checkpoints exist.
- Checkpoints were loaded on CPU with `weights_only=True`; all stored model tensors are finite.
- Every copied file has a SHA-256 recorded in `manifests/copied_files.json`.
- Configurations exported from the checkpoint are authoritative snapshots, including the Burgers configurations absent as standalone JSON files.
- Historical paths in saved configurations and metadata are preserved. Separate `config_release.json` copies now use the archived dataset paths and output paths relative to the release root; see `DATASETS.md`.
- Current code has not been proved identical to the code used at training time; no training-time source revision has been inferred.
- End-to-end inference, strict model/state-dict compatibility, and environment installation remain unverified.
- Current available CPU PyTorch environment lacks PyTorch Geometric and reports a NumPy ABI incompatibility. It was used only for checkpoint inspection; it is not a recommended release environment.
- Figure hashes and source-image matches are recorded in `manifests/manuscript_figures.json`. Dataset/reference checks and unresolved figure issues are detailed in `MANUSCRIPT_DATA_AUDIT.md`.
