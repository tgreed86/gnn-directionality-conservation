# Directionality and Conservation in Graph Neural PDE Solvers

Research code accompanying **Directionality and Conservation in Graph Neural PDE
Solvers**, by Trevor G. Reed, Alex Gates, H. S. Udaykumar, and Stephen S. Baek.

This study examines how directional information and conservation constraints
influence graph neural network predictions of transport-dominated dynamics.
The benchmarks are the one-dimensional heat equation, linear advection, inviscid
Burgers' equation, Lighthill–Whitham–Richards traffic flow on road networks, and
two-dimensional chemotactic cell migration on irregular meshes. Comparisons
include GraphSAGE, MeshGraphNet, and a flux-based conservative variant called
FluxGraphNet, with and without precomputed advection inputs where applicable.

The repository provides dataset generators, model implementations, training and
evaluation scripts, and a workflow for reproducing the paper's figures and tables
from saved numerical results. It records 27 selected runs: 26 transport-model
runs and one heat-equation control.

**Artifact availability:** the dataset DOI is reserved but the record is not yet
published. Public downloads for checkpoints and saved evaluations have not yet
been established. Figure/table reproduction requires the datasets and saved
results described below; a source-only checkout does not contain those inputs.

## Repository contents

| Directory | Contents |
| --- | --- |
| `code/heat_equation/` | Heat-equation generator, model, trainer, and evaluation tools |
| `code/1D_advection/` | Linear-advection generator, models, training, and evaluation tools |
| `code/Burgers/` | Burgers models, training/evaluation tools, and WENO-A data generator |
| `code/1D_traffic/` | Road-network generators, GNN training, and evaluation/plotting tools |
| `code/chemotactic_cell_migration/` | Mesh-based chemotaxis generator, GNN training, and evaluation tools |
| `code/paper/` | Portable metrics, table-formatting, and saved-result plotting scripts |
| `generation/` | Reconstructed generation commands and supporting configurations |
| `environments/` | Workflow-specific dependencies and observed local/HPC environment records |
| `models/` | Selected-run configurations, split metadata, summaries, and training histories; checkpoint binaries are distributed separately |
| `reproduction/` | Figure recipes, reproduction entry points, and architecture diagram source |
| `manifests/` | Dataset/run mappings, artifact hashes, and records of numerical checks |
| `inventory/` | Detailed model and dataset inventories |
| `provenance/` | Original selection scripts and source snapshots retained for traceability |

See the [model inventory](inventory/MODELS.md) for individual runs and the
[dataset inventory](inventory/DATASETS.md) for data characteristics and provenance.
All commands below assume the repository root is the current directory.

## Installation

### Figures and tables from saved results

This workflow uses NumPy and Matplotlib and does not require model training or
PyTorch. It was exercised with Python 3.10.18, NumPy 1.26.4, and Matplotlib 3.8.4.
With Python 3.10 installed, create a dedicated environment:

```sh
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r environments/requirements-saved-results.txt
```

These versions describe the tested saved-results workflow, not the original
training environment. A clean installation test of this repository has not
been performed.

### Model inference, training, and dataset generation

Model execution additionally requires PyTorch and PyTorch Geometric. Some
entry points also require SciPy, PyYAML, or tqdm; the
[dependency inventory](manifests/dependencies.json) lists imports by project.
The author's local environment and a report from the HPC container have now
been recorded in the [environment guide](environments/README.md). They differ:
the local installation uses Python 3.11.13 and PyTorch 2.7.1; the HPC report uses
Python 3.12.3 and PyTorch `2.7.0a0+79aa17489c.nv25.4` built with CUDA 12.9.
The author confirms that the HPC report used the relocated original
`pytorch-2.7.0.sif` container.
The local core dependencies are pinned in
[requirements-local-core.txt](environments/requirements-local-core.txt).
A fresh installation and complete training/generation test with those pins
have not been performed. The full HPC freeze contains container-local wheel references
and is retained as provenance, not as a portable installation command.

The six Burgers visualization rollouts were tested on CPU with Python 3.11.13
and the versions in
[requirements-burgers-inference.txt](reproduction/requirements-burgers-inference.txt).
For that specific workflow, use a separate environment:

```sh
python3.11 -m venv .venv-burgers
source .venv-burgers/bin/activate
python -m pip install -r reproduction/requirements-burgers-inference.txt
```

This environment records the tested Burgers inference stack. It is not a
validated GPU installation or an environment specification for every benchmark.

## Obtain the research artifacts

See [ARTIFACTS.md](ARTIFACTS.md) for download status, exact extraction steps,
and checks that distinguish a valid source checkout from complete research inputs.
The default check succeeds without downloads but checks any artifacts present:

```sh
python tools/verify_inventory.py
```

### Datasets

Dataset record: [10.5281/zenodo.23369294](https://doi.org/10.5281/zenodo.23369294)
(**reserved; not yet published**).

The deposit contains five archives and accompanying descriptions, split IDs,
license, citation metadata, and SHA-256 checksums:

- `heat_equation.zip`
- `linear_advection.zip`
- `burgers.zip`
- `lwr_traffic.zip`
- `chemotactic_cell_migration.zip`

Once the record is published, download the deposit files into a separate download
folder and verify them using its `SHA256SUMS.txt`. From that download folder, use
`shasum -a 256 -c SHA256SUMS.txt` on macOS or `sha256sum -c SHA256SUMS.txt` on Linux.

Extract only the `datasets/` entries into this repository. Each archive also
contains dataset documentation; extracting those top-level files into the
repository would overwrite its README and potentially its citation metadata.
From the repository root, replace `/path/to/downloads` with your download folder:

```sh
for archive in heat_equation.zip linear_advection.zip burgers.zip lwr_traffic.zip chemotactic_cell_migration.zip; do
  unzip "/path/to/downloads/$archive" 'datasets/*' -d .
done
```

The resulting directories must be:

```text
datasets/
  heat_equation/
  1D_advection/
  Burgers/
  1D_traffic/lwr_structured_directional/
  chemotactic_cell_migration/unique_meshes_17x17/
```

Burgers includes two distinct banks. The length-one bank supplies training and
quantitative test trajectories. The length-three bank supplies the qualitative
space-time examples, using zero-based trajectory index 1. Do not substitute the
visualization bank for the quantitative test data.

### Checkpoints and saved evaluations

Public download locations for these artifacts are not yet available. Their
required repository-relative destinations are recorded in
[runs.json](manifests/runs.json), [evaluations.json](manifests/evaluations.json),
and [generated_evaluations.json](manifests/generated_evaluations.json):

- Checkpoint binaries belong under `models/<benchmark>/<run>/`, alongside the
  configuration and history files already in the repository.
- Saved numerical results belong under `evaluations/<benchmark>/<run>/`, or
  the other exact paths specified in the evaluation manifests.

The archived results comprise 101 original evaluation/prediction archives and
six newly generated Burgers visualization archives. They are inputs to the
figure/table workflow. Checkpoint binaries are needed for fresh model inference;
the saved-results workflow uses exported model metadata instead.

## Reproduce figures and tables

List the available figure recipes without downloading research artifacts:

```sh
python reproduction/reproduce.py list
```

After installing the saved-results environment and obtaining the required data
and evaluation archives, run:

```sh
python tools/verify_inventory.py --require saved-results
python reproduction/reproduce.py tables
python reproduction/reproduce.py figures
```

Or run both together:

```sh
python reproduction/reproduce.py all
```

The workflow writes three LaTeX tables to `outputs/tables/`, 23 PNG figure assets
to `outputs/figures/`, and an execution record to `outputs/reproduction_run.json`.
It recomputes metrics and renders saved predictions; it does not retrain models
or run learned-model inference. Table metrics are unsmoothed. The training table
reports completed epochs separately from the validation-selected best epoch.
Each run first verifies the complete dataset/evaluation collection and reports
missing or corrupt inputs before launching jobs; checkpoint binaries are optional
for this workflow. The `list` command works without artifact downloads.

Use `--only RECIPE_ID` with the `figures` task to select a figure; identifiers
are listed by the `list` command and in the [figure index](reproduction/FIGURE_INDEX.md).
Use `--output-dir DIRECTORY` to select a different output directory.

The architecture diagram is separate from the PNG workflow. With an existing
LaTeX installation containing its required packages:

```sh
mkdir -p outputs/figures
pdflatex -interaction=nonstopmode -halt-on-error \
  -output-directory=outputs/figures reproduction/architecture_standalone.tex
```

## Generate data, train models, and run inference

### Generate new datasets

The [generation guide](generation/README.md) supplies commands for all five
benchmarks and both Burgers banks. Execute those commands from this repository's
root. They write to `regenerated/`, preserving the downloaded data in `datasets/`.
To train on newly generated data, point a working copy of the training
configuration to the generated files.

The generation commands are reconstructed from saved metadata and current
source snapshots. They are not recovered historical shell commands. In
particular, the original Burgers initial-condition RNG state is unavailable;
use the archived banks to reproduce the paper's specific samples.

### Train models

[Each selected run](manifests/runs.json) specifies a `training_entry_point` and
`release_configuration`. Use the corresponding `config_release.json`: its data
paths are relative to the repository root, and its training outputs go under
`outputs/`. Original checkpoint configurations retain historical paths for
provenance and should not be used as portable defaults.

| Benchmark | Training entry point |
| --- | --- |
| Heat | `code/heat_equation/scripts/train_gnn_heat.py` |
| Linear advection | `code/1D_advection/scripts/train_gnn_advection.py` |
| Burgers | `code/Burgers/scripts/train_gnn_burgers.py` |
| Traffic | `code/1D_traffic/scripts/train.py` |
| Chemotaxis | `code/chemotactic_cell_migration/scripts/train.py` |

All five entry points accept `--config`. For example, with the required dataset,
training dependencies, and configured device available:

```sh
python code/Burgers/scripts/train_gnn_burgers.py \
  --config models/Burgers/fluxgraphnet_with_advection/config_release.json
```

Configurations retain their original device choices (`cuda` or `mps` where
applicable). Before training, copy the selected configuration and set its
`device` to an available backend, such as `cpu`, and choose a new output directory
for each experiment. Heat, advection, and Burgers use `train.save_dir`; traffic
and chemotaxis use `output`. Change the `--config` argument to that working copy.
Full retraining of all selected runs in a clean environment has not been tested.

### Regenerate Burgers visualization predictions

With the Burgers inference environment active, both Burgers datasets, the six
selected checkpoints, and their saved quantitative evaluation archives present:

```sh
python reproduction/generate_burgers_visualization.py
```

This runs CPU inference for the six length-three examples and writes fresh arrays
and comparison records under `outputs/burgers_visualization/`. It first compares
each model against a historical length-one test trajectory. The ordinary figure
runner continues to use the archived arrays under `evaluations/`; this inference
command does not replace them automatically.

Other benchmark evaluation entry points are included in their respective source
directories. Their original checkpoint paths and dataset overrides require
attention when running fresh inference; they have not all been validated as a
single portable workflow.

## Validation and reproducibility limits

The following checks were performed on the complete archival package before
separating source code from downloadable research artifacts:

- All three tables and 23 PNG assets were generated from a relocated package
  with reads of the original project directories blocked. All 26 numerical rows
  of the results table matched the manuscript values.
- All 27 checkpoints loaded on CPU with finite model-state tensors. Six Burgers
  models additionally passed strict model/checkpoint loading and completed the
  length-three visualization rollouts. Inference for the other 21 models remains
  unvalidated.
- Heat and advection generation replays reproduced all stored float32 states.
  Traffic replays reproduced stored states and transfer fluxes for all 32
  topologies. Two chemotaxis trajectories matched after aligning cell IDs;
  triangulation ordering can vary between numerical-library versions.
- Fresh Burgers rollouts did not exactly reproduce all historical predictions.
  The largest full-rollout absolute difference was 0.03269, with relative L2
  difference 0.004666, for the absolute-edge MeshGraphNet. The cause has not been
  established. Original quantitative evaluation archives remain unchanged, and
  the six new visualization archives have separate provenance records.

Local package versions and an HPC runtime report are recorded in the
[environment guide](environments/README.md). The author confirms use of the
original HPC image; its digest, per-run historical package versions, and
training-time source revision have not been established. Rendering can also
differ with fonts and library versions. These
checks do not establish bitwise-identical retraining or universal platform
compatibility. Fresh-installation and public artifact-download testing have not
been performed for the separated source repository.

Numerical check records are available in
[reproduction_checks.json](manifests/reproduction_checks.json),
[dataset_audit.json](manifests/dataset_audit.json), and
[generator_replay_checks.json](manifests/generator_replay_checks.json).
The standard-library [inventory verifier](tools/verify_inventory.py) checks
repository structure and manifest consistency, plus SHA-256 hashes for research
artifacts present. `--require datasets`, `--require saved-results`, and
`--require all` enforce progressively broader artifact completeness. Its checks
do not run models or validate a fresh environment installation. Fresh-installation
testing was not performed for this release preparation.

## License and citation

Copyright (c) 2026 Trevor G. Reed.

Original software is licensed under the [MIT License](LICENSES/MIT.txt).
Original documentation, reference figures, tables, and diagrams are licensed
under [CC BY 4.0](LICENSES/CC-BY-4.0.txt), as are the separately distributed
datasets, trained checkpoints, and numerical evaluation outputs.
See the [component license notice](LICENSE) for scope. Third-party dependencies
and material bearing separate notices retain their own terms.

The sole software creator is Trevor G. Reed. Please use the software metadata
in [CITATION.cff](CITATION.cff) to cite these implementations. The software's
creator list is distinct from the associated paper's author list.

Please cite the associated paper when using its methods or results, and cite the
[dataset record](https://doi.org/10.5281/zenodo.23369294) when using the datasets
(its DOI remains reserved until publication). The dataset creator is Trevor G.
Reed, [ORCID 0000-0002-4230-6687](https://orcid.org/0000-0002-4230-6687).

The source repository is hosted at
[tgreed86/gnn-directionality-conservation](https://github.com/tgreed86/gnn-directionality-conservation).
The software release version/date, software DOI, and the paper's final
bibliographic details will be added when available. The dataset
DOI identifies the datasets and should not be used as the identifier for this
software repository.
