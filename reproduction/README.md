# Reproduce the paper's tables and figures

This workflow reads the archived datasets and saved predictions in this release.
The figure/table runner does not retrain models or rerun learned-model inference.
All 27 selected runs have archived evaluations. The recipes cover all 23 active
PNG assets in the local `drafts/10-13-26_APLML` manuscript and are runnable. The
architecture schematic is supplied as LaTeX source. Six length-three Burgers
prediction archives were newly generated from checkpoints, as described below.

Run from the repository root. Datasets and evaluation archives are distributed
separately; see [artifact installation](../ARTIFACTS.md). The code-only checkout
can list recipes and pass the default source check without those downloads.

```sh
python tools/verify_inventory.py
python reproduction/reproduce.py list
python tools/verify_inventory.py --require saved-results
python reproduction/reproduce.py tables
python reproduction/reproduce.py figures
```

Or run the tables and every established figure recipe together:

```sh
python reproduction/reproduce.py all
```

Table and figure commands run a strict dataset/evaluation preflight automatically.
Missing or corrupt archives produce installation guidance before any jobs launch.
This checks file integrity, not fresh-installation or model-execution validity.

There are no unresolved figure selections. The `--skip-unresolved` option remains
available if future recipes need clarification; it records each omission.
Use `--only RECIPE_ID` (repeatable) for individual figures;
see [the figure index](FIGURE_INDEX.md). `--output-dir DIRECTORY` selects another
output directory. Commands work from another working directory when the script
is addressed by its path.

Outputs go to `outputs/tables` and `outputs/figures`. Generated outputs and
`regenerated/` datasets are excluded from the immutable-input checksum manifest.
`outputs/reproduction_run.json` records the commands and omissions. Original
manuscript PNGs and tables are preserved under `reference/` for comparison;
copying them is not presented as scientific regeneration.

## Dependencies and tested scope

The saved-results workflow needs Python 3.10 or later, NumPy, and Matplotlib.
It was exercised with Python 3.10.18, NumPy 1.26.4, and Matplotlib 3.8.4. The
inventory checker itself uses only the standard library. These are the versions
used for this reproduction check, not a recovered training environment.

All three tables and all 23 figure assets were generated after moving the
release to another directory and blocking reads of the original manuscript and
`toy_model` trees. See `manifests/reproduction_checks.json`. Six Burgers models
also passed strict checkpoint loading and completed the confirmed visualization
rollouts. Inference for the other 21 selected models and a clean installation
test remain separate work. The original training/evaluation entry points retain
their original dependencies and historical checkpoint paths.

## Tables

`code/paper/make_results_metrics.py` gets the selected models and dataset paths
from the release manifests. It reads only the archived evaluations and data;
historical paths embedded in original summaries are not used as inputs.
`make_results_table.py` formats the metrics, while
`make_hyperparameter_table.py` formats the run-level and dataset-level settings.
The unmodified workspace scripts remain in `provenance/paper_scripts/`.

The resulting `results_table_metrics.tex` reproduces all 26 numerical rows of
the local manuscript table, including standard deviations, high-gradient error,
and conservation error. Final-time table values are unsmoothed. The generated
LaTeX can be included in the manuscript with `\input{...}`; caption wording and
minor formatting remain editorial choices.

The run-level table now reports **Epochs trained**, the final epoch recorded in
`history.json`, `history_checkpoint.json`, or the checkpoint's embedded training
history. **Best epoch** remains the validation-selected checkpoint. The new
`epochs_trained` field is exported alongside `epochs`, which retains the configured
maximum budget. Each row records its history source under `_sources.training_history`.
The formatter rejects older JSON without the new field; regenerate the metrics
JSON before formatting. Missing histories are never replaced with a maximum
budget or best epoch.

All 26 completed-epoch counts were checked against the archived histories and
agree between the workspace and portable release scripts. Apply the generated
run-level table to Table IV; this replaces the earlier instruction to correct
individual maximum-epoch cells. See `inventory/MANUSCRIPT_DATA_AUDIT.md` for the
remaining traffic and smoothing wording fixes.

## Saved evaluations and quantitative figures

`manifests/evaluations.json` maps every selected checkpoint to its primary
archive and associated files. The release contains 101 original numerical
evaluation archives (approximately 266 MB with their summaries), plus six newly
generated Burgers visualization archives and their provenance records.

- Burgers: seven `gnn_vs_godunov_eval_local_ssim.npz` archives. Every model's
  references match the same ten length-one test trajectories, with zero-based
  IDs `[1,4,7,14,31,52,58,70,82,95]`. The quantitative plots use these archives.
- Advection: seven corresponding local-SSIM archives, plus the leading
  FluxGraphNet run's `gnn_vs_godunov_eval.npz`, which also contains the WENO-3
  baseline. Its saved learned-model curves agree with the primary archive.
- Traffic: six all-test metric archives plus all 72 topology-specific prediction
  archives. The table and appendix plot cover 48 trajectories per model across
  three test splits. The appendix averages trajectories within topology,
  topologies within split, then the three splits equally. The main figure now
  uses the same 48 trajectories and weighting, as confirmed by the author on
  2026-10-13. The current manuscript reference PNG also uses these 48 trajectories;
  the old 16-trajectory reference is preserved in the history folder.
- Chemotaxis: six test-prediction archives and the saved comparison-metric
  archive, including finite-volume baseline predictions. The comparison archive
  records CFL subcycling, 15 test trajectories, and local mesh SSIM.
- Heat: the saved evaluation and summary for the selected control. Its plotted
  sample is trajectory 19 at time zero; the archive's 16 metric starts are not
  16 independent trajectories.

The curve recipes explicitly specify models, metrics, baseline arrays, time
axis, and smoothing. The Burgers curves use a nine-point display smoother;
linear advection uses no smoothing; traffic and chemotaxis use nine-point
smoothing. Tables always use unsmoothed values. The main comparison figures
show three learned models; the appendix curves show five, following their
legends. Additional models present in the table remain archived even when they
are absent from these curve figures.

## Rollout figures and schematic

Advection examples use trajectory 26 and 100 transitions. The saved learned
predictions and archived exact reference are used directly; the Godunov panel
is regenerated with the archived finite-volume implementation. Heat, traffic,
and chemotaxis snapshots use the explicitly listed frames in `figures.json`.
Rendering helpers in `code/paper/plot_helpers` retain the original source
function bodies and identify their source files in comments, except for the
heat snapshot x-axis label, corrected to Node Index to match the 10-13 draft. This avoids
requiring PyTorch merely to plot saved arrays.

The heat and traffic rollout PNGs, the chemotaxis rollout PNG, and both
chemotaxis error-curve PNGs reproduced with identical pixels in the tested
environment. Other figures may differ in layout, typography, colormap sampling,
or antialiasing; their scientific inputs and recipes are recorded. Historical
render commands were not recovered for every figure, so byte-identical rendering
is not claimed. The original reference figures are included for visual checking.

`architecture_standalone.tex` contains the original TikZ diagram in a standalone
article wrapper, with its manuscript caption removed. It compiled successfully
with the desktop editor's compiler. With an existing LaTeX installation:

```sh
mkdir -p outputs/figures
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=outputs/figures reproduction/architecture_standalone.tex
```

The unchanged manuscript figure source, including its caption, is available at
`reference/architecture_schematic.tex`.

## Regenerated Burgers visualization predictions

On 2026-10-13 the author confirmed zero-based trajectory index **1** (the second
trajectory) of the length-three bank. All six examples start at time zero
(pair ID 300), use 768 cells, and cover 300 transitions / 301 states. The
historical length-three prediction arrays were not recovered. New CPU rollouts
using the archived model code and checkpoints are stored separately as
`evaluations/Burgers/<method>/length3_visualization.npz`. Companion JSON files
record checkpoint/data hashes, versions, selection, and validation results;
`manifests/generated_evaluations.json` indexes them.

To rerun this inference in an environment containing the packages listed in
`requirements-burgers-inference.txt`:

```sh
python reproduction/generate_burgers_visualization.py
```

This writes fresh arrays and records under `outputs/burgers_visualization/`.
It does not overwrite the packaged inputs. The ordinary figure runner plots
the packaged arrays and requires only NumPy and Matplotlib.

The inference check used Python 3.11.13, NumPy 1.26.4, PyTorch 2.2.2, and PyTorch
Geometric 2.6.1 on CPU. Each model passed strict state-dict loading and was checked
against one archived length-one test trajectory before evaluating index 1 of the
length-three bank. Reference states agreed exactly; first-transition prediction
differences were at most 5.96e-8. Full 300-step predictions were not identical:
the largest absolute difference was 0.03269 for the absolute-edge MeshGraphNet,
whose relative L2 difference over the whole trajectory was 0.004666. The other
five maximum absolute differences ranged from 1.49e-6 to 0.001211. Full per-step
differences are recorded, without asserting recovery of the original inference
environment or bitwise reproducibility. All six new visualization trajectories
were finite and complete. The original quantitative archives remain unchanged.

No further figure-selection information is required. No manuscript files have
been modified.
