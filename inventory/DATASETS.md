# Dataset inventory and generation provenance

The five user-selected datasets are archived unchanged under `datasets/`: 137 files,
319,000,935 bytes, including the additional length-3 Burgers visualization bank. `manifests/datasets.json` maps their original and
release locations. Original-file and whole-snapshot SHA-256 checks cover them.
The supplied chemotaxis generator was found in the project's `scripts/` directory.

| Dataset | Trajectories | Stored states per trajectory | Spatial representation | Saved interval |
|---|---:|---:|---|---|
| Heat | 100 | 131 | 64 periodic nodes, length 1 | 0.0048828125 |
| Linear advection | 100 | 129 | 384 periodic cell averages, length 3 | 0.00234375 |
| Burgers | 100 | 301 | 256 periodic approximate cell averages, length 1 | 0.0015 |
| Burgers visualization | 10 | 301 | 768 periodic approximate cell averages, length 3 | 0.0015 |
| LWR traffic | 192 on 32 topologies | 257 | Road finite-volume cells, varying geometry | 0.0330898765–0.0831493181 by topology |
| Chemotaxis | 100 | 129 | 512 triangular cells on each of 100 distinct meshes | 0.010339876433737077 |

## Evidence linking the data to the results

`manifests/dataset_audit.json` records the following comparisons against the original
saved evaluation archives for the 27 selected runs. This checks references and
initial conditions; it does not rerun trained-model inference.

- **Burgers:** all 10 complete test-reference trajectories in each of seven runs
  match the supplied dataset exactly. Test global IDs are
  `[1,4,7,14,31,52,58,70,82,95]`. In these metric archives, sample ID 300 is a flattened test-pair
  index selecting global trajectory 4. This mapping must not be assumed for
  the separate length-3 visualization bank.
- **Linear advection:** the full representative reference saved by each of seven
  runs matches exactly; all ten saved prediction initial states also match the
  corresponding dataset initial states. Test IDs are
  `[19,26,40,56,66,81,84,86,89,99]`. The paper's example is trajectory 26.
  These archives do not save all ten full reference trajectories, so their
  unsaved reference histories were not independently compared.
- **Traffic:** all four trajectories on each of the 12 test topologies match
  exactly for all six runs: 48 test trajectories per model. The paper's network
  example is topology 26, trajectory 0, in `test_large`.
- **Chemotaxis:** all 15 complete test-reference trajectories match exactly in
  all six runs, despite the historical dataset path differing from today's
  supplied path. The paper's example is trajectory 48. The 100 archived vertex
  arrays have 100 distinct hashes, confirming different meshes by trajectory.
- **Heat:** the plotted initial state is trajectory 19, time zero (flattened
  pair ID 2470). Its saved rollout reference differs from the stored dataset by
  at most 2.384185791015625e-7. The evaluator evolves from the float32 initial
  state and casts after every spectral step; dataset generation evolves in
  float64 before storage. Replaying the evaluator's calculation reproduces its
  reference exactly. The heat evaluation's 16 sample IDs are consecutive pair
  starts in trajectory 19; they must not be described as 16 independent test
  trajectories. The paper currently uses a single illustrative heat rollout.

All stored states inspected are finite. Saved split-index sets are disjoint.
The split counts are 80/10/10 for heat, advection and Burgers; 70/15/15 for
chemotaxis; and 16/4/4/4/4 topologies for traffic training, validation, in-size,
large-network, and parameter/structure-OOD testing.

The user clarified on 2026-10-13 that the Burgers space-time figures use a
separate length-3 bank. Its file and metadata are now included; the earlier
axis-error diagnosis has been withdrawn. The saved metric references checked
above still correspond to the original length-1 holdout bank.

## Reconstructed generation settings

[Generation commands](../generation/README.md) and `generation/commands.json`
give explicit recipes. They write to `regenerated/`, preserving `datasets/`.
No original shell history is claimed to have been recovered.

- **Heat:** seed 1234, diffusivity 0.01, diffusion sampling factor 0.2, modes
  1–8, Gaussian coefficients with amplitude 1 and decay exponent 1; no RMS
  normalization. The saved target is a rate. Replaying the copied numerical
  functions reproduces every stored float32 rollout state exactly.
- **Advection:** seed 1234, speed 1, CFL 0.3, analytic translations, exact cell
  averages, 6–12 piecewise-constant segments/jumps, amplitude 1, no RMS
  normalization. The saved target is the next state; model training uses
  increments derived from the states. Replaying the copied numerical functions
  reproduces every stored float32 rollout state exactly.
- **Burgers:** the archive records Nx=256, length 1, 300 transitions, dt=0.0015,
  CFL=0.2, quadrature count 8, and initial maximum speed 0.5208333333. It does
  **not** record the IC configuration, RNG seed/state, or historical source
  revision. The supplied current generator does not seed the RNG in its CLI.
  `generation/burgers_reconstructed.yaml` explicitly selects the current
  top-hat defaults (2–6 hats, widths 0.05–0.25, amplitudes −1.5–1.5); those
  defaults agree with the appendix but their historical use is not proved by
  the saved metadata. An independently regenerated bank will differ. The
  archived dataset is the exact reference for reproducing the paper.
- **Traffic:** `manifest.json` records the complete generation configuration,
  including seed 1234 and topology split IDs. All 32 topologies were regenerated
  from it. Stored float32 states and transfer fluxes match exactly, as do the
  compared positions, edges, and cell lengths. A float64 speed differs by at
  most 2.22e-16; generation diagnostics can vary at roundoff across libraries.
- **Chemotaxis:** seed 1234, separate meshes and static landscapes, D=0.01,
  chi=0.1, 17×17 vertices, jitter 0.30, one refinement level, coarse/reference
  stability factors 0.45/0.80, and four nominal coarse steps per saved frame.
  The reference actually uses 13 substeps per frame. Trajectories 0 and 48 were
  regenerated using the recorded common solver step. Their float32 states
  match exactly after matching triangle IDs by their vertices. The available
  numerical libraries reorder the Delaunay triangles for trajectory 48 and
  change vertex values by at most 1.11e-16. Thus the archive's original cell
  ordering must be preserved when using saved predictions. The complete
  100-trajectory generation was not replayed.

Detailed replay outcomes are in `manifests/generator_replay_checks.json` and
`manifests/dataset_audit.json`. These checks establish numerical consistency of
the tested generator snapshots; they do not establish an original environment
lockfile or historical code revision.

## Using the archived data with the models

Each model now has `config_release.json`, also indexed in `manifests/runs.json`.
These derived copies replace the data and output paths with paths relative to
the `paper_release/` root. Run scripts from that root. Training outputs go to
`outputs/<run-id>/`; scientific settings and original device selections are
retained. Choose an available device when testing locally. Original checkpoints
and `config_checkpoint.json` are unchanged; checkpoint-based evaluation scripts
still need their dataset-path override because the checkpoint embeds its
historical configuration.

The generator dependency snapshots include the Burgers `weno_a` equation/IC
modules and the traffic network-flux helper. The previously deferred chemotaxis
plotting entry points are now included. Generation adds SciPy (Delaunay), PyYAML,
and tqdm to the dependency inventory; no validated installation or pinned
training environment is claimed. Full trained-model inference remains pending.

See [the manuscript audit](MANUSCRIPT_DATA_AUDIT.md) for required figure and
text corrections. Dataset identity alone does not validate plot axes or which
test subset was averaged.
