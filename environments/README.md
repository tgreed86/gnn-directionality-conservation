# Computational environments

This repository distinguishes the author's local environment, the reported HPC
container environment, and the environments used to check the archival
reproduction workflows. Their Python and package versions differ. None of the
records below establishes a complete per-run history of package versions.

## Observed versions

| Component | Author's local environment | Reported HPC environment | Saved-results check | Burgers inference check |
| --- | --- | --- | --- | --- |
| Python | 3.11.13 | 3.12.3 | 3.10.18 | 3.11.13 |
| Platform | macOS arm64 | Linux x86_64 | Local CPU workflow | Local CPU workflow |
| PyTorch | 2.7.1 | 2.7.0a0+79aa17489c.nv25.4 | Not required | 2.2.2 |
| PyTorch CUDA build | Not applicable to this macOS installation | 12.9 | Not required | CPU execution |
| PyTorch Geometric | 2.6.1 | 2.6.1 | Not required | 2.6.1 |
| NumPy | 2.3.3 | 1.26.4 | 1.26.4 | 1.26.4 |
| Matplotlib | 3.10.6 | 3.10.1 | 3.8.4 | 3.8.4 |
| SciPy | 1.16.2 | 1.15.2 | Not required | Not specified |
| PyYAML | 6.0.2 | 6.0.2 | Not required | 6.0.2 |
| tqdm | 4.67.1 | 4.67.1 | Not required | Not specified |

The local `amr-gnn-pt27` versions were read from installed distribution metadata
on 2026-10-15. The author supplied the HPC report on that date. These observations
do not prove that either environment remained unchanged throughout training.

## Saved figures and tables

Use Python 3.10 with the dependencies in
[requirements-saved-results.txt](requirements-saved-results.txt). From the
repository root:

```sh
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r environments/requirements-saved-results.txt
```

The complete archival workflow was exercised with Python 3.10.18 and these
versions: three tables and 23 PNG assets were generated after relocation with
access to the original source directories blocked. This check used archived
data and saved predictions, not retraining. A fresh installation and public
download test of the separated GitHub repository have not been performed.

## Author's local environment

[requirements-local-core.txt](requirements-local-core.txt) records the seven
direct dependencies used across the model, generation, and plotting code,
as observed in the author's macOS environment. To prepare a corresponding
environment on a compatible system, from the repository root:

```sh
python3.11 -m venv .venv-local
source .venv-local/bin/activate
python -m pip install -r environments/requirements-local-core.txt
```

This specifies core package versions. It does not pin every transitive
dependency or reproduce the original Conda builds. A fresh installation and
all-model training/inference check with these pins have not been performed.
Do not treat this file as a reconstruction of the HPC container.

## HPC container and inherited packages

[hpc-runtime.txt](hpc-runtime.txt) transcribes the runtime report supplied by the
author, with whitespace normalized. [hpc-freeze.txt](hpc-freeze.txt) preserves
the supplied `pip freeze --all` output byte-for-byte. The export contains 333
package entries, 84 of which reference local `file://` locations. The source
hash and comparison results are recorded in
[environment-records.json](environment-records.json).

The environment `shock-ramp-hpc-env` was created with
`include-system-site-packages = true`. It inherits packages from the container:
the report places PyTorch under `/usr/local/lib/python3.12/dist-packages/torch/`,
outside the virtual environment. Its actual reported version is
`2.7.0a0+79aa17489c.nv25.4`, not the standard `2.7.0` release or the local `2.7.1`
installation. CUDA 12.9 is the PyTorch build version; the report does not identify
the host NVIDIA driver, GPU model, or runtime GPU availability.

**The full freeze is a provenance record, not a portable installation file.**
Entries refer to wheels under container paths such as `/rapids/` and
`/opt/transfer/`. Do not run `pip install -r environments/hpc-freeze.txt` as a
generic setup command or silently substitute a standard PyTorch wheel for the
reported vendor-tagged build. Many exported packages belong to the wider
container stack rather than the paper's direct dependencies.

The supplied historical SLURM script selected `pytorch-2.7.0.sif` and then
activated the virtual environment inside Apptainer. On 2026-10-15 the author
confirmed that the original image had been relocated and that this report was
generated using that original image. Its new path and image digest have not
been supplied; the image identity is author-confirmed rather than independently
verified by hash. No replacement image or download URL is inferred. A reusable
HPC setup recipe still requires access to this image; the package export alone
cannot reconstruct the container. The current report does not establish that
virtual-environment packages were unchanged throughout training.

The report lists `torch-scatter`, `torch-sparse`, `torch-cluster`, and `pyg-lib`
as not found. This records the supplied environment rather than adding packages
that were not reported as installed.

## Checked Burgers inference environment

The separate
[Burgers inference requirements](../reproduction/requirements-burgers-inference.txt)
record the Python 3.11.13 / PyTorch 2.2.2 CPU environment used for six archived
visualization rollouts. Use the command in the main README to install it.
This is a reproduction-check environment, not the author's original HPC stack.
Full-rollout differences from historical predictions are documented in the
[reproduction guide](../reproduction/README.md); their cause is not established
by the version differences alone.

## Validation scope

The reported core HPC versions agree with the supplied freeze, including the
versions encoded in its PyTorch and PyTorch Geometric wheel references. No HPC
execution, container rebuild, fresh package installation, or model retraining
was performed as part of documenting these reports. The previously completed
reproduction checks retain their original environment provenance.
