# Research artifacts: downloads, placement, and verification

Run the commands below from the repository root. The verifier uses only Python's
standard library. No fresh environment installation or model execution is needed
to check file integrity.

## Availability

| Collection | Expected files | Availability | Needed for |
| --- | ---: | --- | --- |
| Datasets | 137 original files | [Zenodo DOI 10.5281/zenodo.23369294](https://doi.org/10.5281/zenodo.23369294) reserved; record not yet published | Saved-results reproduction, training, inference |
| Saved evaluations | 141 files: 107 numerical NPZ archives and 34 JSON records | Public download location not yet assigned | Saved-results reproduction; historical comparisons during Burgers inference |
| Model checkpoints | 27 PT files | Public download location not yet assigned | Fresh model inference |

The repository already includes run configurations, histories, split metadata,
and summaries under `models/`. Preserve those files when adding checkpoint
binaries. The saved-results workflow reads that exported metadata and does not
need the checkpoint binaries.

Exact artifact paths, byte counts, and SHA-256 hashes are in
[manifests/artifacts.json](manifests/artifacts.json). Download URLs remain unset
until public archives are available; the verifier does not attempt network
downloads or treat an unpublished DOI as a working download.

## 1. Check the source checkout

```sh
python tools/verify_inventory.py
```

This checks required repository files, Python syntax, JSON syntax, run selection,
portable data paths, and figure/evaluation manifest consistency. It also hashes
every expected research artifact that is present. Absent external artifacts are
reported separately and do not cause this default check to fail. A present but
corrupt artifact always causes failure, even in the default mode.

`manifests/source_files.json` lists the required repository files. Source checks
validate presence and syntax, not original-source byte identity. The historical
`manifests/copied_files.json` records provenance for the complete preparation
snapshot; it is not the source inventory for this checkout. Maintainers adding
or removing shipped files should update `source_files.json` accordingly.

## 2. Install the datasets

Once the dataset record is published, download all 11 deposit files to a folder
outside the repository. The five ZIP names are:

```text
heat_equation.zip
linear_advection.zip
burgers.zip
lwr_traffic.zip
chemotactic_cell_migration.zip
```

In the download folder, validate the deposit's `SHA256SUMS.txt` using
`shasum -a 256 -c SHA256SUMS.txt` on macOS or
`sha256sum -c SHA256SUMS.txt` on Linux. That checksum list covers the archives and
supporting deposit files; the repository verifier checks the extracted originals.

From the repository root, replace `/path/to/downloads` with the download folder:

```sh
for archive in heat_equation.zip linear_advection.zip burgers.zip lwr_traffic.zip chemotactic_cell_migration.zip; do
  unzip "/path/to/downloads/$archive" 'datasets/*' -d .
done
python tools/verify_inventory.py --require datasets
```

Extract only `datasets/*`. The ZIPs include top-level dataset documentation that
would otherwise overwrite the repository README or citation files. Do not place
the ZIPs inside `datasets/` or introduce an extra nested `datasets/datasets/` level.

The two Burgers PT files belong in `datasets/Burgers/`: the length-one training
and quantitative bank and the length-three visualization bank are both required
for the complete set of figures.

## 3. Install saved evaluations and checkpoints

These collections are not yet publicly downloadable. Once their archives are
published, extract their `evaluations/` and `models/` entries at the repository
root, preserving the exact paths in the artifact manifest. Examples:

```text
evaluations/Burgers/fluxgraphnet_with_advection/gnn_vs_godunov_eval_local_ssim.npz
evaluations/Burgers/fluxgraphnet_with_advection/length3_visualization.npz
models/Burgers/fluxgraphnet_with_advection/checkpoint_best.pt
```

Evaluation JSON summaries and provenance records must accompany their NPZ
archives. Existing model JSON configurations and histories remain in Git; the
checkpoint download should add binaries alongside them.

After adding datasets and evaluations:

```sh
python tools/verify_inventory.py --require saved-results
python reproduction/reproduce.py all
```

The first command validates all 137 dataset files and 141 evaluation files.
The second requires the documented NumPy/Matplotlib environment and generates
the tables and figures. Its automatic preflight performs the same strict
artifact check before launching any jobs. Selecting an individual figure with
`--only` still uses this complete saved-results preflight.

After adding checkpoints as well:

```sh
python tools/verify_inventory.py --require all
```

This validates all 305 external artifact files. It does not assert successful
model inference or fresh installation of the training environment.

## Existing local archival package

If the complete archival package is already available on your machine, you can
check its artifacts against this checkout without copying them:

```sh
python tools/verify_inventory.py --require all --artifact-root /path/to/complete/archive
```

`--artifact-root` changes only where the verifier reads external research
artifacts. Source checks still use this repository. It does not redirect
reproduction scripts or make a source-only checkout ready to run them.

To install the existing artifacts locally, replace `/path/to/complete/archive`
below and run from the repository root:

```sh
mkdir -p datasets evaluations models
rsync -av /path/to/complete/archive/datasets/ datasets/
rsync -av /path/to/complete/archive/evaluations/ evaluations/
rsync -av --include='*/' --include='*.pt' --include='*.pth' --exclude='*' \
  /path/to/complete/archive/models/ models/
python tools/verify_inventory.py --require all
```

Downloaded datasets, evaluations, and checkpoint binaries are ignored by Git.
The transfers above add or update those artifacts without deleting destination
files or replacing the repository's source/documentation.

## Interpreting failures

- **Missing repository file:** restore the file from the source checkout.
- **Missing research artifact:** download/extract the indicated collection into
  the expected directory. One example path per incomplete group is printed.
- **Checksum/size mismatch:** the file differs from the archived research input.
  Check the download or extraction; do not replace expected hashes just to make
  the check pass. Put newly generated results under `outputs/` or `regenerated/`.
- **Manifest inconsistency:** run, dataset, or figure metadata disagree. This
  requires correcting the repository metadata, not a package installation.

Strict modes exit with status 1 when required artifacts are missing. All modes
exit with status 1 for malformed source metadata, invalid Python syntax, or
corrupt expected artifacts. New training outputs and unlisted files are not
certified by the verifier.
