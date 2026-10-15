#!/usr/bin/env python3
"""Validate source manifests and separately downloaded research artifacts.

Uses only Python's standard library; does not import model code or load pickles.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {
    'available': set(),
    'datasets': {'datasets'},
    'saved-results': {'datasets', 'evaluations'},
    'all': {'datasets', 'evaluations', 'checkpoints'},
}


def check(condition, message):
    if not condition:
        raise ValueError(message)


def relative_path(value):
    check(isinstance(value, str) and bool(value), 'Empty or invalid manifest path')
    p = PurePosixPath(value)
    check(not p.is_absolute() and '..' not in p.parts and '\\' not in value
          and p.as_posix() == value and value != '.', f'Unsafe manifest path: {value}')
    return value


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def read_json(root, relative):
    return json.loads((root / relative_path(relative)).read_text())


def artifact_index(manifest):
    check(manifest['schema_version'] == 1, 'Unsupported artifact manifest version')
    check(set(manifest['groups']) == REQUIRED['all'], 'Unexpected artifact groups')
    index = {}
    for group, entry in manifest['groups'].items():
        for record in entry['files']:
            path = relative_path(record['path'])
            check(path not in index, f'Duplicate artifact: {path}')
            prefix = 'models/' if group == 'checkpoints' else group + '/'
            check(path.startswith(prefix), f'Artifact in wrong group: {path}')
            check(isinstance(record['bytes'], int) and record['bytes'] >= 0,
                  f'Invalid byte count: {path}')
            check(re.fullmatch('[0-9a-f]{64}', record['sha256']) is not None,
                  f'Invalid SHA-256: {path}')
            index[path] = record
    return index


def verify_sources(root, artifacts):
    inventory = read_json(root, 'manifests/source_files.json')
    check(inventory['schema_version'] == 1, 'Unsupported source inventory version')
    sources = inventory['files']
    check(len(sources) == len(set(sources)), 'Duplicate source inventory entries')
    source_set = set(sources)
    check(not source_set.intersection(artifacts), 'Source/artifact inventories overlap')
    python_count = 0
    for name in sources:
        path = root / relative_path(name)
        check(path.is_file(), f'Missing repository file: {name}')
        if path.suffix == '.py':
            ast.parse(path.read_text(), filename=name)
            python_count += 1
        elif path.suffix == '.json':
            json.loads(path.read_text())

    def source(name):
        check(name in source_set, f'Unlisted repository dependency: {name}')

    def artifact(name):
        check(name in artifacts, f'Unlisted downloadable dependency: {name}')

    manifest = read_json(root, 'manifests/runs.json')
    runs = manifest['runs']
    source(manifest['selection_source'])
    source(manifest['additional_selection_source'])
    tree = ast.parse((root / manifest['selection_source']).read_text())
    selection = next(ast.literal_eval(n.value) for n in tree.body
                     if isinstance(n, ast.Assign) and any(
                         isinstance(t, ast.Name) and t.id == 'RUNS' for t in n.targets))
    transport = [r for r in runs if r['runs_index'] is not None]
    check([(r['dataset'], r['method'], r['original_run_directory']) for r in transport]
          == selection, 'Selected runs differ from original RUNS')
    check([r['runs_index'] for r in transport] == list(range(1, len(transport) + 1)),
          'Invalid RUNS ordering')
    additional = read_json(root, manifest['additional_selection_source'])['runs']
    extras = [r for r in runs if r['runs_index'] is None]
    check(len(extras) == len(additional), 'Additional run count differs')
    for run, selected in zip(extras, additional):
        check(all(run[k] == selected[k] for k in
                  ('id', 'dataset', 'method', 'original_run_directory')),
              'Additional run selection differs')
        check(Path(run['checkpoint']).name == selected['checkpoint_filename'],
              'Additional checkpoint selection differs')
    run_ids = {r['id'] for r in runs}
    check(len(run_ids) == len(runs), 'Duplicate run IDs')
    datasets = read_json(root, 'manifests/datasets.json')['datasets']
    by_dataset = {d['id']: d for d in datasets}
    check(len(by_dataset) == len(datasets), 'Duplicate dataset IDs')
    dataset_paths = set()
    for dataset in datasets:
        source(dataset['generator'])
        for name in dataset['files']:
            artifact(name)
            dataset_paths.add(name)
        check(sum(artifacts[n]['bytes'] for n in dataset['files']) == dataset['bytes'],
              f'Dataset size metadata differs: {dataset["id"]}')
    check(dataset_paths == {n for n in artifacts if n.startswith('datasets/')},
          'Dataset artifact coverage differs')
    checkpoint_paths = set()
    for run in runs:
        name = run['checkpoint']
        artifact(name)
        checkpoint_paths.add(name)
        check(artifacts[name]['bytes'] == run['checkpoint_bytes'] and
              artifacts[name]['sha256'] == run['checkpoint_sha256'],
              f'Checkpoint metadata differs: {name}')
        for key in ('model_source', 'training_entry_point', 'configuration', 'release_configuration'):
            source(run[key])
        for name in run['artifacts']:
            artifact(name) if name in artifacts else source(name)
        cfg = read_json(root, run['release_configuration'])
        data = cfg['data']
        data_path = data.get('data_dir', data.get('dataset_path')) if isinstance(data, dict) else data
        check(data_path == by_dataset[run['dataset_id']]['path'],
              f'Portable dataset path differs: {run["id"]}')
        artifact(run['evaluation_archive'])
    check(checkpoint_paths == {n for n in artifacts if n.startswith('models/')},
          'Checkpoint artifact coverage differs')
    evaluations = read_json(root, 'manifests/evaluations.json')['runs']
    check(len(evaluations) == len(runs) and {r['run_id'] for r in evaluations} == run_ids,
          'Evaluation run coverage differs')
    evaluation_paths = set()
    for record in evaluations:
        check(record['primary_archive'] in record['files'], 'Missing primary evaluation entry')
        for name in record['files']:
            artifact(name)
            evaluation_paths.add(name)
    check(evaluation_paths == {n for n in artifacts if n.startswith('evaluations/')},
          'Evaluation artifact coverage differs')
    generated = read_json(root, 'manifests/generated_evaluations.json')
    source(generated['generator'])
    for record in generated['evaluations']:
        for key, hash_key in (('checkpoint', 'checkpoint_sha256'),
                              ('dataset', 'dataset_sha256'), ('archive', 'archive_sha256')):
            artifact(record[key])
            check(artifacts[record[key]]['sha256'] == record[hash_key],
                  f'Generated evaluation provenance differs: {record[key]}')
        check(record['trajectory_id'] == 1 and record['time_index'] == 0,
              'Burgers visualization selection differs')
    recipes = read_json(root, 'reproduction/figures.json')['figures']
    figures = read_json(root, 'manifests/manuscript_figures.json')['figures']
    check({r['asset'] for r in recipes} == {Path(f['asset']).name for f in figures},
          'Figure coverage differs')
    check(len({r['id'] for r in recipes}) == len(recipes), 'Duplicate figure IDs')
    for recipe in recipes:
        source(recipe['reference'])
        if 'run_id' in recipe:
            check(recipe['run_id'] in run_ids, 'Unknown figure run')
        for key in ('archive', 'dataset'):
            if key in recipe:
                artifact(recipe[key])
        for curve in recipe.get('curves', []):
            artifact(curve['archive'])
        check(recipe['status'] in ('ready', 'needs_information'), 'Invalid figure status')
        if recipe['status'] != 'ready':
            check(bool(recipe.get('reason')), 'Unexplained unresolved figure')
    return len(sources), python_count, len(runs), len(recipes)


def verify_artifacts(root, manifest):
    results = {}
    errors = []
    for group, entry in manifest['groups'].items():
        missing, verified = [], 0
        for record in entry['files']:
            path = root / record['path']
            if not path.exists():
                if path.is_symlink():
                    errors.append(f'Broken artifact symlink: {record["path"]}')
                else:
                    missing.append(record['path'])
                continue
            if not path.is_file():
                errors.append(f'Not an artifact file: {record["path"]}')
            elif path.stat().st_size != record['bytes'] or digest(path) != record['sha256']:
                errors.append(f'Artifact checksum/size mismatch: {record["path"]}')
            else:
                verified += 1
        results[group] = {'verified': verified, 'missing': missing, 'total': len(entry['files'])}
    return results, errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT, help='Repository root (default: this checkout).')
    parser.add_argument('--artifact-root', type=Path,
                        help='Verify artifacts in another root; does not relocate reproduction inputs.')
    parser.add_argument('--require', choices=REQUIRED, default='available',
                        help='Fail on missing inputs for this scope; default checks all present artifacts.')
    args = parser.parse_args(argv)
    root = args.root.resolve()
    artifact_root = args.artifact_root.resolve() if args.artifact_root else root
    manifest = read_json(root, 'manifests/artifacts.json')
    artifacts = artifact_index(manifest)
    count, python_count, run_count, figure_count = verify_sources(root, artifacts)
    print(f'PASS: repository structure, {count} files, {python_count} Python syntax checks; '
          f'{run_count} run mappings and {figure_count} figure recipes.')
    results, errors = verify_artifacts(artifact_root, manifest)
    for group, record in results.items():
        print(f'{group}: {record["verified"]}/{record["total"]} verified, '
              f'{len(record["missing"])} missing.')
        if record['missing']:
            entry = manifest['groups'][group]
            print(f'  Availability: {entry["availability"]}')
            print('  Example missing file: ' + record['missing'][0])
            if group in REQUIRED[args.require]:
                errors.append(f'Required {group} incomplete: {len(record["missing"])} missing files.')
    for error in errors:
        print('FAIL: ' + error, file=sys.stderr)
    if errors:
        print('See ARTIFACTS.md for download/extraction instructions.', file=sys.stderr)
        return 1
    if any(r['missing'] for r in results.values()):
        print(f'PASS for scope {args.require!r}; omitted artifacts are not verified or declared ready.')
    else:
        print(f'PASS: all {sum(r["total"] for r in results.values())} expected '
              'research artifact files verified by SHA-256.')
    print('This checks files and metadata, not installation, model execution, or scientific results.')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, StopIteration, SyntaxError, TypeError) as exc:
        print(f'FAIL: {exc}', file=sys.stderr)
        sys.exit(1)
