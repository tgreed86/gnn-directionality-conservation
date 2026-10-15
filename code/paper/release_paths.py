"""Resolve scientific inputs solely through the portable release manifests."""
import json
from pathlib import Path

RELEASE_ROOT = Path(__file__).resolve().parents[2]


def release_runs(root=RELEASE_ROOT):
    return json.loads((Path(root) / "manifests/runs.json").read_text())["runs"]


def dataset_path(root, name):
    aliases = {"Inviscid Burgers": "Burgers", "Linear Advection": "1D_advection",
               "LWR Traffic": "1D_traffic", "Chemotactic Cell Migration": "chemotactic_cell_migration",
               "Heat Equation": "heat_equation"}
    entries = json.loads((Path(root) / "manifests/datasets.json").read_text())["datasets"]
    return Path(root) / next(d["path"] for d in entries if d["id"] == aliases[name])


def evaluation_paths(root, model_directory):
    root = Path(root).resolve()
    directory = (root / model_directory).resolve()
    run = next(r for r in release_runs(root) if (root / r["checkpoint"]).parent == directory)
    archive = root / run["evaluation_archive"]
    summary_name = {"Burgers": "gnn_vs_godunov_summary.json",
                    "1D_advection": "gnn_vs_godunov_summary.json",
                    "1D_traffic": "summary.json",
                    "chemotactic_cell_migration": "summary_metrics.json",
                    "heat_equation": "rollout_eval_summary.json"}[run["dataset_id"]]
    return directory, archive, archive.parent / summary_name
