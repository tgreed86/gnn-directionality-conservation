#!/usr/bin/env python3
"""Portable entry point for the archived paper tables and figures."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=["list", "tables", "figures", "all"])
    parser.add_argument("--only", action="append", help="Figure recipe ID; repeat to select several.")
    parser.add_argument("--skip-unresolved", action="store_true",
                        help="Explicitly run only established recipes and report the omissions.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    jobs = json.loads((ROOT / "reproduction/figures.json").read_text())["figures"]
    if args.only:
        unknown = set(args.only) - {j["id"] for j in jobs}
        if unknown:
            parser.error("Unknown recipe(s): " + ", ".join(sorted(unknown)))
        jobs = [j for j in jobs if j["id"] in args.only]
    if args.task == "list":
        for job in jobs:
            print(job["status"].ljust(19), job["id"])
            if job["status"] != "ready":
                print("  " + job["reason"])
        print("Architecture schematic: reproduction/reference/architecture_schematic.tex (LaTeX source).")
        return
    pending = [j for j in jobs if j["status"] != "ready"] if args.task in ("all", "figures") else []
    if pending and not args.skip_unresolved:
        parser.error("Unresolved scientific selections: " + ", ".join(j["id"] for j in pending) +
                     ". Run 'list' for details; --skip-unresolved explicitly permits a partial run.")
    # Fail with artifact-installation guidance before launching any plotting jobs.
    verified = subprocess.run([
        sys.executable, str(ROOT / "tools/verify_inventory.py"),
        "--require", "saved-results",
    ])
    if verified.returncode:
        parser.exit(verified.returncode,
                    "Research inputs are incomplete or invalid. See ARTIFACTS.md.\n")
    output = args.output_dir.resolve()
    # Guard against accidental writes into the immutable archive directories.
    for protected in ("datasets", "models", "evaluations", "code", "provenance", "manifests", "reproduction"):
        if output == ROOT / protected or (ROOT / protected) in output.parents:
            parser.error("Output directory must not be inside archived inputs: " + protected)
    env = dict(os.environ)
    cache = Path(tempfile.gettempdir()) / "paper-release-cache"
    env.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    env.setdefault("XDG_CACHE_HOME", str(cache))
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    commands = []
    scripts = ROOT / "code/paper"
    if args.task in ("tables", "all"):
        tables = output / "tables"
        commands.extend([
            [str(scripts / "make_results_metrics.py"), "--output-json", str(tables / "results_table_metrics.json"),
             "--output-csv", str(tables / "results_table_metrics.csv")],
            [str(scripts / "make_results_table.py"), "--input", str(tables / "results_table_metrics.json"),
             "--output", str(tables / "results_table_metrics.tex"), "--sort-by", "rel_l2"],
            [str(scripts / "make_hyperparameter_table.py"), "--input", str(tables / "results_table_metrics.json"),
             "--output", str(tables / "model_training_hyperparameters.tex"),
             "--dataset-output", str(tables / "model_training_hyperparameters_by_dataset.tex")],
        ])
    if args.task in ("figures", "all"):
        commands.extend([str(scripts / "plot_saved_results.py"), j["id"], "--output-dir", str(output / "figures")]
                        for j in jobs if j["status"] == "ready")
    completed = []
    for command in commands:
        subprocess.run([sys.executable, *command], cwd=ROOT, env=env, check=True)
        completed.append(command)
    output.mkdir(parents=True, exist_ok=True)
    (output / "reproduction_run.json").write_text(json.dumps({
        "completed_commands": completed, "skipped": pending,
        "model_inference_performed": False,
        "status": "partial: unresolved figure selections" if pending else "selected tasks completed",
    }, indent=2) + "\n")
    for job in pending:
        print("NOT REPRODUCED:", job["id"], "—", job["reason"])


if __name__ == "__main__":
    main()
