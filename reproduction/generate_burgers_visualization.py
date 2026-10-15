#!/usr/bin/env python3
"""Regenerate the confirmed length-three Burgers example from release checkpoints.

Uses the archived data loader, model builder, and rollout implementation. Each
model is first checked against one saved length-one quantitative trajectory.
Outputs are new evaluations, not recovered historical predictions.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/Burgers/src"))
sys.path.insert(0, str(ROOT / "code/Burgers/scripts"))

import numpy as np
import torch
import torch_geometric
from burgers_data import filter_burgers_dataset_by_regime, load_burgers_dataset
from compare_gnn_vs_godunov import _evaluate_pair_rollout, _load_data_source_from_cfg
from gnn_burgers import build_model_from_config, predict_type_from_target_mode
from input_features import count_input_feature_channels, resolve_input_feature_cfg, u_column_index
from physics_inputs import count_physics_input_channels, resolve_physics_input_cfg


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dataset(config, path, all_trajectories):
    data = config["data"]
    _, source_format, options = _load_data_source_from_cfg(data, override_data_dir=path)
    # Never follow historical split paths embedded in the checkpoint.
    options.pop("split_paths", None)
    if all_trajectories:
        options["split_paths"] = {"test": str(path)}
    result = load_burgers_dataset(path, split="test", include_rollout_states=True,
                                 dtype=torch.float32, source_format=source_format,
                                 source_options=options)
    return filter_burgers_dataset_by_regime(result, data.get("regime_filter", "all"))


def evaluate(model, data, trajectory_id, input_cfg, physics_cfg):
    local_id = data.trajectory_ids.tolist().index(trajectory_id)
    matches = ((data.pair_trajectory_id == local_id) & (data.pair_time_index == 0)).nonzero().flatten()
    if len(matches) != 1:
        raise ValueError(f"Expected one t=0 start for trajectory {trajectory_id}")
    pair_id = int(matches[0])
    horizon = int(data.rollout_states.shape[1] - 1)
    result = _evaluate_pair_rollout(dset=data, pid=pair_id, window=horizon, horizon=horizon,
        model=model, edge_index=data.edge_index.cpu().long(), edge_attr=data.edge_attr.cpu(),
        dt=float(data.metadata["dt"]), dx=float(data.metadata["dx"]), device=torch.device("cpu"),
        input_cfg=input_cfg, physics_cfg=physics_cfg)
    assert result[1] == 0 and result[2] == trajectory_id
    return pair_id, result[3], result[4], result[5]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", action="append", help="Run ID, for example Burgers/fluxgraphnet_with_advection")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/burgers_visualization")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    for protected in ("datasets", "models", "evaluations", "code", "provenance", "manifests", "reproduction"):
        if output == ROOT / protected or (ROOT / protected) in output.parents:
            parser.error("Output must be outside immutable release inputs.")
    jobs = [j for j in json.loads((ROOT / "reproduction/figures.json").read_text())["figures"]
            if j["renderer"] == "burgers_space"]
    if args.only:
        unknown = set(args.only) - {j["run_id"] for j in jobs}
        if unknown:
            parser.error("Unknown Burgers run: " + ", ".join(sorted(unknown)))
        jobs = [j for j in jobs if j["run_id"] in args.only]
    runs = {r["id"]: r for r in json.loads((ROOT / "manifests/runs.json").read_text())["runs"]}
    quantitative_data = ROOT / next(d["path"] for d in
        json.loads((ROOT / "manifests/datasets.json").read_text())["datasets"] if d["id"] == "Burgers")
    torch.set_num_threads(1)
    for job in jobs:
        run = runs[job["run_id"]]
        checkpoint = ROOT / run["checkpoint"]
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        config = payload["config"]
        config["model"] = dict(config["model"])
        config["model"]["predict_type"] = predict_type_from_target_mode(config["data"].get("target_mode", "delta"))
        data = dataset(config, quantitative_data, False)
        input_cfg = resolve_input_feature_cfg(config, source_format=data.metadata["source_format"])
        physics_cfg = resolve_physics_input_cfg(config)
        if config["model"].get("state_channel") is None:
            config["model"]["state_channel"] = u_column_index(input_cfg)
        model = build_model_from_config(config,
            in_channels=count_input_feature_channels(input_cfg) + count_physics_input_channels(physics_cfg),
            edge_attr_channels=int(data.edge_attr.shape[-1])).cpu()
        model.load_state_dict(payload["model_state_dict"], strict=True)
        with np.load(ROOT / run["evaluation_archive"], allow_pickle=False) as old:
            test_id = int(old["sample_global_traj_id"][0])
            _, truth, prediction, _ = evaluate(model, data, test_id, input_cfg, physics_cfg)
            np.testing.assert_array_equal(truth, old["sample_true_traj_by_sample"][0])
            saved = old["sample_gnn_traj_by_sample"][0]
            # A full autoregressive rollout can accumulate backend-dependent
            # roundoff. Check the first transition separately and report all
            # 300-step differences rather than claiming historical identity.
            np.testing.assert_allclose(prediction[1], saved[1], rtol=1e-4, atol=2e-5)
            difference = prediction - saved
            comparison = {"trajectory_id": test_id, "truth_max_abs_difference": 0.0,
                "prediction_max_abs_difference": float(np.max(np.abs(difference))),
                "prediction_rmse_difference": float(np.sqrt(np.mean(difference ** 2))),
                "prediction_relative_l2_difference": float(np.linalg.norm(difference) / np.linalg.norm(saved)),
                "first_transition_max_abs_difference": float(np.max(np.abs(difference[1]))),
                "max_abs_difference_by_step": np.max(np.abs(difference), axis=1).tolist(),
                "first_transition_rtol": 1e-4, "first_transition_atol": 2e-5,
                "first_transition_passed": True,
                "full_rollout_matches_at_first_transition_tolerance": bool(np.allclose(prediction, saved, rtol=1e-4, atol=2e-5))}
        del data
        visual_path = ROOT / job["dataset"]
        data = dataset(config, visual_path, True)
        pair_id, truth, prediction, godunov = evaluate(model, data, job["trajectory_id"], input_cfg, physics_cfg)
        for label, array in (("truth", truth), ("prediction", prediction), ("godunov", godunov)):
            if array.shape != (301, 768) or not np.isfinite(array).all():
                raise ValueError(f"Incomplete/nonfinite {label} for {run['id']}: {array.shape}")
        directory = output / run["id"]
        directory.mkdir(parents=True, exist_ok=True)
        archive = directory / "length3_visualization.npz"
        np.savez_compressed(archive, truth=truth, prediction=prediction, godunov=godunov,
            pos=data.pos.cpu().numpy().reshape(-1), dt=float(data.metadata["dt"]), dx=float(data.metadata["dx"]),
            trajectory_id=job["trajectory_id"], pair_id=pair_id, time_index=0)
        record = {"origin": "New CPU inference from archived checkpoint on author-confirmed trajectory",
            "run_id": run["id"], "checkpoint": run["checkpoint"], "checkpoint_sha256": sha256(checkpoint),
            "dataset": job["dataset"], "dataset_sha256": sha256(visual_path),
            "trajectory_id": job["trajectory_id"], "pair_id": pair_id, "time_index": 0,
            "shape": list(truth.shape), "strict_checkpoint_loading": True,
            "length_one_validation": comparison, "device": "cpu", "torch_threads": 1,
            "versions": {"python": platform.python_version(), "numpy": np.__version__,
                         "torch": torch.__version__, "torch_geometric": torch_geometric.__version__},
            "archive_sha256": sha256(archive)}
        archive.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
        print(f"Wrote {archive}; length-one maximum difference {comparison['prediction_max_abs_difference']:.3g}", flush=True)
        del data, model, payload


if __name__ == "__main__":
    main()
