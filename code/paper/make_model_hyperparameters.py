#!/usr/bin/env python
"""Portable release variant of make_model_hyperparameters.py.

Uses archived inputs under paper_release and writes to outputs/tables by default.
The unchanged workspace original is in provenance/paper_scripts/.
See reproduction/README.md for the complete workflow.
"""

from __future__ import print_function

import argparse
import csv
import io
import json
import math
import pickle
import re
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from release_paths import RELEASE_ROOT, dataset_path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

DEFAULT_BASE_DIR = RELEASE_ROOT
DEFAULT_OUTPUT_JSON = RELEASE_ROOT / "outputs/tables/model_training_hyperparameters.json"
DEFAULT_OUTPUT_CSV = RELEASE_ROOT / "outputs/tables/model_training_hyperparameters.csv"


COLUMNS = [
    "dataset",
    "method",
    "run_dir",
    "config_source",
    "summary_json",
    "dataset_path",
    "source_format",
    "num_trajectories",
    "num_trajectories_train",
    "num_trajectories_val",
    "num_trajectories_test",
    "num_pairs",
    "train_pairs",
    "val_pairs",
    "test_pairs",
    "num_nodes",
    "time_steps_per_trajectory",
    "snapshots_per_trajectory",
    "dt",
    "dx",
    "cfl",
    "wavespeed",
    "target_mode",
    "training_mode",
    "rollout_steps",
    "epochs",  # Configured maximum; retained in machine-readable exports.
    "epochs_trained",
    "best_epoch",
    "early_stopping_metric",
    "best_val_loss",
    "batch_size",
    "optimizer",
    "lr_initial",
    "lr_final",
    "lr_schedule_enabled",
    "lr_schedule",
    "weight_decay",
    "grad_clip",
    "early_stopping_patience",
    "model_type",
    "hidden_dim",
    "message_passing_layers",
    "dropout",
    "activation",
    "use_skip",
    "use_layernorm",
    "predict_type",
    "state_residual_mode",
    "model_in_channels",
    "input_features",
    "normalize_state_to_minus1_1",
    "normalize_state_clip",
    "physics_enabled",
    "include_adv",
    "advection_scheme",
    "physics_input_form",
    "adv_all_steps",
    "normalize_adv_to_minus1_1",
    "normalize_adv_clip",
    "tv_loss_enabled",
    "tv_loss_lambda",
    "tv_loss_mode",
    "gradient_weighted_loss_enabled",
    "gradient_weighted_loss_strength",
    "relative_mse_enabled",
    "relative_mse_alpha",
    "mass_conservation_loss_enabled",
    "mass_conservation_loss_lambda",
    "flux_loss_weight",
    "node_weight_loss",
    "mass_projection",
    "include_absolute_positions",
    "include_boundary_distances",
    "include_drift_velocity",
    "include_chi",
    "feature_names",
    "normalization",
    "feature_normalization",
    "device",
    "seed",
    "notes",
]


class _DummyStorage(object):
    pass


class _SkipTensorUnpickler(pickle.Unpickler):
    """Unpickle checkpoint metadata while replacing tensors with placeholders."""

    def persistent_load(self, pid):
        return ("PERSISTENT", pid)

    def find_class(self, module, name):
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return lambda *args, **kwargs: ("TENSOR",)
        if module == "torch" and name.endswith("Storage"):
            return _DummyStorage
        return super(_SkipTensorUnpickler, self).find_class(module, name)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract model and training hyperparameters from saved runs."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help="Base directory prepended to the configured run paths.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help="Output JSON path.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--omit-raw-config",
        action="store_true",
        help="Do not embed the raw recovered config in the JSON output.",
    )
    return parser.parse_args()


def iter_run_entries(runs):
    for run in runs:
        if len(run) < 3:
            raise ValueError("Each RUNS entry must have at least dataset, method, and path.")
        dataset_name, method = run[0], run[1]
        relative_run_dir = Path(str(run[2]))
        for path_part in run[3:]:
            relative_run_dir = relative_run_dir / str(path_part)
        yield dataset_name, method, relative_run_dir


def load_json(path):
    if not path.exists():
        return {}
    with path.open("r") as fh:
        return json.load(fh)


def to_builtin(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return value.item()
        return [to_builtin(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    return value


def clean_value(value):
    value = to_builtin(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def csv_value(value):
    value = clean_value(value)
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def get_nested(mapping, keys, default=None):
    current = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def coalesce(*values):
    for value in values:
        if value is not None:
            return value
    return None


def parse_encoded_float(text, label):
    if not text:
        return None
    pattern = r"(?:^|[\\/_-]){0}-([0-9]+(?:p[0-9]+)?(?:e[-+]?[0-9]+)?)".format(
        re.escape(label)
    )
    match = re.search(pattern, str(text))
    if not match:
        return None
    try:
        return float(match.group(1).replace("p", "."))
    except ValueError:
        return None


def load_checkpoint_record(path):
    if not path.exists():
        return {}

    with zipfile.ZipFile(str(path)) as zf:
        data_names = [name for name in zf.namelist() if name.endswith("/data.pkl")]
        if not data_names:
            raise ValueError("Could not find data.pkl inside {0}".format(path))
        raw = zf.read(data_names[0])
    return _SkipTensorUnpickler(io.BytesIO(raw)).load()


def resolve_project_path(base_dir, dataset_name, path_text):
    return dataset_path(base_dir, dataset_name)


def summarize_checkpoint_metadata(metadata):
    if not isinstance(metadata, dict):
        return {}

    train_meta = metadata.get("train", {}) or {}
    val_meta = metadata.get("val", {}) or {}
    test_meta = metadata.get("test", {}) or {}

    return {
        "source_format": train_meta.get("source_format"),
        "source_path": train_meta.get("source_path"),
        "time_steps_per_trajectory": train_meta.get("Nt"),
        "snapshots_per_trajectory": train_meta.get("snapshots_per_trajectory"),
        "dt": train_meta.get("dt"),
        "dx": train_meta.get("dx"),
        "domain_length": train_meta.get("domain_length"),
        "num_pairs": train_meta.get("num_pairs"),
        "train_pairs": train_meta.get("num_pairs"),
        "val_pairs": val_meta.get("num_pairs"),
        "test_pairs": test_meta.get("num_pairs"),
        "num_trajectories_train": train_meta.get("num_trajectories_split"),
        "num_trajectories_val": val_meta.get("num_trajectories_split"),
        "num_trajectories_test": test_meta.get("num_trajectories_split"),
    }


def summarize_npz_dataset(path):
    if path is None or not path.exists() or path.suffix != ".npz":
        return {}

    data = np.load(path, allow_pickle=False)

    def scalar(name):
        if name not in data.files:
            return None
        arr = np.asarray(data[name])
        if arr.shape == ():
            return clean_value(arr.item())
        return None

    rollout_states = data["rollout_states"] if "rollout_states" in data.files else None
    snapshots = None
    if rollout_states is not None and rollout_states.ndim >= 2:
        snapshots = int(rollout_states.shape[1])

    return {
        "source_format": "npz_rollout",
        "source_path": str(path),
        "num_nodes": scalar("N"),
        "domain_length": scalar("L"),
        "wavespeed": scalar("c"),
        "cfl": scalar("CFL"),
        "dt": scalar("dt"),
        "dx": scalar("dx"),
        "num_trajectories": scalar("num_samples"),
        "num_pairs": scalar("num_pairs"),
        "time_steps_per_trajectory": scalar("window"),
        "snapshots_per_trajectory": snapshots,
        "initial_condition_type": scalar("ic_type"),
        "label_method": scalar("label_method"),
    }


def infer_dataset_metadata(
    base_dir, dataset_name, config, summary, checkpoint_metadata, relative_run_dir
):
    metadata = summarize_checkpoint_metadata(checkpoint_metadata)
    data_cfg = config.get("data", {}) or {}

    dataset_path_text = coalesce(
        data_cfg.get("dataset_path"),
        data_cfg.get("data_dir"),
        summary.get("dataset_path"),
        summary.get("data_dir"),
        metadata.get("source_path"),
    )
    dataset_path = resolve_project_path(base_dir, dataset_name, dataset_path_text)
    npz_metadata = summarize_npz_dataset(dataset_path)
    metadata.update({k: v for k, v in npz_metadata.items() if v is not None})

    if dataset_path is not None:
        metadata["source_path"] = str(dataset_path)

    encoded_text = " ".join(
        [str(v) for v in [dataset_path_text, dataset_path, relative_run_dir] if v is not None]
    )
    if metadata.get("cfl") is None:
        metadata["cfl"] = parse_encoded_float(encoded_text, "cfl")
    if metadata.get("dt") is None:
        metadata["dt"] = parse_encoded_float(encoded_text, "dt")

    train_n = metadata.get("num_trajectories_train")
    val_n = metadata.get("num_trajectories_val")
    test_n = metadata.get("num_trajectories_test")
    num_trajectories = metadata.get("num_trajectories")
    if num_trajectories is not None and any(v is None for v in [train_n, val_n, test_n]):
        val_fraction = data_cfg.get("val_fraction")
        test_fraction = data_cfg.get("test_fraction")
        if val_fraction is not None and test_fraction is not None:
            val_n = int(round(float(num_trajectories) * float(val_fraction)))
            test_n = int(round(float(num_trajectories) * float(test_fraction)))
            train_n = int(num_trajectories) - val_n - test_n
            metadata["num_trajectories_train"] = train_n
            metadata["num_trajectories_val"] = val_n
            metadata["num_trajectories_test"] = test_n

    if metadata.get("num_trajectories") is None and all(v is not None for v in [train_n, val_n, test_n]):
        metadata["num_trajectories"] = int(train_n) + int(val_n) + int(test_n)

    train_pairs = metadata.get("train_pairs")
    val_pairs = metadata.get("val_pairs")
    test_pairs = metadata.get("test_pairs")
    if all(v is not None for v in [train_pairs, val_pairs, test_pairs]):
        metadata["num_pairs"] = int(train_pairs) + int(val_pairs) + int(test_pairs)

    return metadata


def effective_flag(enabled, flag):
    if enabled is False:
        return False
    if enabled is True:
        if flag is None:
            return None
        return bool(flag)
    return flag


def loss_config(config, summary, name):
    return coalesce(get_nested(config, ["train", name]), summary.get(name), {})


def completed_training_epochs(run_dir, checkpoint_path, checkpoint):
    """Read the final recorded epoch, keeping it distinct from the budget/best epoch."""
    def last_epoch(history, source):
        if isinstance(history, dict):
            history = history.get("history")
        if not history:
            return None
        if not isinstance(history, list):
            raise ValueError("Training history must be a list: {0}".format(source))
        epochs = []
        for entry in history:
            value = entry.get("epoch") if isinstance(entry, dict) else None
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("Invalid training epoch in {0}: {1!r}".format(source, value))
            if not math.isfinite(value) or value < 1 or int(value) != value:
                raise ValueError("Invalid training epoch in {0}: {1!r}".format(source, value))
            epochs.append(int(value))
        if any(right <= left for left, right in zip(epochs, epochs[1:])):
            raise ValueError("Training epochs are not increasing: {0}".format(source))
        return epochs[-1]

    for name in ("history.json", "history_checkpoint.json"):
        path = run_dir / name
        if path.exists():
            epoch = last_epoch(load_json(path), str(path))
            if epoch is not None:
                return epoch, str(path)
    # The original traffic runs store their full history inside checkpoint.pt.
    if not checkpoint and checkpoint_path.exists():
        checkpoint = load_checkpoint_record(checkpoint_path)
    source = str(checkpoint_path) + ":history"
    epoch = last_epoch(checkpoint.get("history"), source)
    return epoch, source if epoch is not None else None


def build_row(base_dir, dataset_name, method, relative_run_dir, omit_raw_config=False):
    run_dir = base_dir / relative_run_dir
    summary_candidates = [
        run_dir / "summary.json",
        run_dir / "rollout_plots" / "summary_metrics.json",
        run_dir / "summary_metrics.json",
    ]
    summary_path = next((path for path in summary_candidates if path.exists()), summary_candidates[0])
    config_resolved_path = run_dir / "config_resolved.json"
    config_input_path = run_dir / "config_input.json"
    config_json_path = run_dir / "config.json"
    checkpoint_candidates = [run_dir / "checkpoint_best.pt", run_dir / "checkpoint.pt"]
    checkpoint_path = next(
        (path for path in checkpoint_candidates if path.exists()), checkpoint_candidates[0]
    )

    notes = []
    summary = load_json(summary_path)
    checkpoint = {}
    checkpoint_metadata = {}
    config = {}
    config_source = None

    if config_resolved_path.exists():
        config = load_json(config_resolved_path)
        config_source = str(config_resolved_path)
    elif config_json_path.exists():
        config = load_json(config_json_path)
        config_source = str(config_json_path)

    needs_checkpoint = not config or not any(
        key in summary for key in ("best_epoch", "best_val_loss")
    )
    if checkpoint_path.exists() and needs_checkpoint:
        checkpoint = load_checkpoint_record(checkpoint_path)
        if not config:
            config = coalesce(
                checkpoint.get("config"),
                checkpoint.get("training_args"),
                {},
            )
            config_source = str(checkpoint_path) + ":config"
        checkpoint_metadata = checkpoint.get("metadata", {}) or {}
        notes.append("checkpoint metadata read without importing torch")

    if not config:
        notes.append("no resolved training config found")

    if config_input_path.exists() and not config_resolved_path.exists():
        notes.append("config_input.json exists but another resolved config source was used")

    checkpoint_training = checkpoint.get("training_args", {}) or {}
    checkpoint_model = checkpoint.get("model_kwargs", {}) or {}

    raw_data_cfg = config.get("data", {})
    data_cfg = dict(raw_data_cfg) if isinstance(raw_data_cfg, dict) else {}
    if raw_data_cfg and not isinstance(raw_data_cfg, dict):
        data_cfg.setdefault("dataset_path", raw_data_cfg)
    data_cfg.setdefault("dataset_path", checkpoint.get("data_path"))
    for target_key, source_key in (
        ("batch_size", "batch_size"),
        ("train_fraction", "train_fraction"),
        ("val_fraction", "validation_fraction"),
        ("test_fraction", "test_fraction"),
    ):
        value = coalesce(config.get(source_key), checkpoint_training.get(source_key))
        if value is not None:
            data_cfg.setdefault(target_key, value)

    raw_model_cfg = config.get("model", {})
    model_cfg = dict(raw_model_cfg) if isinstance(raw_model_cfg, dict) else {}
    if raw_model_cfg and not isinstance(raw_model_cfg, dict):
        model_cfg.setdefault("type", raw_model_cfg)
    for target_key, source_key in (
        ("hidden", "hidden_channels"),
        ("layers", "num_layers"),
        ("dropout", "dropout"),
        ("activation", "activation"),
        ("use_layernorm", "layer_norm"),
    ):
        value = coalesce(
            config.get(source_key),
            checkpoint_training.get(source_key),
            checkpoint_model.get(source_key),
        )
        if value is not None:
            model_cfg.setdefault(target_key, value)

    raw_train_cfg = config.get("train", {})
    train_cfg = dict(raw_train_cfg) if isinstance(raw_train_cfg, dict) else {}
    for target_key, source_keys in (
        ("epochs", ("epochs",)),
        ("lr", ("learning_rate", "lr")),
        ("weight_decay", ("weight_decay",)),
        ("grad_clip", ("gradient_clip_norm", "grad_clip")),
        ("early_stopping_metric", ("early_stopping_metric",)),
        ("early_stopping_patience", ("patience", "early_stopping_patience")),
        ("rollout_steps", ("training_rollout_steps", "autoregressive_steps", "rollout_steps")),
    ):
        value = None
        for source_key in source_keys:
            value = coalesce(config.get(source_key), checkpoint_training.get(source_key))
            if value is not None:
                break
        if value is not None:
            train_cfg.setdefault(target_key, value)
    if dataset_name == "LWR Traffic" and train_cfg.get("rollout_steps") is None:
        train_cfg["rollout_steps"] = 1
        notes.append("training rollout width inferred from the legacy LWR trainer default")

    feature_options = coalesce(
        checkpoint.get("feature_options"),
        {},
    )
    input_features = dict(
        coalesce(config.get("input_features"), summary.get("input_features"), {})
    )
    feature_names = coalesce(checkpoint.get("feature_names"), summary.get("feature_names"), [])
    include_absolute_positions = coalesce(
        config.get("include_absolute_positions"),
        feature_options.get("include_absolute_positions"),
    )
    if include_absolute_positions is None and feature_names:
        include_absolute_positions = any("pos" in str(name).lower() for name in feature_names)
    if include_absolute_positions is not None:
        input_features.setdefault("include_pos", bool(include_absolute_positions))
    if feature_names:
        input_features.setdefault("feature_names", feature_names)

    physics_inputs = coalesce(
        config.get("physics_inputs"),
        checkpoint.get("physics_inputs"),
        summary.get("physics_inputs"),
        {},
    )

    normalized_config = dict(config)
    normalized_config["data"] = data_cfg
    normalized_config["model"] = model_cfg
    normalized_config["train"] = train_cfg
    normalized_config["input_features"] = input_features
    normalized_config["physics_inputs"] = physics_inputs
    dataset_metadata = infer_dataset_metadata(
        base_dir,
        dataset_name,
        normalized_config,
        summary,
        checkpoint_metadata,
        relative_run_dir,
    )

    lr_schedule_cfg = train_cfg.get("lr_schedule", {}) or {}
    lr_schedule_entries = coalesce(
        summary.get("lr_schedule"),
        lr_schedule_cfg.get("entries"),
        [],
    )

    gradient_loss = loss_config(config, summary, "gradient_weighted_loss")
    relative_mse = loss_config(config, summary, "relative_mse")
    mass_loss = loss_config(config, summary, "mass_conservation_loss")
    tv_loss = loss_config(config, summary, "tv_loss")

    target_mode = coalesce(
        data_cfg.get("target_mode"),
        config.get("target_type"),
        checkpoint.get("target_type"),
        summary.get("target_mode"),
    )
    model_in_channels = coalesce(summary.get("model_in_channels"), checkpoint_model.get("in_channels"))
    normalize_state = coalesce(
        input_features.get("normalize_state_to_minus1_1"),
        input_features.get("normalize_u_to_minus1_1"),
    )
    training_mode = coalesce(summary.get("training_mode"), "rollout")
    early_stopping_metric = coalesce(
        checkpoint.get("early_stopping_metric"),
        summary.get("early_stopping_metric"),
        train_cfg.get("early_stopping_metric"),
    )
    if early_stopping_metric is None:
        early_stopping_metric = {
            "Inviscid Burgers": "val_loss",
            "Linear Advection": "val_loss",
            "LWR Traffic": "normalized_mse",
            "Chemotactic Cell Migration": "normalized_mse",
        }.get(dataset_name)
    physics_enabled = physics_inputs.get("enabled")
    include_adv = effective_flag(physics_enabled, physics_inputs.get("include_adv"))
    advection_active = bool(physics_enabled) and bool(include_adv)
    physics_active = bool(physics_enabled)

    epochs_trained, training_history_source = completed_training_epochs(
        run_dir, checkpoint_path, checkpoint
    )
    if epochs_trained is None:
        notes.append("completed training epochs unavailable: no recorded training history")

    flat = {
        "dataset": dataset_name,
        "method": method,
        "run_dir": str(run_dir),
        "config_source": config_source,
        "summary_json": str(summary_path) if summary_path.exists() else None,
        "dataset_path": dataset_metadata.get("source_path"),
        "source_format": coalesce(dataset_metadata.get("source_format"), data_cfg.get("source_format")),
        "num_trajectories": dataset_metadata.get("num_trajectories"),
        "num_trajectories_train": dataset_metadata.get("num_trajectories_train"),
        "num_trajectories_val": dataset_metadata.get("num_trajectories_val"),
        "num_trajectories_test": dataset_metadata.get("num_trajectories_test"),
        "num_pairs": coalesce(dataset_metadata.get("num_pairs"), summary.get("num_pairs")),
        "train_pairs": coalesce(dataset_metadata.get("train_pairs"), summary.get("train_size")),
        "val_pairs": coalesce(dataset_metadata.get("val_pairs"), summary.get("val_size")),
        "test_pairs": coalesce(dataset_metadata.get("test_pairs"), summary.get("test_size")),
        "num_nodes": coalesce(dataset_metadata.get("num_nodes"), summary.get("num_nodes")),
        "time_steps_per_trajectory": dataset_metadata.get("time_steps_per_trajectory"),
        "snapshots_per_trajectory": dataset_metadata.get("snapshots_per_trajectory"),
        "dt": dataset_metadata.get("dt"),
        "dx": dataset_metadata.get("dx"),
        "cfl": dataset_metadata.get("cfl"),
        "wavespeed": dataset_metadata.get("wavespeed"),
        "target_mode": target_mode,
        "training_mode": training_mode,
        "rollout_steps": coalesce(train_cfg.get("rollout_steps"), summary.get("rollout_steps")),
        "epochs": train_cfg.get("epochs"),
        "epochs_trained": epochs_trained,
        "best_epoch": coalesce(
            summary.get("best_epoch"),
            checkpoint.get("best_epoch"),
            checkpoint.get("epoch"),
        ),
        "early_stopping_metric": early_stopping_metric,
        "best_val_loss": coalesce(
            summary.get("best_val_loss"),
            checkpoint.get("best_val_loss"),
            checkpoint.get("best_validation_normalized_mse"),
        ),
        "batch_size": data_cfg.get("batch_size"),
        "optimizer": "AdamW",
        "lr_initial": coalesce(summary.get("lr_initial"), train_cfg.get("lr")),
        "lr_final": coalesce(
            summary.get("lr_final"),
            lr_schedule_entries[-1].get("lr") if lr_schedule_entries else None,
            train_cfg.get("lr"),
        ),
        "lr_schedule_enabled": coalesce(
            summary.get("lr_schedule_enabled"),
            lr_schedule_cfg.get("enabled"),
            bool(lr_schedule_entries),
        ),
        "lr_schedule": lr_schedule_entries,
        "weight_decay": train_cfg.get("weight_decay"),
        "grad_clip": train_cfg.get("grad_clip"),
        "early_stopping_patience": train_cfg.get("early_stopping_patience"),
        "model_type": model_cfg.get("type"),
        "hidden_dim": model_cfg.get("hidden"),
        "message_passing_layers": model_cfg.get("layers"),
        "dropout": model_cfg.get("dropout"),
        "activation": model_cfg.get("activation"),
        "use_skip": model_cfg.get("use_skip"),
        "use_layernorm": model_cfg.get("use_layernorm"),
        "predict_type": coalesce(model_cfg.get("predict_type"), target_mode),
        "state_residual_mode": coalesce(model_cfg.get("state_residual_mode"), summary.get("state_residual_mode")),
        "model_in_channels": model_in_channels,
        "input_features": input_features,
        "normalize_state_to_minus1_1": normalize_state,
        "normalize_state_clip": input_features.get("normalize_clip"),
        "physics_enabled": physics_enabled,
        "include_adv": include_adv,
        "advection_scheme": physics_inputs.get("advection_scheme") if advection_active else None,
        "physics_input_form": physics_inputs.get("input_form") if physics_active else None,
        "adv_all_steps": physics_inputs.get("adv_all_steps") if advection_active else None,
        "normalize_adv_to_minus1_1": (
            physics_inputs.get("normalize_adv_to_minus1_1") if advection_active else None
        ),
        "normalize_adv_clip": physics_inputs.get("normalize_adv_clip") if advection_active else None,
        "tv_loss_enabled": coalesce(tv_loss.get("enabled"), False),
        "tv_loss_lambda": tv_loss.get("lambda_tv"),
        "tv_loss_mode": tv_loss.get("mode"),
        "gradient_weighted_loss_enabled": coalesce(gradient_loss.get("enabled"), False),
        "gradient_weighted_loss_strength": gradient_loss.get("strength"),
        "relative_mse_enabled": coalesce(relative_mse.get("enabled"), False),
        "relative_mse_alpha": relative_mse.get("alpha"),
        "mass_conservation_loss_enabled": coalesce(mass_loss.get("enabled"), False),
        "mass_conservation_loss_lambda": mass_loss.get("lambda_mass"),
        "flux_loss_weight": coalesce(
            config.get("flux_loss_weight"),
            checkpoint.get("flux_loss_weight"),
        ),
        "node_weight_loss": coalesce(
            config.get("node_weight_loss"),
            checkpoint.get("node_weight_loss"),
        ),
        "mass_projection": coalesce(
            config.get("mass_projection"),
            checkpoint.get("mass_projection"),
        ),
        "include_absolute_positions": include_absolute_positions,
        "include_boundary_distances": coalesce(
            config.get("include_boundary_distances"),
            feature_options.get("include_boundary_distances"),
        ),
        "include_drift_velocity": coalesce(
            config.get("include_drift_velocity"),
            feature_options.get("include_drift_velocity"),
        ),
        "include_chi": coalesce(config.get("include_chi"), feature_options.get("include_chi")),
        "feature_names": feature_names,
        "normalization": checkpoint.get("normalization"),
        "feature_normalization": checkpoint.get("feature_normalization"),
        "device": coalesce(summary.get("device"), config.get("device")),
        "seed": config.get("seed"),
        "notes": "; ".join(notes),
    }

    json_row = dict(flat)
    json_row["_sources"] = {
        "run_dir": str(run_dir),
        "training_history": training_history_source,
        "summary_json": str(summary_path) if summary_path.exists() else None,
        "config_resolved_json": str(config_resolved_path) if config_resolved_path.exists() else None,
        "config_input_json": str(config_input_path) if config_input_path.exists() else None,
        "config_json": str(config_json_path) if config_json_path.exists() else None,
        "checkpoint": str(checkpoint_path) if checkpoint_path.exists() else None,
    }
    json_row["_dataset_metadata"] = dataset_metadata
    json_row["_checkpoint"] = {
        "epoch": checkpoint.get("epoch"),
        "best_epoch": checkpoint.get("best_epoch"),
        "best_val_loss": coalesce(
            checkpoint.get("best_val_loss"),
            checkpoint.get("best_validation_normalized_mse"),
        ),
        "has_metadata": bool(checkpoint_metadata),
        "model_kwargs": checkpoint_model,
        "feature_options": feature_options,
        "training_args": checkpoint_training,
    }
    if not omit_raw_config:
        json_row["_config"] = config

    return clean_value(flat), clean_value(json_row)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: csv_value(row.get(col)) for col in COLUMNS})


def write_json(path, rows, csv_path, base_dir, skipped_rows=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_script": str(Path(__file__).resolve()),
        "base_dir": str(base_dir),
        "csv_path": str(csv_path),
        "row_count": len(rows),
        "skipped_row_count": len(skipped_rows or []),
        "skipped_rows": skipped_rows or [],
        "columns": COLUMNS,
        "rows": rows,
    }
    with path.open("w") as fh:
        json.dump(clean_value(payload), fh, indent=2, sort_keys=True)
        fh.write("\n")


def main():
    from make_results_metrics import RUNS

    args = parse_args()
    flat_rows = []
    json_rows = []
    skipped_rows = []
    for dataset_name, method, relative_run_dir in iter_run_entries(RUNS):
        flat, rich = build_row(
            args.base_dir,
            dataset_name,
            method,
            relative_run_dir,
            omit_raw_config=args.omit_raw_config,
        )
        flat_rows.append(flat)
        json_rows.append(rich)

    write_csv(args.output_csv, flat_rows)
    write_json(args.output_json, json_rows, args.output_csv, args.base_dir, skipped_rows=skipped_rows)

    print("Wrote {0}".format(args.output_csv))
    print("Wrote {0}".format(args.output_json))
    if skipped_rows:
        print("Skipped {0} rows not included in this table.".format(len(skipped_rows)))


if __name__ == "__main__":
    main()
