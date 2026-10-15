#!/usr/bin/env python3
"""Train a vanilla SAGEConv GNN surrogate for periodic heat-equation data.

Typical usage from repo root:

    python scripts/train_gnn_heat.py --config config_gnn_heat.json

To enable multi-step rollout training, set `train.rollout_steps` in the config
to an integer > 1 (requires rollout dataset fields).

When trajectory metadata is present in the dataset, train/val/test splits are
computed by trajectory id (then expanded to pair indices).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from heat_data import load_heat_dataset
from gnn_heat import (
    TARGET_MODE_DELTA,
    EpochMetrics,
    TARGET_MODE_RATE,
    TARGET_MODE_STATE,
    TARGET_MODES,
    build_batched_edge_index,
    build_model_from_config,
    make_target,
    predict_type_from_target_mode,
    resolve_device,
    save_json,
    set_seed,
    split_indices,
)
from input_features import (
    InputFeatureConfig,
    apply_state_minmax_minus1_1,
    append_position_channels,
    count_input_feature_channels,
    fit_state_minmax,
    resolve_input_feature_cfg,
)


class IndexDataset(Dataset):
    """Simple dataset returning precomputed sample indices."""

    def __init__(self, indices: np.ndarray) -> None:
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, idx: int) -> int:
        return int(self.indices[idx])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SAGE GNN for 1D or 2D heat-equation data."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config_gnn_heat.json"),
        help="Path to JSON config.",
    )
    return parser.parse_args()


def _parse_lr_schedule(
    train_cfg: Mapping[str, Any],
    *,
    epochs: int,
) -> Tuple[bool, Dict[int, float]]:
    """Parse optional epoch-based learning-rate schedule from config.

    Supported config shape:
      train.lr_schedule: [{"epoch": 200, "lr": 5e-5}, {"epoch": 400, "lr": 1e-5}]

    A single object is also accepted:
      train.lr_schedule: {"epoch": 200, "lr": 5e-5}

    New object form with boolean toggle:
      train.lr_schedule: {
        "enabled": true,
        "entries": [{"epoch": 200, "lr": 5e-5}]
      }
    """
    raw = train_cfg.get("lr_schedule", None)
    if raw is None:
        return False, {}

    entries: List[Mapping[str, Any]]
    enabled = True
    if isinstance(raw, Mapping):
        enabled = bool(raw.get("enabled", True))
        if ("epoch" in raw) or ("lr" in raw):
            entries = [raw]
        else:
            entries_raw = raw.get("entries", raw.get("milestones", []))
            if isinstance(entries_raw, Mapping):
                entries = [entries_raw]
            elif isinstance(entries_raw, list):
                entries = []
                for item in entries_raw:
                    if not isinstance(item, Mapping):
                        raise ValueError(
                            "train.lr_schedule entries must be objects with keys "
                            "'epoch' and 'lr'."
                        )
                    entries.append(item)
            else:
                raise ValueError(
                    "train.lr_schedule.entries must be an object or list of objects."
                )
    elif isinstance(raw, list):
        entries = []
        for item in raw:
            if not isinstance(item, Mapping):
                raise ValueError(
                    "train.lr_schedule entries must be objects with keys "
                    "'epoch' and 'lr'."
                )
            entries.append(item)
    else:
        raise ValueError(
            "train.lr_schedule must be either an object "
            "{epoch: <int>, lr: <float>} or a list of such objects."
        )

    if not enabled:
        return False, {}

    schedule: Dict[int, float] = {}
    for entry in entries:
        if "epoch" not in entry or "lr" not in entry:
            raise ValueError(
                "Each train.lr_schedule entry must include keys 'epoch' and 'lr'."
            )
        ep = int(entry["epoch"])
        lr_val = float(entry["lr"])
        if ep < 1:
            raise ValueError(f"train.lr_schedule epoch must be >= 1, got {ep}.")
        if lr_val <= 0.0:
            raise ValueError(f"train.lr_schedule lr must be > 0, got {lr_val}.")
        if ep > int(epochs):
            continue
        if ep in schedule:
            raise ValueError(
                f"Duplicate train.lr_schedule epoch {ep}. "
                "Provide at most one LR value per epoch."
            )
        schedule[ep] = lr_val

    return True, schedule


def _resolve_config_spatial_dim(cfg: Mapping[str, Any]) -> int | None:
    """Read optional spatial-dim setting from config."""
    data_cfg = cfg.get("data", {})
    raw = None
    if isinstance(data_cfg, Mapping):
        raw = data_cfg.get("spatial-dim", data_cfg.get("spatial_dim", None))
    if raw is None:
        raw = cfg.get("spatial-dim", cfg.get("spatial_dim", None))
    if raw is None:
        return None

    spatial_dim = int(raw)
    if spatial_dim not in {1, 2}:
        raise ValueError(
            f"spatial-dim must be 1 or 2 when provided, got {spatial_dim}."
        )
    return spatial_dim


def _batch_metrics(pred: torch.Tensor, target: torch.Tensor) -> Tuple[float, float]:
    err = pred - target
    mse = float(torch.mean(err * err).item())
    rmse = math.sqrt(max(mse, 0.0))
    denom = float(torch.norm(target).item())
    rel_l2 = float(torch.norm(err).item() / max(denom, 1e-12))
    return rmse, rel_l2


def _run_split(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    x_all: torch.Tensor,
    u_next_all: torch.Tensor,
    edge_index: torch.Tensor,
    pos_nodes: torch.Tensor,
    dt: float,
    input_cfg: InputFeatureConfig,
    target_mode: str,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    grad_clip: float,
) -> EpochMetrics:
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_rmse = 0.0
    total_rel_l2 = 0.0
    total_count = 0

    for batch_indices in loader:
        idx = batch_indices.long()
        x_b = x_all[idx].to(device=device)
        u_next_b = u_next_all[idx].to(device=device)
        target_b = make_target(x_b, u_next_b, dt=dt, target_mode=target_mode)

        bsz, n_nodes, n_feat = x_b.shape
        x_flat = x_b.reshape(bsz * n_nodes, n_feat)
        target_flat = target_b.reshape(bsz * n_nodes, n_feat)
        ei_batched = build_batched_edge_index(
            edge_index=edge_index,
            num_nodes=n_nodes,
            batch_size=bsz,
            device=device,
        )

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        x_model_flat = append_position_channels(
            x_state=x_flat,
            pos_nodes=pos_nodes,
            input_cfg=input_cfg,
            batch_size=bsz,
        )
        x_model_flat = apply_state_minmax_minus1_1(
            x_model=x_model_flat,
            input_cfg=input_cfg,
        )

        pred_state_flat = model.predict_state(x_model_flat, ei_batched, dt=dt)
        if target_mode == TARGET_MODE_STATE:
            pred_target_flat = pred_state_flat
        elif target_mode == TARGET_MODE_DELTA:
            pred_target_flat = pred_state_flat - x_flat
        elif target_mode == TARGET_MODE_RATE:
            pred_target_flat = (pred_state_flat - x_flat) / float(dt)
        else:
            raise RuntimeError(f"Unexpected target_mode='{target_mode}'.")

        loss = torch.mean((pred_target_flat - target_flat) ** 2)

        if is_train:
            loss.backward()
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

        batch_n = int(bsz)
        rmse, rel_l2 = _batch_metrics(pred_target_flat.detach(), target_flat.detach())
        total_loss += float(loss.item()) * batch_n
        total_rmse += rmse * batch_n
        total_rel_l2 += rel_l2 * batch_n
        total_count += batch_n

    if total_count == 0:
        return EpochMetrics(loss=float("nan"), rmse=float("nan"), rel_l2=float("nan"))
    return EpochMetrics(
        loss=total_loss / total_count,
        rmse=total_rmse / total_count,
        rel_l2=total_rel_l2 / total_count,
    )


def _target_from_states(
    *,
    x_now_flat: torch.Tensor,
    state_next_flat: torch.Tensor,
    dt: float,
    target_mode: str,
) -> torch.Tensor:
    if target_mode == TARGET_MODE_STATE:
        return state_next_flat
    if target_mode == TARGET_MODE_DELTA:
        return state_next_flat - x_now_flat
    if target_mode == TARGET_MODE_RATE:
        return (state_next_flat - x_now_flat) / float(dt)
    raise RuntimeError(f"Unexpected target_mode='{target_mode}'.")


def _select_rollout_start_indices(
    *,
    split_indices_np: np.ndarray,
    pair_time_index_all: torch.Tensor,
    window: int,
    rollout_steps: int,
) -> np.ndarray:
    if rollout_steps < 1:
        raise ValueError(f"rollout_steps must be >= 1, got {rollout_steps}.")
    if window < rollout_steps:
        return np.zeros((0,), dtype=np.int64)
    max_start_t = int(window - rollout_steps)
    split_idx_t = torch.as_tensor(split_indices_np, dtype=torch.long)
    split_t = pair_time_index_all[split_idx_t].detach().cpu().numpy()
    keep = split_t <= max_start_t
    return np.asarray(split_indices_np[keep], dtype=np.int64)


def _split_pair_indices_by_trajectory(
    *,
    num_pairs: int,
    pair_trajectory_id: torch.Tensor | None,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    """Split pair indices by trajectory ids when available."""
    if pair_trajectory_id is None:
        train_idx, val_idx, test_idx = split_indices(
            num_pairs,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            seed=seed,
        )
        return train_idx, val_idx, test_idx, None, None, None

    pair_traj_np = pair_trajectory_id.detach().cpu().numpy().astype(np.int64)
    if pair_traj_np.shape[0] != int(num_pairs):
        raise ValueError(
            "pair_trajectory_id length mismatch: "
            f"expected {num_pairs}, got {pair_traj_np.shape[0]}."
        )

    unique_traj = np.unique(pair_traj_np)
    train_ti, val_ti, test_ti = split_indices(
        int(unique_traj.shape[0]),
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )
    train_traj = unique_traj[train_ti]
    val_traj = unique_traj[val_ti]
    test_traj = unique_traj[test_ti]

    train_idx = np.nonzero(np.isin(pair_traj_np, train_traj))[0].astype(np.int64)
    val_idx = np.nonzero(np.isin(pair_traj_np, val_traj))[0].astype(np.int64)
    test_idx = np.nonzero(np.isin(pair_traj_np, test_traj))[0].astype(np.int64)

    return train_idx, val_idx, test_idx, train_traj, val_traj, test_traj


def _run_split_rollout(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    rollout_states: torch.Tensor,
    pair_trajectory_id_all: torch.Tensor,
    pair_time_index_all: torch.Tensor,
    edge_index: torch.Tensor,
    pos_nodes: torch.Tensor,
    dt: float,
    input_cfg: InputFeatureConfig,
    target_mode: str,
    rollout_steps: int,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    grad_clip: float,
) -> EpochMetrics:
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_rmse = 0.0
    total_rel_l2 = 0.0
    total_count = 0

    for batch_indices in loader:
        idx = batch_indices.long()
        traj_idx = pair_trajectory_id_all[idx]
        t0_idx = pair_time_index_all[idx]

        x_curr = rollout_states[traj_idx, t0_idx].to(device=device)
        bsz, n_nodes, n_feat = x_curr.shape
        ei_batched = build_batched_edge_index(
            edge_index=edge_index,
            num_nodes=n_nodes,
            batch_size=bsz,
            device=device,
        )

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        loss_accum = torch.zeros((), dtype=x_curr.dtype, device=device)
        rmse_accum = 0.0
        rel_accum = 0.0

        with torch.set_grad_enabled(is_train):
            for step in range(1, rollout_steps + 1):
                true_next = rollout_states[traj_idx, t0_idx + step].to(device=device)

                x_flat = x_curr.reshape(bsz * n_nodes, n_feat)
                true_next_flat = true_next.reshape(bsz * n_nodes, n_feat)
                x_model_flat = append_position_channels(
                    x_state=x_flat,
                    pos_nodes=pos_nodes,
                    input_cfg=input_cfg,
                    batch_size=bsz,
                )
                x_model_flat = apply_state_minmax_minus1_1(
                    x_model=x_model_flat,
                    input_cfg=input_cfg,
                )
                pred_state_flat = model.predict_state(x_model_flat, ei_batched, dt=dt)
                pred_state = pred_state_flat.reshape(bsz, n_nodes, n_feat)

                pred_target_flat = _target_from_states(
                    x_now_flat=x_flat,
                    state_next_flat=pred_state_flat,
                    dt=dt,
                    target_mode=target_mode,
                )
                true_target_flat = _target_from_states(
                    x_now_flat=x_flat,
                    state_next_flat=true_next_flat,
                    dt=dt,
                    target_mode=target_mode,
                )

                step_loss = torch.mean((pred_target_flat - true_target_flat) ** 2)
                loss_accum = loss_accum + step_loss

                rmse_step, rel_step = _batch_metrics(
                    pred_target_flat.detach(), true_target_flat.detach()
                )
                rmse_accum += rmse_step
                rel_accum += rel_step

                x_curr = pred_state

            loss = loss_accum / float(rollout_steps)

            if is_train:
                loss.backward()
                if grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()

        batch_n = int(bsz)
        total_loss += float(loss.item()) * batch_n
        total_rmse += (rmse_accum / float(rollout_steps)) * batch_n
        total_rel_l2 += (rel_accum / float(rollout_steps)) * batch_n
        total_count += batch_n

    if total_count == 0:
        return EpochMetrics(loss=float("nan"), rmse=float("nan"), rel_l2=float("nan"))
    return EpochMetrics(
        loss=total_loss / total_count,
        rmse=total_rmse / total_count,
        rel_l2=total_rel_l2 / total_count,
    )


def main() -> None:
    args = parse_args()
    cfg = {}
    with open(args.config, "r", encoding="utf-8") as f:
        import json

        cfg = json.load(f)

    seed = int(cfg.get("seed", 1234))
    set_seed(seed)

    device = resolve_device(str(cfg.get("device", "auto")))

    data_cfg = dict(cfg.get("data", {}))
    dataset_path = Path(data_cfg.get("dataset_path", "data/single_steps/heat_train.npz"))
    target_mode = str(data_cfg.get("target_mode", TARGET_MODE_STATE)).strip().lower()
    config_spatial_dim = _resolve_config_spatial_dim(cfg)
    if target_mode not in TARGET_MODES:
        raise ValueError(
            f"data.target_mode must be one of {sorted(TARGET_MODES)}, got '{target_mode}'."
        )
    resolved_predict_type = predict_type_from_target_mode(target_mode)

    cfg_model = dict(cfg.get("model", {}))
    old_predict_type = cfg_model.get("predict_type", None)
    if old_predict_type is not None:
        old_key = str(old_predict_type).strip().lower()
        if old_key != resolved_predict_type:
            print(
                "[INFO] Ignoring model.predict_type from config and using "
                f"data.target_mode='{target_mode}' -> predict_type='{resolved_predict_type}'.",
                flush=True,
            )
    cfg_model["predict_type"] = resolved_predict_type
    cfg["model"] = cfg_model

    batch_size = int(data_cfg.get("batch_size", 32))
    num_workers = int(data_cfg.get("num_workers", 0))
    val_fraction = float(data_cfg.get("val_fraction", 0.1))
    test_fraction = float(data_cfg.get("test_fraction", 0.1))
    split_cfg = data_cfg.get("split", {})
    if isinstance(split_cfg, dict):
        split_unit_cfg = split_cfg.get("unit", None)
        if split_unit_cfg is not None:
            key = str(split_unit_cfg).strip().lower()
            if key != "trajectory":
                raise ValueError(
                    "data.split.unit is for documentation only and must be "
                    f"'trajectory' when provided, got '{split_unit_cfg}'."
                )
        train_fraction_cfg = split_cfg.get("train_fraction", None)
        if train_fraction_cfg is not None:
            expected_train_fraction = 1.0 - val_fraction - test_fraction
            if abs(float(train_fraction_cfg) - expected_train_fraction) > 1e-12:
                raise ValueError(
                    "data.split.train_fraction must satisfy "
                    "train_fraction = 1 - val_fraction - test_fraction. "
                    f"Got train_fraction={train_fraction_cfg}, "
                    f"val_fraction={val_fraction}, test_fraction={test_fraction}."
                )

    dset = load_heat_dataset(dataset_path, dtype=torch.float32)
    x_all = dset.x.detach().cpu()
    u_next_all = dset.u_next.detach().cpu()
    edge_index = dset.edge_index.detach().cpu()
    pos_nodes = dset.pos.detach().cpu()
    n_samples = int(len(dset))
    n_nodes = int(dset.x.shape[1])
    dt = float(dset.metadata["dt"])
    spatial_dim = int(dset.metadata.get("spatial_dim", config_spatial_dim or 2))
    nx = int(dset.metadata.get("nx", n_nodes))
    ny = int(dset.metadata.get("ny", 1 if spatial_dim == 1 else n_nodes // nx))
    if spatial_dim not in {1, 2}:
        raise ValueError(
            "Dataset metadata spatial_dim must be 1 or 2 for this trainer, "
            f"got {spatial_dim}."
        )
    if config_spatial_dim is not None and spatial_dim != config_spatial_dim:
        raise ValueError(
            "Config data.spatial-dim does not match dataset metadata for "
            f"{dataset_path}: config has {config_spatial_dim}, dataset has "
            f"{spatial_dim}."
        )
    print(
        "[INFO] Loaded dataset: "
        f"spatial_dim={spatial_dim}, config_spatial_dim={config_spatial_dim}, "
        f"nx={nx}, ny={ny}, "
        f"num_nodes={n_nodes}, num_pairs={n_samples}, dt={dt:.6e}.",
        flush=True,
    )
    input_cfg = resolve_input_feature_cfg(cfg)
    raw_input_features_cfg = cfg.get("input_features", {})
    cfg_input_features = (
        dict(raw_input_features_cfg)
        if isinstance(raw_input_features_cfg, dict)
        else {}
    )
    train_cfg = dict(cfg.get("train", {}))
    rollout_steps = int(train_cfg.get("rollout_steps", 1))
    if rollout_steps < 1:
        raise ValueError(f"train.rollout_steps must be >= 1, got {rollout_steps}.")

    pair_trajectory_id_all = dset.pair_trajectory_id
    pair_time_index_all = dset.pair_time_index
    rollout_states_all = dset.rollout_states
    (
        train_idx,
        val_idx,
        test_idx,
        train_traj_ids,
        val_traj_ids,
        test_traj_ids,
    ) = _split_pair_indices_by_trajectory(
        num_pairs=n_samples,
        pair_trajectory_id=pair_trajectory_id_all,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )
    split_unit = "trajectory" if pair_trajectory_id_all is not None else "pair"
    num_trajectories = (
        int(torch.unique(pair_trajectory_id_all).numel())
        if pair_trajectory_id_all is not None
        else None
    )

    rollout_mode = rollout_steps > 1
    train_effective_idx = train_idx
    val_effective_idx = val_idx
    test_effective_idx = test_idx

    if split_unit == "trajectory":
        assert train_traj_ids is not None
        assert val_traj_ids is not None
        assert test_traj_ids is not None
        print(
            "[INFO] Data split by trajectory ids: "
            f"num_trajectories={num_trajectories}, "
            f"train/val/test trajectories="
            f"{len(train_traj_ids)}/{len(val_traj_ids)}/{len(test_traj_ids)}.",
            flush=True,
        )
    else:
        print(
            "[INFO] Data split by pair indices (no trajectory metadata).",
            flush=True,
        )

    if input_cfg.normalize_state_to_minus1_1:
        train_idx_t = torch.as_tensor(train_idx, dtype=torch.long)
        state_min, state_max = fit_state_minmax(
            x_state=x_all,
            sample_indices=train_idx_t,
        )
        input_cfg = InputFeatureConfig(
            include_pos=bool(input_cfg.include_pos),
            normalize_state_to_minus1_1=True,
            state_min=float(state_min),
            state_max=float(state_max),
            normalize_clip=bool(input_cfg.normalize_clip),
        )
        print(
            "[INFO] State normalization enabled: "
            f"range=[-1,1], state_min={state_min:.6e}, state_max={state_max:.6e}, "
            f"clip={input_cfg.normalize_clip}",
            flush=True,
        )
    else:
        print("[INFO] State normalization disabled.", flush=True)

    cfg_input_features["include_pos"] = bool(input_cfg.include_pos)
    cfg_input_features["normalize_state_to_minus1_1"] = bool(
        input_cfg.normalize_state_to_minus1_1
    )
    cfg_input_features["normalize_clip"] = bool(input_cfg.normalize_clip)
    cfg_input_features["state_min"] = (
        None if input_cfg.state_min is None else float(input_cfg.state_min)
    )
    cfg_input_features["state_max"] = (
        None if input_cfg.state_max is None else float(input_cfg.state_max)
    )
    cfg["input_features"] = cfg_input_features

    if rollout_mode:
        if (
            rollout_states_all is None
            or pair_trajectory_id_all is None
            or pair_time_index_all is None
        ):
            raise ValueError(
                "train.rollout_steps > 1 requires rollout dataset metadata "
                "(rollout_states, pair_trajectory_id, pair_time_index)."
            )
        window = int(
            dset.metadata.get("window", int(rollout_states_all.shape[1] - 1))
        )
        if rollout_steps > window:
            raise ValueError(
                f"train.rollout_steps={rollout_steps} exceeds dataset window={window}."
            )

        train_effective_idx = _select_rollout_start_indices(
            split_indices_np=train_idx,
            pair_time_index_all=pair_time_index_all,
            window=window,
            rollout_steps=rollout_steps,
        )
        val_effective_idx = _select_rollout_start_indices(
            split_indices_np=val_idx,
            pair_time_index_all=pair_time_index_all,
            window=window,
            rollout_steps=rollout_steps,
        )
        test_effective_idx = _select_rollout_start_indices(
            split_indices_np=test_idx,
            pair_time_index_all=pair_time_index_all,
            window=window,
            rollout_steps=rollout_steps,
        )
        if (
            len(train_effective_idx) == 0
            or len(val_effective_idx) == 0
            or len(test_effective_idx) == 0
        ):
            raise ValueError(
                "No valid rollout start samples after filtering split indices. "
                "Try smaller train.rollout_steps or adjust split fractions/dataset size."
            )
        print(
            "[INFO] Rollout training enabled: "
            f"rollout_steps={rollout_steps}, "
            f"effective starts (train/val/test)="
            f"{len(train_effective_idx)}/{len(val_effective_idx)}/{len(test_effective_idx)}.",
            flush=True,
        )

    train_loader = DataLoader(
        IndexDataset(train_effective_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        IndexDataset(val_effective_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        IndexDataset(test_effective_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    in_channels = int(x_all.shape[-1]) + count_input_feature_channels(input_cfg)
    print(
        "[INFO] Input channels: "
        f"include_pos={input_cfg.include_pos}, "
        f"normalize_state_to_minus1_1={input_cfg.normalize_state_to_minus1_1}, "
        f"total_model_in_channels={in_channels}",
        flush=True,
    )
    model = build_model_from_config(cfg, in_channels=in_channels).to(device)
    epochs = int(train_cfg.get("epochs", 200))
    lr = float(train_cfg.get("lr", 1e-3))
    lr_initial = float(lr)
    weight_decay = float(train_cfg.get("weight_decay", 1e-6))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    raw_patience = train_cfg.get("early_stopping_patience", 30)
    patience: int | None
    if raw_patience is None:
        patience = None
    elif isinstance(raw_patience, str):
        key = raw_patience.strip().lower()
        if key in {"none", "null", "off", "disable", "disabled"}:
            patience = None
        else:
            patience = int(raw_patience)
            if patience <= 0:
                patience = None
    else:
        patience = int(raw_patience)
        if patience <= 0:
            patience = None
    log_every = int(train_cfg.get("log_every", 10))
    save_dir = Path(train_cfg.get("save_dir", "runs/heat_sage_state"))
    save_dir.mkdir(parents=True, exist_ok=True)
    lr_schedule_enabled, lr_schedule = _parse_lr_schedule(train_cfg, epochs=epochs)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if not lr_schedule_enabled:
        print("[INFO] LR schedule disabled.", flush=True)
    elif len(lr_schedule) == 0:
        print("[INFO] LR schedule enabled but no entries provided.", flush=True)
    else:
        schedule_str = ", ".join(
            [f"epoch {ep}: {val:.6e}" for ep, val in sorted(lr_schedule.items())]
        )
        print(f"[INFO] LR schedule: {schedule_str}", flush=True)

    if patience is None:
        print("[INFO] Early stopping disabled.", flush=True)
    history: List[Dict[str, float]] = []
    best_val = float("inf")
    best_epoch = -1
    stale = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        if lr_schedule_enabled and (epoch in lr_schedule):
            lr_new = float(lr_schedule[epoch])
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_new
            print(
                f"[INFO] LR schedule kick-in at epoch {epoch}: lr={lr_new:.6e}",
                flush=True,
            )

        current_lr = float(optimizer.param_groups[0]["lr"])
        epoch_t0 = time.time()
        if rollout_mode:
            assert rollout_states_all is not None
            assert pair_trajectory_id_all is not None
            assert pair_time_index_all is not None
            train_m = _run_split_rollout(
                model=model,
                loader=train_loader,
                rollout_states=rollout_states_all,
                pair_trajectory_id_all=pair_trajectory_id_all,
                pair_time_index_all=pair_time_index_all,
                edge_index=edge_index,
                pos_nodes=pos_nodes,
                dt=dt,
                input_cfg=input_cfg,
                target_mode=target_mode,
                rollout_steps=rollout_steps,
                optimizer=optimizer,
                device=device,
                grad_clip=grad_clip,
            )
            val_m = _run_split_rollout(
                model=model,
                loader=val_loader,
                rollout_states=rollout_states_all,
                pair_trajectory_id_all=pair_trajectory_id_all,
                pair_time_index_all=pair_time_index_all,
                edge_index=edge_index,
                pos_nodes=pos_nodes,
                dt=dt,
                input_cfg=input_cfg,
                target_mode=target_mode,
                rollout_steps=rollout_steps,
                optimizer=None,
                device=device,
                grad_clip=grad_clip,
            )
        else:
            train_m = _run_split(
                model=model,
                loader=train_loader,
                x_all=x_all,
                u_next_all=u_next_all,
                edge_index=edge_index,
                pos_nodes=pos_nodes,
                dt=dt,
                input_cfg=input_cfg,
                target_mode=target_mode,
                optimizer=optimizer,
                device=device,
                grad_clip=grad_clip,
            )
            val_m = _run_split(
                model=model,
                loader=val_loader,
                x_all=x_all,
                u_next_all=u_next_all,
                edge_index=edge_index,
                pos_nodes=pos_nodes,
                dt=dt,
                input_cfg=input_cfg,
                target_mode=target_mode,
                optimizer=None,
                device=device,
                grad_clip=grad_clip,
            )

        epoch_time_s = time.time() - epoch_t0

        row = {
            "epoch": float(epoch),
            "lr": current_lr,
            "epoch_time_s": epoch_time_s,
            "train_loss": train_m.loss,
            "train_rmse": train_m.rmse,
            "train_rel_l2": train_m.rel_l2,
            "val_loss": val_m.loss,
            "val_rmse": val_m.rmse,
            "val_rel_l2": val_m.rel_l2,
        }
        history.append(row)

        improved = val_m.loss < best_val
        if improved:
            best_val = val_m.loss
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": cfg,
                    "metadata": dict(dset.metadata),
                    "split_indices": {
                        "train": train_idx.tolist(),
                        "val": val_idx.tolist(),
                        "test": test_idx.tolist(),
                        "train_effective": train_effective_idx.tolist(),
                        "val_effective": val_effective_idx.tolist(),
                        "test_effective": test_effective_idx.tolist(),
                        "train_trajectory_ids": (
                            None if train_traj_ids is None else train_traj_ids.tolist()
                        ),
                        "val_trajectory_ids": (
                            None if val_traj_ids is None else val_traj_ids.tolist()
                        ),
                        "test_trajectory_ids": (
                            None if test_traj_ids is None else test_traj_ids.tolist()
                        ),
                    },
                    "epoch": epoch,
                    "best_val_loss": best_val,
                },
                save_dir / "checkpoint_best.pt",
            )
        else:
            stale += 1

        if (epoch == 1) or (epoch % log_every == 0) or improved:
            print(
                f"[epoch {epoch:04d}] "
                f"lr={current_lr:.6e} "
                f"train_loss={train_m.loss:.6e} val_loss={val_m.loss:.6e} "
                f"train_rmse={train_m.rmse:.6e} val_rmse={val_m.rmse:.6e} "
                f"epoch_time={epoch_time_s:.2f}s",
                flush=True,
            )

        if patience is not None and stale >= patience:
            print(
                f"Early stopping at epoch {epoch} (no val improvement for {patience} epochs).",
                flush=True,
            )
            break

    best_ckpt = torch.load(save_dir / "checkpoint_best.pt", map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    if rollout_mode:
        assert rollout_states_all is not None
        assert pair_trajectory_id_all is not None
        assert pair_time_index_all is not None
        test_m = _run_split_rollout(
            model=model,
            loader=test_loader,
            rollout_states=rollout_states_all,
            pair_trajectory_id_all=pair_trajectory_id_all,
            pair_time_index_all=pair_time_index_all,
            edge_index=edge_index,
            pos_nodes=pos_nodes,
            dt=dt,
            input_cfg=input_cfg,
            target_mode=target_mode,
            rollout_steps=rollout_steps,
            optimizer=None,
            device=device,
            grad_clip=grad_clip,
        )
    else:
        test_m = _run_split(
            model=model,
            loader=test_loader,
            x_all=x_all,
            u_next_all=u_next_all,
            edge_index=edge_index,
            pos_nodes=pos_nodes,
            dt=dt,
            input_cfg=input_cfg,
            target_mode=target_mode,
            optimizer=None,
            device=device,
            grad_clip=grad_clip,
        )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": cfg,
            "metadata": dict(dset.metadata),
            "split_indices": {
                "train": train_idx.tolist(),
                "val": val_idx.tolist(),
                "test": test_idx.tolist(),
                "train_effective": train_effective_idx.tolist(),
                "val_effective": val_effective_idx.tolist(),
                "test_effective": test_effective_idx.tolist(),
                "train_trajectory_ids": (
                    None if train_traj_ids is None else train_traj_ids.tolist()
                ),
                "val_trajectory_ids": (
                    None if val_traj_ids is None else val_traj_ids.tolist()
                ),
                "test_trajectory_ids": (
                    None if test_traj_ids is None else test_traj_ids.tolist()
                ),
            },
            "epoch": len(history),
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
            "test_metrics": test_m.to_dict(),
        },
        save_dir / "checkpoint_last.pt",
    )

    summary: Dict[str, Any] = {
        "run_seconds": float(time.time() - t0),
        "device": str(device),
        "dataset_path": str(dataset_path),
        "target_mode": target_mode,
        "lr_initial": float(lr_initial),
        "lr_final": float(optimizer.param_groups[0]["lr"]),
        "lr_schedule_enabled": bool(lr_schedule_enabled),
        "lr_schedule": [
            {"epoch": int(ep), "lr": float(val)}
            for ep, val in sorted(lr_schedule.items())
        ],
        "split_unit": split_unit,
        "training_mode": "rollout" if rollout_mode else "one_step",
        "rollout_steps": int(rollout_steps),
        "num_samples": n_samples,
        "num_pairs": int(n_samples),
        "num_trajectories": (
            None if num_trajectories is None else int(num_trajectories)
        ),
        "num_nodes": n_nodes,
        "spatial_dim": int(spatial_dim),
        "config_spatial_dim": (
            None if config_spatial_dim is None else int(config_spatial_dim)
        ),
        "nx": int(nx),
        "ny": int(ny),
        "model_in_channels": int(in_channels),
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "test_size": int(len(test_idx)),
        "train_effective_size": int(len(train_effective_idx)),
        "val_effective_size": int(len(val_effective_idx)),
        "test_effective_size": int(len(test_effective_idx)),
        "input_features": {
            "include_pos": bool(input_cfg.include_pos),
            "normalize_state_to_minus1_1": bool(
                input_cfg.normalize_state_to_minus1_1
            ),
            "normalize_clip": bool(input_cfg.normalize_clip),
            "state_min": (
                None if input_cfg.state_min is None else float(input_cfg.state_min)
            ),
            "state_max": (
                None if input_cfg.state_max is None else float(input_cfg.state_max)
            ),
        },
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "test_metrics": test_m.to_dict(),
    }
    save_json(summary, save_dir / "summary.json")
    save_json({"history": history}, save_dir / "history.json")
    save_json(
        {
            "train": train_idx.tolist(),
            "val": val_idx.tolist(),
            "test": test_idx.tolist(),
            "train_effective": train_effective_idx.tolist(),
            "val_effective": val_effective_idx.tolist(),
            "test_effective": test_effective_idx.tolist(),
            "train_trajectory_ids": (
                None if train_traj_ids is None else train_traj_ids.tolist()
            ),
            "val_trajectory_ids": (
                None if val_traj_ids is None else val_traj_ids.tolist()
            ),
            "test_trajectory_ids": (
                None if test_traj_ids is None else test_traj_ids.tolist()
            ),
        },
        save_dir / "split_indices.json",
    )

    print("Training complete.")
    print(f"  save_dir: {save_dir}")
    print(
        f"  training_mode: {'rollout' if rollout_mode else 'one_step'} "
        f"(rollout_steps={rollout_steps})"
    )
    print(f"  best_epoch: {best_epoch}")
    print(f"  best_val_loss: {best_val:.6e}")
    print(f"  test_loss: {test_m.loss:.6e}")
    print(f"  test_rmse: {test_m.rmse:.6e}")
    print(f"  test_rel_l2: {test_m.rel_l2:.6e}")


if __name__ == "__main__":
    main()
