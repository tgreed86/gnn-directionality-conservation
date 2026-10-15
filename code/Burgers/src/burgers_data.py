"""Dataset loader utilities for 1D Burgers GNN training.

Supported sources:
  1) Pre-bundled NPZ directory:
     burgers_data/
       metadata.json
       train_samples.npz
       val_samples.npz
       test_samples.npz
       trajectories/traj_XXXX.npz
  2) WENO-A generated PT trajectories:
     - single `.pt` file with keys: trajectories, dts, dx (and optional meta)
     - loaded in memory and converted to GNN tensors on the fly
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import json
import numpy as np
import torch
from torch.utils.data import Dataset


FEATURE_COLUMNS = ("x", "sin", "cos", "u", "nu")
SPLITS = ("train", "val", "test")
REGIME_FILTERS = ("all", "inviscid", "viscous")
SOURCE_FORMATS = ("auto", "npz_bundle", "wenoa_pt")


def build_periodic_ring_graph(
    N: int,
    *,
    dx: float = 1.0,
    dtype: torch.dtype = torch.float32,
    include_physical_displacement: bool = False,
    include_distance: bool = False,
) -> Tuple[torch.LongTensor, torch.Tensor]:
    """Build directed nearest-neighbor periodic ring graph.

    Edges per node:
      i -> (i + 1) mod N
      i -> (i - 1) mod N

    The first edge-attribute channel is signed normalized displacement (+1, -1).
    """
    if N <= 1:
        raise ValueError(f"N must be > 1, got {N}.")
    if dx <= 0.0:
        raise ValueError(f"dx must be positive, got {dx}.")

    sources = np.repeat(np.arange(N, dtype=np.int64), 2)
    targets = np.empty(2 * N, dtype=np.int64)

    targets[0::2] = (np.arange(N) + 1) % N
    targets[1::2] = (np.arange(N) - 1) % N

    edge_index = torch.tensor(np.stack([sources, targets], axis=0), dtype=torch.long)

    signed = np.empty((2 * N,), dtype=np.float64)
    signed[0::2] = 1.0
    signed[1::2] = -1.0

    attrs = [signed[:, None]]
    if include_physical_displacement:
        attrs.append((signed * dx)[:, None])
    if include_distance:
        attrs.append(np.abs(signed * dx)[:, None])

    edge_attr = torch.tensor(np.concatenate(attrs, axis=1), dtype=dtype)
    return edge_index, edge_attr


class _BaseBurgersDataset(Dataset):
    """Base dataset wrapper returning dictionary samples on a fixed ring graph."""

    def __init__(
        self,
        *,
        x_full: np.ndarray | torch.Tensor,
        target_delta_u: np.ndarray | torch.Tensor,
        u_next: np.ndarray | torch.Tensor,
        pos: np.ndarray | torch.Tensor,
        edge_index: np.ndarray | torch.Tensor,
        edge_attr: np.ndarray | torch.Tensor,
        pair_trajectory_id: np.ndarray | torch.Tensor,
        pair_time_index: np.ndarray | torch.Tensor,
        trajectory_ids: np.ndarray | torch.Tensor,
        rollout_states: np.ndarray | torch.Tensor | None,
        metadata: Mapping[str, Any],
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.x_full = torch.as_tensor(x_full, dtype=dtype)
        self.target_delta_u = torch.as_tensor(target_delta_u, dtype=dtype)
        self.u_next = torch.as_tensor(u_next, dtype=dtype)
        self.pos = torch.as_tensor(pos, dtype=dtype)
        self.edge_index = torch.as_tensor(edge_index, dtype=torch.long)
        self.edge_attr = torch.as_tensor(edge_attr, dtype=dtype)
        self.pair_trajectory_id = torch.as_tensor(pair_trajectory_id, dtype=torch.long)
        self.pair_time_index = torch.as_tensor(pair_time_index, dtype=torch.long)
        self.trajectory_ids = torch.as_tensor(trajectory_ids, dtype=torch.long)
        self.rollout_states = (
            None if rollout_states is None else torch.as_tensor(rollout_states, dtype=dtype)
        )
        self.metadata = dict(metadata)
        self.dtype = dtype

        if self.x_full.ndim != 3 or self.x_full.shape[-1] != len(FEATURE_COLUMNS):
            raise ValueError(
                "x_full must have shape [S, N, 5], got "
                f"{tuple(self.x_full.shape)}."
            )
        if self.target_delta_u.shape != (self.x_full.shape[0], self.x_full.shape[1], 1):
            raise ValueError(
                "target_delta_u must have shape [S,N,1], got "
                f"{tuple(self.target_delta_u.shape)}."
            )
        if self.u_next.shape != (self.x_full.shape[0], self.x_full.shape[1], 1):
            raise ValueError(
                "u_next must have shape [S,N,1], got "
                f"{tuple(self.u_next.shape)}."
            )

        n_nodes = int(self.x_full.shape[1])
        if self.pos.shape != (n_nodes, 1):
            raise ValueError(
                f"pos must have shape [N,1], got {tuple(self.pos.shape)} for N={n_nodes}."
            )
        if self.edge_index.shape != (2, 2 * n_nodes):
            raise ValueError(
                f"edge_index must have shape [2,2N], got {tuple(self.edge_index.shape)}."
            )
        if self.edge_attr.shape[0] != 2 * n_nodes:
            raise ValueError(
                f"edge_attr first dimension must be 2N, got {self.edge_attr.shape[0]}."
            )

        s = int(self.x_full.shape[0])
        if self.pair_trajectory_id.shape != (s,):
            raise ValueError(
                "pair_trajectory_id must have shape [S], got "
                f"{tuple(self.pair_trajectory_id.shape)} for S={s}."
            )
        if self.pair_time_index.shape != (s,):
            raise ValueError(
                "pair_time_index must have shape [S], got "
                f"{tuple(self.pair_time_index.shape)} for S={s}."
            )
        if self.trajectory_ids.ndim != 1:
            raise ValueError(
                "trajectory_ids must be 1D, got "
                f"{tuple(self.trajectory_ids.shape)}."
            )

        if self.rollout_states is not None:
            if self.rollout_states.ndim != 4 or self.rollout_states.shape[-1] != 1:
                raise ValueError(
                    "rollout_states must have shape [num_traj, Nt+1, N, 1], got "
                    f"{tuple(self.rollout_states.shape)}."
                )
            if self.rollout_states.shape[2] != n_nodes:
                raise ValueError(
                    "rollout_states N dimension must match pair N dimension, got "
                    f"{self.rollout_states.shape[2]} vs {n_nodes}."
                )
            if self.rollout_states.shape[0] != self.trajectory_ids.shape[0]:
                raise ValueError(
                    "rollout_states num_traj must match trajectory_ids length, got "
                    f"{self.rollout_states.shape[0]} vs {self.trajectory_ids.shape[0]}."
                )

    def __len__(self) -> int:
        return int(self.x_full.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        local_tid = int(self.pair_trajectory_id[idx].item())
        global_tid = int(self.trajectory_ids[local_tid].item())
        time_index = int(self.pair_time_index[idx].item())

        return {
            "x_full": self.x_full[idx],
            "target_delta_u": self.target_delta_u[idx],
            "u_next": self.u_next[idx],
            "pos": self.pos,
            "edge_index": self.edge_index,
            "edge_attr": self.edge_attr,
            "dt": float(self.metadata["dt"]),
            "dx": float(self.metadata["dx"]),
            "nu_values": self.metadata.get("nu_values", []),
            "sample_id": global_tid,
            "local_trajectory_id": local_tid,
            "pair_index": int(idx),
            "time_index": time_index,
        }


class LoadedBurgersDataset(_BaseBurgersDataset):
    """Dataset loaded from saved `.npz` split arrays and trajectory metadata."""


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_data_dir(data_dir: str | Path) -> Path:
    p = Path(data_dir)
    if p.is_file():
        raise ValueError(
            "data_dir must be a directory containing metadata.json and split files, "
            f"got file: {p}"
        )
    if not p.exists():
        raise FileNotFoundError(f"Dataset directory not found: {p}")
    return p


def _resolve_source_format(
    *,
    data_dir: str | Path,
    source_format: str,
) -> str:
    fmt = str(source_format).strip().lower()
    if fmt not in SOURCE_FORMATS:
        raise ValueError(f"source_format must be one of {SOURCE_FORMATS}, got '{source_format}'.")

    if fmt != "auto":
        return fmt

    p = Path(data_dir)
    if p.is_file() and p.suffix.lower() == ".pt":
        return "wenoa_pt"
    return "npz_bundle"


def _normalize_split_fractions(
    raw: Mapping[str, Any] | None,
) -> dict[str, float]:
    default = {"train": 0.8, "val": 0.1, "test": 0.1}
    if raw is None:
        return default

    out: dict[str, float] = {}
    for k in SPLITS:
        if k not in raw:
            raise ValueError(
                "split_fractions must include keys 'train', 'val', and 'test'. "
                f"Missing '{k}'."
            )
        out[k] = float(raw[k])
        if out[k] < 0.0:
            raise ValueError(f"split_fractions['{k}'] must be >= 0, got {out[k]}.")

    total = out["train"] + out["val"] + out["test"]
    if total <= 0.0:
        raise ValueError("split_fractions must have positive total weight.")

    return {k: (v / total) for k, v in out.items()}


def _counts_from_fractions(
    n_total: int,
    fractions: Mapping[str, float],
) -> dict[str, int]:
    raw = np.asarray([fractions["train"], fractions["val"], fractions["test"]], dtype=np.float64)
    exact = raw * float(n_total)
    counts = np.floor(exact).astype(np.int64)
    remainder = int(n_total - int(counts.sum()))

    if remainder > 0:
        frac_part = exact - counts.astype(np.float64)
        order = np.argsort(-frac_part)
        for i in range(remainder):
            counts[order[i % len(SPLITS)]] += 1

    out = {
        "train": int(counts[0]),
        "val": int(counts[1]),
        "test": int(counts[2]),
    }
    return out


def _split_trajectory_ids(
    *,
    n_trajectories: int,
    split_seed: int,
    split_fractions: Mapping[str, float],
) -> dict[str, np.ndarray]:
    if n_trajectories < 3:
        raise ValueError(
            "Need at least 3 trajectories to create non-empty train/val/test splits. "
            f"Got {n_trajectories}."
        )

    counts = _counts_from_fractions(n_trajectories, split_fractions)
    if any(counts[k] <= 0 for k in SPLITS):
        raise ValueError(
            "Split fractions produced an empty split. "
            f"counts={counts}, n_trajectories={n_trajectories}"
        )

    rng = np.random.default_rng(int(split_seed))
    perm = rng.permutation(n_trajectories).astype(np.int64)

    a = counts["train"]
    b = a + counts["val"]
    ids = {
        "train": np.sort(perm[:a]),
        "val": np.sort(perm[a:b]),
        "test": np.sort(perm[b:]),
    }
    return ids


def _to_numpy_f32(x: Any, *, name: str) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    return arr


def _load_wenoa_pt_dataset(
    *,
    pt_path: Path,
    split: str,
    include_rollout_states: bool,
    dtype: torch.dtype,
    source_options: Mapping[str, Any] | None,
) -> LoadedBurgersDataset:
    if not pt_path.exists():
        raise FileNotFoundError(f"WENO-A dataset file not found: {pt_path}")
    if not pt_path.is_file():
        raise ValueError(f"WENO-A dataset path must be a file, got: {pt_path}")
    if pt_path.suffix.lower() != ".pt":
        raise ValueError(f"WENO-A dataset file must end with .pt, got: {pt_path}")

    options = dict(source_options or {})
    split_key = str(split).strip().lower()
    if split_key not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got '{split}'.")

    payload = torch.load(pt_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError(f"WENO-A .pt payload must be a dict-like object, got {type(payload)}.")

    for key in ("trajectories", "dts", "dx"):
        if key not in payload:
            raise ValueError(f"WENO-A .pt is missing required key '{key}'.")

    trajectories = _to_numpy_f32(payload["trajectories"], name="trajectories")
    dts = _to_numpy_f32(payload["dts"], name="dts")
    dx = float(payload["dx"])
    if not np.isfinite(dx) or dx <= 0.0:
        raise ValueError(f"dx must be finite and > 0, got {dx}.")

    if trajectories.ndim != 3:
        raise ValueError(
            "WENO-A trajectories must have shape [num_traj, Nt+1, N], got "
            f"{trajectories.shape}."
        )
    n_traj_total, snapshots_per_trajectory, n_nodes = trajectories.shape
    if snapshots_per_trajectory < 2:
        raise ValueError(
            "WENO-A trajectories must have at least two snapshots (t0 and t1), got "
            f"{snapshots_per_trajectory}."
        )
    nt = snapshots_per_trajectory - 1

    if dts.ndim != 1 or dts.shape[0] != n_traj_total:
        raise ValueError(
            "WENO-A dts must have shape [num_traj], got "
            f"{dts.shape} with num_traj={n_traj_total}."
        )

    split_paths = options.get("split_paths")
    if isinstance(split_paths, Mapping):
        if split_key not in split_paths:
            raise ValueError(
                "source_options.split_paths is missing split "
                f"'{split_key}'. Available keys: {list(split_paths.keys())}."
            )
        # In split-path mode, each file is already a split-local subset.
        split_ids = np.arange(n_traj_total, dtype=np.int64)
        split_ids_all = {k: None for k in SPLITS}
    else:
        split_fractions = _normalize_split_fractions(options.get("split_fractions"))
        split_seed = int(options.get("split_seed", 1234))
        split_ids_all = _split_trajectory_ids(
            n_trajectories=n_traj_total,
            split_seed=split_seed,
            split_fractions=split_fractions,
        )
        split_ids = split_ids_all[split_key]

    if split_ids.size <= 0:
        raise ValueError(
            f"No trajectories assigned to split '{split_key}' from WENO-A source."
        )

    traj_split = trajectories[split_ids]  # [num_traj_split, Nt+1, N]
    dts_split = dts[split_ids]
    if not np.allclose(dts_split, dts_split[0], rtol=1e-6, atol=1e-12):
        raise ValueError(
            "WENO-A trajectories have varying dt across samples, but this GNN pipeline "
            "currently expects a single scalar dt per split."
        )
    dt = float(dts_split[0])
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and > 0, got {dt}.")

    # WENO-A stores cell-centered values on a uniform periodic grid.
    domain_length = float(payload.get("meta", {}).get("domain_length", float(n_nodes) * dx))
    if not np.isfinite(domain_length) or domain_length <= 0.0:
        raise ValueError(f"domain_length must be finite and > 0, got {domain_length}.")
    x = (np.arange(n_nodes, dtype=np.float32) + 0.5) * np.float32(dx)
    ang = (2.0 * np.pi * x / np.float32(domain_length)).astype(np.float32, copy=False)
    sin = np.sin(ang).astype(np.float32, copy=False)
    cos = np.cos(ang).astype(np.float32, copy=False)
    nu_value = float(options.get("nu_value", 0.0))

    u_now = traj_split[:, :-1, :]   # [num_traj_split, Nt, N]
    u_nxt = traj_split[:, 1:, :]    # [num_traj_split, Nt, N]
    delta = u_nxt - u_now

    num_traj_split = int(traj_split.shape[0])
    n_pairs = num_traj_split * nt
    u_now_pairs = u_now.reshape(n_pairs, n_nodes)
    u_next_pairs = u_nxt.reshape(n_pairs, n_nodes)
    delta_pairs = delta.reshape(n_pairs, n_nodes)

    x_full = np.empty((n_pairs, n_nodes, len(FEATURE_COLUMNS)), dtype=np.float32)
    x_full[..., 0] = x[None, :]
    x_full[..., 1] = sin[None, :]
    x_full[..., 2] = cos[None, :]
    x_full[..., 3] = u_now_pairs
    x_full[..., 4] = np.float32(nu_value)

    pair_trajectory_id = np.repeat(np.arange(num_traj_split, dtype=np.int64), nt)
    pair_time_index = np.tile(np.arange(nt, dtype=np.int64), num_traj_split)

    pos = x[:, None]
    edge_index, edge_attr = build_periodic_ring_graph(N=n_nodes, dx=dx, dtype=dtype)

    rollout_states = None
    if include_rollout_states:
        rollout_states = traj_split[..., None]  # [num_traj_split, Nt+1, N, 1]

    metadata: dict[str, Any] = {
        "source_format": "wenoa_pt",
        "source_path": str(pt_path),
        "split": split_key,
        "Nt": int(nt),
        "snapshots_per_trajectory": int(snapshots_per_trajectory),
        "dx": float(dx),
        "dt": float(dt),
        "domain_length": float(domain_length),
        "num_pairs": int(n_pairs),
        "num_trajectories_split": int(num_traj_split),
        "split_trajectory_ids": split_ids.tolist(),
        "feature_columns": list(FEATURE_COLUMNS),
        "nu_values": [float(nu_value)],
    }
    if isinstance(split_ids_all["train"], np.ndarray):
        metadata["split_ids"] = {
            "train": split_ids_all["train"].tolist(),
            "val": split_ids_all["val"].tolist(),
            "test": split_ids_all["test"].tolist(),
        }

    return LoadedBurgersDataset(
        x_full=x_full,
        target_delta_u=delta_pairs[..., None],
        u_next=u_next_pairs[..., None],
        pos=pos,
        edge_index=edge_index.detach().cpu().numpy(),
        edge_attr=edge_attr.detach().cpu().numpy(),
        pair_trajectory_id=pair_trajectory_id,
        pair_time_index=pair_time_index,
        trajectory_ids=split_ids.astype(np.int64, copy=False),
        rollout_states=rollout_states,
        metadata=metadata,
        dtype=dtype,
    )


def _load_rollout_states(
    *,
    data_dir: Path,
    trajectory_ids: np.ndarray,
    snapshots_per_trajectory: int,
    n_nodes: int,
) -> np.ndarray:
    traj_dir = data_dir / "trajectories"
    if not traj_dir.exists():
        raise FileNotFoundError(f"Trajectory directory not found: {traj_dir}")

    out = np.empty(
        (trajectory_ids.shape[0], snapshots_per_trajectory, n_nodes, 1),
        dtype=np.float32,
    )
    for i, tid in enumerate(trajectory_ids.tolist()):
        path = traj_dir / f"traj_{int(tid):04d}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing trajectory file: {path}")
        with np.load(path) as data:
            u = np.asarray(data["u"], dtype=np.float32)
        if u.shape != (snapshots_per_trajectory, n_nodes):
            raise ValueError(
                "Unexpected trajectory shape in "
                f"{path}: got {u.shape}, expected ({snapshots_per_trajectory}, {n_nodes})."
            )
        out[i, :, :, 0] = u
    return out


def load_burgers_dataset(
    data_dir: str | Path,
    *,
    split: str,
    include_rollout_states: bool = True,
    dtype: torch.dtype = torch.float32,
    source_format: str = "auto",
    source_options: Mapping[str, Any] | None = None,
) -> LoadedBurgersDataset:
    """Load one split from the provided Burgers dataset bundle.

    Args:
        data_dir: Path to dataset directory (`npz_bundle`) or `.pt` file (`wenoa_pt`).
        split: One of {'train', 'val', 'test'}.
        include_rollout_states: If True, load full per-trajectory rollout states
            for this split.
        source_format:
            - "auto": infer from `data_dir` (`.pt` => wenoa_pt, else npz_bundle)
            - "npz_bundle": load the pre-bundled split NPZ directory
            - "wenoa_pt": load a WENO-A `.pt` trajectory file in memory
        source_options: optional loader-specific options. For `wenoa_pt`:
            - split_fractions: {"train": float, "val": float, "test": float}
            - split_seed: int
            - nu_value: float (default 0.0)
            - split_paths: optional mapping {"train": "...pt", "val": "...pt", "test": "...pt"}
    """
    split_key = str(split).strip().lower()
    if split_key not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got '{split}'.")

    resolved_format = _resolve_source_format(data_dir=data_dir, source_format=source_format)
    if resolved_format == "wenoa_pt":
        options = dict(source_options or {})
        split_paths = options.get("split_paths")
        if isinstance(split_paths, Mapping):
            if split_key not in split_paths:
                raise ValueError(
                    "source_options.split_paths is missing split "
                    f"'{split_key}'. Available keys: {list(split_paths.keys())}."
                )
            pt_path = Path(split_paths[split_key])
        else:
            pt_path = Path(data_dir)
        return _load_wenoa_pt_dataset(
            pt_path=pt_path,
            split=split_key,
            include_rollout_states=include_rollout_states,
            dtype=dtype,
            source_options=source_options,
        )

    root = _resolve_data_dir(data_dir)
    meta_path = root / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {meta_path}")
    meta = _load_json(meta_path)

    sample_path = root / f"{split_key}_samples.npz"
    if not sample_path.exists():
        raise FileNotFoundError(f"Missing split file: {sample_path}")

    with np.load(sample_path) as npz:
        x_full = np.asarray(npz["input_node_features"], dtype=np.float32)
        target_delta_u = np.asarray(npz["target_delta_u"], dtype=np.float32)
        u_next = np.asarray(npz["target_next_u"], dtype=np.float32)

    if x_full.ndim != 3 or x_full.shape[-1] != len(FEATURE_COLUMNS):
        raise ValueError(
            f"input_node_features must have shape [S,N,5], got {x_full.shape}."
        )

    n_samples, n_nodes, _ = x_full.shape
    nt = int(meta.get("Nt", 200))
    snapshots_per_trajectory = int(meta.get("snapshots_per_trajectory", nt + 1))
    expected_nt = snapshots_per_trajectory - 1
    if nt != expected_nt:
        nt = expected_nt

    split_ids_raw = meta.get("split_ids", {}).get(split_key, None)
    if split_ids_raw is None:
        raise ValueError(f"metadata.json missing split_ids['{split_key}'].")
    trajectory_ids = np.asarray(split_ids_raw, dtype=np.int64)

    if trajectory_ids.size <= 0:
        raise ValueError(f"No trajectory ids found for split '{split_key}'.")
    if n_samples != int(trajectory_ids.size * nt):
        raise ValueError(
            "Sample count does not match split trajectory ids and Nt: "
            f"samples={n_samples}, n_traj={trajectory_ids.size}, Nt={nt}."
        )

    pair_trajectory_id = np.repeat(np.arange(trajectory_ids.size, dtype=np.int64), nt)
    pair_time_index = np.tile(np.arange(nt, dtype=np.int64), trajectory_ids.size)

    x_coords = x_full[0, :, 0]
    pos = x_coords[:, None]

    dx = float(meta.get("dx", 1.0 / n_nodes))
    edge_index, edge_attr = build_periodic_ring_graph(N=n_nodes, dx=dx, dtype=dtype)

    rollout_states = None
    if include_rollout_states:
        rollout_states = _load_rollout_states(
            data_dir=root,
            trajectory_ids=trajectory_ids,
            snapshots_per_trajectory=snapshots_per_trajectory,
            n_nodes=n_nodes,
        )

    dataset_meta = dict(meta)
    dataset_meta["split"] = split_key
    dataset_meta["num_pairs"] = int(n_samples)
    dataset_meta["num_trajectories_split"] = int(trajectory_ids.size)
    dataset_meta["split_trajectory_ids"] = trajectory_ids.tolist()
    dataset_meta["feature_columns"] = list(FEATURE_COLUMNS)

    return LoadedBurgersDataset(
        x_full=x_full,
        target_delta_u=target_delta_u,
        u_next=u_next,
        pos=pos,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pair_trajectory_id=pair_trajectory_id,
        pair_time_index=pair_time_index,
        trajectory_ids=trajectory_ids,
        rollout_states=rollout_states,
        metadata=dataset_meta,
        dtype=dtype,
    )


def filter_burgers_dataset_by_regime(
    dset: LoadedBurgersDataset,
    regime_filter: str,
) -> LoadedBurgersDataset:
    """Filter dataset to all / inviscid / viscous trajectories.

    Args:
        dset: Loaded split dataset.
        regime_filter:
            - "all": keep all trajectories/samples
            - "inviscid": keep only trajectories with nu == 0
            - "viscous": keep only trajectories with nu > 0
    """
    mode = str(regime_filter).strip().lower()
    if mode not in REGIME_FILTERS:
        raise ValueError(
            f"regime_filter must be one of {REGIME_FILTERS}, got '{regime_filter}'."
        )
    if mode == "all":
        return dset

    # Nu is constant per trajectory in this dataset.
    nu_by_pair = dset.x_full[:, 0, 4].detach().cpu().numpy()
    traj_local = dset.pair_trajectory_id.detach().cpu().numpy()
    num_traj = int(dset.trajectory_ids.shape[0])

    keep_local_traj: list[int] = []
    for t in range(num_traj):
        mask_t = traj_local == t
        if not np.any(mask_t):
            continue
        nu_t = float(nu_by_pair[np.nonzero(mask_t)[0][0]])
        if mode == "inviscid":
            keep = np.isclose(nu_t, 0.0)
        else:
            keep = nu_t > 0.0 and (not np.isclose(nu_t, 0.0))
        if keep:
            keep_local_traj.append(t)

    if len(keep_local_traj) == 0:
        raise ValueError(
            f"regime_filter='{mode}' produced an empty split for split='{dset.metadata.get('split')}'."
        )

    keep_local_traj_np = np.asarray(keep_local_traj, dtype=np.int64)
    pair_keep_mask = np.isin(traj_local, keep_local_traj_np)
    pair_idx = np.nonzero(pair_keep_mask)[0].astype(np.int64)
    if pair_idx.size == 0:
        raise ValueError(
            f"regime_filter='{mode}' produced zero pair samples for split='{dset.metadata.get('split')}'."
        )

    # Remap kept local trajectory ids to dense [0, ..., num_kept-1].
    remap = {old: new for new, old in enumerate(keep_local_traj)}
    pair_local_kept_old = traj_local[pair_idx]
    pair_local_kept_new = np.asarray([remap[int(v)] for v in pair_local_kept_old], dtype=np.int64)

    pair_time_kept = dset.pair_time_index.detach().cpu().numpy()[pair_idx]
    traj_ids_kept = dset.trajectory_ids.detach().cpu().numpy()[keep_local_traj_np]

    rollout_kept = None
    if dset.rollout_states is not None:
        rollout_kept = dset.rollout_states.detach().cpu().numpy()[keep_local_traj_np]

    meta = dict(dset.metadata)
    meta["regime_filter"] = mode
    meta["num_pairs_before_regime_filter"] = int(len(dset))
    meta["num_trajectories_before_regime_filter"] = int(dset.trajectory_ids.shape[0])
    meta["num_pairs"] = int(pair_idx.size)
    meta["num_trajectories_split"] = int(traj_ids_kept.shape[0])
    meta["split_trajectory_ids"] = traj_ids_kept.tolist()

    return LoadedBurgersDataset(
        x_full=dset.x_full.detach().cpu().numpy()[pair_idx],
        target_delta_u=dset.target_delta_u.detach().cpu().numpy()[pair_idx],
        u_next=dset.u_next.detach().cpu().numpy()[pair_idx],
        pos=dset.pos.detach().cpu().numpy(),
        edge_index=dset.edge_index.detach().cpu().numpy(),
        edge_attr=dset.edge_attr.detach().cpu().numpy(),
        pair_trajectory_id=pair_local_kept_new,
        pair_time_index=pair_time_kept,
        trajectory_ids=traj_ids_kept,
        rollout_states=rollout_kept,
        metadata=meta,
        dtype=dset.dtype,
    )
