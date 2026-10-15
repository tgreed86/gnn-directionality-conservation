#!/usr/bin/env python3
"""Compare Burgers GNN rollout performance against finite-volume baselines.

Typical usage:

    python scripts/compare_gnn_vs_godunov.py \
      --checkpoint runs/.../checkpoint_best.pt

Optional overrides:
    --data-dir <path>     # default: dataset path from checkpoint config
    --horizon <int>       # default: rollout_eval.horizon from config
    --num-samples <int>   # default: all test trajectories
    --device <name>       # auto | cpu | cuda | mps
    --allow-full-test-fallback
    --eval-pair-samples   # opt into legacy first-N pair-start evaluation
    --error-curves-log-y
    --error-curves-mean-only
    --error-curves-spread {none,std,iqr}
    --error-curves-smooth-window W
    --error-curves-metrics RMSE SSIM
    --ssim-mode local-periodic
    --ssim-data-range 1.0
    --error-curves-autoscale-source GNN or Godunov or etc.
    --curve-color gnn=black godunov=gray
    --baseline-schemes godunov muscl weno3
    --baseline-schemes none
    --gnn-label <name>
    --godunov-label <name>
    --compare-label <name> # legend names for --compare-eval-npz (repeat in order)
    --compare-eval-npz P   # overlay another model run from gnn_vs_godunov_eval.npz
    --plot-all-samples
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from burgers_data import filter_burgers_dataset_by_regime, load_burgers_dataset
from gnn_burgers import build_model_from_config, predict_type_from_target_mode, resolve_device, save_json
from input_features import (
    apply_u_minmax_minus1_1,
    count_input_feature_channels,
    resolve_input_feature_cfg,
    select_feature_channels,
    u_column_index,
)
from physics_inputs import (
    PhysicsInputConfig,
    build_physics_augmented_inputs_1d,
    count_physics_input_channels,
    resolve_physics_input_cfg,
)

BASELINE_SCHEME_GODUNOV = "godunov"
BASELINE_SCHEME_MUSCL = "muscl"
BASELINE_SCHEME_WENO3 = "weno3"
BASELINE_SCHEMES = (
    BASELINE_SCHEME_GODUNOV,
    BASELINE_SCHEME_MUSCL,
    BASELINE_SCHEME_WENO3,
)
BASELINE_SCHEME_DEFAULT_LABELS = {
    BASELINE_SCHEME_GODUNOV: "Godunov",
    BASELINE_SCHEME_MUSCL: "MUSCL",
    BASELINE_SCHEME_WENO3: "WENO-3",
}
_BASELINE_SCHEME_ALIASES = {
    "upwind": BASELINE_SCHEME_GODUNOV,
    "godunov-upwind": BASELINE_SCHEME_GODUNOV,
    "godunov_upwind": BASELINE_SCHEME_GODUNOV,
    "muscl2": BASELINE_SCHEME_MUSCL,
    "muscl-2": BASELINE_SCHEME_MUSCL,
    "muscl_2": BASELINE_SCHEME_MUSCL,
    "weno-3": BASELINE_SCHEME_WENO3,
    "weno_3": BASELINE_SCHEME_WENO3,
}

ERROR_CURVE_METRIC_RMSE = "rmse"
ERROR_CURVE_METRIC_REL_L2 = "rel"
ERROR_CURVE_METRIC_L1 = "l1"
ERROR_CURVE_METRIC_SSIM = "ssim"
ERROR_CURVE_METRICS = (
    ERROR_CURVE_METRIC_RMSE,
    ERROR_CURVE_METRIC_REL_L2,
    ERROR_CURVE_METRIC_L1,
    ERROR_CURVE_METRIC_SSIM,
)
ERROR_CURVE_METRIC_ALIASES = {
    "rmse": ERROR_CURVE_METRIC_RMSE,
    "rel": ERROR_CURVE_METRIC_REL_L2,
    "rell2": ERROR_CURVE_METRIC_REL_L2,
    "relative": ERROR_CURVE_METRIC_REL_L2,
    "relativel2": ERROR_CURVE_METRIC_REL_L2,
    "l2rel": ERROR_CURVE_METRIC_REL_L2,
    "l2relative": ERROR_CURVE_METRIC_REL_L2,
    "l1": ERROR_CURVE_METRIC_L1,
    "mae": ERROR_CURVE_METRIC_L1,
    "meanabsoluteerror": ERROR_CURVE_METRIC_L1,
    "ssim": ERROR_CURVE_METRIC_SSIM,
}

ERROR_CURVE_SPREAD_NONE = "none"
ERROR_CURVE_SPREAD_STD = "std"
ERROR_CURVE_SPREAD_IQR = "iqr"
ERROR_CURVE_SPREAD_MODES = (
    ERROR_CURVE_SPREAD_NONE,
    ERROR_CURVE_SPREAD_STD,
    ERROR_CURVE_SPREAD_IQR,
)

SSIM_MODE_LEGACY_GLOBAL = "legacy-global"
SSIM_MODE_LOCAL_PERIODIC = "local-periodic"
SSIM_MODES = (SSIM_MODE_LEGACY_GLOBAL, SSIM_MODE_LOCAL_PERIODIC)
SSIM_DEFAULT_WINDOW_SIZE = 11
SSIM_DEFAULT_WINDOW_SIGMA = 1.5
SSIM_DEFAULT_K1 = 0.01
SSIM_DEFAULT_K2 = 0.03


@dataclass(frozen=True)
class SSIMConfig:
    """Resolved SSIM settings used consistently for every evaluated method."""

    mode: str = SSIM_MODE_LEGACY_GLOBAL
    data_range: float | None = None
    data_range_source: str = "prediction-pair"
    window_size: int = SSIM_DEFAULT_WINDOW_SIZE
    window_sigma: float = SSIM_DEFAULT_WINDOW_SIGMA
    k1: float = SSIM_DEFAULT_K1
    k2: float = SSIM_DEFAULT_K2

    @property
    def display_name(self) -> str:
        if self.mode == SSIM_MODE_LOCAL_PERIODIC:
            return "Local SSIM"
        return "SSIM"


def _replace_nonfinite_with_nan(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float64).copy()
    out[~np.isfinite(out)] = np.nan
    return out


def _nanmean_over_samples(arr: np.ndarray, horizon: int) -> np.ndarray:
    if arr.shape[0] == 0:
        return np.zeros((horizon,), dtype=np.float64)
    out = np.full((horizon,), np.nan, dtype=np.float64)
    for j in range(horizon):
        col = arr[:, j]
        finite = col[np.isfinite(col)]
        if finite.size > 0:
            out[j] = float(finite.mean())
    return out


def _safe_nanmean_last(arr: np.ndarray) -> float | None:
    if arr.size == 0:
        return None
    val = float(np.nanmean(arr[:, -1]))
    return val if np.isfinite(val) else None


def _final_rmse_win_fractions(
    gnn_rmse: np.ndarray,
    baseline_rmse: np.ndarray,
) -> tuple[float, float, float]:
    finite_mask = np.isfinite(gnn_rmse[:, -1]) & np.isfinite(baseline_rmse[:, -1])
    if not np.any(finite_mask):
        return float("nan"), float("nan"), float("nan")
    gnn_final = gnn_rmse[finite_mask, -1]
    baseline_final = baseline_rmse[finite_mask, -1]
    gnn_better = gnn_final < baseline_final
    baseline_better = baseline_final < gnn_final
    ties = ~(gnn_better | baseline_better)
    return (
        float(np.mean(gnn_better)),
        float(np.mean(baseline_better)),
        float(np.mean(ties)),
    )


def _moving_average_nan_1d(arr: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float64).reshape(-1)
    w = max(int(window), 1)
    if w <= 1 or x.size == 0:
        return x.copy()
    kernel = np.ones((w,), dtype=np.float64)
    mask = np.isfinite(x).astype(np.float64)
    vals = np.where(np.isfinite(x), x, 0.0)
    num = np.convolve(vals, kernel, mode="same")
    den = np.convolve(mask, kernel, mode="same")
    out = np.full_like(x, np.nan, dtype=np.float64)
    valid = den > 0.0
    out[valid] = num[valid] / den[valid]
    return out


def _smooth_curves(arr: np.ndarray, *, window: int) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    w = max(int(window), 1)
    if (w % 2) == 0:
        w += 1
    if w <= 1:
        return a.copy()
    if a.ndim == 1:
        return _moving_average_nan_1d(a, w)
    if a.ndim == 2:
        out = np.empty_like(a, dtype=np.float64)
        for i in range(a.shape[0]):
            out[i] = _moving_average_nan_1d(a[i], w)
        return out
    raise ValueError(f"_smooth_curves expects 1D/2D arrays, got {tuple(a.shape)}.")


def _error_curve_spread(
    arr: np.ndarray,
    *,
    mode: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return NaN-aware per-time spread bounds for sample curves."""
    if mode == ERROR_CURVE_SPREAD_NONE:
        return None
    if mode not in ERROR_CURVE_SPREAD_MODES:
        raise ValueError(
            f"Unsupported error-curve spread mode {mode!r}; "
            f"expected one of {ERROR_CURVE_SPREAD_MODES}."
        )

    values = np.asarray(arr, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(
            "Error-curve spread requires a [sample, time] array, "
            f"got shape {tuple(values.shape)}."
        )
    lower = np.full((values.shape[1],), np.nan, dtype=np.float64)
    upper = np.full((values.shape[1],), np.nan, dtype=np.float64)
    for time_index in range(values.shape[1]):
        finite = values[:, time_index]
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            continue
        if mode == ERROR_CURVE_SPREAD_STD:
            center = float(np.mean(finite))
            spread = float(np.std(finite, ddof=0))
            lower[time_index] = center - spread
            upper[time_index] = center + spread
        else:
            lower[time_index], upper[time_index] = np.percentile(
                finite,
                (25.0, 75.0),
            )
    return lower, upper


def _rmse_1d(pred: np.ndarray, true: np.ndarray) -> float:
    p = np.asarray(pred, dtype=np.float64).reshape(-1)
    t = np.asarray(true, dtype=np.float64).reshape(-1)
    mask = np.isfinite(p) & np.isfinite(t)
    if np.count_nonzero(mask) != p.size:
        return float("nan")
    err = p[mask] - t[mask]
    if not np.all(np.isfinite(err)):
        return float("nan")
    scale = float(np.max(np.abs(err))) if err.size > 0 else 0.0
    if scale <= 0.0:
        return 0.0
    return float(scale * np.sqrt(np.mean((err / scale) ** 2)))


def _safe_norm_1d(arr: np.ndarray) -> float:
    x = np.asarray(arr, dtype=np.float64).reshape(-1)
    if x.size == 0 or not np.all(np.isfinite(x)):
        return float("nan")
    scale = float(np.max(np.abs(x)))
    if scale <= 0.0:
        return 0.0
    return float(scale * np.linalg.norm(x / scale))


def _ssim_1d_global(
    pred: np.ndarray,
    true: np.ndarray,
    *,
    data_range: float | None = None,
    k1: float = 0.01,
    k2: float = 0.03,
    eps: float = 1e-12,
) -> float:
    x = np.asarray(true, dtype=np.float64).reshape(-1)
    y = np.asarray(pred, dtype=np.float64).reshape(-1)
    if x.shape != y.shape:
        return float("nan")
    mask = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(mask) != x.size or np.count_nonzero(mask) < 2:
        return float("nan")
    x = x[mask]
    y = y[mask]

    max_abs = max(float(np.max(np.abs(x))), float(np.max(np.abs(y))))
    # Extremely large but still finite values can overflow Python float
    # arithmetic while computing SSIM constants. Treat those trajectories as
    # failed instead of letting diagnostics crash.
    if not np.isfinite(max_abs) or max_abs > 1.0e150:
        return float("nan")

    with np.errstate(over="ignore", invalid="ignore"):
        mu_x = float(np.mean(x))
        mu_y = float(np.mean(y))
        x_centered = x - mu_x
        y_centered = y - mu_y
        var_x = float(np.mean(x_centered * x_centered))
        var_y = float(np.mean(y_centered * y_centered))
        cov_xy = float(np.mean(x_centered * y_centered))
    if not all(np.isfinite(v) for v in (mu_x, mu_y, var_x, var_y, cov_xy)):
        return float("nan")

    if data_range is None or not np.isfinite(data_range):
        lo = min(float(np.min(x)), float(np.min(y)))
        hi = max(float(np.max(x)), float(np.max(y)))
        dyn = hi - lo
    else:
        dyn = float(data_range)
    dyn = max(abs(dyn), float(eps))
    if not np.isfinite(dyn) or dyn > 1.0e150:
        return float("nan")

    try:
        c1 = (float(k1) * dyn) ** 2
        c2 = (float(k2) * dyn) ** 2
        denom = (mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2)
    except OverflowError:
        return float("nan")
    if not np.isfinite(denom):
        return float("nan")
    if abs(denom) <= float(eps):
        return float("nan")
    with np.errstate(over="ignore", invalid="ignore"):
        num = (2.0 * mu_x * mu_y + c1) * (2.0 * cov_xy + c2)
    if not np.isfinite(num):
        return float("nan")
    return float(num / denom)


def _gaussian_window_1d(*, window_size: int, sigma: float) -> np.ndarray:
    """Return normalized symmetric Gaussian weights for an odd 1D window."""
    size = int(window_size)
    width = float(sigma)
    if size < 1 or (size % 2) == 0:
        raise ValueError(f"SSIM window size must be a positive odd integer, got {size}.")
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError(f"SSIM window sigma must be finite and positive, got {sigma}.")
    offsets = np.arange(-(size // 2), size // 2 + 1, dtype=np.float64)
    weights = np.exp(-0.5 * (offsets / width) ** 2)
    return weights / float(np.sum(weights))


def _ssim_1d_local_periodic(
    pred: np.ndarray,
    true: np.ndarray,
    *,
    data_range: float,
    window_size: int = SSIM_DEFAULT_WINDOW_SIZE,
    window_sigma: float = SSIM_DEFAULT_WINDOW_SIGMA,
    k1: float = SSIM_DEFAULT_K1,
    k2: float = SSIM_DEFAULT_K2,
) -> float:
    """Mean local 1D SSIM with Gaussian weights and periodic boundaries.

    Local statistics use population-weighted variance and covariance. A
    nonfinite entry invalidates the complete spatial time slice, matching the
    Burgers evaluator's existing numerical-failure policy.
    """
    x = np.asarray(true, dtype=np.float64).reshape(-1)
    y = np.asarray(pred, dtype=np.float64).reshape(-1)
    if x.shape != y.shape or x.size < 2:
        return float("nan")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        return float("nan")

    dyn = float(data_range)
    if not np.isfinite(dyn) or dyn <= 0.0:
        raise ValueError(f"SSIM data range must be finite and positive, got {data_range}.")
    if int(window_size) > int(x.size):
        raise ValueError(
            f"SSIM window size ({window_size}) cannot exceed the number of "
            f"spatial nodes ({x.size})."
        )

    weights = _gaussian_window_1d(
        window_size=int(window_size),
        sigma=float(window_sigma),
    )
    max_abs = max(float(np.max(np.abs(x))), float(np.max(np.abs(y))))
    if not np.isfinite(max_abs) or max_abs > 1.0e150 or dyn > 1.0e150:
        return float("nan")

    offsets = np.arange(-(len(weights) // 2), len(weights) // 2 + 1)
    x_neighbors = np.stack([np.roll(x, -int(offset)) for offset in offsets], axis=0)
    y_neighbors = np.stack([np.roll(y, -int(offset)) for offset in offsets], axis=0)
    weight_col = weights[:, None]

    with np.errstate(over="ignore", invalid="ignore"):
        mu_x = np.sum(weight_col * x_neighbors, axis=0)
        mu_y = np.sum(weight_col * y_neighbors, axis=0)
        x_centered = x_neighbors - mu_x[None, :]
        y_centered = y_neighbors - mu_y[None, :]
        var_x = np.sum(weight_col * x_centered * x_centered, axis=0)
        var_y = np.sum(weight_col * y_centered * y_centered, axis=0)
        cov_xy = np.sum(weight_col * x_centered * y_centered, axis=0)
    if not all(
        np.all(np.isfinite(values))
        for values in (mu_x, mu_y, var_x, var_y, cov_xy)
    ):
        return float("nan")

    try:
        c1 = (float(k1) * dyn) ** 2
        c2 = (float(k2) * dyn) ** 2
    except OverflowError:
        return float("nan")
    if not (np.isfinite(c1) and np.isfinite(c2)):
        return float("nan")

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * cov_xy + c2)
        denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (
            var_x + var_y + c2
        )
    if not (
        np.all(np.isfinite(numerator))
        and np.all(np.isfinite(denominator))
        and np.all(denominator > 0.0)
    ):
        return float("nan")
    score = float(np.mean(numerator / denominator))
    return score if np.isfinite(score) else float("nan")


def _ssim_1d(pred: np.ndarray, true: np.ndarray, *, config: SSIMConfig) -> float:
    """Dispatch to the selected backward-compatible SSIM definition."""
    if config.mode == SSIM_MODE_LEGACY_GLOBAL:
        return _ssim_1d_global(
            pred,
            true,
            k1=config.k1,
            k2=config.k2,
        )
    if config.mode == SSIM_MODE_LOCAL_PERIODIC:
        if config.data_range is None:
            raise ValueError("Local periodic SSIM requires a fixed data range.")
        return _ssim_1d_local_periodic(
            pred,
            true,
            data_range=config.data_range,
            window_size=config.window_size,
            window_sigma=config.window_sigma,
            k1=config.k1,
            k2=config.k2,
        )
    raise ValueError(f"Unsupported SSIM mode '{config.mode}'. Expected one of {SSIM_MODES}.")


def _fmt_metric(value: float, fmt: str) -> str:
    return format(value, fmt) if np.isfinite(value) else "nan"


@dataclass
class ExtraEvalCurves:
    """Optional comparison model curves loaded from an eval NPZ artifact."""

    label: str
    source_npz: str
    horizon: int
    rmse_by_sample: np.ndarray
    rel_by_sample: np.ndarray
    l1_by_sample: np.ndarray
    ssim_by_sample: np.ndarray
    mean_rmse: np.ndarray
    mean_rel: np.ndarray
    mean_l1: np.ndarray
    mean_ssim: np.ndarray
    sample_gnn_traj_by_sample: np.ndarray | None = None
    sample_global_traj_id: np.ndarray | None = None
    sample_time_index: np.ndarray | None = None
    sample_valid_steps: np.ndarray | None = None
    ssim_mode: str = SSIM_MODE_LEGACY_GLOBAL
    ssim_data_range: float | None = None
    ssim_window_size: int | None = None
    ssim_window_sigma: float | None = None


@dataclass
class BaselineEvalCurves:
    """Curves for a finite-volume Burgers baseline computed for this run."""

    scheme: str
    label: str
    rmse_by_sample: np.ndarray
    rel_by_sample: np.ndarray
    l1_by_sample: np.ndarray
    ssim_by_sample: np.ndarray
    mean_rmse: np.ndarray
    mean_rel: np.ndarray
    mean_l1: np.ndarray
    mean_ssim: np.ndarray


def _metric_2d_or_raise(arr: np.ndarray, *, key: str, npz_path: Path) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float64)
    if out.ndim == 1:
        out = out[None, :]
    if out.ndim != 2:
        raise ValueError(
            f"{npz_path}: key '{key}' must be 1D/2D, got shape {tuple(out.shape)}."
        )
    return _replace_nonfinite_with_nan(out)


def _reserve_unique_label(base: str, used_labels: set[str]) -> str:
    root = str(base).strip()
    if root == "":
        root = "model"
    label = root
    i = 2
    while label in used_labels:
        label = f"{root} ({i})"
        i += 1
    used_labels.add(label)
    return label


def _derive_extra_label(npz_path: Path, used_labels: set[str]) -> str:
    base = npz_path.parent.name.strip() if npz_path.parent.name else npz_path.stem
    if base == "":
        base = npz_path.stem
    return _reserve_unique_label(base, used_labels)


def normalize_baseline_scheme(scheme: str) -> str:
    s = str(scheme).strip().lower()
    if s in BASELINE_SCHEMES:
        return s
    if s in _BASELINE_SCHEME_ALIASES:
        return _BASELINE_SCHEME_ALIASES[s]
    allowed = sorted(set(BASELINE_SCHEMES).union(_BASELINE_SCHEME_ALIASES))
    raise ValueError(f"baseline scheme must be one of {allowed}, got '{scheme}'.")


def _resolve_baseline_schemes(raw_schemes: List[str] | None) -> List[str]:
    raw = [BASELINE_SCHEME_GODUNOV] if raw_schemes is None else list(raw_schemes)
    out: List[str] = []
    disabled = False
    for raw_item in raw:
        for part in str(raw_item).split(","):
            token = part.strip()
            if token == "":
                continue
            token_lower = token.lower()
            if token_lower in {"none", "off", "no-baselines", "no_baselines"}:
                if len(out) > 0:
                    raise ValueError(
                        "--baseline-schemes none cannot be combined with other schemes."
                    )
                disabled = True
                continue
            if disabled:
                raise ValueError(
                    "--baseline-schemes none cannot be combined with other schemes."
                )
            if token_lower == "all":
                for scheme in BASELINE_SCHEMES:
                    if scheme not in out:
                        out.append(scheme)
                continue
            scheme = normalize_baseline_scheme(token)
            if scheme not in out:
                out.append(scheme)
    if disabled:
        return []
    if len(out) == 0:
        raise ValueError("At least one baseline scheme must be requested.")
    return out


def normalize_error_curve_metric(metric: str) -> str:
    key = "".join(ch for ch in str(metric).strip().lower() if ch.isalnum())
    if key == "all":
        return "all"
    if key in ERROR_CURVE_METRIC_ALIASES:
        return ERROR_CURVE_METRIC_ALIASES[key]
    allowed = sorted(set(ERROR_CURVE_METRICS).union(ERROR_CURVE_METRIC_ALIASES))
    raise ValueError(
        f"error curve metric must be one of {allowed} or 'all', got '{metric}'."
    )


def _resolve_error_curve_metrics(raw_metrics: List[str] | None) -> List[str]:
    raw = ["all"] if raw_metrics is None else list(raw_metrics)
    out: List[str] = []
    saw_all = False
    for raw_item in raw:
        for part in str(raw_item).split(","):
            token = part.strip()
            if token == "":
                continue
            metric = normalize_error_curve_metric(token)
            if metric == "all":
                if len(out) > 0:
                    raise ValueError(
                        "--error-curves-metrics all cannot be combined with "
                        "specific metrics."
                    )
                saw_all = True
                continue
            if saw_all:
                raise ValueError(
                    "--error-curves-metrics all cannot be combined with "
                    "specific metrics."
                )
            if metric not in out:
                out.append(metric)
    if saw_all:
        return list(ERROR_CURVE_METRICS)
    if len(out) == 0:
        raise ValueError("At least one error curve metric must be requested.")
    return out


def _normalize_curve_color_label_key(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def _compare_curve_color_key(raw_key: str) -> str | None:
    key = str(raw_key).strip().lower()
    for prefix in ("compare", "comparison"):
        for sep in ("", "-", "_"):
            root = f"{prefix}{sep}"
            if key.startswith(root):
                suffix = key[len(root) :]
                if suffix.isdigit():
                    idx = int(suffix)
                    if idx >= 1:
                        return f"compare:{idx - 1}"
    return None


def _canonical_curve_color_key(raw_key: str) -> str:
    key = str(raw_key).strip()
    if key == "":
        raise ValueError("--curve-color keys must be non-empty.")
    key_lower = key.lower()
    if key_lower in {"gnn", "model", "current-model", "current_model"}:
        return "gnn"

    compare_key = _compare_curve_color_key(key)
    if compare_key is not None:
        return compare_key

    try:
        return normalize_baseline_scheme(key)
    except ValueError:
        return f"label:{_normalize_curve_color_label_key(key)}"


def _parse_curve_color_specs(raw_specs: List[str] | None) -> List[Tuple[str, str]]:
    if raw_specs is None:
        return []

    specs: List[Tuple[str, str]] = []
    for raw_item in raw_specs:
        text = str(raw_item).strip()
        if text == "":
            continue
        if "=" not in text:
            raise ValueError(
                "--curve-color entries must be KEY=COLOR pairs, "
                f"got '{text}'."
            )
        raw_key, raw_color = text.split("=", 1)
        key = raw_key.strip()
        color = raw_color.strip()
        if key == "" or color == "":
            raise ValueError(
                "--curve-color entries must have non-empty KEY and COLOR "
                f"values, got '{text}'."
            )
        specs.append((_canonical_curve_color_key(key), color))
    return specs


def _resolve_curve_color_overrides(
    raw_specs: List[str] | None,
    *,
    gnn_label: str,
    baseline_label_by_scheme: Dict[str, str],
    extra_eval_curves: List[ExtraEvalCurves],
) -> Dict[str, str]:
    parsed_specs = _parse_curve_color_specs(raw_specs)
    if len(parsed_specs) == 0:
        return {}

    known_keys: Dict[str, str] = {
        "gnn": "gnn",
        f"label:{_normalize_curve_color_label_key(gnn_label)}": "gnn",
    }
    for scheme in BASELINE_SCHEMES:
        known_keys[scheme] = scheme
    for scheme, label in baseline_label_by_scheme.items():
        known_keys[scheme] = scheme
        known_keys[f"label:{_normalize_curve_color_label_key(label)}"] = scheme
    for idx, extra in enumerate(extra_eval_curves):
        compare_key = f"compare:{idx}"
        known_keys[compare_key] = compare_key
        known_keys[f"label:{_normalize_curve_color_label_key(extra.label)}"] = compare_key

    resolved: Dict[str, str] = {}
    for key, color in parsed_specs:
        resolved_key = known_keys.get(key)
        if resolved_key is None:
            allowed = [
                "gnn",
                *BASELINE_SCHEMES,
                *[f"compare{i + 1}" for i in range(len(extra_eval_curves))],
                gnn_label,
                *baseline_label_by_scheme.values(),
                *[extra.label for extra in extra_eval_curves],
            ]
            allowed_desc = ", ".join(str(v) for v in allowed if str(v).strip() != "")
            raise ValueError(
                f"Unknown --curve-color key '{key}'. Expected one of: "
                f"{allowed_desc}."
            )
        resolved[resolved_key] = color
    return resolved


def _label_arg_for_baseline(args: argparse.Namespace, scheme: str) -> str:
    if scheme == BASELINE_SCHEME_GODUNOV:
        return str(args.godunov_label)
    if scheme == BASELINE_SCHEME_MUSCL:
        return str(args.muscl_label)
    if scheme == BASELINE_SCHEME_WENO3:
        return str(args.weno3_label)
    return BASELINE_SCHEME_DEFAULT_LABELS.get(scheme, scheme)


def _load_extra_eval_curves(npz_path: Path, *, label: str) -> ExtraEvalCurves:
    with np.load(npz_path) as data:
        required = ("rmse_gnn_by_sample", "rel_l2_gnn_by_sample", "l1_gnn_by_sample")
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(
                f"{npz_path}: missing required key(s): {', '.join(missing)}."
            )

        rmse = _metric_2d_or_raise(
            data["rmse_gnn_by_sample"], key="rmse_gnn_by_sample", npz_path=npz_path
        )
        rel = _metric_2d_or_raise(
            data["rel_l2_gnn_by_sample"], key="rel_l2_gnn_by_sample", npz_path=npz_path
        )
        l1 = _metric_2d_or_raise(
            data["l1_gnn_by_sample"], key="l1_gnn_by_sample", npz_path=npz_path
        )
        if "ssim_gnn_by_sample" in data:
            ssim = _metric_2d_or_raise(
                data["ssim_gnn_by_sample"], key="ssim_gnn_by_sample", npz_path=npz_path
            )
        else:
            ssim = np.full_like(rmse, np.nan, dtype=np.float64)
        ssim_mode = (
            str(np.asarray(data["ssim_mode"]).reshape(()).item())
            if "ssim_mode" in data
            else SSIM_MODE_LEGACY_GLOBAL
        )
        ssim_data_range = None
        if "ssim_data_range" in data:
            stored_range = float(np.asarray(data["ssim_data_range"]).reshape(()))
            if np.isfinite(stored_range):
                ssim_data_range = stored_range
        ssim_window_size = None
        if "ssim_window_size" in data:
            stored_window_size = int(
                np.asarray(data["ssim_window_size"]).reshape(())
            )
            if stored_window_size > 0:
                ssim_window_size = stored_window_size
        ssim_window_sigma = None
        if "ssim_window_sigma" in data:
            stored_window_sigma = float(
                np.asarray(data["ssim_window_sigma"]).reshape(())
            )
            if np.isfinite(stored_window_sigma):
                ssim_window_sigma = stored_window_sigma

        sample_gnn_traj_by_sample: np.ndarray | None = None
        sample_global_traj_id: np.ndarray | None = None
        sample_time_index: np.ndarray | None = None
        sample_valid_steps: np.ndarray | None = None
        if "sample_gnn_traj_by_sample" in data:
            traj = np.asarray(data["sample_gnn_traj_by_sample"], dtype=np.float64)
            if traj.ndim != 3:
                raise ValueError(
                    f"{npz_path}: key 'sample_gnn_traj_by_sample' must be rank-3 "
                    f"[S,T,N], got shape {tuple(traj.shape)}."
                )
            n_samp = int(traj.shape[0])
            sample_gnn_traj_by_sample = _replace_nonfinite_with_nan(traj)

            if "sample_valid_steps" in data:
                sample_valid_steps = np.asarray(
                    data["sample_valid_steps"], dtype=np.int64
                ).reshape(-1)
                if int(sample_valid_steps.shape[0]) != n_samp:
                    raise ValueError(
                        f"{npz_path}: sample_valid_steps length {sample_valid_steps.shape[0]} "
                        f"does not match sample_gnn_traj_by_sample samples {n_samp}."
                    )
            if "sample_global_traj_id" in data:
                sample_global_traj_id = np.asarray(
                    data["sample_global_traj_id"], dtype=np.int64
                ).reshape(-1)
                if int(sample_global_traj_id.shape[0]) != n_samp:
                    raise ValueError(
                        f"{npz_path}: sample_global_traj_id length {sample_global_traj_id.shape[0]} "
                        f"does not match sample_gnn_traj_by_sample samples {n_samp}."
                    )
            if "sample_time_index" in data:
                sample_time_index = np.asarray(
                    data["sample_time_index"], dtype=np.int64
                ).reshape(-1)
                if int(sample_time_index.shape[0]) != n_samp:
                    raise ValueError(
                        f"{npz_path}: sample_time_index length {sample_time_index.shape[0]} "
                        f"does not match sample_gnn_traj_by_sample samples {n_samp}."
                    )

    h = min(rmse.shape[1], rel.shape[1], l1.shape[1], ssim.shape[1])
    if h < 1:
        raise ValueError(f"{npz_path}: no time dimension found in metric arrays.")
    if any(v != h for v in (rmse.shape[1], rel.shape[1], l1.shape[1], ssim.shape[1])):
        print(
            f"[WARN] {npz_path}: metric horizons differ; truncating all to horizon={h}.",
            flush=True,
        )

    rmse = rmse[:, :h]
    rel = rel[:, :h]
    l1 = l1[:, :h]
    ssim = ssim[:, :h]

    return ExtraEvalCurves(
        label=label,
        source_npz=str(npz_path),
        horizon=int(h),
        rmse_by_sample=rmse,
        rel_by_sample=rel,
        l1_by_sample=l1,
        ssim_by_sample=ssim,
        mean_rmse=_nanmean_over_samples(rmse, h),
        mean_rel=_nanmean_over_samples(rel, h),
        mean_l1=_nanmean_over_samples(l1, h),
        mean_ssim=_nanmean_over_samples(ssim, h),
        sample_gnn_traj_by_sample=sample_gnn_traj_by_sample,
        sample_global_traj_id=sample_global_traj_id,
        sample_time_index=sample_time_index,
        sample_valid_steps=sample_valid_steps,
        ssim_mode=ssim_mode,
        ssim_data_range=ssim_data_range,
        ssim_window_size=ssim_window_size,
        ssim_window_sigma=ssim_window_sigma,
    )


def _ssim_comparison_incompatibility(
    primary: SSIMConfig,
    extra: ExtraEvalCurves,
) -> str | None:
    if extra.ssim_mode != primary.mode:
        return f"mode '{extra.ssim_mode}' versus '{primary.mode}'"
    if primary.mode != SSIM_MODE_LOCAL_PERIODIC:
        return None
    if (
        extra.ssim_data_range is None
        or extra.ssim_window_size is None
        or extra.ssim_window_sigma is None
    ):
        return "missing local-SSIM metadata in the comparison artifact"
    assert primary.data_range is not None
    if not np.isclose(extra.ssim_data_range, primary.data_range, rtol=1e-12, atol=0.0):
        return (
            f"data range {extra.ssim_data_range:.9g} versus "
            f"{primary.data_range:.9g}"
        )
    if extra.ssim_window_size != primary.window_size:
        return f"window size {extra.ssim_window_size} versus {primary.window_size}"
    if not np.isclose(
        extra.ssim_window_sigma,
        primary.window_sigma,
        rtol=1e-12,
        atol=0.0,
    ):
        return (
            f"window sigma {extra.ssim_window_sigma:g} versus "
            f"{primary.window_sigma:g}"
        )
    return None


def _warn_if_ssim_comparison_incompatible(
    primary: SSIMConfig,
    extra: ExtraEvalCurves,
) -> bool:
    incompatibility = _ssim_comparison_incompatibility(primary, extra)
    if incompatibility is None:
        return False
    print(
        f"[WARN] Comparison model '{extra.label}' has incompatible SSIM "
        f"settings ({incompatibility}). Its overlaid SSIM curve is not directly "
        "comparable to the primary curve.",
        flush=True,
    )
    return True


def _godunov_flux_burgers(u_l: np.ndarray, u_r: np.ndarray) -> np.ndarray:
    """Exact scalar Godunov flux for inviscid Burgers with f(u)=u^2/2."""
    with np.errstate(over="ignore", invalid="ignore"):
        f_l = 0.5 * u_l * u_l
        f_r = 0.5 * u_r * u_r
        shock_speed = 0.5 * (u_l + u_r)
        shock_flux = np.where(shock_speed >= 0.0, f_l, f_r)
        rare_flux = np.where(u_l >= 0.0, f_l, np.where(u_r <= 0.0, f_r, 0.0))
        return np.where(u_l >= u_r, shock_flux, rare_flux)


def _godunov_step_burgers(u: np.ndarray, *, dt: float, dx: float) -> np.ndarray:
    u_l = u
    u_r = np.roll(u, -1)
    f_iph = _godunov_flux_burgers(u_l, u_r)
    f_imh = np.roll(f_iph, 1)
    with np.errstate(over="ignore", invalid="ignore"):
        return u - (float(dt) / float(dx)) * (f_iph - f_imh)


def _minmod_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    same_sign = (a * b) > 0.0
    return np.where(same_sign, np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)


def _muscl_step_burgers(u: np.ndarray, *, dt: float, dx: float) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        d_minus = u - np.roll(u, 1)
        d_plus = np.roll(u, -1) - u
        sigma = _minmod_np(d_minus, d_plus)
        u_l = u + 0.5 * sigma
        u_r = np.roll(u - 0.5 * sigma, -1)
        f_iph = _godunov_flux_burgers(u_l, u_r)
        f_imh = np.roll(f_iph, 1)
        return u - (float(dt) / float(dx)) * (f_iph - f_imh)


def _weno3_left_burgers(u: np.ndarray, *, eps: float = 1e-6) -> np.ndarray:
    nx = int(u.shape[0])
    ext = np.pad(u, (2, 2), mode="wrap")
    im1 = ext[1 : nx + 1]
    i0 = ext[2 : nx + 2]
    ip1 = ext[3 : nx + 3]
    q0 = -0.5 * im1 + 1.5 * i0
    q1 = 0.5 * i0 + 0.5 * ip1
    beta0 = (i0 - im1) ** 2
    beta1 = (ip1 - i0) ** 2
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        alpha0 = (1.0 / 3.0) / (eps + beta0) ** 2
        alpha1 = (2.0 / 3.0) / (eps + beta1) ** 2
        denom = alpha0 + alpha1
        return (alpha0 * q0 + alpha1 * q1) / denom


def _weno3_right_burgers(u: np.ndarray, *, eps: float = 1e-6) -> np.ndarray:
    nx = int(u.shape[0])
    ext = np.pad(u, (2, 2), mode="wrap")
    i0 = ext[2 : nx + 2]
    ip1 = ext[3 : nx + 3]
    ip2 = ext[4 : nx + 4]
    q0 = 1.5 * ip1 - 0.5 * ip2
    q1 = 0.5 * i0 + 0.5 * ip1
    beta0 = (ip2 - ip1) ** 2
    beta1 = (ip1 - i0) ** 2
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        alpha0 = (1.0 / 3.0) / (eps + beta0) ** 2
        alpha1 = (2.0 / 3.0) / (eps + beta1) ** 2
        denom = alpha0 + alpha1
        return (alpha0 * q0 + alpha1 * q1) / denom


def _weno3_step_burgers(u: np.ndarray, *, dt: float, dx: float) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        u_l = _weno3_left_burgers(u)
        u_r = _weno3_right_burgers(u)
        f_iph = _godunov_flux_burgers(u_l, u_r)
        f_imh = np.roll(f_iph, 1)
        return u - (float(dt) / float(dx)) * (f_iph - f_imh)


def finite_volume_step_burgers(
    u: np.ndarray,
    *,
    dt: float,
    dx: float,
    scheme: str,
) -> np.ndarray:
    scheme_norm = normalize_baseline_scheme(scheme)
    u_arr = np.asarray(u, dtype=np.float64)
    if scheme_norm == BASELINE_SCHEME_GODUNOV:
        return _godunov_step_burgers(u_arr, dt=dt, dx=dx)
    if scheme_norm == BASELINE_SCHEME_MUSCL:
        return _muscl_step_burgers(u_arr, dt=dt, dx=dx)
    if scheme_norm == BASELINE_SCHEME_WENO3:
        return _weno3_step_burgers(u_arr, dt=dt, dx=dx)
    raise ValueError(f"Unsupported Burgers baseline scheme: {scheme!r}")


def _rollout_baseline(
    *,
    x0: np.ndarray,
    dt: float,
    dx: float,
    max_h: int,
    scheme: str,
) -> np.ndarray:
    scheme_norm = normalize_baseline_scheme(scheme)
    traj = np.empty((max_h + 1, x0.shape[0]), dtype=np.float64)
    traj[0] = np.asarray(x0, dtype=np.float64)
    for k in range(max_h):
        u = traj[k].copy()
        if not np.all(np.isfinite(u)):
            traj[k + 1 :] = np.nan
            break
        traj[k + 1] = finite_volume_step_burgers(
            u,
            dt=dt,
            dx=dx,
            scheme=scheme_norm,
        )
    return traj


def _rollout_godunov(
    *,
    x0: np.ndarray,
    dt: float,
    dx: float,
    max_h: int,
) -> np.ndarray:
    return _rollout_baseline(
        x0=x0,
        dt=dt,
        dx=dx,
        max_h=max_h,
        scheme=BASELINE_SCHEME_GODUNOV,
    )



def _rollout_gnn(
    *,
    model: torch.nn.Module,
    x_base_full: torch.Tensor,
    x0_state: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    dt: float,
    dx: float,
    max_h: int,
    device: torch.device,
    input_cfg,
    physics_cfg: PhysicsInputConfig,
) -> np.ndarray:
    model.eval()

    n_nodes = int(x_base_full.shape[0])
    x_pred = x0_state.clone().detach()
    traj = np.full((max_h + 1, n_nodes), np.nan, dtype=np.float64)
    traj[0] = x_pred.squeeze(-1).cpu().numpy().astype(np.float64)

    with torch.no_grad():
        for k in range(max_h):
            x_full_step = x_base_full.clone()
            x_full_step[:, 3:4] = x_pred

            x_model = select_feature_channels(x_full=x_full_step, input_cfg=input_cfg)
            x_in = x_model.to(device=device, dtype=torch.float32)
            ei = edge_index.to(device=device)
            ea = edge_attr.to(device=device, dtype=x_in.dtype)

            x_in = build_physics_augmented_inputs_1d(
                x_base=x_in,
                x_state=x_pred.to(device=device, dtype=x_in.dtype),
                edge_index=ei,
                edge_attr=ea,
                dt=dt,
                dx=dx,
                physics_cfg=physics_cfg,
                step_k=k,
            )
            x_in = apply_u_minmax_minus1_1(x_model=x_in, input_cfg=input_cfg)

            pred_next = model.predict_state(
                x_in,
                ei,
                edge_attr=ea,
                dt=dt,
                state_override=x_pred.to(device=device, dtype=x_in.dtype),
            ).detach().cpu()
            traj[k + 1] = pred_next.squeeze(-1).numpy().astype(np.float64)
            x_pred = pred_next

    return traj


def _compute_curves(
    *,
    pred_traj: np.ndarray,
    true_traj: np.ndarray,
    horizon: int,
    ssim_config: SSIMConfig | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-step metrics for t=1..H: RMSE, rel-L2, L1, and configured SSIM."""
    resolved_ssim = SSIMConfig() if ssim_config is None else ssim_config
    rmse = np.full((horizon,), np.nan, dtype=np.float64)
    rel = np.full((horizon,), np.nan, dtype=np.float64)
    l1 = np.full((horizon,), np.nan, dtype=np.float64)
    ssim = np.full((horizon,), np.nan, dtype=np.float64)

    max_h = min(horizon, int(pred_traj.shape[0] - 1), int(true_traj.shape[0] - 1))
    if max_h <= 0:
        return rmse, rel, l1, ssim

    for k in range(max_h):
        pred = np.asarray(pred_traj[k + 1], dtype=np.float64).reshape(-1)
        true = np.asarray(true_traj[k + 1], dtype=np.float64).reshape(-1)
        if pred.shape != true.shape:
            continue
        if not (np.all(np.isfinite(pred)) and np.all(np.isfinite(true))):
            continue

        with np.errstate(over="ignore", invalid="ignore"):
            err = pred - true
        if not np.all(np.isfinite(err)):
            continue

        scale = float(np.max(np.abs(err))) if err.size > 0 else 0.0
        if scale <= 0.0:
            rmse[k] = 0.0
            l1[k] = 0.0
        else:
            err_scaled = err / scale
            rmse[k] = float(scale * np.sqrt(np.mean(err_scaled * err_scaled)))
            l1[k] = float(scale * np.mean(np.abs(err_scaled)))

        err_norm = _safe_norm_1d(err)
        true_norm = _safe_norm_1d(true)
        if np.isfinite(err_norm) and np.isfinite(true_norm):
            try:
                rel_v = float(err_norm / max(true_norm, 1e-12))
            except OverflowError:
                rel_v = float("nan")
            rel[k] = rel_v if np.isfinite(rel_v) else float("nan")
        ssim[k] = _ssim_1d(pred, true, config=resolved_ssim)

    return rmse, rel, l1, ssim


def _reference_data_range(reference_trajs: List[np.ndarray]) -> float:
    """Return one finite range over all selected reference trajectories."""
    lo = float("inf")
    hi = float("-inf")
    for traj in reference_trajs:
        values = np.asarray(traj, dtype=np.float64)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        lo = min(lo, float(np.min(finite)))
        hi = max(hi, float(np.max(finite)))
    data_range = hi - lo
    if not np.isfinite(data_range) or data_range <= 0.0:
        raise ValueError(
            "Could not derive a positive SSIM data range from the selected "
            "reference trajectories. Pass --ssim-data-range explicitly."
        )
    return float(data_range)


def _resolve_ssim_config(
    args: argparse.Namespace,
    *,
    reference_trajs: List[np.ndarray],
    num_nodes: int,
) -> SSIMConfig:
    mode = str(args.ssim_mode)
    if mode == SSIM_MODE_LEGACY_GLOBAL:
        if args.ssim_data_range is not None:
            raise ValueError(
                "--ssim-data-range applies only to --ssim-mode local-periodic. "
                "The legacy-global mode intentionally reproduces the current "
                "prediction-pair-dependent calculation."
            )
        return SSIMConfig(mode=mode)

    if mode != SSIM_MODE_LOCAL_PERIODIC:
        raise ValueError(f"Unsupported SSIM mode '{mode}'. Expected one of {SSIM_MODES}.")

    window_size = int(args.ssim_window_size)
    window_sigma = float(args.ssim_window_sigma)
    _gaussian_window_1d(window_size=window_size, sigma=window_sigma)
    if window_size > int(num_nodes):
        raise ValueError(
            f"--ssim-window-size ({window_size}) cannot exceed the number of "
            f"spatial nodes ({num_nodes})."
        )

    if args.ssim_data_range is None:
        data_range = _reference_data_range(reference_trajs)
        data_range_source = "evaluation-reference"
    else:
        data_range = float(args.ssim_data_range)
        if not np.isfinite(data_range) or data_range <= 0.0:
            raise ValueError(
                "--ssim-data-range must be finite and positive, "
                f"got {args.ssim_data_range}."
            )
        data_range_source = "command-line"

    return SSIMConfig(
        mode=mode,
        data_range=data_range,
        data_range_source=data_range_source,
        window_size=window_size,
        window_sigma=window_sigma,
    )


def _safe_index_list(raw: object) -> List[int] | None:
    if raw is None:
        return None
    try:
        return [int(i) for i in list(raw)]
    except Exception:
        return None


def _default_test_effective_indices(dset, horizon: int) -> List[int]:
    window = int(dset.rollout_states.shape[1] - 1)
    max_start_t = int(window - horizon)
    if max_start_t < 0:
        max_start_t = 0
    t = dset.pair_time_index.detach().cpu().numpy()
    keep = np.nonzero(t <= max_start_t)[0]
    return keep.astype(np.int64).tolist()


def _select_representative_pairs_per_test_trajectory(
    *,
    dset,
    candidate_pair_ids: List[int],
    prefer_time_index: int = 0,
) -> List[int]:
    """Pick one pair index per local test trajectory from candidate pair ids.

    Selection prefers time indices nearest to `prefer_time_index` (default: t=0),
    then smaller time index, then smaller pair index as deterministic tiebreakers.
    """
    if len(candidate_pair_ids) == 0:
        return []

    pair_ids = np.asarray(candidate_pair_ids, dtype=np.int64)
    local_tid_all = dset.pair_trajectory_id.detach().cpu().numpy()[pair_ids]
    t_idx_all = dset.pair_time_index.detach().cpu().numpy()[pair_ids]

    best_by_local: Dict[int, Tuple[Tuple[int, int, int], int]] = {}
    t_pref = int(prefer_time_index)
    for pid, local_tid, t_idx in zip(pair_ids.tolist(), local_tid_all.tolist(), t_idx_all.tolist()):
        key = (abs(int(t_idx) - t_pref), int(t_idx), int(pid))
        prev = best_by_local.get(int(local_tid))
        if prev is None or key < prev[0]:
            best_by_local[int(local_tid)] = (key, int(pid))

    selected = [best_by_local[k][1] for k in sorted(best_by_local.keys())]
    return selected


def _select_t0_pairs_per_test_trajectory(*, dset) -> List[int]:
    """Pick one pair index per local test trajectory with exact start time t=0."""
    local_tid = dset.pair_trajectory_id.detach().cpu().numpy().astype(np.int64)
    t_idx = dset.pair_time_index.detach().cpu().numpy().astype(np.int64)
    unique_local = np.unique(local_tid)
    out: List[int] = []
    for lt in unique_local.tolist():
        cand = np.nonzero((local_tid == int(lt)) & (t_idx == 0))[0]
        if cand.size == 0:
            continue
        out.append(int(cand[0]))
    return out


def _sum_field_curve(traj: np.ndarray) -> np.ndarray:
    """Compute sum(u) over nodes at each trajectory time index."""
    arr = np.asarray(traj, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected trajectory shape [T,N], got {tuple(arr.shape)}.")
    with np.errstate(over="ignore", invalid="ignore"):
        return _replace_nonfinite_with_nan(np.sum(arr, axis=1))


def _lookup_extra_sum_curve_for_traj(
    extra: ExtraEvalCurves,
    *,
    global_tid: int,
    time_index: int = 0,
) -> np.ndarray | None:
    """Return comparison sum(u) curve matching one test trajectory start."""
    traj = extra.sample_gnn_traj_by_sample
    gtid = extra.sample_global_traj_id
    t0 = extra.sample_time_index
    if traj is None or gtid is None or t0 is None:
        return None
    mask = (gtid == int(global_tid)) & (t0 == int(time_index))
    idx = np.nonzero(mask)[0]
    if idx.size == 0:
        return None
    pick = int(idx[0])
    if extra.sample_valid_steps is not None:
        valid = int(np.clip(extra.sample_valid_steps[pick], 0, traj.shape[1] - 1))
        use = traj[pick, : valid + 1, :]
    else:
        use = traj[pick]
    return _sum_field_curve(use)


def _evaluate_pair_rollout(
    *,
    dset,
    pid: int,
    window: int,
    horizon: int,
    model: torch.nn.Module,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    dt: float,
    dx: float,
    device: torch.device,
    input_cfg,
    physics_cfg: PhysicsInputConfig,
) -> Tuple[int, int, int, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate exact / GNN / Godunov trajectories for one pair index."""
    local_tid = int(dset.pair_trajectory_id[pid].item())
    t0 = int(dset.pair_time_index[pid].item())
    global_tid = int(dset.trajectory_ids[local_tid].item())

    max_h = int(min(horizon, max(0, window - t0)))
    true_traj = (
        dset.rollout_states[local_tid, t0 : t0 + max_h + 1, :, 0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    x0_state = torch.from_numpy(true_traj[0][:, None]).to(dtype=torch.float32)

    gnn_traj = _rollout_gnn(
        model=model,
        x_base_full=dset.x_full[pid].detach().cpu(),
        x0_state=x0_state,
        edge_index=edge_index,
        edge_attr=edge_attr,
        dt=dt,
        dx=dx,
        max_h=max_h,
        device=device,
        input_cfg=input_cfg,
        physics_cfg=physics_cfg,
    )
    god_traj = _rollout_godunov(
        x0=true_traj[0],
        dt=dt,
        dx=dx,
        max_h=max_h,
    )
    return local_tid, t0, global_tid, true_traj, gnn_traj, god_traj


def _load_data_source_from_cfg(
    data_cfg: Mapping[str, Any],
    *,
    override_data_dir: Path | None,
) -> Tuple[Path, str, Dict[str, Any]]:
    source_format = str(data_cfg.get("source_format", "auto")).strip().lower()
    data_dir = Path(data_cfg.get("data_dir", "data/burgers_1d_dataset_v1/burgers_data"))
    if "wenoa_pt_path" in data_cfg:
        data_dir = Path(data_cfg["wenoa_pt_path"])
    if override_data_dir is not None:
        data_dir = Path(override_data_dir)

    source_options: Dict[str, Any] = {}
    if "wenoa_split_fractions" in data_cfg:
        source_options["split_fractions"] = data_cfg["wenoa_split_fractions"]
    if "wenoa_split_seed" in data_cfg:
        source_options["split_seed"] = int(data_cfg["wenoa_split_seed"])
    if "wenoa_nu" in data_cfg:
        source_options["nu_value"] = float(data_cfg["wenoa_nu"])
    if "wenoa_split_paths" in data_cfg:
        source_options["split_paths"] = data_cfg["wenoa_split_paths"]

    return data_dir, source_format, source_options


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Burgers GNN rollout against finite-volume baselines."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to checkpoint_best.pt or checkpoint_last.pt from training.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Optional dataset path override (directory for npz_bundle, .pt for wenoa_pt).",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Optional rollout horizon override.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help=(
            "Optional evaluation sample limit. In default trajectory mode, this "
            "limits the number of test trajectories; in --eval-pair-samples mode, "
            "this limits the number of valid pair starts."
        ),
    )
    parser.add_argument(
        "--ssim-mode",
        type=str,
        choices=SSIM_MODES,
        default=SSIM_MODE_LEGACY_GLOBAL,
        help=(
            "SSIM definition: 'legacy-global' exactly preserves the existing "
            "single-window, prediction-dependent calculation; 'local-periodic' "
            "uses Gaussian local statistics, periodic boundaries, and one fixed "
            "reference-only range. Default: legacy-global."
        ),
    )
    parser.add_argument(
        "--ssim-data-range",
        type=float,
        default=None,
        help=(
            "Fixed physical/reference range for local-periodic SSIM. If omitted, "
            "one range is derived from all selected reference trajectories. "
            "Not applicable to legacy-global mode."
        ),
    )
    parser.add_argument(
        "--ssim-window-size",
        type=int,
        default=SSIM_DEFAULT_WINDOW_SIZE,
        help=(
            "Odd Gaussian window size in spatial nodes for local-periodic SSIM. "
            f"Default: {SSIM_DEFAULT_WINDOW_SIZE}."
        ),
    )
    parser.add_argument(
        "--ssim-window-sigma",
        type=float,
        default=SSIM_DEFAULT_WINDOW_SIGMA,
        help=(
            "Gaussian standard deviation in spatial nodes for local-periodic "
            f"SSIM. Default: {SSIM_DEFAULT_WINDOW_SIGMA}."
        ),
    )
    parser.add_argument(
        "--allow-full-test-fallback",
        action="store_true",
        help=(
            "Allow fallback to all valid test start indices when checkpoint "
            "split_indices.test_effective is missing."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output NPZ path. Default: <checkpoint_dir>/gnn_vs_godunov_eval.npz",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help="Directory for comparison visualizations.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable plot generation.",
    )
    parser.add_argument(
        "--plot-all-samples",
        action="store_true",
        help=(
            "Also write a single multi-page PDF containing spacetime comparison "
            "figures for one representative sample per test trajectory."
        ),
    )
    eval_group = parser.add_mutually_exclusive_group()
    eval_group.add_argument(
        "--eval-test-trajectories",
        dest="eval_test_trajectories",
        action="store_true",
        default=True,
        help=(
            "Use one rollout per test trajectory, preferring t0=0, for the primary "
            "evaluation/error curves. This is the default. If --horizon is omitted, "
            "evaluates the full available rollout window instead of "
            "rollout_eval.horizon. If --num-samples is explicitly provided, limits "
            "to that many test trajectories."
        ),
    )
    eval_group.add_argument(
        "--eval-pair-samples",
        dest="eval_test_trajectories",
        action="store_false",
        help=(
            "Legacy evaluation mode: use the first --num-samples valid test pair "
            "starts instead of one representative start per test trajectory."
        ),
    )
    parser.add_argument(
        "--error-curves-log-y",
        action="store_true",
        help="Use logarithmic y-axis for RMSE/rel-L2/L1 in error_curves_compare.png.",
    )
    parser.add_argument(
        "--error-curves-mean-only",
        action="store_true",
        help="Plot only average curves (hide per-sample trajectories) in error_curves_compare.png.",
    )
    parser.add_argument(
        "--error-curves-spread",
        choices=ERROR_CURVE_SPREAD_MODES,
        default=ERROR_CURVE_SPREAD_NONE,
        help=(
            "Optional shaded spread around each mean error curve: 'std' uses "
            "mean plus/minus one population standard deviation and 'iqr' uses "
            "the 25th-75th percentiles. Default: none."
        ),
    )
    parser.add_argument(
        "--error-curves-smooth-window",
        type=int,
        default=1,
        help=(
            "NaN-aware centered moving-average window over time for error curves "
            "(1 disables smoothing)."
        ),
    )
    parser.add_argument(
        "--error-curves-metrics",
        type=str,
        nargs="+",
        default=["all"],
        help=(
            "Accuracy metrics to show in error_curves_compare.png. Use "
            "space-separated or comma-separated values from {rmse, rel_l2, l1, "
            "ssim}, or 'all'. Aliases include relative_l2 and mae. Default: all."
        ),
    )
    parser.add_argument(
        "--error-curves-autoscale-source",
        type=str,
        default="all",
        help=(
            "Which prediction source controls y-axis autoscaling in "
            "error_curves_compare.png. Use 'all' for current behavior, or a "
            "single source such as 'gnn', 'godunov', 'muscl', 'weno3', or a "
            "legend label. Other curves are still plotted but ignored for limits."
        ),
    )
    parser.add_argument(
        "--curve-color",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Override colors for curves in comparison plots with KEY=COLOR "
            "pairs. KEY can be gnn, godunov, muscl, weno3, compare1, "
            "compare2, or a displayed legend label. Repeat or pass multiple "
            "pairs, e.g. --curve-color gnn=black godunov=gray "
            "compare1='#9467bd'. Quote hex colors in the shell."
        ),
    )
    parser.add_argument(
        "--summed-field-exclude-exact",
        action="store_true",
        help=(
            "Exclude the Exact/reference curve from summed_field_totals_compare.png "
            "and summed_field_totals_compare_all.pdf."
        ),
    )
    parser.add_argument(
        "--compare-eval-npz",
        type=Path,
        action="append",
        default=None,
        help=(
            "Optional path to another gnn_vs_godunov_eval.npz to overlay in plots. "
            "Repeat this flag to include multiple models."
        ),
    )
    parser.add_argument(
        "--baseline-schemes",
        type=str,
        nargs="+",
        default=[BASELINE_SCHEME_GODUNOV],
        help=(
            "Finite-volume baselines to show in plots. Use space-separated or "
            "comma-separated values from {godunov, muscl, weno3}, 'all', or 'none'. "
            "Default: godunov."
        ),
    )
    parser.add_argument(
        "--gnn-label",
        type=str,
        default="GNN",
        help="Legend label for GNN curves.",
    )
    parser.add_argument(
        "--godunov-label",
        type=str,
        default="Godunov",
        help="Legend label for Godunov curves.",
    )
    parser.add_argument(
        "--muscl-label",
        type=str,
        default=BASELINE_SCHEME_DEFAULT_LABELS[BASELINE_SCHEME_MUSCL],
        help="Legend label for MUSCL baseline curves.",
    )
    parser.add_argument(
        "--weno3-label",
        type=str,
        default=BASELINE_SCHEME_DEFAULT_LABELS[BASELINE_SCHEME_WENO3],
        help="Legend label for WENO-3 baseline curves.",
    )
    parser.add_argument(
        "--compare-label",
        type=str,
        action="append",
        default=None,
        help=(
            "Optional legend label for a comparison NPZ. Repeat in the same order "
            "as --compare-eval-npz."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=("auto", "cpu", "cuda", "mps"),
        help=(
            "Optional device override for rollout inference. "
            "If omitted, uses the device stored in the checkpoint config."
        ),
    )
    return parser.parse_args()


def _write_plots(
    *,
    plots_dir: Path,
    horizon: int,
    dt: float,
    rmse_gnn: np.ndarray,
    rel_gnn: np.ndarray,
    l1_gnn: np.ndarray,
    ssim_gnn: np.ndarray,
    rmse_god: np.ndarray,
    rel_god: np.ndarray,
    l1_god: np.ndarray,
    ssim_god: np.ndarray,
    mean_rmse_gnn: np.ndarray,
    mean_rel_gnn: np.ndarray,
    mean_l1_gnn: np.ndarray,
    mean_ssim_gnn: np.ndarray,
    mean_rmse_god: np.ndarray,
    mean_rel_god: np.ndarray,
    mean_l1_god: np.ndarray,
    mean_ssim_god: np.ndarray,
    gnn_label: str,
    godunov_label: str,
    plot_baseline_schemes: List[str],
    pos: np.ndarray,
    sample_id: int | None,
    sample_global_id: int | None,
    sample_local_id: int | None,
    sample_time_idx: int | None,
    true_traj: np.ndarray | None,
    gnn_traj: np.ndarray | None,
    god_traj: np.ndarray | None,
    sample_extra_baseline_trajs: Dict[str, np.ndarray] | None,
    plot_all_samples: bool,
    all_sample_ids: List[int] | None,
    all_sample_global_ids: List[int] | None,
    all_sample_local_ids: List[int] | None,
    all_sample_time_indices: List[int] | None,
    all_true_trajs: List[np.ndarray] | None,
    all_gnn_trajs: List[np.ndarray] | None,
    all_god_trajs: List[np.ndarray] | None,
    all_extra_baseline_trajs: Dict[str, List[np.ndarray]] | None,
    error_curves_log_y: bool,
    error_curves_mean_only: bool,
    error_curves_spread: str,
    error_curves_smooth_window: int,
    error_curves_metrics: List[str],
    ssim_config: SSIMConfig,
    error_curves_autoscale_source: str,
    curve_colors: Dict[str, str] | None,
    summed_field_show_exact: bool,
    extra_baseline_curves: List[BaselineEvalCurves] | None,
    extra_eval_curves: List[ExtraEvalCurves] | None,
    sumplot_pages: List[Dict[str, Any]] | None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir.mkdir(parents=True, exist_ok=True)

    steps = np.arange(1, horizon + 1, dtype=np.int64)
    times = steps.astype(np.float64) * float(dt)
    smooth_w = max(int(error_curves_smooth_window), 1)
    plot_scheme_set = set(plot_baseline_schemes)
    show_godunov_baseline = BASELINE_SCHEME_GODUNOV in plot_scheme_set
    extra_baselines = [
        baseline
        for baseline in ([] if extra_baseline_curves is None else list(extra_baseline_curves))
        if baseline.scheme in plot_scheme_set
    ]
    color_overrides = {} if curve_colors is None else dict(curve_colors)
    gnn_color = color_overrides.get("gnn", "tab:blue")
    godunov_color = color_overrides.get(BASELINE_SCHEME_GODUNOV, "tab:green")

    if show_godunov_baseline:
        god_failure_source = np.asarray(rmse_god, dtype=np.float64)
        if god_failure_source.ndim == 2:
            n_error_curve_samples = int(god_failure_source.shape[0])
            god_failed_mask = np.any(~np.isfinite(god_failure_source), axis=1)
        elif god_failure_source.size > 0:
            n_error_curve_samples = 1
            god_failed_mask = np.array(
                [bool(np.any(~np.isfinite(god_failure_source)))],
                dtype=bool,
            )
        else:
            n_error_curve_samples = 0
            god_failed_mask = np.zeros((0,), dtype=bool)
    else:
        gnn_failure_source = np.asarray(rmse_gnn, dtype=np.float64)
        n_error_curve_samples = (
            int(gnn_failure_source.shape[0])
            if gnn_failure_source.ndim == 2
            else int(gnn_failure_source.size > 0)
        )
        god_failed_mask = np.zeros((n_error_curve_samples,), dtype=bool)
    n_god_failed_excluded = int(np.count_nonzero(god_failed_mask))
    error_curve_keep_mask = ~god_failed_mask

    def _filter_godunov_failed_rows(arr: np.ndarray) -> np.ndarray:
        a = np.asarray(arr, dtype=np.float64)
        if (
            a.ndim == 2
            and n_error_curve_samples > 0
            and int(a.shape[0]) == n_error_curve_samples
        ):
            return a[error_curve_keep_mask]
        return a

    def _mean_for_error_curve(arr: np.ndarray) -> np.ndarray:
        a = np.asarray(arr, dtype=np.float64)
        if a.ndim == 1:
            return a
        if a.ndim == 2 and int(a.shape[0]) == 0:
            return np.full((horizon,), np.nan, dtype=np.float64)
        return _nanmean_over_samples(a, horizon)

    rmse_gnn_error = _filter_godunov_failed_rows(rmse_gnn)
    rel_gnn_error = _filter_godunov_failed_rows(rel_gnn)
    l1_gnn_error = _filter_godunov_failed_rows(l1_gnn)
    ssim_gnn_error = _filter_godunov_failed_rows(ssim_gnn)
    rmse_god_error = _filter_godunov_failed_rows(rmse_god)
    rel_god_error = _filter_godunov_failed_rows(rel_god)
    l1_god_error = _filter_godunov_failed_rows(l1_god)
    ssim_god_error = _filter_godunov_failed_rows(ssim_god)

    rmse_gnn_plot = _smooth_curves(rmse_gnn_error, window=smooth_w)
    rel_gnn_plot = _smooth_curves(rel_gnn_error, window=smooth_w)
    l1_gnn_plot = _smooth_curves(l1_gnn_error, window=smooth_w)
    ssim_gnn_plot = _smooth_curves(ssim_gnn_error, window=smooth_w)
    rmse_god_plot = _smooth_curves(rmse_god_error, window=smooth_w)
    rel_god_plot = _smooth_curves(rel_god_error, window=smooth_w)
    l1_god_plot = _smooth_curves(l1_god_error, window=smooth_w)
    ssim_god_plot = _smooth_curves(ssim_god_error, window=smooth_w)
    mean_rmse_gnn_plot = _smooth_curves(_mean_for_error_curve(rmse_gnn_error), window=smooth_w)
    mean_rel_gnn_plot = _smooth_curves(_mean_for_error_curve(rel_gnn_error), window=smooth_w)
    mean_l1_gnn_plot = _smooth_curves(_mean_for_error_curve(l1_gnn_error), window=smooth_w)
    mean_ssim_gnn_plot = _smooth_curves(_mean_for_error_curve(ssim_gnn_error), window=smooth_w)
    mean_rmse_god_plot = _smooth_curves(_mean_for_error_curve(rmse_god_error), window=smooth_w)
    mean_rel_god_plot = _smooth_curves(_mean_for_error_curve(rel_god_error), window=smooth_w)
    mean_l1_god_plot = _smooth_curves(_mean_for_error_curve(l1_god_error), window=smooth_w)
    mean_ssim_god_plot = _smooth_curves(_mean_for_error_curve(ssim_god_error), window=smooth_w)
    error_curve_baselines: List[BaselineEvalCurves] = []
    for baseline in extra_baselines:
        rmse_b = _filter_godunov_failed_rows(baseline.rmse_by_sample)
        rel_b = _filter_godunov_failed_rows(baseline.rel_by_sample)
        l1_b = _filter_godunov_failed_rows(baseline.l1_by_sample)
        ssim_b = _filter_godunov_failed_rows(baseline.ssim_by_sample)
        error_curve_baselines.append(
            BaselineEvalCurves(
                scheme=baseline.scheme,
                label=baseline.label,
                rmse_by_sample=rmse_b,
                rel_by_sample=rel_b,
                l1_by_sample=l1_b,
                ssim_by_sample=ssim_b,
                mean_rmse=_mean_for_error_curve(rmse_b),
                mean_rel=_mean_for_error_curve(rel_b),
                mean_l1=_mean_for_error_curve(l1_b),
                mean_ssim=_mean_for_error_curve(ssim_b),
            )
        )

    def _baseline_color(idx: int, scheme: str) -> str:
        if scheme in color_overrides:
            return color_overrides[scheme]
        fixed = {
            BASELINE_SCHEME_WENO3: "tab:red",
        }
        if scheme in fixed:
            return fixed[scheme]
        return f"C{(idx + 4) % 10}"

    def _extra_model_color(idx: int) -> str:
        return color_overrides.get(
            f"compare:{idx}",
            f"C{(idx + 4 + len(extra_baselines)) % 10}",
        )

    def _nonfinite_count_note(label: str, values: np.ndarray) -> str | None:
        vals = np.asarray(values, dtype=np.float64)
        total = int(vals.size)
        if total <= 0:
            return None
        bad = int(total - np.count_nonzero(np.isfinite(vals)))
        if bad <= 0:
            return None
        return f"{label}: {bad}/{total}"

    def _annotate_nonfinite(
        ax,
        *,
        gnn_values: np.ndarray,
        god_values: np.ndarray | None,
        extra_values: List[tuple[str, np.ndarray]] | None = None,
    ) -> None:
        notes = []
        note = _nonfinite_count_note(gnn_label, gnn_values)
        if note is not None:
            notes.append(note)
        if god_values is not None:
            note = _nonfinite_count_note(godunov_label, god_values)
            if note is not None:
                notes.append(note)
        for label, values in ([] if extra_values is None else extra_values):
            note = _nonfinite_count_note(label, values)
            if note is not None:
                notes.append(note)
        if len(notes) == 0:
            return
        ax.text(
            0.98,
            0.98,
            "NaN/nonfinite metric entries omitted\n" + "\n".join(notes),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": "0.45",
                "alpha": 0.88,
            },
        )

    # 1) Mean error curves.
    metric_cfg: Dict[str, Dict[str, Any]] = {
        ERROR_CURVE_METRIC_RMSE: {
            "ylabel": "RMSE",
            "log_y": True,
            "gnn_sample": rmse_gnn_plot,
            "gnn_mean": mean_rmse_gnn_plot,
            "god_sample": rmse_god_plot,
            "god_mean": mean_rmse_god_plot,
            "sample_attr": "rmse_by_sample",
            "mean_attr": "mean_rmse",
            "gnn_error": rmse_gnn_error,
            "god_error": rmse_god_error,
        },
        ERROR_CURVE_METRIC_REL_L2: {
            "ylabel": "rel-L2",
            "log_y": True,
            "gnn_sample": rel_gnn_plot,
            "gnn_mean": mean_rel_gnn_plot,
            "god_sample": rel_god_plot,
            "god_mean": mean_rel_god_plot,
            "sample_attr": "rel_by_sample",
            "mean_attr": "mean_rel",
            "gnn_error": rel_gnn_error,
            "god_error": rel_god_error,
        },
        ERROR_CURVE_METRIC_L1: {
            "ylabel": "L1",
            "log_y": True,
            "gnn_sample": l1_gnn_plot,
            "gnn_mean": mean_l1_gnn_plot,
            "god_sample": l1_god_plot,
            "god_mean": mean_l1_god_plot,
            "sample_attr": "l1_by_sample",
            "mean_attr": "mean_l1",
            "gnn_error": l1_gnn_error,
            "god_error": l1_god_error,
        },
        ERROR_CURVE_METRIC_SSIM: {
            "ylabel": ssim_config.display_name,
            "log_y": False,
            "gnn_sample": ssim_gnn_plot,
            "gnn_mean": mean_ssim_gnn_plot,
            "god_sample": ssim_god_plot,
            "god_mean": mean_ssim_god_plot,
            "sample_attr": "ssim_by_sample",
            "mean_attr": "mean_ssim",
            "gnn_error": ssim_gnn_error,
            "god_error": ssim_god_error,
        },
    }
    selected_metrics = [m for m in error_curves_metrics if m in metric_cfg]
    if len(selected_metrics) == 0:
        raise ValueError("At least one valid error curve metric must be requested.")

    n_metrics = len(selected_metrics)
    if n_metrics == 1:
        n_rows, n_cols = 1, 1
        figsize = (8.0, 4.8)
    elif n_metrics == 2:
        n_rows, n_cols = 1, 2
        figsize = (15.5, 4.8)
    else:
        n_rows, n_cols = 2, 2
        figsize = (15.5, 8.3)

    fig, axes_obj = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        constrained_layout=True,
        squeeze=False,
    )
    axes_list = list(np.asarray(axes_obj, dtype=object).reshape(-1))
    metric_axes = dict(zip(selected_metrics, axes_list[:n_metrics]))
    for ax in axes_list[n_metrics:]:
        ax.set_visible(False)

    extra_models = [] if extra_eval_curves is None else list(extra_eval_curves)
    axes_title_size = 20

    def _fill_spread_band(
        ax,
        sample_values: np.ndarray,
        *,
        color: str,
        x_values: np.ndarray | None = None,
    ) -> None:
        bounds = _error_curve_spread(
            np.asarray(sample_values, dtype=np.float64),
            mode=error_curves_spread,
        )
        if bounds is None:
            return
        lower, upper = bounds
        finite = np.isfinite(lower) & np.isfinite(upper)
        if not np.any(finite):
            return
        x_plot = times if x_values is None else np.asarray(x_values, dtype=np.float64)
        ax.fill_between(
            x_plot,
            lower,
            upper,
            where=finite,
            interpolate=False,
            color=color,
            alpha=0.16,
            linewidth=0.0,
            zorder=1,
        )

    def _plot_metric_panel(ax, metric: str, cfg: Dict[str, Any], panel_idx: int) -> None:
        gnn_sample = np.asarray(cfg["gnn_sample"], dtype=np.float64)
        god_sample = np.asarray(cfg["god_sample"], dtype=np.float64)
        _fill_spread_band(ax, gnn_sample, color=gnn_color)
        if show_godunov_baseline:
            _fill_spread_band(ax, god_sample, color=godunov_color)
        for j, baseline in enumerate(error_curve_baselines):
            _fill_spread_band(
                ax,
                _smooth_curves(
                    getattr(baseline, str(cfg["sample_attr"])),
                    window=smooth_w,
                ),
                color=_baseline_color(j, baseline.scheme),
            )
        for j, extra in enumerate(extra_models):
            t_extra = np.arange(1, extra.horizon + 1, dtype=np.float64) * float(dt)
            _fill_spread_band(
                ax,
                _smooth_curves(
                    getattr(extra, str(cfg["sample_attr"])),
                    window=smooth_w,
                ),
                color=_extra_model_color(j),
                x_values=t_extra,
            )
        if (gnn_sample.ndim == 2 and gnn_sample.shape[0] > 0) and (not error_curves_mean_only):
            for i in range(gnn_sample.shape[0]):
                ax.plot(times, gnn_sample[i], color=gnn_color, alpha=0.3, lw=0.9)
                if show_godunov_baseline:
                    ax.plot(
                        times,
                        god_sample[i],
                        color=godunov_color,
                        alpha=0.3,
                        lw=0.9,
                        ls="--",
                    )
            for j, baseline in enumerate(error_curve_baselines):
                c_base = _baseline_color(j, baseline.scheme)
                sample_arr = _smooth_curves(
                    getattr(baseline, str(cfg["sample_attr"])),
                    window=smooth_w,
                )
                for i in range(sample_arr.shape[0]):
                    ax.plot(
                        times,
                        sample_arr[i],
                        color=c_base,
                        alpha=0.22,
                        lw=0.85,
                        ls="--",
                    )

        ax.plot(
            times,
            np.asarray(cfg["gnn_mean"], dtype=np.float64),
            color=gnn_color,
            lw=2.3,
            label=gnn_label,
        )
        if show_godunov_baseline:
            ax.plot(
                times,
                np.asarray(cfg["god_mean"], dtype=np.float64),
                color=godunov_color,
                lw=2.3,
                ls="--",
                label=godunov_label,
            )
        for j, baseline in enumerate(error_curve_baselines):
            c_base = _baseline_color(j, baseline.scheme)
            ax.plot(
                times,
                _smooth_curves(getattr(baseline, str(cfg["mean_attr"])), window=smooth_w),
                color=c_base,
                lw=2.1,
                ls="--",
                label=baseline.label,
            )
        for j, extra in enumerate(extra_models):
            c = _extra_model_color(j)
            t_extra = np.arange(1, extra.horizon + 1, dtype=np.float64) * float(dt)
            ax.plot(
                t_extra,
                _smooth_curves(getattr(extra, str(cfg["mean_attr"])), window=smooth_w),
                color=c,
                lw=2.0,
                label=extra.label,
            )

        if n_rows == 1 or panel_idx >= (n_rows - 1) * n_cols:
            ax.set_xlabel("Time", fontsize=axes_title_size)
        ax.set_ylabel(str(cfg["ylabel"]), fontsize=axes_title_size)

        ax.tick_params(axis="x", labelsize=16)
        ax.tick_params(axis="y", labelsize=16)

        ax.grid(alpha=0.3)
        if metric == ERROR_CURVE_METRIC_SSIM:
            ax.legend(loc="lower left", fontsize=16)
        #else:
        #    ax.legend(loc="upper left", fontsize=18)

    for panel_idx, metric in enumerate(selected_metrics):
        _plot_metric_panel(metric_axes[metric], metric, metric_cfg[metric], panel_idx)

    def _autoscale_key(value: str) -> str:
        return "".join(ch for ch in str(value).strip().lower() if ch.isalnum())

    def _autoscale_metric_values(sample_arr: np.ndarray, mean_arr: np.ndarray) -> np.ndarray:
        pieces = [np.asarray(mean_arr, dtype=np.float64).reshape(-1)]
        if not error_curves_mean_only:
            pieces.insert(0, np.asarray(sample_arr, dtype=np.float64).reshape(-1))
        bounds = _error_curve_spread(sample_arr, mode=error_curves_spread)
        if bounds is not None:
            pieces.extend(np.asarray(bound, dtype=np.float64) for bound in bounds)
        return np.concatenate(pieces) if len(pieces) > 0 else np.array([], dtype=np.float64)

    def _register_autoscale_source(
        source_map: Dict[str, Dict[str, np.ndarray]],
        names: List[str],
        *,
        rmse_sample: np.ndarray,
        rmse_mean: np.ndarray,
        rel_sample: np.ndarray,
        rel_mean: np.ndarray,
        l1_sample: np.ndarray,
        l1_mean: np.ndarray,
        ssim_sample: np.ndarray,
        ssim_mean: np.ndarray,
    ) -> None:
        values = {
            "rmse": _autoscale_metric_values(rmse_sample, rmse_mean),
            "rel": _autoscale_metric_values(rel_sample, rel_mean),
            "l1": _autoscale_metric_values(l1_sample, l1_mean),
            "ssim": _autoscale_metric_values(ssim_sample, ssim_mean),
        }
        for name in names:
            key = _autoscale_key(name)
            if key != "":
                source_map[key] = values

    autoscale_sources: Dict[str, Dict[str, np.ndarray]] = {}
    autoscale_source_names: List[str] = ["all"]

    def _add_autoscale_names(names: List[str]) -> None:
        for name in names:
            label = str(name).strip()
            if label != "" and label not in autoscale_source_names:
                autoscale_source_names.append(label)

    _register_autoscale_source(
        autoscale_sources,
        ["gnn", gnn_label],
        rmse_sample=rmse_gnn_plot,
        rmse_mean=mean_rmse_gnn_plot,
        rel_sample=rel_gnn_plot,
        rel_mean=mean_rel_gnn_plot,
        l1_sample=l1_gnn_plot,
        l1_mean=mean_l1_gnn_plot,
        ssim_sample=ssim_gnn_plot,
        ssim_mean=mean_ssim_gnn_plot,
    )
    _add_autoscale_names(["gnn", gnn_label])
    if show_godunov_baseline:
        _register_autoscale_source(
            autoscale_sources,
            ["godunov", godunov_label],
            rmse_sample=rmse_god_plot,
            rmse_mean=mean_rmse_god_plot,
            rel_sample=rel_god_plot,
            rel_mean=mean_rel_god_plot,
            l1_sample=l1_god_plot,
            l1_mean=mean_l1_god_plot,
            ssim_sample=ssim_god_plot,
            ssim_mean=mean_ssim_god_plot,
        )
        _add_autoscale_names(["godunov", godunov_label])
    for baseline in error_curve_baselines:
        _register_autoscale_source(
            autoscale_sources,
            [baseline.scheme, baseline.label],
            rmse_sample=_smooth_curves(baseline.rmse_by_sample, window=smooth_w),
            rmse_mean=_smooth_curves(baseline.mean_rmse, window=smooth_w),
            rel_sample=_smooth_curves(baseline.rel_by_sample, window=smooth_w),
            rel_mean=_smooth_curves(baseline.mean_rel, window=smooth_w),
            l1_sample=_smooth_curves(baseline.l1_by_sample, window=smooth_w),
            l1_mean=_smooth_curves(baseline.mean_l1, window=smooth_w),
            ssim_sample=_smooth_curves(baseline.ssim_by_sample, window=smooth_w),
            ssim_mean=_smooth_curves(baseline.mean_ssim, window=smooth_w),
        )
        _add_autoscale_names([baseline.scheme, baseline.label])
    for extra in extra_models:
        _register_autoscale_source(
            autoscale_sources,
            [extra.label],
            rmse_sample=_smooth_curves(extra.rmse_by_sample, window=smooth_w),
            rmse_mean=_smooth_curves(extra.mean_rmse, window=smooth_w),
            rel_sample=_smooth_curves(extra.rel_by_sample, window=smooth_w),
            rel_mean=_smooth_curves(extra.mean_rel, window=smooth_w),
            l1_sample=_smooth_curves(extra.l1_by_sample, window=smooth_w),
            l1_mean=_smooth_curves(extra.mean_l1, window=smooth_w),
            ssim_sample=_smooth_curves(extra.ssim_by_sample, window=smooth_w),
            ssim_mean=_smooth_curves(extra.mean_ssim, window=smooth_w),
        )
        _add_autoscale_names([extra.label])

    def _collect_plotted_metric_values(metric: str, cfg: Dict[str, Any]) -> np.ndarray:
        pieces = []

        def _add(arr: np.ndarray) -> None:
            pieces.append(np.asarray(arr, dtype=np.float64).reshape(-1))

        if not error_curves_mean_only:
            _add(cfg["gnn_sample"])
        _add(cfg["gnn_mean"])
        bounds = _error_curve_spread(cfg["gnn_sample"], mode=error_curves_spread)
        if bounds is not None:
            _add(bounds[0])
            _add(bounds[1])
        if show_godunov_baseline:
            if not error_curves_mean_only:
                _add(cfg["god_sample"])
            _add(cfg["god_mean"])
            bounds = _error_curve_spread(cfg["god_sample"], mode=error_curves_spread)
            if bounds is not None:
                _add(bounds[0])
                _add(bounds[1])
        for baseline in error_curve_baselines:
            sample_values = _smooth_curves(
                getattr(baseline, str(cfg["sample_attr"])),
                window=smooth_w,
            )
            if not error_curves_mean_only:
                _add(sample_values)
            _add(_smooth_curves(getattr(baseline, str(cfg["mean_attr"])), window=smooth_w))
            bounds = _error_curve_spread(sample_values, mode=error_curves_spread)
            if bounds is not None:
                _add(bounds[0])
                _add(bounds[1])
        for extra in extra_models:
            sample_values = _smooth_curves(
                getattr(extra, str(cfg["sample_attr"])),
                window=smooth_w,
            )
            if not error_curves_mean_only:
                _add(sample_values)
            _add(_smooth_curves(getattr(extra, str(cfg["mean_attr"])), window=smooth_w))
            bounds = _error_curve_spread(sample_values, mode=error_curves_spread)
            if bounds is not None:
                _add(bounds[0])
                _add(bounds[1])
        return np.concatenate(pieces) if len(pieces) > 0 else np.array([], dtype=np.float64)

    if error_curves_log_y:
        def _set_log_y_if_possible(ax, values: np.ndarray) -> None:
            vals = np.asarray(values, dtype=np.float64).reshape(-1)
            finite = vals[np.isfinite(vals)]
            positive = finite[finite > 0.0]
            if positive.size == 0:
                return
            ymin = float(np.min(positive))
            floor = max(1e-12, ymin * 0.9)
            ax.set_yscale("log")
            lo, hi = ax.get_ylim()
            if (not np.isfinite(lo)) or (lo <= 0.0):
                lo = floor
            else:
                lo = max(lo, floor)
            if np.isfinite(hi) and hi > lo:
                ax.set_ylim(lo, hi)
            else:
                ax.set_ylim(bottom=lo)

        for metric in selected_metrics:
            cfg = metric_cfg[metric]
            if bool(cfg["log_y"]):
                _set_log_y_if_possible(
                    metric_axes[metric],
                    _collect_plotted_metric_values(metric, cfg),
                )

    def _set_linear_ylim_from_values(ax, values: np.ndarray, *, nonnegative: bool) -> None:
        vals = np.asarray(values, dtype=np.float64).reshape(-1)
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            return
        lo = float(np.min(finite))
        hi = float(np.max(finite))
        if not np.isfinite(lo) or not np.isfinite(hi):
            return
        if hi <= lo:
            pad = max(1e-12, abs(hi) * 0.05)
        else:
            pad = 0.05 * (hi - lo)
        y0 = lo - pad
        y1 = hi + pad
        if nonnegative:
            y0 = max(0.0, y0)
        if y1 <= y0:
            y1 = y0 + max(1e-12, abs(y0) * 0.1)
        ax.set_ylim(y0, y1)

    def _set_log_ylim_from_values(ax, values: np.ndarray) -> None:
        vals = np.asarray(values, dtype=np.float64).reshape(-1)
        finite = vals[np.isfinite(vals)]
        positive = finite[finite > 0.0]
        if positive.size == 0:
            return
        lo = max(1e-12, float(np.min(positive)) * 0.9)
        hi = float(np.max(positive)) * 1.1
        if not np.isfinite(lo) or not np.isfinite(hi):
            return
        if hi <= lo:
            hi = lo * 10.0
        ax.set_ylim(lo, hi)

    def _pin_ssim_ymax(ax, ymax: float = 1.0) -> None:
        lo, _ = ax.get_ylim()
        if not np.isfinite(lo) or lo >= ymax:
            lo = 0.0
        ax.set_ylim(lo, ymax)

    autoscale_source_key = _autoscale_key(error_curves_autoscale_source)
    if autoscale_source_key not in ("", "all"):
        selected = autoscale_sources.get(autoscale_source_key)
        if selected is None:
            raise ValueError(
                "--error-curves-autoscale-source must be 'all' or one of: "
                + ", ".join(repr(v) for v in autoscale_source_names)
                + f"; got {error_curves_autoscale_source!r}."
            )
        for metric in selected_metrics:
            ax = metric_axes[metric]
            cfg = metric_cfg[metric]
            values = selected[metric]
            if error_curves_log_y and bool(cfg["log_y"]):
                _set_log_ylim_from_values(ax, values)
            else:
                _set_linear_ylim_from_values(
                    ax,
                    values,
                    nonnegative=(metric != ERROR_CURVE_METRIC_SSIM),
                )

    for metric in selected_metrics:
        if metric == ERROR_CURVE_METRIC_SSIM:
            _pin_ssim_ymax(metric_axes[metric])

    for metric in selected_metrics:
        cfg = metric_cfg[metric]
        _annotate_nonfinite(
            metric_axes[metric],
            gnn_values=np.asarray(cfg["gnn_error"], dtype=np.float64),
            god_values=(
                np.asarray(cfg["god_error"], dtype=np.float64)
                if show_godunov_baseline
                else None
            ),
            extra_values=[
                (b.label, getattr(b, str(cfg["sample_attr"])))
                for b in error_curve_baselines
            ]
            + [
                (m.label, getattr(m, str(cfg["sample_attr"])))
                for m in extra_models
            ],
        )

    title_lines = ["1D Inviscid Burger's Equation: Rollout Accuracy"]
    #if show_godunov_baseline:
        #title_lines.append(
        #    f"Excluded {n_god_failed_excluded}/{n_error_curve_samples} "
        #    f"test trajectories where {godunov_label} produced nonfinite RMSE."
        #)
    fig.suptitle("\n".join(title_lines), fontsize=20)
    fig.savefig(plots_dir / "error_curves_compare.png", dpi=170)
    plt.close(fig)

    # 2) Final-step distributions.
    if rmse_gnn.shape[0] > 0:
        fig2, axes2 = plt.subplots(1, 3, figsize=(15.5, 4.3), constrained_layout=True)
        ax20, ax21, ax22 = axes2

        final_rmse_gnn = rmse_gnn[:, -1]
        final_rmse_god = rmse_god[:, -1]
        final_rel_gnn = rel_gnn[:, -1]
        final_rel_god = rel_god[:, -1]
        final_l1_gnn = l1_gnn[:, -1]
        final_l1_god = l1_god[:, -1]

        def _finite(a: np.ndarray) -> np.ndarray:
            return a[np.isfinite(a)]

        frg = _finite(final_rmse_gnn)
        fru = _finite(final_rmse_god)
        flg = _finite(final_rel_gnn)
        flu = _finite(final_rel_god)
        f1g = _finite(final_l1_gnn)
        f1u = _finite(final_l1_god)

        if frg.size > 0:
            ax20.hist(frg, bins=min(20, max(5, frg.size)), alpha=0.6, color=gnn_color, label=gnn_label)
        if show_godunov_baseline and fru.size > 0:
            ax20.hist(fru, bins=min(20, max(5, fru.size)), alpha=0.6, color=godunov_color, label=godunov_label)
        if flg.size > 0:
            ax21.hist(flg, bins=min(20, max(5, flg.size)), alpha=0.6, color=gnn_color, label=gnn_label)
        if show_godunov_baseline and flu.size > 0:
            ax21.hist(flu, bins=min(20, max(5, flu.size)), alpha=0.6, color=godunov_color, label=godunov_label)
        if f1g.size > 0:
            ax22.hist(f1g, bins=min(20, max(5, f1g.size)), alpha=0.6, color=gnn_color, label=gnn_label)
        if show_godunov_baseline and f1u.size > 0:
            ax22.hist(f1u, bins=min(20, max(5, f1u.size)), alpha=0.6, color=godunov_color, label=godunov_label)

        for j, baseline in enumerate(extra_baselines):
            c_base = _baseline_color(j, baseline.scheme)
            f_rmse = _finite(baseline.rmse_by_sample[:, -1])
            f_rel = _finite(baseline.rel_by_sample[:, -1])
            f_l1 = _finite(baseline.l1_by_sample[:, -1])
            if f_rmse.size > 0:
                ax20.hist(
                    f_rmse,
                    bins=min(20, max(5, f_rmse.size)),
                    alpha=0.45,
                    color=c_base,
                    label=baseline.label,
                )
            if f_rel.size > 0:
                ax21.hist(
                    f_rel,
                    bins=min(20, max(5, f_rel.size)),
                    alpha=0.45,
                    color=c_base,
                    label=baseline.label,
                )
            if f_l1.size > 0:
                ax22.hist(
                    f_l1,
                    bins=min(20, max(5, f_l1.size)),
                    alpha=0.45,
                    color=c_base,
                    label=baseline.label,
                )

        for j, extra in enumerate(extra_models):
            c = _extra_model_color(j)
            f_rmse = _finite(extra.rmse_by_sample[:, -1])
            f_rel = _finite(extra.rel_by_sample[:, -1])
            f_l1 = _finite(extra.l1_by_sample[:, -1])
            if f_rmse.size > 0:
                ax20.hist(
                    f_rmse,
                    bins=min(20, max(5, f_rmse.size)),
                    alpha=0.35,
                    color=c,
                    label=extra.label,
                )
            if f_rel.size > 0:
                ax21.hist(
                    f_rel,
                    bins=min(20, max(5, f_rel.size)),
                    alpha=0.35,
                    color=c,
                    label=extra.label,
                )
            if f_l1.size > 0:
                ax22.hist(
                    f_l1,
                    bins=min(20, max(5, f_l1.size)),
                    alpha=0.35,
                    color=c,
                    label=extra.label,
                )

        ax20.set_title("Final-Step RMSE")
        ax21.set_title("Final-Step rel-L2")
        ax22.set_title("Final-Step L1")
        for ax in axes2:
            ax.set_xlabel("error")
            ax.set_ylabel("count")
            ax.grid(alpha=0.25)
            ax.legend(loc="upper right")
        fig2.savefig(plots_dir / "final_step_hist_compare.png", dpi=170)
        plt.close(fig2)

    # 3) Summed field totals over rollout steps (single trajectory per figure).
    pages = [] if sumplot_pages is None else list(sumplot_pages)
    if len(pages) > 0:
        extra_model_label_set = {m.label for m in extra_models}
        extra_model_color_by_label = {
            m.label: _extra_model_color(i) for i, m in enumerate(extra_models)
        }
        baseline_color_by_label = {
            baseline.label: _baseline_color(i, baseline.scheme)
            for i, baseline in enumerate(extra_baselines)
        }

        def _build_sumtot_figure(page: Dict[str, Any]):
            sum_true = np.asarray(page["sum_true"], dtype=np.float64).reshape(-1)
            sum_gnn = np.asarray(page["sum_gnn"], dtype=np.float64).reshape(-1)
            sum_god = np.asarray(page["sum_god"], dtype=np.float64).reshape(-1)
            t_true = np.arange(sum_true.shape[0], dtype=np.int64)
            t_gnn = np.arange(sum_gnn.shape[0], dtype=np.int64)
            t_god = np.arange(sum_god.shape[0], dtype=np.int64)

            #fig_tot, ax_tot = plt.subplots(1, 1, figsize=(10.5, 4.8), constrained_layout=True)
            SUMMED_FIELD_FIG_WIDTH = 8.0
            SUMMED_FIELD_FIG_HEIGHT = 4.8

            fig_tot, ax_tot = plt.subplots(
                1,
                1,
                figsize=(SUMMED_FIELD_FIG_WIDTH, SUMMED_FIELD_FIG_HEIGHT),
                constrained_layout=True,
            )
            if summed_field_show_exact:
                ax_tot.plot(t_true, sum_true, color="k", lw=2.2, label="Exact")
            ax_tot.plot(t_gnn, sum_gnn, color=gnn_color, lw=2.0, label=gnn_label)
            if show_godunov_baseline:
                ax_tot.plot(
                    t_god,
                    sum_god,
                    color=godunov_color,
                    lw=2.0,
                    ls="--",
                    label=godunov_label,
                )

            compare_map = page.get("compare_sum_by_label", {})
            if isinstance(compare_map, dict):
                for j, (lbl, curve) in enumerate(compare_map.items()):
                    if curve is None:
                        continue
                    label = str(lbl)
                    c = extra_model_color_by_label.get(
                        label,
                        baseline_color_by_label.get(label, f"C{(j + 3) % 10}"),
                    )
                    arr = np.asarray(curve, dtype=np.float64).reshape(-1)
                    t_arr = np.arange(arr.shape[0], dtype=np.int64)
                    ax_tot.plot(
                        t_arr,
                        arr,
                        color=c,
                        lw=2.0,
                        ls="-" if label in extra_model_label_set else "--",
                        label=label,
                    )

            pid = int(page.get("pair_id", -1))
            local_tid = int(page.get("local_traj_id", -1))
            global_tid = int(page.get("global_traj_id", -1))
            t0 = int(page.get("time_index", -1))
            #ax_tot.set_title(
            #    "Summed Field Total vs Rollout Step\n"
            #    f"pair_index={pid}, local_traj={local_tid}, global_traj={global_tid}, t0={t0}"
            #)
            #ax_tot.set_xlabel("rollout step")
            #ax_tot.set_ylabel("sum(u)")
            ax_tot.set_title(f"Inviscid Burgers: Field Total vs Time", fontsize=20)
            ax_tot.set_xlabel("Time Step", fontsize=15)
            ax_tot.set_ylabel(r"$\sum_i u_i$", fontsize=15)
            ax_tot.grid(alpha=0.3)
            ax_tot.legend(loc="best", fontsize=15)
            ax_tot.tick_params(axis="x", labelsize=14)
            ax_tot.tick_params(axis="y", labelsize=14)
            return fig_tot

        fig0 = _build_sumtot_figure(pages[0])
        fig0.savefig(plots_dir / "summed_field_totals_compare.png", dpi=170)
        plt.close(fig0)

        from matplotlib.backends.backend_pdf import PdfPages

        pdf_sum = plots_dir / "summed_field_totals_compare_all.pdf"
        with PdfPages(pdf_sum) as pdf:
            for page in pages:
                figp = _build_sumtot_figure(page)
                pdf.savefig(figp, dpi=170)
                plt.close(figp)

    def _build_spacetime_compare_figure(
        *,
        sid: int,
        global_tid: int | None,
        local_tid: int | None,
        time_index: int | None,
        true_arr: np.ndarray,
        gnn_arr: np.ndarray,
        baseline_rows: List[Tuple[str, np.ndarray]],
        title_prefix: str,
    ):
        true_plot = _replace_nonfinite_with_nan(true_arr)
        gnn_plot = _replace_nonfinite_with_nan(gnn_arr)
        err_gnn = np.abs(gnn_plot - true_plot)
        baseline_plots = [
            (label, _replace_nonfinite_with_nan(arr))
            for label, arr in baseline_rows
        ]
        baseline_errs = [
            (label, np.abs(arr - true_plot))
            for label, arr in baseline_plots
        ]

        true_final = true_plot[-1]
        gnn_final = gnn_plot[-1]
        gnn_rmse_final = _rmse_1d(gnn_final, true_final)
        gnn_ssim_final = _ssim_1d(
            gnn_final,
            true_final,
            config=ssim_config,
        )

        t_axis = np.arange(true_plot.shape[0], dtype=np.float64) * float(dt)
        extent = [float(pos.min()), float(pos.max()), float(t_axis.max()), float(t_axis.min())]

        gt_finite = true_plot[np.isfinite(true_plot)]
        if gt_finite.size == 0:
            gt_min, gt_max = -1.0, 1.0
        else:
            gt_min = float(np.min(gt_finite))
            gt_max = float(np.max(gt_finite))
            if not np.isfinite(gt_min) or not np.isfinite(gt_max):
                gt_min, gt_max = -1.0, 1.0
        if gt_max <= gt_min:
            eps = max(1e-6, abs(gt_min) * 1e-6)
            gt_min -= eps
            gt_max += eps

        def _safe_nanmax(arr: np.ndarray) -> float:
            vals = np.asarray(arr, dtype=np.float64)
            finite = vals[np.isfinite(vals)]
            return float(np.max(finite)) if finite.size > 0 else 0.0

        err_max = max(
            _safe_nanmax(err_gnn),
            *[_safe_nanmax(err) for _, err in baseline_errs],
            1e-8,
        )

        n_extra = max(0, len(baseline_rows) - 1)
        n_rows = 1 + int(math.ceil(n_extra / 2.0))
        n_cols = 3 if len(baseline_rows) == 0 else 5
        fig_width = 13.3 if n_cols == 3 else 22.2
        fig_height = max(4.05, 4.05 * n_rows)
        subplot_wspace = 0.36 if n_cols == 3 else 0.40
        fig3, axes3 = plt.subplots(
            n_rows,
            n_cols,
            figsize=(fig_width, fig_height),
            constrained_layout=False,
            squeeze=False,
        )
        fig3.subplots_adjust(
            left=0.055,
            right=0.935,
            bottom=0.13,
            top=0.875,
            wspace=subplot_wspace,
            hspace=0.48,
        )
        state_axes = []
        err_axes = []
        state_im = None
        err_im = None

        def _baseline_final_label(baseline_plot: np.ndarray) -> str:
            baseline_final = baseline_plot[-1]
            baseline_rmse_final = _rmse_1d(baseline_final, true_final)
            baseline_ssim_final = _ssim_1d(
                baseline_final,
                true_final,
                config=ssim_config,
            )
            #return (
            #    "x\n"
            #    f"Final RMSE={_fmt_metric(baseline_rmse_final, '.3e')}, "
            #    f"Final SSIM={_fmt_metric(baseline_ssim_final, '.4f')}"
            #)
            return ("Position")

        axes_row = axes3[0]

        state_im = axes_row[0].imshow(
            true_plot,
            aspect="auto",
            extent=extent,
            cmap="viridis",
            vmin=gt_min,
            vmax=gt_max,
        )
        axes_row[1].imshow(
            gnn_plot,
            aspect="auto",
            extent=extent,
            cmap="viridis",
            vmin=gt_min,
            vmax=gt_max,
        )
        gnn_err_col = 2 if n_cols == 3 else 3
        err_im = axes_row[gnn_err_col].imshow(
            err_gnn,
            aspect="auto",
            extent=extent,
            cmap="magma",
            vmin=0.0,
            vmax=err_max,
        )
        titles = ["Exact", f"{gnn_label} Predicted", f"|{gnn_label} - Exact|"]
        if len(baseline_plots) > 0:
            primary_label, primary_plot = baseline_plots[0]
            _, primary_err = baseline_errs[0]
            axes_row[2].imshow(
                primary_plot,
                aspect="auto",
                extent=extent,
                cmap="viridis",
                vmin=gt_min,
                vmax=gt_max,
            )
            axes_row[4].imshow(
                primary_err,
                aspect="auto",
                extent=extent,
                cmap="magma",
                vmin=0.0,
                vmax=err_max,
            )
            titles = [
                "Exact",
                f"{gnn_label} Predicted",
                f"{primary_label} Predicted",
                f"|{gnn_label} - Exact|",
                f"|{primary_label} - Exact|",
            ]
        for ax, tt in zip(axes_row, titles):
            ax.set_title(tt, fontsize=20)
            ax.set_xlabel("Position", fontsize=20)
            ax.set_ylabel("Time", fontsize=20)
        #axes_row[1].set_xlabel(
            #"x\n"
            #f"Final RMSE={_fmt_metric(gnn_rmse_final, '.3e')}, "
            #f"Final SSIM={_fmt_metric(gnn_ssim_final, '.4f')}",
            #fontsize=9,
        #)
        state_axes.extend([axes_row[0], axes_row[1]])
        err_axes.append(axes_row[gnn_err_col])
        if len(baseline_plots) > 0:
            axes_row[2].set_xlabel(_baseline_final_label(baseline_plots[0][1]), fontsize=12)
            state_axes.append(axes_row[2])
            err_axes.append(axes_row[4])

        for row_idx in range(1, n_rows):
            for ax in axes3[row_idx]:
                ax.set_visible(False)

        for extra_idx, ((baseline_label, baseline_plot), (_, err_baseline)) in enumerate(
            zip(baseline_plots[1:], baseline_errs[1:])
        ):
            row_idx = 1 + (extra_idx // 2)
            col_offset = 2 * (extra_idx % 2)
            pred_ax = axes3[row_idx, col_offset]
            err_ax = axes3[row_idx, col_offset + 1]
            pred_ax.set_visible(True)
            err_ax.set_visible(True)

            pred_ax.imshow(
                baseline_plot,
                aspect="auto",
                extent=extent,
                cmap="viridis",
                vmin=gt_min,
                vmax=gt_max,
            )
            err_ax.imshow(
                err_baseline,
                aspect="auto",
                extent=extent,
                cmap="magma",
                vmin=0.0,
                vmax=err_max,
            )
            pred_ax.set_title(f"{baseline_label} Predicted")
            err_ax.set_title(f"|{baseline_label} - Exact|")
            pred_ax.set_xlabel(_baseline_final_label(baseline_plot), fontsize=9)
            err_ax.set_xlabel("x")
            pred_ax.set_ylabel("time")
            err_ax.set_ylabel("time")
            state_axes.append(pred_ax)
            err_axes.append(err_ax)

        '''
        if global_tid is None:
            title = f"{title_prefix} | pair_index={sid}"
        else:
            title = f"{title_prefix} {global_tid} | pair_index={sid}"
        if local_tid is not None:
            title += f"{local_tid}"
        if time_index is not None:
            title += f" | t0={time_index}"
        '''
        title = f"{title_prefix} {local_tid}"
        fig3.suptitle(title, fontsize=11)

        # Place colorbars in reserved subplot gaps. Matplotlib's automatic
        # colorbar aspect can otherwise make narrow error colorbars much shorter
        # than the axes, while unconstrained add_axes avoids stealing plot space.
        fig3.canvas.draw()

        def _add_explicit_colorbar(mappable, axes_list):
            visible_axes = [ax for ax in axes_list if ax.get_visible()]
            if len(visible_axes) == 0:
                return None
            boxes = [ax.get_position() for ax in visible_axes]
            y0 = min(ax.get_position().y0 for ax in visible_axes)
            y1 = max(ax.get_position().y1 for ax in visible_axes)
            x1 = max(box.x1 for box in boxes)
            width = 0.006
            pad = 0.007
            x0 = x1 + pad
            cax = fig3.add_axes([x0, y0, width, y1 - y0])
            cax.set_in_layout(False)
            cbar = fig3.colorbar(mappable, cax=cax)
            if hasattr(cbar.ax, "set_box_aspect"):
                cbar.ax.set_box_aspect(None)
            cbar.ax.set_aspect("auto")
            return cbar

        if state_im is not None:
            _add_explicit_colorbar(state_im, state_axes)
        if err_im is not None:
            _add_explicit_colorbar(err_im, err_axes)
        return fig3

    # 4) Representative sample spacetime.
    if (
        sample_id is not None
        and true_traj is not None
        and gnn_traj is not None
        and god_traj is not None
    ):
        baseline_rows: List[Tuple[str, np.ndarray]] = []
        if show_godunov_baseline:
            baseline_rows.append((godunov_label, god_traj))
        sample_extra_map = (
            {} if sample_extra_baseline_trajs is None else sample_extra_baseline_trajs
        )
        for baseline in extra_baselines:
            traj = sample_extra_map.get(baseline.scheme)
            if traj is not None:
                baseline_rows.append((baseline.label, traj))
        fig3 = _build_spacetime_compare_figure(
            sid=int(sample_id),
            global_tid=sample_global_id,
            local_tid=sample_local_id,
            time_index=sample_time_idx,
            true_arr=true_traj,
            gnn_arr=gnn_traj,
            baseline_rows=baseline_rows,
            title_prefix="Representative Rollout Sample",
        )
        fig3.savefig(plots_dir / "sample_spacetime_compare.png", dpi=180)
        plt.close(fig3)

    # 5) Optional multi-page PDF with one spacetime comparison page per sample.
    if (
        plot_all_samples
        and all_sample_ids is not None
        and all_true_trajs is not None
        and all_gnn_trajs is not None
        and all_god_trajs is not None
        and len(all_sample_ids) == len(all_true_trajs)
        and len(all_sample_ids) == len(all_gnn_trajs)
        and len(all_sample_ids) == len(all_god_trajs)
    ):
        from matplotlib.backends.backend_pdf import PdfPages

        pdf_path = plots_dir / "sample_spacetime_compare_all.pdf"
        with PdfPages(pdf_path) as pdf:
            extra_all_trajs = (
                {} if all_extra_baseline_trajs is None else all_extra_baseline_trajs
            )
            for sample_idx, (sid, true_arr, gnn_arr, god_arr) in enumerate(zip(
                all_sample_ids,
                all_true_trajs,
                all_gnn_trajs,
                all_god_trajs,
            )):
                baseline_rows: List[Tuple[str, np.ndarray]] = []
                if show_godunov_baseline:
                    baseline_rows.append((godunov_label, god_arr))
                for baseline in extra_baselines:
                    scheme_trajs = extra_all_trajs.get(baseline.scheme)
                    if scheme_trajs is None or sample_idx >= len(scheme_trajs):
                        continue
                    baseline_rows.append((baseline.label, scheme_trajs[sample_idx]))
                global_tid = (
                    None
                    if all_sample_global_ids is None or sample_idx >= len(all_sample_global_ids)
                    else int(all_sample_global_ids[sample_idx])
                )
                local_tid = (
                    None
                    if all_sample_local_ids is None or sample_idx >= len(all_sample_local_ids)
                    else int(all_sample_local_ids[sample_idx])
                )
                time_index = (
                    None
                    if all_sample_time_indices is None or sample_idx >= len(all_sample_time_indices)
                    else int(all_sample_time_indices[sample_idx])
                )
                fig3 = _build_spacetime_compare_figure(
                    sid=int(sid),
                    global_tid=global_tid,
                    local_tid=local_tid,
                    time_index=time_index,
                    true_arr=true_arr,
                    gnn_arr=gnn_arr,
                    baseline_rows=baseline_rows,
                    title_prefix="Rollout Sample",
                )
                pdf.savefig(fig3, dpi=180)
                plt.close(fig3)


def main() -> None:
    args = parse_args()
    if int(args.error_curves_smooth_window) < 1:
        raise ValueError(
            "--error-curves-smooth-window must be >= 1, "
            f"got {args.error_curves_smooth_window}."
        )
    error_curves_metrics = _resolve_error_curve_metrics(args.error_curves_metrics)
    plot_baseline_schemes = _resolve_baseline_schemes(args.baseline_schemes)
    baseline_schemes = list(plot_baseline_schemes)
    if BASELINE_SCHEME_GODUNOV not in baseline_schemes:
        baseline_schemes.insert(0, BASELINE_SCHEME_GODUNOV)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]

    data_cfg = dict(cfg.get("data", {}))
    rollout_cfg = dict(cfg.get("rollout_eval", {}))
    target_mode = str(data_cfg.get("target_mode", "delta")).strip().lower()
    regime_filter = str(data_cfg.get("regime_filter", "all")).strip().lower()

    cfg_model = dict(cfg.get("model", {}))
    cfg_model["predict_type"] = predict_type_from_target_mode(target_mode)
    cfg["model"] = cfg_model

    data_dir, source_format, source_options = _load_data_source_from_cfg(
        data_cfg,
        override_data_dir=args.data_dir,
    )

    if args.num_samples is None:
        num_samples = (
            None
            if bool(args.eval_test_trajectories)
            else int(rollout_cfg.get("num_test_samples", 16))
        )
    else:
        num_samples = int(args.num_samples)
    if num_samples is not None and num_samples <= 0:
        raise ValueError(f"--num-samples must be positive, got {num_samples}.")

    dset = load_burgers_dataset(
        data_dir,
        split="test",
        include_rollout_states=True,
        dtype=torch.float32,
        source_format=source_format,
        source_options=source_options,
    )
    dset = filter_burgers_dataset_by_regime(dset, regime_filter)
    assert dset.rollout_states is not None
    source_format_resolved = str(dset.metadata.get("source_format", source_format)).strip().lower()

    print(
        "[INFO] Dataset source: "
        f"source_format={source_format_resolved} | path={data_dir}",
        flush=True,
    )
    print(
        "[INFO] Regime filter: "
        f"{regime_filter} | pairs={len(dset)} | "
        f"trajectories={int(dset.trajectory_ids.shape[0])}",
        flush=True,
    )

    dt = float(dset.metadata["dt"])
    dx = float(dset.metadata["dx"])
    plotted_baseline_names = (
        "none" if len(plot_baseline_schemes) == 0 else ", ".join(plot_baseline_schemes)
    )
    print(
        "[INFO] Finite-volume baselines: "
        f"plotted={plotted_baseline_names} | "
        f"computed={', '.join(baseline_schemes)} | "
        "one explicit step per dataset dt "
        f"(dt={dt:g}, dx={dx:g}).",
        flush=True,
    )
    window = int(dset.rollout_states.shape[1] - 1)
    horizon_default = int(rollout_cfg.get("horizon", 100))
    if bool(args.eval_test_trajectories) and args.horizon is None:
        horizon_requested = window
    else:
        horizon_requested = int(args.horizon) if args.horizon is not None else horizon_default
    horizon = min(horizon_requested, window)
    if horizon_requested > window:
        print(
            f"[INFO] Requested horizon {horizon_requested} exceeds available rollout "
            f"window {window}; using horizon={horizon}.",
            flush=True,
        )
    if horizon < 1:
        raise ValueError(f"Rollout horizon must be >= 1, got {horizon}.")

    input_cfg = resolve_input_feature_cfg(cfg, source_format=source_format_resolved)
    physics_cfg = resolve_physics_input_cfg(cfg)
    if cfg_model.get("state_channel", None) is None:
        u_col = u_column_index(input_cfg)
        if u_col is None:
            raise ValueError(
                "input_features.feature_columns must include 'u' for rollout evaluation."
            )
        cfg_model["state_channel"] = int(u_col)
        cfg["model"] = cfg_model

    if input_cfg.normalize_u_to_minus1_1:
        if input_cfg.u_min is None or input_cfg.u_max is None:
            raise ValueError(
                "input_features.normalize_u_to_minus1_1 is enabled, but "
                "u_min/u_max are missing from checkpoint config."
            )
        print(
            "[INFO] u normalization enabled: "
            f"range=[-1,1], u_min={float(input_cfg.u_min):.6e}, "
            f"u_max={float(input_cfg.u_max):.6e}, "
            f"clip={bool(input_cfg.normalize_clip)}",
            flush=True,
        )
    else:
        print("[INFO] u normalization disabled.", flush=True)
    print(
        "[INFO] Physics inputs: "
        f"enabled={physics_cfg.enabled}, "
        f"include_adv={physics_cfg.include_adv}, "
        f"input_form={physics_cfg.input_form}, "
        f"advection_scheme={physics_cfg.advection_scheme}, "
        f"normalize_adv_to_minus1_1={physics_cfg.normalize_adv_to_minus1_1}, "
        f"normalize_adv_percentile_low={physics_cfg.normalize_adv_percentile_low}, "
        f"normalize_adv_percentile_high={physics_cfg.normalize_adv_percentile_high}",
        flush=True,
    )

    split = ckpt.get("split_indices", {})
    test_effective_raw = (
        split.get("test_effective", None) if isinstance(split, dict) else None
    )

    test_effective = _safe_index_list(test_effective_raw)
    selection_mode = "checkpoint_test_effective"
    if test_effective is None or len(test_effective) == 0:
        if not bool(args.allow_full_test_fallback):
            raise ValueError(
                "checkpoint split_indices.test_effective is missing/empty. "
                "Rerun with --allow-full-test-fallback to use all valid test starts."
            )
        selection_mode = "full_valid_test_fallback"
        test_effective = _default_test_effective_indices(dset, horizon=horizon)

    if len(test_effective) == 0:
        raise ValueError("No rollout starts available for requested horizon.")

    bad_ids = [i for i in test_effective if (i < 0 or i >= len(dset))]
    if bad_ids:
        raise ValueError(
            "Invalid pair indices in selected test set. "
            f"Dataset size={len(dset)}, bad ids (first 10)={bad_ids[:10]}"
        )

    if bool(args.eval_test_trajectories):
        representative_ids = _select_representative_pairs_per_test_trajectory(
            dset=dset,
            candidate_pair_ids=test_effective,
            prefer_time_index=0,
        )
        if len(representative_ids) == 0:
            raise ValueError(
                "No representative test-trajectory starts found for "
                "--eval-test-trajectories."
            )
        if args.num_samples is None:
            sample_ids = representative_ids
            selection_mode = "one_per_test_trajectory_t0_nearest"
        else:
            sample_ids = list(representative_ids[: min(num_samples, len(representative_ids))])
            selection_mode = "one_per_test_trajectory_t0_nearest_limited"
    else:
        sample_ids = list(test_effective[: min(num_samples, len(test_effective))])
    plot_all_sample_ids = sample_ids
    plot_all_selection_mode = "evaluated_samples"
    if bool(args.plot_all_samples):
        if bool(args.eval_test_trajectories):
            plot_all_sample_ids = sample_ids
            plot_all_selection_mode = "evaluated_one_per_test_trajectory"
        else:
            representative_ids = _select_representative_pairs_per_test_trajectory(
                dset=dset,
                candidate_pair_ids=test_effective,
                prefer_time_index=0,
            )
            if len(representative_ids) > 0:
                plot_all_sample_ids = representative_ids
                plot_all_selection_mode = "one_per_test_trajectory_t0_nearest"
        print(
            "[INFO] Plot-all sample selection: "
            f"mode={plot_all_selection_mode} | pages={len(plot_all_sample_ids)}.",
            flush=True,
        )
    print(
        "[INFO] Evaluation sample selection: "
        f"mode={selection_mode} | samples={len(sample_ids)}.",
        flush=True,
    )

    reference_trajs: List[np.ndarray] = []
    for pid in sample_ids:
        local_tid = int(dset.pair_trajectory_id[int(pid)].item())
        t0 = int(dset.pair_time_index[int(pid)].item())
        max_h = int(min(horizon, max(0, window - t0)))
        true_traj = (
            dset.rollout_states[local_tid, t0 : t0 + max_h + 1, :, 0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        reference_trajs.append(true_traj)
    ssim_config = _resolve_ssim_config(
        args,
        reference_trajs=reference_trajs,
        num_nodes=int(dset.rollout_states.shape[2]),
    )
    if ssim_config.mode == SSIM_MODE_LOCAL_PERIODIC:
        assert ssim_config.data_range is not None
        print(
            "[INFO] SSIM: "
            f"mode={ssim_config.mode}, data_range={ssim_config.data_range:.9g} "
            f"({ssim_config.data_range_source}), window={ssim_config.window_size}, "
            f"sigma={ssim_config.window_sigma:g}, boundary=periodic",
            flush=True,
        )
    else:
        print(
            "[INFO] SSIM: mode=legacy-global "
            "(prediction-pair-dependent data range)",
            flush=True,
        )

    device_pref = str(args.device if args.device is not None else cfg.get("device", "auto"))
    device = resolve_device(device_pref)

    in_channels = (
        count_input_feature_channels(input_cfg)
        + count_physics_input_channels(physics_cfg)
    )
    model = build_model_from_config(
        cfg,
        in_channels=in_channels,
        edge_attr_channels=int(dset.edge_attr.shape[-1]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    edge_index = dset.edge_index.detach().cpu().long()
    edge_attr = dset.edge_attr.detach().cpu()

    rmse_gnn_all: List[np.ndarray] = []
    rel_gnn_all: List[np.ndarray] = []
    l1_gnn_all: List[np.ndarray] = []
    ssim_gnn_all: List[np.ndarray] = []
    rmse_god_all: List[np.ndarray] = []
    rel_god_all: List[np.ndarray] = []
    l1_god_all: List[np.ndarray] = []
    ssim_god_all: List[np.ndarray] = []
    extra_baseline_metric_lists: Dict[str, Dict[str, List[np.ndarray]]] = {
        scheme: {"rmse": [], "rel": [], "l1": [], "ssim": []}
        for scheme in baseline_schemes
        if scheme != BASELINE_SCHEME_GODUNOV
    }

    n_eval = int(len(sample_ids))
    n_nodes = int(dset.x_full.shape[1])
    gnn_traj_by_sample = np.full((n_eval, horizon + 1, n_nodes), np.nan, dtype=np.float64)
    god_traj_by_sample = np.full((n_eval, horizon + 1, n_nodes), np.nan, dtype=np.float64)
    true_traj_by_sample = np.full((n_eval, horizon + 1, n_nodes), np.nan, dtype=np.float64)
    valid_steps_by_sample = np.zeros((n_eval,), dtype=np.int64)
    sample_time_index = np.zeros((n_eval,), dtype=np.int64)
    sample_local_traj_id = np.zeros((n_eval,), dtype=np.int64)
    sample_global_traj_id = np.zeros((n_eval,), dtype=np.int64)

    sample_pair_id: int | None = None
    sample_true_traj: np.ndarray | None = None
    sample_gnn_traj: np.ndarray | None = None
    sample_god_traj: np.ndarray | None = None
    sample_extra_baseline_trajs: Dict[str, np.ndarray] = {}
    all_sample_ids: List[int] = []
    all_sample_global_ids: List[int] = []
    all_sample_local_ids: List[int] = []
    all_sample_time_indices: List[int] = []
    all_true_trajs: List[np.ndarray] = []
    all_gnn_trajs: List[np.ndarray] = []
    all_god_trajs: List[np.ndarray] = []
    all_extra_baseline_trajs: Dict[str, List[np.ndarray]] = {
        scheme: []
        for scheme in baseline_schemes
        if scheme != BASELINE_SCHEME_GODUNOV
    }

    for si, pid in enumerate(sample_ids):
        local_tid, t0, global_tid, true_traj, gnn_traj, god_traj = _evaluate_pair_rollout(
            dset=dset,
            pid=int(pid),
            window=window,
            horizon=horizon,
            model=model,
            edge_index=edge_index,
            edge_attr=edge_attr,
            dt=dt,
            dx=dx,
            device=device,
            input_cfg=input_cfg,
            physics_cfg=physics_cfg,
        )

        baseline_trajs: Dict[str, np.ndarray] = {
            BASELINE_SCHEME_GODUNOV: god_traj,
        }
        max_h_sample = int(max(0, true_traj.shape[0] - 1))
        for scheme in baseline_schemes:
            if scheme == BASELINE_SCHEME_GODUNOV:
                continue
            baseline_trajs[scheme] = _rollout_baseline(
                x0=true_traj[0],
                dt=dt,
                dx=dx,
                max_h=max_h_sample,
                scheme=scheme,
            )

        rmse_gnn, rel_gnn, l1_gnn, ssim_gnn = _compute_curves(
            pred_traj=gnn_traj,
            true_traj=true_traj,
            horizon=horizon,
            ssim_config=ssim_config,
        )
        rmse_god, rel_god, l1_god, ssim_god = _compute_curves(
            pred_traj=god_traj,
            true_traj=true_traj,
            horizon=horizon,
            ssim_config=ssim_config,
        )
        for scheme, metrics in extra_baseline_metric_lists.items():
            rmse_b, rel_b, l1_b, ssim_b = _compute_curves(
                pred_traj=baseline_trajs[scheme],
                true_traj=true_traj,
                horizon=horizon,
                ssim_config=ssim_config,
            )
            metrics["rmse"].append(rmse_b)
            metrics["rel"].append(rel_b)
            metrics["l1"].append(l1_b)
            metrics["ssim"].append(ssim_b)

        rmse_gnn_all.append(rmse_gnn)
        rel_gnn_all.append(rel_gnn)
        l1_gnn_all.append(l1_gnn)
        ssim_gnn_all.append(ssim_gnn)
        rmse_god_all.append(rmse_god)
        rel_god_all.append(rel_god)
        l1_god_all.append(l1_god)
        ssim_god_all.append(ssim_god)

        n_states = int(gnn_traj.shape[0])
        valid_steps = max(0, n_states - 1)
        gnn_traj_by_sample[si, :n_states, :] = gnn_traj
        god_traj_by_sample[si, :n_states, :] = god_traj
        true_traj_by_sample[si, :n_states, :] = true_traj
        valid_steps_by_sample[si] = int(valid_steps)
        sample_time_index[si] = int(t0)
        sample_local_traj_id[si] = int(local_tid)
        sample_global_traj_id[si] = int(global_tid)

        if sample_pair_id is None:
            sample_pair_id = int(pid)
            sample_true_traj = true_traj
            sample_gnn_traj = gnn_traj
            sample_god_traj = god_traj
            sample_extra_baseline_trajs = {
                scheme: traj
                for scheme, traj in baseline_trajs.items()
                if scheme != BASELINE_SCHEME_GODUNOV
            }

    if bool(args.plot_all_samples):
        all_sample_ids = []
        all_sample_global_ids = []
        all_sample_local_ids = []
        all_sample_time_indices = []
        all_true_trajs = []
        all_gnn_trajs = []
        all_god_trajs = []
        for pid in plot_all_sample_ids:
            local_tid, t0, global_tid, true_traj, gnn_traj, god_traj = _evaluate_pair_rollout(
                dset=dset,
                pid=int(pid),
                window=window,
                horizon=horizon,
                model=model,
                edge_index=edge_index,
                edge_attr=edge_attr,
                dt=dt,
                dx=dx,
                device=device,
                input_cfg=input_cfg,
                physics_cfg=physics_cfg,
            )
            max_h_sample = int(max(0, true_traj.shape[0] - 1))
            baseline_trajs = {BASELINE_SCHEME_GODUNOV: god_traj}
            for scheme in baseline_schemes:
                if scheme == BASELINE_SCHEME_GODUNOV:
                    continue
                baseline_trajs[scheme] = _rollout_baseline(
                    x0=true_traj[0],
                    dt=dt,
                    dx=dx,
                    max_h=max_h_sample,
                    scheme=scheme,
                )
            all_sample_ids.append(int(pid))
            all_sample_global_ids.append(int(global_tid))
            all_sample_local_ids.append(int(local_tid))
            all_sample_time_indices.append(int(t0))
            all_true_trajs.append(true_traj)
            all_gnn_trajs.append(gnn_traj)
            all_god_trajs.append(god_traj)
            for scheme, trajs in all_extra_baseline_trajs.items():
                trajs.append(baseline_trajs[scheme])

    rmse_gnn_arr_raw = np.stack(rmse_gnn_all, axis=0) if rmse_gnn_all else np.zeros((0, horizon))
    rel_gnn_arr_raw = np.stack(rel_gnn_all, axis=0) if rel_gnn_all else np.zeros((0, horizon))
    l1_gnn_arr_raw = np.stack(l1_gnn_all, axis=0) if l1_gnn_all else np.zeros((0, horizon))
    ssim_gnn_arr_raw = np.stack(ssim_gnn_all, axis=0) if ssim_gnn_all else np.zeros((0, horizon))
    rmse_god_arr_raw = np.stack(rmse_god_all, axis=0) if rmse_god_all else np.zeros((0, horizon))
    rel_god_arr_raw = np.stack(rel_god_all, axis=0) if rel_god_all else np.zeros((0, horizon))
    l1_god_arr_raw = np.stack(l1_god_all, axis=0) if l1_god_all else np.zeros((0, horizon))
    ssim_god_arr_raw = np.stack(ssim_god_all, axis=0) if ssim_god_all else np.zeros((0, horizon))

    rmse_gnn_arr = _replace_nonfinite_with_nan(rmse_gnn_arr_raw)
    rel_gnn_arr = _replace_nonfinite_with_nan(rel_gnn_arr_raw)
    l1_gnn_arr = _replace_nonfinite_with_nan(l1_gnn_arr_raw)
    ssim_gnn_arr = _replace_nonfinite_with_nan(ssim_gnn_arr_raw)
    rmse_god_arr = _replace_nonfinite_with_nan(rmse_god_arr_raw)
    rel_god_arr = _replace_nonfinite_with_nan(rel_god_arr_raw)
    l1_god_arr = _replace_nonfinite_with_nan(l1_god_arr_raw)
    ssim_god_arr = _replace_nonfinite_with_nan(ssim_god_arr_raw)

    mean_rmse_gnn = _nanmean_over_samples(rmse_gnn_arr, horizon)
    mean_rel_gnn = _nanmean_over_samples(rel_gnn_arr, horizon)
    mean_l1_gnn = _nanmean_over_samples(l1_gnn_arr, horizon)
    mean_ssim_gnn = _nanmean_over_samples(ssim_gnn_arr, horizon)
    mean_rmse_god = _nanmean_over_samples(rmse_god_arr, horizon)
    mean_rel_god = _nanmean_over_samples(rel_god_arr, horizon)
    mean_l1_god = _nanmean_over_samples(l1_god_arr, horizon)
    mean_ssim_god = _nanmean_over_samples(ssim_god_arr, horizon)
    extra_baseline_arrays: Dict[str, Dict[str, np.ndarray]] = {}
    for scheme, metrics in extra_baseline_metric_lists.items():
        if len(metrics["rmse"]) == 0:
            rmse_arr = np.zeros((0, horizon), dtype=np.float64)
            rel_arr = np.zeros((0, horizon), dtype=np.float64)
            l1_arr = np.zeros((0, horizon), dtype=np.float64)
            ssim_arr = np.zeros((0, horizon), dtype=np.float64)
        else:
            rmse_arr = _replace_nonfinite_with_nan(np.stack(metrics["rmse"], axis=0))
            rel_arr = _replace_nonfinite_with_nan(np.stack(metrics["rel"], axis=0))
            l1_arr = _replace_nonfinite_with_nan(np.stack(metrics["l1"], axis=0))
            ssim_arr = _replace_nonfinite_with_nan(np.stack(metrics["ssim"], axis=0))
        extra_baseline_arrays[scheme] = {
            "rmse": rmse_arr,
            "rel": rel_arr,
            "l1": l1_arr,
            "ssim": ssim_arr,
            "mean_rmse": _nanmean_over_samples(rmse_arr, horizon),
            "mean_rel": _nanmean_over_samples(rel_arr, horizon),
            "mean_l1": _nanmean_over_samples(l1_arr, horizon),
            "mean_ssim": _nanmean_over_samples(ssim_arr, horizon),
        }

    used_labels: set[str] = set()
    gnn_label = _reserve_unique_label(args.gnn_label, used_labels)
    if BASELINE_SCHEME_GODUNOV in plot_baseline_schemes:
        godunov_label = _reserve_unique_label(args.godunov_label, used_labels)
    else:
        godunov_label = str(args.godunov_label)
    baseline_label_by_scheme: Dict[str, str] = {
        BASELINE_SCHEME_GODUNOV: godunov_label,
    }
    for scheme in baseline_schemes:
        if scheme == BASELINE_SCHEME_GODUNOV:
            continue
        baseline_label_by_scheme[scheme] = _reserve_unique_label(
            _label_arg_for_baseline(args, scheme),
            used_labels,
        )
    extra_baseline_curves: List[BaselineEvalCurves] = []
    for scheme in baseline_schemes:
        if scheme == BASELINE_SCHEME_GODUNOV:
            continue
        arrs = extra_baseline_arrays[scheme]
        extra_baseline_curves.append(
            BaselineEvalCurves(
                scheme=scheme,
                label=baseline_label_by_scheme[scheme],
                rmse_by_sample=arrs["rmse"],
                rel_by_sample=arrs["rel"],
                l1_by_sample=arrs["l1"],
                ssim_by_sample=arrs["ssim"],
                mean_rmse=arrs["mean_rmse"],
                mean_rel=arrs["mean_rel"],
                mean_l1=arrs["mean_l1"],
                mean_ssim=arrs["mean_ssim"],
            )
        )

    extra_eval_curves: List[ExtraEvalCurves] = []
    compare_eval_paths = (
        []
        if args.compare_eval_npz is None
        else [Path(p) for p in args.compare_eval_npz]
    )
    compare_eval_labels = (
        []
        if args.compare_label is None
        else [str(v) for v in args.compare_label]
    )
    if len(compare_eval_labels) > 0 and len(compare_eval_labels) != len(compare_eval_paths):
        raise ValueError(
            "When --compare-label is provided, it must be repeated exactly once per "
            "--compare-eval-npz (same order)."
        )
    if len(compare_eval_paths) > 0:
        for idx, npz_path in enumerate(compare_eval_paths):
            if not npz_path.exists():
                raise FileNotFoundError(
                    f"--compare-eval-npz path does not exist: {npz_path}"
                )
            if idx < len(compare_eval_labels):
                label = _reserve_unique_label(compare_eval_labels[idx], used_labels)
            else:
                label = _derive_extra_label(npz_path, used_labels)
            extra = _load_extra_eval_curves(npz_path, label=label)
            if ERROR_CURVE_METRIC_SSIM in error_curves_metrics:
                _warn_if_ssim_comparison_incompatible(ssim_config, extra)
            extra_eval_curves.append(extra)
            if int(extra.horizon) != int(horizon):
                print(
                    f"[INFO] Comparison model '{label}' horizon={extra.horizon} "
                    f"differs from primary horizon={horizon}; plotting with its "
                    "available horizon.",
                    flush=True,
                )
            print(
                f"[INFO] Loaded comparison model '{label}' from {npz_path}",
                flush=True,
            )

    curve_color_overrides = _resolve_curve_color_overrides(
        args.curve_color,
        gnn_label=gnn_label,
        baseline_label_by_scheme=baseline_label_by_scheme,
        extra_eval_curves=extra_eval_curves,
    )

    # Per-sample final-step "wins".
    gnn_beats_god_rmse, god_beats_gnn_rmse, tie_fraction_rmse = (
        _final_rmse_win_fractions(rmse_gnn_arr, rmse_god_arr)
    )
    baseline_win_fractions: Dict[str, Dict[str, float | None]] = {
        BASELINE_SCHEME_GODUNOV: {
            "gnn_better": (
                None
                if not np.isfinite(gnn_beats_god_rmse)
                else float(gnn_beats_god_rmse)
            ),
            "baseline_better": (
                None
                if not np.isfinite(god_beats_gnn_rmse)
                else float(god_beats_gnn_rmse)
            ),
            "tie": (
                None if not np.isfinite(tie_fraction_rmse) else float(tie_fraction_rmse)
            ),
        }
    }
    for baseline in extra_baseline_curves:
        gnn_frac, baseline_frac, tie_frac = _final_rmse_win_fractions(
            rmse_gnn_arr,
            baseline.rmse_by_sample,
        )
        baseline_win_fractions[baseline.scheme] = {
            "gnn_better": None if not np.isfinite(gnn_frac) else float(gnn_frac),
            "baseline_better": (
                None if not np.isfinite(baseline_frac) else float(baseline_frac)
            ),
            "tie": None if not np.isfinite(tie_frac) else float(tie_frac),
        }

    out_npz = args.out if args.out is not None else (args.checkpoint.parent / "gnn_vs_godunov_eval.npz")
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    npz_payload: Dict[str, Any] = {
        "sample_ids": np.array(sample_ids, dtype=np.int64),
        "horizon": np.int64(horizon),
        "ssim_mode": np.array(ssim_config.mode),
        "ssim_data_range": np.float64(
            np.nan if ssim_config.data_range is None else ssim_config.data_range
        ),
        "ssim_data_range_source": np.array(ssim_config.data_range_source),
        "ssim_window_size": np.int64(
            ssim_config.window_size
            if ssim_config.mode == SSIM_MODE_LOCAL_PERIODIC
            else 0
        ),
        "ssim_window_sigma": np.float64(
            ssim_config.window_sigma
            if ssim_config.mode == SSIM_MODE_LOCAL_PERIODIC
            else np.nan
        ),
        "ssim_boundary": np.array(
            "periodic"
            if ssim_config.mode == SSIM_MODE_LOCAL_PERIODIC
            else "single-global-window"
        ),
        "ssim_covariance": np.array("population"),
        "ssim_k1": np.float64(ssim_config.k1),
        "ssim_k2": np.float64(ssim_config.k2),
        "baseline_schemes": np.array(plot_baseline_schemes, dtype=str),
        "computed_baseline_schemes": np.array(baseline_schemes, dtype=str),
        "baseline_labels": np.array(
            [baseline_label_by_scheme[s] for s in plot_baseline_schemes],
            dtype=str,
        ),
        "rmse_gnn_by_sample": rmse_gnn_arr,
        "rel_l2_gnn_by_sample": rel_gnn_arr,
        "l1_gnn_by_sample": l1_gnn_arr,
        "ssim_gnn_by_sample": ssim_gnn_arr,
        "rmse_godunov_by_sample": rmse_god_arr,
        "rel_l2_godunov_by_sample": rel_god_arr,
        "l1_godunov_by_sample": l1_god_arr,
        "ssim_godunov_by_sample": ssim_god_arr,
        "mean_rmse_gnn": mean_rmse_gnn,
        "mean_rel_l2_gnn": mean_rel_gnn,
        "mean_l1_gnn": mean_l1_gnn,
        "mean_ssim_gnn": mean_ssim_gnn,
        "mean_rmse_godunov": mean_rmse_god,
        "mean_rel_l2_godunov": mean_rel_god,
        "mean_l1_godunov": mean_l1_god,
        "mean_ssim_godunov": mean_ssim_god,
        "x_nodes": dset.pos.detach().cpu().numpy().reshape(-1),
        "sample_time_index": sample_time_index,
        "sample_local_traj_id": sample_local_traj_id,
        "sample_global_traj_id": sample_global_traj_id,
        "sample_valid_steps": valid_steps_by_sample,
        "sample_gnn_traj_by_sample": gnn_traj_by_sample,
        "sample_godunov_traj_by_sample": god_traj_by_sample,
        "sample_true_traj_by_sample": true_traj_by_sample,
        "sample_pair_id": np.int64(-1 if sample_pair_id is None else sample_pair_id),
        "sample_true_traj": (
            np.array([]) if sample_true_traj is None else sample_true_traj
        ),
        "sample_gnn_traj": (
            np.array([]) if sample_gnn_traj is None else sample_gnn_traj
        ),
        "sample_godunov_traj": (
            np.array([]) if sample_god_traj is None else sample_god_traj
        ),
        "curve_color_override_keys": np.array(list(curve_color_overrides.keys())),
        "curve_color_override_values": np.array(list(curve_color_overrides.values())),
    }
    for baseline in extra_baseline_curves:
        prefix = baseline.scheme
        npz_payload[f"rmse_{prefix}_by_sample"] = baseline.rmse_by_sample
        npz_payload[f"rel_l2_{prefix}_by_sample"] = baseline.rel_by_sample
        npz_payload[f"l1_{prefix}_by_sample"] = baseline.l1_by_sample
        npz_payload[f"ssim_{prefix}_by_sample"] = baseline.ssim_by_sample
        npz_payload[f"mean_rmse_{prefix}"] = baseline.mean_rmse
        npz_payload[f"mean_rel_l2_{prefix}"] = baseline.mean_rel
        npz_payload[f"mean_l1_{prefix}"] = baseline.mean_l1
        npz_payload[f"mean_ssim_{prefix}"] = baseline.mean_ssim
        if baseline.scheme in sample_extra_baseline_trajs:
            npz_payload[f"sample_{prefix}_traj"] = sample_extra_baseline_trajs[
                baseline.scheme
            ]
    np.savez_compressed(out_npz, **npz_payload)

    summary = {
        "checkpoint": str(args.checkpoint),
        "data_dir": str(data_dir),
        "source_format": source_format_resolved,
        "regime_filter": regime_filter,
        "sample_selection_mode": selection_mode,
        "eval_test_trajectories": bool(args.eval_test_trajectories),
        "allow_full_test_fallback": bool(args.allow_full_test_fallback),
        "device_requested": device_pref,
        "device_resolved": str(device),
        "num_samples": int(len(sample_ids)),
        "horizon": int(horizon),
        "dt": float(dt),
        "dx": float(dx),
        "ssim": {
            "mode": ssim_config.mode,
            "data_range": (
                None
                if ssim_config.data_range is None
                else float(ssim_config.data_range)
            ),
            "data_range_source": ssim_config.data_range_source,
            "window_size": (
                int(ssim_config.window_size)
                if ssim_config.mode == SSIM_MODE_LOCAL_PERIODIC
                else None
            ),
            "window_sigma": (
                float(ssim_config.window_sigma)
                if ssim_config.mode == SSIM_MODE_LOCAL_PERIODIC
                else None
            ),
            "boundary": (
                "periodic"
                if ssim_config.mode == SSIM_MODE_LOCAL_PERIODIC
                else "single-global-window"
            ),
            "covariance": "population",
            "k1": float(ssim_config.k1),
            "k2": float(ssim_config.k2),
        },
        "baseline_schemes": list(plot_baseline_schemes),
        "computed_baseline_schemes": list(baseline_schemes),
        "baseline_labels": {
            scheme: baseline_label_by_scheme[scheme]
            for scheme in plot_baseline_schemes
        },
        "model_in_channels": int(in_channels),
        "input_features": {
            "feature_columns": list(input_cfg.feature_columns),
            "use_pos": bool(input_cfg.use_pos),
            "normalize_u_to_minus1_1": bool(input_cfg.normalize_u_to_minus1_1),
            "normalize_clip": bool(input_cfg.normalize_clip),
            "u_min": None if input_cfg.u_min is None else float(input_cfg.u_min),
            "u_max": None if input_cfg.u_max is None else float(input_cfg.u_max),
        },
        "physics_inputs": {
            "enabled": bool(physics_cfg.enabled),
            "include_adv": bool(physics_cfg.include_adv),
            "input_form": str(physics_cfg.input_form),
            "advection_scheme": str(physics_cfg.advection_scheme),
            "adv_all_steps": bool(physics_cfg.adv_all_steps),
            "input_weighted": bool(physics_cfg.input_weighted),
            "adv_weight": float(physics_cfg.adv_weight),
            "normalize_adv_to_minus1_1": bool(physics_cfg.normalize_adv_to_minus1_1),
            "normalize_adv_percentile_low": float(physics_cfg.normalize_adv_percentile_low),
            "normalize_adv_percentile_high": float(physics_cfg.normalize_adv_percentile_high),
            "normalize_adv_clip": bool(physics_cfg.normalize_adv_clip),
            "adv_min": None if physics_cfg.adv_min is None else float(physics_cfg.adv_min),
            "adv_max": None if physics_cfg.adv_max is None else float(physics_cfg.adv_max),
            "detach_inputs": bool(physics_cfg.detach_inputs),
        },
        "final_mean_rmse_gnn": _safe_nanmean_last(rmse_gnn_arr),
        "final_mean_rmse_godunov": _safe_nanmean_last(rmse_god_arr),
        "final_mean_rel_l2_gnn": _safe_nanmean_last(rel_gnn_arr),
        "final_mean_rel_l2_godunov": _safe_nanmean_last(rel_god_arr),
        "final_mean_l1_gnn": _safe_nanmean_last(l1_gnn_arr),
        "final_mean_l1_godunov": _safe_nanmean_last(l1_god_arr),
        "final_mean_ssim_gnn": _safe_nanmean_last(ssim_gnn_arr),
        "final_mean_ssim_godunov": _safe_nanmean_last(ssim_god_arr),
        "final_mean_rmse_baselines": {
            BASELINE_SCHEME_GODUNOV: _safe_nanmean_last(rmse_god_arr),
            **{
                baseline.scheme: _safe_nanmean_last(baseline.rmse_by_sample)
                for baseline in extra_baseline_curves
            },
        },
        "final_mean_rel_l2_baselines": {
            BASELINE_SCHEME_GODUNOV: _safe_nanmean_last(rel_god_arr),
            **{
                baseline.scheme: _safe_nanmean_last(baseline.rel_by_sample)
                for baseline in extra_baseline_curves
            },
        },
        "final_mean_l1_baselines": {
            BASELINE_SCHEME_GODUNOV: _safe_nanmean_last(l1_god_arr),
            **{
                baseline.scheme: _safe_nanmean_last(baseline.l1_by_sample)
                for baseline in extra_baseline_curves
            },
        },
        "final_mean_ssim_baselines": {
            BASELINE_SCHEME_GODUNOV: _safe_nanmean_last(ssim_god_arr),
            **{
                baseline.scheme: _safe_nanmean_last(baseline.ssim_by_sample)
                for baseline in extra_baseline_curves
            },
        },
        "gnn_beats_godunov_fraction_final_rmse": (
            None if not np.isfinite(gnn_beats_god_rmse) else float(gnn_beats_god_rmse)
        ),
        "godunov_beats_gnn_fraction_final_rmse": (
            None if not np.isfinite(god_beats_gnn_rmse) else float(god_beats_gnn_rmse)
        ),
        "tie_fraction_final_rmse": (
            None if not np.isfinite(tie_fraction_rmse) else float(tie_fraction_rmse)
        ),
        "gnn_vs_baseline_fraction_final_rmse": baseline_win_fractions,
        "plot_all_samples": bool(args.plot_all_samples),
        "plot_all_samples_selection_mode": (
            str(plot_all_selection_mode) if bool(args.plot_all_samples) else None
        ),
        "plot_all_samples_count": (
            int(len(plot_all_sample_ids)) if bool(args.plot_all_samples) else 0
        ),
        "error_curves_log_y": bool(args.error_curves_log_y),
        "error_curves_mean_only": bool(args.error_curves_mean_only),
        "error_curves_spread": str(args.error_curves_spread),
        "error_curves_smooth_window": int(args.error_curves_smooth_window),
        "error_curves_metrics": list(error_curves_metrics),
        "error_curves_autoscale_source": str(args.error_curves_autoscale_source),
        "curve_color_overrides": dict(curve_color_overrides),
        "summed_field_show_exact": not bool(args.summed_field_exclude_exact),
        "legend_label_gnn": gnn_label,
        "legend_label_godunov": godunov_label,
        "num_compare_models": int(len(extra_eval_curves)),
        "compare_model_labels": [m.label for m in extra_eval_curves],
        "compare_eval_npz": [m.source_npz for m in extra_eval_curves],
        "out_npz": str(out_npz),
    }

    plots_dir = args.plots_dir if args.plots_dir is not None else (args.checkpoint.parent / "gnn_vs_godunov_plots")
    sumplot_pages: List[Dict[str, Any]] = []
    if not args.no_plots:
        sumplot_pair_ids = _select_t0_pairs_per_test_trajectory(dset=dset)
        if len(sumplot_pair_ids) == 0:
            print(
                "[WARN] No t=0 test trajectory starts found for summed-field plots.",
                flush=True,
            )
        else:
            for pid in sumplot_pair_ids:
                local_tid, t0, global_tid, true_traj, gnn_traj, god_traj = _evaluate_pair_rollout(
                    dset=dset,
                    pid=int(pid),
                    window=window,
                    horizon=window,
                    model=model,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    dt=dt,
                    dx=dx,
                    device=device,
                    input_cfg=input_cfg,
                    physics_cfg=physics_cfg,
                )
                max_h_sample = int(max(0, true_traj.shape[0] - 1))
                baseline_trajs = {BASELINE_SCHEME_GODUNOV: god_traj}
                for baseline in extra_baseline_curves:
                    baseline_trajs[baseline.scheme] = _rollout_baseline(
                        x0=true_traj[0],
                        dt=dt,
                        dx=dx,
                        max_h=max_h_sample,
                        scheme=baseline.scheme,
                    )
                compare_sum_by_label: Dict[str, np.ndarray | None] = {}
                for baseline in extra_baseline_curves:
                    compare_sum_by_label[baseline.label] = _sum_field_curve(
                        baseline_trajs[baseline.scheme]
                    )
                for extra in extra_eval_curves:
                    compare_sum_by_label[extra.label] = _lookup_extra_sum_curve_for_traj(
                        extra,
                        global_tid=global_tid,
                        time_index=t0,
                    )
                sumplot_pages.append(
                    {
                        "pair_id": int(pid),
                        "local_traj_id": int(local_tid),
                        "global_traj_id": int(global_tid),
                        "time_index": int(t0),
                        "sum_true": _sum_field_curve(true_traj),
                        "sum_gnn": _sum_field_curve(gnn_traj),
                        "sum_god": _sum_field_curve(god_traj),
                        "compare_sum_by_label": compare_sum_by_label,
                    }
                )
            print(
                "[INFO] Summed-field trajectories: "
                f"{len(sumplot_pages)} test trajectories (t0=0, full window).",
                flush=True,
            )

    if not args.no_plots:
        pos = dset.pos.detach().cpu().numpy().reshape(-1)
        _write_plots(
            plots_dir=plots_dir,
            horizon=horizon,
            dt=dt,
            rmse_gnn=rmse_gnn_arr,
            rel_gnn=rel_gnn_arr,
            l1_gnn=l1_gnn_arr,
            ssim_gnn=ssim_gnn_arr,
            rmse_god=rmse_god_arr,
            rel_god=rel_god_arr,
            l1_god=l1_god_arr,
            ssim_god=ssim_god_arr,
            mean_rmse_gnn=mean_rmse_gnn,
            mean_rel_gnn=mean_rel_gnn,
            mean_l1_gnn=mean_l1_gnn,
            mean_ssim_gnn=mean_ssim_gnn,
            mean_rmse_god=mean_rmse_god,
            mean_rel_god=mean_rel_god,
            mean_l1_god=mean_l1_god,
            mean_ssim_god=mean_ssim_god,
            gnn_label=gnn_label,
            godunov_label=godunov_label,
            plot_baseline_schemes=plot_baseline_schemes,
            pos=pos,
            sample_id=sample_pair_id,
            sample_global_id=(
                int(sample_global_traj_id[0])
                if sample_pair_id is not None and sample_global_traj_id.size > 0
                else None
            ),
            sample_local_id=(
                int(sample_local_traj_id[0])
                if sample_pair_id is not None and sample_local_traj_id.size > 0
                else None
            ),
            sample_time_idx=(
                int(sample_time_index[0])
                if sample_pair_id is not None and sample_time_index.size > 0
                else None
            ),
            true_traj=sample_true_traj,
            gnn_traj=sample_gnn_traj,
            god_traj=sample_god_traj,
            sample_extra_baseline_trajs=sample_extra_baseline_trajs,
            plot_all_samples=bool(args.plot_all_samples),
            all_sample_ids=all_sample_ids,
            all_sample_global_ids=all_sample_global_ids,
            all_sample_local_ids=all_sample_local_ids,
            all_sample_time_indices=all_sample_time_indices,
            all_true_trajs=all_true_trajs,
            all_gnn_trajs=all_gnn_trajs,
            all_god_trajs=all_god_trajs,
            all_extra_baseline_trajs=all_extra_baseline_trajs,
            error_curves_log_y=bool(args.error_curves_log_y),
            error_curves_mean_only=bool(args.error_curves_mean_only),
            error_curves_spread=str(args.error_curves_spread),
            error_curves_smooth_window=int(args.error_curves_smooth_window),
            error_curves_metrics=error_curves_metrics,
            ssim_config=ssim_config,
            error_curves_autoscale_source=str(args.error_curves_autoscale_source),
            curve_colors=curve_color_overrides,
            summed_field_show_exact=not bool(args.summed_field_exclude_exact),
            extra_baseline_curves=extra_baseline_curves,
            extra_eval_curves=extra_eval_curves,
            sumplot_pages=sumplot_pages,
        )
        summary["plots_dir"] = str(plots_dir)
        summary["summed_field_totals_plot"] = str(
            plots_dir / "summed_field_totals_compare.png"
        )
        summary["summed_field_totals_num_trajectories"] = int(len(sumplot_pages))
        if len(sumplot_pages) > 0:
            summary["summed_field_totals_all_pdf"] = str(
                plots_dir / "summed_field_totals_compare_all.pdf"
            )
        if args.plot_all_samples:
            summary["all_sample_plots_pdf"] = str(plots_dir / "sample_spacetime_compare_all.pdf")

    save_json(summary, args.checkpoint.parent / "gnn_vs_godunov_summary.json")

    print("GNN vs finite-volume baseline comparison complete.")
    print(f"  samples: {len(sample_ids)}")
    print(f"  horizon: {horizon}")
    print(f"  device: {device}")
    print(f"  SSIM mode: {ssim_config.mode}")
    if ssim_config.data_range is not None:
        print(
            "  SSIM data range: "
            f"{ssim_config.data_range:.9g} ({ssim_config.data_range_source})"
        )
    plotted_baselines = (
        "none"
        if len(plot_baseline_schemes) == 0
        else ", ".join(baseline_label_by_scheme[s] for s in plot_baseline_schemes)
    )
    print(f"  baselines: {plotted_baselines}")
    print(
        "  computed baselines: "
        + ", ".join(baseline_label_by_scheme[s] for s in baseline_schemes)
    )
    print(f"  out: {out_npz}")
    if not args.no_plots:
        print(f"  plots: {plots_dir}")
        print(f"  summed field totals plot: {plots_dir / 'summed_field_totals_compare.png'}")
        print(f"  summed field trajectories: {len(sumplot_pages)}")
        if len(sumplot_pages) > 0:
            print(
                "  summed-field totals PDF: "
                f"{plots_dir / 'summed_field_totals_compare_all.pdf'}"
            )
        if bool(args.summed_field_exclude_exact):
            print("  summed-field Exact curve: excluded")
        print(f"  error-curve smoothing window: {int(args.error_curves_smooth_window)}")
        print(f"  error-curve spread: {args.error_curves_spread}")
        print(f"  error-curve metrics: {', '.join(error_curves_metrics)}")
        print(f"  error-curve autoscale source: {args.error_curves_autoscale_source}")
        if len(curve_color_overrides) > 0:
            print(
                "  curve color overrides: "
                + ", ".join(
                    f"{key}={color}"
                    for key, color in curve_color_overrides.items()
                )
            )
        if args.plot_all_samples:
            print(
                "  all-sample spacetime PDF: "
                f"{plots_dir / 'sample_spacetime_compare_all.pdf'}"
            )
            print(
                "  plot-all selection: "
                f"{plot_all_selection_mode} ({len(plot_all_sample_ids)} pages)"
            )

    if summary["final_mean_rmse_gnn"] is not None:
        print(f"  final mean RMSE ({gnn_label}): {summary['final_mean_rmse_gnn']:.6e}")
    show_godunov_in_output = BASELINE_SCHEME_GODUNOV in plot_baseline_schemes
    if show_godunov_in_output and summary["final_mean_rmse_godunov"] is not None:
        print(
            f"  final mean RMSE ({godunov_label}): "
            f"{summary['final_mean_rmse_godunov']:.6e}"
        )
    for baseline in extra_baseline_curves:
        mean_rmse = summary["final_mean_rmse_baselines"].get(baseline.scheme)
        if mean_rmse is not None:
            print(f"  final mean RMSE ({baseline.label}): {float(mean_rmse):.6e}")

    mean_gnn = summary["final_mean_rmse_gnn"]
    mean_god = summary["final_mean_rmse_godunov"]
    if show_godunov_in_output and (mean_gnn is not None) and (mean_god is not None):
        if float(mean_gnn) < float(mean_god):
            print(f"  mean final RMSE winner: {gnn_label}")
        elif float(mean_god) < float(mean_gnn):
            print(f"  mean final RMSE winner: {godunov_label}")
        else:
            print("  mean final RMSE winner: tie")

    frac_gnn = summary["gnn_beats_godunov_fraction_final_rmse"]
    frac_god = summary["godunov_beats_gnn_fraction_final_rmse"]
    frac_tie = summary["tie_fraction_final_rmse"]
    if (
        show_godunov_in_output
        and (frac_gnn is not None)
        and (frac_god is not None)
        and (frac_tie is not None)
    ):
        print(
            "  final-step per-sample RMSE: "
            f"{gnn_label} better on {100.0 * float(frac_gnn):.2f}%, "
            f"{godunov_label} better on {100.0 * float(frac_god):.2f}%, "
            f"tie on {100.0 * float(frac_tie):.2f}%."
        )
    for baseline in extra_baseline_curves:
        fractions = baseline_win_fractions.get(baseline.scheme, {})
        frac_gnn = fractions.get("gnn_better")
        frac_base = fractions.get("baseline_better")
        frac_tie = fractions.get("tie")
        if (
            frac_gnn is not None
            and frac_base is not None
            and frac_tie is not None
        ):
            print(
                "  final-step per-sample RMSE: "
                f"{gnn_label} better than {baseline.label} on "
                f"{100.0 * float(frac_gnn):.2f}%, "
                f"{baseline.label} better on {100.0 * float(frac_base):.2f}%, "
                f"tie on {100.0 * float(frac_tie):.2f}%."
            )

    if len(extra_eval_curves) > 0:
        print(
            "  comparison model overlays: "
            + ", ".join(m.label for m in extra_eval_curves)
        )


if __name__ == "__main__":
    main()
