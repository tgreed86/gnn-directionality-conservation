#!/usr/bin/env python3
"""Compare GNN rollout performance against finite-volume baselines.

Run from an activated `amr-gnn-pt27` shell at repo root:

    python scripts/compare_gnn_vs_godunov.py \
        --checkpoint runs/with_physics/advection/my_run/checkpoint_best.pt \
        --device cpu

Optional overrides:
    --dataset <path>       # default: dataset path from checkpoint config
    --horizon <int>        # default: rollout_eval.horizon from config
    --num-samples <int>    # default: rollout_eval.num_test_samples from config
    --error-curves-log-y   # use log-scale y-axis for error_curves_compare.png
    --error-curves-mean-only  # hide per-sample curves in error_curves_compare.png
    --error-curves-smooth-window W  # NaN-aware moving-average window (1 disables)
    --error-curves-metrics RMSE SSIM rel_l2 L1 # choose panels in error_curves_compare.png
    --error-curves-spread std  # none | std | iqr shaded test-trajectory bands
    --ssim-mode legacy-global  # reproduce the former global SSIM (default: local-periodic)
    --ssim-data-range 1.0       # optional physical range override for local SSIM
    --curve-color exact=black gnn=blue godunov=gray  # override curve colors
    --baseline-schemes godunov muscl weno3  # FV baselines to plot
    --baseline-schemes muscl weno3  # omit Godunov from plots
    --baseline-schemes none  # omit all FV baselines from plots
    --gnn-label <name>     # legend name for current model (default: GNN)
    --godunov-label <name> # legend name for Godunov baseline (default: Godunov)
    --compare-label <name> # legend names for --compare-eval-npz (repeat in order)
    --compare-eval-npz P   # overlay another model run from gnn_vs_godunov_eval.npz
    --no-field-totals-exact  # omit Exact from field_totals_compare_all.pdf
    --plot-all-samples     # write one multi-page PDF with all sample spacetime plots
    --spacetime-delta-plots gnn godunov  # choose delta/error panels in that PDF
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from advection_data import exact_advect_spectral_shift, load_advection_dataset
from advection_schemes import (
    BASELINE_SCHEME_DEFAULT_LABELS,
    BASELINE_SCHEME_GODUNOV,
    BASELINE_SCHEME_MUSCL,
    BASELINE_SCHEME_WENO3,
    BASELINE_SCHEMES,
    normalize_baseline_scheme,
    rollout_finite_volume,
)
from gnn_advection import (
    build_model_from_config,
    predict_type_from_target_mode,
    resolve_device,
    resolve_state_residual_mode,
    save_json,
    state_residual_from_mode,
)
from input_features import (
    InputFeatureConfig,
    apply_state_minmax_minus1_1,
    append_position_channel,
    count_input_feature_channels,
    resolve_input_feature_cfg,
)
from physics_inputs import (
    ADVECTION_SCHEME_UPWIND,
    ADVECTION_SCHEMES,
    PhysicsInputConfig,
    build_physics_augmented_inputs_1d,
    compute_advdiff_terms_abs_1d,
    count_physics_input_channels,
    resolve_physics_input_cfg,
)

ERROR_CURVE_METRIC_RMSE = "rmse"
ERROR_CURVE_METRIC_REL_L2 = "rel_l2"
ERROR_CURVE_METRIC_L1 = "l1"
ERROR_CURVE_METRIC_SSIM = "ssim"
ERROR_CURVE_METRICS_DEFAULT = [
    ERROR_CURVE_METRIC_RMSE,
    ERROR_CURVE_METRIC_REL_L2,
    ERROR_CURVE_METRIC_L1,
    ERROR_CURVE_METRIC_SSIM,
]
ERROR_CURVE_METRIC_ALIASES = {
    "rmse": ERROR_CURVE_METRIC_RMSE,
    "root-mean-square-error": ERROR_CURVE_METRIC_RMSE,
    "root_mean_square_error": ERROR_CURVE_METRIC_RMSE,
    "relative-l2": ERROR_CURVE_METRIC_REL_L2,
    "relative_l2": ERROR_CURVE_METRIC_REL_L2,
    "rel-l2": ERROR_CURVE_METRIC_REL_L2,
    "rel_l2": ERROR_CURVE_METRIC_REL_L2,
    "rell2": ERROR_CURVE_METRIC_REL_L2,
    "l2rel": ERROR_CURVE_METRIC_REL_L2,
    "l1": ERROR_CURVE_METRIC_L1,
    "mae": ERROR_CURVE_METRIC_L1,
    "mean-absolute-error": ERROR_CURVE_METRIC_L1,
    "mean_absolute_error": ERROR_CURVE_METRIC_L1,
    "ssim": ERROR_CURVE_METRIC_SSIM,
}
ERROR_CURVE_METRIC_DISPLAY = {
    ERROR_CURVE_METRIC_RMSE: "RMSE",
    ERROR_CURVE_METRIC_REL_L2: "rel-L2",
    ERROR_CURVE_METRIC_L1: "L1",
    ERROR_CURVE_METRIC_SSIM: "SSIM",
}
ERROR_CURVE_SPREAD_NONE = "none"
ERROR_CURVE_SPREAD_STD = "std"
ERROR_CURVE_SPREAD_IQR = "iqr"
ERROR_CURVE_SPREAD_CHOICES = (
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
    """NaN-aware centered moving average for one curve."""
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
    """Apply NaN-aware temporal moving average to 1D/2D curve arrays."""
    a = np.asarray(arr, dtype=np.float64)
    w = max(int(window), 1)
    if (w % 2) == 0:
        # Keep centered smoothing without half-step phase shift.
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
    raise ValueError(
        f"_smooth_curves expects 1D/2D arrays, got shape {tuple(a.shape)}."
    )


def _error_curve_spread_bounds(
    curves: np.ndarray,
    spread: str,
    *,
    center: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-time spread bounds across equally weighted trajectories.

    ``std`` is the supplied/plotted center plus or minus one population
    standard deviation. ``iqr`` is the 25th--75th percentile interval. Each
    time column uses its finite trajectory values only.
    """
    values = np.asarray(curves, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError(
            "Spread curves must have shape [num_trajectories, num_times]."
        )
    if spread not in (ERROR_CURVE_SPREAD_STD, ERROR_CURVE_SPREAD_IQR):
        raise ValueError(
            f"Spread bounds require 'std' or 'iqr', got {spread!r}."
        )

    if center is None:
        center_values = np.full((values.shape[1],), np.nan, dtype=np.float64)
    else:
        center_values = np.asarray(center, dtype=np.float64).reshape(-1)
        if center_values.size != values.shape[1]:
            raise ValueError(
                "The spread center must have the same time dimension as curves."
            )

    lower = np.full((values.shape[1],), np.nan, dtype=np.float64)
    upper = np.full((values.shape[1],), np.nan, dtype=np.float64)
    for time_index in range(values.shape[1]):
        column = values[:, time_index]
        finite = column[np.isfinite(column)]
        if finite.size == 0:
            continue
        if spread == ERROR_CURVE_SPREAD_IQR:
            lower[time_index], upper[time_index] = np.quantile(
                finite,
                (0.25, 0.75),
            )
            continue

        mean = float(np.mean(finite))
        deviation = float(np.std(finite, ddof=0))
        band_center = float(center_values[time_index])
        if not np.isfinite(band_center):
            band_center = mean
        lower[time_index] = band_center - deviation
        upper[time_index] = band_center + deviation
    return lower, upper


def _clip_error_curve_spread_bounds(
    metric: str,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Clip shaded bands to the mathematical range of the selected metric."""
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    if metric == ERROR_CURVE_METRIC_SSIM:
        return np.maximum(lo, -1.0), np.minimum(hi, 1.0)
    return np.maximum(lo, 0.0), hi


def _rmse_1d(pred: np.ndarray, true: np.ndarray) -> float:
    """RMSE over finite paired entries of 1D arrays."""
    p = np.asarray(pred, dtype=np.float64).reshape(-1)
    t = np.asarray(true, dtype=np.float64).reshape(-1)
    mask = np.isfinite(p) & np.isfinite(t)
    if not np.any(mask):
        return float("nan")
    err = p[mask] - t[mask]
    return float(np.sqrt(np.mean(err * err)))


def _ssim_1d_global(
    pred: np.ndarray,
    true: np.ndarray,
    *,
    data_range: float | None = None,
    k1: float = 0.01,
    k2: float = 0.03,
    eps: float = 1e-12,
) -> float:
    """Compute a global SSIM score for 1D signals over finite paired entries."""
    x = np.asarray(true, dtype=np.float64).reshape(-1)
    y = np.asarray(pred, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(mask) < 2:
        return float("nan")
    x = x[mask]
    y = y[mask]

    mu_x = float(np.mean(x))
    mu_y = float(np.mean(y))
    var_x = float(np.mean((x - mu_x) ** 2))
    var_y = float(np.mean((y - mu_y) ** 2))
    cov_xy = float(np.mean((x - mu_x) * (y - mu_y)))

    if data_range is None or not np.isfinite(data_range):
        lo = min(float(np.min(x)), float(np.min(y)))
        hi = max(float(np.max(x)), float(np.max(y)))
        dyn = hi - lo
    else:
        dyn = float(data_range)
    dyn = max(abs(dyn), float(eps))

    c1 = (float(k1) * dyn) ** 2
    c2 = (float(k2) * dyn) ** 2
    denom = (mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2)
    if abs(denom) <= float(eps):
        return float("nan")
    num = (2.0 * mu_x * mu_y + c1) * (2.0 * cov_xy + c2)
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

    Local statistics use population-weighted variance and covariance. Unlike
    the legacy implementation, a nonfinite entry invalidates the whole spatial
    time slice rather than silently changing the evaluated spatial domain.
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
    offsets = np.arange(-(len(weights) // 2), len(weights) // 2 + 1)
    x_neighbors = np.stack([np.roll(x, -int(offset)) for offset in offsets], axis=0)
    y_neighbors = np.stack([np.roll(y, -int(offset)) for offset in offsets], axis=0)
    weight_col = weights[:, None]

    mu_x = np.sum(weight_col * x_neighbors, axis=0)
    mu_y = np.sum(weight_col * y_neighbors, axis=0)
    x_centered = x_neighbors - mu_x[None, :]
    y_centered = y_neighbors - mu_y[None, :]
    var_x = np.sum(weight_col * x_centered * x_centered, axis=0)
    var_y = np.sum(weight_col * y_centered * y_centered, axis=0)
    cov_xy = np.sum(weight_col * x_centered * y_centered, axis=0)

    c1 = (float(k1) * dyn) ** 2
    c2 = (float(k2) * dyn) ** 2
    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * cov_xy + c2)
    denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2)
    if not (
        np.all(np.isfinite(numerator))
        and np.all(np.isfinite(denominator))
        and np.all(denominator > 0.0)
    ):
        return float("nan")
    return float(np.mean(numerator / denominator))


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
    sample_ids: np.ndarray | None = None
    gnn_trajs_by_sample: np.ndarray | None = None
    ssim_mode: str = SSIM_MODE_LEGACY_GLOBAL
    ssim_data_range: float | None = None
    ssim_window_size: int | None = None
    ssim_window_sigma: float | None = None


@dataclass
class BaselineEvalCurves:
    """Curves for a finite-volume baseline computed for the primary run."""

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


def _derive_extra_label(npz_path: Path, used_labels: set[str]) -> str:
    base = npz_path.parent.name.strip() if npz_path.parent.name else npz_path.stem
    if base == "":
        base = npz_path.stem
    return _reserve_unique_label(base, used_labels)


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


def _resolve_spacetime_delta_plots(
    raw_plots: List[str] | None,
    *,
    plot_baseline_schemes: List[str],
) -> List[str] | None:
    """Return requested spacetime delta/error panels, or None for default behavior."""
    if raw_plots is None:
        return None

    out: List[str] = []
    disabled = False
    allowed = ["gnn"] + list(plot_baseline_schemes)

    def _append(key: str) -> None:
        if key not in out:
            out.append(key)

    for raw_item in raw_plots:
        for part in str(raw_item).split(","):
            token = part.strip()
            if token == "":
                continue
            token_lower = token.lower()
            if token_lower in {
                "none",
                "off",
                "no-deltas",
                "no_deltas",
                "no-delta-plots",
                "no_delta_plots",
            }:
                if len(out) > 0:
                    raise ValueError(
                        "--spacetime-delta-plots none cannot be combined with "
                        "other delta plot selections."
                    )
                disabled = True
                continue
            if disabled:
                raise ValueError(
                    "--spacetime-delta-plots none cannot be combined with "
                    "other delta plot selections."
                )
            if token_lower == "all":
                for key in allowed:
                    _append(key)
                continue
            if token_lower in {"gnn", "model", "current-model", "current_model"}:
                _append("gnn")
                continue

            scheme = normalize_baseline_scheme(token)
            if scheme not in plot_baseline_schemes:
                raise ValueError(
                    f"--spacetime-delta-plots requested '{scheme}', but that "
                    "baseline is not included in --baseline-schemes."
                )
            _append(scheme)

    if disabled:
        return []
    if len(out) == 0:
        raise ValueError("At least one spacetime delta plot must be requested.")
    return out


def _resolve_error_curve_metrics(raw_metrics: List[str] | None) -> List[str]:
    """Return requested error-curve metric keys in plotting order."""
    if raw_metrics is None:
        return list(ERROR_CURVE_METRICS_DEFAULT)

    out: List[str] = []

    def _append(metric: str) -> None:
        if metric not in out:
            out.append(metric)

    for raw_item in raw_metrics:
        for part in str(raw_item).split(","):
            token = part.strip()
            if token == "":
                continue
            token_lower = token.lower()
            if token_lower == "all":
                for metric in ERROR_CURVE_METRICS_DEFAULT:
                    _append(metric)
                continue
            metric = ERROR_CURVE_METRIC_ALIASES.get(token_lower)
            if metric is None:
                allowed = ", ".join(
                    ERROR_CURVE_METRIC_DISPLAY[m]
                    for m in ERROR_CURVE_METRICS_DEFAULT
                )
                raise ValueError(
                    f"Unknown --error-curves-metrics value '{token}'. "
                    f"Expected one of: {allowed}, or 'all'."
                )
            _append(metric)

    if len(out) == 0:
        raise ValueError("At least one error-curve metric must be requested.")
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
    if key_lower in {"exact", "truth", "ground-truth", "ground_truth"}:
        return "exact"
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
        "exact": "exact",
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
                "exact",
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

        sample_ids = None
        gnn_trajs_by_sample = None
        if "all_sample_ids" in data and "all_gnn_trajs" in data:
            sample_ids = np.asarray(data["all_sample_ids"], dtype=np.int64).reshape(-1)
            gnn_trajs_by_sample = np.asarray(
                data["all_gnn_trajs"],
                dtype=np.float64,
            )
            if gnn_trajs_by_sample.ndim != 3:
                raise ValueError(
                    f"{npz_path}: all_gnn_trajs must have shape [S,T,N], "
                    f"got {tuple(gnn_trajs_by_sample.shape)}."
                )
            if int(gnn_trajs_by_sample.shape[0]) != int(sample_ids.shape[0]):
                raise ValueError(
                    f"{npz_path}: all_gnn_trajs first dimension "
                    f"({gnn_trajs_by_sample.shape[0]}) must match "
                    f"all_sample_ids length ({sample_ids.shape[0]})."
                )
        elif "sample_traj_id" in data and "sample_gnn_traj" in data:
            one_traj = np.asarray(data["sample_gnn_traj"], dtype=np.float64)
            if one_traj.size > 0:
                if one_traj.ndim != 2:
                    raise ValueError(
                        f"{npz_path}: sample_gnn_traj must have shape [T,N], "
                        f"got {tuple(one_traj.shape)}."
                    )
                sample_id_val = int(np.asarray(data["sample_traj_id"]).reshape(()))
                sample_ids = np.array([sample_id_val], dtype=np.int64)
                gnn_trajs_by_sample = one_traj[None, :, :]

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
        sample_ids=sample_ids,
        gnn_trajs_by_sample=gnn_trajs_by_sample,
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
        return (
            f"window size {extra.ssim_window_size} versus {primary.window_size}"
        )
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


@dataclass(frozen=True)
class GodunovCorrectionConfig:
    enabled: bool = False
    advection_scheme: str = ADVECTION_SCHEME_UPWIND
    c_override: float | None = None


def _resolve_godunov_correction_cfg(cfg: Dict) -> GodunovCorrectionConfig:
    train_cfg_raw = cfg.get("train", {})
    model_cfg_raw = cfg.get("model", {})
    train_cfg = train_cfg_raw if isinstance(train_cfg_raw, dict) else {}
    model_cfg = model_cfg_raw if isinstance(model_cfg_raw, dict) else {}
    raw = train_cfg.get("godunov_correction", model_cfg.get("godunov_correction", {}))

    if isinstance(raw, bool):
        enabled = bool(raw)
        scheme = ADVECTION_SCHEME_UPWIND
        c_override = None
    elif isinstance(raw, dict):
        enabled = bool(raw.get("enabled", False))
        scheme = str(
            raw.get("advection_scheme", raw.get("scheme", ADVECTION_SCHEME_UPWIND))
        ).strip().lower()
        c_raw = raw.get("c_override", raw.get("c", None))
        c_override = None if c_raw is None else float(c_raw)
    else:
        raise ValueError(
            "godunov_correction must be either a boolean or object, got "
            f"{type(raw).__name__}."
        )
    if scheme not in ADVECTION_SCHEMES:
        raise ValueError(
            "godunov_correction.advection_scheme must be one of "
            f"{sorted(ADVECTION_SCHEMES)}, got '{scheme}'."
        )
    return GodunovCorrectionConfig(
        enabled=enabled,
        advection_scheme=scheme,
        c_override=c_override,
    )


def _target_from_states(
    *,
    x_now: torch.Tensor,
    state_next: torch.Tensor,
    dt: float,
    target_mode: str,
) -> torch.Tensor:
    if target_mode == "state":
        return state_next
    if target_mode == "delta":
        return state_next - x_now
    if target_mode == "rate":
        return (state_next - x_now) / float(dt)
    raise RuntimeError(f"Unexpected target_mode='{target_mode}'.")


def _state_from_target(
    *,
    x_now: torch.Tensor,
    target: torch.Tensor,
    dt: float,
    target_mode: str,
) -> torch.Tensor:
    if target_mode == "state":
        return target
    if target_mode == "delta":
        return x_now + target
    if target_mode == "rate":
        return x_now + float(dt) * target
    raise RuntimeError(f"Unexpected target_mode='{target_mode}'.")


def _godunov_next_state(
    *,
    x_now: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    dt: float,
    dx: float,
    c_from_data: float,
    godunov_cfg: GodunovCorrectionConfig,
) -> torch.Tensor:
    c_eff = (
        float(godunov_cfg.c_override)
        if godunov_cfg.c_override is not None
        else float(c_from_data)
    )
    r_adv_abs, _ = compute_advdiff_terms_abs_1d(
        x_abs=x_now,
        edge_index=edge_index,
        edge_attr=edge_attr,
        c=c_eff,
        dx=float(dx),
        nu=0.0,
        advection_scheme=godunov_cfg.advection_scheme,
        compute_adv=True,
        compute_diff=False,
    )
    assert r_adv_abs is not None
    return x_now + float(dt) * r_adv_abs


def _rollout_baseline(
    *,
    x0: np.ndarray,
    c: float,
    dt: float,
    dx: float,
    horizon: int,
    scheme: str,
) -> np.ndarray:
    return rollout_finite_volume(
        x0=x0,
        c=c,
        dt=dt,
        dx=dx,
        horizon=horizon,
        scheme=scheme,
    )


def _rollout_gnn(
    *,
    model: torch.nn.Module,
    x0: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    pos_nodes: torch.Tensor,
    c: float,
    dt: float,
    dx: float,
    horizon: int,
    device: torch.device,
    physics_cfg: PhysicsInputConfig,
    input_cfg: InputFeatureConfig,
    target_mode: str,
    state_residual_mode: str,
    godunov_cfg: GodunovCorrectionConfig,
) -> np.ndarray:
    model.eval()
    x_pred = x0.clone().detach()
    traj = np.empty((horizon + 1, int(x0.shape[0])), dtype=np.float64)
    traj[0] = x_pred.squeeze(-1).cpu().numpy().astype(np.float64)

    with torch.no_grad():
        ei = edge_index.to(device=device)
        ea = edge_attr.to(device=device, dtype=torch.float32)
        for k in range(horizon):
            x_in = x_pred.to(device=device, dtype=torch.float32)
            x_model = build_physics_augmented_inputs_1d(
                x_state=x_in,
                edge_index=ei,
                edge_attr=ea,
                dt=dt,
                dx=dx,
                c_from_data=c,
                physics_cfg=physics_cfg,
                step_k=k,
            )
            x_model = append_position_channel(
                x_state=x_model,
                pos_nodes=pos_nodes,
                input_cfg=input_cfg,
                batch_size=None,
            )
            x_model = apply_state_minmax_minus1_1(
                x_model=x_model,
                input_cfg=input_cfg,
            )
            if godunov_cfg.enabled:
                if target_mode == "state":
                    raise ValueError(
                        "godunov_correction currently requires target_mode in "
                        "{delta, rate}; state-mode correction is not supported."
                    )
                god_state = _godunov_next_state(
                    x_now=x_in,
                    edge_index=ei,
                    edge_attr=ea,
                    dt=dt,
                    dx=dx,
                    c_from_data=c,
                    godunov_cfg=godunov_cfg,
                )
                god_target = _target_from_states(
                    x_now=x_in,
                    state_next=god_state,
                    dt=dt,
                    target_mode=target_mode,
                )
                pred_corr = model(x_model, ei, edge_attr=ea, dt=dt)
                pred_target = god_target + pred_corr
                y = _state_from_target(
                    x_now=x_in,
                    target=pred_target,
                    dt=dt,
                    target_mode=target_mode,
                )
            else:
                y = model.predict_state(
                    x_model,
                    ei,
                    edge_attr=ea,
                    dt=dt,
                    state_residual=state_residual_from_mode(
                        x_raw=x_in,
                        state_residual_mode=state_residual_mode,
                    ),
                )
            x_pred = y.detach().cpu()
            traj[k + 1] = x_pred.squeeze(-1).cpu().numpy().astype(np.float64)
    return traj


def _compute_curves(
    *,
    pred_traj: np.ndarray,
    true_traj: np.ndarray,
    ssim_config: SSIMConfig | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-step metrics for t=1..H: RMSE, rel-L2, L1, SSIM."""
    resolved_ssim = SSIMConfig() if ssim_config is None else ssim_config
    err = pred_traj[1:] - true_traj[1:]
    rmse = np.sqrt(np.mean(err * err, axis=1))
    l1 = np.mean(np.abs(err), axis=1)
    denom = np.linalg.norm(true_traj[1:], axis=1)
    rel = np.linalg.norm(err, axis=1) / np.maximum(denom, 1e-12)
    ssim = np.empty((pred_traj.shape[0] - 1,), dtype=np.float64)
    for k in range(pred_traj.shape[0] - 1):
        ssim[k] = _ssim_1d(
            pred_traj[k + 1],
            true_traj[k + 1],
            config=resolved_ssim,
        )
    return (
        rmse.astype(np.float64),
        rel.astype(np.float64),
        l1.astype(np.float64),
        ssim.astype(np.float64),
    )


def _resolve_eval_mode_and_ids(
    *,
    dset,
    split: Dict,
    num_samples: int,
) -> Tuple[str, List[int]]:
    """Choose trajectory IDs (preferred) or pair IDs for evaluation."""
    if dset.rollout_states is not None:
        test_traj_ids = split.get("test_trajectory_ids", None)
        if isinstance(test_traj_ids, list) and len(test_traj_ids) > 0:
            ids = [int(v) for v in test_traj_ids if v is not None]
            if len(ids) > 0:
                return "trajectory", ids[: min(num_samples, len(ids))]

        if dset.pair_trajectory_id is not None:
            test_pairs = split.get("test", list(range(len(dset))))
            if len(test_pairs) > 0:
                test_pairs_t = torch.as_tensor(test_pairs, dtype=torch.long)
                uniq = torch.unique(dset.pair_trajectory_id[test_pairs_t]).cpu().tolist()
                ids = sorted(int(v) for v in uniq)
                return "trajectory", ids[: min(num_samples, len(ids))]

        # Fallback: use first trajectories in file.
        n_traj = int(dset.rollout_states.shape[0])
        ids = list(range(n_traj))
        return "trajectory", ids[: min(num_samples, len(ids))]

    # No full trajectories available; evaluate by pair IDs.
    test_pairs = split.get("test", list(range(len(dset))))
    ids = [int(v) for v in test_pairs]
    return "pair", ids[: min(num_samples, len(ids))]


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare GNN rollout against finite-volume baselines."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to checkpoint_best.pt or checkpoint_last.pt from training.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Optional dataset override. Defaults to checkpoint config path.",
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
        help="Number of test trajectories/pairs to evaluate.",
    )
    parser.add_argument(
        "--ssim-mode",
        type=str,
        choices=SSIM_MODES,
        default=SSIM_MODE_LOCAL_PERIODIC,
        help=(
            "SSIM definition: 'legacy-global' exactly preserves the existing "
            "single-window, prediction-dependent calculation; 'local-periodic' "
            "uses Gaussian local statistics, periodic boundaries, and one fixed "
            "reference-only range. Default: local-periodic."
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
        "--device",
        type=str,
        default=None,
        choices=("auto", "cpu", "cuda", "mps"),
        help="Optional inference device override.",
    )
    parser.add_argument(
        "--state-residual-mode",
        type=str,
        default=None,
        help=(
            "Override model.state_residual_mode from the checkpoint config. "
            "Use 'raw' for current behavior or 'legacy'/'model_input' for the "
            "older normalized-input residual behavior."
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
        help=(
            "Directory for comparison plots. "
            "Default: <checkpoint_dir>/gnn_vs_godunov_plots"
        ),
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
            "figures for all evaluated samples."
        ),
    )
    parser.add_argument(
        "--no-field-totals-exact",
        action="store_true",
        help="Omit the Exact reference curve from field_totals_compare_all.pdf.",
    )
    parser.add_argument(
        "--spacetime-delta-plots",
        type=str,
        nargs="+",
        default=None,
        help=(
            "For sample_spacetime_compare_all.pdf, choose which delta/error "
            "panels to include. Use space-separated or comma-separated values "
            "from {gnn, godunov, muscl, weno3}, 'all', or 'none'. Default: "
            "current behavior, meaning every shown prediction gets a delta panel."
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
        default=None,
        help=(
            "Accuracy metrics to include in error_curves_compare.png. Use "
            "space-separated or comma-separated values from {RMSE, rel-L2, "
            "L1, SSIM}, or 'all'. Default: all four metrics."
        ),
    )
    parser.add_argument(
        "--error-curves-spread",
        type=str,
        choices=ERROR_CURVE_SPREAD_CHOICES,
        default=ERROR_CURVE_SPREAD_NONE,
        help=(
            "Shaded dispersion across test-trajectory metric curves: 'none' "
            "(default), 'std' for the mean plus/minus one population standard "
            "deviation, or 'iqr' for the 25th--75th percentile band."
        ),
    )
    parser.add_argument(
        "--curve-color",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Override colors for curves in comparison plots with KEY=COLOR "
            "pairs. KEY can be exact, gnn, godunov, muscl, weno3, compare1, "
            "compare2, or a displayed legend label. Repeat or pass multiple "
            "pairs, e.g. --curve-color gnn=black godunov=gray "
            "compare1='#9467bd'. Quote hex colors in the shell."
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
        help="Legend label for the current model curves.",
    )
    parser.add_argument(
        "--godunov-label",
        type=str,
        default="Godunov",
        help="Legend label for the Godunov baseline curves.",
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
    pos: np.ndarray,
    plot_all_samples: bool,
    spacetime_delta_plots: List[str] | None,
    error_curves_log_y: bool,
    error_curves_mean_only: bool,
    error_curves_smooth_window: int,
    error_curves_spread: str,
    error_curve_metrics: List[str],
    ssim_config: SSIMConfig,
    curve_colors: Dict[str, str] | None,
    field_totals_include_exact: bool,
    extra_baseline_curves: List[BaselineEvalCurves] | None,
    extra_eval_curves: List[ExtraEvalCurves] | None,
    plot_baseline_schemes: List[str],
    all_sample_ids: List[int] | None,
    all_true_trajs: List[np.ndarray] | None,
    all_gnn_trajs: List[np.ndarray] | None,
    all_god_trajs: List[np.ndarray] | None,
    all_extra_baseline_trajs: Dict[str, List[np.ndarray]] | None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir.mkdir(parents=True, exist_ok=True)

    steps = np.arange(1, horizon + 1, dtype=np.int64)
    times = steps.astype(np.float64) * float(dt)
    smooth_w = max(int(error_curves_smooth_window), 1)

    rmse_gnn_plot = _smooth_curves(rmse_gnn, window=smooth_w)
    rel_gnn_plot = _smooth_curves(rel_gnn, window=smooth_w)
    l1_gnn_plot = _smooth_curves(l1_gnn, window=smooth_w)
    ssim_gnn_plot = _smooth_curves(ssim_gnn, window=smooth_w)
    rmse_god_plot = _smooth_curves(rmse_god, window=smooth_w)
    rel_god_plot = _smooth_curves(rel_god, window=smooth_w)
    l1_god_plot = _smooth_curves(l1_god, window=smooth_w)
    ssim_god_plot = _smooth_curves(ssim_god, window=smooth_w)
    mean_rmse_gnn_plot = _smooth_curves(mean_rmse_gnn, window=smooth_w)
    mean_rel_gnn_plot = _smooth_curves(mean_rel_gnn, window=smooth_w)
    mean_l1_gnn_plot = _smooth_curves(mean_l1_gnn, window=smooth_w)
    mean_ssim_gnn_plot = _smooth_curves(mean_ssim_gnn, window=smooth_w)
    mean_rmse_god_plot = _smooth_curves(mean_rmse_god, window=smooth_w)
    mean_rel_god_plot = _smooth_curves(mean_rel_god, window=smooth_w)
    mean_l1_god_plot = _smooth_curves(mean_l1_god, window=smooth_w)
    mean_ssim_god_plot = _smooth_curves(mean_ssim_god, window=smooth_w)
    plot_scheme_set = set(plot_baseline_schemes)
    show_godunov_baseline = BASELINE_SCHEME_GODUNOV in plot_scheme_set
    extra_baselines = [
        baseline
        for baseline in ([] if extra_baseline_curves is None else list(extra_baseline_curves))
        if baseline.scheme in plot_scheme_set
    ]
    color_overrides = {} if curve_colors is None else dict(curve_colors)
    exact_color = color_overrides.get("exact", "black")
    gnn_color = color_overrides.get("gnn", "tab:blue")
    godunov_color = color_overrides.get(BASELINE_SCHEME_GODUNOV, "tab:green")

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

    spread_mode = str(error_curves_spread)

    def _plot_spread_band(
        ax,
        *,
        x_values: np.ndarray,
        sample_curves: np.ndarray,
        mean_curve: np.ndarray,
        metric: str,
        color: str,
    ) -> None:
        if spread_mode == ERROR_CURVE_SPREAD_NONE:
            return
        sample_values = np.asarray(sample_curves, dtype=np.float64)
        if sample_values.ndim == 2 and sample_values.shape[0] == 0:
            return
        lower, upper = _error_curve_spread_bounds(
            sample_values,
            spread_mode,
            center=mean_curve,
        )
        lower, upper = _clip_error_curve_spread_bounds(metric, lower, upper)
        ax.fill_between(
            x_values,
            lower,
            upper,
            color=color,
            alpha=0.14,
            linewidth=0.0,
        )

    # 1) Mean error curves.
    metric_plot_data = {
        ERROR_CURVE_METRIC_RMSE: {
            "metric": ERROR_CURVE_METRIC_RMSE,
            "title": "RMSE vs Time",
            "ylabel": "RMSE",
            "log_y": True,
            "gnn_by_sample": rmse_gnn_plot,
            "god_by_sample": rmse_god_plot,
            "gnn_mean": mean_rmse_gnn_plot,
            "god_mean": mean_rmse_god_plot,
            "sample_attr": "rmse_by_sample",
            "mean_attr": "mean_rmse",
        },
        ERROR_CURVE_METRIC_REL_L2: {
            "metric": ERROR_CURVE_METRIC_REL_L2,
            "title": "Relative L2 vs Time",
            "ylabel": "rel-L2",
            "log_y": True,
            "gnn_by_sample": rel_gnn_plot,
            "god_by_sample": rel_god_plot,
            "gnn_mean": mean_rel_gnn_plot,
            "god_mean": mean_rel_god_plot,
            "sample_attr": "rel_by_sample",
            "mean_attr": "mean_rel",
        },
        ERROR_CURVE_METRIC_L1: {
            "metric": ERROR_CURVE_METRIC_L1,
            "title": "L1 vs Time",
            "ylabel": "L1",
            "log_y": True,
            "gnn_by_sample": l1_gnn_plot,
            "god_by_sample": l1_god_plot,
            "gnn_mean": mean_l1_gnn_plot,
            "god_mean": mean_l1_god_plot,
            "sample_attr": "l1_by_sample",
            "mean_attr": "mean_l1",
        },
        ERROR_CURVE_METRIC_SSIM: {
            "metric": ERROR_CURVE_METRIC_SSIM,
            "title": f"{ssim_config.display_name} vs Time",
            "ylabel": ssim_config.display_name,
            "log_y": False,
            "gnn_by_sample": ssim_gnn_plot,
            "god_by_sample": ssim_god_plot,
            "gnn_mean": mean_ssim_gnn_plot,
            "god_mean": mean_ssim_god_plot,
            "sample_attr": "ssim_by_sample",
            "mean_attr": "mean_ssim",
        },
    }
    selected_metric_specs = [metric_plot_data[m] for m in error_curve_metrics]
    num_error_metrics = len(selected_metric_specs)
    if num_error_metrics == 1:
        error_fig_shape = (1, 1)
        error_fig_size = (7.7, 4.3)
    elif num_error_metrics == 2:
        error_fig_shape = (1, 2)
        error_fig_size = (15.5, 4.3)
    else:
        error_fig_shape = (int(math.ceil(num_error_metrics / 2.0)), 2)
        error_fig_size = (
            15.5,
            8.3 if num_error_metrics <= 4 else 4.15 * error_fig_shape[0],
        )

    fig, axes = plt.subplots(
        error_fig_shape[0],
        error_fig_shape[1],
        figsize=error_fig_size,
        constrained_layout=True,
        squeeze=False,
    )
    axes_flat = axes.ravel()
    plot_axes = list(axes_flat[:num_error_metrics])
    for ax in axes_flat[num_error_metrics:]:
        ax.set_visible(False)

    extra_models = [] if extra_eval_curves is None else list(extra_eval_curves)
    for ax, spec in zip(plot_axes, selected_metric_specs):
        metric = str(spec["metric"])
        gnn_by_sample = np.asarray(spec["gnn_by_sample"], dtype=np.float64)
        god_by_sample = np.asarray(spec["god_by_sample"], dtype=np.float64)
        if (gnn_by_sample.shape[0] > 0) and (not error_curves_mean_only):
            for i in range(gnn_by_sample.shape[0]):
                ax.plot(
                    times,
                    gnn_by_sample[i],
                    color=gnn_color,
                    alpha=0.3,
                    lw=0.9,
                )
                if show_godunov_baseline:
                    ax.plot(
                        times,
                        god_by_sample[i],
                        color=godunov_color,
                        alpha=0.3,
                        lw=0.9,
                        ls="--",
                    )
            for j, baseline in enumerate(extra_baselines):
                c_base = _baseline_color(j, baseline.scheme)
                base_by_sample = _smooth_curves(
                    getattr(baseline, str(spec["sample_attr"])),
                    window=smooth_w,
                )
                for i in range(base_by_sample.shape[0]):
                    ax.plot(
                        times,
                        base_by_sample[i],
                        color=c_base,
                        alpha=0.22,
                        lw=0.85,
                        ls="--",
                    )

        _plot_spread_band(
            ax,
            x_values=times,
            sample_curves=gnn_by_sample,
            mean_curve=np.asarray(spec["gnn_mean"], dtype=np.float64),
            metric=metric,
            color=gnn_color,
        )
        if show_godunov_baseline:
            _plot_spread_band(
                ax,
                x_values=times,
                sample_curves=god_by_sample,
                mean_curve=np.asarray(spec["god_mean"], dtype=np.float64),
                metric=metric,
                color=godunov_color,
            )
        for j, baseline in enumerate(extra_baselines):
            c_base = _baseline_color(j, baseline.scheme)
            _plot_spread_band(
                ax,
                x_values=times,
                sample_curves=_smooth_curves(
                    getattr(baseline, str(spec["sample_attr"])),
                    window=smooth_w,
                ),
                mean_curve=_smooth_curves(
                    getattr(baseline, str(spec["mean_attr"])),
                    window=smooth_w,
                ),
                metric=metric,
                color=c_base,
            )
        for j, extra in enumerate(extra_models):
            c = _extra_model_color(j)
            t_extra = np.arange(1, extra.horizon + 1, dtype=np.float64) * float(dt)
            _plot_spread_band(
                ax,
                x_values=t_extra,
                sample_curves=_smooth_curves(
                    getattr(extra, str(spec["sample_attr"])),
                    window=smooth_w,
                ),
                mean_curve=_smooth_curves(
                    getattr(extra, str(spec["mean_attr"])),
                    window=smooth_w,
                ),
                metric=metric,
                color=c,
            )

        ax.plot(
            times,
            np.asarray(spec["gnn_mean"], dtype=np.float64),
            color=gnn_color,
            lw=2.3,
            label=gnn_label,
        )
        if show_godunov_baseline:
            ax.plot(
                times,
                np.asarray(spec["god_mean"], dtype=np.float64),
                color=godunov_color,
                lw=2.3,
                ls="--",
                label=godunov_label,
            )
        for j, baseline in enumerate(extra_baselines):
            c_base = _baseline_color(j, baseline.scheme)
            ax.plot(
                times,
                _smooth_curves(
                    getattr(baseline, str(spec["mean_attr"])),
                    window=smooth_w,
                ),
                color=c_base,
                lw=2.1,
                ls="--",
                label=baseline.label,
            )

        # Keep primary GNN/Godunov colors stable when overlaying comparison runs.
        for j, extra in enumerate(extra_models):
            c = _extra_model_color(j)
            t_extra = np.arange(1, extra.horizon + 1, dtype=np.float64) * float(dt)
            ax.plot(
                t_extra,
                _smooth_curves(
                    getattr(extra, str(spec["mean_attr"])),
                    window=smooth_w,
                ),
                color=c,
                lw=2.0,
                label=extra.label,
            )

       # ax.set_title(str(spec["title"]), fontsize=15)
        ax.set_xlabel("Time", fontsize=18)
        ax.set_ylabel(str(spec["ylabel"]), fontsize=18)
        ax.tick_params(axis="x", labelsize=14)
        ax.tick_params(axis="y", labelsize=14)
        ax.grid(alpha=0.3)

        if str(spec["sample_attr"]) == "ssim_by_sample":
            ax.legend(loc="lower left", fontsize=16)
        #if (str(spec["ylabel"]) == "L1"):
        #    ax.legend(loc="upper left", fontsize=16)
        #else:
        #    ax.legend(loc="upper left")
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

        for ax, spec in zip(plot_axes, selected_metric_specs):
            if not bool(spec["log_y"]):
                continue
            vals = [
                np.asarray(spec["gnn_by_sample"], dtype=np.float64).reshape(-1),
                np.asarray(spec["gnn_mean"], dtype=np.float64).reshape(-1),
            ]
            if show_godunov_baseline:
                vals.extend(
                    [
                        np.asarray(
                            spec["god_by_sample"],
                            dtype=np.float64,
                        ).reshape(-1),
                        np.asarray(spec["god_mean"], dtype=np.float64).reshape(-1),
                    ]
                )
            for baseline in extra_baselines:
                vals.extend(
                    [
                        _smooth_curves(
                            getattr(baseline, str(spec["sample_attr"])),
                            window=smooth_w,
                        ).reshape(-1),
                        _smooth_curves(
                            getattr(baseline, str(spec["mean_attr"])),
                            window=smooth_w,
                        ).reshape(-1),
                    ]
                )
            for extra in extra_models:
                vals.extend(
                    [
                        _smooth_curves(
                            getattr(extra, str(spec["sample_attr"])),
                            window=smooth_w,
                        ).reshape(-1),
                        _smooth_curves(
                            getattr(extra, str(spec["mean_attr"])),
                            window=smooth_w,
                        ).reshape(-1),
                    ]
                )
            _set_log_y_if_possible(ax, np.concatenate(vals))
    fig.suptitle("1D Linear Advection: Rollout Accuracy", fontsize=20)
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
            ax20.hist(
                frg,
                bins=min(20, max(5, frg.size)),
                alpha=0.6,
                color=gnn_color,
                label=gnn_label,
            )
        if show_godunov_baseline and fru.size > 0:
            ax20.hist(
                fru,
                bins=min(20, max(5, fru.size)),
                alpha=0.6,
                color=godunov_color,
                label=godunov_label,
            )
        if flg.size > 0:
            ax21.hist(
                flg,
                bins=min(20, max(5, flg.size)),
                alpha=0.6,
                color=gnn_color,
                label=gnn_label,
            )
        if show_godunov_baseline and flu.size > 0:
            ax21.hist(
                flu,
                bins=min(20, max(5, flu.size)),
                alpha=0.6,
                color=godunov_color,
                label=godunov_label,
            )
        if f1g.size > 0:
            ax22.hist(
                f1g,
                bins=min(20, max(5, f1g.size)),
                alpha=0.6,
                color=gnn_color,
                label=gnn_label,
            )
        if show_godunov_baseline and f1u.size > 0:
            ax22.hist(
                f1u,
                bins=min(20, max(5, f1u.size)),
                alpha=0.6,
                color=godunov_color,
                label=godunov_label,
            )

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

    def _build_spacetime_compare_figure(
        *,
        sid: int,
        true_arr: np.ndarray,
        gnn_arr: np.ndarray,
        baseline_rows: List[Tuple[str, str, np.ndarray]],
        title_prefix: str,
    ):
        true_plot = _replace_nonfinite_with_nan(true_arr)
        gnn_plot = _replace_nonfinite_with_nan(gnn_arr)
        err_gnn = np.abs(gnn_plot - true_plot)
        baseline_plots = [
            (scheme, label, _replace_nonfinite_with_nan(arr))
            for scheme, label, arr in baseline_rows
        ]
        baseline_errs = [
            (scheme, label, np.abs(arr - true_plot))
            for scheme, label, arr in baseline_plots
        ]
        delta_plot_set = (
            None
            if spacetime_delta_plots is None
            else set(str(v) for v in spacetime_delta_plots)
        )

        def _show_delta(key: str) -> bool:
            return delta_plot_set is None or key in delta_plot_set

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
            *[_safe_nanmax(err) for _, _, err in baseline_errs],
            1e-8,
        )

        def _baseline_final_label(baseline_plot: np.ndarray) -> str:
            baseline_final = baseline_plot[-1]
            baseline_rmse_final = _rmse_1d(baseline_final, true_final)
            baseline_ssim_final = _ssim_1d(
                baseline_final,
                true_final,
                config=ssim_config,
            )
            #return (
            #    "Spatial coordinate x_i\n"
            #    f"Final RMSE={_fmt_metric(baseline_rmse_final, '.3e')}, "
            #    f"Final SSIM={_fmt_metric(baseline_ssim_final, '.4f')}"
            #)
            return ("Position")

        def _add_spacetime_colorbars(
            *,
            fig3,
            state_im,
            err_im,
            state_axes,
            err_axes,
        ) -> None:
            colorbar_fraction = 0.025
            colorbar_pad = 0.02

            def _axes_union_width(ax_list) -> float:
                boxes = [ax.get_position() for ax in ax_list if ax.get_visible()]
                if len(boxes) == 0:
                    return 0.0
                return float(max(box.x1 for box in boxes) - min(box.x0 for box in boxes))

            err_colorbar_fraction = colorbar_fraction
            state_width = _axes_union_width(state_axes)
            err_width = _axes_union_width(err_axes)
            if state_width > 0.0 and err_width > 0.0:
                err_colorbar_fraction = colorbar_fraction * (state_width / err_width)

            if state_im is not None:
                fig3.colorbar(
                    state_im,
                    ax=state_axes,
                    fraction=colorbar_fraction,
                    pad=colorbar_pad,
                )
            if err_im is not None:
                fig3.colorbar(
                    err_im,
                    ax=err_axes,
                    fraction=err_colorbar_fraction,
                    pad=colorbar_pad,
                )

        if delta_plot_set is not None:
            panels = [
                ("state", "Exact", true_plot, "Position"),
                ("state", f"{gnn_label} Predicted", gnn_plot, "Position"),
            ]
            for _, baseline_label, baseline_plot in baseline_plots:
                panels.append(
                    (
                        "state",
                        f"{baseline_label} Predicted",
                        baseline_plot,
                        _baseline_final_label(baseline_plot),
                    )
                )
            if _show_delta("gnn"):
                panels.append(("err", f"|{gnn_label} - Exact|", err_gnn, "Position"))
            for baseline_scheme, baseline_label, err_baseline in baseline_errs:
                if _show_delta(baseline_scheme):
                    panels.append(
                        (
                            "err",
                            f"|{baseline_label} - Exact|",
                            err_baseline,
                            "Position",
                        )
                    )

            n_cols = min(5, max(1, len(panels)))
            n_rows = int(math.ceil(len(panels) / float(n_cols)))
            fig3, axes3 = plt.subplots(
                n_rows,
                n_cols,
                figsize=(4.6 * n_cols, max(4.2, 4.1 * n_rows)),
                constrained_layout=True,
                squeeze=False,
            )
            state_axes = []
            err_axes = []
            state_im = None
            err_im = None
            for panel_idx, (kind, panel_title, panel_arr, xlabel) in enumerate(panels):
                ax = axes3[panel_idx // n_cols, panel_idx % n_cols]
                if kind == "state":
                    im = ax.imshow(
                        panel_arr,
                        aspect="auto",
                        extent=extent,
                        cmap="viridis",
                        vmin=gt_min,
                        vmax=gt_max,
                    )
                    if state_im is None:
                        state_im = im
                    state_axes.append(ax)
                else:
                    im = ax.imshow(
                        panel_arr,
                        aspect="auto",
                        extent=extent,
                        cmap="magma",
                        vmin=0.0,
                        vmax=err_max,
                    )
                    if err_im is None:
                        err_im = im
                    err_axes.append(ax)
                ax.set_title(panel_title, fontsize=20)
                ax.set_xlabel(xlabel, fontsize=20)
                ax.set_ylabel("Time", fontsize=20)

            for panel_idx in range(len(panels), n_rows * n_cols):
                axes3[panel_idx // n_cols, panel_idx % n_cols].set_visible(False)

            _add_spacetime_colorbars(
                fig3=fig3,
                state_im=state_im,
                err_im=err_im,
                state_axes=state_axes,
                err_axes=err_axes,
            )
            fig3.suptitle(
                f"{title_prefix} | id={sid}",
                fontsize=11,
            )
            return fig3

        n_extra = max(0, len(baseline_rows) - 1)
        n_rows = 1 + int(math.ceil(n_extra / 2.0))
        n_cols = 3 if len(baseline_rows) == 0 else 5
        fig3, axes3 = plt.subplots(
            n_rows,
            n_cols,
            figsize=(14.0 if n_cols == 3 else 23.0, max(4.2, 4.1 * n_rows)),
            constrained_layout=True,
            squeeze=False,
        )
        state_axes = []
        err_axes = []
        state_im = None
        err_im = None

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
            _, primary_label, primary_plot = baseline_plots[0]
            _, _, primary_err = baseline_errs[0]
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
        axes_row[1].set_xlabel(
            "Position",
            #"Spatial coordinate x_i\n"
            #f"Final RMSE={_fmt_metric(gnn_rmse_final, '.3e')}, "
            #f"Final SSIM={_fmt_metric(gnn_ssim_final, '.4f')}",
            fontsize=20,
        )
        state_axes.extend([axes_row[0], axes_row[1]])
        err_axes.append(axes_row[gnn_err_col])
        if len(baseline_plots) > 0:
            axes_row[2].set_xlabel(_baseline_final_label(baseline_plots[0][2]), fontsize=12)
            state_axes.append(axes_row[2])
            err_axes.append(axes_row[4])

        for row_idx in range(1, n_rows):
            for ax in axes3[row_idx]:
                ax.set_visible(False)

        for extra_idx, ((_, baseline_label, baseline_plot), (_, _, err_baseline)) in enumerate(
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
            err_ax.set_xlabel("Spatial coordinate x_i")
            pred_ax.set_ylabel("time")
            err_ax.set_ylabel("time")
            state_axes.append(pred_ax)
            err_axes.append(err_ax)

        _add_spacetime_colorbars(
            fig3=fig3,
            state_im=state_im,
            err_im=err_im,
            state_axes=state_axes,
            err_axes=err_axes,
        )
        fig3.suptitle(
            f"{title_prefix} | id={sid}",
            fontsize=11,
        )
        return fig3

    def _field_total_curve(arr: np.ndarray) -> np.ndarray:
        vals = np.asarray(arr, dtype=np.float64)
        if vals.ndim != 2:
            raise ValueError(
                f"Expected a 2D trajectory array for field totals, got {vals.shape}."
            )
        finite = np.isfinite(vals)
        totals = np.full((vals.shape[0],), np.nan, dtype=np.float64)
        valid_rows = np.any(finite, axis=1)
        totals[valid_rows] = np.nansum(
            np.where(finite[valid_rows], vals[valid_rows], np.nan),
            axis=1,
        )
        return totals

    baseline_index_by_scheme = {
        baseline.scheme: idx for idx, baseline in enumerate(extra_baselines)
    }
    extra_model_traj_maps: List[Dict[int, np.ndarray]] = []
    for extra in extra_models:
        traj_map: Dict[int, np.ndarray] = {}
        if extra.sample_ids is not None and extra.gnn_trajs_by_sample is not None:
            for sid, traj in zip(extra.sample_ids, extra.gnn_trajs_by_sample):
                traj_map[int(sid)] = np.asarray(traj, dtype=np.float64)
        extra_model_traj_maps.append(traj_map)

    def _baseline_plot_color(scheme: str) -> str:
        if scheme == BASELINE_SCHEME_GODUNOV:
            return godunov_color
        return _baseline_color(
            baseline_index_by_scheme.get(scheme, 0),
            scheme,
        )

    def _build_field_totals_figure(
        *,
        sid: int,
        true_arr: np.ndarray,
        gnn_arr: np.ndarray,
        baseline_rows: List[Tuple[str, str, np.ndarray]],
        extra_model_rows: List[Tuple[int, str, np.ndarray]],
    ):
        total_steps = np.arange(true_arr.shape[0], dtype=np.int64)
        fig_total, ax_total = plt.subplots(
            1,
            1,
            figsize=(8.0, 4.8),
            constrained_layout=True,
        )
        if field_totals_include_exact:
            ax_total.plot(
                total_steps,
                _field_total_curve(true_arr),
                color=exact_color,
                lw=2.4,
                label="Exact",
            )
        ax_total.plot(
            total_steps,
            _field_total_curve(gnn_arr),
            color=gnn_color,
            lw=2.3,
            label=gnn_label,
        )
        for scheme, label, arr in baseline_rows:
            ax_total.plot(
                total_steps,
                _field_total_curve(arr),
                color=_baseline_plot_color(scheme),
                lw=2.1,
                ls="--",
                label=label,
            )
        for extra_idx, label, arr in extra_model_rows:
            extra_steps = np.arange(arr.shape[0], dtype=np.int64)
            ax_total.plot(
                extra_steps,
                _field_total_curve(arr),
                color=_extra_model_color(extra_idx),
                lw=2.0,
                label=label,
        )
        #ax_total.set_title(f"Field Total vs Time | id={sid}")
        ax_total.set_title(f"Linear Advection: Field Total vs Time", fontsize=20)
        ax_total.set_xlabel("Time Step", fontsize=15)
        ax_total.set_ylabel(r"$\sum_i u_i$", fontsize=15)
        ax_total.grid(alpha=0.3)
        ax_total.legend(loc="best", fontsize=15)
        ax_total.tick_params(axis="x", labelsize=14)
        ax_total.tick_params(axis="y", labelsize=14)
        return fig_total

    have_all_sample_trajs = (
        all_sample_ids is not None
        and all_true_trajs is not None
        and all_gnn_trajs is not None
        and all_god_trajs is not None
        and len(all_sample_ids) == len(all_true_trajs)
        and len(all_sample_ids) == len(all_gnn_trajs)
        and len(all_sample_ids) == len(all_god_trajs)
    )
    extra_all_trajs = (
        {} if all_extra_baseline_trajs is None else all_extra_baseline_trajs
    )

    def _baseline_rows_for_sample(
        *,
        sample_idx: int,
        god_arr: np.ndarray,
    ) -> List[Tuple[str, str, np.ndarray]]:
        baseline_rows: List[Tuple[str, str, np.ndarray]] = []
        if show_godunov_baseline:
            baseline_rows.append((BASELINE_SCHEME_GODUNOV, godunov_label, god_arr))
        for baseline in extra_baselines:
            scheme_trajs = extra_all_trajs.get(baseline.scheme)
            if scheme_trajs is None or sample_idx >= len(scheme_trajs):
                continue
            baseline_rows.append(
                (baseline.scheme, baseline.label, scheme_trajs[sample_idx])
            )
        return baseline_rows

    def _extra_model_rows_for_sample(
        *,
        sample_id: int,
    ) -> List[Tuple[int, str, np.ndarray]]:
        rows: List[Tuple[int, str, np.ndarray]] = []
        for extra_idx, extra in enumerate(extra_models):
            traj = extra_model_traj_maps[extra_idx].get(int(sample_id))
            if traj is None:
                continue
            rows.append((extra_idx, extra.label, traj))
        return rows

    if have_all_sample_trajs:
        from matplotlib.backends.backend_pdf import PdfPages

        totals_pdf_path = plots_dir / "field_totals_compare_all.pdf"
        with PdfPages(totals_pdf_path) as pdf:
            for sample_idx, (sid, true_arr, gnn_arr, god_arr) in enumerate(
                zip(all_sample_ids, all_true_trajs, all_gnn_trajs, all_god_trajs)
            ):
                fig_total = _build_field_totals_figure(
                    sid=int(sid),
                    true_arr=true_arr,
                    gnn_arr=gnn_arr,
                    baseline_rows=_baseline_rows_for_sample(
                        sample_idx=sample_idx,
                        god_arr=god_arr,
                    ),
                    extra_model_rows=_extra_model_rows_for_sample(
                        sample_id=int(sid),
                    ),
                )
                pdf.savefig(fig_total, dpi=170)
                plt.close(fig_total)

    # 4) Optional multi-page PDF with one spacetime comparison page per sample.
    if plot_all_samples and have_all_sample_trajs:
        from matplotlib.backends.backend_pdf import PdfPages

        pdf_path = plots_dir / "sample_spacetime_compare_all.pdf"
        with PdfPages(pdf_path) as pdf:
            for sample_idx, (sid, true_arr, gnn_arr, god_arr) in enumerate(
                zip(all_sample_ids, all_true_trajs, all_gnn_trajs, all_god_trajs)
            ):
                fig3 = _build_spacetime_compare_figure(
                    sid=int(sid),
                    true_arr=true_arr,
                    gnn_arr=gnn_arr,
                    baseline_rows=_baseline_rows_for_sample(
                        sample_idx=sample_idx,
                        god_arr=god_arr,
                    ),
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
    plot_baseline_schemes = _resolve_baseline_schemes(args.baseline_schemes)
    spacetime_delta_plots = _resolve_spacetime_delta_plots(
        args.spacetime_delta_plots,
        plot_baseline_schemes=plot_baseline_schemes,
    )
    error_curve_metrics = _resolve_error_curve_metrics(args.error_curves_metrics)
    field_totals_include_exact = not bool(args.no_field_totals_exact)
    baseline_schemes = list(plot_baseline_schemes)
    if BASELINE_SCHEME_GODUNOV not in baseline_schemes:
        baseline_schemes.insert(0, BASELINE_SCHEME_GODUNOV)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    data_cfg = dict(cfg.get("data", {}))
    rollout_cfg = dict(cfg.get("rollout_eval", {}))
    target_mode = str(data_cfg.get("target_mode", "state")).strip().lower()
    godunov_cfg = _resolve_godunov_correction_cfg(cfg)
    if godunov_cfg.enabled and (target_mode == "state"):
        raise ValueError(
            "godunov_correction is enabled but data.target_mode='state'. "
            "Use target_mode='delta' or 'rate' for correction learning."
        )

    cfg_model = dict(cfg.get("model", {}))
    if args.state_residual_mode is not None:
        cfg_model["state_residual_mode"] = str(args.state_residual_mode)
    cfg_model["predict_type"] = predict_type_from_target_mode(target_mode)
    cfg["model"] = cfg_model
    state_residual_mode = resolve_state_residual_mode(cfg)
    cfg_model["state_residual_mode"] = state_residual_mode
    cfg["model"] = cfg_model

    dataset_path = args.dataset or Path(data_cfg.get("dataset_path", "data/advection_train.npz"))
    num_samples = int(
        args.num_samples
        if args.num_samples is not None
        else rollout_cfg.get("num_test_samples", 16)
    )
    if num_samples < 1:
        raise ValueError(f"--num-samples must be >= 1, got {num_samples}.")

    dset = load_advection_dataset(dataset_path, dtype=torch.float32)
    c = float(dset.metadata["c"])
    dt = float(dset.metadata["dt"])
    dx = float(dset.metadata["dx"])
    L = float(dset.metadata["L"])
    pos = dset.pos.squeeze(-1).detach().cpu().numpy()
    physics_cfg = resolve_physics_input_cfg(cfg)
    input_cfg = resolve_input_feature_cfg(cfg)
    if input_cfg.normalize_state_to_minus1_1:
        if input_cfg.state_min is None or input_cfg.state_max is None:
            raise ValueError(
                "input_features.normalize_state_to_minus1_1 is enabled, but "
                "state_min/state_max are missing from checkpoint config."
            )
    if (
        physics_cfg.enabled
        and physics_cfg.include_adv
        and physics_cfg.normalize_adv_to_minus1_1
        and (physics_cfg.adv_min is None or physics_cfg.adv_max is None)
    ):
        raise ValueError(
            "physics_inputs.normalize_adv_to_minus1_1 is enabled, but adv_min/adv_max "
            "are missing from checkpoint config."
        )

    split = ckpt.get("split_indices", {})
    eval_mode, eval_ids = _resolve_eval_mode_and_ids(
        dset=dset,
        split=split,
        num_samples=num_samples,
    )
    if len(eval_ids) == 0:
        raise ValueError("No evaluation samples found in checkpoint split indices.")
    print(
        f"[INFO] Evaluation samples: mode={eval_mode}, count={len(eval_ids)}",
        flush=True,
    )

    horizon_default = int(rollout_cfg.get("horizon", 100))
    if dset.rollout_states is not None:
        max_h = int(dset.rollout_states.shape[1] - 1)
        if args.horizon is None:
            horizon = min(horizon_default, max_h)
        else:
            horizon = min(int(args.horizon), max_h)
            if int(args.horizon) > max_h:
                print(
                    f"[INFO] Requested horizon {args.horizon} exceeds available rollout "
                    f"window {max_h}; using horizon={horizon}.",
                    flush=True,
                )
    else:
        horizon = int(args.horizon) if args.horizon is not None else horizon_default
    if horizon < 1:
        raise ValueError(f"Rollout horizon must be >= 1, got {horizon}.")

    reference_trajs_by_id: Dict[int, np.ndarray] = {}
    for sid in eval_ids:
        if eval_mode == "trajectory":
            assert dset.rollout_states is not None
            true_traj = (
                dset.rollout_states[int(sid), : horizon + 1, :, 0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
        else:
            initial_state = (
                dset.x[int(sid)].detach().cpu().numpy().reshape(-1).astype(np.float64)
            )
            true_traj = np.empty(
                (horizon + 1, int(initial_state.size)),
                dtype=np.float64,
            )
            true_traj[0] = initial_state
            for k in range(horizon):
                true_traj[k + 1] = exact_advect_spectral_shift(
                    true_traj[k], c=c, dt=dt, L=L
                )
        reference_trajs_by_id[int(sid)] = true_traj

    ssim_config = _resolve_ssim_config(
        args,
        reference_trajs=list(reference_trajs_by_id.values()),
        num_nodes=int(dset.x.shape[1]),
    )
    if ssim_config.mode == SSIM_MODE_LOCAL_PERIODIC:
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

    cfl_mag = abs(c) * dt / dx
    if cfl_mag > 1.0:
        print(
            f"[WARN] Explicit finite-volume baselines use |c|*dt/dx={cfl_mag:.3f} > 1; "
            "baseline rollouts may be unstable.",
            flush=True,
        )

    device_pref = str(args.device if args.device is not None else cfg.get("device", "auto"))
    device = resolve_device(device_pref)
    print(
        "[INFO] State residual mode: "
        f"{state_residual_mode}",
        flush=True,
    )
    in_channels = 1 + count_physics_input_channels(physics_cfg) + count_input_feature_channels(input_cfg)
    model = build_model_from_config(
        cfg,
        in_channels=in_channels,
        edge_attr_channels=int(dset.edge_attr.shape[-1]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    edge_index = dset.edge_index.detach().cpu().long()
    edge_attr = dset.edge_attr.detach().cpu()
    pos_nodes = dset.pos.detach().cpu()

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

    sample_id: int | None = None
    sample_true_traj: np.ndarray | None = None
    sample_gnn_traj: np.ndarray | None = None
    sample_god_traj: np.ndarray | None = None
    sample_extra_baseline_trajs: Dict[str, np.ndarray] = {}
    all_sample_ids: List[int] = []
    all_true_trajs: List[np.ndarray] = []
    all_gnn_trajs: List[np.ndarray] = []
    all_god_trajs: List[np.ndarray] = []
    all_extra_baseline_trajs: Dict[str, List[np.ndarray]] = {
        scheme: []
        for scheme in baseline_schemes
        if scheme != BASELINE_SCHEME_GODUNOV
    }

    for sid in eval_ids:
        true_traj = reference_trajs_by_id[int(sid)]
        x0 = torch.from_numpy(true_traj[0][:, None]).to(dtype=torch.float32)

        gnn_traj = _rollout_gnn(
            model=model,
            x0=x0,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos_nodes=pos_nodes,
            c=c,
            dt=dt,
            dx=dx,
            horizon=horizon,
            device=device,
            physics_cfg=physics_cfg,
            input_cfg=input_cfg,
            target_mode=target_mode,
            state_residual_mode=state_residual_mode,
            godunov_cfg=godunov_cfg,
        )
        baseline_trajs = {
            scheme: _rollout_baseline(
                x0=true_traj[0],
                c=c,
                dt=dt,
                dx=dx,
                horizon=horizon,
                scheme=scheme,
            )
            for scheme in baseline_schemes
        }
        god_traj = baseline_trajs[BASELINE_SCHEME_GODUNOV]

        rmse_gnn, rel_gnn, l1_gnn, ssim_gnn = _compute_curves(
            pred_traj=gnn_traj,
            true_traj=true_traj,
            ssim_config=ssim_config,
        )
        rmse_god, rel_god, l1_god, ssim_god = _compute_curves(
            pred_traj=god_traj,
            true_traj=true_traj,
            ssim_config=ssim_config,
        )
        for scheme, metrics in extra_baseline_metric_lists.items():
            rmse_b, rel_b, l1_b, ssim_b = _compute_curves(
                pred_traj=baseline_trajs[scheme],
                true_traj=true_traj,
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
        all_sample_ids.append(int(sid))
        all_true_trajs.append(true_traj)
        all_gnn_trajs.append(gnn_traj)
        all_god_trajs.append(god_traj)
        for scheme, trajs in all_extra_baseline_trajs.items():
            trajs.append(baseline_trajs[scheme])

        if sample_id is None:
            sample_id = int(sid)
            sample_true_traj = true_traj
            sample_gnn_traj = gnn_traj
            sample_god_traj = god_traj
            sample_extra_baseline_trajs = {
                scheme: traj
                for scheme, traj in baseline_trajs.items()
                if scheme != BASELINE_SCHEME_GODUNOV
            }

    rmse_gnn_arr_raw = np.stack(rmse_gnn_all, axis=0)
    rel_gnn_arr_raw = np.stack(rel_gnn_all, axis=0)
    l1_gnn_arr_raw = np.stack(l1_gnn_all, axis=0)
    ssim_gnn_arr_raw = np.stack(ssim_gnn_all, axis=0)
    rmse_god_arr_raw = np.stack(rmse_god_all, axis=0)
    rel_god_arr_raw = np.stack(rel_god_all, axis=0)
    l1_god_arr_raw = np.stack(l1_god_all, axis=0)
    ssim_god_arr_raw = np.stack(ssim_god_all, axis=0)

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
            extra_eval_curves.append(extra)
            ssim_incompatibility = _ssim_comparison_incompatibility(
                ssim_config,
                extra,
            )
            if ERROR_CURVE_METRIC_SSIM in error_curve_metrics and (
                ssim_incompatibility is not None
            ):
                print(
                    f"[WARN] Comparison model '{label}' has incompatible SSIM "
                    f"settings ({ssim_incompatibility}). Its overlaid SSIM curve is not "
                    "definition-compatible and should not be compared directly.",
                    flush=True,
                )
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
            if extra.sample_ids is None or extra.gnn_trajs_by_sample is None:
                print(
                    f"[INFO] Comparison model '{label}' has no saved rollout "
                    "trajectories; it will not appear in field_totals_compare_all.pdf.",
                    flush=True,
                )
            else:
                print(
                    f"[INFO] Comparison model '{label}' has saved rollout "
                    f"trajectories for {len(extra.sample_ids)} sample(s); "
                    "field-total overlays appear on matching sample-id pages.",
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
    npz_payload = {
        "eval_mode": np.array(eval_mode),
        "sample_ids": np.array(eval_ids, dtype=np.int64),
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
        "baseline_schemes": np.array(plot_baseline_schemes),
        "computed_baseline_schemes": np.array(baseline_schemes),
        "baseline_labels": np.array(
            [baseline_label_by_scheme[s] for s in plot_baseline_schemes]
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
        "sample_traj_id": np.int64(-1 if sample_id is None else sample_id),
        "sample_true_traj": np.array([]) if sample_true_traj is None else sample_true_traj,
        "sample_gnn_traj": np.array([]) if sample_gnn_traj is None else sample_gnn_traj,
        "sample_godunov_traj": np.array([]) if sample_god_traj is None else sample_god_traj,
        "all_sample_ids": np.array(all_sample_ids, dtype=np.int64),
        "all_gnn_trajs": (
            np.stack(all_gnn_trajs, axis=0)
            if len(all_gnn_trajs) > 0
            else np.empty((0, 0, 0), dtype=np.float64)
        ),
        "field_totals_include_exact": np.array(field_totals_include_exact),
        "error_curve_metrics": np.array(error_curve_metrics),
        "error_curves_spread": np.array(str(args.error_curves_spread)),
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
        "dataset": str(dataset_path),
        "device_requested": device_pref,
        "device_resolved": str(device),
        "eval_mode": eval_mode,
        "num_samples": int(len(eval_ids)),
        "horizon": int(horizon),
        "c": float(c),
        "dt": float(dt),
        "dx": float(dx),
        "cfl_abs": float(cfl_mag),
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
            scheme: baseline_label_by_scheme[scheme] for scheme in plot_baseline_schemes
        },
        "model_in_channels": int(in_channels),
        "state_residual_mode": str(state_residual_mode),
        "physics_inputs": {
            "enabled": bool(physics_cfg.enabled),
            "include_adv": bool(physics_cfg.include_adv),
            "include_diff": bool(physics_cfg.include_diff),
            "input_form": str(physics_cfg.input_form),
            "backend": str(physics_cfg.backend),
            "advection_scheme": str(physics_cfg.advection_scheme),
            "mls_ridge_eps": float(physics_cfg.mls_ridge_eps),
            "nu": float(physics_cfg.nu),
            "c_override": (
                None if physics_cfg.c_override is None else float(physics_cfg.c_override)
            ),
            "adv_all_steps": bool(physics_cfg.adv_all_steps),
            "input_weighted": bool(physics_cfg.input_weighted),
            "adv_weight": float(physics_cfg.adv_weight),
            "diff_weight": float(physics_cfg.diff_weight),
            "detach_inputs": bool(physics_cfg.detach_inputs),
            "normalize_adv_to_minus1_1": bool(
                physics_cfg.normalize_adv_to_minus1_1
            ),
            "adv_min": (
                None if physics_cfg.adv_min is None else float(physics_cfg.adv_min)
            ),
            "adv_max": (
                None if physics_cfg.adv_max is None else float(physics_cfg.adv_max)
            ),
            "normalize_adv_clip": bool(physics_cfg.normalize_adv_clip),
        },
        "godunov_correction": {
            "enabled": bool(godunov_cfg.enabled),
            "advection_scheme": str(godunov_cfg.advection_scheme),
            "c_override": (
                None if godunov_cfg.c_override is None else float(godunov_cfg.c_override)
            ),
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
        "spacetime_delta_plots": (
            None if spacetime_delta_plots is None else list(spacetime_delta_plots)
        ),
        "error_curves_log_y": bool(args.error_curves_log_y),
        "error_curves_mean_only": bool(args.error_curves_mean_only),
        "error_curves_smooth_window": int(args.error_curves_smooth_window),
        "error_curves_spread": str(args.error_curves_spread),
        "error_curves_spread_definition": {
            "std": "mean plus/minus one population standard deviation across test trajectories",
            "iqr": "25th--75th percentiles across test trajectories",
        },
        "error_curve_metrics": list(error_curve_metrics),
        "field_totals_include_exact": bool(field_totals_include_exact),
        "curve_color_overrides": dict(curve_color_overrides),
        "legend_label_gnn": gnn_label,
        "legend_label_godunov": godunov_label,
        "num_compare_models": int(len(extra_eval_curves)),
        "compare_model_labels": [m.label for m in extra_eval_curves],
        "compare_eval_npz": [m.source_npz for m in extra_eval_curves],
        "out_npz": str(out_npz),
    }

    plots_dir = args.plots_dir if args.plots_dir is not None else (args.checkpoint.parent / "gnn_vs_godunov_plots")
    if not args.no_plots:
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
            pos=pos,
            plot_all_samples=bool(args.plot_all_samples),
            spacetime_delta_plots=spacetime_delta_plots,
            error_curves_log_y=bool(args.error_curves_log_y),
            error_curves_mean_only=bool(args.error_curves_mean_only),
            error_curves_smooth_window=int(args.error_curves_smooth_window),
            error_curves_spread=str(args.error_curves_spread),
            error_curve_metrics=error_curve_metrics,
            ssim_config=ssim_config,
            curve_colors=curve_color_overrides,
            field_totals_include_exact=field_totals_include_exact,
            extra_baseline_curves=extra_baseline_curves,
            extra_eval_curves=extra_eval_curves,
            plot_baseline_schemes=plot_baseline_schemes,
            all_sample_ids=all_sample_ids,
            all_true_trajs=all_true_trajs,
            all_gnn_trajs=all_gnn_trajs,
            all_god_trajs=all_god_trajs,
            all_extra_baseline_trajs=all_extra_baseline_trajs,
        )
        summary["plots_dir"] = str(plots_dir)
        summary["field_totals_pdf"] = str(
            plots_dir / "field_totals_compare_all.pdf"
        )
        if args.plot_all_samples:
            summary["all_sample_plots_pdf"] = str(
                plots_dir / "sample_spacetime_compare_all.pdf"
            )

    save_json(summary, args.checkpoint.parent / "gnn_vs_godunov_summary.json")

    print("GNN vs finite-volume baseline comparison complete.")
    print(f"  eval_mode: {eval_mode}")
    print(f"  samples: {len(eval_ids)}")
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
    print(f"  out: {out_npz}")
    if not args.no_plots:
        print(f"  plots: {plots_dir}")
        print(
            "  field totals PDF: "
            f"{plots_dir / 'field_totals_compare_all.pdf'}"
        )
        print(
            "  field totals include Exact: "
            f"{field_totals_include_exact}"
        )
        print(f"  error-curve smoothing window: {int(args.error_curves_smooth_window)}")
        print(f"  error-curve spread: {args.error_curves_spread}")
        print(
            "  error-curve metrics: "
            + ", ".join(ERROR_CURVE_METRIC_DISPLAY[m] for m in error_curve_metrics)
        )
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
            delta_desc = (
                "default"
                if spacetime_delta_plots is None
                else (
                    "none"
                    if len(spacetime_delta_plots) == 0
                    else ", ".join(spacetime_delta_plots)
                )
            )
            print(f"  spacetime delta plots: {delta_desc}")
    if len(extra_eval_curves) > 0:
        print(
            "  comparison models: "
            + ", ".join(m.label for m in extra_eval_curves)
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
        and frac_gnn is not None
        and frac_god is not None
        and frac_tie is not None
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


if __name__ == "__main__":
    main()
