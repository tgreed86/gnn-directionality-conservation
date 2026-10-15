#!/usr/bin/env python3
"""Create Figure-6-style rollout-error comparisons for LWR traffic models.

The metric calculations are imported from ``plot_model_comparisons.py`` so
that both plotting entry points use the same cell-length-weighted relative L2
and roadwise local spatial SSIM definitions.  This script changes only the
presentation and the set of generated artifacts.

For example, the following produces the two-panel layout used by each row of
Figure 6 in ``7_6_26_scirep_v2.pdf``::

    python scripts/plot_model_comparisons_paper.py \
        --model "FluxGraphNet=path/to/fluxgraphnet/run" \
        --model "MeshGraphNet=path/to/meshgraphnet/run" \
        --data data/lwr_structured_directional \
        --splits test_in_size \
        --error-curves-metrics rel_l2 ssim \
        --output-dir output/figure6_style

The default nine-step centered moving average matches the smoothing stated in
the Figure 6 caption.  The horizontal axis remains rollout step because the
traffic trajectories do not share one physical time step.

Use ``--combined-mean-only`` to hide individual test-split curves in the
combined-test figure, leaving one combined mean curve per model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import is_color_like
import numpy as np

from plot_model_comparisons import (
    SSIM_K1,
    SSIM_K2,
    SSIM_SIGMA_CELLS,
    SSIM_WINDOW_SIZE,
    ModelSpec,
    RolloutData,
    TrajectoryMetrics,
    common_values,
    compute_metrics,
    discover_rollout_files,
    infer_dataset_root,
    load_rollout,
    load_topology_metric_metadata,
    parse_model_spec,
    resolve_ssim_reference_spacing,
    topology_balanced_mean_curve,
    topology_paths_from_manifest,
    trajectory_index,
    validate_common_rollouts,
)


ERROR_CURVE_METRIC_RMSE = "rmse"
ERROR_CURVE_METRIC_REL_L2 = "rel_l2"
ERROR_CURVE_METRIC_L1 = "l1"
ERROR_CURVE_METRIC_SSIM = "ssim"
ERROR_CURVE_METRICS_DEFAULT = (
    ERROR_CURVE_METRIC_RMSE,
    ERROR_CURVE_METRIC_REL_L2,
    ERROR_CURVE_METRIC_L1,
    ERROR_CURVE_METRIC_SSIM,
)
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
    "local-ssim": ERROR_CURVE_METRIC_SSIM,
    "local_ssim": ERROR_CURVE_METRIC_SSIM,
}
ERROR_CURVE_METRIC_DISPLAY = {
    ERROR_CURVE_METRIC_RMSE: "RMSE",
    ERROR_CURVE_METRIC_REL_L2: r"rel-$L_2$",
    ERROR_CURVE_METRIC_L1: "L1",
    ERROR_CURVE_METRIC_SSIM: "Local SSIM",
}
ERROR_CURVE_METRIC_ATTRIBUTE = {
    ERROR_CURVE_METRIC_RMSE: "rmse_by_time",
    ERROR_CURVE_METRIC_REL_L2: "weighted_relative_l2_by_time",
    ERROR_CURVE_METRIC_L1: "mae_by_time",
    ERROR_CURVE_METRIC_SSIM: "local_ssim_by_time",
}
ERROR_CURVE_SPREAD_NONE = "none"
ERROR_CURVE_SPREAD_STD = "std"
ERROR_CURVE_SPREAD_IQR = "iqr"
ERROR_CURVE_SPREAD_CHOICES = (
    ERROR_CURVE_SPREAD_NONE,
    ERROR_CURVE_SPREAD_STD,
    ERROR_CURVE_SPREAD_IQR,
)

FIGURE6_FAMILY_STYLES = {
    "fluxgraphnet": {"color": "tab:blue", "linestyle": "-"},
    "godunov": {"color": "tab:green", "linestyle": "--"},
    "weno3": {"color": "tab:red", "linestyle": "--"},
    "meshgraphnet": {"color": "magenta", "linestyle": "-"},
    "sageconv": {"color": "tab:orange", "linestyle": "-"},
}
FALLBACK_COLORS = (
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
    "black",
)


def resolve_error_curve_metrics(
    raw_metrics: Optional[Sequence[str]],
) -> List[str]:
    """Normalize metric aliases while preserving the requested panel order."""
    if raw_metrics is None:
        return list(ERROR_CURVE_METRICS_DEFAULT)

    selected: List[str] = []
    for raw_item in raw_metrics:
        for part in str(raw_item).split(","):
            token = part.strip().lower()
            if not token:
                continue
            if token == "all":
                for metric in ERROR_CURVE_METRICS_DEFAULT:
                    if metric not in selected:
                        selected.append(metric)
                continue
            metric = ERROR_CURVE_METRIC_ALIASES.get(token)
            if metric is None:
                allowed = ", ".join(ERROR_CURVE_METRICS_DEFAULT)
                raise ValueError(
                    f"Unknown --error-curves-metrics value {part!r}. "
                    f"Expected one of {allowed}, or 'all'."
                )
            if metric not in selected:
                selected.append(metric)
    if not selected:
        raise ValueError("At least one error-curve metric must be requested.")
    return selected


def centered_moving_average(curve: np.ndarray, window: int) -> np.ndarray:
    """Return a NaN-aware centered mean with truncated endpoint windows."""
    values = np.asarray(curve, dtype=np.float64).reshape(-1)
    if window <= 0 or window % 2 == 0:
        raise ValueError("The smoothing window must be a positive odd integer.")
    if window == 1 or values.size == 0:
        return values.copy()

    radius = window // 2
    smoothed = np.full(values.shape, np.nan, dtype=np.float64)
    for index in range(values.size):
        start = max(0, index - radius)
        stop = min(values.size, index + radius + 1)
        local = values[start:stop]
        finite = local[np.isfinite(local)]
        if finite.size:
            smoothed[index] = float(np.mean(finite))
    return smoothed


def smooth_curves(curves: np.ndarray, window: int) -> np.ndarray:
    """Apply the centered smoother to one curve or a curve matrix."""
    values = np.asarray(curves, dtype=np.float64)
    if values.ndim == 1:
        return centered_moving_average(values, window)
    if values.ndim == 2:
        return np.stack(
            [centered_moving_average(row, window) for row in values],
            axis=0,
        )
    raise ValueError(
        f"Expected one- or two-dimensional curves, got {values.shape}."
    )


def metric_curve_samples(
    metrics: Sequence[TrajectoryMetrics],
    attribute: str,
    *,
    topology_balanced: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stack trajectory curves and return normalized aggregation weights."""
    if not metrics:
        raise ValueError("At least one trajectory metric is required.")

    curves = np.stack(
        [
            np.asarray(getattr(item, attribute), dtype=np.float64)
            for item in metrics
        ],
        axis=0,
    )
    if topology_balanced:
        counts_by_topology: Dict[int, int] = {}
        for item in metrics:
            counts_by_topology[item.topology_id] = (
                counts_by_topology.get(item.topology_id, 0) + 1
            )
        num_topologies = len(counts_by_topology)
        weights = np.asarray(
            [
                1.0
                / (
                    num_topologies
                    * counts_by_topology[item.topology_id]
                )
                for item in metrics
            ],
            dtype=np.float64,
        )
    else:
        weights = np.full(len(metrics), 1.0 / len(metrics), dtype=np.float64)
    return curves, weights


def combined_spread_curve_samples(
    metrics_by_split: Mapping[
        str, Mapping[str, Sequence[TrajectoryMetrics]]
    ],
    selected_metrics: Sequence[str],
    averaging: str,
) -> Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """Collect trajectory curves with weights matching the combined mean."""
    if not metrics_by_split:
        raise ValueError("At least one split is required for a spread band.")
    if averaging not in ("macro", "standard"):
        raise ValueError(f"Unknown combined averaging mode {averaging!r}.")

    split_names = list(metrics_by_split)
    labels = list(metrics_by_split[split_names[0]])
    samples: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {
        label: {} for label in labels
    }
    for label in labels:
        for metric in selected_metrics:
            attribute = ERROR_CURVE_METRIC_ATTRIBUTE[metric]
            curves: List[np.ndarray] = []
            curve_weights: List[float] = []
            for split in split_names:
                split_metrics = metrics_by_split[split][label]
                split_curves, split_weights = metric_curve_samples(
                    split_metrics,
                    attribute,
                    topology_balanced=(averaging == "macro"),
                )
                if averaging == "macro":
                    split_weights = split_weights / len(split_names)
                else:
                    split_weights = np.ones(
                        len(split_metrics), dtype=np.float64
                    )
                curves.extend(list(split_curves))
                curve_weights.extend(split_weights.tolist())

            common_length = min(curve.size for curve in curves)
            aligned = np.stack(
                [curve[:common_length] for curve in curves], axis=0
            )
            weights = np.asarray(curve_weights, dtype=np.float64)
            weights = weights / np.sum(weights)
            samples[label][metric] = aligned, weights
    return samples


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float:
    """Return a linearly interpolated weighted quantile for finite values."""
    finite = (
        np.isfinite(values)
        & np.isfinite(weights)
        & (weights > 0.0)
    )
    finite_values = np.asarray(values[finite], dtype=np.float64)
    finite_weights = np.asarray(weights[finite], dtype=np.float64)
    if finite_values.size == 0:
        return float("nan")
    if finite_values.size == 1:
        return float(finite_values[0])
    if np.allclose(finite_weights, finite_weights[0]):
        return float(np.quantile(finite_values, quantile))

    order = np.argsort(finite_values)
    sorted_values = finite_values[order]
    sorted_weights = finite_weights[order]
    midpoints = np.cumsum(sorted_weights) - 0.5 * sorted_weights
    span = midpoints[-1] - midpoints[0]
    if span <= 0.0:
        return float(sorted_values[0])
    positions = (midpoints - midpoints[0]) / span
    return float(np.interp(quantile, positions, sorted_values))


def error_curve_spread_bounds(
    curves: np.ndarray,
    spread: str,
    *,
    weights: Optional[np.ndarray] = None,
    center: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate weighted std or 25th--75th percentile curve bounds."""
    values = np.asarray(curves, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError(
            "Spread curves must have shape [num_trajectories, num_times]."
        )
    if spread not in (ERROR_CURVE_SPREAD_STD, ERROR_CURVE_SPREAD_IQR):
        raise ValueError(
            f"Spread bounds require 'std' or 'iqr', got {spread!r}."
        )

    if weights is None:
        sample_weights = np.full(
            values.shape[0], 1.0 / values.shape[0], dtype=np.float64
        )
    else:
        sample_weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        if sample_weights.size != values.shape[0]:
            raise ValueError(
                "The number of spread weights must match the number of curves."
            )
        if (
            np.any(~np.isfinite(sample_weights))
            or np.any(sample_weights < 0.0)
            or np.sum(sample_weights) <= 0.0
        ):
            raise ValueError("Spread weights must be finite and nonnegative.")
        sample_weights = sample_weights / np.sum(sample_weights)

    lower = np.full(values.shape[1], np.nan, dtype=np.float64)
    upper = np.full(values.shape[1], np.nan, dtype=np.float64)
    if center is None:
        center_values = np.full(values.shape[1], np.nan, dtype=np.float64)
    else:
        center_values = np.asarray(center, dtype=np.float64).reshape(-1)
        if center_values.size != values.shape[1]:
            raise ValueError(
                "The spread center must have the same time dimension as curves."
            )

    for time_index in range(values.shape[1]):
        column = values[:, time_index]
        finite = np.isfinite(column) & (sample_weights > 0.0)
        if not np.any(finite):
            continue
        local_values = column[finite]
        local_weights = sample_weights[finite]
        local_weights = local_weights / np.sum(local_weights)
        if spread == ERROR_CURVE_SPREAD_IQR:
            lower[time_index] = _weighted_quantile(
                local_values, local_weights, 0.25
            )
            upper[time_index] = _weighted_quantile(
                local_values, local_weights, 0.75
            )
            continue

        weighted_mean = float(np.sum(local_weights * local_values))
        deviation = float(
            np.sqrt(
                np.sum(local_weights * (local_values - weighted_mean) ** 2)
            )
        )
        band_center = center_values[time_index]
        if not np.isfinite(band_center):
            band_center = weighted_mean
        lower[time_index] = band_center - deviation
        upper[time_index] = band_center + deviation
    return lower, upper


def clip_spread_bounds(
    metric: str,
    lower: np.ndarray,
    upper: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Clip shaded bands to the mathematical range of each metric."""
    if metric == ERROR_CURVE_METRIC_SSIM:
        return np.maximum(lower, -1.0), np.minimum(upper, 1.0)
    return np.maximum(lower, 0.0), upper


def normalize_style_key(value: str) -> str:
    """Normalize a model label for semantic style and override matching."""
    return "".join(character for character in value.lower() if character.isalnum())


def model_family(label: str) -> Optional[str]:
    """Recognize the model families styled explicitly in Figure 6."""
    key = normalize_style_key(label)
    if "fluxgraphnet" in key:
        return "fluxgraphnet"
    if "meshgraphnet" in key:
        return "meshgraphnet"
    if "sageconv" in key:
        return "sageconv"
    if "godunov" in key:
        return "godunov"
    if "weno3" in key or key == "weno":
        return "weno3"
    return None


def parse_curve_color_overrides(
    raw_values: Optional[Sequence[str]],
) -> Dict[str, str]:
    """Parse LABEL=COLOR settings used to override the paper palette."""
    overrides: Dict[str, str] = {}
    for raw_item in raw_values or ():
        for part in str(raw_item).split(","):
            token = part.strip()
            if not token:
                continue
            if "=" not in token:
                raise ValueError(
                    f"Invalid --curve-color value {token!r}; expected LABEL=COLOR."
                )
            label, color = token.split("=", 1)
            key = normalize_style_key(label)
            color = color.strip()
            if not key or not color or not is_color_like(color):
                raise ValueError(
                    f"Invalid --curve-color value {token!r}; {color!r} is not "
                    "a Matplotlib color."
                )
            overrides[key] = color
    return overrides


def figure6_model_styles(
    labels: Sequence[str],
    *,
    color_overrides: Optional[Mapping[str, str]] = None,
) -> Dict[str, Dict[str, object]]:
    """Assign the label-driven colors and line styles visible in Figure 6."""
    overrides = dict(color_overrides or {})
    used_colors = set()
    family_uses: Dict[str, int] = {}
    styles: Dict[str, Dict[str, object]] = {}
    fallback_index = 0

    for label in labels:
        family = model_family(label)
        family_use = 0 if family is None else family_uses.get(family, 0)
        if family is not None:
            family_uses[family] = family_use + 1

        if family is not None and family_use == 0:
            style = dict(FIGURE6_FAMILY_STYLES[family])
        else:
            while (
                fallback_index < len(FALLBACK_COLORS)
                and FALLBACK_COLORS[fallback_index] in used_colors
            ):
                fallback_index += 1
            color = FALLBACK_COLORS[fallback_index % len(FALLBACK_COLORS)]
            fallback_index += 1
            style = {"color": color, "linestyle": "-"}

        label_key = normalize_style_key(label)
        override = overrides.get(label_key)
        if override is None and family is not None:
            override = overrides.get(family)
        if override is not None:
            style["color"] = override
        style["linewidth"] = 2.0
        used_colors.add(str(style["color"]))
        styles[label] = style
    return styles


def pretty_split_name(split: str) -> str:
    """Make a compact publication title from a dataset split key."""
    replacements = {
        "test_in_size": "In-size test",
        "test_large": "Large-network test",
        "test_parameter_ood": "Parameter-OOD test",
    }
    return replacements.get(split, split.replace("_", " ").title())


def resolve_figure_title(template: Optional[str], split: str) -> str:
    """Resolve the default title or interpolate a user title template."""
    pretty_split = pretty_split_name(split)
    if template is None:
        #return f"1D LWR Traffic ({pretty_split}): Rollout Errors"
        return f"1D LWR Traffic: Rollout Accuracy"
    return template.format(split=split, split_title=pretty_split)


def plot_figure6_error_curves(
    *,
    split: str,
    metrics_by_model: Mapping[str, Sequence[TrajectoryMetrics]],
    selected_metrics: Sequence[str],
    styles: Mapping[str, Mapping[str, object]],
    output_path: Path,
    dpi: int,
    smooth_window: int,
    mean_only: bool,
    log_y: bool,
    title: str,
    spread: str = ERROR_CURVE_SPREAD_NONE,
) -> None:
    """Plot selectable metric panels using the Figure 6 visual language."""
    if not selected_metrics:
        raise ValueError("At least one metric panel is required.")
    if not metrics_by_model:
        raise ValueError("At least one model is required.")

    panel_count = len(selected_metrics)
    if panel_count == 1:
        n_rows, n_columns, figure_size = 1, 1, (8.0, 4.8)
    elif panel_count == 2:
        n_rows, n_columns, figure_size = 1, 2, (15.5, 4.8)
    else:
        n_rows, n_columns, figure_size = 2, 2, (15.5, 8.3)

    figure, axes_object = plt.subplots(
        n_rows,
        n_columns,
        figsize=figure_size,
        constrained_layout=True,
        squeeze=False,
    )
    axes = list(np.asarray(axes_object, dtype=object).reshape(-1))
    for unused_axis in axes[panel_count:]:
        unused_axis.set_visible(False)

    labels = list(metrics_by_model)
    legend_axis = (
        axes[list(selected_metrics).index(ERROR_CURVE_METRIC_SSIM)]
        if ERROR_CURVE_METRIC_SSIM in selected_metrics
        else axes[panel_count - 1]
    )

    for panel_index, metric in enumerate(selected_metrics):
        axis = axes[panel_index]
        attribute = ERROR_CURVE_METRIC_ATTRIBUTE[metric]
        plotted_means: List[np.ndarray] = []

        for label in labels:
            model_metrics = metrics_by_model[label]
            if not model_metrics:
                raise ValueError(f"Model {label!r} has no trajectory metrics.")
            mean_curve = topology_balanced_mean_curve(
                model_metrics, attribute
            )[1:]
            mean_curve = centered_moving_average(mean_curve, smooth_window)
            plotted_means.append(mean_curve)
            rollout_steps = np.arange(1, mean_curve.size + 1)

            trajectory_curves: Optional[np.ndarray] = None
            trajectory_weights: Optional[np.ndarray] = None
            if not mean_only or spread != ERROR_CURVE_SPREAD_NONE:
                raw_curves, trajectory_weights = metric_curve_samples(
                    model_metrics,
                    attribute,
                    topology_balanced=True,
                )
                trajectory_curves = smooth_curves(
                    raw_curves[:, 1:], smooth_window
                )

            if spread != ERROR_CURVE_SPREAD_NONE:
                assert trajectory_curves is not None
                lower, upper = error_curve_spread_bounds(
                    trajectory_curves,
                    spread,
                    weights=trajectory_weights,
                    center=mean_curve,
                )
                lower, upper = clip_spread_bounds(metric, lower, upper)
                axis.fill_between(
                    rollout_steps,
                    lower,
                    upper,
                    color=styles[label]["color"],
                    alpha=0.14,
                    linewidth=0.0,
                )

            if not mean_only:
                assert trajectory_curves is not None
                for curve in trajectory_curves:
                    axis.plot(
                        rollout_steps,
                        curve,
                        color=styles[label]["color"],
                        linestyle=styles[label]["linestyle"],
                        linewidth=0.75,
                        alpha=0.16,
                    )

            axis.plot(
                rollout_steps,
                mean_curve,
                label=label,
                **styles[label],
            )

        if n_rows == 1 or panel_index >= (n_rows - 1) * n_columns:
            axis.set_xlabel("Rollout step", fontsize=20)
        axis.set_ylabel(ERROR_CURVE_METRIC_DISPLAY[metric], fontsize=20)
        axis.tick_params(axis="both", labelsize=16)
        axis.grid(alpha=0.3)

        if log_y and metric != ERROR_CURVE_METRIC_SSIM:
            finite_positive = np.concatenate(plotted_means)
            finite_positive = finite_positive[
                np.isfinite(finite_positive) & (finite_positive > 0.0)
            ]
            if finite_positive.size:
                axis.set_yscale("log")
        if metric == ERROR_CURVE_METRIC_SSIM:
            finite_ssim = np.concatenate(plotted_means)
            finite_ssim = finite_ssim[np.isfinite(finite_ssim)]
            if finite_ssim.size:
                minimum = max(-1.0, float(np.min(finite_ssim)))
                span = max(1.0 - minimum, 1.0e-3)
                axis.set_ylim(max(-1.0, minimum - 0.05 * span), 1.0)

    legend_axis.legend(
        loc="lower left" if ERROR_CURVE_METRIC_SSIM in selected_metrics else "best",
        fontsize=16,
    )
    figure.suptitle(title, fontsize=20)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def macro_average_split_curves(
    metrics_by_split: Mapping[
        str, Mapping[str, Sequence[TrajectoryMetrics]]
    ],
    selected_metrics: Sequence[str],
) -> Tuple[
    Dict[str, Dict[str, np.ndarray]],
    Dict[str, Dict[str, List[np.ndarray]]],
]:
    """Average topology-balanced split curves equally across test regimes."""
    if not metrics_by_split:
        raise ValueError("At least one split is required for a macro-average.")
    split_names = list(metrics_by_split)
    labels = list(metrics_by_split[split_names[0]])
    if not labels:
        raise ValueError("At least one model is required for a macro-average.")
    expected_labels = set(labels)
    for split, metrics_by_model in metrics_by_split.items():
        if set(metrics_by_model) != expected_labels:
            raise ValueError(
                f"Split {split!r} does not contain the same model labels."
            )

    macro_curves: Dict[str, Dict[str, np.ndarray]] = {
        label: {} for label in labels
    }
    split_curves: Dict[str, Dict[str, List[np.ndarray]]] = {
        label: {} for label in labels
    }
    for label in labels:
        for metric in selected_metrics:
            attribute = ERROR_CURVE_METRIC_ATTRIBUTE[metric]
            curves = [
                topology_balanced_mean_curve(
                    metrics_by_split[split][label], attribute
                )
                for split in split_names
            ]
            common_length = min(curve.size for curve in curves)
            if common_length < 2:
                raise ValueError(
                    "Combined error curves require at least one rollout step."
                )
            aligned = [curve[:common_length] for curve in curves]
            split_curves[label][metric] = aligned
            macro_curves[label][metric] = np.mean(
                np.stack(aligned, axis=0), axis=0
            )
    return macro_curves, split_curves


def standard_average_split_curves(
    metrics_by_split: Mapping[
        str, Mapping[str, Sequence[TrajectoryMetrics]]
    ],
    selected_metrics: Sequence[str],
) -> Tuple[
    Dict[str, Dict[str, np.ndarray]],
    Dict[str, Dict[str, List[np.ndarray]]],
]:
    """Pool all selected topology-trajectory curves with equal weight."""
    if not metrics_by_split:
        raise ValueError("At least one split is required for an average.")
    split_names = list(metrics_by_split)
    labels = list(metrics_by_split[split_names[0]])
    if not labels:
        raise ValueError("At least one model is required for an average.")
    expected_labels = set(labels)
    for split, metrics_by_model in metrics_by_split.items():
        if set(metrics_by_model) != expected_labels:
            raise ValueError(
                f"Split {split!r} does not contain the same model labels."
            )

    pooled_curves: Dict[str, Dict[str, np.ndarray]] = {
        label: {} for label in labels
    }
    split_curves: Dict[str, Dict[str, List[np.ndarray]]] = {
        label: {} for label in labels
    }
    for label in labels:
        for metric in selected_metrics:
            attribute = ERROR_CURVE_METRIC_ATTRIBUTE[metric]
            curves_by_split: List[List[np.ndarray]] = []
            for split in split_names:
                curves = [
                    np.asarray(getattr(item, attribute), dtype=np.float64)
                    for item in metrics_by_split[split][label]
                ]
                if not curves:
                    raise ValueError(
                        f"Split {split!r}, model {label!r} has no metrics."
                    )
                curves_by_split.append(curves)

            common_length = min(
                curve.size
                for split_values in curves_by_split
                for curve in split_values
            )
            if common_length < 2:
                raise ValueError(
                    "Combined error curves require at least one rollout step."
                )
            aligned_by_split = [
                [curve[:common_length] for curve in split_values]
                for split_values in curves_by_split
            ]
            split_curves[label][metric] = [
                np.mean(np.stack(split_values, axis=0), axis=0)
                for split_values in aligned_by_split
            ]
            pooled_curves[label][metric] = np.mean(
                np.stack(
                    [
                        curve
                        for split_values in aligned_by_split
                        for curve in split_values
                    ],
                    axis=0,
                ),
                axis=0,
            )
    return pooled_curves, split_curves


def plot_combined_error_curves(
    *,
    combined_curves: Mapping[str, Mapping[str, np.ndarray]],
    split_curves: Mapping[str, Mapping[str, Sequence[np.ndarray]]],
    selected_metrics: Sequence[str],
    styles: Mapping[str, Mapping[str, object]],
    output_path: Path,
    dpi: int,
    smooth_window: int,
    log_y: bool,
    title: str,
    mean_only: bool = False,
    spread: str = ERROR_CURVE_SPREAD_NONE,
    spread_samples: Optional[
        Mapping[str, Mapping[str, Tuple[np.ndarray, np.ndarray]]]
    ] = None,
) -> None:
    """Plot a combined mean, optionally with faint individual split means."""
    if not selected_metrics:
        raise ValueError("At least one metric panel is required.")
    if not combined_curves:
        raise ValueError("At least one model is required.")

    panel_count = len(selected_metrics)
    if panel_count == 1:
        n_rows, n_columns, figure_size = 1, 1, (8.0, 4.8)
    elif panel_count == 2:
        n_rows, n_columns, figure_size = 1, 2, (15.5, 4.8)
    else:
        n_rows, n_columns, figure_size = 2, 2, (15.5, 8.3)

    figure, axes_object = plt.subplots(
        n_rows,
        n_columns,
        figsize=figure_size,
        constrained_layout=True,
        squeeze=False,
    )
    axes = list(np.asarray(axes_object, dtype=object).reshape(-1))
    for unused_axis in axes[panel_count:]:
        unused_axis.set_visible(False)

    labels = list(combined_curves)
    legend_axis = (
        axes[list(selected_metrics).index(ERROR_CURVE_METRIC_SSIM)]
        if ERROR_CURVE_METRIC_SSIM in selected_metrics
        else axes[panel_count - 1]
    )

    for panel_index, metric in enumerate(selected_metrics):
        axis = axes[panel_index]
        plotted_means: List[np.ndarray] = []
        for label in labels:
            mean_curve = centered_moving_average(
                np.asarray(combined_curves[label][metric], dtype=np.float64)[1:],
                smooth_window,
            )
            plotted_means.append(mean_curve)
            rollout_steps = np.arange(1, mean_curve.size + 1)

            if spread != ERROR_CURVE_SPREAD_NONE:
                if spread_samples is None:
                    raise ValueError(
                        "Combined spread bands require trajectory samples."
                    )
                raw_curves, weights = spread_samples[label][metric]
                sample_curves = smooth_curves(
                    np.asarray(raw_curves, dtype=np.float64)[:, 1:],
                    smooth_window,
                )
                lower, upper = error_curve_spread_bounds(
                    sample_curves,
                    spread,
                    weights=weights,
                    center=mean_curve,
                )
                lower, upper = clip_spread_bounds(metric, lower, upper)
                axis.fill_between(
                    rollout_steps,
                    lower,
                    upper,
                    color=styles[label]["color"],
                    alpha=0.14,
                    linewidth=0.0,
                )

            if not mean_only:
                for split_curve in split_curves[label][metric]:
                    background = centered_moving_average(
                        np.asarray(split_curve, dtype=np.float64)[1:],
                        smooth_window,
                    )
                    axis.plot(
                        rollout_steps,
                        background,
                        color=styles[label]["color"],
                        linestyle=styles[label]["linestyle"],
                        linewidth=0.8,
                        alpha=0.18,
                    )
            axis.plot(
                rollout_steps,
                mean_curve,
                label=label,
                **styles[label],
            )

        if n_rows == 1 or panel_index >= (n_rows - 1) * n_columns:
            axis.set_xlabel("Rollout step", fontsize=20)
        axis.set_ylabel(ERROR_CURVE_METRIC_DISPLAY[metric], fontsize=20)
        axis.tick_params(axis="both", labelsize=16)
        axis.grid(alpha=0.3)

        if log_y and metric != ERROR_CURVE_METRIC_SSIM:
            finite_positive = np.concatenate(plotted_means)
            finite_positive = finite_positive[
                np.isfinite(finite_positive) & (finite_positive > 0.0)
            ]
            if finite_positive.size:
                axis.set_yscale("log")
        if metric == ERROR_CURVE_METRIC_SSIM:
            finite_ssim = np.concatenate(plotted_means)
            finite_ssim = finite_ssim[np.isfinite(finite_ssim)]
            if finite_ssim.size:
                minimum = max(-1.0, float(np.min(finite_ssim)))
                span = max(1.0 - minimum, 1.0e-3)
                axis.set_ylim(max(-1.0, minimum - 0.05 * span), 1.0)

    legend_axis.legend(
        loc="lower left" if ERROR_CURVE_METRIC_SSIM in selected_metrics else "best",
        fontsize=16,
    )
    figure.suptitle(title, fontsize=20)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for Figure-6-style comparisons."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot LWR traffic rollout-error curves in the style of Figure 6 "
            "of 7_6_26_scirep_v2.pdf."
        )
    )
    parser.add_argument(
        "--model",
        action="append",
        type=parse_model_spec,
        required=True,
        metavar="LABEL=PATH",
        help="Model label and run/rollout path. Repeat once per model.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Dataset directory; inferred from rollout summaries by default.",
    )
    parser.add_argument("--splits", nargs="+", default=None)
    parser.add_argument("--topologies", nargs="+", type=int, default=None)
    parser.add_argument("--trajectories", nargs="+", type=int, default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument(
        "--error-curves-metrics",
        nargs="+",
        default=None,
        metavar="METRIC",
        help=(
            "Panels to plot, in order. Accepts space- or comma-separated "
            "values from {rmse, rel_l2, l1, ssim}, or all."
        ),
    )
    parser.add_argument(
        "--error-curves-smooth-window",
        type=int,
        default=9,
        help=(
            "Positive odd width of the centered moving average; default 9 "
            "matches Figure 6. Use 1 to disable smoothing."
        ),
    )
    parser.add_argument(
        "--error-curves-log-y",
        action="store_true",
        help="Use logarithmic y axes for RMSE, rel_l2, and L1 panels.",
    )
    parser.add_argument(
        "--error-curves-spread",
        choices=ERROR_CURVE_SPREAD_CHOICES,
        default=ERROR_CURVE_SPREAD_NONE,
        help=(
            "Shaded dispersion around error curves: 'none' (default), "
            "'std' for mean plus/minus one population standard deviation, "
            "or 'iqr' for the 25th--75th percentile band."
        ),
    )
    parser.add_argument(
        "--combined-average",
        "--combined-averaging",
        choices=("macro", "standard"),
        default="macro",
        help=(
            "Averaging for the combined-test figure: 'macro' averages "
            "trajectories within topology, topologies within split, then "
            "splits equally; 'standard' pools all selected "
            "topology-trajectory pairs equally. Default: macro."
        ),
    )
    parser.add_argument(
        "--combined-mean-only",
        action="store_true",
        help=(
            "Hide individual test-split curves in the combined-test figure, "
            "leaving one combined mean curve per model. Any requested spread "
            "bands remain visible. Default: show individual split curves."
        ),
    )
    trajectory_group = parser.add_mutually_exclusive_group()
    trajectory_group.add_argument(
        "--error-curves-mean-only",
        dest="error_curves_mean_only",
        action="store_true",
        help=(
            "Show only topology-balanced mean curves in per-split figures "
            "(the default)."
        ),
    )
    trajectory_group.add_argument(
        "--error-curves-show-trajectories",
        dest="error_curves_mean_only",
        action="store_false",
        help="Also show faint individual trajectory curves in per-split figures.",
    )
    parser.set_defaults(error_curves_mean_only=True)
    parser.add_argument(
        "--curve-color",
        nargs="+",
        default=None,
        metavar="LABEL=COLOR",
        help=(
            "Override Figure 6 colors by displayed label or model family, "
            "for example FluxGraphNet=black or sageconv='#9467bd'."
        ),
    )
    parser.add_argument(
        "--figure-title",
        default=None,
        help=(
            "Optional title template. {split} and {split_title} placeholders "
            "are supported."
        ),
    )
    parser.add_argument(
        "--ssim-window-size", type=int, default=SSIM_WINDOW_SIZE
    )
    parser.add_argument(
        "--ssim-sigma-cells", type=float, default=SSIM_SIGMA_CELLS
    )
    parser.add_argument(
        "--ssim-reference-spacing",
        type=float,
        default=None,
        help=(
            "Physical reference spacing for local SSIM. The dataset-defined "
            "spacing is used by default."
        ),
    )
    parser.add_argument("--dpi", type=int, default=170)
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    """Validate numeric and cross-option constraints."""
    specs: List[ModelSpec] = args.model
    if len(specs) < 2:
        raise ValueError("Specify at least two --model arguments.")
    labels = [spec.label for spec in specs]
    if len(set(labels)) != len(labels):
        raise ValueError("Every --model label must be unique.")
    if args.rollout_steps is not None and args.rollout_steps <= 0:
        raise ValueError("--rollout-steps must be positive.")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive.")
    if (
        args.error_curves_smooth_window <= 0
        or args.error_curves_smooth_window % 2 == 0
    ):
        raise ValueError(
            "--error-curves-smooth-window must be a positive odd integer."
        )
    if args.ssim_window_size < 3 or args.ssim_window_size % 2 == 0:
        raise ValueError("--ssim-window-size must be an odd integer >= 3.")
    if (
        not np.isfinite(args.ssim_sigma_cells)
        or args.ssim_sigma_cells <= 0.0
    ):
        raise ValueError("--ssim-sigma-cells must be positive and finite.")
    if args.ssim_reference_spacing is not None and (
        not np.isfinite(args.ssim_reference_spacing)
        or args.ssim_reference_spacing <= 0.0
    ):
        raise ValueError(
            "--ssim-reference-spacing must be positive and finite."
        )


def select_shared_splits(
    files_by_model: Mapping[str, Mapping[Tuple[str, int], Path]],
    requested: Optional[Sequence[str]],
) -> List[str]:
    """Select requested splits or the intersection available to every model."""
    split_sets = {
        label: {split for split, _ in files}
        for label, files in files_by_model.items()
    }
    shared = set.intersection(*split_sets.values())
    if requested is None:
        selected = sorted(shared)
        if any(values != shared for values in split_sets.values()):
            print(
                "warning: comparing only splits shared by every model: "
                + ", ".join(selected)
            )
        return selected

    selected = list(dict.fromkeys(str(split) for split in requested))
    missing = {
        label: [split for split in selected if split not in available]
        for label, available in split_sets.items()
    }
    missing = {label: values for label, values in missing.items() if values}
    if missing:
        details = "; ".join(
            f"{label}: {values}" for label, values in missing.items()
        )
        raise ValueError(
            f"Requested splits are not available in every model ({details})."
        )
    return selected


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Generate one Figure-6-style error-curve figure for each split."""
    args = parse_args(argv)
    validate_args(args)
    selected_metrics = resolve_error_curve_metrics(
        args.error_curves_metrics
    )
    color_overrides = parse_curve_color_overrides(args.curve_color)

    specs: List[ModelSpec] = args.model
    labels = [spec.label for spec in specs]
    files_by_model = {
        spec.label: discover_rollout_files(spec.rollout_root)
        for spec in specs
    }
    selected_splits = select_shared_splits(files_by_model, args.splits)
    if not selected_splits:
        raise ValueError("The models have no rollout splits in common.")

    topology_ids_by_split: Dict[str, List[int]] = {}
    for split in selected_splits:
        topologies_by_label = {
            label: [
                topology_id
                for candidate_split, topology_id in files
                if candidate_split == split
            ]
            for label, files in files_by_model.items()
        }
        topology_ids_by_split[split] = common_values(
            topologies_by_label,
            requested=args.topologies,
            description=f"topologies for {split}",
        )

    dataset_root = infer_dataset_root(specs, explicit_path=args.data)
    topology_paths = topology_paths_from_manifest(dataset_root)
    selected_topology_ids = sorted(
        {
            topology_id
            for split_ids in topology_ids_by_split.values()
            for topology_id in split_ids
        }
    )
    missing_metadata = [
        topology_id
        for topology_id in selected_topology_ids
        if topology_id not in topology_paths
    ]
    if missing_metadata:
        raise KeyError(
            f"Dataset {dataset_root} has no metadata for topology IDs "
            f"{missing_metadata}."
        )
    metadata_by_topology = {
        topology_id: load_topology_metric_metadata(
            topology_paths[topology_id],
            expected_topology_id=topology_id,
        )
        for topology_id in selected_topology_ids
    }
    (
        ssim_reference_spacing,
        ssim_reference_spacing_source,
    ) = resolve_ssim_reference_spacing(
        dataset_root,
        explicit_spacing=args.ssim_reference_spacing,
        topology_paths=topology_paths,
    )

    styles = figure6_model_styles(
        labels, color_overrides=color_overrides
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_summaries: Dict[str, object] = {}
    metrics_by_split: Dict[
        str, Dict[str, List[TrajectoryMetrics]]
    ] = {}

    for split in selected_splits:
        topology_ids = topology_ids_by_split[split]
        rollouts_by_topology: Dict[int, Dict[str, RolloutData]] = {}
        available_times: Dict[int, int] = {}
        for topology_id in topology_ids:
            rollouts = {
                label: load_rollout(
                    files_by_model[label][(split, topology_id)]
                )
                for label in labels
            }
            rollouts_by_topology[topology_id] = rollouts
            available_times[topology_id] = validate_common_rollouts(
                rollouts, rollout_steps=args.rollout_steps
            )

        common_num_times = min(available_times.values())
        if len(set(available_times.values())) > 1:
            print(
                f"warning: {split} topologies have different rollout lengths; "
                f"plotting their first {common_num_times - 1} shared steps."
            )

        metrics_by_model: Dict[str, List[TrajectoryMetrics]] = {
            label: [] for label in labels
        }
        trajectory_count = 0
        dt_by_topology: Dict[str, float] = {}
        physical_horizon_by_topology: Dict[str, float] = {}

        for topology_id in topology_ids:
            rollouts = rollouts_by_topology[topology_id]
            reference = rollouts[labels[0]]
            topology_key = str(topology_id)
            dt_by_topology[topology_key] = reference.dt
            physical_horizon_by_topology[topology_key] = (
                (common_num_times - 1) * reference.dt
            )
            trajectory_ids = common_values(
                {
                    label: rollout.trajectory_ids
                    for label, rollout in rollouts.items()
                },
                requested=args.trajectories,
                description=(
                    f"trajectories for {split}/topology_{topology_id:04d}"
                ),
            )

            for trajectory_id in trajectory_ids:
                reference_index = trajectory_index(reference, trajectory_id)
                reference_truth = reference.ground_truth[
                    reference_index, :common_num_times
                ]
                for label in labels:
                    rollout = rollouts[label]
                    index = trajectory_index(rollout, trajectory_id)
                    truth = rollout.ground_truth[index, :common_num_times]
                    if not np.allclose(
                        truth,
                        reference_truth,
                        rtol=1.0e-6,
                        atol=1.0e-7,
                    ):
                        maximum_difference = float(
                            np.max(np.abs(truth - reference_truth))
                        )
                        raise ValueError(
                            f"Ground truth differs between {labels[0]} and "
                            f"{label} for {split}, topology {topology_id}, "
                            f"trajectory {trajectory_id} (max difference "
                            f"{maximum_difference:.3e})."
                        )
                    metrics_by_model[label].append(
                        compute_metrics(
                            rollout,
                            trajectory_id,
                            num_times=common_num_times,
                            metadata=metadata_by_topology[topology_id],
                            ssim_reference_spacing=ssim_reference_spacing,
                            ssim_window_size=args.ssim_window_size,
                            ssim_sigma_cells=args.ssim_sigma_cells,
                        )
                    )
                trajectory_count += 1

        output_path = args.output_dir / split / "error_curves_compare.png"
        title = resolve_figure_title(args.figure_title, split)
        plot_figure6_error_curves(
            split=split,
            metrics_by_model=metrics_by_model,
            selected_metrics=selected_metrics,
            styles=styles,
            output_path=output_path,
            dpi=args.dpi,
            smooth_window=args.error_curves_smooth_window,
            mean_only=args.error_curves_mean_only,
            log_y=args.error_curves_log_y,
            title=title,
            spread=args.error_curves_spread,
        )

        final_metric_means = {
            label: {
                metric: float(
                    topology_balanced_mean_curve(
                        metrics_by_model[label],
                        ERROR_CURVE_METRIC_ATTRIBUTE[metric],
                    )[-1]
                )
                for metric in selected_metrics
            }
            for label in labels
        }
        split_summaries[split] = {
            "figure": str(output_path),
            "title": title,
            "topology_ids": topology_ids,
            "num_topology_trajectory_pairs": trajectory_count,
            "rollout_steps": common_num_times - 1,
            "dt_by_topology": dt_by_topology,
            "physical_horizon_by_topology": physical_horizon_by_topology,
            "final_unsmoothed_topology_balanced_means": final_metric_means,
        }
        print(
            f"[{split}] wrote {output_path} using {len(topology_ids)} "
            f"topologies and {trajectory_count} topology-trajectory pairs"
        )
        metrics_by_split[split] = metrics_by_model

    combined_summary: Optional[Dict[str, object]] = None
    if len(metrics_by_split) >= 2:
        if args.combined_average == "macro":
            combined_curves, individual_split_curves = (
                macro_average_split_curves(
                    metrics_by_split, selected_metrics
                )
            )
            combined_title = (
                "1D LWR Traffic: Rollout Errors"
            )
            aggregation_description = "equal_across_selected_splits"
            background_description = (
                "individual_topology_balanced_split_means"
            )
        else:
            combined_curves, individual_split_curves = (
                standard_average_split_curves(
                    metrics_by_split, selected_metrics
                )
            )
            combined_title = (
                "1D LWR Traffic: Test Rollout Errors"
            )
            aggregation_description = (
                "equal_across_all_selected_topology_trajectory_pairs"
            )
            background_description = (
                "individual_pooled_trajectory_split_means"
            )
        combined_spread_samples = None
        if args.error_curves_spread != ERROR_CURVE_SPREAD_NONE:
            combined_spread_samples = combined_spread_curve_samples(
                metrics_by_split,
                selected_metrics,
                args.combined_average,
            )
        combined_output_path = (
            args.output_dir / "combined_test" / "error_curves_compare.png"
        )
        plot_combined_error_curves(
            combined_curves=combined_curves,
            split_curves=individual_split_curves,
            selected_metrics=selected_metrics,
            styles=styles,
            output_path=combined_output_path,
            dpi=args.dpi,
            smooth_window=args.error_curves_smooth_window,
            log_y=args.error_curves_log_y,
            title=combined_title,
            mean_only=args.combined_mean_only,
            spread=args.error_curves_spread,
            spread_samples=combined_spread_samples,
        )
        combined_rollout_steps = min(
            combined_curves[label][metric].size - 1
            for label in labels
            for metric in selected_metrics
        )
        combined_summary = {
            "figure": str(combined_output_path),
            "title": combined_title,
            "included_splits": list(metrics_by_split),
            "rollout_steps": combined_rollout_steps,
            "averaging": args.combined_average,
            "spread": args.error_curves_spread,
            "aggregation": aggregation_description,
            "background_curves": (
                "none" if args.combined_mean_only else background_description
            ),
            "final_unsmoothed_combined_means": {
                label: {
                    metric: float(combined_curves[label][metric][-1])
                    for metric in selected_metrics
                }
                for label in labels
            },
        }
        print(
            "[combined_test] wrote "
            f"{combined_output_path} using {args.combined_average} "
            "averaging"
        )

    summary = {
        "source_figure": "Figure 6 of 7_6_26_scirep_v2.pdf",
        "models": {spec.label: str(spec.rollout_root) for spec in specs},
        "dataset": str(dataset_root),
        "error_curves_metrics": selected_metrics,
        "error_curves_smooth_window": args.error_curves_smooth_window,
        "error_curves_mean_only": args.error_curves_mean_only,
        "error_curves_spread": args.error_curves_spread,
        "error_curves_spread_definition": {
            "std": "weighted mean plus/minus one population standard deviation",
            "iqr": "weighted 25th--75th percentiles",
        },
        "horizontal_axis": "rollout_step",
        "physical_time_alignment": "not_applied",
        "relative_l2": "cell_length_weighted",
        "split_aggregation": {
            "trajectory_aggregation": "equal_within_topology",
            "topology_aggregation": "equal_across_topologies",
        },
        "combined_average": args.combined_average,
        "combined_mean_only": args.combined_mean_only,
        "ssim": {
            "type": "roadwise_local_gaussian",
            "data_range": "rho_max",
            "window_size_reference_cells": args.ssim_window_size,
            "sigma_reference_cells": args.ssim_sigma_cells,
            "reference_spacing": ssim_reference_spacing,
            "reference_spacing_source": ssim_reference_spacing_source,
            "k1": SSIM_K1,
            "k2": SSIM_K2,
        },
        "splits": split_summaries,
        "combined_test_summary": combined_summary,
    }
    summary_path = args.output_dir / "error_curves_summary.json"
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(f"wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
