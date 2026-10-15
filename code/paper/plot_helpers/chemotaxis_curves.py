# Rendering-only extraction; numerical/rendering function bodies unchanged.
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Sequence, Mapping, Optional, Iterable
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import ScalarFormatter


# Source: code/chemotactic_cell_migration/scripts/plot_model_comparisons.py (save_figure)
def save_figure(
    figure: plt.Figure,
    base_path: Path,
    *,
    formats: Sequence[str],
    dpi: int,
) -> List[Path]:
    """Save a figure in every requested format and return written paths."""
    base_path.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for extension in formats:
        output_path = base_path.with_suffix(f".{extension}")
        figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
        written.append(output_path)
    plt.close(figure)
    return written


# Source: code/chemotactic_cell_migration/scripts/plot_model_comparisons_paper.py (ERROR_CURVE_METRIC_SSIM)
ERROR_CURVE_METRIC_SSIM = "ssim"


# Source: code/chemotactic_cell_migration/scripts/plot_model_comparisons_paper.py (centered_moving_average)
def centered_moving_average(curve: np.ndarray, window: int) -> np.ndarray:
    """Return a NaN-aware centered average with truncated endpoint windows."""
    values = np.asarray(curve, dtype=np.float64).reshape(-1)
    if window <= 0 or window % 2 == 0:
        raise ValueError("The smoothing window must be a positive odd integer.")
    if window == 1 or values.size == 0:
        return values.copy()
    radius = window // 2
    smoothed = np.full(values.shape, np.nan, dtype=np.float64)
    for index in range(values.size):
        local = values[
            max(0, index - radius) : min(values.size, index + radius + 1)
        ]
        finite = local[np.isfinite(local)]
        if finite.size:
            smoothed[index] = float(np.mean(finite))
    return smoothed


# Source: code/chemotactic_cell_migration/scripts/plot_model_comparisons_paper.py (smooth_curves)
def smooth_curves(curves: np.ndarray, window: int) -> np.ndarray:
    """Apply the centered moving average to a curve matrix."""
    values = np.asarray(curves, dtype=np.float64)
    if values.ndim == 1:
        return centered_moving_average(values, window)
    if values.ndim == 2:
        return np.stack(
            [centered_moving_average(row, window) for row in values], axis=0
        )
    raise ValueError(f"Expected one- or two-dimensional curves, got {values.shape}.")


# Source: code/chemotactic_cell_migration/scripts/plot_model_comparisons_paper.py (FIGURE6_FAMILY_STYLES)
FIGURE6_FAMILY_STYLES = {
    "fluxgraphnet": {"color": "tab:blue", "linestyle": "-"},
    "fvupwind": {"color": "tab:green", "linestyle": "--"},
    "fvmuscl": {"color": "black", "linestyle": "--"},
    "meshgraphnet": {"color": "magenta", "linestyle": "-"},
    "sageconv": {"color": "tab:orange", "linestyle": "-"},
    "persistence": {"color": "0.35", "linestyle": ":"},
}


# Source: code/chemotactic_cell_migration/scripts/plot_model_comparisons_paper.py (ERROR_CURVE_METRIC_RMSE)
ERROR_CURVE_METRIC_RMSE = "rmse"


# Source: code/chemotactic_cell_migration/scripts/plot_model_comparisons_paper.py (ERROR_CURVE_METRIC_REL_L2)
ERROR_CURVE_METRIC_REL_L2 = "rel_l2"


# Source: code/chemotactic_cell_migration/scripts/plot_model_comparisons_paper.py (ERROR_CURVE_METRIC_L1)
ERROR_CURVE_METRIC_L1 = "l1"


# Source: code/chemotactic_cell_migration/scripts/plot_model_comparisons_paper.py (ERROR_CURVE_METRIC_DISPLAY)
ERROR_CURVE_METRIC_DISPLAY = {
    ERROR_CURVE_METRIC_RMSE: "RMSE",
    ERROR_CURVE_METRIC_REL_L2: r"rel-$L_2$",
    ERROR_CURVE_METRIC_L1: "L1",
    ERROR_CURVE_METRIC_SSIM: "Local SSIM",
}


# Source: code/chemotactic_cell_migration/scripts/plot_model_comparisons.py (TrajectoryMetrics)
@dataclass(frozen=True)
class TrajectoryMetrics:
    """Time-resolved and scalar diagnostics for one trajectory."""

    mae_by_time: np.ndarray
    rmse_by_time: np.ndarray
    bias_by_time: np.ndarray
    mass_error_by_time: np.ndarray
    negative_fraction_by_time: np.ndarray
    persistence_mae_by_time: np.ndarray
    persistence_rmse_by_time: np.ndarray
    area_weighted_relative_l2_by_time: Optional[np.ndarray]
    persistence_area_weighted_relative_l2_by_time: Optional[np.ndarray]
    local_ssim_by_time: Optional[np.ndarray]
    persistence_local_ssim_by_time: Optional[np.ndarray]
    rollout_mae: float
    rollout_rmse: float
    persistence_rollout_rmse: float
    persistence_improvement_fraction: float
    final_step_mae: float
    final_step_rmse: float
    mean_signed_error: float
    relative_l2_error: float
    area_weighted_relative_l2_error: Optional[float]
    persistence_area_weighted_relative_l2_error: Optional[float]
    mean_rollout_ssim: Optional[float]
    final_step_ssim: Optional[float]
    r_squared: float
    pearson_correlation: float
    mean_absolute_mass_error: float
    max_absolute_mass_error: float
    negative_density_fraction: float
    max_negative_fraction: float


# Source: code/chemotactic_cell_migration/scripts/plot_model_comparisons_paper.py (_metric_matrix)
def _metric_matrix(
    metrics: Sequence[TrajectoryMetrics], attribute: str
) -> np.ndarray:
    curves = []
    for item in metrics:
        value = getattr(item, attribute)
        if value is None:
            raise ValueError(f"Metric {attribute} was not computed.")
        curves.append(np.asarray(value, dtype=np.float64))
    return np.stack(curves, axis=0)


# Source: code/chemotactic_cell_migration/scripts/plot_model_comparisons_paper.py (ERROR_CURVE_METRIC_ATTRIBUTE)
ERROR_CURVE_METRIC_ATTRIBUTE = {
    ERROR_CURVE_METRIC_RMSE: "rmse_by_time",
    ERROR_CURVE_METRIC_REL_L2: "area_weighted_relative_l2_by_time",
    ERROR_CURVE_METRIC_L1: "mae_by_time",
    ERROR_CURVE_METRIC_SSIM: "local_ssim_by_time",
}


# Source: code/chemotactic_cell_migration/scripts/plot_model_comparisons_paper.py (PERSISTENCE_METRIC_ATTRIBUTE)
PERSISTENCE_METRIC_ATTRIBUTE = {
    ERROR_CURVE_METRIC_RMSE: "persistence_rmse_by_time",
    ERROR_CURVE_METRIC_REL_L2: (
        "persistence_area_weighted_relative_l2_by_time"
    ),
    ERROR_CURVE_METRIC_L1: "persistence_mae_by_time",
    ERROR_CURVE_METRIC_SSIM: "persistence_local_ssim_by_time",
}


# Source: code/chemotactic_cell_migration/scripts/plot_model_comparisons_paper.py (plot_figure6_error_curves)
def plot_figure6_error_curves(
    *,
    metrics_by_model: Mapping[str, Sequence[TrajectoryMetrics]],
    times: np.ndarray,
    selected_metrics: Sequence[str],
    styles: Mapping[str, Mapping[str, object]],
    output_base: Path,
    formats: Sequence[str],
    dpi: int,
    smooth_window: int,
    show_trajectories: bool,
    spread: str,
    log_y: bool,
    title: str,
    show_persistence: bool,
) -> List[Path]:
    """Plot selectable metrics using the visual language of paper Figure 6."""
    if not selected_metrics or not metrics_by_model:
        raise ValueError("At least one metric and one model are required.")
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
    plot_times = np.asarray(times, dtype=np.float64)[1:]
    for panel_index, metric in enumerate(selected_metrics):
        axis = axes[panel_index]
        plotted_means: List[np.ndarray] = []
        for label in labels:
            raw_curves = _metric_matrix(
                metrics_by_model[label], ERROR_CURVE_METRIC_ATTRIBUTE[metric]
            )[:, 1:]
            display_curves = smooth_curves(raw_curves, smooth_window)
            mean_curve = centered_moving_average(
                np.mean(raw_curves, axis=0), smooth_window
            )
            plotted_means.append(mean_curve)
            if show_trajectories:
                for curve in display_curves:
                    axis.plot(
                        plot_times,
                        curve,
                        color=styles[label]["color"],
                        linestyle=styles[label]["linestyle"],
                        linewidth=0.75,
                        alpha=0.16,
                    )
            if spread == "iqr":
                lower, upper = np.quantile(display_curves, (0.25, 0.75), axis=0)
                axis.fill_between(
                    plot_times,
                    lower,
                    upper,
                    color=styles[label]["color"],
                    alpha=0.12,
                    linewidth=0.0,
                )
            elif spread == "std":
                standard_deviation = np.std(display_curves, axis=0)
                axis.fill_between(
                    plot_times,
                    mean_curve - standard_deviation,
                    mean_curve + standard_deviation,
                    color=styles[label]["color"],
                    alpha=0.12,
                    linewidth=0.0,
                )
            axis.plot(plot_times, mean_curve, label=label, **styles[label])

        if show_persistence:
            first_metrics = next(iter(metrics_by_model.values()))
            raw_persistence = _metric_matrix(
                first_metrics, PERSISTENCE_METRIC_ATTRIBUTE[metric]
            )[:, 1:]
            persistence_mean = centered_moving_average(
                np.mean(raw_persistence, axis=0), smooth_window
            )
            plotted_means.append(persistence_mean)
            persistence_style = FIGURE6_FAMILY_STYLES["persistence"]
            axis.plot(
                plot_times,
                persistence_mean,
                label="Persistence",
                linewidth=2.0,
                **persistence_style,
            )

        if n_rows == 1 or panel_index >= (n_rows - 1) * n_columns:
            axis.set_xlabel("Time", fontsize=20)
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
    return save_figure(figure, output_base, formats=formats, dpi=dpi)


# Source: code/chemotactic_cell_migration/scripts/plot_model_comparisons_paper.py (_normalized_token)
def _normalized_token(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


# Source: code/chemotactic_cell_migration/scripts/plot_model_comparisons_paper.py (model_family)
def model_family(label: str) -> Optional[str]:
    """Recognize model families carrying fixed colors in Figure 6."""
    key = _normalized_token(label)
    if "fluxgraphnet" in key:
        return "fluxgraphnet"
    if "fvupwind" in key or "godunov" in key:
        return "fvupwind"
    if "fvmuscl" in key:
        return "fvmuscl"
    if "meshgraphnet" in key:
        return "meshgraphnet"
    if "sageconv" in key:
        return "sageconv"
    if "persistence" in key:
        return "persistence"
    return None


# Source: code/chemotactic_cell_migration/scripts/plot_model_comparisons_paper.py (FALLBACK_COLORS)
FALLBACK_COLORS = (
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:olive",
    "tab:cyan",
    "black",
)


# Source: code/chemotactic_cell_migration/scripts/plot_model_comparisons_paper.py (figure6_model_styles)
def figure6_model_styles(
    labels: Sequence[str],
    *,
    color_overrides: Optional[Mapping[str, str]] = None,
) -> Dict[str, Dict[str, object]]:
    """Assign semantic Figure 6 colors and stable fallbacks."""
    overrides = dict(color_overrides or {})
    used_families = set()
    fallback_index = 0
    styles: Dict[str, Dict[str, object]] = {}
    for label in labels:
        family = model_family(label)
        if family is not None and family not in used_families:
            style = dict(FIGURE6_FAMILY_STYLES[family])
            used_families.add(family)
        else:
            style = {
                "color": FALLBACK_COLORS[fallback_index % len(FALLBACK_COLORS)],
                "linestyle": "-",
            }
            fallback_index += 1
        override = overrides.get(_normalized_token(label))
        if override is None and family is not None:
            override = overrides.get(family)
        if override is not None:
            style["color"] = override
        style["linewidth"] = 2.0
        styles[label] = style
    return styles

