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


# Source: code/1D_advection/scripts/plot_paper_spacetime_compare.py (SampleRollout)
@dataclass
class SampleRollout:
    sample_id: int
    true_traj: np.ndarray
    gnn_traj: np.ndarray
    baseline_trajs: Dict[str, np.ndarray]


# Source: code/1D_advection/scripts/compare_gnn_vs_godunov.py (_replace_nonfinite_with_nan)
def _replace_nonfinite_with_nan(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float64).copy()
    out[~np.isfinite(out)] = np.nan
    return out


# Source: code/1D_advection/scripts/plot_paper_spacetime_compare.py (_finite_minmax)
def _finite_minmax(arr: np.ndarray) -> Tuple[float, float]:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return -1.0, 1.0
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return -1.0, 1.0
    if hi <= lo:
        eps = max(1e-6, abs(lo) * 1e-6)
        return lo - eps, hi + eps
    return lo, hi


# Source: code/1D_advection/scripts/plot_paper_spacetime_compare.py (_resolve_imshow_sampling_kwargs)
def _resolve_imshow_sampling_kwargs(
    *,
    true_dx_dt_aspect: bool,
    imshow_interpolation: str | None,
) -> Dict[str, object]:
    if true_dx_dt_aspect:
        return {"interpolation": "none", "resample": False}
    if imshow_interpolation is None:
        return {}
    return {"interpolation": imshow_interpolation}


# Source: code/1D_advection/scripts/plot_paper_spacetime_compare.py (_auto_page_height)
def _auto_page_height(num_panels: int) -> float:
    return max(4.0, 2.35 * float(num_panels) + 0.6)


# Source: code/1D_advection/scripts/plot_paper_spacetime_compare.py (_finite_max)
def _finite_max(arrays: List[np.ndarray], *, fallback: float) -> float:
    vals = []
    for arr in arrays:
        finite = arr[np.isfinite(arr)]
        if finite.size > 0:
            vals.append(float(np.max(finite)))
    if len(vals) == 0:
        return float(fallback)
    return max(max(vals), float(fallback))


# Source: code/1D_advection/scripts/plot_paper_spacetime_compare.py (_make_vertical_spacetime_page)
def _make_vertical_spacetime_page(
    *,
    sample: SampleRollout,
    pos: np.ndarray,
    dt: float,
    gnn_label: str,
    baseline_labels: Dict[str, str],
    spacetime_delta_plots: List[str] | None,
    page_width: float,
    page_height: float | None,
    imshow_interpolation: str | None,
    true_dx_dt_aspect: bool,
):
    import matplotlib.pyplot as plt

    true_plot = _replace_nonfinite_with_nan(sample.true_traj)
    gnn_plot = _replace_nonfinite_with_nan(sample.gnn_traj)
    baseline_plots = [
        (scheme, baseline_labels[scheme], _replace_nonfinite_with_nan(arr))
        for scheme, arr in sample.baseline_trajs.items()
    ]

    err_gnn = np.abs(gnn_plot - true_plot)
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

    t_axis = np.arange(true_plot.shape[0], dtype=np.float64) * float(dt)
    extent = [
        float(pos.min()),
        float(pos.max()),
        float(t_axis.max()),
        float(t_axis.min()),
    ]
    state_min, state_max = _finite_minmax(true_plot)

    panels: List[Tuple[str, str, np.ndarray]] = [
        ("state", "Exact", true_plot),
        ("state", f"{gnn_label} Predicted", gnn_plot),
    ]
    for _, label, arr in baseline_plots:
        panels.append(("state", f"{label} Predicted", arr))
    if _show_delta("gnn"):
        panels.append(("err", f"|{gnn_label} - Exact|", err_gnn))
    for scheme, label, err in baseline_errs:
        if _show_delta(scheme):
            panels.append(("err", f"|{label} - Exact|", err))

    fig_height = (
        _auto_page_height(len(panels))
        if page_height is None
        else float(page_height)
    )
    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(float(page_width), fig_height),
        constrained_layout=True,
        squeeze=False,
    )
    axes_col = axes[:, 0]
    imshow_kwargs = _resolve_imshow_sampling_kwargs(
        true_dx_dt_aspect=true_dx_dt_aspect,
        imshow_interpolation=imshow_interpolation,
    )

    for panel_idx, (ax, (kind, title, arr)) in enumerate(zip(axes_col, panels)):
        if kind == "state":
            im = ax.imshow(
                arr,
                aspect="auto",
                extent=extent,
                cmap="viridis",
                vmin=state_min,
                vmax=state_max,
                **imshow_kwargs,
            )
        else:
            err_max = _finite_max([arr], fallback=1e-8)
            im = ax.imshow(
                arr,
                aspect="auto",
                extent=extent,
                cmap="magma",
                vmin=0.0,
                vmax=err_max,
                **imshow_kwargs,
            )
        ax.set_title(title, fontsize=15)
        if panel_idx == len(panels) - 1:
            ax.set_xlabel("Position", fontsize=12)
        else:
            ax.set_xlabel("")
            ax.tick_params(axis="x", labelbottom=False)
        ax.set_ylabel("Time", fontsize=12)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    #fig.suptitle(f"Rollout Sample | id={sample.sample_id}", fontsize=11)
    fig.suptitle(f"1D Linear Advection", fontsize=18)
    return fig

