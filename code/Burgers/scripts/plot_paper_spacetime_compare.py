#!/usr/bin/env python3
"""Build paper-style vertical spacetime comparison PDFs.

This plotting-only companion to ``compare_gnn_vs_godunov.py`` reads a model
checkpoint, evaluates the requested rollout samples directly, and writes one
multi-page PDF. Each page contains one rollout sample, with the comparison
heatmaps stacked in a single column instead of laid out across a row.

Example:

    python scripts/plot_paper_spacetime_compare.py \
      --checkpoint runs/.../checkpoint_best.pt \
      --page-width 6.5 \
      --page-height 10.0
"""

# CLI options:
#   --checkpoint: required path to checkpoint_best.pt or checkpoint_last.pt.
#   --data-dir: optional dataset path override.
#   --use-all-trajectories: treat a WENO-A .pt override as the full test split.
#   --horizon: optional rollout horizon; default follows checkpoint config/full test window.
#   --num-samples: optional limit on selected test samples.
#   --allow-full-test-fallback: use all valid test starts when checkpoint split is missing.
#   --eval-test-trajectories: select one representative start per test trajectory (default).
#   --eval-pair-samples: legacy first-N pair-start selection.
#   --device: device for GNN rollout inference: auto, cpu, cuda, or mps.
#   --out: output PDF path; defaults to <checkpoint-dir>/paper_plots/.
#   --page-width / --page-height: PDF page size in inches.
#   --dpi: rasterization DPI for PDF pages and optional PNG exports.
#   --summary-json: optional summary JSON for display labels.
#   --baseline-schemes: baselines to include; accepts all, none, godunov, muscl, weno3.
#   --spacetime-delta-plots: difference panels to include; accepts all, none,
#     gnn, baselines, or selected baseline schemes/labels.
#   --no-recompute-missing-baselines: skip non-Godunov baseline rows.
#   --gnn-label: override the GNN panel label.
#   --sample-index: zero-based selected-sample index to plot; repeat for multiple pages.
#   --sample-id: saved pair/sample id to plot; repeat for multiple pages.
#   --max-pages: limit the number of written pages after sample selection.
#   --save-png-page / --png-page: also save the selected 1-based PDF page(s) as PNG.
#   --save-png-time-index / --png-time-index: also save page(s) with matching t0 as PNG.
#   --state-cmap / --error-cmap: Matplotlib colormaps for state and error panels.
#   --title-prefix: prefix for the optional per-page title.
#   --no-page-title: hide the per-page title.
#   --hide-colorbars: hide all per-subplot colorbars.
#   --hspace: optional constrained-layout vertical spacing override.

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np

_TMPDIR = Path(os.environ.get("TMPDIR", "/tmp"))
os.environ.setdefault("MPLCONFIGDIR", str(_TMPDIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_TMPDIR))

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

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
DELTA_SOURCE_GNN = "gnn"
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


def _replace_nonfinite_with_nan(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float64).copy()
    out[~np.isfinite(out)] = np.nan
    return out


def _string_list(arr: np.ndarray) -> list[str]:
    out: list[str] = []
    for value in np.asarray(arr).reshape(-1).tolist():
        if isinstance(value, bytes):
            out.append(value.decode("utf-8"))
        else:
            out.append(str(value))
    return out


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def _load_summary(checkpoint: Path, summary_json: Path | None) -> dict[str, Any]:
    candidates: list[Path] = []
    if summary_json is not None:
        candidates.append(summary_json)
    candidates.append(checkpoint.parent / "gnn_vs_godunov_summary.json")
    candidates.append(checkpoint.parent / "summary.json")

    for path in candidates:
        summary = _read_json(path)
        if summary:
            return summary
    return {}


def normalize_baseline_scheme(scheme: str) -> str:
    s = str(scheme).strip().lower()
    if s in BASELINE_SCHEMES:
        return s
    if s in _BASELINE_SCHEME_ALIASES:
        return _BASELINE_SCHEME_ALIASES[s]
    allowed = sorted(set(BASELINE_SCHEMES).union(_BASELINE_SCHEME_ALIASES))
    raise ValueError(
        f"Unknown baseline scheme {scheme!r}. Expected one of: {', '.join(allowed)}."
    )


def _split_requested_schemes(raw: Iterable[str]) -> list[str]:
    items: list[str] = []
    for item in raw:
        for part in str(item).split(","):
            part = part.strip()
            if part:
                items.append(part)
    return items


def _normalized_label_key(value: str) -> str:
    text = str(value).strip().lower().replace("_", " ").replace("-", " ")
    return " ".join(text.split())


def _resolve_baseline_schemes(
    raw_schemes: list[str] | None,
    *,
    available_schemes: list[str],
) -> list[str]:
    if raw_schemes is None:
        raw = list(available_schemes)
    else:
        raw = _split_requested_schemes(raw_schemes)

    lowered = [v.strip().lower() for v in raw]
    if len(lowered) == 0 or "none" in lowered:
        return []

    if "all" in lowered:
        selected = list(available_schemes) if available_schemes else list(BASELINE_SCHEMES)
    else:
        selected = [normalize_baseline_scheme(v) for v in raw]

    out: list[str] = []
    for scheme in selected:
        norm = normalize_baseline_scheme(scheme)
        if norm not in out:
            out.append(norm)
    return out


def _resolve_spacetime_delta_plots(
    raw_plots: list[str] | None,
    *,
    selected_baselines: list[str],
    label_by_scheme: dict[str, str],
    gnn_label: str,
) -> list[str]:
    default_sources = [DELTA_SOURCE_GNN, *selected_baselines]
    if raw_plots is None:
        return default_sources

    raw = _split_requested_schemes(raw_plots)
    if len(raw) == 0:
        return []

    lowered = {_normalized_label_key(v) for v in raw}
    if lowered.intersection({"none", "no", "off", "false"}):
        return []
    if lowered.intersection({"all", "default"}):
        return default_sources

    label_to_scheme = {
        _normalized_label_key(label): scheme
        for scheme, label in label_by_scheme.items()
    }
    gnn_keys = {
        "gnn",
        "model",
        "network",
        _normalized_label_key(gnn_label),
    }

    out: list[str] = []

    def _add(source: str) -> None:
        if source not in out:
            out.append(source)

    for item in raw:
        key = _normalized_label_key(item)
        if key in gnn_keys:
            _add(DELTA_SOURCE_GNN)
            continue
        if key in {"baseline", "baselines"}:
            for scheme in selected_baselines:
                _add(scheme)
            continue
        if key in label_to_scheme:
            scheme = label_to_scheme[key]
        else:
            try:
                scheme = normalize_baseline_scheme(item)
            except ValueError as exc:
                allowed = [
                    "all",
                    "none",
                    "gnn",
                    "baselines",
                    *selected_baselines,
                    *[label_by_scheme.get(s, s) for s in selected_baselines],
                ]
                raise ValueError(
                    "--spacetime-delta-plots must use 'all', 'none', 'gnn', "
                    "'baselines', or one of the selected baseline schemes/labels. "
                    f"Allowed here: {', '.join(repr(v) for v in allowed)}."
                ) from exc
        if scheme not in selected_baselines:
            raise ValueError(
                f"--spacetime-delta-plots requested {item!r}, but that baseline "
                "is not included by --baseline-schemes."
            )
        _add(scheme)
    return out


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


def _finite_volume_step_burgers(
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
        traj[k + 1] = _finite_volume_step_burgers(
            u,
            dt=dt,
            dx=dx,
            scheme=scheme_norm,
        )
    return traj


def _trim_traj(traj: np.ndarray, valid_steps: int | None) -> np.ndarray:
    arr = np.asarray(traj, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected trajectory array with shape [T, N], got {arr.shape}.")

    if valid_steps is None:
        finite_rows = np.any(np.isfinite(arr), axis=1)
        if not np.any(finite_rows):
            return arr[:0]
        last_idx = int(np.nonzero(finite_rows)[0][-1])
    else:
        last_idx = int(np.clip(valid_steps, 0, arr.shape[0] - 1))
    return arr[: last_idx + 1]


def _align_to_length(arr: np.ndarray, length: int) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float64)
    if out.shape[0] >= length:
        return out[:length]
    pad = np.full((length - out.shape[0], out.shape[1]), np.nan, dtype=np.float64)
    return np.concatenate([out, pad], axis=0)


def _infer_dx(x_nodes: np.ndarray) -> float:
    pos = np.asarray(x_nodes, dtype=np.float64).reshape(-1)
    if pos.size < 2:
        return 1.0
    diffs = np.diff(np.sort(pos))
    finite = diffs[np.isfinite(diffs) & (diffs > 0.0)]
    if finite.size == 0:
        return 1.0
    return float(np.median(finite))


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _format_page_title(
    *,
    title_prefix: str,
    sample_index: int,
    sample_id: int | None,
    global_tid: int | None,
    local_tid: int | None,
    time_index: int | None,
) -> str:
    return str(title_prefix)

def _page_png_path(
    *,
    out_pdf: Path,
    page_number: int,
    sample_index: int,
    time_index: int,
    sample_id: int | None,
) -> Path:
    parts = [
        out_pdf.stem,
        f"t{int(time_index)}",
        f"page-{int(page_number):03d}",
        f"sample-index-{int(sample_index)}",
    ]
    if sample_id is not None:
        parts.append(f"sample-id-{int(sample_id)}")
    return out_pdf.with_name("_".join(parts) + ".png")


def _resolve_indices(
    *,
    n_samples: int,
    sample_ids: np.ndarray | None,
    requested_indices: list[int] | None,
    requested_sample_ids: list[int] | None,
    max_pages: int | None,
) -> list[int]:
    out: list[int] = []
    if requested_indices is None and requested_sample_ids is None:
        out = list(range(n_samples))
    if requested_indices is not None:
        for idx in requested_indices:
            i = int(idx)
            if i < 0 or i >= n_samples:
                raise ValueError(
                    f"Requested --sample-index {i} is out of range [0, {n_samples - 1}]."
                )
            if i not in out:
                out.append(i)
    if requested_sample_ids is not None:
        if sample_ids is None:
            raise ValueError("--sample-id requires selected sample ids to be available.")
        id_arr = np.asarray(sample_ids, dtype=np.int64).reshape(-1)
        for sid in requested_sample_ids:
            matches = np.nonzero(id_arr == int(sid))[0]
            if matches.size == 0:
                raise ValueError(f"Requested --sample-id {sid} was not found.")
            i = int(matches[0])
            if i not in out:
                out.append(i)
    if max_pages is not None:
        if int(max_pages) < 1:
            raise ValueError(f"--max-pages must be >= 1, got {max_pages}.")
        out = out[: int(max_pages)]
    return out


def _add_panel_colorbar(fig, mappable, ax, *, label: str | None) -> None:
    cbar = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.03)
    #if label:
    #    cbar.set_label(label, fontsize=8)
    #cbar.ax.tick_params(labelsize=7)
    if hasattr(cbar.ax, "set_box_aspect"):
        cbar.ax.set_box_aspect(None)
    cbar.ax.set_aspect("auto")


def _build_page_figure(
    *,
    sample_index: int,
    sample_id: int | None,
    global_tid: int | None,
    local_tid: int | None,
    time_index: int | None,
    x_nodes: np.ndarray,
    dt: float | None,
    true_arr: np.ndarray,
    gnn_arr: np.ndarray,
    baseline_rows: list[tuple[str, str, np.ndarray]],
    gnn_label: str,
    delta_plot_sources: list[str],
    page_width: float,
    page_height: float,
    state_cmap: str,
    error_cmap: str,
    title_prefix: str,
    show_page_title: bool,
    show_colorbars: bool,
    hspace: float | None,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    true_plot = _replace_nonfinite_with_nan(true_arr)
    gnn_plot = _replace_nonfinite_with_nan(_align_to_length(gnn_arr, true_plot.shape[0]))
    baseline_plots = [
        (
            scheme,
            label,
            _replace_nonfinite_with_nan(_align_to_length(arr, true_plot.shape[0])),
        )
        for scheme, label, arr in baseline_rows
    ]

    err_gnn = np.abs(gnn_plot - true_plot)
    baseline_errs = [
        (scheme, label, np.abs(arr - true_plot))
        for scheme, label, arr in baseline_plots
    ]

    gt_finite = true_plot[np.isfinite(true_plot)]
    if gt_finite.size == 0:
        gt_min, gt_max = -1.0, 1.0
    else:
        gt_min = float(np.min(gt_finite))
        gt_max = float(np.max(gt_finite))
    if (not np.isfinite(gt_min)) or (not np.isfinite(gt_max)) or gt_max <= gt_min:
        center = 0.0 if not np.isfinite(gt_min) else gt_min
        eps = max(1e-6, abs(center) * 1e-6)
        gt_min = center - eps
        gt_max = center + eps

    def _safe_nanmax(arr: np.ndarray) -> float:
        vals = np.asarray(arr, dtype=np.float64)
        finite = vals[np.isfinite(vals)]
        return float(np.max(finite)) if finite.size > 0 else 0.0

    delta_source_set = set(delta_plot_sources)
    panels: list[tuple[str, str, np.ndarray]] = [
        ("state", "Exact", true_plot),
        ("state", f"{gnn_label} Predicted", gnn_plot),
    ]
    if baseline_plots:
        panels.append(("state", f"{baseline_plots[0][1]} Predicted", baseline_plots[0][2]))
    if DELTA_SOURCE_GNN in delta_source_set:
        panels.append(("error", f"|{gnn_label} - Exact|", err_gnn))
    if baseline_errs and baseline_errs[0][0] in delta_source_set:
        panels.append(("error", f"|{baseline_errs[0][1]} - Exact|", baseline_errs[0][2]))
    for (scheme, label, baseline_plot), (_, _, baseline_err) in zip(
        baseline_plots[1:],
        baseline_errs[1:],
    ):
        panels.append(("state", f"{label} Predicted", baseline_plot))
        if scheme in delta_source_set:
            panels.append(("error", f"|{label} - Exact|", baseline_err))

    n_rows = len(panels)
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(float(page_width), float(page_height)),
        constrained_layout=True,
        squeeze=False,
    )
    if hspace is not None:
        fig.set_constrained_layout_pads(hspace=float(hspace))

    pos = np.asarray(x_nodes, dtype=np.float64).reshape(-1)
    if pos.size == 0:
        pos = np.arange(true_plot.shape[1], dtype=np.float64)
    x0 = float(np.nanmin(pos))
    x1 = float(np.nanmax(pos))
    if (not np.isfinite(x0)) or (not np.isfinite(x1)) or x1 <= x0:
        x0 = 0.0
        x1 = float(max(1, true_plot.shape[1] - 1))

    if dt is None:
        t_max = float(max(0, true_plot.shape[0] - 1))
        y_label = "Rollout step"
    else:
        t_max = float(max(0, true_plot.shape[0] - 1)) * float(dt)
        y_label = "Time"
    extent = [x0, x1, t_max, 0.0]

    for row_idx, (kind, title, arr) in enumerate(panels):
        ax = axes[row_idx, 0]
        if kind == "state":
            im = ax.imshow(
                arr,
                aspect="auto",
                extent=extent,
                cmap=state_cmap,
                vmin=gt_min,
                vmax=gt_max,
            )
            cbar_label = "u"
        else:
            err_panel_max = max(_safe_nanmax(arr), 1e-8)
            im = ax.imshow(
                arr,
                aspect="auto",
                extent=extent,
                cmap=error_cmap,
                vmin=0.0,
                vmax=err_panel_max,
            )
            cbar_label = "absolute error"
        if show_colorbars:
            _add_panel_colorbar(fig, im, ax, label=cbar_label)
        ax.set_title(title, fontsize=15, pad=2)
        #ax.set_title("Inviscid Burger's", loc="left", fontsize=9, pad=2)
        ax.set_ylabel(y_label, fontsize=12)
        #ax.tick_params(axis="both", labelsize=7)
        if row_idx == n_rows - 1:
            ax.set_xlabel("Position", fontsize=12)
        else:
            ax.set_xlabel("")
            ax.tick_params(axis="x", which="both", labelbottom=False)
            ax.xaxis.get_offset_text().set_visible(False)

    if show_page_title:
        title = _format_page_title(
            title_prefix=title_prefix,
            sample_index=sample_index,
            sample_id=sample_id,
            global_tid=global_tid,
            local_tid=local_tid,
            time_index=time_index,
        )
        fig.suptitle(title, fontsize=18)

    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a paper-style multi-page PDF from a checkpoint by directly "
            "evaluating GNN and finite-volume rollout trajectories."
        )
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
        "--use-all-trajectories",
        action="store_true",
        help=(
            "For a WENO-A .pt passed with --data-dir, treat that file as the "
            "entire test split and ignore checkpoint test split indices. In the "
            "default trajectory mode, this plots one rollout per trajectory."
        ),
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
            "Optional sample limit. In default trajectory mode, this limits test "
            "trajectories; in --eval-pair-samples mode, valid pair starts."
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
    eval_group = parser.add_mutually_exclusive_group()
    eval_group.add_argument(
        "--eval-test-trajectories",
        dest="eval_test_trajectories",
        action="store_true",
        default=True,
        help=(
            "Use one rollout per test trajectory, preferring t0=0. This is the "
            "default. If --horizon is omitted, evaluate the full available window."
        ),
    )
    eval_group.add_argument(
        "--eval-pair-samples",
        dest="eval_test_trajectories",
        action="store_false",
        help="Legacy mode: use the first --num-samples valid test pair starts.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=("auto", "cpu", "cuda", "mps"),
        help="Optional device override for rollout inference.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Output PDF path. Default: "
            "<checkpoint-dir>/paper_plots/paper_spacetime_compare_column.pdf."
        ),
    )
    parser.add_argument(
        "--page-width",
        type=float,
        default=6.5,
        help="PDF page width in inches.",
    )
    parser.add_argument(
        "--page-height",
        type=float,
        default=10.0,
        help="PDF page height in inches.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="DPI used when embedding figure raster content in the PDF.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional summary JSON path used to infer display labels.",
    )
    parser.add_argument(
        "--baseline-schemes",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Baselines to include. Default: all available finite-volume baselines. "
            "Use 'none', 'all', or values from godunov, muscl, weno3."
        ),
    )
    parser.add_argument(
        "--spacetime-delta-plots",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Choose which spacetime delta/error panels to include. Default: all "
            "shown prediction sources. Use 'all', 'none', 'gnn', 'baselines', "
            "or selected baseline schemes/labels such as godunov, muscl, weno3. "
            "Comma-separated lists are accepted."
        ),
    )
    parser.add_argument(
        "--no-recompute-missing-baselines",
        action="store_true",
        help=(
            "Skip non-Godunov baseline rows that require extra finite-volume "
            "rollouts from the exact initial state."
        ),
    )
    parser.add_argument(
        "--gnn-label",
        type=str,
        default=None,
        help="Label for GNN panels. Default: summary JSON label or 'GNN'.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        action="append",
        default=None,
        help="Zero-based sample index to plot. Repeat to select multiple pages.",
    )
    parser.add_argument(
        "--sample-id",
        type=int,
        action="append",
        default=None,
        help="Saved sample/pair id to plot. Repeat to select multiple pages.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Limit the number of pages written after sample selection.",
    )
    parser.add_argument(
        "--save-png-time-index",
        "--png-time-index",
        dest="save_png_time_indices",
        type=int,
        action="append",
        default=None,
        help=(
            "Also save generated page(s) whose saved test time index t0 matches "
            "this value as PNG files next to the output PDF. Repeat to export "
            "multiple time indices."
        ),
    )
    parser.add_argument(
        "--save-png-page",
        "--png-page",
        dest="save_png_pages",
        type=int,
        action="append",
        default=None,
        help=(
            "Also save the selected 1-based generated PDF page number as a PNG "
            "file next to the output PDF. Repeat to export multiple pages."
        ),
    )
    parser.add_argument(
        "--state-cmap",
        type=str,
        default="viridis",
        help="Matplotlib colormap for state heatmaps.",
    )
    parser.add_argument(
        "--error-cmap",
        type=str,
        default="magma",
        help="Matplotlib colormap for absolute-error heatmaps.",
    )
    parser.add_argument(
        "--title-prefix",
        type=str,
        default="1D Inviscid Burger's",
        help="Prefix for the optional page title.",
    )
    parser.add_argument(
        "--no-page-title",
        action="store_true",
        help="Do not draw the per-page title.",
    )
    parser.add_argument(
        "--hide-colorbars",
        action="store_true",
        help="Do not draw per-subplot colorbars.",
    )
    parser.add_argument(
        "--hspace",
        type=float,
        default=None,
        help=(
            "Optional vertical spacing override for constrained layout. "
            "Default uses Matplotlib's automatic spacing."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint
    if not checkpoint.exists():
        raise FileNotFoundError(f"--checkpoint path does not exist: {checkpoint}")
    if float(args.page_width) <= 0.0 or float(args.page_height) <= 0.0:
        raise ValueError("--page-width and --page-height must be positive.")
    if int(args.dpi) < 1:
        raise ValueError("--dpi must be positive.")

    import torch
    from burgers_data import filter_burgers_dataset_by_regime, load_burgers_dataset
    from compare_gnn_vs_godunov import (
        _default_test_effective_indices,
        _evaluate_pair_rollout,
        _load_data_source_from_cfg,
        _safe_index_list,
        _select_representative_pairs_per_test_trajectory,
    )
    from gnn_burgers import (
        build_model_from_config,
        predict_type_from_target_mode,
        resolve_device,
    )
    from input_features import (
        count_input_feature_channels,
        resolve_input_feature_cfg,
        u_column_index,
    )
    from physics_inputs import count_physics_input_channels, resolve_physics_input_cfg

    ckpt = torch.load(checkpoint, map_location="cpu")
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
    if bool(args.use_all_trajectories):
        if args.data_dir is None:
            raise ValueError("--use-all-trajectories requires --data-dir.")
        if data_dir.suffix.lower() != ".pt":
            raise ValueError(
                "--use-all-trajectories currently supports WENO-A .pt files only; "
                f"got --data-dir={data_dir}."
            )
        if source_format not in ("auto", "wenoa_pt"):
            raise ValueError(
                "--use-all-trajectories requires source_format='auto' or "
                f"'wenoa_pt', got {source_format!r} from the checkpoint config."
            )
        source_format = "wenoa_pt"
        source_options = dict(source_options)
        source_options["split_paths"] = {"test": str(data_dir)}

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
    source_format_resolved = str(
        dset.metadata.get("source_format", source_format)
    ).strip().lower()

    dt = float(dset.metadata["dt"])
    dx = float(dset.metadata["dx"])
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

    if bool(args.use_all_trajectories):
        test_effective = _default_test_effective_indices(dset, horizon=horizon)
        selection_mode = "all_data_dir_trajectories"
    else:
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

    if test_effective is None or len(test_effective) == 0:
        raise ValueError("No rollout starts available for requested horizon.")
    bad_ids = [i for i in test_effective if (i < 0 or i >= len(dset))]
    if bad_ids:
        raise ValueError(
            "Invalid pair indices in selected test set. "
            f"Dataset size={len(dset)}, bad ids (first 10)={bad_ids[:10]}"
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
        if num_samples is None:
            sample_pair_ids = list(representative_ids)
            selection_mode = "one_per_test_trajectory_t0_nearest"
        else:
            sample_pair_ids = list(
                representative_ids[: min(num_samples, len(representative_ids))]
            )
            selection_mode = "one_per_test_trajectory_t0_nearest_limited"
    else:
        assert num_samples is not None
        sample_pair_ids = list(test_effective[: min(num_samples, len(test_effective))])

    if len(sample_pair_ids) == 0:
        raise ValueError("No samples selected for plotting.")

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

    sample_ids = np.asarray(sample_pair_ids, dtype=np.int64)
    local_tid_all = dset.pair_trajectory_id.detach().cpu().numpy().astype(np.int64)
    time_index_all = dset.pair_time_index.detach().cpu().numpy().astype(np.int64)
    trajectory_ids = dset.trajectory_ids.detach().cpu().numpy().astype(np.int64)
    sample_local_ids = local_tid_all[sample_ids]
    sample_time_indices = time_index_all[sample_ids]
    sample_global_ids = trajectory_ids[sample_local_ids]

    summary = _load_summary(checkpoint, args.summary_json)
    gnn_label = str(
        args.gnn_label
        if args.gnn_label is not None
        else summary.get("legend_label_gnn", "GNN")
    )

    x_nodes = dset.pos.detach().cpu().numpy().reshape(-1)
    available_schemes = list(BASELINE_SCHEMES)
    selected_baselines = _resolve_baseline_schemes(
        args.baseline_schemes,
        available_schemes=available_schemes,
    )

    label_by_scheme = {
        scheme: BASELINE_SCHEME_DEFAULT_LABELS.get(scheme, scheme)
        for scheme in BASELINE_SCHEMES
    }
    summary_labels = summary.get("baseline_labels")
    if isinstance(summary_labels, dict):
        for scheme, label in summary_labels.items():
            try:
                label_by_scheme[normalize_baseline_scheme(str(scheme))] = str(label)
            except ValueError:
                pass
    spacetime_delta_plots = _resolve_spacetime_delta_plots(
        args.spacetime_delta_plots,
        selected_baselines=selected_baselines,
        label_by_scheme=label_by_scheme,
        gnn_label=gnn_label,
    )

    page_indices = _resolve_indices(
        n_samples=len(sample_pair_ids),
        sample_ids=sample_ids,
        requested_indices=args.sample_index,
        requested_sample_ids=args.sample_id,
        max_pages=args.max_pages,
    )
    if not page_indices:
        raise ValueError("No pages selected.")

    save_png_pages = (
        []
        if args.save_png_pages is None
        else [int(v) for v in args.save_png_pages]
    )
    if any(v < 1 for v in save_png_pages):
        raise ValueError("--save-png-page values must be 1-based positive integers.")
    save_png_page_set = set(save_png_pages)
    if save_png_page_set:
        max_page = len(page_indices)
        bad_pages = [p for p in sorted(save_png_page_set) if p > max_page]
        if bad_pages:
            raise ValueError(
                "--save-png-page requested page(s) outside the generated PDF: "
                f"{bad_pages}. This run generates {max_page} page(s)."
            )

    save_png_time_indices = (
        []
        if args.save_png_time_indices is None
        else [int(v) for v in args.save_png_time_indices]
    )
    if any(v < 0 for v in save_png_time_indices):
        raise ValueError("--save-png-time-index values must be nonnegative.")
    save_png_time_set = set(save_png_time_indices)
    if save_png_time_set:
        matching_png_pages = [
            idx
            for idx in page_indices
            if (
                idx < sample_time_indices.shape[0]
                and int(sample_time_indices[idx]) in save_png_time_set
            )
        ]
        if len(matching_png_pages) == 0:
            requested = ", ".join(str(v) for v in sorted(save_png_time_set))
            raise ValueError(
                "No selected/generated page has sample_time_index in "
                f"{{{requested}}}."
            )

    out = (
        args.out
        if args.out is not None
        else checkpoint.parent / "paper_plots" / "paper_spacetime_compare_column.pdf"
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    skipped_baselines: set[str] = set()
    png_paths: list[Path] = []
    with PdfPages(out) as pdf:
        for page_number, sample_index in enumerate(page_indices, start=1):
            pid = int(sample_pair_ids[sample_index])
            (
                local_tid,
                t0,
                global_tid,
                true_traj,
                gnn_traj,
                god_traj,
            ) = _evaluate_pair_rollout(
                dset=dset,
                pid=pid,
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
            if true_traj.shape[0] == 0:
                raise ValueError(f"Sample index {sample_index} has no finite true states.")

            baseline_rows: list[tuple[str, str, np.ndarray]] = []
            max_h_sample = int(max(0, true_traj.shape[0] - 1))
            for scheme in selected_baselines:
                label = label_by_scheme.get(scheme, scheme)
                if scheme == BASELINE_SCHEME_GODUNOV:
                    baseline_rows.append(
                        (
                            scheme,
                            label,
                            god_traj,
                        )
                    )
                    continue
                if args.no_recompute_missing_baselines:
                    skipped_baselines.add(scheme)
                    continue
                baseline_rows.append(
                    (
                        scheme,
                        label,
                        _rollout_baseline(
                            x0=true_traj[0],
                            dt=float(dt),
                            dx=float(dx),
                            max_h=max_h_sample,
                            scheme=scheme,
                        ),
                    )
                )

            fig = _build_page_figure(
                sample_index=int(sample_index),
                sample_id=pid,
                global_tid=int(global_tid),
                local_tid=int(local_tid),
                time_index=int(t0),
                x_nodes=x_nodes,
                dt=dt,
                true_arr=true_traj,
                gnn_arr=gnn_traj,
                baseline_rows=baseline_rows,
                gnn_label=gnn_label,
                delta_plot_sources=spacetime_delta_plots,
                page_width=float(args.page_width),
                page_height=float(args.page_height),
                state_cmap=str(args.state_cmap),
                error_cmap=str(args.error_cmap),
                title_prefix=str(args.title_prefix),
                show_page_title=not bool(args.no_page_title),
                show_colorbars=not bool(args.hide_colorbars),
                hspace=None if args.hspace is None else float(args.hspace),
            )
            pdf.savefig(fig, dpi=int(args.dpi))
            should_save_png = (
                page_number in save_png_page_set
                or int(t0) in save_png_time_set
            )
            if should_save_png:
                png_path = _page_png_path(
                    out_pdf=out,
                    page_number=page_number,
                    sample_index=int(sample_index),
                    time_index=int(t0),
                    sample_id=pid,
                )
                fig.savefig(png_path, dpi=int(args.dpi))
                png_paths.append(png_path)
            plt.close(fig)

    if skipped_baselines:
        print(
            "[WARN] Skipped non-Godunov baseline trajectory rows for: "
            + ", ".join(sorted(skipped_baselines))
            + ". Omit --no-recompute-missing-baselines to include them.",
            flush=True,
        )
    time_msg = f"dt={dt:.6g}" if dt is not None else "rollout-step y-axis"
    print(f"Saved paper spacetime comparison PDF: {out}")
    print(f"  pages: {len(page_indices)}")
    print(f"  checkpoint: {checkpoint}")
    print(f"  sample selection: {selection_mode}")
    print(f"  horizon: {horizon}")
    print(f"  device: {device}")
    print(f"  page size: {float(args.page_width):.3g} x {float(args.page_height):.3g} in")
    print(f"  baselines: {', '.join(selected_baselines) if selected_baselines else 'none'}")
    print(
        "  spacetime delta plots: "
        + (", ".join(spacetime_delta_plots) if spacetime_delta_plots else "none")
    )
    print(f"  time axis: {time_msg}")
    if save_png_page_set or save_png_time_set:
        if save_png_page_set:
            print(
                "  png page numbers requested: "
                + ", ".join(str(v) for v in sorted(save_png_page_set))
            )
        if save_png_time_set:
            print(
                "  png t0 values requested: "
                + ", ".join(str(v) for v in sorted(save_png_time_set))
            )
        print(f"  png pages saved: {len(png_paths)}")
        for png_path in png_paths:
            print(f"    {png_path}")


if __name__ == "__main__":
    main()
