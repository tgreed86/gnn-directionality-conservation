#!/usr/bin/env python3
"""Make paper-style vertical spacetime comparison plots.

This script produces a single multi-page PDF similar to
`sample_spacetime_compare_all.pdf`, but each page stacks panels in a column
instead of arranging them in a row.

Available options:
    --checkpoint PATH
        Required checkpoint_best.pt/checkpoint_last.pt path.
    --dataset PATH
        Optional dataset override; defaults to the dataset path in the checkpoint.
    --horizon N
        Optional rollout horizon override.
    --num-samples N
        Number of test trajectories or pairs to plot.
    --device {auto,cpu,cuda,mps}
        Optional inference device override.
    --state-residual-mode MODE
        Override model.state_residual_mode from the checkpoint config.
    --out PATH
        Output PDF path; defaults to
        <checkpoint_dir>/paper_plots/paper_spacetime_compare_all.pdf.
    --png-test-time-step N
        Also save one plotted page to PNG in the PDF output directory. The
        value first matches the evaluated sample id shown in the page title;
        if not found, it is treated as a zero-based PDF page index.
    --baseline-schemes SCHEME [SCHEME ...]
        Baselines to include. Accepts comma-separated or space-separated
        values from {godunov, muscl, weno3}, plus 'all' or 'none'.
    --spacetime-delta-plots KEY [KEY ...]
        Delta/error panels to include. Accepts comma-separated or
        space-separated values from {gnn, godunov, muscl, weno3}, plus
        'all' or 'none'. If omitted, every shown prediction gets a delta panel.
    --gnn-label TEXT
        Panel label for the current model.
    --godunov-label TEXT
        Panel label for the Godunov baseline.
    --muscl-label TEXT
        Panel label for the MUSCL baseline.
    --weno3-label TEXT
        Panel label for the WENO-3 baseline.
    --page-width INCHES
        PDF page/figure width in inches.
    --page-height INCHES
        PDF page/figure height in inches; if omitted, it is chosen from the
        number of vertical panels.
    --nearest-interpolation
        Use interpolation='nearest' for all spacetime heatmaps.
    --true-dx-dt-aspect
        Render the underlying spacetime array as native heatmap cells using
        interpolation='none' and resample=False. This does not change the
        axes aspect; page width and height still control panel proportions.
    --dpi N
        Rasterization DPI used when saving heatmaps into the PDF.

Example:
    python scripts/plot_paper_spacetime_compare.py \
        --checkpoint runs/my_run/checkpoint_best.pt \
        --baseline-schemes godunov \
        --page-width 4.8 \
        --page-height 12.0
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from advection_data import exact_advect_spectral_shift, load_advection_dataset
from advection_schemes import (
    BASELINE_SCHEME_DEFAULT_LABELS,
    BASELINE_SCHEME_GODUNOV,
    BASELINE_SCHEME_MUSCL,
    BASELINE_SCHEME_WENO3,
)
from gnn_advection import (
    build_model_from_config,
    predict_type_from_target_mode,
    resolve_device,
    resolve_state_residual_mode,
)
from input_features import (
    count_input_feature_channels,
    resolve_input_feature_cfg,
)
from physics_inputs import count_physics_input_channels, resolve_physics_input_cfg

from compare_gnn_vs_godunov import (
    _label_arg_for_baseline,
    _replace_nonfinite_with_nan,
    _reserve_unique_label,
    _resolve_baseline_schemes,
    _resolve_eval_mode_and_ids,
    _resolve_godunov_correction_cfg,
    _resolve_spacetime_delta_plots,
    _rollout_baseline,
    _rollout_gnn,
)


@dataclass
class SampleRollout:
    sample_id: int
    true_traj: np.ndarray
    gnn_traj: np.ndarray
    baseline_trajs: Dict[str, np.ndarray]


@dataclass
class EvalContext:
    samples: List[SampleRollout]
    pos: np.ndarray
    dt: float
    dx: float
    baseline_labels: Dict[str, str]
    gnn_label: str
    eval_mode: str
    horizon: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Make vertical paper-style spacetime comparison plots."
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
        help="Number of test trajectories/pairs to plot.",
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
        help="Override model.state_residual_mode from the checkpoint config.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Output PDF path. Default: "
            "<checkpoint_dir>/paper_plots/paper_spacetime_compare_all.pdf"
        ),
    )
    parser.add_argument(
        "--png-test-time-step",
        type=int,
        default=None,
        help=(
            "Also save one plotted PDF page as a PNG in the PDF output "
            "directory. The value first matches the evaluated sample id shown "
            "in the page title; if not found, it is treated as a zero-based "
            "PDF page index."
        ),
    )
    parser.add_argument(
        "--baseline-schemes",
        type=str,
        nargs="+",
        default=[BASELINE_SCHEME_GODUNOV],
        help=(
            "Finite-volume baselines to include. Use space-separated or "
            "comma-separated values from {godunov, muscl, weno3}, 'all', or "
            "'none'. Default: godunov."
        ),
    )
    parser.add_argument(
        "--spacetime-delta-plots",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Choose which delta/error panels to include. Use space-separated "
            "or comma-separated values from {gnn, godunov, muscl, weno3}, "
            "'all', or 'none'. Default: current behavior, meaning every shown "
            "prediction gets a delta panel."
        ),
    )
    parser.add_argument(
        "--gnn-label",
        type=str,
        default="GNN",
        help="Panel label for the current model.",
    )
    parser.add_argument(
        "--godunov-label",
        type=str,
        default="Godunov",
        help="Panel label for the Godunov baseline.",
    )
    parser.add_argument(
        "--muscl-label",
        type=str,
        default=BASELINE_SCHEME_DEFAULT_LABELS[BASELINE_SCHEME_MUSCL],
        help="Panel label for the MUSCL baseline.",
    )
    parser.add_argument(
        "--weno3-label",
        type=str,
        default=BASELINE_SCHEME_DEFAULT_LABELS[BASELINE_SCHEME_WENO3],
        help="Panel label for the WENO-3 baseline.",
    )
    parser.add_argument(
        "--page-width",
        type=float,
        default=4.8,
        help="PDF page/figure width in inches.",
    )
    parser.add_argument(
        "--page-height",
        type=float,
        default=None,
        help=(
            "PDF page/figure height in inches. Default: choose a height from "
            "the number of vertical panels."
        ),
    )
    parser.add_argument(
        "--nearest-interpolation",
        action="store_true",
        help=(
            "Use interpolation='nearest' for all spacetime heatmaps so raw grid "
            "cells are shown without smoothing."
        ),
    )
    parser.add_argument(
        "--true-dx-dt-aspect",
        action="store_true",
        help=(
            "Render the underlying spacetime array as native heatmap cells "
            "using interpolation='none' and resample=False. This does not "
            "change the axes aspect; --page-width and --page-height control "
            "panel proportions. Default: use Matplotlib's default image "
            "sampling unless --nearest-interpolation is set."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Rasterization DPI used when saving heatmaps into the PDF.",
    )
    return parser.parse_args()


def _validate_plot_args(args: argparse.Namespace) -> None:
    if float(args.page_width) <= 0.0:
        raise ValueError(f"--page-width must be positive, got {args.page_width}.")
    if args.page_height is not None and float(args.page_height) <= 0.0:
        raise ValueError(f"--page-height must be positive, got {args.page_height}.")
    if int(args.dpi) < 1:
        raise ValueError(f"--dpi must be positive, got {args.dpi}.")
    if args.num_samples is not None and int(args.num_samples) < 1:
        raise ValueError(f"--num-samples must be >= 1, got {args.num_samples}.")
    if args.png_test_time_step is not None and int(args.png_test_time_step) < 0:
        raise ValueError(
            "--png-test-time-step must be >= 0, "
            f"got {args.png_test_time_step}."
        )


def _resolve_labels(
    args: argparse.Namespace,
    *,
    plot_baseline_schemes: List[str],
) -> Tuple[str, Dict[str, str]]:
    used_labels: set[str] = set()
    gnn_label = _reserve_unique_label(args.gnn_label, used_labels)
    baseline_labels: Dict[str, str] = {}
    for scheme in plot_baseline_schemes:
        baseline_labels[scheme] = _reserve_unique_label(
            _label_arg_for_baseline(args, scheme),
            used_labels,
        )
    return gnn_label, baseline_labels


def _load_rollouts(
    *,
    args: argparse.Namespace,
    plot_baseline_schemes: List[str],
    gnn_label: str,
    baseline_labels: Dict[str, str],
) -> EvalContext:
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    data_cfg = dict(cfg.get("data", {}))
    rollout_cfg = dict(cfg.get("rollout_eval", {}))
    target_mode = str(data_cfg.get("target_mode", "state")).strip().lower()
    godunov_cfg = _resolve_godunov_correction_cfg(cfg)
    if godunov_cfg.enabled and target_mode == "state":
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

    dataset_path = args.dataset or Path(
        data_cfg.get("dataset_path", "data/advection_train.npz")
    )
    num_samples = int(
        args.num_samples
        if args.num_samples is not None
        else rollout_cfg.get("num_test_samples", 16)
    )

    dset = load_advection_dataset(dataset_path, dtype=torch.float32)
    c = float(dset.metadata["c"])
    dt = float(dset.metadata["dt"])
    dx = float(dset.metadata["dx"])
    length = float(dset.metadata["L"])
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
            "physics_inputs.normalize_adv_to_minus1_1 is enabled, but "
            "adv_min/adv_max are missing from checkpoint config."
        )

    split = ckpt.get("split_indices", {})
    eval_mode, eval_ids = _resolve_eval_mode_and_ids(
        dset=dset,
        split=split,
        num_samples=num_samples,
    )
    if len(eval_ids) == 0:
        raise ValueError("No evaluation samples found in checkpoint split indices.")

    horizon_default = int(rollout_cfg.get("horizon", 100))
    if dset.rollout_states is not None:
        max_h = int(dset.rollout_states.shape[1] - 1)
        if args.horizon is None:
            horizon = min(horizon_default, max_h)
        else:
            horizon = min(int(args.horizon), max_h)
            if int(args.horizon) > max_h:
                print(
                    f"[INFO] Requested horizon {args.horizon} exceeds available "
                    f"rollout window {max_h}; using horizon={horizon}.",
                    flush=True,
                )
    else:
        horizon = int(args.horizon) if args.horizon is not None else horizon_default
    if horizon < 1:
        raise ValueError(f"Rollout horizon must be >= 1, got {horizon}.")

    device_pref = str(args.device if args.device is not None else cfg.get("device", "auto"))
    device = resolve_device(device_pref)
    in_channels = (
        1
        + count_physics_input_channels(physics_cfg)
        + count_input_feature_channels(input_cfg)
    )
    model = build_model_from_config(
        cfg,
        in_channels=in_channels,
        edge_attr_channels=int(dset.edge_attr.shape[-1]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    edge_index = dset.edge_index.detach().cpu().long()
    edge_attr = dset.edge_attr.detach().cpu()
    pos_nodes = dset.pos.detach().cpu()

    samples: List[SampleRollout] = []
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
            x0 = torch.from_numpy(true_traj[0][:, None]).to(dtype=torch.float32)
        else:
            x0 = dset.x[int(sid)].detach().cpu()
            true_traj = np.empty((horizon + 1, int(x0.shape[0])), dtype=np.float64)
            true_traj[0] = x0.squeeze(-1).numpy().astype(np.float64)
            for k in range(horizon):
                true_traj[k + 1] = exact_advect_spectral_shift(
                    true_traj[k],
                    c=c,
                    dt=dt,
                    L=length,
                )

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
            for scheme in plot_baseline_schemes
        }
        samples.append(
            SampleRollout(
                sample_id=int(sid),
                true_traj=true_traj,
                gnn_traj=gnn_traj,
                baseline_trajs=baseline_trajs,
            )
        )

    print(
        "[INFO] Loaded rollouts: "
        f"eval_mode={eval_mode}, samples={len(samples)}, horizon={horizon}, "
        f"device={device}, model_label={gnn_label}, "
        f"baselines={', '.join(baseline_labels.values()) if baseline_labels else 'none'}",
        flush=True,
    )
    return EvalContext(
        samples=samples,
        pos=pos,
        dt=dt,
        dx=dx,
        baseline_labels=baseline_labels,
        gnn_label=gnn_label,
        eval_mode=eval_mode,
        horizon=horizon,
    )


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


def _finite_max(arrays: List[np.ndarray], *, fallback: float) -> float:
    vals = []
    for arr in arrays:
        finite = arr[np.isfinite(arr)]
        if finite.size > 0:
            vals.append(float(np.max(finite)))
    if len(vals) == 0:
        return float(fallback)
    return max(max(vals), float(fallback))


def _auto_page_height(num_panels: int) -> float:
    return max(4.0, 2.35 * float(num_panels) + 0.6)


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


def _resolve_png_page_index(
    *,
    samples: List[SampleRollout],
    requested: int | None,
) -> int | None:
    if requested is None:
        return None
    for idx, sample in enumerate(samples):
        if int(sample.sample_id) == int(requested):
            return idx
    if 0 <= int(requested) < len(samples):
        return int(requested)
    sample_ids = ", ".join(str(sample.sample_id) for sample in samples[:10])
    suffix = "" if len(samples) <= 10 else ", ..."
    raise ValueError(
        "--png-test-time-step must match an evaluated sample id or zero-based "
        f"page index. Got {requested}; available first sample ids: "
        f"[{sample_ids}{suffix}], number of pages={len(samples)}."
    )


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


def write_pdf(
    *,
    ctx: EvalContext,
    out_path: Path,
    args: argparse.Namespace,
    spacetime_delta_plots: List[str] | None,
) -> Path | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    out_path.parent.mkdir(parents=True, exist_ok=True)
    imshow_interpolation = "nearest" if args.nearest_interpolation else None
    png_page_idx = _resolve_png_page_index(
        samples=ctx.samples,
        requested=args.png_test_time_step,
    )
    png_path = None
    if png_page_idx is not None:
        png_sample_id = int(ctx.samples[png_page_idx].sample_id)
        png_path = (
            out_path.parent
            / (
                "paper_spacetime_compare"
                f"_test_step_{int(args.png_test_time_step)}"
                f"_sample_{png_sample_id}.png"
            )
        )
    with PdfPages(out_path) as pdf:
        for page_idx, sample in enumerate(ctx.samples):
            fig = _make_vertical_spacetime_page(
                sample=sample,
                pos=ctx.pos,
                dt=ctx.dt,
                gnn_label=ctx.gnn_label,
                baseline_labels=ctx.baseline_labels,
                spacetime_delta_plots=spacetime_delta_plots,
                page_width=float(args.page_width),
                page_height=args.page_height,
                imshow_interpolation=imshow_interpolation,
                true_dx_dt_aspect=bool(args.true_dx_dt_aspect),
            )
            pdf.savefig(fig, dpi=int(args.dpi))
            if png_page_idx is not None and page_idx == png_page_idx:
                assert png_path is not None
                fig.savefig(png_path, dpi=int(args.dpi))
            plt.close(fig)
    return png_path


def main() -> None:
    args = parse_args()
    _validate_plot_args(args)
    plot_baseline_schemes = _resolve_baseline_schemes(args.baseline_schemes)
    spacetime_delta_plots = _resolve_spacetime_delta_plots(
        args.spacetime_delta_plots,
        plot_baseline_schemes=plot_baseline_schemes,
    )
    gnn_label, baseline_labels = _resolve_labels(
        args,
        plot_baseline_schemes=plot_baseline_schemes,
    )
    ctx = _load_rollouts(
        args=args,
        plot_baseline_schemes=plot_baseline_schemes,
        gnn_label=gnn_label,
        baseline_labels=baseline_labels,
    )
    out_path = (
        args.out
        if args.out is not None
        else args.checkpoint.parent / "paper_plots" / "paper_spacetime_compare_all.pdf"
    )
    png_path = write_pdf(
        ctx=ctx,
        out_path=out_path,
        args=args,
        spacetime_delta_plots=spacetime_delta_plots,
    )
    height_desc = (
        "auto"
        if args.page_height is None
        else f"{float(args.page_height):.3g} in"
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
    print("Paper spacetime comparison PDF complete.")
    print(f"  out: {out_path}")
    print(f"  pages: {len(ctx.samples)}")
    print(f"  page width: {float(args.page_width):.3g} in")
    print(f"  page height: {height_desc}")
    print(f"  spacetime delta plots: {delta_desc}")
    if png_path is not None:
        print(f"  png: {png_path}")
    if args.true_dx_dt_aspect:
        print(
            "  imshow sampling: native data cells "
            "(interpolation='none', resample=False)"
        )
    elif args.nearest_interpolation:
        print("  imshow interpolation: nearest")


if __name__ == "__main__":
    main()
