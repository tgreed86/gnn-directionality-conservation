"""Render paper figures from archived evaluations without trained-model inference.

Scientific selections live in reproduction/figures.json. Rendering helpers were
extracted from the archived project sources; appearance can vary with Matplotlib.
"""
import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from release_paths import RELEASE_ROOT, release_runs, dataset_path


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def source_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def save(fig, out, dpi=180):
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def mean_smooth(values, window):
    # Same centered, finite-value, edge-renormalized average as the source scripts.
    curve = np.asarray(values, dtype=float).mean(axis=0)
    kernel = np.ones(window)
    finite = np.isfinite(curve)
    numerator = np.convolve(np.where(finite, curve, 0), kernel, "same")
    denominator = np.convolve(finite.astype(float), kernel, "same")
    return np.divide(numerator, denominator, out=np.full_like(curve, np.nan), where=denominator > 0)


def curves_1d(job, out):
    metrics = job["metrics"]
    fig, axes = plt.subplots(1 if len(metrics) == 2 else 2, 2,
                             figsize=(15.5, 4.8 if len(metrics) == 2 else 8.3),
                             constrained_layout=True, squeeze=False)
    names = {"rmse": "RMSE", "rel_l2": "rel-L2", "l1": "L1", "ssim": "Local SSIM"}
    exported = {}
    for ax, metric in zip(axes.flat, metrics):
        for curve in job["curves"]:
            z = load(RELEASE_ROOT / curve["archive"])
            raw = z[metric + "_" + curve["source"] + "_by_sample"]
            if not np.isfinite(raw).all():
                raise ValueError("Nonfinite saved metric: " + curve["archive"])
            times = np.arange(1, raw.shape[1] + 1) * job["dt"]
            mean = mean_smooth(raw, job["smooth_window"])
            ax.plot(times, mean, label=curve["label"], color=curve["color"],
                    linestyle=curve.get("linestyle", "-"), lw=2)
            exported[curve["label"] + "/" + metric] = {"raw_mean": raw.mean(axis=0).tolist(),
                                                        "display_mean": mean.tolist()}
        ax.set_xlabel("Time", fontsize=20)
        ax.set_ylabel(names[metric], fontsize=20)
        ax.tick_params(labelsize=16)
        ax.grid(alpha=.3)
    list(axes.flat)[metrics.index("ssim")].legend(loc="lower left", fontsize=16)
    fig.suptitle(job["title"], fontsize=20)
    save(fig, out, job["dpi"])
    out.with_suffix(".json").write_text(json.dumps({"curves": exported, "settings": job}, indent=2) + "\n")


def advection_space(job, out):
    from plot_helpers.advection import SampleRollout, _make_vertical_spacetime_page
    scheme = source_module("release_advection_schemes", RELEASE_ROOT / "code/1D_advection/src/advection_schemes.py")
    run = next(r for r in release_runs() if r["id"] == job["run_id"])
    z = load(RELEASE_ROOT / run["evaluation_archive"])
    data = load(dataset_path(RELEASE_ROOT, "Linear Advection"))
    index = list(z["all_sample_ids"]).index(job["trajectory_id"])
    pred = z["all_gnn_trajs"][index]
    true = data["rollout_states"][job["trajectory_id"], :pred.shape[0]]
    if true.ndim == 3:
        true = true[..., 0]
    dt = float(data["dt"])
    dx = float(data["dx"])
    baseline = scheme.rollout_finite_volume(x0=true[0], horizon=pred.shape[0]-1,
                                            c=float(data["c"]), dt=dt, dx=dx, scheme="godunov")
    sample = SampleRollout(job["trajectory_id"], true, pred, {"godunov": baseline})
    pos = np.asarray(data["pos"]).reshape(-1)
    fig = _make_vertical_spacetime_page(sample=sample, pos=pos, dt=dt, gnn_label="GNN",
        baseline_labels={"godunov": "Godunov"}, spacetime_delta_plots=["gnn"],
        page_width=job["page_width"], page_height=job["page_height"],
        imshow_interpolation="nearest", true_dx_dt_aspect=True)
    save(fig, out, job["dpi"])


def heat(job, out):
    from plot_helpers.heat import _write_rollout_plots
    z = load(RELEASE_ROOT / job["archive"])
    data = load(dataset_path(RELEASE_ROOT, "Heat Equation"))
    _write_rollout_plots(plots_dir=out.parent / "heat_diagnostics", horizon=int(z["horizon"]),
        sample_ids=z["sample_ids"].tolist(), rmse_arr=z["rmse_by_sample"], rel_l2_arr=z["rel_l2_by_sample"],
        mean_rmse=z["mean_rmse"], mean_rel_l2=z["mean_rel_l2"], dt=float(data["dt"]),
        nx=int(z["nx"]), ny=int(z["ny"]), spatial_dim=1, delta_error_mode="absolute",
        delta_percent_eps=1e-8, make_sample_gif=False, gif_out=None, gif_fps=10,
        sample_plot_steps=z["sample_plot_steps"].tolist(), sample_traj_id=int(z["sample_traj_id"]),
        sample_pred_traj=z["sample_pred_traj"], sample_true_traj=z["sample_true_traj"])
    import shutil
    shutil.copy2(out.parent / "heat_diagnostics/rollout_sample_final_state.png", out)


def burgers_space(job, out):
    source = source_module("release_burgers_plotter",
        RELEASE_ROOT / "code/Burgers/scripts/plot_paper_spacetime_compare.py")
    z = load(RELEASE_ROOT / job["archive"])
    if int(z["trajectory_id"]) != job["trajectory_id"] or int(z["time_index"]) != 0:
        raise ValueError("Burgers archive does not match the confirmed trajectory selection")
    fig = source._build_page_figure(sample_index=job["trajectory_id"], sample_id=int(z["pair_id"]),
        global_tid=job["trajectory_id"], local_tid=job["trajectory_id"], time_index=0,
        x_nodes=z["pos"], dt=float(z["dt"]), true_arr=z["truth"], gnn_arr=z["prediction"],
        baseline_rows=[("godunov", "Godunov", z["godunov"])], gnn_label="GNN",
        delta_plot_sources=["gnn"], page_width=job["page_width"], page_height=job["page_height"],
        state_cmap="viridis", error_cmap="magma", title_prefix="1D Inviscid Burger's",
        show_page_title=True, show_colorbars=True, hspace=None)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Match the source plotter's full-page export (no tight bounding-box crop).
    fig.savefig(out, dpi=job["dpi"])
    fig.savefig(out.with_suffix(".pdf"), dpi=job["dpi"])
    plt.close(fig)


def traffic_space(job, out):
    from plot_helpers.traffic import plot_network_comparison
    data = load(dataset_path(RELEASE_ROOT, "LWR Traffic") / "topology_0026.npz")
    z = load(RELEASE_ROOT / job["archive"])
    index = list(z["trajectory_ids"]).index(job["trajectory_id"])
    plot_network_comparison(data, z["ground_truth"][index], z["prediction"][index],
        job["trajectory_id"], out, snapshot_steps=job["snapshot_steps"], dt=float(z["dt"]),
        rho_max=float(data["rho_max"]), dpi=job["dpi"], cmap="viridis",
        flow_direction=float(z["trajectory_flow_direction"][index]), time_axis="columns")


def chem_space(job, out):
    from plot_helpers.chemotaxis import plot_spatial_comparison
    from chemotaxis_archive import load_archive
    data = load_archive(dataset_path(RELEASE_ROOT, "Chemotactic Cell Migration") /
                        ("trajectory_%05d.npz" % job["trajectory_id"]))
    z = load(RELEASE_ROOT / job["archive"])
    i = list(z["trajectory_ids"]).index(job["trajectory_id"])
    plot_spatial_comparison(data, z["truth"][i, ..., 0], z["prediction"][i, ..., 0], 0, out,
        snapshot_steps=job["snapshot_steps"], dt=float(z["time"][1]-z["time"][0]),
        model_label="FluxGraphNet", dpi=job["dpi"], cmap="magma", chemo_contours=True,
        chemo_contour_levels=job["contour_levels"], chemo_contour_alpha=.45,
        chemo_contour_method="analytic-grid", chemo_contour_resolution=300, spatial_time_axis="columns")


def chem_curves(job, out):
    from plot_helpers.chemotaxis_curves import plot_figure6_error_curves, figure6_model_styles, _normalized_token
    z = load(RELEASE_ROOT / job["archive"])
    attrs = {"rmse": "rmse_by_time", "rel_l2": "area_weighted_relative_l2_by_time",
             "l1": "mae_by_time", "ssim": "local_ssim_by_time"}
    metric_names = list(z["metric_names"])
    by_model = {}
    for label in job["labels"]:
        i = list(z["model_labels"]).index(label)
        by_model[label] = [SimpleNamespace(**{attrs[m]: z["values"][i, j, metric_names.index(m)]
                                             for m in job["metrics"]})
                           for j in range(len(z["trajectory_ids"]))]
    styles = figure6_model_styles(job["labels"], color_overrides={
        _normalized_token(k): v for k, v in job.get("colors", {}).items()})
    plot_figure6_error_curves(metrics_by_model=by_model, times=z["time"], selected_metrics=job["metrics"],
        styles=styles, output_base=out.with_suffix(""), formats=["png", "pdf"], dpi=job["dpi"],
        smooth_window=job["smooth_window"], show_trajectories=False, spread="none", log_y=False,
        title=job["title"], show_persistence=False)


def traffic_curves(job, out):
    import subprocess
    import shutil
    script = RELEASE_ROOT / "code/1D_traffic/scripts/plot_model_comparisons_paper.py"
    directory = out.parent / job["id"]
    command = [sys.executable, str(script), "--output-dir", str(directory), "--data",
               str(dataset_path(RELEASE_ROOT, "LWR Traffic")), "--error-curves-metrics", *job["metrics"],
               "--error-curves-smooth-window", str(job["smooth_window"]), "--error-curves-spread", "none",
               "--combined-average", "macro", "--combined-mean-only", "--error-curves-mean-only",
               "--ssim-reference-spacing", "0.125", "--dpi", str(job["dpi"])]
    runs = {r["id"]: r for r in release_runs()}
    for label, run_id in job["models"].items():
        command += ["--model", label + "=" + str(RELEASE_ROOT / runs[run_id]["evaluation_directory"] / "rollout_plots")]
    if job.get("colors"):
        command += ["--curve-color", *[k + "=" + v for k, v in job["colors"].items()]]
    subprocess.run(command, check=True)
    source = directory / job["selected_split"] / "error_curves_compare.png"
    shutil.copy2(source, out)
    if source.with_suffix(".pdf").exists():
        shutil.copy2(source.with_suffix(".pdf"), out.with_suffix(".pdf"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("job")
    p.add_argument("--output-dir", type=Path, default=RELEASE_ROOT / "outputs/figures")
    args = p.parse_args()
    jobs = json.loads((RELEASE_ROOT / "reproduction/figures.json").read_text())["figures"]
    job = next(j for j in jobs if j["id"] == args.job)
    if job["status"] != "ready":
        p.error(job["reason"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(RELEASE_ROOT / "code/chemotactic_cell_migration/scripts"))
    renderers = {"curves_1d": curves_1d, "advection_space": advection_space, "heat": heat,
                 "traffic_space": traffic_space, "chem_space": chem_space, "chem_curves": chem_curves,
                 "traffic_curves": traffic_curves, "burgers_space": burgers_space}
    renderers[job["renderer"]](job, args.output_dir / job["asset"])
    print("Wrote", args.output_dir / job["asset"])


if __name__ == "__main__":
    main()
