#!/usr/bin/env python
"""Generate a LaTeX main-results table from saved metric JSON.

Default usage from the repository root:

    python scripts/make_results_table.py

This reads drafts/6-25-26/results_table_metrics.json and writes
drafts/6-25-26/results_table_metrics.tex. The JSON remains the canonical
reproducibility artifact; this script only formats it for LaTeX.
"""

from __future__ import print_function

import argparse
import json
import math
from pathlib import Path


DEFAULT_INPUT = Path("drafts/6-25-26/results_table_metrics.json")
DEFAULT_OUTPUT = Path("drafts/6-25-26/results_table_metrics.tex")
SORT_METRIC_ALIASES = {
    "rel_l2": "final_rel_l2_mean",
    "relative_l2": "final_rel_l2_mean",
    "final_rel_l2": "final_rel_l2_mean",
    "hg_rel_l2": "final_high_gradient_rel_l2_mean",
    "high_gradient_rel_l2": "final_high_gradient_rel_l2_mean",
    "final_high_gradient_rel_l2": "final_high_gradient_rel_l2_mean",
    "ssim": "final_ssim_mean",
    "final_ssim": "final_ssim_mean",
    "conservation": "final_conservation_error_mean",
    "conservation_error": "final_conservation_error_mean",
    "final_conservation_error": "final_conservation_error_mean",
}
TABLE_METRIC_HEADERS = (
    ("final_rel_l2_mean", r"Rel. $L^2$ $\downarrow$"),
    ("final_high_gradient_rel_l2_mean", r"HG Rel. $L^2$ $\downarrow$"),
    ("final_ssim_mean", r"SSIM $\uparrow$"),
    ("final_conservation_error_mean", r"Cons. err. $\downarrow$"),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a booktabs LaTeX table from results_table_metrics.json."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input metrics JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output LaTeX table file.",
    )
    parser.add_argument(
        "--caption",
        default=(
            "Final-time rollout metrics for the benchmark problems. Values are "
            "mean $\\pm$ standard deviation over the "
            "available test trajectories."
        ),
        help="Table caption.",
    )
    parser.add_argument(
        "--label",
        default="tab:main_results",
        help="LaTeX label for the generated table.",
    )
    parser.add_argument(
        "--mean-only",
        action="store_true",
        help="Show only means rather than mean +/- standard deviation.",
    )
    parser.add_argument(
        "--no-resizebox",
        action="store_true",
        help="Do not wrap the tabular in resizebox{\\textwidth}{!}{...}.",
    )
    parser.add_argument(
        "--conservation-threshold",
        type=float,
        default=1.0e-5,
        help=(
            "Display conservation-error means with absolute value below this "
            "threshold as a less-than bound. Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--sort-by",
        default=None,
        help=(
            "Sort rows within each dataset group by this metric. Use a JSON "
            "field name such as final_rel_l2_mean, or an alias: rel_l2, "
            "high_gradient_rel_l2, ssim, conservation_error. By default, keep "
            "the JSON/RUNS order. The corresponding displayed metric header "
            "is bolded when sorting is enabled."
        ),
    )
    parser.add_argument(
        "--sort-descending",
        action="store_true",
        help="Sort larger metric values first. Default is ascending.",
    )
    return parser.parse_args()


def latex_escape(text):
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in str(text))


def finite_float(value):
    if value is None:
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    return x


def latex_number(value, sig=3):
    x = finite_float(value)
    if x is None:
        return r"\textemdash{}"
    if sig < 1:
        raise ValueError("sig must be at least 1")
    if x == 0.0:
        return ("{0:." + str(sig - 1) + "f}").format(0.0)

    ax = abs(x)
    if ax < 1.0e-3 or ax >= 1.0e3:
        mantissa_text, exponent_text = ("{0:." + str(sig - 1) + "e}").format(x).split("e")
        exponent = int(exponent_text)
        return r"{0}\times 10^{{{1}}}".format(mantissa_text, exponent)

    exponent = int(math.floor(math.log10(ax)))
    decimals = max(sig - exponent - 1, 0)
    rounded = round(x, decimals)
    if abs(rounded) >= 1.0e3:
        mantissa_text, exponent_text = ("{0:." + str(sig - 1) + "e}").format(
            rounded
        ).split("e")
        return r"{0}\times 10^{{{1}}}".format(
            mantissa_text,
            int(exponent_text),
        )
    return ("{0:." + str(decimals) + "f}").format(x)


def significant_figure_place(value, sig=3):
    x = finite_float(value)
    if x is None:
        return None
    if x == 0.0:
        return -(sig - 1)
    rounded = float(("{0:." + str(sig) + "g}").format(x))
    exponent = int(math.floor(math.log10(abs(rounded))))
    return exponent - sig + 1


def uncertainty_number(uncertainty, mean, mean_sig=3):
    uncertainty_x = finite_float(uncertainty)
    place = significant_figure_place(mean, sig=mean_sig)
    if uncertainty_x is None or place is None:
        return r"\textemdash{}"
    if uncertainty_x == 0.0:
        if place < 0:
            return ("{0:." + str(-place) + "f}").format(0.0)
        return "0"

    uncertainty_exponent = int(math.floor(math.log10(abs(uncertainty_x))))
    uncertainty_sig = max(1, uncertainty_exponent - place + 1)
    return latex_number(uncertainty_x, sig=uncertainty_sig)


def metric_cell(mean, std, mean_only=False, sig=3, marker=""):
    mean_text = latex_number(mean, sig=sig)
    if mean_only:
        return r"${0}${1}".format(mean_text, marker)

    std_x = finite_float(std)
    if std_x is None:
        return r"${0}${1}".format(mean_text, marker)

    std_text = uncertainty_number(std_x, mean, mean_sig=sig)
    return r"${0} \pm {1}${2}".format(mean_text, std_text, marker)


def less_than_cell(threshold, marker=""):
    threshold_x = finite_float(threshold)
    if threshold_x is None or threshold_x <= 0.0:
        return None
    exponent = int(round(math.log10(threshold_x)))
    if math.isclose(threshold_x, 10.0 ** exponent, rel_tol=1.0e-12, abs_tol=0.0):
        threshold_text = r"10^{{{0}}}".format(exponent)
    else:
        threshold_text = latex_number(threshold_x, sig=1)
    return r"$<{0}${1}".format(threshold_text, marker)


def conservation_cell(row, mean_only=False, threshold=1.0e-5):
    marker = r"\textsuperscript{*}" if should_mark_conservation(row) else ""
    mean = finite_float(row.get("final_conservation_error_mean"))
    threshold_x = finite_float(threshold)
    if (
        mean is not None
        and threshold_x is not None
        and threshold_x > 0.0
        and abs(mean) < threshold_x
    ):
        return less_than_cell(threshold_x, marker=marker)

    return metric_cell(
        mean,
        row.get("final_conservation_error_std"),
        mean_only=mean_only,
        sig=3,
        marker=marker,
    )


def should_mark_conservation(row):
    try:
        cons_n = int(row.get("final_conservation_error_n", 0))
        metric_n = int(row.get("num_samples_metrics", 0))
    except (TypeError, ValueError):
        return False
    return cons_n > 0 and metric_n > 0 and cons_n != metric_n


def dataset_display_name(dataset):
    aliases = {
        "Inviscid Burgers": "Burgers",
        "Linear Advection": "Linear advection",
        "LWR Traffic": "LWR traffic",
        "Chemotactic Cell Migration": "Chemotactic cell migration",
    }
    return aliases.get(dataset, dataset)


def dataset_latex_name(dataset):
    if dataset == "Chemotactic Cell Migration":
        return r"\shortstack{Chemotactic\\cell migration}"
    return latex_escape(dataset_display_name(dataset))


def model_latex_name(method):
    text = str(method)
    for suffix in (" + advection", " (no advection)"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
            break
    return latex_escape(text)


def physics_input_cell(row):
    physics_enabled = row.get("physics_enabled")
    include_adv = row.get("include_adv")
    if physics_enabled is True and include_adv is True:
        return "Yes"
    if physics_enabled is False or include_adv is False:
        return "No"

    # Some legacy LWR runs do not store the physics-input flags.
    method = str(row.get("method", ""))
    if method.endswith(" + advection"):
        return "Yes"
    if method.endswith(" (no advection)"):
        return "No"
    return r"\textemdash{}"


def grouped_rows(rows):
    groups = []
    seen = {}
    for row in rows:
        dataset = row.get("dataset", "")
        if dataset not in seen:
            seen[dataset] = []
            groups.append((dataset, seen[dataset]))
        seen[dataset].append(row)
    return groups


def resolve_sort_metric(sort_by):
    if sort_by is None or str(sort_by).strip() == "":
        return None
    key = str(sort_by).strip()
    return SORT_METRIC_ALIASES.get(key, key)


def sort_group_rows(group, sort_metric=None, descending=False):
    if sort_metric is None:
        return list(group)

    indexed = list(enumerate(group))

    def key(item):
        index, row = item
        value = finite_float(row.get(sort_metric))
        missing = value is None
        if value is None:
            value = 0.0
        if descending:
            value = -value
        return missing, value, index

    return [row for _, row in sorted(indexed, key=key)]


def table_header(sort_metric=None):
    metric_headers = []
    for metric, label in TABLE_METRIC_HEADERS:
        if metric == sort_metric:
            label = r"{\bfseries\boldmath " + label + "}"
        metric_headers.append(label)
    return "Dataset & Model & \\shortstack{{Physics\\\\input}} & {0} \\\\".format(
        " & ".join(metric_headers)
    )


def build_table(
    data,
    caption,
    label,
    mean_only=False,
    use_resizebox=True,
    conservation_threshold=1.0e-5,
    sort_by=None,
    sort_descending=False,
):
    rows = list(data.get("rows", []))
    sort_metric = resolve_sort_metric(sort_by)
    lines = [
        r"% Auto-generated by scripts/make_results_table.py from {0}".format(
            data.get("source_path", "results_table_metrics.json")
        ),
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{" + caption + r"}",
        r"\label{" + label + r"}",
    ]

    if use_resizebox:
        lines.append(r"\resizebox{\textwidth}{!}{%")

    lines.extend(
        [
            r"\begin{tabular}{llccccc}",
            r"\toprule",
            table_header(sort_metric=sort_metric),
            r"\midrule",
        ]
    )

    groups = grouped_rows(rows)
    for group_index, (dataset, group) in enumerate(groups):
        group = sort_group_rows(
            group,
            sort_metric=sort_metric,
            descending=sort_descending,
        )
        if group_index > 0:
            lines.append(r"\midrule")
        for row_index, row in enumerate(group):
            dataset_text = dataset_latex_name(dataset) if row_index == 0 else ""
            model_text = model_latex_name(row.get("method", ""))
            physics_text = physics_input_cell(row)
            rel_cell = metric_cell(
                row.get("final_rel_l2_mean"),
                row.get("final_rel_l2_std"),
                mean_only=mean_only,
                sig=3,
            )
            high_grad_cell = metric_cell(
                row.get("final_high_gradient_rel_l2_mean"),
                row.get("final_high_gradient_rel_l2_std"),
                mean_only=mean_only,
                sig=3,
            )
            ssim_cell = metric_cell(
                row.get("final_ssim_mean"),
                row.get("final_ssim_std"),
                mean_only=mean_only,
                sig=3,
            )
            cons_cell = conservation_cell(
                row,
                mean_only=mean_only,
                threshold=conservation_threshold,
            )
            lines.append(
                "{0} & {1} & {2} & {3} & {4} & {5} & {6} \\\\".format(
                    dataset_text,
                    model_text,
                    physics_text,
                    rel_cell,
                    high_grad_cell,
                    ssim_cell,
                    cons_cell,
                )
            )

    lines.extend([r"\bottomrule", r"\end{tabular}"])
    if use_resizebox:
        lines.append(r"}")

    if any(should_mark_conservation(row) for row in rows):
        lines.append(
            r"\par\smallskip"
            r"\footnotesize \textsuperscript{*}Conservation error was computed "
            r"from one saved representative trajectory; accuracy metrics use the "
            r"full saved test set for that row."
        )

    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    with args.input.open("r") as fh:
        data = json.load(fh)
    data["source_path"] = str(args.input)

    table = build_table(
        data,
        caption=args.caption,
        label=args.label,
        mean_only=args.mean_only,
        use_resizebox=not args.no_resizebox,
        conservation_threshold=args.conservation_threshold,
        sort_by=args.sort_by,
        sort_descending=args.sort_descending,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(table)
    print("Wrote {0}".format(args.output))


if __name__ == "__main__":
    main()
