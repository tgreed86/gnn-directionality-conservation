#!/usr/bin/env python
"""Portable release variant of make_hyperparameter_table.py.

Uses archived inputs under paper_release and writes to outputs/tables by default.
The unchanged workspace original is in provenance/paper_scripts/.
See reproduction/README.md for the complete workflow.
"""

from __future__ import print_function

import argparse
import json
import math
from pathlib import Path
from release_paths import RELEASE_ROOT


DEFAULT_INPUT = RELEASE_ROOT / "outputs/tables/results_table_metrics.json"
DEFAULT_OUTPUT = RELEASE_ROOT / "outputs/tables/model_training_hyperparameters.tex"

# Verified in the two benchmarks' scripts/train.py: SAGENet and _make_mlp
# use SiLU, and density inputs use (density - density_mean) / density_std.
# These are fallbacks for legacy JSON, never overrides of recorded settings.
DATASET_SETTING_DEFAULTS = {
    "LWR Traffic": {"activation": "silu", "state_normalization": "standardization"},
    "Chemotactic Cell Migration": {
        "activation": "silu",
        "state_normalization": "standardization",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create run-level and dataset-level booktabs LaTeX tables."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input hyperparameter JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output run-level LaTeX table file.",
    )
    parser.add_argument(
        "--dataset-output",
        type=Path,
        help="Dataset-level table file (default: OUTPUT_STEM_by_dataset.tex).",
    )
    parser.add_argument(
        "--caption",
        default=(
            "Training and model hyperparameters that differ across the final "
            "reported runs."
        ),
        help="Table caption.",
    )
    parser.add_argument(
        "--label",
        default="tab:model_hyperparameters",
        help="LaTeX label for the generated table.",
    )
    parser.add_argument(
        "--dataset-caption",
        default="Dataset-level model and training settings for the reported runs.",
        help="Dataset-level table caption.",
    )
    parser.add_argument(
        "--dataset-label",
        default="tab:dataset_hyperparameters",
        help="LaTeX label for the dataset-level table.",
    )
    parser.add_argument(
        "--no-resizebox",
        action="store_true",
        help="Do not wrap either tabular in resizebox{\\textwidth}{!}{...}.",
    )
    args = parser.parse_args()
    if args.dataset_output is None:
        args.dataset_output = args.output.with_name(args.output.stem + "_by_dataset.tex")
    if args.output.resolve() == args.dataset_output.resolve():
        parser.error("--output and --dataset-output must name different files")
    if args.label == args.dataset_label:
        parser.error("--label and --dataset-label must be different")
    return args


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


def trim_float_text(text):
    if "." not in text:
        return text
    text = text.rstrip("0").rstrip(".")
    if text == "-0":
        text = "0"
    return text


def latex_number(value, sig=3):
    x = finite_float(value)
    if x is None:
        return r"\textemdash{}"
    if x == 0.0:
        return "0"

    ax = abs(x)
    if ax < 1.0e-3 or ax >= 1.0e3:
        exponent = int(math.floor(math.log10(ax)))
        mantissa = x / (10.0 ** exponent)
        decimals = max(sig - 1, 0)
        mantissa_text = trim_float_text(("{0:." + str(decimals) + "f}").format(mantissa))
        if mantissa_text == "1":
            return r"10^{{{0}}}".format(exponent)
        return r"{0}\times 10^{{{1}}}".format(mantissa_text, exponent)

    return trim_float_text(("{0:." + str(sig) + "g}").format(x))


def math_number(value, sig=3):
    return r"${0}$".format(latex_number(value, sig=sig))


def bool_text(value):
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return r"\textemdash{}"


def dash_if_blank(text):
    if text is None or text == "":
        return r"\textemdash{}"
    return text


def dataset_latex_name(dataset):
    if dataset == "Chemotactic Cell Migration":
        return r"\shortstack{Chemotactic cell\\migration}"
    return latex_escape(dataset_display_name(dataset))


def dataset_display_name(dataset):
    aliases = {
        "Inviscid Burgers": "Burgers",
        "Linear Advection": "Linear advection",
        "LWR Traffic": "LWR traffic",
        "Chemotactic Cell Migration": r"\shortstack{Chemotactic cell\\migration}",
    }
    return aliases.get(dataset, dataset)


def method_latex_name(method):
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


def selection_metric_cell(row):
    metric = str(row.get("early_stopping_metric") or "").strip().lower()
    if not metric:
        # Legacy result JSON predates the explicit selection-metric field.
        metric = (
            "rollout_state_rel_l2"
            if row.get("dataset") == "Linear Advection"
            else "val_loss"
        )
    labels = {
        "val_loss": r"Val. MSE",
        "rollout_state_rel_l2": r"\shortstack{Val.\\rel. $L_2$}",
        "normalized_mse": r"Val. MSE",
    }
    return labels.get(metric, latex_escape(metric) if metric else r"\textemdash{}")


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


def lr_schedule_cell(row):
    lr0 = finite_float(row.get("lr_initial"))
    lrf = finite_float(row.get("lr_final"))
    entries = row.get("lr_schedule") or []

    if row.get("lr_schedule_enabled") and entries:
        if len(entries) == 1:
            entry = entries[0]
            lr_text = latex_number(lr0, sig=2)
            new_lr = latex_number(entry.get("lr"), sig=2)
            return r"\shortstack{{${0}\rightarrow {1}$\\@ {2}}}".format(
                lr_text,
                new_lr,
                latex_escape(entry.get("epoch", "")),
            )

        pieces = []
        for entry in entries:
            pieces.append(
                r"${0}$ @ {1}".format(
                    latex_number(entry.get("lr"), sig=2),
                    latex_escape(entry.get("epoch", "")),
                )
            )
        return r"${0}$; ".format(latex_number(lr0, sig=2)) + "; ".join(pieces)

    if lr0 is not None and lrf is not None and not math.isclose(lr0, lrf):
        return r"${0}\rightarrow {1}$".format(
            latex_number(lr0, sig=2), latex_number(lrf, sig=2)
        )

    return math_number(lr0, sig=2)


def tv_loss_cell(row):
    if row.get("tv_loss_enabled") is not True:
        return "none"

    lambda_text = latex_number(row.get("tv_loss_lambda"), sig=2)
    #return r"$\lambda_{{\mathrm{{TV}}}}={0}$".format(lambda_text)
    return r"${0}$".format(lambda_text)


def flux_loss_cell(row):
    weight = finite_float(row.get("flux_loss_weight"))
    if weight is None or weight <= 0.0:
        return "none"
    #return r"$\lambda_{{\mathrm{{flux}}}}={0}$".format(
    return r"${0}$".format(
        latex_number(weight, sig=2)
    )


def grad_clip_cell(row):
    clip = finite_float(row.get("grad_clip"))
    if clip is None or clip < 0.0:
        return "none"
    return math_number(clip, sig=2)


def integer_cell(value):
    x = finite_float(value)
    if x is None:
        return r"\textemdash{}"
    if math.isclose(x, round(x), rel_tol=0.0, abs_tol=1.0e-12):
        return str(int(round(x)))
    return latex_escape(x)


def shared_settings_note(rows):
    if not rows:
        return ""

    candidates = [
        ("message_passing_layers", lambda v: "{0} message-passing layers".format(int(v))),
        ("hidden_dim", lambda v: "{0} hidden channels".format(int(v))),
        (
            "activation",
            lambda v: "{0} activations".format("ReLU" if str(v).lower() == "relu" else latex_escape(v)),
        ),
        (
            "dropout",
            lambda v: "zero dropout" if finite_float(v) == 0.0 else "dropout {0}".format(math_number(v, sig=2)),
        ),
        ("batch_size", lambda v: "batch size {0}".format(int(v))),
        ("optimizer", lambda v: "{0} optimizer".format(latex_escape(v))),
        ("weight_decay", lambda v: "weight decay {0}".format(math_number(v, sig=2))),
        ("rollout_steps", lambda v: "{0}-step rollout training".format(int(v))),
        ("target_mode", lambda v: "{0}-state targets".format(latex_escape(v))),
        ("predict_type", lambda v: "{0}-state prediction".format(latex_escape(v))),
    ]

    phrases = []
    for key, formatter in candidates:
        values = [row.get(key) for row in rows]
        if any(value is None for value in values):
            continue
        first = values[0]
        if all(value == first for value in values):
            phrases.append(formatter(first))

    disabled = []
    if all(row.get("use_skip") is False for row in rows):
        disabled.append("skip connections")
    if all(row.get("use_layernorm") is False for row in rows):
        disabled.append("layer normalization")
    if all(row.get("gradient_weighted_loss_enabled") in [False, None] for row in rows):
        disabled.append("gradient-weighted loss")
    if all(row.get("mass_conservation_loss_enabled") in [False, None] for row in rows):
        disabled.append("mass-conservation loss")

    note = ""
    if phrases:
        note = "Shared settings: " + ", ".join(phrases) + "."
    if disabled:
        note += " Disabled in all runs: " + ", ".join(disabled) + "."
    if all(state_normalization_value(row) == "minmax" for row in rows):
        note += r" The scalar state input was normalized to $[-1,1]$"
        if all(row.get("normalize_state_clip") is False for row in rows):
            note += " without clipping"
        note += " in all runs."
    if all(
        (row.get("input_features") or {}).get("use_pos", (row.get("input_features") or {}).get("include_pos"))
        is False
        for row in rows
    ):
        note += " Position was not included as an input feature."
    note += (
        " Epochs trained denotes the number of training epochs completed; Best epoch denotes the "
        "validation-selected checkpoint used for plots and metrics."
    )
    if any(row.get("epochs_trained") is None for row in rows):
        note += " An em dash in Epochs trained indicates that the training history is unavailable."
    if any((finite_float(row.get("flux_loss_weight")) or 0.0) > 0.0 for row in rows):
        note += " Flux loss denotes supervised interface-flux loss weight."
    return note


def position_feature_value(row):
    explicit = row.get("include_absolute_positions")
    if explicit is not None:
        return bool(explicit)
    features = row.get("input_features") or {}
    return features.get("use_pos", features.get("include_pos"))


def setting_name(value):
    if value is None:
        return None
    return str(value).strip().lower() or None


def activation_value(row):
    explicit = setting_name(row.get("activation"))
    if explicit is not None:
        return explicit
    return DATASET_SETTING_DEFAULTS.get(row.get("dataset"), {}).get("activation")


def activation_text(value):
    return {"relu": "ReLU", "silu": "SiLU", "gelu": "GELU"}.get(
        value, latex_escape(value)
    )


def state_normalization_value(row):
    """Prefer recorded methods/flags/statistics to benchmark defaults."""
    explicit = setting_name(row.get("state_normalization"))
    if explicit is not None:
        aliases = {
            "min-max": "minmax",
            "min_max": "minmax",
            "standardize": "standardization",
            "standardisation": "standardization",
            "zscore": "standardization",
            "z-score": "standardization",
        }
        return aliases.get(explicit, explicit)

    scaling = row.get("normalize_state_to_minus1_1")
    if scaling is True:
        return "minmax"

    normalization = row.get("normalization") or {}
    if isinstance(normalization, dict):
        mean = finite_float(normalization.get("density_mean"))
        std = finite_float(normalization.get("density_std"))
        if mean is not None and std is not None and std > 0.0:
            return "standardization"

    if scaling is False:
        # This flag rules out min-max scaling but does not identify another method.
        return "not_minmax"
    return DATASET_SETTING_DEFAULTS.get(row.get("dataset"), {}).get("state_normalization")


def state_normalization_text(value):
    labels = {
        "minmax": r"\shortstack{Scaling to\\$[-1,1]$}",
        "standardization": "Standardization",
        "none": "None",
        "not_minmax": r"No $[-1,1]$ scaling",
    }
    return labels.get(value, latex_escape(value))


def dataset_setting_cell(rows, key, formatter):
    """Report a shared value only when it is known for every run in a dataset."""
    values = [key(row) if callable(key) else row.get(key) for row in rows]
    if not values or any(value is None or value == "" for value in values):
        return r"\textemdash{}"
    if not all(value == values[0] for value in values):
        return "Varies"
    return formatter(values[0])


def build_dataset_table(data, caption, label, use_resizebox=True):
    rows = list(data.get("rows", []))
    groups = grouped_rows(rows)
    settings = [
        ("Message-passing layers", "message_passing_layers", integer_cell),
        ("Activation", activation_value, activation_text),
        ("Dropout", "dropout", math_number),
        ("Batch size", "batch_size", integer_cell),
        ("Weight decay", "weight_decay", math_number),
        ("Training rollout steps", "rollout_steps", integer_cell),
        ("Selection metric", selection_metric_cell, lambda value: value),
        ("Learning rate (LR)", lr_schedule_cell, lambda value: value),
        ("TV loss weight", tv_loss_cell, lambda value: value),
        ("Grad. clip", grad_clip_cell, lambda value: value),
        ("State target", "target_mode", latex_escape),
        ("Absolute position inputs", position_feature_value, bool_text),
        ("State normalization", state_normalization_value, state_normalization_text),
    ]
    lines = [
        r"% Auto-generated by scripts/make_hyperparameter_table.py from {0}".format(
            data.get("source_script", "model_training_hyperparameters.json")
        ),
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{" + caption + "}",
        r"\label{" + label + "}",
    ]
    if use_resizebox:
        lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.extend([
        r"\begin{tabular}{l" + "c" * len(groups) + "}",
        r"\toprule",
        " & ".join(["Setting"] + [dataset_latex_name(name) for name, _ in groups]) + r" \\",
        r"\midrule",
    ])
    cells = []
    for title, key, formatter in settings:
        values = [dataset_setting_cell(rows, key, formatter) for _, rows in groups]
        cells.extend(values)
        lines.append(" & ".join([title] + values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    if use_resizebox:
        lines.append("}")
    notes = [
        "Selection metric denotes the validation quantity minimized for checkpoint "
        "selection and, when enabled, early stopping."
    ]
    if any(row.get("lr_schedule_enabled") and row.get("lr_schedule") for row in rows):
        notes.append("LR changes are shown with their scheduled epochs.")
    if any(row.get("tv_loss_enabled") is True for row in rows):
        notes.append(
            "When enabled, the TV loss penalizes only prediction total variation "
            "above the reference total variation."
        )
    if any(state_normalization_value(row) == "standardization" for row in rows):
        notes.append(r"Standardization subtracts the training-set mean and divides by its standard deviation.")
    if r"\textemdash{}" in cells:
        notes.append("An em dash indicates that the setting is not recorded for every run in that dataset.")
    if "Varies" in cells:
        notes.append("Varies indicates different recorded values among runs within a dataset.")
    if notes:
        lines.append(r"\par\smallskip\footnotesize " + " ".join(notes))
    lines.extend([r"\end{table}", ""])
    return "\n".join(lines)


def build_table(data, caption, label, use_resizebox=True):
    rows = list(data.get("rows", []))
    if any("epochs_trained" not in row for row in rows):
        raise ValueError(
            "Input JSON lacks epochs_trained. Regenerate it with "
            "make_results_metrics.py or make_model_hyperparameters.py before formatting the table."
        )
    source = data.get("source_script", "model_training_hyperparameters.json")
    lines = [
        r"% Auto-generated by scripts/make_hyperparameter_table.py from {0}".format(
            source
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
            r"\begin{tabular}{llcccc}",
            r"\toprule",
            r"Dataset & Method & \shortstack{Physics\\input} & \shortstack{Epochs\\trained} & Best epoch & \shortstack{Flux loss\\weight} \\",
            r"\midrule",
        ]
    )

    groups = grouped_rows(rows)
    for group_index, (dataset, group) in enumerate(groups):
        if group_index > 0:
            lines.append(r"\midrule")
        for row_index, row in enumerate(group):
            #dataset_text = latex_escape(dataset_display_name(dataset)) if row_index == 0 else ""
            dataset_text = dataset_latex_name(dataset) if row_index == 0 else ""
            method_text = method_latex_name(row.get("method", ""))
            lines.append(
                "{0} & {1} & {2} & {3} & {4} & {5} \\\\".format(
                    dataset_text,
                    method_text,
                    physics_input_cell(row),
                    integer_cell(row.get("epochs_trained")),
                    integer_cell(row.get("best_epoch")),
                    flux_loss_cell(row),
                )
            )

    lines.extend([r"\bottomrule", r"\end{tabular}"])
    if use_resizebox:
        lines.append(r"}")

    shared_note = shared_settings_note(rows)
    if shared_note:
        lines.append(r"\par\smallskip\footnotesize " + shared_note)
    lines.append(r"\end{table}")
    lines.append("")
    return "\n".join(lines)


def main():
    args = parse_args()
    with args.input.open("r") as fh:
        data = json.load(fh)

    table = build_table(
        data,
        caption=args.caption,
        label=args.label,
        use_resizebox=not args.no_resizebox,
    )
    dataset_table = build_dataset_table(
        data,
        caption=args.dataset_caption,
        label=args.dataset_label,
        use_resizebox=not args.no_resizebox,
    )
    for path, content in ((args.output, table), (args.dataset_output, dataset_table)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            fh.write(content)
        print("Wrote {0}".format(path))


if __name__ == "__main__":
    main()
