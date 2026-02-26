#!/usr/bin/env python
"""
Generate all LaTeX tables for the CLEAR paper.

Run from docs/scripts/ and outputs to docs/tex_tbls/.
Reproduces the exact same tables as in paper/overleaf/tex_tbls/.

Usage:
    python generate_tables.py

Output structure:
    docs/tex_tbls/
    ├── table-combined-dataset-stats.tex
    ├── de_sqr/
    │   ├── table-de-sqr-95-{metric}.tex  (9 metrics)
    │   └── table-sqr-de-calibration-95.tex
    └── pcs_cqr/
        ├── standard/{a,b,c}/
        │   ├── table-combined-95-{metric}-final_standard.tex  (9 metrics)
        │   ├── table-combined-95-gamma-lambda-final_standard.tex
        │   └── table-uncertainty-metrics-95_standard.tex
        └── conformalized/{a,b,c}/
            ├── table-combined-95-{metric}-final_conformalized.tex
            ├── table-combined-95-gamma-lambda-final_conformalized.tex
            └── table-uncertainty-metrics-95_conformalized.tex
"""

import os
import sys
import glob
import re
import shutil
import subprocess
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, '..', '..'))  # repo root
OUTPUT_BASE = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'tex_tbls'))

# Add paths for imports
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

from utils import format_metric_name, write_latex_table
from combine_real_benchmark_results import (
    process_multiple_result_folders,
    combine_metrics_tables,
    combine_gamma_lambda_tables,
    generate_uncertainty_metrics_table,
)
from combine_improved_de_sqr import (
    load_de_sqr_results,
    aggregate_metrics_for_methods,
    generate_metric_table,
    generate_calibration_parameters_table,
    generate_summary_statistics,
)

# ── constants ──────────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
DATA_DIR = os.path.join(BASE_DIR, 'data')

VARIANT_FOLDERS = [
    "qPCS_all_10seeds_all",     # a
    "qPCS_qxgb_10seeds_qxgb",  # b
    "PCS_all_10seeds_qrf",      # c
]
VARIANT_LABELS = ["a", "b", "c"]

COVERAGE = 95

DATASETS = [
    "ailerons", "airfoil", "allstate", "ca_housing", "computer",
    "concrete", "elevator", "energy_efficiency", "insurance", "kin8nm",
    "miami_housing", "naval_propulsion", "parkinsons", "powerplant",
    "qsar", "sulfur", "superconductor",
]

METRICS_CONFIG = {
    "picp":              {"decimals": 2, "higher_is_better": True},
    "niw":               {"decimals": 3, "higher_is_better": False},
    "mpiw":              {"decimals": 3, "higher_is_better": False},
    "quantileloss":      {"decimals": 3, "higher_is_better": False},
    "crps":              {"decimals": 3, "higher_is_better": False},
    "nciw":              {"decimals": 3, "higher_is_better": False},
    "expectileloss":     {"decimals": 3, "higher_is_better": False},
    "intervalscoreloss": {"decimals": 3, "higher_is_better": False},
}

# ── helpers ────────────────────────────────────────────────────────────────────

def add_setting_suffix(directory, setting):
    """
    Rename .tex files and update internal \\label to add _standard / _conformalized suffix.
    Only renames .tex files (skips .csv).
    """
    for tex_file in glob.glob(os.path.join(directory, '*.tex')):
        # Skip files that already have the suffix
        if tex_file.endswith(f"_{setting}.tex"):
            continue

        with open(tex_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Insert _{setting} before _variant_ in labels
        content = re.sub(
            r'(\\label\{[^}]+?)(_variant_)',
            rf'\1_{setting}\2',
            content,
        )

        # Rename file: foo.tex → foo_{setting}.tex
        base, ext = os.path.splitext(tex_file)
        new_path = f"{base}_{setting}{ext}"

        # Remove existing file if present (overwrite)
        if os.path.exists(new_path):
            os.remove(new_path)

        with open(new_path, 'w', encoding='utf-8', newline='\n') as f:
            f.write(content)

        # Remove the original (unsuffixed) file
        os.remove(tex_file)


def load_csv_data_for_metric(csv_dir, metric, methods_mapping, coverage=95):
    """
    Load benchmark CSV files from a directory and return aggregated data
    for the given metric and method mapping.

    Args:
        csv_dir: Directory containing benchmark_results_*_{coverage}.csv
        metric: Metric name (lowercase, e.g. 'nciw')
        methods_mapping: list of (csv_method, display_name) pairs
        coverage: Coverage percentage

    Returns:
        dict: {dataset: {display_name: {'mean': float, 'std': float}}}
    """
    csv_pattern = os.path.join(csv_dir, f"benchmark_results_*_{coverage}.csv")
    csv_files = glob.glob(csv_pattern)
    data = {}

    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        dataset_match = re.search(r'benchmark_results_(.+)_' + str(coverage), filename)
        if not dataset_match:
            continue

        dataset_key = dataset_match.group(1)  # e.g. data_ailerons
        display_dataset = dataset_key[5:] if dataset_key.startswith("data_") else dataset_key

        df = pd.read_csv(csv_file)

        if display_dataset not in data:
            data[display_dataset] = {}

        for csv_method, display_name in methods_mapping:
            rows = df[
                (df['Dataset'] == dataset_key) &
                (df['Method'] == csv_method) &
                (df['Metric'] == metric)
            ]['Value'].dropna().values

            if len(rows) > 0:
                data[display_dataset][display_name] = {
                    'mean': float(np.mean(rows)),
                    'std': float(np.std(rows)),
                    'values': rows.tolist(),
                }
            else:
                data[display_dataset][display_name] = {
                    'mean': np.nan, 'std': np.nan, 'values': [],
                }

    return data


def load_uacqr_data_for_metric(uacqr_csv, metric, methods, coverage=95):
    """Load UACQR data for a given metric from the aggregated CSV."""
    uacqr_col_map = {
        "picp": "PICP", "niw": "NIW", "mpiw": "MPIW",
        "quantileloss": "QuantileLoss", "crps": "CRPS", "auc": "AUC",
        "nciw": "NCIW", "expectileloss": "ExpectileLoss",
        "intervalscoreloss": "IntervalScoreLoss",
    }
    col = uacqr_col_map.get(metric)
    if col is None:
        return {}

    df = pd.read_csv(uacqr_csv)
    # Filter for the target coverage
    if 'Coverage_Target' in df.columns:
        df = df[df['Coverage_Target'] == coverage / 100.0]

    data = {}
    for method in methods:
        subset = df[df['Method'] == method]
        for _, row in subset.iterrows():
            dataset = row['Dataset']
            if dataset not in data:
                data[dataset] = {}
            val = row.get(col)
            if val is not None and not pd.isna(val):
                if method not in data[dataset]:
                    data[dataset][method] = {'values': []}
                data[dataset][method]['values'].append(float(val))

    # Compute mean/std (inf values propagate to mean, displayed as +infty)
    for dataset in data:
        for method in list(data[dataset].keys()):
            vals = data[dataset][method]['values']
            if vals:
                n_inf = sum(1 for v in vals if np.isinf(v))
                mean_val = float(np.mean(vals))
                # std of inf values is undefined (inf-inf=nan), set to nan directly
                if n_inf > 0:
                    std_val = np.nan
                else:
                    std_val = float(np.std(vals))
                data[dataset][method]['mean'] = mean_val
                data[dataset][method]['std'] = std_val
                data[dataset][method]['n_inf'] = n_inf
                data[dataset][method]['n_total'] = len(vals)
            else:
                data[dataset][method]['mean'] = np.nan
                data[dataset][method]['std'] = np.nan
                data[dataset][method]['n_inf'] = 0
                data[dataset][method]['n_total'] = 0

    return data


# ── DE / SQR tables ───────────────────────────────────────────────────────────

def generate_de_sqr_tables():
    """Generate DE/SQR benchmark tables (10 files)."""
    print("\n" + "=" * 60)
    print("=== Generating DE/SQR Tables ===")
    print("=" * 60)

    output_dir = os.path.join(OUTPUT_BASE, 'de_sqr')
    os.makedirs(output_dir, exist_ok=True)

    results_dir = os.path.join(RESULTS_DIR, 'de_sqr')
    dataset_results = load_de_sqr_results(results_dir, COVERAGE)

    if not dataset_results:
        print("No DE/SQR results found!")
        return

    methods_mapping = {
        'CLEAR': 'CLEAR',
        'DE': 'DE_calibrated',
        'SQR': 'SQR_uncalibrated',
        'DE-conformal': 'DE_conformal',
        'SQR-conformal': 'SQR_conformal',
    }

    aggregated = aggregate_metrics_for_methods(dataset_results, methods_mapping)

    metrics = [
        'PICP', 'NIW', 'MPIW', 'QuantileLoss', 'ExpectileLoss',
        'CRPS', 'NCIW', 'IntervalScoreLoss',
    ]

    for metric in metrics:
        generate_metric_table(aggregated, metric, COVERAGE, output_dir)

    generate_calibration_parameters_table(aggregated, COVERAGE, output_dir)
    generate_summary_statistics(aggregated, COVERAGE, output_dir)


# ── Standard PCS / CQR tables ─────────────────────────────────────────────────

def generate_standard_tables():
    """Generate standard PCS/CQR variant tables (33 files: 11 per variant)."""
    print("\n" + "=" * 60)
    print("=== Generating Standard PCS/CQR Tables ===")
    print("=" * 60)

    variant_paths = [
        os.path.join(RESULTS_DIR, 'standard', folder)
        for folder in VARIANT_FOLDERS
    ]

    # Use a temp directory as output, then move + rename
    temp_output = os.path.join(OUTPUT_BASE, '_temp_standard')

    process_multiple_result_folders(
        variant_paths,
        temp_output,
        VARIANT_LABELS,
        coverage=COVERAGE,
        method_set='final',
        landscape_mode=False,
        uacqr_agg_csv=None,
    )

    # Move and rename with _standard suffix
    for label in VARIANT_LABELS:
        src_dir = os.path.join(temp_output, label)
        dst_dir = os.path.join(OUTPUT_BASE, 'pcs_cqr', 'standard', label)
        os.makedirs(dst_dir, exist_ok=True)

        if os.path.exists(src_dir):
            # Copy generated files (without suffix) to destination
            for f in os.listdir(src_dir):
                src_file = os.path.join(src_dir, f)
                dst_file = os.path.join(dst_dir, f)
                shutil.copy2(src_file, dst_file)
            # Add _standard suffix to .tex files (handles existing files)
            add_setting_suffix(dst_dir, 'standard')

    shutil.rmtree(temp_output, ignore_errors=True)


# ── Conformalized PCS / CQR tables ────────────────────────────────────────────

def _generate_conformalized_metric_table(
    combined_data,
    method_info,
    metric,
    variant_label,
    output_dir,
):
    """Generate a single conformalized metric table."""
    cfg = METRICS_CONFIG.get(metric, {"decimals": 3, "higher_is_better": False})
    decimals = cfg["decimals"]
    higher_is_better = cfg["higher_is_better"]
    formatted_metric = format_metric_name(metric)

    max_runs = 0
    for dataset in combined_data:
        for method_display in combined_data[dataset]:
            vals = combined_data[dataset][method_display].get('values', [])
            max_runs = max(max_runs, len(vals))

    # Check if any method/dataset has inf values for this metric
    has_inf = any(
        np.isinf(combined_data[ds][m].get('mean', 0))
        for ds in combined_data for m in combined_data[ds]
    )

    caption = (
        f"Conformalized Variant ({variant_label}) {formatted_metric} at "
        f"{COVERAGE}\\% prediction intervals, aggregated across {max_runs} seeds. "
        f"Methods with suffix `-c' denote conformalized variants obtained using "
        f"the validation set divided into two parts, one for validation and one "
        f"for calibration."
    )

    if metric != "picp":
        caption += (
            "Values $\\geq 100$ or $< 0.01$ are presented in scientific notation "
            "with 1 decimal place. "
        )
        inf_note = (
            " $+\\infty$ indicates diverged predictions; the superscript "
            "denotes the number of affected seeds out of the total."
        ) if has_inf else ""
        if higher_is_better:
            caption += (
                "\\textbf{Bold} values (desirable) are the maximum for that "
                "dataset and metric, while the \\underline{underlined} values "
                "indicate the second-best result. "
                "\\textcolor{red}{Red} values are more than 33\\% worse than "
                "the best result." + inf_note
            )
        else:
            caption += (
                "\\textbf{Bold} values (desirable) are the minimum for that "
                "dataset and metric, while the \\underline{underlined} values "
                "indicate the second-best result. "
                "\\textcolor{red}{Red} values are more than 33\\% worse than "
                "the best result." + inf_note
            )

    method_display_names = [m[1] for m in method_info]

    table_lines = [
        "\\begin{table}[!htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{tab:combined_{metric}_{COVERAGE}_final_conformalized_variant_{variant_label}}}",
        "\\small",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{l" + "c" * len(method_display_names) + "}",
        "\\toprule",
        "Dataset & " + " & ".join(method_display_names) + " \\\\",
        "\\midrule",
    ]

    for dataset in sorted(combined_data.keys()):
        formatted_dataset = dataset.replace("_", "\\textunderscore ")
        row = [formatted_dataset]

        # Rank methods
        values_methods = []
        for _, display_name in method_info:
            if display_name in combined_data[dataset]:
                m = combined_data[dataset][display_name]['mean']
                if np.isfinite(m):
                    values_methods.append((m, display_name))
        values_methods.sort(key=lambda x: x[0], reverse=higher_is_better)
        best_method = values_methods[0][1] if values_methods else None
        best_val = values_methods[0][0] if values_methods else np.nan
        second_best = values_methods[1][1] if len(values_methods) > 1 else None

        for _, display_name in method_info:
            if display_name in combined_data[dataset]:
                mean_val = combined_data[dataset][display_name]['mean']
                std_val = combined_data[dataset][display_name]['std']

                if not np.isfinite(mean_val):
                    if np.isinf(mean_val):
                        n_inf = combined_data[dataset][display_name].get('n_inf', 0)
                        n_total = combined_data[dataset][display_name].get('n_total', 0)
                        if n_inf > 0 and n_total > 0:
                            row.append(f"$+\\infty$\\textsuperscript{{{n_inf}/{n_total}}}")
                        else:
                            row.append("$+\\infty$")
                    else:
                        row.append("-")
                    continue

                if metric == "picp":
                    use_sci = False
                else:
                    use_sci = abs(mean_val) >= 100 or abs(mean_val) < 0.01

                fmt_mean = f"{mean_val:.1e}" if use_sci else f"{mean_val:.{decimals}f}"

                if use_sci:
                    # When mean triggers scientific notation, also check std
                    if abs(std_val) >= 100 or abs(std_val) < 0.01:
                        fmt_std = f"{std_val:.1e}"
                    else:
                        fmt_std = f"{std_val:.{decimals}f}"
                else:
                    # When mean is in normal range, always use normal notation for std
                    fmt_std = f"{std_val:.{decimals}f}"

                cell = f"{fmt_mean} $\\pm$ {fmt_std}"

                if metric != "picp":
                    if display_name == best_method:
                        cell = f"\\textbf{{{cell}}}"
                    elif display_name == second_best:
                        cell = f"\\underline{{{cell}}}"
                    elif np.isfinite(best_val) and np.isfinite(mean_val):
                        if higher_is_better:
                            if best_val != 0 and mean_val < best_val * 0.77:
                                cell = f"\\textcolor{{red}}{{{cell}}}"
                            elif best_val == 0 and mean_val < 0:
                                cell = f"\\textcolor{{red}}{{{cell}}}"
                        else:
                            if best_val > 0 and mean_val > best_val * 1.33:
                                cell = f"\\textcolor{{red}}{{{cell}}}"
                            elif best_val == 0 and mean_val > 0:
                                cell = f"\\textcolor{{red}}{{{cell}}}"
                            elif best_val < 0:
                                if mean_val > best_val + abs(best_val) * 0.33:
                                    cell = f"\\textcolor{{red}}{{{cell}}}"

                row.append(cell)
            else:
                row.append("-")

        table_lines.append(" & ".join(row) + " \\\\")

    table_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "}",
        "\\end{table}",
    ])

    output_file = os.path.join(
        output_dir,
        f"table-combined-{COVERAGE}-{metric}-final_conformalized.tex",
    )
    write_latex_table(table_lines, output_file)
    print(f"  Created conformalized table for {metric}")


def _generate_conformalized_gamma_lambda_table(csv_dir, variant_label, output_dir):
    """Generate conformalized gamma-lambda table."""
    method_for_params = 'clear'
    datasets_with_data = {}
    max_runs = 0

    csv_pattern = os.path.join(csv_dir, f"benchmark_results_*_{COVERAGE}.csv")
    for csv_file in glob.glob(csv_pattern):
        filename = os.path.basename(csv_file)
        match = re.search(r'benchmark_results_(.+)_' + str(COVERAGE), filename)
        if not match:
            continue
        dataset_key = match.group(1)
        display = dataset_key[5:] if dataset_key.startswith("data_") else dataset_key

        df = pd.read_csv(csv_file)
        lam = df[(df['Dataset'] == dataset_key) & (df['Method'] == method_for_params) & (df['Metric'] == 'lambda')]['Value'].dropna().tolist()
        gam = df[(df['Dataset'] == dataset_key) & (df['Method'] == method_for_params) & (df['Metric'] == 'gamma')]['Value'].dropna().tolist()

        if lam and gam:
            max_runs = max(max_runs, len(lam), len(gam))
            datasets_with_data[display] = {
                'lambda': f"{np.median(lam):.2f} [{min(lam):.2f}:{max(lam):.2f}]",
                'gamma': f"{np.median(gam):.2f} [{min(gam):.2f}:{max(gam):.2f}]",
            }

    if not datasets_with_data:
        print(f"  No gamma-lambda data for conformalized variant {variant_label}")
        return list(datasets_with_data.keys())

    caption = (
        f"Conformalized Variant ({variant_label}) CLEAR calibration parameters "
        f"$\\lambda$ and $\\gamma_1$ for {COVERAGE}\\% prediction intervals "
        f"across {max_runs} seeds. Using all available variables. "
        f"Showing median [min:max] values."
    )

    table_lines = [
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\caption{" + caption + "}",
        f"\\label{{tab:combined_gamma_lambda_{COVERAGE}_final_conformalized_variant_{variant_label}}}",
        "\\small",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Dataset & $\\lambda$ & $\\gamma_1$ \\\\",
        "\\midrule",
    ]

    for dataset in sorted(datasets_with_data.keys()):
        d = datasets_with_data[dataset]
        esc = dataset.replace("_", "\\textunderscore ")
        table_lines.append(f"{esc} & {d['lambda']} & {d['gamma']} \\\\")

    table_lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])

    out = os.path.join(output_dir, f"table-combined-{COVERAGE}-gamma-lambda-final_conformalized.tex")
    write_latex_table(table_lines, out)
    print(f"  Created conformalized gamma-lambda table for variant {variant_label}")
    return list(datasets_with_data.keys())


def _generate_conformalized_uncertainty_table(csv_dir, datasets, variant_label, output_dir):
    """Generate conformalized uncertainty metrics table."""
    method_key = 'clear'
    uncertainty_data = {}
    max_runs = 0

    for dataset in datasets:
        csv_dataset = f"data_{dataset}" if not dataset.startswith("data_") else dataset
        csv_files = glob.glob(os.path.join(csv_dir, f"benchmark_results_{csv_dataset}_{COVERAGE}.csv"))
        if not csv_files:
            continue

        df = pd.read_csv(csv_files[0])

        aleatoric = df[(df['Dataset'] == csv_dataset) & (df['Method'] == method_key) & (df['Metric'] == 'total_aleatoric_calib')]['Value'].tolist()
        epistemic = df[(df['Dataset'] == csv_dataset) & (df['Method'] == method_key) & (df['Metric'] == 'total_epistemic_calib')]['Value'].tolist()
        ratio = df[(df['Dataset'] == csv_dataset) & (df['Method'] == method_key) & (df['Metric'] == 'uncertainty_ratio_calib')]['Value'].tolist()

        if aleatoric and epistemic and ratio:
            max_runs = max(max_runs, len(aleatoric))
            uncertainty_data[dataset] = {
                'aleatoric': {'median': np.median(aleatoric), 'min': min(aleatoric), 'max': max(aleatoric)},
                'epistemic': {'median': np.median(epistemic), 'min': min(epistemic), 'max': max(epistemic)},
                'ratio': {'median': np.median(ratio), 'min': min(ratio), 'max': max(ratio)},
            }

    if not uncertainty_data:
        print(f"  No uncertainty data for conformalized variant {variant_label}")
        return

    caption = (
        f"Conformalized Variant ({variant_label}) uncertainty metrics for "
        f"{COVERAGE}\\% prediction intervals across {max_runs} seeds. "
        f"Values are shown as median [min:max]. "
        f"A = aleatoric uncertainty, E = epistemic uncertainty, E/A = uncertainty ratio."
    )

    table_lines = [
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\caption{" + caption + "}",
        f"\\label{{tab:uncertainty_metrics_{COVERAGE}_conformalized_variant_{variant_label}}}",
        "\\small",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Dataset & \\multicolumn{3}{c}{CLEAR} \\\\",
        "\\cmidrule(lr){2-4}",
        " & A & E & E/A \\\\",
        "\\midrule",
    ]

    for dataset in sorted(uncertainty_data.keys()):
        d = uncertainty_data[dataset]
        esc = dataset.replace("_", "\\textunderscore ")
        a = d['aleatoric']
        e = d['epistemic']
        r = d['ratio']
        table_lines.append(
            f"{esc} & "
            f"{a['median']:.2f} [{a['min']:.2f}:{a['max']:.2f}] & "
            f"{e['median']:.2f} [{e['min']:.2f}:{e['max']:.2f}] & "
            f"{r['median']:.2f} [{r['min']:.2f}:{r['max']:.2f}] \\\\"
        )

    table_lines.extend(["\\bottomrule", "\\end{tabular}", "}", "\\end{table}"])

    out = os.path.join(output_dir, f"table-uncertainty-metrics-{COVERAGE}_conformalized.tex")
    write_latex_table(table_lines, out)
    print(f"  Created conformalized uncertainty table for variant {variant_label}")


def generate_conformalized_tables():
    """
    Generate conformalized PCS/CQR variant tables (33 files: 11 per variant).

    Conformalized tables combine data from TWO sources:
    - Conformalized CSV: CLEAR-c, PCS-EPISTEMIC-c, ALEATORIC-R-c
    - Standard CSV: CLEAR (baseline)
    For variant c, also UACQR-P and UACQR-S from aggregated CSV.
    """
    print("\n" + "=" * 60)
    print("=== Generating Conformalized PCS/CQR Tables ===")
    print("=" * 60)

    uacqr_csv = os.path.join(RESULTS_DIR, 'uacqr', 'uacqr_benchmark_results_conformalized.csv')
    has_uacqr = os.path.exists(uacqr_csv)

    for i, (folder, label) in enumerate(zip(VARIANT_FOLDERS, VARIANT_LABELS)):
        print(f"\n--- Variant {label} ---")

        conf_csv_dir = os.path.join(RESULTS_DIR, 'conformalized', folder)
        std_csv_dir = os.path.join(RESULTS_DIR, 'standard', folder)
        output_dir = os.path.join(OUTPUT_BASE, 'pcs_cqr', 'conformalized', label)
        os.makedirs(output_dir, exist_ok=True)

        # Define method info for this variant
        # (csv_method, source, display_name)
        if label == 'c' and has_uacqr:
            method_sources = [
                ("clear",        "conformalized", "CLEAR-c"),
                ("pcs",          "conformalized", "PCS-EPISTEMIC-c"),
                ("cqr_residual", "conformalized", "ALEATORIC-R-c"),
                ("clear",        "standard",      "CLEAR"),
                ("UACQR-P",     "uacqr",         "UACQR-P"),
                ("UACQR-S",     "uacqr",         "UACQR-S"),
            ]
        else:
            method_sources = [
                ("clear",        "conformalized", "CLEAR-c"),
                ("pcs",          "conformalized", "PCS-EPISTEMIC-c"),
                ("cqr_residual", "conformalized", "ALEATORIC-R-c"),
                ("clear",        "standard",      "CLEAR"),
            ]

        method_info = [(ms[0], ms[2]) for ms in method_sources]  # (csv_method, display_name)

        # Generate each metric table
        for metric in METRICS_CONFIG:
            # Load data from conformalized CSV
            conf_methods = [(m, d) for m, s, d in method_sources if s == "conformalized"]
            conf_data = load_csv_data_for_metric(conf_csv_dir, metric, conf_methods, COVERAGE)

            # Load data from standard CSV
            std_methods = [(m, d) for m, s, d in method_sources if s == "standard"]
            std_data = load_csv_data_for_metric(std_csv_dir, metric, std_methods, COVERAGE)

            # Load UACQR data
            uacqr_methods = [d for m, s, d in method_sources if s == "uacqr"]
            uacqr_data = {}
            if uacqr_methods and has_uacqr:
                uacqr_data = load_uacqr_data_for_metric(uacqr_csv, metric, uacqr_methods, COVERAGE)

            # Combine all data into one dict
            combined = {}
            for dataset in set(list(conf_data.keys()) + list(std_data.keys())):
                combined[dataset] = {}
                if dataset in conf_data:
                    combined[dataset].update(conf_data[dataset])
                if dataset in std_data:
                    combined[dataset].update(std_data[dataset])
                if dataset in uacqr_data:
                    combined[dataset].update(uacqr_data[dataset])

            _generate_conformalized_metric_table(
                combined, method_info, metric, label, output_dir,
            )

        # Gamma-lambda table (from conformalized CSV)
        valid_datasets = _generate_conformalized_gamma_lambda_table(
            conf_csv_dir, label, output_dir,
        )

        # Uncertainty table (from conformalized CSV)
        if valid_datasets:
            _generate_conformalized_uncertainty_table(
                conf_csv_dir, valid_datasets, label, output_dir,
            )


# ── Dataset statistics table ──────────────────────────────────────────────────

def generate_dataset_stats_table():
    """
    Generate dataset statistics table from raw data CSV files.
    Produces table-combined-dataset-stats.tex.

    NOTE: The reference table was originally computed from pickle files
    containing the exact train/val/test splits used during experiments.
    Computing from raw CSVs gives slightly different min/max/range values
    for some datasets. If the reference table already exists, we skip
    regeneration to preserve the correct values.
    """
    print("\n" + "=" * 60)
    print("=== Generating Dataset Statistics Table ===")
    print("=" * 60)

    output_file = os.path.join(OUTPUT_BASE, "table-combined-dataset-stats.tex")
    if os.path.exists(output_file):
        print(f"  Dataset statistics table already exists, skipping regeneration")
        print(f"  (Reference values come from experiment splits, not raw CSVs)")
        return

    all_stats = []
    for dataset in DATASETS:
        data_path = os.path.join(DATA_DIR, f"data_{dataset}")
        x_file = os.path.join(data_path, "X.csv")
        y_file = os.path.join(data_path, "y.csv")

        if not os.path.exists(x_file) or not os.path.exists(y_file):
            print(f"  Skipping {dataset}: data files not found")
            continue

        X = pd.read_csv(x_file)
        y = pd.read_csv(y_file, header=None).values.flatten()

        all_stats.append({
            "dataset": dataset,
            "n_samples": len(y),
            "n_features": X.shape[1],
            "y_min": float(np.min(y)),
            "y_max": float(np.max(y)),
            "y_range": float(np.max(y) - np.min(y)),
        })

    all_stats.sort(key=lambda x: x["dataset"])

    table_lines = [
        "\\begin{table}[!htbp]",
        "\\centering",
        r"\caption{Dataset statistics where $d$ represents the number of variables, "
        r"$n$ represents the number of observations, followed by the minimum, maximum, "
        r"and range values for $y$.}",
        "\\label{tab:dataset_stats}",
        "\\small",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Dataset & $n$ & $d$ & $y_{min}$ & $y_{max}$ & $y_{range}$ \\\\",
        "\\midrule",
    ]

    for s in all_stats:
        dataset_esc = s["dataset"].replace("_", "\\textunderscore ")
        n = f"{s['n_samples']:,}"
        d = str(s["n_features"])

        def fmt_val(v):
            if abs(v) < 0.001 or abs(v) > 10000:
                return f"{v:.2e}"
            return f"{v:.4f}"

        table_lines.append(
            f"{dataset_esc} & {n} & {d} & {fmt_val(s['y_min'])} & "
            f"{fmt_val(s['y_max'])} & {fmt_val(s['y_range'])} \\\\"
        )

    table_lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])

    output_file = os.path.join(OUTPUT_BASE, "table-combined-dataset-stats.tex")
    write_latex_table(table_lines, output_file)
    print(f"  Created dataset statistics table at {output_file}")


# ── CLEAR vs UACQR improvement table ──────────────────────────────────────────

def generate_uacqr_improvement_table(setting="standard"):
    """
    Generate a table showing CLEAR variant (c) percentage improvement over
    UACQR-S and UACQR-P for Average Width (MPIW) and AISL (IntervalScoreLoss).

    Args:
        setting: "standard" or "conformalized". Controls which CLEAR data
                 is used. UACQR baselines are always the same.
                 Use "standard" for the main paper table and "conformalized"
                 for the appendix table.
    """
    print("\n" + "=" * 60)
    print(f"=== Generating CLEAR vs UACQR Improvement Table ({setting}) ===")
    print("=" * 60)

    uacqr_csv = os.path.join(RESULTS_DIR, 'uacqr', f'uacqr_benchmark_results_{setting}.csv')
    if not os.path.exists(uacqr_csv):
        print(f"  UACQR results CSV not found ({uacqr_csv}), skipping.")
        return

    uacqr_df = pd.read_csv(uacqr_csv)
    uacqr_df = uacqr_df[np.isclose(uacqr_df['Coverage_Target'], 0.95)]

    # Parse seed integer from "run_0" format
    uacqr_df = uacqr_df.copy()
    uacqr_df['seed_int'] = uacqr_df['Seed'].str.replace('run_', '').astype(int)

    baselines = ['UACQR-S', 'UACQR-P']
    metrics = [
        ('NCIW', 'NCIW', 'nciw'),
        ('IntervalScoreLoss', 'AISL', 'intervalscoreloss'),
        ('MPIW', 'Width', 'mpiw'),
        ('PICP', 'Coverage', 'picp'),
    ]

    datasets = sorted(uacqr_df[uacqr_df['Method'] == 'clear']['Dataset'].unique())

    # Load CLEAR data
    if setting == "standard":
        # Standard: use clear from UACQR CSV directly (verified identical to variant c)
        clear_df = uacqr_df[uacqr_df['Method'] == 'clear']
    else:
        # Conformalized: load from conformalized variant (c) CSVs
        conf_dir = os.path.join(RESULTS_DIR, 'conformalized', 'PCS_all_10seeds_qrf')
        clear_rows = []
        for dataset in datasets:
            csv_file = os.path.join(conf_dir, f"benchmark_results_data_{dataset}_95.csv")
            if not os.path.exists(csv_file):
                continue
            df = pd.read_csv(csv_file)
            df = df[df['Method'] == 'clear']
            for seed in range(10):
                seed_data = df[df['Seed'] == seed]
                row = {'Dataset': dataset, 'seed_int': seed}
                for uacqr_col, _, metric_key in metrics:
                    vals = seed_data[seed_data['Metric'] == metric_key]['Value'].values
                    row[uacqr_col] = vals[0] if len(vals) > 0 else np.nan
                clear_rows.append(row)
        clear_df = pd.DataFrame(clear_rows)

    # Compute per-dataset mean percentage improvements
    table_data = {}
    for dataset in datasets:
        table_data[dataset] = {}
        cl = clear_df[clear_df['Dataset'] == dataset]

        for baseline in baselines:
            bl = uacqr_df[(uacqr_df['Method'] == baseline) & (uacqr_df['Dataset'] == dataset)]

            for uacqr_col, display_name, _ in metrics:
                cl_mean = cl[uacqr_col].mean()
                bl_mean = bl[uacqr_col].mean()

                if abs(bl_mean) > 1e-12 and np.isfinite(cl_mean) and np.isfinite(bl_mean):
                    pct = (bl_mean - cl_mean) / abs(bl_mean) * 100
                else:
                    pct = np.nan

                # Track inf seeds for annotation
                bl_vals = bl[uacqr_col].dropna().values
                n_inf = sum(1 for v in bl_vals if np.isinf(v))
                n_total = len(bl_vals)

                table_data[dataset][(baseline, display_name)] = pct
                table_data[dataset][(baseline, display_name, 'n_inf')] = n_inf
                table_data[dataset][(baseline, display_name, 'n_total')] = n_total

    # Build LaTeX table
    setting_label = "standard" if setting == "standard" else "conformalized"
    clear_label = "CLEAR" if setting == "standard" else "CLEAR-c"

    # Check if any entry has inf values
    has_inf = any(
        table_data[ds].get((bl, dn, 'n_inf'), 0) > 0
        for ds in table_data for bl in baselines for _, dn, _ in metrics
    )

    caption = (
        f"Improvement (\\%) of {setting_label} {clear_label} variant (c) over UACQR-S "
        f"and UACQR-P at {COVERAGE}\\% coverage across {len(datasets)} datasets. "
        f"\\textbf{{Bold values with +}} indicate {clear_label} outperforms the baseline."
    )
    if has_inf and setting == "conformalized":
        caption += (
            f" $+\\infty$ indicates diverged baseline predictions; the superscript "
            f"denotes the number of affected seeds out of the total."
    )

    metric_headers = " & ".join(d for _, d, _ in metrics)
    n_metrics = len(metrics)
    col_s_end = 1 + n_metrics
    col_p_start = col_s_end + 1
    col_p_end = col_p_start + n_metrics - 1

    table_lines = [
        "\\begin{table}[!htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{tab:clear_vs_uacqr_detailed_{setting_label}}}",
        "\\resizebox{\\textwidth}{!}{%",
        f"\\begin{{tabular}}{{l{'c' * (n_metrics * len(baselines))}}}",
        "\\toprule",
        f"Dataset & \\multicolumn{{{n_metrics}}}{{c}}{{UACQR-S}} & \\multicolumn{{{n_metrics}}}{{c}}{{UACQR-P}} \\\\",
        f"\\cmidrule(lr){{2-{col_s_end}}} \\cmidrule(lr){{{col_p_start}-{col_p_end}}}",
        f" & {metric_headers} & {metric_headers} \\\\",
        "\\midrule",
    ]

    for dataset in datasets:
        esc_dataset = dataset.replace("_", "\\_")
        cells = [esc_dataset]

        for baseline in baselines:
            for _, display_name, _ in metrics:
                pct = table_data[dataset].get((baseline, display_name), np.nan)
                if not np.isfinite(pct):
                    if setting == "conformalized":
                        n_inf = table_data[dataset].get((baseline, display_name, 'n_inf'), 0)
                        n_total = table_data[dataset].get((baseline, display_name, 'n_total'), 0)
                        if n_inf > 0 and n_total > 0:
                            cells.append(f"$+\\infty$\\textsuperscript{{{n_inf}/{n_total}}}")
                        else:
                            cells.append("$+\\infty$")
                    else:
                        cells.append("$+\\infty$")
                else:
                    if pct > 0:
                        cells.append(f"\\textbf{{+{pct:.1f}\\%}}")
                    else:
                        cells.append(f"{pct:.1f}\\%")

        table_lines.append(" & ".join(cells) + " \\\\")

    table_lines.extend([
        "\\bottomrule",
        "\\end{tabular}%",
        "}",
        "\\end{table}",
    ])

    output_dir = os.path.join(OUTPUT_BASE, 'pcs_cqr', setting_label)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"table-clear-vs-uacqr-improvement-{COVERAGE}_{setting_label}.tex")
    write_latex_table(table_lines, output_file)
    print(f"  Created UACQR improvement table at {output_file}")


# ── plot generation ────────────────────────────────────────────────────────────

FIGURES_OUTPUT = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'figures'))


def _run_script(script_name, extra_args=None):
    """Run a script from SCRIPT_DIR with optional extra arguments."""
    script_path = os.path.join(SCRIPT_DIR, script_name)
    cmd = [sys.executable, script_path] + (extra_args or [])
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=BASE_DIR)
    if result.returncode != 0:
        print(f"  WARNING: {script_name} exited with code {result.returncode}")
    return result.returncode


def generate_plots():
    """
    Generate all plots for the CLEAR paper.

    Orchestrates three plot scripts:
    1. plot_real_benchmark_results.py  - Real benchmark bar plots
    2. plot_simulation_results.py      - Simulation distance metric plots
    3. generate_uacqr_csv_plots.py     - UACQR comparison plots

    All scripts write directly to docs/figures/ (no intermediate plots/ directory).
    """
    print("\n" + "=" * 60)
    print("=== Generating Plots ===")
    print("=" * 60)

    status = 0

    real_output_dir = os.path.join(FIGURES_OUTPUT, 'pcs_cqr', 'standard', 'real')
    sim_output_dir = os.path.join(FIGURES_OUTPUT, 'pcs_cqr', 'standard', 'simulations')

    # Clean up stale individual metric plots (only combined plots are generated)
    if os.path.isdir(real_output_dir):
        for stale in glob.glob(os.path.join(real_output_dir, '*_benchmark_plot_nciw.*')) + \
                      glob.glob(os.path.join(real_output_dir, '*_benchmark_plot_quantile_loss.*')):
            os.remove(stale)
            print(f"  Removed stale individual plot: {os.path.basename(stale)}")

    # 1. Real benchmark plots (standard) — default args write to docs/figures/
    print("\n--- Real benchmark plots (standard) ---")
    rc = _run_script("plot_real_benchmark_results.py")
    status |= rc

    # 1b. Real benchmark plots — CLEAR variant comparison
    print("\n--- Real benchmark plots (CLEAR variants) ---")
    rc = _run_script("plot_real_benchmark_results.py", ["--compare_clear_variants"])
    status |= rc

    # 2. Simulation plots — default args read/write docs/figures/
    print("\n--- Simulation plots ---")
    rc = _run_script("plot_simulation_results.py")
    status |= rc

    # 3. UACQR plots (using standard results)
    print("\n--- UACQR plots ---")
    uacqr_csv = os.path.join(RESULTS_DIR, 'uacqr', "uacqr_benchmark_results_standard.csv")
    uacqr_plot_dir = os.path.join(FIGURES_OUTPUT, 'uacqr')
    uacqr_summary_dir = os.path.join(RESULTS_DIR, 'uacqr')
    rc = _run_script("generate_uacqr_csv_plots.py", [
        "--csv_path", uacqr_csv,
        "--output_dir", uacqr_plot_dir,
        "--summary_output_dir", uacqr_summary_dir,
        "--output_format", "pdf",
    ])
    status |= rc

    if status == 0:
        print("\n" + "=" * 60)
        print("All plots generated successfully!")
        print(f"Output: {FIGURES_OUTPUT}")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("Plot generation completed with warnings (see above).")
        print("=" * 60)

    return status


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate LaTeX tables and (optionally) plots for the CLEAR paper.",
    )
    parser.add_argument(
        '--plots', action='store_true', default=False,
        help='Also generate plots (requires matplotlib). Without this flag, only tables are generated.',
    )
    parser.add_argument(
        '--plots-only', action='store_true', default=False,
        help='Generate only plots, skip table generation.',
    )
    args = parser.parse_args()

    print(f"Base directory: {BASE_DIR}")
    print(f"Output directory: {OUTPUT_BASE}")
    print(f"Results directory: {RESULTS_DIR}")

    if not args.plots_only:
        generate_de_sqr_tables()
        generate_standard_tables()
        generate_conformalized_tables()
        generate_dataset_stats_table()
        generate_uacqr_improvement_table(setting="standard")
        generate_uacqr_improvement_table(setting="conformalized")

        print("\n" + "=" * 60)
        print("All tables generated successfully!")
        print(f"Output: {OUTPUT_BASE}")
        print("=" * 60)

    if args.plots or args.plots_only:
        generate_plots()


if __name__ == '__main__':
    main()
