#!/usr/bin/env python
"""Compute and render CLEAR percentage improvement over DE/SQR methods as LaTeX tables."""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

from utils import format_metric_name, write_latex_table

HIGHER_IS_BETTER: Dict[str, bool] = {
    "PICP": True,
    "NIW": False,
    "MPIW": False,
    "QuantileLoss": False,
    "ExpectileLoss": False,
    "CRPS": False,
    "AUC": False,
    "NCIW": False,
    "IntervalScoreLoss": False,
}

# CSV method name -> display name
METHODS_MAPPING = {
    "DE_calibrated": "DE",
    "SQR_uncalibrated": "SQR",
    "DE_conformal": "DE-conformal",
    "SQR_conformal": "SQR-conformal",
}


# ---------------------------------------------------------------------------
# 1. Load & compute
# ---------------------------------------------------------------------------

def load_csvs(results_dir: Path, coverage: int) -> Iterable[pd.DataFrame]:
    """Yield DataFrames from de_sqr_results_*_{coverage}.csv files."""
    pattern = f"de_sqr_results_*_{coverage}.csv"
    for csv_path in sorted(results_dir.glob(pattern)):
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        df["__dataset"] = csv_path.stem.replace("de_sqr_results_", "").rsplit("_", 1)[0]
        yield df


def collect_improvements(
    dfs: Iterable[pd.DataFrame],
    csv_methods: List[str],
) -> pd.DataFrame:
    """Compute per-dataset, per-run percentage improvement of CLEAR over each method."""
    records: List[Dict[str, object]] = []

    for df in dfs:
        dataset = df["__dataset"].iloc[0]
        run_col = "run" if "run" in df.columns else None
        runs = df[run_col].unique().tolist() if run_col else [None]

        for run_id in runs:
            mask = (df[run_col] == run_id) if run_col else slice(None)
            subset = df[mask]
            clear_rows = subset[subset["method"] == "CLEAR"]
            if clear_rows.empty:
                continue
            clear_row = clear_rows.iloc[0]

            for csv_method in csv_methods:
                comp_rows = subset[subset["method"] == csv_method]
                if comp_rows.empty:
                    continue
                comp_row = comp_rows.iloc[0]
                display_name = METHODS_MAPPING.get(csv_method, csv_method)

                for metric, higher_is_better in HIGHER_IS_BETTER.items():
                    if metric not in subset.columns:
                        continue
                    clear_val = clear_row.get(metric)
                    comp_val = comp_row.get(metric)
                    if pd.isna(clear_val) or pd.isna(comp_val) or np.isclose(comp_val, 0.0):
                        continue

                    if higher_is_better:
                        percent = (clear_val - comp_val) / abs(comp_val) * 100.0
                    else:
                        percent = (comp_val - clear_val) / abs(comp_val) * 100.0

                    records.append({
                        "dataset": dataset,
                        "run": run_id if run_id is not None else "overall",
                        "method": display_name,
                        "metric": metric,
                        "percent_improvement": percent,
                    })

    return pd.DataFrame(records)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate improvements by (method, metric)."""
    if df.empty:
        return pd.DataFrame(
            columns=["method", "metric", "n", "mean_percent", "std_percent", "median_percent"]
        )
    return (
        df.groupby(["method", "metric"])["percent_improvement"]
        .agg([("n", "count"), ("mean_percent", "mean"), ("std_percent", "std"), ("median_percent", "median")])
        .reset_index()
    )


def pivot_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot so rows=metrics, columns=methods, values=mean_percent."""
    if summary_df.empty:
        return pd.DataFrame()
    return summary_df.pivot_table(
        index="metric", columns="method", values="mean_percent", aggfunc="first",
    )


# ---------------------------------------------------------------------------
# 2. Render LaTeX
# ---------------------------------------------------------------------------

def generate_percentage_table(
    pivot_df: pd.DataFrame,
    coverage: int,
    output_dir: str,
    metrics: List[str] | None = None,
    landscape_mode: bool = False,
):
    """Write a LaTeX table of percentage improvements to output_dir."""
    method_order = ["DE", "SQR", "DE-conformal", "SQR-conformal"]
    columns = [m for m in method_order if m in pivot_df.columns]

    if metrics is None:
        metrics = [m for m in HIGHER_IS_BETTER if m in pivot_df.index]

    n_datasets = "all"  # placeholder; updated in main()
    caption = (
        f"Mean (\\%) improvement of CLEAR over DE \\& SQR across {n_datasets} datasets "
        f"at {coverage}\\% coverage (higher is better). "
        "\\textbf{Bold} values indicate CLEAR outperforms the baseline."
    )
    label = "tab:clear_percentage_improvement"

    header = "Metric & " + " & ".join(columns) + " \\\\"

    table_lines = [
        "\\begin{table}[!htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\small",
        f"\\begin{{tabular}}{{l{'c' * len(columns)}}}",
        "\\toprule",
        header,
        "\\midrule",
    ]

    for metric in metrics:
        if metric not in pivot_df.index:
            continue
        row_vals = []
        for col in columns:
            val = pivot_df.loc[metric, col] if col in pivot_df.columns else None
            if val is None or pd.isna(val):
                row_vals.append("-")
            else:
                formatted = f"{val:+.2f}\\%"
                if val > 0:
                    formatted = f"\\textbf{{{formatted}}}"
                row_vals.append(formatted)
        table_lines.append(f"{metric} & " + " & ".join(row_vals) + " \\\\")

    table_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    output_file = os.path.join(output_dir, f"table-clear-percentage-improvement-{coverage}.tex")
    write_latex_table(table_lines, output_file, landscape_mode)
    print(f"  Created percentage improvement table at {output_file}")
    return table_lines


# ---------------------------------------------------------------------------
# 3. Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", type=str, default="../../results/de_sqr",
                        help="Directory containing de_sqr_results_*_{coverage}.csv files")
    parser.add_argument("--output_dir", type=str, default="../tex_tbls/de_sqr",
                        help="Directory to save the LaTeX table")
    parser.add_argument("--coverage", type=int, default=95)
    parser.add_argument("--metrics", type=str, default="PICP,NIW,NCIW,QuantileLoss",
                        help="Comma-separated metrics to include in the table, or 'all'")
    parser.add_argument("--landscape_mode", action="store_true")
    parser.add_argument("--save_csv", action="store_true",
                        help="Also save detailed and summary CSVs")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load
    print(f"\nLoading DE/SQR results from {results_dir}")
    dfs = list(load_csvs(results_dir, args.coverage))
    if not dfs:
        raise SystemExit("No matching CSV files found.")

    n_datasets = len(dfs)
    print(f"  Found {n_datasets} datasets")

    # Compute
    csv_methods = list(METHODS_MAPPING.keys())
    detail_df = collect_improvements(dfs, csv_methods)
    summary_df = summarise(detail_df)
    pivot_df = pivot_summary(summary_df)

    # Print to console
    pd.set_option("display.float_format", lambda x: f"{x:+.2f}" if isinstance(x, float) else f"{x}")
    print("\nMean percentage improvement (positive = CLEAR better):")
    if pivot_df.empty:
        print("  <empty>")
    else:
        print(pivot_df.to_string(float_format=lambda x: f"{x:+.2f}"))

    # Select metrics
    if args.metrics.lower() == "all":
        metrics = None  # use all available
    else:
        metrics = [m.strip() for m in args.metrics.split(",")]

    # Render LaTeX
    print("\nGenerating LaTeX table...")
    table_lines = generate_percentage_table(
        pivot_df, args.coverage, args.output_dir,
        metrics=metrics, landscape_mode=args.landscape_mode,
    )

    # Patch caption with actual dataset count
    for i, line in enumerate(table_lines):
        if "across all datasets" in line:
            table_lines[i] = line.replace("across all datasets", f"across {n_datasets} datasets")

    # Optionally save CSVs
    if args.save_csv:
        detail_path = os.path.join(args.output_dir, f"clear_improvement_detail_{args.coverage}.csv")
        summary_path = os.path.join(args.output_dir, f"clear_improvement_summary_{args.coverage}.csv")
        detail_df.to_csv(detail_path, index=False)
        summary_df.to_csv(summary_path, index=False)
        print(f"  Saved detail CSV to {detail_path}")
        print(f"  Saved summary CSV to {summary_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
