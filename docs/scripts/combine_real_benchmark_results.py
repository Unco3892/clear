#!/usr/bin/env python
import os
import sys
import numpy as np
import pickle
from pathlib import Path
import argparse
import glob
import re
import pandas as pd  # Add pandas import for CSV reading

# Import our scripts
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)
sys.path.append(os.path.join(script_dir, '..','..', 'src'))
from clear.utils import load_ensemble_pickle
from utils import format_metric_name, write_latex_table, extract_data_from_table

def extract_dataset_stats(dataset, results_dir, coverage):
    """
    Extract statistics from a dataset's ensemble pickle file.
    
    Args:
        dataset: Dataset name
        results_dir: Directory containing ensemble results
        coverage: Coverage percentage
        
    Returns:
        dict: Statistics including n, d, and value range
    """
    # Check if dataset already starts with "data_" prefix
    dataset_key = dataset if dataset.startswith("data_") else f"data_{dataset}"
    # Use the pcs_results pickle file instead of ensemble_results
    ensemble_file = Path(results_dir) / f"{dataset_key}_pcs_results_{coverage}.pkl"
    
    if not os.path.exists(ensemble_file):
        print(f"Error: PCS-EPISTEMIC results file {ensemble_file} not found.")
        return None
    
    print(f"Loading PCS-EPISTEMIC results from {ensemble_file}")
    ensemble_dict = load_ensemble_pickle(ensemble_file)
    
    # Get the first run
    run_key = next(iter(ensemble_dict))
    run_data = ensemble_dict[run_key]
    
    # Extract dimensions
    X_train = np.array(run_data.get("x_train"))
    y_train = np.array(run_data.get("y_train"))
    X_val = np.array(run_data.get("x_val"))
    y_val = np.array(run_data.get("y_val"))
    X_test = np.array(run_data.get("x_test"))
    y_test = np.array(run_data.get("y_test"))
    
    # Combine train and test data to get total observations
    if y_train is not None and y_val is not None and y_test is not None:
        all_y = np.concatenate([y_train.flatten(), y_val.flatten(), y_test.flatten()])
        n_samples = len(all_y)
        print(f"Total number of samples: {n_samples}")
    else:
        print("Warning: No train, val, or test data found")
        n_samples = len(y_test) if y_test is not None else 0
    
    # Get number of features
    if X_train is not None:
        n_features = X_train.shape[1]
    elif X_test is not None:
        n_features = X_test.shape[1]
    else:
        n_features = 0
    
    # Get range of target values
    if y_train is not None and y_test is not None:
        y_min = min(np.min(y_train), np.min(y_test))
        y_max = max(np.max(y_train), np.max(y_test))
    elif y_test is not None:
        y_min = np.min(y_test)
        y_max = np.max(y_test)
    else:
        y_min = y_max = 0
    
    # Calculate range
    y_range = y_max - y_min
    
    # Use the original dataset name without the data_ prefix for consistency
    # This ensures the dataset names in stats match the ones in other parts of the code
    return {
        "dataset": dataset,
        "n_samples": n_samples,
        "n_features": n_features,
        "y_min": y_min,
        "y_max": y_max,
        "y_range": y_range
    }

def generate_dataset_stats_table(dataset_stats, output_dir, coverage=90, landscape_mode=False):
    """
    Generate a LaTeX table with the dataset statistics.
    
    Args:
        dataset_stats: List of dictionaries containing dataset statistics
        output_dir: Directory to save the table
        coverage: Coverage percentage for the table caption
        landscape_mode: Whether to generate tables in landscape mode
    """
    table_lines = [
        "\\begin{table}[!htbp]",
        "\\centering",
        r"\caption{Dataset statistics where $d$ represents the number of variables, $n$ represents the number of observations, followed by the minimum, maximum, and range values for $y$.}",
        "\\label{tab:dataset_stats}",
        "\\small",
        # "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Dataset & $n$ & $d$ & $y_{min}$ & $y_{max}$ & $y_{range}$ \\\\",
        "\\midrule"
    ]
    
    # Sort by dataset name
    dataset_stats.sort(key=lambda x: x["dataset"])
    
    for stats in dataset_stats:
        # Replace underscores with \textunderscore to prevent LaTeX errors
        dataset = stats["dataset"].replace("_", "\\textunderscore ")
        n_samples = f"{stats['n_samples']:,}"
        n_features = str(stats["n_features"])
        
        # Format min/max/range according to their scale
        y_min = stats["y_min"]
        y_max = stats["y_max"]
        y_range = stats["y_range"]
        
        # Use scientific notation for very small or large values
        if abs(y_min) < 0.001 or abs(y_min) > 10000:
            y_min_str = f"{y_min:.2e}"
        else:
            y_min_str = f"{y_min:.4f}"
            
        if abs(y_max) < 0.001 or abs(y_max) > 10000:
            y_max_str = f"{y_max:.2e}"
        else:
            y_max_str = f"{y_max:.4f}"
            
        if abs(y_range) < 0.001 or abs(y_range) > 10000:
            y_range_str = f"{y_range:.2e}"
        else:
            y_range_str = f"{y_range:.4f}"
        
        table_lines.append(f"{dataset} & {n_samples} & {n_features} & {y_min_str} & {y_max_str} & {y_range_str} \\\\")
    
    table_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        # "}",
        "\\end{table}"
    ])
    
    # Write the table
    output_file = os.path.join(output_dir, f"table-combined-dataset-stats.tex")
    write_latex_table(table_lines, output_file, landscape_mode)
    
    print(f"Created dataset statistics table with {len(dataset_stats)} datasets at {output_file}")

def combine_metrics_tables(input_dir, output_dir, coverage=90, decimal_places=4, method_set='standard', source_csv_dir='results', landscape_mode=False, current_variant_label=None, uacqr_agg_csv=None):
    """
    Combine individual metric tables into combined tables.
    
    Args:
        input_dir: Directory containing individual tables (can be None when reading directly from CSV)
        output_dir: Directory to save combined tables
        coverage: Coverage percentage for table caption
        decimal_places: Default decimal places for formatting values
        method_set: Predefined set of methods to include in tables
        source_csv_dir: Directory containing the CSV files
        landscape_mode: Whether to generate tables in landscape mode
        current_variant_label: Current variant label for conditional updates (e.g., 'a', 'b', 'c')
        uacqr_agg_csv: Path to the aggregated CSV file containing UACQR-S and UACQR-P results. Used with --process_variants for variant 'c'.
    """
    # Define metrics to process with their specific decimal places (if different from default)
    metrics_config = {
        "picp": {"decimals": 2, "higher_is_better": True},  # PICP: higher is better
        "niw": {"decimals": 3, "higher_is_better": False},
        "mpiw": {"decimals": 3, "higher_is_better": False},
        "quantileloss": {"decimals": 3, "higher_is_better": False},
        "crps": {"decimals": 3, "higher_is_better": False},
        "nciw": {"decimals": 3, "higher_is_better": False},
        "expectileloss": {"decimals": 3, "higher_is_better": False},
        "intervalscoreloss": {"decimals": 3, "higher_is_better": False}
    }
    
    # Define predefined method sets
    method_sets = {
        'standard': [
            ("clear_vanilla", "CLEAR-Vanilla"),
            ("clear_vanilla_c", "CLEAR-c-Vanilla"),
            ("cqr", "CQR"),
            ("pcs", "PCS-EPISTEMIC"),
            ("a_naive", "A-Naive"),
            ("mean_pcs_cqr", "Mean(PCS-EPISTEMIC+CQR)"),
            ("uacqr_s", "UACQR-S"),
            ("uacqr_s_c", "UACQR-S-c"),
            ("lambda_one", "Lambda=1"),
            ("lambda_one_c", "Lambda=1-c")
        ],
        'residual': [
            ("clear", "CLEAR-R"),
            ("clear_c", "CLEAR-R-c"),
            ("cqr_residual", "CQR-R"),
            ("pcs", "PCS-EPISTEMIC"),
            ("a_naive", "A-Naive"),
            ("mean_pcs_cqr_residual", "Mean(PCS-EPISTEMIC+CQR-R)"),
            ("gamma_1_r", "UACQR-S-R"),
            ("gamma_1_c_r", "UACQR-S-c-R"),
            ("lambda_one_r", "Lambda=1-R"),
            ("lambda_one_c_r", "Lambda=1-c-R")
        ],
        'final': [
            ("clear", "CLEAR"),
            ("cqr", "ALEATORIC"),
            ("cqr_residual", "ALEATORIC-R"),
            ("pcs", "PCS-EPISTEMIC"),
            # ("a_naive", "A-Naive"),
            ("s_naive", "Naive"),
            ("gamma_1_r", "$\\gamma_1=1$"),  # Using LaTeX for gamma_1=1
            ("lambda_one_r", "$\\lambda=1$")  # Using LaTeX for lambda=1
        ],
        'final_c_uacqr': [
            ("clear", "CLEAR"),
            ("cqr", "ALEATORIC"),
            ("cqr_residual", "ALEATORIC-R"),
            ("pcs", "PCS-EPISTEMIC"),
            ("s_naive", "Naive"),
            # ("gamma_1_r", "$\\gamma_1=1$"),
            # ("lambda_one_r", "$\\lambda=1$"),
            ("UACQR-P", "UACQR-P"),
            ("UACQR-S", "UACQR-S")
        ]
    }
    
    # Get the specified method set
    method_set_to_use = method_set
    if method_set not in method_sets and not (method_set == 'final' and current_variant_label == 'c'):
        print(f"Warning: Method set '{method_set}' not defined. Using standard.")
        method_set_to_use = 'standard'
    
    # Default method_info from the chosen set
    if method_set_to_use == 'final' and current_variant_label == 'c' and uacqr_agg_csv is not None:
        print(f"INFO: For variant 'c' (method_set 'final'), using custom method list including UACQR-P and UACQR-S.")
        method_info_list = method_sets['final_c_uacqr']
    else:
        method_info_list = method_sets[method_set_to_use]
        
    method_info = method_info_list
    methods = [m[0] for m in method_info]
    
    # Get base dir for CSV files
    # Handle source_csv_dir as the exact path to look for CSV files
    results_dir = source_csv_dir
    
    # Debug output
    print(f"  Looking for CSV files in: {results_dir}")
    
    # Load aggregated UACQR CSV if applicable
    df_uacqr_aggregated = None
    if uacqr_agg_csv and current_variant_label == 'c' and method_set == 'final':
        if os.path.exists(uacqr_agg_csv):
            try:
                df_uacqr_aggregated = pd.read_csv(uacqr_agg_csv)
                print(f"  Successfully loaded aggregated UACQR CSV: {uacqr_agg_csv}")
            except Exception as e:
                print(f"  Warning: Failed to load aggregated UACQR CSV '{uacqr_agg_csv}': {e}")
        else:
            print(f"  Warning: Aggregated UACQR CSV '{uacqr_agg_csv}' not found.")

    # Process each metric
    metrics = list(metrics_config.keys())
    # To report format errors only once per file
    reported_format_errors_for_files = set()

    for metric in metrics:
        print(f"Processing metric: {metric}")
        
        # Get decimal places for this metric (use default if not specified)
        decimal_places_for_metric = metrics_config.get(metric, {}).get("decimals", decimal_places)
        higher_is_better = metrics_config.get(metric, {}).get("higher_is_better", False)
        
        # Dictionary to store processed data
        data = {}  # Format: {dataset: {method: {'mean': val, 'std': val}}}
        max_runs = 0  # Track maximum number of runs for any dataset
        
        # Find all CSV files matching the pattern
        csv_pattern = os.path.join(results_dir, f"benchmark_results_*_{coverage}.csv")
        csv_files = glob.glob(csv_pattern)
        
        if not csv_files:
            print(f"  No CSV files found for coverage {coverage}% in {results_dir}")
            # Debug info
            print(f"  CSV pattern used: {csv_pattern}")
            print(f"  Try listing all files in directory:")
            if os.path.exists(results_dir):
                files = os.listdir(results_dir)
                for f in files[:10]:  # Show first 10 files
                    print(f"    - {f}")
                if len(files) > 10:
                    print(f"    ... and {len(files)-10} more files")
            else:
                print(f"  Directory {results_dir} does not exist!")
            continue
            
        print(f"  Found {len(csv_files)} CSV files")
        
        # Process each CSV file
        for csv_file in csv_files:
            try:
                # Extract dataset name from the filename
                # This pattern gets the part between "benchmark_results_" and the last underscore followed by coverage
                filename = os.path.basename(csv_file)
                dataset_match = re.search(r'benchmark_results_(.+)_' + str(coverage), filename)
                
                if not dataset_match:
                    print(f"  Could not extract dataset name from {filename}")
                    continue
                
                dataset = dataset_match.group(1)
                print(f"  Processing dataset: {dataset}")
                
                # Read the CSV file
                df = pd.read_csv(csv_file)
                
                # Verify dataset name from CSV content
                if 'Dataset' in df.columns and not df.empty:
                    # Get the actual dataset name from the CSV content
                    csv_dataset = df['Dataset'].iloc[0]
                    if csv_dataset != dataset:
                        print(f"  Dataset name from CSV ({csv_dataset}) doesn't match filename ({dataset})")
                        dataset = csv_dataset  # Use the dataset name from CSV content
                
                # Initialize dataset in the data dictionary
                if dataset not in data:
                    data[dataset] = {}
                
                # Process each method
                for method in methods:
                    values_for_metric = []
                    source_description = "unknown"

                    is_uacqr_target_case = (method in ['UACQR-P', 'UACQR-S'] and \
                                           df_uacqr_aggregated is not None and \
                                           current_variant_label == 'c' and \
                                           method_set == 'final')

                    if is_uacqr_target_case:
                        source_description = f"aggregated UACQR CSV ({uacqr_agg_csv})"
                        # Dataset name in aggregated CSV is without 'data_' prefix (e.g., 'ailerons')
                        dataset_for_uacqr_query = dataset if not dataset.startswith("data_") else dataset[5:]
                        
                        # Map lowercase metric key (from metrics_config) to potential column name in uacqr_benchmark_results.csv
                        # Common column names: PICP, NIW, MPIW, QuantileLoss, ExpectileLoss, IntervalScoreLoss, CRPS, AUC, NCIW
                        uacqr_col_map = {
                            "picp": "PICP", "niw": "NIW", "mpiw": "MPIW",
                            "quantileloss": "QuantileLoss", "crps": "CRPS",
                            "nciw": "NCIW", "expectileloss": "ExpectileLoss",
                            "intervalscoreloss": "IntervalScoreLoss"
                        }
                        uacqr_metric_col_name = uacqr_col_map.get(metric)

                        if uacqr_metric_col_name and uacqr_metric_col_name in df_uacqr_aggregated.columns:
                            method_subset_agg = df_uacqr_aggregated[
                                (df_uacqr_aggregated['Dataset'] == dataset_for_uacqr_query) &
                                (df_uacqr_aggregated['Method'] == method)
                            ]
                            if not method_subset_agg.empty and uacqr_metric_col_name in method_subset_agg:
                                values_for_metric = method_subset_agg[uacqr_metric_col_name].dropna().values.tolist()
                        # else: print(f"    Metric {uacqr_metric_col_name} (from {metric}) not in UACQR CSV or data missing for {dataset_for_uacqr_query}/{method}")
                    
                    else:
                        # Source from current per-dataset CSV `df` (expected long format)
                        source_description = f"per-dataset CSV {filename}"
                        # `dataset` here is the canonical name (either from filename or CSV content, e.g. data_ailerons)
                        # `df` is from pd.read_csv(csv_file)
                        required_cols = ['Dataset', 'Method', 'Metric', 'Value']
                        if all(col in df.columns for col in required_cols):
                            method_data_long = df[
                                (df['Dataset'] == dataset) & 
                                (df['Method'] == method) & 
                                (df['Metric'] == metric) # `metric` is lowercase key
                            ]
                            if not method_data_long.empty:
                                values_for_metric = method_data_long['Value'].dropna().values.tolist()
                        else:
                            if csv_file not in reported_format_errors_for_files:
                                print(f"    Warning: Per-dataset CSV {filename} for dataset {dataset} "
                                      f"is missing required columns ({required_cols}) or is not in expected long format. "
                                      f"Cannot process method '{method}', metric '{metric}' from this file.")
                                reported_format_errors_for_files.add(csv_file)
                    
                    # Calculate statistics if values were found
                    if values_for_metric:
                        max_runs = max(max_runs, len(values_for_metric))
                        # Filter out potential non-numeric string placeholders like '-' before np.mean/std
                        numeric_values = [v for v in values_for_metric if isinstance(v, (int, float)) and not np.isnan(v)]
                        if numeric_values:
                            mean_val = float(np.mean(numeric_values))
                            std_val = float(np.std(numeric_values))
                            data[dataset][method] = {
                                'mean': mean_val,
                                'std': std_val,
                                'values': numeric_values # Store cleaned numeric values
                            }
                        else:
                            # print(f"    No numeric values found for method {method}, metric {metric}, dataset {dataset} from {source_description}")
                            data[dataset][method] = {'mean': np.nan, 'std': np.nan, 'values': []}
                    else:
                        # This case means values_for_metric remained empty
                        # print(f"    No data rows found for method {method}, metric {metric}, dataset {dataset} from {source_description}")
                        if dataset not in data: data[dataset] = {}
                        data[dataset][method] = {'mean': np.nan, 'std': np.nan, 'values': []}

            except Exception as e:
                print(f"  Error processing file {csv_file}: {str(e)}")
        
        # If no data was found, skip this metric
        if not data:
            print(f"  No data found for metric {metric}")
            continue
        
        # Generate the LaTeX table
        formatted_metric = format_metric_name(metric)
        
        # Build caption with explanation of bold and underline
        # Add variant information to caption if applicable
        if current_variant_label:
            caption = f"Variant ({current_variant_label}) {formatted_metric} at {coverage}\\% prediction intervals, aggregated across {max_runs} seeds."
        else:
            caption = f"{formatted_metric} at {coverage}\\% prediction intervals, aggregated across {max_runs} seeds."

        # caption += f"Using all available variables. "
        
        # Add method set information to the caption
        # if method_set == 'standard':
        #     caption += "Using standard (non-residual) methods. "
        # elif method_set == 'residual':
        #     caption += "Using residual-based methods. "
        # elif method_set == 'final':
        #     caption += "Using selected methods. "
        
        # Only include scientific notation message if not PICP
        if metric != "picp":
            caption += r"Values $\geq 100$ or $< 0.01$ are presented in scientific notation with 1 decimal place. "
    
        # Add bold/underline explanation only if the metric is not PICP
        if metric != "picp":
            if higher_is_better:
                caption += "\\textbf{Bold} values (desirable) are the maximum for that dataset and metric"
                caption += ", while the \\underline{underlined} values indicate the second-best result."
                caption += " \\textcolor{red}{Red} values are more than 33\\% worse than the best result."
            else:
                caption += "\\textbf{Bold} values (desirable) are the minimum for that dataset and metric"
                caption += ", while the \\underline{underlined} values indicate the second-best result."
                caption += " \\textcolor{red}{Red} values are more than 33\\% worse than the best result."
                
        table_lines = [
            "\\begin{table}[!htbp]",
            "\\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{tab:combined_{metric}_{coverage}_{method_set}" + (f"_variant_{current_variant_label}" if current_variant_label else "") + "}",
            "\\small",
            "\\resizebox{\\columnwidth}{!}{%",
            "\\begin{tabular}{l" + "c" * len(methods) + "}",
            "\\toprule",
            "Dataset & " + " & ".join([m[1] for m in method_info]) + " \\\\",
            "\\midrule"
        ]
        
        # Add data rows
        for dataset in sorted(data.keys()):
            # Format the dataset name for LaTeX display - escape underscores properly
            # Remove "data_" prefix if it exists
            display_dataset = dataset
            if display_dataset.startswith("data_"):
                display_dataset = display_dataset[5:]
            formatted_dataset = display_dataset.replace("_", "\\textunderscore ")
            row = [formatted_dataset]
            
            # Find the best and second-best values for this dataset based on the metric type
            best_val = np.nan # Initialize best_val
            if higher_is_better:
                # For PICP, the higher the better
                values_methods = []
                for method_key in methods: # Use method_key consistently
                    if method_key in data[dataset] and 'mean' in data[dataset][method_key]: # Check if 'mean' exists
                        mean_val_current = data[dataset][method_key]['mean'] # Use a different var name
                        if not np.isnan(mean_val_current): # Ensure value is not NaN
                           values_methods.append((mean_val_current, method_key))
                
                # Sort by value (descending for higher_is_better)
                values_methods.sort(key=lambda x: x[0], reverse=True) # Sort by value
                
                # Get best and second best if available
                best_method = values_methods[0][1] if values_methods else None
                best_val = values_methods[0][0] if values_methods else np.nan
                second_best_method = values_methods[1][1] if len(values_methods) > 1 else None
                
            else:
                # For all other metrics, the lower the better
                values_methods = []
                for method_key in methods: # Use method_key consistently
                    if method_key in data[dataset] and 'mean' in data[dataset][method_key]: # Check if 'mean' exists
                        mean_val_current = data[dataset][method_key]['mean'] # Use a different var name
                        # Ensure value is not NaN. For "lower is better", also consider positive values as per original logic,
                        # but allow non-positive for finding the best if that's what data contains.
                        # The original `if mean_val > 0:` might be too restrictive if all values are negative or zero.
                        # Let's keep it for now, but note if it causes issues. Original: if mean_val_current > 0 and not np.isnan(mean_val_current):
                        if not np.isnan(mean_val_current): 
                            values_methods.append((mean_val_current, method_key))
                
                # Sort by value (ascending for lower_is_better)
                values_methods.sort(key=lambda x: x[0]) # Sort by value
                
                # Get best and second best if available
                best_method = values_methods[0][1] if values_methods else None
                best_val = values_methods[0][0] if values_methods else np.nan
                second_best_method = values_methods[1][1] if len(values_methods) > 1 else None
            
            # Process each method
            for method in methods:
                if method in data[dataset]:
                    mean_val = data[dataset][method]['mean']
                    std_val = data[dataset][method]['std']
                    
                    # For PICP, never use scientific notation
                    if metric == "picp":
                        use_scientific = False
                    else:
                        # Determine if scientific notation should be used for other metrics
                        use_scientific = abs(mean_val) >= 100 or abs(mean_val) < 0.01 # User specified condition
                    
                    # Format mean and std
                    if use_scientific:
                        # Use 1 decimal place for scientific notation
                        formatted_mean = f"{mean_val:.1e}"
                        # Apply the same rule for std for consistency, though not explicitly requested for std
                        if abs(std_val) >= 100 or abs(std_val) < 0.01:
                             formatted_std = f"{std_val:.1e}"
                        else:
                             formatted_std = f"{std_val:.{decimal_places_for_metric}f}"
                    else:
                        formatted_mean = f"{mean_val:.{decimal_places_for_metric}f}"
                        formatted_std = f"{std_val:.{decimal_places_for_metric}f}"
                    
                    # Create cell content with mean ± std
                    cell_content = f"{formatted_mean} $\\pm$ {formatted_std}"
                    
                    # Apply formatting based on ranking, but not for PICP
                    if metric != "picp":
                        is_best = (method == best_method)
                        is_second_best = (method == second_best_method)

                        if is_best:
                            cell_content = f"\\textbf{{{cell_content}}}"
                        elif is_second_best:
                            cell_content = f"\\underline{{{cell_content}}}"
                        # Add red color for values > 33% worse than best, if not best or second best
                        # and ensure best_val and mean_val are not NaN
                        elif not np.isnan(best_val) and not np.isnan(mean_val):
                            if higher_is_better: # Higher is better, so "worse" means significantly lower
                                # Avoid division by zero if best_val is 0
                                if best_val != 0 and mean_val < best_val * 0.77:
                                    cell_content = f"\\textcolor{{red}}{{{cell_content}}}"
                                elif best_val == 0 and mean_val < 0: # If best is 0, any negative is worse
                                     cell_content = f"\\textcolor{{red}}{{{cell_content}}}"
                            else: # Lower is better, so "worse" means significantly higher
                                # Avoid issues if best_val is 0 or negative
                                if best_val > 0 and mean_val > best_val * 1.33: # Standard case
                                    cell_content = f"\\textcolor{{red}}{{{cell_content}}}"
                                elif best_val == 0 and mean_val > 0: # If best is 0, any positive is worse
                                    cell_content = f"\\textcolor{{red}}{{{cell_content}}}"
                                elif best_val < 0: # If best_val is negative, 'more positive' or 'less negative' by 33% of its magnitude
                                    # Example: best_val = -100. Threshold = -100 * 0.77 = -77. mean_val > -77 is worse.
                                    # Example: best_val = -100. Threshold = -100 + abs(-100)*0.33 = -80.
                                    if mean_val > best_val + abs(best_val) * 0.33:
                                         cell_content = f"\\textcolor{{red}}{{{cell_content}}}"

                    row.append(cell_content)
                else:
                    row.append("-")
            
            table_lines.append(" & ".join(row) + " \\\\")
        
        # Finish the table
        table_lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "}",
            "\\end{table}"
        ])
        
        # Write the table to a file - include method_set in filename to avoid overwriting
        output_file = os.path.join(output_dir, f"table-combined-{coverage}-{metric}-{method_set}.tex")
        write_latex_table(table_lines, output_file, landscape_mode)
        
        print(f"  Created table for metric {metric} with {len(data)} datasets using '{method_set}' method set")

def combine_gamma_lambda_tables(input_dir, output_dir, coverage=90, source_csv_dir='results', method_set='standard', landscape_mode=False, current_variant_label=None):
    """
    Combine gamma and lambda tables into a single table.
    
    Args:
        input_dir: Directory containing gamma and lambda tables
        output_dir: Directory to save combined table
        coverage: Coverage percentage for table caption
        source_csv_dir: Directory containing the CSV files
        method_set: Method set to use ('standard' or 'residual')
        landscape_mode: Whether to generate tables in landscape mode
        current_variant_label: Current variant label for conditional updates (e.g., 'a', 'b', 'c')
        
    Returns:
        List of datasets with valid gamma and lambda values
    """
    # Find all gamma and lambda tables
    gamma_pattern = f"table-*-all-variables-{coverage}-gamma.tex"
    lambda_pattern = f"table-*-all-variables-{coverage}-lambda.tex"
    
    gamma_files = glob.glob(os.path.join(input_dir, gamma_pattern))
    lambda_files = glob.glob(os.path.join(input_dir, lambda_pattern))
    
    # Get unique datasets from both gamma and lambda tables
    all_datasets = set()
    dataset_data = {}
    run_counts = []
    var_info = []  # Track variable information
    
    results_dir = source_csv_dir # Directly use the provided source_csv_dir
    
    # Determine which method to use for lambda and gamma values
    # When method_set is 'final', we are interested in the parameters of what's displayed as 'CLEAR'
    # which is internally 'clear_residual'.
    if method_set == 'final':
        method_for_params = 'clear'
    elif method_set == 'residual':
        method_for_params = 'clear'
    else: # 'standard' or other
        method_for_params = 'clear_vanilla'

    print(f"DEBUG: In combine_gamma_lambda_tables, method_for_params = {method_for_params} (method_set='{method_set}')") # Debug print
    
    # Keep track of datasets with valid CSV data
    datasets_with_csv_data = set()
    dataset_name_mapping = {} # Maps dataset names from .tex files (if any) to CSV dataset names

    # Attempt to get dataset names from .tex files first
    # This part is more relevant when not in --process_variants mode
    datasets_from_tex = set()
    if gamma_files or lambda_files:
        for f in gamma_files + lambda_files:
            match = re.search(r'table-([^-]+)-all', os.path.basename(f))
            if match:
                datasets_from_tex.add(match.group(1))

    # If no datasets from .tex files (e.g., during variant processing where input_dir is empty of these)
    # or if we always want to prioritize CSVs for dataset listing:
    # Get dataset names by scanning CSV files in results_dir
    
    # Scan CSVs to find all available datasets in the current source_csv_dir
    available_datasets_in_csvs = set()
    csv_pattern_for_scan = os.path.join(results_dir, f"benchmark_results_*_{coverage}.csv")
    all_csv_files_in_source = glob.glob(csv_pattern_for_scan)

    print(f"DEBUG: Found {len(all_csv_files_in_source)} CSV files in {results_dir} for dataset scanning.") # Debug print

    for csv_f in all_csv_files_in_source:
        filename = os.path.basename(csv_f)
        # Updated regex to be more robust for dataset name extraction from CSV filenames
        dataset_match_csv = re.search(r'benchmark_results_(data_[a-zA-Z0-9_]+(?:_small|_medium)?|[^_]+(?:_small|_medium)?)_' + str(coverage), filename)
        if dataset_match_csv:
            raw_dataset_name = dataset_match_csv.group(1)
            # Store the raw name (e.g., data_ailerons or ailerons)
            # The code below will handle adding/removing "data_" prefix as needed.
            available_datasets_in_csvs.add(raw_dataset_name)
        else:
            print(f"DEBUG: Could not extract dataset name from CSV filename: {filename}")


    # Determine the primary list of datasets to iterate over
    # If datasets_from_tex is populated, use that, otherwise use available_datasets_in_csvs
    # For variant processing, datasets_from_tex will be empty.
    iteration_dataset_list = datasets_from_tex if datasets_from_tex else available_datasets_in_csvs
    if not iteration_dataset_list and available_datasets_in_csvs: # Fallback if tex files were expected but not found
        iteration_dataset_list = available_datasets_in_csvs

    print(f"DEBUG: Datasets to iterate for gamma/lambda check: {iteration_dataset_list}") # Debug print

    # First, find all datasets that have valid CSV files
    # for dataset in set(re.search(r'table-([^-]+)-all', os.path.basename(f)).group(1) 
    #                 for f in gamma_files + lambda_files if re.search(r'table-([^-]+)-all', os.path.basename(f))):
    for dataset_from_source in iteration_dataset_list: # Iterate over datasets found from .tex or scanned CSVs
        
        # dataset_from_source could be 'ailerons' or 'data_ailerons'
        # We need to ensure csv_dataset_name is what's in the CSV filenames (usually 'data_...')
        # and dataset_key_for_df is what's inside the 'Dataset' column of the CSV (also usually 'data_...')
        
        if dataset_from_source.startswith("data_"):
            csv_dataset_name = dataset_from_source # e.g. data_ailerons
            dataset_key_for_df = dataset_from_source
            original_dataset_for_output = dataset_from_source[5:] # e.g. ailerons (for output table if needed)
        else:
            csv_dataset_name = f"data_{dataset_from_source}" # e.g. data_ailerons
            dataset_key_for_df = f"data_{dataset_from_source}" 
            original_dataset_for_output = dataset_from_source # e.g. ailerons
            
        dataset_name_mapping[original_dataset_for_output] = dataset_key_for_df
            
        # Use exact match for CSV filename for this specific dataset
        # Ensure results_dir is the direct path to the folder containing these CSVs
        specific_csv_file_pattern = os.path.join(results_dir, f"benchmark_results_{csv_dataset_name}_{coverage}.csv")
        csv_files_for_dataset = glob.glob(specific_csv_file_pattern)

        if csv_files_for_dataset:
            try:
                df = pd.read_csv(csv_files_for_dataset[0])
                # Check if the required method and metrics exist in the CSV
                # dataset_key_for_df should be used for filtering df['Dataset']
                has_lambda = not df[(df['Dataset'] == dataset_key_for_df) & 
                                   (df['Method'] == method_for_params) & 
                                   (df['Metric'] == 'lambda')].empty
                has_gamma = not df[(df['Dataset'] == dataset_key_for_df) & 
                                  (df['Method'] == method_for_params) & 
                                  (df['Metric'] == 'gamma')].empty
                
                if has_lambda and has_gamma:
                    datasets_with_csv_data.add(original_dataset_for_output) # Add the non-'data_' prefixed version
                    print(f"Dataset {original_dataset_for_output}: Found valid lambda and gamma data for method {method_for_params}")
                else:
                    print(f"Dataset {original_dataset_for_output}: Missing lambda or gamma data for method {method_for_params} in {csv_files_for_dataset[0]}")
            except Exception as e:
                print(f"Error processing CSV for {original_dataset_for_output} ({csv_files_for_dataset[0]}): {e}")
        else:
            print(f"DEBUG: No specific CSV file found for {csv_dataset_name} with pattern {specific_csv_file_pattern}")

    
    # Process gamma files, but only for datasets with valid CSV data
    # This part might be largely skipped if datasets_from_tex was empty
    for gamma_file in gamma_files: # This loop will not run if gamma_files is empty
        dataset_match = re.search(r'table-([^-]+)-all', os.path.basename(gamma_file))
        if dataset_match:
            dataset = dataset_match.group(1)
            
            # Skip datasets without valid CSV data
            if dataset not in datasets_with_csv_data:
                print(f"Skipping dataset {dataset} - no valid CSV data found")
                continue
                
            all_datasets.add(dataset)
            
            with open(gamma_file, 'r', encoding="utf-8") as f:
                content = f.read()
                
                # Extract gamma value
                data_match = re.search(r'\\midrule\n.*?& (.*?) \\\\', content, re.DOTALL)
                if data_match:
                    gamma_value = data_match.group(1)
                    
                    # Try to find the CSV file with actual data
                    # Use the mapped CSV dataset name
                    csv_dataset = dataset_name_mapping.get(dataset, dataset)
                    csv_files = glob.glob(os.path.join(results_dir, f"benchmark_results_{csv_dataset}_{coverage}.csv"))
                    
                    if csv_files:
                        # Try to extract raw gamma values from CSV
                        try:
                            df = pd.read_csv(csv_files[0])
                            gamma_values = df[(df['Dataset'] == csv_dataset) & 
                                              (df['Method'] == method_for_params) & 
                                              (df['Metric'] == 'gamma')]['Value'].tolist()
                            
                            if gamma_values:
                                gamma_median = np.median(gamma_values)
                                gamma_min = min(gamma_values)
                                gamma_max = max(gamma_values)
                                gamma_value = f"{gamma_median:.2f} [{gamma_min:.2f}:{gamma_max:.2f}]"
                            else:
                                print(f"  No gamma values found for {dataset}")
                                continue
                        except Exception as e:
                            print(f"  Error processing CSV for gamma values of {dataset}: {e}")
                            continue
                    else:
                        print(f"  No CSV file found for {dataset}")
                        continue
                    
                    if dataset not in dataset_data:
                        dataset_data[dataset] = {'gamma': gamma_value, 'lambda': '-'}
                    else:
                        dataset_data[dataset]['gamma'] = gamma_value
                
                # Extract run count
                run_match = re.search(r'across (\d+) seeds', content)
                if run_match:
                    run_counts.append(int(run_match.group(1)))
                
                # Extract variable information from caption
                var_match = re.search(r'Using (.*?)\.', content)
                if var_match:
                    var_info.append(var_match.group(1))
    
    # Process lambda files, but only for datasets with valid CSV data
    # This part might be largely skipped if datasets_from_tex was empty
    for lambda_file in lambda_files: # This loop will not run if lambda_files is empty
        dataset_match = re.search(r'table-([^-]+)-all', os.path.basename(lambda_file))
        if dataset_match:
            dataset = dataset_match.group(1)
            
            # Skip datasets without valid CSV data
            if dataset not in datasets_with_csv_data:
                continue
                
            all_datasets.add(dataset)
            
            with open(lambda_file, 'r', encoding="utf-8") as f:
                content = f.read()
                
                # Extract lambda value
                data_match = re.search(r'\\midrule\n.*?& (.*?) \\\\', content, re.DOTALL)
                if data_match:
                    lambda_value = data_match.group(1)
                    # Clean up the lambda value - remove trailing $ if it exists without a matching opening $
                    lambda_value = lambda_value.strip()
                    if lambda_value.endswith('$') and not lambda_value.startswith('$') and lambda_value.count('$') == 1:
                        lambda_value = lambda_value[:-1].strip()
                    
                    # Try to find the CSV file with actual data
                    # Use the mapped CSV dataset name
                    csv_dataset = dataset_name_mapping.get(dataset, dataset)
                    csv_files = glob.glob(os.path.join(results_dir, f"benchmark_results_{csv_dataset}_{coverage}.csv"))
                    
                    if csv_files:
                        # Try to extract raw lambda values from CSV
                        try:
                            df = pd.read_csv(csv_files[0])
                            lambda_values = df[(df['Dataset'] == csv_dataset) & 
                                               (df['Method'] == method_for_params) & 
                                               (df['Metric'] == 'lambda')]['Value'].tolist()
                            
                            if lambda_values:
                                lambda_median = np.median(lambda_values)
                                lambda_min = min(lambda_values)
                                lambda_max = max(lambda_values)
                                lambda_value = f"{lambda_median:.2f} [{lambda_min:.2f}:{lambda_max:.2f}]"
                            else:
                                print(f"  No lambda values found for {dataset}")
                                continue
                        except Exception as e:
                            print(f"  Error processing CSV for lambda values of {dataset}: {e}")
                            continue
                    else:
                        print(f"  No CSV file found for {dataset}")
                        continue
                    
                    if dataset not in dataset_data:
                        dataset_data[dataset] = {'gamma': '-', 'lambda': lambda_value}
                    else:
                        dataset_data[dataset]['lambda'] = lambda_value
                
                # Extract run count
                run_match = re.search(r'across (\d+) seeds', content)
                if run_match:
                    run_counts.append(int(run_match.group(1)))
                
                # Extract variable information from caption
                var_match = re.search(r'Using (.*?)\.', content)
                if var_match:
                    var_info.append(var_match.group(1))
    
    # Get the actual number of runs (use maximum, not sum)
    num_runs = max(run_counts) if run_counts else 0 # If run_counts is empty (e.g. no tex files), num_runs will be 0.
                                                  # We should get num_runs from CSV processing later.
    
    # If dataset_data is empty (because no .tex files were processed)
    # but we have datasets_with_csv_data, populate dataset_data directly from CSVs
    if not dataset_data and datasets_with_csv_data:
        print("DEBUG: No .tex files processed for gamma/lambda, populating directly from CSVs for datasets:", datasets_with_csv_data)
        max_runs_from_csv = 0
        for dataset_name_orig in sorted(list(datasets_with_csv_data)): # e.g. ailerons
            # dataset_name_csv = f"data_{dataset_name_orig}" if not dataset_name_orig.startswith("data_") else dataset_name_orig
            dataset_name_csv = dataset_name_mapping.get(dataset_name_orig, dataset_name_orig)


            specific_csv_file_pattern = os.path.join(results_dir, f"benchmark_results_{dataset_name_csv}_{coverage}.csv")
            csv_files_for_ds = glob.glob(specific_csv_file_pattern)

            if csv_files_for_ds:
                df_ds = pd.read_csv(csv_files_for_ds[0])
                lambda_values = df_ds[(df_ds['Dataset'] == dataset_name_csv) & 
                                   (df_ds['Method'] == method_for_params) & 
                                   (df_ds['Metric'] == 'lambda')]['Value'].tolist()
                gamma_values = df_ds[(df_ds['Dataset'] == dataset_name_csv) & 
                                  (df_ds['Method'] == method_for_params) & 
                                  (df_ds['Metric'] == 'gamma')]['Value'].tolist()

                if lambda_values:
                    max_runs_from_csv = max(max_runs_from_csv, len(lambda_values))
                    lambda_median = np.median(lambda_values)
                    lambda_min = min(lambda_values)
                    lambda_max = max(lambda_values)
                    lambda_str = f"{lambda_median:.2f} [{lambda_min:.2f}:{lambda_max:.2f}]"
                else:
                    lambda_str = "-"
                
                if gamma_values:
                    max_runs_from_csv = max(max_runs_from_csv, len(gamma_values))
                    gamma_median = np.median(gamma_values)
                    gamma_min = min(gamma_values)
                    gamma_max = max(gamma_values)
                    gamma_str = f"{gamma_median:.2f} [{gamma_min:.2f}:{gamma_max:.2f}]"
                else:
                    gamma_str = "-"

                if dataset_name_orig not in dataset_data:
                    dataset_data[dataset_name_orig] = {}
                dataset_data[dataset_name_orig]['lambda'] = lambda_str
                dataset_data[dataset_name_orig]['gamma'] = gamma_str
            else:
                print(f"DEBUG: CSV file for {dataset_name_csv} not found when populating dataset_data.")
        
        if num_runs == 0: # Update num_runs if it wasn't set from .tex file processing
             num_runs = max_runs_from_csv
        
        # Ensure all_datasets also reflects these datasets for table generation
        all_datasets.update(datasets_with_csv_data)


    # Determine variable information for the combined caption
    combined_var_info = "all available variables"
    if var_info:
        # Check if all entries are the same
        if all(v == var_info[0] for v in var_info):
            combined_var_info = var_info[0]
        else:
            # If different, default to "all variables"
            combined_var_info = "all variables"
        
        # Extract variable count if present
        var_count_match = re.search(r'(\d+) variables', combined_var_info)
        if var_count_match:
            var_count = var_count_match.group(1)
            combined_var_info = f"{var_count} variables"
    
    # Create combined gamma-lambda table
    if dataset_data:
        # method_name = "CLEAR" if method_set == 'residual' else "CLEAR-Vanilla"
        method_name = "CLEAR"
        if current_variant_label is not None:
            caption = f"Variant ({current_variant_label}) {method_name} calibration parameters $\\lambda$ and $\\gamma_1$ for {coverage}\\% prediction intervals across {num_runs} seeds. Using {combined_var_info}. Showing median [min:max] values."
        else:
            caption = f"{method_name} calibration parameters $\\lambda$ and $\\gamma_1$ for {coverage}\\% prediction intervals across {num_runs} seeds. Using {combined_var_info}. Showing median [min:max] values."
        
        table_lines = [
            "\\begin{table}[!htbp]",
            "\\centering",
            r"\caption{" + caption + r"}",
            f"\\label{{tab:combined_gamma_lambda_{coverage}_{method_set}" + (f"_variant_{current_variant_label}" if current_variant_label else "") + "}",
            "\\small",
            "\\begin{tabular}{lccc}",
            "\\toprule",
            "Dataset & $\\lambda$ & $\\gamma_1$ \\\\",
            "\\midrule"
        ]
        
        # Sort datasets for consistent ordering
        for dataset in sorted(all_datasets):
            # Skip datasets that don't have both lambda and gamma
            if 'lambda' not in dataset_data[dataset] or 'gamma' not in dataset_data[dataset]:
                continue
                
            # Replace underscores with \textunderscore to prevent LaTeX errors
            dataset_escaped = dataset.replace("_", "\\textunderscore ")
            lambda_value = dataset_data[dataset]['lambda']
            gamma_value = dataset_data[dataset]['gamma']
            table_lines.append(f"{dataset_escaped} & {lambda_value} & {gamma_value} \\\\")
        
        table_lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}"
        ])
        
        # Write combined gamma-lambda table
        output_file = os.path.join(output_dir, f"table-combined-{coverage}-gamma-lambda-{method_set}.tex")
        write_latex_table(table_lines, output_file, landscape_mode)
        
        print(f"Created combined gamma-lambda table with {len(all_datasets)} datasets and {num_runs} seeds.")
        return list(datasets_with_csv_data)
    else:
        print("No gamma or lambda data found to create combined table.")
        return []

def generate_uncertainty_metrics_table(datasets, output_dir, coverage=90, source_csv_dir='results', landscape_mode=False, method_set=None, current_variant_label=None):
    """
    Generate a table with uncertainty metrics for all model variants.
    
    Args:
        datasets: List of datasets to include in the table
        output_dir: Directory to save the table
        coverage: Coverage percentage for the table caption
        source_csv_dir: Directory containing the CSV files
        landscape_mode: Whether to generate tables in landscape mode
        method_set: Method set to use
        current_variant_label: Current variant label for conditional updates (e.g., 'a', 'b', 'c')
    """
    print("\n=== Generating Uncertainty Metrics Table ===")
    
    # Get base dir for CSV files
    results_dir = source_csv_dir # Directly use the provided source_csv_dir
    
    # Define the models we want to extract uncertainty metrics for
    # Adapt based on context (standard run vs. variant run)
    is_variant_mode = method_set == 'final' and current_variant_label is not None

    if is_variant_mode:
        # For variant processing, only show uncertainty for the 'CLEAR' method ('clear')
        models = [
            ('clear', 'CLEAR')
        ]
        print(f"DEBUG: Uncertainty Table - Variant Mode (Variant: {current_variant_label}), showing only CLEAR (clear_residual).")
    else:
        # Standard mode: Show multiple CLEAR versions
        models = [
            ('clear_vanilla', 'CLEAR-Vanilla'),
            ('clear_vanilla_c', 'CLEAR-c_vanilla'),
            ('clear', 'CLEAR'),
            ('clear_c', 'CLEAR-R-c')
        ]
        print("DEBUG: Uncertainty Table - Standard Mode, showing multiple CLEAR versions.")

    
    # Dictionary to store uncertainty metrics for each dataset and model
    uncertainty_data = {}
    max_runs = 0
    
    # Process each dataset
    for dataset in datasets:
        print(f"Processing uncertainty metrics for dataset: {dataset}")
        
        # Convert to the format used in CSV files (may include "data_" prefix)
        csv_dataset = dataset
        if not csv_dataset.startswith("data_"):
            csv_dataset = f"data_{csv_dataset}"
        
        # Look for CSV file
        csv_files = glob.glob(os.path.join(results_dir, f"benchmark_results_{csv_dataset}_{coverage}.csv"))
        if not csv_files:
            print(f"  No CSV file found for dataset {dataset} with {coverage}% coverage")
            continue
            
        csv_file = csv_files[0]
        
        try:
            # Read the CSV file
            df = pd.read_csv(csv_file)
            
            # Initialize dataset in the data dictionary
            if dataset not in uncertainty_data:
                uncertainty_data[dataset] = {}
            
            # Process each model
            for model_key, model_name in models:
                # Initialize model in the dataset dictionary
                if model_key not in uncertainty_data[dataset]:
                    uncertainty_data[dataset][model_key] = {}
                
                # Extract uncertainty metrics
                aleatoric_values = df[(df['Dataset'] == csv_dataset) & 
                                     (df['Method'] == model_key) & 
                                     (df['Metric'] == 'total_aleatoric_calib')]['Value'].tolist()
                
                epistemic_values = df[(df['Dataset'] == csv_dataset) & 
                                     (df['Method'] == model_key) & 
                                     (df['Metric'] == 'total_epistemic_calib')]['Value'].tolist()
                
                ratio_values = df[(df['Dataset'] == csv_dataset) & 
                                 (df['Method'] == model_key) & 
                                 (df['Metric'] == 'uncertainty_ratio_calib')]['Value'].tolist()
                
                # Update max_runs
                max_runs = max(max_runs, len(aleatoric_values), len(epistemic_values), len(ratio_values))
                
                # Store metrics if available
                if aleatoric_values:
                    uncertainty_data[dataset][model_key]['aleatoric'] = {
                        'median': np.median(aleatoric_values),
                        'min': min(aleatoric_values),
                        'max': max(aleatoric_values)
                    }
                
                if epistemic_values:
                    uncertainty_data[dataset][model_key]['epistemic'] = {
                        'median': np.median(epistemic_values),
                        'min': min(epistemic_values),
                        'max': max(epistemic_values)
                    }
                
                if ratio_values:
                    uncertainty_data[dataset][model_key]['ratio'] = {
                        'median': np.median(ratio_values),
                        'min': min(ratio_values),
                        'max': max(ratio_values)
                    }
            
        except Exception as e:
            print(f"  Error processing CSV for dataset {dataset}: {e}")
    
    # Filter datasets to only include those with complete data
    datasets_with_data = []
    for dataset in datasets:
        if dataset in uncertainty_data:
            has_all_metrics = True
            for model_key, _ in models:
                if model_key not in uncertainty_data[dataset]:
                    has_all_metrics = False
                    break
                for metric in ['aleatoric', 'epistemic', 'ratio']:
                    if metric not in uncertainty_data[dataset][model_key]:
                        has_all_metrics = False
                        break
            if has_all_metrics:
                datasets_with_data.append(dataset)
    
    if not datasets_with_data:
        print("No datasets with complete uncertainty metrics found")
        return
    
    # Create table header
    # Add variant label to caption if in variant mode
    if is_variant_mode:
        caption = f"Variant ({current_variant_label}) uncertainty "
    else:
        caption = f"Uncertainty "

    caption += f"metrics for {coverage}\\% prediction intervals across {max_runs} seeds. Values are shown as median [min:max]. A = aleatoric uncertainty, E = epistemic uncertainty, E/A = uncertainty ratio."
    
    
    table_lines = [
        "\\begin{table}[!htbp]",
        "\\centering",
        r"\caption{" + caption + r"}",
        f"\\label{{tab:uncertainty_metrics_{coverage}" + (f"_variant_{current_variant_label}" if is_variant_mode else "") + "}",
        "\\small",
        "\\resizebox{\\columnwidth}{!}{%",
        # Adjust header based on number of models
        "\\begin{tabular}{l" + "ccc" * len(models) + "}",
        "\\toprule"
    ]

    # Dynamically create the header rows based on the models list
    header_row1 = "Dataset"
    header_row2 = ""
    rules = ""
    col_index = 2 # Start column index for multicolumn/cmidrule
    for model_key, model_display_name in models:
        header_row1 += f" & \\multicolumn{{3}}{{c}}{{{model_display_name}}}"
        header_row2 += f" & A & E & E/A"
        rules += f"\\cmidrule(lr){{{col_index}-{col_index+2}}}"
        col_index += 3
    header_row1 += " \\\\"
    header_row2 += " \\\\"

    table_lines.append(header_row1)
    table_lines.append(rules)
    table_lines.append(header_row2)
    table_lines.append("\\midrule")

    
    # Add data rows
    for dataset in sorted(datasets_with_data):
        # Format the dataset name for LaTeX display - escape underscores properly
        formatted_dataset = dataset.replace("_", "\\textunderscore ")
        row = [formatted_dataset]
        
        # Add data for each model
        for model_key, _ in models:
            # Add aleatoric metric
            if 'aleatoric' in uncertainty_data[dataset][model_key]:
                aleatoric_data = uncertainty_data[dataset][model_key]['aleatoric']
                row.append(f"{aleatoric_data['median']:.2f} [{aleatoric_data['min']:.2f}:{aleatoric_data['max']:.2f}]")
            else:
                row.append("-")
            
            # Add epistemic metric
            if 'epistemic' in uncertainty_data[dataset][model_key]:
                epistemic_data = uncertainty_data[dataset][model_key]['epistemic']
                row.append(f"{epistemic_data['median']:.2f} [{epistemic_data['min']:.2f}:{epistemic_data['max']:.2f}]")
            else:
                row.append("-")
            
            # Add ratio metric
            if 'ratio' in uncertainty_data[dataset][model_key]:
                ratio_data = uncertainty_data[dataset][model_key]['ratio']
                row.append(f"{ratio_data['median']:.2f} [{ratio_data['min']:.2f}:{ratio_data['max']:.2f}]")
            else:
                row.append("-")
        
        # Add the row to the table
        table_lines.append(" & ".join(row) + " \\\\")
    
    # Finish the table
    table_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "}",
        "\\end{table}"
    ])
    
    # Write the table to a file
    output_file = os.path.join(output_dir, f"table-uncertainty-metrics-{coverage}.tex")
    write_latex_table(table_lines, output_file, landscape_mode)
    
    print(f"Created uncertainty metrics table with {len(datasets_with_data)} datasets at {output_file}")

def process_multiple_result_folders(base_dirs, output_base_dir, variants, coverage=95, method_set='final', landscape_mode=False, uacqr_agg_csv=None):
    """
    Process multiple result folders and generate tables for each.
    
    Args:
        base_dirs: List of directories containing result CSV files
        output_base_dir: Base directory to save output tables
        variants: List of variant names corresponding to base_dirs
        coverage: Coverage percentage
        method_set: Method set to use
        landscape_mode: Whether to generate tables in landscape mode
        uacqr_agg_csv: Path to the aggregated UACQR results CSV.
    """
    # Ensure the output base directory exists
    os.makedirs(output_base_dir, exist_ok=True)
    
    for i, (result_dir, variant) in enumerate(zip(base_dirs, variants)):
        print(f"\n======= Processing Variant {variant} from {result_dir} =======")
        
        # Check if the directory exists and list contents
        if os.path.exists(result_dir):
            print(f"Directory exists. Contents:")
            for item in os.listdir(result_dir):
                if item.startswith("benchmark_results_") and item.endswith(f"_{coverage}.csv"):
                    print(f"  - {item}")
        else:
            print(f"WARNING: Directory {result_dir} does not exist!")
        
        # Create variant-specific output directory
        output_dir = os.path.join(output_base_dir, f"{variant}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Process this variant
        try:
            # Skip dataset stats extraction since we only need metric tables
            # Combine metric tables for this variant
            combine_metrics_tables(
                None,  # No separate_dir needed for CSV-based processing
                output_dir,
                coverage,
                4,  # decimal_places
                method_set,
                result_dir,  # Use this variant's result dir
                landscape_mode,
                current_variant_label=variant,
                uacqr_agg_csv=uacqr_agg_csv # Pass it here
            )
            
            # Also generate gamma-lambda and uncertainty tables for each variant
            valid_datasets_for_variant = combine_gamma_lambda_tables(
                output_dir,  # input_dir (for pre-existing .tex, likely won't find any here, will use CSVs)
                output_dir,  # output_dir for the new combined table
                coverage,
                result_dir,  # source_csv_dir for this variant
                method_set,  # Should be 'final' when processing variants
                landscape_mode,
                current_variant_label=variant # Pass variant label
            )

            if valid_datasets_for_variant:
                generate_uncertainty_metrics_table(
                    valid_datasets_for_variant,
                    output_dir, # output_dir for the new table
                    coverage,
                    result_dir, # source_csv_dir for this variant
                    landscape_mode,
                    method_set=method_set, # Pass method_set
                    current_variant_label=variant # Pass variant label
                )
            else:
                print(f"Skipping uncertainty metrics table for variant {variant} due to no valid datasets from gamma/lambda step.")

            print(f"Completed processing for variant {variant}")
        except Exception as e:
            print(f"Error processing variant {variant}: {e}")
            import traceback
            traceback.print_exc()

def main():
    # Setup default paths
    script_parent_dir = os.path.dirname(script_dir)
    base_dir = os.path.dirname(script_parent_dir)  # Base dir is two levels up from script
    
    # default_results_dir = os.path.join(base_dir, "data", "results")
    default_results_dir = os.path.join(base_dir, "models","pcs_top1_qpcs_10")
    default_separate_dir = os.path.join(base_dir, "paper", "tex_tbls", "separate")
    default_combined_dir = os.path.join(base_dir, "paper", "tex_tbls", "combined")

    # Default list of datasets
    default_datasets = "ailerons,airfoil,allstate,ca_housing,computer,concrete,diamond,elevator,energy_efficiency,insurance,kin8nm,miami_housing,naval_propulsion,parkinsons,powerplant,qsar,sulfur,superconductor"
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Extract dataset statistics and generate combined LaTeX tables.")
    parser.add_argument("--datasets", type=str, default=default_datasets, 
                        help="Dataset key(s) to evaluate. Can be a single dataset or comma-separated list.")
    parser.add_argument("--results_dir", type=str, default=default_results_dir, 
                        help="Directory containing ensemble results.")
    parser.add_argument("--separate_dir", type=str, default=default_separate_dir, 
                        help="Directory containing individual LaTeX tables.")
    parser.add_argument("--combined_dir", type=str, default=default_combined_dir, 
                        help="Directory to save combined LaTeX tables.")
    parser.add_argument("--coverage", type=int, default=90, 
                        help="Coverage percentage for table caption.")
    parser.add_argument("--decimal_places", type=int, default=4,
                        help="Default number of decimal places for metric values.")
    parser.add_argument("--skip-stats", action="store_true", default=False,
                        help="Skip dataset statistics extraction.")
    parser.add_argument("--skip-combine", action="store_true", default=False,
                        help="Skip combining tables.")
    parser.add_argument("--method_set", type=str, default="both",
                        help="Set of methods to include in tables: 'standard', 'residual', 'final', or 'both'")
    parser.add_argument("--source_csv_dir", type=str, default="../../results",
                        help="Directory containing the CSV files. Defaults to a common results subfolder.")
    parser.add_argument("--landscape_mode", action="store_true", default=False,
                        help="Whether to generate tables in landscape mode.")
    parser.add_argument("--process_variants", action="store_true", default=False,
                        help="Process multiple variant folders (a, b, c) as in plot_real_benchmark_results.py")
    parser.add_argument("--uacqr_agg_csv", type=str, default=None,
                        help="Path to the aggregated CSV file containing UACQR-P and UACQR-S results. Used with --process_variants for variant 'c'.")
    args = parser.parse_args()
    
    # Check if we should process variants
    if args.process_variants:
        print("\n=== Processing Multiple Result Folders for Variants A, B, C ===")
        
        # Define the three variant folders
        variant_folders = [
            "qPCS_all_10seeds_all",    # Variant A
            "qPCS_qxgb_10seeds_qxgb",   # Variant B
            "PCS_all_10seeds_qrf"      # Variant C
        ]
        
        # Handle source_csv_dir path
        if os.path.isabs(args.source_csv_dir):
            # If it's an absolute path, use it directly
            results_base_dir = args.source_csv_dir
        elif args.source_csv_dir.startswith("../") or args.source_csv_dir.startswith("..\\"):
            # If it starts with ../, it's relative to the script location
            results_base_dir = os.path.abspath(os.path.join(script_dir, args.source_csv_dir))
        else:
            # Otherwise, resolve it relative to the base_dir
            results_base_dir = os.path.join(base_dir, args.source_csv_dir)
            
        # Use normpath to clean up any path issues
        results_base_dir = os.path.normpath(results_base_dir)
        
        print(f"Looking for variant folders in: {results_base_dir}")
        
        # Full paths to the variant folders
        variant_paths = [os.path.join(results_base_dir, folder) for folder in variant_folders]
        
        # Variant labels
        variant_labels = ["a", "b", "c"]
        
        # Output directory for variants
        variant_output_dir = os.path.join(base_dir, "paper", "tex_tbls", "combined_variants")
        
        # Check if the variant folders exist
        for path in variant_paths:
            if not os.path.exists(path):
                print(f"WARNING: Variant folder doesn't exist: {path}")
        
        # Process the variants
        process_multiple_result_folders(
            variant_paths,
            variant_output_dir,
            variant_labels,
            args.coverage,
            "final",  # Always use final method set for variants
            args.landscape_mode,
            args.uacqr_agg_csv  # Pass the new argument
        )
        
        print("\nVariant processing complete! Check the table outputs at:")
        print(f"{variant_output_dir}/[a,b,c]/table-combined-{args.coverage}-*.tex")
        return
    
    # Original processing path for single folder
    # Create output directories if they don't exist
    os.makedirs(args.separate_dir, exist_ok=True)
    os.makedirs(args.combined_dir, exist_ok=True)
    
    # List of datasets with valid statistics
    all_datasets = []
    
    # Extract dataset statistics if not skipped
    if not args.skip_stats:
        print("\n=== Extracting Dataset Statistics ===")
        datasets = [d.strip() for d in args.datasets.split(",")]
        all_stats = []
        
        for dataset in datasets:
            print(f"Processing dataset: {dataset}")
            stats = extract_dataset_stats(dataset, args.results_dir, args.coverage)
            if stats:
                all_stats.append(stats)
                all_datasets.append(dataset)
        
        # Generate dataset statistics table
        if all_stats:
            print(f"Generating dataset statistics table for {args.coverage}% coverage")
            generate_dataset_stats_table(all_stats, args.combined_dir, args.coverage, args.landscape_mode)
        else:
            print(f"No statistics extracted for {args.coverage}% coverage. Check dataset names and results directory.")
    
    # If no datasets were found through statistics extraction, use the list from args.datasets
    if not all_datasets:
        all_datasets = [d.strip() for d in args.datasets.split(",")]
    
    # Combine tables if not skipped
    if not args.skip_combine:
        print("\n=== Combining Metric Tables ===")
        
        # Determine which method sets to process
        method_sets_to_process = []
        if args.method_set.lower() == "both":
            method_sets_to_process = ["standard", "residual"]
        elif args.method_set.lower() == "all":
            method_sets_to_process = ["standard", "residual", "final"]
        else:
            method_sets_to_process = [args.method_set]
        
        # Dictionary to track datasets with valid values for uncertainty metrics table
        datasets_with_valid_values = {}
        
        # Process each method set
        for current_method_set in method_sets_to_process:
            print(f"\n--- Processing {current_method_set} method set ---")
            
            # Combine metric tables for the current method set
            combine_metrics_tables(
                args.separate_dir, 
                args.combined_dir, 
                args.coverage, 
                args.decimal_places, 
                current_method_set, 
                args.source_csv_dir, 
                args.landscape_mode,
                None,  # current_variant_label, not applicable here
                args.uacqr_agg_csv if current_method_set == 'final' else None # Pass only if relevant
            )
            
            # Combine gamma and lambda tables for the current method set
            valid_datasets = combine_gamma_lambda_tables(
                args.separate_dir, 
                args.combined_dir, 
                args.coverage, 
                args.source_csv_dir, 
                current_method_set, 
                args.landscape_mode
            )
            
            # Store datasets with valid values for this method set
            if valid_datasets:
                datasets_with_valid_values[current_method_set] = valid_datasets
        
        # Generate uncertainty metrics table using the datasets with valid values from any method set
        all_valid_datasets = set()
        for datasets in datasets_with_valid_values.values():
            if datasets:
                all_valid_datasets.update(datasets)
        
        if all_valid_datasets:
            generate_uncertainty_metrics_table(
                list(all_valid_datasets),
                args.combined_dir,
                args.coverage,
                args.source_csv_dir,
                args.landscape_mode
            )
        
    # Display completion message
    print("\nProcessing complete! Check the table outputs at:")
    
    if args.method_set.lower() == "both":
        print(f"Metric tables created with standard method set: {args.combined_dir}/table-combined-{args.coverage}-*-standard.tex")
        print(f"Metric tables created with residual method set: {args.combined_dir}/table-combined-{args.coverage}-*-residual.tex")
        if datasets_with_valid_values.get("standard") or datasets_with_valid_values.get("residual"):
            print(f"Gamma-lambda table (standard): {args.combined_dir}/table-combined-{args.coverage}-stats-gamma-lambda-standard.tex")
            print(f"Gamma-lambda table (residual): {args.combined_dir}/table-combined-{args.coverage}-stats-gamma-lambda-residual.tex")
        if all_valid_datasets:
            print(f"Uncertainty metrics table: {args.combined_dir}/table-uncertainty-metrics-{args.coverage}.tex")
        print(f"Note: Calibration parameters for residual tables are from CLEAR-R instead of CLEAR")
    elif args.method_set.lower() == "all":
        print(f"Metric tables created with standard, residual, and final method sets: {args.combined_dir}/table-combined-{args.coverage}-*-[method_set].tex")
        if all_valid_datasets:
            print(f"Uncertainty metrics table: {args.combined_dir}/table-uncertainty-metrics-{args.coverage}.tex")
    else:
        method_description = args.method_set
        print(f"Metric tables created with {method_description} method set: {args.combined_dir}/table-combined-{args.coverage}-*-{args.method_set}.tex")
        if datasets_with_valid_values.get(args.method_set):
            print(f"Gamma-lambda table: {args.combined_dir}/table-combined-{args.coverage}-stats-gamma-lambda-{args.method_set}.tex")
        if all_valid_datasets:
            print(f"Uncertainty metrics table: {args.combined_dir}/table-uncertainty-metrics-{args.coverage}.tex")
        if args.method_set == "residual":
            print(f"Calibration parameters (lambda, gamma) are from CLEAR-R instead of CLEAR")
    
    print(f"Note: Only datasets with valid lambda and gamma values in the CSV files are included in the gamma-lambda and uncertainty metrics tables.")

if __name__ == "__main__":
    main() 

## To get the all table of the results at once
# python combine_real_benchmark_results.py --method_set final --coverage 95 --process_variants
# To get the variant c table with UACQR-S and UACQR-P
# python combine_real_benchmark_results.py --process_variants --coverage 95 --uacqr_agg_csv ../../results/uacqr_benchmark_results.csv