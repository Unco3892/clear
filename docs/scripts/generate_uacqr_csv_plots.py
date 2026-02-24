#!/usr/bin/env python
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import argparse
import re

# Global list to store normalized data for a potential future summary CSV (optional)
_ALL_NORMALIZED_DATA_FOR_CSV = []

def apply_consistent_font_settings():
    """Apply consistent font settings for all plots."""
    plt.style.use('seaborn-v0_8-whitegrid')
    mpl.rcParams.update({
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 17,
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "savefig.dpi": 600,
        "font.family": "serif",
        "font.serif": ["Palatino", "Times New Roman", "DejaVu Serif"],
    })

def load_specific_metric_data(csv_path, metric_col_name, coverage_target_float):
    """
    Load data for a specific metric from the uacqr_benchmark_results.csv file.
    
    Args:
        csv_path: Path to the uacqr_benchmark_results.csv file.
        metric_col_name: The name of the column for the metric (e.g., 'NCIW', 'QuantileLoss').
        coverage_target_float: The coverage target as a float (e.g., 0.95).
        
    Returns:
        DataFrame with columns: Dataset, Method, mean, std.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return pd.DataFrame()

    # Filter by Coverage_Target
    # Need to be careful with float comparisons
    df_filtered = df[np.isclose(df['Coverage_Target'], coverage_target_float)]

    if df_filtered.empty:
        print(f"Warning: No data found for Coverage_Target ~ {coverage_target_float} in {csv_path}")
        return pd.DataFrame()

    if metric_col_name not in df_filtered.columns:
        print(f"Warning: Metric column '{metric_col_name}' not found in the CSV.")
        return pd.DataFrame()

    # Group by Dataset and Method, then calculate mean and std for the metric
    # The CSV already contains one row per Dataset, Seed, Method. We need to average over seeds.
    agg_df = df_filtered.groupby(['Dataset', 'Method']).agg(
        mean=(metric_col_name, 'mean'),
        std=(metric_col_name, 'std')
    ).reset_index()

    # Format dataset names for better readability
    agg_df['Dataset'] = agg_df['Dataset'].str.replace('_', ' ').str.title()
    
    return agg_df

def normalize_data(df, baseline_method_internal, method_display_map):
    """
    Normalize data relative to the baseline method.
    Args:
        df: DataFrame with benchmark data (columns: Dataset, Method, mean, std)
        baseline_method_internal: Internal name of the method to use as baseline (lowercase).
        method_display_map: Mapping of internal method names (lowercase) to display names.
    Returns:
        DataFrame with normalized data.
    """
    df_copy = df.copy()
    df_copy['Method_Original'] = df_copy['Method'] # Keep original method name for display mapping
    df_copy['Method'] = df_copy['Method'].str.lower() # Normalize internal method names for comparison
    
    norm_data = []
    
    for dataset in df_copy['Dataset'].unique():
        dataset_data = df_copy[df_copy['Dataset'] == dataset].copy()
        
        baseline = dataset_data[dataset_data['Method'] == baseline_method_internal.lower()]
        if baseline.empty:
            print(f"Warning: Baseline method '{baseline_method_internal}' not found for dataset '{dataset}'. Skipping normalization for this dataset.")
            continue
            
        base_mean = baseline['mean'].values[0]
        base_std = baseline['std'].values[0]
        
        for _, row in dataset_data.iterrows():
            method_mean = row['mean']
            method_std = row['std']
            
            normalized_mean = method_mean / base_mean if base_mean != 0 else (1 if method_mean == 0 else float('inf'))
            
            if base_mean != 0 and method_mean != 0:
                normalized_std = np.sqrt((method_std / method_mean)**2 + (base_std / base_mean)**2) * normalized_mean
            else:
                normalized_std = 0 # Or some other indicator of undefined relative std

            if method_mean == base_mean:
                relative_improvement = 0.0
            elif base_mean != 0:
                relative_improvement = (method_mean - base_mean) / base_mean * 100.0
            elif method_mean != 0:
                relative_improvement = 100.0 if method_mean > 0 else -100.0 # Base is 0, method is not.
            else: 
                relative_improvement = 0.0
                
            norm_data.append({
                'Dataset': dataset,
                'Method': row['Method_Original'], # Use original case for mapping to display name
                'mean': method_mean,
                'std': method_std,
                'normalized_mean': normalized_mean,
                'normalized_std': normalized_std,
                'relative_improvement': relative_improvement
            })
            
    norm_df = pd.DataFrame(norm_data)
    if norm_df.empty:
        return norm_df

    # Map method names to display names
    norm_df['Method_Display'] = norm_df['Method'].map(lambda x: method_display_map.get(x.lower(), x))
    
    global _ALL_NORMALIZED_DATA_FOR_CSV # Optional: collect data if needed later
    if not norm_df.empty:
         _ALL_NORMALIZED_DATA_FOR_CSV.append(norm_df.copy())
            
    return norm_df

def create_plot(df, metric_name, ordered_methods_display, palette, baseline_method_display_name,
                output_file_base=None, y_axis_from_zero=False, error_bar_type='std',
                fig_to_use=None, ax_to_use=None, make_legend=True, save_fig_if_internal=True,
                inset_config_from_caller=None, set_styling_if_creating_fig=True, output_format="png",
                predefined_dataset_order=None):
    """
    Create a publication-ready plot with the normalized data.
    """
    fig_created_internally = False
    if fig_to_use is None or ax_to_use is None:
        fig = plt.figure(figsize=(12, 8))
        ax_main = fig.add_subplot(111)
        fig_created_internally = True
        if set_styling_if_creating_fig:
            apply_consistent_font_settings()
    else:
        fig = fig_to_use
        ax_main = ax_to_use

    if df.empty:
        print(f"Warning: DataFrame is empty for metric {metric_name}. Skipping plot elements.")
        if fig_created_internally and save_fig_if_internal: plt.close(fig)
        return fig, ax_main, None, ([], [])

    # Filter for just the methods we want (based on display names)
    df_filtered = df[df['Method_Display'].isin(ordered_methods_display)].copy()
    if df_filtered.empty:
        print(f"Warning: No data for methods {ordered_methods_display} in metric {metric_name}. Skipping plot.")
        if fig_created_internally and save_fig_if_internal: plt.close(fig)
        return fig, ax_main, None, ([], [])

    # Determine dataset order
    if predefined_dataset_order:
        # Filter to only include datasets present in the current filtered data
        current_datasets_in_df = df_filtered['Dataset'].unique()
        sorted_datasets = [ds for ds in predefined_dataset_order if ds in current_datasets_in_df]
        if not sorted_datasets: # Fallback if no overlap
             max_by_dataset = df_filtered.groupby('Dataset')['normalized_mean'].max().sort_values()
             sorted_datasets = max_by_dataset.index.tolist()
    else:
        max_by_dataset = df_filtered.groupby('Dataset')['normalized_mean'].max().sort_values()
        sorted_datasets = max_by_dataset.index.tolist()
    
    df_filtered['dataset_index'] = pd.Categorical(df_filtered['Dataset'], categories=sorted_datasets, ordered=True)
    df_filtered.sort_values('dataset_index', inplace=True)
    # Convert categorical to numerical index for plotting
    df_filtered['dataset_idx_num'] = df_filtered['dataset_index'].cat.codes


    n_methods = len(ordered_methods_display)
    bar_width = 0.75 / n_methods
    
    plot_handles = []
    plot_labels = []

    for i, method_disp_name in enumerate(ordered_methods_display):
        method_data = df_filtered[df_filtered['Method_Display'] == method_disp_name]
        if method_data.empty:
            continue

        x_pos = method_data['dataset_idx_num'] + (i - n_methods/2 + 0.5) * bar_width
        
        yerr_values = method_data['normalized_std']
        if error_bar_type == 'ci95': # Original script had this reversed for some reason
             yerr_values = method_data['normalized_std'] * 1.96 


        bar_container = ax_main.bar(
            x_pos, method_data['normalized_mean'], width=bar_width*0.8,
            label=method_disp_name, color=palette.get(method_disp_name, f'C{i}'),
            alpha=0.85
        )
        if bar_container:
            plot_handles.append(bar_container[0])
            plot_labels.append(method_disp_name)
        
        ax_main.errorbar(
            x_pos, method_data['normalized_mean'], yerr=yerr_values,
            fmt='none', color='black', linewidth=1, capsize=3
        )
    
    ax_main.set_xticks(range(len(sorted_datasets)))
    ax_main.set_xticklabels(sorted_datasets, rotation=45, ha='right', fontsize=14)
    ax_main.set_ylabel(f"{metric_name} (Normalized for {baseline_method_display_name} = 1)", fontsize=18)
    ax_main.axhline(y=1, color='black', linestyle='--', linewidth=0.8, alpha=0.7)
    
    y_max_val = df_filtered['normalized_mean'].max() if not df_filtered.empty else 1.8
    y_max = max(y_max_val * 1.05, 1.8)
    if y_max > 3.5: y_max = 3.5
    y_min = 0.65
    if y_axis_from_zero: y_min = 0
    ax_main.set_ylim(y_min, y_max)
    
    # Simplified y-ticks
    current_yticks = np.linspace(y_min, y_max, num=5 if y_max > 1.5 else 4) 
    ax_main.set_yticks(current_yticks)

    ax_main.grid(axis='y', linestyle='--', alpha=0.3)
    ax_main.grid(axis='x', alpha=0.1)
    for spine in ax_main.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
    
    if make_legend and plot_handles:
        legend_ncol = max(1, len(plot_handles) // 2 if len(plot_handles) > 4 else len(plot_handles))
        ax_main.legend(
            handles=plot_handles, labels=plot_labels,
            loc='upper center', bbox_to_anchor=(0.5, 0.97), ncol=legend_ncol,
            frameon=True, fancybox=False, edgecolor='black', fontsize=15
        )
    
    # --- Inset Plot ---
    ax_inset = None
    # Exclude baseline method from improvement calculation for inset
    improvement_df = df_filtered[df_filtered['Method_Display'] != baseline_method_display_name].copy()

    if not improvement_df.empty:
        default_inset_placement = {'subplot_y_bottom': 0.0, 'subplot_height': 1.0, 'is_part_of_combined': False}
        current_inset_placement = default_inset_placement.copy()
        if inset_config_from_caller:
            current_inset_placement.update(inset_config_from_caller)

        boxplot_height_orig = 0.40
        boxplot_width_orig = 0.32
        boxplot_bottom_orig = 0.48 
        boxplot_left_orig = 0.16

        fig_coord_inset_left = boxplot_left_orig
        fig_coord_inset_width = boxplot_width_orig
        fig_coord_inset_bottom = current_inset_placement['subplot_y_bottom'] + \
                                 boxplot_bottom_orig * current_inset_placement['subplot_height']
        fig_coord_inset_height = boxplot_height_orig * current_inset_placement['subplot_height']
        
        ax_inset = fig.add_axes([fig_coord_inset_left, fig_coord_inset_bottom, fig_coord_inset_width, fig_coord_inset_height])
        
        methods_for_boxplot_display = [m for m in ordered_methods_display if m != baseline_method_display_name]
        
        improvement_df_for_boxplot = improvement_df[improvement_df['relative_improvement'] <= 300].copy()

        if not improvement_df_for_boxplot.empty and methods_for_boxplot_display:
            improvement_df_for_boxplot['Method_Order'] = pd.Categorical(
                improvement_df_for_boxplot['Method_Display'], categories=methods_for_boxplot_display, ordered=True
            )
            improvement_df_for_boxplot = improvement_df_for_boxplot.sort_values('Method_Order')
            
            avg_improvements = improvement_df_for_boxplot.groupby('Method_Display', observed=True)['relative_improvement'].mean()
            
            min_val_in_plot = improvement_df_for_boxplot['relative_improvement'].min()
            max_val_in_plot = improvement_df_for_boxplot['relative_improvement'].max()
            y_min_inset = min_val_in_plot - 5
            y_max_inset_initial = max(max_val_in_plot + 20, 85)
            ax_inset.set_ylim(y_min_inset, y_max_inset_initial)
            
            sns.boxplot(
                data=improvement_df_for_boxplot, x='Method_Display', y='relative_improvement', 
                ax=ax_inset, palette=palette, width=0.6, fliersize=0,
                order=methods_for_boxplot_display, showmeans=False, whis=[0, 100]
            )
            
            # Simplified inset label positioning
            label_y_pos = y_max_inset_initial * 0.95 # Position near top of inset
            if metric_name.lower() == "quantile loss": label_y_pos = y_max_inset_initial * 0.97 
            if metric_name.lower() == "nciw": label_y_pos = y_max_inset_initial * 0.98


            font_size_inset_percentage = 12 if not current_inset_placement.get('is_part_of_combined') else 10.5
            for i_m, meth_disp_name in enumerate(methods_for_boxplot_display):
                if meth_disp_name in avg_improvements:
                    ax_inset.text(
                        i_m, label_y_pos, f"{avg_improvements[meth_disp_name]:.1f}%", 
                        ha='center', va='top', fontsize=font_size_inset_percentage,
                        weight='bold', color='black',
                        bbox=dict(facecolor='white', alpha=0.9, edgecolor='lightgray', pad=2.0, boxstyle="round,pad=0.3")
                    )
            ax_inset.set_ylim(y_min_inset, y_max_inset_initial) # Re-apply ylim after text

            font_size_inset_ticks = 12 if not current_inset_placement.get('is_part_of_combined') else 10
            font_size_inset_label = 14 if not current_inset_placement.get('is_part_of_combined') else 12
            font_size_inset_note = 12 if not current_inset_placement.get('is_part_of_combined') else 12

            yticks_inset = ax_inset.get_yticks()
            ax_inset.set_yticklabels([f"{int(y)}%" for y in yticks_inset], fontsize=font_size_inset_ticks)
            ax_inset.xaxis.set_ticks_position('top')
            ax_inset.xaxis.set_label_position('top')
            ax_inset.set_xticklabels(methods_for_boxplot_display, fontsize=font_size_inset_ticks)
            
            note_text = f"Relative increase over    {baseline_method_display_name} (%)"
            ax_inset.text(0.5, -0.02, note_text, ha='center', va='top', fontsize=font_size_inset_note,
                          style='italic', transform=ax_inset.transAxes)
            baseline_color_for_note = palette.get(baseline_method_display_name, 'grey')
            ax_inset.plot(0.592, -0.045, marker='s', markersize=7, color=baseline_color_for_note, 
                          transform=ax_inset.transAxes, clip_on=False, linestyle='None')

            ax_inset.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
            ax_inset.set_ylabel("Relative Improvement (%)", fontsize=font_size_inset_label, labelpad=2)
            ax_inset.set_xlabel("")
            ax_inset.tick_params(axis='both', which='major', labelsize=font_size_inset_ticks, pad=2)
            ax_inset.grid(axis='y', linestyle='--', alpha=0.3)
            ax_inset.grid(axis='x', alpha=0.1)
            ax_inset.set_zorder(10)
        else:
            print(f"Warning: No data for inset boxplot ({metric_name}). Skipping inset.")
            if ax_inset: ax_inset.set_visible(False) # Hide if created but no data


    if fig_created_internally and save_fig_if_internal and output_file_base:
        # Ensure metric name is filename-safe
        safe_metric_name = re.sub(r'[^a-zA-Z0-9_]', '', metric_name.replace(' ', '_')).lower()
        final_output_file_base = f"{output_file_base}_{safe_metric_name}"
        
        if output_format in ['pdf', 'both']:
            pdf_file = f"{final_output_file_base}.pdf"
            plt.savefig(pdf_file, dpi=600, bbox_inches='tight')
            print(f"Figure saved to {pdf_file}")
        if output_format in ['png', 'both']:
            png_file = f"{final_output_file_base}.png"
            plt.savefig(png_file, dpi=800, bbox_inches='tight')
            print(f"Figure saved to {png_file}")
        plt.close(fig)

    return fig, ax_main, ax_inset, (plot_handles, plot_labels)


def create_combined_uacqr_plot(csv_path, coverage_val, baseline_method_internal,
                               ordered_methods_internal, method_display_map, palette,
                               output_file_prefix, output_dir, alpha_val,
                               error_bar_type='std', y_axis_from_zero=False, output_format="png",
                               upper_plot_show_xlabels=False):
    """
    Creates a combined plot with Quantile Loss on top and NCIW at the bottom.
    """
    print(f"\n--- Generating Combined Plot from {csv_path} ---")
    apply_consistent_font_settings()

    metric_display_names = {"QuantileLoss": "Quantile Loss", "NCIW": "NCIW"}
    
    # Load data for Quantile Loss
    data_qgloss = load_specific_metric_data(csv_path, "QuantileLoss", coverage_val)
    if data_qgloss.empty:
        print("Error: Could not load Quantile Loss data. Aborting combined plot.")
        return
    norm_data_qgloss = normalize_data(data_qgloss, baseline_method_internal, method_display_map)

    # Load data for NCIW
    data_nciw = load_specific_metric_data(csv_path, "NCIW", coverage_val)
    if data_nciw.empty:
        print("Error: Could not load NCIW data. Aborting combined plot.")
        return
    norm_data_nciw = normalize_data(data_nciw, baseline_method_internal, method_display_map)

    if norm_data_qgloss.empty or norm_data_nciw.empty:
        print("Error: Normalization failed for one or both metrics. Aborting combined plot.")
        return

    # Determine common dataset order
    common_datasets = set(norm_data_qgloss['Dataset'].unique()).intersection(set(norm_data_nciw['Dataset'].unique()))
    if not common_datasets:
        print("Error: No common datasets between Quantile Loss and NCIW data. Aborting combined plot.")
        return
        
    temp_df_qgloss = norm_data_qgloss[norm_data_qgloss['Dataset'].isin(common_datasets)]
    temp_df_nciw = norm_data_nciw[norm_data_nciw['Dataset'].isin(common_datasets)]
    
    max_vals_qgloss = temp_df_qgloss.groupby('Dataset')['normalized_mean'].max()
    max_vals_nciw = temp_df_nciw.groupby('Dataset')['normalized_mean'].max()
    
    combined_max_vals = pd.concat([max_vals_qgloss, max_vals_nciw], axis=1).max(axis=1).sort_values()
    sorted_datasets = combined_max_vals.index.tolist()
    
    print(f"  Common dataset order for plots: {sorted_datasets}")

    fig = plt.figure(figsize=(12, 14))
    ax_top = fig.add_axes([0.1, 0.55, 0.85, 0.4])
    ax_bottom = fig.add_axes([0.1, 0.1, 0.85, 0.4])

    ordered_methods_display = [method_display_map[m.lower()] for m in ordered_methods_internal if m.lower() in method_display_map]
    baseline_method_display = method_display_map.get(baseline_method_internal.lower(), baseline_method_internal)

    # Quantile Loss Plot (Top)
    inset_config_top = {'subplot_y_bottom': 0.48, 'subplot_height': 0.5, 'is_part_of_combined': True}
    _, _, _, (handles_top, labels_top) = create_plot(
        norm_data_qgloss, metric_display_names["QuantileLoss"], ordered_methods_display, palette, baseline_method_display,
        y_axis_from_zero=y_axis_from_zero, error_bar_type=error_bar_type,
        fig_to_use=fig, ax_to_use=ax_top, make_legend=False, save_fig_if_internal=False,
        inset_config_from_caller=inset_config_top, set_styling_if_creating_fig=False,
        predefined_dataset_order=sorted_datasets
    )

    # NCIW Plot (Bottom)
    inset_config_bottom = {'subplot_y_bottom': 0.03, 'subplot_height': 0.5, 'is_part_of_combined': True}
    create_plot(
        norm_data_nciw, metric_display_names["NCIW"], ordered_methods_display, palette, baseline_method_display,
        y_axis_from_zero=y_axis_from_zero, error_bar_type=error_bar_type,
        fig_to_use=fig, ax_to_use=ax_bottom, make_legend=False, save_fig_if_internal=False,
        inset_config_from_caller=inset_config_bottom, set_styling_if_creating_fig=False,
        predefined_dataset_order=sorted_datasets
    )
    
    # Shared X-axis configuration
    x_ticks = list(range(len(sorted_datasets)))
    x_limits = (-0.5, len(sorted_datasets) - 0.5)
    ax_bottom.set_xlim(x_limits)
    ax_bottom.set_xticks(x_ticks)
    ax_bottom.set_xticklabels(sorted_datasets, rotation=45, ha='right', fontsize=14)
    
    ax_top.set_xlim(x_limits)
    ax_top.set_xticks(x_ticks)
    if upper_plot_show_xlabels: # Conditional display of top x-axis labels
        ax_top.set_xticklabels(sorted_datasets, rotation=45, ha='right', fontsize=14)
    else:
        ax_top.set_xticklabels([])
        ax_top.set_xlabel('')


    # Shared Legend
    if handles_top:
        legend_ncol = max(1, len(handles_top) // 2 if len(handles_top) > 4 else len(handles_top))
        fig.legend(handles=handles_top, labels=labels_top, loc='center', bbox_to_anchor=(0.5, 0.525),
                   ncol=legend_ncol, frameon=True, fancybox=True, edgecolor='black', fontsize=14)

    # Save combined figure
    os.makedirs(output_dir, exist_ok=True)
    combined_output_file_base = os.path.join(output_dir, f"{output_file_prefix}_alpha{int(alpha_val*100)}_combined")
    
    if output_format in ['pdf', 'both']:
        pdf_file = f"{combined_output_file_base}.pdf"
        plt.savefig(pdf_file, dpi=600, bbox_inches='tight')
        print(f"Combined figure saved to {pdf_file}")
    if output_format in ['png', 'both']:
        png_file = f"{combined_output_file_base}.png"
        plt.savefig(png_file, dpi=800, bbox_inches='tight')
        print(f"Combined figure saved to {png_file}")
    plt.close(fig)
    print(f"--- Finished Combined Plot from {csv_path} ---")


def generate_all_metrics_summary_table(csv_path, coverage_target_float, output_table_csv_path):
    """
    Generates a CSV table summarizing all metrics (mean and std) for each dataset and method.
    """
    print(f"\n--- Generating All Metrics Summary Table from {csv_path} ---")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return

    df_filtered = df[np.isclose(df['Coverage_Target'], coverage_target_float)]
    if df_filtered.empty:
        print(f"Warning: No data found for Coverage_Target ~ {coverage_target_float}. Cannot generate summary table.")
        return

    metrics_to_summarize = ['PICP', 'NIW', 'MPIW', 'QuantileLoss', 'CRPS', 'AUC', 'NCIW']
    
    # Check if all metric columns exist
    for metric in metrics_to_summarize:
        if metric not in df_filtered.columns:
            print(f"Warning: Metric column '{metric}' not found. It will be excluded from the summary table.")
    
    # Filter metrics_to_summarize to only include existing columns
    existing_metrics = [m for m in metrics_to_summarize if m in df_filtered.columns]
    if not existing_metrics:
        print("Warning: No specified metric columns exist in the data. Cannot generate summary table.")
        return

    # Group by Dataset and Method, then calculate mean and std for each metric
    # The CSV already contains one row per Dataset, Seed, Method. We need to average over seeds.
    grouped = df_filtered.groupby(['Dataset', 'Method'])
    
    summary_dfs = []
    for metric in existing_metrics:
        metric_summary = grouped[metric].agg(['mean', 'std']).reset_index()
        metric_summary.rename(columns={'mean': f'{metric}_mean', 'std': f'{metric}_std'}, inplace=True)
        summary_dfs.append(metric_summary)

    # Merge all summary DataFrames
    if not summary_dfs:
        print("No metric summaries to merge. Cannot generate summary table.")
        return

    final_summary_df = summary_dfs[0]
    for i in range(1, len(summary_dfs)):
        final_summary_df = pd.merge(final_summary_df, summary_dfs[i], on=['Dataset', 'Method'], how='outer')
        
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_table_csv_path), exist_ok=True)
    
    final_summary_df.to_csv(output_table_csv_path, index=False, float_format='%.4f')
    print(f"All metrics summary table saved to: {output_table_csv_path}")
    print(f"--- Finished All Metrics Summary Table ---")


def main():
    parser = argparse.ArgumentParser(description="Generate plots and tables from uacqr_benchmark_results.csv.")
    parser.add_argument("--csv_path", type=str, default="results/uacqr_benchmark_results.csv",
                        help="Path to the uacqr_benchmark_results.csv file.")
    parser.add_argument("--output_dir", type=str, default="plots_and_tables/uacqr_summary",
                        help="Directory to save output plots and tables.")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Alpha level for confidence intervals (e.g., 0.05 for 95%% coverage).")
    parser.add_argument("--output_format", type=str, default="png", choices=["pdf", "png", "both"],
                        help="Output format for saving figures.")
    parser.add_argument("--upper_plot_show_xlabels", action='store_true', default=False,
                        help="Show x-axis labels on the upper plot in combined plots.")
    
    args = parser.parse_args()

    coverage_val = 1.0 - args.alpha

    # Define methods, display names, and palette
    # Ensure internal names match those in the CSV's 'Method' column for loading
    # And ensure lowercase versions are used for map keys if normalize_data uses lowercase
    ordered_methods_internal = ['clear_residual', 'UACQR-P', 'UACQR-S']
    method_display_map = {
        'clear_residual': 'CLEAR', # Baseline
        'uacqr-p': 'UACQR-P',       # Note: CSV has 'UACQR-P', normalize_data converts to 'uacqr-p' for map key
        'uacqr-s': 'UACQR-S'      # Note: CSV has 'UACQR-S', normalize_data converts to 'uacqr-s' for map key
    }
    palette = {
        'CLEAR': '#4C72B0',      # Seaborn Blue
        'UACQR-P': '#D62728',     # Tableau Red
        'UACQR-S': '#009E73'    # Bluish Green
    }
    baseline_method_internal = 'clear_residual' # This is the key used internally for normalization

    # Create combined plot
    create_combined_uacqr_plot(
        csv_path=args.csv_path,
        coverage_val=coverage_val,
        baseline_method_internal=baseline_method_internal,
        ordered_methods_internal=ordered_methods_internal,
        method_display_map=method_display_map,
        palette=palette,
        output_file_prefix="uacqr_benchmark_summary",
        output_dir=args.output_dir,
        alpha_val=args.alpha,
        error_bar_type='std',
        y_axis_from_zero=False,
        output_format=args.output_format,
        upper_plot_show_xlabels=args.upper_plot_show_xlabels
    )

    # Generate all metrics summary table
    summary_table_path = os.path.join(args.output_dir, f"all_metrics_summary_alpha{int(args.alpha*100)}.csv")
    generate_all_metrics_summary_table(
        csv_path=args.csv_path,
        coverage_target_float=coverage_val,
        output_table_csv_path=summary_table_path
    )

    print("\nScript execution finished.")

if __name__ == "__main__":
    main() 