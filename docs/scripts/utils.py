#!/usr/bin/env python
import os
import re

def format_metric_name(metric):
    if metric == "quantile_loss" or metric == "quantileloss":
        return "Quantile Loss"
    elif metric == "expectile_loss" or metric == "expectileloss":
        return "Expectile Loss"
    elif metric == "picp":
        return "PICP"
    elif metric == "niw":
        return "NIW"
    elif metric == "crps":
        return "CRPS"
    elif metric == "auc":
        return "AUC"
    elif metric == "nciw":
        return "NCIW"
    elif metric == "lambda":
        return "$\\lambda$"
    elif metric == "gamma":
        return "$\\gamma_1$"
    elif metric == "mpiw":
        return "MPIW"
    elif metric == "interval_score_loss" or metric == "intervalscoreloss":
        return "Interval Score Loss"
    else:
        return metric.upper().replace("_", " ")
        
def write_latex_table(table_lines, output_file, landscape_mode=False):
    """Write a LaTeX table to a file, optionally in landscape mode."""
    # If landscape mode is enabled, modify table lines
    if landscape_mode:
        # Find where the table begins
        begin_table_idx = -1
        for i, line in enumerate(table_lines):
            if line.startswith("\\begin{table}"):
                begin_table_idx = i
                break
        
        if begin_table_idx >= 0:
            # Insert landscape begin before table
            table_lines.insert(begin_table_idx, "\\begin{landscape}")
            
            # Replace standard table with adjusted row height
            table_lines[begin_table_idx+1] = "\\begin{table}[htbp]"
            table_lines.insert(begin_table_idx+2, "    \\renewcommand{\\arraystretch}{1.2} % slightly increase row height")
            
            # Find where the table ends and add landscape end
            for i in range(len(table_lines)):
                if i >= begin_table_idx and table_lines[i].startswith("\\end{table}"):
                    table_lines.insert(i+1, "\\end{landscape}")
                    break
    
    # Write table lines to file (use newline='\n' for consistent LF line endings)
    with open(output_file, 'w', encoding="utf-8", newline='\n') as f:
        f.write("\n".join(table_lines))

def extract_data_from_table(file_path, pattern):
    """
    Extract data from a LaTeX table file using a regex pattern.
    
    Args:
        file_path: Path to the LaTeX table file
        pattern: Regex pattern to extract data rows
        
    Returns:
        list: Extracted data rows matching the pattern
    """
    with open(file_path, 'r', encoding="utf-8") as f:
        content = f.read()
    
    return re.findall(pattern, content, re.DOTALL) 