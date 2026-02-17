import subprocess
import pandas as pd
import os
import sys

# Define the root directory
current_file = os.path.abspath(__file__)
case_study_dir = os.path.dirname(current_file)
src_dir = os.path.dirname(case_study_dir)
root_dir = os.path.dirname(src_dir)
results_dir = os.path.join(root_dir, 'results', 'case_study')

# Define the script to run
script_path = os.path.join(case_study_dir, 'ames_clear_case_study.py')

# Define the experiments
experiments = [
    {
        "name": "2 features",
        "args": ["--aleatoric_model", "linear", "--use_top_features"],
        "output_suffix": "top2_linear",
        "description": "2 features"
    },
    {
        "name": "All features",
        "args": ["--aleatoric_model", "linear"],
        "output_suffix": "all_linear",
        "description": "All features"
    },
    {
        "name": "All features 50% data",
        "args": ["--aleatoric_model", "linear", "--subsample_train", "0.5"],
        "output_suffix": "all_linear_subsample_0.5",
        "description": "All features\\\\\n50% data" # Latex newline
    },
    {
        "name": "All features 20% data",
        "args": ["--aleatoric_model", "linear", "--subsample_train", "0.2"],
        "output_suffix": "all_linear_subsample_0.2",
        "description": "All features\\\\\n20% data" # Latex newline
    }
]

def run_experiment(experiment):
    print(f"Running experiment: {experiment['name']}")
    cmd = [sys.executable, script_path] + experiment["args"]
    # Run from case_study_dir so that relative paths in prepare_ames_data.py work (../../data)
    subprocess.run(cmd, check=True, cwd=case_study_dir)
    
    # Construct expected output filename
    output_file = os.path.join(results_dir, f"prediction_intervals_{experiment['output_suffix']}.csv")
    if not os.path.exists(output_file):
        raise FileNotFoundError(f"Expected output file not found: {output_file}")
    
    df = pd.read_csv(output_file)
    df['Experiment'] = experiment['description']
    return df

all_results = []

try:
    for exp in experiments:
        df = run_experiment(exp)
        # Reorder/Rename columns to match Table 1 in the paper prompt/latex
        # Expected Table 15 columns: Experiment, Method, NCIW, Quantile Loss, Average Width ($), Coverage
        
        # The CSV has: Method, Coverage, Mean_Width, Quantile_Loss, NCIW, NIW, CRPS, AUC
        
        # We need to reshape/process a bit
        # Select relevant columns
        df_subset = df[['Method', 'NCIW', 'Quantile_Loss', 'Mean_Width', 'Coverage', 'Experiment']]
        
        # Rename columns to match desired output
        df_subset = df_subset.rename(columns={
            'Quantile_Loss': 'Quantile Loss',
            'Mean_Width': 'Average Width ($)'
        })
        
        # Order columns
        df_subset = df_subset[['Experiment', 'Method', 'NCIW', 'Quantile Loss', 'Average Width ($)', 'Coverage']]
        
        all_results.append(df_subset)

    final_df = pd.concat(all_results, ignore_index=True)

    # Format for Latex
    # We want to group by Experiment essentially, or just list them
    
    print("\n\nGenerated Results DataFrame:")
    print(final_df)
    
    print("\n\nLaTeX Table code (approximation):")
    
    latex_rows = []
    
    # Group by Experiment to handle the multirow looking structure
    # But since we are just appending, we can iterate
    
    # We want to preserve the order of experiments list
    
    unique_experiments = [exp['description'] for exp in experiments]
    
    print("\\begin{table}[!htbp]")
    print("\\centering")
    print("\\caption{Ames Housing results for all four scenarios (90\\% coverage target).}")
    print("\\setlength{\\tabcolsep}{4pt}")
    print("\\small")
    print("\\begin{tabular}{llcccc}")
    print("\\toprule")
    print("Experiment & Method & NCIW & Quantile Loss & Average Width (\\$) & Coverage \\\\")
    print("\\midrule")
    
    for exp_desc in unique_experiments:
        # Get rows for this experiment
        exp_rows = final_df[final_df['Experiment'] == exp_desc]
        
        # Define specific method order if needed (PCS-PPI, CQR, CLEAR)
        method_order = ['PCS-PPI', 'CQR', 'CLEAR']
        
        first = True
        for method in method_order:
            row = exp_rows[exp_rows['Method'] == method].iloc[0]
            
            # Format values
            nciw = f"{row['NCIW']:.3f}"
            ql = f"{row['Quantile Loss']:,.0f}".replace(",", "{,}") # Latex comma formatting
            width = f"{row['Average Width ($)']:,.0f}".replace(",", "{,}")
            cov = f"{row['Coverage']:.2f}"
            
            # Highlight best values (min NCIW, min QL, min Width)
            
            # Experiment column
            if first:
                # Use multirow if I were using the true latex package, but for simple print:
                # We can just put the name in the first row, or use \multirow
                exp_col = f"\\multirow{{3}}{{*}}{{{exp_desc}}}" 
                first = False
            else:
                exp_col = ""
                
            print(f"{exp_col} & {method} & {nciw} & {ql} & {width} & {cov} \\\\")
        
        print("\\midrule")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

except Exception as e:
    print(f"An error occurred: {e}")