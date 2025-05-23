# Standard imports
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from datetime import datetime
import logging
import argparse
import os

# Sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

# PCS imports
from src.PCS.regression.pcs_uq import PCS_UQ
from src.PCS.regression.pcs_oob import PCS_OOB

# Conformal prediction imports
from src.conformal_methods.regression.split_conformal import SplitConformal
from src.conformal_methods.regression.studentized_conformal import StudentizedConformal
from src.conformal_methods.regression.local_conformal import LocalConformalRegressor

# Metrics imports
from src.metrics.regression_metrics import get_all_metrics, evaluate_majority_vote

# Experiment configs
from experiments.configs.regression_configs import get_regression_datasets, get_conformal_methods, get_pcs_methods
from experiments.configs.regression_consts import VALID_UQ_METHODS, VALID_ESTIMATORS, MODELS, DATASETS, SINGLE_CONFORMAL_METHODS


def get_subgroup_metrics(X_test_df, y_test, y_pred, bin_df_test, importance, method_name):
    all_subgroup_metrics = {}
    for imp_var in importance['feature']:
        subgroup_indicator = bin_df_test[imp_var]
        # Add subgroup indicator to X_test_df
        X_test_df_subgroup = X_test_df.copy()
        X_test_df_subgroup[f'subgroup_{imp_var}'] = subgroup_indicator
        X_test_df_subgroup[f'y_test'] = y_test
        X_test_df_subgroup[f'y_pred_lb'] = y_pred[:,0]
        X_test_df_subgroup[f'y_pred_ub'] = y_pred[:,1]
        
        # Calculate metrics for each subgroup
        subgroup_metrics = {}
        for subgroup in X_test_df_subgroup[f'subgroup_{imp_var}'].unique():
            subgroup_df = X_test_df_subgroup[X_test_df_subgroup[f'subgroup_{imp_var}'] == subgroup]
            subgroup_y_test = subgroup_df['y_test'].values
            subgroup_y_pred = np.column_stack((subgroup_df['y_pred_lb'].values, 
                                             subgroup_df['y_pred_ub'].values))
            if method_name == "majority_vote":
                subgroup_metrics[subgroup] = evaluate_majority_vote(subgroup_y_test, subgroup_y_pred)
            else:
                subgroup_metrics[subgroup] = get_all_metrics(subgroup_y_test, subgroup_y_pred)
            
        all_subgroup_metrics[imp_var] = subgroup_metrics
    return all_subgroup_metrics

def run_regression_experiments(
    dataset_name,
    seed,
    uq_method,
    uq_method_name,
    method_name,
    results_dir="experiments/results/regression",
    max_samples=5000, 
    train_size=0.8
):
    X_df, y, bin_df, importance = get_regression_datasets(dataset_name)
    X = X_df.to_numpy()
    #X,y, bin_df, X_df = X[:max_samples], y[:max_samples], bin_df[:max_samples], X_df[:max_samples]

    # Create results directory structure
    results_path = Path(results_dir)
    dataset_path = results_path / dataset_name
    #seed_path = dataset_path / str(seed)
    
    # Create directories if they don't exist
    dataset_path.mkdir(parents=True, exist_ok=True)
    metrics_file = f'{dataset_path}/{method_name}_seed_{seed}_train_size_{train_size}_metrics.pkl'

    if os.path.exists(metrics_file):
        print(f"Metrics file {metrics_file} already exists. Skipping experiment.\n", flush=True)
        return

    X_train, X_test, y_train, y_test, bin_df_train, bin_df_test, X_df_train, X_df_test = train_test_split(X, y, bin_df, X_df, train_size=train_size, random_state=seed)
    X_train, y_train, bin_df_train, X_df_train = X_train[:max_samples], y_train[:max_samples], bin_df_train[:max_samples], X_df_train[:max_samples]

    print(f"Fitting {method_name} on {dataset_name} with seed {seed}\n", flush=True)
    uq_method.fit(X_train, y_train)
    y_pred = uq_method.predict(X_test)
    if method_name == "majority_vote":
        metrics = evaluate_majority_vote(y_test, y_pred)
    else:
        metrics = get_all_metrics(y_test, y_pred)
    print(f"{method_name}: {metrics}\n", flush=True)
    # Save metrics as pickle file

    #print(f"Saving metrics to {seed_path / f'{method_name}_metrics.pkl'}\n", flush=True)
    #metrics_file = seed_path / f"{method_name}_metrics.pkl"
    metrics_file = f'{dataset_path}/{method_name}_seed_{seed}_train_size_{train_size}_metrics.pkl'
    with open(metrics_file, 'wb') as f:
        pickle.dump(metrics, f)

    print("Finished fitting and saving metrics\n", flush=True)

    print("Calculating subgroup metrics\n", flush=True)
    # Calculate subgroup metrics
    all_subgroup_metrics = get_subgroup_metrics(X_df_test, y_test, y_pred, bin_df_test, importance, uq_method_name)
    print("Finished calculating subgroup metrics\n", flush=True)
    print(all_subgroup_metrics)
    # Save subgroup metrics
    #subgroup_metrics_file = seed_path / f"{method_name}_subgroup_metrics.pkl"
    subgroup_metrics_file = f'{dataset_path}/{method_name}_seed_{seed}_train_size_{train_size}_subgroup_metrics.pkl'
    with open(subgroup_metrics_file, 'wb') as f:
        pickle.dump(all_subgroup_metrics, f)

def agg_results(dataset_name=None, results_dir="experiments/results/regression", train_size=0.8):
    """
    Aggregate results across all seeds for a given dataset and method.
    
    Parameters:
    -----------
    dataset_name : str, optional
        Name of the dataset to aggregate results for. If None, aggregates for all datasets.
    results_dir : str, default="experiments/results/regression"
        Directory where results are stored.
    train_size : float, default=0.8
        Train size used in the experiments.
    """
    import pickle
    import numpy as np
    from pathlib import Path
    
    results_path = Path(results_dir)
    
    # If dataset_name is None, process all datasets
    if dataset_name is None:
        datasets = [d.name for d in results_path.iterdir() if d.is_dir()]
    else:
        datasets = [dataset_name]
    
    for dataset in datasets:
        dataset_path = results_path / dataset
        if not dataset_path.exists():
            print(f"Dataset path {dataset_path} does not exist. Skipping.")
            continue
        
        # Find all unique methods by looking at the metrics files
        all_files = list(dataset_path.glob(f"*_seed_*_train_size_{train_size}_metrics.pkl"))
        methods = set()
        for file in all_files:
            # Extract method name from filename
            filename = file.name
            method_name = filename.split("_seed_")[0]
            methods.add(method_name)
        
        # For each method, aggregate results across seeds
        for method in methods:
            print(f"Aggregating results for {method} on {dataset}")
            
            # Find all seed files for this method
            method_files = list(dataset_path.glob(f"{method}_seed_*_train_size_{train_size}_metrics.pkl"))
            
            if not method_files:
                print(f"No files found for method {method}. Skipping.")
                continue
            
            # Load all metrics
            all_metrics = []
            seeds = []
            for file in method_files:
                # Extract seed from filename
                filename = file.name
                seed = int(filename.split("_seed_")[1].split("_train_size_")[0])
                seeds.append(seed)
                
                with open(file, 'rb') as f:
                    metrics = pickle.load(f)
                    all_metrics.append(metrics)
            
            # Calculate mean and std for each metric
            agg_metrics = {}
            for key in all_metrics[0].keys():
                values = [m[key] for m in all_metrics]
                agg_metrics[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'values': values,
                    'seeds': seeds
                }
            
            # Save aggregated metrics
            agg_file = f'{dataset_path}/{method}_train_size_{train_size}_agg_metrics.pkl'
            with open(agg_file, 'wb') as f:
                pickle.dump(agg_metrics, f)
            
            print(f"Saved aggregated metrics to {agg_file}")

# Example usage:
if __name__ == "__main__":
    # Example methods dictionary
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="data_parkinsons", help="Name of dataset to run experiments on")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--UQ_method", type=str, default="split_conformal", help="UQ method to use")
    parser.add_argument("--estimator", type=str, default="XGBoost", help="Estimator to use")
    parser.add_argument("--train_size", type=float, default=0.8, help="Train size")
    args = parser.parse_args()

    # Validate UQ method argument
   

    if args.UQ_method not in VALID_UQ_METHODS:
        raise ValueError(f"Invalid UQ method '{args.UQ_method}'. Must be one of: {VALID_UQ_METHODS}")

    if args.estimator not in VALID_ESTIMATORS:
        raise ValueError(f"Invalid estimator '{args.estimator}'. Must be one of: {VALID_ESTIMATORS}")
    
    if args.UQ_method in SINGLE_CONFORMAL_METHODS:
        uq_method, method_name  = get_conformal_methods(args.UQ_method, args.estimator, args.seed)
    
    elif args.UQ_method == "majority_vote":
        uq_method, method_name = get_conformal_methods("majority_vote", args.estimator, args.seed)
        method_name = f"majority_vote"
    
    elif args.UQ_method == "pcs_uq":
        uq_method  = get_pcs_methods("pcs_uq", args.seed)
        method_name = "pcs_uq"
    
    elif args.UQ_method == "pcs_oob":
        uq_method  = get_pcs_methods("pcs_oob", args.seed)
        method_name = "pcs_oob"
    
    # Set random seed
    np.random.seed(args.seed)

    run_regression_experiments(dataset_name=args.dataset, seed=args.seed, uq_method=uq_method, uq_method_name = args.UQ_method, method_name=method_name, train_size=args.train_size)
    agg_results()