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
from src.PCS.classification.multi_class_pcs import MultiClassPCS
from src.PCS.classification.multi_class_pcs_oob import MultiClassPCS_OOB

# Conformal prediction imports
from src.conformal_methods.regression.split_conformal import SplitConformal
from src.conformal_methods.regression.studentized_conformal import StudentizedConformal
from src.conformal_methods.regression.local_conformal import LocalConformalRegressor

# Metrics imports
from src.metrics.classification_metrics import get_all_metrics, get_all_class_metrics


# Experiment configs
from experiments.configs.classification_configs import get_classification_datasets, get_conformal_methods, get_pcs_methods
from experiments.configs.classification_consts import VALID_UQ_METHODS, VALID_ESTIMATORS, MODELS, DATASETS, SINGLE_CONFORMAL_METHODS


def run_classification_experiments(
    dataset_name,
    seed,
    uq_method,
    uq_method_name,
    method_name,
    results_dir="experiments/results/classification",
    max_samples=5000, 
    train_size=0.8
):
   
    X_df, y, bin_df, importance = get_classification_datasets(dataset_name)
    X = X_df.to_numpy()
    #X,y, bin_df, X_df = X[:max_samples], y[:max_samples], bin_df[:max_samples], X_df[:max_samples]

    # Create results directory structure
    results_path = Path(results_dir)
    dataset_path = results_path / dataset_name
    #seed_path = dataset_path / str(seed)
    print(f'data created\n', flush=True)
    # Create directories if they don't exist
    dataset_path.mkdir(parents=True, exist_ok=True)
    metrics_file = f'{dataset_path}/{method_name}_seed_{seed}_train_size_{train_size}_metrics.pkl'

    if os.path.exists(metrics_file):
        print(f"Metrics file {metrics_file} already exists. Skipping experiment.\n", flush=True)
        return

    X_train, X_test, y_train, y_test, bin_df_train, bin_df_test, X_df_train, X_df_test = train_test_split(X, y, bin_df, X_df, train_size=train_size, random_state=seed, stratify=y)
    X_train, y_train, bin_df_train, X_df_train = X_train[:max_samples], y_train[:max_samples], bin_df_train[:max_samples], X_df_train[:max_samples]

    print(f"Fitting {method_name} on {dataset_name} with seed {seed}\n", flush=True)
    uq_method.fit(X_train, y_train)
    y_pred = uq_method.predict(X_test)
   
    full_metrics = get_all_metrics(y_test, y_pred, empty_set='to_full')
    full_class_metrics = get_all_class_metrics(y_test, y_pred, empty_set='to_full')
    metrics = get_all_metrics(y_test, y_pred)
    class_metrics = get_all_class_metrics(y_test, y_pred)
    print(f"{method_name} metrics: {metrics}\n", flush=True)
    print(f"{method_name} full metrics: {full_metrics}\n", flush=True)
    print(f"{method_name} class metrics: {class_metrics}\n", flush=True)
    print(f"{method_name} full class metrics: {full_class_metrics}\n", flush=True)


    full_metrics_file = f'{dataset_path}/{method_name}_seed_{seed}_train_size_{train_size}_full_metrics.pkl'
    with open(full_metrics_file, 'wb') as f:
        pickle.dump(full_metrics, f)
    full_class_metrics_file = f'{dataset_path}/{method_name}_seed_{seed}_train_size_{train_size}_full_class_metrics.pkl'
    with open(full_class_metrics_file, 'wb') as f:
        pickle.dump(full_class_metrics, f)
    metrics_file = f'{dataset_path}/{method_name}_seed_{seed}_train_size_{train_size}_metrics.pkl'
    with open(metrics_file, 'wb') as f:
        pickle.dump(metrics, f)
    class_metrics_file = f'{dataset_path}/{method_name}_seed_{seed}_train_size_{train_size}_class_metrics.pkl'
    with open(class_metrics_file, 'wb') as f:
        pickle.dump(class_metrics, f)

if __name__ == "__main__":
    # Example methods dictionary
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="data_chess", help="Name of dataset to run experiments on")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--UQ_method", type=str, default="pcs_uq", help="UQ method to use")
    parser.add_argument("--estimator", type=str, default="ExtraTrees", help="Estimator to use")
    parser.add_argument("--train_size", type=float, default=0.8, help="Train size")
    args = parser.parse_args()

    # Validate UQ method argument
    print(f"Running {args.UQ_method} on {args.dataset} with seed {args.seed} and train size {args.train_size}\n", flush=True)

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
    
    else:
        raise ValueError(f"Invalid UQ method '{args.UQ_method}'. Must be one of: {VALID_UQ_METHODS}")
    
    # elif args.UQ_method == "pcs_uq_model_prop":
    #     uq_method  = get_pcs_methods("pcs_uq_model_prop", args.seed)
    #     method_name = "pcs_uq_model_prop"
    
    # elif args.UQ_method == "pcs_oob_model_prop":
    #     uq_method  = get_pcs_methods("pcs_oob_model_prop", args.seed)
    # Set random seed
    np.random.seed(args.seed)
    print(f'starting experiment\n', flush=True)
    run_classification_experiments(dataset_name=args.dataset, seed=args.seed, uq_method=uq_method, uq_method_name = args.UQ_method, method_name=method_name, train_size=args.train_size)
    