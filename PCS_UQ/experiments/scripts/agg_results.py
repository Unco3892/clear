# Standard imports
import pandas as pd
import numpy as np
import argparse
import os
import pickle
import json
from pathlib import Path

# Experiment configs
from experiments.configs.regression_consts import VALID_UQ_METHODS, VALID_ESTIMATORS, MODELS, DATASETS, SINGLE_CONFORMAL_METHODS
SEEDS = [777, 778, 779, 780, 781, 782, 783, 784, 785, 786]


def agg_results_dataset_method(results_dir, dataset_name = 'data_parkinsons', uq_method = 'split_conformal', estimator = 'XGBoost', train_size = 0.8):
    results = []
    for seed in SEEDS:
        try: 
            seed_results_dir = Path(f"{results_dir}/{dataset_name}/{uq_method}{estimator}seed_{seed}_train_size_{train_size}_metrics.pkl")
            results.append(pickle.load(open(seed_results_dir, "rb")))
        except: 
            print(f"Error in loading results for seed {seed}")
            pass
    # Convert list of dictionaries to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate mean and standard error for each metric
    agg_results = {}
    for column in results_df.columns:
        values = results_df[column].values
        agg_results[f"{column}_mean"] = np.mean(values)
        agg_results[f"{column}_std"] = np.std(values) 
    return agg_results

def average_across_dicts(dicts):
    results_df = pd.DataFrame(dicts)
    agg_results = {}
    for column in results_df.columns:
        values = results_df[column].values
        agg_results[f"{column}_mean"] = np.mean(values)
        agg_results[f"{column}_std"] = np.std(values) 
    return agg_results

def agg_subgroup_results_dataset_method(results_dir, dataset_name = 'data_parkinsons', uq_method = 'split_conformal', estimator = 'XGBoost', train_size = 0.8):
    results = []
    for seed in SEEDS:
        try: 
            seed_results_dir = Path(f"{results_dir}/{dataset_name}/{uq_method}{estimator}seed_{seed}_train_size_{train_size}_subgroup_metrics.pkl")
            results.append(pickle.load(open(seed_results_dir, "rb")))
        except: 
            print(f"Error in loading results for seed {seed} for dataset {dataset_name} and uq method {uq_method} and estimator {estimator} and train size {train_size}")
            pass
    # Convert list of dictionaries to DataFrame
    # Initialize a dictionary to store aggregated results
    agg_results = {}
    if len(results) == 0:
        return {}
    features = list(results[0].keys())
    
    # For each outer key, aggregate the inner dictionary values
    for feature in features:
        # Initialize a dictionary to collect all values for each inner key
        feature_results = [result[feature] for result in results]
        subgroups = list(feature_results[0].keys())
        agg_results[feature] = {}
        for subgroup in subgroups:
            try: 
                subgroup_results = [result[subgroup] for result in feature_results]
                agg_results[feature][subgroup] = average_across_dicts(subgroup_results)
            except: 
                print(f"Error in averaging results for feature {feature} and subgroup {subgroup}")
                print(subgroup)
                pass
        
        # Store the aggregated inner dictionary under the original outer key
        #agg_results[subgroup] = average_across_dicts(subgroup_results)
    
    return agg_results

def get_all_uq_methods():
    method_list = []
    for uq_method in VALID_UQ_METHODS:
        for estimator in VALID_ESTIMATORS:
            if uq_method not in SINGLE_CONFORMAL_METHODS:
                method_list.append((f'{uq_method}_', ''))
            elif uq_method == 'LocalConformalRegressor':
                pass
            else:
                method_list.append((f'{uq_method}_', f'{estimator}_'))
    method_list = list(set(method_list))
    return method_list


def main(task = "regression", train_size = 0.8):
    results_dir = Path(f"experiments/results/{task}/")
    aggregated_results_dir = Path(f'{results_dir}/aggregated_results')
    os.makedirs(aggregated_results_dir, exist_ok=True)
    uq_methods = get_all_uq_methods()
    all_results = {}
    for dataset in DATASETS:
        results_dataset = {}
        for uq_method, estimator in uq_methods:
            results = agg_results_dataset_method(results_dir = results_dir, dataset_name = dataset, uq_method = uq_method, estimator = estimator, train_size = train_size)
            # Remove trailing underscores from method and estimator names
            results_dataset[(uq_method.rstrip('_'), estimator.rstrip('_'))] = results
        # Save dataset-specific results
        dataset_results_file = aggregated_results_dir / f"{dataset}_train_size_{train_size}_results.pkl"
        with open(dataset_results_file, 'wb') as f:
            pickle.dump(results_dataset, f)
        all_results[dataset] = results_dataset
    # Save all results
    all_results_file = aggregated_results_dir / f"all_results_train_size_{train_size}.pkl"
    with open(all_results_file, 'wb') as f:
        pickle.dump(all_results, f)
    return all_results
    #print(results)

def agg_subgroup_results(task = "regression", train_size = 0.8):
    results_dir = Path(f"experiments/results/{task}/")
    aggregated_results_dir = Path(f'{results_dir}/aggregated_results')
    os.makedirs(aggregated_results_dir, exist_ok=True)
    uq_methods = get_all_uq_methods()
    all_results = {}
    for dataset in DATASETS:
        results_dataset = {}
        for uq_method, estimator in uq_methods:
            results = agg_subgroup_results_dataset_method(results_dir = results_dir, dataset_name = dataset, uq_method = uq_method, estimator = estimator, train_size = train_size)
            results_dataset[(uq_method.rstrip('_'), estimator.rstrip('_'))] = results
        # Save dataset-specific results
        dataset_results_file = aggregated_results_dir / f"{dataset}_train_size_{train_size}_subgroup_results.pkl"
        with open(dataset_results_file, 'wb') as f:
            pickle.dump(results_dataset, f)
        all_results[dataset] = results_dataset
    all_results_file = aggregated_results_dir / f"all_subgroup_results_train_size_{train_size}.pkl"
    with open(all_results_file, 'wb') as f:
        pickle.dump(all_results, f)
    return all_results
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="regression", help="Task to run")
    parser.add_argument("--train_size", type=float, default=0.8, help="Train size")
    args = parser.parse_args()
   # main(args.task, args.train_size)
    agg_subgroup_results(args.task, args.train_size)
    