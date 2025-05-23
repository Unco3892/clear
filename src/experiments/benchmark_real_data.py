#!/usr/bin/env python
import os
import sys
import pickle
import numpy as np
import pandas as pd
import argparse
import traceback
import platform
import logging
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
import matplotlib
import time
# Import utilty functions for the experiments
from utils import reconstruct_dataframe, safe_flatten, get_top_model_info, setup_logging, StreamToLogger, format_metric_name
# Import our scripts
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)
sys.path.append(os.path.join(script_dir, '..'))
from clear.metrics import evaluate_intervals
# Import the CLEAR implementation
from clear.clear import CLEAR
# Import utility functions
from clear.utils import load_ensemble_pickle

def compute_cqr_intervals(y_calib, aleatoric_median, aleatoric_lower, aleatoric_upper, 
                          pcs_median, pcs_median_test, aleatoric_median_test, aleatoric_lower_test, 
                          aleatoric_upper_test, coverage):
    """
    Compute Conformalized Quantile Regression (CQR) intervals.
    
    Args:
        y_calib: Calibration targets
        aleatoric_median/lower/upper: Aleatoric predictions for calibration data
        pcs_median: PCS median for calibration data
        pcs_median_test: PCS median for test data
        aleatoric_median/lower/upper_test: Aleatoric predictions for test data
        coverage: Target coverage probability
        
    Returns:
        (cqr_lower, cqr_upper, adjustment)
    """
    logger = logging.getLogger()
    logger.info("Computing CQR intervals...")
    
    # Center aleatoric bounds around PCS median for calibration
    calib_lower_std = pcs_median - (aleatoric_median - aleatoric_lower)
    calib_upper_std = pcs_median + (aleatoric_upper - aleatoric_median)
    
    # Calculate non-conformity scores
    scores_std = np.maximum(
        calib_lower_std - y_calib,
        y_calib - calib_upper_std
    )
    
    # Apply conformal calibration with correction term
    n_calib = len(y_calib)
    alpha = 1 - coverage
    q_level_std = (1 - alpha) * (1 + (1 / n_calib))
    q_level_std = min(q_level_std, 1.0)
    adjustment_std = np.quantile(scores_std, q_level_std, method='higher')
    
    # Generate CQR intervals using PCS median as the center
    cqr_lower = (pcs_median_test - (aleatoric_median_test - aleatoric_lower_test)) - adjustment_std
    cqr_upper = (pcs_median_test + (aleatoric_upper_test - aleatoric_median_test)) + adjustment_std
    
    return cqr_lower, cqr_upper, adjustment_std

def evaluate_and_log_metrics(all_metrics, y_test, method_name, lower_bounds, upper_bounds, alpha, evaluation_median, 
                             method_params=None):
    """
    Evaluate prediction intervals and log metrics.
    
    Args:
        all_metrics: Dictionary to update with metrics
        y_test: Test targets
        method_name: Name of the method
        lower_bounds: Lower bounds of prediction intervals
        upper_bounds: Upper bounds of prediction intervals
        alpha: Significance level (1 - coverage)
        evaluation_median: Median predictions for evaluation
        method_params: Optional parameters to store (e.g., lambda, gamma)
    
    Returns:
        Dictionary of computed metrics
    """
    metrics = evaluate_intervals(y_test, lower_bounds, upper_bounds, alpha=alpha, f=evaluation_median)
    
    # Store standard metrics
    standard_metrics = ["PICP", "NIW", "MPIW", "QuantileLoss", "ExpectileLoss", "CRPS", "AUC", "NCIW", "IntervalScoreLoss"]
    for metric in standard_metrics:
        metric_key = metric.lower()
        if metric_key not in all_metrics[method_name]:
            all_metrics[method_name][metric_key] = []
        all_metrics[method_name][metric_key].append(metrics[metric])
    
    # Store additional parameters if provided
    if method_params:
        for param_name, param_value in method_params.items():
            if param_name not in all_metrics[method_name]:
                all_metrics[method_name][param_name] = []
            all_metrics[method_name][param_name].append(param_value)
    
    return metrics

def create_metrics_dict(include_lambda=False, include_gamma=False, include_uncertainty_ratio=True):
    """Create a metrics dictionary with common metrics and optional lambda/gamma."""
    metrics = {
        "picp": [], "niw": [], "mpiw": [], "quantile_loss": [], "expectile_loss": [], "crps": [], "auc": [], "nciw": [], "interval_score_loss": []
    }
    if include_lambda:
        metrics["lambda"] = []
    if include_gamma:
        metrics["gamma"] = []
    # Add uncertainty ratio metrics
    if include_uncertainty_ratio:
        metrics["total_aleatoric_calib"] = []
        metrics["total_epistemic_calib"] = []
        metrics["uncertainty_ratio_calib"] = []
    return metrics

def generate_tables(all_metrics, dataset, coverage, results_dir, tex_dir):
    """Generate CSV tables for the given dataset and metrics"""
    # Import csv module which is needed for this function
    import csv
    logger = logging.getLogger()
    
    # Create directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)

    # Convert coverage to a string without the decimal point
    coverage_str = str(int(coverage * 100))

    # Save individual seed results to CSV
    csv_path = os.path.join(results_dir, f"benchmark_results_{dataset}_{coverage_str}.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Dataset", "Method", "Metric", "Seed", "Value"])
        for method, metrics in all_metrics.items():
            for metric, values in metrics.items():
                for seed_idx, value in enumerate(values):
                    writer.writerow([dataset, method, metric, seed_idx, value])
    logger.info(f"CSV table saved to {os.path.abspath(csv_path)}")
    
    # LaTeX table generation has been removed as requested

def process_approach(X_train_df, y_train_df, X_valid_df, y_valid, X_test_df, y_test,
                  pcs_median_val, pcs_lower_val, pcs_upper_val,
                  pcs_median_test, pcs_lower_test, pcs_upper_test,
                  run_data, args, is_residual=False, train_pcs_median=None, 
                  pcs_lower_test_calib = None, pcs_upper_test_calib =None):
    """
    Process either the residual or non-residual approach to generate prediction intervals.
    
    Args:
        X_train/valid/test_df: Feature dataframes
        y_train/valid/test: Target values
        pcs_median/lower/upper_val: PCS bounds for validation data
        pcs_median/lower/upper_test: PCS bounds for test data
        run_data: Dictionary containing run data
        args: Command-line arguments
        is_residual: Whether to use the residual approach
        train_pcs_median: Training set PCS median (required for residual approach)
        
    Returns:
        Dictionary containing all computed intervals and models
    """
    logger = logging.getLogger()
    random_state = 777
    result = {}
    
    # Set approach name for logging
    approach = "residual" if is_residual else "standard"
    logger.info(f"Processing {approach} approach...")
    
    # Remove this check since we now have better handling in process_dataset
    # if (args.approach == "standard" and is_residual) or (args.approach == "residual" and not is_residual):
    #     logger.info(f"Skipping {approach} approach based on --approach setting")
    #     return {
    #         'skipped': True,
    #         'reason': 'approach_setting'
    #     }
    
    # Check if we have what we need for residual approach
    if is_residual and train_pcs_median is None:
        logger.warning("Training set PCS median required for residual approach but not provided.")
        return {
            'skipped': True,
            'reason': 'missing_train_pcs_median'
        }
        
    # Store residual flag in result for use during evaluation
    result['is_residual'] = is_residual
    
    # Define lambdas to search over - same for all models
    lambdas = np.concatenate((np.linspace(0, 0.09, 10), np.logspace(-1, 2, 4001)))

    # Get top model information from the PCS run data - always use this
    model_type, model_params, quantile_models = get_top_model_info(run_data)
    # model_params = None
    # quantile_models = "xgb"
    
    # Create, fit, and calibrate CLEAR model
    clear_model = create_clear_model(
        coverage=args.coverage,
        lambdas=lambdas,
        n_bootstraps=args.n_bootstraps,
        random_state=random_state,
        n_jobs=args.n_jobs
    )
    
    # Fit aleatoric model - with or without residuals
    fit_aleatoric_model(
        clear_model=clear_model,
        X_train_df=X_train_df,
        y_train_df=y_train_df,
        quantile_models=quantile_models,
        model_params=model_params,
        fit_on_residuals=is_residual,
        epistemic_preds=train_pcs_median if is_residual else None
    )
    
    # Generate aleatoric predictions
    aleatoric_median_val, aleatoric_lower_val, aleatoric_upper_val = clear_model.predict_aleatoric(
        X_valid_df,
        epistemic_preds=pcs_median_val if is_residual else None
    )
    
    aleatoric_median_test, aleatoric_lower_test, aleatoric_upper_test = clear_model.predict_aleatoric(
        X_test_df,
        epistemic_preds=pcs_median_test if is_residual else None
    )
        
    # Compute CQR intervals
    cqr_lower, cqr_upper, adjustment = compute_cqr_intervals(
        y_calib=y_valid,
        aleatoric_median=aleatoric_median_val,
        aleatoric_lower=aleatoric_lower_val,
        aleatoric_upper=aleatoric_upper_val,
        pcs_median=pcs_median_val,
        pcs_median_test=pcs_median_test,
        aleatoric_median_test=aleatoric_median_test,
        aleatoric_lower_test=aleatoric_lower_test,
        aleatoric_upper_test=aleatoric_upper_test,
        coverage=args.coverage
    )
    
    # Save CQR results
    result['cqr_lower'] = cqr_lower
    result['cqr_upper'] = cqr_upper
    result['cqr_adjustment'] = adjustment
    
    # Calibrate standard CLEAR model
    calibrate_clear_model(
        clear_model=clear_model,
        y_valid=y_valid,
        median_epistemic=pcs_median_val,
        aleatoric_median=aleatoric_median_val,
        aleatoric_lower=aleatoric_lower_val,
        aleatoric_upper=aleatoric_upper_val,
        epistemic_lower=pcs_lower_val,
        epistemic_upper=pcs_upper_val
    )
    
    # Generate CLEAR predictions
    clear_lower, clear_upper = predict_with_clear(
        clear_model=clear_model,
        X_test_df=X_test_df,
        pcs_median_test=pcs_median_test,
        pcs_lower_test=pcs_lower_test,
        pcs_upper_test=pcs_upper_test,
        aleatoric_median_test=aleatoric_median_test, # Use the computed aleatoric median for non-conformalized CLEAR
        aleatoric_lower_test=aleatoric_lower_test,
        aleatoric_upper_test=aleatoric_upper_test
    )
    
    # Save CLEAR results
    result['clear_model'] = clear_model
    result['clear_lower'] = clear_lower
    result['clear_upper'] = clear_upper
    result['clear_optimal_lambda'] = clear_model.optimal_lambda
    result['clear_gamma'] = clear_model.gamma
    # Add uncertainty metrics
    result['clear_total_aleatoric_calib'] = clear_model.total_aleatoric_calib
    result['clear_total_epistemic_calib'] = clear_model.total_epistemic_calib
    result['clear_uncertainty_ratio_calib'] = clear_model.uncertainty_ratio_calib
    
    # Create, calibrate and predict with CLEAR-c model (conformalized)
    clear_c_model = create_clear_model(
        coverage=args.coverage,
        lambdas=lambdas,
        n_bootstraps=args.n_bootstraps,
        random_state=random_state,
        n_jobs=args.n_jobs
    )
    
    # Calibrate CLEAR-c model with conformalized bounds
    calibrate_clear_model(
        clear_model=clear_c_model,
        y_valid=y_valid,
        median_epistemic=pcs_median_val,
        aleatoric_median=aleatoric_median_val,
        aleatoric_lower=aleatoric_lower_val,
        aleatoric_upper=aleatoric_upper_val,
        epistemic_lower=pcs_lower_val,
        epistemic_upper=pcs_upper_val,
        is_conformalized=True,
        adjustment=adjustment
    )
    
    # Generate CLEAR-c predictions
    clear_c_lower, clear_c_upper = predict_with_clear(
        clear_model=clear_c_model,
        X_test_df=X_test_df,
        pcs_median_test=pcs_median_test,
        pcs_lower_test=pcs_lower_test,
        pcs_upper_test=pcs_upper_test,
        aleatoric_median_test=pcs_median_test, # Use PCS median as the center for adjusted bounds
        aleatoric_lower_test=cqr_lower,       # Use conformalized bounds
        aleatoric_upper_test=cqr_upper        # Use conformalized bounds
    )
    
    # Save CLEAR-c results
    result['clear_c_model'] = clear_c_model
    result['clear_c_lower'] = clear_c_lower
    result['clear_c_upper'] = clear_c_upper
    result['clear_c_optimal_lambda'] = clear_c_model.optimal_lambda
    result['clear_c_gamma'] = clear_c_model.gamma
    # Add uncertainty metrics
    result['clear_c_total_aleatoric_calib'] = clear_c_model.total_aleatoric_calib
    result['clear_c_total_epistemic_calib'] = clear_c_model.total_epistemic_calib
    result['clear_c_uncertainty_ratio_calib'] = clear_c_model.uncertainty_ratio_calib
    
    # Calculate MEAN(PCS+CQR) baseline
    mean_pcs_cqr_lower = (pcs_lower_test_calib + cqr_lower) / 2.0
    mean_pcs_cqr_upper = (pcs_upper_test_calib + cqr_upper) / 2.0
    
    # Save MEAN results
    result['mean_pcs_cqr_lower'] = mean_pcs_cqr_lower
    result['mean_pcs_cqr_upper'] = mean_pcs_cqr_upper
    
    # Implement Gamma=1 (formerly UACQR_S) method
    gamma_1_model = create_clear_model(
        coverage=args.coverage,
        lambdas=lambdas,
        n_bootstraps=args.n_bootstraps,
        random_state=random_state,
        n_jobs=args.n_jobs,
        fixed_gamma=1.0  # Fix gamma=1
    )
    
    # Calibration for Gamma=1 model
    logger.info(f"Calibrating Gamma=1 model (gamma=1, {approach} approach)")
    gamma_1_model.calibrate(
        y_calib=y_valid.flatten(),
        median_epistemic=pcs_median_val,
        aleatoric_median=aleatoric_median_val,
        aleatoric_lower=aleatoric_lower_val,
        aleatoric_upper=aleatoric_upper_val,
        epistemic_lower=pcs_lower_val,
        epistemic_upper=pcs_upper_val
    )
    logger.info(f"Gamma=1 optimal lambda: {gamma_1_model.optimal_lambda:.6f}")
    
    # Generate Gamma=1 predictions
    gamma_1_lower, gamma_1_upper = predict_with_clear(
        clear_model=gamma_1_model,
        X_test_df=X_test_df,
        pcs_median_test=pcs_median_test,
        pcs_lower_test=pcs_lower_test,
        pcs_upper_test=pcs_upper_test,
        aleatoric_median_test=aleatoric_median_test,
        aleatoric_lower_test=aleatoric_lower_test,
        aleatoric_upper_test=aleatoric_upper_test
    )
    
    # Save Gamma=1 results
    result['gamma_1_model'] = gamma_1_model
    result['gamma_1_lower'] = gamma_1_lower
    result['gamma_1_upper'] = gamma_1_upper
    result['gamma_1_lambda'] = gamma_1_model.optimal_lambda
    
    # Implement Gamma=1-c (conformalized Gamma=1) method
    gamma_1_c_model = create_clear_model(
        coverage=args.coverage,
        lambdas=lambdas,
        n_bootstraps=args.n_bootstraps,
        random_state=random_state,
        n_jobs=args.n_jobs,
        fixed_gamma=1.0  # Fix gamma=1
    )
    
    logger.info(f"Calibrating Gamma=1-c model (gamma=1, {approach} approach, conformalized)")
    gamma_1_c_model.calibrate(
        y_calib=y_valid.flatten(),
        median_epistemic=pcs_median_val,
        aleatoric_median=pcs_median_val,  # Use PCS median as center
        aleatoric_lower=pcs_median_val - (aleatoric_median_val - aleatoric_lower_val) - adjustment,  # Apply conformation
        aleatoric_upper=pcs_median_val + (aleatoric_upper_val - aleatoric_median_val) + adjustment,  # Apply conformation
        epistemic_lower=pcs_lower_val,
        epistemic_upper=pcs_upper_val
    )
    logger.info(f"Gamma=1-c optimal lambda: {gamma_1_c_model.optimal_lambda:.6f}")
    
    # Generate Gamma=1-c predictions
    gamma_1_c_lower, gamma_1_c_upper = predict_with_clear(
        clear_model=gamma_1_c_model,
        X_test_df=X_test_df,
        pcs_median_test=pcs_median_test,
        pcs_lower_test=pcs_lower_test,
        pcs_upper_test=pcs_upper_test,
        aleatoric_median_test=pcs_median_test, # Use PCS median as center
        aleatoric_lower_test=cqr_lower,        # Use conformalized bounds directly
        aleatoric_upper_test=cqr_upper         # Use conformalized bounds directly
    )
    
    # Save Gamma=1-c results
    result['gamma_1_c_model'] = gamma_1_c_model
    result['gamma_1_c_lower'] = gamma_1_c_lower
    result['gamma_1_c_upper'] = gamma_1_c_upper
    result['gamma_1_c_lambda'] = gamma_1_c_model.optimal_lambda
    
    # Implement fixed lambda=1 method
    lambda_one_model = create_clear_model(
        coverage=args.coverage,
        lambdas=lambdas,
        n_bootstraps=args.n_bootstraps,
        random_state=random_state,
        n_jobs=args.n_jobs,
        fixed_lambda=1.0  # Fix lambda=1
    )
    
    # Improved calibration for Lambda=1 model - direct calibration
    logger.info(f"Calibrating Lambda=1 model ({approach} approach)")
    
    # Direct calibration to match demo_consistent.py implementation
    lambda_one_model.calibrate(
        y_calib=y_valid.flatten(),
        median_epistemic=pcs_median_val,
        aleatoric_median=aleatoric_median_val,
        aleatoric_lower=aleatoric_lower_val,
        aleatoric_upper=aleatoric_upper_val,
        epistemic_lower=pcs_lower_val,
        epistemic_upper=pcs_upper_val
    )
    
    logger.info(f"Lambda=1 gamma: {lambda_one_model.gamma:.6f}")
    
    # Generate lambda=1 predictions
    lambda_one_lower, lambda_one_upper = predict_with_clear(
        clear_model=lambda_one_model,
        X_test_df=X_test_df,
        pcs_median_test=pcs_median_test,
        pcs_lower_test=pcs_lower_test,
        pcs_upper_test=pcs_upper_test,
        aleatoric_median_test=aleatoric_median_test,
        aleatoric_lower_test=aleatoric_lower_test,
        aleatoric_upper_test=aleatoric_upper_test
    )
    
    # Save lambda=1 results
    result['lambda_one_model'] = lambda_one_model
    result['lambda_one_lower'] = lambda_one_lower
    result['lambda_one_upper'] = lambda_one_upper
    result['lambda_one_gamma'] = lambda_one_model.gamma
    
    # Implement fixed lambda=1 with conformalized CQR (lambda=1-c)
    lambda_one_c_model = create_clear_model(
        coverage=args.coverage,
        lambdas=lambdas,
        n_bootstraps=args.n_bootstraps,
        random_state=random_state,
        n_jobs=args.n_jobs,
        fixed_lambda=1.0  # Fix lambda=1
    )
    
    # Improved calibration for Lambda=1-c model - direct calibration with conformalization
    logger.info(f"Calibrating Lambda=1-c model ({approach} approach, conformalized)")
    
    # Direct calibration with conformalized bounds
    lambda_one_c_model.calibrate(
        y_calib=y_valid.flatten(),
        median_epistemic=pcs_median_val,
        aleatoric_median=pcs_median_val,
        aleatoric_lower=pcs_median_val - (aleatoric_median_val - aleatoric_lower_val) - adjustment,
        aleatoric_upper=pcs_median_val + (aleatoric_upper_val - aleatoric_median_val) + adjustment,
        epistemic_lower=pcs_lower_val,
        epistemic_upper=pcs_upper_val
    )
    
    logger.info(f"Lambda=1-c gamma: {lambda_one_c_model.gamma:.6f}")
    
    # Generate lambda=1-c predictions
    lambda_one_c_lower, lambda_one_c_upper = predict_with_clear(
        clear_model=lambda_one_c_model,
        X_test_df=X_test_df,
        pcs_median_test=pcs_median_test,
        pcs_lower_test=pcs_lower_test,
        pcs_upper_test=pcs_upper_test,
        aleatoric_median_test=pcs_median_test,
        aleatoric_lower_test=cqr_lower,
        aleatoric_upper_test=cqr_upper
    )
    
    # Save lambda=1-c results
    result['lambda_one_c_model'] = lambda_one_c_model
    result['lambda_one_c_lower'] = lambda_one_c_lower
    result['lambda_one_c_upper'] = lambda_one_c_upper
    result['lambda_one_c_gamma'] = lambda_one_c_model.gamma
    
    # Save aleatoric predictions
    result['aleatoric_median_val'] = aleatoric_median_val
    result['aleatoric_lower_val'] = aleatoric_lower_val 
    result['aleatoric_upper_val'] = aleatoric_upper_val
    result['aleatoric_median_test'] = aleatoric_median_test
    result['aleatoric_lower_test'] = aleatoric_lower_test
    result['aleatoric_upper_test'] = aleatoric_upper_test
    
    # Save final median to use for evaluation
    result['evaluation_median'] = pcs_median_test
    
    return result

def process_dataset(dataset, args, run_nums=None):
    """Process a single dataset with specified parameters"""
    logger = logging.getLogger()

    # Determine dataset key and results directory based on PCS quantile model option
    dataset_key = dataset if dataset.startswith("data_") else f"data_{dataset}"
    # Base PCS results directory (point predictors by default)
    results_base = args.models_dir
    # Path to ensemble pickle file
    ensemble_file = Path(results_base) / f"{dataset_key}_pcs_results_{int(args.coverage*100)}.pkl"
    if not os.path.exists(ensemble_file):
        logger.error(f"Error: Ensemble file {ensemble_file} not found.")
        return None
    
    logger.info(f"Loading ensemble results from {ensemble_file}")
    ensemble_dict = load_ensemble_pickle(ensemble_file)
    
    # Results storage
    all_metrics = {
        "clear_vanilla": create_metrics_dict(include_lambda=True, include_gamma=True), # Was "clear"
        "clear_vanilla_c": create_metrics_dict(include_lambda=True, include_gamma=True), # Was "clear_c"
        "clear": create_metrics_dict(include_lambda=True, include_gamma=True),         # Was "clear_residual"
        "clear_c": create_metrics_dict(include_lambda=True, include_gamma=True),     # Was "clear_residual_c"
        "cqr": create_metrics_dict(),
        "cqr_residual": create_metrics_dict(),
        "pcs": create_metrics_dict(),
        "a_naive": create_metrics_dict(),
        "s_naive": create_metrics_dict(),
        "mean_pcs_cqr": create_metrics_dict(),
        "mean_pcs_cqr_residual": create_metrics_dict(),
        "gamma_1": create_metrics_dict(include_lambda=True),
        "gamma_1_r": create_metrics_dict(include_lambda=True),
        "gamma_1_c": create_metrics_dict(include_lambda=True),
        "gamma_1_c_r": create_metrics_dict(include_lambda=True),
        "lambda_one": create_metrics_dict(include_gamma=True),
        "lambda_one_r": create_metrics_dict(include_gamma=True),
        "lambda_one_c": create_metrics_dict(include_gamma=True),
        "lambda_one_c_r": create_metrics_dict(include_gamma=True)
    }
    
    # Determine runs to process
    if run_nums is not None:
        runs_to_process = [f"run_{r}" for r in run_nums]
    else:
        runs_to_process = ensemble_dict.keys()
    
    # Filter to existing runs only
    runs_to_process = [run for run in runs_to_process if run in ensemble_dict]
    
    if not runs_to_process:
        logger.warning(f"No valid runs found to process for dataset {dataset}.")
        return None
    
    logger.info(f"Processing {len(runs_to_process)} runs for dataset {dataset}: {', '.join(runs_to_process)}")
    
    # Process each run
    for run_key in runs_to_process:
        start_time = time.time()
        if run_key not in ensemble_dict:
            logger.warning(f"Run {run_key} not found in ensemble pickle. Available runs: {list(ensemble_dict.keys())}")
            continue
            
        logger.info(f"\nProcessing {run_key}...")
        run_data = ensemble_dict[run_key]
        
        # Extract train, validation and test data
        X_train = run_data.get("x_train", None)
        y_train = run_data.get("y_train", None)
        X_valid = run_data.get("x_val", None)
        y_valid = run_data.get("y_val", None)
        X_test = run_data.get("x_test", None)
        y_test = run_data.get("y_test", None)
        
        X_train = np.array(X_train) if isinstance(X_train, list) else X_train
        y_train = np.array(y_train) if isinstance(y_train, list) else y_train
        X_valid = np.array(X_valid) if isinstance(X_valid, list) else X_valid
        y_valid = np.array(y_valid) if isinstance(y_valid, list) else y_valid
        X_test = np.array(X_test) if isinstance(X_test, list) else X_test
        y_test = np.array(y_test) if isinstance(y_test, list) else y_test

        y_test = safe_flatten(y_test)
        y_valid = safe_flatten(y_valid)
        y_train = safe_flatten(y_train)

        # Convert to DataFrames for consistent API
        X_train_df = pd.DataFrame(X_train)
        y_train_df = pd.DataFrame(y_train, columns=['y'])
        X_valid_df = pd.DataFrame(X_valid)
        y_valid_df = pd.DataFrame(y_valid, columns=['y'])
        X_test_df = pd.DataFrame(X_test)
        y_test_df = pd.DataFrame(y_test, columns=['y'])
        alpha = 1 - args.coverage  # Target miscoverage rate

        # print(run_data.keys())
                
        # Get validation intervals if available
        # pcs_intervals_val =np.array(run_data['val_intervals'])
        pcs_intervals_val = np.array(run_data['val_intervals_raw'])
        pcs_lower_val = pcs_intervals_val [:, 0]
        # pcs_median_val = np.array(run_data['val_median_preds'])
        pcs_median_val = pcs_intervals_val[:, 1]
        pcs_upper_val = pcs_intervals_val[:, 2]

        # Extract the raw intervals
        pcs_intervals_test = np.array(run_data['test_intervals_raw'])
        pcs_lower_test = pcs_intervals_test[:, 0]
        pcs_median_test = pcs_intervals_test[:, 1]
        pcs_upper_test = pcs_intervals_test[:, 2]

        # Extract the calibrated intervals
        pcs_intervals_test_calib = np.asarray(run_data.get("test_intervals"))
        pcs_lower_test_calib = pcs_intervals_test_calib[:, 0]
        pcs_upper_test_calib = pcs_intervals_test_calib[:, 1]

        #  Get gamma value from pickle file
        logger.info(f"Using gamma value from pickle file: {run_data['gamma']:.4f}")
        gamma_value = run_data['gamma']

        # Check for median predictions format from train_pcs_quantile.py
        # Create a 2D array with the predictions as a single model
        # if isinstance(test_predictions, np.ndarray) and test_predictions.ndim == 1:
        #     test_predictions = np.array([test_predictions]).T

        # # Create a 2D array with the predictions as a single model
        # if isinstance(val_intervals, np.ndarray) and val_intervals.ndim == 1:
        #     val_intervals = np.array([val_intervals]).T
        
        # Process ensemble predictions to get PCS bounds
        logger.info("Computing PCS bounds from ensemble predictions...")

        # Validation set PCS bounds
        if isinstance(pcs_intervals_val, pd.DataFrame):
            val_ensemble_preds = pcs_intervals_val.values.T
        else:
            val_ensemble_preds = np.array(pcs_intervals_val).T
                
        # Test set PCS bounds - Properly handle different formats of test_predictions
        if isinstance(pcs_intervals_test, pd.DataFrame):
            test_ensemble_preds = pcs_intervals_test.values.T  # Shape: (n_models, n_samples)
        else:
            test_ensemble_preds = np.array(pcs_intervals_test).T  # Convert to numpy array first
        
        logger.info(f"Test ensemble predictions shape: {test_ensemble_preds.shape}")
        
        # Get train ensemble predictions for residual approach if needed
        train_ensemble_preds = None
        train_pcs_median = None
        if args.approach in ["residual", "both"]:
            pcs_intervals_train = np.array(run_data.get("train_intervals_raw"))
            if pcs_intervals_train is not None:
                if isinstance(pcs_intervals_train, pd.DataFrame):
                    train_ensemble_preds = pcs_intervals_train.values.T
                else:
                    train_ensemble_preds = np.array(pcs_intervals_train).T
                
                train_pcs_median = train_ensemble_preds[1, :]
                train_pcs_lower = train_ensemble_preds[0, :]
                train_pcs_upper = train_ensemble_preds[2, :]
                print(f"Shape of train_ensemble_preds after: {train_ensemble_preds.shape}")
            else:
                logger.warning("No training predictions found for residual approach.")
               
        try:
            # When approach is "both", process both standard and residual approaches
            if args.approach == "both":
                # First process standard approach
                standard_results = process_approach(
                    X_train_df=X_train_df, 
                    y_train_df=y_train_df,
                    X_valid_df=X_valid_df,
                    y_valid=y_valid,
                    X_test_df=X_test_df,
                    y_test=y_test,
                    pcs_median_val=pcs_median_val,
                    pcs_lower_val=pcs_lower_val,
                    pcs_upper_val=pcs_upper_val,
                    pcs_median_test=pcs_median_test,
                    pcs_lower_test=pcs_lower_test,
                    pcs_upper_test=pcs_upper_test,
                    run_data=run_data,
                    args=args,
                    is_residual=False,  # Standard approach
                    train_pcs_median=train_pcs_median,
                    pcs_lower_test_calib=pcs_lower_test_calib,
                    pcs_upper_test_calib=pcs_upper_test_calib
                )
                
                # Then process residual approach (if we have train_pcs_median)
                residual_results = None
                if train_pcs_median is not None:
                    residual_results = process_approach(
                        X_train_df=X_train_df, 
                        y_train_df=y_train_df,
                        X_valid_df=X_valid_df,
                        y_valid=y_valid,
                        X_test_df=X_test_df,
                        y_test=y_test,
                        pcs_median_val=pcs_median_val,
                        pcs_lower_val=pcs_lower_val,
                        pcs_upper_val=pcs_upper_val,
                        pcs_median_test=pcs_median_test,
                        pcs_lower_test=pcs_lower_test,
                        pcs_upper_test=pcs_upper_test,
                        run_data=run_data,
                        args=args,
                        is_residual=True,  # Residual approach
                        train_pcs_median=train_pcs_median,
                        pcs_lower_test_calib=pcs_lower_test_calib,
                        pcs_upper_test_calib=pcs_upper_test_calib
                    )
                
                # Process results from both approaches
                results_list = [r for r in [standard_results, residual_results] if r is not None]
                
                # Compute naive baselines once
                calib_predictions = pcs_median_val  # Use PCS median for calibration
                test_predictions = pcs_median_test  # Use PCS median for test
                a_naive_lower, a_naive_upper, s_naive_lower, s_naive_upper = compute_naive_intervals(
                    y_calib=y_valid,
                    calib_predictions=calib_predictions,
                    test_predictions=test_predictions,
                    coverage=args.coverage
                )
                
                # Evaluate PCS and naive baselines once
                if pcs_lower_test_calib is not None and pcs_upper_test_calib is not None:
                    evaluate_and_log_metrics(
                        all_metrics=all_metrics, y_test=y_test, method_name="pcs",
                        lower_bounds=pcs_lower_test_calib, upper_bounds=pcs_upper_test_calib,
                        alpha=alpha, evaluation_median=pcs_median_test,
                        method_params={"gamma": gamma_value} if gamma_value is not None else None
                    )
                
                evaluate_and_log_metrics(
                    all_metrics=all_metrics, y_test=y_test, method_name="a_naive",
                    lower_bounds=a_naive_lower, upper_bounds=a_naive_upper,
                    alpha=alpha, evaluation_median=pcs_median_test
                )
                
                evaluate_and_log_metrics(
                    all_metrics=all_metrics, y_test=y_test, method_name="s_naive",
                    lower_bounds=s_naive_lower, upper_bounds=s_naive_upper,
                    alpha=alpha, evaluation_median=pcs_median_test
                )
                
                # Process each set of results
                alpha = 1 - args.coverage
                for results in results_list:
                    # Skip if the approach was skipped
                    if results.get('skipped', False):
                        logger.info(f"Skipping evaluation for {run_key} because process_approach was skipped (Reason: {results.get('reason')}).")
                        continue
                        
                    # Determine evaluation median
                    evaluation_median = results.get('evaluation_median', pcs_median_test)
                    
                    # Set method prefixes based on whether this is residual or standard
                    is_res = results.get('is_residual', False)
                    # method_prefix = "clear_residual" if is_res else "clear"
                    # method_prefix_c = "clear_residual_c" if is_res else "clear_c"
                    if is_res: # Residual approach results
                        method_prefix = "clear" # Formerly "clear_residual"
                        method_prefix_c = "clear_c" # Formerly "clear_residual_c"
                    else: # Standard approach results
                        method_prefix = "clear_vanilla" # Formerly "clear"
                        method_prefix_c = "clear_vanilla_c" # Formerly "clear_c"

                    cqr_prefix = "cqr_residual" if is_res else "cqr"
                    mean_prefix = "mean_pcs_cqr_residual" if is_res else "mean_pcs_cqr"
                    gamma_1_prefix = "gamma_1_r" if is_res else "gamma_1"
                    gamma_1_c_prefix = "gamma_1_c_r" if is_res else "gamma_1_c"
                    lambda_one_prefix = "lambda_one_r" if is_res else "lambda_one"
                    lambda_one_c_prefix = "lambda_one_c_r" if is_res else "lambda_one_c"
                    
                    # Evaluate CLEAR
                    if results.get('clear_lower') is not None:
                        method_params = {
                            "lambda": results.get('clear_optimal_lambda'), 
                            "gamma": results.get('clear_gamma'),
                            "total_aleatoric_calib": results.get('clear_total_aleatoric_calib'),
                            "total_epistemic_calib": results.get('clear_total_epistemic_calib'),
                            "uncertainty_ratio_calib": results.get('clear_uncertainty_ratio_calib')
                        }
                        evaluate_and_log_metrics(
                            all_metrics=all_metrics, y_test=y_test, method_name=method_prefix,
                            lower_bounds=results['clear_lower'], upper_bounds=results['clear_upper'],
                            alpha=alpha, evaluation_median=evaluation_median,
                            method_params=method_params
                        )
                        
                    # Evaluate CLEAR-c
                    if results.get('clear_c_lower') is not None:
                        method_params = {
                            "lambda": results.get('clear_c_optimal_lambda'), 
                            "gamma": results.get('clear_c_gamma'),
                            "total_aleatoric_calib": results.get('clear_c_total_aleatoric_calib'),
                            "total_epistemic_calib": results.get('clear_c_total_epistemic_calib'),
                            "uncertainty_ratio_calib": results.get('clear_c_uncertainty_ratio_calib')
                        }
                        evaluate_and_log_metrics(
                            all_metrics=all_metrics, y_test=y_test, method_name=method_prefix_c,
                            lower_bounds=results['clear_c_lower'], upper_bounds=results['clear_c_upper'],
                            alpha=alpha, evaluation_median=evaluation_median,
                            method_params=method_params
                        )
                        
                    # Evaluate CQR
                    if results.get('cqr_lower') is not None:
                        evaluate_and_log_metrics(
                            all_metrics=all_metrics, y_test=y_test, method_name=cqr_prefix,
                            lower_bounds=results['cqr_lower'], upper_bounds=results['cqr_upper'],
                            alpha=alpha, evaluation_median=evaluation_median
                        )

                    # Evaluate Mean(PCS+CQR)
                    if results.get('mean_pcs_cqr_lower') is not None:
                        evaluate_and_log_metrics(
                            all_metrics=all_metrics, y_test=y_test, method_name=mean_prefix,
                            lower_bounds=results['mean_pcs_cqr_lower'], upper_bounds=results['mean_pcs_cqr_upper'],
                            alpha=alpha, evaluation_median=evaluation_median
                        )

                    # Evaluate Gamma=1
                    if results.get('gamma_1_lower') is not None:
                        evaluate_and_log_metrics(
                            all_metrics=all_metrics, y_test=y_test, method_name=gamma_1_prefix,
                            lower_bounds=results['gamma_1_lower'], upper_bounds=results['gamma_1_upper'],
                            alpha=alpha, evaluation_median=evaluation_median,
                            method_params={"lambda": results.get('gamma_1_lambda')}
                        )
                    
                    # Evaluate Gamma=1-c
                    if results.get('gamma_1_c_lower') is not None:
                        evaluate_and_log_metrics(
                            all_metrics=all_metrics, y_test=y_test, method_name=gamma_1_c_prefix,
                            lower_bounds=results['gamma_1_c_lower'], upper_bounds=results['gamma_1_c_upper'],
                            alpha=alpha, evaluation_median=evaluation_median,
                            method_params={"lambda": results.get('gamma_1_c_lambda')}
                        )
                    
                    # Evaluate Lambda=1
                    if results.get('lambda_one_lower') is not None:
                        evaluate_and_log_metrics(
                            all_metrics=all_metrics, y_test=y_test, method_name=lambda_one_prefix,
                            lower_bounds=results['lambda_one_lower'], upper_bounds=results['lambda_one_upper'],
                            alpha=alpha, evaluation_median=evaluation_median,
                            method_params={"gamma": results.get('lambda_one_gamma')}
                        )
                    
                    # Evaluate Lambda=1-c
                    if results.get('lambda_one_c_lower') is not None:
                        evaluate_and_log_metrics(
                            all_metrics=all_metrics, y_test=y_test, method_name=lambda_one_c_prefix,
                            lower_bounds=results['lambda_one_c_lower'], upper_bounds=results['lambda_one_c_upper'],
                            alpha=alpha, evaluation_median=evaluation_median,
                            method_params={"gamma": results.get('lambda_one_c_gamma')}
                        )
                
                # Print metrics for this run
                logger.info(f"\n{'-'*40}")
                logger.info(f"Metrics for {run_key}:")
                logger.info(f"{'-'*40}")
                
                # Define a mapping of method keys to display names
                display_name_map = {
                    "clear_vanilla": "CLEAR-Vanilla",       # Was "clear": "CLEAR"
                    "clear_vanilla_c": "CLEAR-Vanilla-c",   # Was "clear_c": "CLEAR-c"
                    "clear": "CLEAR",                 # Was "clear_residual": "CLEAR-R"
                    "clear_c": "CLEAR-c",             # Was "clear_residual_c": "CLEAR-R-c"
                    "cqr": "CQR",
                    "cqr_residual": "CQR-R",
                    "pcs": "PCS",
                    "a_naive": "A-Naive",
                    "s_naive": "S-Naive",
                    "mean_pcs_cqr": "Mean(PCS+CQR)",
                    "mean_pcs_cqr_residual": "Mean(PCS+CQR)-R",
                    "gamma_1": "Gamma=1",
                    "gamma_1_r": "Gamma=1-R",
                    "gamma_1_c": "Gamma=1-c",
                    "gamma_1_c_r": "Gamma=1-c-R",
                    "lambda_one": "Lambda=1",
                    "lambda_one_r": "Lambda=1-R",
                    "lambda_one_c": "Lambda=1-c",
                    "lambda_one_c_r": "Lambda=1-c-R"
                }
                
                # Log metrics for this run organized by method
                current_run_index = runs_to_process.index(run_key)
                for method_name, metrics in all_metrics.items():
                    if method_name in display_name_map:
                        # Check if this method has results for the current run
                        has_results = False
                        for metric_values in metrics.values():
                            if len(metric_values) > current_run_index:
                                has_results = True
                                break
                        
                        if has_results:
                            logger.info(f"\n  {display_name_map[method_name]}:")
                            for metric_name, values in metrics.items():
                                if len(values) > current_run_index:
                                    value = values[current_run_index]
                                    logger.info(f"    {format_metric_name(metric_name)}: {value:.4f}")
                
                # Skip the rest of the processing since we've already done it
                continue
            
            # Process only standard or residual approach based on args.approach
            results = process_approach(
                X_train_df=X_train_df, 
                y_train_df=y_train_df,
                X_valid_df=X_valid_df,
                y_valid=y_valid,
                X_test_df=X_test_df,
                y_test=y_test,
                pcs_median_val=pcs_median_val,
                pcs_lower_val=pcs_lower_val,
                pcs_upper_val=pcs_upper_val,
                pcs_median_test=pcs_median_test,
                pcs_lower_test=pcs_lower_test, # Raw PCS lower for CLEAR input
                pcs_upper_test=pcs_upper_test, # Raw PCS upper for CLEAR input
                run_data=run_data,
                args=args,
                is_residual=(args.approach == "residual"),
                train_pcs_median=train_pcs_median if train_pcs_median is not None else None,
                pcs_lower_test_calib=pcs_lower_test_calib, # Pass calibrated for Mean(PCS+CQR)
                pcs_upper_test_calib=pcs_upper_test_calib  # Pass calibrated for Mean(PCS+CQR)
            )

            # Check if the approach was skipped (e.g., due to --residual_only)
            if results.get('skipped', False):
                logger.info(f"Skipping evaluation for {run_key} because process_approach was skipped (Reason: {results.get('reason')}).")
                continue # Skip to the next run

            # Compute naive baselines using median of raw PCS predictions
            calib_predictions = pcs_median_val # Use PCS median for calibration
            test_predictions = pcs_median_test # Use PCS median for test
            a_naive_lower, a_naive_upper, s_naive_lower, s_naive_upper = compute_naive_intervals(
                y_calib=y_valid,
                calib_predictions=calib_predictions,
                test_predictions=test_predictions,
                coverage=args.coverage
            )
            
            # Determine evaluation median
            evaluation_median = results.get('evaluation_median', pcs_median_test) # Use PCS median as fallback
            alpha = 1 - args.coverage

            # --- Start Refactored Evaluation Block ---

            # Evaluate PCS (using calibrated intervals from run_data)
            if pcs_lower_test_calib is not None and pcs_upper_test_calib is not None:
                evaluate_and_log_metrics(
                    all_metrics=all_metrics, y_test=y_test, method_name="pcs",
                    lower_bounds=pcs_lower_test_calib, upper_bounds=pcs_upper_test_calib,
                    alpha=alpha, evaluation_median=evaluation_median,
                    method_params={"gamma": gamma_value} if gamma_value is not None else None # gamma_value from run_data
                )
            else:
                logger.warning(f"Skipping PCS evaluation for {run_key} due to missing calibrated intervals.")

            # Evaluate Naive Baselines
            evaluate_and_log_metrics(
                all_metrics=all_metrics, y_test=y_test, method_name="a_naive",
                lower_bounds=a_naive_lower, upper_bounds=a_naive_upper,
                alpha=alpha, evaluation_median=evaluation_median
            )
            evaluate_and_log_metrics(
                all_metrics=all_metrics, y_test=y_test, method_name="s_naive",
                lower_bounds=s_naive_lower, upper_bounds=s_naive_upper,
                alpha=alpha, evaluation_median=evaluation_median
            )
                        
            # Define prefixes for metric keys based on whether residuals were used
            # Note: The results dict contains the *output* of process_approach.
            # If args.use_residual=True, these results are residual-based.
            # method_prefix = "clear_residual" if args.approach == "residual" else "clear"
            # method_prefix_c = "clear_residual_c" if args.approach == "residual" else "clear_c"
            if args.approach == "residual":
                method_prefix = "clear" # Formerly "clear_residual"
                method_prefix_c = "clear_c" # Formerly "clear_residual_c"
            elif args.approach == "standard":
                method_prefix = "clear_vanilla" # Formerly "clear"
                method_prefix_c = "clear_vanilla_c" # Formerly "clear_c"
            else: # Should not happen if args.approach is 'both', handled above
                method_prefix = "clear_vanilla" 
                method_prefix_c = "clear_vanilla_c"

            cqr_prefix = "cqr_residual" if args.approach == "residual" else "cqr"
            mean_prefix = "mean_pcs_cqr_residual" if args.approach == "residual" else "mean_pcs_cqr"
            gamma_1_prefix = "gamma_1_r" if args.approach == "residual" else "gamma_1"
            gamma_1_c_prefix = "gamma_1_c_r" if args.approach == "residual" else "gamma_1_c"
            lambda_one_prefix = "lambda_one_r" if args.approach == "residual" else "lambda_one"
            lambda_one_c_prefix = "lambda_one_c_r" if args.approach == "residual" else "lambda_one_c"

            # Evaluate CLEAR (Standard or Residual)
            if results.get('clear_lower') is not None:
                method_params = {
                    "lambda": results.get('clear_optimal_lambda'), 
                    "gamma": results.get('clear_gamma'),
                    "total_aleatoric_calib": results.get('clear_total_aleatoric_calib'),
                    "total_epistemic_calib": results.get('clear_total_epistemic_calib'),
                    "uncertainty_ratio_calib": results.get('clear_uncertainty_ratio_calib')
                }
                evaluate_and_log_metrics(
                    all_metrics=all_metrics, y_test=y_test, method_name=method_prefix,
                    lower_bounds=results['clear_lower'], upper_bounds=results['clear_upper'],
                    alpha=alpha, evaluation_median=evaluation_median,
                    method_params=method_params
                )

            # Evaluate CLEAR-c (Standard or Residual)
            if results.get('clear_c_lower') is not None:
                method_params = {
                    "lambda": results.get('clear_c_optimal_lambda'), 
                    "gamma": results.get('clear_c_gamma'),
                    "total_aleatoric": results.get('clear_c_total_aleatoric'),
                    "total_epistemic": results.get('clear_c_total_epistemic'),
                    "uncertainty_ratio": results.get('clear_c_uncertainty_ratio')
                }
                evaluate_and_log_metrics(
                    all_metrics=all_metrics, y_test=y_test, method_name=method_prefix_c,
                    lower_bounds=results['clear_c_lower'], upper_bounds=results['clear_c_upper'],
                    alpha=alpha, evaluation_median=evaluation_median,
                    method_params=method_params
                )

            # Evaluate CQR (Standard or Residual)
            if results.get('cqr_lower') is not None:
                evaluate_and_log_metrics(
                    all_metrics=all_metrics, y_test=y_test, method_name=cqr_prefix,
                    lower_bounds=results['cqr_lower'], upper_bounds=results['cqr_upper'],
                    alpha=alpha, evaluation_median=evaluation_median
                )

            # Evaluate Mean(PCS+CQR) (Standard or Residual)
            if results.get('mean_pcs_cqr_lower') is not None:
                evaluate_and_log_metrics(
                    all_metrics=all_metrics, y_test=y_test, method_name=mean_prefix,
                    lower_bounds=results['mean_pcs_cqr_lower'], upper_bounds=results['mean_pcs_cqr_upper'],
                    alpha=alpha, evaluation_median=evaluation_median
                )

            # Evaluate Gamma=1
            if results.get('gamma_1_lower') is not None:
                evaluate_and_log_metrics(
                    all_metrics=all_metrics, y_test=y_test, method_name=gamma_1_prefix,
                    lower_bounds=results['gamma_1_lower'], upper_bounds=results['gamma_1_upper'],
                    alpha=alpha, evaluation_median=evaluation_median,
                    method_params={"lambda": results.get('gamma_1_lambda')}
                )

            # Evaluate Gamma=1-c
            if results.get('gamma_1_c_lower') is not None:
                evaluate_and_log_metrics(
                    all_metrics=all_metrics, y_test=y_test, method_name=gamma_1_c_prefix,
                    lower_bounds=results['gamma_1_c_lower'], upper_bounds=results['gamma_1_c_upper'],
                    alpha=alpha, evaluation_median=evaluation_median,
                    method_params={"lambda": results.get('gamma_1_c_lambda')}
                )

            # Evaluate Lambda=1
            if results.get('lambda_one_lower') is not None:
                evaluate_and_log_metrics(
                    all_metrics=all_metrics, y_test=y_test, method_name=lambda_one_prefix,
                    lower_bounds=results['lambda_one_lower'], upper_bounds=results['lambda_one_upper'],
                    alpha=alpha, evaluation_median=evaluation_median,
                    method_params={"gamma": results.get('lambda_one_gamma')} # Store gamma if available
                )
            
            # Evaluate Lambda=1-c
            if results.get('lambda_one_c_lower') is not None:
                evaluate_and_log_metrics(
                    all_metrics=all_metrics, y_test=y_test, method_name=lambda_one_c_prefix,
                    lower_bounds=results['lambda_one_c_lower'], upper_bounds=results['lambda_one_c_upper'],
                    alpha=alpha, evaluation_median=evaluation_median,
                    method_params={"gamma": results.get('lambda_one_c_gamma')} # Store gamma if available
                )
            
            # Print metrics for this run
            logger.info(f"\nMetrics for {run_key}:")
            # Use the display name map defined below in the summary section
            display_name_map_local = {
                "clear_vanilla": "CLEAR-Vanilla",       # Was "clear": "CLEAR"
                "clear_vanilla_c": "CLEAR-Vanilla-c",   # Was "clear_c": "CLEAR-c"
                "clear": "CLEAR",                 # Was "clear_residual": "CLEAR-R"
                "clear_c": "CLEAR-c",             # Was "clear_residual_c": "CLEAR-R-c"
                "cqr": "CQR", "cqr_residual": "CQR-R",
                "pcs": "PCS", "a_naive": "A-Naive", "s_naive": "S-Naive",
                "mean_pcs_cqr": "Mean(PCS+CQR)", "mean_pcs_cqr_residual": "Mean(PCS+CQR)-R",
                "gamma_1": "Gamma=1", "gamma_1_r": "Gamma=1-R", 
                "gamma_1_c": "Gamma=1-c", "gamma_1_c_r": "Gamma=1-c-R",
                "lambda_one": "Lambda=1", "lambda_one_r": "Lambda=1-R",
                "lambda_one_c": "Lambda=1-c", "lambda_one_c_r": "Lambda=1-c-R"
            }
            
            # Log metrics for methods that have results for the current run
            methods_in_run = [m for m, metrics in all_metrics.items() if any(v for v in metrics.values())]
            for method in methods_in_run:
                 # Check if the method has results *for this specific run*
                has_current_run_data = False
                metrics_dict_for_log = {}
                for metric_name, values in all_metrics[method].items():
                    if len(values) == runs_to_process.index(run_key) + 1: # Check if data for current run exists
                        has_current_run_data = True
                        value = values[-1] # Get the last added value (current run)
                        metrics_dict_for_log[metric_name] = value

                if has_current_run_data:
                    display_name = display_name_map_local.get(method, method.upper())
                    logger.info(f"  {display_name}:")
                    for metric_name, value in metrics_dict_for_log.items():
                         logger.info(f"    {format_metric_name(metric_name)}: {value:.4f}")

            
            # Save results for this dataset if requested
            if args.save_results:
                # Create dataset-specific filename
                result_filename = f"{dataset}_{run_key}.pkl"
                # Ensure save_results is treated as a directory path if it doesn't end with .pkl or similar
                save_dir = args.save_results
                if not os.path.isdir(save_dir):
                     # Try to infer directory if a file path was given
                     potential_dir = os.path.dirname(save_dir)
                     if os.path.isdir(potential_dir):
                         save_dir = potential_dir
                     else:
                         # Default to a 'run_results' subdirectory if inference fails
                         save_dir = os.path.join(os.getcwd(), "run_results")
                         logger.warning(f"Specified save_results path '{args.save_results}' is not a directory. Saving to '{save_dir}'.")
                result_path = os.path.join(save_dir, result_filename)

                try:
                    run_results_save = {}
                    run_results_save['dataset'] = dataset
                    run_results_save['run'] = run_key
                    run_results_save['X_test'] = X_test
                    run_results_save['y_test'] = y_test
                    run_results_save['evaluation_median'] = evaluation_median # Include the median used

                    # --- Revised saving logic for clarity with new naming ---
                    # These are the final keys we want in the output .pkl file's metric blocks
                    # and also the keys in all_metrics.
                    target_metric_keys_map = {
                        # For standard run results:
                        "clear_vanilla": "clear_vanilla",
                        "clear_vanilla_c": "clear_vanilla_c",
                        # For residual run results:
                        "clear": "clear", # This is the new "clear" (formerly clear_residual)
                        "clear_c": "clear_c", # This is the new "clear_c" (formerly clear_residual_c)
                        
                        # Other methods (these are processed once per run, their naming in all_metrics is already set)
                        "cqr": "cqr" if not results.get('is_residual', False) else "cqr_residual",
                        "mean_pcs_cqr": "mean_pcs_cqr" if not results.get('is_residual', False) else "mean_pcs_cqr_residual",
                        "gamma_1": "gamma_1" if not results.get('is_residual', False) else "gamma_1_r",
                        "gamma_1_c": "gamma_1_c" if not results.get('is_residual', False) else "gamma_1_c_r",
                        "lambda_one": "lambda_one" if not results.get('is_residual', False) else "lambda_one_r",
                        "lambda_one_c": "lambda_one_c" if not results.get('is_residual', False) else "lambda_one_c_r",
                        
                        # Baselines (processed once per run_key)
                        "pcs": "pcs", 
                        "a_naive": "a_naive", 
                        "s_naive": "s_naive"
                    }

                    # Keys within the 'results' dict from process_approach are generic
                    # e.g., 'clear_lower', 'gamma_1_lower'
                    # The 'is_residual' flag from 'results' tells us if these are from a standard or residual context.
                    
                    current_run_index = runs_to_process.index(run_key)
                    is_current_results_from_residual_run = results.get('is_residual', False)

                    # Loop through the methods we want to save based on the current 'results' context
                    methods_to_save_in_this_iteration = {}
                    if is_current_results_from_residual_run:
                        methods_to_save_in_this_iteration["clear"] = ("clear_lower", "clear_upper", "clear")
                        methods_to_save_in_this_iteration["clear_c"] = ("clear_c_lower", "clear_c_upper", "clear_c")
                        methods_to_save_in_this_iteration["cqr_residual"] = ("cqr_lower", "cqr_upper", "cqr_residual")
                        methods_to_save_in_this_iteration["mean_pcs_cqr_residual"] = ("mean_pcs_cqr_lower", "mean_pcs_cqr_upper", "mean_pcs_cqr_residual")
                        methods_to_save_in_this_iteration["gamma_1_r"] = ("gamma_1_lower", "gamma_1_upper", "gamma_1_r")
                        methods_to_save_in_this_iteration["gamma_1_c_r"] = ("gamma_1_c_lower", "gamma_1_c_upper", "gamma_1_c_r")
                        methods_to_save_in_this_iteration["lambda_one_r"] = ("lambda_one_lower", "lambda_one_upper", "lambda_one_r")
                        methods_to_save_in_this_iteration["lambda_one_c_r"] = ("lambda_one_c_lower", "lambda_one_c_upper", "lambda_one_c_r")
                    else: # Standard run
                        methods_to_save_in_this_iteration["clear_vanilla"] = ("clear_lower", "clear_upper", "clear_vanilla")
                        methods_to_save_in_this_iteration["clear_vanilla_c"] = ("clear_c_lower", "clear_c_upper", "clear_vanilla_c")
                        methods_to_save_in_this_iteration["cqr"] = ("cqr_lower", "cqr_upper", "cqr")
                        methods_to_save_in_this_iteration["mean_pcs_cqr"] = ("mean_pcs_cqr_lower", "mean_pcs_cqr_upper", "mean_pcs_cqr")
                        methods_to_save_in_this_iteration["gamma_1"] = ("gamma_1_lower", "gamma_1_upper", "gamma_1")
                        methods_to_save_in_this_iteration["gamma_1_c"] = ("gamma_1_c_lower", "gamma_1_c_upper", "gamma_1_c")
                        methods_to_save_in_this_iteration["lambda_one"] = ("lambda_one_lower", "lambda_one_upper", "lambda_one")
                        methods_to_save_in_this_iteration["lambda_one_c"] = ("lambda_one_c_lower", "lambda_one_c_upper", "lambda_one_c")

                    # Add baselines (PCS, Naive) to be saved only once, typically with non-residual/first pass
                    if not is_current_results_from_residual_run:
                        methods_to_save_in_this_iteration["pcs"] = (pcs_lower_test_calib, pcs_upper_test_calib, "pcs")
                        methods_to_save_in_this_iteration["a_naive"] = (a_naive_lower, a_naive_upper, "a_naive")
                        methods_to_save_in_this_iteration["s_naive"] = (s_naive_lower, s_naive_upper, "s_naive")
                    
                    for pkl_method_key, (res_lower_key, res_upper_key, metrics_key) in methods_to_save_in_this_iteration.items():
                        lower_val, upper_val = None, None
                        if pkl_method_key in ["pcs", "a_naive", "s_naive"]: # Already have values
                            lower_val, upper_val = res_lower_key, res_upper_key
                        else: # Fetch from 'results' dict
                            lower_val = results.get(res_lower_key)
                            upper_val = results.get(res_upper_key)

                        if lower_val is not None and upper_val is not None:
                            run_results_save[f'{pkl_method_key}_lower'] = lower_val
                            run_results_save[f'{pkl_method_key}_upper'] = upper_val

                            if metrics_key in all_metrics:
                                current_metrics_for_pkl = {}
                                for k, v_list in all_metrics[metrics_key].items():
                                    if len(v_list) > current_run_index:
                                        current_metrics_for_pkl[k] = v_list[current_run_index]
                                if current_metrics_for_pkl:
                                    run_results_save[f'{pkl_method_key}_metrics'] = current_metrics_for_pkl
                    # --- End of revised saving logic ---
                    
                    # Create directory if it doesn't exist
                    os.makedirs(os.path.dirname(result_path), exist_ok=True)
                    
                    # Save results to pickle file using binary write mode
                    with open(result_path, 'wb') as f: # Use 'wb' for pickle
                        pickle.dump(run_results_save, f)
                    
                    logger.info(f"Results saved to {os.path.abspath(result_path)}")
                
                except Exception as e:
                    logger.error(f"Error saving results for {run_key}: {e}")
                    logger.error(traceback.format_exc())
            
            
        except Exception as e:
            logger.error(f"Error processing {run_key}: {e}")
            logger.error(traceback.format_exc())
        
    
    # Print summary statistics for this dataset
    logger.info("\n" + "="*40)
    logger.info(f"Summary Statistics for {dataset} ({len(runs_to_process)} runs):")
    logger.info("="*40)
    
    # Define display names mapping (consistent with acronyms in generate_tables)
    method_display = {
        "clear_vanilla": "CLEAR-Vanilla",       # Was "clear": "CLEAR"
        "clear_vanilla_c": "CLEAR-Vanilla-c",   # Was "clear_c": "CLEAR-c"
        "clear": "CLEAR",                 # Was "clear_residual": "CLEAR-R"
        "clear_c": "CLEAR-c",             # Was "clear_residual_c": "CLEAR-R-c"
        "cqr": "CQR",
        "cqr_residual": "CQR-R",
        "pcs": "PCS",
        "a_naive": "A-Naive",
        "s_naive": "S-Naive",
        "mean_pcs_cqr": "Mean(PCS+CQR)",
        "mean_pcs_cqr_residual": "Mean(PCS+CQR)-R",
        "gamma_1": "Gamma=1",
        "gamma_1_r": "Gamma=1-R",
        "gamma_1_c": "Gamma=1-c",
        "gamma_1_c_r": "Gamma=1-c-R",
        "lambda_one": "Lambda=1",
        "lambda_one_r": "Lambda=1-R",
        "lambda_one_c": "Lambda=1-c",
        "lambda_one_c_r": "Lambda=1-c-R"
    }

    # Determine which methods have results accumulated across runs
    methods_with_results = [m for m, metrics in all_metrics.items() if any(v for v in metrics.values())]

    # Order methods for display
    ordered_methods = [
        "clear", "clear_c", "clear_vanilla", "clear_vanilla_c", 
        "cqr", "cqr_residual", "pcs", "a_naive", "s_naive",
        "mean_pcs_cqr", "mean_pcs_cqr_residual",
        "gamma_1", "gamma_1_r", "gamma_1_c", "gamma_1_c_r",
        "lambda_one", "lambda_one_r", "lambda_one_c", "lambda_one_c_r"
    ]
    methods_to_show = [m for m in ordered_methods if m in methods_with_results]

    # Make sure to show all metrics in a consistent order
    metric_order = ["picp", "niw", "mpiw", "quantile_loss", "expectile_loss", "crps", "interval_score_loss", "auc", "nciw", "lambda", "gamma"]
    
    for method_name in methods_to_show:
        # Skip if no display name defined (shouldn't happen with current setup)
        if method_name not in method_display: continue

        metrics = all_metrics[method_name]
        # Check if there are actually any values logged for this method
        if not any(len(v) > 0 for v in metrics.values()):
            continue
        
        logger.info(f"\n{method_display[method_name]} Metrics:")
        # Use metric_order to ensure consistent display order of metrics
        for metric_name in metric_order:
            values = metrics.get(metric_name, [])
            # values = metrics.get(metric_name)
            if len(values) > 0:  # Only display metrics that have values
            # if values: # Check if list is not empty
                avg_value = np.mean(values)
                std_value = np.std(values)
                logger.info(f"  {format_metric_name(metric_name)}: {avg_value:.4f} ± {std_value:.4f}")
                
                # If multiple runs, also print min and max values
                if len(values) > 1:
                    min_value = np.min(values)
                    max_value = np.max(values)
                    logger.info(f"    -> range: [{min_value:.4f}, {max_value:.4f}]")
    
    # Clean up large objects to release memory
    if 'ensemble_dict' in locals():
        del ensemble_dict
    import gc
    gc.collect()

    return all_metrics

def create_clear_model(coverage, lambdas, n_bootstraps, random_state, n_jobs, fixed_gamma=None, fixed_lambda=None):
    """
    Create a CLEAR model with the given parameters.
    
    Args:
        coverage: Target coverage probability
        lambdas: List of lambda values to try
        n_bootstraps: Number of bootstrap samples
        random_state: Random seed
        n_jobs: Number of parallel jobs
        fixed_gamma: Optional fixed gamma value
        fixed_lambda: Optional fixed lambda value
        
    Returns:
        Initialized CLEAR model
    """
    return CLEAR(
        desired_coverage=coverage,
        lambdas=lambdas,
        n_bootstraps=n_bootstraps,
        random_state=random_state,
        n_jobs=n_jobs,
        fixed_gamma=fixed_gamma,
        fixed_lambda=fixed_lambda
    )

def fit_aleatoric_model(clear_model, X_train_df, y_train_df, quantile_models, model_params=None, fit_on_residuals=False, epistemic_preds=None):
    """
    Fit aleatoric uncertainty model.
    
    Args:
        clear_model: CLEAR model instance
        X_train_df: Training features
        y_train_df: Training targets
        quantile_models: List of quantile models to fit
        model_params: Optional model parameters
        fit_on_residuals: Whether to fit on residuals
        epistemic_preds: Required for residual fitting
    """
    logger = logging.getLogger()
    
    if fit_on_residuals:
        if epistemic_preds is None:
            raise ValueError("epistemic_preds must be provided when fit_on_residuals=True")
        
        logger.info(f"Fitting aleatoric model on residuals with model(s): {quantile_models}")
        if model_params:
            clear_model.fit_aleatoric(
                X_train_df, y_train_df,
                quantile_model=quantile_models,
                model_params=model_params,
                fit_on_residuals=True,
                epistemic_preds=epistemic_preds
            )
        else:
            clear_model.fit_aleatoric(
                X_train_df, y_train_df,
                quantile_model=quantile_models,
                fit_on_residuals=True,
                epistemic_preds=epistemic_preds
            )
    else:
        logger.info(f"Fitting standard aleatoric model with model(s): {quantile_models}")
        if model_params:
            clear_model.fit_aleatoric(
                X_train_df, y_train_df,
                quantile_model=quantile_models,
                model_params=model_params
            )
        else:
            clear_model.fit_aleatoric(
                X_train_df, y_train_df,
                quantile_model=quantile_models
            )

def calibrate_clear_model(clear_model, y_valid, median_epistemic, aleatoric_median, 
                          aleatoric_lower, aleatoric_upper, epistemic_lower, epistemic_upper,
                          is_conformalized=False, adjustment=None):
    """
    Calibrate a CLEAR model.
    
    Args:
        clear_model: CLEAR model instance
        y_valid: Validation targets
        median_epistemic: Epistemic median predictions
        aleatoric_median: Aleatoric median predictions
        aleatoric_lower: Aleatoric lower predictions
        aleatoric_upper: Aleatoric upper predictions
        epistemic_lower: Epistemic lower predictions
        epistemic_upper: Epistemic upper predictions
        is_conformalized: Whether to use conformalized bounds
        adjustment: Conformal adjustment (required if is_conformalized=True)
    """
    logger = logging.getLogger()
    
    if is_conformalized:
        if adjustment is None:
            raise ValueError("adjustment must be provided when is_conformalized=True")
        print("-"*40)
        logger.info("Calibrating with conformalized bounds...")
        # Calculate the PCS-median-centered aleatoric widths/bounds for calibration
        calib_aleatoric_left = aleatoric_median - aleatoric_lower
        calib_aleatoric_right = aleatoric_upper - aleatoric_median

        # Apply adjustment to these centered bounds
        # The bounds are now centered around the epistemic median
        adj_lower = median_epistemic - calib_aleatoric_left - adjustment
        adj_upper = median_epistemic + calib_aleatoric_right + adjustment

        # Calibrate using the correctly centered and adjusted bounds.
        # Use median_epistemic as the aleatoric_median for calibration centering.
        clear_model.calibrate(
            y_valid.flatten(),
            median_epistemic=median_epistemic,
            aleatoric_median=median_epistemic, # Use PCS median as center
            aleatoric_lower=adj_lower,         # Pass adjusted, PCS-centered lower bound
            aleatoric_upper=adj_upper,         # Pass adjusted, PCS-centered upper bound
            epistemic_lower=epistemic_lower,
            epistemic_upper=epistemic_upper
        )
    else:
        print("-"*40)
        logger.info("Calibrating with standard bounds...")
        clear_model.calibrate(
            y_valid.flatten(),
            median_epistemic=median_epistemic,
            aleatoric_median=aleatoric_median,
            aleatoric_lower=aleatoric_lower,
            aleatoric_upper=aleatoric_upper,
            epistemic_lower=epistemic_lower,
            epistemic_upper=epistemic_upper
        )

def predict_with_clear(clear_model, X_test_df, pcs_median_test, pcs_lower_test, pcs_upper_test, aleatoric_median_test, aleatoric_lower_test, aleatoric_upper_test):
    """
    Generate predictions with a calibrated CLEAR model.
    
    Args:
        clear_model: Calibrated CLEAR model
        X_test_df: Test features
        pcs_median_test: PCS median for test data
        pcs_lower_test: PCS lower bounds for test data
        pcs_upper_test: PCS upper bounds for test data
        aleatoric_median_test: Aleatoric median for test data
        aleatoric_lower_test: Aleatoric lower bounds for test data
        aleatoric_upper_test: Aleatoric upper bounds for test data
        
    Returns:
        (lower_bounds, upper_bounds)
    """
    return clear_model.predict(
        X_test_df,
        external_epistemic={
            'median': pcs_median_test,
            'lower': pcs_lower_test,
            'upper': pcs_upper_test
        },
        external_aleatoric={
            'median': aleatoric_median_test,
            'lower': aleatoric_lower_test,
            'upper': aleatoric_upper_test
        }
    )

def compute_naive_intervals(y_calib, calib_predictions, test_predictions, coverage):
    """
    Compute A-Naive (constant width) and S-Naive (constant absolute width) intervals.
    
    Args:
        y_calib: Calibration targets
        calib_predictions: Point predictions for calibration data
        test_predictions: Point predictions for test data
        coverage: Target coverage probability
        
    Returns:
        (a_naive_lower, a_naive_upper, s_naive_lower, s_naive_upper)
    """
    logger = logging.getLogger()
    
    # A-Naive (asymmetric constant width)
    logger.info("Computing A-Naive intervals (constant width)...")
    
    # Compute residuals on calibration data
    residuals = y_calib - calib_predictions
    
    # Find the α/2 and 1-α/2 quantiles of the residuals
    alpha = 1 - coverage
    lower_quantile = np.quantile(residuals, alpha/2, method='higher')
    upper_quantile = np.quantile(residuals, 1-alpha/2, method='higher')
    
    # Create constant-width prediction intervals
    a_naive_lower = test_predictions + lower_quantile
    a_naive_upper = test_predictions + upper_quantile
    
    # S-Naive (symmetric constant width)
    logger.info("Computing S-Naive intervals (constant absolute width)...")
    
    # Calculate absolute residuals on calibration data
    abs_residuals = np.abs(y_calib - calib_predictions)
    
    # Find the quantile corresponding to desired coverage
    gamma_naive = np.quantile(abs_residuals, coverage, method='higher')
    
    # Create fixed-width intervals
    s_naive_lower = test_predictions - gamma_naive
    s_naive_upper = test_predictions + gamma_naive
    
    return a_naive_lower, a_naive_upper, s_naive_lower, s_naive_upper


def main():
    default_datasets = [
        'data_computer',
        'data_ailerons',
        'data_airfoil',
        'data_ca_housing',
        'data_concrete',
        'data_elevator',
        'data_energy_efficiency',
        'data_insurance',
        'data_kin8nm',
        'data_miami_housing',
        'data_naval_propulsion',
        'data_parkinsons',
        'data_powerplant',
        'data_sulfur',
        'data_superconductor',
        'data_qsar',
        'data_allstate' # takes computationally long, but we have maintained it
    ]
    parser = argparse.ArgumentParser(description='Inference script for CLEAR-based prediction intervals.')
    parser.add_argument("--datasets", type=str, default=','.join(default_datasets), 
                        help='Dataset key(s) to evaluate. Can be a single dataset or comma-separated list.')
    parser.add_argument("--run", type=int, help="Specific run number to process (default: process all runs).")
    parser.add_argument("--runs", type=str, help="Comma-separated list of run numbers to process (e.g., '0,1,2,3,4').")
    parser.add_argument("--coverage", type=float, default=0.95, help="Target coverage probability (default: 0.95).")
    parser.add_argument("--min_lambda", type=float, default=0, help="Minimum lambda value for grid search.")
    parser.add_argument("--max_lambda", type=float, default=100, help="Maximum lambda value for grid search.")
    parser.add_argument("--models_dir", type=str, default="../../models/pcs_top1_qpcs_10", help="Directory containing ensemble results.")
    parser.add_argument("--save_results", type=str, default=None, help="Path to save results pickle file")
    parser.add_argument("--quantile_model", type=str, default="rf", 
        help="Quantile regression model to use by CLEAR. Valid options include 'linear','xgb','rf','extratrees', 'sqr', 'simultaneousquantileregressor', 'gradientboostingregressor' or a custom class.")
    parser.add_argument("--n_jobs", type=int, default=30, help="Number of jobs to run in parallel (default: 30 for high-performance systems).")
    parser.add_argument("--n_bootstraps", type=int, default=100, help="Number of bootstrap samples (default: 100)")
    parser.add_argument("--generate_tables", action="store_true", help="Generate CSV tables for aggregated metrics.")
    # Add new argument for median source
    parser.add_argument("--median_source", type=str, choices=["pcs","cqr"], default="pcs", help="Source for point predictor baseline: 'pcs' for ensemble median or 'cqr' for quantile regression median.")
    # Add logging-related arguments
    parser.add_argument("--log_file", type=str, default="auto", 
                        help="Path to log file. Use 'none' to disable file logging. Default 'auto' creates timestamped logs.")
    parser.add_argument("--global_log", action="store_true",
                        help="Create a single log file for all datasets rather than separate files.")
    parser.add_argument("--log_level", type=str, default="INFO", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level.")
    parser.add_argument("--approach", type=str, choices=["standard", "residual", "both"], 
                    default="standard", help="Approach to use: standard, residual, or both")
    parser.add_argument("--csv_results_dir", type=str, default=None,
                        help="Directory where the CSV table files will be saved. Can be an absolute path or "
                             "relative to the current working directory. If not provided, defaults to "
                             "'[PROJECT_ROOT]/results'. Example: '../../results/my_variant_results'.")
    args = parser.parse_args()

    # Remove the validation check for residual_only
    # if args.residual_only and not args.use_residual:
    #     print("Error: --residual_only requires --use_residual to be enabled.")
    #     return

    # Set up logging
    log_level = getattr(logging, args.log_level)
    
    # Base directory is two levels up from the script
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    logs_dir = os.path.join(base_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Global log setup - create a single log file for all datasets
    if args.global_log and args.log_file.lower() != "none":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get all datasets as a single string
        datasets = "_".join([d.strip() for d in args.datasets.split(",")])
        if len(datasets) > 30:  # Truncate if too long
            datasets = datasets[:27] + "..."
            
        # Format a descriptive filename
        coverage = f"{int(args.coverage*100)}"
        
        # Use custom name or auto-generated
        if args.log_file.lower() != "auto":
            base_name = os.path.splitext(args.log_file)[0]
            global_log_file = os.path.join(logs_dir, f"{base_name}_{timestamp}.log")
        else:
            global_log_file = os.path.join(logs_dir, f"benchmark_all_{timestamp}.log")
        
        # Set up the global logger
        logger = setup_logging(global_log_file, log_level)
        logger.info(f"Starting global log for all experiments at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Datasets: {args.datasets}")
        logger.info(f"Command-line arguments: {args}")
        
        # Store the global log file
        args.current_log_file = global_log_file
    else:
        # No global log, will set up individual loggers for each dataset
        args.current_log_file = None
        logger = logging.getLogger()
        logger.setLevel(log_level)

    # Check for mutually exclusive arguments
    if args.run is not None and args.runs is not None:
        logger.error("Error: Cannot specify both --run and --runs. Please use only one.")
        return

    # Define output directory for CSV tables
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(os.path.dirname(script_dir)) # base_dir is the project root
    
    if args.csv_results_dir:
        # User-provided path, resolve it (can be relative to CWD or absolute)
        tables_csv_dir = os.path.abspath(args.csv_results_dir)
    else:
        # Default path relative to project root
        tables_csv_dir = os.path.join(base_dir, "results")
    
    # Create directory if it doesn't exist
    os.makedirs(tables_csv_dir, exist_ok=True)
    
    # Determine run numbers to process
    run_nums = None
    if args.run is not None:
        run_nums = [args.run]
    elif args.runs is not None:
        run_nums = [int(r.strip()) for r in args.runs.split(",")]
    
    # Process datasets
    datasets = [d.strip() for d in args.datasets.split(",")]
    all_dataset_metrics = {}
    
    for dataset in datasets:
        # Set up individual logging for this dataset if we're not using a global log
        if not args.global_log and args.log_file.lower() != "none":
            # Create a descriptive filename based on the command
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            coverage = f"{int(args.coverage*100)}"
            
            # Include run numbers in filename if specified
            run_part = ""
            if args.run is not None:
                run_part = f"_run{args.run}"
            elif args.runs is not None:
                run_nums = [r.strip() for r in args.runs.split(",")]
                run_part = f"_runs{'_'.join(run_nums)}"
            
            # If user specified a custom log_file name that's not "auto", use it as a base name
            if args.log_file.lower() != "auto":
                # Strip extension if present
                base_name = os.path.splitext(args.log_file)[0]
                log_file = os.path.join(logs_dir, f"{base_name}_{dataset}_{timestamp}.log")
            else:
                # Use auto-generated name with dataset, coverage, and timestamp
                log_file = os.path.join(logs_dir, f"benchmark_{dataset}_cov{coverage}{run_part}_{timestamp}.log")
                
            # Set up dataset-specific logger
            logger = setup_logging(log_file, log_level)
            logger.info(f"Logging dataset {dataset} to file: {log_file}")
        else:
            # Use the global logger or console-only logger already set up
            logger = logging.getLogger()
            if args.current_log_file:
                logger.info(f"\n{'-'*80}\nProcessing dataset: {dataset}\n{'-'*80}")
        
        logger.info(f"\n{'='*40}\nProcessing dataset: {dataset}\n{'='*40}")
        
        try:
            # Process the dataset and get metrics
            dataset_metrics = process_dataset(dataset, args, run_nums)
            
            if dataset_metrics:
                all_dataset_metrics[dataset] = dataset_metrics
                
                # Generate tables for this dataset if requested
                if args.generate_tables:
                    generate_tables(dataset_metrics, dataset, args.coverage, tables_csv_dir, None)
        except Exception as e:
            logger.error(f"Error processing dataset {dataset}: {e}")
            logger.error(traceback.format_exc())
        
        # Force garbage collection between datasets to free memory
        import gc
        gc.collect()
    
    logger.info("\nBenchmark completed!")
    
if __name__ == "__main__":
    # Set the plot rendering to non-interactive
    matplotlib.use('Agg')
    main()


###################################################################
# To run the experiments with the models for `qpcs_10` (variant a) then followed by `qxgb_10` (variant b) and `pcs_10` (variant c), use the following command:
# python benchmark_real_data.py --coverage 0.95 --generate_tables --n_jobs 25 --global_log --approach both --models_dir ../../models/pcs_top1_qpcs_10 --csv_results_dir ../../results/qPCS_all_10seeds_all
# python benchmark_real_data.py --coverage 0.95 --generate_tables --n_jobs 30 --global_log --approach both --models_dir ../../models/pcs_top1_qxgb_10 --csv_results_dir ../../results/qPCS_qxgb_10seeds_qxgb
# python benchmark_real_data.py --coverage 0.95 --generate_tables --n_jobs 30 --global_log --approach both --models_dir ../../models/pcs_top1_mean_10 --csv_results_dir ../../results/PCS_all_10seeds_qrf
# Note that the GAM in variant (a) may not converge for `data_naval_propulsion`. This has been documented in the paper as a footnote and it's a bug related to the `pygam` package: https://github.com/dswah/pyGAM/issues/357

###################################################################

# One liner to do variant b, c and a (fastest order of variants execution)
# python benchmark_real_data.py --coverage 0.95 --generate_tables --n_jobs 25 --global_log --approach both --models_dir ../../models/pcs_top1_qxgb_10 --csv_results_dir ../../results/qPCS_qxgb_10seeds_qxgb ; python benchmark_real_data.py --coverage 0.95 --generate_tables --n_jobs 30 --global_log --approach both --models_dir ../../models/pcs_top1_mean_10 --csv_results_dir ../../results/PCS_all_10seeds_qrf ; python benchmark_real_data.py --coverage 0.95 --generate_tables --n_jobs 30 --global_log --approach both --models_dir ../../models/pcs_top1_qpcs_10 --csv_results_dir ../../results/qPCS_all_10seeds_all