"""
Benchmark of real data with SQR/DE components.
"""

import os
import sys
import time
import argparse
import pickle
import copy
from pathlib import Path
import logging
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

from utils import safe_flatten, get_available_datasets, setup_logging

# Ensure local imports work
# if str(ROOT_DIR) not in sys.path:
#     sys.path.append(str(ROOT_DIR))
ROOT_DIR = Path(__file__).resolve().parents[2]
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)
sys.path.append(os.path.join(script_dir, '..'))

# Import the aleatoric and epistemic components
from clear.models.deep_ensembles import PureEpistemicEnsemble
from clear.models.sqr import SQR

# CLEAR imports
from clear.clear import CLEAR
from clear.metrics import evaluate_intervals
from clear.utils import load_ensemble_pickle


def evaluate_method(y_test, lower_bounds, upper_bounds, method_name, alpha=0.05, f=None):
    """Evaluate prediction intervals and return comprehensive metrics."""
    # Use the same alpha as the original benchmark (1 - coverage)
    metrics = evaluate_intervals(y_test, lower_bounds, upper_bounds, alpha=alpha, f=f)
    
    # Add method name
    metrics['method'] = method_name
    
    # Convert numpy values to Python scalars for better display
    for key, value in metrics.items():
        if isinstance(value, (np.ndarray, np.number)):
            metrics[key] = float(value)
    
    # Ensure all standard metrics are present (use NaN if missing)
    standard_metrics = ["PICP", "NIW", "MPIW", "QuantileLoss", "ExpectileLoss", 
                       "CRPS", "AUC", "NCIW", "IntervalScoreLoss", "c_test_cal"]
    for metric in standard_metrics:
        if metric not in metrics:
            metrics[metric] = np.nan
    
    return metrics

def predict_pure_epistemic_ensemble(ensemble_model, X_test, coverage=0.95):
    """Get predictions from pure epistemic ensemble."""
    mean_pred, lower_bound, upper_bound = ensemble_model.predict_interval(X_test, coverage)
    return mean_pred, lower_bound, upper_bound


def process_dataset(dataset_name: str, args: argparse.Namespace):
    """Process dataset with SQR/DE components and full metrics."""
    print(f"\n{'='*60}")
    print(f"Processing dataset: {dataset_name}")
    print(f"{'='*60}")
    
    # Load data
    dataset_key = dataset_name if dataset_name.startswith("data_") else f"data_{dataset_name}"
    pkl_path = Path(args.models_dir) / f"{dataset_key}_pcs_results_{int(args.coverage*100)}.pkl"
    
    if not pkl_path.exists():
        raise FileNotFoundError(f"No results found for {dataset_name} at {pkl_path}")
    
    print(f"[{dataset_name}] Loading data from {pkl_path}")
    ensemble_dict = load_ensemble_pickle(pkl_path)
    
    # Initialize results storage
    all_results = []
    
    # Determine which runs to process
    if args.runs is None:
        all_available_runs = [k for k in ensemble_dict.keys() if k.startswith('run_')]
        
        rng_global = np.random.RandomState(args.seed)

        if args.random_runs:
            # Randomly select runs using seeded RNG
            if len(all_available_runs) > args.max_runs:
                runs_to_process = sorted(rng_global.choice(all_available_runs, size=args.max_runs, replace=False))
            else:
                runs_to_process = all_available_runs
            print(f"[{dataset_name}] Randomly selected runs: {runs_to_process}")
        else:
            # Take first N runs (original behavior)
            runs_to_process = all_available_runs[:args.max_runs]
    else:
        if isinstance(args.runs, int):
            runs_to_process = [f"run_{args.runs}"]
        elif isinstance(args.runs, (list, tuple)):
            runs_to_process = [f"run_{r}" for r in args.runs]
        else:
            runs_to_process = [f"run_{int(args.runs)}"]
    
    runs_to_process = [r for r in runs_to_process if r in ensemble_dict]
    
    if not runs_to_process:
        raise ValueError(f"No valid runs found in {pkl_path}")
    
    print(f"[{dataset_name}] Processing {len(runs_to_process)} runs: {runs_to_process}")
    
    for run_idx, run_key in enumerate(runs_to_process):
        print(f"\n[{dataset_name}] Processing {run_key}...")
        run_data = ensemble_dict[run_key]
        
        # Extract data splits
        X_train = np.array(run_data.get("x_train", run_data.get("X_train")))
        y_train = safe_flatten(run_data.get("y_train", run_data.get("Y_train")))
        X_valid = np.array(run_data.get("x_val", run_data.get("X_valid", run_data.get("x_valid"))))
        y_valid = safe_flatten(run_data.get("y_val", run_data.get("Y_valid", run_data.get("y_valid"))))
        X_test = np.array(run_data.get("x_test", run_data.get("X_test")))
        y_test = safe_flatten(run_data.get("y_test", run_data.get("Y_test")))
        
        # Set run-specific RNGs (align with PCS benchmark behavior)
        run_seed = args.seed + run_idx
        rng_run = np.random.RandomState(run_seed)
        torch.manual_seed(run_seed)
        
        n_samples = len(X_train)

        # Original parameters (ran it like this until `computer`)
        # if n_samples < 1000:
        #     n_ensemble = 15
        #     # sqr_hidden = (128, 128, 64)
        #     sqr_hidden = (256, 256, 128)
        #     sqr_epochs = 4000
        # elif n_samples < 5000:
        #     n_ensemble = 10
        #     sqr_hidden = (256, 256, 128)
        #     sqr_epochs = 3000
        # else:
        #     n_ensemble = 7
        #     sqr_hidden = (512, 256, 128)
        #     sqr_epochs = 2000        

        # Adaptive parameters based on dataset size
        n_ensemble = 5
        sqr_hidden = (128, 128, 64)
        sqr_epochs = 4000
        # Override with args if specified
        if args.n_ensemble is not None:
            n_ensemble = args.n_ensemble
        
        # 1. Train Pure Epistemic Deep Ensemble
        print(f"[{dataset_name}] Training Pure Epistemic Deep Ensemble (n_ensemble={n_ensemble})...")
        start_time = time.time()
        
        epistemic_model = PureEpistemicEnsemble(
            n_members=n_ensemble,
            hidden_sizes=(256, 128),
            learning_rate=1e-3,
            n_epochs=args.ensemble_epochs,
            batch_size=args.batch_size,
            dropout_rate=0.1,
            weight_decay=1e-5,
            diversity_strategies=['random_init', 'architecture', 'dropout', 
                                'bootstrap', 'lr_schedule', 'data_augment'],
            calibrate_c=True,  # Calibrate the confidence multiplier
            random_state=run_seed,
            verbose=args.verbose
        )
        
        epistemic_model.fit(X_train, y_train, X_valid, y_valid)
        
        # Get predictions
        f_val, epi_lower_val, epi_upper_val = epistemic_model.predict_interval(X_valid, args.coverage)
        f_test, epi_lower_test, epi_upper_test = epistemic_model.predict_interval(X_test, args.coverage)
        f_train, _, _ = epistemic_model.predict_interval(X_train, args.coverage)
        
        # Check performance
        y_range = float(np.max(y_train) - np.min(y_train) + 1e-12)
        ensemble_rmse = np.sqrt(np.mean((f_val - y_valid)**2))
        epi_coverage = np.mean((y_valid >= epi_lower_val) & (y_valid <= epi_upper_val))
        ensemble_width = np.mean(epi_upper_val - epi_lower_val)
        
        ensemble_time = time.time() - start_time
        print(f"[{dataset_name}] Pure Epistemic DE training completed in {ensemble_time:.1f}s")
        print(f"[{dataset_name}] Ensemble validation RMSE: {ensemble_rmse:.3f}")
        print(f"[{dataset_name}] Ensemble validation PICP: {epi_coverage:.3f} (target: {args.coverage:.3f})")
        print(f"[{dataset_name}] Ensemble validation interval width: {ensemble_width:.3f} ({ensemble_width/y_range*100:.1f}% of y range)")
        print(f"[{dataset_name}] Calibrated c multiplier: {epistemic_model.c_multiplier:.3f}")
        
        # 2. Train SQR for aleatoric
        print(f"[{dataset_name}] Training SQR aleatoric model...")
        start_time = time.time()
        
        # Create SQR model
        sqr_model = SQR(
            alpha=1 - args.coverage,
            hidden_sizes=sqr_hidden,
            learning_rate=args.sqr_lr,
            n_epochs=sqr_epochs,
            batch_size=128,
            dropout=0.2,
            weight_decay=1e-5,
            crossing_penalty=1.0,
            ensemble_size=1 if args.fast_mode else 3,  # Single model for speed in fast mode
            patience=200,
            random_state=run_seed,
            verbose=args.verbose
        )
        
        # Train on raw targets (not residuals for standalone baseline)
        sqr_model.fit(X_train, y_train)
        sqr_time = time.time() - start_time
        
        # Get predictions
        sqr_median_val, sqr_lower_val, sqr_upper_val = sqr_model.predict(X_valid)
        sqr_median_test, sqr_lower_test, sqr_upper_test = sqr_model.predict(X_test)
        
        # Check SQR coverage
        sqr_coverage = np.mean((y_valid >= sqr_lower_val) & (y_valid <= sqr_upper_val))
        sqr_width_val = np.mean(sqr_upper_val - sqr_lower_val)
        print(f"[{dataset_name}] SQR training completed in {sqr_time:.1f}s")
        print(f"[{dataset_name}] SQR validation PICP: {sqr_coverage:.3f} (target: {args.coverage:.3f})")
        print(f"[{dataset_name}] SQR validation interval width: {sqr_width_val:.3f} ({sqr_width_val/y_range*100:.1f}% of y range)")
        
        # 3. CLEAR with components
        print(f"[{dataset_name}] Calibrating CLEAR...")
        clear_model = CLEAR(
            desired_coverage=args.coverage,
            lambdas=np.concatenate((np.linspace(0, 0.09, 10), np.logspace(-1, 2, 401))),
            n_bootstraps=1,
            random_state=run_seed,
            n_jobs=1
        )
        
        # Train SQR on residuals for CLEAR (if requested)
        if args.sqr_residuals:
            print(f"[{dataset_name}] CLEAR mode: residual-based SQR (using epistemic center + residual quantiles)")
            print(f"[{dataset_name}] Training residual-based SQR for CLEAR...")
            clear_sqr_model = SQR(
                alpha=1 - args.coverage,
                hidden_sizes=sqr_hidden,
                learning_rate=args.sqr_lr,
                n_epochs=sqr_epochs,
                batch_size=128,
                dropout=0.2,
                weight_decay=1e-5,
                crossing_penalty=1.0,
                ensemble_size=1 if args.fast_mode else 3,
                patience=200,
                random_state=run_seed,
                verbose=args.verbose
            )
            
            # Train on residuals
            residuals_train = y_train - f_train
            clear_sqr_model.fit(X_train, residuals_train)
            
            # Get residual predictions and add back epistemic center
            _, res_lower_val, res_upper_val = clear_sqr_model.predict(X_valid)
            _, res_lower_test, res_upper_test = clear_sqr_model.predict(X_test)
            
            clear_ale_median_val = f_val  # Use epistemic as median
            clear_ale_lower_val = f_val + res_lower_val
            clear_ale_upper_val = f_val + res_upper_val
            clear_ale_median_test = f_test
            clear_ale_lower_test = f_test + res_lower_test
            clear_ale_upper_test = f_test + res_upper_test
        else:
            print(f"[{dataset_name}] CLEAR mode: direct SQR (using standalone SQR aleatoric quantiles)")
            # Use standalone SQR predictions for CLEAR
            clear_ale_median_val = sqr_median_val
            clear_ale_lower_val = sqr_lower_val
            clear_ale_upper_val = sqr_upper_val
            clear_ale_median_test = sqr_median_test
            clear_ale_lower_test = sqr_lower_test
            clear_ale_upper_test = sqr_upper_test
        
        # Calibrate CLEAR
        clear_model.calibrate(
            y_calib=y_valid,
            median_epistemic=f_val,
            aleatoric_median=clear_ale_median_val,
            aleatoric_lower=clear_ale_lower_val,
            aleatoric_upper=clear_ale_upper_val,
            epistemic_lower=epi_lower_val,
            epistemic_upper=epi_upper_val,
            pythagoras=False,
            verbose=False
        )
        
        print(f"[{dataset_name}] Calibration complete: lambda={clear_model.optimal_lambda:.4f}, gamma={clear_model.gamma:.4f}")

        # Build auxiliary CLEAR models for fixed gamma or lambda (direct SQR)
        lambda_one_direct = CLEAR(
            desired_coverage=args.coverage,
            lambdas=clear_model.lambdas,
            n_bootstraps=1,
            random_state=run_seed,
            n_jobs=1,
            fixed_lambda=1.0
        )
        lambda_one_direct.calibrate(
            y_calib=y_valid,
            median_epistemic=f_val,
            aleatoric_median=sqr_median_val,
            aleatoric_lower=sqr_lower_val,
            aleatoric_upper=sqr_upper_val,
            epistemic_lower=epi_lower_val,
            epistemic_upper=epi_upper_val,
            verbose=False
        )

        gamma_one_direct = CLEAR(
            desired_coverage=args.coverage,
            lambdas=np.linspace(0, 5, 51),
            n_bootstraps=1,
            random_state=run_seed,
            n_jobs=1,
            fixed_gamma=1.0
        )
        gamma_one_direct.calibrate(
            y_calib=y_valid,
            median_epistemic=f_val,
            aleatoric_median=sqr_median_val,
            aleatoric_lower=sqr_lower_val,
            aleatoric_upper=sqr_upper_val,
            epistemic_lower=epi_lower_val,
            epistemic_upper=epi_upper_val,
            verbose=False
        )

        lambda_one_direct_bounds = lambda_one_direct.predict(
            X_test,
            external_epistemic={'median': f_test, 'lower': epi_lower_test, 'upper': epi_upper_test},
            external_aleatoric={'median': sqr_median_test, 'lower': sqr_lower_test, 'upper': sqr_upper_test}
        )
        gamma_one_direct_bounds = gamma_one_direct.predict(
            X_test,
            external_epistemic={'median': f_test, 'lower': epi_lower_test, 'upper': epi_upper_test},
            external_aleatoric={'median': sqr_median_test, 'lower': sqr_lower_test, 'upper': sqr_upper_test}
        )

        if args.sqr_residuals:
            lambda_one_residual = CLEAR(
                desired_coverage=args.coverage,
                lambdas=clear_model.lambdas,
                n_bootstraps=1,
                random_state=run_seed,
                n_jobs=1,
                fixed_lambda=1.0
            )
            lambda_one_residual.calibrate(
                y_calib=y_valid,
                median_epistemic=f_val,
                aleatoric_median=clear_ale_median_val,
                aleatoric_lower=clear_ale_lower_val,
                aleatoric_upper=clear_ale_upper_val,
                epistemic_lower=epi_lower_val,
                epistemic_upper=epi_upper_val,
                verbose=False
            )

            gamma_one_residual = CLEAR(
                desired_coverage=args.coverage,
                lambdas=np.linspace(0, 5, 51),
                n_bootstraps=1,
                random_state=run_seed,
                n_jobs=1,
                fixed_gamma=1.0
            )
            gamma_one_residual.calibrate(
                y_calib=y_valid,
                median_epistemic=f_val,
                aleatoric_median=clear_ale_median_val,
                aleatoric_lower=clear_ale_lower_val,
                aleatoric_upper=clear_ale_upper_val,
                epistemic_lower=epi_lower_val,
                epistemic_upper=epi_upper_val,
                verbose=False
            )

            lambda_one_residual_bounds = lambda_one_residual.predict(
                X_test,
                external_epistemic={'median': f_test, 'lower': epi_lower_test, 'upper': epi_upper_test},
                external_aleatoric={'median': clear_ale_median_test, 'lower': clear_ale_lower_test, 'upper': clear_ale_upper_test}
            )
            gamma_one_residual_bounds = gamma_one_residual.predict(
                X_test,
                external_epistemic={'median': f_test, 'lower': epi_lower_test, 'upper': epi_upper_test},
                external_aleatoric={'median': clear_ale_median_test, 'lower': clear_ale_lower_test, 'upper': clear_ale_upper_test}
            )
        else:
            lambda_one_residual = gamma_one_residual = None
            lambda_one_residual_bounds = gamma_one_residual_bounds = (None, None)

        lambda_one_direct_lower, lambda_one_direct_upper = lambda_one_direct_bounds
        gamma_one_direct_lower, gamma_one_direct_upper = gamma_one_direct_bounds

        if args.sqr_residuals:
            lambda_one_residual_lower, lambda_one_residual_upper = lambda_one_residual_bounds
            gamma_one_residual_lower, gamma_one_residual_upper = gamma_one_residual_bounds
        else:
            lambda_one_residual_lower = lambda_one_residual_upper = None
            gamma_one_residual_lower = gamma_one_residual_upper = None

        # Get CLEAR predictions
        clear_lower, clear_upper = clear_model.predict(
            X_test,
            external_epistemic={
                'median': f_test,
                'lower': epi_lower_test,
                'upper': epi_upper_test
            },
            external_aleatoric={
                'median': clear_ale_median_test,
                'lower': clear_ale_lower_test,
                'upper': clear_ale_upper_test
            }
        )
        
        # 4. Compute conformalized versions
        n_calib = len(y_valid)
        q_level_std = min(args.coverage * (1 + 1 / max(1, n_calib)), 1.0)
        
        # Conformalized SQR
        sqr_scores_val = np.maximum(sqr_lower_val - y_valid, y_valid - sqr_upper_val)
        sqr_adj = np.quantile(sqr_scores_val, q_level_std, method='higher')
        sqr_lower_conf = sqr_lower_test - sqr_adj
        sqr_upper_conf = sqr_upper_test + sqr_adj
        
        # Conformalized Ensemble
        de_scores_val = np.maximum(epi_lower_val - y_valid, y_valid - epi_upper_val)
        de_adj = np.quantile(de_scores_val, q_level_std, method='higher')
        de_lower_conf = epi_lower_test - de_adj
        de_upper_conf = epi_upper_test + de_adj
        
        # 5. Evaluate all methods
        alpha = 1 - args.coverage
        
        # CLEAR (Components)
        clear_median = (clear_lower + clear_upper) / 2
        clear_metrics = evaluate_method(y_test, clear_lower, clear_upper, 'CLEAR', 
                                      alpha=alpha, f=clear_median)
        clear_metrics['lambda'] = clear_model.optimal_lambda
        clear_metrics['gamma'] = clear_model.gamma
        clear_metrics['run'] = run_key
        clear_metrics['dataset'] = dataset_name
        all_results.append(clear_metrics)
        
        # Pure Epistemic DE (calibrated)
        ensemble_metrics = evaluate_method(y_test, epi_lower_test, epi_upper_test, 'DE_calibrated',
                                         alpha=alpha, f=f_test)
        ensemble_metrics['run'] = run_key
        ensemble_metrics['dataset'] = dataset_name
        ensemble_metrics['c_multiplier'] = epistemic_model.c_multiplier
        all_results.append(ensemble_metrics)
        
        # Conformalized Pure Epistemic DE
        ensemble_conf_metrics = evaluate_method(y_test, de_lower_conf, de_upper_conf, 'DE_conformal',
                                               alpha=alpha, f=f_test)
        ensemble_conf_metrics['run'] = run_key
        ensemble_conf_metrics['dataset'] = dataset_name
        all_results.append(ensemble_conf_metrics)
        
        # SQR (uncalibrated)
        sqr_metrics = evaluate_method(y_test, sqr_lower_test, sqr_upper_test, 'SQR_uncalibrated',
                                    alpha=alpha, f=sqr_median_test)
        sqr_metrics['run'] = run_key
        sqr_metrics['dataset'] = dataset_name
        sqr_metrics['residual_mode'] = False
        all_results.append(sqr_metrics)

        # Conformalized SQR
        sqr_conf_metrics = evaluate_method(y_test, sqr_lower_conf, sqr_upper_conf, 'SQR_conformal',
                                           alpha=alpha, f=sqr_median_test)
        sqr_conf_metrics['run'] = run_key
        sqr_conf_metrics['dataset'] = dataset_name
        sqr_conf_metrics['residual_mode'] = False
        all_results.append(sqr_conf_metrics)

        # Lambda=1 (direct SQR) using fixed-lambda CLEAR
        lambda_one_metrics = evaluate_method(
            y_test,
            lambda_one_direct_lower,
            lambda_one_direct_upper,
            'lambda_one',
            alpha=alpha,
            f=sqr_median_test
        )
        lambda_one_metrics['run'] = run_key
        lambda_one_metrics['dataset'] = dataset_name
        lambda_one_metrics['lambda'] = 1.0
        lambda_one_metrics['gamma'] = lambda_one_direct.gamma
        lambda_one_metrics['residual_mode'] = False
        all_results.append(lambda_one_metrics)

        # Gamma=1 (direct SQR) using fixed-gamma CLEAR
        gamma_one_metrics = evaluate_method(
            y_test,
            gamma_one_direct_lower,
            gamma_one_direct_upper,
            'gamma_1',
            alpha=alpha,
            f=sqr_median_test
        )
        gamma_one_metrics['run'] = run_key
        gamma_one_metrics['dataset'] = dataset_name
        gamma_one_metrics['lambda'] = gamma_one_direct.optimal_lambda
        gamma_one_metrics['gamma'] = 1.0
        gamma_one_metrics['residual_mode'] = False
        all_results.append(gamma_one_metrics)

        # Gamma=1 residual baseline (if residuals enabled)
        if args.sqr_residuals:
            # Residual-based SQR baseline
            sqr_r_metrics = evaluate_method(
                y_test,
                clear_ale_lower_test,
                clear_ale_upper_test,
                'SQR_R_uncalibrated',
                alpha=alpha,
                f=clear_ale_median_test
            )
            sqr_r_metrics['run'] = run_key
            sqr_r_metrics['dataset'] = dataset_name
            sqr_r_metrics['residual_mode'] = True
            all_results.append(sqr_r_metrics)

            # Residual-based SQR conformalized baseline
            sqr_r_scores_val = np.maximum(clear_ale_lower_val - y_valid, y_valid - clear_ale_upper_val)
            sqr_r_adj = np.quantile(sqr_r_scores_val, q_level_std, method='higher')
            sqr_r_lower_conf = clear_ale_lower_test - sqr_r_adj
            sqr_r_upper_conf = clear_ale_upper_test + sqr_r_adj

            sqr_r_conf_metrics = evaluate_method(
                y_test,
                sqr_r_lower_conf,
                sqr_r_upper_conf,
                'SQR_R_conformal',
                alpha=alpha,
                f=clear_ale_median_test
            )
            sqr_r_conf_metrics['run'] = run_key
            sqr_r_conf_metrics['dataset'] = dataset_name
            sqr_r_conf_metrics['residual_mode'] = True
            all_results.append(sqr_r_conf_metrics)

            lambda_one_r_metrics = evaluate_method(
                y_test,
                lambda_one_residual_lower,
                lambda_one_residual_upper,
                'lambda_one_r',
                alpha=alpha,
                f=clear_ale_median_test
            )
            lambda_one_r_metrics['run'] = run_key
            lambda_one_r_metrics['dataset'] = dataset_name
            lambda_one_r_metrics['lambda'] = 1.0
            lambda_one_r_metrics['gamma'] = lambda_one_residual.gamma
            lambda_one_r_metrics['residual_mode'] = True
            all_results.append(lambda_one_r_metrics)

            gamma_one_r_metrics = evaluate_method(
                y_test,
                gamma_one_residual_lower,
                gamma_one_residual_upper,
                'gamma_1_r',
                alpha=alpha,
                f=clear_ale_median_test
            )
            gamma_one_r_metrics['run'] = run_key
            gamma_one_r_metrics['dataset'] = dataset_name
            gamma_one_r_metrics['lambda'] = gamma_one_residual.optimal_lambda
            gamma_one_r_metrics['gamma'] = 1.0
            gamma_one_r_metrics['residual_mode'] = True
            all_results.append(gamma_one_r_metrics)
        
        # Print results for this run - show all metrics like original
        print(f"\n[{dataset_name}] {run_key} Results:")
        # Add explicit note of CLEAR aleatoric source
        print(f"  CLEAR aleatoric source: {'residual-based SQR' if args.sqr_residuals else 'direct SQR'}")
        print(f"  ** Components: Pure Epistemic DE + SQR ensemble (3 models, each predicts both quantiles) **")
        for method in ['CLEAR', 'DE_calibrated', 'DE_conformal', 
                      'SQR_uncalibrated', 'SQR_conformal', 'lambda_one', 'gamma_1',
                      'SQR_R_uncalibrated', 'SQR_R_conformal', 'lambda_one_r', 'gamma_1_r']:
            method_results = [m for m in all_results if m['method'] == method and m['run'] == run_key]
            if method_results:
                m = method_results[0]
                print(f"  {method}:")
                # Show all key metrics
                keys = ['PICP', 'NIW', 'MPIW', 'NCIW', 'CRPS', 'QuantileLoss', 'AUC']
                for k in keys:
                    val = m.get(k, np.nan)
                    print(f"    {k}: {val:.4f}")
                if 'lambda' in m:
                    print(f"    Lambda: {m['lambda']:.4f}")
                    print(f"    Gamma: {m['gamma']:.4f}")
                if 'c_multiplier' in m:
                    print(f"    C_multiplier: {m['c_multiplier']:.4f}")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Aggregate results across runs
    metrics_to_aggregate = ['PICP', 'NIW', 'MPIW', 'QuantileLoss', 'ExpectileLoss', 
                           'CRPS', 'AUC', 'NCIW', 'IntervalScoreLoss', 'c_test_cal']
    
    agg_dict = {}
    for metric in metrics_to_aggregate:
        if metric in results_df.columns:
            agg_dict[metric] = ['mean', 'std']
    
    agg_results = results_df.groupby('method').agg(agg_dict).round(4)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = output_dir / f"de_sqr_results_{dataset_name}_{int(args.coverage*100)}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n[{dataset_name}] Results saved to {csv_path}")
    
    # Print summary
    print(f"\n[{dataset_name}] Summary across {len(runs_to_process)} runs:")
    print(agg_results)
    
    return results_df, agg_results


def main():
    parser = argparse.ArgumentParser(description='Benchmark CLEAR with DE and SQR components')
    parser.add_argument('--datasets', nargs='+', default=None,
                       help='Datasets to process (default: auto-detect)')
    parser.add_argument('--coverage', type=float, default=0.95,
                       help='Target coverage level (default: 0.95)')
    parser.add_argument('--runs', type=int, default=None,
                       help='Specific run number to process (default: all available)')
    parser.add_argument('--max_runs', type=int, default=10,
                       help='Maximum number of runs to process per dataset (default: 3)')
    parser.add_argument('--random_runs', default=False,
                       help='Randomly select runs instead of taking first N runs')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed (default: 0)')
    parser.add_argument('--models_dir', type=str, 
                       default='../../models/pcs_top1_qxgb_10_standard',
                       help='Directory containing model pickle files')
    parser.add_argument('--output_dir', type=str, 
                       default='../../results/de_sqr',
                       help='Output directory for results')
    
    # Ensemble parameters
    parser.add_argument('--n_ensemble', type=int, default=None,
                       help='Number of ensemble members (default: adaptive based on dataset size)')
    parser.add_argument('--ensemble_epochs', type=int, default=1500,
                       help='Ensemble training epochs (default: 1500)')
    parser.add_argument('--ensemble_lr', type=float, default=1e-3,
                       help='Ensemble learning rate (default: 1e-3)')
    
    # SQR parameters
    parser.add_argument('--sqr_epochs', type=int, default=None,
                       help='SQR training epochs (default: adaptive)')
    parser.add_argument('--sqr_lr', type=float, default=5e-4,
                       help='SQR learning rate (default: 5e-4)')
    parser.add_argument('--sqr_residuals', default = True,
                       help='Fit SQR on residuals for CLEAR (not for baselines)')
    
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training (default: 64)')
    parser.add_argument('--fast_mode', action='store_true',
                       help='Fast mode with single SQR model instead of ensemble')
    parser.add_argument('--verbose', action='store_true', 
                       help='Verbose printing')
    
    args = parser.parse_args()
    # Ensure logging is configured so that prints redirected by utils go to console
    setup_logging(log_level=(logging.DEBUG if args.verbose else logging.INFO))
    
    # Resolve paths
    script_dir = Path(__file__).resolve().parent

    models_dir = Path(args.models_dir).expanduser()
    if not models_dir.is_absolute():
        models_dir = (script_dir / models_dir).resolve()
    else:
        models_dir = models_dir.resolve()
    args.models_dir = str(models_dir)
    
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = (script_dir / output_dir).resolve()
    else:
        output_dir = output_dir.resolve()
    args.output_dir = str(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    print(f"Output directory: {output_dir}")
    
    # Auto-detect datasets if not specified
    if args.datasets is None:
        args.datasets = get_available_datasets(models_dir, int(args.coverage * 100))
        print(f"\nAuto-detected {len(args.datasets)} datasets: {args.datasets}")
    
    # Process each dataset
    all_results = {}
    failed_datasets = []
    
    for dataset in args.datasets:
        try:
            results_df, agg_results = process_dataset(dataset, args)
            all_results[dataset] = (results_df, agg_results)
        except Exception as e:
            print(f"\nError processing {dataset}: {str(e)}")
            failed_datasets.append(dataset)
            import traceback
            traceback.print_exc()
            continue
    
    # Report summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Successfully processed: {len(all_results)}/{len(args.datasets)} datasets")
    if failed_datasets:
        print(f"Failed datasets: {failed_datasets}")
    
    # Generate comparison table if we have results
    if len(all_results) > 0:
        # Create a summary CSV with all results
        summary_data = []
        for dataset, (results_df, _) in all_results.items():
            summary_data.append(results_df)
        
        if summary_data:
            summary_df = pd.concat(summary_data, ignore_index=True)
            summary_path = output_dir / f"all_de_sqr_results_{int(args.coverage*100)}.csv"
            summary_df.to_csv(summary_path, index=False)
            print(f"\nAll results saved to {summary_path}")
            
            # Create aggregated summary
            print("\n" + "="*60)
            print("FINAL COMPARISON (Average across all datasets)")
            print("="*60)
            
            comparison_metrics = ['PICP', 'NIW', 'MPIW', 'NCIW', 'CRPS', 'QuantileLoss', 'AUC']
            comparison_dict = {}
            for metric in comparison_metrics:
                if metric in summary_df.columns:
                    comparison_dict[metric] = ['mean', 'std']
            
            overall_summary = summary_df.groupby('method').agg(comparison_dict).round(4)
            
            # Print in a more readable format matching original
            print("\nMean values across all datasets and runs:")
            print(f"{'Method':<30}", end='')
            for metric in comparison_metrics:
                if metric in overall_summary.columns:
                    print(f"{metric:>10}", end='')
            print()
            print("-" * 100)
            
            for method in overall_summary.index:
                print(f"{method:<30}", end='')
                for metric in comparison_metrics:
                    if metric in overall_summary.columns:
                        mean_val = overall_summary.loc[method, (metric, 'mean')]
                        print(f"{mean_val:>10.4f}", end='')
                print()


if __name__ == "__main__":
    main()


# To run the experiments for the benchmark datasets, use the following command from the current `experiments` directory:
# python benchmark_real_data_de_sqr.py --coverage 0.95 --models_dir ../../models/pcs_top1_qxgb_10_standard --output_dir ../../results/de_sqr --seed 42 --batch_size 64 --ensemble_epochs 1500 --sqr_lr 5e-4 --verbose