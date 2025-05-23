import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from copy import deepcopy
import os
import sys
import argparse
import pickle
import time
import logging
# Define models - exactly as in originald
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from xgboost import XGBRegressor
from joblib import Parallel, delayed
from utils import convert_to_serializable, get_intervals_manually, setup_logging, print_model_performance, WorkerLogger
from sklearn.linear_model import LinearRegression, RidgeCV
from celer import LassoCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
from sklearn.neural_network import MLPRegressor

# Update imports to use the new PCS implementation
# Try to import from the installed package first, fall back to local path if not available
try:
    from PCS.regression import PCS_UQ, PCS_OOB
except ImportError:
    # Add PCS_UQ to path if needed
    pcs_uq_path = os.path.join("..", "..", "PCS_UQ", "src")
    if pcs_uq_path not in sys.path:
        sys.path.append(pcs_uq_path)
    from PCS.regression import PCS_UQ, PCS_OOB

def run_single_experiment_pcs(dataset_name, seed_index, n_boot=50, base_data_path=None, alphas=[0.1, 0.05, 0.01], top_k=1, use_oob=False, calibration_method='multiplicative'):
    """
    Run a single PCS ensemble experiment for a dataset using the specified seed index.
    
    Args:
        dataset_name: Name of the dataset to process (e.g., 'data_ailerons' or just 'ailerons')
        seed_index: The index to use for generating the seed (seed = 777 + seed_index)
        n_boot: Number of bootstrap samples
        base_data_path: Base directory path where data is located
        alphas: List of alpha values to use for calibration (default: [0.1, 0.05, 0.01])
        top_k: Number of top-performing models to use (default: 1)
        use_oob: Whether to use PCS_OOB instead of PCS_UQ (default: False)
        calibration_method: Method for calibration, 'multiplicative' or 'additive' (default: 'multiplicative')
    
    Returns:
        Dict containing model results and trained models for each alpha value
    """
    # Create a worker logger to capture logs
    worker_logger = WorkerLogger()
    
    # Normalize base path
    if base_data_path is None:
        base_data_path = os.getcwd()
    
    # Make sure dataset name has 'data_' prefix
    if not dataset_name.startswith("data_"):
        dataset_name = f"data_{dataset_name}"
    
    # List all possible paths where the data might be
    search_paths = [
        os.path.join(base_data_path, dataset_name),
        # Try parent directories
        os.path.join(os.path.dirname(base_data_path), dataset_name),
        os.path.join(os.path.dirname(base_data_path), "data", dataset_name)
    ]
    
    # Find an existing path that contains the dataset
    data_dir = next((path for path in search_paths if os.path.exists(path)), None)
    
    if not data_dir:
        worker_logger.log(f"Searched for dataset in these directories: {search_paths}")
        raise FileNotFoundError(f"Could not find directory for {dataset_name}")
    
    # Now construct paths to the X and y files
    X_path = os.path.join(data_dir, "X.csv")
    y_path = os.path.join(data_dir, "y.csv")
    
    # Verify files exist
    if not os.path.exists(X_path) or not os.path.exists(y_path):
        worker_logger.log(f"Found directory {data_dir} but missing X.csv or y.csv files")
        raise FileNotFoundError(f"Missing data files in {data_dir}")
    
    worker_logger.log(f"Using X data from: {X_path}")
    worker_logger.log(f"Using y data from: {y_path}")
    
    # Read the data
    X = pd.read_csv(X_path)
    y = np.loadtxt(y_path, delimiter=',')
    
    # Calculate seed from index - exactly as in original
    seed = 777 + seed_index
    
    # Train, validation, and test split - exactly as in original
    x_train_val, x_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.25, random_state=seed)
    
    # Create split_dict for storing the data
    split_dict = {
        "x_train": x_train.to_numpy(),
        "y_train": y_train,
        "x_val": x_val.to_numpy(),
        "y_val": y_val,
        "x_test": x_test.to_numpy(),
        "y_test": y_test
    }
    
    estimators = {
    "OLS": LinearRegression(n_jobs = -1),
    "Ridge": RidgeCV(),
    "Lasso": LassoCV(cv = 3, n_jobs = -1),
    "ElasticNet": ElasticNetCV(cv = 3, n_jobs = -1),
    "RandomForest": RandomForestRegressor(min_samples_leaf = 5, max_features = 0.33, n_estimators = 100, random_state = 42, n_jobs = -1),
    "ExtraTrees": ExtraTreesRegressor(min_samples_leaf = 5, max_features = 0.33, n_estimators = 100, random_state = 42, n_jobs = -1),
    "AdaBoost": AdaBoostRegressor(random_state = 42),
    "XGBoost": XGBRegressor(random_state = 42, n_jobs = -1),
    "MLP": MLPRegressor(random_state = 42, hidden_layer_sizes = (64,))
    }

    # Print the models we're using for this run
    worker_logger.log(f"[Seed {seed_index}] Using models: {', '.join(estimators.keys())}")
    
    # Create a dictionary to store results for each alpha
    all_results = {}
    
    # Process each alpha value
    for alpha in alphas:
        worker_logger.log(f"[Seed {seed_index}] Training and calibrating for alpha={alpha} (confidence level: {(1-alpha)*100}%)")
        
        # Create the appropriate PCS object
        if use_oob:
            worker_logger.log(f"[Seed {seed_index}] Using OOB method with top_k={top_k}")
            pcs_model = PCS_OOB(
                models=estimators,
                num_bootstraps=n_boot,
                alpha=alpha,
                seed=seed,
                top_k=top_k,
                calibration_method=calibration_method
            )
            # Fit the model on train+val combined data
            pcs_model.fit(
                X = np.concatenate([split_dict["x_train"], split_dict["x_val"]]), 
                y = np.concatenate([split_dict["y_train"], split_dict["y_val"]])
            )
        else:
            pcs_model = PCS_UQ(
                models=estimators,
                num_bootstraps=n_boot,
                alpha=alpha,
                seed=seed,
                top_k=top_k,
                calibration_method=calibration_method
            )
            pcs_model.fit(X = split_dict["x_train"], y = split_dict["y_train"], X_calib = split_dict["x_val"], y_calib = split_dict["y_val"])
        
        worker_logger.log(f"[Seed {seed_index}] Calibration complete. Gamma value: {pcs_model.gamma}")
        
        # Print model performance rankings to the worker logger
        print_model_performance(pcs_model, worker_logger)

        if use_oob:
            worker_logger.log(f"[Seed {seed_index}] Using OOB method with custom get_intervals")
            
            # Use get_intervals on train+val combined data
            train_intervals_raw = get_intervals_manually(
                pcs_model, 
                # np.concatenate([split_dict["x_train"], split_dict["x_val"]]),
                split_dict["x_train"],
                alpha=alpha
            )
            
            val_intervals_raw = get_intervals_manually(
                pcs_model, 
                split_dict["x_val"],
                alpha=alpha
            )

            # Split the combined results
            # train_size = len(split_dict["x_train"])
            # train_intervals_raw = train_val_intervals_raw[:train_size]
            # val_intervals_raw = train_val_intervals_raw[train_size:]
            
            # Use the same method for test data directly
            test_intervals_raw = get_intervals_manually(pcs_model, split_dict['x_test'], alpha=alpha)
            
            # Get calibrated intervals
            train_intervals = pcs_model.predict(split_dict['x_train'])
            val_intervals = pcs_model.predict(split_dict['x_val'])
            test_intervals = pcs_model.predict(split_dict['x_test'])
            
            worker_logger.log(f"[Seed {seed_index}] OOB intervals and medians calculated using custom method")
        else:
            # For standard PCS_UQ, use the built-in methods
            train_intervals_raw = pcs_model.get_intervals(split_dict['x_train'])
            val_intervals_raw = pcs_model.get_intervals(split_dict['x_val'])
            test_intervals_raw = pcs_model.get_intervals(split_dict['x_test'])
            
            # Get calibrated intervals
            train_intervals = pcs_model.predict(split_dict['x_train'])
            val_intervals = pcs_model.predict(split_dict['x_val'])
            test_intervals = pcs_model.predict(split_dict['x_test'])
            
            worker_logger.log(f"[Seed {seed_index}] Standard intervals and medians calculated")
        
        # Store results
        pcs_ensemble = {
            'x_train': split_dict['x_train'], 
            'y_train': split_dict['y_train'],
            'x_val': split_dict['x_val'], 
            'y_val': split_dict['y_val'],
            'x_test': split_dict['x_test'], 
            'y_test': split_dict['y_test'],
            # Raw intervals for all datasets
            'train_intervals_raw': train_intervals_raw,
            'val_intervals_raw': val_intervals_raw,
            'test_intervals_raw': test_intervals_raw,
            # Calibrated intervals for all datasets
            'train_intervals': train_intervals,
            'val_intervals': val_intervals,
            'test_intervals': test_intervals,
            # Median predictions for all datasets
            # 'train_median_preds': train_median_preds,
            # 'val_median_preds': val_median_preds,
            # 'test_median_preds': test_median_preds,
            'alpha': alpha,
            'confidence_level': f"{(1-alpha)*100:.0f}",
            'gamma': pcs_model.gamma,
            'top_k': top_k,
            'calibration_method': calibration_method,
            'use_oob': use_oob,
            'top_model_names': list(pcs_model.top_k_models.keys()),
            'model_performances': {model: score for model, score in sorted(pcs_model.pred_scores.items(), key=lambda x: x[1], reverse=True)},
        }
        
        all_results[f"alpha_{alpha}"] = pcs_ensemble
    
    # Return both results and logs
    return {
        'results': all_results,
        'logs': worker_logger.get_logs(),
        'seed_index': seed_index
    }

def run_single_experiment_pcs_and_save(dataset_name, seed_index, n_boot=50, base_data_path=None, save_path_template=None, alphas=[0.1, 0.05, 0.01], top_k=1, use_oob=False, calibration_method='multiplicative'):
    """Modified function that runs the experiment and saves result directly to disk"""
    full_result = run_single_experiment_pcs(dataset_name, seed_index, n_boot, base_data_path, alphas, top_k, use_oob, calibration_method)
    results = full_result['results']
    
    # Save results for each alpha to individual files
    if save_path_template:
        for alpha_key, result in results.items():
            alpha = result['alpha']
            confidence_level = result['confidence_level']
            save_path = save_path_template.format(f"{dataset_name}_run_{seed_index}_{confidence_level}")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Convert data to a serializable format before saving
            serializable_result = convert_to_serializable(result)
            with open(save_path, "wb") as f:
                pickle.dump(serializable_result, f)
        
    return full_result

if __name__ == "__main__":
    # Default list of datasets
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
        'data_allstate', # also takes long computationally, but we have maintained it
    ]

    parser = argparse.ArgumentParser(description='Run PCS ensemble experiments')
    parser.add_argument('--datasets', type=str, nargs='?', default=','.join(default_datasets), 
                        help='Comma-separated list of dataset names (default: all datasets)')
    parser.add_argument('--seed_index', type=int, default=0, help='Seed index')
    parser.add_argument('--n_boot', type=int, default=100, help='Number of bootstrap samples')
    parser.add_argument('--n_seeds', type=int, default=10, help='Number of seeds for the experiments')
    parser.add_argument('--n_jobs', type=int, default=5, help='Number of jobs for parallelizing the training')
    parser.add_argument('--project_root', type=str, default=os.path.join("..", ".."), help='Base path for the data')
    parser.add_argument('--alphas', type=str, default='0.05', 
                       help='Comma-separated list of alpha values for calibration (default: 0.1,0.05,0.01)')
    parser.add_argument('--top_k', type=int, default=1, help='Number of top-performing models to use (default: 1)')
    parser.add_argument('--use_oob', action='store_true', help='Use PCS_OOB instead of PCS_UQ')
    parser.add_argument('--calibration_method', type=str, choices=['multiplicative', 'additive'], 
                       default='multiplicative', help='Method for calibration (default: multiplicative)')
    parser.add_argument('--log_file', type=str, default=None, 
                       help='Path to log file (default: logs/pcs_training_<timestamp>.log)')
    args = parser.parse_args()

    # Set up logging
    logger = setup_logging(args.log_file)
    logging.info("=" * 80)
    logging.info("Starting PCS training with the following parameters:")
    logging.info(f"- Datasets: {args.datasets}")
    logging.info(f"- Seed index: {args.seed_index}")
    logging.info(f"- Bootstrap samples: {args.n_boot}")
    logging.info(f"- Jobs: {args.n_jobs}")
    logging.info(f"- Project root: {args.project_root}")
    logging.info(f"- Use OOB: {args.use_oob}")
    logging.info(f"- Calibration method: {args.calibration_method}")
    logging.info(f"- Top k: {args.top_k}")
    logging.info("=" * 80)

    # Parse the datasets from command-line input (or use the default list)
    datasets = args.datasets.split(',')
    
    # Parse alpha values
    alphas = [float(alpha) for alpha in args.alphas.split(',')]
    logging.info(f"Using alpha values: {alphas} (confidence levels: {[(1-alpha)*100 for alpha in alphas]}%)")

    data_base_path = os.path.join(args.project_root, "data")
    model_save_path = os.path.join(args.project_root, "models", f"pcs_top{args.top_k}_pcs_{args.n_seeds}", "{}_pcs_results_{}.pkl")
    # Create results directory if it doesn't exist
    os.makedirs(os.path.dirname(model_save_path.format("", "")), exist_ok=True)

    logging.info("Starting to process datasets...")
    start_time = time.time()
    #---------------------------------------------------
    # Option 1: Process everything at once
    # Process each dataset
    for dataset_name in datasets:
        dataset_name = dataset_name.strip()
        logging.info(f"Processing dataset: {dataset_name}")

        # Special handling for Allstate due to convergence issues
        n_jobs_to_use = 1 if dataset_name == 'data_allstate' else args.n_jobs
        
        all_results = Parallel(n_jobs=n_jobs_to_use)(
            delayed(run_single_experiment_pcs)(
                dataset_name, i, 
                n_boot=args.n_boot, 
                base_data_path=data_base_path, 
                alphas=alphas,
                top_k=args.top_k,
                use_oob=args.use_oob,
                calibration_method=args.calibration_method
            )
            for i in range(args.n_seeds)
        )
        
        # Print worker logs from each seed
        for result in all_results:
            seed_idx = result['seed_index']
            logging.info(f"\n----- Logs from worker process (seed {seed_idx}) -----")
            for log_line in result['logs']:
                logging.info(log_line)
        
        # For each alpha, create a separate results file
        for alpha in alphas:
            confidence_level = f"{(1-alpha)*100:.0f}"
            results_for_alpha = {f'run_{i}': all_results[i]['results'][f"alpha_{alpha}"] for i in range(args.n_seeds)}
            
            # Convert to serializable before saving
            serializable_results = convert_to_serializable(results_for_alpha)
            
            # Save results
            with open(model_save_path.format(dataset_name, confidence_level), "wb") as f:
                pickle.dump(serializable_results, f)
            
            logging.info(f"Results saved for dataset {dataset_name} at confidence level {confidence_level}")

        # end time
        end_time = time.time()
        logging.info(f"Total time taken for {dataset_name}: {end_time - start_time} seconds")

    logging.info("All processing complete!")


# To run the experiments, use the following command (for our paper):
# python .\train_pcs_quantile.py --top_k 1
## Alternatively, for double-logging for each top_k, use the following command:
## python .\train_pcs_quantile.py --top_k 1 *> logs/pcs_training_top1.log ; python .\train_pcs_quantile.py --top_k 2 *> logs/pcs_training_top2.log