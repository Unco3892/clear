import pickle
import pandas as pd
import numpy as np
import pandas as pd
from pygam import ExpectileGAM
import time
import concurrent.futures
import threading
import logging
import datetime
import os

# Set up logging
def setup_logging(log_file=None):
    """Set up logging to file and console"""
    if log_file is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"logs/pcs_training_{timestamp}.log"
    
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file, mode='a')
    file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(file_format)
    logger.addHandler(console_handler)
    
    return logger

# Helper class to capture outputs from worker processes
class WorkerLogger:
    def __init__(self):
        self.logs = []
    
    def log(self, message):
        self.logs.append(message)
    
    def get_logs(self):
        return self.logs
    
    # Add info method to match logging interface
    def info(self, message):
        self.log(message)

def print_model_performance(pcs_model, logger=None):
    """Print a simple table of model performances and selection status."""
    log_func = logger.info if logger else logging.info
    
    log_func("\n----- Model Performance Rankings -----")
    # Sort models by performance score (higher is better)
    sorted_models = sorted(pcs_model.pred_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Print each model's performance
    for i, (model_name, score) in enumerate(sorted_models):
        selected = i < pcs_model.top_k
        marker = "*" if selected else " "
        log_func(f"{marker} {i+1}. {model_name}: {score:.6f}")
    
    # Print selected models summary
    selected_models = [model_name for i, (model_name, _) in enumerate(sorted_models) if i < pcs_model.top_k]
    log_func(f"\nSelected: {', '.join(selected_models)}")
    log_func(f"Top {pcs_model.top_k} of {len(sorted_models)} available models")

def get_intervals_manually(pcs_model, X, alpha=None):
    """
    Custom implementation of get_intervals that works with any dataset for OOB models.
    This replicates the core functionality of the get_intervals method but bypasses OOB index access.
    `PCS_OOB.get_intervals` only works for the training data and not new data.get_intervals

    Args:
        pcs_model: The fitted PCS model (either PCS_UQ or PCS_OOB)
        X: Features to generate intervals for
        alpha: Alpha value to use (defaults to model's alpha)
    
    Returns:
        intervals: numpy array of shape (n_samples, 3) with [lower, median, upper] bounds
    """
    if alpha is None:
        alpha = pcs_model.alpha
    
    n_samples = X.shape[0]
    
    # Initialize array to store predictions from all bootstrap models
    all_predictions = []
    
    # Collect predictions from all bootstrap models
    for model_name, bootstrap_models in pcs_model.bootstrap_models.items():
        for model in bootstrap_models:
            predictions = model.predict(X)
            all_predictions.append(predictions)
    
    # Convert to numpy array of shape (n_bootstrap_models, n_samples)
    all_predictions = np.array(all_predictions)
    
    # Create intervals array to store [lower, median, upper] for each sample
    intervals = np.zeros((n_samples, 3))
    
    # Calculate quantiles across bootstrap models (axis 0)
    intervals[:, 0] = np.quantile(all_predictions, alpha/2, axis=0)           # Lower bound
    intervals[:, 1] = np.quantile(all_predictions, 0.5, axis=0)               # Median
    intervals[:, 2] = np.quantile(all_predictions, 1.0 - alpha/2, axis=0)     # Upper bound
    
    return intervals

def _run_gam_gridsearch(X, y, n_splines):
    """Helper function to run GAM gridsearch that can be executed with a timeout"""
    try:
        gam50 = ExpectileGAM(expectile=0.5, n_splines=n_splines).gridsearch(X, y, progress=False)
        optimal_lam = gam50.lam
        
        # Handle the case when optimal_lam is a list
        if isinstance(optimal_lam, list):
            # For simplicity, use the first/average value
            optimal_lam = optimal_lam[0] if len(optimal_lam) == 1 else np.mean(optimal_lam)
        
        return optimal_lam
    except Exception as e:
        return f"error: {str(e)}"

def optimize_gam_smoothing(X, y, n_splines=10, timeout=300):
    """
    Find the optimal smoothing parameter for ExpectileGAM using grid search.
    
    Args:
        X: Feature matrix
        y: Target values
        n_splines: Number of splines to use (default: 10)
        timeout: Maximum time in seconds to wait for GAM optimization (default: 300 = 5 minutes)
    
    Returns:
        optimal_lam: Optimal smoothing parameter
        convergence_status: True if converged within timeout, False otherwise
    """
    print(f"Finding optimal smoothing parameter for ExpectileGAM via grid search (timeout: {timeout}s)...")
    
    # Use ThreadPoolExecutor to implement timeout which is more compatible with other parallel processing like joblib
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    
    try:
        start_time = time.time()
        future = executor.submit(_run_gam_gridsearch, X, y, n_splines)
        
        try:
            # Wait for the result with timeout
            result = future.result(timeout=timeout)
            
            # Check if there was an error
            if isinstance(result, str) and result.startswith("error"):
                print(f"Error during GAM smoothing optimization: {result}")
                print("Using default smoothing parameter (lam=0.01)")
                return 0.01, False
            else:
                optimal_lam = result
                print(f"Optimal smoothing parameter (lam): {optimal_lam:.4f} (found in {time.time() - start_time:.1f}s)")
                return optimal_lam, True
                
        except concurrent.futures.TimeoutError:
            print(f"GAM optimization exceeded timeout of {timeout} seconds - ExpectileGAM will be excluded")
            print("Using default smoothing parameter (lam=0.01) but GAM will NOT be included in models")
            return 0.01, False
    finally:
        # Make sure to shut down the executor properly to avoid zombie threads
        executor.shutdown(wait=False)

def convert_to_serializable(data):
    """
    Convert data to a format that's safe for pickling, especially for NumPy arrays.
    This helps avoid version compatibility issues.
    """
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, pd.DataFrame):
        # Convert DataFrame to dict but keep structure for reconstruction
        return {
            "_pandas_dataframe_": True,
            "columns": data.columns.tolist(),
            "index": data.index.tolist(),
            "data": data.values.tolist()
        }
    elif isinstance(data, dict):
        return {k: convert_to_serializable(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_to_serializable(item) for item in data]
    else:
        return data

def load_results_safely(file_path):
    """
    Load pickled results safely, converting serialized data back to original format.
    This handles numpy array version incompatibilities.
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    def reconstruct_data(obj):
        if isinstance(obj, dict) and obj.get("_pandas_dataframe_", False):
            # Reconstruct DataFrame
            return pd.DataFrame(
                data=obj["data"],
                columns=obj["columns"],
                index=obj["index"]
            )
        elif isinstance(obj, dict):
            return {k: reconstruct_data(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [reconstruct_data(item) for item in list]
        else:
            return obj
    
    return reconstruct_data(data)