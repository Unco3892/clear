import pandas as pd
import numpy as np
import logging
import sys
import os
# Set up logging
def setup_logging(log_file=None, log_level=logging.INFO):
    """Set up logging configuration"""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    # Get the root logger
    logger = logging.getLogger()
    
    # Clear any existing handlers to avoid duplicates
    if logger.handlers:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
    
    # Configure root logger
    logger.setLevel(log_level)
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console_handler)
    
    # If log file is specified, add file handler
    if log_file:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)
        
    return logger

# hijack sys.stdout so that every print goes through the Python logge
class StreamToLogger:
    def __init__(self, logger, level):
        self.logger = logger
        self.level  = level
        self.linebuf = ''
    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())
    def flush(self):
        pass

sys.stdout = StreamToLogger(logging.getLogger(), logging.INFO)

# Function to format metric names nicely
def format_metric_name(metric):
    if metric == "quantile_loss":
        return "Quantile Loss"
    elif metric == "expectile_loss":
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
        return "Lambda"
    elif metric == "gamma":
        return "Gamma"
    elif metric == "mpiw":
        return "MPIW"
    elif metric == "interval_score_loss":
        return "Interval Score Loss"
    else:
        return metric.upper().replace("_", " ")

def reconstruct_dataframe(df_object):
    if isinstance(df_object, dict) and df_object.get("_pandas_dataframe_", False):
        return pd.DataFrame(
            data=df_object["data"],
            columns=df_object["columns"],
            index=df_object["index"]
        )
    return df_object

def safe_flatten(arr):
    if hasattr(arr, 'flatten'):
        return arr.flatten()
    else:
        return np.array(arr).flatten()

def get_top_model_info(run_data):
    """
    Get information about the top-performing model from PCS run data.
    
    Args:
        run_data: Dictionary containing run data
    
    Returns:
        (model_type, model_params, quantile_models)
    """
    logger = logging.getLogger()

    if 'top_model_names' in run_data and len(run_data['top_model_names']) > 0:
        top_model_name = run_data['top_model_names'][0]
        logger.info(f"Found top-performing model in pickle: {top_model_name}")
    else:
        if 'model_performances' in run_data and run_data['model_performances']:
            sorted_models = sorted(run_data['model_performances'].items(), 
                                key=lambda x: x[1], reverse=True)
            top_model_name = sorted_models[0][0]
            logger.info(f"Determined top model from performances: {top_model_name}")
        else:
            top_model_name = "QRF"
            logger.warning(f"Warning: No top model info found in pickle, defaulting to: {top_model_name}")
    
    model_name_map = {
        "QRF": "rf",
        "QXGB": "xgb",
        "ExpectileGAM": "qgam",
        # mean predictors are always mapped to xgb
        "OLS": "rf",
        "Ridge": "rf",
        "Lasso": "rf",
        "ElasticNet": "rf",
        "RandomForest": "rf",
        "ExtraTrees": "rf",
        "AdaBoost": "rf",
        "XGBoost": "rf",
        "MLP": "rf"
    }
    
    if top_model_name in model_name_map:
        model_type = model_name_map[top_model_name]
    else:
        for pcs_name, clear_name in model_name_map.items():
            if pcs_name.lower() == top_model_name.lower():
                model_type = clear_name
                break
        else:
            model_type = "rf"
            logger.warning(f"Warning: Unknown model type {top_model_name}, defaulting to {model_type}")

    print(f"Using top PCS model '{top_model_name}' mapped to CLEAR model type '{model_type}'")

    logger.info(f"Using top PCS model '{top_model_name}' mapped to CLEAR model type '{model_type}'")
    
    if model_type == "rf":
        model_params = {
            "n_estimators": 100,
            "random_state": 777,
            "min_samples_leaf": 10
        }
    elif model_type == "xgb":
        model_params = {
            "n_estimators": 100,
            "tree_method": "hist",
            "random_state": 777,
            "min_child_weight": 10
        }
    elif model_type == "qgam":
        try:
            if 'gam_parameters' in run_data and 'lam' in run_data['gam_parameters']:
                logger.info("Using optimal lambda from PCS: %s", run_data['gam_parameters']['lam'])
                optimal_lam = run_data['gam_parameters']['lam']
                n_splines = run_data['gam_parameters'].get('n_splines', 10)
            else:
                logger.warning("No optimal lambda found in PCS, using default value: 0.6")
                optimal_lam = 0.6
                n_splines = 10
        except Exception as e:
            logger.error(f"Error getting GAM parameters: {e}")
            optimal_lam = 0.6
            n_splines = 10
            
        model_params = {
            "n_splines": n_splines,
            "lam": optimal_lam,
            "spline_order": 3,
            "basis": "ps"
        }
    
    quantile_models = [model_type]
    
    return model_type, model_params, quantile_models