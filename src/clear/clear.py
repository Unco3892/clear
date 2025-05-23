"""
CLEAR Framework: Calibrated Learning for Epistemic and Aleatoric Risk.
Implementation of the CLEAR method as described in the paper.
"""

import numpy as np
from xgboost import XGBRegressor
from quantile_forest import ExtraTreesQuantileRegressor, RandomForestQuantileRegressor
# alternative implementation that yields the same results
from sklearn.linear_model import QuantileRegressor
from pygam import ExpectileGAM, LinearGAM
# from pygam import s, GAM
from tqdm import tqdm
from sklearn.utils.validation import check_is_fitted
import joblib
import inspect
import matplotlib.pyplot as plt
import warnings
import os
# --- Start of conditional sys.path modification ---
if __name__ == '__main__' and __package__ is None:
    import sys
    import os
    # Get the directory of the current script (e.g., /path/to/src/clear)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Get the parent directory (e.g., /path/to/src)
    parent_dir = os.path.dirname(current_dir)
    # Add the parent directory to sys.path if it's not already there
    # This allows imports like 'from clear.utils import ...'
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
# --- End of conditional sys.path modification ---#
from clear.utils import compute_quantile_ensemble_bounds
from clear.models import SimultaneousQuantileRegressor, QuantileGAM

class CLEAR:
    """
    CLEAR: Calibrated Learning for Epistemic and Aleatoric Risk
    
    This class implements the CLEAR framework as described in the paper, which combines
    epistemic uncertainty and aleatoric uncertainty with adaptive weighting.
    
    The prediction interval is formed as:
    [f̂ - γ₁×(q̂₀.₅ᵃˡᵉ - q̂₀.₀₅ᵃˡᵉ) - γ₂×(f̂ - q̂₀.₀₅ᵉᵖⁱ), 
     f̂ + γ₁×(q̂₀.₉₅ᵃˡᵉ - q̂₀.₅ᵃˡᵉ) + γ₂×(q̂₀.₉₅ᵉᵖⁱ - f̂)]
     
    where γ₂ = λ×γ₁, and λ is chosen to optimize pinball loss on validation data.
    """
    
    def __init__(self, desired_coverage=0.9, lambdas=None, n_bootstraps=50, n_models=None, random_state=None, n_jobs=-1, fixed_gamma=None, fixed_lambda=None):
        """
        Initialize the CLEAR model.
        
        Args:
            desired_coverage: Target coverage probability (default: 0.9)
            lambdas: List of lambda values to try for balancing aleatoric vs epistemic uncertainty.
                     If None, a default range will be used.
            n_bootstraps: Number of bootstrap samples for uncertainty estimation (default: 50)
            n_models: Number of top performing models to use (default: None = use all)
            random_state: Seed for random number generation to ensure reproducible results
            n_jobs: Number of jobs for parallel processing (default: -1 = use all cores)
            fixed_gamma: Fixed value for gamma to use (if None, gamma will be optimized)
            fixed_lambda: Fixed value for lambda to use (if None, lambda will be optimized)
        """
        self.desired_coverage = desired_coverage
        self.alpha = 1 - desired_coverage  # typically 0.1 for 90% coverage
        self.n_bootstraps = n_bootstraps
        self.n_models = n_models
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.fixed_gamma = fixed_gamma
        self.fixed_lambda = fixed_lambda
        
        # Add new attributes for storing uncertainty metrics
        self.total_aleatoric_calib = None
        self.total_epistemic_calib = None
        self.uncertainty_ratio_calib = None
        
        # Set the random seed if provided
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Default lambdas: Log-spaced grid focusing more on smaller values,
        # but still covering a wide range. Includes 0 explicitly.
        if lambdas is None:
            # Default grid: 0, then logspace from 0.01 to 100 (e.g., 50 points)
            #log_lambdas = np.logspace(-2, 2, 50) # From 0.01 to 100
            #self.lambdas = np.concatenate(([0], log_lambdas))
            self.lambdas = np.concatenate((np.linspace(0, 0.09, 10), np.logspace(-1, 2, 4001)))
        else:
            self.lambdas = np.sort(np.unique(lambdas)) # Ensure sorted unique values if provided
        
        # Initialize model parameters
        self.optimal_lambda = None
        self.gamma = None
        
        # Quantile levels based on desired coverage
        # This follows the convention of alpha=0.1 gives 0.05, 0.95 quantiles
        self.lower_quantile = self.alpha / 2
        self.upper_quantile = 1 - self.alpha / 2
        
        # Flag to track if aleatoric models were fit on residuals
        self.fit_on_residuals = False
    
    def _initialize_aleatoric_model(self, quantile_model, quantile, model_params):
        # Ensure model_params is a dict
        if model_params is None:
            model_params = {}
        else:
            model_params = model_params.copy()

        if isinstance(quantile_model, str):
            qmodel_str = quantile_model.lower()
            if qmodel_str in ['qgam', 'gam']:
                defaults = {'n_splines': 10, 'lam': 0.6}
                defaults.update(model_params)
                # Use pygam.ExpectileGAM, mapping quantile to expectile
                # Ensure required params like spline_order, basis are in defaults
                gam_params = {'expectile': quantile} # Use quantile as expectile
                gam_params.update(defaults)
                # Add random_state if ExpectileGAM supports it (it doesn't directly in __init__)
                # if self.random_state is not None:
                #     # Check signature if needed, but pygam GAMs usually handle randomness internally
                #     pass 
                return ExpectileGAM(**gam_params)
            elif qmodel_str in ['quantileregressor', 'linear']:
                quantile_model = QuantileRegressor
                defaults = {'alpha': 0, 'solver': 'highs', 'quantile': quantile}
            elif qmodel_str in ['xgb', 'qxgb', 'xgboost']:
                quantile_model = XGBRegressor
                # Use same n_estimators as PCS in train_pcs_quantile (100 trees)
                defaults = {'objective': 'reg:quantileerror', 'n_estimators': 100, 'tree_method': 'hist', 'quantile_alpha': quantile, 'min_child_weight': 10, 'n_jobs': -1}
                if self.random_state is not None:
                    defaults['random_state'] = self.random_state
            elif qmodel_str in ['rf', 'qrf', 'randomforestquantileregressor']:
                quantile_model = RandomForestQuantileRegressor
                # Match PCS default RF quantile settings: 100 trees, same quantile level
                defaults = {'n_estimators': 100, 'default_quantiles': quantile, 'n_jobs': -1, 'min_samples_leaf' : 10}
                if self.random_state is not None:
                    defaults['random_state'] = self.random_state
            elif qmodel_str in ['extratrees', 'qextratrees', 'extratreesquantileregressor']:
                quantile_model = ExtraTreesQuantileRegressor
                defaults = {'n_estimators': 100, 'default_quantiles': quantile, 'n_jobs': 1, 'min_samples_leaf' : 10}
                if self.random_state is not None:
                    defaults['random_state'] = self.random_state
            elif qmodel_str in ['sqr', 'simultaneousquantileregressor', 'neuralnet']:
                quantile_model = SimultaneousQuantileRegressor
                defaults = {'hidden_layers': [128, 128], 'learning_rate': 1e-3, 'weight_decay': 1e-4,
                            'n_epochs': 2000, 'batch_size': 64, 'quantile': None, 'random_state': self.random_state}
            else:
                raise ValueError(f"Unsupported string model type: {quantile_model}. Supported types: 'linear'/'quantileregressor', 'xgb'/'xgboost', 'rf'/'qrf', 'extratrees'/'qextratrees', 'gam'/'qgam'")
        else:
            defaults = {'quantile': quantile}
            sig = inspect.signature(quantile_model.__init__)
            if 'random_state' in sig.parameters and self.random_state is not None:
                defaults['random_state'] = self.random_state

        # Merge provided parameters over the defaults
        defaults.update(model_params)
        if quantile_model == QuantileRegressor:
            defaults.pop('random_state', None)
        return quantile_model(**defaults)

    def _fit_single_bootstrap(self, X, y, quantile_model, model_params, bootstrap_idx, median_preds=None):
        """
        Fit lower, median, and upper quantile models on a bootstrap sample.
        
        Args:
            X: Feature matrix
            y: Target values (or pre-computed residuals when fit_on_residuals=True)
            quantile_model: Quantile regression model
            model_params: Parameters for the model
            bootstrap_idx: Index of the bootstrap sample (used for reproducibility)
            median_preds: Optional median predictions for residual-based fitting
        """
        n_samples = len(y)
        
        # Set a unique random seed for this bootstrap iteration to ensure reproducibility
        # while still getting different bootstrap samples for each iteration
        if self.random_state is not None:
            bootstrap_seed = self.random_state + bootstrap_idx
            np.random.seed(bootstrap_seed)
        
        bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        X_bootstrap = X[bootstrap_indices]
        y_bootstrap = y[bootstrap_indices]
        
        # If fitting on residuals and median predictions are provided
        if self.fit_on_residuals and median_preds is not None:
            # Add median predictions as a feature
            median_bootstrap = median_preds[bootstrap_indices]
            
            # Create augmented feature matrix with median predictions as the last column
            X_bootstrap = np.column_stack((X_bootstrap, median_bootstrap))
            
        # Fit models for each either quantile level on the original targets or the residuals with augmented features
        lower_model = self._initialize_aleatoric_model(quantile_model, self.lower_quantile, model_params)
        median_model = self._initialize_aleatoric_model(quantile_model, 0.5, model_params)
        upper_model = self._initialize_aleatoric_model(quantile_model, self.upper_quantile, model_params)
        
        lower_model.fit(X_bootstrap, y_bootstrap)
        median_model.fit(X_bootstrap, y_bootstrap)
        upper_model.fit(X_bootstrap, y_bootstrap)
        
        return lower_model, median_model, upper_model

    def fit_aleatoric(self, X, y, quantile_model="rf", model_params=None, fit_on_residuals=False, epistemic_preds=None):
        """
        Fit aleatoric uncertainty estimators using bootstrapping with the provided quantile model(s).
        
        Args:
            X: Feature matrix
            y: Target values
            quantile_model: A scikit-learn style quantile regression model class, string, or list of such
                            Default is "rf" which uses RandomForestQuantileRegressor. Note that only QXGB, QRF, and QGAM have been fully tested and supported.
            model_params: Optional dictionary of parameters for the quantile model.
            fit_on_residuals: Whether to fit aleatoric models on residuals instead of raw targets.
                              If True, epistemic_preds must be provided or epistemic models must be fitted.
            epistemic_preds: Optional pre-computed epistemic predictions to use for residual calculation.
                              If None and fit_on_residuals is True, uses median of epistemic_models predictions.
        
        Returns:
            self, with fitted bootstrap models for each quantile level.
        """
        X = np.asarray(X)
        y = np.asarray(y).flatten()
        
        # Store the fitting mode for future reference
        self.fit_on_residuals = fit_on_residuals
        
        # Prepare epistemic predictions for residual-based fitting if needed
        median_preds = None
        if fit_on_residuals:
            if epistemic_preds is not None:
                # Use provided epistemic predictions
                print("Using provided epistemic predictions for residual-based aleatoric modeling")
                median_preds = np.asarray(epistemic_preds).flatten()
            elif hasattr(self, 'epistemic_models') and len(self.epistemic_models) > 0:
                # Generate predictions from fitted epistemic models
                print("Generating epistemic predictions from fitted models for residual-based aleatoric modeling")
                all_preds = [model.predict(X) for model in self.epistemic_models]
                median_preds = np.median(all_preds, axis=0)
            else:
                raise ValueError("For residual-based fitting, either epistemic_preds must be provided or epistemic models must be fitted first")
            
            # Pre-compute residuals to use as target values
            residuals = y - median_preds
            
            # Print some diagnostic information about residuals
            print(f"Residual statistics - Mean: {np.mean(residuals):.6f}, Std: {np.std(residuals):.6f}")
            print(f"Min residual: {np.min(residuals):.6f}, Max residual: {np.max(residuals):.6f}")
            
            # For residual-based fitting, use residuals as the target values
            y = residuals
        
        # Find optimal lam via grid search if using GAM models
        if isinstance(quantile_model, str) and quantile_model.lower() in ['qgam', 'gam']:
            if model_params is None or 'lam' not in model_params:
                print("Finding optimal smoothing parameter via grid search...")
                gam50 = ExpectileGAM(expectile=0.5).gridsearch(X, y)
                optimal_lam = gam50.lam

                # Add this section to handle the case when optimal_lam is a list
                if isinstance(optimal_lam, list):
                    print(f"Multiple smoothing parameters found: {optimal_lam}")
                    # For simplicity, use the first/average value
                    optimal_lam = optimal_lam[0] if len(optimal_lam) == 1 else np.mean(optimal_lam)
                    print(f"Using smoothing parameter: {optimal_lam:.4f}")
                else:
                    print(f"Optimal smoothing parameter (lam): {optimal_lam:.4f}")
                
                if model_params is None:
                    model_params = {'n_splines': 10, 'lam': optimal_lam}
                else:
                    model_params = model_params.copy()
                    model_params['lam'] = optimal_lam
        
        # Support multiple quantile models (e.g., QRF & QXGB)
        if isinstance(quantile_model, (list, tuple)):
            model_list = list(quantile_model)
        else:
            model_list = [quantile_model]
        # Prepare parameter list for each quantile model
        if isinstance(model_params, list):
            if len(model_params) != len(model_list):
                raise ValueError("Length of model_params list must match number of quantile models")
            params_list = model_params
        else:
            params_list = [model_params] * len(model_list)
        # Record number of base aleatoric model types
        self.n_aleatoric_model_types = len(model_list)
        
        # Initialize model storage
        self.lower_models = []
        self.median_models = []
        self.upper_models = []
        
        # Use adaptive parallelism with job reduction for resource management
        print(f"Fitting aleatoric models with models: {[str(m) for m in model_list]}..." + 
              (" on residuals" if fit_on_residuals else ""))
        
        # Start from the given number of jobs and go down from there
        effective_n_jobs = self.n_jobs

        # Try fitting with reducing number of jobs until success
        max_attempts = 3  # Maximum number of reduction attempts
        for attempt in range(max_attempts):
            try:
                if attempt > 0:
                    # Reduce jobs by a factor of 4 on each attempt
                    effective_n_jobs = max(1, self.n_jobs // 1.2)
                    print(f"Retry attempt {attempt}: Reducing to {effective_n_jobs} parallel jobs")
                
                # Fit each quantile model type across all bootstrap samples
                results = joblib.Parallel(n_jobs=effective_n_jobs, verbose=5, backend='loky', max_nbytes='100M')(
                    joblib.delayed(self._fit_single_bootstrap)(X, y, model, params, i, median_preds)
                    for model, params in zip(model_list, params_list)
                    for i in range(self.n_bootstraps)
                )
                
                # If we get here, processing succeeded
                print(f"Parallel processing successful with {effective_n_jobs} jobs")
                break
                
            except (OSError, MemoryError) as e:
                print(f"Warning: Parallel processing failed with error: {str(e)}")
                
                if attempt == max_attempts - 1:
                    # Last attempt failed, raise the error
                    raise RuntimeError(f"Parallel processing failed after {max_attempts} attempts with reduced jobs. Please try with fewer bootstraps or manually set n_jobs=1.") from e

        # Unpack results of bootstrap fitting
        self.lower_models, self.median_models, self.upper_models = map(list, zip(*results))
        # Record total number of aleatoric bootstrap models
        self.total_aleatoric_models = len(self.lower_models)
        print(f"Fitted {self.total_aleatoric_models} bootstrap models across {self.n_aleatoric_model_types} aleatoric model types and {self.n_bootstraps} bootstraps using {effective_n_jobs} jobs")
        return self

    def predict_aleatoric(self, X, epistemic_preds=None, return_samples=False):
        """
        Predict aleatoric uncertainty bounds by aggregating bootstrap predictions.
        
        Args:
            X: Feature matrix
            epistemic_preds: Optional epistemic predictions for residual-based prediction
            return_samples: If True, return raw bootstrap sample predictions
            
        Returns:
            If return_samples is False:
                median_pred: Median predictions
                lower_pred: Lower bound predictions
                upper_pred: Upper bound predictions
            If return_samples is True:
                all_median_preds: Raw median predictions array (n_models, n_samples)
                all_lower_preds: Raw lower quantile predictions array (n_models, n_samples)
                all_upper_preds: Raw upper quantile predictions array (n_models, n_samples)
        """
        if not hasattr(self, 'lower_models') or len(self.lower_models) == 0:
            raise ValueError("Aleatoric models not fitted. Call fit_aleatoric first.")
        
        X = np.asarray(X)
        n_samples = X.shape[0]
        
        # Prepare input for prediction
        if self.fit_on_residuals:
            if epistemic_preds is None:
                # If no epistemic predictions provided, try to generate them
                if hasattr(self, 'epistemic_models') and len(self.epistemic_models) > 0:
                    # Generate predictions from fitted epistemic models
                    all_preds = [model.predict(X) for model in self.epistemic_models]
                    epistemic_preds = np.median(all_preds, axis=0)
                else:
                    raise ValueError("For residual-based prediction, epistemic_preds must be provided or epistemic models must be fitted")
            
            # Augment features with epistemic predictions
            epistemic_preds = np.asarray(epistemic_preds).flatten()
            X_aug = np.column_stack((X, epistemic_preds))
        else:
            # Use original features
            X_aug = X
        
        # Initialize arrays to store all predictions from fitted models
        n_models = len(self.lower_models)
        all_lower_preds = np.zeros((n_models, n_samples))
        all_median_preds = np.zeros((n_models, n_samples))
        all_upper_preds = np.zeros((n_models, n_samples))
        
        # Get predictions from each fitted aleatoric model
        for i in range(n_models):
            check_is_fitted(self.lower_models[i])
            check_is_fitted(self.median_models[i])
            check_is_fitted(self.upper_models[i])
            
            try:
                # Try normal prediction - might use parallel processing internally
                all_lower_preds[i] = self.lower_models[i].predict(X_aug)
                all_median_preds[i] = self.median_models[i].predict(X_aug)
                all_upper_preds[i] = self.upper_models[i].predict(X_aug)
            except TypeError as e:
                # Check if it's the NoneType context manager error
                if "NoneType" in str(e) and "context manager" in str(e):
                    print(f"Warning: Parallel processing error detected. Using sequential prediction for model {i}.")
                    
                    # For quantile forest models with the joblib error, try setting n_jobs=1
                    if hasattr(self.lower_models[i], "n_jobs"):
                        original_n_jobs = self.lower_models[i].n_jobs
                        try:
                            self.lower_models[i].n_jobs = 1
                            self.median_models[i].n_jobs = 1
                            self.upper_models[i].n_jobs = 1
                            
                            all_lower_preds[i] = self.lower_models[i].predict(X_aug)
                            all_median_preds[i] = self.median_models[i].predict(X_aug)
                            all_upper_preds[i] = self.upper_models[i].predict(X_aug)
                            
                            # Restore original settings
                            self.lower_models[i].n_jobs = original_n_jobs
                            self.median_models[i].n_jobs = original_n_jobs
                            self.upper_models[i].n_jobs = original_n_jobs
                        except:
                            # If that still fails, resort to a more direct looping approach
                            print(f"Model {i}: Sequential prediction with n_jobs=1 failed, using manual approach.")
                            raise
                    else:
                        # If we can't set n_jobs, re-raise the error
                        raise
                else:
                    # For other errors, re-raise
                    raise
        
        # If fitting was done on residuals, adjust the predictions
        if self.fit_on_residuals and epistemic_preds is not None:
            # Adjust predictions by adding back the epistemic predictions
            for i in range(n_models):
                all_lower_preds[i] = all_lower_preds[i] + epistemic_preds
                all_median_preds[i] = all_median_preds[i] + epistemic_preds
                all_upper_preds[i] = all_upper_preds[i] + epistemic_preds
        
        # Replace each upper bound with upper bound and median and the same for lower bound (mins)
        all_upper_preds = np.maximum(all_upper_preds, all_median_preds)
        all_lower_preds = np.minimum(all_lower_preds, all_median_preds)

        # If requested, return raw bootstrap sample predictions
        # print(f"[predict_aleatoric] return_samples shapes: median_preds {all_median_preds.shape} (n_models, n_samples), lower_preds {all_lower_preds.shape} (n_models, n_samples), upper_preds {all_upper_preds.shape} (n_models, n_samples)")

        if return_samples:
            return all_median_preds, all_lower_preds, all_upper_preds

        # Aggregate bootstrap predictions using appropriate quantiles (median for each)
        lower_pred = np.median(all_lower_preds, axis=0)
        median_pred = np.median(all_median_preds, axis=0)
        upper_pred = np.median(all_upper_preds, axis=0)
        
        # print(f"[predict_aleatoric] aggregated shapes: median_pred {median_pred.shape} (n_samples,), lower_pred {lower_pred.shape} (n_samples,), upper_pred {upper_pred.shape} (n_samples,)")
        return median_pred, lower_pred, upper_pred

    def _calculate_pinball_loss(self, y_calib, lower_bound, upper_bound):
        """
        Calculate pinball loss for interval predictions.
        
        Args:
            y_calib: Calibration target values
            lower_bound: Lower prediction bounds
            upper_bound: Upper prediction bounds
            
        Returns:
            Average pinball loss
        """
        tau_L = self.alpha / 2
        tau_U = 1 - self.alpha / 2
        
        loss_lower = np.where(
            y_calib < lower_bound,
            (1 - tau_L) * (lower_bound - y_calib),
            tau_L * (y_calib - lower_bound)
        )
        
        loss_upper = np.where(
            y_calib < upper_bound,
            (1 - tau_U) * (upper_bound - y_calib),
            tau_U * (y_calib - upper_bound)
        )
        
        return np.mean(loss_lower + loss_upper)/2  # Divide by 2 to match original implementation

    def _compute_combined_devs(self, lambda_val, aleatoric_left, aleatoric_right, epistemic_left, epistemic_right):
        """Helper function to compute combined deviations based on lambda and pythagoras flag."""
        if self.pythagoras:
            left_devs = np.sqrt(aleatoric_left**2 + (lambda_val * epistemic_left)**2)
            right_devs = np.sqrt(aleatoric_right**2 + (lambda_val * epistemic_right)**2)
        else:
            # This follows the formula in the paper:
            # l(x) = (q̂₀.₅ᵃˡᵉ - q̂₀.₀₅ᵃˡᵉ) + λ(f̂ - q̂₀.₀₅ᵉᵖⁱ)
            # u(x) = (q̂₀.₉₅ᵃˡᵉ - q̂₀.₅ᵃˡᵉ) + λ(q̂₀.₉₅ᵉᵖⁱ - f̂)
            left_devs = aleatoric_left + lambda_val * epistemic_left
            right_devs = aleatoric_right + lambda_val * epistemic_right
        return left_devs, right_devs

    def _calculate_conformal_scores(self, y_calib, median_epistemic, left_dev, right_dev):
        """
        Calculate conformal scores for calibration.
        
        Args:
            y_calib: Calibration target values
            median_epistemic: Epistemic median predictions
            left_dev: Left deviations (combined aleatoric and epistemic)
            right_dev: Right deviations (combined aleatoric and epistemic)
            
        Returns:
            numpy array of scores
        """
        epsilon = 1e-8  # Small value to avoid division by zero
        left_devs_nonzero = np.maximum(left_dev, epsilon)
        right_devs_nonzero = np.maximum(right_dev, epsilon)
        scores = np.maximum((median_epistemic - y_calib) / left_devs_nonzero,
                            (y_calib - median_epistemic) / right_devs_nonzero)
        return scores
        
    def _calculate_fixed_gamma_scores(self, y_calib, median_epistemic, aleatoric_left, aleatoric_right, epistemic_left, epistemic_right, epsilon=1e-10):
        """Calculate scores when gamma is fixed (e.g., gamma=1 specific scoring)."""
        scores = []
        for i in range(len(y_calib)):
            left_dev = max(epistemic_left[i], epsilon)
            right_dev = max(epistemic_right[i], epsilon)
            
            score = max(
                (median_epistemic[i] - aleatoric_left[i] - y_calib[i]) / left_dev,
                (y_calib[i] - (median_epistemic[i] + aleatoric_right[i])) / right_dev
            )
            scores.append(score)
        return np.array(scores)

    def calibrate(self, y_calib, median_epistemic, aleatoric_median, aleatoric_lower, aleatoric_upper, epistemic_lower, epistemic_upper, pythagoras=False, plot=False, verbose=True):
        """
        Calibrate the prediction intervals by finding optimal lambda and gamma parameters.

        Args:
            y_calib: Calibration set target values
            median_epistemic: Central predictions from the epistemic component
            aleatoric_median: Central predictions from aleatoric quantile regression (i.e., prediction_quant0.5)
            aleatoric_lower: Lower aleatoric bounds (q̂₀.₀₅ᵃˡᵉ)
            aleatoric_upper: Upper aleatoric bounds (q̂₀.₉₅ᵃˡᵉ)
            epistemic_lower: Lower epistemic bounds (q̂₀.₀₅ᵉᵖⁱ)
            epistemic_upper: Upper epistemic bounds (q̂₀.₉₅ᵉᵖⁱ)
            pythagoras: Whether to use Pythagorean combination of uncertainties (default: False)
            plot: Whether to plot lambda-loss curve (default: False)
            verbose: Whether to print diagnostic information (default: True)

        Returns:
            self: The calibrated model
        """
        self.pythagoras = pythagoras

        # Ensure inputs are numpy arrays
        y_calib = np.asarray(y_calib).flatten()
        median_epistemic = np.asarray(median_epistemic).flatten()
        aleatoric_median = np.asarray(aleatoric_median).flatten()
        aleatoric_lower = np.asarray(aleatoric_lower).flatten()
        aleatoric_upper = np.asarray(aleatoric_upper).flatten()
        epistemic_lower = np.asarray(epistemic_lower).flatten()
        epistemic_upper = np.asarray(epistemic_upper).flatten()

        # Raise warnings instead of errors in case of crossing quantiles
        if np.any(aleatoric_lower > aleatoric_median) or np.any(aleatoric_upper < aleatoric_median):
            warnings.warn("Aleatoric quantiles are crossing - lower quantile should be <= median <= upper quantile")
        if np.any(epistemic_lower > median_epistemic) or np.any(epistemic_upper < median_epistemic):
            warnings.warn("Epistemic quantiles are crossing - lower quantile should be <= median <= upper quantile")

        # Apply proper quantile ordering - ensuring quantiles don't cross
        aleatoric_upper = np.maximum(aleatoric_upper, aleatoric_median)
        aleatoric_lower = np.minimum(aleatoric_lower, aleatoric_median)
        epistemic_upper = np.maximum(epistemic_upper, median_epistemic)
        epistemic_lower = np.minimum(epistemic_lower, median_epistemic)

        # Calculate left and right deviations for aleatoric uncertainty
        # Following the paper formula:
        # (q̂₀.₅ᵃˡᵉ - q̂₀.₀₅ᵃˡᵉ) and (q̂₀.₉₅ᵃˡᵉ - q̂₀.₅ᵃˡᵉ)
        aleatoric_left = aleatoric_median - aleatoric_lower
        aleatoric_right = aleatoric_upper - aleatoric_median

        # Calculate left and right deviations for epistemic uncertainty
        # (f̂ - q̂₀.₀₅ᵉᵖⁱ) and (q̂₀.₉₅ᵉᵖⁱ - f̂)
        epistemic_left = median_epistemic - epistemic_lower
        epistemic_right = epistemic_upper - median_epistemic

        if verbose:
          # Print diagnostic information about average magnitudes
          print("\n===== UNCERTAINTY MAGNITUDE DIAGNOSTICS =====")
          print(f"Dataset size: {len(y_calib)} samples")
          print("Average magnitudes of uncertainty components before weighting:")
          print(f"  Aleatoric left: {np.mean(aleatoric_left):.6f}")
          print(f"  Aleatoric right: {np.mean(aleatoric_right):.6f}")
          print(f"  Epistemic left: {np.mean(epistemic_left):.6f}")
          print(f"  Epistemic right: {np.mean(epistemic_right):.6f}")
          print(f"  Total aleatoric: {np.mean(aleatoric_left + aleatoric_right):.6f}")
          print(f"  Total epistemic: {np.mean(epistemic_left + epistemic_right):.6f}")
          print(f"  Ratio (epistemic/aleatoric): {np.mean(epistemic_left + epistemic_right) / np.mean(aleatoric_left + aleatoric_right):.6f}")

          # Print median values as well
          print("\nMedian magnitudes of uncertainty components:")
          print(f"  Aleatoric left: {np.median(aleatoric_left):.6f}")
          print(f"  Aleatoric right: {np.median(aleatoric_right):.6f}")
          print(f"  Epistemic left: {np.median(epistemic_left):.6f}")
          print(f"  Epistemic right: {np.median(epistemic_right):.6f}")

          # Check for zero or very small values
          small_threshold = 1e-6
          small_aleatoric = np.mean(np.logical_or(aleatoric_left < small_threshold, aleatoric_right < small_threshold))
          small_epistemic = np.mean(np.logical_or(epistemic_left < small_threshold, epistemic_right < small_threshold))
          print(f"\nPercentage of samples with very small uncertainty (<{small_threshold}):")
          print(f"  Aleatoric: {small_aleatoric * 100:.2f}%")
          print(f"  Epistemic: {small_epistemic * 100:.2f}%")
          print("===========================================\n")

        # Compute calibration adjustment factor for conformal prediction
        n_calib = len(y_calib)
        
        # Calculate adjusted conformal level for valid coverage
        adj_factor = 1.0 + (1.0 / n_calib)
        target_coverage = 1.0 - self.alpha
        conf_level = min(target_coverage * adj_factor, 1.0)
        
        # Store for reporting
        self.conformal_level = conf_level
        self.conformal_adjustment = adj_factor

        # Handle different calibration approaches based on fixed parameters
        calibration_mode = None
        if self.fixed_gamma is not None and self.fixed_lambda is not None:
            calibration_mode = "both_fixed"
            print(f"Using fixed gamma={self.fixed_gamma:.4f} and lambda={self.fixed_lambda:.4f}")
            self.gamma = self.fixed_gamma
            self.optimal_lambda = self.fixed_lambda
        elif self.fixed_gamma is not None:
            calibration_mode = "gamma_fixed"
            print(f"Using fixed gamma={self.fixed_gamma:.4f}, optimizing lambda")
            self.gamma = self.fixed_gamma
        elif self.fixed_lambda is not None:
            calibration_mode = "lambda_fixed"
            print(f"Using fixed lambda={self.fixed_lambda:.4f}, optimizing gamma")
            self.optimal_lambda = self.fixed_lambda
        else:
            calibration_mode = "optimize_both"
            print(f"Optimizing both gamma and lambda")
            if verbose:
                print(f"Calibrating with {len(self.lambdas)} lambda values from {min(self.lambdas):.4f} to {max(self.lambdas):.4f}")
        
        if calibration_mode == "both_fixed":
            # Both gamma and lambda are fixed - just calculate expected coverage
            left_devs, right_devs = self._compute_combined_devs(self.optimal_lambda, aleatoric_left, aleatoric_right, epistemic_left, epistemic_right)
            lower_bound = median_epistemic - self.gamma * left_devs
            upper_bound = median_epistemic + self.gamma * right_devs
            
            coverage = np.mean((y_calib >= lower_bound) & (y_calib <= upper_bound))
            print(f"Expected empirical coverage on calibration set: {coverage:.4f} (target: {self.desired_coverage:.4f})")

        elif calibration_mode == "gamma_fixed":
            # Find optimal lambda using conformal prediction (coverage-based approach)
            # This replaces the pinball loss approach for consistency with how other methods handle fixed gamma
            # UACQR_S approach: special calculation of lambda
            # This uses a specific conformal scoring formula:
            # score = max((f_hat - aleatoric_left - y) / epistemic_left, (y - (f_hat + aleatoric_right)) / epistemic_right)
            # The key differences from regular conformal scores:
            # 1. Normalization is by epistemic uncertainty only (not combined)
            # 2. Aleatoric uncertainty directly shifts the median prediction
            # 3. Lambda is calibrated to achieve desired coverage with this scoring method

            # Check for unsupported combination of fixed gamma with Pythagoras
            if pythagoras:
                raise NotImplementedError("The combination of fixed_gamma with pythagoras=True is not implemented")
            
            # Calculate scores using the specific formula for fixed gamma
            scores = self._calculate_fixed_gamma_scores(y_calib, median_epistemic, self.fixed_gamma*aleatoric_left, self.fixed_gamma*aleatoric_right, self.fixed_gamma*epistemic_left, self.fixed_gamma*epistemic_right)
    
            # Get optimal lambda from quantile of scores
            self.optimal_lambda = np.quantile(scores, conf_level, method='higher')
            
            print(f"Optimal lambda with fixed gamma={self.gamma:.4f}: {self.optimal_lambda:.4f}")
            print(f"Conformal level: {conf_level:.4f} (desired coverage: {target_coverage:.4f}, adjustment factor: {adj_factor:.4f})")
        
        elif calibration_mode == "lambda_fixed":
            # Find optimal gamma with fixed lambda using conformal prediction
            # Calculate combined uncertainty with fixed lambda
            left_devs, right_devs = self._compute_combined_devs(self.optimal_lambda, aleatoric_left, aleatoric_right, epistemic_left, epistemic_right)
            
            # Calculate scores for conformal prediction
            scores = self._calculate_conformal_scores(y_calib, median_epistemic, left_devs, right_devs)
            
            # Get optimal gamma from quantile of scores
            self.gamma = np.quantile(scores, conf_level, method='higher')
            
            print(f"Optimal gamma with fixed lambda={self.optimal_lambda:.4f}: {self.gamma:.4f}")
            print(f"Conformal level: {conf_level:.4f} (desired coverage: {target_coverage:.4f}, adjustment factor: {adj_factor:.4f})")
        
        elif calibration_mode == "optimize_both":
            # Storage for each lambda
            gamma_values = []
            loss_values = []

            print(f"Calibrating CLEAR with {len(self.lambdas)} lambda values from {min(self.lambdas):.4f} to {max(self.lambdas):.4f}")

            # For each lambda, find the optimal gamma and calculate loss
            for lambda_val in self.lambdas:
                # Calculate combined deviations for this lambda
                left_devs, right_devs = self._compute_combined_devs(lambda_val, aleatoric_left, aleatoric_right, epistemic_left, epistemic_right)
                
                # Calculate scores for conformal prediction
                scores = self._calculate_conformal_scores(y_calib, median_epistemic, left_devs, right_devs)
                
                # Get optimal gamma from quantile of scores
                gamma = np.quantile(scores, conf_level, method='higher')
                gamma_values.append(gamma)
                
                # Calculate bounds with this gamma and lambda
                lower_bound = median_epistemic - gamma * left_devs
                upper_bound = median_epistemic + gamma * right_devs
                
                # Calculate pinball loss for these bounds
                loss = self._calculate_pinball_loss(y_calib, lower_bound, upper_bound)
                loss_values.append(loss)

            # Find lambda and gamma with minimum loss - THIS BLOCK IS NOW INDENTED
            min_index = np.argmin(loss_values)
            self.optimal_lambda = self.lambdas[min_index]
            self.gamma = gamma_values[min_index]
            
            # Print results specific to optimize_both mode
            print(f"Optimal lambda (via min loss): {self.optimal_lambda:.4f}, Corresponding Gamma: {self.gamma:.4f}")
            print(f"Minimized pinball loss: {loss_values[min_index]:.6f}")

        # This print statement now runs for all modes, reporting the final gamma/lambda
        print(f"Final calibration parameters - Lambda: {self.optimal_lambda:.4f}, Gamma: {self.gamma:.4f}")
        print(f"Conformal level used: {conf_level:.4f} (target coverage: {target_coverage:.4f}, adjustment factor: {adj_factor:.4f})")

        # Store the calibrated uncertainty metrics
        self.total_aleatoric_calib = self.gamma * np.mean(aleatoric_left + aleatoric_right)
        self.total_epistemic_calib = self.gamma * self.optimal_lambda * np.mean(epistemic_left + epistemic_right)
        self.uncertainty_ratio_calib = self.optimal_lambda * np.mean(epistemic_left + epistemic_right) / np.mean(aleatoric_left + aleatoric_right)

        if verbose:
          # Print diagnostic information about average magnitudes
          print("\n===== CALIBRATED UNCERTAINTY MAGNITUDE DIAGNOSTICS =====")
          print(f"  Total calibrated aleatoric: {self.total_aleatoric_calib:.6f}")
          print(f"  Total calibrated epistemic: {self.total_epistemic_calib:.6f}")
          print(f"  Calibrated Ratio (epistemic/aleatoric): {self.uncertainty_ratio_calib:.6f}")
          print("===========================================\n")

        if plot and calibration_mode == "optimize_both": # Only plot if loss_values were calculated
          plt.figure(figsize=(10, 6))
          plt.plot(self.lambdas, loss_values)
          plt.xlabel('Lambda')
          plt.ylabel('Pinball Loss')
          plt.title('Pinball Loss vs Lambda')
          plt.axvline(x=self.optimal_lambda, color='red', linestyle='--')
          plt.savefig('lambda_loss_curve.png')

        return self

    def fit_epistemic(self, X, y):
        """
        Fit epistemic uncertainty models using LinearGAM.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            self: The fitted model
        """
        X = np.asarray(X)
        y = np.asarray(y).flatten()
        
        # Initialize epistemic models list
        self.epistemic_models = []
        
        # Print progress message
        print("Fitting epistemic models...")
        
        # Fit bootstrap epistemic models
        for i in tqdm(range(self.n_bootstraps)):
            # Bootstrap sample indices
            if self.random_state is not None:
                np.random.seed(self.random_state + i)
            
            # Resample with replacement
            indices = np.random.choice(len(y), size=len(y), replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]
                        
            # Allow customization of GAM parameters
            gam_params = {'n_splines': 10, 'lam': 0.0000001}
            
            # Fit GAM model with the specified parameters
            gam_model = LinearGAM(**gam_params).fit(X_bootstrap, y_bootstrap)
            self.epistemic_models.append(gam_model)
        
        return self

    def predict_epistemic(self, X, aleatoric_median=None, symmetric_noise=False):
        """
        Predict epistemic uncertainty using GAM models.
        
        Args:
            X: Feature matrix
            aleatoric_median: Optional median predictions from aleatoric models
            symmetric_noise: If True, include aleatoric_median in the ensemble predictions
        Returns:
            median_pred: Median predictions
            lower_pred: Lower bound predictions
            upper_pred: Upper bound predictions
            ensemble_preds: Raw ensemble predictions (n_models, n_samples)
        """
        X = np.asarray(X)
        
        # Get predictions from each model
        epistemic_preds = []
        
        for model in self.epistemic_models:
            epistemic_preds.append(model.predict(X))
        
        # Prepare ensemble predictions in the expected format (n_models, n_samples)
        # Each row is a model, each column is a sample
        ensemble_preds = np.array(epistemic_preds)
        
        # Ensure ensemble_preds has shape (n_models, n_samples)
        if ensemble_preds.ndim == 1:
            ensemble_preds = ensemble_preds.reshape(1, -1)
        
        # Use compute_quantile_ensemble_bounds to get median and quantiles
        median_pred, lower_pred, upper_pred = compute_quantile_ensemble_bounds(
            ensemble_preds,
            lower_quantile=self.lower_quantile,
            upper_quantile=self.upper_quantile,
            random_state=self.random_state,
            aleatoric_median=aleatoric_median,
            symmetric_noise=symmetric_noise
        )
        
        print(f"[predict_epistemic] shapes: median_pred {median_pred.shape} (n_samples,), lower_pred {lower_pred.shape} (n_samples,), upper_pred {upper_pred.shape} (n_samples,), ensemble_preds {ensemble_preds.shape} (n_models, n_samples)")
        return median_pred, lower_pred, upper_pred, ensemble_preds

    def predict(self, X, external_epistemic=None, external_aleatoric=None):
        """
        Generate prediction intervals for test data using the calibrated model.

        Args:
            X: Feature matrix
            external_epistemic: Optional dictionary with pre-computed epistemic predictions:
                                {'median': median_predictions,
                                 'lower': lower_bound_predictions,
                                 'upper': upper_bound_predictions}
            external_aleatoric: Optional dictionary with pre-computed aleatoric predictions:
                                {'median': median_predictions,
                                 'lower': lower_bound_predictions,
                                 'upper': upper_bound_predictions}

        Returns:
            lower_bounds: Calibrated lower prediction bounds
            upper_bounds: Calibrated upper prediction bounds
        """
        if self.gamma is None:
            raise ValueError("Model not calibrated. Call calibrate first.")

        # Compute epistemic predictions if needed
        if external_epistemic is not None:
            median_epistemic = external_epistemic.get('median')
            epistemic_lower = external_epistemic.get('lower')
            epistemic_upper = external_epistemic.get('upper')
        else:
            if not hasattr(self, 'epistemic_models'):
                raise AttributeError("No epistemic models available; either provide external epistemic predictions or call fit_epistemic() first.")

            # Get temporary aleatoric predictions to use for epistemic prediction
            if external_aleatoric is not None and 'median' in external_aleatoric:
                temp_aleatoric_median = external_aleatoric['median']
            else:
                # Skip this step as we'll generate aleatoric predictions later
                temp_aleatoric_median = None

            median_epistemic, epistemic_lower, epistemic_upper, _ = self.predict_epistemic(X, temp_aleatoric_median)

        # Compute aleatoric predictions using external input if provided, otherwise use fitted models
        if external_aleatoric is not None:
            aleatoric_median = external_aleatoric.get('median')
            aleatoric_lower = external_aleatoric.get('lower')
            aleatoric_upper = external_aleatoric.get('upper')
        else:
            # Pass epistemic predictions to aleatoric prediction if using residual-based modeling
            if self.fit_on_residuals:
                aleatoric_median, aleatoric_lower, aleatoric_upper = self.predict_aleatoric(X, epistemic_preds=median_epistemic)
            else:
                aleatoric_median, aleatoric_lower, aleatoric_upper = self.predict_aleatoric(X)

        # Ensure predictions are numpy arrays
        median_epistemic = np.asarray(median_epistemic).flatten()
        epistemic_lower = np.asarray(epistemic_lower).flatten()
        epistemic_upper = np.asarray(epistemic_upper).flatten()
        aleatoric_median = np.asarray(aleatoric_median).flatten()
        aleatoric_lower = np.asarray(aleatoric_lower).flatten()
        aleatoric_upper = np.asarray(aleatoric_upper).flatten()

        # Apply proper quantile ordering - ensuring quantiles don't cross
        # (Important to do this *before* calculating deviations)
        aleatoric_upper = np.maximum(aleatoric_upper, aleatoric_median)
        aleatoric_lower = np.minimum(aleatoric_lower, aleatoric_median)
        epistemic_upper = np.maximum(epistemic_upper, median_epistemic)
        epistemic_lower = np.minimum(epistemic_lower, median_epistemic)

        # Calculate base deviations
        aleatoric_left = aleatoric_median - aleatoric_lower
        aleatoric_right = aleatoric_upper - aleatoric_median
        epistemic_left = median_epistemic - epistemic_lower
        epistemic_right = epistemic_upper - median_epistemic

        # Compute combined deviations using the helper method and stored lambda
        combined_left_devs, combined_right_devs = self._compute_combined_devs(
            self.optimal_lambda, aleatoric_left, aleatoric_right, epistemic_left, epistemic_right
        )

        # Compute final calibrated bounds using stored gamma
        lower_bounds = median_epistemic - self.gamma * combined_left_devs
        upper_bounds = median_epistemic + self.gamma * combined_right_devs

        # Ensure lower_bounds are always less than or equal to upper_bounds to prevent invalid intervals
        if np.any(lower_bounds > upper_bounds):
            print("WARNING: Lower bounds should be less than or equal to upper bounds")
            #----------------------------------
            # Force bounds not to cross in case of incorrectly ordered bounds which may occur in case of
            # 1. Large negative adjustments being applied.
            # 2. Floating-point inaccuracies in complex calculations.
            median_point = (lower_bounds + upper_bounds) / 2
            half_width = np.abs(upper_bounds - lower_bounds) / 2
            lower_bounds = median_point - half_width
            upper_bounds = median_point + half_width
            #----------------------------------
            # Re-check minimum difference to avoid floating point issues
            lower_bounds = np.minimum(lower_bounds, upper_bounds) 

        # print(f"[predict] lower_bounds {lower_bounds.shape} (n_samples,), upper_bounds {upper_bounds.shape} (n_samples,)")

        return lower_bounds, upper_bounds

if __name__ == "__main__":
    # Demonstrate CLEAR class functionality
    print("=== CLEAR Class Demonstration ===")
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 200
    X = np.random.rand(n_samples, 3) * 10
    y_true_func = lambda x: 5 * np.sin(x[:, 0]) + 2 * x[:, 1] - 0.5 * x[:, 2]**2
    y = y_true_func(X) + np.random.normal(0, 2, n_samples) # Aleatoric noise
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.5, random_state=42)
    X_calib, X_test, y_calib, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print(f"Data shapes: X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"             X_calib: {X_calib.shape}, y_calib: {y_calib.shape}")
    print(f"             X_test: {X_test.shape}, y_test: {y_test.shape}")

    # Initialize CLEAR model
    desired_coverage = 0.9
    clear_model = CLEAR(desired_coverage=desired_coverage, n_bootstraps=10, random_state=42, n_jobs=1) # Reduced bootstraps for demo

    # 1. Fit Epistemic Models (using internal GAMs)
    print("\n--- Fitting Epistemic Models ---")
    clear_model.fit_epistemic(X_train, y_train)
    
    # Predict with epistemic models (raw, uncalibrated)
    median_epistemic_calib, lower_epistemic_calib, upper_epistemic_calib, _ = clear_model.predict_epistemic(X_calib)
    median_epistemic_test, lower_epistemic_test, upper_epistemic_test, _ = clear_model.predict_epistemic(X_test)

    # 2. Fit Aleatoric Models 
    # (Example with RandomForestQuantileRegressor, can also use 'xgb', 'qgam')
    print("\n--- Fitting Aleatoric Models ---")
    clear_model.fit_aleatoric(X_train, y_train, quantile_model="rf", model_params={'n_estimators': 20}) # Reduced estimators for demo
    
    # Predict with aleatoric models (raw, uncalibrated)
    # For standard CLEAR (not residual-based for this demo part)
    median_aleatoric_calib, lower_aleatoric_calib, upper_aleatoric_calib = clear_model.predict_aleatoric(X_calib)
    median_aleatoric_test, lower_aleatoric_test, upper_aleatoric_test = clear_model.predict_aleatoric(X_test)

    # 3. Calibrate CLEAR
    print("\n--- Calibrating CLEAR ---")
    clear_model.calibrate(
        y_calib,
        median_epistemic=median_epistemic_calib,
        aleatoric_median=median_aleatoric_calib,
        aleatoric_lower=lower_aleatoric_calib,
        aleatoric_upper=upper_aleatoric_calib,
        epistemic_lower=lower_epistemic_calib,
        epistemic_upper=upper_epistemic_calib,
        verbose=True
    )
    print(f"Optimal Lambda: {clear_model.optimal_lambda:.4f}, Optimal Gamma: {clear_model.gamma:.4f}")

    # 4. Predict with calibrated CLEAR
    print("\n--- Predicting with Calibrated CLEAR ---")
    # We can pass the pre-computed epistemic and aleatoric components
    lower_pred_clear, upper_pred_clear = clear_model.predict(
        X_test,
        external_epistemic={
            'median': median_epistemic_test, 
            'lower': lower_epistemic_test, 
            'upper': upper_epistemic_test
        },
        external_aleatoric={
            'median': median_aleatoric_test, 
            'lower': lower_aleatoric_test, 
            'upper': upper_aleatoric_test
        }
    )
    
    # Evaluate results (basic example)
    from metrics import picp, mpiw # Assuming metrics.py is in the same directory
    
    coverage_clear = picp(y_test, lower_pred_clear, upper_pred_clear)
    width_clear = mpiw(lower_pred_clear, upper_pred_clear)
    
    print(f"CLEAR - PICP on test set: {coverage_clear:.4f} (Target: {desired_coverage:.2f})")
    print(f"CLEAR - MPIW on test set: {width_clear:.4f}")

    # --- Demonstrate Residual-based Aleatoric Fitting ---
    print("\n\n--- Demonstrating Residual-based Aleatoric Fitting ---")
    clear_model_residual = CLEAR(desired_coverage=desired_coverage, n_bootstraps=10, random_state=42, n_jobs=1)
    
    # Epistemic part is the same (or could be loaded externally)
    clear_model_residual.epistemic_models = clear_model.epistemic_models # Reuse fitted epistemic
    median_epistemic_train_res, _, _, _ = clear_model_residual.predict_epistemic(X_train) # Need train epistemic for residual calculation
    
    print("\n--- Fitting Aleatoric Models on Residuals ---")
    clear_model_residual.fit_aleatoric(
        X_train, y_train, 
        quantile_model="rf", model_params={'n_estimators': 20},
        fit_on_residuals=True,
        epistemic_preds=median_epistemic_train_res # Provide epistemic predictions on training data
    )

    # Predict aleatoric (residual-based)
    median_aleatoric_calib_res, lower_aleatoric_calib_res, upper_aleatoric_calib_res = \
        clear_model_residual.predict_aleatoric(X_calib, epistemic_preds=median_epistemic_calib)
    median_aleatoric_test_res, lower_aleatoric_test_res, upper_aleatoric_test_res = \
        clear_model_residual.predict_aleatoric(X_test, epistemic_preds=median_epistemic_test)
        
    print("\n--- Calibrating CLEAR (Residual-based) ---")
    clear_model_residual.calibrate(
        y_calib,
        median_epistemic=median_epistemic_calib,
        aleatoric_median=median_aleatoric_calib_res, # Use residual-based aleatoric median
        aleatoric_lower=lower_aleatoric_calib_res,
        aleatoric_upper=upper_aleatoric_calib_res,
        epistemic_lower=lower_epistemic_calib,
        epistemic_upper=upper_epistemic_calib,
        verbose=True
    )
    
    lower_pred_clear_res, upper_pred_clear_res = clear_model_residual.predict(
        X_test,
        external_epistemic={
            'median': median_epistemic_test, 
            'lower': lower_epistemic_test, 
            'upper': upper_epistemic_test
        },
        external_aleatoric={ # Pass residual-based aleatoric predictions
            'median': median_aleatoric_test_res, 
            'lower': lower_aleatoric_test_res, 
            'upper': upper_aleatoric_test_res
        }
    )
    
    coverage_clear_res = picp(y_test, lower_pred_clear_res, upper_pred_clear_res)
    width_clear_res = mpiw(lower_pred_clear_res, upper_pred_clear_res)
    
    print(f"CLEAR (Residual) - PICP on test set: {coverage_clear_res:.4f} (Target: {desired_coverage:.2f})")
    print(f"CLEAR (Residual) - MPIW on test set: {width_clear_res:.4f}")

    print("\nDemonstration finished.")
