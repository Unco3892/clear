# General Imports
import numpy as np
import pandas as pd
import os
import pickle
import copy
# Sklearn Imports
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.ensemble import RandomForestRegressor  
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.datasets import make_regression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.base import clone
# PCS UQ Imports
from metrics.regression_metrics import *

class PCS_UQ:
    def __init__(self, models, num_bootstraps=10, alpha=0.1, seed=42, top_k = 1, save_path = None, 
                 load_models = True, val_size = 0.25, metric = r2_score, calibration_method = 'multiplicative'):
        """
        PCS UQ

        Args:
            models: dictionary of model names and models
            alpha: significance level
            seed: random seed
            top_k: number of top models to use
            save_path: path to save the models
            load_models: whether to load the models from the save_path
            metric: metric to use for the prediction scores -- assume that higher is better
        """
        self.models = {model_name: copy.deepcopy(model) for model_name, model in models.items()}
        self.alpha = alpha
        self.num_bootstraps = num_bootstraps
        self.seed = seed
        self.top_k = top_k
        self.save_path = save_path
        self.load_models = load_models
        self.val_size = val_size
        self.metric = metric
        self.pred_scores = {model: -np.inf for model in self.models}
        self.top_k_models = None
        self.bootstrap_models = None
        assert calibration_method in ['multiplicative', 'additive'], "Invalid calibration method"
        self.calibration_method = calibration_method
    
    def fit(self, X, y, X_calib = None, y_calib = None, alpha = 0.1):
        """
        Args: 
            X: features
            y: target
            X_calib: features for calibration
            y_calib: target for calibration
        Returns: 
            None
        Steps: 
        1. Split the data into training and calibration sets
        2. Train the models
        3. Check the predictions of the models
        4. Get the top k models
        5. Calibrate the top-k models 
        """
        if alpha is None:
            alpha = self.alpha
        
        if X_calib is None:
            X_train, X_calib, y_train, y_calib = train_test_split(X, y, test_size=self.val_size, random_state=self.seed)
        else:
            X_train, y_train = X, y
            X_calib, y_calib = X_calib, y_calib

        self._train(X_train, y_train) # train the models such that they are ready for calibration, saved in self.models
        self._pred_check(X_calib, y_calib) # check the predictions of the models, saved in self.models
        self.top_k_models = self._get_top_k()
        self._fit_bootstraps(X_train, y_train)
        uncalibrated_intervals  = self.get_intervals(X_calib) # get the uncalibrated intervals and raw width/coverage
        self.uncalibrated_metrics = get_all_metrics(y_calib, uncalibrated_intervals[:,[0,2]]) # drop median to assess raw coverage and width
        gamma = self.calibrate(uncalibrated_intervals, y_calib, self.calibration_method) # calibrate the intervals to get the best gamma 

                

    def _train(self, X, y):
        if self.load_models and (self.save_path is not None):
            print(f"Loading models from {self.save_path}")
            for model in self.models:
                try: 
                    with open(f"{self.save_path}/pcs_uq/{model}.pkl", "rb") as f:
                        self.models[model] = pickle.load(f)
                except FileNotFoundError:
                    print(f"No saved model found for {model}, fitting new model")
                    self.models[model].fit(X, y)
                    os.makedirs(f"{self.save_path}/pcs_uq", exist_ok=True)
                    with open(f"{self.save_path}/pcs_uq/{model}.pkl", "wb") as f:
                        pickle.dump(self.models[model], f)
        else: 
            for model in self.models:
                self.models[model].fit(X, y)
                if self.save_path is not None:
                    os.makedirs(f"{self.save_path}/pcs_uq", exist_ok=True)
                    with open(f"{self.save_path}/pcs_uq/{model}.pkl", "wb") as f:
                        pickle.dump(self.models[model], f)

    # For now, assume only one metric. 
    # TODO: Add support for multiple metrics by picking in average highest rank
    def _pred_check(self, X, y):
        """
        Args: 
            X: features
            y: target
        Steps: 
        1. Predict the target using the models
        2. Calculate the prediction score for each model
        """
        for model in self.models:
            y_pred = self.models[model].predict(X)
            self.pred_scores[model] = self.metric(y, y_pred)

    def _get_top_k(self):
        """
        Args: 
            None
        Steps: 
        1. Sort the models by the prediction score
        2. Return the top k models
        """
        sorted_models = sorted(self.pred_scores, key=self.pred_scores.get, reverse=True)
        top_k_model_names = sorted_models[:self.top_k]
        self.top_k_models = {model: self.models[model] for model in top_k_model_names}
        return self.top_k_models
    
    def _fit_bootstraps(self, X, y):
        """
        Generate prediction intervals using bootstrap resampling for each top-k model
        
        Args:
            X: features
            y: target
        Returns:
            Dictionary of model predictions for each bootstrap sample
        """
        bootstrap_predictions = {model: [] for model in self.top_k_models}
        bootstrap_models = {model: [] for model in self.top_k_models}
        
        for i in range(self.num_bootstraps):
            for model_name, model in self.top_k_models.items():
                if self.load_models and self.save_path is not None:
                    # Try to load existing bootstrap model
                    bootstrap_dir = os.path.join(self.save_path, 'pcs_uq', 'bootstrap_models', model_name)
                    bootstrap_path = f"{bootstrap_dir}/bootstrap_{model_name}_{i}.pkl"
                    
                    try:
                        with open(bootstrap_path, "rb") as f:
                            bootstrap_model = pickle.load(f)
                            print(f"Loaded bootstrap model {i} for {model_name}")
                    except (FileNotFoundError, EOFError):
                        # If loading fails, fit a new bootstrap model
                        print(f"Fitting new bootstrap model {i} for {model_name}")
                        X_boot, y_boot = resample(X, y, random_state=self.seed + i)
                        bootstrap_model = copy.deepcopy(model)
                        bootstrap_model.fit(X_boot, y_boot)
                        
                        # Save the newly fitted model
                        if self.save_path is not None:
                            os.makedirs(bootstrap_dir, exist_ok=True)
                            with open(bootstrap_path, "wb") as f:
                                pickle.dump(bootstrap_model, f)
                else:
                    # Fit new bootstrap model without attempting to load
                    X_boot, y_boot = resample(X, y, random_state=self.seed + i)
                    bootstrap_model = copy.deepcopy(model)
                    bootstrap_model.fit(X_boot, y_boot)
                    
                    # Save the model if save_path is specified
                    if self.save_path is not None:
                        bootstrap_dir = os.path.join(self.save_path, 'pcs_uq', 'bootstrap_models', model_name)
                        os.makedirs(bootstrap_dir, exist_ok=True)
                        with open(f"{bootstrap_dir}/bootstrap_{model_name}_{i}.pkl", "wb") as f:
                            pickle.dump(bootstrap_model, f)
                
                # Store the bootstrap model
                bootstrap_models[model_name].append(bootstrap_model)
                
                # Get predictions for the original data
                predictions = bootstrap_model.predict(X)
                bootstrap_predictions[model_name].append(predictions)
        
        self.bootstrap_models = bootstrap_models
        #return bootstrap_predictions


    def get_intervals(self, X):
        """
        Args: 
            X: features
            gamma: expansion factor
        Returns: 
            Dictionary containing lower and upper bounds of prediction intervals combining all models
            Format: {'lower': array, 'upper': array, 'predictions': array}
        """
        # Get number of samples
        n_samples = X.shape[0]
        
        # Initialize array to store all predictions
        # Shape will be (n_samples, K*B) - each row represents all predictions for one data point
        all_predictions = np.zeros((n_samples, 0))

        
        # Collect predictions for each data point
        for model_name, bootstrap_models in self.bootstrap_models.items():
            for model in bootstrap_models:
                pred = model.predict(X).reshape(-1, 1)  # Shape: (n_samples, 1)
                all_predictions = np.hstack((all_predictions, pred))
                
        
        # Sort predictions for each data point
        intervals = np.array((n_samples, 3))
        # Calculate quantiles for each row
        intervals = np.zeros((n_samples, 3))
        intervals[:, 0] = np.quantile(all_predictions, self.alpha/2, axis=1)  # Lower bound
        intervals[:, 1] = np.quantile(all_predictions, 0.5, axis=1)  # Median
        intervals[:, 2] = np.quantile(all_predictions, 1.0 - self.alpha/2, axis=1)  # Upper bound
        return intervals

    def calibrate(self,intervals, y_calib, calibration_method = 'multiplicative', gamma_min = 1.0, gamma_max = 1000.0, tol = 1e-6):
        if self.calibration_method == 'multiplicative':
            return self._calibrate_multiplicative(intervals, y_calib, gamma_min, gamma_max, tol)
        elif self.calibration_method == 'additive':
            return self._calibrate_additive(intervals, y_calib)
        
    def _calibrate_multiplicative(self, intervals, y_calib, gamma_min = 1.0, gamma_max = 1000.0, tol = 1e-6):
        """
        Calibrate the intervals
        """
        # gamma_range = np.linspace(start = gamma_min, stop = gamma_max, num = num_points)
        # coverage_list = []
        # width_list = []
        # Binary search to find minimum gamma that achieves target coverage
        left = gamma_min
        right = gamma_max
        best_gamma = gamma_max
        target_coverage = 1.0 - self.alpha
        
        while right - left > tol:
            gamma = (left + right) / 2
            lb = intervals[:, 1] - gamma * (intervals[:, 1] - intervals[:, 0])
            ub = intervals[:, 1] + gamma * (intervals[:, 2] - intervals[:, 1])
            coverage = np.mean((y_calib >= lb) & (y_calib <= ub))
            
            if coverage >= target_coverage:
                # Current gamma achieves coverage, try smaller gamma
                best_gamma = gamma
                right = gamma
            else:
                # Current gamma doesn't achieve coverage, try larger gamma
                left = gamma
                
        self.gamma = best_gamma
        return best_gamma

    def _calibrate_additive(self, intervals, y_calib, step_size = 1e-6):
        """
        Calibrate the intervals
        """
        min_gamma = 0.0
        max_gamma = np.max(y_calib) - np.min(y_calib)
        target_coverage = 1.0 - self.alpha
        coverage = np.mean((y_calib >= intervals[:, 0]) & (y_calib <= intervals[:, 2]))
        if coverage >= target_coverage:
            self.gamma = 0.0
            return 0.0
        else:
            left = min_gamma
            right = max_gamma
            best_gamma = max_gamma

            while right - left > step_size:
                gamma = (left + right) / 2
                lb = intervals[:, 0] - gamma
                ub = intervals[:, 2] + gamma
                coverage = np.mean((y_calib >= lb) & (y_calib <= ub))

                if coverage >= target_coverage:
                    best_gamma = gamma
                    right = gamma
                else:
                    left = gamma

            self.gamma = best_gamma
            return best_gamma
        
        # while coverage < target_coverage:
        #     lb = intervals[:, 0] - gamma
        #     ub = intervals[:, 2] + gamma
        #     coverage = np.mean((y_calib >= lb) & (y_calib <= ub))
        #     gamma += step_size
        # return gamma
    
        # gamma_range = np.linspace(start = gamma_min, stop = gamma_max, num = num_points)
        # coverage_list = []
        # width_list = []

    def predict(self, X,):
        uncalibrated_intervals = self.get_intervals(X)
        if self.calibration_method == 'multiplicative':
            lower_bound = uncalibrated_intervals[:, 1] - self.gamma * (uncalibrated_intervals[:, 1] - uncalibrated_intervals[:, 0])
            upper_bound  = uncalibrated_intervals[:, 1] + self.gamma * (uncalibrated_intervals[:, 2] - uncalibrated_intervals[:, 1])
        elif self.calibration_method == 'additive':
            lower_bound = uncalibrated_intervals[:, 0] - self.gamma
            upper_bound = uncalibrated_intervals[:, 2] + self.gamma
        intervals = np.zeros((X.shape[0], 2))
        intervals[:, 0] = lower_bound
        intervals[:, 1] = upper_bound
        return intervals



if __name__ == "__main__":
    
    models = {
        "rf": RandomForestRegressor(),
        "lr": LinearRegression(), 
        'ridge': RidgeCV()
    }
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    calibration_method = ['multiplicative', 'additive']
    for method in calibration_method:
        pcs_uq = PCS_UQ(models, save_path = 'test', load_models = True, calibration_method = method)
        pcs_uq.fit(X_train, y_train)
        intervals = pcs_uq.predict(X_test)
        print(f'{method}: {get_all_metrics(y_test, intervals)}')

 # # Calculate indices for lower and upper bounds based on alpha
        # n_total = sorted_predictions.shape[1]  # Total number of predictions per point (K * B)
        # lower_idx = int(np.floor(self.alpha/2 * n_total))
        # upper_idx = int(np.ceil((1 - self.alpha/2) * n_total))
        
        # # Get the prediction intervals
        # intervals = {
        #     'lower': sorted_predictions[:, lower_idx],
        #     'upper': sorted_predictions[:, upper_idx],
        #     'predictions': sorted_predictions  # Including all predictions for potential further use
        # }
        
        # return intervals
