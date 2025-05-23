# General Imports
import numpy as np
import pandas as pd
import os
import pickle
import copy
from tqdm import tqdm
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
from PCS.regression.pcs_uq import PCS_UQ

class PCS_OOB(PCS_UQ):
    def __init__(self, models, num_bootstraps=500, alpha=0.1, seed=42, top_k=1, val_size = 0.25, save_path=None, load_models=True, metric=r2_score, calibration_method = 'multiplicative'):
        """
        PCS OOB

        Args:
            models: dictionary of model names and models
            num_bootstraps: number of bootstraps
            alpha: significance level
            seed: random seed
            top_k: number of top models to use
            save_path: path to save the models
            load_models: whether to load the models from the save_path
            metric: metric to use for the prediction scores -- assume that higher is better
        """
        self.models = {model_name: copy.deepcopy(model) for model_name, model in models.items()}
        self.num_bootstraps = num_bootstraps
        self.alpha = alpha
        self.seed = seed
        self.top_k = top_k
        self.save_path = save_path
        self.load_models = load_models
        self.metric = metric
        self.pred_scores = {model: -np.inf for model in self.models}
        self.top_k_models = None
        self.bootstrap_models = None
        self.calibration_method = calibration_method
        self.val_size = val_size
    def fit(self, X, y, alpha = None):
        """
        Fit the models
        """
        if alpha is None:
            alpha = self.alpha
        X_train, X_calib, y_train, y_calib = train_test_split(X, y, test_size=self.val_size, random_state=self.seed)
        self._train(X_train, y_train)
        self._pred_check(X_calib, y_calib)
        self._get_top_k()
        self._train_top_k(X, y)
        uncalibrated_intervals = self.get_intervals(X)
        self.uncalibrated_metrics = get_all_metrics(y, uncalibrated_intervals[:,[0,2]])
        self.gamma = self.calibrate(uncalibrated_intervals, y, self.calibration_method)
        # self._train(X, y)
        # self._pred_check(X, y)
        # self._get_top_k()
        # self._train_top_k(X, y)
        # uncalibrated_intervals = self.get_intervals(X)
        # self.uncalibrated_metrics = get_all_metrics(y, uncalibrated_intervals[:,[0,2]])
        # self.gamma = self.calibrate(uncalibrated_intervals, y)
        

    def _train_top_k(self, X, y):
        """
        Train the models and store out-of-bag indices
        """
        # Initialize dictionaries once outside the model loop
        self.bootstrap_models = {}
        self.oob_indices = {}
        
        for model_name, model in self.top_k_models.items():
            # Initialize lists for each model
            self.bootstrap_models[model_name] = []
            self.oob_indices[model_name] = []
            
            for i in tqdm(range(self.num_bootstraps), desc=f"Training {model_name} models"):
                bootstrap_seed = self.seed + i
                # Try to load existing bootstrap model and OOB indices if enabled
                model_path = f"{self.save_path}/pcs_oob/{model_name}_model_seed_{bootstrap_seed}.pkl" if self.save_path else None
                oob_path = f"{self.save_path}/pcs_oob/{model_name}_oob_seed_{bootstrap_seed}.pkl" if self.save_path else None
                bootstrap_model = None
                
                if self.load_models and model_path and os.path.exists(model_path) and os.path.exists(oob_path):
                    with open(model_path, "rb") as f:
                        bootstrap_model = pickle.load(f)
                    with open(oob_path, "rb") as f:
                        oob_indices = pickle.load(f)
                    self.oob_indices[model_name].append(oob_indices)
                else:
                    # Bootstrap the data
                    n_samples = len(X)
                    bootstrap_indices = np.random.choice(range(n_samples), size=n_samples, replace=True)
                    oob_indices = list(set(range(n_samples)) - set(bootstrap_indices))
                    
                    X_boot = X[bootstrap_indices]
                    y_boot = y[bootstrap_indices]
                    
                    # Store OOB indices
                    self.oob_indices[model_name].append(oob_indices)
                    
                    # Create and fit bootstrap model
                    bootstrap_model = copy.deepcopy(model)
                    bootstrap_model.fit(X_boot, y_boot)
                    
                    # Save the bootstrap model and OOB indices if save path is provided
                    if model_path:
                        os.makedirs(os.path.dirname(model_path), exist_ok=True)
                        with open(model_path, "wb") as f:
                            pickle.dump(bootstrap_model, f)
                        with open(oob_path, "wb") as f:
                            pickle.dump(oob_indices, f)
                
                self.bootstrap_models[model_name].append(bootstrap_model)
            print(f"Finished training {model_name} models")


    def get_intervals(self, X):
        """
        Get predictions from all bootstrap models for the top k models,
        but only for samples that were out-of-bag for each bootstrap model.
        Returns row-wise sorted predictions.
        
        Args:
            X: Input features of shape (n_samples, n_features)
            
        Returns:
            predictions: numpy array of shape (n_samples, top_k * num_bootstraps)
                Contains sorted predictions from all bootstrap models for each sample.
                Values will be np.nan for samples that were not OOB for a given bootstrap.
        """
        n_samples = X.shape[0]
        n_predictions = self.top_k * self.num_bootstraps
        predictions = np.full((n_samples, n_predictions), np.nan)
        
        for k, model_name in enumerate(self.top_k_models):
            # For each bootstrap model of this top-k model
            for i in range(self.num_bootstraps):
                # Get OOB indices for this bootstrap model
                oob_indices = self.oob_indices[model_name][i]
                
                # Get predictions only for OOB samples
                bootstrap_model = self.bootstrap_models[model_name][i]
                X_oob = X[oob_indices]
                y_pred = bootstrap_model.predict(X_oob)
                
                # Store predictions in the appropriate column
                col_idx = k * self.num_bootstraps + i
                predictions[oob_indices, col_idx] = y_pred
        
        # Sort each row, keeping nan values at the end
        #predictions = np.sort(predictions, axis=1)
        intervals = np.zeros((n_samples, 3))
        intervals[:, 0] = np.nanquantile(predictions, self.alpha/2, axis=1)
        intervals[:, 1] = np.nanquantile(predictions, 0.5, axis=1)
        intervals[:, 2] = np.nanquantile(predictions, 1 - self.alpha/2, axis=1)
        return intervals

    def predict(self, X):
        """
        Make predictions with uncertainty intervals using the top k models.
        
        Args:
            X: Input features of shape (n_samples, n_features)
            
        Returns:
            predictions: numpy array of shape (n_samples, 2)
                Contains the lower/upper bounds for each sample
                [:,0] = lower bound
                [:,1] = upper bound
        """
        n_samples = X.shape[0]
        all_predictions = np.zeros((n_samples, self.top_k * self.num_bootstraps))
        
        # Get predictions from all bootstrap models
        col_idx = 0
        for model_name in self.top_k_models:
            for bootstrap_model in self.bootstrap_models[model_name]:
                all_predictions[:, col_idx] = bootstrap_model.predict(X)
                col_idx += 1
        
        # Sort predictions for each sample
        all_predictions.sort(axis=1)
        lower_bound = np.nanquantile(all_predictions, self.alpha/2, axis=1)
        median = np.nanquantile(all_predictions, 0.5, axis=1)
        upper_bound = np.nanquantile(all_predictions, 1 - self.alpha/2, axis=1)
        if self.calibration_method == 'multiplicative':
            lower_bound = median - self.gamma * (median - lower_bound)
            upper_bound = median + self.gamma * (upper_bound - median)
        elif self.calibration_method == 'additive':
            lower_bound = lower_bound - self.gamma
            upper_bound = upper_bound + self.gamma
        
        return np.column_stack([lower_bound, upper_bound])

if __name__ == "__main__":
    models = {
        "rf": RandomForestRegressor(n_estimators=100, max_depth=5),
        "lr": LinearRegression(),
        "ridge": RidgeCV()
    }
    calibration_method = ['multiplicative', 'additive']
    for method in calibration_method:
        X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        pcs_oob = PCS_OOB(models, save_path="test", load_models=False, calibration_method = method)
        pcs_oob.fit(X_train, y_train)
        intervals = pcs_oob.predict(X_test)
        print(f'{method}: {get_all_metrics(y_test, intervals)}')