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

class JackknifeBootstrap(PCS_UQ):
    def __init__(self, model, num_bootstraps=500, alpha=0.1, seed=42, save_path=None, load_models=True):
        """
        Jackknife Bootstrap

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
        self.model = model
        self.num_bootstraps = num_bootstraps
        self.alpha = alpha
        self.seed = seed
        self.save_path = save_path
        self.load_models = load_models
        self.bootstrap_models = None
        self.oob_indices = None

    def fit(self, X, y, alpha = None):
        """
        Fit the models
        """
        if alpha is None:
            alpha = self.alpha
        self._train_bootstrap(X, y)
        self._get_residuals(X, y)
       
    def _train_bootstrap(self, X, y):
        """
        Train the models and store out-of-bag indices
        """
        # Initialize dictionaries once outside the model loop
        self.bootstrap_models = []
        self.oob_indices = []
        self._train_sample_idx_to_oob_indices = []
        
        for i in tqdm(range(self.num_bootstraps),   ):
            bootstrap_seed = self.seed + i
            # Try to load existing bootstrap model and OOB indices if enabled
            
            
            n_samples = len(X)
            bootstrap_indices = np.random.choice(range(n_samples), size=n_samples, replace=True)
            oob_indices = list(set(range(n_samples)) - set(bootstrap_indices))
            
            X_boot = X[bootstrap_indices]
            y_boot = y[bootstrap_indices]
            
            # Store OOB indices
            self.oob_indices.append(oob_indices)            
            bootstrap_model = copy.deepcopy(self.model)
            bootstrap_model.fit(X_boot, y_boot)
            self.bootstrap_models.append(bootstrap_model)
                
    


    def _get_residuals(self, X, y):
        """
        Average the residuals of the bootstrap models for each sample.
        
        Args:
            X: Input features of shape (n_samples, n_features)
            
        Returns:
            predictions: numpy array of shape (n_samples, num_bootstraps)
                Contains sorted predictions from all bootstrap models for each sample.
                Values will be np.nan for samples that were not OOB for a given bootstrap.
        """
        n_samples = X.shape[0]
        n_predictions = self.num_bootstraps
        predictions = np.full((n_samples, n_predictions), np.nan)
        residuals = np.zeros(n_samples)
        
        
        
        for i in range(self.num_bootstraps):
                # Get OOB indices for this bootstrap model
            oob_indices = self.oob_indices[i]
            bootstrap_model = self.bootstrap_models[i]
            X_oob = X[oob_indices]
            y_pred = bootstrap_model.predict(X_oob)
            predictions[oob_indices, i] = y_pred
        
        # Sort each row, keeping nan values at the end
        #predictions = np.sort(predictions, axis=1)
        mean_predictions = np.nanmean(predictions, axis=1)
        residuals = y - mean_predictions

        # Get boolean mask of non-nan entries
        mask = ~np.isnan(predictions)

        # Get list of column indices for each row where the values are not nan
        non_nan_indices = [np.where(row)[0] for row in mask]
        self._train_sample_idx_to_oob_indices = non_nan_indices

        self.residuals = residuals
    
    def _get_mean_predictions(self, X, model_list):
        predictions = np.zeros((X.shape[0]))
        for model in model_list:
            predictions += model.predict(X)
        predictions /= len(model_list)
        return predictions
    
    def predict(self, X):
        """
        Make predictions with uncertainty intervals 
        """
        # for each training sample, get the oob models
        num_training_samples = len(self._train_sample_idx_to_oob_indices)
        self.bootstrap_models = np.array(self.bootstrap_models)
        predictions = np.zeros((X.shape[0], num_training_samples))
        for i in range(num_training_samples):
            oob_models = self.bootstrap_models[self._train_sample_idx_to_oob_indices[i]]
            oob_predictions = self._get_mean_predictions(X, oob_models) + self.residuals[i]
            predictions[:, i] = oob_predictions

        intervals = np.zeros((X.shape[0], 2))
        lb = np.quantile(predictions, self.alpha/2, axis=1)
        ub = np.quantile(predictions, 1-self.alpha/2, axis=1)
        intervals[:, 0] = lb
        intervals[:, 1] = ub

        return intervals

if __name__ == "__main__":
    models = LinearRegression()
    calibration_method = ['multiplicative', 'additive']
    X, y = make_regression(n_samples=500, n_features=1, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    jackknife_bootstrap = JackknifeBootstrap(models, save_path="test", load_models=False)
    jackknife_bootstrap.fit(X_train, y_train)
    intervals = jackknife_bootstrap.predict(X_test)
    print(f'{get_all_metrics(y_test, intervals)}')
