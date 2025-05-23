import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from copy import deepcopy
from sklearn.base import clone

class StudentizedConformal:
    def __init__(self, mean_model, sd_model, alpha = 0.1, seed = 42):
        self.mean_model = clone(mean_model)
        self.sd_model = clone(sd_model)
        self.alpha = alpha
        self.seed = seed
        self.q = None
    
    def fit(self, X, y, alpha = None):
        """
        Fit the model on the training data. 
        Steps:
            1. Split the data into training and validation sets
            2. Fit the mean model on the training set
            3. Compute the residuals on the validation set
            4. Fit the sd model on the training set with the residuals

        Args:
            X: training features
            y: training labels
        
        """
        if alpha is None:
            alpha = self.alpha
        X_train, X_calib, y_train, y_calib = train_test_split(X, y, test_size=0.5, random_state=self.seed)
        self._train(X_train, y_train)
        self._calibrate(X_calib, y_calib, alpha)
    
    def _train(self, X, y):
        """
        Train the model on the training set.
        """
        self.mean_model.fit(X, y)
        train_residuals = np.abs(y - self.mean_model.predict(X))
        self.sd_model.fit(X, np.abs(train_residuals))
    
    def _calibrate(self, X, y, alpha = 0.1):
        """
        Calibrate the model on the validation set.
        """
        y_pred = self.mean_model.predict(X)
        residuals = np.abs(y - y_pred)
        resid_pred = np.abs(self.sd_model.predict(X))
        weighted_errors = residuals / resid_pred

        self.q = np.sort(weighted_errors)[int(np.ceil((len(X) + 1) * (1 - alpha)) - 1)]
        return self.q
    
    def predict(self, X):
        """
        Predict the intervals for the test set.
        """
        y_pred = self.mean_model.predict(X)
        resid_pred = np.abs(self.sd_model.predict(X))
        lower_bounds = y_pred - self.q * resid_pred
        upper_bounds = y_pred + self.q * resid_pred
        return np.column_stack((lower_bounds, upper_bounds))
    

if __name__ == "__main__":
    X = np.random.randn(100, 10)
    y = np.random.randn(100)
    model = StudentizedConformal(RandomForestRegressor(), LinearRegression())
    model.fit(X, y)
    print(model.predict(X))