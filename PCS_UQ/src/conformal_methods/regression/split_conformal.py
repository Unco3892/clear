# General imports
import numpy as np
import pandas as pd

# Sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.base import clone
# Local imports

class SplitConformal:
    """
    Split conformal prediction intervals for regression. 

    Args:
        model: sklearn regression model
        alpha: significance level
        seed: random seed for train-test split
    """
    def __init__(self, model, alpha = 0.1, seed = 42):
        """
        Initialize the SplitConformal class.

        Args:
            model: sklearn regression model
            alpha: significance level
            seed: random seed for train-test split
        """
        self.model = clone(model)
        self.alpha = alpha
        self.seed = seed
        self.q = None 
        self._trained = False
        self._calibrated = False
    
    def fit(self, X, y, alpha = None):
        """
        Fit and calibrate the model 

        Args:
            X: training features
            y: training labels
        """
        if alpha is None:
            alpha = self.alpha
        X_train, X_calib, y_train, y_calib = train_test_split(X, y, test_size=0.5, random_state=self.seed)    
        self._train(X_train, y_train)
        self._calibrate(X_calib, y_calib, alpha)

        return self 
    
    def _train(self, X, y):
        """
        Train the model on the training set.
        """
        self.model.fit(X, y)
        self._trained = True

    def _calibrate(self, X, y, alpha):
        """
        Calibrate the model on the validation set.
        """
        if self._trained == False:
            raise ValueError("Model must be fitted before calling calibrate.")
        residuals = np.abs(y - self.model.predict(X))
        self.q = np.sort(residuals)[int(np.ceil((len(X) + 1) * (1 - alpha)) - 1)]
        return self.q

    def predict(self, X):
        """
        Predicts intervals for test data using calibration errors.

        Parameters:
        ----------
        X : np.ndarray
            
        Returns:
        -------
        np.ndarray
            An array of shape (n_samples, 2) containing the lower and upper bounds of the prediction intervals.
        """
    
        # Predict on the test set
        y_pred_test = self.model.predict(X)

        # Compute prediction intervals
        lower_bounds = y_pred_test - self.q
        upper_bounds = y_pred_test + self.q

        return np.column_stack((lower_bounds, upper_bounds))


if __name__ == "__main__":
    X = np.random.randn(100, 10)
    y = np.random.randn(100)
    model = RandomForestRegressor()
    conformal = SplitConformal(model)
    conformal.fit(X, y)
    print(conformal.predict(X))
