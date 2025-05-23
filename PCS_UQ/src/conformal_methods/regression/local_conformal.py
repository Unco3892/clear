import numpy as np
import pandas as pd
from sklearn.base import clone


# Import local conformal utils
from scipy.spatial.distance import cdist
from src.conformal_methods.regression.local_conformal_utils import *

# Sklearn imports
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


class LocalConformalRegressor:
    """
    Conformal Localized Regression
    """
    def __init__ (self, model, alpha = 0.1, seed = 42):
        self.model = clone(model)
        self.alpha = alpha
        self.LCR = None
        self.seed = seed
        self._trained = False
        self.LCR = None
    
    def fit(self, X, y, alpha = None):
        
        if alpha is None:
            alpha = self.alpha
        
        X_train, X_calib, y_train, y_calib = train_test_split(X, y, test_size=0.5, random_state=self.seed)
        self.X_calib = X_calib
        self.model.fit(X_train, y_train)
        Vcv = np.abs(y_train).flatten()
        Dcv = cdist(X_train, X_train, metric='euclidean')

        # Set up a grid of localizer bandwidths. (In 1D the code used quantiles of Dcv;
        # here we use the same idea applied to the multivariate distances.)
        max0 = np.max(Dcv) * 2
        min0 = np.quantile(Dcv, 0.01)
        hs = np.exp(np.linspace(np.log(min0), np.log(max0), 20))
        self._calibrate(X_calib, y_calib, Vcv, Dcv, hs, alpha)

    def _calibrate(self, X, y,  Vcv, Dcv, hs, alpha = 0.1):
        eps = np.abs(y - self.model.predict(X)).flatten()
        D_calib = cdist(X, X, metric='euclidean')

        # For the LCP module we need to order the calibration scores.
        order1 = np.argsort(eps) 
        self._order1 = order1
        D_ordered = D_calib[order1][:, order1]
        eps_ordered = eps[order1]

        # make lcp
        LCR = LCP(H=D_ordered, V=eps_ordered, h=0.2, alpha=alpha, type="distance")

        # Auto-tuning: here we call the auto-tune function using the training data
        auto_ret = LCR.LCP_auto_tune(V0=Vcv, H0=Dcv, 
                                          hs=hs, B=2, delta=alpha/2, 
                                          lambda_=1, trace=True)
        LCR.h = auto_ret['h']

        # Prepare the calibration/localizer quantities:
        LCR.lower_idx()
        LCR.cumsum_unnormalized()
        self.LCR = LCR

       
    def predict(self, X):
        
        y_pred = self.model.predict(X)

        # computer distances
        Dnew = cdist(X, self.X_calib, metric='euclidean')
    

        # make predictions
        self.LCR.LCP_construction(Hnew=Dnew[:, self._order1], HnewT=Dnew.T[self._order1, :])
        deltaLCP = self.LCR.band_V
        lb = y_pred - deltaLCP
        ub = y_pred + deltaLCP

        return np.column_stack((lb, ub))
    
if __name__ == "__main__":
    X, y = make_regression(n_samples=100, n_features=10, noise=10)
    conformal = LocalConformalRegressor(LinearRegression())
    conformal.fit(X, y)
    y_pred = conformal.predict(X)
