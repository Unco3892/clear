
import pandas as pd
import numpy as np
from copy import deepcopy
from sklearn.base import clone
from src.conformal_methods.regression.split_conformal import SplitConformal
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

class MajorityVote:
    def __init__ (self, models, alpha = 0.1, seed = 42):
        """
        Initialize the Conformal Majority Vote model

        Args:
            models: dictionary of model names and models
            alpha: significance level
            seed: random seed
        """
        self.models = {model_name: clone(model) for model_name, model in models.items()}
        self.conformals = {model_name: None for model_name in models.keys()}
        self.K = len(models)
        self.alpha = alpha
        self.seed = seed
        self.trained = False


    def fit(self, X, y, alpha = None):
        """
        Fit the model on the training data.

        Args:
            X: training features
            y: training labels
        """
        if alpha is None:
            alpha = self.alpha
        self.conformals = {}
        X_train, X_calib, y_train, y_calib = train_test_split(X, y, test_size=0.5, random_state=self.seed)
        self._train(X_train, y_train)
        self._calibrate(X_calib, y_calib)
        return self
        
    def _train(self, X, y):
        for model_name, model in self.models.items():
            split_conformal_model = SplitConformal(model, self.alpha/2)
            split_conformal_model._train(X, y)
            self.conformals[model_name] = split_conformal_model
        self.trained = True
    
    def _calibrate(self, X, y, alpha=0.1):
        if self.trained == False:
            raise ValueError("Model must be trained before calling calibrate.")
        qs = {}
        for model_name, conformal in self.conformals.items():
            q = conformal._calibrate(X, y, alpha/2)
            qs[model_name] = q

        self.calibrated = True
        self.qs = qs
        return qs
    
    def predict(self, X, tau = 0.5):
        """
        Predict the intervals for the test set.

        Args:
            X: test features
            tau: significance level
        """
        model_intervals = {}
        for model_name, conformal in self.conformals.items():
            model_intervals[model_name] = conformal.predict(X) # Shape: (n_samples, 2)
        all_bounds = np.zeros((X.shape[0], 2*self.K))
        for i, model_name in enumerate(self.conformals.keys()):
            all_bounds[:, i] = model_intervals[model_name][:, 0]
            all_bounds[:, self.K + i] = model_intervals[model_name][:, 1]
        all_bounds_df = pd.DataFrame(all_bounds)
        pred_intervals = all_bounds_df.apply(
            lambda row: self._majority_vote_helper(row, 
                                                   self.K, 
                                                   tau), 
            axis=1, 
            result_type='expand').rename(columns={0: 'lower', 1: 'upper'})
        pred_intervals = pred_intervals.to_numpy()
        # # Convert lists to single values
        # pred_intervals = np.array([[interval[0][0], interval[1][0]] for interval in pred_intervals])
        return pred_intervals
    
    def _majority_vote_helper(self, row, K, tau):
        lower_bounds = row.iloc[:K].to_numpy()
        upper_bounds = row.iloc[K:].to_numpy()
        q = np.sort(row)
        i = 1
        lower = []
        upper = []
        while i < 2*K:
            cond_i = np.mean((lower_bounds <= (q[i-1] + q[i])/ 2) & (upper_bounds >= (q[i-1] + q[i])/ 2))
            if cond_i > tau:
                lower.append(q[i-1])
                j = i
                cond_j = np.mean((lower_bounds <= (q[j-1] + q[j])/ 2) & (upper_bounds >= (q[j-1] + q[j])/ 2))
                while (cond_j > tau) and (j < 2*K):
                    j += 1
                    cond_j = np.mean((lower_bounds <= (q[j-1] + q[j])/ 2) & (upper_bounds >= (q[j-1] + q[j])/ 2))
                i = j
                upper.append(q[i-1])
            else:
                i += 1
        return [lower, upper]
    
if __name__ == "__main__":
    X,y = make_regression(n_samples=1000, n_features=10, noise=10)
    models = {
        'rf': RandomForestRegressor(),
        'lr': LinearRegression(),
        'extraTrees': ExtraTreesRegressor()
    }
    conformal_majority_vote = MajorityVote(models)
    conformal_majority_vote.fit(X, y)
    print(conformal_majority_vote.predict(X))

    
    
    #print(evaluate_majority_vote(y, conformal_majority_vote.predict(X)))
    
