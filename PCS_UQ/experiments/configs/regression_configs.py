import json
from pathlib import Path
import pandas as pd
import pickle
import numpy as np
# Model imports
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# PCS imports
from src.PCS.regression.pcs_uq import PCS_UQ
from src.PCS.regression.pcs_oob import PCS_OOB

# Conformal prediction imports
from src.conformal_methods.regression.split_conformal import SplitConformal
from src.conformal_methods.regression.studentized_conformal import StudentizedConformal
from src.conformal_methods.regression.local_conformal import LocalConformalRegressor
from src.conformal_methods.regression.majority_vote import MajorityVote

from experiments.configs.regression_consts import MODELS, DATASETS, VALID_UQ_METHODS, VALID_ESTIMATORS, SINGLE_CONFORMAL_METHODS, TEST_MODELS

#MODELS = {"XGBoost": XGBRegressor(random_state = 42)}#, "RandomForest": RandomForestRegressor(min_samples_leaf = 5, max_features = 0.33, n_estimators = 100, random_state = 42)}
#MODELS = {"LightGBM": LGBMRegressor(random_state = 42)}


def get_conformal_methods(conformal_type, model_name= 'XGBoost', seed = 0):
    if conformal_type == "split_conformal":
        return SplitConformal(model=MODELS[model_name], seed = seed), f"split_conformal_{model_name}"
    elif conformal_type == "studentized_conformal":
        return StudentizedConformal(mean_model=MODELS[model_name], sd_model=MODELS[model_name], seed = seed), f"studentized_conformal_{model_name}"
    elif conformal_type == "LocalConformalRegressor":
        return LocalConformalRegressor(model=MODELS[model_name], seed = seed), f"local_conformal_{model_name}"
    elif conformal_type == "majority_vote":
        return MajorityVote(models=MODELS, seed = seed), f"majority_vote"
    else:
        raise ValueError(f"Invalid conformal method: {conformal_type}")

def get_pcs_methods(pcs_type, seed = 0):
    if pcs_type == "pcs_uq":
        return PCS_UQ(models=MODELS, num_bootstraps=1000, alpha=0.1, top_k=1, load_models=False, seed = seed)
    elif pcs_type == "pcs_oob":
        return PCS_OOB(models=MODELS, num_bootstraps=1000, alpha=0.1, top_k=1, load_models=False, seed = seed)
    else:
        raise ValueError(f"Invalid PCS method: {pcs_type}")



def get_regression_datasets(dataset_name):
    if dataset_name not in DATASETS:
        raise ValueError(f"Dataset '{dataset_name}' not found. Available datasets are: {DATASETS}")
    
    X = pd.read_csv(f"experiments/data/{dataset_name}/X.csv")
    y = np.loadtxt(f"experiments/data/{dataset_name}/y.csv")
    with open(f'experiments/data/{dataset_name}/bin_df.pkl', 'rb') as f:
        bin_df = pickle.load(f)
    importance = pd.read_csv(f"experiments/data/{dataset_name}/importances.csv")
    return X, y, bin_df, importance


