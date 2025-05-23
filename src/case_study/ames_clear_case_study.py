# %% [markdown]
# # [Chapter 13] The final Ames sale price predictions
# 
# ## [DSLC stages]: Analysis
# 
# 
# The following code sets up the libraries and creates cleaned and pre-processed training, validation and test data that we will use in this document.
# 

# %%
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklego.linear_model import LADRegression
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.preprocessing import OneHotEncoder, scale, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from itertools import product
from joblib import Parallel, delayed
from itertools import compress
import warnings
import os
import sys
import argparse

# Set random seeds for reproducibility
np.random.seed(299433)  # Same as R version's first seed
import random
random.seed(299433)

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)
sys.path.append(os.path.join(script_dir, '..'))

# Imports for CQR and CLEAR
from sklearn.linear_model import QuantileRegressor
from clear.clear import CLEAR
from clear.metrics import evaluate_intervals # Assuming evaluate_intervals is accessible


# Suppress warnings
warnings.filterwarnings("ignore")

# Setup paths for imports
current_file = os.path.abspath(__file__)
case_study_dir = os.path.dirname(current_file)
src_dir = os.path.dirname(case_study_dir)
root_dir = os.path.dirname(src_dir)

# Add root directory to path
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Add dslc_documentation directory to path
dslc_dir = os.path.join(case_study_dir, 'dslc_documentation')
if dslc_dir not in sys.path:
    sys.path.insert(0, dslc_dir)

# define all of the objects we need
# Replace Jupyter magic with standard import
from dslc_documentation.prepare_ames_data import *

# pd.set_option('display.max_columns', None)
# pd.options.display.max_colwidth = 500
# pd.options.display.max_rows = 100

# --- Argument Parser ---
parser = argparse.ArgumentParser(description='Run Ames Housing prediction interval analysis with model options.')
parser.add_argument('--aleatoric_model', type=str, default='linear', choices=['gam', 'linear'],
                    help='Model type for aleatoric uncertainty (CQR baseline and CLEAR): "gam" or "linear". Default is "gam".')
parser.add_argument('--use_top_features', action='store_true', default=False,
                    help='Use only the top 2 features (overall_qual, gr_liv_area).')
# Parse arguments if running as a script, otherwise use default 'gam' for interactive use
if __name__ == '__main__' and '__file__' in globals():
    args = parser.parse_args()
else:
    # Default args for interactive/notebook environment
    args = parser.parse_args([]) # Use default 'gam'

print(f"Using aleatoric model: {args.aleatoric_model}")
# --- End Argument Parser ---

# --- Removing argument parser section entirely ---
# No more argument parser code here

# %% [markdown]
# 
# In this document we will demonstrate how to use the principles of PCS to choose the final prediction. We will demonstrate three different formats of the final prediction based on:
# 
# 1. The **single "best"** predictive algorithm, in terms of validation set performance from among a range of different algorithms each trained on several different cleaning/pre-processing judgment call-perturbed versions of the training dataset.
# 
# 1. An **ensemble** prediction, which combines the predictions from a range of predictive fits from across different algorithms and cleaning/pre-processing judgment call-perturbations that pass a predictability screening test.
# 
# 1. An **interval** of predictions from a range of predictive fits from across different algorithms and data- and cleaning/pre-processing judgment call-perturbations that pass a predictability screening test. We will compare the original PCS-PPI approach with CQR and CLEAR.
# 


# %% [markdown]
# 
# ## Computing the perturbed predictions
# 
# Since each of these approaches will involve each perturbed version of the cleaning/pre-processing judgment call training (and validation) datasets that we used in our stability analyses, (only the final interval approach will involve the data-perturbations), we will create the cleaning/pre-processing judgment call-perturbed datasets and fit the algorithms here.
# 
# 
# ### Create the perturbed datasets
# 
# First, let's create the list of the cleaning/pre-processing judgment call perturbed datasets:
# 

# %%
perturb_options = list(product([0.65, 0.8, 0.95], 
                               [10, 20],
                               ['other', 'mode'],
                               [True, False],
                               ['none', 'log', 'sqrt'],
                               [0, 0.5],
                               ['numeric', 'simplified_dummy', 'dummy']))
perturb_options = pd.DataFrame(perturb_options, columns=('max_identical_thresh', 
                                                         'n_neighborhoods',
                                                         'impute_missing_categorical',
                                                         'simplify_vars',
                                                         'transform_response',
                                                         'cor_feature_selection_threshold',
                                                         'convert_categorical'))

# %%
# conduct judgment call perturbations of training data
ames_jc_perturb = [preprocess_ames_data(ames_train_clean,
                                        max_identical_thresh=perturb_options['max_identical_thresh'][i],
                                        n_neighborhoods=perturb_options['n_neighborhoods'][i],
                                        impute_missing_categorical=perturb_options['impute_missing_categorical'][i],
                                        simplify_vars=perturb_options['simplify_vars'][i],
                                        transform_response=perturb_options['transform_response'][i],
                                        cor_feature_selection_threshold=perturb_options['cor_feature_selection_threshold'][i],
                                        convert_categorical=perturb_options['convert_categorical'][i],
                                        use_top_features=args.use_top_features, # Pass arg
                                        keep_pid=False # No need to keep pid for training set preprocessing
                                        )
                   for i in range(perturb_options.shape[0])]
print(f"Number of perturbed datasets: {len(ames_jc_perturb)}")

# %% [markdown]
# Note that several of the cleaning/pre-processing judgment call combinations will lead to identical datasets, so we will remove the duplicates from the list of perturbed datasets:

# %%
# since some judgment-call combinations yield the same data frames,
# remove duplicate data frames from the ames_jc_perturb list
new_list = []
# keep a record of which perturbations are retained
ames_jc_perturb_id = []
# add each perturbed dataset to the new_list if it isn't already included 
for i in range(len(ames_jc_perturb)):
  if not any (df.equals(ames_jc_perturb[i]) for df in new_list):
      new_list.append(ames_jc_perturb[i])
      ames_jc_perturb_id.append(i)
# update ames_jc_perturb to be this new_list object
ames_jc_perturb = new_list
# check how many perturbed datasets remain - note that this is about 40 fewer than the R version
# (presumably because I have slight differences in the implementation of the R/Python pre-processing code)
len(ames_jc_perturb)

# %%
# filter the set of perturb_options to only include the perturbations that are retained
perturb_options = perturb_options.iloc[ames_jc_perturb_id, :].reset_index(drop=True)
perturb_options.shape

# %%

# conduct judgment call perturbations of validation data data (we need to make sure each validation set is compartible with the relevant training set)
ames_val_jc_perturb = []
for i in range(len(ames_jc_perturb)):
    
    # extract relevant neighborhoods from  relevant training data
    train_neighborhood_cols = list(ames_jc_perturb[i].filter(regex="neighborhood").columns)
    train_neighborhoods = [x.replace("neighborhood_", "") for x in train_neighborhood_cols]
    
    # create preprocessed validation set
    ames_val_jc_perturb.append(
        preprocess_ames_data(ames_val_clean,
                             max_identical_thresh=perturb_options['max_identical_thresh'][i],
                             n_neighborhoods=perturb_options['n_neighborhoods'][i],
                             impute_missing_categorical=perturb_options['impute_missing_categorical'][i],
                             simplify_vars=perturb_options['simplify_vars'][i],
                             transform_response=perturb_options['transform_response'][i],
                             cor_feature_selection_threshold=perturb_options['cor_feature_selection_threshold'][i],
                             convert_categorical=perturb_options['convert_categorical'][i],
                             use_top_features=args.use_top_features, # Pass arg
                             # make sure val set matches training set
                             column_selection=list(ames_jc_perturb[i].columns),
                             neighborhood_levels=train_neighborhoods,
                             keep_pid=True # Keep pid for validation
                             )
        )

# create a standardized version of the validation datasets
ames_val_jc_perturb_std = []
for i in range(len(ames_val_jc_perturb)):
    df = ames_val_jc_perturb[i].drop(columns=['saleprice'])
    df_std = (df - df.mean()) / df.std()
    df_std['saleprice'] = ames_val_jc_perturb[i]['saleprice']
    ames_val_jc_perturb_std.append(df_std)



# %%
# conduct judgment call perturbations of test data data (we need to make sure each test set is compartible with the relevant training set)
ames_test_jc_perturb = []
for i in range(perturb_options.shape[0]):
    
    # extract relevant neighborhoods from  relevant training data
    train_neighborhood_cols = list(ames_jc_perturb[i].filter(regex="neighborhood").columns)
    train_neighborhoods = [x.replace("neighborhood_", "") for x in train_neighborhood_cols]
    
    # create preprocessed test set
    ames_test_jc_perturb.append(
        preprocess_ames_data(ames_test_clean,
                             max_identical_thresh=perturb_options['max_identical_thresh'][i],
                             n_neighborhoods=perturb_options['n_neighborhoods'][i],
                             impute_missing_categorical=perturb_options['impute_missing_categorical'][i],
                             simplify_vars=perturb_options['simplify_vars'][i],
                             transform_response=perturb_options['transform_response'][i],
                             cor_feature_selection_threshold=perturb_options['cor_feature_selection_threshold'][i],
                             convert_categorical=perturb_options['convert_categorical'][i],
                             use_top_features=args.use_top_features, # Pass arg
                             # make sure test set matches training set
                             column_selection=list(ames_jc_perturb[i].columns),
                             neighborhood_levels=train_neighborhoods,
                             keep_pid=True # Keep pid for test set
                             )
        )

# create a standardized version of the test datasets
ames_test_jc_perturb_std = []
for i in range(len(ames_test_jc_perturb)):
    df = ames_test_jc_perturb[i].drop(columns=['saleprice'])
    df_std = (df - df.mean()) / df.std()
    df_std['saleprice'] = ames_test_jc_perturb[i]['saleprice']
    ames_test_jc_perturb_std.append(df_std)


# %% [markdown]
# 
# 
# ### Fitting the algorithms to each perturbed dataset
# 
# 
# Below is a function that will fit all models simultaneously:
# 

# %%
# This code takes a while to run, so we will use the joblib library to parallelize the code
def fit_models(df):
    
    # standardize predictor variables in df for ridge and lasso
    df_x = df.drop(columns='saleprice')
    df_x_std = (df_x - df_x.mean()) / df_x.std()
    df_y = df['saleprice']
    
    ls_fit = LinearRegression().fit(X=df_x, y=df_y) 
    lad_fit = LADRegression().fit(X=df_x, y=df_y)
    rf_fit = RandomForestRegressor().fit(X=df_x, y=df_y)
        
    
    alphas = np.logspace(-1, 5, 100)
    ridge_cv_scores = []
    for alpha in alphas:
        ridge = Ridge(alpha=alpha)
        ridge_cv = cross_validate(estimator=ridge,
                                X=df_x_std,
                                y=df_y,
                                cv=10,
                                scoring='neg_root_mean_squared_error')
        ridge_cv_scores.append({'alpha': alpha,
                                'log_alpha': np.log(alpha),
                                'test_mse': -np.mean(ridge_cv['test_score'])})
        
    ridge_cv_scores_df = pd.DataFrame(ridge_cv_scores)
    ridge_alpha_min = ridge_cv_scores_df.sort_values(by='test_mse').head(1).alpha.values[0]
    # identify the 1SE value
    mse_se_ridge = ridge_cv_scores_df['test_mse'].std() / np.sqrt(10)
    mse_min_ridge = ridge_cv_scores_df['test_mse'].min()
    ridge_alpha_1se = ridge_cv_scores_df[(ridge_cv_scores_df['test_mse'] <= mse_min_ridge + mse_se_ridge) & 
                                        (ridge_cv_scores_df['test_mse'] >= mse_min_ridge - mse_se_ridge)].sort_values(by='alpha', ascending=False).head(1).alpha.values[0]
    ridge_fit = Ridge(alpha=ridge_alpha_1se).fit(X=df_x_std, y=df_y)
    
    alphas = np.logspace(-2, 7, 100)
    lasso_cv_scores = []
    for alpha in alphas:
        lasso = Lasso(alpha=alpha, max_iter=10000)
        lasso_cv = cross_validate(estimator=lasso,
                                X=df_x_std,
                                y=df_y,
                                cv=10,
                                scoring='neg_root_mean_squared_error')
        lasso_cv_scores.append({'alpha': alpha,
                                'log_alpha': np.log(alpha),
                                'test_mse': -np.mean(lasso_cv['test_score'])})
    lasso_cv_scores_df = pd.DataFrame(lasso_cv_scores)
    lasso_alpha_min = lasso_cv_scores_df.sort_values(by='test_mse').head(1).alpha.values[0]
    # identify the 1SE value
    mse_se_lasso = lasso_cv_scores_df['test_mse'].std() / np.sqrt(10)
    mse_min_lasso = lasso_cv_scores_df['test_mse'].min()
    lasso_alpha_1se = lasso_cv_scores_df[(lasso_cv_scores_df['test_mse'] <= mse_min_lasso + mse_se_lasso) & 
                                        (lasso_cv_scores_df['test_mse'] >= mse_min_lasso - mse_se_lasso)].sort_values(by='alpha', ascending=False).head(1).alpha.values[0]
    lasso_fit = Lasso(alpha=lasso_alpha_1se).fit(X=df_x_std, y=df_y)
    
    return (ls_fit, lad_fit, ridge_fit, lasso_fit, rf_fit)

# %%
results_jc = Parallel(n_jobs=-1)(delayed(fit_models)(df) for df in ames_jc_perturb)
ls_jc_perturbed, lad_jc_perturbed, ridge_jc_perturbed, lasso_jc_perturbed, rf_jc_perturbed = zip(*results_jc)

# %%
# compute the predictions on the validaion set for ls_area_perturbed, ls_multi_perturbed, ls_all_perturbed, cart_perturbed, and rf_perturbed
ls_val_jc_pred_perturbed = [ls_jc_perturbed[i].predict(X=ames_val_jc_perturb[i].drop(columns='saleprice')) for i in range(len(ames_val_jc_perturb))]
lad_val_jc_pred_perturbed = [lad_jc_perturbed[i].predict(X=ames_val_jc_perturb[i].drop(columns='saleprice')) for i in range(len(ames_val_jc_perturb))]
ridge_val_jc_pred_perturbed = [ridge_jc_perturbed[i].predict(X=ames_val_jc_perturb_std[i].drop(columns='saleprice')) for i in range(len(ames_val_jc_perturb_std))]
lasso_val_jc_pred_perturbed = [lasso_jc_perturbed[i].predict(X=ames_val_jc_perturb_std[i].drop(columns='saleprice')) for i in range(len(ames_val_jc_perturb_std))]
rf_val_jc_pred_perturbed = [rf_jc_perturbed[i].predict(X=ames_val_jc_perturb[i].drop(columns='saleprice')) for i in range(len(ames_val_jc_perturb))]

# %%
# for predictions where the response was log-transformed, undo the log transformation
ls_val_jc_pred_perturbed = [np.exp(pred) if perturb_options['transform_response'][i] == 'log' else pred for i, pred in enumerate(ls_val_jc_pred_perturbed)]
lad_val_jc_pred_perturbed = [np.exp(pred) if perturb_options['transform_response'][i] == 'log' else pred for i, pred in enumerate(lad_val_jc_pred_perturbed)]
ridge_val_jc_pred_perturbed = [np.exp(pred) if perturb_options['transform_response'][i] == 'log' else pred for i, pred in enumerate(ridge_val_jc_pred_perturbed)]
lasso_val_jc_pred_perturbed = [np.exp(pred) if perturb_options['transform_response'][i] == 'log' else pred for i, pred in enumerate(lasso_val_jc_pred_perturbed)]
rf_val_jc_pred_perturbed = [np.exp(pred) if perturb_options['transform_response'][i] == 'log' else pred for i, pred in enumerate(rf_val_jc_pred_perturbed)]

# for predictions where the response was sqrt-transformed, undo the sqrt transformation
ls_val_jc_pred_perturbed = [pred**2 if perturb_options['transform_response'][i] == 'sqrt' else pred for i, pred in enumerate(ls_val_jc_pred_perturbed)]
lad_val_jc_pred_perturbed = [pred**2 if perturb_options['transform_response'][i] == 'sqrt' else pred for i, pred in enumerate(lad_val_jc_pred_perturbed)]
ridge_val_jc_pred_perturbed = [pred**2 if perturb_options['transform_response'][i] == 'sqrt' else pred for i, pred in enumerate(ridge_val_jc_pred_perturbed)]
lasso_val_jc_pred_perturbed = [pred**2 if perturb_options['transform_response'][i] == 'sqrt' else pred for i, pred in enumerate(lasso_val_jc_pred_perturbed)]
rf_val_jc_pred_perturbed = [pred**2 if perturb_options['transform_response'][i] == 'sqrt' else pred for i, pred in enumerate(rf_val_jc_pred_perturbed)]

# %% [markdown]
# Next, we will compute the performance of each model on the validation set for each perturbed dataset:

# %%
# LS model
ls_val_jc_perturbed_rmse = [np.sqrt(np.mean((ames_val_preprocessed['saleprice'] - ls_val_jc_pred_perturbed[i])**2)) for i in range(len(ames_val_jc_perturb))]
ls_val_jc_perturbed_mae = [np.mean(np.abs(ames_val_preprocessed['saleprice'] - ls_val_jc_pred_perturbed[i])) for i in range(len(ames_val_jc_perturb))]
ls_val_jc_perturbed_corr = [np.corrcoef(ames_val_preprocessed['saleprice'], ls_val_jc_pred_perturbed[i])[0,1] for i in range(len(ames_val_jc_perturb))]

# LAD model
lad_val_jc_perturbed_rmse = [np.sqrt(np.mean((ames_val_preprocessed['saleprice'] - lad_val_jc_pred_perturbed[i])**2)) for i in range(len(ames_val_jc_perturb))]
lad_val_jc_perturbed_mae = [np.mean(np.abs(ames_val_preprocessed['saleprice'] - lad_val_jc_pred_perturbed[i])) for i in range(len(ames_val_jc_perturb))]
lad_val_jc_perturbed_corr = [np.corrcoef(ames_val_preprocessed['saleprice'], lad_val_jc_pred_perturbed[i])[0,1] for i in range(len(ames_val_jc_perturb))]

# Ridge model
ridge_val_jc_perturbed_rmse = [np.sqrt(np.mean((ames_val_preprocessed['saleprice'] - ridge_val_jc_pred_perturbed[i])**2)) for i in range(len(ames_val_jc_perturb))]
ridge_val_jc_perturbed_mae = [np.mean(np.abs(ames_val_preprocessed['saleprice'] - ridge_val_jc_pred_perturbed[i])) for i in range(len(ames_val_jc_perturb))]
ridge_val_jc_perturbed_corr = [np.corrcoef(ames_val_preprocessed['saleprice'], ridge_val_jc_pred_perturbed[i])[0,1] for i in range(len(ames_val_jc_perturb))]

# Lasso model
lasso_val_jc_perturbed_rmse = [np.sqrt(np.mean((ames_val_preprocessed['saleprice'] - lasso_val_jc_pred_perturbed[i])**2)) for i in range(len(ames_val_jc_perturb))]
lasso_val_jc_perturbed_mae = [np.mean(np.abs(ames_val_preprocessed['saleprice'] - lasso_val_jc_pred_perturbed[i])) for i in range(len(ames_val_jc_perturb))]
lasso_val_jc_perturbed_corr = [np.corrcoef(ames_val_preprocessed['saleprice'], lasso_val_jc_pred_perturbed[i])[0,1] for i in range(len(ames_val_jc_perturb))]

# Random Forest model
rf_val_jc_perturbed_rmse = [np.sqrt(np.mean((ames_val_preprocessed['saleprice'] - rf_val_jc_pred_perturbed[i])**2)) for i in range(len(ames_val_jc_perturb))]
rf_val_jc_perturbed_mae = [np.mean(np.abs(ames_val_preprocessed['saleprice'] - rf_val_jc_pred_perturbed[i])) for i in range(len(ames_val_jc_perturb))]
rf_val_jc_perturbed_corr = [np.corrcoef(ames_val_preprocessed['saleprice'], rf_val_jc_pred_perturbed[i])[0,1] for i in range(len(ames_val_jc_perturb))]


# %%
# place all of the correlation performance lists in a data frame
perturbed_jc_corr = pd.DataFrame({
    'ls': ls_val_jc_perturbed_corr,
    'lad': lad_val_jc_perturbed_corr,
    'ridge': ridge_val_jc_perturbed_corr,
    'lasso': lasso_val_jc_perturbed_corr,
    'rf': rf_val_jc_perturbed_corr,
    'max_identical_thresh': perturb_options['max_identical_thresh'],
    'n_neighborhoods': perturb_options['n_neighborhoods'],
    'impute_missing_categorical': perturb_options['impute_missing_categorical'],
    'simplify_vars': perturb_options['simplify_vars'],
    'transform_response': perturb_options['transform_response'],
    'cor_feature_selection_threshold': perturb_options['cor_feature_selection_threshold'],
    'convert_categorical': perturb_options['convert_categorical']
    }).melt(id_vars=['max_identical_thresh', 'n_neighborhoods', 'impute_missing_categorical', 'simplify_vars', 'transform_response', 'cor_feature_selection_threshold', 'convert_categorical'], 
            var_name='model', 
            value_name='corr')

# place all of the RMSE performance lists in a data frame
perturbed_jc_rmse = pd.DataFrame({
    'ls': ls_val_jc_perturbed_rmse,
    'lad': lad_val_jc_perturbed_rmse,
    'ridge': ridge_val_jc_perturbed_rmse,
    'lasso': lasso_val_jc_perturbed_rmse,
    'rf': rf_val_jc_perturbed_rmse,
    'max_identical_thresh': perturb_options['max_identical_thresh'],
    'n_neighborhoods': perturb_options['n_neighborhoods'],
    'impute_missing_categorical': perturb_options['impute_missing_categorical'],
    'simplify_vars': perturb_options['simplify_vars'],
    'transform_response': perturb_options['transform_response'],
    'cor_feature_selection_threshold': perturb_options['cor_feature_selection_threshold'],
    'convert_categorical': perturb_options['convert_categorical']
    }).melt(id_vars=['max_identical_thresh', 'n_neighborhoods', 'impute_missing_categorical', 'simplify_vars', 'transform_response', 'cor_feature_selection_threshold', 'convert_categorical'], 
            var_name='model', 
            value_name='rmse')

# place all of the MAE performance lists in a data frame
perturbed_jc_mae = pd.DataFrame({
    'ls': ls_val_jc_perturbed_mae,
    'lad': lad_val_jc_perturbed_mae,
    'ridge': ridge_val_jc_perturbed_mae,
    'lasso': lasso_val_jc_perturbed_mae,
    'rf': rf_val_jc_perturbed_mae,
    'max_identical_thresh': perturb_options['max_identical_thresh'],
    'n_neighborhoods': perturb_options['n_neighborhoods'],
    'impute_missing_categorical': perturb_options['impute_missing_categorical'],
    'simplify_vars': perturb_options['simplify_vars'],
    'transform_response': perturb_options['transform_response'],
    'cor_feature_selection_threshold': perturb_options['cor_feature_selection_threshold'],
    'convert_categorical': perturb_options['convert_categorical']
    }).melt(id_vars=['max_identical_thresh', 'n_neighborhoods', 'impute_missing_categorical', 'simplify_vars', 'transform_response', 'cor_feature_selection_threshold', 'convert_categorical'], 
            var_name='model', 
            value_name='mae')


# %% [markdown]
# 
# ## Approach 1: Choosing a single predictive fit using PCS
# 
# Having computed the performance of each of our judgment-call perturbed fits for each algorithm we considered in this book, we can then identify which fit yields the "best" performance.
# 
# The following code prints the details of the fits with the highest correlation performance:

# %%
perturbed_jc_corr.sort_values(by='corr', ascending=False).head(6)

# %% [markdown]
# 
# Then we can print the details of the fits with the lowest rMSE (best performance):

# %%
ames_top_rmse = perturbed_jc_rmse.sort_values(by='rmse')
ames_top_rmse.head(6)

# %% [markdown]
# 
# And lastly, we can print the details of the fits with the lowest MAE (best performance):

# %%
perturbed_jc_mae.sort_values(by='mae', ascending=True).head(6)

# %% [markdown]
# 
# The "best" fit in terms of the correlation measure is the LAD algorithm with the following cleaning/pre-processing judgment call options:
# 
# - `max_identical_thresh = 0.95`
# 
# - `n_neighborhoods = 20`
# 
# - `impute_missing_categorical = "mode"`
# 
# - `simplify_vars = FALSE`
# 
# - `transform_response = "log"`
# 
# - `cor_feature_selection_threshold = 0`
# 
# - `convert_categorical = "dummy"`
# 
# (Note that these results are slightly different to the R/book version which had the square-root transformation and the dummy coding of categorical variables, but this R/book version is second-best in terms of correlation here with very little difference overall.)
# 
# The "best" fit in terms of the rMSE measure has mostly the same set of judgment calls, but involves the LAD algorithm instead of the LS algorithm and the square-root transformation instead of the logarithmic transformation, and the "best" fit in terms of the MAE involves the simplified dummy encoding of the categorical variables.
# 
# Since the rMSE and MAE measures are slightly more precise than the correlation algorithm, we will use the **LAD algorithm trained on the training set with the particular cleaning/pre-processing judgment calls aligned with the book's version: LAD with the square-root transformation and dummy encoding of categorical variables.**
# 

# %%
ames_train_preprocessed_selected = preprocess_ames_data(ames_train_clean,
                                                        max_identical_thresh=0.95,
                                                        n_neighborhoods=20,
                                                        impute_missing_categorical='mode',
                                                        simplify_vars=False,
                                                        transform_response='sqrt',
                                                        cor_feature_selection_threshold=0,
                                                        convert_categorical='dummy',
                                                        use_top_features=args.use_top_features, # Pass arg
                                                        keep_pid=False # No need to keep pid for training
                                                        )

# No more feature subsetting logic here - removed

single_fit = LADRegression()
single_fit.fit(X=ames_train_preprocessed_selected.drop(columns='saleprice'), y=ames_train_preprocessed_selected['saleprice'])

# %% [markdown]
# 
# ### Test set evaluation
# 
# Let's then evaluate this final fit using the test set (since our validation set was used to choose it, it can no longer provide an independent assessment of its performance).
# 
# First we must create the relevant pre-processed test set.

# %%
# Extract relevant neighborhoods from the training data
train_neighborhood_cols = ames_train_preprocessed_selected.filter(regex="neighborhood").columns
train_neighborhoods = [x.replace("neighborhood_", "") for x in train_neighborhood_cols]
    
# Pre-process the test set using exactly the same parameters and feature selection as training set
print("Creating test dataset with exact same parameters as training dataset...")
ames_test_preprocessed_selected = preprocess_ames_data(ames_test_clean,
                                                        max_identical_thresh=0.95,
                                                        n_neighborhoods=20,
                                                        impute_missing_categorical='mode',
                                                        simplify_vars=False,
                                                        transform_response='sqrt',
                                                        cor_feature_selection_threshold=0,
                                                        convert_categorical='dummy',
                                                        use_top_features=args.use_top_features, # Pass arg
                                                        neighborhood_levels=train_neighborhoods,
                                                        column_selection=list(ames_train_preprocessed_selected.columns),
                                                        keep_pid=True # Keep pid for test set
                                                        )

# No more feature subsetting logic here - removed

# %% [markdown]
# And then we can compute the predictions for the test set and evaluate them.

# %%
ames_test_pred = single_fit.predict(X=ames_test_preprocessed_selected.drop(columns='saleprice'))

# %%
# print out the correlation performance
corr = np.corrcoef(ames_test_preprocessed_selected['saleprice']**2, ames_test_pred**2)[0,1]
rmse = np.sqrt(np.mean((ames_test_preprocessed_selected['saleprice']**2 - ames_test_pred**2)**2))
mae = np.mean(np.abs(ames_test_preprocessed_selected['saleprice']**2 - ames_test_pred**2))

# print out the results
print("Correlation performance:", round(corr, 3))
print("RMSE:", round(rmse, 2))
print("MAE:", round(mae, 2))



# %% [markdown]
# 
# 
# The correlation of the predicted and true test set sale prices are very high. The rMSE and MAE both indicate that the typical sale price error is less than \$20,000.

# %% [markdown]
# ## Approach 2: PCS ensemble prediction 
# 
# In this approach, we take a look at all of the predictions that we computed above (across all algorithms and judgment call combinations), and we first conduct a predictability screening test to ensure that we are not using particularly poorly performing fits to create our ensemble.
# 
# Let's visualize the distribution of the correlation performance measure across all of the algorithms and cleaning/pre-processing judgment calls (grouping by algorithm) using boxplots:

# %%
px.box(perturbed_jc_rmse, x='model', y='rmse')

# %% [markdown]
# We can also look at the distributions of the judgment calls grouping by the judgment call options, such as the response transformation:

# %%
px.box(perturbed_jc_rmse, x='transform_response', y='rmse')

# %% [markdown]
# 
# Note that the log- and square root-transformations are much more accurate in general than the fits with the untransformed response ("none") (but there are still some fits with the untransformed response that perform quite well).

# %% [markdown]
# 
# A histogram below shows the overall distribution:

# %%
px.histogram(perturbed_jc_rmse, x='rmse')

# %% [markdown]
# 
# When it comes to an ensemble fit, generally if you have a range of performance measures, you will be able to generate more accurate response predictions if you filter to just the best performing fits. Let's thus conduct a fairly arbitrary **predictability screening test of that requires a validation set correlation performance of at least 0.94**, which will filter our just some of the worse performing untransformed response fits. An ensemble prediction can then be computed based on the average prediction based on the fits that remain.

# %% [markdown]
# 
# ### Test set evaluation
# 
# To evaluate the ensemble, let's compute the ensemble predictions for each of the *test set* data points using just the fits that passed the predictability screening test.

# %% [markdown]
# Let's first compute all of the predictions for each test set house

# %%
# compute the predictions on the test set for ls_area_perturbed, ls_multi_perturbed, ls_all_perturbed, cart_perturbed, and rf_perturbed
ls_test_jc_pred_perturbed = [ls_jc_perturbed[i].predict(X=ames_test_jc_perturb[i].drop(columns='saleprice')) for i in range(len(ames_test_jc_perturb))]
lad_test_jc_pred_perturbed = [lad_jc_perturbed[i].predict(X=ames_test_jc_perturb[i].drop(columns='saleprice')) for i in range(len(ames_test_jc_perturb))]
ridge_test_jc_pred_perturbed = [ridge_jc_perturbed[i].predict(X=ames_test_jc_perturb_std[i].drop(columns='saleprice')) for i in range(len(ames_test_jc_perturb_std))]
lasso_test_jc_pred_perturbed = [lasso_jc_perturbed[i].predict(X=ames_test_jc_perturb_std[i].drop(columns='saleprice')) for i in range(len(ames_test_jc_perturb_std))]
rf_test_jc_pred_perturbed = [rf_jc_perturbed[i].predict(X=ames_test_jc_perturb[i].drop(columns='saleprice')) for i in range(len(ames_test_jc_perturb))]

# for predictions where the response was log-transformed, undo the log transformation
ls_test_jc_pred_perturbed = [np.exp(pred) if perturb_options['transform_response'][i] == 'log' else pred for i, pred in enumerate(ls_test_jc_pred_perturbed)]
lad_test_jc_pred_perturbed = [np.exp(pred) if perturb_options['transform_response'][i] == 'log' else pred for i, pred in enumerate(lad_test_jc_pred_perturbed)]
ridge_test_jc_pred_perturbed = [np.exp(pred) if perturb_options['transform_response'][i] == 'log' else pred for i, pred in enumerate(ridge_test_jc_pred_perturbed)]
lasso_test_jc_pred_perturbed = [np.exp(pred) if perturb_options['transform_response'][i] == 'log' else pred for i, pred in enumerate(lasso_test_jc_pred_perturbed)]
rf_test_jc_pred_perturbed = [np.exp(pred) if perturb_options['transform_response'][i] == 'log' else pred for i, pred in enumerate(rf_test_jc_pred_perturbed)]

# for predictions where the response was sqrt-transformed, undo the sqrt transformation
ls_test_jc_pred_perturbed = [pred**2 if perturb_options['transform_response'][i] == 'sqrt' else pred for i, pred in enumerate(ls_test_jc_pred_perturbed)]
lad_test_jc_pred_perturbed = [pred**2 if perturb_options['transform_response'][i] == 'sqrt' else pred for i, pred in enumerate(lad_test_jc_pred_perturbed)]
ridge_test_jc_pred_perturbed = [pred**2 if perturb_options['transform_response'][i] == 'sqrt' else pred for i, pred in enumerate(ridge_test_jc_pred_perturbed)]
lasso_test_jc_pred_perturbed = [pred**2 if perturb_options['transform_response'][i] == 'sqrt' else pred for i, pred in enumerate(lasso_test_jc_pred_perturbed)]
rf_test_jc_pred_perturbed = [pred**2 if perturb_options['transform_response'][i] == 'sqrt' else pred for i, pred in enumerate(rf_test_jc_pred_perturbed)]


# %% [markdown]
# Let's next identify which fits correspond to the top 10% of fits according to the rMSE.

# %%
# identify the 10th quantile of rMSE values across all perturbed models 
rmse_screening_threshold = perturbed_jc_rmse['rmse'].quantile(0.1)
# identify which models are in the top 10% rMSE performance band (i.e. have the lowest rMSE values)
ls_rmse_top10p = perturbed_jc_rmse.query('model == "ls"')['rmse'] <= rmse_screening_threshold
lad_rmse_top10p = perturbed_jc_rmse.query('model == "lad"')['rmse'] <= rmse_screening_threshold
ridge_rmse_top10p = perturbed_jc_rmse.query('model == "ridge"')['rmse'] <= rmse_screening_threshold
lasso_rmse_top10p = perturbed_jc_rmse.query('model == "lasso"')['rmse'] <= rmse_screening_threshold
rf_rmse_top10p = perturbed_jc_rmse.query('model == "rf"')['rmse'] <= rmse_screening_threshold

# %% [markdown]
# We can then compute the ensemble predictions using the average of all of the screened predictions across all of the models and we then aggregate the predictions from each fit into a single DataFrame

# %%
# define a concat function that can handle the cases where no models are selected
def concat_nonempty(df_list):
    if df_list == []:
        return pd.DataFrame()   
    else:
        return pd.concat([x for x in df_list if not x.empty])

# LS model
ls_test_jc_pred_perturbed_screened_df = [pd.DataFrame({
    'pred': ls_test_jc_pred_perturbed[i], 
    'model': 'ls',
    'house': list(range(len(ls_test_jc_pred_perturbed[i]))),
    'perturb_index': i
}) for i, include in enumerate(ls_rmse_top10p) if include]
ls_test_jc_pred_perturbed_screened_df = concat_nonempty(ls_test_jc_pred_perturbed_screened_df)

# LAD model
lad_test_jc_pred_perturbed_screened_df = [pd.DataFrame({
    'pred': lad_test_jc_pred_perturbed[i], 
    'model': 'lad',
    'house': list(range(len(ls_test_jc_pred_perturbed[i]))),
    'perturb_index': i
}) for i, include in enumerate(lad_rmse_top10p) if include]
lad_test_jc_pred_perturbed_screened_df = concat_nonempty(lad_test_jc_pred_perturbed_screened_df)

# Ridge model
ridge_test_jc_pred_perturbed_screened_df = [pd.DataFrame({
    'pred': ridge_test_jc_pred_perturbed[i], 
    'model': 'ridge',
    'house': list(range(len(ls_test_jc_pred_perturbed[i]))),
    'perturb_index': i
}) for i, include in enumerate(ridge_rmse_top10p) if include]
ridge_test_jc_pred_perturbed_screened_df = concat_nonempty(ridge_test_jc_pred_perturbed_screened_df)

# Lasso model
lasso_test_jc_pred_perturbed_screened_df = [pd.DataFrame({
    'pred': lasso_test_jc_pred_perturbed[i], 
    'model': 'lasso',
    'house': list(range(len(ls_test_jc_pred_perturbed[i]))),
    'perturb_index': i
}) for i, include in enumerate(lasso_rmse_top10p) if include]
lasso_test_jc_pred_perturbed_screened_df = concat_nonempty(lasso_test_jc_pred_perturbed_screened_df)

# Random Forest model
rf_test_jc_pred_perturbed_screened_df = [pd.DataFrame({
    'pred': rf_test_jc_pred_perturbed[i], 
    'model': 'rf',
    'house': list(range(len(ls_test_jc_pred_perturbed[i]))),
    'perturb_index': i
}) for i, include in enumerate(rf_rmse_top10p) if include]
rf_test_jc_pred_perturbed_screened_df = concat_nonempty(rf_test_jc_pred_perturbed_screened_df)


# %%
# aggregate the predictions from all of the models into a single data frame
test_jc_pred_perturbed_screened_df = pd.concat([
    ls_test_jc_pred_perturbed_screened_df, 
    lad_test_jc_pred_perturbed_screened_df,
    ridge_test_jc_pred_perturbed_screened_df,
    lasso_test_jc_pred_perturbed_screened_df,
    rf_test_jc_pred_perturbed_screened_df
    ])
test_jc_pred_perturbed_screened_df


# %% [markdown]
# Since this contains all of the test set predictions for all fits that passed the predictability screening test, we can then compute the ensemble predictions for each house using the average of all of the screened predictions across all of the models:

# %%
pred_ensemble_test = test_jc_pred_perturbed_screened_df.groupby('house')['pred'].mean()

# %% [markdown]
# We can then compute the performance of the ensemble predictions using the regular metrics

# %%
# compute the correlation, rmse, and mae of the ensemble predictions for the test set
test_ensemble_rmse = np.sqrt(np.mean((ames_test_preprocessed['saleprice'].values - pred_ensemble_test)**2))
test_ensemble_mae = np.mean(np.abs(ames_test_preprocessed['saleprice'].values - pred_ensemble_test))
test_ensemble_corr = np.corrcoef(ames_test_preprocessed['saleprice'].values, pred_ensemble_test)[0,1]


# %%
# print out the results
print("Correlation performance:", round(test_ensemble_corr, 3))
print("RMSE:", round(test_ensemble_rmse, 2))
print("MAE:", round(test_ensemble_mae, 2))

# %% [markdown]
# According to all three measures, the ensemble performance on the test set is slightly worse than the single-best fit performance.
# 
# 

# %% [markdown]
# 
# ## Approach 3: Calibrated PCS perturbation prediction intervals (and comparison with CQR/CLEAR)
# 
# 
# The process for computing the perturbation prediction intervals (PPIs) is similar to the ensemble prediction process, but instead of averaging the fits that pass the predictability screening test, we instead compute an interval from them. Since we want to include fits from data perturbations (bootstrap samples) when we compute the intervals, we will need to re-fit our algorithms on 10 bootstrapped versions of each of our fits that passed the (top 10% rMSE) screening stage above.
# 
# Note that only the LS and LAD fits make it into the top 10% of fits in terms of rMSE, so we will only use these two algorithms in the PPIs/epistemic component.
# 
# These bootstrapped models represent the epistemic uncertainty - uncertainty in the model itself.
# Later, we'll compare this approach with the CLEAR method, which combines both epistemic and aleatoric uncertainty.

# %%
# how many LS fits made it into the top 10%?
sum(ls_rmse_top10p)

# %%
# extract just the perturbed datasets that are in the top 10% of RMSE performance for the LS model
ames_jc_perturbed_screened_ls = list(compress(ames_jc_perturb, ls_rmse_top10p))
# fit a linear regression model to each of these perturbed datasets using 10 bootstrap samples
ls_jc_perturbed_screened_boot_fit = []
ls_orig_indices = [] # keep track of which original dataset each bootstrapped model came from
for i in range(10):
    # compute a bootstrapped version of each relevant dataset
    ames_jc_perturbed_screened_ls_boot = [df.sample(n = df.shape[0], replace=True) for df in ames_jc_perturbed_screened_ls]
    # fit the LS model to each bootstrapped dataset
    for j, df in enumerate(ames_jc_perturbed_screened_ls_boot):
        ls_jc_perturbed_screened_boot_fit.append(
            LinearRegression().fit(X=df.drop(columns='saleprice'), y=df['saleprice'])
        )
        # Store the index of the original dataset in the perturbed list
        orig_idx = np.where(ls_rmse_top10p)[0][j]
        ls_orig_indices.append(orig_idx)

# %%
# how many LAD fits made it into the top 10%?
sum(lad_rmse_top10p)

# %%
# extract just the perturbed datasets that are in the top 10% of RMSE performance for the LAD model
ames_jc_perturbed_screened_lad = list(compress(ames_jc_perturb, lad_rmse_top10p))
# fit a linear regression model to each of these perturbed datasets using 10 bootstrap samples
lad_jc_perturbed_screened_boot_fit = []
lad_orig_indices = [] # keep track of which original dataset each bootstrapped model came from
for i in range(10):
    # compute a bootstrapped version of each relevant dataset
    ames_jc_perturbed_screened_lad_boot = [df.sample(n = df.shape[0], replace=True) for df in ames_jc_perturbed_screened_lad]
    # fit the LAD model to each bootstrapped dataset
    for j, df in enumerate(ames_jc_perturbed_screened_lad_boot):
        lad_jc_perturbed_screened_boot_fit.append(
            LADRegression().fit(X=df.drop(columns='saleprice'), y=df['saleprice'])
        )
        # Store the index of the original dataset in the perturbed list
        orig_idx = np.where(lad_rmse_top10p)[0][j]
        lad_orig_indices.append(orig_idx)

# %%
# how many Ridge fits made it into the top 10%?
sum(ridge_rmse_top10p)

# %%
# how many Lasso fits made it into the top 10%?
sum(lasso_rmse_top10p)

# %%
# how many RF fits made it into the top 10%?
sum(rf_rmse_top10p)

# %% [markdown]
# Next, we need to generate predictions for each of the validation set houses using each of these bootstrapped perturbed fits that passed the screening test.

# %%
# Generate predictions from bootstrapped LS models for validation data
ls_val_jc_perturbed_screened_boot_pred = []
for i, model in enumerate(ls_jc_perturbed_screened_boot_fit):
    # Get the corresponding validation dataset
    val_data = ames_val_jc_perturb[ls_orig_indices[i]]
    # Generate predictions
    preds = model.predict(val_data.drop(columns='saleprice'))
    # Handle transformations (log or sqrt)
    if perturb_options['transform_response'][ls_orig_indices[i]] == 'log':
        preds = np.exp(preds)
    elif perturb_options['transform_response'][ls_orig_indices[i]] == 'sqrt':
        preds = preds ** 2
    ls_val_jc_perturbed_screened_boot_pred.append(preds)

# Generate predictions from bootstrapped LAD models for validation data
lad_val_jc_perturbed_screened_boot_pred = []
for i, model in enumerate(lad_jc_perturbed_screened_boot_fit):
    # Get the corresponding validation dataset
    val_data = ames_val_jc_perturb[lad_orig_indices[i]]
    # Generate predictions
    preds = model.predict(val_data.drop(columns='saleprice'))
    # Handle transformations (log or sqrt)
    if perturb_options['transform_response'][lad_orig_indices[i]] == 'log':
        preds = np.exp(preds)
    elif perturb_options['transform_response'][lad_orig_indices[i]] == 'sqrt':
        preds = preds ** 2
    lad_val_jc_perturbed_screened_boot_pred.append(preds)

# %%
# Organize predictions for validation data in a dataframe
val_pred_intervals = []
# Make sure validation data has reset indices to avoid KeyError
ames_val_preprocessed_indexed = ames_val_preprocessed.reset_index(drop=True)

for house_id in range(len(ames_val_preprocessed_indexed)):
    house_preds = []
    # Add all LS predictions for this house
    for pred_set in ls_val_jc_perturbed_screened_boot_pred:
        house_preds.append(pred_set[house_id])
    # Add all LAD predictions for this house
    for pred_set in lad_val_jc_perturbed_screened_boot_pred:
        house_preds.append(pred_set[house_id])
    
    # Calculate intervals
    val_pred_intervals.append({
        'pid': house_id,
        'true_val': ames_val_preprocessed_indexed['saleprice'].iloc[house_id],
        'median_pred': np.median(house_preds),
        'q05': np.quantile(house_preds, 0.05),
        'q95': np.quantile(house_preds, 0.95)
    })

# Convert to DataFrame
val_pred_intervals_df = pd.DataFrame(val_pred_intervals)

# %%
# Calculate uncalibrated coverage
val_pred_intervals_df['covered'] = (val_pred_intervals_df['true_val'] >= val_pred_intervals_df['q05']) & \
                                   (val_pred_intervals_df['true_val'] <= val_pred_intervals_df['q95'])
val_coverage = val_pred_intervals_df['covered'].mean()
print(f"Validation uncalibrated coverage: {val_coverage:.3f}")

# %% [markdown]
# This is unfortunately far lower than the 90% coverage that we were aiming for! But that's ok, because we can compute *calibrated* intervals based on the median prediction, of the form:
#
# [median - γ(median - q0.05), median + γ(q0.95 - median)]
#
# where the constant γ is chosen so that the calibrated interval will have a coverage of 0.9.
#
# We will calculate this gamma dynamically based on the validation set.

# %%
# --- Dynamic Gamma Calculation for PCS-PPI ---
# Define target coverage
target_coverage = 0.90
alpha_coverage = 1 - target_coverage

# Function to calculate coverage for a given gamma
def calculate_coverage(gamma_val, intervals_df):
    lower_calib = intervals_df['median_pred'] - gamma_val * (intervals_df['median_pred'] - intervals_df['q05'])
    upper_calib = intervals_df['median_pred'] + gamma_val * (intervals_df['q95'] - intervals_df['median_pred'])
    covered = (intervals_df['true_val'] >= lower_calib) & (intervals_df['true_val'] <= upper_calib)
    return covered.mean()

# Find the gamma that achieves target coverage on the validation set
# This uses a simple search, more sophisticated methods exist but this should be sufficient
gamma_candidates = np.logspace(-1, 2, 100) # Search between 0.1 and 100
coverages = [calculate_coverage(g, val_pred_intervals_df) for g in gamma_candidates]
# Find the gamma closest to the target coverage
gamma_pcs_ppi = gamma_candidates[np.argmin(np.abs(np.array(coverages) - target_coverage))]

print(f"Dynamically calculated PCS-PPI gamma: {gamma_pcs_ppi:.4f}")
# --- End Dynamic Gamma Calculation ---


# %%
# Apply the dynamically calculated gamma for calibration
val_pred_intervals_df['q05_calibrated'] = val_pred_intervals_df['median_pred'] - \
                                          gamma_pcs_ppi * (val_pred_intervals_df['median_pred'] - val_pred_intervals_df['q05'])
val_pred_intervals_df['q95_calibrated'] = val_pred_intervals_df['median_pred'] + \
                                          gamma_pcs_ppi * (val_pred_intervals_df['q95'] - val_pred_intervals_df['median_pred'])

# %%
# Calculate calibrated coverage
val_pred_intervals_df['covered_calibrated'] = (val_pred_intervals_df['true_val'] >= val_pred_intervals_df['q05_calibrated']) & \
                                             (val_pred_intervals_df['true_val'] <= val_pred_intervals_df['q95_calibrated'])
val_calibrated_coverage = val_pred_intervals_df['covered_calibrated'].mean()
print(f"Validation calibrated coverage: {val_calibrated_coverage:.3f}")

# %% [markdown]
# It seems like a constant value of γ = 2.04 yields calibrated intervals with coverage of approximately 0.9 (or 90%), as shown above.
#
# We can visualize the calibrated intervals using a prediction stability plot:

# %%
# import random
# random.seed(3864)
# Sample some validation points to visualize
# sample_val_indices = random.sample(range(len(val_pred_intervals_df)), 150)
# sample_intervals = val_pred_intervals_df.iloc[sample_val_indices].copy()

# # Create a plot
# fig = go.Figure()

# # Add interval segments
# for idx, row in sample_intervals.iterrows():
#     fig.add_trace(go.Scatter(
#         x=[row['q05_calibrated'], row['q95_calibrated']],
#         y=[row['true_val'], row['true_val']],
#         mode='lines',
#         line=dict(color='blue' if row['covered_calibrated'] else 'red'),
#         showlegend=False
#     ))

# # Add diagonal line
# max_val = max(sample_intervals['true_val'].max(), sample_intervals['q95_calibrated'].max())
# min_val = min(sample_intervals['true_val'].min(), sample_intervals['q05_calibrated'].min())
# fig.add_trace(go.Scatter(
#     x=[min_val, max_val],
#     y=[min_val, max_val],
#     mode='lines',
#     line=dict(color='black', dash='dash'),
#     showlegend=False
# ))

# fig.update_layout(
#     title='Prediction Stability Plot with Calibrated Intervals',
#     xaxis_title='Predicted sale price',
#     yaxis_title='Observed sale price'
# )

# fig.show()

# %% [markdown]
# ### Test set evaluation
#
# Finally, using our calibration constant that we computed using the validation set, we can generate calibrated prediction perturbation intervals for our test set houses and compute the coverage of the intervals.

# %%
# Generate predictions from bootstrapped LS models for test data
ls_test_jc_perturbed_screened_boot_pred = []
for i, model in enumerate(ls_jc_perturbed_screened_boot_fit):
    # Get the corresponding test dataset
    test_data = ames_test_jc_perturb[ls_orig_indices[i]]
    # Generate predictions
    preds = model.predict(test_data.drop(columns='saleprice'))
    # Handle transformations (log or sqrt)
    if perturb_options['transform_response'][ls_orig_indices[i]] == 'log':
        preds = np.exp(preds)
    elif perturb_options['transform_response'][ls_orig_indices[i]] == 'sqrt':
        preds = preds ** 2
    ls_test_jc_perturbed_screened_boot_pred.append(preds)

# Generate predictions from bootstrapped LAD models for test data
lad_test_jc_perturbed_screened_boot_pred = []
for i, model in enumerate(lad_jc_perturbed_screened_boot_fit):
    # Get the corresponding test dataset
    test_data = ames_test_jc_perturb[lad_orig_indices[i]]
    # Generate predictions
    preds = model.predict(test_data.drop(columns='saleprice'))
    # Handle transformations (log or sqrt)
    if perturb_options['transform_response'][lad_orig_indices[i]] == 'log':
        preds = np.exp(preds)
    elif perturb_options['transform_response'][lad_orig_indices[i]] == 'sqrt':
        preds = preds ** 2
    lad_test_jc_perturbed_screened_boot_pred.append(preds)

# %%
# Organize predictions for test data in a dataframe
test_pred_intervals = []
# Make sure test data has reset indices to avoid KeyError
ames_test_preprocessed_indexed = ames_test_preprocessed.reset_index(drop=True)

for house_id in range(len(ames_test_preprocessed_indexed)):
    house_preds = []
    # Add all LS predictions for this house
    for pred_set in ls_test_jc_perturbed_screened_boot_pred:
        house_preds.append(pred_set[house_id])
    # Add all LAD predictions for this house
    for pred_set in lad_test_jc_perturbed_screened_boot_pred:
        house_preds.append(pred_set[house_id])
    
    # Calculate intervals
    test_pred_intervals.append({
        'pid': house_id,
        'true_test': ames_test_preprocessed_indexed['saleprice'].iloc[house_id],
        'median_pred_test': np.median(house_preds),
        'q05': np.quantile(house_preds, 0.05),
        'q95': np.quantile(house_preds, 0.95),
        # Apply the calculated gamma_pcs_ppi here for test set intervals
        'q05_calibrated': np.median(house_preds) - gamma_pcs_ppi * (np.median(house_preds) - np.quantile(house_preds, 0.05)),
        'q95_calibrated': np.median(house_preds) + gamma_pcs_ppi * (np.quantile(house_preds, 0.95) - np.median(house_preds))
    })

# Convert to DataFrame
test_pred_intervals_df = pd.DataFrame(test_pred_intervals)

# %%
# Calculate test set coverage
test_pred_intervals_df['covered'] = (test_pred_intervals_df['true_test'] >= test_pred_intervals_df['q05_calibrated']) & \
                                   (test_pred_intervals_df['true_test'] <= test_pred_intervals_df['q95_calibrated'])
test_calibrated_coverage = test_pred_intervals_df['covered'].mean()
print(f"Test set calibrated coverage: {test_calibrated_coverage:.3f}")

# %% [markdown]
# The test set coverage is fairly close to 90%, which is what we were hoping to see.

# %%
# Calculate width of the intervals
test_pred_intervals_df['width'] = test_pred_intervals_df['q95_calibrated'] - test_pred_intervals_df['q05_calibrated']
mean_width = test_pred_intervals_df['width'].mean()
print(f"Mean interval width: ${mean_width:.2f}")

# If you uncomment the visualization code, use this seed to match the R version
# np.random.seed(3864)
# random.seed(3864)
# Sample some validation points to visualize
# sample_val_indices = random.sample(range(len(val_pred_intervals_df)), 150)
# sample_intervals = val_pred_intervals_df.iloc[sample_val_indices].copy()

# # Create a plot
# fig = go.Figure()

# %% [markdown]
# ### Additive calibration 
#
# As an alternative to the multiplicative calibration above, we could also consider additive calibration of the form:
#
# [median - (q0.05 - median + c), median + (q0.95 - median + c)]
#
# where c is a constant that is chosen to achieve the desired coverage.

# %% [markdown]
# ### Prepare Data and Models for Interval Methods

# We need the preprocessed training, validation, and test sets corresponding to the 
# *single best* judgment call combination identified earlier.
# We also need the bootstrapped LS and LAD models fitted on the screened judgment call perturbations.

# Set the selected transformation directly since we're using this specific transformation
# in all our training, validation, and test data
selected_transform = 'sqrt'
print(f"Using transformation: {selected_transform}")

def inverse_transform(preds, transform_type):
    if transform_type == 'log':
        return np.exp(preds)
    elif transform_type == 'sqrt':
        return preds**2
    else:
        return preds

# Prepare training, validation, and test data (already created and subsetted above)
X_train_final = ames_train_preprocessed_selected.drop(columns='saleprice')
y_train_final = ames_train_preprocessed_selected['saleprice']

X_test_final = ames_test_preprocessed_selected.drop(columns='saleprice')
# Get the true test y values *before* potential transformation for evaluation
# Check which column name is actually in the dataframe
if 'sale_price' in ames_test_clean.columns:
    y_test_true_orig = ames_test_clean.loc[ames_test_preprocessed_selected.index, 'sale_price']
elif 'saleprice' in ames_test_clean.columns:
    y_test_true_orig = ames_test_clean.loc[ames_test_preprocessed_selected.index, 'saleprice']
else:
    # Print available columns to help diagnose the issue
    print("Available columns in ames_test_clean:", ames_test_clean.columns.tolist())
    raise KeyError("Neither 'sale_price' nor 'saleprice' column found in ames_test_clean")

# --- REMOVE THIS BLOCK ---
# Find the corresponding validation set for calibration
# # We need the validation set that matches the `ames_train_preprocessed_selected` settings
# selected_perturb_idx = perturb_options.loc[
#     (perturb_options['max_identical_thresh'] == 0.95) &
#     (perturb_options['n_neighborhoods'] == 20) &
#     (perturb_options['impute_missing_categorical'] == 'mode') &
#     (perturb_options['simplify_vars'] == False) &
#     (perturb_options['transform_response'] == 'sqrt') &
#     (perturb_options['cor_feature_selection_threshold'] == 0) &
#     (perturb_options['convert_categorical'] == 'dummy')
# ].index[0]
# 
# ames_val_preprocessed_selected = ames_val_jc_perturb[selected_perturb_idx]
# # No more feature subsetting logic here - removed
# # Re-run preprocessing for the selected validation set to ensure correct features if use_top_features is True
# # and correct keep_pid status
# ames_val_preprocessed_selected = preprocess_ames_data(ames_val_clean,
#                                         max_identical_thresh=perturb_options['max_identical_thresh'][selected_perturb_idx],
#                                         n_neighborhoods=perturb_options['n_neighborhoods'][selected_perturb_idx],
#                                         impute_missing_categorical=perturb_options['impute_missing_categorical'][selected_perturb_idx],
#                                         simplify_vars=perturb_options['simplify_vars'][selected_perturb_idx],
#                                         transform_response=perturb_options['transform_response'][selected_perturb_idx],
#                                         cor_feature_selection_threshold=perturb_options['cor_feature_selection_threshold'][selected_perturb_idx],
#                                         convert_categorical=perturb_options['convert_categorical'][selected_perturb_idx],
#                                         use_top_features=args.use_top_features, # Pass arg
#                                         column_selection=list(ames_jc_perturb[selected_perturb_idx].columns),
#                                         neighborhood_levels=train_neighborhoods, # Use neighborhoods from selected training set
#                                         keep_pid=True # Keep pid for validation
#                                         )
# --- END REMOVE THIS BLOCK ---

# Define X_val_final and y_val_final by creating a new validation dataset with the exact same parameters
# as the training dataset (ames_train_preprocessed_selected)
print("Creating validation dataset with exact same parameters as training dataset...")

# Extract relevant neighborhoods from training data for consistency
train_neighborhood_cols = list(ames_train_preprocessed_selected.filter(regex="neighborhood").columns)
train_neighborhoods = [x.replace("neighborhood_", "") for x in train_neighborhood_cols]

# Create preprocessed validation set with EXACT same parameters as training
ames_val_preprocessed_selected = preprocess_ames_data(ames_val_clean,
                                                     max_identical_thresh=0.95,
                                                     n_neighborhoods=20,
                                                     impute_missing_categorical='mode',
                                                     simplify_vars=False,
                                                     transform_response='sqrt',
                                                     cor_feature_selection_threshold=0,
                                                     convert_categorical='dummy',
                                                     use_top_features=args.use_top_features,  # Pass the same flag
                                                     neighborhood_levels=train_neighborhoods,
                                                     column_selection=list(ames_train_preprocessed_selected.columns),
                                                     keep_pid=True  # Keep pid for validation
                                                     )

X_val_final = ames_val_preprocessed_selected.drop(columns='saleprice')
y_val_final = ames_val_preprocessed_selected['saleprice']
# Get the true validation y values *before* potential transformation
# Check which column name is actually in the dataframe
if 'sale_price' in ames_val_clean.columns:
    y_val_true_orig = ames_val_clean.loc[ames_val_preprocessed_selected.index, 'sale_price']
elif 'saleprice' in ames_val_clean.columns:
    y_val_true_orig = ames_val_clean.loc[ames_val_preprocessed_selected.index, 'saleprice']
else:
    # Print available columns to help diagnose the issue
    print("Available columns in ames_val_clean:", ames_val_clean.columns.tolist())
    raise KeyError("Neither 'sale_price' nor 'saleprice' column found in ames_val_clean")

# %% [markdown]
# ### Bootstrapped Epistemic Models (LS and LAD)

# %%

# Identify top 10% performing LS/LAD fits based on validation RMSE
rmse_screening_threshold = perturbed_jc_rmse['rmse'].quantile(0.1)
ls_rmse_top10p = perturbed_jc_rmse.query('model == "ls"')['rmse'] <= rmse_screening_threshold
lad_rmse_top10p = perturbed_jc_rmse.query('model == "lad"')['rmse'] <= rmse_screening_threshold

# --- Re-fit bootstrapped LS and LAD models (Epistemic Component) ---
print("Fitting bootstrapped epistemic models (LS/LAD)...")
# Use fewer bootstraps for testing if needed
n_bootstraps_epistemic = 10  # Default to 10 bootstraps
print(f"Using {n_bootstraps_epistemic} epistemic bootstraps")

# Set random seed for bootstrap sampling to match R version (1789)
np.random.seed(1789)
random.seed(1789)

ls_jc_perturbed_screened_ls = list(compress(ames_jc_perturb, ls_rmse_top10p))
lad_jc_perturbed_screened_lad = list(compress(ames_jc_perturb, lad_rmse_top10p))

# LS Models
print("Fitting bootstrapped LS models...")
ls_jc_perturbed_screened_boot_fit = []
ls_orig_indices = []
for i in range(n_bootstraps_epistemic):
    print(f"  LS Bootstrap {i+1}/{n_bootstraps_epistemic}")
    ames_jc_perturbed_screened_ls_boot = [df.sample(n = df.shape[0], replace=True) for df in ls_jc_perturbed_screened_ls]
    for j, df_boot in enumerate(ames_jc_perturbed_screened_ls_boot):
        # No more feature subsetting logic here - removed
        ls_jc_perturbed_screened_boot_fit.append(
            LinearRegression().fit(X=df_boot.drop(columns='saleprice'), y=df_boot['saleprice'])
        )
        orig_idx = np.where(ls_rmse_top10p)[0][j]
        ls_orig_indices.append(orig_idx)

# LAD Models
print("Fitting bootstrapped LAD models...")
lad_jc_perturbed_screened_boot_fit = []
lad_orig_indices = []
for i in range(n_bootstraps_epistemic):
    print(f"  LAD Bootstrap {i+1}/{n_bootstraps_epistemic}")
    ames_jc_perturbed_screened_lad_boot = [df.sample(n = df.shape[0], replace=True) for df in lad_jc_perturbed_screened_lad]
    for j, df_boot in enumerate(ames_jc_perturbed_screened_lad_boot):
        # No more feature subsetting logic here - removed
        lad_jc_perturbed_screened_boot_fit.append(
            LADRegression().fit(X=df_boot.drop(columns='saleprice'), y=df_boot['saleprice'])
        )
        orig_idx = np.where(lad_rmse_top10p)[0][j]
        lad_orig_indices.append(orig_idx)

print("Finished fitting bootstrapped models.")

# --- Generate Epistemic Predictions (Validation and Test) ---
print("Generating epistemic predictions...")

all_epistemic_train_preds = []
all_epistemic_val_preds = []
all_epistemic_test_preds = []

# LS Predictions
for i, model in enumerate(ls_jc_perturbed_screened_boot_fit):
    orig_idx = ls_orig_indices[i]
    transform_type = perturb_options['transform_response'][orig_idx]

    train_data = ames_jc_perturb[orig_idx] # <-- Add access to train data
    val_data = ames_val_jc_perturb[orig_idx]
    test_data = ames_test_jc_perturb[orig_idx]
    # Predict
    preds_train = model.predict(train_data.drop(columns='saleprice'))
    preds_val = model.predict(val_data.drop(columns='saleprice'))
    preds_test = model.predict(test_data.drop(columns='saleprice'))

    # Inverse transform
    all_epistemic_train_preds.append(inverse_transform(preds_train, transform_type)) # <-- Append train pred
    all_epistemic_val_preds.append(inverse_transform(preds_val, transform_type))
    all_epistemic_test_preds.append(inverse_transform(preds_test, transform_type))

# LAD Predictions
for i, model in enumerate(lad_jc_perturbed_screened_boot_fit):
    orig_idx = lad_orig_indices[i]
    transform_type = perturb_options['transform_response'][orig_idx]

    train_data = ames_jc_perturb[orig_idx]
    val_data = ames_val_jc_perturb[orig_idx]
    test_data = ames_test_jc_perturb[orig_idx]

    # No more feature subsetting logic here - removed

    # Predict
    preds_train = model.predict(train_data.drop(columns='saleprice'))
    preds_val = model.predict(val_data.drop(columns='saleprice'))
    preds_test = model.predict(test_data.drop(columns='saleprice'))

    # Inverse transform
    all_epistemic_train_preds.append(inverse_transform(preds_train, transform_type)) # <-- Append train pred
    all_epistemic_val_preds.append(inverse_transform(preds_val, transform_type))
    all_epistemic_test_preds.append(inverse_transform(preds_test, transform_type))

# Combine into arrays (n_models, n_samples)
all_epistemic_train_preds_arr = np.array(all_epistemic_train_preds) # <-- Define array
all_epistemic_val_preds_arr = np.array(all_epistemic_val_preds)
all_epistemic_test_preds_arr = np.array(all_epistemic_test_preds)

print(f"Epistemic train preds shape: {all_epistemic_train_preds_arr.shape}") # <-- Add print
print(f"Epistemic validation preds shape: {all_epistemic_val_preds_arr.shape}")
print(f"Epistemic test preds shape: {all_epistemic_test_preds_arr.shape}")

# Calculate epistemic bounds (median, q05, q95)
desired_coverage = 0.90 # Corresponds to 0.05 and 0.95 quantiles
alpha = 1 - desired_coverage
lower_q = alpha / 2
upper_q = 1 - alpha / 2

median_epistemic_train = np.median(all_epistemic_train_preds_arr, axis=0)
median_epistemic_val = np.median(all_epistemic_val_preds_arr, axis=0)
lower_epistemic_val = np.quantile(all_epistemic_val_preds_arr, lower_q, axis=0)
upper_epistemic_val = np.quantile(all_epistemic_val_preds_arr, upper_q, axis=0)
median_epistemic_test = np.median(all_epistemic_test_preds_arr, axis=0)
lower_epistemic_test = np.quantile(all_epistemic_test_preds_arr, lower_q, axis=0)
upper_epistemic_test = np.quantile(all_epistemic_test_preds_arr, upper_q, axis=0)

# --- Method 1: Original Calibrated PCS-PPI ---
print("\nCalculating original PCS-PPI intervals...")
# Apply the dynamically calculated gamma_pcs_ppi instead of hardcoded values
pcs_ppi_lower_val = median_epistemic_val - gamma_pcs_ppi * (median_epistemic_val - lower_epistemic_val)
pcs_ppi_upper_val = median_epistemic_val + gamma_pcs_ppi * (upper_epistemic_val - median_epistemic_val)
pcs_ppi_lower_test = median_epistemic_test - gamma_pcs_ppi * (median_epistemic_test - lower_epistemic_test)
pcs_ppi_upper_test = median_epistemic_test + gamma_pcs_ppi * (upper_epistemic_test - median_epistemic_test)

val_covered_pcs = (y_val_true_orig >= pcs_ppi_lower_val) & (y_val_true_orig <= pcs_ppi_upper_val)
print(f"PCS-PPI Validation Coverage (gamma={gamma_pcs_ppi:.4f}): {np.mean(val_covered_pcs):.3f}") # Use calculated gamma

# --- Calculate PCS Width ---
# Calculate the mean width of the PCS-PPI intervals
pcs_ppi_width = np.mean(pcs_ppi_upper_test - pcs_ppi_lower_test)
print(f"Mean PCS-PPI interval width: ${pcs_ppi_width:.2f}")

# --- Method 2: CQR Baseline ---
print("\nCalculating CQR baseline intervals...")
# Convert data to float and then to NumPy arrays to avoid pandas indexing issues with pygam
X_train_final = X_train_final.astype(float).values  # Convert to NumPy array
X_val_final = X_val_final.astype(float).values      # Convert to NumPy array
X_test_final = X_test_final.astype(float).values     # Convert to NumPy array
y_train_final = y_train_final.values # Ensure y is also NumPy
y_val_final = y_val_final.values

# --- Add Data Inspection ---
print("\nInspecting data before CQR baseline fit...")
print(f"X_train_final shape: {X_train_final.shape}")
print(f"y_train_final shape: {y_train_final.shape}")

# Check for NaNs
x_nans = np.isnan(X_train_final).sum().sum()
y_nans = np.isnan(y_train_final).sum()
print(f"NaNs in X_train_final: {x_nans}")
print(f"NaNs in y_train_final: {y_nans}")

# Check for Infs
x_infs = np.isinf(X_train_final).sum().sum()
y_infs = np.isinf(y_train_final).sum()
print(f"Infs in X_train_final: {x_infs}")
print(f"Infs in y_train_final: {y_infs}")

# Check for zero variance columns in X (NumPy array handling)
zero_var_cols_count = np.sum(np.std(X_train_final, axis=0) == 0)
print(f"Zero variance columns in X_train_final: {zero_var_cols_count}")
if zero_var_cols_count > 0:
     # You might want to identify which columns have zero variance if needed for debugging
    print(f"Warning: {zero_var_cols_count} columns with zero variance detected.")

if x_nans > 0 or y_nans > 0 or x_infs > 0 or y_infs > 0:
    print("WARNING: NaNs or Infs detected in data. This might cause issues with fitting.")
    # Optional: Add handling here, e.g., imputation or raising an error
    # For now, we'll just print the warning
# --- End Data Inspection ---


if args.aleatoric_model == 'gam':
    # Import pyGAM for quantile GAM
    from pygam import ExpectileGAM
    print("Fitting ExpectileGAM models for CQR...")
    # Use ExpectileGAM instead of QuantileRegressor
    alpha_lower = lower_q
    alpha_upper = upper_q

    # Fit GAM models
    try:
        gam_lower = ExpectileGAM(expectile=alpha_lower).gridsearch(X_train_final, y_train_final)
        gam_median = ExpectileGAM(expectile=0.5).gridsearch(X_train_final, y_train_final)
        gam_upper = ExpectileGAM(expectile=alpha_upper).gridsearch(X_train_final, y_train_final)
        print("Successfully fit ExpectileGAM models for CQR")

        # Predict using the GAM models
        pred_cqr_lower_val = inverse_transform(gam_lower.predict(X_val_final), selected_transform)
        pred_cqr_median_val = inverse_transform(gam_median.predict(X_val_final), selected_transform)
        pred_cqr_upper_val = inverse_transform(gam_upper.predict(X_val_final), selected_transform)

        pred_cqr_lower_test = inverse_transform(gam_lower.predict(X_test_final), selected_transform)
        pred_cqr_median_test = inverse_transform(gam_median.predict(X_test_final), selected_transform)
        pred_cqr_upper_test = inverse_transform(gam_upper.predict(X_test_final), selected_transform)

    except Exception as e:
        print(f"Error fitting ExpectileGAM for CQR: {str(e)}")
        print("Cannot proceed without a CQR baseline model.")
        # Handle error appropriately, e.g., sys.exit() or raise
        raise e # Re-raise the exception

elif args.aleatoric_model == 'linear':
    # Import QuantileRegressor
    from sklearn.linear_model import QuantileRegressor
    print("Fitting QuantileRegressor models for CQR...")

    # Fit linear QuantileRegressor models
    qr_lower = QuantileRegressor(quantile=lower_q, alpha=0, solver='highs')
    qr_median = QuantileRegressor(quantile=0.5, alpha=0, solver='highs')
    qr_upper = QuantileRegressor(quantile=upper_q, alpha=0, solver='highs')

    qr_lower.fit(X_train_final, y_train_final)
    qr_median.fit(X_train_final, y_train_final)
    qr_upper.fit(X_train_final, y_train_final)
    print("Finished fitting QuantileRegressors.")

    # Predict using the linear models
    pred_cqr_lower_val = inverse_transform(qr_lower.predict(X_val_final), selected_transform)
    pred_cqr_median_val = inverse_transform(qr_median.predict(X_val_final), selected_transform)
    pred_cqr_upper_val = inverse_transform(qr_upper.predict(X_val_final), selected_transform)

    pred_cqr_lower_test = inverse_transform(qr_lower.predict(X_test_final), selected_transform)
    pred_cqr_median_test = inverse_transform(qr_median.predict(X_test_final), selected_transform)
    pred_cqr_upper_test = inverse_transform(qr_upper.predict(X_test_final), selected_transform)

# Ensure quantiles don't cross (applies to both GAM and linear predictions)
pred_cqr_upper_val = np.maximum(pred_cqr_upper_val, pred_cqr_median_val)
pred_cqr_lower_val = np.minimum(pred_cqr_lower_val, pred_cqr_median_val)
pred_cqr_upper_test = np.maximum(pred_cqr_upper_test, pred_cqr_median_test)
pred_cqr_lower_test = np.minimum(pred_cqr_lower_test, pred_cqr_median_test)

# CQR Calibration (applies to both GAM and linear predictions)
calib_targets_val = y_val_true_orig.values
scores_cqr = np.maximum(pred_cqr_lower_val - calib_targets_val, calib_targets_val - pred_cqr_upper_val)
n_calib = len(calib_targets_val)
alpha = 1 - desired_coverage # Make sure alpha is defined
q_level_cqr = min((1 - alpha) * (1 + 1 / n_calib), 1.0)
calib_term_cqr = np.quantile(scores_cqr, q_level_cqr, method='higher')
print(f"CQR Calibration Term: {calib_term_cqr:.4f}")

# Final CQR test intervals
cqr_lower_test = pred_cqr_lower_test - calib_term_cqr
cqr_upper_test = pred_cqr_upper_test + calib_term_cqr

# --- Method 3: CLEAR ---
print("\nCalculating CLEAR intervals...")
# Use fewer bootstraps for aleatoric model if we're working with a subset of data
n_bootstraps_clear = 1
print(f"Using {n_bootstraps_clear} aleatoric bootstraps")

# Set random seed for CLEAR model
np.random.seed(42)
random.seed(42)

clear_model = CLEAR(
    desired_coverage=desired_coverage,
    n_bootstraps=n_bootstraps_clear,
    random_state=42,
    n_jobs=-1 # Use available cores
)

# Data already converted to float above
# Fit aleatoric component (using model specified by args)
print("Fitting CLEAR aleatoric component...")
clear_model.fit_aleatoric(
    X_train_final,
    y_train_final,
    quantile_model=args.aleatoric_model, # Use the selected model type
    fit_on_residuals=True, # Fit directly on (potentially transformed) y
    epistemic_preds=median_epistemic_train
)

# Predict aleatoric bounds
print("Predicting CLEAR aleatoric bounds...")
aleatoric_median_val_clear, aleatoric_lower_val_clear, aleatoric_upper_val_clear = clear_model.predict_aleatoric(
    X_val_final, 
    epistemic_preds=median_epistemic_val # <-- Add validation epistemic median
)
aleatoric_median_test_clear, aleatoric_lower_test_clear, aleatoric_upper_test_clear = clear_model.predict_aleatoric(
    X_test_final, 
    epistemic_preds=median_epistemic_test # <-- Add test epistemic median
)

# Inverse transform aleatoric predictions
aleatoric_median_val_clear = inverse_transform(aleatoric_median_val_clear, selected_transform)
aleatoric_lower_val_clear = inverse_transform(aleatoric_lower_val_clear, selected_transform)
aleatoric_upper_val_clear = inverse_transform(aleatoric_upper_val_clear, selected_transform)
aleatoric_median_test_clear = inverse_transform(aleatoric_median_test_clear, selected_transform)
aleatoric_lower_test_clear = inverse_transform(aleatoric_lower_test_clear, selected_transform)
aleatoric_upper_test_clear = inverse_transform(aleatoric_upper_test_clear, selected_transform)

# Calibrate CLEAR model
print("Calibrating CLEAR model...")
clear_model.calibrate(
    y_calib=calib_targets_val,
    median_epistemic=median_epistemic_val,
    aleatoric_median=aleatoric_median_val_clear,
    aleatoric_lower=aleatoric_lower_val_clear,
    aleatoric_upper=aleatoric_upper_val_clear,
    epistemic_lower=lower_epistemic_val,
    epistemic_upper=upper_epistemic_val
)

print(f"CLEAR Optimal Lambda: {clear_model.optimal_lambda:.4f}")
print(f"CLEAR Gamma: {clear_model.gamma:.4f}")

# Predict CLEAR intervals
print("Predicting CLEAR intervals...")
clear_lower_test, clear_upper_test = clear_model.predict(
    X_test_final,
    external_epistemic={
        'median': median_epistemic_test,
        'lower': lower_epistemic_test,
        'upper': upper_epistemic_test
    },
    external_aleatoric={
        'median': aleatoric_median_test_clear,
        'lower': aleatoric_lower_test_clear,
        'upper': aleatoric_upper_test_clear
    }
)


# --- Final Evaluation on Test Set ---
print("\n--- Test Set Evaluation ---")
y_test_eval = y_test_true_orig.values

# Method comparison explanation:
# - PCS-PPI: Uses an ensemble of bootstrapped models to capture epistemic uncertainty
#   and calibrates the intervals to get the desired coverage.
# - CQR: Uses quantile regression to model the aleatoric uncertainty directly
#   and applies conformal prediction to ensure coverage.
# - CLEAR: Combines both epistemic uncertainty (from ensemble) and aleatoric 
#   uncertainty (from quantile regression) with an adaptive weighting scheme.

# Debug output to understand structure of evaluate_intervals result
print("\nDebug: Checking what evaluate_intervals returns")
debug_eval = evaluate_intervals(y_test_eval[:10], pcs_ppi_lower_test[:10], pcs_ppi_upper_test[:10], alpha=alpha)
print(f"Debug: evaluate_intervals returns: {type(debug_eval)}")
print(f"Debug: evaluate_intervals keys: {debug_eval.keys() if isinstance(debug_eval, dict) else 'Not a dict'}")
print(f"Debug: evaluate_intervals first few values: {list(debug_eval.values())[:3] if isinstance(debug_eval, dict) else 'Not a dict'}")

# Evaluate each method and store the full evaluation dictionaries
pcs_eval = evaluate_intervals(y_test_eval, pcs_ppi_lower_test, pcs_ppi_upper_test, alpha=alpha)
cqr_eval = evaluate_intervals(y_test_eval, cqr_lower_test, cqr_upper_test, alpha=alpha)
clear_eval = evaluate_intervals(y_test_eval, clear_lower_test, clear_upper_test, alpha=alpha)

# Calculate mean widths and coverage directly as a backup
pcs_mean_width = np.mean(pcs_ppi_upper_test - pcs_ppi_lower_test)
cqr_mean_width = np.mean(cqr_upper_test - cqr_lower_test)
clear_mean_width = np.mean(clear_upper_test - clear_lower_test)

pcs_coverage = np.mean((y_test_eval >= pcs_ppi_lower_test) & (y_test_eval <= pcs_ppi_upper_test))
cqr_coverage = np.mean((y_test_eval >= cqr_lower_test) & (y_test_eval <= cqr_upper_test))
clear_coverage = np.mean((y_test_eval >= clear_lower_test) & (y_test_eval <= clear_upper_test))

# Print summary table with metrics from evaluate_intervals if available, otherwise use direct calculations
print(f"\nResults for all features (Desired Coverage: {desired_coverage*100}%):")
print("---------------------------------------------------------------------")
print(f"Method      | Coverage | Mean Width ($) | Quantile Loss | NCIW")
print("---------------------------------------------------------------------")
print(f"PCS-PPI     | {pcs_eval.get('PICP', pcs_coverage):.3f}    | {pcs_mean_width:<10.2f} | {pcs_eval.get('QuantileLoss', 'N/A'):<12.2f} | {pcs_eval.get('NCIW', 'N/A'):.4f}")
print(f"CQR         | {cqr_eval.get('PICP', cqr_coverage):.3f}    | {cqr_mean_width:<10.2f} | {cqr_eval.get('QuantileLoss', 'N/A'):<12.2f} | {cqr_eval.get('NCIW', 'N/A'):.4f}")
print(f"CLEAR       | {clear_eval.get('PICP', clear_coverage):.3f}    | {clear_mean_width:<10.2f} | {clear_eval.get('QuantileLoss', 'N/A'):<12.2f} | {clear_eval.get('NCIW', 'N/A'):.4f}")
print("---------------------------------------------------------------------")

# Save results to CSV
feature_suffix = "top2" if args.use_top_features else "all"
results_df = pd.DataFrame({
    'Method': ['PCS-PPI', 'CQR', 'CLEAR'],
    'Coverage': [
        pcs_eval.get('PICP', pcs_coverage),
        cqr_eval.get('PICP', cqr_coverage),
        clear_eval.get('PICP', clear_coverage)
    ],
    'Mean_Width': [pcs_mean_width, cqr_mean_width, clear_mean_width],
    'Quantile_Loss': [
        pcs_eval.get('QuantileLoss', np.nan),
        cqr_eval.get('QuantileLoss', np.nan),
        clear_eval.get('QuantileLoss', np.nan)
    ],
    'NCIW': [
        pcs_eval.get('NCIW', np.nan),
        cqr_eval.get('NCIW', np.nan),
        clear_eval.get('NCIW', np.nan)
    ],
    'NIW': [
        pcs_eval.get('NIW', np.nan),
        cqr_eval.get('NIW', np.nan),
        clear_eval.get('NIW', np.nan)
    ],
    'CRPS': [
        pcs_eval.get('CRPS', np.nan),
        cqr_eval.get('CRPS', np.nan),
        clear_eval.get('CRPS', np.nan)
    ],
    'AUC': [
        pcs_eval.get('AUC', np.nan),
        cqr_eval.get('AUC', np.nan),
        clear_eval.get('AUC', np.nan)
    ]
})

# Create results directory if it doesn't exist
results_dir = os.path.join(root_dir, 'results', 'case_study')
os.makedirs(results_dir, exist_ok=True)

# Save to CSV with appropriate suffix
csv_filename = os.path.join(results_dir, f'prediction_intervals_{feature_suffix}_{args.aleatoric_model}.csv')
results_df.to_csv(csv_filename, index=False)
print(f"\nResults saved to: {csv_filename}")
