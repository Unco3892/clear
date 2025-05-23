# %% [markdown]
# # [Chapter 13] The final Ames sale price predictions
# 
# ## [DSLC stages]: Analysis
# 
# 
# The following code sets up the libraries and creates cleaned and pre-processed training, validation and test data that we will use in this document.
# 

# %%

#######################################
# Add new parts to run the notebook as a script
import warnings
# Filter out the specific FutureWarning about downcasting in replace
warnings.filterwarnings("ignore", category=FutureWarning, 
                        message="Downcasting behavior in `replace` is deprecated")
# from sklearn.exceptions import ConvergenceWarning
# ConvergenceWarning('ignore')
# warnings.filterwarnings('ignore')

#######################################

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

#######################################
# Filter out scikit-learn convergence warnings
# from warnings import simplefilter
# from sklearn.exceptions import ConvergenceWarning
# simplefilter("ignore", category=ConvergenceWarning)
# warnings.filterwarnings('ignore')
# define all of the objects we need
# from prepare_ames_data import *
with open('prepare_ames_data.py', 'r') as file:
    exec(file.read())
#######################################

pd.set_option('display.max_columns', None)
pd.options.display.max_colwidth = 500
pd.options.display.max_rows = 100

# %% [markdown]
# 
# In this document we will demonstrate how to use the principles of PCS to choose the final prediction. We will demonstrate three different formats of the final prediction based on:
# 
# 1. The **single "best"** predictive algorithm, in terms of validation set performance from among a range of different algorithms each trained on several different cleaning/pre-processing judgment call-perturbed versions of the training dataset.
# 
# 1. An **ensemble** prediction, which combines the predictions from a range of predictive fits from across different algorithms and cleaning/pre-processing judgment call-perturbations that pass a predictability screening test.
# 
# 1. An **interval** of predictions from a range of predictive fits from across different algorithms and data- and cleaning/pre-processing judgment call-perturbations that pass a predictability screening test.
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
                                        convert_categorical=perturb_options['convert_categorical'][i])
                   for i in range(perturb_options.shape[0])]
len(ames_jc_perturb)

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
                             # make sure val set matches training set
                             column_selection=list(ames_jc_perturb[i].columns),
                             neighborhood_levels=train_neighborhoods)
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
                             # make sure test set matches training set
                             column_selection=list(ames_jc_perturb[i].columns),
                             neighborhood_levels=train_neighborhoods)
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
        lasso = Lasso(alpha=alpha)
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
# - `convert_categorical = "numeric"`
# 
# (Note that these results are slightly different to the R/book version which had the square-root transformation and the dummy coding of categorical variables, but this R/book version is second-best in terms of correlation here with very little difference overall.)
# 
# The "best" fit in terms of the rMSE measure has mostly the same set of judgment calls, but involves the LAD algorithm instead of the LS algorithm and the square-root transformation instead of the logarithmic transformation, and the "best" fit in terms of the MAE involves the simplified dummy encoding of the categorical variables.
# 
# Since the rMSE and MAE measures are slightly more precise than the correlation algorithm, we will use the **LAD algorithm trained on the training set with the particular cleaning/pre-processing judgment calls corresponding to the best rMSE here (the only difference between the version here and the book version is the `convert_cateogrical` variable is "dummy" in the book but we found the best version to be "numeric" here)**
# 

# %%
ames_train_preprocessed_selected = preprocess_ames_data(ames_train_clean,
                                                        max_identical_thresh=0.95,
                                                        n_neighborhoods=20,
                                                        impute_missing_categorical='mode',
                                                        simplify_vars=False,
                                                        transform_response='sqrt',
                                                        cor_feature_selection_threshold=0,
                                                        convert_categorical='numeric')

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
# extract relevant neighborhoods from  relevant training data
train_neighborhood_cols = ames_train_preprocessed_selected.filter(regex="neighborhood").columns
train_neighborhoods = [x.replace("neighborhood_", "") for x in train_neighborhood_cols]
    
# pre-processs the test set
ames_test_preprocessed_selected = preprocess_ames_data(ames_test_clean,
                                                        max_identical_thresh=0.95,
                                                        n_neighborhoods=20,
                                                        impute_missing_categorical='mode',
                                                        simplify_vars=False,
                                                        transform_response='sqrt',
                                                        cor_feature_selection_threshold=0,
                                                        convert_categorical='numeric',
                                                        neighborhood_levels=train_neighborhoods,
                                                        column_selection=list(ames_train_preprocessed_selected.columns))

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
# ## Approach 3: Calibrated PCS perturbation prediction intervals
# 
# 
# The process for computing the perturbation prediction intervals (PPIs) is similar to the ensemble prediction process, but instead of averaging the fits that pass the predictability screening test, we instead compute an interval from them. Since we want to include fits from data perturbations (bootstrap samples) when we compute the intervals, we will need to re-fit our algorithms on 10 bootstrapped versions of each of our fits that passed the (top 10% rMSE) screening stage above.
# 
# Note that only the LS and LAD fits make it into the top 10% of fits in terms of rMSE, so we will only use these two algorithms in the PPIs.

# %%
# how many LS fits made it into the top 10%?
sum(ls_rmse_top10p)

# %%
# extract just the perturbed datasets that are in the top 10% of RMSE performance for the LS model
ames_jc_perturbed_screened_ls = list(compress(ames_jc_perturb, ls_rmse_top10p))
# fit a linear regression model to each of these perturbed datasets using 10 bootstrap samples
ls_jc_perturbed_screened_boot_fit = []
for i in range(10):
    # compute a bootstrapped version of each relevant dataset
    ames_jc_perturbed_screened_ls_boot = [df.sample(n = df.shape[0], replace=True) for df in ames_jc_perturbed_screened_ls]
    # fit the LS model to each bootstrapped dataset
    for df in ames_jc_perturbed_screened_ls_boot:
        ls_jc_perturbed_screened_boot_fit.append(
            LinearRegression().fit(X=df.drop(columns='saleprice'), y=df['saleprice'])
        )

# %%
# how many LAD fits made it into the top 10%?
sum(lad_rmse_top10p)

# %%

# extract just the perturbed datasets that are in the top 10% of RMSE performance for the LAD model
ames_jc_perturbed_screened_lad = list(compress(ames_jc_perturb, lad_rmse_top10p))
# fit a linear regression model to each of these perturbed datasets using 10 bootstrap samples
lad_jc_perturbed_screened_boot_fit = []
for i in range(10):
    # compute a bootstrapped version of each relevant dataset
    ames_jc_perturbed_screened_lad_boot = [df.sample(n = df.shape[0], replace=True) for df in ames_jc_perturbed_screened_lad]
    # fit the LAD model to each bootstrapped dataset
    for df in ames_jc_perturbed_screened_lad_boot:
        lad_jc_perturbed_screened_boot_fit.append(
            LADRegression().fit(X=df.drop(columns='saleprice'), y=df['saleprice'])
        )

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


