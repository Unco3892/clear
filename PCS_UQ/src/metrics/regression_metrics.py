import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor

from conformal_methods.regression.split_conformal import SplitConformal


def evaluate_majority_vote(y_true, y_pred):
        cover = 0
        width = []
        for i in range(len(y_pred)):
            tv = y_true[i]
            lower = y_pred[i, 0]
            upper = y_pred[i, 1]
            cover += np.any([l <= tv and u >= tv for l, u in zip(lower, upper)])
            width.append(np.sum([u - l for l, u in zip(lower, upper)]))

        coverage = cover / len(y_true)
        avg_length = np.mean(width)
        med_length = np.median(width)
        range_y_test = y_true.max() - y_true.min()

        # Compile results into a DataFrame
        result_dict = {'coverage': coverage,
        'mean_width': avg_length,
        'median_width': med_length,
        'mean_width_scaled': avg_length / range_y_test,
        'median_width_scaled': med_length / range_y_test}

        return result_dict


def get_coverage(y_true, y_pred):
    if len(y_true) == 0:
        return np.nan
    coverage_indicator = (y_true >= y_pred[:, 0]) & (y_true <= y_pred[:, 1]).astype(int)
    return np.mean(coverage_indicator)

def get_mean_width(y_true, y_pred, return_scaled=False):
    if len(y_true) == 0:
        return np.nan
    mean_width = np.mean(np.abs(y_pred[:, 1] - y_pred[:, 0]))
    if return_scaled:
        scaled_mean_width = mean_width / np.abs(y_true.max() - y_true.min())
        return scaled_mean_width
    return mean_width

def get_median_width(y_true, y_pred, return_scaled=False):
    if len(y_true) == 0:
        return np.nan
    median_width = np.median(np.abs(y_pred[:, 1] - y_pred[:, 0]))
    if return_scaled:
        scaled_median_width = median_width / np.abs(y_true.max() - y_true.min())
        return scaled_median_width
    return median_width

def get_all_metrics(y_true, y_pred):
    return {
        'coverage': get_coverage(y_true, y_pred),
        'mean_width': get_mean_width(y_true, y_pred, return_scaled=False),
        'median_width': get_median_width(y_true, y_pred, return_scaled=False),
        'mean_width_scaled': get_mean_width(y_true, y_pred, return_scaled=True),
        'median_width_scaled': get_median_width(y_true, y_pred, return_scaled=True),
    }

if __name__ == "__main__":
    X, y = make_regression(n_samples=100, n_features=10, noise=10)
    conformal = SplitConformal(RandomForestRegressor())
    conformal.fit(X, y)
    y_pred = conformal.predict(X)
    print(get_all_metrics(y, y_pred))