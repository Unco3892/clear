#!/usr/bin/env python
# %% [markdown]
"""
# Demo: Calibrated Learning for Epistemic and Aleatoric Risk (CLEAR)

This demo shows how to use the CLEAR framework to build calibrated prediction intervals that combine aleatoric (noise) and epistemic (model) uncertainty.

**Six example scenarios are covered:**
1. CLEAR with built-in aleatoric and epistemic models (default GAM + Quantile Random Forest)
2. CLEAR with built-in aleatoric and flexible epistemic models (Random Forest + Quantile XGBoost)
3. CLEAR as a pure calibration layer (fully external predictions) **with baseline comparison**
4. California Housing dataset with XGBoost epistemic model
5. PCS ensemble (Predictability-Computability-Stability) on the Parkinsons dataset for the epistemic part with built-in aleatoric. Here the pre-trained results auto-downloaded, and then we compare CLEAR vs Aleatoric-R (CQR trained on the PCS residuals) vs Aleatoric (CQR) vs Epistemic (PCS)
6. Retrain PCS from scratch on the Airfoil dataset. This shows the full CLEAR + PCS pipeline end-to-end, no pre-trained files required

**Datasets:** All the datasets are retrieved automatically. Example scenarios 1,2,3 and 6 use the `Airfoil Self-Noise` dataset from our paper (1503 samples, 5 features). Example 4 uses the `California Housing` dataset built into sklearn. Example 5 uses the `Parkinsons Telemonitoring` dataset also from our paper.

**To run on Google Colab**, simply run all cells (the first cell installs the package automatically). Note that you can either run `demo.py` or `demo.ipynb` as the content are the same.
"""

# %% [markdown]
"""
## Setup
"""

# %%
# -------------------
# COLAB INSTALLATION
# -------------------
# Installs clear-uq if not already available (e.g., on Google Colab).
# Locally this is a no-op if the package is already installed.
# NOTE: On Colab, the first install takes ~3-5 minutes and will
#       automatically restart the runtime to pick up new dependencies.
#       After the restart, simply re-run all cells (the install is instant).
import subprocess, sys

try:
    import clear
    import importlib.metadata
    version = importlib.metadata.version("clear-uq")
    print(f"\u2713 clear-uq already installed (version {version})")
except ImportError:
    print("Installing clear-uq from PyPI (this may take 3-5 minutes) ...")
    # Show pip progress so users see what's happening during the install
    subprocess.check_call([sys.executable, "-m", "pip", "install", "clear-uq"])
    import importlib.metadata
    version = importlib.metadata.version("clear-uq")
    print(f"\n\u2713 clear-uq {version} installed successfully!")
    # On Colab, restart the runtime so that pre-loaded C-extension packages
    # (numpy, pandas, etc.) are reloaded against the newly installed versions.
    # Without this, you get "numpy.dtype size changed" errors.
    try:
        from google.colab import runtime  # must use 'from' import
        print("\u21BB Restarting Colab runtime \u2014 please re-run all cells after restart ...")
        runtime.restart()
    except ImportError:
        pass  # Not on Colab — no restart needed
    except Exception:
        # Fallback: force-kill the process to trigger a Colab restart
        import os, signal
        os.kill(os.getpid(), signal.SIGKILL)

# %%
# --------------------------
# IMPORTS & CONFIGURATION
# --------------------------
import numpy as np
import pandas as pd
import os
import io
import copy
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import matplotlib.pyplot as plt

from clear.clear import CLEAR
from clear.metrics import evaluate_intervals
from clear.utils import plot_prediction_intervals

# Disable Colab's vertical-scroll cap on cell output (no-op outside Colab)
try:
    from google.colab import output as _colab_output
    _colab_output.no_vertical_scroll()
except ImportError:
    pass

# Configuration
DESIRED_COVERAGE = 0.95         # Target coverage for prediction intervals
N_BOOTSTRAPS = 10               # Number of bootstrap samples (use 100 for paper-level results)
RANDOM_STATE = 777              # Random seed for reproducibility
N_JOBS = -1                     # Parallel jobs (-1 = all cores)
ALPHA = 1 - DESIRED_COVERAGE    # Miscoverage level
# Optional: pass a custom lambda grid to CLEAR() to match the paper's benchmark exactly.
# The default grid is sufficient for most use cases. Uncomment to use the paper's grid:
# LAMBDAS_CLEAR = np.concatenate((np.linspace(0, 0.09, 10), np.logspace(-1, 2, 4001)))
# Then pass it as: CLEAR(..., lambdas=LAMBDAS_CLEAR)

np.random.seed(RANDOM_STATE)
print("Imports and configuration loaded.")

# %%
# --------------------------
# HELPER FUNCTIONS
# --------------------------

def print_all_metrics(metrics, method_name=""):
    """Pretty-print key metrics returned by evaluate_intervals."""
    title = f"  {method_name}:" if method_name else "  Results:"
    print(title)
    for key in COMPARISON_METRICS:
        print(f"    {key:<20s} {metrics[key]:>10.4f}")
    print()

COMPARISON_METRICS = ["PICP", "MPIW", "NCIW", "QuantileLoss"]


def comparison_table(results_dict, target_coverage):
    """
    Print a side-by-side comparison table for several methods.

    Only the four key metrics from the paper are shown:
    PICP, MPIW, NCIW, and QuantileLoss.

    Parameters
    ----------
    results_dict : dict[str, dict]
        {method_name: metrics_dict}
    target_coverage : float
    """
    header = f"{'Method':<20}" + "".join(f"{m:>16}" for m in COMPARISON_METRICS)
    print(header)
    print("-" * len(header))
    for method, metrics in results_dict.items():
        row = f"{method:<20}"
        for m in COMPARISON_METRICS:
            row += f"{metrics.get(m, float('nan')):>16.4f}"
        print(row)
    print(f"\nTarget coverage: {target_coverage}")

# %% [markdown]
"""
## Load Data: Airfoil Self-Noise Dataset

The Airfoil dataset (from our paper) has 1503 samples and 5 features. We load it from the GitHub repository, falling back to local files if available.
"""

# %%
def load_airfoil_data():
    """Load the Airfoil dataset from local files or GitHub."""
    local_X = os.path.join("data", "data_airfoil", "X.csv")
    local_y = os.path.join("data", "data_airfoil", "y.csv")

    if os.path.exists(local_X) and os.path.exists(local_y):
        print("Loading Airfoil data from local files...")
        X = pd.read_csv(local_X)
        y = pd.read_csv(local_y, header=None).squeeze("columns")
    else:
        print("Downloading Airfoil data from GitHub...")
        from urllib.request import urlopen
        base_url = "https://raw.githubusercontent.com/Unco3892/clear/main/data/data_airfoil"
        X = pd.read_csv(io.StringIO(urlopen(f"{base_url}/X.csv").read().decode()))
        y = pd.read_csv(io.StringIO(urlopen(f"{base_url}/y.csv").read().decode()), header=None).squeeze("columns")

    return X, y

X, y = load_airfoil_data()

# Split: 60% train, 20% calibration, 20% test
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)
X_train, X_calib, y_train, y_calib = train_test_split(
    X_train_full, y_train_full, test_size=0.25, random_state=RANDOM_STATE
)
print(f"Data shapes: Train {X_train.shape}, Calibration {X_calib.shape}, Test {X_test.shape}")

# %% [markdown]
"""
## Example 1: CLEAR with Built-in Models

This is the simplest way to use CLEAR. We fit an epistemic model (GAM bootstrap ensemble) and an aleatoric model (Quantile Random Forest) internally, then calibrate and predict.
"""

# %%
print("=" * 60)
print("Example 1: CLEAR with Built-in Models (GAM + QRF)")
print("=" * 60)

clear_1 = CLEAR(
    desired_coverage=DESIRED_COVERAGE,
    n_bootstraps=N_BOOTSTRAPS,
    random_state=RANDOM_STATE,
    n_jobs=N_JOBS
)

# Step 1: Fit epistemic model (default = LinearGAM bootstrap ensemble)
clear_1.fit_epistemic(X_train, y_train)

# Step 2: Get epistemic predictions for calibration and test sets
ep_med_calib, ep_low_calib, ep_up_calib, _ = clear_1.predict_epistemic(X_calib)
ep_med_test, ep_low_test, ep_up_test, _ = clear_1.predict_epistemic(X_test)

# Step 3: Fit aleatoric model (QRF on residuals of epistemic median)
ep_med_train, _, _, _ = clear_1.predict_epistemic(X_train)
clear_1.fit_aleatoric(
    X_train, y_train,
    quantile_model="rf",
    fit_on_residuals=True,
    epistemic_preds=ep_med_train
)

# Step 4: Get aleatoric predictions
al_med_calib, al_low_calib, al_up_calib = clear_1.predict_aleatoric(X_calib, epistemic_preds=ep_med_calib)
al_med_test, al_low_test, al_up_test = clear_1.predict_aleatoric(X_test, epistemic_preds=ep_med_test)

# Step 5: Calibrate CLEAR (find optimal lambda and gamma)
clear_1.calibrate(
    y_calib,
    median_epistemic=ep_med_calib,
    aleatoric_median=al_med_calib,
    aleatoric_lower=al_low_calib,
    aleatoric_upper=al_up_calib,
    epistemic_lower=ep_low_calib,
    epistemic_upper=ep_up_calib,
    verbose=False
)
print(f"Optimal Lambda: {clear_1.optimal_lambda:.4f}, Gamma: {clear_1.gamma:.4f}")

# Step 6: Predict calibrated intervals
lower_1, upper_1 = clear_1.predict(
    X_test,
    external_epistemic={'median': ep_med_test, 'lower': ep_low_test, 'upper': ep_up_test},
    external_aleatoric={'median': al_med_test, 'lower': al_low_test, 'upper': al_up_test}
)

# Evaluate — all metrics
metrics_1 = evaluate_intervals(y_test, lower_1, upper_1, alpha=ALPHA, f=ep_med_test)
print_all_metrics(metrics_1, "Example 1 (GAM + QRF)")

# %% [markdown]
"""
## Example 2: CLEAR with Flexible Epistemic Model

Here we use a **Random Forest** as the epistemic model instead of GAM. You can pass any sklearn-compatible model class or use string shortcuts: ``"rf"``, ``"xgb"``, ``"gam"``, ``"ridge"``, or a custom class.
"""

# %%
print("\n" + "=" * 60)
print("Example 2: CLEAR with Flexible Epistemic Model (RF + QXGB)")
print("=" * 60)

clear_2 = CLEAR(
    desired_coverage=DESIRED_COVERAGE,
    n_bootstraps=N_BOOTSTRAPS,
    random_state=RANDOM_STATE,
    n_jobs=N_JOBS
)

# Step 1: Fit epistemic model — now using Random Forest instead of GAM
clear_2.fit_epistemic(X_train, y_train, epistemic_model="rf", model_params={"n_estimators": 50})

# Step 2: Get epistemic predictions
ep_med_calib_2, ep_low_calib_2, ep_up_calib_2, _ = clear_2.predict_epistemic(X_calib)
ep_med_test_2, ep_low_test_2, ep_up_test_2, _ = clear_2.predict_epistemic(X_test)

# Step 3: Fit aleatoric model — using Quantile XGBoost on residuals
ep_med_train_2, _, _, _ = clear_2.predict_epistemic(X_train)
clear_2.fit_aleatoric(
    X_train, y_train,
    quantile_model="xgb",
    fit_on_residuals=True,
    epistemic_preds=ep_med_train_2
)

# Step 4: Get aleatoric predictions
al_med_calib_2, al_low_calib_2, al_up_calib_2 = clear_2.predict_aleatoric(X_calib, epistemic_preds=ep_med_calib_2)
al_med_test_2, al_low_test_2, al_up_test_2 = clear_2.predict_aleatoric(X_test, epistemic_preds=ep_med_test_2)

# Step 5: Calibrate
clear_2.calibrate(
    y_calib,
    median_epistemic=ep_med_calib_2,
    aleatoric_median=al_med_calib_2,
    aleatoric_lower=al_low_calib_2,
    aleatoric_upper=al_up_calib_2,
    epistemic_lower=ep_low_calib_2,
    epistemic_upper=ep_up_calib_2,
    verbose=False
)
print(f"Optimal Lambda: {clear_2.optimal_lambda:.4f}, Gamma: {clear_2.gamma:.4f}")

# Step 6: Predict
lower_2, upper_2 = clear_2.predict(
    X_test,
    external_epistemic={'median': ep_med_test_2, 'lower': ep_low_test_2, 'upper': ep_up_test_2},
    external_aleatoric={'median': al_med_test_2, 'lower': al_low_test_2, 'upper': al_up_test_2}
)

metrics_2 = evaluate_intervals(y_test, lower_2, upper_2, alpha=ALPHA, f=ep_med_test_2)
print_all_metrics(metrics_2, "Example 2 (RF + QXGB)")

# %% [markdown]
"""
## Example 3: Fully External Predictions + Baseline Comparison

CLEAR can also be used as a **pure calibration layer**. Train your own models outside of CLEAR, compute epistemic and aleatoric bounds yourself, and pass them directly to ``calibrate()`` and ``predict()``.

We then compare CLEAR against two standard baselines **using the same underlying models**, only the calibration strategy differs:
- **CQR** (Conformalized Quantile Regression, Romano et al., 2019): conformal correction on quantile bounds
- **Split-Conformal**: symmetric constant-width intervals from absolute errors (``f_hat +/- gamma``)
"""

# %%
print("\n" + "=" * 60)
print("Example 3: Fully External Predictions (GradientBoosting + QRF)")
print("=" * 60)

from sklearn.ensemble import GradientBoostingRegressor
from quantile_forest import RandomForestQuantileRegressor

X_train_np = np.asarray(X_train)
X_calib_np = np.asarray(X_calib)
X_test_np = np.asarray(X_test)
y_train_np = np.asarray(y_train).flatten()
y_calib_np = np.asarray(y_calib).flatten()

# --- External Epistemic: Bootstrap ensemble of GradientBoosting ---
print("Training external epistemic ensemble (GradientBoosting)...")
n_ensemble = N_BOOTSTRAPS
epistemic_models = []
for i in range(n_ensemble):
    np.random.seed(RANDOM_STATE + i)
    idx = np.random.choice(len(y_train_np), len(y_train_np), replace=True)
    model = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=RANDOM_STATE + i)
    model.fit(X_train_np[idx], y_train_np[idx])
    epistemic_models.append(model)

# Get epistemic bounds from ensemble quantiles
def get_ensemble_bounds(models, X, lower_q=0.025, upper_q=0.975):
    preds = np.array([m.predict(X) for m in models])
    return np.median(preds, axis=0), np.quantile(preds, lower_q, axis=0), np.quantile(preds, upper_q, axis=0)

ep_med_calib_3, ep_low_calib_3, ep_up_calib_3 = get_ensemble_bounds(epistemic_models, X_calib_np)
ep_med_test_3, ep_low_test_3, ep_up_test_3 = get_ensemble_bounds(epistemic_models, X_test_np)

# --- External Aleatoric: Quantile Random Forest on residuals ---
print("Training external aleatoric model (QRF on residuals)...")
ep_med_train_3, _, _ = get_ensemble_bounds(epistemic_models, X_train_np)
residuals_train = y_train_np - ep_med_train_3

# Augment features with epistemic median (residual-based approach)
X_train_aug = np.column_stack((X_train_np, ep_med_train_3))
X_calib_aug = np.column_stack((X_calib_np, ep_med_calib_3))
X_test_aug = np.column_stack((X_test_np, ep_med_test_3))

qrf_lower = RandomForestQuantileRegressor(
    n_estimators=100, default_quantiles=ALPHA / 2,
    min_samples_leaf=10, random_state=RANDOM_STATE
)
qrf_median = RandomForestQuantileRegressor(
    n_estimators=100, default_quantiles=0.5,
    min_samples_leaf=10, random_state=RANDOM_STATE
)
qrf_upper = RandomForestQuantileRegressor(
    n_estimators=100, default_quantiles=1 - ALPHA / 2,
    min_samples_leaf=10, random_state=RANDOM_STATE
)
qrf_lower.fit(X_train_aug, residuals_train)
qrf_median.fit(X_train_aug, residuals_train)
qrf_upper.fit(X_train_aug, residuals_train)

# Predict aleatoric bounds (add back epistemic median to get back to y-space)
al_low_calib_3 = qrf_lower.predict(X_calib_aug) + ep_med_calib_3
al_med_calib_3 = qrf_median.predict(X_calib_aug) + ep_med_calib_3
al_up_calib_3 = qrf_upper.predict(X_calib_aug) + ep_med_calib_3

al_low_test_3 = qrf_lower.predict(X_test_aug) + ep_med_test_3
al_med_test_3 = qrf_median.predict(X_test_aug) + ep_med_test_3
al_up_test_3 = qrf_upper.predict(X_test_aug) + ep_med_test_3

# --- Use CLEAR as a pure calibration layer ---
clear_3 = CLEAR(
    desired_coverage=DESIRED_COVERAGE,
    random_state=RANDOM_STATE
)

# No fit_epistemic or fit_aleatoric needed — just calibrate with external arrays
clear_3.calibrate(
    y_calib_np,
    median_epistemic=ep_med_calib_3,
    aleatoric_median=al_med_calib_3,
    aleatoric_lower=al_low_calib_3,
    aleatoric_upper=al_up_calib_3,
    epistemic_lower=ep_low_calib_3,
    epistemic_upper=ep_up_calib_3,
    verbose=False
)
print(f"Optimal Lambda: {clear_3.optimal_lambda:.4f}, Gamma: {clear_3.gamma:.4f}")

# Predict with external predictions
lower_3, upper_3 = clear_3.predict(
    X_test_np,
    external_epistemic={'median': ep_med_test_3, 'lower': ep_low_test_3, 'upper': ep_up_test_3},
    external_aleatoric={'median': al_med_test_3, 'lower': al_low_test_3, 'upper': al_up_test_3}
)

metrics_3 = evaluate_intervals(y_test, lower_3, upper_3, alpha=ALPHA, f=ep_med_test_3)
print_all_metrics(metrics_3, "CLEAR (External GradientBoosting + QRF)")

# %% [markdown]
"""
### Baseline Comparison

We compare CLEAR against two standard calibration baselines:
- **CQR** (Conformalized Quantile Regression): a QRF trained directly on raw
  targets (same hyperparameters, same features — no residual decomposition),
  with standard conformal correction (Romano et al., 2019).
- **Split-Conformal**: uses the bootstrap ensemble's median as the point predictor
  ``f_hat``, computes the nonconformity scores ``|y - f_hat|`` on the calibration
  set, takes the ``(1-alpha)``-quantile of those scores as ``gamma``, and outputs
  the constant-width interval ``f_hat ± gamma``.
"""

# %%
print("--- Baseline Methods ---\n")

n_calib = len(y_calib_np)
q_level = min((1 - ALPHA) * (1 + 1 / n_calib), 1.0)

# --- CQR (Conformalized Quantile Regression on raw targets) ---
# Train a standard QRF on raw y (not residuals) — same hyperparameters as above
print("Training CQR baseline (QRF on raw targets)...")
cqr_qrf_lower = RandomForestQuantileRegressor(
    n_estimators=100, default_quantiles=ALPHA / 2,
    min_samples_leaf=10, random_state=RANDOM_STATE
)
cqr_qrf_upper = RandomForestQuantileRegressor(
    n_estimators=100, default_quantiles=1 - ALPHA / 2,
    min_samples_leaf=10, random_state=RANDOM_STATE
)
cqr_qrf_lower.fit(X_train_np, y_train_np)
cqr_qrf_upper.fit(X_train_np, y_train_np)

# Calibration scores
cqr_calib_lower = cqr_qrf_lower.predict(X_calib_np)
cqr_calib_upper = cqr_qrf_upper.predict(X_calib_np)
cqr_scores = np.maximum(cqr_calib_lower - y_calib_np, y_calib_np - cqr_calib_upper)
cqr_adjustment = np.quantile(cqr_scores, q_level, method='higher')

# Test intervals
cqr_lower = cqr_qrf_lower.predict(X_test_np) - cqr_adjustment
cqr_upper = cqr_qrf_upper.predict(X_test_np) + cqr_adjustment
metrics_cqr = evaluate_intervals(y_test, cqr_lower, cqr_upper, alpha=ALPHA, f=ep_med_test_3)

# --- Split-Conformal ---
# Uses the ensemble median as f_hat, computes |y - f_hat| on calibration data,
# and takes the (1-alpha)-quantile as the half-width gamma → f_hat ± gamma.
abs_residuals_calib = np.abs(y_calib_np - ep_med_calib_3)
gamma_naive = np.quantile(abs_residuals_calib, q_level, method='higher')
s_naive_lower = ep_med_test_3 - gamma_naive
s_naive_upper = ep_med_test_3 + gamma_naive
metrics_s_naive = evaluate_intervals(y_test, s_naive_lower, s_naive_upper, alpha=ALPHA, f=ep_med_test_3)

# --- Side-by-side comparison ---
print("=" * 60)
print("Comparison: CLEAR vs Baselines (Airfoil, GradientBoosting + QRF)")
print("=" * 60)
comparison_table(
    {
        "CLEAR": metrics_3,
        "CQR": metrics_cqr,
        "Split-Conformal": metrics_s_naive,
    },
    DESIRED_COVERAGE
)

# %% [markdown]
"""
All methods achieve valid coverage. CLEAR produces tighter intervals byoptimally weighting epistemic and aleatoric uncertainty components.
"""

# %% [markdown]
"""
## Example 4: California Housing (sklearn built-in dataset)

Demonstrates CLEAR on a larger, well-known dataset that requires no downloads. Uses the flexible epistemic API with XGBoost and aleatoric QRF.
"""

# %%
print("\n" + "=" * 60)
print("Example 4: California Housing (XGBoost + QRF)")
print("=" * 60)

from sklearn.datasets import fetch_california_housing

# Load data — built into sklearn, no downloads needed
housing = fetch_california_housing()
X_housing = pd.DataFrame(housing.data, columns=housing.feature_names)
y_housing = pd.Series(housing.target, name='MedHouseVal')
print(f"California Housing: {X_housing.shape[0]} samples, {X_housing.shape[1]} features")

# Split: 60% train, 20% calibration, 20% test
X_train_full_h, X_test_h, y_train_full_h, y_test_h = train_test_split(
    X_housing, y_housing, test_size=0.2, random_state=RANDOM_STATE
)
X_train_h, X_calib_h, y_train_h, y_calib_h = train_test_split(
    X_train_full_h, y_train_full_h, test_size=0.25, random_state=RANDOM_STATE
)
print(f"Data shapes: Train {X_train_h.shape}, Calibration {X_calib_h.shape}, Test {X_test_h.shape}")

clear_4 = CLEAR(
    desired_coverage=DESIRED_COVERAGE,
    n_bootstraps=N_BOOTSTRAPS,
    random_state=RANDOM_STATE,
    n_jobs=N_JOBS
)

# Epistemic: XGBoost bootstrap ensemble
clear_4.fit_epistemic(X_train_h, y_train_h, epistemic_model="xgb", model_params={"n_estimators": 100})

ep_med_calib_4, ep_low_calib_4, ep_up_calib_4, _ = clear_4.predict_epistemic(X_calib_h)
ep_med_test_4, ep_low_test_4, ep_up_test_4, _ = clear_4.predict_epistemic(X_test_h)

# Aleatoric: QRF on residuals
ep_med_train_4, _, _, _ = clear_4.predict_epistemic(X_train_h)
clear_4.fit_aleatoric(
    X_train_h, y_train_h,
    quantile_model="rf",
    fit_on_residuals=True,
    epistemic_preds=ep_med_train_4
)

al_med_calib_4, al_low_calib_4, al_up_calib_4 = clear_4.predict_aleatoric(X_calib_h, epistemic_preds=ep_med_calib_4)
al_med_test_4, al_low_test_4, al_up_test_4 = clear_4.predict_aleatoric(X_test_h, epistemic_preds=ep_med_test_4)

# Calibrate and predict
clear_4.calibrate(
    y_calib_h,
    median_epistemic=ep_med_calib_4,
    aleatoric_median=al_med_calib_4,
    aleatoric_lower=al_low_calib_4,
    aleatoric_upper=al_up_calib_4,
    epistemic_lower=ep_low_calib_4,
    epistemic_upper=ep_up_calib_4,
    verbose=False
)
print(f"Optimal Lambda: {clear_4.optimal_lambda:.4f}, Gamma: {clear_4.gamma:.4f}")

lower_4, upper_4 = clear_4.predict(
    X_test_h,
    external_epistemic={'median': ep_med_test_4, 'lower': ep_low_test_4, 'upper': ep_up_test_4},
    external_aleatoric={'median': al_med_test_4, 'lower': al_low_test_4, 'upper': al_up_test_4}
)

metrics_4 = evaluate_intervals(y_test_h, lower_4, upper_4, alpha=ALPHA, f=ep_med_test_4)
print_all_metrics(metrics_4, "Example 4 (XGBoost + QRF, CA Housing)")

# --- Baselines for California Housing ---
X_train_h_np = np.asarray(X_train_h)
X_calib_h_np = np.asarray(X_calib_h)
X_test_h_np = np.asarray(X_test_h)
y_train_h_np = np.asarray(y_train_h).flatten()
y_calib_h_np = np.asarray(y_calib_h).flatten()

n_calib_h = len(y_calib_h_np)
q_level_h = min((1 - ALPHA) * (1 + 1 / n_calib_h), 1.0)

# CQR (Romano et al., 2019) — QRF on raw targets
print("Training CQR baseline for CA Housing...")
cqr_qrf_lower_h = RandomForestQuantileRegressor(
    n_estimators=100, default_quantiles=ALPHA / 2,
    min_samples_leaf=10, random_state=RANDOM_STATE
)
cqr_qrf_upper_h = RandomForestQuantileRegressor(
    n_estimators=100, default_quantiles=1 - ALPHA / 2,
    min_samples_leaf=10, random_state=RANDOM_STATE
)
cqr_qrf_lower_h.fit(X_train_h_np, y_train_h_np)
cqr_qrf_upper_h.fit(X_train_h_np, y_train_h_np)

cqr_calib_lower_h = cqr_qrf_lower_h.predict(X_calib_h_np)
cqr_calib_upper_h = cqr_qrf_upper_h.predict(X_calib_h_np)
cqr_scores_h = np.maximum(cqr_calib_lower_h - y_calib_h_np, y_calib_h_np - cqr_calib_upper_h)
cqr_adj_h = np.quantile(cqr_scores_h, q_level_h, method='higher')

cqr_lower_h = cqr_qrf_lower_h.predict(X_test_h_np) - cqr_adj_h
cqr_upper_h = cqr_qrf_upper_h.predict(X_test_h_np) + cqr_adj_h
metrics_cqr_h = evaluate_intervals(y_test_h, cqr_lower_h, cqr_upper_h, alpha=ALPHA, f=ep_med_test_4)

# Split-Conformal — symmetric split-conformal around ensemble median
abs_res_calib_h = np.abs(y_calib_h_np - np.asarray(ep_med_calib_4))
gamma_naive_h = np.quantile(abs_res_calib_h, q_level_h, method='higher')
s_naive_lower_h = np.asarray(ep_med_test_4) - gamma_naive_h
s_naive_upper_h = np.asarray(ep_med_test_4) + gamma_naive_h
metrics_s_naive_h = evaluate_intervals(y_test_h, s_naive_lower_h, s_naive_upper_h, alpha=ALPHA, f=ep_med_test_4)

print("\n" + "=" * 60)
print("Comparison: CLEAR vs Baselines (California Housing)")
print("=" * 60)
comparison_table(
    {
        "CLEAR": metrics_4,
        "CQR": metrics_cqr_h,
        "Split-Conformal": metrics_s_naive_h,
    },
    DESIRED_COVERAGE
)

# %% [markdown]
"""
## Example 5: PCS Ensemble on Parkinsons Dataset

Demonstrates CLEAR layered on top of a pre-trained **PCS** (Predictability-Computability-Stability) ensemble, the epistemic component used in the paper. The pre-trained results (~13 MB) are downloaded automatically from the repository the first time this cell runs.

Four methods are compared at the same target coverage (95%):
- **CLEAR**: adaptively balances aleatoric and epistemic uncertainty
- **Aleatoric-R**: aleatoric QRF intervals centered on PCS median, conformally calibrated
- **Aleatoric (CQR)**: plain CQR, QRF trained on raw targets, no epistemic component
- **Epistemic (PCS)**: standard multiplicative PCS calibration (epistemic-only, no aleatoric model)
"""

# %%
print("\n" + "=" * 60)
print("Example 5: PCS Ensemble (Parkinsons Dataset)")
print("=" * 60)

import pickle
from urllib.request import urlretrieve

# --- Auto-download pre-trained PCS results ---
# Prefer the full model directory (qpcs_10_standard) when available locally;
# fall back to the demo copy committed to the repo (numerically identical).
PKL_PATH_FULL = os.path.join("models", "pcs_top1_qpcs_10_standard", "data_parkinsons_pcs_results_95.pkl")
PKL_PATH_DEMO = os.path.join("models", "demo", "data_parkinsons_pcs_results_95.pkl")
PKL_URL = (
    "https://raw.githubusercontent.com/Unco3892/clear/main/"
    "models/demo/data_parkinsons_pcs_results_95.pkl"
)

if os.path.exists(PKL_PATH_FULL):
    PKL_PATH = PKL_PATH_FULL
    print(f"Loading PCS results from full model directory: {PKL_PATH}")
elif os.path.exists(PKL_PATH_DEMO):
    PKL_PATH = PKL_PATH_DEMO
    print(f"Loading PCS results from: {PKL_PATH}")
else:
    PKL_PATH = PKL_PATH_DEMO
    os.makedirs(os.path.dirname(PKL_PATH), exist_ok=True)
    print(f"Downloading pre-trained PCS results (~13 MB) ...")
    urlretrieve(PKL_URL, PKL_PATH)
    print("Download complete.")

with open(PKL_PATH, "rb") as f:
    pcs_data = pickle.load(f)

# Use a single run (run_0) for the demo
run = pcs_data["run_0"]

# --- Reconstruct data splits from the pickle ---
X_train_p    = np.array(run["x_train"])
y_train_p    = np.array(run["y_train"]).flatten()
X_calib_p    = np.array(run["x_val"])
y_calib_p    = np.array(run["y_val"]).flatten()
X_test_p     = np.array(run["x_test"])
y_test_p     = np.array(run["y_test"]).flatten()
print(f"Parkinsons splits: Train {X_train_p.shape}, Calib {X_calib_p.shape}, Test {X_test_p.shape}")

# --- PCS raw predictions: (n, 3) → [lower, median, upper] ---
ep_low_train_5  = np.array(run["train_intervals_raw"])[:, 0]
ep_med_train_5  = np.array(run["train_intervals_raw"])[:, 1]
ep_up_train_5   = np.array(run["train_intervals_raw"])[:, 2]

ep_low_calib_5  = np.array(run["val_intervals_raw"])[:, 0]
ep_med_calib_5  = np.array(run["val_intervals_raw"])[:, 1]
ep_up_calib_5   = np.array(run["val_intervals_raw"])[:, 2]

ep_low_test_5   = np.array(run["test_intervals_raw"])[:, 0]
ep_med_test_5   = np.array(run["test_intervals_raw"])[:, 1]
ep_up_test_5    = np.array(run["test_intervals_raw"])[:, 2]

ALPHA_5 = ALPHA   # same as global (0.05 → 95% coverage)

# --- Fit aleatoric model via CLEAR (bootstrapped QRF on residuals) ---
# NOTE: Set to 10 here for speed. To exactly replicate the paper's benchmark
# results (results/standard/qPCS_all_10seeds_all/), increase this to 100;
# all reported metrics (PICP, MPIW, NCIW, QuantileLoss, lambda, gamma) match
# the benchmark to 6+ decimal places at N_BOOTSTRAPS_5 = 100.
N_BOOTSTRAPS_5 = 10
print(f"Fitting aleatoric QRF on PCS training residuals (via CLEAR, {N_BOOTSTRAPS_5} bootstraps)...")
clear_5 = CLEAR(desired_coverage=DESIRED_COVERAGE, n_bootstraps=N_BOOTSTRAPS_5,
                random_state=RANDOM_STATE, n_jobs=N_JOBS)
clear_5.fit_aleatoric(
    X_train_p, y_train_p,
    quantile_model="rf",
    fit_on_residuals=True,
    epistemic_preds=ep_med_train_5
)

al_med_calib_5, al_low_calib_5, al_up_calib_5 = clear_5.predict_aleatoric(X_calib_p, epistemic_preds=ep_med_calib_5)
al_med_test_5,  al_low_test_5,  al_up_test_5  = clear_5.predict_aleatoric(X_test_p,  epistemic_preds=ep_med_test_5)

# --- Calibrate CLEAR ---
clear_5.calibrate(
    y_calib_p,
    median_epistemic=ep_med_calib_5,
    aleatoric_median=al_med_calib_5,
    aleatoric_lower=al_low_calib_5,
    aleatoric_upper=al_up_calib_5,
    epistemic_lower=ep_low_calib_5,
    epistemic_upper=ep_up_calib_5,
    verbose=False
)
print(f"Optimal Lambda: {clear_5.optimal_lambda:.4f}, Gamma: {clear_5.gamma:.4f}")

lower_5, upper_5 = clear_5.predict(
    X_test_p,
    external_epistemic={"median": ep_med_test_5, "lower": ep_low_test_5, "upper": ep_up_test_5},
    external_aleatoric={"median": al_med_test_5, "lower": al_low_test_5, "upper": al_up_test_5},
)
metrics_5 = evaluate_intervals(y_test_p, lower_5, upper_5, alpha=ALPHA_5, f=ep_med_test_5)
print_all_metrics(metrics_5, "CLEAR (PCS + QRF, Parkinsons)")

# --- Aleatoric-R baseline: aleatoric intervals centred on PCS median ---
# Intervals centred on PCS median using aleatoric width, then conformally adjusted.
q_level_5 = min((1 - ALPHA_5) * (1 + 1 / len(y_calib_p)), 1.0)
calib_lo_ar = ep_med_calib_5 - (al_med_calib_5 - al_low_calib_5)
calib_hi_ar = ep_med_calib_5 + (al_up_calib_5  - al_med_calib_5)
ar_scores   = np.maximum(calib_lo_ar - y_calib_p, y_calib_p - calib_hi_ar)
ar_adj      = np.quantile(ar_scores, q_level_5, method="higher")
ar_lower_5  = (ep_med_test_5 - (al_med_test_5 - al_low_test_5)) - ar_adj
ar_upper_5  = (ep_med_test_5 + (al_up_test_5  - al_med_test_5)) + ar_adj
metrics_ar_5 = evaluate_intervals(y_test_p, ar_lower_5, ar_upper_5, alpha=ALPHA_5, f=ep_med_test_5)

# --- PCS baseline: multiplicative recalibration to 95% on val set ---
# The raw PCS intervals encode epistemic uncertainty only; here we recalibrate
# them to the same 95% target so the comparison is fair.
raw_half_5  = (ep_up_calib_5 - ep_low_calib_5) / 2.0
pcs_scores  = np.abs(y_calib_p - ep_med_calib_5) / np.maximum(raw_half_5, 1e-10)
pcs_gamma   = np.quantile(pcs_scores, q_level_5, method="higher")
pcs_lower_5 = ep_med_test_5 - pcs_gamma * (ep_up_test_5 - ep_low_test_5) / 2.0
pcs_upper_5 = ep_med_test_5 + pcs_gamma * (ep_up_test_5 - ep_low_test_5) / 2.0
metrics_pcs_5 = evaluate_intervals(y_test_p, pcs_lower_5, pcs_upper_5, alpha=ALPHA_5, f=ep_med_test_5)

# --- Aleatoric baseline (CQR): QRF on raw targets, no epistemic component ---
al_raw_lo_5 = RandomForestQuantileRegressor(
    n_estimators=100, default_quantiles=ALPHA_5 / 2,
    min_samples_leaf=10, random_state=RANDOM_STATE
)
al_raw_hi_5 = RandomForestQuantileRegressor(
    n_estimators=100, default_quantiles=1 - ALPHA_5 / 2,
    min_samples_leaf=10, random_state=RANDOM_STATE
)
al_raw_lo_5.fit(X_train_p, y_train_p)
al_raw_hi_5.fit(X_train_p, y_train_p)
al_raw_sc_5 = np.maximum(
    al_raw_lo_5.predict(X_calib_p) - y_calib_p,
    y_calib_p - al_raw_hi_5.predict(X_calib_p),
)
al_raw_adj_5  = np.quantile(al_raw_sc_5, q_level_5, method="higher")
al_lower_5    = al_raw_lo_5.predict(X_test_p) - al_raw_adj_5
al_upper_5    = al_raw_hi_5.predict(X_test_p) + al_raw_adj_5
metrics_al_5  = evaluate_intervals(y_test_p, al_lower_5, al_upper_5, alpha=ALPHA_5, f=ep_med_test_5)

print("\n" + "=" * 60)
print("Comparison: CLEAR vs Baselines (Parkinsons, PCS)")
print("=" * 60)
comparison_table(
    {
        "CLEAR": metrics_5,
        "Aleatoric-R": metrics_ar_5,
        "Aleatoric (CQR)": metrics_al_5,
        "Epistemic (PCS)": metrics_pcs_5,
    },
    DESIRED_COVERAGE
)

# %% [markdown]
"""
## Example 6: Retrain PCS from Scratch on Airfoil

Shows the **full CLEAR + PCS pipeline** trained from scratch, no pre-trained models needed. We use the same Airfoil splits as Examples 1–3.

A lightweight PCS bootstrap ensemble is implemented inline:
1. Bootstrap `N_BOOTSTRAPS` QRF median predictors on the training set.
2. Raw epistemic intervals = quantiles of the bootstrap median predictions.
3. Multiplicative calibration → PCS baseline.
4. Aleatoric QRF trained on residuals of the PCS median.
5. CLEAR combines both components and finds the optimal λ.

Four methods are compared (same target coverage):
- **CLEAR**: adaptive epistemic + aleatoric combination
- **Aleatoric-R**: aleatoric QRF on residuals, centred on PCS median
- **Aleatoric (CQR)**: plain CQR — QRF trained on raw targets, no epistemic component
- **Epistemic (PCS)**: multiplicative calibration, epistemic-only
"""

# %%
print("\n" + "=" * 60)
print("Example 6: Retrain PCS from Scratch (Airfoil, QRF ensemble)")
print("=" * 60)

# Reuse the Airfoil splits from Examples 1–3 (already in memory)
X_train_np6 = np.asarray(X_train)
X_calib_np6 = np.asarray(X_calib)
X_test_np6  = np.asarray(X_test)
y_train_np6 = np.asarray(y_train).flatten()
y_calib_np6 = np.asarray(y_calib).flatten()

# ------------------------------------------------------------------
# Helper: lightweight PCS bootstrap (no external dependency)
# ------------------------------------------------------------------
def train_pcs_bootstrap(X_tr, y_tr, X_cal, X_te,
                        n_bootstraps=N_BOOTSTRAPS, alpha=ALPHA,
                        random_state=RANDOM_STATE):
    """
    Simplified PCS ensemble:
    - Bootstrap N QRF median predictors
    - Raw intervals = quantiles of bootstrap median predictions
    - Returns (train, calib, test) raw interval tuples + PCS gamma
    """
    from quantile_forest import RandomForestQuantileRegressor
    base = RandomForestQuantileRegressor(
        n_estimators=100, default_quantiles=0.5,
        min_samples_leaf=10, random_state=random_state
    )
    models = []
    for i in range(n_bootstraps):
        X_b, y_b = resample(X_tr, y_tr, random_state=random_state + i)
        m = copy.deepcopy(base)
        m.set_params(random_state=random_state + i)
        m.fit(X_b, y_b)
        models.append(m)
        print(f"  Bootstrap {i+1}/{n_bootstraps} done", end="\r")
    print()

    def intervals_from_bootstraps(X):
        preds = np.column_stack([m.predict(X) for m in models])  # (n, B)
        lo  = np.quantile(preds, alpha / 2,       axis=1)
        med = np.quantile(preds, 0.5,             axis=1)
        hi  = np.quantile(preds, 1 - alpha / 2,  axis=1)
        return lo, med, hi

    return intervals_from_bootstraps(X_tr), \
           intervals_from_bootstraps(X_cal), \
           intervals_from_bootstraps(X_te),  \
           models

# ------------------------------------------------------------------
# Step 1: Train PCS bootstrap ensemble
# ------------------------------------------------------------------
print("Training PCS bootstrap ensemble (QRF)...")
(ep_lo_tr6, ep_med_tr6, ep_hi_tr6), \
(ep_lo_cal6, ep_med_cal6, ep_hi_cal6), \
(ep_lo_te6,  ep_med_te6,  ep_hi_te6), \
pcs_models_6 = train_pcs_bootstrap(
    X_train_np6, y_train_np6, X_calib_np6, X_test_np6
)
print(f"PCS raw intervals ready. Epistemic spread (test median MPIW): "
      f"{np.mean(ep_hi_te6 - ep_lo_te6):.4f}")

# ------------------------------------------------------------------
# Step 2: Fit aleatoric QRF on PCS residuals
# ------------------------------------------------------------------
print("Fitting aleatoric QRF on PCS training residuals...")
residuals_tr6   = y_train_np6 - ep_med_tr6
X_tr_aug6  = np.column_stack([X_train_np6, ep_med_tr6])
X_cal_aug6 = np.column_stack([X_calib_np6, ep_med_cal6])
X_te_aug6  = np.column_stack([X_test_np6,  ep_med_te6])

al_lo6  = RandomForestQuantileRegressor(n_estimators=100, default_quantiles=ALPHA/2,       min_samples_leaf=10, random_state=RANDOM_STATE)
al_med6 = RandomForestQuantileRegressor(n_estimators=100, default_quantiles=0.5,           min_samples_leaf=10, random_state=RANDOM_STATE)
al_hi6  = RandomForestQuantileRegressor(n_estimators=100, default_quantiles=1 - ALPHA/2,  min_samples_leaf=10, random_state=RANDOM_STATE)
al_lo6.fit(X_tr_aug6, residuals_tr6)
al_med6.fit(X_tr_aug6, residuals_tr6)
al_hi6.fit(X_tr_aug6, residuals_tr6)

# Add PCS median back to put predictions in y-space
al_low_cal6 = al_lo6.predict(X_cal_aug6)  + ep_med_cal6
al_med_cal6 = al_med6.predict(X_cal_aug6) + ep_med_cal6
al_hi_cal6  = al_hi6.predict(X_cal_aug6)  + ep_med_cal6

al_low_te6  = al_lo6.predict(X_te_aug6)   + ep_med_te6
al_med_te6  = al_med6.predict(X_te_aug6)  + ep_med_te6
al_hi_te6   = al_hi6.predict(X_te_aug6)   + ep_med_te6

# ------------------------------------------------------------------
# Step 3: CLEAR calibration
# ------------------------------------------------------------------
clear_6 = CLEAR(desired_coverage=DESIRED_COVERAGE, random_state=RANDOM_STATE)
clear_6.calibrate(
    y_calib_np6,
    median_epistemic=ep_med_cal6,
    aleatoric_median=al_med_cal6,
    aleatoric_lower=al_low_cal6,
    aleatoric_upper=al_hi_cal6,
    epistemic_lower=ep_lo_cal6,
    epistemic_upper=ep_hi_cal6,
    verbose=False
)
print(f"Optimal Lambda: {clear_6.optimal_lambda:.4f}, Gamma: {clear_6.gamma:.4f}")

lower_6, upper_6 = clear_6.predict(
    X_test_np6,
    external_epistemic={"median": ep_med_te6, "lower": ep_lo_te6, "upper": ep_hi_te6},
    external_aleatoric={"median": al_med_te6, "lower": al_low_te6, "upper": al_hi_te6},
)
metrics_6 = evaluate_intervals(y_test, lower_6, upper_6, alpha=ALPHA, f=ep_med_te6)
print_all_metrics(metrics_6, "CLEAR (PCS retrained, Airfoil)")

# ------------------------------------------------------------------
# Step 4: Baselines
# ------------------------------------------------------------------
q_level_6 = min((1 - ALPHA) * (1 + 1 / len(y_calib_np6)), 1.0)

# Aleatoric-R: aleatoric-width intervals centred on PCS median, conformally adjusted
cal_lo_ar6 = ep_med_cal6 - (al_med_cal6 - al_low_cal6)
cal_hi_ar6 = ep_med_cal6 + (al_hi_cal6  - al_med_cal6)
ar_adj6    = np.quantile(
    np.maximum(cal_lo_ar6 - y_calib_np6, y_calib_np6 - cal_hi_ar6),
    q_level_6, method="higher"
)
ar_lo6 = (ep_med_te6 - (al_med_te6 - al_low_te6)) - ar_adj6
ar_hi6 = (ep_med_te6 + (al_hi_te6  - al_med_te6)) + ar_adj6
metrics_ar6 = evaluate_intervals(y_test, ar_lo6, ar_hi6, alpha=ALPHA, f=ep_med_te6)

# PCS: multiplicative recalibration to 95% on the calibration set
raw_half6  = (ep_hi_cal6 - ep_lo_cal6) / 2.0
pcs_sc6    = np.abs(y_calib_np6 - ep_med_cal6) / np.maximum(raw_half6, 1e-10)
pcs_gam6   = np.quantile(pcs_sc6, q_level_6, method="higher")
pcs_lo6    = ep_med_te6 - pcs_gam6 * (ep_hi_te6 - ep_lo_te6) / 2.0
pcs_hi6    = ep_med_te6 + pcs_gam6 * (ep_hi_te6 - ep_lo_te6) / 2.0
metrics_pcs6 = evaluate_intervals(y_test, pcs_lo6, pcs_hi6, alpha=ALPHA, f=ep_med_te6)

# Aleatoric: plain QRF on raw targets (no epistemic component)
al_raw_lo6 = RandomForestQuantileRegressor(
    n_estimators=100, default_quantiles=ALPHA / 2,
    min_samples_leaf=10, random_state=RANDOM_STATE
)
al_raw_hi6 = RandomForestQuantileRegressor(
    n_estimators=100, default_quantiles=1 - ALPHA / 2,
    min_samples_leaf=10, random_state=RANDOM_STATE
)
al_raw_lo6.fit(X_train_np6, y_train_np6)
al_raw_hi6.fit(X_train_np6, y_train_np6)
al_raw_sc6  = np.maximum(
    al_raw_lo6.predict(X_calib_np6) - y_calib_np6,
    y_calib_np6 - al_raw_hi6.predict(X_calib_np6),
)
al_raw_adj6 = np.quantile(al_raw_sc6, q_level_6, method="higher")
al_lower6   = al_raw_lo6.predict(X_test_np6) - al_raw_adj6
al_upper6   = al_raw_hi6.predict(X_test_np6) + al_raw_adj6
metrics_al6 = evaluate_intervals(y_test, al_lower6, al_upper6, alpha=ALPHA, f=ep_med_te6)

print("\n" + "=" * 60)
print("Comparison: CLEAR vs Baselines (Airfoil, retrained PCS)")
print("=" * 60)
comparison_table(
    {
        "CLEAR": metrics_6,
        "Aleatoric-R": metrics_ar6,
        "Aleatoric (CQR)": metrics_al6,
        "Epistemic (PCS)": metrics_pcs6,
    },
    DESIRED_COVERAGE
)

# %% [markdown]
"""
## Visualization

Plot prediction intervals for Example 5 (Parkinsons + PCS) comparing all four methods.
Intervals are sorted by target value so width and coverage differences are visible.
Plots are saved to ``plots/demo/`` and displayed inline.
"""

# %%
plot_dir = os.path.join("plots", "demo")
os.makedirs(plot_dir, exist_ok=True)

print("\nPlotting intervals — Example 5 (Parkinsons, PCS)...")
intervals_5 = {
    "CLEAR":            (lower_5,     upper_5,     metrics_5),
    "Aleatoric-R":      (ar_lower_5,  ar_upper_5,  metrics_ar_5),
    "Aleatoric (CQR)":  (al_lower_5,  al_upper_5,  metrics_al_5),
    "Epistemic (PCS)":  (pcs_lower_5, pcs_upper_5, metrics_pcs_5),
}
plot_prediction_intervals(
    X_test_p, y_test_p, intervals_5,
    dataset_name="parkinsons",
    run_key="demo_run_0",
    coverage_target=DESIRED_COVERAGE,
    base_plot_dir=plot_dir,
    display=True,
)
print(f"Saved to: {os.path.join(plot_dir, 'parkinsons')}")
