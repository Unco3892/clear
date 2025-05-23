"""
Python script to run simulations as described in the CLEAR paper.
Evaluates CLEAR and baseline model performance at different distances from the origin
for various dimensions and noise settings.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import argparse
import time
from pygam import ExpectileGAM, LinearGAM
import matplotlib as mpl

# --- Path Setup ---
# Add project root and other necessary paths to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..')) # Assuming src/experiments -> clear/
src_path = os.path.join(project_root, 'src') # Explicitly define src path

if project_root not in sys.path:
    sys.path.append(project_root)
    print(f"Appended project root: {project_root}")
if src_path not in sys.path: # Add src path
    sys.path.append(src_path)
    print(f"Appended src path: {src_path}")

# Add path for CLEAR module (now relative to src path is less critical)
clear_module_path = os.path.join(project_root, "src", "clear")
if clear_module_path not in sys.path:
    sys.path.append(clear_module_path)
    print(f"Appended CLEAR module path: {clear_module_path}")

# Add path for PCS_UQ module (assuming it's parallel to the main project dir)
pcs_uq_base_path = os.path.abspath(os.path.join(project_root, 'PCS_UQ'))
pcs_uq_src_path = os.path.join(pcs_uq_base_path, 'src')
external_pcs_available = False
if os.path.isdir(pcs_uq_src_path):
    if pcs_uq_src_path not in sys.path:
        sys.path.insert(0, pcs_uq_src_path) # Insert at beginning
        print(f"Added PCS_UQ src path: {pcs_uq_src_path}")
    try:
        from PCS.regression.pcs_uq import PCS_UQ
        # Import necessary base models if using external PCS_UQ
        # from sklearn.ensemble import RandomForestRegressor
        # from xgboost import XGBRegressor
        # Using ExpectileGAM as in the original benchmark script for now
        from pygam import ExpectileGAM
        external_pcs_available = True
        print("External PCS_UQ module loaded successfully.")
    except ImportError as e:
        print(f"Warning: Failed to import PCS_UQ or its dependencies ({e}). External PCS mode will be unavailable.")
        PCS_UQ = None
    except Exception as e:
        print(f"An unexpected error occurred during PCS_UQ import: {e}")
        PCS_UQ = None
else:
    print(f"Warning: PCS_UQ directory not found at expected location: {pcs_uq_base_path}")
    PCS_UQ = None

# Import CLEAR components AFTER setting up paths
try:
    from clear.clear import CLEAR
    from clear.utils import generate_distance_points, compute_coverage_by_distance, plot_distance_metrics
    # Note: calibrate_ensemble_bounds is internal to CLEAR now or handled differently
except ImportError as e:
    print(f"Error importing CLEAR components: {e}")
    print("Please ensure the project structure is correct and paths are set up properly.")
    sys.exit(1)

# --- Simulation Functions ---

def generate_mean_function(d):
    """Generates a mean function mu(X) and the random betas for a given dimension d."""
    # Generate random betas for the simulation instance
    betas = np.random.normal(1, 0.5, d) # Generate d betas

    def mu(X):
        """Mean function for dimension d."""
        n_samples = X.shape[0]
        y_mu = np.full(n_samples, 5.0) # Start with base value 5
        for i in range(d):
            term = betas[i] * np.abs(X[:, i])**(1.5 if i % 2 == 0 else 1.25) # Alternate exponents slightly
            if i % 2 == 0:
                y_mu += term
            else:
                y_mu -= term
        return y_mu

    return mu, betas # Return the function and the betas used

def get_sigma(X, d, noise_type='homo'):
    """Calculates the standard deviation sigma(X) based on dimension and noise type."""
    if d == 1:
        x_norm = np.abs(X[:, 0]) # For d=1, ||x|| is |x|
        if noise_type == 'homo': # sigma_1
            return np.ones_like(x_norm)
        elif noise_type == 'hetero1': # sigma_2
            return 1.0 + x_norm
        elif noise_type == 'hetero2': # sigma_3
            return 1.0 + 1.0 / (1.0 + x_norm**2)
        else:
            raise ValueError(f"Unknown noise_type '{noise_type}' for d=1")
    else: # For d > 1, use homo noise as per paper interpretation
        if noise_type != 'homo':
            print(f"Warning: noise_type '{noise_type}' ignored for d={d}. Using homoscedastic noise.")
        return np.ones(X.shape[0])

def run_simulation(
    d_fixed=None, # Changed from d
    randomize_d_flag=False, # New flag
    dims_to_randomize=None, # List of dims if randomizing
    noise_type='homo',
    num_simulations=20,
    n_train_calib=2000,
    coverage_level=0.90,
    base_path=".",
    log_details=False,
    use_external_pcs=False,
    n_boot_clear=50, # Defaulting to 50 for CLEAR internal models
    n_boot_pcs=50,   # Defaulting to 50 for external PCS
    cqr_center_source="pcs",
    fit_on_residuals=True
    ):
    """
    Run the simulation for a specific dimension and noise type,
    or randomize dimension across simulations.
    """
    if use_external_pcs and not external_pcs_available:
        raise RuntimeError("External PCS mode selected, but PCS_UQ module or dependencies could not be imported.")
    if d_fixed != 1 and noise_type != 'homo':
        print(f"Warning: For d={d_fixed}, only 'homoscedastic' noise_type is supported based on paper. Ignoring '{noise_type}'.")
        noise_type = 'homo'

    alpha = 1.0 - coverage_level
    epistemic_mode_str = ("External_PCS_UQ" if use_external_pcs else "Internal_GAM").lower()
    if randomize_d_flag:
        current_d_str = "random"
        title_d_part = f"d=randomized in {dims_to_randomize}"
    else:
        current_d_str = str(d_fixed)
        title_d_part = f"d={d_fixed}"
    output_subdir = f"d{current_d_str}/{noise_type}_{epistemic_mode_str}"
    output_dir = os.path.join(base_path, "plots/simulations", output_subdir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"--- Starting Simulation: {title_d_part}, noise='{noise_type}', epistemic='{epistemic_mode_str}' ---")
    print(f"Output directory: {output_dir}")

    # Initialize storage for results across all simulations
    all_results = {'CLEAR': [], 'PCS': [], 'CQR': [], 'S-Naive': []}
    all_results_width = {'CLEAR': [], 'PCS': [], 'CQR': [], 'S-Naive': []}
    optimal_lambdas = []
    optimal_gammas = []
    simulation_betas = [] # Store betas for logging

    # Distance sequence (matches R code: conditional_sequence = 0.01 + seq(0, 6, 0.25))
    distances = 0.01 + np.arange(0, 6.01, 0.25)
    n_points_per_distance = 1000  # matches R code
    # n_points_per_distance = n_train_calib

    log_file = None
    if log_details:
        log_filename = f"simulation_log_d{current_d_str}_{noise_type}_{epistemic_mode_str}.txt"
        log_file = os.path.join(output_dir, log_filename)
        with open(log_file, 'w') as f:
            f.write(f"Simulation Log: d={current_d_str}, noise='{noise_type}', epistemic='{epistemic_mode_str}'\n")
            f.write(f"=====================================================\n")
            # Use current_d_str for logging consistency if d_fixed might be None
            f.write(f"Dimension (d): {current_d_str if randomize_d_flag else d_fixed}\n")
            f.write(f"Noise type: {noise_type}\n")
            f.write(f"Using external PCS: {use_external_pcs}\n")
            f.write(f"CQR centering source: {cqr_center_source}\n")
            f.write(f"Fit aleatoric on residuals: {fit_on_residuals}\n")
            f.write(f"Number of simulations: {num_simulations}\n")
            f.write(f"Train/Calib size: {n_train_calib} (Split 70/30, with training size as {n_train_calib})\n")
            f.write(f"Target coverage: {coverage_level:.2f}\n")
            f.write(f"CLEAR bootstraps: {n_boot_clear}\n")
            if use_external_pcs:
                f.write(f"PCS_UQ bootstraps: {n_boot_pcs}\n")
            f.write(f"Points per distance: {n_points_per_distance}\n")
            f.write(f"Distances: {distances}\n\n")

    # Run multiple simulations
    for sim_idx in range(num_simulations):
        # --- Determine Dimension 'd' for this run ---
        if randomize_d_flag:
            current_d = np.random.choice(dims_to_randomize)
            dimensions_used = dims_to_randomize if randomize_d_flag else [d_fixed]
            print(f"\nRunning simulation {sim_idx+1}/{num_simulations} (Random d={current_d}, noise='{noise_type}')")
        else:
            current_d = d_fixed
            print(f"\nRunning simulation {sim_idx+1}/{num_simulations} (Fixed d={current_d}, noise='{noise_type}')")
        # -------------------------------------------

        # Determine noise type based on current_d for this run
        current_noise_type = noise_type if current_d == 1 else 'homo'

        # --- Use 'current_d' and 'current_noise_type' throughout the rest of the loop ---
        # e.g., in generate_mean_function(current_d), get_sigma(X, current_d, current_noise_type),
        # generate_distance_points(n_dims=current_d), etc.
        # Also update logging messages within the loop if log_details is True

        start_time = time.time()
        sim_seed = sim_idx # Use simulation index as base seed

        # Generate mean function for this simulation run
        mu, current_betas = generate_mean_function(current_d)
        simulation_betas.append(current_betas)

        if log_details and log_file:
            with open(log_file, 'a') as f:
                f.write(f"Simulation {sim_idx+1}\n")
                f.write(f"-------------\n")
                f.write(f"Current d for this run: {current_d}, Betas: {np.round(current_betas, 4)}\n")

        # Generate training data
        X = np.random.normal(0, 1, size=(n_train_calib, current_d))
        sigma_vals = get_sigma(X, current_d, current_noise_type)
        Y = mu(X) + sigma_vals * np.random.normal(0, 1, n_train_calib)

        # Generate test data at specified distances from origin
        X_test_flat = generate_distance_points(
            n_points_per_distance=n_points_per_distance,
            distances=distances,
            n_dims=current_d,
            random_state=sim_seed + 1000 # Offset seed
        )

        # Generate the true Y values for test data
        sigma_vals_test = get_sigma(X_test_flat, current_d, current_noise_type)
        Y_test = mu(X_test_flat) + sigma_vals_test * np.random.normal(0, 1, len(X_test_flat))

        # Split data into training and calibration sets (70/30)
        X_train, X_calib, y_train, y_calib = train_test_split(
            X, Y, test_size=0.3, random_state=sim_seed + 42
        )

        # Create CLEAR model instance
        clear_model = CLEAR(
            desired_coverage=coverage_level,
            n_bootstraps=n_boot_clear,
            lambdas=np.concatenate((np.linspace(0, 0.09, 10), np.logspace(-1, 2, 100))),
            # lambdas=np.arange(1, 10.1, 0.1), # Reduced range for speed, adjust if needed
            random_state=sim_seed + 100,
            n_jobs=-1 # Use all available cores
        )

        # --- Epistemic Part ---
        if use_external_pcs:
            print("Fitting epistemic models (External PCS_UQ)...")
            # Define base models for PCS ensemble - Using ExpectileGAM for now
            # TODO: Consider allowing RF/XGBoost selection via args
            # models = {"ExpectileGAM": ExpectileGAM(expectile=0.5, n_splines=10, lam=1e-11, spline_order=3)}
            from sklearn.exceptions import ConvergenceWarning
            from pygam import ExpectileGAM
            from xgboost import XGBRegressor
            from quantile_forest import RandomForestQuantileRegressor, ExtraTreesQuantileRegressor
            import warnings
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            from sklearn.neural_network import MLPRegressor
            models = {
                # "MLP": MLPRegressor(random_state = 42, hidden_layer_sizes = (64,)),
                # # "ExpectileGAM": ExpectileGAM(expectile=0.5, n_splines=10, lam=0),
                "QRF": RandomForestQuantileRegressor(random_state=777, n_estimators=100, default_quantiles=0.5, min_samples_leaf = 10),
                "QXGB": XGBRegressor(random_state=777, objective='reg:quantileerror', n_estimators=100, tree_method='hist', quantile_alpha=0.5, min_child_weight = 10),
                "LinearGAM": LinearGAM(n_splines=10, lam=1)
                }

            pcs_uq = PCS_UQ(
                models=models,
                num_bootstraps=n_boot_pcs,
                alpha=alpha, # Pass alpha directly
                seed=sim_seed + 400,
                top_k=1, # Select top 1 model based on validation performance
                calibration_method='multiplicative' # Or 'additive'
            )
            # Note: PCS_UQ might need separate validation set, but benchmark script uses calib set
            pcs_uq.fit(X=X_train, y=y_train, X_calib=X_calib, y_calib=y_calib)

            # Get RAW predictions needed later
            raw_intervals_train = pcs_uq.get_intervals(X_train)
            raw_intervals_calib = pcs_uq.get_intervals(X_calib)
            raw_intervals_test = pcs_uq.get_intervals(X_test_flat)

            pcs_median_train_raw = raw_intervals_train[:, 1]
            pcs_lower_train_raw = raw_intervals_train[:, 0]
            pcs_upper_train_raw = raw_intervals_train[:, 2]
            pcs_median_calib_raw = raw_intervals_calib[:, 1]
            pcs_lower_calib_raw = raw_intervals_calib[:, 0]
            pcs_upper_calib_raw = raw_intervals_calib[:, 2]
            pcs_median_test_raw = raw_intervals_test[:, 1]
            pcs_lower_test_raw = raw_intervals_test[:, 0]
            pcs_upper_test_raw = raw_intervals_test[:, 2]

            # Get CALIBRATED PCS bounds for the 'PCS' baseline comparison
            print("  Getting CALIBRATED PCS intervals for benchmark comparison.")
            calibrated_intervals_pcs_test = pcs_uq.predict(X_test_flat)
            lower_bounds_pcs = calibrated_intervals_pcs_test[:, 0]
            upper_bounds_pcs = calibrated_intervals_pcs_test[:, 1] # PCS_UQ predict returns [lower, upper]

            # Print Model Performance Scores if available
            if hasattr(pcs_uq, 'pred_scores') and pcs_uq.pred_scores:
                 print("============================")
                 print("  External PCS_UQ Base Model Performance Scores (on calib):")
                 sorted_scores = sorted(pcs_uq.pred_scores.items(), key=lambda item: item[1])
                 for model_name, score in sorted_scores:
                     print(f"    - {model_name}: {score:.4f}")
                 print("============================")

        else: # Use internal GAM-based epistemic model
            print("Fitting epistemic models (CLEAR internal GAM)...")
            clear_model.fit_epistemic(X_train, y_train)

            # Predict epistemic bounds on ALL splits (raw, uncalibrated)
            pcs_median_train_raw, pcs_lower_train_raw, pcs_upper_train_raw, _ = \
                clear_model.predict_epistemic(X_train, symmetric_noise=False)
            pcs_median_calib_raw, pcs_lower_calib_raw, pcs_upper_calib_raw, _ = \
                clear_model.predict_epistemic(X_calib, symmetric_noise=False)
            pcs_median_test_raw, pcs_lower_test_raw, pcs_upper_test_raw, _ = \
                clear_model.predict_epistemic(X_test_flat, symmetric_noise=False)

            # For the "PCS" comparison method, use simple conformal calibration on GAM ensemble
            print("Generating PCS-only comparison bounds (from internal GAM ensemble)...")
            # Calculate scores (normalized by raw epistemic deviations)
            epsilon = 1e-8
            left_dev_calib = np.maximum(pcs_median_calib_raw - pcs_lower_calib_raw, epsilon)
            right_dev_calib = np.maximum(pcs_upper_calib_raw - pcs_median_calib_raw, epsilon)
            left_dev_test = np.maximum(pcs_median_test_raw - pcs_lower_test_raw, epsilon)
            right_dev_test = np.maximum(pcs_upper_test_raw - pcs_median_test_raw, epsilon)

            scores_pcs = np.maximum(
                (pcs_median_calib_raw - y_calib) / left_dev_calib,
                (y_calib - pcs_median_calib_raw) / right_dev_calib
            )

            # Get conformal quantile adjustment
            n_calib_pcs = len(y_calib)
            conf_level_pcs = (1.0 - alpha) * (1.0 + 1.0 / n_calib_pcs)
            conf_level_pcs = min(conf_level_pcs, 1.0) # Ensure level doesn't exceed 1
            gamma_pcs = np.quantile(scores_pcs, conf_level_pcs, method='higher')

            # Apply adjustment to test deviations
            lower_bounds_pcs = pcs_median_test_raw - gamma_pcs * left_dev_test
            upper_bounds_pcs = pcs_median_test_raw + gamma_pcs * right_dev_test
            print(f"  Internal GAM PCS calibration: gamma_pcs={gamma_pcs:.4f}")

        # --- Aleatoric Part ---
        print("Fitting aleatoric models...")
        clear_model.fit_aleatoric(
            X_train, y_train,
            # quantile_model="qgam", # Using GAM to match R code simulation setup
            # model_params={'n_splines': 10, 'lam': 1e-11}, # Params used in benchmark script
            quantile_model="xgb",
            fit_on_residuals=fit_on_residuals,
            epistemic_preds=pcs_median_train_raw if fit_on_residuals else None
        )

        # Get aleatoric predictions for calibration and test data
        # Need to provide epistemic predictions if fitting on residuals
        X_all = np.vstack([X_calib, X_test_flat])
        pcs_median_all_raw = np.concatenate([pcs_median_calib_raw, pcs_median_test_raw]) if fit_on_residuals else None

        aleatoric_median, aleatoric_lower, aleatoric_upper = clear_model.predict_aleatoric(
            X_all,
            epistemic_preds=pcs_median_all_raw
        )

        # Split predictions back into calibration and test sets
        n_calib_split = len(X_calib)
        aleatoric_median_calib = aleatoric_median[:n_calib_split]
        aleatoric_lower_calib = aleatoric_lower[:n_calib_split]
        aleatoric_upper_calib = aleatoric_upper[:n_calib_split]
        aleatoric_median_test = aleatoric_median[n_calib_split:]
        aleatoric_lower_test = aleatoric_lower[n_calib_split:]
        aleatoric_upper_test = aleatoric_upper[n_calib_split:]

        # --- Calibrate CLEAR ---
        print("Calibrating CLEAR model...")
        clear_model.calibrate(
            y_calib,
            median_epistemic=pcs_median_calib_raw, # RAW epistemic median
            aleatoric_median=aleatoric_median_calib, # Aleatoric median
            aleatoric_lower=aleatoric_lower_calib,   # Aleatoric bounds
            aleatoric_upper=aleatoric_upper_calib,
            epistemic_lower=pcs_lower_calib_raw,   # RAW epistemic bounds
            epistemic_upper=pcs_upper_calib_raw
        )
        optimal_lambdas.append(clear_model.optimal_lambda)
        optimal_gammas.append(clear_model.gamma)

        # --- Generate Final Predictions ---
        print("Generating final CLEAR predictions...")
        lower_bounds_clear, upper_bounds_clear = clear_model.predict(
            X_test_flat,
            external_epistemic={ # Pass RAW epistemic preds
                'median': pcs_median_test_raw,
                'lower': pcs_lower_test_raw,
                'upper': pcs_upper_test_raw
            },
            external_aleatoric={ # Pass aleatoric preds
                'median': aleatoric_median_test,
                'lower': aleatoric_lower_test,
                'upper': aleatoric_upper_test
            }
        )

        # --- CQR Baseline ---
        print("Generating CQR-only comparison bounds...")
        # Select Median Source for CQR Centering
        if cqr_center_source == "aleatoric":
            cqr_median_calib = aleatoric_median_calib
            cqr_median_test = aleatoric_median_test
            print("  Using ALEATORIC median for CQR centering.")
        elif cqr_center_source == "pcs":
            cqr_median_calib = pcs_median_calib_raw # Use RAW epistemic median
            cqr_median_test = pcs_median_test_raw
            print("  Using EPISTEMIC (PCS/GAM) median for CQR centering.")
        else:
            raise ValueError(f"Invalid cqr_center_source: {cqr_center_source}.")

        # Center raw aleatoric bounds around the CHOSEN median
        # Widths: aleatoric_upper - aleatoric_median, aleatoric_median - aleatoric_lower
        raw_upper_cqr_calib = cqr_median_calib + (aleatoric_upper_calib - aleatoric_median_calib)
        raw_lower_cqr_calib = cqr_median_calib - (aleatoric_median_calib - aleatoric_lower_calib)

        # Calculate non-conformity scores (CQR style)
        scores_cqr = np.maximum(raw_lower_cqr_calib - y_calib, y_calib - raw_upper_cqr_calib)

        # Apply conformal calibration adjustment
        n_calib_cqr = len(y_calib)
        q_level_cqr = (1.0 - alpha) * (1.0 + 1.0 / n_calib_cqr)
        q_level_cqr = min(q_level_cqr, 1.0)
        adjustment_cqr = np.quantile(scores_cqr, q_level_cqr, method='higher')

        # Generate calibrated CQR intervals using the CHOSEN median center
        raw_upper_cqr_test = cqr_median_test + (aleatoric_upper_test - aleatoric_median_test)
        raw_lower_cqr_test = cqr_median_test - (aleatoric_median_test - aleatoric_lower_test)
        lower_bounds_cqr = raw_lower_cqr_test - adjustment_cqr
        upper_bounds_cqr = raw_upper_cqr_test + adjustment_cqr

        # --- Naive Baseline ---
        print("Generating Naive comparison bounds...")
        # Select Median Source for Naive Centering (consistent with CQR choice)
        if cqr_center_source == "aleatoric":
            naive_median_calib = aleatoric_median_calib
            naive_median_test = aleatoric_median_test
            print("  Using ALEATORIC median for Naive centering.")
        else: # pcs
            naive_median_calib = pcs_median_calib_raw
            naive_median_test = pcs_median_test_raw
            print("  Using EPISTEMIC (PCS/GAM) median for Naive centering.")

        # Calculate absolute residuals on calibration set using the chosen median
        calib_scores_naive = np.abs(y_calib - naive_median_calib)
        q_level_naive = (1.0 - alpha) * (1.0 + 1.0 / n_calib_cqr) # Same n_calib
        q_level_naive = min(q_level_naive, 1.0)
        adjustment_naive = np.quantile(calib_scores_naive, q_level_naive, method='higher')

        # Apply constant adjustment to test set median
        lower_bounds_naive = naive_median_test - adjustment_naive
        upper_bounds_naive = naive_median_test + adjustment_naive

        # --- Compute and Store Metrics ---
        print("Computing metrics...")
        metrics_clear = compute_coverage_by_distance(X_test_flat, lower_bounds_clear, upper_bounds_clear, Y_test, distances, n_points_per_distance)
        metrics_pcs = compute_coverage_by_distance(X_test_flat, lower_bounds_pcs, upper_bounds_pcs, Y_test, distances, n_points_per_distance)
        metrics_cqr = compute_coverage_by_distance(X_test_flat, lower_bounds_cqr, upper_bounds_cqr, Y_test, distances, n_points_per_distance)
        metrics_naive = compute_coverage_by_distance(X_test_flat, lower_bounds_naive, upper_bounds_naive, Y_test, distances, n_points_per_distance)

        all_results['CLEAR'].append(metrics_clear['coverage'])
        all_results['PCS'].append(metrics_pcs['coverage'])
        all_results['CQR'].append(metrics_cqr['coverage'])
        all_results['S-Naive'].append(metrics_naive['coverage'])

        all_results_width['CLEAR'].append(metrics_clear['width'])
        all_results_width['PCS'].append(metrics_pcs['width'])
        all_results_width['CQR'].append(metrics_cqr['width'])
        all_results_width['S-Naive'].append(metrics_naive['width'])

        # Log detailed metrics if enabled
        if log_details and log_file:
             with open(log_file, 'a') as f:
                 f.write(f"Optimal lambda: {clear_model.optimal_lambda:.4f}, Gamma: {clear_model.gamma:.4f}\n")
                 f.write(f"Avg Coverage - CLEAR: {np.mean(metrics_clear['coverage']):.4f}, PCS: {np.mean(metrics_pcs['coverage']):.4f}, CQR: {np.mean(metrics_cqr['coverage']):.4f}, Naive: {np.mean(metrics_naive['coverage']):.4f}\n")
                 f.write(f"Avg Width    - CLEAR: {np.mean(metrics_clear['width']):.4f}, PCS: {np.mean(metrics_pcs['width']):.4f}, CQR: {np.mean(metrics_cqr['width']):.4f}, Naive: {np.mean(metrics_naive['width']):.4f}\n")
                 # Add current_d if randomized, for clarity in log
                 if randomize_d_flag:
                     f.write(f"Dimension for this run (d): {current_d}\n")
                 f.write("\n") # Separator

        elapsed = time.time() - start_time
        print(f"Simulation {sim_idx+1} completed in {elapsed:.1f} seconds")
        if not log_details:
            print(f"  Optimal lambda: {clear_model.optimal_lambda:.4f}, Gamma: {clear_model.gamma:.4f}")
            print(f"  Avg Coverage - CLEAR: {np.mean(metrics_clear['coverage']):.4f}")

    # --- Aggregate and Plot Results ---
    avg_results = {m: np.mean(all_results[m], axis=0) for m in all_results}
    avg_results_width = {m: np.mean(all_results_width[m], axis=0) for m in all_results_width}

    # Log overall average results
    avg_lambda = np.mean(optimal_lambdas)
    std_lambda = np.std(optimal_lambdas)
    avg_gamma = np.mean(optimal_gammas)
    std_gamma = np.std(optimal_gammas)
    print("\n" + "="*40)
    print(f"Summary ({title_d_part}, noise='{noise_type}', epistemic='{epistemic_mode_str}')")
    print("="*40)
    print(f"Average optimal lambda: {avg_lambda:.4f} (std: {std_lambda:.4f})")
    print(f"Average gamma: {avg_gamma:.4f} (std: {std_gamma:.4f})")
    print(f"\nOverall average coverage:")
    for m in avg_results: print(f"  {m}: {np.mean(avg_results[m]):.4f}")
    print(f"\nOverall average width:")
    for m in avg_results_width: print(f"  {m}: {np.mean(avg_results_width[m]):.4f}")

    if log_details and log_file:
        with open(log_file, 'a') as f:
            f.write("\nSummary of All Simulations\n")
            f.write("========================\n")
            f.write(f"Average optimal lambda: {avg_lambda:.4f} (std: {std_lambda:.4f})\n")
            f.write(f"Average gamma: {avg_gamma:.4f} (std: {std_gamma:.4f})\n")
            f.write("\nOverall average coverage:\n")
            for m in avg_results: f.write(f"  {m}: {np.mean(avg_results[m]):.4f}\n")
            f.write("\nOverall average width:\n")
            for m in avg_results_width: f.write(f"  {m}: {np.mean(avg_results_width[m]):.4f}\n")
            # Log average by distance maybe?

    # Apply font and style settings for plotting
    plt.style.use('seaborn-v0_8-whitegrid')
    mpl.rcParams.update({
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 17,
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        # "savefig.dpi": 600, # Overridden locally for PNG/PDF
        "font.family": "serif",
        "font.serif": ["Palatino", "Times New Roman", "DejaVu Serif"],
    })

    # Plot final results
    # Original method keys and their desired plotting order, colors, and display names
    all_method_keys_original_order = ['CLEAR', 'PCS', 'CQR', 'S-Naive']
    original_colors_map = {'CLEAR': 'red', 'PCS': 'blue', 'CQR': 'green', 'S-Naive': 'purple'}
    original_display_names_map = {'CLEAR': 'CLEAR', 'PCS': 'PCS', 'CQR': 'CQR-R+', 'S-Naive': 'S-Naive'}

    # Start with all methods and their data
    methods_to_plot_final = list(all_method_keys_original_order)
    # Filter methods_to_plot_final to only those present in avg_results to avoid KeyErrors
    methods_to_plot_final = [m for m in methods_to_plot_final if m in avg_results]

    current_avg_coverage = {k: avg_results[k] for k in methods_to_plot_final if k in avg_results}
    current_avg_width = {k: avg_results_width[k] for k in methods_to_plot_final if k in avg_results_width}

    # Conditional removal of 'S-Naive' for fixed d=1, homo noise plot
    # Note: current_d_str, noise_type, and randomize_d_flag determine the plot type
    is_fixed_d1_homo_plot_scenario = (current_d_str == '1' and not randomize_d_flag and noise_type == 'homo')

    if is_fixed_d1_homo_plot_scenario:
        print("Modifying plot data for d=1, homo noise: Removing 'S-Naive' method.")
        if 'S-Naive' in current_avg_coverage: del current_avg_coverage['S-Naive']
        if 'S-Naive' in current_avg_width: del current_avg_width['S-Naive']
        if 'S-Naive' in methods_to_plot_final: methods_to_plot_final.remove('S-Naive')

    # Prepare colors and display names for the methods that will actually be plotted
    plot_colors_final = [original_colors_map[m] for m in methods_to_plot_final]
    plot_display_names_final = [original_display_names_map[m] for m in methods_to_plot_final]

    # Output filenames (base name without extension)
    plot_filename_base = f"avg_metrics_d{current_d_str}_{noise_type}_{epistemic_mode_str}"
    base_plot_path = os.path.join(output_dir, plot_filename_base)

    title_suffix_for_plot = f"d={current_d_str}, noise={noise_type}, {epistemic_mode_str}, {num_simulations} sims"
    
    if current_d_str == '1' and not randomize_d_flag:
        # For fixed d=1, use the symmetric plot
        # Pass the (potentially filtered) data and the base path for output files
        plot_1d_simulation_results_symmetric(
            positive_distances=distances,
            avg_coverage_metrics_dict=current_avg_coverage,
            avg_width_metrics_dict=current_avg_width,
            target_coverage=coverage_level,
            output_path_base=base_plot_path, # Pass base path
            title_suffix=title_suffix_for_plot
        )
    else:
        # For d > 1 or randomized d, use the original plot_distance_metrics
        # 'S-Naive' method is kept for these scenarios as per logic above
        # metrics_plot_dict_avg is already based on all methods from avg_results
        
        # Reconstruct metrics_plot_dict using current_avg_coverage and current_avg_width
        # which for this branch will contain all methods.
        metrics_plot_dict_for_clear_utils = {
            method: {'distances': distances, 'coverage': current_avg_coverage[method], 'width': current_avg_width[method]}
            for method in methods_to_plot_final # these are all methods if not fixed d1 homo
        }

        fig, axes = plot_distance_metrics(
            metrics_plot_dict_for_clear_utils, # Use the dict with all methods for this case
            methods_to_plot_final,             # Full list of method keys
            plot_colors_final,                 # Colors aligned with full list
            plot_display_names_final,          # Display names aligned with full list
            target_coverage=coverage_level
        )
        axes[1].set_ylim(bottom=0, top=30) # Cap width plot y-axis at 30
        
        fig.suptitle(None) # No title for any output

        # Remove individual legends from subplots if they exist
        for ax in axes:
            if ax.get_legend() is not None:
                ax.get_legend().remove()

        # Create a shared legend below the plots
        handles, labels = [], []
        # Collect handles and labels from the axes (plot_distance_metrics should have plotted them)
        # We use plot_display_names_final and plot_colors_final to reconstruct proxy artists if needed,
        # but first try to get them from axes.
        # Assuming plot_distance_metrics plots lines that can be retrieved for the legend.
        # It's safer to use the predefined labels and colors to create proxy artists for clarity.
        
        proxy_handles = []
        for i, method_key in enumerate(methods_to_plot_final):
            proxy_handles.append(mpl.lines.Line2D([0], [0], color=plot_colors_final[i], lw=2.5, label=plot_display_names_final[i]))
        
        fig.legend(handles=proxy_handles, loc='lower center', ncol=len(methods_to_plot_final), bbox_to_anchor=(0.5, 0))
        
        plt.tight_layout(rect=[0, 0.08, 1, 0.92]) # Adjust rect for no suptitle and external legend

        png_output_path = base_plot_path + ".png"
        pdf_output_path = base_plot_path + ".pdf"

        plt.savefig(pdf_output_path, format='pdf')
        print(f"Saved multivariate PDF plot (no title, shared legend): {pdf_output_path}")

        plt.savefig(png_output_path, dpi=800)
        print(f"Saved multivariate PNG plot (no title, shared legend): {png_output_path}")
        
        plt.close(fig)

    # Create and save DataFrames
    coverage_df = pd.DataFrame({'distance': distances, **avg_results})
    width_df = pd.DataFrame({'distance': distances, **avg_results_width})
    csv_filename_cov = f"coverage_results_d{current_d_str}_{noise_type}_{epistemic_mode_str}.csv"
    csv_filename_wid = f"width_results_d{current_d_str}_{noise_type}_{epistemic_mode_str}.csv"
    coverage_df.to_csv(os.path.join(output_dir, csv_filename_cov), index=False)
    width_df.to_csv(os.path.join(output_dir, csv_filename_wid), index=False)
    print(f"Saved results CSV: {os.path.join(output_dir, csv_filename_cov)}")
    print(f"Saved results CSV: {os.path.join(output_dir, csv_filename_wid)}")

    return coverage_df, width_df


def plot_1d_simulation_results_symmetric(
    positive_distances,
    avg_coverage_metrics_dict, # {'CLEAR': array, 'PCS': array, ...}
    avg_width_metrics_dict,    # {'CLEAR': array, 'PCS': array, ...}
    target_coverage,
    output_path_base, # Changed from output_path_and_filename
    title_suffix
):
    """
    Generates a 1D plot similar to the provided R-generated image,
    assuming results are symmetric around x=0.
    """
    x_plot = np.concatenate((-positive_distances[::-1], positive_distances))
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5)) # Adjusted figsize for legend
    
    # Ensure methods_order matches the keys present in the dicts
    methods_order = [m for m in ['CLEAR', 'PCS', 'CQR', 'S-Naive'] if m in avg_coverage_metrics_dict]

    colors = {'CLEAR': 'red', 'PCS': 'blue', 'CQR': 'green', 'S-Naive': 'purple'}
    # Labels as in the user-provided image, with CQR updated
    plot_labels = {'CLEAR': 'CLEAR', 'PCS': 'PCS', 'CQR': 'CQR-R+', 'S-Naive': 'S-Naive'}

    # Coverage subplot (axes[0])
    for method_key in methods_order:
        if method_key in avg_coverage_metrics_dict:
            coverage_values = avg_coverage_metrics_dict[method_key]
            plot_coverage = np.concatenate((coverage_values[::-1], coverage_values))
            axes[0].plot(x_plot, plot_coverage, color=colors[method_key], label=plot_labels.get(method_key, method_key), linewidth=2.5)
    
    axes[0].axhline(target_coverage, color='black', linestyle='--', linewidth=1)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('Conditional Coverage')
    axes[0].set_ylim([0.35, 1.05]) # Based on user image
    axes[0].grid(True, alpha=0.3, linestyle=':')

    # Width subplot (axes[1])
    max_width_plot = 0
    for method_key in methods_order:
        if method_key in avg_width_metrics_dict:
            width_values = avg_width_metrics_dict[method_key]
            plot_width = np.concatenate((width_values[::-1], width_values))
            axes[1].plot(x_plot, plot_width, color=colors[method_key], label=plot_labels.get(method_key, method_key), linewidth=2.5)
            if len(plot_width) > 0: # Ensure plot_width is not empty
                 max_width_plot = max(max_width_plot, np.nanmax(plot_width[np.isfinite(plot_width)]))

    axes[1].set_xlabel('x')
    axes[1].set_ylabel('Average width')
    if max_width_plot > 0 : # Set ylim only if there's valid data
        axes[1].set_ylim(bottom=0, top=30) # Dynamic based on data, cap at 30
    else:
        axes[1].set_ylim(bottom=0, top=25) # Default if no data
    axes[1].grid(True, alpha=0.3, linestyle=':')

    # Shared legend below plots
    handles, labels = [], []
    for ax in fig.axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    
    fig.legend(handles, labels, loc='lower center', ncol=len(methods_order), bbox_to_anchor=(0.5, 0))

    fig.suptitle(None) # No title for any output
    plt.tight_layout(rect=[0, 0.08, 1, 0.92]) # Adjust rect for no suptitle and legend, equal top/bottom margins
    
    png_output_path = output_path_base + ".png"
    pdf_output_path = output_path_base + ".pdf"

    # Save PDF without title
    plt.savefig(pdf_output_path, format='pdf')
    print(f"Saved 1D-style PDF plot (no title): {pdf_output_path}")

    # Save PNG also without title
    plt.savefig(png_output_path, dpi=800)
    print(f"Saved 1D-style PNG plot (no title): {png_output_path}")
    
    plt.close(fig)


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CLEAR paper simulations.")
    parser.add_argument('-d', '--dimension', type=int, default=0, # Default to 0, indicates random if flag set
                        help='Dimension of the input features (X). Ignored if --randomize_d is set.')
    parser.add_argument('--randomize_d', action='store_true',
                        help='Randomly sample d from {2, 3, 20, 50} for each simulation run, ignoring -d.')
    parser.add_argument('--noise_type', type=str, default='homo', choices=['homo', 'hetero1', 'hetero2'],
                        help="Type of noise variance function (only applies when d=1). 'homoscedastic' -> sigma=1, 'hetero1' -> sigma=1+|x|, 'hetero2' -> sigma=1+1/(1+x^2).")
    parser.add_argument('--num_simulations', type=int, default=100, help='Number of simulation runs.')
    parser.add_argument('--n_samples', type=int, default=5000, help='Total number of samples (train + calibration).')
    parser.add_argument('--coverage', type=float, default=0.90, help='Target coverage level (e.g., 0.90 for 90%).')
    parser.add_argument('--log_details', action='store_true', help='Enable detailed logging to file.')
    parser.add_argument('--use_external_pcs', action='store_true', help='Use external PCS_UQ for epistemic uncertainty.')
    parser.add_argument('--n_boot_clear', type=int, default=50, help="Number of bootstraps for CLEAR's internal components.")
    parser.add_argument('--n_boot_pcs', type=int, default=50, help='Number of bootstraps for external PCS_UQ.')
    parser.add_argument('--cqr_center', type=str, default="pcs", choices=["pcs", "aleatoric"], help="Median source for centering CQR/Naive intervals ('pcs' or 'aleatoric').")
    parser.add_argument('--no_residuals', action='store_true', help='Fit aleatoric models directly on Y instead of residuals.')

    args = parser.parse_args()

    # Validate arguments
    dims_to_randomize = [2, 3, 10, 20] # Define the set for randomization
    fixed_d = args.dimension
    run_random_d_mode = args.randomize_d

    if run_random_d_mode:
        if fixed_d != 0:
            print(f"Warning: --randomize_d is set. Ignoring provided -d {fixed_d}.")
        fixed_d = None # Signal to run_simulation that d is randomized
        print(f"Running in randomized dimension mode. Sampling d from {dims_to_randomize}.")
        # Noise type will be forced to homo inside the loop for d>1
        if args.noise_type != 'homo':
             print(f"Warning: --randomize_d is set. Noise type '{args.noise_type}' will be ignored for d>1 runs (homoscedastic used).")
    elif fixed_d is None or fixed_d < 1:
         parser.error("Argument -d/--dimension is required and must be >= 1 if --randomize_d is not set.")
    elif fixed_d == 1 and args.noise_type not in ['homo', 'hetero1', 'hetero2']:
         parser.error(f"Invalid noise_type '{args.noise_type}' for d=1.")
    else:
         print(f"Running in fixed dimension mode with d={fixed_d}.")


    # Set random seed for overall experiment setup
    np.random.seed(42)

    # Determine project root path
    script_base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_path = os.path.abspath(os.path.join(script_base_dir, '..', '..'))

    print(f"Project root determined as: {project_root_path}")
    print(f"Running simulation with options:")
    if run_random_d_mode:
        print(f"  Dimension (d): Randomized in {dims_to_randomize}")
        print(f"  Noise Type: {args.noise_type} (Used only if d=1 is sampled, else homoscedastic)")
    else:
        print(f"  Dimension (d): {fixed_d}")
        print(f"  Noise Type: {args.noise_type if fixed_d == 1 else 'homoscedastic (d>1)'}")
    print(f"  Number of simulations: {args.num_simulations}")
    print(f"  Samples (Train+Calib): {args.n_samples}")
    print(f"  Coverage Level: {args.coverage}")
    print(f"  Log details: {args.log_details}")
    print(f"  Use external PCS: {args.use_external_pcs}")
    print(f"  CLEAR bootstraps: {args.n_boot_clear}")
    if args.use_external_pcs: print(f"  PCS_UQ bootstraps: {args.n_boot_pcs}")
    print(f"  CQR/Naive Center Source: {args.cqr_center}")
    print(f"  Fit aleatoric on residuals: {not args.no_residuals}")
    print(f"Outputs will be relative to project root: {project_root_path}")


    # Run the simulation
    run_simulation(
        d_fixed=fixed_d, # Pass the fixed dimension (or None if randomizing)
        randomize_d_flag=run_random_d_mode,
        dims_to_randomize=dims_to_randomize if run_random_d_mode else None,
        noise_type=args.noise_type, # Pass the user choice (will be handled internally for d>1)
        num_simulations=args.num_simulations,
        n_train_calib=args.n_samples,
        coverage_level=args.coverage,
        base_path=project_root_path,
        log_details=args.log_details,
        use_external_pcs=args.use_external_pcs,
        n_boot_clear=args.n_boot_clear,
        n_boot_pcs=args.n_boot_pcs,
        cqr_center_source=args.cqr_center,
        fit_on_residuals=not args.no_residuals
    )

    print("\nSimulation completed successfully!")

    # Add example run commands
    print("\n##########################################################")
    print("## Example Commands:")
    print("# Fixed d=5, 5 simulations:")
    print("# > python benchmark_simulations.py -d 5 --num_simulations 5")
    print("# Randomized d in {2,3,20,50}, 20 simulations:")
    print("# > python benchmark_simulations.py --randomize_d --num_simulations 20 --noise_type homo --use_external_pcs")
    print("##########################################################")

##########################################################
## To run the simulation exactly as done in the paper, use the following command:
# Homoscedastic noise with sigma=1, d=1, 100 simulations
# python .\benchmark_simulations.py --d 1 --num_simulations 100 --noise_type homo --use_external_pcs

# Heteroscedastic noise with sigma=1+|x|, d=1, 100 simulations
# python .\benchmark_simulations.py --d 1 --num_simulations 100 --noise_type hetero1 --use_external_pcs

# Heteroscedastic noise with sigma=1+1/(1+x^2), d=1, 100 simulations
# python .\benchmark_simulations.py --d 1 --num_simulations 100 --noise_type hetero2 --use_external_pcs

# Multivariate: Homoscedastic noise with sigma=1, d=1, 100 simulations, no external PCS
# python .\benchmark_simulations.py --randomize_d --num_simulations 100 --noise_type homo --use_external_pcs

##########################################################
# One liner command to run all simulations
# python .\benchmark_simulations.py --d 1 --num_simulations 100 --noise_type homo --use_external_pcs ; python .\benchmark_simulations.py --d 1 --num_simulations 100 --noise_type hetero1 --use_external_pcs ; python .\benchmark_simulations.py --d 1 --num_simulations 100 --noise_type hetero2 --use_external_pcs ; python .\benchmark_simulations.py --randomize_d --num_simulations 100 --noise_type homo --use_external_pcs
##########################################################