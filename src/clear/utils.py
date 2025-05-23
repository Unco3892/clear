"""
Utility functions for the CLEAR framework.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

# --- Start of conditional sys.path modification ---
if __name__ == '__main__' and __package__ is None:
    import sys
    import os
    # Get the directory of the current script (e.g., /path/to/src/clear)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Get the parent directory (e.g., /path/to/src)
    parent_dir = os.path.dirname(current_dir)
    # Add the parent directory to sys.path if it's not already there
    # This allows imports like 'from clear.metrics import ...'
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
# --- End of conditional sys.path modification ---
from clear.metrics import compute_auc

def load_ensemble_pickle(pkl_path):
    """Load the ensemble pickle file containing pre-trained ensembles."""
    # Fallback to standard pickle if custom loader fails
    with open(pkl_path, "rb") as f:
        ensemble_dict = pickle.load(f)
    return ensemble_dict

def compute_coverage_by_distance(X, lower_bounds, upper_bounds, y_true, distances, n_points_per_distance):
    """
    Compute coverage and interval width by distance from origin, matching the R implementation.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        lower_bounds: Lower bounds of prediction intervals
        upper_bounds: Upper bounds of prediction intervals
        y_true: True target values
        distances: List of distance values used to generate the data
        n_points_per_distance: Number of points generated at each distance
        
    Returns:
        Dict containing distances, coverage, and interval width per distance
    """
    # Initialize arrays to store results
    coverage_by_distance = []
    width_by_distance = []
    
    # Calculate metrics for each distance
    for j, distance in enumerate(distances):
        # Get indices for points at this distance
        start_idx = j * n_points_per_distance
        end_idx = (j + 1) * n_points_per_distance
        
        # Calculate coverage (percentage of points where y is within bounds)
        in_bounds = ((y_true[start_idx:end_idx] >= lower_bounds[start_idx:end_idx]) & 
                      (y_true[start_idx:end_idx] <= upper_bounds[start_idx:end_idx]))
        coverage = np.mean(in_bounds)
        coverage_by_distance.append(coverage)
        
        # Calculate average interval width
        width = np.mean(upper_bounds[start_idx:end_idx] - lower_bounds[start_idx:end_idx])
        width_by_distance.append(width)
    
    return {
        'distances': distances,
        'coverage': np.array(coverage_by_distance),
        'width': np.array(width_by_distance)
    }


def plot_distance_metrics(metrics_dict, methods, colors, method_names=None, target_coverage=0.9):
    """
    Plot coverage and interval width by distance from origin, matching the R implementation.
    
    Args:
        metrics_dict: Dictionary with methods as keys and metrics dicts as values
        methods: List of method keys to plot
        colors: List of colors to use for each method
        method_names: Optional list of display names for each method
        target_coverage: Target coverage level (for horizontal line)
    """
    if method_names is None:
        method_names = methods
    
    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot coverage by distance
    for method, color, name in zip(methods, colors, method_names):
        metrics = metrics_dict[method]
        axes[0].plot(metrics['distances'], metrics['coverage'], color=color, label=name, linewidth=2)
    
    axes[0].axhline(y=target_coverage, color='black', linestyle='--', linewidth=1)
    axes[0].set_xlabel('Distance from Origin (Radius)')
    axes[0].set_ylabel('Conditional Coverage')
    axes[0].set_ylim([0.4, 1.05])
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Plot interval width by distance
    for method, color, name in zip(methods, colors, method_names):
        metrics = metrics_dict[method]
        axes[1].plot(metrics['distances'], metrics['width'], color=color, label=name, linewidth=2)
    
    axes[1].set_xlabel('Distance from Origin (Radius)')
    axes[1].set_ylabel('Average Interval Width')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    return fig, axes

def generate_distance_points(n_points_per_distance=100, distances=None, n_dims=2, random_state=None):
    """
    Generate multivariate points distributed on concentric circles/spheres of different distances from origin.
    
    Args:
        n_points_per_distance: Number of points to generate for each distance
        distances: List of distance values from origin
        n_dims: Number of dimensions
        random_state: Random seed for reproducibility
        
    Returns:
        Array of shape (n_points_per_distance * len(distances), n_dims)
    """
    if random_state is not None:
        np.random.seed(random_state)
        
    if distances is None:
        distances = np.linspace(0.1, 5.0, 20)
    
    all_points = []
    
    for distance in distances:
        points = []
        for _ in range(n_points_per_distance):
            # Generate a random point on the unit sphere
            if n_dims == 1:
                # In 1D, just use -1 or 1 with equal probability
                point = np.array([np.random.choice([-1, 1])])
            else:
                # In higher dimensions, generate random direction
                point = np.random.randn(n_dims)
                point = point / np.linalg.norm(point)  # Normalize to unit length
            
            # Scale by the desired distance
            point = point * distance
            points.append(point)
        
        all_points.extend(points)
    
    return np.array(all_points)

# Create a function to visualize prediction intervals
def plot_prediction_intervals(X_test, y_test, intervals_dict, dataset_name, run_key, coverage_target=0.9, sample_size=200, base_plot_dir="../../plots", display=False):
    """
    Plot prediction intervals using a grid of subplots - one for each method.

    Args:
        X_test (array-like): Test features.
        y_test (array-like): True target values.
        intervals_dict (dict): Dictionary mapping method names to a tuple 
            (lower bounds, upper bounds, metrics).
        dataset_name (str): Name of the dataset (used for saving plots).
        run_key (str): Run identifier (used for saving plots).
        coverage_target (float): Target coverage level (default is 0.9).
        sample_size (int): Number of examples to sample for visualization.
        base_plot_dir (str): Base directory for saving plots.
        display (bool): If True, do not close the figures and return them for interactive display.
                        Default is False.
    """
    # Create the full directory for saving plots.
    save_dir = os.path.join(base_plot_dir, dataset_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # Sort instances by target value
    # Ensure y_test is a NumPy array for consistent indexing
    y_test_np = np.asarray(y_test).flatten()
    sort_idx = np.argsort(y_test_np)
    y_sorted = y_test_np[sort_idx]
    instance_idx = np.arange(len(y_test_np))
    
    # Create stratified sample if needed
    sampled_indices = None
    if sample_size < len(y_test):
        num_strata = 10  # Number of bins to stratify by
        strata_size = len(y_test) // num_strata
        sampled_indices = []
        
        for i in range(num_strata):
            start_idx = i * strata_size
            end_idx = start_idx + strata_size if i < num_strata - 1 else len(y_test)
            stratum_size = min(sample_size // num_strata, end_idx - start_idx)
            if stratum_size > 0:
                stratum_indices = np.random.choice(
                    np.arange(start_idx, end_idx), 
                    size=stratum_size, 
                    replace=False
                )
                sampled_indices.extend(stratum_indices)
        
        sampled_indices = sorted(sampled_indices)
        instance_idx = instance_idx[sampled_indices]
        y_sorted = y_sorted[sampled_indices]
    
    # Set up the subplot grid
    num_methods = len(intervals_dict)
    fig, axes = plt.subplots(num_methods, 1, figsize=(12, 4*num_methods), sharex=True)
    if not hasattr(axes, '__iter__'):
        axes = [axes]
    
    # Expanded color palette to handle more methods
    colors = ['red', 'blue', 'purple', 'green', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta', 'yellow']
    
    # Generate additional colors if needed using color map
    if num_methods > len(colors):
        import matplotlib.cm as cm
        color_map = cm.get_cmap('tab20')
        colors = [color_map(i) for i in np.linspace(0, 1, num_methods)]
    
    # Use axes[0] when there's a single method to ensure ax is an Axes instance
    for i, (method, (lower, upper, metrics)) in enumerate(intervals_dict.items()):
        ax = axes[i] if num_methods > 1 else axes[0]
        lower_sorted = np.array(lower)[sort_idx]
        upper_sorted = np.array(upper)[sort_idx]
        if sampled_indices is not None:
            lower_sorted = lower_sorted[sampled_indices]
            upper_sorted = upper_sorted[sampled_indices]
        ax.scatter(instance_idx, y_sorted, s=5, color='black', alpha=0.5)
        ax.fill_between(
            instance_idx, lower_sorted, upper_sorted, 
            alpha=0.3, color=colors[i % len(colors)]
        )
        ax.set_title(f"{method} (PICP={metrics['PICP']:.3f}, NIW={metrics['NIW']:.3f})")
        ax.grid(alpha=0.3)
        ax.set_ylabel('Target Value')
    
    axes[-1].set_xlabel('Sorted Instance Index')
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{dataset_name}_{run_key}_intervals_grid.png")
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    if not display:
        plt.close(fig)
    
    # Also create the coverage vs width scatter plot
    fig2 = plt.figure(figsize=(8, 6))
    
    for i, (method, (_, _, metrics)) in enumerate(intervals_dict.items()):
        plt.scatter(
            metrics['NIW'], 
            metrics['PICP'], 
            s=100, 
            color=colors[i % len(colors)], 
            label=method
        )
    
    plt.axhline(y=coverage_target, color='black', linestyle='--', label=f'Target Coverage ({coverage_target:.2f})')
    plt.title(f'Coverage vs. Width: {dataset_name} - {run_key}')
    plt.xlabel('Normalized Mean Prediction Interval Width (NIW)')
    plt.ylabel('Prediction Interval Coverage Probability (PICP)')
    plt.legend(loc='best')
    plt.grid(alpha=0.3)
    
    save_path2 = os.path.join(save_dir, f"{dataset_name}_{run_key}_coverage_width_comparison.png")
    fig2.savefig(save_path2, dpi=300, bbox_inches='tight')
    if not display:
        plt.close(fig2)

    if display:
        return fig, fig2

def plot_prediction_intervals_examples(y_test, intervals_dict, dataset_name, run_key, num_examples=5, base_plot_dir="../../plots"):
    """
    Plot prediction intervals for a few carefully selected examples to highlight differences.

    Args:
        X_test (array-like): Test features.
        y_test (array-like): True target values.
        intervals_dict (dict): Dictionary mapping method names to a tuple 
            (lower bounds, upper bounds, other metrics).
        dataset_name (str): Name of the dataset (used for saving plots).
        run_key (str): Run identifier (used for saving plots).
        coverage_target (float): Target coverage probability.
        num_examples (int): Number of examples to plot.
        base_plot_dir (str): Base directory where the plots will be saved.
    """
    # Create the full directory for saving plots.
    save_dir = os.path.join(base_plot_dir, dataset_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # Sort instances by target value
    # Ensure y_test is a NumPy array for consistent indexing
    y_test_np = np.asarray(y_test).flatten()
    sort_idx = np.argsort(y_test_np)
    y_sorted = y_test_np[sort_idx]
    
    # Select examples strategically from different parts of the distribution
    percentiles = np.linspace(0, 100, num_examples+2)[1:-1]
    example_indices = [int(p/100 * len(y_test)) for p in percentiles]
    example_indices = [sort_idx[i] for i in example_indices]
    
    # Set up figure with multiple subplots
    fig, axes = plt.subplots(1, num_examples, figsize=(4*num_examples, 6), sharey=True)
    
    # Expanded color palette to handle more methods
    colors = ['red', 'blue', 'purple', 'green', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta', 'yellow']
    methods = list(intervals_dict.keys())
    
    # Generate additional colors if needed using color map
    if len(methods) > len(colors):
        import matplotlib.cm as cm
        color_map = cm.get_cmap('tab20')
        colors = [color_map(i) for i in np.linspace(0, 1, len(methods))]
    
    for i, idx in enumerate(example_indices):
        ax = axes[i]
        y_true_val = y_test[idx]
        ax.axhline(y=y_true_val, color='black', linestyle='-', linewidth=1.5, label='True Value')
        for j, method in enumerate(methods):
            lower, upper, _ = intervals_dict[method]
            ax.plot([j, j], [lower[idx], upper[idx]], color=colors[j % len(colors)], linewidth=3, alpha=0.7)
            ax.plot([j-0.1, j+0.1], [lower[idx], lower[idx]], color=colors[j % len(colors)], linewidth=3)
            ax.plot([j-0.1, j+0.1], [upper[idx], upper[idx]], color=colors[j % len(colors)], linewidth=3)
        ax.set_title(f'Example {i+1}')
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=45)
        if i == 0:
            ax.set_ylabel('Target Value')
    
    # Add a legend below the subplots
    handles = [plt.Line2D([0], [0], color=c, linewidth=3) for c in colors[:len(methods)]]
    handles.append(plt.Line2D([0], [0], color='black', linewidth=1.5))
    labels = methods + ['True Value']
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=len(labels))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    save_path = os.path.join(save_dir, f"{dataset_name}_{run_key}_interval_examples.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# New function to plot the AUC curve for each method.
def plot_auc_curves(X_test, y_test, intervals_dict, dataset_name, run_key, num_points=100, base_plot_dir="../../plots"):
    """
    Plot AUC curves for each method using the full integration mode.
    
    For each method, we call compute_auc from clear.metrics to obtain the
    NIW and PICP values over the full range of c (from 0 to c*), and plot
    the corresponding AUC curve.
    
    Args:
        X_test (array-like): Test features (not used here).
        y_test (array-like): True target values.
        intervals_dict (dict): Dictionary mapping method names to a tuple 
            (lower bounds, upper bounds, other metrics). 
        dataset_name (str): Name of the dataset (used for saving plots).
        run_key (str): Run identifier (used for saving plots).
        num_points (int): Number of grid points in compute_auc.
        base_plot_dir (str): Base directory where the plots will be saved.
        
    Saves:
        A plot image file containing the AUC curves.
    """
    plt.figure(figsize=(8, 6))
    
    # Expanded color palette to handle more methods
    colors = ['red', 'blue', 'purple', 'green', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta', 'yellow']
    
    # Generate additional colors if needed using color map
    num_methods = len(intervals_dict)
    if num_methods > len(colors):
        import matplotlib.cm as cm
        color_map = cm.get_cmap('tab20')
        colors = [color_map(i) for i in np.linspace(0, 1, num_methods)]
    
    for i, (method, (lower, upper, metrics)) in enumerate(intervals_dict.items()):
        lower = np.asarray(lower).flatten()
        upper = np.asarray(upper).flatten()
        y_true = np.asarray(y_test).flatten()
        
        try:
            # Use compute_auc to get the NIW and PICP curve values.
            auc_value, auc_info = compute_auc(y_true, lower, upper, num_points=num_points)
            nm_values = auc_info["NMPIWs"]
            picp_values = auc_info["PICPs"]
            plt.plot(nm_values, picp_values, color=colors[i % len(colors)], 
                     label=f"{method} (AUC: {auc_value:.3f})", linewidth=2)
        except Exception as e:
            print(f"Error computing AUC for {method}: {str(e)}")
    
    plt.axhline(y=0.90, color='black', linestyle='--', label='Target Coverage (0.90)')
    plt.xlabel('Normalized Mean Prediction Interval Width (NIW)')
    plt.ylabel('Prediction Interval Coverage Probability (PICP)')
    plt.title(f"AUC Curves for {dataset_name}, {run_key}")
    plt.legend(loc='best')
    plt.grid(alpha=0.3)
    
    # Build the full save directory using the provided base_plot_dir and dataset_name.
    save_dir = os.path.join(base_plot_dir, dataset_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{dataset_name}_{run_key}_auc_curve.png")
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def compute_quantile_ensemble_bounds(ensemble_preds, lower_quantile=0.05, upper_quantile=0.95, random_state=None, aleatoric_median=None, symmetric_noise=False):
    """
    Compute uncertainty bounds from ensemble predictions.
    
    Args:
        ensemble_preds: Ensemble predictions with shape (n_models, n_samples)
        lower_quantile: Lower quantile (default: 0.05)
        upper_quantile: Upper quantile (default: 0.95)
        random_state: Optional random seed for reproducibility
        aleatoric_median: Optional aleatoric median predictions to include in ensemble
        symmetric_noise: If True, include aleatoric_median in the ensemble predictions
    
    Returns:
        median: Median predictions
        lower: Lower bound predictions
        upper: Upper bound predictions
    """
    # MODIFIED: Handle the case where aleatoric_median is None
    
    # If symmetric_noise and aleatoric_median is provided, include the aleatoric median
    if symmetric_noise and aleatoric_median is not None:
        # Make sure aleatoric_median is a 1D array before reshaping
        aleatoric_median = np.asarray(aleatoric_median).flatten()
        
        # Ensure ensemble_preds is 2D
        if ensemble_preds.ndim == 1:
            ensemble_preds = ensemble_preds.reshape(1, -1)
            
        # Reshape aleatoric_median to match ensemble_preds dimensions
        # This ensures it's a single row with the same number of columns as ensemble_preds
        aleatoric_median_reshaped = aleatoric_median.reshape(1, -1)
        
        # Make sure the second dimension (columns) matches
        if aleatoric_median_reshaped.shape[1] != ensemble_preds.shape[1]:
            # If they don't match, it could mean our ensemble_preds is transposed
            # Try transposing if it makes the dimensions match
            if aleatoric_median_reshaped.shape[1] == ensemble_preds.shape[0]:
                ensemble_preds = ensemble_preds.T
            else:
                raise ValueError(f"Cannot combine ensemble_preds shape {ensemble_preds.shape} with aleatoric_median shape {aleatoric_median_reshaped.shape}")
        
        # Now stack them
        combined_preds = np.vstack([ensemble_preds, aleatoric_median_reshaped])
    else:
        combined_preds = ensemble_preds
        
    # Compute quantiles over the first axis (n_models)
    median = np.median(combined_preds, axis=0)
    lower = np.quantile(combined_preds, lower_quantile, axis=0, method='higher')
    upper = np.quantile(combined_preds, upper_quantile, axis=0, method='higher')
    
    # Add a sanity check to avoid overlapping quantiles using any method
    if np.any(upper < median) or np.any(lower > median):
        # Ensure non-crossing quantiles
        print("Warning: Crossing quantiles detected. Recalibrating...") 
        upper = np.maximum(upper, median)
        lower = np.minimum(lower, median)
    
    return median, lower, upper

def calibrate_ensemble_bounds(y_calib, ensemble_preds_calib, ensemble_preds_test, desired_coverage=0.9, random_state=None, aleatoric_median_calib=None, aleatoric_median_test=None, symmetric_noise=False):
    """
    Apply conformal calibration to raw ensemble prediction intervals.
    
    Args:
        y_calib: Calibration targets
        ensemble_preds_calib: Ensemble predictions for calibration data (n_models, n_calib_samples)
        ensemble_preds_test: Ensemble predictions for test data (n_models, n_test_samples)
        desired_coverage: Target coverage level (default: 0.9)
        random_state: Optional random seed for reproducibility
        aleatoric_median_calib: Optional aleatoric median predictions for calibration data
        aleatoric_median_test: Optional aleatoric median predictions for test data
        symmetric_noise: If True, include aleatoric median in ensemble (default: True)
    
    Returns:
        lower_bounds: Calibrated lower bounds for test data
        upper_bounds: Calibrated upper bounds for test data
    """
    # Set random seed if provided
    if random_state is not None:
        np.random.seed(random_state)

    alpha = 1 - desired_coverage
    lower_quantile = alpha / 2
    upper_quantile = 1 - alpha / 2

    # MODIFIED: Handle the pure PCS case properly when aleatoric_median is None
    if aleatoric_median_calib is None and aleatoric_median_test is None:
        # Pure PCS approach without aleatoric component
        pcs_median_calib, pcs_lower_calib, pcs_upper_calib = compute_quantile_ensemble_bounds(
            ensemble_preds_calib, 
            lower_quantile=alpha/2, 
            upper_quantile=1-alpha/2,
            aleatoric_median=None,
            symmetric_noise=symmetric_noise
        )
        
        pcs_median_test, pcs_lower_test, pcs_upper_test = compute_quantile_ensemble_bounds(
            ensemble_preds_test, 
            lower_quantile=alpha/2, 
            upper_quantile=1-alpha/2,
            aleatoric_median=None,
            symmetric_noise=symmetric_noise
        )
    else:
        # Original code with aleatoric median
        pcs_median_calib, pcs_lower_calib, pcs_upper_calib = compute_quantile_ensemble_bounds(
            ensemble_preds_calib, 
            lower_quantile=alpha/2, 
            upper_quantile=1-alpha/2,
            aleatoric_median=aleatoric_median_calib,
            symmetric_noise=symmetric_noise
        )
        
        pcs_median_test, pcs_lower_test, pcs_upper_test = compute_quantile_ensemble_bounds(
            ensemble_preds_test, 
            lower_quantile=alpha/2, 
            upper_quantile=1-alpha/2,
            aleatoric_median=aleatoric_median_test,
            symmetric_noise=symmetric_noise
        )
    
    # Conformalize the intervals
    n_calib = len(y_calib)
    scores = np.maximum(
        (pcs_lower_calib - y_calib) / (pcs_lower_calib - pcs_median_calib + 1e-8),
        (y_calib - pcs_upper_calib) / (pcs_median_calib - pcs_upper_calib + 1e-8)
    )
    
    # Get adjustment factor as the quantile of scores
    q_level = np.ceil((n_calib + 1) * (1 - alpha)) / n_calib
    adjustment = np.quantile(scores, q_level)
    
    # Apply calibration
    lower_bounds = pcs_median_test - adjustment * (pcs_median_test - pcs_lower_test)
    upper_bounds = pcs_median_test + adjustment * (pcs_upper_test - pcs_median_test)
    
    return lower_bounds, upper_bounds

if __name__ == '__main__':
    print("=== CLEAR Utils Module Demonstration ===")

    # --- Demonstrate compute_quantile_ensemble_bounds ---
    print("\n--- Demonstrating compute_quantile_ensemble_bounds ---")
    np.random.seed(0)
    n_models = 10
    n_samples = 50
    ensemble_preds_demo = np.random.rand(n_models, n_samples) * 10 # Example ensemble predictions
    aleatoric_median_demo = np.median(ensemble_preds_demo, axis=0) + np.random.normal(0, 0.5, n_samples) # Example aleatoric median

    print(f"Shape of ensemble_preds_demo: {ensemble_preds_demo.shape}")
    print(f"Shape of aleatoric_median_demo: {aleatoric_median_demo.shape}")

    # Scenario 1: Standard quantiles from ensemble
    median_1, lower_1, upper_1 = compute_quantile_ensemble_bounds(ensemble_preds_demo, lower_quantile=0.1, upper_quantile=0.9)
    print("\nScenario 1: Standard quantiles")
    print(f"  Median shape: {median_1.shape}, Lower shape: {lower_1.shape}, Upper shape: {upper_1.shape}")
    print(f"  Example median_1[0]: {median_1[0]:.2f}, lower_1[0]: {lower_1[0]:.2f}, upper_1[0]: {upper_1[0]:.2f}")

    # Scenario 2: Including aleatoric median with symmetric_noise=True
    median_2, lower_2, upper_2 = compute_quantile_ensemble_bounds(
        ensemble_preds_demo, 
        lower_quantile=0.1, 
        upper_quantile=0.9,
        aleatoric_median=aleatoric_median_demo,
        symmetric_noise=True
    )
    print("\nScenario 2: Including aleatoric median (symmetric_noise=True)")
    print(f"  Median shape: {median_2.shape}, Lower shape: {lower_2.shape}, Upper shape: {upper_2.shape}")
    print(f"  Example median_2[0]: {median_2[0]:.2f}, lower_2[0]: {lower_2[0]:.2f}, upper_2[0]: {upper_2[0]:.2f}")

    # --- Demonstrate generate_distance_points ---
    print("\n\n--- Demonstrating generate_distance_points ---")
    distances_demo = np.array([0.5, 1.0, 1.5])
    n_points_per_distance_demo = 3
    n_dims_demo = 2

    points_2d = generate_distance_points(n_points_per_distance_demo, distances_demo, n_dims_demo, random_state=42)
    print(f"\nGenerated points for d={n_dims_demo}:")
    print(f"  Shape: {points_2d.shape}") # Expected: (n_distances * n_points_per_distance, n_dims)
    # Verify distances for the first few points of each distance group
    for i, dist in enumerate(distances_demo):
        start_idx = i * n_points_per_distance_demo
        end_idx = start_idx + n_points_per_distance_demo
        points_at_dist = points_2d[start_idx:end_idx]
        norms = np.linalg.norm(points_at_dist, axis=1)
        print(f"  Distance {dist:.1f}: Norms of generated points: {np.round(norms, 2)}")

    n_dims_demo_3d = 3
    points_3d = generate_distance_points(n_points_per_distance_demo, distances_demo, n_dims_demo_3d, random_state=42)
    print(f"\nGenerated points for d={n_dims_demo_3d}:")
    print(f"  Shape: {points_3d.shape}")
    for i, dist in enumerate(distances_demo):
        start_idx = i * n_points_per_distance_demo
        end_idx = start_idx + n_points_per_distance_demo
        points_at_dist = points_3d[start_idx:end_idx]
        norms = np.linalg.norm(points_at_dist, axis=1)
        print(f"  Distance {dist:.1f}: Norms of generated points: {np.round(norms, 2)}")

    # --- Demonstrate compute_coverage_by_distance (example setup) ---
    print("\n\n--- Demonstrating compute_coverage_by_distance (setup) ---")
    # Using the points_2d generated earlier as X_test_flat
    X_test_flat_demo = points_2d
    y_test_demo = np.sum(X_test_flat_demo, axis=1) + np.random.normal(0, 0.1, X_test_flat_demo.shape[0]) # Dummy y_test
    
    # Dummy bounds for demonstration
    lower_bounds_demo = y_test_demo - 0.5 
    upper_bounds_demo = y_test_demo + 0.5
    
    # Ensure all points are covered for a simple check, then reduce for variation
    # For a more realistic demo, one might create bounds that don't always cover y_test_demo.
    # Make some bounds intentionally miss for varied coverage
    miss_indices = np.random.choice(len(y_test_demo), size=len(y_test_demo)//3, replace=False)
    lower_bounds_demo[miss_indices] = y_test_demo[miss_indices] + 0.1 # Force lower bound above y_test
    
    print(f"X_test_flat_demo shape: {X_test_flat_demo.shape}")
    print(f"y_test_demo shape: {y_test_demo.shape}")
    print(f"lower_bounds_demo shape: {lower_bounds_demo.shape}")
    print(f"upper_bounds_demo shape: {upper_bounds_demo.shape}")
    print(f"Distances used for grouping: {distances_demo}")
    print(f"Points per distance: {n_points_per_distance_demo}")

    coverage_metrics = compute_coverage_by_distance(
        X_test_flat_demo, 
        lower_bounds_demo, 
        upper_bounds_demo, 
        y_test_demo, 
        distances_demo, 
        n_points_per_distance_demo
    )
    print("\nCoverage by distance results:")
    print(f"  Distance centers: {np.round(coverage_metrics['distances'], 2)}")
    print(f"  Conditional coverage: {np.round(coverage_metrics['coverage'], 2)}")
    print(f"  Conditional width: {np.round(coverage_metrics['width'], 2)}")

    # Example of plotting (requires matplotlib, which might not be available in all test environments)
    import matplotlib.pyplot as plt
    fig, axes = plot_distance_metrics(
        {'DemoMethod': coverage_metrics},
        methods=['DemoMethod'],
        colors=['blue'],
        # labels=['Demo Method'],
        target_coverage=0.9 # Example target
    )
    plt.suptitle("Coverage and Width by Distance (Utils Demo)")
    plt.tight_layout(rect=[0,0,1,0.96])
    plt.show()

    print("\nUtils module demonstration finished.")
