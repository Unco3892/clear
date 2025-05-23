"""
Metrics for evaluating uncertainty quantification methods in the CLEAR framework.
These metrics assess both calibration and accuracy of prediction intervals.
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings

def picp(y_true, lower, upper):
    """
    Prediction Interval Coverage Probability (PICP).
    Computes the proportion of true values that fall within the [lower, upper] prediction interval.
    
    Parameters:
    y_true (array-like): Ground truth values.
    lower (array-like): Lower bounds of the prediction intervals.
    upper (array-like): Upper bounds of the prediction intervals.

    Returns:
    float: Coverage probability.
    """
    y_true = np.array(y_true)
    lower = np.array(lower)
    upper = np.array(upper)
    coverage = np.mean((y_true >= lower) & (y_true <= upper))
    return coverage

def interval_score_loss(upper, lower, y_true, alpha):
    """
    Interval Score Loss (ISL).
    Computes a score that penalizes both the width of the interval and violations 
    of the interval bounds, weighted by alpha.
    Source: https://github.com/rrross/UACQR/blob/main/helper.py
    
    Parameters:
    upper (array-like): Upper bounds of the prediction intervals.
    lower (array-like): Lower bounds of the prediction intervals.
    y_true (array-like): Ground truth values.
    alpha (float): Significance level (e.g., 0.05 for 95% intervals).
    
    Returns:
    array-like: Element-wise interval score loss.
    """
    y_true = np.array(y_true)
    upper = np.array(upper)
    lower = np.array(lower)
    
    return (upper - lower) + \
           (2/alpha) * (lower - y_true) * (y_true < lower) + \
           (2/alpha) * (y_true - upper) * (y_true > upper)

def average_interval_score_loss(upper, lower, y_true, alpha):
    """
    Average Interval Score Loss (AISL).
    Computes the mean interval score loss across all samples.
    Source: https://github.com/rrross/UACQR/blob/main/helper.py
    
    Parameters:
    upper (array-like): Upper bounds of the prediction intervals.
    lower (array-like): Lower bounds of the prediction intervals.
    y_true (array-like): Ground truth values.
    alpha (float): Significance level (e.g., 0.05 for 95% intervals).
    
    Returns:
    float: Mean interval score loss.
    """
    return np.mean(interval_score_loss(upper, lower, y_true, alpha))

def mpiw(lower, upper):
    """
    Mean Prediction Interval Width (MPIW).
    Computes the average width of the prediction intervals.
    
    Parameters:
    lower (array-like): Lower bounds of the prediction intervals.
    upper (array-like): Upper bounds of the prediction intervals.

    Returns:
    float: Mean width.
    """
    lower = np.array(lower)
    upper = np.array(upper)
    interval_width = upper - lower
    mean_width = np.mean(interval_width)
    return mean_width

def niw(y_true, lower, upper):
    """
    Normalized Mean Prediction Interval Width (NIW).
    Computes the average width of the prediction intervals normalized by the range of y_true.
    
    Parameters:
    y_true (array-like): Ground truth values.
    lower (array-like): Lower bounds of the prediction intervals.
    upper (array-like): Upper bounds of the prediction intervals.

    Returns:
    float: Normalized mean width.
    """
    mean_width = mpiw(lower, upper)
    data_range = np.max(y_true) - np.min(y_true)
    if data_range == 0:
        return mean_width
    return mean_width / data_range

def quantile_loss(y_true, q, tau):
    """
    Computes the quantile loss for predictions at a given quantile level tau.
    The quantile loss is defined as:

        L_tau(y, q) = (y - q)(tau - 𝟙{y ≤ q})

    Parameters:
    y_true (array-like): Ground truth values.
    q (array-like): Predicted quantile values.
    tau (float): Quantile level (between 0 and 1).

    Returns:
    float: Average quantile loss.
    """
    y_true = np.array(y_true)
    q = np.array(q)
    # If y >= q: loss = tau*(y - q); else loss = (1-tau)*(q - y)
    loss = np.where(y_true >= q, tau * (y_true - q), (1 - tau) * (q - y_true))
    
    # if np.max(np.abs(loss)) < 0.001:
    #     print(f"Info: Quantile loss values are less than 0.001. May impact readability of results.")

    return np.mean(loss)

def expectile_loss(y_true, q, tau):
    """
    Computes the expectile loss for predictions at a given expectile level tau.
    The expectile loss for a single observation y and prediction q is:
        L_tau(y, q) = (1-tau)*(y-q)^2 if y <= q
                      tau*(y-q)^2   if y > q

    This definition is based on the asymmetric L2 loss minimized by expectile regression,
    as described at https://en.m.wikipedia.org/wiki/Expectile.

    Parameters:
    y_true (array-like): Ground truth values.
    q (array-like): Predicted expectile values.
    tau (float): Expectile level (between 0 and 1, exclusive).

    Returns:
    float: Average expectile loss.
    """
    y_true = np.array(y_true)
    q = np.array(q)
    
    if not (0 < tau < 1):
        raise ValueError("Expectile level tau must be between 0 and 1 (exclusive).")
        
    squared_errors = (y_true - q)**2
    # Loss is (1-tau)*(y-q)^2 if y <= q, else tau*(y-q)^2
    loss = np.where(y_true <= q, (1 - tau) * squared_errors, tau * squared_errors)
    return np.mean(loss)

def compute_average_expectile_loss(lower, upper, y_true, alpha=0.1):
    """
    Computes the average expectile loss for lower and upper expectile predictions.

    This function calculates the average of two expectile losses:
    1. Expectile loss for `lower` predictions at `tau = alpha / 2`.
    2. Expectile loss for `upper` predictions at `tau = 1 - alpha / 2`.

    This is analogous to `compute_quantile_loss` but for expectiles.

    Parameters:
    lower (array-like): Lower expectile predictions (e.g., for tau = alpha/2).
    upper (array-like): Upper expectile predictions (e.g., for tau = 1 - alpha/2).
    y_true (array-like): Ground truth values.
    alpha (float): Significance level, used to determine expectile levels.
                   tau_lower = alpha / 2, tau_upper = 1 - alpha / 2.
                   Default is 0.1. Ensures 0 < tau_lower, tau_upper < 1.

    Returns:
    float: Average of the two expectile losses.
    """
    tau_lower = alpha / 2
    tau_upper = 1 - alpha / 2
    
    # Alpha is typically the miscoverage rate, so 0 < alpha < 1.
    # This check ensures tau_lower and tau_upper will be in (0,1).
    if not (0 < alpha < 2): 
        raise ValueError("alpha must be in (0, 2) to ensure valid expectile levels tau_lower and tau_upper in (0,1). Typically, alpha is in (0,1).")

    lower_loss = expectile_loss(y_true, lower, tau_lower)
    upper_loss = expectile_loss(y_true, upper, tau_upper)
    return (lower_loss + upper_loss) / 2

def compute_quantile_loss(lower, upper, y_true, alpha=0.1):
    """
    Average quantile loss at the lower and upper quantiles (for alpha-level intervals).
    
    Parameters:
    lower (array-like): Lower bounds of prediction intervals.
    upper (array-like): Upper bounds of prediction intervals. 
    y_true (array-like): Ground truth values.
    alpha (float): Target miscoverage level, default is 0.1 (for 90% coverage).
    
    Returns:
    float: Average of lower and upper quantile losses.
    """
    tau_lower = alpha / 2
    tau_upper = 1 - alpha / 2
    lower_loss = quantile_loss(y_true, lower, tau_lower)
    upper_loss = quantile_loss(y_true, upper, tau_upper)
    return (lower_loss + upper_loss) / 2

def crps_interval(y_true, lower, upper):
    """
    Computes the Continuous Ranked Probability Score (CRPS) assuming a uniform predictive
    distribution over the interval [lower, upper].
    
    For standard uniform distribution on [a,b], analytical formula is used:
    - If y < a: CRPS = (a - y) + (b - a) / 3
    - If y > b: CRPS = (y - b) + (b - a) / 3
    - If a ≤ y ≤ b: CRPS = ((y - a)^3 + (b - y)^3) / (3 * (b - a)^2)
    
    Parameters:
    y_true (array-like): Ground truth values.
    lower (array-like): Lower bounds of prediction intervals.
    upper (array-like): Upper bounds of prediction intervals.
    
    Returns:
    float: Mean CRPS over all samples.
    """
    y_true = np.asarray(y_true).flatten()
    lower = np.asarray(lower).flatten()
    upper = np.asarray(upper).flatten()
    
    # Ensure all arrays have the same length
    n = len(y_true)
    if len(lower) != n or len(upper) != n:
        raise ValueError(f"All input arrays must have the same length. Got y_true: {len(y_true)}, lower: {len(lower)}, upper: {len(upper)}")
    
    # Small epsilon to avoid division by zero
    epsilon = 1e-10
    
    # Initialize array for CRPS values
    crps_values = np.zeros(n, dtype=float)
    
    # Calculate CRPS for each prediction
    for i in range(n):
        y = y_true[i]
        a = lower[i]
        b = upper[i]
        
        # Handle degenerate case (lower = upper or lower > upper)
        if np.abs(b - a) < epsilon:
            crps_values[i] = np.abs(y - a)
            continue
            
        # Ensure interval width is always positive
        interval_width = max(b - a, epsilon)
        
        # Calculate CRPS based on where y falls relative to the interval
        if y < a:
            crps_values[i] = (a - y) + interval_width / 3.0
        elif y > b:
            crps_values[i] = (y - b) + interval_width / 3.0
        else:  # a ≤ y ≤ b
            crps_values[i] = ((y - a)**3 + (b - y)**3) / (3.0 * interval_width**2)

    # if np.max(np.abs(crps_values)) < 0.001:
    #     print(f"Info: CRPS values are less than 0.001. May impact readability of results.")

    # Return the mean CRPS value
    return np.mean(crps_values)

def compute_auc(y_true, lower, upper, f=None):
    """
    Compute the Area under the AUC-like curve defined by varying the scaling factor c 
    for the prediction intervals [hat{f} - c*l, hat{f} + c*u], where 
    hat{f} is the midpoint of (lower, upper) and l = hat{f} - lower, u = upper - hat{f}.
    
    This implementation computes AUC by integrating over the full range of the scaling factor c, 
    from c = 0 to c = c_star, where c_star is defined as:
    
        c* := argmin_{c >= 0} { PICP(hat{f} - c*l, hat{f} + c*u) = 1 }.
    
    The AUC is then obtained by approximating the integral of the curve 
    { (NIW(hat{f} - c*l, hat{f} + c*u), PICP(hat{f} - c*l, hat{f} + c*u)) : c in [0, c*] }
    using the trapezoidal rule.

    An optional parameter ``f`` can be provided to represent the central estimate (which need not 
    be symmetric). If ``f`` is not provided, it defaults to the symmetric midpoint computed as 
    (lower+upper)/2.
    
    Args:
        y_true (array-like): True target values.
        lower (array-like): Original lower bounds.
        upper (array-like): Original upper bounds.
        f (array-like, optional): Central estimate for each sample. If None, it is computed as (lower+upper)/2.
    
    Returns:
        float: AUC value (area under the PICP versus NIW curve).
        dict: Additional information, including the sorted candidate scaling factors, PICPs, and NMPIWs.
    """
    y_true = np.asarray(y_true).flatten()
    lower = np.asarray(lower).flatten()
    upper = np.asarray(upper).flatten()
    
    # Use provided median f if available; otherwise compute as symmetric midpoint.
    if f is None:
        f = (lower + upper) / 2.0
    
    # Base deviations.
    l_base = f - lower
    u_base = upper - f

    # Compute candidate scaling factors for each sample.
    c_candidates = []
    for i in range(len(y_true)):
        if y_true[i] < f[i]:
            c_val = (f[i] - y_true[i]) / l_base[i] if l_base[i] > 0 else 0
        elif y_true[i] > f[i]:
            c_val = (y_true[i] - f[i]) / u_base[i] if u_base[i] > 0 else 0
        else:
            c_val = 0
        c_candidates.append(c_val)
    
    # Use sorted candidate values instead of a uniform grid.
    sorted_c = np.sort(c_candidates)
    n_points = len(sorted_c)
    # PICP at a given candidate is simply the fraction of data points covered.
    PICPs = np.arange(1, n_points + 1) / n_points
    # Note: new interval widths scale linearly, so NIW(new_lower,new_upper)= c * NIW(y_true, lower, upper).
    base_nmpiw = niw(y_true, lower, upper)
    NMPIWs = sorted_c * base_nmpiw

    # Integrate using the trapezoidal rule.
    auc_value = np.trapz(PICPs, NMPIWs)
    extra_info = {"sorted_c": sorted_c, "PICPs": PICPs, "NMPIWs": NMPIWs}
    # print(f"AUC computed using {n_points} candidate scaling factors ranging from {sorted_c[0]:.2f} to {sorted_c[-1]:.2f}.")
    return auc_value, extra_info


def compute_nciw(y_true, lower, upper, alpha=0.1, f=None):
    """
    Compute the Normalized Mean Test-Calibrated Prediction Interval Width (NCIW) 
    with improved efficiency.
    
    NCIW is defined as NIW evaluated on intervals that are calibrated to achieve a 
    target coverage of 1 - alpha. In other words, we search for c_test_cal defined by:
      
        c_test_cal := inf{ c >= 0 : PICP(f̂ - c*l, f̂ + c*u) >= (1 - alpha) },
      
    and then compute:
      
        NCIW = NIW(f̂ - c_test_cal*l, f̂ + c_test_cal*u).
        
    The prediction intervals are assumed to scale linearly with a scaling factor c, allowing the 
    NCIW to be computed efficiently.
    
    An optional parameter ``f`` can be provided to specify the central estimate (which may be 
    asymmetric). If ``f`` is not provided, it is computed as (lower+upper)/2.
    
    Args:
        y_true (array-like): True target values.
        lower (array-like): Original lower prediction bounds.
        upper (array-like): Original upper prediction bounds.
        alpha (float): Miscoverage level (default 0.1 for 90% coverage).
        f (array-like, optional): Central estimate for each sample. If None, computed as (lower+upper)/2.
    
    Returns:
        float: NCIW value.
        dict: Additional information, including c_test_cal, the sorted candidate scaling factors, and the corresponding PICPs.
    """
    y_true = np.asarray(y_true).flatten()
    lower = np.asarray(lower).flatten()
    upper = np.asarray(upper).flatten()
    
    if f is None:
        f = (lower + upper) / 2.0
    
    l_base = f - lower
    u_base = upper - f
    
    # Compute candidate scaling factors for each data point.
    c_candidates = []
    for i in range(len(y_true)):
        if y_true[i] < f[i]:
            c_val = (f[i] - y_true[i]) / l_base[i] if l_base[i] > 0 else 0
        elif y_true[i] > f[i]:
            c_val = (y_true[i] - f[i]) / u_base[i] if u_base[i] > 0 else 0
        else:
            c_val = 0
        c_candidates.append(c_val)
    
    sorted_c = np.sort(c_candidates)
    n = len(sorted_c)
    # PICP for a candidate c at index i is (i+1)/n.
    PICPs = np.arange(1, n + 1) / n
    target = 1 - alpha
    # Find the smallest candidate for which PICP >= target.
    idx = np.searchsorted(PICPs, target)
    if idx < n:
        c_test_cal = sorted_c[idx]
    else:
        c_test_cal = sorted_c[-1]
    
    # New prediction interval widths scale linearly.
    base_nmpiw = niw(y_true, lower, upper)
    nciw_value = c_test_cal * base_nmpiw
    extra_info = {"c_test_cal": c_test_cal, "sorted_c": sorted_c, "PICPs": PICPs}
    return nciw_value, extra_info

# Add a function to plot the AUC-like curve of PICP vs NIW by varying scaling factor c
# Plot PICP on the x-axis and NIW on the y-axis

def plot_auc_curve(y_true, lower, upper, f=None, ax=None, label=None, **plot_kwargs):
    """
    Plot the PICP vs NIW curve for varying scaling factors c using compute_auc.

    Parameters:
        y_true (array-like): Ground truth values.
        lower (array-like): Lower bounds of the prediction intervals.
        upper (array-like): Upper bounds of the prediction intervals.
        f (array-like, optional): Central estimates; defaults to midpoint of [lower, upper].
        ax (matplotlib.axes.Axes, optional): Axis to plot on. If None, a new figure and axis are created.
        label (str, optional): Label for the curve.
        **plot_kwargs: Additional keyword arguments passed to ax.plot().

    Returns:
        matplotlib.axes.Axes: The axis with the plotted curve.
    """
    # Compute curve points using compute_auc (which returns sorted_c, PICPs, NMPIWs)
    _, extra_info = compute_auc(y_true, lower, upper, f=f)
    PICPs = extra_info["PICPs"]
    NMPIWs = extra_info["NMPIWs"]
    # Use current axis if not provided, so multiple calls overlay on the same plot
    if ax is None:
        ax = plt.gca()
    # Plot PICP vs NIW on the existing axis
    ax.plot(PICPs, NMPIWs, label=label, **plot_kwargs)
    # Label axes if this is the first call
    ax.set_xlabel("PICP")
    ax.set_ylabel("NIW")
    return ax

def evaluate_intervals(y_true, lower, upper, alpha=0.1, f=None):
    """
    Evaluates prediction intervals using multiple metrics.
    
    Parameters:
    y_true (array-like): Ground truth values.
    lower (array-like): Lower bounds of prediction intervals.
    upper (array-like): Upper bounds of prediction intervals.
    alpha (float): Miscoverage level (default: 0.1 for 90% coverage).
    f (array-like, optional): Central estimates provided by the model. If not provided, these 
        are computed as (lower+upper)/2.
    
    Returns:
    dict: A dictionary of calculated metrics including PICP, NIW, Quantile Loss, Expectile Loss, CRPS, AUC, and NCIW.
    """
    # Ensure input arrays are numpy arrays and flattened.
    y_true = np.asarray(y_true).flatten()
    lower = np.asarray(lower).flatten()
    upper = np.asarray(upper).flatten()
    
    lengths = [len(y_true), len(lower), len(upper)]
    if len(set(lengths)) > 1:
        min_length = min(lengths)
        warnings.warn(f"Input arrays have different lengths: {lengths}. Using first {min_length} elements for all arrays.", UserWarning)
        y_true = y_true[:min_length]
        lower = lower[:min_length]
        upper = upper[:min_length]
    
    metrics = {}
    try:
        metrics["PICP"] = picp(y_true, lower, upper)
    except Exception as e:
        print(f"Error calculating PICP: {str(e)}")
        metrics["PICP"] = np.nan
        
    try:
        metrics["NIW"] = niw(y_true, lower, upper)
    except Exception as e:
        print(f"Error calculating NIW: {str(e)}")
        metrics["NIW"] = np.nan
        
    try:
        metrics["MPIW"] = mpiw(lower, upper)
    except Exception as e:
        print(f"Error calculating MPIW: {str(e)}")
        metrics["MPIW"] = np.nan

    try:
        metrics["QuantileLoss"] = compute_quantile_loss(lower, upper, y_true, alpha=alpha)
    except Exception as e:
        print(f"Error calculating QuantileLoss: {str(e)}")
        metrics["QuantileLoss"] = np.nan
        
    # Add Expectile Loss
    try:
        metrics["ExpectileLoss"] = compute_average_expectile_loss(lower, upper, y_true, alpha=alpha)
    except Exception as e:
        print(f"Error calculating ExpectileLoss: {str(e)}")
        metrics["ExpectileLoss"] = np.nan
        
    try:
        metrics["CRPS"] = crps_interval(y_true, lower, upper)
    except Exception as e:
        print(f"Error calculating CRPS: {str(e)}")
        metrics["CRPS"] = np.nan
    
    # Add Interval Score Loss
    try:
        metrics["IntervalScoreLoss"] = average_interval_score_loss(upper, lower, y_true, alpha)
    except Exception as e:
        print(f"Error calculating IntervalScoreLoss: {str(e)}")
        metrics["IntervalScoreLoss"] = np.nan
    
    # New metrics: AUC and NCIW.
    try:
        auc_value, _ = compute_auc(y_true, lower, upper, f=f)
        metrics["AUC"] = auc_value
    except Exception as e:
        print(f"Error calculating AUC: {str(e)}")
        metrics["AUC"] = np.nan
        
    try:
        nciw_value, nciw_info = compute_nciw(y_true, lower, upper, alpha=alpha, f=f)
        metrics["NCIW"] = nciw_value
        # For compatibility we keep c_test_cal if available.
        if "c_test_cal" in nciw_info:
            metrics["c_test_cal"] = nciw_info["c_test_cal"]
    except Exception as e:
        print(f"Error calculating NCIW: {str(e)}")
        metrics["NCIW"] = np.nan
        metrics["c_test_cal"] = np.nan
    
    return metrics


if __name__ == '__main__':
    print("=== CLEAR Metrics Module Demonstration ===")
    # Generate synthetic data with varying uncertainty for demonstration
    np.random.seed(42)
    n_samples = 100
    
    # Feature: x values from -3 to 3
    X = np.linspace(-3, 3, n_samples).reshape(-1, 1)
    
    # True function: f(x) = x^2 with increasing noise away from origin
    true_func = lambda x: x**2
    noise_std_func = lambda x: 0.5 + 0.5 * np.abs(x)
    
    # Generate y values with heteroskedastic noise
    y_true = true_func(X.flatten()) + np.random.normal(0, noise_std_func(X.flatten()), n_samples)
    
    # Define alpha for 95% intervals
    alpha_level = 0.05
    target_coverage_level = 1 - alpha_level

    # Create prediction intervals with different properties
    # Median prediction (can be the true function for this demo, or a model's prediction)
    f_pred = true_func(X.flatten())

    # 1. "Perfect" intervals (oracle, centered on true values with correct width for ~95% coverage)
    # Using 1.96 for ~95% coverage with normal noise assumption
    perfect_half_width = noise_std_func(X.flatten()) * 1.96 
    perfect_lower = y_true - perfect_half_width # Centered on y_true for an oracle view
    perfect_upper = y_true + perfect_half_width
    
    # 2. Constant width intervals (good in center, poor at edges, centered on f_pred)
    constant_width_val = 2.0 
    constant_lower = f_pred - constant_width_val / 2
    constant_upper = f_pred + constant_width_val / 2
    
    # 3. Too narrow intervals (poor coverage, centered on f_pred)
    narrow_half_width = noise_std_func(X.flatten()) * 0.5
    narrow_lower = f_pred - narrow_half_width
    narrow_upper = f_pred + narrow_half_width
    
    # 4. Too wide intervals (good coverage but inefficient, centered on f_pred)
    wide_half_width = noise_std_func(X.flatten()) * 3.0 # Wider than perfect
    wide_lower = f_pred - wide_half_width
    wide_upper = f_pred + wide_half_width

    # 5. Intervals that cross (lower > upper for some points)
    crossing_lower = f_pred.copy()
    crossing_upper = f_pred.copy()
    crossing_lower[::2] = f_pred[::2] + 0.5 # Lower is above median
    crossing_upper[::2] = f_pred[::2] - 0.5 # Upper is below median
    
    methods = {
        "Perfect (Oracle)": (perfect_lower, perfect_upper, y_true), # Oracle uses y_true as its median for demo
        "Constant Width": (constant_lower, constant_upper, f_pred),
        "Too Narrow": (narrow_lower, narrow_upper, f_pred),
        "Too Wide": (wide_lower, wide_upper, f_pred),
        "Crossing Intervals": (crossing_lower, crossing_upper, f_pred)
    }
    
    print(f"\n=== Interval Evaluation Metrics (Target Coverage: {target_coverage_level:.2f}) ===")
    
    all_methods_metrics_summary = {}

    for method_name, (lower, upper, median_for_eval) in methods.items():
        print(f"\n--- {method_name} Intervals ---")
        
        # Use the combined evaluation function
        metrics_results = evaluate_intervals(y_true, lower, upper, alpha=alpha_level, f=median_for_eval)
        all_methods_metrics_summary[method_name] = metrics_results
        
        for metric_name_key, value in metrics_results.items():
            # Handle c_test_cal which might be np.nan
            if isinstance(value, float) and np.isnan(value):
                print(f"  {metric_name_key}: nan")
            else:
                print(f"  {metric_name_key}: {value:.4f}")
    
    # --- Visualization --- 
    # Reduce number of methods for cleaner plots if needed, or plot all
    methods_to_plot = ["Perfect (Oracle)", "Constant Width", "Too Narrow", "Too Wide"]
    
    plt.figure(figsize=(15, 6)) # Adjusted figsize for 1x2 layout
    plt.suptitle(f"Metrics Demonstration (Target Coverage: {target_coverage_level:.2f})", fontsize=16)

    # Plot 1: Intervals for selected methods
    ax1 = plt.subplot(1, 2, 1) # Changed to 1x2 layout
    ax1.scatter(X.flatten(), y_true, s=15, color='black', alpha=0.7, label='Data points')
    ax1.plot(X.flatten(), true_func(X.flatten()), 'k--', lw=1.5, label='True function (underlying)')
    
    colors = {
        "Perfect (Oracle)": 'forestgreen',
        "Constant Width": 'royalblue',
        "Too Narrow": 'darkorange',
        "Too Wide": 'mediumpurple'
    }

    for i, method_name_plot in enumerate(methods_to_plot):
        lower_p, upper_p, _ = methods[method_name_plot]
        picp_val = all_methods_metrics_summary[method_name_plot]['PICP']
        niw_val = all_methods_metrics_summary[method_name_plot]['NIW']
        
        # Sort by X for nicer visualization
        idx = np.argsort(X.flatten())
        x_sorted = X.flatten()[idx]
        lower_sorted = lower_p[idx]
        upper_sorted = upper_p[idx]
        
        ax1.fill_between(x_sorted, lower_sorted, upper_sorted, 
                        color=colors.get(method_name_plot, 'gray'), alpha=0.25, 
                        label=f'{method_name_plot}\n  PICP={picp_val:.2f}, NIW={niw_val:.2f}')
    
    ax1.set_title('Prediction Intervals')
    ax1.set_xlabel('Feature (x)')
    ax1.set_ylabel('Target (y)')
    ax1.legend(fontsize=9)
    ax1.grid(True, linestyle=':', alpha=0.5)

    # Plot 2: PICP vs NIW (AUC-like curve) for selected methods
    ax2 = plt.subplot(1, 2, 2) # Changed to 1x2 layout
    for method_name_plot in methods_to_plot:
        lower_p, upper_p, median_p = methods[method_name_plot]
        auc_val = all_methods_metrics_summary[method_name_plot].get('AUC', np.nan)
        plot_auc_curve(y_true, lower_p, upper_p, f=median_p, ax=ax2, 
                       label=f'{method_name_plot} (AUC={auc_val:.2f})', 
                       color=colors.get(method_name_plot, 'gray'), lw=2)
    ax2.set_title('PICP vs. NIW Curve')
    ax2.set_xlabel('PICP')
    ax2.set_ylabel('NIW')
    ax2.legend(fontsize=9)
    ax2.grid(True, linestyle=':', alpha=0.5)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
    
    # Show the plot
    plt.show()
    
    print("\nThis module provides metrics for evaluating prediction intervals in the CLEAR framework.")