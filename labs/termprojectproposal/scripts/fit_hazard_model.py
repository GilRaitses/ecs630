#!/usr/bin/env python3
"""
Fit event-hazard GLM models to larval trajectory data.

This script:
1. Loads trajectory and stimulus data
2. Extracts features (stimulus history, contextual features)
3. Fits GLM with temporal kernel for hazard rates
4. Validates model (KS test, PSTH comparison)
5. Saves fitted model and results
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import pickle
import sys
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def load_config(config_path="config/model_config.json"):
    """Load model configuration."""
    with open(config_path, 'r') as f:
        return json.load(f)

def create_temporal_kernel_basis(times, n_basis=10, time_range=(-2.0, 20.0)):
    """
    Create raised cosine basis functions for temporal kernel.
    
    Parameters
    ----------
    times : array-like
        Time points relative to stimulus onset (seconds)
    n_basis : int
        Number of basis functions
    time_range : tuple
        (min_time, max_time) for kernel support
    
    Returns
    -------
    basis_matrix : ndarray, shape (n_times, n_basis)
        Basis function evaluations at each time point
    """
    times = np.array(times)
    t_min, t_max = time_range
    knot_positions = np.linspace(t_min, t_max, n_basis)
    delta_tau = (t_max - t_min) / (n_basis - 1)
    
    basis_matrix = np.zeros((len(times), n_basis))
    
    for j, tau_j in enumerate(knot_positions):
        # Raised cosine: cos^2(pi*(t - tau_j)/(2*delta_tau)) for |t - tau_j| < delta_tau
        tau_diff = times - tau_j
        mask = np.abs(tau_diff) < delta_tau
        
        if np.any(mask):
            cos_arg = np.pi * tau_diff[mask] / (2 * delta_tau)
            basis_matrix[mask, j] = np.cos(cos_arg) ** 2
    
    return basis_matrix

def extract_stimulus_kernel_features(events_df, config):
    """
    Extract stimulus history features using temporal kernel convolution.
    
    For each time bin, computes stimulus history features by convolving
    stimulus signal with temporal kernel basis functions.
    
    Uses configurable analysis window from config (default: [-3, +8] seconds).
    Only stimulus events within the analysis window contribute to kernel features.
    
    Parameters
    ----------
    events_df : DataFrame
        Event records with time, led_intensity, stimulus_on, time_since_stimulus
    config : dict
        Model configuration (must contain 'model.temporal_kernel.time_range')
    
    Returns
    -------
    kernel_features : ndarray, shape (n_samples, n_basis)
        Kernel feature matrix, computed using configurable analysis window
    """
    n_basis = config['model']['temporal_kernel']['n_basis']
    time_range = config['model']['temporal_kernel']['time_range']
    t_min, t_max = time_range  # Analysis window: e.g., [-3.0, 8.0] seconds
    
    n_samples = len(events_df)
    kernel_features = np.zeros((n_samples, n_basis))
    
    # Get stimulus onsets
    if 'stimulus_onset' in events_df.columns:
        onset_mask = events_df['stimulus_onset'].values == 1
        if np.sum(onset_mask) == 0:
            # No onsets detected, use time_since_stimulus
            return extract_stimulus_kernel_features_from_time_since(events_df, config)
    else:
        # Use time_since_stimulus to infer stimulus history
        return extract_stimulus_kernel_features_from_time_since(events_df, config)
    
    # Create kernel basis for time window
    time_window = np.arange(t_min, t_max, 0.05)  # 50ms resolution
    kernel_basis = create_temporal_kernel_basis(time_window, n_basis, time_range)
    
    # For each time bin, compute convolution with recent stimulus history
    # Use led1Val if available, otherwise led_intensity
    if 'led1Val' in events_df.columns:
        led_intensity = events_df['led1Val'].values
    elif 'led_intensity' in events_df.columns:
        led_intensity = events_df['led_intensity'].values
    else:
        # Fallback: use stimulus_on as binary intensity
        led_intensity = events_df['stimulus_on'].values.astype(float) * 255.0
    
    times = events_df['time'].values
    
    for i in range(n_samples):
        t_current = times[i]
        
        # Extract stimulus history window
        window_start = max(0, int((t_current - t_max) / 0.05))
        window_end = int((t_current - t_min) / 0.05) + 1
        
        if window_start < len(led_intensity) and window_end > 0:
            window_start = max(0, window_start)
            window_end = min(len(led_intensity), window_end)
            
            if window_end > window_start:
                # Get stimulus values in window (reversed: most recent first)
                stimulus_window = led_intensity[window_start:window_end][::-1]
                
                # Align with kernel (most recent stimulus at delay 0)
                kernel_aligned = np.zeros(len(time_window))
                n_copy = min(len(stimulus_window), len(kernel_aligned))
                kernel_aligned[:n_copy] = stimulus_window[:n_copy]
                
                # Convolve with each basis function
                for j in range(n_basis):
                    kernel_features[i, j] = np.sum(kernel_basis[:, j] * kernel_aligned)
    
    return kernel_features

def extract_stimulus_kernel_features_from_time_since(events_df, config):
    """
    Extract kernel features using time_since_stimulus when onsets not explicitly marked.
    
    Uses configurable analysis window from config (default: [-3, +8] seconds).
    Only time points within the analysis window contribute to kernel features.
    
    Parameters
    ----------
    events_df : DataFrame
        Event records with time_since_stimulus column
    config : dict
        Model configuration (must contain 'model.temporal_kernel.time_range')
    
    Returns
    -------
    kernel_features : ndarray, shape (n_samples, n_basis)
        Kernel feature matrix, with zero features for time points outside analysis window
    """
    n_basis = config['model']['temporal_kernel']['n_basis']
    time_range = config['model']['temporal_kernel']['time_range']
    t_min, t_max = time_range  # Analysis window: e.g., [-3.0, 8.0] seconds
    
    n_samples = len(events_df)
    kernel_features = np.zeros((n_samples, n_basis))
    
    # Create kernel basis
    time_window = np.arange(t_min, t_max, 0.05)
    kernel_basis = create_temporal_kernel_basis(time_window, n_basis, time_range)
    
    # Get time since stimulus and current intensity
    time_since = events_df['time_since_stimulus'].values
    # Use led1Val if available, otherwise led_intensity
    if 'led1Val' in events_df.columns:
        led_intensity = events_df['led1Val'].values
    elif 'led_intensity' in events_df.columns:
        led_intensity = events_df['led_intensity'].values
    else:
        # Fallback: use stimulus_on as binary intensity
        led_intensity = events_df['stimulus_on'].values.astype(float) * 255.0
    
    for i in range(n_samples):
        # Find which kernel basis to use based on time_since_stimulus
        tau = time_since[i]
        
        # CRITICAL: Only include time points within analysis window [-3, +8]
        # Values outside this window should have zero kernel contribution
        if tau < t_min or tau > t_max:
            # Outside analysis window - set kernel features to zero
            kernel_features[i, :] = 0.0
            continue
        
        # Clamp to time_range bounds (for numerical stability near edges)
        tau = np.clip(tau, t_min, t_max)
        
        # Find corresponding time index in kernel basis array
        time_idx = int((tau - t_min) / 0.05)
        time_idx = np.clip(time_idx, 0, len(time_window) - 1)
        
        # Get kernel basis values at this delay
        # Weight by current stimulus intensity
        if led_intensity[i] > 0:  # Stimulus is on
            kernel_features[i, :] = kernel_basis[time_idx, :] * led_intensity[i]
        else:
            # Stimulus off, but include kernel based on recency (small baseline)
            kernel_features[i, :] = kernel_basis[time_idx, :] * 0.1
    return kernel_features

def extract_stimulus_history(stimulus_times, stimulus_values, time_window=(-2.0, 20.0), bin_width=0.05):
    """
    Extract stimulus history features using temporal kernel.
    
    Parameters
    ----------
    stimulus_times : array-like
        Absolute times of stimulus events
    stimulus_values : array-like
        Stimulus intensity values
    time_window : tuple
        (before, after) stimulus onset in seconds
    bin_width : float
        Time bin width in seconds
    
    Returns
    -------
    stimulus_features : dict
        Dictionary with 'times' and 'kernel_features' arrays
    """
    t_min, t_max = time_window
    times = np.arange(t_min, t_max, bin_width)
    
    # Create kernel basis
    kernel_basis = create_temporal_kernel_basis(times, n_basis=10, time_range=time_window)
    
    # For each stimulus event, compute kernel convolution
    # Simplified: assume single stimulus at t=0 for now
    # In full implementation, would convolve with stimulus signal
    
    kernel_features = kernel_basis.sum(axis=1)  # Placeholder: sum of all basis functions
    
    return {
        'times': times,
        'kernel_features': kernel_features
    }

def prepare_feature_matrix(events_df, config, baseline_only=False, event_type='turn'):
    """
    Prepare feature matrix for GLM fitting from event records.
    
    Supports multiple event types: turn, pause, reversal (stop).
    Includes spine curve energy as a feature.
    
    Parameters
    ----------
    events_df : DataFrame
        Event records with time bins, features, and event indicators
    config : dict
        Model configuration
    baseline_only : bool
        If True, only fit constant baseline (no features)
    event_type : str
        Type of event to model ('turn', 'pause', 'stop', 'reversal', 'reverse')
    
    Returns
    -------
    X : ndarray
        Feature matrix (n_samples, n_features)
    y : ndarray
        Binary event indicators (n_samples,)
    feature_names : list
        Names of features
    bin_width : float
        Time bin width (for converting hazard to rate)
    """
    # Map event type to column name
    event_column_map = {
        'turn': 'is_turn',
        'pause': 'is_pause',
        'stop': 'is_pause',
        'reversal': 'is_reversal',
        'reverse': 'is_reversal',
        'reorientation': 'is_reorientation'  # MAGAT-compatible reorientations
    }
    event_column = event_column_map.get(event_type.lower(), 'is_turn')
    
    # Filter out invalid rows
    valid_mask = ~(events_df['time'].isna())
    if event_column in events_df.columns:
        valid_mask = valid_mask & ~(events_df[event_column].isna())
    events_df = events_df[valid_mask].copy()
    
    if len(events_df) == 0:
        raise ValueError(f"No valid event records found for {event_type}")
    
    # Get event indicators
    if event_column in events_df.columns:
        y = events_df[event_column].values.astype(int)
    else:
        raise ValueError(f"Event column '{event_column}' not found in data")
    
    if baseline_only:
        # Null model: constant hazard (intercept only)
        X = np.ones((len(events_df), 1))
        feature_names = ['intercept']
    else:
        # Full model: intercept + stimulus kernel + contextual features
        X_features = []
        feature_names = []
        
        # Intercept
        X_features.append(np.ones(len(events_df)))
        feature_names.append('intercept')
        
        # Stimulus kernel features
        print(f"  Extracting stimulus kernel features...")
        kernel_features = extract_stimulus_kernel_features(events_df, config)
        n_basis = kernel_features.shape[1]
        for j in range(n_basis):
            X_features.append(kernel_features[:, j])
            feature_names.append(f'kernel_{j}')
        
        # Contextual features (normalized)
        if 'speed' in events_df.columns:
            speed = events_df['speed'].values
            speed_normalized = (speed - speed.mean()) / (speed.std() + 1e-6)
            X_features.append(speed_normalized)
            feature_names.append('speed_normalized')
        
        if 'heading' in events_df.columns:
            heading = events_df['heading'].values
            # Normalize heading to [-pi, pi] range, then use sin/cos
            heading_norm = (heading - heading.mean()) / (heading.std() + 1e-6)
            X_features.append(np.sin(heading_norm))
            X_features.append(np.cos(heading_norm))
            feature_names.append('heading_sin')
            feature_names.append('heading_cos')
        
        # Spine curve energy (bending energy) - continuous feature
        if 'spine_curve_energy' in events_df.columns:
            spine_energy = events_df['spine_curve_energy'].values
            # Log-transform to handle large dynamic range, then normalize
            spine_energy_log = np.log(spine_energy + 1e-6)  # Add small epsilon to avoid log(0)
            spine_energy_normalized = (spine_energy_log - spine_energy_log.mean()) / (spine_energy_log.std() + 1e-6)
            X_features.append(spine_energy_normalized)
            feature_names.append('spine_curve_energy_normalized')
        
        X = np.column_stack(X_features)
    
    # Get bin width from config
    bin_width = config['model'].get('bin_width', 0.05)
    
    return X, y, feature_names, bin_width

def fit_hazard_glm(X, y, config, baseline_only=False):
    """
    Fit GLM for event hazard.
    
    For baseline model: fits constant hazard (Poisson process with rate λ)
    For full model: fits logistic regression (approximation to Poisson GLM)
    
    Parameters
    ----------
    X : ndarray
        Feature matrix
    y : ndarray
        Binary event indicators
    config : dict
        Model configuration
    baseline_only : bool
        If True, fit as Poisson (rate estimation), else logistic regression
    
    Returns
    -------
    model : dict
        Dictionary with fitted parameters and model info
    """
    if baseline_only:
        # Constant hazard model: estimate rate λ
        # For rare events in small bins, P(event) ≈ λ * bin_width
        n_events = np.sum(y)
        n_bins = len(y)
        bin_width = config['model'].get('bin_width', 0.05)
        
        # Maximum likelihood estimate: λ = (n_events) / (total_time)
        total_time = n_bins * bin_width
        lambda_hat = n_events / total_time
        
        # Confidence interval (Poisson rate)
        from scipy.stats import poisson
        alpha = 1 - config['confidence_intervals']['target_coverage']
        ci_lower = poisson.ppf(alpha/2, n_events) / total_time
        ci_upper = poisson.ppf(1 - alpha/2, n_events) / total_time
        
        # Convert to turn rate (turns per minute)
        rate_per_min = lambda_hat * 60
        ci_lower_per_min = ci_lower * 60
        ci_upper_per_min = ci_upper * 60
        
        model = {
            'type': 'baseline_constant',
            'lambda_hat': lambda_hat,
            'rate_per_min': rate_per_min,
            'ci_lower': ci_lower_per_min,
            'ci_upper': ci_upper_per_min,
            'n_events': int(n_events),
            'n_bins': n_bins,
            'total_time': total_time,
            'bin_width': bin_width
        }
    else:
        # Full GLM model (will implement later)
        # For now, scale features and fit logistic regression
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        lambda_reg = config['model']['regularization']['lambda']
        C = 1.0 / lambda_reg
        
        glm_model = LogisticRegression(
            C=C,
            penalty='l2',
            solver='lbfgs',
            max_iter=1000
        )
        
        glm_model.fit(X_scaled, y)
        
        model = {
            'type': 'glm_logistic',
            'sklearn_model': glm_model,
            'scaler': scaler,
            'coefficients': glm_model.coef_[0],
            'intercept': glm_model.intercept_[0],
            'config': config  # Store config for KS test
        }
    
    return model

def compute_ks_test_baseline(model, events_df, bin_width, event_type='turn'):
    """
    Time-rescaled KS test for baseline constant hazard model.
    
    For constant hazard λ, rescaled times should be uniform.
    
    Parameters
    ----------
    model : dict
        Fitted baseline model
    events_df : DataFrame
        Event records with time and event columns
    bin_width : float
        Time bin width
    event_type : str
        Type of event ('turn', 'pause', 'reversal')
    
    Returns
    -------
    ks_statistic : float
        KS test statistic
    p_value : float
        P-value for uniformity test
    """
    # Map event type to column name
    event_column_map = {
        'turn': 'is_turn',
        'pause': 'is_pause',
        'stop': 'is_pause',
        'reversal': 'is_reversal',
        'reverse': 'is_reversal',
        'reorientation': 'is_reorientation'  # MAGAT-compatible reorientations
    }
    event_column = event_column_map.get(event_type.lower(), 'is_turn')
    
    lambda_hat = model['lambda_hat']
    
    # Get event times
    event_mask = events_df[event_column].values == 1
    event_times = events_df.loc[event_mask, 'time'].values
    
    if len(event_times) == 0:
        return np.nan, np.nan
    
    # For constant hazard λ, cumulative hazard = λ * t
    lambda_hat = model['lambda_hat']
    
    # Time-rescale: transform event times by cumulative hazard
    rescaled_times = lambda_hat * event_times
    
    # Under correct model, rescaled times should be Poisson process increments
    # Normalize to [0, 1] for KS test
    max_time = events_df['time'].max()
    max_rescaled = lambda_hat * max_time
    
    if max_rescaled > 0:
        normalized = rescaled_times / max_rescaled
        # KS test: should be uniform on [0, 1]
        ks_statistic, p_value = stats.kstest(normalized, 'uniform')
    else:
        ks_statistic, p_value = np.nan, np.nan
    
    return ks_statistic, p_value

def compute_ks_test_full(model, events_df, bin_width, event_type='turn'):
    """
    Time-rescaled KS test for full GLM model.
    
    Parameters
    ----------
    model : dict
        Fitted GLM model with sklearn_model and scaler
    events_df : DataFrame
        Event records
    bin_width : float
        Time bin width
    event_type : str
        Type of event ('turn', 'pause', 'reversal')
    
    Returns
    -------
    ks_statistic : float
        KS test statistic
    p_value : float
        P-value for uniformity test
    """
    # Map event type to column name
    event_column_map = {
        'turn': 'is_turn',
        'pause': 'is_pause',
        'stop': 'is_pause',
        'reversal': 'is_reversal',
        'reverse': 'is_reversal',
        'reorientation': 'is_reorientation'  # MAGAT-compatible reorientations
    }
    event_column = event_column_map.get(event_type.lower(), 'is_turn')
    
    # Prepare features (same as during fitting)
    config = model.get('config', {'model': {'bin_width': bin_width}})
    X, y, _, _ = prepare_feature_matrix(events_df, config, baseline_only=False, event_type=event_type)
    
    # Compute predicted hazards
    X_scaled = model['scaler'].transform(X)
    hazards = model['sklearn_model'].predict_proba(X_scaled)[:, 1]
    
    # Convert probabilities to hazard rates (per second)
    # P(event in bin) ≈ λ(t) * bin_width
    hazard_rates = hazards / bin_width
    
    # Get event times
    event_mask = events_df[event_column].values == 1
    event_times = events_df.loc[event_mask, 'time'].values
    
    if len(event_times) == 0:
        return np.nan, np.nan
    
    # Compute cumulative hazard: Λ(t) = ∫₀ᵗ λ(s) ds
    # Approximate as sum of hazard rates * bin_width
    times = events_df['time'].values
    
    # Sort by time to ensure proper integration
    sort_idx = np.argsort(times)
    times_sorted = times[sort_idx]
    hazard_rates_sorted = hazard_rates[sort_idx]
    
    cumulative_hazard = np.zeros(len(times_sorted))
    
    for i in range(1, len(times_sorted)):
        dt = times_sorted[i] - times_sorted[i-1]
        if dt > 0:
            cumulative_hazard[i] = cumulative_hazard[i-1] + hazard_rates_sorted[i] * dt
        else:
            cumulative_hazard[i] = cumulative_hazard[i-1]
    
    # Time-rescale: transform event times (need to map back to sorted order)
    event_indices_sorted = np.searchsorted(times_sorted, event_times)
    event_indices_sorted = np.clip(event_indices_sorted, 0, len(times_sorted) - 1)
    rescaled_times = cumulative_hazard[event_indices_sorted]
    
    # Normalize to [0, 1] for KS test
    max_cumulative = cumulative_hazard[-1]
    if max_cumulative > 0:
        normalized = rescaled_times / max_cumulative
        ks_statistic, p_value = stats.kstest(normalized, 'uniform')
    else:
        ks_statistic, p_value = np.nan, np.nan
    
    return ks_statistic, p_value

def main():
    """Main fitting pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fit event-hazard GLM model')
    parser.add_argument('--trajectory-dir', type=str, required=True,
                       help='Directory containing event CSV files')
    parser.add_argument('--events-file', type=str, default=None,
                       help='Path to events CSV file (if None, looks for *_events.csv)')
    parser.add_argument('--event-type', type=str, default='turn',
                       choices=['turn', 'pause', 'stop', 'reversal', 'reverse', 'reorientation'],
                       help='Type of event to model (reorientation uses MAGAT-compatible reorientations)')
    parser.add_argument('--output-dir', type=str, default='output/fitted_models',
                       help='Output directory for fitted model')
    parser.add_argument('--config', type=str, default='config/model_config.json',
                       help='Path to model config JSON')
    parser.add_argument('--baseline-only', action='store_true',
                       help='Fit only baseline (constant hazard) model')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    baseline_flag = args.baseline_only
    
    print(f"Fitting {args.event_type} hazard model ({'baseline' if baseline_flag else 'full'}...")
    print(f"Loading data from {args.trajectory_dir}")
    
    # Find events file
    trajectory_dir = Path(args.trajectory_dir)
    if args.events_file:
        events_path = Path(args.events_file)
    else:
        # Look for *_events.csv files
        events_files = list(trajectory_dir.glob("*_events.csv"))
        if not events_files:
            print(f"ERROR: No events CSV files found in {trajectory_dir}")
            sys.exit(1)
        events_path = events_files[0]
        print(f"Using events file: {events_path.name}")
    
    # Load events data
    print(f"Loading events from {events_path}...")
    events_df = pd.read_csv(events_path)
    print(f"  Loaded {len(events_df)} event records")
    if 'track_id' in events_df.columns:
        print(f"  Tracks: {events_df['track_id'].nunique()}")
    
    # Get event column name for reporting
    event_column_map = {
        'turn': 'is_turn',
        'pause': 'is_pause',
        'stop': 'is_pause',
        'reversal': 'is_reversal',
        'reverse': 'is_reversal'
    }
    event_column = event_column_map.get(args.event_type.lower(), 'is_turn')
    
    if event_column in events_df.columns:
        print(f"  {args.event_type.title()} events: {events_df[event_column].sum()}")
    else:
        print(f"  Warning: Event column '{event_column}' not found")
    
    # Prepare features
    print("Preparing feature matrix...")
    X, y, feature_names, bin_width = prepare_feature_matrix(events_df, config, baseline_only=baseline_flag, event_type=args.event_type)
    print(f"  Features: {len(feature_names)}")
    print(f"  Samples: {len(y)}")
    print(f"  Events: {np.sum(y)} ({100*np.mean(y):.2f}%)")
    
    # Fit model
    print("Fitting model...")
    model = fit_hazard_glm(X, y, config, baseline_only=baseline_flag)
    
    if baseline_flag:
        print(f"\nBaseline Model Results:")
        print(f"  Hazard rate (λ): {model['lambda_hat']:.6f} events/sec")
        event_rate_field = f'{args.event_type}_rate_per_min'
        if 'rate_per_min' in model:
            print(f"  {args.event_type.title()} rate: {model['rate_per_min']:.2f} {args.event_type}s/min")
            print(f"  95% CI: [{model['ci_lower']:.2f}, {model['ci_upper']:.2f}] {args.event_type}s/min")
        print(f"  Total events: {model['n_events']}")
        print(f"  Total time: {model['total_time']:.1f} seconds ({model['total_time']/60:.1f} minutes)")
        
        # Validation
        print("\nValidating model...")
        ks_stat, p_value = compute_ks_test_baseline(model, events_df, bin_width, event_type=args.event_type)
        print(f"  KS test statistic: {ks_stat:.4f}")
        print(f"  KS test p-value: {p_value:.4f}")
        
        if p_value > 0.05:
            print("  ✓ Model not rejected (p > 0.05)")
        else:
            print("  ⚠ Model rejected (p ≤ 0.05)")
    else:
        print(f"\nGLM Model Results:")
        print(f"  Intercept: {model['intercept']:.4f}")
        print(f"  Total features: {len(feature_names)}")
        print(f"  Coefficients:")
        for name, coef in zip(feature_names, np.concatenate([[model['intercept']], model['coefficients']])):
            print(f"    {name}: {coef:.4f}")
        
        # Show kernel coefficient summary
        kernel_coefs = [coef for name, coef in zip(feature_names, np.concatenate([[model['intercept']], model['coefficients']])) 
                        if name.startswith('kernel_')]
        if kernel_coefs:
            print(f"\n  Kernel coefficients: min={min(kernel_coefs):.4f}, max={max(kernel_coefs):.4f}, mean={np.mean(kernel_coefs):.4f}")
        
        # Validation
        print("\nValidating model...")
        ks_stat, p_value = compute_ks_test_full(model, events_df, bin_width, event_type=args.event_type)
        print(f"  KS test statistic: {ks_stat:.4f}")
        print(f"  KS test p-value: {p_value:.4f}")
        
        if p_value > 0.05:
            print("  ✓ Model not rejected (p > 0.05)")
        else:
            print("  ⚠ Model rejected (p ≤ 0.05)")
        
        # Compare to baseline
        baseline_summary_path = Path(args.output_dir) / f'{args.event_type}_baseline_summary.json'
        if baseline_summary_path.exists():
            with open(baseline_summary_path, 'r') as f:
                baseline = json.load(f)
            
            # Get rate field name (could be turn_rate_per_min, pause_rate_per_min, etc.)
            rate_key = f'{args.event_type}_rate_per_min' if f'{args.event_type}_rate_per_min' in baseline else 'turn_rate_per_min'
            baseline_rate = baseline.get(rate_key, baseline.get('rate_per_min', 0))
            
            # Compute model-predicted rate
            X_scaled = model['scaler'].transform(X)
            predicted_probs = model['sklearn_model'].predict_proba(X_scaled)[:, 1]
            predicted_rate = np.mean(predicted_probs) / bin_width * 60
            
            print(f"\n  Comparison to baseline:")
            print(f"    Baseline rate: {baseline_rate:.2f} {args.event_type}s/min")
            print(f"    Model rate: {predicted_rate:.2f} {args.event_type}s/min")
            print(f"    Improvement: {100*(predicted_rate - baseline_rate)/baseline_rate:.1f}%")
    
    # Save model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / f'{args.event_type}_{"baseline" if baseline_flag else "full"}_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n✓ Saved model to {model_path}")
    
    # Save summary
    if baseline_flag:
        summary = {
            'model_type': 'baseline_constant',
            'event_type': args.event_type,
            'hazard_rate_per_sec': float(model['lambda_hat']),
            'rate_per_min': float(model['rate_per_min']),
            'ci_lower_per_min': float(model['ci_lower']),
            'ci_upper_per_min': float(model['ci_upper']),
            'n_events': int(model['n_events']),
            'n_bins': int(model['n_bins']),
            'total_time_sec': float(model['total_time']),
            'ks_statistic': float(ks_stat) if not np.isnan(ks_stat) else None,
            'ks_p_value': float(p_value) if not np.isnan(p_value) else None
        }
    else:
        ks_stat, p_value = compute_ks_test_full(model, events_df, bin_width, event_type=args.event_type)
        
        # Compute predicted rate
        X_scaled = model['scaler'].transform(X)
        predicted_probs = model['sklearn_model'].predict_proba(X_scaled)[:, 1]
        predicted_rate = np.mean(predicted_probs) / bin_width * 60
        
        summary = {
            'model_type': 'glm_logistic',
            'intercept': float(model['intercept']),
            'coefficients': {name: float(coef) for name, coef in zip(feature_names, np.concatenate([[model['intercept']], model['coefficients']]))},
            'kernel_coefficients': {name: float(coef) for name, coef in zip(feature_names, np.concatenate([[model['intercept']], model['coefficients']])) 
                                   if name.startswith('kernel_')},
            'n_samples': len(y),
            'n_events': int(np.sum(y)),
            'event_rate': float(np.mean(y)),
            'predicted_rate_per_min': float(predicted_rate),
            'ks_statistic': float(ks_stat) if not np.isnan(ks_stat) else None,
            'ks_p_value': float(p_value) if not np.isnan(p_value) else None
        }
    
    summary_path = output_dir / f'{args.event_type}_{"baseline" if baseline_flag else "full"}_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved summary to {summary_path}")
    
    # Save coefficients table (if applicable)
    if not baseline_flag:
        # feature_names includes 'intercept' as first element
        # model['coefficients'] excludes intercept (it's separate)
        # So we need to match them up correctly
        all_coefs = []
        for name in feature_names:
            if name == 'intercept':
                all_coefs.append(model['intercept'])
            else:
                # Find index in feature_names (skip intercept)
                idx = feature_names.index(name) - 1  # -1 because intercept is first
                all_coefs.append(model['coefficients'][idx])
        
        coef_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': all_coefs
        })
        coef_path = output_dir / f'{args.event_type}_coefficients.csv'
        coef_df.to_csv(coef_path, index=False)
        print(f"✓ Saved coefficients to {coef_path}")

if __name__ == '__main__':
    main()

