#!/usr/bin/env python3
"""
Simulate larval trajectories using fitted hazard models.

Generates trajectories by:
1. Sampling events (reorientations, pauses, heading_reversals) from fitted hazard model
2. Updating position/orientation based on events
3. Integrating position with empirical speed distributions
4. Outputting trajectory data compatible with DOE analysis
"""

import sys
import argparse
import json
import pickle
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def load_config(config_path="config/model_config.json"):
    """Load model configuration."""
    with open(config_path, 'r') as f:
        return json.load(f)

def load_fitted_model(model_path):
    """Load fitted model from pickle file."""
    # Import sklearn before loading (models contain sklearn objects)
    try:
        import sklearn.linear_model
        import sklearn.preprocessing
    except ImportError:
        pass  # Continue anyway - model may not need sklearn
    
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def compute_speed_distribution_params(events_df):
    """
    Pre-compute speed distribution parameters for efficient sampling.
    
    Parameters
    ----------
    events_df : DataFrame
        Empirical trajectory data
    
    Returns
    -------
    params : dict
        Dictionary with mu, sigma for log-normal distribution
    """
    speeds = events_df['speed'].values
    speeds = speeds[speeds > 0]  # Filter out zeros
    
    if len(speeds) == 0:
        return {'mu': np.log(0.01), 'sigma': 0.1, 'default': 0.01}
    
    # Fit log-normal distribution
    log_speeds = np.log(speeds + 1e-6)
    mu = np.mean(log_speeds)
    sigma = np.std(log_speeds)
    
    return {'mu': mu, 'sigma': sigma, 'default': 0.01}

def sample_speed_from_params(params, n_samples=1):
    """
    Sample speeds using pre-computed distribution parameters.
    
    Parameters
    ----------
    params : dict
        Distribution parameters (mu, sigma) from compute_speed_distribution_params
    n_samples : int
        Number of samples to draw
    
    Returns
    -------
    speeds : ndarray
        Sampled speeds
    """
    if 'default' in params and params.get('default') is not None:
        sampled_log = np.random.normal(params['mu'], params['sigma'], n_samples)
        sampled_speeds = np.exp(sampled_log)
        return sampled_speeds
    else:
        return np.full(n_samples, params.get('default', 0.01))

def compute_hazard_rate(model, features_dict, bin_width):
    """
    Compute hazard rate from fitted model given features.
    
    Parameters
    ----------
    model : dict
        Fitted GLM model
    features_dict : dict
        Dictionary with feature values (kernel_features, speed, heading, spine_curve_energy, etc.)
    bin_width : float
        Time bin width
    
    Returns
    -------
    hazard_rate : float
        Hazard rate (events per second)
    """
    if model['type'] == 'baseline_constant':
        return model['lambda_hat']
    
    elif model['type'] == 'glm_logistic':
        # Reconstruct feature vector (same order as in fit_hazard_model.py)
        feature_names = ['intercept']
        feature_values = [1.0]  # intercept
        
        # Add kernel features
        if 'kernel_features' in features_dict:
            kernel_feats = features_dict['kernel_features']
            for j in range(len(kernel_feats)):
                feature_names.append(f'kernel_{j}')
                feature_values.append(kernel_feats[j])
        else:
            # If no kernel features, use zeros
            n_basis = features_dict.get('n_basis', 10)
            for j in range(n_basis):
                feature_names.append(f'kernel_{j}')
                feature_values.append(0.0)
        
        # Add contextual features (in same order as training)
        if 'speed_normalized' in features_dict:
            feature_names.append('speed_normalized')
            feature_values.append(features_dict['speed_normalized'])
        else:
            feature_names.append('speed_normalized')
            feature_values.append(0.0)
        
        if 'heading_sin' in features_dict:
            feature_names.append('heading_sin')
            feature_values.append(features_dict['heading_sin'])
        else:
            feature_names.append('heading_sin')
            feature_values.append(0.0)
        
        if 'heading_cos' in features_dict:
            feature_names.append('heading_cos')
            feature_values.append(features_dict['heading_cos'])
        else:
            feature_names.append('heading_cos')
            feature_values.append(0.0)
        
        # Add spine curve energy (if available)
        if 'spine_curve_energy_normalized' in features_dict:
            feature_names.append('spine_curve_energy_normalized')
            feature_values.append(features_dict['spine_curve_energy_normalized'])
        elif 'spine_curve_energy' in features_dict:
            # Normalize spine curve energy (log-transform then normalize)
            spine_energy = features_dict['spine_curve_energy']
            # Use empirical mean/std if provided, otherwise use defaults
            spine_mean = features_dict.get('spine_energy_mean', 0.0)
            spine_std = features_dict.get('spine_energy_std', 1.0)
            spine_log = np.log(spine_energy + 1e-6)
            spine_normalized = (spine_log - spine_mean) / (spine_std + 1e-6)
            feature_names.append('spine_curve_energy_normalized')
            feature_values.append(spine_normalized)
        else:
            # No spine curve energy available
            feature_names.append('spine_curve_energy_normalized')
            feature_values.append(0.0)
        
        # Create feature vector
        X = np.array(feature_values).reshape(1, -1)
        
        # Scale using model's scaler
        X_scaled = model['scaler'].transform(X)
        
        # Predict probability
        prob = model['sklearn_model'].predict_proba(X_scaled)[0, 1]
        
        # Convert to hazard rate: P(event) ≈ λ * bin_width
        hazard_rate = prob / bin_width
        
        return hazard_rate
    
    else:
        raise ValueError(f"Unknown model type: {model['type']}")

def extract_stimulus_kernel_features_at_time(t, stimulus_onsets, stimulus_intensity_at_t, config):
    """
    Extract kernel features at a specific time given stimulus onset history.
    
    Uses configurable analysis window from config (default: [-3, +8] seconds relative to onset).
    Only stimulus onsets within the analysis window contribute to kernel features.
    
    Parameters
    ----------
    t : float
        Current time
    stimulus_onsets : list
        List of (onset_time, onset_intensity) tuples for recent stimulus onsets
    stimulus_intensity_at_t : float
        Current stimulus intensity (for ongoing pulses)
    config : dict
        Model configuration (must contain 'model.temporal_kernel.time_range')
    
    Returns
    -------
    kernel_features : ndarray
        Kernel feature vector (n_basis elements)
    """
    n_basis = config['model']['temporal_kernel']['n_basis']
    time_range = config['model']['temporal_kernel']['time_range']
    t_min, t_max = time_range  # e.g., [-3.0, 8.0] seconds
    bin_width = config['model'].get('bin_width', 0.05)
    
    kernel_features = np.zeros(n_basis)
    
    # Create kernel basis functions for the analysis window
    time_window = np.arange(t_min, t_max, bin_width)
    kernel_basis = create_temporal_kernel_basis(time_window, n_basis, time_range)
    
    # Compute convolution: sum over stimulus onsets within analysis window
    for onset_time, onset_intensity in stimulus_onsets:
        # Time since this stimulus onset (tau = t - t_onset)
        tau = t - onset_time
        
        # CRITICAL: Only include if within analysis window [-3, +8]
        # This enforces that stimulus effects are localized to the analysis window
        if tau < t_min or tau > t_max:
            continue  # Outside analysis window - no contribution
        
        # Find index in kernel basis array
        time_idx = int((tau - t_min) / bin_width)
        time_idx = np.clip(time_idx, 0, len(time_window) - 1)
        
        # Weight kernel basis by current stimulus intensity (not just onset intensity)
        # This allows ongoing pulses to have continued effect
        current_weight = stimulus_intensity_at_t if tau >= 0 else onset_intensity
        
        # Add contribution from this stimulus onset (weighted by kernel basis)
        kernel_features += kernel_basis[time_idx, :] * current_weight
    
    return kernel_features

def create_temporal_kernel_basis(times, n_basis=10, time_range=(-2.0, 20.0)):
    """Create raised cosine basis functions (same as in fit_hazard_model.py)."""
    times = np.array(times)
    t_min, t_max = time_range
    knot_positions = np.linspace(t_min, t_max, n_basis)
    delta_tau = (t_max - t_min) / (n_basis - 1)
    
    basis_matrix = np.zeros((len(times), n_basis))
    
    for j, tau_j in enumerate(knot_positions):
        tau_diff = times - tau_j
        mask = np.abs(tau_diff) < delta_tau
        
        if np.any(mask):
            cos_arg = np.pi * tau_diff[mask] / (2 * delta_tau)
            basis_matrix[mask, j] = np.cos(cos_arg) ** 2
    
    return basis_matrix

def simulate_single_trajectory(models_dict, stimulus_schedule, config, 
                               empirical_data=None, max_time=300.0, 
                               dt=0.1, random_seed=None):
    """
    Simulate a single larval trajectory using multiple event models.
    
    Parameters
    ----------
    models_dict : dict
        Dictionary with keys 'reorientation', 'pause', 'heading_reversal' mapping to fitted models
        (can use None for models not available)
        NOTE: 'heading_reversal' is a discrete turn event (~180°), different from 'reverse crawling'
    stimulus_schedule : callable or dict
        Function or dict mapping time -> stimulus intensity
    config : dict
        Model configuration
    empirical_data : DataFrame, optional
        Empirical trajectory data for sampling speeds/starting positions
    max_time : float
        Maximum simulation time (seconds)
    dt : float
        Integration time step (seconds)
    random_seed : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    trajectory : DataFrame
        Simulated trajectory with columns: time, x, y, heading, speed, events
    events : list
        List of (event_type, time) tuples
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    bin_width = config['model'].get('bin_width', 0.05)
    time_range = config['model']['temporal_kernel']['time_range']
    t_min, t_max = time_range  # Analysis window: e.g., [-3.0, 8.0] seconds
    
    # Calculate maximum history buffer size needed
    # Need to keep onsets for at least (abs(t_min) + t_max) seconds to cover full window
    max_history_duration = abs(t_min) + t_max  # e.g., 3 + 8 = 11 seconds
    
    # Initialize state
    # Pre-compute empirical statistics if available (avoids recomputing every timestep)
    empirical_stats = None
    speed_params = None
    
    if empirical_data is not None:
        # Pre-compute statistics once
        speed_mean = empirical_data['speed'].mean()
        speed_std = empirical_data['speed'].std()
        speed_params = compute_speed_distribution_params(empirical_data)
        
        if 'spine_curve_energy' in empirical_data.columns:
            spine_energy_vals = empirical_data['spine_curve_energy'].values
            spine_energy_vals = spine_energy_vals[spine_energy_vals > 0]
            if len(spine_energy_vals) > 0:
                spine_log_vals = np.log(spine_energy_vals + 1e-6)
                spine_energy_mean = np.mean(spine_log_vals)
                spine_energy_std = np.std(spine_log_vals)
            else:
                spine_energy_mean = 0.0
                spine_energy_std = 1.0
        else:
            spine_energy_mean = 0.0
            spine_energy_std = 1.0
        
        empirical_stats = {
            'speed_mean': speed_mean,
            'speed_std': speed_std,
            'spine_energy_mean': spine_energy_mean,
            'spine_energy_std': spine_energy_std
        }
        
        # Sample starting position from empirical data
        start_idx = np.random.randint(len(empirical_data))
        x = empirical_data.iloc[start_idx]['x']
        y = empirical_data.iloc[start_idx]['y']
        heading = empirical_data.iloc[start_idx].get('heading', np.random.uniform(-np.pi, np.pi))
    else:
        x, y = 0.0, 0.0
        heading = np.random.uniform(-np.pi, np.pi)
        speed_mean = 0.0
        speed_std = 1.0
        spine_energy_mean = 0.0
        spine_energy_std = 1.0
    
    # Sample initial speed
    if speed_params is not None:
        current_speed = sample_speed_from_params(speed_params, 1)[0]
    else:
        current_speed = 0.01  # Default speed
    
    # Stimulus onset tracking (for kernel convolution)
    # Track stimulus onsets (not all events) for analysis window computation
    stimulus_onsets = []  # List of (onset_time, onset_intensity) tuples
    
    # State tracking
    times = []
    positions_x = []
    positions_y = []
    headings = []
    speeds = []
    events = []
    stimulus_intensities_recorded = []
    spine_curve_energy_track = []  # Track spine curve energy (MAGAT-compatible)
    spine_theta_track = []  # Track spineTheta (MAGAT body bend angle)
    
    # Import MAGAT spine analysis functions
    try:
        import sys
        from pathlib import Path
        script_dir = Path(__file__).parent
        sys.path.insert(0, str(script_dir))
        from magat_spine_analysis import (
            calculate_spine_theta_magat, 
            calculate_spine_curve_energy_magat,
            lowpass1d
        )
        MAGAT_SPINE_AVAILABLE = True
    except ImportError:
        MAGAT_SPINE_AVAILABLE = False
        print("Warning: MAGAT spine analysis not available, using simplified method")
    
    # Initialize spine tracking buffers for MAGAT analysis
    # Sample spine points from empirical data if available
    spine_points_buffer = []  # Buffer of recent spine configurations for MAGAT analysis
    max_spine_buffer_size = 50  # Keep last N frames for spineTheta computation
    
    # REQUIRE real spine points from empirical data - no fallbacks
    if empirical_data is None:
        raise ValueError("empirical_data is required - must contain spine points from H5 files")
    
    spine_cols_x = [col for col in empirical_data.columns if col.startswith('spine_x_')]
    spine_cols_y = [col for col in empirical_data.columns if col.startswith('spine_y_')]
    
    if len(spine_cols_x) == 0 or len(spine_cols_y) == 0 or len(spine_cols_x) != len(spine_cols_y):
        raise ValueError(f"empirical_data must contain spine point columns (spine_x_0, spine_y_0, etc.). "
                        f"Found {len(spine_cols_x)} x-columns and {len(spine_cols_y)} y-columns")
    
    has_empirical_spines = True
    n_spine_points = len(spine_cols_x)
    
    if len(empirical_data) == 0:
        raise ValueError("empirical_data is empty - cannot sample spine points")
    
    # Initialize spine curve energy and spineTheta
    current_spine_curve_energy = 0.0
    current_spine_theta = 0.0
    
    t = 0.0
    last_event_time = 0.0
    prev_stimulus_intensity = 0.0
    last_progress_time = 0.0
    wall_clock_start = time.time()
    
    # Pause state tracking
    is_paused = False
    pause_start_time = None
    
    while t < max_time:
        # Get stimulus intensity at current time
        if callable(stimulus_schedule):
            stimulus_intensity = stimulus_schedule(t)
        elif isinstance(stimulus_schedule, dict):
            stimulus_intensity = np.interp(t, stimulus_schedule['times'], 
                                           stimulus_schedule['intensities'])
        else:
            stimulus_intensity = 0.0
        
        # Detect stimulus onsets (transitions from low to high)
        # Threshold: 50% of max intensity (assuming normalized 0-1 scale)
        if prev_stimulus_intensity < 0.5 and stimulus_intensity >= 0.5:
            # Stimulus onset detected - add to onset history
            stimulus_onsets.append((t, stimulus_intensity))
        
        # Prune stimulus onset history: keep only onsets within buffer window
        # Need to keep onsets for at least max_history_duration to cover full analysis window
        # This ensures we can look back t_min seconds and forward t_max seconds
        stimulus_onsets = [(t_onset, i_onset) for t_onset, i_onset in stimulus_onsets 
                          if (t - t_onset) <= max_history_duration]
        
        # Extract kernel features using configurable analysis window
        # Only onsets within [-3, +8] window relative to current time contribute
        kernel_features = extract_stimulus_kernel_features_at_time(
            t, stimulus_onsets, stimulus_intensity, config
        )
        
        # Normalize speed and heading for feature vector
        speed_normalized = (current_speed - speed_mean) / (speed_std + 1e-6)
        heading_sin = np.sin(heading)
        heading_cos = np.cos(heading)
        
        # Get spine points from empirical data - REQUIRED, no fallbacks
        # Sample spine shape from empirical data based on similar speed/heading
        # Compute similarity scores (prefer similar speed and heading)
        speed_diff = np.abs(empirical_data['speed'].values - current_speed)
        heading_diff = np.abs(np.sin(empirical_data['heading'].values - heading))
        
        # Weighted combination (speed more important)
        similarity = 1.0 / (1.0 + speed_diff * 10.0 + heading_diff * 5.0)
        
        # Sample with probability proportional to similarity
        probs = similarity / np.sum(similarity)
        sample_idx = np.random.choice(len(empirical_data), p=probs)
        
        # Extract spine points from sampled row
        sampled_row = empirical_data.iloc[sample_idx]
        spine_points = np.zeros((n_spine_points, 2))
        for i in range(n_spine_points):
            col_x = f'spine_x_{i}'
            col_y = f'spine_y_{i}'
            if col_x not in sampled_row or col_y not in sampled_row:
                raise ValueError(f"Missing spine point columns {col_x} or {col_y} in empirical data")
            spine_points[i, 0] = sampled_row[col_x]
            spine_points[i, 1] = sampled_row[col_y]
        
        # Transform spine to current position and heading
        # Get original spine center (use mid x,y from sampled row)
        orig_mid_x = sampled_row['x']
        orig_mid_y = sampled_row['y']
        orig_heading = sampled_row['heading']
        
        # Translate to origin
        spine_points[:, 0] -= orig_mid_x
        spine_points[:, 1] -= orig_mid_y
        
        # Rotate to align with current heading
        heading_rotation = heading - orig_heading
        cos_rot = np.cos(heading_rotation)
        sin_rot = np.sin(heading_rotation)
        spine_points_rotated = np.zeros_like(spine_points)
        spine_points_rotated[:, 0] = spine_points[:, 0] * cos_rot - spine_points[:, 1] * sin_rot
        spine_points_rotated[:, 1] = spine_points[:, 0] * sin_rot + spine_points[:, 1] * cos_rot
        
        # Translate to current position
        spine_points = spine_points_rotated
        spine_points[:, 0] += x
        spine_points[:, 1] += y
        
        # Add to buffer (keep recent frames for temporal smoothing)
        spine_points_buffer.append(spine_points.copy())
        if len(spine_points_buffer) > max_spine_buffer_size:
            spine_points_buffer.pop(0)
        
        # Compute MAGAT spineTheta and curve energy
        if MAGAT_SPINE_AVAILABLE and len(spine_points_buffer) >= 3:
            try:
                # Convert buffer to frame-by-frame format: (n_frames, n_spine_points, 2)
                spine_array = np.array(spine_points_buffer)  # (buffer_size, n_spine_points, 2)
                
                # Compute spineTheta for current frame (last in buffer)
                # MAGAT computes spineTheta from optimal split point
                current_spine_theta = calculate_spine_theta_magat(spine_array[-1:])[0]  # Single frame
                
                # Compute spine curve energy from spineTheta² (MAGAT method)
                current_spine_curve_energy = calculate_spine_curve_energy_magat(
                    np.array([current_spine_theta]), spine_curv=None
                )[0]
                
            except Exception as e:
                # Fallback to simple approximation if MAGAT computation fails
                heading_change_rate = abs(np.sin(heading - (headings[-1] if len(headings) > 0 else heading))) / dt if dt > 0 else 0.0
                current_spine_theta = heading_change_rate
                current_spine_curve_energy = heading_change_rate ** 2
        else:
            # Fallback: simple approximation from heading change
            heading_change_rate = abs(np.sin(heading - (headings[-1] if len(headings) > 0 else heading))) / dt if dt > 0 else 0.0
            current_spine_theta = heading_change_rate
            current_spine_curve_energy = heading_change_rate ** 2
        
        # Prepare features for all models
        features = {
            'kernel_features': kernel_features,
            'n_basis': len(kernel_features),
            'speed_normalized': speed_normalized,
            'heading_sin': heading_sin,
            'heading_cos': heading_cos,
            'spine_curve_energy': current_spine_curve_energy,
            'spine_energy_mean': spine_energy_mean,
            'spine_energy_std': spine_energy_std
        }
        
        # Compute hazard rates for all event types
        hazard_rates = {}
        # Use reorientation instead of turn for biologically plausible rates
        # Reorientation models produce ~0.4-11 turns/min vs turn models which produce 60+ turns/min
        event_types = ['reorientation', 'pause', 'heading_reversal']  # Prefer reorientation over turn
        
        for event_type in event_types:
            if event_type in models_dict and models_dict[event_type] is not None:
                hazard_rates[event_type] = compute_hazard_rate(
                    models_dict[event_type], features, bin_width
                )
            else:
                hazard_rates[event_type] = 0.0
        
        # Sample events independently from each hazard model
        # Each event type can occur independently (Poisson processes)
        for event_type in event_types:
            if hazard_rates[event_type] > 0:
                event_prob = hazard_rates[event_type] * dt
                if np.random.random() < event_prob:
                    events.append((event_type, t))
                    last_event_time = t
                    
                    # Update state based on event type
                    if event_type == 'turn' or event_type == 'reorientation':
                        # Turn/Reorientation: change heading
                        # Reorientations are larger than simple turns
                        if event_type == 'reorientation':
                            turn_angle = np.random.normal(0, np.pi/2)  # ~90 deg std for reorientations
                        else:
                            turn_angle = np.random.normal(0, np.pi/3)  # ~60 deg std for simple turns
                        heading = (heading + turn_angle) % (2 * np.pi)
                        
                    elif event_type == 'pause':
                        # Pause: set speed to very low
                        is_paused = True
                        pause_start_time = t
                        current_speed = 0.001  # Very slow
                        
                    elif event_type == 'heading_reversal':
                        # Heading reversal: large heading change (~180 degrees)
                        # NOTE: This is different from "reverse crawling" (Klein method)
                        # Reverse crawling is continuous backward movement, this is a discrete turn event
                        reversal_angle = np.random.normal(np.pi, np.pi/6)  # ~180 deg ± 30 deg
                        heading = (heading + reversal_angle) % (2 * np.pi)
        
        # Handle pause state: resume after some time or based on speed model
        if is_paused:
            # Resume from pause after random duration (or if pause hazard is low)
            pause_duration = t - pause_start_time if pause_start_time else 0.0
            # Simple pause duration model: resume when pause hazard drops or after min duration
            if pause_duration > 0.2 and hazard_rates.get('pause', 1.0) < 0.1:
                is_paused = False
                pause_start_time = None
                # Resume with sampled speed
                if speed_params is not None:
                    current_speed = sample_speed_from_params(speed_params, 1)[0]
                else:
                    current_speed = 0.01
        
        # Update position (forward Euler integration)
        # Only move if not paused
        if not is_paused:
            dx = current_speed * dt * np.cos(heading)
            dy = current_speed * dt * np.sin(heading)
            x += dx
            y += dy
            
            # Update speed (add small random variation)
            speed_change = np.random.normal(0, 0.001)
            current_speed = max(0.001, current_speed + speed_change)
            
            # Sample new speed occasionally (to match empirical distribution)
            if np.random.random() < 0.01:  # 1% chance per timestep
                if speed_params is not None:
                    current_speed = sample_speed_from_params(speed_params, 1)[0]
        
        # Record state
        times.append(t)
        positions_x.append(x)
        positions_y.append(y)
        headings.append(heading)
        speeds.append(current_speed)
        stimulus_intensities_recorded.append(stimulus_intensity)
        spine_curve_energy_track.append(current_spine_curve_energy)
        spine_theta_track.append(current_spine_theta)
        
        # Progress tracking: print every 10 seconds of simulation time
        if t - last_progress_time >= 10.0:
            elapsed_wall = time.time() - wall_clock_start
            reorientation_count = len([e for e in events if e[0] == 'reorientation'])
            pause_count = len([e for e in events if e[0] == 'pause'])
            heading_reversal_count = len([e for e in events if e[0] == 'heading_reversal'])
            print(f"      t={t:.1f}s/{max_time:.1f}s | Reorientations: {reorientation_count} | Pauses: {pause_count} | Heading Reversals: {heading_reversal_count} | Speed: {current_speed:.4f} | Elapsed: {elapsed_wall:.1f}s")
            sys.stdout.flush()
            last_progress_time = t
        
        prev_stimulus_intensity = stimulus_intensity
        t += dt
    
    # Create trajectory DataFrame with MAGAT-compatible fields
    trajectory = pd.DataFrame({
        'time': times,
        'x': positions_x,
        'y': positions_y,
        'heading': headings,
        'speed': speeds,
        'stimulus_intensity': stimulus_intensities_recorded,
        'spine_curve_energy': spine_curve_energy_track,
        'spineTheta_magat': spine_theta_track  # MAGAT body bend angle
    })
    
    # Compute smoothed spineTheta (sspineTheta) using MAGAT's lowpass filter
    if MAGAT_SPINE_AVAILABLE and len(spine_theta_track) > 1:
        try:
            # MAGAT smooths spineTheta: smoothTime/interpTime (typically ~0.1s / 0.1s = 1 sample)
            smooth_time = 0.1  # seconds
            sigma_samples = smooth_time / dt
            trajectory['sspineTheta_magat'] = lowpass1d(np.array(spine_theta_track), sigma_samples)
        except Exception:
            trajectory['sspineTheta_magat'] = spine_theta_track  # Fallback: no smoothing
    else:
        trajectory['sspineTheta_magat'] = spine_theta_track
    
    # Add event markers (MAGAT-compatible)
    trajectory['is_turn'] = False  # Simple turn detection (backwards compatibility)
    trajectory['is_reorientation'] = False  # MAGAT reorientation (gaps between runs)
    trajectory['is_pause'] = False
    trajectory['is_heading_reversal'] = False
    trajectory['is_run'] = False  # MAGAT run detection (periods of forward movement)
    
    # Mark events in trajectory
    for event_type, event_time in events:
        # Find closest time index
        idx = np.argmin(np.abs(trajectory['time'].values - event_time))
        if event_type == 'turn':
            trajectory.loc[idx, 'is_turn'] = True
        elif event_type == 'reorientation':
            trajectory.loc[idx, 'is_reorientation'] = True  # MAGAT reorientation
            trajectory.loc[idx, 'is_turn'] = True  # Also mark as turn for compatibility
        elif event_type == 'pause':
            trajectory.loc[idx, 'is_pause'] = True
        elif event_type == 'heading_reversal':
            trajectory.loc[idx, 'is_heading_reversal'] = True
    
    # Post-process: detect runs using MAGAT segmentation (if available)
    # Runs are periods of forward movement between reorientations
    if MAGAT_SPINE_AVAILABLE and len(trajectory) > 10:
        try:
            import sys
            from pathlib import Path
            script_dir = Path(__file__).parent
            sys.path.insert(0, str(script_dir))
            from magat_segmentation import magat_segment_track
            
            # Prepare DataFrame for MAGAT segmentation
            magat_df = pd.DataFrame({
                'time': trajectory['time'].values,
                'speed': trajectory['speed'].values,
                'curvature': np.abs(np.diff(trajectory['heading'].values, prepend=trajectory['heading'].values[0])),
                'curv': np.abs(np.diff(trajectory['heading'].values, prepend=trajectory['heading'].values[0])),
                'spineTheta': trajectory['spineTheta_magat'].values,
                'sspineTheta': trajectory['sspineTheta_magat'].values,
                'heading': trajectory['heading'].values,
                'x': trajectory['x'].values,
                'y': trajectory['y'].values
            })
            
            # Add vel_dp (approximate as cos(angle between heading and velocity))
            magat_df['vel_dp'] = np.ones(len(magat_df)) * 0.707  # Approximate alignment
            
            # Run MAGAT segmentation
            frame_rate_actual = 1.0 / dt
            segmentation = magat_segment_track(magat_df, frame_rate=frame_rate_actual)
            
            # Mark runs in trajectory
            trajectory['is_run'] = segmentation['is_run']
            
            # Update reorientations to match MAGAT's detected reorientations
            # (only at start of reorientation events, not all frames)
            trajectory['is_reorientation'] = False  # Reset
            reorientation_starts = np.where(segmentation['is_reorientation'])[0]
            for idx in reorientation_starts:
                if idx < len(trajectory):
                    trajectory.loc[idx, 'is_reorientation'] = True
                    
        except Exception as e:
            # Fallback: simple run detection (speed-based)
            # Runs are periods where speed > threshold and not paused
            speed_threshold = np.percentile(trajectory['speed'].values, 25)
            trajectory['is_run'] = (trajectory['speed'] > speed_threshold) & (~trajectory['is_pause'])
    
    return trajectory, events

def create_stimulus_schedule(intensity_pct, pulse_duration, inter_pulse_interval, 
                            max_time=300.0, frame_rate=10.0):
    """
    Create stimulus schedule function for DOE condition.
    
    Uses variable pulse durations (10s, 15s, 20s, 25s, 30s, ...) from DOE table
    and maps intensity percentages to PWM values (Pulse Width Modulation).
    
    Parameters
    ----------
    intensity_pct : float
        LED intensity percentage (25, 50, 100, 200, 300, 400)
        Maps to PWM values: 25% -> 250, 50% -> 500, 100% -> 1000, etc.
    pulse_duration : float
        Pulse duration in seconds (from DOE: 10s, 15s, 20s, 25s, 30s, ...)
    inter_pulse_interval : float
        Time between pulse starts in seconds (20s, 40s, 60s)
        This is the gap between pulse end and next pulse start
    max_time : float
        Maximum time for schedule
    frame_rate : float
        Frame rate for discrete time points
    
    Returns
    -------
    schedule : dict
        Dictionary with 'times' and 'intensities' arrays
        'intensities' are PWM values (Pulse Width Modulation duty cycle)
    """
    # Use pulse duration from DOE parameter (variable: 10s, 15s, 20s, 25s, 30s, ...)
    pulse_duration_actual = pulse_duration
    
    # Map DOE intensity percentages to PWM values (Pulse Width Modulation)
    # NOTE: Observed values (83.3, 166.7, 250.0) are PWM values at the LOWER BOUNDS of perception
    # PWM values represent duty cycle: higher PWM = higher LED power
    # Assuming higher bit-depth PWM (e.g., 10-bit: 0-1023, or 12-bit: 0-4095)
    # where 250 is low percentage, representing near-threshold power
    # Extended mapping explores suprathreshold responses with higher PWM values
    intensity_mapping = {
        25: 250.0,    # Near perception threshold (PWM ≈ 250, low duty cycle)
        50: 500.0,    # Low suprathreshold (PWM ≈ 500, 2× threshold PWM)
        100: 1000.0   # Maximum tested (PWM ≈ 1000, 4× threshold PWM)
    }
    
    # Get actual LED power value for this intensity percentage
    if intensity_pct in intensity_mapping:
        led_power = intensity_mapping[intensity_pct]
    else:
        # Interpolate for other values
        sorted_pcts = sorted(intensity_mapping.keys())
        if intensity_pct < sorted_pcts[0]:
            led_power = intensity_mapping[sorted_pcts[0]]
        elif intensity_pct > sorted_pcts[-1]:
            led_power = intensity_mapping[sorted_pcts[-1]]
        else:
            # Linear interpolation
            for i in range(len(sorted_pcts) - 1):
                if sorted_pcts[i] <= intensity_pct <= sorted_pcts[i+1]:
                    pct1, power1 = sorted_pcts[i], intensity_mapping[sorted_pcts[i]]
                    pct2, power2 = sorted_pcts[i+1], intensity_mapping[sorted_pcts[i+1]]
                    led_power = power1 + (power2 - power1) * (intensity_pct - pct1) / (pct2 - pct1)
                    break
    
    times = np.arange(0, max_time, 1.0/frame_rate)
    intensities = np.zeros(len(times))
    
    # Create periodic stimulus pattern
    # Cycle = pulse (20s) + inter-pulse interval
    cycle_period = pulse_duration_actual + inter_pulse_interval
    
    for i, t in enumerate(times):
        cycle_time = t % cycle_period
        if cycle_time < pulse_duration_actual:
            intensities[i] = led_power  # Use actual LED power value
        else:
            intensities[i] = 0.0
    
    return {'times': times, 'intensities': intensities}

def _extract_stimulus_onsets(stimulus_schedule, max_time):
    """
    Extract all stimulus cycle onset times (tOn) from stimulus schedule.
    
    Parameters
    ----------
    stimulus_schedule : callable or dict
        Stimulus schedule function or dict with 'times' and 'intensities'
    max_time : float
        Maximum time to search
    
    Returns
    -------
    stimulus_onsets : list
        List of stimulus onset times (tOn) in seconds
    """
    stimulus_onsets = []
    
    if isinstance(stimulus_schedule, dict) and 'times' in stimulus_schedule:
        # Extract from dict format
        times = stimulus_schedule['times']
        intensities = stimulus_schedule['intensities']
        
        # Find rising edges (transitions from low to high)
        prev_intensity = 0.0
        threshold = np.max(intensities) * 0.5 if len(intensities) > 0 else 0.5
        
        for i, (t, intensity) in enumerate(zip(times, intensities)):
            if prev_intensity < threshold and intensity >= threshold:
                stimulus_onsets.append(t)
            prev_intensity = intensity
    
    elif callable(stimulus_schedule):
        # Sample stimulus schedule to find onsets
        dt = 0.1  # Sample at 10 Hz
        times = np.arange(0, max_time, dt)
        prev_intensity = 0.0
        
        for t in times:
            intensity = stimulus_schedule(t)
            threshold = 0.5  # Assuming normalized 0-1 scale
            
            if prev_intensity < threshold and intensity >= threshold:
                stimulus_onsets.append(t)
            prev_intensity = intensity
    
    return sorted(stimulus_onsets)

def compute_kpis(trajectory, events, stimulus_schedule=None, config=None, event_params=None):
    """
    Compute key performance indicators for simulated trajectory.
    
    Parameters
    ----------
    trajectory : DataFrame
        Simulated trajectory
    events : list
        List of (event_type, time) tuples
    stimulus_schedule : callable or dict, optional
        Stimulus schedule function or dict. Required for correct latency calculation.
    config : dict, optional
        Model configuration. Required for integration window definition.
    
    Returns
    -------
    kpis : dict
        Dictionary of KPI values
    """
    total_time = trajectory['time'].max()
    
    # Turn rate - use reorientations (MAGAT-compatible) instead of simple turns
    # NOTE: For biologically plausible rates, use reorientations (0.4-11 turns/min)
    # not simple turns (which can be 60+ turns/min)
    reorientation_events = [e for e in events if e[0] == 'reorientation']
    turn_events = [e for e in events if e[0] == 'turn']
    
    # Prefer reorientations if available, otherwise fall back to turns
    if reorientation_events:
        turn_rate = len(reorientation_events) / (total_time / 60.0) if total_time > 0 else 0.0
        turn_event_count = len(reorientation_events)
        turn_times = [e[1] for e in reorientation_events]
    else:
        # Fallback to simple turns (less biologically accurate)
        turn_rate = len(turn_events) / (total_time / 60.0) if total_time > 0 else 0.0
        turn_event_count = len(turn_events)
        turn_times = [e[1] for e in turn_events]
    
    # Latency to first turn/reorientation WITHIN integration window
    # Integration window: [-3s before tOn, +8s after tOn] for each stimulus cycle
    latency = np.nan
    
    if turn_times and stimulus_schedule is not None and config is not None:
        # Extract all stimulus cycle onsets (tOn times)
        stimulus_onsets = _extract_stimulus_onsets(stimulus_schedule, total_time)
        
        if len(stimulus_onsets) > 0:
            # Get integration window from config
            time_range = config.get('model', {}).get('temporal_kernel', {}).get('time_range', [-3.0, 8.0])
            t_min, t_max = time_range  # e.g., [-3.0, 8.0] seconds
            
            # Find first turn that falls within ANY stimulus cycle's integration window
            for turn_time in sorted(turn_times):
                for tOn in stimulus_onsets:
                    # Check if turn is within integration window for this cycle
                    window_start = tOn + t_min  # e.g., tOn - 3.0
                    window_end = tOn + t_max    # e.g., tOn + 8.0
                    
                    if window_start <= turn_time <= window_end:
                        # Turn is within integration window - compute latency relative to tOn
                        latency = turn_time - tOn
                        break  # Found first valid latency, stop searching
                
                if not np.isnan(latency):
                    break  # Found valid latency, stop searching
    elif turn_times:
        # Fallback: if no stimulus schedule provided, use old method (but warn)
        # This should not happen in normal operation
        latency = turn_times[0]
    
    # Stop fraction (time with speed < threshold, or pause events)
    # Use learned threshold if available, otherwise default
    if event_params and 'stop_speed_threshold' in event_params:
        speed_threshold = event_params['stop_speed_threshold']
    else:
        speed_threshold = 0.001  # Default fallback
    stop_time = np.sum(trajectory['speed'] < speed_threshold) * (trajectory['time'].iloc[1] - trajectory['time'].iloc[0])
    stop_fraction = stop_time / total_time if total_time > 0 else 0.0
    
    # Pause-related KPIs
    # Use learned threshold to detect pauses from speed (MAGAT approach)
    # This matches how empirical data pauses are detected
    if event_params and 'pause_speed_threshold' in event_params:
        pause_speed_threshold = event_params['pause_speed_threshold']
        pause_min_duration = event_params.get('pause_min_duration', 0.2)
    else:
        pause_speed_threshold = 0.001  # Default fallback
        pause_min_duration = 0.2
    
    # Detect pause periods from speed trajectory (like MAGAT)
    is_pause = trajectory['speed'] < pause_speed_threshold
    
    # Find pause start/end events (transitions)
    pause_starts = []
    pause_ends = []
    in_pause = False
    
    for i in range(len(is_pause)):
        if is_pause.iloc[i] and not in_pause:
            pause_starts.append(i)
            in_pause = True
        elif not is_pause.iloc[i] and in_pause:
            pause_ends.append(i)
            in_pause = False
    
    # Handle pause that extends to end
    if in_pause:
        pause_ends.append(len(is_pause))
    
    # Filter pauses by minimum duration
    dt = trajectory['time'].iloc[1] - trajectory['time'].iloc[0] if len(trajectory) > 1 else 0.1
    valid_pause_events = 0
    
    for start_idx, end_idx in zip(pause_starts, pause_ends):
        pause_duration = (end_idx - start_idx) * dt
        if pause_duration >= pause_min_duration:
            valid_pause_events += 1
    
    # Calculate pause rate (events per minute)
    pause_rate = valid_pause_events / (total_time / 60.0) if total_time > 0 else 0.0
    
    # Heading reversal-related KPIs (discrete turn events, not reverse crawling)
    # Use learned threshold if available
    if event_params and 'reversal_angle_threshold' in event_params:
        reversal_threshold = event_params['reversal_angle_threshold']
    else:
        reversal_threshold = np.pi / 2  # Default: 90 degrees
    
    heading_reversal_events = [e for e in events if e[0] == 'heading_reversal']
    heading_reversal_rate = len(heading_reversal_events) / (total_time / 60.0) if total_time > 0 else 0.0
    
    # Path tortuosity
    path_length = np.sum(np.sqrt(np.diff(trajectory['x'])**2 + np.diff(trajectory['y'])**2))
    euclidean_distance = np.sqrt((trajectory['x'].iloc[-1] - trajectory['x'].iloc[0])**2 + 
                                 (trajectory['y'].iloc[-1] - trajectory['y'].iloc[0])**2)
    tortuosity = euclidean_distance / path_length if path_length > 0 else 0.0
    
    # Spatial dispersal (mean distance from start)
    start_x, start_y = trajectory['x'].iloc[0], trajectory['y'].iloc[0]
    distances = np.sqrt((trajectory['x'] - start_x)**2 + (trajectory['y'] - start_y)**2)
    dispersal = np.mean(distances)
    
    # Mean spine curve energy
    mean_spine_energy = trajectory['spine_curve_energy'].mean() if 'spine_curve_energy' in trajectory.columns else np.nan
    
    return {
        'turn_rate': turn_rate,
        'latency': latency,
        'stop_fraction': stop_fraction,
        'pause_rate': pause_rate,
        'reversal_rate': heading_reversal_rate,  # Use 'reversal_rate' for consistency with export mapping
        'heading_reversal_rate': heading_reversal_rate,  # Keep for backward compatibility
        'tortuosity': tortuosity,
        'dispersal': dispersal,
        'mean_spine_curve_energy': mean_spine_energy,
        'total_time': total_time,
        'total_turns': turn_event_count,  # Reorientations if available, else simple turns
        'total_pauses': valid_pause_events,  # Pause events detected from speed trajectory
        'total_heading_reversals': len(heading_reversal_events),
        'path_length': path_length,
        'euclidean_distance': euclidean_distance
    }

def load_event_models(models_dir):
    """
    Load all event models once (bottleneck fix - don't reload every condition).
    
    Parameters
    ----------
    models_dir : Path or str
        Directory containing fitted model pickle files
    
    Returns
    -------
    models_dict : dict
        Dictionary with 'reorientation', 'pause', 'heading_reversal' keys mapping to models
    """
    models_dir = Path(models_dir)
    models_dict = {}
    
    # Use reorientation instead of turn for biologically plausible rates
    # Reorientation models produce ~0.4-11 turns/min vs turn models which produce 60+ turns/min
    event_types = ['reorientation', 'pause', 'heading_reversal']  # Prefer reorientation over turn
    for event_type in event_types:
        # Handle backward compatibility: check for old "reversal" model name
        if event_type == 'heading_reversal':
            # Try new name first
            model_path = models_dir / 'heading_reversal_full_model.pkl'
            if not model_path.exists():
                # Fallback to old "reversal" name for backward compatibility
                model_path = models_dir / 'reversal_full_model.pkl'
        else:
            model_path = models_dir / f'{event_type}_full_model.pkl'
        
        if model_path.exists():
            try:
                models_dict[event_type] = load_fitted_model(model_path)
            except Exception as e:
                print(f"    Warning: Could not load {event_type} model: {e}")
                models_dict[event_type] = None
        else:
            # Try baseline model as fallback
            if event_type == 'heading_reversal':
                baseline_path = models_dir / 'heading_reversal_baseline_model.pkl'
                if not baseline_path.exists():
                    baseline_path = models_dir / 'reversal_baseline_model.pkl'
            else:
                baseline_path = models_dir / f'{event_type}_baseline_model.pkl'
            
            if baseline_path.exists():
                try:
                    models_dict[event_type] = load_fitted_model(baseline_path)
                except Exception as e:
                    models_dict[event_type] = None
            else:
                models_dict[event_type] = None
    
    if all(m is None for m in models_dict.values()):
        raise ValueError("No models found! Cannot simulate.")
    
    return models_dict

def simulate_doe_condition(models_dict, condition, config, empirical_data=None,
                          n_replications=30, max_time=300.0, random_seed_base=42,
                          continue_on_error=False, event_params=None):
    """
    Simulate multiple replications for a single DOE condition using multi-event models.
    
    NOTE: Models are pre-loaded (passed in) to avoid reloading bottleneck.
    
    Parameters
    ----------
    models_dict : dict
        Pre-loaded models dictionary (from load_event_models)
    condition : dict
        DOE condition with intensity_pct, pulse_duration, inter_pulse_interval
    config : dict
        Model configuration
    empirical_data : DataFrame, optional
        Empirical data for speed/position sampling
    n_replications : int
        Number of replications (larvae) to simulate
    max_time : float
        Maximum simulation time
    random_seed_base : int
        Base random seed
    
    Returns
    -------
    results : DataFrame
        Results with one row per replication
    """
    
    # Create stimulus schedule
    stimulus_schedule = create_stimulus_schedule(
        condition['intensity_pct'],
        condition['pulse_duration_s'],
        condition['inter_pulse_interval_s'],
        max_time=max_time
    )
    
    all_results = []
    condition_start_time = time.time()
    error_count = 0
    
    print(f"  Simulating condition {condition['condition_id']}...")
    print(f"    Intensity: {condition['intensity_pct']}%, "
          f"Pulse: {condition['pulse_duration_s']}s, "
          f"Interval: {condition['inter_pulse_interval_s']}s")
    
    for rep in range(n_replications):
        rep_start_time = time.time()
        random_seed = int(random_seed_base + condition['condition_id'] * 1000 + rep)
        
        try:
            # Simulate trajectory
            trajectory, events = simulate_single_trajectory(
                models_dict, stimulus_schedule, config,
                empirical_data=empirical_data,
                max_time=max_time,
                random_seed=random_seed
            )
            
            # Compute KPIs (pass stimulus schedule and config for correct latency calculation)
            kpis = compute_kpis(trajectory, events, stimulus_schedule=stimulus_schedule, config=config, event_params=event_params)
            
            # Store results
            result = {
                'condition_id': condition['condition_id'],
                'replication': rep,
                'intensity_pct': condition['intensity_pct'],
                'pulse_duration_s': condition['pulse_duration_s'],
                'inter_pulse_interval_s': condition['inter_pulse_interval_s'],
                **kpis
            }
            
            all_results.append(result)
            
            # Print replication completion progress
            rep_elapsed = time.time() - rep_start_time
            print(f"    ✓ Replication {rep + 1}/{n_replications} completed ({rep_elapsed:.1f}s)")
            sys.stdout.flush()
            
        except Exception as e:
            error_count += 1
            rep_elapsed = time.time() - rep_start_time
            print(f"    ✗ Replication {rep + 1}/{n_replications} FAILED ({rep_elapsed:.1f}s): {e}")
            sys.stdout.flush()
            
            if not continue_on_error:
                raise
            
            # Store error record
            error_result = {
                'condition_id': condition['condition_id'],
                'replication': rep,
                'intensity_pct': condition['intensity_pct'],
                'pulse_duration_s': condition['pulse_duration_s'],
                'inter_pulse_interval_s': condition['inter_pulse_interval_s'],
                'error': str(e),
                'turn_rate': np.nan,
                'latency': np.nan,
                'stop_fraction': np.nan,
                'pause_rate': np.nan,
                'reversal_rate': np.nan,
                'heading_reversal_rate': np.nan,  # Keep for backward compatibility
                'tortuosity': np.nan,
                'dispersal': np.nan,
                'mean_spine_curve_energy': np.nan,
                'total_time': np.nan,
                'total_turns': np.nan,
                'total_pauses': np.nan,
                'total_heading_reversals': np.nan,
                'path_length': np.nan,
                'euclidean_distance': np.nan
            }
            all_results.append(error_result)
    
    # Print condition summary
    condition_elapsed = time.time() - condition_start_time
    success_count = n_replications - error_count
    print(f"  ✓ Condition {condition['condition_id']} complete: {success_count}/{n_replications} successful in {condition_elapsed:.1f}s")
    if error_count > 0:
        print(f"    ⚠ {error_count} replication(s) failed")
    sys.stdout.flush()
    
    results_df = pd.DataFrame(all_results)
    return results_df

def main():
    """Main simulation pipeline."""
    parser = argparse.ArgumentParser(description='Simulate larval trajectories using multi-event models')
    parser.add_argument('--models-dir', type=str, 
                       default='output/fitted_models',
                       help='Directory containing fitted model pickle files (reorientation_full_model.pkl, pause_full_model.pkl, reversal_full_model.pkl or heading_reversal_full_model.pkl)')
    parser.add_argument('--doe-table', type=str,
                       default='config/doe_table.csv',
                       help='Path to DOE table CSV')
    parser.add_argument('--empirical-data', type=str, 
                       default='data/engineered/GMR61_tier2_complete_trajectories.csv',
                       help='Path to empirical trajectory CSV (must include spine points)')
    parser.add_argument('--condition-id', type=int, default=None,
                       help='Simulate single condition (if None, simulates all)')
    parser.add_argument('--n-replications', type=int, default=30,
                       help='Number of replications per condition')
    parser.add_argument('--max-time', type=float, default=300.0,
                       help='Maximum simulation time (seconds)')
    parser.add_argument('--output-dir', type=str, default='output/simulation_results',
                       help='Output directory for results')
    parser.add_argument('--config', type=str, default='config/model_config.json',
                       help='Path to model config JSON')
    parser.add_argument('--force-full-doe', action='store_true',
                       help='Enable full DOE execution (requires explicit flag)')
    parser.add_argument('--checkpoint-interval', type=int, default=5,
                       help='Save checkpoint every N conditions (default: 5)')
    parser.add_argument('--continue-on-error', action='store_true',
                       help='Continue simulation even if individual conditions fail')
    parser.add_argument('--max-errors', type=int, default=10,
                       help='Maximum errors before stopping (default: 10)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint file (path to checkpoint CSV)')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Load DOE table (handle unquoted commas in description column)
    # Use manual CSV parsing to handle description field with commas
    import csv
    rows = []
    with open(args.doe_table, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        expected_cols = len(header)
        for row in reader:
            # Handle description column with commas by joining extra fields
            if len(row) > expected_cols:
                # Join extra fields into description (last column)
                row = row[:expected_cols-1] + [', '.join(row[expected_cols-1:])]
            elif len(row) < expected_cols:
                # Pad with empty strings if too short
                row = row + [''] * (expected_cols - len(row))
            rows.append(row)
    
    doe_df = pd.DataFrame(rows, columns=header)
    doe_df['condition_id'] = pd.to_numeric(doe_df['condition_id'], errors='coerce')
    doe_df['intensity_pct'] = pd.to_numeric(doe_df['intensity_pct'], errors='coerce')
    doe_df['pulse_duration_s'] = pd.to_numeric(doe_df['pulse_duration_s'], errors='coerce')
    doe_df['inter_pulse_interval_s'] = pd.to_numeric(doe_df['inter_pulse_interval_s'], errors='coerce')
    
    print(f"Loaded DOE table with {len(doe_df)} conditions")
    print(f"  Condition IDs: {int(doe_df['condition_id'].min())} to {int(doe_df['condition_id'].max())}")
    
    # Load empirical data (required for spine points)
    empirical_data = None
    if args.empirical_data:
        print(f"Loading empirical data from {args.empirical_data}...")
        empirical_data = pd.read_csv(args.empirical_data)
        print(f"  Loaded {len(empirical_data)} trajectory points")
        
        # Validate that spine points are present
        spine_x_cols = [c for c in empirical_data.columns if c.startswith('spine_x_')]
        spine_y_cols = [c for c in empirical_data.columns if c.startswith('spine_y_')]
        if len(spine_x_cols) == 0 or len(spine_y_cols) == 0:
            print(f"  ERROR: Empirical data must contain spine point columns (spine_x_0, spine_y_0, etc.)")
            print(f"    Found {len(spine_x_cols)} x-columns and {len(spine_y_cols)} y-columns")
            sys.exit(1)
        print(f"  ✓ Found {len(spine_x_cols)} spine point pairs")
        
        # Check required columns
        required_cols = ['speed', 'heading', 'x', 'y']
        missing_cols = [c for c in required_cols if c not in empirical_data.columns]
        if missing_cols:
            print(f"  ERROR: Missing required columns: {missing_cols}")
            sys.exit(1)
        print(f"  ✓ All required columns present")
    
    # Select conditions to simulate
    if args.condition_id is not None:
        conditions = doe_df[doe_df['condition_id'] == args.condition_id]
        if len(conditions) == 0:
            print(f"ERROR: Condition {args.condition_id} not found in DOE table")
            print(f"  Available condition IDs: {sorted(doe_df['condition_id'].unique())}")
            sys.exit(1)
    else:
        conditions = doe_df
    
    print(f"Simulating {len(conditions)} condition(s) with {args.n_replications} replications each")
    print(f"Models directory: {args.models_dir}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if resuming from checkpoint
    all_results = []
    completed_condition_ids = set()
    checkpoint_file = output_dir / 'checkpoint.csv'
    overall_start_time = time.time()
    
    if args.resume:
        if Path(args.resume).exists():
            print(f"Resuming from checkpoint: {args.resume}")
            checkpoint_df = pd.read_csv(args.resume)
            all_results.append(checkpoint_df)
            completed_condition_ids = set(checkpoint_df['condition_id'].unique())
            print(f"  Found {len(completed_condition_ids)} completed conditions")
        else:
            print(f"ERROR: Checkpoint file not found: {args.resume}")
            sys.exit(1)
    
    # Load models once (bottleneck fix)
    print("\nLoading models...")
    models_dict = load_event_models(args.models_dir)
    print(f"✓ Loaded {len(models_dict)} models")
    
    # Load event parameters if available
    event_params = None
    # Try to infer experiment_id from empirical_data path or use default
    experiment_id = 'GMR61_tier2_complete'  # Default
    if args.empirical_data:
        # Extract experiment ID from path if possible
        emp_path = Path(args.empirical_data)
        if 'GMR61' in emp_path.name:
            experiment_id = 'GMR61_tier2_complete'
    
    event_params_file = Path('data/simulation') / f'{experiment_id}_event_parameters.json'
    if event_params_file.exists():
        print(f"\nLoading event parameters from {event_params_file}...")
        import json
        with open(event_params_file) as f:
            event_params = json.load(f)
        print(f"  ✓ Pause threshold: {event_params.get('pause_speed_threshold', 'N/A'):.6f}")
        print(f"  ✓ Reversal threshold: {event_params.get('reversal_angle_threshold_degrees', 'N/A'):.1f}°")
    else:
        print(f"\n⚠️  Event parameters not found at {event_params_file}")
        print(f"  Using default thresholds (may not match empirical data)")
        print(f"  Run prepare_simulation_dataset.py to learn parameters")
    
    # Determine if running full DOE
    run_full_doe = args.condition_id is None
    
    # Auto-launch monitoring window for full DOE
    if run_full_doe and args.force_full_doe:
        print("\n" + "="*70)
        print("LAUNCHING MONITORING WINDOW")
        print("="*70)
        try:
            import subprocess
            import os
            # Path is already imported at top of file
            
            # Get absolute path to monitor script
            script_dir = Path(__file__).parent
            monitor_script = script_dir / 'monitor_doe.py'
            
            # Get absolute path to project root (for working directory)
            project_root = Path(__file__).parent.parent
            
            # Launch in new Terminal window (macOS)
            if os.name == 'posix':  # macOS/Linux
                # Use osascript to open new Terminal window
                # Use absolute path to ensure it works
                monitor_cmd = f"cd '{project_root}' && python3 scripts/monitor_doe.py"
                applescript = f'''
                tell application "Terminal"
                    activate
                    do script "{monitor_cmd}"
                end tell
                '''
                try:
                    result = subprocess.run(
                        ['osascript', '-e', applescript],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if result.returncode == 0:
                        print("✓ Monitoring window launched in new Terminal")
                        print("  Monitor will update automatically every 2 seconds")
                    else:
                        print(f"⚠️  Could not auto-launch monitor (exit code {result.returncode})")
                        if result.stderr:
                            print(f"   Error: {result.stderr[:200]}")
                        print(f"   Please manually run: python3 {monitor_script}")
                except subprocess.TimeoutExpired:
                    print("⚠️  Monitor launch timed out")
                    print(f"   Please manually run: python3 {monitor_script}")
                except Exception as e:
                    print(f"⚠️  Error launching monitor: {e}")
                    print(f"   Please manually run: python3 {monitor_script}")
            else:
                print("⚠️  Auto-launch not supported on this OS")
                print(f"  Please manually run: python3 {monitor_script}")
        except Exception as e:
            print(f"⚠️  Could not auto-launch monitor: {e}")
            print(f"  Please manually run: python3 scripts/monitor_doe.py")
    
    if not run_full_doe:
        # Single condition mode (no --force-full-doe needed)
        print(f"\nRunning single condition {args.condition_id}...")
        condition = conditions.iloc[0]
        condition_id = condition['condition_id']
        condition_dict = condition.to_dict()
        
        results = simulate_doe_condition(
            models_dict,
            condition_dict,
            config,
            empirical_data=empirical_data,
            n_replications=args.n_replications,
            max_time=args.max_time,
            continue_on_error=args.continue_on_error,
            event_params=event_params
        )
        
        all_results.append(results)
        
        # Combine results and save
        combined_results = pd.concat(all_results, ignore_index=True)
        results_file = output_dir / 'all_results.csv'
        combined_results.to_csv(results_file, index=False)
        
        # Print final summary
        total_elapsed = time.time() - overall_start_time
        total_replications = len(combined_results)
        total_successful_reps = len(combined_results[combined_results['turn_rate'].notna()])
        avg_time_per_rep = total_elapsed / total_successful_reps if total_successful_reps > 0 else 0
        
        print(f"\n✓ Saved {total_replications} simulation results to {results_file}")
        print("\nSimulation Summary:")
        print(f"  Total replications: {total_replications}")
        print(f"  Successful replications: {total_successful_reps}")
        if 'turn_rate' in combined_results.columns:
            valid_turn_rates = combined_results['turn_rate'].dropna()
            if len(valid_turn_rates) > 0:
                print(f"  Mean turn rate: {valid_turn_rates.mean():.2f} turns/min")
        print(f"  Conditions simulated: {combined_results['condition_id'].nunique()}")
        print(f"  Total elapsed time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
        if total_successful_reps > 0:
            print(f"  Average time per replication: {avg_time_per_rep:.2f}s")
        return
    
    # Full DOE mode
    if run_full_doe:
        if not args.force_full_doe:
            print("\n" + "="*70)
            print("FULL DOE EXECUTION")
            print("="*70)
            print(f"About to simulate {len(conditions)} conditions × {args.n_replications} replications")
            print(f"Estimated time: ~{len(conditions) * args.n_replications * 0.2 / 60:.1f} minutes")
            print("\n⚠️  This will generate a large dataset.")
            print("Use --force-full-doe to enable full DOE execution.")
            print("="*70 + "\n")
            sys.exit(0)
        
        print("\n" + "="*70)
        print("RUNNING FULL DOE")
        print("="*70)
        print(f"Conditions: {len(conditions)}")
        print(f"Replications per condition: {args.n_replications}")
        print(f"Checkpoint interval: {args.checkpoint_interval} conditions")
        print(f"Continue on error: {args.continue_on_error}")
        print(f"Max errors: {args.max_errors}")
        print("="*70 + "\n")
    
    total_errors = 0
    total_successful_reps = 0
    
    # Process conditions
    conditions_to_run = conditions[~conditions['condition_id'].isin(completed_condition_ids)]
    
    if len(conditions_to_run) == 0:
        print("All conditions already completed in checkpoint.")
        combined_results = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    else:
        print(f"Running {len(conditions_to_run)} condition(s)...")
        
        for idx, (_, condition_row) in enumerate(conditions_to_run.iterrows()):
            condition_id = condition_row['condition_id']
            condition_dict = condition_row.to_dict()
            
            try:
                # Show progress
                completed = len(completed_condition_ids) + idx
                total = len(conditions)
                progress_pct = (completed / total) * 100
                
                # Estimate ETA
                elapsed = time.time() - overall_start_time
                if completed > 0:
                    avg_time_per_condition = elapsed / completed
                    remaining = total - completed
                    eta_seconds = avg_time_per_condition * remaining
                    eta_minutes = eta_seconds / 60
                    print(f"\n[{progress_pct:.1f}%] Condition {condition_id}/{total} "
                          f"(ETA: {eta_minutes:.1f} min)")
                else:
                    print(f"\n[{progress_pct:.1f}%] Condition {condition_id}/{total}")
                
                # Simulate condition
                results = simulate_doe_condition(
                    models_dict,
                    condition_dict,
                    config,
                    empirical_data=empirical_data,
                    n_replications=args.n_replications,
                    event_params=event_params,
                    max_time=args.max_time,
                    continue_on_error=args.continue_on_error
                )
                
                all_results.append(results)
                completed_condition_ids.add(condition_id)
                total_successful_reps += len(results[results['turn_rate'].notna()])
                
                # Save checkpoint periodically AND immediately if resuming (to avoid redoing work)
                should_save = (idx + 1) % args.checkpoint_interval == 0 or (idx + 1) == len(conditions_to_run)
                if args.resume:
                    # When resuming, save after EVERY condition to avoid losing progress
                    should_save = True
                
                if should_save:
                    combined_results = pd.concat(all_results, ignore_index=True)
                    combined_results.to_csv(checkpoint_file, index=False)
                    print(f"  💾 Checkpoint saved: {len(combined_results)} total replications")
                
            except Exception as e:
                total_errors += 1
                print(f"  ✗ Condition {condition_id} FAILED: {e}")
                sys.stdout.flush()
                
                if args.continue_on_error and total_errors < args.max_errors:
                    print(f"  ⚠ Continuing (error {total_errors}/{args.max_errors})")
                    continue
                else:
                    print(f"  ❌ Stopping due to error limit ({total_errors}/{args.max_errors})")
                    if all_results:
                        combined_results = pd.concat(all_results, ignore_index=True)
                        combined_results.to_csv(checkpoint_file, index=False)
                        print(f"  💾 Checkpoint saved before exit")
                    raise
    
    # Combine all results
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
    else:
        combined_results = pd.DataFrame()
    
    # Save final results
    results_file = output_dir / 'all_results.csv'
    combined_results.to_csv(results_file, index=False)
    
    # Print final summary
    total_elapsed = time.time() - overall_start_time
    total_replications = len(combined_results)
    avg_time_per_rep = total_elapsed / total_successful_reps if total_successful_reps > 0 else 0
    
    print("\n" + "="*70)
    print("SIMULATION COMPLETE")
    print("="*70)
    print(f"✓ Saved {total_replications} simulation results to {results_file}")
    print(f"✓ Checkpoint saved to {checkpoint_file}")
    print("\nSimulation Summary:")
    print(f"  Total replications: {total_replications}")
    print(f"  Successful replications: {total_successful_reps}")
    if total_errors > 0:
        print(f"  Errors: {total_errors}")
    if 'turn_rate' in combined_results.columns:
        valid_turn_rates = combined_results['turn_rate'].dropna()
        if len(valid_turn_rates) > 0:
            print(f"  Mean turn rate: {valid_turn_rates.mean():.2f} turns/min")
    print(f"  Conditions simulated: {combined_results['condition_id'].nunique() if len(combined_results) > 0 else 0}")
    print(f"  Total elapsed time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    if total_successful_reps > 0:
        print(f"  Average time per replication: {avg_time_per_rep:.2f}s")
    print("="*70)

if __name__ == '__main__':
    main()

