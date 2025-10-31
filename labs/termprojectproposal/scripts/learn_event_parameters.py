#!/usr/bin/env python3
"""
Event Parameter Learning System

Learns optimal thresholds for behavioral event detection (pauses, heading reversals)
from empirical data distributions, similar to MAGAT parameter learning.

This ensures pause rate, reversal rate, and other event-based KPIs are calculated
using biologically plausible thresholds calibrated from real larval behavior.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from pathlib import Path

class EventDetectionParameters:
    """Learned parameters for event detection."""
    def __init__(self):
        # Pause detection
        self.pause_speed_threshold = 0.001  # Speed below which larva is paused
        self.pause_min_duration = 0.2  # Minimum pause duration (seconds)
        
        # Stop fraction (similar to pause but continuous measure)
        self.stop_speed_threshold = 0.001  # Speed threshold for stop fraction
        
        # Heading reversal detection
        self.reversal_angle_threshold = np.pi / 2  # ~90 degrees
        
        # Store analysis results
        self._learned_params = {}

def analyze_pause_parameters(speed: np.ndarray, time: np.ndarray, 
                             empirical_pause_rate: Optional[float] = None,
                             use_magat_approach: bool = True) -> Dict[str, float]:
    """
    Analyze speed distribution to learn optimal pause detection thresholds.
    
    Uses MAGAT's approach: pauses are periods where speed < stop_speed_cut.
    MAGAT default: stop_speed_cut = 2 mm/s, start_speed_cut = 3 mm/s
    
    Parameters
    ----------
    speed : ndarray
        Speed values from empirical data (units: pixels/second or mm/s)
    time : ndarray
        Time values (for duration calculations)
    empirical_pause_rate : float, optional
        Known pause rate from empirical data (pauses/min)
        Used to calibrate thresholds to match observed rates
    use_magat_approach : bool
        If True, use MAGAT's approach: pauses = non-run periods (speed < stop_speed_cut)
        If False, use simple threshold-based pause detection
    
    Returns
    -------
    dict
        Analysis results with suggested thresholds
    """
    speed_positive = speed[speed > 0]
    
    if len(speed_positive) == 0:
        return {
            'speed_threshold': 0.001,
            'min_duration': 0.2,
            'percentiles': {},
            'pause_fraction': 0.0
        }
    
    # Calculate percentiles
    percentiles = {}
    for p in [1, 2, 5, 10, 15, 25, 50, 75, 90]:
        percentiles[f'p{p}'] = np.percentile(speed_positive, p)
    
    if use_magat_approach:
        # MAGAT approach: stop_speed_cut ends runs, periods below this are "pauses"
        # MAGAT default: stop_speed_cut = 2 mm/s
        # Learn from data: use 10th-15th percentile as stop threshold
        # This marks slow periods as pauses (similar to MAGAT's logic)
        suggested_threshold = percentiles['p10']  # 10th percentile
        
        # Ensure threshold captures slow periods but not noise
        # Should be between p5 and p25
        suggested_threshold = np.clip(suggested_threshold, 
                                     percentiles['p5'], 
                                     percentiles['p25'])
        
        # If empirical rate provided, calibrate to match
        # Higher threshold -> more pause periods -> more pause events
        if empirical_pause_rate is not None:
            # Test different thresholds to match empirical rate
            test_thresholds = [percentiles['p5'], percentiles['p10'], 
                             percentiles['p15'], percentiles['p25']]
            best_threshold = suggested_threshold
            best_diff = float('inf')
            
            for test_thresh in test_thresholds:
                is_pause = speed < test_thresh
                pause_starts = np.diff(is_pause.astype(int), prepend=0) == 1
                n_pause_events = pause_starts.sum()
                
                if len(time) > 1:
                    total_time = time.max() - time.min()
                    test_rate = n_pause_events / (total_time / 60.0) if total_time > 0 else 0.0
                else:
                    test_rate = 0.0
                
                diff = abs(test_rate - empirical_pause_rate)
                if diff < best_diff:
                    best_diff = diff
                    best_threshold = test_thresh
            
            suggested_threshold = best_threshold
        
        # Minimum duration: MAGAT doesn't explicitly filter pauses by duration
        # But we use 0.2s to match detect_events.py and filter noise
        min_duration = 0.2
    else:
        # Simple threshold-based approach (original)
        suggested_threshold = max(percentiles['p5'], np.median(speed_positive) * 0.05)
        suggested_threshold = max(suggested_threshold, 0.0001)
        min_duration = 0.2
    
    # Calculate pause fraction with suggested threshold
    pause_frames = (speed < suggested_threshold).sum()
    pause_fraction = pause_frames / len(speed)
    
    # Estimate pause rate based on pause events (not frames)
    is_pause = speed < suggested_threshold
    pause_starts = np.diff(is_pause.astype(int), prepend=0) == 1
    n_pause_events = pause_starts.sum()
    
    if len(time) > 1:
        total_time = time.max() - time.min()
        estimated_rate = n_pause_events / (total_time / 60.0) if total_time > 0 else 0.0
    else:
        estimated_rate = 0.0
    
    return {
        'speed_threshold': suggested_threshold,
        'min_duration': min_duration,
        'percentiles': percentiles,
        'pause_fraction': pause_fraction,
        'estimated_pause_rate': estimated_rate,
        'n_pause_events': n_pause_events,
        'method': 'magat' if use_magat_approach else 'simple'
    }

def analyze_reversal_parameters(heading_change: np.ndarray,
                                empirical_reversal_rate: Optional[float] = None) -> Dict[str, float]:
    """
    Analyze heading change distribution to learn optimal reversal detection threshold.
    
    Parameters
    ----------
    heading_change : ndarray
        Absolute heading change values (radians)
    empirical_reversal_rate : float, optional
        Known reversal rate from empirical data (reversals/min)
    
    Returns
    -------
    dict
        Analysis results with suggested threshold
    """
    heading_change_abs = np.abs(heading_change)
    heading_change_positive = heading_change_abs[heading_change_abs > 0]
    
    if len(heading_change_positive) == 0:
        return {
            'angle_threshold': np.pi / 2,
            'percentiles': {},
            'reversal_fraction': 0.0
        }
    
    # Calculate percentiles
    percentiles = {}
    for p in [75, 80, 85, 90, 95, 99]:
        percentiles[f'p{p}'] = np.percentile(heading_change_abs, p)
    
    # Suggested threshold: 90th percentile (large heading changes = reversals)
    # This marks ~10% of heading changes as reversals
    suggested_threshold = percentiles.get('p90', np.pi / 2)
    
    # Ensure threshold is reasonable (between 45° and 135°)
    suggested_threshold = np.clip(suggested_threshold, np.pi / 4, 3 * np.pi / 4)
    
    # Calculate reversal fraction
    reversal_frames = (heading_change_abs > suggested_threshold).sum()
    reversal_fraction = reversal_frames / len(heading_change_abs)
    
    return {
        'angle_threshold': suggested_threshold,
        'percentiles': percentiles,
        'reversal_fraction': reversal_fraction,
        'threshold_degrees': np.rad2deg(suggested_threshold)
    }

def learn_optimal_event_parameters(trajectory_df: pd.DataFrame,
                                   empirical_pause_rate: Optional[float] = None,
                                   empirical_reversal_rate: Optional[float] = None,
                                   use_magat_approach: bool = True) -> EventDetectionParameters:
    """
    Learn optimal event detection parameters from empirical trajectory data.
    
    Parameters
    ----------
    trajectory_df : pd.DataFrame
        Trajectory data with columns: time, speed, heading (or heading_change)
    empirical_pause_rate : float, optional
        Known pause rate from empirical data (pauses/min)
    empirical_reversal_rate : float, optional
        Known reversal rate from empirical data (reversals/min)
    
    Returns
    -------
    EventDetectionParameters
        Calibrated event detection parameters
    """
    params = EventDetectionParameters()
    
    # Analyze pause parameters
    if 'speed' in trajectory_df.columns and 'time' in trajectory_df.columns:
        speed = trajectory_df['speed'].values
        time = trajectory_df['time'].values
        
        pause_analysis = analyze_pause_parameters(speed, time, empirical_pause_rate, use_magat_approach)
        params.pause_speed_threshold = pause_analysis['speed_threshold']
        params.pause_min_duration = pause_analysis['min_duration']
        params.stop_speed_threshold = pause_analysis['speed_threshold']  # Use same for stop fraction
        
        params._learned_params['pause_analysis'] = pause_analysis
    
    # Analyze reversal parameters
    if 'heading_change' in trajectory_df.columns:
        heading_change = trajectory_df['heading_change'].values
    elif 'heading' in trajectory_df.columns:
        heading = trajectory_df['heading'].values
        heading_change = np.diff(heading, prepend=heading[0])
        # Wrap angles to [-pi, pi]
        heading_change = np.where(heading_change > np.pi, heading_change - 2*np.pi, heading_change)
        heading_change = np.where(heading_change < -np.pi, heading_change + 2*np.pi, heading_change)
    else:
        heading_change = None
    
    if heading_change is not None:
        reversal_analysis = analyze_reversal_parameters(heading_change, empirical_reversal_rate)
        params.reversal_angle_threshold = reversal_analysis['angle_threshold']
        params._learned_params['reversal_analysis'] = reversal_analysis
    
    return params

def print_learned_parameters(params: EventDetectionParameters):
    """Print learned parameters in a readable format."""
    print("\n" + "="*70)
    print("LEARNED EVENT DETECTION PARAMETERS")
    print("="*70)
    
    print(f"\nPause Detection:")
    print(f"  speed_threshold: {params.pause_speed_threshold:.6f}")
    print(f"  min_duration: {params.pause_min_duration:.3f} seconds")
    
    if 'pause_analysis' in params._learned_params:
        analysis = params._learned_params['pause_analysis']
        print(f"  Estimated pause rate: {analysis.get('estimated_pause_rate', 0):.1f} pauses/min")
        print(f"  Pause fraction: {analysis.get('pause_fraction', 0):.4f}")
    
    print(f"\nStop Fraction:")
    print(f"  speed_threshold: {params.stop_speed_threshold:.6f}")
    
    print(f"\nHeading Reversal Detection:")
    print(f"  angle_threshold: {np.rad2deg(params.reversal_angle_threshold):.2f}° ({params.reversal_angle_threshold:.6f} rad)")
    
    if 'reversal_analysis' in params._learned_params:
        analysis = params._learned_params['reversal_analysis']
        print(f"  Reversal fraction: {analysis.get('reversal_fraction', 0):.4f}")
    
    print("="*70 + "\n")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Learn event detection parameters from empirical data')
    parser.add_argument('--empirical-data', type=str, 
                       default='data/engineered/GMR61_tier2_complete_trajectories.csv',
                       help='Path to empirical trajectory data CSV')
    parser.add_argument('--sample-size', type=int, default=50000,
                       help='Number of rows to sample for analysis')
    args = parser.parse_args()
    
    empirical_path = Path(args.empirical_data)
    if not empirical_path.exists():
        print(f"Error: Empirical data file not found: {empirical_path}")
        exit(1)
    
    print(f"Loading empirical data from {empirical_path}...")
    df = pd.read_csv(empirical_path, nrows=args.sample_size)
    
    print(f"Loaded {len(df):,} trajectory frames")
    
    # Extract empirical rates if available
    empirical_pause_rate = None
    empirical_reversal_rate = None
    
    if 'is_pause' in df.columns and 'time' in df.columns:
        pause_starts = df['is_pause'].diff().fillna(False) & df['is_pause']
        n_pause_events = pause_starts.sum()
        total_time = df['time'].max() - df['time'].min()
        if total_time > 0:
            empirical_pause_rate = n_pause_events / (total_time / 60.0)
            print(f"Empirical pause rate: {empirical_pause_rate:.1f} pauses/min")
    
    # Learn parameters
    params = learn_optimal_event_parameters(df, 
                                          empirical_pause_rate=empirical_pause_rate,
                                          empirical_reversal_rate=empirical_reversal_rate)
    
    print_learned_parameters(params)

