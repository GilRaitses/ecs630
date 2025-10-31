#!/usr/bin/env python3
"""
Prepare simulation-ready dataset from tier2_complete H5 file.

Extracts:
1. Empirical distributions (speed, heading, position) for sampling
2. Stimulus schedule parameters (verified pulse timing)
3. Baseline statistics for validation
4. Simulation-ready trajectory data

GOLD STANDARD: Uses tier2_complete.h5 with proper LED1/LED2 extraction
"""

import sys
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import h5py

def extract_stimulus_parameters(events_df, stimulus_df):
    """
    Extract and verify stimulus parameters from data.
    
    Returns verified pulse duration, inter-pulse intervals, intensity levels.
    """
    # Get inter-onset intervals from stimulus onsets
    # Check for stimulus_onset column, or use led1Val_ton transitions
    if 'stimulus_onset' in events_df.columns:
        onsets = events_df[events_df['stimulus_onset'] == True]['time'].values
    elif 'led1Val_ton' in events_df.columns:
        # Detect onsets as transitions from False to True in led1Val_ton
        led1_ton = events_df['led1Val_ton'].values
        led1_ton_diff = np.diff(led1_ton.astype(int), prepend=0)
        onset_indices = np.where(led1_ton_diff == 1)[0]
        onsets = events_df.iloc[onset_indices]['time'].values
    else:
        onsets = np.array([])
    
    # Detect actual pulse duration from LED drop
    pulse_duration = None
    if 'led1Val' in events_df.columns and len(onsets) > 0:
        # Find when LED drops after each onset
        threshold = 50.0  # LED threshold for "off"
        pulse_durations = []
        
        for onset_time in onsets[:10]:  # Check first 10 pulses
            onset_idx = np.argmin(np.abs(events_df['time'].values - onset_time))
            # Look ahead up to 100 seconds
            pulse_data = events_df.iloc[onset_idx:min(onset_idx+1000, len(events_df))]
            led1_vals = pulse_data['led1Val'].values
            
            drop_idx = np.where(led1_vals < threshold)[0]
            if len(drop_idx) > 0:
                duration = events_df.iloc[onset_idx + drop_idx[0]]['time'] - onset_time
                pulse_durations.append(duration)
        
        if len(pulse_durations) > 0:
            pulse_duration = np.mean(pulse_durations)
            print(f"  Detected pulse duration: {pulse_duration:.1f}s (from LED drop)")
    
    # Fallback to protocol value if detection failed
    if pulse_duration is None:
        pulse_duration = 10.0  # Protocol value
        print(f"  Using protocol pulse duration: {pulse_duration}s")
    
    if len(onsets) > 1:
        onsets = np.sort(onsets)  # Ensure sorted
        intervals = np.diff(onsets)
        # Filter positive intervals only
        positive_intervals = intervals[intervals > 0]
        if len(positive_intervals) > 0:
            inter_pulse_intervals = np.unique(np.round(positive_intervals, 1))
            # Inter-pulse gap = interval - pulse_duration
            inter_pulse_gaps = inter_pulse_intervals - pulse_duration
        else:
            inter_pulse_gaps = np.array([40.0])  # Default based on observed ~40s gap
        
        print(f"  Inter-onset intervals: {inter_pulse_intervals}")
        print(f"  Inter-pulse gaps: {inter_pulse_gaps}")
    else:
        inter_pulse_gaps = np.array([50.0])  # Default from analysis
    
    # Get intensity levels from LED1 values
    led1_on = events_df[events_df['led1Val_ton'] == True]['led1Val'].values
    intensity_levels = np.unique(np.round(led1_on[led1_on > 0], 0))
    
    # Convert to percentage (LED range 0-250)
    intensity_pcts = (intensity_levels / 250.0 * 100).astype(int)
    
    print(f"  LED1 intensity levels: {intensity_levels} ({intensity_pcts}%)")
    
    return {
        'pulse_duration_s': pulse_duration,
        'inter_pulse_intervals_s': sorted(inter_pulse_gaps.tolist()),
        'intensity_levels': sorted(intensity_pcts.tolist()),
        'led1_max': float(np.max(events_df['led1Val'])),
        'led2_mean': float(events_df['led2Val'].mean()),
        'led2_std': float(events_df['led2Val'].std())
    }

def extract_empirical_distributions(trajectories_df):
    """
    Extract empirical distributions for simulation sampling.
    
    Returns fitted distributions for:
    - Speed (log-normal)
    - Heading (circular uniform/von Mises)
    - Starting positions (x, y)
    """
    print("\nExtracting empirical distributions...")
    
    # Speed distribution (log-normal fit)
    speeds = trajectories_df['speed'].values
    speeds = speeds[speeds > 0]  # Filter zeros
    
    if len(speeds) > 0:
        log_speeds = np.log(speeds + 1e-6)
        speed_mu = float(np.mean(log_speeds))
        speed_sigma = float(np.std(log_speeds))
        speed_mean = float(np.mean(speeds))
        speed_std = float(np.std(speeds))
        speed_min = float(np.min(speeds))
        speed_max = float(np.max(speeds))
    else:
        speed_mu = speed_sigma = speed_mean = speed_std = 0.0
        speed_min = speed_max = 0.01
    
    print(f"  Speed: log-normal(μ={speed_mu:.4f}, σ={speed_sigma:.4f})")
    print(f"    Mean: {speed_mean:.4f}, Std: {speed_std:.4f}, Range: [{speed_min:.4f}, {speed_max:.4f}]")
    
    # Heading distribution (circular)
    headings = trajectories_df['heading'].values
    heading_mean = float(np.mean(headings))
    heading_std = float(np.std(headings))
    
    # Convert to circular statistics if needed
    heading_sin_mean = float(np.mean(np.sin(headings)))
    heading_cos_mean = float(np.mean(np.cos(headings)))
    
    print(f"  Heading: circular mean={heading_mean:.4f}, std={heading_std:.4f}")
    
    # Position distribution (starting positions)
    # Group by track to get initial positions
    starting_positions = []
    starting_headings = []
    
    # Check for track_id column (events file) or use frame/experiment_id grouping
    if 'track_id' in trajectories_df.columns:
        track_col = 'track_id'
    elif 'experiment_id' in trajectories_df.columns:
        # If no track_id, use first N positions as starting positions
        # Sample first position from each time window
        unique_times = trajectories_df['time'].unique()[:12]  # Assuming ~12 tracks
        for t in unique_times:
            track_data = trajectories_df[trajectories_df['time'] == t].sort_values('x')
            if len(track_data) > 0:
                starting_positions.append([track_data.iloc[0]['x'], track_data.iloc[0]['y']])
                starting_headings.append(track_data.iloc[0]['heading'])
    else:
        # Fallback: sample random positions from first time points
        first_time_points = trajectories_df.nsmallest(12, 'time')
        for _, row in first_time_points.iterrows():
            starting_positions.append([row['x'], row['y']])
            starting_headings.append(row['heading'])
    
    if 'track_id' in trajectories_df.columns:
        for track_id in trajectories_df['track_id'].unique():
            track_data = trajectories_df[trajectories_df['track_id'] == track_id].sort_values('time')
            if len(track_data) > 0:
                starting_positions.append([track_data.iloc[0]['x'], track_data.iloc[0]['y']])
                starting_headings.append(track_data.iloc[0]['heading'])
    
    starting_positions = np.array(starting_positions)
    starting_headings = np.array(starting_headings)
    
    pos_x_mean = float(np.mean(starting_positions[:, 0]))
    pos_x_std = float(np.std(starting_positions[:, 0]))
    pos_y_mean = float(np.mean(starting_positions[:, 1]))
    pos_y_std = float(np.std(starting_positions[:, 1]))
    
    print(f"  Starting positions: x={pos_x_mean:.2f}±{pos_x_std:.2f}, y={pos_y_mean:.2f}±{pos_y_std:.2f}")
    print(f"  Starting headings: {len(starting_headings)} samples")
    
    return {
        'speed': {
            'distribution': 'lognormal',
            'mu': speed_mu,
            'sigma': speed_sigma,
            'mean': speed_mean,
            'std': speed_std,
            'min': speed_min,
            'max': speed_max
        },
        'heading': {
            'distribution': 'circular',
            'mean': heading_mean,
            'std': heading_std,
            'sin_mean': heading_sin_mean,
            'cos_mean': heading_cos_mean
        },
        'starting_position': {
            'x_mean': pos_x_mean,
            'x_std': pos_x_std,
            'y_mean': pos_y_mean,
            'y_std': pos_y_std,
            'n_samples': len(starting_positions)
        },
        'starting_headings': starting_headings.tolist()
    }

def extract_baseline_statistics(events_df):
    """
    Extract baseline statistics for simulation validation.
    """
    print("\nExtracting baseline statistics...")
    
    total_time = events_df['time'].max() - events_df['time'].min()
    n_turns = events_df['is_turn'].sum()
    turn_rate = n_turns / (total_time / 60.0) if total_time > 0 else 0.0
    
    # Average speed
    avg_speed = events_df['speed'].mean()
    
    # Stop fraction (speed < threshold)
    speed_threshold = 0.001
    stop_fraction = (events_df['speed'] < speed_threshold).sum() / len(events_df)
    
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Total turns: {n_turns}")
    print(f"  Turn rate: {turn_rate:.2f} turns/min")
    print(f"  Average speed: {avg_speed:.4f}")
    print(f"  Stop fraction: {stop_fraction:.3f}")
    
    return {
        'total_time_s': float(total_time),
        'total_turns': int(n_turns),
        'turn_rate_per_min': float(turn_rate),
        'avg_speed': float(avg_speed),
        'stop_fraction': float(stop_fraction)
    }

def extract_klein_run_statistics(klein_run_table_file):
    """
    Extract run/turn statistics from Klein run table for better empirical sampling.
    
    Returns distributions of run durations, lengths, speeds, and turn characteristics.
    """
    if not Path(klein_run_table_file).exists():
        print(f"  Warning: Klein run table not found at {klein_run_table_file}")
        return None
    
    print("\nExtracting Klein run table statistics...")
    klein_df = pd.read_csv(klein_run_table_file)
    
    # Run statistics
    run_durations = klein_df['runT'].values
    run_lengths = klein_df['runL'].values
    run_speeds = klein_df['runQ'].values / (run_durations + 1e-6)  # Average speed during run
    
    # Filter valid values
    run_durations = run_durations[run_durations > 0]
    run_lengths = run_lengths[run_lengths > 0]
    run_speeds = run_speeds[run_speeds > 0]
    
    # Turn statistics (runs ending in reorientations)
    turns_df = klein_df[klein_df['reoYN'] == 1]
    turn_magnitudes = turns_df['turn_magnitude'].values if 'turn_magnitude' in turns_df.columns else np.array([])
    turn_directions = turns_df['turn_direction'].values if 'turn_direction' in turns_df.columns else np.array([])
    n_head_swings = turns_df['reo#HS'].values if 'reo#HS' in turns_df.columns else np.array([])
    
    # Ensure turn_directions are numeric
    if len(turn_directions) > 0:
        turn_directions = pd.to_numeric(turn_directions, errors='coerce')
        turn_directions = turn_directions[~np.isnan(turn_directions)]
    
    print(f"  Total runs: {len(klein_df)}")
    print(f"  Runs with turns: {len(turns_df)}")
    print(f"  Mean run duration: {np.mean(run_durations):.2f}s")
    print(f"  Mean run length: {np.mean(run_lengths):.4f}")
    print(f"  Mean run speed: {np.mean(run_speeds):.4f}")
    
    if len(turn_magnitudes) > 0:
        print(f"  Mean turn magnitude: {np.mean(np.abs(turn_magnitudes)):.4f} rad")
        print(f"  Mean head swings per turn: {np.mean(n_head_swings):.1f}")
    
    return {
        'run_durations': {
            'values': run_durations.tolist(),
            'mean': float(np.mean(run_durations)),
            'std': float(np.std(run_durations)),
            'min': float(np.min(run_durations)),
            'max': float(np.max(run_durations))
        },
        'run_lengths': {
            'values': run_lengths.tolist(),
            'mean': float(np.mean(run_lengths)),
            'std': float(np.std(run_lengths)),
            'min': float(np.min(run_lengths)),
            'max': float(np.max(run_lengths))
        },
        'run_speeds': {
            'values': run_speeds.tolist(),
            'mean': float(np.mean(run_speeds)),
            'std': float(np.std(run_speeds)),
            'min': float(np.min(run_speeds)),
            'max': float(np.max(run_speeds))
        },
        'turn_magnitudes': {
            'values': turn_magnitudes.tolist() if len(turn_magnitudes) > 0 else [],
            'mean': float(np.mean(np.abs(turn_magnitudes))) if len(turn_magnitudes) > 0 else 0.0,
            'std': float(np.std(turn_magnitudes)) if len(turn_magnitudes) > 0 else 0.0
        },
        'turn_directions': {
            'values': turn_directions.tolist() if len(turn_directions) > 0 else [],
            'left_turn_fraction': float(np.sum(turn_directions > 0) / len(turn_directions)) if len(turn_directions) > 0 else 0.5,
            'right_turn_fraction': float(np.sum(turn_directions < 0) / len(turn_directions)) if len(turn_directions) > 0 else 0.5
        },
        'head_swings_per_turn': {
            'values': n_head_swings.tolist() if len(n_head_swings) > 0 else [],
            'mean': float(np.mean(n_head_swings)) if len(n_head_swings) > 0 else 0.0,
            'std': float(np.std(n_head_swings)) if len(n_head_swings) > 0 else 0.0
        }
    }

def extract_event_parameters(trajectories_file: str) -> Optional[Dict]:
    """
    Learn optimal event detection parameters from empirical data.
    
    Returns learned parameters for pause detection, stop fraction, and reversal detection.
    """
    try:
        from learn_event_parameters import learn_optimal_event_parameters
        
        print("\nLearning event detection parameters from empirical data...")
        trajectories_df = pd.read_csv(trajectories_file, nrows=100000)  # Sample for speed
        
        # Extract empirical pause rate if available
        empirical_pause_rate = None
        if 'is_pause' in trajectories_df.columns and 'time' in trajectories_df.columns:
            pause_starts = trajectories_df['is_pause'].diff().fillna(False) & trajectories_df['is_pause']
            n_pause_events = pause_starts.sum()
            total_time = trajectories_df['time'].max() - trajectories_df['time'].min()
            if total_time > 0:
                empirical_pause_rate = n_pause_events / (total_time / 60.0)
                print(f"  Empirical pause rate: {empirical_pause_rate:.1f} pauses/min")
        
        # Learn parameters
        params = learn_optimal_event_parameters(trajectories_df, 
                                                empirical_pause_rate=empirical_pause_rate,
                                                use_magat_approach=True)
        
        # Convert to JSON-serializable dict
        event_params = {
            'pause_speed_threshold': float(params.pause_speed_threshold),
            'pause_min_duration': float(params.pause_min_duration),
            'stop_speed_threshold': float(params.stop_speed_threshold),
            'reversal_angle_threshold': float(params.reversal_angle_threshold),
            'reversal_angle_threshold_degrees': float(np.rad2deg(params.reversal_angle_threshold))
        }
        
        # Add analysis results if available
        if hasattr(params, '_learned_params'):
            if 'pause_analysis' in params._learned_params:
                pause_analysis = params._learned_params['pause_analysis']
                event_params['pause_analysis'] = {
                    'estimated_pause_rate': float(pause_analysis.get('estimated_pause_rate', 0)),
                    'pause_fraction': float(pause_analysis.get('pause_fraction', 0)),
                    'method': pause_analysis.get('method', 'unknown')
                }
            
            if 'reversal_analysis' in params._learned_params:
                reversal_analysis = params._learned_params['reversal_analysis']
                event_params['reversal_analysis'] = {
                    'reversal_fraction': float(reversal_analysis.get('reversal_fraction', 0)),
                    'threshold_degrees': float(reversal_analysis.get('threshold_degrees', 90))
                }
        
        print(f"  ✓ Learned pause threshold: {event_params['pause_speed_threshold']:.6f}")
        print(f"  ✓ Learned reversal threshold: {event_params['reversal_angle_threshold_degrees']:.1f}°")
        
        return event_params
    except Exception as e:
        print(f"  Warning: Could not learn event parameters: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Prepare simulation dataset from tier2_complete')
    parser.add_argument('--events-file', type=str,
                       default='data/engineered/GMR61_tier2_complete_events.csv',
                       help='Path to events CSV')
    parser.add_argument('--trajectories-file', type=str,
                       default='data/engineered/GMR61_tier2_complete_trajectories.csv',
                       help='Path to trajectories CSV')
    parser.add_argument('--klein-run-table-file', type=str,
                       default='data/engineered/GMR61_tier2_complete_klein_run_table.csv',
                       help='Path to Klein run table CSV (optional)')
    parser.add_argument('--output-dir', type=str,
                       default='data/simulation',
                       help='Output directory for simulation datasets')
    parser.add_argument('--experiment-id', type=str,
                       default='GMR61_tier2_complete',
                       help='Experiment ID')
    
    args = parser.parse_args()
    
    print("="*80)
    print("Preparing Simulation Dataset from tier2_complete")
    print("="*80)
    
    # Load data
    print(f"\nLoading data...")
    print(f"  Events: {args.events_file}")
    print(f"  Trajectories: {args.trajectories_file}")
    
    events_df = pd.read_csv(args.events_file)
    trajectories_df = pd.read_csv(args.trajectories_file)
    
    print(f"  Loaded {len(events_df)} event records, {len(trajectories_df)} trajectory points")
    
    # Extract Klein run table statistics if available
    klein_stats = None
    if args.klein_run_table_file:
        klein_stats = extract_klein_run_statistics(args.klein_run_table_file)
    
    # Extract stimulus parameters
    print(f"\n{'='*80}")
    print("EXTRACTING STIMULUS PARAMETERS")
    print(f"{'='*80}")
    stimulus_params = extract_stimulus_parameters(events_df, None)
    
    # Extract empirical distributions
    print(f"\n{'='*80}")
    print("EXTRACTING EMPIRICAL DISTRIBUTIONS")
    print(f"{'='*80}")
    empirical_dists = extract_empirical_distributions(trajectories_df)
    
    # Extract baseline statistics
    print(f"\n{'='*80}")
    print("EXTRACTING BASELINE STATISTICS")
    print(f"{'='*80}")
    baseline_stats = extract_baseline_statistics(events_df)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save simulation-ready trajectory data (filtered/optimized)
    print(f"\n{'='*80}")
    print("SAVING SIMULATION DATASETS")
    print(f"{'='*80}")
    
    # Save empirical distributions as JSON
    dists_file = output_dir / f'{args.experiment_id}_empirical_distributions.json'
    with open(dists_file, 'w') as f:
        json.dump(empirical_dists, f, indent=2)
    print(f"✓ Saved empirical distributions to {dists_file}")
    
    # Save stimulus parameters
    stim_file = output_dir / f'{args.experiment_id}_stimulus_parameters.json'
    with open(stim_file, 'w') as f:
        json.dump(stimulus_params, f, indent=2)
    print(f"✓ Saved stimulus parameters to {stim_file}")
    
    # Save baseline statistics
    baseline_file = output_dir / f'{args.experiment_id}_baseline_statistics.json'
    with open(baseline_file, 'w') as f:
        json.dump(baseline_stats, f, indent=2)
    print(f"✓ Saved baseline statistics to {baseline_file}")
    
    # Save Klein run statistics if available
    if klein_stats is not None:
        klein_file = output_dir / f'{args.experiment_id}_klein_run_statistics.json'
        with open(klein_file, 'w') as f:
            json.dump(klein_stats, f, indent=2)
        print(f"✓ Saved Klein run statistics to {klein_file}")
    
    # Learn event detection parameters
    event_params = extract_event_parameters(args.trajectories_file)
    event_params_file = None
    if event_params:
        event_params_file = output_dir / f'{args.experiment_id}_event_parameters.json'
        with open(event_params_file, 'w') as f:
            json.dump(event_params, f, indent=2)
        print(f"✓ Saved event parameters to {event_params_file}")
    
    # Save simulation-ready trajectory data (every Nth point for efficiency)
    # Include all columns needed for simulation
    sim_cols = ['time', 'x', 'y', 'heading', 'speed', 'led1Val', 'led1Val_ton', 
                'led1Val_toff', 'led2Val', 'led2Val_ton', 'led2Val_toff',
                'stimulus_on', 'time_since_stimulus', 'track_id']
    
    available_cols = [c for c in sim_cols if c in trajectories_df.columns]
    sim_traj = trajectories_df[available_cols].copy()
    
    # Downsample if too large (keep every 10th point = 1Hz sampling)
    if len(sim_traj) > 100000:
        sim_traj = sim_traj.iloc[::10].reset_index(drop=True)
        print(f"  Downsampled to {len(sim_traj)} points (1Hz sampling)")
    
    sim_traj_file = output_dir / f'{args.experiment_id}_simulation_trajectories.csv'
    sim_traj.to_csv(sim_traj_file, index=False)
    print(f"✓ Saved simulation trajectories to {sim_traj_file}")
    
    # Create summary
    summary = {
        'experiment_id': args.experiment_id,
        'source': 'tier2_complete.h5',
        'n_events': len(events_df),
        'n_trajectory_points': len(trajectories_df),
        'n_simulation_points': len(sim_traj),
        'stimulus_parameters': stimulus_params,
        'baseline_statistics': baseline_stats,
        'empirical_distributions': {
            'speed': empirical_dists['speed'],
            'heading': empirical_dists['heading'],
            'starting_position': empirical_dists['starting_position']
        },
        'has_klein_statistics': klein_stats is not None,
        'has_event_parameters': event_params is not None
    }
    
    if klein_stats is not None:
        summary['klein_statistics'] = {
            'n_runs': len(klein_stats['run_durations']['values']),
            'mean_run_duration': klein_stats['run_durations']['mean'],
            'mean_run_speed': klein_stats['run_speeds']['mean']
        }
    
    if event_params is not None:
        summary['event_parameters'] = {
            'pause_speed_threshold': event_params['pause_speed_threshold'],
            'reversal_angle_threshold_degrees': event_params['reversal_angle_threshold_degrees']
        }
    
    summary_file = output_dir / f'{args.experiment_id}_simulation_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved summary to {summary_file}")
    
    print(f"\n{'='*80}")
    print("✓ SIMULATION DATASET PREPARATION COMPLETE")
    print(f"{'='*80}")
    print(f"\nOutput files in: {output_dir}")
    print(f"  - Empirical distributions: {dists_file.name}")
    print(f"  - Stimulus parameters: {stim_file.name}")
    print(f"  - Baseline statistics: {baseline_file.name}")
    if klein_stats is not None:
        print(f"  - Klein run statistics: {klein_file.name}")
    if event_params is not None:
        print(f"  - Event parameters: {event_params_file.name}")
    print(f"  - Simulation trajectories: {sim_traj_file.name}")
    print(f"  - Summary: {summary_file.name}")

if __name__ == '__main__':
    main()

