#!/usr/bin/env python3
"""
Comprehensive visualization tools for behavioral events.

Includes:
1. Distribution plots for turn duration, pause duration, body bend (spine curve energy), turn rate
2. Stimulus-locked turn rate analysis (PSTH with cycle-by-cycle tracking)
3. Salience envelope (aggregated across all cycles)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import h5py
from scipy import stats
from typing import Dict, List, Tuple, Optional
import json

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300


def load_stimulus_cycles(h5_path: str, fps: float = 10.0, threshold: float = 50.0) -> Tuple[List[Dict], np.ndarray]:
    """
    Extract stimulus cycle information from H5 file.
    
    Returns cycles list and LED1 data.
    """
    cycles = []
    
    with h5py.File(h5_path, 'r') as f:
        # Get onset frames
        onset_frames = f['stimulus']['onset_frames'][:]
        onset_frames = np.sort(onset_frames)
        
        # Get LED1 data
        if 'global_quantities' in f and 'led1Val' in f['global_quantities']:
            gq_item = f['global_quantities']['led1Val']
            if isinstance(gq_item, h5py.Group) and 'yData' in gq_item:
                led1_data = gq_item['yData'][:]
            elif isinstance(gq_item, h5py.Dataset):
                led1_data = gq_item[:]
        else:
            raise ValueError("LED1 data not found in H5 file")
        
        # Extract cycle information
        for i, onset_frame in enumerate(onset_frames):
            onset_frame_int = int(onset_frame)
            onset_time = onset_frame_int / fps
            
            # Find when LED drops below threshold
            pulse_window_start = onset_frame_int
            pulse_window_end = min(onset_frame_int + 1000, len(led1_data))
            pulse_window = led1_data[pulse_window_start:pulse_window_end]
            
            drop_indices = np.where(pulse_window < threshold)[0]
            
            if len(drop_indices) > 0:
                drop_frame_int = pulse_window_start + drop_indices[0]
                drop_time = drop_frame_int / fps
            else:
                if i < len(onset_frames) - 1:
                    drop_frame_int = int(onset_frames[i+1])
                    drop_time = drop_frame_int / fps
                else:
                    drop_frame_int = len(led1_data) - 1
                    drop_time = drop_frame_int / fps
            
            # Calculate ETI
            if i < len(onset_frames) - 1:
                next_onset_frame = int(onset_frames[i+1])
                next_onset_time = next_onset_frame / fps
                eti = next_onset_time - drop_time
            else:
                eti = None
            
            cycle = {
                'cycle_num': i + 1,
                'frame_start': onset_frame_int,
                'frame_end': drop_frame_int,
                'time_start': onset_time,
                'time_end': drop_time,
                'eti': eti,
                'cycle_period': (next_onset_time - onset_time) if i < len(onset_frames) - 1 else None
            }
            cycles.append(cycle)
    
    return cycles, led1_data


def plot_event_distributions(trajectories_df: pd.DataFrame, output_dir: Path):
    """
    Plot distributions for behavioral events.
    
    Creates:
    - Turn duration distribution
    - Pause duration distribution
    - Body bend (spine curve energy) distribution
    - Turn rate distribution (per track)
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Behavioral Event Distributions', fontsize=16, fontweight='bold')
    
    # 1. Turn Duration Distribution
    turn_events = trajectories_df[trajectories_df['turn_duration'] > 0].copy()
    if len(turn_events) > 0:
        unique_turns = turn_events.groupby('turn_event_id')['turn_duration'].first()
        axes[0, 0].hist(unique_turns.values, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(unique_turns.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {unique_turns.mean():.3f}s')
        axes[0, 0].axvline(unique_turns.median(), color='orange', linestyle='--', linewidth=2, label=f'Median: {unique_turns.median():.3f}s')
        axes[0, 0].set_xlabel('Turn Duration (s)')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Turn Duration Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Pause Duration Distribution
    pause_events = trajectories_df[trajectories_df['is_pause'] == True].copy()
    if len(pause_events) > 0:
        unique_pauses = pause_events.groupby(pause_events.index)['pause_duration'].first()
        unique_pauses = unique_pauses[unique_pauses > 0]
        if len(unique_pauses) > 0:
            axes[0, 1].hist(unique_pauses.values, bins=30, color='coral', alpha=0.7, edgecolor='black')
            axes[0, 1].axvline(unique_pauses.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {unique_pauses.mean():.3f}s')
            axes[0, 1].axvline(unique_pauses.median(), color='orange', linestyle='--', linewidth=2, label=f'Median: {unique_pauses.median():.3f}s')
            axes[0, 1].set_xlabel('Pause Duration (s)')
            axes[0, 1].set_ylabel('Count')
            axes[0, 1].set_title('Pause Duration Distribution')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Body Bend (Spine Curve Energy) Distribution
    if 'spine_curve_energy' in trajectories_df.columns:
        spine_energy = trajectories_df['spine_curve_energy'].values
        spine_energy = spine_energy[spine_energy > 0]  # Filter zeros
        if len(spine_energy) > 0:
            # Log transform for visualization
            spine_energy_log = np.log(spine_energy + 1e-6)
            axes[1, 0].hist(spine_energy_log, bins=50, color='green', alpha=0.7, edgecolor='black')
            axes[1, 0].axvline(spine_energy_log.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.exp(spine_energy_log.mean()):.3f}')
            axes[1, 0].axvline(np.median(spine_energy_log), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.exp(np.median(spine_energy_log)):.3f}')
            axes[1, 0].set_xlabel('Log(Spine Curve Energy)')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].set_title('Body Bend Distribution (Spine Curve Energy)')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Turn Rate Distribution (per track or overall)
    if 'track_id' in trajectories_df.columns and trajectories_df['track_id'].nunique() > 1:
        track_turn_rates = []
        for track_id in trajectories_df['track_id'].unique():
            track_data = trajectories_df[trajectories_df['track_id'] == track_id]
            if len(track_data) > 0:
                total_time = track_data['time'].max() - track_data['time'].min()
                n_turns = track_data['is_turn'].sum()
                if total_time > 0:
                    turn_rate = (n_turns / total_time) * 60  # turns per minute
                    track_turn_rates.append(turn_rate)
        
        if len(track_turn_rates) > 0:
            axes[1, 1].hist(track_turn_rates, bins=20, color='purple', alpha=0.7, edgecolor='black')
            axes[1, 1].axvline(np.mean(track_turn_rates), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(track_turn_rates):.2f} turns/min')
            axes[1, 1].axvline(np.median(track_turn_rates), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(track_turn_rates):.2f} turns/min')
            axes[1, 1].set_xlabel('Turn Rate (turns/min)')
            axes[1, 1].set_ylabel('Count (tracks)')
            axes[1, 1].set_title('Turn Rate Distribution (per track)')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
    else:
        # Overall turn rate
        total_time = trajectories_df['time'].max() - trajectories_df['time'].min()
        n_turns = trajectories_df['is_turn'].sum()
        if total_time > 0:
            overall_rate = (n_turns / total_time) * 60
            axes[1, 1].text(0.5, 0.5, f'Overall Turn Rate: {overall_rate:.2f} turns/min', 
                           transform=axes[1, 1].transAxes, ha='center', va='center', fontsize=14)
            axes[1, 1].set_title('Turn Rate (overall)')
            axes[1, 1].axis('off')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    output_path = output_dir / 'behavioral_event_distributions.png'
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved distribution plots to {output_path}")


def compute_stimulus_locked_turn_rate(trajectories_df: pd.DataFrame, cycles: List[Dict], 
                                      bin_width: float = 0.5, fps: float = 10.0,
                                      analysis_window: Tuple[float, float] = (-3.0, 8.0)) -> Dict:
    """
    Compute stimulus-locked turn rate analysis with cycle-by-cycle tracking.
    
    For each cycle:
    - Extract tracks that have data in that cycle
    - Create 0.5 second bins relative to cycle onset
    - For each bin, sample turn rate from each track
    - Average across tracks with confidence intervals
    
    Then aggregate all cycles to get the salience envelope.
    
    Parameters
    ----------
    trajectories_df : pd.DataFrame
        Trajectory data with track_id, time, is_turn columns
    cycles : List[Dict]
        List of cycle dictionaries with time_start, time_end
    bin_width : float
        Bin width in seconds (default 0.5s)
    fps : float
        Frame rate in Hz
    analysis_window : Tuple[float, float]
        Time window relative to cycle onset (default -3s to +8s)
    
    Returns
    -------
    dict
        Dictionary with per-cycle and aggregated turn rate data
    """
    import time
    
    t_min, t_max = analysis_window
    
    # Create time bins relative to cycle onset
    bin_edges = np.arange(t_min, t_max + bin_width, bin_width)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Storage for per-cycle results
    cycle_results = []
    all_cycle_turn_rates = []  # For salience envelope
    
    total_cycles = len(cycles)
    print(f"  Processing {total_cycles} cycles...")
    start_time = time.time()
    
    for cycle_idx, cycle in enumerate(cycles):
        cycle_start_time_wall = time.time()
        cycle_num = cycle['cycle_num']
        cycle_start_time = cycle['time_start']
        cycle_end_time = cycle['time_end']
        
        # Progress update every cycle or every 5 seconds
        cycle_elapsed = time.time() - cycle_start_time_wall if cycle_idx > 0 else 0
        elapsed = time.time() - start_time
        
        if cycle_idx == 0 or cycle_idx % max(1, total_cycles // 10) == 0 or cycle_elapsed > 5:
            print(f"    Cycle {cycle_num}/{total_cycles} ({cycle_idx+1}/{total_cycles}) - Elapsed: {elapsed:.1f}s", end='')
            if cycle_idx > 0:
                avg_time_per_cycle = elapsed / (cycle_idx + 1)
                remaining_cycles = total_cycles - (cycle_idx + 1)
                est_remaining = avg_time_per_cycle * remaining_cycles
                print(f" - Est. remaining: {est_remaining:.1f}s - Avg: {avg_time_per_cycle:.2f}s/cycle")
            else:
                print()
        
        # Find tracks that have data in this cycle - optimized with vectorized operations
        # Use query for faster filtering
        try:
            cycle_tracks_df = trajectories_df[
                (trajectories_df['time'] >= cycle_start_time) & 
                (trajectories_df['time'] <= cycle_end_time)
            ].copy()
        except:
            continue
        
        if len(cycle_tracks_df) == 0:
            continue
        
        # Compute relative time for each data point
        cycle_tracks_df['time_rel_onset'] = cycle_tracks_df['time'] - cycle_start_time
        
        # Filter to analysis window
        cycle_tracks_df = cycle_tracks_df[
            (cycle_tracks_df['time_rel_onset'] >= t_min) & 
            (cycle_tracks_df['time_rel_onset'] <= t_max)
        ]
        
        if len(cycle_tracks_df) == 0:
            continue
        
        # Bin data points using searchsorted for speed
        cycle_tracks_df['time_bin'] = np.searchsorted(bin_edges[1:], cycle_tracks_df['time_rel_onset'].values)
        cycle_tracks_df['time_bin'] = np.clip(cycle_tracks_df['time_bin'], 0, len(bin_centers) - 1)
        
        # Vectorized computation: group by time_bin and track_id
        n_bins = len(bin_centers)
        grouped = cycle_tracks_df.groupby(['time_bin', 'track_id'], observed=True).agg({
            'is_turn': 'sum',
            'time': 'count'  # Count frames per track per bin
        }).reset_index()
        grouped['turn_rate'] = (grouped['is_turn'] / (grouped['time'] / fps)) * 60
        
        # Build bin_turn_rates using vectorized operations
        bin_turn_rates = [[] for _ in range(n_bins)]
        bin_turn_rates_dict = {i: [] for i in range(n_bins)}
        
        for _, row in grouped.iterrows():
            bin_idx = int(row['time_bin'])
            if 0 <= bin_idx < n_bins:
                rate = row['turn_rate']
                bin_turn_rates[bin_idx].append(rate)
                bin_turn_rates_dict[bin_idx].append(rate)
        
        # Aggregate: mean turn rate per bin across tracks
        mean_turn_rates = []
        sem_turn_rates = []
        ci_lower = []
        ci_upper = []
        n_tracks_per_bin = []
        
        for bin_idx, track_rates in enumerate(bin_turn_rates):
            if len(track_rates) > 0:
                mean_rate = np.mean(track_rates)
                sem_rate = stats.sem(track_rates) if len(track_rates) > 1 else 0
                # 95% CI
                ci = stats.t.interval(0.95, len(track_rates) - 1, loc=mean_rate, scale=sem_rate) if len(track_rates) > 1 else (mean_rate, mean_rate)
                
                mean_turn_rates.append(mean_rate)
                sem_turn_rates.append(sem_rate)
                ci_lower.append(ci[0])
                ci_upper.append(ci[1])
                n_tracks_per_bin.append(len(track_rates))
            else:
                mean_turn_rates.append(np.nan)
                sem_turn_rates.append(np.nan)
                ci_lower.append(np.nan)
                ci_upper.append(np.nan)
                n_tracks_per_bin.append(0)
        
        cycle_result = {
            'cycle_num': cycle_num,
            'bin_centers': bin_centers.tolist(),
            'mean_turn_rates': mean_turn_rates,
            'sem_turn_rates': sem_turn_rates,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n_tracks_per_bin': n_tracks_per_bin
        }
        cycle_results.append(cycle_result)
        
        # Store for salience envelope
        all_cycle_turn_rates.append({
            'cycle_num': cycle_num,
            'bin_centers': bin_centers,
            'turn_rates': bin_turn_rates_dict
        })
    
    print(f"  ✓ Processed {len(cycle_results)} cycles in {time.time() - start_time:.1f}s")
    
    # Aggregate across all cycles to get salience envelope
    print("  Aggregating cycles to compute salience envelope...")
    agg_start_time = time.time()
    
    salience_envelope = {
        'bin_centers': bin_centers.tolist(),
        'mean_turn_rates': [],
        'sem_turn_rates': [],
        'ci_lower': [],
        'ci_upper': [],
        'n_samples_per_bin': []
    }
    
    n_bins = len(bin_centers)
    print(f"    Aggregating {n_bins} bins across {len(all_cycle_turn_rates)} cycles...")
    
    # Optimize: pre-compute pooled rates for all bins
    pooled_rates_by_bin = {i: [] for i in range(n_bins)}
    for cycle_data in all_cycle_turn_rates:
        for bin_idx, rates in cycle_data['turn_rates'].items():
            if bin_idx in pooled_rates_by_bin:
                pooled_rates_by_bin[bin_idx].extend(rates)
    
    for bin_idx in range(n_bins):
        if bin_idx % 5 == 0 or bin_idx == n_bins - 1:
            print(f"    Processing bin {bin_idx+1}/{n_bins}...", end='\r')
        
        pooled_rates = pooled_rates_by_bin[bin_idx]
        
        if len(pooled_rates) > 0:
            mean_rate = np.mean(pooled_rates)
            sem_rate = stats.sem(pooled_rates) if len(pooled_rates) > 1 else 0
            ci = stats.t.interval(0.95, len(pooled_rates) - 1, loc=mean_rate, scale=sem_rate) if len(pooled_rates) > 1 else (mean_rate, mean_rate)
            
            salience_envelope['mean_turn_rates'].append(mean_rate)
            salience_envelope['sem_turn_rates'].append(sem_rate)
            salience_envelope['ci_lower'].append(ci[0])
            salience_envelope['ci_upper'].append(ci[1])
            salience_envelope['n_samples_per_bin'].append(len(pooled_rates))
        else:
            salience_envelope['mean_turn_rates'].append(np.nan)
            salience_envelope['sem_turn_rates'].append(np.nan)
            salience_envelope['ci_lower'].append(np.nan)
            salience_envelope['ci_upper'].append(np.nan)
            salience_envelope['n_samples_per_bin'].append(0)
    
    print(f"  ✓ Aggregated salience envelope in {time.time() - agg_start_time:.1f}s")
    
    return {
        'per_cycle_results': cycle_results,
        'salience_envelope': salience_envelope,
        'n_cycles': len(cycle_results)
    }


def plot_stimulus_locked_turn_rate(turn_rate_analysis: Dict, output_dir: Path, 
                                   analysis_window: Tuple[float, float] = (-3.0, 8.0)):
    """
    Plot stimulus-locked turn rate analysis.
    
    Creates:
    1. Individual cycle plots (first few cycles)
    2. Salience envelope (aggregated across all cycles)
    """
    t_min, t_max = analysis_window
    
    # Plot 1: Salience envelope (aggregated)
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Salience envelope
    salience = turn_rate_analysis['salience_envelope']
    bin_centers = np.array(salience['bin_centers'])
    mean_rates = np.array(salience['mean_turn_rates'])
    ci_lower = np.array(salience['ci_lower'])
    ci_upper = np.array(salience['ci_upper'])
    
    # Filter out NaN values
    valid_mask = ~np.isnan(mean_rates)
    bin_centers_valid = bin_centers[valid_mask]
    mean_rates_valid = mean_rates[valid_mask]
    ci_lower_valid = ci_lower[valid_mask]
    ci_upper_valid = ci_upper[valid_mask]
    
    axes[0].plot(bin_centers_valid, mean_rates_valid, 'k-', linewidth=2, label='Mean Turn Rate')
    axes[0].fill_between(bin_centers_valid, ci_lower_valid, ci_upper_valid, 
                         alpha=0.3, color='blue', label='95% CI')
    axes[0].axvline(0, color='red', linestyle='--', linewidth=1.5, label='Stimulus Onset')
    axes[0].axvspan(0, 10, color='red', alpha=0.1, label='Stimulus Duration')
    axes[0].set_xlabel('Time relative to stimulus onset (s)')
    axes[0].set_ylabel('Turn Rate (turns/min)')
    axes[0].set_title(f'Salience Envelope: Stimulus-Locked Turn Rate (Aggregated across {turn_rate_analysis["n_cycles"]} cycles)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_xlim(t_min, t_max)
    
    # Plot 2: Individual cycles (first 6 cycles)
    per_cycle = turn_rate_analysis['per_cycle_results']
    n_cycles_to_plot = min(6, len(per_cycle))
    
    colors = plt.cm.viridis(np.linspace(0, 1, n_cycles_to_plot))
    
    for i, cycle_result in enumerate(per_cycle[:n_cycles_to_plot]):
        bin_centers_cycle = np.array(cycle_result['bin_centers'])
        mean_rates_cycle = np.array(cycle_result['mean_turn_rates'])
        
        valid_mask = ~np.isnan(mean_rates_cycle)
        bin_centers_valid = bin_centers_cycle[valid_mask]
        mean_rates_valid = mean_rates_cycle[valid_mask]
        
        axes[1].plot(bin_centers_valid, mean_rates_valid, 
                    color=colors[i], alpha=0.7, linewidth=1.5, 
                    label=f'Cycle {cycle_result["cycle_num"]}')
    
    axes[1].axvline(0, color='red', linestyle='--', linewidth=1.5, label='Stimulus Onset')
    axes[1].axvspan(0, 10, color='red', alpha=0.1, label='Stimulus Duration')
    axes[1].set_xlabel('Time relative to stimulus onset (s)')
    axes[1].set_ylabel('Turn Rate (turns/min)')
    axes[1].set_title(f'Individual Cycles (first {n_cycles_to_plot} cycles)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper right', fontsize=8)
    axes[1].set_xlim(t_min, t_max)
    
    plt.tight_layout()
    output_path = output_dir / 'stimulus_locked_turn_rate.png'
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved stimulus-locked turn rate plot to {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize behavioral events and stimulus-locked turn rate')
    parser.add_argument('--trajectories-file', type=str,
                       default='data/engineered_tier2/GMR61_tier2_trajectories.csv',
                       help='Path to trajectories CSV file')
    parser.add_argument('--events-file', type=str, default=None,
                       help='Path to events CSV file (for track_id if not in trajectories)')
    parser.add_argument('--h5-file', type=str,
                       default='/Users/gilraitses/mechanosensation/h5tests/GMR61_tier2_complete.h5',
                       help='Path to H5 file for stimulus cycle extraction')
    parser.add_argument('--output-dir', type=str,
                       default='output/visualizations',
                       help='Output directory for plots')
    parser.add_argument('--bin-width', type=float, default=0.5,
                       help='Bin width in seconds for stimulus-locked analysis')
    parser.add_argument('--analysis-window', type=float, nargs=2, default=[-3.0, 8.0],
                       help='Analysis window relative to stimulus onset (e.g., -3.0 8.0)')
    parser.add_argument('--fps', type=float, default=10.0,
                       help='Frame rate in Hz')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("BEHAVIORAL EVENT VISUALIZATION")
    print("="*80)
    
    # Load trajectory data
    print(f"\nLoading trajectory data from {args.trajectories_file}...")
    trajectories_df = pd.read_csv(args.trajectories_file)
    print(f"  Loaded {len(trajectories_df):,} trajectory points")
    
    # Check for track_id or get from events file
    if 'track_id' not in trajectories_df.columns:
        if args.events_file and Path(args.events_file).exists():
            print(f"  Loading track_id from events file: {args.events_file}")
            events_df = pd.read_csv(args.events_file)
            # Merge track_id using time as key (approximate)
            # Get unique track_ids and map them
            if 'track_id' in events_df.columns and 'time' in events_df.columns:
                # Create a mapping from time to track_id (use nearest time match)
                track_id_map = events_df.set_index('time')['track_id'].to_dict()
                # For each trajectory time, find nearest event time
                trajectories_df['track_id'] = trajectories_df['time'].apply(
                    lambda t: track_id_map.get(t, track_id_map.get(min(track_id_map.keys(), key=lambda k: abs(k - t)), 1)))
                print(f"  Mapped track_id from events file: {trajectories_df['track_id'].nunique()} tracks")
            else:
                print("  Warning: track_id not found in events file, using single track")
                trajectories_df['track_id'] = 1
        else:
            print("  Warning: track_id not found and no events file provided, using single track")
            trajectories_df['track_id'] = 1
    else:
        print(f"  Tracks: {trajectories_df['track_id'].nunique()}")
    
    # Load stimulus cycles
    print(f"\nLoading stimulus cycles from {args.h5_file}...")
    cycles, led1_data = load_stimulus_cycles(args.h5_file, fps=args.fps)
    print(f"  Found {len(cycles)} stimulus cycles")
    
    # Plot event distributions
    print("\n" + "="*80)
    print("PLOTTING EVENT DISTRIBUTIONS")
    print("="*80)
    plot_event_distributions(trajectories_df, output_dir)
    
    # Compute stimulus-locked turn rate
    print("\n" + "="*80)
    print("COMPUTING STIMULUS-LOCKED TURN RATE")
    print("="*80)
    turn_rate_analysis = compute_stimulus_locked_turn_rate(
        trajectories_df, cycles, 
        bin_width=args.bin_width,
        fps=args.fps,
        analysis_window=tuple(args.analysis_window)
    )
    print(f"  Processed {turn_rate_analysis['n_cycles']} cycles")
    
    # Plot stimulus-locked turn rate
    print("\n" + "="*80)
    print("PLOTTING STIMULUS-LOCKED TURN RATE")
    print("="*80)
    plot_stimulus_locked_turn_rate(turn_rate_analysis, output_dir, 
                                   analysis_window=tuple(args.analysis_window))
    
    # Save analysis results
    results_path = output_dir / 'stimulus_locked_turn_rate_results.json'
    # Convert numpy arrays to lists for JSON serialization
    results_for_json = {
        'n_cycles': turn_rate_analysis['n_cycles'],
        'salience_envelope': {
            k: v if not isinstance(v, np.ndarray) else v.tolist() 
            for k, v in turn_rate_analysis['salience_envelope'].items()
        },
        'per_cycle_results': [
            {
                k: v if not isinstance(v, np.ndarray) else v.tolist()
                for k, v in cycle.items()
            }
            for cycle in turn_rate_analysis['per_cycle_results']
        ]
    }
    
    with open(results_path, 'w') as f:
        json.dump(results_for_json, f, indent=2)
    print(f"✓ Saved analysis results to {results_path}")
    
    print("\n" + "="*80)
    print("✓ VISUALIZATION COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()

