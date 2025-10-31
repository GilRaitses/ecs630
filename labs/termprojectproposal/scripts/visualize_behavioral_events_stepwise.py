#!/usr/bin/env python3
"""
Stepwise visualization - uses pre-aggregated events CSV for speed.

Step 1: Load data and verify
Step 2: Extract cycles from H5
Step 3: Compute stimulus-locked turn rate (using events CSV)
Step 4: Plot distributions
Step 5: Plot stimulus-locked analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import h5py
from scipy import stats
from typing import Dict, List, Tuple
import json
import time

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150


def step1_load_data(events_file: str) -> pd.DataFrame:
    """Step 1: Load and verify events data."""
    print("\n" + "="*80)
    print("STEP 1: LOADING DATA")
    print("="*80)
    
    print(f"Loading events from {events_file}...")
    events_df = pd.read_csv(events_file)
    print(f"  ✓ Loaded {len(events_df):,} event records")
    print(f"  ✓ Tracks: {events_df['track_id'].nunique()}")
    print(f"  ✓ Turn events: {events_df['is_turn'].sum():,}")
    print(f"  ✓ Time range: {events_df['time'].min():.1f}s - {events_df['time'].max():.1f}s")
    
    return events_df


def step2_extract_cycles(h5_file: str, fps: float = 10.0) -> Tuple[List[Dict], np.ndarray]:
    """Step 2: Extract stimulus cycles from H5."""
    print("\n" + "="*80)
    print("STEP 2: EXTRACTING STIMULUS CYCLES")
    print("="*80)
    
    cycles = []
    
    print(f"Loading cycles from {h5_file}...")
    with h5py.File(h5_file, 'r') as f:
        onset_frames = f['stimulus']['onset_frames'][:]
        onset_frames = np.sort(onset_frames)
        
        if 'global_quantities' in f and 'led1Val' in f['global_quantities']:
            gq_item = f['global_quantities']['led1Val']
            if isinstance(gq_item, h5py.Group) and 'yData' in gq_item:
                led1_data = gq_item['yData'][:]
            else:
                led1_data = gq_item[:]
        else:
            raise ValueError("LED1 data not found")
        
        print(f"  Found {len(onset_frames)} stimulus onsets")
        
        for i, onset_frame in enumerate(onset_frames):
            onset_frame_int = int(onset_frame)
            onset_time = onset_frame_int / fps
            
            # Find drop point (simplified - use next onset or end)
            if i < len(onset_frames) - 1:
                drop_frame_int = int(onset_frames[i+1])
                drop_time = drop_frame_int / fps
            else:
                drop_frame_int = len(led1_data) - 1
                drop_time = drop_frame_int / fps
            
            cycle = {
                'cycle_num': i + 1,
                'frame_start': onset_frame_int,
                'frame_end': drop_frame_int,
                'time_start': onset_time,
                'time_end': drop_time,
            }
            cycles.append(cycle)
    
    print(f"  ✓ Extracted {len(cycles)} cycles")
    return cycles, led1_data


def step3_compute_stimulus_locked_turn_rate(events_df: pd.DataFrame, cycles: List[Dict],
                                            bin_width: float = 0.5,
                                            analysis_window: Tuple[float, float] = (-3.0, 8.0)) -> Dict:
    """Step 3: Compute stimulus-locked turn rate using events CSV."""
    print("\n" + "="*80)
    print("STEP 3: COMPUTING STIMULUS-LOCKED TURN RATE")
    print("="*80)
    
    t_min, t_max = analysis_window
    bin_edges = np.arange(t_min, t_max + bin_width, bin_width)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    cycle_results = []
    all_cycle_turn_rates = {}
    
    print(f"Processing {len(cycles)} cycles...")
    start_time = time.time()
    
    for cycle_idx, cycle in enumerate(cycles):
        if (cycle_idx + 1) % 5 == 0 or cycle_idx == 0:
            elapsed = time.time() - start_time
            print(f"  Cycle {cycle_idx + 1}/{len(cycles)} - Elapsed: {elapsed:.1f}s")
        
        cycle_start = cycle['time_start']
        cycle_end = cycle['time_end']
        
        # Filter events in this cycle
        cycle_events = events_df[
            (events_df['time'] >= cycle_start) & 
            (events_df['time'] <= cycle_end)
        ].copy()
        
        if len(cycle_events) == 0:
            continue
        
        # Compute relative time
        cycle_events['time_rel_onset'] = cycle_events['time'] - cycle_start
        
        # Filter to analysis window
        cycle_events = cycle_events[
            (cycle_events['time_rel_onset'] >= t_min) & 
            (cycle_events['time_rel_onset'] <= t_max)
        ]
        
        if len(cycle_events) == 0:
            continue
        
        # Bin using searchsorted (fast)
        cycle_events['time_bin'] = np.searchsorted(bin_edges[1:], cycle_events['time_rel_onset'].values)
        cycle_events['time_bin'] = np.clip(cycle_events['time_bin'], 0, len(bin_centers) - 1)
        
        # Group by bin and track to compute turn rate
        # events CSV already has 50ms bins, so each row represents a bin
        grouped = cycle_events.groupby(['time_bin', 'track_id'], observed=True).agg({
            'is_turn': 'sum',  # Count turns in this bin-track combination
            'time': 'count'    # Count bins (each bin is already 50ms)
        }).reset_index()
        
        # Turn rate = (turns / time) * 60
        # time is in number of 50ms bins, so duration = time * 0.05 seconds
        grouped['turn_rate'] = (grouped['is_turn'] / (grouped['time'] * 0.05)) * 60
        
        # Aggregate per bin across tracks
        bin_stats = grouped.groupby('time_bin', observed=True).agg({
            'turn_rate': ['mean', 'std', 'count']
        }).reset_index()
        bin_stats.columns = ['time_bin', 'mean_rate', 'std_rate', 'n_tracks']
        
        # Fill in all bins
        mean_rates = [np.nan] * len(bin_centers)
        sem_rates = [np.nan] * len(bin_centers)
        ci_lower = [np.nan] * len(bin_centers)
        ci_upper = [np.nan] * len(bin_centers)
        n_tracks_per_bin = [0] * len(bin_centers)
        
        for _, row in bin_stats.iterrows():
            bin_idx = int(row['time_bin'])
            if 0 <= bin_idx < len(bin_centers):
                mean_rates[bin_idx] = row['mean_rate']
                std_rate = row['std_rate']
                n_tracks = int(row['n_tracks'])
                sem_rates[bin_idx] = std_rate / np.sqrt(n_tracks) if n_tracks > 1 else 0
                n_tracks_per_bin[bin_idx] = n_tracks
                
                if n_tracks > 1:
                    ci = stats.t.interval(0.95, n_tracks - 1, loc=row['mean_rate'], scale=sem_rates[bin_idx])
                    ci_lower[bin_idx] = ci[0]
                    ci_upper[bin_idx] = ci[1]
                else:
                    ci_lower[bin_idx] = row['mean_rate']
                    ci_upper[bin_idx] = row['mean_rate']
        
        cycle_result = {
            'cycle_num': cycle['cycle_num'],
            'bin_centers': bin_centers.tolist(),
            'mean_turn_rates': mean_rates,
            'sem_turn_rates': sem_rates,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n_tracks_per_bin': n_tracks_per_bin
        }
        cycle_results.append(cycle_result)
        
        # Store raw rates for salience envelope
        for _, row in grouped.iterrows():
            bin_idx = int(row['time_bin'])
            if bin_idx not in all_cycle_turn_rates:
                all_cycle_turn_rates[bin_idx] = []
            all_cycle_turn_rates[bin_idx].append(row['turn_rate'])
    
    print(f"  ✓ Processed {len(cycle_results)} cycles in {time.time() - start_time:.1f}s")
    
    # Compute salience envelope
    print("  Computing salience envelope...")
    salience_envelope = {
        'bin_centers': bin_centers.tolist(),
        'mean_turn_rates': [],
        'sem_turn_rates': [],
        'ci_lower': [],
        'ci_upper': [],
        'n_samples_per_bin': []
    }
    
    for bin_idx in range(len(bin_centers)):
        pooled_rates = all_cycle_turn_rates.get(bin_idx, [])
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
    
    print(f"  ✓ Salience envelope computed")
    
    return {
        'per_cycle_results': cycle_results,
        'salience_envelope': salience_envelope,
        'n_cycles': len(cycle_results)
    }


def step4_plot_distributions(events_df: pd.DataFrame, output_dir: Path):
    """Step 4: Plot event distributions."""
    print("\n" + "="*80)
    print("STEP 4: PLOTTING DISTRIBUTIONS")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Behavioral Event Distributions', fontsize=16, fontweight='bold')
    
    # 1. Turn Duration (from events)
    if 'turn_duration' in events_df.columns:
        turn_durations = events_df[events_df['turn_duration'] > 0]['turn_duration'].unique()
        if len(turn_durations) > 0:
            axes[0, 0].hist(turn_durations, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
            axes[0, 0].axvline(np.mean(turn_durations), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(turn_durations):.3f}s')
            axes[0, 0].set_xlabel('Turn Duration (s)')
            axes[0, 0].set_ylabel('Count')
            axes[0, 0].set_title('Turn Duration Distribution')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Pause Duration
    if 'pause_duration' in events_df.columns:
        pause_durations = events_df[events_df['pause_duration'] > 0]['pause_duration'].unique()
        if len(pause_durations) > 0:
            axes[0, 1].hist(pause_durations, bins=30, color='coral', alpha=0.7, edgecolor='black')
            axes[0, 1].axvline(np.mean(pause_durations), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(pause_durations):.3f}s')
            axes[0, 1].set_xlabel('Pause Duration (s)')
            axes[0, 1].set_ylabel('Count')
            axes[0, 1].set_title('Pause Duration Distribution')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Spine Curve Energy
    if 'spine_curve_energy' in events_df.columns:
        spine_energy = events_df['spine_curve_energy'].values
        spine_energy = spine_energy[spine_energy > 0]
        if len(spine_energy) > 0:
            spine_energy_log = np.log(spine_energy + 1e-6)
            axes[1, 0].hist(spine_energy_log, bins=50, color='green', alpha=0.7, edgecolor='black')
            axes[1, 0].axvline(spine_energy_log.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.exp(spine_energy_log.mean()):.3f}')
            axes[1, 0].set_xlabel('Log(Spine Curve Energy)')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].set_title('Body Bend Distribution')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Turn Rate per Track
    track_rates = []
    for track_id in events_df['track_id'].unique():
        track_data = events_df[events_df['track_id'] == track_id]
        total_time = track_data['time'].max() - track_data['time'].min()
        n_turns = track_data['is_turn'].sum()
        if total_time > 0:
            turn_rate = (n_turns / total_time) * 60
            track_rates.append(turn_rate)
    
    if len(track_rates) > 0:
        axes[1, 1].hist(track_rates, bins=20, color='purple', alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(np.mean(track_rates), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(track_rates):.2f} turns/min')
        axes[1, 1].set_xlabel('Turn Rate (turns/min)')
        axes[1, 1].set_ylabel('Count (tracks)')
        axes[1, 1].set_title('Turn Rate Distribution (per track)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    output_path = output_dir / 'behavioral_event_distributions.png'
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved to {output_path}")


def step5_plot_stimulus_locked(turn_rate_analysis: Dict, output_dir: Path,
                               analysis_window: Tuple[float, float] = (-3.0, 8.0)):
    """Step 5: Plot stimulus-locked turn rate."""
    print("\n" + "="*80)
    print("STEP 5: PLOTTING STIMULUS-LOCKED TURN RATE")
    print("="*80)
    
    t_min, t_max = analysis_window
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Salience envelope
    salience = turn_rate_analysis['salience_envelope']
    bin_centers = np.array(salience['bin_centers'])
    mean_rates = np.array(salience['mean_turn_rates'])
    ci_lower = np.array(salience['ci_lower'])
    ci_upper = np.array(salience['ci_upper'])
    
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
    
    # Individual cycles
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
    print(f"  ✓ Saved to {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Stepwise behavioral event visualization')
    parser.add_argument('--events-file', type=str,
                       default='data/engineered_tier2/GMR61_tier2_events.csv',
                       help='Path to events CSV file')
    parser.add_argument('--h5-file', type=str,
                       default='/Users/gilraitses/mechanosensation/h5tests/GMR61_tier2_complete.h5',
                       help='Path to H5 file for stimulus cycles')
    parser.add_argument('--output-dir', type=str,
                       default='output/visualizations',
                       help='Output directory')
    parser.add_argument('--bin-width', type=float, default=0.5,
                       help='Bin width in seconds')
    parser.add_argument('--analysis-window', type=float, nargs=2, default=[-3.0, 8.0],
                       help='Analysis window')
    parser.add_argument('--fps', type=float, default=10.0,
                       help='Frame rate')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("STEPWISE BEHAVIORAL EVENT VISUALIZATION")
    print("="*80)
    print("Using pre-aggregated events CSV for speed")
    
    # Execute steps
    events_df = step1_load_data(args.events_file)
    cycles, led1_data = step2_extract_cycles(args.h5_file, args.fps)
    turn_rate_analysis = step3_compute_stimulus_locked_turn_rate(
        events_df, cycles, args.bin_width, tuple(args.analysis_window)
    )
    step4_plot_distributions(events_df, output_dir)
    step5_plot_stimulus_locked(turn_rate_analysis, output_dir, tuple(args.analysis_window))
    
    # Save results
    results_path = output_dir / 'stimulus_locked_turn_rate_results.json'
    results_for_json = {
        'n_cycles': turn_rate_analysis['n_cycles'],
        'salience_envelope': {
            k: v if not isinstance(v, np.ndarray) else v.tolist() 
            for k, v in turn_rate_analysis['salience_envelope'].items()
        }
    }
    
    with open(results_path, 'w') as f:
        json.dump(results_for_json, f, indent=2)
    print(f"\n✓ Saved results to {results_path}")
    
    print("\n" + "="*80)
    print("✓ ALL STEPS COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()



