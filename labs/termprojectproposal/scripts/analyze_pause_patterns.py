#!/usr/bin/env python3
"""
Analyze pause patterns: relationship to stimulus and pause frequency.

Creates:
- Pause frequency statistics
- Peri-stimulus pause probability (PSTH)
- Pause duration distributions
- Pause vs stimulus timing analysis
"""

import sys
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from cinnamoroll_palette import CINNAMOROLL_COLORS, CINNAMOROLL_PALETTE, setup_cinnamoroll_style

def analyze_pause_frequency(trajectory_df, events_df):
    """
    Analyze overall pause frequency and duration statistics.
    
    Parameters
    ----------
    trajectory_df : pd.DataFrame
        Full-resolution trajectory data
    events_df : pd.DataFrame
        Event records (binned)
    
    Returns
    -------
    dict
        Statistics dictionary
    """
    stats_dict = {}
    
    # Overall pause statistics
    total_frames = len(trajectory_df)
    total_time = trajectory_df['time'].max() - trajectory_df['time'].min()
    
    pause_frames = trajectory_df['is_pause'].sum()
    pause_fraction = pause_frames / total_frames if total_frames > 0 else 0
    
    # Pause events (unique pauses)
    pause_durations = trajectory_df[trajectory_df['pause_duration'] > 0]['pause_duration'].unique()
    n_pause_events = len(pause_durations)
    
    if n_pause_events > 0:
        mean_pause_duration = np.mean(pause_durations)
        median_pause_duration = np.median(pause_durations)
        pause_rate = n_pause_events / (total_time / 60.0)  # pauses per minute
    else:
        mean_pause_duration = 0
        median_pause_duration = 0
        pause_rate = 0
    
    stats_dict['pause_frequency'] = {
        'total_frames': int(total_frames),
        'pause_frames': int(pause_frames),
        'pause_fraction': float(pause_fraction),
        'n_pause_events': int(n_pause_events),
        'pause_rate_per_min': float(pause_rate),
        'mean_pause_duration_s': float(mean_pause_duration),
        'median_pause_duration_s': float(median_pause_duration),
        'pause_duration_range_s': [float(np.min(pause_durations)), float(np.max(pause_durations))] if len(pause_durations) > 0 else [0, 0]
    }
    
    # Pause duration distribution
    if len(pause_durations) > 0:
        stats_dict['pause_duration_distribution'] = {
            'percentiles': {
                'p10': float(np.percentile(pause_durations, 10)),
                'p25': float(np.percentile(pause_durations, 25)),
                'p50': float(np.percentile(pause_durations, 50)),
                'p75': float(np.percentile(pause_durations, 75)),
                'p90': float(np.percentile(pause_durations, 90))
            }
        }
    
    return stats_dict

def analyze_pause_stimulus_relationship(trajectory_df, events_df, analysis_window=(-3.0, 8.0)):
    """
    Analyze pause patterns relative to stimulus onsets.
    
    Parameters
    ----------
    trajectory_df : pd.DataFrame
        Full-resolution trajectory data
    events_df : pd.DataFrame
        Event records
    analysis_window : tuple
        Time window relative to stimulus onset (before, after) in seconds
    
    Returns
    -------
    dict
        Analysis results
    """
    results = {}
    
    # Get stimulus onsets
    if 'stimulus_onset' in trajectory_df.columns:
        onset_mask = trajectory_df['stimulus_onset'] == True
        onset_times = trajectory_df[onset_mask]['time'].values
    elif 'time_since_stimulus' in trajectory_df.columns:
        # Detect onsets as transitions from large to small time_since_stimulus
        time_since = trajectory_df['time_since_stimulus'].values
        onset_mask = np.abs(np.diff(time_since, prepend=time_since[0])) > 50
        onset_times = trajectory_df[onset_mask]['time'].values
    else:
        print("  Warning: No stimulus onset data found")
        return results
    
    if len(onset_times) == 0:
        print("  Warning: No stimulus onsets detected")
        return results
    
    t_min, t_max = analysis_window
    
    # Peri-stimulus pause probability (PSTH)
    bin_width = 0.1  # 100ms bins
    bins = np.arange(t_min, t_max, bin_width)
    pause_counts = np.zeros(len(bins) - 1)
    total_counts = np.zeros(len(bins) - 1)
    
    for onset_time in onset_times:
        # Extract data around this onset
        window_start = onset_time + t_min
        window_end = onset_time + t_max
        
        window_data = trajectory_df[
            (trajectory_df['time'] >= window_start) & 
            (trajectory_df['time'] <= window_end)
        ].copy()
        
        if len(window_data) > 0:
            # Convert to time relative to onset
            window_data['time_rel_onset'] = window_data['time'] - onset_time
            
            # Bin relative to onset
            window_data['time_bin'] = pd.cut(
                window_data['time_rel_onset'], 
                bins=bins, 
                labels=False
            )
            
            # Count pauses in each bin
            binned = window_data.groupby('time_bin').agg({
                'is_pause': 'sum',
                'time': 'count'
            })
            
            for bin_idx in binned.index:
                if not np.isnan(bin_idx):
                    bin_idx = int(bin_idx)
                    if 0 <= bin_idx < len(pause_counts):
                        pause_counts[bin_idx] += binned.loc[bin_idx, 'is_pause']
                        total_counts[bin_idx] += binned.loc[bin_idx, 'time']
    
    # Compute pause probability
    pause_probability = pause_counts / (total_counts + 1e-6)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    results['peri_stimulus_pause'] = {
        'time_rel_onset': bin_centers.tolist(),
        'pause_probability': pause_probability.tolist(),
        'pause_counts': pause_counts.tolist(),
        'total_counts': total_counts.tolist()
    }
    
    # Compare pause probability during vs outside stimulus
    if 'stimulus_on' in trajectory_df.columns:
        during_stimulus = trajectory_df[trajectory_df['stimulus_on'] == True]
        outside_stimulus = trajectory_df[trajectory_df['stimulus_on'] == False]
        
        pause_prob_during = during_stimulus['is_pause'].mean() if len(during_stimulus) > 0 else 0
        pause_prob_outside = outside_stimulus['is_pause'].mean() if len(outside_stimulus) > 0 else 0
        
        results['pause_stimulus_comparison'] = {
            'pause_prob_during_stimulus': float(pause_prob_during),
            'pause_prob_outside_stimulus': float(pause_prob_outside),
            'ratio': float(pause_prob_during / (pause_prob_outside + 1e-6))
        }
    
    return results

def plot_pause_analysis(trajectory_df, events_df, analysis_results, output_dir):
    """
    Create visualization plots for pause analysis.
    
    Parameters
    ----------
    trajectory_df : pd.DataFrame
        Trajectory data
    events_df : pd.DataFrame
        Event records
    analysis_results : dict
        Analysis results dictionary
    output_dir : Path
        Output directory for plots
    """
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: Peri-stimulus pause probability (PSTH)
    ax1 = plt.subplot(3, 2, 1)
    if 'peri_stimulus_pause' in analysis_results:
        psth = analysis_results['peri_stimulus_pause']
        ax1.plot(psth['time_rel_onset'], psth['pause_probability'], 'b-', linewidth=2)
        ax1.axvline(0, color='r', linestyle='--', linewidth=1, label='Stimulus Onset')
        ax1.set_xlabel('Time Relative to Stimulus Onset (s)')
        ax1.set_ylabel('Pause Probability')
        ax1.set_title('Peri-Stimulus Pause Probability (PSTH)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
    
    # Plot 2: Pause duration distribution
    ax2 = plt.subplot(3, 2, 2)
    pause_durations = trajectory_df[trajectory_df['pause_duration'] > 0]['pause_duration'].values
    if len(pause_durations) > 0:
        ax2.hist(pause_durations, bins=50, alpha=0.7, edgecolor='black')
        ax2.axvline(np.mean(pause_durations), color='r', linestyle='--', 
                   label=f'Mean: {np.mean(pause_durations):.2f}s')
        ax2.set_xlabel('Pause Duration (s)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Pause Duration Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Pause probability during vs outside stimulus
    ax3 = plt.subplot(3, 2, 3)
    if 'pause_stimulus_comparison' in analysis_results:
        comp = analysis_results['pause_stimulus_comparison']
        ax3.bar(['During Stimulus', 'Outside Stimulus'], 
               [comp['pause_prob_during_stimulus'], comp['pause_prob_outside_stimulus']],
               color=['red', 'blue'], alpha=0.7)
        ax3.set_ylabel('Pause Probability')
        ax3.set_title('Pause Probability: During vs Outside Stimulus')
        ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Pause frequency over time
    ax4 = plt.subplot(3, 2, 4)
    if 'time' in trajectory_df.columns and 'is_pause' in trajectory_df.columns:
        # Bin by time windows
        time_bins = np.arange(0, trajectory_df['time'].max(), 60)  # 1-minute bins
        trajectory_df['time_bin'] = pd.cut(trajectory_df['time'], bins=time_bins, labels=False)
        pause_rate_over_time = trajectory_df.groupby('time_bin').agg({
            'is_pause': 'sum',
            'time': lambda x: (x.max() - x.min()) / 60.0  # minutes
        })
        pause_rate_over_time['pause_rate_per_min'] = pause_rate_over_time['is_pause'] / pause_rate_over_time['time']
        
        bin_centers = (time_bins[:-1] + time_bins[1:]) / 2
        ax4.plot(bin_centers[:len(pause_rate_over_time)], 
                pause_rate_over_time['pause_rate_per_min'].values, 'g-', linewidth=2)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Pause Rate (per minute)')
        ax4.set_title('Pause Rate Over Time')
        ax4.grid(True, alpha=0.3)
    
    # Plot 5: Pause duration vs stimulus timing
    ax5 = plt.subplot(3, 2, 5)
    if 'time_since_stimulus' in trajectory_df.columns:
        pause_data = trajectory_df[trajectory_df['pause_duration'] > 0].copy()
        if len(pause_data) > 0:
            # Scatter plot: pause duration vs time since stimulus
            ax5.scatter(pause_data['time_since_stimulus'], pause_data['pause_duration'],
                       alpha=0.5, s=10)
            ax5.set_xlabel('Time Since Stimulus Onset (s)')
            ax5.set_ylabel('Pause Duration (s)')
            ax5.set_title('Pause Duration vs Time Since Stimulus')
            ax5.grid(True, alpha=0.3)
    
    # Plot 6: Summary statistics
    ax6 = plt.subplot(3, 2, 6)
    ax6.axis('off')
    if 'pause_frequency' in analysis_results:
        freq = analysis_results['pause_frequency']
        stats_text = f"""
Pause Statistics:
──────────────────
Total frames: {freq['total_frames']:,}
Pause frames: {freq['pause_frames']:,}
Pause fraction: {freq['pause_fraction']:.3f}

Pause Events:
──────────────────
Number of pauses: {freq['n_pause_events']}
Pause rate: {freq['pause_rate_per_min']:.2f} pauses/min
Mean duration: {freq['mean_pause_duration_s']:.2f}s
Median duration: {freq['median_pause_duration_s']:.2f}s
"""
        ax6.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                verticalalignment='center')
    
    plt.tight_layout()
    
    output_path = output_dir / 'pause_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved pause analysis plot to {output_path}")
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Analyze pause patterns')
    parser.add_argument('--trajectories-file', type=str,
                       default='data/engineered_tier2/GMR61_tier2_trajectories.csv',
                       help='Path to trajectories CSV')
    parser.add_argument('--events-file', type=str,
                       default='data/engineered_tier2/GMR61_tier2_events.csv',
                       help='Path to events CSV')
    parser.add_argument('--output-dir', type=str,
                       default='output/analysis',
                       help='Output directory for results')
    parser.add_argument('--analysis-window', type=float, nargs=2,
                       default=[-3.0, 8.0],
                       help='Analysis window relative to stimulus onset (before, after)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("PAUSE PATTERN ANALYSIS")
    print("="*80)
    
    # Load data
    print(f"\nLoading data...")
    print(f"  Trajectories: {args.trajectories_file}")
    print(f"  Events: {args.events_file}")
    
    trajectory_df = pd.read_csv(args.trajectories_file)
    events_df = pd.read_csv(args.events_file)
    
    print(f"  Loaded {len(trajectory_df):,} trajectory points")
    print(f"  Loaded {len(events_df):,} event records")
    
    # Analysis
    print(f"\n{'='*80}")
    print("ANALYZING PAUSE FREQUENCY")
    print(f"{'='*80}")
    pause_freq_stats = analyze_pause_frequency(trajectory_df, events_df)
    
    print(f"\nPause Frequency Statistics:")
    freq = pause_freq_stats['pause_frequency']
    print(f"  Total frames: {freq['total_frames']:,}")
    print(f"  Pause frames: {freq['pause_frames']:,} ({freq['pause_fraction']*100:.2f}%)")
    print(f"  Pause events: {freq['n_pause_events']}")
    print(f"  Pause rate: {freq['pause_rate_per_min']:.2f} pauses/min")
    print(f"  Mean pause duration: {freq['mean_pause_duration_s']:.2f}s")
    print(f"  Median pause duration: {freq['median_pause_duration_s']:.2f}s")
    if len(freq['pause_duration_range_s']) == 2:
        print(f"  Duration range: {freq['pause_duration_range_s'][0]:.2f}s - {freq['pause_duration_range_s'][1]:.2f}s")
    
    print(f"\n{'='*80}")
    print("ANALYZING PAUSE-STIMULUS RELATIONSHIP")
    print(f"{'='*80}")
    pause_stimulus_stats = analyze_pause_stimulus_relationship(
        trajectory_df, events_df, 
        analysis_window=tuple(args.analysis_window)
    )
    
    if 'pause_stimulus_comparison' in pause_stimulus_stats:
        comp = pause_stimulus_stats['pause_stimulus_comparison']
        print(f"\nPause-Stimulus Comparison:")
        print(f"  Pause prob during stimulus: {comp['pause_prob_during_stimulus']:.4f}")
        print(f"  Pause prob outside stimulus: {comp['pause_prob_outside_stimulus']:.4f}")
        print(f"  Ratio (during/outside): {comp['ratio']:.2f}x")
    
    # Combine results
    analysis_results = {
        **pause_freq_stats,
        **pause_stimulus_stats
    }
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / 'pause_analysis_results.json'
    with open(results_file, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    print(f"\n✓ Saved analysis results to {results_file}")
    
    # Create plots
    print(f"\n{'='*80}")
    print("CREATING VISUALIZATIONS")
    print(f"{'='*80}")
    plot_pause_analysis(trajectory_df, events_df, analysis_results, output_dir)
    
    print(f"\n{'='*80}")
    print("✓ PAUSE ANALYSIS COMPLETE")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()



