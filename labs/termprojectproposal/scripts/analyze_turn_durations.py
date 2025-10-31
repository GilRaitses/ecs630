#!/usr/bin/env python3
"""
Analyze turn durations: distribution and stimulus effects.

Creates:
- Turn duration distribution statistics
- Peri-stimulus turn duration analysis
- Turn duration vs stimulus timing
- Comparison of turn durations during vs outside stimulus
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

def analyze_turn_duration_distribution(trajectory_df, events_df):
    """
    Analyze turn duration distribution statistics.
    
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
    
    # Get unique turn events (each turn has a unique turn_event_id)
    turn_events = trajectory_df[trajectory_df['turn_event_id'] > 0].copy()
    
    if len(turn_events) == 0:
        return {'turn_duration': {'n_turn_events': 0}}
    
    # Get unique turn durations (one per turn event)
    unique_turn_durations = turn_events.groupby('turn_event_id')['turn_duration'].first().values
    
    total_frames = len(trajectory_df)
    total_time = trajectory_df['time'].max() - trajectory_df['time'].min()
    
    n_turn_events = len(unique_turn_durations)
    turn_rate = n_turn_events / (total_time / 60.0) if total_time > 0 else 0
    
    # Turn duration statistics
    mean_duration = np.mean(unique_turn_durations)
    median_duration = np.median(unique_turn_durations)
    std_duration = np.std(unique_turn_durations)
    
    stats_dict['turn_duration'] = {
        'total_frames': int(total_frames),
        'n_turn_events': int(n_turn_events),
        'turn_rate_per_min': float(turn_rate),
        'mean_duration_s': float(mean_duration),
        'median_duration_s': float(median_duration),
        'std_duration_s': float(std_duration),
        'min_duration_s': float(np.min(unique_turn_durations)),
        'max_duration_s': float(np.max(unique_turn_durations)),
        'duration_range_s': [float(np.min(unique_turn_durations)), float(np.max(unique_turn_durations))]
    }
    
    # Duration distribution percentiles
    stats_dict['turn_duration_distribution'] = {
        'percentiles': {
            'p10': float(np.percentile(unique_turn_durations, 10)),
            'p25': float(np.percentile(unique_turn_durations, 25)),
            'p50': float(np.percentile(unique_turn_durations, 50)),
            'p75': float(np.percentile(unique_turn_durations, 75)),
            'p90': float(np.percentile(unique_turn_durations, 90)),
            'p95': float(np.percentile(unique_turn_durations, 95))
        }
    }
    
    # Duration distribution shape
    # Check if distribution is log-normal
    log_durations = np.log(unique_turn_durations + 1e-6)
    stats_dict['turn_duration_distribution']['log_normal'] = {
        'mean': float(np.mean(log_durations)),
        'std': float(np.std(log_durations))
    }
    
    return stats_dict

def analyze_turn_duration_stimulus_relationship(trajectory_df, events_df, analysis_window=(-3.0, 8.0)):
    """
    Analyze turn duration patterns relative to stimulus onsets.
    
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
    
    # Get unique turn events
    turn_events = trajectory_df[trajectory_df['turn_event_id'] > 0].copy()
    
    if len(turn_events) == 0:
        return results
    
    # Get turn start times (first frame of each turn)
    turn_starts = turn_events.groupby('turn_event_id').agg({
        'time': 'first',
        'turn_duration': 'first',
        'time_since_stimulus': 'first',
        'stimulus_on': 'first'
    }).reset_index()
    
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
    
    # Peri-stimulus turn duration analysis
    bin_width = 0.1  # 100ms bins
    bins = np.arange(t_min, t_max, bin_width)
    turn_durations_per_bin = []
    bin_centers = []
    
    for onset_time in onset_times:
        # Find turns that start within analysis window relative to this onset
        window_start = onset_time + t_min
        window_end = onset_time + t_max
        
        window_turns = turn_starts[
            (turn_starts['time'] >= window_start) & 
            (turn_starts['time'] <= window_end)
        ].copy()
        
        if len(window_turns) > 0:
            # Convert to time relative to onset
            window_turns['time_rel_onset'] = window_turns['time'] - onset_time
            
            # Bin relative to onset
            window_turns['time_bin'] = pd.cut(
                window_turns['time_rel_onset'], 
                bins=bins, 
                labels=False
            )
            
            # Aggregate turn durations by bin
            binned = window_turns.groupby('time_bin').agg({
                'turn_duration': ['mean', 'count']
            })
            
            for bin_idx in binned.index:
                if not np.isnan(bin_idx):
                    bin_idx = int(bin_idx)
                    if 0 <= bin_idx < len(bins) - 1:
                        bin_center = (bins[bin_idx] + bins[bin_idx + 1]) / 2
                        mean_duration = binned.loc[bin_idx, ('turn_duration', 'mean')]
                        n_turns = binned.loc[bin_idx, ('turn_duration', 'count')]
                        
                        turn_durations_per_bin.append({
                            'time_rel_onset': bin_center,
                            'mean_duration': mean_duration,
                            'n_turns': n_turns
                        })
    
    # Aggregate across all onsets
    if turn_durations_per_bin:
        bin_df = pd.DataFrame(turn_durations_per_bin)
        aggregated = bin_df.groupby('time_rel_onset').agg({
            'mean_duration': 'mean',
            'n_turns': 'sum'
        }).reset_index()
        
        results['peri_stimulus_turn_duration'] = {
            'time_rel_onset': aggregated['time_rel_onset'].tolist(),
            'mean_duration': aggregated['mean_duration'].tolist(),
            'n_turns': aggregated['n_turns'].tolist()
        }
    
    # Compare turn durations during vs outside stimulus
    if 'stimulus_on' in turn_starts.columns:
        during_stimulus = turn_starts[turn_starts['stimulus_on'] == True]
        outside_stimulus = turn_starts[turn_starts['stimulus_on'] == False]
        
        if len(during_stimulus) > 0 and len(outside_stimulus) > 0:
            mean_dur_during = during_stimulus['turn_duration'].mean()
            mean_dur_outside = outside_stimulus['turn_duration'].mean()
            
            results['turn_duration_stimulus_comparison'] = {
                'mean_duration_during_stimulus_s': float(mean_dur_during),
                'mean_duration_outside_stimulus_s': float(mean_dur_outside),
                'ratio': float(mean_dur_during / (mean_dur_outside + 1e-6)),
                'n_turns_during': int(len(during_stimulus)),
                'n_turns_outside': int(len(outside_stimulus))
            }
            
            # Statistical test
            try:
                stat, p_value = stats.mannwhitneyu(
                    during_stimulus['turn_duration'].values,
                    outside_stimulus['turn_duration'].values,
                    alternative='two-sided'
                )
                results['turn_duration_stimulus_comparison']['statistical_test'] = {
                    'test': 'Mann-Whitney U',
                    'statistic': float(stat),
                    'p_value': float(p_value),
                    'significant': bool(p_value < 0.05)
                }
            except Exception as e:
                print(f"    Warning: Statistical test failed: {e}")
                pass
    
    return results

def plot_turn_duration_analysis(trajectory_df, events_df, analysis_results, output_dir):
    """
    Create visualization plots for turn duration analysis.
    
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
    
    # Plot 1: Turn duration distribution
    ax1 = plt.subplot(3, 2, 1)
    turn_events = trajectory_df[trajectory_df['turn_event_id'] > 0]
    if len(turn_events) > 0:
        unique_durations = turn_events.groupby('turn_event_id')['turn_duration'].first().values
        ax1.hist(unique_durations, bins=50, alpha=0.7, edgecolor='black')
        ax1.axvline(np.mean(unique_durations), color='r', linestyle='--', 
                   label=f'Mean: {np.mean(unique_durations):.3f}s')
        ax1.axvline(np.median(unique_durations), color='g', linestyle='--', 
                   label=f'Median: {np.median(unique_durations):.3f}s')
        ax1.set_xlabel('Turn Duration (s)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Turn Duration Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: Log-normal fit
    ax2 = plt.subplot(3, 2, 2)
    if len(turn_events) > 0:
        unique_durations = turn_events.groupby('turn_event_id')['turn_duration'].first().values
        log_durations = np.log(unique_durations + 1e-6)
        ax2.hist(log_durations, bins=50, alpha=0.7, edgecolor='black', density=True)
        
        # Fit normal distribution
        mu, sigma = np.mean(log_durations), np.std(log_durations)
        x = np.linspace(log_durations.min(), log_durations.max(), 100)
        ax2.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2,
                label=f'Normal fit (μ={mu:.2f}, σ={sigma:.2f})')
        ax2.set_xlabel('Log(Turn Duration)')
        ax2.set_ylabel('Density')
        ax2.set_title('Turn Duration Distribution (Log Scale)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Peri-stimulus turn duration
    ax3 = plt.subplot(3, 2, 3)
    if 'peri_stimulus_turn_duration' in analysis_results:
        psth = analysis_results['peri_stimulus_turn_duration']
        ax3.plot(psth['time_rel_onset'], psth['mean_duration'], 'b-', linewidth=2, marker='o', markersize=3)
        ax3.axvline(0, color='r', linestyle='--', linewidth=1, label='Stimulus Onset')
        
        # Add bar chart showing number of turns per bin
        ax3_twin = ax3.twinx()
        ax3_twin.bar(psth['time_rel_onset'], psth['n_turns'], alpha=0.3, color='gray', width=0.08)
        ax3_twin.set_ylabel('Number of Turns', color='gray')
        ax3_twin.tick_params(axis='y', labelcolor='gray')
        
        ax3.set_xlabel('Time Relative to Stimulus Onset (s)')
        ax3.set_ylabel('Mean Turn Duration (s)', color='b')
        ax3.tick_params(axis='y', labelcolor='b')
        ax3.set_title('Peri-Stimulus Turn Duration')
        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Turn duration during vs outside stimulus
    ax4 = plt.subplot(3, 2, 4)
    if 'turn_duration_stimulus_comparison' in analysis_results:
        comp = analysis_results['turn_duration_stimulus_comparison']
        ax4.bar(['During Stimulus', 'Outside Stimulus'], 
               [comp['mean_duration_during_stimulus_s'], comp['mean_duration_outside_stimulus_s']],
               color=['red', 'blue'], alpha=0.7)
        ax4.set_ylabel('Mean Turn Duration (s)')
        ax4.set_title('Turn Duration: During vs Outside Stimulus')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add text with ratio
        ratio = comp['ratio']
        ax4.text(0.5, ax4.get_ylim()[1] * 0.9, f'Ratio: {ratio:.2f}x', 
                ha='center', fontsize=12, fontweight='bold')
    
    # Plot 5: Turn duration vs time since stimulus
    ax5 = plt.subplot(3, 2, 5)
    turn_events = trajectory_df[trajectory_df['turn_event_id'] > 0].copy()
    if len(turn_events) > 0 and 'time_since_stimulus' in turn_events.columns:
        unique_turns = turn_events.groupby('turn_event_id').agg({
            'turn_duration': 'first',
            'time_since_stimulus': 'first'
        })
        
        # Filter to reasonable range
        valid_mask = (unique_turns['time_since_stimulus'] >= 0) & (unique_turns['time_since_stimulus'] <= 60)
        ax5.scatter(unique_turns[valid_mask]['time_since_stimulus'], 
                   unique_turns[valid_mask]['turn_duration'],
                   alpha=0.5, s=10)
        ax5.set_xlabel('Time Since Stimulus Onset (s)')
        ax5.set_ylabel('Turn Duration (s)')
        ax5.set_title('Turn Duration vs Time Since Stimulus')
        ax5.grid(True, alpha=0.3)
    
    # Plot 6: Summary statistics
    ax6 = plt.subplot(3, 2, 6)
    ax6.axis('off')
    if 'turn_duration' in analysis_results:
        dur = analysis_results['turn_duration']
        stats_text = f"""
Turn Duration Statistics:
─────────────────────────
Turn events: {dur['n_turn_events']}
Turn rate: {dur['turn_rate_per_min']:.2f} turns/min

Duration Statistics:
─────────────────────────
Mean: {dur['mean_duration_s']:.3f}s
Median: {dur['median_duration_s']:.3f}s
Std: {dur['std_duration_s']:.3f}s
Range: {dur['min_duration_s']:.3f}s - {dur['max_duration_s']:.3f}s
"""
        if 'turn_duration_distribution' in analysis_results:
            pct = analysis_results['turn_duration_distribution']['percentiles']
            stats_text += f"""
Percentiles:
─────────────────────────
P10: {pct['p10']:.3f}s
P25: {pct['p25']:.3f}s
P50: {pct['p50']:.3f}s
P75: {pct['p75']:.3f}s
P90: {pct['p90']:.3f}s
"""
        ax6.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                verticalalignment='center')
    
    plt.tight_layout()
    
    output_path = output_dir / 'turn_duration_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved turn duration analysis plot to {output_path}")
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Analyze turn duration patterns')
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
    print("TURN DURATION ANALYSIS")
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
    print("ANALYZING TURN DURATION DISTRIBUTION")
    print(f"{'='*80}")
    duration_stats = analyze_turn_duration_distribution(trajectory_df, events_df)
    
    if duration_stats['turn_duration']['n_turn_events'] > 0:
        print(f"\nTurn Duration Statistics:")
        dur = duration_stats['turn_duration']
        print(f"  Turn events: {dur['n_turn_events']}")
        print(f"  Turn rate: {dur['turn_rate_per_min']:.2f} turns/min")
        print(f"  Mean duration: {dur['mean_duration_s']:.3f}s")
        print(f"  Median duration: {dur['median_duration_s']:.3f}s")
        print(f"  Std duration: {dur['std_duration_s']:.3f}s")
        print(f"  Duration range: {dur['min_duration_s']:.3f}s - {dur['max_duration_s']:.3f}s")
        
        if 'turn_duration_distribution' in duration_stats:
            pct = duration_stats['turn_duration_distribution']['percentiles']
            print(f"\n  Duration percentiles:")
            print(f"    P10: {pct['p10']:.3f}s | P25: {pct['p25']:.3f}s | P50: {pct['p50']:.3f}s")
            print(f"    P75: {pct['p75']:.3f}s | P90: {pct['p90']:.3f}s | P95: {pct['p95']:.3f}s")
    else:
        print("  No turn events found")
    
    print(f"\n{'='*80}")
    print("ANALYZING TURN DURATION-STIMULUS RELATIONSHIP")
    print(f"{'='*80}")
    duration_stimulus_stats = analyze_turn_duration_stimulus_relationship(
        trajectory_df, events_df, 
        analysis_window=tuple(args.analysis_window)
    )
    
    if 'turn_duration_stimulus_comparison' in duration_stimulus_stats:
        comp = duration_stimulus_stats['turn_duration_stimulus_comparison']
        print(f"\nTurn Duration-Stimulus Comparison:")
        print(f"  Mean duration during stimulus: {comp['mean_duration_during_stimulus_s']:.3f}s")
        print(f"  Mean duration outside stimulus: {comp['mean_duration_outside_stimulus_s']:.3f}s")
        print(f"  Ratio (during/outside): {comp['ratio']:.2f}x")
        print(f"  Turns during: {comp['n_turns_during']} | Turns outside: {comp['n_turns_outside']}")
        
        if 'statistical_test' in comp:
            test = comp['statistical_test']
            print(f"\n  Statistical test ({test['test']}):")
            print(f"    p-value: {test['p_value']:.4f}")
            print(f"    Significant: {'Yes' if test['significant'] else 'No'}")
    
    # Combine results
    analysis_results = {
        **duration_stats,
        **duration_stimulus_stats
    }
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / 'turn_duration_analysis_results.json'
    with open(results_file, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    print(f"\n✓ Saved analysis results to {results_file}")
    
    # Create plots
    print(f"\n{'='*80}")
    print("CREATING VISUALIZATIONS")
    print(f"{'='*80}")
    plot_turn_duration_analysis(trajectory_df, events_df, analysis_results, output_dir)
    
    print(f"\n{'='*80}")
    print("✓ TURN DURATION ANALYSIS COMPLETE")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()

