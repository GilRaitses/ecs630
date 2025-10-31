#!/usr/bin/env python3
"""
Create EDA figures for experimental dataset:
1. Stimulus conditions visualization
2. Stimulus-locked turn rate analysis (using ACTUAL experimental cycles)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from scipy.stats import t as t_dist
from cinnamoroll_palette import CINNAMOROLL_COLORS, CINNAMOROLL_PALETTE, setup_cinnamoroll_style

import h5py

def load_experimental_data(trajectories_file):
    """Load experimental trajectory data."""
    print(f"Loading experimental data from {trajectories_file}...")
    df = pd.read_csv(trajectories_file, nrows=100000)  # Load subset for efficiency
    print(f"  Loaded {len(df)} rows")
    print(f"  Columns: {list(df.columns)[:10]}...")
    return df

def extract_cycles_from_h5(h5_file):
    """
    Extract actual stimulus cycles from H5 file.
    Returns cycles with actual pulse duration and timing.
    
    Uses the same approach as visualize_behavioral_events_stepwise.py
    """
    
    cycles = []
    led1_data = None
    
    print(f"Extracting cycles from {h5_file}...")
    try:
        with h5py.File(h5_file, 'r') as f:
            fps = 10.0
            
            # Get onset frames
            if 'stimulus' in f and 'onset_frames' in f['stimulus']:
                onset_frames = f['stimulus']['onset_frames'][:]
                onset_frames = np.sort(onset_frames)
                onset_times = onset_frames / fps
            else:
                print("  No stimulus onsets found")
                return cycles, None
            
            # Get LED1 data
            if 'global_quantities' in f and 'led1Val' in f['global_quantities']:
                gq_item = f['global_quantities']['led1Val']
                if isinstance(gq_item, h5py.Group) and 'yData' in gq_item:
                    led1_data = gq_item['yData'][:]
                else:
                    led1_data = gq_item[:]
            else:
                print("  No LED1 data found")
                return cycles, None
            
            print(f"  Found {len(onset_frames)} stimulus onsets")
            print(f"  LED1 data: {len(led1_data)} frames")
            
            # Extract cycles - find actual LED ON and OFF times
            threshold = 50.0
            pulse_durations = []
            
            for i, onset_frame in enumerate(onset_frames):
                onset_frame_int = int(onset_frame)
                onset_time = onset_frame_int / fps
                
                # Find when LED actually turns ON (may be before onset_frame)
                # Look backwards from onset_frame to find LED transition
                lookback_start = max(0, onset_frame_int - 50)  # Look back up to 5 seconds
                lookback_window = led1_data[lookback_start:onset_frame_int + 1]
                on_indices = np.where(lookback_window >= threshold)[0]
                
                if len(on_indices) > 0:
                    # LED ON is the first frame where LED >= threshold before/at onset
                    led_on_frame = lookback_start + on_indices[0]
                    led_on_time = led_on_frame / fps
                else:
                    # Fallback: use onset_frame as LED ON
                    led_on_frame = onset_frame_int
                    led_on_time = onset_time
                
                # Find when LED turns OFF after LED ON
                pulse_window_start = led_on_frame
                pulse_window_end = min(led_on_frame + 600, len(led1_data))  # Check up to 60 seconds
                pulse_window = led1_data[pulse_window_start:pulse_window_end]
                
                drop_indices = np.where(pulse_window < threshold)[0]
                
                if len(drop_indices) > 0:
                    led_off_frame = pulse_window_start + drop_indices[0]
                    led_off_time = led_off_frame / fps
                    pulse_duration = led_off_time - led_on_time
                else:
                    # If no drop found, use next onset as end
                    if i < len(onset_frames) - 1:
                        next_onset_frame = int(onset_frames[i+1])
                        led_off_frame = next_onset_frame
                        led_off_time = next_onset_frame / fps
                        pulse_duration = led_off_time - led_on_time
                    else:
                        led_off_frame = len(led1_data) - 1
                        led_off_time = led_off_frame / fps
                        pulse_duration = led_off_time - led_on_time
                
                pulse_durations.append(pulse_duration)
                
                # Analysis window: 10 seconds BEFORE LED ON + full pulse duration
                # Cycle is aligned to LED ON time (not onset_frame)
                baseline_period = 10.0
                baseline_start_time = max(0, led_on_time - baseline_period)
                analysis_end_time = led_off_time
                
                cycle = {
                    'cycle_num': i + 1,
                    'onset_frame': onset_frame_int,  # Original onset frame (for reference)
                    'onset_time': onset_time,  # Original onset time (for reference)
                    'led_on_frame': led_on_frame,
                    'led_on_time': led_on_time,  # Actual LED ON time (cycle start = 0)
                    'led_off_frame': led_off_frame,
                    'led_off_time': led_off_time,
                    'pulse_duration': pulse_duration,
                    'baseline_start_time': baseline_start_time,
                    'cycle_start_time': baseline_start_time,  # Start of analysis window (10s before LED ON)
                    'cycle_end_time': analysis_end_time  # End of analysis window (LED OFF)
                }
                cycles.append(cycle)
            
            if len(pulse_durations) > 0:
                print(f"  Pulse durations: min={min(pulse_durations):.1f}s, max={max(pulse_durations):.1f}s, mean={np.mean(pulse_durations):.1f}s")
        
        return cycles, led1_data
    except FileNotFoundError as e:
        print(f"  Error: H5 file not found: {h5_file}")
        raise
    except Exception as e:
        print(f"  Error extracting cycles: {e}")
        import traceback
        traceback.print_exc()
        raise

def calculate_stimulus_locked_turn_rate_from_data(events_df, cycles, h5_file=None):
    """
    Calculate actual stimulus-locked turn rate from events data.
    
    Proper methodology:
    1. For each cycle, bin into 0.5s bins
    2. For each bin in each cycle, aggregate tracks that have data
    3. Calculate turn rate per bin: (n_turns / bin_duration) * 60
    4. Store rate and n_tracks for each cycle-bin combination
    5. Aggregate across cycles with proper weighting by n_tracks
    
    Parameters
    ----------
    events_df : pd.DataFrame
        Events CSV with track_id, time, is_turn columns
    cycles : List[Dict]
        Actual cycles extracted from H5 file
    h5_file : str, optional
        Path to H5 file for LED data
    
    Returns
    -------
    dict
        Dictionary with bin_centers, mean_rates, ci_lower, ci_upper, led_pattern
    """
    from scipy import stats
    
    if len(cycles) == 0:
        print("  No cycles found")
        return None
    
    # Determine analysis window based on actual pulse duration
    pulse_durations = [c['pulse_duration'] for c in cycles]
    max_pulse_duration = max(pulse_durations)
    mean_pulse_duration = np.mean(pulse_durations)
    
    # Analysis window: 10 seconds before stimulus + full pulse duration
    baseline_period = 10.0
    analysis_window = (-baseline_period, max_pulse_duration)
    
    print(f"  Pulse duration: {mean_pulse_duration:.1f}s (range: {min(pulse_durations):.1f}-{max(pulse_durations):.1f}s)")
    print(f"  Analysis window: {analysis_window[0]:.1f}s to {analysis_window[1]:.1f}s (total: {analysis_window[1] - analysis_window[0]:.1f}s)")
    
    BIN_SIZE = 0.5  # 0.5 second bins
    t_min, t_max = analysis_window
    bin_edges = np.arange(t_min, t_max + BIN_SIZE, BIN_SIZE)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    n_bins = len(bin_centers)
    
    # Use proper reorientation detection (MAGAT-based, not simple is_turn)
    if 'is_reorientation' not in events_df.columns:
        print("  Error: 'is_reorientation' column not found in events CSV")
        print("  This requires proper dataset engineering with MAGAT segmentation")
        print("  Please run engineer_dataset_from_h5.py first")
        return None
    
    # Use is_reorientation (MAGAT-compatible reorientation detection)
    print("  Using is_reorientation (MAGAT-based reorientation detection)")
    
    # Structure: bin_idx -> list of (rate, n_tracks) tuples for each cycle
    bin_data_by_cycle = {bin_idx: [] for bin_idx in range(n_bins)}
    
    # Process each cycle
    cycles_processed = 0
    
    for cycle in cycles:
        cycle_start = cycle['cycle_start_time']
        cycle_end = cycle['cycle_end_time']
        onset_time = cycle['onset_time']
        
        # Extract events for this cycle
        cycle_events = events_df[
            (events_df['time'] >= cycle_start) & 
            (events_df['time'] <= cycle_end)
        ].copy()
        
        if len(cycle_events) == 0:
            continue
        
        # Calculate relative time from LED ON (cycle alignment point)
        # Cycle time 0 = LED ON, not onset_frame
        led_on_time = cycle['led_on_time']
        cycle_events['time_rel_onset'] = cycle_events['time'] - led_on_time
        
        # Filter to analysis window
        cycle_events = cycle_events[
            (cycle_events['time_rel_onset'] >= t_min) & 
            (cycle_events['time_rel_onset'] <= t_max)
        ]
        
        if len(cycle_events) == 0:
            continue
        
        # Bin using digitize (correctly handles bin boundaries)
        # digitize returns bin index where bin_edges[i] <= x < bin_edges[i+1]
        # Result is 1-indexed, so subtract 1 to get 0-indexed bin
        cycle_events['time_bin'] = np.digitize(cycle_events['time_rel_onset'].values, bin_edges) - 1
        cycle_events['time_bin'] = np.clip(cycle_events['time_bin'], 0, n_bins - 1)
        
        # For each bin, calculate turn rate per track, then aggregate
        for bin_idx in range(n_bins):
            bin_events = cycle_events[cycle_events['time_bin'] == bin_idx]
            
            if len(bin_events) == 0:
                continue
            
            # Get unique tracks that have data in this bin
            unique_tracks = bin_events['track_id'].unique()
            n_tracks = len(unique_tracks)
            
            if n_tracks == 0:
                continue
            
            # Gold standard: calculate per-track per-cycle bin rates, then aggregate
            track_rates = []
            for track_id in unique_tracks:
                track_bin_events = bin_events[bin_events['track_id'] == track_id]
                track_reorientations = track_bin_events['is_reorientation'].sum()
                track_rate = (track_reorientations / BIN_SIZE) * 60.0
                track_rates.append(track_rate)
            
            if len(track_rates) > 0:
                # Store list of per-track rates for this cycle-bin
                bin_data_by_cycle[bin_idx].append(track_rates)
        
        cycles_processed += 1
    
    print(f"  Processed {cycles_processed} cycles")
    
    # Aggregate across cycles: mean of per-track per-cycle bin rates (gold standard)
    mean_rates = []
    sem_rates = []
    ci_lower = []
    ci_upper = []
    
    for bin_idx in range(n_bins):
        cycle_data = bin_data_by_cycle[bin_idx]
        
        if len(cycle_data) == 0:
            mean_rates.append(0)
            ci_lower.append(0)
            ci_upper.append(0)
            continue
        
        # Flatten all per-track rates for this bin across cycles
        all_track_rates = []
        for track_rates in cycle_data:
            all_track_rates.extend(track_rates)
        
        if len(all_track_rates) == 0:
            mean_rates.append(0)
            sem_rates.append(0)
            ci_lower.append(0)
            ci_upper.append(0)
            continue
        
        aggregate_rate = float(np.mean(all_track_rates))
        
        if len(all_track_rates) > 1:
            se = float(np.std(all_track_rates, ddof=1) / np.sqrt(len(all_track_rates)))
            t_val = t_dist.ppf(0.975, len(all_track_rates) - 1)
            ci_low = aggregate_rate - t_val * se
            ci_high = aggregate_rate + t_val * se
        else:
            se = 0.0
            ci_low = aggregate_rate
            ci_high = aggregate_rate
        
        mean_rates.append(aggregate_rate)
        sem_rates.append(se)
        ci_lower.append(max(0, ci_low))
        ci_upper.append(ci_high)
    
    # Create validation table for plausibility checking
    validation_df = pd.DataFrame({
        'bin_center': bin_centers,
        'mean_rate': mean_rates,
        'sem_rate': sem_rates,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n_cycles': [len(bin_data_by_cycle[i]) for i in range(n_bins)],
        'n_track_cycles': [sum(len(track_rates) for track_rates in bin_data_by_cycle[i]) if len(bin_data_by_cycle[i]) > 0 else 0 
                           for i in range(n_bins)]
    })
    
    # Add time relative to LED ON (cycle time 0 = LED ON)
    validation_df['time_rel_stimulus'] = validation_df['bin_center']
    validation_df['is_baseline'] = validation_df['time_rel_stimulus'] < 0
    validation_df['is_during_stimulus'] = (validation_df['time_rel_stimulus'] >= 0) & (validation_df['time_rel_stimulus'] <= mean_pulse_duration)
    
    # Add cycle alignment info
    validation_df['cycle_time_0'] = 'LED ON'  # Time 0 = LED ON
    
    # Print validation summary
    print("\n  VALIDATION SUMMARY:")
    print("  " + "="*60)
    baseline_mask = validation_df['is_baseline']
    stimulus_mask = validation_df['is_during_stimulus']
    
    if baseline_mask.sum() > 0:
        baseline_mean = validation_df[baseline_mask]['mean_rate'].mean()
        baseline_min = validation_df[baseline_mask]['mean_rate'].min()
        baseline_max = validation_df[baseline_mask]['mean_rate'].max()
        print(f"  Baseline period (-10s to 0s):")
        print(f"    Mean rate: {baseline_mean:.2f} turns/min")
        print(f"    Range: {baseline_min:.2f} - {baseline_max:.2f} turns/min")
    
    if stimulus_mask.sum() > 0:
        stimulus_mean = validation_df[stimulus_mask]['mean_rate'].mean()
        stimulus_min = validation_df[stimulus_mask]['mean_rate'].min()
        stimulus_max = validation_df[stimulus_mask]['mean_rate'].max()
        print(f"  Stimulus period (0s to {mean_pulse_duration:.1f}s):")
        print(f"    Mean rate: {stimulus_mean:.2f} turns/min")
        print(f"    Range: {stimulus_min:.2f} - {stimulus_max:.2f} turns/min")
    
    overall_mean = validation_df['mean_rate'].mean()
    overall_max = validation_df['mean_rate'].max()
    print(f"\n  Overall:")
    print(f"    Mean rate: {overall_mean:.2f} turns/min")
    print(f"    Max rate: {overall_max:.2f} turns/min")
    
    # Check for plausibility (typical larval turn rates: 0-15 turns/min)
    if overall_max > 50:
        print(f"\n  ⚠️  WARNING: Max turn rate ({overall_max:.2f} turns/min) exceeds typical range (0-15 turns/min)")
    elif overall_max > 20:
        print(f"\n  ⚠️  CAUTION: Max turn rate ({overall_max:.2f} turns/min) is high but may be valid")
    else:
        print(f"\n  ✓ Turn rates appear biologically plausible")
    
    # Create LED pattern for plotting - square wave with proper transitions
    # Create time points that include bin edges for clean square wave
    bin_edges_full = np.arange(t_min, t_max + BIN_SIZE, BIN_SIZE)
    
    # LED pattern: OFF before time 0, ON from 0 to pulse_duration, OFF after
    led_pattern = {
        'time': bin_edges_full,
        'values': np.zeros_like(bin_edges_full)
    }
    
    # LED ON from time 0 to pulse_duration
    led_on_mask = (led_pattern['time'] >= 0) & (led_pattern['time'] <= mean_pulse_duration)
    led_pattern['values'][led_on_mask] = 1.0
    
    # Ensure clean transitions at boundaries
    # Find indices closest to 0 and pulse_duration
    zero_idx = np.argmin(np.abs(led_pattern['time']))
    pulse_end_idx = np.argmin(np.abs(led_pattern['time'] - mean_pulse_duration))
    
    # Set transition points
    if zero_idx < len(led_pattern['time']):
        if led_pattern['time'][zero_idx] < 0:
            # Insert point at exactly 0
            led_pattern['time'] = np.insert(led_pattern['time'], zero_idx + 1, 0.0)
            led_pattern['values'] = np.insert(led_pattern['values'], zero_idx + 1, 1.0)
        elif led_pattern['time'][zero_idx] > 0:
            # Insert point at exactly 0
            led_pattern['time'] = np.insert(led_pattern['time'], zero_idx, 0.0)
            led_pattern['values'] = np.insert(led_pattern['values'], zero_idx, 1.0)
    
    if pulse_end_idx < len(led_pattern['time']):
        if led_pattern['time'][pulse_end_idx] < mean_pulse_duration:
            # Insert point at pulse end
            led_pattern['time'] = np.insert(led_pattern['time'], pulse_end_idx + 1, mean_pulse_duration)
            led_pattern['values'] = np.insert(led_pattern['values'], pulse_end_idx + 1, 0.0)
        elif led_pattern['time'][pulse_end_idx] > mean_pulse_duration:
            # Insert point at pulse end
            led_pattern['time'] = np.insert(led_pattern['time'], pulse_end_idx, mean_pulse_duration)
            led_pattern['values'] = np.insert(led_pattern['values'], pulse_end_idx, 0.0)
    
    return {
        'bin_centers': bin_centers,
        'mean_rates': mean_rates,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'led_pattern': led_pattern,
        'pulse_duration': mean_pulse_duration,
        'baseline_period': baseline_period,
        'validation_table': validation_df
    }

def plot_stimulus_conditions(df, output_path):
    """
    Plot stimulus conditions distribution from experimental dataset.
    Shows PWM intensity, pulse duration, and inter-pulse interval distributions.
    """
    plt = setup_cinnamoroll_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor(CINNAMOROLL_COLORS['cream'])
    
    # Extract stimulus parameters if available
    if 'led1_pwm' in df.columns:
        pwm_values = df['led1_pwm'].dropna()
        axes[0, 0].hist(pwm_values, bins=30, color=CINNAMOROLL_COLORS['light_blue'],
                       edgecolor=CINNAMOROLL_COLORS['blue'], linewidth=1.5, alpha=0.7)
        axes[0, 0].set_xlabel('LED1 PWM Intensity', color=CINNAMOROLL_COLORS['dark_blue'])
        axes[0, 0].set_ylabel('Frequency', color=CINNAMOROLL_COLORS['dark_blue'])
        axes[0, 0].set_title('LED Intensity Distribution', color=CINNAMOROLL_COLORS['dark_blue'])
        axes[0, 0].set_facecolor('white')
        axes[0, 0].grid(True, alpha=0.3, color=CINNAMOROLL_COLORS['tan'])
    else:
        axes[0, 0].text(0.5, 0.5, 'LED PWM data not available', 
                        ha='center', va='center', transform=axes[0, 0].transAxes,
                        color=CINNAMOROLL_COLORS['brown'])
        axes[0, 0].set_facecolor('white')
    
    if 'pulse_duration' in df.columns:
        pulse_durations = df['pulse_duration'].dropna()
        axes[0, 1].hist(pulse_durations, bins=20, color=CINNAMOROLL_COLORS['lavender'],
                       edgecolor=CINNAMOROLL_COLORS['purple'], linewidth=1.5, alpha=0.7)
        axes[0, 1].set_xlabel('Pulse Duration (s)', color=CINNAMOROLL_COLORS['dark_blue'])
        axes[0, 1].set_ylabel('Frequency', color=CINNAMOROLL_COLORS['dark_blue'])
        axes[0, 1].set_title('Pulse Duration Distribution', color=CINNAMOROLL_COLORS['dark_blue'])
        axes[0, 1].set_facecolor('white')
        axes[0, 1].grid(True, alpha=0.3, color=CINNAMOROLL_COLORS['tan'])
    else:
        axes[0, 1].text(0.5, 0.5, 'Pulse duration data not available',
                        ha='center', va='center', transform=axes[0, 1].transAxes,
                        color=CINNAMOROLL_COLORS['brown'])
        axes[0, 1].set_facecolor('white')
    
    if 'time' in df.columns and 'led1_pwm' in df.columns:
        time_sample = df.groupby(df.index // 100)['time'].first()
        pwm_sample = df.groupby(df.index // 100)['led1_pwm'].mean()
        axes[1, 0].plot(time_sample, pwm_sample, color=CINNAMOROLL_COLORS['pink'],
                       linewidth=2, alpha=0.8)
        axes[1, 0].set_xlabel('Time (s)', color=CINNAMOROLL_COLORS['dark_blue'])
        axes[1, 0].set_ylabel('LED1 PWM', color=CINNAMOROLL_COLORS['dark_blue'])
        axes[1, 0].set_title('Stimulus Time Series', color=CINNAMOROLL_COLORS['dark_blue'])
        axes[1, 0].set_facecolor('white')
        axes[1, 0].grid(True, alpha=0.3, color=CINNAMOROLL_COLORS['tan'])
    else:
        axes[1, 0].text(0.5, 0.5, 'Time series data not available',
                        ha='center', va='center', transform=axes[1, 0].transAxes,
                        color=CINNAMOROLL_COLORS['brown'])
        axes[1, 0].set_facecolor('white')
    
    axes[1, 1].axis('off')
    stats_text = []
    stats_text.append("Experimental Dataset Summary")
    stats_text.append("=" * 30)
    stats_text.append(f"Total rows: {len(df):,}")
    stats_text.append(f"Unique tracks: {df['track_id'].nunique() if 'track_id' in df.columns else 'N/A'}")
    if 'time' in df.columns:
        stats_text.append(f"Time range: {df['time'].min():.1f} - {df['time'].max():.1f} s")
    if 'led1_pwm' in df.columns:
        stats_text.append(f"PWM range: {df['led1_pwm'].min():.0f} - {df['led1_pwm'].max():.0f}")
    
    axes[1, 1].text(0.1, 0.5, '\n'.join(stats_text),
                   transform=axes[1, 1].transAxes,
                   fontsize=11, family='monospace',
                   color=CINNAMOROLL_COLORS['dark_blue'],
                   verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='white',
                            edgecolor=CINNAMOROLL_COLORS['light_blue'], linewidth=2))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=CINNAMOROLL_COLORS['cream'])
    print(f"Saved stimulus conditions plot to {output_path}")
    plt.close()

def create_stimulus_locked_turn_rate_analysis(trajectories_file, events_file, h5_file, output_path):
    """
    Create stimulus-locked turn rate analysis using ACTUAL experimental cycles.
    
    Matches MATLAB gold standard but uses real data:
    - Extracts actual cycles from H5 file
    - Determines actual pulse duration
    - Analysis window: 10 seconds before stimulus + full pulse duration
    - Calculates turn rates from actual events data
    """
    plt = setup_cinnamoroll_style()
    
    # Load events data (has is_turn column)
    print(f"Loading events data from {events_file}...")
    events_df = pd.read_csv(events_file, nrows=500000)  # Load subset
    print(f"  Loaded {len(events_df)} event records")
    
    # Extract actual cycles from H5 file (required, no fallback)
    cycles, led1_data = extract_cycles_from_h5(h5_file)
    
    if len(cycles) == 0:
        raise ValueError(f"No cycles found in {h5_file}. Cannot create analysis.")
    
    # Calculate turn rates from actual data
    analysis_data = calculate_stimulus_locked_turn_rate_from_data(events_df, cycles, h5_file)
    
    if analysis_data is None:
        print("  Warning: Could not calculate turn rates")
        return
    
    bin_centers = analysis_data['bin_centers']
    mean_rates = analysis_data['mean_rates']
    ci_lower = analysis_data['ci_lower']
    ci_upper = analysis_data['ci_upper']
    led_pattern = analysis_data['led_pattern']
    pulse_duration = analysis_data['pulse_duration']
    baseline_period = analysis_data['baseline_period']
    validation_df = analysis_data['validation_table']
    
    # Save validation table
    validation_path = output_path.parent / (output_path.stem + '_validation.csv')
    validation_df.to_csv(validation_path, index=False)
    print(f"  Saved validation table to {validation_path}")
    
    # Create composite figure (upper: LED, lower: turn rate)
    fig = plt.figure(figsize=(10, 7))
    fig.patch.set_facecolor('white')
    
    # Upper subplot: LED stimulus square wave
    ax_led = plt.subplot2grid((4, 1), (0, 0), rowspan=1, fig=fig)
    
    # Plot LED pattern as square wave
    # Sort by time to ensure proper plotting
    sort_idx = np.argsort(led_pattern['time'])
    led_time_sorted = np.array(led_pattern['time'])[sort_idx]
    led_values_sorted = np.array(led_pattern['values'])[sort_idx]
    
    ax_led.plot(led_time_sorted, led_values_sorted, color='#CC3333', linewidth=3, drawstyle='steps-post')
    ax_led.fill_between(led_time_sorted, led_values_sorted, 0, color='#CC3333', alpha=0.3, step='post')
    
    x_min = min(bin_centers)
    x_max = max(bin_centers)
    ax_led.set_xlim(x_min, x_max)
    ax_led.set_ylim(-0.1, 1.1)
    ax_led.set_ylabel('Fictive vibration', fontsize=12, fontweight='bold', color='#333333')
    ax_led.set_xticks([])
    ax_led.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='LED ON')
    ax_led.spines['top'].set_visible(False)
    ax_led.spines['right'].set_visible(False)
    ax_led.spines['bottom'].set_visible(False)
    ax_led.tick_params(colors='#333333')
    ax_led.grid(True, alpha=0.5, color='#F2F2F2')
    
    # Lower subplot: Turn rate analysis
    ax_turn = plt.subplot2grid((4, 1), (1, 0), rowspan=3, fig=fig)
    
    # Plot confidence interval (light grey)
    ax_turn.fill_between(bin_centers, ci_lower, ci_upper, 
                        color='#E6E6E6', alpha=0.8, edgecolor='none')
    
    # Plot mean turn rate (blue)
    ax_turn.plot(bin_centers, mean_rates, color='#3366CC', linewidth=2,
                marker='o', markersize=4, markerfacecolor='#3366CC')
    
    # Vertical line at LED ON (cycle time 0)
    ax_turn.axvline(x=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.6, label='LED ON')
    
    # Styling
    ax_turn.set_xlim(x_min, x_max)
    max_y = max(ci_upper) if ci_upper else 10
    ax_turn.set_ylim(0, max_y * 1.1 if max_y > 0 else 10)
    ax_turn.set_xlabel('time in cycle (s)', fontsize=12, color='black')
    ax_turn.set_ylabel('Turn rate (min^-1)', fontsize=12, color='black')
    ax_turn.set_title(f'Aggregate (All Tracks) - Pulse: {pulse_duration:.1f}s', 
                     fontsize=14, fontweight='bold', color='black')
    ax_turn.grid(True, alpha=0.5, color='#F2F2F2')
    ax_turn.set_xticks(np.arange(int(x_min), int(x_max) + 1, 5))
    ax_turn.spines['top'].set_visible(False)
    ax_turn.spines['right'].set_visible(False)
    ax_turn.set_facecolor('white')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved stimulus-locked turn rate analysis to {output_path}")
    plt.close()

def main():
    """Main function to generate EDA figures."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create EDA figures for experimental dataset')
    parser.add_argument('--trajectories-file', type=str,
                       default='data/engineered/GMR61_tier2_complete_trajectories.csv',
                       help='Path to engineered trajectories CSV')
    parser.add_argument('--events-file', type=str,
                       default='data/engineered/GMR61_tier2_complete_events.csv',
                       help='Path to events CSV with is_turn column')
    parser.add_argument('--h5-file', type=str,
                       default='/Users/gilraitses/mechanosensation/h5tests/GMR61_tier2_complete.h5',
                       help='Path to H5 file with cycles and LED data')
    parser.add_argument('--output-dir', type=str, 
                       default='/Users/gilraitses/ecs630/labs/termprojectproposal/output/figures/eda',
                       help='Output directory for figures')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_experimental_data(args.trajectories_file)
    
    # Create figures
    print("\nCreating EDA figures...")
    
    # 1. Stimulus conditions
    stimulus_plot_path = output_dir / 'stimulus_conditions.png'
    plot_stimulus_conditions(df, stimulus_plot_path)
    
    # 2. Stimulus-locked turn rate analysis (using ACTUAL cycles)
    turnrate_plot_path = output_dir / 'stimulus_locked_turn_rate_analysis.png'
    create_stimulus_locked_turn_rate_analysis(args.trajectories_file, args.events_file, args.h5_file, turnrate_plot_path)
    
    print(f"\nEDA figures saved to {output_dir}")

if __name__ == '__main__':
    main()
