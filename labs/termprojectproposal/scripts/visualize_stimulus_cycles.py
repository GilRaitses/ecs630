#!/usr/bin/env python3
"""
Visualize LED1 stimulus cycles as square wave and list cycle details.

Extracts:
- Cycle number
- Frame start (onset)
- Frame end (when LED drops)
- ETI (Estimated Time Interval / Inter-trial interval)
"""

import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import h5py

def extract_stimulus_cycles(h5_path, threshold=50.0, fps=10.0):
    """
    Extract stimulus cycle information from H5 file.
    
    Parameters
    ----------
    h5_path : str
        Path to H5 file
    threshold : float
        LED threshold for detecting "off" state
    fps : float
        Frame rate in Hz
    
    Returns
    -------
    cycles : list of dict
        List of cycle dictionaries with cycle_num, frame_start, frame_end, eti
    led1_data : ndarray
        Full LED1 data array (red pulsing)
    led2_data : ndarray or None
        Full LED2 data array (blue constant) if available
    """
    cycles = []
    led2_data = None
    
    with h5py.File(h5_path, 'r') as f:
        # Get onset frames
        onset_frames = f['stimulus']['onset_frames'][:]
        onset_frames = np.sort(onset_frames)
        
        # Get LED1 data (red pulsing)
        if 'global_quantities' in f and 'led1Val' in f['global_quantities']:
            led1_data = f['global_quantities']['led1Val']['yData'][:]
        else:
            raise ValueError("LED1 data not found in H5 file")
        
        # Get LED2 data (blue constant)
        if 'global_quantities' in f and 'led2Val' in f['global_quantities']:
            gq_item = f['global_quantities']['led2Val']
            if isinstance(gq_item, h5py.Group) and 'yData' in gq_item:
                led2_data = gq_item['yData'][:]
            elif isinstance(gq_item, h5py.Dataset):
                led2_data = gq_item[:]
        
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
                pulse_duration = drop_indices[0] / fps
            else:
                # If no drop found, use next onset as end (or end of data)
                if i < len(onset_frames) - 1:
                    drop_frame_int = int(onset_frames[i+1])
                    drop_time = drop_frame_int / fps
                    pulse_duration = (drop_frame_int - onset_frame_int) / fps
                else:
                    drop_frame_int = len(led1_data) - 1
                    drop_time = drop_frame_int / fps
                    pulse_duration = (drop_frame_int - onset_frame_int) / fps
            
            # Calculate ETI (time until next onset, or remaining time)
            if i < len(onset_frames) - 1:
                next_onset_frame = int(onset_frames[i+1])
                next_onset_time = next_onset_frame / fps
                eti = next_onset_time - drop_time  # Time from drop to next onset
            else:
                eti = None  # Last cycle
            
            cycle = {
                'cycle_num': i + 1,
                'frame_start': onset_frame_int,
                'frame_end': drop_frame_int,
                'time_start': onset_time,
                'time_end': drop_time,
                'pulse_duration': pulse_duration,
                'eti': eti,
                'next_onset_frame': int(onset_frames[i+1]) if i < len(onset_frames) - 1 else None,
                'cycle_period': (onset_frames[i+1] - onset_frame_int) / fps if i < len(onset_frames) - 1 else None
            }
            
            cycles.append(cycle)
    
    return cycles, led1_data, led2_data

def plot_stimulus_squarewave(cycles, led1_data, led2_data=None, fps=10.0, max_time=None, output_path=None):
    """
    Plot LED1 and LED2 square wave showing stimulus cycles.
    
    Parameters
    ----------
    cycles : list of dict
        Cycle information
    led1_data : ndarray
        Full LED1 data array (red pulsing)
    led2_data : ndarray, optional
        Full LED2 data array (blue constant)
    fps : float
        Frame rate
    max_time : float, optional
        Maximum time to plot (seconds)
    output_path : str, optional
        Path to save plot
    """
    times = np.arange(len(led1_data)) / fps
    
    if max_time is not None:
        mask = times <= max_time
        times = times[mask]
        led1_data = led1_data[mask]
        if led2_data is not None:
            led2_data = led2_data[mask]
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Plot LED2 signal (blue constant) first, so it's behind
    if led2_data is not None:
        ax.plot(times, led2_data, 'b-', linewidth=2, alpha=0.6, label='LED2 (Blue Constant)', zorder=1)
    
    # Fill area under LED1 signal (red pulsing)
    ax.fill_between(times, 0, led1_data, color='red', alpha=0.4, label='LED1 (Red Pulsing)', zorder=2)
    ax.plot(times, led1_data, 'r-', linewidth=1.5, alpha=0.8, zorder=3)
    
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('LED Intensity', fontsize=12)
    ax.set_title('LED Stimulus Cycles: Red (Pulsing) and Blue (Constant)', fontsize=14, fontweight='bold')
    
    # Highlight cycles
    colors = plt.cm.tab10(np.linspace(0, 1, min(10, len(cycles))))
    
    for i, cycle in enumerate(cycles):
        if max_time is not None and cycle['time_start'] > max_time:
            break
        
        cycle_time_start = cycle['time_start']
        cycle_time_end = cycle['time_end']
        
        # Mark onset
        color = colors[i % len(colors)]
        ax.axvline(cycle_time_start, color=color, linestyle='--', linewidth=1, alpha=0.5, zorder=4)
        
        # Mark drop
        ax.axvline(cycle_time_end, color=color, linestyle=':', linewidth=1, alpha=0.5, zorder=4)
        
        # Annotate cycle number
        mid_time = (cycle_time_start + cycle_time_end) / 2
        if max_time is None or mid_time <= max_time:
            max_val = max(np.max(led1_data), np.max(led2_data) if led2_data is not None else 0)
            ax.text(mid_time, max_val * 0.95, f"C{cycle['cycle_num']}", 
                   ha='center', va='bottom', fontsize=8, fontweight='bold', zorder=5)
    
    ax.grid(True, alpha=0.3, zorder=0)
    ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved plot to {output_path}")
    
    plt.show()

def print_cycle_table(cycles):
    """
    Print formatted table of cycle information.
    """
    print("\n" + "="*100)
    print("STIMULUS CYCLE INFORMATION")
    print("="*100)
    print(f"{'Cycle':<8} {'Frame Start':<12} {'Frame End':<12} {'Time Start (s)':<16} {'Time End (s)':<16} {'Duration (s)':<14} {'ETI (s)':<12} {'Period (s)':<12}")
    print("-"*100)
    
    for cycle in cycles:
        eti_str = f"{cycle['eti']:.1f}" if cycle['eti'] is not None else "N/A"
        period_str = f"{cycle['cycle_period']:.1f}" if cycle['cycle_period'] is not None else "N/A"
        
        print(f"{cycle['cycle_num']:<8} "
              f"{cycle['frame_start']:<12} "
              f"{cycle['frame_end']:<12} "
              f"{cycle['time_start']:<16.1f} "
              f"{cycle['time_end']:<16.1f} "
              f"{cycle['pulse_duration']:<14.1f} "
              f"{eti_str:<12} "
              f"{period_str:<12}")
    
    print("="*100)
    
    # Summary statistics
    durations = [c['pulse_duration'] for c in cycles]
    etis = [c['eti'] for c in cycles if c['eti'] is not None]
    periods = [c['cycle_period'] for c in cycles if c['cycle_period'] is not None]
    
    print(f"\nSummary Statistics:")
    print(f"  Total cycles: {len(cycles)}")
    print(f"  Pulse duration: {np.mean(durations):.1f}s (range: {np.min(durations):.1f}-{np.max(durations):.1f}s)")
    if etis:
        print(f"  ETI (Inter-trial interval): {np.mean(etis):.1f}s (range: {np.min(etis):.1f}-{np.max(etis):.1f}s)")
    if periods:
        print(f"  Cycle period: {np.mean(periods):.1f}s (range: {np.min(periods):.1f}-{np.max(periods):.1f}s)")

def main():
    parser = argparse.ArgumentParser(description='Visualize stimulus cycles and list cycle details')
    parser.add_argument('--h5-file', type=str,
                       default='/Users/gilraitses/mechanosensation/h5tests/GMR61_tier2_complete.h5',
                       help='Path to H5 file')
    parser.add_argument('--threshold', type=float, default=50.0,
                       help='LED threshold for detecting "off" state')
    parser.add_argument('--fps', type=float, default=10.0,
                       help='Frame rate in Hz')
    parser.add_argument('--max-time', type=float, default=None,
                       help='Maximum time to plot (seconds, None = all)')
    parser.add_argument('--output-plot', type=str, default=None,
                       help='Path to save plot (None = show interactively)')
    parser.add_argument('--output-csv', type=str, default=None,
                       help='Path to save cycle table as CSV')
    
    args = parser.parse_args()
    
    print("="*100)
    print("EXTRACTING STIMULUS CYCLES")
    print("="*100)
    print(f"H5 file: {args.h5_file}")
    print(f"Threshold: {args.threshold}")
    print(f"FPS: {args.fps}")
    
    # Extract cycles
    cycles, led1_data, led2_data = extract_stimulus_cycles(args.h5_file, args.threshold, args.fps)
    
    print(f"\n✓ Extracted {len(cycles)} cycles")
    if led2_data is not None:
        print(f"✓ LED2 (blue constant) data available: mean={np.mean(led2_data):.2f}, std={np.std(led2_data):.2f}")
    else:
        print("⚠ LED2 data not found")
    
    # Print cycle table
    print_cycle_table(cycles)
    
    # Plot square wave
    print(f"\n{'='*100}")
    print("PLOTTING SQUARE WAVE")
    print(f"{'='*100}")
    
    plot_stimulus_squarewave(cycles, led1_data, led2_data, args.fps, args.max_time, args.output_plot)
    
    # Save CSV if requested
    if args.output_csv:
        cycles_df = pd.DataFrame(cycles)
        cycles_df.to_csv(args.output_csv, index=False)
        print(f"\n✓ Saved cycle table to {args.output_csv}")

if __name__ == '__main__':
    main()

