#!/usr/bin/env python3
"""
Analyze H5 files to extract LED stimulus data and verify experiment structure.
Looking for:
- Red pulsing LED (led1Val) with 10-second pulses
- Blue constant LED (led2Val)
- Verify pulse duration = 10 seconds
- Match real track structure
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_stimulus_from_h5(filepath):
    """Analyze stimulus structure from H5 file."""
    print(f"\n{'='*80}")
    print(f"Analyzing: {Path(filepath).name}")
    print(f"{'='*80}")
    
    with h5py.File(filepath, 'r') as f:
        # Get metadata
        fps = f['metadata'].attrs.get('fps', 10.0)
        num_frames = f['metadata'].attrs.get('num_frames', 0)
        total_time = num_frames / fps
        
        print(f"\nExperiment metadata:")
        print(f"  FPS: {fps}")
        print(f"  Total frames: {num_frames}")
        print(f"  Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        
        # Check LED data
        print(f"\n=== LED DATA ===")
        if 'led_data' in f:
            led_data = f['led_data'][:]
            print(f"  led_data: shape={led_data.shape}, dtype={led_data.dtype}")
            print(f"  Range: [{led_data.min():.2f}, {led_data.max():.2f}]")
            print(f"  Mean: {led_data.mean():.2f}, Std: {led_data.std():.2f}")
            print(f"  Non-zero: {np.sum(led_data > 0)}/{len(led_data)} ({100*np.sum(led_data > 0)/len(led_data):.1f}%)")
            
            # Check if it's pulsing (high variance = pulsing, low = constant)
            std_dev = led_data.std()
            if std_dev > 10:
                print(f"  → HIGH variance ({std_dev:.2f}) = PULSING LED (likely LED1/Red)")
            else:
                print(f"  → LOW variance ({std_dev:.2f}) = CONSTANT LED (likely LED2/Blue)")
            
            # Create time array
            times = np.arange(len(led_data)) / fps
            
            # Analyze pulse structure
            print(f"\n=== PULSE ANALYSIS ===")
            # Find pulses (onsets where LED goes from low to high)
            threshold = led_data.max() * 0.1  # 10% of max
            pulse_mask = led_data > threshold
            
            # Find transitions
            pulse_starts = []
            pulse_ends = []
            in_pulse = False
            for i in range(len(pulse_mask)):
                if pulse_mask[i] and not in_pulse:
                    pulse_starts.append(i)
                    in_pulse = True
                elif not pulse_mask[i] and in_pulse:
                    pulse_ends.append(i)
                    in_pulse = False
            
            if in_pulse:
                pulse_ends.append(len(pulse_mask))
            
            if pulse_starts:
                print(f"  Found {len(pulse_starts)} pulses")
                pulse_durations = []
                for start, end in zip(pulse_starts, pulse_ends):
                    duration_frames = end - start
                    duration_sec = duration_frames / fps
                    pulse_durations.append(duration_sec)
                    print(f"    Pulse {len(pulse_durations)}: frame {start}-{end} ({duration_sec:.2f}s)")
                
                avg_duration = np.mean(pulse_durations)
                print(f"\n  Average pulse duration: {avg_duration:.2f} seconds")
                if abs(avg_duration - 10.0) < 0.5:
                    print(f"  ✓ MATCHES expected 10-second pulse duration!")
                else:
                    print(f"  ⚠ Does NOT match expected 10 seconds (difference: {abs(avg_duration - 10.0):.2f}s)")
                
                # Inter-pulse intervals
                if len(pulse_starts) > 1:
                    intervals = []
                    for i in range(len(pulse_starts) - 1):
                        interval_frames = pulse_starts[i+1] - pulse_ends[i]
                        interval_sec = interval_frames / fps
                        intervals.append(interval_sec)
                    print(f"\n  Inter-pulse intervals: {[f'{x:.1f}s' for x in intervals[:5]]}...")
                    print(f"  Average interval: {np.mean(intervals):.1f} seconds")
        else:
            print("  No led_data found")
        
        # Check stimulus onsets
        print(f"\n=== STIMULUS ONSETS ===")
        if 'stimulus' in f:
            if 'onset_frames' in f['stimulus']:
                onset_frames = f['stimulus']['onset_frames'][:]
                onset_times = f['stimulus']['onset_times'][:]
                print(f"  Found {len(onset_frames)} stimulus onsets")
                print(f"  First onset: frame {onset_frames[0]} ({onset_times[0]:.1f}s)")
                print(f"  Last onset: frame {onset_frames[-1]} ({onset_times[-1]:.1f}s)")
                
                # Verify pulse duration from onsets
                if len(onset_frames) > 1:
                    # Calculate inter-onset intervals
                    intervals = np.diff(onset_times)
                    print(f"\n  Inter-onset intervals: {intervals[:5]} seconds...")
                    print(f"  Average interval: {np.mean(intervals):.1f} seconds")
                    
                    # Estimate pulse duration (time between onset and next onset minus gap)
                    # Assuming pulse_duration + inter_pulse_interval = total_cycle
                    if len(intervals) > 0:
                        avg_interval = np.mean(intervals)
                        # Common values: 5s, 10s, 20s intervals
                        # Pulse duration = cycle - interval, where cycle might be 15s, 20s, or 30s
                        # User says pulse is always 10s, so cycle = 10s + interval
                        estimated_cycles = [avg_interval + 10.0]  # pulse (10s) + interval
                        print(f"\n  Estimated cycle time: {estimated_cycles[0]:.1f}s (pulse 10s + interval {avg_interval:.1f}s)")
        
        # Check tracks structure
        print(f"\n=== TRACKS STRUCTURE ===")
        if 'tracks' in f:
            tracks = f['tracks']
            track_keys = list(tracks.keys())
            print(f"  Found {len(track_keys)} tracks")
            
            if track_keys:
                first_track = tracks[track_keys[0]]
                print(f"\n  First track ({track_keys[0]}) structure:")
                for key in first_track.keys():
                    item = first_track[key]
                    if isinstance(item, h5py.Dataset):
                        print(f"    {key}: shape={item.shape}, dtype={item.dtype}")
                    elif isinstance(item, h5py.Group):
                        print(f"    {key}/: {list(item.keys())}")
                
                # Check derived features
                if 'derived' in first_track:
                    derived = first_track['derived']
                    print(f"\n  Derived features:")
                    for key in derived.keys():
                        if isinstance(derived[key], h5py.Dataset):
                            print(f"    {key}: shape={derived[key].shape}")
        
        # Look for LED2 (might be stored separately or as a different dataset)
        print(f"\n=== SEARCHING FOR LED2 (BLUE CONSTANT) ===")
        led2_candidates = []
        
        def search_led2(name, obj):
            if isinstance(obj, h5py.Dataset):
                name_lower = name.lower()
                if any(term in name_lower for term in ['led2', 'blue', 'constant']):
                    try:
                        data = obj[...]
                        if len(data.shape) == 1 and len(data) == num_frames:
                            std_dev = np.std(data)
                            if std_dev < 5:  # Constant LED should have low variance
                                led2_candidates.append((name, std_dev, np.mean(data)))
                    except:
                        pass
        
        f.visititems(search_led2)
        
        if led2_candidates:
            print("  Found LED2 candidates (low variance = constant):")
            for name, std_dev, mean_val in led2_candidates:
                print(f"    {name}: std={std_dev:.2f}, mean={mean_val:.2f}")
        else:
            print("  No obvious LED2 found (might be in a different location or format)")
            print("  Note: LED2 might be constant and stored differently, or may not be exported")

if __name__ == '__main__':
    files = [
        '/Users/gilraitses/mechanosensation/h5tests/GMR61_202509051201_tier1 1.h5',
        '/Users/gilraitses/mechanosensation/h5tests/GMR61_tier2_complete.h5'
    ]
    
    for filepath in files:
        if Path(filepath).exists():
            try:
                analyze_stimulus_from_h5(filepath)
            except Exception as e:
                print(f"Error analyzing {filepath}: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"File not found: {filepath}")

