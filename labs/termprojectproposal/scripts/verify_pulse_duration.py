#!/usr/bin/env python3
"""
Verify pulse duration is 10 seconds by examining LED values around stimulus onsets.
"""

import h5py
import numpy as np
from pathlib import Path

def verify_pulse_duration(filepath):
    """Verify pulse duration is 10 seconds."""
    print(f"\n{'='*80}")
    print(f"Verifying pulse duration: {Path(filepath).name}")
    print(f"{'='*80}")
    
    with h5py.File(filepath, 'r') as f:
        fps = f['metadata'].attrs.get('fps', 10.0)
        led_data = f['led_data'][:]
        onset_frames = f['stimulus']['onset_frames'][:]
        
        print(f"\nFPS: {fps}")
        print(f"LED data length: {len(led_data)} frames")
        print(f"Number of onsets: {len(onset_frames)}")
        
        # Check first few pulses in detail
        print(f"\n=== DETAILED PULSE ANALYSIS (first 3 pulses) ===")
        for i in range(min(3, len(onset_frames))):
            onset_frame = onset_frames[i]
            onset_time = onset_frame / fps
            
            # Check LED values around onset
            # Pulse should be 10 seconds = 100 frames at 10 fps
            pulse_end_frame = onset_frame + int(10 * fps)  # 10 seconds
            pulse_end_frame = min(pulse_end_frame, len(led_data))
            
            # Extract LED values during pulse
            pulse_frames = np.arange(onset_frame, pulse_end_frame)
            pulse_led = led_data[pulse_frames]
            
            print(f"\nPulse {i+1}:")
            print(f"  Onset: frame {onset_frame} ({onset_time:.1f}s)")
            print(f"  Expected end: frame {pulse_end_frame} ({pulse_end_frame/fps:.1f}s)")
            print(f"  LED during pulse: min={pulse_led.min():.1f}, max={pulse_led.max():.1f}, mean={pulse_led.mean():.1f}")
            print(f"  LED after pulse (next 10 frames): {led_data[pulse_end_frame:pulse_end_frame+10]}")
            
            # Check if LED drops after 10 seconds
            if pulse_end_frame + 10 < len(led_data):
                post_pulse = led_data[pulse_end_frame:pulse_end_frame+10]
                print(f"  Post-pulse mean: {post_pulse.mean():.1f} (should be near 0)")
        
        # Verify inter-onset intervals match expected (pulse 10s + interval)
        print(f"\n=== INTER-ONSET INTERVALS ===")
        if len(onset_frames) > 1:
            intervals_frames = np.diff(onset_frames)
            intervals_sec = intervals_frames / fps
            print(f"  Intervals: {intervals_sec[:5]} seconds")
            print(f"  Mean interval: {np.mean(intervals_sec):.1f} seconds")
            print(f"  This means: pulse ({10.0}s) + gap ({np.mean(intervals_sec) - 10.0:.1f}s) = cycle ({np.mean(intervals_sec):.1f}s)")

if __name__ == '__main__':
    filepath = '/Users/gilraitses/mechanosensation/h5tests/GMR61_202509051201_tier1 1.h5'
    if Path(filepath).exists():
        verify_pulse_duration(filepath)
    else:
        print(f"File not found: {filepath}")

