#!/usr/bin/env python3
"""
Check actual pulse duration by examining LED values and onset frames carefully.
"""

import h5py
import numpy as np
from pathlib import Path

def check_pulse_duration(filepath):
    """Carefully check pulse duration from onset frames and LED values."""
    print(f"\n{'='*80}")
    print(f"Checking pulse duration: {Path(filepath).name}")
    print(f"{'='*80}")
    
    with h5py.File(filepath, 'r') as f:
        fps = f['metadata'].attrs.get('fps', 10.0)
        led_data = f['led_data'][:]
        onset_frames = f['stimulus']['onset_frames'][:]
        onset_times = f['stimulus']['onset_times'][:]
        
        print(f"\nFPS: {fps}")
        print(f"Total frames: {len(led_data)}")
        print(f"Number of onsets: {len(onset_frames)}")
        
        # Check first few pulses in detail
        print(f"\n=== DETAILED PULSE EXAMINATION ===")
        for i in range(min(5, len(onset_frames))):
            onset_frame = int(onset_frames[i])
            onset_time = onset_times[i]
            
            print(f"\nPulse {i+1}:")
            print(f"  Onset frame: {onset_frame} ({onset_time:.1f}s)")
            
            # Check LED values at onset and around it
            check_range_start = max(0, onset_frame - 5)
            check_range_end = min(len(led_data), onset_frame + 120)  # Check up to 12 seconds after
            
            led_section = led_data[check_range_start:check_range_end]
            frames_section = np.arange(check_range_start, check_range_end)
            times_section = frames_section / fps
            
            print(f"  LED values around onset (frames {check_range_start}-{check_range_end}):")
            print(f"    Before onset (5 frames): {led_data[max(0, onset_frame-5):onset_frame]}")
            print(f"    At onset (frame {onset_frame}): {led_data[onset_frame]:.1f}")
            print(f"    After onset (first 10 frames): {led_data[onset_frame:onset_frame+10]}")
            
            # Find where LED drops (if it does)
            # Check if LED stays high after 10 seconds
            pulse_end_frame_expected = onset_frame + int(10 * fps)  # 10 seconds = 100 frames
            pulse_end_frame_expected = min(pulse_end_frame_expected, len(led_data))
            
            print(f"\n  Expected pulse end (10s): frame {pulse_end_frame_expected} ({pulse_end_frame_expected/fps:.1f}s)")
            print(f"    LED at expected end: {led_data[pulse_end_frame_expected]:.1f}")
            print(f"    LED around expected end (±5 frames): {led_data[max(0, pulse_end_frame_expected-5):min(len(led_data), pulse_end_frame_expected+5)]}")
            
            # Look for where LED actually drops to near zero
            # Search from expected end forward
            search_start = pulse_end_frame_expected
            search_end = min(search_start + int(50 * fps), len(led_data))  # Search up to 50 seconds ahead
            
            if search_start < len(led_data):
                led_after_pulse = led_data[search_start:search_end]
                drop_indices = np.where(led_after_pulse < 10)[0]  # Find where LED < 10
                
                if len(drop_indices) > 0:
                    first_drop_frame = search_start + drop_indices[0]
                    actual_pulse_duration = (first_drop_frame - onset_frame) / fps
                    print(f"    LED drops to <10 at frame {first_drop_frame} ({first_drop_frame/fps:.1f}s)")
                    print(f"    → Actual pulse duration: {actual_pulse_duration:.2f} seconds")
                else:
                    print(f"    LED stays high (>10) for next 50 seconds")
                    print(f"    This suggests LED record may continue beyond actual pulse")
            
            # Check what the next onset tells us
            if i < len(onset_frames) - 1:
                next_onset_frame = int(onset_frames[i+1])
                next_onset_time = onset_times[i+1]
                interval = next_onset_time - onset_time
                print(f"\n  Next onset: frame {next_onset_frame} ({next_onset_time:.1f}s)")
                print(f"  Interval between onsets: {interval:.1f} seconds")
                print(f"  If pulse is 10s, gap should be: {interval - 10:.1f} seconds")
                
                # Check LED values between pulses
                mid_point_frame = onset_frame + int((next_onset_frame - onset_frame) / 2)
                print(f"  LED at midpoint between pulses (frame {mid_point_frame}): {led_data[mid_point_frame]:.1f}")

if __name__ == '__main__':
    filepath = '/Users/gilraitses/mechanosensation/h5tests/GMR61_202509051201_tier1 1.h5'
    if Path(filepath).exists():
        check_pulse_duration(filepath)
    else:
        print(f"File not found: {filepath}")

