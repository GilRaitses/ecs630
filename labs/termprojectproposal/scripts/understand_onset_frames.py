#!/usr/bin/env python3
"""
Understand what onset_frames represent - do they mark 10-second pulse starts?
"""

import h5py
import numpy as np
from pathlib import Path

def understand_onsets(filepath):
    """Understand what onset frames represent."""
    print(f"\n{'='*80}")
    print(f"Understanding onset frames: {Path(filepath).name}")
    print(f"{'='*80}")
    
    with h5py.File(filepath, 'r') as f:
        fps = f['metadata'].attrs.get('fps', 10.0)
        led_data = f['led_data'][:]
        onset_frames = f['stimulus']['onset_frames'][:]
        onset_times = f['stimulus']['onset_times'][:]
        
        print(f"\nAssuming pulse duration = 10 seconds (as stated)")
        print(f"Onset intervals = 60 seconds (implies 10s pulse + 50s gap)")
        
        # Check first few pulses assuming 10-second duration
        print(f"\n=== CHECKING IF ONSETS MARK 10-SECOND PULSE STARTS ===")
        for i in range(min(3, len(onset_frames))):
            onset_frame = int(onset_frames[i])
            onset_time = onset_times[i]
            
            # If pulse is 10 seconds, pulse should end at onset + 100 frames (10s * 10fps)
            pulse_end_frame_10s = onset_frame + int(10 * fps)
            pulse_end_time_10s = pulse_end_frame_10s / fps
            
            print(f"\nPulse {i+1}:")
            print(f"  Onset: frame {onset_frame} ({onset_time:.1f}s)")
            print(f"  If 10s pulse: ends at frame {pulse_end_frame_10s} ({pulse_end_time_10s:.1f}s)")
            
            # Check LED values
            print(f"  LED at onset: {led_data[onset_frame]:.1f}")
            print(f"  LED at pulse end (10s): {led_data[min(pulse_end_frame_10s, len(led_data)-1)]:.1f}")
            print(f"  LED 5 frames after pulse end: {led_data[min(pulse_end_frame_10s+5, len(led_data)-1)]:.1f}")
            
            # Check what happens between this pulse end and next onset
            if i < len(onset_frames) - 1:
                next_onset_frame = int(onset_frames[i+1])
                gap_start_frame = pulse_end_frame_10s
                gap_end_frame = next_onset_frame
                
                print(f"  Gap to next onset: frames {gap_start_frame} to {gap_end_frame}")
                print(f"  Gap duration: {(gap_end_frame - gap_start_frame) / fps:.1f} seconds")
                print(f"  LED during gap (first 10 frames): {led_data[gap_start_frame:min(gap_start_frame+10, len(led_data))]}")
                print(f"  LED during gap (last 10 frames before next onset): {led_data[max(gap_start_frame, gap_end_frame-10):gap_end_frame]}")
        
        # Summary
        print(f"\n=== SUMMARY ===")
        print(f"If onsets mark 10-second pulse starts:")
        print(f"  - Pulse duration: 10 seconds")
        print(f"  - Inter-onset interval: 60 seconds")
        print(f"  - Gap between pulses: 50 seconds")
        print(f"\nLED recording shows:")
        print(f"  - LED stays high (~250) until ~20 seconds after onset")
        print(f"  - This suggests LED recording may have delay/extended recording")
        print(f"  - OR onsets mark something different than pulse start")
        print(f"\nRecommendation: Use onset frames + 10 seconds to define pulses,")
        print(f"regardless of LED value recording, since protocol is known to be 10s.")

if __name__ == '__main__':
    filepath = '/Users/gilraitses/mechanosensation/h5tests/GMR61_202509051201_tier1 1.h5'
    if Path(filepath).exists():
        understand_onsets(filepath)
    else:
        print(f"File not found: {filepath}")

