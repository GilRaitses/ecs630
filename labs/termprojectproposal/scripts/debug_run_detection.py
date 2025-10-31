#!/usr/bin/env python3
"""
Debug why MAGAT segmentation isn't detecting runs.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

scripts_dir = Path(__file__).parent
sys.path.insert(0, str(scripts_dir))

from engineer_dataset_from_h5 import extract_trajectory_features, load_h5_file
from magat_segmentation import magat_segment_track, MaggotSegmentOptions

def debug_run_detection(h5_file_path: str, track_key: str = None):
    """
    Debug why runs aren't being detected.
    """
    print(f"Loading H5 file: {h5_file_path}")
    h5_data = load_h5_file(h5_file_path)
    
    if track_key is None:
        track_keys = list(h5_data['tracks'].keys())
        track_key = track_keys[0]
    
    track_data = h5_data['tracks'][track_key]
    df = extract_trajectory_features(track_data, frame_rate=10.0)
    
    print(f"\n{'='*60}")
    print("Data Summary")
    print('='*60)
    print(f"Total frames: {len(df)}")
    print(f"Speed range: {df['speed'].min():.6f} to {df['speed'].max():.6f}")
    print(f"Speed median: {df['speed'].median():.6f}")
    print(f"Speed mean: {df['speed'].mean():.6f}")
    print(f"Speed > 0: {(df['speed'] > 0).sum()} frames")
    
    # Prepare MAGAT DataFrame
    magat_df = pd.DataFrame({
        'time': df['time'],
        'speed': df['speed'],
        'curvature': df['curvature'],
        'curv': df['curvature'],
        'spineTheta': df.get('spineTheta_magat', np.zeros(len(df))),
        'sspineTheta': df.get('sspineTheta_magat', np.zeros(len(df))),
        'heading': df['heading'],
        'x': df['x'],
        'y': df['y']
    })
    
    magat_df['vel_dp'] = np.ones(len(df)) * 0.707
    
    segment_options = MaggotSegmentOptions()
    segment_options.minRunTime = 2.5
    segment_options.minHeadSwingDuration = 0.05
    segment_options.minHeadSwingAmplitude = np.deg2rad(10)
    
    print(f"\n{'='*60}")
    print("MAGAT Segmentation Options")
    print('='*60)
    print(f"stop_speed_cut: {segment_options.stop_speed_cut}")
    print(f"start_speed_cut: {segment_options.start_speed_cut}")
    print(f"minRunTime: {segment_options.minRunTime}")
    print(f"theta_cut: {segment_options.theta_cut}")
    print(f"curv_cut: {segment_options.curv_cut}")
    print(f"aligned_dp: {segment_options.aligned_dp}")
    
    # Check what frames meet run criteria
    speed = magat_df['speed'].values
    curvature = np.abs(magat_df['curvature'].values)
    body_theta = np.abs(magat_df['spineTheta'].values)
    vel_dp = magat_df['vel_dp'].values
    
    # Adaptive thresholds (from magat_segmentation.py)
    speed_median = np.median(speed[speed > 0])
    stop_speed_cut = segment_options.stop_speed_cut
    start_speed_cut = segment_options.start_speed_cut
    
    if speed_median < 0.1:  # Likely in cm/s
        speed_sorted = np.sort(speed[speed > 0])
        if len(speed_sorted) > 10:
            stop_speed_cut = np.percentile(speed_sorted, 5)
            start_speed_cut = np.percentile(speed_sorted, 15)
            stop_speed_cut = max(stop_speed_cut, speed_median * 0.1)
            start_speed_cut = max(start_speed_cut, speed_median * 0.2)
    
    print(f"\n{'='*60}")
    print("Adjusted Speed Thresholds")
    print('='*60)
    print(f"stop_speed_cut (adjusted): {stop_speed_cut:.6f}")
    print(f"start_speed_cut (adjusted): {start_speed_cut:.6f}")
    
    # Check criteria
    speedlow = speed < stop_speed_cut
    speedhigh = speed >= start_speed_cut
    
    body_theta_abs = np.abs(body_theta)
    if np.any(body_theta_abs > 0):
        theta_cut_actual = np.percentile(body_theta_abs[body_theta_abs > 0], 90)
        theta_cut_actual = min(theta_cut_actual, segment_options.theta_cut)
    else:
        theta_cut_actual = segment_options.theta_cut
    
    head_swinging = body_theta_abs > theta_cut_actual
    highcurv = curvature > segment_options.curv_cut
    
    notarun = highcurv | head_swinging | speedlow
    headaligned = vel_dp >= segment_options.aligned_dp
    
    isarun = (~notarun) & speedhigh & headaligned
    
    print(f"\n{'='*60}")
    print("Run Detection Criteria")
    print('='*60)
    print(f"Frames with speedlow: {speedlow.sum()} ({speedlow.sum()/len(df)*100:.1f}%)")
    print(f"Frames with speedhigh: {speedhigh.sum()} ({speedhigh.sum()/len(df)*100:.1f}%)")
    print(f"Frames with head_swinging: {head_swinging.sum()} ({head_swinging.sum()/len(df)*100:.1f}%)")
    print(f"Frames with highcurv: {highcurv.sum()} ({highcurv.sum()/len(df)*100:.1f}%)")
    print(f"Frames with notarun: {notarun.sum()} ({notarun.sum()/len(df)*100:.1f}%)")
    print(f"Frames with headaligned: {headaligned.sum()} ({headaligned.sum()/len(df)*100:.1f}%)")
    print(f"Frames meeting ALL run criteria (isarun): {isarun.sum()} ({isarun.sum()/len(df)*100:.1f}%)")
    
    # Run segmentation
    frame_rate = 1.0 / np.mean(np.diff(df['time'].values[df['time'].values > 0]))
    segmentation = magat_segment_track(magat_df, segment_options=segment_options, frame_rate=frame_rate)
    
    print(f"\n{'='*60}")
    print("Segmentation Results")
    print('='*60)
    print(f"Runs detected: {segmentation['n_runs']}")
    print(f"Head swings detected: {segmentation['n_head_swings']}")
    print(f"Reorientations detected: {segmentation['n_reorientations']}")
    
    if segmentation['n_runs'] == 0:
        print(f"\n{'='*60}")
        print("DIAGNOSIS: Why no runs detected?")
        print('='*60)
        
        # Check if isarun has any True values
        if isarun.sum() == 0:
            print("✓ Problem: NO frames meet all run criteria simultaneously")
            print("\nBreakdown of why frames fail:")
            
            failing_notarun = ~notarun
            failing_speedhigh = speedhigh
            failing_headaligned = headaligned
            
            # Frames that pass individual criteria
            pass_notarun = (~notarun).sum()
            pass_speedhigh = speedhigh.sum()
            pass_headaligned = headaligned.sum()
            
            print(f"  - Pass 'not a run' check (~notarun): {pass_notarun} frames")
            print(f"  - Pass speed high check: {pass_speedhigh} frames")
            print(f"  - Pass head aligned check: {pass_headaligned} frames")
            
            # Find frames that pass 2 out of 3
            pass_2_of_3 = ((~notarun) & speedhigh).sum()
            pass_2_of_3b = ((~notarun) & headaligned).sum()
            pass_2_of_3c = (speedhigh & headaligned).sum()
            
            print(f"\n  Frames passing 2/3 criteria:")
            print(f"    - (~notarun) & speedhigh: {pass_2_of_3}")
            print(f"    - (~notarun) & headaligned: {pass_2_of_3b}")
            print(f"    - speedhigh & headaligned: {pass_2_of_3c}")
            
            # Check most common failure
            failures = {
                'speedlow': speedlow.sum(),
                'head_swinging': head_swinging.sum(),
                'highcurv': highcurv.sum(),
                'speed_not_high': (~speedhigh).sum(),
                'head_not_aligned': (~headaligned).sum()
            }
            
            print(f"\n  Most common individual failures:")
            for reason, count in sorted(failures.items(), key=lambda x: x[1], reverse=True):
                print(f"    - {reason}: {count} frames ({count/len(df)*100:.1f}%)")
        else:
            print(f"✗ Problem: {isarun.sum()} frames meet criteria but segmentation found 0 runs")
            print("  This suggests a problem with run start/stop matching logic")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Debug run detection')
    parser.add_argument('h5_file', help='Path to H5 file')
    parser.add_argument('--track', help='Track key')
    args = parser.parse_args()
    
    debug_run_detection(args.h5_file, args.track)

