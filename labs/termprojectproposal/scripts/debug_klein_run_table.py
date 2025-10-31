#!/usr/bin/env python3
"""
Debug script to check what data is available for Klein run table generation.
"""

import sys
import h5py
import numpy as np
import pandas as pd
from pathlib import Path

# Add scripts to path
scripts_dir = Path(__file__).parent
sys.path.insert(0, str(scripts_dir))

from engineer_dataset_from_h5 import extract_trajectory_features, load_h5_file

def debug_klein_requirements(h5_file_path: str, track_key: str = None):
    """
    Debug what data is available for Klein run table generation.
    """
    print(f"Loading H5 file: {h5_file_path}")
    h5_data = load_h5_file(h5_file_path)
    
    # Get first track if track_key not specified
    if track_key is None:
        if 'tracks' in h5_data:
            track_keys = list(h5_data['tracks'].keys())
            if len(track_keys) == 0:
                print("ERROR: No tracks found in H5 file")
                return
            track_key = track_keys[0]
            print(f"Using first track: {track_key}")
        else:
            print("ERROR: No 'tracks' group in H5 file")
            return
    
    track_data = h5_data['tracks'][track_key]
    
    print(f"\n{'='*60}")
    print("Step 1: Extract trajectory features")
    print('='*60)
    
    try:
        df = extract_trajectory_features(track_data, frame_rate=10.0)
        print(f"✓ Successfully extracted trajectory features")
        print(f"  DataFrame shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        
        # Check required columns
        required_cols = ['time', 'x', 'y', 'heading']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"\n✗ MISSING REQUIRED COLUMNS: {missing_cols}")
        else:
            print(f"\n✓ All required columns present: {required_cols}")
            # Check for NaN values
            for col in required_cols:
                nan_count = df[col].isna().sum()
                if nan_count > 0:
                    print(f"  WARNING: {col} has {nan_count} NaN values")
                else:
                    print(f"  ✓ {col}: {len(df)} valid values")
        
    except Exception as e:
        print(f"✗ ERROR extracting trajectory features: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\n{'='*60}")
    print("Step 2: Check MAGAT segmentation")
    print('='*60)
    
    # Check if run table was generated
    if hasattr(df, 'attrs') and 'klein_run_table' in df.attrs:
        run_table = df.attrs['klein_run_table']
        print(f"✓ Klein run table found in df.attrs")
        print(f"  Run table shape: {run_table.shape}")
        print(f"  Columns: {list(run_table.columns)}")
        print(f"  Number of runs/turns: {len(run_table)}")
    else:
        print("✗ Klein run table NOT found in df.attrs")
        print("  This means either:")
        print("    1. MAGAT segmentation failed")
        print("    2. Run table generation failed")
        print("    3. Run table generation was not attempted")
    
    # Try to manually check segmentation
    print(f"\n{'='*60}")
    print("Step 3: Manual MAGAT segmentation check")
    print('='*60)
    
    try:
        from magat_segmentation import magat_segment_track, MaggotSegmentOptions
        
        # Prepare DataFrame for MAGAT segmentation
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
        
        # Add vel_dp if available or approximate
        if 'vel_dp' in df.columns:
            magat_df['vel_dp'] = df['vel_dp']
        else:
            magat_df['vel_dp'] = np.ones(len(df)) * 0.707
        
        # Configure segmentation options
        segment_options = MaggotSegmentOptions()
        segment_options.minRunTime = 2.5
        segment_options.minHeadSwingDuration = 0.05
        segment_options.minHeadSwingAmplitude = np.deg2rad(10)
        
        # Run segmentation
        frame_rate = 1.0 / np.mean(np.diff(df['time'].values[df['time'].values > 0]))
        segmentation = magat_segment_track(magat_df, segment_options=segment_options, frame_rate=frame_rate)
        
        print(f"✓ MAGAT segmentation succeeded")
        print(f"  Runs: {segmentation['n_runs']}")
        print(f"  Head swings: {segmentation['n_head_swings']}")
        print(f"  Reorientations: {segmentation['n_reorientations']}")
        
        # Check required keys
        required_keys = ['runs', 'head_swings', 'reorientations']
        missing_keys = [key for key in required_keys if key not in segmentation]
        if missing_keys:
            print(f"\n✗ MISSING REQUIRED KEYS: {missing_keys}")
        else:
            print(f"\n✓ All required keys present: {required_keys}")
            print(f"  runs: {len(segmentation['runs'])} items")
            print(f"  head_swings: {len(segmentation['head_swings'])} items")
            print(f"  reorientations: {len(segmentation['reorientations'])} items")
        
        # Try to generate run table
        print(f"\n{'='*60}")
        print("Step 4: Generate Klein run table")
        print('='*60)
        
        try:
            from klein_run_table import generate_klein_run_table
            
            run_table = generate_klein_run_table(
                trajectory_df=df,
                segmentation=segmentation,
                track_id=1,
                experiment_id=1,
                set_id=1
            )
            
            print(f"✓ Successfully generated Klein run table!")
            print(f"  Shape: {run_table.shape}")
            print(f"  Columns: {list(run_table.columns)}")
            print(f"\nFirst few rows:")
            print(run_table.head())
            
        except Exception as e:
            print(f"✗ ERROR generating Klein run table: {e}")
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        print(f"✗ ERROR running MAGAT segmentation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Debug Klein run table generation')
    parser.add_argument('h5_file', help='Path to H5 file')
    parser.add_argument('--track', help='Track key (default: first track)')
    args = parser.parse_args()
    
    debug_klein_requirements(args.h5_file, args.track)

