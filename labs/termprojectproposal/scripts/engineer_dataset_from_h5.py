#!/usr/bin/env python3
"""
Engineer dataset from H5 files for term project modeling.

Extracts trajectory data, stimulus timing, and behavioral events from H5 files
and creates feature matrices suitable for hazard model fitting.

GOLD STANDARD: tier2_complete.h5 format
- LED data: global_quantities/led1Val/yData and global_quantities/led2Val/yData
- Track structure: tracks/track_N/points/{head,mid,tail} and derived_quantities/{speed,theta,curv}
- Creates addTonToff-equivalent fields: led1Val_ton, led1Val_toff, led2Val_ton, led2Val_toff

Usage:
    python scripts/engineer_dataset_from_h5.py \
        --h5-dir /Users/gilraitses/mechanosensation/h5tests \
        --output-dir data/engineered \
        --experiment-id GMR61_202509051201
"""

import sys
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    print("WARNING: h5py not installed. Install with: pip install h5py")

def load_h5_file(h5_path: Path) -> Dict:
    """Load H5 file and return dictionary of datasets."""
    if not HAS_H5PY:
        raise ImportError("h5py required. Install with: pip install h5py")
    
    data = {}
    with h5py.File(h5_path, 'r') as f:
        # Load metadata
        if 'metadata' in f:
            data['metadata'] = dict(f['metadata'].attrs)
        
        # Load tracks
        if 'tracks' in f:
            data['tracks'] = {}
            for track_key in f['tracks'].keys():
                track_group = f[f'tracks/{track_key}']
                track_data = {}
                
                # Load position data (head, mid, tail, and spine points)
                # Gold standard (tier2_complete): positions in points/head, points/mid, points/tail
                # Also includes spine_points and spine_indices for full spine tracking
                if 'points' in track_group and isinstance(track_group['points'], h5py.Group):
                    # Tier2 structure: positions in points/head, points/mid, points/tail
                    points_group = track_group['points']
                    for pos_key in ['head', 'mid', 'tail']:
                        if pos_key in points_group:
                            track_data[pos_key] = points_group[pos_key][:]
                    
                    # Load spine points (multiple points per frame)
                    if 'spine_points' in points_group:
                        spine_data = points_group['spine_points'][:]
                        # Ensure it's a 2D array (N_points, 2)
                        if spine_data.size > 0:
                            if spine_data.ndim == 1:
                                # Reshape if needed: might be (2*N,) -> (N, 2)
                                if len(spine_data) % 2 == 0:
                                    spine_data = spine_data.reshape(-1, 2)
                            track_data['spine_points'] = spine_data
                        else:
                            print(f"      Warning: Empty spine_points for {track_key}")
                    if 'spine_indices' in points_group:
                        idx_data = points_group['spine_indices'][:]
                        if idx_data.size > 0:
                            track_data['spine_indices'] = idx_data.flatten()
                        else:
                            print(f"      Warning: Empty spine_indices for {track_key}")
                else:
                    # Tier1 fallback: positions directly in track_group
                    for pos_key in ['head', 'mid', 'tail']:
                        if pos_key in track_group:
                            track_data[pos_key] = track_group[pos_key][:]
                    # Spine points may not be available in Tier1
                    if 'spine_points' in track_group:
                        spine_data = track_group['spine_points'][:]
                        # Ensure it's a 2D array (N_points, 2)
                        if spine_data.size > 0:
                            if spine_data.ndim == 1:
                                # Reshape if needed: might be (2*N,) -> (N, 2)
                                if len(spine_data) % 2 == 0:
                                    spine_data = spine_data.reshape(-1, 2)
                            track_data['spine_points'] = spine_data
                        else:
                            print(f"      Warning: Empty spine_points for {track_key}")
                    if 'spine_indices' in track_group:
                        idx_data = track_group['spine_indices'][:]
                        if idx_data.size > 0:
                            track_data['spine_indices'] = idx_data.flatten()
                        else:
                            print(f"      Warning: Empty spine_indices for {track_key}")
                
                # Load derived features if available
                # Gold standard (tier2_complete): uses 'derived_quantities' with speed, theta, curv
                derived_group = None
                if 'derived_quantities' in track_group:
                    # Tier2 structure (gold standard)
                    derived_group = track_group['derived_quantities']
                elif 'derived' in track_group:
                    # Tier1 fallback structure
                    derived_group = track_group['derived']
                
                if derived_group is not None:
                    track_data['derived'] = {}
                    for derived_key in derived_group.keys():
                        if isinstance(derived_group[derived_key], h5py.Dataset):
                            track_data['derived'][derived_key] = derived_group[derived_key][:]
                
                # Load track attributes
                track_data['attrs'] = dict(track_group.attrs)
                data['tracks'][track_key] = track_data
        
        # Load LED data - check multiple possible locations
        # Gold standard (tier2_complete): global_quantities/led1Val/yData and global_quantities/led2Val/yData
        led1_found = False
        led2_found = False
        
        # Method 1: Check global_quantities (gold standard tier2_complete format)
        if 'global_quantities' in f:
            gq = f['global_quantities']
            # Check for led1Val and led2Val in global_quantities (exact match, not variations)
            for key in gq.keys():
                # Match exactly 'led1Val' (not 'led1ValDeriv', 'led1ValDiff', etc.)
                if key == 'led1Val' and not led1_found:
                    gq_item = gq[key]
                    if isinstance(gq_item, h5py.Group):
                        # Check for yData within the group
                        if 'yData' in gq_item:
                            data['led1Val'] = gq_item['yData'][:]
                            led1_found = True
                    elif isinstance(gq_item, h5py.Dataset):
                        data['led1Val'] = gq_item[:]
                        led1_found = True
                
                # Match exactly 'led2Val' (not 'led2ValDeriv', 'led2ValDiff', etc.)
                elif key == 'led2Val' and not led2_found:
                    gq_item = gq[key]
                    if isinstance(gq_item, h5py.Group):
                        # Check for yData within the group
                        if 'yData' in gq_item:
                            data['led2Val'] = gq_item['yData'][:]
                            led2_found = True
                    elif isinstance(gq_item, h5py.Dataset):
                        data['led2Val'] = gq_item[:]
                        led2_found = True
        
        # Method 2: Check top-level led_data (fallback for led1Val)
        if not led1_found and 'led_data' in f:
            data['led_data'] = f['led_data'][:]
            data['led1Val'] = f['led_data'][:]
            led1_found = True
        
        # Method 3: Check separate led2_data dataset
        if not led2_found and 'led2_data' in f:
            data['led2Val'] = f['led2_data'][:]
            led2_found = True
        
        # If led1Val found but led2Val not found, create placeholder
        if led1_found and not led2_found:
            n_frames = len(data['led1Val'])
            data['led2Val'] = np.zeros(n_frames)  # Placeholder
            print(f"  WARNING: led2Val not found, creating zero placeholder")
        
        # Also store led_data for backwards compatibility
        if 'led1Val' in data and 'led_data' not in data:
            data['led_data'] = data['led1Val']
        
        # Load stimulus group (onset frames/times)
        if 'stimulus' in f:
            data['stimulus'] = {}
            for subkey in f['stimulus'].keys():
                if isinstance(f['stimulus'][subkey], h5py.Dataset):
                    data['stimulus'][subkey] = f['stimulus'][subkey][:]
        
        # Load metadata attributes
        if 'metadata' in f:
            data['metadata'] = {'attrs': dict(f['metadata'].attrs)}
    
    return data

def extract_trajectory_features(track_data: Dict, frame_rate: float = 10.0) -> pd.DataFrame:
    """
    Extract trajectory features from track data.
    
    Based on actual H5 structure: head, mid, tail positions and derived features.
    
    Parameters
    ----------
    track_data : dict
        Track data dictionary with head/mid/tail positions and derived features
    frame_rate : float
        Frame rate in Hz (default 10 fps from H5 metadata)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: time, x, y, speed, heading, etc.
    """
    # Use mid position as primary location (centroid)
    if 'mid' not in track_data:
        return pd.DataFrame()
    
    mid_pos = track_data['mid']  # N×2 array (x, y)
    n_frames = len(mid_pos)
    
    if n_frames == 0:
        return pd.DataFrame()
    
    x = mid_pos[:, 0]
    y = mid_pos[:, 1]
    frames = np.arange(n_frames)
    time = frames / frame_rate
    
    # Use pre-computed derived features if available (tier2_complete is gold standard)
    if 'derived' in track_data and isinstance(track_data['derived'], dict):
        derived = track_data['derived']
        if 'speed' in derived:
            speed = derived['speed']
            # Tier2 structure: speed is shape (1, N), flatten to (N,)
            if speed.ndim > 1:
                speed = speed.flatten()
            # Ensure correct length
            if len(speed) > n_frames:
                speed = speed[:n_frames]
            elif len(speed) < n_frames:
                # Pad with last value if shorter
                speed = np.pad(speed, (0, n_frames - len(speed)), mode='edge')
        else:
            # Compute speed from positions
            dx = np.diff(x, prepend=x[0])
            dy = np.diff(y, prepend=y[0])
            dt = np.diff(time, prepend=time[0])
            speed = np.sqrt(dx**2 + dy**2) / np.maximum(dt, 1e-6)
        
        if 'direction' in derived:
            heading = derived['direction']
            # Tier2 structure: may be 2D, flatten
            if heading.ndim > 1:
                heading = heading.flatten()
            if len(heading) > n_frames:
                heading = heading[:n_frames]
            elif len(heading) < n_frames:
                heading = np.pad(heading, (0, n_frames - len(heading)), mode='edge')
        elif 'theta' in derived:
            # Tier2 uses 'theta' instead of 'direction'
            heading = derived['theta']
            if heading.ndim > 1:
                heading = heading.flatten()
            if len(heading) > n_frames:
                heading = heading[:n_frames]
            elif len(heading) < n_frames:
                heading = np.pad(heading, (0, n_frames - len(heading)), mode='edge')
        else:
            # Compute heading from positions
            dx = np.diff(x, prepend=x[0])
            dy = np.diff(y, prepend=y[0])
            heading = np.arctan2(dy, dx)
            heading = np.concatenate([[heading[0]], heading[1:]])
        
        if 'curvature' in derived:
            curvature = derived['curvature']
            if curvature.ndim > 1:
                curvature = curvature.flatten()
            if len(curvature) > n_frames:
                curvature = curvature[:n_frames]
            elif len(curvature) < n_frames:
                curvature = np.pad(curvature, (0, n_frames - len(curvature)), mode='constant', constant_values=0)
        elif 'curv' in derived:
            # Tier2 uses 'curv' instead of 'curvature'
            curvature = derived['curv']
            if curvature.ndim > 1:
                curvature = curvature.flatten()
            if len(curvature) > n_frames:
                curvature = curvature[:n_frames]
            elif len(curvature) < n_frames:
                curvature = np.pad(curvature, (0, n_frames - len(curvature)), mode='constant', constant_values=0)
        else:
            curvature = np.zeros(n_frames)
    else:
        # Compute speed using MAGAT algorithm (if not already computed)
        if not speed_magat_computed:
            try:
                import sys
                from pathlib import Path
                script_dir = Path(__file__).parent
                sys.path.insert(0, str(script_dir))
                from magat_speed_analysis import calculate_speed_magat
                
                # Compute MAGAT speed from positions
                positions = np.array([x, y])  # (2, n_frames)
                interp_time = np.mean(np.diff(time)) if len(time) > 1 else 0.1
                smooth_time = 0.1  # MAGAT default
                deriv_time = 0.1   # MAGAT default
                
                speed, velocity, smoothed_locs = calculate_speed_magat(
                    positions, interp_time, smooth_time, deriv_time
                )
                speed_magat_computed = True
                
            except Exception as e:
                # Fallback to simple diff-based speed
                print(f"  Warning: MAGAT speed calculation failed ({e}), using simple diff")
                dx = np.diff(x, prepend=x[0])
                dy = np.diff(y, prepend=y[0])
                dt = np.diff(time, prepend=time[0])
                speed = np.sqrt(dx**2 + dy**2) / np.maximum(dt, 1e-6)
        
        # Compute heading and curvature
        dx = np.diff(x, prepend=x[0])
        dy = np.diff(y, prepend=y[0])
        heading = np.arctan2(dy, dx)
        heading = np.concatenate([[heading[0]], heading[1:]])
        curvature = np.zeros(n_frames)
    
    # Ensure speed is 1D before computing acceleration
    if speed.ndim > 1:
        speed = speed.flatten()
    if len(speed) != n_frames:
        speed = speed[:n_frames] if len(speed) > n_frames else np.pad(speed, (0, n_frames - len(speed)), mode='edge')
    
    # Compute acceleration
    dt = np.diff(time, prepend=time[0])
    accel = np.diff(speed, prepend=speed[0]) / np.maximum(dt, 1e-6)
    
    # Extract deltatheta (heading change per frame) from H5 if available
    # This is the basis for reorientation detection (more accurate than diff(heading))
    # 
    # NOTE: In MAGAT (https://github.com/GilRaitses/magniphyq), reorientations are accessed as 
    # track.reorientation (abbreviated "reo") and are MaggotReorientation objects with startInd/endInd.
    # In tier2_complete H5, reorientations are not pre-computed, so we detect them from deltatheta
    # using MAGAT-compatible thresholds. MAGAT's reorientation detection algorithm uses deltatheta
    # along with speed thresholds and head swing detection (numHS >= 1).
    #
    # MAGAT Reference: @MaggotReorientation class in magniphyq repository
    deltatheta = None
    if 'derived' in track_data and isinstance(track_data['derived'], dict):
        if 'deltatheta' in track_data['derived']:
            deltatheta_raw = track_data['derived']['deltatheta']
            # Flatten if 2D (tier2 structure: (1, N))
            if deltatheta_raw.ndim > 1:
                deltatheta_raw = deltatheta_raw.flatten()
            # Ensure correct length
            if len(deltatheta_raw) >= n_frames:
                deltatheta = deltatheta_raw[:n_frames]  # Keep sign for directional info
            else:
                # Pad with zeros if shorter
                deltatheta = np.pad(deltatheta_raw, (0, n_frames - len(deltatheta_raw)), mode='constant')
        # TODO: If tier2_complete adds a 'reo' or 'reorientation' field (from MAGAT), use it directly
    
    # Compute heading_change: use deltatheta if available, otherwise compute from heading
    if deltatheta is not None:
        # Use pre-computed deltatheta from H5 (more accurate, already accounts for angle wrapping)
        # Keep absolute value for magnitude
        heading_change = np.abs(deltatheta)
    else:
        # Fallback: compute from heading differences
        heading_change = np.abs(np.diff(heading, prepend=heading[0]))
        # Wrap angles to [-pi, pi]
        heading_change = np.mod(heading_change + np.pi, 2*np.pi) - np.pi
        heading_change = np.abs(heading_change)
    
    # Initialize MAGAT spine analysis flags (computed in spine analysis section below)
    spine_theta_magat_computed = False
    spine_theta_magat = None
    spine_theta_smoothed = None
    
    # Initialize MAGAT segmentation (will be set if segmentation succeeds)
    magat_segmentation = None
    
    # Extract spineTheta (body bend angle) from derived quantities if available (for fallback)
    spine_theta = None
    if 'derived' in track_data and isinstance(track_data['derived'], dict):
        if 'sspineTheta' in track_data['derived']:
            spine_theta_raw = track_data['derived']['sspineTheta']
            if spine_theta_raw.ndim > 1:
                spine_theta_raw = spine_theta_raw.flatten()
            if len(spine_theta_raw) >= n_frames:
                spine_theta = spine_theta_raw[:n_frames]
        elif 'spineTheta' in track_data['derived']:
            spine_theta_raw = track_data['derived']['spineTheta']
            if spine_theta_raw.ndim > 1:
                spine_theta_raw = spine_theta_raw.flatten()
            if len(spine_theta_raw) >= n_frames:
                spine_theta = spine_theta_raw[:n_frames]
    
    # Get vel_dp (velocity dot product) if available
    vel_dp = None
    if 'derived' in track_data and isinstance(track_data['derived'], dict):
        if 'vel_dp' in track_data['derived']:
            vel_dp_raw = track_data['derived']['vel_dp']
            if vel_dp_raw.ndim > 1:
                vel_dp_raw = vel_dp_raw.flatten()
            if len(vel_dp_raw) >= n_frames:
                vel_dp = vel_dp_raw[:n_frames]
    
    # Also keep simple turn detection for backwards compatibility (heading_change > threshold)
    is_turn = heading_change > np.pi/6  # 30 degrees threshold (simpler detection)
    
    # Extract spine points per frame (if available)
    # Multiple spine points allow more accurate curvature computation
    spine_points_per_frame = None
    if 'spine_points' in track_data and 'spine_indices' in track_data:
        spine_points_all = track_data['spine_points']  # (N_total, 2) array
        spine_indices = track_data['spine_indices'].astype(int)  # Indices marking frame boundaries
        
        # Debug: print spine data info
        print(f"    Loading spine points: {len(spine_points_all)} total points, {len(spine_indices)} frame indices")
        
        # Extract spine points for each frame
        if len(spine_indices) > 1:
            n_spine_points_per_frame = int(spine_indices[1] - spine_indices[0])
        else:
            # Fallback: estimate from total points
            n_spine_points_per_frame = len(spine_points_all) // max(n_frames, 1) if n_frames > 0 else 11
            if n_spine_points_per_frame == 0:
                n_spine_points_per_frame = 11  # Default
        
        print(f"    Spine points per frame: {n_spine_points_per_frame}")
        spine_points_per_frame = np.zeros((n_frames, n_spine_points_per_frame, 2))
        
        for i in range(n_frames):
            if i < len(spine_indices) - 1:
                idx_start = spine_indices[i]
                idx_end = spine_indices[i + 1]
                if idx_end <= len(spine_points_all) and idx_start < len(spine_points_all):
                    frame_spine = spine_points_all[idx_start:idx_end]
                    if len(frame_spine) == n_spine_points_per_frame:
                        spine_points_per_frame[i] = frame_spine
                    elif len(frame_spine) > 0:
                        # Handle variable-length spines (pad or truncate)
                        min_len = min(len(frame_spine), n_spine_points_per_frame)
                        spine_points_per_frame[i, :min_len] = frame_spine[:min_len]
        
        frames_with_spines = np.sum(np.any(spine_points_per_frame != 0, axis=(1,2)))
        print(f"    Extracted spine points for {frames_with_spines}/{n_frames} frames")
        
        if frames_with_spines == 0:
            track_id = track_data.get('attrs', {}).get('id', 'unknown')
            raise ValueError(f"No valid spine points extracted from track (id={track_id}). "
                           f"Check that spine_points and spine_indices are correctly formatted in H5 file.")
    else:
        missing = []
        if 'spine_points' not in track_data:
            missing.append('spine_points')
        if 'spine_indices' not in track_data:
            missing.append('spine_indices')
        track_id = track_data.get('attrs', {}).get('id', 'unknown')
        raise ValueError(f"REQUIRED: Spine data not available in H5 file for track (id={track_id}). "
                        f"Missing: {missing}. "
                        f"The H5 file must contain spine_points and spine_indices in tracks/track_*/points/")
    
    # Compute spineTheta and curvature using MAGAT algorithms
    # MAGAT Reference: @MaggotTrack/calculateDerivedQuantity.m
    try:
        import sys
        from pathlib import Path
        script_dir = Path(__file__).parent
        sys.path.insert(0, str(script_dir))
        from magat_spine_analysis import calculate_spine_theta_magat, calculate_spine_curv_magat, calculate_spine_curve_energy_magat, lowpass1d
        
        if spine_points_per_frame is not None:
            # Use MAGAT's algorithm to compute spineTheta (body bend angle)
            spine_theta_magat = calculate_spine_theta_magat(spine_points_per_frame)  # (n_frames,)
            
            # Use MAGAT's algorithm to compute spineCurv
            spine_curv_magat = calculate_spine_curv_magat(spine_points_per_frame)  # (n_frames,)
            
            # Compute smoothed spineTheta (sspineTheta) - MAGAT lowpass filters spineTheta
            # MAGAT: sspineTheta = lowpass1D(spineTheta, smoothTime/interpTime)
            if len(spine_theta_magat) > 1:
                # Estimate smoothTime/interpTime from frame rate
                dt_avg = np.mean(dt[dt > 0]) if np.any(dt > 0) else 1.0 / frame_rate
                smooth_time = 0.1  # Typical MAGAT smoothTime (seconds)
                sigma_samples = smooth_time / dt_avg
                spine_theta_smoothed = lowpass1d(spine_theta_magat, sigma_samples)
            else:
                spine_theta_smoothed = spine_theta_magat
            
            # Use MAGAT-computed values
            # Override curvature with MAGAT's calculation
            curvature = spine_curv_magat
            
            # Store MAGAT spineTheta (body bend angle) for use in segmentation
            spine_theta_magat_computed = True
            print(f"    MAGAT spine analysis: computed spineTheta and spineCurv from {spine_points_per_frame.shape[1]} spine points")
        else:
            spine_theta_magat = None
            spine_theta_smoothed = None
            spine_theta_magat_computed = False
    except Exception as e:
        # Fallback to original method if MAGAT algorithms fail
        import traceback
        print(f"    Warning: MAGAT spine analysis failed ({e}), using simplified calculation")
        traceback.print_exc()
        spine_theta_magat = None
        spine_theta_smoothed = None
        spine_theta_magat_computed = False
        
        # Original simple curvature computation
        if spine_points_per_frame is not None:
            spine_curvatures = []
            for i in range(n_frames):
                spine = spine_points_per_frame[i]
                if len(spine) >= 3:
                    vecs = np.diff(spine, axis=0)
                    vec_norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                    vec_norms[vec_norms < 1e-6] = 1e-6
                    vecs_normalized = vecs / vec_norms
                    angles = []
                    for j in range(len(vecs_normalized) - 1):
                        dot = np.clip(np.dot(vecs_normalized[j], vecs_normalized[j+1]), -1, 1)
                        angle = np.arccos(dot)
                        angles.append(angle)
                    if len(angles) > 0:
                        total_angle = np.sum(angles)
                        total_length = np.sum(vec_norms.flatten())
                        frame_curvature = total_angle / (total_length + 1e-6)
                    else:
                        frame_curvature = 0.0
                else:
                    frame_curvature = 0.0
                spine_curvatures.append(frame_curvature)
            spine_curvature_from_points = np.array(spine_curvatures)
            if len(spine_curvature_from_points) == n_frames:
                curvature = spine_curvature_from_points
    
    # Compute spine curve energy using MAGAT method
    # MAGAT uses spineTheta^2 for curve energy, or curvature^2
    if spine_theta_magat_computed and spine_theta_magat is not None:
        # Use MAGAT's spineTheta-based energy
        spine_curve_energy = calculate_spine_curve_energy_magat(spine_theta_magat, spine_curv=None)
    else:
        # Fallback: use curvature^2
        curvature_normalized = np.clip(curvature, -1e6, 1e6)  # Clip extreme outliers
        spine_curve_energy = curvature_normalized ** 2  # Energy ∝ curvature²
    
    # Also compute absolute curvature for additional features
    curvature_abs = np.abs(curvature)
    
    # FULL MAGAT SEGMENTATION ALGORITHM (after spine analysis)
    # MAGAT Reference: https://github.com/GilRaitses/magniphyq
    # 
    # MAGAT's Algorithm (from @MaggotTrack/segmentTrack.m):
    #   1. Find runs (periods of forward movement: high speed, head aligned, low curvature)
    #   2. Find head swings (head swinging wide periods between runs)
    #   3. Group head swings into reorientations - a reorientation is the period BETWEEN runs
    #      (whether or not it contains head swings). Reorientations are gaps between runs.
    #
    try:
        import sys
        from pathlib import Path
        script_dir = Path(__file__).parent
        sys.path.insert(0, str(script_dir))
        from magat_segmentation import magat_segment_track, MaggotSegmentOptions
        
        # Prepare DataFrame for MAGAT segmentation
        # Need: time, speed, curvature, body_theta (spineTheta), vel_dp
        # Use MAGAT-computed spineTheta if available (from spine analysis above)
        # Otherwise use H5 values or approximate
        if spine_theta_magat_computed and spine_theta_magat is not None:
            body_theta_for_seg = spine_theta_magat
            body_theta_smooth = spine_theta_smoothed if spine_theta_smoothed is not None else spine_theta_magat
        elif spine_theta is not None:
            body_theta_for_seg = spine_theta
            body_theta_smooth = spine_theta  # Approximate
        else:
            # Fallback approximation
            body_theta_for_seg = np.abs(curvature) * 10
            body_theta_smooth = body_theta_for_seg
        
        magat_df = pd.DataFrame({
            'time': time,
            'speed': speed,
            'curvature': curvature,
            'curv': curvature,  # MAGAT uses 'curv'
            'spineTheta': body_theta_for_seg,  # MAGAT body bend angle
            'sspineTheta': body_theta_smooth,  # MAGAT smoothed body bend angle
            'heading': heading,
            'x': x,
            'y': y
        })
        
        # Add vel_dp if available
        if vel_dp is not None:
            magat_df['vel_dp'] = vel_dp
        else:
            # Approximate: assume good alignment if speed is high
            magat_df['vel_dp'] = np.ones(n_frames) * 0.707  # cos(45°)
        
        # Learn optimal segmentation parameters from data (parameter learning)
        try:
            from learn_magat_parameters import learn_optimal_parameters
            print("    Learning optimal MAGAT parameters from data...")
            segment_options = learn_optimal_parameters(
                trajectory_df=pd.DataFrame({
                    'time': time,
                    'speed': speed,
                    'curvature': curvature,
                    'heading': heading,
                    'spineTheta': body_theta_for_seg
                }),
                target_runs_per_minute=1.0,
                min_run_duration=2.5
            )
            print(f"    Learned parameters: curv_cut={segment_options.curv_cut:.4f}, "
                  f"theta_cut={np.rad2deg(segment_options.theta_cut):.1f}°, "
                  f"stop_speed={segment_options.stop_speed_cut:.6f}, "
                  f"start_speed={segment_options.start_speed_cut:.6f}")
        except Exception as e:
            # Fallback to defaults if learning fails
            print(f"    Warning: Parameter learning failed ({e}), using defaults")
            segment_options = MaggotSegmentOptions()
            segment_options.minRunTime = 2.5
            segment_options.minHeadSwingDuration = 0.05
            segment_options.minHeadSwingAmplitude = np.deg2rad(10)
        
        # Run quality filters
        segment_options.minRunLength = 0.0  # Minimum path length in cm (0 = disabled)
        segment_options.minRunSpeed = 0.0  # Minimum average speed during run (0 = disabled)
        segment_options.requireRunContinuous = True  # Require runs to be continuous (no gaps)
        
        # Head swing quality filters (if not set by learning)
        if not hasattr(segment_options, 'minHeadSwingDuration'):
            segment_options.minHeadSwingDuration = 0.05  # 50ms minimum
        if not hasattr(segment_options, 'minHeadSwingAmplitude'):
            segment_options.minHeadSwingAmplitude = np.deg2rad(10)  # 10° minimum
        segment_options.requireAccepted = False  # Set to True for MAGAT strict mode
        segment_options.requireValid = False  # Set to True if htValid data is available
        
        # Run MAGAT segmentation
        frame_rate_actual = 1.0 / np.mean(dt[dt > 0]) if np.any(dt > 0) else frame_rate
        segmentation = magat_segment_track(magat_df, segment_options=segment_options, frame_rate=frame_rate_actual)
        
        # Extract MAGAT results
        is_reorientation = segmentation['is_reorientation']  # Start events only
        is_run = segmentation['is_run']
        n_reorientations = segmentation['n_reorientations']
        
        print(f"    MAGAT segmentation: {segmentation['n_runs']} runs, {segmentation['n_head_swings']} head swings, {n_reorientations} reorientations")
        
        # Store segmentation for Klein run table generation
        magat_segmentation = segmentation
        
    except Exception as e:
        # Fallback to simplified detection if MAGAT segmentation fails
        import traceback
        print(f"    Warning: MAGAT segmentation failed ({e}), using simplified detection")
        traceback.print_exc()
        frame_rate_actual = 1.0 / np.mean(dt[dt > 0]) if np.any(dt > 0) else 10.0
        angular_velocity = heading_change / np.maximum(dt, 1e-6)  # rad/s
        
        # Simplified thresholds
        turn_threshold_rad_per_sec = 2.3
        speed_threshold = 0.0003
        
        is_reorientation_frame = (angular_velocity > turn_threshold_rad_per_sec) & (speed > speed_threshold)
        is_reorientation = np.zeros(n_frames, dtype=bool)
        if n_frames > 1:
            is_reorientation[1:] = is_reorientation_frame[1:] & (~is_reorientation_frame[:-1])
            if is_reorientation_frame[0]:
                is_reorientation[0] = True
        is_run = np.zeros(n_frames, dtype=bool)  # No run info in fallback
    
    # Also include head and tail positions
    head_x = track_data.get('head', mid_pos)[:, 0] if 'head' in track_data else x
    head_y = track_data.get('head', mid_pos)[:, 1] if 'head' in track_data else y
    tail_x = track_data.get('tail', mid_pos)[:, 0] if 'tail' in track_data else x
    tail_y = track_data.get('tail', mid_pos)[:, 1] if 'tail' in track_data else y
    
    # Store spine points data (if available)
    spine_data = {}
    if spine_points_per_frame is not None:
        # Store spine points as columns (spine_x_0, spine_y_0, spine_x_1, spine_y_1, ...)
        n_spine_pts = spine_points_per_frame.shape[1]
        for i in range(n_spine_pts):
            spine_data[f'spine_x_{i}'] = spine_points_per_frame[:, i, 0]
            spine_data[f'spine_y_{i}'] = spine_points_per_frame[:, i, 1]
    
    df = pd.DataFrame({
        'frame': frames,
        'time': time,
        'x': x,  # mid/centroid x
        'y': y,  # mid/centroid y
        'head_x': head_x,
        'head_y': head_y,
        'tail_x': tail_x,
        'tail_y': tail_y,
        'speed': speed,
        'heading': heading,
        'curvature': curvature,
        'curvature_abs': curvature_abs,
        'spine_curve_energy': spine_curve_energy,
        'acceleration': accel,
        'heading_change': np.concatenate([[0], heading_change[1:]]),
        'is_turn': is_turn,  # Simple heading change detection (backwards compatibility)
        'is_reorientation': is_reorientation,  # MAGAT reorientation detection (start events)
        'is_run': is_run if 'is_run' in locals() else np.zeros(n_frames, dtype=bool),  # MAGAT run detection
        **spine_data  # Add spine point columns
    })
    
    # Add MAGAT-computed spineTheta fields if available
    if spine_theta_magat_computed and spine_theta_magat is not None:
        df['spineTheta_magat'] = spine_theta_magat  # MAGAT body bend angle
        if spine_theta_smoothed is not None:
            df['sspineTheta_magat'] = spine_theta_smoothed  # MAGAT smoothed body bend angle
    
    # Add pause detection and turn duration quantification
    try:
        import sys
        from pathlib import Path
        # Add scripts directory to path if not already there
        scripts_dir = Path(__file__).parent
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from detect_events import add_event_detection
        df = add_event_detection(df)
    except (ImportError, Exception) as e:
        # Fallback: simple pause detection if module not available
        print(f"  Warning: Could not import detect_events, using simple detection: {e}")
        speed_threshold = 0.001
        pause_min_duration = 0.2  # seconds
        is_pause = (df['speed'] < speed_threshold).values
        pause_durations = np.zeros(len(df))
        turn_durations = np.zeros(len(df))
        turn_event_ids = np.zeros(len(df), dtype=int)
        is_reversal = (df['heading_change'] > np.pi/2).values
        
        df['is_pause'] = is_pause
        df['pause_duration'] = pause_durations
        df['turn_duration'] = turn_durations
        df['turn_event_id'] = turn_event_ids
        df['is_reversal'] = is_reversal
    
    # Generate Klein run table if MAGAT segmentation succeeded (NO FALLBACKS)
    klein_run_table = None
    if 'magat_segmentation' in locals() and magat_segmentation is not None:
        # Check if we have runs - Klein run table requires at least 1 run
        if magat_segmentation.get('n_runs', 0) == 0:
            print(f"    Skipping Klein run table generation: No runs detected (need at least 1 run)")
        else:
            try:
                import sys
                from pathlib import Path
                scripts_dir = Path(__file__).parent
                if str(scripts_dir) not in sys.path:
                    sys.path.insert(0, str(scripts_dir))
                from klein_run_table import generate_klein_run_table
                
                # Get track and experiment IDs
                track_id = track_data.get('attrs', {}).get('id', 1)
                experiment_id = track_data.get('attrs', {}).get('experiment_id', 1)
                
                # Generate Klein run table (NO FALLBACKS - will raise error if data missing)
                klein_run_table = generate_klein_run_table(
                    trajectory_df=df,
                    segmentation=magat_segmentation,
                    track_id=track_id,
                    experiment_id=experiment_id,
                    set_id=1
                )
                
                print(f"    Generated Klein run table: {len(klein_run_table)} runs/turns")
                
            except Exception as e:
                # NO FALLBACKS - re-raise error for data quality issues
                # But provide helpful error message
                track_id = track_data.get('attrs', {}).get('id', 1)
                raise ValueError(f"Failed to generate Klein run table for track {track_id}: {e}") from e
    
    # Store run table as attribute (for later access)
    if klein_run_table is not None:
        df.attrs['klein_run_table'] = klein_run_table
    
    return df

def compute_ton_toff(led_values: np.ndarray, threshold: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute _ton and _toff boolean arrays from LED values (equivalent to MAGAT addTonToff).
    
    Parameters
    ----------
    led_values : ndarray
        LED intensity values (1D array)
    threshold : float, optional
        Threshold for detecting ON state. If None, uses 10% of max value.
    
    Returns
    -------
    ton : ndarray
        Boolean array where True indicates LED is ON
    toff : ndarray
        Boolean array where True indicates LED is OFF
    """
    if threshold is None:
        threshold = np.max(led_values) * 0.1  # 10% of max
    
    # Determine ON/OFF state
    is_on = led_values > threshold
    
    # ton: True when LED is ON
    ton = is_on
    
    # toff: True when LED is OFF
    toff = ~is_on
    
    return ton, toff

def extract_stimulus_timing(h5_data: Dict, frame_rate: float = 10.0) -> pd.DataFrame:
    """
    Extract stimulus timing from H5 data.
    
    Uses stimulus onset frames to create 10-second pulses (fixed duration).
    
    Parameters
    ----------
    h5_data : dict
        H5 data dictionary with 'led_data' and 'stimulus' groups
    frame_rate : float
        Frame rate in Hz (default 10 fps)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: time, stimulus_on, intensity, time_since_stimulus, etc.
    """
    # Get LED1 data (red pulsing) and create time array
    if 'led1Val' in h5_data:
        led1_values = h5_data['led1Val']
        n_frames = len(led1_values)
        times = np.arange(n_frames) / frame_rate
    elif 'led_data' in h5_data:
        led1_values = h5_data['led_data']
        n_frames = len(led1_values)
        times = np.arange(n_frames) / frame_rate
    else:
        return pd.DataFrame()
    
    # Compute led1Val_ton and led1Val_toff (equivalent to addTonToff)
    led1Val_ton, led1Val_toff = compute_ton_toff(led1_values)
    
    # Get LED2 data (blue constant) if available
    led2_values = None
    led2Val_ton = None
    led2Val_toff = None
    if 'led2Val' in h5_data:
        led2_values = h5_data['led2Val']
        if len(led2_values) == n_frames:
            led2Val_ton, led2Val_toff = compute_ton_toff(led2_values)
    
    # FIXED: Pulse duration is always 10 seconds
    pulse_duration = 10.0
    pulse_duration_frames = int(pulse_duration * frame_rate)  # 100 frames at 10 fps
    
    # Create stimulus_on array from onset frames (more accurate than LED detection)
    stimulus_on = np.zeros(n_frames, dtype=bool)
    stimulus_onset = np.zeros(n_frames, dtype=bool)
    
    # Use onset frames if available (most accurate)
    if 'stimulus' in h5_data and 'onset_frames' in h5_data['stimulus']:
        onset_frames = h5_data['stimulus']['onset_frames']
        
        # Mark each pulse: from onset to onset + 10 seconds
        for onset_frame in onset_frames:
            onset_frame = int(onset_frame)
            pulse_end_frame = min(onset_frame + pulse_duration_frames, n_frames)
            
            # Mark pulse as ON
            stimulus_on[onset_frame:pulse_end_frame] = True
            # Mark onset
            if onset_frame < n_frames:
                stimulus_onset[onset_frame] = True
    else:
        # Fallback: use led1Val_ton (from addTonToff equivalent)
        stimulus_on = led1Val_ton
        # Detect onsets as transitions from OFF to ON
        led_diff = np.diff(led1_values, prepend=led1_values[0])
        stimulus_onset = (led_diff > 0) & led1Val_ton
    
    # Compute time since last stimulus onset
    time_since_stimulus = np.full(n_frames, np.nan)
    last_onset_time = np.nan
    
    for i, t in enumerate(times):
        if stimulus_onset[i]:
            last_onset_time = t
        
        if not np.isnan(last_onset_time):
            time_since_stimulus[i] = t - last_onset_time
    
    # Build DataFrame with all LED timing fields
    df_dict = {
        'time': times,
        'frame': np.arange(len(times)),
        'led1Val': led1_values,
        'led1Val_ton': led1Val_ton,
        'led1Val_toff': led1Val_toff,
        'stimulus_on': stimulus_on,
        'stimulus_onset': stimulus_onset,
        'time_since_stimulus': time_since_stimulus
    }
    
    # Add LED2 fields if available
    if led2_values is not None:
        df_dict['led2Val'] = led2_values
        df_dict['led2Val_ton'] = led2Val_ton
        df_dict['led2Val_toff'] = led2Val_toff
    
    df = pd.DataFrame(df_dict)
    
    return df

def align_trajectory_with_stimulus(trajectory_df: pd.DataFrame, 
                                   stimulus_df: pd.DataFrame) -> pd.DataFrame:
    """
    Align trajectory data with stimulus timing.
    
    Parameters
    ----------
    trajectory_df : pd.DataFrame
        Trajectory features
    stimulus_df : pd.DataFrame
        Stimulus timing
    
    Returns
    -------
    pd.DataFrame
        Combined DataFrame with aligned data
    """
    # Merge on time (with tolerance for frame alignment)
    merged = pd.merge_asof(
        trajectory_df.sort_values('time'),
        stimulus_df.sort_values('time'),
        on='time',
        direction='nearest',
        tolerance=0.05  # 50ms tolerance
    )
    
    # Compute time since last stimulus onset
    stimulus_onsets = merged[merged['stimulus_onset'] == True]['time'].values
    if len(stimulus_onsets) > 0:
        time_since_stimulus = np.zeros(len(merged))
        for i, t in enumerate(merged['time']):
            prev_onsets = stimulus_onsets[stimulus_onsets <= t]
            if len(prev_onsets) > 0:
                time_since_stimulus[i] = t - prev_onsets[-1]
        merged['time_since_stimulus'] = time_since_stimulus
    else:
        merged['time_since_stimulus'] = np.inf
    
    return merged

def create_event_records(trajectory_df: pd.DataFrame, 
                        track_id: int,
                        experiment_id: str) -> pd.DataFrame:
    """
    Create event records for hazard modeling.
    
    Parameters
    ----------
    trajectory_df : pd.DataFrame
        Aligned trajectory+stimulus data
    track_id : int
        Track identifier
    experiment_id : str
        Experiment identifier
    
    Returns
    -------
    pd.DataFrame
        Event records with time bins and event indicators
    """
    # Create time bins (e.g., 50ms bins)
    bin_width = 0.05  # 50ms
    time_bins = np.arange(
        trajectory_df['time'].min(),
        trajectory_df['time'].max() + bin_width,
        bin_width
    )
    
    # Assign each time point to a bin
    trajectory_df['time_bin'] = np.digitize(trajectory_df['time'], time_bins) - 1
    
    # Aggregate to bins (only aggregate columns that exist)
    # NOTE: Spine point coordinates (spine_x_*, spine_y_*) are NOT aggregated here
    # They remain at full resolution in the trajectory DataFrame. Only derived features
    # like spine_curve_energy are aggregated for event records.
    agg_dict = {
        'time': 'mean',
        'speed': 'mean',
        'heading': 'mean',
        'x': 'mean',
        'y': 'mean',
        'stimulus_on': 'any',
        'time_since_stimulus': 'mean',
        'is_turn': 'any',  # Simple heading change detection (backwards compatibility)
        'is_reorientation': 'any',  # Proper reorientation detection (USE THIS FOR TURN RATES)
        'is_pause': 'any',  # Event occurred if any frame in bin was paused
        'is_reversal': 'any',  # Event occurred if any frame in bin was reversal
        'curvature': 'mean',
        'spine_curve_energy': 'mean',  # Average bending energy per bin
        'turn_duration': 'mean',  # Average turn duration in bin
        'pause_duration': 'mean'  # Average pause duration in bin
    }
    
    # Add LED columns if they exist
    if 'led1Val' in trajectory_df.columns:
        agg_dict['led1Val'] = 'mean'
    if 'led1Val_ton' in trajectory_df.columns:
        agg_dict['led1Val_ton'] = 'any'
    if 'led1Val_toff' in trajectory_df.columns:
        agg_dict['led1Val_toff'] = 'any'
    if 'led2Val' in trajectory_df.columns:
        agg_dict['led2Val'] = 'mean'
    if 'led2Val_ton' in trajectory_df.columns:
        agg_dict['led2Val_ton'] = 'any'
    if 'led2Val_toff' in trajectory_df.columns:
        agg_dict['led2Val_toff'] = 'any'
    
    binned = trajectory_df.groupby('time_bin').agg(agg_dict).reset_index()
    
    # Add metadata
    binned['track_id'] = track_id
    binned['experiment_id'] = experiment_id
    
    return binned

def process_h5_file(h5_path: Path, output_dir: Path, experiment_id: str):
    """Process a single H5 file and extract data for modeling."""
    print(f"\nProcessing: {h5_path.name}")
    
    # Load H5 file
    try:
        h5_data = load_h5_file(h5_path)
    except Exception as e:
        print(f"  ERROR loading H5 file: {e}")
        return
    
    # Get frame rate from metadata
    frame_rate = 10.0  # default
    if 'metadata' in h5_data and 'attrs' in h5_data['metadata']:
        metadata_attrs = h5_data['metadata']['attrs']
        if 'fps' in metadata_attrs:
            frame_rate = float(metadata_attrs['fps'])
    
    # Extract stimulus timing
    stimulus_df = extract_stimulus_timing(h5_data, frame_rate=frame_rate)
    if len(stimulus_df) == 0:
        print("  WARNING: No stimulus data found")
    else:
        print(f"  Stimulus data: {len(stimulus_df)} frames, {stimulus_df['stimulus_onset'].sum()} onsets")
    
    # Process each track
    all_event_records = []
    all_trajectories = []
    all_klein_run_tables = []  # Collect Klein run tables for all tracks
    
    if 'tracks' in h5_data:
        for track_key, track_data in h5_data['tracks'].items():
            # Extract track ID from track_key (e.g., "track_1" -> 1)
            try:
                track_id = int(track_key.split('_')[-1])
            except:
                track_id = len(all_trajectories) + 1
            
            # Extract trajectory features
            traj_df = extract_trajectory_features(track_data, frame_rate=frame_rate)
            if len(traj_df) == 0:
                continue
            
            # Align with stimulus (skip if no stimulus data)
            if len(stimulus_df) > 0:
                aligned_df = align_trajectory_with_stimulus(traj_df, stimulus_df)
            else:
                # No stimulus data - just use trajectory data
                aligned_df = traj_df.copy()
                # Add empty stimulus columns
                aligned_df['led1Val'] = 0.0
                aligned_df['led1Val_ton'] = False
                aligned_df['led1Val_toff'] = True
                aligned_df['led2Val'] = 0.0
                aligned_df['led2Val_ton'] = False
                aligned_df['led2Val_toff'] = True
                aligned_df['stimulus_on'] = False
                aligned_df['stimulus_onset'] = False
                aligned_df['time_since_stimulus'] = np.nan
            
            # Create event records (aggregated for hazard modeling)
            event_records = create_event_records(aligned_df, track_id, experiment_id)
            all_event_records.append(event_records)
            
            # Keep full-resolution trajectories (including all spine points)
            # These are NOT aggregated - preserve full frame-level resolution
            all_trajectories.append(aligned_df)
            
            # Collect Klein run table if available
            if hasattr(traj_df, 'attrs') and 'klein_run_table' in traj_df.attrs:
                klein_rt = traj_df.attrs['klein_run_table'].copy()
                # Ensure track_id and experiment_id are set
                klein_rt['track_id'] = track_id
                klein_rt['experiment_id'] = experiment_id
                all_klein_run_tables.append(klein_rt)
            
            n_turns = event_records['is_turn'].sum()
            n_reorientations = event_records['is_reorientation'].sum() if 'is_reorientation' in event_records.columns else 0
            n_runs = len(traj_df.attrs.get('klein_run_table', [])) if hasattr(traj_df, 'attrs') and 'klein_run_table' in traj_df.attrs else 0
            print(f"  Track {track_id}: {len(traj_df)} frames, {n_turns} turns (simple), {n_reorientations} reorientations (proper), {n_runs} runs (Klein)")
    
    # Combine all tracks
    if all_event_records:
        combined_events = pd.concat(all_event_records, ignore_index=True)
        combined_trajectories = pd.concat(all_trajectories, ignore_index=True)
        
        # Save outputs
        output_dir.mkdir(parents=True, exist_ok=True)
        
        events_file = output_dir / f"{experiment_id}_events.csv"
        combined_events.to_csv(events_file, index=False)
        print(f"  ✓ Saved {len(combined_events)} event records to {events_file}")
        
        trajectories_file = output_dir / f"{experiment_id}_trajectories.csv"
        combined_trajectories.to_csv(trajectories_file, index=False)
        print(f"  ✓ Saved trajectory data to {trajectories_file}")
        
        # Save Klein run tables if available
        if all_klein_run_tables:
            combined_klein_runs = pd.concat(all_klein_run_tables, ignore_index=True)
            klein_runs_file = output_dir / f"{experiment_id}_klein_run_table.csv"
            combined_klein_runs.to_csv(klein_runs_file, index=False)
            print(f"  ✓ Saved {len(combined_klein_runs)} Klein run table rows to {klein_runs_file}")
            
            # Add to summary
            total_runs = int(len(combined_klein_runs))
            total_turns = int(combined_klein_runs['reoYN'].sum())
            total_head_swings = int(combined_klein_runs['reo#HS'].sum())
        else:
            total_runs = 0
            total_turns = 0
            total_head_swings = 0
        
        # Save summary
        summary = {
            'experiment_id': experiment_id,
            'n_tracks': len(all_event_records),
            'n_event_records': len(combined_events),
            'n_trajectory_points': len(combined_trajectories),
            'n_turns': int(combined_events['is_turn'].sum()),
            'n_reorientations': int(combined_events['is_reorientation'].sum()) if 'is_reorientation' in combined_events.columns else 0,
            'mean_turn_rate': float(combined_events['is_turn'].mean() / 0.05 * 60),  # turns/min (simple detection)
            'mean_reorientation_rate': float(combined_events['is_reorientation'].mean() / 0.05 * 60) if 'is_reorientation' in combined_events.columns else 0.0,  # reorientations/min (proper detection)
            'n_klein_runs': total_runs,
            'n_klein_turns': total_turns,
            'n_klein_head_swings': total_head_swings
        }
        
        summary_file = output_dir / f"{experiment_id}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"  ✓ Saved summary to {summary_file}")
    else:
        print("  WARNING: No tracks processed")

def main():
    parser = argparse.ArgumentParser(description='Engineer dataset from H5 files')
    parser.add_argument('--h5-dir', type=str, 
                       default='/Users/gilraitses/mechanosensation/h5tests',
                       help='Directory containing H5 files')
    parser.add_argument('--output-dir', type=str,
                       default='data/engineered',
                       help='Output directory for engineered data')
    parser.add_argument('--experiment-id', type=str, default=None,
                       help='Experiment ID (if None, uses filename)')
    parser.add_argument('--file', type=str, default=None,
                       help='Process single file (if None, processes all)')
    
    args = parser.parse_args()
    
    h5_dir = Path(args.h5_dir)
    output_dir = Path(args.output_dir)
    
    if not HAS_H5PY:
        print("ERROR: h5py not installed. Install with: pip install h5py")
        sys.exit(1)
    
    # Find H5 files
    if args.file:
        h5_files = [h5_dir / args.file]
    else:
        h5_files = sorted(h5_dir.glob("*.h5"))
    
    if not h5_files:
        print(f"No H5 files found in {h5_dir}")
        sys.exit(1)
    
    print(f"Found {len(h5_files)} H5 file(s)")
    
    # Process each file
    for h5_file in h5_files:
        experiment_id = args.experiment_id or h5_file.stem.replace(' ', '_')
        process_h5_file(h5_file, output_dir, experiment_id)
    
    print(f"\n✓ Processing complete. Outputs in {output_dir}")

if __name__ == '__main__':
    main()

