#!/usr/bin/env python3
"""
MAGAT Segmentation Algorithm Implementation in Python

Implements the full MAGAT track segmentation algorithm:
1. Detect runs (periods of forward movement)
2. Detect head swings (between runs)
3. Group into reorientations (gaps between runs)

Based on @MaggotTrack/segmentTrack.m from magniphyq repository.
Reference: https://github.com/GilRaitses/magniphyq
"""

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
from typing import Dict, List, Tuple, Optional

class MaggotSegmentOptions:
    """MAGAT segmentation options (defaults from MaggotSegmentOptions.m)"""
    def __init__(self):
        self.curv_cut = 0.4  # If track curvature > curv_cut, end a run
        self.autoset_curv_cut = False
        self.autoset_curv_cut_mult = 5
        self.theta_cut = np.pi / 2  # If body theta > theta_cut, end a run
        self.speed_field = 'speed'
        self.stop_speed_cut = 2.0  # If speed < stop_speed_cut, end a run (mm/s, convert to cm/s if needed)
        self.start_speed_cut = 3.0  # If speed > start_speed_cut && vel_dp > aligned_dp, run can start
        self.aligned_dp = np.cos(np.deg2rad(45))  # cos(45°) ≈ 0.707
        # Run quality filters (based on MAGAT @MaggotSegmentOptions)
        # These filters ensure only meaningful runs are detected:
        #
        # 1. minRunTime: Minimum run duration in seconds (MAGAT default: 2.5s)
        #    Runs shorter than this are discarded. Typical runs last 2-10 seconds.
        # 
        # 2. minRunLength: Minimum path length in cm (0 = disabled)
        #    Ensures runs cover meaningful distance. Typical runs: 0.1-1.0 cm.
        #    Uses path length (cumulative distance), not Euclidean displacement.
        #
        # 3. minRunSpeed: Minimum average speed during run (0 = disabled)
        #    Ensures runs involve meaningful forward movement.
        #    Units must match speed field (typically mm/s or cm/s).
        #
        # 4. requireRunContinuous: If True, verify run continuity (default: True)
        #    Ensures no gaps violate run criteria between start and end.
        #
        # Configuration examples:
        #   - Lenient: minRunTime=1.0, minRunLength=0.0, minRunSpeed=0.0
        #   - Default: minRunTime=2.5, minRunLength=0.0, minRunSpeed=0.0
        #   - Strict: minRunTime=2.5, minRunLength=0.1, minRunSpeed=0.001
        self.minRunTime = 2.5  # Minimum run duration in seconds
        self.minRunLength = 0.0  # Minimum run length (cm) - path length, not Euclidean distance
        self.minRunSpeed = 0.0  # Minimum average speed during run (same units as speed field)
        self.requireRunContinuous = True  # If True, runs must be continuous (no gaps)
        self.headswing_start = np.deg2rad(20)  # If body theta > headswing_start and not in run, start headswing
        self.headswing_stop = np.deg2rad(10)  # If body theta < headswing_stop (or changes sign), end headswing
        self.smoothBodyFromPeriFreq = False
        self.smoothBodyTime = None
        
        # Head swing quality filters (based on MAGAT analysis scripts)
        # These filters ensure only meaningful head swings are detected:
        # 
        # 1. minHeadSwingDuration: Removes very brief head movements (< 50ms default)
        #    Typical head swings last 0.2-0.4 seconds. Set to 0.0 to disable.
        # 
        # 2. minHeadSwingAmplitude: Minimum body bend angle (maxTheta) required
        #    Default: 10° (matches headswing_stop threshold). MAGAT analysis scripts
        #    typically use 10-20° for first head swings. Set to 0.0 to disable.
        # 
        # 3. requireAccepted: MAGAT's "accepted" criterion - head swing must end in a run
        #    This ensures the head swing led to forward movement. Set True for strict mode.
        # 
        # 4. requireValid: Require valid head-tail detection throughout head swing
        #    Only use if htValid/ihtValid data is available and reliable. Set True to enable.
        #
        # Configuration examples:
        #   - Lenient: minHeadSwingDuration=0.0, minHeadSwingAmplitude=0.0, requireAccepted=False
        #   - Default: minHeadSwingDuration=0.05, minHeadSwingAmplitude=deg2rad(10), requireAccepted=False
        #   - Strict: minHeadSwingDuration=0.1, minHeadSwingAmplitude=deg2rad(15), requireAccepted=True
        self.minHeadSwingDuration = 0.05  # Minimum head swing duration in seconds (default: 50ms, ~0.5 frames at 10Hz)
        self.minHeadSwingAmplitude = np.deg2rad(10)  # Minimum maxTheta amplitude in radians (default: 10°, matches headswing_stop)
        self.requireAccepted = False  # If True, only accept head swings that end in a run (MAGAT default: True)
        self.requireValid = False  # If True, require valid head-tail detection (if htValid available)


def magat_segment_track(trajectory_df: 'pd.DataFrame', 
                       segment_options: Optional[MaggotSegmentOptions] = None,
                       frame_rate: float = 10.0) -> Dict:
    """
    Full MAGAT segmentation algorithm implementation.
    
    Segments a maggot track into runs and reorientations following MAGAT's algorithm.
    
    Parameters
    ----------
    trajectory_df : pd.DataFrame
        Trajectory DataFrame with columns: time, speed, curvature, body_theta, vel_dp
        Must also have: x, y, heading (or theta)
    segment_options : MaggotSegmentOptions, optional
        Segmentation options (uses defaults if None)
    frame_rate : float
        Frame rate in Hz (default 10.0)
    
    Returns
    -------
    segmentation : dict
        Dictionary containing:
        - runs: List of (start_idx, end_idx) tuples for each run
        - head_swings: List of (start_idx, end_idx) tuples for each head swing
        - reorientations: List of (start_idx, end_idx) tuples for each reorientation
        - is_run: Boolean array indicating run frames
        - is_head_swing: Boolean array indicating head swing frames
        - is_reorientation: Boolean array indicating reorientation frames (start events)
    """
    if segment_options is None:
        segment_options = MaggotSegmentOptions()
    
    n_frames = len(trajectory_df)
    time = trajectory_df['time'].values
    dt = np.mean(np.diff(time)) if len(time) > 1 else 1.0 / frame_rate
    
    # Get derived quantities
    # Note: MAGAT uses different field names, we'll use what's available in our DataFrame
    speed = trajectory_df['speed'].values
    
    # Curvature (curv)
    if 'curvature' in trajectory_df.columns:
        curv = trajectory_df['curvature'].values
    elif 'curv' in trajectory_df.columns:
        curv = trajectory_df['curv'].values
    else:
        curv = np.zeros(n_frames)
    
    # Body theta (sspineTheta) - body bend angle
    if 'spineTheta' in trajectory_df.columns:
        body_theta = trajectory_df['spineTheta'].values
    elif 'sspineTheta' in trajectory_df.columns:
        body_theta = trajectory_df['sspineTheta'].values
    elif 'body_theta' in trajectory_df.columns:
        body_theta = trajectory_df['body_theta'].values
    else:
        # Approximate from heading changes or curvature
        if 'heading_change' in trajectory_df.columns:
            body_theta = np.abs(trajectory_df['heading_change'].values)
        else:
            body_theta = np.abs(curv) * 100  # Rough approximation
    
    # Velocity dot product (vel_dp) - alignment of velocity with head direction
    # This measures how aligned the head is with the direction of motion
    if 'vel_dp' in trajectory_df.columns:
        vel_dp = trajectory_df['vel_dp'].values
    else:
        # Approximate: compute from speed and heading alignment
        # vel_dp measures cos(angle between velocity and heading)
        # For now, assume good alignment if speed is high (simplified)
        vel_dp = np.ones(n_frames) * segment_options.aligned_dp
    
    # Auto-set curvature cut if requested
    if segment_options.autoset_curv_cut:
        if 'spineLength' in trajectory_df.columns:
            median_spine_length = np.median(trajectory_df['spineLength'].values)
        else:
            # Approximate spine length from track
            dx = np.diff(trajectory_df['x'].values, prepend=trajectory_df['x'].values[0])
            dy = np.diff(trajectory_df['y'].values, prepend=trajectory_df['y'].values[0])
            median_spine_length = np.median(np.sqrt(dx**2 + dy**2)) * 10  # Rough estimate
        segment_options.curv_cut = segment_options.autoset_curv_cut_mult / median_spine_length
    
    # Use learned parameters if available, otherwise auto-adjust
    # Check if parameters were learned (has _learned_params attribute)
    use_learned_params = hasattr(segment_options, '_learned_params')
    
    if use_learned_params:
        # Use learned parameters directly (already calibrated from data)
        stop_speed_cut = segment_options.stop_speed_cut
        start_speed_cut = segment_options.start_speed_cut
    else:
        # Auto-adjust thresholds based on speed distribution (legacy behavior)
    speed_median = np.median(speed[speed > 0])
    speed_max = np.max(speed[speed > 0])
    
    # Determine unit scale and adjust thresholds
    # MAGAT defaults: stop=2 mm/s, start=3 mm/s
    # If speeds are very small (< 0.1), likely cm/s
    # If speeds are larger (> 1), likely mm/s
    if speed_median < 0.1:  # Likely in cm/s (e.g., 0.01-0.05 cm/s)
        # Convert MAGAT thresholds from mm/s to cm/s
        stop_speed_cut = segment_options.stop_speed_cut / 10.0  # 2 mm/s = 0.2 cm/s
        start_speed_cut = segment_options.start_speed_cut / 10.0  # 3 mm/s = 0.3 cm/s
        # But these might still be too high - use percentile-based thresholds instead
            # Use more lenient thresholds: 5th percentile for stop, 15th percentile for start
            # This allows more frames to qualify as runs
        speed_sorted = np.sort(speed[speed > 0])
        if len(speed_sorted) > 10:
                stop_speed_cut = np.percentile(speed_sorted, 5)  # 5th percentile (more lenient)
                start_speed_cut = np.percentile(speed_sorted, 15)  # 15th percentile (more lenient)
                # Don't let thresholds get too low though
                stop_speed_cut = max(stop_speed_cut, speed_median * 0.1)
                start_speed_cut = max(start_speed_cut, speed_median * 0.2)
    else:
        stop_speed_cut = segment_options.stop_speed_cut
        start_speed_cut = segment_options.start_speed_cut
    
    # Step 1: Find everywhere NOT a run
    # A run ends if: high curvature OR head swinging OR speed too low
    # Note: Curvature values from tier2_complete can be very large (raw values)
    # Use relative thresholds or scale curvature
    curv_abs = np.abs(curv)
    
    # Use learned curvature threshold if available, otherwise auto-adjust
    if use_learned_params:
        curv_cut_actual = segment_options.curv_cut  # Use learned value directly
    else:
        # Auto-adjust curvature cut if values are very large (legacy behavior)
    curv_median = np.median(curv_abs[curv_abs > 0])
    if curv_median > 10.0:  # Curvature values seem to be in different units
        # Use percentile-based threshold instead
        curv_cut_actual = np.percentile(curv_abs[curv_abs > 0], 75)  # 75th percentile
    else:
        curv_cut_actual = segment_options.curv_cut
    
    highcurv = curv_abs > curv_cut_actual
    
    # Head swinging threshold for ending runs should be stricter than headswing_start
    # Use a larger threshold (body bend must be very severe to end a run)
    body_theta_abs = np.abs(body_theta)
    
    # Use learned theta threshold if available, otherwise auto-adjust
    if use_learned_params:
        theta_cut_actual = segment_options.theta_cut  # Use learned value directly
    else:
        # Auto-adjust using percentile-based approach (legacy behavior)
        # theta_cut = π/2 is 90°, which is too lenient - most frames will be marked
        # Use a percentile-based approach: only mark frames with extreme body bending
        if np.any(body_theta_abs > 0):
            # Use 90th percentile as threshold - only extreme bends end runs
            theta_cut_actual = np.percentile(body_theta_abs[body_theta_abs > 0], 90)
            # But don't make it too lenient - minimum of percentile and original cut
            theta_cut_actual = min(theta_cut_actual, segment_options.theta_cut)
        else:
            theta_cut_actual = segment_options.theta_cut
    
    head_swinging = body_theta_abs > theta_cut_actual
    speedlow = speed < stop_speed_cut
    
    notarun = highcurv | head_swinging | speedlow
    
    # Find run end indices (transitions from False to True in notarun)
    endarun = np.where(np.diff(notarun.astype(int)) >= 1)[0] + 1
    
    # Step 2: Find run start indices
    # A run can start if: NOT at stop point AND speed high enough AND head aligned
    speedhigh = speed >= start_speed_cut
    # vel_dp may be approximated - be more lenient if it's a constant value
    # If vel_dp is constant (approximated), don't use it as a strict requirement
    vel_dp_unique = len(np.unique(vel_dp))
    if vel_dp_unique <= 2:  # Constant or nearly constant (approximated)
        # Don't require head alignment if vel_dp is approximated
        headaligned = np.ones(n_frames, dtype=bool)
    else:
    headaligned = vel_dp >= segment_options.aligned_dp
    
    isarun = (~notarun) & speedhigh & headaligned
    
    # Find run start indices (transitions from False to True in isarun)
    startarun = np.where(np.diff(isarun.astype(int)) >= 1)[0] + 1
    
    # Step 3: Match starts and stops, create run intervals
    runs = []
    if len(startarun) > 0:
        si = 0
        while si < len(startarun):
            start_idx = startarun[si]
            # Find next end after this start
            ei = np.where(endarun > start_idx)[0]
            if len(ei) > 0:
                end_idx = endarun[ei[0]]
            else:
                end_idx = n_frames - 1
            
            # Find next start after this end
            next_si = np.where(startarun > end_idx)[0]
            if len(next_si) > 0:
                si = next_si[0]
            else:
                si = len(startarun)
            
            runs.append((start_idx, end_idx))
    
    # Step 4: Apply quality filters to runs
    valid_runs = []
    for start_idx, end_idx in runs:
        # Filter 1: Minimum duration
        run_duration = time[end_idx] - time[start_idx] if end_idx < len(time) else time[-1] - time[start_idx]
        if run_duration < segment_options.minRunTime:
            continue
        
        # Filter 2: Minimum path length (if enabled)
        if segment_options.minRunLength > 0:
            # Calculate path length (cumulative distance along trajectory)
            if 'pathLength' in trajectory_df.columns:
                path_start = trajectory_df['pathLength'].values[start_idx]
                path_end = trajectory_df['pathLength'].values[end_idx] if end_idx < len(trajectory_df) else trajectory_df['pathLength'].values[-1]
                run_path_length = path_end - path_start
            else:
                # Approximate path length from positions
                run_x = trajectory_df['x'].values[start_idx:end_idx+1]
                run_y = trajectory_df['y'].values[start_idx:end_idx+1]
                if len(run_x) > 1:
                    dx = np.diff(run_x)
                    dy = np.diff(run_y)
                    run_path_length = np.sum(np.sqrt(dx**2 + dy**2))
                else:
                    run_path_length = 0.0
            
            if run_path_length < segment_options.minRunLength:
                continue
        
        # Filter 3: Minimum average speed (if enabled)
        if segment_options.minRunSpeed > 0:
            run_speeds = speed[start_idx:end_idx+1]
            avg_speed = np.mean(run_speeds[run_speeds > 0]) if np.any(run_speeds > 0) else 0.0
            if avg_speed < segment_options.minRunSpeed:
                continue
        
        # Filter 4: Check continuity (runs should not have gaps if required)
        # Note: The segmentation algorithm already ensures continuity by matching starts/stops,
        # but we can verify that intermediate frames meet run criteria
        if segment_options.requireRunContinuous:
            # Verify that frames between start and end meet run criteria
            # (not in notarun, speed high enough, head aligned)
            mid_frames = notarun[start_idx+1:end_idx] if end_idx > start_idx + 1 else []
            if len(mid_frames) > 0 and np.any(mid_frames):
                # Some intermediate frames violate run criteria - skip this run
                continue
        
        # All filters passed - add to valid runs
            valid_runs.append((start_idx, end_idx))
    runs = valid_runs
    
    # Create is_run boolean array
    is_run = np.zeros(n_frames, dtype=bool)
    for start_idx, end_idx in runs:
        is_run[start_idx:end_idx+1] = True
    
    notrun = ~is_run
    
    # Step 5: Find head swings
    # Head swings are periods where head swings wide AND not in a run
    # Buffer around runs to allow head sweeps to start/end near run boundaries
    buffer = max(1, int((0.2 + 0.1) / dt))  # Approximate smoothTime + derivTime
    
    # Only look for head swings between first and last run (with buffer)
    if np.any(is_run):
        first_run_ind = np.where(is_run)[0][0] + buffer
        last_run_ind = np.where(is_run)[0][-1] - buffer
        inrange = np.zeros(n_frames, dtype=bool)
        inrange[first_run_ind:last_run_ind+1] = True
    else:
        inrange = np.ones(n_frames, dtype=bool)
    
    # Dilate notrun to allow head sweeps to start right at end of runs
    notrun_dilated = binary_dilation(notrun, structure=np.ones(buffer))
    
    # Head swinging condition: body theta > headswing_start AND not in run AND in range
    head_swinging_condition = (np.abs(body_theta) > segment_options.headswing_start) & notrun_dilated & inrange
    
    # Find head swing start indices
    head_swinging_indices = np.where(head_swinging_condition)[0]
    
    # Erode runs to allow head sweeps to end inside runs
    isrun_eroded = binary_erosion(is_run, structure=np.ones(buffer))
    
    # Head swing ends: body theta < headswing_stop OR sign change OR back in run
    body_theta_sign_change = np.zeros(n_frames, dtype=bool)
    if n_frames > 1:
        body_theta_sign_change[1:] = np.diff(np.sign(body_theta)) != 0
    
    not_head_swing_condition = ((np.abs(body_theta) < segment_options.headswing_stop) | 
                                body_theta_sign_change | 
                                isrun_eroded) & inrange
    
    not_head_swing_indices = np.where(not_head_swing_condition)[0]
    
    # Match head swing starts and stops
    head_swings_raw = []
    if len(head_swinging_indices) > 0:
        si = 0
        while si < len(head_swinging_indices):
            start_idx = head_swinging_indices[si]
            # Find next stop after this start
            ei = np.where(not_head_swing_indices > start_idx)[0]
            if len(ei) > 0:
                end_idx = not_head_swing_indices[ei[0]]
            else:
                end_idx = n_frames - 1
            
            # Head swing is only valid if it includes at least one point that is not a run
            if np.any(notrun[start_idx:end_idx+1]):
                head_swings_raw.append((start_idx, end_idx))
            
            # Find next start after this end
            next_si = np.where(head_swinging_indices > end_idx)[0]
            if len(next_si) > 0:
                si = next_si[0]
            else:
                si = len(head_swinging_indices)
    
    # Apply quality filters to head swings
    head_swings = []
    for start_idx, end_idx in head_swings_raw:
        # Filter 1: Minimum duration
        head_swing_duration = time[end_idx] - time[start_idx] if end_idx < len(time) else time[-1] - time[start_idx]
        if head_swing_duration < segment_options.minHeadSwingDuration:
            continue
        
        # Filter 2: Minimum amplitude (maxTheta)
        # Find maximum body_theta within the head swing
        hs_body_theta = body_theta[start_idx:end_idx+1]
        max_theta = np.max(np.abs(hs_body_theta)) if len(hs_body_theta) > 0 else 0.0
        if max_theta < segment_options.minHeadSwingAmplitude:
            continue
        
        # Filter 3: Must contain at least one non-run frame (already checked above, but verify)
        if not np.any(notrun[start_idx:end_idx+1]):
            continue
        
        # Filter 4: Require accepted (ends in a run) if enabled
        if segment_options.requireAccepted:
            if end_idx >= len(is_run) or not is_run[end_idx]:
                continue
        
        # Filter 5: Require valid head-tail detection if enabled and available
        if segment_options.requireValid:
            if 'htValid' in trajectory_df.columns or 'ihtValid' in trajectory_df.columns:
                ht_valid_col = 'htValid' if 'htValid' in trajectory_df.columns else 'ihtValid'
                hs_ht_valid = trajectory_df[ht_valid_col].values[start_idx:end_idx+1]
                if not np.all(hs_ht_valid):
                    continue
        
        # All filters passed - add to valid head swings
        head_swings.append((start_idx, end_idx))
    
    # Create is_head_swing boolean array
    is_head_swing = np.zeros(n_frames, dtype=bool)
    for start_idx, end_idx in head_swings:
        is_head_swing[start_idx:end_idx+1] = True
    
    # Step 6: Group head swings into reorientations
    # A reorientation is the period BETWEEN runs (whether or not it contains head swings)
    # Reorientations are gaps between consecutive runs
    reorientations = []
    
    if len(runs) > 1:
        for i in range(len(runs) - 1):
            # Reorientation is the gap between run i and run i+1
            prev_run_end = runs[i][1]
            next_run_start = runs[i+1][0]
            
            # Reorientation start is right after previous run ends
            reo_start = prev_run_end + 1
            # Reorientation end is right before next run starts
            reo_end = next_run_start - 1
            
            if reo_start <= reo_end:
                reorientations.append((reo_start, reo_end))
    
    # Create is_reorientation boolean array (mark START of each reorientation)
    is_reorientation = np.zeros(n_frames, dtype=bool)
    for start_idx, end_idx in reorientations:
        is_reorientation[start_idx] = True  # Mark start event only
    
    return {
        'runs': runs,
        'head_swings': head_swings,
        'reorientations': reorientations,
        'is_run': is_run,
        'is_head_swing': is_head_swing,
        'is_reorientation': is_reorientation,  # Start events only
        'n_runs': len(runs),
        'n_head_swings': len(head_swings),
        'n_reorientations': len(reorientations)
    }

