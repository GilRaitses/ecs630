# H5 File Stimulus Structure

## Key Findings

### Pulse Duration
- **Pulse duration is ALWAYS 10 seconds** (fixed, not variable)
- This contradicts the DOE table which has variable pulse durations (0.5s, 1.0s, 2.0s)
- The simulation should use **fixed 10-second pulses**

### Stimulus Parameters
- **LED1 (Red pulsing)**: Stored in `/led_data` array
  - Shape: (24001,) frames
  - Range: 0-250 (LED power)
  - Pulsing pattern: 10 seconds ON, 50 seconds OFF
  - 40 pulses total in experiment
  
- **LED2 (Blue constant)**: Not found in H5 files
  - May not be exported, or stored in different location
  - If needed, can be simulated as constant value

### Experiment Structure
- **FPS**: 10 frames/second
- **Total frames**: 24001 frames
- **Total time**: 2400.1 seconds (40 minutes)
- **Stimulus onsets**: Stored in `/stimulus/onset_frames` and `/stimulus/onset_times`
  - 40 onsets at 60-second intervals
  - First onset: frame 427 (42.7s)
  - Last onset: frame 23827 (2382.7s)

### Track Structure
```
/tracks/track_N/
  /head: (N_frames, 2) - head position
  /mid: (N_frames, 2) - centroid position  
  /tail: (N_frames, 2) - tail position
  /derived/
    /speed: (N_frames,) - speed
    /direction: (N_frames,) - heading direction
    /curvature: (N_frames,) - curvature
  /contour_points, /spine_points: (for tier1+)
```

## Implications for Simulation

### DOE Table Update Needed
The current DOE table has variable pulse durations (0.5s, 1.0s, 2.0s), but the actual experiment uses:
- **Fixed pulse duration**: 10 seconds
- **Variable inter-pulse intervals**: Can be 5s, 10s, 20s (as in DOE table)
- **Variable intensity**: 25%, 50%, 100% (as in DOE table)

### Stimulus Schedule
For simulation, create stimulus schedules with:
```python
pulse_duration = 10.0  # FIXED
inter_pulse_interval = [5, 10, 20]  # Variable (from DOE)
intensity_pct = [25, 50, 100]  # Variable (from DOE)
```

Cycle time = pulse_duration + inter_pulse_interval = 10 + interval

## Files to Update

1. **`scripts/simulate_trajectories.py`**:
   - Fix `create_stimulus_schedule()` to use **fixed 10-second pulse duration**
   - Update DOE condition handling to use `pulse_duration=10.0` always

2. **`scripts/engineer_dataset_from_h5.py`**:
   - Extract pulse timing from `/stimulus/onset_frames` 
   - Create pulses with 10-second duration from onsets
   - Map LED1 intensity to stimulus values

3. **`config/doe_table.csv`**:
   - Update pulse_duration_s column to all be 10.0 (or remove if fixed)
   - Keep inter_pulse_interval_s and intensity_pct as factors

4. **`config/model_config.json`**:
   - Document that pulse_duration is fixed at 10 seconds

