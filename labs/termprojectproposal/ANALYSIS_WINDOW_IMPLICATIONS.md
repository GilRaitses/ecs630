# Analysis Window Implications for Simulation Model

## Updated Analysis Window
- **Previous**: [-2.0, 20.0] seconds (22 seconds total)
- **New**: [-3.0, 8.0] seconds (11 seconds total)
  - **3 seconds before** stimulus onset
  - **8 seconds after** stimulus onset

## What This Means for the Model

### 1. **Temporal Kernel Coverage**
- The raised cosine basis functions now span **-3s to +8s** relative to each stimulus onset
- This is a **shorter window** focused on the immediate stimulus response
- The kernel captures:
  - **Pre-stimulus baseline** (3s before): Spontaneous behavior patterns
  - **Stimulus response** (0-8s after): Immediate behavioral changes
  - **Post-stimulus decay** (covered within 8s window)

### 2. **Feature Extraction**
- When computing kernel features at time `t`, we only consider stimulus events within the window:
  - `t - 3 <= stimulus_time <= t + 8` (relative to nearest onset)
- Stimulus events outside this window have **zero contribution** to the hazard rate
- This reduces computational load (smaller history buffer needed)

### 3. **Simulation Implications**

#### Stimulus History Tracking
- **Buffer size**: Only need to track stimulus history for the **last 11 seconds** (actually 8s forward + 3s backward lookback)
- For stimuli with 60s cycle period, only the **current pulse** matters (pulses are 20s duration, so previous pulses are outside the window)
- Need to track stimulus onsets and maintain history buffer: `[(time, intensity), ...]` for window [-3, 8] seconds

#### Stimulus Window Relative to Onset
For each stimulus onset at time `t_onset`:
- Analysis window: `[t_onset - 3, t_onset + 8]`
- At simulation time `t`, compute kernel features relative to nearest onset:
  ```python
  tau = t - t_onset  # Time since onset
  if tau < -3.0 or tau > 8.0:
      # Outside analysis window - no stimulus effect
      kernel_features = 0
  else:
      # Within window - compute kernel convolution
      kernel_features = kernel_basis[tau] * stimulus_intensity
  ```

#### Overlapping Windows
- With **60s cycle period** and **20s pulse duration**:
  - Pulse 1: 0-20s → Analysis window: [-3, 28s] for onset at 0s (but only -3 to +8 matters)
  - Pulse 2: 60-80s → Analysis window: [57, 68s] for onset at 60s
  - **No overlap** between analysis windows (60s gap > 8s window)
  - Each pulse is analyzed independently

#### Memory Efficiency
- Previously: Needed to track stimulus history for 20+ seconds
- Now: Only need to track **11 seconds** of history
- Stimulus buffer: `deque(maxlen=int(11 / dt))` where `dt` is timestep

### 4. **Model Fitting**
- Feature extraction in `fit_hazard_model.py` will now:
  - Only extract kernel features when `time_since_stimulus` is in [-3, 8]
  - Zero out features outside this window
  - This focuses the model on **immediate stimulus responses**

### 5. **Behavioral Interpretation**
- **-3 to 0s**: Pre-stimulus behavior (baseline)
- **0 to 8s**: Immediate response window
  - Captures early turn responses (< 8s latency)
  - Misses late responses (> 8s latency)
- **Trade-off**: Focused on **fast, stimulus-locked responses** rather than delayed effects

### 6. **Simulation Code Changes Needed**

#### In `simulate_trajectories.py`:

1. **Update stimulus history tracking**:
   ```python
   # Only keep history within window
   max_history_time = 11.0  # 3s before + 8s after
   stimulus_history = [(t, intensity) for t, intensity in stimulus_history 
                        if current_time - t <= max_history_time]
   ```

2. **Update kernel feature extraction**:
   ```python
   # In extract_stimulus_kernel_features_at_time():
   # Filter stimulus_history to only include events within [-3, 8] window
   for hist_time, hist_intensity in stimulus_history:
       tau = t - hist_time
       if tau < -3.0 or tau > 8.0:
           continue  # Skip events outside analysis window
   ```

3. **Stimulus effect cutoff**:
   - After 8 seconds post-onset, stimulus no longer affects hazard rate
   - Behavior returns to baseline (or affected by next pulse if it starts)

### 7. **Expected Impact**

#### Advantages:
- **Focused model**: Captures immediate stimulus responses
- **Computational efficiency**: Smaller history buffer
- **Clearer interpretation**: Kernel shows response within 8s window
- **Matches experimental design**: Aligns with analysis window

#### Limitations:
- **Misses delayed responses**: Any behavioral changes > 8s after onset won't be captured
- **May underestimate**: If larvae respond with long latency, model will miss it
- **Cycle interactions**: Previous pulses won't affect current analysis (but with 60s gaps, this is fine)

### 8. **Validation**
- Check that most turn responses occur within 8s window (PSTH analysis)
- Verify that kernel coefficients drop to ~0 near +8s boundary
- Ensure baseline behavior (pre-stimulus) is well-captured in -3 to 0s window

## Summary
The **[-3, +8] second analysis window** means:
1. ✅ **Shorter, focused** kernel (11s vs 22s)
2. ✅ **Lower memory** requirements (smaller history buffer)
3. ✅ **Faster computation** (fewer kernel features to compute)
4. ⚠️ **Focuses on immediate responses** (may miss delayed effects)
5. ✅ **Clean separation** between pulses (60s gap >> 8s window)

