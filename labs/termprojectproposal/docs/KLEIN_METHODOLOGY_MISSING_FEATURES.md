# Missing Features from Klein Methodology

Based on comparison with Klein Lab's run table methodology, here are the key features that are currently missing from our implementation:

## 1. **18-Column Run Table Structure**

We don't generate Klein-style run tables. Each row should represent one RUN or TURN event with 18 standardized columns:

### Core Identification (Columns 1-4)
- `set`: Experiment set number (usually 1)
- `expt`: Experiment number within set
- `track`: Individual track ID
- `time0`: Run start time (seconds)

### Behavioral State (Column 5)
- `reoYN`: Binary (0=NO, 1=YES) - whether run ends in a valid reorientation/turn

### Run Characteristics (Columns 6-9)
- `runQ`: Average run direction (radians, -π to +π) - **MISSING**
- `runL`: Run path length (pixels) - **MISSING (have frame-level, not run-level)**
- `runT`: Run duration (seconds) - **MISSING (have frame-level, not run-level)**
- `runX`: End X position (redundant with column 17)

### Turn Analysis (Columns 10-13)
- `reo#HS`: Number of head swings in turn - **MISSING (not associating head swings with turns)**
- `reoQ1`: Direction at end of run (radians) - **MISSING**
- `reoQ2`: Direction at end of turn (radians) - **MISSING**
- `reoHS1`: First head swing magnitude and direction (radians) - **MISSING**

### Directional Analysis (Column 14)
- `runQ0`: Direction at start of run (radians) - **MISSING**

### Spatial Coordinates (Columns 15-18)
- `runX0`: Start X position (pixels) - **MISSING (have frame-level, not run-level)**
- `runY0`: Start Y position (pixels) - **MISSING (have frame-level, not run-level)**
- `runX1`: End X position (pixels) - **MISSING (have frame-level, not run-level)**
- `runY1`: End Y position (pixels) - **MISSING (have frame-level, not run-level)**

## 2. **Run-Level Metrics**

Currently we have frame-level metrics but not run-level aggregations:

### Missing Calculations:
- **runQ (average direction)**: Mean heading over entire run
- **runQ0 (start direction)**: Heading at first frame of run
- **runQ1 (end direction)**: Heading at last frame of run (becomes reoQ1)
- **runL (path length)**: Cumulative distance traveled along run trajectory
- **runT (duration)**: Time span from start to end of run
- **runX0, runY0 (start position)**: Spatial coordinates at run start
- **runX1, runY1 (end position)**: Spatial coordinates at run end

## 3. **Turn-Level Metrics**

### Missing Calculations:
- **reoQ2 (turn end direction)**: Heading at end of turn (start of next run)
- **reo#HS (head swing count)**: Number of head swings within each turn
- **reoHS1 (first head swing)**: Magnitude and direction of first head swing in turn

### Missing Associations:
- **Head swing → Turn mapping**: We detect head swings but don't associate them with specific turns
- **Multiple head swings per turn**: We don't count or analyze multiple head swings within a single turn

## 4. **Derived Metrics**

### Turn Analysis:
```python
# Turn magnitude with angle wrapping - MISSING
Δθ = θ₂ - θ₁
if Δθ < -π: Δθ += 2π
if Δθ > +π: Δθ -= 2π

# Turn direction - MISSING
LEFT turn: Δθ > 0
RIGHT turn: Δθ < 0
```

### Run Characteristics:
```python
# Run efficiency - MISSING
displacement = √[(x₁-x₀)² + (y₁-y₀)²]
efficiency = displacement / path_length

# Average speed - MISSING
speed = path_length / duration

# Turn rate - MISSING (proper calculation)
turn_rate = (Σ reoYN) / (Σ runT) * 60  # turns per minute
```

### Run Drift Analysis:
```python
# Run direction change - MISSING
drift = θ₁ - θ₀
LEFT drift: drift > 0
RIGHT drift: drift < 0
```

## 5. **Head Swing Analysis**

### Missing Features:
- **Head swing acceptance/rejection**: First head swing may be "rejected" (larva doesn't commit to that direction) - we don't track this
- **Head swing magnitude per turn**: We detect head swings but don't calculate their contribution to turn magnitude
- **Head swing direction**: We don't track LEFT vs RIGHT for individual head swings

## 6. **Behavioral Classification**

### Missing:
- **Run vs Turn distinction**: We detect runs and reorientations separately, but don't create unified run table where each row is a run OR turn
- **Turn completion criteria**: Turn only complete when larva resumes forward motion - we don't validate this
- **Pause detection**: Short pauses don't have turns - we detect pauses but don't integrate with run table structure

## 7. **Turn Rate Calculation**

### Current Implementation:
- We calculate turn rates at frame-level or temporal bin level
- We don't use Klein's specific formula: `turn_rate = (Σ reoYN) / (Σ runT) * 60`

## Implementation Priority

### High Priority (Core Klein Methodology):
1. ✅ Run detection (MAGAT-based) - **DONE**
2. ✅ Head swing detection (MAGAT-based) - **DONE**
3. ✅ Reorientation detection (MAGAT-based) - **DONE**
4. ✅ Reverse crawl detection (Klein methodology) - **DONE**
5. ❌ **18-column run table generation** - **HIGH PRIORITY**
6. ❌ **Run-level metrics** (runQ, runQ0, runQ1, runL, runT, coordinates) - **HIGH PRIORITY**
7. ❌ **Turn-level metrics** (reoQ2, reo#HS, reoHS1) - **HIGH PRIORITY**

### Medium Priority (Derived Metrics):
8. ❌ **Turn magnitude calculation** (with angle wrapping)
9. ❌ **Turn direction** (LEFT vs RIGHT)
10. ❌ **Run efficiency** (displacement/path_length)
11. ❌ **Run drift analysis** (θ₁ - θ₀)
12. ❌ **Head swing → Turn association**

### Lower Priority (Advanced Analysis):
13. ❌ **Head swing acceptance/rejection** tracking
14. ❌ **Multiple head swings per turn** analysis
15. ❌ **Turn rate** using Klein formula

## Current Status

### What We Have:
- ✅ Frame-level trajectory features (x, y, speed, heading, curvature)
- ✅ MAGAT segmentation (runs, head swings, reorientations)
- ✅ Reverse crawl detection (movement-orientation angle analysis)
- ✅ Simple turn detection (heading change threshold)
- ✅ Pause detection
- ✅ Temporal binning for stimulus-response analysis

### What We're Missing:
- ❌ **Run table structure** (rows = runs/turns, not frames)
- ❌ **Run-level aggregations** (direction, length, duration, coordinates)
- ❌ **Turn magnitude and direction** calculations
- ❌ **Head swing → Turn associations**
- ❌ **Run efficiency and drift** metrics
- ❌ **First head swing analysis**

## Next Steps

To fully implement Klein methodology, we need to:

1. **Create run table generator** that processes MAGAT segmentation output
2. **Calculate run-level metrics** from frame-level data
3. **Associate head swings with turns** to count reo#HS
4. **Calculate turn magnitude** (reoQ2 - reoQ1) with angle wrapping
5. **Generate 18-column Klein run table** per track
6. **Calculate derived metrics** (efficiency, drift, turn direction)

This would enable proper Klein-style analysis including:
- Turn rate calculations
- Spatial navigation analysis
- Directional bias studies
- Stimulus-response analysis at run/turn level

