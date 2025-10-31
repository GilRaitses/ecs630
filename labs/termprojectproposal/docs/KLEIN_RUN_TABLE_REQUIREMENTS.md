# Klein Run Table Generator - Data Requirements

**NO FALLBACKS Policy**: The Klein run table generator strictly requires all data and raises `ValueError` if anything is missing. This ensures data quality and prevents silent failures.

## Required Input Data

### 1. **Trajectory DataFrame** (`trajectory_df`)

Must be a pandas DataFrame with the following **required columns**:

| Column | Type | Description | Used For |
|--------|------|-------------|----------|
| `time` | float | Time in seconds for each frame | Run duration (`runT`), start time (`time0`) |
| `x` | float | X coordinates (centroid position) | Spatial coordinates (`runX0`, `runX1`), path length (`runL`) |
| `y` | float | Y coordinates (centroid position) | Spatial coordinates (`runY0`, `runY1`), path length (`runL`) |
| `heading` | float | Heading angle in radians [-π, +π] | Direction metrics (`runQ`, `runQ0`, `reoQ1`, `reoQ2`) |

**Error if missing:**
```python
ValueError("Missing required columns in trajectory_df: ['time', 'x', 'y', 'heading']")
```

**Note:** The `speed` column is NOT required (it's mentioned in docstring but not actually used in the code).

### 2. **MAGAT Segmentation Dictionary** (`segmentation`)

Must be a dictionary with the following **required keys**:

| Key | Type | Description | Used For |
|-----|------|-------------|----------|
| `runs` | `List[Tuple[int, int]]` | List of (start_idx, end_idx) for each run | Core run table structure |
| `head_swings` | `List[Tuple[int, int]]` | List of (start_idx, end_idx) for each head swing | Turn analysis (`reo#HS`, `reoHS1`) |
| `reorientations` | `List[Tuple[int, int]]` | List of (start_idx, end_idx) for each reorientation | Turn identification (`reoYN`) |

**Errors if missing:**
```python
ValueError("segmentation must contain 'runs'")
ValueError("segmentation must contain 'head_swings'")
ValueError("segmentation must contain 'reorientations'")
```

## Data Quality Validations

### 3. **Runs Must Exist**

**Requirement:** At least one run must be present in segmentation.

**Error if missing:**
```python
ValueError(f"No runs found in segmentation for track {track_id}")
```

### 4. **Valid Run Indices**

**Requirements:**
- `run_start >= 0` (not negative)
- `run_end < n_frames` (within trajectory length)
- `run_end >= run_start` (end after start)

**Error if invalid:**
```python
ValueError(f"Invalid run indices for track {track_id}, run {run_idx}: "
           f"start={run_start}, end={run_end}, n_frames={n_frames}")
```

**Example invalid cases:**
- Run indices out of bounds: `(0, 150)` when trajectory has only 100 frames
- Negative indices: `(-1, 10)`
- Invalid order: `(50, 20)` (end before start)

### 5. **Valid Time Progression**

**Requirement:** Run duration must be positive (`time[end] - time[start] > 0`).

**Error if invalid:**
```python
ValueError(f"Negative run duration for track {track_id}, run {run_idx}")
```

**Note:** This happens if time array is not monotonically increasing or has duplicate timestamps.

### 6. **Turn Metrics Require Next Run**

**Requirement:** If a run ends in a turn (`reoYN == 1`), there must be a next run to calculate `reoQ2`.

**Error if missing:**
```python
ValueError(f"Run {run_idx} has reoYN=1 but no next run")
```

**Note:** This shouldn't happen if segmentation is correct (last run shouldn't have `reoYN=1`).

### 7. **Next Run Start Index Valid**

**Requirement:** When calculating `reoQ2` (direction at end of turn), the next run start index must be within trajectory bounds.

**Error if invalid:**
```python
ValueError(f"Next run start index {next_run_start_idx} >= n_frames {n_frames}")
```

### 8. **Turn Metrics Must Have Valid Angles**

**Requirement:** For rows with `reoYN == 1`, both `reoQ1` and `reoQ2` must be finite (not NaN).

**Error if invalid:**
```python
ValueError(f"NaN values in reoQ1 or reoQ2 for turns at indices: {invalid_idx}")
```

**Note:** This happens in `calculate_klein_derived_metrics()` when calculating turn magnitude.

### 9. **Turn Rate Calculation Requires Time**

**Requirement:** Total run time must be > 0 for turn rate calculation.

**Error if invalid:**
```python
ValueError("Total run time must be > 0 for turn rate calculation")
```

## Integration Requirements

### 10. **MAGAT Segmentation Must Succeed**

When integrated into `engineer_dataset_from_h5.py`:

**Requirement:** MAGAT segmentation must complete successfully. If it fails, `magat_segmentation` will be `None` and no run table is generated.

**Behavior:** No error is raised - run table generation is simply skipped. The frame-level DataFrame is still returned, but `df.attrs['klein_run_table']` will not exist.

**To enforce strict mode** (raise error if segmentation fails), you would need to modify `engineer_dataset_from_h5.py` to remove the fallback exception handling.

## What Happens When Data is Missing?

### **Strict "No Fallbacks" Behavior:**

1. **Missing columns** → `ValueError` immediately
2. **Missing segmentation keys** → `ValueError` immediately  
3. **Invalid indices** → `ValueError` when processing that run
4. **Invalid turn data** → `ValueError` when calculating derived metrics
5. **MAGAT segmentation failure** → Run table not generated (no error, but no table)

### **No Approximations Made:**

- ❌ No default values substituted
- ❌ No interpolation of missing data
- ❌ No skipping of invalid runs (processes all or fails)
- ❌ No fallback to simpler calculations

### **What IS Allowed:**

- ✅ `NaN` values in turn metrics for runs without turns (`reoYN == 0`)
- ✅ Empty head swing lists (counts as 0 head swings)
- ✅ Runs without following turns (last run has `reoYN == 0`)

## Example Error Scenarios

### Scenario 1: Missing 'heading' Column
```python
df = pd.DataFrame({'time': [0, 1, 2], 'x': [0, 1, 2], 'y': [0, 1, 2]})
# Missing 'heading'
seg = {'runs': [(0, 2)], 'head_swings': [], 'reorientations': []}
generate_klein_run_table(df, seg, track_id=1)
# Raises: ValueError("Missing required columns in trajectory_df: ['heading']")
```

### Scenario 2: No Runs Detected
```python
df = pd.DataFrame({'time': [0, 1, 2], 'x': [0, 1, 2], 'y': [0, 1, 2], 
                   'heading': [0, 0, 0]})
seg = {'runs': [], 'head_swings': [], 'reorientations': []}
generate_klein_run_table(df, seg, track_id=1)
# Raises: ValueError("No runs found in segmentation for track 1")
```

### Scenario 3: Invalid Run Indices
```python
df = pd.DataFrame({'time': [0, 1, 2], 'x': [0, 1, 2], 'y': [0, 1, 2], 
                   'heading': [0, 0, 0]})
seg = {'runs': [(0, 10)], 'head_swings': [], 'reorientations': []}  # end > n_frames
generate_klein_run_table(df, seg, track_id=1)
# Raises: ValueError("Invalid run indices for track 1, run 0: start=0, end=10, n_frames=3")
```

### Scenario 4: Turn Without Next Run
```python
# This is a logic error - shouldn't happen with proper MAGAT segmentation
# But if it does, the code will catch it:
# Raises: ValueError("Run 0 has reoYN=1 but no next run")
```

## Summary

The Klein run table generator requires:

1. **Complete trajectory data**: `time`, `x`, `y`, `heading` columns
2. **Complete MAGAT segmentation**: `runs`, `head_swings`, `reorientations` lists
3. **Valid indices**: All run/reorientation indices within trajectory bounds
4. **Logical consistency**: Turns must have associated runs, time must progress

**No data is approximated or guessed** - if anything is missing or invalid, an error is raised immediately.

