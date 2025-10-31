# Spine Curve Energy Tracking

## Overview
Spine curve energy is now tracked as a feature in the hazard model to capture larval body bending dynamics.

## Definition
**Spine Curve Energy** = `curvature²` (bending energy)

This measures the "energy" stored in body bending, proportional to the square of curvature. Higher values indicate more pronounced body curvature/bending.

## Computation

### Data Extraction (`engineer_dataset_from_h5.py`)
1. **Load curvature** from H5 `derived_quantities/curv` or `curvature`
2. **Clip extreme values**: `curvature_clipped = np.clip(curvature, -1e6, 1e6)`
   - Handles outliers from H5 data (observed range: ~-850k to 2.6M)
3. **Compute energy**: `spine_curve_energy = curvature_clipped²`
4. **Store in DataFrame**: Added as `spine_curve_energy` column

### Feature Engineering (`fit_hazard_model.py`)
1. **Log-transform**: `spine_energy_log = log(spine_curve_energy + ε)`
   - Handles large dynamic range (energy spans many orders of magnitude)
   - Prevents log(0) with small epsilon (1e-6)
2. **Normalize**: `spine_energy_normalized = (log - mean) / std`
   - Standardized for GLM fitting
3. **Feature name**: `spine_curve_energy_normalized`

## Integration Points

### 1. Trajectory Extraction
- **File**: `scripts/engineer_dataset_from_h5.py`
- **Function**: `extract_trajectory_features()`
- **Output**: DataFrame with `spine_curve_energy` column

### 2. Event Record Creation
- **File**: `scripts/engineer_dataset_from_h5.py`
- **Function**: `create_event_records()`
- **Aggregation**: Mean spine curve energy per time bin (50ms)

### 3. Model Feature Matrix
- **File**: `scripts/fit_hazard_model.py`
- **Function**: `prepare_feature_matrix()`
- **Feature**: `spine_curve_energy_normalized` (log-transformed, normalized)

## Usage in Model

The spine curve energy feature is included in the full GLM model:
```
hazard_rate = intercept + kernel_features + speed + heading_sin + heading_cos + spine_curve_energy
```

## Interpretation

- **High spine curve energy**: Larva is more curved/bent (e.g., during turns, C-bends)
- **Low spine curve energy**: Larva is straighter (e.g., during runs)
- **Stimulus effect**: Changes in spine curve energy may indicate stimulus-induced body shape changes

## Re-running Data Engineering

To include spine curve energy in your datasets, re-run:
```bash
python3 scripts/engineer_dataset_from_h5.py \
    --h5-dir /Users/gilraitses/mechanosensation/h5tests \
    --output-dir data/engineered_tier2 \
    --experiment-id GMR61_tier2
```

This will create new event/trajectory CSVs with `spine_curve_energy` included.

