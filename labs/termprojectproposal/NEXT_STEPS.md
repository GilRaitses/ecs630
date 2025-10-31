# Next Steps for Term Project

## ✅ Completed

1. **Proposal Created**: `TermProject_Proposal.qmd` with full methodology
2. **H5 Files Inspected**: Actual structure documented
3. **Data Extraction Working**: Successfully extracted from H5 → CSV

**Results from test extraction:**
- 12 tracks processed
- 273,157 event records (50ms bins)
- Turn detection working (track_11 had 388 turns!)
- Stimulus alignment successful (81 onsets detected)

## 🎯 Immediate Next Steps

### Step 1: Validate Extracted Data (5 min)
```bash
cd /Users/gilraitses/ecs630/labs/termprojectproposal
source venv/bin/activate

# Quick check of data quality
python3 -c "
import pandas as pd
events = pd.read_csv('data/engineered/GMR61_202509051201_events.csv')
print('Event records:', len(events))
print('Turn rate:', events['is_turn'].mean() * 60 / 0.05, 'turns/min')
print('Stimulus coverage:', events['stimulus_on'].mean())
"
```

### Step 2: Explore Data Visually (15 min)
Create a quick visualization script to understand:
- Turn rate over time
- Relationship between stimulus and turns
- Speed distributions
- Heading distributions

**File to create**: `scripts/explore_data.py`

### Step 3: Fit Baseline Model (30 min)
Implement and fit the null model (constant hazard):
```bash
python3 scripts/fit_hazard_model.py \
    --trajectory-dir data/engineered \
    --stimulus-file data/engineered/GMR61_202509051201_trajectories.csv \
    --output-dir output/fitted_models \
    --event-type turn \
    --baseline-only
```

This will:
- Load event records
- Fit constant hazard (no stimulus effects)
- Compute baseline turn rate
- Validate model (KS test, etc.)

### Step 4: Implement Temporal Kernel (1 hour)
Add the stimulus-locked kernel to `fit_hazard_model.py`:
- Implement raised cosine basis functions
- Create stimulus history features (convolution)
- Fit GLM with kernel features

### Step 5: Fit Full Model (30 min)
Fit the stimulus-locked hazard model:
```bash
python3 scripts/fit_hazard_model.py \
    --trajectory-dir data/engineered \
    --stimulus-file data/engineered/GMR61_202509051201_trajectories.csv \
    --output-dir output/fitted_models \
    --event-type turn
```

### Step 6: Implement Simulation Engine (2 hours)
Complete `simulate_trajectories.py`:
- Load fitted model
- Implement event generation (thinning algorithm)
- Run simulation for single condition
- Validate against empirical data

### Step 7: Run DOE (1 hour)
Execute full factorial design:
```bash
python3 scripts/run_doe.py \
    --doe-table config/doe_table.csv \
    --model-config config/model_config.json \
    --output-dir output/doe_results
```

### Step 8: Generate Arena CSVs (15 min)
Export results:
```bash
python3 scripts/export_arena_format.py \
    --results output/doe_results/all_results.csv \
    --output-dir output/arena_csvs
```

### Step 9: Write Report (ongoing)
Update `TermProject_Report.qmd` with:
- Actual data description
- Model fitting results
- DOE results and CI analysis
- Figures and tables

## 📋 Priority Order

**This Week:**
1. ✅ Data extraction (DONE)
2. Validate derived data quality
3. Fit baseline/null model
4. Implement temporal kernel

**Next Week:**
5. Fit full stimulus-locked model
6. Implement simulation engine
7. Run DOE for one condition (test)

**Following Week:**
8. Run full DOE (all 27 conditions)
9. Generate Arena CSVs
10. Write report with results

## 🔧 Files to Complete

**High Priority:**
- `scripts/fit_hazard_model.py` - Need to implement actual GLM fitting (currently placeholder)
- `scripts/simulate_trajectories.py` - Create from scratch
- `scripts/run_doe.py` - Create from scratch

**Medium Priority:**
- `scripts/explore_data.py` - Quick visualization script
- Update `TermProject_Proposal.qmd` → `TermProject_Report.qmd` with actual results

**Low Priority:**
- Add more error handling
- Add unit tests
- Documentation updates

## 📊 Current Data Stats

From extraction:
- **Tracks**: 12 larvae
- **Total frames**: ~240K frames (varies by track)
- **Event records**: 273,157 bins (50ms each = ~227 minutes total)
- **Turn events**: Hundreds detected per track
- **Stimulus**: 81 onsets detected (should be ~40 cycles × 2 = 80, close!)

## 🐛 Known Issues to Address

1. **Turn detection**: Track 11 has 388 turns - may be too sensitive or that larva is hyperactive
2. **Stimulus onsets**: Detected 81, expected ~80 - minor difference, check detection logic
3. **Frame rate**: Using 10 fps from metadata - verify this matches actual experiment

## 💡 Tips

- If turn detection is too sensitive, adjust threshold in `extract_trajectory_features()` (currently 30°)
- Use pre-computed `derived/direction` instead of computing from positions for consistency
- Consider using `derived/curvature` as additional feature for turn prediction
- Test simulation engine on a single track first before running full DOE

