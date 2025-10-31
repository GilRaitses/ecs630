# Quick Start Guide

Get started with the term project in 4 steps.

## Prerequisites

Install Python dependencies:
```bash
pip install pandas numpy scipy h5py scikit-learn statsmodels
```

## Step 1: Inspect H5 Files

First, check what's in your H5 files:

```bash
cd /Users/gilraitses/ecs630/labs/termprojectproposal
python3 scripts/inspect_h5_structure.py \
    /Users/gilraitses/mechanosensation/h5tests/GMR61_202509051201_tier1\ 1.h5
```

This will show you the structure of the H5 file so you know what data is available.

## Step 2: Extract Data from H5 Files

Extract trajectory and stimulus data into modeling-ready CSV format:

```bash
python3 scripts/engineer_dataset_from_h5.py \
    --h5-dir /Users/gilraitses/mechanosensation/h5tests \
    --output-dir data/engineered \
    --experiment-id GMR61_202509051201
```

This creates:
- `data/engineered/GMR61_202509051201_events.csv`: Event records (50ms bins)
- `data/engineered/GMR61_202509051201_trajectories.csv`: Full trajectories aligned with stimulus
- `data/engineered/GMR61_202509051201_summary.json`: Summary statistics

## Step 3: Fit Hazard Model

Fit the event-hazard GLM model to your data:

```bash
python3 scripts/fit_hazard_model.py \
    --trajectory-dir data/engineered \
    --stimulus-file data/engineered/GMR61_202509051201_trajectories.csv \
    --output-dir output/fitted_models \
    --event-type turn
```

This will:
- Load the engineered data
- Extract features (stimulus history, speed, heading, etc.)
- Fit GLM with temporal kernel
- Validate the model (KS test, PSTH comparison)
- Save fitted model and coefficients

## Step 4: Run DOE Simulation

Run the full factorial design of experiments:

```bash
python3 scripts/run_doe.py \
    --doe-table config/doe_table.csv \
    --model-config config/model_config.json \
    --output-dir output/doe_results
```

This simulates 27 conditions × 30 replications = 810 trajectories and computes KPIs.

## Step 5: Export Arena Format

Convert results to Arena-style CSV format:

```bash
python3 scripts/export_arena_format.py \
    --results output/doe_results/all_results.csv \
    --output-dir output/arena_csvs
```

This creates:
- `output/arena_csvs/AcrossReplicationsSummary.csv`
- `output/arena_csvs/ContinuousTimeStatsByRep.csv`
- `output/arena_csvs/DiscreteTimeStatsByRep.csv`

## Troubleshooting

### H5 file not found
- Check path: `/Users/gilraitses/mechanosensation/h5tests/`
- List files: `ls /Users/gilraitses/mechanosensation/h5tests/*.h5`

### h5py not installed
```bash
pip install h5py
```

### H5 file structure doesn't match
- Run `inspect_h5_structure.py` to see actual structure
- Modify `engineer_dataset_from_h5.py` to match your H5 format

### No tracks found in H5 file
- Check if H5 file has `/tracks/` group
- Verify track keys (may be `track_1`, `track_2`, etc. or different naming)

### No stimulus data found
- Check if H5 file has `/led_data/`, `/stimulus/`, or `/experiment/` group
- May need to use backup CSV stimulus files if H5 doesn't contain LED data

## Next Steps

1. Review extracted data: Check `data/engineered/*_summary.json` for statistics
2. Explore trajectories: Load `*_trajectories.csv` in Python/R to visualize
3. Fit models: Experiment with different kernel designs in `fit_hazard_model.py`
4. Validate: Compare simulated vs. empirical KPIs

For more details, see:
- `README.md`: Project overview
- `DATA_SOURCES.md`: Data file descriptions
- `scripts/README.md`: Script documentation

