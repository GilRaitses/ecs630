# Scripts Directory

Supporting Python scripts for the term project.

## Scripts

### `inspect_h5_structure.py`
Inspect the structure of H5 files to understand data organization.

**Usage**:
```bash
python3 scripts/inspect_h5_structure.py <h5_file_path>
```

**Example**:
```bash
python3 scripts/inspect_h5_structure.py /Users/gilraitses/mechanosensation/h5tests/GMR61_202509051201_tier1\ 1.h5
```

### `engineer_dataset_from_h5.py`
Extract trajectory and stimulus data from H5 files and create feature matrices for modeling.

**Usage**:
```bash
python3 scripts/engineer_dataset_from_h5.py \
    --h5-dir /Users/gilraitses/mechanosensation/h5tests \
    --output-dir data/engineered \
    --experiment-id GMR61_202509051201
```

**Outputs**:
- `{experiment_id}_events.csv`: Event records with time bins and turn indicators
- `{experiment_id}_trajectories.csv`: Full trajectory data with aligned stimulus
- `{experiment_id}_summary.json`: Summary statistics

**Features Extracted**:
- Trajectory features: position (x, y), speed, heading, acceleration
- Behavioral events: turn indicators (based on heading changes > 30°)
- Stimulus alignment: LED intensity, stimulus on/off, time since stimulus onset
- Time binning: 50ms bins for discrete-time hazard modeling

### `fit_hazard_model.py`
Fits event-hazard GLM models to trajectory data.

**Usage**:
```bash
python3 scripts/fit_hazard_model.py \
    --trajectory-dir data/engineered \
    --stimulus-file data/engineered/GMR61_202509051201_trajectories.csv \
    --output-dir output/fitted_models \
    --event-type turn
```

**Outputs**:
- Fitted model object (pickle)
- Coefficient table (CSV)
- Kernel visualization (PNG)
- Validation metrics (JSON)

### `simulate_trajectories.py`
Simulates larval trajectories using fitted hazard models.

**Usage**:
```bash
python3 scripts/simulate_trajectories.py \
    --model-file output/fitted_models/turn_model.pkl \
    --stimulus-schedule config/doe_table.csv \
    --n-replications 30 \
    --output-dir output/simulation_results
```

**Outputs**:
- Simulated trajectories (CSV per replication)
- Event logs (CSV)
- KPI summaries (CSV)

### `run_doe.py`
Executes full factorial DOE across all conditions.

**Usage**:
```bash
python3 scripts/run_doe.py \
    --doe-table config/doe_table.csv \
    --model-config config/model_config.json \
    --output-dir output/doe_results
```

**Outputs**:
- Combined results CSV
- Individual replication files
- Calls `export_arena_format.py` to generate Arena CSVs

### `export_arena_format.py`
Converts simulation results to Arena-style CSV format.

**Usage**:
```bash
python3 scripts/export_arena_format.py \
    --results output/simulation_results/all_results.csv \
    --output-dir output/arena_csvs
```

**Outputs**:
- `AcrossReplicationsSummary.csv`
- `ContinuousTimeStatsByRep.csv`
- `DiscreteTimeStatsByRep.csv`

## Data Pipeline

### Step 1: Extract from H5 Files
```bash
python3 scripts/engineer_dataset_from_h5.py \
    --h5-dir /Users/gilraitses/mechanosensation/h5tests \
    --output-dir data/engineered
```

### Step 2: Fit Models
```bash
python3 scripts/fit_hazard_model.py \
    --trajectory-dir data/engineered \
    --output-dir output/fitted_models
```

### Step 3: Run DOE Simulations
```bash
python3 scripts/run_doe.py \
    --doe-table config/doe_table.csv \
    --output-dir output/doe_results
```

### Step 4: Export Results
```bash
python3 scripts/export_arena_format.py \
    --results output/doe_results/all_results.csv \
    --output-dir output/arena_csvs
```

## Dependencies

- `pandas`: Data manipulation
- `numpy`: Numerical computing
- `scipy`: Statistical functions
- `h5py`: HDF5 file reading
- `scikit-learn`: GLM fitting (if used)
- `statsmodels`: Alternative GLM option

Install with:
```bash
pip install pandas numpy scipy h5py scikit-learn statsmodels
```

## H5 File Structure

The scripts expect H5 files with the following structure:

```
/tracks/
  /track_1/
    /positions (N×2 or N×3 array: x, y, [frame])
    /attrs (metadata)
/stimulus/ or /led_data/ or /experiment/
  /led1Val or /intensity (LED values)
  /elapsedTime or /time (timestamps)
/metadata/
  /attrs (experiment metadata)
```

If your H5 files have a different structure, modify `engineer_dataset_from_h5.py` accordingly.
