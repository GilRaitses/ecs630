# Data Sources for Term Project

## Primary Data: H5 Files

The primary data source for this project is HDF5 files exported from MAGAT experiments.

### Location
```
/Users/gilraitses/mechanosensation/h5tests/
```

### Available Files

1. **`GMR61_202509051201_tier1 1.h5`** (16 MB)
   - Complete tier1 export
   - Contains: tracks, positions, LED stimulus data, metadata
   - **Recommended for initial modeling**

2. **`GMR61_tier2_complete.h5`** (83 MB)
   - Full tier2 export with contour data
   - Contains: tracks, positions, contours, LED data
   - **Use if contour-based features needed**

3. **`GMR61_202509051201_tier3.h5`** (701 KB)
   - Tier3 export with FID (Foreground Image Data)
   - Contains: minimal structure with FID references
   - **Use if image analysis needed**

4. **`GMR61_202509051201.h5`** (1.1 MB)
   - Basic export
   - Contains: tracks and basic metadata

5. **`GMR61_tier2 1.h5`** (1.1 MB)
   - Tier2 summary export

6. **`GMR61_tier2_final.h5`** (1.1 MB)
   - Final tier2 export

### H5 File Structure

Expected structure (based on `export_experiment_to_hdf5.py`):

```
/metadata/
  /attrs: num_tracks, num_frames, experiment_id, etc.
/tracks/
  /track_1/
    /positions: N×2 array (x, y coordinates)
    /attrs: start_frame, end_frame, etc.
    /contours/ (if tier2+)
      /frame_00001: variable-length contour array
/led_data/ or /stimulus/
  /led1Val or /intensity: LED values per frame
  /elapsedTime or /time: timestamps
```

### Data Extraction

Use `scripts/engineer_dataset_from_h5.py` to extract modeling-ready data:

```bash
python3 scripts/engineer_dataset_from_h5.py \
    --h5-dir /Users/gilraitses/mechanosensation/h5tests \
    --output-dir data/engineered \
    --experiment-id GMR61_202509051201
```

This will create:
- `{experiment_id}_events.csv`: Event records (50ms bins) with turn indicators
- `{experiment_id}_trajectories.csv`: Full trajectory data aligned with stimulus
- `{experiment_id}_summary.json`: Summary statistics

## Backup Data Sources

If H5 files are unavailable or incomplete:

### CSV Trajectory Data
```
/Users/gilraitses/mechanosensation/output/spatial_analysis/
  - runs.csv: Run segments with directions, durations, path lengths
  - reorientations.csv: Turn events with angles, head swings
  - head_swings.csv: Individual head swing events
```

### CSV Stimulus Data
```
/Users/gilraitses/mechanosensation/led_stimulus_data_*.csv
  - Frame-level LED intensity values
  - Stimulus onset indicators
  - Pulse timing information
```

## Data Inspection

To inspect H5 file structure:

```bash
python3 scripts/inspect_h5_structure.py \
    /Users/gilraitses/mechanosensation/h5tests/GMR61_202509051201_tier1\ 1.h5
```

## Data Requirements for Modeling

### Minimum Requirements
- Trajectory positions (x, y) per frame
- Frame timestamps or frame rate
- LED stimulus values (intensity or on/off)

### Optional but Helpful
- Heading/orientation data
- Speed data
- Contour data (for shape analysis)
- Reorientation event markers

### Expected Data Volume
- **Per experiment**: ~20-30 tracks × ~20 minutes × 20 fps = ~24,000 frames
- **Per track**: ~800 frames average (40 seconds at 20 fps)
- **Event rate**: ~1-5 turns per minute per larva

## Notes

- H5 files are MATLAB v7.3 format, readable with `h5py` in Python
- Frame rate is typically 20 fps
- Time units: seconds (derived from frame numbers ÷ frame_rate)
- Position units: pixels (may need calibration to mm/cm)
- Stimulus cycle: Typically 20 seconds (10s baseline, 10s stimulus)

