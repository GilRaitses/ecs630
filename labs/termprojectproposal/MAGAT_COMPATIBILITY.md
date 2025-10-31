# MAGAT Compatibility Notes

## MAGAT Reference

The MAGAT (Maggot Analysis Tool) codebase is available at:
- Repository: https://github.com/GilRaitses/magniphyq
- Example Scripts: `codebase/Matlab-Track-Analysis-SkanataLab/MAGATAnalyzer Example Scripts/`

## MAGAT Reorientation Detection Algorithm

From `@MaggotTrack/segmentTrack.m`:

1. **Detect Runs**: Periods of forward movement where:
   - Speed >= `start_speed_cut`
   - Head aligned (`vel_dp >= aligned_dp`)
   - Low curvature (`abs(curv) <= curv_cut`)
   - Head not swinging (`abs(spineTheta) <= theta_cut`)

2. **Detect Head Swings**: Periods between runs where:
   - Head swinging (`abs(spineTheta) > headswing_start`)
   - Not in a run
   - Between first and last run (with buffer)

3. **Group into Reorientations**: 
   - A reorientation is the period **BETWEEN runs**
   - Whether or not it contains head swings
   - Reorientations are gaps between runs

## Our Implementation

Since tier2_complete H5 files don't have pre-segmented runs, we use a simplified detection:

- **Detection Method**: Large angular velocity (`deltatheta/dt > 2.3 rad/s`) AND moving (`speed > 0.0003`)
- **Event Marking**: We mark reorientation **start events** (equivalent to `track.reorientation(i).startInd`)
- **Data Source**: Uses `deltatheta` from `derived_quantities/deltatheta` in tier2_complete H5

## Full MAGAT Workflow (MATLAB)

To use full MAGAT segmentation with tier2_complete H5 on MATLAB:

```matlab
% Load experiment set from H5 (if H5 loader exists)
eset = ExperimentSet.fromH5Files('path/to/tier2_complete.h5');

% Segment tracks into runs and reorientations
eset.executeTrackFunction('segmentTrack');

% Access reorientations
for track = eset.expt(1).track
    for reo = track.reorientation
        startInd = reo.startInd;  % Frame index where reorientation starts
        endInd = reo.endInd;      % Frame index where reorientation ends
        numHS = reo.numHS;        % Number of head swings in this reorientation
    end
end
```

## Validation

Our Python implementation produces reorientation start events that approximate MAGAT's `startInd` values. For full validation, compare:

1. Load tier2_complete H5 in MAGAT and run `segmentTrack`
2. Extract `[track.reorientation.startInd]` for each track
3. Compare with our `is_reorientation` flags (where True indicates reorientation start)

## Files Reference

- MAGAT Segmentation: `magniphyq/codebase/Matlab-Track-Analysis-SkanataLab/@MaggotTrack/segmentTrack.m`
- Reorientation Class: `magniphyq/codebase/Matlab-Track-Analysis-SkanataLab/@MaggotReorientation/`
- Example Scripts: `magniphyq/codebase/Matlab-Track-Analysis-SkanataLab/MAGATAnalyzer Example Scripts/MAGAT_ANALYZER_DEMO.m`



