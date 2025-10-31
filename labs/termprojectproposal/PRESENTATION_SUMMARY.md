# Simulation Project One-Page Summary

## Behavioral Simulation of Drosophila Larvae: Data Engineering, DOE Design, and Event-Hazard Modeling

**Author:** Gil Raitses  
**Course:** ECS630 Simulation Modeling  
**Date:** October 2024

---

### Objective

Develop a stimulus-response modeling framework for Drosophila larval behavior using event-hazard methods from survival analysis. The model predicts behavioral events (turns, stops, reversals) as functions of LED stimulus intensity, temporal history, and contextual features, enabling simulation-based prediction of behavioral metrics.

---

### Data Engineering Pipeline

**Input:** H5 files containing larval trajectory data (position, speed, angle at 20 Hz) with synchronized LED stimulus timing.

**Process:**
1. Extract spatial features (speed, curvature, body bend angle)
2. Detect behavioral events using MAGAT segmentation
3. Generate Klein run table (18-column standardized structure)
4. Extract stimulus-locked features using temporal kernel bases
5. Learn detection parameters from empirical distributions

**Output:** Engineered dataset with ~100+ features per trajectory, ready for hazard model fitting.

---

### Design of Experiments (DOE)

**Factors:**
- **Intensity**: 3 levels (PWM 250, 500, 1000)
- **Pulse Duration**: 5 levels (10s, 15s, 20s, 25s, 30s)
- **Inter-Pulse Interval**: 3 levels (5s, 10s, 20s)

**Design:** 45 conditions × 30 replications = 1,350 total simulations (~6 hours runtime)

**Rationale:** Covers biologically relevant parameter space, enables main effects and interaction analysis, provides statistical power for validation.

---

### Simulation Methodology

**Event-Hazard Models:**
- Fitted generalized linear models (Poisson/logistic) for reorientations, pauses, and heading reversals
- Features include temporal kernel bases (raised cosine), stimulus intensity, speed, orientation, wall proximity
- Cross-validation: leave-one-larva-out
- Regularization: L2

**Simulation Process:**
1. Load fitted hazard models and learned event parameters
2. For each DOE condition, set stimulus schedule
3. Time-step simulation: calculate hazard rates, sample event times, update trajectory state
4. Compute 7 KPIs per replication, aggregate across 30 replications with confidence intervals

---

### Key Performance Indicators (KPIs)

1. **Turn Rate**: Reorientations per minute
2. **Latency**: Time to first turn after stimulus onset (within integration window)
3. **Stop Fraction**: Proportion of time spent paused
4. **Pause Rate**: Pauses per minute (speed-based detection)
5. **Reversal Rate**: Heading reversals per minute
6. **Tortuosity**: Path efficiency (displacement/path_length)
7. **Mean Spine Curve Energy**: Body bend energy

All KPIs validated against empirical data ranges to ensure biological plausibility.

---

### Validation & Results

**Pre-DOE Validation:**
- Single condition test: 30 replications
- All KPIs within expected ranges:
  - Turn rate: 0.4-11 turns/min
  - Latency: 0.5-5 seconds
  - Pause rate: 1-10 pauses/min
  - Stop fraction: 0-1

**Parameter Learning:**
- Pause detection: learned speed threshold (0.003 mm/s) and min duration (0.2s) to match empirical rate (~4 pauses/min)
- Reversal detection: learned angle threshold from heading change distribution
- MAGAT segmentation: learned curvature and body bend thresholds from empirical data

**Status:** Full DOE simulation running (45 conditions × 30 replications). Expected completion: ~4-5 hours.

---

### Analysis Pipeline (Post-Simulation)

1. **Export Results**: Arena-format CSV files (AcrossReplicationsSummary.csv)
2. **Main Effects Analysis**: ANOVA for each KPI, factor importance ranking
3. **Interaction Analysis**: Factor interaction plots and statistical tests
4. **Visualization**: Main effects plots, distribution comparisons, confidence intervals

**Output Format:** Arena-style summary statistics with 95% confidence intervals, enabling statistical comparison to empirical observations.

---

### Technical Implementation

**Languages/Tools:** Python 3 (data processing, simulation), R/Quarto (analysis, reporting), Pandas/NumPy, Scikit-learn

**Key Scripts:**
- `engineer_dataset_from_h5.py`: Data engineering and feature extraction
- `fit_hazard_models.py`: Hazard model fitting with cross-validation
- `simulate_trajectories.py`: DOE simulation with checkpointing and error recovery
- `analyze_doe_results.py`: Post-simulation analysis pipeline

**Output:** Arena-style CSV files, summary statistics with CIs, formatted tables for reports, visualization plots (PNG/PDF).

---

### Significance

This project demonstrates how discrete-event simulation principles apply to behavioral systems where events (behavioral transitions) occur stochastically as functions of external stimuli and internal state. Results are reported using Arena-style summary statistics including across-replications summaries, confidence intervals, and performance metrics analogous to manufacturing system analysis, bridging simulation modeling methods with biological data analysis.

