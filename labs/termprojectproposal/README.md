# Term Project Proposal: Stimulus-Driven Behavioral Modeling

This folder contains all materials for the ECS630 term project proposal on modeling Drosophila larval behavior using stimulus-response event-hazard methods.

## Folder Structure

```
termprojectproposal/
├── TermProject_Proposal.qmd          # Main proposal document (Quarto)
├── README.md                          # This file
├── scripts/                           # Supporting analysis scripts
│   ├── fit_hazard_model.py           # GLM hazard model fitting
│   ├── simulate_trajectories.py       # Trajectory simulation engine
│   ├── run_doe.py                     # DOE execution script
│   └── export_arena_format.py         # Convert results to Arena CSV format
├── config/                            # Configuration files
│   ├── doe_table.csv                  # Design of experiments table (27 conditions)
│   └── model_config.json              # Model hyperparameters, CI targets
├── data/                              # Data files (symbolic links or copies)
│   └── [links to mechanosensation data]
└── output/                            # Generated outputs
    ├── fitted_models/                 # Saved model objects
    ├── simulation_results/            # DOE simulation outputs
    └── arena_csvs/                    # Arena-style summary CSVs
```

## Quick Start

1. **Review Proposal**: Open `TermProject_Proposal.qmd` in RStudio/Quarto
2. **Install Dependencies**: 
   ```bash
   pip install -r requirements.txt  # (if created)
   ```
3. **Run Example Analysis**: 
   ```bash
   python scripts/fit_hazard_model.py --help
   ```
4. **Generate Report**: 
   ```bash
   quarto render TermProject_Proposal.qmd
   ```

## Key Components

### Proposal Document
- **File**: `TermProject_Proposal.qmd`
- **Format**: Quarto markdown → PDF
- **Style**: Matches Lab01/Lab02 formatting (Avenir Next fonts, technical terminology)

### Design of Experiments
- **File**: `config/doe_table.csv`
- **Design**: Full factorial 3³ = 27 conditions
- **Factors**: Stimulus Intensity, Pulse Duration, Inter-Pulse Interval
- **Replications**: 30 per condition (810 total simulations)

### Model Implementation
- **Core Method**: Event-hazard GLM with temporal kernels
- **Events**: Turn starts, stop starts, reversal starts
- **Features**: Stimulus history (kernel convolution), speed, orientation, wall distance

### Output Format
- **Arena-Style CSVs**: `AcrossReplicationsSummary.csv`, `ContinuousTimeStatsByRep.csv`, `DiscreteTimeStatsByRep.csv`
- **Matching Lab Format**: Compatible with existing lab analysis workflows

## Data Requirements

The project uses H5 files from `/Users/gilraitses/mechanosensation/h5tests/`:

- **Primary**: H5 files (e.g., `GMR61_202509051201_tier1 1.h5`) containing trajectory and stimulus data
- **Backup**: CSV files in `output/spatial_analysis/` if H5 files unavailable
- **Experimental Metadata**: Experiment IDs embedded in H5 metadata

See `DATA_SOURCES.md` for detailed information about available H5 files.

Data paths are configured in `config/model_config.json`.

## Timeline

- **Week 1**: Data preparation
- **Week 2**: Model development
- **Week 3**: Model fitting
- **Week 4**: Simulation engine
- **Week 5**: DOE execution
- **Week 6**: Report writing

## Contact

Questions about the proposal? See the proposal document or contact Gil Raitses.

