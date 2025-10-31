# DOE Analysis Pipeline

## After Running DOE Simulation

Once the DOE simulation completes (using `run_doe.py`), run the analysis pipeline:

### Quick Analysis

```bash
python3 scripts/analyze_doe_results.py \
    --results-dir output/simulation_results \
    --output-dir output/analysis
```

This single command will:
1. ✅ Export results to Arena format (`AcrossReplicationsSummary.csv`)
2. ✅ Analyze main effects and interactions (ANOVA)
3. ✅ Generate visualization plots
4. ✅ Create report tables for Quarto

### Step-by-Step Analysis

If you prefer to run steps individually:

#### Step 1: Export to Arena Format
```bash
python3 scripts/export_arena_format.py \
    --results output/simulation_results/all_results.csv \
    --output-dir output/arena_csvs
```

**Outputs:**
- `AcrossReplicationsSummary.csv` - Mean, CI, min, max for each KPI by condition
- `ContinuousTimeStatsByRep.csv` - Time-persistent statistics per replication
- `DiscreteTimeStatsByRep.csv` - Event-based statistics per replication

#### Step 2: Analyze Main Effects
```bash
python3 scripts/analyze_main_effects.py \
    --summary output/arena_csvs/AcrossReplicationsSummary.csv \
    --output-dir output/figures/doe_analysis
```

**Outputs:**
- Main effects plots (one for each factor)
- Interaction plots (one for each factor pair)
- ANOVA results JSON

#### Step 3: Generate Report Tables
```bash
python3 scripts/generate_report_tables.py \
    --doe-table config/doe_table.csv \
    --summary output/arena_csvs/AcrossReplicationsSummary.csv \
    --output-dir output/report_tables
```

**Outputs:**
- `doe_table_report.csv` - Formatted DOE table for report
- `summary_table_top10.csv` - Top 10 conditions with CIs
- `*_coefficients_report.csv` - Model coefficients tables

## Output Structure

```
output/
├── simulation_results/
│   ├── all_results.csv          # All simulation results
│   ├── checkpoint.csv            # Checkpoint file for resume
│   └── summary_statistics.json   # Basic summary (if generated)
│
├── analysis/                    # Complete analysis results
│   ├── arena_csvs/
│   │   └── AcrossReplicationsSummary.csv
│   ├── figures/
│   │   └── doe_analysis/        # Main effects & interaction plots
│   ├── report_tables/           # Tables for Quarto report
│   └── main_effects_analysis.json
│
└── fitted_models/                # Hazard models
    └── *_model.pkl
```

## Integration with Quarto Report

After analysis, update `TermProject_Report.qmd` to include:
- New DOE results from `output/analysis/arena_csvs/`
- New figures from `output/analysis/figures/doe_analysis/`
- Updated tables from `output/analysis/report_tables/`

Then re-render:
```bash
quarto render TermProject_Report.qmd
```



