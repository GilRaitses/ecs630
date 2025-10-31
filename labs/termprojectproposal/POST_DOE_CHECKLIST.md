# Post-DOE Analysis Checklist

## After DOE Simulation Completes

### 1. Verify DOE Results
```bash
# Check that all_results.csv exists
ls -lh output/simulation_results/all_results.csv

# Quick check of results
python3 -c "
import pandas as pd
df = pd.read_csv('output/simulation_results/all_results.csv')
print(f'Total rows: {len(df)}')
print(f'Conditions: {df[\"condition_id\"].nunique()}')
print(f'Replications per condition: {len(df) / df[\"condition_id\"].nunique():.0f}')
print(f'\\nSample KPIs:')
print(df[[c for c in df.columns if 'rate' in c.lower()]].head())
"
```

### 2. Run Complete Analysis Pipeline
```bash
python3 scripts/analyze_doe_results.py \
    --results-dir output/simulation_results \
    --output-dir output/analysis
```

**This will generate:**
- ✅ `output/analysis/arena_csvs/AcrossReplicationsSummary.csv`
- ✅ `output/analysis/figures/doe_analysis/*.png` (main effects & interactions)
- ✅ `output/analysis/report_tables/*.csv` (formatted tables for report)
- ✅ `output/analysis/main_effects_analysis.json` (ANOVA results)

### 3. Review Results
- Check `AcrossReplicationsSummary.csv` for biologically plausible values
- Review main effects plots to see which factors matter most
- Check interaction plots for factor interactions

### 4. Update Quarto Report
- Update figures paths in `TermProject_Report.qmd`
- Update table paths to new analysis results
- Re-render: `quarto render TermProject_Report.qmd`

## Quick Reference

**DOE Configuration:**
- 45 conditions total
- 3 intensities (PWM 250, 500, 1000)
- 5 pulse durations (10s, 15s, 20s, 25s, 30s)
- 3 inter-pulse intervals (20s, 40s, 60s)
- Each condition: 5 replications (test) or 30 replications (full)

**Expected Output Files:**
```
output/simulation_results/
├── all_results.csv              # All simulation results
├── checkpoint.csv               # Checkpoint file for resume
└── summary_statistics.json      # Basic summary (if generated)

output/analysis/
├── arena_csvs/
│   └── AcrossReplicationsSummary.csv
├── figures/doe_analysis/
│   ├── main_effects_*.png
│   └── interactions_*.png
└── report_tables/
    ├── doe_table_report.csv
    └── summary_table_top10.csv
```



