# Full DOE Run Status

## Configuration
- **Conditions**: 45 total
- **Replications**: 30 per condition
- **Total Simulations**: 1,350
- **Max Time**: 300 seconds per simulation

## Expected Timeline
- **Per replication**: ~10 seconds
- **Per condition**: ~5 minutes (30 reps × 10s)
- **Total runtime**: ~3-4 hours

## Monitoring Commands

### Check Progress
```bash
# Count completed conditions
ls output/doe_results/condition_*_results.csv | wc -l

# Watch log in real-time
tail -f output/doe_full_run.log

# Use monitor script
python3 scripts/monitor_doe.py
```

### Check Process Status
```bash
ps aux | grep run_doe.py | grep -v grep
```

## After Completion

Once all 45 conditions are complete, run the analysis:

```bash
python3 scripts/analyze_doe_results.py \
    --results-dir output/doe_results \
    --output-dir output/analysis
```

This will generate:
- Complete `AcrossReplicationsSummary.csv` with 45 conditions
- Main effects and interaction plots
- Report tables for Quarto
- Statistical analysis (ANOVA, main effects)

## Current Status

Started: $(date)
Expected completion: ~$(date -v+4H 2>/dev/null || date -d '+4 hours')



