# Progress Window Guide

## Quick Progress Check

Simply run:
```bash
python3 scripts/show_progress.py
```

This shows:
- ✅ Current progress (conditions completed)
- ✅ Visual progress bar
- ✅ Time elapsed and estimated time remaining
- ✅ Completion time estimate
- ✅ Current status

## Live Monitoring (Auto-Refresh)

For a live updating display:
```bash
python3 scripts/show_progress.py --watch
```

This will:
- Clear and refresh the screen every 5 seconds
- Show real-time updates
- Keep the display clean and current

Press **Ctrl+C** to stop monitoring (simulation continues running).

## Custom Refresh Rate

```bash
# Refresh every 2 seconds (more frequent)
python3 scripts/show_progress.py --watch --refresh 2

# Refresh every 10 seconds (less frequent)
python3 scripts/show_progress.py --watch --refresh 10
```

## What You'll See

```
======================================================================
DOE SIMULATION PROGRESS
======================================================================

Conditions: 3/45 (6.7%)
[████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 6.7%

Time elapsed: 0.3min
Time remaining: ~13.7min
Estimated completion: 20:14:32

Latest file: condition_3_results.csv
Status: ✓ Running (2.1s ago)

Current condition: [████████████████████████░░░░░░] 13.3%
Replication: 4/30

Rate: 10.00 conditions/minute

✓ Process is running

======================================================================
Auto-refreshing every 5 seconds... (Ctrl+C to stop)
```

## Alternative Quick Checks

### Count completed conditions
```bash
ls output/doe_results/condition_*_results.csv | wc -l
```

### Watch log file
```bash
tail -f output/doe_full_run.log
```

### Check process status
```bash
ps aux | grep run_doe.py | grep -v grep
```



