# DOE Monitoring Guide

## What is Monitoring?

Monitoring means **automatically checking** the status of a running process and showing you updates in real-time. Instead of manually checking if something is done, a monitoring script watches it for you and alerts you when it's complete.

## How to Use the Monitor

### Basic Usage
```bash
python3 scripts/monitor_doe.py
```

This will:
- Check progress every 5 seconds
- Show current condition count (e.g., "23/45")
- Display percentage complete
- Show time remaining estimate
- Alert you when simulation completes

### Custom Refresh Rate
```bash
# Check every 2 seconds (more frequent updates)
python3 scripts/monitor_doe.py --refresh 2

# Check every 10 seconds (less frequent, quieter)
python3 scripts/monitor_doe.py --refresh 10
```

### Quiet Mode
```bash
# Only print when progress actually changes
python3 scripts/monitor_doe.py --quiet
```

## What You'll See

```
======================================================================
DOE Simulation Monitor
======================================================================
Monitoring: output/doe_results
Refresh interval: 5 seconds
Press Ctrl+C to stop monitoring (simulation will continue)
======================================================================

[18:36:45] ✓ RUNNING | Progress: 23/45 (51.1%) | Latest: condition_23_results.csv (2s ago) | ETA: 3.7min
[18:36:50] ✓ RUNNING | Progress: 24/45 (53.3%) | Latest: condition_24_results.csv (1s ago) | ETA: 3.5min
[18:36:55] ✓ RUNNING | Progress: 25/45 (55.6%) | Latest: condition_25_results.csv (0s ago) | ETA: 3.3min
...
[18:40:12] ✓ RUNNING | Progress: 45/45 (100.0%) | Latest: condition_45_results.csv (3s ago)

======================================================================
✓ SIMULATION COMPLETE!
======================================================================
Total conditions: 45/45
Total time: 6.2min

Next steps:
  1. Run analysis: python3 scripts/analyze_doe_results.py \
     --results-dir output/doe_results --output-dir output/analysis
  2. Review results and proceed with full DOE
======================================================================
```

## Tips

1. **Run in a separate terminal** - Keep your main terminal free for other work
2. **Use quiet mode** - If you just want to see when it's done, use `--quiet`
3. **Stop anytime** - Press Ctrl+C to stop monitoring (simulation keeps running)
4. **Check manually** - You can still check progress manually:
   ```bash
   ls output/doe_results/condition_*_results.csv | wc -l
   ```

## Other Monitoring Commands

### Watch log file in real-time
```bash
tail -f output/doe_test_run.log
```

### Check process status
```bash
ps aux | grep run_doe.py | grep -v grep
```

### Count completed conditions
```bash
ls output/doe_results/condition_*_results.csv | wc -l
```



