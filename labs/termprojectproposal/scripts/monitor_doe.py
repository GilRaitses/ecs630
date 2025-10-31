#!/usr/bin/env python3
"""
Real-time DOE simulation monitoring script with BIG ASCII numbers.
Run this in a separate terminal window to monitor progress.
"""

import time
import os
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

# ASCII art digits (5 rows tall)
ASCII_DIGITS = {
    '0': [' ████ ', '█    █', '█    █', '█    █', ' ████ '],
    '1': ['  ██  ', ' ███  ', '  ██  ', '  ██  ', '██████'],
    '2': [' ████ ', '█    █', '   ██ ', '  ██  ', '██████'],
    '3': [' ████ ', '█    █', '  ███ ', '█    █', ' ████ '],
    '4': ['█   █ ', '█   █ ', '██████', '    █ ', '    █ '],
    '5': ['██████', '█     ', ' ████ ', '     █', '██████'],
    '6': [' ████ ', '█     ', '█████ ', '█    █', ' ████ '],
    '7': ['██████', '    █ ', '   █  ', '  █   ', ' █    '],
    '8': [' ████ ', '█    █', ' ████ ', '█    █', ' ████ '],
    '9': [' ████ ', '█    █', ' █████', '     █', ' ████ '],
    '.': ['      ', '      ', '      ', '      ', '  ██  '],
    '/': ['     █', '    █ ', '   █  ', '  █   ', ' █    '],
    '-': ['      ', '      ', '██████', '      ', '      '],
    ' ': ['      ', '      ', '      ', '      ', '      '],
    'm': ['      ', '      ', ' ████ ', '█ █ ██', '█ █ ██'],
    'i': ['  ██  ', '      ', '  ██  ', '  ██  ', '██████'],
    'n': ['      ', '      ', '████  ', '█  █  ', '█  █  '],
    'K': ['█   █ ', '█  █  ', '███   ', '█  █  ', '█   █ '],
    'B': ['████  ', '█   █ ', '████  ', '█   █ ', '████  '],
}

def get_ascii_digit(char):
    """Get ASCII art for a single character."""
    return ASCII_DIGITS.get(char, ASCII_DIGITS[' '])

def create_progress_bar(current, total, width=20):
    """Create ASCII progress bar."""
    if total == 0:
        return "[]" + " " * width + "[]"
    
    filled = int((current / total) * width)
    bar = "[" + "█" * filled + "░" * (width - filled) + "]"
    return bar

def format_big_number(value, label, unit='', progress_bar=None):
    """Format a number as big ASCII art with label."""
    # Convert to string, handle floats
    if isinstance(value, float):
        if value < 100:
            str_val = f"{value:.1f}"
        else:
            str_val = f"{int(value)}"
    else:
        str_val = str(value)
    
    # Create ASCII art rows
    rows = [[''] * 5 for _ in range(5)]
    
    for char in str_val:
        digit_rows = get_ascii_digit(char)
        for i, row in enumerate(digit_rows):
            rows[i].append(row)
            rows[i].append(' ')  # Spacing between digits
    
    # Combine rows
    ascii_lines = [''.join(row).rstrip() for row in rows]
    
    # Add label and unit
    label_line = f"{label}"
    if unit:
        label_line += f" {unit}"
    
    # Add progress bar if provided
    if progress_bar:
        # Append progress bar on the same line as label, or on a new line below
        label_line += f"  {progress_bar}"
    
    return ascii_lines, label_line

def clear_screen():
    """Clear terminal screen."""
    os.system('clear' if os.name != 'nt' else 'cls')

def get_progress():
    """Get current progress from checkpoint and log files."""
    # Use absolute paths based on script location
    script_dir = Path(__file__).parent.parent
    checkpoint_file = script_dir / 'output/simulation_results/checkpoint.csv'
    log_file = script_dir / 'output/simulation_results/doe_run.log'
    results_file = script_dir / 'output/simulation_results/all_results.csv'
    
    progress = {
        'n_replications': 0,
        'n_conditions': 0,
        'progress_pct': 0.0,
        'remaining_conditions': 45,
        'eta_minutes': 0.0,
        'log_size_kb': 0.0,
        'is_complete': False,
        'current_replication': 0,
        'total_replications_per_condition': 30
    }
    
    # Parse log file FIRST to get real-time progress
    # (This will override CSV data which only updates when conditions complete)
    log_has_data = False
    
    if log_file.exists():
        try:
            import re
            with open(log_file, 'r') as f:
                lines = f.readlines()
            
            # Find simulation start time (first line with timestamp or first "Simulating" message)
            simulation_start_time = None
            if len(lines) > 0:
                # Try to find first meaningful log entry
                for line in lines[:100]:  # Check first 100 lines
                    if 'Simulating' in line or 'DOE' in line or 'Condition' in line:
                        # Use log file creation time as proxy for start time
                        simulation_start_time = log_file.stat().st_ctime
                        break
                if simulation_start_time is None:
                    simulation_start_time = log_file.stat().st_ctime
            
            # Find latest replication first
            latest_rep = None
            latest_rep_line_idx = None
            
            for i, line in enumerate(reversed(lines[-500:])):
                rep_match = re.search(r'Replication\s+(\d+)/(\d+)', line)
                if rep_match and latest_rep is None:
                    latest_rep = (int(rep_match.group(1)), int(rep_match.group(2)))
                    latest_rep_line_idx = len(lines) - i - 1
                    progress['current_replication'] = latest_rep[0]
                    progress['total_replications_per_condition'] = latest_rep[1]
                    break
            
            # Now find condition - search from beginning of log for most recent condition marker
            latest_condition_from_log = None
            latest_condition_line_idx = None
            
            # Search entire log file for condition markers (start from beginning)
            for i, line in enumerate(lines):
                # Look for "[X.X%] Condition X/45"
                cond_match = re.search(r'\[([\d.]+)%\]\s+Condition\s+(\d+)/45', line)
                if cond_match:
                    cond_num = int(cond_match.group(2))
                    latest_condition_from_log = cond_num
                    latest_condition_line_idx = i
                    # Don't break - we want the most recent one
                
                # Look for "Simulating condition X"
                sim_match = re.search(r'Simulating condition\s+(\d+)', line)
                if sim_match:
                    cond_num = int(sim_match.group(1))
                    latest_condition_from_log = cond_num
                    latest_condition_line_idx = i
                    # Don't break - we want the most recent one
            
            # If still not found, check if we're in condition 1
            if latest_condition_from_log is None:
                # If replication found but no condition marker, assume condition 1
                latest_condition_from_log = 1
            
            # Use log data to calculate progress
            if latest_condition_from_log and latest_rep:
                # Currently running a condition
                # The condition number shown is the one we're currently working on
                # Completed conditions = current condition - 1
                completed_conditions = latest_condition_from_log - 1
                progress['n_conditions'] = completed_conditions
                progress['remaining_conditions'] = 45 - completed_conditions
                
                # Calculate total replications: completed conditions × reps_per_condition + current replication
                completed_reps = completed_conditions * latest_rep[1]  # e.g., 1 condition × 30 = 30
                progress['n_replications'] = completed_reps + latest_rep[0]  # e.g., 30 + 20 = 50
                
                # Progress percentage based on total replications (45 conditions × 30 reps = 1350 total)
                total_expected_reps = 45 * latest_rep[1]
                progress['progress_pct'] = (progress['n_replications'] / total_expected_reps) * 100
                
                log_has_data = True
                
                # Estimate ETA based on simulation start time
                if simulation_start_time is not None:
                    current_time = time.time()
                    elapsed_seconds = current_time - simulation_start_time
                    
                    # If we have enough progress (at least 5 replications), estimate time per replication
                    if progress['n_replications'] >= 5 and elapsed_seconds > 10:
                        seconds_per_rep = elapsed_seconds / progress['n_replications']
                        remaining_reps = total_expected_reps - progress['n_replications']
                        eta_seconds = remaining_reps * seconds_per_rep
                        progress['eta_minutes'] = max(0.0, eta_seconds / 60.0)  # Ensure non-negative
                    elif progress['n_replications'] > 0:
                        # Early in simulation: use a conservative estimate
                        # Assume ~12 seconds per replication (from observed ~11.5s average)
                        seconds_per_rep = 12.0
                        remaining_reps = total_expected_reps - progress['n_replications']
                        eta_seconds = remaining_reps * seconds_per_rep
                        progress['eta_minutes'] = eta_seconds / 60.0
                    else:
                        # Fallback: 5 minutes per condition
                        avg_time_per_condition = 300.0
                        progress['eta_minutes'] = (progress['remaining_conditions'] * avg_time_per_condition) / 60.0
                else:
                    # Fallback: 5 minutes per condition
                    avg_time_per_condition = 300.0
                    progress['eta_minutes'] = (progress['remaining_conditions'] * avg_time_per_condition) / 60.0
                
                # Debug output (can remove later)
                # print(f"DEBUG: condition={latest_condition_from_log}, rep={latest_rep[0]}/{latest_rep[1]}, completed={completed_conditions}, total_reps={progress['n_replications']}")
            elif latest_condition_from_log:
                # Found condition but no replication - check if complete
                for line in reversed(lines[-100:]):
                    if f"Condition {latest_condition_from_log} complete" in line:
                        progress['n_conditions'] = latest_condition_from_log
                        progress['progress_pct'] = (latest_condition_from_log / 45) * 100
                        progress['remaining_conditions'] = 45 - latest_condition_from_log
                        progress['n_replications'] = latest_condition_from_log * 30
                        log_has_data = True
                        break
        except Exception:
            pass
    
    # Fallback to CSV data if log parsing didn't work
    if not log_has_data:
        checkpoint_data = None
        results_data = None
        
        if checkpoint_file.exists():
            try:
                checkpoint_data = pd.read_csv(checkpoint_file)
                checkpoint_mtime = checkpoint_file.stat().st_mtime
            except:
                checkpoint_mtime = 0
        else:
            checkpoint_mtime = 0
        
        if results_file.exists():
            try:
                results_data = pd.read_csv(results_file)
                results_mtime = results_file.stat().st_mtime
            except:
                results_mtime = 0
        else:
            results_mtime = 0
        
        # Use the most recent file, or combine if both exist
        if checkpoint_data is not None and results_data is not None:
            data = pd.concat([checkpoint_data, results_data], ignore_index=True)
            data = data.drop_duplicates(subset=['condition_id', 'replication'], keep='last')
        elif checkpoint_data is not None:
            data = checkpoint_data
        elif results_data is not None:
            data = results_data
        else:
            data = None
        
        if data is not None and len(data) > 0:
            try:
                progress['n_replications'] = len(data)
                completed_conditions = sorted(data['condition_id'].unique().tolist())
                progress['n_conditions'] = len(completed_conditions)
                
                # Progress percentage based on total replications (45 × 30 = 1350)
                total_expected_reps = 45 * 30
                progress['progress_pct'] = (progress['n_replications'] / total_expected_reps) * 100
                progress['remaining_conditions'] = 45 - progress['n_conditions']
                
                # Estimate ETA based on file modification times
                if progress['n_replications'] > 0:
                    most_recent_mtime = max(checkpoint_mtime, results_mtime)
                    current_time = time.time()
                    elapsed_seconds = current_time - most_recent_mtime
                    
                    # Estimate time per replication
                    seconds_per_rep = elapsed_seconds / progress['n_replications']
                    remaining_reps = total_expected_reps - progress['n_replications']
                    progress['eta_minutes'] = (remaining_reps * seconds_per_rep) / 60.0
                else:
                    # Fallback: 5 minutes per condition
                    avg_time_per_condition = 300.0
                    progress['eta_minutes'] = (progress['remaining_conditions'] * avg_time_per_condition) / 60.0
            except Exception:
                pass
    
    # Get log file size
    if log_file.exists():
        progress['log_size_kb'] = log_file.stat().st_size / 1024
    
    if results_file.exists():
        try:
            results = pd.read_csv(results_file)
            if len(results) >= 45 * 30:  # Expected total
                progress['is_complete'] = True
        except:
            pass
    
    return progress

def display_progress():
    """Display progress with BIG ASCII numbers."""
    clear_screen()
    
    progress = get_progress()
    
    print("="*80)
    print(" " * 25 + "DOE SIMULATION MONITOR")
    print("="*80)
    print()
    
    # Counter 1: Total replications with progress bar
    current_rep = progress.get('current_replication', 0)
    total_reps = progress.get('total_replications_per_condition', 30)
    
    # Always show progress bar if we have replication info
    if current_rep > 0:
        rep_progress = create_progress_bar(current_rep, total_reps, width=25)
        rep_label = f" (Rep {current_rep}/{total_reps})"
        reps_lines, reps_label = format_big_number(progress['n_replications'], "Total replications", progress_bar=rep_progress + rep_label)
    else:
        # No current replication info - just show total
        reps_lines, reps_label = format_big_number(progress['n_replications'], "Total replications")
    
    print(reps_label)
    for line in reps_lines:
        print(line)
    print()
    
    # Counter 2: Conditions completed
    cond_str = f"{progress['n_conditions']}/45"
    cond_lines, cond_label = format_big_number(cond_str, "Conditions completed")
    print(cond_label)
    for line in cond_lines:
        print(line)
    print()
    
    # Counter 3: Progress percentage
    pct_lines, pct_label = format_big_number(progress['progress_pct'], "Progress", "%")
    print(pct_label)
    for line in pct_lines:
        print(line)
    print()
    
    # Counter 4: Remaining conditions
    rem_lines, rem_label = format_big_number(progress['remaining_conditions'], "Remaining conditions")
    print(rem_label)
    for line in rem_lines:
        print(line)
    print()
    
    # Counter 5: Estimated time remaining
    eta_lines, eta_label = format_big_number(progress['eta_minutes'], "Estimated time remaining", "minutes")
    print(eta_label)
    for line in eta_lines:
        print(line)
    print()
    
    # Counter 6: Log file size
    log_lines, log_label = format_big_number(progress['log_size_kb'], "LOG FILE Size", "KB")
    print(log_label)
    for line in log_lines:
        print(line)
    print()
    
    if progress['is_complete']:
        print(" " * 20 + "✅ SIMULATION COMPLETE!")
    else:
        print(" " * 25 + "Press Ctrl+C to exit")
    
    print("="*80)
    print(f"Last update: {datetime.now().strftime('%H:%M:%S')}")

def main():
    """Main monitoring loop."""
    print("Starting DOE monitor with BIG numbers...")
    time.sleep(1)
    
    try:
        while True:
            display_progress()
            time.sleep(2)  # Update every 2 seconds
    except KeyboardInterrupt:
        clear_screen()
        print("\n\nMonitor stopped.")
        sys.exit(0)

if __name__ == '__main__':
    main()
