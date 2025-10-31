#!/usr/bin/env python3
"""
Display a progress window for DOE simulation.

Shows current progress, time estimates, and a visual progress bar.

Usage:
    python3 scripts/show_progress.py
    python3 scripts/show_progress.py --watch  # Auto-refresh every 5 seconds
"""

import time
import sys
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
import argparse


def clear_screen():
    """Clear terminal screen."""
    print("\033[2J\033[H", end="")


def get_progress(results_dir="output/doe_results"):
    """Get current progress from result files."""
    results_path = Path(results_dir)
    if not results_path.exists():
        return None, None, None, None, None, None
    
    condition_files = sorted(results_path.glob('condition_*_results.csv'))
    
    # Check replication count from files
    n_reps = None
    full_doe_files = []
    
    if condition_files:
        try:
            import pandas as pd
            # Check all files to find ones with 30 replications (full DOE)
            for f in condition_files:
                try:
                    df = pd.read_csv(f)
                    if len(df) == 30:
                        full_doe_files.append(f)
                        if n_reps is None:
                            n_reps = 30
                except:
                    pass
            
            # If we found full DOE files, count those
            if full_doe_files:
                n_complete = len(full_doe_files)
                most_recent = max(full_doe_files, key=lambda p: p.stat().st_mtime)
            else:
                # All files are test files (5 reps) or full DOE hasn't started yet
                n_complete = 0
                most_recent = max(condition_files, key=lambda p: p.stat().st_mtime) if condition_files else None
                # Check if process is running - if so, count files being overwritten
                try:
                    result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
                    processes = [line for line in result.stdout.split('\n') 
                               if 'run_doe.py' in line and '30' in line and 'grep' not in line]
                    if processes:
                        # Full DOE is running, count files (will be overwritten)
                        n_complete = len(condition_files)
                        most_recent = max(condition_files, key=lambda p: p.stat().st_mtime)
                except:
                    pass
        except:
            n_complete = len(condition_files)
            most_recent = max(condition_files, key=lambda p: p.stat().st_mtime) if condition_files else None
    
    # Get most recent file info
    if condition_files:
        most_recent = max(condition_files, key=lambda p: p.stat().st_mtime)
        mod_time = most_recent.stat().st_mtime
        age_seconds = time.time() - mod_time
        return n_complete, 45, most_recent.name, age_seconds, n_reps, len(full_doe_files) > 0
    else:
        return 0, 45, None, None, None, False


def get_log_info(log_file="output/doe_full_run.log"):
    """Extract current condition and replication from log."""
    log_path = Path(log_file)
    if not log_path.exists():
        return None, None
    
    try:
        # Read last 50 lines
        with open(log_path, 'r') as f:
            lines = f.readlines()
        
        # Look for replication info in last lines
        for line in reversed(lines[-50:]):
            if "Replication" in line and "/" in line:
                # Extract "X/30" pattern
                parts = line.split("Replication")
                if len(parts) > 1:
                    rep_part = parts[1].strip().split()[0]
                    if "/" in rep_part:
                        current_rep = int(rep_part.split("/")[0])
                        total_reps = int(rep_part.split("/")[1])
                        return current_rep, total_reps
            
            # Look for condition info
            if "Condition" in line and "complete" not in line.lower():
                # Try to extract condition number
                for word in line.split():
                    if word.isdigit() and len(word) <= 2:
                        condition_num = int(word)
                        if 1 <= condition_num <= 45:
                            return None, None  # Will use condition files instead
    
    except Exception:
        pass
    
    return None, None


def draw_progress_bar(current, total, width=50):
    """Draw a visual progress bar."""
    if total == 0:
        return "[" + " " * width + "]"
    
    filled = int((current / total) * width)
    bar = "█" * filled + "░" * (width - filled)
    pct = (current / total * 100) if total > 0 else 0
    return f"[{bar}] {pct:.1f}%"


def format_time(seconds):
    """Format seconds into readable time."""
    if seconds is None or seconds < 0:
        return "N/A"
    
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}min"


def show_progress(watch=False, refresh_interval=5):
    """Display progress window."""
    results_dir = "output/doe_results"
    log_file = "output/doe_full_run.log"
    
    start_time = None
    last_progress = 0
    
    try:
        while True:
            # Get progress
            progress_result = get_progress(results_dir)
            if progress_result[0] is None:
                n_complete, total, latest_file, age_seconds, n_reps_file, is_full_doe = 0, 45, None, None, None, False
            else:
                n_complete, total, latest_file, age_seconds, n_reps_file, is_full_doe = progress_result
            
            current_rep, total_reps = get_log_info(log_file)
            
            # Determine replication count
            if n_reps_file:
                n_reps = n_reps_file
            elif total_reps:
                n_reps = total_reps
            else:
                n_reps = 30  # Default assumption
            
            # Clear screen if watching
            if watch:
                clear_screen()
            
            # Header
            print("=" * 70)
            if is_full_doe or (latest_file and age_seconds and age_seconds < 300):
                print("DOE SIMULATION PROGRESS (FULL RUN - 30 reps/condition)")
            else:
                print("DOE SIMULATION PROGRESS (TEST RUN - 5 reps/condition)")
            print("=" * 70)
            print()
            
            # Overall progress
            if n_complete is not None:
                pct = (n_complete / total * 100) if total > 0 else 0
                bar = draw_progress_bar(n_complete, total, width=50)
                
                print(f"Conditions: {n_complete}/{total} ({pct:.1f}%)")
                print(bar)
                print()
                
                # Calculate time estimates
                if start_time is None and n_complete > 0:
                    start_time = time.time()
                
                if start_time and n_complete > 0:
                    elapsed = time.time() - start_time
                    if n_complete < total:
                        # Estimate remaining
                        avg_time_per_condition = elapsed / n_complete
                        remaining_conditions = total - n_complete
                        est_remaining = avg_time_per_condition * remaining_conditions
                        
                        print(f"Time elapsed: {format_time(elapsed)}")
                        print(f"Time remaining: ~{format_time(est_remaining)}")
                        
                        # Estimate completion time
                        est_completion = datetime.now() + timedelta(seconds=est_remaining)
                        print(f"Estimated completion: {est_completion.strftime('%H:%M:%S')}")
                    else:
                        elapsed = time.time() - start_time
                        print(f"✓ COMPLETE! Total time: {format_time(elapsed)}")
                
                print()
                
                # Current condition info
                if latest_file:
                    print(f"Latest file: {latest_file}")
                    if age_seconds is not None:
                        if age_seconds < 30:
                            print(f"Status: ✓ Running ({age_seconds:.1f}s ago)")
                        elif age_seconds < 120:
                            print(f"Status: ⚠ Slow ({age_seconds:.1f}s ago)")
                        else:
                            print(f"Status: ✗ Stalled ({age_seconds:.1f}s ago)")
                    print()
                
                # Current replication (if available)
                if current_rep and total_reps:
                    rep_pct = (current_rep / total_reps * 100) if total_reps > 0 else 0
                    rep_bar = draw_progress_bar(current_rep, total_reps, width=30)
                    print(f"Current condition: ~{rep_bar}")
                    print(f"Replication: {current_rep}/{total_reps}")
                    print()
                
                # Progress rate
                if start_time and n_complete > last_progress:
                    elapsed_since_start = time.time() - start_time
                    rate = n_complete / elapsed_since_start if elapsed_since_start > 0 else 0
                    print(f"Rate: {rate:.2f} conditions/minute")
                    print()
                
                last_progress = n_complete
            else:
                print("⚠ No progress files found yet")
                print("  Check if simulation has started...")
                print()
            
            # Check if process is running
            try:
                result = subprocess.run(
                    ['ps', 'aux'],
                    capture_output=True,
                    text=True
                )
                processes = [
                    line for line in result.stdout.split('\n')
                    if 'run_doe.py' in line and 'grep' not in line
                ]
                
                if processes:
                    print("✓ Process is running")
                else:
                    if n_complete and n_complete < total:
                        print("✗ Process appears to have stopped!")
                    else:
                        print("✓ Process completed")
            except:
                pass
            
            print()
            print("=" * 70)
            
            if watch:
                print(f"Auto-refreshing every {refresh_interval} seconds... (Ctrl+C to stop)")
                time.sleep(refresh_interval)
            else:
                break
                
    except KeyboardInterrupt:
        print()
        print("Monitoring stopped.")


def main():
    parser = argparse.ArgumentParser(description='Show DOE simulation progress')
    parser.add_argument('--watch', action='store_true',
                       help='Auto-refresh every 5 seconds')
    parser.add_argument('--refresh', type=int, default=5,
                       help='Refresh interval in seconds (default: 5)')
    
    args = parser.parse_args()
    
    show_progress(watch=args.watch, refresh_interval=args.refresh)


if __name__ == '__main__':
    main()

