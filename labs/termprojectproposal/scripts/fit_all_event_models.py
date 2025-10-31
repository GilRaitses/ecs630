#!/usr/bin/env python3
"""
Fit hazard models for all event types: turn, pause, reversal.

Includes spine curve energy as a feature.
Fits both baseline and full models for each event type.
"""

import subprocess
import sys
from pathlib import Path

def fit_all_event_models(events_file, output_dir, config_file='config/model_config.json'):
    """
    Fit baseline and full models for all event types.
    
    Parameters
    ----------
    events_file : str
        Path to events CSV file
    output_dir : str
        Output directory for fitted models
    config_file : str
        Path to model config JSON
    """
    event_types = ['turn', 'pause', 'reversal']
    
    print("="*80)
    print("FITTING HAZARD MODELS FOR ALL EVENT TYPES")
    print("="*80)
    print(f"Events file: {events_file}")
    print(f"Output directory: {output_dir}")
    print(f"Event types: {event_types}")
    print(f"\nFeatures included:")
    print(f"  - Stimulus kernel (temporal basis functions)")
    print(f"  - Speed (normalized)")
    print(f"  - Heading (sin/cos)")
    print(f"  - Spine curve energy (log-normalized)")
    
    for event_type in event_types:
        print(f"\n{'='*80}")
        print(f"FITTING {event_type.upper()} MODEL")
        print(f"{'='*80}")
        
        # Fit baseline model
        print(f"\n[1/2] Fitting baseline model...")
        # Extract trajectory directory from events file path
        events_path = Path(events_file)
        trajectory_dir = str(events_path.parent)
        
        cmd_baseline = [
            sys.executable, 'scripts/fit_hazard_model.py',
            '--trajectory-dir', trajectory_dir,
            '--events-file', events_file,
            '--event-type', event_type,
            '--output-dir', output_dir,
            '--config', config_file,
            '--baseline-only'
        ]
        
        result = subprocess.run(cmd_baseline, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        if result.returncode != 0:
            print(f"⚠ Warning: Baseline model fitting failed for {event_type}")
        
        # Fit full model
        print(f"\n[2/2] Fitting full model...")
        cmd_full = [
            sys.executable, 'scripts/fit_hazard_model.py',
            '--trajectory-dir', trajectory_dir,
            '--events-file', events_file,
            '--event-type', event_type,
            '--output-dir', output_dir,
            '--config', config_file
        ]
        
        result = subprocess.run(cmd_full, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        if result.returncode != 0:
            print(f"⚠ Warning: Full model fitting failed for {event_type}")
    
    print(f"\n{'='*80}")
    print("✓ ALL MODELS FITTED")
    print(f"{'='*80}")
    print(f"\nModels saved to: {output_dir}")
    print("\nGenerated files:")
    for event_type in event_types:
        print(f"  - {event_type}_baseline_model.pkl")
        print(f"  - {event_type}_baseline_summary.json")
        print(f"  - {event_type}_full_model.pkl")
        print(f"  - {event_type}_full_summary.json")
        print(f"  - {event_type}_coefficients.csv")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Fit hazard models for all event types')
    parser.add_argument('--events-file', type=str,
                       default='data/engineered_tier2/GMR61_tier2_events.csv',
                       help='Path to events CSV file')
    parser.add_argument('--output-dir', type=str,
                       default='output/fitted_models',
                       help='Output directory for fitted models')
    parser.add_argument('--config', type=str,
                       default='config/model_config.json',
                       help='Path to model config JSON')
    
    args = parser.parse_args()
    
    fit_all_event_models(args.events_file, args.output_dir, args.config)

