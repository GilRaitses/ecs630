#!/usr/bin/env python3
"""
Run full factorial Design of Experiments (DOE) simulations.

Executes all 27 conditions from doe_table.csv with 30 replications each,
generating 810 total simulated trajectories.

Usage:
    python3 scripts/run_doe.py \
        --doe-table config/doe_table.csv \
        --models-dir output/fitted_models \
        --config config/model_config.json \
        --output-dir output/doe_results \
        --n-replications 30
"""

import pandas as pd
import numpy as np
import json
import argparse
import sys
from pathlib import Path
import time

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))
from simulate_trajectories import simulate_doe_condition, load_event_models, load_config
from prepare_simulation_dataset import extract_empirical_distributions


def load_doe_table(doe_table_path):
    """
    Load DOE table from CSV.
    
    Parameters
    ----------
    doe_table_path : str
        Path to DOE table CSV
    
    Returns
    -------
    doe_table : DataFrame
        DOE conditions with columns: condition_id, intensity_pct, pulse_duration_s, inter_pulse_interval_s
    """
    # Read CSV - handle quoted descriptions
    doe_table = pd.read_csv(doe_table_path, quotechar='"')
    
    # Validate required columns
    required_cols = ['condition_id', 'intensity_pct', 'pulse_duration_s', 'inter_pulse_interval_s']
    missing_cols = [c for c in required_cols if c not in doe_table.columns]
    if missing_cols:
        raise ValueError(f"DOE table missing required columns: {missing_cols}")
    
    # Ensure numeric columns are float (not string)
    numeric_cols = ['intensity_pct', 'pulse_duration_s', 'inter_pulse_interval_s']
    for col in numeric_cols:
        if col in doe_table.columns:
            doe_table[col] = pd.to_numeric(doe_table[col], errors='coerce')
    
    # Sort by condition_id to ensure consistent ordering
    doe_table = doe_table.sort_values('condition_id').reset_index(drop=True)
    
    return doe_table


def run_full_doe(doe_table, models_dict, config, empirical_data=None,
                n_replications=30, max_time=300.0, output_dir='output/doe_results',
                random_seed_base=42):
    """
    Run full factorial DOE across all conditions.
    
    Parameters
    ----------
    doe_table : DataFrame
        DOE conditions table
    models_dict : dict
        Pre-loaded event models dictionary
    config : dict
        Model configuration
    empirical_data : DataFrame, optional
        Empirical data for speed/position sampling
    n_replications : int
        Number of replications per condition
    max_time : float
        Maximum simulation time per trajectory
    output_dir : str
        Output directory for results
    random_seed_base : int
        Base random seed for reproducibility
    
    Returns
    -------
    all_results : DataFrame
        Combined results from all conditions and replications
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    n_conditions = len(doe_table)
    total_simulations = n_conditions * n_replications
    
    print("=" * 80)
    print("FULL FACTORIAL DOE SIMULATION")
    print("=" * 80)
    print(f"Conditions: {n_conditions}")
    print(f"Replications per condition: {n_replications}")
    print(f"Total simulations: {total_simulations}")
    print(f"Max time per simulation: {max_time}s")
    print("=" * 80)
    print()
    
    all_results = []
    overall_start_time = time.time()
    
    # Iterate through all conditions
    for idx, condition in doe_table.iterrows():
        condition_num = idx + 1
        print(f"[{condition_num}/{n_conditions}] Condition {condition['condition_id']}: "
              f"Intensity={condition['intensity_pct']}%, "
              f"Pulse={condition['pulse_duration_s']}s, "
              f"Interval={condition['inter_pulse_interval_s']}s")
        
        try:
            # Convert condition row to dict with proper types
            condition_dict = {
                'condition_id': int(condition['condition_id']),
                'intensity_pct': float(condition['intensity_pct']),
                'pulse_duration_s': float(condition['pulse_duration_s']),
                'inter_pulse_interval_s': float(condition['inter_pulse_interval_s'])
            }
            
            # Simulate all replications for this condition
            condition_results = simulate_doe_condition(
                models_dict=models_dict,
                condition=condition_dict,
                config=config,
                empirical_data=empirical_data,
                n_replications=n_replications,
                max_time=max_time,
                random_seed_base=random_seed_base
            )
            
            # Save condition results immediately (in case of crash)
            condition_file = output_path / f"condition_{condition['condition_id']}_results.csv"
            condition_results.to_csv(condition_file, index=False)
            
            all_results.append(condition_results)
            
            # Print condition summary
            print(f"  ✓ Condition {condition['condition_id']} complete: "
                  f"{len(condition_results)} replications saved to {condition_file}")
            print()
            
        except Exception as e:
            print(f"  ✗ ERROR in condition {condition['condition_id']}: {e}")
            import traceback
            traceback.print_exc()
            print()
            continue
    
    # Combine all results
    if all_results:
        all_results_df = pd.concat(all_results, ignore_index=True)
        
        # Save combined results
        combined_file = output_path / "all_results.csv"
        all_results_df.to_csv(combined_file, index=False)
        print(f"✓ Combined results saved to {combined_file}")
        print(f"  Total rows: {len(all_results_df)}")
        
        # Save summary statistics
        summary = compute_summary_statistics(all_results_df)
        summary_file = output_path / "summary_statistics.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"✓ Summary statistics saved to {summary_file}")
        
        # Print overall summary
        overall_elapsed = time.time() - overall_start_time
        print()
        print("=" * 80)
        print("DOE SIMULATION COMPLETE")
        print("=" * 80)
        print(f"Total conditions completed: {len(all_results)}/{n_conditions}")
        print(f"Total simulations: {len(all_results_df)}")
        print(f"Total elapsed time: {overall_elapsed/60:.1f} minutes")
        print(f"Average time per simulation: {overall_elapsed/len(all_results_df):.2f} seconds")
        print("=" * 80)
        
        return all_results_df
    else:
        print("✗ ERROR: No results collected!")
        return None


def compute_summary_statistics(results_df):
    """
    Compute summary statistics across all conditions.
    
    Parameters
    ----------
    results_df : DataFrame
        Combined results from all DOE simulations
    
    Returns
    -------
    summary : dict
        Summary statistics dictionary
    """
    summary = {
        'n_conditions': results_df['condition_id'].nunique(),
        'n_replications_total': len(results_df),
        'n_replications_per_condition': results_df.groupby('condition_id').size().to_dict(),
        'conditions': []
    }
    
    # KPI columns (exclude metadata columns)
    kpi_cols = [c for c in results_df.columns 
                if c not in ['condition_id', 'replication', 'intensity_pct', 
                            'pulse_duration_s', 'inter_pulse_interval_s']]
    
    # Compute statistics per condition
    for condition_id in sorted(results_df['condition_id'].unique()):
        condition_data = results_df[results_df['condition_id'] == condition_id]
        condition_summary = {
            'condition_id': int(condition_id),
            'n_replications': len(condition_data),
            'kpis': {}
        }
        
        for kpi in kpi_cols:
            if condition_data[kpi].dtype in [np.float64, np.int64]:
                condition_summary['kpis'][kpi] = {
                    'mean': float(condition_data[kpi].mean()),
                    'std': float(condition_data[kpi].std()),
                    'min': float(condition_data[kpi].min()),
                    'max': float(condition_data[kpi].max()),
                    'median': float(condition_data[kpi].median()),
                    'q25': float(condition_data[kpi].quantile(0.25)),
                    'q75': float(condition_data[kpi].quantile(0.75))
                }
        
        summary['conditions'].append(condition_summary)
    
    # Overall statistics
    summary['overall_kpis'] = {}
    for kpi in kpi_cols:
        if results_df[kpi].dtype in [np.float64, np.int64]:
            summary['overall_kpis'][kpi] = {
                'mean': float(results_df[kpi].mean()),
                'std': float(results_df[kpi].std()),
                'min': float(results_df[kpi].min()),
                'max': float(results_df[kpi].max())
            }
    
    return summary


def main():
    """Main DOE execution pipeline."""
    parser = argparse.ArgumentParser(description='Run full factorial DOE simulations')
    parser.add_argument('--doe-table', type=str, required=True,
                       help='Path to DOE table CSV (config/doe_table.csv)')
    parser.add_argument('--models-dir', type=str,
                       default='output/fitted_models',
                       help='Directory containing fitted model pickle files')
    parser.add_argument('--config', type=str,
                       default='config/model_config.json',
                       help='Path to model configuration JSON')
    parser.add_argument('--output-dir', type=str,
                       default='output/doe_results',
                       help='Output directory for DOE results')
    parser.add_argument('--n-replications', type=int,
                       default=30,
                       help='Number of replications per condition (default: 30)')
    parser.add_argument('--max-time', type=float,
                       default=300.0,
                       help='Maximum simulation time per trajectory (default: 300s)')
    parser.add_argument('--trajectories-file', type=str,
                       help='Path to empirical trajectories CSV (for sampling speed/position)')
    parser.add_argument('--random-seed-base', type=int,
                       default=42,
                       help='Base random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Load DOE table
    print("Loading DOE table...")
    doe_table = load_doe_table(args.doe_table)
    print(f"  ✓ Loaded {len(doe_table)} conditions")
    
    # Load configuration
    print("Loading configuration...")
    config = load_config(args.config)
    print(f"  ✓ Configuration loaded")
    
    # Load models (pre-load once to avoid bottleneck)
    print("Loading event models...")
    models_dict = load_event_models(args.models_dir)
    print(f"  ✓ Loaded models: {list(models_dict.keys())}")
    
    # Load empirical data if provided (for speed/position sampling)
    # Pass DataFrame directly (not extracted distributions) - simulate_single_trajectory expects DataFrame
    empirical_data = None
    if args.trajectories_file:
        print("Loading empirical data for sampling...")
        empirical_data = pd.read_csv(args.trajectories_file)
        print(f"  ✓ Empirical data loaded from {args.trajectories_file} ({len(empirical_data)} rows)")
    
    # Run full DOE
    results_df = run_full_doe(
        doe_table=doe_table,
        models_dict=models_dict,
        config=config,
        empirical_data=empirical_data,
        n_replications=args.n_replications,
        max_time=args.max_time,
        output_dir=args.output_dir,
        random_seed_base=args.random_seed_base
    )
    
    if results_df is not None:
        print("\n✓ DOE execution complete!")
        print(f"  Results saved to: {args.output_dir}")
        print(f"  Next step: Run scripts/generate_doe_summaries.py to compute CIs and generate Arena CSVs")
    else:
        print("\n✗ DOE execution failed!")
        sys.exit(1)


if __name__ == '__main__':
    main()

