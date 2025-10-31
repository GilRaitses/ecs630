#!/usr/bin/env python3
"""
Validate simulated trajectories against empirical data.

Compares:
- Turn rate distributions
- Pause fraction
- Reversal rate
- Spatial distributions
- KPI matching
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path
from scipy import stats
import argparse
import sys

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))
from simulate_trajectories import simulate_doe_condition, load_config, create_stimulus_schedule, load_event_models

def load_empirical_kpis(events_file, trajectories_file=None):
    """
    Compute KPIs from empirical data.
    
    Parameters
    ----------
    events_file : str
        Path to events CSV file
    trajectories_file : str, optional
        Path to trajectories CSV file (for additional metrics)
    
    Returns
    -------
    kpis : dict
        Dictionary of empirical KPI values
    """
    print("Loading empirical data...")
    events_df = pd.read_csv(events_file)
    
    total_time = events_df['time'].max() - events_df['time'].min()
    
    # Turn rate (use reorientations if available, otherwise falls back to simple turns)
    # NOTE: is_reorientation flags are already event starts (one per reorientation)
    # so summing them gives the total number of reorientation events
    if 'is_reorientation' in events_df.columns:
        total_turns = events_df['is_reorientation'].sum()
        print(f"  Using reorientations for turn rate (proper detection)")
        print(f"    Total reorientation events: {total_turns}")
    else:
        total_turns = events_df['is_turn'].sum()
        print(f"  Using simple turns for turn rate (fallback)")
    turn_rate = (total_turns / total_time) * 60 if total_time > 0 else 0.0
    
    # Pause rate and fraction
    total_pauses = events_df['is_pause'].sum()
    pause_rate = (total_pauses / total_time) * 60 if total_time > 0 else 0.0
    
    # Stop fraction (time paused)
    if 'pause_duration' in events_df.columns:
        stop_time = events_df[events_df['is_pause'] == True]['pause_duration'].sum()
    else:
        # Approximate: count pause bins
        dt = events_df['time'].iloc[1] - events_df['time'].iloc[0] if len(events_df) > 1 else 0.05
        stop_time = events_df['is_pause'].sum() * dt
    stop_fraction = stop_time / total_time if total_time > 0 else 0.0
    
    # Reversal rate
    total_reversals = events_df['is_reversal'].sum()
    reversal_rate = (total_reversals / total_time) * 60 if total_time > 0 else 0.0
    
    # Spine curve energy
    mean_spine_energy = events_df['spine_curve_energy'].mean() if 'spine_curve_energy' in events_df.columns else np.nan
    
    # Per-track statistics
    track_turn_rates = []
    track_pause_rates = []
    
    for track_id in events_df['track_id'].unique():
        track_data = events_df[events_df['track_id'] == track_id]
        track_time = track_data['time'].max() - track_data['time'].min()
        if track_time > 0:
            # Use reorientations if available, otherwise simple turns
            if 'is_reorientation' in track_data.columns:
                track_turn_rate = (track_data['is_reorientation'].sum() / track_time) * 60
            else:
                track_turn_rate = (track_data['is_turn'].sum() / track_time) * 60
            track_pause_rate = (track_data['is_pause'].sum() / track_time) * 60
            track_turn_rates.append(track_turn_rate)
            track_pause_rates.append(track_pause_rate)
    
    empirical_kpis = {
        'turn_rate': turn_rate,
        'pause_rate': pause_rate,
        'reversal_rate': reversal_rate,
        'stop_fraction': stop_fraction,
        'mean_spine_curve_energy': mean_spine_energy,
        'total_time': total_time,
        'total_turns': total_turns,
        'total_pauses': total_pauses,
        'total_reversals': total_reversals,
        'n_tracks': events_df['track_id'].nunique(),
        'track_turn_rates': track_turn_rates,
        'track_pause_rates': track_pause_rates
    }
    
    print(f"  Loaded empirical KPIs:")
    print(f"    Turn rate: {turn_rate:.2f} turns/min")
    print(f"    Pause rate: {pause_rate:.2f} pauses/min")
    print(f"    Reversal rate: {reversal_rate:.2f} reversals/min")
    print(f"    Stop fraction: {stop_fraction:.3f}")
    print(f"    Tracks: {empirical_kpis['n_tracks']}")
    
    return empirical_kpis


def run_validation_simulations(models_dict, config, empirical_data, n_replications=10, max_time=100.0):
    """
    Run validation simulations matching empirical stimulus conditions.
    
    Uses shorter simulation time for faster validation.
    Models are pre-loaded to avoid reloading bottleneck.
    
    Parameters
    ----------
    models_dict : dict
        Pre-loaded models dictionary
    config : dict
        Model configuration
    empirical_data : DataFrame
        Empirical trajectory data (sample)
    n_replications : int
        Number of simulation replications (default 10 for quick validation)
    max_time : float
        Maximum simulation time (default 100s for quick validation)
    
    Returns
    -------
    simulated_kpis : list
        List of KPI dictionaries (one per replication)
    """
    print(f"\nRunning {n_replications} validation simulations (max_time={max_time}s)...")
    print(f"  Note: Using shorter simulation time for quick validation")
    
    # Extract stimulus parameters from empirical data
    # Use the actual experimental conditions (intensity, pulse duration, interval)
    # For now, use a baseline condition similar to the empirical experiment
    condition = {
        'condition_id': 0,  # Validation condition
        'intensity_pct': 100.0,  # Use empirical intensity level
        'pulse_duration_s': 10.0,  # Fixed 10s pulses
        'inter_pulse_interval_s': 50.0  # Approximate from empirical (60s interval - 10s pulse)
    }
    
    import time
    start_time = time.time()
    
    # Use pre-loaded models (no reloading bottleneck)
    from simulate_trajectories import simulate_doe_condition
    results = simulate_doe_condition(
        models_dict,  # Pre-loaded models
        condition,
        config,
        empirical_data=empirical_data,
        n_replications=n_replications,
        max_time=max_time
    )
    
    elapsed = time.time() - start_time
    simulated_kpis = results.to_dict('records')
    
    print(f"  ✓ Completed {len(simulated_kpis)} simulations in {elapsed:.1f}s")
    
    return simulated_kpis


def compare_kpis(empirical_kpis, simulated_kpis_list):
    """
    Compare empirical and simulated KPIs.
    
    Parameters
    ----------
    empirical_kpis : dict
        Empirical KPI values
    simulated_kpis_list : list
        List of simulated KPI dictionaries
    
    Returns
    -------
    comparison : dict
        Comparison statistics
    """
    print("\n" + "="*80)
    print("COMPARING EMPIRICAL VS SIMULATED KPIs")
    print("="*80)
    
    # Convert simulated KPIs to DataFrame
    sim_df = pd.DataFrame(simulated_kpis_list)
    
    comparison = {}
    
    # Compare each KPI
    kpi_names = ['turn_rate', 'pause_rate', 'reversal_rate', 'stop_fraction', 'mean_spine_curve_energy']
    
    for kpi_name in kpi_names:
        if kpi_name in empirical_kpis and kpi_name in sim_df.columns:
            emp_value = empirical_kpis[kpi_name]
            sim_values = sim_df[kpi_name].values
            sim_mean = np.mean(sim_values)
            sim_std = np.std(sim_values)
            sim_ci_lower = np.percentile(sim_values, 2.5)
            sim_ci_upper = np.percentile(sim_values, 97.5)
            
            # Check if empirical value is within simulated CI
            within_ci = (sim_ci_lower <= emp_value <= sim_ci_upper)
            
            # Relative error
            relative_error = abs(sim_mean - emp_value) / (emp_value + 1e-6) * 100
            
            comparison[kpi_name] = {
                'empirical': float(emp_value),
                'simulated_mean': float(sim_mean),
                'simulated_std': float(sim_std),
                'simulated_ci_lower': float(sim_ci_lower),
                'simulated_ci_upper': float(sim_ci_upper),
                'within_ci': bool(within_ci),
                'relative_error_pct': float(relative_error)
            }
            
            print(f"\n{kpi_name.upper()}:")
            print(f"  Empirical: {emp_value:.4f}")
            print(f"  Simulated: {sim_mean:.4f} ± {sim_std:.4f}")
            print(f"  95% CI: [{sim_ci_lower:.4f}, {sim_ci_upper:.4f}]")
            print(f"  Within CI: {'✓' if within_ci else '✗'}")
            print(f"  Relative error: {relative_error:.1f}%")
    
    # Statistical tests
    print("\n" + "="*80)
    print("STATISTICAL TESTS")
    print("="*80)
    
    # Turn rate distribution comparison
    if 'track_turn_rates' in empirical_kpis and len(empirical_kpis['track_turn_rates']) > 0:
        emp_turn_rates = np.array(empirical_kpis['track_turn_rates'])
        sim_turn_rates = sim_df['turn_rate'].values
        
        # Two-sample KS test
        ks_stat, ks_p = stats.ks_2samp(emp_turn_rates, sim_turn_rates)
        comparison['turn_rate_ks_test'] = {
            'statistic': float(ks_stat),
            'p_value': float(ks_p),
            'significant': bool(ks_p < 0.05)
        }
        
        print(f"\nTurn Rate Distribution (KS Test):")
        print(f"  KS statistic: {ks_stat:.4f}")
        print(f"  p-value: {ks_p:.4f}")
        print(f"  Significant difference: {'Yes' if ks_p < 0.05 else 'No'}")
    
    # Pause rate distribution comparison
    if 'track_pause_rates' in empirical_kpis and len(empirical_kpis['track_pause_rates']) > 0:
        emp_pause_rates = np.array(empirical_kpis['track_pause_rates'])
        sim_pause_rates = sim_df['pause_rate'].values
        
        ks_stat, ks_p = stats.ks_2samp(emp_pause_rates, sim_pause_rates)
        comparison['pause_rate_ks_test'] = {
            'statistic': float(ks_stat),
            'p_value': float(ks_p),
            'significant': bool(ks_p < 0.05)
        }
        
        print(f"\nPause Rate Distribution (KS Test):")
        print(f"  KS statistic: {ks_stat:.4f}")
        print(f"  p-value: {ks_p:.4f}")
        print(f"  Significant difference: {'Yes' if ks_p < 0.05 else 'No'}")
    
    return comparison


def plot_validation_results(empirical_kpis, simulated_kpis_list, comparison, output_dir):
    """
    Plot validation comparison.
    
    Parameters
    ----------
    empirical_kpis : dict
        Empirical KPI values
    simulated_kpis_list : list
        List of simulated KPI dictionaries
    comparison : dict
        Comparison statistics
    output_dir : Path
        Output directory for plots
    """
    sim_df = pd.DataFrame(simulated_kpis_list)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Simulation Validation: Empirical vs Simulated', fontsize=16, fontweight='bold')
    
    # Panel 1: Turn Rate Distribution
    if 'track_turn_rates' in empirical_kpis:
        emp_rates = empirical_kpis['track_turn_rates']
        sim_rates = sim_df['turn_rate'].values
        
        axes[0, 0].hist(emp_rates, bins=15, alpha=0.6, label='Empirical', color='blue', edgecolor='black')
        axes[0, 0].hist(sim_rates, bins=15, alpha=0.6, label='Simulated', color='red', edgecolor='black')
        axes[0, 0].axvline(np.mean(emp_rates), color='blue', linestyle='--', linewidth=2, label=f'Emp Mean: {np.mean(emp_rates):.2f}')
        axes[0, 0].axvline(np.mean(sim_rates), color='red', linestyle='--', linewidth=2, label=f'Sim Mean: {np.mean(sim_rates):.2f}')
        axes[0, 0].set_xlabel('Turn Rate (turns/min)')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Turn Rate Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # Panel 2: Pause Rate Distribution
    if 'track_pause_rates' in empirical_kpis:
        emp_rates = empirical_kpis['track_pause_rates']
        sim_rates = sim_df['pause_rate'].values
        
        axes[0, 1].hist(emp_rates, bins=15, alpha=0.6, label='Empirical', color='blue', edgecolor='black')
        axes[0, 1].hist(sim_rates, bins=15, alpha=0.6, label='Simulated', color='red', edgecolor='black')
        axes[0, 1].axvline(np.mean(emp_rates), color='blue', linestyle='--', linewidth=2, label=f'Emp Mean: {np.mean(emp_rates):.2f}')
        axes[0, 1].axvline(np.mean(sim_rates), color='red', linestyle='--', linewidth=2, label=f'Sim Mean: {np.mean(sim_rates):.2f}')
        axes[0, 1].set_xlabel('Pause Rate (pauses/min)')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Pause Rate Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Panel 3: Reversal Rate Comparison
    if 'reversal_rate' in comparison:
        comp = comparison['reversal_rate']
        axes[0, 2].bar(['Empirical', 'Simulated'], 
                       [comp['empirical'], comp['simulated_mean']],
                       yerr=[[comp['empirical'] - comp['empirical']], 
                             [comp['simulated_mean'] - comp['simulated_ci_lower']]],
                       color=['blue', 'red'], alpha=0.7, capsize=5)
        axes[0, 2].set_ylabel('Reversal Rate (reversals/min)')
        axes[0, 2].set_title('Reversal Rate Comparison')
        axes[0, 2].grid(axis='y', alpha=0.3)
    
    # Panel 4: Stop Fraction Comparison
    if 'stop_fraction' in comparison:
        comp = comparison['stop_fraction']
        axes[1, 0].bar(['Empirical', 'Simulated'], 
                       [comp['empirical'], comp['simulated_mean']],
                       yerr=[[comp['empirical'] - comp['empirical']], 
                             [comp['simulated_mean'] - comp['simulated_ci_lower']]],
                       color=['blue', 'red'], alpha=0.7, capsize=5)
        axes[1, 0].set_ylabel('Stop Fraction')
        axes[1, 0].set_title('Stop Fraction Comparison')
        axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Panel 5: KPI Comparison Summary
    kpi_names = ['turn_rate', 'pause_rate', 'reversal_rate', 'stop_fraction']
    emp_values = []
    sim_means = []
    sim_cis = []
    kpi_labels = []
    
    for kpi_name in kpi_names:
        if kpi_name in comparison:
            comp = comparison[kpi_name]
            emp_values.append(comp['empirical'])
            sim_means.append(comp['simulated_mean'])
            sim_cis.append([comp['simulated_mean'] - comp['simulated_ci_lower'],
                           comp['simulated_ci_upper'] - comp['simulated_mean']])
            kpi_labels.append(kpi_name.replace('_', ' ').title())
    
    if len(emp_values) > 0:
        x_pos = np.arange(len(kpi_labels))
        width = 0.35
        
        axes[1, 1].bar(x_pos - width/2, emp_values, width, label='Empirical', color='blue', alpha=0.7)
        axes[1, 1].bar(x_pos + width/2, sim_means, width, yerr=np.array(sim_cis).T, 
                       label='Simulated', color='red', alpha=0.7, capsize=5)
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].set_title('KPI Comparison Summary')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(kpi_labels, rotation=45, ha='right')
        axes[1, 1].legend()
        axes[1, 1].grid(axis='y', alpha=0.3)
    
    # Panel 6: Validation Summary Text
    summary_text = "Validation Summary:\n\n"
    for kpi_name in kpi_names:
        if kpi_name in comparison:
            comp = comparison[kpi_name]
            status = "✓" if comp['within_ci'] else "✗"
            summary_text += f"{status} {kpi_name}: {comp['relative_error_pct']:.1f}% error\n"
    
    if 'turn_rate_ks_test' in comparison:
        ks_comp = comparison['turn_rate_ks_test']
        summary_text += f"\nTurn Rate KS Test:\n"
        summary_text += f"  p={ks_comp['p_value']:.4f}\n"
        summary_text += f"  {'Match' if not ks_comp['significant'] else 'Mismatch'}\n"
    
    axes[1, 2].text(0.1, 0.5, summary_text, transform=axes[1, 2].transAxes,
                    fontsize=10, verticalalignment='center', fontfamily='monospace')
    axes[1, 2].set_title('Validation Summary')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    output_path = output_dir / 'simulation_validation.png'
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"\n✓ Saved validation plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Validate simulation against empirical data')
    parser.add_argument('--models-dir', type=str, default='output/fitted_models',
                       help='Directory containing fitted models')
    parser.add_argument('--events-file', type=str,
                       default='data/engineered_tier2/GMR61_tier2_events.csv',
                       help='Path to empirical events CSV')
    parser.add_argument('--trajectories-file', type=str, default=None,
                       help='Path to empirical trajectories CSV (optional)')
    parser.add_argument('--config', type=str, default='config/model_config.json',
                       help='Path to model config JSON')
    parser.add_argument('--n-replications', type=int, default=10,
                       help='Number of simulation replications (default 10 for quick validation)')
    parser.add_argument('--max-time', type=float, default=100.0,
                       help='Maximum simulation time (seconds, default 100s for quick validation)')
    parser.add_argument('--output-dir', type=str, default='output/validation',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("SIMULATION VALIDATION")
    print("="*80)
    
    # Load config
    config = load_config(args.config)
    
    # Load empirical KPIs
    empirical_kpis = load_empirical_kpis(args.events_file, args.trajectories_file)
    
    # Phase 1: Load models once (bottleneck fix)
    print("\n" + "="*80)
    print("PHASE 1: LOADING MODELS")
    print("="*80)
    print(f"Loading models from {args.models_dir}...")
    import time
    model_load_start = time.time()
    models_dict = load_event_models(args.models_dir)
    model_load_time = time.time() - model_load_start
    print(f"  ✓ Loaded models in {model_load_time:.2f}s")
    for event_type, model in models_dict.items():
        if model is not None:
            print(f"    {event_type}: {model.get('type', 'unknown')} model")
    
    # Phase 2: Load empirical data (sample subset for speed/position sampling)
    print("\n" + "="*80)
    print("PHASE 2: LOADING EMPIRICAL DATA")
    print("="*80)
    print("  Loading sample of empirical data (for speed/position sampling)...")
    data_load_start = time.time()
    # Only need a sample for speed/position sampling - don't need all 273K rows
    empirical_data_full = pd.read_csv(args.events_file)
    # Sample 10K rows for speed/position sampling (enough for statistics)
    if len(empirical_data_full) > 10000:
        sample_size = 10000
        empirical_data = empirical_data_full.sample(n=sample_size, random_state=42).reset_index(drop=True)
        print(f"  Sampled {sample_size:,} rows from {len(empirical_data_full):,} total records")
    else:
        empirical_data = empirical_data_full
        print(f"  Loaded {len(empirical_data):,} event records")
    data_load_time = time.time() - data_load_start
    print(f"  ✓ Loaded empirical data in {data_load_time:.2f}s")
    
    # Phase 3: Run validation simulations
    print("\n" + "="*80)
    print("PHASE 3: RUNNING VALIDATION SIMULATIONS")
    print("="*80)
    simulated_kpis = run_validation_simulations(
        models_dict,  # Pre-loaded models
        config,
        empirical_data,
        n_replications=args.n_replications,
        max_time=args.max_time
    )
    
    # Compare KPIs
    comparison = compare_kpis(empirical_kpis, simulated_kpis)
    
    # Plot results
    print("\n" + "="*80)
    print("GENERATING VALIDATION PLOTS")
    print("="*80)
    plot_validation_results(empirical_kpis, simulated_kpis, comparison, output_dir)
    
    # Save results (convert numpy types to Python native types for JSON)
    results_path = output_dir / 'validation_results.json'
    
    # Convert numpy types to Python native types
    def convert_to_native(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        return obj
    
    empirical_kpis_for_json = {k: convert_to_native(v) for k, v in empirical_kpis.items() 
                               if k not in ['track_turn_rates', 'track_pause_rates']}
    
    with open(results_path, 'w') as f:
        json.dump({
            'empirical_kpis': empirical_kpis_for_json,
            'simulated_kpis': convert_to_native(simulated_kpis),
            'comparison': convert_to_native(comparison)
        }, f, indent=2)
    print(f"\n✓ Saved validation results to {results_path}")
    
    # Save simulated KPIs CSV
    sim_df = pd.DataFrame(simulated_kpis)
    sim_df.to_csv(output_dir / 'simulated_kpis.csv', index=False)
    print(f"✓ Saved simulated KPIs to {output_dir / 'simulated_kpis.csv'}")
    
    print("\n" + "="*80)
    print("✓ VALIDATION COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()

