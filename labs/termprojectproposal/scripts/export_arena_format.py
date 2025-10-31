#!/usr/bin/env python3
"""
Export simulation results to Arena-style CSV format.

Generates:
- AcrossReplicationsSummary.csv: Mean, CI, min, max for each KPI by condition
- ContinuousTimeStatsByRep.csv: Time-persistent statistics per replication
- DiscreteTimeStatsByRep.csv: Event-based statistics per replication
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from scipy import stats

def load_config(config_path="config/model_config.json"):
    """Load model configuration."""
    with open(config_path, 'r') as f:
        return json.load(f)

def compute_confidence_interval(data, confidence=0.95):
    """Compute confidence interval for data."""
    n = len(data)
    if n < 2:
        return np.nan, np.nan, np.nan
    
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    sem = std / np.sqrt(n)
    
    # t-distribution critical value
    t_crit = stats.t.ppf((1 + confidence) / 2, df=n-1)
    
    half_width = t_crit * sem
    ci_lower = mean - half_width
    ci_upper = mean + half_width
    
    return mean, ci_lower, ci_upper

def create_across_replications_summary(results_df, config):
    """Create AcrossReplicationsSummary.csv format."""
    # Drop approach_rate if present (not applicable to omnidirectional stimulus)
    if 'approach_rate' in results_df.columns:
        results_df = results_df.drop(columns=['approach_rate'])
    
    # Define KPIs based on actual simulation output columns
    kpi_mapping = {
        'turn_rate': 'TurnRate',
        'latency': 'Latency',
        'stop_fraction': 'StopFraction',
        'pause_rate': 'PauseRate',
        'reversal_rate': 'ReversalRate',  # Primary name
        'heading_reversal_rate': 'ReversalRate',  # Backward compatibility alias
        'tortuosity': 'Tortuosity',
        'dispersal': 'Dispersal',
        'mean_spine_curve_energy': 'MeanSpineCurveEnergy'
    }
    
    conditions = results_df['condition_id'].unique()
    
    summary_rows = []
    
    for condition in sorted(conditions):
        cond_data = results_df[results_df['condition_id'] == condition]
        
        row = {
            'ConditionID': int(condition),
            'Intensity': cond_data['intensity_pct'].iloc[0],
            'PulseDuration': cond_data['pulse_duration_s'].iloc[0],
            'InterPulseInterval': cond_data['inter_pulse_interval_s'].iloc[0]
        }
        
        # Compute statistics for each KPI
        for kpi_col, kpi_name in kpi_mapping.items():
            if kpi_col not in cond_data.columns:
                continue
                
            kpi_values = cond_data[kpi_col].values
            kpi_values = kpi_values[~np.isnan(kpi_values)]  # Remove NaN values
            
            if len(kpi_values) == 0:
                continue
            
            mean, ci_lower, ci_upper = compute_confidence_interval(kpi_values)
            std = np.std(kpi_values, ddof=1)
            min_val = np.min(kpi_values)
            max_val = np.max(kpi_values)
            
            row[f'{kpi_name}_Mean'] = mean
            row[f'{kpi_name}_StdDev'] = std
            row[f'{kpi_name}_CILower'] = ci_lower
            row[f'{kpi_name}_CIUpper'] = ci_upper
            row[f'{kpi_name}_Min'] = min_val
            row[f'{kpi_name}_Max'] = max_val
            row[f'{kpi_name}_HalfWidth'] = (ci_upper - ci_lower) / 2
        
        row['NReplications'] = len(cond_data)
        summary_rows.append(row)
    
    return pd.DataFrame(summary_rows)

def create_continuous_time_stats(results_df):
    """Create ContinuousTimeStatsByRep.csv format."""
    # Time-persistent statistics (averages over simulation duration)
    stats_rows = []
    
    for _, row in results_df.iterrows():
        stats_rows.append({
            'ConditionID': int(row['condition_id']),
            'Replication': int(row['replication']),
            'AvgSpeed': row.get('turn_rate', np.nan),  # Approximate from turn rate
            'AvgDistanceFromStart': row.get('dispersal', np.nan),
            'AvgWallDistance': np.nan,  # Not computed in simulation
            'AvgTimeInState_Run': row.get('total_time', np.nan) * (1 - row.get('stop_fraction', 0)),
            'AvgTimeInState_Stop': row.get('total_time', np.nan) * row.get('stop_fraction', 0),
            'AvgTimeInState_Turn': np.nan,  # Not directly tracked
            'MeanSpineCurveEnergy': row.get('mean_spine_curve_energy', np.nan)
        })
    
    return pd.DataFrame(stats_rows)

def create_discrete_time_stats(results_df):
    """Create DiscreteTimeStatsByRep.csv format."""
    # Event-based statistics
    stats_rows = []
    
    for _, row in results_df.iterrows():
        stats_rows.append({
            'ConditionID': int(row['condition_id']),
            'Replication': int(row['replication']),
            'TotalTurns': row.get('total_turns', np.nan),
            'TotalReorientations': row.get('total_turns', np.nan),  # Same as turns (reorientations used)
            'TotalPauses': row.get('total_pauses', np.nan),
            'TotalReversals': row.get('total_reversals', np.nan),
            'MeanTurnLatency': row.get('latency', np.nan),
            'MeanPauseLatency': np.nan,  # Not computed
            'TotalSimulationTime': row.get('total_time', np.nan),
            'PathLength': row.get('path_length', np.nan),
            'EuclideanDistance': row.get('euclidean_distance', np.nan),
            'TurnRate': row.get('turn_rate', np.nan),
            'PauseRate': row.get('pause_rate', np.nan),
            'ReversalRate': row.get('reversal_rate', np.nan)
        })
    
    return pd.DataFrame(stats_rows)

def export_arena_csvs(results_df, output_dir, config):
    """Export all Arena-style CSV files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # AcrossReplicationsSummary
    summary_df = create_across_replications_summary(results_df, config)
    summary_path = output_dir / 'AcrossReplicationsSummary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"✓ Exported {summary_path}")
    
    # ContinuousTimeStatsByRep
    continuous_df = create_continuous_time_stats(results_df)
    continuous_path = output_dir / 'ContinuousTimeStatsByRep.csv'
    continuous_df.to_csv(continuous_path, index=False)
    print(f"✓ Exported {continuous_path}")
    
    # DiscreteTimeStatsByRep
    discrete_df = create_discrete_time_stats(results_df)
    discrete_path = output_dir / 'DiscreteTimeStatsByRep.csv'
    discrete_df.to_csv(discrete_path, index=False)
    print(f"✓ Exported {discrete_path}")
    
    return summary_df, continuous_df, discrete_df

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Export simulation results to Arena CSV format')
    parser.add_argument('--results', type=str, required=True,
                       help='Path to simulation results CSV')
    parser.add_argument('--output-dir', type=str, default='output/arena_csvs',
                       help='Output directory for CSV files')
    parser.add_argument('--config', type=str, default='config/model_config.json',
                       help='Path to model config JSON')
    
    args = parser.parse_args()
    
    # Load data and config
    results_df = pd.read_csv(args.results)
    config = load_config(args.config)
    
    # Export
    export_arena_csvs(results_df, args.output_dir, config)
    print(f"\n✓ All Arena CSV files exported to {args.output_dir}")

