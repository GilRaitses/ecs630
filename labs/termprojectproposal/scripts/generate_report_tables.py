#!/usr/bin/env python3
"""
Generate tables for Quarto report.

Creates:
1. DOE table (formatted)
2. AcrossReplicationsSummary table (formatted subset)
3. Model coefficients tables (for each event type)
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List

def load_doe_table(doe_path: str) -> pd.DataFrame:
    """Load and format DOE table."""
    df = pd.read_csv(doe_path, quotechar='"')
    return df

def format_doe_table_for_report(doe_df: pd.DataFrame, output_path: str):
    """Format DOE table for report (select key columns)."""
    # Select relevant columns
    report_df = doe_df[['condition_id', 'intensity_pct', 'pulse_duration_s', 
                        'inter_pulse_interval_s', 'description']].copy()
    
    # Rename for readability
    report_df.columns = ['Condition ID', 'Intensity (%)', 'Pulse Duration (s)',
                         'Inter-Pulse Interval (s)', 'Description']
    
    # Round numeric columns
    report_df['Intensity (%)'] = report_df['Intensity (%)'].astype(int)
    report_df['Pulse Duration (s)'] = report_df['Pulse Duration (s)'].round(1)
    report_df['Inter-Pulse Interval (s)'] = report_df['Inter-Pulse Interval (s)'].astype(int)
    
    # Save as CSV (can be converted to LaTeX/HTML in Quarto)
    report_df.to_csv(output_path, index=False)
    print(f"✓ Saved DOE table to {output_path}")
    
    return report_df

def format_summary_table_for_report(summary_path: str, output_path: str, 
                                    top_n: int = 10):
    """Format AcrossReplicationsSummary for report (top N conditions)."""
    df = pd.read_csv(summary_path)
    
    # Select key columns for report
    cols = ['ConditionID', 'Intensity', 'PulseDuration', 'InterPulseInterval',
            'TurnRate_Mean', 'TurnRate_CILower', 'TurnRate_CIUpper',
            'Latency_Mean', 'StopFraction_Mean', 'NReplications']
    
    # Filter to available columns
    available_cols = [c for c in cols if c in df.columns]
    report_df = df[available_cols].copy()
    
    # Sort by Turn Rate Mean BEFORE formatting (use numeric column)
    if 'TurnRate_Mean' in report_df.columns:
        report_df = report_df.sort_values('TurnRate_Mean', ascending=False)
    
    # Format column names
    report_df.columns = [c.replace('_', ' ') for c in report_df.columns]
    
    # Round numeric columns
    numeric_cols = report_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if 'Mean' in col or 'CI' in col:
            report_df[col] = report_df[col].round(3)
        elif 'Intensity' in col or 'Duration' in col or 'Interval' in col:
            report_df[col] = report_df[col].round(1)
    
    # Format CIs as strings: "mean [lower, upper]"
    if 'TurnRate Mean' in report_df.columns and 'TurnRate CILower' in report_df.columns:
        report_df['Turn Rate (turns/min)'] = (
            report_df['TurnRate Mean'].round(2).astype(str) + ' [' +
            report_df['TurnRate CILower'].round(2).astype(str) + ', ' +
            report_df['TurnRate CIUpper'].round(2).astype(str) + ']'
        )
        # Drop individual CI columns
        report_df = report_df.drop(columns=['TurnRate CILower', 'TurnRate CIUpper'])
        report_df = report_df.drop(columns=['TurnRate Mean'])
    
    # Select top N
    report_df = report_df.head(top_n)
    
    # Rename remaining columns
    report_df = report_df.rename(columns={
        'ConditionID': 'Condition',
        'Intensity': 'Intensity (%)',
        'PulseDuration': 'Pulse Duration (s)',
        'InterPulseInterval': 'Inter-Pulse Interval (s)',
        'Latency Mean': 'Latency (s)',
        'StopFraction Mean': 'Stop Fraction',
        'NReplications': 'N Reps'
    })
    
    # Reorder columns
    col_order = ['Condition', 'Intensity (%)', 'Pulse Duration (s)', 
                 'Inter-Pulse Interval (s)', 'Turn Rate (turns/min)',
                 'Latency (s)', 'Stop Fraction', 'N Reps']
    report_df = report_df[[c for c in col_order if c in report_df.columns]]
    
    report_df.to_csv(output_path, index=False)
    print(f"✓ Saved summary table (top {top_n}) to {output_path}")
    
    return report_df

def format_model_coefficients_table(coef_path: str, output_path: str,
                                    event_type: str = 'reorientation'):
    """Format model coefficients table for report."""
    df = pd.read_csv(coef_path)
    
    # Select and format key columns
    if 'coefficient' in df.columns and 'feature' in df.columns:
        # Check which columns are available
        available_cols = ['feature', 'coefficient']
        if 'std_err' in df.columns:
            available_cols.append('std_err')
        elif 'std_error' in df.columns:
            available_cols.append('std_error')
        report_df = df[available_cols].copy()
        
        # Filter to kernel coefficients (most important)
        kernel_coefs = report_df[report_df['feature'].str.startswith('kernel_', na=False)].copy()
        
        # Extract kernel index
        kernel_coefs['kernel_idx'] = kernel_coefs['feature'].str.extract(r'kernel_(\d+)').astype(int)
        kernel_coefs = kernel_coefs.sort_values('kernel_idx')
        
        # Format feature names
        kernel_coefs['Feature'] = kernel_coefs['feature'].str.replace('kernel_', 'Kernel Basis ')
        kernel_coefs['Coefficient'] = kernel_coefs['coefficient'].round(4)
        
        # Select formatted columns
        if 'std_err' in kernel_coefs.columns or 'std_error' in kernel_coefs.columns:
            std_col = 'std_err' if 'std_err' in kernel_coefs.columns else 'std_error'
            kernel_coefs['Std Error'] = kernel_coefs[std_col].round(4)
            report_df_formatted = kernel_coefs[['Feature', 'Coefficient', 'Std Error']].copy()
        else:
            report_df_formatted = kernel_coefs[['Feature', 'Coefficient']].copy()
        
        # Add other important features (intercept, speed, etc.)
        other_features = report_df[~report_df['feature'].str.startswith('kernel_', na=False)].copy()
        if len(other_features) > 0:
            other_features['Feature'] = other_features['feature'].str.replace('_', ' ').str.title()
            other_features['Coefficient'] = other_features['coefficient'].round(4)
            if 'std_err' in other_features.columns or 'std_error' in other_features.columns:
                std_col = 'std_err' if 'std_err' in other_features.columns else 'std_error'
                other_features['Std Error'] = other_features[std_col].round(4)
                other_formatted = other_features[['Feature', 'Coefficient', 'Std Error']].copy()
            else:
                other_formatted = other_features[['Feature', 'Coefficient']].copy()
            
            # Combine (intercept first, then other features, then kernel)
            report_df_formatted = pd.concat([
                other_formatted[other_formatted['Feature'].str.contains('Intercept', case=False)],
                other_formatted[~other_formatted['Feature'].str.contains('Intercept', case=False)],
                report_df_formatted
            ], ignore_index=True)
        
        report_df_formatted.to_csv(output_path, index=False)
        print(f"✓ Saved {event_type} model coefficients to {output_path}")
        
        return report_df_formatted
    else:
        print(f"Warning: Unexpected coefficient file format in {coef_path}")
        return None

def generate_all_tables(doe_table_path: str, summary_path: str,
                       models_dir: str, output_dir: str):
    """Generate all report tables."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating report tables...")
    print("="*60)
    
    # 1. DOE table
    doe_df = load_doe_table(doe_table_path)
    format_doe_table_for_report(doe_df, output_dir / 'doe_table_report.csv')
    
    # 2. Summary table (top 10 conditions)
    format_summary_table_for_report(summary_path, 
                                   output_dir / 'summary_table_top10.csv',
                                   top_n=10)
    
    # 3. Model coefficients tables
    event_types = ['reorientation', 'pause', 'reversal']
    for event_type in event_types:
        coef_path = Path(models_dir) / f'{event_type}_coefficients.csv'
        if coef_path.exists():
            format_model_coefficients_table(
                str(coef_path),
                output_dir / f'{event_type}_coefficients_report.csv',
                event_type
            )
        else:
            print(f"  Warning: {coef_path} not found, skipping")
    
    print("\n" + "="*60)
    print("✓ All report tables generated!")
    print(f"  Output directory: {output_dir}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate tables for Quarto report')
    parser.add_argument('--doe-table', type=str,
                       default='config/doe_table.csv',
                       help='Path to DOE table CSV')
    parser.add_argument('--summary', type=str,
                       default='output/doe_test3/arena_csvs/AcrossReplicationsSummary.csv',
                       help='Path to AcrossReplicationsSummary CSV')
    parser.add_argument('--models-dir', type=str,
                       default='output/fitted_models',
                       help='Directory containing model coefficient CSVs')
    parser.add_argument('--output-dir', type=str,
                       default='output/report_tables',
                       help='Output directory for report tables')
    
    args = parser.parse_args()
    
    generate_all_tables(args.doe_table, args.summary, args.models_dir, args.output_dir)

