#!/usr/bin/env python3
"""
Complete analysis pipeline for DOE results.

After running the DOE simulation, this script:
1. Exports results to Arena format
2. Analyzes main effects and interactions
3. Generates visualizations
4. Creates summary tables for report

Usage:
    python3 scripts/analyze_doe_results.py \
        --results-dir output/doe_results \
        --output-dir output/analysis
"""

import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path
import sys

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))
from export_arena_format import create_across_replications_summary, load_config
from analyze_main_effects import load_summary, compute_main_effects, compute_anova
from generate_report_tables import load_doe_table, format_doe_table_for_report, format_summary_table_for_report

def convert_to_json_serializable(obj):
    """Convert numpy types and other non-serializable types to JSON-compatible types."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {str(k): convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    elif pd.isna(obj):
        return None
    else:
        return obj

def run_analysis_pipeline(results_dir, output_dir, config_path='config/model_config.json'):
    """
    Run complete analysis pipeline on DOE results.
    
    Parameters
    ----------
    results_dir : str
        Directory containing DOE results (all_results.csv)
    output_dir : str
        Output directory for analysis results
    config_path : str
        Path to model configuration
    """
    results_path = Path(results_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (output_path / 'arena_csvs').mkdir(exist_ok=True)
    (output_path / 'figures').mkdir(exist_ok=True)
    (output_path / 'report_tables').mkdir(exist_ok=True)
    
    print("=" * 80)
    print("DOE RESULTS ANALYSIS PIPELINE")
    print("=" * 80)
    print()
    
    # Step 1: Load results
    print("Step 1: Loading DOE results...")
    results_file = results_path / "all_results.csv"
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    results_df = pd.read_csv(results_file)
    print(f"  ✓ Loaded {len(results_df)} simulation results")
    print(f"  ✓ {results_df['condition_id'].nunique()} conditions")
    print(f"  ✓ {results_df['replication'].nunique()} replications per condition")
    print()
    
    # Step 2: Export to Arena format
    print("Step 2: Exporting to Arena format...")
    config = load_config(config_path)
    summary_df = create_across_replications_summary(results_df, config)
    
    arena_dir = output_path / 'arena_csvs'
    summary_file = arena_dir / "AcrossReplicationsSummary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"  ✓ Saved AcrossReplicationsSummary.csv ({len(summary_df)} conditions)")
    print()
    
    # Step 3: Analyze main effects
    print("Step 3: Analyzing main effects and interactions...")
    # Drop ApproachRate if present (not applicable to omnidirectional stimulus)
    if 'ApproachRate_Mean' in summary_df.columns:
        summary_df = summary_df.drop(columns=[c for c in summary_df.columns if 'ApproachRate' in c])
    
    # All KPIs use _Mean suffix in the exported CSV
    kpis = ['TurnRate_Mean', 'Latency_Mean', 'StopFraction_Mean', 'PauseRate_Mean', 
            'ReversalRate_Mean', 'Tortuosity_Mean', 'Dispersal_Mean', 'MeanSpineCurveEnergy_Mean']
    # Note: MeanSpineCurveEnergy_Mean is included but may not be analyzed in main effects
    # (it's a spatial metric, not a behavioral rate)
    available_kpis = [k for k in kpis if k in summary_df.columns]
    
    main_effects_results = {}
    anova_results = {}
    
    for kpi in available_kpis:
        print(f"  Analyzing {kpi}...")
        main_effects = compute_main_effects(summary_df, kpi_col=kpi)
        anova = compute_anova(summary_df, kpi_col=kpi)
        
        main_effects_results[kpi] = main_effects
        anova_results[kpi] = anova
    
    # Save analysis results
    analysis_file = output_path / "main_effects_analysis.json"
    with open(analysis_file, 'w') as f:
        json.dump({
            'main_effects': convert_to_json_serializable(main_effects_results),
            'anova': convert_to_json_serializable(anova_results)
        }, f, indent=2)
    print(f"  ✓ Saved analysis results to {analysis_file}")
    print()
    
    # Step 4: Generate visualizations
    print("Step 4: Generating visualizations...")
    try:
        from analyze_main_effects import plot_main_effects, plot_interaction_effects
        
        fig_dir = output_path / 'figures' / 'doe_analysis'
        fig_dir.mkdir(parents=True, exist_ok=True)
        
        for kpi in available_kpis[:3]:  # Limit to first 3 KPIs for now
            print(f"  Plotting {kpi}...")
            kpi_name = kpi.replace('_Mean', '')
            plot_main_effects(summary_df, kpi_col=kpi, 
                            output_path=str(fig_dir / f'main_effects_{kpi_name}.png'))
            plot_interaction_effects(summary_df, kpi_col=kpi,
                            output_path=str(fig_dir / f'interactions_{kpi_name}.png'))
        
        print(f"  ✓ Saved figures to {fig_dir}")
    except Exception as e:
        print(f"  ⚠ Visualization generation failed: {e}")
        import traceback
        traceback.print_exc()
        print("  (You can run analyze_main_effects.py separately)")
    print()
    
    # Step 5: Generate report tables
    print("Step 5: Generating report tables...")
    report_tables_dir = output_path / 'report_tables'
    
    # DOE table for report
    doe_df = load_doe_table('config/doe_table.csv')
    format_doe_table_for_report(doe_df, str(report_tables_dir / 'doe_table_report.csv'))
    print(f"  ✓ Saved DOE table for report")
    
    # Summary table (top N conditions)
    if 'TurnRate_Mean' in summary_df.columns:
        summary_file = arena_dir / "AcrossReplicationsSummary.csv"
        format_summary_table_for_report(str(summary_file), 
                                      str(report_tables_dir / 'summary_table_top10.csv'),
                                      top_n=10)
        print(f"  ✓ Saved summary table (top 10 conditions)")
    
    print()
    
    # Step 6: Print summary
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Results directory: {output_path}")
    print(f"  • Arena CSV: {arena_dir}")
    print(f"  • Figures: {output_path / 'figures'}")
    print(f"  • Report tables: {report_tables_dir}")
    print(f"  • Analysis JSON: {analysis_file}")
    print()
    print("Next steps:")
    print("  1. Review AcrossReplicationsSummary.csv")
    print("  2. Check main effects plots in figures/doe_analysis/")
    print("  3. Use report tables for Quarto report")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Analyze DOE simulation results')
    parser.add_argument('--results-dir', type=str, required=True,
                       help='Directory containing DOE results (all_results.csv)')
    parser.add_argument('--output-dir', type=str, default='output/analysis',
                       help='Output directory for analysis results')
    parser.add_argument('--config', type=str, default='config/model_config.json',
                       help='Path to model configuration')
    
    args = parser.parse_args()
    
    run_analysis_pipeline(args.results_dir, args.output_dir, args.config)


if __name__ == '__main__':
    main()

