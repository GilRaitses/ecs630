#!/usr/bin/env python3
"""
Analyze main effects and interactions from DOE results.

Performs statistical analysis to determine which factors (intensity, pulse duration,
inter-pulse interval) significantly affect behavioral KPIs.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from scipy import stats
from scipy.stats import f_oneway
import matplotlib.pyplot as plt
import seaborn as sns
from cinnamoroll_palette import CINNAMOROLL_COLORS, CINNAMOROLL_PALETTE, setup_cinnamoroll_style

def load_summary(summary_path):
    """Load AcrossReplicationsSummary CSV."""
    return pd.read_csv(summary_path)

def compute_main_effects(summary_df, kpi_col='TurnRate_Mean'):
    """
    Compute main effects for each factor level.
    
    Parameters
    ----------
    summary_df : DataFrame
        AcrossReplicationsSummary DataFrame
    kpi_col : str
        Column name for KPI to analyze
    
    Returns
    -------
    main_effects : dict
        Dictionary with main effects for each factor
    """
    main_effects = {}
    
    # Intensity main effects
    intensity_levels = sorted(summary_df['Intensity'].unique())
    intensity_effects = {}
    for level in intensity_levels:
        level_data = summary_df[summary_df['Intensity'] == level][kpi_col].values
        intensity_effects[level] = {
            'mean': np.mean(level_data),
            'std': np.std(level_data),
            'n': len(level_data)
        }
    main_effects['Intensity'] = intensity_effects
    
    # Pulse duration main effects
    pulse_levels = sorted(summary_df['PulseDuration'].unique())
    pulse_effects = {}
    for level in pulse_levels:
        level_data = summary_df[summary_df['PulseDuration'] == level][kpi_col].values
        pulse_effects[level] = {
            'mean': np.mean(level_data),
            'std': np.std(level_data),
            'n': len(level_data)
        }
    main_effects['PulseDuration'] = pulse_effects
    
    # Inter-pulse interval main effects
    interval_levels = sorted(summary_df['InterPulseInterval'].unique())
    interval_effects = {}
    for level in interval_levels:
        level_data = summary_df[summary_df['InterPulseInterval'] == level][kpi_col].values
        interval_effects[level] = {
            'mean': np.mean(level_data),
            'std': np.std(level_data),
            'n': len(level_data)
        }
    main_effects['InterPulseInterval'] = interval_effects
    
    return main_effects

def compute_anova(summary_df, kpi_col='TurnRate_Mean'):
    """
    Perform ANOVA to test for significant main effects and interactions.
    
    Parameters
    ----------
    summary_df : DataFrame
        AcrossReplicationsSummary DataFrame
    kpi_col : str
        Column name for KPI to analyze
    
    Returns
    -------
    anova_results : dict
        ANOVA F-statistics and p-values
    """
    # Two-way ANOVA for each pair of factors
    results = {}
    
    # Intensity × PulseDuration
    intensity_levels = sorted(summary_df['Intensity'].unique())
    pulse_levels = sorted(summary_df['PulseDuration'].unique())
    groups = []
    for intensity in intensity_levels:
        for pulse in pulse_levels:
            group_data = summary_df[
                (summary_df['Intensity'] == intensity) & 
                (summary_df['PulseDuration'] == pulse)
            ][kpi_col].values
            if len(group_data) > 0:
                groups.append(group_data)
    
    if len(groups) >= 2:
        f_stat, p_value = f_oneway(*groups)
        results['Intensity_x_PulseDuration'] = {
            'F_statistic': f_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    # Intensity × InterPulseInterval
    interval_levels = sorted(summary_df['InterPulseInterval'].unique())
    groups = []
    for intensity in intensity_levels:
        for interval in interval_levels:
            group_data = summary_df[
                (summary_df['Intensity'] == intensity) & 
                (summary_df['InterPulseInterval'] == interval)
            ][kpi_col].values
            if len(group_data) > 0:
                groups.append(group_data)
    
    if len(groups) >= 2:
        f_stat, p_value = f_oneway(*groups)
        results['Intensity_x_InterPulseInterval'] = {
            'F_statistic': f_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    # PulseDuration × InterPulseInterval
    groups = []
    for pulse in pulse_levels:
        for interval in interval_levels:
            group_data = summary_df[
                (summary_df['PulseDuration'] == pulse) & 
                (summary_df['InterPulseInterval'] == interval)
            ][kpi_col].values
            if len(group_data) > 0:
                groups.append(group_data)
    
    if len(groups) >= 2:
        f_stat, p_value = f_oneway(*groups)
        results['PulseDuration_x_InterPulseInterval'] = {
            'F_statistic': f_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    return results

def plot_main_effects(summary_df, kpi_col='TurnRate_Mean', output_path=None):
    """
    Plot main effects for each factor.
    
    Parameters
    ----------
    summary_df : DataFrame
        AcrossReplicationsSummary DataFrame
    kpi_col : str
        Column name for KPI to plot
    output_path : str, optional
        Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Intensity main effects
    intensity_levels = sorted(summary_df['Intensity'].unique())
    intensity_means = []
    intensity_stds = []
    for level in intensity_levels:
        level_data = summary_df[summary_df['Intensity'] == level][kpi_col].values
        intensity_means.append(np.mean(level_data))
        intensity_stds.append(np.std(level_data))
    
    axes[0].errorbar(intensity_levels, intensity_means, yerr=intensity_stds,
                    marker='o', capsize=5, capthick=2, 
                    color=CINNAMOROLL_COLORS['blue'], 
                    markerfacecolor=CINNAMOROLL_COLORS['light_blue'],
                    markeredgecolor=CINNAMOROLL_COLORS['dark_blue'],
                    ecolor=CINNAMOROLL_COLORS['lavender'], linewidth=2)
    axes[0].set_xlabel('Intensity (%)', color=CINNAMOROLL_COLORS['dark_blue'])
    axes[0].set_ylabel(kpi_col.replace('_', ' '), color=CINNAMOROLL_COLORS['dark_blue'])
    axes[0].set_title('Intensity Main Effect', color=CINNAMOROLL_COLORS['dark_blue'])
    axes[0].grid(True, alpha=0.3, color=CINNAMOROLL_COLORS['tan'])
    axes[0].set_facecolor('white')
    
    # Pulse duration main effects
    pulse_levels = sorted(summary_df['PulseDuration'].unique())
    pulse_means = []
    pulse_stds = []
    for level in pulse_levels:
        level_data = summary_df[summary_df['PulseDuration'] == level][kpi_col].values
        pulse_means.append(np.mean(level_data))
        pulse_stds.append(np.std(level_data))
    
    axes[1].errorbar(pulse_levels, pulse_means, yerr=pulse_stds,
                    marker='o', capsize=5, capthick=2,
                    color=CINNAMOROLL_COLORS['blue'],
                    markerfacecolor=CINNAMOROLL_COLORS['light_blue'],
                    markeredgecolor=CINNAMOROLL_COLORS['dark_blue'],
                    ecolor=CINNAMOROLL_COLORS['lavender'], linewidth=2)
    axes[1].set_xlabel('Pulse Duration (s)', color=CINNAMOROLL_COLORS['dark_blue'])
    axes[1].set_ylabel(kpi_col.replace('_', ' '), color=CINNAMOROLL_COLORS['dark_blue'])
    axes[1].set_title('Pulse Duration Main Effect', color=CINNAMOROLL_COLORS['dark_blue'])
    axes[1].grid(True, alpha=0.3, color=CINNAMOROLL_COLORS['tan'])
    axes[1].set_facecolor('white')
    
    # Inter-pulse interval main effects
    interval_levels = sorted(summary_df['InterPulseInterval'].unique())
    interval_means = []
    interval_stds = []
    for level in interval_levels:
        level_data = summary_df[summary_df['InterPulseInterval'] == level][kpi_col].values
        interval_means.append(np.mean(level_data))
        interval_stds.append(np.std(level_data))
    
    axes[2].errorbar(interval_levels, interval_means, yerr=interval_stds,
                    marker='o', capsize=5, capthick=2,
                    color=CINNAMOROLL_COLORS['blue'],
                    markerfacecolor=CINNAMOROLL_COLORS['light_blue'],
                    markeredgecolor=CINNAMOROLL_COLORS['dark_blue'],
                    ecolor=CINNAMOROLL_COLORS['lavender'], linewidth=2)
    axes[2].set_xlabel('Inter-Pulse Interval (s)', color=CINNAMOROLL_COLORS['dark_blue'])
    axes[2].set_ylabel(kpi_col.replace('_', ' '), color=CINNAMOROLL_COLORS['dark_blue'])
    axes[2].set_title('Inter-Pulse Interval Main Effect', color=CINNAMOROLL_COLORS['dark_blue'])
    axes[2].grid(True, alpha=0.3, color=CINNAMOROLL_COLORS['tan'])
    axes[2].set_facecolor('white')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved main effects plot to {output_path}")
    else:
        plt.show()
    
    plt.close()

def plot_interaction_effects(summary_df, kpi_col='TurnRate_Mean', output_path=None):
    """
    Plot interaction effects between factors.
    
    Parameters
    ----------
    summary_df : DataFrame
        AcrossReplicationsSummary DataFrame
    kpi_col : str
        Column name for KPI to plot
    output_path : str, optional
        Path to save figure
    """
    plt = setup_cinnamoroll_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor(CINNAMOROLL_COLORS['cream'])
    
    # Intensity × PulseDuration interaction
    intensity_levels = sorted(summary_df['Intensity'].unique())
    pulse_levels = sorted(summary_df['PulseDuration'].unique())
    x = np.arange(len(intensity_levels))
    width = 0.25
    
    for i, pulse in enumerate(pulse_levels):
        means = []
        for intensity in intensity_levels:
            level_data = summary_df[
                (summary_df['Intensity'] == intensity) & 
                (summary_df['PulseDuration'] == pulse)
            ][kpi_col].values
            means.append(np.mean(level_data) if len(level_data) > 0 else 0)
        axes[0].bar(x + i*width, means, width, label=f'Pulse={pulse}s',
                   color=CINNAMOROLL_PALETTE[i % len(CINNAMOROLL_PALETTE)],
                   edgecolor=CINNAMOROLL_COLORS['dark_blue'], linewidth=1.5)
    
    axes[0].set_xlabel('Intensity (%)', color=CINNAMOROLL_COLORS['dark_blue'])
    axes[0].set_ylabel(kpi_col.replace('_', ' '), color=CINNAMOROLL_COLORS['dark_blue'])
    axes[0].set_title('Intensity × Pulse Duration Interaction', color=CINNAMOROLL_COLORS['dark_blue'])
    axes[0].set_xticks(x + width)
    axes[0].set_xticklabels(intensity_levels)
    axes[0].legend(framealpha=0.9, facecolor='white', edgecolor=CINNAMOROLL_COLORS['light_blue'])
    axes[0].grid(True, alpha=0.3, axis='y', color=CINNAMOROLL_COLORS['tan'])
    axes[0].set_facecolor('white')
    
    # Intensity × InterPulseInterval interaction
    interval_levels = sorted(summary_df['InterPulseInterval'].unique())
    for i, interval in enumerate(interval_levels):
        means = []
        for intensity in intensity_levels:
            level_data = summary_df[
                (summary_df['Intensity'] == intensity) & 
                (summary_df['InterPulseInterval'] == interval)
            ][kpi_col].values
            means.append(np.mean(level_data) if len(level_data) > 0 else 0)
        axes[1].bar(x + i*width, means, width, label=f'Interval={interval}s',
                   color=CINNAMOROLL_PALETTE[i % len(CINNAMOROLL_PALETTE)],
                   edgecolor=CINNAMOROLL_COLORS['dark_blue'], linewidth=1.5)
    
    axes[1].set_xlabel('Intensity (%)', color=CINNAMOROLL_COLORS['dark_blue'])
    axes[1].set_ylabel(kpi_col.replace('_', ' '), color=CINNAMOROLL_COLORS['dark_blue'])
    axes[1].set_title('Intensity × Inter-Pulse Interval Interaction', color=CINNAMOROLL_COLORS['dark_blue'])
    axes[1].set_xticks(x + width)
    axes[1].set_xticklabels(intensity_levels)
    axes[1].legend(framealpha=0.9, facecolor='white', edgecolor=CINNAMOROLL_COLORS['light_blue'])
    axes[1].grid(True, alpha=0.3, axis='y', color=CINNAMOROLL_COLORS['tan'])
    axes[1].set_facecolor('white')
    
    # PulseDuration × InterPulseInterval interaction
    pulse_x = np.arange(len(pulse_levels))
    for i, interval in enumerate(interval_levels):
        means = []
        for pulse in pulse_levels:
            level_data = summary_df[
                (summary_df['PulseDuration'] == pulse) & 
                (summary_df['InterPulseInterval'] == interval)
            ][kpi_col].values
            means.append(np.mean(level_data) if len(level_data) > 0 else 0)
        axes[2].bar(pulse_x + i*width, means, width, label=f'Interval={interval}s',
                   color=CINNAMOROLL_PALETTE[i % len(CINNAMOROLL_PALETTE)],
                   edgecolor=CINNAMOROLL_COLORS['dark_blue'], linewidth=1.5)
    
    axes[2].set_xlabel('Pulse Duration (s)', color=CINNAMOROLL_COLORS['dark_blue'])
    axes[2].set_ylabel(kpi_col.replace('_', ' '), color=CINNAMOROLL_COLORS['dark_blue'])
    axes[2].set_title('Pulse Duration × Inter-Pulse Interval Interaction', color=CINNAMOROLL_COLORS['dark_blue'])
    axes[2].set_xticks(pulse_x + width)
    axes[2].set_xticklabels(pulse_levels)
    axes[2].legend(framealpha=0.9, facecolor='white', edgecolor=CINNAMOROLL_COLORS['light_blue'])
    axes[2].grid(True, alpha=0.3, axis='y', color=CINNAMOROLL_COLORS['tan'])
    axes[2].set_facecolor('white')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved interaction effects plot to {output_path}")
    else:
        plt.show()
    
    plt.close()

def analyze_all_kpis(summary_path, output_dir):
    """
    Analyze main effects and interactions for all KPIs.
    
    Parameters
    ----------
    summary_path : str
        Path to AcrossReplicationsSummary.csv
    output_dir : str
        Output directory for results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary_df = load_summary(summary_path)
    
    # Define KPIs to analyze
    kpi_mapping = {
        'TurnRate_Mean': 'Turn Rate',
        'Latency_Mean': 'Latency',
        'StopFraction_Mean': 'Stop Fraction',
        'PauseRate_Mean': 'Pause Rate',
        'ReversalRate_Mean': 'Reversal Rate',
        'Tortuosity_Mean': 'Tortuosity',
        'Dispersal_Mean': 'Dispersal',
        'MeanSpineCurveEnergy_Mean': 'Mean Spine Curve Energy'
    }
    
    all_results = {}
    
    for kpi_col, kpi_name in kpi_mapping.items():
        if kpi_col not in summary_df.columns:
            continue
        
        print(f"\nAnalyzing {kpi_name}...")
        
        # Compute main effects
        main_effects = compute_main_effects(summary_df, kpi_col)
        
        # Compute ANOVA
        anova_results = compute_anova(summary_df, kpi_col)
        
        # Plot main effects
        plot_main_effects(summary_df, kpi_col, 
                         output_dir / f'main_effects_{kpi_name.replace(" ", "_")}.png')
        
        # Plot interactions
        plot_interaction_effects(summary_df, kpi_col,
                                output_dir / f'interaction_effects_{kpi_name.replace(" ", "_")}.png')
        
        all_results[kpi_name] = {
            'main_effects': main_effects,
            'anova': anova_results
        }
    
    # Save results to JSON
    results_path = output_dir / 'main_effects_analysis.json'
    
    # Convert to JSON-serializable format
    json_results = {}
    for kpi_name, results in all_results.items():
        json_results[kpi_name] = {
            'main_effects': {k: {kk: float(vv) if isinstance(vv, (np.float64, np.float32)) else vv 
                               for kk, vv in v.items()} 
                           for k, v in results['main_effects'].items()},
            'anova': {k: {kk: float(vv) if isinstance(vv, (np.float64, np.float32)) else vv 
                        for kk, vv in v.items()} 
                    for k, v in results['anova'].items()}
        }
    
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\n✓ Analysis complete! Results saved to {results_path}")
    print(f"✓ Plots saved to {output_dir}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze main effects and interactions from DOE results')
    parser.add_argument('--summary', type=str, required=True,
                       help='Path to AcrossReplicationsSummary.csv')
    parser.add_argument('--output-dir', type=str, default='output/analysis',
                       help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    analyze_all_kpis(args.summary, args.output_dir)



