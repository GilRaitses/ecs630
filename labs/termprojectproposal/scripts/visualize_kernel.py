#!/usr/bin/env python3
"""
Visualize fitted temporal kernel.

Creates plots showing:
1. Kernel basis functions
2. Fitted kernel (weighted sum of bases)
3. Predicted hazard over time relative to stimulus
"""

import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def load_model_summary(summary_path):
    """Load model summary JSON."""
    with open(summary_path, 'r') as f:
        return json.load(f)

def create_kernel_basis(times, n_basis=10, time_range=(-2.0, 20.0)):
    """Create raised cosine basis functions."""
    times = np.array(times)
    t_min, t_max = time_range
    knot_positions = np.linspace(t_min, t_max, n_basis)
    delta_tau = (t_max - t_min) / (n_basis - 1)
    
    basis_matrix = np.zeros((len(times), n_basis))
    
    for j, tau_j in enumerate(knot_positions):
        tau_diff = times - tau_j
        mask = np.abs(tau_diff) < delta_tau
        
        if np.any(mask):
            cos_arg = np.pi * tau_diff[mask] / (2 * delta_tau)
            basis_matrix[mask, j] = np.cos(cos_arg) ** 2
    
    return basis_matrix, knot_positions

def plot_kernel(summary_path, output_path=None):
    """Plot the fitted temporal kernel."""
    summary = load_model_summary(summary_path)
    
    if 'kernel_coefficients' not in summary:
        print("No kernel coefficients found in summary")
        return
    
    # Get kernel coefficients
    kernel_coefs = {}
    for name, coef in summary['kernel_coefficients'].items():
        if name.startswith('kernel_'):
            idx = int(name.split('_')[1])
            kernel_coefs[idx] = coef
    
    n_basis = len(kernel_coefs)
    time_range = (-2.0, 20.0)
    times = np.arange(time_range[0], time_range[1], 0.05)
    
    # Create basis functions
    basis_matrix, knot_positions = create_kernel_basis(times, n_basis, time_range)
    
    # Compute fitted kernel
    fitted_kernel = np.zeros(len(times))
    for idx, coef in kernel_coefs.items():
        fitted_kernel += coef * basis_matrix[:, idx]
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot 1: Basis functions
    ax1 = axes[0]
    for idx in range(n_basis):
        ax1.plot(times, basis_matrix[:, idx], alpha=0.3, label=f'Basis {idx}')
    ax1.set_xlabel('Time since stimulus (seconds)')
    ax1.set_ylabel('Basis function value')
    ax1.set_title('Temporal Kernel Basis Functions')
    ax1.axvline(0, color='r', linestyle='--', alpha=0.5, label='Stimulus onset')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Fitted kernel
    ax2 = axes[1]
    ax2.plot(times, fitted_kernel, 'b-', linewidth=2, label='Fitted kernel')
    ax2.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax2.axvline(0, color='r', linestyle='--', alpha=0.5, label='Stimulus onset')
    ax2.set_xlabel('Time since stimulus (seconds)')
    ax2.set_ylabel('Kernel coefficient')
    ax2.set_title('Fitted Temporal Kernel (Weighted Sum of Bases)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved kernel plot to {output_path}")
    else:
        plt.show()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize fitted temporal kernel')
    parser.add_argument('--summary', type=str, 
                       default='output/fitted_models/turn_full_summary.json',
                       help='Path to model summary JSON')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for plot (PNG)')
    
    args = parser.parse_args()
    
    plot_kernel(args.summary, args.output)

