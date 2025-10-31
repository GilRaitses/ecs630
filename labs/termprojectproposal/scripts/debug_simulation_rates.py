#!/usr/bin/env python3
"""
Debug script to investigate simulation rate mismatch.

Compares:
1. Model-predicted rates vs baseline rates
2. Hazard rates computed during simulation
3. Event sampling probabilities
4. Timestep mismatch (bin_width vs dt)
"""

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from simulate_trajectories import load_config, load_event_models, compute_hazard_rate, sample_speed_from_params, compute_speed_distribution_params

def test_hazard_rate_computation():
    """Test hazard rate computation with baseline features."""
    print("="*80)
    print("TESTING HAZARD RATE COMPUTATION")
    print("="*80)
    
    # Load config
    config = load_config('config/model_config.json')
    bin_width = config['model'].get('bin_width', 0.05)
    dt = 0.1  # Simulation timestep
    
    print(f"\nConfiguration:")
    print(f"  Training bin_width: {bin_width}s (50ms)")
    print(f"  Simulation dt: {dt}s (100ms)")
    print(f"  Ratio: {dt/bin_width:.1f}x")
    
    # Load models
    models_dict = load_event_models('output/fitted_models')
    
    # Test with baseline features (no stimulus, average speed, etc.)
    features_baseline = {
        'kernel_features': np.zeros(10),  # No stimulus
        'n_basis': 10,
        'speed_normalized': 0.0,  # Average speed
        'heading_sin': 0.0,
        'heading_cos': 1.0,
        'spine_curve_energy': 0.0,
        'spine_energy_mean': 0.0,
        'spine_energy_std': 1.0
    }
    
    print(f"\nBaseline Features (no stimulus, average conditions):")
    print(f"  kernel_features: all zeros")
    print(f"  speed_normalized: 0.0")
    print(f"  heading: 0.0")
    print(f"  spine_curve_energy: 0.0")
    
    # Compute hazard rates
    print(f"\nComputed Hazard Rates:")
    hazard_rates = {}
    for event_type in ['turn', 'pause', 'reversal']:
        if event_type in models_dict and models_dict[event_type] is not None:
            hazard_rate = compute_hazard_rate(models_dict[event_type], features_baseline, bin_width)
            hazard_rates[event_type] = hazard_rate
            
            # Convert to rate per minute
            rate_per_min = hazard_rate * 60
            
            # Event probability per timestep
            event_prob_per_timestep = hazard_rate * dt
            
            print(f"\n{event_type.upper()}:")
            print(f"  Hazard rate (λ): {hazard_rate:.6f} events/sec")
            print(f"  Rate per minute: {rate_per_min:.2f} events/min")
            print(f"  Event prob per timestep (dt={dt}s): {event_prob_per_timestep:.6f}")
            print(f"  Expected events in 60s: {hazard_rate * 60:.1f}")
        else:
            hazard_rates[event_type] = 0.0
    
    # Compare to baseline model rates
    print(f"\n" + "="*80)
    print("COMPARISON TO BASELINE MODEL RATES")
    print("="*80)
    
    for event_type in ['turn', 'pause', 'reversal']:
        baseline_path = Path(f'output/fitted_models/{event_type}_baseline_model.pkl')
        if baseline_path.exists():
            with open(baseline_path, 'rb') as f:
                baseline_model = pickle.load(f)
            baseline_rate = baseline_model.get('rate_per_min', 0)
            baseline_lambda = baseline_model.get('lambda_hat', 0)
            
            computed_rate = hazard_rates[event_type] * 60
            
            print(f"\n{event_type.upper()}:")
            print(f"  Baseline model rate: {baseline_rate:.2f} events/min")
            print(f"  Computed rate (baseline features): {computed_rate:.2f} events/min")
            print(f"  Difference: {abs(computed_rate - baseline_rate):.2f} events/min")
            print(f"  Ratio: {computed_rate / baseline_rate:.2f}x")
    
    return hazard_rates

def test_event_sampling():
    """Test event sampling logic."""
    print("\n" + "="*80)
    print("TESTING EVENT SAMPLING LOGIC")
    print("="*80)
    
    config = load_config('config/model_config.json')
    bin_width = config['model'].get('bin_width', 0.05)
    dt = 0.1
    
    # Simulate 100 timesteps (10 seconds)
    n_timesteps = 100
    hazard_rate = 0.1  # Example: 0.1 events/sec = 6 events/min
    
    print(f"\nSimulation parameters:")
    print(f"  Hazard rate: {hazard_rate} events/sec")
    print(f"  Timestep (dt): {dt}s")
    print(f"  Number of timesteps: {n_timesteps} ({n_timesteps * dt}s)")
    
    # Sample events
    np.random.seed(42)
    events = []
    for i in range(n_timesteps):
        event_prob = hazard_rate * dt
        if np.random.random() < event_prob:
            events.append(i * dt)
    
    n_events = len(events)
    expected_events = hazard_rate * n_timesteps * dt
    actual_rate = n_events / (n_timesteps * dt) * 60  # events per minute
    
    print(f"\nSampling results:")
    print(f"  Expected events: {expected_events:.1f}")
    print(f"  Actual events: {n_events}")
    print(f"  Actual rate: {actual_rate:.2f} events/min")
    print(f"  Target rate: {hazard_rate * 60:.2f} events/min")
    
    # Check if multiple events can occur per timestep
    print(f"\nEvent timing:")
    if len(events) > 0:
        print(f"  First event: {events[0]:.2f}s")
        print(f"  Last event: {events[-1]:.2f}s")
        # Check for events in same timestep (shouldn't happen)
        event_timesteps = [int(e / dt) for e in events]
        unique_timesteps = len(set(event_timesteps))
        if unique_timesteps < len(events):
            print(f"  WARNING: {len(events) - unique_timesteps} duplicate timesteps (multiple events per dt)")

def analyze_simulation_output():
    """Analyze actual simulation output to find rate issues."""
    print("\n" + "="*80)
    print("ANALYZING SIMULATION OUTPUT")
    print("="*80)
    
    results_path = Path('output/validation/simulated_kpis.csv')
    if not results_path.exists():
        print("  No simulation results found. Run validation first.")
        return
    
    sim_df = pd.read_csv(results_path)
    
    print(f"\nSimulation Results ({len(sim_df)} replications):")
    print(f"  Mean turn rate: {sim_df['turn_rate'].mean():.2f} turns/min")
    print(f"  Mean pause rate: {sim_df['pause_rate'].mean():.2f} pauses/min")
    print(f"  Mean reversal rate: {sim_df['reversal_rate'].mean():.2f} reversals/min")
    
    print(f"\nTotal Events:")
    print(f"  Total turns: {sim_df['total_turns'].sum()}")
    print(f"  Total pauses: {sim_df['total_pauses'].sum()}")
    print(f"  Total reversals: {sim_df['total_reversals'].sum()}")
    
    # Check if multiple events per timestep might be occurring
    # If hazard rates are high, multiple events could fire in same dt
    print(f"\nEvent Multiplicity Check:")
    print(f"  If hazard_rate * dt > 1.0, multiple events per timestep possible")
    
    # Estimate average hazard rates from event counts
    total_time = sim_df['total_time'].mean()
    avg_turn_rate = sim_df['turn_rate'].mean()
    avg_pause_rate = sim_df['pause_rate'].mean()
    
    avg_turn_hazard = avg_turn_rate / 60  # events/sec
    avg_pause_hazard = avg_pause_rate / 60
    
    dt = 0.1
    turn_prob_per_dt = avg_turn_hazard * dt
    pause_prob_per_dt = avg_pause_hazard * dt
    
    print(f"  Estimated turn hazard: {avg_turn_hazard:.4f} events/sec")
    print(f"  Turn prob per dt (0.1s): {turn_prob_per_dt:.4f}")
    print(f"  Estimated pause hazard: {avg_pause_hazard:.4f} events/sec")
    print(f"  Pause prob per dt (0.1s): {pause_prob_per_dt:.4f}")
    
    if turn_prob_per_dt > 1.0 or pause_prob_per_dt > 1.0:
        print(f"\n  WARNING: Event probability > 1.0! This means multiple events per timestep.")
        print(f"  Should use Poisson sampling instead of Bernoulli.")

if __name__ == '__main__':
    hazard_rates = test_hazard_rate_computation()
    test_event_sampling()
    analyze_simulation_output()



