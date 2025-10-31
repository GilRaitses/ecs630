# Model 4-1 Simulation - Python Implementation

This directory contains a Python simulation script to replicate Arena Model 4-1 and calculate resource utilization.

## Files

- `simulate_model_4_1.py` - Main simulation script
- `Model 04-01(1).doe` - Original Arena model file (for reference)
- `model_4_1_results.csv` - Output CSV file with resource utilizations (generated after running)

## Quick Start

1. **Configure the simulation** based on your Model 4-1 specifications (see Configuration section below)

2. **Run the simulation**:
   ```bash
   python3 simulate_model_4_1.py
   ```

3. **Review results**: The script will:
   - Display resource utilizations in the terminal
   - Identify the resource with highest utilization
   - Save results to `model_4_1_results.csv`

## Configuration Required

You need to configure the simulation based on your specific Model 4-1 specifications. Open `simulate_model_4_1.py` and modify:

### 1. Resources

In the `__init__` method, add/modify resources based on your Model 4-1:
```python
# Example: Two-resource serial system
self.resource1_capacity = 1
self.resource1_queue: deque = deque()
self.resource1_busy = False

self.resource2_capacity = 1
self.resource2_queue: deque = deque()
self.resource2_busy = False
```

### 2. Interarrival Distribution

Modify `sample_interarrival()` method:
```python
def sample_interarrival(self) -> float:
    # Example: Exponential with mean 5
    return random.expovariate(1.0 / 5.0)
    
    # Or constant: return 5.0
    # Or uniform: return random.uniform(3.0, 7.0)
    # Or triangular: return random.triangular(1.0, 5.0, 10.0)
```

### 3. Service Time Distributions

Modify `sample_service_time_r1()`, `sample_service_time_r2()`, etc.:
```python
def sample_service_time_r1(self) -> float:
    # Example: Exponential with mean 4
    return random.exponential(4.0)
    
    # Or triangular: return random.triangular(1.0, 3.0, 6.0)
    # Or uniform: return random.uniform(2.0, 8.0)
```

### 4. Simulation Time

Modify the `sim_time` parameter in `main()`:
```python
sim = Model4_1Simulation(sim_time=1000.0)  # Change as needed
```

### 5. Processing Flow

If your Model 4-1 has a different flow (not simple serial), modify the event handlers in `process_until_T()`.

## Common Model 4-1 Scenarios

### Scenario 1: Two-Station Serial Queue
- Resource 1 → Resource 2
- Entities flow sequentially through both resources
- Already implemented in the example

### Scenario 2: Multi-Server Parallel Queue
- Single resource with capacity > 1
- Modify `resource1_capacity` to desired value
- Adjust queue logic if needed

### Scenario 3: Three-Station Serial Process
- Resource 1 → Resource 2 → Resource 3
- Add Resource 3 following the same pattern as Resource 1 and 2

## Output Format

The CSV file (`model_4_1_results.csv`) contains:
```csv
Resource,Utilization (%)
Resource1,75.234567
Resource2,82.123456
```

## Getting Model 4-1 Specifications

To find your Model 4-1 specifications:

1. **Open the Arena model** (`Model 04-01(1).doe`) in Arena software
2. **Check the modules**:
   - Create module: Look for interarrival distribution
   - Process modules: Look for resource names and service time distributions
   - Resource module: Check resource capacities
3. **Check your textbook** (Chapter 4) for Model 4-1 description
4. **Check Run Setup**: Look at replication length (simulation time)

## Troubleshooting

- **Zero utilization**: Check that service times are correctly configured
- **100% utilization**: Check that interarrival times aren't too fast
- **No entities processed**: Check that simulation time is long enough
- **Import errors**: Make sure you're using Python 3 with standard library only (no external dependencies needed)

## Example Output

```
======================================================================
MODEL 4-1 SIMULATION - RESOURCE UTILIZATION ANALYSIS
======================================================================

======================================================================
SIMULATION RESULTS
======================================================================

Simulation Time: 1000.00
Entities Processed: 200

----------------------------------------------------------------------
Resource                         Utilization (%)
----------------------------------------------------------------------
Resource1                                   75.2346%
Resource2                                   82.1235%
----------------------------------------------------------------------

HIGHEST UTILIZATION:
  Resource: Resource2
  Utilization: 82.1235%

✓ Results saved to: model_4_1_results.csv
======================================================================
```


