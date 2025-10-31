#!/usr/bin/env python3
"""
Simulation script for Arena Model 4-1
Calculates resource utilization for each resource in the system.

This script is based on the discrete-event simulation framework from hw3.
You need to configure it based on your Model 4-1 specifications by:
1. Modifying the resources in __init__
2. Updating distributions in sample_* methods
3. Adjusting the event handlers for your specific flow
"""

import json
import csv
import heapq
import random
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Optional

# Set seed for reproducibility
random.seed(42)

@dataclass(order=True)
class Event:
    time: float
    priority: int
    event_type: str = field(compare=False)
    entity_id: Optional[int] = field(compare=False, default=None)

class Model4_1Simulation:
    """
    Discrete-event simulation for Model 4-1
    
    CONFIGURATION REQUIRED:
    Modify this class based on your Model 4-1 specifications.
    Common Model 4-1 scenarios:
    - Two-station serial queue (Resource1 -> Resource2)
    - Multi-server parallel queue (1 resource with capacity > 1)
    - Three-station serial process (Resource1 -> Resource2 -> Resource3)
    """
    
    def __init__(self, sim_time: float = 1000.0):
        self.T = sim_time
        
        # Simulation state
        self.clock: float = 0.0
        self.next_entity_id: int = 0
        self.event_list: List[Event] = []
        
        # Resources - MODIFY THESE based on your Model 4-1
        # Format: (capacity, queue, busy_count)
        # Example for two-resource serial system:
        self.resource1_capacity = 1
        self.resource1_queue: deque = deque()
        self.resource1_busy = False
        
        self.resource2_capacity = 1
        self.resource2_queue: deque = deque()
        self.resource2_busy = False
        
        # Add more resources as needed:
        # self.resource3_capacity = 1
        # self.resource3_queue: deque = deque()
        # self.resource3_busy = False
        
        # Resource utilization tracking (time integrals)
        self.last_time: float = 0.0
        self.area_busy_r1: float = 0.0
        self.area_busy_r2: float = 0.0
        # self.area_busy_r3: float = 0.0  # Add if needed
        
        # Entity tracking
        self.entity_info: Dict[int, Dict[str, float]] = {}
        self.num_in_system: int = 0
        
    def update_areas(self, new_time: float) -> None:
        """Update time-averaged statistics"""
        dt = new_time - self.last_time
        if dt < 0:
            dt = 0
        
        # Update resource busy time (for utilization)
        self.area_busy_r1 += (1.0 if self.resource1_busy else 0.0) * dt
        self.area_busy_r2 += (1.0 if self.resource2_busy else 0.0) * dt
        # self.area_busy_r3 += (1.0 if self.resource3_busy else 0.0) * dt
        
        self.last_time = new_time
    
    def schedule(self, time: float, priority: int, event_type: str, entity_id: Optional[int] = None) -> None:
        """Schedule an event"""
        heapq.heappush(self.event_list, Event(time, priority, event_type, entity_id))
    
    # ============================================================================
    # DISTRIBUTIONS - MODIFY THESE based on your Model 4-1
    # ============================================================================
    
    def sample_interarrival(self) -> float:
        """
        Sample interarrival time
        
        MODIFY based on your Model 4-1 interarrival distribution.
        Common distributions:
        - Exponential: random.expovariate(1.0 / mean)
        - Constant: return constant_value
        - Uniform: random.uniform(min, max)
        - Triangular: random.triangular(min, mode, max)
        """
        # EXAMPLE: Exponential with mean 5
        return random.expovariate(1.0 / 5.0)
    
    def sample_service_time_r1(self) -> float:
        """
        Sample service time for Resource 1
        
        MODIFY based on your Model 4-1 Resource 1 service time distribution
        """
        # EXAMPLE: Exponential with mean 4
        return random.expovariate(1.0 / 4.0)
        # Other examples:
        # return random.triangular(1.0, 3.0, 6.0)  # Triangular
        # return random.uniform(2.0, 8.0)  # Uniform
    
    def sample_service_time_r2(self) -> float:
        """
        Sample service time for Resource 2
        
        MODIFY based on your Model 4-1 Resource 2 service time distribution
        """
        # EXAMPLE: Exponential with mean 3
        return random.expovariate(1.0 / 3.0)
    
    # Add more service time methods if you have more resources:
    # def sample_service_time_r3(self) -> float:
    #     return random.expovariate(1.0 / 5.0)
    
    # ============================================================================
    # RESOURCE MANAGEMENT - MODIFY IF YOUR MODEL HAS DIFFERENT FLOW
    # ============================================================================
    
    def start_resource1_if_possible(self, entity_id: int) -> None:
        """Try to start service at Resource 1"""
        if not self.resource1_busy:
            self.resource1_busy = True
            self.entity_info[entity_id]["r1_start"] = self.clock
            st = self.sample_service_time_r1()
            self.schedule(self.clock + st, 1, "r1_done", entity_id)
        else:
            self.entity_info[entity_id]["r1_queue_enter"] = self.clock
            self.resource1_queue.append(entity_id)
    
    def start_resource2_if_possible(self, entity_id: int) -> None:
        """Try to start service at Resource 2"""
        if not self.resource2_busy:
            self.resource2_busy = True
            self.entity_info[entity_id]["r2_start"] = self.clock
            st = self.sample_service_time_r2()
            self.schedule(self.clock + st, 2, "r2_done", entity_id)
        else:
            self.entity_info[entity_id]["r2_queue_enter"] = self.clock
            self.resource2_queue.append(entity_id)
    
    # Add more resource start methods if needed:
    # def start_resource3_if_possible(self, entity_id: int) -> None:
    #     ...
    
    # ============================================================================
    # EVENT PROCESSING - MODIFY IF YOUR MODEL HAS DIFFERENT FLOW
    # ============================================================================
    
    def process_until_T(self) -> None:
        """Main simulation loop"""
        # Schedule first arrival
        self.schedule(0.0, 0, "arrival", None)
        next_arrival_time = 0.0
        
        while self.event_list:
            evt = heapq.heappop(self.event_list)
            
            # Stop at simulation end time
            if evt.time > self.T:
                self.update_areas(self.T)
                self.clock = self.T
                break
            
            # Advance time and update statistics
            self.update_areas(evt.time)
            self.clock = evt.time
            
            if evt.event_type == "arrival":
                # Create new entity
                eid = self.next_entity_id
                self.next_entity_id += 1
                self.entity_info[eid] = {"arrival": self.clock}
                self.num_in_system += 1
                
                # Start processing at Resource 1
                self.start_resource1_if_possible(eid)
                
                # Schedule next arrival
                next_arrival_time += self.sample_interarrival()
                if next_arrival_time <= self.T:
                    self.schedule(next_arrival_time, 0, "arrival", None)
            
            elif evt.event_type == "r1_done":
                eid = evt.entity_id
                self.entity_info[eid]["r1_done"] = self.clock
                
                # Free Resource 1 and start next entity if queue not empty
                if self.resource1_queue:
                    next_eid = self.resource1_queue.popleft()
                    self.entity_info[next_eid]["r1_start"] = self.clock
                    st = self.sample_service_time_r1()
                    self.schedule(self.clock + st, 1, "r1_done", next_eid)
                    self.resource1_busy = True
                else:
                    self.resource1_busy = False
                
                # Move entity to Resource 2
                self.start_resource2_if_possible(eid)
            
            elif evt.event_type == "r2_done":
                eid = evt.entity_id
                self.entity_info[eid]["r2_done"] = self.clock
                
                # Free Resource 2 and start next entity if queue not empty
                if self.resource2_queue:
                    next_eid = self.resource2_queue.popleft()
                    self.entity_info[next_eid]["r2_start"] = self.clock
                    st = self.sample_service_time_r2()
                    self.schedule(self.clock + st, 2, "r2_done", next_eid)
                    self.resource2_busy = True
                else:
                    self.resource2_busy = False
                
                # Entity leaves system
                if self.num_in_system > 0:
                    self.num_in_system -= 1
                # Entity is complete
        
        # Finalize if simulation ended before T
        if self.clock < self.T:
            self.update_areas(self.T)
            self.clock = self.T
    
    def calculate_utilizations(self) -> Dict[str, float]:
        """Calculate resource utilizations"""
        T = self.T
        util_r1 = (self.area_busy_r1 / T * 100.0) if T > 0 else 0.0
        util_r2 = (self.area_busy_r2 / T * 100.0) if T > 0 else 0.0
        # util_r3 = (self.area_busy_r3 / T * 100.0) if T > 0 else 0.0
        
        utilizations = {
            "Resource1": util_r1,
            "Resource2": util_r2,
            # "Resource3": util_r3,  # Add if needed
        }
        
        return utilizations


def main():
    """
    Main function
    
    CONFIGURE THIS based on your Model 4-1:
    1. Set sim_time to match your simulation length
    2. Modify resources in __init__ method
    3. Update distributions in sample_* methods
    4. Adjust event handlers if flow is different
    """
    
    print("="*70)
    print("MODEL 4-1 SIMULATION - RESOURCE UTILIZATION ANALYSIS")
    print("="*70)
    print("\n⚠️  CONFIGURATION REQUIRED ⚠️")
    print("\nThis script needs to be configured based on your Model 4-1.")
    print("Please modify:")
    print("  1. Resources in __init__ (add/modify resources)")
    print("  2. sample_interarrival() - Set interarrival distribution")
    print("  3. sample_service_time_r*() - Set service time distributions")
    print("  4. Event handlers if your flow differs")
    print("  5. sim_time below")
    print("\nRunning with EXAMPLE configuration (2-resource serial)...\n")
    
    # Create and run simulation
    sim = Model4_1Simulation(sim_time=1000.0)  # MODIFY simulation time
    sim.process_until_T()
    
    # Calculate utilizations
    utilizations = sim.calculate_utilizations()
    
    # Display results
    print("="*70)
    print("SIMULATION RESULTS")
    print("="*70)
    print(f"\nSimulation Time: {sim.T:.2f}")
    print(f"Entities Processed: {sim.next_entity_id}")
    print("\n" + "-"*70)
    print(f"{'Resource':<30} {'Utilization (%)':>20}")
    print("-"*70)
    
    for resource_name in sorted(utilizations.keys()):
        utilization = utilizations[resource_name]
        print(f"{resource_name:<30} {utilization:>19.4f}%")
    
    # Find highest utilization
    if utilizations:
        max_resource = max(utilizations.items(), key=lambda x: x[1])
        print("-"*70)
        print(f"\nHIGHEST UTILIZATION:")
        print(f"  Resource: {max_resource[0]}")
        print(f"  Utilization: {max_resource[1]:.4f}%")
    
    # Save to CSV
    output_file = 'model_4_1_results.csv'
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Resource', 'Utilization (%)'])
        for resource_name in sorted(utilizations.keys()):
            writer.writerow([resource_name, f'{utilizations[resource_name]:.6f}'])
    
    print(f"\n✓ Results saved to: {output_file}")
    print("="*70 + "\n")
    
    return utilizations


if __name__ == '__main__':
    main()
