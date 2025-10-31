#!/usr/bin/env python3
import json
import csv
import heapq
import random
import os
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

# Reproducibility
random.seed(6303)

@dataclass(order=True)
class Event:
    time: float
    priority: int
    event_type: str = field(compare=False)
    entity_id: Optional[int] = field(compare=False, default=None)

class PaintingSystemSimulation:
    def __init__(self, horizon_minutes: float = 1440.0, drying_capacity: int = 10):
        self.T: float = horizon_minutes
        self.drying_capacity: int = drying_capacity

        # State
        self.clock: float = 0.0
        self.next_entity_id: int = 0
        self.event_list: List[Event] = []

        # Queues and resources
        self.painting_queue: deque[int] = deque()
        self.finishing_queue: deque[int] = deque()
        self.drying_queue: deque[int] = deque()
        self.painting_busy: bool = False
        self.finishing_busy: bool = False
        self.drying_in_progress: int = 0

        # Accounting for time-average metrics
        self.last_time: float = 0.0
        self.area_num_in_system: float = 0.0
        self.area_paint_queue: float = 0.0
        self.area_finish_queue: float = 0.0
        self.area_busy_paint: float = 0.0
        self.area_busy_finish: float = 0.0

        # Per-entity tracking
        self.entity_info: Dict[int, Dict[str, float]] = {}

        # Averages and counters
        self.sum_wq_paint: float = 0.0
        self.count_wq_paint: int = 0
        self.sum_wq_finish: float = 0.0
        self.count_wq_finish: int = 0
        self.sum_time_in_system: float = 0.0
        self.count_completed_in_T: int = 0
        self.num_in_system: int = 0

        # Output timeline (optional)
        self.timeline_rows: List[Tuple] = []

    # Distributions
    def sample_interarrival(self) -> float:
        # Exponential with mean 5 minutes => rate = 1/5
        return random.expovariate(1.0 / 5.0)

    def sample_painting_time(self) -> float:
        # Triangular(min=1, mode=4.5, max=7)
        return random.triangular(1.0, 7.0, 4.5)

    def sample_finishing_time(self) -> float:
        # Uniform(0.5, 9)
        return random.uniform(0.5, 9.0)

    # Event scheduling
    def schedule(self, time: float, priority: int, event_type: str, entity_id: Optional[int] = None) -> None:
        heapq.heappush(self.event_list, Event(time, priority, event_type, entity_id))

    def update_areas(self, new_time: float) -> None:
        dt = new_time - self.last_time
        if dt < 0:
            dt = 0
        # Time-average number in system includes everything in system, including drying
        self.area_num_in_system += self.num_in_system * dt
        self.area_paint_queue += len(self.painting_queue) * dt
        self.area_finish_queue += len(self.finishing_queue) * dt
        self.area_busy_paint += (1.0 if self.painting_busy else 0.0) * dt
        self.area_busy_finish += (1.0 if self.finishing_busy else 0.0) * dt
        self.last_time = new_time

    def log_timeline(self, label: str) -> None:
        self.timeline_rows.append((self.clock,
                                   label,
                                   self.num_in_system,
                                   len(self.painting_queue),
                                   int(self.painting_busy),
                                   self.drying_in_progress,
                                   len(self.finishing_queue),
                                   int(self.finishing_busy)))

    def start_painting_if_possible(self, entity_id: int) -> None:
        if not self.painting_busy:
            # Start service immediately
            self.painting_busy = True
            self.entity_info[entity_id]["paint_start"] = self.clock
            if "paint_queue_enter" in self.entity_info[entity_id]:
                wq = self.clock - self.entity_info[entity_id]["paint_queue_enter"]
                self.sum_wq_paint += wq
                self.count_wq_paint += 1
            st = self.sample_painting_time()
            self.schedule(self.clock + st, 1, "paint_done", entity_id)
        else:
            # Queue for painting
            self.entity_info[entity_id]["paint_queue_enter"] = self.clock
            self.painting_queue.append(entity_id)

    def start_drying_if_possible(self, entity_id: int) -> None:
        if self.drying_in_progress < self.drying_capacity:
            self.drying_in_progress += 1
            self.entity_info[entity_id]["dry_start"] = self.clock
            # Fixed drying time 15 minutes
            self.schedule(self.clock + 15.0, 2, "drying_done", entity_id)
        else:
            self.entity_info[entity_id]["dry_queue_enter"] = self.clock
            self.drying_queue.append(entity_id)

    def start_finishing_if_possible(self, entity_id: int) -> None:
        if not self.finishing_busy:
            self.finishing_busy = True
            self.entity_info[entity_id]["finish_start"] = self.clock
            if "finish_queue_enter" in self.entity_info[entity_id]:
                wq = self.clock - self.entity_info[entity_id]["finish_queue_enter"]
                self.sum_wq_finish += wq
                self.count_wq_finish += 1
            st = self.sample_finishing_time()
            self.schedule(self.clock + st, 3, "finish_done", entity_id)
        else:
            self.entity_info[entity_id]["finish_queue_enter"] = self.clock
            self.finishing_queue.append(entity_id)

    def process_until_T(self) -> None:
        # Initial arrival at time 0
        self.schedule(0.0, 0, "arrival", None)
        next_arrival_time = 0.0

        while self.event_list:
            evt = heapq.heappop(self.event_list)
            # Stop integrating at T
            if evt.time > self.T:
                # Update areas to T and stop
                self.update_areas(self.T)
                self.clock = self.T
                break

            # Advance time and integrate
            self.update_areas(evt.time)
            self.clock = evt.time

            if evt.event_type == "arrival":
                # Generate entity
                eid = self.next_entity_id
                self.next_entity_id += 1
                self.entity_info[eid] = {"arrival": self.clock}
                self.num_in_system += 1
                self.log_timeline("arrival")

                # Attempt to start painting
                self.start_painting_if_possible(eid)

                # Schedule next arrival within T
                next_arrival_time += self.sample_interarrival()
                if next_arrival_time <= self.T:
                    self.schedule(next_arrival_time, 0, "arrival", None)

            elif evt.event_type == "paint_done":
                eid = evt.entity_id
                self.entity_info[eid]["paint_done"] = self.clock
                # Painting server becomes available, start next if queue not empty
                if self.painting_queue:
                    next_eid = self.painting_queue.popleft()
                    # Start immediately for next
                    self.entity_info[next_eid]["paint_start"] = self.clock
                    if "paint_queue_enter" in self.entity_info[next_eid]:
                        wq = self.clock - self.entity_info[next_eid]["paint_queue_enter"]
                        self.sum_wq_paint += wq
                        self.count_wq_paint += 1
                    st = self.sample_painting_time()
                    self.schedule(self.clock + st, 1, "paint_done", next_eid)
                    self.painting_busy = True
                else:
                    self.painting_busy = False
                self.log_timeline("paint_done")
                # Move to drying
                self.start_drying_if_possible(eid)

            elif evt.event_type == "drying_done":
                eid = evt.entity_id
                self.entity_info[eid]["dry_done"] = self.clock
                # Free drying spot
                if self.drying_in_progress > 0:
                    self.drying_in_progress -= 1
                # Start drying for next waiting if any
                if self.drying_queue:
                    next_eid = self.drying_queue.popleft()
                    self.drying_in_progress += 1
                    self.entity_info[next_eid]["dry_start"] = self.clock
                    self.schedule(self.clock + 15.0, 2, "drying_done", next_eid)
                self.log_timeline("drying_done")
                # Move to finishing
                self.start_finishing_if_possible(eid)

            elif evt.event_type == "finish_done":
                eid = evt.entity_id
                self.entity_info[eid]["finish_done"] = self.clock
                # Completion within horizon contributes to time-in-system average if within T
                if self.clock <= self.T and "arrival" in self.entity_info[eid]:
                    self.sum_time_in_system += self.clock - self.entity_info[eid]["arrival"]
                    self.count_completed_in_T += 1
                # Finishing server becomes available, start next if queue not empty
                if self.finishing_queue:
                    next_eid = self.finishing_queue.popleft()
                    self.entity_info[next_eid]["finish_start"] = self.clock
                    if "finish_queue_enter" in self.entity_info[next_eid]:
                        wq = self.clock - self.entity_info[next_eid]["finish_queue_enter"]
                        self.sum_wq_finish += wq
                        self.count_wq_finish += 1
                    st = self.sample_finishing_time()
                    self.schedule(self.clock + st, 3, "finish_done", next_eid)
                    self.finishing_busy = True
                else:
                    self.finishing_busy = False
                # Entity leaves system
                if self.num_in_system > 0:
                    self.num_in_system -= 1
                self.log_timeline("finish_done")

        # If loop exits due to empty event list before T, integrate to T
        if self.clock < self.T:
            self.update_areas(self.T)
            self.clock = self.T

    def results(self) -> Dict[str, float]:
        T = self.T
        avg_num_in_system = self.area_num_in_system / T if T > 0 else 0.0
        lq_paint = self.area_paint_queue / T if T > 0 else 0.0
        lq_finish = self.area_finish_queue / T if T > 0 else 0.0
        util_paint = self.area_busy_paint / T if T > 0 else 0.0
        util_finish = self.area_busy_finish / T if T > 0 else 0.0
        avg_wq_paint = (self.sum_wq_paint / self.count_wq_paint) if self.count_wq_paint > 0 else 0.0
        avg_wq_finish = (self.sum_wq_finish / self.count_wq_finish) if self.count_wq_finish > 0 else 0.0
        avg_time_in_system = (self.sum_time_in_system / self.count_completed_in_T) if self.count_completed_in_T > 0 else 0.0
        return {
            "T": T,
            "avg_total_time_in_system": avg_time_in_system,
            "l_sys": avg_num_in_system,
            "avg_wq_paint": avg_wq_paint,
            "lq_paint": lq_paint,
            "util_paint": util_paint,
            "avg_wq_finish": avg_wq_finish,
            "lq_finish": lq_finish,
            "util_finish": util_finish,
            "num_completed": self.count_completed_in_T
        }

    def write_outputs(self, results_path_json: str, results_path_csv: str, timeline_csv: str) -> None:
        # Write metrics JSON
        with open(results_path_json, "w") as f:
            json.dump(self.results(), f, indent=2)
        # Write metrics CSV
        metrics = self.results()
        with open(results_path_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            for k, v in metrics.items():
                writer.writerow([k, v])
        # Write timeline CSV (optional diagnostics)
        with open(timeline_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time", "event", "N_system", "Q_paint", "Busy_paint", "Drying_in_progress", "Q_finish", "Busy_finish"])
            writer.writerows(self.timeline_rows)


def main() -> None:
    sim = PaintingSystemSimulation(horizon_minutes=1440.0, drying_capacity=10)
    sim.process_until_T()
    out_dir = os.path.dirname(os.path.abspath(__file__))
    sim.write_outputs(
        results_path_json=os.path.join(out_dir, "results_week3.json"),
        results_path_csv=os.path.join(out_dir, "results_week3.csv"),
        timeline_csv=os.path.join(out_dir, "timeline_week3.csv"),
    )

if __name__ == "__main__":
    main()
