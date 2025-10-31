# Week 1 Homework Assignment - Complete Solutions

## Problem Description

Consider the following drilling center example used in our textbook (see Figure 2.1 on page 15) and the lecture. The drilling center has a single server (machine) and an infinite queue with first-in-first-out (FIFO) discipline.

### Assumptions and Conventions

- Measurement window: t ∈ [0,10].
- Time interpretation: The diagram represents real time; a "logical" (event-count) clock is not used. All rates and time-averages are with respect to real time over the window.
- Completed-parts convention: Entity averages (e.g., waiting time, service time) use only parts that complete within [0,10], unless otherwise stated. Time averages (e.g., L and L_q) are computed over [0,10].
- Notation: S(t) is total in system; Q(t) = max(0, S(t) - 1).

## System Trajectory Analysis

From the given trajectory diagram:

```
Total number of parts in the system, S(t)
  ↑
3 ┤
  │
2 ┤     ┌───┐
  │     │   │
1 ┤─────┤   └─────┐     ┌───┐
  │                 │     │   │
0 ┤─────────────────┴─────┤   └───────→
  └─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─
    0 1 2 3 4 5 6 7 8 9 10  time, t
```

### Event Timeline:
- **t=0**: Arrival of Part 1 (S(t): 0→1)
- **t=2**: Arrival of Part 2 (S(t): 1→2)
- **t=3**: Arrival of Part 3 (S(t): 2→3)
- **t=5**: Departure of Part 1 (S(t): 3→2)
- **t=7**: Departure of Part 2 (S(t): 2→1)
- **t=8**: Departure of Part 3 (S(t): 1→0)
- **t=9**: Arrival of Part 4 (S(t): 0→1)

### System State Over Time:
- Time 0-1: S(t) = 1
- Time 1-2: S(t) = 1
- Time 2-3: S(t) = 2
- Time 3-4: S(t) = 3
- Time 4-5: S(t) = 3
- Time 5-6: S(t) = 2
- Time 6-7: S(t) = 2
- Time 7-8: S(t) = 1
- Time 8-9: S(t) = 0
- Time 9-10: S(t) = 1

---

## Question 1: How many parts complete service (departures) in t ∈ [0,10]?

**Solution:**
Completed parts (departures) in t ∈ [0,10]

From the trajectory analysis:
- Part 1 departs at t=5
- Part 2 departs at t=7
- Part 3 departs at t=8
- Part 4 arrives at t=9 but doesn't depart by t=10

**Answer: 3 completed parts**

---

## Question 2: When does the first part (p1) leave the system?

**Solution:**
From the event timeline, Part 1 arrives at t=0 and the first departure occurs at t=5.

**Answer: t=5**

---

## Question 3: What is the average time that a part spends in the system?

**Solution:**
We need to calculate the time in system for each completed part:

- Part 1: Arrives at t=0, departs at t=5 → Time in system = 5-0 = 5
- Part 2: Arrives at t=2, departs at t=7 → Time in system = 7-2 = 5
- Part 3: Arrives at t=3, departs at t=8 → Time in system = 8-3 = 5

Average time in system = (5 + 5 + 5) ÷ 3 = 15 ÷ 3 = 5

**Answer: 5 time units**

---

## Question 4: What is the time-averaged number of parts in the whole system?

**Solution:**
Time-averaged number = ∫₀¹⁰ S(t)dt ÷ 10

Calculate area under the S(t) curve:
- Time 0-1: S(t) = 1, Area = 1 × 1 = 1
- Time 1-2: S(t) = 1, Area = 1 × 1 = 1
- Time 2-3: S(t) = 2, Area = 2 × 1 = 2
- Time 3-4: S(t) = 3, Area = 3 × 1 = 3
- Time 4-5: S(t) = 3, Area = 3 × 1 = 3
- Time 5-6: S(t) = 2, Area = 2 × 1 = 2
- Time 6-7: S(t) = 2, Area = 2 × 1 = 2
- Time 7-8: S(t) = 1, Area = 1 × 1 = 1
- Time 8-9: S(t) = 0, Area = 0 × 1 = 0
- Time 9-10: S(t) = 1, Area = 1 × 1 = 1

Total area = 1 + 1 + 2 + 3 + 3 + 2 + 2 + 1 + 0 + 1 = 16

Time-averaged number = 16 ÷ 10 = 1.6

**Answer: 1.6 parts**

---

## Question 5: What is the utilization of the drill press?

**Solution:**
Utilization = Time the server is busy ÷ Total time

The server is busy when S(t) ≥ 1 (at least one part in the system):
- Busy periods: t=0 to t=8 (8 time units) and t=9 to t=10 (1 time unit)
- Idle period: t=8 to t=9 (1 time unit)

Total busy time = 8 + 1 = 9 time units
Total time = 10 time units

Utilization = 9 ÷ 10 = 0.9 = 90%

**Answer: 0.9 or 90%**

---

## Question 6: What is the average waiting time in the queue of parts?

**Solution:**
We need to determine when parts are waiting in queue (S(t) > 1):

**Part 1:** 
- Arrives at t=0, starts service immediately
- Waiting time = 0

**Part 2:**
- Arrives at t=2, at this time S(t)=2, so Part 1 is in service
- Part 2 waits from t=2 to t=5 (when Part 1 departs)
- Waiting time = 5-2 = 3

**Part 3:**
- Arrives at t=3, at this time S(t)=3, so Part 1 is in service and Part 2 is waiting
- Part 3 waits from t=3 to t=7 (when Part 2 starts service after Part 1 departs at t=5)
- Waiting time = 7-3 = 4

**Part 4:**
- Arrives at t=9, system is empty, starts service immediately
- Waiting time = 0

All arrivals in [0,10]: (0 + 3 + 4 + 0) ÷ 4 = 7 ÷ 4 = 1.75.

Completed-parts convention (Parts 1–3): (0 + 3 + 4) ÷ 3 = 7 ÷ 3 ≈ 2.33.

Consistency check over [0,10]: L_q = 0.7 (from Q8) and departures per unit time = 3/10, so L_q/throughput = 0.7/0.3 = 7/3.

**Answer: 7/3 time units (≈ 2.33), using the completed-parts convention**

---

## Question 7: What is the average service time of a part?

**Solution:**
Using the hint: Average service time = Average time in system - Average waiting time in queue

From Question 3: Average time in system = 5
From Question 6: Average waiting time in queue = 7/3

Average service time = 5 - 7/3 = 8/3 ≈ 2.67

**Verification by direct calculation:**
- Part 1: Service time = 5-0 = 5 (arrives at t=0, departs at t=5)
- Part 2: Service time = 7-5 = 2 (starts service at t=5, departs at t=7)
- Part 3: Service time = 8-7 = 1 (starts service at t=7, departs at t=8)

Average = (5 + 2 + 1) ÷ 3 = 8 ÷ 3 = 2.67

**Answer: 8/3 time units (≈ 2.67)**

---

## Question 8: What is the time-averaged number of parts waiting in the queue?

**Solution:**
Using the hint: Time-averaged number waiting = Time-averaged number in system - Utilization

From Question 4: Time-averaged number in system = 1.6
From Question 5: Utilization = 0.9

Time-averaged number waiting in queue = 1.6 - 0.9 = 0.7

**Verification by direct calculation:**
Queue length Q(t) = max(0, S(t) - 1):
- Time 0-1: Q(t) = max(0, 1-1) = 0
- Time 1-2: Q(t) = max(0, 1-1) = 0
- Time 2-3: Q(t) = max(0, 2-1) = 1
- Time 3-4: Q(t) = max(0, 3-1) = 2
- Time 4-5: Q(t) = max(0, 3-1) = 2
- Time 5-6: Q(t) = max(0, 2-1) = 1
- Time 6-7: Q(t) = max(0, 2-1) = 1
- Time 7-8: Q(t) = max(0, 1-1) = 0
- Time 8-9: Q(t) = max(0, 0-1) = 0
- Time 9-10: Q(t) = max(0, 1-1) = 0

Area under Q(t) curve = 0 + 0 + 1 + 2 + 2 + 1 + 1 + 0 + 0 + 0 = 7
Time-averaged queue length = 7 ÷ 10 = 0.7

**Answer: 0.7 parts**

---

## Summary of Answers

1. Completed parts (departures) in [0,10]: **3**
2. When first part leaves: **t = 5**
3. Average time in system: **5 time units**
4. Time-averaged number in system: **1.6 parts**
5. Utilization of drill press: **0.9 or 90%**
6. Average waiting time in queue: **7/3 time units (≈ 2.33)**
7. Average service time: **8/3 time units (≈ 2.67)**
8. Time-averaged number waiting in queue: **0.7 parts**
