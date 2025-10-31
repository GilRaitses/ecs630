# Week 1 Homework Assignment

## Problem Description

Consider the following drilling center example used in our textbook (see Figure 2.1 on page 15) and the lecture. The drilling center has a single server (machine) and an infinite queue with first-in-first-out (FIFO) discipline.

Using the drilling center, we have completed a hand simulation during 10 time units and recorded the following system trajectory in Figure 1. The x-axis and y-axis represent time and the number of parts in the system S(t), respectively. Note that S(t) includes both the number of parts in the queue, Q(t), and the number of parts in service in the machine, B(t). That is, S(t) = Q(t) + B(t) at any time. Using this trajectory, answer the following questions. Each question will have five points (total 40 points).

## Assumptions

- The system is idle at time zero.
- The first arrival occurs at time zero.
- An event changing the system status occurs only at an integer time (0, 1, …, 10).
- There are no simultaneous arrival and departure at any time point.

## Figure 1: Hand Simulation Trajectory of the Drilling Center

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

From the trajectory shown:
- Time 0-1: S(t) = 1 (first arrival at t=0)
- Time 1-2: S(t) = 1 
- Time 2-3: S(t) = 2 (arrival at t=2)
- Time 3-4: S(t) = 3 (arrival at t=3)
- Time 4-5: S(t) = 3
- Time 5-6: S(t) = 2 (departure at t=5)
- Time 6-7: S(t) = 2
- Time 7-8: S(t) = 1 (departure at t=7)
- Time 8-9: S(t) = 0 (departure at t=8)
- Time 9-10: S(t) = 1 (arrival at t=9)

## Questions

1. **What is the total number of productions?**

2. **When does the first part (p1) leave the system?**

3. **What is the average time that a part spends in the system?**

4. **What is the time-averaged number of parts in the whole system?**

5. **What is the utilization of the drill press?**

6. **What is the average waiting time in the queue of parts?**

7. **What is the average service time of a part?**
   (Hint: average service time of a part = average time spent in the system - average waiting time in the queue of a part)

8. **What is the time-averaged number of parts waiting in the queue?**
   (Hint: time-averaged number of parts waiting in the queue = time-averaged number of parts – utilization of the drill press since the utilization is the same as the number of parts in the machine)

## Solution Approach

To solve these problems, you will need to:

1. Analyze the system trajectory to identify arrival and departure events
2. Track individual parts through the system
3. Calculate time-weighted averages for system performance metrics
4. Apply Little's Law and other queueing theory relationships

## Event Timeline Analysis

Based on the trajectory:
- **t=0**: Arrival of part 1 (S(t) goes from 0 to 1)
- **t=2**: Arrival of part 2 (S(t) goes from 1 to 2)  
- **t=3**: Arrival of part 3 (S(t) goes from 2 to 3)
- **t=5**: Departure of part 1 (S(t) goes from 3 to 2)
- **t=7**: Departure of part 2 (S(t) goes from 2 to 1)
- **t=8**: Departure of part 3 (S(t) goes from 1 to 0)
- **t=9**: Arrival of part 4 (S(t) goes from 0 to 1)

This analysis will be essential for answering the questions about system performance.
