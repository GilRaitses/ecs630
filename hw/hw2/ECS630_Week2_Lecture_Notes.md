# ECS 630 — Week 2 Notes (for Homework Support)
## 1) Inverse Transform Sampling (INT and TOC in Table 1)

- Goal: Map U ~ Uniform(0,1) to target discrete distributions using cumulative probability tables from slides.
- Procedure (Inverse Transform):
  1. Take next uniform number u from the column “U(0,1)”.
  2. Find the first bin in the CDF table where CDF >= u.
  3. Assign the corresponding outcome/value for that bin.
- Apply separately to INT (interarrival time of trains, days) and TOC (tons of coal removed by trucks).
- Practical tip: When two outcomes share the same CDF boundary, use left-closed, right-open intervals [low, high).

Example (illustrative):

- Suppose INT has the CDF (from slides):
- P(INT=2)=0.30 → CDF 0.30
- P(INT=5)=0.60 → CDF 0.90
- P(INT=8)=0.10 → CDF 1.00
- If u=0.0105 → INT=2 (since 0.0105 ≤ 0.30)
- Repeat for all 10 rows to fill the “INT (days)” column.
- Do the same with TOC using its CDF table from the slides to fill “TOC (tons)”.

### Exact CDF Tables (from slides OCR)

INT (days):

| Value | Probability | CDF |
|:-----:|:-----------:|:---:|
| 2 | 0.05 | 0.05 |
| 3 | 0.13 | 0.18 |
| 4 | 0.17 | 0.35 |
| 5 | 0.27 | 0.62 |
| 6 | 0.23 | 0.85 |
| 7 | 0.10 | 0.95 |
| 8 | 0.05 | 1.00 |

TOC (tons):

| Value | Probability | CDF |
|:-----:|:-----------:|:---:|
| 70 | 0.07 | 0.07 |
| 80 | 0.10 | 0.17 |
| 90 | 0.18 | 0.35 |
| 100 | 0.28 | 0.63 |
| 110 | 0.21 | 0.84 |
| 120 | 0.10 | 0.94 |
| 130 | 0.06 | 1.00 |

Mapping rule: for a uniform u in [0, 1), choose the smallest x with CDF(x) ≥ u. Use left-closed, right-open intervals for bins: [prev_CDF, next_CDF).

## 2) Flowchart-driven Hand Simulation (Table 2, 6 iterations)

- Variables (as used in Table 2):
  - D: Day index or event count (0..5 for six iterations)
  - DOA: Day-of-arrival of train (or current arrival clock)
  - INT used: Interarrival sampled this step (from Table 1)
  - TOC used: Truck-outgoing coal this step (from Table 1)
  - PLI: Pile level inventory (tons) after updates
  - CCT: Cumulative coal transported/consumed total (if defined in slide flowchart)

Generic step logic (adapt to the lecture’s flowchart states):

1. Generate/consume INT in chronological order; update arrival time: DOA_next = DOA_prev + INT.
2. When a train arrives, add its shipment to the pile (if specified by the model).
3. For each day/iteration, remove TOC from the pile: PLI = max(0, PLI_prev − TOC).
4. Track CCT or any required KPIs after each update.
5. Proceed for 6 iterations with the INT and TOC values in order.

Tips:

- Use a running pointer into your Table 1 rows so you don’t reuse INT/TOC out of order.
- If the pile goes negative, cap at 0 unless the flowchart defines backlogs.
- Align exactly with the states/decisions in the provided flowchart (Generate INT → Advance clock → Arrival? → Update pile → Truck removes TOC → Record outputs).

## 3) What to capture in Table 2 (per iteration)

- D: iteration index (0..5)
- DOA: updated arrival time/clock based on cumulative INT
- INT used, TOC used: from Table 1, current row
- PLI: pile level after arrival and truck removal
- CCT: total removed so far (if required)

## 4) Quality checks

- Use all 6 INT/TOC pairs in order, no skipping.
- Each iteration updates the state once and records outputs consistently.
- Units: INT in days, TOC in tons; keep integer/decimal formatting per table conventions.

## 5) Deliverables alignment

- Table 1: Fully filled INT and TOC using inverse transform.
- Table 2: Six rows completed following the flowchart sequence.

## 6) Flowchart (Mermaid)

```mermaid
flowchart TD
  A[Start] --> B[Get next u_INT from Table 1]
  B --> C[Map u_INT via INT CDF → INT_k]
  C --> D[Update DOA: DOA_k = DOA_{k-1} + INT_k]
  D --> E{Train arrival event?}
  E -- Yes --> F[Add 500 tons: PLI = PLI + 500]
  E -- No --> G[No addition]
  F --> H[Get next u_TOC from Table 1]
  G --> H
  H --> I[Map u_TOC via TOC CDF → TOC_k]
  I --> J[Remove coal: PLI = max(0, PLI − TOC_k)]
  J --> K[Record D, DOA, INT_k, TOC_k, PLI, CCT]
  K --> L{Iterations < 6?}
  L -- Yes --> B
  L -- No --> M[End]

  subgraph Inverse_Transform [Inverse Transform]
    X[u in [0,1)] --> Y[Find first CDF ≥ u] --> Z[Emit outcome]
  end
```

Rendered PNG of this chart:

![Week 2 Flowchart](./ECS630_Week2_Flowchart.png)

## 7) Flowchart (ASCII)

```text
Start
  |
  v
Get next u_INT  --->  Map via INT CDF  --->  INT_k
                                            |
                                            v
                                   DOA_k = DOA_{k-1} + INT_k
                                            |
                                            v
                                   +------------------------+
                                   | Train arrival event?   |
                                   +-----------+------------+
                                               |
                           Yes ----------------+---------------- No
                             |                                    |
                PLI = PLI + 500                                  |
                             |                                    |
                             +---------------------+--------------+
                                                   |
                                                   v
                                  Get next u_TOC -> Map via TOC CDF -> TOC_k
                                                   |
                                                   v
                                  PLI = max(0, PLI - TOC_k)
                                                   |
                                                   v
                           Record D, DOA, INT_k, TOC_k, PLI, CCT
                                                   |
                                                   v
                                   Iterations < 6 ?  --Yes--> (loop)
                                                   |
                                                  No
                                                   |
                                                   v
                                                  End
```

## 8) Method and equations (how results are obtained)

Inverse transform for discrete distributions: let outcomes be x_i with probabilities p_i, and cumulative distribution F(x_i) = \(\sum_{j \le i} p_j\). For \(u \in [0,1)\), choose the smallest i such that \(F(x_i) \ge u\). Example using INT CDF above: \(u=0.375\) maps to INT = 5 because \(F(4)=0.35 < 0.375\) and \(F(5)=0.62 \ge 0.375\).

Hand simulation state updates per iteration k:
- Arrival clock: \(\mathrm{DOA}_k = \mathrm{DOA}_{k-1} + \mathrm{INT}_k\), with \(\mathrm{DOA}_0 = 0\).
- Inventory (pile) update with train arrivals adding 500 tons: \(\mathrm{PLI}_k = \max\{0,\ \mathrm{PLI}_{k-1} + 500\cdot \mathbb{1}_{\text{arrival}_k} - \mathrm{TOC}_k\}\).
- Cumulative removed (if tracked): \(\mathrm{CCT}_k = \mathrm{CCT}_{k-1} + \mathrm{TOC}_k\).

Work example (one step): If \(u_{\text{INT}}=0.375\) → INT = 5; advance DOA by 5. If a train arrives at that event, add 500 to PLI. Then with \(u_{\text{TOC}}=0.764\) (Table 1 row 1), TOC = 110 by the TOC CDF (since \(0.63 < 0.764 \le 0.84\)), so \(\mathrm{PLI} \leftarrow \max(0, \mathrm{PLI} - 110)\), and record the row.

