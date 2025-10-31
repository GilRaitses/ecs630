# Lab02 Arena Model Build Instructions - OAPM3

## Model OAPM3: Basic Auto Part Manufacturing System

### Overview
This model represents the foundational OAPM manufacturing system with CAD inspection and CAM file generation processes. The system includes complex scheduling with multiple workers, shift patterns, and time-varying arrival rates throughout a 9-hour workday.

### Key System Features:
- **Multi-step Process**: Order review → CAD inspection → CAM file generation → Order completion
- **Complex Workforce**: Part-time, full-time, and site machinists with different schedules
- **Resource Constraints**: Only 2 computers for CAD inspection and CAM file generation
- **Time-varying Arrivals**: Different order rates throughout the 9-hour day
- **Realistic Timing**: Triangular and uniform processing distributions

---

## System Description

### **Process Flow:**
1. **Orders Arrive**: Variable rates by hour (see arrival schedule)
2. **Order Review**: Machinist reviews order (2-4 minutes, mode 3)
3. **CAD Inspection**: Computer-based process (30-60 seconds)
4. **CAM File Generation**: Computer-based process (exactly 45 seconds)
5. **Order Summary Printing**: Final step before completion

### **Workforce Schedule:**
- **Rich (Part-time)**: 10 AM - 2 PM daily
- **Ann (Full-time)**: 8 AM - 5 PM with break 10:30-11:30 AM
- **Dina (Site Machinist)**: Available when others are busy
- **Plant Manager**: 8 AM - 5 PM with break 1:15-2:15 PM

### **Resource Constraints:**
- **Computers**: Only 2 available for CAD inspection and CAM file generation
- **Shared Usage**: Same computers used for both CAD and CAM processes

---

## Arrival Schedule Implementation

### **Hourly Arrival Rates:**
| Time Period | Average Orders/Hour |
|-------------|-------------------|
| 8 AM - 9 AM | 10 |
| 9 AM - 10 AM | 15 |
| 10 AM - 11 AM | 25 |
| 11 AM - Noon | 37 |
| Noon - 1 PM | 36 |
| 1 PM - 2 PM | 14 |
| 2 PM - 3 PM | 11 |
| 3 PM - 4 PM | 21 |
| 4 PM - 5 PM | 24 |

---

## Build Steps

### 1. **Create Module with Schedule**

#### **1.1 Schedule Data Module**
   - Go to **Data Definition** → **Schedule**
   - Create new schedule: "Order_Arrival_Schedule"
   - **Type**: Arrival
   - **Time Units**: Minutes
   - **COMMENT**: Controls time-varying arrivals - slow mornings, peak lunch, varying afternoons.
   - Define hourly rates in the Durations popup:
     ```
     Value | Duration | Time Period
     10    | 60       | (8-9 AM) - Slow morning start
     15    | 60       | (9-10 AM) - Building up
     25    | 60       | (10-11 AM) - Getting busier
     37    | 60       | (11-12 PM) - PEAK HOUR - lunch rush
     36    | 60       | (12-1 PM) - PEAK HOUR continues
     14    | 60       | (1-2 PM) - Post-lunch slowdown
     11    | 60       | (2-3 PM) - Slowest period
     21    | 60       | (3-4 PM) - Afternoon pickup
     24    | 60       | (4-5 PM) - End of day rush
     ```

#### **1.2 Create Module**
   - Drag **Create** module from Basic Process panel
   - Name: "Orders Arrive"
   - Entity Type: "Order"
   - **Type**: Schedule
   - **Schedule Name**: "Order_Arrival_Schedule"
   - **Time Units**: Minutes
   - Max Arrivals: Infinite
   - First Creation: 0.0
   - **COMMENT**: Orders "birth" point. Uses schedule instead of constant arrivals like OAPM2.

### 2. **Order Review Process**

#### **2.1 Process Module**
   - Drag **Process** module from Basic Process panel
   - Name: "Order Review"
   - Type: Standard
   - Action: **Seize Delay Release**
   - Resources: "r Machinist" (will define multiple units)
   - **Delay Type**: Triangular
   - **Minimum**: 2
   - **Value (Mode)**: 3
   - **Maximum**: 4
   - **Units**: Minutes
   - Allocation: Value Added
   - Report Statistics: Checked
   - **COMMENT**: Machinists review orders. Triangular 2-4 min (mode 3). Orders wait if all busy.

### 3. **Resource Scheduling Setup**

#### **3.1 Schedule Data Modules for Workers**

**Rich's Schedule (Part-time):**
   - Schedule Name: "Rich_Schedule"
   - Type: Capacity
   - Time Units: Minutes
   ```
   Start Time | Duration | Capacity
   0          | 120      | 0        (8-10 AM: not working)
   120        | 240      | 1        (10 AM-2 PM: working)
   360        | 180      | 0        (2-5 PM: not working)
   ```

**Ann's Schedule (Full-time with break):**
   - Schedule Name: "Ann_Schedule"
   - Type: Capacity
   - Time Units: Minutes
   ```
   Start Time | Duration | Capacity
   0          | 150      | 1        (8-10:30 AM: working)
   150        | 60       | 0        (10:30-11:30 AM: break)
   210        | 330      | 1        (11:30 AM-5 PM: working)
   ```

**Dina's Schedule (Site Machinist):**
   - Schedule Name: "Dina_Schedule"
   - Type: Capacity
   - Time Units: Minutes
   - Capacity: 1 (always available when needed)

#### **3.2 Resource Data Module**
   - Go to **Data Definition** → **Resource**
   - **r Machinist**:
     - Type: Based on Schedule
     - Schedule Name: Combine schedules or use multiple resources
   - **Alternative**: Create separate resources (r_Rich, r_Ann, r_Dina)

### 4. **CAD Inspection Process**

#### **4.1 Process Module**
   - Name: "CAD Inspection"
   - Type: Standard
   - Action: **Seize Delay Release**
   - Resources: "r Computer" (capacity 2)
   - **Delay Type**: Uniform
   - **Minimum**: 0.5 minutes (30 seconds)
   - **Maximum**: 1.0 minutes (60 seconds)
   - **Units**: Minutes
   - Allocation: Value Added
   - Report Statistics: Checked
   - **COMMENT**: CAD inspection using 1 of 2 computers. Uniform 30-60 sec. Key bottleneck.

### 5. **CAM File Generation Process**

#### **5.1 Process Module**
   - Name: "CAM File Generation"
   - Type: Standard
   - Action: **Seize Delay Release**
   - Resources: "r Computer" (same resource as CAD)
   - **Delay Type**: Constant
   - **Value**: 0.75 minutes (45 seconds)
   - **Units**: Minutes
   - Allocation: Value Added
   - Report Statistics: Checked
   - **COMMENT**: CAM generation using same 2 computers as CAD. Constant 45 sec. Competes for computers.

### 6. **Order Summary Printing**

#### **6.1 Process Module**
   - Name: "Print Order Summary"
   - Type: Standard
   - Action: **Delay** (no resource needed)
   - **Delay Type**: Constant
   - **Value**: 0.75 minutes (45 seconds)
   - **Units**: Minutes
   - Allocation: Value Added
   - **COMMENT**: Final printing step. Constant 45 sec. No resource needed.

### 7. **System Flow Connection**
```
Orders Arrive → Order Review → CAD Inspection → CAM File Generation → Print Order Summary → Dispose
```
- **COMMENT**: Connect modules with arrows. Creates order flow path through manufacturing.

### 8. **Resource Configuration**

#### **8.1 Computer Resource**
   - **r Computer**:
     - Type: Fixed Capacity
     - Capacity: 2
     - Report Statistics: Yes
   - **COMMENT**: 2 computers shared by CAD and CAM. Creates competition/queuing.

#### **8.2 Machinist Resources (Recommended: Three Separate Resources)**

**Step 1: Create Three Resources in Resource Data Module**
   - Go to **Data Definition** → **Resource**
   - **r Rich**:
     - Type: Based on Schedule
     - Schedule Name: "Rich_Schedule"
     - **COMMENT**: Part-time worker, 10 AM - 2 PM only
   
   - **r Ann**:
     - Type: Based on Schedule  
     - Schedule Name: "Ann_Schedule"
     - **COMMENT**: Full-time with break 10:30-11:30 AM
   
   - **r Dina**:
     - Type: Based on Schedule
     - Schedule Name: "Dina_Schedule" 
     - **COMMENT**: Site machinist, available all day as backup

**Step 2: Update Order Review Process Module**
   - Edit your "Order Review" Process module
   - **Resources**: Delete "r Machinist", Add all three:
     - Add "r Rich"
     - Add "r Ann" 
     - Add "r Dina"
   - **COMMENT**: Orders can seize ANY available machinist. Arena picks first available worker.

### 9. **Run Settings**
   - **Run** → **Setup** → **Replication Parameters**
   - **Replication Length**: 9 × 60 = **540** minutes (9-hour day)
   - **Time Units**: Minutes
   - **Base Time Units**: Minutes
   - **Number of Replications**: 1
   - **Warm-up Period**: 0
   - **COMMENT**: Simulates 9-hour workday. Shows performance during slow/peak periods.

---

## Advanced Implementation Options

### **Option 1: Single Machinist Resource with Complex Schedule**
- Combine all worker schedules into one resource
- Use expressions to handle overlapping availability

### **Option 2: Multiple Machinist Resources**
- Create separate resources for each worker
- Use **Seize** module to grab any available machinist
- More complex but more realistic

### **Option 3: Dina as Backup Resource**
- Primary resources: Rich and Ann
- Dina only activated when others are busy
- Use **Queue** ranking or **Priority** settings

---

## Expected Model Behavior

### **Peak Hours:**
- **11 AM - 1 PM**: Highest arrival rates (36-37 orders/hour)
- Computer resources likely to be bottleneck
- Queue buildup expected during peak times

### **Low Activity Periods:**
- **1 PM - 3 PM**: Reduced arrivals (11-14 orders/hour)
- Ann's break (1:15-2:15 PM) coincides with low activity
- System should catch up during these periods

### **Resource Utilization:**
- **Computers**: High utilization during peak hours
- **Rich**: Only available 10 AM - 2 PM (peak period)
- **Ann**: Available most of the day except break
- **Dina**: Utilization depends on system load

---

## Key Performance Indicators

### **System Metrics:**
- **Total orders processed** in 9-hour day
- **Average time in system** per order
- **Maximum queue lengths** at each process
- **Resource utilization rates** for computers and machinists

### **Bottleneck Analysis:**
- **Computer utilization** (likely bottleneck with only 2 units)
- **Peak hour performance** (11 AM - 1 PM)
- **Impact of worker breaks** on system performance

---

## Verification Checklist

- [ ] **Arrival schedule** matches hourly rates from table
- [ ] **Worker schedules** reflect correct availability periods
- [ ] **Computer sharing** between CAD and CAM processes
- [ ] **Processing times** match specifications
- [ ] **Resource capacities** set correctly (2 computers, multiple machinists)
- [ ] **9-hour simulation** length configured
- [ ] **Statistics collection** enabled for all processes

---

## Troubleshooting

### **Common Issues:**
1. **Schedule conflicts**: Verify time units are consistent (minutes)
2. **Resource sharing**: Ensure same computer resource used for CAD and CAM
3. **Worker availability**: Check schedule start/end times align with workday
4. **Peak hour bottlenecks**: Normal behavior - computers are constrained

### **Performance Validation:**
1. **Total arrivals**: Should approximate sum of hourly rates × time
2. **Computer utilization**: Should be high during peak hours
3. **Worker utilization**: Should reflect their availability schedules
4. **Queue statistics**: Expect buildup during 11 AM - 1 PM peak

---

## Analysis Framework

### **Key Questions:**
1. **What is the system bottleneck?** (Likely computers with capacity 2)
2. **How do worker breaks affect performance?**
3. **What happens during peak arrival periods?**
4. **Is the current workforce adequate for demand?**

### **Scenarios to Test:**
- **Additional computers**: What if 3 or 4 computers available?
- **Extended worker hours**: Impact of longer shifts
- **Demand smoothing**: Effect of more uniform arrival rates

---

## Submission Requirements

1. **Model File**: "OAPM3_[YourName].doe"
2. **Results Report**: Focus on bottleneck analysis and resource utilization
3. **Screenshots**: Show schedule implementation and peak hour queues
4. **Analysis**: 
   - Bottleneck identification
   - Resource utilization patterns
   - Peak vs. off-peak performance comparison

This model demonstrates **complex scheduling**, **resource sharing**, **time-varying arrivals**, and **workforce management** - foundational concepts for manufacturing simulation.
