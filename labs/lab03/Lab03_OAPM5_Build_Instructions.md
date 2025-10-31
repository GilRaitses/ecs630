# Lab03 Arena Model Build Instructions - OAPM5

## Model OAPM5: Multi-Part Type Manufacturing with Priority and Machine Cleaning

### Overview
This model introduces **two different part types** with **priority processing** and **machine cleaning cycles**. The system processes wheel hubs and brake rotors with different arrival rates, includes priority queuing for wheel hubs, and requires machine cleaning after every 100 parts.

### Key Advanced Features:
- **Two Part Types**: Wheel Hub (priority) and Brake Rotor (standard)
- **Different Arrival Rates**: Wheel Hub (5 min), Brake Rotor (8 min)
- **Priority Processing**: Wheel hubs get priority at drilling preparation
- **Machine Cleaning**: Required after 100 parts machined (5-10 min, mode 7)
- **Quality Inspection**: 90% pass rate for all parts
- **Performance Analysis**: Bottleneck identification and output optimization

---

## System Description

### **Part Type Specifications:**

#### **Wheel Hub:**
- **Arrival Pattern**: Exponential with mean 5 minutes
- **Priority**: HIGH (gets priority at drilling preparation)
- **Processing**: Same as brake rotor after priority assignment

#### **Brake Rotor:**
- **Arrival Pattern**: Exponential with mean 8 minutes
- **Priority**: STANDARD (normal processing)
- **Processing**: Same as wheel hub but lower priority

### **Process Flow:**
1. **Parts arrive** (two separate streams: Wheel Hub and Brake Rotor)
2. **Drilling Preparation** (2-8 minutes uniform, priority to wheel hubs)
3. **Drilling Machine** (exactly 2.5 minutes, FIFO, cleaning required)
4. **Quality Inspection** (2-4 minutes triangular, mode 3, 90% pass rate)
5. **Exit system** (passed parts only)

### **Machine Cleaning Logic:**
- **Trigger**: After 100 parts machined
- **Duration**: Triangular 5-10 minutes (mode 7)
- **Impact**: Machine unavailable during cleaning

---

## Build Steps

### 1. **Two-Part Type Creation**

#### **1.1 Entity Data Module**
   - Go to **Data Definition** → **Entity**
   - Create two entity types:
     - **Wheel_Hub**
     - **Brake_Rotor**
   - Set different pictures for visual distinction

#### **1.2 Create Module (Wheel Hub)**
   - Drag **Create** module from Basic Process panel
   - Name: "Wheel Hub Arrives"
   - Entity Type: "Wheel_Hub"
   - **Type**: Random (Expo)
   - **Value**: 5
   - **Units**: Minutes
   - **COMMENT**: High-priority parts arrive every 5 minutes on average.

#### **1.3 Create Module (Brake Rotor)**
   - Drag **Create** module from Basic Process panel
   - Name: "Brake Rotor Arrives"
   - Entity Type: "Brake_Rotor"
   - **Type**: Random (Expo)
   - **Value**: 8
   - **Units**: Minutes
   - **COMMENT**: Standard-priority parts arrive every 8 minutes on average.

### 2. **Priority Assignment**

#### **2.1 Assign Module (Wheel Hub)**
   - Drag **Assign** module after Wheel Hub Create
   - Name: "Set Wheel Hub Priority"
   - **Assignments**:
     - Part_Type = 1 (Wheel Hub identifier)
     - Priority = 1 (High priority for queuing)
   - **COMMENT**: Marks wheel hubs for priority processing.

#### **2.2 Assign Module (Brake Rotor)**
   - Drag **Assign** module after Brake Rotor Create
   - Name: "Set Brake Rotor Priority"
   - **Assignments**:
     - Part_Type = 2 (Brake Rotor identifier)
     - Priority = 2 (Standard priority)
   - **COMMENT**: Marks brake rotors for standard processing.

### 3. **Combine Flows and Priority Processing**

#### **3.1 Process Module (Drilling Preparation)**
   - Drag **Process** module from Basic Process panel
   - Name: "Drilling Preparation"
   - Type: Standard
   - Action: **Seize Delay Release**
   - Resources: Add "r Prep Operator" (capacity 2)
   - **Priority**: Use attribute "Priority" (wheel hubs get preference)
   - **Delay Type**: Uniform
   - **Minimum**: 2
   - **Maximum**: 8
   - **Units**: Minutes
   - Report Statistics: Checked
   - **COMMENT**: Two operators available. Wheel hubs processed first when both types waiting.

### 4. **Drilling Machine with Cleaning Logic**

#### **4.1 Variable Data Module (Parts Counter)**
   - Go to **Data Definition** → **Variable**
   - Create variable: "Parts_Machined_Count"
   - **Initial Value**: 0
   - **COMMENT**: Tracks parts processed for cleaning trigger.

#### **4.2 Process Module (Drilling Machine)**
   - Name: "Drilling Machine"
   - Type: Standard
   - Action: **Seize Delay Release**
   - Resources: Add "r Drilling Machine" (capacity 1)
   - **Delay Type**: Constant
   - **Value**: 2.5
   - **Units**: Minutes
   - Report Statistics: Checked
   - **COMMENT**: Processes exactly one part at a time. FIFO after priority assignment.

#### **4.3 Assign Module (Count Parts)**
   - Place after Drilling Machine
   - Name: "Count Machined Parts"
   - **Assignments**:
     - Parts_Machined_Count = Parts_Machined_Count + 1
   - **COMMENT**: Increments counter for cleaning trigger.

#### **4.4 Decide Module (Cleaning Check)**
   - Name: "Check Cleaning Requirement"
   - Type: **2-way by Condition**
   - **Condition**: Parts_Machined_Count >= 100
   - If True: Send to cleaning process
   - If False: Continue to inspection
   - **COMMENT**: Checks if 100 parts processed, triggers cleaning.

### 5. **Machine Cleaning Process**

#### **5.1 Process Module (Machine Cleaning)**
   - Name: "Machine Cleaning"
   - Type: Standard
   - Action: **Seize Delay Release**
   - Resources: "r Drilling Machine" (same machine)
   - **Delay Type**: Triangular
   - **Minimum**: 5
   - **Value (Mode)**: 7
   - **Maximum**: 10
   - **Units**: Minutes
   - Report Statistics: Checked
   - **COMMENT**: Cleaning takes 5-10 min (mode 7). Machine unavailable during cleaning.

#### **5.2 Assign Module (Reset Counter)**
   - Place after Machine Cleaning
   - Name: "Reset Parts Counter"
   - **Assignments**:
     - Parts_Machined_Count = 0
   - **COMMENT**: Resets counter after cleaning for next 100-part cycle.

### 6. **Quality Inspection Process**

#### **6.1 Process Module (Inspection)**
   - Name: "Quality Inspection"
   - Type: Standard
   - Action: **Seize Delay Release**
   - Resources: "r Inspector"
   - **Delay Type**: Triangular
   - **Minimum**: 2
   - **Value (Mode)**: 3
   - **Maximum**: 4
   - **Units**: Minutes
   - Report Statistics: Checked
   - **COMMENT**: Single inspector, triangular 2-4 min (mode 3).

#### **6.2 Decide Module (Pass/Fail)**
   - Name: "Quality Decision"
   - Type: **2-way by Chance**
   - **Percent True**: 90 (90% pass rate)
   - If True: Part passes → Exit
   - If False: Part fails → Dispose (or rework if desired)
   - **COMMENT**: 90% of parts pass inspection and exit system.

### 7. **System Flow Logic**
```
Wheel Hub Arrives → Set Priority → \
                                   → Drilling Preparation → Drilling Machine → Count Parts
Brake Rotor Arrives → Set Priority → /                                            ↓
                                                                              Check Cleaning
                                                                                ↓        ↓
                                                                          (100 parts) (Continue)
                                                                                ↓        ↓
                                                                        Machine Cleaning  ↓
                                                                                ↓        ↓
                                                                          Reset Counter   ↓
                                                                                ↓        ↓
                                                                           Quality Inspection
                                                                                    ↓
                                                                             Pass/Fail Decision
                                                                                    ↓
                                                                              Parts Exit
```

### 8. **Resource Configuration**

#### **8.1 Resource Data Module**
   - **r Prep Operator**:
     - Type: Fixed Capacity
     - Capacity: 2
     - **COMMENT**: Two operators for drilling preparation.
   
   - **r Drilling Machine**:
     - Type: Fixed Capacity
     - Capacity: 1
     - **COMMENT**: Single machine, shared for processing and cleaning.
   
   - **r Inspector**:
     - Type: Fixed Capacity
     - Capacity: 1
     - **COMMENT**: Single inspector for quality check.

#### **8.2 Queue Data Module**
   - **Drilling Preparation.Queue**:
     - Type: **Lowest Attribute Value**
     - Attribute Name: **Priority**
     - **COMMENT**: Wheel hubs (Priority=1) processed before brake rotors (Priority=2).

### 9. **Run Settings**
   - **Run** → **Setup** → **Replication Parameters**
   - **Replication Length**: 5 × 8 × 60 = **2400** minutes (5 eight-hour days)
   - **Time Units**: Minutes
   - **Base Time Units**: Minutes
   - **Number of Replications**: 1
   - **Warm-up Period**: 0
   - **COMMENT**: Simulates 5 full working days to capture cleaning cycles.

---

## Expected Model Behavior

### **Priority Processing:**
- **Wheel hubs** get processed first when both types waiting
- **Brake rotors** wait longer during busy periods
- **Different throughput rates** due to arrival rate differences

### **Machine Cleaning Cycles:**
- **Every 100 parts**: Machine stops for cleaning
- **5-10 minute cleaning** (mode 7 minutes)
- **Temporary throughput reduction** during cleaning
- **Counter resets** after each cleaning

### **Quality Control:**
- **90% pass rate** for both part types
- **10% failure rate** removes parts from system
- **Independent decisions** for each part

---

## Key Performance Indicators

### **Part-Type-Specific Metrics:**
- **Wheel Hub Performance**: Higher priority, potentially lower wait times
- **Brake Rotor Performance**: Standard priority, potentially higher wait times
- **Arrival Rate Impact**: Different processing loads

### **System-Wide Metrics:**
- **Total throughput** (accounting for failures and cleaning)
- **Machine utilization** (including cleaning time)
- **Cleaning frequency** (should occur every ~100 parts)
- **Priority queue effectiveness**

### **Bottleneck Analysis:**
- **Drilling preparation** (2 operators vs demand)
- **Drilling machine** (single unit with cleaning requirements)
- **Quality inspection** (single inspector)

---

## Verification Checklist

- [ ] **Two part types** arrive with different rates (5 min vs 8 min)
- [ ] **Priority queuing** works (wheel hubs processed first)
- [ ] **Machine cleaning** triggers after 100 parts
- [ ] **Parts counter** resets after cleaning
- [ ] **90% pass rate** achieved in inspection
- [ ] **Resource capacities** set correctly
- [ ] **Priority attribute** assigned properly

---

## Troubleshooting

### **Priority Issues:**
1. **Queue Type**: Must be "Lowest Attribute Value" with "Priority" attribute
2. **Priority Assignment**: Wheel Hub=1, Brake Rotor=2 (lower number = higher priority)
3. **Attribute Names**: Must match exactly between Assign and Queue modules

### **Cleaning Logic Issues:**
1. **Counter Logic**: Verify Parts_Machined_Count increments correctly
2. **Condition Check**: Ensure >= 100 comparison works
3. **Reset Logic**: Counter must reset to 0 after cleaning

### **Part Type Issues:**
1. **Entity Types**: Verify two distinct entity types created
2. **Arrival Rates**: Check exponential parameters (5 vs 8 minutes)
3. **Flow Logic**: Ensure both types reach same processes

---

## Analysis Framework

### **Key Questions:**
1. **How effective is priority processing?** (Wheel Hub vs Brake Rotor wait times)
2. **What is the cleaning impact?** (Throughput reduction during cleaning)
3. **Where are the bottlenecks?** (Preparation, drilling, or inspection)
4. **How do arrival rates affect performance?** (5 min vs 8 min intervals)

### **Performance Metrics to Track:**
- **Part-type-specific cycle times**
- **Priority queue effectiveness**
- **Machine cleaning frequency**
- **Resource utilization patterns**
- **Overall system throughput**

---

## Submission Requirements

1. **Model File**: "OAPM5_[YourName].doe"
2. **Results Report**: Focus on priority processing and cleaning cycles
3. **Analysis**: 
   - Part type performance comparison
   - Priority queuing effectiveness
   - Machine cleaning impact analysis
   - Bottleneck identification
4. **Screenshots**: Show priority queuing and cleaning cycles

This model demonstrates **priority queuing**, **part-type differentiation**, **maintenance scheduling**, and **performance optimization** - key concepts for advanced manufacturing simulation.
