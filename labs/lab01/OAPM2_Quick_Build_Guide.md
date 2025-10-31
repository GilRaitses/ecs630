# OAPM2 Quick Build Guide - URGENT

## Model 3-2: Specialized Serial Processing - Loan Application

### IMMEDIATE STEPS TO BUILD

#### 1. Open Arena and Create New Model
- File → New
- Save as "OAPM2_LoanApplication.doe"

#### 2. Build the Flow (5 modules total)

**Module 1: Create**
- Drag Create module
- Name: "Application Arrives"
- Entity Type: "Application"
- Type: Random (Expo)
- Value: 1.25
- Units: Hours

**Module 2: Process (Ali)**
- Drag Process module
- Name: "Ali Checks Credit"
- Action: Seize Delay Release
- Resources: Add "r Ali"
- Delay Type: Expression
- Expression: EXPO(1)
- Units: Hours

**Module 3: Process (Bianca)**
- Drag Process module
- Name: "Bianca Prepares Covenant"
- Action: Seize Delay Release
- Resources: Add "r Bianca"
- Delay Type: Expression
- Expression: EXPO(1)
- Units: Hours

**Module 4: Process (Carl)**
- Drag Process module
- Name: "Carl Prices Loan"
- Action: Seize Delay Release
- Resources: Add "r Carl"
- Delay Type: Expression
- Expression: EXPO(1)
- Units: Hours

**Module 5: Process (Deeta)**
- Drag Process module
- Name: "Deeta Disburses Funds"
- Action: Seize Delay Release
- Resources: Add "r Deeta"
- Delay Type: Expression
- Expression: EXPO(1)
- Units: Hours

**Module 6: Dispose**
- Drag Dispose module
- Name: "Application Departs"
- Record Entity Statistics: Yes

#### 3. Connect All Modules
Connect in sequence: Application Arrives → Ali → Bianca → Carl → Deeta → Application Departs

#### 4. Set Run Parameters
- Run → Setup
- Replication Length: 160
- Time Units: Hours
- Number of Replications: 1

#### 5. Run the Model
- Click Run button
- When complete, click "Yes" to see results
- Export/screenshot the results

### KEY STATISTICS TO CAPTURE
- Total WIP Average/Max
- Total Time in System Average/Max
- Total Waiting Time Average/Max
- Number Processed
- Resource Utilizations (Ali, Bianca, Carl, Deeta)

### TROUBLESHOOTING
- If resources not found: Check spelling of "r Ali", "r Bianca", etc.
- If EXPO(1) error: Make sure Expression is selected as Delay Type
- If no results: Check that Replication Length = 160 Hours

This should take 15-20 minutes to build and run.
