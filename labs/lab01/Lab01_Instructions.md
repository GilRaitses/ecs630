### Slide 1: Guided Tour of Arena
Guided Tour of Arena
Chapter 3
© McGraw Hill LLC. All rights reserved. No reproduction or distribution without the prior written consent of McGraw Hill LLC.

### Slide 2: What We’ll Do . . .
What We’ll Do . . .
Start Arena – A Tour of the Application
Load, explore,  and run an existing model
Construct a model from scratch
Use just these basic building blocks in case study to address real operational question
Tour menus, toolbars, drawing, printing
Help system
Options for running and control
2

### Slide 3: Arena Simulation Software
Arena Simulation Software
Arena simulation software is owned and developed by Rockwell Automation (ROK). 
www.rockwellautomation.com and www.arenasimulation.com
Arena is . . .
40-year leader in discrete event simulation
a Windows based application
the GUI (Graphical User Interface) to the SIMAN simulation engine
capable of being automated via its object model using VBA, Python, Visual Studio
Installing the Application
The application can be downloaded from the www.arenasimulation.com web site and installed on your Windows laptop or computer. For end users using Apple products, you will need to install a Windows emulator in order to install the software. 
Questions about installing the software can be directed to arena-support@rockwellautomation.com
3

### Slide 4: Open the Application
Open the Application
Access the text alternative for slide images.
4

### Slide 5: Tour of the Application Begins!
Tour of the Application Begins!
Access the text alternative for slide images.
5

### Slide 6: This is the Arena application
This is the Arena application
Access the text alternative for slide images.
6

### Slide 7: Ribbon Tab Tour
Ribbon Tab Tour
The ribbons contain features of the software related to the name on the tab.
Looking to animate your model, click on the Animate tab
Features related to running your model will be found by selecting the Run tab . . .
On the next few slides, a general overview of each of these tabs will be covered.
Access the text alternative for slide images.
7

### Slide 8: File Tab
File Tab
Access the text alternative for slide images.
8

### Slide 9: Home Tab
Home Tab
Access the text alternative for slide images.
9

### Slide 10: Animate Tab
Animate Tab
Access the text alternative for slide images.
10

### Slide 11: Draw Tab
Draw Tab
Access the text alternative for slide images.
11

### Slide 12: Run Tab
Run Tab
Access the text alternative for slide images.
12

### Slide 13: View Tab
View Tab
Access the text alternative for slide images.
13

### Slide 14: Tools Tab
Tools Tab
Access the text alternative for slide images.
14

### Slide 15: Template Panels
Template Panels
Access the text alternative for slide images.
15

### Slide 16: Modules
Modules
Basic building blocks of simulation model
Two types:  flowchart and data
Different types of modules for different actions, specifications
“Blank” modules: on Project Bar
Add a flowchart module to model by dragging it from Project Bar into flowchart view of model window
Can have many instances of same kind of flowchart module in model
To add a data module: select it (single-click) in Project Bar, edit in spreadsheet view of model window
Only one instance of each kind of data module in model, but it can have many entries (rows) in spreadsheet view
Can use Edit via Dialog – double-click on number in leftmost column
16

### Slide 17: Data Modules 1
Data Modules 1
Access the text alternative for slide images.
17

### Slide 18: Data Modules 2
Data Modules 2
Set values, conditions, etc. for whole model
No entity flow, no connections
Data Definition panel data modules:
Attribute, Entity, Queue, Resource, Variable, Schedule, Set
Data Module Icons in Project Bar will look like little spreadsheets but may have a different color to identify their purpose.  For example, data modules on the Animation panel are going to be orange in color.
To use a data module, select it (single-click) in Project Bar, edit in spreadsheet view
Can edit via dialog – double-click in leftmost column, or right-click and select Edit via Dialog
Double-click where indicated to add new row
Right-click on row, column to do different things
At most one instance of each kind of data module in a model
But each one can have many entries (rows)
18

### Slide 19: Flowchart Modules 1
Flowchart Modules 1
Access the text alternative for slide images.
19

### Slide 20: Flowchart Modules 2
Flowchart Modules 2
Describe dynamic processes
Nodes/places through which entities flow
Typically connected to each other in some way
Discrete Processing panel flowchart module types:
Create, Dispose, Process, Assign, Seize, Delay and Release
Other panels – many other kinds
Two ways to edit
Double-click to open up, then fill out the dialog
Select (single-click) a module type in the model or Project Bar, get all modules of that type in spreadsheet view
20

### Slide 21: Flowchart and Spreadsheet Views
Flowchart and Spreadsheet Views
Model window split into two views
Flowchart view
Graphics
Process flowchart
Animation, drawing
Edit things by double-clicking on them, to access a dialog box
Spreadsheet view
Displays model data directly
Can edit, add, delete data in spreadsheet view
Displays all similar kinds of modeling elements at once
Many model parameters can be edited in either view
Horizontal splitter bar to apportion two views
Access the text alternative for slide images.
21

### Slide 22: Relations Among Modules
Relations Among Modules
Flowchart, data modules related via names for objects
Queues, Resources, Entity types, Variables, Expressions, Sets, . . . many others
Arena keeps internal lists of different kinds of names
Presents existing lists to you where appropriate
Helps you remember names, protects you from typos
All names you make up in a model must be unique across model, even across different types of modules
22

### Slide 23: Browsing through Model 3-1
Browsing through Model 3-1
Open Model 03-01.doe, Book Examples folder
www.mhhe.com/kelton, Student Edition, BookExamples.zip, unzip and put folder where you want on your system
Three flowchart modules
Create, Process, Dispose
Entries in three data modules
Entity, Queue, Resource
Animation objects
Resource animation
Two plots
Some (passive) labels, “art” work
Access the text alternative for slide images.
23

### Slide 24: Create Flowchart Module 1
Create Flowchart Module 1
“Birth” node for entities
Gave this instance of Create-type module the Name Part Arrives to System
If we had other Create modules (we don’t) they’d all have different Names
Double-click on module to open property
Access the text alternative for slide images.
24

### Slide 25: Create Flowchart Module 2
Create Flowchart Module 2
Name – for module (type it in, overriding default)
Entity Type – enter descriptive name
Can have multiple Entity Types with distinct names
Time Between Arrivals area
Specify nature of time separating consecutive arrivals
Type – pull-down list, several options
Value – depends on Type . . . for Random (Expo) is mean
Units – time units for Value
Entities per Arrival – constant, random variable, very general “Expression” (more later . . .)
Max Arrivals – choke off arrivals (from here) after this many arrivals (batches, not entities)
First Creation – time of first arrival (need not be 0)
25

### Slide 26: Editing Flowchart Modules in Spreadsheet View
Editing Flowchart Modules in Spreadsheet View
Alternative to dialog for each instance of a module type
See all instances of a module type at once
Convenient for seeing, editing many things at once
Selecting a module in either flowchart or spreadsheet view also selects it in the other view
Click, double-click fields to view, edit
Right-click in expression fields to get Expression Builder for help in constructing complex expressions with Arena variables (more later . . .)
26

### Slide 27: Entity Data Module
Entity Data Module
A data module, so edit in spreadsheet view only
View, edit aspects of different entity Types in your model (we have just one entity Type, e Part)
Pull-down lists activated as you select fields
Our only edit – Initial Picture for animation
Picked Picture.Blue Ball from default list
Animate> Pictures > Edit Entity Pictures . . . to see or modify
Access the text alternative for slide images.
27

### Slide 28: Process Flowchart Module 1
Process Flowchart Module 1
Represents machine, including:
Resource
Queue
Entity delay time (processing)
Enter Name – r Drilling Center
Type – picked Standard to define logic here rather than in a submodel (more later . . .)
Report Statistics check box at bottom
To get utilizations, queue lengths, queue waiting times, etc.
Access the text alternative for slide images.
28

### Slide 29: Process Flowchart Module 2
Process Flowchart Module 2
Logic area – what happens to entities here
Action
Seize Delay Release – entity Seizes some number of units of a Resource (maybe after a wait in queue), Delay itself there for processing time, then Release units of Resource it had Seized – chose this option
Delay entity (red traffic light) – no Resources or queueing, just sit here for a time duration
Seize Delay (no Release . . . presumably Release downstream)
Delay Release (if Resource had been Seized upstream)
Priority for seizing – lower numbers ⇒ higher priority
Different Action choices could allow stringing together several Process modules for modeling flexibility
Resources – define Resource(s) to be seized, released
Double-click on row to open subdialog
Define Resource Name, Quantity of units underlined to be Seized/Released here
  underlined Not where you say there are multiple Resource units . . . do that in Resource underlined data module
Several Resources present (Add) – entities must first Seize all
29

### Slide 30: Process Flowchart Module 3
Process Flowchart Module 3
Delay Type – choice of probability distributions, constant or general Expression (more later . . .)
Units – time units for delay (don’t ignore)
Allocation – how to “charge” delay in costing (more later . . .)
Prompts on next line – change depending on choice of Delay Type – specify numerical parameters involved
Can also edit in spreadsheet view
Subdialogs (e.g., Resource here) become secondary spreadsheets that pop up, must be closed
30

### Slide 31: Resource Data Module
Resource Data Module
Defining r Drill Press Resource in Process module automatically creates entry (row) for it in Resource data module
Can edit it here for more options
Type – could vary capacity Based on Schedule instead of having a Fixed Capacity
Would define Schedule in Schedule data module . . . later
Capacity (if Type = Capacity) is number of units of this resource that exist
Failures – cause resource to fail according to some pattern
Define this pattern via Failure data module (Data Definition panel) . . . later
Access the text alternative for slide images.
31

### Slide 32: Queue Data Module
Queue Data Module
Specify aspects of queues in model
We only have one, named Drilling Center.Queue (default name, given Process module name)
Type – specifies queue discipline or ranking rule
If Lowest or Highest Attribute Value, then another field appears where you specify which attribute to use
Shared – if this queue will be shared among several resources (later . . .)
Report Statistics – check for automatic collection and reporting of queue length, time in queue
Access the text alternative for slide images.
32

### Slide 33: Animating Resources and Queues 1
Animating Resources and Queues 1
We get queue animation automatically by specifying a Seize in the Process module
Entity pictures line up here in animation
33

### Slide 34: Animating Resources and Queues 2
Animating Resources and Queues 2
Don’t get Resource animation automatically
To add it, use Animate>Pictures>Resource to get to the Resource Picture Placement dialog
Identifier – link to Resource name in pull-down list
Specify different pictures for Idle, Busy states
For pre-defined “art” work, Open a picture library (.plb filename extension)
Scroll up/down on right, select (single-click) a picture on right, select Idle or Busy state on left, then click <<  to copy picture over
To edit later, double-click on picture in flowchart view
Access the text alternative for slide images.
34

### Slide 35: Dispose Flowchart Module
Dispose Flowchart Module
Represents entities leaving model boundaries
Name the module
Decide on Record Entity Statistics (average, maximum time in system of entities exiting here, costing information)
Check boxes for statistics collection and reporting:
Most are checked (turned on) by default
Little or no modeling effort to say yes to these
But in some models can slow execution markedly
Moral – if you have speed problems, clear these if you don’t care
Access the text alternative for slide images.
35

### Slide 36: Connecting Flowchart Modules
Connecting Flowchart Modules
Access the text alternative for slide images.
36

### Slide 37: Dynamic Plots 1
Dynamic Plots 1
Trace variables (e.g., queue lengths) as simulation runs – “data animation”
Disappear after run ends
To keep, save data, postprocess in Output Analyzer . . . later
Animate>Status>Charts and select Plot
Six tabs across top; many options (best just to explore)
Data Series tab – click Add button for each curve to be plotted on same set of axes
In right “Properties” area, enter Name, define Expression
Pull down Build Expression, “+” Current Model Variables and Functions, “+” Queue, Current Number in Queue, select Drilling Center.Queue in Queue Name field pull-down, note Current Expression NQ(Drilling Center.Queue) automatically filled in at bottom, OK button to copy this expression back out
DrawMode – Stairs or PointToPoint
Line/fill color, vertical-axis on left/right
Access the text alternative for slide images.
37

### Slide 38: Dynamic Plots 2
Dynamic Plots 2
Axes tab – choose Time (X) Axis on left
X axis is always simulated time
Scale area on right (“+” to open it) – specify Min/Max, MajorIncrement, AutoScroll (“windows” axis during simulation)
Title on right – type in Text (mention units!), set Visible to True
Axes tab – choose Left Value (Y) Axis on left
Note possibility for a different right Y axis scale for multiple curves
Scale area on right – specify Min/Max, MajorIncrement, usually leave AutoScaleMaximum at True so Y axis scale will automatically adjust to contain whole plot during run
Title on right
Legend tab – clear Show Legend box since we have only one curve, and Y axis is labeled
Other tabs – Titles, Areas, 3-D View . . . just explore
Drop plot in via crosshairs (resize, move later)
Access the text alternative for slide images.
38

### Slide 39: Dressing Things Up
Dressing Things Up
Add drawing objects from Draw ribbon
Similar to other drawing, CAD packages
Object-oriented drawing tools (layers, etc.), not just a paint tool
Add Text to annotate
Control font, size, color, orientation
Access the text alternative for slide images.
39

### Slide 40: Setting Run Conditions
Setting Run Conditions
Run > Settings > Setup menu dialog – seven options
Project Parameters – Title, Name, Project Description, stats
Replication Parameters
Number of Replications
Initialization options Between Replications
Start Date/Time to associate with start of simulation
Warm-up Period (when statistics are cleared)
Replication Length (and Time Units)
Hours per “Day” (convenience for 16-hour days, etc.)
Base Time Units (output measures, internal computations, units where not specified in dialog, e.g. Plot X Axis time units)
Terminating Condition (complex stopping rules)
Tabs for run speed, run control, reports, array sizes, visuals
Terminating your simulation:
You must specify – part of modeling.
Arena has no default termination!
If you don’t specify termination, Arena will usually keep running forever.
Access the text alternative for slide images.
40

### Slide 41: Running the Model
Running the Model
Plain-vanilla run:  Click  the Run button from Home or Run ribbons (like audio/video players)
When the model run starts, Arena will check the model for errors.  
Once Arena enters run mode — can move around but not edit
When model run is complete you will be asked to view the reports. 
Click the end button to end the model run.
Can pause run with the pause button or hit ESC.
Access the text alternative for slide images.
41

### Slide 42: Moving Around in the Model Window
Moving Around in the Model Window
Access the text alternative for slide images.
42

### Slide 43: Named Views – Organizing and Presenting
Named Views – Organizing and Presenting
Access the text alternative for slide images.
43

### Slide 44: Opening Models
Opening Models
Click > Open or double-click Model 03-01.doe
Book example models: www.mhhe.com/kelton, Student Edition, BookExamples.zip, put where you want on your computer so that you can browse to them easily
Access the text alternative for slide images.
44

### Slide 45: Viewing Reports
Viewing Reports
Click Yes in Arena box at end of run
Opens Excel and displays the model reports
Remember to close all reports windows before future runs or save the Excel file under a different name if you wish to reference it later
Times are in Base Time Units for model
45

### Slide 46: Arena Report
Arena Report
Access the text alternative for slide images.
46

### Slide 47: Types of Statistics Reported
Types of Statistics Reported
Many output statistics are one of three types:
Discrete Time Statistics (Tally) – avg., max, min of a discrete list of numbers
Used for discrete-time output processes like waiting times in queue, total times in system
Continuous-Time (Time persistent) – time-average, max, min of a plot of something where x-axis is continuous time
Used for continuous-time output processes like queue lengths, WIP, server-busy functions (for utilizations)
Counter – accumulated sums of something, usually just nose counts of how many times something happened
Often used to count entities passing through a point in model
47

### Slide 48: Build It Yourself – Best Way to Learn
Build It Yourself – Best Way to Learn
Build same model from scratch – details in text
Handy user-interface tricks:
Right-click in an empty spot in flowchart view – small box of options, including Repeat Last Action . . . useful in repetitive editing like placing lots of same module type
Ctrl+D or Ins key – duplicates whatever’s selected in flowchart view, offsetting it a bit . . . drag elsewhere, edit
Open new (blank) model window – name it, save it, maybe maximize it
Attach modeling panels you’ll need to Project Bar if not there
48

### Slide 49: Build It Yourself
Build It Yourself
Access the text alternative for slide images.
49

### Slide 50: Case Study:  Specialized Serial vs. Generalized Parallel Processing
Case Study:  Specialized Serial vs. Generalized Parallel Processing
Loan applications go through four steps
Check credit, prepare covenant, price loan, disburse funds
Each step takes expo (1 hour)
Applications arrive with expo (1.25 hour) interarrival times
First application arrives at time 0
Run for 160 hours
Watch avg, max no. applications in process (WIP); avg, max total time in system of applications
Four employees, each can do any process step
Serial specialized processing or generalized parallel processing?
What’s the effect of service-time variability on decision?
50

### Slide 51: Case Study: Model 3-2, Specialized Serial Processing
Case Study: Model 3-2, Specialized Serial Processing
File Model 03-02.doe
Create module – similar to Model 3-1 except expo mean, time units
Set Entity Type to e Application
All files in book:  www.mhhe.com/kelton, Student Edition, BookExamples.zip
Access the text alternative for slide images.
51

### Slide 52: Case Study:  Model 3-2, Specialized Serial Processing
Case Study:  Model 3-2, Specialized Serial Processing
Four Process modules – similar to Model 3-1
Four separate Resources
Expo process time:  Expression (via Expression Builder)
Dispose module similar to Model 3-1
Default entity picture (report) is OK
Default Resource animations almost OK
Make Idle picture same as Busy
Select correct Resource name in Identifier field
Queue, Resource data modules OK
Plot WIP – use Expression builder to find EntitiesWIP(Application)
Fixed Y axis max = 25 to compare with next three models
Fill in Run > Setup, lengthen queue animations
52

### Slide 53: Case Study: Model 3-3, Generalized Parallel Processing 1
Case Study: Model 3-3, Generalized Parallel Processing 1
File Model 03-03.doe
Create, Dispose, plot, Run > Settings > Setup almost same
Just change some labels, etc.
Only one place to queue, but processing times are longer – sum of four IID expo (1 hour) times
Access the text alternative for slide images.
53

### Slide 54: Case Study:  Model 3-3, Generalized Parallel Processing 2
Case Study:  Model 3-3, Generalized Parallel Processing 2
Replace four earlier Process modules with just a single Process module
One Resource (r Loan Officer), but four units of it
Still set Quantity to 1 since application just needs 1 officer
Delay type – Expression
EXPO(1) + EXPO(1) + EXPO(1) + EXPO(1)
Why not
Modify Resource Animation for four units
Open Model 3-2 Resource Animation to get Resource Picture Placement window, open Idle picture
Duplicate white square three times, realign; copy to Busy
In model window, double-click Seize Area, then Add three
Still not completely accurate animation (order) – need Sets
54

### Slide 55: Case Study: Compare Model 3-2 vs. 3-3
Case Study: Compare Model 3-2 vs. 3-3
Model | Total WIP Avg. | Total WIP Max. | Total Time in System Avg. | Total Time in System Max. | Total Waiting Time        Avg. | Total Waiting Time       Max. | Number Processed | Avg. Utilization
3-2 (serial) | 12.39 | 21 | 16.08 | 27.21 | 11.98 | 22.27 | 117 | 0.78
3-3 (parallel) | 4.61 | 10 | 5.38 | 13.73 | 1.33 | 6.82 | 135 | 0.87
Caution:  This is from only one replication of each configuration, so there’s output variability
Are differences statistically significant?  (Exercise 6-19)
55

### Slide 56: Case Study: Effect of Task-Time Variability 1
Case Study: Effect of Task-Time Variability 1
Is parallel always better than serial under any conditions?
Many aspects could matter
Focus on task-time variability
Now, each task time
expo (1 hour)
Highly variable distribution
In serial config., just one large task time congests greatly
In parallel config. it would also congest, but probably not by as much since other three tasks are probably not all large too
Other extreme – each task time is exactly 1 hour
Leave interarrival times as expo (1.25 hours)
Models 3-4 (serial), 3-5 (parallel) – alter Process modules
56

### Slide 57: Case Study: Effect of Task-Time Variability 2
Case Study: Effect of Task-Time Variability 2
For constant service, parallel improvement appears minor
Maybe not even statistically significant (Exercise 6-19)
Some further questions
In parallel, work is integrated/generalized, so would it be slower per task? (Exercises 3-13, 6-20)
Effect of worker breaks? (Chapters 4, 5)
Differences statistically significant? (Exercises 6-19, 6-20)
A table summarizes the results from four Scenarios of the Loan-Processing Model. Column 1 with no header lists the row headers, expo service, and constant service, respectively.
 | Model | Total WIP Avg. | Total WIP Max. | Total Time in System Avg. | Total Time in System Max. | Total Waiting Time Avg. | Total Waiting Time Max. | Number Processed | Avg. Utilization
Expo service | 3-2 (serial) | 12.39 | 21 | 16.08 | 27.21 | 11.98 | 22.27 | 117 | 0.78
 | 3-3 (parallel) | 4.61 | 10 | 5.38 | 13.73 | 1.33 | 6.82 | 135 | 0.87
Constant service | 3-4 (serial) | 3.49 | 12 | 5.32 | 11.38 | 1.32 | 7.38 | 102 | 0.65
 | 3-5 (parallel) | 3.17 | 11 | 4.81 | 10.05 | 0.81 | 6.05 | 102 | 0.66
57

### Slide 58: End of Main Content
End of Main Content
© McGraw Hill LLC. All rights reserved. No reproduction or distribution without the prior written consent of McGraw Hill LLC.

### Slide 59: Accessibility Content: Text Alternatives for Images
Accessibility Content: Text Alternatives for Images
59

### Slide 60: Open the Application - Text Alternative
Open the Application - Text Alternative
Return to parent-slide containing images.
The first column in the start menu shows the categories icon at the top and five quick icons at the bottom. From bottom to top, these are power off, settings, photos, documents, and the profile. The second column shows lists of flies and apps in the Alphabets P and R. In the Alphabet R column, the folder named Rockwell software is selected and nine files from it are displayed. The third column shows ten Windows applications such as Office, Word, One Note, Outlook, PowerPoint, Microsoft Edge, Microsoft Store, and Arena. In the taskbar, the Windows start icon, a search bar, a small circle icon, a playlist icon, a file explorer icon, an Outlook icon, and a Microsoft Edge icon are displayed. The first callout box pointed toward the Start button reads, 1. After the software is installed select the Start menu. The second callout box pointed toward the Arena application reads, Open the Rockwell Software folder and click on the Arena application to open it.
Return to parent-slide containing images.
60

### Slide 61: Tour of the Application Begins! - Text Alternative
Tour of the Application Begins! - Text Alternative
Return to parent-slide containing images.
The blocks on the top from left to right, read, application window, ribbons, template panels, and help. The template panels gets divided into two blocks, flowchart modules, and data modules.
Return to parent-slide containing images.
61

### Slide 62: This is the Arena application - Text Alternative
This is the Arena application - Text Alternative
Return to parent-slide containing images.
Ribbon of the Arena Application Window is at the top side. A message box contains test Ribbon Tabs is pointing towards Ribbon. Project Bar is below the ribbon. Model tab labeled Model 03-01.doe is opened. Assign, Clone, Delay, Dispose, Go to Label, Pick Station, Release, Service, Assign Attribute, Create, Dispense, Gather, Label, Process, Route, Station are the icons inside the Template Panel. Model window spreadsheet view is on the lower side of the model window. A status bar is on the lower side of the screen grab.
Return to parent-slide containing images.
62

### Slide 63: Ribbon Tab Tour - Text Alternative
Ribbon Tab Tour - Text Alternative
Return to parent-slide containing images.
The quick access toolbar has several icons including play, save, undo, and redo, followed by the customized button. The eight tabs are File (highlighted red), Home (selected), Animate, Draw, Run, View, Tools, and Developer. Seven ribbon groups are labeled Template, Clipboard, Navigation, Connections, Editing, Run, and Help and Manuals. Template has two icons labeled Attach and Detach. Clipboard has three icons labeled Paste, Cut, Copy. Navigation has five icons labeled Select and zoom, zoom percentage (selected as 58 percent), Submodel, Find, Replace, and Layers. Connections has an icon labeled Connect, with three checkboxes for Auto-connect, Smart connect, and connector Arrows. The first two checkboxes are selected and the third checkbox is not selected. Editing has three icons labeled Select All, Expression Builder, and Arrange. Run has three icons of Check model, review errors, and a player. Help and Manuals has three icons of Arena Help, Release Notes, and Arena Product Manuals. A callout box pointed toward the customized button reads, Quick Access toolbar. A callout box pointed at the developer tab reads, Ribbon tabs. A callout box pointed toward the editing group reads, Ribbon groups.
Return to parent-slide containing images.
63

### Slide 64: File Tab - Text Alternative
File Tab - Text Alternative
Return to parent-slide containing images.
It is a vertical rectangular-shaped shaded block. The following options are enlisted in it New, Open, Info, Save, Save As, Print, Share, Close, Browse SMART, Brows Examples, Accounts, and Options. The message box is pointing toward the Return icon, and it contains the following text Return to Arena applications. The message box is pointing toward the option Open, and it contains the following text Open existing models. The message box is pointing toward the option New, and it contains the following text click here to open the new model window. The message box is pointing toward the option Print, and it contains the following text Options to save, print, share or close the active model. The message box is pointing toward the option Browse SMART, and it contains the following text SMARTS are small example models for learning various concepts in Arena.
Return to parent-slide containing images.
64

### Slide 65: Home Tab - Text Alternative
Home Tab - Text Alternative
Return to parent-slide containing images.
The message box is pointing toward the Home tab and it contains the following information The Home tab is Where commonly used functions are organized. The message box is pointing toward the Connect tab and it contains the following information Connections is where you find the Connect button and Settings for how it can be customized.  The message box is pointing toward the Clipboard group and it contains the following information Clipboard is where you will find the Cut and Paste Functions. The message box is pointing toward the Navigation group and it contains the following information Navigation is where you will search and replace functions found are found as well as methods of locating logic. The message box is pointing toward the Help and Manuals group and it contains the following information Have a question? Arena Help and Product Manuals are found here.
Return to parent-slide containing images.
65

### Slide 66: Animate Tab - Text Alternative
Animate Tab - Text Alternative
Return to parent-slide containing images.
The quick access toolbar has three icons for save, undo, and redo, respectively, followed by the customized button. Six ribbon groups are labeled Status, Waiting, Pictures, Locations, Paths, and Object, respectively. Status has five icons labeled Clock, Date, Scoreboard, Variable, and Charts. Waiting has four icons labeled Queue, Storage, Seize, and Parking. Pictures has five icons labeled Resource, Global, Entity, Transporter, and Edit Entity Pictures. Locations have two icons labeled Station and Intersection. Paths have six icons labeled Route, Segment, Distance, Network Link, Animated Network Link, and Promote Path. The Promote path icon is blurred. The object has an icon labeled, Arrange.
Return to parent-slide containing images.
66

### Slide 67: Draw Tab - Text Alternative
Draw Tab - Text Alternative
Return to parent-slide containing images.
The quick access toolbar has three icons for save, undo, and redo, respectively, followed by the customized button. Five ribbon groups are Object, Color, Characteristic, Tools, and view. Object has eight icons labeled Line, Polyline, Arc, Bezier Curve, Box, Polygon, Ellipse, and Text. Color has four icons labeled Line, Fill, Text, and Window Background. Characteristic has five icons labeled Line Width, Line Style, Arrow Style, Line Pattern, and Fill Pattern. Tools has Five icons labeled Show Dimensions, DXF Import, Open color Palette, and Save Color Palette, and Arena symbol factory. View has an icon labeled arrange.
Return to parent-slide containing images.
67

### Slide 68: Run Tab - Text Alternative
Run Tab - Text Alternative
Return to parent-slide containing images.
The quick access toolbar has three icons for save, undo, and redo, respectively, followed by the customized button. Four ribbon groups are labeled Interaction, Debug, Visualization, and Settings. Interaction has three icons, a player, Check Model, and Review Errors. The check model icon has a checkbox and is checked. Debug has five icons labeled Command, SIMAN, Break on Module, Debug Bar, and Runtime Elements bar. Debug bar and the Runtime Elements Bar have checkboxes and are unchecked. Visualization has three icons labeled Animate Connectors, Animate At Desktop Color Depth, and Highlight Active Module. All three of them have checkboxes and the last one for highlight active module is unchecked. Settings have three icons labeled Batch Run (No Animation), a slider, and Setup. The batch run icon has a checkbox and is unchecked.
Return to parent-slide containing images.
68

### Slide 69: View Tab - Text Alternative
View Tab - Text Alternative
Return to parent-slide containing images.
The quick access toolbar has three icons for save, undo, and redo, respectively, followed by the customized button. Seven group tabs are labeled Zoom, Views, Show, Select, Settings, Data Tips, and Layers. Zoom has five icons labeled Zoom Out, Zoom In, Zoom Factor, 75 percent, and Select and Zoom. Views have four icons labeled Submodel, Views, Named Views, and Split Screen with a checked checkbox. Show has seven checkboxes labeled project Bar, Status Bar, Animate Bar, Rulers, Grid, Guides, and Page Breaks. The checkboxes for Project bar, status bar, animate bar, and guides are checked. Select has two icons labeled Snap to Grid, and Glue to Guides, both with checkboxes that are checked. Settings have one icon labeled Grid and Snap Settings. Data Tips has two icons labeled Show Default Descriptions, and Show User-Specified Descriptions. Both have checkboxes, which are checked. Layers have an icon labeled Layers.
Return to parent-slide containing images.
69

### Slide 70: Tools Tab - Text Alternative
Tools Tab - Text Alternative
Return to parent-slide containing images.
The quick access toolbar has three icons for save, undo, and redo, respectively, followed by the customized button. Six group tabs are labeled Visualization, analysis, review, integration, visual basic, and macros, respectively. Visualization has icons labeled Arena visual designer, insert new object, and links. Links icon is blurred. Analysis has four icons labeled OptQuest for Arena, process analyzer, input analyzer, and multi-computing. Review has an icon labeled model report. Integration has five icons labeled export model to database (blurred), import model from database (blurred), export summary statistics to CSV file, module data transfer, and custom add-ins (blurred). Visual basic has icons labeled visual basic, unchecked design mode checkbox, and insert new control. Macros shows an icon for macros.
Return to parent-slide containing images.
70

### Slide 71: Template Panels - Text Alternative
Template Panels - Text Alternative
Return to parent-slide containing images.
The project bar consists of data definition and discrete processing templates. The flowchart modules with the icons are Assign, Assign attribute, Clone, Create, Delay, Disperse, Dispose, Gather, Go to Label, Label, Pickstation, Process, Release, Route, Seize, and Station, respectively. Reports and Navigate panels are given at the bottom of the project bar. Other options like Decisions, Grouping, Input Output, Animation, and Material Handling are displayed on top of it. A callout box from the project bar reads, The project bar hosts Arena's Template Panels when they're attached. A callout box from the template panel reads, Click on the Template Panel title to reveal its contents. A callout box from the icon of Process module reads, Flowchart Modules define the logical flow of entities and are dragged and dropped into the model window to define the actions of the system. A callout box from the navigate panel reads, Reports are accessed only after the model is run. The Navigate panel is another method of moving around the model window and displays the model hierarchy and views.
Return to parent-slide containing images.
71

### Slide 72: Data Modules 1 - Text Alternative
Data Modules 1 - Text Alternative
Return to parent-slide containing images.
The screen grab shows the panel Data Definition given below the project bar. The data elements in the project bar are Activity area, Attribute, Entity, Expression, Failure, Queue, Resource, Schedule, Sequence, Set, StateSet, Station data, and Variable respectively. A callout box reads, data modules define the various process elements. They cannot be dragged into the model window and are viewed in the spreadsheet view.
Return to parent-slide containing images.
72

### Slide 73: Flowchart Modules 1 - Text Alternative
Flowchart Modules 1 - Text Alternative
Return to parent-slide containing images.
The project bar consists of data definition and discrete processing templates. The flowchart modules with the icons are Assign, Assign attribute, Clone, Create, Delay, Disperse, Dispose, Gather, Go to Label, Label, Pickstation, Process, Release, Route, Seize, and Station, respectively. A block labeled create 1 is shown in the model 1 window. A dotted bi-directional arrow points at the create module icon in the project bar panel, and create 1 block in the model 1 window. A callout box pointed at the create module icon in the project bar panel reads, flowchart modules are those modules, which are dragged from the panel and dropped into the model window.
Return to parent-slide containing images.
73

### Slide 74: Flowchart and Spreadsheet Views - Text Alternative
Flowchart and Spreadsheet Views - Text Alternative
Return to parent-slide containing images.
The project bar consists of data definition and discrete processing templates. The flowchart modules with the icons are Assign, Assign attribute, Clone, Create, Delay, Disperse, Dispose, Gather, Go to Label, Label, PickStation, Process, Release, Route, Seize, and Station, respectively. A flowchart made up of three blocks is shown in the model 1 window. The blocks from left to right read, Create 1, Delay 1, and Dispose 1. A callout box pointed at the create 1 block in the flowchart reads, When the module is selected its module dialog settings are visible in the Spreadsheet view. The spreadsheet view shows ten columns and two rows. The first column has no header and shows the row header, number 1. Columns 2 through 10 are labeled, Name, Entity Type, Type, Value, Units, Entities per Arrival, Max Arrivals, First Creation, and comment, respectively. The data in row 2 for columns 2 through 9 is as follows: Create 1, Entity 1, Random (Expo), 1 (selected), Hours, 1, Infinite, and 0.0. The last column labeled comment is blank. A callout box pointed toward the first row reads, Splitter bar can be moved up or down based on user preference. A callout box pointed toward 1 in the column labeled value reads, Spreadsheet View.
Return to parent-slide containing images.
74

### Slide 75: Browsing through Model 3-1 - Text Alternative
Browsing through Model 3-1 - Text Alternative
Return to parent-slide containing images.
A simple processing system is displayed. Block diagram showing three blocks namely Part Arrives to System, Drilling Centre, Part Leaves System from left. 7, 2, 5 are the numbers mentioned below the blocks respectively. A pop-up if Arena is displayed with two command buttons labeled Yes, No. Follow text is written on the pop-up, The simulation has run to completion. Would you like to see the results?.
The vertical axis is labeled as Queue Length and the horizontal axis is labeled as Time (Minutes). The values along the vertical axis range from 0 through 3 in increments of 1. The values along the horizontal axis range from 0 through 20 in increments of 5. The curve starts approximately from (0, 0) and passes approximately through (2, 1), (5, 3), (12.5, 1), and so on, and ends approximately at (20,1).
For the second graph, the vertical axis is labeled as Number Busy and the horizontal axis is labeled as Time (Minutes). The values along the vertical axis range from 0 through 2 in increments of 1. The values along the horizontal axis range from 0 through 20 in increments of 5. The square wave-like curve starts from (0, 1) and passes through (17, 1), (18.5, 0), and (18.5, 1). The data points are approximate.
Return to parent-slide containing images.
75

### Slide 76: Create Flowchart Module - Text Alternative
Create Flowchart Module - Text Alternative
Return to parent-slide containing images.
The Name is Part Arrives to System. Entity Type is e Part. The Time Between Arrivals section has three options Type is Random (expo), Value is 5, and Units are Minutes. Entities per Arrival is 1, Maximum Arrivals are infinite, and First Creation is 0.0. The Comment section is empty. Three command buttons are OK, Cancel, and Help.
Return to parent-slide containing images.
76

### Slide 77: Entity Data Module - Text Alternative
Entity Data Module - Text Alternative
Return to parent-slide containing images.
The columns are labeled as Entity Type, Initial Picture, Holding Cost or Hour, Initial VA Cost, Initial NVA Cost, Initial Waiting Cost, Initial Train Cost, Initial Other Cost, Report Statistics, and Comment respectively. The data in the second row is as follows Entity Type - ePart, Initial Picture - Picture blue ball, Holding Cost or Hour - 0.0, Initial VA Cost - 0.0, Initial NVA Cost - 0.0, Initial Waiting Cost - 0.0, Initial Train Cost - 0.0, Initial Other Cost - 0.0, Report Statistics -  correct mark, respectively.
Return to parent-slide containing images.
77

### Slide 78: Process Flowchart Module - Text Alternative
Process Flowchart Module - Text Alternative
Return to parent-slide containing images.
Name is mentioned as drilling center, type is selected as standard.  the logic section has some options labeled as action, priority, and resources.  add, edit,  and delete are the buttons on the right side of the resources drop-down list. On the lower side, there are some options such as delay type, units and unit allocation. Delay type is selected as triangular, units are selected as minutes and allocation is selected as a value-added. The minimum value is selected as a one, the value is selected as a three and the maximum value is related as a 6. There is one option button labeled report statistics selected. Below the options button, a text box labeled comment is empty. 3 command buttons okay, cancel, and help are shown at the lower right corner of the screen grab.
Return to parent-slide containing images.
78

### Slide 79: Resource Data Module - Text Alternative
Resource Data Module - Text Alternative
Return to parent-slide containing images.
In the spreadsheet, the first column has no header and shows the row header, number 1. Columns 2 through 10 are labeled Name, Type, Capacity, Busy per Hour, Idle per hour, Per use, StateSet Name, Failures, Report Statistics, and Comment, respectively. The data in the second row is as follows. Name, r Drill Press; Type, Fixed Capacity; Capacity, 1; Busy per Hour, 0.0; Idle per hour, 0.0; Per use, 0.0; StateSet name, blank; Failures, 0 rows; Report Statistics, a selected checkbox; comment, blank. Below the second row, the text reads, Double-click here to add a new row. Below the spreadsheet, the Resource dialog box is displayed. The Name selected is r Drill Press. The Type selected is Fixed capacity. The capacity selected is 1. Costs have three options, Busy per Hour, Idle per Hour, and Per Use, respectively. The value in each of the three options is 0.0. StateSet Name has a blank drop-down list box. The failures textbox reads, <End of the list> (selected). Three buttons, Add, Edit, and Delete are on the right side of the textbox. Below, a checkbox labeled Report Statistics is selected. The Comment section has a blank textbox. Three buttons OK, Cancel, and Help, with the Ok button selected are displayed at the right bottom. Two arrows, one pointed toward the spreadsheet and the other pointed toward the dialog box are jointly labeled, You can edit via the spreadsheet view or the dialog.
Return to parent-slide containing images.
79

### Slide 80: Queue Data Module - Text Alternative
Queue Data Module - Text Alternative
Return to parent-slide containing images.
The columns are labeled as follows Name, Type, Shared,  Report Statistics, and Comment respectively. The data in the second row is as follows, Name - Drilling center. Queue, Type - First in first out, Report Statistics - correct mark respectively.
Return to parent-slide containing images.
80

### Slide 81: Animating Resources and Queues 2 - Text Alternative
Animating Resources and Queues 2 - Text Alternative
Return to parent-slide containing images.
In the second tab, an image icon followed by the text reads Model 03-01.doe - Resource Picture Placement.
On the top left of the dialogue box Identifier: is mentioned. A rectangular box in front of the text identifier contains a text that reads r Drill Press and at the right end of the box down pointing arrowhead is mentioned. Below the identifier State: is mentioned followed by the blank rectangular box and at the right end, the box has a down-pointing arrowhead. The text Picture ID: is mentioned below the state followed by the blank rectangular box containing the cursor. There are 5 small shaded boxes are mentioned below it. The text Add is mentioned in the first box. The text Copy is mentioned in the second box and the text is blurred. The text delete is mentioned in the third box and the text is blurred. A black up-pointing triangle is indicated in the fourth box. A black down-pointing triangle is indicated in the fifth box. On the right side of these boxes, another box is indicated and it consists of 4 shaded small boxes. The texts Idle, Busy, Inactive, and Failed are mentioned along with the icons inside the 4 shaded boxes.
Size Factor: 2.25 and blurred text Auto Scale with the blank square are indicated.
The text at the top right of the dialogue box reads Current Library: in the first line. The second line reads c:\ . . . \general.plb. Below the text 7 shaded small boxes are arranged vertically. Add is mentioned in the first box. Copy is mentioned in the second box. Delete is mentioned in the third box. The fourth box has two parts. In the first part, two left-pointing arrowheads are indicated and in the second part, two right-pointing arrowheads are indicated. New is mentioned in the fifth box. Open . . . is mentioned in the sixth box. Save . . . is mentioned in the seventh box. 5 icons are mentioned vertically on the right side of these small boxes.
The text at the middle of the dialog box reads Select Draw a right-pointing arrowhead Arena Symbol Factory to open and view additional images. To use Symbol Factory, select an image from any category. drag it from the Preview pane and drop it into the Arena Picture Editor.
A box at the bottom of the window is labeled Effects and the text inside the box reads When multiple pictures are defined for the same resource state, use this simulation timing to animate a series of pictures:. Two small boxes are mentioned below the text. The first box is blank and consists of a down-pointing arrowhead at the right end. The second box is shaded and the text inside the box reads Hours and a down-pointing arrowhead is indicated at the right end. The text on the outside of the box reads per picture. The text below the first box reads Rotate By Expression: and Seize Area. There is a checkmark with the text Seize Area. A blank small box with the down-pointing arrowhead at the right end is indicated below the second box.
The dialog window has 3 horizontal shaded boxes at the bottom left of the window and the text in the boxes reads OK, Cancel, and Help respectively.
Return to parent-slide containing images.
81

### Slide 82: Dispose Flowchart Module - Text Alternative
Dispose Flowchart Module - Text Alternative
Return to parent-slide containing images.
The Name is Part Leaves to System. Record Entity statistics is mentioned with the correct mark. The Comment section is empty. Three command buttons are OK, Cancel, and Help.
Return to parent-slide containing images.
82

### Slide 83: Connecting Flowchart Modules - Text Alternative
Connecting Flowchart Modules - Text Alternative
Return to parent-slide containing images.
The quick access toolbar has three icons for save, undo, and redo, respectively, followed by the customized button. The eight tabs are File (highlighted red), Home (selected), Animate, Draw, Run, View, Tools, and Developer. Seven ribbon groups are labeled Template, Clipboard, Navigation, Connections, Editing, Run, and Help and Manuals. Template has two icons labeled Attach and Detach. Clipboard has three icons labeled Paste, Cut, Copy. Navigation has five icons labeled Select and zoom, zoom percentage (selected as 68 percent), Submodel, Find, Replace, and Layers. Connections has an icon labeled Connect, with three checkboxes for Auto-connect, Smart connect, and connector Arrows. First two checkboxes are selected and the third checkbox is not selected. Editing has three icons labeled Select All, Expression Builder, and Arrange. Run has three icons of Check model, review errors, and a player. Help and Manuals has three icons of Arena Help, Release Notes, and Arena Product Manuals. A callout box pointed toward the connect icon reads, Connect: Click the connect button to connect modules manually. A callout box pointed at the Auto-Connect checkbox reads, Auto-Connect: Automatically will connect modules as you drag and drop them in the model window, so long as the last module placed is selected. A Right click menu is displayed below the ribbon and has the following commands, Connect (selected), Enable Connect Mode, Repeat Last Action, Find, Replace, Undo, Redo, Paste, Select All, View, and Run. Some of the commands are blurred, they are repeat last action, find, replace, and redo. A callout box pointed at the Connect in the right-click menu reads, Right-clicking on the mouse within the model window will provide you with the option to make a single connection with Connect, or you can go into a Connect Mode that allows you to manually make multiple connections using Enable Connect Mode.
Return to parent-slide containing images.
83

### Slide 84: Dynamic Plots 1 - Text Alternative
Dynamic Plots 1 - Text Alternative
Return to parent-slide containing images.
The cross mark is mentioned at the top right corner of the window. The first part has six tabs and the first tab is mentioned in the window. The six tabs are labeled Data Series, Axes, Titles, Areas, Legend, and 3-D View respectively. The text Data Series: is mentioned below the tab. A rectangle box below the text data series consists of a checkmark in the box followed by the text Queue Length. There are two shaded boxes below the rectangle box and the text in the shaded box reads Add and Remove. The letter A and R in the text Add and Remove has an underscore. The Black up-pointing and down-pointing arrows are mentioned in the two shaded boxes at the top right corner of the rectangular box. There is a rectangular box at the right of the arrows. The text above the box reads Queue Length Properties: is mentioned. The box consists of two tables. The heading for the first table reads Source Data. The first table consists of two rows and two columns. In the first column, Name is mentioned on the first row, and Expression is mentioned on the second row and it is shaded. In the second column, Queue Length is mentioned in the first row and NQ(Drilling Center Queue) is mentioned in the second row. The down-pointing arrowhead is mentioned at the right end of the second row in the second column. The table labeled Fill has 4 rows and 4 columns. In the first column Pattern, ForeColor, BackColor, and GradientBalance are mentioned in 4 rows respectively. In the second column, a blank square followed by (No Fill), a shaded black square followed by RGB (128,0,0), a blank square followed by RGB (255, 255, 255), and 50 are mentioned in 4 rows respectively. The text at the end of the box reads Line. An up-pointing arrowhead is indicated at the top right corner of the box and a down-pointing arrowhead is indicated at the bottom right corner of the box. A small shaded rectangle is indicated between these two arrowheads. The text in the other shaded box reads Source Data\Expression in the first line and the text is in bold letters. The text in the second line reads The simulation expression to monitor and plot.
The second part of the dialog window is labeled Sample. A plot is shown on a graph of Queue Length along the vertical axis against Time (Minutes) along the horizontal axis. The values along the vertical axis range from 24 through 80 in increments of 8. The values along the horizontal axis range from 0 through 20 in increments of 5. The curve is like a square wave, which starts from point (0, 46) and ends at (20, 72). The approximate points in the square wave include (2, 46), (4.5, 52), (7, 52), (11, 44), (15, 55), and (20, 72).
The dialog window has 3 horizontal shaded boxes at the bottom left of the window and the text in the boxes reads OK, Cancel, and Help respectively. The letter O in Ok, C in Cancel, and H in help have an underscore.
Return to parent-slide containing images.
84

### Slide 85: Dynamic Plots 2 - Text Alternative
Dynamic Plots 2 - Text Alternative
Return to parent-slide containing images.
The cross mark is mentioned at the top right corner of the window. The first part has six tabs and the first tab is mentioned in the window. The six tabs are labeled Data Series, Axes, Titles, Areas, Legend, and 3-D View respectively. The text Axes: is mentioned at the top left of the window below the tab. A rectangle box below the text axes consists of 3 check marks in the box followed by the texts Time (X) Axis, Left Value (Y) Axis, and Right Value (Y) Axis respectively and the first text is shaded. There is a box at the top right of the window. The text above the box reads Time (X) Axis Properties:. Inside the box, 3 boxes with positive signs are followed by the texts Gridlines, Labels, and Line respectively. The negative enclosed by the small square box followed by the text Scale is indicated below the text Line. Below the scale, a table is indicated. The table consists of 7 rows and 2 columns. In the first column the texts Minimum, Maximum, Major Increment, Minor Count, AutoScroll, AutoScrollPercent, and Reversed are mentioned in 7 rows respectively. In the second column 0, 20, 5, 0, True, 100, and False are mentioned in 7 rows respectively. Below the table, a negative sign enclosed with the small box followed by the text Title is indicated. Another table is mentioned below it and it has 4 rows and 2 columns. In the first column Color, Font, Text, and Visible are mentioned in 4 rows respectively. In the second column, a shaded black square followed by RGB (0,0,0), Times New Roman 10pt, Time (Minutes), and True is mentioned in 4 rows respectively. The text in the new box reads Line.
The second part of the dialog window is labeled Sample. A plot is shown on a graph of Queue Length along the vertical axis against Time (Minutes) along the horizontal axis. The values along the vertical axis range from 0 through 80 in increments of 8. The values along the horizontal axis range from 0 through 20 in increments of 5. The curve is like a square wave and it is starting from the point (0, 84) and the curve ends at ( 20, 70). The approximate points in the square wave are (3, 76), (4.5, 30), (7, 0), (9, 76), (11.5, 38), (13.5, 36), (14, 46), and (17.5, 70).
The dialog window has 3 horizontal shaded boxes at the bottom left of the window and the text in the boxes reads OK, Cancel, and Help respectively. The letter O in Ok, C in Cancel, and H in help have an underscore.
At the top left of the window, Charts are mentioned in the shaded box. A black down-pointing small triangle is indicated below the text charts. Histogram, Level, and Plot are mentioned below the charts with the respective icons. A shaded arrow points toward the text plot.
Return to parent-slide containing images.
85

### Slide 86: Dressing Things Up - Text Alternative
Dressing Things Up - Text Alternative
Return to parent-slide containing images.
The quick access toolbar has three icons for save, undo, and redo, respectively, followed by the customized button. The eight tabs are File (highlighted red), Home, Animate, Draw (selected), Run, View, Tools, and Developer. Four ribbon groups are labeled Object, Color, Characteristic, and Tools. Object has eight icons labeled Line, Polyline, Arc, Bezier Curve, Box, Polygon, Ellipse, and Text. Color has four icons labeled Line, Fill, Text, and Window Background. Characteristic has five icons labeled Line Width, Line Style, Arrow Style, Line Pattern, and Fill Pattern. Tools have Five icons labeled Show Dimensions, DXF Import, Open color Palette, and Save Color Palette, and Arena symbol factory. The project bar shows the panel, Data Definition. The data elements are Activity area, Attribute, Entity, Expression, Failure, Queue, Resource, Schedule, Sequence, Set, StateSet, Station data, and Variable, respectively. The model window spreadsheet view is on the lower side of the model window and is titled, Model 3-1; A simple Processing System. A flow diagram with three blocks in series reads, Part Arrives to System 0, Drilling Center 0, and Part Leaves System. Below are two graphs. The first one plots Queue length on the vertical axis against the time in minutes on the horizontal axis. The vertical axis shows 0. The horizontal axis ranges from 0 through 20 in increments of 5. The second graph plots, Number Busy on the vertical axis against time in minutes on the horizontal axis. The vertical axis ranges from 0 through 2 in increments of 1. The horizontal axis ranges from 0 through 20 in increments of 5. An arrow points  toward the title in the model window and a callout box from the title reads, This box and text were added using the objects from the Draw ribbon.
Return to parent-slide containing images.
86

### Slide 87: Setting Run Conditions - Text Alternative
Setting Run Conditions - Text Alternative
Return to parent-slide containing images.
Run speed, Run Control, Reports, Project Parameters, Replication Parameters, Array Sizes, and Arena visual Designer are the options in the list at the top left corner. Established replication-related options for the current model. Settings include the number of simulation replications to be run, the length of the replication, the start date and time of the simulation, warm-up time length, time units, and the type of initialization to be performed between replications in the text mentioned in the upper side of the screen grab. The replication Parameters section has text box options labeled Number of Replication, Start Date and Time, Warm-up Period, Replication Length, Hours per Day, Terminating Condition, and Base time units. The parallel section has one option button labeled Run Replication Parallel. Several parallel Processes and Parallel Replication Input Data Files are two text boxes displayed in the section. OK, Cancel, Apply, and Help are the command buttons at the lower right corner of the screen grab.
Return to parent-slide containing images.
87

### Slide 88: Running the Model - Text Alternative
Running the Model - Text Alternative
Return to parent-slide containing images.
The quick access toolbar has three icons for save, undo, and redo, respectively, followed by the customized button. Four ribbon groups are labeled Interaction, Debug, Visualization, and Settings. Interaction has three icons, a player, Check Model, and Review Errors. The check model icon has a checkbox and is checked. Debug has five icons labeled Command, SIMAN, Break on Module, Debug Bar, and Runtime Elements bar. Debug bar and the Runtime Elements Bar have checkboxes and are unchecked. Visualization has three icons labeled Animate Connectors, Animate At Desktop Color Depth, and Highlight Active Module. All three of them have checkboxes and the last two are unchecked. Settings have three icons labeled Batch Run (No Animation), a slider, and Setup. The batch run icon has a checkbox and is unchecked. A callout box from the play button reads, Run the model. A callout box pointed at the pause button in the player icon reads, Pause the model run. A callout box pointed at the stop icon reads, End the model run ahead of it ending on its own. A callout box from the slider bar reads, Slide bar to speed up or slow down the model run.
Return to parent-slide containing images.
88

### Slide 89: Moving Around in the Model Window - Text Alternative
Moving Around in the Model Window - Text Alternative
Return to parent-slide containing images.
The quick access toolbar has three icons for save, undo, and redo, respectively, followed by the customized button. Seven group tabs are labeled Zoom, Views, Show, Select, Settings, Data Tips, and Layers. Zoom has five icons labeled Zoom Out, Zoom In, Zoom Factor, 83 percent, and Select and Zoom. Views have four icons labeled Submodel, Views, Named Views, and Split Screen with a checked checkbox. Show has seven checkboxes labeled project Bar, Status Bar, Animate Bar, Rulers, Grid, Guides, and Page Breaks. The checkboxes for Project bar, status bar, and guides are checked. Select has two icons labeled Snap to Grid, and Give to Guides, both with checkboxes that are checked. Settings have one icon labeled Grid and Snap Settings. Data Tips has two icons labeled Show Default Descriptions, and Show User-Specified Descriptions. Both have checkboxes, which are checked. Layers have an icon labeled Layers. A callout box pointed toward the Zoom out and zoom in icons read, You can use these buttons or use the plus or minus keys on your keyboard to zoom in and out. Pressing the asterisk button will pan out and show you all model contents in one view. A callout box pointed toward the Select ribbon group reads, Use snap to grid settings to make your models more orderly and line up modules or animation the way you want it.
Return to parent-slide containing images.
89

### Slide 90: Named Views – Organizing and Presenting - Text Alternative
Named Views – Organizing and Presenting - Text Alternative
Return to parent-slide containing images.
In the ribbon at the top, the quick access toolbar has three icons for save, undo, and redo, respectively, followed by the customized button. The eight tabs are labeled File (highlighted red), Home, Animate, Draw, Run, View (selected), Tools, and Developer. Seven group tabs are labeled Zoom, Views, Show, Select, Settings, Data Tips, and Layers. Zoom has five icons labeled Zoom Out, Zoom In, Zoom Factor, 82 percent, and Select and Zoom. Views have four icons labeled Submodel, Views, Named Views, and Split Screen with a checked checkbox. Show has seven checkboxes labeled project Bar, Status Bar, Animate Bar, Rulers, Grid, and Page Breaks. The checkboxes for Project bar, status bar are checked. Select has two icons labeled Snap to Grid, and Give to Guides, both with checkboxes that are checked. Settings has one icon labeled Grid and Snap Settings. Data Tips has two icons labeled Show Default Descriptions, and Show User-Specified Descriptions. Both have checkboxes, which are checked. Layers has an icon labeled Layers. A callout box pointed at named views in the views group reads, Named Views allow you to define hot keys for specific views that you may wish to reference for your own modeling use or to make it easier to navigate during a presentation.
Project Bar is displayed below the ribbon. The template panels are Data Definition, Discrete Processing, Decisions, Grouping, Input output, Animation, and Material Handling. Below that Reports and Navigate panels are shown. A box shows a small icon at the center. At the bottom is option top-level, followed by suboptions, dialog 1 (1), dialog 2 (2), dialog 3 (3), dialog 4 (4), dialog 5 (5), overview (o).
The process Module.doe window is to the right. It shows a logo along with its basic concepts followed by the text information. A callout box on top of the text reads, named views and their associated hot keys will appear on the navigate panel. A dialog box titled Named Views shows, Key, View Name as 1-dialog 1 (selected); 2-dialog 2; 3- dialog 3; 4-dialog 4; 5- dialog 5; and o- Overview. Six buttons, Show, Close, Help, Add, Edit, and Delete are shown to the right. A callout box pointing toward the named views dialog box reads, This is what the Named View dialog looks like. As stated in the text, try out the features and functions to see how they work. Right to this box is a flowchart with two blocks that read, Process 1 and Dispose 1. An arrow pointed toward the blocks from the named views dialog box is shown. Another dialog box titled View Name shows Hot key as 1 (highlighted), and Name as dialog 1. Below is a checkbox labeled Assign Name to Current View. Three buttons Ok, Cancel, and Help, with the OK button selected are placed on the right. A callout box inside the Process module window reads, Arena's SMARTS use Named Views to help make it easier to navigate the models. Another box with number 3 reads, if modeling of a resource is necessary, then the "Seize, Delay, Release" Action is appropriate. The Process module will then allow you to specify the resource or resources that you need to perform the process. The model will calculate statistics on the utilization and availability of the resource, in addition to the process time of the entity.
Return to parent-slide containing images.
90

### Slide 91: Opening Models - Text Alternative
Opening Models - Text Alternative
Return to parent-slide containing images.
A window for open is on the right side and shows two tabs labeled, Recent and Browse, where Recent is selected. File tab at the left lists the options like new, open (selected), info, save, browse Smarts, browse examples, account, and options. A callout box pointed toward the Open command in the File tab reads, Select Open to open up models that you have created or the book examples available with the text. A callout box pointed toward the Recent option in the Open window reads, You can either look to open files you recently edited or you can browse to a different location on your computer or network.
Return to parent-slide containing images.
91

### Slide 92: Arena Report - Text Alternative
Arena Report - Text Alternative
Return to parent-slide containing images.
On the top Microsoft, Excel Ribbon is given. The 13 columns are labeled from A through M, respectively. The 12 rows are labeled from 1 through 12, respectively. Row A1 contains the text that reads, Discrete-Time Statistics (Tally). The data in the tab labeled, acrossreplicationssummary is as follows: A2: Project Name. B2: Name C2: Type. D2: Source. E2: Average of Replication Averages. F2: observed equals Requested. G2: Half-Width. H2: stDev of Replication Averages. I2: Minimum Replication Average. J2: Maximum Replication Average. K2: Overall Min Value. L2: Overall max values. M2: Average observations per replication, respectively. Column A shows the text, simple processing, in row 3. Remaining rows are blank.
Column B shows the text reading, drilling center in row 3, drilling center.queue in row 6, and epart in row 7. Remaining rows are blank.
Column C shows the data as follows: Total time per entity, VA time per entity, Wait time per entity, waiting time, NVA time, Other time, total time, transfer time, VA time, and wait time.
Column D shows the text, process in rows 3 through 5, the text, queue in row 6, and the text, entity in rows 7 though 12.
Column E shows the values 0 in rows 7, 8, and 10, and the remaining rows have other values.
Column F shows the text reading, yes in rows 3 through 12.
Column G shows the text reading, insufficient in rows 3 through 12.
Column H shows the value, 0 in rows 3 through 12.
Column I shows the values 0 in rows 7, 8, and 10, and the remaining rows have other values.
Column J shows the values 0 in rows 7, 8, and 10, and the remaining rows have other values.
Column K shows the values 0 in rows 5 through 8, and 10, and the remaining rows have other values.
Column L shows the values 0 in rows 7, 8, and 10, and the remaining rows have other values.
Column M shows the value, 5 in rows 3 through 12, except for the value 6 in row 6.
Return to parent-slide containing images.
92

### Slide 93: Build It Yourself - Text Alternative
Build It Yourself - Text Alternative
Return to parent-slide containing images.
The first block leads to the second block that reads, Edit flowchart, data modules as needed; Experiment with Expression Builder - right-click in expression field. It leads to the third block that reads Add plots, animation, artwork, further leads to the fourth block that reads, Add named views and ends with the fifth block.
Return to parent-slide containing images.
93

### Slide 94: Case Study: Model 3-2, Specialized Serial Processing - Text Alternative
Case Study: Model 3-2, Specialized Serial Processing - Text Alternative
Return to parent-slide containing images.
It is titled, Model 3-2; Specialized Serial Loan Application. In the pop-up, Type is selected as Resource, Resource Name is selected as r Ali, Units to seize or Release is 1. Ok, cancel, and Help buttons, with the OK button selected are at the lower right corner of the pop-up. In the dialog box labeled Process, the name is selected as Ali Checks credit, and Type is selected as Standard. The logic section has options such as Action, Priority, and Resources. The list box for resources shows two values, of which, the value, Resource, r Ali, 1, is selected. The second value is <end of list>. Add, edit, and delete are the buttons on the right side of the resources. On the lower side, there are options, delay type, units, allocation, and expression. Delay type is selected as Expression, units are selected as Hours, allocation is selected as value added, and expression is EXPO (1). There is a checkbox labeled report statistics, checked. Below, a text box labeled comment is blank. OK, cancel, and Help are the three buttons at the lower right corner of the dialog box and OK is selected. A flowchart is displayed on top of it, from the top, showing, Application Arrives leads to Ali checks credit, further leads to Bianca Prepares Covenant, leads to Carl Prices Loan, leads to Deeta Disburses Funds, and finally ends with Application Departs. All the blocks are marked 0.
Return to parent-slide containing images.
94

### Slide 95: Case Study: Model 3-3, Generalized Parallel Processing 1 - Text Alternative
Case Study: Model 3-3, Generalized Parallel Processing 1 - Text Alternative
Return to parent-slide containing images.
These are titled Model 3-3; Generalized Parallel Loan Application. In the resources dialog box, type is selected as Resources, Resource Name is selected as r Loan Officer, Units to seize or Release is 1. Ok, cancel, and Help are the three buttons at the lower right corner, with the OK button selected. In the dialog box labeled Process, name is selected as One of the Employees Processes All 4 Steps, Type is selected as Standard. Logic section has options as Action, Priority, and Resources. Action is selected as seize delay release. Priority is selected as medium (2). The list box for resources shows two values, of which, the value, Resource, r Loan Officer, 1 is selected. The second value is <end of list>. Delay type is selected as expression. Units as hours. Expression is EXPO (1) plus EXPO (1) plus EXPO (1) plus EXPO (1). A checkbox labeled report statistics is checked. The comment box is blank. Ok, cancel, and Help are the three buttons at the lower right corner of the dialog box, with the OK button selected. The flowchart is displayed on the top, which starts with Application Arrives, leading to, One of the Employees Processes All 4 steps, which further leads to Application Departs. The graph displays the vertical axis labeled applications in process ranging from 0 through 20 in increments of 10. An illustration of four circles in four boxes, all connected by a vertical line is displayed next to the flowchart.
Return to parent-slide containing images.
95
