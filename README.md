# UR3 Motor Variants Simulation

This repository contains scripts and data for generating, testing, and analyzing different motor configurations for a UR3 robot.

## Folder Structure

```

.
├── generate_xml.py             # Generates mjcf for UR3 with different motor selections
├── meshes
│   ├── collision               
│   └── visual                  
├── motors.py                   # Motor definitions, classes, and joint-motor mappings
├── numerical_dynamic.py        # Computes dynamic torques
├── numerical_static.py         # Computes static torques
├── universalUR3.urdf           # Base URDF of the UR3 robot
├── universalUR3_variants/       # Generated XML variants with different motor combinations
└── universalUR3.xml            # Default XML version of UR3

````

## Scripts

- **generate_xml.py**  
  Generates all motor variants based on the `motors.py` definitions and saves them in `universalUR3_variants/`.

- **numerical_static.py**  
  Performs static torque analysis for all joints and configurations, compares torques with rated and peak motor limits, and visualizes the results.

- **numerical_dynamic.py**  
  Simulates dynamic Cartesian trajectories, computes joint torques over time, and plots torque-speed curves and motor performance envelopes.

## Usage

1. Generate all motor variants:

```bash
python generate_xml.py
````

2. Run static analysis on all variants:

```bash
python numerical_static.py
```

3. Run dynamic analysis for Cartesian trajectories:

```bash
python numerical_dynamic.py
```
