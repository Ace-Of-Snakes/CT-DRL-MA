# Container Terminal Simulation

This repository contains simulation tools for container terminal operations, developed as part of the CT-DRL-MA (Container Terminal Deep Reinforcement Learning Multi-Agent system) master's thesis project.

## Components

The simulation framework consists of two main components:

1. **Container Terminal Simulator** (CTSimulator.py) - Creates and manages a realistic container terminal layout
2. **RMG Movement Calculator** (RMGMovementCalculator.py) - Calculates realistic movement times for Rail Mounted Gantry cranes

## Container Terminal Simulator

The `ContainerTerminal` class generates a configurable layout for a container terminal, including rail tracks, parking spots, driving lanes, and storage yards. It calculates distances between all objects in the terminal and provides visualization capabilities.

### Features

- Customizable terminal layout with flexible parameters
- Configurable ratio parameters between different terminal sections
- Distance matrix calculation between all terminal objects
- Terminal visualization with proper labeling and scaling
- Save/load functionality for the distance matrix

### Usage

```python
from CTSimulator import ContainerTerminal

# Create a container terminal with custom parameters
terminal = ContainerTerminal(
    layout_order=['rails', 'parking', 'driving_lane', 'yard_storage'],
    num_railtracks=6,                # Number of rail tracks
    num_railslots_per_track=40,      # Rail slots per track
    num_storage_rows=5,              # Storage rows (A-E)
    # Ratio parameters
    parking_to_railslot_ratio=1.0,   # One parking spot per rail slot
    storage_to_railslot_ratio=2.0,   # Two storage slots per rail slot
    # Dimension parameters (in meters)
    rail_slot_length=20.0,
    track_width=3.0,
    space_between_tracks=1.5,
    space_rails_to_parking=5.0,
    space_driving_to_storage=2.0,
    parking_width=4.0,
    driving_lane_width=8.0,
    storage_slot_width=10.0
)

# Visualize the terminal
fig, ax = terminal.visualize(figsize=(30, 15))
plt.savefig('terminal_layout.png', dpi=300)

# Save the distance matrix for later use
terminal.save_distance_matrix('distance_matrix.pkl')

# Get distance between two objects
distance = terminal.get_distance('t1_1', 'A1')
print(f"Distance between t1_1 and A1: {distance:.2f} meters")
```

### Important Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `layout_order` | Order of terminal sections | `['rails', 'parking', 'driving_lane', 'yard_storage']` |
| `num_railtracks` | Number of rail tracks | 3 |
| `num_railslots_per_track` | Number of slots per rail track | 10 |
| `num_storage_rows` | Number of rows in storage yard | 6 |
| `parking_to_railslot_ratio` | Ratio of parking spots to rail slots | 1.0 |
| `storage_to_railslot_ratio` | Ratio of storage slots per rail slot | 2.0 |
| `rail_slot_length` | Length of a rail slot in meters | 20.0 |

## RMG Movement Calculator

The `RMGMovementCalculator` class calculates realistic movement times for Rail Mounted Gantry (RMG) cranes based on physics models with proper acceleration/deceleration.

### Features

- Physics-based movement calculations with realistic acceleration/deceleration
- Separate modeling of gantry, trolley, and hoisting movements
- Different movement patterns based on source and destination types
- Support for various container types and heights
- Optimization features: movement time caching, batch calculations
- Option to generate time matrices for lookup during reinforcement learning

### Usage

```python
from RMGMovementCalculator import RMGMovementCalculator

# Create calculator with realistic RMG parameters
calculator = RMGMovementCalculator(
    distance_matrix_path='distance_matrix.pkl',  # Path to saved distance matrix
    trolley_speed=70.0,         # m/min - from Liebherr specs
    hoisting_speed=28.0,        # m/min with load - from Liebherr specs
    trolley_acceleration=0.3,   # m/s²
    hoisting_acceleration=0.2,  # m/s²
    max_height=20.0,            # meters
    ground_vehicle_height=1.5   # meters
)

# Calculate time for a train to storage movement
movement_time = calculator.calculate_movement_time(
    src='t1_1',             # Source position (train slot)
    dst='A1',               # Destination position (storage slot)
    container_type="TEU",   # Container type
    stack_height=0.0        # Height of stack at destination
)
print(f"Movement time: {movement_time:.2f} seconds")

# Batch calculate multiple movements
movements = [
    ('t1_1', 'A1', 'TEU', 0.0),
    ('A5', 'p_2', 'FEU', 2.59),
    ('p_3', 't2_3', 'HQ', 0.0)
]
times = calculator.batch_calculate_times(movements)

# Generate a time matrix for a subset of objects
calculator.export_time_matrix('time_matrix.pkl', objects=['t1_1', 't1_2', 'A1', 'A2', 'p_1'])
```

### Important Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `trolley_speed` | Trolley travel speed in m/min | 70.0 |
| `hoisting_speed` | Hoisting speed with load in m/min | 28.0 |
| `gantry_speed` | Hoisting speed with load in m/min | 130.0 |
| `trolley_acceleration` | Trolley acceleration in m/s² | 0.3 |
| `hoisting_acceleration` | Hoisting acceleration in m/s² | 0.2 |
| `gantry_acceleration` | Hoisting acceleration in m/s² | 0.1 |
| `max_height` | Maximum height of the crane in meters | 20.0 |
| `ground_vehicle_height` | Height of trucks/trains from ground in meters | 1.5 |
| `container_heights` | Dictionary mapping container types to heights | `{"TEU": 2.59, "FEU": 2.59, "HQ": 2.90}` |

## Movement Types

The RMG Movement Calculator handles different types of container movements:

1. **Train to Storage**: Moving a container from a train slot to a storage yard location
2. **Storage to Train**: Moving a container from the storage yard to a train slot
3. **Truck to Storage**: Moving a container from a parking spot (truck) to storage
4. **Storage to Truck**: Moving a container from storage to a parking spot
5. **Train to Truck**: Direct transfer from train to truck (or vice versa)
6. **Storage to Storage**: Reshuffling containers within the storage yard

Each movement type has specific rules for calculating vertical distances based on container height and stack height.

## Coordinate System

The coordinate system used in the simulation:
- X-axis: Position along the trolley travel direction (across the bridge)
- Y-axis: Position along the gantry travel direction (along the rails)

## Next Steps

This simulation framework provides the foundation for implementing reinforcement learning agents to optimize container terminal operations:

1. Build a reinforcement learning environment using these components
2. Implement the Terminal Manager Agent and Pre-Marshaling Agent
3. Develop the training infrastructure and curriculum learning approach