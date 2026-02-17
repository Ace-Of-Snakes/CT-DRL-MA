# CT-DRL-MA: Deep Reinforcement Learning for RMGC Control in Container Terminals

Master's thesis project by **Franko Jolić** — training DQN-based agents to control Rail-Mounted Gantry Cranes (RMGCs) in a simulated container terminal environment.

## Overview

This project implements a full-stack simulation of a container terminal storage yard and trains a Deep RL agent to schedule crane operations — deciding **which container to move** and **where to place it** — to maximize throughput while minimizing truck wait times, stacking inversions, and missed train deadlines.

The agent observes the terminal as a 10-channel spatial tensor (occupancy, urgency, demand, etc.) and outputs two-stage spatial actions: first selecting a **source** position (container to pick up), then a **destination** position (where to place it). Training uses a progressive curriculum that scales from 20 to 220 daily import containers across 11 stages, preceded by a structured tutorial phase with 25 mastery-gated scenarios.

### Key Features

- **Realistic terminal simulation** with trains, trucks, terminal trucks, rail yards, parking areas, and a multi-tier storage yard
- **10-channel CNN state encoding** capturing occupancy, container type, departure urgency, stacking inversions, truck/train demand, and more
- **2-stage spatial action selection** (source then destination) with dynamic masking for valid moves
- **7 DQN variants** (Baseline, Munchausen, Spectral Norm, NoisyNet, Dueling, QR-DQN, IQN) + a Kitchen Sink combination
- **6 CNN backbone variants** (Baseline, Wider, Deeper, Residual, Narrow-Deep, Kitchen Sink)
- **Curriculum learning** with 11 progressive difficulty stages
- **25 tutorial scenarios** organized in 11 mastery-gated tiers
- **Multi-crane support** with anti-reversal blacklisting
- **Realistic crane physics** using trapezoidal motion profiles (gantry, trolley, hoist)
- **Comprehensive logging** with CSV move tracking, NDJSON step logs, and daily/stage metrics

## Architecture

### Terminal Layout

```
Row 0-6:   Rail Tracks (7 tracks for train arrivals/departures)
Row 7:     Parking Area (truck parking with bay-level allocation)
Row 8-12:  Storage Yard (5 rows x 58 bays x 5 tiers, 20 splits/bay)
Row 13-14: Queue (waiting trucks before parking assignment)
```

### State Representation

The terminal state is encoded as a `(10, 15, 1160, 5)` float32 tensor:

| Channel | Name | Range | Description |
|---------|------|-------|-------------|
| 0 | Occupancy | [0, 1] | Is position occupied? |
| 1 | Container Start | [0, 1] | Start of a container span |
| 2 | Container Type | [0.25, 1.0] | Regular / Reefer / Dangerous / Swap |
| 3 | Accessible | [0, 1] | Can crane reach this container? |
| 4 | Departure Urgency | [0, 1] | Normalized days until departure |
| 5 | Blocks Urgent | [0, 1] | Stacking inversion severity |
| 6 | Direction | [0, 1] | Import (0) / Export (1) / Terminal (0.5) |
| 7 | Container Hash | [0.1, 1.0] | Deterministic container ID hash |
| 8 | Truck Demand | [0, 1] | Parked truck waiting for this container |
| 9 | Train Demand | [0.1, 1.0] | Train wanting this container |

Dimensions: **C**=10 channels, **R**=15 unified rows, **S**=1160 splits (58 bays x 20), **T**=5 tiers.

### Decision Flow

```
1. Encode state -> (10, 15, 1160, 5) tensor
2. CNN backbone -> feature map (64, 15, 290, 5) [stride=4 downsampling]
3. OccupiedPooling -> global feature (128-dim)
4. Source Head: Q(source positions) with source mask
5. Select source (epsilon-greedy or NoisyNet)
6. Extract source feature from feature map
7. Destination Head: Q(dest positions) conditioned on source
8. Select destination with dynamic destination mask
9. Infer move type from source/dest regions
10. Execute crane operation in simulation
```

### Move Types

| Move | Direction | Description |
|------|-----------|-------------|
| PARK_TRUCK | Queue -> Parking | Assign waiting truck to parking spot |
| TRAIN_TO_YARD | Rail -> Yard | Unload import container from train |
| TRUCK_TO_YARD | Parking -> Yard | Unload delivery container from truck |
| YARD_TO_TRAIN | Yard -> Rail | Load export container onto train |
| YARD_TO_TRUCK | Yard -> Parking | Load pickup container onto truck |
| YARD_TO_YARD | Yard -> Yard | Restack for better accessibility |
| YARD_TO_TERMINAL_TRUCK | Yard -> Parking | Dispatch swap body via terminal truck |

### Reward Structure

**Per-move rewards:**
- Distance + time cost (negative, proportional to crane travel)
- Move-type bonuses: export to train (+5.0), pickup to truck (+3.0), import to yard (+1.0), restack (-0.2)
- Parking proximity bonus (up to +0.5, incentivizes placing trucks near their target containers)

**Event rewards:**
- Train departure: +10.0 (successful batch export)
- Leftover penalty: -3.0 per missed container on departing train
- Truck wait shaping: sliding scale based on service time (fast=+2.0, slow=0.0)
- Idle waiting: -0.01 per truck per minute while cranes idle

**End-of-day penalties:**
- Leftover containers: -0.3 per container
- Stacking inversions: -0.5 per inversion

## DQN Variants

| Variant | Key Innovation | Exploration | Loss |
|---------|---------------|-------------|------|
| **Baseline** | Double DQN + aux dest loss | Epsilon-greedy | Huber TD |
| **Munchausen** | Reward augmentation via log-policy | Epsilon-greedy | Munchausen TD |
| **Spectral Norm** | Backbone Lipschitz constraint | Epsilon-greedy | Huber TD |
| **NoisyNet** | Parametric noise layers | State-dependent noise | Huber TD |
| **Dueling** | V(s) + A(s,a) decomposition | Epsilon-greedy | Huber TD |
| **QR-DQN** | 32 fixed quantiles | Epsilon-greedy | Quantile Huber |
| **IQN** | Sampled implicit quantiles | Epsilon-greedy | IQN Huber |
| **Kitchen Sink** | Deeper-Residual CNN + SN + Munchausen + NoisyNet | NoisyNet | Munchausen TD |

## CNN Backbone Variants

| Backbone | Layers | Channels | Special |
|----------|--------|----------|---------|
| **Baseline** | 4 | 32 -> 64 | Factored 1D convolutions |
| **Wider** | 4 | 64 -> 128 | 2x channel width |
| **Deeper** | 5 | 32 -> 64 | Extra cross-region pass (RF_R=5) |
| **Residual** | 4 | 32 -> 64 | Skip connections around conv2/conv3 |
| **Narrow-Deep** | 6 | 16 -> 32 | More layers, fewer channels |
| **Kitchen Sink** | 5 | 32 -> 64 | Deeper + Residual combined |

All backbones use factored 1D CNN design: container profile extraction (1x21x1), cross-region awareness (3x1x3), and spatial neighborhood refinement (1x5x1).

## Training Pipeline

### Phase 0: Tutorial Learning (25 Scenarios, 11 Tiers)

Structured skill-building before curriculum training:

| Tier | Scenarios | Skill |
|------|-----------|-------|
| 0 | S1-S4 | Primitives: park, import, export, load |
| 1 | S5-S6 | Two-action chains |
| 2 | S7-S8 | Full three-action chains |
| 3 | S9-S10 | Restack + load (with distractors) |
| 4 | S11-S12 | Random container sizes |
| 5 | S13-S14 | Multi-truck serving, deep unbury |
| 6 | S15-S16 | Multi-vehicle batch parking |
| 7 | S17 | Bidirectional train ops |
| 8 | S18-S19 | Concurrent import/export |
| 9 | S20-S21 | Full complexity (crowded yard, mini day) |
| 10 | S22-S25 | Terminal truck specialization |

Mastery gate: 90% pass rate on all scenarios in a tier before advancing. Periodic maintenance replays of mastered tiers to prevent catastrophic forgetting.

### Phase 1: Curriculum Training (11 Stages)

| Stage | Imports/Day | Days | Difficulty |
|-------|-------------|------|------------|
| 0 | 20 | 7 | Minimal |
| 1 | 40 | 7 | Early |
| 2 | 60 | 7 | Foundational |
| ... | ... | 7 | ... |
| 10 | 220 | 7 | Full capacity |

Epsilon resets at each stage transition. Carryover mechanism passes unfinished vehicles between days.

## Project Structure

```
CT-DRL-MA/
|-- simulation/
|   |-- config/
|   |   |-- crane_config.py          # Crane geometry & kinematics
|   |   |-- operations_config.py     # Reward weights & operational params
|   |   |-- train_config.py          # Train defaults (wagon count)
|   |   |-- yard_config.py           # Yard zones (reefer, DG, swap body)
|   |   +-- paths.py                 # Data & output file paths
|   |-- core/
|   |   |-- constants.py             # Global constants
|   |   |-- enums.py                 # Direction, MoveType, Status enums
|   |   |-- containers/
|   |   |   +-- container.py         # Container dataclass
|   |   |-- facilities/
|   |   |   |-- yard.py              # OptimizedStorageYard (numpy-backed)
|   |   |   |-- railyard.py          # OptimizedRailYard (track allocation)
|   |   |   +-- parking.py           # OptimizedParkingArea
|   |   |-- vehicles/
|   |   |   |-- truck.py             # Truck (delivery/pickup)
|   |   |   |-- train.py             # Train (with O(1) container lookup)
|   |   |   |-- wagon.py             # Wagon (OrderedDict containers)
|   |   |   +-- terminal_truck.py    # TerminalTruck (swap bodies)
|   |   +-- factories/
|   |       |-- container_factory.py # Vectorized container generation (KDE)
|   |       +-- truck_factory.py     # Truck generation (KDE arrival times)
|   |-- env/
|   |   |-- env.py                   # Base ContainerTerminalEnv
|   |   |-- unified_env.py           # UnifiedContainerTerminalEnv (2-stage spatial)
|   |   |-- unified_state_encoder.py # 10-channel state encoder
|   |   +-- reward_engine.py         # Reward calculation
|   |-- operations/
|   |   |-- crane_movements.py       # TerminalRMGC (3D crane physics)
|   |   |-- _rmgc_math.py            # Numba-compiled motion profiles
|   |   |-- gate.py                  # TerminalGate (truck/train scheduling)
|   |   +-- terminal_manager.py      # TerminalLogisticsManager (move execution)
|   |-- planning/
|   |   |-- logistics_manager.py     # LogisticsManager (daily planning)
|   |   |-- train_scheduler.py       # Best-fit track assignment
|   |   |-- train_loader.py          # First-fit-decreasing bin packing
|   |   |-- time_encoder.py          # Time encoding utilities
|   |   +-- driving_plan_parser.py   # Driving plan JSON parser
|   |-- rl/
|   |   |-- agent_registry.py        # Factory for agents + backbones
|   |   |-- base_agent.py            # BaseSpatialDQNAgent (shared infra)
|   |   |-- backbone_factory.py      # CNN backbone selection & spectral norm
|   |   |-- features/
|   |   |   +-- featurizers.py       # Container/Destination/Parking features
|   |   |-- multihead_dqn/
|   |   |   |-- config.py            # UnifiedDims, CNNConfig, DQNConfig
|   |   |   |-- unified_agent.py     # Baseline UnifiedDQNAgent
|   |   |   |-- unified_networks.py  # CNN backbone, Q-heads, full pipeline
|   |   |   +-- unified_replay_buffer.py  # Circular + PER buffers
|   |   +-- variants/
|   |       |-- dueling_agent.py     # DuelingDQNAgent
|   |       |-- iqn_agent.py         # IQNAgent (implicit quantile)
|   |       |-- munchausen_agent.py  # MunchausenDQNAgent
|   |       |-- noisynet_agent.py    # NoisyNetDQNAgent
|   |       |-- qrdqn_agent.py       # QRDQNAgent (quantile regression)
|   |       |-- spectralnorm_agent.py# SpectralNormDQNAgent
|   |       |-- kitchen_sink_agent.py# Combined best improvements
|   |       |-- backbones/           # 5 CNN backbone variants
|   |       +-- networks/            # Dueling, quantile, IQN, noisy heads
|   |-- training/
|   |   |-- unified_curriculum_trainer.py  # Main training loop
|   |   |-- unified_tutorial_runner.py     # Tutorial tier runner
|   |   +-- tutorial_scenarios.py          # 25 tutorial scenario definitions
|   |-- analytics/
|   |   |-- async_logger.py          # Thread-safe NDJSON logger
|   |   |-- move_csv_logger.py       # Per-move CSV logging
|   |   +-- stats_tracker.py         # Daily statistics aggregation
|   +-- utils/
|       |-- direction_utils.py       # Direction helpers
|       |-- id_generator.py          # Unique ID generation
|       +-- serialization.py         # Object serialization
|-- tests/                           # Unit tests & stress tests (19 files)
|-- notebooks/                       # Jupyter analysis notebooks
|   |-- cnn_comparison.ipynb         # CNN backbone benchmarks
|   |-- variant_comparison.ipynb     # DQN variant benchmarks
|   |-- kitchen_sink_tutorial.ipynb  # Kitchen sink analysis
|   +-- phase2_simulation_training.ipynb
|-- tools/
|   +-- visualise_crane_movements.py # Crane movement visualization
|-- runs/                            # Experiment outputs (checkpoints + logs)
|-- latex/                           # Master's thesis (LaTeX)
|-- docs/                            # BooleanStorage documentation
+-- misc/                            # Performance benchmarks & images
```

## Simulation Components

### Container Terminal

The terminal simulates a realistic storage yard with:
- **Storage Yard**: 5 rows x 58 bays x 5 tiers with 20 sub-bay splits per bay (1,160 split positions per row). Numpy-backed with O(1) lookups.
- **Rail Yard**: 7 tracks for train arrivals/departures. Best-fit scheduling with weekly wraparound.
- **Parking Area**: Bay-level truck parking with split-granularity allocation.
- **Queue**: 2-row waiting area for trucks before parking assignment.
- **2 RMGC Cranes**: Zone-divided with 4-bay overlap. Realistic trapezoidal motion profiles for gantry (130 m/min), trolley (70 m/min), and hoist (28 m/min).
- **2 Terminal Trucks**: Internal vehicles for swap body/trailer dispatch.

### Vehicle Operations

- **Trains**: Arrive with import containers, depart with export containers. 29 wagons per train. First-fit-decreasing bin packing for loading.
- **Trucks**: Delivery trucks bring export containers; pickup trucks collect import containers. KDE-based arrival time sampling.
- **Terminal Trucks**: Internal vehicles that carry only swap bodies and trailers between yard and parking.

### Container Types

- **Regular**: Standard 20ft/40ft containers
- **Reefer**: Refrigerated, stored at yard edges
- **Dangerous Goods**: Stored in center rows
- **Swap Body / Trailer**: Handled by terminal trucks, stored in designated row

### Data-Driven Generation

Container properties, dwell times, and truck arrival patterns are sampled from KDE models fitted to real-world distributions. Import/export container type distributions are operator-specific.

## Usage

### Dependencies

```bash
# Python 3.12.3, managed via Poetry
poetry install
```

**Core dependencies**: PyTorch 2.10+, NumPy, SciPy, Pandas, Matplotlib, Numba, Scikit-learn, Jupyter

### Training

```bash
# Full training pipeline (tutorials + curriculum)
python -m simulation.training.unified_curriculum_trainer \
    --output-dir runs/curriculum \
    --seed 42

# Resume from checkpoint
python -m simulation.training.unified_curriculum_trainer \
    --output-dir runs/curriculum \
    --start-stage 3 \
    --load-checkpoint runs/curriculum/checkpoints/stage2_complete.pt

# Select agent variant and backbone
python -m simulation.training.unified_curriculum_trainer \
    --variant kitchen_sink \
    --backbone kitchen_sink \
    --output-dir runs/kitchen_sink
```

### Testing

```bash
# Run unit tests
python -m pytest tests/

# Test tutorial scenario setups
python -m simulation.training.test_tutorials
```

### Analysis

```bash
# Jupyter notebooks for variant/backbone comparison
jupyter notebook notebooks/variant_comparison.ipynb
jupyter notebook notebooks/cnn_comparison.ipynb
```

## Output Structure

```
runs/<experiment>/
|-- checkpoints/
|   |-- stage0_complete.pt
|   |-- stage1_complete.pt
|   +-- final.pt
+-- logs/
    |-- daily_metrics.csv       # Per-day: reward, moves, epsilon, loss
    |-- stage_metrics.csv       # Per-stage: avg reward, total moves
    |-- moves_day<N>.csv        # Per-move: container, regions, crane metrics
    |-- events_day<N>.csv       # Vehicle arrivals/departures
    +-- episodes/               # Detailed NDJSON step logs
```

## Hardware

- Developed and tested on RTX 2060 Super 8GB
- Estimated VRAM usage: ~500MB
- Platform: WSL2 (Ubuntu on Windows)

## License

See [LICENSE](LICENSE) for details.
