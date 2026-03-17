# Learning to Orchestrate Intermodal Terminals

**A CNN-Augmented Deep Reinforcement Learning Approach for Crane Scheduling, Truck Parking and Train Loading**

Master's thesis by [Franko Jolić](https://github.com/Ace-Of-Snakes), University of Hamburg, in collaboration with the Fraunhofer Center for Maritime Logistics and Services (CML).

> **Research question:** Can a single deep reinforcement learning agent learn to jointly schedule crane operations, park trucks, and load trains in a realistic container terminal simulation?

## What This Project Does

This repository contains the full source code for a simulated intermodal container terminal and a DQN-based reinforcement learning agent that learns to operate it. The agent controls two Rail-Mounted Gantry Cranes (RMGCs) to move containers between trains, trucks, and a storage yard, handling 10 distinct move types across three interconnected facility regions.

The agent observes the terminal as a 14-channel spatial tensor and produces two-stage spatial actions: first selecting a **source** (which container to pick up, or IDLE) and then a **destination** (where to place it). Move types are inferred from source and destination regions, eliminating explicit action-type prediction. Training uses a two-phase curriculum: 33 tutorial scenarios organised into 14 mastery-gated tiers (teaching atomic skills before compositions), followed by 11 progressive difficulty stages scaling from 20 to 220 daily container imports.

### Key Results

- **7 DQN variants** and **6 CNN backbone variants** systematically compared
- **Kitchen Sink agent** (Munchausen + NoisyNet + Spectral Norm + Deeper-Residual CNN) achieves 99.2% tutorial pass rate across all 33 scenarios
- All agents reach 98–100% completion on import/export tasks up to 400 containers per day
- Agents exhibit emergent behaviours: export prioritisation over imports, yard flattening strategies, and multi-step restacking plans

## Architecture Overview

```
State Tensor (14, 15, 1160, 5)
        │
        ▼
┌─────────────────────┐
│   CNN Backbone       │  Factored 1D convolutions:
│   (4 stages)         │  bay-level (1×21×1) → cross-region (3×1×3)
│                      │  → neighbourhood (1×5×1)
└─────────┬───────────┘
          │  Feature map (64, 15, 290, 5)
          ▼
┌─────────────────────┐     ┌─────────────────────┐
│   Source Head        │────▶│   Destination Head   │
│   + IDLE action      │     │   (source-conditioned)│
│   + validity mask    │     │   + validity mask     │
└─────────────────────┘     └─────────────────────┘
          │                           │
          ▼                           ▼
     Source position            Dest position
          │                           │
          └───────────┬───────────────┘
                      ▼
            Move type inferred from
            source/dest regions
```

### Terminal Layout

```
Row  0–6:   Rail Yard     (7 tracks for train arrivals/departures)
Row  7:     Parking Area  (truck parking, bay-level allocation)
Row  8–12:  Storage Yard  (5 rows × 58 bays × 5 tiers, 20 splits/bay)
Row 13–14:  Queue         (trucks waiting for parking assignment)
```

### Move Types

| Move | Source → Dest | Description |
|------|---------------|-------------|
| PARK_TRUCK | Queue → Parking | Assign waiting truck to a parking bay |
| TRAIN_TO_YARD | Rail → Yard | Unload import container from train |
| TRUCK_TO_YARD | Parking → Yard | Unload delivery container from truck |
| YARD_TO_TRAIN | Yard → Rail | Load export container onto train |
| YARD_TO_TRUCK | Yard → Parking | Load pickup container onto truck |
| YARD_TO_YARD | Yard → Yard | Restack for better accessibility |
| TRAIN_TO_TRUCK | Rail → Parking | Direct transfer (train to pickup truck) |
| TRUCK_TO_TRAIN | Parking → Rail | Direct transfer (delivery truck to train) |
| YARD_TO_TERMINAL_TRUCK | Yard → Parking | Dispatch swap body via terminal truck |
| IDLE | — | Crane voluntarily waits |

## DQN Variants

| Variant | Key Modification | Exploration |
|---------|-----------------|-------------|
| **Baseline** | Double DQN + auxiliary destination loss | ε-greedy |
| **Munchausen** | Reward augmentation via scaled log-policy | ε-greedy |
| **Spectral Norm** | Lipschitz-constrained backbone | ε-greedy |
| **NoisyNet** | Parametric noise layers (state-dependent exploration) | Learned noise |
| **Dueling** | V(s) + A(s,a) stream decomposition | ε-greedy |
| **QR-DQN** | 32-quantile distributional value estimation | ε-greedy |
| **IQN** | Implicit quantile sampling | ε-greedy |
| **Kitchen Sink** | Munchausen + NoisyNet + Spectral Norm + Deeper-Residual CNN | Learned noise |

## Training Pipeline

### Phase 1: Tutorial Learning (33 Scenarios, 14 Tiers)

Structured skill-building from atomic moves to full-complexity operations:

| Tiers | Skills |
|-------|--------|
| 0–1 | Single-move primitives (park, import, export) and two-action chains |
| 2–4 | Three-action chains, restacking with distractors, variable container sizes |
| 5–7 | Multi-truck serving, terminal truck dispatch, batch parking |
| 8–10 | Bidirectional train ops, concurrent import/export, crowded-yard stress |
| 11–13 | Full-day mini-simulations combining all move types |

Mastery gate: ≥90% pass rate on all tier scenarios before advancing. Periodic replay of mastered tiers prevents catastrophic forgetting.

### Phase 2: Curriculum Training (11 Stages)

Traffic intensity scales from 20 to 220 daily imports across 11 stages, each running 7 simulated days. Exploration resets at each stage boundary.

## Project Structure

```
simulation/
├── config/           Crane geometry, reward weights, yard zones
├── core/             Domain objects (containers, vehicles, facilities)
│   └── facilities/   StorageYard, RailYard, ParkingArea (numpy-backed grids)
├── env/              TerminalEnv, StateEncoder, RewardEngine, FacilityCoordinator
├── operations/       Crane physics (trapezoidal motion), gate scheduling
├── planning/         Daily logistics, train scheduling, bin packing
├── rl/               Agent code, backbone factory, 7 DQN variants
│   ├── multihead_dqn/  Core agent, networks, replay buffer
│   └── variants/       Dueling, IQN, Munchausen, NoisyNet, QR-DQN, etc.
├── training/         Curriculum trainer, tutorial runner
│   └── scenarios/    33 tutorial scenarios (modular, per-tier)
└── analytics/        Logging (NDJSON, CSV, daily stats)

tests/                Unit and integration tests
notebooks/            Analysis notebooks (CNN comparison, variant comparison, eval)
latex/                Master's thesis (LaTeX source + figures)
```

## Quick Start

```bash
# Install dependencies (Python 3.12, Poetry)
poetry install

# Run tests
python -m pytest tests/

# Train with default settings (Baseline DQN, Baseline CNN)
python -m simulation.training.curriculum_trainer \
    --output-dir runs/baseline --seed 42

# Train Kitchen Sink variant
python -m simulation.training.curriculum_trainer \
    --variant kitchen_sink --backbone kitchen_sink \
    --output-dir runs/kitchen_sink --seed 42
```

## Hardware Requirements

- **GPU**: Developed on RTX 2060 Super (8 GB VRAM); ~500 MB VRAM usage
- **Platform**: WSL2 (Ubuntu on Windows) or native Linux
- **Dependencies**: PyTorch 2.1+, NumPy, SciPy, Pandas, Matplotlib, Numba

## Thesis

The full master's thesis is included in `latex/`. It covers the simulation design, state representation, agent architecture, curriculum system, experimental setup, results across all variant and backbone combinations, and a discussion of emergent agent behaviours. The compiled PDF is at [`latex/main.pdf`](latex/main.pdf).

## Citation

If you use this work in your research, please cite:

```bibtex
@mastersthesis{jolic2026orchestrate,
  author  = {Jolić, Franko},
  title   = {Learning to Orchestrate Intermodal Terminals: A CNN-Augmented
             Deep Reinforcement Learning Approach for Crane Scheduling,
             Truck Parking and Train Loading},
  school  = {University of Hamburg},
  year    = {2026},
  note    = {In collaboration with Fraunhofer CML}
}
```

## License

See [LICENSE](LICENSE) for details.
