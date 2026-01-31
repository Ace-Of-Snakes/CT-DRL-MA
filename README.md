# Hierarchical DQN for Container Terminal - Implementation Summary

## Overview

This implementation transforms the container terminal RL system from a single-stage move selection to a **two-stage hierarchical decision process**:

1. **Stage 1 - Container Selection**: Agent selects which container to move (from yard, train, or truck) OR selects a parking action
2. **Stage 2 - Destination Selection**: Agent selects where to move the selected container

This dramatically reduces action space complexity at each decision point.

## Architecture

### Network Design (~50K parameters total)

```
┌─────────────────────────────────────────────────────────────┐
│                    SHARED BACKBONE (3D CNN)                 │
│  State [R,B,T,21] → Conv3D(21→32) → Conv3D(32→64)          │
│                   → AdaptivePool → Linear(64→128) → Tanh   │
│                   Output: state_emb [128]                   │
└─────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┴────────────────────┐
         ▼                                         ▼
┌─────────────────────┐                 ┌─────────────────────┐
│   CONTAINER SCORER  │                 │  DESTINATION SCORER │
│   (Stage 1)         │                 │  (Stage 2)          │
│                     │                 │                     │
│ cont_feat [16]      │                 │ dest_feat [12]      │
│ → MLP(16→64→64)     │                 │ + cont_feat [16]    │
│ → cont_emb [64]     │                 │ → MLP(28→64→64)     │
│                     │                 │ → dest_emb [64]     │
│ [state_emb,cont_emb]│                 │                     │
│ → MLP(192→64→1)     │                 │ [state_emb,dest_emb]│
│ → Q_container       │                 │ → MLP(192→64→1)     │
└─────────────────────┘                 │ → Q_destination     │
                                        └─────────────────────┘
```

### Feature Engineering

**Container Features (16 dims):**
- Position: row, bay, tier, start_split (normalized)
- Length: container length in meters
- Goods type: one-hot (Regular, Reefer, DangerousGoods)
- Special flags: is_swap_body, is_trailer
- Urgency: days_until_departure / 30
- Source type: one-hot (YARD, TRAIN, TRUCK)
- Source anchor bay (normalized)
- Opens access below (binary)

**Destination Features (12 dims):**
- Target position: row, bay, tier, start_split
- Destination type: one-hot (YARD, TRAIN, TRUCK)
- Bay distance from source
- Tier delta from source
- Zone match (goods type matches zone)
- Heat proximity to train anchors
- Ground level (tier == 0)

## File Structure

```
simulation/
├── config/
│   ├── curriculum_config.py    # Curriculum & network configs
│   └── yard_config.py          # Yard zone configuration
├── rl/
│   ├── agents/
│   │   └── hierarchical_dqn_agent.py   # Main agent
│   ├── policy/
│   │   └── hierarchical_networks.py    # Neural networks
│   └── features/
│       └── featurizers.py              # Feature extraction
├── env/
│   └── hierarchical_env.py     # Environment wrapper
├── operations/
│   └── hierarchical_moves.py   # Move generation
└── training/
    ├── curriculum_trainer.py   # Training loop
    └── visualize_training.py   # Plotting utilities
```

## Curriculum Training

The agent is trained through 11 stages with increasing complexity:

| Stage | Imports/Day | Exports/Day | Days |
|-------|-------------|-------------|------|
| 0 | 20 | 15 | 365 |
| 1 | 40 | 30 | 365 |
| 2 | 60 | 45 | 365 |
| ... | ... | ... | ... |
| 10 | 220 | 165 | 365 |

**Total: 4,015 training days**

### Epsilon Schedule
- Start: 0.3 (30% random exploration)
- End: 0.02 (2% random exploration)
- Decay: Linear over 50,000 steps
- Optional reset at each stage transition

## Decision Flow

```
1. Encode state once → state_emb (cached)

2. Build action pool:
   - Moveable containers (yard accessible + train + truck)
   - Pending parking moves

3. STAGE 1: Score all items in pool
   - Containers: ContainerScorer(state_emb, cont_feats)
   - Parking: ParkingScorer(state_emb, park_feats)
   - ε-greedy selection → chosen_item

4. If chosen_item is PARKING:
   - Execute parking move
   - reward = 0.5 (existing value)
   - Store experience, return

5. If chosen_item is CONTAINER:
   - Compute all valid destinations for this container
   
6. If no valid destinations:
   - reward = -1.0
   - Remove container from pool
   - Go back to step 3 (same timestep, max 10 retries)

7. STAGE 2: Score all destinations
   - DestinationScorer(state_emb, cont_feat, dest_feats)
   - ε-greedy selection → chosen_destination

8. Execute move (container → destination)
   - reward = RewardEngine.immediate_reward(...)

9. Store transition for training
```

## Multi-Crane Support

Cranes operate **sequentially**:
1. Crane 1 completes full two-stage decision
2. Crane 2 completes full two-stage decision
3. ...and so on for all idle cranes

This maintains simplicity while supporting multiple cranes.

## Usage

### Training
```bash
python -m simulation.training.curriculum_trainer \
    --output-dir runs/hierarchical \
    --seed 42 \
    --rows 5 \
    --bays 58 \
    --tiers 5 \
    --tracks 7 \
    --days-per-stage 365 \
    --start-imports 20 \
    --max-imports 220
```

### Resume Training
```bash
python -m simulation.training.curriculum_trainer \
    --output-dir runs/hierarchical \
    --start-stage 3 \
    --load-checkpoint runs/hierarchical/checkpoints/stage2_complete.pt
```

### Visualization
```bash
python -m simulation.training.visualize_training \
    runs/hierarchical/logs \
    --output-dir runs/hierarchical/plots
```

## Output Files

```
runs/hierarchical/
├── checkpoints/
│   ├── stage0_complete.pt
│   ├── stage1_complete.pt
│   ├── ...
│   └── final.pt
└── logs/
    ├── daily_metrics.csv      # Per-day metrics
    ├── stage_metrics.csv      # Per-stage summaries
    └── episodes/              # Detailed episode data
```

## Key Design Decisions

1. **Shared backbone with caching**: State encoding computed once per timestep
2. **Combined reward**: Single reward after full move execution
3. **Retry on no-destinations**: Same timestep, exclude failed container, max 10 retries
4. **Parking as direct action**: Competes with containers in Stage 1, no Stage 2
5. **Conservative bootstrapping**: Uses 0 for next-state Q-value estimate (can improve later)

## Future Improvements

1. **Double DQN**: Reduce overestimation bias
2. **Prioritized Experience Replay**: Focus on important transitions
3. **Rainbow DQN**: Combine multiple improvements
4. **Proper next-state Q estimation**: Store next pool in transitions
5. **Attention mechanism**: For variable-size action sets

## Hardware Requirements

- RTX 2060 Super 8GB: More than sufficient
- Estimated VRAM usage: ~500MB
- Training speed: ~1000 days/hour (estimate)
