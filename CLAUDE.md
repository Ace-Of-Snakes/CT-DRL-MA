# Claude Code Project Notes

## Project
CT-DRL-MA — Deep Reinforcement Learning for RMGC Control in Container Terminals.
Master's thesis by Franko Jolić. Training DQN agents to control Rail-Mounted Gantry Cranes in a simulated container terminal.

## Environment
- Platform: WSL2 (Ubuntu on Windows), RTX 2060 Super 8GB
- Python 3.12.3, managed via Poetry (`poetry install`)
- Core deps: PyTorch 2.10+, NumPy, SciPy, Pandas, Matplotlib, Numba, Scikit-learn

## Key Commands
```bash
# Run tests
python -m pytest tests/

# Test tutorial scenarios
python -m simulation.training.test_tutorials

# Full training pipeline
python -m simulation.training.curriculum_trainer --output-dir runs/curriculum --seed 42
```

## Architecture at a Glance
- **State**: 13-channel (13, 15, 1160, 5) float32 tensor
- **Action**: 2-stage spatial — source position (+ learnable IDLE) then destination position with dynamic masking
- **Agent**: 7 DQN variants + Kitchen Sink combo, 6 CNN backbone variants
- **Training**: Phase 0 (27 tutorial scenarios, 11 mastery-gated tiers) → Phase 1 (11 curriculum stages, 20→220 imports/day)

## Key Directories
- `simulation/env/` — Environment (`terminal_env.py`), state encoder, reward engine
- `simulation/rl/` — Agent code, backbone factory, DQN variants
- `simulation/training/` — Curriculum trainer, tutorial runner, scenario definitions
- `simulation/core/` — Domain objects (containers, vehicles, facilities)
- `simulation/operations/` — Crane physics, gate scheduling, move execution
- `tests/` — Unit tests (19 files)
- `notebooks/` — Analysis notebooks (CNN comparison, variant comparison, etc.)
- `runs/` — Experiment outputs
- `latex/` — Master's thesis

## Current Work (as of 2026-02-23)
- Modular scenario system complete (`simulation/training/scenarios/`)
- IDLE action added to source head with pessimistic bias init (-5.0)
- Direct transfers: TRAIN_TO_TRUCK and TRUCK_TO_TRAIN (S26, S27)
- Per-container-size destination masks (replaces global MIN_CONTAINER_SPLITS)
- Fixed _resolve_dest_split to check contiguous free space + tier support
- Investigating S3/S4 greedy IDLE preference (training dynamics, not mechanical)

## Recent History
- Added learnable IDLE action with IdleSourceWrapper (pessimistic init)
- Added TRAIN_TO_TRUCK / TRUCK_TO_TRAIN direct transfer moves
- Refactored tutorial_scenarios.py into modular scenarios/ package (27 scenarios)
- Per-container-size dest masks (7 sizes: 20ft–45ft, 10–23 splits)
- Fixed mask/resolver mismatch in _resolve_dest_split (contiguous space check)
- Added 3 state channels: Crane Proximity, Train Departure Urgency, Time Progress
- Added real-world container size diversity to tutorial scenarios
- Refactored tests
- Phase 2: added terminal trucks back in
- Added kitchen sink agent variant
- Benchmarked different CNN configs and DQN variants
- Fixed memory leak, double crane container import/export
- Implemented curriculum learning with tutorial system

## Git Conventions
- Add `Co-Authored-By: Claude Opus 4.6` to commits (no email needed)
- Exception: commits touching only `latex/` should NOT have the co-author tag
