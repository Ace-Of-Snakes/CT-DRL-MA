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
python -m simulation.training.unified_curriculum_trainer --output-dir runs/curriculum --seed 42
```

## Architecture at a Glance
- **State**: 10-channel (10, 15, 1160, 5) float32 tensor
- **Action**: 2-stage spatial — source position then destination position with dynamic masking
- **Agent**: 7 DQN variants + Kitchen Sink combo, 6 CNN backbone variants
- **Training**: Phase 0 (25 tutorial scenarios, 11 mastery-gated tiers) → Phase 1 (11 curriculum stages, 20→220 imports/day)

## Key Directories
- `simulation/env/` — Environment (`unified_env.py`), state encoder, reward engine
- `simulation/rl/` — Agent code, backbone factory, DQN variants
- `simulation/training/` — Curriculum trainer, tutorial runner, scenario definitions
- `simulation/core/` — Domain objects (containers, vehicles, facilities)
- `simulation/operations/` — Crane physics, gate scheduling, move execution
- `tests/` — Unit tests (19 files)
- `notebooks/` — Analysis notebooks (CNN comparison, variant comparison, etc.)
- `runs/` — Experiment outputs
- `latex/` — Master's thesis

## Current Work (as of 2026-02-22)
Refactoring `tutorial_scenarios.py` (monolithic) into modular scenario system:
- New `simulation/training/scenarios/` directory with individual modules per tier:
  `0_primitives`, `1_chains`, `2_restack`, `3_generalization`, `4_multi_step`,
  `5_terminal_truck`, `6_multi_vehicle`, `7_bidirectional`, `8_stress`
- Base class in `_base.py`, registry in `_registry.py`, visualization in `_visualization.py`
- Updates to `unified_tutorial_runner.py` and `unified_env.py` to support new structure
- Uncommitted changes in progress

## Recent History
- Added real-world container size diversity to tutorial scenarios
- Refactored tests
- Phase 2: added terminal trucks back in
- Added kitchen sink agent variant
- Benchmarked different CNN configs and DQN variants
- Fixed memory leak, double crane container import/export
- Implemented curriculum learning with tutorial system
