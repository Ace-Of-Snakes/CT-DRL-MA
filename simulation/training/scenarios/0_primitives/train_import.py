# simulation/training/scenarios/0_primitives/train_import.py
"""Scenario 2: Import container from train to yard."""
from __future__ import annotations

from simulation.training.scenarios._base import (
    Direction,
    TutorialScenario,
    _make_container,
    _make_train,
    _slot_train,
)


class TrainImport(TutorialScenario):
    """Train with Import container, no trucks.

    Agent must IMPORT_VEHICLE (train->yard).
    """
    id = 2
    name = "train_import"
    description = "Import container from train to yard"
    max_steps = 15
    repeatable = False

    def setup(self, env) -> None:
        c = _make_container("S2_C1", Direction.IMPORT)
        tr = _make_train("S2_TR1", containers=[c])
        _slot_train(env, tr)

    def check_success(self, env) -> bool:
        return env.yard.get_container("S2_C1") is not None
