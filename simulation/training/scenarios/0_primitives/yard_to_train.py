# simulation/training/scenarios/0_primitives/yard_to_train.py
"""Scenario 4: Load Export container onto waiting train."""
from __future__ import annotations

from simulation.training.scenarios._base import (
    ANCHOR_BAY,
    Direction,
    TutorialScenario,
    _make_container,
    _make_train,
    _slot_train,
    _yard_placement,
)


class YardToTrain(TutorialScenario):
    """Export container in yard, train wants it (in pickup manifest).

    Agent must MOVE_CONTAINER -> TRAIN.
    """
    id = 4
    name = "yard_to_train"
    description = "Load Export container onto waiting train"
    max_steps = 15
    repeatable = False

    def setup(self, env) -> None:
        c = _make_container("S4_C1", Direction.EXPORT)
        env.yard.add_container(c, _yard_placement(bay=ANCHOR_BAY, row=0, tier=0))
        tr = _make_train("S4_TR1", pickup_ids=[c.container_id])
        _slot_train(env, tr)

    def check_success(self, env) -> bool:
        tr = env.trains.get("S4_TR1")
        return tr is not None and tr.has_container("S4_C1")
