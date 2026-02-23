# simulation/training/scenarios/1_chains/full_chain_truck.py
"""Scenario 7: Park truck, import from train, load onto truck."""
from __future__ import annotations

from simulation.training.scenarios._base import (
    Direction,
    TutorialScenario,
    _make_container,
    _make_train,
    _make_truck,
    _slot_train,
)


class FullChainTruck(TutorialScenario):
    """Train has Import container, unparked pickup truck waiting.

    Agent: SLOT_PARKING -> IMPORT_VEHICLE -> MOVE_CONTAINER->TRUCK.
    """
    id = 7
    name = "full_chain_truck"
    description = "Park truck, import from train, load onto truck"
    max_steps = 35
    repeatable = False

    def setup(self, env) -> None:
        c = _make_container("S7_C1", Direction.IMPORT)
        tr = _make_train("S7_TR1", containers=[c])
        _slot_train(env, tr)
        tk = _make_truck("S7_TK1", pickup_ids=[c.container_id])
        env.trucks[tk.truck_id] = tk

    def check_success(self, env) -> bool:
        return "S7_TK1" not in env.trucks
