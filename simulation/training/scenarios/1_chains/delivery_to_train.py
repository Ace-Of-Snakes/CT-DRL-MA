# simulation/training/scenarios/1_chains/delivery_to_train.py
"""Scenario 8: Park delivery truck, unload to yard, load onto train."""
from __future__ import annotations

from simulation.training.scenarios._base import (
    Direction,
    TutorialScenario,
    _make_container,
    _make_train,
    _make_truck,
    _slot_train,
)


class DeliveryToTrain(TutorialScenario):
    """Delivery truck (unparked) with Export container, train wants it.

    Agent: SLOT_PARKING -> IMPORT_VEHICLE -> MOVE_CONTAINER->TRAIN.
    """
    id = 8
    name = "delivery_to_train"
    description = "Park delivery truck, unload to yard, load onto train"
    max_steps = 35
    repeatable = False

    def setup(self, env) -> None:
        c = _make_container("S8_C1", Direction.EXPORT)
        tr = _make_train("S8_TR1", pickup_ids=[c.container_id])
        _slot_train(env, tr)
        tk = _make_truck("S8_TK1", containers=[c])
        env.trucks[tk.truck_id] = tk

    def check_success(self, env) -> bool:
        tr = env.trains.get("S8_TR1")
        return tr is not None and tr.has_container("S8_C1")
