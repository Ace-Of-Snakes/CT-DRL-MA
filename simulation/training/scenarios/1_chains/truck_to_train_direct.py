# simulation/training/scenarios/1_chains/truck_to_train_direct.py
"""Scenario 27: Park delivery truck, then direct-transfer export container from truck to train.

Teaches the agent to skip the yard entirely when a train is already waiting
for a container that sits on the delivery truck.  This is the most efficient
export flow: TRUCK_TO_TRAIN saves one crane move compared to the two-step
TRUCK_TO_YARD + YARD_TO_TRAIN route.
"""
from __future__ import annotations

from simulation.training.scenarios._base import (
    Direction,
    TutorialScenario,
    _make_container,
    _make_train,
    _make_truck,
    _slot_train,
)


class TruckToTrainDirect(TutorialScenario):
    """Delivery truck carries export container, train wants it.

    Agent: SLOT_PARKING → TRUCK_TO_TRAIN (direct).
    Optimal: 2 moves (park + direct transfer).
    """
    id = 27
    name = "truck_to_train_direct"
    description = "Park delivery truck, then direct-transfer container from truck to train"
    max_steps = 25
    repeatable = False

    def setup(self, env) -> None:
        c = _make_container("S27_C1", Direction.EXPORT)
        tr = _make_train("S27_TR1", pickup_ids=[c.container_id])
        _slot_train(env, tr)
        tk = _make_truck("S27_TK1", containers=[c])
        env.trucks[tk.truck_id] = tk

    def check_success(self, env) -> bool:
        tr = env.trains.get("S27_TR1")
        return tr is not None and tr.has_container("S27_C1")
