# simulation/training/scenarios/1_chains/train_to_truck_direct.py
"""Scenario 26: Park pickup truck, then direct-transfer import container from train to truck.

Teaches the agent to skip the yard entirely when a parked truck is already
waiting for a container that sits on the train.  This is the most efficient
import flow: TRAIN_TO_TRUCK saves one crane move compared to the two-step
TRAIN_TO_YARD + YARD_TO_TRUCK route.
"""
from __future__ import annotations

from simulation.training.scenarios._base import (
    ANCHOR_BAY,
    Direction,
    TutorialScenario,
    _make_container,
    _make_train,
    _make_truck,
    _slot_train,
)


class TrainToTruckDirect(TutorialScenario):
    """Train carries import container, unparked pickup truck waiting.

    Agent: SLOT_PARKING → TRAIN_TO_TRUCK (direct).
    Optimal: 2 moves (park + direct transfer).
    """
    id = 26
    name = "train_to_truck_direct"
    description = "Park pickup truck, then direct-transfer container from train to truck"
    max_steps = 25
    repeatable = False

    def setup(self, env) -> None:
        c = _make_container("S26_C1", Direction.IMPORT)
        tr = _make_train("S26_TR1", containers=[c])
        _slot_train(env, tr)
        tk = _make_truck("S26_TK1", pickup_ids=[c.container_id])
        env.trucks[tk.truck_id] = tk

    def check_success(self, env) -> bool:
        # Truck picked up and departed
        return "S26_TK1" not in env.trucks
