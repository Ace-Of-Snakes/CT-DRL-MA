# simulation/training/scenarios/6_multi_vehicle/delivery_to_train_pipeline.py
"""Scenario 16: Park 2 delivery trucks, import to yard, load train."""
from __future__ import annotations

from simulation.training.scenarios._base import (
    Direction,
    TutorialScenario,
    _make_container,
    _make_train,
    _make_truck,
    _slot_train,
)


class DeliveryToTrainPipeline(TutorialScenario):
    """2 unparked delivery trucks (Export), 1 train wanting both.

    Full export pipeline: park -> truck-to-yard -> yard-to-train.
    """
    id = 16
    name = "delivery_to_train_pipeline"
    description = "Park 2 delivery trucks, import to yard, load train"
    max_steps = 55
    repeatable = True

    def setup(self, env) -> None:
        cids = []
        for i in range(2):
            c = _make_container(f"S16_C{i}", Direction.EXPORT)
            cids.append(c.container_id)
            tk = _make_truck(f"S16_TK{i}", containers=[c])
            env.trucks[tk.truck_id] = tk
        tr = _make_train("S16_TR1", pickup_ids=cids)
        _slot_train(env, tr)

    def check_success(self, env) -> bool:
        tr = env.trains.get("S16_TR1")
        if tr is None:
            return False
        return all(tr.has_container(f"S16_C{i}") for i in range(2))
