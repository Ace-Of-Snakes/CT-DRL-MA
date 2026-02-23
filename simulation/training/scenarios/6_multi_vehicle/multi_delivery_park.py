# simulation/training/scenarios/6_multi_vehicle/multi_delivery_park.py
"""Scenario 15: Park 3 delivery trucks and import all containers."""
from __future__ import annotations

from simulation.training.scenarios._base import (
    ANCHOR_BAY,
    Direction,
    TutorialScenario,
    _make_container,
    _make_truck,
)

_N_DELIVERY = 3
_DELIVERY_BAYS = [ANCHOR_BAY, ANCHOR_BAY + 8, ANCHOR_BAY + 16]


class MultiDeliveryPark(TutorialScenario):
    """3 unparked delivery trucks with Export containers.

    Agent must park all 3, then import all containers to yard.
    Tests batch parking + batch truck-to-yard import pipeline.
    """
    id = 15
    name = "multi_delivery_park"
    description = "Park 3 delivery trucks and import all containers"
    max_steps = 50
    repeatable = True

    def setup(self, env) -> None:
        for i in range(_N_DELIVERY):
            c = _make_container(f"S15_C{i}", Direction.EXPORT)
            tk = _make_truck(f"S15_TK{i}", containers=[c])
            env.trucks[tk.truck_id] = tk

    def check_success(self, env) -> bool:
        # All 3 containers in yard, all 3 trucks departed (empty delivery)
        in_yard = all(
            env.yard.get_container(f"S15_C{i}") is not None
            for i in range(_N_DELIVERY)
        )
        trucks_gone = all(
            f"S15_TK{i}" not in env.trucks
            for i in range(_N_DELIVERY)
        )
        return in_yard and trucks_gone
