# simulation/training/scenarios/0_primitives/park_truck.py
"""Scenario 1: Park an unparked pickup truck."""
from __future__ import annotations

from simulation.training.scenarios._base import (
    ANCHOR_BAY,
    Direction,
    TutorialScenario,
    _make_container,
    _make_truck,
    _yard_placement,
)


class ParkTruck(TutorialScenario):
    """Unparked pickup truck, Import container in yard.

    Agent must SLOT_PARKING.
    """
    id = 1
    name = "park_truck"
    description = "Park an unparked pickup truck"
    max_steps = 10
    repeatable = False

    def setup(self, env) -> None:
        c = _make_container("S1_C1", Direction.IMPORT)
        env.yard.add_container(c, _yard_placement(bay=ANCHOR_BAY, row=0, tier=0))
        tk = _make_truck("S1_TK1", pickup_ids=[c.container_id])
        env.trucks[tk.truck_id] = tk

    def check_success(self, env) -> bool:
        truck = env.trucks.get("S1_TK1")
        if truck is None:
            # Truck departed — must have been parked + loaded first
            return True
        return truck.parking_spot is not None
