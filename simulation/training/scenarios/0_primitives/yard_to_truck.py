# simulation/training/scenarios/0_primitives/yard_to_truck.py
"""Scenario 3: Load Import container onto parked pickup truck."""
from __future__ import annotations

from simulation.training.scenarios._base import (
    ANCHOR_BAY,
    Direction,
    TutorialScenario,
    _make_container,
    _make_truck,
    _park_truck_near,
    _yard_placement,
)


class YardToTruck(TutorialScenario):
    """Import container in yard, pre-parked pickup truck wants it.

    Agent must MOVE_CONTAINER -> TRUCK.
    """
    id = 3
    name = "yard_to_truck"
    description = "Load Import container onto parked pickup truck"
    max_steps = 15
    repeatable = False

    def setup(self, env) -> None:
        c = _make_container("S3_C1", Direction.IMPORT)
        env.yard.add_container(c, _yard_placement(bay=ANCHOR_BAY, row=0, tier=0))
        tk = _make_truck("S3_TK1", pickup_ids=[c.container_id])
        env.trucks[tk.truck_id] = tk
        _park_truck_near(env, tk, ANCHOR_BAY)

    def check_success(self, env) -> bool:
        return (env.yard.get_container("S3_C1") is None
                and "S3_TK1" not in env.trucks)
