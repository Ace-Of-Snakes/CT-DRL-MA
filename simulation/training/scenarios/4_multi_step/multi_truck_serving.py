# simulation/training/scenarios/4_multi_step/multi_truck_serving.py
"""Scenario 13: Match 3 containers to 3 pickup trucks."""
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


class MultiTruckServing(TutorialScenario):
    """3 Import containers in yard, 3 pre-parked trucks each wanting one.

    Tests vehicle selection precision -- agent must match the right
    container to the right truck.
    """
    id = 13
    name = "multi_truck_serving"
    description = "Match 3 containers to 3 pickup trucks"
    max_steps = 30
    repeatable = True

    def setup(self, env) -> None:
        bays = [ANCHOR_BAY, ANCHOR_BAY + 5, ANCHOR_BAY + 10]
        for i, bay in enumerate(bays):
            c = _make_container(f"S13_C{i}", Direction.IMPORT)
            env.yard.add_container(c, _yard_placement(bay=bay, row=0, tier=0))
            tk = _make_truck(f"S13_TK{i}", pickup_ids=[c.container_id])
            env.trucks[tk.truck_id] = tk
            _park_truck_near(env, tk, bay)

    def check_success(self, env) -> bool:
        return all(f"S13_TK{i}" not in env.trucks for i in range(3))
