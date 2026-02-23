# simulation/training/scenarios/8_stress/crowded_yard_pickup.py
"""Scenario 20: Serve 3 pickup trucks from a cluttered yard."""
from __future__ import annotations

import random

from simulation.training.scenarios._base import (
    Direction,
    TutorialScenario,
    _make_container,
    _make_truck,
    _place_distractors,
    _yard_placement,
)

_STRESS_DISTRACTORS = 12


class CrowdedYardPickup(TutorialScenario):
    """12 distractor containers + 3 Import targets scattered across yard.
    3 unparked pickup trucks, each wanting a different target.

    Tests finding specific containers in a cluttered yard and
    correct vehicle-to-container matching.
    """
    id = 20
    name = "crowded_yard_pickup"
    description = "Serve 3 pickup trucks from a cluttered yard"
    max_steps = 60
    repeatable = True

    def setup(self, env) -> None:
        rng = random.Random(2026)
        target_bays = [3, 20, 35]

        # Place 3 target Import containers
        for i, bay in enumerate(target_bays):
            c = _make_container(f"S20_T{i}", Direction.IMPORT)
            env.yard.add_container(c, _yard_placement(bay=bay, row=0, tier=0))
            tk = _make_truck(f"S20_TK{i}", pickup_ids=[c.container_id])
            env.trucks[tk.truck_id] = tk

        # 12 distractor containers to create clutter
        _place_distractors(
            env, rng, _STRESS_DISTRACTORS, "S20",
            exclude_bays=set(target_bays),
        )

    def check_success(self, env) -> bool:
        return all(f"S20_TK{i}" not in env.trucks for i in range(3))
