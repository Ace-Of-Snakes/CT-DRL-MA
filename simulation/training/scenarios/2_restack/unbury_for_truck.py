# simulation/training/scenarios/2_restack/unbury_for_truck.py
"""Scenario 9: Restack blocker to reach buried Import, load onto truck."""
from __future__ import annotations

import random
from datetime import timedelta

from simulation.training.scenarios._base import (
    ANCHOR_BAY,
    TUTORIAL_TIME,
    Direction,
    TutorialScenario,
    _make_container,
    _make_truck,
    _park_truck_near,
    _place_distractors,
    _random_container_spec,
    _yard_placement,
)


class UnburyForTruck(TutorialScenario):
    """Import target buried under blocker, 5 distractors.

    Agent: RESTACK blocker -> MOVE_CONTAINER->TRUCK.
    """
    id = 9
    name = "unbury_for_truck"
    description = "Restack blocker to reach buried Import, load onto truck"
    max_steps = 25
    repeatable = True

    def setup(self, env) -> None:
        rng = random.Random(42)
        # All stacked containers must share the same size
        ft, m, ctype, _w, _h, _hc = _random_container_spec(rng)
        target = _make_container("S9_TARGET", Direction.IMPORT,
                                  length_ft=ft, length_m=m, container_type=ctype)
        blocker = _make_container("S9_BLK1", Direction.IMPORT,
                                   departure=TUTORIAL_TIME + timedelta(days=10),
                                   length_ft=ft, length_m=m, container_type=ctype)
        env.yard.add_container(target,  _yard_placement(bay=ANCHOR_BAY, row=0, tier=0))
        env.yard.add_container(blocker, _yard_placement(bay=ANCHOR_BAY, row=0, tier=1))

        _place_distractors(env, rng, 5, "S9", exclude_bays={ANCHOR_BAY})

        tk = _make_truck("S9_TK1", pickup_ids=[target.container_id])
        env.trucks[tk.truck_id] = tk
        _park_truck_near(env, tk, ANCHOR_BAY)

    def check_success(self, env) -> bool:
        return (env.yard.get_container("S9_TARGET") is None
                and "S9_TK1" not in env.trucks)
