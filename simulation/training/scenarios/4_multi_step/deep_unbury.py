# simulation/training/scenarios/4_multi_step/deep_unbury.py
"""Scenario 14: Dig through 2 blockers to reach buried Import container."""
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


class DeepUnbury(TutorialScenario):
    """Import target at tier 0, 2 blockers above (tier 1, tier 2).

    Agent must restack twice before loading target onto truck.
    5 random distractors for noise.
    """
    id = 14
    name = "deep_unbury"
    description = "Dig through 2 blockers to reach buried Import container"
    max_steps = 30
    repeatable = True

    def setup(self, env) -> None:
        rng = random.Random(7)
        # All stacked containers must share the same size
        ft, m, ctype, _w, _h, _hc = _random_container_spec(rng)
        target = _make_container("S14_TARGET", Direction.IMPORT,
                                  length_ft=ft, length_m=m, container_type=ctype)
        blocker1 = _make_container(
            "S14_BLK1", Direction.IMPORT,
            departure=TUTORIAL_TIME + timedelta(days=10),
            length_ft=ft, length_m=m, container_type=ctype,
        )
        blocker2 = _make_container(
            "S14_BLK2", Direction.IMPORT,
            departure=TUTORIAL_TIME + timedelta(days=10),
            length_ft=ft, length_m=m, container_type=ctype,
        )
        env.yard.add_container(target,   _yard_placement(bay=ANCHOR_BAY, row=0, tier=0))
        env.yard.add_container(blocker1, _yard_placement(bay=ANCHOR_BAY, row=0, tier=1))
        env.yard.add_container(blocker2, _yard_placement(bay=ANCHOR_BAY, row=0, tier=2))

        _place_distractors(env, rng, 5, "S14", exclude_bays={ANCHOR_BAY})

        tk = _make_truck("S14_TK1", pickup_ids=[target.container_id])
        env.trucks[tk.truck_id] = tk
        _park_truck_near(env, tk, ANCHOR_BAY)

    def check_success(self, env) -> bool:
        return (env.yard.get_container("S14_TARGET") is None
                and "S14_TK1" not in env.trucks)
