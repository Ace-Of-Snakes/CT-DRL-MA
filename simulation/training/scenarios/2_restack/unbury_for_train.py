# simulation/training/scenarios/2_restack/unbury_for_train.py
"""Scenario 10: Restack blocker to reach buried Export, load onto train."""
from __future__ import annotations

import random
from datetime import timedelta

from simulation.training.scenarios._base import (
    ANCHOR_BAY,
    TUTORIAL_TIME,
    Direction,
    TutorialScenario,
    _make_container,
    _make_train,
    _place_distractors,
    _random_container_spec,
    _slot_train,
    _yard_placement,
)


class UnburyForTrain(TutorialScenario):
    """Export target buried under blocker, 5 distractors, train wants it.

    Agent: RESTACK blocker -> MOVE_CONTAINER->TRAIN.
    """
    id = 10
    name = "unbury_for_train"
    description = "Restack blocker to reach buried Export, load onto train"
    max_steps = 25
    repeatable = True

    def setup(self, env) -> None:
        rng = random.Random(99)
        # All stacked containers must share the same size
        ft, m, ctype, _w, _h, _hc = _random_container_spec(rng)
        target = _make_container("S10_TARGET", Direction.EXPORT,
                                  length_ft=ft, length_m=m, container_type=ctype)
        blocker = _make_container("S10_BLK1", Direction.EXPORT,
                                   departure=TUTORIAL_TIME + timedelta(days=10),
                                   length_ft=ft, length_m=m, container_type=ctype)
        env.yard.add_container(target,  _yard_placement(bay=ANCHOR_BAY, row=0, tier=0))
        env.yard.add_container(blocker, _yard_placement(bay=ANCHOR_BAY, row=0, tier=1))

        _place_distractors(env, rng, 5, "S10", exclude_bays={ANCHOR_BAY})

        tr = _make_train("S10_TR1", pickup_ids=[target.container_id])
        _slot_train(env, tr)

    def check_success(self, env) -> bool:
        tr = env.trains.get("S10_TR1")
        return (tr is not None
                and tr.has_container("S10_TARGET")
                and env.yard.get_container("S10_TARGET") is None)
