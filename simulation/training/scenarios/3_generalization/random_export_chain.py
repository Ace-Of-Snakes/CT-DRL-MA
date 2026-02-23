# simulation/training/scenarios/3_generalization/random_export_chain.py
"""Scenario 12: Full export chain with random sizes and distractors."""
from __future__ import annotations

import random

from simulation.training.scenarios._base import (
    ANCHOR_BAY,
    CONTAINER_SIZES,
    Direction,
    TutorialScenario,
    _make_container,
    _make_train,
    _make_truck,
    _place_distractors,
    _slot_train,
)


class RandomExportChain(TutorialScenario):
    """Random-size Export on delivery truck, random distractors, train waiting.

    Mirror of S11 for the export path.
    """
    id = 12
    name = "random_export_chain"
    description = "Full export chain with random sizes and distractors"
    max_steps = 40
    repeatable = True

    def setup(self, env) -> None:
        rng = random.Random()
        ft, m, label = rng.choice(CONTAINER_SIZES)
        c = _make_container(
            "S12_C1", Direction.EXPORT,
            length_ft=ft, length_m=m, container_type=label,
        )
        tr = _make_train("S12_TR1", pickup_ids=[c.container_id])
        _slot_train(env, tr)

        n_distractors = rng.randint(3, 8)
        _place_distractors(env, rng, n_distractors, "S12",
                           exclude_bays={ANCHOR_BAY})

        tk = _make_truck("S12_TK1", containers=[c])
        env.trucks[tk.truck_id] = tk

    def check_success(self, env) -> bool:
        tr = env.trains.get("S12_TR1")
        return tr is not None and tr.has_container("S12_C1")
