# simulation/training/scenarios/3_generalization/random_import_chain.py
"""Scenario 11: Full import chain with random sizes and distractors."""
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


class RandomImportChain(TutorialScenario):
    """Random-size Import on train, random distractors, unparked pickup truck.

    Different every run to force generalization over container sizes
    and yard layouts. No fixed RNG seed.
    """
    id = 11
    name = "random_import_chain"
    description = "Full import chain with random sizes and distractors"
    max_steps = 40
    repeatable = True

    def setup(self, env) -> None:
        rng = random.Random()  # unseeded -> different every run
        ft, m, label = rng.choice(CONTAINER_SIZES)
        c = _make_container(
            "S11_C1", Direction.IMPORT,
            length_ft=ft, length_m=m, container_type=label,
        )
        tr = _make_train("S11_TR1", containers=[c])
        _slot_train(env, tr)

        n_distractors = rng.randint(3, 8)
        _place_distractors(env, rng, n_distractors, "S11",
                           exclude_bays={ANCHOR_BAY})

        tk = _make_truck("S11_TK1", pickup_ids=[c.container_id])
        env.trucks[tk.truck_id] = tk

    def check_success(self, env) -> bool:
        return "S11_TK1" not in env.trucks
