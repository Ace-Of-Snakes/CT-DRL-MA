# simulation/training/scenarios/8_stress/full_mini_day.py
"""Scenario 21: Complete terminal day -- 5 trucks, 1 train, all action types."""
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
    _make_truck,
    _place_distractors,
    _slot_train,
    _yard_placement,
)


class FullMiniDay(TutorialScenario):
    """Maximum integration complexity.

    1 train with 3 Imports + wants 3 Exports.
    3 Export containers in yard.
    2 unparked delivery trucks (bringing 2 more Exports -- only 3 needed by train).
    3 unparked pickup trucks (wanting the Imports).
    8 distractor containers.

    Agent must orchestrate: park 5 trucks, import from train,
    import from delivery trucks, load exports onto train, load imports
    onto pickup trucks. Not all delivery containers go to train (surplus).
    """
    id = 21
    name = "full_mini_day"
    description = "Complete terminal day: 5 trucks, 1 train, all action types"
    max_steps = 100
    repeatable = True

    def setup(self, env) -> None:
        rng = random.Random(42)

        # 3 Import containers on train
        imports = []
        for i in range(3):
            c = _make_container(f"S21_IMP{i}", Direction.IMPORT)
            imports.append(c)

        # 3 Export containers already in yard -> train wants these
        export_ids = []
        for i in range(3):
            c = _make_container(f"S21_EXP{i}", Direction.EXPORT)
            bay = ANCHOR_BAY + i * 6
            env.yard.add_container(c, _yard_placement(bay=bay, row=0, tier=0))
            export_ids.append(c.container_id)

        tr = _make_train("S21_TR1", containers=imports, pickup_ids=export_ids,
                         num_wagons=10)
        _slot_train(env, tr)

        # 2 delivery trucks with surplus Export containers (not in train manifest)
        for i in range(2):
            c = _make_container(f"S21_DEXP{i}", Direction.EXPORT,
                                departure=TUTORIAL_TIME + timedelta(days=10))
            tk = _make_truck(f"S21_DTK{i}", containers=[c])
            env.trucks[tk.truck_id] = tk

        # 3 pickup trucks wanting imports
        for i, imp in enumerate(imports):
            tk = _make_truck(f"S21_PTK{i}", pickup_ids=[imp.container_id])
            env.trucks[tk.truck_id] = tk

        # Distractors for clutter
        _place_distractors(env, rng, 8, "S21",
                           exclude_bays={ANCHOR_BAY + i * 6 for i in range(3)})

    def check_success(self, env) -> bool:
        tr = env.trains.get("S21_TR1")
        if tr is None:
            return False
        # All 3 exports loaded on train
        exports_loaded = all(tr.has_container(f"S21_EXP{i}") for i in range(3))
        # All 3 pickup trucks served and departed
        pickups_served = all(f"S21_PTK{i}" not in env.trucks for i in range(3))
        # Delivery truck containers imported to yard
        deliveries_in_yard = all(
            env.yard.get_container(f"S21_DEXP{i}") is not None
            for i in range(2)
        )
        # Delivery trucks departed
        deliveries_gone = all(f"S21_DTK{i}" not in env.trucks for i in range(2))
        return exports_loaded and pickups_served and deliveries_in_yard and deliveries_gone
