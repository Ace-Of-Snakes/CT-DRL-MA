# simulation/training/scenarios/9_complex_ops/full_train_load.py
"""S29 — Fully load a train (6 export containers from yard → train).

3 containers are directly accessible, 3 are each buried under 1 blocker.
The agent must restack blockers (YARD_TO_YARD) then load (YARD_TO_TRAIN).
This tests the longest restack-to-load dependency chains in the tutorial
system (~12-18 moves total).
"""
import random
from datetime import timedelta

from simulation.core.enums import Direction
from simulation.training.scenarios._base import (
    TutorialScenario,
    _make_container,
    _make_train,
    _place_distractors,
    _random_container_spec,
    _slot_train,
    _yard_placement,
    ANCHOR_BAY,
    TUTORIAL_TIME,
)


class FullTrainLoad(TutorialScenario):
    id = 29
    name = "full_train_load"
    description = "Load 6 exports onto train (3 accessible, 3 buried under blockers)"
    max_steps = 60
    repeatable = True

    def setup(self, env) -> None:
        rng = random.Random(29)

        export_ids = []
        bays = [ANCHOR_BAY, ANCHOR_BAY + 6, ANCHOR_BAY + 12,
                ANCHOR_BAY + 18, ANCHOR_BAY + 24, ANCHOR_BAY + 30]

        for i in range(6):
            # All containers in a stack must share the same size
            ft, m, ctype, _w, _h, _hc = _random_container_spec(rng)

            target = _make_container(
                f"S29_EXP{i}", Direction.EXPORT,
                length_ft=ft, length_m=m, container_type=ctype,
            )
            env.yard.add_container(target, _yard_placement(bay=bays[i], row=0, tier=0))
            export_ids.append(target.container_id)

            # First 3 are accessible (tier 0 only).
            # Last 3 have a blocker on top (tier 1).
            if i >= 3:
                blocker = _make_container(
                    f"S29_BLK{i}", Direction.IMPORT,
                    departure=TUTORIAL_TIME + timedelta(days=15),
                    length_ft=ft, length_m=m, container_type=ctype,
                )
                env.yard.add_container(
                    blocker, _yard_placement(bay=bays[i], row=0, tier=1),
                )

        # Train that wants all 6 exports
        tr = _make_train("S29_TR1", pickup_ids=export_ids, num_wagons=10)
        _slot_train(env, tr)

        # Distractors
        _place_distractors(env, rng, 8, "S29",
                           exclude_bays=set(bays))

    def check_success(self, env) -> bool:
        tr = env.trains.get("S29_TR1")
        if tr is None:
            return False
        return all(tr.has_container(f"S29_EXP{i}") for i in range(6))
