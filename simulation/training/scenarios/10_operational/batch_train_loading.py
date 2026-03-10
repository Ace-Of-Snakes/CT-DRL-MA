# simulation/training/scenarios/10_operational/batch_train_loading.py
"""S32 — Batch train loading under time pressure.

Goal: Load 10 export containers onto a train departing in 90 minutes.
Tests the agent's ability to prioritise YARD_TO_TRAIN over YARD_TO_YARD
when train departure is imminent.

Setup:
  - 10 export containers in yard bays 5–32 (clustered near train)
  - 1 train with 90-minute departure window, wants all 10 exports
  - 5 import distractor containers (should be ignored)
  - No trucks — pure yard→train focus

What it teaches:
  - Urgency-driven train loading (high EXPORT_PICKUP_DEMAND signal)
  - Avoid wasting steps on YARD_TO_YARD reshuffles when train is urgent
  - Sequential batch loading pattern
"""
from __future__ import annotations

import random
from datetime import timedelta

from simulation.training.scenarios._base import (
    ANCHOR_BAY,
    TUTORIAL_TIME,
    TutorialScenario,
    _make_container,
    _make_train,
    _place_distractors,
    _slot_train,
    _yard_placement,
)
from simulation.core.enums import Direction


class BatchTrainLoading(TutorialScenario):
    id = 32
    name = "batch_train_loading"
    description = "Load 10 exports onto urgent train (90-min departure)"
    max_steps = 60
    repeatable = True

    def setup(self, env) -> None:
        rng = random.Random(32)

        # 10 export containers clustered near train (bays 5-32)
        export_ids = []
        bays = [5, 8, 11, 14, 17, 20, 23, 26, 29, 32]
        for i, bay in enumerate(bays):
            c = _make_container(
                f"S32_EXP{i}",
                direction=Direction.EXPORT,
                departure=TUTORIAL_TIME + timedelta(hours=2),
                rng=rng,
            )
            # Spread across rows 0-4 (cycle through yard rows)
            row = i % env.yard.n_rows
            env.yard.add_container(c, _yard_placement(bay=bay, row=row, tier=0))
            export_ids.append(c.container_id)

        # Train with 90-min departure — urgency is high but achievable
        train = _make_train(
            "S32_TR1",
            pickup_ids=export_ids,
            num_wagons=10,
        )
        train.departure_time = TUTORIAL_TIME + timedelta(minutes=90)
        _slot_train(env, train, track_id=0, anchor_bay=ANCHOR_BAY)

        # 5 import distractors (should be ignored)
        _place_distractors(env, rng, 5, "S32", exclude_bays=set(bays))

    def check_success(self, env) -> bool:
        tr = env.trains.get("S32_TR1")
        if tr is None:
            return False
        return all(tr.has_container(f"S32_EXP{i}") for i in range(10))
