# simulation/training/scenarios/10_operational/mixed_urgency_ops.py
"""S33 — Mixed urgency: imports + exports + truck parking under time pressure.

Goal: Handle all three operation types simultaneously with an urgent train.
Tests interleaving parking, importing, and exporting without getting stuck
on reshuffles.

Setup:
  - 5 export containers in yard (train wants them)
  - 1 train with 2-hour departure carrying 5 import containers
  - 3 delivery trucks in queue waiting to park (bring extra exports)
  - No distractors — clean but multi-modal

Success criteria:
  - All 5 exports loaded on train
  - All 5 imports moved to yard
  - All 3 delivery trucks parked and containers unloaded

What it teaches:
  - Interleave PARK_TRUCK + TRUCK_TO_YARD + TRAIN_TO_YARD + YARD_TO_TRAIN
  - Avoid getting stuck in reshuffle loops when multiple priorities compete
  - Time management: train exports are urgent, truck parking is less so
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
    _make_truck,
    _slot_train,
    _yard_placement,
)
from simulation.core.enums import Direction


class MixedUrgencyOps(TutorialScenario):
    id = 33
    name = "mixed_urgency_ops"
    description = "Mixed ops: 5 exports to train + 5 imports to yard + 3 truck parks"
    max_steps = 80
    repeatable = True

    def setup(self, env) -> None:
        rng = random.Random(33)

        # ── 5 export containers in yard (train wants them) ──────────
        export_ids = []
        export_bays = [ANCHOR_BAY + i * 8 for i in range(5)]  # 5,13,21,29,37
        for i, bay in enumerate(export_bays):
            c = _make_container(
                f"S33_EXP{i}",
                direction=Direction.EXPORT,
                departure=TUTORIAL_TIME + timedelta(hours=2),
                rng=rng,
            )
            env.yard.add_container(c, _yard_placement(bay=bay, row=0, tier=0))
            export_ids.append(c.container_id)

        # ── 1 train with 2-hour departure, carrying 5 imports ──────
        imports = []
        for i in range(5):
            c = _make_container(
                f"S33_IMP{i}",
                direction=Direction.IMPORT,
                departure=TUTORIAL_TIME + timedelta(days=5),
                rng=rng,
            )
            imports.append(c)

        train = _make_train(
            "S33_TR1",
            containers=imports,
            pickup_ids=export_ids,
            num_wagons=10,
        )
        train.departure_time = TUTORIAL_TIME + timedelta(hours=2)
        _slot_train(env, train, track_id=0, anchor_bay=ANCHOR_BAY)

        # ── 3 delivery trucks in queue (bring extra export containers) ──
        for i in range(3):
            c = _make_container(
                f"S33_DEXP{i}",
                direction=Direction.EXPORT,
                departure=TUTORIAL_TIME + timedelta(days=3),
                rng=rng,
            )
            tk = _make_truck(f"S33_DTK{i}", containers=[c])
            env.trucks[tk.truck_id] = tk

    def check_success(self, env) -> bool:
        tr = env.trains.get("S33_TR1")
        if tr is None:
            return False

        # All 5 exports loaded on train
        exports_loaded = all(
            tr.has_container(f"S33_EXP{i}") for i in range(5)
        )

        # All 5 imports in yard (off train)
        imports_in_yard = all(
            env.yard.get_container(f"S33_IMP{i}") is not None
            for i in range(5)
        )

        # All 3 delivery trucks parked and containers unloaded to yard
        deliveries_stored = all(
            env.yard.get_container(f"S33_DEXP{i}") is not None
            for i in range(3)
        )
        trucks_done = all(
            f"S33_DTK{i}" not in env.trucks for i in range(3)
        )

        return exports_loaded and imports_in_yard and deliveries_stored and trucks_done
