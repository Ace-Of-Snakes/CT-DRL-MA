# simulation/training/scenarios/9_complex_ops/full_bidirectional_day.py
"""S31 — Full bidirectional day (capstone benchmark).

1 train carrying 4 imports + wanting 4 exports from yard.
4 unparked pickup trucks for the imports.
2 delivery trucks each bringing 1 extra export (stored in yard).
10 distractors.

Every move type in a single scenario.  This is the hardest tutorial
benchmark — validates the agent can orchestrate the complete terminal
workflow.
"""
import random
from datetime import timedelta

from simulation.core.enums import Direction
from simulation.training.scenarios._base import (
    TutorialScenario,
    _make_container,
    _make_train,
    _make_truck,
    _place_distractors,
    _slot_train,
    _yard_placement,
    ANCHOR_BAY,
    TUTORIAL_TIME,
)


class FullBidirectionalDay(TutorialScenario):
    id = 31
    name = "full_bidirectional_day"
    description = "Full bidirectional: 4 imports to trucks, 4 exports to train, 2 deliveries"
    max_steps = 100
    repeatable = True

    def setup(self, env) -> None:
        rng = random.Random(31)

        # ── 4 Import containers on the train ───────────────────────────
        imports = []
        for i in range(4):
            c = _make_container(f"S31_IMP{i}", Direction.IMPORT, rng=rng)
            imports.append(c)

        # ── 4 Export containers already in yard → train wants them ──────
        export_ids = []
        export_bays = [ANCHOR_BAY + i * 7 for i in range(4)]
        for i in range(4):
            c = _make_container(f"S31_EXP{i}", Direction.EXPORT, rng=rng)
            env.yard.add_container(c, _yard_placement(bay=export_bays[i], row=0, tier=0))
            export_ids.append(c.container_id)

        # ── Train: carries imports, wants exports ──────────────────────
        tr = _make_train(
            "S31_TR1", containers=imports, pickup_ids=export_ids,
            num_wagons=10,
        )
        _slot_train(env, tr)

        # ── 4 unparked pickup trucks for the imports ───────────────────
        for i, imp in enumerate(imports):
            tk = _make_truck(f"S31_PTK{i}", pickup_ids=[imp.container_id])
            env.trucks[tk.truck_id] = tk

        # ── 2 delivery trucks each bringing 1 extra export ─────────────
        for i in range(2):
            c = _make_container(
                f"S31_DEXP{i}", Direction.EXPORT,
                departure=TUTORIAL_TIME + timedelta(days=10),
                rng=rng,
            )
            tk = _make_truck(f"S31_DTK{i}", containers=[c])
            env.trucks[tk.truck_id] = tk

        # ── Distractors ───────────────────────────────────────────────
        _place_distractors(env, rng, 10, "S31", exclude_bays=set(export_bays))

    def check_success(self, env) -> bool:
        tr = env.trains.get("S31_TR1")
        if tr is None:
            return False

        # All 4 exports loaded on train
        exports_loaded = all(
            tr.has_container(f"S31_EXP{i}") for i in range(4)
        )
        # All 4 pickup trucks served and departed
        pickups_gone = all(
            f"S31_PTK{i}" not in env.trucks for i in range(4)
        )
        # Delivery containers stored in yard
        deliveries_in_yard = all(
            env.yard.get_container(f"S31_DEXP{i}") is not None
            for i in range(2)
        )
        # Delivery trucks departed
        deliveries_gone = all(
            f"S31_DTK{i}" not in env.trucks for i in range(2)
        )
        return exports_loaded and pickups_gone and deliveries_in_yard and deliveries_gone
