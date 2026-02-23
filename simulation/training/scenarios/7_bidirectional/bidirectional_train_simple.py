# simulation/training/scenarios/7_bidirectional/bidirectional_train_simple.py
"""Scenario 17: Export to train + import from train + deliver to truck (pre-parked)."""
from __future__ import annotations

from simulation.training.scenarios._base import (
    ANCHOR_BAY,
    Direction,
    TutorialScenario,
    _make_container,
    _make_train,
    _make_truck,
    _park_truck_near,
    _slot_train,
    _yard_placement,
)


class BidirectionalTrainSimple(TutorialScenario):
    """1 train with 1 Import + wants 1 Export (already in yard).
    1 pre-parked pickup truck for the import.

    Bridge scenario: teaches concurrent import + export on the SAME
    train at minimal scale, with no parking required.
      - Load 1 Export from yard onto train
      - Import 1 Import from train to yard
      - Load import onto pre-parked pickup truck
    """
    id = 17
    name = "bidirectional_train_simple"
    description = "Export to train + import from train + deliver to truck (pre-parked)"
    max_steps = 35
    repeatable = True

    def setup(self, env) -> None:
        # 1 Import container on the train
        imp = _make_container("S17_IMP0", Direction.IMPORT)

        # 1 Export container in the yard, train wants it
        exp = _make_container("S17_EXP0", Direction.EXPORT)
        env.yard.add_container(exp, _yard_placement(bay=ANCHOR_BAY + 5, row=0, tier=0))

        tr = _make_train("S17_TR1", containers=[imp], pickup_ids=[exp.container_id])
        _slot_train(env, tr)

        # 1 pre-parked pickup truck (no parking step needed)
        tk = _make_truck("S17_TK0", pickup_ids=[imp.container_id])
        env.trucks[tk.truck_id] = tk
        _park_truck_near(env, tk, ANCHOR_BAY)

    def check_success(self, env) -> bool:
        tr = env.trains.get("S17_TR1")
        if tr is None:
            return False
        export_loaded = tr.has_container("S17_EXP0")
        truck_served = "S17_TK0" not in env.trucks
        return export_loaded and truck_served
