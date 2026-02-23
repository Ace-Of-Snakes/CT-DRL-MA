# simulation/training/scenarios/7_bidirectional/concurrent_import_export.py
"""Scenario 18: Bidirectional train -- export to train + import to trucks."""
from __future__ import annotations

from simulation.training.scenarios._base import (
    ANCHOR_BAY,
    Direction,
    TutorialScenario,
    _make_container,
    _make_train,
    _make_truck,
    _slot_train,
    _yard_placement,
)


class ConcurrentImportExport(TutorialScenario):
    """1 train with 2 Imports. 2 Exports in yard that train wants.
    2 unparked pickup trucks for the imports.

    Tests bidirectional flow on a shared train:
      - Load 2 Exports from yard onto train
      - Import 2 Imports from train to yard
      - Park 2 pickup trucks, load them
    """
    id = 18
    name = "concurrent_import_export"
    description = "Bidirectional train: export to train + import to trucks"
    max_steps = 65
    repeatable = True

    def setup(self, env) -> None:
        # 2 Import containers on the train
        imports = []
        for i in range(2):
            c = _make_container(f"S18_IMP{i}", Direction.IMPORT)
            imports.append(c)

        # 2 Export containers already in yard, train wants them
        export_ids = []
        for i in range(2):
            c = _make_container(f"S18_EXP{i}", Direction.EXPORT)
            bay = ANCHOR_BAY + 5 + i * 4
            env.yard.add_container(c, _yard_placement(bay=bay, row=0, tier=0))
            export_ids.append(c.container_id)

        tr = _make_train("S18_TR1", containers=imports, pickup_ids=export_ids)
        _slot_train(env, tr)

        # 2 unparked pickup trucks for the imports
        for i, imp in enumerate(imports):
            tk = _make_truck(f"S18_TK{i}", pickup_ids=[imp.container_id])
            env.trucks[tk.truck_id] = tk

    def check_success(self, env) -> bool:
        tr = env.trains.get("S18_TR1")
        if tr is None:
            return False
        exports_loaded = all(tr.has_container(f"S18_EXP{i}") for i in range(2))
        trucks_served = all(f"S18_TK{i}" not in env.trucks for i in range(2))
        return exports_loaded and trucks_served
