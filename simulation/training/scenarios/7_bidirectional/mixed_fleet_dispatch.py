# simulation/training/scenarios/7_bidirectional/mixed_fleet_dispatch.py
"""Scenario 19: Park 4 trucks, import/export through yard and train."""
from __future__ import annotations

from simulation.training.scenarios._base import (
    Direction,
    TutorialScenario,
    _make_container,
    _make_train,
    _make_truck,
    _slot_train,
)


class MixedFleetDispatch(TutorialScenario):
    """2 delivery trucks (Export, unparked) + 2 pickup trucks (unparked).
    1 train with 2 Imports + wants 2 Exports.

    All four action types needed:
      SLOT_PARKING (4 trucks)
      IMPORT_VEHICLE (train->yard for imports, truck->yard for exports)
      MOVE_CONTAINER->TRAIN (exports)
      MOVE_CONTAINER->TRUCK (imports)
    """
    id = 19
    name = "mixed_fleet_dispatch"
    description = "Park 4 trucks, import/export through yard and train"
    max_steps = 75
    repeatable = True

    def setup(self, env) -> None:
        # 2 Import containers on train -> pickup trucks will want them
        imports = []
        for i in range(2):
            c = _make_container(f"S19_IMP{i}", Direction.IMPORT)
            imports.append(c)

        # 2 Export containers on delivery trucks -> train wants them
        export_cids = []
        for i in range(2):
            c = _make_container(f"S19_EXP{i}", Direction.EXPORT)
            export_cids.append(c.container_id)
            tk = _make_truck(f"S19_DTK{i}", containers=[c])
            env.trucks[tk.truck_id] = tk

        tr = _make_train("S19_TR1", containers=imports, pickup_ids=export_cids)
        _slot_train(env, tr)

        # 2 pickup trucks wanting the imports
        for i, imp in enumerate(imports):
            tk = _make_truck(f"S19_PTK{i}", pickup_ids=[imp.container_id])
            env.trucks[tk.truck_id] = tk

    def check_success(self, env) -> bool:
        tr = env.trains.get("S19_TR1")
        if tr is None:
            return False
        exports_loaded = all(tr.has_container(f"S19_EXP{i}") for i in range(2))
        pickups_served = all(f"S19_PTK{i}" not in env.trucks for i in range(2))
        deliveries_done = all(f"S19_DTK{i}" not in env.trucks for i in range(2))
        return exports_loaded and pickups_served and deliveries_done
