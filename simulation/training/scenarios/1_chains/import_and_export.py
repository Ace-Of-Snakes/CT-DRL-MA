# simulation/training/scenarios/1_chains/import_and_export.py
"""Scenario 5: Import one container from train, load a different export onto it."""
from __future__ import annotations

from simulation.training.scenarios._base import (
    ANCHOR_BAY,
    Direction,
    TutorialScenario,
    _make_container,
    _make_train,
    _slot_train,
    _yard_placement,
)


class ImportAndExport(TutorialScenario):
    """Two independent actions on one train.

    Train carries 1 Import container + wants 1 Export from yard.
    Agent: IMPORT_VEHICLE(train->yard) + MOVE_CONTAINER->TRAIN.
    """
    id = 5
    name = "import_and_export"
    description = "Import one container from train, load a different export onto it"
    max_steps = 25
    repeatable = False

    def setup(self, env) -> None:
        imp = _make_container("S5_IMP", Direction.IMPORT)
        exp = _make_container("S5_EXP", Direction.EXPORT)
        env.yard.add_container(exp, _yard_placement(bay=ANCHOR_BAY, row=0, tier=0))
        tr = _make_train("S5_TR1", containers=[imp], pickup_ids=[exp.container_id])
        _slot_train(env, tr)

    def check_success(self, env) -> bool:
        tr = env.trains.get("S5_TR1")
        if tr is None:
            return False
        export_loaded = tr.has_container("S5_EXP")
        import_in_yard = env.yard.get_container("S5_IMP") is not None
        return export_loaded and import_in_yard
