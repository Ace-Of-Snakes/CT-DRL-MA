# simulation/training/scenarios/1_chains/park_then_import.py
"""Scenario 6: Park delivery truck, then import its container to yard."""
from __future__ import annotations

from simulation.training.scenarios._base import (
    Direction,
    TutorialScenario,
    _make_container,
    _make_truck,
)


class ParkThenImport(TutorialScenario):
    """Delivery truck with Export container (unparked), plus train.

    Agent: SLOT_PARKING -> IMPORT_VEHICLE(truck->yard).
    """
    id = 6
    name = "park_then_import"
    description = "Park delivery truck, then import its container to yard"
    max_steps = 25
    repeatable = False

    def setup(self, env) -> None:
        c = _make_container("S6_C1", Direction.EXPORT)
        tk = _make_truck("S6_TK1", containers=[c])
        env.trucks[tk.truck_id] = tk

    def check_success(self, env) -> bool:
        return (env.yard.get_container("S6_C1") is not None
                and "S6_TK1" not in env.trucks)
