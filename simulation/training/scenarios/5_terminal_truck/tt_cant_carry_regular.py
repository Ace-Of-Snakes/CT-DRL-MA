# simulation/training/scenarios/5_terminal_truck/tt_cant_carry_regular.py
"""Scenario 22: TT cannot carry regular containers -- use regular truck."""
from __future__ import annotations

from simulation.training.scenarios._base import (
    ANCHOR_BAY,
    Direction,
    TutorialScenario,
    _add_terminal_truck,
    _make_container,
    _make_truck,
    _yard_placement,
)


class TTCantCarryRegular(TutorialScenario):
    """Terminal truck + regular pickup truck in queue.  Regular Import in yard.

    The agent must learn that TTs cannot carry regular containers and
    use the regular pickup truck instead.

    Success: regular truck departs with the container.
    """
    id = 22
    name = "terminal_truck_dispatch"
    description = "Park TT, dispatch swap body via terminal truck"
    max_steps = 15
    repeatable = True

    def setup(self, env) -> None:
        # A regular Import container in the yard
        c = _make_container("S22_C1", Direction.IMPORT)
        env.yard.add_container(c, _yard_placement(bay=ANCHOR_BAY, row=0, tier=0))

        # Regular pickup truck wanting that container
        tk = _make_truck("S22_TK1", pickup_ids=[c.container_id])
        env.trucks[tk.truck_id] = tk

        # Terminal truck (cannot carry regular containers)
        _add_terminal_truck(env, "S22", 0)

    def check_success(self, env) -> bool:
        # Regular truck departed with the container
        return (env.yard.get_container("S22_C1") is None
                and "S22_TK1" not in env.trucks)
