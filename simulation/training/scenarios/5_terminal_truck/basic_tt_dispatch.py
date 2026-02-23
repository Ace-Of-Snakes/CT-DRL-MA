# simulation/training/scenarios/5_terminal_truck/basic_tt_dispatch.py
"""Scenario 25: Park TT and dispatch swap body (simplest TT case)."""
from __future__ import annotations

from simulation.training.scenarios._base import (
    ANCHOR_BAY,
    TutorialScenario,
    _add_terminal_truck,
    _make_swap_body,
    _yard_placement,
)


class BasicTTDispatch(TutorialScenario):
    """Simplest terminal truck scenario: swap body in yard, only a TT available.

    No competing trucks, no regular containers.  Agent just needs to
    park the TT and dispatch the swap body to it.

    Success: swap body removed from yard.
    """
    id = 25
    name = "basic_tt_dispatch"
    description = "Park TT and dispatch swap body (simplest TT case)"
    max_steps = 15
    repeatable = False

    def setup(self, env) -> None:
        sb = _make_swap_body("S25", 0)
        env.yard.add_container(sb, _yard_placement(bay=ANCHOR_BAY, row=0, tier=0))

        # Only a terminal truck — no regular trucks
        _add_terminal_truck(env, "S25", 0)

    def check_success(self, env) -> bool:
        return env.yard.get_container("S25_SB0") is None
