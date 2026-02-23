# simulation/training/scenarios/5_terminal_truck/tt_only_option.py
"""Scenario 24: TT is only option for swap body; regular truck takes its own container."""
from __future__ import annotations

from simulation.training.scenarios._base import (
    ANCHOR_BAY,
    Direction,
    TutorialScenario,
    _add_terminal_truck,
    _make_container,
    _make_swap_body,
    _make_truck,
    _yard_placement,
)


class TTOnlyOption(TutorialScenario):
    """Swap body in yard + regular Import in yard.
    Regular pickup truck is assigned to the regular container (not the swap body).
    Terminal truck is the only vehicle that can remove the swap body.

    Agent must use TT for swap body AND regular truck for its container.

    Success: swap body removed from yard AND regular truck departs.
    """
    id = 24
    name = "tt_only_option"
    description = "TT is only option for swap body; regular truck takes its own container"
    max_steps = 30
    repeatable = True

    def setup(self, env) -> None:
        # Swap body — only TT can carry it (no regular truck assigned to it)
        sb = _make_swap_body("S24", 0)
        env.yard.add_container(sb, _yard_placement(bay=ANCHOR_BAY, row=0, tier=0))

        # Regular Import — regular truck is assigned to it
        c = _make_container("S24_C1", Direction.IMPORT)
        env.yard.add_container(c, _yard_placement(bay=ANCHOR_BAY + 5, row=0, tier=0))

        # Regular pickup truck wants the regular container (NOT the swap body)
        tk = _make_truck("S24_TK1", pickup_ids=[c.container_id])
        env.trucks[tk.truck_id] = tk

        # Terminal truck — only vehicle that can remove the swap body
        _add_terminal_truck(env, "S24", 0)

    def check_success(self, env) -> bool:
        sb_gone = env.yard.get_container("S24_SB0") is None
        truck_served = "S24_TK1" not in env.trucks
        return sb_gone and truck_served
