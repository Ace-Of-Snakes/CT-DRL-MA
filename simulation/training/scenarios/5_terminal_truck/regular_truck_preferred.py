# simulation/training/scenarios/5_terminal_truck/regular_truck_preferred.py
"""Scenario 23: Prefer regular truck over TT for swap body (higher reward)."""
from __future__ import annotations

from simulation.training.scenarios._base import (
    ANCHOR_BAY,
    TutorialScenario,
    _add_terminal_truck,
    _make_swap_body,
    _make_truck,
    _yard_placement,
)


class RegularTruckPreferred(TutorialScenario):
    """Swap body in yard.  Both a regular pickup truck (assigned to it)
    and a terminal truck are in the queue.

    Regular truck gives +3.0 reward vs TT's +2.0.  Agent should learn
    to prefer the regular truck when it is an option.

    Success: regular truck departs with the swap body.
    """
    id = 23
    name = "regular_truck_preferred"
    description = "Prefer regular truck over TT for swap body (higher reward)"
    max_steps = 20
    repeatable = True

    def setup(self, env) -> None:
        sb = _make_swap_body("S23", 0)
        env.yard.add_container(sb, _yard_placement(bay=ANCHOR_BAY, row=0, tier=0))

        # Regular pickup truck that wants the swap body
        tk = _make_truck("S23_TK1", pickup_ids=[sb.container_id])
        env.trucks[tk.truck_id] = tk

        # Terminal truck (also could carry it, but less reward)
        _add_terminal_truck(env, "S23", 0)

    def check_success(self, env) -> bool:
        # Regular truck departed with the swap body
        return (env.yard.get_container("S23_SB0") is None
                and "S23_TK1" not in env.trucks)
