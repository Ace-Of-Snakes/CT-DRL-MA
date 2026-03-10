# simulation/training/eval/metrics.py
"""Dataclasses for parameterized evaluation results."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class SubTaskProgress:
    """Progress on a single sub-task within an eval scenario."""

    name: str
    completed: int
    total: int

    @property
    def pct(self) -> float:
        return self.completed / max(1, self.total)

    @property
    def done(self) -> bool:
        return self.completed >= self.total


@dataclass
class EvalRunResult:
    """Rich result from a single evaluation run."""

    # Identity
    params: "EvalParams"  # forward ref to avoid circular import
    repeat: int = 0

    # Outcome
    passed: bool = False
    steps: int = 0
    agent_moves: int = 0
    total_reward: float = 0.0

    # Granular progress
    sub_tasks: List[SubTaskProgress] = field(default_factory=list)

    # Move distribution
    move_type_counts: Dict[str, int] = field(default_factory=dict)

    # Timing
    wall_time_s: float = 0.0

    # ── Derived metrics ──────────────────────────────────────────

    @property
    def completion_pct(self) -> float:
        """Overall completion percentage across all sub-tasks."""
        total = sum(st.total for st in self.sub_tasks)
        completed = sum(st.completed for st in self.sub_tasks)
        return completed / max(1, total)

    @property
    def total_containers(self) -> int:
        return (
            self.params.n_imports
            + self.params.n_exports
            + self.params.n_delivery_trucks
            + self.params.n_pickup_trucks
        )

    @property
    def moves_per_container(self) -> float:
        return self.agent_moves / max(1, self.total_containers)

    @property
    def reshuffle_count(self) -> int:
        return self.move_type_counts.get("YARD_TO_YARD", 0)

    @property
    def reshuffle_ratio(self) -> float:
        """Fraction of moves that are YARD_TO_YARD."""
        return self.reshuffle_count / max(1, self.agent_moves)

    @property
    def productive_move_ratio(self) -> float:
        """Fraction of moves that are NOT reshuffles."""
        return 1.0 - self.reshuffle_ratio
