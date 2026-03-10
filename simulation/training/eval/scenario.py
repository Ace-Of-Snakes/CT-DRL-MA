# simulation/training/eval/scenario.py
"""Parameterized evaluation scenario with guaranteed feasibility.

Usage::

    from simulation.training.eval import ScalableEvalScenario, EvalParams
    sc = ScalableEvalScenario(EvalParams(n_imports=20, n_exports=20, seed=42))
    # sc is a standard TutorialScenario — run with TutorialRunner.run_scenario()
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import timedelta
from typing import List, Optional, Set, Tuple

from simulation.core.enums import Direction
from simulation.training.eval.metrics import SubTaskProgress
from simulation.training.eval.timing import WorkloadEstimate, estimate_feasible_time
from simulation.training.scenarios._base import (
    TUTORIAL_TIME,
    TutorialScenario,
    _make_container,
    _make_train,
    _make_truck,
    _place_distractors,
    _slot_train,
    _yard_placement,
)

# ================================================================
# Constants
# ================================================================

# Conservative avg containers per wagon (mixed 20ft–45ft sizes)
_CONTAINERS_PER_WAGON: float = 1.5
_MAX_WAGONS_PER_TRACK: int = 29
_MAX_CONTAINERS_PER_TRAIN: int = int(_MAX_WAGONS_PER_TRACK * _CONTAINERS_PER_WAGON)
_MAX_TRACKS: int = 7

# Reshuffle overhead estimate (fraction of productive moves)
_RESHUFFLE_OVERHEAD: float = 0.15


# ================================================================
# EvalParams
# ================================================================

@dataclass
class EvalParams:
    """Parameters for a single evaluation scenario."""

    n_imports: int = 10
    n_exports: int = 10
    n_delivery_trucks: int = 0
    n_pickup_trucks: int = 0
    yard_fill_pct: float = 0.0  # 0.0 to 0.7
    seed: int = 42
    safety_factor: float = 2.0

    @property
    def label(self) -> str:
        """Human-readable label for this parameter set."""
        parts = [f"imp={self.n_imports}", f"exp={self.n_exports}"]
        if self.n_delivery_trucks:
            parts.append(f"dtk={self.n_delivery_trucks}")
        if self.n_pickup_trucks:
            parts.append(f"ptk={self.n_pickup_trucks}")
        parts.append(f"fill={self.yard_fill_pct:.0%}")
        return "_".join(parts)

    @property
    def total_containers(self) -> int:
        return (
            self.n_imports
            + self.n_exports
            + self.n_delivery_trucks
            + self.n_pickup_trucks
        )


# ================================================================
# ScalableEvalScenario
# ================================================================

class ScalableEvalScenario(TutorialScenario):
    """A parameterized evaluation scenario with guaranteed feasibility.

    Departure times are computed from actual crane physics so the task
    is always achievable by a competent agent.  The ``safety_factor``
    (default 2.0) controls how generous the time budget is.
    """

    def __init__(self, params: EvalParams):
        self.params = params
        self._prefix = f"EVAL{params.seed}"

        # Plan trains
        self._n_trains, self._wagons_per_train = self._plan_trains()

        # Compute workload and timing
        self._workload = self._compute_workload()
        self._feasible_seconds = estimate_feasible_time(
            self._workload, safety_factor=params.safety_factor,
        )
        # Minimum 10 minutes so very small scenarios don't have 0-second windows
        self._feasible_seconds = max(600.0, self._feasible_seconds)

        # TutorialScenario interface
        self.id = 9000 + params.seed % 1000
        self.name = f"eval_{params.label}"
        self.description = f"Eval: {params.label}"
        self.max_steps = self._compute_max_steps()
        self.repeatable = True
        self.expected_moves = None

    # ── Train planning ─────────────────────────────────────────

    def _plan_trains(self) -> Tuple[int, List[int]]:
        """Determine number of trains and wagons per train."""
        total_rail = self.params.n_imports + self.params.n_exports
        if total_rail == 0:
            return 0, []

        n_trains = max(1, math.ceil(total_rail / _MAX_CONTAINERS_PER_TRAIN))
        n_trains = min(n_trains, _MAX_TRACKS)

        per_train = math.ceil(total_rail / n_trains)
        wagons_needed = math.ceil(per_train / _CONTAINERS_PER_WAGON)
        wagons_needed = min(wagons_needed, _MAX_WAGONS_PER_TRACK)

        return n_trains, [wagons_needed] * n_trains

    # ── Workload estimation ────────────────────────────────────

    def _compute_workload(self) -> WorkloadEstimate:
        p = self.params
        productive = (
            p.n_imports + p.n_exports
            + p.n_delivery_trucks + p.n_pickup_trucks
        )
        reshuffles = max(0, int(productive * _RESHUFFLE_OVERHEAD))

        # Average bay spread depends on container count vs yard size
        # More containers → more spread → longer gantry travel
        avg_spread = min(15.0, max(3.0, p.total_containers * 0.5))

        return WorkloadEstimate(
            n_train_to_yard=p.n_imports,
            n_yard_to_train=p.n_exports,
            n_park_truck=p.n_delivery_trucks + p.n_pickup_trucks,
            n_truck_to_yard=p.n_delivery_trucks,
            n_yard_to_truck=p.n_pickup_trucks,
            n_yard_to_yard=reshuffles,
            avg_bay_spread=avg_spread,
        )

    def _compute_max_steps(self) -> int:
        """Generous step budget based on total expected moves."""
        total = self._workload.total_moves
        # Each step can produce 2 moves (2 cranes), give 3× buffer
        return max(30, math.ceil(total * 3.0))

    # ── Bay layout helpers ─────────────────────────────────────

    @staticmethod
    def _spread_bays(
        n: int,
        n_bays: int,
        rng: random.Random,
        exclude: Optional[Set[int]] = None,
    ) -> List[int]:
        """Spread n items across the bay range, avoiding excluded bays."""
        exclude = exclude or set()
        available = [b for b in range(n_bays) if b not in exclude]
        if not available or n == 0:
            return []
        if n <= len(available):
            step = max(1, len(available) // n)
            selected = available[::step][:n]
        else:
            # More items than bays: allow repeats (different rows)
            selected = (available * math.ceil(n / len(available)))[:n]
        rng.shuffle(selected)
        return selected

    @staticmethod
    def _distribute(total: int, n_buckets: int) -> List[int]:
        """Distribute total items across n_buckets as evenly as possible."""
        if n_buckets == 0:
            return []
        base = total // n_buckets
        remainder = total % n_buckets
        return [base + (1 if i < remainder else 0) for i in range(n_buckets)]

    @staticmethod
    def _train_anchor(train_idx: int, n_bays: int, n_trains: int) -> int:
        """Spread train anchors across bay range."""
        if n_trains <= 1:
            return 5  # default ANCHOR_BAY
        step = max(1, (n_bays - 10) // n_trains)
        return min(5 + train_idx * step, n_bays - 5)

    # ── Setup ──────────────────────────────────────────────────

    def setup(self, env) -> None:  # noqa: C901 (complexity is inherent)
        rng = random.Random(self.params.seed)
        p = self.params
        prefix = self._prefix
        active_bays: Set[int] = set()

        n_bays = env.yard.n_bays
        n_rows = env.yard.n_rows
        departure = TUTORIAL_TIME + timedelta(seconds=self._feasible_seconds)

        # ── 1. Export containers in yard (for train pickup) ────
        export_ids: List[str] = []
        if p.n_exports > 0:
            export_bays = self._spread_bays(p.n_exports, n_bays, rng)
            for i, bay in enumerate(export_bays):
                c = _make_container(
                    f"{prefix}_EXP{i}",
                    direction=Direction.EXPORT,
                    departure=departure,
                    rng=rng,
                )
                row = i % n_rows
                env.yard.add_container(c, _yard_placement(bay=bay, row=row, tier=0))
                export_ids.append(c.container_id)
                active_bays.add(bay)

        # ── 2. Pickup target containers in yard ────────────────
        pickup_ids: List[str] = []
        if p.n_pickup_trucks > 0:
            pickup_bays = self._spread_bays(
                p.n_pickup_trucks, n_bays, rng, exclude=active_bays,
            )
            for i, bay in enumerate(pickup_bays):
                c = _make_container(
                    f"{prefix}_PKP{i}",
                    direction=Direction.IMPORT,
                    rng=rng,
                )
                row = (i + p.n_exports) % n_rows
                env.yard.add_container(c, _yard_placement(bay=bay, row=row, tier=0))
                pickup_ids.append(c.container_id)
                active_bays.add(bay)

        # ── 3. Trains with imports + export pickup lists ───────
        if self._n_trains > 0:
            imports_per_train = self._distribute(p.n_imports, self._n_trains)
            exports_per_train = self._distribute(p.n_exports, self._n_trains)

            export_cursor = 0
            imp_global = 0
            for t_idx in range(self._n_trains):
                # Import containers loaded on this train
                train_imports = []
                for _ in range(imports_per_train[t_idx]):
                    c = _make_container(
                        f"{prefix}_IMP{imp_global}",
                        direction=Direction.IMPORT,
                        rng=rng,
                    )
                    train_imports.append(c)
                    imp_global += 1

                # Export pickup IDs for this train
                n_exp = exports_per_train[t_idx]
                train_pickup_ids = export_ids[export_cursor:export_cursor + n_exp]
                export_cursor += n_exp

                tr = _make_train(
                    f"{prefix}_TR{t_idx}",
                    containers=train_imports,
                    pickup_ids=train_pickup_ids,
                    num_wagons=self._wagons_per_train[t_idx],
                )
                tr.departure_time = departure

                anchor = self._train_anchor(t_idx, n_bays, self._n_trains)
                _slot_train(env, tr, track_id=t_idx, anchor_bay=anchor)

        # ── 4. Delivery trucks ─────────────────────────────────
        for i in range(p.n_delivery_trucks):
            c = _make_container(
                f"{prefix}_DEL{i}",
                direction=Direction.EXPORT,
                departure=TUTORIAL_TIME + timedelta(days=10),
                rng=rng,
            )
            tk = _make_truck(f"{prefix}_DTK{i}", containers=[c])
            env.trucks[tk.truck_id] = tk

        # ── 5. Pickup trucks ──────────────────────────────────
        for i in range(p.n_pickup_trucks):
            tk = _make_truck(
                f"{prefix}_PTK{i}",
                pickup_ids=[pickup_ids[i]],
            )
            env.trucks[tk.truck_id] = tk

        # ── 6. Distractors (yard fill) ────────────────────────
        if p.yard_fill_pct > 0:
            total_tier0_slots = n_bays * n_rows
            n_distractors = int(total_tier0_slots * p.yard_fill_pct)
            _place_distractors(
                env, rng, n_distractors, prefix,
                exclude_bays=active_bays,
            )

    # ── Success / progress checking ────────────────────────────

    def check_progress(self, env) -> List[SubTaskProgress]:
        """Return granular progress on each sub-task."""
        p = self.params
        prefix = self._prefix
        progress: List[SubTaskProgress] = []

        # Imports unloaded: import container in yard = success
        if p.n_imports > 0:
            done = sum(
                1 for i in range(p.n_imports)
                if env.yard.get_container(f"{prefix}_IMP{i}") is not None
            )
            progress.append(SubTaskProgress("imports_unloaded", done, p.n_imports))

        # Exports loaded: export container NOT in yard = loaded onto train
        if p.n_exports > 0:
            done = sum(
                1 for i in range(p.n_exports)
                if env.yard.get_container(f"{prefix}_EXP{i}") is None
            )
            progress.append(SubTaskProgress("exports_loaded", done, p.n_exports))

        # Delivery trucks: container in yard = unloaded
        if p.n_delivery_trucks > 0:
            done = sum(
                1 for i in range(p.n_delivery_trucks)
                if env.yard.get_container(f"{prefix}_DEL{i}") is not None
            )
            progress.append(SubTaskProgress("deliveries_served", done, p.n_delivery_trucks))

        # Pickup trucks: container NOT in yard = picked up
        if p.n_pickup_trucks > 0:
            done = sum(
                1 for i in range(p.n_pickup_trucks)
                if env.yard.get_container(f"{prefix}_PKP{i}") is None
            )
            progress.append(SubTaskProgress("pickups_served", done, p.n_pickup_trucks))

        return progress

    def check_success(self, env) -> bool:
        """All sub-tasks complete."""
        progress = self.check_progress(env)
        return len(progress) > 0 and all(st.done for st in progress)
