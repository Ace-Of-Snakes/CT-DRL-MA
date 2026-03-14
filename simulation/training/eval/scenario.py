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
from typing import Dict, List, Optional, Set, Tuple

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
    n_train_to_truck: int = 0   # imports on train destined for pickup trucks
    n_truck_to_train: int = 0   # exports on delivery trucks destined for trains
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
        if self.n_train_to_truck:
            parts.append(f"t2tk={self.n_train_to_truck}")
        if self.n_truck_to_train:
            parts.append(f"tk2t={self.n_truck_to_train}")
        parts.append(f"fill={self.yard_fill_pct:.0%}")
        return "_".join(parts)

    @property
    def total_containers(self) -> int:
        return (
            self.n_imports
            + self.n_exports
            + self.n_delivery_trucks
            + self.n_pickup_trucks
            + self.n_train_to_truck
            + self.n_truck_to_train
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

        # Tracking for progress checks (populated during setup)
        self._t2tk_truck_ids: List[str] = []
        self._tk2t_map: Dict[str, str] = {}  # {container_id: train_id}

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
        p = self.params
        total_rail = (
            p.n_imports + p.n_exports
            + p.n_train_to_truck + p.n_truck_to_train
        )
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
            + p.n_train_to_truck + p.n_truck_to_train
        )
        reshuffles = max(0, int(productive * _RESHUFFLE_OVERHEAD))

        # Extra reshuffles for buried exports/pickups when yard is filled
        if p.yard_fill_pct > 0:
            buried_fraction = min(0.3, p.yard_fill_pct)
            n_buried = int(
                (p.n_exports + p.n_pickup_trucks) * buried_fraction
            )
            reshuffles += n_buried

        # Average bay spread depends on container count vs yard size
        # More containers → more spread → longer gantry travel
        avg_spread = min(15.0, max(3.0, p.total_containers * 0.5))

        # Budget direct transfers conservatively as via-yard (2 crane moves)
        # Agent may use direct TRAIN_TO_TRUCK / TRUCK_TO_TRAIN (1 move) and
        # finish early, but we must guarantee feasibility for the slower path.
        n_all_trucks = (
            p.n_delivery_trucks + p.n_pickup_trucks
            + p.n_train_to_truck + p.n_truck_to_train
        )

        return WorkloadEstimate(
            n_train_to_yard=p.n_imports + p.n_train_to_truck,
            n_yard_to_train=p.n_exports + p.n_truck_to_train,
            n_park_truck=n_all_trucks,
            n_truck_to_yard=p.n_delivery_trucks + p.n_truck_to_train,
            n_yard_to_truck=p.n_pickup_trucks + p.n_train_to_truck,
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

        # Reset tracking state (setup may be called multiple times)
        self._t2tk_truck_ids = []
        self._tk2t_map = {}

        n_bays = env.yard.n_bays
        n_rows = env.yard.n_rows
        departure = TUTORIAL_TIME + timedelta(seconds=self._feasible_seconds)
        has_fill = p.yard_fill_pct > 0

        # Staggered truck arrivals: spread across the first third of the
        # feasible window so they arrive well before train departure.
        n_total_trucks = (
            p.n_delivery_trucks + p.n_pickup_trucks
            + p.n_train_to_truck + p.n_truck_to_train
        )
        arrival_window_s = self._feasible_seconds / 3.0

        def _truck_arrival(idx: int) -> datetime:
            """Return a staggered arrival time for truck *idx* (0-based)."""
            if n_total_trucks <= 1:
                return TUTORIAL_TIME
            frac = idx / max(1, n_total_trucks - 1)
            return TUTORIAL_TIME + timedelta(seconds=frac * arrival_window_s)

        truck_idx = 0  # running counter across all truck categories

        # ── 1. Export containers in yard (for train pickup) ────
        #    Always use find_single_placement to handle all container sizes
        #    (including 45ft that span multiple bays) and avoid out-of-bounds.
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
                placement = env.yard.find_single_placement(
                    c, target_bay=bay,
                )
                if placement is not None:
                    env.yard.add_container(c, placement)
                export_ids.append(c.container_id)
                active_bays.add(bay)

        # ── 2. Pickup target containers in yard ────────────────
        pickup_ids: List[str] = []
        if p.n_pickup_trucks > 0:
            pickup_bays = self._spread_bays(
                p.n_pickup_trucks, n_bays, rng, exclude=active_bays,
            )
            # Fall back to unrestricted bays when too many are excluded
            # (e.g. 68 exports already claimed all 58 bays).
            if not pickup_bays:
                pickup_bays = self._spread_bays(
                    p.n_pickup_trucks, n_bays, rng,
                )
            for i, bay in enumerate(pickup_bays):
                c = _make_container(
                    f"{prefix}_PKP{i}",
                    direction=Direction.IMPORT,
                    rng=rng,
                )
                placement = env.yard.find_single_placement(
                    c, target_bay=bay,
                )
                if placement is not None:
                    env.yard.add_container(c, placement)
                pickup_ids.append(c.container_id)
                active_bays.add(bay)

        # ── 3. Trains with imports + export/truck-to-train pickups ──
        #    Train-to-truck containers are loaded as imports on the train.
        #    Truck-to-train containers are registered as train pickup IDs.
        t2tk_per_train = self._distribute(p.n_train_to_truck, self._n_trains)
        tk2t_per_train = self._distribute(p.n_truck_to_train, self._n_trains)

        if self._n_trains > 0:
            imports_per_train = self._distribute(p.n_imports, self._n_trains)
            exports_per_train = self._distribute(p.n_exports, self._n_trains)

            export_cursor = 0
            imp_global = 0
            t2tk_global = 0
            tk2t_global = 0

            for t_idx in range(self._n_trains):
                train_id = f"{prefix}_TR{t_idx}"

                # Regular import containers loaded on this train
                train_imports = []
                for _ in range(imports_per_train[t_idx]):
                    c = _make_container(
                        f"{prefix}_IMP{imp_global}",
                        direction=Direction.IMPORT,
                        rng=rng,
                    )
                    train_imports.append(c)
                    imp_global += 1

                # Train-to-truck containers (also loaded as imports)
                for _ in range(t2tk_per_train[t_idx]):
                    c = _make_container(
                        f"{prefix}_T2TK{t2tk_global}",
                        direction=Direction.IMPORT,
                        rng=rng,
                    )
                    train_imports.append(c)
                    t2tk_global += 1

                # Export pickup IDs for this train
                n_exp = exports_per_train[t_idx]
                train_pickup_ids = export_ids[export_cursor:export_cursor + n_exp]
                export_cursor += n_exp

                # Truck-to-train pickup IDs (will be added after trucks)
                tk2t_ids_for_train: List[str] = []
                for _ in range(tk2t_per_train[t_idx]):
                    cid = f"{prefix}_TK2T{tk2t_global}"
                    tk2t_ids_for_train.append(cid)
                    self._tk2t_map[cid] = train_id
                    tk2t_global += 1

                tr = _make_train(
                    train_id,
                    containers=train_imports,
                    pickup_ids=train_pickup_ids + tk2t_ids_for_train,
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
            tk = _make_truck(
                f"{prefix}_DTK{i}", containers=[c],
                arrival_time=_truck_arrival(truck_idx),
            )
            env.trucks[tk.truck_id] = tk
            truck_idx += 1

        # ── 5. Pickup trucks ──────────────────────────────────
        for i in range(p.n_pickup_trucks):
            tk = _make_truck(
                f"{prefix}_PTK{i}",
                pickup_ids=[pickup_ids[i]],
                arrival_time=_truck_arrival(truck_idx),
            )
            env.trucks[tk.truck_id] = tk
            truck_idx += 1

        # ── 6. Train-to-truck pickup trucks ────────────────────
        #    These trucks want specific import containers that arrive
        #    on trains.  Agent can deliver directly (TRAIN_TO_TRUCK)
        #    or route via yard (TRAIN_TO_YARD + YARD_TO_TRUCK).
        for i in range(p.n_train_to_truck):
            cid = f"{prefix}_T2TK{i}"
            truck_id = f"{prefix}_T2TK_TK{i}"
            tk = _make_truck(
                truck_id, pickup_ids=[cid],
                arrival_time=_truck_arrival(truck_idx),
            )
            env.trucks[tk.truck_id] = tk
            self._t2tk_truck_ids.append(truck_id)
            truck_idx += 1

        # ── 7. Truck-to-train delivery trucks ──────────────────
        #    These trucks carry export containers that must reach a
        #    train.  Agent can deliver directly (TRUCK_TO_TRAIN)
        #    or route via yard (TRUCK_TO_YARD + YARD_TO_TRAIN).
        for i in range(p.n_truck_to_train):
            cid = f"{prefix}_TK2T{i}"
            c = _make_container(
                cid,
                direction=Direction.EXPORT,
                departure=departure,
                rng=rng,
            )
            tk = _make_truck(
                f"{prefix}_TK2T_TK{i}", containers=[c],
                arrival_time=_truck_arrival(truck_idx),
            )
            env.trucks[tk.truck_id] = tk
            truck_idx += 1

        # ── 8. Distractors (yard fill) ────────────────────────
        #    When fill > 0, exports and pickups already count toward
        #    fill (placed at tier 0-2 above).  Fill remaining slots.
        if has_fill:
            total_tier0_slots = n_bays * n_rows
            n_fill_target = int(total_tier0_slots * p.yard_fill_pct)
            # Exports + pickups already occupy some slots
            n_already = p.n_exports + p.n_pickup_trucks
            n_distractors = max(0, n_fill_target - n_already)
            if n_distractors > 0:
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

        # Train-to-truck: pickup truck has departed (got its container)
        if p.n_train_to_truck > 0:
            done = sum(
                1 for tk_id in self._t2tk_truck_ids
                if tk_id not in env.trucks
            )
            progress.append(SubTaskProgress(
                "train_to_truck_done", done, p.n_train_to_truck,
            ))

        # Truck-to-train: container successfully loaded onto train.
        # Check is departure-resilient: if the train has departed
        # (popped from env.trains), we verify that the container is
        # no longer in the yard AND no longer on any delivery truck.
        # If absent from both, it must have been loaded before departure.
        if p.n_truck_to_train > 0:
            done = 0
            for cid, train_id in self._tk2t_map.items():
                # Fast path: train still present → direct check
                tr = env.trains.get(train_id)
                if tr is not None and tr.has_container(cid):
                    done += 1
                    continue
                # Slow path: train departed → check by exclusion
                in_yard = env.yard.get_container(cid) is not None
                on_truck = any(
                    any(c.container_id == cid for c in tk.containers)
                    for tk in env.trucks.values()
                )
                if not in_yard and not on_truck:
                    done += 1  # successfully loaded before departure
            progress.append(SubTaskProgress(
                "truck_to_train_done", done, p.n_truck_to_train,
            ))

        return progress

    def check_success(self, env) -> bool:
        """All sub-tasks complete."""
        progress = self.check_progress(env)
        return len(progress) > 0 and all(st.done for st in progress)
