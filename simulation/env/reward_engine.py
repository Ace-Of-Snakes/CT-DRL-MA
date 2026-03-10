# simulation/env/reward_engine.py
"""Reward calculation engine for terminal operations."""
from datetime import datetime
from typing import Optional, Set

from simulation.core.facilities.yard import StorageYard
from simulation.core.vehicles.train import Train
from simulation.core.enums import MoveType
from simulation.config.operations_config import RewardWeights


# Dispatch: MoveType.value -> attribute name on RewardWeights
_MOVE_BONUS_ATTR = {
    MoveType.YARD_TO_TERMINAL_TRUCK.value: "reward_terminal_truck",
    MoveType.YARD_TO_TRUCK.value: "reward_yard_to_truck",
    MoveType.TRAIN_TO_TRUCK.value: "reward_train_to_truck",
    MoveType.YARD_TO_TRAIN.value: "reward_yard_to_train",
    MoveType.TRAIN_TO_YARD.value: "reward_train_to_yard",
    MoveType.TRUCK_TO_YARD.value: "reward_truck_to_yard",
    MoveType.TRUCK_TO_TRAIN.value: "reward_truck_to_train",
    MoveType.YARD_TO_YARD.value: "penalty_yard_to_yard",
}

_EPSILON = 1e-6
MAX_URGENCY_HOURS: float = 12.0  # matches state encoder's MAX_TRAIN_DEPARTURE_HOURS


class RewardEngine:
    """Computes rewards for container moves, truck waits, train departures, and end-of-day state."""

    def __init__(self, yard: StorageYard, weights: RewardWeights = None):
        self.yard = yard
        self.weights = weights or RewardWeights()
        self._departed_trains: Set[str] = set()

    # ==================== Move Rewards ====================

    def immediate_reward(
        self,
        move_type: str,
        distance_m: float,
        time_s: float,
        now: Optional[datetime] = None,
        train: Optional[Train] = None,
    ) -> float:
        """Base cost (distance + time) plus move-specific bonus.

        When *train* and *now* are provided for train-loading moves
        (YARD_TO_TRAIN, TRUCK_TO_TRAIN), the bonus is scaled by departure
        urgency: ``bonus *= (1 + urgency)``.  A flat +7 becomes up to +14
        when the train departs imminently.
        """
        w = self.weights
        base = w.per_meter_cost * distance_m + w.per_second_cost * time_s
        attr = _MOVE_BONUS_ATTR.get(move_type)
        bonus = getattr(w, attr, 0.0) if attr else 0.0
        # Urgency scaling for train-loading moves
        if train is not None and now is not None and move_type in (
            MoveType.YARD_TO_TRAIN.value, MoveType.TRUCK_TO_TRAIN.value,
        ):
            urgency = self._train_urgency(train, now)
            bonus *= (1.0 + urgency)  # +7.0 → up to +14.0
        return base + bonus

    def _train_urgency(self, train: Train, now: datetime) -> float:
        """0.0 = train departs in 12+ hours, 1.0 = departing NOW."""
        dep = getattr(train, "departure_time", None)
        if dep is None:
            return 0.0
        hours_until = max(0.0, (dep - now).total_seconds() / 3600.0)
        return 1.0 - min(1.0, hours_until / MAX_URGENCY_HOURS)

    # ==================== Truck Wait Rewards ====================

    def truck_first_service_reward(self, wait_minutes: float) -> float:
        """Reward shaping when a truck receives its first container."""
        return self._compute_wait_reward(wait_minutes)

    def truck_wait_reward(self, wait_minutes: float) -> float:
        """Reward for truck departure wait time."""
        return self._compute_wait_reward(wait_minutes)

    def _compute_wait_reward(self, wait_minutes: float) -> float:
        """Sliding scale: fast_minutes -> max reward, slow_minutes -> min reward."""
        w = self.weights
        if wait_minutes <= w.truck_wait_fast_minutes:
            return w.truck_wait_reward_fast
        if wait_minutes >= w.truck_wait_slow_minutes:
            return w.truck_wait_reward_min
        span = max(_EPSILON, w.truck_wait_slow_minutes - w.truck_wait_fast_minutes)
        alpha = (wait_minutes - w.truck_wait_fast_minutes) / span
        return (1.0 - alpha) * w.truck_wait_reward_fast + alpha * w.truck_wait_reward_min

    def waiting_penalty(self, num_trucks_present: int, minutes_advanced: float) -> float:
        """Penalty for trucks waiting during idle time."""
        rate = self.weights.waiting_penalty_per_truck_minute
        if rate <= 0.0:
            return 0.0
        return -rate * num_trucks_present * minutes_advanced

    # ==================== Train Departures ====================

    def on_train_departure(self, train: Train) -> float:
        """Reward/penalty for train departure (once per train)."""
        if train.train_id in self._departed_trains:
            return 0.0
        self._departed_trains.add(train.train_id)
        leftover_count = len(train.get_all_pickup_container_ids())
        penalty = self.weights.penalty_train_leftover_per_id * leftover_count
        return self.weights.reward_train_departed + penalty

    # ==================== End-of-Day ====================

    def end_of_day_penalty(self, now: datetime) -> float:
        """Penalty for missed containers and inversions at end of day.

        Only containers whose departure_date is on or before today are
        considered "missed" — they should have left but didn't.
        Containers waiting for future pickup are legitimate carryover
        and receive no penalty.
        """
        w = self.weights
        today = now.date()
        missed = self._count_missed(today)
        inversions = self._count_inversions()
        return w.endday_leftover_weight * missed + w.endday_blocking_weight * inversions

    def _count_missed(self, today) -> int:
        """Count containers whose departure_date is on or before today."""
        missed = 0
        for rec in self.yard.iter_records():
            dep = getattr(rec.container, "departure_date", None)
            if dep is not None and dep.date() <= today:
                missed += 1
        return missed

    def _count_inversions(self) -> int:
        """Count stacking inversions (earlier departure below later departure)."""
        by_slot: dict = {}
        for rec in self.yard.iter_records():
            key = (rec.placement.row, rec.placement.bay)
            by_slot.setdefault(key, []).append(
                (rec.placement.tier, rec.container)
            )

        inversions = 0
        for stack in by_slot.values():
            stack.sort(key=lambda x: x[0])  # sort by tier, bottom-up
            dates = []
            for _tier, container in stack:
                dep = getattr(container, "departure_date", None)
                if dep is None:
                    break
                dates.append(dep)
            for i in range(1, len(dates)):
                if dates[i - 1] > dates[i]:
                    inversions += 1
        return inversions

    def compute_stacking_quality(self) -> float:
        """Quality metric: 1.0 = perfect, 0.0 = worst."""
        inversions = self._count_inversions()
        n = self.yard.container_count
        max_inv = max(1, n * (n - 1) // 2)
        return 1.0 - min(1.0, inversions / max_inv)

    # ==================== Reset ====================

    def reset_train_tracking(self) -> None:
        """Reset departed train tracking (call at start of new day)."""
        self._departed_trains.clear()