# simulation/terminal_components/systems/RewardEngine.py
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
from datetime import datetime
from simulation.core.facilities.yard import BooleanStorageYard
from simulation.operations.terminal_manager import (
    YARD_TO_YARD, TRAIN_TO_YARD, YARD_TO_TRAIN, TRUCK_TO_YARD, YARD_TO_TRUCK,
    TRAIN_TO_TRUCK, TRUCK_TO_TRAIN, YARD_TO_TERMINAL_TRUCK
)

@dataclass
class RewardWeights:
    per_meter_cost: float = -0.001
    per_second_cost: float = -0.0005
    reward_train_departed: float = 5.0
    penalty_train_leftover_per_id: float = -2.0
    reward_tt: float = 0.5
    reward_yard_to_truck: float = 1.0
    reward_train_to_truck: float = 1.0
    reward_yard_to_train: float = 1.5
    reward_train_to_yard: float = 1.0
    reward_truck_to_yard: float = 0.8
    reward_truck_to_train: float = 1.2
    penalty_yard_to_yard: float = -0.05
    endday_blocking_weight: float = -0.2
    endday_leftover_weight: float = -0.1
    waiting_penalty_per_truck_minute: float = 0.0  # set >0 for real penalty

    # New: truck wait-time shaping
    truck_wait_fast_minutes: float = 60.0     # <= 1 hour => max reward
    truck_wait_slow_minutes: float = 180.0    # >= 3 hours => min reward
    truck_wait_reward_fast: float = 1.0       # reward at <= 1h
    truck_wait_reward_min: float = 0.05       # reward floor at >= 3h

class RewardEngine:
    def __init__(self, yard: BooleanStorageYard, weights: RewardWeights = RewardWeights()):
        self.yard = yard
        self.w = weights
        self._departed_trains: set[str] = set()

    def immediate_reward(self, move_type: str, distance_m: float, time_s: float) -> float:
        base = self.w.per_meter_cost * distance_m + self.w.per_second_cost * time_s
        bonus = 0.0
        if move_type == YARD_TO_TERMINAL_TRUCK:
            bonus = self.w.reward_tt
        elif move_type == YARD_TO_TRUCK:
            bonus = self.w.reward_yard_to_truck
        elif move_type == TRAIN_TO_TRUCK:
            bonus = self.w.reward_train_to_truck
        elif move_type == YARD_TO_TRAIN:
            bonus = self.w.reward_yard_to_train
        elif move_type == TRAIN_TO_YARD:
            bonus = self.w.reward_train_to_yard
        elif move_type == TRUCK_TO_YARD:
            bonus = self.w.reward_truck_to_yard
        elif move_type == TRUCK_TO_TRAIN:
            bonus = self.w.reward_truck_to_train
        elif move_type == YARD_TO_YARD:
            bonus = self.w.penalty_yard_to_yard
        return base + bonus

    def truck_first_service_reward(self, wait_minutes: float) -> float:
        """
        Reward shaping at the moment a truck receives its first container.
        Uses the same curve as truck_wait_reward (high reward for fast service, minimum after slow).
        """
        return self.truck_wait_reward(wait_minutes)


    def waiting_penalty(self, num_trucks_present: int, minutes_advanced: float) -> float:
        if self.w.waiting_penalty_per_truck_minute <= 0.0:
            return 0.0
        return -self.w.waiting_penalty_per_truck_minute * num_trucks_present * minutes_advanced

    def truck_wait_reward(self, wait_minutes: float) -> float:
        # High reward at <= 60 min; linear down to 180 min; minimal afterward
        w = self.w
        if wait_minutes <= w.truck_wait_fast_minutes:
            return w.truck_wait_reward_fast
        if wait_minutes >= w.truck_wait_slow_minutes:
            return w.truck_wait_reward_min
        span = max(1e-6, (w.truck_wait_slow_minutes - w.truck_wait_fast_minutes))
        alpha = (wait_minutes - w.truck_wait_fast_minutes) / span
        return (1.0 - alpha) * w.truck_wait_reward_fast + alpha * w.truck_wait_reward_min

    def on_train_departure(self, train) -> float:
        if train.train_id in self._departed_trains:
            return 0.0
        self._departed_trains.add(train.train_id)
        leftover_ids = train.get_all_pickup_container_ids()
        penalty = self.w.penalty_train_leftover_per_id * len(leftover_ids)
        return self.w.reward_train_departed + penalty

    def end_of_day_penalty(self, now: datetime) -> float:
        leftovers = len(self.yard.containers)
        pen = self.w.endday_leftover_weight * leftovers
        by_slot = {}
        for cid, rec in self.yard.containers.items():
            key = (rec.placement.row, rec.placement.bay)
            by_slot.setdefault(key, []).append((rec.placement.tier, cid))
        inversions = 0
        for (_r,_b), lst in by_slot.items():
            lst.sort(key=lambda x: x[0])
            dep = []
            for _tier, cid in lst:
                c = self.yard.get_container(cid)
                if not c:
                    continue
                d = c.departure_date
                dep.append(d)
            for i in range(1, len(dep)):
                if dep[i-1] > dep[i]:
                    inversions += 1
        pen += self.w.endday_blocking_weight * inversions
        return pen