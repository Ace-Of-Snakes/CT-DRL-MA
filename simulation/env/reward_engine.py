# simulation/env/reward_engine.py
from datetime import datetime
from typing import Set

from simulation.core.facilities.yard import BooleanStorageYard
from simulation.core.vehicles.train import Train
from simulation.core.enums import MoveType
from simulation.config.operations_config import RewardWeights


class RewardEngine:
    """
    Reward calculation engine for terminal operations.
    
    Computes rewards for:
    - Container moves (distance/time penalties, move-specific bonuses)
    - Train departures (completion bonus, leftover penalties)
    - Truck wait times (fast service rewards)
    - End-of-day penalties (inversions, leftovers)
    """
    
    def __init__(
        self,
        yard: BooleanStorageYard,
        weights: RewardWeights = None
    ):
        """
        Initialize reward engine.
        
        Args:
            yard: Reference to yard for state queries
            weights: Reward weights configuration (uses defaults if None)
        """
        self.yard = yard
        self.weights = weights or RewardWeights()
        self._departed_trains: Set[str] = set()
    
    def immediate_reward(
        self,
        move_type: str,
        distance_m: float,
        time_s: float
    ) -> float:
        """
        Calculate immediate reward for a container move.
        
        Args:
            move_type: Type of move (MoveType enum value as string)
            distance_m: Distance traveled in meters
            time_s: Time taken in seconds
            
        Returns:
            Total reward (base cost + move-specific bonus)
        """
        w = self.weights
        
        # Base cost (distance + time penalties)
        base = w.per_meter_cost * distance_m + w.per_second_cost * time_s
        
        # Move-specific bonuses
        bonus = 0.0
        
        if move_type == MoveType.YARD_TO_TERMINAL_TRUCK.value:
            bonus = w.reward_terminal_truck
        elif move_type == MoveType.YARD_TO_TRUCK.value:
            bonus = w.reward_yard_to_truck
        elif move_type == MoveType.TRAIN_TO_TRUCK.value:
            bonus = w.reward_train_to_truck
        elif move_type == MoveType.YARD_TO_TRAIN.value:
            bonus = w.reward_yard_to_train
        elif move_type == MoveType.TRAIN_TO_YARD.value:
            bonus = w.reward_train_to_yard
        elif move_type == MoveType.TRUCK_TO_YARD.value:
            bonus = w.reward_truck_to_yard
        elif move_type == MoveType.TRUCK_TO_TRAIN.value:
            bonus = w.reward_truck_to_train
        elif move_type == MoveType.YARD_TO_YARD.value:
            bonus = w.penalty_yard_to_yard
        
        return base + bonus
    
    def truck_first_service_reward(self, wait_minutes: float) -> float:
        """
        Reward shaping when a truck receives its first container.
        
        Uses sliding scale:
        - <= fast_minutes: max reward
        - >= slow_minutes: min reward
        - Between: linear interpolation
        
        Args:
            wait_minutes: Minutes truck waited since arrival
            
        Returns:
            Shaped reward for service speed
        """
        return self._compute_wait_reward(wait_minutes)
    
    def truck_wait_reward(self, wait_minutes: float) -> float:
        """
        Calculate reward for truck departure wait time.
        
        Args:
            wait_minutes: Total wait time from arrival to departure
            
        Returns:
            Shaped reward based on wait duration
        """
        return self._compute_wait_reward(wait_minutes)
    
    def _compute_wait_reward(self, wait_minutes: float) -> float:
        """
        Internal: Compute wait time reward using configured thresholds.
        
        Args:
            wait_minutes: Wait time in minutes
            
        Returns:
            Reward value between reward_min and reward_fast
        """
        w = self.weights
        
        if wait_minutes <= w.truck_wait_fast_minutes:
            return w.truck_wait_reward_fast
        
        if wait_minutes >= w.truck_wait_slow_minutes:
            return w.truck_wait_reward_min
        
        # Linear interpolation between fast and slow thresholds
        span = max(1e-6, w.truck_wait_slow_minutes - w.truck_wait_fast_minutes)
        alpha = (wait_minutes - w.truck_wait_fast_minutes) / span
        
        return (1.0 - alpha) * w.truck_wait_reward_fast + alpha * w.truck_wait_reward_min
    
    def waiting_penalty(
        self,
        num_trucks_present: int,
        minutes_advanced: float
    ) -> float:
        """
        Calculate penalty for trucks waiting during idle time.
        
        Args:
            num_trucks_present: Number of trucks in terminal
            minutes_advanced: Minutes of simulation time advanced
            
        Returns:
            Penalty value (non-positive)
        """
        if self.weights.waiting_penalty_per_truck_minute <= 0.0:
            return 0.0
        
        return -self.weights.waiting_penalty_per_truck_minute * num_trucks_present * minutes_advanced
    
    def on_train_departure(self, train: Train) -> float:
        """
        Calculate reward/penalty for train departure.
        
        Rewards:
        - Base departure bonus
        - Penalty for each container not loaded (leftover pickup IDs)
        
        Args:
            train: Departing train
            
        Returns:
            Total reward (bonus - leftover penalties)
        """
        if train.train_id in self._departed_trains:
            return 0.0
        
        self._departed_trains.add(train.train_id)
        
        leftover_ids = train.get_all_pickup_container_ids()
        penalty = self.weights.penalty_train_leftover_per_id * len(leftover_ids)
        
        return self.weights.reward_train_departed + penalty
    
    def end_of_day_penalty(self, now: datetime) -> float:
        """
        Calculate penalty for end-of-day yard state.
        
        Penalizes:
        - Containers left in yard (should have departed)
        - Inversions (containers stacked in wrong order)
        
        Args:
            now: Current simulation time
            
        Returns:
            Total penalty (non-positive)
        """
        # Leftover container penalty
        leftovers = len(self.yard.containers)
        penalty = self.weights.endday_leftover_weight * leftovers
        
        # Inversion penalty
        inversions = self._count_inversions()
        penalty += self.weights.endday_blocking_weight * inversions
        
        return penalty
    
    def _count_inversions(self) -> int:
        """
        Count inversions in yard stacks.
        
        An inversion occurs when a container with earlier departure
        is stacked below a container with later departure in the same slot.
        
        Returns:
            Total number of inversions
        """
        # Group containers by slot (row, bay)
        by_slot = {}
        for cid, rec in self.yard.containers.items():
            key = (rec.placement.row, rec.placement.bay)
            by_slot.setdefault(key, []).append((rec.placement.tier, cid))
        
        inversions = 0
        
        for (_row, _bay), containers in by_slot.items():
            # Sort by tier (bottom to top)
            containers.sort(key=lambda x: x[0])
            
            # Get departure dates
            departure_dates = []
            for _tier, cid in containers:
                container = self.yard.get_container(cid)
                if container and container.departure_date:
                    departure_dates.append(container.departure_date)
                else:
                    # Missing departure date - skip this slot
                    break
            
            # Count inversions (earlier departure below later departure)
            for i in range(1, len(departure_dates)):
                if departure_dates[i - 1] > departure_dates[i]:
                    inversions += 1
        
        return inversions
    
    def compute_stacking_quality(self) -> float:
        """
        Compute stacking quality metric (0-1 scale).
        
        Quality = 1 - (inversions / max_possible_inversions)
        
        Returns:
            Quality score between 0 (worst) and 1 (perfect)
        """
        inversions = self._count_inversions()
        
        # Maximum possible inversions
        total_slots = sum(len(containers) for containers in self._get_containers_by_slot().values())
        max_inversions = max(1, total_slots * (total_slots - 1) // 2)  # Upper bound
        
        return 1.0 - min(1.0, inversions / max_inversions)
    
    def _get_containers_by_slot(self) -> dict:
        """Get containers grouped by slot for quality calculations."""
        by_slot = {}
        for cid, rec in self.yard.containers.items():
            key = (rec.placement.row, rec.placement.bay)
            by_slot.setdefault(key, []).append((rec.placement.tier, cid))
        return by_slot
    
    def reset_train_tracking(self) -> None:
        """Reset departed train tracking (call at start of new day)."""
        self._departed_trains.clear()