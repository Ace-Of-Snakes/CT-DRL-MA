import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

class TemporalStateEnhancement:
    """
    Provides temporal awareness features for the DRL agent.
    Encodes the current day's schedule into state features.
    """
    
    def __init__(self, time_per_day: float = 86400.0):
        self.time_per_day = time_per_day
        self.feature_dim = 20  # Number of temporal features
        
    def encode_daily_schedule(
        self,
        current_time: float,
        current_day: int,
        trains: Dict,  # Active trains with departure times
        trucks: Dict,  # Active trucks
        train_queue: Any,  # Train queue with scheduled arrivals
        truck_queue: Any,  # Truck queue with scheduled arrivals
        container_pickup_schedules: Dict[str, datetime],
        container_arrival_times: Dict[str, datetime]
    ) -> np.ndarray:
        """
        Encode the current day's schedule into feature vector.
        
        Returns:
            Feature vector of shape (feature_dim,)
        """
        features = np.zeros(self.feature_dim, dtype=np.float32)
        
        # Current time features
        time_in_day = current_time / self.time_per_day  # [0, 1]
        hours_remaining = (self.time_per_day - current_time) / 3600.0
        
        features[0] = time_in_day
        features[1] = hours_remaining / 24.0  # Normalize to [0, 1]
        
        # Time buckets (next 1h, 2h, 4h, 8h)
        time_buckets = [1, 2, 4, 8]
        current_datetime = datetime(2025, 1, 1) + timedelta(days=current_day, seconds=current_time)
        
        # Count scheduled arrivals in time buckets
        for i, hours in enumerate(time_buckets):
            bucket_end = current_datetime + timedelta(hours=hours)
            
            # Train arrivals
            train_arrivals = self._count_arrivals_in_window(
                train_queue.scheduled_arrivals, current_datetime, bucket_end
            )
            features[2 + i] = min(train_arrivals / 10.0, 1.0)  # Normalize
            
            # Truck arrivals
            truck_arrivals = self._count_arrivals_in_window(
                truck_queue.scheduled_arrivals, current_datetime, bucket_end
            )
            features[6 + i] = min(truck_arrivals / 20.0, 1.0)  # Normalize
        
        # Departure pressure (trains leaving soon)
        departures_1h = 0
        departures_2h = 0
        departures_4h = 0
        
        for train in trains.values():
            if hasattr(train, 'departure_time') and train.departure_time:
                time_until = (train.departure_time - current_datetime).total_seconds() / 3600.0
                if 0 <= time_until <= 1:
                    departures_1h += 1
                elif 1 < time_until <= 2:
                    departures_2h += 1
                elif 2 < time_until <= 4:
                    departures_4h += 1
        
        features[10] = min(departures_1h / 3.0, 1.0)
        features[11] = min(departures_2h / 3.0, 1.0)
        features[12] = min(departures_4h / 5.0, 1.0)
        
        # Container pickup pressure
        pickups_due_1h = 0
        pickups_due_4h = 0
        pickups_overdue = 0
        
        for container_id, pickup_time in container_pickup_schedules.items():
            time_until = (pickup_time - current_datetime).total_seconds() / 3600.0
            if time_until < 0:
                pickups_overdue += 1
            elif 0 <= time_until <= 1:
                pickups_due_1h += 1
            elif 1 < time_until <= 4:
                pickups_due_4h += 1
        
        features[13] = min(pickups_due_1h / 10.0, 1.0)
        features[14] = min(pickups_due_4h / 20.0, 1.0)
        features[15] = min(pickups_overdue / 5.0, 1.0)  # Overdue is bad
        
        # Utilization features
        features[16] = len(trains) / 10.0  # Active trains normalized
        features[17] = len(trucks) / 30.0  # Active trucks normalized
        
        # Schedule density (how busy is the upcoming period)
        total_events_4h = train_arrivals + truck_arrivals + departures_1h + departures_2h
        features[18] = min(total_events_4h / 20.0, 1.0)
        
        # Critical period indicator (rush hours, etc.)
        hour_of_day = (current_time % self.time_per_day) / 3600.0
        is_rush_hour = (6 <= hour_of_day <= 10) or (14 <= hour_of_day <= 18)
        features[19] = 1.0 if is_rush_hour else 0.0
        
        return features
    
    def _count_arrivals_in_window(
        self, 
        scheduled_arrivals: List[Tuple[datetime, Any]], 
        start: datetime, 
        end: datetime
    ) -> int:
        """Count arrivals within time window."""
        count = 0
        for arrival_time, _ in scheduled_arrivals:
            if start <= arrival_time < end:
                count += 1
        return count


class MoveEvaluator:
    """
    Evaluates the urgency and value of moves based on terminal state.
    """
    
    def __init__(self):
        self.urgency_weights = {
            'train_departure': 10.0,
            'truck_waiting': 5.0,
            'pickup_due': 8.0,
            'yard_optimization': 2.0
        }
    
    def evaluate_move_urgency(
        self,
        move: Dict[str, Any],
        current_datetime: datetime,
        trains: Dict,
        trucks: Dict,
        container_pickup_schedules: Dict[str, datetime],
        logistics: Any
    ) -> Tuple[float, float]:
        """
        Evaluate move urgency and value.
        
        Returns:
            (urgency_score, value_score)
        """
        urgency = 0.0
        value = 0.0
        
        container_id = move.get('container_id')
        move_type = move.get('move_type')
        
        # 1. Train departure urgency
        if move['dest_type'] == 'train':
            railtrack_id, wagon_idx = move['dest_pos']
            if railtrack_id in trains:
                train = trains[railtrack_id]
                if hasattr(train, 'departure_time') and train.departure_time:
                    hours_until = (train.departure_time - current_datetime).total_seconds() / 3600.0
                    if hours_until < 1:
                        urgency += self.urgency_weights['train_departure'] * 2.0
                        value += 20.0
                    elif hours_until < 2:
                        urgency += self.urgency_weights['train_departure']
                        value += 15.0
                    elif hours_until < 4:
                        urgency += self.urgency_weights['train_departure'] * 0.5
                        value += 10.0
        
        # 2. Truck pickup urgency
        if move['source_type'] == 'yard' and move['dest_type'] == 'truck':
            truck_pos = move['dest_pos']
            if truck_pos in trucks:
                truck = trucks[truck_pos]
                # Truck is waiting for this container
                if hasattr(truck, 'pickup_container_ids') and container_id in truck.pickup_container_ids:
                    urgency += self.urgency_weights['truck_waiting']
                    value += 15.0
        
        # 3. Container pickup schedule
        if container_id in container_pickup_schedules:
            pickup_time = container_pickup_schedules[container_id]
            hours_until_pickup = (pickup_time - current_datetime).total_seconds() / 3600.0
            
            if hours_until_pickup < 0:  # Overdue
                urgency += self.urgency_weights['pickup_due'] * 2.0
                value += 5.0  # Lower value because it's late
            elif hours_until_pickup < 2:
                urgency += self.urgency_weights['pickup_due']
                value += 15.0
            elif hours_until_pickup < 4:
                urgency += self.urgency_weights['pickup_due'] * 0.5
                value += 10.0
        
        # 4. Vehicle-to-vehicle transfers (high value)
        if move_type == 'train_to_truck' or move_type == 'truck_to_train':
            urgency += 5.0
            value += 20.0
        
        # 5. Yard optimization moves (low urgency but still valuable)
        if move_type == 'yard_to_yard':
            # Check if this move improves yard organization
            if self._is_optimization_move(move, container_pickup_schedules, current_datetime):
                urgency += self.urgency_weights['yard_optimization']
                value += 5.0
            else:
                value += 1.0  # Minimal value for reshuffling
        
        return urgency, value
    
    def _is_optimization_move(
        self, 
        move: Dict, 
        pickup_schedules: Dict,
        current_datetime: datetime
    ) -> bool:
        """Check if a yard-to-yard move improves organization."""
        container_id = move.get('container_id')
        
        # Move is good if it positions container for upcoming pickup
        if container_id in pickup_schedules:
            hours_until = (pickup_schedules[container_id] - current_datetime).total_seconds() / 3600.0
            if 0 < hours_until < 8:  # Pickup within 8 hours
                # Check if destination is closer to pickup area
                dest_pos = move['dest_pos']
                if isinstance(dest_pos, tuple) and len(dest_pos) >= 2:
                    _, bay, tier, _ = dest_pos
                    # Lower tiers and edge bays are better for pickup
                    if tier <= 1 or bay <= 2 or bay >= 15:
                        return True
        
        return False


class ProgressBasedRewardCalculator:
    """
    Calculates rewards based on progress toward daily goals.
    """
    
    def __init__(self, distance_weight: float = 0.1):
        self.distance_weight = distance_weight
        self.move_evaluator = MoveEvaluator()
        
    def calculate_reward(
        self,
        move: Dict[str, Any],
        distance: float,
        time: float,
        current_datetime: datetime,
        trains: Dict,
        trucks: Dict,
        container_pickup_schedules: Dict[str, datetime],
        logistics: Any
    ) -> float:
        """
        Calculate reward based on move value and urgency.
        Distance is used as a minor factor, not a penalty.
        """
        # Get move urgency and value
        urgency, value = self.move_evaluator.evaluate_move_urgency(
            move, current_datetime, trains, trucks, 
            container_pickup_schedules, logistics
        )
        
        # Base reward is value weighted by urgency
        base_reward = value * (1.0 + urgency / 10.0)
        
        # Distance factor (not penalty) - prefer shorter moves
        # Normalize distance (assume max ~200m in terminal)
        normalized_distance = min(distance / 200.0, 1.0)
        distance_factor = 1.0 - (normalized_distance * self.distance_weight)
        
        # Time efficiency bonus
        time_bonus = 0.0
        if time < 60:  # Less than 1 minute
            time_bonus = 2.0
        elif time < 120:  # Less than 2 minutes
            time_bonus = 1.0
        
        # Final reward
        reward = base_reward * distance_factor + time_bonus
        
        # Ensure minimum positive reward for any completed move
        reward = max(reward, 0.5)
        
        return reward