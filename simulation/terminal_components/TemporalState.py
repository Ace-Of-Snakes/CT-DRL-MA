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
        trains: Dict,  # Active trains (now properly extracted from new tracking)
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
        
        # Count departures from all trains (properly handling new structure)
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
        # Need to recount since we might have different train counts
        total_trains = len(trains)
        total_trucks = len(trucks)
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
    Updated to work with new BooleanLogistics train tracking.
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
        trains: Dict,  # Now contains train_id -> train mapping
        trucks: Dict,
        container_pickup_schedules: Dict[str, datetime],
        logistics: Any
    ) -> Tuple[float, float]:
        """
        Evaluate move urgency and value with departure awareness.
        Updated to handle new train tracking system.
        
        Returns:
            (urgency_score, value_score)
        """
        urgency = 0.0
        value = 0.0
        
        container_id = move.get('container_id')
        move_type = move.get('move_type')
        source_type = move.get('source_type')
        dest_type = move.get('dest_type')
        
        # CORE PRINCIPLE: Any move involving vehicles gets high base value
        if source_type != 'yard' or dest_type != 'yard':
            # Vehicle involved - high base value
            value = 30.0
            urgency = 5.0
            
            # 1. Vehicle-to-vehicle transfers (HIGHEST PRIORITY)
            if source_type != 'yard' and dest_type != 'yard':
                value = 40.0
                urgency = 10.0
                
                # Check departure urgency for both vehicles
                if source_type == 'train' and dest_type == 'train':
                    # NEW: Handle new position format (train_id, wagon_idx)
                    src_train_id, _ = move['source_pos']
                    dst_train_id, _ = move['dest_pos']
                    
                    # Check source train departure
                    if src_train_id in trains:
                        src_train = trains[src_train_id]
                        if hasattr(src_train, 'departure_time') and src_train.departure_time:
                            hours_until = (src_train.departure_time - current_datetime).total_seconds() / 3600.0
                            if hours_until < 1:
                                urgency += 10.0
                                value += 10.0
                    
                    # Check destination train departure
                    if dst_train_id in trains:
                        dst_train = trains[dst_train_id]
                        if hasattr(dst_train, 'departure_time') and dst_train.departure_time:
                            hours_until = (dst_train.departure_time - current_datetime).total_seconds() / 3600.0
                            if hours_until < 1:
                                urgency += 10.0
                                value += 10.0
                                
            # 2. From yard to vehicle (EXPORTS - HIGH PRIORITY)
            elif source_type == 'yard' and dest_type in ['train', 'truck']:
                value = 35.0
                urgency = 8.0
                
                if dest_type == 'train':
                    # NEW: Handle new position format
                    train_id, wagon_idx = move['dest_pos']
                    if train_id in trains:
                        train = trains[train_id]
                        
                        # Check for imminent departure
                        if hasattr(train, 'departure_time') and train.departure_time:
                            hours_until = (train.departure_time - current_datetime).total_seconds() / 3600.0
                            
                            # CRITICAL URGENCY for trains about to leave
                            if hours_until < 0.5:  # 30 minutes
                                urgency = 20.0
                                value = 50.0
                            elif hours_until < 1:  # 1 hour
                                urgency = 15.0
                                value = 45.0
                            elif hours_until < 2:  # 2 hours
                                urgency = 10.0
                                value = 40.0
                        
                        # Extra bonus for fulfilling pickup request
                        for wagon in train.wagons:
                            if container_id in wagon.pickup_container_ids:
                                value += 15.0  # Big bonus for fulfilling request
                                urgency += 5.0
                                
                                # Even more urgent if train leaving soon
                                if hasattr(train, 'departure_time') and train.departure_time:
                                    hours_until = (train.departure_time - current_datetime).total_seconds() / 3600.0
                                    if hours_until < 1:
                                        urgency += 5.0  # Total urgency could be 25+
                                        value += 5.0
                                break
                                
                elif dest_type == 'truck':
                    truck_pos = move['dest_pos']
                    if truck_pos in trucks:
                        truck = trucks[truck_pos]
                        
                        # Check truck time in terminal
                        arrival_time = getattr(truck, 'arrival_time', current_datetime)
                        hours_in_terminal = (current_datetime - arrival_time).total_seconds() / 3600.0
                        
                        # Urgency based on time in terminal (trucks forced out after 8 hours)
                        if hours_in_terminal > 7:  # About to be forced out
                            urgency = 15.0
                            value = 40.0
                        elif hours_in_terminal > 6:
                            urgency = 10.0
                            value = 35.0
                        
                        # Bonus for fulfilling pickup request
                        if hasattr(truck, 'pickup_container_ids') and container_id in truck.pickup_container_ids:
                            value += 10.0  # Fulfilling request bonus
                            urgency += 3.0
                            
                            # More urgent if truck been waiting long
                            if hours_in_terminal > 4:
                                urgency += 2.0
                                
            # 3. From vehicle to yard (IMPORTS - MEDIUM-HIGH PRIORITY)
            elif source_type in ['train', 'truck'] and dest_type == 'yard':
                value = 25.0
                urgency = 6.0
                
                # Urgent if vehicle needs to leave soon
                if source_type == 'train':
                    # NEW: Handle new position format
                    train_id, _ = move['source_pos']
                    if train_id in trains:
                        train = trains[train_id]
                        if hasattr(train, 'departure_time') and train.departure_time:
                            hours_until = (train.departure_time - current_datetime).total_seconds() / 3600.0
                            
                            if hours_until < 0.5:  # 30 minutes
                                urgency = 18.0  # Very urgent to unload
                                value = 40.0
                            elif hours_until < 1:
                                urgency = 12.0
                                value = 35.0
                            elif hours_until < 2:
                                urgency = 8.0
                                value = 30.0
                                
                elif source_type == 'truck':
                    truck_pos = move['source_pos']
                    if truck_pos in trucks:
                        truck = trucks[truck_pos]
                        arrival_time = getattr(truck, 'arrival_time', current_datetime)
                        hours_in_terminal = (current_datetime - arrival_time).total_seconds() / 3600.0
                        
                        if hours_in_terminal > 7:  # About to be forced out
                            urgency = 12.0
                            value = 35.0
                        elif hours_in_terminal > 6:
                            urgency = 8.0
                            value = 30.0
                            
        # 4. Yard to stack (MEDIUM PRIORITY - clearing space)
        elif move_type == 'yard_to_stack':
            value = 15.0
            urgency = 3.0
            
            # Higher priority if it's a swap body/trailer that's been sitting
            if container_id and hasattr(logistics, 'yard_container_index'):
                # Try to find container info
                container_pos = logistics.yard_container_index.get(container_id)
                if container_pos:
                    row, bay, tier, split = container_pos
                    try:
                        container = logistics.yard.get_container_at(row, bay, tier, split)
                        if container and container.container_type in ['Swap Body', 'Trailer']:
                            # Check how long it's been in yard
                            if container.arrival_date:
                                days_in_yard = (current_datetime - container.arrival_date).days
                                if days_in_yard > 5:
                                    urgency = 5.0
                                    value = 20.0
                                elif days_in_yard > 3:
                                    urgency = 4.0
                                    value = 17.0
                    except:
                        pass
                        
        # 5. Yard to yard (LOWEST PRIORITY - reshuffling)
        else:  # yard_to_yard
            value = 2.0
            urgency = 1.0
            
            # Check if it's an optimization move
            if self._is_optimization_move(move, container_pickup_schedules, current_datetime):
                value += 3.0
                urgency += 1.0
                
            # Check if moving to better position for upcoming pickup
            if container_id in container_pickup_schedules:
                pickup_time = container_pickup_schedules[container_id]
                hours_until_pickup = (pickup_time - current_datetime).total_seconds() / 3600.0
                
                if 0 < hours_until_pickup < 4:
                    # Move is preparing for upcoming pickup
                    dest_pos = move['dest_pos']
                    if isinstance(dest_pos, tuple) and len(dest_pos) >= 2:
                        # Handle both yard position formats
                        if len(dest_pos) == 4:  # (row, bay, tier, split)
                            _, bay, tier, _ = dest_pos
                        elif len(dest_pos) == 3:  # (row, bay, tier)
                            _, bay, tier = dest_pos
                        else:
                            bay, tier = 10, 2  # Default middle values
                            
                        # Lower tiers are better
                        if tier == 0:
                            value += 2.0
                            urgency += 1.0
                        # Edge bays are better for access
                        if bay <= 2 or bay >= 17:
                            value += 1.0
        
        # Additional urgency based on container pickup schedule
        if container_id and container_id in container_pickup_schedules:
            pickup_time = container_pickup_schedules[container_id]
            hours_until_pickup = (pickup_time - current_datetime).total_seconds() / 3600.0
            
            if hours_until_pickup < 0:  # Overdue
                urgency += 5.0
                value += 2.0
            elif hours_until_pickup < 1:  # Very urgent
                urgency += 4.0
                value += 2.0
            elif hours_until_pickup < 2:
                urgency += 3.0
                value += 1.0
            elif hours_until_pickup < 4:
                urgency += 1.0
        
        # Cap values to prevent overflow
        urgency = min(urgency, 30.0)
        value = min(value, 60.0)
        
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
                    # Handle both formats
                    if len(dest_pos) >= 4:
                        _, bay, tier, _ = dest_pos
                    else:
                        _, bay, tier = dest_pos[:3]
                    # Lower tiers and edge bays are better for pickup
                    if tier <= 1 or bay <= 2 or bay >= 15:
                        return True
        
        return False


class ProgressBasedRewardCalculator:
    """
    Calculates rewards based on progress toward daily goals.
    No changes needed - works with the refactored system.
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
        
        # Special handling for yard_to_stack moves
        if move.get('move_type') == 'yard_to_stack':
            # Small reward for clearing swap bodies/trailers
            return 3.0 * distance_factor + time_bonus

        # Ensure minimum positive reward for any completed move
        reward = max(reward, 0.5)
        
        return reward