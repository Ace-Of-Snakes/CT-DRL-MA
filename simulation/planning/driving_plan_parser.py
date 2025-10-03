# simulation/planning/driving_plan_parser.py
import json
import os
from typing import Dict, List, Optional

from simulation.core.vehicles.train import Train
from simulation.planning.time_encoder import WeeklyTimeEncoder
from simulation.config.paths import DataPaths
from simulation.config.train_config import TrainDefaults


# JSON field indices
ARRIVAL_IDX_DAY = 0
ARRIVAL_IDX_TIME = 1
ARRIVAL_IDX_DAY_COMPLETE = 2
ARRIVAL_IDX_TIME_COMPLETE = 3
DEPARTURE_IDX_DAY = 0
DEPARTURE_IDX_TIME = 1
DEPARTURE_IDX_DAY_COMPLETE = 2
DEPARTURE_IDX_TIME_COMPLETE = 3


class DrivingPlanParser:
    """
    Parser for converting driving plan JSON to Train objects.
    
    Loads weekly train schedules from JSON and creates Train objects
    with encoded temporal information for scheduling.
    """
    
    def __init__(self, json_path: Optional[str] = None):
        """
        Initialize parser with JSON file path.
        
        Args:
            json_path: Path to driving plan JSON file (uses default if None)
        """
        self.json_path = json_path or str(DataPaths.DRIVING_PLAN)
        self.encoder = WeeklyTimeEncoder()
        self.trains_data = self._load_json()
    
    def _load_json(self) -> Dict:
        """
        Load and return the driving plan JSON data.
        
        Returns:
            Dictionary of train data from JSON
            
        Raises:
            FileNotFoundError: If JSON file doesn't exist
            json.JSONDecodeError: If JSON is malformed
        """
        with open(self.json_path, 'r') as f:
            data = json.load(f)
        return data['driving_plan']['trains']
    
    def _parse_time(self, day: str, time_str: str) -> Dict[str, float]:
        """
        Parse day and time string to encoded values.
        
        Args:
            day: Day name (e.g., "Monday")
            time_str: Time string (e.g., "14:30")
            
        Returns:
            Dictionary with angle, sin, cos, seconds
        """
        hour, minute = map(int, time_str.split(':'))
        return self.encoder.encode(day, hour, minute)
    
    def _generate_train_id(self, base_id: str, operator: str, instance: int = 0) -> str:
        """
        Generate unique train ID.
        
        Args:
            base_id: Base train identifier
            operator: Operator code
            instance: Instance number for mirrored trains
            
        Returns:
            Unique train ID string
        """
        if instance == 0:
            return f"{base_id}_{operator}"
        return f"{base_id}_{operator}_{instance}"
    
    def create_trains(self) -> List[Train]:
        """
        Create Train objects from driving plan.
        
        Returns:
            List of Train objects with encoded schedules
        """
        trains = []
        
        for train_id, train_info in self.trains_data.items():
            operator = train_info['operator_short']
            
            for plan_key, schedule in train_info['plan'].items():
                # Base train
                train = self._create_single_train(
                    train_id,
                    operator,
                    schedule['arrival'],
                    schedule['departure'],
                    0
                )
                trains.append(train)
                
                # Mirrored trains
                if 'mirrored_on' in schedule:
                    for i, mirror_day in enumerate(schedule['mirrored_on'], 1):
                        # Calculate day offset from original
                        orig_day = schedule['arrival'][ARRIVAL_IDX_DAY].lower()
                        mirror_day_clean = mirror_day.lower().replace('tueday', 'tuesday')  # Fix typo
                        
                        # Create mirrored train with offset
                        mirror_train = self._create_mirrored_train(
                            train_id,
                            operator,
                            schedule['arrival'],
                            schedule['departure'],
                            orig_day,
                            mirror_day_clean,
                            i
                        )
                        if mirror_train:
                            trains.append(mirror_train)
        
        return trains
    
    def _create_single_train(
        self,
        base_id: str,
        operator: str,
        arrival: List[str],
        departure: List[str],
        instance: int
    ) -> Train:
        """
        Create a single Train object.
        
        Args:
            base_id: Base train identifier
            operator: Operator code
            arrival: Arrival schedule [day, time, day_complete, time_complete]
            departure: Departure schedule [day, time, day_complete, time_complete]
            instance: Instance number
            
        Returns:
            Train object with encoded schedule
        """
        train_id = self._generate_train_id(base_id, operator, instance)
        
        # Parse arrival and departure times
        arrival_enc = self._parse_time(arrival[ARRIVAL_IDX_DAY], arrival[ARRIVAL_IDX_TIME])
        arrival_complete_enc = self._parse_time(
            arrival[ARRIVAL_IDX_DAY_COMPLETE],
            arrival[ARRIVAL_IDX_TIME_COMPLETE]
        )
        
        departure_enc = self._parse_time(departure[DEPARTURE_IDX_DAY], departure[DEPARTURE_IDX_TIME])
        departure_complete_enc = self._parse_time(
            departure[DEPARTURE_IDX_DAY_COMPLETE],
            departure[DEPARTURE_IDX_TIME_COMPLETE]
        )
        
        # Calculate length of stay
        arrival_str = f"{arrival[ARRIVAL_IDX_DAY]}-{arrival[ARRIVAL_IDX_TIME].replace(':', '-')}"
        departure_str = f"{departure[DEPARTURE_IDX_DAY]}-{departure[DEPARTURE_IDX_TIME].replace(':', '-')}"
        stay_duration = self.encoder.subtract(arrival_str, departure_str)
        
        # Create Train with encoded times stored as metadata
        train = Train(
            train_id=train_id,
            num_wagons=TrainDefaults.NUM_WAGONS,
            wagon_length=TrainDefaults.WAGON_LENGTH_M,
            arrival_time=None,  # Will be set during simulation
            departure_time=None,  # Will be set during simulation
            operator=operator
        )
        
        # Store encoded schedule as metadata
        train.schedule_encoded = {
            'arrival': arrival_enc,
            'arrival_complete': arrival_complete_enc,
            'departure': departure_enc,
            'departure_complete': departure_complete_enc,
            'operator': operator,
            'stay_duration': stay_duration
        }
        
        return train
    
    def _create_mirrored_train(
        self,
        base_id: str,
        operator: str,
        arrival: List[str],
        departure: List[str],
        orig_day: str,
        mirror_day: str,
        instance: int
    ) -> Optional[Train]:
        """
        Create a mirrored train with day offset.
        
        Args:
            base_id: Base train identifier
            operator: Operator code
            arrival: Original arrival schedule
            departure: Original departure schedule
            orig_day: Original day of week
            mirror_day: Mirror day of week
            instance: Instance number
            
        Returns:
            Train object or None if invalid days
        """
        # Calculate day offset
        orig_day_idx = self.encoder.DAY_MAP.get(orig_day.lower())
        mirror_day_idx = self.encoder.DAY_MAP.get(mirror_day)
        
        if orig_day_idx is None or mirror_day_idx is None:
            return None  # Skip invalid days
        
        day_offset = mirror_day_idx - orig_day_idx
        
        # Apply offset to arrival and departure days
        arrival_day_idx = (
            self.encoder.DAY_MAP[arrival[ARRIVAL_IDX_DAY].lower()] + day_offset
        ) % 7
        arrival_complete_day_idx = (
            self.encoder.DAY_MAP[arrival[ARRIVAL_IDX_DAY_COMPLETE].lower()] + day_offset
        ) % 7
        departure_day_idx = (
            self.encoder.DAY_MAP[departure[DEPARTURE_IDX_DAY].lower()] + day_offset
        ) % 7
        departure_complete_day_idx = (
            self.encoder.DAY_MAP[departure[DEPARTURE_IDX_DAY_COMPLETE].lower()] + day_offset
        ) % 7
        
        # Create modified schedule
        mirror_arrival = [
            self.encoder.DAY_NAMES[arrival_day_idx],
            arrival[ARRIVAL_IDX_TIME],
            self.encoder.DAY_NAMES[arrival_complete_day_idx],
            arrival[ARRIVAL_IDX_TIME_COMPLETE]
        ]
        
        mirror_departure = [
            self.encoder.DAY_NAMES[departure_day_idx],
            departure[DEPARTURE_IDX_TIME],
            self.encoder.DAY_NAMES[departure_complete_day_idx],
            departure[DEPARTURE_IDX_TIME_COMPLETE]
        ]
        
        return self._create_single_train(base_id, operator, mirror_arrival, mirror_departure, instance)


# Example usage
if __name__ == "__main__":
    parser = DrivingPlanParser()
    trains = parser.create_trains()
    
    print(f"Created {len(trains)} trains")
    for train in trains[:5]:  # Show first 5
        if hasattr(train, 'schedule_encoded'):
            print(f"Train {train.train_id}: {train.schedule_encoded['operator']}")
            print(f"  Arrival angle: {train.schedule_encoded['arrival']['angle']:.4f}")
            print(f"  Departure angle: {train.schedule_encoded['departure']['angle']:.4f}")
            print(f"  Stay duration: {train.schedule_encoded['stay_duration']['hours']:.2f} hours")