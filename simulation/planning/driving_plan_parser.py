# simulation/planning/driving_plan_parser.py
"""Parser for converting driving plan JSON to Train objects."""
import json
from typing import Dict, List, Optional

from simulation.core.vehicles.train import Train
from simulation.planning.time_encoder import WeeklyTimeEncoder
from simulation.config.paths import DataPaths
from simulation.config.train_config import TrainDefaults
from simulation.core.constants import STANDARD_VEHICLE_LENGTH_M

# JSON field indices for arrival/departure arrays
_DAY = 0
_TIME = 1
_DAY_COMPLETE = 2
_TIME_COMPLETE = 3

# Known typo in driving plan JSON data
_TYPO_MAP: Dict[str, str] = {'tueday': 'tuesday'}


class DrivingPlanParser:
    """Parser for weekly driving plan JSON -> Train objects."""

    def __init__(self, json_path: Optional[str] = None):
        self.json_path = json_path or str(DataPaths.DRIVING_PLAN)
        self.encoder = WeeklyTimeEncoder()
        self.trains_data = self._load_json()

    def _load_json(self) -> Dict:
        """Load driving plan JSON data."""
        with open(self.json_path, 'r') as f:
            data = json.load(f)
        return data['driving_plan']['trains']

    def _parse_time(self, day: str, time_str: str) -> Dict[str, float]:
        """Parse day + 'HH:MM' string to encoded values."""
        hour, minute = map(int, time_str.split(':'))
        return self.encoder.encode(day, hour, minute)

    @staticmethod
    def _train_id(base_id: str, operator: str, instance: int = 0) -> str:
        """Generate unique train ID."""
        if instance == 0:
            return f"{base_id}_{operator}"
        return f"{base_id}_{operator}_{instance}"

    def _offset_day(self, day_name: str, offset: int) -> str:
        """Apply a day offset (mod 7) to a day name."""
        idx = self.encoder.DAY_MAP.get(day_name.lower())
        if idx is None:
            return day_name
        return self.encoder.DAY_NAMES[(idx + offset) % 7]

    # ================================================================
    # Train creation
    # ================================================================

    def create_trains(self) -> List[Train]:
        """Create Train objects from driving plan."""
        trains: List[Train] = []

        for train_id, train_info in self.trains_data.items():
            operator = train_info['operator_short']

            for _plan_key, schedule in train_info['plan'].items():
                trains.append(self._create_single_train(
                    train_id, operator,
                    schedule['arrival'], schedule['departure'],
                    instance=0,
                ))

                if 'mirrored_on' in schedule:
                    orig_day = schedule['arrival'][_DAY].lower()
                    for i, mirror_day in enumerate(schedule['mirrored_on'], 1):
                        mirror_day_clean = _TYPO_MAP.get(
                            mirror_day.lower(), mirror_day.lower()
                        )
                        mirror = self._create_mirrored_train(
                            train_id, operator,
                            schedule['arrival'], schedule['departure'],
                            orig_day, mirror_day_clean,
                            instance=i,
                        )
                        if mirror:
                            trains.append(mirror)

        return trains

    def _create_single_train(
        self,
        base_id: str,
        operator: str,
        arrival: List[str],
        departure: List[str],
        instance: int,
    ) -> Train:
        """Create a single Train with encoded schedule metadata."""
        tid = self._train_id(base_id, operator, instance)

        arrival_enc = self._parse_time(arrival[_DAY], arrival[_TIME])
        arrival_complete_enc = self._parse_time(arrival[_DAY_COMPLETE], arrival[_TIME_COMPLETE])
        departure_enc = self._parse_time(departure[_DAY], departure[_TIME])
        departure_complete_enc = self._parse_time(departure[_DAY_COMPLETE], departure[_TIME_COMPLETE])

        arrival_str = f"{arrival[_DAY]}-{arrival[_TIME].replace(':', '-')}"
        departure_str = f"{departure[_DAY]}-{departure[_TIME].replace(':', '-')}"
        stay_duration = self.encoder.subtract(arrival_str, departure_str)

        train = Train(
            train_id=tid,
            num_wagons=TrainDefaults.NUM_WAGONS,
            wagon_length=STANDARD_VEHICLE_LENGTH_M,
            arrival_time=None,
            departure_time=None,
            operator=operator,
        )

        train.schedule_encoded = {
            'arrival': arrival_enc,
            'arrival_complete': arrival_complete_enc,
            'departure': departure_enc,
            'departure_complete': departure_complete_enc,
            'operator': operator,
            'stay_duration': stay_duration,
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
        instance: int,
    ) -> Optional[Train]:
        """Create a mirrored train with day offset applied."""
        orig_idx = self.encoder.DAY_MAP.get(orig_day.lower())
        mirror_idx = self.encoder.DAY_MAP.get(mirror_day)
        if orig_idx is None or mirror_idx is None:
            return None

        offset = mirror_idx - orig_idx

        mirror_arrival = [
            self._offset_day(arrival[_DAY], offset),
            arrival[_TIME],
            self._offset_day(arrival[_DAY_COMPLETE], offset),
            arrival[_TIME_COMPLETE],
        ]
        mirror_departure = [
            self._offset_day(departure[_DAY], offset),
            departure[_TIME],
            self._offset_day(departure[_DAY_COMPLETE], offset),
            departure[_TIME_COMPLETE],
        ]

        return self._create_single_train(
            base_id, operator, mirror_arrival, mirror_departure, instance
        )