"""Configuration for terminal operations."""
from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class RewardWeights:
    """Reward weights for the reward engine."""
    
    # Distance/time costs
    per_meter_cost: float = -0.001
    per_second_cost: float = -0.0005
    
    # Move rewards
    reward_train_departed: float = 5.0
    penalty_train_leftover_per_id: float = -2.0
    reward_terminal_truck: float = 0.5
    reward_yard_to_truck: float = 1.0
    reward_train_to_truck: float = 1.0
    reward_yard_to_train: float = 1.5
    reward_train_to_yard: float = 1.0
    reward_truck_to_yard: float = 0.8
    reward_truck_to_train: float = 1.2
    penalty_yard_to_yard: float = -0.05
    
    # End of day penalties
    endday_blocking_weight: float = -0.2
    endday_leftover_weight: float = -0.1
    
    # Truck wait time shaping
    waiting_penalty_per_truck_minute: float = 0.0
    truck_wait_fast_minutes: float = 60.0
    truck_wait_slow_minutes: float = 180.0
    truck_wait_reward_fast: float = 1.0
    truck_wait_reward_min: float = 0.05


class OperationsDefaults:
    """Default operational parameters."""
    
    PROXIMITY_SEARCH_BAYS: Final[int] = 3
    RECALC_WINDOW_MINUTES: Final[int] = 30
    
    # Logistics
    EXPORT_PER_IMPORT_RATIO: Final[float] = 0.75
    TRAIN_OVERGENERATION_FACTOR: Final[float] = 3.0
    DAILY_TRAIN_IMPORT_CAP: Final[int] = 220


class GateConfig:
    """Terminal gate configuration."""
    
    # Dwell time classification
    SHORT_DWELL_THRESHOLD_DAYS: Final[float] = 3.0
    DWELL_COMPARISON_PERCENTILE: Final[int] = 25
    
    # Early arrival for short-dwell operators
    SHORT_DWELL_EARLY_ARRIVAL_HOURS: Final[int] = 12
    
    # Truck arrival windows
    TRUCK_ARRIVAL_HOUR_START: Final[int] = 6
    TRUCK_ARRIVAL_HOUR_END: Final[int] = 22
    
    # Processing
    MIN_TRUCK_LOAD_FACTOR: Final[float] = 0.5
    CONTAINER_BATCH_SIZE: Final[int] = 1000
    MAX_WORKERS: Final[int] = 4