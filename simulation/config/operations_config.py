"""Configuration for terminal operations."""
from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class RewardWeights:
    """Reward weights for the reward engine.
    
    Export rewards are HIGH to encourage agent to prioritize exports.
    Restack penalty is negative to discourage unnecessary moves.
    """
    
    # Distance/time costs (small penalty for inefficient moves)
    per_meter_cost: float = -0.001
    per_second_cost: float = -0.0005
    
    # EXPORT rewards (agent must learn these are HIGH PRIORITY)
    reward_yard_to_train: float = 5.0    # Trains have deadlines!
    reward_yard_to_truck: float = 3.0    # Customer service
    
    # IMPORT rewards (auto-transfer does these)
    reward_train_to_yard: float = 1.0
    reward_truck_to_yard: float = 0.8
    
    # Direct transfers (auto-transfer optimization)
    reward_train_to_truck: float = 4.0   # Skip yard = efficient
    reward_truck_to_train: float = 4.0   # Skip yard = efficient
    
    # Other moves
    reward_terminal_truck: float = 0.5
    
    # RESTACK penalty (agent should minimize these)
    penalty_yard_to_yard: float = -0.2   # Discourage restacking
    
    # Train departure
    reward_train_departed: float = 10.0  # Successful export batch
    penalty_train_leftover_per_id: float = -3.0  # Missed exports hurt
    
    # End of day penalties
    endday_blocking_weight: float = -0.5  # Blocking is bad
    endday_leftover_weight: float = -0.3  # Leftovers are bad
    
    # Truck wait time shaping
    waiting_penalty_per_truck_minute: float = 0.01  # Waiting costs
    truck_wait_fast_minutes: float = 60.0
    truck_wait_slow_minutes: float = 180.0
    truck_wait_reward_fast: float = 2.0   # Fast service = good
    truck_wait_reward_min: float = 0.0


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