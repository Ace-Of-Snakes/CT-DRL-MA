"""Configuration for crane operations and geometry."""
from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class CraneGeometry:
    """Physical dimensions and layout for RMGC operations."""
    
    # Yard dimensions
    bay_length_m: float = 12.2  # 40 feet in meters
    slot_width_m: float = 2.44    # Row spacing (y-direction)
    tier_height_m: float = 2.59   # Vertical tier spacing
    
    # Rail and parking layout
    track_width_m: float = 3.0
    space_rails_to_parking_m: float = 5.0
    parking_lane_width_m: float = 4.0
    driving_lane_width_m: float = 4.0
    space_driving_to_storage_m: float = 2.0
    
    # Vehicle heights
    ground_vehicle_height_m: float = 1.5  # Truck/train hook height


@dataclass(frozen=True)
class CranePerformance:
    """Performance parameters for crane movements."""
    
    # Speeds (meters per second)
    trolley_speed_mps: float = 70.0 / 60.0
    hoist_speed_mps: float = 28.0 / 60.0
    gantry_speed_mps: float = 130.0 / 60.0
    
    # Accelerations (meters per second squared)
    trolley_acc_mps2: float = 0.3
    hoist_acc_mps2: float = 0.2
    gantry_acc_mps2: float = 0.1
    
    # Operational limits
    max_hook_height_m: float = 20.0
    handling_time_s: float = 30.0  # Container lock/unlock time


class CraneDefaults:
    """Default crane operational parameters."""
    
    NUM_CRANES: Final[int] = 2
    ZONE_OVERLAP_BAYS: Final[int] = 4  # Bay overlap between crane zones
    PROXIMITY_SEARCH_BAYS: Final[int] = 3  # Search radius for placements