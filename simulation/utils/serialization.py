"""JSON serialization utilities for simulation objects."""
from typing import Any, Union, Dict, List
from datetime import datetime
import numpy as np

try:
    from simulation.core.facilities.yard import PlacementResult
except ImportError:
    PlacementResult = None


def to_json_serializable(obj: Any) -> Union[Dict, List, str, int, float, bool, None]:
    """
    Convert simulation objects to JSON-serializable types.
    
    Args:
        obj: Any Python object from the simulation
        
    Returns:
        JSON-serializable representation
        
    Examples:
        >>> to_json_serializable(datetime(2024, 1, 1))
        '2024-01-01T00:00:00'
        >>> to_json_serializable(np.float32(3.14))
        3.14
    """
    # PlacementResult
    if PlacementResult is not None and isinstance(obj, PlacementResult):
        return {
            "row": int(obj.row),
            "bay": int(obj.bay),
            "tier": int(obj.tier),
            "start_split": int(obj.start_split)
        }
    
    # Datetime
    if isinstance(obj, datetime):
        return obj.isoformat()
    
    # Numpy types
    if isinstance(obj, np.generic):
        return obj.item()
    
    # Collections
    if isinstance(obj, (set, tuple)):
        return [to_json_serializable(x) for x in obj]
    
    if isinstance(obj, list):
        return [to_json_serializable(x) for x in obj]
    
    if isinstance(obj, dict):
        return {str(k): to_json_serializable(v) for k, v in obj.items()}
    
    # Primitives
    return obj


def serialize_move_record(
    timestamp: datetime,
    crane_id: int,
    move_type: str,
    args: Dict[str, Any],
    distance_m: float,
    time_s: float,
    reward: float,
    **extra_fields
) -> Dict[str, Any]:
    """
    Create a standardized move record for logging.
    
    Args:
        timestamp: When the move occurred
        crane_id: Which crane performed the move
        move_type: Type of move (from MoveType enum)
        args: Move-specific arguments
        distance_m: Distance traveled
        time_s: Time taken
        reward: Reward received
        **extra_fields: Additional fields to include
        
    Returns:
        JSON-serializable dictionary
    """
    record = {
        "timestamp": timestamp.isoformat(),
        "crane_id": crane_id,
        "move_type": move_type,
        "args": to_json_serializable(args),
        "distance_m": float(distance_m),
        "time_s": float(time_s),
        "reward": float(reward),
    }
    record.update(to_json_serializable(extra_fields))
    return record