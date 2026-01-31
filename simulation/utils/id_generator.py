"""Centralized ID generation for simulation entities."""
import random
from typing import Final


class IDGenerator:
    """Generates unique IDs for simulation entities."""
    
    # Prefixes
    _TRAIN_PREFIX: Final[str] = "TRN"
    _TRUCK_PREFIX: Final[str] = "TRK"
    _TERMINAL_TRUCK_PREFIX: Final[str] = "TTR"
    _CONTAINER_PREFIX: Final[str] = "C"
    
    # Random ID range
    _MIN_RANDOM: Final[int] = 10_000
    _MAX_RANDOM: Final[int] = 99_999
    
    @staticmethod
    def generate_train_id() -> str:
        """Generate unique train ID."""
        return f"{IDGenerator._TRAIN_PREFIX}{random.randint(IDGenerator._MIN_RANDOM, IDGenerator._MAX_RANDOM)}"
    
    @staticmethod
    def generate_truck_id(prefix: str = None) -> str:
        """
        Generate unique truck ID.
        
        Args:
            prefix: Optional custom prefix (defaults to TRK)
        """
        prefix = prefix or IDGenerator._TRUCK_PREFIX
        return f"{prefix}{random.randint(IDGenerator._MIN_RANDOM, IDGenerator._MAX_RANDOM)}"
    
    @staticmethod
    def generate_terminal_truck_id(index: int = None) -> str:
        """
        Generate terminal truck ID.
        
        Args:
            index: Optional sequential index for terminal trucks
        """
        if index is not None:
            return f"{IDGenerator._TERMINAL_TRUCK_PREFIX}_{index + 1}"
        return f"{IDGenerator._TERMINAL_TRUCK_PREFIX}{random.randint(IDGenerator._MIN_RANDOM, IDGenerator._MAX_RANDOM)}"
    
    @staticmethod
    def generate_container_id(counter: int) -> str:
        """
        Generate container ID with counter.
        
        Args:
            counter: Sequential counter value
        """
        return f"{IDGenerator._CONTAINER_PREFIX}{counter:06d}"