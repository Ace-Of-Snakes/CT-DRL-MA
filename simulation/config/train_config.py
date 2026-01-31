# simulation/config/train_config.py
"""
Configuration for train-related constants and defaults.
"""
import os


class TrainDefaults:
    """Default configuration for trains."""
    
    # Wagon configuration
    NUM_WAGONS = int(os.getenv("WAGON_COUNT", "29"))  # Can be overridden via env var
    WAGON_LENGTH_M = 24.4  # Standard wagon length in meters (80 feet)
    
    # ID generation
    ID_PREFIX = "TRN"
    ID_MIN_RANDOM = 10_000
    ID_MAX_RANDOM = 99_999
    
    # Status values (kept as strings for compatibility)
    STATUS_ARRIVING = "arriving"
    STATUS_WAITING = "waiting"
    STATUS_LOADING = "loading"
    STATUS_DEPARTING = "departing"
    STATUS_DEPARTED = "departed"


class TrainLoaderConfig:
    """Configuration for train loading operations."""
    
    # Overgeneration factor: generate N times more containers than wagons
    OVERGENERATION_FACTOR = 5.0
    
    # Direction - trains only handle imports
    TRAIN_DIRECTION = "Import"
    
    # Minimum container length for placement checks (meters)
    MIN_CONTAINER_LENGTH_M = 6.1