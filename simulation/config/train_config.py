# simulation/config/train_config.py
"""Configuration for train-related constants and defaults."""
import os
from typing import Final


class TrainDefaults:
    """Default configuration for trains."""

    # Wagon configuration (overridable via env var)
    NUM_WAGONS: Final[int] = int(os.getenv("WAGON_COUNT", "29"))