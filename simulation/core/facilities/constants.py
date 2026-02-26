"""Central constants for the container terminal simulation."""
from typing import Final

# === VEHICLE DIMENSIONS ===
STANDARD_VEHICLE_LENGTH_M: Final[float] = 24.4  # 80 feet in meters
STANDARD_VEHICLE_LENGTH_FT: Final[int] = 80
MIN_CONTAINER_LENGTH_M: Final[float] = 6.1  # Minimum container length

# === TIME CONSTANTS ===
SECONDS_PER_MINUTE: Final[int] = 60
SECONDS_PER_HOUR: Final[int] = 3_600
SECONDS_PER_DAY: Final[int] = 86_400
SECONDS_PER_WEEK: Final[int] = 604_800

# === TERMINAL OPERATIONS ===
TERMINAL_TRUCK_TASK_DURATION_S: Final[float] = 300.0  # 5 minutes for TT jobs
DEFAULT_STEP_MINUTES: Final[int] = 5  # Default simulation step size

# === PARKING ===
DEFAULT_PARKING_PREFIX: Final[str] = "P"

# === NUMERIC DEFAULTS ===
EPSILON_TOLERANCE: Final[float] = 1e-3  # For floating point comparisons
DEFAULT_BUFFER_TIME_HOURS: Final[int] = 4

# === TERMINAL LAYOUT ===
import os
import pandas as pd
import numpy as np
# this is going to be a funcitonal draft which is to be replaced by a more sophisticated
# implementation in the future

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
df = pd.read_csv(os.path.join(_DATA_DIR, "container_data.csv"))

BAY_LENGTH_FT = 40
'''A bay is the basic unit of storage in a block. It is 40 feet long.'''
BAY_SPLIT_FACTOR = 20
'''The bay-split factor defines how many sub-bays a bay is split into.'''
SUB_BAY_LENGTH_FT = BAY_LENGTH_FT / BAY_SPLIT_FACTOR
'''A sub-bay is the smallest unit of storage in a bay. Containers can be spanned across multiple sub-bays.'''

# read out all unique container lengths from the CSV
CONTAINER_LENGTHS_FT = sorted(df["LENGTH_IN_FEET"].unique().tolist())

CONTAINER_LENGTH_TO_SUB_BAYS = {
    length: int(np.ceil(length / SUB_BAY_LENGTH_FT)) for length in CONTAINER_LENGTHS_FT
}
'''Mapping from container length in feet to number of sub-bays it occupies.'''

