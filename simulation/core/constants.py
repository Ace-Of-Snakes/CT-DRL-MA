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

# === TRAIN OPERATIONS ===
DEFAULT_TRAIN_NUM_WAGONS: Final[int] = 29
TRAIN_IMPORT_DIRECTION: Final[str] = "Import"

# === NUMERIC DEFAULTS ===
EPSILON_TOLERANCE: Final[float] = 1e-3  # For floating point comparisons
DEFAULT_BUFFER_TIME_HOURS: Final[int] = 4