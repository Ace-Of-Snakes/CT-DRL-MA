"""Enumerations for simulation entities."""
from enum import Enum


class Direction(str, Enum):
    """Container direction types."""
    IMPORT = "Import"
    EXPORT = "Export"


class GoodsType(str, Enum):
    """Container goods classification."""
    REGULAR = "Regular"
    REEFER = "Reefer"
    DANGEROUS_GOODS = "DangerousGoods"


class TruckStatus(str, Enum):
    """Truck operational status."""
    ARRIVING = "arriving"
    WAITING = "waiting"
    LOADING = "loading"
    DEPARTING = "departing"
    DEPARTED = "departed"


class TrainStatus(str, Enum):
    """Train operational status."""
    ARRIVING = "arriving"
    WAITING = "waiting"
    LOADING = "loading"
    DEPARTING = "departing"
    DEPARTED = "departed"


class TerminalTruckStatus(str, Enum):
    """Terminal truck status."""
    IDLE = "idle"
    BUSY = "busy"


class Region(str, Enum):
    """Unified spatial regions in the CNN state matrix."""
    RAIL = "RAIL"
    PARKING = "PARKING"
    YARD = "YARD"
    QUEUE = "QUEUE"


class MoveType(str, Enum):
    """Types of container moves."""
    YARD_TO_YARD = "YARD_TO_YARD"
    TRAIN_TO_YARD = "TRAIN_TO_YARD"
    YARD_TO_TRAIN = "YARD_TO_TRAIN"
    TRUCK_TO_YARD = "TRUCK_TO_YARD"
    YARD_TO_TRUCK = "YARD_TO_TRUCK"
    TRAIN_TO_TRUCK = "TRAIN_TO_TRUCK"
    TRUCK_TO_TRAIN = "TRUCK_TO_TRAIN"
    YARD_TO_TERMINAL_TRUCK = "YARD_TO_TERMINAL_TRUCK"
    SLOT_TRUCK_PARKING = "SLOT_TRUCK_PARKING"
    PARK_TRUCK = "PARK_TRUCK"


class PositionType(str, Enum):
    """Special position types in yard."""
    REEFER = "r"
    DANGEROUS_GOODS = "dg"
    SWAP_BODY_TRAILER = "sb_t"