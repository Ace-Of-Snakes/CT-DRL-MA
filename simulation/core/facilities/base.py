# simulation/core/facilities/base.py
"""Shared types for facility grid interfaces."""
from dataclasses import dataclass
from typing import Any, Optional

from simulation.core.containers.container import Container


EMPTY_SLOT: int = -1
"""Sentinel value for empty position_grid entries."""


@dataclass(slots=True)
class FacilityPlacement:
    """Position within a facility's local coordinate system."""
    row: int          # local row (track_id for rail, 0 for parking, yard row)
    abs_split: int    # absolute split index (bay * split_factor + offset)
    tier: int         # 0 for rail/parking, 0-4 for yard
    n_splits: int     # how many splits this entity occupies


@dataclass(slots=True)
class FacilityRecord:
    """An entity placed in a facility with its position."""
    entity_id: str                       # container_id or truck_id
    container: Optional[Container]       # the container (None for empty truck)
    placement: FacilityPlacement
    metadata: dict                       # facility-specific extras
