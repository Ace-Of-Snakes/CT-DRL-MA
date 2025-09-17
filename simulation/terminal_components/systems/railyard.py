# simulation/terminal_components/systems/railyard.py
from dataclasses import dataclass
from typing import Dict, Optional, Union
from simulation.terminal_components.vehicles.Train import Train

TrackId = Union[int, str]

@dataclass(frozen=True)
class RailSlot:
    track_id: TrackId
    anchor_bay: int  # yard bay index used to bias yard placement searches

class BooleanRailYard:
    """
    Minimal rail yard: only remembers which train is slotted to which track
    and exposes the anchor_bay used for yard proximity searches.
    """
    def __init__(self):
        self._train_to_slot: Dict[str, RailSlot] = {}

    def slot_train(self, train: Train, slot: RailSlot) -> None:
        self._train_to_slot[train.train_id] = slot
        train.rail_track = str(slot.track_id)

    def release_train(self, train_id: str) -> None:
        self._train_to_slot.pop(train_id, None)

    def get_anchor_bay(self, train_id: str) -> Optional[int]:
        s = self._train_to_slot.get(train_id)
        return s.anchor_bay if s else None
    
    def get_slot(self, train_id: str) -> Optional[RailSlot]:
        return self._train_to_slot.get(train_id)