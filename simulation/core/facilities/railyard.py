# simulation/core/facilities/railyard_optimized.py
"""
Optimized rail yard with slots and optional array-based storage.
Mostly unchanged from original - not a performance bottleneck.
"""
from dataclasses import dataclass
from typing import Dict, Optional, Union, List, Iterator
from simulation.core.vehicles.train import Train


TrackId = Union[int, str]


@dataclass(slots=True, frozen=True)
class RailSlot:
    """Rail slot assignment."""
    track_id: TrackId
    anchor_bay: int  # Yard bay index for proximity searches


class OptimizedRailYard:
    """
    Rail yard with train-to-track slot mapping.
    
    Optimizations:
    - __slots__ for memory efficiency
    - Optional array-based storage for dense train IDs
    """
    
    __slots__ = ('_train_to_slot', '_track_trains', 'n_tracks')
    
    def __init__(self, n_tracks: int = 10):
        """
        Initialize rail yard.
        
        Args:
            n_tracks: Number of rail tracks
        """
        self.n_tracks = n_tracks
        self._train_to_slot: Dict[str, RailSlot] = {}
        # Track -> list of train_ids for reverse lookup
        self._track_trains: Dict[TrackId, List[str]] = {}
    
    def slot_train(self, train: Train, slot: RailSlot) -> None:
        """Assign train to slot."""
        train_id = train.train_id
        
        # Remove from old slot if exists
        old_slot = self._train_to_slot.get(train_id)
        if old_slot:
            self._remove_from_track(old_slot.track_id, train_id)
        
        # Add to new slot
        self._train_to_slot[train_id] = slot
        self._add_to_track(slot.track_id, train_id)
        train.rail_track = str(slot.track_id)
    
    def release_train(self, train_id: str) -> Optional[RailSlot]:
        """Release train from slot. Returns the slot if found."""
        slot = self._train_to_slot.pop(train_id, None)
        if slot:
            self._remove_from_track(slot.track_id, train_id)
        return slot
    
    def get_anchor_bay(self, train_id: str) -> Optional[int]:
        """Get anchor bay for train."""
        slot = self._train_to_slot.get(train_id)
        return slot.anchor_bay if slot else None
    
    def get_slot(self, train_id: str) -> Optional[RailSlot]:
        """Get full slot info for train."""
        return self._train_to_slot.get(train_id)

    def iter_slotted_trains(self) -> Iterator[tuple[str, RailSlot]]:
        """Iterate over (train_id, slot) pairs."""
        return iter(self._train_to_slot.items())
    
    def get_all_anchor_bays(self) -> Dict[str, int]:
        """Get mapping of train_id -> anchor_bay for all slotted trains."""
        return {
            train_id: slot.anchor_bay
            for train_id, slot in self._train_to_slot.items()
        }
    
    def _add_to_track(self, track_id: TrackId, train_id: str):
        """Internal: add train to track list."""
        if track_id not in self._track_trains:
            self._track_trains[track_id] = []
        if train_id not in self._track_trains[track_id]:
            self._track_trains[track_id].append(train_id)
    
    def _remove_from_track(self, track_id: TrackId, train_id: str):
        """Internal: remove train from track list."""
        if track_id in self._track_trains:
            try:
                self._track_trains[track_id].remove(train_id)
            except ValueError:
                pass