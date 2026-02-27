# simulation/core/facilities/railyard.py
"""
Grid-based rail yard with spatial container tracking.

Each train is assigned a slot (track_id, anchor_bay). Containers on the
train's wagons are mirrored in a spatial grid (occupancy_mask, position_grid)
for O(1) lookups — the same pattern used by StorageYard and ParkingArea.

The grid is synchronised via sync_train() after any operation that mutates
a train's container state (add/remove via TLM).
"""
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Union, List, Iterator, Tuple

from simulation.core.containers.container import Container
from simulation.core.vehicles.train import Train
from simulation.core.facilities.base import EMPTY_SLOT
from simulation.core.facilities.constants import BAY_SPLIT_FACTOR

TrackId = Union[int, str]

MAX_RAIL_CONTAINERS: int = 500


@dataclass(slots=True, frozen=True)
class RailSlot:
    """Rail slot assignment."""
    track_id: TrackId
    anchor_bay: int  # Yard bay index for proximity searches


@dataclass(slots=True)
class RailContainerRecord:
    """A container placed on a train in the rail yard."""
    container: Container
    track: int           # track row (0..n_tracks-1)
    abs_split: int       # absolute split start
    n_splits: int        # container footprint in splits
    train_id: str
    wagon_index: int
    idx: int             # internal index in _records


def _container_n_splits(container: Container, split_factor: int) -> int:
    """Number of splits a container occupies."""
    length_ft = getattr(container, "length_ft", None) or getattr(
        container, "length", 20
    )
    sub_bay_ft = 40.0 / split_factor
    return max(1, int(np.ceil(length_ft / sub_bay_ft)))


class RailYard:
    """
    Grid-based rail yard with train-to-track slot mapping.

    Maintains spatial arrays for O(1) container lookups:
    - occupancy_mask: (n_tracks, total_splits) bool
    - position_grid:  (n_tracks, total_splits) int32 → record index

    Call sync_train() after any external mutation to a train's containers.
    """

    __slots__ = (
        'n_tracks', 'n_bays', 'split_factor', 'total_splits',
        '_train_to_slot', '_track_trains',
        'occupancy_mask', 'position_grid',
        '_records', '_id_to_idx', '_free_indices',
        '_train_records',  # train_id → list of record indices
    )

    def __init__(
        self,
        n_tracks: int = 10,
        n_bays: int = 58,
        split_factor: int = BAY_SPLIT_FACTOR,
    ):
        self.n_tracks = n_tracks
        self.n_bays = n_bays
        self.split_factor = split_factor
        self.total_splits = n_bays * split_factor

        # Slot tracking (existing logic)
        self._train_to_slot: Dict[str, RailSlot] = {}
        self._track_trains: Dict[TrackId, List[str]] = {}

        # Spatial grid
        self.occupancy_mask = np.zeros(
            (n_tracks, self.total_splits), dtype=bool,
        )
        self.position_grid = np.full(
            (n_tracks, self.total_splits), EMPTY_SLOT, dtype=np.int32,
        )

        # Record storage
        self._records: List[Optional[RailContainerRecord]] = [
            None
        ] * MAX_RAIL_CONTAINERS
        self._id_to_idx: Dict[str, int] = {}
        self._free_indices: List[int] = list(
            range(MAX_RAIL_CONTAINERS - 1, -1, -1)
        )

        # Track which record indices belong to which train
        self._train_records: Dict[str, List[int]] = {}

    # ══════════════════════════════════════════════════════════════════
    # Train slot management
    # ══════════════════════════════════════════════════════════════════

    def slot_train(self, train: Train, slot: RailSlot) -> None:
        """Assign train to slot and populate grid from its current containers."""
        train_id = train.train_id

        # Remove from old slot if exists
        old_slot = self._train_to_slot.get(train_id)
        if old_slot:
            self._clear_train_grid(train_id)
            self._remove_from_track(old_slot.track_id, train_id)

        # Add to new slot
        self._train_to_slot[train_id] = slot
        self._add_to_track(slot.track_id, train_id)
        train.rail_track = str(slot.track_id)

        # Populate grid
        self._sync_train_grid(train_id, train)

    def release_train(self, train_id: str) -> Optional[RailSlot]:
        """Release train from slot and clear its grid entries."""
        slot = self._train_to_slot.pop(train_id, None)
        if slot:
            self._clear_train_grid(train_id)
            self._remove_from_track(slot.track_id, train_id)
        return slot

    def sync_train(self, train_id: str, train: Train) -> None:
        """Rebuild grid for one train from its current wagon state.

        Call this after any container add/remove on the train.
        """
        slot = self._train_to_slot.get(train_id)
        if slot is None:
            return
        self._clear_train_grid(train_id)
        self._sync_train_grid(train_id, train)

    # ══════════════════════════════════════════════════════════════════
    # O(1) spatial lookups
    # ══════════════════════════════════════════════════════════════════

    def get_container_at(
        self, track: int, split: int,
    ) -> Optional[Container]:
        """O(1) container lookup at (track, split)."""
        if track < 0 or track >= self.n_tracks:
            return None
        if split < 0 or split >= self.total_splits:
            return None
        idx = self.position_grid[track, split]
        if idx == EMPTY_SLOT:
            return None
        rec = self._records[idx]
        return rec.container if rec is not None else None

    def get_record_at(
        self, track: int, split: int,
    ) -> Optional[RailContainerRecord]:
        """O(1) full record lookup at (track, split)."""
        if track < 0 or track >= self.n_tracks:
            return None
        if split < 0 or split >= self.total_splits:
            return None
        idx = self.position_grid[track, split]
        if idx == EMPTY_SLOT:
            return None
        return self._records[idx]

    def get_record_by_id(
        self, container_id: str,
    ) -> Optional[RailContainerRecord]:
        """O(1) record lookup by container_id."""
        idx = self._id_to_idx.get(container_id)
        if idx is None:
            return None
        return self._records[idx]

    # ══════════════════════════════════════════════════════════════════
    # Iteration
    # ══════════════════════════════════════════════════════════════════

    def iter_records(self) -> Iterator[RailContainerRecord]:
        """Iterate over all container records with valid placements."""
        for idx_list in self._train_records.values():
            for idx in idx_list:
                rec = self._records[idx]
                if rec is not None:
                    yield rec

    def container_count(self) -> int:
        """Total containers currently tracked in the grid."""
        return len(self._id_to_idx)

    # ══════════════════════════════════════════════════════════════════
    # Slot queries (existing API preserved)
    # ══════════════════════════════════════════════════════════════════

    def get_anchor_bay(self, train_id: str) -> Optional[int]:
        """Get anchor bay for train."""
        slot = self._train_to_slot.get(train_id)
        return slot.anchor_bay if slot else None

    def get_slot(self, train_id: str) -> Optional[RailSlot]:
        """Get full slot info for train."""
        return self._train_to_slot.get(train_id)

    def iter_slotted_trains(self) -> Iterator[Tuple[str, RailSlot]]:
        """Iterate over (train_id, slot) pairs."""
        return iter(self._train_to_slot.items())

    def get_all_anchor_bays(self) -> Dict[str, int]:
        """Get mapping of train_id -> anchor_bay for all slotted trains."""
        return {
            train_id: slot.anchor_bay
            for train_id, slot in self._train_to_slot.items()
        }

    def get_trains_on_track(self, track_id: TrackId) -> List[str]:
        """Get list of train_ids on a given track."""
        return list(self._track_trains.get(track_id, []))

    def get_occupancy_mask(self) -> np.ndarray:
        """Return occupancy_mask reference (n_tracks, total_splits)."""
        return self.occupancy_mask

    # ══════════════════════════════════════════════════════════════════
    # Internal: grid sync
    # ══════════════════════════════════════════════════════════════════

    def _sync_train_grid(self, train_id: str, train: Train) -> None:
        """Populate grid from train's current wagon containers."""
        slot = self._train_to_slot.get(train_id)
        if slot is None:
            return

        track = slot.track_id
        if isinstance(track, str):
            try:
                track = int(track)
            except (ValueError, TypeError):
                return
        if track < 0 or track >= self.n_tracks:
            return

        anchor = slot.anchor_bay
        sf = self.split_factor
        record_indices: List[int] = []

        for wagon_idx, wagon in enumerate(train.wagons):
            wagon_bay = anchor + wagon_idx
            if wagon_bay < 0 or wagon_bay >= self.n_bays:
                continue

            split_cursor = 0
            for container in wagon.containers.values():
                n_splits = _container_n_splits(container, sf)
                abs_split = wagon_bay * sf + split_cursor
                s_end = min(abs_split + n_splits, self.total_splits)

                if abs_split >= self.total_splits:
                    break

                # Allocate record index
                if not self._free_indices:
                    break  # out of space
                idx = self._free_indices.pop()

                rec = RailContainerRecord(
                    container=container,
                    track=track,
                    abs_split=abs_split,
                    n_splits=n_splits,
                    train_id=train_id,
                    wagon_index=wagon_idx,
                    idx=idx,
                )
                self._records[idx] = rec
                self._id_to_idx[container.container_id] = idx
                record_indices.append(idx)

                # Fill grid
                self.occupancy_mask[track, abs_split:s_end] = True
                self.position_grid[track, abs_split:s_end] = idx

                split_cursor += n_splits

        self._train_records[train_id] = record_indices

    def _clear_train_grid(self, train_id: str) -> None:
        """Clear all grid entries for a train."""
        indices = self._train_records.pop(train_id, [])
        for idx in indices:
            rec = self._records[idx]
            if rec is not None:
                track = rec.track
                s0 = rec.abs_split
                s1 = min(s0 + rec.n_splits, self.total_splits)
                self.occupancy_mask[track, s0:s1] = False
                self.position_grid[track, s0:s1] = EMPTY_SLOT
                self._id_to_idx.pop(rec.container.container_id, None)
                self._records[idx] = None
                self._free_indices.append(idx)

    # ══════════════════════════════════════════════════════════════════
    # Internal: track bookkeeping
    # ══════════════════════════════════════════════════════════════════

    def _add_to_track(self, track_id: TrackId, train_id: str) -> None:
        if track_id not in self._track_trains:
            self._track_trains[track_id] = []
        if train_id not in self._track_trains[track_id]:
            self._track_trains[track_id].append(train_id)

    def _remove_from_track(self, track_id: TrackId, train_id: str) -> None:
        if track_id in self._track_trains:
            try:
                self._track_trains[track_id].remove(train_id)
            except ValueError:
                pass
