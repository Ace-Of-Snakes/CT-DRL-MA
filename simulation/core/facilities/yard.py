# simulation/core/facilities/yard_optimized.py
"""
Optimized storage yard with array-based lookups.
Key changes:
- position_grid: 3D int32 array replaces Dict[Tuple, str]
- Container indexing: internal int indices, string ID mapping at boundary
- Slots on dataclasses
- Single-placement fast path for RL
"""
import numpy as np
from datetime import datetime
from typing import Dict, Tuple, List, Optional, Iterator
from dataclasses import dataclass
from simulation.core.containers.container import Container
from simulation.core.facilities.constants import (
    BAY_SPLIT_FACTOR, CONTAINER_LENGTH_TO_SUB_BAYS
)

# Constants
EMPTY_SLOT: int = -1
MAX_CONTAINERS: int = 2000  # Pre-allocate for this many


@dataclass(slots=True)
class PlacementResult:
    """Result of container placement search."""
    row: int
    bay: int
    tier: int
    start_split: int
    
    @property
    def abs_start(self) -> int:
        """Absolute split index."""
        return self.bay * BAY_SPLIT_FACTOR + self.start_split


@dataclass(slots=True)
class ContainerRecord:
    """Complete record of a container and its placement."""
    container: Container
    placement: PlacementResult
    n_splits: int
    idx: int  # Internal index
    is_accessible: bool = True


class OptimizedStorageYard:
    """
    High-performance storage yard optimized for RL training.
    
    Key optimizations:
    - O(1) position lookups via numpy grid
    - Integer container indexing internally
    - Single-placement search with early exit
    - Minimal allocations in hot paths
    """
    
    __slots__ = (
        'n_rows', 'n_bays', 'n_tiers', 'split_factor', 'total_splits',
        'occupancy_mask', 'position_grid', 'container_length_map',
        '_records', '_id_to_idx', '_free_indices', '_next_idx',
        '_accessible_mask', '_tier_counts',
        'reefer_mask', 'dangerous_mask', 'swapbody_mask', 'regular_mask',
        '_support_cache', '_empty_int32'
    )
    
    def __init__(
        self,
        n_rows: int,
        n_bays: int,
        n_tiers: int,
        coordinates: List[Tuple[int, int, str]],
        max_containers: int = MAX_CONTAINERS,
        validate: bool = False
    ):
        self.n_rows = n_rows
        self.n_bays = n_bays
        self.n_tiers = n_tiers
        self.split_factor = BAY_SPLIT_FACTOR
        self.total_splits = n_bays * self.split_factor
        
        # Primary spatial structure: (tier, row, split) -> container_idx
        self.position_grid = np.full(
            (n_tiers, n_rows, self.total_splits),
            EMPTY_SLOT,
            dtype=np.int32
        )
        
        # Occupancy mask for fast availability checks
        self.occupancy_mask = np.zeros(
            (n_tiers, n_rows, self.total_splits),
            dtype=bool
        )
        
        # Container storage: index-based
        self._records: List[Optional[ContainerRecord]] = [None] * max_containers
        self._id_to_idx: Dict[str, int] = {}
        self._free_indices: List[int] = list(range(max_containers - 1, -1, -1))
        self._next_idx = 0
        
        # Accessibility tracking: bool array
        self._accessible_mask = np.zeros(max_containers, dtype=bool)
        
        # Tier membership counts (for iteration optimization)
        self._tier_counts = np.zeros(n_tiers, dtype=np.int32)
        
        # Special position masks
        self._init_special_masks(coordinates)
        
        # Container length mapping
        self.container_length_map = CONTAINER_LENGTH_TO_SUB_BAYS
        
        # Support cache: (tier, n_splits) -> mask or None
        self._support_cache: Dict[Tuple[int, int], Optional[np.ndarray]] = {}
        
        # Pre-allocated empty array for returns
        self._empty_int32 = np.empty(0, dtype=np.int32)
        
        if validate:
            self._print_masks()
    
    def _init_special_masks(self, coordinates: List[Tuple[int, int, str]]):
        """Initialize special position masks."""
        shape = (self.n_tiers, self.n_rows, self.total_splits)
        self.reefer_mask = np.zeros(shape, dtype=bool)
        self.dangerous_mask = np.zeros(shape, dtype=bool)
        self.swapbody_mask = np.zeros(shape, dtype=bool)
        
        for bay, row, pos_type in coordinates:
            bay_idx = bay - 1
            row_idx = row - 1
            s0 = bay_idx * self.split_factor
            s1 = s0 + self.split_factor
            
            if pos_type == "r":
                self.reefer_mask[:, row_idx, s0:s1] = True
            elif pos_type == "dg":
                self.dangerous_mask[:, row_idx, s0:s1] = True
            elif pos_type == "sb_t":
                self.swapbody_mask[0, row_idx, s0:s1] = True
        
        self.regular_mask = ~(self.reefer_mask | self.dangerous_mask)
    
    # ========== Container Management ==========
    
    def add_container(self, container: Container, placement: PlacementResult) -> int:
        """
        Add container to yard.
        Returns internal index.
        """
        n_splits = self.container_length_map.get(container.length_ft, 0)
        if n_splits <= 0:
            raise ValueError(f"Unsupported container length: {container.length_ft}")
        
        # Validate bounds
        row, tier = placement.row, placement.tier
        abs_start = placement.abs_start
        abs_end = abs_start + n_splits
        
        if not (0 <= row < self.n_rows and 0 <= tier < self.n_tiers):
            raise IndexError("Placement out of bounds (row/tier)")
        if abs_end > self.total_splits:
            raise IndexError("Placement out of bounds (splits)")
        
        # Allocate index
        if not self._free_indices:
            raise RuntimeError("Container capacity exceeded")
        idx = self._free_indices.pop()
        
        # Update occupancy
        self.occupancy_mask[tier, row, abs_start:abs_end] = True
        self.position_grid[tier, row, abs_start:abs_end] = idx
        
        # Check accessibility
        is_accessible = not np.any(
            self.occupancy_mask[tier + 1:, row, abs_start:abs_end]
        ) if tier < self.n_tiers - 1 else True
        
        # Create record
        record = ContainerRecord(
            container=container,
            placement=placement,
            n_splits=n_splits,
            idx=idx,
            is_accessible=is_accessible
        )
        
        self._records[idx] = record
        self._id_to_idx[container.container_id] = idx
        self._accessible_mask[idx] = is_accessible
        self._tier_counts[tier] += 1
        
        # Update support cache
        self._update_support_cache(tier + 1, n_splits, row, abs_start, abs_end, True)
        
        # Update accessibility of container below
        self._update_below_accessibility(row, tier, abs_start, make_accessible=False)
        
        return idx
    
    def remove_container(self, container: Container) -> Optional[Container]:
        """Remove container from yard."""
        cid = container.container_id
        idx = self._id_to_idx.get(cid)
        if idx is None:
            return None
        
        record = self._records[idx]
        if record is None:
            return None
        
        row = record.placement.row
        tier = record.placement.tier
        abs_start = record.placement.abs_start
        abs_end = abs_start + record.n_splits
        
        # Clear occupancy
        self.occupancy_mask[tier, row, abs_start:abs_end] = False
        self.position_grid[tier, row, abs_start:abs_end] = EMPTY_SLOT
        
        # Update support cache
        self._update_support_cache(tier + 1, record.n_splits, row, abs_start, abs_end, False)
        
        # Update accessibility below
        self._update_below_accessibility(row, tier, abs_start, make_accessible=True)
        
        # Clean up
        self._records[idx] = None
        del self._id_to_idx[cid]
        self._free_indices.append(idx)
        self._accessible_mask[idx] = False
        self._tier_counts[tier] -= 1
        
        return record.container
    
    def move_container(self, container_id: str, dest: PlacementResult) -> bool:
        """Move container to new position."""
        idx = self._id_to_idx.get(container_id)
        if idx is None:
            return False
        
        record = self._records[idx]
        if record is None:
            return False
        
        container = record.container
        self.remove_container(container)
        self.add_container(container, dest)
        return True
    
    # ========== Lookups ==========
    
    def get_container(self, container_id: str) -> Optional[Container]:
        """Get container by ID - O(1)."""
        idx = self._id_to_idx.get(container_id)
        if idx is None:
            return None
        record = self._records[idx]
        return record.container if record else None
    
    def get_record(self, container_id: str) -> Optional[ContainerRecord]:
        """Get full record by ID - O(1)."""
        idx = self._id_to_idx.get(container_id)
        if idx is None:
            return None
        return self._records[idx]
    
    def get_container_at(self, row: int, tier: int, split: int) -> Optional[Container]:
        """Get container at position - O(1)."""
        if not (0 <= tier < self.n_tiers and 0 <= row < self.n_rows and 0 <= split < self.total_splits):
            return None
        idx = self.position_grid[tier, row, split]
        if idx == EMPTY_SLOT:
            return None
        record = self._records[idx]
        return record.container if record else None
    
    def get_placement(self, container_id: str) -> Optional[PlacementResult]:
        """Get placement by ID - O(1)."""
        idx = self._id_to_idx.get(container_id)
        if idx is None:
            return None
        record = self._records[idx]
        return record.placement if record else None
    
    # ========== Iteration ==========
    
    def iter_records(self) -> Iterator[ContainerRecord]:
        """Iterate over all container records efficiently."""
        for idx in self._id_to_idx.values():
            record = self._records[idx]
            if record is not None:
                yield record
    
    def iter_accessible(self) -> Iterator[ContainerRecord]:
        """Iterate over accessible containers."""
        for idx in self._id_to_idx.values():
            if self._accessible_mask[idx]:
                record = self._records[idx]
                if record is not None:
                    yield record
    
    @property
    def container_count(self) -> int:
        """Total container count."""
        return len(self._id_to_idx)
    
    # ========== Placement Search ==========
    
    def find_single_placement(
        self,
        container: Container,
        target_bay: int,
        max_proximity: int = 3,
        prefer_lower_tier: bool = True
    ) -> Optional[PlacementResult]:
        """
        Fast single-placement search for RL.
        Returns first valid placement, spiraling from target bay.
        """
        n_splits = self.container_length_map.get(container.length_ft, 0)
        if n_splits <= 0:
            return None
        
        target_bay = max(0, min(target_bay, self.n_bays - 1))
        
        # Tier order
        tiers = range(self.n_tiers) if prefer_lower_tier else range(self.n_tiers - 1, -1, -1)
        
        for tier in tiers:
            # Spiral outward from target bay
            for offset in range(max_proximity + 1):
                for bay in (target_bay + offset, target_bay - offset):
                    if bay < 0 or bay >= self.n_bays:
                        continue
                    if offset == 0 and bay != target_bay:
                        continue
                    
                    result = self._find_in_bay(container, tier, bay, n_splits)
                    if result is not None:
                        return result
        
        return None
    
    def _find_in_bay(
        self,
        container: Container,
        tier: int,
        bay: int,
        n_splits: int
    ) -> Optional[PlacementResult]:
        """Find valid placement within a specific bay."""
        base = bay * self.split_factor
        max_start = self.split_factor - n_splits + 1
        
        goods_mask = self._get_goods_mask_tier(container, tier)
        
        for start_split in range(max_start):
            abs_start = base + start_split
            abs_end = abs_start + n_splits
            
            for row in range(self.n_rows):
                if self._is_valid_placement(tier, row, abs_start, abs_end, n_splits, goods_mask):
                    return PlacementResult(row=row, bay=bay, tier=tier, start_split=start_split)
        
        return None
    
    def _is_valid_placement(
        self,
        tier: int,
        row: int,
        abs_start: int,
        abs_end: int,
        n_splits: int,
        goods_mask: np.ndarray
    ) -> bool:
        """Check if position is valid for placement - minimal allocations."""
        # 1. Occupancy check
        if np.any(self.occupancy_mask[tier, row, abs_start:abs_end]):
            return False
        
        # 2. Goods type check
        if not np.all(goods_mask[row, abs_start:abs_end]):
            return False
        
        # 3. Support check (tier > 0)
        if tier > 0:
            support = self._get_support_mask(tier, n_splits)
            if support is not None and not np.all(support[row, abs_start:abs_end]):
                return False
        
        return True
    
    # ========== Mask Helpers ==========
    
    def _get_goods_mask_tier(self, container: Container, tier: int) -> np.ndarray:
        """Get goods type mask for specific tier."""
        if container.goods_type == "Reefer":
            return self.reefer_mask[tier]
        elif container.goods_type == "DangerousGoods":
            return self.dangerous_mask[tier]
        elif container.is_swap_body or container.is_trailer:
            if tier == 0:
                return self.swapbody_mask[tier]
            return np.zeros((self.n_rows, self.total_splits), dtype=bool)
        return self.regular_mask[tier]
    
    def _get_support_mask(self, tier: int, n_splits: int) -> Optional[np.ndarray]:
        """Get or build support mask for tier."""
        if tier == 0:
            return None
        
        key = (tier, n_splits)
        mask = self._support_cache.get(key)
        
        if mask is None:
            # Build from scratch (cold path)
            mask = np.zeros((self.n_rows, self.total_splits), dtype=bool)
            
            # Check tier below for containers of same length
            below = tier - 1
            for cid, idx in self._id_to_idx.items():
                rec = self._records[idx]
                if rec and rec.placement.tier == below and rec.n_splits == n_splits:
                    r = rec.placement.row
                    s = rec.placement.abs_start
                    mask[r, s:s + n_splits] = True
            
            self._support_cache[key] = mask
        
        return mask
    
    def _update_support_cache(
        self,
        tier: int,
        n_splits: int,
        row: int,
        abs_start: int,
        abs_end: int,
        value: bool
    ):
        """Incrementally update support cache."""
        if tier <= 0 or tier >= self.n_tiers:
            return
        
        key = (tier, n_splits)
        mask = self._support_cache.get(key)
        
        if mask is None:
            mask = np.zeros((self.n_rows, self.total_splits), dtype=bool)
            self._support_cache[key] = mask
        
        mask[row, abs_start:abs_end] = value
    
    def _update_below_accessibility(
        self,
        row: int,
        tier: int,
        abs_start: int,
        make_accessible: bool
    ):
        """Update accessibility of container directly below."""
        if tier == 0:
            return
        
        below_tier = tier - 1
        below_idx = self.position_grid[below_tier, row, abs_start]
        
        if below_idx == EMPTY_SLOT:
            return
        
        rec: ContainerRecord = self._records[below_idx]
        if rec is None:
            return
        
        abs_end = abs_start + rec.n_splits
        
        if make_accessible:
            # Check if anything above
            nothing_above = not np.any(
                self.occupancy_mask[below_tier + 1:, row, abs_start:abs_end]
            )
            rec.is_accessible = nothing_above
            self._accessible_mask[below_idx] = nothing_above
        else:
            rec.is_accessible = False
            self._accessible_mask[below_idx] = False
    
    # ========== State Export (for RL) ==========
    
    def get_occupancy_mask(self) -> np.ndarray:
        """Get occupancy mask for state encoding."""
        return self.occupancy_mask
    
    def get_accessibility_grid(self) -> np.ndarray:
        """
        Get accessibility as spatial grid.
        Returns (n_tiers, n_rows, total_splits) bool array.
        """
        grid = np.zeros_like(self.occupancy_mask)
        
        for cid, idx in self._id_to_idx.items():
            if self._accessible_mask[idx]:
                rec = self._records[idx]
                if rec:
                    t = rec.placement.tier
                    r = rec.placement.row
                    s = rec.placement.abs_start
                    grid[t, r, s:s + rec.n_splits] = True
        
        return grid
    
    def get_blocking_pairs(self) -> List[Tuple[str, str]]:
        """
        Get list of (blocker_id, blocked_id) pairs.
        Blocker is on top, blocked is below and less urgent.
        """
        pairs = []
        
        for cid, idx in self._id_to_idx.items():
            rec = self._records[idx]
            if rec is None or rec.placement.tier == 0:
                continue
            
            # Check below
            below_tier = rec.placement.tier - 1
            below_idx = self.position_grid[below_tier, rec.placement.row, rec.placement.abs_start]
            
            if below_idx != EMPTY_SLOT:
                below_rec: ContainerRecord = self._records[below_idx]
                if below_rec:
                    pairs.append((cid, below_rec.container.container_id))
        
        return pairs
    
    # ========== Debug ==========
    
    def _print_masks(self):
        """Print masks for validation."""
        np.set_printoptions(threshold=50, linewidth=200)
        print("=== Special Masks (Tier 0) ===")
        print(f"Reefer:\n{self.reefer_mask[0]}")
        print(f"Dangerous:\n{self.dangerous_mask[0]}")
        print(f"Swapbody:\n{self.swapbody_mask[0]}")
        print(f"Regular:\n{self.regular_mask[0]}")