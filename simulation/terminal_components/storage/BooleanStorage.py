import numpy as np
from typing import Dict, Tuple, List, Optional, Set
from dataclasses import dataclass
from simulation.terminal_components.storage_units.Container import Container
from simulation.terminal_components.storage.constants import (
    BAY_SPLIT_FACTOR, CONTAINER_LENGTH_TO_SUB_BAYS
)


@dataclass
class PlacementResult:
    """Result of container placement search."""
    row: int
    bay: int
    tier: int
    start_split: int
    score: float = 0.0


@dataclass
class ContainerRecord:
    """Complete record of a container and its placement."""
    container: Container
    placement: PlacementResult
    n_splits: int
    is_accessible: bool = True


class BooleanStorageYard:
    """
    Hybrid storage yard using both spatial masks and direct lookups.
    Optimized for both placement search and container operations.
    """
    
    def __init__(self, 
                 n_rows: int, 
                 n_bays: int, 
                 n_tiers: int,
                 coordinates: List[Tuple[int, int, str]],
                 validate: bool = False):
        """
        Initialize storage yard with hybrid data structures.
        
        Args:
            n_rows: Number of rows in yard
            n_bays: Number of bays (40ft units)
            n_tiers: Maximum stacking height
            coordinates: Special position definitions [(bay, row, type)]
            validate: Whether to print masks for validation
        """
        self.n_rows = n_rows
        self.n_bays = n_bays
        self.n_tiers = n_tiers
        self.split_factor = BAY_SPLIT_FACTOR
        self.total_splits = n_bays * self.split_factor
        
        # Hybrid data structures
        self.occupancy_mask = np.zeros((n_tiers, n_rows, self.total_splits), dtype=bool)
        self.containers: Dict[str, ContainerRecord] = {}
        self.position_map: Dict[Tuple[int, int, int], str] = {}  # (row, tier, start_split) -> container_id
        self.tier_containers: List[Set[str]] = [set() for _ in range(n_tiers)]
        self.accessible_containers: Set[str] = set()
        self._support_masks: Dict[tuple[int, int], np.ndarray] = {}

        # Special position masks
        self._init_special_masks(coordinates)
        
        # Container length mapping
        self.container_length_map = CONTAINER_LENGTH_TO_SUB_BAYS
        
        if validate:
            self._print_masks()
    
    def _init_special_masks(self, coordinates: List[Tuple[int, int, str]]):
        """Initialize special position masks."""
        self.reefer_mask = np.zeros((self.n_tiers, self.n_rows, self.total_splits), dtype=bool)
        self.dangerous_mask = np.zeros_like(self.reefer_mask)
        self.swapbody_mask = np.zeros_like(self.reefer_mask)
        
        for bay, row, position_type in coordinates:
            bay_idx = bay - 1
            row_idx = row - 1
            start_pos = bay_idx * self.split_factor
            end_pos = start_pos + self.split_factor
            
            if position_type == "r":
                self.reefer_mask[:, row_idx, start_pos:end_pos] = True
            elif position_type == "dg":
                self.dangerous_mask[:, row_idx, start_pos:end_pos] = True
            elif position_type == "sb_t":
                self.swapbody_mask[0, row_idx, start_pos:end_pos] = True
        
        self.regular_mask = ~(self.reefer_mask | self.dangerous_mask)
    
    def _get_goods_mask(self, container: Container, tier: int) -> np.ndarray:
        """Get appropriate mask based on container goods type."""
        if container.goods_type == "Reefer":
            return self.reefer_mask[tier]
        elif container.goods_type == "DangerousGoods":
            return self.dangerous_mask[tier]
        elif container.is_swap_body or container.is_trailer:
            return self.swapbody_mask[tier] if tier == 0 else np.zeros_like(self.swapbody_mask[0])
        return self.regular_mask[tier]
    
    def _get_support_mask(self, tier: int, n_splits: int) -> np.ndarray:
        if tier == 0:
            # Never used for base tier; keep shape for consistency
            if (tier, n_splits) not in self._support_masks:
                self._support_masks[(tier, n_splits)] = np.zeros((self.n_rows, self.total_splits), dtype=bool)
            return self._support_masks[(tier, n_splits)]
        key = (tier, n_splits)
        mask = self._support_masks.get(key)
        if mask is None:
            mask = np.zeros((self.n_rows, self.total_splits), dtype=bool)
            # cold-build once from current tier-1 containers
            for cid in self.tier_containers[tier - 1]:
                rec = self.containers.get(cid)
                if rec and rec.n_splits == n_splits:
                    r = rec.placement.row
                    s = rec.placement.bay * self.split_factor + rec.placement.start_split
                    e = s + n_splits
                    mask[r, s:e] = True
            self._support_masks[key] = mask
        return mask

    def _k(self, row: int, tier: int, abs_start: int) -> int:
        return (tier * self.n_rows + row) * self.total_splits + abs_start
    
    def _support_cache_apply(self, tier: int, n_splits: int, row: int, abs_start: int, abs_end: int, value: bool):
        # Update support for placements at 'tier' (supported by tier-1)
        if tier <= 0 or tier > self.n_tiers - 1:
            return
        key = (tier, n_splits)
        if key not in self._support_masks:
            self._support_masks[key] = np.zeros((self.n_rows, self.total_splits), dtype=bool)
        self._support_masks[key][row, abs_start:abs_end] = value

    def search_placement_all_tiers(
        self,
        container: "Container",
        target_bay: int,
        max_proximity: int = 3
    ) -> List["PlacementResult"]:
        """
        Find all valid placements near target_bay across all tiers.
        Vectorized across rows via a single 2D sliding-window sum per tier.
        """
        n_splits = self.container_length_map.get(container.length_ft, 0)
        if n_splits <= 0:
            return []

        # Clamp target window in split units (0-based bays)
        target_bay = max(0, min(target_bay, self.n_bays - 1))
        min_split = max(0, target_bay - max_proximity) * self.split_factor
        max_split_excl = min(self.n_bays, target_bay + max_proximity + 1) * self.split_factor
        # Note: stop_exclusive is the window edge; _find_runs_2d clamps to valid start indices internally.

        results: List["PlacementResult"] = []

        for tier in range(self.n_tiers):
            available = self._get_available_mask(container, tier)  # (n_rows, total_splits), bool

            rows, starts = self._find_runs_2d(
                available_brs=available,
                run_len=n_splits,
                start=min_split,
                stop_exclusive=max_split_excl
            )
            if rows.size == 0:
                continue

            bays = starts // self.split_factor
            start_splits = starts % self.split_factor
            center_bays = (starts + n_splits / 2.0) / self.split_factor
            scores = np.abs(center_bays - target_bay)

            # Build PlacementResult list
            for r, b, ss, sc in zip(rows.tolist(), bays.tolist(), start_splits.tolist(), scores.tolist()):
                results.append(
                    PlacementResult(row=int(r), bay=int(b), tier=int(tier), start_split=int(ss), score=float(sc))
                )

        results.sort(key=lambda p: (p.tier, p.score))
        return results
    
    def _get_available_mask(self, container: "Container", tier: int) -> np.ndarray:
        goods_mask = self._get_goods_mask(container, tier)
        unoccupied = ~self.occupancy_mask[tier]
        n_splits = self.container_length_map.get(container.length_ft, 0)
        if tier == 0:
            return goods_mask & unoccupied
        support = self._get_support_mask(tier, n_splits)
        return goods_mask & support & unoccupied
    
    def add_container(self, container: "Container", placement: "PlacementResult"):
        """
        Add container to yard.
        Accessibility: container is accessible iff there is nothing above its span.
        Updates below: the container directly beneath (same start, tier-1) becomes inaccessible.
        """
        n_splits = self.container_length_map.get(container.length_ft, 0)
        if n_splits <= 0:
            raise ValueError(f"Unsupported container length: {container.length_ft}")

        # Bounds
        if not (0 <= placement.row < self.n_rows and 0 <= placement.tier < self.n_tiers):
            raise IndexError("Placement out of yard bounds (row/tier).")

        abs_start = placement.bay * self.split_factor + placement.start_split
        abs_end = abs_start + n_splits
        if not (0 <= abs_start < self.total_splits) or abs_end > self.total_splits:
            raise IndexError("Placement out of yard bounds (splits).")

        # Vectorized occupancy set
        self.occupancy_mask[placement.tier, placement.row, abs_start:abs_end] = True

        # make starts above available for same-length placements
        self._support_cache_apply(placement.tier + 1, n_splits, placement.row, abs_start, abs_end, True)

        # Is new container top-of-its-stack?
        above = self.occupancy_mask[placement.tier + 1:, placement.row, abs_start:abs_end]
        is_free_above = not np.any(above)

        # Record and indices
        record = ContainerRecord(
            container=container,
            placement=placement,
            n_splits=n_splits,
            is_accessible=is_free_above
        )
        self.containers[container.container_id] = record
        self.position_map[(placement.row, placement.tier, abs_start)] = container.container_id
        self.tier_containers[placement.tier].add(container.container_id)

        if record.is_accessible:
            self.accessible_containers.add(container.container_id)
        else:
            self.accessible_containers.discard(container.container_id)

        # Under the same-length stacking rule, the only impacted below-container
        # is the one that starts exactly at abs_start on tier-1 (if any).
        if placement.tier > 0:
            self._update_accessibility_below(placement, make_accessible=False)
    
    def remove_container(self, container: "Container") -> Optional["Container"]:
        """
        Remove container from yard.
        - Clear occupancy.
        - Make the directly-below container (same start, tier-1) accessible iff nothing remains above it.
        """
        container_id = container.container_id
        record = self.containers.get(container_id)
        if record is None:
            return None

        # Use authoritative record
        row = record.placement.row
        tier = record.placement.tier
        abs_start = record.placement.bay * self.split_factor + record.placement.start_split
        abs_end = abs_start + record.n_splits

        # Clear occupancy first
        self.occupancy_mask[tier, row, abs_start:abs_end] = False

        # remove support above this container
        self._support_cache_apply(tier + 1, record.n_splits, row, abs_start, abs_end, False)

        # Update accessibility for directly-below container (if any)
        if tier > 0:
            self._update_accessibility_below(record.placement, make_accessible=True)

        # Remove indices
        self.tier_containers[tier].discard(container_id)
        self.accessible_containers.discard(container_id)
        self.position_map.pop((row, tier, abs_start), None)
        self.containers.pop(container_id, None)

        return record.container
    
    def move_container(self, container_id: str, destination: "PlacementResult") -> bool:
        """
        Minimal move: remove then add with no validity checks.
        Expects 'destination' to be a PlacementResult (e.g., from find_moveable_containers).

        Returns:
            True if the container existed and was moved, False if container_id not found.
        """
        rec = self.containers.get(container_id)
        if rec is None:
            return False

        # Remove from current placement
        self.remove_container(rec.container)

        # Add at destination
        self.add_container(rec.container, destination)
        return True

    def _update_accessibility_below(self, placement: "PlacementResult", make_accessible: bool):
        """
        Update accessibility for the container directly below the given placement:
        - same row, tier-1, and same start (aligned), if one exists.
        """
        if placement.tier == 0:
            return

        row = placement.row
        below_tier = placement.tier - 1
        abs_start = placement.bay * self.split_factor + placement.start_split

        cid_below = self.position_map.get((row, below_tier, abs_start))
        if not cid_below:
            return
        rec = self.containers.get(cid_below)
        if not rec:
            return

        s2 = abs_start
        e2 = s2 + rec.n_splits

        if make_accessible:
            # Accessible iff no occupancy on any higher tier over its span
            nothing_above = not np.any(self.occupancy_mask[below_tier + 1:, row, s2:e2])
            rec.is_accessible = bool(nothing_above)
            if rec.is_accessible:
                self.accessible_containers.add(cid_below)
            else:
                self.accessible_containers.discard(cid_below)
        else:
            # Adding a container above => not accessible
            rec.is_accessible = False
            self.accessible_containers.discard(cid_below)

    def _find_runs_2d(
        self,
        available_brs: np.ndarray,
        run_len: int,
        start: int,
        stop_exclusive: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Vectorized run detection across all rows using 2D cumsum/diff.
        Returns:
        - rows: 1D int array of row indices
        - starts: 1D int array of absolute start split indices (same length as rows)
        Only considers start positions in [start, stop_exclusive).
        """
        if run_len <= 0:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

        R, S = available_brs.shape
        valid_len = S - run_len + 1
        if valid_len <= 0:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

        # Clamp the start window to valid start indices
        lo = max(0, min(start, valid_len))
        hi = max(lo, min(stop_exclusive, valid_len))
        if lo >= hi:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

        # Slice only the needed region to reduce work
        # We need columns [lo : hi + run_len - 1] to compute windowed sums over starts in [lo, hi)
        end_col = hi + run_len - 1  # exclusive upper bound for the sliced columns
        sub = available_brs[:, lo:end_col].astype(np.int8, copy=False)  # [R, (hi-lo)+run_len-1]

        # 2D rolling sum along splits using cumsum/diff
        c = np.cumsum(sub, axis=1, dtype=np.int32)
        # Window sums at each start: sum over run_len = c[:, i+run_len-1] - c[:, i-1]
        prev = np.concatenate([np.zeros((R, 1), dtype=c.dtype), c[:, :-run_len]], axis=1)
        win = c[:, run_len - 1:] - prev  # shape [R, hi-lo]

        # Valid starts where the window sum equals run_len
        valid = (win == run_len)
        rows, starts_rel = np.where(valid)
        if rows.size == 0:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

        starts = starts_rel + lo
        return rows.astype(np.int32, copy=False), starts.astype(np.int32, copy=False)

    def find_moveable_containers(self, max_proximity: int = 5) -> Dict[str, List[PlacementResult]]:
        """
        Find all accessible containers and their possible destinations.
        
        Returns:
            Dict mapping container_id to list of possible placements
        """
        moveable = {}
        
        for container_id in self.accessible_containers:
            record = self.containers[container_id]
            current = record.placement
            
            # Find alternative placements
            destinations = self.search_placement_all_tiers(
                record.container, current.bay, max_proximity
            )
            
            # Filter out current position
            abs_current = current.bay * self.split_factor + current.start_split
            destinations = [
                d for d in destinations 
                if not (d.row == current.row and d.tier == current.tier and 
                       d.bay * self.split_factor + d.start_split == abs_current)
            ]
            
            if destinations:
                moveable[container_id] = destinations
        
        return moveable
    
    def get_container(self, container_id: str) -> Optional[Container]:
        """Get container by ID - O(1) operation."""
        record = self.containers.get(container_id)
        return record.container if record else None
    
    def get_container_placement(self, container_id: str) -> Optional[PlacementResult]:
        """Get container placement by ID - O(1) operation."""
        record = self.containers.get(container_id)
        return record.placement if record else None
    
    def get_tier_containers(self, tier: int) -> List[Container]:
        """Get all containers in a specific tier - O(tier_size) operation."""
        return [self.containers[cid].container 
                for cid in self.tier_containers[tier] 
                if cid in self.containers]
    
    def get_all_containers(self) -> List[Container]:
        """Get all containers - O(n) where n is number of containers."""
        return [record.container for record in self.containers.values()]
    
    def _print_masks(self):
        """Print masks for validation."""
        np.set_printoptions(threshold=50, linewidth=200)
        
        print("=== Special Masks (Tier 0) ===")
        print(f"Reefer positions:\n{self.reefer_mask[0]}")
        print(f"Dangerous goods positions:\n{self.dangerous_mask[0]}")
        print(f"Swap body positions:\n{self.swapbody_mask[0]}")
        print(f"Regular positions:\n{self.regular_mask[0]}")
        
        print("\n=== Occupancy (Tier 0) ===")
        print(f"Occupied positions:\n{self.occupancy_mask[0]}")