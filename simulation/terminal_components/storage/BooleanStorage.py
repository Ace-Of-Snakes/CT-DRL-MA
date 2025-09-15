import numpy as np
from typing import Dict, Tuple, List, Optional, Set
from dataclasses import dataclass
from simulation.terminal_components.storage_units.Container import Container
from simulation.terminal_components.storage.constants import (
    BAY_LENGTH_FT, BAY_SPLIT_FACTOR, SUB_BAY_LENGTH_FT,
    CONTAINER_LENGTHS_FT, CONTAINER_LENGTH_TO_SUB_BAYS
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
    
    def search_placement_all_tiers(self, 
                                   container: Container, 
                                   target_bay: int, 
                                   max_proximity: int = 3) -> List[PlacementResult]:
        """
        Search for valid placements across all tiers.
        
        Args:
            container: Container to place
            target_bay: Preferred bay index
            max_proximity: Maximum distance from target bay
            
        Returns:
            List of PlacementResult sorted by tier and proximity
        """
        n_splits = self.container_length_map.get(container.length_ft, 0)
        if n_splits == 0:
            return []
        
        min_split = max(0, target_bay - max_proximity) * self.split_factor
        max_split = min(self.n_bays, target_bay + max_proximity + 1) * self.split_factor
        
        all_placements = []
        
        for tier in range(self.n_tiers):
            # Get available positions mask
            available = self._get_available_mask(container, tier)
            
            # Find placements in this tier
            for row in range(self.n_rows):
                row_mask = available[row]
                
                for start_pos in range(min_split, min(max_split, self.total_splits - n_splits + 1)):
                    if np.all(row_mask[start_pos:start_pos + n_splits]):
                        bay = start_pos // self.split_factor
                        start_split = start_pos % self.split_factor
                        
                        # Score based on container center
                        center_bay = (start_pos + n_splits // 2) / self.split_factor
                        score = abs(center_bay - target_bay)
                        
                        all_placements.append(PlacementResult(
                            row=row, bay=bay, tier=tier, 
                            start_split=start_split, score=score
                        ))
        
        all_placements.sort(key=lambda p: (p.tier, p.score))
        return all_placements
    
    def _get_available_mask(self, container: Container, tier: int) -> np.ndarray:
        """Get combined availability mask for container at tier."""
        goods_mask = self._get_goods_mask(container, tier)
        
        # Check occupancy and support
        if tier == 0:
            return goods_mask & ~self.occupancy_mask[tier]
        else:
            # Must have support from below and be unoccupied
            has_support = self.occupancy_mask[tier - 1]
            return goods_mask & has_support & ~self.occupancy_mask[tier]
    
    def add_container(self, container: Container, placement: PlacementResult):
        """Add container to yard at specified placement."""
        n_splits = self.container_length_map.get(container.length_ft, 0)
        abs_start = placement.bay * self.split_factor + placement.start_split
        
        # Update occupancy mask
        for i in range(n_splits):
            pos = abs_start + i
            if pos < self.total_splits:
                self.occupancy_mask[placement.tier, placement.row, pos] = True
        
        # Create container record
        record = ContainerRecord(
            container=container,
            placement=placement,
            n_splits=n_splits,
            is_accessible=placement.tier == self.n_tiers - 1
        )
        
        # Update data structures
        self.containers[container.container_id] = record
        self.position_map[(placement.row, placement.tier, abs_start)] = container.container_id
        self.tier_containers[placement.tier].add(container.container_id)
        
        if record.is_accessible:
            self.accessible_containers.add(container.container_id)
        
        # Update accessibility of container below
        if placement.tier > 0:
            self._update_accessibility_below(placement, False)
    
    def remove_container(self, placement: PlacementResult, container: Container) -> Container:
        """Remove container from yard."""
        container_id = container.container_id
        if container_id not in self.containers:
            return None
        
        record = self.containers[container_id]
        abs_start = placement.bay * self.split_factor + placement.start_split
        
        # Update occupancy mask
        for i in range(record.n_splits):
            pos = abs_start + i
            if pos < self.total_splits:
                self.occupancy_mask[placement.tier, placement.row, pos] = False
        
        # Remove from data structures
        del self.containers[container_id]
        del self.position_map[(placement.row, placement.tier, abs_start)]
        self.tier_containers[placement.tier].discard(container_id)
        self.accessible_containers.discard(container_id)
        
        # Update accessibility of container below
        if placement.tier > 0:
            self._update_accessibility_below(placement, True)
        
        return record.container
    
    def _update_accessibility_below(self, placement: PlacementResult, make_accessible: bool):
        """Update accessibility of containers below the given placement."""
        if placement.tier == 0:
            return
        
        tier_below = placement.tier - 1
        abs_start = placement.bay * self.split_factor + placement.start_split
        
        # Check for containers at positions below
        containers_below = set()
        for pos in range(abs_start, abs_start + self.container_length_map.get(
            self.containers.get(list(self.position_map.values())[0]).container.length_ft 
            if self.position_map else 20, 1)):
            
            # Check all possible container starts that could cover this position
            for check_start in range(max(0, pos - 40), pos + 1):
                key = (placement.row, tier_below, check_start)
                if key in self.position_map:
                    containers_below.add(self.position_map[key])
        
        # Update accessibility
        for container_id in containers_below:
            if container_id in self.containers:
                record = self.containers[container_id]
                if make_accessible:
                    # Check if truly accessible (nothing else above)
                    abs_pos = record.placement.bay * self.split_factor + record.placement.start_split
                    nothing_above = not np.any(
                        self.occupancy_mask[tier_below + 1:, record.placement.row, 
                                          abs_pos:abs_pos + record.n_splits]
                    )
                    if nothing_above:
                        record.is_accessible = True
                        self.accessible_containers.add(container_id)
                else:
                    record.is_accessible = False
                    self.accessible_containers.discard(container_id)
    
    def find_moveable_containers(self, max_proximity: int = 2) -> Dict[str, List[PlacementResult]]:
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