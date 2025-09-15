import numpy as np
from typing import Dict, Tuple, List, Optional, Set
from collections import defaultdict
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
    score: float = 0.0  # For prioritization


class BooleanStorageYard:
    """
    Optimized storage yard with dynamic container lengths and multi-tier support.
    Uses vectorized operations and dynamic space finding for maximum flexibility.
    """
    
    def __init__(self, 
                 n_rows: int, 
                 n_bays: int, 
                 n_tiers: int,
                 coordinates: List[Tuple[int, int, str]],
                 validate: bool = False):
        """
        Initialize storage yard with dynamic container length support.
        
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
        
        # Total splits across all bays
        self.total_splits = n_bays * self.split_factor
        
        # Container storage array
        self.containers = np.full((n_rows, n_bays, n_tiers, self.split_factor), 
                                  None, dtype=object)
        
        # Create tier-aware dynamic masks for each container length
        self._init_dynamic_masks()
        
        # Create special position masks (reefer, dangerous goods, swap body/trailer)
        self._init_special_masks(coordinates)
        
        # Container length mapping from constants 
        self.container_length_map = CONTAINER_LENGTH_TO_SUB_BAYS
        
        if validate:
            self._print_masks()
    
    def _init_dynamic_masks(self):
        """Initialize dynamic occupancy masks for each tier and container length."""
        # Base occupancy mask: (n_tiers, n_rows, n_bays * split_factor)
        self.base_mask = np.zeros((self.n_tiers, self.n_rows, self.total_splits), 
                                   dtype=bool)
        
        # Only ground tier is initially available
        self.base_mask[0, :, :] = True
        
        # Dynamic masks for each container length at each tier
        # Key: (length_ft, tier) -> mask
        self.length_tier_masks = {}
        
        for length_ft in CONTAINER_LENGTHS_FT:
            for tier in range(self.n_tiers):
                # Initially copy base mask for this tier
                mask = self.base_mask[tier].copy()
                self.length_tier_masks[(length_ft, tier)] = mask
    
    def _init_special_masks(self, coordinates: List[Tuple[int, int, str]]):
        """Initialize special position masks for reefer, dangerous goods, and swap bodies."""
        # Shape: (n_tiers, n_rows, n_bays * split_factor)
        self.reefer_mask = np.zeros((self.n_tiers, self.n_rows, self.total_splits), 
                                     dtype=bool)
        self.dangerous_mask = np.zeros_like(self.reefer_mask)
        self.swapbody_mask = np.zeros_like(self.reefer_mask)
        
        for bay, row, position_type in coordinates:
            # Convert to 0-indexed
            bay_idx = bay - 1
            row_idx = row - 1
            
            # Calculate position range in flattened array
            start_pos = bay_idx * self.split_factor
            end_pos = start_pos + self.split_factor
            
            if position_type == "r":  # Reefer
                self.reefer_mask[:, row_idx, start_pos:end_pos] = True
            elif position_type == "dg":  # Dangerous goods
                self.dangerous_mask[:, row_idx, start_pos:end_pos] = True
            elif position_type == "sb_t":  # Swap body/trailer
                # Only ground tier for swap bodies
                self.swapbody_mask[0, row_idx, start_pos:end_pos] = True
        
        # Regular container mask (everything except special positions)
        self.regular_mask = ~(self.reefer_mask | self.dangerous_mask)
    
    def _get_container_length_ft(self, container: Container) -> int:
        """Extract container length in feet from container object."""
        return container.length_ft
    
    def _get_goods_mask(self, container: Container, tier: int) -> np.ndarray:
        """Get appropriate mask based on container goods type."""
        goods_type = container.goods_type
        
        if goods_type == "Reefer":
            return self.reefer_mask[tier]
        elif goods_type == "DangerousGoods":
            return self.dangerous_mask[tier]
        elif container.is_swap_body or container.is_trailer:
            if tier > 0:
                return np.zeros_like(self.swapbody_mask[0])  # Not stackable
            return self.swapbody_mask[tier]
        else:  # Regular
            return self.regular_mask[tier]
    
    def search_placement_all_tiers(self, 
                                   container: Container, 
                                   target_bay: int, 
                                   max_proximity: int = 3) -> List[PlacementResult]:
        """
        Search for valid placements across all tiers using dynamic space finding.
        
        Args:
            container: Container to place
            target_bay: Preferred bay index
            max_proximity: Maximum distance from target bay
            
        Returns:
            List of PlacementResult sorted by tier (ground first) and proximity
        """
        length_ft = self._get_container_length_ft(container)
        n_splits = self.container_length_map.get(length_ft, 0)
        
        if n_splits == 0:
            return []
        
        # Calculate bay range for proximity
        min_bay = max(0, target_bay - max_proximity)
        max_bay = min(self.n_bays, target_bay + max_proximity + 1)
        
        # Convert to split positions
        min_split = min_bay * self.split_factor
        max_split = max_bay * self.split_factor
        
        all_placements = []
        
        # Search all tiers
        for tier in range(self.n_tiers):
            # Get combined mask for this tier
            goods_mask = self._get_goods_mask(container, tier)
            dynamic_mask = self.length_tier_masks[(length_ft, tier)]
            
            # Combine masks
            available = goods_mask & dynamic_mask
            
            # Find valid placements for this tier
            tier_placements = self._find_placements_in_tier(
                available, tier, n_splits, min_split, max_split, target_bay
            )
            all_placements.extend(tier_placements)
        
        # Sort by tier (ground first) then by proximity score
        all_placements.sort(key=lambda p: (p.tier, p.score))
        
        return all_placements
    
    def _find_placements_in_tier(self, 
                                 available_mask: np.ndarray, 
                                 tier: int, 
                                 n_splits: int,
                                 min_split: int,
                                 max_split: int,
                                 target_bay: int) -> List[PlacementResult]:
        """
        Find all valid placements within a single tier using sliding window.
        This method finds all consecutive spaces where a container fits,
        regardless of bay boundaries.
        """
        placements = []
        
        # Process each row
        for row in range(self.n_rows):
            row_mask = available_mask[row]
            
            # Use sliding window to find all consecutive available spaces
            # that can accommodate the container
            for start_pos in range(min_split, min(max_split, self.total_splits - n_splits + 1)):
                # Check if all required positions are available
                if np.all(row_mask[start_pos:start_pos + n_splits]):
                    # Convert back to bay and split coordinates
                    bay = start_pos // self.split_factor
                    start_split = start_pos % self.split_factor
                    
                    # Calculate proximity score based on the center of the container
                    container_center_bay = (start_pos + n_splits // 2) / self.split_factor
                    score = abs(container_center_bay - target_bay)
                    
                    placements.append(PlacementResult(
                        row=row, 
                        bay=bay, 
                        tier=tier, 
                        start_split=start_split, 
                        score=score
                    ))
        
        return placements
    
    def add_container(self, container: Container, placement: PlacementResult):
        """
        Add container to yard at specified placement.
        
        Args:
            container: Container to add
            placement: PlacementResult from search
        """
        length_ft = self._get_container_length_ft(container)
        n_splits = self.container_length_map.get(length_ft, 0)
        
        # Calculate absolute starting position
        abs_start = placement.bay * self.split_factor + placement.start_split
        
        # Update container array
        for i in range(n_splits):
            abs_pos = abs_start + i
            bay = abs_pos // self.split_factor
            split = abs_pos % self.split_factor
            
            if bay < self.n_bays:
                self.containers[placement.row, bay, placement.tier, split] = container
        
        # Update masks
        self._update_masks_on_add(placement, n_splits, length_ft)
    
    def remove_container(self, placement: PlacementResult, container: Container) -> Container:
        """
        Remove container from yard.
        
        Args:
            placement: PlacementResult indicating position
            container: Container being removed (for length info)
            
        Returns:
            Removed container
        """
        length_ft = self._get_container_length_ft(container)
        n_splits = self.container_length_map.get(length_ft, 0)
        
        # Calculate absolute starting position
        abs_start = placement.bay * self.split_factor + placement.start_split
        
        # Remove from container array
        removed = None
        for i in range(n_splits):
            abs_pos = abs_start + i
            bay = abs_pos // self.split_factor
            split = abs_pos % self.split_factor
            
            if bay < self.n_bays:
                if removed is None:
                    removed = self.containers[placement.row, bay, placement.tier, split]
                self.containers[placement.row, bay, placement.tier, split] = None
        
        # Update masks
        self._update_masks_on_remove(placement, n_splits, length_ft)
        
        return removed
    
    def _update_masks_on_add(self, placement: PlacementResult, n_splits: int, length_ft: int):
        """Update dynamic masks when container is added."""
        row = placement.row
        tier = placement.tier
        
        # Calculate absolute starting position
        abs_start = placement.bay * self.split_factor + placement.start_split
        
        # Mark positions as occupied for all container lengths at this tier
        for i in range(n_splits):
            pos = abs_start + i
            if pos < self.total_splits:
                for other_length in CONTAINER_LENGTHS_FT:
                    self.length_tier_masks[(other_length, tier)][row, pos] = False
        
        # Enable next tier if applicable
        if tier < self.n_tiers - 1:
            for i in range(n_splits):
                pos = abs_start + i
                if pos < self.total_splits:
                    self.base_mask[tier + 1, row, pos] = True
                    for other_length in CONTAINER_LENGTHS_FT:
                        self.length_tier_masks[(other_length, tier + 1)][row, pos] = True
    
    def _update_masks_on_remove(self, placement: PlacementResult, n_splits: int, length_ft: int):
        """Update dynamic masks when container is removed."""
        row = placement.row
        tier = placement.tier
        
        # Calculate absolute starting position
        abs_start = placement.bay * self.split_factor + placement.start_split
        
        # Mark positions as available
        for i in range(n_splits):
            pos = abs_start + i
            if pos < self.total_splits:
                for other_length in CONTAINER_LENGTHS_FT:
                    self.length_tier_masks[(other_length, tier)][row, pos] = True
        
        # Disable next tier if it becomes unsupported
        if tier < self.n_tiers - 1:
            for i in range(n_splits):
                pos = abs_start + i
                if pos < self.total_splits:
                    # Check if position has support from below
                    bay = pos // self.split_factor
                    split = pos % self.split_factor
                    if self.containers[row, bay, tier, split] is None:
                        self.base_mask[tier + 1, row, pos] = False
                        for other_length in CONTAINER_LENGTHS_FT:
                            self.length_tier_masks[(other_length, tier + 1)][row, pos] = False
    
    def find_moveable_containers(self, max_proximity: int = 2) -> Dict[str, List[PlacementResult]]:
        """
        Find all containers that can be moved and their possible destinations.
        
        Returns:
            Dict mapping container_id to list of possible placements
        """
        moveable = {}
        
        # Find accessible containers (top of stacks or ground level non-stackables)
        for row in range(self.n_rows):
            for bay in range(self.n_bays):
                for tier in range(self.n_tiers):
                    for split in range(self.split_factor):
                        container = self.containers[row, bay, tier, split]
                        
                        if container is None:
                            continue
                        
                        # Check if accessible (nothing above or special type on ground)
                        is_accessible = tier == self.n_tiers - 1  # Top tier
                        
                        if not is_accessible and tier < self.n_tiers - 1:
                            # Check if nothing above
                            nothing_above = all(
                                self.containers[row, bay, t, split] is None 
                                for t in range(tier + 1, self.n_tiers)
                            )
                            is_accessible = nothing_above
                        
                        if is_accessible and container.container_id not in moveable:
                            # Find alternative placements
                            current_placement = PlacementResult(row, bay, tier, split)
                            destinations = self.search_placement_all_tiers(
                                container, bay, max_proximity
                            )
                            
                            # Filter out current position
                            destinations = [
                                d for d in destinations 
                                if not (d.row == row and d.bay == bay and 
                                       d.tier == tier and d.start_split == split)
                            ]
                            
                            if destinations:
                                moveable[container.container_id] = destinations
        
        return moveable
    
    def _print_masks(self):
        """Print masks for validation."""
        np.set_printoptions(threshold=50, linewidth=200)
        
        print("=== Special Masks (Tier 0) ===")
        print(f"Reefer positions:\n{self.reefer_mask[0]}")
        print(f"Dangerous goods positions:\n{self.dangerous_mask[0]}")
        print(f"Swap body positions:\n{self.swapbody_mask[0]}")
        print(f"Regular positions:\n{self.regular_mask[0]}")
        
        print("\n=== Dynamic Masks Sample ===")
        print(f"20ft container tier 0:\n{self.length_tier_masks[(20, 0)]}")
        print(f"40ft container tier 0:\n{self.length_tier_masks[(40, 0)]}")