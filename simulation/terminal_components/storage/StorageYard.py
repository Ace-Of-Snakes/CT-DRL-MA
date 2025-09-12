# simulation/terminal_components/storage/optimized_storage.py

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from simulation.terminal_components.storage_units.Container import Container
from simulation.terminal_components.storage.storage_constants import *
from simulation.terminal_components.storage.SlotMapper import ContainerSlotMapper

@dataclass
class StoragePosition:
    """Represents a container's position in the yard."""
    row: int
    bay: int
    tier: int
    start_split: int
    container_type: str
    
    def get_subslot_positions(self, mapper: ContainerSlotMapper) -> List[Tuple[int, int, int, int]]:
        """Get all (row, bay, tier, split) positions occupied."""
        positions = []
        occupied = mapper.get_occupied_subslots(self.container_type, self.bay, self.start_split)
        for bay, split in occupied:
            positions.append((self.row, bay, self.tier, split))
        return positions

class OptimizedStorageYard:
    """Optimized storage yard using bit manipulation and numpy operations."""
    
    def __init__(self, n_rows: int, n_bays: int, n_tiers: int,
                 special_areas: List[Tuple[int, int, str]]):
        """
        Initialize yard with special area designations.
        
        Args:
            n_rows: Number of rows in yard
            n_bays: Number of bays in yard
            n_tiers: Maximum stacking height
            special_areas: List of (bay, row, type) where type is 'r', 'dg', or 'sb_t'
                          Note: Uses 1-based indexing, converted internally to 0-based
        """
        self.n_rows = n_rows
        self.n_bays = n_bays
        self.n_tiers = n_tiers
        self.mapper = ContainerSlotMapper()
        
        # Container storage - sparse for memory efficiency
        self.containers: Dict[str, Tuple[Container, StoragePosition]] = {}
        
        # Occupancy tracking using uint64 for bit manipulation
        # Each uint64 represents 64 subslots (8 bays * 8 subslots/bay)
        bays_per_uint = 64 // YARD_SPLIT_FACTOR  # 8 bays per uint64
        n_uints = (n_bays + bays_per_uint - 1) // bays_per_uint
        
        # 3D occupancy array: [row, tier, uint64_index]
        self.occupancy = np.zeros((n_rows, n_tiers, n_uints), dtype=np.uint64)
        
        # Special area masks (2D: row x bay)
        self.reefer_mask = np.zeros((n_rows, n_bays), dtype=bool)
        self.dg_mask = np.zeros((n_rows, n_bays), dtype=bool)
        self.ground_only_mask = np.zeros((n_rows, n_bays), dtype=bool)
        
        self._init_special_areas(special_areas)
        
        # Stack accessibility tracking (which positions have containers on top)
        self.stack_blocked = np.zeros((n_rows, n_bays, n_tiers), dtype=bool)
        
    def _init_special_areas(self, special_areas: List[Tuple[int, int, str]]):
        """Initialize special area masks from coordinates."""
        for bay, row, area_type in special_areas:
            # Convert from 1-based to 0-based indexing
            bay_idx = bay - 1
            row_idx = row - 1
            
            if area_type == 'r':
                self.reefer_mask[row_idx, bay_idx] = True
            elif area_type == 'dg':
                self.dg_mask[row_idx, bay_idx] = True
            elif area_type == 'sb_t':
                self.ground_only_mask[row_idx, bay_idx] = True
    
    def _get_occupancy_bits(self, row: int, tier: int, bay: int) -> int:
        """Get occupancy bits for a specific bay."""
        uint_idx = bay // 8
        bay_offset = (bay % 8) * YARD_SPLIT_FACTOR
        
        if uint_idx >= self.occupancy.shape[2]:
            return (1 << YARD_SPLIT_FACTOR) - 1  # All occupied if out of bounds
        
        bits = (self.occupancy[row, tier, uint_idx] >> bay_offset) & ((1 << YARD_SPLIT_FACTOR) - 1)
        return int(bits)
    
    def _set_occupancy_bits(self, row: int, tier: int, bay: int, start_split: int, 
                           length: int, occupied: bool):
        """Set occupancy bits for container placement."""
        positions = []
        for i in range(length):
            current_bay = bay + ((start_split + i) // YARD_SPLIT_FACTOR)
            current_split = (start_split + i) % YARD_SPLIT_FACTOR
            
            uint_idx = current_bay // 8
            bay_offset = (current_bay % 8) * YARD_SPLIT_FACTOR + current_split
            
            if uint_idx < self.occupancy.shape[2]:
                if occupied:
                    self.occupancy[row, tier, uint_idx] |= np.uint64(1 << bay_offset)
                else:
                    # Fix: Use numpy's bitwise NOT to stay within uint64 bounds
                    mask = np.uint64(1 << bay_offset)
                    self.occupancy[row, tier, uint_idx] &= np.uint64(~mask)
    
    def _is_position_accessible(self, row: int, bay: int, tier: int) -> bool:
        """Check if position is accessible (nothing above it)."""
        if tier >= self.n_tiers - 1:
            return True  # Top tier always accessible
        
        # Check if any tier above has a container
        return not np.any(self.stack_blocked[row, bay, tier+1:])
    
    def _update_stack_blocking(self, row: int, bay: int, tier: int, 
                              start_split: int, container_type: str, adding: bool):
        """Update stack blocking when adding/removing containers."""
        subslots = self.mapper.get_subslots_needed(container_type)
        occupied_positions = self.mapper.get_occupied_subslots(container_type, bay, start_split)
        
        for bay_pos, _ in occupied_positions:
            if bay_pos < self.n_bays:
                if adding:
                    # Block all tiers below
                    if tier > 0:
                        self.stack_blocked[row, bay_pos, :tier] = True
                else:
                    # Check if this bay position is still blocked by other containers
                    still_blocked = False
                    for check_tier in range(tier + 1, self.n_tiers):
                        if self._get_occupancy_bits(row, check_tier, bay_pos) != 0:
                            still_blocked = True
                            break
                    
                    if not still_blocked and tier > 0:
                        self.stack_blocked[row, bay_pos, :tier] = False
    
    def _check_placement_rules(self, container: Container, row: int, bay: int, tier: int) -> bool:
        """Check if container can be placed according to type rules."""
        # Ground-only containers
        if container.container_type in EXCLUSIVE_TYPES:
            if tier != 0:
                return False
            if not self.ground_only_mask[row, bay]:
                return False
        
        # Check goods type compatibility with area
        if container.goods_type == GOODS_REEFER:
            if not self.reefer_mask[row, bay]:
                return False
        elif container.goods_type == GOODS_DANGEROUS:
            if not self.dg_mask[row, bay]:
                return False
        else:  # Regular
            # Regular can go on regular or reefer slots
            if self.dg_mask[row, bay] or self.ground_only_mask[row, bay]:
                if not self.reefer_mask[row, bay]:  # Unless it's also a reefer slot
                    return False
        
        # Check stacking rules - containers below must be same type
        if tier > 0:
            # Get container below
            below_bits = self._get_occupancy_bits(row, tier - 1, bay)
            if below_bits == 0:
                return False  # Nothing below to stack on
            
            # Check if container types match (simplified - would need actual container lookup)
            # This is where we'd verify same goods type stacking
        
        return True
    
    def add_container(self, container: Container, row: int, bay: int, tier: int, start_split: int) -> bool:
        """
        Add container to yard.
        
        Returns:
            True if successfully added, False otherwise
        """
        if container.container_id in self.containers:
            return False
        
        # Validate placement
        if not self.mapper.validate_placement(container.container_type, bay, start_split, self.n_bays):
            return False
        
        if not self._check_placement_rules(container, row, bay, tier):
            return False
        
        # Check if position is free
        subslots = self.mapper.get_subslots_needed(container.container_type)
        for i in range(subslots):
            check_bay = bay + ((start_split + i) // YARD_SPLIT_FACTOR)
            check_split = (start_split + i) % YARD_SPLIT_FACTOR
            
            if check_bay >= self.n_bays:
                return False
            
            bits = self._get_occupancy_bits(row, tier, check_bay)
            if bits & (1 << check_split):
                return False  # Position occupied
        
        # Add container
        position = StoragePosition(row, bay, tier, start_split, container.container_type)
        self.containers[container.container_id] = (container, position)
        
        # Update occupancy
        self._set_occupancy_bits(row, tier, bay, start_split, subslots, True)
        
        # Update stack blocking
        self._update_stack_blocking(row, bay, tier, start_split, container.container_type, True)
        
        return True
    
    def remove_container(self, container_id: str) -> Optional[Container]:
        """
        Remove container from yard if accessible.
        
        Returns:
            Container if removed, None if not found or not accessible
        """
        if container_id not in self.containers:
            return None
        
        container, position = self.containers[container_id]
        
        # Check accessibility
        if not self._is_position_accessible(position.row, position.bay, position.tier):
            return None  # Container blocked by others above
        
        # Remove container
        del self.containers[container_id]
        
        # Update occupancy
        subslots = self.mapper.get_subslots_needed(position.container_type)
        self._set_occupancy_bits(position.row, position.tier, position.bay, 
                                position.start_split, subslots, False)
        
        # Update stack blocking
        self._update_stack_blocking(position.row, position.bay, position.tier, 
                                   position.start_split, position.container_type, False)
        
        return container
    
    def find_placement_positions(self, container: Container, near_bay: int, 
                                max_distance: int = 3) -> List[Tuple[int, int, int, int]]:
        """
        Find valid placement positions for container near specified bay.
        
        Returns:
            List of (row, bay, tier, start_split) tuples
        """
        valid_positions = []
        valid_starts = self.mapper.get_valid_start_positions(container.container_type)
        subslots = self.mapper.get_subslots_needed(container.container_type)
        
        # Search range
        min_bay = max(0, near_bay - max_distance)
        max_bay = min(self.n_bays, near_bay + max_distance + 1)
        
        # Use numpy to find candidate positions efficiently
        for bay in range(min_bay, max_bay):
            for row in range(self.n_rows):
                # Check placement rules for this row/bay
                if not self._check_placement_rules(container, row, bay, 0):
                    continue
                
                # Check each tier
                for tier in range(self.n_tiers):
                    if container.container_type in EXCLUSIVE_TYPES and tier > 0:
                        continue  # Ground only
                    
                    # Check each valid start position
                    for start_split in valid_starts:
                        if not self.mapper.validate_placement(container.container_type, bay, start_split, self.n_bays):
                            continue
                        
                        # Check if all required positions are free
                        all_free = True
                        for i in range(subslots):
                            check_bay = bay + ((start_split + i) // YARD_SPLIT_FACTOR)
                            check_split = (start_split + i) % YARD_SPLIT_FACTOR
                            
                            if check_bay >= self.n_bays:
                                all_free = False
                                break
                            
                            bits = self._get_occupancy_bits(row, tier, check_bay)
                            if bits & (1 << check_split):
                                all_free = False
                                break
                        
                        if all_free:
                            valid_positions.append((row, bay, tier, start_split))
        
        # Sort by distance from target bay, then by tier (prefer ground level)
        valid_positions.sort(key=lambda x: (abs(x[1] - near_bay), x[2]))
        
        return valid_positions
    
    def get_moveable_containers(self) -> List[str]:
        """Get IDs of all containers that can be moved (accessible)."""
        moveable = []
        
        for container_id, (container, position) in self.containers.items():
            if self._is_position_accessible(position.row, position.bay, position.tier):
                moveable.append(container_id)
        
        return moveable
    
    def get_yard_moves(self, max_distance: int = 2) -> Dict[str, List[Tuple[int, int, int, int]]]:
        """
        Get all possible moves for accessible containers.
        
        Returns:
            Dict mapping container_id to list of valid destination positions
        """
        moves = {}
        
        for container_id in self.get_moveable_containers():
            container, position = self.containers[container_id]
            
            # Find alternative positions
            destinations = self.find_placement_positions(
                container, position.bay, max_distance
            )
            
            # Filter out current position
            destinations = [
                (r, b, t, s) for r, b, t, s in destinations
                if not (r == position.row and b == position.bay and 
                       t == position.tier and s == position.start_split)
            ]
            
            if destinations:
                moves[container_id] = destinations
        
        return moves