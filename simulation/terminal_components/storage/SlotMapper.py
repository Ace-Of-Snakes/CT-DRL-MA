# simulation/terminal_components/storage/slot_mapper.py

import numpy as np
from typing import Dict, List, Tuple, Optional
from simulation.terminal_components.storage.storage_constants import *

class ContainerSlotMapper:
    """Maps container types to sub-slot requirements and validates placements."""
    
    def __init__(self):
        self.container_subslots = CONTAINER_SUBSLOTS
        self.split_factor = YARD_SPLIT_FACTOR
        
    def get_subslots_needed(self, container_type: str) -> int:
        """Get number of sub-slots needed for container type."""
        return self.container_subslots.get(container_type, YARD_SPLIT_FACTOR)
    
    def get_valid_start_positions(self, container_type: str) -> List[int]:
        """
        Get valid starting positions based on container length and placement rules.
        
        Rules:
        - Containers < 40ft: Can start at 0 or align to end of bay
        - 40ft containers: Must start at position 0 (full bay)
        - 45ft containers: Can start at 0 or end at position 8 (cross-bay)
        """
        subslots = self.get_subslots_needed(container_type)
        
        if subslots >= CROSS_BAY_MIN_LENGTH:  # 45ft container
            # Can start at beginning or align to end of next bay
            return [0, YARD_SPLIT_FACTOR - (subslots - YARD_SPLIT_FACTOR)]
        elif subslots == FULL_BAY_LENGTH:  # 40ft container
            return [0]  # Must use full bay
        else:  # Smaller containers
            # Can start at beginning or align to end
            start_pos = 0
            end_pos = YARD_SPLIT_FACTOR - subslots
            return [start_pos, end_pos] if start_pos != end_pos else [start_pos]
    
    def validate_placement(self, container_type: str, bay: int, start_split: int, 
                          n_bays: int) -> bool:
        """
        Validate if a container can be placed at the given position.
        
        Returns:
            True if placement is valid according to all rules
        """
        subslots = self.get_subslots_needed(container_type)
        valid_starts = self.get_valid_start_positions(container_type)
        
        # Check if start position is valid
        if start_split not in valid_starts:
            return False
        
        # Check bay boundaries
        if subslots >= CROSS_BAY_MIN_LENGTH:  # Cross-bay container
            end_bay = bay + ((start_split + subslots - 1) // YARD_SPLIT_FACTOR)
            if end_bay >= n_bays:
                return False
        else:  # Regular container - cannot cross bay boundary
            if start_split + subslots > YARD_SPLIT_FACTOR:
                return False
        
        return True
    
    def get_occupied_subslots(self, container_type: str, bay: int, 
                             start_split: int) -> List[Tuple[int, int]]:
        """
        Get all (bay, subslot) positions occupied by a container.
        
        Returns:
            List of (bay, subslot) tuples
        """
        subslots = self.get_subslots_needed(container_type)
        positions = []
        
        for i in range(subslots):
            current_bay = bay + ((start_split + i) // YARD_SPLIT_FACTOR)
            current_split = (start_split + i) % YARD_SPLIT_FACTOR
            positions.append((current_bay, current_split))
        
        return positions