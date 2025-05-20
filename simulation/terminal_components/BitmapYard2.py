import numpy as np
import torch
import time
from typing import Dict, List, Tuple, Optional, Any
import re


class EnhancedBitmapYard:
    """
    GPU-accelerated bitmap representation of a container storage yard.
    Uses bit operations for efficient container placement and movement calculations.
    
    Enhanced to handle container sizes using 10ft subslots (4 subslots per standard bay).
    """
    
    def __init__(self, 
                num_rows: int, 
                num_bays: int, 
                max_tier_height: int = 5,
                row_names: List[str] = None,
                special_areas: Dict[str, List[Tuple[str, int, int]]] = None,
                subslots_per_bay: int = 4,
                device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the enhanced bitmap storage yard with subslot handling.
        
        Args:
            num_rows: Number of rows in the yard
            num_bays: Number of bays per row
            max_tier_height: Maximum stacking height
            row_names: Names for each row (defaults to A, B, C...)
            special_areas: Dictionary mapping special types to areas
            subslots_per_bay: Number of subslots per bay (default 4 for 10ft subslots)
            device: Computation device ('cuda' or 'cpu')
        """
        self.num_rows = num_rows
        self.num_bays = num_bays
        self.max_tier_height = max_tier_height
        self.device = device
        self.subslots_per_bay = subslots_per_bay
        
        # Initialize row names if not provided
        if row_names is None:
            self.row_names = [chr(65 + i) for i in range(num_rows)]  # A, B, C, ...
        else:
            self.row_names = row_names[:num_rows]
        
        # Calculate bitmap dimensions with subslots
        # Each bay now has subslots_per_bay positions
        self.total_subslots = num_bays * subslots_per_bay
        
        # Calculate bits per row with 64-bit alignment
        # This ensures we have enough words to hold all subslots
        self.bits_per_row = ((self.total_subslots + 63) // 64) * 64
        self.total_bits = self.num_rows * self.bits_per_row
        
        # Get number of 64-bit words needed for each bitmap
        self.num_words = (self.total_bits + 63) // 64
        
        print(f"Initializing bitmap with dimensions:")
        print(f"- Rows: {self.num_rows}, Bays: {self.num_bays}, Subslots per bay: {self.subslots_per_bay}")
        print(f"- Total subslots: {self.total_subslots}, Bits per row: {self.bits_per_row}")
        print(f"- Total bits: {self.total_bits}, Number of 64-bit words: {self.num_words}")
        
        # Special areas for different container types
        if special_areas is None:
            # Default special areas
            self.special_areas = {
                'reefer': [('A', 1, 5)],  # Row A, bays 1-5
                'dangerous': [('C', 6, 10)],  # Row C, bays 6-10
                'trailer': [('A', 15, 25)],  # Row A, bays 15-25
                'swap_body': [('B', 30, 40)]  # Row B, bays 30-40
            }
        else:
            self.special_areas = special_areas
        
        # Initialize bitmaps for each tier (using PyTorch for GPU compatibility)
        # Each tier gets its own bitmap; bit=1 means occupied, bit=0 means free
        self.tier_bitmaps = [torch.zeros(self.num_words, dtype=torch.int64, device=device) 
                            for _ in range(max_tier_height)]
        
        # Initialize bitmaps for special areas
        self.special_area_bitmaps = {
            'reefer': torch.zeros(self.num_words, dtype=torch.int64, device=device),
            'dangerous': torch.zeros(self.num_words, dtype=torch.int64, device=device),
            'trailer': torch.zeros(self.num_words, dtype=torch.int64, device=device),
            'swap_body': torch.zeros(self.num_words, dtype=torch.int64, device=device)
        }
        
        # Container length mapping (in terms of 10ft subslots)
        self.container_length_subslots = {
            'TWEU': 2,      # 20ft = 2 subslots
            'THEU': 3,      # 30ft = 3 subslots 
            'FEU': 4,       # 40ft = 4 subslots
            'FFEU': 5,      # 45ft = 5 subslots
            'Trailer': 4,   # Trailer = 4 subslots
            'Swap Body': 3, # Swap body = 3 subslots
            'default': 2    # Default to 20ft (2 subslots)
        }
        
        # Initialize position mapping with subslots
        self.position_to_bit = {}  # Maps position string to bit index
        self.bit_to_position = {}  # Maps bit index to position string
        self._build_position_mapping()
        
        # Initialize special area bitmaps with subslot resolution
        self._initialize_special_area_bitmaps()
        
        # Container registry for efficient lookup
        self.container_registry = {}  # position -> container object
        
        # Track container occupation with subslot mapping
        self.container_occupation = {}  # container_id -> list of subslot positions
        
        # Pre-compute proximity masks for efficient proximity searching
        self.proximity_masks = {}  # (bit_idx, n) -> proximity mask
        
        # Create an "occupied" bitmap that combines all tiers for quick checks
        self.update_occupied_bitmap()
        
        # Generate container placement masks
        self._generate_container_placement_masks()
    
    def _build_position_mapping(self):
        """
        Build mapping between position strings and bit indices.
        Positions now include subslot notation: 'A1.1', 'A1.2', etc.
        """
        for row_idx, row in enumerate(self.row_names):
            row_offset = row_idx * self.bits_per_row
            for bay in range(1, self.num_bays + 1):
                for subslot in range(1, self.subslots_per_bay + 1):
                    # Position format: 'A1.1', 'A1.2', etc.
                    position = f"{row}{bay}.{subslot}"
                    
                    # Calculate bit index
                    # Convert from 1-based to 0-based indexing
                    bay_idx = bay - 1
                    subslot_idx = subslot - 1
                    
                    # Overall subslot index within the row
                    overall_subslot_idx = bay_idx * self.subslots_per_bay + subslot_idx
                    
                    # Skip if it would exceed our bitmap size
                    if overall_subslot_idx >= self.bits_per_row:
                        continue
                    
                    # Final bit index
                    bit_idx = row_offset + overall_subslot_idx
                    
                    # Skip if it would exceed total bits
                    if bit_idx >= self.total_bits:
                        continue
                    
                    # Store mappings
                    self.position_to_bit[position] = bit_idx
                    self.bit_to_position[bit_idx] = position
    
    def _initialize_special_area_bitmaps(self):
        """
        Initialize bitmaps for special areas based on configuration.
        Handles the conversion from bay-level to subslot-level areas.
        """
        # Initialize special area bitmaps
        for area_type, areas in self.special_areas.items():
            bitmap = self.special_area_bitmaps[area_type]
            for area_row, start_bay, end_bay in areas:
                if area_row in self.row_names:
                    row_idx = self.row_names.index(area_row)
                    row_offset = row_idx * self.bits_per_row
                    
                    # Validate and limit bay range if needed
                    start_bay = max(1, min(start_bay, self.num_bays))
                    end_bay = max(1, min(end_bay, self.num_bays))
                    
                    # Convert bay range to subslot range
                    start_subslot = (start_bay - 1) * self.subslots_per_bay
                    end_subslot = (end_bay * self.subslots_per_bay) - 1
                    
                    # Ensure we don't exceed the total subslots
                    end_subslot = min(end_subslot, self.total_subslots - 1)
                    
                    for subslot_idx in range(start_subslot, end_subslot + 1):
                        if 0 <= subslot_idx < self.total_subslots:
                            bit_idx = row_offset + subslot_idx
                            
                            # Ensure we don't exceed the total bits
                            if bit_idx < self.total_bits:
                                word_idx = bit_idx // 64
                                bit_offset = bit_idx % 64
                                
                                # Safety check to prevent overflow
                                if 0 <= bit_offset < 64 and word_idx < len(bitmap):
                                    bitmap[word_idx] |= (1 << bit_offset)
    
    def _generate_container_placement_masks(self):
        """
        Generate specialized masks for each container type's valid placement positions.
        These are precomputed to speed up placement operations.
        """
        # Create a mask for each container type
        self.container_placement_masks = {}
        
        # Get all possible container types from length mapping
        for container_type, length in self.container_length_subslots.items():
            # Create an empty mask
            mask = torch.zeros(self.num_words, dtype=torch.int64, device=self.device)
            
            # For each row
            for row_idx, row in enumerate(self.row_names):
                row_offset = row_idx * self.bits_per_row
                
                # For each bay
                for bay_idx in range(self.num_bays):
                    # For each potential starting subslot position
                    for start_subslot in range(self.subslots_per_bay):
                        # Check if there are enough subslots remaining in this bay
                        remaining_subslots = self.subslots_per_bay - start_subslot
                        
                        if length <= remaining_subslots:
                            # This is a valid starting position for this container type
                            # Calculate the bit index
                            subslot_idx = bay_idx * self.subslots_per_bay + start_subslot
                            bit_idx = row_offset + subslot_idx
                            
                            # Skip if out of range
                            if bit_idx >= self.total_bits:
                                continue
                                
                            word_idx = bit_idx // 64
                            bit_offset = bit_idx % 64
                            
                            # Safety check
                            if 0 <= bit_offset < 64 and word_idx < self.num_words:
                                # Set the bit in the mask
                                mask[word_idx] |= (1 << bit_offset)
            
            # Store the mask
            self.container_placement_masks[container_type] = mask
    
    def update_occupied_bitmap(self):
        """Update the combined occupied bitmap based on all tiers."""
        # Create a bitmap that has a 1 wherever any tier has a container
        self.occupied_bitmap = torch.zeros(self.num_words, dtype=torch.int64, device=self.device)
        for tier_bitmap in self.tier_bitmaps:
            self.occupied_bitmap |= tier_bitmap
    
    def parse_position(self, position: str) -> Tuple[str, int, int]:
        """
        Parse a position string into row, bay, and subslot components.
        Accepts formats: 'A1' (full bay) or 'A1.1' (specific subslot)
        
        Returns:
            Tuple of (row, bay, subslot) - subslot is None for full bay references
        """
        # Pattern for positions with subslot
        subslot_pattern = r"([A-Za-z])(\d+)\.(\d+)"
        # Pattern for positions without subslot
        bay_pattern = r"([A-Za-z])(\d+)"
        
        # Try to match with subslot first
        match = re.match(subslot_pattern, position)
        if match:
            row = match.group(1).upper()
            bay = int(match.group(2))
            subslot = int(match.group(3))
            return row, bay, subslot
        
        # Try to match without subslot
        match = re.match(bay_pattern, position)
        if match:
            row = match.group(1).upper()
            bay = int(match.group(2))
            return row, bay, None
        
        raise ValueError(f"Invalid position format: {position}")
    
    def encode_position(self, position: str) -> List[int]:
        """
        Convert a position string to bit indices.
        For positions without subslot (e.g., 'A1'), returns all subslot bit indices.
        For positions with subslot (e.g., 'A1.1'), returns that specific bit index.
        
        Returns:
            List of bit indices
        """
        # Check if in direct mapping first
        if position in self.position_to_bit:
            return [self.position_to_bit[position]]
        
        # Parse the position
        try:
            row, bay, subslot = self.parse_position(position)
        except ValueError:
            raise ValueError(f"Invalid position: {position}")
        
        # Validate components
        if row not in self.row_names or bay < 1 or bay > self.num_bays:
            raise ValueError(f"Invalid position: {position}")
        
        # If specific subslot specified, get just that bit index
        if subslot is not None:
            if subslot < 1 or subslot > self.subslots_per_bay:
                raise ValueError(f"Invalid subslot: {subslot}")
            
            # Look up the specific position
            position_with_subslot = f"{row}{bay}.{subslot}"
            if position_with_subslot in self.position_to_bit:
                return [self.position_to_bit[position_with_subslot]]
            
            # Calculate bit index for the specific subslot
            row_idx = self.row_names.index(row)
            row_offset = row_idx * self.bits_per_row
            bay_idx = bay - 1
            subslot_idx = subslot - 1
            overall_subslot_idx = bay_idx * self.subslots_per_bay + subslot_idx
            bit_idx = row_offset + overall_subslot_idx
            
            return [bit_idx]
        
        # If no subslot specified, get all subslot bit indices for the bay
        bit_indices = []
        row_idx = self.row_names.index(row)
        row_offset = row_idx * self.bits_per_row
        bay_idx = bay - 1
        
        for subslot_idx in range(self.subslots_per_bay):
            overall_subslot_idx = bay_idx * self.subslots_per_bay + subslot_idx
            bit_idx = row_offset + overall_subslot_idx
            bit_indices.append(bit_idx)
        
        return bit_indices
    
    def decode_position(self, bit_idx: int) -> str:
        """Convert a bit index to a position string with subslot (e.g., 'A1.1')."""
        if bit_idx in self.bit_to_position:
            return self.bit_to_position[bit_idx]
        
        # Calculate position if not in mapping
        row_idx = bit_idx // self.bits_per_row
        relative_idx = bit_idx % self.bits_per_row
        
        # Calculate bay and subslot
        overall_subslot_idx = relative_idx
        bay_idx = overall_subslot_idx // self.subslots_per_bay
        subslot_idx = overall_subslot_idx % self.subslots_per_bay
        
        # Convert to 1-based indexing
        bay = bay_idx + 1
        subslot = subslot_idx + 1
        
        if 0 <= row_idx < len(self.row_names) and 1 <= bay <= self.num_bays and 1 <= subslot <= self.subslots_per_bay:
            return f"{self.row_names[row_idx]}{bay}.{subslot}"
        
        raise ValueError(f"Invalid bit index: {bit_idx}")
    
    def is_occupied(self, position: str, tier: int = 1) -> bool:
        """
        Check if a position at a specific tier is occupied.
        
        Args:
            position: Position string (e.g., 'A1' or 'A1.1')
            tier: Tier level to check
            
        Returns:
            Boolean indicating if the position is occupied at the tier
        """
        try:
            # Get bit indices for the position
            bit_indices = self.encode_position(position)
            
            # Check if tier is valid
            if 1 <= tier <= self.max_tier_height:
                # Check if any of the subslots are occupied
                for bit_idx in bit_indices:
                    word_idx = bit_idx // 64
                    bit_offset = bit_idx % 64
                    
                    if (self.tier_bitmaps[tier-1][word_idx] >> bit_offset) & 1:
                        return True
            
            return False
        except ValueError:
            return False
    
    def is_position_in_special_area(self, position: str, area_type: str) -> bool:
        """
        Check if a position is in a specific special area.
        
        Args:
            position: Position string (e.g., 'A1' or 'A1.1')
            area_type: Special area type (e.g., 'reefer', 'dangerous')
            
        Returns:
            Boolean indicating if the position is in the special area
        """
        try:
            # Convert area type to lowercase
            area_type = area_type.lower()
            
            # Check if valid area type
            if area_type not in self.special_area_bitmaps:
                return False
            
            # Get bit indices for the position
            bit_indices = self.encode_position(position)
            
            # Check if any of the subslots are in the special area
            for bit_idx in bit_indices:
                word_idx = bit_idx // 64
                bit_offset = bit_idx % 64
                
                if (self.special_area_bitmaps[area_type][word_idx] >> bit_offset) & 1:
                    return True
            
            return False
        except ValueError:
            return False
    
    def get_stack_height(self, position: str) -> int:
        """
        Get the current stack height at a position.
        
        Args:
            position: Position string (e.g., 'A1' or 'A1.1')
            
        Returns:
            Current stack height (0 if empty)
        """
        try:
            # Get bit indices for the position
            bit_indices = self.encode_position(position)
            
            # Find the maximum stack height across all subslots
            max_height = 0
            
            for bit_idx in bit_indices:
                word_idx = bit_idx // 64
                bit_offset = bit_idx % 64
                
                # Check from top to bottom
                for tier in range(self.max_tier_height, 0, -1):
                    if (self.tier_bitmaps[tier-1][word_idx] >> bit_offset) & 1:
                        max_height = max(max_height, tier)
                        break
            
            return max_height
        except ValueError:
            return 0
    
    def get_container_length_in_subslots(self, container: Any) -> int:
        """
        Get the length of a container in subslots.
        
        Args:
            container: Container object
            
        Returns:
            Number of subslots the container occupies
        """
        # Get container type
        container_type = getattr(container, 'container_type', 'default')
        
        # Get length in subslots
        return self.container_length_subslots.get(container_type, self.container_length_subslots['default'])
    
    def find_contiguous_free_subslots(self, position: str, length: int, tier: int = 1) -> List[str]:
        """
        Find contiguous free subslots starting from a position.
        
        Args:
            position: Position string (e.g., 'A1' or 'A1.1')
            length: Number of contiguous subslots needed
            tier: Tier level to check
            
        Returns:
            List of subslot positions that are free (empty if not enough free space)
        """
        try:
            # Parse the position to determine the starting point
            row, bay, subslot = self.parse_position(position)
            
            # If no specific subslot provided, use the first subslot in the bay
            if subslot is None:
                subslot = 1
            
            # Build the full starting position
            start_position = f"{row}{bay}.{subslot}"
            start_bit_indices = self.encode_position(start_position)
            
            if not start_bit_indices:
                return []
            
            start_bit_idx = start_bit_indices[0]
            
            # Calculate row and bay indices
            row_idx = self.row_names.index(row)
            bay_idx = bay - 1
            subslot_idx = subslot - 1
            
            # Start checking from this subslot
            free_positions = []
            
            # Need to check if we have enough space in this bay
            remaining_in_bay = self.subslots_per_bay - subslot_idx
            
            if remaining_in_bay < length:
                # Not enough space in this bay
                return []
            
            # Check each potential subslot
            for i in range(length):
                current_subslot = subslot + i
                current_position = f"{row}{bay}.{current_subslot}"
                
                # Check if this subslot is free
                if self.is_occupied(current_position, tier):
                    # Subslot is occupied, can't fit the container
                    return []
                
                free_positions.append(current_position)
            
            return free_positions
        except ValueError:
            return []
    
    def add_container(self, position: str, container: Any, tier: int = None) -> bool:
        """
        Add a container to a specific position and tier.
        
        Args:
            position: Position string (e.g., 'A1' or 'A1.1')
            container: Container object to add
            tier: Tier level (if None, adds to the top tier + 1)
            
        Returns:
            True if container was added successfully, False otherwise
        """
        try:
            # Determine the container's length in subslots
            container_length = self.get_container_length_in_subslots(container)
            
            # Determine tier if not specified
            if tier is None:
                tier = self.get_stack_height(position) + 1
            
            # Check if tier is valid
            if tier > self.max_tier_height:
                return False
            
            # Check if container can be accepted at this position
            if not self.can_accept_container(position, container):
                return False
            
            # Find contiguous free subslots for the container
            free_subslots = self.find_contiguous_free_subslots(position, container_length, tier)
            
            if not free_subslots or len(free_subslots) < container_length:
                return False
            
            # Get bit indices for all subslots
            all_bit_indices = []
            for subslot_position in free_subslots:
                bit_indices = self.encode_position(subslot_position)
                all_bit_indices.extend(bit_indices)
            
            # Mark all subslots as occupied in the tier bitmap
            for bit_idx in all_bit_indices:
                word_idx = bit_idx // 64
                bit_offset = bit_idx % 64
                
                self.tier_bitmaps[tier-1][word_idx] |= (1 << bit_offset)
            
            # Update container registry
            # We only store the container once, with the starting position
            main_position = free_subslots[0]
            
            # Register container in the position registry
            if position not in self.container_registry:
                self.container_registry[position] = {}
            
            self.container_registry[position][tier] = container
            
            # Update container occupation tracking
            if hasattr(container, 'container_id'):
                self.container_occupation[container.container_id] = {
                    'position': position,
                    'tier': tier,
                    'subslots': free_subslots
                }
            
            # Update occupied bitmap
            self.update_occupied_bitmap()
            
            return True
        except ValueError:
            return False
    
    def remove_container(self, position: str, tier: int = None) -> Optional[Any]:
        """
        Remove a container from a position and tier.
        
        Args:
            position: Position string (e.g., 'A1' or 'A1.1')
            tier: Tier level (if None, removes the top container)
            
        Returns:
            The removed container or None if no container was removed
        """
        try:
            # Determine tier if not specified
            if tier is None:
                tier = self.get_stack_height(position)
                if tier == 0:
                    return None
            
            # Check if there are containers above this one
            for t in range(tier + 1, self.max_tier_height + 1):
                if self.is_occupied(position, t):
                    return None  # Can't remove with containers on top
            
            # Get container before removing
            container = None
            if position in self.container_registry and tier in self.container_registry[position]:
                container = self.container_registry[position][tier]
            
            if not container:
                return None
            
            # Get the container's subslots
            container_id = getattr(container, 'container_id', None)
            
            if container_id in self.container_occupation:
                occupied_subslots = self.container_occupation[container_id]['subslots']
            else:
                # Determine container length
                container_length = self.get_container_length_in_subslots(container)
                
                # Find the container's subslots
                # First, determine if we're specifying a subslot or bay position
                row, bay, subslot = self.parse_position(position)
                
                # If no specific subslot provided, we need to find where the container starts
                if subslot is None:
                    # Look through all subslots in the bay to find the container
                    for s in range(1, self.subslots_per_bay + 1):
                        test_position = f"{row}{bay}.{s}"
                        bit_indices = self.encode_position(test_position)
                        
                        if bit_indices:
                            bit_idx = bit_indices[0]
                            word_idx = bit_idx // 64
                            bit_offset = bit_idx % 64
                            
                            if (self.tier_bitmaps[tier-1][word_idx] >> bit_offset) & 1:
                                # Found the beginning of the container
                                occupied_subslots = self.find_contiguous_free_subslots(
                                    test_position, container_length, tier)
                                break
                else:
                    occupied_subslots = self.find_contiguous_free_subslots(
                        position, container_length, tier)
            
            # Mark all subslots as free in the tier bitmap
            for subslot_position in occupied_subslots:
                bit_indices = self.encode_position(subslot_position)
                
                for bit_idx in bit_indices:
                    word_idx = bit_idx // 64
                    bit_offset = bit_idx % 64
                    
                    self.tier_bitmaps[tier-1][word_idx] &= ~(1 << bit_offset)
            
            # Update container registry
            if position in self.container_registry and tier in self.container_registry[position]:
                del self.container_registry[position][tier]
                if not self.container_registry[position]:
                    del self.container_registry[position]
            
            # Update container occupation tracking
            if container_id and container_id in self.container_occupation:
                del self.container_occupation[container_id]
            
            # Update occupied bitmap
            self.update_occupied_bitmap()
            
            return container
        except ValueError:
            return None
    
    def can_accept_container(self, position: str, container: Any) -> bool:
        """
        Check if a container can be accepted at a position.
        
        Args:
            position: Position string (e.g., 'A1' or 'A1.1')
            container: Container to check
            
        Returns:
            True if the container can be accepted, False otherwise
        """
        try:
            # Get container type and length
            container_type = getattr(container, 'container_type', 'default')
            container_length = self.get_container_length_in_subslots(container)
            
            # Check special area constraints
            if hasattr(container, 'goods_type'):
                goods_type = container.goods_type
                
                if goods_type == 'Reefer':
                    # Reefer containers must be in reefer areas
                    if not self.is_position_in_special_area(position, 'reefer'):
                        return False
                elif goods_type == 'Dangerous':
                    # Dangerous containers must be in dangerous goods areas
                    if not self.is_position_in_special_area(position, 'dangerous'):
                        return False
            
            # Check container type constraints
            if container_type == 'Trailer':
                # Trailers must be in trailer areas
                if not self.is_position_in_special_area(position, 'trailer'):
                    return False
                # Trailers can't be stacked
                if self.get_stack_height(position) > 0:
                    return False
            elif container_type == 'Swap Body':
                # Swap bodies must be in swap body areas
                if not self.is_position_in_special_area(position, 'swap_body'):
                    return False
            
            # Check height constraints
            current_height = self.get_stack_height(position)
            if current_height >= self.max_tier_height:
                return False
            
            # Check if there are enough contiguous free subslots
            row, bay, subslot = self.parse_position(position)
            
            # If no specific subslot provided, check from the beginning of the bay
            if subslot is None:
                subslot = 1
            
            # Check if there are enough subslots left in the bay
            remaining_subslots = self.subslots_per_bay - (subslot - 1)
            if remaining_subslots < container_length:
                return False
            
            # Check if the subslots are free at the next tier
            next_tier = current_height + 1
            free_subslots = self.find_contiguous_free_subslots(
                f"{row}{bay}.{subslot}", container_length, next_tier)
            
            if len(free_subslots) < container_length:
                return False
            
            # Check stacking compatibility with container below
            if current_height > 0:
                container_below = None
                for t in range(current_height, 0, -1):
                    if position in self.container_registry and t in self.container_registry[position]:
                        container_below = self.container_registry[position][t]
                        break
                
                if container_below and hasattr(container, 'can_stack_with'):
                    if not container.can_stack_with(container_below):
                        return False
            
            # Check container-specific can_be_stacked rules
            if hasattr(container, 'is_stackable') and not container.is_stackable and current_height > 0:
                return False
            
            return True
        except ValueError:
            return False
    
    def get_proximity_mask(self, position: str, n: int, container=None) -> torch.Tensor:
        """
        Create a proximity mask for subslots within range n of the original position.
        
        Args:
            position: Position string (e.g., 'A1' or 'A1.1')
            n: Number of bays to include in proximity
            container: Optional container for filtering
            
        Returns:
            Bitmap mask of proximate positions
        """
        try:
            # Parse the position
            row, bay, subslot = self.parse_position(position)
            
            # If no specific subslot provided, use the first subslot in the bay
            if subslot is None:
                subslot = 1
            
            # Build the full starting position
            start_position = f"{row}{bay}.{subslot}"
            start_bit_indices = self.encode_position(start_position)
            
            if not start_bit_indices:
                return torch.zeros(self.num_words, dtype=torch.int64, device=self.device)
            
            start_bit_idx = start_bit_indices[0]
            
            # Use cache for repeated calculations
            cache_key = (start_bit_idx, n)
            if container:
                container_type = getattr(container, 'container_type', 'default')
                goods_type = getattr(container, 'goods_type', 'Regular')
                cache_key = (start_bit_idx, n, container_type, goods_type)
                
            if cache_key in self.proximity_masks:
                return self.proximity_masks[cache_key]
            
            # Create an empty proximity mask
            proximity_mask = torch.zeros(self.num_words, dtype=torch.int64, device=self.device)
            
            # Calculate row and bay indices
            row_idx = self.row_names.index(row)
            bay_idx = bay - 1
            
            # Calculate bay range for proximity
            min_bay = max(0, bay_idx - n)
            max_bay = min(self.num_bays - 1, bay_idx + n)
            
            # Create a 2D tensor of the yard (rows x total_subslots)
            yard_tensor = torch.zeros((self.num_rows, self.total_subslots), dtype=torch.bool, device=self.device)
            
            # Convert bay range to subslot range
            min_subslot = min_bay * self.subslots_per_bay
            max_subslot = (max_bay + 1) * self.subslots_per_bay - 1
            max_subslot = min(max_subslot, self.total_subslots - 1)  # Ensure we don't exceed dimensions
            
            # Set rectangular region to True (all rows, bays within range)
            yard_tensor[:, min_subslot:max_subslot+1] = True
            
            # Set original position to False (exclude it)
            orig_subslot_idx = bay_idx * self.subslots_per_bay + (subslot - 1)
            if 0 <= row_idx < self.num_rows and 0 <= orig_subslot_idx < self.total_subslots:
                yard_tensor[row_idx, orig_subslot_idx] = False
            
            # Apply container type filtering
            if container:
                container_type = getattr(container, 'container_type', 'default')
                container_length = self.get_container_length_in_subslots(container)
                goods_type = getattr(container, 'goods_type', 'Regular')
                
                # Create a subslot-level mask based on container constraints
                valid_positions_mask = torch.zeros_like(yard_tensor)
                
                # For each row and subslot
                for r in range(self.num_rows):
                    for s in range(self.total_subslots - container_length + 1):
                        # For container to fit, we need contiguous subslots
                        # and they need to be in the same bay or adjacent bays
                        start_bay = s // self.subslots_per_bay
                        end_bay = (s + container_length - 1) // self.subslots_per_bay
                        
                        # Must be contiguous within same bay or adjacent bays
                        if end_bay > start_bay + 1:
                            continue
                        
                        # Must have correct alignment within bay
                        start_subslot_in_bay = s % self.subslots_per_bay
                        end_subslot_in_bay = (s + container_length - 1) % self.subslots_per_bay
                        
                        # Check if spans multiple bays
                        if start_bay != end_bay:
                            # Can only span to adjacent bay if aligned properly
                            if start_subslot_in_bay + container_length > self.subslots_per_bay:
                                valid_positions_mask[r, s] = True
                        else:
                            # Within same bay, any position works
                            valid_positions_mask[r, s] = True
                
                # Apply special area constraints
                if container_type == "Trailer":
                    # Get trailer area mask
                    trailer_mask = torch.zeros_like(yard_tensor)
                    for r in range(self.num_rows):
                        for s in range(self.total_subslots):
                            if s < self.total_subslots:  # Safety check
                                pos = f"{self.row_names[r]}{s // self.subslots_per_bay + 1}.{s % self.subslots_per_bay + 1}"
                                if self.is_position_in_special_area(pos, 'trailer'):
                                    trailer_mask[r, s] = True
                    valid_positions_mask &= trailer_mask
                elif container_type == "Swap Body":
                    # Get swap body area mask
                    swap_body_mask = torch.zeros_like(yard_tensor)
                    for r in range(self.num_rows):
                        for s in range(self.total_subslots):
                            if s < self.total_subslots:  # Safety check
                                pos = f"{self.row_names[r]}{s // self.subslots_per_bay + 1}.{s % self.subslots_per_bay + 1}"
                                if self.is_position_in_special_area(pos, 'swap_body'):
                                    swap_body_mask[r, s] = True
                    valid_positions_mask &= swap_body_mask
                elif goods_type == "Reefer":
                    # Get reefer area mask
                    reefer_mask = torch.zeros_like(yard_tensor)
                    for r in range(self.num_rows):
                        for s in range(self.total_subslots):
                            if s < self.total_subslots:  # Safety check
                                pos = f"{self.row_names[r]}{s // self.subslots_per_bay + 1}.{s % self.subslots_per_bay + 1}"
                                if self.is_position_in_special_area(pos, 'reefer'):
                                    reefer_mask[r, s] = True
                    valid_positions_mask &= reefer_mask
                elif goods_type == "Dangerous":
                    # Get dangerous area mask
                    dangerous_mask = torch.zeros_like(yard_tensor)
                    for r in range(self.num_rows):
                        for s in range(self.total_subslots):
                            if s < self.total_subslots:  # Safety check
                                pos = f"{self.row_names[r]}{s // self.subslots_per_bay + 1}.{s % self.subslots_per_bay + 1}"
                                if self.is_position_in_special_area(pos, 'dangerous'):
                                    dangerous_mask[r, s] = True
                    valid_positions_mask &= dangerous_mask
                
                # Only keep positions that satisfy both proximity and container constraints
                yard_tensor &= valid_positions_mask
            
            # Convert to bit indices
            rows, subslots = torch.where(yard_tensor)
            bit_indices = rows * self.bits_per_row + subslots
            
            # Convert to word indices and bit positions
            word_indices = bit_indices // 64
            bit_positions = bit_indices % 64
            
            # Set bits in proximity mask
            unique_words = torch.unique(word_indices)
            for word_idx in unique_words:
                word_idx_int = word_idx.item()
                
                # Skip if out of range
                if word_idx_int >= self.num_words:
                    continue
                    
                bits_in_word = bit_positions[word_indices == word_idx]
                
                for bit in bits_in_word:
                    bit_offset = bit.item()
                    # Safety check
                    if 0 <= bit_offset < 64:
                        proximity_mask[word_idx_int] |= (1 << bit_offset)
            
            # Cache result
            self.proximity_masks[cache_key] = proximity_mask
            return proximity_mask
            
        except ValueError:
            return torch.zeros(self.num_words, dtype=torch.int64, device=self.device)
    
    def calc_possible_moves(self, position: str, n: int) -> List[str]:
        """
        Calculate all possible positions a container can be moved to within n bays.
        
        Args:
            position: Starting position string (e.g., 'A1' or 'A1.1')
            n: Number of bays to consider in each direction
            
        Returns:
            List of valid destination bay positions (not subslots)
        """
        try:
            # Check if position has a container to move
            container = None
            tier = None
            
            # Try direct lookup in registry first
            if position in self.container_registry:
                # Get the top container
                if self.container_registry[position]:
                    tier = max(self.container_registry[position].keys())
                    container = self.container_registry[position][tier]
            
            # If not found, it might be a subslot position
            if container is None:
                row, bay, subslot = self.parse_position(position)
                
                # Try to find container in the registry with the bay position
                bay_position = f"{row}{bay}"
                if bay_position in self.container_registry:
                    tier = max(self.container_registry[bay_position].keys())
                    container = self.container_registry[bay_position][tier]
            
            # Still no container found
            if container is None:
                return []
            
            # Get proximity mask WITH container type filtering
            proximity_mask = self.get_proximity_mask(position, n, container)
            
            # Get the container's length in subslots
            container_length = self.get_container_length_in_subslots(container)
            
            # We need to check each potential position against container rules
            valid_destinations = []
            
            # Get bit indices from mask
            for word_idx in range(len(proximity_mask)):
                word = proximity_mask[word_idx].item()
                if word == 0:
                    continue
                
                # Process each set bit in the word
                for bit_offset in range(64):
                    if (word >> bit_offset) & 1:
                        dest_bit_idx = word_idx * 64 + bit_offset
                        
                        # Determine if this bit index is within our yard
                        if dest_bit_idx < self.total_bits:
                            try:
                                # Get the subslot position
                                dest_subslot_position = self.decode_position(dest_bit_idx)
                                
                                # Extract the bay position (remove the subslot suffix)
                                row, bay, subslot = self.parse_position(dest_subslot_position)
                                dest_bay_position = f"{row}{bay}"
                                
                                # Check if this is a valid starting position for the container
                                if self.can_accept_container(dest_subslot_position, container):
                                    # We add the bay position, not the subslot
                                    if dest_bay_position not in valid_destinations:
                                        valid_destinations.append(dest_bay_position)
                            except ValueError:
                                # Skip invalid positions
                                continue
            
            return valid_destinations
        except ValueError:
            return []

    def get_container(self, position: str, tier: int = None) -> Optional[Any]:
        """
        Get the container at a specific position and tier.
        
        Args:
            position: Position string (e.g., 'A1' or 'A1.1')
            tier: Tier level (if None, gets the top container)
            
        Returns:
            Container object or None if no container found
        """
        try:
            # Parse the position
            row, bay, subslot = self.parse_position(position)
            
            # Convert to bay position
            bay_position = f"{row}{bay}"
            
            # Default to top container if tier not specified
            if tier is None:
                tier = self.get_stack_height(bay_position)
                if tier == 0:
                    return None
            
            # Check container registry
            if bay_position in self.container_registry and tier in self.container_registry[bay_position]:
                return self.container_registry[bay_position][tier]
            
            return None
        except ValueError:
            return None
    
    def get_top_container(self, position: str) -> Tuple[Optional[Any], Optional[int]]:
        """
        Get the top container at a position and its tier.
        
        Args:
            position: Position string (e.g., 'A1' or 'A1.1')
            
        Returns:
            Tuple of (container, tier) or (None, None) if no container found
        """
        try:
            # Parse the position
            row, bay, subslot = self.parse_position(position)
            
            # Convert to bay position
            bay_position = f"{row}{bay}"
            
            # Get the stack height
            tier = self.get_stack_height(bay_position)
            if tier == 0:
                return None, None
            
            # Get the top container
            container = self.get_container(bay_position, tier)
            return container, tier
        except ValueError:
            return None, None
    
    def find_container(self, container_id: str) -> Optional[Tuple[str, int]]:
        """
        Find a container by ID.
        
        Args:
            container_id: ID of the container to find
            
        Returns:
            Tuple of (position, tier) or None if not found
        """
        # Check direct lookup first
        if container_id in self.container_occupation:
            return (self.container_occupation[container_id]['position'],
                    self.container_occupation[container_id]['tier'])
        
        # Fallback to searching the container registry
        for position, tiers in self.container_registry.items():
            for tier, container in tiers.items():
                if hasattr(container, 'container_id') and container.container_id == container_id:
                    return position, tier
        
        return None
    
    def get_containers_at_position(self, position: str) -> Dict[int, Any]:
        """
        Get all containers at a position.
        
        Args:
            position: Position string (e.g., 'A1')
            
        Returns:
            Dictionary mapping tier to container
        """
        try:
            # Parse the position
            row, bay, subslot = self.parse_position(position)
            
            # Convert to bay position
            bay_position = f"{row}{bay}"
            
            if bay_position in self.container_registry:
                return dict(self.container_registry[bay_position])
            return {}
        except ValueError:
            return {}
    
    def clear(self):
        """Clear the entire storage yard."""
        # Reset all bitmaps
        for i in range(self.max_tier_height):
            self.tier_bitmaps[i].zero_()
        
        # Clear container registry
        self.container_registry = {}
        
        # Clear container occupation tracking
        self.container_occupation = {}
        
        # Clear proximity mask cache
        self.proximity_masks = {}
        
        # Update occupied bitmap
        self.update_occupied_bitmap()
    
    def get_state_representation(self) -> torch.Tensor:
        """
        Get a tensor representation of the yard state.
        
        Returns:
            A 3D tensor of shape (num_rows, num_bays, max_tier_height)
        """
        # Create tensor for occupied positions
        state = torch.zeros((self.num_rows, self.num_bays, self.max_tier_height), 
                           dtype=torch.int8, device=self.device)
        
        # Loop through every bay position
        for row_idx, row in enumerate(self.row_names):
            for bay in range(1, self.num_bays + 1):
                bay_position = f"{row}{bay}"
                
                # Check if any containers at this position
                if bay_position in self.container_registry:
                    # Mark all tiers that have containers
                    for tier, _ in self.container_registry[bay_position].items():
                        if 1 <= tier <= self.max_tier_height:
                            state[row_idx, bay - 1, tier - 1] = 1
        
        return state
    
    def visualize_bitmap(self, bitmap, title="Bitmap Visualization"):
        """
        Visualize a bitmap as a 2D grid using vectorized operations.
        
        Args:
            bitmap: The bitmap to visualize
            title: Title for the visualization
        """
        import matplotlib.pyplot as plt
        
        # Create a 2D tensor representation with subslot resolution
        grid = torch.zeros((self.num_rows, self.total_subslots), dtype=torch.float32, device=self.device)
        
        # Convert bitmap to 2D grid using vectorized operations
        bitmap_expanded = bitmap.view(-1, 1)  # Shape (n_words, 1)
        bits_expanded = torch.arange(64, device=self.device).repeat(len(bitmap), 1)  # Shape (n_words, 64)
        
        # Generate all bit values
        bit_values = (bitmap_expanded & (1 << bits_expanded)) != 0  # Shape (n_words, 64)
        
        # Flatten to 1D array of bits
        all_bits = bit_values.view(-1)[:self.total_bits]  # Ensure we don't exceed total bits
        
        # Reshape to match our grid layout
        reshaped_bits = all_bits.view(self.num_rows, self.bits_per_row)
        
        # Extract only the valid subslots
        grid = reshaped_bits[:, :self.total_subslots].float()
        
        # Visualize using matplotlib
        plt.figure(figsize=(15, 6))
        plt.imshow(grid.cpu().numpy(), cmap='Blues', interpolation='none')
        plt.title(title)
        plt.xlabel('Subslot')
        plt.ylabel('Row')
        
        # Add row names on y-axis
        plt.yticks(range(self.num_rows), self.row_names)
        
        # Add bay dividers
        for bay in range(1, self.num_bays):
            plt.axvline(x=bay * self.subslots_per_bay - 0.5, color='red', linestyle='-', linewidth=0.5)
        
        # Add grid lines
        plt.grid(which='both', color='lightgrey', linestyle='-', linewidth=0.5)
        
        plt.colorbar(label='Occupied')
        plt.tight_layout()
        plt.show()
    
    def visualize_3d(self, show_container_types=True, figsize=(30, 20)):
        """
        Create a 3D visualization with proper container orientation.
        
        Each slot is 40ft long and containers are oriented with:
        - Their long side along the bay axis
        - Their short side along the row axis
        - Their height upward in tiers
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import numpy as np
        
        # Create the figure and 3D axis
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Color mapping
        container_colors = {
            'TWEU': 'royalblue',
            'THEU': 'purple',
            'FEU': 'lightseagreen',
            'FFEU': 'teal',
            'Trailer': 'darkred',
            'Swap Body': 'goldenrod',
            'default': 'gray'
        }
        
        # Edge color mapping
        goods_edge_colors = {
            'Regular': 'black',
            'Reefer': 'blue',
            'Dangerous': 'red'
        }
        
        # Set up legend elements
        legend_elements = []
        from matplotlib.patches import Patch
        
        # Collect all containers to visualize
        for position, tiers in self.container_registry.items():
            try:
                # Skip anything that's not a bay position
                if '.' in position:
                    continue
                
                # Parse bay position
                row, bay, subslot = self.parse_position(position)
                
                row_idx = self.row_names.index(row)
                bay_idx = bay - 1
                
                # Process each tier
                for tier, container in tiers.items():
                    if 1 <= tier <= self.max_tier_height:
                        container_type = getattr(container, 'container_type', 'default')
                        goods_type = getattr(container, 'goods_type', 'Regular')
                        
                        # Get colors
                        color = container_colors.get(container_type, container_colors['default'])
                        edge_color = goods_edge_colors.get(goods_type, goods_edge_colors['Regular'])
                        
                        # Add to legend if needed
                        legend_key = f"{container_type} - {goods_type}"
                        if not any(le.get_label() == legend_key for le in legend_elements):
                            legend_elements.append(
                                Patch(facecolor=color, edgecolor=edge_color, label=legend_key)
                            )
                        
                        # Get container length in subslots
                        container_length = self.get_container_length_in_subslots(container)
                        
                        # Determine starting subslot
                        # If container_id in occupation map, we know exactly where it is
                        container_id = getattr(container, 'container_id', None)
                        starting_subslot = 1
                        
                        if container_id and container_id in self.container_occupation:
                            # Get the first subslot position
                            first_subslot = self.container_occupation[container_id]['subslots'][0]
                            _, _, starting_subslot = self.parse_position(first_subslot)
                        
                        # Calculate position and dimensions in the 3D space
                        # Map to plot coordinates where 1 bay = 1 unit
                        
                        # Calculate x position (bay axis)
                        # If the container doesn't start at subslot 1, adjust the position
                        x_start = bay_idx + (starting_subslot - 1) / self.subslots_per_bay
                        
                        # Width is container_length / subslots_per_bay of a bay
                        dx = container_length / self.subslots_per_bay
                        
                        # All containers have same width
                        dy = 0.7  # Slightly less than 1 to see gaps
                        
                        # Height is one tier
                        dz = 0.9  # Slightly less than 1 to see layers
                        
                        # Draw the container with proper dimensions and position
                        ax.bar3d(
                            x_start, row_idx + 0.15, tier - 1,      # Position (x, y, z)
                            dx, dy, dz,                           # Dimensions (dx, dy, dz)
                            color=color,
                            shade=True,
                            alpha=0.8,
                            edgecolor=edge_color,
                            linewidth=0.5
                        )
            except ValueError:
                continue
        
        # Set axis labels and limits
        ax.set_xlabel('Bay')
        ax.set_ylabel('Row')
        ax.set_zlabel('Tier')
        
        # Set row labels to be A, B, C, etc.
        ax.set_yticks(range(self.num_rows))
        ax.set_yticklabels(self.row_names)
        
        # Set limits
        ax.set_xlim(0, self.num_bays)
        ax.set_ylim(0, self.num_rows)
        ax.set_zlim(0, self.max_tier_height)
        
        # Adjust aspect ratio for better visualization
        ax.set_box_aspect([self.num_bays/3, self.num_rows/6, self.max_tier_height/3])
        
        # Set title and legend
        ax.set_title('3D Container Yard Visualization with Subslot Support')
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
        
        # Set the viewing angle to better see the layout
        ax.view_init(elev=20, azim=230)
        
        plt.tight_layout()
        return fig, ax
    
    def visualize_container_placement(self, container_type='FEU'):
        """
        Visualize valid placement positions for a specific container type.
        
        Args:
            container_type: Type of container to visualize placement for
        """
        import matplotlib.pyplot as plt
        
        # Get the placement mask for this container type
        if container_type in self.container_placement_masks:
            placement_mask = self.container_placement_masks[container_type]
            self.visualize_bitmap(placement_mask, f"Valid placement positions for {container_type}")
        else:
            print(f"No placement mask for container type: {container_type}")
    
    def __str__(self):
        """String representation of the storage yard."""
        container_count = sum(len(tiers) for tiers in self.container_registry.values())
        
        # Get containers per row
        row_counts = {}
        for position in self.container_registry:
            if '.' not in position:  # Skip subslot positions
                row = position[0]
                if row in row_counts:
                    row_counts[row] += len(self.container_registry[position])
                else:
                    row_counts[row] = len(self.container_registry[position])
        
        # Format string
        lines = []
        lines.append(f"Enhanced Bitmap Storage Yard with Subslots:")
        lines.append(f"- {self.num_rows} rows, {self.num_bays} bays, {self.subslots_per_bay} subslots per bay")
        lines.append(f"- {container_count} containers")
        
        for row in self.row_names:
            count = row_counts.get(row, 0)
            lines.append(f"Row {row}: {count} containers")
        
        return '\n'.join(lines)