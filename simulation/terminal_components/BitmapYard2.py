import numpy as np
import torch
import time
import re
from typing import Dict, List, Tuple, Optional, Any, Set

class SlotTierBitmapYard:
    """
    GPU-accelerated bitmap representation of a container storage yard
    with flattened slot-tier representation.
    
    Uses a position format of {Row}{Bay.Slot}-T{Tier}, where:
    - Each bay is divided into 4 slots of 10 feet each
    - Tiers are explicitly represented in the position string
    
    Container placement rules:
    - TWEU (20ft): 2 consecutive slots within the same bay
    - THEU (30ft): 3 consecutive slots within the same bay
    - FEU (40ft): All 4 slots of a bay (full bay)
    - FFEU (45ft): 5 slots (spanning into adjacent bay)
    """
    
    def __init__(self, 
                num_rows: int, 
                num_bays: int, 
                max_tier_height: int = 5,
                row_names: List[str] = None,
                special_areas: Dict[str, List[Tuple[str, int, int]]] = None,
                device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the flattened slot-tier bitmap storage yard.
        
        Args:
            num_rows: Number of rows in the yard
            num_bays: Number of bays per row
            max_tier_height: Maximum stacking height
            row_names: Names for each row (defaults to A, B, C...)
            special_areas: Dictionary mapping special types to areas
            device: Computation device ('cuda' or 'cpu')
        """
        self.num_rows = num_rows
        self.num_bays = num_bays
        self.slots_per_bay = 4  # Each bay has 4 slots of 10ft each
        self.max_tier_height = max_tier_height
        self.device = device
        
        # Initialize row names if not provided
        if row_names is None:
            self.row_names = [chr(65 + i) for i in range(num_rows)]  # A, B, C, ...
        else:
            self.row_names = row_names[:num_rows]
        
        # Calculate bitmap dimensions
        # Each position is now {Row}{Bay.Slot}-T{Tier}
        # Total slots = num_rows * num_bays * slots_per_bay * max_tier_height
        self.total_slots = num_rows * num_bays * self.slots_per_bay * max_tier_height
        
        # Align to 64-bit boundaries for efficient operations
        self.total_words = (self.total_slots + 63) // 64
        
        # Special areas for different container types
        if special_areas is None:
            # Default special areas
            self.special_areas = {
                'reefer': [('A', 1, 5)],     # Row A, bays 1-5
                'dangerous': [('C', 6, 10)],  # Row C, bays 6-10
                'trailer': [('A', 15, 25)],   # Row A, bays 15-25
                'swap_body': [('B', 30, 40)]  # Row B, bays 30-40
            }
        else:
            self.special_areas = special_areas
        
        # Pre-compute powers of 2 for bit operations (to avoid overflow on CUDA)
        self.powers_of_2 = torch.zeros(64, dtype=torch.int64, device=device)
        for i in range(64):
            if i == 0:
                self.powers_of_2[i] = 1
            else:
                self.powers_of_2[i] = self.powers_of_2[i-1] * 2
        
        # Initialize bitmap (1 for occupied, 0 for free)
        self.occupied_bitmap = torch.zeros(self.total_words, dtype=torch.int64, device=device)
        
        # Initialize bitmaps for special areas
        self.special_area_bitmaps = {
            'reefer': torch.zeros(self.total_words, dtype=torch.int64, device=device),
            'dangerous': torch.zeros(self.total_words, dtype=torch.int64, device=device),
            'trailer': torch.zeros(self.total_words, dtype=torch.int64, device=device),
            'swap_body': torch.zeros(self.total_words, dtype=torch.int64, device=device)
        }
        
        # Initialize container type-specific placement masks
        self.container_placement_masks = {
            'TWEU': None,  # 20ft (2 slots) - will be generated per position
            'THEU': None,  # 30ft (3 slots) - will be generated per position
            'FEU': None,   # 40ft (4 slots/full bay) - will be generated per position
            'FFEU': None,  # 45ft (5 slots) - will be generated per position
        }
        
        # Initialize position mapping
        self.position_to_bit = {}  # Maps position string to bit index
        self.bit_to_position = {}  # Maps bit index to position string
        self._build_position_mapping()
        
        # Initialize special area bitmaps
        self._initialize_special_area_bitmaps()
        
        # Container registry for efficient lookup
        self.container_registry = {}  # position -> container
        
        # Pre-compute common masks for operations
        self._precompute_container_masks()
        
        # Cache for proximity calculations
        self.proximity_masks = {}
    
    def _build_position_mapping(self):
        """Build mapping between position strings and bit indices."""
        bit_idx = 0
        
        for row_idx, row in enumerate(self.row_names):
            for bay in range(1, self.num_bays + 1):
                for slot in range(1, self.slots_per_bay + 1):
                    for tier in range(1, self.max_tier_height + 1):
                        position = f"{row}{bay}.{slot}-T{tier}"
                        self.position_to_bit[position] = bit_idx
                        self.bit_to_position[bit_idx] = position
                        bit_idx += 1
    
    def _initialize_special_area_bitmaps(self):
        """Initialize bitmaps for special areas based on configuration."""
        for area_type, areas in self.special_areas.items():
            bitmap = self.special_area_bitmaps[area_type]
            
            for area_row, start_bay, end_bay in areas:
                if area_row in self.row_names:
                    row_idx = self.row_names.index(area_row)
                    
                    # For each bay in the special area
                    for bay in range(start_bay, end_bay + 1):
                        # For each slot in the bay
                        for slot in range(1, self.slots_per_bay + 1):
                            # For each tier (special areas apply to all tiers)
                            for tier in range(1, self.max_tier_height + 1):
                                position = f"{area_row}{bay}.{slot}-T{tier}"
                                if position in self.position_to_bit:
                                    bit_idx = self.position_to_bit[position]
                                    word_idx = bit_idx // 64
                                    bit_offset = bit_idx % 64
                                    bitmap[word_idx] = bitmap[word_idx] | self.powers_of_2[bit_offset]
    
    def _precompute_container_masks(self):
        """Pre-compute common masks for container placement operations."""
        # These are template masks for different container types
        # Actual masks will be generated for specific positions when needed
        
        # Create sets to store valid starting positions
        self.valid_starts = {
            'TWEU': set(),  # 20ft: positions where a TWEU can start
            'THEU': set(),  # 30ft: positions where a THEU can start
            'FEU': set(),   # 40ft: positions where a FEU can start
            'FFEU': set(),  # 45ft: positions where a FFEU can start
        }
        
        # For each row and bay, compute valid starting positions
        for row in self.row_names:
            for bay in range(1, self.num_bays + 1):
                # For TWEU (20ft): can start at slots 1 or 3
                for start_slot in [1, 3]:
                    # Check if both required slots exist
                    if start_slot + 1 <= self.slots_per_bay:
                        for tier in range(1, self.max_tier_height + 1):
                            start_pos = f"{row}{bay}.{start_slot}-T{tier}"
                            self.valid_starts['TWEU'].add(start_pos)
                
                # For THEU (30ft): can only start at slot 1
                if 3 <= self.slots_per_bay:  # Need 3 slots
                    for tier in range(1, self.max_tier_height + 1):
                        start_pos = f"{row}{bay}.1-T{tier}"
                        self.valid_starts['THEU'].add(start_pos)
                
                # For FEU (40ft): always starts at slot 1 (takes whole bay)
                if self.slots_per_bay == 4:  # Must have all 4 slots
                    for tier in range(1, self.max_tier_height + 1):
                        start_pos = f"{row}{bay}.1-T{tier}"
                        self.valid_starts['FEU'].add(start_pos)
                
                # For FFEU (45ft): can only start at slot 1 if next bay exists
                if bay < self.num_bays:  # Need next bay for 5th slot
                    for tier in range(1, self.max_tier_height + 1):
                        start_pos = f"{row}{bay}.1-T{tier}"
                        self.valid_starts['FFEU'].add(start_pos)
    
    def encode_position(self, position: str) -> int:
        """Convert a position string to a bit index."""
        if position in self.position_to_bit:
            return self.position_to_bit[position]
        
        # Parse position if not in mapping
        match = re.match(r'([A-Z])(\d+)\.(\d+)-T(\d+)', position)
        if match:
            row, bay, slot, tier = match.groups()
            bay, slot, tier = int(bay), int(slot), int(tier)
            
            if (row in self.row_names and 
                1 <= bay <= self.num_bays and 
                1 <= slot <= self.slots_per_bay and 
                1 <= tier <= self.max_tier_height):
                
                row_idx = self.row_names.index(row)
                bit_idx = (((row_idx * self.num_bays + (bay - 1)) * self.slots_per_bay + (slot - 1)) 
                          * self.max_tier_height + (tier - 1))
                return bit_idx
        
        raise ValueError(f"Invalid position: {position}")
    
    def decode_position(self, bit_idx: int) -> str:
        """Convert a bit index to a position string."""
        if bit_idx in self.bit_to_position:
            return self.bit_to_position[bit_idx]
        
        # Calculate position if not in mapping
        if 0 <= bit_idx < self.total_slots:
            tier = (bit_idx % self.max_tier_height) + 1
            slot = ((bit_idx // self.max_tier_height) % self.slots_per_bay) + 1
            bay = (((bit_idx // self.max_tier_height) // self.slots_per_bay) % self.num_bays) + 1
            row_idx = ((bit_idx // self.max_tier_height) // self.slots_per_bay) // self.num_bays
            
            if 0 <= row_idx < len(self.row_names):
                row = self.row_names[row_idx]
                return f"{row}{bay}.{slot}-T{tier}"
        
        raise ValueError(f"Invalid bit index: {bit_idx}")
    
    def get_container_mask(self, position: str, container_type: str) -> torch.Tensor:
        """
        Generate a bitmap mask for a specific container type at the given position.
        
        Args:
            position: Starting position string
            container_type: Type of container (TWEU, THEU, FEU, FFEU)
            
        Returns:
            Tensor mask with 1s for all slots that would be occupied by this container
        """
        # Parse the starting position
        match = re.match(r'([A-Z])(\d+)\.(\d+)-T(\d+)', position)
        if not match:
            raise ValueError(f"Invalid position format: {position}")
            
        row, bay, slot, tier = match.groups()
        bay, slot, tier = int(bay), int(slot), int(tier)
        
        # Create empty mask
        mask = torch.zeros(self.total_words, dtype=torch.int64, device=self.device)
        
        # Generate mask based on container type
        slots_to_occupy = []
        
        if container_type == 'TWEU':  # 20ft: 2 slots
            # TWEU must start at slot 1 or 3
            if slot not in [1, 3]:
                return mask  # Invalid starting slot
                
            # Check if we have enough slots in this bay
            if slot + 1 > self.slots_per_bay:
                return mask  # Not enough slots
                
            # Occupy 2 consecutive slots
            slots_to_occupy = [(bay, s, tier) for s in range(slot, slot + 2)]
            
        elif container_type == 'THEU':  # 30ft: 3 slots
            # THEU must start at slot 1
            if slot != 1:
                return mask  # Invalid starting slot
                
            # Check if we have enough slots in this bay
            if slot + 2 > self.slots_per_bay:
                return mask  # Not enough slots
                
            # Occupy 3 consecutive slots
            slots_to_occupy = [(bay, s, tier) for s in range(slot, slot + 3)]
            
        elif container_type == 'FEU':  # 40ft: 4 slots (full bay)
            # FEU must start at slot 1
            if slot != 1:
                return mask  # Invalid starting slot
                
            # Check if we have enough slots in this bay
            if self.slots_per_bay != 4:
                return mask  # Bay must have exactly 4 slots
                
            # Occupy all 4 slots in this bay
            slots_to_occupy = [(bay, s, tier) for s in range(1, 5)]
            
        elif container_type == 'FFEU':  # 45ft: 5 slots
            # FFEU must start at slot 1
            if slot != 1:
                return mask  # Invalid starting slot
                
            # Check if we have enough slots (need 1 slot in next bay)
            if bay >= self.num_bays:
                return mask  # No next bay available
                
            # Occupy all 4 slots in this bay and 1 slot in next bay
            slots_to_occupy = [(bay, s, tier) for s in range(1, 5)]
            slots_to_occupy.append((bay + 1, 1, tier))
            
        else:
            # Unknown container type
            return mask
        
        # Set bits for all slots to occupy
        for bay_num, slot_num, tier_num in slots_to_occupy:
            pos = f"{row}{bay_num}.{slot_num}-T{tier_num}"
            if pos in self.position_to_bit:
                bit_idx = self.position_to_bit[pos]
                word_idx = bit_idx // 64
                bit_offset = bit_idx % 64
                mask[word_idx] = mask[word_idx] | self.powers_of_2[bit_offset]
        
        return mask
    
    def is_position_occupied(self, position: str) -> bool:
        """Check if a position is occupied."""
        try:
            bit_idx = self.encode_position(position)
            word_idx = bit_idx // 64
            bit_offset = bit_idx % 64
            return bool((self.occupied_bitmap[word_idx] & self.powers_of_2[bit_offset]) != 0)
        except ValueError:
            return False
    
    def is_position_in_special_area(self, position: str, area_type: str) -> bool:
        """Check if a position is in a specific special area."""
        try:
            # Convert area type to lowercase
            area_type = area_type.lower()
            
            # Check if valid area type
            if area_type not in self.special_area_bitmaps:
                return False
            
            # Get bit index
            bit_idx = self.encode_position(position)
            word_idx = bit_idx // 64
            bit_offset = bit_idx % 64
            
            # Check special area bitmap
            return bool((self.special_area_bitmaps[area_type][word_idx] & self.powers_of_2[bit_offset]) != 0)
        except ValueError:
            return False
    
    def is_container_placement_valid(self, position: str, container: Any) -> bool:
        """
        Check if a container can be placed at the given position.
        
        Args:
            position: Starting position string
            container: Container object to check
            
        Returns:
            True if the container can be placed, False otherwise
        """
        # Get container type
        if not hasattr(container, 'container_type'):
            return False
            
        container_type = container.container_type
        goods_type = getattr(container, 'goods_type', 'Regular')
        
        # Check if container type is valid
        if container_type not in ['TWEU', 'THEU', 'FEU', 'FFEU', 'Trailer', 'Swap Body']:
            return False
        
        # Special handling for trailers and swap bodies
        if container_type == 'Trailer':
            # Trailers must be in trailer areas
            if not self.is_position_in_special_area(position, 'trailer'):
                return False
                
            # Trailers can't be stacked (must be tier 1)
            if '-T1' not in position:
                return False
                
        elif container_type == 'Swap Body':
            # Swap bodies must be in swap body areas
            if not self.is_position_in_special_area(position, 'swap_body'):
                return False
        
        # Check special area constraints for goods type
        if goods_type == 'Reefer':
            # Reefer containers must be in reefer areas
            if not self.is_position_in_special_area(position, 'reefer'):
                return False
        elif goods_type == 'Dangerous':
            # Dangerous goods must be in dangerous areas
            if not self.is_position_in_special_area(position, 'dangerous'):
                return False
        
        # Parse the starting position
        match = re.match(r'([A-Z])(\d+)\.(\d+)-T(\d+)', position)
        if not match:
            return False
            
        row, bay, slot, tier = match.groups()
        bay, slot, tier = int(bay), int(slot), int(tier)
        
        # Stacking checks (evaluate stackability for containers above tier 1)
        if tier > 1:
            # Check container directly below 
            position_below = f"{row}{bay}.{slot}-T{tier-1}"
            
            # Get container below
            container_below = self.get_container(position_below)
            
            # If no container below, can't stack
            if container_below is None:
                return False
                
            # Check stacking compatibility
            if (hasattr(container, 'can_stack_with') and 
                not container.can_stack_with(container_below)):
                return False
                
            # Check if container is stackable
            if hasattr(container, 'is_stackable') and not container.is_stackable:
                return False
        
        # Check if starting position is valid for this container type
        # Only needed for containers with specific slot requirements
        if container_type in ['TWEU', 'THEU', 'FEU', 'FFEU']:
            # TWEU (20ft): must start at slot 1 or 3
            if container_type == 'TWEU' and slot not in [1, 3]:
                return False
                
            # THEU (30ft): must start at slot 1
            elif container_type == 'THEU' and slot != 1:
                return False
                
            # FEU (40ft): must start at slot 1
            elif container_type == 'FEU' and slot != 1:
                return False
                
            # FFEU (45ft): must start at slot 1 and have next bay available
            elif container_type == 'FFEU':
                if slot != 1:
                    return False
                if bay >= self.num_bays:  # No next bay available
                    return False
        
        # Generate mask for container placement
        placement_mask = self.get_container_mask(position, container_type)
        
        # Check if placement would overlap with existing containers
        overlap = placement_mask & self.occupied_bitmap
        
        if overlap.any():
            return False  # Overlaps with existing containers
        
        return True
    
    def add_container(self, position: str, container: Any) -> bool:
        """
        Add a container to the yard.
        
        Args:
            position: Starting position string
            container: Container object to add
            
        Returns:
            True if container was added successfully, False otherwise
        """
        # Check if container can be placed
        if not self.is_container_placement_valid(position, container):
            return False
        
        # Get container type
        container_type = container.container_type
        
        # Get mask for container placement
        placement_mask = self.get_container_mask(position, container_type)
        
        # Update occupied bitmap
        self.occupied_bitmap |= placement_mask
        
        # Add to container registry
        self.container_registry[position] = container
        
        # If container has ID, add to lookup
        if hasattr(container, 'container_id'):
            self.container_id_to_position = getattr(self, 'container_id_to_position', {})
            self.container_id_to_position[container.container_id] = position
        
        return True
    
    def remove_container(self, position: str) -> Optional[Any]:
        """
        Remove a container from the yard.
        
        Args:
            position: Starting position string
            
        Returns:
            The removed container or None if no container was removed
        """
        # Check if position has a container
        if position not in self.container_registry:
            return None
        
        # Get the container and its type
        container = self.container_registry[position]
        container_type = container.container_type
        
        # Get mask for container placement
        placement_mask = self.get_container_mask(position, container_type)
        
        # Update occupied bitmap (using ~mask to clear bits)
        self.occupied_bitmap &= ~placement_mask
        
        # Remove from container registry
        del self.container_registry[position]
        
        # Remove from ID lookup if exists
        if hasattr(container, 'container_id') and hasattr(self, 'container_id_to_position'):
            if container.container_id in self.container_id_to_position:
                del self.container_id_to_position[container.container_id]
        
        return container
    
    def get_container(self, position: str) -> Optional[Any]:
        """
        Get the container at a position without removing it.
        
        Args:
            position: Position string
            
        Returns:
            Container object or None if no container found
        """
        return self.container_registry.get(position)
    
    def find_container(self, container_id: str) -> Optional[str]:
        """
        Find a container by ID.
        
        Args:
            container_id: ID of the container to find
            
        Returns:
            Position string or None if not found
        """
        # Check direct lookup first
        if hasattr(self, 'container_id_to_position') and container_id in self.container_id_to_position:
            return self.container_id_to_position[container_id]
        
        # Fallback to searching all positions
        for position, container in self.container_registry.items():
            if hasattr(container, 'container_id') and container.container_id == container_id:
                return position
        
        return None
    
    def get_proximity_mask(self, position: str, n: int, container_type: str = None) -> torch.Tensor:
        """
        Create a proximity mask for positions within n bays of the specified position.
        
        Args:
            position: Position string
            n: Number of bays in each direction
            container_type: Optional container type for special filtering
            
        Returns:
            Bitmap mask with 1s for positions within proximity
        """
        # Parse the position
        match = re.match(r'([A-Z])(\d+)\.(\d+)-T(\d+)', position)
        if not match:
            raise ValueError(f"Invalid position format: {position}")
            
        row, bay, slot, tier = match.groups()
        bay, slot, tier = int(bay), int(slot), int(tier)
        
        # Create cache key
        cache_key = (position, n, container_type)
        if cache_key in self.proximity_masks:
            return self.proximity_masks[cache_key]
        
        # Create empty mask
        mask = torch.zeros(self.total_words, dtype=torch.int64, device=self.device)
        
        # Calculate bay range
        min_bay = max(1, bay - n)
        max_bay = min(self.num_bays, bay + n)
        
        # For all positions within range
        for row_name in self.row_names:
            for bay_num in range(min_bay, max_bay + 1):
                for slot_num in range(1, self.slots_per_bay + 1):
                    # Use same tier as source position
                    pos = f"{row_name}{bay_num}.{slot_num}-T{tier}"
                    
                    # Skip the source position
                    if pos == position:
                        continue
                    
                    try:
                        bit_idx = self.encode_position(pos)
                        word_idx = bit_idx // 64
                        bit_offset = bit_idx % 64
                        mask[word_idx] |= mask[word_idx] | self.powers_of_2[bit_offset]
                    except ValueError:
                        continue
        
        # Apply container type filtering if specified
        if container_type:
            valid_positions_mask = torch.zeros_like(mask)
            
            # For each position in the proximity
            for word_idx in range(len(mask)):
                word = mask[word_idx].item()
                if word == 0:
                    continue
                
                # Process each set bit
                for bit_offset in range(64):
                    if (word & self.powers_of_2[bit_offset].item()) != 0:
                        bit_idx = word_idx * 64 + bit_offset
                        if bit_idx < self.total_slots:
                            try:
                                pos = self.decode_position(bit_idx)
                                # Check if container type can be placed here
                                if pos in self.valid_starts.get(container_type, set()):
                                    # Set bit in the filtered mask
                                    valid_positions_mask[word_idx] |= (1 << bit_offset)
                            except ValueError:
                                continue
            
            # Use the filtered mask
            mask = valid_positions_mask
            
            # Apply additional filtering based on special areas if needed
            if container_type == "Trailer":
                mask &= self.special_area_bitmaps['trailer']
            elif container_type == "Swap Body":
                mask &= self.special_area_bitmaps['swap_body']
        
        # Cache the result
        self.proximity_masks[cache_key] = mask
        
        return mask
    
    def calc_possible_moves(self, position: str, n: int) -> List[str]:
        """
        Calculate all possible positions a container can be moved to within n bays.
        
        Args:
            position: Starting position string
            n: Number of bays to consider in each direction
            
        Returns:
            List of valid destination positions
        """
        # Check if position has a container to move
        container = self.get_container(position)
        if container is None:
            return []
        
        # Get container type
        container_type = container.container_type
        
        # Get proximity mask
        proximity_mask = self.get_proximity_mask(position, n, container_type)
        
        # Initialize valid destinations list
        valid_destinations = []
        
        # Process each potential destination in the mask
        for word_idx in range(len(proximity_mask)):
            word = proximity_mask[word_idx].item()
            if word == 0:
                continue
            
            # Process each set bit
            for bit_offset in range(64):
                if (word >> bit_offset) & 1:
                    bit_idx = word_idx * 64 + bit_offset
                    if bit_idx < self.total_slots:
                        try:
                            dest_position = self.decode_position(bit_idx)
                            # Check if this is a valid starting position for this container
                            if self.is_container_placement_valid(dest_position, container):
                                valid_destinations.append(dest_position)
                        except ValueError:
                            continue
        
        return valid_destinations
    
    def batch_calc_possible_moves(self, positions: List[str], n: int) -> Dict[str, List[str]]:
        """
        Calculate possible moves for multiple containers in parallel.
        
        Args:
            positions: List of position strings to process
            n: Proximity range to use for all calculations
            
        Returns:
            Dictionary mapping each position to a list of valid destinations
        """
        # Filter to only positions with containers
        valid_positions = [pos for pos in positions if pos in self.container_registry]
        
        # Prepare storage for results
        all_moves = {}
        
        # Process each position
        for position in valid_positions:
            container = self.container_registry[position]
            valid_moves = self.calc_possible_moves(position, n)
            all_moves[position] = valid_moves
        
        return all_moves
    
    def visualize_bitmap(self, bitmap, title="Bitmap Visualization"):
        """
        Visualize a bitmap as a 2D grid.
        
        Args:
            bitmap: The bitmap to visualize
            title: Title for the visualization
        """
        import matplotlib.pyplot as plt
        
        # Create a 3D tensor representation (row, bay, slot+tier)
        # For each row and bay, we'll show slots*tiers as a block
        grid = np.zeros((self.num_rows, self.num_bays, self.slots_per_bay * self.max_tier_height), dtype=np.float32)
        
        # Fill the grid from the bitmap
        for bit_idx in range(self.total_slots):
            word_idx = bit_idx // 64
            bit_offset = bit_idx % 64
            
            # Check if there's a bit set at this index
            if word_idx < len(bitmap) and bit_offset < 64:
                if (bitmap[word_idx] & self.powers_of_2[bit_offset]) != 0:
                # Decode the position
                    pos = self.decode_position(bit_idx)
                    match = re.match(r'([A-Z])(\d+)\.(\d+)-T(\d+)', pos)
                    if match:
                        row, bay, slot, tier = match.groups()
                        row_idx = self.row_names.index(row)
                        bay_idx = int(bay) - 1
                        slot_idx = int(slot) - 1
                        tier_idx = int(tier) - 1
                        
                        # Convert to flattened index (slot first, then tier)
                        flattened_idx = slot_idx * self.max_tier_height + tier_idx
                        
                        # Set the grid cell
                        grid[row_idx, bay_idx, flattened_idx] = 1
        
        # Create the figure
        fig = plt.figure(figsize=(15, 8))
        ax = fig.add_subplot(111)
        
        # Visualize as a 2D heatmap (flattening the slots+tiers dimension)
        occupancy = np.sum(grid, axis=2) / (self.slots_per_bay * self.max_tier_height)
        plt.imshow(occupancy, cmap='YlOrRd', interpolation='none')
        
        # Add labels
        plt.xlabel('Bay')
        plt.ylabel('Row')
        plt.title(title)
        
        # Add row names on y-axis
        plt.yticks(range(self.num_rows), self.row_names)
        
        # Add bay numbers on x-axis
        plt.xticks(range(self.num_bays), range(1, self.num_bays + 1))
        
        # Add grid
        plt.grid(False)
        
        # Add colorbar
        plt.colorbar(label='Occupancy Rate')
        
        plt.tight_layout()
        return fig, ax
    
    def visualize_3d(self, show_container_types=True, figsize=(15, 10)):
        """
        Create a 3D visualization of the container yard.
        
        Args:
            show_container_types: Whether to color containers by type
            figsize: Figure size
            
        Returns:
            Figure and axis objects
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import numpy as np
        
        # Create figure and 3D axis
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Container type colors
        container_colors = {
            'TWEU': 'royalblue',       # 20ft
            'THEU': 'purple',          # 30ft
            'FEU': 'lightseagreen',    # 40ft
            'FFEU': 'teal',            # 45ft
            'Trailer': 'darkred',      # Trailer
            'Swap Body': 'goldenrod',  # Swap Body
            'default': 'gray'          # Default
        }
        
        # Goods type edge colors
        goods_edge_colors = {
            'Regular': 'black',
            'Reefer': 'blue',
            'Dangerous': 'red'
        }
        
        # Legend elements
        from matplotlib.patches import Patch
        legend_elements = []
        
        # Containers to visualize
        for position, container in self.container_registry.items():
            # Get container properties
            container_type = getattr(container, 'container_type', 'default')
            goods_type = getattr(container, 'goods_type', 'Regular')
            
            # Get position coordinates
            match = re.match(r'([A-Z])(\d+)\.(\d+)-T(\d+)', position)
            if match:
                row, bay, slot, tier = match.groups()
                row_idx = self.row_names.index(row)
                bay_idx = int(bay) - 1
                slot_idx = int(slot) - 1
                tier_idx = int(tier) - 1
                
                # Get container dimensions
                if container_type == 'TWEU':  # 20ft (2 slots)
                    slots = 2
                elif container_type == 'THEU':  # 30ft (3 slots)
                    slots = 3
                elif container_type == 'FEU':  # 40ft (4 slots)
                    slots = 4
                elif container_type == 'FFEU':  # 45ft (5 slots)
                    slots = 5
                else:  # Trailer, Swap Body, etc.
                    slots = 4  # Default to full bay
                
                # Get colors
                color = container_colors.get(container_type, container_colors['default'])
                edge_color = goods_edge_colors.get(goods_type, goods_edge_colors['Regular'])
                
                # Add to legend if not already present
                legend_key = f"{container_type} - {goods_type}"
                if not any(le.get_label() == legend_key for le in legend_elements):
                    legend_elements.append(
                        Patch(facecolor=color, edgecolor=edge_color, label=legend_key)
                    )
                
                # Draw the container based on its type
                x = bay_idx
                y = row_idx
                z = tier_idx
                
                # For 20ft containers (TWEU), place in appropriate half of bay
                if container_type == 'TWEU':
                    # Check if slot is 1 or 3
                    if int(slot) == 1:
                        x_length = 0.5  # Half bay
                    elif int(slot) == 3:
                        x = bay_idx + 0.5  # Second half of bay
                        x_length = 0.5  # Half bay
                    else:
                        continue  # Invalid slot
                elif container_type == 'THEU':
                    x_length = 0.75  # 3/4 of a bay
                elif container_type == 'FEU':
                    x_length = 1.0  # Full bay
                elif container_type == 'FFEU':
                    # FFEU spans into next bay
                    x_length = 1.25  # 5/4 of a bay
                else:
                    x_length = 1.0  # Full bay for other types
                
                # Draw as 3D box
                dx = x_length
                dy = 0.8  # Slightly less than 1 to see gaps
                dz = 0.8  # Slightly less than 1 to see gaps
                
                # Draw the container
                ax.bar3d(
                    x, y, z,          # Position
                    dx, dy, dz,       # Dimensions
                    color=color,
                    shade=True,
                    alpha=0.8,
                    edgecolor=edge_color,
                    linewidth=0.5
                )
        
        # Set labels and limits
        ax.set_xlabel('Bay')
        ax.set_ylabel('Row')
        ax.set_zlabel('Tier')
        
        # Set row labels
        ax.set_yticks(range(self.num_rows))
        ax.set_yticklabels(self.row_names)
        
        # Set bay labels
        ax.set_xticks(range(0, self.num_bays, 2))
        ax.set_xticklabels(range(1, self.num_bays + 1, 2))
        
        # Set tier labels
        ax.set_zticks(range(self.max_tier_height))
        ax.set_zticklabels(range(1, self.max_tier_height + 1))
        
        # Set limits
        ax.set_xlim(-0.5, self.num_bays - 0.5)
        ax.set_ylim(-0.5, self.num_rows - 0.5)
        ax.set_zlim(-0.5, self.max_tier_height - 0.5)
        
        # Set title and legend
        ax.set_title('3D Container Yard Visualization')
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
        
        # Set view angle
        ax.view_init(elev=30, azim=225)
        
        plt.tight_layout()
        return fig, ax
    
    def clear(self):
        """Clear the entire storage yard."""
        # Reset bitmap
        self.occupied_bitmap.zero_()
        
        # Clear container registry
        self.container_registry.clear()
        
        # Clear container ID mapping
        if hasattr(self, 'container_id_to_position'):
            self.container_id_to_position.clear()

    def __str__(self):
        """String representation of the storage yard."""
        container_count = len(self.container_registry)
        
        # Count containers by row
        row_counts = {}
        for position in self.container_registry:
            match = re.match(r'([A-Z])(\d+)\.(\d+)-T(\d+)', position)
            if match:
                row = match.group(1)
                if row in row_counts:
                    row_counts[row] += 1
                else:
                    row_counts[row] = 1
        
        # Format string
        lines = []
        lines.append(f"Slot-Tier Bitmap Yard: {self.num_rows} rows, {self.num_bays} bays, {container_count} containers")
        
        for row in self.row_names:
            count = row_counts.get(row, 0)
            lines.append(f"Row {row}: {count} containers")
        
        return '\n'.join(lines)