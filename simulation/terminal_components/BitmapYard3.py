import numpy as np
import torch
import time
from typing import Dict, List, Tuple, Optional, Any, Set

class OptimizedSlotTierBitmapYard:
    """
    Highly optimized GPU-accelerated bitmap representation of a container storage yard
    with mathematical position encoding and vectorized operations.
    
    Key optimizations:
    - Mathematical position encoding (no strings/regex)
    - GPU-vectorized operations
    - O(1) container lookups
    - Lazy initialization
    - Batch processing capabilities
    """
    
    def __init__(self, 
                num_rows: int, 
                num_bays: int, 
                max_tier_height: int = 5,
                row_names: List[str] = None,
                special_areas: Dict[str, List[Tuple[str, int, int]]] = None,
                device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the optimized slot-tier bitmap storage yard.
        
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
        
        # Initialize row names
        if row_names is None:
            self.row_names = [chr(65 + i) for i in range(num_rows)]  # A, B, C, ...
        else:
            self.row_names = row_names[:num_rows]
        
        # Create row name to index mapping for string interface compatibility
        self.row_name_to_idx = {name: idx for idx, name in enumerate(self.row_names)}
        
        # Calculate total slots and bitmap size
        self.total_slots = num_rows * num_bays * self.slots_per_bay * max_tier_height
        self.total_words = (self.total_slots + 63) // 64
        
        # Pre-compute powers of 2 for efficient bit operations
        self.powers_of_2 = torch.pow(2, torch.arange(64, dtype=torch.int64, device=device))
        
        # Initialize main occupied bitmap
        self.occupied_bitmap = torch.zeros(self.total_words, dtype=torch.int64, device=device)
        
        # Initialize special area configuration (store as numerical ranges)
        self.special_areas_numerical = self._convert_special_areas_to_numerical(special_areas)
        
        # Initialize special area bitmaps using vectorized operations  
        self.special_area_bitmaps = self._initialize_special_area_bitmaps_vectorized()
        
        # Container registries - using numerical indexing for O(1) operations
        self.container_by_bit_idx = {}  # bit_idx -> container (O(1) lookup)
        self.container_id_to_bit_idx = {}  # container_id -> bit_idx (O(1) search)
        
        # Pre-compute container type slot requirements for vectorized operations
        self.container_slot_requirements = {
            'TWEU': torch.tensor([0, 1], device=device),  # 2 consecutive slots
            'THEU': torch.tensor([0, 1, 2], device=device),  # 3 consecutive slots  
            'FEU': torch.tensor([0, 1, 2, 3], device=device),  # 4 slots (full bay)
            'FFEU': torch.tensor([0, 1, 2, 3, 4], device=device),  # 5 slots (spans bays)
            'Trailer': torch.tensor([0, 1, 2, 3], device=device),  # Full bay
            'Swap Body': torch.tensor([0, 1, 2, 3], device=device),  # Full bay
        }
        
        # Valid starting slots for each container type (for validation)
        self.valid_starting_slots = {
            'TWEU': [1, 3],  # Can start at slot 1 or 3
            'THEU': [1],     # Must start at slot 1
            'FEU': [1],      # Must start at slot 1
            'FFEU': [1],     # Must start at slot 1
            'Trailer': [1],  # Must start at slot 1
            'Swap Body': [1] # Must start at slot 1
        }
        
        # Cache for frequently computed masks
        self.mask_cache = {}
        self.proximity_cache = {}
    
    def encode_position(self, row_idx: int, bay: int, slot: int, tier: int) -> int:
        """
        Convert position coordinates to bit index using pure mathematics.
        O(1) operation - no strings, no regex, no dictionaries.
        
        Args:
            row_idx: 0-based row index (0=A, 1=B, etc.)
            bay: 1-based bay number
            slot: 1-based slot number (1-4)
            tier: 1-based tier number (1-5)
            
        Returns:
            Bit index for this position
        """
        return (((row_idx * self.num_bays + (bay - 1)) * self.slots_per_bay + (slot - 1)) 
                * self.max_tier_height + (tier - 1))
    
    def decode_position(self, bit_idx: int) -> Tuple[int, int, int, int]:
        """
        Convert bit index to position coordinates using pure mathematics.
        O(1) operation.
        
        Args:
            bit_idx: Bit index to decode
            
        Returns:
            Tuple of (row_idx, bay, slot, tier)
        """
        tier = (bit_idx % self.max_tier_height) + 1
        slot = ((bit_idx // self.max_tier_height) % self.slots_per_bay) + 1
        bay = (((bit_idx // self.max_tier_height) // self.slots_per_bay) % self.num_bays) + 1
        row_idx = ((bit_idx // self.max_tier_height) // self.slots_per_bay) // self.num_bays
        return row_idx, bay, slot, tier
    
    def batch_encode_positions(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Vectorized encoding of multiple positions using GPU operations.
        
        Args:
            positions: Tensor of shape (N, 4) with [row_idx, bay, slot, tier]
            
        Returns:
            Tensor of bit indices
        """
        row_indices = positions[:, 0]
        bays = positions[:, 1] - 1  # Convert to 0-based
        slots = positions[:, 2] - 1  # Convert to 0-based  
        tiers = positions[:, 3] - 1  # Convert to 0-based
        
        # Vectorized encoding
        bit_indices = (((row_indices * self.num_bays + bays) * self.slots_per_bay + slots) 
                       * self.max_tier_height + tiers)
        return bit_indices
    
    def _convert_special_areas_to_numerical(self, special_areas: Dict) -> Dict:
        """Convert string-based special areas to numerical ranges."""
        if special_areas is None:
            return {
                'reefer': [(0, 1, 5), (4, 1, 5)],  # Row A and E, bays 1-5
                'dangerous': [(2, 28, 36)],  # Row C, bays 28-36
                'trailer': [(4, 1, 58)],     # Row E, bays 1-58
                'swap_body': [(4, 1, 58)]    # Row E, bays 1-58
            }
        
        numerical_areas = {}
        for area_type, areas in special_areas.items():
            numerical_areas[area_type] = []
            for area_row, start_bay, end_bay in areas:
                if area_row in self.row_name_to_idx:
                    row_idx = self.row_name_to_idx[area_row]
                    numerical_areas[area_type].append((row_idx, start_bay, end_bay))
        
        return numerical_areas
    
    def _initialize_special_area_bitmaps_vectorized(self) -> Dict[str, torch.Tensor]:
        """Initialize special area bitmaps using vectorized operations."""
        special_bitmaps = {}
        
        for area_type, areas in self.special_areas_numerical.items():
            bitmap = torch.zeros(self.total_words, dtype=torch.int64, device=self.device)
            
            for row_idx, start_bay, end_bay in areas:
                if 0 <= row_idx < self.num_rows:
                    # Generate all positions for this area using vectorization
                    bays = torch.arange(start_bay, end_bay + 1, device=self.device)
                    slots = torch.arange(1, self.slots_per_bay + 1, device=self.device)
                    tiers = torch.arange(1, self.max_tier_height + 1, device=self.device)
                    
                    # Create meshgrid of all combinations
                    bay_grid, slot_grid, tier_grid = torch.meshgrid(bays, slots, tiers, indexing='ij')
                    
                    # Flatten and create position tensor
                    num_positions = bay_grid.numel()
                    positions = torch.stack([
                        torch.full((num_positions,), row_idx, device=self.device),
                        bay_grid.flatten(),
                        slot_grid.flatten(), 
                        tier_grid.flatten()
                    ], dim=1)
                    
                    # Batch encode all positions
                    bit_indices = self.batch_encode_positions(positions)
                    
                    # Set bits efficiently using scatter operations
                    word_indices = bit_indices // 64
                    bit_offsets = bit_indices % 64
                    
                    # Use scatter_add to set multiple bits efficiently
                    bitmap.scatter_add_(0, word_indices, self.powers_of_2[bit_offsets])
            
            special_bitmaps[area_type] = bitmap
        
        return special_bitmaps
    
    def get_container_mask_vectorized(self, row_idx: int, bay: int, slot: int, tier: int, 
                                     container_type: str) -> torch.Tensor:
        """
        Generate container placement mask using GPU-vectorized operations.
        
        Args:
            row_idx: 0-based row index
            bay: 1-based bay number
            slot: 1-based slot number
            tier: 1-based tier number
            container_type: Type of container
            
        Returns:
            Bitmap mask with occupied slots marked
        """
        # Check cache first
        cache_key = (row_idx, bay, slot, tier, container_type)
        if cache_key in self.mask_cache:
            return self.mask_cache[cache_key]
        
        # Get slot requirements for this container type
        if container_type not in self.container_slot_requirements:
            return torch.zeros(self.total_words, dtype=torch.int64, device=self.device)
        
        slot_offsets = self.container_slot_requirements[container_type]
        
        # Validate starting position
        if slot not in self.valid_starting_slots.get(container_type, []):
            return torch.zeros(self.total_words, dtype=torch.int64, device=self.device)
        
        # Calculate actual slots this container will occupy
        if container_type == 'FFEU' and bay < self.num_bays:
            # FFEU spans into next bay - handle specially
            # 4 slots in current bay + 1 slot in next bay
            current_bay_slots = slot_offsets[:4] + (slot - 1)
            next_bay_slots = torch.tensor([0], device=self.device)  # First slot of next bay
            
            # Create positions for current bay
            current_positions = torch.stack([
                torch.full((4,), row_idx, device=self.device),
                torch.full((4,), bay, device=self.device),
                current_bay_slots + 1,  # Convert to 1-based
                torch.full((4,), tier, device=self.device)
            ], dim=1)
            
            # Create position for next bay
            next_positions = torch.stack([
                torch.tensor([row_idx], device=self.device),
                torch.tensor([bay + 1], device=self.device),
                torch.tensor([1], device=self.device),  # First slot
                torch.tensor([tier], device=self.device)
            ], dim=1)
            
            # Combine positions
            all_positions = torch.cat([current_positions, next_positions], dim=0)
        else:
            # Regular container - all slots in same bay
            actual_slots = slot_offsets + (slot - 1)
            
            # Filter slots that are within bay bounds
            valid_mask = actual_slots < self.slots_per_bay
            if not valid_mask.all():
                # Container doesn't fit in bay
                return torch.zeros(self.total_words, dtype=torch.int64, device=self.device)
            
            actual_slots = actual_slots[valid_mask]
            num_slots = len(actual_slots)
            
            # Create position tensor
            all_positions = torch.stack([
                torch.full((num_slots,), row_idx, device=self.device),
                torch.full((num_slots,), bay, device=self.device), 
                actual_slots + 1,  # Convert to 1-based
                torch.full((num_slots,), tier, device=self.device)
            ], dim=1)
        
        # Batch encode all positions
        bit_indices = self.batch_encode_positions(all_positions)
        
        # Create mask using scatter operations
        mask = torch.zeros(self.total_words, dtype=torch.int64, device=self.device)
        word_indices = bit_indices // 64
        bit_offsets = bit_indices % 64
        
        # Set bits efficiently
        mask.scatter_add_(0, word_indices, self.powers_of_2[bit_offsets])
        
        # Cache the result
        self.mask_cache[cache_key] = mask
        
        return mask
    
    def is_position_occupied(self, row_idx: int, bay: int, slot: int, tier: int) -> bool:
        """Check if a position is occupied using O(1) bit operation."""
        bit_idx = self.encode_position(row_idx, bay, slot, tier)
        word_idx = bit_idx // 64
        bit_offset = bit_idx % 64
        
        if word_idx >= len(self.occupied_bitmap):
            return False
            
        return bool((self.occupied_bitmap[word_idx] & self.powers_of_2[bit_offset]) != 0)
    
    def is_position_in_special_area(self, row_idx: int, bay: int, slot: int, tier: int, 
                                  area_type: str) -> bool:
        """Check if position is in special area using O(1) bit operation."""
        if area_type not in self.special_area_bitmaps:
            return False
            
        bit_idx = self.encode_position(row_idx, bay, slot, tier)
        word_idx = bit_idx // 64
        bit_offset = bit_idx % 64
        
        if word_idx >= len(self.special_area_bitmaps[area_type]):
            return False
            
        return bool((self.special_area_bitmaps[area_type][word_idx] & self.powers_of_2[bit_offset]) != 0)
    
    def is_container_placement_valid(self, row_idx: int, bay: int, slot: int, tier: int, 
                                   container: Any) -> bool:
        """
        Validate container placement using optimized checks.
        
        Args:
            row_idx: 0-based row index
            bay: 1-based bay number
            slot: 1-based slot number
            tier: 1-based tier number
            container: Container object
            
        Returns:
            True if placement is valid
        """
        container_type = getattr(container, 'container_type', 'Unknown')
        goods_type = getattr(container, 'goods_type', 'Regular')
        
        # Quick bounds check
        if not (0 <= row_idx < self.num_rows and 1 <= bay <= self.num_bays and
                1 <= slot <= self.slots_per_bay and 1 <= tier <= self.max_tier_height):
            return False
        
        # Container type validation
        if container_type not in self.container_slot_requirements:
            return False
        
        # Starting slot validation
        if slot not in self.valid_starting_slots.get(container_type, []):
            return False
        
        # Special area constraints
        if container_type == 'Trailer':
            if not self.is_position_in_special_area(row_idx, bay, slot, tier, 'trailer'):
                return False
            if tier != 1:  # Trailers can't be stacked
                return False
        elif container_type == 'Swap Body':
            if not self.is_position_in_special_area(row_idx, bay, slot, tier, 'swap_body'):
                return False
        
        # Goods type constraints
        if goods_type == 'Reefer':
            if not self.is_position_in_special_area(row_idx, bay, slot, tier, 'reefer'):
                return False
        elif goods_type == 'Dangerous':
            if not self.is_position_in_special_area(row_idx, bay, slot, tier, 'dangerous'):
                return False
        
        # Stacking validation for tier > 1
        if tier > 1:
            container_below = self.get_container(row_idx, bay, slot, tier - 1)
            if container_below is None:
                return False
            
            # Check stacking compatibility
            if (hasattr(container, 'can_stack_with') and 
                not container.can_stack_with(container_below)):
                return False
        
        # Check if placement mask overlaps with occupied positions
        placement_mask = self.get_container_mask_vectorized(row_idx, bay, slot, tier, container_type)
        overlap = placement_mask & self.occupied_bitmap
        
        return not overlap.any().item()
    
    def add_container(self, row_idx: int, bay: int, slot: int, tier: int, container: Any) -> bool:
        """
        Add container using optimized operations.
        
        Args:
            row_idx: 0-based row index  
            bay: 1-based bay number
            slot: 1-based slot number
            tier: 1-based tier number
            container: Container object
            
        Returns:
            True if container was added successfully
        """
        # Validate placement
        if not self.is_container_placement_valid(row_idx, bay, slot, tier, container):
            return False
        
        container_type = container.container_type
        
        # Get placement mask
        placement_mask = self.get_container_mask_vectorized(row_idx, bay, slot, tier, container_type)
        
        # Update occupied bitmap
        self.occupied_bitmap |= placement_mask
        
        # Add to registries using bit index for O(1) operations
        start_bit_idx = self.encode_position(row_idx, bay, slot, tier)
        self.container_by_bit_idx[start_bit_idx] = container
        
        # Add to ID lookup if container has ID
        if hasattr(container, 'container_id'):
            self.container_id_to_bit_idx[container.container_id] = start_bit_idx
        
        return True
    
    def remove_container(self, row_idx: int, bay: int, slot: int, tier: int) -> Optional[Any]:
        """
        Remove container using optimized operations.
        
        Args:
            row_idx: 0-based row index
            bay: 1-based bay number
            slot: 1-based slot number  
            tier: 1-based tier number
            
        Returns:
            Removed container or None
        """
        start_bit_idx = self.encode_position(row_idx, bay, slot, tier)
        
        # Check if container exists at this position
        if start_bit_idx not in self.container_by_bit_idx:
            return None
        
        container = self.container_by_bit_idx[start_bit_idx]
        container_type = container.container_type
        
        # Get placement mask
        placement_mask = self.get_container_mask_vectorized(row_idx, bay, slot, tier, container_type)
        
        # Update occupied bitmap (clear bits)
        self.occupied_bitmap &= ~placement_mask
        
        # Remove from registries
        del self.container_by_bit_idx[start_bit_idx]
        
        # Remove from ID lookup if exists
        if hasattr(container, 'container_id') and container.container_id in self.container_id_to_bit_idx:
            del self.container_id_to_bit_idx[container.container_id]
        
        return container
    
    def get_container(self, row_idx: int, bay: int, slot: int, tier: int) -> Optional[Any]:
        """
        Get container using O(1) lookup.
        
        Args:
            row_idx: 0-based row index
            bay: 1-based bay number
            slot: 1-based slot number
            tier: 1-based tier number
            
        Returns:
            Container at position or None
        """
        # First check if this exact position has a container starting here
        bit_idx = self.encode_position(row_idx, bay, slot, tier)
        if bit_idx in self.container_by_bit_idx:
            return self.container_by_bit_idx[bit_idx]
        
        # If not, check if this slot is occupied (might be part of multi-slot container)
        if not self.is_position_occupied(row_idx, bay, slot, tier):
            return None
        
        # Search for multi-slot containers that might occupy this position
        # Check all possible starting positions that could reach this slot
        for start_slot in [1, 3]:  # Possible starting slots
            if start_slot <= slot:
                start_bit_idx = self.encode_position(row_idx, bay, start_slot, tier)
                if start_bit_idx in self.container_by_bit_idx:
                    container = self.container_by_bit_idx[start_bit_idx]
                    container_type = container.container_type
                    
                    # Check if this container's mask covers our target slot
                    mask = self.get_container_mask_vectorized(row_idx, bay, start_slot, tier, container_type)
                    target_bit_idx = self.encode_position(row_idx, bay, slot, tier)
                    word_idx = target_bit_idx // 64
                    bit_offset = target_bit_idx % 64
                    
                    if word_idx < len(mask) and (mask[word_idx] & self.powers_of_2[bit_offset]) != 0:
                        return container
        
        # Check FFEU containers from previous bay (they can span into this bay)
        if bay > 1:
            prev_bay_bit_idx = self.encode_position(row_idx, bay - 1, 1, tier)
            if prev_bay_bit_idx in self.container_by_bit_idx:
                container = self.container_by_bit_idx[prev_bay_bit_idx]
                if container.container_type == 'FFEU' and slot == 1:
                    return container
        
        return None
    
    def find_container_by_id(self, container_id: str) -> Optional[Tuple[int, int, int, int]]:
        """
        Find container by ID using O(1) lookup.
        
        Args:
            container_id: Container ID to search for
            
        Returns:
            Tuple of (row_idx, bay, slot, tier) or None
        """
        if container_id in self.container_id_to_bit_idx:
            bit_idx = self.container_id_to_bit_idx[container_id]
            return self.decode_position(bit_idx)
        return None
    
    def get_proximity_mask_vectorized(self, row_idx: int, bay: int, tier: int, n: int, 
                                     container_type: str = None) -> torch.Tensor:
        """
        Generate proximity mask using GPU-vectorized operations.
        
        Args:
            row_idx: Center row index
            bay: Center bay
            tier: Target tier 
            n: Proximity range in bays
            container_type: Optional container type filter
            
        Returns:
            Bitmap mask with valid proximity positions
        """
        cache_key = (row_idx, bay, tier, n, container_type)
        if cache_key in self.proximity_cache:
            return self.proximity_cache[cache_key]
        
        # Calculate bay range
        min_bay = max(1, bay - n)
        max_bay = min(self.num_bays, bay + n)
        
        # Generate all positions within proximity using vectorization
        bays = torch.arange(min_bay, max_bay + 1, device=self.device)
        rows = torch.arange(self.num_rows, device=self.device)
        slots = torch.arange(1, self.slots_per_bay + 1, device=self.device)
        
        # Create meshgrid for all combinations
        bay_grid, row_grid, slot_grid = torch.meshgrid(bays, rows, slots, indexing='ij')
        
        # Flatten and create position tensor
        num_positions = bay_grid.numel()
        positions = torch.stack([
            row_grid.flatten(),
            bay_grid.flatten(),
            slot_grid.flatten(),
            torch.full((num_positions,), tier, device=self.device)
        ], dim=1)
        
        # Remove center position
        center_mask = ((positions[:, 0] != row_idx) | 
                      (positions[:, 1] != bay) | 
                      (positions[:, 2] != 1))  # Simplified check
        positions = positions[center_mask]
        
        # Batch encode positions
        bit_indices = self.batch_encode_positions(positions)
        
        # Create proximity mask
        mask = torch.zeros(self.total_words, dtype=torch.int64, device=self.device)
        word_indices = bit_indices // 64
        bit_offsets = bit_indices % 64
        
        # Filter for valid bit indices
        valid_mask = bit_indices < self.total_slots
        word_indices = word_indices[valid_mask]
        bit_offsets = bit_offsets[valid_mask]
        
        mask.scatter_add_(0, word_indices, self.powers_of_2[bit_offsets])
        
        # Apply container type filtering if specified
        if container_type and container_type in self.container_slot_requirements:
            # Filter for valid starting positions for this container type
            valid_starting_slots = torch.tensor(self.valid_starting_slots[container_type], device=self.device)
            
            # Create filter mask for valid starting slots
            valid_positions = positions[valid_mask]
            slot_filter = torch.isin(valid_positions[:, 2], valid_starting_slots)
            
            if slot_filter.any():
                filtered_word_indices = word_indices[slot_filter]
                filtered_bit_offsets = bit_offsets[slot_filter]
                
                filtered_mask = torch.zeros(self.total_words, dtype=torch.int64, device=self.device)
                filtered_mask.scatter_add_(0, filtered_word_indices, self.powers_of_2[filtered_bit_offsets])
                mask = filtered_mask
        
        # Cache result
        self.proximity_cache[cache_key] = mask
        return mask
    
    def calc_possible_moves_vectorized(self, row_idx: int, bay: int, slot: int, tier: int, 
                                     n: int = 5) -> List[Tuple[int, int, int, int]]:
        """
        Calculate possible moves using GPU-vectorized operations.
        
        Args:
            row_idx: Source row index
            bay: Source bay
            slot: Source slot
            tier: Source tier
            n: Proximity range
            
        Returns:
            List of valid destination coordinates
        """
        # Get container at source position
        container = self.get_container(row_idx, bay, slot, tier)
        if container is None:
            return []
        
        container_type = container.container_type
        
        # Get proximity mask
        proximity_mask = self.get_proximity_mask_vectorized(row_idx, bay, tier, n, container_type)
        
        # Find all set bits efficiently using torch operations
        valid_destinations = []
        
        # Process each word
        for word_idx in range(len(proximity_mask)):
            word = proximity_mask[word_idx].item()
            if word == 0:
                continue
            
            # Find set bits in this word
            bit_positions = torch.nonzero(proximity_mask[word_idx:word_idx+1].view(-1), as_tuple=False).flatten()
            
            for bit_offset in bit_positions:
                bit_idx = word_idx * 64 + bit_offset.item()
                if bit_idx < self.total_slots:
                    dest_row_idx, dest_bay, dest_slot, dest_tier = self.decode_position(bit_idx)
                    
                    # Validate this destination
                    if self.is_container_placement_valid(dest_row_idx, dest_bay, dest_slot, dest_tier, container):
                        valid_destinations.append((dest_row_idx, dest_bay, dest_slot, dest_tier))
        
        return valid_destinations
    
    def batch_calc_possible_moves(self, positions: List[Tuple[int, int, int, int]], n: int = 5) -> Dict:
        """
        Calculate possible moves for multiple containers in parallel.
        
        Args:
            positions: List of (row_idx, bay, slot, tier) tuples
            n: Proximity range
            
        Returns:
            Dictionary mapping positions to valid destinations
        """
        results = {}
        
        # Process each position (could be further optimized with full batching)
        for pos in positions:
            row_idx, bay, slot, tier = pos
            valid_moves = self.calc_possible_moves_vectorized(row_idx, bay, slot, tier, n)
            results[pos] = valid_moves
        
        return results
    
    # String interface compatibility methods
    def parse_string_position(self, position: str) -> Tuple[int, int, int, int]:
        """Parse string position to numerical coordinates."""
        import re
        match = re.match(r'([A-Z])(\d+)\.(\d+)-T(\d+)', position)
        if not match:
            raise ValueError(f"Invalid position format: {position}")
        
        row, bay, slot, tier = match.groups()
        row_idx = self.row_name_to_idx[row]
        return row_idx, int(bay), int(slot), int(tier)
    
    def format_position_string(self, row_idx: int, bay: int, slot: int, tier: int) -> str:
        """Format numerical coordinates to string position."""
        row_name = self.row_names[row_idx]
        return f"{row_name}{bay}.{slot}-T{tier}"
    
    def add_container_string(self, position: str, container: Any) -> bool:
        """Add container using string position (compatibility method)."""
        row_idx, bay, slot, tier = self.parse_string_position(position)
        return self.add_container(row_idx, bay, slot, tier, container)
    
    def get_container_string(self, position: str) -> Optional[Any]:
        """Get container using string position (compatibility method)."""
        row_idx, bay, slot, tier = self.parse_string_position(position)
        return self.get_container(row_idx, bay, slot, tier)
    
    def clear(self):
        """Clear the entire storage yard."""
        self.occupied_bitmap.zero_()
        self.container_by_bit_idx.clear()
        self.container_id_to_bit_idx.clear()
        self.mask_cache.clear()
        self.proximity_cache.clear()
    
    def get_container_count(self) -> int:
        """Get total number of containers in the yard."""
        return len(self.container_by_bit_idx)
    
    def get_occupancy_rate(self) -> float:
        """Get overall occupancy rate of the yard."""
        occupied_bits = self.occupied_bitmap.count_nonzero().item() * 64  # Approximate
        return min(1.0, occupied_bits / self.total_slots)
    
    def __str__(self):
        """String representation of the storage yard."""
        container_count = self.get_container_count()
        occupancy = self.get_occupancy_rate() * 100
        
        return (f"Optimized SlotTier Bitmap Yard: {self.num_rows} rows, {self.num_bays} bays, "
                f"{container_count} containers ({occupancy:.1f}% occupancy)")