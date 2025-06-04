import numpy as np
import torch
import time
from typing import Dict, List, Tuple, Optional, Any
import re


class BitmapStorageYard:
    """
    GPU-accelerated bitmap representation of a container storage yard.
    Uses bit operations for efficient container placement and movement calculations.
    """
    
    def __init__(self, 
                num_rows: int, 
                num_bays: int, 
                max_tier_height: int = 5,
                row_names: List[str] = None,
                special_areas: Dict[str, List[Tuple[str, int, int]]] = None,
                device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the bitmap storage yard.
        
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
        self.max_tier_height = max_tier_height
        self.device = device
        
        # Initialize row names if not provided
        if row_names is None:
            self.row_names = [chr(65 + i) for i in range(num_rows)]  # A, B, C, ...
        else:
            self.row_names = row_names[:num_rows]
        
        # Calculate bitmap dimensions
        # We'll use a 64-bit alignment for efficient operations
        self.bits_per_row = ((num_bays + 63) // 64) * 64
        self.total_bits = self.num_rows * self.bits_per_row
        
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
        self.tier_bitmaps = [torch.zeros(self.total_bits // 64, dtype=torch.int64, device=device) 
                            for _ in range(max_tier_height)]
        
        # Initialize bitmaps for special areas
        self.special_area_bitmaps = {
            'reefer': torch.zeros(self.total_bits // 64, dtype=torch.int64, device=device),
            'dangerous': torch.zeros(self.total_bits // 64, dtype=torch.int64, device=device),
            'trailer': torch.zeros(self.total_bits // 64, dtype=torch.int64, device=device),
            'swap_body': torch.zeros(self.total_bits // 64, dtype=torch.int64, device=device)
        }
        
        # Initialize position mapping
        self.position_to_bit = {}  # Maps position string to bit index
        self.bit_to_position = {}  # Maps bit index to position string
        self._build_position_mapping()
        
        # Initialize special area bitmaps
        self._initialize_special_area_bitmaps()
        
        # Container registry for efficient lookup
        self.container_registry = {}  # position -> {tier -> container}
        
        # Pre-compute proximity masks for efficient proximity searching
        self.proximity_masks = {}  # (bit_idx, n) -> proximity mask
        
        # Create an "occupied" bitmap that combines all tiers for quick checks
        self.update_occupied_bitmap()
    
    def _build_position_mapping(self):
        """Build mapping between position strings and bit indices."""
        for row_idx, row in enumerate(self.row_names):
            row_offset = row_idx * self.bits_per_row
            for bay in range(1, self.num_bays + 1):
                position = f"{row}{bay}"
                bit_idx = row_offset + (bay - 1)
                self.position_to_bit[position] = bit_idx
                self.bit_to_position[bit_idx] = position
    
    def _initialize_special_area_bitmaps(self):
        """Initialize bitmaps for special areas based on configuration."""
        # Initialize special area bitmaps
        for area_type, areas in self.special_areas.items():
            bitmap = self.special_area_bitmaps[area_type]
            for area_row, start_bay, end_bay in areas:
                if area_row in self.row_names:
                    row_idx = self.row_names.index(area_row)
                    row_offset = row_idx * self.bits_per_row
                    for bay in range(start_bay, end_bay + 1):
                        if 1 <= bay <= self.num_bays:
                            bit_idx = row_offset + (bay - 1)
                            word_idx = bit_idx // 64
                            bit_offset = bit_idx % 64
                            bitmap[word_idx] |= (1 << bit_offset)
    
    def update_occupied_bitmap(self):
        """Update the combined occupied bitmap based on all tiers."""
        # Create a bitmap that has a 1 wherever any tier has a container
        self.occupied_bitmap = torch.zeros(self.total_bits // 64, dtype=torch.int64, device=self.device)
        for tier_bitmap in self.tier_bitmaps:
            self.occupied_bitmap |= tier_bitmap
    
    def encode_position(self, position: str) -> int:
        """Convert a position string to a bit index."""
        if position in self.position_to_bit:
            return self.position_to_bit[position]
        
        # Parse position if not in mapping
        if len(position) >= 2 and position[0].isalpha() and position[1:].isdigit():
            row = position[0]
            bay = int(position[1:])
            if row in self.row_names and 1 <= bay <= self.num_bays:
                row_idx = self.row_names.index(row)
                return row_idx * self.bits_per_row + (bay - 1)
        
        raise ValueError(f"Invalid position: {position}")
    
    def decode_position(self, bit_idx: int) -> str:
        """Convert a bit index to a position string."""
        if bit_idx in self.bit_to_position:
            return self.bit_to_position[bit_idx]
        
        # Calculate position if not in mapping
        row_idx = bit_idx // self.bits_per_row
        bay = (bit_idx % self.bits_per_row) + 1
        
        if 0 <= row_idx < len(self.row_names) and 1 <= bay <= self.num_bays:
            return f"{self.row_names[row_idx]}{bay}"
        
        raise ValueError(f"Invalid bit index: {bit_idx}")
    
    def is_occupied(self, position: str, tier: int = 1) -> bool:
        """Check if a position at a specific tier is occupied."""
        try:
            bit_idx = self.encode_position(position)
            word_idx = bit_idx // 64
            bit_offset = bit_idx % 64
            
            # Check the specific tier bitmap
            if 1 <= tier <= self.max_tier_height:
                return bool((self.tier_bitmaps[tier-1][word_idx] >> bit_offset) & 1)
            
            return False
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
            return bool((self.special_area_bitmaps[area_type][word_idx] >> bit_offset) & 1)
        except ValueError:
            return False
    
    def get_stack_height(self, position: str) -> int:
        """Get the current stack height at a position."""
        try:
            bit_idx = self.encode_position(position)
            word_idx = bit_idx // 64
            bit_offset = bit_idx % 64
            
            # Check from top to bottom
            for tier in range(self.max_tier_height, 0, -1):
                if (self.tier_bitmaps[tier-1][word_idx] >> bit_offset) & 1:
                    return tier
            
            return 0
        except ValueError:
            return 0
    
    def get_container(self, position: str, tier: int = None) -> Optional[Any]:
        """Get the container at a specific position and tier."""
        try:
            # Default to top container if tier not specified
            if tier is None:
                tier = self.get_stack_height(position)
                if tier == 0:
                    return None
            
            # Check container registry
            if position in self.container_registry and tier in self.container_registry[position]:
                return self.container_registry[position][tier]
            
            return None
        except ValueError:
            return None
    
    def get_top_container(self, position: str) -> Tuple[Optional[Any], Optional[int]]:
        """Get the top container at a position and its tier."""
        tier = self.get_stack_height(position)
        if tier == 0:
            return None, None
        
        container = self.get_container(position, tier)
        return container, tier
    
    def add_container(self, position: str, container: Any, tier: int = None) -> bool:
        """
        Add a container to a specific position and tier.
        
        Args:
            position: Position string (e.g., 'A1')
            container: Container object to add
            tier: Tier level (if None, adds to the top tier + 1)
            
        Returns:
            True if container was added successfully, False otherwise
        """
        try:
            # Validate position
            bit_idx = self.encode_position(position)
            word_idx = bit_idx // 64
            bit_offset = bit_idx % 64
            
            # Check if container can be accepted at this position
            if not self.can_accept_container(position, container):
                return False
            
            # Determine tier if not specified
            if tier is None:
                tier = self.get_stack_height(position) + 1
            
            # Check if tier is already occupied
            if self.is_occupied(position, tier):
                return False
            
            # Check tier height limit
            if tier > self.max_tier_height:
                return False
            
            # Update tier bitmap
            self.tier_bitmaps[tier-1][word_idx] |= (1 << bit_offset)
            
            # Update container registry
            if position not in self.container_registry:
                self.container_registry[position] = {}
            self.container_registry[position][tier] = container
            
            # Update occupied bitmap
            self.update_occupied_bitmap()
            
            # Register container by ID for efficient lookup
            if hasattr(container, 'container_id'):
                self.container_id_to_position = getattr(self, 'container_id_to_position', {})
                self.container_id_to_position[container.container_id] = (position, tier)
            
            return True
        except ValueError:
            return False
    
    def remove_container(self, position: str, tier: int = None) -> Optional[Any]:
        """
        Remove a container from a position and tier.
        
        Args:
            position: Position string (e.g., 'A1')
            tier: Tier level (if None, removes the top container)
            
        Returns:
            The removed container or None if no container was removed
        """
        try:
            # Validate position
            bit_idx = self.encode_position(position)
            word_idx = bit_idx // 64
            bit_offset = bit_idx % 64
            
            # Determine tier if not specified
            if tier is None:
                tier = self.get_stack_height(position)
                if tier == 0:
                    return None
            
            # Check if tier is occupied
            if not self.is_occupied(position, tier):
                return None
            
            # Check if there are containers above this one
            for t in range(tier + 1, self.max_tier_height + 1):
                if self.is_occupied(position, t):
                    return None  # Can't remove with containers on top
            
            # Get container before removing
            container = self.get_container(position, tier)
            
            # Update tier bitmap
            self.tier_bitmaps[tier-1][word_idx] &= ~(1 << bit_offset)
            
            # Update container registry
            if position in self.container_registry and tier in self.container_registry[position]:
                del self.container_registry[position][tier]
                if not self.container_registry[position]:
                    del self.container_registry[position]
            
            # Update occupied bitmap
            self.update_occupied_bitmap()
            
            # Remove container ID reference
            if hasattr(container, 'container_id') and hasattr(self, 'container_id_to_position'):
                if container.container_id in self.container_id_to_position:
                    del self.container_id_to_position[container.container_id]
            
            return container
        except ValueError:
            return None
    
    def can_accept_container(self, position: str, container: Any) -> bool:
        """
        Check if a container can be accepted at a position.
        
        Args:
            position: Position string (e.g., 'A1')
            container: Container to check
            
        Returns:
            True if the container can be accepted, False otherwise
        """
        try:
            # Validate position
            bit_idx = self.encode_position(position)
            
            # Check special area constraints
            if hasattr(container, 'goods_type'):
                if container.goods_type == 'Reefer':
                    # Reefer containers must be in reefer areas
                    if not self.is_position_in_special_area(position, 'reefer'):
                        return False
                elif container.goods_type == 'Dangerous':
                    # Dangerous containers must be in dangerous goods areas
                    if not self.is_position_in_special_area(position, 'dangerous'):
                        return False
                else:
                    # Regular containers cannot be in special areas
                    # We'll allow overlap for simplicity, but this could be modified
                    pass
            
            # Check container type constraints
            if hasattr(container, 'container_type'):
                if container.container_type == 'Trailer':
                    # Trailers must be in trailer areas
                    if not self.is_position_in_special_area(position, 'trailer'):
                        return False
                    # Trailers can't be stacked
                    if self.get_stack_height(position) > 0:
                        return False
                elif container.container_type == 'Swap Body':
                    # Swap bodies must be in swap body areas
                    if not self.is_position_in_special_area(position, 'swap_body'):
                        return False
            
            # Check height constraints
            current_height = self.get_stack_height(position)
            if current_height >= self.max_tier_height:
                return False
            
            # Check stacking compatibility with container below
            if current_height > 0:
                container_below = self.get_container(position, current_height)
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
        Create a proximity mask as a rectangular box that includes all rows
        for bays within range n of the original bay.
        
        Uses tensor operations for efficiency.
        """
        try:
            # Get bit index and position details
            bit_idx = self.encode_position(position)
            orig_row_idx = bit_idx // self.bits_per_row
            orig_bay_idx = bit_idx % self.bits_per_row
            
            # Use cache for repeated calculations
            cache_key = (bit_idx, n)
            if container:
                container_type = getattr(container, 'container_type', 'default')
                goods_type = getattr(container, 'goods_type', 'Regular')
                cache_key = (bit_idx, n, container_type, goods_type)
                
            if cache_key in self.proximity_masks:
                return self.proximity_masks[cache_key]
            
            # Create an empty proximity mask
            proximity_mask = torch.zeros(self.total_bits // 64, dtype=torch.int64, device=self.device)
            
            # TENSOR-BASED APPROACH
            # 1. Create a 2D tensor of the yard (rows x bays)
            yard_tensor = torch.zeros((self.num_rows, self.num_bays), dtype=torch.bool, device=self.device)
            
            # 2. Calculate bay range
            min_bay = max(0, orig_bay_idx - n)
            max_bay = min(self.num_bays - 1, orig_bay_idx + n)
            
            # 3. Set rectangular region to True (all rows, bays within range)
            yard_tensor[:, min_bay:max_bay+1] = True
            
            # 4. Set original position to False (exclude it)
            if 0 <= orig_row_idx < self.num_rows and 0 <= orig_bay_idx < self.num_bays:
                yard_tensor[orig_row_idx, orig_bay_idx] = False
            
            # 5. Convert to bit indices
            rows, bays = torch.where(yard_tensor)
            bit_indices = rows * self.bits_per_row + bays
            
            # 6. Convert to word indices and bit positions
            word_indices = bit_indices // 64
            bit_positions = bit_indices % 64
            
            # 7. Set bits in proximity mask
            # Group by word idx for efficiency
            unique_words = torch.unique(word_indices)
            for word_idx in unique_words:
                word_idx_int = word_idx.item()
                bits_in_word = bit_positions[word_indices == word_idx]
                
                # Set bits for this word
                for bit in bits_in_word:
                    proximity_mask[word_idx_int] |= (1 << bit.item())
            
            # 8. Apply container type filtering
            if container:
                # Apply appropriate filtering based on container type/goods
                if container_type == "Trailer":
                    proximity_mask &= self.special_area_bitmaps['trailer']
                elif container_type == "Swap Body":
                    proximity_mask &= self.special_area_bitmaps['swap_body']
                elif goods_type == "Reefer":
                    proximity_mask &= self.special_area_bitmaps['reefer']
                elif goods_type == "Dangerous":
                    proximity_mask &= self.special_area_bitmaps['dangerous']
                else:
                    # Regular containers - use non-special areas
                    if not hasattr(self, 'non_special_mask'):
                        special_mask = torch.zeros_like(proximity_mask)
                        for area in ['reefer', 'dangerous', 'trailer', 'swap_body']:
                            special_mask |= self.special_area_bitmaps[area]
                        self.non_special_mask = ~special_mask
                    
                    proximity_mask &= self.non_special_mask
            
            # Cache result
            self.proximity_masks[cache_key] = proximity_mask
            return proximity_mask
            
        except ValueError:
            return torch.zeros(self.total_bits // 64, dtype=torch.int64, device=self.device)
    
    def calc_possible_moves(self, position: str, n: int) -> List[str]:
        """
        Calculate all possible positions a container can be moved to within n bays.
        Now properly distinguishes between N=1 (pre-marshalling) and N=5 (transfer).
        
        Args:
            position: Starting position string (e.g., 'A1')
            n: Number of bays to consider (1 for pre-marshalling, 5 for transfers)
            
        Returns:
            List of valid destination positions
        """
        try:
            # Check if position has a container to move
            container, tier = self.get_top_container(position)
            if container is None:
                return []
            
            # Get proximity mask WITH container type filtering and proper N value
            proximity_mask = self.get_proximity_mask(position, n, container)
            
            # For N=1 (pre-marshalling), only return storage positions
            # For N=5 (transfers), return all valid positions
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
                        
                        if dest_bit_idx < self.total_bits:
                            try:
                                dest_position = self.decode_position(dest_bit_idx)
                                
                                # Apply N-specific filtering
                                if n == 1:
                                    # Pre-marshalling: only storage-to-storage moves
                                    if not self._is_storage_position_internal(dest_position):
                                        continue
                                
                                # Check if container can be placed here
                                if self.can_accept_container(dest_position, container):
                                    valid_destinations.append(dest_position)
                            except ValueError:
                                continue
            
            return valid_destinations
        except ValueError:
            return []


    def _is_storage_position_internal(self, position: str) -> bool:
        """Internal helper to check if position is storage (for bitmap yard)."""
        return position and position[0].isalpha() and position[1:].isdigit()

    def find_all_moves_gpu(self):
        """
        GPU-accelerated version of find_all_moves.
        Uses advanced parallelism for maximum throughput.
        """
        start_time = time.time()
        
        # Find all positions with containers on top of stacks
        occupied_positions = []
        
        # Get all occupied positions in a single tensor operation
        # by converting our bitmap to positions
        occupied_bitmap = self.occupied_bitmap
        
        # Extract all set bits efficiently using PyTorch operations
        occupied_indices = []
        for word_idx, word in enumerate(occupied_bitmap):
            if word == 0:
                continue
                
            # Extract set bits in this word
            bits = torch.arange(64, device=self.device)
            mask = (word & (1 << bits)) != 0
            set_bits = bits[mask]
            
            # Convert to global bit indices
            global_indices = word_idx * 64 + set_bits
            occupied_indices.extend(global_indices.tolist())
        
        # Convert bit indices to position strings
        for bit_idx in occupied_indices:
            if bit_idx < self.total_bits:
                try:
                    position = self.decode_position(bit_idx)
                    # Check if it's a top container
                    if self.get_stack_height(position) == self.get_top_container(position)[1]:
                        occupied_positions.append(position)
                except ValueError:
                    continue
        
        # Use the batch processing method to process all positions
        all_moves = self.batch_calc_possible_moves(occupied_positions, n=5)
        
        end_time = time.time()
        # print(f"GPU-accelerated find_all_moves processed {len(occupied_positions)} containers in {end_time - start_time:.4f} seconds")
        
        return all_moves

    def batch_calc_possible_moves(self, positions: List[str], n: int = 5) -> Dict[str, List[str]]:
        """
        Calculate possible moves for multiple containers in parallel.
        
        Args:
            positions: List of position strings to process
            n: Proximity range to use for all calculations
            
        Returns:
            Dictionary mapping each position to a list of valid destinations
        """
        # Start timing
        import time
        start_time = time.time()
        
        # Filter to only positions with containers
        valid_positions = []
        containers = []
        
        for pos in positions:
            container, _ = self.get_top_container(pos)
            if container is not None:
                valid_positions.append(pos)
                containers.append(container)
        
        # Prepare storage for results
        all_moves = {}
        
        # Process in batches of 100 positions or fewer
        batch_size = 100
        for i in range(0, len(valid_positions), batch_size):
            batch_positions = valid_positions[i:i+batch_size]
            batch_containers = containers[i:i+batch_size]
            
            # Process each position in the batch
            batch_results = {}
            
            # Here we use PyTorch's parallelism by operating on tensors
            # Create a tensor to hold all positions' proximity masks
            batch_masks = []
            
            # Gather all proximity masks (this leverages GPU parallelism)
            for j, (pos, container) in enumerate(zip(batch_positions, batch_containers)):
                mask = self.get_proximity_mask(pos, n, container)
                batch_masks.append(mask)
            
            # Now convert masks to valid positions
            for j, (pos, mask) in enumerate(zip(batch_positions, batch_masks)):
                # Get all set bits in the mask
                valid_bits = []
                for word_idx in range(len(mask)):
                    word = mask[word_idx].item()
                    if word == 0:
                        continue
                    
                    # Find set bits using fast bit operations
                    # We can use torch.nonzero for this in a vectorized way
                    for bit_offset in range(64):
                        if (word >> bit_offset) & 1:
                            bit_idx = word_idx * 64 + bit_offset
                            if bit_idx < self.total_bits:
                                valid_bits.append(bit_idx)
                
                # Convert bits to positions
                container = batch_containers[j]
                valid_destinations = []
                for bit_idx in valid_bits:
                    try:
                        dest_pos = self.decode_position(bit_idx)
                        if self.can_accept_container(dest_pos, container):
                            valid_destinations.append(dest_pos)
                    except ValueError:
                        continue
                
                # Store results for this position
                all_moves[pos] = valid_destinations
        
        # Report timing
        end_time = time.time()
        # print(f"Batch processed {len(valid_positions)} positions in {end_time - start_time:.4f} seconds")
        # print(f"Average time per position: {(end_time - start_time) * 1000 / len(valid_positions):.2f} ms")
        
        return all_moves

    def find_container(self, container_id: str) -> Optional[Tuple[str, int]]:
        """
        Find a container by ID.
        
        Args:
            container_id: ID of the container to find
            
        Returns:
            Tuple of (position, tier) or None if not found
        """
        # Check direct lookup first
        if hasattr(self, 'container_id_to_position') and container_id in self.container_id_to_position:
            return self.container_id_to_position[container_id]
        
        # Fallback to searching all positions
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
        if position in self.container_registry:
            return dict(self.container_registry[position])
        return {}
    
    def clear(self):
        """Clear the entire storage yard."""
        # Reset all bitmaps
        for i in range(self.max_tier_height):
            self.tier_bitmaps[i].zero_()
        
        # Clear container registry
        self.container_registry = {}
        
        # Clear container ID mapping
        if hasattr(self, 'container_id_to_position'):
            self.container_id_to_position = {}
        
        # Update occupied bitmap
        self.update_occupied_bitmap()

    def get_containers_by_type(self, container_type: str) -> List[Tuple[str, int, Any]]:
        """
        Get all containers of a specific type using GPU-accelerated search.
        
        Args:
            container_type: Type of container to find
            
        Returns:
            List of tuples (position, tier, container)
        """
        results = []
        
        # Iterate through all positions in the container registry
        for position, tiers in self.container_registry.items():
            for tier, container in tiers.items():
                # Check if container has the desired type
                if hasattr(container, 'container_type') and container.container_type == container_type:
                    results.append((position, tier, container))
        
        return results
    
    def get_container_count(self) -> int:
        """
        Get the total number of containers in the storage yard.
        
        Returns:
            Total container count
        """
        count = 0
        for position, tiers in self.container_registry.items():
            count += len(tiers)
        
        return count

    def get_state_representation(self) -> torch.Tensor:
        """
        Get a tensor representation of the yard state.
        
        Returns:
            A 3D tensor of shape (num_rows, num_bays, max_tier_height)
        """
        # Create tensor for occupied positions
        state = torch.zeros((self.num_rows, self.num_bays, self.max_tier_height), 
                           dtype=torch.int8, device=self.device)
        
        # Fill tensor from bitmaps
        for tier in range(self.max_tier_height):
            tier_bitmap = self.tier_bitmaps[tier]
            
            # Process each word in the bitmap
            for word_idx in range(len(tier_bitmap)):
                word = tier_bitmap[word_idx].item()
                if word == 0:
                    continue
                
                # Process each bit in the word
                for bit_offset in range(64):
                    if (word >> bit_offset) & 1:
                        bit_idx = word_idx * 64 + bit_offset
                        
                        # Calculate row and bay
                        row_idx = bit_idx // self.bits_per_row
                        bay_idx = bit_idx % self.bits_per_row
                        
                        # Skip if outside actual yard dimensions
                        if row_idx >= self.num_rows or bay_idx >= self.num_bays:
                            continue
                        
                        # Set the corresponding tensor element
                        state[row_idx, bay_idx, tier] = 1
        
        return state
    
    def visualize_bitmap(self, bitmap, title="Bitmap Visualization"):
        """
        Visualize a bitmap as a 2D grid using vectorized operations.
        
        Args:
            bitmap: The bitmap to visualize
            title: Title for the visualization
        """
        import matplotlib.pyplot as plt
        
        # Create a 2D tensor representation
        grid = torch.zeros((self.num_rows, self.num_bays), dtype=torch.float32, device=self.device)
        
        # Convert bitmap to 2D grid using vectorized operations
        bitmap_expanded = bitmap.view(-1, 1)  # Shape (n_words, 1)
        bits_expanded = torch.arange(64, device=self.device).repeat(len(bitmap), 1)  # Shape (n_words, 64)
        
        # Generate all bit values
        bit_values = (bitmap_expanded & (1 << bits_expanded)) != 0  # Shape (n_words, 64)
        
        # Flatten to 1D array of bits
        all_bits = bit_values.view(-1)[:self.total_bits]  # Ensure we don't exceed total bits
        
        # Reshape to match our grid layout
        reshaped_bits = all_bits.view(self.num_rows, self.bits_per_row)
        
        # Extract only the valid bays
        grid = reshaped_bits[:, :self.num_bays].float()
        
        # Visualize using matplotlib
        plt.figure(figsize=(12, 6))
        plt.imshow(grid.cpu().numpy(), cmap='Blues', interpolation='none')
        plt.title(title)
        plt.xlabel('Bay')
        plt.ylabel('Row')
        
        # Add row names on y-axis
        plt.yticks(range(self.num_rows), self.row_names)
        
        # Add grid lines
        plt.grid(which='both', color='lightgrey', linestyle='-', linewidth=0.5)
        
        plt.colorbar(label='Occupied')
        plt.tight_layout()
        plt.show()
    
    def visualize_proximity(self, position, n):
        """
        Visualize proximity for a position with container constraints.
        
        Args:
            position: Position to visualize proximity for
            n: Number of bays in each direction
        """
        # Get container at this position
        container, _ = self.get_top_container(position)
        
        # Get proximity mask WITH container type filtering if there's a container
        proximity_mask = self.get_proximity_mask(position, n, container)
        
        # Set title based on container type
        title = f"Proximity Mask for {position} (n={n})"
        if container:
            container_type = getattr(container, 'container_type', 'Unknown')
            goods_type = getattr(container, 'goods_type', 'Regular')
            title += f" - {container_type}, {goods_type}"
        
        # Visualize the mask
        self.visualize_bitmap(proximity_mask, title)
    
    def find_all_moves(self):
        """Find all possible moves for all top containers."""
        start_time = time.time()
        
        # Find all movable containers (ones on top of stacks)
        all_moves = {}
        
        # Use pytorch to parallelize this computation
        # First, we'll create a tensor with all positions that have containers
        occupied_positions = []
        
        for position, tiers in self.container_registry.items():
            if tiers:  # Position has containers
                # Get the highest tier
                max_tier = max(tiers.keys())
                
                # Check if it's a top container
                if max_tier == self.get_stack_height(position):
                    occupied_positions.append(position)
        
        # Process each position
        for position in occupied_positions:
            moves = self.calc_possible_moves(position, 5)  # Standard 5-bay radius
            all_moves[position] = moves
        
        end_time = time.time()
        # print(f"Found all possible moves in {end_time - start_time:.6f} seconds")
        
        return all_moves
    
    def visualize_3d(self, show_container_types=True, figsize=(30, 20)):
        """
        Create a 3D visualization with proper container orientation.
        
        Each slot is 40ft long and containers are oriented with:
        - Their long side (20ft or 40ft) along the bay axis
        - Their short side (8ft/2.4m width) along the row axis
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
        
        # Container dimensions (relative to slot size)
        container_lengths = {
            'TWEU': 0.5,      # 20ft = 1/2 of slot
            'THEU': 0.75,     # 30ft = 3/4 of slot
            'FEU': 1.0,       # 40ft = full slot
            'FFEU': 1.125,    # 45ft = slightly more than slot
            'Trailer': 1.0,   # Trailer = full slot
            'Swap Body': 0.75,# Swap body = 3/4 of slot
            'default': 0.5    # Default to 20ft
        }
        
        # Set up legend elements
        legend_elements = []
        from matplotlib.patches import Patch
        
        # Collect all containers to visualize
        for position, tiers in self.container_registry.items():
            try:
                bit_idx = self.encode_position(position)
                row_idx = bit_idx // self.bits_per_row
                bay_idx = bit_idx % self.bits_per_row
                
                if row_idx < self.num_rows and bay_idx < self.num_bays:
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
                            
                            # CORRECTLY ORIENTED:
                            # dx = length of container along bay axis (20ft or 40ft)
                            # dy = width of container along row axis (all ~8ft)
                            # dz = height of container (all ~8-9.5ft)
                            
                            # Get length for this container type
                            length = container_lengths.get(container_type, 0.5)
                            
                            # For 20ft containers, need to show position within slot
                            # If position has odd index, place in second half of slot
                            if length < 1.0:  # Less than full slot
                                dx = length
                                
                                # Check if we should place it in first or second half
                                # This is a simplification - in reality this would be based on
                                # additional data about exact position within slot
                                if bay_idx % 2 == 0:  # Even position
                                    x_pos = bay_idx
                                else:  # Odd position
                                    x_pos = bay_idx - 0.5  # Place in second half of previous slot
                            else:  # Full slot or larger
                                dx = length
                                x_pos = bay_idx
                            
                            # All containers have same width
                            dy = 0.7  # Slightly less than 1 to see gaps
                            
                            # Height is one tier
                            dz = 0.9  # Slightly less than 1 to see layers
                            
                            # Draw the container with proper dimensions
                            ax.bar3d(
                                x_pos, row_idx + 0.15, tier - 1,  # Position (x, y, z)
                                dx, dy, dz,                       # Dimensions (dx, dy, dz)
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
        ax.set_title('3D Container Yard Visualization')
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
        
        # Set the viewing angle to better see the layout
        ax.view_init(elev=20, azim=230)
        
        plt.tight_layout()
        return fig, ax

    def __str__(self):
        """String representation of the storage yard."""
        container_count = sum(len(tiers) for tiers in self.container_registry.values())
        
        # Get containers per row
        row_counts = {}
        for position in self.container_registry:
            row = position[0]
            if row in row_counts:
                row_counts[row] += len(self.container_registry[position])
            else:
                row_counts[row] = len(self.container_registry[position])
        
        # Format string
        lines = []
        lines.append(f"Bitmap Storage Yard: {self.num_rows} rows, {self.num_bays} bays, {container_count} containers")
        
        for row in self.row_names:
            count = row_counts.get(row, 0)
            lines.append(f"Row {row}: {count} containers")
        
        return '\n'.join(lines)
