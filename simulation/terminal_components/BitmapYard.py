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
        Get a bit mask for positions within n bays using efficient bit operations.
        
        Args:
            position: Position string (e.g., 'A1')
            n: Number of bays to search in each direction
            container: Optional - container to filter valid positions by type
            
        Returns:
            Bit mask as a PyTorch tensor
        """
        try:
            # Get bit index and position details
            bit_idx = self.encode_position(position)
            orig_row_idx = bit_idx // self.bits_per_row
            orig_bay_idx = bit_idx % self.bits_per_row
            
            # Create cache key including container type info
            cache_key = (bit_idx, n)
            if container:
                cache_key = (bit_idx, n, getattr(container, 'container_type', ''), 
                            getattr(container, 'goods_type', ''))
                
            if cache_key in self.proximity_masks:
                return self.proximity_masks[cache_key]
            
            # Create proximity mask using bit operations
            proximity_mask = torch.zeros(self.total_bits // 64, dtype=torch.int64, device=self.device)
            
            # Calculate bay range
            min_bay = max(0, orig_bay_idx - n)
            max_bay = min(self.num_bays - 1, orig_bay_idx + n)
            
            # For each row, create a bay mask (bits set for bays min_bay to max_bay)
            for row_idx in range(self.num_rows):
                # Calculate bit range for this row
                start_bit = row_idx * self.bits_per_row + min_bay
                end_bit = row_idx * self.bits_per_row + max_bay
                
                # Set bits in range (optimized bit mask creation)
                for bit in range(start_bit, end_bit + 1):
                    word_idx = bit // 64
                    bit_offset = bit % 64
                    proximity_mask[word_idx] |= (1 << bit_offset)
            
            # Clear the original position's bit
            orig_word_idx = bit_idx // 64
            orig_bit_offset = bit_idx % 64
            proximity_mask[orig_word_idx] &= ~(1 << orig_bit_offset)
            
            # APPLY CONTAINER TYPE CONSTRAINTS USING BIT OPERATIONS
            if container is not None:
                container_type = getattr(container, 'container_type', None)
                goods_type = getattr(container, 'goods_type', 'Regular')
                
                # Create appropriate constraint mask
                if container_type == "Trailer":
                    # For trailers, only trailer areas
                    constraint_mask = self.special_area_bitmaps['trailer']
                elif container_type == "Swap Body":
                    # For swap bodies, only swap body areas
                    constraint_mask = self.special_area_bitmaps['swap_body']
                elif goods_type == "Reefer":
                    # For reefer, only reefer areas
                    constraint_mask = self.special_area_bitmaps['reefer']
                elif goods_type == "Dangerous":
                    # For dangerous goods, only dangerous areas
                    constraint_mask = self.special_area_bitmaps['dangerous']
                else:
                    # For regular containers, exclude all special areas
                    # Start with all bits set
                    constraint_mask = torch.ones_like(proximity_mask)
                    
                    # Clear bits for special areas using bitwise operations
                    special_areas_mask = torch.zeros_like(proximity_mask)
                    for area_type in ['reefer', 'dangerous', 'trailer', 'swap_body']:
                        special_areas_mask |= self.special_area_bitmaps[area_type]
                    
                    # Get regular areas by inverting special areas
                    constraint_mask &= ~special_areas_mask
                
                # Apply constraint using bitwise AND
                proximity_mask &= constraint_mask
            
            # Cache the result
            self.proximity_masks[cache_key] = proximity_mask
            
            return proximity_mask
            
        except ValueError:
            return torch.zeros(self.total_bits // 64, dtype=torch.int64, device=self.device)
    
    def calc_possible_moves(self, position: str, n: int) -> List[str]:
        """
        Calculate all possible positions a container can be moved to within n bays.
        
        Args:
            position: Starting position string (e.g., 'A1')
            n: Number of bays to consider in each direction
            
        Returns:
            List of valid destination positions
        """
        try:
            # Check if position has a container to move
            container, tier = self.get_top_container(position)
            if container is None:
                return []
            
            # Get proximity mask WITH container type filtering
            proximity_mask = self.get_proximity_mask(position, n, container)
            
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
                                dest_position = self.decode_position(dest_bit_idx)
                                
                                # Double-check if container can be placed here
                                # (this includes additional checks beyond just area type)
                                if self.can_accept_container(dest_position, container):
                                    valid_destinations.append(dest_position)
                            except ValueError:
                                # Skip invalid positions
                                continue
            
            return valid_destinations
        except ValueError:
            return []
    
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
        Visualize a bitmap as a 2D grid.
        
        Args:
            bitmap: The bitmap to visualize
            title: Title for the visualization
        """
        import matplotlib.pyplot as plt
        
        # Create a 2D array
        grid = torch.zeros((self.num_rows, self.num_bays), dtype=torch.int8)
        
        # Fill grid from bitmap
        for word_idx in range(len(bitmap)):
            word = bitmap[word_idx].item()
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
                    
                    # Set the corresponding grid element
                    grid[row_idx, bay_idx] = 1
        
        # Visualize the grid
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
        print(f"Found all possible moves in {end_time - start_time:.6f} seconds")
        
        return all_moves

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


def main():
    """Test the bitmap-based storage yard implementation with Container class from repository."""
    import sys
    import os
    import time
    import matplotlib.pyplot as plt
    from datetime import datetime
    
    # Ensure the repository root is in the path for imports
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if repo_root not in sys.path:
        sys.path.append(repo_root)
    
    # Import the Container class and ContainerFactory from the repository
    from Container import Container, ContainerFactory
    
    # Determine compute device - use GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    print("\n===== Testing BitmapStorageYard Implementation with Repository Containers =====")
    
    # 1. Create a bitmap storage yard
    yard = BitmapStorageYard(
        num_rows=6,            # 6 rows (A-F)
        num_bays=40,           # 40 bays per row
        max_tier_height=5,     # Maximum 5 containers high
        special_areas={
            'reefer': [('A', 1, 5), ('F', 35, 40)],
            'dangerous': [('C', 25, 30)],
            'trailer': [('A', 15, 25)],
            'swap_body': [('B', 30, 40)]
        },
        device=device
    )
    
    print(f"Created bitmap storage yard with {yard.num_rows} rows and {yard.num_bays} bays")
    
    # 2. Create test containers using ContainerFactory
    print("\n----- Creating test containers with ContainerFactory -----")
    
    # Current date for container attributes
    current_date = datetime.now()
    
    containers = [
        # Regular containers
        ContainerFactory.create_container("REG001", "TWEU", "Import", "Regular", weight=20000),
        ContainerFactory.create_container("REG002", "FEU", "Export", "Regular", weight=25000),
        ContainerFactory.create_container("REG003", "THEU", "Import", "Regular", weight=18000),
        
        # Reefer containers
        ContainerFactory.create_container("REEF001", "TWEU", "Import", "Reefer", weight=22000),
        ContainerFactory.create_container("REEF002", "FEU", "Export", "Reefer", weight=24000),
        
        # Dangerous goods
        ContainerFactory.create_container("DG001", "TWEU", "Import", "Dangerous", weight=19000),
        ContainerFactory.create_container("DG002", "FEU", "Export", "Dangerous", weight=27000),
        
        # Special types
        ContainerFactory.create_container("TRL001", "Trailer", "Export", "Regular", weight=15000),
        ContainerFactory.create_container("SB001", "Swap Body", "Export", "Regular", weight=12000),
    ]
    
    # Print container details
    for i, container in enumerate(containers):
        print(f"{i+1}. Created {container.container_id}: {container.container_type}, {container.goods_type}, " +
              f"Stackable: {container.is_stackable}, Compatibility: {container.stack_compatibility}")
    
    # 3. Test adding containers to the yard
    print("\n----- Testing container placement -----")
    
    # Add containers to appropriate areas
    placements = [
        # Regular containers in regular areas
        ('D10', containers[0]),
        ('D11', containers[1]),
        ('D12', containers[2]),
        # Reefer containers in reefer areas
        ('A3', containers[3]),
        ('F38', containers[4]),
        # Dangerous goods in dangerous area
        ('C27', containers[5]),
        ('C28', containers[6]),
        # Special containers in special areas
        ('A18', containers[7]),  # Trailer
        ('B35', containers[8])   # Swap body
    ]
    
    for position, container in placements:
        success = yard.add_container(position, container)
        print(f"Adding {container.container_id} to {position}: {'Success' if success else 'Failed'}")
    
    # 4. Test stacking
    print("\n----- Testing stacking -----")
    stack_position = 'D15'
    stack_containers = [
        ContainerFactory.create_container("STACK001", "TWEU", "Import", "Regular", weight=24000),
        ContainerFactory.create_container("STACK002", "TWEU", "Import", "Regular", weight=20000),
        ContainerFactory.create_container("STACK003", "TWEU", "Import", "Regular", weight=18000),
    ]
    
    for i, container in enumerate(stack_containers):
        tier = i + 1
        success = yard.add_container(stack_position, container, tier)
        print(f"Adding {container.container_id} to {stack_position} tier {tier}: {'Success' if success else 'Failed'}")
    
    # 5. Test invalid placements
    print("\n----- Testing invalid placements -----")
    
    # Try to add a reefer container to a non-reefer area
    reefer_container = ContainerFactory.create_container("INVALID01", "TWEU", "Import", "Reefer")
    success = yard.add_container('D20', reefer_container)
    print(f"Adding reefer container to non-reefer area: {'Success' if success else 'Failed (expected)'}")
    
    # Try to add a trailer outside trailer area
    trailer_container = ContainerFactory.create_container("INVALID02", "Trailer", "Export", "Regular")
    success = yard.add_container('E30', trailer_container)
    print(f"Adding trailer outside trailer area: {'Success' if success else 'Failed (expected)'}")
    
    # Try to stack on a trailer
    regular_container = ContainerFactory.create_container("INVALID03", "TWEU", "Import", "Regular")
    success = yard.add_container('A18', regular_container, tier=2)
    print(f"Adding container on top of trailer: {'Success' if success else 'Failed (expected)'}")
    
    # 6. Visualize yard state
    print("\n----- Visualizing yard state -----")
    
    # Get the combined occupied bitmap (combined from all tiers)
    print("Visualizing all occupied positions...")
    yard.visualize_bitmap(yard.occupied_bitmap, "All Occupied Positions")
    
    # 7. Test proximity calculation
    print("\n----- Testing proximity calculation -----")
    
    test_positions = ['D10', 'A3', 'C27', 'A18', stack_position]
    for position in test_positions:
        container, tier = yard.get_top_container(position)
        if container:
            print(f"\nCalculating proximity for {container.container_id} at {position}:")
            
            # Calculate proximity mask for different distances
            for n in [3, 5, 10]:
                # Visualize the proximity mask
                yard.visualize_proximity(position, n)
                
                # Calculate valid moves
                start_time = time.time()
                valid_moves = yard.calc_possible_moves(position, n)
                calc_time = (time.time() - start_time) * 1000  # ms
                
                print(f"Valid moves within {n} bays ({len(valid_moves)}): {valid_moves}")
                print(f"Calculated in {calc_time:.3f} ms")
    
    # 8. Test batch computation
    print("\n----- Batch computation with all containers -----")
    
    # Create a large batch of random containers for testing
    batch_size = 100
    batch_containers = []
    
    for i in range(batch_size):
        container_id = f"BATCH{i:03d}"
        # Create a mix of container types
        container_type = ["TWEU", "FEU", "THEU", "Trailer", "Swap Body"][i % 5]
        goods_type = ["Regular", "Reefer", "Dangerous"][i % 3]
        
        # Create container with ContainerFactory
        container = ContainerFactory.create_container(
            container_id=container_id,
            container_type=container_type,
            goods_type=goods_type
        )
        
        batch_containers.append(container)
    
    # Test find_all_moves performance
    start_time = time.time()
    all_moves = yard.find_all_moves()
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"Find all moves for {len(all_moves)} containers: {total_time:.6f} seconds")
    
    # Print summary
    move_count = sum(len(moves) for moves in all_moves.values())
    print(f"Found {move_count} possible moves for {len(all_moves)} containers")
    print(f"Average of {move_count / len(all_moves):.2f} moves per container")
    
    # 9. Profile performance with different proximity ranges
    print("\n----- Performance testing with different proximity ranges -----")
    
    n_values = [3, 5, 10, 20]
    times = []
    
    for n in n_values:
        start_time = time.time()
        
        # Calculate valid moves for all test positions
        for position in test_positions:
            yard.calc_possible_moves(position, n)
        
        end_time = time.time()
        avg_time = (end_time - start_time) / len(test_positions) * 1000  # ms
        times.append(avg_time)
        
        print(f"N={n}: Average {avg_time:.3f} ms per position")
    
    # Visualize performance
    plt.figure(figsize=(10, 5))
    plt.plot(n_values, times, marker='o')
    plt.title('Proximity Calculation Performance')
    plt.xlabel('Proximity Range (n)')
    plt.ylabel('Average Time (ms)')
    plt.grid(True)
    plt.show()
    
    # 10. Print final stats
    print("\n----- Final Yard State -----")
    print(yard)
    
    container_count = sum(len(tiers) for position, tiers in yard.container_registry.items())
    print(f"Total containers: {container_count}")
    print(f"Containers per row:")
    
    for row in yard.row_names:
        count = 0
        for position, tiers in yard.container_registry.items():
            if position[0] == row:
                count += len(tiers)
        print(f"  Row {row}: {count} containers")

if __name__ == "__main__":
    main()