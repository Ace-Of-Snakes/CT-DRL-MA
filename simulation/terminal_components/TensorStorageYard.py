import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Set, Any
import re


class TensorStorageYard:
    """
    Tensor-based implementation of the container storage yard in the terminal.
    Uses tensor operations for fast computation of valid container placements.
    
    Attributes:
        num_rows: Number of rows in the storage yard
        num_bays: Number of bays in each row
        max_tier_height: Maximum allowed stacking height
        row_names: Names of the rows (e.g., A, B, C...)
        special_areas: Dictionary mapping special area types to areas
    """
    
    def __init__(self, 
                num_rows: int, 
                num_bays: int, 
                max_tier_height: int = 5,
                row_names: List[str] = None,
                special_areas: Dict[str, List[Tuple[str, int, int]]] = None,
                device: str = None):
        """
        Initialize the tensor-based storage yard.
        
        Args:
            num_rows: Number of rows in the yard
            num_bays: Number of bays per row
            max_tier_height: Maximum stacking height
            row_names: Names for each row (defaults to A, B, C...)
            special_areas: Dictionary mapping special types to areas
            device: Device to use for tensors ('cuda' for GPU if available)
        """
        self.num_rows = num_rows
        self.num_bays = num_bays
        self.max_tier_height = max_tier_height
        
        # Set device for tensors
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device} for tensor operations")
        
        # Initialize row names if not provided
        if row_names is None:
            self.row_names = [chr(65 + i) for i in range(num_rows)]  # A, B, C, ...
        else:
            self.row_names = row_names[:num_rows]  # Use provided names up to num_rows
        
        # Special areas for different container types (including trailer/swap body)
        if special_areas is None:
            # Default special areas
            self.special_areas = {
                'reefer': [('A', 1, 5)],  # Row A, bays 1-5
                'dangerous': [('F', 6, 10)],  # Row F, bays 6-10
                'trailer': [('A', 15, 25)],  # Row A, bays 15-25
                'swap_body': [('A', 30, 40)]  # Row A, bays 30-40
            }
        else:
            self.special_areas = special_areas
            
        # Validate special areas
        self._validate_special_areas()
        
        # Initialize storage for containers
        # This remains a dictionary for compatibility with existing code
        self.yard = {row: {bay: {} for bay in range(1, num_bays + 1)} for row in self.row_names}
        
        # Create tensor representations
        self._initialize_tensors()
        
        # Mapping from position string to indices
        self.position_to_indices = {}
        self._build_position_mapping()
        
        # Create static masks for special areas
        self._create_static_masks()

    def _validate_special_areas(self):
        """Validate that special areas are within the yard boundaries."""
        for container_type, areas in self.special_areas.items():
            for area in areas:
                row, bay_start, bay_end = area
                if row not in self.row_names:
                    raise ValueError(f"Invalid row '{row}' in special area for {container_type}")
                if bay_start < 1 or bay_end > self.num_bays or bay_start > bay_end:
                    raise ValueError(f"Invalid bay range {bay_start}-{bay_end} in special area for {container_type}")

    def _initialize_tensors(self):
        """Initialize tensor representations of the yard state."""
        # Main occupancy tensor: [row, bay, tier]
        self.occupancy_tensor = torch.zeros((self.num_rows, self.num_bays, self.max_tier_height), 
                                            dtype=torch.bool, device=self.device)
        
        # Container type tensors: [row, bay, tier, type_idx]
        # Type indices: 0=regular, 1=reefer, 2=dangerous, 3=trailer, 4=swap_body
        self.type_tensor = torch.zeros((self.num_rows, self.num_bays, self.max_tier_height, 5), 
                                      dtype=torch.bool, device=self.device)
        
        # Container size tensors: [row, bay, tier, size_idx]
        # Size indices: 0=20ft, 1=30ft, 2=40ft, 3=45ft
        self.size_tensor = torch.zeros((self.num_rows, self.num_bays, self.max_tier_height, 4), 
                                      dtype=torch.bool, device=self.device)
        
        # Priority tensor: [row, bay, tier]
        # Stores normalized priority values (0-1)
        self.priority_tensor = torch.zeros((self.num_rows, self.num_bays, self.max_tier_height), 
                                          dtype=torch.float32, device=self.device)
        
        # Height/tier occupancy: [row, bay]
        # Stores the current height of each stack
        self.stack_height_tensor = torch.zeros((self.num_rows, self.num_bays), 
                                               dtype=torch.int32, device=self.device)

    def _build_position_mapping(self):
        """Build mapping from position strings (e.g., 'A1') to tensor indices."""
        for row_idx, row in enumerate(self.row_names):
            for bay_idx in range(self.num_bays):
                position = f"{row}{bay_idx+1}"
                self.position_to_indices[position] = (row_idx, bay_idx)
    
    def _create_static_masks(self):
        """Create static boolean masks for special areas."""
        # Initialize masks for each special area type
        self.special_area_masks = {}
        
        # Create mask for each special area type
        for area_type, areas in self.special_areas.items():
            # Initialize mask with zeros
            mask = torch.zeros((self.num_rows, self.num_bays), dtype=torch.bool, device=self.device)
            
            # Set mask to 1 for positions in this special area
            for area_row, start_bay, end_bay in areas:
                row_idx = self.row_names.index(area_row)
                # Adjust for 0-based indexing
                start_bay_idx = start_bay - 1
                end_bay_idx = end_bay - 1
                
                # Set the mask for this area
                mask[row_idx, start_bay_idx:end_bay_idx+1] = True
            
            # Store the mask
            self.special_area_masks[area_type] = mask
            
        # For regular positions, create a mask where no special areas exist
        regular_mask = torch.ones((self.num_rows, self.num_bays), dtype=torch.bool, device=self.device)
        for mask in self.special_area_masks.values():
            regular_mask = regular_mask & ~mask
        
        self.special_area_masks['regular'] = regular_mask

    def clear(self):
        """Clear the entire storage yard."""
        # Clear dictionary representation
        for row in self.row_names:
            for bay in range(1, self.num_bays + 1):
                self.yard[row][bay] = {}
        
        # Reset all tensors
        self.occupancy_tensor.zero_()
        self.type_tensor.zero_()
        self.size_tensor.zero_()
        self.priority_tensor.zero_()
        self.stack_height_tensor.zero_()

    def _get_type_index(self, container) -> int:
        """Convert container type and goods type to a type index."""
        if hasattr(container, 'container_type') and hasattr(container, 'goods_type'):
            if container.container_type == "Trailer":
                return 3
            elif container.container_type == "Swap Body":
                return 4
            elif container.goods_type == "Reefer":
                return 1
            elif container.goods_type == "Dangerous":
                return 2
            else:
                return 0  # Regular
        return 0  # Default to regular if unknown

    def _get_size_index(self, container) -> int:
        """Convert container physical type to a size index."""
        if hasattr(container, 'container_type'):
            if container.container_type == "TWEU":  # 20ft
                return 0
            elif container.container_type == "THEU":  # 30ft
                return 1
            elif container.container_type == "FEU":  # 40ft
                return 2
            elif container.container_type == "FFEU":  # 45ft
                return 3
            elif container.container_type == "Trailer" or container.container_type == "Swap Body":
                return 2  # Treat as 40ft for size calculations
        return 2  # Default to 40ft if unknown

    def add_container(self, position: str, container: Any, tier: int = None) -> bool:
        """
        Add a container to the specified position.
        Updates both dictionary and tensor representations.
        
        Args:
            position: Position string (e.g., 'A1')
            container: Container object to add
            tier: Specific tier to add the container to (if None, adds to the top)
            
        Returns:
            Boolean indicating success
        """
        # Check if position is valid
        if position not in self.position_to_indices:
            return False
            
        # Check if container can be accepted at this position
        if not self.can_accept_container(position, container):
            return False
            
        row_idx, bay_idx = self.position_to_indices[position]
        row, bay = self.position_to_row_bay(position)
        
        # If tier not specified, add to the top
        if tier is None:
            tier = int(self.stack_height_tensor[row_idx, bay_idx].item()) + 1
            
        # Check tier bounds
        if tier < 1 or tier > self.max_tier_height:
            return False
            
        # Check if the tier is already occupied
        if self.occupancy_tensor[row_idx, bay_idx, tier-1]:
            return False
        
        # Add to dictionary
        self.yard[row][bay][tier] = container
        
        # Update tensors
        self.occupancy_tensor[row_idx, bay_idx, tier-1] = True
        
        # Update container type
        type_idx = self._get_type_index(container)
        self.type_tensor[row_idx, bay_idx, tier-1, type_idx] = True
        
        # Update container size
        size_idx = self._get_size_index(container)
        self.size_tensor[row_idx, bay_idx, tier-1, size_idx] = True
        
        # Update priority (normalized to 0-1)
        priority = getattr(container, 'priority', 50)  # Default to 50 if not set
        normalized_priority = 1.0 - (min(max(priority, 1), 100) / 100.0)  # Invert so higher priority = higher value
        self.priority_tensor[row_idx, bay_idx, tier-1] = normalized_priority
        
        # Update stack height if this is a new top container
        current_height = int(self.stack_height_tensor[row_idx, bay_idx].item())
        if tier > current_height:
            self.stack_height_tensor[row_idx, bay_idx] = tier
        
        return True

    def remove_container(self, position: str, tier: int = None) -> Optional[Any]:
        """
        Remove a container from the specified position.
        Updates both dictionary and tensor representations.
        
        Args:
            position: Position string (e.g., 'A1')
            tier: Tier number (if None, removes the top container)
            
        Returns:
            Removed container or None if no container was removed
        """
        # Check if position is valid
        if position not in self.position_to_indices:
            return None
            
        row_idx, bay_idx = self.position_to_indices[position]
        row, bay = self.position_to_row_bay(position)
        
        # If tier not specified, find the top container
        if tier is None:
            tier = int(self.stack_height_tensor[row_idx, bay_idx].item())
            if tier == 0:  # No containers in this position
                return None
        
        # Check if there's a container at the specified tier
        if tier not in self.yard[row][bay]:
            return None
            
        # Check if there are containers above this one
        current_height = int(self.stack_height_tensor[row_idx, bay_idx].item())
        if tier < current_height:
            # Can't remove container with containers on top
            return None
        
        # Remove from dictionary
        container = self.yard[row][bay].pop(tier)
        
        # Update tensors
        self.occupancy_tensor[row_idx, bay_idx, tier-1] = False
        self.type_tensor[row_idx, bay_idx, tier-1] = 0
        self.size_tensor[row_idx, bay_idx, tier-1] = 0
        self.priority_tensor[row_idx, bay_idx, tier-1] = 0
        
        # Update stack height
        if tier == current_height:
            # Find the new top container
            new_height = 0
            for t in range(tier-1, 0, -1):
                if self.occupancy_tensor[row_idx, bay_idx, t-1]:
                    new_height = t
                    break
            self.stack_height_tensor[row_idx, bay_idx] = new_height
        
        return container

    def can_accept_container(self, position: str, container: Any) -> bool:
        """
        Check if a container can be added to the specified position using tensors.
        
        Args:
            position: Position string (e.g., 'A1')
            container: Container to check
            
        Returns:
            Boolean indicating if the container can be added
        """
        # Check if position is valid
        if position not in self.position_to_indices:
            return False
            
        row_idx, bay_idx = self.position_to_indices[position]
        
        # Check height constraints
        current_height = int(self.stack_height_tensor[row_idx, bay_idx].item())
        if current_height >= self.max_tier_height:
            return False
        
        # Get container type and special area requirements
        type_idx = self._get_type_index(container)
        
        # Check special area constraints using masks
        if type_idx == 1:  # Reefer
            if not self.special_area_masks['reefer'][row_idx, bay_idx]:
                return False
        elif type_idx == 2:  # Dangerous
            if not self.special_area_masks['dangerous'][row_idx, bay_idx]:
                return False
        elif type_idx == 3:  # Trailer
            if not self.special_area_masks['trailer'][row_idx, bay_idx]:
                return False
        elif type_idx == 4:  # Swap Body
            if not self.special_area_masks['swap_body'][row_idx, bay_idx]:
                return False
        elif type_idx == 0:  # Regular
            # Regular containers can't go in special areas (strict rule for better organization)
            special_areas = (
                self.special_area_masks['reefer'][row_idx, bay_idx] |
                self.special_area_masks['dangerous'][row_idx, bay_idx] |
                self.special_area_masks['trailer'][row_idx, bay_idx] |
                self.special_area_masks['swap_body'][row_idx, bay_idx]
            )
            if special_areas:
                return False
        
        # Check stacking compatibility with container below
        if current_height > 0:
            # Traditional check for stacking compatibility using the dictionary
            row, bay = self.position_to_row_bay(position)
            container_below = self.yard[row][bay][current_height]
            if not self._can_stack(container, container_below):
                return False
        
        return True

    def _can_stack(self, upper_container, lower_container) -> bool:
        """Check if one container can stack on another using container attributes."""
        if hasattr(upper_container, 'can_stack_with'):
            return upper_container.can_stack_with(lower_container)
        
        # Basic compatibility checks if can_stack_with not available
        # 1. Non-stackable containers can't be stacked
        if (hasattr(upper_container, 'is_stackable') and not upper_container.is_stackable or
            hasattr(lower_container, 'is_stackable') and not lower_container.is_stackable):
            return False
            
        # 2. Trailers and swap bodies can't be stacked
        if (hasattr(upper_container, 'container_type') and 
            upper_container.container_type in ["Trailer", "Swap Body"]):
            return False
            
        # 3. Different types usually can't stack (simplified rule)
        if (hasattr(upper_container, 'container_type') and 
            hasattr(lower_container, 'container_type') and
            upper_container.container_type != lower_container.container_type):
            return False
            
        return True

    def get_top_container(self, position: str) -> Tuple[Optional[Any], Optional[int]]:
        """
        Get the top container at the specified position.
        
        Args:
            position: Position string (e.g., 'A1')
            
        Returns:
            Tuple of (container, tier) or (None, None) if no container is found
        """
        # Check if position is valid
        if position not in self.position_to_indices:
            return None, None
            
        row_idx, bay_idx = self.position_to_indices[position]
        row, bay = self.position_to_row_bay(position)
        
        # Get current height
        current_height = int(self.stack_height_tensor[row_idx, bay_idx].item())
        if current_height == 0:  # No containers in this position
            return None, None
        
        # Return the container at the highest tier
        return self.yard[row][bay][current_height], current_height

    def get_containers_at_position(self, position: str) -> Dict[int, Any]:
        """
        Get all containers at a specified position.
        
        Args:
            position: Position string (e.g., 'A1')
            
        Returns:
            Dictionary mapping tier to container
        """
        # Check if position is valid
        if position not in self.position_to_indices:
            return {}
            
        row, bay = self.position_to_row_bay(position)
        
        # Return a copy of the containers dictionary
        return dict(self.yard[row][bay])

    def get_state_representation(self) -> torch.Tensor:
        """
        Get a tensor representation of the storage yard state.
        
        Returns:
            Tensor representation of the yard [rows, bays, features]
            where features include occupancy, container type, and more
        """
        # Create a more detailed tensor representation with multiple features
        # Features: [occupancy, stack_height/max_height, 5_container_types, 4_container_sizes, priority]
        state = torch.zeros((self.num_rows, self.num_bays, 11), 
                            dtype=torch.float32, device=self.device)
        
        # Feature 0: Occupancy (1 if bay has any containers)
        state[:, :, 0] = (self.stack_height_tensor > 0).float()
        
        # Feature 1: Normalized stack height
        state[:, :, 1] = self.stack_height_tensor.float() / self.max_tier_height
        
        # Features 2-6: Container types at the top of each stack
        for type_idx in range(5):
            # For each position, check if the top container is of this type
            for row_idx in range(self.num_rows):
                for bay_idx in range(self.num_bays):
                    height = int(self.stack_height_tensor[row_idx, bay_idx].item())
                    if height > 0:
                        if self.type_tensor[row_idx, bay_idx, height-1, type_idx]:
                            state[row_idx, bay_idx, 2 + type_idx] = 1.0
        
        # Features 7-10: Container sizes at the top of each stack
        for size_idx in range(4):
            # For each position, check if the top container is of this size
            for row_idx in range(self.num_rows):
                for bay_idx in range(self.num_bays):
                    height = int(self.stack_height_tensor[row_idx, bay_idx].item())
                    if height > 0:
                        if self.size_tensor[row_idx, bay_idx, height-1, size_idx]:
                            state[row_idx, bay_idx, 7 + size_idx] = 1.0
        
        # Feature 11: Priority of top container
        for row_idx in range(self.num_rows):
            for bay_idx in range(self.num_bays):
                height = int(self.stack_height_tensor[row_idx, bay_idx].item())
                if height > 0:
                    state[row_idx, bay_idx, -1] = self.priority_tensor[row_idx, bay_idx, height-1]
        
        return state
    
    def get_state_representation_3d(self) -> torch.Tensor:
        """
        Get a 3D tensor representation of the storage yard state.
        
        Returns:
            3D Tensor representation of the yard [rows, bays, tiers]
        """
        # Create a simplified 3D tensor just showing occupancy
        return self.occupancy_tensor.clone()

    def get_valid_placement_mask(self, container: Any) -> torch.Tensor:
        """
        Get a boolean mask of valid placement positions for a container.
        
        Args:
            container: Container to place
            
        Returns:
            Boolean tensor of shape [rows, bays] where True indicates valid placement
        """
        # Initialize mask with all positions
        valid_mask = torch.ones((self.num_rows, self.num_bays), dtype=torch.bool, device=self.device)
        
        # Filter out positions where stack is at max height
        valid_mask = valid_mask & (self.stack_height_tensor < self.max_tier_height)
        
        # Handle special area constraints
        type_idx = self._get_type_index(container)
        
        if type_idx == 1:  # Reefer
            valid_mask = valid_mask & self.special_area_masks['reefer']
        elif type_idx == 2:  # Dangerous
            valid_mask = valid_mask & self.special_area_masks['dangerous']
        elif type_idx == 3:  # Trailer
            valid_mask = valid_mask & self.special_area_masks['trailer']
        elif type_idx == 4:  # Swap Body
            valid_mask = valid_mask & self.special_area_masks['swap_body']
        elif type_idx == 0:  # Regular
            # Regular containers can't go in special areas
            for area_type in ['reefer', 'dangerous', 'trailer', 'swap_body']:
                valid_mask = valid_mask & ~self.special_area_masks[area_type]
        
        # For each valid position, check stacking compatibility
        for row_idx in range(self.num_rows):
            for bay_idx in range(self.num_bays):
                if valid_mask[row_idx, bay_idx]:
                    current_height = int(self.stack_height_tensor[row_idx, bay_idx].item())
                    if current_height > 0:
                        position = f"{self.row_names[row_idx]}{bay_idx+1}"
                        row, bay = self.position_to_row_bay(position)
                        container_below = self.yard[row][bay][current_height]
                        
                        # Check if stacking is allowed
                        if not self._can_stack(container, container_below):
                            valid_mask[row_idx, bay_idx] = False
        
        return valid_mask

    def generate_extraction_mask(self) -> torch.Tensor:
        """
        Generate a mask of positions where containers can be extracted.
        Only top containers can be extracted.
        
        Returns:
            Boolean tensor of shape [rows, bays] where True indicates extractable
        """
        # A container can be extracted if it's at the top of a stack
        # This is simply where stack_height > 0
        return self.stack_height_tensor > 0

    def position_to_row_bay(self, position: str) -> Tuple[str, int]:
        """Convert a position string to (row, bay)."""
        if position not in self.position_to_indices:
            raise ValueError(f"Invalid position: {position}")
            
        row_idx, bay_idx = self.position_to_indices[position]
        row = self.row_names[row_idx]
        bay = bay_idx + 1  # Convert to 1-based indexing
        
        return row, bay

    def indices_to_position(self, row_idx: int, bay_idx: int) -> str:
        """Convert tensor indices to position string."""
        if 0 <= row_idx < self.num_rows and 0 <= bay_idx < self.num_bays:
            row = self.row_names[row_idx]
            bay = bay_idx + 1  # Convert to 1-based indexing
            return f"{row}{bay}"
        return None

    def to_cpu(self):
        """Move all tensors to CPU for saving."""
        self.occupancy_tensor = self.occupancy_tensor.cpu()
        self.type_tensor = self.type_tensor.cpu()
        self.size_tensor = self.size_tensor.cpu()
        self.priority_tensor = self.priority_tensor.cpu()
        self.stack_height_tensor = self.stack_height_tensor.cpu()
        
        for area_type in self.special_area_masks:
            self.special_area_masks[area_type] = self.special_area_masks[area_type].cpu()
            
        self.device = 'cpu'

    def to_device(self, device: str):
        """Move all tensors to specified device."""
        self.device = device
        self.occupancy_tensor = self.occupancy_tensor.to(device)
        self.type_tensor = self.type_tensor.to(device)
        self.size_tensor = self.size_tensor.to(device)
        self.priority_tensor = self.priority_tensor.to(device)
        self.stack_height_tensor = self.stack_height_tensor.to(device)
        
        for area_type in self.special_area_masks:
            self.special_area_masks[area_type] = self.special_area_masks[area_type].to(device)

    def __str__(self):
        """String representation of the storage yard."""
        rows = []
        rows.append(f"TensorStorageYard: {self.num_rows} rows, {self.num_bays} bays")
        rows.append(f"Total containers: {torch.sum(self.occupancy_tensor).item()}")
        rows.append(f"Devices: {self.device}")
        
        # Count by container type
        type_counts = []
        for i, name in enumerate(['regular', 'reefer', 'dangerous', 'trailer', 'swap_body']):
            count = torch.sum(self.type_tensor[:, :, :, i]).item()
            type_counts.append(f"{name}={int(count)}")
        
        rows.append(f"Container types: {', '.join(type_counts)}")
        
        return "\n".join(rows)