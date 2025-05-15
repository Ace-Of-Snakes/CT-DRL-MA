import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Set, Any
import time

class GPUStorageYard:
    """
    GPU-accelerated implementation of the container storage yard in the terminal.
    Uses PyTorch tensors for efficient operations on GPU.
    """
    
    def __init__(self, 
                num_rows: int, 
                num_bays: int, 
                max_tier_height: int = 5,
                row_names: List[str] = None,
                special_areas: Dict[str, List[Tuple[str, int, int]]] = None,
                device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the GPU-accelerated storage yard.
        
        Args:
            num_rows: Number of rows in the yard
            num_bays: Number of bays per row
            max_tier_height: Maximum stacking height
            row_names: Names for each row (defaults to A, B, C...)
            special_areas: Dictionary mapping special types to areas
            device: Device to use for tensors ('cuda' for GPU or 'cpu')
        """
        self.num_rows = num_rows
        self.num_bays = num_bays
        self.max_tier_height = max_tier_height
        self.device = device
        
        print(f"Initializing GPU storage yard on device: {device}")
        
        # Initialize row names if not provided
        if row_names is None:
            self.row_names = [chr(65 + i) for i in range(num_rows)]  # A, B, C, ...
        else:
            self.row_names = row_names[:num_rows]  # Use provided names up to num_rows
        
        # Special areas for different container types
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
        
        # Initialize storage for containers (dictionary for compatibility)
        self.yard = {row: {bay: {} for bay in range(1, num_bays + 1)} for row in self.row_names}
        
        # Create tensor representations
        self._initialize_tensors()
        
        # Mapping from position string to indices
        self.position_to_indices = {}
        self.indices_to_position = {}
        self._build_position_mapping()
        
        # Create static masks for special areas
        self._create_special_area_masks()

        # Performance tracking
        self.container_access_times = []
        self.state_update_times = []
    
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
        """Initialize PyTorch tensors for state representation."""
        # Main occupancy tensor: [row, bay, tier]
        self.occupancy_tensor = torch.zeros((self.num_rows, self.num_bays, self.max_tier_height), 
                                           dtype=torch.bool, device=self.device)
        
        # Container type tensor: [row, bay, tier, type_idx]
        # Type indices: 0=regular, 1=reefer, 2=dangerous, 3=trailer, 4=swap_body
        self.type_tensor = torch.zeros((self.num_rows, self.num_bays, self.max_tier_height, 5), 
                                      dtype=torch.bool, device=self.device)
        
        # Container size tensor: [row, bay, tier, size_idx]
        # Size indices: 0=20ft, 1=30ft, 2=40ft, 3=45ft
        self.size_tensor = torch.zeros((self.num_rows, self.num_bays, self.max_tier_height, 4), 
                                      dtype=torch.bool, device=self.device)
        
        # Priority tensor: [row, bay, tier]
        # Stores normalized priority values (0-100)
        self.priority_tensor = torch.zeros((self.num_rows, self.num_bays, self.max_tier_height), 
                                          dtype=torch.int16, device=self.device)
        
        # Stack height tensor: [row, bay]
        # Stores the current height of each stack (0 = empty)
        self.stack_height_tensor = torch.zeros((self.num_rows, self.num_bays), 
                                              dtype=torch.int8, device=self.device)
    
    def _build_position_mapping(self):
        """Build mappings between position strings and tensor indices."""
        for row_idx, row in enumerate(self.row_names):
            for bay_idx in range(self.num_bays):
                position = f"{row}{bay_idx+1}"
                self.position_to_indices[position] = (row_idx, bay_idx)
                self.indices_to_position[(row_idx, bay_idx)] = position
    
    def _create_special_area_masks(self):
        """Create static boolean masks for special areas."""
        # Initialize masks for each special area type
        self.special_area_masks = {}
        
        # Create mask for each special area type
        for area_type, areas in self.special_areas.items():
            # Initialize mask with False
            mask = torch.zeros((self.num_rows, self.num_bays), dtype=torch.bool, device=self.device)
            
            # Set mask to True for positions in this special area
            for area_row, start_bay, end_bay in areas:
                if area_row in self.row_names:
                    row_idx = self.row_names.index(area_row)
                    # Adjust for 0-based indexing
                    start_bay_idx = max(0, start_bay - 1)
                    end_bay_idx = min(self.num_bays - 1, end_bay - 1)
                    
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
        
        # Zero all tensors
        self.occupancy_tensor.zero_()
        self.type_tensor.zero_()
        self.size_tensor.zero_()
        self.priority_tensor.zero_()
        self.stack_height_tensor.zero_()
    
    def _get_type_code(self, container) -> int:
        """Convert container type and goods type to a numeric code."""
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
        return 0  # Default to regular
    
    def _get_size_code(self, container) -> int:
        """Convert container dimensions to numeric code."""
        if hasattr(container, 'container_type'):
            if container.container_type == "TWEU":  # 20ft
                return 0
            elif container.container_type == "THEU":  # 30ft
                return 1
            elif container.container_type == "FEU":  # 40ft
                return 2
            elif container.container_type == "FFEU":  # 45ft
                return 3
            elif container.container_type in ["Trailer", "Swap Body"]:
                return 2  # Treat as 40ft
        return 2  # Default to 40ft
    
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
        start_time = time.time()
        
        # Check if position is valid
        if position not in self.position_to_indices:
            return False
            
        # Check if container can be accepted at this position
        if not self.can_accept_container(position, container):
            return False
            
        row_idx, bay_idx = self.position_to_indices[position]
        row, bay = self._position_to_row_bay(position)
        
        # If tier not specified, add to the top
        if tier is None:
            current_height = int(self.stack_height_tensor[row_idx, bay_idx].item())
            tier = current_height + 1
            
        # Check tier bounds
        if tier < 1 or tier > self.max_tier_height:
            return False
            
        # Check if the tier is already occupied
        if tier <= int(self.stack_height_tensor[row_idx, bay_idx].item()) and self.occupancy_tensor[row_idx, bay_idx, tier-1]:
            return False
        
        # Add to dictionary
        self.yard[row][bay][tier] = container
        
        # Update tensors
        self.occupancy_tensor[row_idx, bay_idx, tier-1] = True
        
        # Update container type
        type_code = self._get_type_code(container)
        self.type_tensor[row_idx, bay_idx, tier-1, type_code] = True
        
        # Update container size
        size_code = self._get_size_code(container)
        self.size_tensor[row_idx, bay_idx, tier-1, size_code] = True
        
        # Update priority
        priority = getattr(container, 'priority', 50)  # Default to 50
        self.priority_tensor[row_idx, bay_idx, tier-1] = max(0, min(100, priority))
        
        # Update stack height if this is a new top container
        current_height = int(self.stack_height_tensor[row_idx, bay_idx].item())
        if tier > current_height:
            self.stack_height_tensor[row_idx, bay_idx] = tier
        
        # Track performance
        self.container_access_times.append(time.time() - start_time)
        
        return True
    
    def remove_container(self, position: str, tier: int = None) -> Optional[Any]:
        """
        Remove a container from the specified position.
        
        Args:
            position: Position string (e.g., 'A1')
            tier: Tier number (if None, removes the top container)
            
        Returns:
            Removed container or None if no container was removed
        """
        start_time = time.time()
        
        # Check if position is valid
        if position not in self.position_to_indices:
            return None
            
        row_idx, bay_idx = self.position_to_indices[position]
        row, bay = self._position_to_row_bay(position)
        
        # Get current height
        current_height = int(self.stack_height_tensor[row_idx, bay_idx].item())
        
        # If tier not specified, remove from the top
        if tier is None:
            tier = current_height
            if tier == 0:  # No containers in this position
                return None
        
        # Check if there's a container at the specified tier
        if tier not in self.yard[row][bay] or tier > current_height:
            return None
        
        # Check if there are containers above this one
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
        
        # Track performance
        self.container_access_times.append(time.time() - start_time)
        
        return container
    
    def can_accept_container(self, position: str, container: Any) -> bool:
        """
        Check if a container can be added to the specified position.
        
        Args:
            position: Position string (e.g., 'A1')
            container: Container to check
            
        Returns:
            Boolean indicating if the container can be accepted
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
        type_code = self._get_type_code(container)
        
        # Check special area constraints using masks
        if type_code == 1:  # Reefer
            if not self.special_area_masks['reefer'][row_idx, bay_idx]:
                return False
        elif type_code == 2:  # Dangerous
            if not self.special_area_masks['dangerous'][row_idx, bay_idx]:
                return False
        elif type_code == 3:  # Trailer
            if not self.special_area_masks['trailer'][row_idx, bay_idx]:
                return False
        elif type_code == 4:  # Swap Body
            if not self.special_area_masks['swap_body'][row_idx, bay_idx]:
                return False
        elif type_code == 0:  # Regular
            # Regular containers should not go in special areas for better organization
            if not self.special_area_masks['regular'][row_idx, bay_idx]:
                return False
        
        # Check stacking compatibility with container below
        if current_height > 0:
            # Get row and bay
            row, bay = self._position_to_row_bay(position)
            container_below = self.yard[row][bay][current_height]
            if not self._can_stack(container, container_below):
                return False
        
        return True
    
    def _can_stack(self, upper_container, lower_container) -> bool:
        """Check if one container can stack on another."""
        if hasattr(upper_container, 'can_stack_with'):
            return upper_container.can_stack_with(lower_container)
        
        # Basic compatibility checks
        if (hasattr(upper_container, 'is_stackable') and not upper_container.is_stackable or
            hasattr(lower_container, 'is_stackable') and not lower_container.is_stackable):
            return False
        
        if (hasattr(upper_container, 'container_type') and 
            upper_container.container_type in ["Trailer", "Swap Body"]):
            return False
        
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
        row, bay = self._position_to_row_bay(position)
        
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
            
        row, bay = self._position_to_row_bay(position)
        
        # Return a copy of the containers dictionary
        return dict(self.yard[row][bay])
    
    def get_state_representation(self) -> torch.Tensor:
        """
        Get a tensor representation of the storage yard state.
        
        Returns:
            Tensor of shape [rows, bays, features]
        """
        start_time = time.time()
        
        # Create a state representation with multiple features
        # [occupancy, stack_height/max_height, 5 type features, 4 size features, priority]
        state = torch.zeros((self.num_rows, self.num_bays, 11), dtype=torch.float32, device=self.device)
        
        # Feature 0: Occupancy (1 if bay has any containers)
        state[:, :, 0] = (self.stack_height_tensor > 0).float()
        
        # Feature 1: Normalized stack height
        state[:, :, 1] = self.stack_height_tensor.float() / self.max_tier_height
        
        # Features 2-6: Container types at the top
        for type_idx in range(5):
            # For each position with containers, check the type of the top container
            for row_idx in range(self.num_rows):
                for bay_idx in range(self.num_bays):
                    height = int(self.stack_height_tensor[row_idx, bay_idx].item())
                    if height > 0 and self.type_tensor[row_idx, bay_idx, height-1, type_idx]:
                        state[row_idx, bay_idx, 2 + type_idx] = 1.0
        
        # Features 7-10: Container sizes at the top
        for size_idx in range(4):
            # For each position with containers, check the size of the top container
            for row_idx in range(self.num_rows):
                for bay_idx in range(self.num_bays):
                    height = int(self.stack_height_tensor[row_idx, bay_idx].item())
                    if height > 0 and self.size_tensor[row_idx, bay_idx, height-1, size_idx]:
                        state[row_idx, bay_idx, 7 + size_idx] = 1.0
        
        # Feature 11: Priority of top container (normalized to 0-1)
        for row_idx in range(self.num_rows):
            for bay_idx in range(self.num_bays):
                height = int(self.stack_height_tensor[row_idx, bay_idx].item())
                if height > 0:
                    state[row_idx, bay_idx, -1] = self.priority_tensor[row_idx, bay_idx, height-1].float() / 100.0
        
        # Track performance
        self.state_update_times.append(time.time() - start_time)
        
        return state
    
    def _position_to_row_bay(self, position: str) -> Tuple[str, int]:
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
    
    def generate_extraction_mask(self) -> torch.Tensor:
        """
        Generate a mask of positions where containers can be extracted.
        
        Returns:
            Boolean tensor of shape [rows, bays]
        """
        # A container can be extracted if it's at the top of a stack
        return self.stack_height_tensor > 0
    
    def get_valid_placement_mask(self, container: Any) -> torch.Tensor:
        """
        Get a boolean mask of valid placement positions for a container.
        
        Args:
            container: Container to check
            
        Returns:
            Boolean tensor of shape [rows, bays]
        """
        # Initialize mask with all positions
        valid_mask = torch.ones((self.num_rows, self.num_bays), dtype=torch.bool, device=self.device)
        
        # Filter out positions where stack is at max height
        valid_mask = valid_mask & (self.stack_height_tensor < self.max_tier_height)
        
        # Handle special area constraints
        type_code = self._get_type_code(container)
        
        if type_code == 1:  # Reefer
            valid_mask = valid_mask & self.special_area_masks['reefer']
        elif type_code == 2:  # Dangerous
            valid_mask = valid_mask & self.special_area_masks['dangerous']
        elif type_code == 3:  # Trailer
            valid_mask = valid_mask & self.special_area_masks['trailer']
        elif type_code == 4:  # Swap Body
            valid_mask = valid_mask & self.special_area_masks['swap_body']
        elif type_code == 0:  # Regular
            valid_mask = valid_mask & self.special_area_masks['regular']
        
        # For each valid position, check stacking compatibility
        for row_idx in range(self.num_rows):
            for bay_idx in range(self.num_bays):
                if valid_mask[row_idx, bay_idx]:
                    current_height = int(self.stack_height_tensor[row_idx, bay_idx].item())
                    if current_height > 0:
                        position = self.indices_to_position(row_idx, bay_idx)
                        row, bay = self._position_to_row_bay(position)
                        container_below = self.yard[row][bay][current_height]
                        
                        # Check if stacking is allowed
                        if not self._can_stack(container, container_below):
                            valid_mask[row_idx, bay_idx] = False
        
        return valid_mask
    
    def find_container(self, container_id: str) -> List[str]:
        """Find a container by its ID and return its position(s)."""
        positions = []
        
        # Search through all positions
        for row in self.row_names:
            for bay in range(1, self.num_bays + 1):
                for tier, container in self.yard[row][bay].items():
                    if hasattr(container, 'container_id') and container.container_id == container_id:
                        positions.append(f"{row}{bay}")
                        break
        
        return positions
    
    def to_cpu(self):
        """Move all tensors to CPU."""
        self.device = 'cpu'
        self.occupancy_tensor = self.occupancy_tensor.cpu()
        self.type_tensor = self.type_tensor.cpu()
        self.size_tensor = self.size_tensor.cpu()
        self.priority_tensor = self.priority_tensor.cpu()
        self.stack_height_tensor = self.stack_height_tensor.cpu()
        
        for area_type in self.special_area_masks:
            self.special_area_masks[area_type] = self.special_area_masks[area_type].cpu()
    
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
        occupied = torch.sum(self.stack_height_tensor > 0).item()
        total_containers = torch.sum(self.occupancy_tensor).item()
        
        # Count containers by type for more detailed information
        type_counts = []
        type_names = ['regular', 'reefer', 'dangerous', 'trailer', 'swap_body']
        for i, name in enumerate(type_names):
            count = torch.sum(self.type_tensor[:, :, :, i]).item()
            type_counts.append(f"{name}: {int(count)}")
        
        return (f"GPUStorageYard: {self.num_rows} rows × {self.num_bays} bays on {self.device}\n"
                f"Occupied positions: {occupied}/{self.num_rows*self.num_bays}\n"
                f"Total containers: {int(total_containers)}\n"
                f"Container types: {', '.join(type_counts)}")