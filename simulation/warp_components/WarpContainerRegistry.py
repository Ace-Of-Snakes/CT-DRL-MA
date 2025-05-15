import warp as wp
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import uuid

class WarpContainerRegistry:
    """
    GPU-accelerated container registry for terminal simulation using NVIDIA Warp.
    Efficiently manages container data, stacking rules, and container operations.
    """
    
    def __init__(self, 
                 terminal_state,
                 max_containers: int = 10000,
                 device: str = None):
        """
        Initialize the container registry.
        
        Args:
            terminal_state: Reference to the WarpTerminalState object
            max_containers: Maximum number of containers that can be in the system
            device: Computation device (if None, will use the terminal_state's device)
        """
        self.terminal_state = terminal_state
        self.max_containers = max_containers
        self.device = device if device else terminal_state.device
        
        # Container ID to index mapping (for CPU-side lookups)
        self.container_id_to_idx = {}
        self.container_idx_to_id = {}
        
        # Container type codes
        self.type_codes = {
            "TWEU": 0,      # 20-foot
            "THEU": 1,      # 30-foot
            "FEU": 2,       # 40-foot
            "FFEU": 3,      # 45-foot
            "Trailer": 4,
            "Swap Body": 5
        }
        
        # Goods type codes
        self.goods_codes = {
            "Regular": 0,
            "Reefer": 1,
            "Dangerous": 2
        }
        
        # Stack compatibility codes
        self.stack_codes = {
            "none": 0,
            "self": 1,
            "size": 2
        }
        
        # Standard container dimensions
        self.standard_dimensions = {
            "TWEU": (6.06, 2.44, 2.59),    # 20-foot: length, width, height
            "THEU": (9.14, 2.44, 2.59),    # 30-foot
            "FEU": (12.19, 2.44, 2.59),    # 40-foot
            "FFEU": (13.72, 2.44, 2.59),   # 45-foot
            "Trailer": (12.19, 2.55, 4.0),
            "Swap Body": (7.45, 2.55, 3.0)
        }
        
        # Standard container weights
        self.standard_weights = {
            "TWEU": 20000,      # 20 tonnes typical for TWEU
            "THEU": 25000,      # 25 tonnes typical for THEU
            "FEU": 30000,       # 30 tonnes typical for FEU
            "FFEU": 32000,      # 32 tonnes typical for FFEU
            "Trailer": 15000,   # Lighter than standard containers
            "Swap Body": 12000  # Lighter than standard containers
        }
        
        # Register warp kernels
        # self._register_kernels()
        
        print(f"WarpContainerRegistry initialized on device: {self.device}")
    
    def _register_kernels(self):
        """Register Warp kernels for container operations."""
        # Register the validation kernel
        # wp.register_kernel(self._kernel_validate_stacking)
    


        # def _kernel_validate_stacking(container_properties: wp.array(dtype=wp.float32),
        #                        container_dimensions: wp.array(dtype=wp.float32),
        #                        stack_heights: wp.array(dtype=wp.int32),
        #                        yard_container_indices: wp.array(dtype=wp.int32),
        #                        container_idx: int,
        #                        row: int,
        #                        bay: int,
        #                        special_area_masks: Dict[str, wp.array],
        #                        results: wp.array(dtype=wp.int32)):
    @wp.kernel
    def _kernel_validate_stacking(container_properties: wp.array,
                               container_dimensions: wp.array,
                               stack_heights: wp.array,
                               yard_container_indices: wp.array,
                               container_idx: int,
                               row: int,
                               bay: int,
                               special_area_masks: Dict[str, wp.array],
                               results: wp.array):
        """
        Warp kernel for validating if a container can be stacked at a position.
        
        Args:
            container_properties: Container properties array
            container_dimensions: Container dimensions array
            stack_heights: Current stack heights array
            yard_container_indices: Container indices in the yard
            container_idx: Index of the container to place
            row: Row index
            bay: Bay index
            special_area_masks: Dictionary of special area masks
            results: Output array for results (0=invalid, 1=valid)
        """
        # Initialize result to invalid
        results[0] = 0
        
        # Get current stack height
        height = stack_heights[row, bay]
        
        # Check if stack is full
        max_height = 5  # Consider making this a parameter
        if height >= max_height:
            return
        
        # Get container properties
        container_type = int(container_properties[container_idx, 0])
        container_goods = int(container_properties[container_idx, 1])
        container_weight = container_properties[container_idx, 3]
        is_stackable = bool(container_properties[container_idx, 4])
        stack_compatibility = int(container_properties[container_idx, 5])
        
        # If container is not stackable, only allow on empty positions
        if not is_stackable and height > 0:
            return
        
        # Check if container can be stacked on the current top container
        if height > 0:
            # Get the top container index and properties
            top_container_idx = yard_container_indices[row, bay, height-1]
            if top_container_idx < 0:
                # This should not happen - height > 0 but no container found
                return
                
            top_container_type = int(container_properties[top_container_idx, 0])
            top_container_goods = int(container_properties[top_container_idx, 1])
            top_container_weight = container_properties[top_container_idx, 3]
            top_container_stackable = bool(container_properties[top_container_idx, 4])
            top_container_compatibility = int(container_properties[top_container_idx, 5])
            
            # Check stackability
            if not top_container_stackable:
                return
                
            # Check stack compatibility
            if stack_compatibility == 0 or top_container_compatibility == 0:
                # None compatibility - can't stack
                return
                
            # Self compatibility - must be same type and goods
            if stack_compatibility == 1 or top_container_compatibility == 1:
                if container_type != top_container_type or container_goods != top_container_goods:
                    return
            
            # Size compatibility - must be same size
            if stack_compatibility == 2:
                if container_type != top_container_type:
                    return
            
            # Check weight constraints - container above should be lighter
            if container_weight > top_container_weight:
                return
        
        # Check special area constraints
        if container_goods == 1:  # Reefer
            if not special_area_masks["reefer"][row, bay]:
                return
        elif container_goods == 2:  # Dangerous
            if not special_area_masks["dangerous"][row, bay]:
                return
        elif container_type == 4:  # Trailer
            if not special_area_masks["trailer"][row, bay]:
                return
        elif container_type == 5:  # Swap Body
            if not special_area_masks["swap_body"][row, bay]:
                return
        
        # All checks passed, container can be stacked here
        results[0] = 1
    
    def create_container(self, 
                        container_id: str = None, 
                        container_type: str = "FEU", 
                        goods_type: str = "Regular",
                        priority: int = 50,
                        weight: float = None,
                        dimensions: Tuple[float, float, float] = None,
                        is_high_cube: bool = False,
                        is_stackable: bool = True,
                        stack_compatibility: str = "size",
                        departure_time: float = 0.0) -> int:
        """
        Create a new container and add it to the registry.
        
        Args:
            container_id: Container ID (generated if None)
            container_type: Container type ("TWEU", "FEU", etc.)
            goods_type: Type of goods ("Regular", "Reefer", "Dangerous")
            priority: Priority value (0-100)
            weight: Container weight in kg (uses standard weight if None)
            dimensions: Container dimensions as (length, width, height) in meters
            is_high_cube: Whether this is a high cube container
            is_stackable: Whether the container can be stacked
            stack_compatibility: Stack compatibility ("none", "self", "size")
            departure_time: Scheduled departure time (simulation seconds)
            
        Returns:
            Container index or -1 if creation failed
        """
        # Generate container ID if not provided
        if container_id is None:
            container_id = f"CONT{uuid.uuid4().hex[:8].upper()}"
        
        # Check if container ID already exists
        if container_id in self.container_id_to_idx:
            print(f"Warning: Container ID {container_id} already exists")
            return self.container_id_to_idx[container_id]
        
        # Convert active flags to NumPy for checking availability
        active_flags = self.terminal_state.container_properties.numpy()[:, 6]
        
        # Find the next available container slot
        container_idx = -1
        for i in range(self.max_containers):
            # Check if slot is inactive
            if active_flags[i] == 0:
                container_idx = i
                break
        
        if container_idx == -1:
            print("Error: No available container slots")
            return -1
        
        # Map container and goods type to codes
        type_code = self.type_codes.get(container_type, 2)  # Default to FEU
        goods_code = self.goods_codes.get(goods_type, 0)    # Default to Regular
        stack_code = self.stack_codes.get(stack_compatibility, 2)  # Default to size
        
        # Set dimensions
        if dimensions is None:
            std_dims = self.standard_dimensions.get(container_type, (12.19, 2.44, 2.59))
            height = std_dims[2]
            if is_high_cube:
                height = 2.90  # High cube is typically 2.90m high
            dimensions = (std_dims[0], std_dims[1], height)
        
        # Set weight
        if weight is None:
            weight = self.standard_weights.get(container_type, 20000)
        
        # Special handling for different container types
        if container_type == "Trailer":
            is_stackable = False
            stack_compatibility = "none"
            stack_code = self.stack_codes["none"]
        elif container_type == "Swap Body":
            stack_compatibility = "none"
            stack_code = self.stack_codes["none"]
        
        # Special handling for goods types
        if goods_type in ["Reefer", "Dangerous"]:
            stack_compatibility = "self"
            stack_code = self.stack_codes["self"]
        
        # Create NumPy arrays for the container properties and dimensions
        property_values = np.array([type_code, goods_code, priority, weight, 
                                1.0 if is_stackable else 0.0, stack_code, 
                                1.0, departure_time], dtype=np.float32)
        
        dimension_values = np.array([dimensions[0], dimensions[1], dimensions[2]], 
                                dtype=np.float32)
        
        # Update Warp arrays using numpy arrays
        # Get container properties for this index
        container_props = self.terminal_state.container_properties.numpy()
        container_dims = self.terminal_state.container_dimensions.numpy()
        
        # Update the arrays
        container_props[container_idx] = property_values
        container_dims[container_idx] = dimension_values
        
        # Create new Warp arrays from the modified NumPy arrays
        self.terminal_state.container_properties = wp.array(container_props, 
                                                        dtype=wp.float32, 
                                                        device=self.device)
        
        self.terminal_state.container_dimensions = wp.array(container_dims, 
                                                        dtype=wp.float32, 
                                                        device=self.device)
        
        # Set container position to "not placed" in a separate step
        positions = self.terminal_state.container_positions.numpy()
        positions[container_idx] = -1
        self.terminal_state.container_positions = wp.array(positions, 
                                                        dtype=wp.int32, 
                                                        device=self.device)
        
        # Update mappings
        self.container_id_to_idx[container_id] = container_idx
        self.container_idx_to_id[container_idx] = container_id
        
        return container_idx
        
    def remove_container(self, container_id_or_idx):
        """
        Remove a container from the registry.
        
        Args:
            container_id_or_idx: Container ID or index
            
        Returns:
            True if container was removed, False otherwise
        """
        # Get container index
        container_idx = self._get_container_idx(container_id_or_idx)
        if container_idx == -1:
            return False
        
        # Check if container is currently placed
        position_idx = self.terminal_state.container_positions[container_idx]
        if position_idx >= 0:
            print("Warning: Container is still placed. Remove from position first.")
            return False
        
        # Mark container as inactive
        self.terminal_state.container_properties[container_idx, 6] = 0.0
        
        # Get container ID from index
        container_id = self.container_idx_to_id.get(container_idx)
        if container_id:
            # Update mappings
            del self.container_id_to_idx[container_id]
            del self.container_idx_to_id[container_idx]
        
        return True
    
    def place_container(self, container_id_or_idx, position_str) -> bool:
        """
        Place a container at a specific position.
        
        Args:
            container_id_or_idx: Container ID or index
            position_str: Position string (e.g., 'A1', 't1_1', 'p_1')
            
        Returns:
            True if container was placed successfully, False otherwise
        """
        # Get container index
        container_idx = self._get_container_idx(container_id_or_idx)
        if container_idx == -1:
            return False
        
        # Check if container is already placed
        if self.terminal_state.container_positions[container_idx] >= 0:
            print(f"Warning: Container {container_id_or_idx} is already placed")
            return False
        
        # Get position index
        if position_str not in self.terminal_state.position_to_idx:
            print(f"Error: Invalid position {position_str}")
            return False
        
        position_idx = self.terminal_state.position_to_idx[position_str]
        
        # Check position type
        position_type = self._get_position_type(position_str)
        
        if position_type == 'storage':
            # Extract row and bay from position string
            row, bay = self._parse_storage_position(position_str)
            if row is None or bay is None:
                return False
            
            # Check if container can be stacked at this position
            if not self.can_place_at(container_idx, position_str):
                return False
            
            # Get current stack height
            height = int(self.terminal_state.stack_heights[row, bay])
            
            # Place container
            self.terminal_state.yard_container_indices[row, bay, height] = container_idx
            self.terminal_state.stack_heights[row, bay] = height + 1
            
        elif position_type == 'truck':
            # Implementation for placing on truck would go here
            # For now, just mark as placed
            pass
            
        elif position_type == 'rail':
            # Implementation for placing on rail would go here
            # For now, just mark as placed
            pass
        
        # Update container position
        self.terminal_state.container_positions[container_idx] = position_idx
        
        return True
    
    def remove_from_position(self, position_str) -> int:
        """
        Remove a container from a position.
        
        Args:
            position_str: Position string (e.g., 'A1', 't1_1', 'p_1')
            
        Returns:
            Container index that was removed or -1 if no container was removed
        """
        # Get position index
        if position_str not in self.terminal_state.position_to_idx:
            print(f"Error: Invalid position {position_str}")
            return -1
        
        # Check position type
        position_type = self._get_position_type(position_str)
        
        if position_type == 'storage':
            # Extract row and bay from position string
            row, bay = self._parse_storage_position(position_str)
            if row is None or bay is None:
                return -1
            
            # Check if there's a container at this position
            height = int(self.terminal_state.stack_heights[row, bay])
            if height <= 0:
                return -1
            
            # Can only remove from the top
            container_idx = self.terminal_state.yard_container_indices[row, bay, height-1]
            if container_idx < 0:
                return -1
            
            # Remove container
            self.terminal_state.yard_container_indices[row, bay, height-1] = -1
            self.terminal_state.stack_heights[row, bay] = height - 1
            
            # Update container position
            self.terminal_state.container_positions[container_idx] = -1
            
            return container_idx
            
        elif position_type == 'truck':
            # Implementation for removing from truck would go here
            # For now just return -1
            return -1
            
        elif position_type == 'rail':
            # Implementation for removing from rail would go here
            # For now just return -1
            return -1
        
        return -1
    
    def get_container_at_position(self, position_str) -> int:
        """
        Get the container index at a position (without removing it).
        
        Args:
            position_str: Position string (e.g., 'A1', 't1_1', 'p_1')
            
        Returns:
            Container index or -1 if no container is at the position
        """
        # Get position type
        position_type = self._get_position_type(position_str)
        
        if position_type == 'storage':
            # Extract row and bay from position string
            row, bay = self._parse_storage_position(position_str)
            if row is None or bay is None:
                return -1
            
            # Check if there's a container at this position
            height = int(self.terminal_state.stack_heights[row, bay])
            if height <= 0:
                return -1
            
            # Return top container
            return self.terminal_state.yard_container_indices[row, bay, height-1]
            
        elif position_type == 'truck':
            # Implementation for getting from truck would go here
            # For now just return -1
            return -1
            
        elif position_type == 'rail':
            # Implementation for getting from rail would go here
            # For now just return -1
            return -1
        
        return -1
    
    def can_place_at(self, container_id_or_idx, position_str) -> bool:
        """
        Check if a container can be placed at a position.
        
        Args:
            container_id_or_idx: Container ID or index
            position_str: Position string (e.g., 'A1', 't1_1', 'p_1')
            
        Returns:
            True if container can be placed, False otherwise
        """
        # Get container index
        container_idx = self._get_container_idx(container_id_or_idx)
        if container_idx == -1:
            return False
        
        # Get position type
        position_type = self._get_position_type(position_str)
        
        if position_type == 'storage':
            # Extract row and bay from position string
            row, bay = self._parse_storage_position(position_str)
            if row is None or bay is None:
                return False
            
            # Use the warp kernel for validation
            results = wp.zeros(1, dtype=wp.int32, device=self.device)
            
            # Run the validation kernel
            wp.launch(
                kernel=self._kernel_validate_stacking,
                dim=1,
                inputs=[
                    self.terminal_state.container_properties,
                    self.terminal_state.container_dimensions,
                    self.terminal_state.stack_heights,
                    self.terminal_state.yard_container_indices,
                    container_idx,
                    row,
                    bay,
                    self.terminal_state.special_area_masks,
                    results
                ]
            )
            
            # Check result
            return bool(results[0])
            
        elif position_type == 'truck':
            # Implementation for placing on truck would go here
            # For now, just return True
            return True
            
        elif position_type == 'rail':
            # Implementation for placing on rail would go here
            # For now, just return True
            return True
        
        return False
    
    def get_container_info(self, container_id_or_idx) -> Dict:
        """
        Get information about a container.
        
        Args:
            container_id_or_idx: Container ID or index
            
        Returns:
            Dictionary with container information
        """
        # Get container index
        container_idx = self._get_container_idx(container_id_or_idx)
        if container_idx == -1:
            return {}
        
        # Get container properties
        props = self.terminal_state.container_properties[container_idx].numpy()
        dimensions = self.terminal_state.container_dimensions[container_idx].numpy()
        position_idx = self.terminal_state.container_positions[container_idx]
        
        # Reverse mappings for codes
        type_codes_rev = {v: k for k, v in self.type_codes.items()}
        goods_codes_rev = {v: k for k, v in self.goods_codes.items()}
        stack_codes_rev = {v: k for k, v in self.stack_codes.items()}
        
        # Get position string
        position_str = "Not placed"
        if position_idx >= 0:
            position_str = self.terminal_state.idx_to_position.get(int(position_idx), "Unknown")
        
        # Build container info dictionary
        container_info = {
            "id": self.container_idx_to_id.get(container_idx, f"Unknown-{container_idx}"),
            "type": type_codes_rev.get(int(props[0]), "Unknown"),
            "goods_type": goods_codes_rev.get(int(props[1]), "Unknown"),
            "priority": int(props[2]),
            "weight": props[3],
            "is_stackable": bool(props[4]),
            "stack_compatibility": stack_codes_rev.get(int(props[5]), "Unknown"),
            "is_active": bool(props[6]),
            "departure_time": props[7],
            "dimensions": {
                "length": dimensions[0],
                "width": dimensions[1],
                "height": dimensions[2]
            },
            "position": position_str,
            "position_idx": int(position_idx) if position_idx >= 0 else -1
        }
        
        return container_info
    
    def _get_container_idx(self, container_id_or_idx) -> int:
        """Convert container ID or index to container index."""
        if isinstance(container_id_or_idx, int):
            # Already an index, check if it's valid
            if 0 <= container_id_or_idx < self.max_containers:
                return container_id_or_idx
        else:
            # Container ID, look up index
            return self.container_id_to_idx.get(container_id_or_idx, -1)
        
        return -1
    
    def _get_position_type(self, position_str) -> str:
        """Determine the type of a position."""
        if position_str.startswith('t') and '_' in position_str:
            return 'rail'
        elif position_str.startswith('p_'):
            return 'truck'
        else:
            return 'storage'
    
    def _parse_storage_position(self, position_str) -> Tuple[Optional[int], Optional[int]]:
        """Parse a storage position string into row and bay indices."""
        # Storage positions are in the format 'A1', 'B2', etc.
        if len(position_str) < 2 or not position_str[0].isalpha():
            return None, None
        
        row_letter = position_str[0].upper()
        try:
            bay = int(position_str[1:]) - 1  # Convert to 0-based index
        except ValueError:
            return None, None
        
        # Convert row letter to index
        row = ord(row_letter) - ord('A')
        
        # Check bounds
        if row < 0 or row >= self.terminal_state.num_storage_rows:
            return None, None
            
        if bay < 0 or bay >= self.terminal_state.num_storage_bays:
            return None, None
        
        return row, bay
    
    def create_random_container(self, 
                               container_id: str = None,
                               departure_time: float = None) -> int:
        """
        Create a random container based on probability distributions.
        
        Args:
            container_id: Container ID (generated if None)
            departure_time: Departure time in simulation seconds (random if None)
            
        Returns:
            Container index or -1 if creation failed
        """
        # Generate container ID if not provided
        if container_id is None:
            container_id = f"CONT{uuid.uuid4().hex[:8].upper()}"
        
        # Define probability weights for container types
        container_types = list(self.type_codes.keys())
        container_type_weights = []
        
        # Try to access terminal config if available via terminal_state
        if hasattr(self.terminal_state, 'terminal_config') and self.terminal_state.terminal_config:
            # Get probabilities from config
            config = self.terminal_state.terminal_config
            probs = config.get_container_type_probabilities()
            if probs:
                # Map from config to container types
                length_probs = probs.get('length', {})
                container_type_weights = [
                    length_probs.get('20', {}).get('probability', 0.177),        # TWEU
                    length_probs.get('30', {}).get('probability', 0.018),        # THEU
                    length_probs.get('40', {}).get('probability', 0.521),        # FEU
                    length_probs.get('40', {}).get('probability', 0.0) * 
                    length_probs.get('40', {}).get('probability_high_cube', 0.1115),  # FFEU (40-foot High Cube)
                    length_probs.get('trailer', {}).get('probability', 0.033),   # Trailer
                    length_probs.get('swap body', {}).get('probability', 0.251)  # Swap Body
                ]
        
        # Fallback if no config available or empty weights
        if not container_type_weights or sum(container_type_weights) == 0:
            container_type_weights = [0.532, 0.026, 0.180, 0.032, 0.014, 0.216]  # Default weights
        
        # Normalize weights to ensure they sum to 1
        weight_sum = sum(container_type_weights)
        if weight_sum > 0:
            container_type_weights = [w / weight_sum for w in container_type_weights]
        else:
            # If all weights are zero, use uniform distribution
            container_type_weights = [1.0 / len(container_types) for _ in container_types]

        # Random container type
        container_type = np.random.choice(container_types, p=container_type_weights)
        
        # Random goods type
        goods_type = "Regular"
        if container_type in ["TWEU", "THEU", "FEU", "FFEU"]:
            # Dangerous goods probability depends on container type
            dg_probabilities = {
                "TWEU": 0.0134, "FEU": 0.0023, "FFEU": 0.0, "THEU": 0.2204
            }
            
            if np.random.random() < dg_probabilities.get(container_type, 0.01):
                goods_type = "Dangerous"
            elif np.random.random() < 0.0066:  # Reefer probability
                goods_type = "Reefer"
        
        # Random high cube for standard containers
        is_high_cube = False
        if container_type in ["TWEU", "THEU", "FEU", "FFEU"]:
            is_high_cube = np.random.random() < 0.3
        
        # Random priority
        priority = np.random.randint(1, 101)
        
        # Random weight - slight variation from standard
        std_weight = self.standard_weights.get(container_type, 20000)
        weight = std_weight * (0.8 + 0.4 * np.random.random())  # ±20% variation
        
        # Random departure time if not provided
        if departure_time is None:
            # Between 1 and 10 days from now
            departure_time = float(self.terminal_state.simulation_time.numpy()[0]) + np.random.randint(86400, 864000)
        
        # Create the container
        return self.create_container(
            container_id=container_id,
            container_type=container_type,
            goods_type=goods_type,
            priority=priority,
            weight=weight,
            is_high_cube=is_high_cube,
            departure_time=departure_time
        )