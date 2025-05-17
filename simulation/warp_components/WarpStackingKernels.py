import warp as wp
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import time


# Standalone kernel functions for stacking validation
@wp.kernel
def kernel_validate_container_placement(container_properties: wp.array(dtype=wp.float32, ndim=2),
                                    container_dimensions: wp.array(dtype=wp.float32, ndim=2),
                                    yard_container_indices: wp.array(dtype=wp.int32, ndim=3),
                                    stack_heights: wp.array(dtype=wp.int32, ndim=2),
                                    container_idx: wp.int32,
                                    row: wp.int32,
                                    bay: wp.int32,
                                    max_height: wp.int32,
                                    result: wp.array(dtype=wp.int32, ndim=1)):
    """Kernel to check if a container can be placed at a specific position."""
    # Default to invalid
    result[0] = 0
    
    # Get current stack height
    height = stack_heights[row, bay]
    
    # Check if stack is full
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
        # Get the top container index
        top_container_idx = yard_container_indices[row, bay, height-1]
        
        # Invalid if no container at top (should not happen)
        if top_container_idx < 0:
            return
            
        # Get top container properties
        top_type = int(container_properties[top_container_idx, 0])
        top_goods = int(container_properties[top_container_idx, 1])
        top_weight = container_properties[top_container_idx, 3]
        top_stackable = bool(container_properties[top_container_idx, 4])
        top_compatibility = int(container_properties[top_container_idx, 5])
        
        # Check stackability
        if not top_stackable:
            return
            
        # Check stack compatibility
        if stack_compatibility == 0 or top_compatibility == 0:
            # None compatibility - can't stack
            return
            
        # Self compatibility - must be same type and goods
        if stack_compatibility == 1 or top_compatibility == 1:
            if container_type != top_type or container_goods != top_goods:
                return
        
        # Size compatibility - must be same size
        if stack_compatibility == 2 or top_compatibility == 2:
            if container_type != top_type:
                return
        
        # Weight constraint - container above should be lighter
        if container_weight > top_weight:
            return
    
    # All checks passed, container can be stacked here
    result[0] = 1


class WarpStackingKernels:
    """
    GPU-accelerated kernels for container stacking operations using NVIDIA Warp.
    Provides specialized kernels for stacking validation, optimization, and pre-marshalling.
    """
    
    def __init__(self, 
                 terminal_state,
                 container_registry,
                 device: str = None):
        """
        Initialize the stacking kernels.
        
        Args:
            terminal_state: Reference to the WarpTerminalState object
            container_registry: Reference to the WarpContainerRegistry object
            device: Computation device (if None, will use the terminal_state's device)
        """
        self.terminal_state = terminal_state
        self.container_registry = container_registry
        self.device = device if device else terminal_state.device
        
        # Performance tracking
        self.validation_times = []
        self.optimization_times = []
        self.lookup_times = []
        
        # Compatibility matrix for optimized lookups
        self.compatibility_matrix_np = None
        self.use_matrix_lookup = False
        
        # Initialize flag to track when we've initialized the matrix
        self._matrix_initialized = False
        
        print(f"WarpStackingKernels initialized on device: {self.device}")
    
    def use_compatibility_matrix(self, compatibility_matrix: np.ndarray) -> None:
        """
        Configure the stacking kernels to use a precomputed compatibility matrix.
        
        Args:
            compatibility_matrix: Numpy array of precomputed stacking compatibility
        """
        if compatibility_matrix is None:
            self.use_matrix_lookup = False
            self._matrix_initialized = False
            return
            
        self.compatibility_matrix_np = compatibility_matrix
        self.use_matrix_lookup = True
        self._matrix_initialized = True
        
        print(f"Stacking kernels configured to use compatibility matrix")
        print(f"Matrix shape: {compatibility_matrix.shape}")
        print(f"Matrix memory usage: {compatibility_matrix.nbytes / (1024 * 1024):.2f} MB")
    
    def calculate_compatibility_matrix(self) -> np.ndarray:
        """
        Calculate the full container-to-container compatibility matrix.
        
        Returns:
            Numpy array containing compatibility results [upper_container, lower_container]
        """
        start_time = time.time()
        
        # Create numpy array for compatibility matrix
        max_containers = self.terminal_state.max_containers
        compatibility_matrix = np.zeros((max_containers, max_containers), dtype=np.int32)
        
        # Get container properties as numpy array for efficient processing
        container_props = self.terminal_state.container_properties.numpy()
        
        # Precompute for each container pair
        for upper_idx in range(max_containers):
            # Skip inactive containers
            if container_props[upper_idx, 6] <= 0:  # Check active flag
                continue
                
            for lower_idx in range(max_containers):
                # Skip inactive containers and self-stacking
                if container_props[lower_idx, 6] <= 0 or upper_idx == lower_idx:
                    continue
                
                # Get container properties
                upper_type = int(container_props[upper_idx, 0])
                upper_goods = int(container_props[upper_idx, 1])
                upper_weight = float(container_props[upper_idx, 3])
                upper_compatibility = int(container_props[upper_idx, 5])
                
                lower_type = int(container_props[lower_idx, 0])
                lower_goods = int(container_props[lower_idx, 1])
                lower_weight = float(container_props[lower_idx, 3])
                lower_stackable = bool(container_props[lower_idx, 4])
                lower_compatibility = int(container_props[lower_idx, 5])
                
                # Can't stack on non-stackable container
                if not lower_stackable:
                    continue
                
                # Check compatibility
                can_stack = True
                
                # None compatibility - can't stack
                if lower_compatibility == 0 or upper_compatibility == 0:
                    can_stack = False
                
                # Self compatibility - must be same type and goods
                elif lower_compatibility == 1 or upper_compatibility == 1:
                    if lower_type != upper_type or lower_goods != upper_goods:
                        can_stack = False
                
                # Size compatibility - must be same size
                elif lower_compatibility == 2 or upper_compatibility == 2:
                    if lower_type != upper_type:
                        can_stack = False
                
                # Weight constraint - container above should be lighter
                if upper_weight > lower_weight:
                    can_stack = False
                
                # Update matrix
                if can_stack:
                    compatibility_matrix[upper_idx, lower_idx] = 1
        
        calculation_time = time.time() - start_time
        print(f"Compatibility matrix calculated in {calculation_time:.2f} seconds")
        
        # Store matrix for later use
        self.compatibility_matrix_np = compatibility_matrix
        self.use_matrix_lookup = True
        self._matrix_initialized = True
        
        return compatibility_matrix
    
    def can_place_at(self, container_idx: int, position_str: str) -> bool:
        """
        Check if a container can be placed at a position.
        
        Args:
            container_idx: Container index
            position_str: Position string
            
        Returns:
            True if container can be placed, False otherwise
        """
        # Initialize the compatibility matrix if needed and flagged to use it
        if self.use_matrix_lookup and not self._matrix_initialized:
            self.calculate_compatibility_matrix()
            
        lookup_start = time.time()
        
        # Use compatibility matrix if available and enabled
        if self.use_matrix_lookup and self.compatibility_matrix_np is not None:
            # Parse position to find existing container
            if len(position_str) < 2 or not position_str[0].isalpha():
                return False
                
            row_letter = position_str[0].upper()
            
            try:
                bay_num = int(position_str[1:])
            except ValueError:
                return False
            
            # Convert to indices
            if not hasattr(self.terminal_state, 'row_names') or row_letter not in self.terminal_state.row_names:
                return False
                
            row_idx = self.terminal_state.row_names.index(row_letter)
            bay_idx = bay_num - 1  # Convert to 0-based index
            
            # Check bounds
            if row_idx < 0 or row_idx >= self.terminal_state.num_storage_rows:
                return False
                
            if bay_idx < 0 or bay_idx >= self.terminal_state.num_storage_bays:
                return False
            
            # Get stack height
            stack_heights_np = self.terminal_state.stack_heights.numpy()
            height = int(stack_heights_np[row_idx, bay_idx])
            
            # Empty stack - container can be placed if valid
            if height == 0:
                self.lookup_times.append(time.time() - lookup_start)
                return True
            
            # Check if stack is full
            if height >= self.terminal_state.max_stack_height:
                self.lookup_times.append(time.time() - lookup_start)
                return False
            
            # Get top container index
            yard_indices_np = self.terminal_state.yard_container_indices.numpy()
            lower_container_idx = int(yard_indices_np[row_idx, bay_idx, height-1])
            
            # Check if we got a valid container
            if lower_container_idx < 0:
                self.lookup_times.append(time.time() - lookup_start)
                return False
                
            # Check compatibility matrix
            try:
                result = bool(self.compatibility_matrix_np[container_idx, lower_container_idx])
                self.lookup_times.append(time.time() - lookup_start)
                return result
            except (IndexError, TypeError):
                # Fall back to original method if lookup fails
                pass
        
        # Fall back to standard calculation
        result = self._can_place_at_original(container_idx, position_str)
        self.lookup_times.append(time.time() - lookup_start)
        return result
    
    def _can_place_at_original(self, container_idx: int, position_str: str) -> bool:
        """
        Original stacking validation method using direct property checks.
        
        Args:
            container_idx: Container index
            position_str: Position string
            
        Returns:
            True if container can be placed, False otherwise
        """
        start_time = time.time()
        
        if container_idx < 0:
            return False
        
        # Parse position string
        if len(position_str) < 2 or not position_str[0].isalpha():
            return False
            
        row_letter = position_str[0].upper()
        
        try:
            bay_num = int(position_str[1:])
        except ValueError:
            return False
        
        # Convert to indices
        if not hasattr(self.terminal_state, 'row_names') or row_letter not in self.terminal_state.row_names:
            return False
            
        row_idx = self.terminal_state.row_names.index(row_letter)
        bay_idx = bay_num - 1  # Convert to 0-based index
        
        # Check bounds
        if row_idx < 0 or row_idx >= self.terminal_state.num_storage_rows:
            return False
            
        if bay_idx < 0 or bay_idx >= self.terminal_state.num_storage_bays:
            return False
        
        # Get the stack height at this position
        stack_heights_np = self.terminal_state.stack_heights.numpy()
        height = int(stack_heights_np[row_idx, bay_idx])
        
        # Check if stack is full
        if height >= self.terminal_state.max_stack_height:
            return False
        
        # Get container properties
        container_props = self.terminal_state.container_properties.numpy()
        container_type = int(container_props[container_idx, 0])
        container_goods = int(container_props[container_idx, 1])
        container_weight = float(container_props[container_idx, 3])
        is_stackable = bool(container_props[container_idx, 4])
        stack_compatibility = int(container_props[container_idx, 5])
        
        # If container is not stackable, only allow on empty positions
        if not is_stackable and height > 0:
            return False
        
        # Check if container can be stacked on the current top container
        if height > 0:
            # Get the yard container indices
            yard_indices = self.terminal_state.yard_container_indices.numpy()
            
            # Get top container index
            top_container_idx = int(yard_indices[row_idx, bay_idx, height-1])
            
            # Skip if invalid container (should not happen)
            if top_container_idx < 0:
                return False
                
            # Get top container properties
            top_type = int(container_props[top_container_idx, 0])
            top_goods = int(container_props[top_container_idx, 1])
            top_weight = float(container_props[top_container_idx, 3])
            top_stackable = bool(container_props[top_container_idx, 4])
            top_compatibility = int(container_props[top_container_idx, 5])
            
            # Check stackability
            if not top_stackable:
                return False
                
            # Check stack compatibility
            if stack_compatibility == 0 or top_compatibility == 0:
                # None compatibility - can't stack
                return False
                
            # Self compatibility - must be same type and goods
            if stack_compatibility == 1 or top_compatibility == 1:
                if container_type != top_type or container_goods != top_goods:
                    return False
            
            # Size compatibility - must be same size
            if stack_compatibility == 2 or top_compatibility == 2:
                if container_type != top_type:
                    return False
            
            # Weight constraint - container above should be lighter
            if container_weight > top_weight:
                return False
        
        # All checks passed, container can be stacked here
        self.validation_times.append(time.time() - start_time)
        return True
    
    def are_compatible(self, upper_container_idx: int, lower_container_idx: int) -> bool:
        """
        Check if one container can be stacked on another.
        
        Args:
            upper_container_idx: Index of container to be placed on top
            lower_container_idx: Index of container at the bottom
            
        Returns:
            True if containers are compatible for stacking, False otherwise
        """
        # Initialize the compatibility matrix if needed and flagged to use it
        if self.use_matrix_lookup and not self._matrix_initialized:
            self.calculate_compatibility_matrix()
        
        start_time = time.time()
        
        # Use compatibility matrix if available
        if self.use_matrix_lookup and self.compatibility_matrix_np is not None:
            try:
                result = bool(self.compatibility_matrix_np[upper_container_idx, lower_container_idx])
                self.lookup_times.append(time.time() - start_time)
                return result
            except (IndexError, TypeError):
                # Fall back to direct check if lookup fails
                pass
        
        # Direct property check if matrix not available
        container_props = self.terminal_state.container_properties.numpy()
        
        # Check bounds and active status
        if (upper_container_idx < 0 or upper_container_idx >= len(container_props) or
            lower_container_idx < 0 or lower_container_idx >= len(container_props) or
            container_props[upper_container_idx, 6] <= 0 or  # Check active flag
            container_props[lower_container_idx, 6] <= 0):   # Check active flag
            return False
        
        # Get container properties
        upper_type = int(container_props[upper_container_idx, 0])
        upper_goods = int(container_props[upper_container_idx, 1])
        upper_weight = float(container_props[upper_container_idx, 3])
        upper_compatibility = int(container_props[upper_container_idx, 5])
        
        lower_type = int(container_props[lower_container_idx, 0])
        lower_goods = int(container_props[lower_container_idx, 1])
        lower_weight = float(container_props[lower_container_idx, 3])
        lower_stackable = bool(container_props[lower_container_idx, 4])
        lower_compatibility = int(container_props[lower_container_idx, 5])
        
        # Can't stack on non-stackable container
        if not lower_stackable:
            return False
        
        # Check compatibility
        can_stack = True
        
        # None compatibility - can't stack
        if lower_compatibility == 0 or upper_compatibility == 0:
            can_stack = False
        
        # Self compatibility - must be same type and goods
        elif lower_compatibility == 1 or upper_compatibility == 1:
            if lower_type != upper_type or lower_goods != upper_goods:
                can_stack = False
        
        # Size compatibility - must be same size
        elif lower_compatibility == 2 or upper_compatibility == 2:
            if lower_type != upper_type:
                can_stack = False
        
        # Weight constraint - container above should be lighter
        if upper_weight > lower_weight:
            can_stack = False
        
        self.validation_times.append(time.time() - start_time)
        return can_stack
    
    def validate_stack(self, row: int, bay: int) -> bool:
        """
        Validate if a stack at the given position is valid according to stacking rules.
        
        Args:
            row: Row index
            bay: Bay index
            
        Returns:
            True if the stack is valid, False otherwise
        """
        start_time = time.time()
        
        # Create result array on device
        valid = wp.zeros(1, dtype=wp.int32, device=self.device)
        
        # Run the validation kernel
        wp.launch(
            kernel=self._kernel_validate_stack,
            dim=1,
            inputs=[
                self.terminal_state.container_properties,
                self.terminal_state.container_dimensions,
                self.terminal_state.yard_container_indices,
                self.terminal_state.stack_heights,
                row,
                bay,
                self.terminal_state.max_stack_height,
                valid
            ]
        )
        
        result = bool(valid[0])
        
        # Track performance
        self.validation_times.append(time.time() - start_time)
        
        return result
    
    @wp.kernel
    def _kernel_validate_stack(container_properties: wp.array(dtype=wp.float32, ndim=2),
                            container_dimensions: wp.array(dtype=wp.float32, ndim=2),
                            yard_container_indices: wp.array(dtype=wp.int32, ndim=3),
                            stack_heights: wp.array(dtype=wp.int32, ndim=2),
                            row: wp.int32,
                            bay: wp.int32,
                            max_height: wp.int32,
                            valid: wp.array(dtype=wp.int32, ndim=1)):
        """
        Kernel to validate if a stack is valid according to stacking rules.
        
        Args:
            container_properties: Container properties array
            container_dimensions: Container dimensions array
            yard_container_indices: Container indices in the yard
            stack_heights: Current stack heights array
            row: Row index to validate
            bay: Bay index to validate
            max_height: Maximum allowable stack height
            valid: Output array with validation result (0=invalid, 1=valid)
        """
        # Default to valid
        valid[0] = 1
        
        # Get the stack height
        height = stack_heights[row, bay]
        
        # Check height constraint
        if height > max_height:
            valid[0] = 0
            return
        
        # Check if stack is empty
        if height <= 0:
            return
        
        # Validate each container in the stack
        for i in range(height-1):
            # Get container indices
            lower_idx = yard_container_indices[row, bay, i]
            upper_idx = yard_container_indices[row, bay, i+1]
            
            # Skip if either container is invalid
            if lower_idx < 0 or upper_idx < 0:
                valid[0] = 0
                return
            
            # Get container properties
            lower_type = int(container_properties[lower_idx, 0])
            lower_goods = int(container_properties[lower_idx, 1])
            lower_weight = container_properties[lower_idx, 3]
            lower_stackable = bool(container_properties[lower_idx, 4])
            lower_compat = int(container_properties[lower_idx, 5])
            
            upper_type = int(container_properties[upper_idx, 0])
            upper_goods = int(container_properties[upper_idx, 1])
            upper_weight = container_properties[upper_idx, 3]
            upper_stackable = bool(container_properties[upper_idx, 4])
            upper_compat = int(container_properties[upper_idx, 5])
            
            # Check stackability
            if not lower_stackable:
                valid[0] = 0
                return
            
            # Check compatibility
            if lower_compat == 0 or upper_compat == 0:
                # None compatibility - can't stack
                valid[0] = 0
                return
            
            # Self compatibility - must be same type and goods
            if lower_compat == 1 or upper_compat == 1:
                if lower_type != upper_type or lower_goods != upper_goods:
                    valid[0] = 0
                    return
            
            # Size compatibility - must be same size
            if lower_compat == 2 or upper_compat == 2:
                if lower_type != upper_type:
                    valid[0] = 0
                    return
            
            # Weight constraint - container above should be lighter
            if upper_weight > lower_weight:
                valid[0] = 0
                return
    
    @wp.kernel
    def _kernel_get_container_properties(container_properties: wp.array(dtype=wp.float32, ndim=2),
                                    container_idx: wp.int32,
                                    property_idx: wp.int32,
                                    result: wp.array(dtype=wp.float32, ndim=1)):
        """Get a property of a container."""
        if container_idx >= 0:
            result[0] = container_properties[container_idx, property_idx]
        else:
            result[0] = -1.0

    @wp.kernel
    def _kernel_calculate_stack_quality(container_properties: wp.array(dtype=wp.float32, ndim=2),
                                    yard_container_indices: wp.array(dtype=wp.int32, ndim=3),
                                    stack_heights: wp.array(dtype=wp.int32, ndim=2),
                                    current_time: wp.float32,
                                    row: wp.int32,
                                    bay: wp.int32,
                                    quality_score: wp.array(dtype=wp.float32, ndim=1)):
        """
        Kernel to calculate the quality of a stack based on priority order and departure times.
        Higher score means better organization.
        
        Args:
            container_properties: Container properties array
            yard_container_indices: Container indices in the yard
            stack_heights: Current stack heights array
            current_time: Current simulation time
            row: Row index to evaluate
            bay: Bay index to evaluate
            quality_score: Output array with quality score (higher is better)
        """
        # Initialize quality score
        quality_score[0] = 100.0
        
        # Get the stack height
        height = stack_heights[row, bay]
        
        # Empty stacks are neutral
        if height <= 0:
            return
        
        # Single container stacks are good
        if height == 1:
            return
        
        # Check priority order
        priority_violations = int(0)
        for i in range(height-1):
            # Get container indices
            lower_idx = yard_container_indices[row, bay, i]
            upper_idx = yard_container_indices[row, bay, i+1]
            
            # Skip if either container is invalid
            if lower_idx < 0 or upper_idx < 0:
                quality_score[0] = float(quality_score[0] - 20.0)
                continue
            
            # Get container priorities and departure times
            lower_priority = int(container_properties[lower_idx, 2])
            upper_priority = int(container_properties[upper_idx, 2])
            
            lower_departure = container_properties[lower_idx, 7]
            upper_departure = container_properties[upper_idx, 7]
            
            # Check priority order (higher priority should be on top)
            # Lower priority number means higher actual priority
            if lower_priority < upper_priority:
                priority_violations = int(priority_violations + 1)
                quality_score[0] = float(quality_score[0] - 10.0)
            
            # Check departure time order (containers leaving sooner should be on top)
            if lower_departure > 0 and upper_departure > 0:
                if lower_departure < upper_departure:
                    quality_score[0] = float(quality_score[0] - 15.0)
                    
                    # Extra penalty if lower container is departing soon
                    time_until_departure = lower_departure - current_time
                    if time_until_departure > 0 and time_until_departure < 86400:  # Less than 1 day
                        quality_score[0] = float(quality_score[0] - 25.0)
        
        # Overall stack quality factors
        if height > 3:
            # Tall stacks are slightly penalized
            quality_score[0] = float(quality_score[0] - float(height - 3) * 5.0)
        
        # Ensure score stays in reasonable bounds
        if quality_score[0] < 0:
            quality_score[0] = 0.0

    @wp.kernel
    def _kernel_find_optimal_locations(container_properties: wp.array(dtype=wp.float32, ndim=2),
                                    container_dimensions: wp.array(dtype=wp.float32, ndim=2),
                                    yard_container_indices: wp.array(dtype=wp.int32, ndim=3),  # Fixed dtype, was float32
                                    stack_heights: wp.array(dtype=wp.int32, ndim=2),
                                    reefer_mask: wp.array(dtype=wp.bool, ndim=2),
                                    dangerous_mask: wp.array(dtype=wp.bool, ndim=2),
                                    trailer_mask: wp.array(dtype=wp.bool, ndim=2),
                                    swap_body_mask: wp.array(dtype=wp.bool, ndim=2),
                                    container_idx: wp.int32,
                                    suitability_scores: wp.array(dtype=wp.float32, ndim=2),
                                    num_rows: wp.int32,
                                    num_bays: wp.int32):
        """
        Kernel to find optimal locations for a container based on stacking rules.
        
        Args:
            container_properties: Container properties array
            container_dimensions: Container dimensions array
            yard_container_indices: Container indices in the yard
            stack_heights: Current stack heights array
            reefer_mask: Mask for reefer container positions
            dangerous_mask: Mask for dangerous goods positions
            trailer_mask: Mask for trailer positions
            swap_body_mask: Mask for swap body positions
            container_idx: Index of the container to place
            suitability_scores: Output array with suitability scores [row, bay]
            num_rows: Number of rows in the yard
            num_bays: Number of bays per row
        """
        # Get thread indices
        row, bay = wp.tid()
        
        # Check bounds
        if row >= num_rows or bay >= num_bays:
            return
        
        # Initialize suitability to 0 (not suitable)
        suitability_scores[row, bay] = 0.0
        
        # Get stack height
        height = stack_heights[row, bay]
        
        # Check max height constraint
        max_height = 5  # Should be parameterized
        if height >= max_height:
            return
        
        # Get container properties
        container_type = int(container_properties[container_idx, 0])
        container_goods = int(container_properties[container_idx, 1])
        container_weight = container_properties[container_idx, 3]
        is_stackable = bool(container_properties[container_idx, 4])
        stack_compatibility = int(container_properties[container_idx, 5])
        
        # Special area constraints - using individual mask arrays instead of dictionary
        if container_goods == 1:  # Reefer
            if not reefer_mask[row, bay]:
                return
        elif container_goods == 2:  # Dangerous
            if not dangerous_mask[row, bay]:
                return
        elif container_type == 4:  # Trailer
            if not trailer_mask[row, bay]:
                return
        elif container_type == 5:  # Swap Body
            if not swap_body_mask[row, bay]:
                return
        
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
                
            # Calculate suitability score - higher is better
            suitability = 100.0
            
            # Prefer positions with same container type
            if container_type == top_container_type:
                suitability += 20.0
            
            # Prefer positions with same goods type
            if container_goods == top_container_goods:
                suitability += 10.0
            
            # Prefer lower stacks
            suitability -= float(height * 5)
            
            # Prefer positions closer to the departure point
            # (proximity calculation would go here)
            
            suitability_scores[row, bay] = suitability
        else:
            # Empty position - always valid but lower priority than stacking
            suitability_scores[row, bay] = 50.0

    @wp.kernel
    def _kernel_identify_suboptimal_stacks(container_properties: wp.array(dtype=wp.float32, ndim=2),
                                        yard_container_indices: wp.array(dtype=wp.int32, ndim=3),
                                        stack_heights: wp.array(dtype=wp.int32, ndim=2),
                                        current_time: wp.float32,
                                        problem_scores: wp.array(dtype=wp.float32, ndim=2),
                                        num_rows: wp.int32,
                                        num_bays: wp.int32):
        """
        Kernel to identify suboptimal stacks that need reshuffling.
        
        Args:
            container_properties: Container properties array
            yard_container_indices: Container indices in the yard
            stack_heights: Current stack heights array
            current_time: Current simulation time
            problem_scores: Output array with problem scores [row, bay] (higher score = more problematic)
            num_rows: Number of rows in the yard
            num_bays: Number of bays per row
        """
        # Get thread indices
        row, bay = wp.tid()
        
        # Check bounds
        if row >= num_rows or bay >= num_bays:
            return
        
        # Initialize problem score to 0 (no problems)
        problem_scores[row, bay] = 0.0
        
        # Get stack height
        height = stack_heights[row, bay]
        
        # Empty or single container stacks have no problems
        if height <= 1:
            return
        
        # Check for priority and departure time violations
        for i in range(height-1):
            # Get container indices
            lower_idx = yard_container_indices[row, bay, i]
            upper_idx = yard_container_indices[row, bay, i+1]
            
            # Skip if either container is invalid
            if lower_idx < 0 or upper_idx < 0:
                continue
            
            # Get container priorities and departure times
            lower_priority = int(container_properties[lower_idx, 2])
            upper_priority = int(container_properties[upper_idx, 2])
            
            lower_departure = container_properties[lower_idx, 7]
            upper_departure = container_properties[upper_idx, 7]
            
            # Check priority order (higher priority should be on top)
            # Lower priority number means higher actual priority
            if lower_priority < upper_priority:
                # Higher priority container is below lower priority
                problem_scores[row, bay] += 10.0
            
            # Check departure time order (containers leaving sooner should be on top)
            if lower_departure > 0 and upper_departure > 0:
                if lower_departure < upper_departure:
                    # Container leaving sooner is below
                    problem_scores[row, bay] += 15.0
                    
                    # Urgent problem if departure is imminent
                    time_until_departure = lower_departure - current_time
                    if time_until_departure > 0:
                        if time_until_departure < 3600:  # Less than 1 hour
                            problem_scores[row, bay] += 100.0
                        elif time_until_departure < 86400:  # Less than 1 day
                            problem_scores[row, bay] += 50.0
                        elif time_until_departure < 172800:  # Less than 2 days
                            problem_scores[row, bay] += 20.0
    
    def calculate_stack_quality(self, row: int, bay: int) -> float:
        """
        Calculate the quality score for a stack at the given position.
        
        Args:
            row: Row index
            bay: Bay index
            
        Returns:
            Quality score (higher is better)
        """
        # Create result array on device
        quality_score = wp.zeros(1, dtype=wp.float32, device=self.device)
        
        # Run the quality calculation kernel
        wp.launch(
            kernel=self._kernel_calculate_stack_quality,
            dim=1,
            inputs=[
                self.terminal_state.container_properties,
                self.terminal_state.yard_container_indices,
                self.terminal_state.stack_heights,
                float(self.terminal_state.simulation_time[0]),
                row,
                bay,
                quality_score
            ]
        )
        
        return float(quality_score[0])
    
    def find_optimal_location(self, container_idx: int) -> Tuple[int, int, float]:
        """
        Find the optimal location for a container in the yard.
        
        Args:
            container_idx: Index of the container to place
            
        Returns:
            Tuple of (row, bay, score) for the best location, or (-1, -1, 0) if no valid location
        """
        start_time = time.time()
        
        # Create suitability scores array on device
        suitability_scores = wp.zeros(
            (self.terminal_state.num_storage_rows, self.terminal_state.num_storage_bays),
            dtype=wp.float32,
            device=self.device
        )
        
        # Run the optimal location kernel with unpacked special area masks
        wp.launch(
            kernel=self._kernel_find_optimal_locations,
            dim=[self.terminal_state.num_storage_rows, self.terminal_state.num_storage_bays],
            inputs=[
                self.terminal_state.container_properties,
                self.terminal_state.container_dimensions,
                self.terminal_state.yard_container_indices,
                self.terminal_state.stack_heights,
                self.terminal_state.special_area_masks['reefer'],
                self.terminal_state.special_area_masks['dangerous'],
                self.terminal_state.special_area_masks['trailer'],
                self.terminal_state.special_area_masks['swap_body'],
                container_idx,
                suitability_scores,
                self.terminal_state.num_storage_rows,
                self.terminal_state.num_storage_bays
            ]
        )
        
        # Find the best location
        scores = suitability_scores.numpy()
        best_score = 0.0
        best_row = -1
        best_bay = -1
        
        for row in range(self.terminal_state.num_storage_rows):
            for bay in range(self.terminal_state.num_storage_bays):
                if scores[row, bay] > best_score:
                    best_score = scores[row, bay]
                    best_row = row
                    best_bay = bay
        
        # Track performance
        self.optimization_times.append(time.time() - start_time)
        
        return best_row, best_bay, best_score
    
    def identify_problem_stacks(self, threshold: float = 50.0) -> List[Tuple[int, int, float]]:
        """
        Identify stacks that need reshuffling due to priority or departure time issues.
        
        Args:
            threshold: Problem score threshold (stacks with scores above this are problematic)
            
        Returns:
            List of (row, bay, score) tuples for problematic stacks, sorted by score (highest first)
        """
        start_time = time.time()
        
        # Create problem scores array on device
        problem_scores = wp.zeros(
            (self.terminal_state.num_storage_rows, self.terminal_state.num_storage_bays),
            dtype=wp.float32,
            device=self.device
        )
        
        # Run the problem identification kernel
        wp.launch(
            kernel=self._kernel_identify_suboptimal_stacks,
            dim=[self.terminal_state.num_storage_rows, self.terminal_state.num_storage_bays],
            inputs=[
                self.terminal_state.container_properties,
                self.terminal_state.yard_container_indices,
                self.terminal_state.stack_heights,
                float(self.terminal_state.simulation_time[0]),
                problem_scores,
                self.terminal_state.num_storage_rows,
                self.terminal_state.num_storage_bays
            ]
        )
        
        # Find problematic stacks
        scores = problem_scores.numpy()
        problem_stacks = []
        
        for row in range(self.terminal_state.num_storage_rows):
            for bay in range(self.terminal_state.num_storage_bays):
                if scores[row, bay] >= threshold:
                    problem_stacks.append((row, bay, float(scores[row, bay])))
        
        # Sort by score (highest first)
        problem_stacks.sort(key=lambda x: x[2], reverse=True)
        
        # Track performance
        self.optimization_times.append(time.time() - start_time)
        
        return problem_stacks
    
    def generate_premarshalling_plan(self, max_moves: int = 10) -> List[Tuple[int, int, int, int]]:
        """
        Generate a plan for pre-marshalling (reshuffling) the yard to optimize stacking.
        
        Args:
            max_moves: Maximum number of moves to include in the plan
            
        Returns:
            List of (src_row, src_bay, dst_row, dst_bay) tuples representing moves
        """
        start_time = time.time()
        
        # Identify problem stacks
        problem_stacks = self.identify_problem_stacks()
        
        # Generate moves
        moves = []
        
        for src_row, src_bay, score in problem_stacks[:max_moves]:
            # Get the top container from this stack
            height = int(self.terminal_state.stack_heights[src_row, src_bay])
            if height <= 0:
                continue
                
            container_idx = self.terminal_state.yard_container_indices[src_row, src_bay, height-1]
            if container_idx < 0:
                continue
            
            # Find an optimal destination for this container
            dst_row, dst_bay, dst_score = self.find_optimal_location(container_idx)
            
            # Only include the move if we found a valid destination
            if dst_row >= 0 and dst_bay >= 0 and dst_score > 0:
                # Make sure we're not moving to the same position
                if src_row != dst_row or src_bay != dst_bay:
                    moves.append((src_row, src_bay, dst_row, dst_bay))
                    
                    # If we have enough moves, stop
                    if len(moves) >= max_moves:
                        break
        
        # Track performance
        self.optimization_times.append(time.time() - start_time)
        
        return moves
    
    def print_performance_stats(self):
        """Print performance statistics for stacking operations."""
        print("\nStacking Operations Performance Statistics:")
        
        if self.validation_times:
            avg_validation = sum(self.validation_times) / len(self.validation_times) * 1000
            print(f"  Stack validation: {avg_validation:.2f}ms average")
        
        if self.optimization_times:
            avg_optimization = sum(self.optimization_times) / len(self.optimization_times) * 1000
            print(f"  Optimization operations: {avg_optimization:.2f}ms average")
        
        if self.lookup_times:
            avg_lookup = sum(self.lookup_times) / len(self.lookup_times) * 1000
            print(f"  Compatibility lookup: {avg_lookup:.2f}ms average")
            
            if self.validation_times:
                speedup = sum(self.validation_times) / max(1, sum(self.lookup_times))
                print(f"  Lookup speedup: {speedup:.2f}x faster than direct validation")
        
        if hasattr(self, 'compatibility_matrix_np') and self.compatibility_matrix_np is not None:
            memory_usage = self.compatibility_matrix_np.nbytes / (1024 * 1024)
            print(f"  Compatibility matrix memory usage: {memory_usage:.2f} MB")