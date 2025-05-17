import warp as wp
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Union, Any

class WarpStackingCompatibilityMatrix:
    """
    Precomputes all container stacking compatibility checks.
    
    Trades memory for performance by:
    1. Computing container-to-container stacking compatibility once
    2. Storing results in a binary matrix for O(1) lookup
    3. Eliminating the need for property checks during simulation
    
    Memory usage: O(max_containers²) for full compatibility matrix
    """
    
    def __init__(self, 
                 terminal_state,
                 container_registry,
                 device: str = None,
                 use_bit_packing: bool = False):
        """
        Initialize the stacking compatibility matrix.
        
        Args:
            terminal_state: Reference to the WarpTerminalState object
            container_registry: Reference to the WarpContainerRegistry object
            device: Computation device (if None, will use terminal_state's device)
            use_bit_packing: Use bit packing to reduce memory usage (advanced)
        """
        self.terminal_state = terminal_state
        self.container_registry = container_registry
        self.device = device if device else terminal_state.device
        self.max_containers = terminal_state.max_containers
        self.use_bit_packing = use_bit_packing
        
        # Initialize compatibility matrix
        if use_bit_packing:
            # Use int32 for bit packing (32 containers per entry)
            packed_width = (self.max_containers + 31) // 32
            self.compatibility_matrix = wp.zeros(
                (self.max_containers, packed_width),
                dtype=wp.int32,
                device=self.device
            )
        else:
            # Use standard matrix (one entry per container pair)
            self.compatibility_matrix = wp.zeros(
                (self.max_containers, self.max_containers),
                dtype=wp.int32,
                device=self.device
            )
        
        # Track performance
        self.lookup_times = []
        
        print(f"WarpStackingCompatibilityMatrix initialized on device: {self.device}")
        print(f"Max containers: {self.max_containers}")
        print(f"Memory estimate: {self._estimate_memory_usage():.2f} MB")
    
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in megabytes."""
        if self.use_bit_packing:
            packed_width = (self.max_containers + 31) // 32
            num_entries = self.max_containers * packed_width
        else:
            num_entries = self.max_containers * self.max_containers
            
        # Each entry is an int32 (4 bytes)
        return num_entries * 4 / (1024 * 1024)  # Convert bytes to MB
    
    def precompute_compatibility(self) -> None:
        """Precompute compatibility for all containers."""
        start_time = time.time()
        
        if self.use_bit_packing:
            # Use bit-packed kernel
            wp.launch(
                kernel=self._kernel_compute_compatibility_matrix_packed,
                dim=[self.max_containers, self.max_containers],
                inputs=[
                    self.terminal_state.container_properties,
                    self.terminal_state.container_dimensions,
                    self.compatibility_matrix
                ]
            )
        else:
            # Use standard kernel
            wp.launch(
                kernel=self._kernel_compute_compatibility_matrix,
                dim=[self.max_containers, self.max_containers],
                inputs=[
                    self.terminal_state.container_properties,
                    self.terminal_state.container_dimensions,
                    self.compatibility_matrix
                ]
            )
        
        end_time = time.time()
        print(f"Compatibility matrix computation completed in {end_time - start_time:.2f} seconds")
    
    @wp.kernel
    def _kernel_compute_compatibility_matrix(
        container_properties: wp.array(dtype=wp.float32, ndim=2),
        container_dimensions: wp.array(dtype=wp.float32, ndim=2),
        compatibility_matrix: wp.array(dtype=wp.int32, ndim=2)) -> None:
        """
        Kernel to compute all container stacking compatibility in parallel.
        
        Args:
            container_properties: Container properties array [container_idx, property]
            container_dimensions: Container dimensions array [container_idx, dimension]
            compatibility_matrix: Output matrix for compatibility results
        """
        # Get container indices
        i, j = wp.tid()
        
        # Only compute for active containers and valid stacking (j on top of i)
        if (container_properties[i, 6] <= 0 or  # i inactive
            container_properties[j, 6] <= 0 or  # j inactive
            i == j):  # Same container
            return
        
        # Default: incompatible
        compatibility_matrix[j, i] = 0
        
        # Get container properties
        i_type = int(container_properties[i, 0])
        i_goods = int(container_properties[i, 1])
        i_weight = container_properties[i, 3]
        i_stackable = bool(container_properties[i, 4])
        i_compatibility = int(container_properties[i, 5])
        
        j_type = int(container_properties[j, 0])
        j_goods = int(container_properties[j, 1])
        j_weight = container_properties[j, 3]
        j_compatibility = int(container_properties[j, 5])
        
        # Can't stack on non-stackable container
        if not i_stackable:
            return
        
        # Check compatibility
        can_stack = True
        
        # None compatibility - can't stack
        if i_compatibility == 0 or j_compatibility == 0:
            can_stack = False
        
        # Self compatibility - must be same type and goods
        elif i_compatibility == 1 or j_compatibility == 1:
            if i_type != j_type or i_goods != j_goods:
                can_stack = False
        
        # Size compatibility - must be same size
        elif i_compatibility == 2 or j_compatibility == 2:
            if i_type != j_type:
                can_stack = False
        
        # Weight constraint - container above should be lighter
        if j_weight > i_weight:
            can_stack = False
        
        # Update compatibility matrix
        if can_stack:
            compatibility_matrix[j, i] = 1
    
    @wp.kernel
    def _kernel_compute_compatibility_matrix_packed(
        container_properties: wp.array(dtype=wp.float32, ndim=2),
        container_dimensions: wp.array(dtype=wp.float32, ndim=2),
        compatibility_matrix: wp.array(dtype=wp.int32, ndim=2)) -> None:
        """
        Kernel to compute compatibility with bit packing for memory efficiency.
        
        Args:
            container_properties: Container properties array
            container_dimensions: Container dimensions array
            compatibility_matrix: Output packed matrix for compatibility results
        """
        # Get container indices
        i, j = wp.tid()
        
        # Only compute for active containers and valid stacking (j on top of i)
        if (container_properties[i, 6] <= 0 or  # i inactive
            container_properties[j, 6] <= 0 or  # j inactive
            i == j):  # Same container
            return
        
        # Calculate bit position
        bit_index = i % 32
        word_index = i // 32
        
        # Get container properties
        i_type = int(container_properties[i, 0])
        i_goods = int(container_properties[i, 1])
        i_weight = container_properties[i, 3]
        i_stackable = bool(container_properties[i, 4])
        i_compatibility = int(container_properties[i, 5])
        
        j_type = int(container_properties[j, 0])
        j_goods = int(container_properties[j, 1])
        j_weight = container_properties[j, 3]
        j_compatibility = int(container_properties[j, 5])
        
        # Can't stack on non-stackable container
        if not i_stackable:
            return
        
        # Check compatibility
        can_stack = True
        
        # None compatibility - can't stack
        if i_compatibility == 0 or j_compatibility == 0:
            can_stack = False
        
        # Self compatibility - must be same type and goods
        elif i_compatibility == 1 or j_compatibility == 1:
            if i_type != j_type or i_goods != j_goods:
                can_stack = False
        
        # Size compatibility - must be same size
        elif i_compatibility == 2 or j_compatibility == 2:
            if i_type != j_type:
                can_stack = False
        
        # Weight constraint - container above should be lighter
        if j_weight > i_weight:
            can_stack = False
        
        # Update compatibility matrix using atomic operations
        if can_stack:
            bit_mask = 1 << bit_index
            wp.atomic_or(compatibility_matrix, (j, word_index), bit_mask)
    
    @wp.kernel
    def _kernel_get_compatibility(
        compatibility_matrix: wp.array(dtype=wp.int32, ndim=2),
        upper_container_idx: wp.int32,
        lower_container_idx: wp.int32,
        use_bit_packing: wp.int32,
        result: wp.array(dtype=wp.int32, ndim=1)) -> None:
        """
        Kernel to get compatibility between two containers.
        
        Args:
            compatibility_matrix: Compatibility matrix
            upper_container_idx: Index of upper container
            lower_container_idx: Index of lower container
            use_bit_packing: Whether bit packing is used
            result: Output for compatibility result
        """
        if use_bit_packing:
            # Get from bit-packed matrix
            bit_index = lower_container_idx % 32
            word_index = lower_container_idx // 32
            word = compatibility_matrix[upper_container_idx, word_index]
            bit_mask = 1 << bit_index
            result[0] = 1 if (word & bit_mask) != 0 else 0
        else:
            # Get from standard matrix
            result[0] = compatibility_matrix[upper_container_idx, lower_container_idx]
    
    def can_stack(self, 
                upper_container_idx: int, 
                lower_container_idx: int) -> bool:
        """
        Check if upper container can be stacked on lower container.
        
        Args:
            upper_container_idx: Index of container to be placed on top
            lower_container_idx: Index of container at the bottom
            
        Returns:
            True if stacking is valid, False otherwise
        """
        start_time = time.time()
        
        # Validate indices
        if upper_container_idx < 0 or upper_container_idx >= self.max_containers:
            return False
            
        if lower_container_idx < 0 or lower_container_idx >= self.max_containers:
            return False
            
        # Create result array
        result = wp.zeros(1, dtype=wp.int32, device=self.device)
        
        # Launch kernel to get compatibility
        wp.launch(
            kernel=self._kernel_get_compatibility,
            dim=1,
            inputs=[
                self.compatibility_matrix,
                upper_container_idx,
                lower_container_idx,
                1 if self.use_bit_packing else 0,
                result
            ]
        )
        
        # Get result
        can_stack = bool(result.numpy()[0])
        
        # Track lookup time
        self.lookup_times.ap