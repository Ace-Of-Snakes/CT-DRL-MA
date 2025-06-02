import warp as wp
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import time

class WarpStorageYard:
    """
    GPU-accelerated storage yard for container terminal simulation using NVIDIA Warp.
    Efficiently manages container storage, stacking operations, and storage yard state.
    """
    
    def __init__(self, 
                 terminal_state,
                 container_registry,
                 stacking_kernels,
                 device: str = None):
        """
        Initialize the storage yard.
        
        Args:
            terminal_state: Reference to the WarpTerminalState object
            container_registry: Reference to the WarpContainerRegistry object
            stacking_kernels: Reference to the WarpStackingKernels object
            device: Computation device (if None, will use the terminal_state's device)
        """
        self.terminal_state = terminal_state
        self.container_registry = container_registry
        self.stacking_kernels = stacking_kernels
        self.device = device if device else terminal_state.device
        
        # Basic dimensions from terminal state
        self.num_rows = self.terminal_state.num_storage_rows
        self.num_bays = self.terminal_state.num_storage_bays
        self.max_height = self.terminal_state.max_stack_height
        self.row_names = self.terminal_state.row_names
        
        # Performance tracking
        self.add_container_times = []
        self.remove_container_times = []
        self.query_times = []
        
        # Register Warp kernels
        self._register_kernels()
        
        print(f"WarpStorageYard initialized on device: {self.device} with dimensions {self.num_rows}x{self.num_bays}")
    
    def _register_kernels(self):
        """
        Initialize any kernel-related setup. 
        
        Note: In Warp, @wp.kernel decorator is sufficient to define kernels,
        no explicit registration is needed.
        """
        # Kernels are already defined with @wp.kernel decorators,
        # no additional registration is needed
        pass

    # Utility kernels for array access
    @wp.kernel
    def _kernel_get_stack_height(stack_heights: wp.array(dtype=wp.int32, ndim=2),
                              row: wp.int32,
                              bay: wp.int32,
                              result: wp.array(dtype=wp.int32, ndim=1)):
        """Kernel to retrieve stack height at a specific position."""
        result[0] = stack_heights[row, bay]
    
    @wp.kernel
    def _kernel_get_container_at_position(yard_container_indices: wp.array(dtype=wp.int32, ndim=3),
                                       row: wp.int32,
                                       bay: wp.int32,
                                       tier: wp.int32,
                                       result: wp.array(dtype=wp.int32, ndim=1)):
        """Kernel to retrieve a container index at a specific position and tier."""
        result[0] = yard_container_indices[row, bay, tier]
    
    @wp.kernel
    def _kernel_set_container_at_position(yard_container_indices: wp.array(dtype=wp.int32, ndim=3),
                                       row: wp.int32,
                                       bay: wp.int32,
                                       tier: wp.int32,
                                       container_idx: wp.int32):
        """Kernel to set a container index at a specific position and tier."""
        yard_container_indices[row, bay, tier] = container_idx
    
    @wp.kernel
    def _kernel_update_stack_height(stack_heights: wp.array(dtype=wp.int32, ndim=2),
                                 row: wp.int32,
                                 bay: wp.int32,
                                 height: wp.int32):
        """Kernel to update stack height at a specific position."""
        stack_heights[row, bay] = height
    
    @wp.kernel
    def _kernel_update_container_position(container_positions: wp.array(dtype=wp.int32, ndim=1),
                                       container_idx: wp.int32,
                                       position_idx: wp.int32):
        """Kernel to update a container's position index."""
        container_positions[container_idx] = position_idx

    # Main analysis kernels
    @wp.kernel
    def _kernel_get_yard_occupancy(yard_container_indices: wp.array(dtype=wp.int32, ndim=3),
                                 stack_heights: wp.array(dtype=wp.int32, ndim=2),
                                 occupancy: wp.array(dtype=wp.float32, ndim=1),
                                 num_rows: wp.int32,
                                 num_bays: wp.int32):
        """
        Kernel to calculate yard occupancy.
        
        Args:
            yard_container_indices: Container indices in the yard [row, bay, tier]
            stack_heights: Current stack heights [row, bay]
            occupancy: Output occupancy metrics [3] (spots used, total spots, utilization %)
            num_rows: Number of rows in the yard
            num_bays: Number of bays per row
        """
        # Initialize results
        occupancy[0] = 0.0  # Spots used
        occupancy[1] = float(num_rows * num_bays)  # Total spots
        occupancy[2] = 0.0  # Utilization %
        
        # Count spots with at least one container
        spots_used = int(0)
        
        for row in range(num_rows):
            for bay in range(num_bays):
                if stack_heights[row, bay] > 0:
                    spots_used += 1
        
        # Use atomic add to update the first element
        wp.atomic_add(occupancy, 0, float(spots_used))
        
        # Calculate utilization percentage (only one thread should do this)
        if wp.tid() == 0:
            if occupancy[1] > 0:
                occupancy[2] = occupancy[0] / occupancy[1] * 100.0

    @wp.kernel
    def _kernel_get_container_distribution(yard_container_indices: wp.array(dtype=wp.int32, ndim=3),
                                        container_properties: wp.array(dtype=wp.float32, ndim=2),
                                        stack_heights: wp.array(dtype=wp.int32, ndim=2),
                                        type_counts: wp.array(dtype=wp.int32, ndim=1),
                                        num_rows: wp.int32,
                                        num_bays: wp.int32,
                                        max_height: wp.int32):
        """
        Kernel to count containers by type in the yard.
        
        Args:
            yard_container_indices: Container indices in the yard [row, bay, tier]
            container_properties: Container properties [container_idx, property]
            stack_heights: Current stack heights [row, bay]
            type_counts: Output counts by type [6] (TWEU, THEU, FEU, FFEU, Trailer, Swap Body)
            num_rows: Number of rows in the yard
            num_bays: Number of bays per row
            max_height: Maximum stack height
        """
        # Get thread indices - one thread per position
        row, bay = wp.tid()
        
        if row >= num_rows or bay >= num_bays:
            return
        
        # Get stack height at this position
        height = stack_heights[row, bay]
        
        # Count containers by type
        for tier in range(height):
            container_idx = yard_container_indices[row, bay, tier]
            if container_idx >= 0:
                # Get container type
                container_type = int(container_properties[container_idx, 0])
                
                # Increment count for this type (using atomic add to avoid race conditions)
                wp.atomic_add(type_counts, container_type, 1)

    @wp.kernel
    def _kernel_find_containers_of_type(yard_container_indices: wp.array(dtype=wp.int32, ndim=3),
                                     container_properties: wp.array(dtype=wp.float32, ndim=2), 
                                     stack_heights: wp.array(dtype=wp.int32, ndim=2),
                                     target_type: wp.int32,
                                     target_goods: wp.int32,
                                     location_mask: wp.array(dtype=wp.int32, ndim=2),
                                     count: wp.array(dtype=wp.int32, ndim=1),
                                     num_rows: wp.int32,
                                     num_bays: wp.int32):
        """
        Kernel to find locations of containers of a specific type.
        
        Args:
            yard_container_indices: Container indices in the yard [row, bay, tier]
            container_properties: Container properties [container_idx, property]
            stack_heights: Current stack heights [row, bay]
            target_type: Target container type code
            target_goods: Target goods type code (-1 for any)
            location_mask: Output mask of locations [row, bay] (1 if contains target container)
            count: Output count of matching containers [1]
            num_rows: Number of rows in the yard
            num_bays: Number of bays per row
        """
        # Get thread indices - one thread per position
        row, bay = wp.tid()
        
        if row >= num_rows or bay >= num_bays:
            return
        
        # Initialize mask for this position
        location_mask[row, bay] = 0
        
        # Get stack height at this position
        height = stack_heights[row, bay]
        
        # Check each container in the stack
        for tier in range(height):
            container_idx = yard_container_indices[row, bay, tier]
            if container_idx >= 0:
                # Get container type and goods type
                container_type = int(container_properties[container_idx, 0])
                container_goods = int(container_properties[container_idx, 1])
                
                # Check if this matches the target types
                if container_type == target_type:
                    if target_goods == -1 or container_goods == target_goods:
                        # Mark this location and increment count
                        location_mask[row, bay] = 1
                        wp.atomic_add(count, 0, 1)
                        break  # No need to check other containers in this stack
    
    @wp.kernel
    def _kernel_analyze_yard_utilization(stack_heights: wp.array(dtype=wp.int32, ndim=2), 
                                      max_height: wp.int32,
                                      utilization: wp.array(dtype=wp.float32, ndim=1), 
                                      num_rows: wp.int32,
                                      num_bays: wp.int32):
        """
        Kernel to analyze yard utilization and stacking efficiency.
        
        Args:
            stack_heights: Current stack heights [row, bay]
            max_height: Maximum stack height
            utilization: Output utilization metrics
                [0]: Ground utilization (% of positions used)
                [1]: Volume utilization (% of total possible container slots used)
                [2]: Average stack height
                [3]: Number of full-height stacks
            num_rows: Number of rows in the yard
            num_bays: Number of bays per row
        """
        # Reset utilization metrics
        if wp.tid() == 0:
            utilization[0] = 0.0  # Ground utilization
            utilization[1] = 0.0  # Volume utilization
            utilization[2] = 0.0  # Average stack height
            utilization[3] = 0.0  # Number of full-height stacks
        
        # Calculate metrics
        positions_used = 0
        containers_count = 0
        stack_height_sum = 0
        full_stacks = 0
        
        # Each thread computes for one position
        row, bay = wp.tid()
        
        if row < num_rows and bay < num_bays:
            height = stack_heights[row, bay]
            
            if height > 0:
                positions_used += 1
                containers_count += height
                stack_height_sum += height
                
                if height >= max_height:
                    full_stacks += 1
        
        # Accumulate results using atomic operations
        wp.atomic_add(utilization, 0, float(positions_used))
        wp.atomic_add(utilization, 1, float(containers_count))
        wp.atomic_add(utilization, 2, float(stack_height_sum))
        wp.atomic_add(utilization, 3, float(full_stacks))
        
        # Finalize calculations (only one thread should do this)
        if wp.tid() == 0:
            total_positions = float(num_rows * num_bays)
            total_capacity = total_positions * float(max_height)
            
            # Calculate percentages
            if total_positions > 0:
                utilization[0] = utilization[0] / total_positions * 100.0  # Ground utilization
            
            if total_capacity > 0:
                utilization[1] = utilization[1] / total_capacity * 100.0  # Volume utilization
            
            if utilization[0] > 0:
                utilization[2] = utilization[2] / utilization[0]  # Average stack height (only counting used positions)
    
    def add_container(self, container_id_or_idx, position_str: str) -> bool:
        """
        Add a container to a position in the storage yard.
        
        Args:
            container_id_or_idx: Container ID or index
            position_str: Position string (e.g., 'A1')
            
        Returns:
            True if container was placed successfully, False otherwise
        """
        start_time = time.time()
        
        # Convert container ID to index if needed
        container_idx = container_id_or_idx
        if isinstance(container_id_or_idx, str):
            container_idx = self.container_registry._get_container_idx(container_id_or_idx)
            
        if container_idx < 0:
            return False
        
        # Check if position is valid
        if not self._is_valid_position(position_str):
            return False
        
        # Parse position string
        row, bay = self._parse_position(position_str)
        if row is None or bay is None:
            return False
        
        # Check if container can be placed at this position
        if not self.stacking_kernels.can_place_at(container_idx, position_str):
            return False
        
        # Get current stack height using kernel
        height_result = wp.zeros(1, dtype=wp.int32, device=self.device)
        wp.launch(
            kernel=self._kernel_get_stack_height,
            dim=1,
            inputs=[self.terminal_state.stack_heights, row, bay, height_result]
        )
        height_result_np = height_result.numpy()  # Convert to NumPy array
        height = int(height_result_np[0])         # Now safely index
        
        # Check if stack is full
        if height >= self.max_height:
            return False
        
        # Place container in yard using kernel
        wp.launch(
            kernel=self._kernel_set_container_at_position,
            dim=1,
            inputs=[self.terminal_state.yard_container_indices, row, bay, height, container_idx]
        )
        
        # Update stack height using kernel
        wp.launch(
            kernel=self._kernel_update_stack_height,
            dim=1,
            inputs=[self.terminal_state.stack_heights, row, bay, height + 1]
        )
        
        # Update container position
        position_idx = self.terminal_state.position_to_idx.get(position_str, -1)
        if position_idx >= 0:
            wp.launch(
                kernel=self._kernel_update_container_position,
                dim=1,
                inputs=[self.terminal_state.container_positions, container_idx, position_idx]
            )
        
        # Track performance
        self.add_container_times.append(time.time() - start_time)
        
        return True
    
    @wp.kernel
    def _kernel_get_height_result(height_result: wp.array(dtype=wp.int32, ndim=1),
                            output: wp.array(dtype=wp.int32, ndim=1)):
        """Get height result and store in output array."""
        output[0] = height_result[0]

    def remove_container(self, position_str: str) -> int:
        """
        Remove a container from a position in the storage yard.
        
        Args:
            position_str: Position string (e.g., 'A1')
            
        Returns:
            Container index that was removed or -1 if no container was removed
        """
        start_time = time.time()
        
        # Check if position is valid
        if not self._is_valid_position(position_str):
            return -1
        
        # Parse position string
        row, bay = self._parse_position(position_str)
        if row is None or bay is None:
            return -1
        
        # Get current stack height using kernel
        height_result = wp.zeros(1, dtype=wp.int32, device=self.device)
        wp.launch(
            kernel=self._kernel_get_stack_height,
            dim=1,
            inputs=[self.terminal_state.stack_heights, row, bay, height_result]
        )
        
        # Extract height using a helper kernel
        height_output = wp.zeros(1, dtype=wp.int32, device=self.device) 
        wp.launch(
            kernel=self._kernel_get_height_result,
            dim=1,
            inputs=[height_result, height_output]
        )
        height = int(height_output.numpy()[0])
        
        # Check if stack is empty
        if height <= 0:
            return -1
        
        # Get container at the top of the stack using kernel
        container_result = wp.zeros(1, dtype=wp.int32, device=self.device)
        wp.launch(
            kernel=self._kernel_get_container_at_position,
            dim=1,
            inputs=[self.terminal_state.yard_container_indices, row, bay, height-1, container_result]
        )
        
        # Extract container using same helper kernel
        container_output = wp.zeros(1, dtype=wp.int32, device=self.device)
        wp.launch(
            kernel=self._kernel_get_height_result,  # Reuse this kernel (it works for any int array)
            dim=1,
            inputs=[container_result, container_output]
        )
        container_idx = int(container_output.numpy()[0])
        
        # Check if we got a valid container
        if container_idx < 0:
            return -1
        
        # Remove container from yard (set position to -1)
        wp.launch(
            kernel=self._kernel_set_container_at_position,
            dim=1,
            inputs=[self.terminal_state.yard_container_indices, row, bay, height-1, -1]
        )
        
        # Update stack height
        wp.launch(
            kernel=self._kernel_update_stack_height,
            dim=1,
            inputs=[self.terminal_state.stack_heights, row, bay, height - 1]
        )
        
        # Update container position
        wp.launch(
            kernel=self._kernel_update_container_position,
            dim=1,
            inputs=[self.terminal_state.container_positions, container_idx, -1]
        )
        
        # Track performance
        self.remove_container_times.append(time.time() - start_time)
        
        return container_idx
    
    def get_top_container(self, position_str: str) -> Tuple[Optional[int], Optional[int]]:
        """
        Get the top container at a position (without removing it).
        
        Args:
            position_str: Position string (e.g., 'A1')
            
        Returns:
            Tuple of (container_idx, tier) or (None, None) if no container
        """
        start_time = time.time()
        
        # Check if position is valid
        if not self._is_valid_position(position_str):
            return None, None
        
        # Parse position string
        row, bay = self._parse_position(position_str)
        if row is None or bay is None:
            return None, None
        
        # Get current stack height using kernel
        height_result = wp.zeros(1, dtype=wp.int32, device=self.device)
        wp.launch(
            kernel=self._kernel_get_stack_height,
            dim=1,
            inputs=[self.terminal_state.stack_heights, row, bay, height_result]
        )
        height = int(height_result.numpy()[0])
        
        # Check if stack is empty
        if height <= 0:
            return None, None
        
        # Get container at the top of the stack using kernel
        container_result = wp.zeros(1, dtype=wp.int32, device=self.device)
        wp.launch(
            kernel=self._kernel_get_container_at_position,
            dim=1,
            inputs=[self.terminal_state.yard_container_indices, row, bay, height-1, container_result]
        )
        container_idx = int(container_result.numpy()[0])
        
        # Check if we got a valid container
        if container_idx < 0:
            return None, None
        
        # Track performance
        self.query_times.append(time.time() - start_time)
        
        return container_idx, height - 1
    
    def get_containers_at_position(self, position_str: str) -> Dict[int, int]:
        """
        Get all containers at a position.
        
        Args:
            position_str: Position string (e.g., 'A1')
            
        Returns:
            Dictionary mapping tier to container index
        """
        start_time = time.time()
        
        result = {}
        
        # Check if position is valid
        if not self._is_valid_position(position_str):
            return result
        
        # Parse position string
        row, bay = self._parse_position(position_str)
        if row is None or bay is None:
            return result
        
        # Get current stack height using kernel
        height_result = wp.zeros(1, dtype=wp.int32, device=self.device)
        wp.launch(
            kernel=self._kernel_get_stack_height,
            dim=1,
            inputs=[self.terminal_state.stack_heights, row, bay, height_result]
        )
        height = int(height_result.numpy()[0])
        
        # Check if stack is empty
        if height <= 0:
            return result
        
        # Get containers in the stack, tier by tier
        container_result = wp.zeros(1, dtype=wp.int32, device=self.device)
        for tier in range(height):
            wp.launch(
                kernel=self._kernel_get_container_at_position,
                dim=1,
                inputs=[self.terminal_state.yard_container_indices, row, bay, tier, container_result]
            )
            container_idx = int(container_result.numpy()[0])
            
            if container_idx >= 0:
                result[tier] = container_idx
        
        # Track performance
        self.query_times.append(time.time() - start_time)
        
        return result
    
    def can_accept_container(self, position_str: str, container_id_or_idx) -> bool:
        """
        Check if a container can be placed at a position.
        
        Args:
            position_str: Position string (e.g., 'A1')
            container_id_or_idx: Container ID or index
            
        Returns:
            True if container can be placed, False otherwise
        """
        # Convert container ID to index if needed
        container_idx = container_id_or_idx
        if isinstance(container_id_or_idx, str):
            container_idx = self.container_registry._get_container_idx(container_id_or_idx)
            
        if container_idx < 0:
            return False
        
        return self.stacking_kernels.can_place_at(container_idx, position_str)
    
    def get_yard_state(self) -> np.ndarray:
        """
        Get a numpy array representation of the yard state.
        
        Returns:
            Numpy array with shape [num_rows, num_bays, features]
        """
        # Create a multi-feature representation of the yard state
        # [0]: Stack height
        # [1]: Has reefer container
        # [2]: Has dangerous container
        # [3]: Has trailer
        # [4]: Has swap body
        # [5]: Average priority of containers in stack
        # [6]: Container at top is close to departure
        
        # Initialize state array
        state = np.zeros((self.num_rows, self.num_bays, 7), dtype=np.float32)
        
        # Get stack heights using numpy conversion (this is safe as it's a read-only operation)
        heights = self.terminal_state.stack_heights.numpy()
        state[:,:,0] = heights / self.max_height  # Normalize by max height
        
        # Get container indices and properties using kernel computation
        container_result = wp.zeros(1, dtype=wp.int32, device=self.device)
        
        # Go through each position
        for row in range(self.num_rows):
            for bay in range(self.num_bays):
                height = heights[row, bay]
                if height > 0:
                    # Find container types in stack
                    has_reefer = False
                    has_dangerous = False
                    has_trailer = False
                    has_swap_body = False
                    priority_sum = 0
                    
                    for tier in range(height):
                        # Get container at this position and tier
                        wp.launch(
                            kernel=self._kernel_get_container_at_position,
                            dim=1,
                            inputs=[self.terminal_state.yard_container_indices, row, bay, tier, container_result]
                        )
                        container_idx = int(container_result.numpy()[0])
                        
                        if container_idx >= 0:
                            # Get container properties using numpy for this read operation
                            props = self.terminal_state.container_properties[container_idx].numpy()
                            container_type = int(props[0])
                            goods_type = int(props[1])
                            priority = props[2]
                            departure_time = props[7]
                            
                            # Check container type
                            if container_type == 4:  # Trailer
                                has_trailer = True
                            elif container_type == 5:  # Swap Body
                                has_swap_body = True
                                
                            # Check goods type
                            if goods_type == 1:  # Reefer
                                has_reefer = True
                            elif goods_type == 2:  # Dangerous
                                has_dangerous = True
                                
                            # Add to priority sum
                            priority_sum += priority
                            
                            # Check departure time for top container
                            if tier == height - 1:
                                if departure_time > 0:
                                    current_time = self.terminal_state.simulation_time.numpy()[0]
                                    time_until_departure = departure_time - current_time
                                    if time_until_departure > 0:
                                        # Normalize: 1.0 = departure within 1 hour, 0.5 = 1 day, 0.0 = >1 week
                                        if time_until_departure < 3600:  # 1 hour
                                            state[row, bay, 6] = 1.0
                                        elif time_until_departure < 86400:  # 1 day
                                            state[row, bay, 6] = 0.75
                                        elif time_until_departure < 172800:  # 2 days
                                            state[row, bay, 6] = 0.5
                                        elif time_until_departure < 604800:  # 1 week
                                            state[row, bay, 6] = 0.25
                    
                    # Set feature flags
                    state[row, bay, 1] = 1.0 if has_reefer else 0.0
                    state[row, bay, 2] = 1.0 if has_dangerous else 0.0
                    state[row, bay, 3] = 1.0 if has_trailer else 0.0
                    state[row, bay, 4] = 1.0 if has_swap_body else 0.0
                    
                    # Average priority
                    if height > 0:
                        state[row, bay, 5] = priority_sum / height / 100.0  # Normalize by dividing by 100
        
        return state
    
    def get_yard_occupancy(self) -> Dict[str, float]:
        """
        Get storage yard occupancy metrics.
        
        Returns:
            Dictionary with occupancy metrics
        """
        # Create result array on device
        occupancy = wp.zeros(3, dtype=wp.float32, device=self.device)
        
        # Run the occupancy kernel
        wp.launch(
            kernel=self._kernel_get_yard_occupancy,
            dim=1,
            inputs=[
                self.terminal_state.yard_container_indices,
                self.terminal_state.stack_heights,
                occupancy,
                self.num_rows,
                self.num_bays
            ]
        )
        
        # Return results as a dictionary
        occupancy_np = occupancy.numpy()
        return {
            "spots_used": int(occupancy_np[0]),
            "total_spots": int(occupancy_np[1]),
            "utilization_percent": float(occupancy_np[2])
        }
    
    def get_container_distribution(self) -> Dict[str, int]:
        """
        Get distribution of containers by type in the yard.
        
        Returns:
            Dictionary mapping container types to counts
        """
        # Create type counts array on device
        type_counts = wp.zeros(6, dtype=wp.int32, device=self.device)
        
        # Run the distribution kernel
        wp.launch(
            kernel=self._kernel_get_container_distribution,
            dim=[self.num_rows, self.num_bays],
            inputs=[
                self.terminal_state.yard_container_indices,
                self.terminal_state.container_properties,
                self.terminal_state.stack_heights,
                type_counts,
                self.num_rows,
                self.num_bays,
                self.max_height
            ]
        )
        
        # Return results as a dictionary
        type_counts_np = type_counts.numpy()
        
        # Map types to names
        type_names = ["TWEU", "THEU", "FEU", "FFEU", "Trailer", "Swap Body"]
        return {name: int(type_counts_np[i]) for i, name in enumerate(type_names)}
    
    def find_containers_of_type(self, container_type: str, goods_type: str = None) -> List[str]:
        """
        Find positions where containers of a specific type are stored.
        
        Args:
            container_type: Container type ("TWEU", "FEU", etc.)
            goods_type: Goods type ("Regular", "Reefer", "Dangerous") or None for any
            
        Returns:
            List of position strings where matching containers are found
        """
        # Map container type to code
        type_codes = {
            "TWEU": 0, "THEU": 1, "FEU": 2, "FFEU": 3, "Trailer": 4, "Swap Body": 5
        }
        
        if container_type not in type_codes:
            return []
            
        type_code = type_codes[container_type]
        
        # Map goods type to code
        goods_codes = {
            "Regular": 0, "Reefer": 1, "Dangerous": 2
        }
        
        goods_code = -1  # -1 means any goods type
        if goods_type is not None:
            if goods_type not in goods_codes:
                return []
            goods_code = goods_codes[goods_type]
        
        # Create result arrays on device
        location_mask = wp.zeros((self.num_rows, self.num_bays), dtype=wp.int32, device=self.device)
        count = wp.zeros(1, dtype=wp.int32, device=self.device)
        
        # Run the find containers kernel
        wp.launch(
            kernel=self._kernel_find_containers_of_type,
            dim=[self.num_rows, self.num_bays],
            inputs=[
                self.terminal_state.yard_container_indices,
                self.terminal_state.container_properties,
                self.terminal_state.stack_heights,
                type_code,
                goods_code,
                location_mask,
                count,
                self.num_rows,
                self.num_bays
            ]
        )
        
        # Get results
        location_mask_np = location_mask.numpy()
        
        # Convert to position strings
        result = []
        for row in range(self.num_rows):
            for bay in range(self.num_bays):
                if location_mask_np[row, bay]:
                    position_str = f"{self.row_names[row]}{bay+1}"
                    result.append(position_str)
        
        return result
    
    def analyze_yard_utilization(self) -> Dict[str, float]:
        """
        Analyze yard utilization and stacking efficiency.
        
        Returns:
            Dictionary with utilization metrics
        """
        # Create result array on device
        utilization = wp.zeros(4, dtype=wp.float32, device=self.device)
        
        # Run the utilization kernel
        wp.launch(
            kernel=self._kernel_analyze_yard_utilization,
            dim=[self.num_rows, self.num_bays],
            inputs=[
                self.terminal_state.stack_heights,
                self.max_height,
                utilization,
                self.num_rows,
                self.num_bays
            ]
        )
        
        # Return results as a dictionary
        utilization_np = utilization.numpy()
        return {
            "ground_utilization_percent": float(utilization_np[0]),
            "volume_utilization_percent": float(utilization_np[1]),
            "average_stack_height": float(utilization_np[2]),
            "full_height_stacks": int(utilization_np[3])
        }
    
    def _is_valid_position(self, position_str: str) -> bool:
        """Check if a position string is a valid storage position."""
        # Valid storage positions are in format 'A1', 'B2', etc.
        if len(position_str) < 2 or not position_str[0].isalpha():
            return False
            
        # Check if the row letter is valid
        row_letter = position_str[0].upper()
        if row_letter not in self.row_names:
            return False
            
        # Check if the bay number is valid
        try:
            bay_num = int(position_str[1:])
            return 1 <= bay_num <= self.num_bays
        except ValueError:
            return False
    
    def _parse_position(self, position_str: str) -> Tuple[Optional[int], Optional[int]]:
        """Parse a position string into row and bay indices."""
        if not self._is_valid_position(position_str):
            return None, None
            
        row_letter = position_str[0].upper()
        bay_num = int(position_str[1:])
        
        row_idx = self.row_names.index(row_letter)
        bay_idx = bay_num - 1  # Convert to 0-based index
        
        return row_idx, bay_idx
    
    def find_optimal_location(self, container_id_or_idx) -> Optional[str]:
        """
        Find the optimal location in the storage yard for a container.
        
        Args:
            container_id_or_idx: Container ID or index
            
        Returns:
            Optimal position string or None if no suitable location found
        """
        # Convert container ID to index if needed
        container_idx = container_id_or_idx
        if isinstance(container_id_or_idx, str):
            container_idx = self.container_registry._get_container_idx(container_id_or_idx)
            
        if container_idx < 0:
            return None
        
        # Use the stacking kernels to find optimal location
        row, bay, score = self.stacking_kernels.find_optimal_location(container_idx)
        
        if row < 0 or bay < 0 or score <= 0:
            return None
            
        # Convert to position string
        position_str = f"{self.row_names[row]}{bay+1}"
        return position_str
    
    def get_problem_stacks(self) -> List[Tuple[str, float]]:
        """
        Get a list of positions with problematic stacks that should be optimized.
        
        Returns:
            List of (position_str, problem_score) tuples, sorted by score (highest first)
        """
        # Use the stacking kernels to identify problem stacks
        problem_stacks = self.stacking_kernels.identify_problem_stacks()
        
        # Convert to position strings with scores
        result = []
        for row, bay, score in problem_stacks:
            position_str = f"{self.row_names[row]}{bay+1}"
            result.append((position_str, score))
            
        return result
    
    def get_premarshalling_plan(self) -> List[Tuple[str, str]]:
        """
        Get a plan for pre-marshalling (reshuffling) the yard to optimize stacking.
        
        Returns:
            List of (source_position, destination_position) tuples representing moves
        """
        # Use the stacking kernels to generate a pre-marshalling plan
        moves = self.stacking_kernels.generate_premarshalling_plan()
        
        # Convert to position strings
        result = []
        for src_row, src_bay, dst_row, dst_bay in moves:
            src_pos = f"{self.row_names[src_row]}{src_bay+1}"
            dst_pos = f"{self.row_names[dst_row]}{dst_bay+1}"
            result.append((src_pos, dst_pos))
            
        return result
    
    def print_performance_stats(self):
        """Print performance statistics for the storage yard."""
        if not self.add_container_times and not self.remove_container_times and not self.query_times:
            print("No performance data available.")
            return
        
        print("\nStorage Yard Performance Statistics:")
        
        if self.add_container_times:
            avg_add = sum(self.add_container_times) / len(self.add_container_times) * 1000
            print(f"  Add container: {avg_add:.2f}ms average")
        
        if self.remove_container_times:
            avg_remove = sum(self.remove_container_times) / len(self.remove_container_times) * 1000
            print(f"  Remove container: {avg_remove:.2f}ms average")
        
        if self.query_times:
            avg_query = sum(self.query_times) / len(self.query_times) * 1000
            print(f"  Query operations: {avg_query:.2f}ms average")
            
        # Print utilization statistics
        try:
            utilization = self.analyze_yard_utilization()
            print("\nCurrent Yard Utilization:")
            print(f"  Ground utilization: {utilization['ground_utilization_percent']:.1f}%")
            print(f"  Volume utilization: {utilization['volume_utilization_percent']:.1f}%")
            print(f"  Average stack height: {utilization['average_stack_height']:.2f}")
            print(f"  Full height stacks: {utilization['full_height_stacks']}")
        except Exception as e:
            pass