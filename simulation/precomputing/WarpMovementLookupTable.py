import warp as wp
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Union, Any

class WarpMovementLookupTable:
    """
    Replaces runtime physics calculations with precomputed lookup tables.
    
    Trades memory for performance by:
    1. Precomputing all possible crane movement times between terminal positions
    2. Storing results in 3D lookup tables (source, destination, stack height)
    3. Allowing O(1) lookups instead of runtime calculations
    
    Memory usage: O(positions² × container_types × stack_heights)
    """
    
    def __init__(self, 
                 terminal_state,
                 movement_calculator,
                 container_types: List[str] = None,
                 max_stack_heights: int = 6, 
                 device: str = None):
        """
        Initialize the movement lookup table.
        
        Args:
            terminal_state: Reference to the WarpTerminalState object
            movement_calculator: Reference to the movement calculator
            container_types: List of container types to precompute
            max_stack_heights: Number of stack height levels to precompute
            device: Computation device (if None, will use terminal_state's device)
        """
        self.terminal_state = terminal_state
        self.movement_calculator = movement_calculator
        self.device = device if device else terminal_state.device
        
        # Position information
        self.position_to_idx = terminal_state.position_to_idx
        self.idx_to_position = terminal_state.idx_to_position
        self.num_positions = len(self.position_to_idx)
        
        # Container types to precompute
        if container_types is None:
            self.container_types = ["TEU", "FEU", "HQ", "Trailer", "Swap Body"]
        else:
            self.container_types = container_types
            
        # Map container type names to indices
        self.container_type_to_idx = {t: i for i, t in enumerate(self.container_types)}
        
        # Number of stack heights to precompute
        self.max_stack_heights = max_stack_heights
        
        # Initialize lookup tables
        self.movement_times = None
        
        # Time tracking for performance metrics
        self.lookup_times = []
        
        print(f"WarpMovementLookupTable initialized on device: {self.device}")
        print(f"Positions: {self.num_positions}, Container types: {len(self.container_types)}")
        print(f"Memory estimate: {self._estimate_memory_usage():.2f} MB")
    
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in megabytes."""
        # Each value is a float32 (4 bytes)
        num_entries = self.num_positions * self.num_positions * len(self.container_types) * self.max_stack_heights
        return num_entries * 4 / (1024 * 1024)  # Convert bytes to MB
    
    def precompute_movement_times(self) -> None:
        """Precompute all possible movement times and store in lookup table."""
        start_time = time.time()
        
        # Create lookup table [positions, positions, container_types, stack_heights]
        self.movement_times = wp.zeros(
            (self.num_positions, self.num_positions, len(self.container_types), self.max_stack_heights),
            dtype=wp.float32,
            device=self.device
        )
        
        # Process positions in batches to avoid memory issues
        batch_size = 50  # Process 50 source positions at a time
        num_batches = (self.num_positions + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, self.num_positions)
            print(f"Processing batch {batch_idx+1}/{num_batches} (positions {start_idx}-{end_idx})...")
            
            self._process_position_batch(start_idx, end_idx)
        
        end_time = time.time()
        print(f"Movement lookup table precomputation completed in {end_time - start_time:.2f} seconds")
    
    def _process_position_batch(self, start_src_idx: int, end_src_idx: int) -> None:
        """Process a batch of source positions to populate the lookup table."""
        # List source position names for this batch
        position_indices = list(self.idx_to_position.keys())
        
        # Process each source position
        for src_idx in range(start_src_idx, end_src_idx):
            if src_idx >= self.num_positions:
                break
                
            src_pos = self.idx_to_position.get(src_idx)
            if src_pos is None:
                continue
                
            # Get source position type
            src_type = self._get_position_type(src_pos)
            
            # For each destination position
            for dst_idx in range(self.num_positions):
                dst_pos = self.idx_to_position.get(dst_idx)
                if dst_pos is None or src_pos == dst_pos:
                    continue
                
                # Get destination position type
                dst_type = self._get_position_type(dst_pos)
                
                # For each container type
                for type_idx, container_type in enumerate(self.container_types):
                    # For each stack height
                    for stack_idx in range(self.max_stack_heights):
                        # Calculate stack height in meters
                        stack_height = stack_idx * 2.59  # Approximate height per container
                        
                        # Calculate movement time
                        try:
                            time_value = self.movement_calculator.calculate_movement_time(
                                src_pos, dst_pos, container_type, stack_height
                            )
                            
                            # Store in lookup table using kernel
                            wp.launch(
                                kernel=self._kernel_set_movement_time,
                                dim=1,
                                inputs=[
                                    self.movement_times,
                                    src_idx,
                                    dst_idx,
                                    type_idx,
                                    stack_idx,
                                    float(time_value)
                                ]
                            )
                        except Exception as e:
                            # Handle calculation errors gracefully
                            wp.launch(
                                kernel=self._kernel_set_movement_time,
                                dim=1,
                                inputs=[
                                    self.movement_times,
                                    src_idx,
                                    dst_idx,
                                    type_idx,
                                    stack_idx,
                                    float(100.0)  # Default fallback time
                                ]
                            )
    
    @wp.kernel
    def _kernel_set_movement_time(movement_times: wp.array(dtype=wp.float32, ndim=4),
                               src_idx: wp.int32,
                               dst_idx: wp.int32,
                               type_idx: wp.int32,
                               stack_idx: wp.int32,
                               time_value: wp.float32) -> None:
        """Kernel to set a movement time in the lookup table."""
        movement_times[src_idx, dst_idx, type_idx, stack_idx] = time_value
    
    @wp.kernel
    def _kernel_get_movement_time(movement_times: wp.array(dtype=wp.float32, ndim=4),
                               src_idx: wp.int32,
                               dst_idx: wp.int32,
                               type_idx: wp.int32,
                               stack_idx: wp.int32,
                               result: wp.array(dtype=wp.float32, ndim=1)) -> None:
        """Kernel to get a movement time from the lookup table."""
        result[0] = movement_times[src_idx, dst_idx, type_idx, stack_idx]
    
    def get_movement_time(self, 
                        src_pos: str, 
                        dst_pos: str, 
                        container_type: str = "FEU", 
                        stack_height: float = 0.0) -> float:
        """
        Get movement time from lookup table.
        
        Args:
            src_pos: Source position string
            dst_pos: Destination position string
            container_type: Container type
            stack_height: Height of container stack
            
        Returns:
            Movement time in seconds or None if positions not found
        """
        start_time = time.time()
        
        # Convert positions to indices
        src_idx = self.position_to_idx.get(src_pos, -1)
        dst_idx = self.position_to_idx.get(dst_pos, -1)
        
        if src_idx == -1 or dst_idx == -1:
            return None
        
        # Get container type index
        type_idx = self.container_type_to_idx.get(container_type, 0)  # Default to first type if not found
        
        # Calculate stack height index (convert height to tier count)
        container_height = 2.59  # Standard container height
        stack_idx = min(int(stack_height / container_height), self.max_stack_heights - 1)
        
        # Create result array on device
        result = wp.zeros(1, dtype=wp.float32, device=self.device)
        
        # Get time from lookup table
        wp.launch(
            kernel=self._kernel_get_movement_time,
            dim=1,
            inputs=[
                self.movement_times,
                src_idx,
                dst_idx,
                type_idx,
                stack_idx,
                result
            ]
        )
        
        # Get result as float
        movement_time = float(result.numpy()[0])
        
        # Track lookup time
        self.lookup_times.append(time.time() - start_time)
        
        return movement_time
    
    def _get_position_type(self, position_str: str) -> str:
        """Get the type of an object based on its name."""
        if position_str.startswith('t') and '_' in position_str:
            return 'rail_slot'
        elif position_str.startswith('p_'):
            return 'parking_spot'
        else:
            return 'storage_slot'
            
    def print_performance_stats(self) -> None:
        """Print performance statistics for movement time lookups."""
        if not self.lookup_times:
            print("No lookup performance data available.")
            return
        
        avg_lookup = sum(self.lookup_times) / len(self.lookup_times) * 1000  # Convert to ms
        print(f"Movement lookup performance: {avg_lookup:.4f}ms average")
        
        # Memory usage statistics
        if self.movement_times is not None:
            memory_usage = (self.num_positions * self.num_positions * 
                          len(self.container_types) * self.max_stack_heights * 4) / (1024 * 1024)
            print(f"Movement lookup table memory usage: {memory_usage:.2f} MB")