import warp as wp
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Union, Any

from WarpEventPrecomputer import WarpEventPrecomputer
from WarpMovementLookupTable import WarpMovementLookupTable
from WarpStackingCompatibilityMatrix import WarpStackingCompatibilityMatrix

class WarpOptimizedSimulator:
    """
    Integrates all optimization components to maximize simulation performance.
    
    Provides a unified interface for:
    1. Event-driven time progression
    2. Fast movement calculations via lookup tables
    3. Efficient container stacking validation
    4. Performance monitoring and statistics
    
    Combined memory usage: Sum of all optimization components
    """
    
    def __init__(self, 
                 terminal_env,
                 initialize_on_creation: bool = True,
                 precompute_events: bool = True,
                 precompute_movements: bool = True,
                 precompute_stacking: bool = True,
                 device: str = None):
        """
        Initialize the optimized simulator.
        
        Args:
            terminal_env: Reference to the terminal environment
            initialize_on_creation: Whether to initialize optimizations immediately
            precompute_events: Whether to enable event precomputation
            precompute_movements: Whether to enable movement lookup tables
            precompute_stacking: Whether to enable stacking compatibility matrix
            device: Computation device (if None, will use terminal_env's device)
        """
        self.terminal_env = terminal_env
        self.terminal_state = terminal_env.terminal_state
        self.device = device if device else getattr(terminal_env, 'device', 'cuda')
        
        # Track which optimizations are enabled
        self.precompute_events = precompute_events
        self.precompute_movements = precompute_movements
        self.precompute_stacking = precompute_stacking
        
        # Initialize component instances
        self.event_precomputer = None
        self.movement_lookup = None 
        self.stacking_matrix = None
        
        # Performance monitoring
        self.step_times = []
        self.movement_times = []
        self.stacking_times = []
        self.event_times = []
        
        print(f"WarpOptimizedSimulator initialized on device: {self.device}")
        
        if initialize_on_creation:
            self.initialize()
    
    def initialize(self) -> None:
        """Initialize all enabled optimization components."""
        start_time = time.time()
        
        if self.precompute_events:
            print("Initializing event precomputation...")
            self.event_precomputer = WarpEventPrecomputer(
                self.terminal_state,
                self.terminal_env.max_simulation_time,
                self.terminal_env.max_trucks_per_day,
                self.terminal_env.max_trains_per_day,
                self.device
            )
            self.event_precomputer.precompute_events()
        
        if self.precompute_movements:
            print("Initializing movement lookup tables...")
            self.movement_lookup = WarpMovementLookupTable(
                self.terminal_state,
                self.terminal_env.movement_calculator,
                device=self.device
            )
            self.movement_lookup.precompute_movement_times()
        
        if self.precompute_stacking:
            print("Initializing stacking compatibility matrix...")
            self.stacking_matrix = WarpStackingCompatibilityMatrix(
                self.terminal_state, 
                self.terminal_env.container_registry,
                device=self.device
            )
            self.stacking_matrix.precompute_compatibility()
        
        end_time = time.time()
        print(f"Optimization initialization completed in {end_time - start_time:.2f} seconds")
    
    def optimized_step(self, action: Dict) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Perform an optimized step using all available optimizations.
        
        Args:
            action: Action dictionary from the agent
            
        Returns:
            Tuple of (next_state, reward, terminated, truncated, info)
        """
        step_start_time = time.time()
        
        # Execute action similar to original step method
        action_type = action['action_type'] 
        
        if action_type == 0:  # Crane movement
            if self.precompute_movements and self.movement_lookup is not None:
                # Use movement lookup table instead of physics calculation
                reward = self._execute_optimized_crane_movement(action['crane_movement'])
            else:
                # Use original movement calculation
                reward = self.terminal_env._execute_crane_movement(action['crane_movement'])
        elif action_type == 1:  # Truck parking
            reward = self.terminal_env._execute_truck_parking(action['truck_parking'])
        elif action_type == 2:  # Terminal truck
            reward = self.terminal_env._execute_terminal_truck(action['terminal_truck'])
        else:
            # Invalid action type
            reward = 0.0
        
        # If no action was taken, advance simulation time
        if reward == 0:
            if self.precompute_events and self.event_precomputer is not None:
                # Use event precomputation to jump to next event
                self.fast_forward_to_next_event()
            else:
                # Use standard time advancement
                self.terminal_env._advance_time(300)  # 5 minutes
        
        # Update terminal state simulation time
        current_time = self.terminal_env.current_simulation_time
        wp.launch(
            kernel=self._kernel_set_simulation_time,
            dim=1,
            inputs=[
                self.terminal_state.simulation_time,
                float(current_time)
            ]
        )
        
        # Process arrivals and departures
        if self.precompute_events and self.event_precomputer is not None:
            # Events already processed in fast_forward
            pass
        else:
            # Use original processing
            self.terminal_env._process_vehicle_arrivals()
            self.terminal_env._process_vehicle_departures()
        
        # Check if episode is done
        terminated = False
        truncated = current_time >= self.terminal_env.max_simulation_time
        
        # Get new observation
        observation = self.terminal_env._get_observation()
        
        # Create info dictionary
        info = {
            'simulation_time': current_time,
            'simulation_datetime': self.terminal_env.current_simulation_datetime,
            'trucks_handled': self.terminal_env.trucks_arrived,
            'trains_handled': self.terminal_env.trains_arrived,
            'containers_moved': self.terminal_env.containers_moved,
            'optimization_enabled': True
        }
        
        # Track step time
        step_time = time.time() - step_start_time
        self.step_times.append(step_time)
        
        return observation, reward, terminated, truncated, info
    
    def _execute_optimized_crane_movement(self, crane_action: np.ndarray) -> float:
        """
        Execute a crane movement using optimized lookup tables.
        
        Args:
            crane_action: Crane movement action [crane_idx, src_idx, dst_idx]
            
        Returns:
            Reward for the action
        """
        movement_start_time = time.time()
        
        crane_idx, src_idx, dst_idx = crane_action
        
        # Convert indices to position strings
        src_pos = self.terminal_env.idx_to_position.get(src_idx, None)
        dst_pos = self.terminal_env.idx_to_position.get(dst_idx, None)
        
        if src_pos is None or dst_pos is None:
            # Invalid position indices
            return 0.0
        
        # Check if crane is available
        crane_props_np = self.terminal_state.crane_properties.numpy()
        crane_available_time = float(crane_props_np[crane_idx, 2])
        if crane_available_time > self.terminal_env.current_simulation_time:
            return 0.0
        
        # Get container at source position
        container_idx = -1
        if src_pos[0].isalpha() and src_pos[0].upper() in self.terminal_env.storage_yard.row_names:
            # Storage position
            container_idx, _ = self.terminal_env.storage_yard.get_top_container(src_pos)
            if container_idx is None:
                container_idx = -1
        else:
            # Rail or parking position
            # This would normally check vehicles at these positions
            pass
        
        if container_idx < 0:
            # No container at source position
            return 0.0
        
        # Check if container can be placed at destination
        can_place = False
        if dst_pos[0].isalpha() and dst_pos[0].upper() in self.terminal_env.storage_yard.row_names:
            # Storage position
            if self.precompute_stacking and self.stacking_matrix is not None:
                # Use stacking matrix to check if placement is valid
                row, bay = self.terminal_env._parse_position(dst_pos)
                if row is not None and bay is not None:
                    stack_heights_np = self.terminal_state.stack_heights.numpy()
                    height = int(stack_heights_np[row, bay])
                    
                    if height > 0:
                        # Get top container in destination stack
                        yard_indices_np = self.terminal_state.yard_container_indices.numpy()
                        top_container_idx = int(yard_indices_np[row, bay, height - 1])
                        
                        if top_container_idx >= 0:
                            # Check if can stack using matrix
                            can_place = self.stacking_matrix.can_stack(container_idx, top_container_idx)
                        else:
                            can_place = True  # No container in stack
                    else:
                        can_place = True  # Empty stack
            else:
                # Use original validation
                can_place = self.terminal_env.storage_yard.can_accept_container(dst_pos, container_idx)
        else:
            # Rail or parking position - assume true for simplicity
            can_place = True
        
        if not can_place:
            # Cannot place container at destination
            return 0.0
        
        # Get container type from properties
        container_props_np = self.terminal_state.container_properties.numpy()
        container_type = int(container_props_np[container_idx, 0])
        
        # Get stack height if needed (only for storage destinations)
        stack_height = 0.0
        if dst_pos[0].isalpha() and dst_pos[0].upper() in self.terminal_env.storage_yard.row_names:
            row, bay = self.terminal_env._parse_position(dst_pos)
            if row is not None and bay is not None:
                stack_heights_np = self.terminal_state.stack_heights.numpy()
                stack_height = float(stack_heights_np[row, bay])
        
        # Get movement time from lookup table
        container_type_str = ["TEU", "FEU", "HQ", "Trailer", "Swap Body"][container_type] 
        movement_time = self.movement_lookup.get_movement_time(
            src_pos, dst_pos, container_type_str, stack_height
        )
        
        # Update crane position and time using kernel
        wp.launch(
            kernel=self._kernel_set_crane_available_time,
            dim=1,
            inputs=[
                self.terminal_state.crane_properties,
                crane_idx,
                float(self.terminal_env.current_simulation_time + movement_time)
            ]
        )
        
        # Update crane position for visualization
        dst_coords = self.terminal_env._get_position_coordinates(dst_pos)
        wp.launch(
            kernel=self._kernel_set_crane_position,
            dim=1,
            inputs=[
                self.terminal_state.crane_positions,
                crane_idx,
                0,  # X coordinate
                float(dst_coords[0])
            ]
        )
        
        wp.launch(
            kernel=self._kernel_set_crane_position,
            dim=1,
            inputs=[
                self.terminal_state.crane_positions,
                crane_idx,
                1,  # Y coordinate
                float(dst_coords[1])
            ]
        )
        
        # Remove container from source
        if src_pos[0].isalpha() and src_pos[0].upper() in self.terminal_env.storage_yard.row_names:
            # Storage position
            self.terminal_env.storage_yard.remove_container(src_pos)
        
        # Place container at destination
        if dst_pos[0].isalpha() and dst_pos[0].upper() in self.terminal_env.storage_yard.row_names:
            # Storage position
            self.terminal_env.storage_yard.add_container(container_idx, dst_pos)
        
        # Update container position
        dst_position_idx = self.terminal_env.position_to_idx.get(dst_pos, -1)
        wp.launch(
            kernel=self._kernel_update_container_position,
            dim=1,
            inputs=[
                self.terminal_state.container_positions, 
                container_idx, 
                dst_position_idx
            ]
        )
        
        # Increment containers moved
        self.terminal_env.containers_moved += 1
        
        # Track performance
        self.movement_times.append(time.time() - movement_start_time)
        
        # Reward proportional to movement efficiency (inverse of time)
        reward = 10.0 / (1.0 + movement_time / 60.0)  # Normalize to ~0-10 range
        
        return reward
    
    @wp.kernel
    def _kernel_set_crane_available_time(crane_properties: wp.array(dtype=wp.float32, ndim=2),
                                      crane_idx: wp.int32,
                                      available_time: wp.float32) -> None:
        """Set the available time for a crane."""
        crane_properties[crane_idx, 2] = available_time
    
    @wp.kernel
    def _kernel_set_crane_position(crane_positions: wp.array(dtype=wp.float32, ndim=2),
                                crane_idx: wp.int32,
                                coordinate_idx: wp.int32,
                                value: wp.float32) -> None:
        """Set a coordinate value for a crane position."""
        crane_positions[crane_idx, coordinate_idx] = value
    
    @wp.kernel
    def _kernel_update_container_position(container_positions: wp.array(dtype=wp.int32, ndim=1),
                                       container_idx: wp.int32,
                                       position_idx: wp.int32) -> None:
        """Update a container's position index."""
        container_positions[container_idx] = position_idx
    
    @wp.kernel
    def _kernel_set_simulation_time(simulation_time: wp.array(dtype=wp.float32, ndim=1),
                                 time_value: wp.float32) -> None:
        """Set the simulation time."""
        simulation_time[0] = time_value
    
    def fast_forward_to_next_event(self) -> float:
        """
        Fast forward simulation to next event.
        
        Returns:
            Time advanced in seconds
        """
        event_start_time = time.time()
        
        # Get current time
        current_time = self.terminal_env.current_simulation_time
        
        # Find next event
        next_time, event_type, event_data = self.event_precomputer.fast_forward_to_next_event(current_time)
        
        if next_time is None:
            # No more events, advance by default time
            self.terminal_env._advance_time(300)  # 5 minutes
            self.event_times.append(time.time() - event_start_time)
            return 300.0
        
        # Calculate time to advance
        time_delta = next_time - current_time
        
        # Don't advance more than 30 minutes at once (configurable)
        max_advance = 1800.0  # 30 minutes
        if time_delta > max_advance:
            # Advance by max time instead
            self.terminal_env._advance_time(max_advance)
            self.event_times.append(time.time() - event_start_time)
            return max_advance
        
        # Update simulation time
        self.terminal_env._advance_time(time_delta)
        
        # Process the event
        if event_type == 0:  # Truck arrival
            self.terminal_env._create_truck_arrival()
        elif event_type == 1:  # Train arrival
            self.terminal_env._create_train_arrival()
        
        # Track performance
        self.event_times.append(time.time() - event_start_time)
        
        return time_delta
    
    def replace_env_step(self) -> None:
        """Replace the environment's step method with the optimized version."""
        self.terminal_env.original_step = self.terminal_env.step
        self.terminal_env.step = self.optimized_step
    
    def restore_env_step(self) -> None:
        """Restore the environment's original step method."""
        if hasattr(self.terminal_env, 'original_step'):
            self.terminal_env.step = self.terminal_env.original_step
    
    def print_performance_stats(self) -> None:
        """Print detailed performance statistics."""
        print("\nWarpOptimizedSimulator Performance Statistics:")
        
        if self.step_times:
            avg_step = sum(self.step_times) / len(self.step_times) * 1000  # ms
            print(f"  Overall step time: {avg_step:.2f}ms average")
        
        if self.movement_times:
            avg_movement = sum(self.movement_times) / len(self.movement_times) * 1000  # ms
            print(f"  Crane movement time: {avg_movement:.2f}ms average")
        
        if self.stacking_times:
            avg_stacking = sum(self.stacking_times) / len(self.stacking_times) * 1000  # ms
            print(f"  Stacking validation time: {avg_stacking:.2f}ms average")
        
        if self.event_times:
            avg_event = sum(self.event_times) / len(self.event_times) * 1000  # ms
            print(f"  Event processing time: {avg_event:.2f}ms average")
        
        # Memory usage
        memory_usage = 0.0
        
        if self.precompute_events and self.event_precomputer is not None:
            event_memory = (self.event_precomputer.max_events * 3 * 4) / (1024 * 1024)  # MB
            memory_usage += event_memory
            print(f"  Event precomputation memory: {event_memory:.2f} MB")
        
        if self.precompute_movements and self.movement_lookup is not None:
            self.movement_lookup.print_performance_stats()
            movement_memory = self.movement_lookup._estimate_memory_usage()
            memory_usage += movement_memory
        
        if self.precompute_stacking and self.stacking_matrix is not None:
            self.stacking_matrix.print_performance_stats()
            stacking_memory = self.stacking_matrix._estimate_memory_usage()
            memory_usage += stacking_memory
        
        print(f"\nTotal memory usage: {memory_usage:.2f} MB")
        
        # Calculate speedup estimates
        if hasattr(self.terminal_env, 'step_times') and self.terminal_env.step_times and self.step_times:
            original_avg = sum(self.terminal_env.step_times) / len(self.terminal_env.step_times) * 1000
            optimized_avg = avg_step
            speedup = original_avg / optimized_avg if optimized_avg > 0 else 0
            print(f"\nEstimated speedup: {speedup:.2f}x faster")

def create_optimized_simulator(terminal_env, initialize: bool = True) -> WarpOptimizedSimulator:
    """
    Convenience function to create an optimized simulator and optionally replace step method.
    
    Args:
        terminal_env: Terminal environment to optimize
        initialize: Whether to initialize optimizations immediately
        
    Returns:
        Initialized WarpOptimizedSimulator
    """
    simulator = WarpOptimizedSimulator(terminal_env, initialize_on_creation=initialize)
    simulator.replace_env_step()
    return simulator