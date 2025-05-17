import warp as wp
import numpy as np
import torch
import time
import os
from typing import Dict, List, Tuple, Optional, Union, Any

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
        try:
            start_time = time.time()
            
            # Import optimization classes here to avoid circular imports
            from simulation.precomputing.WarpEventPrecomputer import WarpEventPrecomputer
            from simulation.precomputing.WarpMovementLookupTable import WarpMovementLookupTable
            from simulation.precomputing.WarpStackingCompatibilityMatrix import WarpStackingCompatibilityMatrix
            
            # Initialize movement lookup if enabled
            if self.precompute_movements:
                print("\nInitializing movement lookup tables...")
                self.movement_lookup = WarpMovementLookupTable(
                    self.terminal_state,
                    self.terminal_env.movement_calculator,
                    device=self.device
                )
                self.movement_lookup.precompute_movement_times()
                
                # Configure movement calculator to use lookup tables
                if hasattr(self.terminal_env.movement_calculator, 'use_lookup_tables'):
                    self.terminal_env.movement_calculator.use_lookup_tables(
                        self.movement_lookup.movement_times.numpy(),
                        self.movement_lookup.container_types,
                        self.movement_lookup.max_stack_heights
                    )
            
            # Initialize stacking compatibility if enabled
            if self.precompute_stacking:
                print("\nInitializing stacking compatibility matrix...")
                self.stacking_matrix = WarpStackingCompatibilityMatrix(
                    self.terminal_state, 
                    self.terminal_env.container_registry,
                    device=self.device
                )
                self.stacking_matrix.precompute_compatibility()
                
                # Configure stacking kernels to use compatibility matrix
                if hasattr(self.terminal_env.stacking_kernels, 'use_compatibility_matrix'):
                    self.terminal_env.stacking_kernels.use_compatibility_matrix(
                        self.stacking_matrix.compatibility_matrix.numpy()
                    )
            
            # Initialize event precomputation if enabled
            if self.precompute_events:
                print("\nInitializing event precomputation...")
                self.event_precomputer = WarpEventPrecomputer(
                    self.terminal_state,
                    self.terminal_env.max_simulation_time,
                    self.terminal_env.max_trucks_per_day,
                    self.terminal_env.max_trains_per_day,
                    self.device
                )
                self.event_precomputer.precompute_events()
            
            end_time = time.time()
            print(f"\nOptimization initialization completed in {end_time - start_time:.2f} seconds")
            
            # Report memory usage
            memory_usage = self._calculate_memory_usage()
            print(f"Total memory usage: {memory_usage:.2f} MB")
            
        except ImportError as e:
            print(f"Failed to import optimization components: {e}")
            print("Using standard simulation instead")
        except Exception as e:
            print(f"Error initializing optimizations: {e}")
            import traceback
            traceback.print_exc()
    
    def _calculate_memory_usage(self) -> float:
        """Calculate total memory usage of all optimization components."""
        memory_usage = 0.0
        
        if self.movement_lookup is not None and hasattr(self.movement_lookup, '_estimate_memory_usage'):
            memory_usage += self.movement_lookup._estimate_memory_usage()
            
        if self.stacking_matrix is not None and hasattr(self.stacking_matrix, '_estimate_memory_usage'):
            memory_usage += self.stacking_matrix._estimate_memory_usage()
            
        if self.event_precomputer is not None and hasattr(self.event_precomputer, '_estimate_memory_usage'):
            memory_usage += self.event_precomputer._estimate_memory_usage()
            
        return memory_usage
    
    def optimized_step(self, action: Dict) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Perform an optimized step using all available optimizations.
        
        Args:
            action: Action dictionary from the agent
            
        Returns:
            Tuple of (next_state, reward, terminated, truncated, info)
        """
        step_start_time = time.time()
        
        # Handle None action (wait)
        if action is None:
            return self._execute_wait_action(step_start_time)
        
        # Execute action
        action_type = action.get('action_type', 0)
        
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
            return self._execute_wait_action(step_start_time)
        
        # Update terminal state simulation time
        current_time = self.terminal_env.current_simulation_time
        self.terminal_env.terminal_state.simulation_time[0] = float(current_time)
        
        # Process arrivals and departures
        if not self.precompute_events:
            # Use original processing
            self.terminal_env._process_vehicle_arrivals()
            self.terminal_env._process_vehicle_departures()
        
        # Check if episode is done
        terminated = False
        truncated = current_time >= self.terminal_env.max_simulation_time
        
        # Get new observation
        observation = self.terminal_env._get_observation()
        
        # Create info dictionary
        step_time = time.time() - step_start_time
        info = {
            'simulation_time': current_time,
            'simulation_datetime': self.terminal_env.current_simulation_datetime,
            'trucks_handled': self.terminal_env.trucks_arrived,
            'trains_handled': self.terminal_env.trains_arrived,
            'containers_moved': self.terminal_env.containers_moved,
            'step_compute_time': step_time,
            'optimization_enabled': True
        }
        
        # Track step time
        self.step_times.append(step_time)
        
        return observation, reward, terminated, truncated, info
    
    def _execute_wait_action(self, step_start_time):
        """Execute a wait action with time advancement."""
        # Advance time with event precomputation if available
        if self.precompute_events and self.event_precomputer is not None:
            event_start = time.time()
            time_delta, next_time, event_type, event_data = self.event_precomputer.fast_forward_to_next_event(
                self.terminal_env.current_simulation_time
            )
            
            # Track event processing time
            self.event_times.append(time.time() - event_start)
            
            # Update simulation time
            self.terminal_env._advance_time(time_delta)
            
            # Process events if found
            if next_time is not None:
                if event_type == 0:  # Truck arrival
                    self.terminal_env._create_truck_arrival()
                elif event_type == 1:  # Train arrival
                    self.terminal_env._create_train_arrival()
        else:
            # Use standard time advancement
            self.terminal_env._advance_time(300)  # 5 minutes
            self.terminal_env._process_vehicle_arrivals()
            self.terminal_env._process_vehicle_departures()
        
        # No reward for waiting
        reward = 0.0
        
        # Update terminal state simulation time
        current_time = self.terminal_env.current_simulation_time
        self.terminal_env.terminal_state.simulation_time[0] = float(current_time)
        
        # Check if episode is done
        terminated = False
        truncated = current_time >= self.terminal_env.max_simulation_time
        
        # Get new observation
        observation = self.terminal_env._get_observation()
        
        # Create info dictionary
        step_time = time.time() - step_start_time
        info = {
            'simulation_time': current_time,
            'simulation_datetime': self.terminal_env.current_simulation_datetime,
            'trucks_handled': self.terminal_env.trucks_arrived,
            'trains_handled': self.terminal_env.trains_arrived,
            'containers_moved': self.terminal_env.containers_moved,
            'step_compute_time': step_time,
            'optimization_enabled': True,
            'wait_action': True
        }
        
        # Track step time
        self.step_times.append(step_time)
        
        return observation, reward, terminated, truncated, info
    
    def _execute_optimized_crane_movement(self, crane_action):
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
                    
                    stacking_start = time.time()
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
                    
                    self.stacking_times.append(time.time() - stacking_start)
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
        movement_start = time.time()
        if self.movement_lookup is not None:
            container_type_str = ["TEU", "FEU", "HQ", "Trailer", "Swap Body"][container_type] 
            movement_time = self.movement_lookup.get_movement_time(
                src_pos, dst_pos, container_type_str, stack_height
            )
        else:
            # Fall back to original movement calculator
            movement_time = self.terminal_env.movement_calculator.calculate_movement_time(
                src_pos, dst_pos, crane_idx, container_type, stack_height
            )
        
        self.movement_times.append(time.time() - movement_start)
        
        # Update crane position and time
        self.terminal_state.crane_properties[crane_idx, 2] = self.terminal_env.current_simulation_time + movement_time
        
        # Update crane position for visualization
        dst_coords = self.terminal_env._get_position_coordinates(dst_pos)
        self.terminal_state.crane_positions[crane_idx, 0] = dst_coords[0]
        self.terminal_state.crane_positions[crane_idx, 1] = dst_coords[1]
        
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
        self.terminal_state.container_positions[container_idx] = dst_position_idx
        
        # Increment containers moved
        self.terminal_env.containers_moved += 1
        
        # Reward proportional to movement efficiency (inverse of time)
        reward = 10.0 / (1.0 + movement_time / 60.0)  # Normalize to ~0-10 range
        
        return reward
    
    def replace_env_step(self) -> None:
        """Replace the environment's step method with the optimized version."""
        if not hasattr(self.terminal_env, 'original_step'):
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
            min_step = min(self.step_times) * 1000
            max_step = max(self.step_times) * 1000
            print(f"  Overall step time: {avg_step:.2f}ms average (range: {min_step:.2f}ms - {max_step:.2f}ms)")
            print(f"  Total steps: {len(self.step_times)}")
        
        if self.movement_times:
            avg_movement = sum(self.movement_times) / len(self.movement_times) * 1000  # ms
            print(f"  Movement calculation time: {avg_movement:.2f}ms average")
            print(f"  Total movements: {len(self.movement_times)}")
        
        if self.stacking_times:
            avg_stacking = sum(self.stacking_times) / len(self.stacking_times) * 1000  # ms
            print(f"  Stacking validation time: {avg_stacking:.2f}ms average")
            print(f"  Total validations: {len(self.stacking_times)}")
        
        if self.event_times:
            avg_event = sum(self.event_times) / len(self.event_times) * 1000  # ms
            print(f"  Event processing time: {avg_event:.2f}ms average")
            print(f"  Total event processings: {len(self.event_times)}")
        
        # Memory usage
        memory_usage = self._calculate_memory_usage()
        print(f"\nTotal memory usage: {memory_usage:.2f} MB")
        
        # Calculate speedup estimates
        if hasattr(self.terminal_env, 'step_times') and self.terminal_env.step_times and self.step_times:
            original_avg = sum(self.terminal_env.step_times) / len(self.terminal_env.step_times) * 1000
            optimized_avg = avg_step
            speedup = original_avg / optimized_avg if optimized_avg > 0 else 0
            print(f"\nEstimated speedup: {speedup:.2f}x faster")
        
        # Print component-specific stats
        print("\nOptimization Component Statistics:")
        
        if self.movement_lookup is not None and hasattr(self.movement_lookup, 'print_performance_stats'):
            print("\nMovement Lookup:")
            self.movement_lookup.print_performance_stats()
        
        if self.stacking_matrix is not None and hasattr(self.stacking_matrix, 'print_performance_stats'):
            print("\nStacking Compatibility Matrix:")
            self.stacking_matrix.print_performance_stats()
        
        if self.event_precomputer is not None and hasattr(self.event_precomputer, 'print_performance_stats'):
            print("\nEvent Precomputer:")
            self.event_precomputer.print_performance_stats()

def create_optimized_simulator(terminal_env, 
                             precompute_events=True, 
                             precompute_movements=True, 
                             precompute_stacking=True,
                             tables_dir='precomputed',
                             load_tables=False,
                             save_tables=False):
    """
    Convenience function to create an optimized simulator and replace step method.
    
    Args:
        terminal_env: Terminal environment to optimize
        precompute_events: Whether to precompute events
        precompute_movements: Whether to precompute movement times
        precompute_stacking: Whether to precompute stacking compatibility
        tables_dir: Directory for precomputed tables
        load_tables: Whether to load tables from disk
        save_tables: Whether to save tables to disk
        
    Returns:
        Initialized WarpOptimizedSimulator
    """
    # Create directory if saving tables
    if save_tables:
        os.makedirs(tables_dir, exist_ok=True)
    
    # Create simulator
    simulator = WarpOptimizedSimulator(
        terminal_env=terminal_env,
        initialize_on_creation=False,
        precompute_events=precompute_events,
        precompute_movements=precompute_movements,
        precompute_stacking=precompute_stacking,
        device=getattr(terminal_env, 'device', 'cuda')
    )
    
    # Try to load tables if requested
    tables_loaded = False
    if load_tables:
        print(f"Attempting to load precomputed tables from {tables_dir}...")
        
        # Check for movement lookup table
        if simulator.precompute_movements:
            movement_path = os.path.join(tables_dir, "movement_lookup.npy")
            if os.path.exists(movement_path):
                print(f"Loading movement lookup table from {movement_path}")
                from simulation.precomputing.WarpMovementLookupTable import WarpMovementLookupTable
                simulator.movement_lookup = WarpMovementLookupTable(
                    terminal_env.terminal_state,
                    terminal_env.movement_calculator
                )
                table_data = np.load(movement_path)
                if simulator.movement_lookup.load_table(table_data):
                    print("Successfully loaded movement lookup table")
                    
                    # Configure movement calculator to use the table
                    if hasattr(terminal_env.movement_calculator, 'use_lookup_tables'):
                        terminal_env.movement_calculator.use_lookup_tables(
                            simulator.movement_lookup.movement_times.numpy(),
                            simulator.movement_lookup.container_types,
                            simulator.movement_lookup.max_stack_heights
                        )
                    
                    tables_loaded = True
        
        # Check for stacking compatibility matrix
        if simulator.precompute_stacking:
            stacking_path = os.path.join(tables_dir, "stacking_matrix.npy")
            if os.path.exists(stacking_path):
                print(f"Loading stacking compatibility matrix from {stacking_path}")
                from simulation.precomputing.WarpStackingCompatibilityMatrix import WarpStackingCompatibilityMatrix
                simulator.stacking_matrix = WarpStackingCompatibilityMatrix(
                    terminal_env.terminal_state,
                    terminal_env.container_registry,
                    use_bit_packing=False
                )
                matrix_data = np.load(stacking_path)
                if simulator.stacking_matrix.load_matrix(matrix_data):
                    print("Successfully loaded stacking compatibility matrix")
                    
                    # Configure stacking kernels to use the matrix
                    if hasattr(terminal_env.stacking_kernels, 'use_compatibility_matrix'):
                        terminal_env.stacking_kernels.use_compatibility_matrix(
                            simulator.stacking_matrix.compatibility_matrix.numpy()
                        )
                    
                    tables_loaded = True
        
        # Check for event precomputation
        if simulator.precompute_events:
            events_path = os.path.join(tables_dir, "precomputed_events.npy")
            if os.path.exists(events_path):
                print(f"Loading precomputed events from {events_path}")
                from simulation.precomputing.WarpEventPrecomputer import WarpEventPrecomputer
                simulator.event_precomputer = WarpEventPrecomputer(
                    terminal_env.terminal_state,
                    terminal_env.max_simulation_time,
                    terminal_env.max_trucks_per_day,
                    terminal_env.max_trains_per_day
                )
                events_data = np.load(events_path, allow_pickle=True).item()
                if simulator.event_precomputer.load_events(events_data):
                    print("Successfully loaded precomputed events")
                    tables_loaded = True
    
    # Initialize simulator if not all tables were loaded
    if not tables_loaded:
        simulator.initialize()
    
    # Save tables if requested
    if save_tables:
        print(f"Saving precomputed tables to {tables_dir}...")
        
        # Save movement lookup table
        if simulator.movement_lookup is not None and simulator.movement_lookup.movement_times is not None:
            movement_path = os.path.join(tables_dir, "movement_lookup.npy")
            np.save(movement_path, simulator.movement_lookup.movement_times.numpy())
            print(f"Saved movement lookup table to {movement_path}")
        
        # Save stacking compatibility matrix
        if simulator.stacking_matrix is not None and simulator.stacking_matrix.compatibility_matrix is not None:
            stacking_path = os.path.join(tables_dir, "stacking_matrix.npy")
            np.save(stacking_path, simulator.stacking_matrix.compatibility_matrix.numpy())
            print(f"Saved stacking compatibility matrix to {stacking_path}")
        
        # Save precomputed events
        if simulator.event_precomputer is not None and simulator.event_precomputer.precomputed_events is not None:
            events_path = os.path.join(tables_dir, "precomputed_events.npy")
            events_data = {
                'events': simulator.event_precomputer.precomputed_events.numpy(),
                'count': simulator.event_precomputer.event_count,
                'max_simulation_time': terminal_env.max_simulation_time,
                'max_trucks_per_day': terminal_env.max_trucks_per_day,
                'max_trains_per_day': terminal_env.max_trains_per_day
            }
            np.save(events_path, events_data)
            print(f"Saved precomputed events to {events_path}")
    
    # Replace environment step with optimized version
    simulator.replace_env_step()
    
    return simulator