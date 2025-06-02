import warp as wp
import numpy as np
import time
from typing import Dict, Tuple, List, Optional, Any

class WarpEventPrecomputer:
    """
    Precomputes all vehicle arrivals/departures at simulation start to enable time warping.
    
    Trades memory for dramatic speed improvements by:
    1. Generating all events for the entire simulation at initialization
    2. Allowing simulation to jump directly between events
    3. Eliminating incremental time advancement
    
    Memory usage: O(max_events) where max_events = simulation_days * events_per_day
    """
    
    def __init__(self, terminal_state, max_simulation_time, max_trucks_per_day, max_trains_per_day, device=None):
        """
        Initialize the event precomputer.
        
        Args:
            terminal_state: Reference to the WarpTerminalState object
            max_simulation_time: Maximum simulation time in seconds
            max_trucks_per_day: Maximum number of trucks per day
            max_trains_per_day: Maximum number of trains per day
            device: Computation device (if None, will use terminal_state's device)
        """
        self.terminal_state = terminal_state
        self.max_simulation_time = max_simulation_time
        self.max_trucks_per_day = max_trucks_per_day
        self.max_trains_per_day = max_trains_per_day
        self.device = device if device else terminal_state.device
        
        # Calculate number of simulation days
        self.simulation_days = int(max_simulation_time / 86400) + 1
        
        # Estimate maximum number of events
        events_per_day = (max_trucks_per_day + max_trains_per_day) * 2  # Arrive & depart
        self.max_events = self.simulation_days * events_per_day
        
        # Initialize event arrays
        self.precomputed_events = wp.zeros((self.max_events, 3), dtype=wp.float32, device=self.device)
        self.event_count = 0
        self.current_event_idx = 0
        
        # Flag to track if events have been precomputed
        self.precomputed = False
        
        # Performance tracking
        self.event_lookup_times = []
        self.time_advancements = []
        
        print(f"WarpEventPrecomputer initialized on device: {self.device}")
        print(f"Simulation days: {self.simulation_days}, Max events: {self.max_events}")
        print(f"Memory estimate: {self._estimate_memory_usage():.2f} MB")
    
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in megabytes."""
        # Each event entry has 3 float32 values (time, type, data)
        return self.max_events * 3 * 4 / (1024 * 1024)  # Convert bytes to MB
    
    @wp.kernel
    def _kernel_add_event(precomputed_events: wp.array(dtype=wp.float32, ndim=2),
                       event_idx: wp.int32,
                       time: wp.float32,
                       event_type: wp.int32,
                       event_data: wp.int32):
        """Kernel to add an event to the precomputed events array."""
        precomputed_events[event_idx, 0] = time
        precomputed_events[event_idx, 1] = float(event_type)
        precomputed_events[event_idx, 2] = float(event_data)

    @wp.kernel
    def _kernel_find_next_event(precomputed_events: wp.array(dtype=wp.float32, ndim=2),
                             current_time: wp.float32,
                             event_count: wp.int32,
                             result: wp.array(dtype=wp.float32, ndim=1)):
        """Find the next event after current_time."""
        # Default result: -1 for not found
        result[0] = -1.0
        result[1] = -1.0
        result[2] = -1.0
        
        min_time_diff = float(1e20)  # Very large number
        min_time_idx = int(-1)
        
        # Linear search through events
        for i in range(event_count):
            event_time = precomputed_events[i, 0]
            if event_time > current_time:
                time_diff = event_time - current_time
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    min_time_idx = i
        
        # If found, set result
        if min_time_idx >= 0:
            result[0] = precomputed_events[min_time_idx, 0]  # Time
            result[1] = precomputed_events[min_time_idx, 1]  # Type
            result[2] = precomputed_events[min_time_idx, 2]  # Data

    def _add_event(self, time, event_type, event_data):
        """Add an event to the precomputed array."""
        if self.event_count >= self.max_events:
            return False
        
        # Use kernel to add event
        wp.launch(
            kernel=self._kernel_add_event,
            dim=1,
            inputs=[
                self.precomputed_events,
                self.event_count,
                float(time),
                int(event_type),
                int(event_data)
            ]
        )
        
        self.event_count += 1
        return True

    def precompute_events(self):
        """Generate all events for the entire simulation period."""
        if self.precomputed:
            print("Events already precomputed, skipping.")
            return
            
        start_time = time.time()
        
        # Reset counter
        self.event_count = 0
        self.current_event_idx = 0
        
        # Set seed for reproducibility
        np.random.seed(42)
        
        # Generate events for each day
        for day in range(self.simulation_days):
            day_start = day * 86400
            
            # Generate truck arrivals for this day
            num_trucks = np.random.randint(5, self.max_trucks_per_day + 1)
            for i in range(num_trucks):
                # Random arrival time within the day
                arrival_time = day_start + np.random.uniform(0, 86400)
                self._add_event(arrival_time, 0, i)  # 0 = truck arrival
            
            # Generate train arrivals for this day
            num_trains = np.random.randint(1, self.max_trains_per_day + 1)
            for i in range(num_trains):
                # Trains arrive at more specific times
                arrival_time = day_start + np.random.uniform(0, 86400)
                self._add_event(arrival_time, 1, i)  # 1 = train arrival
        
        print(f"Generated {self.event_count} events for {self.simulation_days} days")
        
        # Sort events by time
        self._sort_events()
        
        end_time = time.time()
        self.precomputed = True
        print(f"Event precomputation completed in {end_time - start_time:.2f} seconds")

    @wp.kernel
    def _kernel_swap_events(precomputed_events: wp.array(dtype=wp.float32, ndim=2),
                         idx1: wp.int32,
                         idx2: wp.int32):
        """Swap two events in the array."""
        # Swap all fields
        for i in range(3):
            temp = precomputed_events[idx1, i]
            precomputed_events[idx1, i] = precomputed_events[idx2, i]
            precomputed_events[idx2, i] = temp

    def _sort_events(self):
        """Sort events by time using a simple bubble sort on GPU."""
        # For simplicity, we convert to NumPy, sort, and convert back
        # This is more efficient than implementing a full sort on GPU for this use case
        events_np = self.precomputed_events.numpy()[:self.event_count]
        
        # Sort by time (first column)
        sorted_indices = np.argsort(events_np[:, 0])
        sorted_events = events_np[sorted_indices]
        
        # Update the GPU array with sorted events
        for i in range(self.event_count):
            wp.launch(
                kernel=self._kernel_add_event,
                dim=1,
                inputs=[
                    self.precomputed_events,
                    i,
                    float(sorted_events[i, 0]),
                    int(sorted_events[i, 1]),
                    int(sorted_events[i, 2])
                ]
            )

    def get_next_event(self, current_time):
        """Find the next event after current_time."""
        start_time = time.time()
        
        # Ensure events have been precomputed
        if not self.precomputed:
            self.precompute_events()
        
        # Prepare result array
        result = wp.zeros(3, dtype=wp.float32, device=self.device)
        
        # Launch kernel to find next event
        wp.launch(
            kernel=self._kernel_find_next_event,
            dim=1,
            inputs=[
                self.precomputed_events,
                float(current_time),
                self.event_count,
                result
            ]
        )
        
        # Convert result to Python values
        result_np = result.numpy()
        next_time = float(result_np[0])
        event_type = int(result_np[1])
        event_data = int(result_np[2])
        
        # Track lookup time
        self.event_lookup_times.append(time.time() - start_time)
        
        if next_time < 0:
            return None, None, None
        
        return next_time, event_type, event_data

    def fast_forward_to_next_event(self, current_time, max_advance=1800.0):
        """
        Fast forward to the next event.
        
        Args:
            current_time: Current simulation time
            max_advance: Maximum time to advance in seconds (default: 30 minutes)
            
        Returns:
            Tuple of (time_advanced, next_event_time, event_type, event_data)
        """
        start_time = time.time()
        
        # Find next event
        next_time, event_type, event_data = self.get_next_event(current_time)
        
        if next_time is None:
            # No more events, advance by default time
            time_advanced = 300.0  # 5 minutes
            self.time_advancements.append(time_advanced)
            return time_advanced, None, None, None
        
        # Calculate time to advance
        time_delta = next_time - current_time
        
        # Don't advance more than max_advance
        if time_delta > max_advance:
            time_advanced = max_advance
            self.time_advancements.append(time_advanced)
            return time_advanced, None, None, None
        
        # Track time advancement
        self.time_advancements.append(time_delta)
        
        # Return time advanced and event info
        return time_delta, next_time, event_type, event_data
    
    def load_events(self, events_data):
        """
        Load precomputed events from NumPy array.
        
        Args:
            events_data: Dictionary containing events data
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            if not isinstance(events_data, dict):
                print("Error: events_data must be a dictionary")
                return False
                
            # Verify configuration
            if (events_data.get('max_simulation_time', 0) != self.max_simulation_time or
                events_data.get('max_trucks_per_day', 0) != self.max_trucks_per_day or
                events_data.get('max_trains_per_day', 0) != self.max_trains_per_day):
                print("Warning: Configuration mismatch, continuing anyway")
            
            # Load events and count
            events = events_data.get('events')
            self.event_count = events_data.get('count', 0)
            
            if events is None or self.event_count <= 0:
                print("Error: Invalid events data")
                return False
                
            # Copy events to GPU array
            if events.shape[0] < self.event_count:
                print(f"Error: Events array has {events.shape[0]} events, but count is {self.event_count}")
                self.event_count = min(self.event_count, events.shape[0])
                
            # Update GPU array
            for i in range(self.event_count):
                wp.launch(
                    kernel=self._kernel_add_event,
                    dim=1,
                    inputs=[
                        self.precomputed_events,
                        i,
                        float(events[i, 0]),
                        int(events[i, 1]),
                        int(events[i, 2])
                    ]
                )
            
            self.precomputed = True
            return True
        except Exception as e:
            print(f"Error loading precomputed events: {e}")
            return False
    
    def print_performance_stats(self):
        """Print performance statistics for event precomputation."""
        print("\nEvent Precomputation Performance:")
        
        # Memory usage
        memory_usage = self._estimate_memory_usage()
        print(f"  Memory usage: {memory_usage:.2f} MB")
        print(f"  Total events: {self.event_count}")
        
        # Event lookup performance
        if self.event_lookup_times:
            avg_lookup = sum(self.event_lookup_times) / len(self.event_lookup_times) * 1000  # Convert to ms
            print(f"  Event lookup time: {avg_lookup:.4f}ms average")
            print(f"  Event lookups: {len(self.event_lookup_times)}")
        
        # Time advancement statistics
        if self.time_advancements:
            avg_advance = sum(self.time_advancements) / len(self.time_advancements)
            print(f"  Average time advancement: {avg_advance:.2f} seconds")
            print(f"  Total time advancements: {len(self.time_advancements)}")