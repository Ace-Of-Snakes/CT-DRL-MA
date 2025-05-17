import warp as wp
import numpy as np
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
        
        # Register kernels
        self._register_kernels()
        
        print(f"WarpEventPrecomputer initialized on device: {self.device}")
        print(f"Simulation days: {self.simulation_days}, Max events: {self.max_events}")

    def _register_kernels(self):
        """Register kernels for event operations."""
        # Define kernels here if needed
        pass

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
        
        min_time_diff = 1e20  # Very large number
        min_time_idx = -1
        
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
        # Reset counter
        self.event_count = 0
        
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
        
        print(f"Precomputed {self.event_count} events for {self.simulation_days} days")
        self._sort_events()

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
        # For simplicity, we use a bubble sort here
        # More efficient sorting algorithms could be implemented
        for _ in range(self.event_count):
            for j in range(self.event_count - 1):
                time1 = self.precomputed_events[j, 0].numpy()
                time2 = self.precomputed_events[j+1, 0].numpy()
                
                if time1 > time2:
                    # Swap events
                    wp.launch(
                        kernel=self._kernel_swap_events,
                        dim=1,
                        inputs=[
                            self.precomputed_events,
                            j,
                            j+1
                        ]
                    )

    def get_next_event(self, current_time):
        """Find the next event after current_time."""
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
        
        if next_time < 0:
            return None, None, None
        
        return next_time, event_type, event_data

    def fast_forward_to_next_event(self, current_time):
        """
        Fast forward to the next event.
        
        Returns:
            Tuple of (next_event_time, event_type, event_data)
        """
        return self.get_next_event(current_time)