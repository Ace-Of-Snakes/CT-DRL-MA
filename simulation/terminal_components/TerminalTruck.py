# simulation/terminal_components/TerminalTruck.py
from datetime import datetime, timedelta
import random
from typing import List, Dict, Optional, Tuple, Set
from simulation.terminal_components.Container import Container

class TerminalTruck:
    """
    Represents a terminal-owned truck for moving containers within the terminal.
    
    These trucks are specialized for moving swap bodies and trailers between
    storage positions, freeing up valuable specialized storage areas.
    """
    
    # Possible terminal truck statuses
    IDLE = "idle"           # Available for use
    BUSY = "busy"           # Currently performing a task
    MAINTENANCE = "maint"   # Unavailable due to maintenance
    
    def __init__(self, 
                 truck_id: str = None, 
                 max_length: float = 15.0,
                 max_weight: float = 60000):
        """Initialize a new terminal truck."""
        self.truck_id = truck_id or f"TTR{random.randint(10000, 99999)}"
        self.max_length = max_length
        self.max_weight = max_weight
        self.containers = []  # List of Container objects currently loaded
        self.status = self.IDLE
        self.current_source = None  # Current source position
        self.current_destination = None  # Current destination position
        self.task_start_time = None  # When current task started
        self.task_completion_time = None  # When current task will complete
    
    def add_container(self, container: Container) -> bool:
        """
        Add a container to the truck if it's a swap body or trailer.
        
        Args:
            container: Container object to add
            
        Returns:
            bool: True if container was added successfully, False otherwise
        """
        # Only allow swap bodies and trailers
        if container.container_type not in ["Trailer", "Swap Body"]:
            return False
        
        # Don't allow multiple containers
        if self.containers:
            return False
        
        # Check weight
        if container.weight > self.max_weight:
            return False
        
        self.containers.append(container)
        return True
    
    def remove_container(self) -> Optional[Container]:
        """
        Remove the container from the truck.
        
        Returns:
            Container or None: The removed container, or None if truck is empty
        """
        if not self.containers:
            return None
        
        return self.containers.pop(0)
    
    def assign_task(self, source: str, destination: str, task_time: float):
        """
        Assign a transport task to the terminal truck.
        
        Args:
            source: Source position
            destination: Destination position
            task_time: Time in seconds the task will take
        """
        self.status = self.BUSY
        self.current_source = source
        self.current_destination = destination
        self.task_start_time = datetime.now()
        self.task_completion_time = self.task_start_time + timedelta(seconds=task_time)
    
    def complete_task(self):
        """Mark the current task as completed."""
        self.status = self.IDLE
        self.current_source = None
        self.current_destination = None
        self.task_start_time = None
        self.task_completion_time = None
        self.containers = []  # Empty the truck
    
    def is_available(self) -> bool:
        """Check if the terminal truck is available for a new task."""
        return self.status == self.IDLE and not self.containers
    
    def get_remaining_task_time(self, current_time: datetime) -> float:
        """
        Get the remaining time for the current task.
        
        Args:
            current_time: Current simulation time
            
        Returns:
            float: Remaining time in seconds, or 0 if no task
        """
        if self.status != self.BUSY or not self.task_completion_time:
            return 0.0
        
        remaining = (self.task_completion_time - current_time).total_seconds()
        return max(0.0, remaining)
    
    def __str__(self):
        if not self.containers:
            status_str = "Empty"
        else:
            container_str = ", ".join(c.container_id for c in self.containers)
            status_str = f"Carrying: {container_str}"
        
        return f"Terminal Truck {self.truck_id}: {status_str}, status: {self.status}"