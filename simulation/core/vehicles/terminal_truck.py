# simulation/terminal_components/TerminalTruck.py

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from simulation.core.containers.container import Container
from simulation.core.vehicles.truck import Truck

# ==================== TERMINAL TRUCK CONSTANTS ====================

# Physical dimensions (meters)
TERMINAL_TRUCK_MAX_LENGTH = 24.4  # Specialized length for terminal trucks

# Terminal truck statuses (extends parent truck statuses)
TERMINAL_TRUCK_STATUS_IDLE = "idle"              # Available for use
TERMINAL_TRUCK_STATUS_BUSY = "busy"              # Currently performing a task

VALID_TERMINAL_TRUCK_STATUSES = {
    TERMINAL_TRUCK_STATUS_IDLE,
    TERMINAL_TRUCK_STATUS_BUSY
}

# ID generation constants
TERMINAL_TRUCK_ID_PREFIX = "TTR"

# Container type restrictions
TERMINAL_TRUCK_ALLOWED_TYPES = {"Trailer", "Swap Body"}  # Only these types allowed

# Task time limits (seconds)
TERMINAL_TRUCK_MIN_TASK_TIME = 0.0
TERMINAL_TRUCK_MAX_TASK_TIME = 300.0  # 5 minutes max for internal moves


class TerminalTruck(Truck):
    """
    Represents a terminal-owned truck for moving containers within the terminal.
    
    These trucks are specialized for moving swap bodies and trailers between
    storage positions, freeing up valuable specialized storage areas.
    
    Inherits from Truck but adds terminal-specific functionality and constraints.
    """
    
    def __init__(self,
                 truck_id: Optional[str] = None,
                 max_length: float = TERMINAL_TRUCK_MAX_LENGTH,
                 arrival_time: Optional[datetime] = None):
        """
        Initialize a new terminal truck.
        
        Args:
            truck_id: Unique identifier (auto-generated if None)
            max_length: Maximum container length capacity in meters
            arrival_time: When the truck enters service
        """
        
        # Initialize parent class
        super().__init__(
            truck_id=truck_id,
            max_length=max_length,
            arrival_time=arrival_time,
            parking_spot=None,  # Terminal trucks don't use regular parking
            prefix=TERMINAL_TRUCK_ID_PREFIX
        )
        
        # Override status to use terminal truck statuses
        self.status = TERMINAL_TRUCK_STATUS_IDLE
        
        # Terminal truck specific attributes
        self.current_source: Optional[str] = None
        self.current_destination: Optional[str] = None
        self.task_start_time: Optional[datetime] = None
        self.task_completion_time: Optional[datetime] = None
        
        # Terminal trucks are always for internal moves
        self.is_pickup_truck = False
        self.is_delivery_truck = False
        self.is_terminal_truck = True
    
    def add_container(self, container: Container) -> bool:
        """
        Add a container to the terminal truck.
        Only swap bodies and trailers are allowed, and only one at a time.
        
        Args:
            container: Container object to add
            
        Returns:
            True if container was added successfully, False otherwise
        """
        assert container is not None, "TerminalTruck.add_container requires a Container"
        assert isinstance(container, Container), "TerminalTruck.add_container expects a Container"

        # Only allow specific container types
        if container.container_type not in TERMINAL_TRUCK_ALLOWED_TYPES:
            return False
        
        # Don't allow multiple containers (terminal trucks carry one at a time)
        if self.containers:
            return False
        
        # Use parent's add_container which already handles exclusive types
        return super().add_container(container)
    
    def assign_task(self, 
                   source: str, 
                   destination: str, 
                   task_time: float,
                   current_time: datetime) -> None:
        """
        Assign a transport task to the terminal truck.
        
        Args:
            source: Source position identifier
            destination: Destination position identifier
            task_time: Time in seconds the task will take
            current_time: Current simulation time
            
        Raises:
            ValueError: If parameters are invalid
        """
        if not source:
            raise ValueError("Source position must be provided")
        if not destination:
            raise ValueError("Destination position must be provided")
        if not current_time:
            raise ValueError("Current time must be provided")
        
        # If not given set to max
        if not task_time:
            task_time = TERMINAL_TRUCK_MAX_TASK_TIME
        if task_time < TERMINAL_TRUCK_MIN_TASK_TIME:
            raise ValueError(f"Task time must be non-negative, got {task_time}")
        if task_time > TERMINAL_TRUCK_MAX_TASK_TIME:
            raise ValueError(
                f"Task time exceeds maximum of {TERMINAL_TRUCK_MAX_TASK_TIME} seconds"
            )
        
        self.status = TERMINAL_TRUCK_STATUS_BUSY
        self.current_source = source
        self.current_destination = destination
        self.task_start_time = current_time
        self.task_completion_time = current_time + timedelta(seconds=task_time)
    
    def complete_task(self, current_time: datetime) -> None:
        """
        Mark the current task as completed.
        
        Args:
            current_time: Current simulation time
            
        Raises:
            ValueError: If current_time is not provided
        """
        if not current_time:
            raise ValueError("Current time must be provided")
        
        # Store completion time for stats
        if not self.loading_complete_time:
            self.loading_complete_time = current_time
        
        # Reset task-specific attributes
        self.status = TERMINAL_TRUCK_STATUS_IDLE
        self.current_source = None
        self.current_destination = None
        self.task_start_time = None
        self.task_completion_time = None
        self.containers = []  # Empty the truck
    
    def is_available(self) -> bool:
        """
        Check if the terminal truck is available for a new task.
        
        Returns:
            True if truck is idle and empty
        """
        return (
            self.status == TERMINAL_TRUCK_STATUS_IDLE and 
            len(self.containers) == 0
        )
    
    def is_busy(self) -> bool:
        """
        Check if the terminal truck is currently busy.
        
        Returns:
            True if truck is performing a task
        """
        return self.status == TERMINAL_TRUCK_STATUS_BUSY
    
    def __str__(self) -> str:
        """String representation of the terminal truck."""
        if not self.containers:
            container_str = "Empty"
        else:
            container_ids = ", ".join(c.container_id for c in self.containers)
            container_str = f"Carrying: {container_ids}"
        
        status_str = f"status: {self.status}"
        
        if self.status == TERMINAL_TRUCK_STATUS_BUSY and self.current_source:
            status_str += f" ({self.current_source} → {self.current_destination})"
        
        return f"Terminal Truck {self.truck_id}: {container_str}, {status_str}"
    
    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return (
            f"TerminalTruck(id={self.truck_id}, "
            f"containers={len(self.containers)}, "
            f"status={self.status})"
        )