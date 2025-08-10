# simulation/terminal_components/Truck.py

from datetime import datetime, timedelta
import random
from typing import List, Dict, Optional, Set
from simulation.terminal_components.storage_units.Container import Container

# ==================== TRUCK CONSTANTS ====================

# Physical dimensions (meters)
TRUCK_MAX_LENGTH_STANDARD = 24.4  # Standard trailer length in meters (80 foot)
TRUCK_MIN_CONTAINER_LENGTH = 6.1  # Minimum space needed for smallest container

# Truck statuses
TRUCK_STATUS_ARRIVING = "arriving"     # Scheduled to arrive but not yet at terminal
TRUCK_STATUS_WAITING = "waiting"       # At terminal waiting to be processed
TRUCK_STATUS_LOADING = "loading"       # Currently being loaded/unloaded
TRUCK_STATUS_DEPARTING = "departing"   # Ready to leave
TRUCK_STATUS_DEPARTED = "departed"     # Has left the terminal

VALID_TRUCK_STATUSES = {
    TRUCK_STATUS_ARRIVING,
    TRUCK_STATUS_WAITING,
    TRUCK_STATUS_LOADING,
    TRUCK_STATUS_DEPARTING,
    TRUCK_STATUS_DEPARTED
}

# Special container types that require exclusive truck usage
EXCLUSIVE_CONTAINER_TYPES = {"Trailer", "Swap Body"}

# ID generation bounds
TRUCK_ID_MIN_RANDOM = 10000
TRUCK_ID_MAX_RANDOM = 99999
TRUCK_ID_PREFIX = "TRK"


class Truck:
    """
    Represents a truck that can transport containers to/from the terminal.
    
    Attributes:
        truck_id: Unique identifier for the truck
        max_length: Maximum length of containers the truck can carry (meters)
        containers: List of containers currently loaded on the truck
        arrival_time: When the truck arrived at the terminal
        departure_time: When the truck departed from the terminal
        parking_spot: Current parking spot position
        status: Current status of the truck
        pickup_container_ids: IDs of containers to pick up
    """
    
    def __init__(self, 
                 truck_id: Optional[str] = None,
                 max_length: float = TRUCK_MAX_LENGTH_STANDARD,
                 containers: Optional[List[Container]] = None,
                 arrival_time: Optional[datetime] = None,
                 parking_spot: Optional[str] = None):
        """
        Initialize a new truck.
        
        Args:
            truck_id: Unique identifier (auto-generated if None)
            max_length: Maximum container length capacity in meters
            containers: Initial containers on the truck
            arrival_time: When the truck arrives at terminal
            parking_spot: Assigned parking spot
        """
        # Validate max_length
        if max_length <= 0:
            raise ValueError(f"max_length must be positive, got {max_length}")
        
        # Core attributes
        self.truck_id = truck_id or self._generate_truck_id()
        self.max_length = max_length
        self.parking_spot = parking_spot
        
        # Container management
        self.containers: List[Container] = []
        if containers:
            if isinstance(containers, list):
                self.containers = containers
            else:
                self.containers = [containers]
        
        # Time tracking - no datetime.now() usage
        self.arrival_time = arrival_time
        self.departure_time: Optional[datetime] = None
        self.loading_start_time: Optional[datetime] = None
        self.loading_complete_time: Optional[datetime] = None
        
        # Status
        self.status = TRUCK_STATUS_ARRIVING
        
        # Purpose tracking
        self.pickup_container_ids: Set[str] = set()
        self.is_pickup_truck = len(self.pickup_container_ids) > 0
        self.is_delivery_truck = len(self.containers) > 0
    
    @staticmethod
    def _generate_truck_id() -> str:
        """Generate a unique truck ID."""
        random_num = random.randint(TRUCK_ID_MIN_RANDOM, TRUCK_ID_MAX_RANDOM)
        return f"{TRUCK_ID_PREFIX}{random_num}"
    
    def add_container(self, container: Container) -> bool:
        """
        Add a container to the truck if there's enough space.
        
        Args:
            container: Container object to add
            
        Returns:
            True if container was added successfully, False otherwise
        """
        # Validate container
        if not container:
            return False
        
        # Special handling for exclusive container types
        if container.container_type in EXCLUSIVE_CONTAINER_TYPES:
            if self.containers:
                return False  # Truck must be empty for exclusive types
            self.containers.append(container)
            return True
        
        # Check if truck already has exclusive container
        if self.containers and any(
            c.container_type in EXCLUSIVE_CONTAINER_TYPES for c in self.containers
        ):
            return False
        
        # Check length constraint
        current_length = sum(c.length for c in self.containers)
        if current_length + container.length > self.max_length:
            return False
        
        self.containers.append(container)
        return True
    
    def remove_container(self, container_id: Optional[str] = None) -> Optional[Container]:
        """
        Remove a container from the truck.
        
        Args:
            container_id: ID of container to remove (removes first if None)
            
        Returns:
            The removed container, or None if not found
        """
        if not self.containers:
            return None
        
        if container_id is None:
            return self.containers.pop(0)
        
        for i, container in enumerate(self.containers):
            if container.container_id == container_id:
                return self.containers.pop(i)
        
        return None
    
    def start_loading(self, current_time: datetime) -> None:
        """
        Mark the truck as being loaded.
        
        Args:
            current_time: Current simulation time
        """
        if not current_time:
            raise ValueError("current_time must be provided")
        
        self.loading_start_time = current_time
        self.status = TRUCK_STATUS_LOADING
    
    def complete_loading(self, current_time: datetime) -> None:
        """
        Mark the truck as finished loading.
        
        Args:
            current_time: Current simulation time
        """
        if not current_time:
            raise ValueError("current_time must be provided")
        
        self.loading_complete_time = current_time
        self.status = TRUCK_STATUS_DEPARTING
    
    def depart(self, current_time: datetime) -> None:
        """
        Mark the truck as departed.
        
        Args:
            current_time: Current simulation time
        """
        if not current_time:
            raise ValueError("current_time must be provided")
        
        self.departure_time = current_time
        self.status = TRUCK_STATUS_DEPARTED
    
    def add_pickup_container_id(self, container_id: str) -> None:
        """Add a container ID to be picked up."""
        if not container_id:
            raise ValueError("container_id cannot be empty")
        
        self.pickup_container_ids.add(container_id)
        self.is_pickup_truck = True
    
    def remove_pickup_container_id(self, container_id: str) -> bool:
        """
        Remove a container ID from the pickup list.
        
        Returns:
            True if removed, False if not in list
        """
        if container_id in self.pickup_container_ids:
            self.pickup_container_ids.remove(container_id)
            return True
        return False
    
    def get_available_length(self) -> float:
        """Get the remaining available length on the truck."""
        current_length = sum(c.length for c in self.containers)
        return max(0.0, self.max_length - current_length)
    
    def has_containers(self) -> bool:
        """Check if the truck has any containers."""
        return len(self.containers) > 0
    
    def is_full(self) -> bool:
        """
        Check if the truck is effectively full.
        
        Returns:
            True if less than minimum container space remains
        """
        return self.get_available_length() < TRUCK_MIN_CONTAINER_LENGTH
    
    def is_ready_to_depart(self) -> bool:
        """
        Check if the truck is ready to depart.
        
        Returns:
            True if all pickup/delivery operations are complete
        """
        if self.is_pickup_truck:
            return len(self.pickup_container_ids) == 0
        else:
            return len(self.containers) == 0
    
    def can_accommodate_container(self, container: Container) -> bool:
        """
        Check if truck can accommodate a specific container.
        
        Args:
            container: Container to check
            
        Returns:
            True if container can be added
        """
        if not container:
            return False
        
        # Check exclusive container constraints
        if container.container_type in EXCLUSIVE_CONTAINER_TYPES:
            return len(self.containers) == 0
        
        # Check if truck has exclusive container
        if self.containers and any(
            c.container_type in EXCLUSIVE_CONTAINER_TYPES for c in self.containers
        ):
            return False
        
        # Check length
        current_length = sum(c.length for c in self.containers)
        return current_length + container.length <= self.max_length
    
    def get_stats(self) -> Dict[str, any]:
        """
        Get comprehensive statistics about the truck.
        
        Returns:
            Dictionary containing truck statistics
        """
        stats = {
            'parking_spot': self.parking_spot,
            'arrival_time': self.arrival_time,
            'loading_start_time': self.loading_start_time,
            'loading_complete_time': self.loading_complete_time,
            'departure_time': self.departure_time
        }

        return stats
    
    def __str__(self) -> str:
        """String representation of the truck."""
        if not self.containers:
            status_str = "Empty"
        elif len(self.containers) == 1:
            status_str = f"Carrying {self.containers[0].container_id}"
        else:
            status_str = f"Carrying {len(self.containers)} containers"
        
        return f"Truck {self.truck_id}: {status_str}, status: {self.status}"
    
    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return (f"Truck(id={self.truck_id}, containers={len(self.containers)}, "
                f"status={self.status}, parking={self.parking_spot})")