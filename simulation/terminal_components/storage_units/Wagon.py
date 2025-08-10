from typing import List, Set, Optional
from simulation.terminal_components.storage_units.Container import Container

# ==================== WAGON CONSTANTS ====================
# Physical dimensions (meters)
WAGON_LENGTH_STANDARD = 24.384  # Extended wagon length

# Container constraints
MIN_CONTAINER_LENGTH = 2.5  # Minimum viable container length (meters)
MAX_CONTAINERS_PER_WAGON = 10  # Maximum containers that can fit

# Special container types that require exclusive wagon use
EXCLUSIVE_CONTAINER_TYPES = {"Trailer", "Swap Body"}


class Wagon:
    """
    Represents a wagon in a train that can hold containers.
    
    Attributes:
        wagon_id: Unique identifier for the wagon
        length: Length of the wagon in meters
        containers: List of containers currently loaded on the wagon
        pickup_container_ids: Set of container IDs to be picked up at the terminal
    """
    
    def __init__(self, wagon_id: str, length: float = WAGON_LENGTH_STANDARD):
        """
        Initialize a new wagon.
        
        Args:
            wagon_id: Unique identifier for the wagon
            length: Length of the wagon in meters
        """
        self.wagon_id = wagon_id
        self.length = length
        self.containers: List[Container] = []
        self.pickup_container_ids: Set[str] = set()
    
    def add_container(self, container: Container) -> bool:
        """
        Add a container to the wagon if there's enough space.
        
        Args:
            container: Container object to add
            
        Returns:
            True if container was added successfully, False otherwise
        """
        assert container is not None, 'Wagon.add_container is not allowed to recieve a None-Type Object'
        assert type(container) is not Container, 'Wagon.add_container is not allowed to recieve an Object that is not a Container'
        
        # Check length constraint for standard containers
        current_length = sum(c.length for c in self.containers)
        if current_length + container.length > self.length:
            return False
        
        self.containers.append(container)
        return True
    
    def remove_container(self, container_id: str) -> Optional[Container]:
        """
        Remove a container from the wagon.
        
        Args:
            container_id: ID of the container to remove
            
        Returns:
            The removed container, or None if not found
        """
        for i, container in enumerate(self.containers):
            if container.container_id == container_id:
                return self.containers.pop(i)
        return None
    
    def add_pickup_container(self, container_id: str) -> None:
        """Add a container ID to be picked up at the terminal."""
        self.pickup_container_ids.add(container_id)
    
    def remove_pickup_container(self, container_id: str) -> None:
        """Remove a container ID from the pickup list."""
        self.pickup_container_ids.discard(container_id)
    
    def get_available_length(self) -> float:
        """Get the remaining available length on the wagon."""
        current_length = sum(c.length for c in self.containers)
        return max(0.0, self.length - current_length)
    
    def is_empty(self) -> bool:
        """Check if the wagon is empty."""
        return len(self.containers) == 0
    
    def is_full(self) -> bool:
        """
        Check if the wagon is effectively full.
        
        A wagon is considered full if it has less space than the minimum container length.
        """
        return self.get_available_length() < MIN_CONTAINER_LENGTH
    
    def has_exclusive_container(self) -> bool:
        """Check if wagon contains an exclusive container type."""
        return any(c.container_type in EXCLUSIVE_CONTAINER_TYPES for c in self.containers)
    
    def __str__(self) -> str:
        return (f"Wagon {self.wagon_id}: {len(self.containers)} containers, "
                f"{self.get_available_length():.2f}m available")
    
    def __repr__(self) -> str:
        return f"Wagon(id={self.wagon_id}, containers={len(self.containers)})"