from datetime import datetime, timedelta
import random
import uuid
from typing import List, Dict, Optional, Tuple, Set
from Container import Container  # Assuming Container is defined in a separate module

class Wagon:
    """
    Represents a wagon in a train that can hold containers.
    
    Attributes:
        wagon_id (str): Unique identifier for the wagon
        length (float): Length of the wagon in meters (matches rail_slot length)
        max_weight (float): Maximum weight capacity of the wagon in kg
        containers (list): List of containers currently loaded on the wagon
        pickup_container_ids (set): Set of container IDs to be picked up at the terminal
    """
    
    def __init__(self, wagon_id: str, length: float = 20.0, max_weight: float = 60000):
        """
        Initialize a new wagon.
        
        Args:
            wagon_id: Unique identifier for the wagon
            length: Length of the wagon in meters (defaults to standard rail slot length)
            max_weight: Maximum weight capacity in kg
        """
        self.wagon_id = wagon_id
        self.length = length
        self.max_weight = max_weight
        self.containers: List[Container] = []  # List of Container objects currently loaded
        self.pickup_container_ids = set()  # Container IDs to pick up at terminal
    
    def add_container(self, container: Container) -> bool:
        """
        Add a container to the wagon if there's enough space.
        
        Args:
            container: Container object to add
            
        Returns:
            bool: True if container was added successfully, False otherwise
        """
        # Special case for trailers and swap bodies - they occupy an entire wagon
        if container.container_type in ["Trailer", "Swap Body"]:
            if len(self.containers) > 0:
                return False  # Wagon must be empty for trailer or swap body
            
            current_weight = sum(c.weight for c in self.containers)
            if current_weight + container.weight > self.max_weight:
                return False  # Exceeds weight limit
            
            self.containers.append(container)
            return True
        
        # For standard containers, check if there's enough space
        current_length = sum(c.length for c in self.containers)
        if current_length + container.length > self.length:
            return False  # Exceeds length limit
        
        current_weight = sum(c.weight for c in self.containers)
        if current_weight + container.weight > self.max_weight:
            return False  # Exceeds weight limit
        
        self.containers.append(container)
        return True
    
    def remove_container(self, container_id: str):
        """
        Remove a container from the wagon.
        
        Args:
            container_id: ID of the container to remove
            
        Returns:
            Container or None: The removed container, or None if not found
        """
        for i, container in enumerate(self.containers):
            if container.container_id == container_id:
                return self.containers.pop(i)
        return None
    
    def add_pickup_container(self, container_id: str):
        """Add a container ID to be picked up at the terminal."""
        self.pickup_container_ids.add(container_id)
    
    def remove_pickup_container(self, container_id: str):
        """Remove a container ID from the pickup list."""
        if container_id in self.pickup_container_ids:
            self.pickup_container_ids.remove(container_id)
    
    def get_available_length(self) -> float:
        """Get the remaining available length on the wagon."""
        current_length = sum(c.length for c in self.containers)
        return max(0, self.length - current_length)
    
    def get_available_weight(self) -> float:
        """Get the remaining available weight capacity on the wagon."""
        current_weight = sum(c.weight for c in self.containers)
        return max(0, self.max_weight - current_weight)
    
    def is_empty(self) -> bool:
        """Check if the wagon is empty."""
        return len(self.containers) == 0
    
    def is_full(self) -> bool:
        """
        Check if the wagon is effectively full.
        
        A wagon is considered full if it has less than 2.5m of space remaining, 
        which isn't enough for even the smallest container.
        """
        return self.get_available_length() < 2.5
    
    def __str__(self):
        return f"Wagon {self.wagon_id}: {len(self.containers)} containers, {self.get_available_length():.2f}m space remaining"
    
    def __repr__(self):
        return f"Wagon(id={self.wagon_id}, containers={len(self.containers)})"

