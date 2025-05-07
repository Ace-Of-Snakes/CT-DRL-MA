from datetime import datetime, timedelta
import random
import uuid
from typing import List, Dict, Optional, Tuple, Set
from terminal_components.Container import Container

class Truck:
    """
    Represents a truck that can transport containers to/from the terminal.
    
    Attributes:
        truck_id (str): Unique identifier for the truck
        max_length (float): Maximum length of containers the truck can carry
        max_weight (float): Maximum weight capacity of the truck in kg
        containers (list): List of containers currently loaded on the truck
        arrival_time (datetime): When the truck arrived at the terminal
        departure_time (datetime): When the truck departed from the terminal
        parking_spot (str): Current parking spot position (e.g., 'p_1', 'p_2')
        status (str): Current status of the truck
    """
    
    # Possible truck statuses
    ARRIVING = "arriving"     # Truck is scheduled to arrive but not yet at terminal
    WAITING = "waiting"       # Truck is at terminal waiting to be processed
    LOADING = "loading"       # Truck is currently being loaded/unloaded
    DEPARTING = "departing"   # Truck is ready to leave
    DEPARTED = "departed"     # Truck has left the terminal
    
    def __init__(self, 
                 truck_id: str = None, 
                 max_length: float = 13.6,  # Standard trailer length
                 max_weight: float = 40000, # Standard max weight
                 containers = None,
                 arrival_time: datetime = None,
                 parking_spot: str = None):
        """
        Initialize a new truck.
        
        Args:
            truck_id: Unique identifier for the truck (auto-generated if None)
            max_length: Maximum length of containers the truck can carry
            max_weight: Maximum weight capacity of the truck in kg
            containers: List of Container objects on the truck (empty list if arriving empty)
            arrival_time: When the truck arrives at the terminal
            parking_spot: Assigned parking spot (e.g., 'p_1', 'p_2')
        """
        self.truck_id = truck_id or f"TRK{random.randint(10000, 99999)}"
        self.max_length = max_length
        self.max_weight = max_weight
        self.containers: List[Container] = [] if containers is None else containers if isinstance(containers, list) else [containers]
        self.arrival_time = arrival_time or datetime.now()
        self.departure_time = None
        self.parking_spot = parking_spot
        self.status = self.ARRIVING
        
        # Tracking variables
        self.loading_start_time = None
        self.loading_complete_time = None
        
        # Destination and purpose
        self.pickup_container_ids = set()  # IDs of containers to pick up (if any)
        self.is_pickup_truck = len(self.containers) == 0  # True if truck arrives empty (to pick up)
        self.is_delivery_truck = len(self.containers) > 0  # True if truck arrives with container(s) (to deliver)
    
    def add_container(self, container: Container) -> bool:
        """
        Add a container to the truck if there's enough space.
        
        Args:
            container: Container object to add
            
        Returns:
            bool: True if container was added successfully, False otherwise
        """
        # Special case for trailers and swap bodies - they occupy the entire truck
        if container.container_type in ["Trailer", "Swap Body"]:
            if len(self.containers) > 0:
                return False  # Truck must be empty for trailer or swap body
            
            current_weight = sum(c.weight for c in self.containers)
            if current_weight + container.weight > self.max_weight:
                return False  # Exceeds weight limit
            
            self.containers.append(container)
            return True
        
        # For standard containers, check if there's enough space
        current_length = sum(c.length for c in self.containers)
        if current_length + container.length > self.max_length:
            return False  # Exceeds length limit
        
        current_weight = sum(c.weight for c in self.containers)
        if current_weight + container.weight > self.max_weight:
            return False  # Exceeds weight limit
        
        self.containers.append(container)
        return True
    
    def remove_container(self, container_id: str = None):
        """
        Remove a container from the truck.
        
        Args:
            container_id: ID of the container to remove (removes first container if None)
            
        Returns:
            Container or None: The removed container, or None if not found or truck is empty
        """
        if not self.containers:
            return None
            
        if container_id is None:
            # Remove the first container if no ID specified
            return self.containers.pop(0)
            
        for i, container in enumerate(self.containers):
            if container.container_id == container_id:
                return self.containers.pop(i)
                
        return None
    
    def start_loading(self):
        """Mark the truck as being loaded and record the start time."""
        self.loading_start_time = datetime.now()
        self.status = self.LOADING
    
    def complete_loading(self):
        """Mark the truck as finished loading and record the completion time."""
        self.loading_complete_time = datetime.now()
        self.status = self.DEPARTING
    
    def get_loading_time(self) -> Optional[timedelta]:
        """Get the time taken to load/unload the truck."""
        if self.loading_start_time and self.loading_complete_time:
            return self.loading_complete_time - self.loading_start_time
        return None
    
    def get_total_time(self) -> Optional[timedelta]:
        """Get the total time from arrival to departure."""
        if self.arrival_time and self.departure_time:
            return self.departure_time - self.arrival_time
        return None
    
    def depart(self):
        """Mark the truck as departed and record the departure time."""
        self.status = self.DEPARTED
        self.departure_time = datetime.now()
    
    def add_pickup_container_id(self, container_id: str):
        """Add a container ID to be picked up."""
        self.pickup_container_ids.add(container_id)
        self.is_pickup_truck = True
    
    def remove_pickup_container_id(self, container_id: str):
        """Remove a container ID from the pickup list."""
        if container_id in self.pickup_container_ids:
            self.pickup_container_ids.remove(container_id)
    
    def has_containers(self) -> bool:
        """Check if the truck has any containers."""
        return len(self.containers) > 0
    
    def is_full(self) -> bool:
        """
        Check if the truck is effectively full.
        
        A truck is considered full if it has less than 2.5m of space remaining,
        which isn't enough for even the smallest container.
        """
        return self.get_available_length() < 2.5
    
    def get_available_length(self) -> float:
        """Get the remaining available length on the truck."""
        current_length = sum(c.length for c in self.containers)
        return max(0, self.max_length - current_length)
    
    def get_available_weight(self) -> float:
        """Get the remaining available weight capacity on the truck."""
        current_weight = sum(c.weight for c in self.containers)
        return max(0, self.max_weight - current_weight)
    
    def is_ready_to_depart(self) -> bool:
        """
        Check if the truck is ready to depart.
        
        A pickup truck is ready if it has picked up all required containers.
        A delivery truck is ready if it has delivered all its containers.
        """
        if self.is_pickup_truck:
            # Ready if picked up all required containers
            return len(self.pickup_container_ids) == 0
        else:
            # Ready if delivered all its containers
            return len(self.containers) == 0
    
    def __str__(self):
        if not self.containers:
            status_str = "Empty"
        elif len(self.containers) == 1:
            status_str = f"Carrying {self.containers[0].container_id}"
        else:
            container_ids = ", ".join(c.container_id for c in self.containers)
            status_str = f"Carrying {len(self.containers)} containers: {container_ids}"
        
        return f"Truck {self.truck_id}: {status_str}, status: {self.status}"
    
    def __repr__(self):
        container_count = len(self.containers)
        return f"Truck(id={self.truck_id}, containers={container_count}, status={self.status})"

