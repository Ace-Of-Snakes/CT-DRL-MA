from datetime import datetime, timedelta
import random
import uuid
from typing import List, Dict, Optional, Tuple, Set
from terminal_components.Container import Container  # Assuming Container is defined in a separate module
from terminal_components.Wagon import Wagon  # Assuming Wagon is defined in a separate module

class Train:
    """
    Represents a train composed of multiple wagons that can transport containers.
    
    Attributes:
        train_id (str): Unique identifier for the train
        wagons (list): List of wagon objects in the train
        arrival_time (datetime): When the train arrived at the terminal
        departure_time (datetime): When the train is scheduled to depart
        rail_track (str): Current rail track position (e.g., 'T1', 'T2')
        status (str): Current status of the train
    """
    
    # Possible train statuses
    ARRIVING = "arriving"  # Train is scheduled to arrive but not yet at terminal
    WAITING = "waiting"    # Train is at terminal waiting to be processed
    LOADING = "loading"    # Train is currently being loaded/unloaded
    DEPARTING = "departing"  # Train is scheduled to leave
    DEPARTED = "departed"  # Train has left the terminal
    
    def __init__(self, 
                 train_id: str = None, 
                 num_wagons: int = 10, 
                 wagon_length: float = 20.0,
                 arrival_time: datetime = None,
                 departure_time: datetime = None,
                 rail_track: str = None):
        """
        Initialize a new train.
        
        Args:
            train_id: Unique identifier for the train (auto-generated if None)
            num_wagons: Number of wagons in the train
            wagon_length: Length of each wagon in meters
            arrival_time: When the train arrives at the terminal
            departure_time: When the train is scheduled to depart
            rail_track: Assigned rail track (e.g., 'T1', 'T2')
        """
        self.train_id = train_id or f"TRN{random.randint(10000, 99999)}"
        self.wagons = [Wagon(f"{self.train_id}_W{i+1}", wagon_length) for i in range(num_wagons)]
        self.arrival_time = arrival_time or datetime.now()
        self.departure_time = departure_time
        self.rail_track = rail_track
        self.status = self.ARRIVING
        
        # Tracking variables
        self.loading_start_time = None
        self.loading_complete_time = None
        
    def get_all_containers(self) -> List[Container]:
        """Get a list of all containers on the train."""
        containers = []
        for wagon in self.wagons:
            containers.extend(wagon.containers)
        return containers
    
    def get_all_container_ids(self) -> List[str]:
        """Get a list of all container IDs on the train."""
        return [container.container_id for container in self.get_all_containers()]
    
    def get_all_pickup_container_ids(self) -> Set[str]:
        """Get a set of all container IDs to be picked up."""
        pickup_ids = set()
        for wagon in self.wagons:
            pickup_ids.update(wagon.pickup_container_ids)
        return pickup_ids
    
    def find_container(self, container_id: str) -> Tuple[Optional[Wagon], Optional[int]]:
        """
        Find a container on the train by its ID.
        
        Args:
            container_id: ID of the container to find
            
        Returns:
            tuple: (wagon, container_index) or (None, None) if not found
        """
        for wagon in self.wagons:
            for i, container in enumerate(wagon.containers):
                if container.container_id == container_id:
                    return wagon, i
        return None, None
    
    def add_container(self, container, wagon_index: int = None) -> bool:
        """
        Add a container to the train, either to a specific wagon or to the first available wagon.
        
        Args:
            container: Container object to add
            wagon_index: Index of the wagon to add to (tries all wagons if None)
            
        Returns:
            bool: True if container was added successfully, False otherwise
        """
        # If wagon index specified, try to add to that wagon
        if wagon_index is not None:
            if 0 <= wagon_index < len(self.wagons):
                return self.wagons[wagon_index].add_container(container)
            return False
        
        # Try to find a wagon that can accommodate the container
        for wagon in self.wagons:
            if wagon.add_container(container):
                return True
        
        return False  # No suitable wagon found
    
    def remove_container(self, container_id: str):
        """
        Remove a container from the train.
        
        Args:
            container_id: ID of the container to remove
            
        Returns:
            Container or None: The removed container, or None if not found
        """
        wagon, _ = self.find_container(container_id)
        if wagon:
            return wagon.remove_container(container_id)
        return None
    
    def add_pickup_container(self, container_id: str, wagon_index: int = None) -> bool:
        """
        Add a container ID to be picked up, either to a specific wagon or to the first available wagon.
        
        Args:
            container_id: ID of the container to be picked up
            wagon_index: Index of the wagon to assign to (assigns to first wagon with space if None)
            
        Returns:
            bool: True if container ID was added successfully, False otherwise
        """
        if wagon_index is not None:
            if 0 <= wagon_index < len(self.wagons):
                self.wagons[wagon_index].add_pickup_container(container_id)
                return True
            return False
        
        # Add to first wagon with available capacity
        # For simplicity, assume each wagon can handle multiple pickups
        if len(self.wagons) > 0:
            self.wagons[0].add_pickup_container(container_id)
            return True
        
        return False
    
    def start_loading(self):
        """Mark the train as being loaded and record the start time."""
        self.loading_start_time = datetime.now()
        self.status = self.LOADING
    
    def complete_loading(self):
        """Mark the train as finished loading and record the completion time."""
        self.loading_complete_time = datetime.now()
        self.status = self.DEPARTING
    
    def get_loading_time(self) -> Optional[timedelta]:
        """Get the time taken to load/unload the train."""
        if self.loading_start_time and self.loading_complete_time:
            return self.loading_complete_time - self.loading_start_time
        return None
    
    def is_fully_loaded(self) -> bool:
        """
        Check if the train is fully loaded based on pickup requirements.
        
        Returns:
            bool: True if all pickup containers have been loaded, False otherwise
        """
        for wagon in self.wagons:
            if len(wagon.pickup_container_ids) > 0:
                return False
        return True
    
    def has_space_for_container(self, container: Container) -> bool:
        """
        Check if any wagon in the train has space for the given container.
        
        Args:
            container: Container object to check
            
        Returns:
            bool: True if there's space for the container, False otherwise
        """
        for wagon in self.wagons:
            # Special case for trailers and swap bodies
            if container.container_type in ["Trailer", "Swap Body"]:
                if wagon.is_empty():
                    return True
            else:
                # For standard containers, check if there's enough space
                if wagon.get_available_length() >= container.length:
                    return True
        return False
    
    def depart(self):
        """Mark the train as departed and record the departure time."""
        self.status = self.DEPARTED
        self.departure_time = datetime.now()
    
    def __str__(self):
        container_count = sum(len(wagon.containers) for wagon in self.wagons)
        return f"Train {self.train_id}: {len(self.wagons)} wagons, {container_count} containers, status: {self.status}"
    
    def __repr__(self):
        return f"Train(id={self.train_id}, wagons={len(self.wagons)}, status={self.status})"