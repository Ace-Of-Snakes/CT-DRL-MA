
from datetime import datetime, timedelta
import random
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from collections import OrderedDict
from simulation.terminal_components.storage_units.Container import Container
from simulation.terminal_components.storage_units.Wagon import Wagon

# ==================== TRAIN CONSTANTS ====================
DEFAULT_NUM_WAGONS = 10
DEFAULT_WAGON_LENGTH = 24.4
TRAIN_ID_PREFIX = "TRN"
TRAIN_ID_MIN_RANDOM = 10000
TRAIN_ID_MAX_RANDOM = 99999
WAGON_ID_SEPARATOR = "_W"

# Train statuses
TRAIN_STATUS_ARRIVING = "arriving"
TRAIN_STATUS_WAITING = "waiting"
TRAIN_STATUS_LOADING = "loading"
TRAIN_STATUS_DEPARTING = "departing"
TRAIN_STATUS_DEPARTED = "departed"


@dataclass
class ContainerLocation:
    """Tracks where a container is located."""
    wagon_id: str
    wagon_index: int
    position_in_wagon: int

class Train:
    """
    Optimized train with O(1) container operations.
    Uses hash maps for direct container lookup across all wagons.
    """ 
    def __init__(self,
                 train_id: Optional[str] = None,
                 num_wagons: int = DEFAULT_NUM_WAGONS,
                 wagon_length: float = DEFAULT_WAGON_LENGTH,
                 arrival_time: Optional[datetime] = None,
                 departure_time: Optional[datetime] = None,
                 rail_track: Optional[str] = None,
                 operator: Optional[str] = None,
                 destination: Optional[str] = None,
                 scheduled_arrival: Optional[datetime] = None,
                 scheduled_departure: Optional[datetime] = None):
        
        if num_wagons <= 0:
            raise ValueError(f"Number of wagons must be positive, got {num_wagons}")
        if wagon_length <= 0:
            raise ValueError(f"Wagon length must be positive, got {wagon_length}")
        
        self.train_id = train_id or self._generate_train_id()
        
        # Create wagons
        self.wagons: List[Wagon] = [
            Wagon(f"{self.train_id}{WAGON_ID_SEPARATOR}{i+1}", wagon_length)
            for i in range(num_wagons)
        ]
        
        # O(1) lookup structures
        self.container_locations: Dict[str, ContainerLocation] = {}  # container_id -> location
        self.wagon_by_id: Dict[str, Wagon] = {w.wagon_id: w for w in self.wagons}
        
        # Track wagons with space for quick placement
        self.wagons_with_space: Set[int] = set(range(num_wagons))  # wagon indices
        self.empty_wagons: Set[int] = set(range(num_wagons))  # for exclusive containers
        
        # Timing and status
        self.arrival_time = arrival_time
        self.departure_time = departure_time
        self.realised_departure_time: Optional[datetime] = None
        self.loading_start_time: Optional[datetime] = None
        self.loading_complete_time: Optional[datetime] = None
        self.rail_track = rail_track
        self.status = TRAIN_STATUS_ARRIVING
        
        # Cache for performance
        self._total_containers = 0
        self._total_pickup_ids = 0

        # Additional attributes
        self.operator = operator
        self.destination = destination
        self.scheduled_arrival = scheduled_arrival
        self.scheduled_departure = scheduled_departure
    
    @staticmethod
    def _generate_train_id() -> str:
        """Generate unique train ID."""
        return f"{TRAIN_ID_PREFIX}{random.randint(TRAIN_ID_MIN_RANDOM, TRAIN_ID_MAX_RANDOM)}"
    
    def find_container(self, container_id: str) -> Tuple[Optional[Wagon], Optional[int]]:
        """Find container - O(1) operation."""
        location = self.container_locations.get(container_id)
        if not location:
            return None, None
        
        wagon = self.wagons[location.wagon_index]
        return wagon, location.position_in_wagon
    
    def add_container(self, container: Container, wagon_index: Optional[int] = None) -> bool:
        """Add container - O(1) for specified wagon, O(w) worst case for auto-placement."""
        if not container or container.container_id in self.container_locations:
            return False
        
        # Specific wagon - O(1)
        if wagon_index is not None:
            if 0 <= wagon_index < len(self.wagons):
                if self._add_to_wagon(wagon_index, container):
                    return True
            return False
        
        # Auto-placement
        # if container.container_type in EXCLUSIVE_CONTAINER_TYPES:
        #     # Try empty wagons only - O(1) average
        #     for idx in self.empty_wagons:
        #         if self._add_to_wagon(idx, container):
        #             return True
        # else:
        # Try wagons with space - O(1) average
        for idx in self.wagons_with_space:
            # wagon = self.wagons[idx]
            if self._add_to_wagon(idx, container):
                return True
        
        return False
    
    def _add_to_wagon(self, wagon_index: int, container: Container) -> bool:
        """Internal method to add container to specific wagon and update indexes."""
        wagon = self.wagons[wagon_index]
        position = len(wagon.containers)
        
        if wagon.add_container(container):
            # Update location index
            self.container_locations[container.container_id] = ContainerLocation(
                wagon_id=wagon.wagon_id,
                wagon_index=wagon_index,
                position_in_wagon=position
            )
            
            # Update space tracking
            if wagon.is_full():
                self.wagons_with_space.discard(wagon_index)
            if not wagon.is_empty():
                self.empty_wagons.discard(wagon_index)
            
            self._total_containers += 1
            return True
        return False
    
    def remove_container(self, container_id: str) -> Optional[Container]:
        """Remove container - O(1) operation."""
        location = self.container_locations.get(container_id)
        if not location:
            return None
        
        wagon = self.wagons[location.wagon_index]
        container = wagon.remove_container(container_id)
        
        if container:
            # Update indexes
            del self.container_locations[container_id]
            
            # Update space tracking
            if not wagon.is_full():
                self.wagons_with_space.add(location.wagon_index)
            if wagon.is_empty():
                self.empty_wagons.add(location.wagon_index)
            
            self._total_containers -= 1
        
        return container
    
    def add_pickup_container(self, container_id: str, wagon_index: Optional[int] = None) -> bool:
        """Add pickup container - O(1) operation."""
        if not container_id:
            return False
        
        target_wagon_index = wagon_index if wagon_index is not None else 0
        
        if 0 <= target_wagon_index < len(self.wagons):
            self.wagons[target_wagon_index].add_pickup_container(container_id)
            self._total_pickup_ids += 1
            return True
        return False
    
    def remove_pickup_container(self, container_id: str) -> bool:
        """Remove pickup container - O(w) where w is number of wagons."""
        if not container_id:
            return False
        
        for wagon in self.wagons:
            if container_id in wagon.pickup_container_ids:
                wagon.remove_pickup_container(container_id)
                self._total_pickup_ids -= 1
                return True
        return False
    
    def has_container(self, container_id: str) -> bool:
        """Check if train has container - O(1) operation."""
        return container_id in self.container_locations
    
    def get_all_containers(self) -> List[Container]:
        """Get all containers - O(n) where n is total containers."""
        containers = []
        for wagon in self.wagons:
            containers.extend(wagon.get_container_list())
        return containers
    
    def get_all_container_ids(self) -> List[str]:
        """Get all container IDs - O(1) operation."""
        return list(self.container_locations.keys())
    
    def get_container_count(self) -> int:
        """Get container count - O(1) operation."""
        return self._total_containers
    
    def get_all_pickup_container_ids(self) -> Set[str]:
        """Get all pickup IDs - O(w) where w is number of wagons."""
        pickup_ids = set()
        for wagon in self.wagons:
            pickup_ids.update(wagon.pickup_container_ids)
        return pickup_ids
    
    def is_fully_loaded(self) -> bool:
        """Check if fully loaded - O(1) operation."""
        return self._total_pickup_ids == 0
    
    def has_space_for_container(self, container: Container) -> bool:
        """Check space availability - O(1) average case."""
        if not container:
            return False
        
        # if container.container_type in EXCLUSIVE_CONTAINER_TYPES:
        #     return len(self.empty_wagons) > 0
        # else:
            # Check if any wagon with space doesn't have exclusive container
        # for idx in self.wagons_with_space:
        #     if not self.wagons[idx].has_exclusive_container():
        #         return True
        return len(self.wagons_with_space) > 0
    
    # Keep existing time/status methods unchanged
    def start_loading(self, current_time: datetime) -> None:
        if not current_time:
            raise ValueError("current_time must be provided")
        self.loading_start_time = current_time
        self.status = TRAIN_STATUS_LOADING
    
    def complete_loading(self, current_time: datetime) -> None:
        if not current_time:
            raise ValueError("current_time must be provided")
        self.loading_complete_time = current_time
        self.status = TRAIN_STATUS_DEPARTING
    
    def depart(self, current_time: datetime) -> None:
        if not current_time:
            raise ValueError("current_time must be provided")
        self.realised_departure_time = current_time
        self.status = TRAIN_STATUS_DEPARTED
    
    def get_stats(self) -> Dict[str, any]:
        """Get train statistics - mostly O(1) operations."""
        total_capacity = sum(w.length for w in self.wagons)
        used_capacity = sum(w._used_length for w in self.wagons)
        
        return {
            'train_id': self.train_id,
            'num_wagons': len(self.wagons),
            'total_containers': self._total_containers,  # O(1)
            'pickup_containers': self._total_pickup_ids,  # O(1)
            'total_capacity': total_capacity,
            'used_capacity': used_capacity,
            'utilization_rate': used_capacity / total_capacity if total_capacity > 0 else 0,
            'status': self.status,
            'rail_track': self.rail_track,
            'departed_on_time': True if self.realised_departure_time is not None and self.departure_time is not None and self.realised_departure_time < self.departure_time else False
        }