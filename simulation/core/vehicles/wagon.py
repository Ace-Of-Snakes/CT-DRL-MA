from datetime import datetime, timedelta
import random
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from collections import OrderedDict
from simulation.core.containers.container import Container

# ==================== WAGON CONSTANTS ====================
WAGON_LENGTH_STANDARD = 24.4
MIN_CONTAINER_LENGTH = 6
MAX_CONTAINERS_PER_WAGON = 10

class Wagon:
    """
    Optimized wagon with O(1) container operations.
    Uses OrderedDict to maintain insertion order while having O(1) lookups.
    """
    
    def __init__(self, wagon_id: str, length: float = WAGON_LENGTH_STANDARD):
        self.wagon_id = wagon_id
        self.length = length
        self.containers: OrderedDict[str, Container] = OrderedDict()  # O(1) lookup by ID
        self.pickup_container_ids: Set[str] = set()  # O(1) membership tests
        self._used_length: float = 0.0  # Cache for performance
    
    def add_container(self, container: Container) -> bool:
        """Add container - O(1) operation."""
        if not container or container.container_id in self.containers:
            return False
        
        # Check length
        if self._used_length + container.length > self.length:
            return False
        
        self.containers[container.container_id] = container
        self._used_length += container.length
        return True
    
    def remove_container(self, container_id: str) -> Optional[Container]:
        """Remove container - O(1) operation."""
        if container_id not in self.containers:
            return None
        
        container = self.containers.pop(container_id)
        self._used_length -= container.length
        return container
    
    def has_container(self, container_id: str) -> bool:
        """Check if wagon has container - O(1) operation."""
        return container_id in self.containers
    
    def add_pickup_container(self, container_id: str) -> None:
        """Add pickup ID - O(1) operation."""
        self.pickup_container_ids.add(container_id)
    
    def remove_pickup_container(self, container_id: str) -> None:
        """Remove pickup ID - O(1) operation."""
        self.pickup_container_ids.discard(container_id)
    
    def get_available_length(self) -> float:
        """Get available space - O(1) operation."""
        return max(0.0, self.length - self._used_length)
    
    def is_empty(self) -> bool:
        """Check if empty - O(1) operation."""
        return len(self.containers) == 0
    
    def is_full(self) -> bool:
        """Check if full - O(1) operation."""
        return self.get_available_length() < MIN_CONTAINER_LENGTH
    
    def get_container_list(self) -> List[Container]:
        """Get containers as list - O(n) but only when needed."""
        return list(self.containers.values())
    
    def __str__(self) -> str:
        return (f"Wagon {self.wagon_id}: {len(self.containers)} containers, "
                f"{self.get_available_length():.2f}m available")
    
    def __repr__(self) -> str:
        return f"Wagon(id={self.wagon_id}, containers={len(self.containers)})"