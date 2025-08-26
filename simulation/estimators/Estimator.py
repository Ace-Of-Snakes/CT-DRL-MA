from datetime import datetime, timedelta
import numpy as np
from typing import Optional, Protocol
from abc import ABC, abstractmethod
from simulation.terminal_components.storage_units.Container import Container

class EstimatorStrategy(ABC):
    """
    Abstract base class for departure estimation strategies.
    Allows for easy swapping of prediction models.
    """
    
    @abstractmethod
    def estimate_departure(self, 
                          container: Container, 
                          current_date: Optional[datetime] = None) -> datetime:
        """Calculate estimated departure date for a container."""
        pass
