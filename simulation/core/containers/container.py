# simulation/core/containers/container.py
from dataclasses import dataclass
from datetime import datetime
from simulation.core.enums import Direction, GoodsType

# Type definitions now use enums
# Direction and GoodsType moved to enums.py


@dataclass
class Container:
    """
    Pure data class representing a shipping container.
    All dimensions and properties are set by the factory based on container type.
    """
    # Required fields
    container_id: str
    direction: Direction
    container_type: str  # Dynamic based on CSV data
    arrival_date: datetime
    departure_date: datetime
    
    # Goods type
    goods_type: GoodsType = GoodsType.REGULAR
    
    # Physical properties - set by factory
    length_ft: int = 40
    length_m: float = 12.2
    width_m: float = 2.44
    height_m: float = 2.44
    is_high_cube: bool = False
    is_swap_body: bool = False
    is_trailer: bool = False
    
    @property
    def length(self) -> float:
        """Compatibility property for existing code."""
        return self.length_m
    
    @property
    def width(self) -> float:
        """Compatibility property for existing code."""
        return self.width_m
    
    @property
    def height(self) -> float:
        """Compatibility property for existing code."""
        return self.height_m
    
    @property
    def is_special_type(self) -> bool:
        """Check if this is a special container type (Trailer/Swap Body)."""
        return self.is_trailer or self.is_swap_body
    
    def days_in_terminal(self, current_date: datetime) -> int:
        """
        Calculate days since arrival.
        
        Args:
            current_date: Current simulation date
            
        Returns:
            Number of days in terminal (non-negative)
        """
        if self.arrival_date and current_date:
            return max(0, (current_date - self.arrival_date).days)
        return 0
    
    def days_until_departure(self, current_date: datetime) -> float:
        """
        Calculate days until scheduled departure.
        
        Args:
            current_date: Current simulation date
            
        Returns:
            Days until departure (or infinity if no departure date)
        """
        if self.departure_date and current_date:
            return max(0.0, (self.departure_date - current_date).days)
        return float('inf')