from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Literal

# ==================== CONTAINER CONSTANTS ====================
# Physical dimensions (meters)
CONTAINER_HEIGHT_STANDARD = 2.59
CONTAINER_HEIGHT_HIGH_CUBE = 2.89
CONTAINER_WIDTH_STANDARD = 2.44
CONTAINER_WIDTH_SPECIAL = 2.55  # For trailers and swap bodies

# Container lengths by type (meters)
CONTAINER_LENGTHS = {
    "TWEU": 6.06,      # 20 feet
    "THEU": 9.14,      # 30 feet
    "FEU": 12.19,      # 40 feet
    "FFEU": 13.72,     # 45 feet
    "Swap Body": 7.45, # Typical swap body length
    "Trailer": 12.19   # Default for trailers
}

# Type definitions
ContainerType = Literal["TWEU", "THEU", "FEU", "FFEU", "Trailer", "Swap Body"]
Direction = Literal["Import", "Export"]
GoodsType = Literal["Regular", "Reefer", "Dangerous"]

SPECIAL_CONTAINER_TYPES = {"Trailer", "Swap Body"}


@dataclass
class Container:
    """
    Pure data class representing a shipping container.
    Contains only data fields without business logic.
    """
    # Required fields
    container_id: str
    direction: Direction
    container_type: ContainerType
    arrival_date: datetime
    departure_date: datetime
    
    # Optional fields with defaults
    goods_type: GoodsType = "Regular"
    is_high_cube: bool = False
    
    # Dimensions - calculated in __post_init__ if not provided
    height: Optional[float] = None
    length: Optional[float] = None
    width: Optional[float] = None
    
    # Estimation field - set by estimator
    estimated_departure: Optional[datetime] = None
    
    def __post_init__(self):
        """Set default dimensions if not provided."""
        if self.height is None:
            self.height = (CONTAINER_HEIGHT_HIGH_CUBE if self.is_high_cube 
                          else CONTAINER_HEIGHT_STANDARD)
        
        if self.length is None:
            self.length = CONTAINER_LENGTHS.get(self.container_type, CONTAINER_LENGTHS["FEU"])
        
        if self.width is None:
            self.width = (CONTAINER_WIDTH_SPECIAL if self.container_type in SPECIAL_CONTAINER_TYPES 
                         else CONTAINER_WIDTH_STANDARD)
    
    @property
    def is_special_type(self) -> bool:
        """Check if this is a special container type (Trailer/Swap Body)."""
        return self.container_type in SPECIAL_CONTAINER_TYPES
    
    def days_in_terminal(self, current_date: datetime) -> int:
        """Calculate days since arrival."""
        if self.arrival_date and current_date:
            return max(0, (current_date - self.arrival_date).days)
        return 0
    
    def days_until_departure(self, current_date: datetime) -> float:
        """Calculate days until estimated departure."""
        if self.estimated_departure and current_date:
            return max(0, (self.estimated_departure - current_date).days)
        return float('inf')