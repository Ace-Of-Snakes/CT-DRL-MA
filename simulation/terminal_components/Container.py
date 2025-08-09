from datetime import datetime, timedelta
import numpy as np
import random

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

# Valid container types
VALID_CONTAINER_TYPES = ["TWEU", "THEU", "FEU", "FFEU", "Trailer", "Swap Body"]
SPECIAL_CONTAINER_TYPES = {"Trailer", "Swap Body"}  # Non-standard containers

# Valid directions and goods types
VALID_DIRECTIONS = {"Import", "Export"}
VALID_GOODS_TYPES = {"Regular", "Reefer", "Dangerous"}
VALID_STACK_COMPATIBILITY = {"self", "size", "none"}

# Stacking rules
ONLY_SELF_STACKABLE = {"Reefer", "Dangerous"}  # Goods that can only stack with same type

# Estimation accuracy parameters
MIN_ACCURACY_DAYS = 2           # Days with 100% accuracy
MIN_ACCURACY_BOOST = 7          # Days before MIN accuracy with higher precision
MAX_HOLDING_DAYS = 160          # Maximum container holding time
MIN_ACCURACY_PERCENT = 0.30     # Minimum accuracy at peak uncertainty
MIN_ACC_BOOST_PERCENT = 0.85    # Boost percentage
LATE_ACCURACY_PERCENT = 0.45    # Accuracy for very long stays
PEAK_UNCERTAINTY_DAY = 120      # Day with lowest accuracy

class Container:
    """
    Container class representing shipping containers in a terminal.
    """
    
    def __init__(self, 
                 container_id: str, 
                 direction: str, 
                 container_type: str, 
                 arrival_date: datetime,
                 departure_date: datetime,
                 goods_type: str = "Regular",
                 stack_compatibility: str = "size",
                 is_high_cube: bool | None = False,
                 is_stackable: bool | None = None,
                 height: float | None = None,
                 length: float | None = None,
                 width: float | None = None):
        
        # Validate inputs
        if direction not in VALID_DIRECTIONS:
            raise ValueError(f"Direction must be one of {VALID_DIRECTIONS}")
        if container_type not in VALID_CONTAINER_TYPES:
            raise ValueError(f"Container type must be one of {VALID_CONTAINER_TYPES}")
        if goods_type not in VALID_GOODS_TYPES:
            raise ValueError(f"Goods type must be one of {VALID_GOODS_TYPES}")
        if stack_compatibility not in VALID_STACK_COMPATIBILITY:
            raise ValueError(f"Stack compatibility must be one of {VALID_STACK_COMPATIBILITY}")

        # Basic identification
        self.container_id = container_id
        self.direction = direction
        self.container_type = container_type
        self.goods_type = goods_type
        self.is_high_cube = is_high_cube
        
        # Set dimensions (use provided values or defaults)
        self.height = height if height is not None else (
            CONTAINER_HEIGHT_HIGH_CUBE if is_high_cube else CONTAINER_HEIGHT_STANDARD
        )
        self.length = length if length is not None else CONTAINER_LENGTHS.get(container_type, CONTAINER_LENGTHS["FEU"])
        self.width = width if width is not None else (
            CONTAINER_WIDTH_SPECIAL if container_type in SPECIAL_CONTAINER_TYPES else CONTAINER_WIDTH_STANDARD
        )
        
        # Stackability properties
        self.is_stackable = is_stackable if is_stackable is not None else (
            False if container_type in SPECIAL_CONTAINER_TYPES else True
        )
        # Special goods types can only stack with themselves
        self.stack_compatibility = "self" if goods_type in ONLY_SELF_STACKABLE else (
            "none" if container_type in SPECIAL_CONTAINER_TYPES else stack_compatibility
        )
        
        # Timing
        self.arrival_date = arrival_date
        self.departure_date = departure_date  # True departure date
        self.estimated_departure = None  # Will be calculated
        
        # Calculate initial estimation
        if self.departure_date:
            self.calculate_estimation()
    
    def calculate_estimation(self, current_date: datetime = None) -> datetime:
        """
        Calculate estimated departure date based on true departure and uncertainty model.
        Lightweight function optimized for frequent recalculation.
        """
        
        current_date = current_date if current_date is not None else self.arrival_date

        # Use helper function for days calculation
        days_in_terminal = self.days_in_terminal(current_date)
        total_stay = (self.departure_date - self.arrival_date).days
        remaining_days = max(0, (self.departure_date - current_date).days)
        
        # Calculate accuracy based on stay duration
        accuracy = self._calculate_accuracy(total_stay, remaining_days)
        
        # Generate estimation error
        if accuracy >= 0.99:
            error_days = 0
        else:
            max_error_days = max(7, total_stay * 0.3)
            std_dev = max_error_days * (1.0 - accuracy)
            error_days = np.clip(np.random.normal(0, std_dev), -max_error_days, max_error_days)
        
        # Calculate estimated departure
        self.estimated_departure = self.departure_date + timedelta(days=error_days)
        
        # Ensure estimation is reasonable for containers still in terminal
        if days_in_terminal > 0 and self.estimated_departure < current_date:
            self.estimated_departure = current_date + timedelta(days=1)
        
        return self.estimated_departure
    
    def _calculate_accuracy(self, total_stay: int, remaining_days: int) -> float:
        """Calculate prediction accuracy based on stay duration."""
        # Perfect accuracy for very short stays or imminent departures
        if total_stay <= MIN_ACCURACY_DAYS or remaining_days <= MIN_ACCURACY_DAYS:
            return 1.0
        
        # Boost accuracy in final week
        if remaining_days <= MIN_ACCURACY_BOOST:
            return min(1.0, MIN_ACC_BOOST_PERCENT)
        
        # Very long stays
        if total_stay >= MAX_HOLDING_DAYS:
            return LATE_ACCURACY_PERCENT
        
        # Interpolate with dip at peak uncertainty
        if total_stay <= PEAK_UNCERTAINTY_DAY:
            progress = (total_stay - MIN_ACCURACY_DAYS) / (PEAK_UNCERTAINTY_DAY - MIN_ACCURACY_DAYS)
            return 1.0 - (1.0 - MIN_ACCURACY_PERCENT) * progress
        else:
            progress = (total_stay - PEAK_UNCERTAINTY_DAY) / (MAX_HOLDING_DAYS - PEAK_UNCERTAINTY_DAY)
            return MIN_ACCURACY_PERCENT + (LATE_ACCURACY_PERCENT - MIN_ACCURACY_PERCENT) * progress
    
    def update_estimation(self, current_date: datetime = None) -> datetime:
        """Update the estimated departure date."""
        return self.calculate_estimation(current_date)
    
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
    
    def can_stack_with(self, other_container: 'Container') -> bool:
        """Check if this container can be stacked on another."""
        # Basic stackability check
        if not self.is_stackable or not other_container.is_stackable:
            return False
        
        if self.stack_compatibility == "none" or other_container.stack_compatibility == "none":
            return False
        
        # Special goods compatibility using ONLY_SELF_STACKABLE constant
        if self.goods_type in ONLY_SELF_STACKABLE:
            if other_container.goods_type != self.goods_type:
                # Can only stack on same type or same-sized regular containers
                if other_container.goods_type != "Regular" or other_container.container_type != self.container_type:
                    return False
        
        # Stack compatibility rules
        if "self" in {self.stack_compatibility, other_container.stack_compatibility}:
            if self.container_type != other_container.container_type or self.goods_type != other_container.goods_type:
                return False
        
        if self.stack_compatibility == "size" and self.container_type != other_container.container_type:
            return False
        
        return True
    
    def __str__(self):
        return f"Container {self.container_id} ({self.container_type}): {self.direction}, {self.goods_type}"
    
    def __repr__(self):
        return f"Container(id={self.container_id}, type={self.container_type}, goods={self.goods_type})"
