from datetime import datetime, timedelta
import numpy as np
from typing import Optional, Protocol
from simulation.estimators.Estimator import EstimatorStrategy
from simulation.terminal_components.storage_units.Container import Container

class SimpleRandomEstimator(EstimatorStrategy):
    """
    Simple random estimator with configurable error range.
    Provides a simpler alternative to the standard estimator.
    """
    
    def __init__(self, max_error_days: int = 7):
        """
        Initialize the estimator.
        
        Args:
            max_error_days: Maximum error in either direction
        """
        self.max_error_days = max_error_days
    
    def estimate_departure(self, 
                          container: Container, 
                          current_date: Optional[datetime] = None) -> datetime:
        """Add random error to true departure date."""
        error_days = np.random.randint(-self.max_error_days, self.max_error_days + 1)
        estimated = container.departure_date + timedelta(days=error_days)
        
        # Don't estimate in the past for containers still in terminal
        current_date = current_date or container.arrival_date
        if container.days_in_terminal(current_date) > 0 and estimated < current_date:
            estimated = current_date + timedelta(days=1)
        
        container.estimated_departure = estimated
        return estimated