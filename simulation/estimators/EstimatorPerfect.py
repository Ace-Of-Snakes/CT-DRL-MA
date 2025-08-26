from datetime import datetime, timedelta
import numpy as np
from typing import Optional, Protocol
from simulation.estimators.Estimator import EstimatorStrategy
from simulation.terminal_components.storage_units.Container import Container

class PerfectEstimator(EstimatorStrategy):
    """
    Perfect estimator that always returns the true departure date.
    Useful for testing or scenarios without uncertainty.
    """
    
    def estimate_departure(self, 
                          container: Container, 
                          current_date: Optional[datetime] = None) -> datetime:
        """Always return the true departure date."""
        container.estimated_departure = container.departure_date
        return container.departure_date