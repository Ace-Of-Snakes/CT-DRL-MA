from datetime import datetime, timedelta
import numpy as np
from typing import Optional, Protocol
from simulation.estimators.Estimator import EstimatorStrategy
from simulation.terminal_components.storage_units.Container import Container
from simulation.estimators.EstimatorDeparture import StandardDepartureEstimator
from simulation.estimators.EstimatorRandom import SimpleRandomEstimator
from simulation.estimators.EstimatorPerfect import PerfectEstimator

class DepartureEstimatorManager:
    """
    Manager class for handling departure estimations.
    Allows switching between different estimation strategies.
    """
    
    def __init__(self, strategy: Optional[EstimatorStrategy] = None):
        """
        Initialize the manager with a strategy.
        
        Args:
            strategy: Estimation strategy to use (defaults to StandardDepartureEstimator)
        """
        self.strategy = strategy or StandardDepartureEstimator()
    
    def set_strategy(self, strategy: EstimatorStrategy):
        """Change the estimation strategy."""
        self.strategy = strategy
    
    def estimate(self, container: Container, current_date: Optional[datetime] = None) -> datetime:
        """
        Estimate departure for a container using current strategy.
        
        Args:
            container: Container to estimate
            current_date: Current simulation date
            
        Returns:
            Estimated departure datetime
        """
        return self.strategy.estimate_departure(container, current_date)
    
    def batch_estimate(self, 
                      containers: list[Container], 
                      current_date: Optional[datetime] = None) -> list[datetime]:
        """
        Estimate departures for multiple containers.
        
        Args:
            containers: List of containers to estimate
            current_date: Current simulation date
            
        Returns:
            List of estimated departure times
        """
        return [self.estimate(c, current_date) for c in containers]
    
    def update_estimation(self, container: Container, current_date: datetime) -> datetime:
        """
        Update the estimation for a container (recalculate).
        
        Args:
            container: Container to update
            current_date: Current simulation date
            
        Returns:
            New estimated departure datetime
        """
        return self.estimate(container, current_date)
    
if __name__ == "__main__":    

    # Create a container (pure data)
    container = Container(
        container_id="CTR001",
        direction="Import",
        container_type="FEU",
        arrival_date=datetime(2024, 1, 1),
        departure_date=datetime(2024, 1, 15),
        goods_type="Reefer"
    )

    # Create an estimator manager
    estimator = DepartureEstimatorManager()  # Uses StandardDepartureEstimator by default

    # Estimate departure
    estimated_date = estimator.estimate(container, datetime(2024, 1, 5))
    print(f"Estimated departure: {container.estimated_departure}")

    # Switch to a different strategy
    estimator.set_strategy(PerfectEstimator())
    estimated_date = estimator.estimate(container, datetime(2024, 1, 5))
    print(f"Perfect estimate: {container.estimated_departure}")

    # Use simple random estimator
    estimator.set_strategy(SimpleRandomEstimator(max_error_days=3))
    estimated_date = estimator.estimate(container, datetime(2024, 1, 5))
    print(f"Simple random estimate: {container.estimated_departure}")