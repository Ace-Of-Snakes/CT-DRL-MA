from datetime import datetime, timedelta
import numpy as np
from typing import Optional, Protocol
from simulation.estimators.Estimator import EstimatorStrategy
from simulation.terminal_components.storage_units.Container import Container

class StandardDepartureEstimator(EstimatorStrategy):
    """
    Standard departure estimator with uncertainty model based on stay duration.
    This implements the original estimation logic from the Container class.
    """
    
    # Estimation accuracy parameters
    MIN_ACCURACY_DAYS = 2           # Days with 100% accuracy
    MIN_ACCURACY_BOOST = 7          # Days before MIN accuracy with higher precision
    MAX_HOLDING_DAYS = 160          # Maximum container holding time
    MIN_ACCURACY_PERCENT = 0.30     # Minimum accuracy at peak uncertainty
    MIN_ACC_BOOST_PERCENT = 0.85    # Boost percentage
    LATE_ACCURACY_PERCENT = 0.45    # Accuracy for very long stays
    PEAK_UNCERTAINTY_DAY = 120      # Day with lowest accuracy
    
    def estimate_departure(self, 
                          container: Container, 
                          current_date: Optional[datetime] = None) -> datetime:
        """
        Calculate estimated departure date based on true departure and uncertainty model.
        
        Args:
            container: Container to estimate
            current_date: Current simulation date (defaults to arrival date)
            
        Returns:
            Estimated departure datetime
        """
        current_date = current_date if current_date is not None else container.arrival_date
        
        # Calculate time metrics
        days_in_terminal = container.days_in_terminal(current_date)
        total_stay = (container.departure_date - container.arrival_date).days
        remaining_days = max(0, (container.departure_date - current_date).days)
        
        # Calculate accuracy based on stay duration
        accuracy = self._calculate_accuracy(total_stay, remaining_days)
        
        # Generate estimation error
        if accuracy >= 0.99:
            error_days = 0
        else:
            max_error_days = max(7, total_stay * 0.3)
            std_dev = max_error_days * (1.0 - accuracy)
            error_days = np.clip(
                np.random.normal(0, std_dev), 
                -max_error_days, 
                max_error_days
            )
        
        # Calculate estimated departure
        estimated = container.departure_date + timedelta(days=error_days)
        
        # Ensure estimation is reasonable for containers still in terminal
        if days_in_terminal > 0 and estimated < current_date:
            estimated = current_date + timedelta(days=1)
        
        # Update the container's estimated departure
        container.estimated_departure = estimated
        
        return estimated
    
    def _calculate_accuracy(self, total_stay: int, remaining_days: int) -> float:
        """
        Calculate prediction accuracy based on stay duration.
        
        Args:
            total_stay: Total days container will stay
            remaining_days: Days remaining until departure
            
        Returns:
            Accuracy value between 0 and 1
        """
        # Perfect accuracy for very short stays or imminent departures
        if total_stay <= self.MIN_ACCURACY_DAYS or remaining_days <= self.MIN_ACCURACY_DAYS:
            return 1.0
        
        # Boost accuracy in final week
        if remaining_days <= self.MIN_ACCURACY_BOOST:
            return min(1.0, self.MIN_ACC_BOOST_PERCENT)
        
        # Very long stays
        if total_stay >= self.MAX_HOLDING_DAYS:
            return self.LATE_ACCURACY_PERCENT
        
        # Interpolate with dip at peak uncertainty
        if total_stay <= self.PEAK_UNCERTAINTY_DAY:
            progress = ((total_stay - self.MIN_ACCURACY_DAYS) / 
                       (self.PEAK_UNCERTAINTY_DAY - self.MIN_ACCURACY_DAYS))
            return 1.0 - (1.0 - self.MIN_ACCURACY_PERCENT) * progress
        else:
            progress = ((total_stay - self.PEAK_UNCERTAINTY_DAY) / 
                       (self.MAX_HOLDING_DAYS - self.PEAK_UNCERTAINTY_DAY))
            return (self.MIN_ACCURACY_PERCENT + 
                   (self.LATE_ACCURACY_PERCENT - self.MIN_ACCURACY_PERCENT) * progress)
