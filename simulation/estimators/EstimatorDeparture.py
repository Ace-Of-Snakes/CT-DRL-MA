from datetime import datetime, timedelta
import numpy as np
from typing import Optional, List
from simulation.estimators.Estimator import EstimatorStrategy
from simulation.terminal_components.storage_units.Container import Container

class StandardDepartureEstimator(EstimatorStrategy):
    """
    Standard departure estimator with uncertainty model based on stay duration.
    Supports both single container and vectorized batch estimation.
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
        Calculate estimated departure date for a single container.
        
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
    
    def estimate_batch(self, 
                      containers: List[Container], 
                      current_date: Optional[datetime] = None) -> None:
        """
        Vectorized batch estimation for multiple containers.
        Modifies containers in-place for efficiency.
        
        Args:
            containers: List of containers to estimate
            current_date: Current simulation date (defaults to arrival date)
        """
        n = len(containers)
        if n == 0:
            return
        
        # Use first container's arrival if no current date provided
        if current_date is None:
            current_date = containers[0].arrival_date
        
        # Extract arrays for vectorized computation
        current_ts = current_date.timestamp()
        arrival_ts = np.array([c.arrival_date.timestamp() for c in containers])
        departure_ts = np.array([c.departure_date.timestamp() for c in containers])
        
        # Calculate time metrics in days
        days_in_terminal = np.maximum(0, (current_ts - arrival_ts) / 86400)
        total_stay = (departure_ts - arrival_ts) / 86400
        remaining_days = np.maximum(0, (departure_ts - current_ts) / 86400)
        
        # Vectorized accuracy calculation
        accuracy = self._vectorized_calculate_accuracy(total_stay, remaining_days)
        
        # Vectorized error generation
        error_days = np.zeros(n)
        non_perfect = accuracy < 0.99
        
        if np.any(non_perfect):
            # Only generate errors for non-perfect accuracy containers
            indices = np.where(non_perfect)[0]
            max_errors = np.maximum(7, total_stay[indices] * 0.3)
            std_devs = max_errors * (1.0 - accuracy[indices])
            
            # Generate all random values at once
            random_errors = np.random.normal(0, std_devs)
            error_days[indices] = np.clip(
                random_errors,
                -max_errors,
                max_errors
            )
        
        # Calculate estimated departures (in seconds since epoch)
        estimated_ts = departure_ts + (error_days * 86400)
        
        # Ensure reasonable estimates for containers in terminal
        in_terminal = days_in_terminal > 0
        too_early = estimated_ts < current_ts
        needs_adjustment = in_terminal & too_early
        estimated_ts[needs_adjustment] = current_ts + 86400  # Add 1 day
        
        # Update containers with estimates
        for i, container in enumerate(containers):
            container.estimated_departure = datetime.fromtimestamp(estimated_ts[i])
    
    def _calculate_accuracy(self, total_stay: float, remaining_days: float) -> float:
        """
        Calculate prediction accuracy for a single container.
        
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
            return self.MIN_ACC_BOOST_PERCENT
        
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
    
    def _vectorized_calculate_accuracy(self, 
                                       total_stay: np.ndarray, 
                                       remaining_days: np.ndarray) -> np.ndarray:
        """
        Vectorized accuracy calculation for batch processing.
        
        Args:
            total_stay: Array of total days containers will stay
            remaining_days: Array of days remaining until departure
            
        Returns:
            Array of accuracy values between 0 and 1
        """
        n = len(total_stay)
        accuracy = np.ones(n)
        
        # Perfect accuracy for short stays or imminent departures
        perfect = (total_stay <= self.MIN_ACCURACY_DAYS) | (remaining_days <= self.MIN_ACCURACY_DAYS)
        # Keep accuracy at 1.0 for perfect cases (already initialized to 1.0)
        
        # Boost accuracy in final week (but not if already perfect)
        boost = (~perfect) & (remaining_days <= self.MIN_ACCURACY_BOOST)
        accuracy[boost] = self.MIN_ACC_BOOST_PERCENT
        
        # Very long stays (but not if already perfect or boosted)
        very_long = (~perfect) & (~boost) & (total_stay >= self.MAX_HOLDING_DAYS)
        accuracy[very_long] = self.LATE_ACCURACY_PERCENT
        
        # Interpolate with dip at peak uncertainty
        normal = (~perfect) & (~boost) & (~very_long)
        
        # Before peak uncertainty
        before_peak = normal & (total_stay <= self.PEAK_UNCERTAINTY_DAY)
        if np.any(before_peak):
            progress = ((total_stay[before_peak] - self.MIN_ACCURACY_DAYS) / 
                       (self.PEAK_UNCERTAINTY_DAY - self.MIN_ACCURACY_DAYS))
            accuracy[before_peak] = 1.0 - (1.0 - self.MIN_ACCURACY_PERCENT) * progress
        
        # After peak uncertainty
        after_peak = normal & (total_stay > self.PEAK_UNCERTAINTY_DAY)
        if np.any(after_peak):
            progress = ((total_stay[after_peak] - self.PEAK_UNCERTAINTY_DAY) / 
                       (self.MAX_HOLDING_DAYS - self.PEAK_UNCERTAINTY_DAY))
            accuracy[after_peak] = (self.MIN_ACCURACY_PERCENT + 
                                   (self.LATE_ACCURACY_PERCENT - self.MIN_ACCURACY_PERCENT) * progress)
        
        return accuracy