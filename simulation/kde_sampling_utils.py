# simulation/kde_sampling_utils.py

import numpy as np
import pickle
from typing import Optional, Tuple, Union


def load_kde_model(model_path: str):
    """
    Load a KDE model from a pickle file.
    
    Args:
        model_path: Path to the pickle file containing the KDE model
        
    Returns:
        Loaded KDE model object
    """
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def sample_from_kde(
    kde_model, 
    n_samples: int = 1, 
    min_val: float = 0, 
    max_val: float = 24
) -> np.ndarray:
    """
    Sample values from a KDE model with bounds.
    
    Args:
        kde_model: The KDE model to sample from
        n_samples: Number of samples to generate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        
    Returns:
        Array of sampled values
    """
    samples = kde_model.sample(n_samples=n_samples)
    
    # Apply modulo 24 to ensure time values stay within 0-24 range
    if min_val == 0 and max_val == 24:
        samples = samples % 24
    
    # Clip values to specified range
    samples = np.clip(samples, min_val, max_val)
    
    return samples.flatten()


def hours_to_time(hours: float) -> Tuple[int, int, int]:
    """
    Convert decimal hours to hours, minutes, seconds.
    
    Args:
        hours: Decimal hours (e.g., 13.5 = 1:30 PM)
        
    Returns:
        Tuple of (hours, minutes, seconds)
    """
    h = int(hours)
    m = int((hours - h) * 60)
    s = int(((hours - h) * 60 - m) * 60)
    return h, m, s


def time_to_hours(h: int, m: int = 0, s: int = 0) -> float:
    """
    Convert hours, minutes, seconds to decimal hours.
    
    Args:
        h: Hours (0-23)
        m: Minutes (0-59)
        s: Seconds (0-59)
        
    Returns:
        Decimal hours
    """
    return h + m / 60.0 + s / 3600.0


def sample_container_weight(kde_model: Optional[object] = None) -> float:
    """
    Sample a container weight using KDE or fallback distribution.
    
    Args:
        kde_model: Optional KDE model for container weights
        
    Returns:
        Container weight in kg
    """
    if kde_model is not None:
        # Sample from KDE
        weight = sample_from_kde(
            kde_model, 
            n_samples=1, 
            min_val=1000,  # 1 tonne minimum
            max_val=31000  # 31 tonnes maximum
        )[0]
        return float(weight)
    else:
        # Fallback: normal distribution centered around 20 tonnes
        weight = np.random.normal(20000, 5000)
        return float(np.clip(weight, 1000, 31000))


def sample_train_schedule(
    n_trains: int,
    train_arrival_kde: Optional[object] = None,
    train_delay_kde: Optional[object] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample train arrival times and delays.
    
    Args:
        n_trains: Number of trains to schedule
        train_arrival_kde: KDE model for arrival times
        train_delay_kde: KDE model for delays
        
    Returns:
        Tuple of (arrival_hours, delay_hours)
    """
    if train_arrival_kde is not None:
        arrival_hours = sample_from_kde(train_arrival_kde, n_samples=n_trains)
    else:
        # Fallback: uniform distribution across the day
        arrival_hours = np.random.uniform(0, 24, n_trains)
    
    if train_delay_kde is not None:
        delays = sample_from_kde(
            train_delay_kde, 
            n_samples=n_trains, 
            min_val=-3,  # Max 3 hours early
            max_val=3    # Max 3 hours late
        )
    else:
        # Fallback: normal distribution with small variance
        delays = np.random.normal(0, 0.5, n_trains)
        delays = np.clip(delays, -3, 3)
    
    return arrival_hours, delays


def sample_truck_arrivals(
    n_trucks: int,
    truck_pickup_kde: Optional[object] = None
) -> np.ndarray:
    """
    Sample truck arrival times during the day.
    
    Args:
        n_trucks: Number of trucks to schedule
        truck_pickup_kde: KDE model for truck arrivals
        
    Returns:
        Array of arrival hours
    """
    if truck_pickup_kde is not None:
        return sample_from_kde(truck_pickup_kde, n_samples=n_trucks)
    else:
        # Fallback: concentrated during business hours
        return np.random.normal(12, 3, n_trucks) % 24


def sample_pickup_wait_time(
    pickup_wait_kde: Optional[object] = None,
    n_samples: int = 1
) -> np.ndarray:
    """
    Sample wait time until pickup in hours.
    
    Args:
        pickup_wait_kde: KDE model for pickup wait times
        n_samples: Number of samples
        
    Returns:
        Array of wait times in hours
    """
    if pickup_wait_kde is not None:
        return sample_from_kde(
            pickup_wait_kde, 
            n_samples=n_samples,
            min_val=0,    # No negative wait
            max_val=168   # Max 1 week
        )
    else:
        # Fallback: exponential distribution with mean 48 hours
        wait_times = np.random.exponential(48, n_samples)
        return np.clip(wait_times, 0, 168)