# simulation/config.py

import json
import os
import pickle
import numpy as np
from datetime import datetime, timedelta

class TerminalConfig:
    """Configuration manager for the terminal simulation."""
    
    def __init__(self, config_path=None):
        """
        Initialize the terminal configuration.
        
        Args:
            config_path: Path to the configuration JSON file
        """
        self.config = {
            "env_name": "CTNürnberg",
            "container_probabilities": {
                "length": {
                    "20": {
                        "probability": 0.177,
                        "probability_high_cube": 0.00023,
                        "probability_reefer": 0.0066,
                        "probability_dangerous_goods": 0.0134,
                        "probability_regular": 0.98
                    },
                    "30": {
                        "probability": 0.018,
                        "probability_high_cube": 0,
                        "probability_reefer": 0.0066,
                        "probability_dangerous_goods": 0.2203,
                        "probability_regular": 0.7731
                    },
                    "40": {
                        "probability": 0.521,
                        "probability_high_cube": 0.1115,
                        "probability_reefer": 0.0066,
                        "probability_dangerous_goods": 0.0023,
                        "probability_regular": 0.9911
                    },
                    "swap body": {
                        "probability": 0.251
                    },
                    "trailer": {
                        "probability": 0.033
                    }
                }
            }
        }
        
        # Load configuration from file if provided
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config.update(json.load(f))
        
        # Initialize KDE models
        self.kde_models = {}
        self.load_kde_models()
    
    def load_kde_models(self, models_dir='data/models'):
        """Load KDE models from the models directory."""
        model_files = {
            'train_arrival': 'train_arrival_kde.pkl',
            'truck_pickup': 'truck_pickup_kde.pkl',
            'train_delay': 'train_delay_kde.pkl',
            'pickup_wait': 'pickup_wait_kde.pkl',
            'container_weight': 'container_weight_kde.pkl'
        }
        
        for model_name, file_name in model_files.items():
            model_path = os.path.join(models_dir, file_name)
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.kde_models[model_name] = pickle.load(f)
            else:
                print(f"Warning: KDE model {model_path} not found")
    
    def sample_from_kde(self, model_name, n_samples=1, min_val=0, max_val=24):
        """
        Sample values from a KDE model.
        
        Args:
            model_name: Name of the model ('train_arrival', 'truck_pickup', etc.)
            n_samples: Number of samples to generate
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            
        Returns:
            Array of sampled values or None if model not found
        """
        if model_name not in self.kde_models:
            print(f"Warning: KDE model {model_name} not loaded")
            return None
        
        kde_model = self.kde_models[model_name]
        samples = kde_model.sample(n_samples=n_samples)
        
        # Apply modulo 24 to ensure time values stay within 0-24 range
        if min_val == 0 and max_val == 24:
            samples = samples % 24
        
        # Clip values to specified range
        samples = np.clip(samples, min_val, max_val)
        
        return samples.flatten()
    
    def get_container_type_probabilities(self):
        """Get the container type probabilities."""
        return self.config['container_probabilities']
    
    def hours_to_datetime(self, hours, base_date=None):
        """
        Convert decimal hours to a datetime object.
        
        Args:
            hours: Decimal hours (0-24)
            base_date: Base date (defaults to today)
            
        Returns:
            datetime object
        """
        if base_date is None:
            base_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        h = int(hours)
        m = int((hours - h) * 60)
        s = int(((hours - h) * 60 - m) * 60)
        
        return base_date + timedelta(hours=h, minutes=m, seconds=s)
    
    def generate_train_arrival_schedule(self, n_trains=10, base_date=None):
        """
        Generate a schedule of train arrivals.
        
        Args:
            n_trains: Number of trains to schedule
            base_date: Base date for the schedule
            
        Returns:
            List of (train_id, planned_arrival, realized_arrival) tuples
        """
        if base_date is None:
            base_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Generate planned arrival times (in hours)
        planned_hours = self.sample_from_kde('train_arrival', n_samples=n_trains)
        
        # Generate delays (in hours)
        delays = self.sample_from_kde('train_delay', n_samples=n_trains, min_val=-3, max_val=3)
        
        schedule = []
        for i, (p_hour, delay) in enumerate(zip(planned_hours, delays)):
            planned_datetime = self.hours_to_datetime(p_hour, base_date)
            realized_datetime = planned_datetime + timedelta(hours=delay)
            train_id = f"TRN{i+1:04d}"
            schedule.append((train_id, planned_datetime, realized_datetime))
        
        # Sort by realized arrival time
        schedule.sort(key=lambda x: x[2])
        
        return schedule
    
    def generate_truck_pickup_schedule(self, container_ids, base_date=None):
        """
        Generate a schedule of truck pickups based on container IDs.
        
        Args:
            container_ids: List of container IDs to be picked up
            base_date: Base date for the schedule
            
        Returns:
            Dictionary mapping container_id to pickup datetime
        """
        if base_date is None:
            base_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Generate pickup wait times (in hours)
        wait_times = self.sample_from_kde('pickup_wait', n_samples=len(container_ids), min_val=0, max_val=72)
        
        # Generate pickup hour of day
        pickup_hours = self.sample_from_kde('truck_pickup', n_samples=len(container_ids))
        
        pickup_schedule = {}
        for i, container_id in enumerate(container_ids):
            # Calculate pickup date based on wait time
            wait_days = int(wait_times[i] // 24)
            pickup_date = base_date + timedelta(days=wait_days)
            
            # Set the hour of day for pickup
            pickup_datetime = self.hours_to_datetime(pickup_hours[i], pickup_date)
            
            pickup_schedule[container_id] = pickup_datetime
        
        return pickup_schedule