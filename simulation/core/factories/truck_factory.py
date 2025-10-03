# simulation/core/factories/truck_factory.py
import os
import pickle
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from scipy.stats import gaussian_kde

from simulation.core.vehicles.truck import Truck
from simulation.core.containers.container import Container
from simulation.core.constants import STANDARD_VEHICLE_LENGTH_M, MIN_CONTAINER_LENGTH_M
from simulation.config.paths import DataPaths
from simulation.config.operations_config import GateConfig


class TruckFactory:
    """
    Factory class for efficient truck generation using KDE models for arrival times.
    All trucks are 80 feet (24.4 meters) long.
    """
    
    def __init__(self, kde_folder: str = None):
        """
        Initialize factory with KDE models for arrival time distributions.
        
        Args:
            kde_folder: Base folder containing 'delivery' and 'pickup' subfolders with KDE models
        """
        kde_folder = kde_folder or str(DataPaths.TRUCK_ARRIVALS_DIR)
        self.kde_folder = kde_folder
        
        # Load KDE models for both delivery and pickup
        self.delivery_kde_models = self._load_kde_models(
            os.path.join(kde_folder, "delivery")
        )
        self.pickup_kde_models = self._load_kde_models(
            os.path.join(kde_folder, "pickup")
        )
        
        # ID counter for unique truck IDs
        self._id_counter = 0
    
    def _load_kde_models(self, folder_path: str) -> Dict[str, gaussian_kde]:
        """
        Load KDE models from a folder for each day of the week.
        
        Args:
            folder_path: Path to folder containing KDE pickle files
            
        Returns:
            Dictionary mapping day names to KDE models
        """
        kde_models = {}
        
        if not os.path.exists(folder_path):
            print(f"Warning: KDE folder {folder_path} does not exist")
            return kde_models
        
        weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        for day in weekdays:
            file_path = os.path.join(folder_path, f"{day}_kde.pkl")
            
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'rb') as f:
                        kde_data = pickle.load(f)
                    
                    if isinstance(kde_data, gaussian_kde):
                        kde_models[day.lower()] = kde_data
                    elif isinstance(kde_data, dict) and 'dataset' in kde_data:
                        kde_models[day.lower()] = gaussian_kde(kde_data['dataset'])
                    
                except Exception as e:
                    print(f"Warning: Could not load KDE for {day}: {e}")
        
        return kde_models
    
    def generate_trucks(
        self,
        day_of_week: str,
        delivery_containers: Optional[List[Container]] = None,
        pickup_containers: Optional[List[Container]] = None,
        base_date: Optional[datetime] = None,
        parking_spot_prefix: str = "P"
    ) -> List[Truck]:
        """
        Generate trucks for a specific day with containers to deliver and/or pick up.
        
        Args:
            day_of_week: Day name (e.g., "Monday", "Tuesday")
            delivery_containers: Containers to be delivered to terminal
            pickup_containers: Containers to be picked up from terminal
            base_date: Base date for the specified day
            parking_spot_prefix: Prefix for parking spot assignments
            
        Returns:
            List of Truck objects with assigned containers and arrival times
        """
        trucks = []
        day_key = day_of_week.lower()
        
        # Generate delivery trucks
        if delivery_containers:
            delivery_trucks = self._generate_delivery_trucks(
                containers=delivery_containers,
                day_key=day_key,
                base_date=base_date,
                parking_spot_prefix=parking_spot_prefix
            )
            trucks.extend(delivery_trucks)
        
        # Generate pickup trucks
        if pickup_containers:
            pickup_trucks = self._generate_pickup_trucks(
                containers=pickup_containers,
                day_key=day_key,
                base_date=base_date,
                parking_spot_prefix=parking_spot_prefix
            )
            trucks.extend(pickup_trucks)
        
        return trucks
    
    def _generate_delivery_trucks(
        self,
        containers: List[Container],
        day_key: str,
        base_date: Optional[datetime],
        parking_spot_prefix: str
    ) -> List[Truck]:
        """Generate trucks for delivering containers to the terminal."""
        if not containers:
            return []
        
        trucks = []
        
        # Sort containers by length for better packing (largest first)
        sorted_containers = sorted(containers, key=lambda c: c.length_m, reverse=True)
        
        # Pack containers into trucks
        current_truck_containers = []
        current_used_length = 0.0
        
        for container in sorted_containers:
            if current_used_length + container.length_m <= STANDARD_VEHICLE_LENGTH_M:
                current_truck_containers.append(container)
                current_used_length += container.length_m
            else:
                # Create truck with current containers
                if current_truck_containers:
                    truck = self._create_delivery_truck(
                        containers=current_truck_containers,
                        day_key=day_key,
                        base_date=base_date,
                        parking_spot_prefix=parking_spot_prefix
                    )
                    trucks.append(truck)
                
                # Start new truck
                current_truck_containers = [container]
                current_used_length = container.length_m
        
        # Create final truck
        if current_truck_containers:
            truck = self._create_delivery_truck(
                containers=current_truck_containers,
                day_key=day_key,
                base_date=base_date,
                parking_spot_prefix=parking_spot_prefix
            )
            trucks.append(truck)
        
        return trucks
    
    def _generate_pickup_trucks(
        self,
        containers: List[Container],
        day_key: str,
        base_date: Optional[datetime]
    ) -> List[Truck]:
        """
        Generate exactly one pickup truck per container.
        No bundling. Arrival time from pickup KDE (6-22h) on same day.
        """
        if not containers:
            return []
        
        trucks: List[Truck] = []
        
        for container in containers:
            arrival_time = self._sample_arrival_time(
                day_key=day_key,
                is_delivery=False,
                base_date=base_date
            )
            
            truck = Truck(
                max_length=STANDARD_VEHICLE_LENGTH_M,
                arrival_time=arrival_time,
                parking_spot=None
            )
            truck.add_pickup_container_id(container.container_id)
            truck.is_pickup_truck = True
            truck.is_delivery_truck = False
            trucks.append(truck)
        
        return trucks
    
    def _create_delivery_truck(
        self,
        containers: List[Container],
        day_key: str,
        base_date: Optional[datetime]
    ) -> Truck:
        """Create a delivery truck with containers."""
        self._id_counter += 1
        truck_id = f"TRK{self._id_counter:05d}"
        
        arrival_time = self._sample_arrival_time(
            day_key=day_key,
            is_delivery=True,
            base_date=base_date
        )
        
        # Do NOT pre-assign parking; agent will handle SLOT_TRUCK_PARKING
        truck = Truck(
            truck_id=truck_id,
            max_length=STANDARD_VEHICLE_LENGTH_M,
            arrival_time=arrival_time,
            parking_spot=None
        )
        
        for container in containers:
            truck.add_container(container)
        
        truck.is_delivery_truck = True
        truck.is_pickup_truck = False
        
        return truck
    
    def _sample_arrival_time(
        self,
        day_key: str,
        is_delivery: bool,
        base_date: Optional[datetime]
    ) -> datetime:
        """
        Sample arrival time from KDE model for the specified day.
        
        Args:
            day_key: Day of week (lowercase)
            is_delivery: True for delivery, False for pickup
            base_date: Base date for arrival
            
        Returns:
            Datetime for truck arrival
        """
        if base_date is None:
            base_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        kde_models = self.delivery_kde_models if is_delivery else self.pickup_kde_models
        
        if day_key in kde_models:
            kde = kde_models[day_key]
            sampled_hours = kde.resample(1)
            
            if sampled_hours.ndim > 1:
                sampled_hours = sampled_hours[0, 0]
            else:
                sampled_hours = sampled_hours[0]
            
            sampled_hours = np.clip(
                sampled_hours,
                GateConfig.TRUCK_ARRIVAL_HOUR_START,
                GateConfig.TRUCK_ARRIVAL_HOUR_END
            )
        else:
            # No KDE model - use uniform distribution
            sampled_hours = np.random.uniform(
                GateConfig.TRUCK_ARRIVAL_HOUR_START,
                GateConfig.TRUCK_ARRIVAL_HOUR_END
            )
        
        hours = int(sampled_hours)
        minutes = int((sampled_hours - hours) * 60)
        
        return base_date.replace(hour=hours, minute=minutes)
    
    def get_kde_summary(self) -> Dict[str, Dict[str, any]]:
        """Get summary of loaded KDE models."""
        return {
            'delivery_models': {
                'available_days': list(self.delivery_kde_models.keys()),
                'count': len(self.delivery_kde_models)
            },
            'pickup_models': {
                'available_days': list(self.pickup_kde_models.keys()),
                'count': len(self.pickup_kde_models)
            }
        }