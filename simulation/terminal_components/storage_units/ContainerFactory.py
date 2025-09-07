import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from scipy.stats import gaussian_kde
from simulation.terminal_components.storage_units.Container import Container, Direction, GoodsType
from simulation.estimators.EstimatorDeparture import StandardDepartureEstimator


class ContainerFactory:
    """
    Factory class for efficient container generation using preloaded models and data.
    Loads all necessary data once at initialization and uses vectorized operations.
    """
    
    def __init__(self, 
                 operator_dist_file: str = "simulation/data/train_operator_container_type_distribution.json",
                 container_data_file: str = "simulation/data/container_data.csv",
                 models_folder: str = "simulation/data/models",
                 export_kde_file: str = "kde_dwelltime_export.pkl",
                 use_estimator: bool = True):
        """
        Initialize factory with all necessary data loaded into memory.
        
        Args:
            operator_dist_file: Path to operator distribution JSON
            container_data_file: Path to container specifications CSV
            models_folder: Folder containing KDE models
            export_kde_file: Name of export KDE file in models folder
            use_estimator: Whether to apply departure estimation to containers
        """
        # Load operator distributions
        with open(operator_dist_file, 'r') as f:
            self.operator_dict = json.load(f)
        
        # Load container specifications
        self._load_container_specs(container_data_file)
        
        # Load KDE models for import containers
        self.import_kde_models = self._load_kde_models(models_folder)
        
        # Load single export KDE model
        export_path = os.path.join(models_folder, export_kde_file)
        with open(export_path, 'rb') as f:
            export_data = pickle.load(f)
        
        # Handle different possible formats of the export KDE file
        if isinstance(export_data, gaussian_kde):
            # Already a KDE object
            self.export_kde = export_data
        elif isinstance(export_data, dict) and "weighted_days" in export_data:
            # Dictionary with parameters
            self.export_kde = gaussian_kde(
                export_data["weighted_days"],
                bw_method=export_data.get("bw_method", None)
            )
        else:
            # Try to use it directly as data for KDE
            try:
                self.export_kde = gaussian_kde(export_data)
            except Exception as e:
                raise ValueError(f"Could not interpret export KDE file format: {e}")
        
        # Initialize departure estimator
        self.use_estimator = use_estimator
        if use_estimator:
            self.estimator = StandardDepartureEstimator()
        
        # ID counter for unique container IDs
        self._id_counter = 0
    
    def _load_container_specs(self, csv_file: str):
        """Load container specifications from CSV."""
        df = pd.read_csv(csv_file)
        
        # Create lookup dictionary with all properties
        self.container_specs = {
            row["CONTAINER"]: {
                "length_ft": int(row["LENGTH_IN_FEET"]),
                "length_m": float(row["LENGTH_IN_METER"]),
                "width_m": float(row["WIDTH_IN_METER"]),
                "height_m": float(row["HEIGHT"]),
                "is_high_cube": bool(row["IS_HIGH_CUBE"]),
                "is_swap_body": bool(row["IS_SWAP_BODY"]),
                "is_trailer": bool(row["IS_TRAILER"])
            }
            for _, row in df.iterrows()
        }
        
        # Store available container types for validation
        self.available_container_types = set(self.container_specs.keys())
    
    def _load_kde_models(self, models_folder: str) -> Dict:
        """Load all import KDE models."""
        kde_models = {}
        for file in os.listdir(models_folder):
            if file.endswith("_dwelltime_kde.pkl") and not file.startswith("kde_dwelltime_export"):
                container_type = file.replace("_dwelltime_kde.pkl", "")
                with open(os.path.join(models_folder, file), 'rb') as f:
                    kde_data = pickle.load(f)
                
                # Handle different possible formats
                if isinstance(kde_data, gaussian_kde):
                    kde_models[container_type] = kde_data
                elif isinstance(kde_data, dict) and "weighted_days" in kde_data:
                    kde_models[container_type] = gaussian_kde(
                        kde_data["weighted_days"],
                        bw_method=kde_data.get("bw_method", None)
                    )
                else:
                    # Skip if format is unrecognized
                    print(f"Warning: Skipping {file} - unrecognized format")
                    
        return kde_models
    
    def create_containers(self, 
                         operator: str, 
                         direction: Direction,
                         n_containers: int = 1,
                         base_arrival_date: Optional[datetime] = None,
                         current_date: Optional[datetime] = None) -> List[Container]:
        """
        Create containers for a given operator and direction.
        
        Args:
            operator: Operator name (must exist in operator_dict)
            direction: "Import" or "Export"
            n_containers: Number of containers to create
            base_arrival_date: Base arrival date (defaults to now)
            current_date: Current simulation date for estimation (defaults to arrival)
            
        Returns:
            List of Container objects with estimated departures
        """
        if operator not in self.operator_dict:
            raise ValueError(f"Unknown operator: {operator}")
        
        if base_arrival_date is None:
            base_arrival_date = datetime.now()
        
        # Sample container properties
        samples = self._sample_containers(operator, direction, n_containers)
        
        # Create Container objects
        containers = []
        for sample in samples:
            container = self._create_single_container(sample, direction, base_arrival_date)
            containers.append(container)
        
        # Apply batch estimation if enabled
        if self.use_estimator and containers:
            self.estimator.estimate_batch(containers, current_date or base_arrival_date)
        
        return containers
    
    def _sample_containers(self, 
                          operator: str, 
                          direction: Direction,
                          n_samples: int) -> np.ndarray:
        """
        Vectorized sampling of container properties.
        
        Returns:
            Structured array with sampled properties
        """
        operator_data = self.operator_dict[operator]
        
        # Extract container types and probabilities (filter to available types)
        all_types = list(operator_data.keys())
        valid_types = [t for t in all_types if t in self.available_container_types]
        
        if not valid_types:
            raise ValueError(f"No valid container types found for operator {operator}")
        
        container_types = np.array(valid_types)
        probs = np.array([operator_data[c]["P_for_operator"] for c in container_types])
        probs /= probs.sum()  # Normalize
        
        # 1. Sample container types
        sampled_types = np.random.choice(container_types, size=n_samples, p=probs)
        
        # 2. Sample goods types (vectorized)
        reefer_probs = np.array([operator_data[t]["P_to_be_Reefer"] for t in sampled_types])
        dg_probs = np.array([operator_data[t]["P_to_be_DangerousGoods"] for t in sampled_types])
        
        reefer_flags = np.random.rand(n_samples) < reefer_probs
        dg_flags = (np.random.rand(n_samples) < dg_probs) & (~reefer_flags)
        goods_types = np.where(reefer_flags, "Reefer", 
                               np.where(dg_flags, "DangerousGoods", "Regular"))
        
        # 3. Sample dwell times based on direction
        dwell_times = self._sample_dwell_times(sampled_types, direction, n_samples)
        
        # 4. Get container specifications (vectorized lookup)
        specs = np.array([self.container_specs[t] for t in sampled_types])
        
        # Create structured array for efficient access
        dtype = [
            ('container_type', 'U10'),
            ('goods_type', 'U15'),
            ('dwell_time', 'f8'),
            ('length_ft', 'i4'),
            ('length_m', 'f8'),
            ('width_m', 'f8'),
            ('height_m', 'f8'),
            ('is_high_cube', 'bool'),
            ('is_swap_body', 'bool'),
            ('is_trailer', 'bool')
        ]
        
        result = np.empty(n_samples, dtype=dtype)
        result['container_type'] = sampled_types
        result['goods_type'] = goods_types
        result['dwell_time'] = dwell_times
        
        for i, spec in enumerate(specs):
            result['length_ft'][i] = spec['length_ft']
            result['length_m'][i] = spec['length_m']
            result['width_m'][i] = spec['width_m']
            result['height_m'][i] = spec['height_m']
            result['is_high_cube'][i] = spec['is_high_cube']
            result['is_swap_body'][i] = spec['is_swap_body']
            result['is_trailer'][i] = spec['is_trailer']
        
        return result
        
    def _sample_dwell_times(self, 
                        sampled_types: np.ndarray,
                        direction: Direction,
                        n_samples: int) -> np.ndarray:
        """Sample dwell times based on direction with precision handling."""
        dwell_times = np.empty(n_samples)
        
        if direction == "Export":
            # All exports use same KDE
            sampled = self.export_kde.resample(n_samples)
            # Handle both 1D and 2D array returns from resample
            if sampled.ndim > 1:
                dwell_times[:] = sampled[0]
            else:
                dwell_times[:] = sampled
        else:
            # Imports use type-specific KDEs
            # Set default value for containers without KDE
            dwell_times[:] = 2.0  # Default if no KDE available
            
            for ct in np.unique(sampled_types):
                if ct in self.import_kde_models:
                    idx = np.where(sampled_types == ct)[0]
                    sampled = self.import_kde_models[ct].resample(len(idx))
                    # Handle both 1D and 2D array returns
                    if sampled.ndim > 1:
                        dwell_times[idx] = sampled[0]
                    else:
                        dwell_times[idx] = sampled
        
        # Ensure non-negative values
        dwell_times = np.maximum(dwell_times, 0)
        
        # Round short dwell times to integers to ensure perfect accuracy works correctly
        # This is critical for containers that should have perfect estimation
        short_stay_mask = dwell_times <= 2.0
        dwell_times[short_stay_mask] = np.round(dwell_times[short_stay_mask])
        
        return dwell_times
    
    def _vectorized_estimate_departures(self, containers: List[Container], current_date: datetime):
        """
        Vectorized departure estimation for a batch of containers.
        Modifies containers in-place for efficiency.
        
        Args:
            containers: List of containers to estimate
            current_date: Current simulation date
        """
        n = len(containers)
        if n == 0:
            return
        
        # Extract arrays for vectorized computation
        arrival_dates = np.array([c.arrival_date for c in containers])
        departure_dates = np.array([c.departure_date for c in containers])
        
        # Convert to days for computation
        current_ts = current_date.timestamp()
        arrival_ts = np.array([d.timestamp() for d in arrival_dates])
        departure_ts = np.array([d.timestamp() for d in departure_dates])
        
        days_in_terminal = np.maximum(0, (current_ts - arrival_ts) / 86400)
        total_stay = (departure_ts - arrival_ts) / 86400
        remaining_days = np.maximum(0, (departure_ts - current_ts) / 86400)
        
        # Vectorized accuracy calculation
        accuracy = self._vectorized_calculate_accuracy(total_stay, remaining_days)
        
        # Vectorized error generation
        error_days = np.zeros(n)
        non_perfect = accuracy < 0.99
        
        if np.any(non_perfect):
            max_errors = np.maximum(7, total_stay * 0.3)
            std_devs = max_errors * (1.0 - accuracy)
            
            # Generate all random values at once
            random_errors = np.random.normal(0, std_devs)
            error_days[non_perfect] = np.clip(
                random_errors[non_perfect],
                -max_errors[non_perfect],
                max_errors[non_perfect]
            )
        
        # Calculate estimated departures
        estimated_ts = departure_ts + (error_days * 86400)
        
        # Ensure reasonable estimates for containers in terminal
        in_terminal = days_in_terminal > 0
        too_early = estimated_ts < current_ts
        needs_adjustment = in_terminal & too_early
        estimated_ts[needs_adjustment] = current_ts + 86400  # Add 1 day
        
        # Convert back to datetime and assign to containers
        for i, container in enumerate(containers):
            container.estimated_departure = datetime.fromtimestamp(estimated_ts[i])
    
    def _vectorized_calculate_accuracy(self, total_stay: np.ndarray, remaining_days: np.ndarray) -> np.ndarray:
        """
        Vectorized accuracy calculation based on StandardDepartureEstimator logic.
        
        Args:
            total_stay: Array of total days containers will stay
            remaining_days: Array of days remaining until departure
            
        Returns:
            Array of accuracy values between 0 and 1
        """
        # Constants from StandardDepartureEstimator
        MIN_ACCURACY_DAYS = 2
        MIN_ACCURACY_BOOST = 7
        MAX_HOLDING_DAYS = 160
        MIN_ACCURACY_PERCENT = 0.30
        MIN_ACC_BOOST_PERCENT = 0.85
        LATE_ACCURACY_PERCENT = 0.45
        PEAK_UNCERTAINTY_DAY = 120
        
        n = len(total_stay)
        accuracy = np.ones(n)
        
        # Perfect accuracy for short stays or imminent departures
        perfect = (total_stay <= MIN_ACCURACY_DAYS) | (remaining_days <= MIN_ACCURACY_DAYS)
        
        # Boost accuracy in final week
        boost = (~perfect) & (remaining_days <= MIN_ACCURACY_BOOST)
        accuracy[boost] = MIN_ACC_BOOST_PERCENT
        
        # Very long stays
        very_long = (~perfect) & (~boost) & (total_stay >= MAX_HOLDING_DAYS)
        accuracy[very_long] = LATE_ACCURACY_PERCENT
        
        # Interpolate with dip at peak uncertainty
        normal = (~perfect) & (~boost) & (~very_long)
        
        # Before peak uncertainty
        before_peak = normal & (total_stay <= PEAK_UNCERTAINTY_DAY)
        if np.any(before_peak):
            progress = ((total_stay[before_peak] - MIN_ACCURACY_DAYS) / 
                       (PEAK_UNCERTAINTY_DAY - MIN_ACCURACY_DAYS))
            accuracy[before_peak] = 1.0 - (1.0 - MIN_ACCURACY_PERCENT) * progress
        
        # After peak uncertainty
        after_peak = normal & (total_stay > PEAK_UNCERTAINTY_DAY)
        if np.any(after_peak):
            progress = ((total_stay[after_peak] - PEAK_UNCERTAINTY_DAY) / 
                       (MAX_HOLDING_DAYS - PEAK_UNCERTAINTY_DAY))
            accuracy[after_peak] = (MIN_ACCURACY_PERCENT + 
                                   (LATE_ACCURACY_PERCENT - MIN_ACCURACY_PERCENT) * progress)
        
        return accuracy
    
    def create_batch(self,
                    operators_directions: List[Tuple[str, Direction, int]],
                    base_arrival_date: Optional[datetime] = None,
                    current_date: Optional[datetime] = None) -> List[Container]:
        """
        Create multiple batches of containers efficiently.
        
        Args:
            operators_directions: List of (operator, direction, count) tuples
            base_arrival_date: Base arrival date for all containers
            current_date: Current simulation date for estimation
            
        Returns:
            Combined list of all containers with estimated departures
        """
        all_containers = []
        for operator, direction, count in operators_directions:
            containers = self.create_containers(
                operator, direction, count, base_arrival_date, current_date
            )
            all_containers.extend(containers)
        return all_containers
    
    def _create_single_container(self, 
                                sample: np.void,
                                direction: Direction,
                                base_arrival_date: datetime) -> Container:
        """Create a single Container object from sampled data."""
        self._id_counter += 1
        
        # Calculate dates with proper rounding for fractional days
        arrival_date = base_arrival_date
        dwell_days = float(sample['dwell_time'])
        
        # Round dwell time to avoid floating point precision issues
        # For perfect accuracy cases (<=2 days), ensure exact integer days
        if dwell_days <= 2.0:
            dwell_days = round(dwell_days)  # Round to nearest integer day
        
        # Use total_seconds for more precise timedelta
        dwell_seconds = dwell_days * 86400  # Convert to seconds
        departure_date = arrival_date + timedelta(seconds=round(dwell_seconds))
        
        return Container(
            container_id=f"C{self._id_counter:06d}",
            direction=direction,
            container_type=str(sample['container_type']),
            arrival_date=arrival_date,
            departure_date=departure_date,
            goods_type=str(sample['goods_type']),
            length_ft=int(sample['length_ft']),
            length_m=float(sample['length_m']),
            width_m=float(sample['width_m']),
            height_m=float(sample['height_m']),
            is_high_cube=bool(sample['is_high_cube']),
            is_swap_body=bool(sample['is_swap_body']),
            is_trailer=bool(sample['is_trailer'])
        )
    
    def update_estimations(self, containers: List[Container], current_date: datetime):
        """
        Update departure estimations for existing containers.
        
        Args:
            containers: List of containers to update
            current_date: Current simulation date
        """
        if self.use_estimator and containers:
            self.estimator.estimate_batch(containers, current_date)

if __name__ == "__main__":
    # Factory uses the estimator internally
    factory = ContainerFactory(use_estimator=True)
    containers = factory.create_containers("BOX", "Import", 1000)
    print(containers[:10])
# Or use estimator directly
# estimator = StandardDepartureEstimator()
# estimator.estimate_batch(containers, current_date)