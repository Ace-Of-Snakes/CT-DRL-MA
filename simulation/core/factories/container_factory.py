# simulation/core/factories/container_factory.py
import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from scipy.stats import gaussian_kde
from simulation.core.containers.container import Container, Direction, GoodsType


class ContainerFactory:
    """
    Factory class for efficient container generation using preloaded models and data.
    Loads all necessary data once at initialization and uses vectorized operations.
    Now supports separate distributions and models for import and export operations.
    """
    
    def __init__(self, 
                 import_dist_file: str = "simulation/data/train_operator_container_type_distribution_import.json",
                 export_dist_file: str = "simulation/data/train_operator_container_type_distribution_export.json",
                 container_data_file: str = "simulation/data/container_data.csv",
                 models_folder: str = "simulation/data/models"
                 ):
        """
        Initialize factory with all necessary data loaded into memory.
        
        Args:
            import_dist_file: Path to importer distribution JSON
            export_dist_file: Path to exporter distribution JSON
            container_data_file: Path to container specifications CSV
            models_folder: Base folder containing 'import' and 'export' subfolders with KDE models
            use_estimator: Whether to apply departure estimation to containers
        """
        # Load operator distributions for both import and export
        with open(import_dist_file, 'r') as f:
            self.import_operator_dict = json.load(f)
        
        with open(export_dist_file, 'r') as f:
            self.export_operator_dict = json.load(f)
        
        # Load container specifications
        self._load_container_specs(container_data_file)
        
        # Load KDE models for both import and export containers
        import_models_folder = os.path.join(models_folder, "import")
        export_models_folder = os.path.join(models_folder, "export")
        
        self.import_kde_models = self._load_kde_models(import_models_folder)
        self.export_kde_models = self._load_kde_models(export_models_folder)
        
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
        """Load all KDE models from a folder."""
        kde_models = {}
        
        if not os.path.exists(models_folder):
            print(f"Warning: Models folder {models_folder} does not exist")
            return kde_models
        
        for file in os.listdir(models_folder):
            if file.endswith("_dwelltime_kde.pkl"):
                # Extract container type from filename
                # Handle both old format (e.g., "FEU_dwelltime_kde.pkl")
                # and potential new format (e.g., "kde_dwelltime_FEU.pkl")
                if file.startswith("kde_dwelltime_"):
                    # Old export format - skip as we'll use new structure
                    continue
                else:
                    # Standard format: CONTAINER_TYPE_dwelltime_kde.pkl
                    container_type = file.replace("_dwelltime_kde.pkl", "")
                
                file_path = os.path.join(models_folder, file)
                try:
                    with open(file_path, 'rb') as f:
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
                        print(f"Warning: Skipping {file} - unrecognized format")
                except Exception as e:
                    print(f"Warning: Could not load {file}: {e}")
                    
        return kde_models
    
    def create_containers(self, 
                         operator: str, 
                         direction: Direction,
                         n_containers: int = 1,
                         base_arrival_date: Optional[datetime] = None) -> List[Container]:
        """
        Create containers for a given operator and direction.
        Estimation removed: only scheduled (true) departure is set.
        """
        if direction == "Import":
            operator_dict = self.import_operator_dict
        else:
            operator_dict = self.export_operator_dict

        if operator not in operator_dict:
            raise ValueError(f"Unknown operator: {operator} for direction: {direction}")

        if base_arrival_date is None:
            base_arrival_date = datetime.now()

        # Sample container properties
        samples = self._sample_containers(operator, direction, n_containers)

        # Create Container objects
        containers = []
        for sample in samples:
            container = self._create_single_container(sample, direction, base_arrival_date)
            containers.append(container)

        # No estimator; return as-is
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
        # Select the appropriate operator dictionary based on direction
        if direction == "Import":
            operator_dict = self.import_operator_dict
        else:  # Export
            operator_dict = self.export_operator_dict
        
        operator_data = operator_dict[operator]
        
        # Extract container types and probabilities (filter to available types)
        all_types = list(operator_data.keys())
        valid_types = [t for t in all_types if t in self.available_container_types]
        
        if not valid_types:
            raise ValueError(f"No valid container types found for operator {operator} in {direction}")
        
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
        """Sample dwell times based on direction and container type with precision handling."""
        dwell_times = np.empty(n_samples)
        
        # Select the appropriate KDE models based on direction
        if direction == "Import":
            kde_models = self.import_kde_models
        else:  # Export
            kde_models = self.export_kde_models
        
        # Set default value for containers without KDE
        dwell_times[:] = 2.0  # Default if no KDE available
        
        # Process each unique container type
        for ct in np.unique(sampled_types):
            if ct in kde_models:
                idx = np.where(sampled_types == ct)[0]
                sampled = kde_models[ct].resample(len(idx))
                # Handle both 1D and 2D array returns from resample
                if sampled.ndim > 1:
                    dwell_times[idx] = sampled[0]
                else:
                    dwell_times[idx] = sampled
            else:
                # If no specific KDE model exists for this container type,
                # keep the default value (2.0 days)
                print(f"Warning: No KDE model found for {direction} container type {ct}, using default dwell time")
        
        # Ensure non-negative values
        dwell_times = np.maximum(dwell_times, 0)
        
        # Round short dwell times to integers to ensure perfect accuracy works correctly
        # This is critical for containers that should have perfect estimation
        short_stay_mask = dwell_times <= 2.0
        dwell_times[short_stay_mask] = np.round(dwell_times[short_stay_mask])
        
        return dwell_times
    
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
    
    def get_available_operators(self, direction: Direction) -> List[str]:
        """
        Get list of available operators for a given direction.
        
        Args:
            direction: "Import" or "Export"
            
        Returns:
            List of operator names
        """
        if direction == "Import":
            return list(self.import_operator_dict.keys())
        else:  # Export
            return list(self.export_operator_dict.keys())
    
    def get_kde_model_summary(self) -> Dict:
        """
        Get summary of loaded KDE models.
        
        Returns:
            Dictionary with model counts and available container types
        """
        return {
            "import_models": {
                "count": len(self.import_kde_models),
                "container_types": sorted(self.import_kde_models.keys())
            },
            "export_models": {
                "count": len(self.export_kde_models),
                "container_types": sorted(self.export_kde_models.keys())
            }
        }


# Example usage
if __name__ == "__main__":
    # Initialize factory with new structure
    factory = ContainerFactory()
    
    # Check loaded models
    model_summary = factory.get_kde_model_summary()
    print("KDE Models loaded:")
    print(f"  Import: {model_summary['import_models']['count']} models")
    print(f"  Export: {model_summary['export_models']['count']} models")
    
    # Check available operators
    import_operators = factory.get_available_operators("Import")
    export_operators = factory.get_available_operators("Export")
    print(f"\nAvailable operators:")
    print(f"  Import: {import_operators}")  # Show first 5
    print(f"  Export: {export_operators}")  # Show first 5
    
    # Test container generation for both directions
    print("\n=== Testing Import Containers ===")
    if import_operators:
        import_containers = factory.create_containers(
            operator=import_operators[0],
            direction="Import",
            n_containers=10000
        )
        print(f"Created {len(import_containers)} import containers")
        for c in import_containers[:3]:
            print(f"  {c.container_id}: {c.container_type}, {c.direction}, "
                  f"dwell: {(c.departure_date - c.arrival_date).days} days")
    
    print("\n=== Testing Export Containers ===")
    if export_operators:
        export_containers = factory.create_containers(
            operator=export_operators[0],
            direction="Export",
            n_containers=10000
        )
        print(f"Created {len(export_containers)} export containers")
        for c in export_containers[:3]:
            print(f"  {c.container_id}: {c.container_type}, {c.direction}, "
                  f"dwell: {(c.departure_date - c.arrival_date).days} days")