"""File path configuration for data and output."""
from pathlib import Path
from typing import Final

# Resolve relative to this file → simulation/config/../data = simulation/data
_SIMULATION_DIR = Path(__file__).resolve().parent.parent


class DataPaths:
    """Paths to input data files."""

    # Base directories
    BASE_DIR: Final[Path] = _SIMULATION_DIR / "data"
    MODELS_DIR: Final[Path] = BASE_DIR / "models"
    
    # Container data
    CONTAINER_CSV: Final[Path] = BASE_DIR / "container_data.csv"
    
    # Operator distributions
    IMPORT_DISTRIBUTION: Final[Path] = BASE_DIR / "train_operator_container_type_distribution_import.json"
    EXPORT_DISTRIBUTION: Final[Path] = BASE_DIR / "train_operator_container_type_distribution_export.json"
    
    # KDE Models
    IMPORT_MODELS_DIR: Final[Path] = MODELS_DIR / "import"
    EXPORT_MODELS_DIR: Final[Path] = MODELS_DIR / "export"
    
    # Truck arrival models
    TRUCK_ARRIVALS_DIR: Final[Path] = BASE_DIR / "truck_arrivals"
    TRUCK_DELIVERY_KDE: Final[Path] = TRUCK_ARRIVALS_DIR / "delivery"
    TRUCK_PICKUP_KDE: Final[Path] = TRUCK_ARRIVALS_DIR / "pickup"
    
    # Driving plan
    DRIVING_PLAN: Final[Path] = BASE_DIR / "driving_plan.json"


class OutputPaths:
    """Paths for simulation outputs."""
    
    BASE_DIR: Final[Path] = Path("runs")
    CHECKPOINTS_SUBDIR: Final[str] = "checkpoints"
    LOGS_SUBDIR: Final[str] = "logs"
    
    @classmethod
    def create_run_dir(cls, base_name: str = "run") -> Path:
        """Create timestamped run directory."""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = cls.BASE_DIR / f"{base_name}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir