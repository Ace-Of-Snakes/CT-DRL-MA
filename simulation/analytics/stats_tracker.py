# simulation/analytics/stats_tracker.py
from typing import Dict, Any, List, Tuple
from datetime import datetime
from collections import defaultdict
import csv
import os

from simulation.core.facilities.yard import BooleanStorageYard
from simulation.core.vehicles.train import Train
from simulation.core.vehicles.truck import Truck
from simulation.core.enums import MoveType
from simulation.utils.serialization import to_json_serializable

# Moved _jsonable to utils/serialization.py


class StatsTracker:
    """
    Tracks per-step moves (async NDJSON) and per-day aggregates (CSV).
    
    Records detailed metrics for:
    - Container movements
    - Train operations
    - Truck operations
    - Yard state
    - Wait times
    """
    
    # CSV column definitions (keep as class constant for clarity)
    CSV_COLUMNS = [
        "day_index", "date", "moves", "cummulative_reward",
        "inversions", "containers_in_yard", "trains_departed",
        "containers_arrived_on_train", "imports_unloaded",
        "trucks_arrived", "trucks_with_containers", "trucks_without_containers",
        "number_trucks_unloaded",
        "containers_loaded_onto_train",
        "should_be_loaded_onto_train",
        "trucks_departed",
        "min_wait_truck", "max_wait_truck", "avg_wait_truck",
        "min_wait_to_first_load", "max_wait_to_first_load", "avg_wait_to_first_load"
    ]
    
    def __init__(
        self,
        moves_path: str,
        daily_csv_path: str,
        yard: BooleanStorageYard
    ):
        """
        Initialize stats tracker.
        
        Args:
            moves_path: Path to NDJSON file for move logging
            daily_csv_path: Path to CSV file for daily summaries
            yard: Reference to the yard for state queries
        """
        os.makedirs(os.path.dirname(moves_path), exist_ok=True)
        self.yard = yard
        
        from simulation.analytics.async_logger import AsyncNDJSONLogger
        self.moves_logger = AsyncNDJSONLogger(moves_path)
        
        self.daily_csv_path = daily_csv_path
        self._ensure_csv()
        self.reset_day_aggregates()
    
    def _ensure_csv(self) -> None:
        """Create CSV file with headers if it doesn't exist."""
        if not os.path.exists(self.daily_csv_path):
            with open(self.daily_csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(self.CSV_COLUMNS)
    
    def reset_day_aggregates(self) -> None:
        """Reset daily counters at start of new day."""
        # Rewards/moves
        self.day_reward: float = 0.0
        self.day_moves: int = 0
        self.move_counts: Dict[str, int] = defaultdict(int)
        
        # Arrivals and departures
        self.trains_departed: int = 0
        self.containers_arrived_on_train: int = 0
        self.imports_unloaded: int = 0
        
        # Per-train load accounting
        self._loaded_to_train_by_train: Dict[str, int] = defaultdict(int)
        self._leftover_by_train_at_departure: Dict[str, int] = {}
        
        # Trucks
        self.trucks_arrived: int = 0
        self.trucks_with_containers: int = 0
        self.trucks_without_containers: int = 0
        self.number_trucks_unloaded: int = 0
        self.trucks_departed: int = 0
        self.truck_wait_times: List[float] = []
        self.truck_first_service_wait_times: List[float] = []
        self._truck_arrived_with_containers: Dict[str, bool] = {}
    
    def log_move(self, record: Dict[str, Any]) -> None:
        """
        Log a single move to NDJSON file.
        
        Args:
            record: Move record dictionary
        """
        try:
            safe = to_json_serializable(record)
            self.moves_logger.log(safe)
        except Exception:
            pass  # Silent fail for performance
        
        self.day_moves += 1
        self.day_reward += float(record.get("reward", 0.0))
        
        move_type = record.get("move_type", "UNKNOWN")
        self.move_counts[move_type] += 1
        
        # Track loads per train
        if move_type in (MoveType.YARD_TO_TRAIN.value, MoveType.TRUCK_TO_TRAIN.value):
            args = record.get("args", {})
            train_id = args.get("train_id")
            if train_id:
                self._loaded_to_train_by_train[train_id] += 1
        
        # Track imports unloaded
        if move_type == MoveType.TRAIN_TO_YARD.value:
            self.imports_unloaded += 1
    
    def on_train_arrival(self, train: Train) -> None:
        """Record train arrival event."""
        try:
            self.containers_arrived_on_train += int(train.get_container_count())
        except Exception:
            pass
    
    def on_train_departure(self, train_id: str, leftover_pickup_ids: int) -> None:
        """Record train departure event."""
        self.trains_departed += 1
        self._leftover_by_train_at_departure[train_id] = int(leftover_pickup_ids)
    
    def on_truck_arrival(self, truck: Truck) -> None:
        """Record truck arrival event."""
        self.trucks_arrived += 1
        has_containers = bool(truck.containers)
        
        if has_containers:
            self.trucks_with_containers += 1
        else:
            self.trucks_without_containers += 1
        
        self._truck_arrived_with_containers[truck.truck_id] = has_containers
    
    def on_truck_departure(self, truck: Truck, wait_minutes: float) -> None:
        """Record truck departure event."""
        self.trucks_departed += 1
        self.truck_wait_times.append(float(wait_minutes))
        
        if self._truck_arrived_with_containers.get(truck.truck_id, False):
            self.number_trucks_unloaded += 1
        
        # Cleanup
        self._truck_arrived_with_containers.pop(truck.truck_id, None)
    
    def on_truck_first_service(self, truck: Truck, wait_minutes: float) -> None:
        """Record wait time from truck arrival until first container received."""
        self.truck_first_service_wait_times.append(float(wait_minutes))
    
    def _compute_inversions_and_leftovers(self) -> Tuple[int, int]:
        """
        Compute yard inversions and leftover container count.
        
        Returns:
            Tuple of (inversions, containers_in_yard)
        """
        leftovers = len(self.yard.containers)
        
        # Group containers by slot
        by_slot: Dict[Tuple[int, int], List[Tuple[int, str]]] = {}
        for cid, rec in self.yard.containers.items():
            key = (rec.placement.row, rec.placement.bay)
            by_slot.setdefault(key, []).append((rec.placement.tier, cid))
        
        # Count inversions
        inversions = 0
        for (_r, _b), lst in by_slot.items():
            lst.sort(key=lambda x: x[0])  # Sort by tier
            
            # Get departure dates
            deps = []
            for _tier, cid in lst:
                c = self.yard.get_container(cid)
                if c and c.departure_date:
                    deps.append(c.departure_date)
            
            # Count inversions (earlier departure below later)
            for i in range(1, len(deps)):
                if deps[i - 1] > deps[i]:
                    inversions += 1
        
        return inversions, leftovers
    
    def write_day_summary(self, day_index: int, date: datetime) -> None:
        """
        Write daily summary to CSV.
        
        Args:
            day_index: Sequential day number
            date: Date for this day
        """
        inversions, containers_in_yard = self._compute_inversions_and_leftovers()
        
        # Truck wait statistics
        waits = sorted(self.truck_wait_times) if self.truck_wait_times else []
        min_w = waits[0] if waits else 0.0
        max_w = waits[-1] if waits else 0.0
        avg_w = (sum(waits) / len(waits)) if waits else 0.0
        
        # First service wait statistics
        waits_first = sorted(self.truck_first_service_wait_times) if self.truck_first_service_wait_times else []
        min_wf = waits_first[0] if waits_first else 0.0
        max_wf = waits_first[-1] if waits_first else 0.0
        avg_wf = (sum(waits_first) / len(waits_first)) if waits_first else 0.0
        
        # Calculate containers loaded onto trains
        containers_loaded_onto_train = (
            self.move_counts.get(MoveType.YARD_TO_TRAIN.value, 0) +
            self.move_counts.get(MoveType.TRUCK_TO_TRAIN.value, 0)
        )
        
        # Calculate should-be-loaded (loaded + leftover)
        should_be_loaded_onto_train = 0
        for train_id, leftover in self._leftover_by_train_at_departure.items():
            loaded = self._loaded_to_train_by_train.get(train_id, 0)
            should_be_loaded_onto_train += (loaded + leftover)
        
        # Write row
        with open(self.daily_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                day_index,
                date.strftime("%Y-%m-%d"),
                self.day_moves,
                f"{self.day_reward:.6f}",
                inversions,
                containers_in_yard,
                self.trains_departed,
                self.containers_arrived_on_train,
                self.imports_unloaded,
                self.trucks_arrived,
                self.trucks_with_containers,
                self.trucks_without_containers,
                self.number_trucks_unloaded,
                containers_loaded_onto_train,
                should_be_loaded_onto_train,
                self.trucks_departed,
                f"{min_w:.2f}",
                f"{max_w:.2f}",
                f"{avg_w:.2f}",
                f"{min_wf:.2f}",
                f"{max_wf:.2f}",
                f"{avg_wf:.2f}"
            ])
    
    def close(self) -> None:
        """Close the async logger."""
        self.moves_logger.close()