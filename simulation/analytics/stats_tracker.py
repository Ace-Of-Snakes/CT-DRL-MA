# simulation/analytics/stats_tracker.py
import csv, os
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
from collections import defaultdict
import numpy as np

from simulation.terminal_components.storage.BooleanStorage import BooleanStorageYard, PlacementResult
from simulation.terminal_components.vehicles.Train import Train
from simulation.terminal_components.vehicles.Truck import Truck

YARD_TO_TRAIN = "YARD_TO_TRAIN"
TRUCK_TO_TRAIN = "TRUCK_TO_TRAIN"
TRAIN_TO_YARD = "TRAIN_TO_YARD"

class StatsTracker:
    """
    Tracks per-step moves (async NDJSON) and per-day aggregates (CSV).
    Adds the exact daily fields requested:
    day_index, date, moves, cummulative_reward, inversions, trains_departed,
    containers_arrived_on_train, trucks_arrived, trucks_with_containers,
    trucks_without_containers, number_trucks_unloaded, containers_in_yard,
    containers_loaded_onto_train, should_be_loaded_onto_train, trucks_departed,
    min_wait_truck, max_wait_truck, avg_wait_truck.
    """
    def __init__(self, moves_path: str, daily_csv_path: str, yard: BooleanStorageYard):
        os.makedirs(os.path.dirname(moves_path), exist_ok=True)
        self.yard = yard

        from simulation.analytics.async_logger import AsyncNDJSONLogger
        self.moves_logger = AsyncNDJSONLogger(moves_path)

        self.daily_csv_path = daily_csv_path
        self._ensure_csv()
        self.reset_day_aggregates()

    def _ensure_csv(self):
        if not os.path.exists(self.daily_csv_path):
            with open(self.daily_csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "day_index","date","moves","cummulative_reward",
                    "inversions","containers_in_yard","trains_departed",
                    "containers_arrived_on_train",
                    "trucks_arrived","trucks_with_containers","trucks_without_containers",
                    "number_trucks_unloaded",
                    "containers_loaded_onto_train",
                    "should_be_loaded_onto_train",
                    "trucks_departed","min_wait_truck","max_wait_truck","avg_wait_truck"
                ])

    def reset_day_aggregates(self):
        # rewards/moves
        self.day_reward: float = 0.0
        self.day_moves: int = 0
        self.move_counts = defaultdict(int)

        # arrivals and departures
        self.trains_departed: int = 0
        self.containers_arrived_on_train: int = 0

        # per-train load accounting for "should have been loaded"
        self._loaded_to_train_by_train: Dict[str, int] = defaultdict(int)
        self._leftover_by_train_at_departure: Dict[str, int] = {}

        # trucks
        self.trucks_arrived: int = 0
        self.trucks_with_containers: int = 0
        self.trucks_without_containers: int = 0
        self.number_trucks_unloaded: int = 0  # trucks that arrived with containers and departed empty (i.e., unloaded)
        self.trucks_departed: int = 0
        self.truck_wait_times: List[float] = []
        self._truck_arrived_with_containers: Dict[str, bool] = {}

    # --- JSON-safety helpers (kept small/fast) ---
    def _jsonable(self, obj):
        if isinstance(obj, PlacementResult):
            return {
                "row": int(obj.row),
                "bay": int(obj.bay),
                "tier": int(obj.tier),
                "start_split": int(obj.start_split),
                "score": float(obj.score),
            }
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, (np.generic,)):
            return obj.item()
        if isinstance(obj, (set, tuple)):
            return [self._jsonable(x) for x in obj]
        if isinstance(obj, list):
            return [self._jsonable(x) for x in obj]
        if isinstance(obj, dict):
            return {str(k): self._jsonable(v) for k, v in obj.items()}
        return obj

    # --- Move logging (called per executed move) ---
    def log_move(self, record: Dict[str, Any]):
        # async NDJSON
        try:
            safe = self._jsonable(record)
            self.moves_logger.log(safe)
        except Exception:
            pass

        self.day_moves += 1
        self.day_reward += float(record.get("reward", 0.0))
        mt = record.get("move_type", "UNKNOWN")
        self.move_counts[mt] += 1

        # Track loads per train for "should_be_loaded_onto_train"
        if mt in (YARD_TO_TRAIN, TRUCK_TO_TRAIN):
            args = record.get("args", {})
            tid = args.get("train_id")
            if tid:
                self._loaded_to_train_by_train[tid] += 1

    # --- Train events ---
    def on_train_arrival(self, train: Train):
        try:
            self.containers_arrived_on_train += int(train.get_container_count())
        except Exception:
            pass

    def on_train_departure(self, train_id: str, leftover_pickup_ids: int):
        self.trains_departed += 1
        self._leftover_by_train_at_departure[train_id] = int(leftover_pickup_ids)

    # --- Truck events ---
    def on_truck_arrival(self, truck: Truck):
        self.trucks_arrived += 1
        has = bool(truck.containers)
        if has:
            self.trucks_with_containers += 1
        else:
            self.trucks_without_containers += 1
        self._truck_arrived_with_containers[truck.truck_id] = has

    def on_truck_departure(self, truck: Truck, wait_minutes: float):
        self.trucks_departed += 1
        self.truck_wait_times.append(float(wait_minutes))
        if self._truck_arrived_with_containers.get(truck.truck_id, False):
            # it arrived with containers and is now leaving (i.e. unloading completed)
            self.number_trucks_unloaded += 1
        # cleanup
        self._truck_arrived_with_containers.pop(truck.truck_id, None)

    # --- Yard metrics ---
    def _compute_inversions_and_leftovers(self) -> Tuple[int, int]:
        leftovers = len(self.yard.containers)
        by_slot = {}
        for cid, rec in self.yard.containers.items():
            key = (rec.placement.row, rec.placement.bay)
            by_slot.setdefault(key, []).append((rec.placement.tier, cid))
        inversions = 0
        for (_r,_b), lst in by_slot.items():
            lst.sort(key=lambda x: x[0])
            deps = []
            for _tier, cid in lst:
                c = self.yard.get_container(cid)
                if not c: 
                    continue
                d = c.estimated_departure or c.departure_date
                deps.append(d)
            for i in range(1, len(deps)):
                if deps[i-1] > deps[i]:
                    inversions += 1
        return inversions, leftovers

    # --- Daily flush ---
    def write_day_summary(self, day_index: int, date: datetime):
        inversions, containers_in_yard = self._compute_inversions_and_leftovers()

        waits = sorted(self.truck_wait_times) if self.truck_wait_times else []
        min_w = waits[0] if waits else 0.0
        max_w = waits[-1] if waits else 0.0
        avg_w = (sum(waits) / len(waits)) if waits else 0.0

        # Move-derived counts
        containers_loaded_onto_train = (
            self.move_counts.get(YARD_TO_TRAIN, 0) + self.move_counts.get(TRUCK_TO_TRAIN, 0)
        )
        # TRAIN_TO_YARD moves answer "How many containers were deloaded from the trains?"
        # If you want this column too, you can add it similarly.
        # imports_unloaded = self.move_counts.get(TRAIN_TO_YARD, 0)

        # "Should have been loaded": for all trains that departed today sum loaded + leftover
        should_be_loaded_onto_train = 0
        for tid, leftover in self._leftover_by_train_at_departure.items():
            loaded = self._loaded_to_train_by_train.get(tid, 0)
            should_be_loaded_onto_train += (loaded + leftover)

        with open(self.daily_csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                day_index, date.strftime("%Y-%m-%d"),
                self.day_moves, f"{self.day_reward:.6f}",
                inversions, containers_in_yard, self.trains_departed,
                self.containers_arrived_on_train,
                self.trucks_arrived, self.trucks_with_containers, self.trucks_without_containers,
                self.number_trucks_unloaded,
                containers_loaded_onto_train,
                should_be_loaded_onto_train,
                self.trucks_departed,
                f"{min_w:.2f}", f"{max_w:.2f}", f"{avg_w:.2f}"
            ])

    def close(self):
        self.moves_logger.close()