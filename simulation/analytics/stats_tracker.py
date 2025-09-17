# simulation/analytics/stats_tracker.py
import csv, os
from typing import Dict, Any, List, Tuple
from datetime import datetime
from collections import defaultdict
import numpy as np
from simulation.terminal_components.storage.BooleanStorage import BooleanStorageYard, PlacementResult

class StatsTracker:
    """
    Tracks per-step moves (async NDJSON) and per-day aggregates (CSV).
    - Moves: what/when, type, source/dest, ids, crane distance/time, reward.
    - Day summary: pre-marshalling inversions, left-overs, train loads/unloads,
                   truck waiting stats, move counts, cumulative reward.
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
                    "day_index","date","moves","cumulative_reward",
                    "inversions","leftovers",
                    "trains_departed","train_leftover_ids","imports_unloaded",
                    "trucks_departed","avg_wait_min","p95_wait_min","max_wait_min"
                ])

    def reset_day_aggregates(self):
        self.day_reward = 0.0
        self.day_moves = 0
        self.move_counts = defaultdict(int)
        self.train_departed = 0
        self.train_leftover_ids = 0
        self.imports_unloaded = 0
        self.truck_wait_times = []  # minutes
        self.trucks_departed = 0

    def _jsonable(self, obj):
        # PlacementResult -> dict
        if isinstance(obj, PlacementResult):
            return {
                "row": int(obj.row),
                "bay": int(obj.bay),
                "tier": int(obj.tier),
                "start_split": int(obj.start_split),
                "score": float(obj.score),
            }
        # datetime -> ISO
        if isinstance(obj, datetime):
            return obj.isoformat()
        # numpy scalar -> Python scalar
        if isinstance(obj, (np.generic,)):
            return obj.item()
        # set/tuple -> list
        if isinstance(obj, (set, tuple)):
            return [self._jsonable(x) for x in obj]
        # list -> list
        if isinstance(obj, list):
            return [self._jsonable(x) for x in obj]
        # dict -> dict
        if isinstance(obj, dict):
            return {str(k): self._jsonable(v) for k, v in obj.items()}
        # fallback
        return obj

    def log_move(self, record: Dict[str, Any]):
        safe = self._jsonable(record)
        self.moves_logger.log(safe)
        self.day_moves += 1
        self.day_reward += float(record.get("reward", 0.0))
        self.move_counts[record.get("move_type", "UNKNOWN")] += 1

    def on_train_departure(self, leftover_ids_count: int, imports_unloaded_count: int):
        self.train_departed += 1
        self.train_leftover_ids += int(leftover_ids_count)
        self.imports_unloaded += int(imports_unloaded_count)

    def on_truck_departure(self, wait_minutes: float):
        self.trucks_departed += 1
        self.truck_wait_times.append(float(wait_minutes))

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
                if not c: continue
                d = c.estimated_departure or c.departure_date
                deps.append(d)
            for i in range(1, len(deps)):
                if deps[i-1] > deps[i]:
                    inversions += 1
        return inversions, leftovers

    def write_day_summary(self, day_index: int, date: datetime):
        inv, left = self._compute_inversions_and_leftovers()
        waits = sorted(self.truck_wait_times) if self.truck_wait_times else []
        avg_w = sum(waits)/len(waits) if waits else 0.0
        p95_w = waits[int(0.95*len(waits))-1] if waits else 0.0
        max_w = waits[-1] if waits else 0.0

        with open(self.daily_csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                day_index, date.strftime("%Y-%m-%d"),
                self.day_moves, f"{self.day_reward:.6f}",
                inv, left,
                self.train_departed, self.train_leftover_ids, self.imports_unloaded,
                self.trucks_departed, f"{avg_w:.2f}", f"{p95_w:.2f}", f"{max_w:.2f}"
            ])

    def close(self):
        self.moves_logger.close()