# simulation/analytics/move_csv_logger.py
"""CSV logger for detailed per-move and per-vehicle-event recording.

Produces two CSV files per simulation run:
  - moves_<label>.csv   — one row per container move (crane pick-place)
  - events_<label>.csv  — one row per vehicle event (truck/train arrival/departure)

These CSVs feed the post-hoc crane movement visualiser and Phase 2 metrics.
"""
from __future__ import annotations

import csv
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional


# ── Column definitions ──────────────────────────────────────────────────

MOVE_COLUMNS = [
    "timestamp",
    "move_type",
    "container_id",
    "length_ft",
    "goods_type",
    "is_swap_body",
    "is_trailer",
    "src_region",
    "src_row",
    "src_bay",
    "src_split",
    "src_tier",
    "dst_region",
    "dst_row",
    "dst_bay",
    "dst_split",
    "dst_tier",
    "crane_id",
    "crane_time_s",
    "crane_distance_m",
]

EVENT_COLUMNS = [
    "timestamp",
    "event_type",
    "vehicle_id",
    "vehicle_type",
    "track_or_bay",
    "extra",
]


class MoveCSVLogger:
    """Lightweight CSV logger for container moves and vehicle events.

    Usage::

        logger = MoveCSVLogger("/output/dir", label="stage3_day5")
        logger.log_move(timestamp=..., move_type="TRAIN_TO_YARD", ...)
        logger.log_vehicle_event(timestamp=..., event_type="TRUCK_ARRIVAL", ...)
        logger.close()

    The logger is intentionally simple — it opens two CSV files on
    construction and appends rows as moves/events occur.  Call ``close()``
    at the end of the simulation day to flush buffers.
    """

    def __init__(self, output_dir: str, label: str = "run") -> None:
        self._dir = Path(output_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

        moves_path = self._dir / f"moves_{label}.csv"
        events_path = self._dir / f"events_{label}.csv"

        self._moves_file = open(moves_path, "w", newline="")
        self._moves_writer = csv.writer(self._moves_file)
        self._moves_writer.writerow(MOVE_COLUMNS)

        self._events_file = open(events_path, "w", newline="")
        self._events_writer = csv.writer(self._events_file)
        self._events_writer.writerow(EVENT_COLUMNS)

        self._closed = False

    # ── Move logging ─────────────────────────────────────────────────

    def log_move(
        self,
        timestamp: datetime,
        move_type: str,
        container_id: str,
        length_ft: float = 0.0,
        goods_type: str = "",
        is_swap_body: bool = False,
        is_trailer: bool = False,
        src_region: str = "",
        src_row: int = -1,
        src_bay: int = -1,
        src_split: int = -1,
        src_tier: int = -1,
        dst_region: str = "",
        dst_row: int = -1,
        dst_bay: int = -1,
        dst_split: int = -1,
        dst_tier: int = -1,
        crane_id: int = -1,
        crane_time_s: float = 0.0,
        crane_distance_m: float = 0.0,
    ) -> None:
        """Append one move row to the moves CSV."""
        if self._closed:
            return
        self._moves_writer.writerow([
            timestamp.isoformat() if timestamp else "",
            move_type,
            container_id,
            round(length_ft, 1),
            goods_type,
            int(is_swap_body),
            int(is_trailer),
            src_region,
            src_row,
            src_bay,
            src_split,
            src_tier,
            dst_region,
            dst_row,
            dst_bay,
            dst_split,
            dst_tier,
            crane_id,
            round(crane_time_s, 2),
            round(crane_distance_m, 2),
        ])

    # ── Vehicle event logging ────────────────────────────────────────

    def log_vehicle_event(
        self,
        timestamp: datetime,
        event_type: str,
        vehicle_id: str,
        vehicle_type: str = "",
        track_or_bay: str = "",
        extra: str = "",
    ) -> None:
        """Append one event row to the events CSV.

        event_type examples: TRUCK_ARRIVAL, TRUCK_DEPARTURE,
        TRAIN_ARRIVAL, TRAIN_DEPARTURE, PARK_TRUCK.
        """
        if self._closed:
            return
        self._events_writer.writerow([
            timestamp.isoformat() if timestamp else "",
            event_type,
            vehicle_id,
            vehicle_type,
            track_or_bay,
            extra,
        ])

    # ── Lifecycle ────────────────────────────────────────────────────

    def flush(self) -> None:
        """Flush both CSV files to disk."""
        if not self._closed:
            self._moves_file.flush()
            self._events_file.flush()

    def close(self) -> None:
        """Close both CSV files."""
        if self._closed:
            return
        self._closed = True
        self._moves_file.close()
        self._events_file.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            logging.debug("MoveCSVLogger: failed to close during __del__", exc_info=True)
