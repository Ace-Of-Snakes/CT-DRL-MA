# simulation/training/eval/timing.py
"""Crane physics timing helper for feasibility computation.

Replicates the trapezoidal motion profile from
``simulation/operations/_rmgc_math.py`` in pure Python so that
scenario construction does not depend on Numba compilation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from simulation.config.crane_config import CraneGeometry, CranePerformance


# ================================================================
# Layout constants (from TerminalRMGC._recompute_layout)
# ================================================================

def _layout_y(geom: CraneGeometry, n_tracks: int = 7):
    """Compute Y-axis positions for rail / parking / storage."""
    rail_y0 = 0.0
    parking_y = rail_y0 + geom.track_width_m * n_tracks + geom.space_rails_to_parking_m
    driving_y = parking_y + geom.parking_lane_width_m
    storage_y0 = (
        driving_y + geom.driving_lane_width_m + geom.space_driving_to_storage_m
    )
    return rail_y0, parking_y, storage_y0


# ================================================================
# Trapezoidal axis time (mirrors _rmgc_math.axis_time)
# ================================================================

def _axis_time(dist: float, vmax: float, acc: float) -> float:
    """Trapezoidal motion profile time for a single axis."""
    if dist <= 0.0:
        return 0.0
    t_acc = vmax / acc
    d_acc = 0.5 * acc * t_acc * t_acc
    if dist <= 2.0 * d_acc:
        return 2.0 * math.sqrt(dist / acc)  # triangular
    return 2.0 * t_acc + (dist - 2.0 * d_acc) / vmax  # trapezoidal


def _move_time(
    x1: float, y1: float, z1: float,
    x2: float, y2: float, z2: float,
    perf: CranePerformance,
) -> float:
    """Pure-Python critical-path move time (mirrors _rmgc_math.move_time)."""
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    plane_z = perf.max_hook_height_m

    hoist_to_pickup = _axis_time(abs(plane_z - z1), perf.hoist_speed_mps, perf.hoist_acc_mps2)
    hoist_from_pickup = _axis_time(abs(plane_z - z1), perf.hoist_speed_mps, perf.hoist_acc_mps2)
    gx = _axis_time(dx, perf.gantry_speed_mps, perf.gantry_acc_mps2)
    gy = _axis_time(dy, perf.trolley_speed_mps, perf.trolley_acc_mps2)
    plane = max(gx, gy)
    hoist_to_dest = _axis_time(abs(plane_z - z2), perf.hoist_speed_mps, perf.hoist_acc_mps2)

    return hoist_to_pickup + hoist_from_pickup + plane + hoist_to_dest + perf.handling_time_s


# ================================================================
# Reference move-type times
# ================================================================

def _avg_move_time(
    move_type: str,
    avg_bay_spread: float,
    geom: CraneGeometry,
    perf: CranePerformance,
    n_tracks: int = 7,
    n_yard_rows: int = 5,
) -> float:
    """Compute average move time for a move type given a bay spread."""
    rail_y0, parking_y, storage_y0 = _layout_y(geom, n_tracks)
    mid_yard_y = storage_y0 + (n_yard_rows // 2) * geom.slot_width_m
    mid_yard_z = geom.tier_height_m  # assume avg tier ~1
    mid_train_y = rail_y0 + (n_tracks // 2) * geom.track_width_m
    train_z = geom.ground_vehicle_height_m
    truck_z = geom.ground_vehicle_height_m
    dx = avg_bay_spread * geom.bay_length_m

    if move_type == "TRAIN_TO_YARD":
        return _move_time(0, mid_train_y, train_z,
                          dx, mid_yard_y, 0.0, perf)
    elif move_type == "YARD_TO_TRAIN":
        return _move_time(0, mid_yard_y, 0.0,
                          dx, mid_train_y, train_z, perf)
    elif move_type == "PARK_TRUCK":
        # Queue → parking: minimal gantry, mostly trolley
        return _move_time(0, storage_y0 + n_yard_rows * geom.slot_width_m + 5.0,
                          truck_z, 0, parking_y, truck_z, perf)
    elif move_type == "TRUCK_TO_YARD":
        return _move_time(0, parking_y, truck_z,
                          dx * 0.3, mid_yard_y, 0.0, perf)
    elif move_type == "YARD_TO_TRUCK":
        return _move_time(0, mid_yard_y, 0.0,
                          dx * 0.3, parking_y, truck_z, perf)
    elif move_type == "YARD_TO_YARD":
        return _move_time(0, mid_yard_y, 0.0,
                          dx * 0.3, mid_yard_y, geom.tier_height_m, perf)
    else:
        # Fallback: conservative estimate
        return _move_time(0, mid_yard_y, 0.0, dx, mid_train_y, train_z, perf)


# ================================================================
# Workload estimate + feasibility computation
# ================================================================

@dataclass
class WorkloadEstimate:
    """Describes crane operations to be performed."""

    n_train_to_yard: int = 0
    n_yard_to_train: int = 0
    n_park_truck: int = 0
    n_truck_to_yard: int = 0
    n_yard_to_truck: int = 0
    n_yard_to_yard: int = 0  # estimated reshuffles
    avg_bay_spread: float = 10.0  # average X distance in bays between moves

    @property
    def total_moves(self) -> int:
        return (
            self.n_train_to_yard
            + self.n_yard_to_train
            + self.n_park_truck
            + self.n_truck_to_yard
            + self.n_yard_to_truck
            + self.n_yard_to_yard
        )

    @property
    def productive_moves(self) -> int:
        return self.total_moves - self.n_yard_to_yard


def estimate_feasible_time(
    workload: WorkloadEstimate,
    n_cranes: int = 2,
    safety_factor: float = 2.0,
    geom: Optional[CraneGeometry] = None,
    perf: Optional[CranePerformance] = None,
) -> float:
    """Return wall-clock seconds needed for the workload.

    Algorithm:
        1. Compute per-move-type average time using physics model.
        2. Sum total time for all moves.
        3. Divide by n_cranes (parallel execution).
        4. Multiply by safety_factor (agent inefficiency).

    Returns:
        Seconds from scenario start.
    """
    geom = geom or CraneGeometry()
    perf = perf or CranePerformance()
    spread = workload.avg_bay_spread

    move_counts = {
        "TRAIN_TO_YARD": workload.n_train_to_yard,
        "YARD_TO_TRAIN": workload.n_yard_to_train,
        "PARK_TRUCK": workload.n_park_truck,
        "TRUCK_TO_YARD": workload.n_truck_to_yard,
        "YARD_TO_TRUCK": workload.n_yard_to_truck,
        "YARD_TO_YARD": workload.n_yard_to_yard,
    }

    total_seconds = 0.0
    for mt, count in move_counts.items():
        if count > 0:
            t = _avg_move_time(mt, spread, geom, perf)
            total_seconds += count * t

    # Parallel crane execution
    parallel_seconds = total_seconds / max(1, n_cranes)

    return parallel_seconds * safety_factor
