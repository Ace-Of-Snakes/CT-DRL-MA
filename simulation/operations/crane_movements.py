# simulation/operations/crane_movements.py
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass
import math
import numpy as np

from simulation.core.facilities.yard import BooleanStorageYard, PlacementResult
from simulation.core.facilities.railyard import BooleanRailYard
from simulation.core.vehicles.train import Train
from simulation.core.vehicles.truck import Truck
from simulation.core.vehicles.terminal_truck import TerminalTruck


@dataclass
class RMGCGeometry:
    bay_length_m: float = 12.192         # 40ft in meters for yard/rail x-axis granularity
    slot_width_m: float = 2.44           # yard row spacing (y-offset between rows)
    track_width_m: float = 3.0           # distance between tracks
    space_rails_to_parking_m: float = 5.0
    parking_lane_width_m: float = 4.0
    driving_lane_width_m: float = 4.0
    space_driving_to_storage_m: float = 2.0
    tier_height_m: float = 2.59
    ground_vehicle_height_m: float = 1.5  # z-height of hooks at vehicles


@dataclass
class RMGCPerformance:
    trolley_speed_mps: float = 70.0 / 60.0
    hoist_speed_mps: float = 28.0 / 60.0
    gantry_speed_mps: float = 130.0 / 60.0
    trolley_acc_mps2: float = 0.3
    hoist_acc_mps2: float = 0.2
    gantry_acc_mps2: float = 0.1
    max_hook_height_m: float = 20.0       # z level for plane/transfer height
    handling_time_s: float = 30.0         # constant for spreader locking/unlocking etc.


@dataclass
class RMGCMoveCost:
    distance_m: float
    time_s: float


class TerminalRMGC:
    """
    Computes crane travel distance/time for any move by mapping endpoints to (x,y,z).
    Movement model: trapezoidal per-axis profiles, critical-path timing (max of axes).
    Reflects the current yard/rail layout and track count; call set_layout() when needed.
    """

    def __init__(
        self,
        yard: BooleanStorageYard,
        rail: BooleanRailYard,
        num_tracks: int,
        geom: RMGCGeometry = RMGCGeometry(),
        perf: RMGCPerformance = RMGCPerformance(),
    ):
        self.yard = yard
        self.rail = rail
        self.num_tracks = int(max(1, num_tracks))
        self.geom = geom
        self.perf = perf
        self._recompute_layout()

    def _recompute_layout(self) -> None:
        # Y positions of strips (0 = rail area start)
        self.rail_y0 = 0.0
        self.parking_y = (
            self.rail_y0
            + max(self.geom.track_width_m, self.geom.track_width_m * self.num_tracks)
            + self.geom.space_rails_to_parking_m
        )
        self.driving_y = self.parking_y + self.geom.parking_lane_width_m
        self.storage_y0 = self.driving_y + self.geom.driving_lane_width_m + self.geom.space_driving_to_storage_m

    def set_layout(
        self,
        yard: Optional[BooleanStorageYard] = None,
        rail: Optional[BooleanRailYard] = None,
        num_tracks: Optional[int] = None,
        geom: Optional[RMGCGeometry] = None,
        perf: Optional[RMGCPerformance] = None,
    ) -> None:
        """
        Update references and/or parameters and recompute dependent layout positions.
        Call this on env.reset() if needed (yard/rail references are the same objects in practice).
        """
        if yard is not None:
            self.yard = yard
        if rail is not None:
            self.rail = rail
        if num_tracks is not None:
            self.num_tracks = int(max(1, num_tracks))
        if geom is not None:
            self.geom = geom
        if perf is not None:
            self.perf = perf
        self._recompute_layout()

    # -------- coordinates --------
    def _yard_xyz(self, pl: PlacementResult) -> Tuple[float, float, float]:
        x = (pl.bay + pl.start_split / max(1, self.yard.split_factor)) * self.geom.bay_length_m
        y = self.storage_y0 + pl.row * self.geom.slot_width_m
        z = pl.tier * self.geom.tier_height_m
        return (x, y, z)

    def _train_xyz(self, train: Train) -> Tuple[float, float, float]:
        # Use slotted anchor_bay for X coordinate; fall back to yard center
        slot = self.rail.get_slot(train.train_id)
        bay = (slot.anchor_bay if slot else self.yard.n_bays // 2)
        x = bay * self.geom.bay_length_m

        # Track index to Y: distribute by track index
        try:
            track_idx = int(train.rail_track) if train.rail_track is not None else 0
        except Exception:
            track_idx = 0
        y = self.rail_y0 + track_idx * self.geom.track_width_m

        z = self.geom.ground_vehicle_height_m
        return (x, y, z)

    def _truck_xyz(self, truck: Truck) -> Tuple[float, float, float]:
        # Expect parking_spot like "P_{bay}_{split}"
        bay, split = 0, 0
        if isinstance(truck.parking_spot, str):
            try:
                _, b, s = truck.parking_spot.split("_")
                bay, split = int(b), int(s)
            except Exception:
                bay, split = 0, 0
        x = (bay + split / max(1, self.yard.split_factor)) * self.geom.bay_length_m
        y = self.parking_y
        z = self.geom.ground_vehicle_height_m
        return (x, y, z)

    def _stack_xyz(self) -> Tuple[float, float, float]:
        # Virtual stack area beyond storage x-range (for TT moves)
        x = (self.yard.n_bays + 2) * self.geom.bay_length_m
        y = self.storage_y0 + (self.yard.n_rows * self.geom.slot_width_m) / 2.0
        z = 0.0
        return (x, y, z)

    # -------- timing model --------
    def _axis_time(self, dist: float, vmax: float, acc: float) -> float:
        if dist <= 0.0:
            return 0.0
        t_acc = vmax / acc
        d_acc = 0.5 * acc * t_acc * t_acc
        if dist <= 2.0 * d_acc:
            return 2.0 * math.sqrt(dist / acc)
        return 2.0 * t_acc + (dist - 2.0 * d_acc) / vmax

    def _move_time(self, p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
        dx = abs(p2[0] - p1[0])
        dy = abs(p2[1] - p1[1])

        # Lower to pick at source z
        hoist_down = self._axis_time(
            abs(self.perf.max_hook_height_m - p1[2]), self.perf.hoist_speed_mps, self.perf.hoist_acc_mps2
        )
        # Lift up to plane height
        hoist_up = self._axis_time(
            abs(self.perf.max_hook_height_m - p1[2]), self.perf.hoist_speed_mps, self.perf.hoist_acc_mps2
        )
        # In-plane move (critical path between gantry and trolley)
        plane = max(
            self._axis_time(dx, self.perf.gantry_speed_mps, self.perf.gantry_acc_mps2),
            self._axis_time(dy, self.perf.trolley_speed_mps, self.perf.trolley_acc_mps2),
        )
        # Lower to destination z
        hoist_lower = self._axis_time(
            abs(self.perf.max_hook_height_m - p2[2]), self.perf.hoist_speed_mps, self.perf.hoist_acc_mps2
        )
        return hoist_down + hoist_up + plane + hoist_lower + self.perf.handling_time_s

    def estimate_move_cost(
        self,
        move_type: str,
        source: Tuple[float, float, float],
        dest: Tuple[float, float, float],
    ) -> RMGCMoveCost:
        # L1 distance approximation for total travel (gantry+trolley+hoist)
        dist = abs(dest[0] - source[0]) + abs(dest[1] - source[1]) + abs(dest[2] - source[2])
        t = self._move_time(source, dest)
        return RMGCMoveCost(distance_m=dist, time_s=t)

    # -------- endpoints per move --------
    def endpoints_for_move(
        self,
        move: Dict[str, Any],
        trains: Dict[str, Train],
        trucks: Dict[str, Truck],
        yard: BooleanStorageYard,
    ) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
        """
        Map a move to (source_xyz, dest_xyz). Returns None if move isn't crane-handled
        or preconditions (like parked truck) are not met.
        """
        t = move["type"]
        a = move["args"]

        if t == "YARD_TO_YARD":
            cid = a.get("container_id")
            src_pl = yard.get_container_placement(cid)
            dst_pl: PlacementResult = a.get("placement")
            if src_pl and dst_pl:
                return self._yard_xyz(src_pl), self._yard_xyz(dst_pl)
            return None

        if t == "YARD_TO_TRAIN":
            cid = a.get("container_id")
            tr = trains.get(a.get("train_id"))
            src_pl = yard.get_container_placement(cid)
            if tr and src_pl:
                return self._yard_xyz(src_pl), self._train_xyz(tr)
            return None

        if t == "TRAIN_TO_YARD":
            tr = trains.get(a.get("train_id"))
            dst_pl: PlacementResult = a.get("placement")
            if tr and dst_pl:
                return self._train_xyz(tr), self._yard_xyz(dst_pl)
            return None

        if t == "TRUCK_TO_YARD":
            trk = trucks.get(a.get("truck_id"))
            dst_pl: PlacementResult = a.get("placement")
            # Require truck to be parked for crane endpoints
            if trk and trk.parking_spot and dst_pl:
                return self._truck_xyz(trk), self._yard_xyz(dst_pl)
            return None

        if t == "YARD_TO_TRUCK":
            cid = a.get("container_id")
            trk = trucks.get(a.get("truck_id"))
            src_pl = yard.get_container_placement(cid)
            if trk and trk.parking_spot and src_pl:
                return self._yard_xyz(src_pl), self._truck_xyz(trk)
            return None

        if t == "TRAIN_TO_TRUCK":
            tr = trains.get(a.get("train_id"))
            trk = trucks.get(a.get("truck_id"))
            if tr and trk and trk.parking_spot:
                return self._train_xyz(tr), self._truck_xyz(trk)
            return None

        if t == "TRUCK_TO_TRAIN":
            tr = trains.get(a.get("train_id"))
            trk = trucks.get(a.get("truck_id"))
            if tr and trk and trk.parking_spot:
                return self._truck_xyz(trk), self._train_xyz(tr)
            return None

        if t == "YARD_TO_TERMINAL_TRUCK":
            # non-crane, handled separately by env (fixed duration)
            return None

        if t == "SLOT_TRUCK_PARKING":
            # non-crane, handled by parking allocator
            return None

        return None

    def endpoints_and_cost_for_move(
        self,
        move: Dict[str, Any],
        trains: Dict[str, Train],
        trucks: Dict[str, Truck],
        yard: BooleanStorageYard,
    ) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float], RMGCMoveCost]]:
        ep = self.endpoints_for_move(move, trains, trucks, yard)
        if ep is None:
            return None
        cost = self.estimate_move_cost(move["type"], ep[0], ep[1])
        return ep[0], ep[1], cost