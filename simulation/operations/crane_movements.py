# simulation/terminal_components/systems/RMGC_v2.py
from typing import Dict, Tuple, Optional
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
    slot_width_m: float = 2.44           # yard row spacing
    track_width_m: float = 3.0
    space_rails_to_parking_m: float = 5.0
    parking_lane_width_m: float = 4.0
    driving_lane_width_m: float = 4.0
    space_driving_to_storage_m: float = 2.0
    tier_height_m: float = 2.59
    ground_vehicle_height_m: float = 1.5

@dataclass
class RMGCPerformance:
    trolley_speed_mps: float = 70.0/60.0
    hoist_speed_mps: float = 28.0/60.0
    gantry_speed_mps: float = 130.0/60.0
    trolley_acc_mps2: float = 0.3
    hoist_acc_mps2: float = 0.2
    gantry_acc_mps2: float = 0.1
    max_hook_height_m: float = 20.0

@dataclass
class RMGCMoveCost:
    distance_m: float
    time_s: float

class TerminalRMGC:
    """
    Computes crane travel distance/time for any move by mapping endpoints to (x,y,z).
    Movement model: trapezoidal per-axis profiles, critical-path timing (max of axes).
    """
    def __init__(self,
                 yard: BooleanStorageYard,
                 rail: BooleanRailYard,
                 num_tracks: int,
                 geom: RMGCGeometry = RMGCGeometry(),
                 perf: RMGCPerformance = RMGCPerformance()):
        self.yard = yard
        self.rail = rail
        self.num_tracks = num_tracks
        self.geom = geom
        self.perf = perf
        # Y positions of strips (0 = rail area start)
        self.rail_y0 = 0.0
        self.parking_y = self.rail_y0 + max(geom.track_width_m, geom.track_width_m * num_tracks) + geom.space_rails_to_parking_m
        self.driving_y = self.parking_y + geom.parking_lane_width_m
        self.storage_y0 = self.driving_y + geom.driving_lane_width_m + geom.space_driving_to_storage_m

    # -------- coordinates --------
    def _yard_xyz(self, pl: PlacementResult) -> Tuple[float, float, float]:
        x = (pl.bay + pl.start_split / self.yard.split_factor) * self.geom.bay_length_m
        y = self.storage_y0 + pl.row * self.geom.slot_width_m
        z = pl.tier * self.geom.tier_height_m
        return (x, y, z)

    def _train_xyz(self, train: Train) -> Tuple[float, float, float]:
        slot = self.rail.get_slot(train.train_id)
        bay = (slot.anchor_bay if slot else self.yard.n_bays // 2)
        x = bay * self.geom.bay_length_m
        # Track index to Y: distribute tracks across track width lines
        track_idx = int(train.rail_track) if train.rail_track and train.rail_track.isdigit() else 0
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
            except:
                pass
        x = (bay + split / self.yard.split_factor) * self.geom.bay_length_m
        y = self.parking_y
        z = self.geom.ground_vehicle_height_m
        return (x, y, z)

    def _stack_xyz(self) -> Tuple[float, float, float]:
        # Virtual stack area beyond storage x-range
        x = (self.yard.n_bays + 2) * self.geom.bay_length_m
        y = self.storage_y0 + (self.yard.n_rows * self.geom.slot_width_m) / 2.0
        z = 0.0
        return (x, y, z)

    # -------- timing model --------
    def _axis_time(self, dist: float, vmax: float, acc: float) -> float:
        if dist <= 0: return 0.0
        t_acc = vmax / acc
        d_acc = 0.5 * acc * t_acc * t_acc
        if dist <= 2*d_acc:
            return 2.0 * math.sqrt(dist / acc)
        return 2.0 * t_acc + (dist - 2*d_acc) / vmax

    def _move_time(self, p1: Tuple[float,float,float], p2: Tuple[float,float,float]) -> float:
        dx = abs(p2[0] - p1[0])
        dy = abs(p2[1] - p1[1])
        # Lower to pick
        hoist_down = self._axis_time(abs(self.perf.max_hook_height_m - p1[2]), self.perf.hoist_speed_mps, self.perf.hoist_acc_mps2)
        # Lift up then move plane then lower
        hoist_up = self._axis_time(abs(self.perf.max_hook_height_m - p1[2]), self.perf.hoist_speed_mps, self.perf.hoist_acc_mps2)
        plane = max(self._axis_time(dx, self.perf.gantry_speed_mps, self.perf.gantry_acc_mps2),
                    self._axis_time(dy, self.perf.trolley_speed_mps, self.perf.trolley_acc_mps2))
        hoist_lower = self._axis_time(abs(self.perf.max_hook_height_m - p2[2]), self.perf.hoist_speed_mps, self.perf.hoist_acc_mps2)
        handling = 30.0  # s
        return hoist_down + hoist_up + plane + hoist_lower + handling

    def estimate_move_cost(self,
                           move_type: str,
                           source: Tuple[float,float,float],
                           dest: Tuple[float,float,float]) -> RMGCMoveCost:
        # L1 distance for simplicity (close to gantry+trolley path sum)
        dist = abs(dest[0]-source[0]) + abs(dest[1]-source[1]) + abs(dest[2]-source[2])
        t = self._move_time(source, dest)
        return RMGCMoveCost(distance_m=dist, time_s=t)

    # -------- helpers to get endpoints per move --------
    def endpoints_for_move(self,
                           move: Dict[str, any],
                           trains: Dict[str, Train],
                           trucks: Dict[str, Truck],
                           yard: BooleanStorageYard) -> Optional[Tuple[Tuple[float,float,float], Tuple[float,float,float]]]:
        t = move["type"]
        a = move["args"]
        if t in ("YARD_TO_YARD",):
            cid = a["container_id"]
            src_rec = yard.get_container_placement(cid)
            dst_pl: PlacementResult = a["placement"]
            if src_rec:
                return self._yard_xyz(src_rec), self._yard_xyz(dst_pl)
            return None
        if t in ("YARD_TO_TRAIN",):
            cid = a["container_id"]
            tr = trains.get(a["train_id"])
            src_rec = yard.get_container_placement(cid)
            if tr and src_rec:
                return self._yard_xyz(src_rec), self._train_xyz(tr)
            return None
        if t in ("TRAIN_TO_YARD",):
            tr = trains.get(a["train_id"])
            dst_pl: PlacementResult = a["placement"]
            if tr and dst_pl:
                return self._train_xyz(tr), self._yard_xyz(dst_pl)
            return None
        if t in ("TRUCK_TO_YARD",):
            trk = trucks.get(a["truck_id"])
            dst_pl: PlacementResult = a["placement"]
            if trk and dst_pl:
                return self._truck_xyz(trk), self._yard_xyz(dst_pl)
            return None
        if t in ("YARD_TO_TRUCK",):
            cid = a["container_id"]
            trk = trucks.get(a["truck_id"])
            src_rec = yard.get_container_placement(cid)
            if trk and src_rec:
                return self._yard_xyz(src_rec), self._truck_xyz(trk)
            return None
        if t in ("TRAIN_TO_TRUCK",):
            tr = trains.get(a["train_id"])
            trk = trucks.get(a["truck_id"])
            if tr and trk:
                return self._train_xyz(tr), self._truck_xyz(trk)
            return None
        if t in ("TRUCK_TO_TRAIN",):
            tr = trains.get(a["train_id"])
            trk = trucks.get(a["truck_id"])
            if tr and trk:
                return self._truck_xyz(trk), self._train_xyz(tr)
            return None
        if t in ("YARD_TO_TERMINAL_TRUCK",):
            cid = a["container_id"]
            src_rec = yard.get_container_placement(cid)
            if src_rec:
                return self._yard_xyz(src_rec), self._stack_xyz()
            return None
        return None