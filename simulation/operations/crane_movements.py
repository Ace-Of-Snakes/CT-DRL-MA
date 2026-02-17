# simulation/operations/crane_movements.py
"""RMGC crane cost estimation with trapezoidal motion profiles."""
from typing import Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum

from simulation.operations._rmgc_math import move_time as _jit_move_time
from simulation.core.facilities.yard import OptimizedStorageYard, PlacementResult
from simulation.core.facilities.railyard import OptimizedRailYard
from simulation.core.facilities.parking import ParkingSpot
from simulation.core.vehicles.train import Train
from simulation.core.vehicles.truck import Truck
from simulation.config.crane_config import CraneGeometry, CranePerformance
from simulation.core.enums import MoveType


@dataclass(slots=True)
class RMGCMoveCost:
    """Result of crane move cost calculation."""
    distance_m: float
    time_s: float


class TerminalRMGC:
    """Computes crane travel distance/time via trapezoidal per-axis profiles."""

    def __init__(
        self,
        yard: OptimizedStorageYard,
        rail: OptimizedRailYard,
        num_tracks: int,
        geom: CraneGeometry = None,
        perf: CranePerformance = None,
    ):
        self.yard = yard
        self.rail = rail
        self.num_tracks = int(max(1, num_tracks))
        self.geom = geom or CraneGeometry()
        self.perf = perf or CranePerformance()
        self._recompute_layout()
        self._build_dispatch()

    # ----- Layout -----

    def _recompute_layout(self) -> None:
        """Recompute strip Y positions from geometry."""
        self.rail_y0 = 0.0
        self.parking_y = (
            self.rail_y0
            + max(self.geom.track_width_m, self.geom.track_width_m * self.num_tracks)
            + self.geom.space_rails_to_parking_m
        )
        self.driving_y = self.parking_y + self.geom.parking_lane_width_m
        self.storage_y0 = (
            self.driving_y
            + self.geom.driving_lane_width_m
            + self.geom.space_driving_to_storage_m
        )

    def set_layout(
        self,
        yard: Optional[OptimizedStorageYard] = None,
        rail: Optional[OptimizedRailYard] = None,
        num_tracks: Optional[int] = None,
        geom: Optional[CraneGeometry] = None,
        perf: Optional[CranePerformance] = None,
    ) -> None:
        """Update references / parameters and recompute."""
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

    # ----- Coordinate helpers -----

    def _yard_xyz(self, placement: PlacementResult) -> Tuple[float, float, float]:
        """(x, y, z) for a yard placement."""
        x = (placement.bay + placement.start_split / max(1, self.yard.split_factor)) * self.geom.bay_length_m
        y = self.storage_y0 + placement.row * self.geom.slot_width_m
        z = placement.tier * self.geom.tier_height_m
        return (x, y, z)

    def _train_xyz(self, train: Train) -> Tuple[float, float, float]:
        """(x, y, z) for a train via its rail slot."""
        slot = self.rail.get_slot(train.train_id)
        bay = slot.anchor_bay if slot else self.yard.n_bays // 2
        x = bay * self.geom.bay_length_m

        try:
            track_idx = int(train.rail_track) if train.rail_track is not None else 0
        except (ValueError, TypeError):
            track_idx = 0
        y = self.rail_y0 + track_idx * self.geom.track_width_m
        z = self.geom.ground_vehicle_height_m
        return (x, y, z)

    def _truck_xyz(self, truck: Truck) -> Tuple[float, float, float]:
        """(x, y, z) for a parked truck."""
        bay, split = 0, 0
        spot = truck.parking_spot
        if spot is not None:
            bay = getattr(spot, 'bay', 0)
            split = getattr(spot, 'split', 0)

        x = (bay + split / max(1, self.yard.split_factor)) * self.geom.bay_length_m
        y = self.parking_y
        z = self.geom.ground_vehicle_height_m
        return (x, y, z)

    def _stack_xyz(self) -> Tuple[float, float, float]:
        """Virtual stack area for terminal truck moves."""
        x = (self.yard.n_bays + 2) * self.geom.bay_length_m
        y = self.storage_y0 + (self.yard.n_rows * self.geom.slot_width_m) / 2.0
        z = 0.0
        return (x, y, z)

    # ----- Timing model -----

    def _move_time(
        self,
        p1: Tuple[float, float, float],
        p2: Tuple[float, float, float],
    ) -> float:
        """Numba-compiled critical-path move time."""
        return _jit_move_time(
            p1[0], p1[1], p1[2],
            p2[0], p2[1], p2[2],
            self.perf.trolley_speed_mps, self.perf.trolley_acc_mps2,
            self.perf.gantry_speed_mps, self.perf.gantry_acc_mps2,
            self.perf.hoist_speed_mps, self.perf.hoist_acc_mps2,
            self.perf.max_hook_height_m,
            self.perf.handling_time_s,
        )

    def estimate_move_cost(
        self,
        source: Tuple[float, float, float],
        dest: Tuple[float, float, float],
    ) -> RMGCMoveCost:
        """L1 distance + trapezoidal time."""
        dist = abs(dest[0] - source[0]) + abs(dest[1] - source[1]) + abs(dest[2] - source[2])
        time = self._move_time(source, dest)
        return RMGCMoveCost(distance_m=dist, time_s=time)

    # ----- Endpoint dispatch -----

    def _build_dispatch(self) -> None:
        """Build move-type -> resolver dispatch table."""
        self._dispatch: Dict[MoveType, Callable] = {
            MoveType.YARD_TO_YARD: self._ep_yard_to_yard,
            MoveType.YARD_TO_TRAIN: self._ep_yard_to_train,
            MoveType.TRAIN_TO_YARD: self._ep_train_to_yard,
            MoveType.TRUCK_TO_YARD: self._ep_truck_to_yard,
            MoveType.YARD_TO_TRUCK: self._ep_yard_to_truck,
            MoveType.TRAIN_TO_TRUCK: self._ep_train_to_truck,
            MoveType.TRUCK_TO_TRAIN: self._ep_truck_to_train,
            MoveType.YARD_TO_TERMINAL_TRUCK: self._ep_yard_to_terminal_truck,
        }

    def _ep_yard_to_yard(self, args, trains, trucks, yard):
        src = yard.get_placement(args["container_id"])
        dst = args.get("placement")
        return (self._yard_xyz(src), self._yard_xyz(dst)) if src and dst else None

    def _ep_yard_to_train(self, args, trains, trucks, yard):
        src = yard.get_placement(args["container_id"])
        train = trains.get(args.get("train_id"))
        return (self._yard_xyz(src), self._train_xyz(train)) if src and train else None

    def _ep_train_to_yard(self, args, trains, trucks, yard):
        train = trains.get(args.get("train_id"))
        dst = args.get("placement")
        return (self._train_xyz(train), self._yard_xyz(dst)) if train and dst else None

    def _ep_truck_to_yard(self, args, trains, trucks, yard):
        truck = trucks.get(args.get("truck_id"))
        dst = args.get("placement")
        if truck and truck.parking_spot and dst:
            return (self._truck_xyz(truck), self._yard_xyz(dst))
        return None

    def _ep_yard_to_truck(self, args, trains, trucks, yard):
        src = yard.get_placement(args["container_id"])
        truck = trucks.get(args.get("truck_id"))
        if src and truck and truck.parking_spot:
            return (self._yard_xyz(src), self._truck_xyz(truck))
        return None

    def _ep_train_to_truck(self, args, trains, trucks, yard):
        train = trains.get(args.get("train_id"))
        truck = trucks.get(args.get("truck_id"))
        if train and truck and truck.parking_spot:
            return (self._train_xyz(train), self._truck_xyz(truck))
        return None

    def _ep_truck_to_train(self, args, trains, trucks, yard):
        truck = trucks.get(args.get("truck_id"))
        train = trains.get(args.get("train_id"))
        if truck and truck.parking_spot and train:
            return (self._truck_xyz(truck), self._train_xyz(train))
        return None

    def _ep_yard_to_terminal_truck(self, args, trains, trucks, yard):
        """Yard swap body → terminal truck side storage (virtual stack area)."""
        src = yard.get_placement(args.get("container_id"))
        if src:
            return (self._yard_xyz(src), self._stack_xyz())
        return None

    # ----- Public API -----

    def endpoints_for_move(
        self,
        move,
        trains: Dict[str, Train],
        trucks: Dict[str, Truck],
        yard: OptimizedStorageYard,
    ) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
        """Resolve move -> (source_xyz, dest_xyz). Accepts Move dataclass or dict."""
        move_type = move.type if hasattr(move, "type") else move["type"]
        args = move.args if hasattr(move, "args") else move["args"]

        # Normalise string keys to MoveType enum
        if isinstance(move_type, str):
            try:
                move_type = MoveType(move_type)
            except ValueError:
                return None

        resolver = self._dispatch.get(move_type)
        if resolver is None:
            return None
        return resolver(args, trains, trucks, yard)

    def endpoints_and_cost_for_move(
        self,
        move,
        trains: Dict[str, Train],
        trucks: Dict[str, Truck],
        yard: OptimizedStorageYard,
    ) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float], RMGCMoveCost]]:
        """Resolve endpoints + compute cost in one call."""
        endpoints = self.endpoints_for_move(move, trains, trucks, yard)
        if endpoints is None:
            return None
        cost = self.estimate_move_cost(endpoints[0], endpoints[1])
        return endpoints[0], endpoints[1], cost