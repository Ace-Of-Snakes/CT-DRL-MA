# simulation/operations/crane_movements.py
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass

from simulation.operations._rmgc_math import move_time as _jit_move_time
from simulation.core.facilities.yard import BooleanStorageYard, PlacementResult
from simulation.core.facilities.railyard import BooleanRailYard
from simulation.core.vehicles.train import Train
from simulation.core.vehicles.truck import Truck
from simulation.config.crane_config import CraneGeometry, CranePerformance

# Import geometry and performance from config
# Remove all hardcoded constants


@dataclass
class RMGCMoveCost:
    """Result of crane move cost calculation."""
    distance_m: float
    time_s: float


class TerminalRMGC:
    """
    Computes crane travel distance/time for any move by mapping endpoints to (x,y,z).
    Movement model: trapezoidal per-axis profiles, critical-path timing.
    """
    
    def __init__(
        self,
        yard: BooleanStorageYard,
        rail: BooleanRailYard,
        num_tracks: int,
        geom: CraneGeometry = None,
        perf: CranePerformance = None
    ):
        """
        Initialize RMGC crane model.
        
        Args:
            yard: Yard storage facility
            rail: Rail yard
            num_tracks: Number of rail tracks
            geom: Crane geometry configuration
            perf: Crane performance configuration
        """
        self.yard = yard
        self.rail = rail
        self.num_tracks = int(max(1, num_tracks))
        self.geom = geom or CraneGeometry()
        self.perf = perf or CranePerformance()
        self._recompute_layout()
    
    def _recompute_layout(self) -> None:
        """Recompute layout positions based on current configuration."""
        # Y positions of strips (0 = rail area start)
        self.rail_y0 = 0.0
        self.parking_y = (
            self.rail_y0 +
            max(self.geom.track_width_m, self.geom.track_width_m * self.num_tracks) +
            self.geom.space_rails_to_parking_m
        )
        self.driving_y = self.parking_y + self.geom.parking_lane_width_m
        self.storage_y0 = (
            self.driving_y +
            self.geom.driving_lane_width_m +
            self.geom.space_driving_to_storage_m
        )
    
    def set_layout(
        self,
        yard: Optional[BooleanStorageYard] = None,
        rail: Optional[BooleanRailYard] = None,
        num_tracks: Optional[int] = None,
        geom: Optional[CraneGeometry] = None,
        perf: Optional[CranePerformance] = None
    ) -> None:
        """
        Update references and/or parameters and recompute layout.
        
        Args:
            yard: New yard reference
            rail: New rail reference
            num_tracks: New track count
            geom: New geometry configuration
            perf: New performance configuration
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
    
    # ----- Coordinate Calculations -----
    
    def _yard_xyz(self, placement: PlacementResult) -> Tuple[float, float, float]:
        """Get (x, y, z) coordinates for a yard placement."""
        x = (placement.bay + placement.start_split / max(1, self.yard.split_factor)) * self.geom.bay_length_m
        y = self.storage_y0 + placement.row * self.geom.slot_width_m
        z = placement.tier * self.geom.tier_height_m
        return (x, y, z)
    
    def _train_xyz(self, train: Train) -> Tuple[float, float, float]:
        """Get (x, y, z) coordinates for a train."""
        # Use slotted anchor_bay for X coordinate
        slot = self.rail.get_slot(train.train_id)
        bay = slot.anchor_bay if slot else self.yard.n_bays // 2
        x = bay * self.geom.bay_length_m
        
        # Track index to Y
        try:
            track_idx = int(train.rail_track) if train.rail_track is not None else 0
        except Exception:
            track_idx = 0
        y = self.rail_y0 + track_idx * self.geom.track_width_m
        
        z = self.geom.ground_vehicle_height_m
        return (x, y, z)
    
    def _truck_xyz(self, truck: Truck) -> Tuple[float, float, float]:
        """Get (x, y, z) coordinates for a truck."""
        bay, split = 0, 0
        if isinstance(truck.parking_spot, str):
            try:
                parts = truck.parking_spot.split("_")
                bay_str, split_str = parts[-2], parts[-1]
                bay, split = int(bay_str), int(split_str)
            except Exception:
                bay, split = 0, 0
        
        x = (bay + split / max(1, self.yard.split_factor)) * self.geom.bay_length_m
        y = self.parking_y
        z = self.geom.ground_vehicle_height_m
        return (x, y, z)
    
    def _stack_xyz(self) -> Tuple[float, float, float]:
        """Get coordinates for virtual stack area (for terminal truck moves)."""
        x = (self.yard.n_bays + 2) * self.geom.bay_length_m
        y = self.storage_y0 + (self.yard.n_rows * self.geom.slot_width_m) / 2.0
        z = 0.0
        return (x, y, z)
    
    # ----- Timing Model -----
    
    def _move_time(
        self,
        p1: Tuple[float, float, float],
        p2: Tuple[float, float, float]
    ) -> float:
        """Call Numba-compiled move time calculation."""
        return _jit_move_time(
            p1[0], p1[1], p1[2],
            p2[0], p2[1], p2[2],
            self.perf.trolley_speed_mps, self.perf.trolley_acc_mps2,
            self.perf.gantry_speed_mps, self.perf.gantry_acc_mps2,
            self.perf.hoist_speed_mps, self.perf.hoist_acc_mps2,
            self.perf.max_hook_height_m,
            self.perf.handling_time_s
        )
    
    def estimate_move_cost(
        self,
        move_type: str,
        source: Tuple[float, float, float],
        dest: Tuple[float, float, float]
    ) -> RMGCMoveCost:
        """
        Estimate distance and time for a move.
        
        Args:
            move_type: Type of move
            source: Source (x, y, z) coordinates
            dest: Destination (x, y, z) coordinates
            
        Returns:
            RMGCMoveCost with distance and time
        """
        # L1 distance approximation
        dist = abs(dest[0] - source[0]) + abs(dest[1] - source[1]) + abs(dest[2] - source[2])
        time = self._move_time(source, dest)
        return RMGCMoveCost(distance_m=dist, time_s=time)
    
    # ----- Endpoint Resolution -----
    
    def endpoints_for_move(
        self,
        move: Dict[str, Any],
        trains: Dict[str, Train],
        trucks: Dict[str, Truck],
        yard: BooleanStorageYard
    ) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
        """
        Map a move to (source_xyz, dest_xyz).
        
        Args:
            move: Move dictionary with 'type' and 'args'
            trains: Active trains
            trucks: Active trucks
            yard: Yard reference
            
        Returns:
            Tuple of (source, dest) coordinates, or None if not applicable
        """
        move_type = move["type"]
        args = move["args"]
        
        if move_type == "YARD_TO_YARD":
            container_id = args.get("container_id")
            src_placement = yard.get_container_placement(container_id)
            dst_placement: PlacementResult = args.get("placement")
            if src_placement and dst_placement:
                return self._yard_xyz(src_placement), self._yard_xyz(dst_placement)
            return None
        
        if move_type == "YARD_TO_TRAIN":
            container_id = args.get("container_id")
            train = trains.get(args.get("train_id"))
            src_placement = yard.get_container_placement(container_id)
            if train and src_placement:
                return self._yard_xyz(src_placement), self._train_xyz(train)
            return None
        
        if move_type == "TRAIN_TO_YARD":
            train = trains.get(args.get("train_id"))
            dst_placement: PlacementResult = args.get("placement")
            if train and dst_placement:
                return self._train_xyz(train), self._yard_xyz(dst_placement)
            return None
        
        if move_type == "TRUCK_TO_YARD":
            truck = trucks.get(args.get("truck_id"))
            dst_placement: PlacementResult = args.get("placement")
            if truck and truck.parking_spot and dst_placement:
                return self._truck_xyz(truck), self._yard_xyz(dst_placement)
            return None
        
        if move_type == "YARD_TO_TRUCK":
            container_id = args.get("container_id")
            truck = trucks.get(args.get("truck_id"))
            src_placement = yard.get_container_placement(container_id)
            if truck and truck.parking_spot and src_placement:
                return self._yard_xyz(src_placement), self._truck_xyz(truck)
            return None
        
        if move_type == "TRAIN_TO_TRUCK":
            train = trains.get(args.get("train_id"))
            truck = trucks.get(args.get("truck_id"))
            if train and truck and truck.parking_spot:
                return self._train_xyz(train), self._truck_xyz(truck)
            return None
        
        if move_type == "TRUCK_TO_TRAIN":
            train = trains.get(args.get("train_id"))
            truck = trucks.get(args.get("truck_id"))
            if train and truck and truck.parking_spot:
                return self._truck_xyz(truck), self._train_xyz(train)
            return None
        
        # Non-crane moves
        if move_type in ("YARD_TO_TERMINAL_TRUCK", "SLOT_TRUCK_PARKING"):
            return None
        
        return None
    
    def endpoints_and_cost_for_move(
        self,
        move: Dict[str, Any],
        trains: Dict[str, Train],
        trucks: Dict[str, Truck],
        yard: BooleanStorageYard
    ) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float], RMGCMoveCost]]:
        """
        Get endpoints and cost for a move.
        
        Returns:
            Tuple of (source, dest, cost) or None
        """
        endpoints = self.endpoints_for_move(move, trains, trucks, yard)
        if endpoints is None:
            return None
        cost = self.estimate_move_cost(move["type"], endpoints[0], endpoints[1])
        return endpoints[0], endpoints[1], cost