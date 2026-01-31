# simulation/operations/terminal_manager.py (COMPLETE)
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Iterable
from datetime import datetime
import numpy as np

from simulation.core.vehicles.train import Train
from simulation.core.vehicles.truck import Truck
from simulation.core.vehicles.terminal_truck import TerminalTruck
from simulation.core.facilities.yard import BooleanStorageYard, PlacementResult
from simulation.core.facilities.railyard import BooleanRailYard
from simulation.core.facilities.parking import ParkingArea
from simulation.core.enums import MoveType
from simulation.config.operations_config import OperationsDefaults

PROXIMITY = OperationsDefaults.PROXIMITY_SEARCH_BAYS


@dataclass(frozen=True)
class Move:
    """Represents a container move operation."""
    type: MoveType
    args: Dict[str, Any]


class TerminalLogisticsManager:
    """
    Manages terminal logistics operations.
    
    Generates and executes container moves between:
    - Yard storage
    - Trains
    - Trucks
    - Terminal trucks
    """
    
    def __init__(
        self,
        yard: BooleanStorageYard,
        rail: BooleanRailYard,
        parking: Optional[ParkingArea] = None
    ):
        """
        Initialize logistics manager.
        
        Args:
            yard: Yard storage facility
            rail: Rail yard for train positioning
            parking: Truck parking area (optional)
        """
        self.yard = yard
        self.rail = rail
        self.parking = parking
        self._zone_anchors = self._compute_zone_anchors()
    
    # ----- Zone Anchors -----
    
    def _center_bay_from_mask(self, mask2d: np.ndarray) -> int:
        """Compute center bay from a 2D boolean mask."""
        cols = np.where(mask2d.any(axis=0))[0]
        if cols.size == 0:
            return self.yard.n_bays // 2
        c = int(round(cols.mean()))
        return min(max(c // self.yard.split_factor, 0), self.yard.n_bays - 1)
    
    def _compute_zone_anchors(self) -> Dict[str, int]:
        """Compute anchor bays for different container types."""
        return {
            "reefer": self._center_bay_from_mask(self.yard.reefer_mask[0]),
            "dg": self._center_bay_from_mask(self.yard.dangerous_mask[0]),
            "sb": self._center_bay_from_mask(self.yard.swapbody_mask[0]),
            "reg": self.yard.n_bays // 2
        }
    
    def _goods_anchor(self, container) -> int:
        """Get anchor bay for a container based on its goods type."""
        if container.goods_type == "Reefer":
            return self._zone_anchors["reefer"]
        if container.goods_type == "DangerousGoods":
            return self._zone_anchors["dg"]
        if getattr(container, "is_swap_body", False) or getattr(container, "is_trailer", False):
            return self._zone_anchors["sb"]
        return self._zone_anchors["reg"]
    
    def _search_goods_aware(
        self,
        container,
        anchors: Iterable[int]
    ) -> List[PlacementResult]:
        """
        Goods-aware placement search with robust fallback.
        
        Args:
            container: Container to place
            anchors: Bay anchors to search near
            
        Returns:
            List of possible placements
        """
        seen = set()
        out: List[PlacementResult] = []
        
        # 1) Search near provided anchors
        for anchor in anchors:
            dests = self.yard.search_placement_all_tiers(
                container,
                target_bay=anchor,
                max_proximity=PROXIMITY
            )
            for d in dests:
                key = (d.row, d.bay, d.tier, d.start_split)
                if key not in seen:
                    seen.add(key)
                    out.append(d)
        
        # 2) Global fallback if nothing found
        if not out:
            goods_anchor = self._goods_anchor(container)
            dests = self.yard.search_placement_all_tiers(
                container,
                target_bay=goods_anchor,
                max_proximity=self.yard.n_bays
            )
            for d in dests:
                key = (d.row, d.bay, d.tier, d.start_split)
                if key not in seen:
                    seen.add(key)
                    out.append(d)
        
        out.sort(key=lambda p: (p.tier, p.bay, p.row, p.start_split))
        return out
    
    # ----- Parking -----
    
    def _preferred_bay_for_truck(self, truck: Truck) -> Optional[int]:
        """Determine preferred parking bay for a truck."""
        # Pickup trucks: median of target container bays
        if getattr(truck, "pickup_container_ids", None):
            bays = []
            for cid in truck.pickup_container_ids:
                pl = self.yard.get_container_placement(cid)
                if pl:
                    bays.append(pl.bay)
            if bays:
                bays.sort()
                return bays[len(bays) // 2]
        
        # Delivery trucks: use goods anchor of first container
        if truck.containers:
            return self._goods_anchor(truck.containers[0])
        
        return None
    
    def list_parking_moves_active(
        self,
        active_trucks: Dict[str, Truck]
    ) -> List[Move]:
        """
        Generate parking moves for trucks without parking spots.
        
        Args:
            active_trucks: Currently active trucks in terminal
            
        Returns:
            List of parking slot moves
        """
        if not self.parking or not active_trucks:
            return []
        
        moves: List[Move] = []
        PARKING_ALLOWED_OFFSETS = (-1, 0, +1)
        
        candidates = [t for t in active_trucks.values() if t and not t.parking_spot]
        
        for truck in candidates:
            preferred_bay = self._preferred_bay_for_truck(truck)
            
            if preferred_bay is None:
                # Fallback: any free spot
                free = self.parking.iter_free()
                if free:
                    spot = free[0]
                    moves.append(Move(
                        MoveType.SLOT_TRUCK_PARKING,
                        {
                            "truck_id": truck.truck_id,
                            "spot": spot,
                            "preferred_bay": None,
                            "delta_bay": 0
                        }
                    ))
                continue
            
            placed = False
            
            # Try exact/offset bays first
            for offset in PARKING_ALLOWED_OFFSETS:
                bay = preferred_bay + offset
                if bay < 0 or bay >= self.yard.n_bays:
                    continue
                
                near_exact = self.parking.iter_free_in_bay_range(bay, bay)
                if near_exact:
                    spot = near_exact[0]
                    moves.append(Move(
                        MoveType.SLOT_TRUCK_PARKING,
                        {
                            "truck_id": truck.truck_id,
                            "spot": spot,
                            "preferred_bay": preferred_bay,
                            "delta_bay": offset
                        }
                    ))
                    placed = True
                    break
            
            # Try within ±2 bays
            if not placed:
                near = self.parking.iter_free_in_bay_range(
                    max(0, preferred_bay - 2),
                    min(self.yard.n_bays - 1, preferred_bay + 2)
                )
                if near:
                    spot = near[0]
                    bay_num = self.parking.spot_bay(spot)
                    delta = (bay_num - preferred_bay) if bay_num is not None else 0
                    moves.append(Move(
                        MoveType.SLOT_TRUCK_PARKING,
                        {
                            "truck_id": truck.truck_id,
                            "spot": spot,
                            "preferred_bay": preferred_bay,
                            "delta_bay": int(delta)
                        }
                    ))
                    placed = True
            
            # Final fallback: any free spot
            if not placed:
                free_any = self.parking.iter_free()
                if free_any:
                    spot = free_any[0]
                    moves.append(Move(
                        MoveType.SLOT_TRUCK_PARKING,
                        {
                            "truck_id": truck.truck_id,
                            "spot": spot,
                            "preferred_bay": preferred_bay,
                            "delta_bay": 0
                        }
                    ))
        
        return moves
    
    # ----- Move Listing -----
    
    def list_train_to_yard(
        self,
        train: Train,
        top_per_container: Optional[int] = None
    ) -> List[Move]:
        """
        Generate moves to unload import containers from train to yard.
        
        Args:
            train: Train to unload from
            top_per_container: Limit placements per container (None = all)
            
        Returns:
            List of TRAIN_TO_YARD moves
        """
        anchor_track = self.rail.get_anchor_bay(train.train_id)
        out: List[Move] = []
        
        for container in train.get_all_containers():
            if getattr(container, "direction", "Import") != "Import":
                continue
            
            anchors = ([anchor_track] if anchor_track is not None else []) + [
                self._goods_anchor(container)
            ]
            dests = self._search_goods_aware(container, anchors)
            
            take = dests if top_per_container is None else dests[:top_per_container]
            
            for dest in take:
                out.append(Move(
                    MoveType.TRAIN_TO_YARD,
                    {
                        "train_id": train.train_id,
                        "container_id": container.container_id,
                        "placement": dest
                    }
                ))
        
        return out
    
    def list_yard_to_train(self, train: Train) -> List[Move]:
        """
        Generate moves to load export containers from yard to train.
        
        Args:
            train: Train to load onto
            
        Returns:
            List of YARD_TO_TRAIN moves
        """
        out: List[Move] = []
        
        for container_id in train.get_all_pickup_container_ids():
            if container_id not in self.yard.accessible_containers:
                continue
            
            container = self.yard.get_container(container_id)
            if container and train.has_space_for_container(container):
                out.append(Move(
                    MoveType.YARD_TO_TRAIN,
                    {
                        "train_id": train.train_id,
                        "container_id": container_id
                    }
                ))
        
        return out
    
    def list_yard_to_yard(self) -> List[Move]:
        """
        Generate moves to relocate containers within yard.
        
        Returns:
            List of YARD_TO_YARD moves
        """
        moveable = self.yard.find_moveable_containers(max_proximity=PROXIMITY)
        out: List[Move] = []
        
        for container_id, dests in moveable.items():
            for dest in dests:
                out.append(Move(
                    MoveType.YARD_TO_YARD,
                    {
                        "container_id": container_id,
                        "placement": dest
                    }
                ))
        
        return out
    
    def list_truck_to_yard(
        self,
        truck: Truck,
        top_per_container: Optional[int] = None
    ) -> List[Move]:
        """
        Generate moves to unload delivery truck to yard.
        
        Args:
            truck: Truck to unload from
            top_per_container: Limit placements per container
            
        Returns:
            List of TRUCK_TO_YARD moves
        """
        # Require parked truck
        if not truck or not truck.parking_spot:
            return []
        if not truck.containers:
            return []
        
        out: List[Move] = []
        
        for container in truck.containers:
            dests = self._search_goods_aware(container, [self._goods_anchor(container)])
            take = dests if top_per_container is None else dests[:top_per_container]
            
            for dest in take:
                out.append(Move(
                    MoveType.TRUCK_TO_YARD,
                    {
                        "truck_id": truck.truck_id,
                        "container_id": container.container_id,
                        "placement": dest
                    }
                ))
        
        return out
    
    def list_yard_to_truck(self, truck: Truck) -> List[Move]:
        """
        Generate moves to load pickup truck from yard.
        
        Args:
            truck: Pickup truck to load
            
        Returns:
            List of YARD_TO_TRUCK moves
        """
        # Require parked truck
        if not truck or not truck.parking_spot:
            return []
        if not truck.pickup_container_ids:
            return []
        
        out: List[Move] = []
        
        for container_id in list(truck.pickup_container_ids):
            if container_id in self.yard.accessible_containers:
                container = self.yard.get_container(container_id)
                if container and truck.can_accommodate_container(container):
                    out.append(Move(
                        MoveType.YARD_TO_TRUCK,
                        {
                            "truck_id": truck.truck_id,
                            "container_id": container_id
                        }
                    ))
        
        return out
    
    def list_train_to_truck(self, train: Train, truck: Truck) -> List[Move]:
        """
        Generate moves for direct train-to-truck transfer.
        
        Args:
            train: Source train
            truck: Destination truck
            
        Returns:
            List of TRAIN_TO_TRUCK moves
        """
        # Require parked truck
        if not truck or not truck.parking_spot:
            return []
        if not truck.pickup_container_ids:
            return []
        
        out: List[Move] = []
        wanted = truck.pickup_container_ids
        
        for container in train.get_all_containers():
            if container.container_id in wanted and truck.can_accommodate_container(container):
                out.append(Move(
                    MoveType.TRAIN_TO_TRUCK,
                    {
                        "train_id": train.train_id,
                        "truck_id": truck.truck_id,
                        "container_id": container.container_id
                    }
                ))
        
        return out
    
    def list_truck_to_train(self, truck: Truck, train: Train) -> List[Move]:
        """
        Generate moves for direct truck-to-train transfer.
        
        Args:
            truck: Source truck
            train: Destination train
            
        Returns:
            List of TRUCK_TO_TRAIN moves
        """
        # Require parked truck
        if not truck or not truck.parking_spot:
            return []
        if not truck.containers:
            return []
        
        out: List[Move] = []
        wanted = train.get_all_pickup_container_ids()
        
        if not wanted:
            return out
        
        for container in truck.containers:
            if container.container_id in wanted and train.has_space_for_container(container):
                out.append(Move(
                    MoveType.TRUCK_TO_TRAIN,
                    {
                        "train_id": train.train_id,
                        "truck_id": truck.truck_id,
                        "container_id": container.container_id
                    }
                ))
        
        return out
    
    def list_yard_to_terminal_truck(self, terminal_truck: TerminalTruck) -> List[Move]:
        """
        Generate moves for terminal truck to pick up swap bodies/trailers.
        
        Args:
            terminal_truck: Available terminal truck
            
        Returns:
            List of YARD_TO_TERMINAL_TRUCK moves
        """
        out: List[Move] = []
        
        if not terminal_truck:
            return out
        
        # Check availability
        if hasattr(terminal_truck, "is_available") and not terminal_truck.is_available():
            return out
        
        for container_id in list(self.yard.accessible_containers):
            container = self.yard.get_container(container_id)
            if not container:
                continue
            
            # Only swap bodies and trailers
            if not (getattr(container, "is_swap_body", False) or 
                    getattr(container, "is_trailer", False)):
                continue
            
            out.append(Move(
                MoveType.YARD_TO_TERMINAL_TRUCK,
                {
                    "terminal_truck_id": getattr(terminal_truck, "truck_id", None),
                    "container_id": container_id
                }
            ))
        
        return out
    
    # ----- Move Execution -----
    
    def _remove_pickup_id_from_all_trucks(
        self,
        trucks: Dict[str, Truck],
        container_id: str
    ) -> None:
        """Remove a container ID from all truck pickup lists."""
        for truck in trucks.values():
            try:
                truck.remove_pickup_container_id(container_id)
            except Exception:
                pass
    
    def execute(
        self,
        move: Move,
        trains: Dict[str, Train],
        trucks: Dict[str, Truck],
        terminal_trucks: Dict[str, TerminalTruck]
    ) -> bool:
        """
        Execute a container move.
        
        Args:
            move: Move to execute
            trains: Active trains
            trucks: Active trucks
            terminal_trucks: Active terminal trucks
            
        Returns:
            True if move executed successfully, False otherwise
        """
        move_type = move.type
        args = move.args
        
        # Helper to check if truck is parked (required for crane moves)
        def _require_parked(truck_id_key: str) -> bool:
            truck = trucks.get(args.get(truck_id_key))
            return bool(truck and truck.parking_spot)
        
        # Parking moves
        if move_type == MoveType.SLOT_TRUCK_PARKING:
            if not self.parking:
                return False
            truck = trucks.get(args["truck_id"])
            spot = args["spot"]
            return bool(truck and self.parking.allocate(truck, spot))
        
        # Train <-> Yard
        if move_type == MoveType.TRAIN_TO_YARD:
            train = trains.get(args["train_id"])
            container_id = args["container_id"]
            if not train:
                return False
            container = train.remove_container(container_id)
            if not container:
                return False
            self.yard.add_container(container, args["placement"])
            return True
        
        if move_type == MoveType.YARD_TO_TRAIN:
            train = trains.get(args["train_id"])
            container_id = args["container_id"]
            if not train:
                return False
            container = self.yard.get_container(container_id)
            if not container or not train.has_space_for_container(container):
                return False
            ok = train.add_container(container)
            if not ok:
                return False
            self.yard.remove_container(container)
            train.remove_pickup_container(container_id)
            return True
        
        # Yard <-> Yard
        if move_type == MoveType.YARD_TO_YARD:
            return self.yard.move_container(args["container_id"], args["placement"])
        
        # Truck <-> Yard
        if move_type == MoveType.TRUCK_TO_YARD:
            if not _require_parked("truck_id"):
                return False
            truck = trucks.get(args["truck_id"])
            container_id = args["container_id"]
            if not truck:
                return False
            container = truck.remove_container(container_id)
            if not container:
                return False
            self.yard.add_container(container, args["placement"])
            return True
        
        if move_type == MoveType.YARD_TO_TRUCK:
            if not _require_parked("truck_id"):
                return False
            truck = trucks.get(args["truck_id"])
            container_id = args["container_id"]
            if not truck:
                return False
            container = self.yard.get_container(container_id)
            if not container or not truck.can_accommodate_container(container):
                return False
            self.yard.remove_container(container)
            if not truck.add_container(container):
                return False
            truck.remove_pickup_container_id(container_id)
            return True
        
        # Train <-> Truck
        if move_type == MoveType.TRAIN_TO_TRUCK:
            if not _require_parked("truck_id"):
                return False
            train = trains.get(args["train_id"])
            truck = trucks.get(args["truck_id"])
            container_id = args["container_id"]
            if not train or not truck:
                return False
            container = train.remove_container(container_id)
            if not container or not truck.can_accommodate_container(container):
                return False
            if not truck.add_container(container):
                return False
            truck.remove_pickup_container_id(container_id)
            return True
        
        if move_type == MoveType.TRUCK_TO_TRAIN:
            if not _require_parked("truck_id"):
                return False
            truck = trucks.get(args["truck_id"])
            train = trains.get(args["train_id"])
            container_id = args["container_id"]
            if not truck or not train:
                return False
            container = truck.remove_container(container_id)
            if not container or not train.has_space_for_container(container):
                return False
            ok = train.add_container(container)
            if not ok:
                return False
            train.remove_pickup_container(container_id)
            return True
        
        # Terminal Truck
        if move_type == MoveType.YARD_TO_TERMINAL_TRUCK:
            terminal_truck = terminal_trucks.get(args["terminal_truck_id"])
            container_id = args["container_id"]
            if not terminal_truck:
                return False
            if hasattr(terminal_truck, "is_available") and not terminal_truck.is_available():
                return False
            container = self.yard.get_container(container_id)
            if not container:
                return False
            if not (getattr(container, "is_swap_body", False) or 
                    getattr(container, "is_trailer", False)):
                return False
            self.yard.remove_container(container)
            self._remove_pickup_id_from_all_trucks(trucks, container_id)
            if not terminal_truck.add_container(container):
                return False
            return True
        
        return False