# simulation/operations/terminal_manager.py
"""Terminal logistics: move listing + execution against optimized facilities."""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Iterable
from datetime import datetime
import numpy as np

from simulation.core.vehicles.train import Train
from simulation.core.vehicles.truck import Truck
from simulation.core.vehicles.terminal_truck import TerminalTruck
from simulation.core.facilities.yard import OptimizedStorageYard, PlacementResult
from simulation.core.facilities.railyard import OptimizedRailYard
from simulation.core.facilities.parking import OptimizedParkingArea, ParkingSpot
from simulation.core.enums import MoveType
from simulation.config.operations_config import OperationsDefaults
from simulation.utils.direction_utils import is_import


@dataclass(frozen=True)
class Move:
    """Represents a container move operation."""
    type: MoveType
    args: Dict[str, Any]


class TerminalLogisticsManager:
    """Manages terminal logistics: generates and executes container moves."""

    def __init__(
        self,
        yard: OptimizedStorageYard,
        rail: OptimizedRailYard,
        parking: Optional[OptimizedParkingArea] = None,
    ):
        self.yard = yard
        self.rail = rail
        self.parking = parking
        self._zone_anchors = self._compute_zone_anchors()

    # ================================================================
    # Zone anchors
    # ================================================================

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
            "reg": self.yard.n_bays // 2,
        }

    def _goods_anchor(self, container) -> int:
        """Get anchor bay based on container goods type."""
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
        anchors: Iterable[int],
    ) -> List[PlacementResult]:
        """Goods-aware placement search with global fallback."""
        seen: set = set()
        out: List[PlacementResult] = []

        for anchor in anchors:
            for d in self.yard.search_placements(
                container, target_bay=anchor, max_proximity=OperationsDefaults.PROXIMITY_SEARCH_BAYS
            ):
                key = (d.row, d.bay, d.tier, d.start_split)
                if key not in seen:
                    seen.add(key)
                    out.append(d)

        # Global fallback
        if not out:
            goods_anchor = self._goods_anchor(container)
            for d in self.yard.search_placements(
                container, target_bay=goods_anchor, max_proximity=self.yard.n_bays
            ):
                key = (d.row, d.bay, d.tier, d.start_split)
                if key not in seen:
                    seen.add(key)
                    out.append(d)

        out.sort(key=lambda p: (p.tier, p.bay, p.row, p.start_split))
        return out

    # ================================================================
    # Parking
    # ================================================================

    def _preferred_bay_for_truck(self, truck: Truck) -> Optional[int]:
        """Determine preferred parking bay for a truck."""
        if getattr(truck, "pickup_container_ids", None):
            bays = []
            for cid in truck.pickup_container_ids:
                pl = self.yard.get_placement(cid)
                if pl:
                    bays.append(pl.bay)
            if bays:
                bays.sort()
                return bays[len(bays) // 2]

        if truck.containers:
            return self._goods_anchor(truck.containers[0])
        return None

    def _try_allocate_near(
        self,
        truck: Truck,
        preferred_bay: int,
        offsets: Iterable[int],
    ) -> Optional[Move]:
        """Try to find and allocate a parking spot near preferred bay."""
        for offset in offsets:
            bay = preferred_bay + offset
            if bay < 0 or bay >= self.yard.n_bays:
                continue
            spot = next(self.parking.iter_free_in_bay_range(bay, bay), None)
            if spot is not None:
                return Move(
                    MoveType.SLOT_TRUCK_PARKING,
                    {
                        "truck_id": truck.truck_id,
                        "spot": spot,
                        "preferred_bay": preferred_bay,
                        "delta_bay": offset,
                    },
                )
        return None

    def _fallback_any_spot(self, truck: Truck, preferred_bay: Optional[int]) -> Optional[Move]:
        """Allocate any free parking spot."""
        spot = next(self.parking.iter_free(), None)
        if spot is None:
            return None
        delta = (spot.bay - preferred_bay) if preferred_bay is not None else 0
        return Move(
            MoveType.SLOT_TRUCK_PARKING,
            {
                "truck_id": truck.truck_id,
                "spot": spot,
                "preferred_bay": preferred_bay,
                "delta_bay": delta,
            },
        )

    def list_parking_moves_active(self, active_trucks: Dict[str, Truck]) -> List[Move]:
        """Generate parking moves for unparked trucks."""
        if not self.parking or not active_trucks:
            return []

        NEAR_OFFSETS = (-1, 0, 1)
        WIDER_OFFSETS = (-2, -1, 0, 1, 2)
        moves: List[Move] = []

        for truck in active_trucks.values():
            if not truck or truck.parking_spot:
                continue

            preferred_bay = self._preferred_bay_for_truck(truck)

            if preferred_bay is None:
                move = self._fallback_any_spot(truck, None)
            else:
                move = (
                    self._try_allocate_near(truck, preferred_bay, NEAR_OFFSETS)
                    or self._try_allocate_near(truck, preferred_bay, WIDER_OFFSETS)
                    or self._fallback_any_spot(truck, preferred_bay)
                )

            if move is not None:
                moves.append(move)

        return moves

    # ================================================================
    # Move listing
    # ================================================================

    def list_train_to_yard(
        self,
        train: Train,
        top_per_container: Optional[int] = None,
    ) -> List[Move]:
        """Unload import containers from train to yard."""
        anchor_track = self.rail.get_anchor_bay(train.train_id)
        out: List[Move] = []

        for container in train.get_all_containers():
            if not is_import(container):
                continue

            anchors = (
                ([anchor_track] if anchor_track is not None else [])
                + [self._goods_anchor(container)]
            )
            dests = self._search_goods_aware(container, anchors)
            take = dests if top_per_container is None else dests[:top_per_container]

            for dest in take:
                out.append(Move(
                    MoveType.TRAIN_TO_YARD,
                    {
                        "train_id": train.train_id,
                        "container_id": container.container_id,
                        "placement": dest,
                    },
                ))
        return out

    def list_yard_to_train(self, train: Train) -> List[Move]:
        """Load export containers from yard onto train."""
        out: List[Move] = []
        for container_id in train.get_all_pickup_container_ids():
            if not self.yard.is_accessible(container_id):
                continue
            container = self.yard.get_container(container_id)
            if container and train.has_space_for_container(container):
                out.append(Move(
                    MoveType.YARD_TO_TRAIN,
                    {"train_id": train.train_id, "container_id": container_id},
                ))
        return out

    def list_yard_to_yard(self) -> List[Move]:
        """Generate yard-to-yard relocation moves for accessible containers."""
        out: List[Move] = []

        for rec in self.yard.iter_accessible():
            container = rec.container
            pl = rec.placement
            anchor = self._goods_anchor(container)
            dest = self.yard.find_single_placement(container, target_bay=anchor)
            if dest is None:
                continue
            # Skip same position
            if (dest.row == pl.row and dest.bay == pl.bay
                    and dest.tier == pl.tier and dest.start_split == pl.start_split):
                continue
            out.append(Move(
                MoveType.YARD_TO_YARD,
                {"container_id": container.container_id, "placement": dest},
            ))
        return out

    def list_truck_to_yard(
        self,
        truck: Truck,
        top_per_container: Optional[int] = None,
    ) -> List[Move]:
        """Unload delivery truck to yard."""
        if not truck or not truck.parking_spot or not truck.containers:
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
                        "placement": dest,
                    },
                ))
        return out

    def list_yard_to_truck(self, truck: Truck) -> List[Move]:
        """Load pickup truck from yard."""
        if not truck or not truck.parking_spot or not truck.pickup_container_ids:
            return []

        out: List[Move] = []
        for container_id in list(truck.pickup_container_ids):
            if not self.yard.is_accessible(container_id):
                continue
            container = self.yard.get_container(container_id)
            if container and truck.can_accommodate_container(container):
                out.append(Move(
                    MoveType.YARD_TO_TRUCK,
                    {"truck_id": truck.truck_id, "container_id": container_id},
                ))
        return out

    def list_train_to_truck(self, train: Train, truck: Truck) -> List[Move]:
        """Direct train-to-truck transfer."""
        if not truck or not truck.parking_spot or not truck.pickup_container_ids:
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
                        "container_id": container.container_id,
                    },
                ))
        return out

    def list_truck_to_train(self, truck: Truck, train: Train) -> List[Move]:
        """Direct truck-to-train transfer."""
        if not truck or not truck.parking_spot or not truck.containers:
            return []

        wanted = train.get_all_pickup_container_ids()
        if not wanted:
            return []

        out: List[Move] = []
        for container in truck.containers:
            if container.container_id in wanted and train.has_space_for_container(container):
                out.append(Move(
                    MoveType.TRUCK_TO_TRAIN,
                    {
                        "train_id": train.train_id,
                        "truck_id": truck.truck_id,
                        "container_id": container.container_id,
                    },
                ))
        return out

    def list_yard_to_terminal_truck(self, terminal_truck: TerminalTruck) -> List[Move]:
        """Pick up swap bodies / trailers with terminal truck."""
        if not terminal_truck:
            return []
        if hasattr(terminal_truck, "is_available") and not terminal_truck.is_available():
            return []

        out: List[Move] = []
        for rec in self.yard.iter_accessible():
            container = rec.container
            if not (getattr(container, "is_swap_body", False) or
                    getattr(container, "is_trailer", False)):
                continue
            out.append(Move(
                MoveType.YARD_TO_TERMINAL_TRUCK,
                {
                    "terminal_truck_id": getattr(terminal_truck, "truck_id", None),
                    "container_id": container.container_id,
                },
            ))
        return out

    # ================================================================
    # Move execution
    # ================================================================

    def _remove_pickup_id_from_all_trucks(
        self,
        trucks: Dict[str, Truck],
        container_id: str,
    ) -> None:
        """Remove container ID from all truck pickup lists."""
        for truck in trucks.values():
            try:
                truck.remove_pickup_container_id(container_id)
            except (AttributeError, ValueError, KeyError):
                pass

    def execute(
        self,
        move: Move,
        trains: Dict[str, Train],
        trucks: Dict[str, Truck],
        terminal_trucks: Dict[str, TerminalTruck],
    ) -> bool:
        """Execute a container move. Returns True on success."""
        mt = move.type
        args = move.args

        def _require_parked(truck_id_key: str) -> bool:
            truck = trucks.get(args.get(truck_id_key))
            return bool(truck and truck.parking_spot)

        # --- Parking ---
        if mt == MoveType.SLOT_TRUCK_PARKING:
            if not self.parking:
                return False
            truck = trucks.get(args["truck_id"])
            spot: ParkingSpot = args["spot"]
            return bool(truck and self.parking.allocate(truck, spot.bay, spot.split))

        # -- Train " Yard ------------------------------------------------
        if mt == MoveType.TRAIN_TO_YARD:
            train = trains.get(args["train_id"])
            if not train:
                return False
            container = train.remove_container(args["container_id"])
            if not container:
                return False
            self.yard.add_container(container, args["placement"])
            return True

        if mt == MoveType.YARD_TO_TRAIN:
            train = trains.get(args["train_id"])
            container_id = args["container_id"]
            if not train:
                return False
            container = self.yard.get_container(container_id)
            if not container or not train.has_space_for_container(container):
                return False
            if not train.add_container(container):
                return False
            self.yard.remove_container(container)
            train.remove_pickup_container(container_id)
            return True

        # -- Yard " Yard -------------------------------------------------
        if mt == MoveType.YARD_TO_YARD:
            return self.yard.move_container(args["container_id"], args["placement"])

        # -- Truck " Yard ------------------------------------------------
        if mt == MoveType.TRUCK_TO_YARD:
            if not _require_parked("truck_id"):
                return False
            truck = trucks[args["truck_id"]]
            container = truck.remove_container(args["container_id"])
            if not container:
                return False
            self.yard.add_container(container, args["placement"])
            return True

        if mt == MoveType.YARD_TO_TRUCK:
            if not _require_parked("truck_id"):
                return False
            truck = trucks[args["truck_id"]]
            container_id = args["container_id"]
            container = self.yard.get_container(container_id)
            if not container or not truck.can_accommodate_container(container):
                return False
            self.yard.remove_container(container)
            if not truck.add_container(container):
                return False
            truck.remove_pickup_container_id(container_id)
            return True

        # -- Train " Truck -----------------------------------------------
        if mt == MoveType.TRAIN_TO_TRUCK:
            if not _require_parked("truck_id"):
                return False
            train = trains.get(args["train_id"])
            truck = trucks[args["truck_id"]]
            if not train:
                return False
            container = train.remove_container(args["container_id"])
            if not container or not truck.can_accommodate_container(container):
                return False
            if not truck.add_container(container):
                return False
            truck.remove_pickup_container_id(args["container_id"])
            return True

        if mt == MoveType.TRUCK_TO_TRAIN:
            if not _require_parked("truck_id"):
                return False
            truck = trucks[args["truck_id"]]
            train = trains.get(args["train_id"])
            if not train:
                return False
            container = truck.remove_container(args["container_id"])
            if not container or not train.has_space_for_container(container):
                return False
            if not train.add_container(container):
                return False
            train.remove_pickup_container(args["container_id"])
            return True

        # --- Terminal truck ---
        if mt == MoveType.YARD_TO_TERMINAL_TRUCK:
            tt = terminal_trucks.get(args["terminal_truck_id"])
            container_id = args["container_id"]
            if not tt:
                return False
            if hasattr(tt, "is_available") and not tt.is_available():
                return False
            container = self.yard.get_container(container_id)
            if not container:
                return False
            if not (getattr(container, "is_swap_body", False) or
                    getattr(container, "is_trailer", False)):
                return False
            self.yard.remove_container(container)
            self._remove_pickup_id_from_all_trucks(trucks, container_id)
            if not tt.add_container(container):
                return False
            return True

        return False