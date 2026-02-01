# simulation/operations/hierarchical_moves.py
"""
Hierarchical move generation for two-stage agent.

Provides:
- list_moveable_containers: All containers that can be moved (Stage 1)
- list_destinations_for_container: Valid destinations for container (Stage 2)
- list_parking_actions: Pending parking moves (Stage 1 alternative)
"""
from typing import List, Dict, Optional, Set
from datetime import datetime
import os

from simulation.core.facilities.yard import (
    OptimizedStorageYard, PlacementResult, ContainerRecord, EMPTY_SLOT,
)
from simulation.core.facilities.railyard import OptimizedRailYard
from simulation.core.facilities.parking import OptimizedParkingArea, ParkingSpot
from simulation.core.vehicles.train import Train
from simulation.core.vehicles.truck import Truck
from simulation.core.containers.container import Container

try:
    from simulation.core.enums import GoodsType
    REEFER_VALUES = {GoodsType.REEFER, GoodsType.REEFER.value, "Reefer"}
    DG_VALUES = {GoodsType.DANGEROUS_GOODS, GoodsType.DANGEROUS_GOODS.value, "DangerousGoods"}
except ImportError:
    REEFER_VALUES = {"Reefer"}
    DG_VALUES = {"DangerousGoods"}

from simulation.rl.features.featurizers import (
    MoveableContainer, Destination, ParkingAction,
    SourceType, DestinationType,
)

DEBUG_DESTINATIONS: bool = os.environ.get("DEBUG_DESTINATIONS", "0") == "1"

SECONDS_PER_DAY: float = 86400.0
DEFAULT_DAYS_UNTIL_DEPARTURE: float = 30.0
DEFAULT_HEAT_PROXIMITY: float = 0.5


class HierarchicalMoveGenerator:
    """Generate moves for hierarchical agent."""

    def __init__(
        self,
        yard: OptimizedStorageYard,
        rail: OptimizedRailYard,
        parking: Optional[OptimizedParkingArea],
        proximity: int = 10,
    ):
        self.yard = yard
        self.rail = rail
        self.parking = parking
        self.proximity = proximity
        self._zone_anchors = self._compute_zone_anchors()

    # ================================================================
    # Zone helpers
    # ================================================================

    def _compute_zone_anchors(self) -> Dict[str, int]:
        """Compute anchor bays for goods zones."""
        import numpy as np

        def center_from_mask(mask2d):
            cols = np.where(mask2d.any(axis=0))[0]
            if cols.size == 0:
                return self.yard.n_bays // 2
            c = int(round(cols.mean()))
            return min(max(c // self.yard.split_factor, 0), self.yard.n_bays - 1)

        return {
            "reefer": center_from_mask(self.yard.reefer_mask[0]),
            "dg": center_from_mask(self.yard.dangerous_mask[0]),
            "sb": center_from_mask(self.yard.swapbody_mask[0]),
            "reg": self.yard.n_bays // 2,
        }

    def _goods_anchor(self, container: Container) -> int:
        """Get anchor bay for container based on goods type."""
        goods = getattr(container, "goods_type", "Regular")
        if goods in REEFER_VALUES:
            return self._zone_anchors["reefer"]
        if goods in DG_VALUES:
            return self._zone_anchors["dg"]
        if getattr(container, "is_swap_body", False) or getattr(container, "is_trailer", False):
            return self._zone_anchors["sb"]
        return self._zone_anchors["reg"]

    def _container_opens_access(self, rec: ContainerRecord) -> bool:
        """Check if removing this container opens access to one below."""
        if rec.placement.tier == 0:
            return False
        below_tier = rec.placement.tier - 1
        abs_start = rec.placement.abs_start
        return self.yard.position_grid[below_tier, rec.placement.row, abs_start] != EMPTY_SLOT

    @staticmethod
    def _days_until(departure_date, current_time: Optional[datetime]) -> float:
        """Compute days until departure."""
        if not current_time or not departure_date:
            return DEFAULT_DAYS_UNTIL_DEPARTURE
        delta = (departure_date - current_time).total_seconds()
        return max(0.0, delta / SECONDS_PER_DAY)

    @staticmethod
    def _parse_parking_bay(truck: Truck) -> int:
        """Extract bay index from truck parking spot string."""
        if not isinstance(truck.parking_spot, str):
            return 0
        parsed = ParkingSpot.from_string(truck.parking_spot)
        return parsed.bay if parsed is not None else 0

    # ================================================================
    # Stage 1: list moveable containers
    # ================================================================

    def list_moveable_containers(
        self,
        trains: Dict[str, Train],
        trucks: Dict[str, Truck],
        current_time: Optional[datetime] = None,
    ) -> List[MoveableContainer]:
        """Get all containers that can be moved."""
        result: List[MoveableContainer] = []

        # 1. Yard accessible containers — direct iteration, no dict lookup
        for rec in self.yard.iter_accessible():
            container = rec.container
            pl = rec.placement
            result.append(MoveableContainer(
                container_id=container.container_id,
                source_type=SourceType.YARD,
                source_id=None,
                row=pl.row,
                bay=pl.bay,
                tier=pl.tier,
                start_split=pl.start_split,
                length_m=container.length_m,
                goods_type=getattr(container, "goods_type", "Regular"),
                is_swap_body=getattr(container, "is_swap_body", False),
                is_trailer=getattr(container, "is_trailer", False),
                days_until_departure=self._days_until(container.departure_date, current_time),
                source_anchor_bay=pl.bay,
                opens_access_below=self._container_opens_access(rec),
            ))

        # 2. Train containers (import)
        for train_id, train in trains.items():
            anchor = self.rail.get_anchor_bay(train_id) or (self.yard.n_bays // 2)
            for container in train.get_all_containers():
                if getattr(container, "direction", "Import") != "Import":
                    continue
                result.append(MoveableContainer(
                    container_id=container.container_id,
                    source_type=SourceType.TRAIN,
                    source_id=train_id,
                    row=0,
                    bay=anchor,
                    tier=0,
                    start_split=0,
                    length_m=container.length_m,
                    goods_type=getattr(container, "goods_type", "Regular"),
                    is_swap_body=getattr(container, "is_swap_body", False),
                    is_trailer=getattr(container, "is_trailer", False),
                    days_until_departure=self._days_until(container.departure_date, current_time),
                    source_anchor_bay=anchor,
                    opens_access_below=False,
                ))

        # 3. Truck containers (delivery trucks)
        for truck_id, truck in trucks.items():
            if not truck.parking_spot or not truck.containers:
                continue
            bay = self._parse_parking_bay(truck)
            for container in truck.containers:
                result.append(MoveableContainer(
                    container_id=container.container_id,
                    source_type=SourceType.TRUCK,
                    source_id=truck_id,
                    row=0,
                    bay=bay,
                    tier=0,
                    start_split=0,
                    length_m=container.length_m,
                    goods_type=getattr(container, "goods_type", "Regular"),
                    is_swap_body=getattr(container, "is_swap_body", False),
                    is_trailer=getattr(container, "is_trailer", False),
                    days_until_departure=self._days_until(container.departure_date, current_time),
                    source_anchor_bay=bay,
                    opens_access_below=False,
                ))

        return result

    # ================================================================
    # Stage 2: list destinations for a selected container
    # ================================================================

    def list_destinations_for_container(
        self,
        moveable: MoveableContainer,
        trains: Dict[str, Train],
        trucks: Dict[str, Truck],
        train_heat_bays: Optional[Set[int]] = None,
    ) -> List[Destination]:
        """Get valid destinations for a specific container."""
        container = self._get_container_object(moveable, trains, trucks)
        if not container:
            if DEBUG_DESTINATIONS:
                print(f"  DEBUG: _get_container_object returned None for "
                      f"{moveable.container_id} from {moveable.source_type}")
            return []

        result: List[Destination] = []

        if moveable.source_type == SourceType.YARD:
            result.extend(self._yard_destinations(moveable, container, train_heat_bays))
            result.extend(self._train_destinations(moveable, container, trains))
            result.extend(self._truck_destinations(moveable, container, trucks))
        elif moveable.source_type == SourceType.TRAIN:
            result.extend(self._yard_destinations(moveable, container, train_heat_bays))
            result.extend(self._truck_destinations(moveable, container, trucks))
        elif moveable.source_type == SourceType.TRUCK:
            result.extend(self._yard_destinations(moveable, container, train_heat_bays))
            result.extend(self._train_destinations(moveable, container, trains))

        if DEBUG_DESTINATIONS and not result:
            print(f"  DEBUG: No destinations for {moveable.container_id} "
                  f"from {moveable.source_type}")
        return result

    def _get_container_object(
        self,
        moveable: MoveableContainer,
        trains: Dict[str, Train],
        trucks: Dict[str, Truck],
    ) -> Optional[Container]:
        """Resolve MoveableContainer → actual Container object."""
        if moveable.source_type == SourceType.YARD:
            return self.yard.get_container(moveable.container_id)

        if moveable.source_type == SourceType.TRAIN and moveable.source_id:
            train = trains.get(moveable.source_id)
            if train:
                # Use dict lookup if available, else linear scan
                container_map = getattr(train, "_container_map", None)
                if container_map:
                    return container_map.get(moveable.container_id)
                for c in train.get_all_containers():
                    if c.container_id == moveable.container_id:
                        return c

        if moveable.source_type == SourceType.TRUCK and moveable.source_id:
            truck = trucks.get(moveable.source_id)
            if truck:
                for c in truck.containers:
                    if c.container_id == moveable.container_id:
                        return c
        return None

    # ----- Yard destinations -----

    def _yard_destinations(
        self,
        moveable: MoveableContainer,
        container: Container,
        train_heat_bays: Optional[Set[int]],
    ) -> List[Destination]:
        """Generate yard placement destinations with global fallback."""
        # Validate container length
        length_ft = getattr(container, "length_ft", None)
        n_splits = self.yard.container_length_map.get(length_ft, 0)

        if n_splits <= 0:
            # Infer from length_m
            length_m = getattr(container, "length_m", 12.2)
            inferred_ft = round(length_m * 3.28084)
            known = list(self.yard.container_length_map.keys())
            if known:
                closest = min(known, key=lambda x: abs(x - inferred_ft))
                container.length_ft = closest
                if DEBUG_DESTINATIONS:
                    print(f"  DEBUG: Inferred length_ft={closest} from length_m={length_m}")

        # Search near goods anchor
        anchor = self._goods_anchor(container)
        placements = self.yard.search_placements(
            container, target_bay=anchor, max_proximity=self.proximity
        )

        # Global fallback
        if not placements:
            placements = self.yard.search_placements(
                container, target_bay=anchor, max_proximity=self.yard.n_bays
            )

        # Filter out current position
        if moveable.source_type == SourceType.YARD:
            curr_abs = moveable.bay * self.yard.split_factor + moveable.start_split
            placements = [
                p for p in placements
                if not (p.row == moveable.row and p.tier == moveable.tier
                        and p.bay * self.yard.split_factor + p.start_split == curr_abs)
            ]

        result: List[Destination] = []
        for pl in placements:
            result.append(Destination(
                dest_type=DestinationType.YARD,
                dest_id=None,
                row=pl.row,
                bay=pl.bay,
                tier=pl.tier,
                start_split=pl.start_split,
                zone_match=self._check_zone_match(container, pl),
                heat_proximity=self._compute_heat_proximity(pl.bay, train_heat_bays),
            ))
        return result

    # ----- Train destinations -----

    def _train_destinations(
        self,
        moveable: MoveableContainer,
        container: Container,
        trains: Dict[str, Train],
    ) -> List[Destination]:
        """Generate train loading destinations."""
        if getattr(container, "direction", "Export") != "Export":
            return []

        result: List[Destination] = []
        for train_id, train in trains.items():
            if moveable.source_type == SourceType.TRAIN and moveable.source_id == train_id:
                continue
            if moveable.container_id not in train.get_all_pickup_container_ids():
                continue
            if not train.has_space_for_container(container):
                continue

            anchor = self.rail.get_anchor_bay(train_id) or (self.yard.n_bays // 2)
            result.append(Destination(
                dest_type=DestinationType.TRAIN,
                dest_id=train_id,
                row=0,
                bay=anchor,
                tier=0,
                start_split=0,
                zone_match=True,
                heat_proximity=1.0,
            ))
        return result

    # ----- Truck destinations -----

    def _truck_destinations(
        self,
        moveable: MoveableContainer,
        container: Container,
        trucks: Dict[str, Truck],
    ) -> List[Destination]:
        """Generate truck loading destinations."""
        result: List[Destination] = []
        for truck_id, truck in trucks.items():
            if moveable.source_type == SourceType.TRUCK and moveable.source_id == truck_id:
                continue
            if not truck.parking_spot:
                continue
            if moveable.container_id not in truck.pickup_container_ids:
                continue
            if not truck.can_accommodate_container(container):
                continue

            bay = self._parse_parking_bay(truck)
            result.append(Destination(
                dest_type=DestinationType.TRUCK,
                dest_id=truck_id,
                row=0,
                bay=bay,
                tier=0,
                start_split=0,
                zone_match=True,
                heat_proximity=0.8,
            ))
        return result

    # ----- Zone / heat helpers -----

    def _check_zone_match(self, container: Container, placement: PlacementResult) -> bool:
        """Check if container goods type matches placement zone."""
        goods = getattr(container, "goods_type", "Regular")
        abs_start = placement.bay * self.yard.split_factor + placement.start_split

        if goods in REEFER_VALUES:
            return bool(self.yard.reefer_mask[placement.tier, placement.row, abs_start])
        if goods in DG_VALUES:
            return bool(self.yard.dangerous_mask[placement.tier, placement.row, abs_start])
        if getattr(container, "is_swap_body", False) or getattr(container, "is_trailer", False):
            return bool(self.yard.swapbody_mask[placement.tier, placement.row, abs_start])
        return bool(self.yard.regular_mask[placement.tier, placement.row, abs_start])

    def _compute_heat_proximity(self, bay: int, heat_bays: Optional[Set[int]]) -> float:
        """Proximity to nearest train heat bay (0-1, higher = closer)."""
        if not heat_bays:
            return DEFAULT_HEAT_PROXIMITY
        min_dist = min(abs(bay - hb) for hb in heat_bays)
        return max(0.0, 1.0 - min_dist / self.yard.n_bays)

    # ================================================================
    # Parking actions
    # ================================================================

    def list_parking_actions(self, trucks: Dict[str, Truck]) -> List[ParkingAction]:
        """Get pending parking actions for unparked trucks."""
        if not self.parking:
            return []

        result: List[ParkingAction] = []
        NEAR_OFFSETS = (0, -1, 1, -2, 2)

        for truck_id, truck in trucks.items():
            if truck.parking_spot:
                continue

            preferred_bay = self._preferred_bay_for_truck(truck)

            if preferred_bay is not None:
                spot = self._find_spot_near(preferred_bay, NEAR_OFFSETS)
                if spot is not None:
                    result.append(ParkingAction(
                        truck_id=truck_id,
                        spot=spot,
                        preferred_bay=preferred_bay,
                        delta_bay=spot.bay - preferred_bay,
                    ))
                    continue

            # Fallback: any free spot
            spot = next(self.parking.iter_free(), None)
            if spot is not None:
                result.append(ParkingAction(
                    truck_id=truck_id,
                    spot=spot,
                    preferred_bay=preferred_bay,
                    delta_bay=0,
                ))

        return result

    def _find_spot_near(self, preferred_bay: int, offsets) -> Optional[ParkingSpot]:
        """Find first free parking spot near preferred bay."""
        for offset in offsets:
            bay = preferred_bay + offset
            if bay < 0 or bay >= self.yard.n_bays:
                continue
            spot = next(self.parking.iter_free_in_bay_range(bay, bay), None)
            if spot is not None:
                return spot
        return None

    def _preferred_bay_for_truck(self, truck: Truck) -> Optional[int]:
        """Determine preferred parking bay for truck."""
        if truck.pickup_container_ids:
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