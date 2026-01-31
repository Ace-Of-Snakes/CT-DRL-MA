# simulation/operations/hierarchical_moves.py
"""
Hierarchical move generation for two-stage agent.

Provides:
- list_moveable_containers: All containers that can be moved (Stage 1)
- list_destinations_for_container: Valid destinations for specific container (Stage 2)
- list_parking_actions: Pending parking moves (Stage 1 alternative)
"""
from typing import List, Dict, Optional, Set
from datetime import datetime

from simulation.core.facilities.yard import BooleanStorageYard, PlacementResult
from simulation.core.facilities.railyard import BooleanRailYard
from simulation.core.facilities.parking import ParkingArea
from simulation.core.vehicles.train import Train
from simulation.core.vehicles.truck import Truck
from simulation.core.containers.container import Container
from simulation.rl.features.featurizers import (
    MoveableContainer, Destination, ParkingAction,
    SourceType, DestinationType
)


class HierarchicalMoveGenerator:
    """Generate moves for hierarchical agent."""
    
    def __init__(
        self,
        yard: BooleanStorageYard,
        rail: BooleanRailYard,
        parking: Optional[ParkingArea],
        proximity: int = 5
    ):
        self.yard = yard
        self.rail = rail
        self.parking = parking
        self.proximity = proximity
        
        # Precompute zone anchors
        self._zone_anchors = self._compute_zone_anchors()
    
    def _compute_zone_anchors(self) -> Dict[str, int]:
        """Compute anchor bays for goods zones."""
        def center_from_mask(mask2d):
            import numpy as np
            cols = np.where(mask2d.any(axis=0))[0]
            if cols.size == 0:
                return self.yard.n_bays // 2
            c = int(round(cols.mean()))
            return min(max(c // self.yard.split_factor, 0), self.yard.n_bays - 1)
        
        return {
            "reefer": center_from_mask(self.yard.reefer_mask[0]),
            "dg": center_from_mask(self.yard.dangerous_mask[0]),
            "sb": center_from_mask(self.yard.swapbody_mask[0]),
            "reg": self.yard.n_bays // 2
        }
    
    def _goods_anchor(self, container: Container) -> int:
        """Get anchor bay for container based on goods type."""
        if container.goods_type == "Reefer":
            return self._zone_anchors["reefer"]
        if container.goods_type == "DangerousGoods":
            return self._zone_anchors["dg"]
        if getattr(container, "is_swap_body", False) or getattr(container, "is_trailer", False):
            return self._zone_anchors["sb"]
        return self._zone_anchors["reg"]
    
    def _container_opens_access(self, container_id: str) -> bool:
        """Check if removing container opens access to one below."""
        rec = self.yard.containers.get(container_id)
        if not rec or rec.placement.tier == 0:
            return False
        
        # Check if there's a container directly below
        below_key = (rec.placement.row, rec.placement.tier - 1, 
                     rec.placement.bay * self.yard.split_factor + rec.placement.start_split)
        return below_key in self.yard.position_map
    
    def list_moveable_containers(
        self,
        trains: Dict[str, Train],
        trucks: Dict[str, Truck],
        current_time: Optional[datetime] = None
    ) -> List[MoveableContainer]:
        """
        Get all containers that can be moved.
        
        Sources:
        - Yard: accessible containers
        - Trains: containers on trains (for unloading)
        - Trucks: containers on delivery trucks (for unloading)
        
        Returns:
            List of MoveableContainer objects
        """
        result: List[MoveableContainer] = []
        
        # 1. Yard accessible containers
        for cid in self.yard.accessible_containers:
            rec = self.yard.containers.get(cid)
            if not rec:
                continue
            
            container = rec.container
            days_until = 30.0
            if current_time and container.departure_date:
                delta = (container.departure_date - current_time).total_seconds()
                days_until = max(0.0, delta / 86400.0)
            
            result.append(MoveableContainer(
                container_id=cid,
                source_type=SourceType.YARD,
                source_id=None,
                row=rec.placement.row,
                bay=rec.placement.bay,
                tier=rec.placement.tier,
                start_split=rec.placement.start_split,
                length_m=container.length_m,
                goods_type=container.goods_type if hasattr(container, 'goods_type') else "Regular",
                is_swap_body=getattr(container, "is_swap_body", False),
                is_trailer=getattr(container, "is_trailer", False),
                days_until_departure=days_until,
                source_anchor_bay=rec.placement.bay,
                opens_access_below=self._container_opens_access(cid)
            ))
        
        # 2. Train containers (import direction for unloading)
        for train_id, train in trains.items():
            anchor = self.rail.get_anchor_bay(train_id) or (self.yard.n_bays // 2)
            
            for container in train.get_all_containers():
                # Only import containers can be unloaded
                if getattr(container, "direction", "Import") != "Import":
                    continue
                
                days_until = 30.0
                if current_time and container.departure_date:
                    delta = (container.departure_date - current_time).total_seconds()
                    days_until = max(0.0, delta / 86400.0)
                
                result.append(MoveableContainer(
                    container_id=container.container_id,
                    source_type=SourceType.TRAIN,
                    source_id=train_id,
                    row=0,
                    bay=anchor,
                    tier=0,
                    start_split=0,
                    length_m=container.length_m,
                    goods_type=container.goods_type if hasattr(container, 'goods_type') else "Regular",
                    is_swap_body=getattr(container, "is_swap_body", False),
                    is_trailer=getattr(container, "is_trailer", False),
                    days_until_departure=days_until,
                    source_anchor_bay=anchor,
                    opens_access_below=False
                ))
        
        # 3. Truck containers (delivery trucks for unloading)
        for truck_id, truck in trucks.items():
            if not truck.parking_spot:
                continue  # Truck must be parked
            if not truck.containers:
                continue
            
            # Parse parking spot for bay
            try:
                parts = truck.parking_spot.split("_")
                bay = int(parts[-2])
            except:
                bay = self.yard.n_bays // 2
            
            for container in truck.containers:
                days_until = 30.0
                if current_time and container.departure_date:
                    delta = (container.departure_date - current_time).total_seconds()
                    days_until = max(0.0, delta / 86400.0)
                
                result.append(MoveableContainer(
                    container_id=container.container_id,
                    source_type=SourceType.TRUCK,
                    source_id=truck_id,
                    row=0,
                    bay=bay,
                    tier=0,
                    start_split=0,
                    length_m=container.length_m,
                    goods_type=container.goods_type if hasattr(container, 'goods_type') else "Regular",
                    is_swap_body=getattr(container, "is_swap_body", False),
                    is_trailer=getattr(container, "is_trailer", False),
                    days_until_departure=days_until,
                    source_anchor_bay=bay,
                    opens_access_below=False
                ))
        
        return result
    
    def list_destinations_for_container(
        self,
        moveable: MoveableContainer,
        trains: Dict[str, Train],
        trucks: Dict[str, Truck],
        train_heat_bays: Optional[Set[int]] = None
    ) -> List[Destination]:
        """
        Get valid destinations for a specific container.
        
        Destinations depend on source:
        - From YARD: can go to YARD (reshuffle), TRAIN (load export), TRUCK (pickup)
        - From TRAIN: can go to YARD (unload import), TRUCK (direct transfer)
        - From TRUCK: can go to YARD (delivery), TRAIN (direct transfer)
        
        Args:
            moveable: The container to find destinations for
            trains: Active trains
            trucks: Active trucks
            train_heat_bays: Bays near trains (for heat proximity)
            
        Returns:
            List of Destination objects
        """
        result: List[Destination] = []
        container = self._get_container_object(moveable, trains, trucks)
        if not container:
            return result
        
        # Determine valid destination types based on source
        if moveable.source_type == SourceType.YARD:
            # Yard container: reshuffle, load to train, load to truck
            result.extend(self._yard_destinations(moveable, container, train_heat_bays))
            result.extend(self._train_destinations(moveable, container, trains))
            result.extend(self._truck_destinations(moveable, container, trucks))
            
        elif moveable.source_type == SourceType.TRAIN:
            # Train container: unload to yard, direct to truck
            result.extend(self._yard_destinations(moveable, container, train_heat_bays))
            result.extend(self._truck_destinations(moveable, container, trucks))
            
        elif moveable.source_type == SourceType.TRUCK:
            # Truck container: delivery to yard, direct to train
            result.extend(self._yard_destinations(moveable, container, train_heat_bays))
            result.extend(self._train_destinations(moveable, container, trains))
        
        return result
    
    def _get_container_object(
        self,
        moveable: MoveableContainer,
        trains: Dict[str, Train],
        trucks: Dict[str, Truck]
    ) -> Optional[Container]:
        """Get actual Container object from moveable."""
        if moveable.source_type == SourceType.YARD:
            return self.yard.get_container(moveable.container_id)
        elif moveable.source_type == SourceType.TRAIN and moveable.source_id:
            train = trains.get(moveable.source_id)
            if train:
                for c in train.get_all_containers():
                    if c.container_id == moveable.container_id:
                        return c
        elif moveable.source_type == SourceType.TRUCK and moveable.source_id:
            truck = trucks.get(moveable.source_id)
            if truck:
                for c in truck.containers:
                    if c.container_id == moveable.container_id:
                        return c
        return None
    
    def _yard_destinations(
        self,
        moveable: MoveableContainer,
        container: Container,
        train_heat_bays: Optional[Set[int]]
    ) -> List[Destination]:
        """Generate yard placement destinations."""
        result: List[Destination] = []
        
        # Search near goods anchor
        anchor = self._goods_anchor(container)
        placements = self.yard.search_placement_all_tiers(
            container, target_bay=anchor, max_proximity=self.proximity
        )
        
        # Filter out current position if from yard
        if moveable.source_type == SourceType.YARD:
            curr_abs = moveable.bay * self.yard.split_factor + moveable.start_split
            placements = [
                p for p in placements
                if not (p.row == moveable.row and p.tier == moveable.tier and
                       p.bay * self.yard.split_factor + p.start_split == curr_abs)
            ]
        
        # Convert to Destination objects
        for pl in placements:
            # Check zone match
            zone_match = self._check_zone_match(container, pl)
            
            # Heat proximity
            heat_prox = self._compute_heat_proximity(pl.bay, train_heat_bays)
            
            result.append(Destination(
                dest_type=DestinationType.YARD,
                dest_id=None,
                row=pl.row,
                bay=pl.bay,
                tier=pl.tier,
                start_split=pl.start_split,
                zone_match=zone_match,
                heat_proximity=heat_prox
            ))
        
        return result
    
    def _train_destinations(
        self,
        moveable: MoveableContainer,
        container: Container,
        trains: Dict[str, Train]
    ) -> List[Destination]:
        """Generate train loading destinations."""
        result: List[Destination] = []
        
        # Only export containers can be loaded onto trains
        if getattr(container, "direction", "Export") != "Export":
            return result
        
        for train_id, train in trains.items():
            # Skip source train
            if moveable.source_type == SourceType.TRAIN and moveable.source_id == train_id:
                continue
            
            # Check if container is wanted by this train
            if moveable.container_id not in train.get_all_pickup_container_ids():
                continue
            
            # Check if train has space
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
                heat_proximity=1.0  # Train is "hot"
            ))
        
        return result
    
    def _truck_destinations(
        self,
        moveable: MoveableContainer,
        container: Container,
        trucks: Dict[str, Truck]
    ) -> List[Destination]:
        """Generate truck loading destinations."""
        result: List[Destination] = []
        
        for truck_id, truck in trucks.items():
            # Skip source truck
            if moveable.source_type == SourceType.TRUCK and moveable.source_id == truck_id:
                continue
            
            # Truck must be parked
            if not truck.parking_spot:
                continue
            
            # Check if container is wanted by this truck (pickup)
            if moveable.container_id not in truck.pickup_container_ids:
                continue
            
            # Check if truck can accommodate
            if not truck.can_accommodate_container(container):
                continue
            
            # Parse parking bay
            try:
                parts = truck.parking_spot.split("_")
                bay = int(parts[-2])
            except:
                bay = self.yard.n_bays // 2
            
            result.append(Destination(
                dest_type=DestinationType.TRUCK,
                dest_id=truck_id,
                row=0,
                bay=bay,
                tier=0,
                start_split=0,
                zone_match=True,
                heat_proximity=0.8
            ))
        
        return result
    
    def _check_zone_match(self, container: Container, placement: PlacementResult) -> bool:
        """Check if container goods type matches placement zone."""
        goods = container.goods_type if hasattr(container, 'goods_type') else "Regular"
        abs_start = placement.bay * self.yard.split_factor + placement.start_split
        
        if goods == "Reefer":
            return bool(self.yard.reefer_mask[placement.tier, placement.row, abs_start])
        elif goods == "DangerousGoods":
            return bool(self.yard.dangerous_mask[placement.tier, placement.row, abs_start])
        elif getattr(container, "is_swap_body", False) or getattr(container, "is_trailer", False):
            return bool(self.yard.swapbody_mask[placement.tier, placement.row, abs_start])
        else:
            return bool(self.yard.regular_mask[placement.tier, placement.row, abs_start])
    
    def _compute_heat_proximity(self, bay: int, heat_bays: Optional[Set[int]]) -> float:
        """Compute proximity to train heat (0-1, higher is closer)."""
        if not heat_bays:
            return 0.5
        
        min_dist = min(abs(bay - hb) for hb in heat_bays)
        # Normalize: 0 distance = 1.0, max_dist = 0.0
        max_dist = self.yard.n_bays
        return max(0.0, 1.0 - min_dist / max_dist)
    
    def list_parking_actions(
        self,
        trucks: Dict[str, Truck]
    ) -> List[ParkingAction]:
        """
        Get pending parking actions for unparked trucks.
        
        Returns:
            List of ParkingAction objects
        """
        if not self.parking:
            return []
        
        result: List[ParkingAction] = []
        
        for truck_id, truck in trucks.items():
            if truck.parking_spot:
                continue  # Already parked
            
            # Determine preferred bay
            preferred_bay = self._preferred_bay_for_truck(truck)
            
            # Find available spots
            if preferred_bay is not None:
                # Try exact and nearby
                for offset in [0, -1, 1, -2, 2]:
                    bay = preferred_bay + offset
                    if bay < 0 or bay >= self.yard.n_bays:
                        continue
                    
                    spots = self.parking.iter_free_in_bay_range(bay, bay)
                    if spots:
                        result.append(ParkingAction(
                            truck_id=truck_id,
                            spot=spots[0],
                            preferred_bay=preferred_bay,
                            delta_bay=offset
                        ))
                        break
                else:
                    # Fallback: any free spot
                    free = self.parking.iter_free()
                    if free:
                        spot = free[0]
                        spot_bay = self.parking.spot_bay(spot) or 0
                        result.append(ParkingAction(
                            truck_id=truck_id,
                            spot=spot,
                            preferred_bay=preferred_bay,
                            delta_bay=spot_bay - preferred_bay
                        ))
            else:
                # No preference: any free spot
                free = self.parking.iter_free()
                if free:
                    result.append(ParkingAction(
                        truck_id=truck_id,
                        spot=free[0],
                        preferred_bay=None,
                        delta_bay=0
                    ))
        
        return result
    
    def _preferred_bay_for_truck(self, truck: Truck) -> Optional[int]:
        """Determine preferred parking bay for truck."""
        # Pickup: median of target container bays
        if truck.pickup_container_ids:
            bays = []
            for cid in truck.pickup_container_ids:
                pl = self.yard.get_container_placement(cid)
                if pl:
                    bays.append(pl.bay)
            if bays:
                bays.sort()
                return bays[len(bays) // 2]
        
        # Delivery: goods anchor of first container
        if truck.containers:
            return self._goods_anchor(truck.containers[0])
        
        return None
