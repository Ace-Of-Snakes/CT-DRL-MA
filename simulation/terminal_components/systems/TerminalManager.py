# simulation/terminal_components/systems/TerminalManager.py
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Iterable
from datetime import datetime
import numpy as np

from simulation.terminal_components.storage.BooleanStorage import BooleanStorageYard, PlacementResult
from simulation.terminal_components.vehicles.Train import Train
from simulation.terminal_components.vehicles.Truck import Truck
from simulation.terminal_components.vehicles.TerminalTruck import TerminalTruck
from simulation.terminal_components.systems.railyard import BooleanRailYard
from simulation.terminal_components.systems.parking import ParkingArea
from simulation.terminal_components.systems.TerminalGate import TerminalGate

YARD_TO_YARD = "YARD_TO_YARD"
TRAIN_TO_YARD = "TRAIN_TO_YARD"
YARD_TO_TRAIN = "YARD_TO_TRAIN"
TRUCK_TO_YARD = "TRUCK_TO_YARD"
YARD_TO_TRUCK = "YARD_TO_TRUCK"
TRAIN_TO_TRUCK = "TRAIN_TO_TRUCK"
TRUCK_TO_TRAIN = "TRUCK_TO_TRAIN"
YARD_TO_TERMINAL_TRUCK = "YARD_TO_TERMINAL_TRUCK"
SLOT_TRUCK_PARKING = "SLOT_TRUCK_PARKING"

PROXIMITY = 3

@dataclass(frozen=True)
class Move:
    type: str
    args: Dict[str, Any]

class TerminalLogisticsManager:
    def __init__(self, yard: BooleanStorageYard, rail: BooleanRailYard, parking: Optional[ParkingArea] = None):
        self.yard = yard
        self.rail = rail
        self.parking = parking
        self._zone_anchors = self._compute_zone_anchors()

    # ----- anchors -----
    def _center_bay_from_mask(self, mask2d: np.ndarray) -> int:
        cols = np.where(mask2d.any(axis=0))[0]
        if cols.size == 0:
            return self.yard.n_bays // 2
        c = int(round(cols.mean()))
        return min(max(c // self.yard.split_factor, 0), self.yard.n_bays - 1)

    def _compute_zone_anchors(self) -> Dict[str, int]:
        return {
            "reefer": self._center_bay_from_mask(self.yard.reefer_mask[0]),
            "dg": self._center_bay_from_mask(self.yard.dangerous_mask[0]),
            "sb": self._center_bay_from_mask(self.yard.swapbody_mask[0]),
            "reg": self.yard.n_bays // 2
        }

    def _goods_anchor(self, container) -> int:
        if container.goods_type == "Reefer":
            return self._zone_anchors["reefer"]
        if container.goods_type == "DangerousGoods":
            return self._zone_anchors["dg"]
        if getattr(container, "is_swap_body", False) or getattr(container, "is_trailer", False):
            return self._zone_anchors["sb"]
        return self._zone_anchors["reg"]

    def _search_goods_aware(self, container, anchors: Iterable[int]) -> List[PlacementResult]:
        seen = set()
        out: List[PlacementResult] = []
        for a in anchors:
            dests = self.yard.search_placement_all_tiers(container, target_bay=a, max_proximity=PROXIMITY)
            for d in dests:
                key = (d.row, d.bay, d.tier, d.start_split)
                if key not in seen:
                    seen.add(key)
                    out.append(d)
        out.sort(key=lambda p: (p.tier, p.score))
        return out

    # ----- parking: within ±2 bays of the "work bay" -----
    def _preferred_bay_for_truck(self, truck: Truck) -> Optional[int]:
        # pickup trucks: median of target container bays
        if getattr(truck, "pickup_container_ids", None):
            bays = []
            for cid in truck.pickup_container_ids:
                pl = self.yard.get_container_placement(cid)
                if pl:
                    bays.append(pl.bay)
            if bays:
                bays.sort()
                return bays[len(bays)//2]
        # delivery trucks: use goods anchor of first container
        if truck.containers:
            return self._goods_anchor(truck.containers[0])
        return None

    def list_parking_moves(self, gate: TerminalGate, todays_trucks: List[Truck], current_time: datetime) -> List[Move]:
        if not self.parking or not todays_trucks:
            return []
        arrived = gate.get_arrived_trucks(todays_trucks, current_time)
        candidates = [t for t in arrived if not t.parking_spot]
        if not candidates:
            return []
        moves: List[Move] = []
        # Allowed bay offsets from preferred bay: left (-1), exact (0), right (+1)
        PARKING_ALLOWED_OFFSETS = (-1, 0, +1)
        for t in candidates:
            pb = self._preferred_bay_for_truck(t)
            if pb is None:
                # fallback: any free
                free = self.parking.iter_free()
                if free:
                    spot = free[0]
                    moves.append(Move(SLOT_TRUCK_PARKING, {
                        "truck_id": t.truck_id,
                        "spot": spot,
                        "preferred_bay": None,
                        "delta_bay": 0
                    }))
                continue
            for off in PARKING_ALLOWED_OFFSETS:
                bay = pb + off
                if bay < 0 or bay >= self.yard.n_bays:
                    continue
                # find a free spot exactly in this bay
                near_exact = self.parking.iter_free_in_bay_range(bay, bay)
                if near_exact:
                    spot = near_exact[0]
                    moves.append(Move(SLOT_TRUCK_PARKING, {
                        "truck_id": t.truck_id,
                        "spot": spot,
                        "preferred_bay": pb,
                        "delta_bay": off
                    }))
            # If no exact bay spots found by offsets, try any within ±2 (rare fallback)
            if not any(m.args.get("truck_id") == t.truck_id for m in moves):
                near = self.parking.iter_free_in_bay_range(max(0, pb - 2), min(self.yard.n_bays - 1, pb + 2))
                if near:
                    spot = near[0]
                    delta = (self.parking.spot_bay(spot) or pb) - pb
                    moves.append(Move(SLOT_TRUCK_PARKING, {
                        "truck_id": t.truck_id,
                        "spot": spot,
                        "preferred_bay": pb,
                        "delta_bay": int(delta)
                    }))
        return moves

    # ----- move listing (broad) -----
    def list_train_to_yard(self, train: Train, top_per_container: Optional[int] = None) -> List[Move]:
        anchor_track = self.rail.get_anchor_bay(train.train_id)
        out: List[Move] = []
        for c in train.get_all_containers():
            if getattr(c, "direction", "Import") != "Import":
                continue
            anchors = ([anchor_track] if anchor_track is not None else []) + [self._goods_anchor(c)]
            dests = self._search_goods_aware(c, anchors)
            if top_per_container is None:
                take = dests
            else:
                take = dests[:top_per_container]
            for d in take:
                out.append(Move(TRAIN_TO_YARD, {"train_id": train.train_id, "container_id": c.container_id, "placement": d}))
        return out

    def list_yard_to_train(self, train: Train) -> List[Move]:
        out: List[Move] = []
        for cid in train.get_all_pickup_container_ids():
            if cid not in self.yard.accessible_containers:
                continue
            cont = self.yard.get_container(cid)
            if cont and train.has_space_for_container(cont):
                out.append(Move(YARD_TO_TRAIN, {"train_id": train.train_id, "container_id": cid}))
        return out

    def list_yard_to_yard(self) -> List[Move]:
        mv = self.yard.find_moveable_containers(max_proximity=PROXIMITY)
        out: List[Move] = []
        for cid, dests in mv.items():
            for d in dests:
                out.append(Move(YARD_TO_YARD, {"container_id": cid, "placement": d}))
        return out

    def list_truck_to_yard(self, truck: Truck, top_per_container: Optional[int] = None) -> List[Move]:
        if not truck.containers:
            return []
        out: List[Move] = []
        for c in truck.containers:
            dests = self._search_goods_aware(c, [self._goods_anchor(c)])
            take = dests if top_per_container is None else dests[:top_per_container]
            for d in take:
                out.append(Move(TRUCK_TO_YARD, {"truck_id": truck.truck_id, "container_id": c.container_id, "placement": d}))
        return out

    def list_yard_to_truck(self, truck: Truck) -> List[Move]:
        if not truck.pickup_container_ids:
            return []
        out: List[Move] = []
        for cid in list(truck.pickup_container_ids):
            if cid in self.yard.accessible_containers:
                c = self.yard.get_container(cid)
                if c and truck.can_accommodate_container(c):
                    out.append(Move(YARD_TO_TRUCK, {"truck_id": truck.truck_id, "container_id": cid}))
        return out

    def list_train_to_truck(self, train: Train, truck: Truck) -> List[Move]:
        if not truck.pickup_container_ids:
            return []
        out: List[Move] = []
        want = truck.pickup_container_ids
        for c in train.get_all_containers():
            if c.container_id in want and truck.can_accommodate_container(c):
                out.append(Move(TRAIN_TO_TRUCK, {"train_id": train.train_id, "truck_id": truck.truck_id, "container_id": c.container_id}))
        return out

    def list_truck_to_train(self, truck: Truck, train: Train) -> List[Move]:
        if not truck.containers:
            return []
        out: List[Move] = []
        want = train.get_all_pickup_container_ids()
        if not want:
            return out
        for c in truck.containers:
            if c.container_id in want and train.has_space_for_container(c):
                out.append(Move(TRUCK_TO_TRAIN, {"train_id": train.train_id, "truck_id": truck.truck_id, "container_id": c.container_id}))
        return out

    def list_yard_to_terminal_truck(self, ttr: TerminalTruck) -> List[Move]:
        """
        Listet alle möglichen TT‑Pickups (nur SwapBody/Trailer, nur wenn TT frei).
        Achtung: keine Kranzeit – die Env blockt die Ressource für 5 Minuten.
        """
        out: List[Move] = []
        if not ttr:
            return out
        # TerminalTruck gilt als verfügbar, wenn leer und nicht busy (Env prüft Busy zusätzlich)
        if hasattr(ttr, "is_available") and not ttr.is_available():
            return out

        for cid in list(self.yard.accessible_containers):
            c = self.yard.get_container(cid)
            if not c:
                continue
            # nur Swap Body / Trailer
            if not (getattr(c, "is_swap_body", False) or getattr(c, "is_trailer", False)):
                continue
            out.append(Move(YARD_TO_TERMINAL_TRUCK, {
                "terminal_truck_id": getattr(ttr, "truck_id", None),
                "container_id": cid
            }))
        return out

    def _remove_pickup_id_from_all_trucks(self, trucks: Dict[str, Truck], cid: str) -> None:
        for tk in trucks.values():
            try:
                tk.remove_pickup_container_id(cid)
            except Exception:
                pass

    # ----- execution -----
    def execute(self, move: Move, trains: Dict[str, Train], trucks: Dict[str, Truck], terminal_trucks: Dict[str, TerminalTruck]) -> bool:
        t = move.type
        a = move.args
        if t == SLOT_TRUCK_PARKING:
            if not self.parking:
                return False
            tr = trucks.get(a["truck_id"])
            spot = a["spot"]
            return bool(tr and self.parking.allocate(tr, spot))

        if t == TRAIN_TO_YARD:
            train = trains.get(a["train_id"])
            cid = a["container_id"]
            if not train:
                return False
            cont = train.remove_container(cid)
            if not cont:
                return False
            self.yard.add_container(cont, a["placement"])
            return True

        if t == YARD_TO_TRAIN:
            train = trains.get(a["train_id"])
            cid = a["container_id"]
            if not train:
                return False
            cont = self.yard.get_container(cid)
            if not cont or not train.has_space_for_container(cont):
                return False
            ok = train.add_container(cont)
            if not ok:
                return False
            self.yard.remove_container(cont)
            train.remove_pickup_container(cid)
            return True

        if t == YARD_TO_YARD:
            return self.yard.move_container(a["container_id"], a["placement"])

        if t == TRUCK_TO_YARD:
            truck = trucks.get(a["truck_id"])
            cid = a["container_id"]
            if not truck:
                return False
            cont = truck.remove_container(cid)
            if not cont:
                return False
            self.yard.add_container(cont, a["placement"])
            return True

        if t == YARD_TO_TRUCK:
            truck = trucks.get(a["truck_id"])
            cid = a["container_id"]
            if not truck:
                return False
            cont = self.yard.get_container(cid)
            if not cont or not truck.can_accommodate_container(cont):
                return False
            self.yard.remove_container(cont)
            if not truck.add_container(cont):
                return False
            truck.remove_pickup_container_id(cid)
            return True

        if t == TRAIN_TO_TRUCK:
            train = trains.get(a["train_id"])
            truck = trucks.get(a["truck_id"])
            cid = a["container_id"]
            if not train or not truck:
                return False
            cont = train.remove_container(cid)
            if not cont or not truck.can_accommodate_container(cont):
                return False
            return truck.add_container(cont)

        if t == TRUCK_TO_TRAIN:
            truck = trucks.get(a["truck_id"])
            train = trains.get(a["train_id"])
            cid = a["container_id"]
            if not truck or not train:
                return False
            cont = truck.remove_container(cid)
            if not cont or not train.has_space_for_container(cont):
                return False
            return train.add_container(cont)

        if t == YARD_TO_TERMINAL_TRUCK:
            tt = terminal_trucks.get(a["terminal_truck_id"])
            cid = a["container_id"]
            if not tt:
                return False
            # nur wenn TT frei (leer) – Busy/Timer macht die Env
            if hasattr(tt, "is_available") and not tt.is_available():
                return False
            cont = self.yard.get_container(cid)
            if not cont:
                return False
            # nur Swap Body / Trailer
            if not (getattr(cont, "is_swap_body", False) or getattr(cont, "is_trailer", False)):
                return False

            # aus Yard entfernen, Pickup-IDs bei allen Trucks entfernen
            self.yard.remove_container(cont)
            self._remove_pickup_id_from_all_trucks(trucks, cid)

            # auf TT laden (TT hält 1 Stück)
            if not tt.add_container(cont):
                # Rückrollen: falls add fehlschlägt, Container zurück in Yard (sollte selten sein)
                # Hier vereinfachen wir und geben False zurück, da zurücklegen Kranlos nicht passt.
                return False
            return True

        return False