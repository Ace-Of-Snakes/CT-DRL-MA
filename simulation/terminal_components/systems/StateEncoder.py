# simulation/terminal_components/systems/StateEncoder.py
import numpy as np
from typing import Dict
from simulation.terminal_components.storage.BooleanStorage import BooleanStorageYard
from simulation.terminal_components.vehicles.Train import Train
from simulation.terminal_components.vehicles.Truck import Truck
from simulation.terminal_components.vehicles.TerminalTruck import TerminalTruck
from simulation.terminal_components.systems.railyard import BooleanRailYard

class TerminalStateEncoder:
    """
    4D tensor [rows, bays, tiers, channels]:
    0: occupancy, 1: regular, 2: reefer, 3: dg, 4: sb/trailer,
    5: accessible, 6: wanted_by_train, 7: wanted_by_truck,
    8: days_until_departure (normalized),
    9: train_pickup_demand_per_bay (broadcast),
    10: train_anchor_heat (broadcast)
    """
    def __init__(self, yard: BooleanStorageYard, rail: BooleanRailYard):
        self.yard = yard
        self.rail = rail

    def encode(self,
               trains: Dict[str, Train],
               trucks: Dict[str, Truck],
               terminal_trucks: Dict[str, TerminalTruck]) -> np.ndarray:
        R, B, T = self.yard.n_rows, self.yard.n_bays, self.yard.n_tiers
        C = 11
        tensor = np.zeros((R, B, T, C), dtype=np.float32)

        # per-slot features
        for cid, rec in self.yard.containers.items():
            r, b, t = rec.placement.row, rec.placement.bay, rec.placement.tier
            c = rec.container
            tensor[r, b, t, 0] = 1.0
            if c.is_swap_body or c.is_trailer:
                tensor[r, b, t, 4] = 1.0
            elif c.goods_type == "Reefer":
                tensor[r, b, t, 2] = 1.0
            elif c.goods_type == "DangerousGoods":
                tensor[r, b, t, 3] = 1.0
            else:
                tensor[r, b, t, 1] = 1.0
            tensor[r, b, t, 5] = 1.0 if rec.is_accessible else 0.0
            try:
                now = c.arrival_date
                days = c.days_until_departure(now)
                tensor[r, b, t, 8] = float(min(30.0, max(0.0, days))) / 30.0
            except:
                pass

        # wanted by train/truck
        train_want = set()
        for tr in trains.values():
            train_want |= set(tr.get_all_pickup_container_ids())
        truck_want = set()
        for tk in trucks.values():
            truck_want |= set(tk.pickup_container_ids)

        for cid in self.yard.containers.keys():
            rec = self.yard.containers[cid]
            r, b, t = rec.placement.row, rec.placement.bay, rec.placement.tier
            if cid in train_want:
                tensor[r, b, t, 6] = 1.0
            if cid in truck_want:
                tensor[r, b, t, 7] = 1.0

        # bay-wise broadcasts
        demand_per_bay = np.zeros(B, dtype=np.float32)
        anchor_heat = np.zeros(B, dtype=np.float32)
        for tr in trains.values():
            anchor = self.rail.get_anchor_bay(tr.train_id) or (B // 2)
            total_len = 0.0
            for cid in tr.get_all_pickup_container_ids():
                c = self.yard.get_container(cid)
                if c:
                    total_len += c.length_m
            if total_len > 0:
                demand_per_bay[anchor] += total_len
                anchor_heat[anchor] += 1.0

        if B > 2:
            demand_per_bay = np.convolve(demand_per_bay, [0.25, 0.5, 0.25], mode='same')
            anchor_heat = np.convolve(anchor_heat, [0.2, 0.6, 0.2], mode='same')
        if demand_per_bay.max() > 0:
            demand_per_bay /= demand_per_bay.max()
        if anchor_heat.max() > 0:
            anchor_heat /= anchor_heat.max()

        tensor[:, :, :, 9] = demand_per_bay[None, :, None]
        tensor[:, :, :, 10] = anchor_heat[None, :, None]
        return tensor