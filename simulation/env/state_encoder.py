# simulation/env/state_encoder.py
import numpy as np
from typing import Dict
from simulation.core.vehicles.train import Train
from simulation.core.vehicles.truck import Truck
from simulation.core.vehicles.terminal_truck import TerminalTruck

from simulation.core.facilities.yard import BooleanStorageYard
from simulation.core.facilities.railyard import BooleanRailYard
from simulation.planning.time_encoder import WeeklyTimeEncoder

class TerminalStateEncoder:
    """
    4D tensor [rows, bays, tiers, channels]:
    0: occupancy, 1: regular, 2: reefer, 3: dg, 4: sb/trailer,
    5: accessible, 6: wanted_by_train, 7: wanted_by_truck,
    8: days_until_departure (normalized),
    9: train_pickup_demand_per_bay (broadcast),
    10: train_anchor_heat (broadcast),
    + Forecast channels (11: trains_arriving_heat, 12: trucks_arriving_heat)
    """
    def __init__(self, yard: BooleanStorageYard, rail: BooleanRailYard):
        self.yard = yard
        self.rail = rail
        self.time_enc = WeeklyTimeEncoder()  # use our own decoder for angles

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
                now = c.arrival_date  # keep your current convention
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

        for cid, rec in self.yard.containers.items():
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

    def encode_with_forecast(self,
                            trains: Dict[str, Train],
                            trucks: Dict[str, Truck],
                            terminal_trucks: Dict[str, TerminalTruck],
                            day_plan,  # LogisticsManager.DayPlan
                            now) -> np.ndarray:
        """
        Extend the 11-channel tensor with bay-wise forecast channels:
        Trains arriving heat and Trucks arriving heat across multiple horizons:
        - horizons (hours): [3, 6, 12, 24, 48]
        Total extra channels: 10 (5 for trains, 5 for trucks).
        """
        base = self.encode(trains, trucks, terminal_trucks)  # [R, B, T, 11]
        R, B, T, _ = base.shape

        # Define horizons (no magic numbers): configurable list
        TRAIN_HEAT_WINDOWS_H = [3, 6, 12, 24, 48]
        TRUCK_HEAT_WINDOWS_H = [3, 6, 12, 24, 48]

        num_extra = len(TRAIN_HEAT_WINDOWS_H) + len(TRUCK_HEAT_WINDOWS_H)
        extra = np.zeros((R, B, T, num_extra), dtype=np.float32)

        # Trains arriving soon by window
        trains_heats = [np.zeros(B, dtype=np.float32) for _ in TRAIN_HEAT_WINDOWS_H]
        if day_plan and getattr(day_plan, "todays_trains", None):
            for st in day_plan.todays_trains:
                if st.train.train_id not in trains:
                    _day, h, m = self.time_enc.decode(st.arrival_angle)
                    arr_dt = day_plan.date.replace(hour=h, minute=m, second=0, microsecond=0)
                    dt_min = (arr_dt - now).total_seconds() / 60.0
                    if dt_min >= 0.0:
                        anchor = self.rail.get_anchor_bay(st.train.train_id) or (B // 2)
                        for i, hrs in enumerate(TRAIN_HEAT_WINDOWS_H):
                            window_min = hrs * 60.0
                            if dt_min <= window_min:
                                w = max(0.0, 1.0 - dt_min / window_min)
                                trains_heats[i][anchor] += w

        # Trucks arriving soon by window
        trucks_heats = [np.zeros(B, dtype=np.float32) for _ in TRUCK_HEAT_WINDOWS_H]
        if day_plan and getattr(day_plan, "trucks_today", None):
            for tk in day_plan.trucks_today:
                if tk and tk.arrival_time and tk.arrival_time > now:
                    dt_min = (tk.arrival_time - now).total_seconds() / 60.0
                    bay = self.yard.n_bays // 2
                    if getattr(tk, "pickup_container_ids", None):
                        bays = []
                        for cid in tk.pickup_container_ids:
                            pl = self.yard.get_container_placement(cid)
                            if pl:
                                bays.append(pl.bay)
                        if bays:
                            bays.sort()
                            bay = bays[len(bays)//2]
                    if dt_min >= 0.0:
                        for i, hrs in enumerate(TRUCK_HEAT_WINDOWS_H):
                            window_min = hrs * 60.0
                            if dt_min <= window_min:
                                w = max(0.0, 1.0 - dt_min / window_min)
                                trucks_heats[i][min(max(0, bay), B-1)] += w

        # Normalize each channel independently
        for i in range(len(trains_heats)):
            mx = trains_heats[i].max()
            if mx > 0:
                trains_heats[i] /= mx
        for i in range(len(trucks_heats)):
            mx = trucks_heats[i].max()
            if mx > 0:
                trucks_heats[i] /= mx

        # Pack into extra tensor: trains first, then trucks
        for i, heat in enumerate(trains_heats):
            extra[:, :, :, i] = heat[None, :, None]
        offset = len(trains_heats)
        for j, heat in enumerate(trucks_heats):
            extra[:, :, :, offset + j] = heat[None, :, None]

        return np.concatenate([base, extra], axis=-1)