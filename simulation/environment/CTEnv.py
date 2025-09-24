# simulation/env/ContainerTerminalEnv.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable, Any
from datetime import datetime, timedelta
import math

from simulation.terminal_components.storage.BooleanStorage import BooleanStorageYard, PlacementResult
from simulation.terminal_components.vehicles.Train import Train
from simulation.terminal_components.vehicles.Truck import Truck
from simulation.terminal_components.vehicles.TerminalTruck import TerminalTruck
from simulation.terminal_components.systems.railyard import BooleanRailYard, RailSlot
from simulation.terminal_components.systems.parking import ParkingArea
from simulation.terminal_components.systems.TerminalManager import TerminalLogisticsManager, Move
from simulation.terminal_components.systems.StateEncoder import TerminalStateEncoder
from simulation.terminal_components.systems.LogisticsManager import LogisticsManager, DayPlan
from simulation.terminal_components.systems.RewardEngine import RewardEngine
from simulation.analytics.stats_tracker import StatsTracker

TT_JOB_SECONDS = 300.0  # 5 minutes

@dataclass
class CraneState:
    id: int
    busy_until: Optional[datetime] = None  # None = idle


class ContainerTerminalEnv:
    """
    Eventgetriebene Umgebung mit zwei RMGC-Kränen:
    - Bei jedem Event wird eine große Move-Liste erzeugt (TerminalLogisticsManager).
    - Der Agent wählt nacheinander zwei Moves (je Kran; Zonen mit Überlappung).
    - Beide Moves werden sofort ausgeführt (ohne Abhängigkeit), Zeiten per RMGC-Kinematik berechnet.
    - Die Simulationszeit springt um die kürzere der beiden Bewegungszeiten vor; der andere Kran bleibt busy.

    Zusätzliche Funktionen:
    - Züge/Lkw nach Zeit einlassen, Abfahrten handhaben (inkl. RewardEngine.on_train_departure).
    - Parking-Moves automatisch ausführen (ohne Kranzeit).
    - Alternative step(action, moves) (single move) für Abwärtskompatibilität.
    """

    def __init__(self,
                yard: BooleanStorageYard,
                rail: BooleanRailYard,
                parking: ParkingArea,
                tlm: TerminalLogisticsManager,
                lm: LogisticsManager,
                num_tracks: int,
                step_minutes: int = 5,
                overflow_penalty: float = -500.0,
                num_cranes: int = 2,
                stats: Optional[StatsTracker] = None,
                auto_park: bool = False):  # default: agent controls parking
        self.yard = yard
        self.rail = rail
        self.parking = parking
        self.tlm = tlm
        self.lm = lm
        self.num_tracks = num_tracks
        self.step_minutes = step_minutes
        self.overflow_penalty = overflow_penalty

        self.encoder = TerminalStateEncoder(yard, rail)
        self.reward_engine = RewardEngine(yard)
        self.stats = stats

        self.auto_park = bool(auto_park)

        self.current_time: Optional[datetime] = None
        self.day_index: int = 0
        self.day_plan: Optional[DayPlan] = None
        self._scheduled_trains: List = []
        self._departed_cache: Dict[str, Train] = {}

        self.trains: Dict[str, Train] = {}
        self.trucks: Dict[str, Truck] = {}
        self.terminal_trucks: Dict[str, TerminalTruck] = {}
        
        # Terminal‑Truck Busy‑Zeiten (Env-gemanagt)
        self._tt_busy_until: Dict[str, datetime] = {}

        # geometry/perf...
        self._bay_length_m = 12.192
        self._row_slot_width_m = 2.44
        self._track_width_m = 3.0
        self._space_rails_to_parking_m = 5.0
        self._parking_lane_width_m = 4.0
        self._driving_lane_width_m = 4.0
        self._space_driving_to_storage_m = 2.0
        self._tier_height_m = 2.59
        self._ground_vehicle_height_m = 1.5

        self._trolley_speed = 70.0 / 60.0
        self._hoist_speed = 28.0 / 60.0
        self._gantry_speed = 130.0 / 60.0
        self._trolley_acc = 0.3
        self._hoist_acc = 0.2
        self._gantry_acc = 0.1
        self._max_hook_height = 20.0
        self._handling_s = 30.0

        self._rail_y0 = 0.0
        self._parking_y = self._rail_y0 + max(self._track_width_m, self._track_width_m * self.num_tracks) + self._space_rails_to_parking_m
        self._driving_y = self._parking_y + self._parking_lane_width_m
        self._storage_y0 = self._driving_y + self._driving_lane_width_m + self._space_driving_to_storage_m

        self.num_cranes = max(1, int(num_cranes))
        self.cranes: List[CraneState] = []
        self.crane_zones: List[Tuple[int, int]] = []

    # ------------- Public API -------------

# simulation/env/ContainerTerminalEnv.py

    def reset(self, day_start: datetime, day_index: int = 0, trains_override: Optional[List[Train]] = None) -> Tuple[Any, List[Move]]:
        self.day_index = day_index
        self.current_time = day_start.replace(hour=0, minute=0, second=0, microsecond=0)

        self.day_plan = self.lm.plan_day(self.current_time, trains_override=trains_override)
        self._scheduled_trains = list(self.day_plan.todays_trains)
        self.trains.clear()
        self.trucks.clear()
        self.terminal_trucks.clear()
        self._tt_busy_until.clear()
        # 2 Terminal‑Trucks pro Modul
        for i in range(2):
            tt = TerminalTruck()
            # sichere ID (falls Konstruktor schon vergibt, überschreiben wir nicht)
            if not getattr(tt, "truck_id", None):
                setattr(tt, "truck_id", f"TTR_{i+1}")
            self.terminal_trucks[tt.truck_id] = tt

        self.cranes = [CraneState(i, None) for i in range(self.num_cranes)]
        self.crane_zones = self._make_crane_zones(overlap_bays=4)

        self._admit_arrivals()
        if self.auto_park:
            self._auto_slot_parking()

        state = self.encoder.encode_with_forecast(self.trains, self.trucks, self.terminal_trucks, self.day_plan, self.current_time)
        moves = self._list_moves()
        return state, moves

    def step(self, action: Optional[int], moves: List[Move]) -> Tuple[Any, List[Move], float, bool, Dict]:
        reward = 0.0
        info: Dict[str, Any] = {"executed": None, "train_departures": [], "truck_departures": []}

        if moves and action is not None and 0 <= action < len(moves):
            mv = moves[action]
            p = self._endpoints_for_move(mv)
            if p is not None:
                cost = self._estimate_move_cost(p[0], p[1])
                ok = self.tlm.execute(mv, self.trains, self.trucks, self.terminal_trucks)
                if ok:
                    r = self.reward_engine.immediate_reward(mv.type, cost["distance_m"], cost["time_s"])
                    reward += r
                    rec = {"timestamp": self.current_time.isoformat(), "move_type": mv.type, "args": mv.args,
                           "distance_m": cost["distance_m"], "time_s": cost["time_s"], "reward": r}
                    if mv.type in ("YARD_TO_TRUCK", "TRAIN_TO_TRUCK"):
                        tk_id = mv.args.get("truck_id")
                        if tk_id:
                            tk = self.trucks.get(tk_id)
                            if tk and tk.loading_start_time is None:
                                tk.loading_start_time = self.current_time
                                wait_min = 0.0
                                if tk.arrival_time:
                                    wait_min = (self.current_time - tk.arrival_time).total_seconds() / 60.0
                                rec["first_service_wait_min"] = wait_min
                                reward += self.reward_engine.truck_first_service_reward(wait_min)
                                if self.stats:
                                    self.stats.on_truck_first_service(truck=tk, wait_minutes=wait_min)
                    info["executed"] = rec
                    if self.stats:
                        self.stats.log_move(rec)
                else:
                    reward -= 0.1
            else:
                # Only crane-less moves allowed here
                if mv.type == "YARD_TO_TERMINAL_TRUCK":
                    ok = self.tlm.execute(mv, self.trains, self.trucks, self.terminal_trucks)
                    if ok:
                        r = self.reward_engine.immediate_reward(mv.type, 0.0, TT_JOB_SECONDS)
                        reward += r
                        tt_id = mv.args.get("terminal_truck_id")
                        if tt_id:
                            self._tt_busy_until[tt_id] = self.current_time + timedelta(seconds=TT_JOB_SECONDS)
                        rec = {"timestamp": self.current_time.isoformat(), "move_type": mv.type, "args": mv.args,
                               "distance_m": 0.0, "time_s": TT_JOB_SECONDS, "reward": r}
                        info["executed"] = rec
                        if self.stats:
                            self.stats.log_move(rec)
                    else:
                        reward -= 0.1
                elif mv.type == "SLOT_TRUCK_PARKING":
                    ok = self.tlm.execute(mv, self.trains, self.trucks, self.terminal_trucks)
                    if ok:
                        r = 0.5
                        reward += r
                        rec = {"timestamp": self.current_time.isoformat(), "move_type": mv.type, "args": mv.args,
                               "distance_m": 0.0, "time_s": 0.0, "reward": r}
                        rec["delta_bay"] = mv.args.get("delta_bay", 0)
                        info["executed"] = rec
                        if self.stats:
                            self.stats.log_move(rec)
                    else:
                        reward -= 0.1
                else:
                    # Invalid: endpoints missing for a move that requires crane endpoints
                    reward -= 0.1

        self.current_time += timedelta(minutes=self.step_minutes)
        self._complete_terminal_truck_jobs()
        self.lm.recalc_assignments_before_arrival(self.current_time, self.day_plan)
        departed = self._admit_arrivals_and_departures()
        for tid, leftover in departed:
            tr = self._departed_cache.get(tid)
            if tr:
                reward += self.reward_engine.on_train_departure(tr)
                info["train_departures"].append(tid)
                if self.stats:
                    self.stats.on_train_departure(tid, leftover)
        self._auto_slot_parking()
        events = self._collect_truck_departures()
        info["truck_departures"].extend(events)
        done = False
        if self.current_time >= self.day_plan.last_departure_dt and not self.trains:
            reward += self.reward_engine.end_of_day_penalty(self.current_time)
            self._rollover_missed_deadlines()
            done = True
        next_moves = self._list_moves()
        state = self.encoder.encode_with_forecast(self.trains, self.trucks, self.terminal_trucks, self.day_plan, self.current_time)
        return state, next_moves, reward, done, info

    def step_dual_agent(self,
                        agent,
                        log_cb: Optional[Callable[[Dict[str, Any]], None]] = None
                        ) -> Tuple[Any, List[Move], float, bool, Dict]:
        reward = 0.0
        info: Dict[str, Any] = {"executed": [], "train_departures": [], "truck_departures": []}
        now = self.current_time

        departed = self._admit_arrivals_and_departures()
        for tid, leftover in departed:
            tr = self._departed_cache.get(tid)
            if tr:
                reward += self.reward_engine.on_train_departure(tr)
                info["train_departures"].append(tid)
                if self.stats:
                    self.stats.on_train_departure(tid, leftover)

        self._auto_slot_parking()

        idle = [c for c in self.cranes if (c.busy_until is None or c.busy_until <= now)]
        if not idle:
            next_t = min(c.busy_until for c in self.cranes if c.busy_until is not None)
            advance_min = (next_t - now).total_seconds() / 60.0
            self.current_time = next_t
            self._complete_terminal_truck_jobs()
            reward += self.reward_engine.waiting_penalty(len(self.trucks), advance_min)
            state = self.encoder.encode_with_forecast(self.trains, self.trucks, self.terminal_trucks, self.day_plan, self.current_time)
            return state, self._list_moves(), reward, False, info

        all_moves = self._list_moves()
        if not all_moves:
            self.current_time += timedelta(minutes=self.step_minutes)
            self._complete_terminal_truck_jobs()
            state = self.encoder.encode_with_forecast(self.trains, self.trucks, self.terminal_trucks, self.day_plan, self.current_time)
            return state, self._list_moves(), reward, False, info

        if log_cb is None and self.stats:
            def log_cb(rec: Dict[str, Any]):
                self.stats.log_move(rec)

        zones = self.crane_zones
        picks: List[Tuple[int, Move, float, float]] = []
        used_bays, used_cids = set(), set()

        for crane in idle:
            cand = [m for m in all_moves if self._eligible_for_crane(m, crane.id, zones)]
            if not cand:
                continue
            filtered = []
            for mv in cand:
                cids = self._move_container_ids(mv)
                bays = self._move_bays(mv)
                if cids and (used_cids & cids):
                    continue
                if bays and (used_bays & bays):
                    continue
                filtered.append(mv)
            if not filtered:
                continue

            state = self.encoder.encode_with_forecast(self.trains, self.trucks, self.terminal_trucks, self.day_plan, now)
            a_idx = agent.act(state, filtered)
            if a_idx < 0 or a_idx >= len(filtered):
                continue
            mv = filtered[a_idx]

            p = self._endpoints_for_move(mv)

            # Only crane-less moves allowed without endpoints
            if mv.type in ("SLOT_TRUCK_PARKING", "YARD_TO_TERMINAL_TRUCK"):
                ok = self.tlm.execute(mv, self.trains, self.trucks, self.terminal_trucks)
                if not ok:
                    continue
                if mv.type == "YARD_TO_TERMINAL_TRUCK":
                    r = self.reward_engine.immediate_reward(mv.type, 0.0, TT_JOB_SECONDS)
                    reward += r
                    tt_id = mv.args.get("terminal_truck_id")
                    if tt_id:
                        self._tt_busy_until[tt_id] = now + timedelta(seconds=TT_JOB_SECONDS)
                    rec = {"timestamp": now.isoformat(), "crane_id": crane.id,
                           "move_type": mv.type, "args": mv.args,
                           "distance_m": 0.0, "time_s": TT_JOB_SECONDS, "reward": r}
                else:  # SLOT_TRUCK_PARKING
                    r = 0.5
                    reward += r
                    rec = {"timestamp": now.isoformat(), "crane_id": crane.id,
                           "move_type": mv.type, "args": mv.args,
                           "distance_m": 0.0, "time_s": 0.0, "reward": r}
                    rec["delta_bay"] = mv.args.get("delta_bay", 0)
                info["executed"].append(rec)
                if log_cb:
                    try: log_cb(rec)
                    except Exception: pass
                continue

            # Skip invalid truck/train moves that lack endpoints (e.g., truck not parked)
            if p is None:
                # small penalty to discourage selecting invalid moves
                reward -= 0.1
                continue

            # normal crane move
            cost = self._estimate_move_cost(p[0], p[1])
            ok = self.tlm.execute(mv, self.trains, self.trucks, self.terminal_trucks)
            if not ok:
                reward -= 0.05
                continue

            r = self.reward_engine.immediate_reward(mv.type, cost["distance_m"], cost["time_s"])
            reward += r
            self.cranes[crane.id].busy_until = now + timedelta(seconds=cost["time_s"])
            picks.append((crane.id, mv, cost["distance_m"], cost["time_s"]))
            used_bays |= self._move_bays(mv)
            used_cids |= self._move_container_ids(mv)

            rec = {"timestamp": now.isoformat(), "crane_id": crane.id,
                   "move_type": mv.type, "args": mv.args,
                   "distance_m": cost["distance_m"], "time_s": cost["time_s"], "reward": r}
            if mv.type in ("YARD_TO_TRUCK", "TRAIN_TO_TRUCK"):
                tk_id = mv.args.get("truck_id")
                if tk_id:
                    tk = self.trucks.get(tk_id)
                    if tk and tk.loading_start_time is None:
                        tk.loading_start_time = now
                        wait_min = 0.0
                        if tk.arrival_time:
                            wait_min = (now - tk.arrival_time).total_seconds() / 60.0
                        rec["first_service_wait_min"] = wait_min
                        reward += self.reward_engine.truck_first_service_reward(wait_min)
                        if self.stats:
                            self.stats.on_truck_first_service(truck=tk, wait_minutes=wait_min)

            info["executed"].append(rec)
            if log_cb:
                try: log_cb(rec)
                except Exception: pass

        if not picks:
            self.current_time += timedelta(minutes=self.step_minutes)
            self._complete_terminal_truck_jobs()
            state = self.encoder.encode_with_forecast(self.trains, self.trucks, self.terminal_trucks, self.day_plan, self.current_time)
            return state, self._list_moves(), reward, False, info

        min_time_s = min(t for (_, _, _, t) in picks)
        self.current_time = now + timedelta(seconds=min_time_s)
        self._complete_terminal_truck_jobs()

        departed2 = self._admit_arrivals_and_departures()
        for tid, leftover in departed2:
            tr = self._departed_cache.get(tid)
            if tr:
                reward += self.reward_engine.on_train_departure(tr)
                info["train_departures"].append(tid)
                if self.stats:
                    self.stats.on_train_departure(tid, leftover)

        events = self._collect_truck_departures()
        info["truck_departures"].extend(events)

        done = False
        if self.current_time >= self.day_plan.last_departure_dt and not self.trains:
            reward += self.reward_engine.end_of_day_penalty(self.current_time)
            self._rollover_missed_deadlines()
            done = True

        state = self.encoder.encode_with_forecast(self.trains, self.trucks, self.terminal_trucks, self.day_plan, self.current_time)
        next_moves = self._list_moves()
        return state, next_moves, reward, done, info

    # ------------- interne Helfer -------------
    def _tt_is_available(self, ttr: TerminalTruck) -> bool:
        """Env‑Sicht: frei wenn keine Busy‑Marke in der Zukunft und TT leer/idle."""
        if not ttr:
            return False
        busy_until = self._tt_busy_until.get(ttr.truck_id)
        if busy_until and busy_until > self.current_time:
            return False
        if hasattr(ttr, "is_available"):
            return bool(ttr.is_available())
        # Fallback: als frei betrachten, wenn keine Container geladen sind
        return len(getattr(ttr, "containers", [])) == 0

    def _complete_terminal_truck_jobs(self):
        """Beende TT‑Jobs, deren Zeit abgelaufen ist: TT leeren und Busy‑Flag löschen."""
        done: List[str] = []
        for tt_id, until in self._tt_busy_until.items():
            if until <= self.current_time:
                tt = self.terminal_trucks.get(tt_id)
                if tt:
                    # Terminal‑Trucks entladen ihr Ziel außerhalb des Yards; wir leeren nur.
                    if hasattr(tt, "containers"):
                        tt.containers.clear()
                    if hasattr(tt, "status"):
                        try:
                            # optional: auf idle setzen (abhängig von TT‑Implementierung)
                            tt.status = "idle"
                        except Exception:
                            pass
                done.append(tt_id)
        for tt_id in done:
            self._tt_busy_until.pop(tt_id, None)

    def _make_crane_zones(self, overlap_bays: int = 4) -> List[Tuple[int, int]]:
        n = max(1, self.num_cranes)
        B = self.yard.n_bays
        if n == 1:
            return [(0, B)]
        base = B // n
        zones = []
        for i in range(n):
            lo = max(0, i * base - overlap_bays)
            hi = B if i == n - 1 else min(B, (i + 1) * base + overlap_bays)
            zones.append((lo, hi))
        return zones

    def _list_moves(self) -> List[Move]:
        out: List[Move] = []
        # Train <-> Yard
        for tr in self.trains.values():
            out.extend(self.tlm.list_yard_to_train(tr))
            out.extend(self.tlm.list_train_to_yard(tr))
        # Truck <-> Yard
        for tk in self.trucks.values():
            out.extend(self.tlm.list_yard_to_truck(tk))
            out.extend(self.tlm.list_truck_to_yard(tk))
        # Train <-> Truck
        if self.trains and self.trucks:
            for tr in self.trains.values():
                for tk in self.trucks.values():
                    out.extend(self.tlm.list_train_to_truck(tr, tk))
                    out.extend(self.tlm.list_truck_to_train(tk, tr))
        # Terminal‑Truck (nur wenn Ressource frei)
        for ttr in self.terminal_trucks.values():
            if self._tt_is_available(ttr):
                out.extend(self.tlm.list_yard_to_terminal_truck(ttr))
        # Parking
        if self.day_plan:
            pmv = self.tlm.list_parking_moves(self.lm.gate, self.day_plan.trucks_today, self.current_time)
            out.extend(pmv)
        # Yard <-> Yard
        out.extend(self.tlm.list_yard_to_yard())
        return out

    def _auto_slot_parking(self) -> None:
        # Only if enabled; otherwise agent parks via SLOT_TRUCK_PARKING
        if not self.auto_park or not self.day_plan:
            return
        from simulation.terminal_components.systems.TerminalGate import TerminalGate
        pmoves = self.tlm.list_parking_moves(self.lm.gate, self.day_plan.trucks_today, self.current_time)
        for mv in pmoves:
            try:
                self.tlm.execute(mv, self.trains, self.trucks, self.terminal_trucks)
            except Exception:
                pass

    def _collect_truck_departures(self) -> List[Dict[str, Any]]:
            events = []
            to_remove = []
            for tid, tk in self.trucks.items():
                if tk.is_ready_to_depart():
                    if tk.departure_time is None:
                        tk.departure_time = self.current_time
                    wait_min = 0.0
                    if tk.arrival_time:
                        wait_min = (tk.departure_time - tk.arrival_time).total_seconds() / 60.0
                    events.append({"truck_id": tk.truck_id, "wait_min": wait_min})
                    if self.stats:
                        self.stats.on_truck_departure(truck=tk, wait_minutes=wait_min)
                    if self.parking:
                        self.parking.release(tk)
                    to_remove.append(tid)
            for tid in to_remove:
                self.trucks.pop(tid, None)
            return events

    def _admit_arrivals(self) -> None:
            still = []
            self._departed_cache = {}
            for st in self._scheduled_trains:
                d_arr, h_arr, m_arr = self.lm.time.decode(st.arrival_angle)
                d_dep, h_dep, m_dep = self.lm.time.decode(st.departure_angle)
                arr_dt = self.day_plan.date.replace(hour=h_arr, minute=m_arr, second=0, microsecond=0)
                dep_dt = self.day_plan.date.replace(hour=h_dep, minute=m_dep, second=0, microsecond=0)
                tid = st.train.train_id
                if tid not in self.trains and self.current_time >= arr_dt and self.current_time < dep_dt:
                    anchor = int(round((st.track_id + 1) * (self.yard.n_bays / (self.num_tracks + 1))))
                    anchor = max(0, min(self.yard.n_bays - 1, anchor))
                    self.rail.slot_train(st.train, RailSlot(track_id=st.track_id, anchor_bay=anchor))
                    st.train.arrival_time = arr_dt
                    st.train.departure_time = dep_dt
                    self.trains[tid] = st.train
                    if self.stats:
                        self.stats.on_train_arrival(st.train)
                still.append(st)
            self._scheduled_trains = still

            for t in self.day_plan.trucks_today:
                if t and t.truck_id not in self.trucks and t.arrival_time and t.arrival_time <= self.current_time:
                    t.status = "waiting"
                    self.trucks[t.truck_id] = t
                    if self.stats:
                        self.stats.on_truck_arrival(t)

    def _admit_arrivals_and_departures(self) -> List[Tuple[str, int]]:
        departed: List[Tuple[str, int]] = []
        still = []
        self._departed_cache = {}

        for st in self._scheduled_trains:
            d_arr, h_arr, m_arr = self.lm.time.decode(st.arrival_angle)
            d_dep, h_dep, m_dep = self.lm.time.decode(st.departure_angle)
            arr_dt = self.day_plan.date.replace(hour=h_arr, minute=m_arr, second=0, microsecond=0)
            dep_dt = self.day_plan.date.replace(hour=h_dep, minute=m_dep, second=0, microsecond=0)
            tid = st.train.train_id

            if tid not in self.trains and self.current_time >= arr_dt and self.current_time < dep_dt:
                anchor = int(round((st.track_id + 1) * (self.yard.n_bays / (self.num_tracks + 1))))
                anchor = max(0, min(self.yard.n_bays - 1, anchor))
                self.rail.slot_train(st.train, RailSlot(track_id=st.track_id, anchor_bay=anchor))
                st.train.arrival_time = arr_dt
                st.train.departure_time = dep_dt
                self.trains[tid] = st.train
                if self.stats:
                    self.stats.on_train_arrival(st.train)

            if tid in self.trains and self.current_time >= dep_dt:
                tr = self.trains.pop(tid)
                leftover = len(tr.get_all_pickup_container_ids())
                self.rail.release_train(tid)
                self._departed_cache[tid] = tr
                departed.append((tid, leftover))
                continue

            still.append(st)

        self._scheduled_trains = still

        for t in self.day_plan.trucks_today:
            if t and t.truck_id not in self.trucks and t.arrival_time and t.arrival_time <= self.current_time:
                t.status = "waiting"
                self.trucks[t.truck_id] = t
                if self.stats:
                    self.stats.on_truck_arrival(t)

        return departed

    def _rollover_missed_deadlines(self) -> None:
        # No rollover via estimated departures; departure_date is authoritative.
        return None

    # ---- RMGC Kinematik/Endpunkte ----

    def _yard_xyz(self, pl: PlacementResult) -> Tuple[float, float, float]:
        x = (pl.bay + pl.start_split / self.yard.split_factor) * self._bay_length_m
        y = self._storage_y0 + pl.row * self._row_slot_width_m
        z = pl.tier * self._tier_height_m
        return (x, y, z)

    def _train_xyz(self, train: Train) -> Tuple[float, float, float]:
        anchor_bay = self.rail.get_anchor_bay(train.train_id) or (self.yard.n_bays // 2)
        x = anchor_bay * self._bay_length_m
        try:
            track_idx = int(train.rail_track) if train.rail_track is not None else 0
        except ValueError:
            track_idx = 0
        y = self._rail_y0 + track_idx * self._track_width_m
        z = self._ground_vehicle_height_m
        return (x, y, z)

    def _truck_xyz(self, truck: Truck) -> Tuple[float, float, float]:
        # Erwartet Parkplatz-String "P_{bay}_{split}" – robust parsen
        bay = self.yard.n_bays // 2
        split = 0
        if isinstance(truck.parking_spot, str):
            try:
                parts = truck.parking_spot.split("_")
                if len(parts) == 3:
                    _, b, s = parts
                    bay = int(b)
                    split = int(s)
            except:
                pass
        x = (bay + split / max(1, self.yard.split_factor)) * self._bay_length_m
        y = self._parking_y
        z = self._ground_vehicle_height_m
        return (x, y, z)

    def _stack_xyz(self) -> Tuple[float, float, float]:
        x = (self.yard.n_bays + 2) * self._bay_length_m
        y = self._storage_y0 + (self.yard.n_rows * self._row_slot_width_m) / 2.0
        z = 0.0
        return (x, y, z)

    def _axis_time(self, dist: float, vmax: float, acc: float) -> float:
        if dist <= 0:
            return 0.0
        t_acc = vmax / acc
        d_acc = 0.5 * acc * t_acc * t_acc
        if dist <= 2 * d_acc:
            return 2.0 * math.sqrt(dist / acc)
        return 2.0 * t_acc + (dist - 2 * d_acc) / vmax

    def _estimate_move_cost(self,
                            p1: Tuple[float, float, float],
                            p2: Tuple[float, float, float]) -> Dict[str, float]:
        # Heber runter (aufnehmen)
        hoist_down = self._axis_time(abs(self._max_hook_height - p1[2]), self._hoist_speed, self._hoist_acc)
        # heben hoch (frei)
        hoist_up = self._axis_time(abs(self._max_hook_height - p1[2]), self._hoist_speed, self._hoist_acc)
        # Ebene Fahrt
        dx = abs(p2[0] - p1[0])
        dy = abs(p2[1] - p1[1])
        plane = max(self._axis_time(dx, self._gantry_speed, self._gantry_acc),
                    self._axis_time(dy, self._trolley_speed, self._trolley_acc))
        # absenken am Ziel
        hoist_lower = self._axis_time(abs(self._max_hook_height - p2[2]), self._hoist_speed, self._hoist_acc)
        t = hoist_down + hoist_up + plane + hoist_lower + self._handling_s
        dist = dx + dy + abs(p2[2] - p1[2])
        return {"distance_m": dist, "time_s": t}

    def _endpoints_for_move(self, mv: Move) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
        t = mv.type
        a = mv.args
        if t == "YARD_TO_YARD":
            cid = a.get("container_id")
            src_pl = self.yard.get_container_placement(cid)
            dst_pl: PlacementResult = a.get("placement")
            if src_pl and dst_pl:
                return self._yard_xyz(src_pl), self._yard_xyz(dst_pl)
            return None
        if t == "YARD_TO_TRAIN":
            cid = a.get("container_id")
            tr = self.trains.get(a.get("train_id"))
            src_pl = self.yard.get_container_placement(cid)
            if tr and src_pl:
                return self._yard_xyz(src_pl), self._train_xyz(tr)
            return None
        if t == "TRAIN_TO_YARD":
            tr = self.trains.get(a.get("train_id"))
            dst_pl: PlacementResult = a.get("placement")
            if tr and dst_pl:
                return self._train_xyz(tr), self._yard_xyz(dst_pl)
            return None
        if t == "TRUCK_TO_YARD":
            tk = self.trucks.get(a.get("truck_id"))
            dst_pl: PlacementResult = a.get("placement")
            # Require parked truck for crane endpoints
            if tk and tk.parking_spot and dst_pl:
                return self._truck_xyz(tk), self._yard_xyz(dst_pl)
            return None
        if t == "YARD_TO_TRUCK":
            cid = a.get("container_id")
            tk = self.trucks.get(a.get("truck_id"))
            src_pl = self.yard.get_container_placement(cid)
            if tk and tk.parking_spot and src_pl:
                return self._yard_xyz(src_pl), self._truck_xyz(tk)
            return None
        if t == "TRAIN_TO_TRUCK":
            tr = self.trains.get(a.get("train_id"))
            tk = self.trucks.get(a.get("truck_id"))
            if tr and tk and tk.parking_spot:
                return self._train_xyz(tr), self._truck_xyz(tk)
            return None
        if t == "TRUCK_TO_TRAIN":
            tr = self.trains.get(a.get("train_id"))
            tk = self.trucks.get(a.get("truck_id"))
            if tr and tk and tk.parking_spot:
                return self._truck_xyz(tk), self._train_xyz(tr)
            return None
        if t == "YARD_TO_TERMINAL_TRUCK":
            # KRANLOS: keine Endpunkte zurückgeben => Sofortausführung
            return None
        if t == "SLOT_TRUCK_PARKING":
            return None
        return None

    # ---- Konflikt-/Zonen-Prüfung und Move-Metadaten ----

    def _eligible_for_crane(self, mv: Move, crane_id: int, zones: List[Tuple[int, int]]) -> bool:
        bays = self._move_bays(mv)
        if not bays:
            return True
        lo, hi = zones[crane_id]
        return all(lo <= b < hi for b in bays)

    def _move_bays(self, mv: Move) -> set:
        t = mv.type
        a = mv.args
        bays = set()
        if t in ("TRAIN_TO_YARD", "TRUCK_TO_YARD", "YARD_TO_YARD"):
            dst = a.get("placement")
            if dst:
                bays.add(dst.bay)
        if t in ("YARD_TO_TRAIN", "YARD_TO_TRUCK", "YARD_TO_TERMINAL_TRUCK"):
            cid = a.get("container_id")
            pl = self.yard.get_container_placement(cid)
            if pl:
                bays.add(pl.bay)
        if t in ("TRAIN_TO_TRUCK", "TRUCK_TO_TRAIN"):
            # train-anchor approximiert
            if "train_id" in a:
                anchor = self.rail.get_anchor_bay(a["train_id"])
                if anchor is not None:
                    bays.add(anchor)
        return bays

    def _move_container_ids(self, mv: Move) -> set:
        # Hilfsfunktion um Containerkonflikte (gleiche IDs) zu vermeiden
        ids = set()
        a = mv.args
        if "container_id" in a and a["container_id"]:
            ids.add(a["container_id"])
        # Bei TRAIN_TO_YARD/TRUCK_TO_YARD steckt ID nicht zwingend in args (doch hier schon vorhanden)
        return ids