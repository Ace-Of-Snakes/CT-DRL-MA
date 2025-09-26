# simulation/env/env.py
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable, Any

# facilities imports
from simulation.core.facilities.yard import BooleanStorageYard
from simulation.core.facilities.parking import ParkingArea
from simulation.core.facilities.railyard import BooleanRailYard, RailSlot

# vehicles imports
from simulation.core.vehicles.train import Train
from simulation.core.vehicles.truck import Truck
from simulation.core.vehicles.terminal_truck import TerminalTruck

# Env imports
from simulation.env.reward_engine import RewardEngine
from simulation.env.state_encoder import TerminalStateEncoder

# Logistics
from simulation.operations.terminal_manager import TerminalLogisticsManager, Move
from simulation.planning.logistics_manager import LogisticsManager, DayPlan
from simulation.analytics.stats_tracker import StatsTracker

# RMGC crane model (distance/time + endpoints)
from simulation.operations.crane_movements import TerminalRMGC

TT_JOB_SECONDS = 300.0  # 5 minutes


@dataclass
class CraneState:
    id: int
    busy_until: Optional[datetime] = None  # None = idle


class ContainerTerminalEnv:
    """
    Event-driven container terminal with n RMGC cranes. All crane distances/times and
    endpoints are computed by TerminalRMGC (simulation/operations/crane_movements.py).

    Step loop:
    - Build broad move list (TerminalLogisticsManager).
    - Agent selects moves for available cranes subject to zone overlap constraints.
    - Execute moves immediately; RMGC provides travel time. Advance time by the shortest move.
    - Handle arrivals/departures, automatic parking (optional), and terminal-truck jobs.

    Note: Single-step API is intentionally omitted; use step_dual_agent.
    """

    def __init__(
        self,
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
        auto_park: bool = False,  # default: agent controls parking
    ):
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

        # Terminal‑Truck busy times (env-managed)
        self._tt_busy_until: Dict[str, datetime] = {}

        # Cranes and zones
        self.num_cranes = max(1, int(num_cranes))
        self.cranes: List[CraneState] = []
        self.crane_zones: List[Tuple[int, int]] = []

        # RMGC crane model (distance/time + endpoints) — stays attached to yard/rail.
        self.rmgc = TerminalRMGC(yard=self.yard, rail=self.rail, num_tracks=self.num_tracks)

    # ------------- Public API -------------

    def reset(
        self,
        day_start: datetime,
        day_index: int = 0,
        trains_override: Optional[List[Train]] = None,
    ) -> Tuple[Any, List[Move]]:
        self.day_index = day_index
        self.current_time = day_start.replace(hour=0, minute=0, second=0, microsecond=0)

        # Plan day and clear state
        self.day_plan = self.lm.plan_day(self.current_time, trains_override=trains_override)
        self._scheduled_trains = list(self.day_plan.todays_trains)
        self.trains.clear()
        self.trucks.clear()
        self.terminal_trucks.clear()
        self._tt_busy_until.clear()

        # Re-sync RMGC to reflect current env references/track count
        # (yard/rail instances are typically unchanged, but this keeps layout in sync)
        self.rmgc.set_layout(yard=self.yard, rail=self.rail, num_tracks=self.num_tracks)

        # Two terminal trucks per module
        for i in range(2):
            tt = TerminalTruck()
            if not getattr(tt, "truck_id", None):
                setattr(tt, "truck_id", f"TTR_{i+1}")
            self.terminal_trucks[tt.truck_id] = tt

        self.cranes = [CraneState(i, None) for i in range(self.num_cranes)]
        self.crane_zones = self._make_crane_zones(overlap_bays=4)

        self._admit_arrivals()
        if self.auto_park:
            self._auto_slot_parking()

        state = self.encoder.encode_with_forecast(
            self.trains, self.trucks, self.terminal_trucks, self.day_plan, self.current_time
        )
        moves = self._list_moves()
        return state, moves

    def step_dual_agent(
        self,
        agent,
        log_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
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
            state = self.encoder.encode_with_forecast(
                self.trains, self.trucks, self.terminal_trucks, self.day_plan, self.current_time
            )
            return state, self._list_moves(), reward, False, info

        all_moves = self._list_moves()
        if not all_moves:
            self.current_time += timedelta(minutes=self.step_minutes)
            self._complete_terminal_truck_jobs()
            state = self.encoder.encode_with_forecast(
                self.trains, self.trucks, self.terminal_trucks, self.day_plan, self.current_time
            )
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

            state = self.encoder.encode_with_forecast(
                self.trains, self.trucks, self.terminal_trucks, self.day_plan, now
            )
            a_idx = agent.act(state, filtered)
            if a_idx < 0 or a_idx >= len(filtered):
                continue
            mv = filtered[a_idx]

            # Non-crane moves
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
                    rec = {
                        "timestamp": now.isoformat(),
                        "crane_id": crane.id,
                        "move_type": mv.type,
                        "args": mv.args,
                        "distance_m": 0.0,
                        "time_s": TT_JOB_SECONDS,
                        "reward": r,
                    }
                else:  # SLOT_TRUCK_PARKING
                    r = 0.5
                    reward += r
                    rec = {
                        "timestamp": now.isoformat(),
                        "crane_id": crane.id,
                        "move_type": mv.type,
                        "args": mv.args,
                        "distance_m": 0.0,
                        "time_s": 0.0,
                        "reward": r,
                        "delta_bay": mv.args.get("delta_bay", 0),
                    }
                info["executed"].append(rec)
                if log_cb:
                    try:
                        log_cb(rec)
                    except Exception:
                        pass
                continue

            # Crane move: use RMGC to build endpoints and cost
            mv_dict = {"type": mv.type, "args": mv.args}
            endpoints = self.rmgc.endpoints_for_move(mv_dict, self.trains, self.trucks, self.yard)
            if endpoints is None:
                # Preconditions (e.g., truck not parked) or non-crane move: small penalty
                reward -= 0.1
                continue

            cost = self.rmgc.estimate_move_cost(mv.type, endpoints[0], endpoints[1])
            ok = self.tlm.execute(mv, self.trains, self.trucks, self.terminal_trucks)
            if not ok:
                reward -= 0.05
                continue

            r = self.reward_engine.immediate_reward(mv.type, cost.distance_m, cost.time_s)
            reward += r
            self.cranes[crane.id].busy_until = now + timedelta(seconds=cost.time_s)
            picks.append((crane.id, mv, cost.distance_m, cost.time_s))
            used_bays |= self._move_bays(mv)
            used_cids |= self._move_container_ids(mv)

            rec = {
                "timestamp": now.isoformat(),
                "crane_id": crane.id,
                "move_type": mv.type,
                "args": mv.args,
                "distance_m": cost.distance_m,
                "time_s": cost.time_s,
                "reward": r,
            }
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
                try:
                    log_cb(rec)
                except Exception:
                    pass

        if not picks:
            self.current_time += timedelta(minutes=self.step_minutes)
            self._complete_terminal_truck_jobs()
            state = self.encoder.encode_with_forecast(
                self.trains, self.trucks, self.terminal_trucks, self.day_plan, self.current_time
            )
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

        state = self.encoder.encode_with_forecast(
            self.trains, self.trucks, self.terminal_trucks, self.day_plan, self.current_time
        )
        next_moves = self._list_moves()
        return state, next_moves, reward, done, info

    # ------------- internal helpers -------------
    def _tt_is_available(self, ttr: TerminalTruck) -> bool:
        if not ttr:
            return False
        busy_until = self._tt_busy_until.get(ttr.truck_id)
        if busy_until and busy_until > self.current_time:
            return False
        if hasattr(ttr, "is_available"):
            return bool(ttr.is_available())
        return len(getattr(ttr, "containers", [])) == 0

    def _complete_terminal_truck_jobs(self):
        done: List[str] = []
        for tt_id, until in self._tt_busy_until.items():
            if until <= self.current_time:
                tt = self.terminal_trucks.get(tt_id)
                if tt:
                    if hasattr(tt, "containers"):
                        tt.containers.clear()
                    if hasattr(tt, "status"):
                        try:
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
        # Terminal‑Truck
        for ttr in self.terminal_trucks.values():
            if self._tt_is_available(ttr):
                out.extend(self.tlm.list_yard_to_terminal_truck(ttr))
        # Parking (ACTIVE trucks only; not day plan)
        out.extend(self.tlm.list_parking_moves_active(self.trucks))
        # Yard <-> Yard
        out.extend(self.tlm.list_yard_to_yard())
        return out

    def _auto_slot_parking(self) -> None:
        if not self.auto_park or not self.day_plan:
            return
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

    # ---- zone/eligibility and move metadata ----
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
            if "train_id" in a:
                anchor = self.rail.get_anchor_bay(a["train_id"])
                if anchor is not None:
                    bays.add(anchor)
        return bays

    def _move_container_ids(self, mv: Move) -> set:
        ids = set()
        a = mv.args
        if "container_id" in a and a["container_id"]:
            ids.add(a["container_id"])
        return ids