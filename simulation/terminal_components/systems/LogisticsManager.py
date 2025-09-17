# simulation/terminal_components/systems/LogisticsManager.py
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import random
from collections import defaultdict

from simulation.terminal_components.storage.BooleanStorage import BooleanStorageYard
from simulation.terminal_components.systems.TerminalGate import TerminalGate, Order
from simulation.terminal_components.systems.train_tools.DPParser import DrivingPlanParser
from simulation.terminal_components.systems.train_tools.TrainScheduler import TrainScheduler, TrainSchedule, ScheduledTrain
from simulation.terminal_components.systems.train_tools.TimeEncoder import WeeklyTimeEncoder
from simulation.terminal_components.systems.train_tools.TrainLoader import TrainLoader
from simulation.terminal_components.vehicles.Train import Train
from simulation.terminal_components.storage_units.Container import Container

RECALC_WINDOW_MIN = 30

@dataclass
class DayPlan:
    date: datetime
    schedule: TrainSchedule
    todays_trains: List[ScheduledTrain]
    last_departure_dt: datetime
    trucks_today: List  # List[Truck]
    pickup_assignments: Dict[str, Dict[int, List[str]]]  # train_id -> wagon_idx -> [container_id]

class LogisticsManager:
    """
    Day planner:
    - Parse driving plan -> schedule trains on tracks
    - Load import containers on trains, rearrange wagons
    - Assign Export containers due today from yard to trains (pickup IDs)
    - Generate trucks:
        * delivery trucks with Export containers (count ~= 0.75 * today's import arrivals)
        * pickup trucks for Import containers due today in yard
    - Recalculate pickup assignments within 30 minutes before train arrival
    """

    def __init__(self,
                 yard: BooleanStorageYard,
                 terminal_gate: TerminalGate,
                 train_loader: TrainLoader,
                 train_scheduler: TrainScheduler,
                 parser: DrivingPlanParser):
        self.yard = yard
        self.gate = terminal_gate
        self.loader = train_loader
        self.scheduler = train_scheduler
        self.parser = parser
        self.time = WeeklyTimeEncoder()

    def plan_day(self, day_start: datetime) -> DayPlan:
        # 1) Build trains and schedule
        trains = self.parser.create_trains()
        schedule = self.scheduler.schedule_trains(trains)
        day_name = day_start.strftime("%A").lower()
        todays_trains = [st for st in schedule.scheduled_trains
                         if self.time.decode(st.arrival_angle)[0] == day_name]

        # 2) Pre-load import containers and rearrange wagons
        for st in todays_trains:
            op = st.operator
            self.loader.load_train(st.train, operator=op, current_date=day_start)
            self.loader.rearrange_wagons_for_goods(st.train, self.yard)

        # Count import containers arriving on trains today
        imports_arriving = sum(st.train.get_container_count() for st in todays_trains)

        # 3) Gather due-today yard containers
        due_pairs = self.yard.get_containers_departing_on(day_start, use_estimated=True, one_based_bay=False)
        # Split by direction
        due_export_ids = []
        due_import_ids = []
        for cid, _bay in due_pairs:
            c = self.yard.get_container(cid)
            if not c: 
                continue
            if c.direction == "Export":
                due_export_ids.append(cid)
            else:
                due_import_ids.append(cid)

        due_export_containers = [self.yard.get_container(cid) for cid in due_export_ids if self.yard.get_container(cid)]
        due_import_containers = [self.yard.get_container(cid) for cid in due_import_ids if self.yard.get_container(cid)]

        # 4) Assign due-today Export containers to trains (operator-agnostic)
        pickup_assignments: Dict[str, Dict[int, List[str]]] = self._assign_pickups_to_trains(
            due_export_containers, [st.train for st in todays_trains], append_to=None
        )

        # 5) Generate trucks for today
        # 5a) Delivery trucks (Export containers), ratio = 0.75 per Import arrival
        target_exports = int(round(0.75 * imports_arriving))
        export_cfg = self._export_operator_split(todays_trains, target_exports)
        # 5b) Pickup trucks for Import containers due today
        order = Order(import_containers=due_import_containers, export_operators=export_cfg)
        trucks_today = self.gate.process_order(order, day_start, day_start.strftime("%A"))

        # 6) Pre-assign IDs of export deliveries to trains as well
        if trucks_today:
            export_delivered: List[Container] = []
            for t in trucks_today:
                if t and getattr(t, "is_delivery_truck", False) and t.containers:
                    export_delivered.extend(t.containers)
            if export_delivered:
                self._assign_pickups_to_trains(export_delivered, [st.train for st in todays_trains],
                                               append_to=pickup_assignments)

        # 7) Compute last departure datetime for cut-off
        last_departure_dt = self._last_departure_datetime(todays_trains, day_start)

        return DayPlan(
            date=day_start,
            schedule=schedule,
            todays_trains=todays_trains,
            last_departure_dt=last_departure_dt,
            trucks_today=[t for t in trucks_today if t],
            pickup_assignments=pickup_assignments
        )

    def recalc_assignments_before_arrival(self, now: datetime, day_plan: DayPlan) -> None:
        horizon = now + timedelta(minutes=RECALC_WINDOW_MIN)
        day_name = now.strftime("%A").lower()
        imminent: List[Train] = []
        for st in day_plan.todays_trains:
            d, h, m = self.time.decode(st.arrival_angle)
            if d != day_name:
                continue
            arr_dt = now.replace(hour=h, minute=m, second=0, microsecond=0)
            if now <= arr_dt <= horizon:
                imminent.append(st.train)
        if not imminent:
            return

        # Gather still-present Export due-today
        due_pairs = self.yard.get_containers_departing_on(now, use_estimated=True, one_based_bay=False)
        due_export = []
        for cid, _ in due_pairs:
            c = self.yard.get_container(cid)
            if c and c.direction == "Export":
                due_export.append(c)

        # Clear existing pickups on imminent trains if not due today
        due_ids = {c.container_id for c in due_export}
        for tr in imminent:
            for w in tr.wagons:
                w.pickup_container_ids.intersection_update(due_ids)

        self._assign_pickups_to_trains(due_export, imminent, append_to=None)

    # ---------------- helpers ----------------
    def _export_operator_split(self, todays: List[ScheduledTrain], n_exports: int) -> Dict[str, Dict]:
        if n_exports <= 0 or not todays:
            return {}
        # Count trains per operator
        cnt = defaultdict(int)
        op_to_earliest_angle: Dict[str, float] = {}
        for st in todays:
            cnt[st.operator] += 1
            if st.operator not in op_to_earliest_angle:
                op_to_earliest_angle[st.operator] = st.arrival_angle
            else:
                if st.arrival_angle < op_to_earliest_angle[st.operator]:
                    op_to_earliest_angle[st.operator] = st.arrival_angle

        total = sum(cnt.values())
        per_op = {op: int(round(n_exports * c / total)) for op, c in cnt.items()}
        spill = n_exports - sum(per_op.values())
        ops = list(cnt.keys())
        random.shuffle(ops)
        for i in range(max(0, spill)):
            per_op[ops[i % len(ops)]] += 1

        cfg = {}
        for op, k in per_op.items():
            if k > 0:
                cfg[op] = {"num_containers": k, "arrival_time": {"angle": op_to_earliest_angle[op]}}
        return cfg

    def _assign_pickups_to_trains(self,
                                  containers: List[Container],
                                  trains: List[Train],
                                  append_to: Optional[Dict[str, Dict[int, List[str]]]] = None
                                  ) -> Dict[str, Dict[int, List[str]]]:
        if append_to is None:
            assignments: Dict[str, Dict[int, List[str]]] = {}
        else:
            assignments = append_to
        # free length per wagon
        free_len: Dict[Tuple[str, int], float] = {}
        for tr in trains:
            if tr.train_id not in assignments:
                assignments[tr.train_id] = {}
            for i, w in enumerate(tr.wagons):
                free_len[(tr.train_id, i)] = w.get_available_length()

        for c in containers:
            needed = c.length_m
            placed = False
            for tr in trains:
                for i, w in enumerate(tr.wagons):
                    key = (tr.train_id, i)
                    if free_len.get(key, 0.0) + 1e-3 >= needed:
                        w.add_pickup_container(c.container_id)
                        free_len[key] = max(0.0, free_len[key] - needed)
                        assignments[tr.train_id].setdefault(i, []).append(c.container_id)
                        placed = True
                        break
                if placed:
                    break
        return assignments

    def _last_departure_datetime(self, todays: List[ScheduledTrain], base_day: datetime) -> datetime:
        if not todays:
            return base_day.replace(hour=23, minute=59, second=0, microsecond=0)
        latest = base_day
        for st in todays:
            d, h, m = self.time.decode(st.departure_angle)
            # same day only
            dt = base_day.replace(hour=h, minute=m, second=0, microsecond=0)
            if dt > latest:
                latest = dt
        return latest