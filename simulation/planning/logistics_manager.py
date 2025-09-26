# simulation/planning/logistics_manager.py
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import random
from collections import defaultdict

from simulation.planning.driving_plan_parser import DrivingPlanParser
from simulation.planning.train_scheduler import TrainScheduler, TrainSchedule, ScheduledTrain
from simulation.planning.time_encoder import WeeklyTimeEncoder
from simulation.planning.train_loader import TrainLoader

from simulation.operations.gate import TerminalGate
from simulation.core.facilities.yard import BooleanStorageYard
from simulation.core.vehicles.train import Train
from simulation.core.vehicles.truck import Truck
from simulation.core.containers.container import Container

RECALC_WINDOW_MIN = 30

@dataclass
class DayPlan:
    date: datetime
    schedule: TrainSchedule
    todays_trains: List[ScheduledTrain]
    last_departure_dt: datetime
    trucks_today: List[Truck]
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
                 parser: DrivingPlanParser,
                 export_per_import: float = 0.75,
                daily_train_import_cap: Optional[int] = 220):
        self.yard = yard
        self.gate = terminal_gate
        self.loader = train_loader
        self.scheduler = train_scheduler
        self.parser = parser
        self.time = WeeklyTimeEncoder()
        self.export_per_import = max(0.0, float(export_per_import))
        self.daily_train_import_cap = daily_train_import_cap

    def plan_day(self, day_start: datetime, trains_override: Optional[List[Train]] = None) -> DayPlan:
        # 1) Trains for the day
        trains = trains_override if trains_override is not None else self.parser.create_trains()
        schedule = self.scheduler.schedule_trains(trains)
        day_name = day_start.strftime("%A").lower()
        todays_trains = [st for st in schedule.scheduled_trains
                        if self.time.decode(st.arrival_angle)[0] == day_name]

        def due_today(c: Container) -> bool:
            d = c.departure_date
            return (d is not None) and (d.date() == day_start.date())

        # 2) Load imports and rearrange wagons
        for st in todays_trains:
            op = st.operator
            self.loader.load_train(st.train, operator=op, current_date=day_start)
            self.loader.rearrange_wagons_for_goods(st.train, self.yard)

        if self.daily_train_import_cap is not None:
            self._throttle_train_imports([st.train for st in todays_trains], cap_total=int(self.daily_train_import_cap))

        # 3) Imports arriving today
        imports_arriving_today = max(0, sum(st.train.get_container_count() for st in todays_trains))

        # 4) Export containers target = 0.75 × imports (containers), then one container per truck
        target_exports = int(round(self.export_per_import * imports_arriving_today))

        export_cfg = self._export_operator_split(todays_trains, target_exports)
        export_trucks = self.gate.create_export_trucks_with_buffer(
            export_operators=export_cfg,
            simulation_date=day_start,
            day_of_week=day_start.strftime("%A"),
            buffer_hours=2
        )

        export_truck_containers_due_today: List[Container] = []
        for t in export_trucks:
            if t and getattr(t, "is_delivery_truck", False) and t.containers:
                for c in t.containers:
                    if due_today(c):
                        export_truck_containers_due_today.append(c)

        # 5) Pickup assignments for Export containers due today (yard + trucks)
        yard_due_pairs = self.yard.get_containers_departing_on(day_start, one_based_bay=False)
        yard_exports_due_today = []
        for cid, _bay in yard_due_pairs:
            c = self.yard.get_container(cid)
            if c and c.direction == "Export" and due_today(c):
                yard_exports_due_today.append(c)

        pickup_assignments = self._assign_pickups_to_trains(
            containers=yard_exports_due_today + export_truck_containers_due_today,
            trains=[st.train for st in todays_trains],
            append_to=None
        )

        # 6) Import pickup trucks by distribution (due today)
        import_pickup_trucks = []
        for st in todays_trains:
            imports_due_today = []
            for c in st.train.get_all_containers():
                if c.direction == "Import" and due_today(c):
                    imports_due_today.append(c)
            if imports_due_today:
                trucks_for_this_train = self.gate.create_pickup_trucks_by_distribution(
                    containers=imports_due_today,
                    simulation_date=day_start,
                    day_of_week=day_start.strftime("%A")
                )
                import_pickup_trucks.extend(trucks_for_this_train)

        # 7) Combine trucks
        trucks_today = []
        if export_trucks:
            trucks_today.extend([t for t in export_trucks if t])
        if import_pickup_trucks:
            trucks_today.extend([t for t in import_pickup_trucks if t])

        # 8) Last departure
        last_departure_dt = self._last_departure_datetime(todays_trains, day_start)

        return DayPlan(
            date=day_start,
            schedule=schedule,
            todays_trains=todays_trains,
            last_departure_dt=last_departure_dt,
            trucks_today=trucks_today,
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

        due_pairs = self.yard.get_containers_departing_on(now, use_estimated=False, one_based_bay=False)
        due_export = []
        for cid, _ in due_pairs:
            c = self.yard.get_container(cid)
            if c and c.direction == "Export":
                d = c.departure_date
                if d and d.date() == now.date():
                    due_export.append(c)
        due_ids = {c.container_id for c in due_export}

        for tr in imminent:
            for w in tr.wagons:
                w.pickup_container_ids.intersection_update(due_ids)

        self._assign_pickups_to_trains(due_export, imminent, append_to=None)

    # ---------------- helpers ----------------
    def _export_operator_split(self, todays: List[ScheduledTrain], n_exports: int) -> Dict[str, Dict]:
        if n_exports <= 0 or not todays:
            return {}
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
        per_op = {op: int(round(n_exports * c / max(1, total))) for op, c in cnt.items()}
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
        """
        Capacity-agnostic "intent" assignment:
        - Mark Export container IDs as desired by trains (pickup_container_ids) without checking space.
        - Space is enforced at execution time (add_container).
        - Stable round-robin over trains; stable wagon index by hash.
        """
        if append_to is None:
            assignments: Dict[str, Dict[int, List[str]]] = {}
        else:
            assignments = append_to

        if not containers or not trains:
            return assignments

        # Ensure dicts exist
        for tr in trains:
            assignments.setdefault(tr.train_id, {})

        k = len(trains)
        if k <= 0:
            return assignments

        rr = 0
        for c in containers:
            if not c or c.direction != "Export":
                continue
            tr = trains[rr % k]
            rr += 1

            n_w = max(1, len(tr.wagons))
            wi = (abs(hash(c.container_id)) % n_w)
            tr.wagons[wi].add_pickup_container(c.container_id)
            assignments[tr.train_id].setdefault(wi, []).append(c.container_id)

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
    
    def _throttle_train_imports(self, trains: List[Train], cap_total: int) -> None:
        """
        Remove import containers across today's trains until total <= cap_total.
        Balanced strategy:
        - Compute equal target removals per train (+1 for first 'remainder' trains).
        - For each train, remove from alternating wagons (0,2,4,... then 1,3,5,...).
        - Within a wagon, remove the middle container (not first/last) to avoid local bias.
        """
        if not trains:
            return
        # Consider only trains that actually have containers
        active = [tr for tr in trains if tr.get_container_count() > 0]
        if not active:
            return

        total = sum(tr.get_container_count() for tr in active)
        if total <= cap_total:
            return

        to_remove_total = total - cap_total
        n = len(active)
        base = to_remove_total // n
        rem = to_remove_total % n

        targets = [base + (1 if i < rem else 0) for i in range(n)]
        removed_so_far = [0] * n
        remaining = to_remove_total

        # Multiple passes in case some trains can't reach their target due to low count
        pass_idx = 0
        while remaining > 0 and pass_idx < 3:
            changed = False
            for i, tr in enumerate(active):
                need = targets[i] - removed_so_far[i]
                if need <= 0:
                    continue
                took = self._remove_from_train_balanced(tr, need, start_offset=(pass_idx + i) & 1)
                if took > 0:
                    removed_so_far[i] += took
                    remaining -= took
                    changed = True
                if remaining <= 0:
                    break
            if not changed:
                break
            pass_idx += 1

    def _remove_from_train_balanced(self, train: Train, k: int, start_offset: int = 0) -> int:
        """
        Remove up to k containers from a train in a balanced pattern:
        - First sweep: every other wagon starting from start_offset parity (0 or 1).
        - Second sweep: the other parity wagons.
        - Within a wagon, remove the middle container (avoids head/tail bias).
        - Final sweep: round-robin over all wagons if still needed.
        Returns the number actually removed.
        """
        removed = 0
        n_w = len(train.wagons)
        if n_w == 0 or k <= 0:
            return 0

        def pop_middle_from_wagon(w) -> str | None:
            if not w.containers:
                return None
            keys = list(w.containers.keys())
            mid = len(keys) // 2
            return keys[mid]

        par = start_offset & 1
        orders = [list(range(par, n_w, 2)), list(range(1 - par, n_w, 2))]
        for order in orders:
            for wi in order:
                if removed >= k:
                    return removed
                w = train.wagons[wi]
                cid = pop_middle_from_wagon(w)
                if cid is None:
                    continue
                ok = train.remove_container(cid)
                if ok:
                    removed += 1
                if removed >= k:
                    return removed

        # Final sweep if still needed
        for wi in range(n_w):
            if removed >= k:
                break
            w = train.wagons[wi]
            cid = pop_middle_from_wagon(w)
            if cid is None:
                continue
            ok = train.remove_container(cid)
            if ok:
                removed += 1

        return removed