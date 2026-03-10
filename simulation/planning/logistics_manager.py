# simulation/planning/logistics_manager.py
"""Day planner for terminal operations."""
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
from simulation.core.facilities.yard import StorageYard
from simulation.core.vehicles.train import Train
from simulation.core.vehicles.truck import Truck
from simulation.core.containers.container import Container
from simulation.core.enums import Direction
from simulation.config.operations_config import OperationsDefaults


@dataclass
class DayPlan:
    """Complete plan for a single day's operations."""
    date: datetime
    schedule: TrainSchedule
    todays_trains: List[ScheduledTrain]
    last_departure_dt: datetime
    trucks_today: List[Truck]
    pickup_assignments: Dict[str, Dict[int, List[str]]]


class LogisticsManager:
    """
    Day planner: schedule trains, load imports, assign exports,
    generate trucks, recalculate assignments before train arrival.
    """

    def __init__(
        self,
        yard: StorageYard,
        terminal_gate: TerminalGate,
        train_loader: TrainLoader,
        train_scheduler: TrainScheduler,
        parser: DrivingPlanParser,
        export_per_import: float = 0.75,
        daily_train_import_cap: Optional[int] = 220,
    ):
        self.yard = yard
        self.gate = terminal_gate
        self.loader = train_loader
        self.scheduler = train_scheduler
        self.parser = parser
        self.time = WeeklyTimeEncoder()
        self.export_per_import = max(0.0, float(export_per_import))
        self.daily_train_import_cap = daily_train_import_cap
        self.yard_tracks = train_scheduler.num_tracks

    # ================================================================
    # Day planning
    # ================================================================

    def plan_day(
        self,
        day_start: datetime,
        trains_override: Optional[List[Train]] = None,
    ) -> DayPlan:
        """Create complete plan for a day's operations."""
        trains = trains_override if trains_override is not None else self.parser.create_trains()
        schedule = self.scheduler.schedule_trains(trains)

        day_name = day_start.strftime("%A").lower()
        day_of_week = day_start.strftime("%A")
        today_date = day_start.date()

        todays_trains = [
            st for st in schedule.scheduled_trains
            if self.time.decode(st.arrival_angle)[0] == day_name
        ]

        # 1) Load imports and rearrange wagons
        for st in todays_trains:
            self.loader.load_train(st.train, operator=st.operator, current_date=day_start)
            self.loader.rearrange_wagons_for_goods(st.train, self.yard)

        if self.daily_train_import_cap is not None:
            self._throttle_train_imports(
                [st.train for st in todays_trains],
                cap_total=int(self.daily_train_import_cap),
            )

        # 2) Count imports -> derive export target
        imports_arriving = max(0, sum(
            st.train.get_container_count() for st in todays_trains
        ))
        target_exports = int(round(self.export_per_import * imports_arriving))
        export_cfg = self._export_operator_split(todays_trains, target_exports)
        export_trucks = self.gate.create_export_trucks_with_buffer(
            export_operators=export_cfg,
            simulation_date=day_start,
            day_of_week=day_of_week,
            buffer_hours=2,
        )

        # 3) Collect export containers due today from delivery trucks
        truck_exports_due: List[Container] = [
            c for t in (export_trucks or [])
            if t and getattr(t, "is_delivery_truck", False) and t.containers
            for c in t.containers
            if c.departure_date and c.departure_date.date() == today_date
        ]

        # 4) Single pass over yard: partition departing containers into export / import
        yard_exports_due, yard_imports_due = self._collect_yard_departing(today_date)

        # 5) Assign pickups to trains (proximity-aware)
        pickup_assignments = self._assign_pickups_to_trains(
            containers=yard_exports_due + truck_exports_due,
            trains=[st.train for st in todays_trains],
            scheduled_trains=todays_trains,
        )

        # 6) Generate pickup trucks for imports due today (on trains)
        import_pickup_trucks: List[Truck] = []
        for st in todays_trains:
            imports_due = [
                c for c in st.train.get_all_containers()
                if c.direction == Direction.IMPORT
                and c.departure_date and c.departure_date.date() == today_date
            ]
            if imports_due:
                import_pickup_trucks.extend(
                    self.gate.create_pickup_trucks_by_distribution(
                        containers=imports_due,
                        simulation_date=day_start,
                        day_of_week=day_of_week,
                    )
                )

        # 7) Generate pickup trucks for imports due today (in yard)
        if yard_imports_due:
            import_pickup_trucks.extend(
                self.gate.create_pickup_trucks_by_distribution(
                    containers=yard_imports_due,
                    simulation_date=day_start,
                    day_of_week=day_of_week,
                )
            )

        # 8) Combine all trucks
        trucks_today = [t for t in (export_trucks or []) if t]
        trucks_today.extend(t for t in import_pickup_trucks if t)

        return DayPlan(
            date=day_start,
            schedule=schedule,
            todays_trains=todays_trains,
            last_departure_dt=self._last_departure_datetime(todays_trains, day_start),
            trucks_today=trucks_today,
            pickup_assignments=pickup_assignments,
        )

    # ================================================================
    # Recalculation
    # ================================================================

    def recalc_assignments_before_arrival(
        self,
        now: datetime,
        day_plan: DayPlan,
    ) -> None:
        """Recalculate pickup assignments for trains arriving within recalc window."""
        horizon = now + timedelta(minutes=OperationsDefaults.RECALC_WINDOW_MINUTES)
        day_name = now.strftime("%A").lower()

        imminent_trains: List[Train] = []
        imminent_scheduled: List[ScheduledTrain] = []
        for st in day_plan.todays_trains:
            d, h, m = self.time.decode(st.arrival_angle)
            if d != day_name:
                continue
            arr_dt = now.replace(hour=h, minute=m, second=0, microsecond=0)
            if now <= arr_dt <= horizon:
                imminent_trains.append(st.train)
                imminent_scheduled.append(st)

        if not imminent_trains:
            return

        today_date = now.date()
        yard_exports_due, _ = self._collect_yard_departing(today_date)
        due_ids = {c.container_id for c in yard_exports_due}

        for train in imminent_trains:
            for wagon in train.wagons:
                wagon.pickup_container_ids.intersection_update(due_ids)

        self._assign_pickups_to_trains(
            yard_exports_due, imminent_trains,
            scheduled_trains=imminent_scheduled,
        )

    # ================================================================
    # Yard departure collection
    # ================================================================

    def _collect_yard_departing(
        self,
        target_date,
    ) -> tuple:
        """
        Single pass over yard records to collect containers departing on target_date.
        Returns (export_containers, import_containers).
        """
        exports: List[Container] = []
        imports: List[Container] = []

        for rec in self.yard.iter_records():
            container = rec.container
            dep = container.departure_date
            if dep is None or dep.date() != target_date:
                continue
            if container.direction == Direction.EXPORT:
                exports.append(container)
            elif container.direction == Direction.IMPORT:
                imports.append(container)

        return exports, imports

    # ================================================================
    # Helpers
    # ================================================================

    def _export_operator_split(
        self,
        todays: List[ScheduledTrain],
        n_exports: int,
    ) -> Dict[str, Dict]:
        """Split export containers among operators proportionally."""
        if n_exports <= 0 or not todays:
            return {}

        cnt: Dict[str, int] = defaultdict(int)
        earliest_angle: Dict[str, float] = {}

        for st in todays:
            cnt[st.operator] += 1
            if st.operator not in earliest_angle or st.arrival_angle < earliest_angle[st.operator]:
                earliest_angle[st.operator] = st.arrival_angle

        total = sum(cnt.values())
        per_op = {
            op: int(round(n_exports * c / max(1, total)))
            for op, c in cnt.items()
        }

        # Distribute remainder randomly
        spill = n_exports - sum(per_op.values())
        if spill > 0:
            ops = list(cnt.keys())
            random.shuffle(ops)
            for i in range(spill):
                per_op[ops[i % len(ops)]] += 1

        return {
            op: {"num_containers": k, "arrival_time": {"angle": earliest_angle[op]}}
            for op, k in per_op.items()
            if k > 0
        }

    def _assign_pickups_to_trains(
        self,
        containers: List[Container],
        trains: List[Train],
        scheduled_trains: Optional[List[ScheduledTrain]] = None,
        append_to: Optional[Dict[str, Dict[int, List[str]]]] = None,
    ) -> Dict[str, Dict[int, List[str]]]:
        """
        Proximity-aware pickup assignment: assign each export container to the
        nearest train (by anchor bay), then to the nearest wagon within that train.

        For containers already in the yard, distance is measured from their bay
        position to the train's anchor bay.  For containers not yet placed (arriving
        via truck), round-robin across trains is used as a fallback.
        """
        assignments = append_to if append_to is not None else {}

        if not containers or not trains:
            return assignments

        for train in trains:
            assignments.setdefault(train.train_id, {})

        # Build train → anchor bay mapping for proximity
        n_bays = self.yard.n_bays
        num_tracks = self.yard_tracks
        train_anchors: Dict[str, int] = {}
        if scheduled_trains:
            for st in scheduled_trains:
                n_wagons = len(st.train.wagons) if st.train.wagons else 1
                anchor = int(round((st.track_id + 1) * (n_bays / (num_tracks + 1))))
                anchor = max(0, min(n_bays - n_wagons, anchor))
                train_anchors[st.train.train_id] = anchor

        # Build container → bay mapping from yard records
        container_bays: Dict[str, int] = {}
        for rec in self.yard.iter_records():
            container_bays[rec.container.container_id] = rec.placement.bay

        rr = 0
        k = len(trains)
        for container in containers:
            if not container or container.direction != Direction.EXPORT:
                continue

            cid = container.container_id
            container_bay = container_bays.get(cid)

            if container_bay is not None and train_anchors:
                # Proximity: pick the closest train by anchor bay distance
                best_train = min(
                    trains,
                    key=lambda t: abs(train_anchors.get(t.train_id, n_bays) - container_bay),
                )
            else:
                # Fallback: round-robin for containers not yet in yard
                best_train = trains[rr % k]
                rr += 1

            # Assign to nearest wagon within the chosen train
            n_wagons = max(1, len(best_train.wagons))
            if container_bay is not None and best_train.train_id in train_anchors:
                anchor = train_anchors[best_train.train_id]
                # Nearest wagon = offset from anchor closest to container bay
                wagon_idx = max(0, min(n_wagons - 1, container_bay - anchor))
            else:
                wagon_idx = abs(hash(cid)) % n_wagons

            best_train.wagons[wagon_idx].add_pickup_container(cid)
            assignments[best_train.train_id].setdefault(wagon_idx, []).append(cid)

        return assignments

    def _last_departure_datetime(
        self,
        todays: List[ScheduledTrain],
        base_day: datetime,
    ) -> datetime:
        """Find the latest departure time for today's trains."""
        if not todays:
            return base_day.replace(hour=23, minute=59, second=0, microsecond=0)

        latest = base_day
        for st in todays:
            _d, h, m = self.time.decode(st.departure_angle)
            dt = base_day.replace(hour=h, minute=m, second=0, microsecond=0)
            if dt > latest:
                latest = dt
        return latest

    def _throttle_train_imports(self, trains: List[Train], cap_total: int) -> None:
        """Remove imports across trains until total <= cap_total (balanced)."""
        active = [t for t in trains if t.get_container_count() > 0]
        if not active:
            return

        total = sum(t.get_container_count() for t in active)
        if total <= cap_total:
            return

        to_remove = total - cap_total
        n = len(active)
        base_per = to_remove // n
        rem = to_remove % n
        targets = [base_per + (1 if i < rem else 0) for i in range(n)]
        removed_so_far = [0] * n
        remaining = to_remove

        for pass_idx in range(3):
            changed = False
            for i, train in enumerate(active):
                need = targets[i] - removed_so_far[i]
                if need <= 0:
                    continue
                took = self._remove_from_train_balanced(
                    train, need, start_offset=(pass_idx + i) & 1
                )
                if took > 0:
                    removed_so_far[i] += took
                    remaining -= took
                    changed = True
                if remaining <= 0:
                    return
            if not changed:
                break

    @staticmethod
    def _remove_from_train_balanced(
        train: Train,
        k: int,
        start_offset: int = 0,
    ) -> int:
        """Remove up to k containers from alternating wagons (middle-first)."""
        n_wagons = len(train.wagons)
        if n_wagons == 0 or k <= 0:
            return 0

        def _pop_middle(wagon) -> Optional[str]:
            if not wagon.containers:
                return None
            keys = list(wagon.containers.keys())
            return keys[len(keys) // 2]

        removed = 0
        parity = start_offset & 1
        orders = [
            range(parity, n_wagons, 2),
            range(1 - parity, n_wagons, 2),
            range(n_wagons),
        ]

        for order in orders:
            for w_idx in order:
                if removed >= k:
                    return removed
                cid = _pop_middle(train.wagons[w_idx])
                if cid and train.remove_container(cid):
                    removed += 1

        return removed