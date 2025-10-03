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
from simulation.core.enums import Direction
from simulation.core.constants import SECONDS_PER_MINUTE


# Configuration
RECALC_WINDOW_MIN = 30  # Recalculate pickup assignments 30 min before train arrival


@dataclass
class DayPlan:
    """Complete plan for a single day's operations."""
    date: datetime
    schedule: TrainSchedule
    todays_trains: List[ScheduledTrain]
    last_departure_dt: datetime
    trucks_today: List[Truck]
    pickup_assignments: Dict[str, Dict[int, List[str]]]  # train_id -> wagon_idx -> [container_id]


class LogisticsManager:
    """
    Day planner for terminal operations.
    
    Responsibilities:
    - Parse driving plan -> schedule trains on tracks
    - Load import containers on trains, rearrange wagons
    - Assign export containers due today from yard to trains (pickup IDs)
    - Generate trucks:
        * Delivery trucks with export containers
        * Pickup trucks for import containers due today in yard
    - Recalculate pickup assignments within 30 minutes before train arrival
    """
    
    def __init__(
        self,
        yard: BooleanStorageYard,
        terminal_gate: TerminalGate,
        train_loader: TrainLoader,
        train_scheduler: TrainScheduler,
        parser: DrivingPlanParser,
        export_per_import: float = 0.75,
        daily_train_import_cap: Optional[int] = 220
    ):
        """
        Initialize logistics manager.
        
        Args:
            yard: Yard storage facility
            terminal_gate: Gate for generating trucks
            train_loader: Loader for train containers
            train_scheduler: Scheduler for train track assignment
            parser: Parser for driving plan
            export_per_import: Ratio of exports to imports
            daily_train_import_cap: Maximum import containers per day (None = no cap)
        """
        self.yard = yard
        self.gate = terminal_gate
        self.loader = train_loader
        self.scheduler = train_scheduler
        self.parser = parser
        self.time = WeeklyTimeEncoder()
        self.export_per_import = max(0.0, float(export_per_import))
        self.daily_train_import_cap = daily_train_import_cap
    
    def plan_day(
        self,
        day_start: datetime,
        trains_override: Optional[List[Train]] = None
    ) -> DayPlan:
        """
        Create complete plan for a day's operations.
        
        Args:
            day_start: Start datetime for the day
            trains_override: Optional train list (overrides driving plan)
            
        Returns:
            DayPlan with all scheduled operations
        """
        # 1) Get trains for the day
        trains = trains_override if trains_override is not None else self.parser.create_trains()
        schedule = self.scheduler.schedule_trains(trains)
        
        day_name = day_start.strftime("%A").lower()
        todays_trains = [
            st for st in schedule.scheduled_trains
            if self.time.decode(st.arrival_angle)[0] == day_name
        ]
        
        # Helper function for departure date check
        def due_today(container: Container) -> bool:
            departure_date = container.departure_date
            return (departure_date is not None) and (departure_date.date() == day_start.date())
        
        # 2) Load imports and rearrange wagons
        for st in todays_trains:
            operator = st.operator
            self.loader.load_train(st.train, operator=operator, current_date=day_start)
            self.loader.rearrange_wagons_for_goods(st.train, self.yard)
        
        # Apply import cap if configured
        if self.daily_train_import_cap is not None:
            self._throttle_train_imports(
                [st.train for st in todays_trains],
                cap_total=int(self.daily_train_import_cap)
            )
        
        # 3) Count imports arriving today
        imports_arriving_today = max(0, sum(
            st.train.get_container_count() for st in todays_trains
        ))
        
        # 4) Generate export trucks
        target_exports = int(round(self.export_per_import * imports_arriving_today))
        export_cfg = self._export_operator_split(todays_trains, target_exports)
        export_trucks = self.gate.create_export_trucks_with_buffer(
            export_operators=export_cfg,
            simulation_date=day_start,
            day_of_week=day_start.strftime("%A"),
            buffer_hours=2
        )
        
        # 5) Collect export containers due today from trucks
        export_truck_containers_due_today: List[Container] = []
        for truck in export_trucks:
            if truck and getattr(truck, "is_delivery_truck", False) and truck.containers:
                for container in truck.containers:
                    if due_today(container):
                        export_truck_containers_due_today.append(container)
        
        # 6) Collect export containers due today from yard
        yard_due_pairs = self.yard.get_containers_departing_on(day_start, one_based_bay=False)
        yard_exports_due_today = []
        for cid, _bay in yard_due_pairs:
            container = self.yard.get_container(cid)
            if container and container.direction == Direction.EXPORT and due_today(container):
                yard_exports_due_today.append(container)
        
        # 7) Assign pickups to trains (for exports due today)
        pickup_assignments = self._assign_pickups_to_trains(
            containers=yard_exports_due_today + export_truck_containers_due_today,
            trains=[st.train for st in todays_trains],
            append_to=None
        )
        
        # 8) Generate pickup trucks for imports due today (on trains)
        import_pickup_trucks: List[Truck] = []
        for st in todays_trains:
            imports_due_today = []
            for container in st.train.get_all_containers():
                if container.direction == Direction.IMPORT and due_today(container):
                    imports_due_today.append(container)
            
            if imports_due_today:
                trucks_for_this_train = self.gate.create_pickup_trucks_by_distribution(
                    containers=imports_due_today,
                    simulation_date=day_start,
                    day_of_week=day_start.strftime("%A")
                )
                import_pickup_trucks.extend(trucks_for_this_train)
        
        # 9) Generate pickup trucks for imports due today (in yard)
        yard_imports_due_today: List[Container] = []
        for cid, _bay in yard_due_pairs:
            container = self.yard.get_container(cid)
            if container and container.direction == Direction.IMPORT and due_today(container):
                yard_imports_due_today.append(container)
        
        if yard_imports_due_today:
            trucks_for_yard_imports = self.gate.create_pickup_trucks_by_distribution(
                containers=yard_imports_due_today,
                simulation_date=day_start,
                day_of_week=day_start.strftime("%A")
            )
            import_pickup_trucks.extend(trucks_for_yard_imports)
        
        # 10) Combine all trucks
        trucks_today: List[Truck] = []
        if export_trucks:
            trucks_today.extend([t for t in export_trucks if t])
        if import_pickup_trucks:
            trucks_today.extend([t for t in import_pickup_trucks if t])
        
        # 11) Calculate last departure time
        last_departure_dt = self._last_departure_datetime(todays_trains, day_start)
        
        return DayPlan(
            date=day_start,
            schedule=schedule,
            todays_trains=todays_trains,
            last_departure_dt=last_departure_dt,
            trucks_today=trucks_today,
            pickup_assignments=pickup_assignments
        )
    
    def recalc_assignments_before_arrival(
        self,
        now: datetime,
        day_plan: DayPlan
    ) -> None:
        """
        Recalculate pickup assignments for trains arriving soon.
        
        Called periodically to update pickup assignments based on current yard state.
        Only affects trains arriving within RECALC_WINDOW_MIN minutes.
        
        Args:
            now: Current simulation time
            day_plan: Current day plan
        """
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
        
        # Get export containers due today from yard
        due_pairs = self.yard.get_containers_departing_on(now, one_based_bay=False)
        due_export = []
        for cid, _ in due_pairs:
            container = self.yard.get_container(cid)
            if container and container.direction == Direction.EXPORT:
                departure_date = container.departure_date
                if departure_date and departure_date.date() == now.date():
                    due_export.append(container)
        
        due_ids = {c.container_id for c in due_export}
        
        # Clean up old assignments for imminent trains
        for train in imminent:
            for wagon in train.wagons:
                wagon.pickup_container_ids.intersection_update(due_ids)
        
        # Reassign
        self._assign_pickups_to_trains(due_export, imminent, append_to=None)
    
    # ==================== Helper Methods ====================
    
    def _export_operator_split(
        self,
        todays: List[ScheduledTrain],
        n_exports: int
    ) -> Dict[str, Dict]:
        """
        Split export containers among operators proportionally.
        
        Args:
            todays: Today's scheduled trains
            n_exports: Total number of exports to generate
            
        Returns:
            Dict mapping operator to config {num_containers, arrival_time}
        """
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
        
        # Proportional split
        total = sum(cnt.values())
        per_op = {
            op: int(round(n_exports * c / max(1, total)))
            for op, c in cnt.items()
        }
        
        # Distribute remainder randomly
        spill = n_exports - sum(per_op.values())
        ops = list(cnt.keys())
        random.shuffle(ops)
        for i in range(max(0, spill)):
            per_op[ops[i % len(ops)]] += 1
        
        # Build config
        cfg = {}
        for op, k in per_op.items():
            if k > 0:
                cfg[op] = {
                    "num_containers": k,
                    "arrival_time": {"angle": op_to_earliest_angle[op]}
                }
        
        return cfg
    
    def _assign_pickups_to_trains(
        self,
        containers: List[Container],
        trains: List[Train],
        append_to: Optional[Dict[str, Dict[int, List[str]]]] = None
    ) -> Dict[str, Dict[int, List[str]]]:
        """
        Capacity-agnostic "intent" assignment.
        
        Marks export container IDs as desired by trains (pickup_container_ids)
        without checking space. Space is enforced at execution time.
        
        Args:
            containers: Containers to assign
            trains: Trains to assign to
            append_to: Existing assignments to append to (None = new dict)
            
        Returns:
            Dict mapping train_id -> wagon_idx -> [container_id]
        """
        if append_to is None:
            assignments: Dict[str, Dict[int, List[str]]] = {}
        else:
            assignments = append_to
        
        if not containers or not trains:
            return assignments
        
        # Ensure dicts exist
        for train in trains:
            assignments.setdefault(train.train_id, {})
        
        k = len(trains)
        if k <= 0:
            return assignments
        
        # Round-robin assignment
        rr = 0
        for container in containers:
            if not container or container.direction != Direction.EXPORT:
                continue
            
            train = trains[rr % k]
            rr += 1
            
            # Hash-based wagon selection (stable)
            n_wagons = max(1, len(train.wagons))
            wagon_idx = (abs(hash(container.container_id)) % n_wagons)
            
            train.wagons[wagon_idx].add_pickup_container(container.container_id)
            assignments[train.train_id].setdefault(wagon_idx, []).append(container.container_id)
        
        return assignments
    
    def _last_departure_datetime(
        self,
        todays: List[ScheduledTrain],
        base_day: datetime
    ) -> datetime:
        """
        Find the latest departure time for today's trains.
        
        Args:
            todays: Today's scheduled trains
            base_day: Base datetime for the day
            
        Returns:
            Latest departure datetime
        """
        if not todays:
            return base_day.replace(hour=23, minute=59, second=0, microsecond=0)
        
        latest = base_day
        for st in todays:
            d, h, m = self.time.decode(st.departure_angle)
            # Same day only
            dt = base_day.replace(hour=h, minute=m, second=0, microsecond=0)
            if dt > latest:
                latest = dt
        
        return latest
    
    def _throttle_train_imports(
        self,
        trains: List[Train],
        cap_total: int
    ) -> None:
        """
        Remove import containers across today's trains until total <= cap_total.
        
        Balanced strategy:
        - Compute equal target removals per train (+1 for first 'remainder' trains)
        - For each train, remove from alternating wagons (0,2,4,... then 1,3,5,...)
        - Within a wagon, remove the middle container (not first/last)
        
        Args:
            trains: Trains to throttle
            cap_total: Maximum total containers across all trains
        """
        if not trains:
            return
        
        # Consider only trains that have containers
        active = [train for train in trains if train.get_container_count() > 0]
        if not active:
            return
        
        total = sum(train.get_container_count() for train in active)
        if total <= cap_total:
            return
        
        to_remove_total = total - cap_total
        n = len(active)
        base = to_remove_total // n
        rem = to_remove_total % n
        
        targets = [base + (1 if i < rem else 0) for i in range(n)]
        removed_so_far = [0] * n
        remaining = to_remove_total
        
        # Multiple passes if some trains can't reach target
        pass_idx = 0
        while remaining > 0 and pass_idx < 3:
            changed = False
            for i, train in enumerate(active):
                need = targets[i] - removed_so_far[i]
                if need <= 0:
                    continue
                
                took = self._remove_from_train_balanced(train, need, start_offset=(pass_idx + i) & 1)
                if took > 0:
                    removed_so_far[i] += took
                    remaining -= took
                    changed = True
                
                if remaining <= 0:
                    break
            
            if not changed:
                break
            pass_idx += 1
    
    def _remove_from_train_balanced(
        self,
        train: Train,
        k: int,
        start_offset: int = 0
    ) -> int:
        """
        Remove up to k containers from a train in a balanced pattern.
        
        Pattern:
        - First sweep: every other wagon starting from start_offset parity
        - Second sweep: the other parity wagons
        - Within a wagon, remove the middle container (avoids head/tail bias)
        - Final sweep: round-robin over all wagons if still needed
        
        Args:
            train: Train to remove from
            k: Number of containers to remove
            start_offset: Starting parity (0 or 1)
            
        Returns:
            Number of containers actually removed
        """
        removed = 0
        n_wagons = len(train.wagons)
        if n_wagons == 0 or k <= 0:
            return 0
        
        def pop_middle_from_wagon(wagon) -> Optional[str]:
            """Pop middle container ID from wagon."""
            if not wagon.containers:
                return None
            keys = list(wagon.containers.keys())
            mid = len(keys) // 2
            return keys[mid]
        
        parity = start_offset & 1
        orders = [list(range(parity, n_wagons, 2)), list(range(1 - parity, n_wagons, 2))]
        
        # Two passes with alternating wagons
        for order in orders:
            for wagon_idx in order:
                if removed >= k:
                    return removed
                
                wagon = train.wagons[wagon_idx]
                cid = pop_middle_from_wagon(wagon)
                if cid is None:
                    continue
                
                ok = train.remove_container(cid)
                if ok:
                    removed += 1
                if removed >= k:
                    return removed
        
        # Final sweep if still needed
        for wagon_idx in range(n_wagons):
            if removed >= k:
                break
            
            wagon = train.wagons[wagon_idx]
            cid = pop_middle_from_wagon(wagon)
            if cid is None:
                continue
            
            ok = train.remove_container(cid)
            if ok:
                removed += 1
        
        return removed