# simulation/operations/gate.py
"""Terminal gate: generates truck arrivals from import/export orders."""
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass

from simulation.core.vehicles.truck import Truck
from simulation.core.containers.container import Container
from simulation.core.factories.container_factory import ContainerFactory
from simulation.core.factories.truck_factory import TruckFactory
from simulation.planning.time_encoder import WeeklyTimeEncoder
from simulation.core.enums import Direction
from simulation.config.operations_config import GateConfig


@dataclass
class Order:
    """Terminal gate order."""
    import_containers: List[Container]
    export_operators: Dict[str, Dict]  # operator -> {num_containers, arrival_time}


class TerminalGate:
    """Generates truck arrivals from import/export orders."""

    def __init__(
        self,
        container_factory: ContainerFactory,
        truck_factory: TruckFactory,
        operator_dwell_stats: Optional[Dict[str, float]] = None,
    ):
        self.container_factory = container_factory
        self.truck_factory = truck_factory
        self.time_encoder = WeeklyTimeEncoder()
        self.operator_dwell_stats = operator_dwell_stats or self._compute_operator_stats()
        self._identify_short_dwell_operators()

    # ----- Init helpers -----

    def _compute_operator_stats(self) -> Dict[str, float]:
        """Sample containers to estimate per-operator dwell times."""
        stats: Dict[str, float] = {}
        SAMPLE_SIZE = 100

        for direction in (Direction.IMPORT, Direction.EXPORT):
            for operator in self.container_factory.get_available_operators(direction):
                try:
                    samples = self.container_factory.create_containers(
                        operator=operator,
                        direction=direction,
                        n_containers=SAMPLE_SIZE,
                    )
                    dwell_days = [(c.departure_date - c.arrival_date).days for c in samples]
                    stats[f"{operator}_{direction.value}"] = np.mean(dwell_days)
                except (AttributeError, TypeError, ValueError):
                    continue
        return stats

    def _identify_short_dwell_operators(self) -> None:
        """Tag operators with short export dwell times."""
        if not self.operator_dwell_stats:
            self.short_dwell_operators: set = set()
            return

        dwell_values = list(self.operator_dwell_stats.values())
        if not dwell_values:
            self.short_dwell_operators = set()
            return

        threshold = min(
            np.percentile(dwell_values, GateConfig.DWELL_COMPARISON_PERCENTILE),
            GateConfig.SHORT_DWELL_THRESHOLD_DAYS,
        )
        self.short_dwell_operators = {
            op.split("_")[0]
            for op, dwell in self.operator_dwell_stats.items()
            if dwell <= threshold and "_Export" in op
        }

    # ----- Order processing -----

    def process_order(
        self,
        order: Order,
        simulation_date: datetime,
        day_of_week: str,
    ) -> List[Truck]:
        """Process an order and return all scheduled trucks."""
        trucks: List[Truck] = []

        if order.import_containers:
            trucks.extend(
                self._process_imports(order.import_containers, simulation_date, day_of_week)
            )
        if order.export_operators:
            trucks.extend(
                self._process_exports(order.export_operators, simulation_date, day_of_week)
            )
        return trucks

    def _process_imports(
        self,
        containers: List[Container],
        simulation_date: datetime,
        day_of_week: str,
    ) -> List[Truck]:
        """Generate one pickup truck per import container."""
        if not containers:
            return []
        return self.truck_factory._generate_pickup_trucks(
            containers=containers,
            day_key=day_of_week.lower(),
            base_date=simulation_date,
        )

    def _process_exports(
        self,
        export_operators: Dict[str, Dict],
        simulation_date: datetime,
        day_of_week: str,
    ) -> List[Truck]:
        """Generate delivery trucks for all export operators (sequential)."""
        trucks: List[Truck] = []
        for operator, cfg in export_operators.items():
            op_trucks = self._process_single_operator(
                operator, cfg, simulation_date, day_of_week
            )
            if op_trucks:
                trucks.extend(op_trucks)
        return trucks

    def _process_single_operator(
        self,
        operator: str,
        config: Dict,
        simulation_date: datetime,
        day_of_week: str,
    ) -> List[Truck]:
        """Process a single export operator."""
        num_containers = int(config.get("num_containers", 0))
        if num_containers <= 0:
            return []

        arrival_time = simulation_date
        if operator in self.short_dwell_operators:
            train_arrival = config.get("arrival_time", {})
            if train_arrival:
                angle = float(train_arrival.get("angle", 0.0))
                _day, hour, minute = self.time_encoder.decode(angle)
                arrival_time = simulation_date.replace(
                    hour=hour, minute=minute, second=0, microsecond=0
                )
                arrival_time -= timedelta(hours=GateConfig.SHORT_DWELL_EARLY_ARRIVAL_HOURS)

        # Batched container generation
        containers: List[Container] = []
        for i in range(0, num_containers, GateConfig.CONTAINER_BATCH_SIZE):
            batch_size = min(GateConfig.CONTAINER_BATCH_SIZE, num_containers - i)
            containers.extend(
                self.container_factory.create_containers(
                    operator=operator,
                    direction=Direction.EXPORT,
                    n_containers=batch_size,
                    base_arrival_date=arrival_time,
                )
            )

        return self.truck_factory._generate_delivery_trucks(
            containers=containers,
            day_key=day_of_week.lower(),
            base_date=arrival_time,
            parking_spot_prefix="P",
        )

    # ----- Filtering -----

    def get_arrived_trucks(
        self,
        trucks: List[Truck],
        current_time: datetime,
    ) -> List[Truck]:
        """Filter trucks that have arrived by current_time."""
        if not trucks:
            return []
        return [t for t in trucks if t.arrival_time is not None and t.arrival_time <= current_time]

    # ----- Convenience methods -----

    def create_delivery_trucks_for_operators(
        self,
        export_operators: Dict[str, Dict],
        simulation_date: datetime,
        day_of_week: str,
    ) -> List[Truck]:
        """Generate export delivery trucks for given operator config."""
        return self._process_exports(export_operators, simulation_date, day_of_week)

    def create_pickup_trucks_after(
        self,
        containers: List[Container],
        earliest_time: datetime,
        day_of_week: str,
    ) -> List[Truck]:
        """Create pickup trucks arriving strictly after earliest_time."""
        if not containers:
            return []

        trucks = self.truck_factory._generate_pickup_trucks(
            containers=containers,
            day_key=day_of_week.lower(),
            base_date=earliest_time,
        )
        for truck in trucks or []:
            if truck.arrival_time is None or truck.arrival_time <= earliest_time:
                jitter = np.random.randint(GateConfig.JITTER_MIN_MINUTES, GateConfig.JITTER_MAX_MINUTES)
                truck.arrival_time = earliest_time + timedelta(minutes=int(jitter))
        return trucks

    def create_export_trucks_with_buffer(
        self,
        export_operators: Dict[str, Dict],
        simulation_date: datetime,
        day_of_week: str,
        buffer_hours: int = 2,
    ) -> List[Truck]:
        """Create export delivery trucks with pre-train buffer."""
        trucks: List[Truck] = []
        day_key = day_of_week.lower()

        for operator, cfg in (export_operators or {}).items():
            n = int(cfg.get("num_containers", 0))
            if n <= 0:
                continue

            # Decode earliest train arrival
            earliest_dt = simulation_date
            angle = cfg.get("arrival_time", {}).get("angle")
            if angle is not None:
                _, h, m = self.time_encoder.decode(angle)
                earliest_dt = simulation_date.replace(hour=h, minute=m, second=0, microsecond=0)

            containers = self.container_factory.create_containers(
                operator=operator,
                direction=Direction.EXPORT,
                n_containers=n,
                base_arrival_date=simulation_date,
            )

            pretrain_arrival = max(
                earliest_dt - timedelta(hours=buffer_hours),
                simulation_date,
            )
            for container in containers:
                is_due_today = (
                    container.departure_date
                    and container.departure_date.date() == simulation_date.date()
                )
                if is_due_today:
                    arr = pretrain_arrival
                else:
                    arr = self.truck_factory._sample_arrival_time(
                        day_key=day_key, is_delivery=True, base_date=simulation_date
                    )
                truck = Truck(arrival_time=arr)
                truck.is_delivery_truck = True
                truck.is_pickup_truck = False
                truck.add_container(container)
                trucks.append(truck)

        return trucks

    def create_pickup_trucks_by_distribution(
        self,
        containers: List[Container],
        simulation_date: datetime,
        day_of_week: str,
    ) -> List[Truck]:
        """Create pickup trucks from KDE distribution."""
        if not containers:
            return []
        return self.truck_factory._generate_pickup_trucks(
            containers=containers,
            day_key=day_of_week.lower(),
            base_date=simulation_date,
        )