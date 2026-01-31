# simulation/operations/gate.py
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

from simulation.core.vehicles.truck import Truck
from simulation.core.containers.container import Container
from simulation.core.factories.container_factory import ContainerFactory
from simulation.core.factories.truck_factory import TruckFactory
from simulation.planning.time_encoder import WeeklyTimeEncoder
from simulation.core.constants import STANDARD_VEHICLE_LENGTH_M
from simulation.core.enums import Direction
from simulation.config.operations_config import GateConfig


@dataclass
class Order:
    """Structure for terminal gate orders."""
    import_containers: List[Container]
    export_operators: Dict[str, Dict]  # operator -> {num_containers: int, arrival_time: dict}


class TerminalGate:
    """
    High-performance terminal gate for generating truck arrivals.
    Processes import/export orders and creates optimized truck schedules.
    """
    
    def __init__(
        self,
        container_factory: ContainerFactory,
        truck_factory: TruckFactory,
        operator_dwell_stats: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the terminal gate.
        
        Args:
            container_factory: Factory for generating containers
            truck_factory: Factory for generating trucks
            operator_dwell_stats: Pre-computed average dwell times by operator
        """
        self.container_factory = container_factory
        self.truck_factory = truck_factory
        self.time_encoder = WeeklyTimeEncoder()
        
        # Pre-compute operator dwell statistics if not provided
        self.operator_dwell_stats = operator_dwell_stats or self._compute_operator_stats()
        self._identify_short_dwell_operators()
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=GateConfig.MAX_WORKERS)
    
    def _compute_operator_stats(self) -> Dict[str, float]:
        """Compute average dwell times for all operators."""
        stats = {}
        
        # Sample containers to estimate dwell times
        for direction in [Direction.IMPORT, Direction.EXPORT]:
            operators = self.container_factory.get_available_operators(direction)
            for operator in operators:
                try:
                    # Generate small sample to estimate dwell
                    samples = self.container_factory.create_containers(
                        operator=operator,
                        direction=direction,
                        n_containers=100
                    )
                    dwell_times = [
                        (c.departure_date - c.arrival_date).days
                        for c in samples
                    ]
                    stats[f"{operator}_{direction.value}"] = np.mean(dwell_times)
                except:
                    continue
        
        return stats
    
    def _identify_short_dwell_operators(self) -> None:
        """Identify operators with short dwell times."""
        if not self.operator_dwell_stats:
            self.short_dwell_operators = set()
            return
        
        # Calculate percentile threshold
        dwell_values = list(self.operator_dwell_stats.values())
        if dwell_values:
            threshold = np.percentile(
                dwell_values,
                GateConfig.DWELL_COMPARISON_PERCENTILE
            )
            threshold = min(threshold, GateConfig.SHORT_DWELL_THRESHOLD_DAYS)
            
            self.short_dwell_operators = {
                op.split('_')[0]  # Extract operator name
                for op, dwell in self.operator_dwell_stats.items()
                if dwell <= threshold and "_Export" in op
            }
        else:
            self.short_dwell_operators = set()
    
    def process_order(
        self,
        order: Order,
        simulation_date: datetime,
        day_of_week: str
    ) -> List[Truck]:
        """
        Process an order and generate all required trucks.
        
        Args:
            order: Order containing import/export requirements
            simulation_date: Current simulation date
            day_of_week: Day of week for truck arrival patterns
            
        Returns:
            List of all trucks scheduled to arrive
        """
        all_trucks = []
        
        # Process imports and exports in parallel
        futures = []
        
        # Submit import processing
        if order.import_containers:
            future = self.executor.submit(
                self._process_imports,
                order.import_containers,
                simulation_date,
                day_of_week
            )
            futures.append(('import', future))
        
        # Submit export processing
        if order.export_operators:
            future = self.executor.submit(
                self._process_exports,
                order.export_operators,
                simulation_date,
                day_of_week
            )
            futures.append(('export', future))
        
        # Collect results
        for task_type, future in futures:
            try:
                trucks = future.result()
                all_trucks.extend(trucks)
            except Exception as e:
                print(f"Error processing {task_type}: {e}")
        
        return all_trucks
    
    def _process_imports(
        self,
        containers: List[Container],
        simulation_date: datetime,
        day_of_week: str
    ) -> List[Truck]:
        """
        Generate pickup trucks 1:1 per container (no bundling).
        
        Args:
            containers: Import containers to generate trucks for
            simulation_date: Current simulation date
            day_of_week: Day of week
            
        Returns:
            List of pickup trucks
        """
        if not containers:
            return []
        
        day_key = day_of_week.lower()
        return self.truck_factory._generate_pickup_trucks(
            containers=containers,
            day_key=day_key,
            base_date=simulation_date
        )
    
    def _process_exports(
        self,
        export_operators: Dict[str, Dict],
        simulation_date: datetime,
        day_of_week: str
    ) -> List[Truck]:
        """
        Generate delivery trucks for export containers.
        
        Args:
            export_operators: Operator configuration
            simulation_date: Current simulation date
            day_of_week: Day of week
            
        Returns:
            List of delivery trucks
        """
        if not export_operators:
            return []
        
        all_trucks: List[Truck] = []
        futures = [
            self.executor.submit(
                self._process_single_operator,
                operator,
                cfg,
                simulation_date,
                day_of_week
            )
            for operator, cfg in export_operators.items()
        ]
        
        for fut in as_completed(futures):
            try:
                trucks = fut.result()
                if trucks:
                    all_trucks.extend(trucks)
            except Exception as e:
                print(f"Error processing operator: {e}")
        
        return all_trucks
    
    def _process_single_operator(
        self,
        operator: str,
        config: Dict,
        simulation_date: datetime,
        day_of_week: str
    ) -> List[Truck]:
        """
        Process a single export operator.
        
        Args:
            operator: Operator name
            config: Configuration dict with num_containers and arrival_time
            simulation_date: Current simulation date
            day_of_week: Day of week
            
        Returns:
            List of delivery trucks for this operator
        """
        num_containers = int(config.get("num_containers", 0))
        if num_containers <= 0:
            return []
        
        # Determine arrival time
        arrival_time = simulation_date
        if operator in self.short_dwell_operators:
            train_arrival = config.get("arrival_time", {})
            if train_arrival:
                angle = float(train_arrival.get("angle", 0.0))
                _day, hour, minute = self.time_encoder.decode(angle)
                arrival_time = simulation_date.replace(
                    hour=hour,
                    minute=minute,
                    second=0,
                    microsecond=0
                )
                arrival_time -= timedelta(hours=GateConfig.SHORT_DWELL_EARLY_ARRIVAL_HOURS)
        
        # Generate export containers (batched)
        containers: List[Container] = []
        for i in range(0, num_containers, GateConfig.CONTAINER_BATCH_SIZE):
            batch_size = min(GateConfig.CONTAINER_BATCH_SIZE, num_containers - i)
            batch = self.container_factory.create_containers(
                operator=operator,
                direction=Direction.EXPORT,
                n_containers=batch_size,
                base_arrival_date=arrival_time
            )
            containers.extend(batch)
        
        # Build delivery trucks
        trucks = self.truck_factory._generate_delivery_trucks(
            containers=containers,
            day_key=day_of_week.lower(),
            base_date=arrival_time,
            parking_spot_prefix="P"
        )
        return trucks
    
    def get_arrived_trucks(
        self,
        trucks: List[Truck],
        current_time: datetime
    ) -> List[Truck]:
        """
        Filter trucks that have arrived by the current time.
        
        Args:
            trucks: List of all scheduled trucks
            current_time: Current simulation time
            
        Returns:
            List of trucks that have arrived
        """
        if not trucks:
            return []
        
        # Vectorized filtering for speed
        arrival_times = np.array([t.arrival_time for t in trucks], dtype=object)
        mask = arrival_times <= current_time
        
        return [truck for truck, arrived in zip(trucks, mask) if arrived]
    
    def create_delivery_trucks_for_operators(
        self,
        export_operators: Dict[str, Dict],
        simulation_date: datetime,
        day_of_week: str
    ) -> List[Truck]:
        """
        Generate export delivery trucks for the given operator config.
        
        Args:
            export_operators: Operator configuration
            simulation_date: Current simulation date
            day_of_week: Day of week
            
        Returns:
            List of delivery trucks
        """
        if not export_operators:
            return []
        
        trucks: List[Truck] = []
        for operator, cfg in export_operators.items():
            num_containers = int(cfg.get("num_containers", 0))
            if num_containers <= 0:
                continue
            
            op_trucks = self._process_single_operator(
                operator,
                cfg,
                simulation_date,
                day_of_week
            )
            if op_trucks:
                trucks.extend(op_trucks)
        
        return trucks
    
    def create_pickup_trucks_after(
        self,
        containers: List[Container],
        earliest_time: datetime,
        day_of_week: str
    ) -> List[Truck]:
        """
        Create import pickup trucks that arrive strictly after earliest_time.
        
        Args:
            containers: Containers to generate trucks for
            earliest_time: Earliest allowed arrival time
            day_of_week: Day of week
            
        Returns:
            List of pickup trucks
        """
        if not containers:
            return []
        
        day_key = day_of_week.lower()
        trucks = self.truck_factory._generate_pickup_trucks(
            containers=containers,
            day_key=day_key,
            base_date=earliest_time
        )
        
        # Enforce 'after' timing
        for truck in trucks or []:
            if truck.arrival_time is None or truck.arrival_time <= earliest_time:
                jitter = np.random.randint(5, 45)  # 5-45 minutes after
                truck.arrival_time = earliest_time + timedelta(minutes=int(jitter))
        
        return trucks
    
    def create_export_trucks_with_buffer(
        self,
        export_operators: Dict[str, Dict],
        simulation_date: datetime,
        day_of_week: str,
        buffer_hours: int = 2
    ) -> List[Truck]:
        """
        Create export delivery trucks (one container per truck).
        
        Args:
            export_operators: Operator configuration
            simulation_date: Current simulation date
            day_of_week: Day of week
            buffer_hours: Hours before train arrival for trucks
            
        Returns:
            List of delivery trucks
        """
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
                earliest_dt = simulation_date.replace(
                    hour=h,
                    minute=m,
                    second=0,
                    microsecond=0
                )
            
            # Generate export containers
            containers = self.container_factory.create_containers(
                operator=operator,
                direction=Direction.EXPORT,
                n_containers=n,
                base_arrival_date=simulation_date
            )
            
            # Split into due today and later
            due_today = [
                c for c in containers
                if c.departure_date and c.departure_date.date() == simulation_date.date()
            ]
            later = [c for c in containers if c not in due_today]
            
            # Due today: arrive buffer_hours before earliest train arrival
            pretrain_arrival = max(
                earliest_dt - timedelta(hours=buffer_hours),
                simulation_date
            )
            for container in due_today:
                truck = Truck(arrival_time=pretrain_arrival)
                truck.is_delivery_truck = True
                truck.is_pickup_truck = False
                truck.add_container(container)
                trucks.append(truck)
            
            # Later: arrive per KDE distribution
            for container in later:
                arr = self.truck_factory._sample_arrival_time(
                    day_key=day_key,
                    is_delivery=True,
                    base_date=simulation_date
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
        day_of_week: str
    ) -> List[Truck]:
        """
        Create import pickup trucks by day-of-week distribution (KDE).
        
        Args:
            containers: Containers to generate trucks for
            simulation_date: Current simulation date
            day_of_week: Day of week
            
        Returns:
            List of pickup trucks
        """
        if not containers:
            return []
        
        day_key = day_of_week.lower()
        return self.truck_factory._generate_pickup_trucks(
            containers=containers,
            day_key=day_key,
            base_date=simulation_date
        )
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.executor.shutdown(wait=True)
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()


# Example usage
if __name__ == "__main__":
    from datetime import datetime
    
    # Initialize factories
    container_factory = ContainerFactory()
    truck_factory = TruckFactory()
    
    # Create terminal gate
    gate = TerminalGate(container_factory, truck_factory)
    
    # Create sample import containers
    import_containers = container_factory.create_containers(
        operator="BOX",
        direction=Direction.IMPORT,
        n_containers=50
    )
    
    # Create sample order
    order = Order(
        import_containers=import_containers,
        export_operators={
            "MET": {
                "num_containers": 30,
                "arrival_time": {"angle": 1.5}
            },
            "BOX": {
                "num_containers": 20,
                "arrival_time": {"angle": 2.0}
            }
        }
    )
    
    # Process order
    simulation_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    trucks = gate.process_order(order, simulation_date, "Monday")
    
    print(f"Generated {len(trucks)} trucks")
    
    # Check arrived trucks
    current_time = simulation_date.replace(hour=10)
    arrived = gate.get_arrived_trucks(trucks, current_time)
    print(f"{len(arrived)} trucks have arrived by 10:00")
    
    # Cleanup
    gate.cleanup()