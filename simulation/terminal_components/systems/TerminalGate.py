# simulation/terminal_components/systems/TerminalGate.py

import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from simulation.terminal_components.vehicles.Truck import Truck
from simulation.terminal_components.storage_units.Container import Container
from simulation.terminal_components.storage_units.ContainerFactory import ContainerFactory
from simulation.terminal_components.vehicles.TruckFactory import TruckFactory
from simulation.terminal_components.systems.train_tools.TimeEncoder import WeeklyTimeEncoder

# ==================== CONSTANTS ====================
# Container type groups
SPECIAL_CONTAINER_TYPES = {"Trailer", "Swap Body"}

# Truck capacity
TRUCK_MAX_LENGTH = 24.4  # 80 feet in meters

# Dwell time thresholds
SHORT_DWELL_THRESHOLD = 3.0  # Days - operators with avg dwell < this are "short dwell"
DWELL_COMPARISON_PERCENTILE = 25  # Compare to 25th percentile of all operators

# Early arrival settings for short-dwell operators
SHORT_DWELL_EARLY_ARRIVAL_HOURS = 12  # Arrive 12 hours before train

# Parallelization
MAX_WORKERS = 4  # Thread pool size for parallel processing

# Batch sizes for efficient processing
CONTAINER_BATCH_SIZE = 1000  # Process containers in batches of this size


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
    
    def __init__(self,
                 container_factory: ContainerFactory,
                 truck_factory: TruckFactory,
                 operator_dwell_stats: Optional[Dict[str, float]] = None):
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
        self.executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    
    def _compute_operator_stats(self) -> Dict[str, float]:
        """Compute average dwell times for all operators."""
        stats = {}
        
        # Sample containers to estimate dwell times
        for direction in ["Import", "Export"]:
            operators = self.container_factory.get_available_operators(direction)
            for operator in operators:
                try:
                    # Generate small sample to estimate dwell
                    samples = self.container_factory.create_containers(
                        operator=operator,
                        direction=direction,
                        n_containers=100
                    )
                    dwell_times = [(c.departure_date - c.arrival_date).days 
                                  for c in samples]
                    stats[f"{operator}_{direction}"] = np.mean(dwell_times)
                except:
                    continue
        
        return stats
    
    def _identify_short_dwell_operators(self):
        """Identify operators with short dwell times."""
        if not self.operator_dwell_stats:
            self.short_dwell_operators = set()
            return
        
        # Calculate percentile threshold
        dwell_values = list(self.operator_dwell_stats.values())
        if dwell_values:
            threshold = np.percentile(dwell_values, DWELL_COMPARISON_PERCENTILE)
            threshold = min(threshold, SHORT_DWELL_THRESHOLD)
            
            self.short_dwell_operators = {
                op.split('_')[0]  # Extract operator name
                for op, dwell in self.operator_dwell_stats.items()
                if dwell <= threshold and "_Export" in op
            }
        else:
            self.short_dwell_operators = set()
    
    def process_order(self, 
                     order: Order,
                     simulation_date: datetime,
                     day_of_week: str) -> List[Truck]:
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
    
    def _process_imports(self,
                        containers: List[Container],
                        simulation_date: datetime,
                        day_of_week: str) -> List[Truck]:
        """
        Generate pickup trucks for import containers.
        Uses vectorized operations for speed.
        """
        if not containers:
            return []
        
        # Separate special containers using numpy for speed
        container_array = np.array(containers, dtype=object)
        types = np.array([c.container_type for c in containers])
        
        # Vectorized classification
        is_special = np.isin(types, list(SPECIAL_CONTAINER_TYPES))
        
        special_containers = container_array[is_special].tolist()
        regular_containers = container_array[~is_special].tolist()
        
        trucks = []
        
        # Generate trucks for special containers (one truck each)
        for container in special_containers:
            truck = self._create_single_pickup_truck(
                [container], simulation_date, day_of_week
            )
            trucks.append(truck)
        
        # Bundle regular containers efficiently
        if regular_containers:
            bundles = self._bundle_containers_vectorized(regular_containers)
            
            # Create trucks for bundles in parallel batches
            batch_size = max(1, len(bundles) // MAX_WORKERS)
            bundle_batches = [bundles[i:i+batch_size] 
                            for i in range(0, len(bundles), batch_size)]
            
            with ThreadPoolExecutor(max_workers=min(len(bundle_batches), MAX_WORKERS)) as executor:
                futures = [
                    executor.submit(self._create_pickup_trucks_batch, 
                                  batch, simulation_date, day_of_week)
                    for batch in bundle_batches
                ]
                
                for future in as_completed(futures):
                    trucks.extend(future.result())
        
        return trucks
    
    def _bundle_containers_vectorized(self, containers: List[Container]) -> List[List[Container]]:
        """
        Bundle containers into truck-sized groups using vectorized operations.
        Optimized for speed using numpy.
        """
        if not containers:
            return []
        
        # Extract lengths as numpy array for fast operations
        lengths = np.array([c.length_m for c in containers])
        n = len(containers)
        
        # Sort by length descending for better packing
        sorted_indices = np.argsort(-lengths)
        
        bundles = []
        used = np.zeros(n, dtype=bool)
        
        for i in sorted_indices:
            if used[i]:
                continue
            
            # Start new bundle
            bundle_indices = [i]
            bundle_length = lengths[i]
            used[i] = True
            
            # Vectorized search for containers that fit
            remaining_space = TRUCK_MAX_LENGTH - bundle_length
            fits = (lengths <= remaining_space) & (~used)
            
            if np.any(fits):
                # Get indices that fit, sorted by length descending
                fit_indices = np.where(fits)[0]
                fit_lengths = lengths[fit_indices]
                fit_sorted = fit_indices[np.argsort(-fit_lengths)]
                
                # Greedily pack
                for j in fit_sorted:
                    if bundle_length + lengths[j] <= TRUCK_MAX_LENGTH:
                        bundle_indices.append(j)
                        bundle_length += lengths[j]
                        used[j] = True
                        
                        if bundle_length >= TRUCK_MAX_LENGTH * 0.95:  # 95% full
                            break
            
            bundles.append([containers[idx] for idx in bundle_indices])
        
        return bundles
    
    def _create_pickup_trucks_batch(self,
                                   bundles: List[List[Container]],
                                   simulation_date: datetime,
                                   day_of_week: str) -> List[Truck]:
        """Create pickup trucks for a batch of container bundles."""
        trucks = []
        for bundle in bundles:
            truck = self._create_single_pickup_truck(bundle, simulation_date, day_of_week)
            trucks.append(truck)
        return trucks
    
    def _create_single_pickup_truck(self,
                                   containers: List[Container],
                                   simulation_date: datetime,
                                   day_of_week: str) -> Truck:
        """Create a single pickup truck."""
        # Use truck factory to generate pickup truck
        pickup_trucks = self.truck_factory._generate_pickup_trucks(
            containers=containers,
            day_key=day_of_week.lower(),
            base_date=simulation_date,
            parking_spot_prefix="P"
        )
        return pickup_trucks[0] if pickup_trucks else None
    
    def _process_exports(self,
                        export_operators: Dict[str, Dict],
                        simulation_date: datetime,
                        day_of_week: str) -> List[Truck]:
        """
        Generate delivery trucks for export containers.
        Handles short-dwell operators with early arrival.
        """
        all_trucks = []
        
        # Process operators in parallel
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            
            for operator, config in export_operators.items():
                future = executor.submit(
                    self._process_single_operator,
                    operator, config, simulation_date, day_of_week
                )
                futures.append(future)
            
            for future in as_completed(futures):
                try:
                    trucks = future.result()
                    if trucks:
                        all_trucks.extend(trucks)
                except Exception as e:
                    print(f"Error processing operator: {e}")
        
        return all_trucks
    
    def _process_single_operator(self,
                                operator: str,
                                config: Dict,
                                simulation_date: datetime,
                                day_of_week: str) -> List[Truck]:
        """Process a single export operator."""
        num_containers = config.get('num_containers', 0)
        if num_containers <= 0:
            return []
        
        # Determine arrival time based on operator characteristics
        arrival_time = simulation_date
        
        if operator in self.short_dwell_operators:
            # Short-dwell operators: arrive early
            train_arrival = config.get('arrival_time', {})
            if train_arrival:
                # Decode train arrival time
                angle = train_arrival.get('angle', 0)
                day, hour, minute = self.time_encoder.decode(angle)
                
                # Set arrival time earlier than train
                arrival_time = simulation_date.replace(hour=hour, minute=minute)
                arrival_time -= timedelta(hours=SHORT_DWELL_EARLY_ARRIVAL_HOURS)
        
        # Generate containers in batches for efficiency
        containers = []
        for i in range(0, num_containers, CONTAINER_BATCH_SIZE):
            batch_size = min(CONTAINER_BATCH_SIZE, num_containers - i)
            batch = self.container_factory.create_containers(
                operator=operator,
                direction="Export",
                n_containers=batch_size,
                base_arrival_date=arrival_time,
                current_date=simulation_date
            )
            containers.extend(batch)
        
        # Generate delivery trucks
        trucks = self.truck_factory._generate_delivery_trucks(
            containers=containers,
            day_key=day_of_week.lower(),
            base_date=arrival_time,
            parking_spot_prefix="D"
        )
        
        return trucks
    
    def get_arrived_trucks(self, 
                          trucks: List[Truck],
                          current_time: datetime) -> List[Truck]:
        """
        Filter trucks that have arrived by the current time.
        
        Args:
            trucks: List of all scheduled trucks
            current_time: Current simulation time
            
        Returns:
            List of trucks that have arrived
        """
        # Vectorized filtering for speed
        if not trucks:
            return []
        
        arrival_times = np.array([t.arrival_time for t in trucks], dtype=object)
        mask = arrival_times <= current_time
        
        return [truck for truck, arrived in zip(trucks, mask) if arrived]
    
    def cleanup(self):
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
        direction="Import",
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