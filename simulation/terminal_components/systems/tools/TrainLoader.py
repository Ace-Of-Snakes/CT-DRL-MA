import numpy as np
from typing import List, Optional, Dict
from dataclasses import dataclass
from datetime import datetime
from simulation.terminal_components.vehicles.Train import Train
from simulation.terminal_components.storage_units.Container import Container
from simulation.terminal_components.storage_units.ContainerFactory import ContainerFactory

# ==================== CONSTANTS ====================
OVERGENERATION_FACTOR = 5.0  # Generate 5x more containers than wagons
TRAIN_DIRECTION = "Import"  # Trains only handle imports (Train -> Truck)
WAGON_UTILIZATION_TARGET = 0.85  # Target 85% wagon utilization
MIN_CONTAINER_LENGTH = 6.0  # Minimum container length in meters (20ft container)


@dataclass
class LoadingStats:
    """Statistics for train loading operation."""
    total_containers_generated: int
    containers_loaded: int
    wagons_used: int
    avg_utilization: float
    time_elapsed_ms: float


class TrainLoader:
    """
    High-performance train loader using vectorized operations.
    Optimized for speed with numpy arrays and efficient algorithms.
    """
    
    def __init__(self, 
                 container_factory: ContainerFactory,
                 overgeneration_factor: float = OVERGENERATION_FACTOR):
        """
        Initialize the train loader.
        
        Args:
            container_factory: Factory for generating containers
            overgeneration_factor: Factor to overgenerate containers (5.0 = 5x containers)
        """
        if not container_factory:
            raise ValueError("Container factory is required")
            
        self.factory = container_factory
        self.overgeneration_factor = overgeneration_factor
    
    def load_train(self, 
                   train: Train,
                   operator: str,
                   current_date: Optional[datetime] = None) -> Train:
        """
        Load a single train with containers using optimized algorithm.
        
        Args:
            train: Train object to load
            operator: Operator name for container generation (required)
            current_date: Current simulation date
            
        Returns:
            The loaded train object (modified in-place)
            
        Raises:
            ValueError: If operator is not provided
        """
        if not operator:
            raise ValueError("Operator must be specified for train loading")
        
        if not train or not train.wagons:
            return train
        
        num_wagons = len(train.wagons)
        num_to_generate = int(num_wagons * self.overgeneration_factor)
        
        # Generate containers in batch (factory already uses vectorized ops)
        containers = self.factory.create_containers(
            operator=operator,
            direction=TRAIN_DIRECTION,  # Always Import for trains
            n_containers=num_to_generate,
            current_date=current_date
        )
        
        if not containers:
            return train
        
        # Fast load using optimized packing algorithm
        self._optimized_load_wagons(train, containers, num_wagons)
        
        return train
    
    def load_multiple_trains(self,
                           trains: List[Train],
                           operator: str,
                           current_date: Optional[datetime] = None) -> List[Train]:
        """
        Load multiple trains efficiently with shared container generation.
        
        Args:
            trains: List of trains to load
            operator: Operator name for container generation (required)
            current_date: Current simulation date
            
        Returns:
            List of loaded trains (modified in-place)
            
        Raises:
            ValueError: If operator is not provided
        """
        if not operator:
            raise ValueError("Operator must be specified for train loading")
        
        if not trains:
            return trains
        
        # Calculate total containers needed
        total_wagons = sum(len(t.wagons) for t in trains)
        total_to_generate = int(total_wagons * self.overgeneration_factor)
        
        # Generate all containers at once for efficiency
        all_containers = self.factory.create_containers(
            operator=operator,
            direction=TRAIN_DIRECTION,  # Always Import for trains
            n_containers=total_to_generate,
            current_date=current_date
        )
        
        if not all_containers:
            return trains
        
        # Distribute containers among trains
        container_idx = 0
        for train in trains:
            num_wagons = len(train.wagons)
            containers_for_train = int(num_wagons * self.overgeneration_factor)
            
            # Slice containers for this train
            train_containers = all_containers[container_idx:container_idx + containers_for_train]
            
            if train_containers:
                self._optimized_load_wagons(train, train_containers, num_wagons)
            
            container_idx += containers_for_train
            
            # Break if we run out of containers
            if container_idx >= len(all_containers):
                break
        
        return trains
    
    def _optimized_load_wagons(self, train: Train, containers: List[Container], num_wagons: int):
        """
        Random wagon loading that maximizes utilization without sorting.
        
        Algorithm:
        1. Randomly shuffle containers for true randomness
        2. Ensure at least one container per wagon for distribution
        3. Use random placement to avoid algorithmic skewing
        """
        # Shuffle containers for true randomness (no sorting by length)
        shuffled_containers = containers.copy()
        np.random.shuffle(shuffled_containers)
        
        # Track which containers have been loaded
        loaded_containers = set()
        
        # Phase 1: Ensure each wagon gets at least one container
        # Pick random containers (not sorted) for initial distribution
        initial_containers = shuffled_containers[:num_wagons]
        
        for wagon_idx, container in enumerate(initial_containers):
            if wagon_idx >= num_wagons:
                break
            if train.add_container(container, wagon_index=wagon_idx):
                loaded_containers.add(container.container_id)
        
        # Phase 2: Load remaining containers randomly
        # Try each container in random order
        for container in shuffled_containers:
            if container.container_id in loaded_containers:
                continue
            
            # Try wagons in sequential order first, then randomize if needed
            # This helps with cache locality while still being random due to shuffled containers
            for wagon_idx in range(num_wagons):
                wagon = train.wagons[wagon_idx]
                
                # Check if container fits with a small tolerance for floating point errors
                # Using direct comparison with wagon's available length
                available = wagon.get_available_length()
                
                # Add tiny tolerance (1mm) for floating point comparison issues
                # This handles cases where 6.096 might not fit in 6.1 due to float precision
                if available >= container.length_m - 0.001:
                    if train.add_container(container, wagon_index=wagon_idx):
                        loaded_containers.add(container.container_id)
                        break
            else:
                # If sequential didn't work, try random order
                wagon_indices = list(range(num_wagons))
                np.random.shuffle(wagon_indices)
                
                for wagon_idx in wagon_indices:
                    wagon = train.wagons[wagon_idx]
                    available = wagon.get_available_length()
                    
                    if available >= container.length_m - 0.001:
                        if train.add_container(container, wagon_index=wagon_idx):
                            loaded_containers.add(container.container_id)
                            break
        
        # Phase 3: Final pass - try to fit any remaining containers
        # using train's auto-placement logic
        for container in shuffled_containers:
            if container.container_id not in loaded_containers:
                # Let the train's built-in logic find any remaining space
                if train.add_container(container):
                    loaded_containers.add(container.container_id)
    
    def _debug_wagon_state(self, train: Train) -> Dict:
        """
        Debug function to analyze wagon loading state.
        
        Returns:
            Dictionary with detailed wagon statistics
        """
        wagon_stats = []
        total_capacity = 0
        total_used = 0
        
        for i, wagon in enumerate(train.wagons):
            capacity = wagon.length
            used = wagon._used_length
            available = wagon.get_available_length()
            num_containers = len(wagon.containers)
            
            # Get container details for debugging
            container_lengths = [c.length_m for c in wagon.get_container_list()]
            container_types = [c.container_type for c in wagon.get_container_list()]
            
            total_capacity += capacity
            total_used += used
            
            wagon_stats.append({
                'wagon_id': i,
                'capacity': capacity,
                'used': used,
                'available': available,
                'utilization': (used / capacity * 100) if capacity > 0 else 0,
                'num_containers': num_containers,
                'container_lengths': container_lengths,
                'container_types': container_types
            })
        
        return {
            'wagon_details': wagon_stats,
            'total_capacity': total_capacity,
            'total_used': total_used,
            'overall_utilization': (total_used / total_capacity * 100) if total_capacity > 0 else 0,
            'wagons_empty': sum(1 for w in train.wagons if w.is_empty()),
            'wagons_full': sum(1 for w in train.wagons if w.is_full())
        }
    
    def analyze_container_distribution(self, containers: List[Container]) -> Dict:
        """
        Analyze the distribution of container types and lengths.
        
        Args:
            containers: List of containers to analyze
            
        Returns:
            Dictionary with distribution statistics
        """
        if not containers:
            return {}
        
        # Group by container type
        type_counts = {}
        length_counts = {}
        
        for container in containers:
            # Count types
            ctype = container.container_type
            if ctype not in type_counts:
                type_counts[ctype] = 0
            type_counts[ctype] += 1
            
            # Count lengths
            length = container.length_m
            length_key = f"{length:.3f}m"
            if length_key not in length_counts:
                length_counts[length_key] = 0
            length_counts[length_key] += 1
        
        # Calculate statistics
        lengths = [c.length_m for c in containers]
        
        return {
            'total_containers': len(containers),
            'type_distribution': type_counts,
            'length_distribution': length_counts,
            'min_length': min(lengths),
            'max_length': max(lengths),
            'avg_length': sum(lengths) / len(lengths),
            'unique_types': len(type_counts),
            'unique_lengths': len(length_counts)
        }
    
    def get_loading_stats(self, train: Train) -> LoadingStats:
        """
        Calculate loading statistics for a train.
        
        Args:
            train: Loaded train to analyze
            
        Returns:
            LoadingStats object with utilization metrics
        """
        stats = train.get_stats()
        
        wagons_used = sum(1 for wagon in train.wagons if not wagon.is_empty())
        
        return LoadingStats(
            total_containers_generated=0,  # Set by caller if needed
            containers_loaded=stats['total_containers'],
            wagons_used=wagons_used,
            avg_utilization=stats['utilization_rate'],
            time_elapsed_ms=0  # Set by caller if timing
        )
    

if __name__ == "__main__":
    import time
    
    factory = ContainerFactory()
    loader = TrainLoader(factory, overgeneration_factor=3.0)

    # First, analyze what containers are being generated
    print("=== Container Generation Analysis ===")
    test_containers = factory.create_containers(
        operator="BOX",
        direction=TRAIN_DIRECTION,
        n_containers=100,
        current_date=datetime.now()
    )
    
    container_dist = loader.analyze_container_distribution(test_containers)
    print(f"Container types: {container_dist['type_distribution']}")
    print(f"Container lengths: {container_dist['length_distribution']}")
    print(f"Min length: {container_dist['min_length']:.3f}m")
    print(f"Max length: {container_dist['max_length']:.3f}m")
    print(f"Avg length: {container_dist['avg_length']:.3f}m")
    
    print("\n=== Train Loading Test ===")
    # Load single train with timing
    train = Train(train_id="TRN001", num_wagons=29)
    
    start_time = time.perf_counter()
    loaded_train = loader.load_train(
        train, 
        operator="BOX",
        current_date=datetime.now()
    )
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    # Get loading statistics
    stats = loader.get_loading_stats(loaded_train)
    stats.time_elapsed_ms = elapsed_ms
    stats.total_containers_generated = int(29 * 3.0)  # Factor of 3
    
    print(f"Generated: {stats.total_containers_generated} containers")
    print(f"Loaded: {stats.containers_loaded} containers")
    print(f"Wagons used: {stats.wagons_used}/{29}")
    print(f"Utilization: {stats.avg_utilization:.1%}")
    print(f"Time: {stats.time_elapsed_ms:.2f}ms")
    
    # Debug output
    debug_info = loader._debug_wagon_state(loaded_train)
    print(f"\nDetailed Analysis:")
    print(f"Empty wagons: {debug_info['wagons_empty']}")
    print(f"Full wagons: {debug_info['wagons_full']}")
    print(f"Total capacity: {debug_info['total_capacity']:.1f}m")
    print(f"Total used: {debug_info['total_used']:.1f}m")
    
    # Show per-wagon stats for ALL wagons to see the pattern
    print(f"\nAll wagon details:")
    for wagon_stat in debug_info['wagon_details']:
        print(f"  Wagon {wagon_stat['wagon_id']:2d}: "
              f"{wagon_stat['num_containers']} containers, "
              f"{wagon_stat['utilization']:5.1f}% full, "
              f"{wagon_stat['available']:6.3f}m available")
        if wagon_stat['container_lengths']:
            print(f"    Lengths: {[f'{l:.3f}m' for l in wagon_stat['container_lengths']]}")
            print(f"    Types: {wagon_stat['container_types']}")
            print(f"    Sum: {sum(wagon_stat['container_lengths']):.3f}m used of {wagon_stat['capacity']:.3f}m capacity")