import numpy as np
import pickle
import time
from typing import Tuple, Dict, Optional, List, Union
import os


class RMGMovementCalculator:
    """
    Class for calculating the time needed to move a container from one place to another
    using a Rail Mounted Gantry (RMG) crane.
    """
    
    def __init__(
        self,
        distance_matrix_path: str,
        trolley_speed: float = 70.0,  # m/min (from Liebherr specs)
        hoisting_speed: float = 28.0,  # m/min with load (from Liebherr specs)
        gantry_speed: float = 130.0,     # m/min (from Liebherr specs)
        trolley_acceleration: float = 0.3,  # m/s²
        hoisting_acceleration: float = 0.2,  # m/s²
        gantry_acceleration: float = 0.1,   # m/s² (slower for the whole crane)
        max_height: float = 20.0,  # meters
        ground_vehicle_height: float = 1.5,  # meters
        container_heights: Dict[str, float] = None,  # meters
    ):
        """
        Initialize the RMG Movement Calculator.
        
        Args:
            distance_matrix_path: Path to the pickle file containing the distance matrix
            trolley_speed: Maximum trolley speed in meters per minute
            hoisting_speed: Maximum hoisting speed in meters per minute
            trolley_acceleration: Trolley acceleration in meters per second squared
            hoisting_acceleration: Hoisting acceleration in meters per second squared
            max_height: Maximum height of the RMG crane in meters
            ground_vehicle_height: Height of trucks/trains from ground in meters
            container_heights: Dictionary mapping container types to their heights
        """
        # Convert speeds from m/min to m/s for easier calculations
        self.trolley_speed = trolley_speed / 60.0  # m/s
        self.hoisting_speed = hoisting_speed / 60.0  # m/s
        self.gantry_speed = gantry_speed / 60.0  # Convert to m/s

        self.trolley_acceleration = trolley_acceleration
        self.hoisting_acceleration = hoisting_acceleration
        self.gantry_acceleration = gantry_acceleration

        self.max_height = max_height
        self.ground_vehicle_height = ground_vehicle_height
        
        # Default container heights if not provided
        if container_heights is None:
            # Standard container heights (TEU, FEU, etc.)
            self.container_heights = {
                "TEU": 2.59,  # 20-foot equivalent unit (standard)
                "FEU": 2.59,  # 40-foot equivalent unit (standard)
                "HQ": 2.90,   # High cube container
                "default": 2.59  # Default height
            }
        else:
            self.container_heights = container_heights
            
        # Load the distance matrix and related data
        self.load_distance_matrix(distance_matrix_path)
        
        # Precompute object types for faster lookup
        self.object_types = {obj_name: obj_info['type'] 
                             for obj_name, obj_info in self.objects.items()}
        
        # Create cache for movement times to avoid recalculating the same paths
        self.movement_time_cache = {}
        
    def load_distance_matrix(self, file_path: str):
        """Load the distance matrix and related data from the pickle file."""
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                
            self.distance_matrix = data['distance_matrix']
            self.object_to_idx = data['object_to_idx']
            self.objects = data['objects']
            self.positions = data['positions']
            
            print(f"Successfully loaded distance matrix with {len(self.objects)} objects")
            
        except Exception as e:
            raise ValueError(f"Error loading distance matrix from {file_path}: {str(e)}")
    
    def get_object_type(self, obj_name: str) -> str:
        """Get the type of an object based on its name."""
        if obj_name in self.object_types:
            return self.object_types[obj_name]
        raise ValueError(f"Object {obj_name} not found in the terminal")
    
    def get_horizontal_distance(self, src: str, dst: str) -> float:
        """Get the horizontal distance between two objects in the terminal."""
        if src not in self.object_to_idx or dst not in self.object_to_idx:
            raise ValueError(f"Object not found: {src if src not in self.object_to_idx else dst}")
            
        idx1 = self.object_to_idx[src]
        idx2 = self.object_to_idx[dst]
        return self.distance_matrix[idx1, idx2]
    
    def calculate_movement_time(
        self, 
        src: str, 
        dst: str, 
        container_type: str = "default",
        stack_height: float = 0.0
    ) -> float:
        """
        Calculate the time needed to move a container from source to destination,
        including potential gantry movement along the rails.
        
        Args:
            src: Source object name (e.g., 't1_1', 'p_5', 'A10')
            dst: Destination object name
            container_type: Type of container for height calculation
            stack_height: Height of container stack at destination (if moving to storage)
                          or source (if moving from storage)
        
        Returns:
            Time in seconds needed for the movement
        """
        # Check the cache first
        cache_key = (src, dst, container_type, stack_height)
        if cache_key in self.movement_time_cache:
            return self.movement_time_cache[cache_key]
        
        # Get source and destination positions
        if src not in self.positions or dst not in self.positions:
            raise ValueError(f"Object not found: {src if src not in self.positions else dst}")
            
        src_pos = self.positions[src]
        dst_pos = self.positions[dst]
        
        # Determine movement components
        
        # 1. Gantry movement (movement of entire crane along the rails)
        # In our coordinate system, y-axis represents the position along the rails
        gantry_distance = abs(src_pos[1] - dst_pos[1])
        
        # 2. Trolley movement (movement of trolley across the crane bridge)
        # In our coordinate system, x-axis represents the position across the bridge
        trolley_distance = abs(src_pos[0] - dst_pos[0])
        
        # Determine if significant gantry movement is needed
        # (small differences might just be points at different positions but reachable by trolley)
        significant_gantry_movement = gantry_distance > 1.0  # threshold in meters
        
        # Calculate gantry movement time if needed
        gantry_time = 0.0
        if significant_gantry_movement:
            # Use gantry speed (would be defined in __init__, typically 130-240 m/min)
            # For now, we'll use a default value based on Liebherr specs
            
            gantry_time = self.calculate_travel_time(
                gantry_distance,
                self.gantry_speed,
                self.gantry_acceleration
            )
        
        # Calculate trolley movement time
        trolley_time = self.calculate_travel_time(
            trolley_distance,
            self.trolley_speed,
            self.trolley_acceleration
        )
        
        # Determine vertical distances based on source and destination types
        src_type = self.get_object_type(src)
        dst_type = self.get_object_type(dst)
        
        # Get container height
        container_height = self.container_heights.get(container_type, 
                                                     self.container_heights["default"])
        
        # Calculate vertical distances
        vertical_distance_up = 0.0
        vertical_distance_down = 0.0
        
        # Determine vertical movement based on source and destination types
        if src_type == 'rail_slot' and dst_type == 'storage_slot':
            # Train to storage
            vertical_distance_up = self.max_height - (self.ground_vehicle_height + container_height)
            vertical_distance_down = self.max_height - stack_height
            
        elif src_type == 'storage_slot' and dst_type == 'rail_slot':
            # Storage to train
            vertical_distance_up = self.max_height - stack_height
            vertical_distance_down = self.max_height - (self.ground_vehicle_height + container_height)
            
        elif src_type == 'parking_spot' and dst_type == 'storage_slot':
            # Truck to storage
            vertical_distance_up = self.max_height - (self.ground_vehicle_height + container_height)
            vertical_distance_down = self.max_height - stack_height
            
        elif src_type == 'storage_slot' and dst_type == 'parking_spot':
            # Storage to truck
            vertical_distance_up = self.max_height - stack_height
            vertical_distance_down = self.max_height - (self.ground_vehicle_height + container_height)
            
        elif src_type == 'rail_slot' and dst_type == 'parking_spot':
            # Train to truck (direct move)
            vertical_distance_up = self.max_height - (self.ground_vehicle_height + container_height)
            vertical_distance_down = self.max_height - (self.ground_vehicle_height + container_height)
            
        elif src_type == 'parking_spot' and dst_type == 'rail_slot':
            # Truck to train (direct move)
            vertical_distance_up = self.max_height - (self.ground_vehicle_height + container_height)
            vertical_distance_down = self.max_height - (self.ground_vehicle_height + container_height)
            
        elif src_type == 'storage_slot' and dst_type == 'storage_slot':
            # Storage to storage (reshuffling)
            src_stack_height = stack_height  # In this case, stack_height is for source
            dst_stack_height = 0.0  # Default, unless specified otherwise
            vertical_distance_up = self.max_height - src_stack_height
            vertical_distance_down = self.max_height - dst_stack_height
        
        # Calculate vertical movement times
        vertical_up_time = self.calculate_travel_time(
            vertical_distance_up, 
            self.hoisting_speed, 
            self.hoisting_acceleration
        )
        
        vertical_down_time = self.calculate_travel_time(
            vertical_distance_down, 
            self.hoisting_speed, 
            self.hoisting_acceleration
        )
        
        # Calculate total time with proper sequencing of operations:
        # 1. Gantry movement (if needed)
        # 2. Simultaneous lifting and trolley movement
        # 3. Lowering
        
        # In most RMG operations, the sequence is:
        # - Move gantry to position (if needed)
        # - Lift container
        # - Move trolley to destination
        # - Lower container
        
        # The trolley can move while lifting, but typically after gantry movement
        horizontal_movement_time = gantry_time + trolley_time
        
        # If trolley can move during lifting, take the max
        # Otherwise, add them sequentially
        concurrent_operations = True  # Set to False if operations must be sequential
        
        if concurrent_operations:
            # Lifting can happen while trolley moves (after gantry movement)
            if significant_gantry_movement:
                # Gantry moves first, then trolley and lifting happen together
                time_after_gantry = max(vertical_up_time, trolley_time)
                total_time = gantry_time + time_after_gantry + vertical_down_time
            else:
                # No gantry movement, lifting and trolley can happen simultaneously
                total_time = max(vertical_up_time, trolley_time) + vertical_down_time
        else:
            # Sequential operations: gantry → lift → trolley → lower
            total_time = gantry_time + vertical_up_time + trolley_time + vertical_down_time
        
        # Add some fixed time for attaching/detaching the container
        attach_detach_time = 10.0  # seconds
        total_time += attach_detach_time
        
        # Cache the result
        self.movement_time_cache[cache_key] = total_time
        
        return total_time
    
    def calculate_travel_time(self, distance: float, max_speed: float, acceleration: float) -> float:
        """
        Calculate travel time with acceleration and deceleration.
        
        Args:
            distance: Distance to travel in meters
            max_speed: Maximum speed in meters per second
            acceleration: Acceleration/deceleration in meters per second squared
        
        Returns:
            Time in seconds needed for the travel
        """
        # No movement needed
        if distance <= 0:
            return 0.0
        
        # Calculate the distance needed to reach max speed
        accel_distance = 0.5 * max_speed**2 / acceleration
        
        # If we can't reach max speed (distance is too short)
        if distance <= 2 * accel_distance:
            # Time to accelerate and then immediately decelerate
            peak_speed = np.sqrt(acceleration * distance)
            time = 2 * peak_speed / acceleration
        else:
            # Time to accelerate + time at max speed + time to decelerate
            accel_time = max_speed / acceleration
            constant_speed_distance = distance - 2 * accel_distance
            constant_speed_time = constant_speed_distance / max_speed
            time = 2 * accel_time + constant_speed_time
            
        return time
    
    def batch_calculate_times(
        self, 
        movements: List[Tuple[str, str, str, float]]
    ) -> List[float]:
        """
        Calculate movement times for a batch of movements.
        
        Args:
            movements: List of tuples (src, dst, container_type, stack_height)
        
        Returns:
            List of movement times in seconds
        """
        return [self.calculate_movement_time(src, dst, container_type, stack_height) 
                for src, dst, container_type, stack_height in movements]
    
    def create_time_matrix(
        self, 
        objects: List[str] = None, 
        container_type: str = "default",
        stack_height: float = 0.0
    ) -> np.ndarray:
        """
        Create a matrix of movement times between all specified objects.
        
        Args:
            objects: List of object names to include (all if None)
            container_type: Type of container for height calculation
            stack_height: Default stack height for storage slots
        
        Returns:
            Matrix of movement times
        """
        if objects is None:
            objects = list(self.object_to_idx.keys())
        
        n_objects = len(objects)
        time_matrix = np.zeros((n_objects, n_objects))
        
        # Create mapping from objects to indices in the time matrix
        obj_to_matrix_idx = {obj: i for i, obj in enumerate(objects)}
        
        # Vectorized calculation is challenging for this due to the different rules
        # for different object types, so we'll use nested loops
        for i, src in enumerate(objects):
            for j, dst in enumerate(objects):
                if i != j:  # Skip diagonal (self-movement)
                    time_matrix[i, j] = self.calculate_movement_time(
                        src, dst, container_type, stack_height
                    )
        
        return time_matrix
    
    def export_time_matrix(
        self, 
        file_path: str,
        objects: List[str] = None, 
        container_type: str = "default",
        stack_height: float = 0.0
    ):
        """
        Calculate and export the time matrix to a pickle file.
        
        Args:
            file_path: Path to save the pickle file
            objects: List of object names to include (all if None)
            container_type: Type of container for height calculation
            stack_height: Default stack height for storage slots
        """
        if objects is None:
            objects = list(self.object_to_idx.keys())
        
        time_matrix = self.create_time_matrix(objects, container_type, stack_height)
        
        data = {
            'time_matrix': time_matrix,
            'objects': objects,
            'object_to_idx': {obj: i for i, obj in enumerate(objects)},
            'container_type': container_type,
            'stack_height': stack_height,
            'trolley_speed': self.trolley_speed * 60.0,  # Convert back to m/min
            'hoisting_speed': self.hoisting_speed * 60.0,  # Convert back to m/min
            'trolley_acceleration': self.trolley_acceleration,
            'hoisting_acceleration': self.hoisting_acceleration,
            'max_height': self.max_height,
            'ground_vehicle_height': self.ground_vehicle_height,
            'container_heights': self.container_heights
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Time matrix exported to {file_path}")


def main():
    """Example usage of the RMGMovementCalculator class."""
    # Path to distance matrix
    distance_matrix_path = 'distance_matrix.pkl'
    
    # Create calculator with realistic RMG parameters based on Liebherr specs
    calculator = RMGMovementCalculator(
        distance_matrix_path=distance_matrix_path,
        trolley_speed=70.0,         # m/min - from Liebherr specs
        hoisting_speed=28.0,        # m/min with load - from Liebherr specs
        trolley_acceleration=0.3,   # m/s² (estimated)
        hoisting_acceleration=0.2,  # m/s² (estimated)
        max_height=20.0,            # meters (typical for 5-6 container stacks)
        ground_vehicle_height=1.5   # meters (typical truck/train height from ground)
    )
    
    # Get valid objects of each type
    rail_slots = [k for k, v in calculator.objects.items() if v['type'] == 'rail_slot']
    storage_slots = [k for k, v in calculator.objects.items() if v['type'] == 'storage_slot']
    parking_spots = [k for k, v in calculator.objects.items() if v['type'] == 'parking_spot']
    
    # Example movement types
    print("\nExample movement times:")
    
    # Train to storage
    src, dst = rail_slots[0], storage_slots[0]
    movement_time = calculator.calculate_movement_time(src, dst, "TEU", 0.0)
    print(f"1. Train to empty storage ({src} to {dst}): {movement_time:.2f} seconds")
    
    # Train to storage with stack
    src, dst = rail_slots[10], storage_slots[10]
    movement_time = calculator.calculate_movement_time(src, dst, "TEU", 2.59*2)  # Two containers high
    print(f"2. Train to stacked storage ({src} to {dst}, stack height 5.18m): {movement_time:.2f} seconds")
    
    # Storage to train
    src, dst = storage_slots[20], rail_slots[20]
    movement_time = calculator.calculate_movement_time(src, dst, "TEU", 2.59)  # One container high
    print(f"3. Storage to train ({src} to {dst}, stack height 2.59m): {movement_time:.2f} seconds")
    
    # Truck to storage
    src, dst = parking_spots[5], storage_slots[50]
    movement_time = calculator.calculate_movement_time(src, dst, "TEU", 0.0)
    print(f"4. Truck to empty storage ({src} to {dst}): {movement_time:.2f} seconds")
    
    # Storage to truck
    src, dst = storage_slots[40], parking_spots[15]
    movement_time = calculator.calculate_movement_time(src, dst, "TEU", 2.59*3)  # Three containers high
    print(f"5. Storage to truck ({src} to {dst}, stack height 7.77m): {movement_time:.2f} seconds")
    
    # Train to truck (direct move)
    src, dst = rail_slots[5], parking_spots[5]
    movement_time = calculator.calculate_movement_time(src, dst, "TEU")
    print(f"6. Train to truck ({src} to {dst}): {movement_time:.2f} seconds")
    
    # Truck to train (direct move)
    src, dst = parking_spots[10], rail_slots[10]
    movement_time = calculator.calculate_movement_time(src, dst, "FEU")
    print(f"7. Truck to train ({src} to {dst}): {movement_time:.2f} seconds")
    
    # Storage to storage (reshuffling)
    src, dst = storage_slots[30], storage_slots[60]
    movement_time = calculator.calculate_movement_time(src, dst, "HQ", 2.90*2)  # Two high cube containers
    print(f"8. Storage to storage ({src} to {dst}, stack height 5.80m): {movement_time:.2f} seconds")
    
    # Performance test
    print("\nPerformance test:")
    
    # Batch calculation
    batch_size = 1000
    movements = []
    for _ in range(batch_size):
        src = np.random.choice(rail_slots + storage_slots + parking_spots)
        dst = np.random.choice(rail_slots + storage_slots + parking_spots)
        container_type = np.random.choice(["TEU", "FEU", "HQ"])
        stack_height = np.random.uniform(0, 10)
        movements.append((src, dst, container_type, stack_height))
    
    start_time = time.time()
    movement_times = calculator.batch_calculate_times(movements)
    end_time = time.time()
    
    print(f"Calculated {batch_size} movement times in {end_time - start_time:.4f} seconds")
    print(f"Average time per calculation: {(end_time - start_time) / batch_size * 1000:.4f} ms")
    
    # Create and export time matrix for a subset of objects
    subset = rail_slots[:10] + storage_slots[:10] + parking_spots[:10]
    start_time = time.time()
    # calculator.export_time_matrix('time_matrix.pkl', subset)
    end_time = time.time()
    
    print(f"Created and exported time matrix for {len(subset)} objects in {end_time - start_time:.4f} seconds")


if __name__ == "__main__":
    main()