from queue import Queue
from typing import List, Dict, Any, Optional
from datetime import datetime


class VehicleQueue:
    """
    Queue manager for vehicles (trains or trucks) arriving at the terminal.
    
    This class manages a queue of vehicles waiting to enter the terminal,
    with scheduling based on arrival times.
    
    Attributes:
        vehicle_type: Type of vehicles in the queue ('Train' or 'Truck')
        vehicles: Queue of vehicles waiting to enter the terminal
        scheduled_arrivals: List of vehicles scheduled to arrive
    """
    
    def __init__(self, vehicle_type: str):
        """
        Initialize the vehicle queue.
        
        Args:
            vehicle_type: Type of vehicles in the queue ('Train' or 'Truck')
        """
        if vehicle_type not in ["Train", "Truck"]:
            raise ValueError("Vehicle type must be 'Train' or 'Truck'")
            
        self.vehicle_type = vehicle_type
        self.vehicles = Queue()
        self.scheduled_arrivals = []
        self.current_time = datetime.now()
    
    def add_vehicle(self, vehicle: Any) -> bool:
        """
        Add a vehicle to the queue.
        
        Args:
            vehicle: Vehicle object to add
            
        Returns:
            Boolean indicating success
        """
        # Check if the vehicle is of the correct type
        if not self._is_correct_vehicle_type(vehicle):
            return False
        
        # Add to the queue
        self.vehicles.put(vehicle)
        return True
    
    def schedule_arrival(self, vehicle: Any, arrival_time: datetime) -> bool:
        """
        Schedule a vehicle to arrive at a specific time.
        
        Args:
            vehicle: Vehicle object to schedule
            arrival_time: Time when the vehicle will arrive
            
        Returns:
            Boolean indicating success
        """
        # Check if the vehicle is of the correct type
        if not self._is_correct_vehicle_type(vehicle):
            return False
        
        # Set the vehicle's arrival time
        vehicle.arrival_time = arrival_time
        
        # Add to scheduled arrivals
        self.scheduled_arrivals.append((arrival_time, vehicle))
        
        # Sort the scheduled arrivals by time
        self.scheduled_arrivals.sort(key=lambda x: x[0])
        
        return True
    
    def get_next_vehicle(self) -> Optional[Any]:
        """
        Get the next vehicle from the queue.
        
        Returns:
            Next vehicle in the queue or None if the queue is empty
        """
        if self.vehicles.empty():
            return None
        
        return self.vehicles.get()
    
    def update(self, current_time: datetime = None) -> List[Any]:
        """
        Update the queue based on the current time.
        
        This processes scheduled arrivals that have reached their arrival time.
        
        Args:
            current_time: Current time (defaults to now)
            
        Returns:
            List of vehicles that have arrived since the last update
        """
        if current_time is None:
            current_time = datetime.now()
        
        self.current_time = current_time
        arrived_vehicles = []
        
        # Check scheduled arrivals
        while self.scheduled_arrivals and self.scheduled_arrivals[0][0] <= current_time:
            _, vehicle = self.scheduled_arrivals.pop(0)
            self.vehicles.put(vehicle)
            arrived_vehicles.append(vehicle)
        
        return arrived_vehicles
    
    def is_empty(self) -> bool:
        """
        Check if the queue is empty.
        
        Returns:
            Boolean indicating if the queue is empty
        """
        return self.vehicles.empty()
    
    def size(self) -> int:
        """
        Get the current size of the queue.
        
        Returns:
            Number of vehicles in the queue
        """
        return self.vehicles.qsize()
    
    def future_arrivals_count(self) -> int:
        """
        Get the number of scheduled future arrivals.
        
        Returns:
            Number of scheduled future arrivals
        """
        return len(self.scheduled_arrivals)
    
    def clear(self) -> None:
        """Clear the queue and scheduled arrivals."""
        # Create a new queue (can't clear an existing one)
        self.vehicles = Queue()
        self.scheduled_arrivals = []
    
    def _is_correct_vehicle_type(self, vehicle: Any) -> bool:
        """
        Check if a vehicle is of the correct type for this queue.
        
        Args:
            vehicle: Vehicle object to check
            
        Returns:
            Boolean indicating if the vehicle is of the correct type
        """
        if self.vehicle_type == "Train":
            return hasattr(vehicle, 'train_id')
        elif self.vehicle_type == "Truck":
            return hasattr(vehicle, 'truck_id')
        return False
    
    def __len__(self) -> int:
        """
        Make the queue support the len() function.
        
        Returns:
            Number of vehicles in the queue
        """
        return self.size()
    
    def __str__(self) -> str:
        """String representation of the queue."""
        return f"{self.vehicle_type} Queue: {self.size()} waiting, {self.future_arrivals_count()} scheduled"


if __name__ == "__main__":
    # Mock classes for testing
    class MockTrain:
        def __init__(self, train_id):
            self.train_id = train_id
            self.arrival_time = None
        
        def __str__(self):
            return f"Train {self.train_id}"
    
    class MockTruck:
        def __init__(self, truck_id):
            self.truck_id = truck_id
            self.arrival_time = None
        
        def __str__(self):
            return f"Truck {self.truck_id}"
    
    # Test the truck queue
    print("Testing Truck Queue:")
    truck_queue = VehicleQueue("Truck")
    
    # Add some trucks
    truck1 = MockTruck("T1")
    truck2 = MockTruck("T2")
    truck3 = MockTruck("T3")
    
    truck_queue.add_vehicle(truck1)
    truck_queue.add_vehicle(truck2)
    
    # Schedule a future arrival
    future_time = datetime(2025, 6, 1, 10, 0)  # 10 AM on 2025-06-01
    truck_queue.schedule_arrival(truck3, future_time)
    
    print(f"Queue size: {truck_queue.size()}")
    print(f"Scheduled arrivals: {truck_queue.future_arrivals_count()}")
    
    # Get a truck from the queue
    next_truck = truck_queue.get_next_vehicle()
    print(f"Next truck: {next_truck}")
    print(f"Queue size after removal: {truck_queue.size()}")
    
    # Test the train queue
    print("\nTesting Train Queue:")
    train_queue = VehicleQueue("Train")
    
    # Add some trains
    train1 = MockTrain("TR1")
    train2 = MockTrain("TR2")
    
    train_queue.add_vehicle(train1)
    
    # Try to add a truck to the train queue (should fail)
    result = train_queue.add_vehicle(truck1)
    print(f"Adding truck to train queue: {result}")
    
    print(f"Train queue size: {train_queue.size()}")
    
    # Test queue update with current time
    print("\nTesting Queue Update:")
    current_time = datetime(2025, 5, 15, 14, 0)  # May 15, 2025, 2 PM
    
    # Schedule an arrival in the past
    past_arrival = datetime(2025, 5, 15, 13, 0)  # 1 PM (1 hour ago)
    train_queue.schedule_arrival(train2, past_arrival)
    
    # Update the queue
    arrived = train_queue.update(current_time)
    print(f"Arrived after update: {len(arrived)} trains")
    print(f"Queue size after update: {train_queue.size()}")
    
    # Clear the queue
    train_queue.clear()
    print(f"Queue size after clear: {train_queue.size()}")