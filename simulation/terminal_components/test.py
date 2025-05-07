from Wagon import Wagon
from Train import Train
from Truck import Truck
from Container import Container


#Example usage
if __name__ == "__main__":
    # Create a mock Container class for testing
    class MockContainer:
        def __init__(self, container_id, container_type, length, weight):
            self.container_id = container_id
            self.container_type = container_type
            self.length = length
            self.weight = weight
    
    # Create some test containers
    tweu1 = MockContainer("TWEU100", "TWEU", 6.06, 15000)
    tweu2 = MockContainer("TWEU200", "TWEU", 6.06, 18000)
    feu1 = MockContainer("FEU100", "FEU", 12.19, 25000)
    trailer1 = MockContainer("TRL100", "Trailer", 18.3, 14000)
    
    # Test Wagon class
    print("\n===== Testing Wagon =====")
    wagon = Wagon("W1", 24.34, 60000)
    print(f"New wagon: {wagon}")
    print(f"Adding TWEU container: {wagon.add_container(tweu1)}")
    print(f"Adding another TWEU container: {wagon.add_container(tweu2)}")
    print(f"Wagon after adding containers: {wagon}")
    print(f"Available length: {wagon.get_available_length():.2f}m")
    print(f"Adding FEU container (should succeed): {wagon.add_container(feu1)}")
    print(f"Wagon is now full: {wagon.is_full()}")
    
    # Test Train class
    print("\n===== Testing Train =====")
    train = Train("TR1001", num_wagons=3, wagon_length=20.0)
    print(f"New train: {train}")
    print(f"Adding TWEU container to train: {train.add_container(tweu1)}")
    print(f"Adding another TWEU container to train: {train.add_container(tweu2)}")
    print(f"Adding FEU container to train: {train.add_container(feu1)}")
    
    # Test Truck class
    print("\n===== Testing Truck =====")
    # Empty truck arriving to pick up
    truck1 = Truck("TK1001")
    print(f"Empty truck: {truck1}")
    truck1.add_pickup_container_id("TWEU100")
    print(f"Truck assigned for pickup: {truck1.pickup_container_ids}")
    
    # Truck arriving with container to deliver
    truck2 = Truck("TK2001", containers=feu1)
    print(f"Delivery truck: {truck2}")
    
    # Test truck loading/unloading
    truck2.start_loading()
    container = truck2.remove_container()
    truck2.complete_loading()
    print(f"Truck after unloading: {truck2}")
    print(f"Loading time: {truck2.get_loading_time()}")
    
    # Test truck with multiple containers
    truck3 = Truck("TK3001", max_length=13.0)
    print(f"\nNew truck for multiple containers: {truck3}")
    print(f"Adding first TWEU container: {truck3.add_container(tweu1)}")
    print(f"Adding second TWEU container: {truck3.add_container(tweu2)}")
    print(f"Available length after 2 TWEUs: {truck3.get_available_length():.2f}m")
    print(f"Is truck full? {truck3.is_full()}")
    print(f"Adding FEU container (should fail): {truck3.add_container(feu1)}")
    print(f"Truck with multiple containers: {truck3}")
    
    # Test complex train operations
    print("\n===== Testing Complex Train Operations =====")
    # Create a new train with multiple wagons
    train2 = Train("TR2001", num_wagons=5, wagon_length=20.0)
    
    # Add pickup requirements
    train2.add_pickup_container("CONT001", 0)
    train2.add_pickup_container("CONT002", 0)
    train2.add_pickup_container("CONT003", 1)
    
    print(f"Train with pickup requirements: {train2}")
    print(f"Pickup container IDs: {train2.get_all_pickup_container_ids()}")
    
    # Add a trailer to an empty wagon (should work)
    print(f"Adding trailer to train: {train2.add_container(trailer1)}")
    
    # Try to add a container to the same wagon as trailer (should fail)
    wagon_with_trailer, _ = train2.find_container(trailer1.container_id)
    wagon_index = train2.wagons.index(wagon_with_trailer)
    print(f"Adding container to wagon with trailer (should fail): {train2.add_container(tweu1, wagon_index)}")
    
    # Start and complete loading process
    train2.start_loading()
    # Simulate loading operations...
    train2.wagons[0].pickup_container_ids.clear()  # Simulate picking up all containers
    train2.wagons[1].pickup_container_ids.clear()
    train2.complete_loading()
    
    print(f"Train after loading: {train2}")
    print(f"Loading time: {train2.get_loading_time()}")
    print(f"Is train fully loaded: {train2.is_fully_loaded()}")