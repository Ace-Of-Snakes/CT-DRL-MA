import numpy as np
from typing import List, Dict, Tuple, Optional, Set, Any
import re


class StorageYard:
    """
    Represents the container storage yard in the terminal.
    
    The storage yard is a 3D grid where:
    - Rows correspond to different areas in the yard (e.g., A, B, C...)
    - Bays are positions along each row
    - Tiers represent the stacking height at each position
    
    Attributes:
        num_rows: Number of rows in the storage yard
        num_bays: Number of bays in each row
        max_tier_height: Maximum allowed stacking height
        row_names: Names of the rows (e.g., A, B, C...)
        special_rows: Dictionary mapping special row types to row names
        yard: 3D dictionary mapping positions to containers
    """
    
    def __init__(self, 
                 num_rows: int, 
                 num_bays: int, 
                 max_tier_height: int = 5,
                 row_names: List[str] = None,
                 special_areas: Dict[str, List[Tuple[str, int, int]]] = None):
        """
        Initialize the storage yard.
        
        Args:
            num_rows: Number of rows in the yard
            num_bays: Number of bays per row
            max_tier_height: Maximum stacking height
            row_names: Names for each row (defaults to A, B, C...)
            special_areas: Dictionary mapping special types to areas
                           e.g., {'reefer': [('A', 1, 5), ('B', 6, 10)],
                                  'dangerous': [('F', 1, 10)]}
                           where each tuple is (row, bay_start, bay_end) inclusive
        """
        self.num_rows = num_rows
        self.num_bays = num_bays
        self.max_tier_height = max_tier_height
        
        # Initialize row names if not provided
        if row_names is None:
            self.row_names = [chr(65 + i) for i in range(num_rows)]  # A, B, C, ...
        else:
            self.row_names = row_names[:num_rows]  # Use provided names up to num_rows
        
        # Special areas for different container types
        if special_areas is None:
            # Default special areas: first 5 bays of row A for reefer, last 5 bays of row F for dangerous
            self.special_areas = {
                'reefer': [('A', 1, 5)],  # Row A, bays 1-5
                'dangerous': [('F', 6, 10)]  # Row F, bays 6-10
            }
        else:
            self.special_areas = special_areas
            
        # Validate special areas
        self._validate_special_areas()
        
        # Initialize empty yard - nested dictionary for efficient sparse representation
        # yard[row][bay][tier] = container
        self.yard = {row: {bay: {} for bay in range(1, num_bays + 1)} for row in self.row_names}
        
        # Keep track of bay occupancy for each row
        self.bay_occupancy = {row: np.zeros(num_bays + 1, dtype=int) for row in self.row_names}
        
        # Mapping from position string to (row, bay) tuple
        self.position_to_coordinates = {}
        self._build_position_mapping()
    
    def _validate_special_areas(self):
        """Validate that special areas are within the yard boundaries."""
        for container_type, areas in self.special_areas.items():
            for area in areas:
                row, bay_start, bay_end = area
                if row not in self.row_names:
                    raise ValueError(f"Invalid row '{row}' in special area for {container_type}")
                if bay_start < 1 or bay_end > self.num_bays or bay_start > bay_end:
                    raise ValueError(f"Invalid bay range {bay_start}-{bay_end} in special area for {container_type}")
    
    def _build_position_mapping(self):
        """Build mapping from position strings (e.g., 'A1') to (row, bay) coordinates."""
        for row in self.row_names:
            for bay in range(1, self.num_bays + 1):
                position = f"{row}{bay}"
                self.position_to_coordinates[position] = (row, bay)
    
    def is_position_in_special_area(self, position: str, container_type: str) -> bool:
        """
        Check if a position is in a special area for a specific container type.
        
        Args:
            position: Position string (e.g., 'A1')
            container_type: Container type to check (e.g., 'Reefer', 'Dangerous')
            
        Returns:
            Boolean indicating if the position is in a special area for the container type
        """
        # Convert position to row, bay
        if position not in self.position_to_coordinates:
            return False
            
        row, bay = self.position_to_coordinates[position]
        
        # Convert container type to lowercase for lookup
        container_type_lower = container_type.lower()
        
        # Check if container type has special areas
        if container_type_lower not in self.special_areas:
            return False
            
        # Check if position is in any of the special areas for this container type
        for area_row, bay_start, bay_end in self.special_areas[container_type_lower]:
            if row == area_row and bay_start <= bay <= bay_end:
                return True
                
        return False

    def _build_position_mapping(self):
        """Build mapping from position strings (e.g., 'A1') to (row, bay) coordinates."""
        for row in self.row_names:
            for bay in range(self.num_bays):
                position = f"{row}{bay+1}"
                self.position_to_coordinates[position] = (row, bay)
    
    def clear(self):
        """Clear the entire storage yard."""
        for row in self.row_names:
            for bay in range(self.num_bays):
                self.yard[row][bay] = {}
        
        # Reset bay occupancy
        self.bay_occupancy = {row: np.zeros(self.num_bays, dtype=int) for row in self.row_names}
    
    def get_container(self, position: str, tier: int = None) -> Optional[Any]:
        """
        Get the container at the specified position and tier.
        
        Args:
            position: Position string (e.g., 'A1')
            tier: Tier number (if None, returns the top container)
            
        Returns:
            Container object or None if no container is found
        """
        # Parse the position string to get row and bay
        if position not in self.position_to_coordinates:
            return None
        
        row, bay = self.position_to_coordinates[position]
        
        # If tier not specified, get the top container
        if tier is None:
            max_tier = max(self.yard[row][bay].keys()) if self.yard[row][bay] else 0
            if max_tier == 0:  # No containers in this position
                return None
            tier = max_tier
        
        # Return the container at the specified tier
        return self.yard[row][bay].get(tier)
    
    def get_top_container(self, position: str) -> Tuple[Optional[Any], Optional[int]]:
        """
        Get the top container at the specified position.
        
        Args:
            position: Position string (e.g., 'A1')
            
        Returns:
            Tuple of (container, tier) or (None, None) if no container is found
        """
        # Parse the position string to get row and bay
        if position not in self.position_to_coordinates:
            return None, None
        
        row, bay = self.position_to_coordinates[position]
        
        # Get the highest tier
        tiers = list(self.yard[row][bay].keys())
        if not tiers:  # No containers in this position
            return None, None
        
        max_tier = max(tiers)
        return self.yard[row][bay][max_tier], max_tier
    
    def add_container(self, position: str, container: Any, tier: int = None) -> bool:
        """
        Add a container to the specified position.
        
        Args:
            position: Position string (e.g., 'A1')
            container: Container object to add
            tier: Specific tier to add the container to (if None, adds to the top)
            
        Returns:
            Boolean indicating success
        """
        # Check if container can be accepted at this position
        if not self.can_accept_container(position, container):
            return False
            
        # Parse the position string to get row and bay
        row, bay = self.position_to_coordinates[position]
        
        # If tier not specified, add to the top
        if tier is None:
            current_tiers = list(self.yard[row][bay].keys())
            tier = 1 if not current_tiers else max(current_tiers) + 1
        
        # Check if the tier is already occupied
        if tier in self.yard[row][bay]:
            return False
        
        # Add the container
        self.yard[row][bay][tier] = container
        self.bay_occupancy[row][bay] = max(self.bay_occupancy[row][bay], tier)
        
        return True
    
    def remove_container(self, position: str, tier: int = None) -> Optional[Any]:
        """
        Remove a container from the specified position.
        
        Args:
            position: Position string (e.g., 'A1')
            tier: Tier number (if None, removes the top container)
            
        Returns:
            Removed container or None if no container was removed
        """
        # Parse the position string to get row and bay
        if position not in self.position_to_coordinates:
            return None
        
        row, bay = self.position_to_coordinates[position]
        
        # If tier not specified, remove the top container
        if tier is None:
            tiers = list(self.yard[row][bay].keys())
            if not tiers:  # No containers in this position
                return None
            
            tier = max(tiers)
        
        # Check if there's a container at the specified tier
        if tier not in self.yard[row][bay]:
            return None
        
        # Check if there are containers above this one
        if any(t > tier for t in self.yard[row][bay].keys()):
            # Can't remove container with containers on top
            return None
        
        # Remove and return the container
        container = self.yard[row][bay].pop(tier)
        
        # Update bay occupancy
        if not self.yard[row][bay]:  # Bay is now empty
            self.bay_occupancy[row][bay] = 0
        else:
            self.bay_occupancy[row][bay] = max(self.yard[row][bay].keys())
        
        return container
    
    def get_stack_height(self, position: str) -> int:
        """
        Get the current stack height at the specified position.
        
        Args:
            position: Position string (e.g., 'A1')
            
        Returns:
            Current stack height (0 if empty)
        """
        # Parse the position string to get row and bay
        if position not in self.position_to_coordinates:
            return 0
        
        row, bay = self.position_to_coordinates[position]
        
        # Return the stack height
        return self.bay_occupancy[row][bay]
    
    def get_containers_at_position(self, position: str) -> Dict[int, Any]:
        """
        Get all containers at a specified position.
        
        Args:
            position: Position string (e.g., 'A1')
            
        Returns:
            Dictionary mapping tier to container
        """
        # Parse the position string to get row and bay
        if position not in self.position_to_coordinates:
            return {}
        
        row, bay = self.position_to_coordinates[position]
        
        # Return a copy of the containers dictionary
        return dict(self.yard[row][bay])
    
    def get_state_representation(self) -> np.ndarray:
        """
        Get a simplified representation of the storage yard state.
        
        Returns:
            3D numpy array representation of the yard
        """
        # Create a 3D array (rows, bays, tiers)
        state = np.zeros((self.num_rows, self.num_bays, self.max_tier_height), dtype=np.int32)
        
        # Fill in the array with container presence (1 for occupied, 0 for empty)
        for row_idx, row in enumerate(self.row_names):
            for bay in range(self.num_bays):
                for tier in range(1, self.max_tier_height + 1):
                    if tier in self.yard[row][bay]:
                        state[row_idx, bay, tier - 1] = 1
        
        return state
    
    def can_accept_container(self, position: str, container: Any) -> bool:
        """
        Check if a container can be added to the specified position.
        
        Args:
            position: Position string (e.g., 'A1')
            container: Container to check
            
        Returns:
            Boolean indicating if the container can be added
        """
        # Parse the position string to get row and bay
        if position not in self.position_to_coordinates:
            return False
        
        row, bay = self.position_to_coordinates[position]
        
        # Check special area constraints
        if container.goods_type == 'Reefer':
            # Reefer containers must be in reefer areas
            if not self.is_position_in_special_area(position, 'reefer'):
                return False
        elif container.goods_type == 'Dangerous':
            # Dangerous containers must be in dangerous goods areas
            if not self.is_position_in_special_area(position, 'dangerous'):
                return False
        else:
            # Regular containers cannot be in special areas
            if (self.is_position_in_special_area(position, 'reefer') or 
                self.is_position_in_special_area(position, 'dangerous')):
                return False
        
        # Check height constraints
        current_height = self.bay_occupancy[row][bay]
        if current_height >= self.max_tier_height:
            return False
        
        # Check stacking compatibility with container below
        if current_height > 0:
            top_tier = max(self.yard[row][bay].keys())
            container_below = self.yard[row][bay][top_tier]
            if not container.can_stack_with(container_below):
                return False
        
        return True
    
    def find_container(self, container_id: str) -> Optional[Tuple[str, int]]:
        """
        Find a container in the storage yard by its ID.
        
        Args:
            container_id: ID of the container to find
            
        Returns:
            Tuple of (position, tier) or None if not found
        """
        for row in self.row_names:
            for bay in range(self.num_bays):
                for tier, container in self.yard[row][bay].items():
                    if container.container_id == container_id:
                        position = f"{row}{bay+1}"
                        return position, tier
        
        return None
    
    def get_containers_by_type(self, container_type: str) -> List[Tuple[str, int, Any]]:
        """
        Get all containers of a specific type.
        
        Args:
            container_type: Type of container to find
            
        Returns:
            List of tuples (position, tier, container)
        """
        results = []
        
        for row in self.row_names:
            for bay in range(self.num_bays):
                for tier, container in self.yard[row][bay].items():
                    if container.container_type == container_type:
                        position = f"{row}{bay+1}"
                        results.append((position, tier, container))
        
        return results
    
    def get_container_count(self) -> int:
        """
        Get the total number of containers in the storage yard.
        
        Returns:
            Total container count
        """
        count = 0
        for row in self.row_names:
            for bay in range(self.num_bays):
                count += len(self.yard[row][bay])
        
        return count
    
    def __str__(self):
        """String representation of the storage yard."""
        rows = []
        rows.append(f"Storage Yard: {self.num_rows} rows, {self.num_bays} bays, {self.get_container_count()} containers")
        
        for row in self.row_names:
            row_str = f"Row {row}: "
            bay_strs = []
            
            for bay in range(self.num_bays):
                if self.yard[row][bay]:
                    tiers = sorted(self.yard[row][bay].keys())
                    containers = [self.yard[row][bay][tier].container_id for tier in tiers]
                    bay_strs.append(f"Bay {bay+1}: {containers}")
                
            if bay_strs:
                row_str += ", ".join(bay_strs)
            else:
                row_str += "Empty"
            
            rows.append(row_str)
        
        return "\n".join(rows)


if __name__ == "__main__":
    # Mock Container class for testing
    class MockContainer:
        def __init__(self, container_id, container_type, goods_type):
            self.container_id = container_id
            self.container_type = container_type
            self.goods_type = goods_type
        
        def can_stack_with(self, other_container):
            # Simple stacking rule for testing
            return (self.container_type == other_container.container_type and 
                    self.goods_type == other_container.goods_type)
        
        def __str__(self):
            return f"{self.container_id} ({self.container_type}, {self.goods_type})"
    
    # Create a test storage yard
    yard = StorageYard(
        num_rows=3,
        num_bays=5,
        max_tier_height=3,
        special_rows={'reefer': ['A'], 'dangerous': ['C']}
    )
    
    # Create some test containers
    containers = [
        MockContainer("CONT1", "TWEU", "Regular"),
        MockContainer("CONT2", "TWEU", "Regular"),
        MockContainer("CONT3", "FEU", "Regular"),
        MockContainer("CONT4", "TWEU", "Reefer"),
        MockContainer("CONT5", "TWEU", "Dangerous")
    ]
    
    # Test adding containers
    print("Adding containers to yard...")
    print(f"Add CONT1 to A1: {yard.add_container('A1', containers[0])}")  # Regular in reefer row (should fail)
    print(f"Add CONT4 to A1: {yard.add_container('A1', containers[3])}")  # Reefer in reefer row (should succeed)
    print(f"Add CONT2 to B1: {yard.add_container('B1', containers[1])}")  # Regular in regular row
    print(f"Add CONT3 to B2: {yard.add_container('B2', containers[2])}")  # Regular in regular row
    print(f"Add CONT5 to C1: {yard.add_container('C1', containers[4])}")  # Dangerous in dangerous row
    
    # Test stacking
    print("\nTesting stacking...")
    cont6 = MockContainer("CONT6", "TWEU", "Reefer")
    print(f"Stack CONT6 on A1: {yard.add_container('A1', cont6)}")  # Should succeed (same type)
    
    cont7 = MockContainer("CONT7", "FEU", "Regular")
    print(f"Stack CONT7 on B1: {yard.add_container('B1', cont7)}")  # Should fail (different type)
    
    # Test get container
    print("\nGetting containers...")
    print(f"Container at A1, tier 1: {yard.get_container('A1', 1)}")
    print(f"Container at A1, tier 2: {yard.get_container('A1', 2)}")
    print(f"Top container at A1: {yard.get_top_container('A1')}")
    
    # Test remove container
    print("\nRemoving containers...")
    print(f"Remove top from A1: {yard.remove_container('A1')}")
    print(f"Remove from B1: {yard.remove_container('B1')}")
    
    # Display yard state
    print("\nFinal yard state:")
    print(yard)