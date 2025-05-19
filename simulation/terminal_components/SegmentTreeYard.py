import numpy as np
from typing import List, Dict, Tuple, Optional, Set, Any
import re


class SegmentNode:
    """Segment tree node for tracking free spaces in a storage yard row."""
    
    def __init__(self, start: int, end: int):
        """Initialize a segment tree node."""
        self.start = start  # Starting bay index (inclusive)
        self.end = end      # Ending bay index (inclusive)
        
        # Maximum free segment length in this range
        self.max_free_segment = end - start + 1
        
        # Free segment length starting from left and ending at right
        self.left_free_segment = end - start + 1
        self.right_free_segment = end - start + 1
        
        # Flag indicating if this entire segment is free
        self.is_fully_free = True
        
        # Track occupied tiers and container info at leaf nodes
        self.occupied_tiers = {}  # Maps tier -> container
        self.max_tier = 0         # Maximum tier height
        
        # Child nodes
        self.left = None
        self.right = None
        
        # For leaf nodes, track position
        self.position = None


class StorageYard:
    """
    Represents the container storage yard in the terminal using segment trees.
    
    The storage yard is a 3D grid where:
    - Rows correspond to different areas in the yard (e.g., A, B, C...)
    - Bays are positions along each row
    - Tiers represent the stacking height at each position
    """
    
    def __init__(self, 
                num_rows: int, 
                num_bays: int, 
                max_tier_height: int = 5,
                row_names: List[str] = None,
                special_areas: Dict[str, List[Tuple[str, int, int]]] = None):
        """
        Initialize the storage yard with segment trees.
        
        Args:
            num_rows: Number of rows in the yard
            num_bays: Number of bays per row
            max_tier_height: Maximum stacking height
            row_names: Names for each row (defaults to A, B, C...)
            special_areas: Dictionary mapping special types to areas
                        e.g., {'reefer': [('A', 1, 5)],
                                'dangerous': [('F', 6, 10)],
                                'trailer': [('A', 15, 25)],
                                'swap_body': [('A', 30, 40)]}
        """
        self.num_rows = num_rows
        self.num_bays = num_bays
        self.max_tier_height = max_tier_height
        
        # Initialize row names if not provided
        if row_names is None:
            self.row_names = [chr(65 + i) for i in range(num_rows)]  # A, B, C, ...
        else:
            self.row_names = row_names[:num_rows]  # Use provided names up to num_rows
        
        # Special areas for different container types (including trailer/swap body)
        if special_areas is None:
            # Default special areas
            self.special_areas = {
                'reefer': [('A', 1, 5)],  # Row A, bays 1-5
                'dangerous': [('F', 6, 10)],  # Row F, bays 6-10
                'trailer': [('A', 15, 25)],  # Row A, bays 15-25
                'swap_body': [('A', 30, 40)]  # Row A, bays 30-40
            }
        else:
            self.special_areas = special_areas
            
        # Validate special areas
        self._validate_special_areas()
        
        # Initialize segment trees for each row
        self.segment_trees = {}
        for row in self.row_names:
            self.segment_trees[row] = self._build_segment_tree(0, self.num_bays - 1, row)
        
        # Keep track of bay occupancy for each row (using numpy arrays for performance)
        self.bay_occupancy = {row: np.zeros(num_bays + 1, dtype=int) for row in self.row_names}
        
        # Mapping from position string to (row, bay) tuple
        self.position_to_coordinates = {}
        self._build_position_mapping()
        
        # Pre-compute special area lookups for faster checking
        self._precompute_special_areas()
        
        # Container registry for efficient lookup
        self.container_registry = {}  # container_id -> (position, tier)

    def _build_segment_tree(self, start: int, end: int, row: str) -> SegmentNode:
        """
        Recursively build a segment tree for a row.
        
        Args:
            start: Start bay index
            end: End bay index
            row: Row identifier
            
        Returns:
            Root node of the segment tree
        """
        node = SegmentNode(start, end)
        
        # Base case: leaf node (single bay)
        if start == end:
            # For leaf nodes, store the position
            bay = start + 1  # Convert 0-based index to 1-based bay number
            node.position = f"{row}{bay}"
            return node
        
        # Recursive case: build left and right subtrees
        mid = (start + end) // 2
        node.left = self._build_segment_tree(start, mid, row)
        node.right = self._build_segment_tree(mid + 1, end, row)
        
        return node
    
    def _update_segment_tree(self, node: SegmentNode, bay: int, tier: int, 
                           add_container: bool, container=None) -> bool:
        """
        Update the segment tree when adding or removing a container.
        
        Args:
            node: Current segment tree node
            bay: Bay index (0-based)
            tier: Tier level
            add_container: True to add container, False to remove
            container: Container object to add
            
        Returns:
            True if update successful, False otherwise
        """
        # Base case: leaf node (single bay)
        if node.start == node.end == bay:
            if add_container:
                # Adding container
                if tier in node.occupied_tiers:
                    return False  # Tier already occupied
                
                node.occupied_tiers[tier] = container
                node.max_tier = max(node.max_tier, tier)
                
                # Mark as occupied if this is the bottom tier
                if tier == 1:
                    node.is_fully_free = False
                    node.max_free_segment = 0
                    node.left_free_segment = 0
                    node.right_free_segment = 0
            else:
                # Removing container
                if tier not in node.occupied_tiers:
                    return False  # No container at this tier
                
                # Remove the container
                container = node.occupied_tiers.pop(tier)
                
                # Update max tier
                if len(node.occupied_tiers) == 0:
                    node.max_tier = 0
                else:
                    node.max_tier = max(node.occupied_tiers.keys())
                
                # Mark as free if this was the bottom tier
                if tier == 1 and 1 not in node.occupied_tiers:
                    node.is_fully_free = True
                    node.max_free_segment = 1
                    node.left_free_segment = 1
                    node.right_free_segment = 1
            
            return True
            
        # Recursive case: update appropriate child
        mid = (node.start + node.end) // 2
        result = False
        
        if bay <= mid:
            result = self._update_segment_tree(node.left, bay, tier, add_container, container)
        else:
            result = self._update_segment_tree(node.right, bay, tier, add_container, container)
        
        # Update node properties based on children
        self._recalculate_node_properties(node)
        
        return result
    
    def _recalculate_node_properties(self, node: SegmentNode):
        """
        Recalculate segment tree node properties based on its children.
        
        Args:
            node: Node to recalculate
        """
        # Skip for leaf nodes without children
        if node.left is None or node.right is None:
            return
        
        # Check if both children are completely free
        node.is_fully_free = node.left.is_fully_free and node.right.is_fully_free
        
        # Calculate maximum free segment:
        # 1. Max segment in left child
        # 2. Max segment in right child
        # 3. Segment that crosses the boundary
        cross_boundary_segment = (node.left.right_free_segment + 
                                 node.right.left_free_segment)
        
        node.max_free_segment = max(
            node.left.max_free_segment,
            node.right.max_free_segment,
            cross_boundary_segment if node.left.right_free_segment > 0 and 
                                   node.right.left_free_segment > 0 else 0
        )
        
        # Update left free segment
        if node.left.is_fully_free:
            # Left child completely free, check if right child's left is free
            node.left_free_segment = (node.left.end - node.left.start + 1 + 
                                     node.right.left_free_segment)
        else:
            node.left_free_segment = node.left.left_free_segment
        
        # Update right free segment (mirror of left calculation)
        if node.right.is_fully_free:
            node.right_free_segment = (node.right.end - node.right.start + 1 + 
                                      node.left.right_free_segment)
        else:
            node.right_free_segment = node.right.right_free_segment
    
    def _precompute_special_areas(self):
        """Pre-compute which positions belong to which special areas for quick lookup."""
        self._special_area_positions = {area_type: set() for area_type in self.special_areas}
        
        for area_type, areas in self.special_areas.items():
            for area_row, start_bay, end_bay in areas:
                for bay in range(start_bay, end_bay + 1):
                    position = f"{area_row}{bay}"
                    self._special_area_positions[area_type].add(position)
    
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
                self.position_to_coordinates[position] = (row, bay - 1)  # Store 0-based bay index
    
    def is_position_in_special_area(self, position: str, container_type: str) -> bool:
        """
        Check if a position is in a special area for a specific container type.
        
        Args:
            position: Position string (e.g., 'A1')
            container_type: Container type to check (e.g., 'Reefer', 'Dangerous', 'Trailer', 'Swap Body')
            
        Returns:
            Boolean indicating if the position is in a special area for the container type
        """
        # Convert container type to lowercase for lookup
        container_type_lower = container_type.lower()
        
        # Use pre-computed positions if available
        if hasattr(self, '_special_area_positions'):
            return position in self._special_area_positions.get(container_type_lower, set())
        
        # Fallback to checking directly
        if position not in self.position_to_coordinates:
            return False
            
        row, bay_idx = self.position_to_coordinates[position]
        bay = bay_idx + 1  # Convert to 1-based bay number
        
        # Check if container type has special areas
        if container_type_lower not in self.special_areas:
            return False
            
        # Check if position is in any of the special areas for this container type
        for area_row, bay_start, bay_end in self.special_areas[container_type_lower]:
            if row == area_row and bay_start <= bay <= bay_end:
                return True
                
        return False

    def clear(self):
        """Clear the entire storage yard."""
        # Reinitialize segment trees
        for row in self.row_names:
            self.segment_trees[row] = self._build_segment_tree(0, self.num_bays - 1, row)
        
        # Reset bay occupancy
        self.bay_occupancy = {row: np.zeros(self.num_bays + 1, dtype=int) for row in self.row_names}
        
        # Clear container registry
        self.container_registry = {}
    
    def get_container(self, position: str, tier: int = None) -> Optional[Any]:
        """
        Get the container at the specified position and tier.
        
        Args:
            position: Position string (e.g., 'A1')
            tier: Tier number (if None, returns the top container)
            
        Returns:
            Container object or None if no container is found
        """
        if position not in self.position_to_coordinates:
            return None
        
        row, bay_idx = self.position_to_coordinates[position]
        bay = bay_idx  # Use 0-based index internally
        
        # Find the leaf node for this position
        node = self._find_leaf_node(self.segment_trees[row], bay)
        
        if node is None:
            return None
        
        # If tier not specified, get the top container
        if tier is None:
            if node.max_tier == 0:
                return None
            tier = node.max_tier
        
        # Return the container at the specified tier
        return node.occupied_tiers.get(tier)
    
    def _find_leaf_node(self, node: SegmentNode, bay: int) -> Optional[SegmentNode]:
        """
        Find the leaf node corresponding to a specific bay.
        
        Args:
            node: Current node in traversal
            bay: Bay index (0-based)
            
        Returns:
            Leaf node or None if not found
        """
        # Base case: leaf node
        if node.start == node.end:
            return node
        
        # Recursive case: traverse to correct child
        mid = (node.start + node.end) // 2
        
        if bay <= mid:
            return self._find_leaf_node(node.left, bay)
        else:
            return self._find_leaf_node(node.right, bay)
    
    def get_top_container(self, position: str) -> Tuple[Optional[Any], Optional[int]]:
        """
        Get the top container at the specified position.
        
        Args:
            position: Position string (e.g., 'A1')
            
        Returns:
            Tuple of (container, tier) or (None, None) if no container is found
        """
        if position not in self.position_to_coordinates:
            return None, None
        
        row, bay_idx = self.position_to_coordinates[position]
        bay = bay_idx  # Use 0-based index internally
        
        # Find the leaf node for this position
        node = self._find_leaf_node(self.segment_trees[row], bay)
        
        if node is None or node.max_tier == 0:
            return None, None
        
        # Get the highest tier
        top_tier = node.max_tier
        return node.occupied_tiers[top_tier], top_tier
    
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
        # Check if position is valid
        if position not in self.position_to_coordinates:
            return False
        
        # Check if container can be accepted at this position
        if not self.can_accept_container(position, container):
            return False
            
        row, bay_idx = self.position_to_coordinates[position]
        bay = bay_idx  # Use 0-based index internally
        
        # Find the leaf node for this position
        node = self._find_leaf_node(self.segment_trees[row], bay)
        
        if node is None:
            return False
        
        # If tier not specified, add to the top
        if tier is None:
            tier = node.max_tier + 1
        
        # Check if the tier is already occupied
        if tier in node.occupied_tiers:
            return False
        
        # Check tier height limit
        if tier > self.max_tier_height:
            return False
        
        # Update segment tree
        success = self._update_segment_tree(self.segment_trees[row], bay, tier, True, container)
        
        if success:
            # Update bay occupancy
            self.bay_occupancy[row][bay + 1] = max(self.bay_occupancy[row][bay + 1], tier)
            
            # Update container registry if container has an ID
            if hasattr(container, 'container_id'):
                self.container_registry[container.container_id] = (position, tier)
        
        return success
    
    def remove_container(self, position: str, tier: int = None) -> Optional[Any]:
        """
        Remove a container from the specified position.
        
        Args:
            position: Position string (e.g., 'A1')
            tier: Tier number (if None, removes the top container)
            
        Returns:
            Removed container or None if no container was removed
        """
        if position not in self.position_to_coordinates:
            return None
        
        row, bay_idx = self.position_to_coordinates[position]
        bay = bay_idx  # Use 0-based index internally
        
        # Find the leaf node for this position
        node = self._find_leaf_node(self.segment_trees[row], bay)
        
        if node is None:
            return None
        
        # If tier not specified, remove the top container
        if tier is None:
            if node.max_tier == 0:
                return None
            tier = node.max_tier
        
        # Check if there's a container at the specified tier
        if tier not in node.occupied_tiers:
            return None
        
        # Check if there are containers above this one
        if any(t > tier for t in node.occupied_tiers.keys()):
            # Can't remove container with containers on top
            return None
        
        # Get the container before removing
        container = node.occupied_tiers[tier]
        
        # Update segment tree
        success = self._update_segment_tree(self.segment_trees[row], bay, tier, False)
        
        if success:
            # Update bay occupancy
            if len(node.occupied_tiers) == 0:
                self.bay_occupancy[row][bay + 1] = 0
            else:
                self.bay_occupancy[row][bay + 1] = max(node.occupied_tiers.keys())
            
            # Update container registry if container has an ID
            if hasattr(container, 'container_id'):
                if container.container_id in self.container_registry:
                    del self.container_registry[container.container_id]
        
        return container if success else None
    
    def get_stack_height(self, position: str) -> int:
        """
        Get the current stack height at the specified position.
        
        Args:
            position: Position string (e.g., 'A1')
            
        Returns:
            Current stack height (0 if empty)
        """
        if position not in self.position_to_coordinates:
            return 0
        
        row, bay_idx = self.position_to_coordinates[position]
        bay = bay_idx  # Use 0-based index internally
        
        # Find the leaf node for this position
        node = self._find_leaf_node(self.segment_trees[row], bay)
        
        if node is None:
            return 0
        
        return node.max_tier
    
    def get_containers_at_position(self, position: str) -> Dict[int, Any]:
        """
        Get all containers at a specified position.
        
        Args:
            position: Position string (e.g., 'A1')
            
        Returns:
            Dictionary mapping tier to container
        """
        if position not in self.position_to_coordinates:
            return {}
        
        row, bay_idx = self.position_to_coordinates[position]
        bay = bay_idx  # Use 0-based index internally
        
        # Find the leaf node for this position
        node = self._find_leaf_node(self.segment_trees[row], bay)
        
        if node is None:
            return {}
        
        # Return a copy of the containers dictionary
        return dict(node.occupied_tiers)
    
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
                position = f"{row}{bay+1}"
                containers = self.get_containers_at_position(position)
                
                for tier in range(1, self.max_tier_height + 1):
                    if tier in containers:
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
        # Check if position is valid
        if position not in self.position_to_coordinates:
            return False
        
        row, bay_idx = self.position_to_coordinates[position]
        
        # Check special area constraints
        if hasattr(container, 'goods_type'):
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
        
        # Check container type constraints
        if hasattr(container, 'container_type'):
            if container.container_type == 'Trailer':
                # Trailers must be in trailer areas
                if not self.is_position_in_special_area(position, 'trailer'):
                    return False
            elif container.container_type == 'Swap Body':
                # Swap bodies must be in swap body areas
                if not self.is_position_in_special_area(position, 'swap_body'):
                    return False
        
        # Check height constraints
        current_height = self.get_stack_height(position)
        if current_height >= self.max_tier_height:
            return False
        
        # Check stacking compatibility with container below
        if current_height > 0:
            container_below, _ = self.get_top_container(position)
            if container_below and hasattr(container, 'can_stack_with'):
                if not container.can_stack_with(container_below):
                    return False
        
        # Check container-specific can_be_stacked rules
        if hasattr(container, 'is_stackable') and not container.is_stackable and current_height > 0:
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
        # Check container registry first (O(1) lookup)
        if container_id in self.container_registry:
            return self.container_registry[container_id]
        
        # Fallback to searching all positions
        for row in self.row_names:
            for bay in range(1, self.num_bays + 1):
                position = f"{row}{bay}"
                containers = self.get_containers_at_position(position)
                
                for tier, container in containers.items():
                    if hasattr(container, 'container_id') and container.container_id == container_id:
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
            for bay in range(1, self.num_bays + 1):
                position = f"{row}{bay}"
                containers = self.get_containers_at_position(position)
                
                for tier, container in containers.items():
                    if (hasattr(container, 'container_type') and 
                        container.container_type == container_type):
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
            for bay in range(1, self.num_bays + 1):
                position = f"{row}{bay}"
                containers = self.get_containers_at_position(position)
                count += len(containers)
        
        return count
        
    def calc_possible_moves(self, start_position: str, n: int) -> List[str]:
        """
        Calculate all possible positions a container can be moved to within n bays left and right.
        
        Args:
            start_position: Starting position string (e.g., 'A1')
            n: Number of bays to consider in each direction
            
        Returns:
            List of valid destination positions
        """
        # Check if position is valid
        if start_position not in self.position_to_coordinates:
            return []
            
        # Get container to move
        container, tier = self.get_top_container(start_position)
        if container is None:
            return []  # No container to move
            
        # Get row and bay for starting position
        row, bay_idx = self.position_to_coordinates[start_position]
        bay = bay_idx + 1  # Convert to 1-based bay
        
        # Calculate bay range to check
        min_bay = max(1, bay - n)
        max_bay = min(self.num_bays, bay + n)
        
        valid_destinations = []
        
        # Check positions in the same row
        for check_bay in range(min_bay, max_bay + 1):
            # Skip the starting position
            if check_bay == bay:
                continue
                
            # Generate position string
            dest_position = f"{row}{check_bay}"
            
            # Check if container can be accepted at this position
            if self.can_accept_container(dest_position, container):
                valid_destinations.append(dest_position)
        
        # Efficiently find positions using segment tree
        # This simulates the container being temporarily removed for validation
        removed_container = None
        try:
            # Temporarily remove container to check valid moves
            removed_container = self.remove_container(start_position)
            
            # Now search for valid placements efficiently using segment tree structure
            for check_bay in range(min_bay, max_bay + 1):
                if check_bay == bay:
                    continue
                
                dest_position = f"{row}{check_bay}"
                if self.can_accept_container(dest_position, container):
                    valid_destinations.append(dest_position)
                    
        finally:
            # Always put the container back
            if removed_container:
                self.add_container(start_position, removed_container)
        
        return valid_destinations
    
    def find_next_valid_position(self, container: Any, start_bay: int = 0, row: str = None) -> Optional[str]:
        """
        Find the next valid position for a container using segment tree.
        
        Args:
            container: Container to place
            start_bay: Bay to start searching from (0-based)
            row: Row to search in (if None, searches all rows)
            
        Returns:
            Position string or None if no valid position found
        """
        if row is None:
            # Search in all rows
            for search_row in self.row_names:
                position = self.find_next_valid_position(container, start_bay, search_row)
                if position:
                    return position
            return None
            
        if row not in self.row_names:
            return None
            
        # Use segment tree to efficiently find space
        root = self.segment_trees[row]
        
        # Special handling for container types
        required_space = 1  # Default for standard containers
        
        # For special containers like trailers that need more space
        if hasattr(container, 'container_type'):
            if container.container_type in ['Trailer', 'Swap Body']:
                # These typically need special areas
                special_area_type = container.container_type.lower()
                
                # Find positions in special areas
                for area_row, bay_start, bay_end in self.special_areas.get(special_area_type, []):
                    if area_row != row:
                        continue
                        
                    # Check each position in the special area
                    for bay in range(max(bay_start, start_bay + 1), bay_end + 1):
                        position = f"{row}{bay}"
                        if self.can_accept_container(position, container):
                            return position
                
                # If we reach here, no valid position in special areas
                return None
        
        # For regular containers, use segment tree to find free space
        if hasattr(container, 'goods_type'):
            # Handle special goods types
            if container.goods_type in ['Reefer', 'Dangerous']:
                special_area_type = container.goods_type.lower()
                
                # Find positions in special areas
                for area_row, bay_start, bay_end in self.special_areas.get(special_area_type, []):
                    if area_row != row:
                        continue
                        
                    # Check each position in the special area
                    for bay in range(max(bay_start, start_bay + 1), bay_end + 1):
                        position = f"{row}{bay}"
                        if self.can_accept_container(position, container):
                            return position
                
                # If we reach here, no valid position in special areas
                return None
        
        # Standard search for regular containers
        # Use segment tree for efficiency
        valid_bays = self._find_valid_bays(root, start_bay, required_space)
        
        # Check each candidate bay for stacking rules
        for bay in valid_bays:
            position = f"{row}{bay + 1}"  # Convert to 1-based bay
            if self.can_accept_container(position, container):
                return position
                
        return None
    
    def _find_valid_bays(self, node: SegmentNode, start_bay: int, required_space: int) -> List[int]:
        """
        Find valid bays using segment tree.
        
        Args:
            node: Current segment tree node
            start_bay: Bay to start search from (0-based)
            required_space: Required space for container
            
        Returns:
            List of valid bay indices
        """
        # If node doesn't have enough space or is before our start position
        if node.max_free_segment < required_space or node.end < start_bay:
            return []
            
        # Base case: leaf node
        if node.start == node.end:
            return [node.start] if node.is_fully_free and node.start >= start_bay else []
            
        # Check if we can find positions in children
        valid_bays = []
        
        # Check left child if it includes or is after start_bay
        if node.left and node.left.end >= start_bay:
            valid_bays.extend(self._find_valid_bays(node.left, start_bay, required_space))
            
        # Check right child
        if node.right:
            valid_bays.extend(self._find_valid_bays(node.right, start_bay, required_space))
            
        return valid_bays
    
    def __str__(self):
        """String representation of the storage yard."""
        rows = []
        rows.append(f"Storage Yard: {self.num_rows} rows, {self.num_bays} bays, {self.get_container_count()} containers")
        
        for row in self.row_names:
            row_str = f"Row {row}: "
            bay_strs = []
            
            for bay in range(1, self.num_bays + 1):
                position = f"{row}{bay}"
                containers = self.get_containers_at_position(position)
                
                if containers:
                    container_ids = []
                    for tier in sorted(containers.keys()):
                        container = containers[tier]
                        container_id = container.container_id if hasattr(container, 'container_id') else str(container)
                        container_ids.append(f"{tier}:{container_id}")
                    bay_strs.append(f"Bay {bay}: [{', '.join(container_ids)}]")
            
            if bay_strs:
                row_str += ", ".join(bay_strs)
            else:
                row_str += "Empty"
            
            rows.append(row_str)
        
        return "\n".join(rows)
    
def main():
    """Test the SegmentTree-based StorageYard implementation with realistic scenarios."""
    try:
        # Try to import from the actual repo
        # from simulation.terminal_components.Container import Container, ContainerFactory
        from Container import Container, ContainerFactory
    except ImportError:
        print("Could not import Container classes from repository, using mock implementations")

    import time
    import random
    
    print("\n===== Testing SegmentTree-based StorageYard Implementation =====")
    
    # 1. Create a storage yard with realistic dimensions
    yard = StorageYard(
        num_rows=6,               # Rows A-F
        num_bays=40,              # 40 bays per row
        max_tier_height=5,        # Maximum stacking height
        special_areas={
            'reefer': [('A', 1, 5), ('F', 35, 40)],     # Reefer areas
            'dangerous': [('C', 25, 30)],                # Dangerous goods area
            'trailer': [('A', 15, 25)],                  # Trailer area
            'swap_body': [('B', 30, 40)]                 # Swap body area
        }
    )
    
    print(f"Created storage yard with {yard.num_rows} rows and {yard.num_bays} bays")
    
    # 2. Create a variety of containers
    print("\n----- Creating test containers -----")
    
    # Regular containers
    containers = [
        ContainerFactory.create_container("CONT001", "TWEU", "Import", "Regular", weight=20000),
        ContainerFactory.create_container("CONT002", "FEU", "Export", "Regular", weight=25000),
        ContainerFactory.create_container("CONT003", "THEU", "Import", "Regular", weight=18000),
        # Reefer containers
        ContainerFactory.create_container("CONT004", "TWEU", "Import", "Reefer", weight=22000),
        ContainerFactory.create_container("CONT005", "FEU", "Export", "Reefer", weight=24000),
        # Dangerous goods
        ContainerFactory.create_container("CONT006", "TWEU", "Import", "Dangerous", weight=19000),
        ContainerFactory.create_container("CONT007", "FEU", "Export", "Dangerous", weight=27000),
        # Special types
        ContainerFactory.create_container("CONT008", "Trailer", "Export", "Regular", weight=15000),
        ContainerFactory.create_container("CONT009", "Swap Body", "Export", "Regular", weight=12000),
    ]
    
    # 3. Test basic yard operations
    print("\n----- Testing basic yard operations -----")
    
    # Add containers to appropriate areas
    placements = [
        # Regular containers in regular areas
        ('D10', containers[0]),
        ('D11', containers[1]),
        ('D12', containers[2]),
        # Reefer containers in reefer areas
        ('A3', containers[3]),
        ('F38', containers[4]),
        # Dangerous goods in dangerous area
        ('C27', containers[5]),
        ('C28', containers[6]),
        # Special containers in special areas
        ('A18', containers[7]),  # Trailer
        ('B35', containers[8])   # Swap body
    ]
    
    # Add containers to the yard
    successful_placements = []
    for position, container in placements:
        success = yard.add_container(position, container)
        print(f"Adding {container.container_id} to {position}: {'Success' if success else 'Failed'}")
        if success:
            successful_placements.append((position, container))
    
    # 4. Test stack operations
    print("\n----- Testing stacking operations -----")
    
    # Create containers for stacking
    stack_position = 'D15'
    stack_containers = [
        ContainerFactory.create_container("STACK001", "TWEU", "Import", "Regular", weight=24000),
        ContainerFactory.create_container("STACK002", "TWEU", "Import", "Regular", weight=20000),
        ContainerFactory.create_container("STACK003", "TWEU", "Import", "Regular", weight=18000),
        ContainerFactory.create_container("STACK004", "TWEU", "Import", "Regular", weight=15000),
    ]
    
    # Create a stack and check stacking rules
    for i, container in enumerate(stack_containers):
        tier = i + 1
        success = yard.add_container(stack_position, container, tier)
        print(f"Adding {container.container_id} to {stack_position} tier {tier}: {'Success' if success else 'Failed'}")
    
    # 5. Test invalid placement scenarios
    print("\n----- Testing invalid placement scenarios -----")
    
    # Try to add a reefer container to a non-reefer area
    reefer_container = ContainerFactory.create_container("INVALID01", "TWEU", "Import", "Reefer")
    success = yard.add_container('D20', reefer_container)
    print(f"Adding reefer container to non-reefer area: {'Success' if success else 'Failed (expected)'}")
    
    # Try to add a dangerous container to a non-dangerous area
    dangerous_container = ContainerFactory.create_container("INVALID02", "TWEU", "Import", "Dangerous")
    success = yard.add_container('E25', dangerous_container)
    print(f"Adding dangerous container to non-dangerous area: {'Success' if success else 'Failed (expected)'}")
    
    # Try to add a trailer outside trailer area
    trailer_container = ContainerFactory.create_container("INVALID03", "Trailer", "Export", "Regular")
    success = yard.add_container('E30', trailer_container)
    print(f"Adding trailer outside trailer area: {'Success' if success else 'Failed (expected)'}")
    
    # Try to stack a container with wrong stacking rules
    wrong_stack_container = ContainerFactory.create_container("INVALID04", "FEU", "Import", "Regular")
    success = yard.add_container(stack_position, wrong_stack_container, 5)
    print(f"Adding incompatible container to stack: {'Success' if success else 'Failed (expected)'}")
    
    # 6. Test finding containers
    print("\n----- Testing container lookups -----")
    
    # Find containers by ID
    for container_id in ["CONT001", "CONT004", "CONT008", "STACK002", "NONEXISTENT"]:
        result = yard.find_container(container_id)
        if result:
            position, tier = result
            print(f"Found {container_id} at {position}, tier {tier}")
        else:
            print(f"Container {container_id} not found")
    
    # 7. Test new calc_possible_moves function
    print("\n----- Testing calc_possible_moves function -----")
    
    # Test within regular areas
    test_positions = [
        # Regular container
        ('D10', 3),
        # Reefer container
        ('A3', 2),
        # Dangerous goods
        ('C27', 3),
        # Trailer
        ('A18', 3),
        # Stack top container
        (stack_position, 5)
    ]
    
    for position, n in test_positions:
        container, tier = yard.get_top_container(position)
        if container:
            print(f"\nCalculating possible moves for {container.container_id} at {position} within {n} bays:")
            start_time = time.time()
            valid_moves = yard.calc_possible_moves(position, n)
            calc_time = (time.time() - start_time) * 1000  # ms
            print(f"Valid destinations ({len(valid_moves)}): {valid_moves}")
            print(f"Calculated in {calc_time:.3f} ms")
    
    # 8. Performance test with many containers
    print("\n----- Performance testing -----")
    
    # Generate a large set of random positions for performance testing
    test_count = 1000
    
    # Create a large batch of random containers
    print(f"Testing with {test_count} containers...")
    start_time = time.time()
    
    for i in range(test_count):
        # Generate random parameters
        container_type = random.choice(["TWEU", "FEU", "THEU", "Trailer", "Swap Body"])
        goods_type = random.choice(["Regular", "Regular", "Regular", "Reefer", "Dangerous"])
        
        # Create container
        container = ContainerFactory.create_container(
            f"PERF{i:04d}", container_type, 
            "Import" if container_type not in ["Trailer", "Swap Body"] else "Export",
            goods_type
        )
        
        # Find suitable position based on container type
        if container_type == "Trailer":
            row = "A"
            bay = random.randint(15, 25)
        elif container_type == "Swap Body":
            row = "B"
            bay = random.randint(30, 40)
        elif goods_type == "Reefer":
            if random.choice([True, False]):
                row = "A"
                bay = random.randint(1, 5)
            else:
                row = "F"
                bay = random.randint(35, 40)
        elif goods_type == "Dangerous":
            row = "C"
            bay = random.randint(25, 30)
        else:
            # Regular container
            row = random.choice(["D", "E"])
            bay = random.randint(1, 30)
        
        # Check if position is valid
        position = f"{row}{bay}"
        yard.can_accept_container(position, container)
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Time for {test_count} container validations: {total_time:.3f} seconds")
    print(f"Average time per container: {total_time / test_count * 1000:.3f} ms")
    
    # 9. Test comprehensive moves across different container types
    print("\n----- Testing comprehensive move scenarios -----")
    
    # 1. Test regular container moves
    regular_position = 'D10'
    print(f"\nTesting regular container at {regular_position}:")
    moves_3 = yard.calc_possible_moves(regular_position, 3)
    moves_5 = yard.calc_possible_moves(regular_position, 5)
    moves_10 = yard.calc_possible_moves(regular_position, 10)
    print(f"Possible moves within 3 bays: {len(moves_3)} positions")
    print(f"Possible moves within 5 bays: {len(moves_5)} positions")
    print(f"Possible moves within 10 bays: {len(moves_10)} positions")
    
    # 2. Test reefer container moves (should only find reefer areas)
    reefer_position = 'A3'
    print(f"\nTesting reefer container at {reefer_position}:")
    reefer_moves = yard.calc_possible_moves(reefer_position, 10)
    print(f"Possible moves for reefer container: {reefer_moves}")
    print(f"Note: Only reefer areas should be included")
    
    # 3. Test trailer container moves (should only find trailer areas)
    trailer_position = 'A18'
    print(f"\nTesting trailer at {trailer_position}:")
    trailer_moves = yard.calc_possible_moves(trailer_position, 10)
    print(f"Possible moves for trailer: {trailer_moves}")
    print(f"Note: Only trailer areas should be included")
    
    # 10. Test container removal
    print("\n----- Testing container removal -----")
    
    # Remove a container from middle of stack (should fail)
    remove_result = yard.remove_container(stack_position, 2)
    print(f"Removing container from middle of stack: {'Success' if remove_result else 'Failed (expected)'}")
    
    # Remove container from top of stack (should succeed)
    top_container, top_tier = yard.get_top_container(stack_position)
    remove_result = yard.remove_container(stack_position, top_tier)
    print(f"Removing container from top of stack: {'Success' if remove_result else 'Failed'}")
    
    # Remove a trailer (should succeed)
    remove_result = yard.remove_container('A18')
    print(f"Removing trailer: {'Success' if remove_result else 'Failed'}")
    
    # 11. Print summary statistics
    print("\n----- Final Yard State -----")
    print(f"Total containers in yard: {yard.get_container_count()}")
    
    for row in yard.row_names:
        row_count = 0
        for bay in range(1, yard.num_bays + 1):
            position = f"{row}{bay}"
            containers = yard.get_containers_at_position(position)
            row_count += len(containers)
        print(f"Row {row}: {row_count} containers")

if __name__ == "__main__":
    main()