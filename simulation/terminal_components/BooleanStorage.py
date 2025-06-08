from typing import Dict, Tuple, List, Optional
from simulation.terminal_components.Container import Container, ContainerFactory
import numpy as np
import torch
from collections import defaultdict

class BooleanStorageYard:
    def __init__(self, 
                 n_rows: int, 
                 n_bays: int, 
                 n_tiers: int,
                 coordinates: List[Tuple[int, int, str]],
                 split_factor: int = 4,
                 validate: bool = False,
                 device: str = 'cuda'):

        # Type assertion to check for wrong input        
        assert type(n_rows) == int
        assert type(n_bays) == int
        assert type(n_tiers) == int
        assert type(split_factor) == int
        assert type(validate) == bool

        # Saving the yard configuration
        self.n_rows = n_rows
        self.n_bays = n_bays
        self.n_tiers = n_tiers
        self.split_factor = split_factor
        self.device = device

        # OPTIMIZED: Replace dictionary with 4D numpy array for O(1) access
        # Shape: (n_rows, n_bays, n_tiers, split_factor)
        self.containers = np.full((n_rows, n_bays, n_tiers, split_factor), 
                                  None, dtype=object)
        
        # Create container property arrays for fast tensor conversion
        self._init_property_arrays()

        # Declaring base placeability of containers
        self.dynamic_yard_mask = self.create_dynamic_yard_mask()
        '''boolean mask for whole yard, where False means that a place is occupied'''

        # Create coordinate mapping for yard
        self.coordinates = self.create_coordinate_mapping()

        # Creating masks for specific container types
        self.r_mask, self.dg_mask, self.sb_t_mask = self.extract_special_masks(coordinates)

        # AND product results in mask of available spots for regular containers
        self.reg_mask = ~np.zeros_like(self.dynamic_yard_mask, dtype=bool)
        self.reg_mask = self.reg_mask & ~self.r_mask & ~self.dg_mask #& ~self.sb_t_mask not this because they share stacking space.
 
        if validate:
            self.print_masks()

        # Define container lengths
        self.container_lengths: dict = {
            "TWEU": 2,
            "THEU": 3,
            "FEU": 4,
            "Swap Body": 4,
            "Trailer": 4, 
            "FFEU": 5
        }
        '''Container lengths defined in ammount of subslots that they use up '''

        self.cldymc = {
            k:self.dynamic_yard_mask.copy() for k in self.container_lengths
        }
        '''Container-Lengths-Dynamic-Yard-Mask-Copy for each different container length'''

        # Create mappings for fast tensor conversion
        self.container_type_to_id = {
            None: 0, "TWEU": 1, "THEU": 2, "FEU": 3, "FFEU": 4, 
            "Swap Body": 5, "Trailer": 6
        }
        self.goods_type_to_id = {
            None: 0, "Regular": 1, "Reefer": 2, "Dangerous": 3
        }

    def _init_property_arrays(self):
        """Initialize arrays to store container properties for fast tensor conversion."""
        # Arrays to store extracted properties for tensor conversion
        self.container_type_ids = np.zeros((self.n_rows, self.n_bays, self.n_tiers, self.split_factor), dtype=np.int8)
        self.goods_type_ids = np.zeros((self.n_rows, self.n_bays, self.n_tiers, self.split_factor), dtype=np.int8)
        self.container_weights = np.zeros((self.n_rows, self.n_bays, self.n_tiers, self.split_factor), dtype=np.float32)
        self.container_priorities = np.zeros((self.n_rows, self.n_bays, self.n_tiers, self.split_factor), dtype=np.float32)
        self.occupied_mask = np.zeros((self.n_rows, self.n_bays, self.n_tiers, self.split_factor), dtype=bool)

    def create_dynamic_yard_mask(self)->np.ndarray:
        bool_arr = np.zeros((self.n_rows*self.n_tiers, self.n_bays*self.split_factor), dtype=bool)
        for row in range(self.n_rows):
            for bay in range(self.n_bays):
                for tier in range(self.n_tiers):
                    for split in range(self.split_factor):
                        if tier == 0:
                            bool_arr[row*self.n_tiers+tier][bay*self.split_factor+split] = True
        
        return bool_arr

    def create_coordinate_mapping(self)->np.ndarray:
        ''' 0-starting coordinate map as identical mask to the dynamic_yard_mask'''
        coordinate_arr = np.zeros((self.n_rows*self.n_tiers, self.n_bays*self.split_factor), dtype=object)
        for row in range(self.n_rows):
            for bay in range(self.n_bays):
                for tier in range(self.n_tiers):
                    for split in range(self.split_factor):
                        coordinate_format = (row, bay, split, tier)
                        coordinate_arr[row*self.n_tiers+tier][bay*self.split_factor+split] = coordinate_format
        
        return coordinate_arr

    def extract_special_masks(self, coordinates: List[Tuple[int, int, str]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''
        Creates special masks for:\n
        - Reefers
        - Dangerous Goods Containers
        - Swap Bodies/Trailers\n
        by copying inverse of dynamic_yard_mask and filling individual masks with given 
        coordinates through match case logic. It also adjusts to stacking height und sub-
        division of container places with class parameters n_tiers and split_factor.
        1-starting coordinates are meant to be injected here as it is an init func
        '''
        r_mask = np.zeros_like(self.dynamic_yard_mask, dtype=bool)  # Start with all FALSE
        dg_mask = np.zeros_like(self.dynamic_yard_mask, dtype=bool)  # Start with all FALSE
        sb_t_mask = np.zeros_like(self.dynamic_yard_mask, dtype=bool)  # Start with all FALSE
        for coordinate in coordinates:
            bay, row, class_type = coordinate
            # Correct the coordinates to 0-start
            bay -= 1
            row -= 1
            match class_type:
                case "r":
                    for tier in range(self.n_tiers):
                        for split in range(self.split_factor):
                            r_mask[row*self.n_tiers+tier, bay*self.split_factor+split] = True
                case "dg":
                    for tier in range(self.n_tiers):
                        for split in range(self.split_factor):
                            dg_mask[row*self.n_tiers+tier, bay*self.split_factor+split] = True
                case "sb_t":
                    for tier in range(self.n_tiers):
                        for split in range(self.split_factor):
                            # non-stackable
                            if tier == 0:
                                sb_t_mask[row*self.n_tiers+tier, bay*self.split_factor+split] = True
                case _:
                    raise Exception("Storage Yard Class: invalid coordinates passed") 
        return (r_mask, dg_mask, sb_t_mask)

    def print_masks(self) -> None:
        'Prints all masks for visual verification of corectness'
                
        # Setting print options to be able to see masks nicely
        np.set_printoptions(
            threshold=np.inf,          # Print entire array, don't truncate
            linewidth=200,             # Reasonable line width
            suppress=True,             # Suppress scientific notation
            precision=0,               # No decimal places for boolean
            formatter={'bool': lambda x: '■' if x else '□'}  # Visual representation
        )

        print('Reefer Mask')
        print(self.r_mask)

        print('DG Mask')
        print(self.dg_mask)
        
        print('Swap Body/Trailer Mask')
        print(self.sb_t_mask)
        
        print("Mask for regular Containers")
        print(self.reg_mask)

        print("Dynamic Yard at init")
        print(self.dynamic_yard_mask)
        print(self.dynamic_yard_mask.shape)

    def _update_property_arrays(self, row: int, bay: int, tier: int, split: int, container: Optional[Container]):
        """Update property arrays when containers are added/removed."""
        if container is not None:
            # Container added
            self.container_type_ids[row, bay, tier, split] = self.container_type_to_id.get(container.container_type, 0)
            self.goods_type_ids[row, bay, tier, split] = self.goods_type_to_id.get(container.goods_type, 0)
            self.container_weights[row, bay, tier, split] = getattr(container, 'weight', 0.0)
            self.container_priorities[row, bay, tier, split] = getattr(container, 'priority', 0.0)
            self.occupied_mask[row, bay, tier, split] = True
        else:
            # Container removed
            self.container_type_ids[row, bay, tier, split] = 0
            self.goods_type_ids[row, bay, tier, split] = 0
            self.container_weights[row, bay, tier, split] = 0.0
            self.container_priorities[row, bay, tier, split] = 0.0
            self.occupied_mask[row, bay, tier, split] = False

    def add_container(self, container: Container, coordinates: List[Tuple[int, int, int, int]]):
        '''
        OPTIMIZED: Direct array access instead of string-based dictionary.
        
        Args:
            - container: Object of Container class that is suposed to be placed into yard
            - coordinates: List[row, bay, sub-bay, tier] 
            -> 
            row, bay(s) - possible use of subdivision of yard places for
            complex containers and tier of placeable container\n
        Result:
            - step 1: updates the self.dynamic_yard_mask[coordinates] == false
            - step 2: updates the self.containers[coordinates] == Container
            - step 3: if possible unlocks next tier
        '''

        for coordinate in coordinates:
            # unpack coordinate
            row, bay, split, tier = coordinate

            # OPTIMIZED: Direct array indexing instead of string key lookup
            self.containers[row, bay, tier, split] = container
            
            # Update property arrays for tensor conversion
            self._update_property_arrays(row, bay, tier, split, container)
            
            self.dynamic_yard_mask[row*self.n_tiers+tier, bay*self.split_factor+split] = False
            
            # lock stack for that container type
            self.cldymc[container.container_type][row*self.n_tiers:(row*self.n_tiers)+self.n_tiers-1, bay*self.split_factor+split] = False

            # unlock the next tier
            if tier < self.n_tiers - 1:
                self.dynamic_yard_mask[row*self.n_tiers+tier+1, bay*self.split_factor+split] = True

    def remove_container(self, coordinates: List[Tuple[int, int, int, int]]) -> Container:
        '''
        OPTIMIZED: Direct array access instead of string-based dictionary.
        
        Args:
            - coordinates: List[Tuple(row, bay, sub-bay, tier)]
            -> 
            row, bay(s) - possible use of subdivision of yard places for
            complex containers and tier of placeable container\n
        Result:
            - step 1: updates the self.dynamic_yard_mask[coordinates] == false
            - step 2: updates the self.containers[coordinates] == Container
        '''
        container_saved: bool = False
        for coordinate in coordinates:
            # unpack coordinate
            row, bay, split, tier = coordinate
            
            if not container_saved:
                # OPTIMIZED: Direct array access instead of string key lookup
                container = self.containers[row, bay, tier, split]
                container_saved = True

            # remove container from yard
            self.containers[row, bay, tier, split] = None
            
            # Update property arrays for tensor conversion
            self._update_property_arrays(row, bay, tier, split, None)
            
            self.dynamic_yard_mask[row*self.n_tiers+tier, bay*self.split_factor+split] = True

            # unlock the stack for all container types if the removal opens a stack
            if tier == 0:
                self.cldymc[container.container_type][row*self.n_tiers:(row*self.n_tiers)+self.n_tiers-1, bay*self.split_factor+split] = True

            # if not max height, lock tier above
            if tier < self.n_tiers - 1:
                self.dynamic_yard_mask[row*self.n_tiers+tier+1, bay*self.split_factor+split] = False

        return container
    
    def move_container(self, 
                       loc_coordinates: List[Tuple[int, int, int, int]], 
                       dest_coordinates: List[Tuple[int, int, int, int]]
                       )->None:
        '''
        Args:
            - loc_coordinates: List of container coordinates in yard
            - dest_coordinates: List of destination coordinates in yard
        Description:
        Moves container in yard through following steps:
            - remove container from old spot
            - add container to new spot 
        '''
        container = self.remove_container(loc_coordinates)
        self.add_container(container, dest_coordinates)

    # ==================== TENSOR CONVERSION FUNCTIONS ====================
    
    def get_occupied_tensor(self, as_tensor: bool = True) -> torch.Tensor:
        """
        Get tensor representing which positions are occupied.
        
        Returns:
            Boolean tensor of shape (n_rows, n_bays, n_tiers, split_factor)
        """
        if as_tensor:
            return torch.from_numpy(self.occupied_mask).to(self.device)
        return self.occupied_mask
    
    def get_container_type_tensor(self, as_tensor: bool = True) -> torch.Tensor:
        """
        Get tensor of container type IDs.
        
        Returns:
            Integer tensor of shape (n_rows, n_bays, n_tiers, split_factor)
            Values: 0=empty, 1=TWEU, 2=THEU, 3=FEU, 4=FFEU, 5=SwapBody, 6=Trailer
        """
        if as_tensor:
            return torch.from_numpy(self.container_type_ids).to(self.device)
        return self.container_type_ids
    
    def get_goods_type_tensor(self, as_tensor: bool = True) -> torch.Tensor:
        """
        Get tensor of goods type IDs.
        
        Returns:
            Integer tensor of shape (n_rows, n_bays, n_tiers, split_factor)
            Values: 0=empty, 1=Regular, 2=Reefer, 3=Dangerous
        """
        if as_tensor:
            return torch.from_numpy(self.goods_type_ids).to(self.device)
        return self.goods_type_ids
    
    def get_weights_tensor(self, as_tensor: bool = True) -> torch.Tensor:
        """
        Get tensor of container weights.
        
        Returns:
            Float tensor of shape (n_rows, n_bays, n_tiers, split_factor)
        """
        if as_tensor:
            return torch.from_numpy(self.container_weights).to(self.device)
        return self.container_weights
    
    def get_priorities_tensor(self, as_tensor: bool = True) -> torch.Tensor:
        """
        Get tensor of container priorities.
        
        Returns:
            Float tensor of shape (n_rows, n_bays, n_tiers, split_factor)
        """
        if as_tensor:
            return torch.from_numpy(self.container_priorities).to(self.device)
        return self.container_priorities
    
    def get_full_state_tensor(self, flatten: bool = False) -> torch.Tensor:
        """
        Get complete state representation as a single tensor.
        
        Args:
            flatten: If True, flatten spatial dimensions for MLP input
            
        Returns:
            Tensor of shape (5, n_rows, n_bays, n_tiers, split_factor) or flattened
            Channels: [occupied, container_type, goods_type, weight, priority]
        """
        occupied = torch.from_numpy(self.occupied_mask.astype(np.float32)).to(self.device)
        container_types = torch.from_numpy(self.container_type_ids.astype(np.float32)).to(self.device)
        goods_types = torch.from_numpy(self.goods_type_ids.astype(np.float32)).to(self.device)
        weights = torch.from_numpy(self.container_weights).to(self.device)
        priorities = torch.from_numpy(self.container_priorities).to(self.device)
        
        # Stack all channels
        state = torch.stack([occupied, container_types, goods_types, weights, priorities], dim=0)
        
        if flatten:
            # Flatten for MLP: (5 * n_rows * n_bays * n_tiers * split_factor,)
            state = state.flatten()
        
        return state
    
    def get_compact_state_tensor(self) -> torch.Tensor:
        """
        Get compact state representation for efficient DRL training.
        
        Returns:
            Tensor with aggregated features per row/bay for fast processing
        """
        # Aggregate by row and bay
        occupied_by_row_bay = torch.from_numpy(
            self.occupied_mask.sum(axis=(2, 3)).astype(np.float32)
        ).to(self.device)
        
        weights_by_row_bay = torch.from_numpy(
            self.container_weights.sum(axis=(2, 3))
        ).to(self.device)
        
        # Stack aggregated features
        compact_state = torch.stack([occupied_by_row_bay, weights_by_row_bay], dim=0)
        
        return compact_state.flatten()

    # ==================== OPTIMIZED LOOKUP FUNCTIONS ====================
    
    def get_container_at(self, row: int, bay: int, tier: int, split: int) -> Optional[Container]:
        """OPTIMIZED: Direct array access for container lookup."""
        return self.containers[row, bay, tier, split]
    
    def set_container_at(self, row: int, bay: int, tier: int, split: int, container: Optional[Container]):
        """OPTIMIZED: Direct array access for container setting."""
        self.containers[row, bay, tier, split] = container
        self._update_property_arrays(row, bay, tier, split, container)
    
    def is_position_occupied(self, row: int, bay: int, tier: int, split: int) -> bool:
        """OPTIMIZED: Direct mask access for occupancy check."""
        return self.occupied_mask[row, bay, tier, split]

    # ==================== REMAINING METHODS ====================

    def _find_valid_container_placements(self, available_coordinates, container_type: str) -> List[Tuple]:
        """
        OPTIMIZED: Find valid starting positions for containers using bit manipulation.
        Time complexity: O(1) per position group.
        
        Args:
            available_coordinates: NumPy array of (row, bay, split, tier) tuples
            container_type: Container type to get length for
            
        Returns:
            List of valid starting positions as (row, bay, tier, start_split) tuples
        """
        if len(available_coordinates) == 0 or container_type not in self.container_lengths:
            return []
        
        container_length = self.container_lengths[container_type]
        
        # Handle cross-bay containers (like FFEU=5 with split_factor=4)
        if container_length > self.split_factor:
            return self._find_cross_bay_placements(available_coordinates, container_type)
        
        # Group coordinates by (row, bay, tier) and create bitmasks
        position_groups = defaultdict(int)
        
        for coord in available_coordinates:
            row, bay, split, tier = coord
            key = (row, bay, tier)  # Group by position
            position_groups[key] |= (1 << split)  # Set bit at split position
        
        valid_placements = []
        
        # Create container mask (e.g., length=2 → 0b11, length=4 → 0b1111)
        container_mask = (1 << container_length) - 1
        
        # Check each position group
        for (row, bay, tier), bitmask in position_groups.items():
            # Only allow START or END positions, not middle positions
            valid_start_positions = self._get_valid_start_positions(container_length)
            
            for start_split in valid_start_positions:
                if start_split + container_length <= self.split_factor:  # Safety check
                    shifted_mask = container_mask << start_split
                    
                    # O(1) check: are all required consecutive positions available?
                    if (bitmask & shifted_mask) == shifted_mask:
                        valid_placements.append((row, bay, tier, start_split))
        
        return valid_placements

    def _get_valid_start_positions(self, container_length: int) -> List[int]:
        """
        Get valid starting positions for containers based on START/END placement rule.
        
        Args:
            container_length: Length of container in split positions
            
        Returns:
            List of valid starting split positions (only start=0 or end positions)
        """
        if container_length > self.split_factor:
            # Special case for containers longer than split_factor (like FFEU=5 in split_factor=4)
            # Maximum allowed length is split_factor * 2
            assert container_length < self.split_factor*2, 'Container is not allowed to span more than 2 full bays'

            start_position = 0
            end_position = -(container_length - self.split_factor)
            return [start_position, end_position]
        
        elif container_length == self.split_factor:
            # Container uses full bay (e.g., FEU=4 in split_factor=4)
            return [0]  # Only start position possible
        
        else:
            # Container fits within bay with room to spare
            # Allow START (position 0) or END (position that aligns to end of bay)
            start_position = 0
            end_position = self.split_factor - container_length
            
            if start_position == end_position:
                # Edge case: only one position possible
                return [start_position]
            else:
                # Both start and end positions available
                return [start_position, end_position]

    def _find_cross_bay_placements(self, available_coordinates, container_type: str) -> List[Tuple]:
        """
        OPTIMIZED: Handle cross-bay containers with minimal loops.
        
        Args:
            available_coordinates: Available positions
            container_type: Container type (FFEU, etc.)
            
        Returns:
            List of valid cross-bay placements as (row, start_bay, tier, start_split) tuples
        """
        if container_type not in self.container_lengths:
            return []
            
        container_length = self.container_lengths[container_type]
        
        if container_length <= self.split_factor or container_length > self.split_factor * 2:
            return []
        
        valid_start_positions = self._get_valid_start_positions(container_length)
        
        # Convert to set for O(1) lookups
        position_set = {(row, bay, split, tier) for row, bay, split, tier in available_coordinates}
        
        # Get unique (row, tier) combinations
        row_tier_combinations = {(row, tier) for row, bay, split, tier in available_coordinates}
        
        valid_placements = []
        
        for row, tier in row_tier_combinations:
            for start_split in valid_start_positions:
                actual_start_split = start_split if start_split >= 0 else self.split_factor + start_split
                
                # Find starting bays that have the required start split available
                starting_bays = {bay for row_check, bay, split, tier_check in position_set 
                            if row_check == row and tier_check == tier and split == actual_start_split}
                
                # Check each potential starting bay
                for start_bay in starting_bays:
                    # Single loop to validate consecutive positions
                    valid = True
                    for j in range(container_length):
                        expected_bay = start_bay + (actual_start_split + j) // self.split_factor
                        expected_split = (actual_start_split + j) % self.split_factor
                        
                        if (row, expected_bay, expected_split, tier) not in position_set:
                            valid = False
                            break
                    
                    if valid:
                        valid_placements.append((row, start_bay, tier, actual_start_split))
        
        return valid_placements

    def search_insertion_position(self, bay: int, goods: str, container_type: str, max_proximity: int, coords = None)-> List:
        '''
        ENHANCED: Now returns valid starting positions for containers based on their length.
        
        Args:
            - bay : index of bay that is parallel to train_slot
            - goods : str in [r,dg,sb_t,reg] to correspond to masks
            - container_type : str in container_lengths dict (TWEU, FEU, etc.)
            - max_proximity : int of bays to left or right to be searched
            
        Returns:
            List of valid starting positions as (row, bay, tier, start_split) tuples
        '''
        # Assemble basic mask of available spaces
        match goods:
            case 'r':
                available_places = self.r_mask & self.dynamic_yard_mask
            case 'dg':
                available_places = self.dg_mask & self.dynamic_yard_mask
            case 'sb_t':
                available_places = self.sb_t_mask & self.dynamic_yard_mask  # Fixed: was using dg_mask
            case 'reg':
                available_places = self.reg_mask & self.dynamic_yard_mask
            case _:
                raise Exception('Storage Yard: invalid goods type passed in search_insertion_position()')
            
        # Assess stacking rules
        for k in self.cldymc:
            if k != container_type:
                available_places = available_places & self.cldymc[k]

        # Determine possible bays
        min_bay = max(0, bay - max_proximity)
        max_bay = min(self.n_bays, bay + max_proximity + 1)

        # Block off everything past min and max bay
        available_places[:, :min_bay*self.split_factor] = False
        available_places[:, max_bay*self.split_factor:] = False

        if coords != None:
            for coord in coords:
                r, b, s, t = coord
                available_places[r*self.n_tiers+t+1, b*self.split_factor+s] = False

        # Convert to coordinates
        available_coordinates = self.coordinates[available_places]

        if len(available_coordinates) > 0:
            # OPTIMIZATION: Use bit manipulation to find valid container placements
            valid_placements = self._find_valid_container_placements(available_coordinates, container_type)
            
            # Sort by row for consistent output
            valid_placements.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
            
            return valid_placements
            
        return []

    def get_container_coordinates_from_placement(self, placement: Tuple[int, int, int, int], container_type: str) -> List[Tuple[int, int, int, int]]:
        """
        Convert a placement tuple to full coordinate list for add_container/remove_container.
        Handles both regular and cross-bay placements.
        
        Args:
            placement: (row, bay, tier, start_split) tuple from search_insertion_position
            container_type: Container type to get length for
            
        Returns:
            List of (row, bay, split, tier) coordinates for all positions the container occupies
        """
        if container_type not in self.container_lengths:
            return []
        
        row, bay, tier, start_split = placement
        container_length = self.container_lengths[container_type]
        
        coordinates = []
        
        for i in range(container_length):
            # Calculate which bay and split this position falls into
            current_bay = bay + (start_split + i) // self.split_factor
            current_split = (start_split + i) % self.split_factor
            
            # Safety checks
            if current_bay < self.n_bays and current_split < self.split_factor:
                coordinates.append((row, current_bay, current_split, tier))
        
        return coordinates

    def return_possible_yard_moves(self, max_proximity: int = 1) -> Dict[str, Dict[str, List[List[Tuple]]]]:
        """Find all possible yard moves for accessible containers."""
        # Get all target coordinates
        target_coordinates = self.coordinates[self.dynamic_yard_mask]
        
        if len(target_coordinates) == 0:
            return {}
        
        # Pre-filter coordinates by tier to avoid checking tier == 0 in loop
        valid_coordinates = [coord for coord in target_coordinates if coord[3] > 0]
        
        if not valid_coordinates:
            return {}
        
        # Use defaultdict to eliminate membership testing
        target_containers: Dict[Container, Dict[str, List[int]]] = defaultdict(lambda: {"source_coords": [], "destinations": []})
        
        # Cache for container attribute -> mask mapping
        container_mask_cache = {}
        
        # Process each valid coordinate
        for coordinate in valid_coordinates:
            row, bay, split, tier = coordinate
            
            # CASE 1: Check container below empty space (original logic)
            if tier < self.n_tiers - 1:
                container_tier = tier - 1
                container = self.containers[row, bay, container_tier, split]
            # CASE 2: At max tier - check current position for container
            else:  # tier == self.n_tiers - 1
                container = self.containers[row, bay, tier, split]
            
            if container is not None:
                # Add coordinate (defaultdict eliminates need for existence check)
                actual_tier = tier - 1 if tier < self.n_tiers - 1 else tier
                target_containers[container]["source_coords"].append((row, bay, split, actual_tier))
                
                # Cache container mask mapping
                if container not in container_mask_cache:
                    if container.goods_type == 'Reefer':
                        container_mask_cache[container] = 'r'
                    elif container.goods_type == 'Dangerous':
                        container_mask_cache[container] = 'dg'
                    elif container.goods_type == 'Regular':
                        container_mask_cache[container] = 'reg'
                    elif container.container_type in ("Trailer", "Swap Body"):
                        container_mask_cache[container] = 'sb_t'
                    else:
                        container_mask_cache[container] = 'reg'
        
        # CASE 3: Check swap bodies/trailers using sb_t_mask
        sb_t_positions = self.coordinates[self.sb_t_mask & self.dynamic_yard_mask]
        for coordinate in sb_t_positions:
            row, bay, split, tier = coordinate
            if tier == 0:  # sb_t only on ground level
                container = self.containers[row, bay, tier, split]
                if container is not None and container.container_type in ("Trailer", "Swap Body"):
                    target_containers[container]["source_coords"].append((row, bay, split, tier))
                    container_mask_cache[container] = 'sb_t'
        
        # Process destinations using cached mask mappings
        result = {}
        for container, data in target_containers.items():
            # Get bay from first coordinate
            bay = data["source_coords"][0][1]
            
            # Use cached mask
            applicable_mask = container_mask_cache[container]
            
            # Get valid destinations
            destinations = self.search_insertion_position(
                bay, 
                applicable_mask, 
                container.container_type, 
                max_proximity,
                data["source_coords"]
            )
            
            # Only include containers that have valid destinations
            if destinations:
                result[container.container_id] = {
                    "source_coords": data["source_coords"],
                    "destinations": destinations
                }
        
        return result

# Example usage and testing
if __name__ == "__main__":
    import time
    
    start = time.time()
    yard = BooleanStorageYard(
        n_rows=5,
        n_bays=15,
        n_tiers=3,
        coordinates=[
            # Reefers on both ends
            (1, 1, "r"), (1, 2, "r"), (1, 3, "r"), (1, 4, "r"), (1, 5, "r"),
            (15, 1, "r"), (15, 2, "r"), (15, 3, "r"), (15, 4, "r"), (15, 5, "r"),
            
            # Row nearest to trucks is for swap bodies and trailers
            (1, 1, "sb_t"), (2, 1, "sb_t"), (3, 1, "sb_t"), (4, 1, "sb_t"), (5, 1, "sb_t"),
            (6, 1, "sb_t"), (7, 1, "sb_t"), (8, 1, "sb_t"), (9, 1, "sb_t"), (10, 1, "sb_t"),
            (11, 1, "sb_t"), (12, 1, "sb_t"), (13, 1, "sb_t"), (14, 1, "sb_t"), (15, 1, "sb_t"),
            
            # Pit in the middle for dangerous goods
            (7, 3, "dg"), (8, 3, "dg"), (9, 3, "dg"),
            (7, 4, "dg"), (8, 4, "dg"), (9, 4, "dg"),
            (7, 5, "dg"), (8, 5, "dg"), (9, 5, "dg"),
        ],
        split_factor=4,
        device='cuda',  # or 'cuda' if available
        validate=True
    )
    end = time.time()
    print(f"Initialization time: {end-start:.4f}s")

    # Test the optimized search
    start = time.time()
    
    # Test with TWEU (length 2)
    print("\n=== Testing TWEU (length 2) placement ===")
    valid_placements = yard.search_insertion_position(6, 'reg', 'TWEU', 3)
    print(f"Found {len(valid_placements)} valid TWEU placements")
    if valid_placements:
        print("First 5 placements:", valid_placements[:5])
        
        # Test coordinate conversion
        first_placement = valid_placements[0]
        coords = yard.get_container_coordinates_from_placement(first_placement, 'TWEU')
        print(f"Placement {first_placement} converts to coordinates: {coords}")
        print("Expected: Only start (0,1) or end (2,3) positions allowed")
    
    # Test with THEU (length 3)
    print("\n=== Testing THEU (length 3) placement ===")
    valid_placements_theu = yard.search_insertion_position(6, 'reg', 'THEU', 3)
    print(f"Found {len(valid_placements_theu)} valid THEU placements")
    if valid_placements_theu:
        print("First 5 THEU placements:", valid_placements_theu[:5])
        print("Expected: Only positions 0 (start) and 1 (end) for each bay")
    
    # Test with FEU (length 4)
    print("\n=== Testing FEU (length 4) placement ===")
    valid_placements_feu = yard.search_insertion_position(6, 'reg', 'FEU', 3)
    print(f"Found {len(valid_placements_feu)} valid FEU placements")
    if valid_placements_feu:
        print("FEU placements:", valid_placements_feu[:5])
        print("Expected: Only position 0 (full bay) for each bay")
    
    # Test with FFEU (length 5) - cross-bay container
    print("\n=== Testing FFEU (length 5) cross-bay placement ===")
    valid_placements_ffeu = yard.search_insertion_position(6, 'reg', 'FFEU', 3)
    print(f"Found {len(valid_placements_ffeu)} valid FFEU placements")
    if valid_placements_ffeu:
        print("FFEU placements:", valid_placements_ffeu[:5])
        print("Expected: Cross-bay spanning positions")
    
    end = time.time()
    print(f"\nSearch time: {end-start:.4f}s")

    # Test container operations
    print("\n=== Testing container add/remove ===")
    new_container = ContainerFactory.create_container("REG001", "TWEU", "Import", "Regular", weight=20000)
    
    start = time.time()
    if valid_placements:
        # Use the optimized coordinate conversion
        placement = valid_placements[0]
        coordinates = yard.get_container_coordinates_from_placement(placement, 'TWEU')
        print(coordinates)
        yard.add_container(new_container, coordinates)
        print(yard.return_possible_yard_moves())    
        end = time.time()
        print(f"Container add/remove time: {end-start:.4f}s")
        removed_container = yard.remove_container(coordinates)
        print(yard.dynamic_yard_mask)
        print(f"Container operations successful: {removed_container.container_id}")
    # Test container operations
    container = ContainerFactory.create_container("TEST001", "TWEU", "Import", "Regular", weight=20000)
    placement = (0, 5, 0, 0)  # row=0, bay=5, tier=0, start_split=0
    coords = yard.get_container_coordinates_from_placement(placement, 'TWEU')
    
    # Add container
    start = time.time()
    yard.add_container(container, coords)
    end = time.time()
    print(f"Container add time: {end-start:.6f}s")
    
    # Test tensor conversion
    start = time.time()
    occupied_tensor = yard.get_occupied_tensor()
    type_tensor = yard.get_container_type_tensor()
    full_state = yard.get_full_state_tensor()
    compact_state = yard.get_compact_state_tensor()
    end = time.time()
    print(f"Tensor conversion time: {end-start:.6f}s")
    
    print(f"Occupied tensor shape: {occupied_tensor.shape}")
    print(f"Type tensor shape: {type_tensor.shape}")
    print(f"Full state shape: {full_state.shape}")
    print(f"Compact state shape: {compact_state.shape}")
    
    # Test possible moves
    start = time.time()
    moves = yard.return_possible_yard_moves()
    end = time.time()
    print(f"Possible moves calculation time: {end-start:.6f}s")
    print(f"Found {len(moves)} containers with possible moves")