from typing import Dict, Tuple, List
from simulation.terminal_components.Container import Container, ContainerFactory
import numpy as np
from collections import defaultdict

class BooleanStorageYard:
    def __init__(self, 
                 n_rows: int, 
                 n_bays: int, 
                 n_tiers: int,
                 coordinates: List[Tuple[int, int, str]],
                 split_factor: int = 4,
                 validate: bool = False):

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

        # Declaring base placeability of containers
        # self.dynamic_yard_mask = ~np.zeros((n_rows*n_tiers*split_factor, n_bays), dtype=bool)
        self.dynamic_yard_mask = self.create_dynamic_yard_mask()
        '''boolean mask for whole yard, where False means that a place is occupied'''

        # Create coordinate mapping for yard
        self.coordinates = self.create_coordinate_mapping()

        print(self.dynamic_yard_mask.shape)
        # Creating masks for specific container types
        self.r_mask, self.dg_mask, self.sb_t_mask = self.extract_special_masks(coordinates)

        # AND product results in mask of available spots for regular containers
        self.reg_mask = ~np.zeros_like(self.dynamic_yard_mask, dtype=bool)
        self.reg_mask = self.reg_mask & ~self.r_mask & ~self.dg_mask & ~self.sb_t_mask

        # Test of AND between bool mask and yard coordinates
        # print(self.coordinates[self.r_mask])
 
        if validate:
            self.print_masks()

        self.containers: Dict[str, Container] = self.create_container_mapping()

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
            k:self.dynamic_yard_mask for k in self.container_lengths
        }
        '''Container-Lengths-Dynamic-Yard-Mask-Copy for each different container length'''

    def create_dynamic_yard_mask(self)->np.ndarray:
        bool_arr = np.zeros((self.n_rows*self.split_factor, self.n_bays*self.n_tiers), dtype=bool)
        for row in range(self.n_rows):
            for bay in range(self.n_bays):
                for tier in range(self.n_tiers):
                    for split in range(self.split_factor):
                        if tier == 0:
                            bool_arr[row*self.split_factor+split][bay*self.n_tiers+tier] = True
        
        return bool_arr

    def create_coordinate_mapping(self)->np.ndarray:
        ''' 0-starting coordinate map as identical mask to the dynamic_yard_mask'''
        coordinate_arr = np.zeros((self.n_rows*self.split_factor, self.n_bays*self.n_tiers), dtype=tuple)
        for row in range(self.n_rows):
            for bay in range(self.n_bays):
                for tier in range(self.n_tiers):
                    for split in range(self.split_factor):
                        coordinate_format = (row, bay, split, tier)
                        coordinate_arr[row*self.split_factor+split][bay*self.n_tiers+tier] = coordinate_format
        
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
                            # print((row, split),(bay, tier)) DEBUGGING PRINT STATEMENT
                            r_mask[row*self.split_factor+split][bay*self.n_tiers+tier] = True
                case "dg":
                    for tier in range(self.n_tiers):
                        for split in range(self.split_factor):
                            dg_mask[row*self.split_factor+split][bay*self.n_tiers+tier] = True
                case "sb_t":
                    for tier in range(self.n_tiers):
                        for split in range(self.split_factor):
                            sb_t_mask[row*self.split_factor+split][bay*self.n_tiers+tier] = True
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

        # print("Coordinate Mapping")
        # print(self.coordinates)

        print("Dinamic Yard at init")
        print(self.dynamic_yard_mask)
        print(self.dynamic_yard_mask.shape)

    def create_container_mapping(self):
        containers = {}
        for row in range(self.n_rows):
            for bay in range(self.n_bays):
                for tier in range(self.n_tiers):
                    for split in range(self.split_factor):
                        key_format = f"R{row}B{bay}.{split}T{tier}"
                        containers[key_format] = None
        return containers

    def add_container(self, container: Container, coordinates: List[Tuple[int, int, int, int]]):
        '''
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

            # place container-piecewise
            self.containers[f"R{row}B{bay}.{split}T{tier}"] = container
            self.dynamic_yard_mask[row*self.split_factor+split][bay*self.n_tiers+tier] = False
            
            # lock stack for that container type
            # [(row-1)*self.n_tiers*self.split_factor + i*self.split_factor + j][bay-1]
            self.cldymc[container.container_type][row*self.split_factor+split][(bay*self.n_tiers):(bay*self.n_tiers)+self.n_tiers-1] = False

            # unlock the next tier
            if tier < self.n_tiers - 1:
                self.dynamic_yard_mask[row*self.split_factor+split][bay*self.n_tiers+tier+1] = True

    def remove_container(self, coordinates: List[Tuple[int, int, int, int]]) -> Container:
        '''
        Args:
            - container: Object of Container class that is suposed to be placed into yard
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
                # save container
                container = self.containers[f"R{row}B{bay}.{split}T{tier}"]
                container_saved = True

            # remove container from yard
            self.containers[f"R{row}B{bay}.{split}T{tier}"] = None
            self.dynamic_yard_mask[row*self.split_factor+split][bay*self.n_tiers+tier] = True

            # unlock the stack for all container types if the removal opens a stack
            if tier == 0:
                self.cldymc[container.container_type][row*self.split_factor+split][(bay*self.n_tiers):(bay*self.n_tiers)+self.n_tiers-1] = True

            # if not max height, lock tier above
            if tier < self.n_tiers - 1:
                self.dynamic_yard_mask[row*self.split_factor+split][bay*self.n_tiers+tier+1] = False

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
            # TODO: Implement cross-bay spanning logic
            return [0]  # For now, only allow start position
        
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
        Handle containers that span across multiple bays (like FFEU=5 in split_factor=4).
        
        Args:
            available_coordinates: Available positions
            container_type: Container type (FFEU, etc.)
            
        Returns:
            List of valid cross-bay placements as (row, start_bay, tier, start_split) tuples
        """
        if container_type not in self.container_lengths:
            return []
            
        container_length = self.container_lengths[container_type]
        
        if container_length <= self.split_factor:
            return []  # Not a cross-bay container
        
        # Group coordinates by (row, tier) for cross-bay analysis
        position_groups = defaultdict(lambda: defaultdict(set))
        
        for coord in available_coordinates:
            row, bay, split, tier = coord
            position_groups[row][tier].add((bay, split))
        
        valid_placements = []
        
        for row in position_groups:
            for tier in position_groups[row]:
                bay_splits = position_groups[row][tier]
                
                # Sort by bay, then split
                sorted_positions = sorted(bay_splits)
                
                # Look for consecutive positions across bays
                # For FFEU (length=5): need 5 consecutive split positions
                # Could be: bay_n splits [2,3] + bay_(n+1) splits [0,1,2]
                
                for i in range(len(sorted_positions) - container_length + 1):
                    # Check if we have container_length consecutive positions
                    consecutive_positions = []
                    
                    current_bay, current_split = sorted_positions[i]
                    for j in range(container_length):
                        expected_bay = current_bay + (current_split + j) // self.split_factor
                        expected_split = (current_split + j) % self.split_factor
                        
                        if (expected_bay, expected_split) in bay_splits:
                            consecutive_positions.append((expected_bay, expected_split))
                        else:
                            break
                    
                    if len(consecutive_positions) == container_length:
                        # Valid cross-bay placement found
                        start_bay, start_split = consecutive_positions[0]
                        valid_placements.append((row, start_bay, tier, start_split))
        
        return valid_placements

    def search_insertion_position(self, bay: int, goods: str, container_type: str, max_proximity: int):
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
        # min_bay = bay*self.split_factor - max_proximity*self.split_factor if bay - max_proximity > 0 else 0
        # max_bay = bay*self.split_factor + max_proximity*self.split_factor if bay + max_proximity < self.n_bays*self.split_factor else self.n_bays*self.split_factor
        min_bay = max(0, bay - max_proximity)
        max_bay = min(self.n_bays, bay + max_proximity + 1)

        # Block off everything past min and max bay
        # available_places[:min_bay, :] = False
        # available_places[max_bay:, :] = False
        available_places[:, :min_bay*self.n_tiers] = False
        available_places[:, max_bay*self.n_tiers:] = False
        # print(available_places)
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

    def return_possible_yard_moves_optimized(self, max_proximity: int = 5) -> Dict[str, Dict[str, List[List[Tuple]]]]:
        """
        OPTIMIZED: Calculate all possible moves using direct mask operations.
        
        Args:
            max_proximity: Maximum bay distance to search for destinations
            
        Returns:
            Dict mapping container_id -> {
                'source_coords': List of (row, bay, split, tier) tuples for removal,
                'destinations': List of possible destination coordinate lists for addition
            }
            
        Runtime: O(mask_operations) - dramatically faster than iterative approach
        """
        possible_moves = {}
        
        # Step 1: Find all movable containers using mask operations
        # Occupied positions are where dynamic_yard_mask is False
        occupied_mask = ~self.dynamic_yard_mask
        
        # Get coordinates of all occupied positions - O(1) numpy operation
        occupied_coords = self.coordinates[occupied_mask]
        
        if len(occupied_coords) == 0:
            return possible_moves
        
        # Step 2: Identify top-tier containers (movable ones)
        movable_positions = []
        
        for coord in occupied_coords:
            row, bay, split, tier = coord
            
            # Check if this is a top container (no container above it)
            is_top = True
            if tier < self.n_tiers - 1:  # Not already at max tier
                # Check if tier above is empty
                above_tier = tier + 1
                above_idx = row * self.n_tiers * self.split_factor + above_tier * self.split_factor + split
                if above_idx < self.dynamic_yard_mask.shape[0] and bay < self.dynamic_yard_mask.shape[1]:
                    is_top = self.dynamic_yard_mask[above_idx, bay]  # True = empty = this is top
            
            if is_top:
                movable_positions.append((row, bay, split, tier))
        
        # Step 3: For each movable container, find destinations using mask operations
        for row, bay, split, tier in movable_positions:
            # Get container from position - O(1) lookup
            position_key = f"R{row}B{bay}.{split}T{tier}"
            container = self.containers.get(position_key)
            
            if container is None:
                continue
            
            # Determine appropriate mask based on container type - O(1)
            if hasattr(container, 'goods_type') and container.goods_type == 'Reefer':
                available_mask = self.r_mask & self.dynamic_yard_mask
            elif hasattr(container, 'goods_type') and container.goods_type == 'Dangerous':
                available_mask = self.dg_mask & self.dynamic_yard_mask
            elif hasattr(container, 'container_type') and container.container_type in ('Swap Body', 'Trailer'):
                available_mask = self.sb_t_mask & self.dynamic_yard_mask
            else:
                available_mask = self.reg_mask & self.dynamic_yard_mask
            
            # Apply stacking rules for container type - O(1) mask operation
            container_length = self.container_lengths.get(container.container_type, 1)
            for other_type, other_mask in self.cldymc.items():
                if other_type != container.container_type:
                    available_mask = available_mask & other_mask
            
            # Apply proximity constraint - O(1) mask operation
            min_bay = max(0, bay - max_proximity)
            max_bay = min(self.n_bays, bay + max_proximity + 1)
            
            # Create proximity mask
            proximity_mask = np.zeros_like(available_mask)
            proximity_mask[:, min_bay:max_bay] = True
            available_mask = available_mask & proximity_mask
            
            # Get available coordinates - O(1) numpy operation
            available_coords = self.coordinates[available_mask]
            
            if len(available_coords) == 0:
                continue
            
            # Calculate source coordinates for this container
            source_coords = []
            for i in range(container_length):
                current_bay = bay + (split + i) // self.split_factor
                current_split = (split + i) % self.split_factor
                if current_bay < self.n_bays and current_split < self.split_factor:
                    source_coords.append((row, current_bay, current_split, tier))
            
            # Convert available coordinates to valid placements using optimized method
            valid_placements = self._find_valid_container_placements(available_coords, container.container_type)
            
            # Convert placements to destination coordinates
            destinations = []
            for placement in valid_placements:
                dest_coords = self.get_container_coordinates_from_placement(placement, container.container_type)
                # Don't include current position as destination
                if dest_coords != source_coords:
                    destinations.append(dest_coords)
            
            # Store results only if destinations exist
            if destinations:
                possible_moves[container.container_id] = {
                    'source_coords': source_coords,
                    'destinations': destinations
                }
        
        # np.set_printoptions(linewidth=(6*self.n_rows*self.n_tiers*+5))
        # print(self.dynamic_yard_mask)
        return possible_moves

    def return_possible_yard_moves_ultra_optimized(self, max_proximity: int = 5) -> Dict[str, Dict[str, List[List[Tuple]]]]:
        """
        ULTRA OPTIMIZED: Calculate moves using pure numpy operations where possible.
        
        This version minimizes Python loops and maximizes numpy vectorization.
        """
        possible_moves = {}
        
        # Step 1: Vectorized identification of movable containers
        occupied_mask = ~self.dynamic_yard_mask
        
        # Get all occupied indices using numpy.where - much faster than coordinate iteration
        occupied_rows, occupied_cols = np.where(occupied_mask)
        
        if len(occupied_rows) == 0:
            return possible_moves
        
        # Step 2: Vectorized top-container detection
        movable_mask = np.zeros_like(self.dynamic_yard_mask, dtype=bool)
        
        for i in range(len(occupied_rows)):
            row_idx, col_idx = occupied_rows[i], occupied_cols[i]
            
            # Extract coordinates from the coordinate mapping
            coord = self.coordinates[row_idx, col_idx]
            row, bay, split, tier = coord
            
            # Check if top container
            if tier == self.n_tiers - 1:  # Already at top tier
                movable_mask[row_idx, col_idx] = True
            else:
                # Check tier above
                above_tier = tier + 1
                above_idx = row * self.n_tiers * self.split_factor + above_tier * self.split_factor + split
                if above_idx < self.dynamic_yard_mask.shape[0]:
                    if self.dynamic_yard_mask[above_idx, bay]:  # Empty above
                        movable_mask[row_idx, col_idx] = True
        
        # Get movable coordinates
        movable_coords = self.coordinates[movable_mask]
        
        # Step 3: Process each movable container
        for coord in movable_coords:
            row, bay, split, tier = coord
            
            # Direct container lookup
            position_key = f"R{row}B{bay}.{split}T{tier}"
            container = self.containers.get(position_key)
            
            if container is None:
                continue
            
            # Fast mask selection using dictionary lookup
            mask_selector = {
                ('Reefer',): self.r_mask,
                ('Dangerous',): self.dg_mask,
                ('Swap Body', 'Trailer'): self.sb_t_mask
            }
            
            selected_mask = self.reg_mask  # default
            goods_type = getattr(container, 'goods_type', None)
            container_type = getattr(container, 'container_type', None)
            
            for key, mask in mask_selector.items():
                if goods_type in key or container_type in key:
                    selected_mask = mask
                    break
            
            # Vectorized available positions calculation
            available_mask = selected_mask & self.dynamic_yard_mask
            
            # Apply stacking rules - vectorized
            container_type = getattr(container, 'container_type', 'TWEU')
            if container_type in self.cldymc:
                for other_type, other_mask in self.cldymc.items():
                    if other_type != container_type:
                        available_mask = available_mask & other_mask
            
            # Apply proximity constraint using array slicing
            min_bay = max(0, bay - max_proximity)
            max_bay = min(self.n_bays, bay + max_proximity + 1)
            
            # Zero out everything outside proximity range
            proximity_constrained = np.zeros_like(available_mask)
            proximity_constrained[:, min_bay:max_bay] = available_mask[:, min_bay:max_bay]
            
            # Get available coordinates
            available_coords = self.coordinates[proximity_constrained]
            
            if len(available_coords) == 0:
                continue
            
            # Use existing optimized placement finding
            container_length = self.container_lengths.get(container_type, 1)
            valid_placements = self._find_valid_container_placements(available_coords, container_type)
            
            if not valid_placements:
                continue
            
            # Calculate source coordinates
            source_coords = []
            for i in range(container_length):
                current_bay = bay + (split + i) // self.split_factor
                current_split = (split + i) % self.split_factor
                if current_bay < self.n_bays and current_split < self.split_factor:
                    source_coords.append((row, current_bay, current_split, tier))
            
            # Convert to destinations
            destinations = []
            for placement in valid_placements:
                dest_coords = self.get_container_coordinates_from_placement(placement, container_type)
                if dest_coords != source_coords:
                    destinations.append(dest_coords)
            
            if destinations:
                possible_moves[container.container_id] = {
                    'source_coords': source_coords,
                    'destinations': destinations
                }
        
        return possible_moves

if __name__ == "__main__":
    import time
    start = time.time()
    new_yard = BooleanStorageYard(
        n_rows=5,
        n_bays=15,
        n_tiers=3,
        # coordinates are in form (bay, row, type = r,dg,sb_t)
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
        validate=True
    )
    end = time.time()
    print(f"Initialization time: {end-start:.4f}s")

    # Test the optimized search
    start = time.time()
    
    # Test with TWEU (length 2)
    print("\n=== Testing TWEU (length 2) placement ===")
    valid_placements = new_yard.search_insertion_position(6, 'reg', 'TWEU', 3)
    print(f"Found {len(valid_placements)} valid TWEU placements")
    if valid_placements:
        print("First 5 placements:", valid_placements[:5])
        
        # Test coordinate conversion
        first_placement = valid_placements[0]
        coords = new_yard.get_container_coordinates_from_placement(first_placement, 'TWEU')
        print(f"Placement {first_placement} converts to coordinates: {coords}")
        print("Expected: Only start (0,1) or end (2,3) positions allowed")
    
    # Test with THEU (length 3)
    print("\n=== Testing THEU (length 3) placement ===")
    valid_placements_theu = new_yard.search_insertion_position(6, 'reg', 'THEU', 3)
    print(f"Found {len(valid_placements_theu)} valid THEU placements")
    if valid_placements_theu:
        print("First 5 THEU placements:", valid_placements_theu[:5])
        print("Expected: Only positions 0 (start) and 1 (end) for each bay")
    
    # Test with FEU (length 4)
    print("\n=== Testing FEU (length 4) placement ===")
    valid_placements_feu = new_yard.search_insertion_position(6, 'reg', 'FEU', 3)
    print(f"Found {len(valid_placements_feu)} valid FEU placements")
    if valid_placements_feu:
        print("FEU placements:", valid_placements_feu[:5])
        print("Expected: Only position 0 (full bay) for each bay")
    
    # Test with FFEU (length 5) - cross-bay container
    print("\n=== Testing FFEU (length 5) cross-bay placement ===")
    valid_placements_ffeu = new_yard.search_insertion_position(6, 'reg', 'FFEU', 3)
    print(f"Found {len(valid_placements_ffeu)} valid FFEU placements")
    if valid_placements_ffeu:
        print("FFEU placements:", valid_placements_ffeu[:3])
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
        coordinates = new_yard.get_container_coordinates_from_placement(placement, 'TWEU')
        print(coordinates)
        new_yard.add_container(new_container, coordinates)
        print(new_yard.return_possible_yard_moves_ultra_optimized())    
        end = time.time()
        print(f"Container add/remove time: {end-start:.4f}s")
        removed_container = new_yard.remove_container(coordinates)
        # print(new_yard.dynamic_yard_mask)
        # print(f"Container operations successful: {removed_container.container_id}")
