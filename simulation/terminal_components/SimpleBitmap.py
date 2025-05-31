from typing import Dict, Tuple, List
from simulation.terminal_components.Container import Container, ContainerFactory
import numpy as np

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

        # Creating masks for specific container types
        self.r_mask, self.dg_mask, self.sb_t_mask = self.extract_special_masks(coordinates)

        # AND product results in mask of available spots for regular containers
        self.reg_mask = self.dynamic_yard_mask & ~self.r_mask & ~self.dg_mask & ~self.sb_t_mask

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
        bool_arr = np.zeros((self.n_rows*self.n_tiers*self.split_factor, self.n_bays), dtype=bool)
        for row in range(self.n_rows):
            for bay in range(self.n_bays):
                for tier in range(self.n_tiers):
                    for split in range(self.split_factor):
                        if tier == 0:
                            bool_arr[(bay)*self.split_factor+split][(row)*self.n_tiers+tier] = True
        
        return bool_arr

    def create_coordinate_mapping(self)->np.ndarray:
        coordinate_arr = np.zeros((self.n_rows*self.n_tiers*self.split_factor, self.n_bays), dtype=tuple)
        for row in range(self.n_rows):
            for bay in range(self.n_bays):
                for tier in range(self.n_tiers):
                    for split in range(self.split_factor):
                        coordinate_format = (row, bay, split, tier)
                        coordinate_arr[(bay)*self.split_factor+split][(row)*self.n_tiers+tier] = coordinate_format
        
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
        '''
        r_mask = ~self.dynamic_yard_mask
        dg_mask = ~self.dynamic_yard_mask
        sb_t_mask = ~self.dynamic_yard_mask
        for coordinate in coordinates:
            bay, row, class_type = coordinate
            match class_type:
                case "r":
                    for i in range(self.n_tiers):
                        for j in range(self.split_factor):
                            r_mask[(bay-1)*self.split_factor+j][(row-1)*self.n_tiers+i] = True
                case "dg":
                    for i in range(self.n_tiers):
                        for j in range(self.split_factor):
                            dg_mask[(bay-1)*self.split_factor+j][(row-1)*self.n_tiers+i] = True
                case "sb_t":
                    for i in range(self.n_tiers):
                        for j in range(self.split_factor):
                            sb_t_mask[(bay-1)*self.split_factor+j][(row-1)*self.n_tiers+i] = True
                case _:
                    raise Exception("Storage Yard Class: invalid coordinates passed") 
        return (r_mask, dg_mask, sb_t_mask)

    def print_masks(self) -> None:
        'Prints all masks for visual verification of corectness'
                
        # Setting print options to be able to see masks nicely
        np.set_printoptions(linewidth=(6*self.n_rows*self.n_tiers*+5))

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
            row, bay, sub_bay, tier = coordinate

            # place container-piecewise
            self.containers[f"R{row}B{bay}.{sub_bay}T{tier}"] = container
            self.dynamic_yard_mask[(bay)*self.split_factor+sub_bay][(row*self.n_tiers)+tier] = False
            
            # lock stack for that container type
            self.cldymc[container.container_type][(bay)*self.split_factor+sub_bay][(row*self.n_tiers):(row*self.n_tiers)+self.n_tiers-1] = False

            # unlock the next tier
            if tier < self.n_tiers:
                self.dynamic_yard_mask[(bay)*self.split_factor+sub_bay][(row*self.n_tiers)+tier+1] = True

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
            row, bay, sub_bay, tier = coordinate
            
            if not container_saved:
                # save container
                container = self.containers[f"R{row}B{bay}.{sub_bay}T{tier}"]
                container_saved = True

            # remove container from yard
            self.containers[f"R{row}B{bay}.{sub_bay}T{tier}"] = None
            self.dynamic_yard_mask[(bay)*self.split_factor+sub_bay][(row*self.n_tiers)+tier] = True

            # unlock the stack for all container types
            self.cldymc[container.container_type][(bay)*self.split_factor+sub_bay][(row*self.n_tiers):(row*self.n_tiers)+self.n_tiers-1] = True

            # if not max height, lock tier above
            if tier < self.n_tiers:
                self.dynamic_yard_mask[(bay)*self.split_factor+sub_bay][(row*self.n_tiers)+tier+1] = False

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

    def search_insertion_position(self, bay: int, goods: str, container_type: str, max_proximity: int):
        '''
        Args:
            - bay : index of bay that is parallel to train_slot
            - container_type : str in [r,dg,sb_t,reg] to correspond to masks
            - container_length :  str in self.container_length_masks
            - max_proximity : int of bays to left or right to be searched
        '''
        # Assemble basic mask of available spaces
        match goods:
            case 'r':
                available_places = self.r_mask & self.dynamic_yard_mask
            case 'dg':
                available_places = self.dg_mask & self.dynamic_yard_mask
            case 'sb_t':
                available_places = self.dg_mask & self.dynamic_yard_mask
            case 'reg':
                available_places = self.reg_mask & self.dynamic_yard_mask
            case _:
                raise Exception('Storage Yard: invalid container_type passed in search_insertion_position()')
            
        # Assess stacking rules
        for k in self.cldymc:
            if k != container_type:
                available_places = available_places & self.cldymc[k]

        # Determine possible bays
        min_bay = bay*self.split_factor - max_proximity*self.split_factor if bay - max_proximity > 0 else 0
        max_bay = bay*self.split_factor + max_proximity*self.split_factor if bay + max_proximity < self.n_bays else self.n_bays*self.split_factor

        # np.set_printoptions(linewidth=(6*self.n_rows*self.n_tiers*+5))
        np.set_printoptions(linewidth=600)
        print(available_places,'\n\n')

        # Block off everything past min and max bay
        available_places[:min_bay, :] = False
        available_places[max_bay:, :] = False

        # TODO: find a way to account for stacking rules i.e. no 20 on 40foot or vice versa


        # Convert to coordinates
        available_coordinates = self.coordinates[available_places]

        if len(available_coordinates) > 0:
            grouped_coordinates = sorted(available_coordinates, key=lambda x: x[0])
            print(grouped_coordinates)
            return grouped_coordinates
            
        # With this we have calculated all possible movements for container
        return available_coordinates


if __name__ == "__main__":
    import time
    start= time.time()
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
            (1, 1, "sb_t"),
            (2, 1, "sb_t"),
            (3, 1, "sb_t"),
            (4, 1, "sb_t"),
            (5, 1, "sb_t"),
            (6, 1, "sb_t"),
            (7, 1, "sb_t"),
            (8, 1, "sb_t"),
            (9, 1, "sb_t"),
            (10, 1, "sb_t"),
            (11, 1, "sb_t"),
            (12, 1, "sb_t"),
            (13, 1, "sb_t"),
            (14, 1, "sb_t"),
            (15, 1, "sb_t"),
            
            # Pit in the middle for dangerous goods
            (7, 3, "dg"), (8, 3, "dg"), (9, 3, "dg"),
            (7, 4, "dg"), (8, 4, "dg"), (9, 4, "dg"),
            (7, 5, "dg"), (8, 5, "dg"), (9, 5, "dg"),
        ],
        split_factor=4,
        validate= False
    )

    new_container = ContainerFactory.create_container("REG001", "TWEU", "Import", "Regular", weight=20000)
    coordinates =[
        (1, 1, 0, 0),
        (1, 1, 1, 0)
    ]
    end=time.time()
    print(end-start)

    start=time.time()
    new_yard.add_container(new_container, coordinates)
    new_yard.remove_container(coordinates)
    end=time.time()
    print(end-start)

    from datetime import timedelta
    start=time.time()
    new_yard.search_insertion_position(6, 'reg', 'xyz', 3)
    end= time.time()
    print(end-start)

    # for now both functions are at <1ms