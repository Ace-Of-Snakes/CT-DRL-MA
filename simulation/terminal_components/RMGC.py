from typing import Dict, Tuple, List, Optional, Set
from simulation.terminal_components.Container import Container, ContainerFactory
from simulation.terminal_components.BooleanStorage import BooleanStorageYard
from simulation.terminal_components.BooleanLogistics import BooleanLogistics
from simulation.terminal_components.Truck import Truck
from simulation.terminal_components.Wagon import Wagon
from simulation.terminal_components.Train import Train
import numpy as np
import torch
from collections import defaultdict
import math

class RMGC_Controller:
    """
    Optimized Rail Mounted Gantry Crane Controller for container terminal operations.
    Manages multiple crane heads with collision avoidance and physics-based movement.
    """
    
    def __init__(
            self,
            yard: BooleanStorageYard,
            logistics: BooleanLogistics,
            # Railroad stats
            rail_slot_length=12.192,
            rail_track_width=3.0,
            space_between_rail_tracks=1.5,
            # Parking stats
            parking_width=4.0,
            driving_lane_width=4.0,
            # Distances between
            space_rails_to_parking=5.0,
            space_driving_to_storage=2.0,
            # Yard stats
            storage_slot_length=12.192,
            storage_slot_width=2.44,
            # Order
            order=['rails', 'parking', 'driving lane', 'storage'],
            # RMGC Count
            heads=2,
            # Crane data from Liebherr Specs
            trolley_speed=70.0,  # m/min
            hoisting_speed=28.0,  # m/min with load
            gantry_speed=130.0,   # m/min
            trolley_acceleration=0.3,  # m/s²
            hoisting_acceleration=0.2,  # m/s²
            gantry_acceleration=0.1,   # m/s²
            max_height=20.0,  # meters
            ground_vehicle_height=1.5,  # meters
            ):
        
        self.yard = yard
        self.logistics = logistics
        
        # Terminal dimensions
        self.rail_slot_length = rail_slot_length
        self.rail_track_width = rail_track_width
        self.space_between_rail_tracks = space_between_rail_tracks
        self.parking_width = parking_width
        self.driving_lane_width = driving_lane_width
        self.space_rails_to_parking = space_rails_to_parking
        self.space_driving_to_storage = space_driving_to_storage
        self.storage_slot_length = storage_slot_length
        self.storage_slot_width = storage_slot_width
        self.order = order
        
        # Crane specifications (convert speeds to m/s)
        self.trolley_speed = trolley_speed / 60.0  # m/s
        self.hoisting_speed = hoisting_speed / 60.0  # m/s
        self.gantry_speed = gantry_speed / 60.0  # m/s
        self.trolley_acceleration = trolley_acceleration
        self.hoisting_acceleration = hoisting_acceleration
        self.gantry_acceleration = gantry_acceleration
        self.max_height = max_height
        self.ground_vehicle_height = ground_vehicle_height
        
        # Crane heads management
        self.heads = heads
        self.crane_heads = [
            {
                'id': i,
                'position': np.array([0.0, 0.0, self.max_height]),  # (x, y, z)
                'busy': False,
                'current_move': None,
                'working_bays': set(),  # Bay indices this head is working in
                'completion_time': 0.0
            }
            for i in range(heads)
        ]
        
        # OPTIMIZATION: Pre-computed position mappings for O(1) lookup
        self.position_to_coords = {}  # Maps position string -> (x, y) meters
        self.position_to_idx = {}     # Maps position string -> matrix index
        self.idx_to_position = {}     # Reverse mapping
        
        # Build coordinate system and distance matrix
        self._build_coordinate_system()
        self.distance_matrix = self._build_distance_matrix()
        
        # Divide yard between crane heads
        self._split_yard_work()
        
        # Cache for physics calculations
        self.physics_cache = {}
    
    def _build_coordinate_system(self):
        """Build unified coordinate system for all terminal positions."""
        idx = 0
        x_offset = 0.0
        
        # Add virtual stack position
        self.position_to_coords["stack_0_0_0"] = (x_offset + 100, 0, 0)  # Far position
        self.position_to_idx["stack_0_0_0"] = idx
        self.idx_to_position[idx] = "stack_0_0_0"
        self.num_positions = idx + 1

        # Process areas in order
        for area in self.order:
            if area == 'rails':
                # Rails area
                for track in range(self.logistics.n_railtracks):
                    y_pos = track * (self.rail_track_width + self.space_between_rail_tracks)
                    
                    for row in range(self.logistics.n_rows):
                        for split in range(self.logistics.split_factor):
                            # Calculate position
                            x_pos = x_offset + (row + split/self.logistics.split_factor) * self.rail_slot_length
                            
                            # Create position identifier
                            pos_str = f"rail_{track}_{row}_{split}"
                            
                            # Store mappings
                            self.position_to_coords[pos_str] = (x_pos, y_pos)
                            self.position_to_idx[pos_str] = idx
                            self.idx_to_position[idx] = pos_str
                            idx += 1
                
                x_offset += self.logistics.n_rows * self.rail_slot_length + self.space_rails_to_parking
            
            elif area == 'parking':
                # Parking area (single lane) - aligned with rail positions
                y_pos = self.logistics.n_railtracks * (self.rail_track_width + self.space_between_rail_tracks) + self.space_rails_to_parking
                
                for row in range(self.logistics.n_rows):
                    for split in range(self.logistics.split_factor):
                        x_pos = (row + split/self.logistics.split_factor) * self.rail_slot_length
                        
                        pos_str = f"parking_{row}_{split}"
                        
                        self.position_to_coords[pos_str] = (x_pos, y_pos)
                        self.position_to_idx[pos_str] = idx
                        self.idx_to_position[idx] = pos_str
                        idx += 1
                
                x_offset = self.logistics.n_rows * self.rail_slot_length + self.space_rails_to_parking + self.parking_width
            
            elif area == 'driving lane':
                # Just add offset, no positions here
                x_offset += self.driving_lane_width + self.space_driving_to_storage
            
            elif area == 'storage':
                # Storage yard area
                for row in range(self.yard.n_rows):
                    y_pos = row * self.storage_slot_width
                    
                    for bay in range(self.yard.n_bays):
                        for tier in range(self.yard.n_tiers):
                            for split in range(self.yard.split_factor):
                                x_pos = x_offset + (bay + split/self.yard.split_factor) * self.storage_slot_length
                                
                                # Add height component for tier
                                z_pos = tier * 2.59  # Standard container height
                                
                                pos_str = f"yard_{row}_{bay}_{tier}_{split}"
                                
                                # For yard, store 3D coords
                                self.position_to_coords[pos_str] = (x_pos, y_pos, z_pos)
                                self.position_to_idx[pos_str] = idx
                                self.idx_to_position[idx] = pos_str
                                idx += 1
        
        self.num_positions = idx
    
    def _build_distance_matrix(self) -> np.ndarray:
        """
        Build optimized distance matrix for O(1) lookup.
        Uses Manhattan distance for crane movements (X-gantry, Y-trolley, Z-hoist).
        """
        # Pre-allocate matrix
        n = self.num_positions
        matrix = np.zeros((n, n), dtype=np.float32)
        
        # Compute distances
        for i in range(n):
            pos1_str = self.idx_to_position[i]
            coords1 = self.position_to_coords[pos1_str]
            
            for j in range(i+1, n):  # Symmetric matrix
                pos2_str = self.idx_to_position[j]
                coords2 = self.position_to_coords[pos2_str]
                
                # Manhattan distance (crane can't move diagonally)
                if len(coords1) == 2 and len(coords2) == 2:
                    # 2D positions (rail/parking)
                    dist = abs(coords1[0] - coords2[0]) + abs(coords1[1] - coords2[1])
                elif len(coords1) == 3 and len(coords2) == 3:
                    # 3D positions (yard)
                    dist = abs(coords1[0] - coords2[0]) + abs(coords1[1] - coords2[1]) + abs(coords1[2] - coords2[2])
                else:
                    # Mixed 2D/3D - assume ground level for 2D
                    c1 = coords1 if len(coords1) == 3 else (coords1[0], coords1[1], self.ground_vehicle_height)
                    c2 = coords2 if len(coords2) == 3 else (coords2[0], coords2[1], self.ground_vehicle_height)
                    dist = abs(c1[0] - c2[0]) + abs(c1[1] - c2[1]) + abs(c1[2] - c2[2])
                
                matrix[i, j] = dist
                matrix[j, i] = dist  # Symmetric
        
        return matrix
    
    def _split_yard_work(self):
        """Divide yard into working zones for each crane head with overlap."""
        if self.heads == 1:
            # Single head covers entire yard
            self.crane_zones = [(0, self.yard.n_bays)]
        else:
            # Multiple heads - divide with overlap
            bays_per_head = self.yard.n_bays // self.heads
            overlap = max(2, bays_per_head // 4)  # 25% overlap or at least 2 bays
            
            self.crane_zones = []
            for i in range(self.heads):
                start = i * bays_per_head
                end = min((i + 1) * bays_per_head + overlap, self.yard.n_bays)
                self.crane_zones.append((start, end))
    
    def _get_bays_from_position(self, pos_str: str) -> Set[int]:
        """Extract bay indices from position string."""
        bays = set()
        
        if pos_str.startswith('yard_'):
            # yard_row_bay_tier_split
            parts = pos_str.split('_')
            bay = int(parts[2])
            bays.add(bay)
        elif pos_str.startswith('rail_'):
            # Map rail position to approximate bay
            parts = pos_str.split('_')
            row = int(parts[2])
            # Approximate bay based on row position
            bay = (row * self.logistics.split_factor) // self.yard.split_factor
            bays.add(bay)
        elif pos_str.startswith('parking_'):
            # Similar mapping for parking
            parts = pos_str.split('_')
            row = int(parts[1])
            bay = (row * self.logistics.split_factor) // self.yard.split_factor
            bays.add(bay)
        
        return bays
    
    def _calculate_movement_time(self, pos1: str, pos2: str, head_pos: np.ndarray) -> float:
        """
        Calculate time for crane movement with acceleration/deceleration.
        Uses physics equations: t = 2*sqrt(d/a) for acceleration-limited moves.
        """
        # Get coordinates
        coords1 = np.array(self.position_to_coords[pos1])
        coords2 = np.array(self.position_to_coords[pos2])
        
        # Ensure 3D coordinates
        if len(coords1) == 2:
            coords1 = np.append(coords1, self.ground_vehicle_height)
        if len(coords2) == 2:
            coords2 = np.append(coords2, self.ground_vehicle_height)
        
        # Movement sequence: head -> pos1 -> lift -> pos2 -> lower
        total_time = 0.0
        
        # 1. Move to pickup position
        dx = abs(head_pos[0] - coords1[0])
        dy = abs(head_pos[1] - coords1[1])
        dz = abs(head_pos[2] - coords1[2])
        
        # Time for each axis (considering acceleration)
        t_gantry = self._axis_time(dx, self.gantry_speed, self.gantry_acceleration)
        t_trolley = self._axis_time(dy, self.trolley_speed, self.trolley_acceleration)
        t_hoist_down = self._axis_time(dz, self.hoisting_speed, self.hoisting_acceleration)
        
        # Parallel movement (max of all axes)
        total_time += max(t_gantry, t_trolley, t_hoist_down)
        
        # 2. Lift container to safe height
        lift_height = self.max_height - coords1[2]
        t_lift = self._axis_time(lift_height, self.hoisting_speed, self.hoisting_acceleration)
        total_time += t_lift
        
        # 3. Move to destination
        dx = abs(coords1[0] - coords2[0])
        dy = abs(coords1[1] - coords2[1])
        
        t_gantry = self._axis_time(dx, self.gantry_speed, self.gantry_acceleration)
        t_trolley = self._axis_time(dy, self.trolley_speed, self.trolley_acceleration)
        
        total_time += max(t_gantry, t_trolley)
        
        # 4. Lower container
        lower_height = self.max_height - coords2[2]
        t_lower = self._axis_time(lower_height, self.hoisting_speed, self.hoisting_acceleration)
        total_time += t_lower
        
        # Add fixed handling time
        total_time += 30.0  # 30 seconds for pickup/release operations
        
        return total_time
    
    def _axis_time(self, distance: float, max_speed: float, acceleration: float) -> float:
        """Calculate time for single axis movement with acceleration."""
        if distance == 0:
            return 0.0
        
        # Time to reach max speed
        t_accel = max_speed / acceleration
        # Distance covered during acceleration
        d_accel = 0.5 * acceleration * t_accel * t_accel
        
        if distance <= 2 * d_accel:
            # Acceleration-limited (triangular profile)
            return 2 * math.sqrt(distance / acceleration)
        else:
            # Trapezoidal profile
            d_constant = distance - 2 * d_accel
            t_constant = d_constant / max_speed
            return 2 * t_accel + t_constant
    
    def lock_head(self, head_id: int, move: Dict):
        """Lock crane head for a move."""
        if 0 <= head_id < self.heads:
            self.crane_heads[head_id]['busy'] = True
            self.crane_heads[head_id]['current_move'] = move
            
            # Update working bays
            bays = set()
            bays.update(self._get_bays_from_position(self._position_to_string(move['source_pos'])))
            bays.update(self._get_bays_from_position(self._position_to_string(move['dest_pos'])))
            self.crane_heads[head_id]['working_bays'] = bays
    
    def unlock_head(self, head_id: int):
        """Unlock crane head after move completion."""
        if 0 <= head_id < self.heads:
            self.crane_heads[head_id]['busy'] = False
            self.crane_heads[head_id]['current_move'] = None
            self.crane_heads[head_id]['working_bays'].clear()

    def _position_to_string(self, pos) -> str:
        """Convert position from move format to string format."""
        if isinstance(pos, tuple):
            if len(pos) == 2:  # Rail position (railtrack_id, wagon_idx)
                # Calculate approximate row based on wagon position
                railtrack_id, wagon_idx = pos
                # Each wagon spans wagon_length positions
                position_offset = wagon_idx * self.logistics.wagon_length + self.logistics.train_head_length
                row = position_offset // self.logistics.split_factor
                split = position_offset % self.logistics.split_factor
                return f"rail_{railtrack_id}_{row}_{split}"
            elif len(pos) == 3:  # Parking position (row, railtrack, split)
                return f"parking_{pos[0]}_{pos[2]}"
            elif len(pos) == 4:  # Yard position (row, bay, tier, split)
                return f"yard_{pos[0]}_{pos[1]}_{pos[2]}_{pos[3]}"
        elif isinstance(pos, list) and len(pos) > 0:
            # List of coordinates - use first one
            first_coord = pos[0]
            if len(first_coord) == 4:
                return f"yard_{first_coord[0]}_{first_coord[1]}_{first_coord[2]}_{first_coord[3]}"
        elif pos is None:
            # Virtual position for stack moves
            return "stack_0_0_0"
        
        # Unknown position format
        return str(pos)
    
    def mask_moves(self, moves: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Filter moves based on available crane heads and working zones.
        O(n) where n is number of moves.
        """
        if not moves:
            return {}
        
        # Find available heads
        available_heads = []
        busy_bays = set()
        
        for i, head in enumerate(self.crane_heads):
            if not head['busy']:
                available_heads.append(i)
            else:
                busy_bays.update(head['working_bays'])
        
        if not available_heads:
            return {}  # No heads available
        
        # If all heads are free, return all moves
        if len(available_heads) == self.heads:
            return moves
        
        # Filter moves that don't conflict with busy heads
        eligible_moves = {}
        
        for move_id, move in moves.items():
            # Get bays involved in this move
            move_bays = set()
            move_bays.update(self._get_bays_from_position(self._position_to_string(move['source_pos'])))
            move_bays.update(self._get_bays_from_position(self._position_to_string(move['dest_pos'])))
            
            # Check if move conflicts with busy bays
            if not move_bays.intersection(busy_bays):
                # Check if any available head can reach these bays
                for head_id in available_heads:
                    zone_start, zone_end = self.crane_zones[head_id]
                    if any(zone_start <= bay < zone_end for bay in move_bays):
                        eligible_moves[move_id] = move
                        break
        
        return eligible_moves
    
    def execute_move(self, move: Dict) -> Tuple[float, float]:
        """
        Execute container move and return (distance, time).
        Handles container removal/addition using Boolean storage/logistics methods.
        """
        source_type = move['source_type']
        dest_type = move['dest_type']
        
        # Find best available crane head
        head_id = self._select_best_head(move)
        if head_id is None:
            return 0.0, 0.0  # No head available
        
        # Lock the head
        self.lock_head(head_id, move)
        
        # Get current head position
        head_pos = self.crane_heads[head_id]['position'].copy()
        
        # Convert positions to strings for lookup
        source_str = self._position_to_string(move['source_pos'])
        dest_str = self._position_to_string(move['dest_pos'])
        
        # Remove the debug print statements after conversion
        
        # Execute the container transfer
        container = None
        
        # Remove from source
        if source_type == 'yard':
            # Parse yard position
            if isinstance(move['source_pos'], tuple) and len(move['source_pos']) == 4:
                coords = [move['source_pos']]  # Single coordinate
            else:
                coords = move['source_pos']  # Already a list
            container = self.logistics.remove_container_from_yard(coords)
        
        elif source_type == 'train':
            railtrack_id, wagon_idx = move['source_pos']
            train = self.logistics.active_trains.get(railtrack_id)
            if train and wagon_idx < len(train.wagons):
                wagon = train.wagons[wagon_idx]
                if wagon.containers:
                    # Get the specific container if specified
                    container_id = move.get('container_id')
                    if container_id:
                        container = wagon.remove_container(container_id)
                    else:
                        container = wagon.remove_container(wagon.containers[0].container_id)
                    # Update logistics lookups
                    if container:
                        self.logistics._update_train_lookups(train, railtrack_id)
        
        elif source_type == 'truck':
            truck_pos = move['source_pos']
            truck = self.logistics.active_trucks.get(truck_pos)
            if truck and truck.containers:
                # Get the specific container if specified
                container_id = move.get('container_id')
                if container_id:
                    container = truck.remove_container(container_id)
                else:
                    container = truck.remove_container(truck.containers[0].container_id)
                # Update logistics lookups
                if container:
                    self.logistics._update_truck_lookups(truck, truck_pos)
        
        # Add to destination
        success = False
        if container:
            if dest_type == 'yard':
                # Find placement coordinates
                coords = self._get_yard_placement_coords(container, move['dest_pos'])
                if coords:
                    success = self.logistics.add_container_to_yard(container, coords)
            
            elif dest_type == 'train':
                railtrack_id, wagon_idx = move['dest_pos']
                train = self.logistics.active_trains.get(railtrack_id)
                if train and wagon_idx < len(train.wagons):
                    success = train.wagons[wagon_idx].add_container(container)
                    if success:
                        # Update logistics lookups
                        self.logistics._update_train_lookups(train, railtrack_id)
                        # Remove from pickup requests if it was one
                        if container.container_id in train.wagons[wagon_idx].pickup_container_ids:
                            train.wagons[wagon_idx].remove_pickup_container(container.container_id)
            
            elif dest_type == 'truck':
                truck_pos = move['dest_pos']
                truck = self.logistics.active_trucks.get(truck_pos)
                if truck:
                    success = truck.add_container(container)
                    if success:
                        # Update logistics lookups
                        self.logistics._update_truck_lookups(truck, truck_pos)
                        # Remove from pickup requests if it was one
                        if hasattr(truck, 'pickup_container_ids') and container.container_id in truck.pickup_container_ids:
                            truck.remove_pickup_container_id(container.container_id)
        

            elif dest_type == 'stack':
                # Terminal truck move - remove from yard
                success = True
                # Container is removed from terminal system
                # Update terminal truck state in logistics
                truck_id = self.logistics.get_available_terminal_truck()
                if truck_id is not None:
                    self.logistics.terminal_trucks[truck_id]['busy'] = True
                    self.logistics.terminal_trucks[truck_id]['completion_time'] = 300.0  # 5 minutes


        # Calculate physics
        if success:
            # Get distance from matrix - with error handling
            idx1 = self.position_to_idx.get(source_str, None)
            idx2 = self.position_to_idx.get(dest_str, None)
            
            if idx1 is not None and idx2 is not None:
                distance = self.distance_matrix[idx1, idx2]
            else:
                
                # Fallback: calculate distance directly
                coords1 = self.position_to_coords.get(source_str)
                coords2 = self.position_to_coords.get(dest_str)
                
                if coords1 and coords2:
                    # Ensure 3D coordinates
                    c1 = coords1 if len(coords1) == 3 else (coords1[0], coords1[1], self.ground_vehicle_height)
                    c2 = coords2 if len(coords2) == 3 else (coords2[0], coords2[1], self.ground_vehicle_height)
                    distance = abs(c1[0] - c2[0]) + abs(c1[1] - c2[1]) + abs(c1[2] - c2[2])
                else:
                    # Last resort: estimate distance
                    distance = 50.0  # Default distance
            
            # Calculate time
            if source_str in self.position_to_coords and dest_str in self.position_to_coords:
                time = self._calculate_movement_time(source_str, dest_str, head_pos)
            else:
                # Estimate time based on distance
                time = distance / self.gantry_speed + 60.0  # Movement + handling time
            
            # Update head position
            new_coords = self.position_to_coords[dest_str]
            if len(new_coords) == 2:
                self.crane_heads[head_id]['position'] = np.array([new_coords[0], new_coords[1], self.max_height])
            else:
                self.crane_heads[head_id]['position'] = np.array(new_coords)
            
            # Set completion time
            self.crane_heads[head_id]['completion_time'] = time
            
            # Schedule unlock (in real system, this would be event-driven)
            # For now, caller should unlock after time expires
            
            return distance, time
        
        # Failed move
        self.unlock_head(head_id)
        return 0.0, 0.0
    
    def _select_best_head(self, move: Dict) -> Optional[int]:
        """Select best available crane head for move."""
        move_bays = set()
        move_bays.update(self._get_bays_from_position(self._position_to_string(move['source_pos'])))
        move_bays.update(self._get_bays_from_position(self._position_to_string(move['dest_pos'])))
        
        best_head = None
        min_distance = float('inf')
        
        for i, head in enumerate(self.crane_heads):
            if not head['busy']:
                # Check if head can reach the bays
                zone_start, zone_end = self.crane_zones[i]
                if any(zone_start <= bay < zone_end for bay in move_bays):
                    # Calculate distance from current position
                    source_str = self._position_to_string(move['source_pos'])
                    if source_str in self.position_to_coords:
                        coords = self.position_to_coords[source_str]
                        coords_3d = coords if len(coords) == 3 else (coords[0], coords[1], self.ground_vehicle_height)
                        dist = np.sum(np.abs(head['position'] - coords_3d))
                        
                        if dist < min_distance:
                            min_distance = dist
                            best_head = i
        
        return best_head
    
    def _get_yard_placement_coords(self, container: Container, dest_pos) -> List[Tuple]:
        """Get yard coordinates for container placement."""
        if isinstance(dest_pos, tuple):
            if len(dest_pos) == 4:
                # Single placement position (row, bay, tier, start_split)
                placement = dest_pos
                return self.yard.get_container_coordinates_from_placement(placement, container.container_type)
            elif len(dest_pos) == 3:
                # Might be missing the split, assume 0
                placement = (dest_pos[0], dest_pos[1], dest_pos[2], 0)
                return self.yard.get_container_coordinates_from_placement(placement, container.container_type)
        elif isinstance(dest_pos, list):
            # Already a list of coordinates
            return dest_pos
        else:
            # Try to parse if it's some other format
            return [dest_pos] if dest_pos else []


# Test implementation
if __name__ == '__main__':
    print("="*60)
    print("RMGC CONTROLLER TEST")
    print("="*60)
    
    yard = BooleanStorageYard(
        n_rows=5,
        n_bays=20,
        n_tiers=4,
        coordinates=[
            (1, 1, "r"), (20, 1, "r"),
            (10, 3, "dg"), (11, 3, "dg"),
            (5, 1, "sb_t"), (6, 1, "sb_t")
        ],
        split_factor=4,
        validate=False
    )
    
    # Create logistics
    logistics = BooleanLogistics(
        n_rows=5,
        n_railtracks=4,
        split_factor=4,
        yard=yard,
        validate=False
    )
    
    # Create RMGC controller
    rmgc = RMGC_Controller(
        yard=yard,
        logistics=logistics,
        heads=2
    )
    
    print(f"Terminal configuration:")
    print(f"  Positions mapped: {rmgc.num_positions}")
    print(f"  Distance matrix shape: {rmgc.distance_matrix.shape}")
    print(f"  Crane zones: {rmgc.crane_zones}")
    
    # Add test containers to yard
    print("\nAdding test containers...")
    test_containers = []
    for i in range(3):
        container = ContainerFactory.create_container(
            f"TEST_{i:03d}", "FEU", "Export", "Regular"
        )
        positions = yard.search_insertion_position(5 + i*2, 'reg', 'FEU', 2)
        if positions:
            coords = yard.get_container_coordinates_from_placement(positions[0], 'FEU')
            logistics.add_container_to_yard(container, coords)
            test_containers.append(container)
            print(f"  Added {container.container_id} at {positions[0]}")
    
    # CRITICAL: Sync yard index after adding containers
    logistics.sync_yard_index()
    print(f"Synced yard index: {len(logistics.yard_container_index)} containers indexed")
    
    # Debug: Check rail availability
    print(f"\nChecking rail availability:")
    print(f"  Wagon length: {logistics.wagon_length} positions")
    print(f"  Train head length: {logistics.train_head_length} positions")
    print(f"  Total positions needed: {logistics.train_head_length + 2 * logistics.wagon_length}")
    print(f"  Total rail positions: {logistics.n_rows * logistics.split_factor}")
    
    # Add test train with pickup requests
    print("\nAdding test train...")
    train = Train("TRAIN_001", num_wagons=2)  # Reduced to 2 wagons to ensure it fits
    
    # Add pickup requests
    train.wagons[0].add_pickup_container("TEST_000")
    train.wagons[1].add_pickup_container("TEST_001")
    
    # Add train to logistics (wagon length is automatically set by logistics)
    success = logistics.add_train_to_yard(train, 0)
    print(f"Train added to railtrack 0: {success}")
    if not success:
        print("  Trying railtrack 1...")
        success = logistics.add_train_to_yard(train, 1)
        print(f"  Train added to railtrack 1: {success}")
    
    # Also test with a truck needing containers
    print("\nAdding test truck...")
    truck = Truck("TRUCK_001")
    truck.add_pickup_container_id("TEST_002")
    
    # Add truck to parking
    truck_pos = (1, 0, 0)  # row 1, railtrack 0 (parking), split 0
    success = logistics.add_truck_to_yard(truck, truck_pos)
    print(f"Truck added to parking: {success}")
    
    # Find moves
    print("\nFinding moves...")
    moves = logistics.find_moves_optimized()
    print(f"Found {len(moves)} possible moves")
    


    
    
    # Test move masking
    print("\nTesting move masking...")
    eligible_moves = rmgc.mask_moves(moves)
    print(f"Eligible moves: {len(eligible_moves)}")
    
    # Execute a move
    if eligible_moves:
        move_id, move = next(iter(eligible_moves.items()))
        print(f"\nExecuting move: {move_id}")
        print(f"  Container: {move['container_id']}")
        print(f"  Route: {move['source_type']} -> {move['dest_type']}")
        
        distance, time = rmgc.execute_move(move)
        print(f"  Distance: {distance:.2f} meters")
        print(f"  Time: {time:.2f} seconds")
        
        # Test with one head busy
        print("\nTesting with one head busy...")
        remaining_moves = {k: v for k, v in eligible_moves.items() if k != move_id}
        masked_moves = rmgc.mask_moves(remaining_moves)
        print(f"Moves available with head 0 busy: {len(masked_moves)}")
        
        # Unlock head
        rmgc.unlock_head(0)
        print("Head 0 unlocked")
    
    print("\n✓ RMGC Controller test completed!")

    
    # Show move details
    if moves:
        print("\nMove details:")
        for move_id, move in moves.items():
            print(f"  {move_id}: {move['container_id']} from {move['source_type']} to {move['dest_type']}")