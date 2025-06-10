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
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.colors as mcolors
import random


class RMGC_Controller:
    """
    Refactored Rail Mounted Gantry Crane Controller for horizontal terminal layout.
    Optimized for the new BooleanLogistics train tracking system.
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
            # Order (horizontal layout)
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
        
        # Verify logistics has n_bays attribute
        if not hasattr(self.logistics, 'n_bays'):
            raise AttributeError(f"BooleanLogistics object missing n_bays attribute. Available attributes: {dir(self.logistics)}")
        
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
                'working_bays': set(),
                'completion_time': 0.0
            }
            for i in range(heads)
        ]
        
        # Position mappings
        self.position_to_coords = {}  # Maps position string -> (x, y, z) meters
        self.position_to_idx = {}     # Maps position string -> matrix index
        self.idx_to_position = {}     # Reverse mapping
        
        # Pre-calculate terminal dimensions (needed for coordinate system)
        self._calculate_terminal_dimensions()
        
        # Build coordinate system for horizontal layout
        self._build_coordinate_system()
        self.distance_matrix = self._build_distance_matrix()
        
        # Divide yard between crane heads
        self._split_yard_work()
        
        # Cache for physics calculations
        self.physics_cache = {}
    
    def _calculate_terminal_dimensions(self):
        """Pre-calculate terminal dimensions for visualization."""
        # X-axis spans the bays (horizontal)
        self.terminal_length = self.logistics.n_bays * self.rail_slot_length
        
        # Y-axis spans from rails to storage (vertical)
        # Calculate rail area height
        if self.logistics.n_railtracks > 1:
            self.rail_area_height = self.logistics.n_railtracks * self.rail_track_width + \
                                   (self.logistics.n_railtracks - 1) * self.space_between_rail_tracks
        else:
            self.rail_area_height = self.rail_track_width
            
        self.parking_y_start = self.rail_area_height + self.space_rails_to_parking
        self.driving_y_start = self.parking_y_start + self.parking_width
        self.storage_y_start = self.driving_y_start + self.driving_lane_width + self.space_driving_to_storage
        self.storage_area_height = self.yard.n_rows * self.storage_slot_width
        
        self.terminal_width = self.storage_y_start + self.storage_area_height
    
    def _build_coordinate_system(self):
        """Build coordinate system for horizontal terminal layout."""
        idx = 0
        
        # Add virtual stack position
        # Use terminal dimensions if available, otherwise use estimates
        stack_x = getattr(self, 'terminal_length', self.logistics.n_bays * self.rail_slot_length) + 50
        stack_y = getattr(self, 'terminal_width', 50) / 2
        self.position_to_coords["stack_0_0_0"] = (stack_x, stack_y, 0)
        self.position_to_idx["stack_0_0_0"] = idx
        self.idx_to_position[idx] = "stack_0_0_0"
        idx += 1
        
        # Rails area (horizontal tracks)
        for track in range(self.logistics.n_railtracks):
            y_pos = track * (self.rail_track_width + self.space_between_rail_tracks)
            
            for bay in range(self.logistics.n_bays):
                for split in range(self.logistics.split_factor):
                    x_pos = (bay + split/self.logistics.split_factor) * self.rail_slot_length
                    
                    pos_str = f"rail_{track}_{bay}_{split}"
                    self.position_to_coords[pos_str] = (x_pos, y_pos, self.ground_vehicle_height)
                    self.position_to_idx[pos_str] = idx
                    self.idx_to_position[idx] = pos_str
                    idx += 1
        
        # Parking area (single horizontal lane)
        y_pos = self.parking_y_start
        for bay in range(self.logistics.n_bays):
            for split in range(self.logistics.split_factor):
                x_pos = (bay + split/self.logistics.split_factor) * self.rail_slot_length
                
                pos_str = f"parking_{bay}_{split}"
                self.position_to_coords[pos_str] = (x_pos, y_pos, self.ground_vehicle_height)
                self.position_to_idx[pos_str] = idx
                self.idx_to_position[idx] = pos_str
                idx += 1
        
        # Storage yard area
        for row in range(self.yard.n_rows):
            y_pos = self.storage_y_start + row * self.storage_slot_width
            
            for bay in range(self.yard.n_bays):
                for tier in range(self.yard.n_tiers):
                    for split in range(self.yard.split_factor):
                        x_pos = (bay + split/self.yard.split_factor) * self.storage_slot_length
                        z_pos = tier * 2.59  # Standard container height
                        
                        pos_str = f"yard_{row}_{bay}_{tier}_{split}"
                        self.position_to_coords[pos_str] = (x_pos, y_pos, z_pos)
                        self.position_to_idx[pos_str] = idx
                        self.idx_to_position[idx] = pos_str
                        idx += 1
        
        self.num_positions = idx
    
    def _build_distance_matrix(self) -> np.ndarray:
        """Build optimized distance matrix for crane movements."""
        n = self.num_positions
        matrix = np.zeros((n, n), dtype=np.float32)
        
        # Pre-compute all distances
        for i in range(n):
            pos1_str = self.idx_to_position[i]
            coords1 = self.position_to_coords[pos1_str]
            
            for j in range(i+1, n):
                pos2_str = self.idx_to_position[j]
                coords2 = self.position_to_coords[pos2_str]
                
                # Manhattan distance (X-gantry, Y-trolley, Z-hoist)
                dist = abs(coords1[0] - coords2[0]) + abs(coords1[1] - coords2[1]) + abs(coords1[2] - coords2[2])
                
                matrix[i, j] = dist
                matrix[j, i] = dist  # Symmetric
        
        return matrix
    
    def _split_yard_work(self):
        """Divide yard bays between crane heads."""
        if self.heads == 1:
            self.crane_zones = [(0, self.yard.n_bays)]
        else:
            bays_per_head = self.yard.n_bays // self.heads
            overlap = max(2, bays_per_head // 4)  # 25% overlap
            
            self.crane_zones = []
            for i in range(self.heads):
                start = i * bays_per_head
                end = min((i + 1) * bays_per_head + overlap, self.yard.n_bays)
                self.crane_zones.append((start, end))
    
    def visualize_terminal(self, save_path: str = None):
        """Create a visual representation of the terminal layout with distances."""
        fig, ax = plt.subplots(1, 1, figsize=(20, 12))
        
        # Set up the plot
        ax.set_xlim(-10, self.terminal_length + 10)
        ax.set_ylim(-10, self.terminal_width + 10)
        ax.set_aspect('equal')
        ax.set_title('Container Terminal Layout - Horizontal Configuration', fontsize=16, fontweight='bold')
        ax.set_xlabel('Distance along terminal (meters)', fontsize=12)
        ax.set_ylabel('Distance across terminal (meters)', fontsize=12)
        
        # Color scheme
        colors = {
            'rail': '#4A90E2',
            'parking': '#F5A623',
            'driving': '#7ED321',
            'storage': '#BD10E0',
            'crane_zone': '#FF6B6B'
        }
        
        # Draw rail tracks
        for track in range(self.logistics.n_railtracks):
            y = track * (self.rail_track_width + self.space_between_rail_tracks)
            rail_rect = Rectangle((0, y), self.terminal_length, self.rail_track_width,
                                facecolor=colors['rail'], edgecolor='black', alpha=0.7)
            ax.add_patch(rail_rect)
            ax.text(self.terminal_length/2, y + self.rail_track_width/2, f'Rail Track {track}',
                   ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Label rail area
        ax.text(-5, self.rail_area_height/2, 'RAIL TRACKS', rotation=90,
               ha='center', va='center', fontsize=14, fontweight='bold')
        
        # Draw parking lane
        parking_rect = Rectangle((0, self.parking_y_start), self.terminal_length, self.parking_width,
                               facecolor=colors['parking'], edgecolor='black', alpha=0.7)
        ax.add_patch(parking_rect)
        ax.text(self.terminal_length/2, self.parking_y_start + self.parking_width/2, 'PARKING LANE',
               ha='center', va='center', fontsize=12, fontweight='bold')
        
        # Draw driving lane
        driving_rect = Rectangle((0, self.driving_y_start), self.terminal_length, self.driving_lane_width,
                               facecolor=colors['driving'], edgecolor='black', alpha=0.5)
        ax.add_patch(driving_rect)
        ax.text(self.terminal_length/2, self.driving_y_start + self.driving_lane_width/2, 'DRIVING LANE',
               ha='center', va='center', fontsize=12, fontweight='bold')
        
        # Draw storage yard with grid
        storage_rect = Rectangle((0, self.storage_y_start), self.terminal_length, self.storage_area_height,
                               facecolor=colors['storage'], edgecolor='black', alpha=0.3)
        ax.add_patch(storage_rect)
        
        # Draw storage grid
        for row in range(self.yard.n_rows):
            y = self.storage_y_start + row * self.storage_slot_width
            ax.axhline(y, color='gray', linewidth=0.5, alpha=0.5)
            ax.text(-3, y + self.storage_slot_width/2, f'R{row}', ha='right', va='center', fontsize=8)
        
        for bay in range(self.yard.n_bays):
            x = bay * self.storage_slot_length
            ax.axvline(x, color='gray', linewidth=0.5, alpha=0.5)
            ax.text(x + self.storage_slot_length/2, self.storage_y_start - 2, f'B{bay}',
                   ha='center', va='top', fontsize=8, rotation=45)
        
        # Label storage area
        ax.text(-5, self.storage_y_start + self.storage_area_height/2, 'STORAGE YARD',
               rotation=90, ha='center', va='center', fontsize=14, fontweight='bold')
        
        # Draw crane zones
        for i, (start_bay, end_bay) in enumerate(self.crane_zones):
            x_start = start_bay * self.storage_slot_length
            x_end = end_bay * self.storage_slot_length
            
            # Crane working area
            crane_rect = Rectangle((x_start, -5), x_end - x_start, self.terminal_width + 10,
                                 facecolor=colors['crane_zone'], alpha=0.1, 
                                 edgecolor=colors['crane_zone'], linewidth=2, linestyle='--')
            ax.add_patch(crane_rect)
            
            # Crane label
            ax.text((x_start + x_end)/2, self.terminal_width + 5, f'Crane {i} Zone',
                   ha='center', va='bottom', fontsize=12, fontweight='bold',
                   color=colors['crane_zone'])
        
        # Add distance annotations
        # Terminal length
        ax.annotate('', xy=(self.terminal_length, -8), xytext=(0, -8),
                   arrowprops=dict(arrowstyle='<->', color='black', lw=2))
        ax.text(self.terminal_length/2, -8, f'{self.terminal_length:.1f}m',
               ha='center', va='top', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor='white'))
        
        # Terminal width
        ax.annotate('', xy=(-8, self.terminal_width), xytext=(-8, 0),
                   arrowprops=dict(arrowstyle='<->', color='black', lw=2))
        ax.text(-8, self.terminal_width/2, f'{self.terminal_width:.1f}m',
               ha='right', va='center', fontsize=10, rotation=90,
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white'))
        
        # Add legend
        legend_elements = [
            patches.Patch(color=colors['rail'], label='Rail Tracks'),
            patches.Patch(color=colors['parking'], label='Parking Lane'),
            patches.Patch(color=colors['driving'], label='Driving Lane'),
            patches.Patch(color=colors['storage'], label='Storage Yard'),
            patches.Patch(color=colors['crane_zone'], alpha=0.3, label='Crane Zones')
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))
        
        # Add terminal statistics
        stats_text = f"""Terminal Statistics:
        Rails: {self.logistics.n_railtracks} tracks
        Parking: {self.logistics.n_bays} × {self.logistics.split_factor} positions
        Storage: {self.yard.n_rows} × {self.yard.n_bays} × {self.yard.n_tiers} positions
        Cranes: {self.heads} heads
        
        Dimensions:
        Bay length: {self.rail_slot_length}m
        Storage slot: {self.storage_slot_length}m × {self.storage_slot_width}m
        Total area: {self.terminal_length:.1f}m × {self.terminal_width:.1f}m"""
        
        ax.text(1.02, 0.5, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='center', bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Terminal layout saved to: {save_path}")
        
        plt.show()
    
    def _get_bays_from_position(self, pos_str: str) -> Set[int]:
        """Extract bay indices from position string."""
        bays = set()
        
        if pos_str.startswith('yard_'):
            parts = pos_str.split('_')
            bay = int(parts[2])
            bays.add(bay)
        elif pos_str.startswith('rail_') or pos_str.startswith('parking_'):
            parts = pos_str.split('_')
            bay = int(parts[1])  # Bay is second element in new format
            bays.add(bay)
        
        return bays
    
    def _position_to_string(self, pos) -> str:
        """Convert position from move format to string format for new tracking system."""
        if isinstance(pos, tuple):
            if len(pos) == 2:
                # Train position format: (train_id, wagon_idx)
                train_id, wagon_idx = pos
                
                # Find the train's location
                if train_id in self.logistics.train_to_track:
                    railtrack_id = self.logistics.train_to_track[train_id]
                    
                    # Find train's position range
                    for dep_time, (start_pos, end_pos), train in self.logistics.trains_on_track[railtrack_id]:
                        if train.train_id == train_id:
                            # Calculate wagon position
                            wagon_start = start_pos + self.logistics.train_head_length + wagon_idx * self.logistics.wagon_length
                            bay = wagon_start // self.logistics.split_factor
                            split = wagon_start % self.logistics.split_factor
                            return f"rail_{railtrack_id}_{bay}_{split}"
                
                # Fallback if train not found
                return f"rail_0_0_0"
                
            elif len(pos) == 3:
                # Truck parking position: (0, bay, split)
                _, bay, split = pos
                return f"parking_{bay}_{split}"
                
            elif len(pos) == 4:
                # Yard position: (row, bay, tier, split)
                return f"yard_{pos[0]}_{pos[1]}_{pos[2]}_{pos[3]}"
                
        elif isinstance(pos, list) and len(pos) > 0:
            # List of coordinates - use first one
            first_coord = pos[0]
            if len(first_coord) == 4:
                return f"yard_{first_coord[0]}_{first_coord[1]}_{first_coord[2]}_{first_coord[3]}"
                
        elif pos is None:
            # Virtual position for stack moves
            return "stack_0_0_0"
        
        # Unknown format
        return str(pos)
    
    def _calculate_movement_time(self, pos1: str, pos2: str, head_pos: np.ndarray) -> float:
        """Calculate time for crane movement with physics model."""
        coords1 = np.array(self.position_to_coords.get(pos1, (0, 0, 0)))
        coords2 = np.array(self.position_to_coords.get(pos2, (0, 0, 0)))
        
        # Movement sequence with proper physics
        total_time = 0.0
        
        # 1. Move to pickup position
        dx = abs(head_pos[0] - coords1[0])
        dy = abs(head_pos[1] - coords1[1])
        dz = abs(head_pos[2] - coords1[2])
        
        t_gantry = self._axis_time(dx, self.gantry_speed, self.gantry_acceleration)
        t_trolley = self._axis_time(dy, self.trolley_speed, self.trolley_acceleration)
        t_hoist_down = self._axis_time(dz, self.hoisting_speed, self.hoisting_acceleration)
        
        total_time += max(t_gantry, t_trolley, t_hoist_down)
        
        # 2. Lift container
        lift_height = self.max_height - coords1[2]
        total_time += self._axis_time(lift_height, self.hoisting_speed, self.hoisting_acceleration)
        
        # 3. Move to destination
        dx = abs(coords1[0] - coords2[0])
        dy = abs(coords1[1] - coords2[1])
        
        t_gantry = self._axis_time(dx, self.gantry_speed, self.gantry_acceleration)
        t_trolley = self._axis_time(dy, self.trolley_speed, self.trolley_acceleration)
        
        total_time += max(t_gantry, t_trolley)
        
        # 4. Lower container
        lower_height = self.max_height - coords2[2]
        total_time += self._axis_time(lower_height, self.hoisting_speed, self.hoisting_acceleration)
        
        # 5. Fixed handling time
        total_time += 30.0
        
        return total_time
    
    def _axis_time(self, distance: float, max_speed: float, acceleration: float) -> float:
        """Calculate time for single axis movement."""
        if distance == 0:
            return 0.0
        
        t_accel = max_speed / acceleration
        d_accel = 0.5 * acceleration * t_accel * t_accel
        
        if distance <= 2 * d_accel:
            # Triangular velocity profile
            return 2 * math.sqrt(distance / acceleration)
        else:
            # Trapezoidal velocity profile
            d_constant = distance - 2 * d_accel
            t_constant = d_constant / max_speed
            return 2 * t_accel + t_constant
    
    def lock_head(self, head_id: int, move: Dict):
        """Lock crane head for move execution."""
        if 0 <= head_id < self.heads:
            self.crane_heads[head_id]['busy'] = True
            self.crane_heads[head_id]['current_move'] = move
            
            # Update working bays
            bays = set()
            bays.update(self._get_bays_from_position(self._position_to_string(move['source_pos'])))
            bays.update(self._get_bays_from_position(self._position_to_string(move['dest_pos'])))
            self.crane_heads[head_id]['working_bays'] = bays
    
    def unlock_head(self, head_id: int):
        """Unlock crane head after completion."""
        if 0 <= head_id < self.heads:
            self.crane_heads[head_id]['busy'] = False
            self.crane_heads[head_id]['current_move'] = None
            self.crane_heads[head_id]['working_bays'].clear()
    
    def mask_moves(self, moves: Dict[str, Dict]) -> Dict[str, Dict]:
        """Filter moves based on available crane heads and zones."""
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
            return {}
        
        # If all heads free, return all moves
        if len(available_heads) == self.heads:
            return moves
        
        # Filter moves by busy bays
        eligible_moves = {}
        
        for move_id, move in moves.items():
            move_bays = set()
            move_bays.update(self._get_bays_from_position(self._position_to_string(move['source_pos'])))
            move_bays.update(self._get_bays_from_position(self._position_to_string(move['dest_pos'])))
            
            # Check for conflicts
            if not move_bays.intersection(busy_bays):
                # Check if any available head can reach
                for head_id in available_heads:
                    zone_start, zone_end = self.crane_zones[head_id]
                    if any(zone_start <= bay < zone_end for bay in move_bays):
                        eligible_moves[move_id] = move
                        break
        
        return eligible_moves
    
    def execute_move(self, move: Dict) -> Tuple[float, float]:
        """Execute container move with new train tracking system."""
        source_type = move['source_type']
        dest_type = move['dest_type']
        
        # Find best crane head
        head_id = self._select_best_head(move)
        if head_id is None:
            return 0.0, 0.0
        
        # Lock the head
        self.lock_head(head_id, move)
        
        # Get current head position
        head_pos = self.crane_heads[head_id]['position'].copy()
        
        # Convert positions to strings
        source_str = self._position_to_string(move['source_pos'])
        dest_str = self._position_to_string(move['dest_pos'])
        
        # Execute container transfer
        container = None
        
        # Remove from source
        if source_type == 'yard':
            # Fix: Ensure source_pos is properly formatted
            if isinstance(move['source_pos'], list):
                coords = move['source_pos']
            else:
                coords = [move['source_pos']]
            container = self.logistics.remove_container_from_yard(coords)
            
            # If removal failed, unlock and return
            if not container:
                self.unlock_head(head_id)
                return 0.0, 0.0
        
        elif source_type == 'train':
            train_id, wagon_idx = move['source_pos']
            
            # Find train by ID - use logistics tracking
            train = None
            for track_trains in self.logistics.trains_on_track.values():
                for _, _, t in track_trains:
                    if t.train_id == train_id:
                        train = t
                        break
                if train:
                    break
            
            if train and wagon_idx < len(train.wagons):
                wagon = train.wagons[wagon_idx]
                if wagon.containers:
                    container_id = move.get('container_id')
                    if container_id:
                        container = wagon.remove_container(container_id)
                    else:
                        container = wagon.remove_container(wagon.containers[0].container_id)
                    
                    if container:
                        self.logistics._update_train_lookups(train)
            
            if not container:
                self.unlock_head(head_id)
                return 0.0, 0.0
        
        elif source_type == 'truck':
            truck_pos = move['source_pos']
            truck = self.logistics.active_trucks.get(truck_pos)
            if truck and truck.containers:
                container_id = move.get('container_id')
                if container_id:
                    container = truck.remove_container(container_id)
                else:
                    container = truck.remove_container(truck.containers[0].container_id)
                
                if container:
                    self.logistics._update_truck_lookups(truck, truck_pos)
            
            if not container:
                self.unlock_head(head_id)
                return 0.0, 0.0
        
        # Add to destination
        success = False
        if container:
            if dest_type == 'yard':
                coords = self._get_yard_placement_coords(container, move['dest_pos'])
                if coords:
                    success = self.logistics.add_container_to_yard(container, coords)
            
            elif dest_type == 'train':
                train_id, wagon_idx = move['dest_pos']
                
                # Find train by ID
                train = None
                for track_trains in self.logistics.trains_on_track.values():
                    for _, _, t in track_trains:
                        if t.train_id == train_id:
                            train = t
                            break
                    if train:
                        break
                
                if train and wagon_idx < len(train.wagons):
                    success = train.wagons[wagon_idx].add_container(container)
                    if success:
                        self.logistics._update_train_lookups(train)
                        if container.container_id in train.wagons[wagon_idx].pickup_container_ids:
                            train.wagons[wagon_idx].remove_pickup_container(container.container_id)
            
            elif dest_type == 'truck':
                truck_pos = move['dest_pos']
                truck = self.logistics.active_trucks.get(truck_pos)
                if truck:
                    success = truck.add_container(container)
                    if success:
                        self.logistics._update_truck_lookups(truck, truck_pos)
                        if hasattr(truck, 'pickup_container_ids') and container.container_id in truck.pickup_container_ids:
                            truck.remove_pickup_container_id(container.container_id)
            
            elif dest_type == 'stack':
                success = True
                truck_id = self.logistics.get_available_terminal_truck()
                if truck_id is not None:
                    self.logistics.terminal_trucks[truck_id]['busy'] = True
                    self.logistics.terminal_trucks[truck_id]['completion_time'] = 300.0
        
        # Calculate physics if successful
        if success:
            # Get distance from matrix - ensure valid indices
            idx1 = self.position_to_idx.get(source_str)
            idx2 = self.position_to_idx.get(dest_str)
            
            if idx1 is not None and idx2 is not None:
                distance = self.distance_matrix[idx1, idx2]
            else:
                # Fallback calculation
                coords1 = self.position_to_coords.get(source_str, (0, 0, 0))
                coords2 = self.position_to_coords.get(dest_str, (0, 0, 0))
                distance = abs(coords1[0] - coords2[0]) + abs(coords1[1] - coords2[1]) + abs(coords1[2] - coords2[2])
            
            # Calculate time
            timeVal = self._calculate_movement_time(source_str, dest_str, head_pos)
            
            # Update head position
            new_coords = self.position_to_coords.get(dest_str, head_pos)
            self.crane_heads[head_id]['position'] = np.array(new_coords)
            self.crane_heads[head_id]['completion_time'] = timeVal
            
            # Ensure we unlock the head after time expires
            self.unlock_head(head_id)
            
            return distance, timeVal
        
        # Failed move
        self.unlock_head(head_id)
        return 0.0, 0.0
    
    def _select_best_head(self, move: Dict) -> Optional[int]:
        """Select optimal crane head for move."""
        move_bays = set()
        move_bays.update(self._get_bays_from_position(self._position_to_string(move['source_pos'])))
        move_bays.update(self._get_bays_from_position(self._position_to_string(move['dest_pos'])))
        
        best_head = None
        min_distance = float('inf')
        
        for i, head in enumerate(self.crane_heads):
            if not head['busy']:
                zone_start, zone_end = self.crane_zones[i]
                if any(zone_start <= bay < zone_end for bay in move_bays):
                    # Calculate distance to source
                    source_str = self._position_to_string(move['source_pos'])
                    if source_str in self.position_to_coords:
                        coords = self.position_to_coords[source_str]
                        dist = np.sum(np.abs(head['position'] - coords))
                        
                        if dist < min_distance:
                            min_distance = dist
                            best_head = i
        
        return best_head
    
    def _get_yard_placement_coords(self, container: Container, dest_pos) -> List[Tuple]:
        """Get yard coordinates for container placement."""
        if isinstance(dest_pos, list) and len(dest_pos) > 0:
            # Already a list of coordinates
            return dest_pos
        elif isinstance(dest_pos, tuple):
            if len(dest_pos) == 4:
                placement = dest_pos
                return self.yard.get_container_coordinates_from_placement(placement, container.container_type)
            elif len(dest_pos) == 3:
                placement = (dest_pos[0], dest_pos[1], dest_pos[2], 0)
                return self.yard.get_container_coordinates_from_placement(placement, container.container_type)
        return []
    
    def get_system_stats(self) -> Dict[str, any]:
        """Get comprehensive system statistics."""
        stats = {
            'terminal_dimensions': {
                'length': self.terminal_length,
                'width': self.terminal_width,
                'area': self.terminal_length * self.terminal_width
            },
            'positions': {
                'total': self.num_positions,
                'rails': self.logistics.n_railtracks * self.logistics.n_bays * self.logistics.split_factor,
                'parking': self.logistics.n_bays * self.logistics.split_factor,
                'storage': self.yard.n_rows * self.yard.n_bays * self.yard.n_tiers * self.yard.split_factor
            },
            'cranes': {
                'heads': self.heads,
                'zones': self.crane_zones,
                'busy_heads': sum(1 for h in self.crane_heads if h['busy'])
            },
            'performance': {
                'gantry_speed': self.gantry_speed * 60,  # m/min
                'trolley_speed': self.trolley_speed * 60,  # m/min
                'hoisting_speed': self.hoisting_speed * 60,  # m/min
                'max_height': self.max_height
            }
        }
        return stats


# Test implementation
if __name__ == '__main__':
    from datetime import datetime, timedelta
    import time
    
    print("="*80)
    print("REFACTORED RMGC CONTROLLER TEST")
    print("Testing horizontal terminal layout with new BooleanLogistics")
    print("="*80)
    
    # Create test yard
    test_yard = BooleanStorageYard(
        n_rows=5,
        n_bays=15,
        n_tiers=4,
        coordinates=[
            # Reefers on both ends
            (1, 1, "r"), (1, 2, "r"), (1, 3, "r"), (1, 4, "r"), (1, 5, "r"),
            (15, 1, "r"), (15, 2, "r"), (15, 3, "r"), (15, 4, "r"), (15, 5, "r"),
            
            # Swap bodies/trailers nearest to trucks
            (1, 1, "sb_t"), (2, 1, "sb_t"), (3, 1, "sb_t"), (4, 1, "sb_t"), (5, 1, "sb_t"),
            (6, 1, "sb_t"), (7, 1, "sb_t"), (8, 1, "sb_t"), (9, 1, "sb_t"), (10, 1, "sb_t"),
            
            # Dangerous goods in middle
            (7, 3, "dg"), (8, 3, "dg"), (9, 3, "dg"),
            (7, 4, "dg"), (8, 4, "dg"), (9, 4, "dg"),
        ],
        split_factor=4,
        validate=False
    )
    
    # Create logistics with horizontal layout
    logistics = BooleanLogistics(
        n_rows=5,
        n_bays=15,
        n_railtracks=4,
        split_factor=4,
        yard=test_yard,
        validate=True
    )
    
    # Create RMGC controller
    rmgc = RMGC_Controller(
        yard=test_yard,
        logistics=logistics,
        heads=2
    )
    
    print(f"\nTerminal Configuration:")
    print(f"  Layout: Horizontal (rails -> parking -> driving -> storage)")
    print(f"  Positions mapped: {rmgc.num_positions}")
    print(f"  Distance matrix shape: {rmgc.distance_matrix.shape}")
    print(f"  Crane zones: {rmgc.crane_zones}")
    
    # Get and print system statistics
    stats = rmgc.get_system_stats()
    print(f"\nSystem Statistics:")
    print(f"  Terminal dimensions: {stats['terminal_dimensions']['length']:.1f}m × {stats['terminal_dimensions']['width']:.1f}m")
    print(f"  Total area: {stats['terminal_dimensions']['area']:.1f} m²")
    print(f"  Position counts:")
    print(f"    Rails: {stats['positions']['rails']}")
    print(f"    Parking: {stats['positions']['parking']}")
    print(f"    Storage: {stats['positions']['storage']}")
    print(f"    Total: {stats['positions']['total']}")
    
    # Visualize the terminal layout
    print("\n" + "="*60)
    print("VISUALIZING TERMINAL LAYOUT")
    print("="*60)
    rmgc.visualize_terminal(save_path="terminal_layout.png")
    
    # Test container operations
    print("\n" + "="*60)
    print("TESTING CONTAINER OPERATIONS")
    print("="*60)
    
    # Add containers to yard
    print("\n1. Adding test containers to yard...")
    test_containers = []
    container_positions = []
    
    for i in range(3):
        container = ContainerFactory.create_container(
            f"TEST_{i:03d}", "FEU", "Export", "Regular"
        )
        positions = test_yard.search_insertion_position(5 + i*2, 'reg', 'FEU', 2)
        if positions:
            coords = test_yard.get_container_coordinates_from_placement(positions[0], 'FEU')
            logistics.add_container_to_yard(container, coords)
            test_containers.append(container)
            container_positions.append(positions[0])
            print(f"  Added {container.container_id} at position {positions[0]}")
    
    # Sync yard index
    logistics.sync_yard_index()
    print(f"\nSynced yard index: {len(logistics.yard_container_index)} containers indexed")
    
    # Create trains with the new system
    print("\n2. Adding trains with departure priorities...")
    
    current_time = datetime.now()
    
    # Train 1 - departs in 2 hours
    train1 = Train("TRAIN_001", num_wagons=3)
    train1.departure_time = current_time + timedelta(hours=2)
    train1.wagons[0].add_pickup_container("TEST_000")
    train1.wagons[1].add_pickup_container("TEST_001")
    
    # Train 2 - departs in 1 hour (higher priority)
    train2 = Train("TRAIN_002", num_wagons=2)
    train2.departure_time = current_time + timedelta(hours=1)
    train2.wagons[0].add_pickup_container("TEST_002")
    
    # Add trains to queue and process
    logistics.add_train_to_queue(train1)
    logistics.add_train_to_queue(train2)
    
    trains_placed = logistics.process_current_trains()
    print(f"\nTrains placed: {trains_placed}")
    
    # Print train positions
    print("\nTrain positions on tracks:")
    for track_id, track_trains in logistics.trains_on_track.items():
        if track_trains:
            print(f"  Track {track_id}:")
            for dep_time, (start, end), train in track_trains:
                print(f"    {train.train_id}: positions [{start}-{end}), departs {dep_time.strftime('%H:%M')}")
    
    # Add trucks
    print("\n3. Adding test trucks...")
    
    truck1 = Truck("TRUCK_001")
    truck1.add_pickup_container_id("TEST_001")
    
    truck2 = Truck("TRUCK_002")
    delivery_container = ContainerFactory.create_container("DELIVERY_001", "TWEU", "Import", "Regular")
    truck2.add_container(delivery_container)
    
    logistics.add_truck_to_queue(truck1)
    logistics.add_truck_to_queue(truck2)
    
    trucks_placed = logistics.process_current_trucks()
    print(f"\nTrucks placed: {trucks_placed}")
    
    # Find and categorize moves
    print("\n4. Finding possible moves...")
    moves = logistics.find_moves_optimized()
    print(f"Found {len(moves)} possible moves")
    
    # Categorize moves
    move_categories = defaultdict(list)
    for move_id, move in moves.items():
        move_categories[move['move_type']].append((move_id, move))
    
    print("\nMove breakdown:")
    for move_type, move_list in move_categories.items():
        print(f"  {move_type}: {len(move_list)} moves")
        # Show first move of each type
        if move_list:
            move_id, move = move_list[0]
            print(f"    Example: {move['container_id']} ({move['source_type']} -> {move['dest_type']})")
    
    # Test move execution
    print("\n5. Testing move execution...")
    eligible_moves = rmgc.mask_moves(moves)
    print(f"Eligible moves (after masking): {len(eligible_moves)}")
    
    if eligible_moves:
        # Execute first available move
        move_id, move = next(iter(eligible_moves.items()))
        print(f"\nExecuting move: {move_id}")
        print(f"  Container: {move['container_id']}")
        print(f"  Type: {move['move_type']}")
        print(f"  Route: {move['source_type']} -> {move['dest_type']}")
        
        # Test position string conversion
        source_str = rmgc._position_to_string(move['source_pos'])
        dest_str = rmgc._position_to_string(move['dest_pos'])
        print(f"  Source position: {move['source_pos']} -> {source_str}")
        print(f"  Dest position: {move['dest_pos']} -> {dest_str}")
        
        # Execute move
        distance, exec_time = rmgc.execute_move(move)
        print(f"\nMove results:")
        print(f"  Distance: {distance:.2f} meters")
        print(f"  Time: {exec_time:.2f} seconds ({exec_time/60:.1f} minutes)")
        print(f"  Average speed: {distance/exec_time:.2f} m/s")
        
        # Check crane status
        print(f"\nCrane status after move:")
        for i, head in enumerate(rmgc.crane_heads):
            status = "BUSY" if head['busy'] else "IDLE"
            print(f"  Crane {i}: {status}, position: ({head['position'][0]:.1f}, {head['position'][1]:.1f}, {head['position'][2]:.1f})")
    
    # Test reorganization
    print("\n6. Testing logistics reorganization...")
    summary, penalty = logistics.reorganize_logistics()
    early_departures, late_departures, truck_departures = summary
    
    print(f"\nReorganization results:")
    print(f"  Trains departed early: {early_departures}")
    print(f"  Trains departed late: {late_departures}")
    print(f"  Trucks departed: {truck_departures}")
    print(f"  Total penalty: {penalty}")
    
    # Performance test
    print("\n" + "="*60)
    print("PERFORMANCE TEST")
    print("="*60)
    
    # Test move finding performance
    print("Testing move finding performance (100 iterations)...")
    start_time = time.time()
    for _ in range(100):
        moves = logistics.find_moves_optimized()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 100
    print(f"  Average time to find moves: {avg_time*1000:.2f} ms")
    print(f"  Moves per second: {1/avg_time:.1f}")
    
    # Test distance calculation performance
    print("\nTesting distance calculation performance...")
    if rmgc.num_positions > 100:
        sample_positions = random.sample(list(rmgc.idx_to_position.keys()), 100)
        
        start_time = time.time()
        distance_sum = 0
        for i in range(100):
            idx1, idx2 = random.sample(sample_positions, 2)
            distance_sum += rmgc.distance_matrix[idx1, idx2]
        end_time = time.time()
        
        print(f"  100 distance lookups: {(end_time - start_time)*1000:.2f} ms")
        print(f"  Average distance: {distance_sum/100:.1f} meters")
    
    print("\n" + "="*80)
    print("✓ All tests completed successfully!")
    print("✓ Terminal visualization saved to: terminal_layout.png")
    print("="*80)