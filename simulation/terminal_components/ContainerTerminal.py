from typing import Dict, Tuple, List, Optional, Any
import numpy as np
import torch
from collections import defaultdict, deque
import gymnasium as gym
from gymnasium import spaces
from datetime import datetime, timedelta
import random
import os
from simulation.terminal_components.TemporalState import TemporalStateEnhancement, ProgressBasedRewardCalculator, MoveEvaluator
from simulation.terminal_components.Container import Container, ContainerFactory
from simulation.terminal_components.BooleanStorage import BooleanStorageYard
from simulation.terminal_components.BooleanLogistics import BooleanLogistics
from simulation.terminal_components.RMGC import RMGC_Controller
from simulation.terminal_components.Train import Train
from simulation.terminal_components.Truck import Truck
from simulation.terminal_components.Wagon import Wagon
from simulation.TerminalConfig import TerminalConfig
from simulation.kde_sampling_utils import load_kde_model, sample_from_kde, hours_to_time


class ContainerTerminal(gym.Env):
    """
    Optimized container terminal environment for continuous DRL training.
    Updated to work with refactored Boolean components.
    """
    
    def __init__(
        self,
        n_rows: int = 15,
        n_bays: int = 20,
        n_tiers: int = 5,
        n_railtracks: int = 4,
        split_factor: int = 4,
        max_days: int = 365,
        time_per_day: float = 86400.0,  # seconds
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()
        
        # Terminal configuration
        self.n_rows = n_rows
        self.n_bays = n_bays
        self.n_tiers = n_tiers
        self.n_railtracks = n_railtracks
        self.split_factor = split_factor
        self.max_days = max_days
        self.time_per_day = time_per_day
        self.device = device
        
        # Initialize components
        self._init_terminal_components()
        
        # Time tracking
        self.current_time = 0.0
        self.current_day = 0
        self.crane_completion_times = np.zeros(self.rmgc.heads)
        
        # Performance tracking
        self.daily_metrics = defaultdict(list)
        self.episode_metrics = defaultdict(float)
        
        # Reward parameters
        self.max_distance = self._calculate_max_distance()
        
        # Temporal awareness components
        self.temporal_encoder = TemporalStateEnhancement(self.time_per_day)
        self.reward_calculator = ProgressBasedRewardCalculator(distance_weight=0.1)
        
        # Daily goals tracking
        self.daily_goals = {
            'train_departures': [],
            'truck_pickups': [],
            'container_movements': defaultdict(list)
        }
        
        # Move ranking cache
        self.ranked_moves_cache = None
        self.cache_timestamp = -1
        
        # Action and observation spaces
        self._init_spaces()
        
        # Load KDE models for generation
        self.kde_models = self._load_kde_models()
        
        # Container tracking
        self.container_arrival_times = {}
        self.container_pickup_schedules = {}
        
        # Daily generation parameters
        self.trains_per_day_range = (3, 8)
        self.trucks_per_day_range = (20, 40)
        self.containers_per_train_range = (15, 30)
        
        # Move history tracking
        self.move_history = []
        
        # Penalty tracking
        self.penalty_metrics = {
            'daily_penalties': [],
            'forced_train_departures': [],
            'forced_truck_departures': [],
            'unfulfilled_pickups': [],
            'undelivered_containers': []
        }

    def _init_terminal_components(self):
        """Initialize yard, logistics, and RMGC components."""
        # Define special storage areas
        coordinates = []
        
        # Reefer areas on ends
        for row in [1, 2]:
            for bay in [1, 2, self.n_bays-1, self.n_bays]:
                coordinates.append((bay, row, "r"))
        
        # Dangerous goods in middle
        mid_bay = self.n_bays // 2
        mid_row = self.n_rows // 2
        for offset in [-1, 0, 1]:
            coordinates.append((mid_bay + offset, mid_row, "dg"))
            coordinates.append((mid_bay + offset, mid_row + 1, "dg"))
        
        # Swap bodies/trailers near trucks (first row)
        for bay in range(1, min(10, self.n_bays + 1)):
            coordinates.append((bay, 1, "sb_t"))
        
        # Initialize yard
        self.yard = BooleanStorageYard(
            n_rows=self.n_rows,
            n_bays=self.n_bays,
            n_tiers=self.n_tiers,
            coordinates=coordinates,
            split_factor=self.split_factor,
            validate=False,
            device=self.device
        )
        
        # Initialize logistics WITH n_bays parameter
        self.logistics = BooleanLogistics(
            n_rows=self.n_rows,
            n_bays=self.n_bays,  # NOW REQUIRED
            n_railtracks=self.n_railtracks,
            split_factor=self.split_factor,
            yard=self.yard,
            validate=False,
            device=self.device
        )
        
        # Initialize RMGC controller
        self.rmgc = RMGC_Controller(
            yard=self.yard,
            logistics=self.logistics,
            heads=2
        )
    
    def _init_spaces(self):
        """Initialize action and observation spaces with temporal features."""
        # State dimensions
        yard_state_dim = 5 * self.n_rows * self.n_bays * self.n_tiers * self.split_factor
        rail_state_dim = self.n_railtracks * self.n_bays * self.split_factor
        parking_state_dim = 1 * self.n_bays * self.split_factor
        train_props_dim = self.n_railtracks * 3
        truck_props_dim = self.n_bays * self.split_factor * 3
        queue_dim = 2
        time_dim = 3
        temporal_dim = self.temporal_encoder.feature_dim
        
        total_state_dim = (yard_state_dim + rail_state_dim + parking_state_dim + 
                          train_props_dim + truck_props_dim + queue_dim + 
                          time_dim + temporal_dim)
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_state_dim,),
            dtype=np.float32
        )
        
        # Action space
        max_moves = 1000
        self.action_space = spaces.Discrete(max_moves)
    
    def _calculate_max_distance(self) -> float:
        """Calculate maximum possible distance in terminal."""
        max_x = self.n_bays * 12.192
        max_y = self.n_rows * 2.44
        max_z = self.rmgc.max_height
        return max_x + max_y + max_z
    
    def _load_kde_models(self) -> Dict[str, Any]:
        """Load KDE models from data/models directory."""
        kde_models = {}
        model_files = {
            'train_arrival': 'train_arrival_kde.pkl',
            'train_delay': 'train_delay_kde.pkl',
            'truck_pickup': 'truck_pickup_kde.pkl',
            'pickup_wait': 'pickup_wait_kde.pkl',
            'container_weight': 'container_weight_kde.pkl'
        }
        
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_dir = os.path.join(base_dir, 'data', 'models')
        
        for model_name, file_name in model_files.items():
            file_path = os.path.join(models_dir, file_name)
            try:
                if os.path.exists(file_path):
                    kde_models[model_name] = load_kde_model(file_path)
                else:
                    kde_models[model_name] = None
            except Exception as e:
                print(f"Error loading {model_name}: {e}")
                kde_models[model_name] = None
                
        return kde_models
    
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset environment for new episode."""
        super().reset(seed=seed)
        
        # Re-initialize components
        self._init_terminal_components()
        
        # Reset time
        self.current_time = 0.0
        self.current_day = 0
        self.crane_completion_times = np.zeros(self.rmgc.heads)
        
        # Reset tracking
        self.container_arrival_times.clear()
        self.container_pickup_schedules.clear()
        self.daily_metrics.clear()
        self.episode_metrics.clear()
        self.move_history.clear()
        
        # Generate initial day
        self._generate_daily_traffic()
        
        # Get initial state
        state = self._get_state()
        info = self._get_info()
        
        return state, info
    
    def _generate_daily_traffic(self):
        """
        CRITICAL: Properly generate daily traffic ensuring moves are always available.
        Focus on container lifecycle management and proper queue handling.
        """
        base_date = datetime(2025, 1, 1) + timedelta(days=self.current_day)
        current_datetime = base_date
        
        # First, ensure we have containers in yard if day > 0
        # containers_in_yard = len(self.logistics.yard_container_index)
        
        if self.current_day == 0:
            # Initial population or replenishment
            n_initial = 50 if self.current_day == 0 else 20
            print(f"Day {self.current_day}: Adding {n_initial} containers to yard")
            
            for i in range(n_initial):
                container = self._create_container_with_weight(
                    f"YARD_{self.current_day:03d}_{i:04d}"
                )
                container.direction = "Export"
                container.arrival_date = base_date - timedelta(days=random.randint(1, 3))
                
                if self._place_container_in_yard(container):
                    # Schedule future pickup
                    pickup_time = self._sample_pickup_time(base_date)
                    self.container_pickup_schedules[container.container_id] = pickup_time
        
        # Sync yard index after initial population
        self.logistics.sync_yard_index()
        
        # Generate trains
        n_trains = random.randint(*self.trains_per_day_range)
        train_counter = 0
        container_counter = 1000
        
        # Sample arrival times
        if self.kde_models.get('train_arrival'):
            arrival_hours = sample_from_kde(self.kde_models['train_arrival'], n_samples=n_trains)
        else:
            arrival_hours = np.random.uniform(0, 24, n_trains)
        
        # Create trains
        for i in range(n_trains):
            train_id = f"TRN_{self.current_day:03d}_{i:04d}"
            
            # Calculate arrival time
            h, m, s = hours_to_time(arrival_hours[i])
            arrival_time = base_date.replace(hour=h, minute=m, second=s)
            
            # Create train
            n_wagons = random.randint(2, 4)
            train = Train(train_id, num_wagons=n_wagons, arrival_time=arrival_time)
            
            # Set departure time (4-8 hours after arrival)
            departure_hours = random.uniform(4, 8)
            train.departure_time = arrival_time + timedelta(hours=departure_hours)
            
            # 50/50 import/export trains
            if random.random() < 0.5:
                # IMPORT TRAIN - brings containers
                n_containers = random.randint(10, 20)
                containers_added = 0
                
                for j in range(n_containers):
                    container = self._create_container_with_weight(
                        f"IMP_{self.current_day:03d}_{container_counter:04d}"
                    )
                    container_counter += 1
                    container.direction = "Import"
                    container.arrival_date = arrival_time
                    
                    # Try to add to wagon
                    wagon_idx = containers_added % len(train.wagons)
                    if train.wagons[wagon_idx].add_container(container):
                        containers_added += 1
                        # Schedule future pickup
                        pickup_time = self._sample_pickup_time(base_date)
                        self.container_pickup_schedules[container.container_id] = pickup_time
            else:
                # EXPORT TRAIN - picks up from yard
                available_for_train = []
                
                # Find unassigned containers in yard
                for cid in list(self.logistics.yard_container_set):
                    if (cid not in self.logistics.pickup_to_train and 
                        cid not in self.logistics.pickup_to_truck):
                        # Check if container exists and is export
                        pos = self.logistics.yard_container_index.get(cid)
                        if pos:
                            try:
                                row, bay, tier, split = pos
                                container = self.yard.get_container_at(row, bay, tier, split)
                                if container and container.direction == "Export":
                                    available_for_train.append(cid)
                            except:
                                continue
                
                # Assign pickups
                if available_for_train:
                    n_pickups = min(random.randint(5, 15), len(available_for_train))
                    selected = random.sample(available_for_train, n_pickups)
                    
                    for container_id in selected:
                        wagon_idx = random.randint(0, len(train.wagons) - 1)
                        train.wagons[wagon_idx].add_pickup_container(container_id)
            
            # Add train to system
            if arrival_time <= current_datetime + timedelta(hours=1):
                # Immediate arrival
                self.logistics.add_train_to_queue(train)
            else:
                # Schedule for later
                self.logistics.trains.schedule_arrival(train, arrival_time)
            
            train_counter += 1
        
        # Process train queue immediately
        trains_placed = self.logistics.process_current_trains()
        
        # Generate trucks
        n_trucks = random.randint(*self.trucks_per_day_range)
        truck_counter = 0
        
        # Sample truck arrival times
        if self.kde_models.get('truck_pickup'):
            truck_hours = sample_from_kde(self.kde_models['truck_pickup'], n_samples=n_trucks)
        else:
            truck_hours = np.random.uniform(6, 18, n_trucks)
        
        for i in range(n_trucks):
            truck_id = f"TRK_{self.current_day:03d}_{i:03d}"
            truck = Truck(truck_id)
            
            # 30% delivery, 70% pickup
            if random.random() < 0.3:
                # DELIVERY TRUCK
                container = self._create_container_with_weight(
                    f"DEL_{self.current_day:03d}_{container_counter:04d}"
                )
                container_counter += 1
                container.direction = "Import"
                
                if truck.add_container(container):
                    # Schedule future pickup
                    pickup_time = self._sample_pickup_time(base_date)
                    self.container_pickup_schedules[container.container_id] = pickup_time
            else:
                # PICKUP TRUCK
                available_for_truck = []
                
                # Find unassigned containers
                for cid in list(self.logistics.yard_container_set):
                    if (cid not in self.logistics.pickup_to_truck and 
                        cid not in self.logistics.pickup_to_train):
                        # Prioritize containers due for pickup
                        if cid in self.container_pickup_schedules:
                            pickup_time = self.container_pickup_schedules[cid]
                            if pickup_time <= base_date + timedelta(days=1):
                                available_for_truck.append((0, cid))  # High priority
                            else:
                                available_for_truck.append((1, cid))  # Lower priority
                        else:
                            available_for_truck.append((2, cid))  # Lowest priority
                
                if available_for_truck:
                    # Sort by priority
                    available_for_truck.sort()
                    n_pickups = min(random.randint(1, 3), len(available_for_truck))
                    
                    for _, container_id in available_for_truck[:n_pickups]:
                        truck.add_pickup_container_id(container_id)
            
            # Set arrival time
            h, m, s = hours_to_time(truck_hours[i])
            arrival_time = base_date.replace(hour=h, minute=m, second=s)
            truck.arrival_time = arrival_time
            truck.max_dwell_time = 8.0
            
            # Add truck to system
            if arrival_time <= current_datetime + timedelta(hours=1):
                # Immediate arrival
                self.logistics.add_truck_to_queue(truck)
            else:
                # Schedule for later
                self.logistics.trucks.schedule_arrival(truck, arrival_time)
            
            truck_counter += 1
        
        # Process truck queue immediately
        trucks_placed = self.logistics.process_current_trucks()
        
        # Handle swap bodies/trailers with terminal trucks
        if self.current_day > 0:
            sb_trailers = []
            for cid, pos in self.logistics.yard_container_index.items():
                row, bay, tier, split = pos
                try:
                    container = self.yard.get_container_at(row, bay, tier, split)
                    if container and container.container_type in ['Swap Body', 'Trailer']:
                        if container.arrival_date:
                            days_in_yard = (base_date - container.arrival_date).days
                            if days_in_yard > 3:
                                sb_trailers.append(container.container_id)
                except:
                    continue
            
            # Mark some for removal via terminal trucks
            if sb_trailers:
                n_to_remove = min(3, len(sb_trailers))
                for cid in random.sample(sb_trailers, n_to_remove):
                    # This will trigger yard_to_stack moves
                    self.container_pickup_schedules[cid] = base_date + timedelta(hours=1)
        
        # Update priorities
        self._update_container_priorities()
        
        # Final sync
        self.logistics.sync_yard_index()
        
        # Ensure moves are available
        moves = self.logistics.find_moves_optimized()
        
        print(f"\nDay {self.current_day} generation complete:")
        print(f"  Trains: {n_trains} scheduled, {trains_placed} placed")
        print(f"  Trucks: {n_trucks} scheduled, {trucks_placed} placed")
        print(f"  Containers in yard: {len(self.logistics.yard_container_index)}")
        print(f"  Available moves: {len(moves)}")
        
        # If no moves available, add emergency vehicles
        if len(moves) == 0:
            print("WARNING: No moves available! Adding emergency vehicles...")
            self._add_emergency_vehicles()
    
    def _add_emergency_vehicles(self):
        """Add emergency vehicles to ensure moves are available."""
        base_date = datetime(2025, 1, 1) + timedelta(days=self.current_day)
        
        # Add pickup truck for any available container
        if self.logistics.yard_container_set:
            truck = Truck(f"EMERGENCY_TRK_{self.current_day}")
            
            # Pick first available container
            for cid in list(self.logistics.yard_container_set)[:3]:
                if (cid not in self.logistics.pickup_to_truck and 
                    cid not in self.logistics.pickup_to_train):
                    truck.add_pickup_container_id(cid)
            
            if truck.pickup_container_ids:
                truck.arrival_time = base_date
                self.logistics.add_truck_to_queue(truck)
                self.logistics.process_current_trucks()
        
        # Add delivery truck with container
        delivery_truck = Truck(f"EMERGENCY_DEL_{self.current_day}")
        container = self._create_container_with_weight(f"EMERGENCY_{self.current_day}")
        container.direction = "Import"
        
        if delivery_truck.add_container(container):
            delivery_truck.arrival_time = base_date
            self.logistics.add_truck_to_queue(delivery_truck)
            self.logistics.process_current_trucks()
        
        # Sync and check
        self.logistics.sync_yard_index()
        moves = self.logistics.find_moves_optimized()
        print(f"  Emergency vehicles added. Moves now: {len(moves)}")
    
    def _create_container_with_weight(self, container_id: str) -> Container:
        """Create container with weight sampled from KDE."""
        container = ContainerFactory.create_random(container_id)
        
        # Avoid FFEU for now (cross-bay complexity)
        while container.container_type == "FFEU":
            container = ContainerFactory.create_random(container_id)
        
        # Sample weight
        if self.kde_models.get('container_weight'):
            weight = sample_from_kde(
                self.kde_models['container_weight'], 
                n_samples=1, 
                min_val=1000, 
                max_val=31000
            )[0]
            container.weight = float(weight)
        
        return container
    
    def _sample_pickup_time(self, base_date: datetime) -> datetime:
        """Sample pickup time using KDE."""
        if self.kde_models.get('pickup_wait'):
            wait_hours = sample_from_kde(
                self.kde_models['pickup_wait'], 
                n_samples=1, 
                min_val=0, 
                max_val=168
            )[0]
            
            wait_days = int(wait_hours // 24)
            pickup_date = base_date + timedelta(days=wait_days)
            
            if self.kde_models.get('truck_pickup'):
                pickup_hour = sample_from_kde(self.kde_models['truck_pickup'], n_samples=1)[0]
            else:
                pickup_hour = random.uniform(6, 18)
                
            h, m, s = hours_to_time(pickup_hour)
            return pickup_date.replace(hour=h, minute=m, second=s)
        else:
            days_ahead = random.randint(1, 5)
            hour = random.randint(6, 18)
            return base_date + timedelta(days=days_ahead, hours=hour)
    
    def _place_container_in_yard(self, container: Container) -> bool:
        """Place container in appropriate yard location."""
        goods_type = 'r' if container.goods_type == 'Reefer' else \
                    'dg' if container.goods_type == 'Dangerous' else \
                    'sb_t' if container.container_type in ['Trailer', 'Swap Body'] else 'reg'
        
        # Try multiple positions
        for attempt in range(10):
            bay = random.randint(2, self.n_bays - 3)
            positions = self.yard.search_insertion_position(
                bay, goods_type, container.container_type, 3
            )
            
            if positions:
                placement = positions[0]
                coords = self.yard.get_container_coordinates_from_placement(
                    placement, container.container_type
                )
                
                # Validate coordinates
                valid = True
                for coord in coords:
                    r, b, s, t = coord
                    if not (0 <= r < self.yard.n_rows and 
                           0 <= b < self.yard.n_bays and 
                           0 <= s < self.yard.split_factor and 
                           0 <= t < self.yard.n_tiers):
                        valid = False
                        break
                
                if valid:
                    success = self.logistics.add_container_to_yard(container, coords)
                    if success:
                        return True
        
        return False
    
    def _update_container_priorities(self):
        """Update priorities for all containers based on pickup schedules."""
        current_date = datetime(2025, 1, 1) + timedelta(
            days=self.current_day, 
            seconds=self.current_time
        )
        
        for container_id, pos in self.logistics.yard_container_index.items():
            row, bay, tier, split = pos
            try:
                container = self.yard.get_container_at(row, bay, tier, split)
                if container:
                    container.update_priority()
                    
                    if container.container_id in self.container_pickup_schedules:
                        pickup_time = self.container_pickup_schedules[container.container_id]
                        hours_until = (pickup_time - current_date).total_seconds() / 3600
                        
                        if hours_until < 0:
                            container.priority = 1  # Overdue
                        elif hours_until <= 4:
                            container.priority = max(1, container.priority - 40)
                        elif hours_until <= 24:
                            container.priority = max(1, container.priority - 20)
                    
                    self.yard._update_property_arrays(row, bay, tier, split, container)
            except:
                continue
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute action with updated move handling."""
        # Get and rank moves
        moves = self.logistics.find_moves_optimized()
        eligible_moves = self.rmgc.mask_moves(moves)
        
        # Rank moves if needed
        if eligible_moves and self.current_time != self.cache_timestamp:
            self.ranked_moves_cache = self._rank_moves_by_urgency(eligible_moves)
            self.cache_timestamp = self.current_time
        
        # Execute action
        if not eligible_moves:
            # No moves - wait and check for new arrivals
            self._advance_time(30.0)
            reward = -0.1  # Small penalty for waiting
        else:
            # Select from available moves
            move_list = self.ranked_moves_cache or list(eligible_moves.items())
            
            if action >= len(move_list):
                action = action % len(move_list)
            
            move_id, move = move_list[action]
            
            # Execute move
            distance, time = self.rmgc.execute_move(move)
            
            if time > 0:
                # Calculate reward
                current_datetime = datetime(2025, 1, 1) + timedelta(
                    days=self.current_day,
                    seconds=self.current_time
                )
                
                reward = self.reward_calculator.calculate_reward(
                    move=move,
                    distance=distance,
                    time=time,
                    current_datetime=current_datetime,
                    trains={t.train_id: t for t in self.logistics.trains_on_track.values() 
                            for _, _, t in t},
                    trucks=self.logistics.active_trucks,
                    container_pickup_schedules=self.container_pickup_schedules,
                    logistics=self.logistics
                )
                
                # Schedule crane unlock
                head_id = self._get_crane_head_for_move(move)
                if head_id is not None:
                    self.crane_completion_times[head_id] = self.current_time + time
                
                # Advance time
                self._advance_time(time)
                
                # Track metrics
                self.daily_metrics['moves'].append(move_id)
                self.daily_metrics['distances'].append(distance)
                self.daily_metrics['times'].append(time)
                self.daily_metrics['rewards'].append(reward)
                
                # Track move history
                self.move_history.append({
                    'day': self.current_day,
                    'move_id': move_id,
                    'container_id': move.get('container_id', ''),
                    'source_type': move.get('source_type', ''),
                    'source_pos': str(move.get('source_pos', '')),
                    'dest_type': move.get('dest_type', ''),
                    'dest_pos': str(move.get('dest_pos', '')),
                    'move_type': move.get('move_type', ''),
                    'reward': reward,
                    'distance': distance,
                    'time': time
                })
            else:
                reward = -1.0  # Failed move
        
        # Update crane states
        self._update_crane_states()
        
        # Reorganize logistics
        departed, penalty = self.logistics.reorganize_logistics()
        trains_early, trains_late, trucks = departed
        
        # Apply penalty and bonuses
        reward += penalty
        if trains_early > 0:
            reward += trains_early * 5.0
        if trains_late > 0 and penalty == 0:
            reward -= trains_late * 2.0
        
        # Check day end
        day_ended = self.current_time >= self.time_per_day
        if day_ended:
            self._end_of_day()
        
        # Check termination
        terminated = self.current_day >= self.max_days
        truncated = False
        
        # Get new state and info
        state = self._get_state()
        info = self._get_info()
        
        # Add ranking info
        if eligible_moves:
            info['ranked_move_list'] = [m[0] for m in (self.ranked_moves_cache or [])]
        
        return state, reward, terminated, truncated, info
    
    def _rank_moves_by_urgency(self, moves: Dict[str, Dict]) -> List[Tuple[str, Dict]]:
        """Rank moves by urgency and value."""
        current_datetime = datetime(2025, 1, 1) + timedelta(
            days=self.current_day,
            seconds=self.current_time
        )
        
        # Get all trains from new tracking system
        all_trains = {}
        for track_trains in self.logistics.trains_on_track.values():
            for _, _, train in track_trains:
                all_trains[train.train_id] = train
        
        move_scores = []
        
        for move_id, move in moves.items():
            urgency, value = self.reward_calculator.move_evaluator.evaluate_move_urgency(
                move=move,
                current_datetime=current_datetime,
                trains=all_trains,
                trucks=self.logistics.active_trucks,
                container_pickup_schedules=self.container_pickup_schedules,
                logistics=self.logistics
            )
            
            # Estimate distance
            try:
                source_str = self.rmgc._position_to_string(move['source_pos'])
                dest_str = self.rmgc._position_to_string(move['dest_pos'])
                
                idx1 = self.rmgc.position_to_idx.get(source_str)
                idx2 = self.rmgc.position_to_idx.get(dest_str)
                
                if idx1 is not None and idx2 is not None:
                    distance = self.rmgc.distance_matrix[idx1, idx2]
                else:
                    distance = 50.0
            except:
                distance = 50.0
            
            score = (urgency + 1.0) * value - (distance * 0.01)
            move_scores.append((score, distance, move_id, move))
        
        # Sort by score desc, distance asc
        move_scores.sort(key=lambda x: (-x[0], x[1]))
        
        return [(item[2], item[3]) for item in move_scores]
    
    def _advance_time(self, seconds: float):
        """Advance simulation time and process arrivals."""
        self.current_time += seconds
        
        # Update vehicle arrivals
        current_datetime = datetime(2025, 1, 1) + timedelta(
            days=self.current_day,
            seconds=self.current_time
        )
        
        arrived_trains = self.logistics.trains.update(current_datetime)
        arrived_trucks = self.logistics.trucks.update(current_datetime)
        
        # Process new arrivals
        if arrived_trains or arrived_trucks:
            print(f"  Time {self.current_time:.0f}s: {len(arrived_trains)} trains and {len(arrived_trucks)} trucks arrived")
            trains_placed = self.logistics.process_current_trains()
            trucks_placed = self.logistics.process_current_trucks()
            self.logistics.sync_yard_index()
            print(f"    Placed: {trains_placed} trains, {trucks_placed} trucks")
    
    def _update_crane_states(self):
        """Update crane head states."""
        for head_id, completion_time in enumerate(self.crane_completion_times):
            if completion_time > 0 and self.current_time >= completion_time:
                self.rmgc.unlock_head(head_id)
                self.crane_completion_times[head_id] = 0.0
    
    def _get_crane_head_for_move(self, move: Dict) -> Optional[int]:
        """Get crane head executing move."""
        for i, head in enumerate(self.rmgc.crane_heads):
            if head['busy'] and head['current_move'] == move:
                return i
        return None
    
    def _end_of_day(self):
        """Process end of day tasks."""
        # Calculate statistics
        if self.daily_metrics['moves']:
            avg_distance = np.mean(self.daily_metrics['distances'])
            total_reward = sum(self.daily_metrics['rewards'])
            
            self.episode_metrics['total_moves'] += len(self.daily_metrics['moves'])
            self.episode_metrics['total_reward'] += total_reward
        
        # Save move history
        if hasattr(self, 'move_log_file') and self.move_history:
            import csv
            with open(self.move_log_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'day', 'move_id', 'container_id', 'source_type', 'source_pos',
                    'dest_type', 'dest_pos', 'move_type', 'reward', 'distance', 'time'
                ])
                if self.current_day == 0:
                    writer.writeheader()
                writer.writerows(self.move_history)
            self.move_history.clear()
        
        # Clear daily metrics
        self.daily_metrics.clear()
        
        # Advance day
        self.current_day += 1
        self.current_time = 0.0
        
        # Generate new traffic
        if self.current_day < self.max_days:
            self._generate_daily_traffic()
    
    def _get_state(self) -> np.ndarray:
        """Get current state with all components."""
        # Yard state
        yard_state = self.yard.get_full_state_tensor(flatten=True).cpu().numpy()
        
        # Rail and parking states
        rail_state = self.logistics.get_rail_state_tensor(as_tensor=False).flatten()
        parking_state = self.logistics.get_parking_state_tensor(as_tensor=False).flatten()
        
        # Vehicle properties
        train_props = self.logistics.get_train_properties_tensor(as_tensor=False).flatten()
        truck_props = self.logistics.get_truck_properties_tensor(as_tensor=False).flatten()
        
        # Queue state
        queue_state = self.logistics.get_queue_state_tensor(as_tensor=False)
        
        # Time state
        time_state = np.array([
            self.current_time / self.time_per_day,
            self.current_day / self.max_days,
            (self.time_per_day - self.current_time) / self.time_per_day
        ], dtype=np.float32)
        
        # Temporal features
        all_trains = {t.train_id: t for track_trains in self.logistics.trains_on_track.values() 
                      for _, _, t in track_trains}
        
        temporal_features = self.temporal_encoder.encode_daily_schedule(
            current_time=self.current_time,
            current_day=self.current_day,
            trains=all_trains,
            trucks=self.logistics.active_trucks,
            train_queue=self.logistics.trains,
            truck_queue=self.logistics.trucks,
            container_pickup_schedules=self.container_pickup_schedules,
            container_arrival_times=self.container_arrival_times
        )
        
        # Concatenate all
        state = np.concatenate([
            yard_state,
            rail_state,
            parking_state,
            train_props,
            truck_props,
            queue_state,
            time_state,
            temporal_features
        ])
        
        return state.astype(np.float32)
    
    def _get_info(self) -> Dict[str, Any]:
        """Get environment info."""
        moves = self.logistics.find_moves_optimized()
        eligible_moves = self.rmgc.mask_moves(moves)
        
        # Count trains
        total_trains = sum(len(trains) for trains in self.logistics.trains_on_track.values())
        
        return {
            'day': self.current_day,
            'time': self.current_time,
            'available_moves': len(eligible_moves),
            'move_list': list(eligible_moves.keys()),
            'trains_in_terminal': total_trains,
            'trucks_in_terminal': len(self.logistics.active_trucks),
            'containers_in_yard': len(self.logistics.yard_container_index),
            'episode_metrics': dict(self.episode_metrics)
        }
    
    def render(self):
        """Render terminal state."""
        total_trains = sum(len(trains) for trains in self.logistics.trains_on_track.values())
        
        print(f"\n=== Day {self.current_day}, Time: {self.current_time:.0f}s ===")
        print(f"Trains: {total_trains}, Trucks: {len(self.logistics.active_trucks)}")
        print(f"Yard containers: {len(self.logistics.yard_container_index)}")
        print(f"Queues - Trains: {self.logistics.trains.size()}, Trucks: {self.logistics.trucks.size()}")
        
        if self.daily_metrics['moves']:
            print(f"Today's moves: {len(self.daily_metrics['moves'])}")
            print(f"Avg distance: {np.mean(self.daily_metrics['distances']):.1f}m")

if __name__ == '__main__':
    """
    Test container terminal generation and natural processes over 60 days.
    Monitors container lifecycle, slot filling, and move availability.
    """
    import matplotlib.pyplot as plt
    from collections import defaultdict
    import pandas as pd
    
    print("="*80)
    print("CONTAINER TERMINAL 60-DAY NATURAL PROCESS TEST")
    print("="*80)
    
    # Create terminal
    terminal = ContainerTerminal(
        n_rows=10,
        n_bays=20,
        n_tiers=5,
        n_railtracks=4,
        split_factor=4,
        max_days=60
    )
    
    # Tracking metrics
    daily_stats = defaultdict(list)
    move_distribution = defaultdict(int)
    container_lifecycle = defaultdict(int)
    hourly_activity = defaultdict(lambda: defaultdict(int))
    
    # Tracking for warnings
    days_without_moves = 0
    min_containers = float('inf')
    max_containers = 0
    
    print("\nStarting 60-day simulation...")
    print("-" * 60)
    
    # Reset environment
    state, info = terminal.reset()
    
    # Run for 60 days
    for day in range(60):
        print(f"\nDay {day + 1}:")
        
        # Daily tracking
        day_start_containers = len(terminal.logistics.yard_container_index)
        day_moves = 0
        move_types_today = defaultdict(int)
        containers_imported = 0
        containers_exported = 0
        trains_processed = 0
        trucks_processed = 0
        
        # Track hourly patterns
        hourly_moves = defaultdict(int)
        
        # Run until day ends
        while terminal.current_day == day:
            # Get current hour
            current_hour = int((terminal.current_time / 3600) % 24)
            
            # Get available moves
            moves = terminal.logistics.find_moves_optimized()
            eligible_moves = terminal.rmgc.mask_moves(moves)
            
            if not eligible_moves:
                # No moves available - advance time
                terminal._advance_time(60.0)  # 1 minute
                
                # Check if this persists
                if terminal.current_time > 3600:  # After 1 hour
                    days_without_moves += 1
                    print(f"  WARNING: No moves available for extended period!")
            else:
                # Categorize available moves
                for move_id, move in eligible_moves.items():
                    move_distribution[move['move_type']] += 1
                
                # Execute random move (simulating agent)
                action = random.randint(0, len(eligible_moves) - 1)
                state, reward, terminated, truncated, info = terminal.step(action)
                
                # Track move
                if terminal.daily_metrics['moves']:
                    last_move = list(eligible_moves.values())[action]
                    move_types_today[last_move['move_type']] += 1
                    hourly_moves[current_hour] += 1
                    hourly_activity[current_hour][last_move['move_type']] += 1
                    day_moves += 1
                    
                    # Track import/export
                    if last_move['move_type'] == 'to_yard':
                        containers_imported += 1
                        container_lifecycle['imported'] += 1
                    elif last_move['source_type'] == 'yard' and last_move['dest_type'] != 'yard':
                        containers_exported += 1
                        container_lifecycle['exported'] += 1
            
            # Count active vehicles periodically
            if terminal.current_time % 3600 < 60:  # Once per hour
                trains_in_terminal = sum(len(trains) for trains in terminal.logistics.trains_on_track.values())
                trucks_in_terminal = len(terminal.logistics.active_trucks)
                
                if trains_in_terminal > trains_processed:
                    trains_processed = trains_in_terminal
                if trucks_in_terminal > trucks_processed:
                    trucks_processed = trucks_in_terminal
        
        # End of day statistics
        day_end_containers = len(terminal.logistics.yard_container_index)
        container_change = day_end_containers - day_start_containers
        
        # Update min/max
        min_containers = min(min_containers, day_end_containers)
        max_containers = max(max_containers, day_end_containers)
        
        # Calculate yard utilization
        total_slots = terminal.n_rows * terminal.n_bays * terminal.n_tiers
        utilization = (day_end_containers / total_slots) * 100
        
        # Store daily stats
        daily_stats['day'].append(day + 1)
        daily_stats['containers_in_yard'].append(day_end_containers)
        daily_stats['container_change'].append(container_change)
        daily_stats['total_moves'].append(day_moves)
        daily_stats['imports'].append(containers_imported)
        daily_stats['exports'].append(containers_exported)
        daily_stats['utilization'].append(utilization)
        daily_stats['trains'].append(trains_processed)
        daily_stats['trucks'].append(trucks_processed)
        
        # Print summary
        print(f"  Containers: {day_start_containers} → {day_end_containers} ({container_change:+d})")
        print(f"  Moves: {day_moves} (Import: {containers_imported}, Export: {containers_exported})")
        print(f"  Vehicles: {trains_processed} trains, {trucks_processed} trucks")
        print(f"  Yard utilization: {utilization:.1f}%")
        
        # Move breakdown
        if move_types_today:
            print("  Move types:", end="")
            for mtype, count in sorted(move_types_today.items()):
                print(f" {mtype}:{count}", end="")
            print()
        
        # Warnings
        if day_end_containers < 10:
            print("  ⚠️  CRITICAL: Very low container count!")
        elif day_end_containers < 30:
            print("  ⚠️  WARNING: Low container count")
        
        if day_moves < 50:
            print("  ⚠️  WARNING: Low activity level")
        
        if abs(containers_imported - containers_exported) > 20:
            print("  ⚠️  WARNING: Import/Export imbalance")
    
    # Final Analysis
    print("\n" + "="*80)
    print("60-DAY SIMULATION ANALYSIS")
    print("="*80)
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(daily_stats)
    
    print("\nContainer Population Statistics:")
    print(f"  Average containers in yard: {df['containers_in_yard'].mean():.1f}")
    print(f"  Min containers: {min_containers}")
    print(f"  Max containers: {max_containers}")
    print(f"  Standard deviation: {df['containers_in_yard'].std():.1f}")
    
    print("\nActivity Statistics:")
    print(f"  Total moves: {df['total_moves'].sum()}")
    print(f"  Average moves per day: {df['total_moves'].mean():.1f}")
    print(f"  Total imports: {df['imports'].sum()}")
    print(f"  Total exports: {df['exports'].sum()}")
    print(f"  Import/Export ratio: {df['imports'].sum() / max(1, df['exports'].sum()):.2f}")
    
    print("\nMove Type Distribution:")
    total_moves = sum(move_distribution.values())
    for move_type, count in sorted(move_distribution.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_moves) * 100
        print(f"  {move_type}: {count} ({percentage:.1f}%)")
    
    print("\nYard Utilization:")
    print(f"  Average utilization: {df['utilization'].mean():.1f}%")
    print(f"  Min utilization: {df['utilization'].min():.1f}%")
    print(f"  Max utilization: {df['utilization'].max():.1f}%")
    
    print("\nNatural Process Health:")
    # Check if system is self-sustaining
    if days_without_moves > 0:
        print(f"  ❌ System had {days_without_moves} days with move shortages")
    else:
        print(f"  ✅ System maintained continuous operations")
    
    # Check container stability
    container_variance = df['containers_in_yard'].std() / df['containers_in_yard'].mean()
    if container_variance < 0.3:
        print(f"  ✅ Container population stable (CV: {container_variance:.2f})")
    else:
        print(f"  ⚠️  Container population unstable (CV: {container_variance:.2f})")
    
    # Check import/export balance
    total_imports = df['imports'].sum()
    total_exports = df['exports'].sum()
    imbalance = abs(total_imports - total_exports) / max(total_imports, total_exports)
    if imbalance < 0.1:
        print(f"  ✅ Import/Export well balanced ({imbalance*100:.1f}% difference)")
    else:
        print(f"  ⚠️  Import/Export imbalanced ({imbalance*100:.1f}% difference)")
    
    # Hourly activity patterns
    print("\nHourly Activity Patterns:")
    hourly_totals = {}
    for hour in range(24):
        total = sum(hourly_activity[hour].values())
        hourly_totals[hour] = total
    
    peak_hours = sorted(hourly_totals.items(), key=lambda x: x[1], reverse=True)[:3]
    print(f"  Peak hours: {', '.join([f'{h}:00 ({c} moves)' for h, c in peak_hours])}")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Container Terminal 60-Day Natural Process Analysis', fontsize=16)
    
    # 1. Container population over time
    axes[0, 0].plot(df['day'], df['containers_in_yard'], 'b-', linewidth=2)
    axes[0, 0].axhline(y=df['containers_in_yard'].mean(), color='r', linestyle='--', label='Average')
    axes[0, 0].fill_between(df['day'], min_containers, max_containers, alpha=0.2)
    axes[0, 0].set_xlabel('Day')
    axes[0, 0].set_ylabel('Containers in Yard')
    axes[0, 0].set_title('Container Population')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Daily moves
    axes[0, 1].bar(df['day'], df['total_moves'], color='green', alpha=0.7)
    axes[0, 1].set_xlabel('Day')
    axes[0, 1].set_ylabel('Number of Moves')
    axes[0, 1].set_title('Daily Move Activity')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Import/Export balance
    width = 0.35
    x = np.arange(len(df['day']))
    axes[0, 2].bar(x - width/2, df['imports'], width, label='Imports', color='blue', alpha=0.7)
    axes[0, 2].bar(x + width/2, df['exports'], width, label='Exports', color='red', alpha=0.7)
    axes[0, 2].set_xlabel('Day')
    axes[0, 2].set_ylabel('Containers')
    axes[0, 2].set_title('Import/Export Flow')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Yard utilization
    axes[1, 0].plot(df['day'], df['utilization'], 'g-', linewidth=2)
    axes[1, 0].axhline(y=50, color='orange', linestyle='--', label='50% Target')
    axes[1, 0].fill_between(df['day'], 0, df['utilization'], alpha=0.3, color='green')
    axes[1, 0].set_xlabel('Day')
    axes[1, 0].set_ylabel('Utilization %')
    axes[1, 0].set_title('Yard Utilization')
    axes[1, 0].set_ylim(0, 100)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Container change rate
    axes[1, 1].plot(df['day'], df['container_change'], 'purple', linewidth=2)
    axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    positive_changes = df[df['container_change'] > 0]['container_change'].sum()
    negative_changes = abs(df[df['container_change'] < 0]['container_change'].sum())
    axes[1, 1].fill_between(df['day'], 0, df['container_change'], 
                           where=(df['container_change'] >= 0), alpha=0.3, color='green', label=f'+{positive_changes}')
    axes[1, 1].fill_between(df['day'], 0, df['container_change'], 
                           where=(df['container_change'] < 0), alpha=0.3, color='red', label=f'-{negative_changes}')
    axes[1, 1].set_xlabel('Day')
    axes[1, 1].set_ylabel('Daily Change')
    axes[1, 1].set_title('Container Population Change')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Move type distribution pie chart
    move_types = list(move_distribution.keys())
    move_counts = list(move_distribution.values())
    axes[1, 2].pie(move_counts, labels=move_types, autopct='%1.1f%%', startangle=90)
    axes[1, 2].set_title('Move Type Distribution')
    
    plt.tight_layout()
    plt.savefig('terminal_60day_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print final verdict
    print("\n" + "="*80)
    print("FINAL VERDICT:")
    if (days_without_moves == 0 and 
        container_variance < 0.3 and 
        imbalance < 0.15 and
        df['containers_in_yard'].min() > 20):
        print("✅ PASS: Terminal demonstrates healthy natural processes!")
        print("   The system is self-sustaining without artificial intervention.")
    else:
        print("❌ FAIL: Terminal requires artificial intervention to maintain operations.")
        print("   Natural processes alone are insufficient for stable operation.")
    
    print("\nRecommendations:")
    if days_without_moves > 0:
        print("  - Adjust import/export ratios to prevent move shortages")
    if container_variance > 0.3:
        print("  - Implement better balance between arrivals and departures")
    if imbalance > 0.15:
        print("  - Fine-tune vehicle generation to balance import/export flows")
    if df['containers_in_yard'].min() < 30:
        print("  - Increase safety buffer of containers in yard")
    
    print("="*80)