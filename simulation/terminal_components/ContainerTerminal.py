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
    Implements day-by-day operations with efficient state representation and reward calculation.
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
        self.distance_reward_scale = 5.0 / self.max_distance  # Maps to [0, -5]
        
        # MOVED: Add temporal awareness components BEFORE _init_spaces()
        self.temporal_encoder = TemporalStateEnhancement(self.time_per_day)
        self.reward_calculator = ProgressBasedRewardCalculator(distance_weight=0.1)
        
        # Additional tracking for daily operations
        self.daily_goals = {
            'train_departures': [],  # List of (train_id, departure_time)
            'truck_pickups': [],     # List of (truck_id, container_ids)
            'container_movements': defaultdict(list)  # container_id -> required moves
        }
        
        # Move ranking cache
        self.ranked_moves_cache = None
        self.cache_timestamp = -1
        
        # Action and observation spaces (NOW temporal_encoder exists)
        self._init_spaces()
        
        # Load KDE models for generation
        self.kde_models = self._load_kde_models()
        
        # Container tracking for multi-day operations
        self.container_arrival_times = {}
        self.container_pickup_schedules = {}
        
        # Daily generation parameters
        self.trains_per_day_range = (3, 8)
        self.trucks_per_day_range = (20, 40)
        self.containers_per_train_range = (15, 30)

        # History of moves to print out
        self.move_history = []
    def _init_terminal_components(self):
        """Initialize yard, logistics, and RMGC components."""
        # Define special storage areas
        coordinates = [
            # Reefer areas on ends
            (1, 1, "r"), (1, 2, "r"), (self.n_bays, 1, "r"), (self.n_bays, 2, "r"),
            # Dangerous goods in middle
            (self.n_bays//2, self.n_rows//2, "dg"), (self.n_bays//2+1, self.n_rows//2, "dg"),
            # Swap bodies/trailers near trucks
            (5, 1, "sb_t"), (6, 1, "sb_t"), (7, 1, "sb_t")
        ]
        
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
        
        # Initialize logistics
        self.logistics = BooleanLogistics(
            n_rows=self.n_rows,
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
            heads=2,  # Two crane heads for efficiency
            trolley_speed=70.0,
            hoisting_speed=28.0,
            gantry_speed=130.0
        )
        
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
        
        # Get base directory
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_dir = os.path.join(base_dir, 'data', 'models')
        
        for model_name, file_name in model_files.items():
            file_path = os.path.join(models_dir, file_name)
            try:
                if os.path.exists(file_path):
                    kde_models[model_name] = load_kde_model(file_path)
                    print(f"✓ Loaded KDE model: {model_name}")
                else:
                    print(f"⚠ KDE model file not found: {file_path}")
                    kde_models[model_name] = None
            except Exception as e:
                print(f"✗ Error loading {model_name} KDE model: {e}")
                kde_models[model_name] = None
                
        # Print summary
        loaded_count = sum(1 for m in kde_models.values() if m is not None)
        print(f"Loaded {loaded_count}/{len(model_files)} KDE models successfully")
                
        return kde_models
    
    def _init_spaces(self):
        """Initialize action and observation spaces with temporal features."""
        # Original state dimensions
        yard_state_dim = 5 * self.n_rows * self.n_bays * self.n_tiers * self.split_factor
        rail_state_dim = self.n_rows * self.split_factor * self.n_railtracks
        parking_state_dim = self.n_rows * self.split_factor
        queue_dim = 4
        time_dim = 3
        
        # Add temporal features
        temporal_dim = self.temporal_encoder.feature_dim
        
        total_state_dim = (yard_state_dim + rail_state_dim + parking_state_dim + 
                        queue_dim + time_dim + temporal_dim)
        
        print(f"State dimension breakdown:")
        print(f"  Yard: {yard_state_dim}")
        print(f"  Rail: {rail_state_dim}")
        print(f"  Parking: {parking_state_dim}")
        print(f"  Queue: {queue_dim}")
        print(f"  Time: {time_dim}")
        print(f"  Temporal: {temporal_dim} (NEW)")
        print(f"  Total: {total_state_dim}")
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_state_dim,),
            dtype=np.float32
        )
        
        # Action space remains the same
        max_moves = 1000
        self.action_space = spaces.Discrete(max_moves)
        
    def _calculate_max_distance(self) -> float:
        """Calculate maximum possible distance in terminal."""
        # Approximate maximum Manhattan distance
        max_x = self.n_bays * 12.192  # storage length
        max_y = self.n_rows * 2.44  # storage width
        max_z = self.rmgc.max_height
        return max_x + max_y + max_z
        
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset environment for new episode."""
        super().reset(seed=seed)
        
        # Clear all components
        self.yard = BooleanStorageYard(
            n_rows=self.n_rows,
            n_bays=self.n_bays,
            n_tiers=self.n_tiers,
            coordinates=[
                (1, 1, "r"), (1, 2, "r"), (self.n_bays, 1, "r"), (self.n_bays, 2, "r"),
                (self.n_bays//2, self.n_rows//2, "dg"), (self.n_bays//2+1, self.n_rows//2, "dg"),
                (5, 1, "sb_t"), (6, 1, "sb_t"), (7, 1, "sb_t")
            ],
            split_factor=self.split_factor,
            validate=False,
            device=self.device
        )
        
        self.logistics = BooleanLogistics(
            n_rows=self.n_rows,
            n_railtracks=self.n_railtracks,
            split_factor=self.split_factor,
            yard=self.yard,
            validate=False,
            device=self.device
        )
        
        self.rmgc = RMGC_Controller(
            yard=self.yard,
            logistics=self.logistics,
            heads=2
        )
        
        # Reset time
        self.current_time = 0.0
        self.current_day = 0
        self.crane_completion_times = np.zeros(self.rmgc.heads)
        
        # Reset tracking
        self.container_arrival_times.clear()
        self.container_pickup_schedules.clear()
        self.daily_metrics.clear()
        self.episode_metrics.clear()
        
        # Generate initial day
        self._generate_daily_traffic()
        
        # Get initial state
        state = self._get_state()
        info = self._get_info()
        
        return state, info
        
    def _generate_daily_traffic(self):
        """Generate trains, trucks, and containers for the day using KDE models."""
        base_date = datetime(2025, 1, 1) + timedelta(days=self.current_day)
        
        # Generate train arrivals
        n_trains = random.randint(*self.trains_per_day_range)
        
        # On first day, pre-populate yard with some containers
        if self.current_day == 0:
            print("Initializing yard with containers...")
            n_initial_containers = 50  # Start with some containers in yard
            
            for i in range(n_initial_containers):
                container = self._create_container_with_weight(f"INIT_{i:04d}")
                container.direction = "Export"  # Available for pickup
                container.arrival_date = base_date - timedelta(days=random.randint(1, 5))
                
                # Place in yard
                self._place_container_in_yard(container)
                
                # Schedule pickup in next few days
                pickup_time = self._sample_pickup_time(base_date)
                self.container_pickup_schedules[container.container_id] = pickup_time

        # Sample train arrival times using KDE
        if self.kde_models.get('train_arrival'):
            # Sample arrival hours
            arrival_hours = sample_from_kde(self.kde_models['train_arrival'], n_samples=n_trains)
            
            # Sample delays
            if self.kde_models.get('train_delay'):
                delays = sample_from_kde(self.kde_models['train_delay'], n_samples=n_trains, min_val=-3, max_val=3)
            else:
                delays = np.random.normal(0, 0.5, n_trains)  # Fallback
        else:
            # Fallback: uniform distribution
            arrival_hours = np.random.uniform(0, 24, n_trains)
            delays = np.random.normal(0, 0.5, n_trains)
        
        # Create trains
        train_arrivals = []
        for i in range(n_trains):
            train_id = f"TRN_{self.current_day:03d}_{i:04d}"
            
            # Convert hours to datetime
            h, m, s = hours_to_time(arrival_hours[i])
            planned_arrival = base_date.replace(hour=h, minute=m, second=s)
            
            # Add delay
            delay_hours = delays[i]
            realized_arrival = planned_arrival + timedelta(hours=delay_hours)
            
            train_arrivals.append((train_id, planned_arrival, realized_arrival))
        
        # Sort by realized arrival time
        train_arrivals.sort(key=lambda x: x[2])
        
        # Process trains in arrival order
        for train_id, planned_arrival, realized_arrival in train_arrivals:
            n_wagons = random.randint(2, 5)
            train = Train(train_id, num_wagons=n_wagons, arrival_time=realized_arrival)
            
            # 70% import trains, 30% export trains
            if random.random() < 0.7:
                # IMPORT TRAIN - brings containers
                n_containers = random.randint(*self.containers_per_train_range)
                for j in range(n_containers):
                    container = self._create_container_with_weight(f"{train_id}_C{j:03d}")
                    container.direction = "Import"
                    container.arrival_date = realized_arrival
                    
                    # Try to add to wagon
                    for wagon in train.wagons:
                        if wagon.add_container(container):
                            # Schedule future pickup by truck
                            pickup_time = self._sample_pickup_time(base_date)
                            self.container_pickup_schedules[container.container_id] = pickup_time
                            break
            else:
                # EXPORT TRAIN - needs existing containers
                n_pickups = random.randint(5, 15)
                
                # Find existing containers in yard without pickup requests
                available_containers = []
                for cid in list(self.logistics.yard_container_set):
                    if (cid not in self.logistics.pickup_to_train and 
                        cid not in self.logistics.pickup_to_truck and
                        not cid.startswith('TRN_')):  # Not already assigned
                        available_containers.append(cid)
                
                # Assign pickups from available containers
                n_actual_pickups = min(n_pickups, len(available_containers))
                if n_actual_pickups > 0:
                    selected = random.sample(available_containers, n_actual_pickups)
                    for container_id in selected:
                        wagon_idx = random.randint(0, len(train.wagons) - 1)
                        train.wagons[wagon_idx].add_pickup_container(container_id)
            
            # Schedule train arrival
            self.logistics.trains.schedule_arrival(train, realized_arrival)
        
        # Generate truck arrivals
        n_trucks = random.randint(*self.trucks_per_day_range)
        
        # Sample truck arrival times using KDE
        if self.kde_models.get('truck_pickup'):
            truck_hours = sample_from_kde(self.kde_models['truck_pickup'], n_samples=n_trucks)
        else:
            truck_hours = np.random.uniform(6, 18, n_trucks)  # Business hours fallback
        
        for i in range(n_trucks):
            truck_id = f"TRK_{self.current_day:03d}_{i:03d}"
            truck = Truck(truck_id)
            
            if random.random() < 0.3:
                # DELIVERY TRUCK - brings new containers
                n_containers = random.randint(1, 2)
                for j in range(n_containers):
                    container = self._create_container_with_weight(f"{truck_id}_C{j}")
                    container.direction = "Import"
                    truck.add_container(container)
                    
                    # These will be placed in yard, available for future trains
            else:
                # PICKUP TRUCK - picks up train-delivered containers
                # Find containers delivered by trains (Import containers in yard)
                available_for_pickup = []
                for cid in list(self.logistics.yard_container_set):
                    if (cid not in self.logistics.pickup_to_truck and 
                        cid.startswith('TRN_') and '_C' in cid):  # Train-delivered
                        available_for_pickup.append(cid)
                
                if available_for_pickup:
                    n_pickups = min(random.randint(1, 3), len(available_for_pickup))
                    selected = random.sample(available_for_pickup, n_pickups)
                    for container_id in selected:
                        truck.add_pickup_container_id(container_id)
            
            # Schedule truck arrival
            h, m, s = hours_to_time(truck_hours[i])
            arrival_time = base_date.replace(hour=h, minute=m, second=s)
            truck.arrival_time = arrival_time
            self.logistics.trucks.schedule_arrival(truck, arrival_time)
        
        # Update logistics to process arrivals
        self.logistics.trains.update(base_date)
        self.logistics.trucks.update(base_date)
        
        # Process queues
        self.logistics.process_current_trains()
        self.logistics.process_current_trucks()
        
        # Sync yard index
        self.logistics.sync_yard_index()
        
        # Update container priorities
        self._update_container_priorities()
        
        # Mark initialization as complete
        self._initialization_complete = True
    
    def _create_container_with_weight(self, container_id: str) -> Container:
        """Create container with weight sampled from KDE."""
        # Use ContainerFactory to create random container
        container = ContainerFactory.create_random(container_id)
        
        # Avoid FFEU containers for now as they cause cross-bay issues
        while container.container_type == "FFEU":
            container = ContainerFactory.create_random(container_id)
        
        # Sample weight from KDE if available
        if self.kde_models.get('container_weight'):
            weight = sample_from_kde(
                self.kde_models['container_weight'], 
                n_samples=1, 
                min_val=1000, 
                max_val=31000
            )[0]
            container.weight = float(weight)
        else:
            # Use factory default weight if KDE not available
            container.weight = ContainerFactory.sample_container_weight(None)
            
        return container
    
    def _sample_pickup_time(self, base_date: datetime) -> datetime:
        """Sample pickup time using wait time KDE."""
        if self.kde_models.get('pickup_wait'):
            # Sample wait time in hours
            wait_hours = sample_from_kde(
                self.kde_models['pickup_wait'], 
                n_samples=1, 
                min_val=0, 
                max_val=168  # Max 1 week
            )[0]
            
            # Calculate pickup date
            wait_days = int(wait_hours // 24)
            pickup_date = base_date + timedelta(days=wait_days)
            
            # Sample pickup hour of day
            if self.kde_models.get('truck_pickup'):
                pickup_hour = sample_from_kde(self.kde_models['truck_pickup'], n_samples=1)[0]
            else:
                pickup_hour = random.uniform(6, 18)
                
            h, m, s = hours_to_time(pickup_hour)
            return pickup_date.replace(hour=h, minute=m, second=s)
        else:
            # Fallback: 1-7 days in future
            days_ahead = random.randint(1, 7)
            hour = random.randint(6, 18)
            return base_date + timedelta(days=days_ahead, hours=hour)
    
    def _place_container_in_yard(self, container: Container) -> bool:
        """Place container in appropriate yard location."""
        goods_type = 'r' if container.goods_type == 'Reefer' else \
                    'dg' if container.goods_type == 'Dangerous' else \
                    'sb_t' if container.container_type in ['Trailer', 'Swap Body'] else 'reg'
        
        # Try multiple bay positions
        for _ in range(5):
            bay = random.randint(2, self.n_bays-3)  # Leave more margin for cross-bay containers
            positions = self.yard.search_insertion_position(
                bay,
                goods_type,
                container.container_type,
                3  # Search proximity
            )
            
            if positions:
                # Get coordinates for the first valid position
                placement = positions[0]
                row, bay, tier, start_split = placement
                
                # For FFEU containers (length 5), ensure we don't go out of bounds
                container_length = self.yard.container_lengths.get(container.container_type, 1)
                if container_length > self.yard.split_factor:
                    # Skip if this would go out of bounds
                    if bay >= self.n_bays - 1:
                        continue
                
                coords = self.yard.get_container_coordinates_from_placement(
                    placement,
                    container.container_type
                )
                
                # Validate ALL coordinates before adding
                valid = True
                for coord in coords:
                    r, b, s, t = coord
                    if not (0 <= r < self.yard.n_rows and 
                           0 <= b < self.yard.n_bays and 
                           0 <= s < self.yard.split_factor and 
                           0 <= t < self.yard.n_tiers):
                        valid = False
                        if not hasattr(self, '_coord_warning_shown'):
                            print(f"Debug: Invalid coord {coord} for {container.container_type} (length={container_length})")
                            self._coord_warning_shown = True
                        break
                
                if valid:
                    success = self.logistics.add_container_to_yard(container, coords)
                    if success:
                        return True
                
        return False
        
    def _update_container_priorities(self):
        """Update priorities for all containers based on time until pickup."""
        current_date = datetime(2025, 1, 1) + timedelta(days=self.current_day, seconds=self.current_time)
        
        # Update yard containers
        for row in range(self.yard.n_rows):
            for bay in range(self.yard.n_bays):
                for tier in range(self.yard.n_tiers):
                    for split in range(self.yard.split_factor):
                        container = self.yard.get_container_at(row, bay, tier, split)
                        if container:
                            container.update_priority()
                            
                            # Additional priority based on pickup schedule
                            if container.container_id in self.container_pickup_schedules:
                                pickup_time = self.container_pickup_schedules[container.container_id]
                                days_until_pickup = (pickup_time - current_date).days
                                
                                if days_until_pickup <= 1:
                                    container.priority = max(1, container.priority - 50)
                                elif days_until_pickup <= 3:
                                    container.priority = max(1, container.priority - 25)
                            
                            # Update priority in the yard arrays
                            self.yard._update_property_arrays(row, bay, tier, split, container)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute action with forced move selection when available."""
        # Get available moves and rank them
        moves = self.logistics.find_moves_optimized()
        eligible_moves = self.rmgc.mask_moves(moves)
        
        # NEW: Rank moves by urgency if cache is stale
        if eligible_moves and self.current_time != self.cache_timestamp:
            self.ranked_moves_cache = self._rank_moves_by_urgency(eligible_moves)
            self.cache_timestamp = self.current_time
        
        # CHANGED: No waiting when moves are available
        if not eligible_moves:
            # No moves available - short wait
            self._advance_time(30.0)  # 30 seconds
            reward = 0.0  # Neutral reward for forced wait
        else:
            # Must select from available moves
            move_list = self.ranked_moves_cache or list(eligible_moves.items())
            
            # If action is str, reformat
            if type(action) == str:
                action = int(str(action).replace('move_', ''))
                # print(action)
            # Ensure action is within bounds
            if action >= len(move_list):
                action = action % len(move_list)
            
            move_id, move = move_list[action]
            
            # Execute move
            distance, time = self.rmgc.execute_move(move)
            
            if time > 0:
                # NEW: Calculate progress-based reward
                current_datetime = datetime(2025, 1, 1) + timedelta(
                    days=self.current_day,
                    seconds=self.current_time
                )
                
                reward = self.reward_calculator.calculate_reward(
                    move=move,
                    distance=distance,
                    time=time,
                    current_datetime=current_datetime,
                    trains=self.logistics.active_trains,
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

                # Track move
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
                # Failed move - small negative reward
                reward = -0.5
        
        # Check and process completed cranes
        self._update_crane_states()
        
        # Reorganize logistics
        departed = self.logistics.reorganize_logistics()
        trains_early, trains_late, trucks = departed
        
        # Bonus/penalty for departures
        if trains_early > 0:
            reward += trains_early * 5.0  # Bonus for on-time departure
        if trains_late > 0:
            reward -= trains_late * 10.0  # Penalty for late trains
        
        # Check if day ended
        day_ended = self.current_time >= self.time_per_day
        
        if day_ended:
            self._end_of_day()
        
        # Check episode termination
        terminated = self.current_day >= self.max_days
        truncated = False
        
        # Get new state
        state = self._get_state()
        info = self._get_info()
        
        # Add ranked moves to info for action masking
        if eligible_moves:
            info['ranked_move_list'] = [m[0] for m in (self.ranked_moves_cache or [])]
        
        return state, reward, terminated, truncated, info
    
    def _rank_moves_by_urgency(self, moves: Dict[str, Dict]) -> List[Tuple[str, Dict]]:
        """
        Rank moves by urgency and value, with distance as tiebreaker.
        Returns list of (move_id, move) tuples sorted by priority.
        """
        current_datetime = datetime(2025, 1, 1) + timedelta(
            days=self.current_day,
            seconds=self.current_time
        )
        
        move_scores = []
        
        for move_id, move in moves.items():
            # Evaluate move urgency and value
            urgency, value = self.reward_calculator.move_evaluator.evaluate_move_urgency(
                move=move,
                current_datetime=current_datetime,
                trains=self.logistics.active_trains,
                trucks=self.logistics.active_trucks,
                container_pickup_schedules=self.container_pickup_schedules,
                logistics=self.logistics
            )
            
            # Get estimated distance for tiebreaking
            try:
                source_str = self.rmgc._position_to_string(move['source_pos'])
                dest_str = self.rmgc._position_to_string(move['dest_pos'])
                
                idx1 = self.rmgc.position_to_idx.get(source_str)
                idx2 = self.rmgc.position_to_idx.get(dest_str)
                
                if idx1 is not None and idx2 is not None:
                    distance = self.rmgc.distance_matrix[idx1, idx2]
                else:
                    distance = 50.0  # Default distance
            except:
                distance = 50.0
            
            # Composite score: urgency * value - small distance factor
            score = (urgency + 1.0) * value - (distance * 0.01)
            
            move_scores.append((score, distance, move_id, move))
        
        # Sort by score (descending), then by distance (ascending) for tiebreaking
        move_scores.sort(key=lambda x: (-x[0], x[1]))
        
        # Return as list of (move_id, move) tuples
        return [(item[2], item[3]) for item in move_scores]

    def _update_daily_goals(self):
        """Update daily goals based on current vehicles and schedules."""
        self.daily_goals['train_departures'].clear()
        self.daily_goals['truck_pickups'].clear()
        self.daily_goals['container_movements'].clear()
        
        # Record train departures
        for railtrack_id, train in self.logistics.active_trains.items():
            if hasattr(train, 'departure_time') and train.departure_time:
                self.daily_goals['train_departures'].append(
                    (train.train_id, train.departure_time)
                )
        
        # Record truck pickup requirements
        for truck_pos, truck in self.logistics.active_trucks.items():
            if hasattr(truck, 'pickup_container_ids') and truck.pickup_container_ids:
                self.daily_goals['truck_pickups'].append(
                    (truck.truck_id, list(truck.pickup_container_ids))
                )
        
        # Track container movements needed
        for container_id, pickup_time in self.container_pickup_schedules.items():
            current_date = datetime(2025, 1, 1) + timedelta(days=self.current_day)
            if current_date <= pickup_time < current_date + timedelta(days=1):
                self.daily_goals['container_movements'][container_id].append({
                    'type': 'pickup',
                    'deadline': pickup_time
                })

    # Override the old _calculate_reward method
    def _calculate_reward(self, move: Dict, distance: float, time: float) -> float:
        """This method is replaced by the ProgressBasedRewardCalculator."""
        raise NotImplementedError("Use reward_calculator.calculate_reward() instead")
    
    def _advance_time(self, seconds: float):
        """Advance simulation time."""
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
            self.logistics.process_current_trains()
            self.logistics.process_current_trucks()
            self.logistics.sync_yard_index()
    
    def _update_crane_states(self):
        """Update crane head states based on completion times."""
        for head_id, completion_time in enumerate(self.crane_completion_times):
            if completion_time > 0 and self.current_time >= completion_time:
                # Crane completed its task
                self.rmgc.unlock_head(head_id)
                self.crane_completion_times[head_id] = 0.0
    
    def _get_crane_head_for_move(self, move: Dict) -> Optional[int]:
        """Get which crane head is executing a move."""
        for i, head in enumerate(self.rmgc.crane_heads):
            if head['busy'] and head['current_move'] == move:
                return i
        return None
    
    def _find_container(self, container_id: str) -> Optional[Container]:
        """Find container in yard, trains, or trucks."""
        # Check yard
        if container_id in self.logistics.yard_container_index:
            pos = self.logistics.yard_container_index[container_id]
            # Validate position bounds
            row, bay, tier, split = pos
            if (0 <= row < self.yard.n_rows and 
                0 <= bay < self.yard.n_bays and 
                0 <= tier < self.yard.n_tiers and 
                0 <= split < self.yard.split_factor):
                return self.yard.get_container_at(row, bay, tier, split)
            else:
                # Remove invalid entry silently
                self.logistics.yard_container_index.pop(container_id, None)
                self.logistics.yard_container_set.discard(container_id)
        
        # Check trains
        for train in self.logistics.active_trains.values():
            for wagon in train.wagons:
                for container in wagon.containers:
                    if container.container_id == container_id:
                        return container
        
        # Check trucks
        for truck in self.logistics.active_trucks.values():
            for container in truck.containers:
                if container.container_id == container_id:
                    return container
        
        return None
    
    def _end_of_day(self):
        """Process end of day tasks."""
        # Calculate daily statistics
        if self.daily_metrics['moves']:
            avg_distance = np.mean(self.daily_metrics['distances'])
            avg_time = np.mean(self.daily_metrics['times'])
            total_reward = sum(self.daily_metrics['rewards'])
            
            self.episode_metrics['total_moves'] += len(self.daily_metrics['moves'])
            self.episode_metrics['total_reward'] += total_reward
            self.episode_metrics['avg_distance'] = \
                (self.episode_metrics.get('avg_distance', 0) * self.current_day + avg_distance) / (self.current_day + 1)
        
        # Clear daily metrics
        self.daily_metrics.clear()
        
        # Advance to next day
        self.current_day += 1
        self.current_time = 0.0
        
        # Generate new day's traffic
        if self.current_day < self.max_days:
            self._generate_daily_traffic()

        # Write out history and clear
        if hasattr(self, 'move_log_file') and self.move_history:
            import csv
            with open(self.move_log_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['day', 'move_id', 'container_id', 
                                                    'source_type', 'source_pos', 'dest_type', 
                                                    'dest_pos', 'move_type', 'reward', 
                                                    'distance', 'time'])
                if self.current_day == 0:  # Write header on first day
                    writer.writeheader()
                writer.writerows(self.move_history)
            self.move_history.clear()
    
    def _get_state(self) -> np.ndarray:
        """Get current environment state with temporal awareness."""
        # Original state components
        yard_state = self.yard.get_full_state_tensor(flatten=True)
        rail_state = self.logistics.get_rail_state_tensor(as_tensor=False).flatten()
        parking_state = self.logistics.get_parking_state_tensor(as_tensor=False).flatten()
        
        # Queue information
        queue_state = np.array([
            self.logistics.trains.size(),
            self.logistics.trucks.size(),
            len(self.logistics.active_trains),
            len(self.logistics.active_trucks)
        ], dtype=np.float32)
        
        # Time information
        time_state = np.array([
            self.current_time / self.time_per_day,
            self.current_day / self.max_days,
            (self.time_per_day - self.current_time) / self.time_per_day
        ], dtype=np.float32)
        
        # NEW: Temporal schedule features
        temporal_features = self.temporal_encoder.encode_daily_schedule(
            current_time=self.current_time,
            current_day=self.current_day,
            trains=self.logistics.active_trains,
            trucks=self.logistics.active_trucks,
            train_queue=self.logistics.trains,
            truck_queue=self.logistics.trucks,
            container_pickup_schedules=self.container_pickup_schedules,
            container_arrival_times=self.container_arrival_times
        )
        
        # Concatenate all states
        state = np.concatenate([
            yard_state.cpu().numpy(),
            rail_state,
            parking_state,
            queue_state,
            time_state,
            temporal_features  # NEW
        ])
        
        return state.astype(np.float32)
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional environment information."""
        # Get available moves for action masking
        moves = self.logistics.find_moves_optimized()
        eligible_moves = self.rmgc.mask_moves(moves)
        
        return {
            'day': self.current_day,
            'time': self.current_time,
            'available_moves': len(eligible_moves),
            'move_list': list(eligible_moves.keys()),
            'trains_in_terminal': len(self.logistics.active_trains),
            'trucks_in_terminal': len(self.logistics.active_trucks),
            'containers_in_yard': len(self.logistics.yard_container_index),
            'episode_metrics': dict(self.episode_metrics)
        }
    
    def render(self):
        """Render environment state (text-based)."""
        print(f"\n=== Day {self.current_day}, Time: {self.current_time:.0f}s ===")
        print(f"Trains: {len(self.logistics.active_trains)}, " 
              f"Trucks: {len(self.logistics.active_trucks)}, "
              f"Yard containers: {len(self.logistics.yard_container_index)}")
        print(f"Train queue: {self.logistics.trains.size()}, "
              f"Truck queue: {self.logistics.trucks.size()}")
        
        if self.daily_metrics['moves']:
            print(f"Today's moves: {len(self.daily_metrics['moves'])}, "
                  f"Avg distance: {np.mean(self.daily_metrics['distances']):.1f}m")