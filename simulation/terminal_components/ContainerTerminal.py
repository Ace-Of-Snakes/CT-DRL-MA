# simulation/terminal_components/ContainerTerminal.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Set
from datetime import datetime, timedelta
from collections import defaultdict, deque
import random

from simulation.terminal_components.BooleanStorage import BooleanStorageYard
from simulation.terminal_components.BooleanLogistics import BooleanLogistics
from simulation.terminal_components.Container import Container, ContainerFactory
from simulation.terminal_components.Train import Train
from simulation.terminal_components.Truck import Truck
from simulation.terminal_components.TemporalState import (
    TemporalStateEnhancement, 
    MoveEvaluator, 
    ProgressBasedRewardCalculator
)
from simulation.TerminalConfig import TerminalConfig
import logging

logger = logging.getLogger(__name__)


class ContainerTerminal(gym.Env):
    """
    Gymnasium environment for container terminal simulation.
    
    This environment simulates a container terminal with trains, trucks, and containers
    moving through the system based on realistic KDE distributions.
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}
    
    def __init__(
        self,
        n_rows: int = 5,
        n_bays: int = 20,
        n_tiers: int = 4,
        n_railtracks: int = 4,
        split_factor: int = 4,
        max_days: int = 365,
        config_path: str = None,
        seed: int = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        validate: bool = False
    ):
        super().__init__()
        
        # Terminal configuration
        self.n_rows = n_rows
        self.n_bays = n_bays
        self.n_tiers = n_tiers
        self.n_railtracks = n_railtracks
        self.split_factor = split_factor
        self.max_days = max_days
        self.device = device
        self.validate = validate

        # Set random seed
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            random.seed(seed)
        
        # Load configuration with KDE models
        self.config = TerminalConfig(config_path)
        
        # Time tracking
        self.seconds_per_day = 86400  # 24 hours
        self.current_time = 0.0  # Seconds since start of current day
        self.current_day = 0
        self.current_datetime = datetime(2025, 1, 1)  # Base date
        
        # Initialize yard with special zones
        self._init_yard_zones()
        
        # Create yard and logistics
        self.yard = BooleanStorageYard(
            n_rows=n_rows,
            n_bays=n_bays,
            n_tiers=n_tiers,
            coordinates=self.yard_zones,
            split_factor=split_factor,
            device=device,
            validate=False
        )
        
        self.logistics = BooleanLogistics(
            n_rows=n_rows,
            n_bays=n_bays,
            n_railtracks=n_railtracks,
            split_factor=split_factor,
            yard=self.yard,
            validate=False,
            device=device
        )
        
        # Temporal components
        self.temporal_encoder = TemporalStateEnhancement(time_per_day=self.seconds_per_day)
        self.move_evaluator = MoveEvaluator()
        self.reward_calculator = ProgressBasedRewardCalculator(distance_weight=0.1)
        
        # Container lifecycle tracking
        self.container_arrival_times = {}  # container_id -> arrival datetime
        self.container_pickup_schedules = {}  # container_id -> scheduled pickup datetime
        self.container_destinations = {}  # container_id -> 'train_id' or 'truck_id'
        
        # Vehicle tracking
        self.scheduled_trains = []  # List of (arrival_time, train) tuples
        self.scheduled_trucks = []  # List of (arrival_time, truck) tuples
        self.active_trains = {}  # train_id -> train object
        self.active_trucks = {}  # truck_id -> truck object
        
        # Metrics tracking
        self.episode_metrics = defaultdict(lambda: 0)
        self.daily_metrics = defaultdict(list)
        
        # Move history
        self.move_history = deque(maxlen=1000)
        self.move_log_file = None
        
        # Define action and observation spaces
        self._define_spaces()
        
        # Initialize environment
        self.reset()
    
    def _init_yard_zones(self):
        """Initialize yard zones for different container types."""
        self.yard_zones = []
        
        # Reefers on both ends (bays 1-5 and last 5 bays)
        for row in range(1, 6):
            for bay in [1, 2, 3, 4, 5]:
                self.yard_zones.append((bay, row, "r"))
            for bay in range(self.n_bays - 4, self.n_bays + 1):
                self.yard_zones.append((bay, row, "r"))
        
        # Swap bodies/trailers nearest to trucks (row 1)
        for bay in range(1, self.n_bays + 1):
            self.yard_zones.append((bay, 1, "sb_t"))
        
        # Dangerous goods in middle
        middle_bay = self.n_bays // 2
        for offset in [-1, 0, 1]:
            for row in range(3, min(6, self.n_rows + 1)):
                self.yard_zones.append((middle_bay + offset, row, "dg"))
    
    def _define_spaces(self):
        """Define action and observation spaces."""
        # Action space: index of move to execute (dynamic size)
        # We use a large discrete space and mask invalid actions
        self.action_space = spaces.Discrete(10000)
        
        # Observation space components
        obs_components = []
        
        # Yard state tensors
        yard_state_size = self.n_rows * self.n_bays * self.n_tiers * self.split_factor * 5
        obs_components.append(yard_state_size)  # Yard tensor (flattened)
        
        # Rail and parking states
        rail_state_size = self.n_railtracks * self.n_bays * self.split_factor
        parking_state_size = self.n_bays * self.split_factor
        obs_components.append(rail_state_size + parking_state_size)
        
        # Vehicle properties
        train_props_size = self.n_railtracks * 3  # trains, containers, pickups per track
        truck_props_size = parking_state_size * 3  # containers, pickups, weights per spot
        obs_components.append(train_props_size + truck_props_size)
        
        # Queue information
        obs_components.append(2)  # train queue size, truck queue size
        
        # Temporal features
        obs_components.append(self.temporal_encoder.feature_dim)
        
        # Total observation size
        total_obs_size = sum(obs_components)
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_obs_size,),
            dtype=np.float32
        )
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        # Reset time
        self.current_time = 0.0
        self.current_day = 0
        self.current_datetime = datetime(2025, 1, 1)
        
        # Clear tracking
        self.container_arrival_times.clear()
        self.container_pickup_schedules.clear()
        self.container_destinations.clear()
        self.scheduled_trains.clear()
        self.scheduled_trucks.clear()
        self.active_trains.clear()
        self.active_trucks.clear()
        self.episode_metrics.clear()
        self.daily_metrics.clear()
        self.move_history.clear()
        
        # Reset yard and logistics
        self.yard = BooleanStorageYard(
            n_rows=self.n_rows,
            n_bays=self.n_bays,
            n_tiers=self.n_tiers,
            coordinates=self.yard_zones,
            split_factor=self.split_factor,
            device=self.device,
            validate=False
        )
        
        self.logistics = BooleanLogistics(
            n_rows=self.n_rows,
            n_bays=self.n_bays,
            n_railtracks=self.n_railtracks,
            split_factor=self.split_factor,
            yard=self.yard,
            validate=False,
            device=self.device
        )
        
        # Initialize yard with 30% capacity
        self._initialize_yard()
        
        # Generate initial day's schedule
        self._generate_daily_schedule()
        
        # Place some vehicles immediately for initial moves
        # Schedule some arrivals at time 0
        for i in range(min(2, len(self.scheduled_trains))):
            if self.scheduled_trains[i][0].hour < 6:  # Early morning trains
                self.scheduled_trains[i] = (self.current_datetime, self.scheduled_trains[i][1])
        
        for i in range(min(4, len(self.scheduled_trucks))):
            if self.scheduled_trucks[i][0].hour < 6:  # Early morning trucks
                self.scheduled_trucks[i] = (self.current_datetime, self.scheduled_trucks[i][1])
        
        # Process initial arrivals
        self._process_arrivals()
        
        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def _initialize_yard(self):
        """Initialize yard with 30% capacity."""
        target_containers = int(self.n_rows * self.n_bays * self.n_tiers * 0.3)
        containers_placed = 0
        
        # Generate variety of container types
        container_types = ['TWEU', 'THEU', 'FEU', 'FFEU', 'Swap Body', 'Trailer']
        goods_types = ['Regular', 'Reefer', 'Dangerous']
        
        attempts = 0
        max_attempts = target_containers * 3
        
        while containers_placed < target_containers and attempts < max_attempts:
            attempts += 1
            
            # Random container properties
            container_type = random.choice(container_types)
            
            # Match goods type to yard zones
            if container_type in ['Swap Body', 'Trailer']:
                goods_type = 'Regular'
                search_type = 'sb_t'
            else:
                if random.random() < 0.05:  # 5% dangerous
                    goods_type = 'Dangerous'
                    search_type = 'dg'
                elif random.random() < 0.1:  # 10% reefer
                    goods_type = 'Reefer'
                    search_type = 'r'
                else:
                    goods_type = 'Regular'
                    search_type = 'reg'
            
            # Fixed weight sampling
            weight = None
            if self.config and 'container_weight' in self.config.kde_models:
                weight_samples = self.config.sample_from_kde('container_weight', 1, 1000, 31000)
                if weight_samples is not None and len(weight_samples) > 0:
                    weight = float(weight_samples[0])
            
            # Create container
            container_id = f"INIT_{containers_placed:06d}"
            container = ContainerFactory.create_container(
                container_id,
                container_type,
                direction=random.choice(['Import', 'Export']),
                goods_type=goods_type,
                arrival_date=self.current_datetime - timedelta(days=random.randint(0, 5)),
                weight=weight,
                config=self.config
            )
            
            # Find placement
            bay = random.randint(0, self.n_bays - 1)
            positions = self.yard.search_insertion_position(bay, search_type, container_type, max_proximity=5)
            
            if positions:
                placement = positions[0]
                coords = self.yard.get_container_coordinates_from_placement(placement, container_type)
                
                try:
                    self.yard.add_container(container, coords)
                    self.logistics._update_yard_container_index(container_id, coords[0])
                    
                    # Track container
                    self.container_arrival_times[container_id] = container.arrival_date
                    
                    # Schedule pickup based on wait time distribution
                    if self.config and 'pickup_wait' in self.config.kde_models:
                        wait_samples = self.config.sample_from_kde('pickup_wait', 1, 0, 168)
                        if wait_samples is not None and len(wait_samples) > 0:
                            wait_hours = float(wait_samples[0])
                            pickup_time = self.current_datetime + timedelta(hours=wait_hours)
                            self.container_pickup_schedules[container_id] = pickup_time
                    
                    containers_placed += 1
                except:
                    continue
        
        # Sync yard index
        self.logistics.sync_yard_index()
        logger.info(f"Initialized yard with {containers_placed} containers ({containers_placed/target_containers*100:.1f}% of target)")
    
    def _generate_daily_schedule(self):
        """Generate trains and trucks for the current day using KDE distributions."""
        # Clear previous day's unprocessed schedules
        self.scheduled_trains = [
            (time, train) for time, train in self.scheduled_trains 
            if time.date() >= self.current_datetime.date()
        ]
        self.scheduled_trucks = [
            (time, truck) for time, truck in self.scheduled_trucks 
            if time.date() >= self.current_datetime.date()
        ]
        
        # Generate trains
        self._generate_trains()
        
        # Generate trucks
        self._generate_trucks()
        
        # Sort schedules
        self.scheduled_trains.sort(key=lambda x: x[0])
        self.scheduled_trucks.sort(key=lambda x: x[0])
    
    def _generate_trains(self):
        """Generate trains for the day."""
        trains_per_track = 4  # ~4 trains per track per day
        
        for track_id in range(self.n_railtracks):
            track_trains = []
            
            # Generate arrival times using KDE
            if self.config and 'train_arrival' in self.config.kde_models:
                arrival_hours = self.config.sample_from_kde(
                    'train_arrival', 
                    n_samples=trains_per_track,
                    min_val=0,
                    max_val=24
                )
            else:
                # Fallback: uniform distribution
                arrival_hours = np.random.uniform(0, 24, trains_per_track)
            
            # Sort arrival times
            arrival_hours.sort()
            
            for i, arrival_hour in enumerate(arrival_hours):
                # Create arrival datetime
                arrival_time = self.current_datetime.replace(
                    hour=int(arrival_hour),
                    minute=int((arrival_hour % 1) * 60),
                    second=0,
                    microsecond=0
                )
                
                # Stay duration: 4-6 hours (normal distribution)
                stay_hours = np.random.normal(5, 0.5)
                stay_hours = np.clip(stay_hours, 4, 6)
                departure_time = arrival_time + timedelta(hours=stay_hours)
                
                # Check for overlaps and determine wagon count
                max_wagons = self._calculate_max_wagons(track_id, arrival_time, departure_time, track_trains)
                
                if max_wagons < 3:  # Need at least 3 wagons (changed from 2)
                    continue
                
                # Create train - ensure valid range for randint
                num_wagons = random.randint(3, max(3, min(10, max_wagons)))  # Fixed logic
                train_id = f"TRN_{self.current_day:03d}_{track_id:02d}_{i:02d}"
                
                train = Train(
                    train_id=train_id,
                    num_wagons=num_wagons,
                    wagon_length=24.384,
                    arrival_time=arrival_time,
                    departure_time=departure_time
                )
                
                # Add to track schedule
                track_trains.append((arrival_time, departure_time, train))
                
                # Generate containers for train
                self._generate_train_containers(train)
                
                # Schedule train
                self.scheduled_trains.append((arrival_time, train))
    
    def _calculate_max_wagons(self, track_id: int, arrival: datetime, departure: datetime, 
                             track_trains: List[Tuple[datetime, datetime, Any]]) -> int:
        """Calculate maximum wagons for a train given track constraints."""
        track_length = self.n_bays * self.split_factor
        wagon_length = self.logistics.wagon_length  # From BooleanLogistics
        head_length = self.logistics.train_head_length
        
        # Find overlapping trains
        overlapping_trains = []
        for arr, dep, train in track_trains:
            if not (departure <= arr or arrival >= dep):  # Overlap exists
                overlapping_trains.append(train)
        
        if not overlapping_trains:
            # No overlap - can use 70%+ of track
            max_positions = int(track_length * 0.8)
            return (max_positions - head_length) // wagon_length
        else:
            # Must share track
            total_heads = len(overlapping_trains) + 1
            total_head_space = total_heads * head_length
            available_space = track_length - total_head_space
            
            # Divide space fairly
            space_per_train = available_space // total_heads
            return max(0, space_per_train // wagon_length)
    
    def _generate_train_containers(self, train: Train):
        """Generate containers for a train's wagons."""
        for wagon_idx, wagon in enumerate(train.wagons):
            if random.random() < 0.5:  # Import containers
                # Generate realistic container combinations
                combo_type = random.choice(['4tweu', 'feu_2tweu', 'theu_ffeu', 'mixed'])
                
                if combo_type == '4tweu':
                    # 4 TWEUs
                    for _ in range(4):
                        container = self._create_import_container(source='train', force_type='TWEU')
                        if not wagon.add_container(container):
                            break
                        self.container_arrival_times[container.container_id] = train.arrival_time
                        
                elif combo_type == 'feu_2tweu':
                    # 1 FEU + 2 TWEUs
                    container = self._create_import_container(source='train', force_type='FEU')
                    wagon.add_container(container)
                    self.container_arrival_times[container.container_id] = train.arrival_time
                    
                    for _ in range(2):
                        container = self._create_import_container(source='train', force_type='TWEU')
                        if not wagon.add_container(container):
                            break
                        self.container_arrival_times[container.container_id] = train.arrival_time
                        
                elif combo_type == 'theu_ffeu':
                    # 1 THEU + 1 FFEU
                    container = self._create_import_container(source='train', force_type='THEU')
                    wagon.add_container(container)
                    self.container_arrival_times[container.container_id] = train.arrival_time
                    
                    container = self._create_import_container(source='train', force_type='FFEU')
                    wagon.add_container(container)
                    self.container_arrival_times[container.container_id] = train.arrival_time
                    
                else:  # mixed
                    # Random realistic mix
                    remaining_length = 23.0
                    while remaining_length > 6.0:  # Minimum container length
                        if remaining_length >= 12.19:
                            container_type = random.choice(['TWEU', 'FEU', 'THEU'])
                        else:
                            container_type = 'TWEU'
                        
                        container = self._create_import_container(source='train', force_type=container_type)
                        if wagon.add_container(container):
                            self.container_arrival_times[container.container_id] = train.arrival_time
                            # Update remaining length
                            if container_type == 'TWEU':
                                remaining_length -= 6.06
                            elif container_type == 'FEU':
                                remaining_length -= 12.19
                            elif container_type == 'THEU':
                                remaining_length -= 9.14
                        else:
                            break
    
    def _generate_trucks(self):
        """Generate trucks for the day."""
        # Number of trucks based on terminal capacity
        num_trucks = int(self.n_bays * 2)  # ~2 trucks per bay per day
        
        # Generate arrival times using KDE
        if 'truck_pickup' in self.config.kde_models:
            arrival_hours = self.config.sample_from_kde(
                'truck_pickup',
                n_samples=num_trucks,
                min_val=0,
                max_val=24
            )
        else:
            # Fallback: concentrated during business hours
            arrival_hours = np.random.normal(12, 3, num_trucks) % 24
        
        # Get train schedule for coordination
        train_times = [(t[0].hour + t[0].minute/60, t[1]) for t in self.scheduled_trains]
        
        for i, arrival_hour in enumerate(arrival_hours):
            # Create arrival datetime
            arrival_time = self.current_datetime.replace(
                hour=int(arrival_hour),
                minute=int((arrival_hour % 1) * 60),
                second=0,
                microsecond=0
            )
            
            # Create truck
            truck_id = f"TRK_{self.current_day:03d}_{i:04d}"
            truck = Truck(
                truck_id=truck_id,
                arrival_time=arrival_time
            )
            
            # Check if arrival matches train time (within 1 hour)
            near_train = any(
                abs(arrival_hour - train_hour) < 1.0 
                for train_hour, _ in train_times
            )
            
            # 50/50 import/export, higher train connection chance if near train
            if random.random() < 0.5:  # Import
                container = self._create_import_container(source='truck')
                truck.add_container(container)
                self.container_arrival_times[container.container_id] = arrival_time
            else:  # Export
                if near_train and random.random() < 0.7:  # 70% chance to connect with train
                    # Find train container to relay
                    matching_train = self._find_matching_train(arrival_hour, train_times)
                    if matching_train:
                        train = matching_train[1]
                        # Pick container from train's imports
                        train_containers = [c.container_id for wagon in train.wagons for c in wagon.containers]
                        if train_containers:
                            container_id = random.choice(train_containers)
                            truck.add_pickup_container_id(container_id)
                            self.container_destinations[container_id] = truck_id
                else:
                    # Regular yard pickup
                    yard_container = self._find_suitable_yard_container()
                    if yard_container:
                        truck.add_pickup_container_id(yard_container)
                        self.container_destinations[yard_container] = truck_id
            
            # Schedule truck
            self.scheduled_trucks.append((arrival_time, truck))
    
    def _get_high_priority_containers(self, limit: int) -> List[str]:
        """Get high-priority containers from yard based on wait time."""
        priority_list = []
        
        for container_id, arrival_time in self.container_arrival_times.items():
            if container_id in self.logistics.yard_container_index:
                days_waiting = (self.current_datetime - arrival_time).days
                priority = days_waiting
                
                # Add bonus for scheduled pickups
                if container_id in self.container_pickup_schedules:
                    if self.container_pickup_schedules[container_id] <= self.current_datetime:
                        priority += 10  # Overdue
                    elif self.container_pickup_schedules[container_id] <= self.current_datetime + timedelta(days=1):
                        priority += 5  # Due soon
                
                priority_list.append((priority, container_id))
        
        # Sort by priority (highest first) and return top N
        priority_list.sort(reverse=True, key=lambda x: x[0])
        return [cid for _, cid in priority_list[:limit]]
    
    def _create_import_container(self, source: str, force_type: str = None) -> Container:
        """Create an import container with appropriate properties."""
        if force_type:
            container_type = force_type
        else:
            container_types = ['TWEU', 'THEU', 'FEU', 'FFEU']
            weights = [0.18, 0.02, 0.52, 0.01]
            container_type = np.random.choice(container_types, p=weights/np.sum(weights))
        
        # Determine goods type
        if random.random() < 0.02:
            goods_type = 'Dangerous'
        elif random.random() < 0.01:
            goods_type = 'Reefer'
        else:
            goods_type = 'Regular'
        
        container_id = f"{source.upper()}_{self.current_day:03d}_{random.randint(0, 9999):04d}"
        
        # Fixed KDE sampling for weight
        weight = None
        if self.config and 'container_weight' in self.config.kde_models:
            weight_samples = self.config.sample_from_kde('container_weight', 1, 1000, 31000)
            if weight_samples is not None and len(weight_samples) > 0:
                weight = float(weight_samples[0])
        
        return ContainerFactory.create_container(
            container_id=container_id,
            container_type=container_type,
            direction='Import',
            goods_type=goods_type,
            arrival_date=self.current_datetime,
            weight=weight,
            config=self.config
        )
    
    def _find_suitable_yard_container(self) -> Optional[str]:
        """Find a suitable container in yard for export."""
        export_containers = []
        
        for container_id in self.logistics.yard_container_index:
            if container_id not in self.container_destinations:
                # Check if it's an export container
                pos = self.logistics.yard_container_index[container_id]
                row, bay, tier, split = pos
                
                try:
                    container = self.yard.get_container_at(row, bay, tier, split)
                    if container and container.direction == 'Export':
                        export_containers.append(container_id)
                except:
                    continue
        
        return random.choice(export_containers) if export_containers else None
    
    def _find_matching_train(self, truck_hour: float, train_times: List[Tuple[float, Any]]) -> Optional[Tuple[float, Any]]:
        """Find train arriving near truck time."""
        for train_hour, train in train_times:
            if abs(truck_hour - train_hour) < 1.0:
                return (train_hour, train)
        return None
    
    # In ContainerTerminal._process_arrivals:
    def _process_arrivals(self):
        """Process vehicle arrivals up to current time."""
        current_dt = self.current_datetime + timedelta(seconds=self.current_time)
        
        # Process train arrivals
        trains_arrived = 0
        while self.scheduled_trains and self.scheduled_trains[0][0] <= current_dt:
            arrival_time, train = self.scheduled_trains.pop(0)
            self.logistics.add_train_to_queue(train)
            trains_arrived += 1
        
        # Process truck arrivals  
        trucks_arrived = 0
        while self.scheduled_trucks and self.scheduled_trucks[0][0] <= current_dt:
            arrival_time, truck = self.scheduled_trucks.pop(0)
            self.logistics.add_truck_to_queue(truck)
            trucks_arrived += 1
        
        # Process queues to place vehicles
        trains_placed = self.logistics.process_current_trains()
        trucks_placed = self.logistics.process_current_trucks()
        
        # CRITICAL: Assign pickups AFTER placement
        if trains_placed > 0 or trucks_placed > 0:
            # Get available containers for export
            available_containers = [
                cid for cid in self.logistics.yard_container_index.keys()
                if cid not in self.container_destinations
            ]
            
            # Assign pickups to placed trains
            for track_id, track_trains in self.logistics.trains_on_track.items():
                for dep_time, pos_range, train in track_trains:
                    # Update active trains
                    self.active_trains[train.train_id] = train
                    
                    # Assign pickups to wagons
                    for wagon_idx, wagon in enumerate(train.wagons):
                        wagon_capacity = 2  # Typical wagon capacity
                        current_load = len(wagon.containers) + len(wagon.pickup_container_ids)
                        
                        if current_load < wagon_capacity and available_containers:
                            num_pickups = min(wagon_capacity - current_load, len(available_containers))
                            for _ in range(num_pickups):
                                container_id = available_containers.pop(random.randint(0, len(available_containers)-1))
                                wagon.add_pickup_container(container_id)
                                self.container_destinations[container_id] = train.train_id
            
            # Assign pickups to placed trucks
            for pos, truck in self.logistics.active_trucks.items():
                self.active_trucks[truck.truck_id] = truck
                
                has_pickups = len(getattr(truck, 'pickup_container_ids', set())) > 0
                has_containers = len(truck.containers) > 0
                
                if not has_pickups and available_containers:
                    container_id = available_containers.pop(random.randint(0, len(available_containers)-1))
                    truck.add_pickup_container_id(container_id)
                    self.container_destinations[container_id] = truck.truck_id
            
            # CRITICAL FIX: Force sync of all pickup mappings after assignment
            # This ensures the mappings are correctly populated
            self.logistics.sync_pickup_mappings()
            
            # Clear and rebuild the move cache to reflect new pickups
            self.logistics.available_moves_cache = self.logistics.find_moves_optimized()
        
        if trains_arrived > 0 or trucks_arrived > 0 or trains_placed > 0 or trucks_placed > 0:
            if self.validate:
                print(f"Arrivals: {trains_arrived} trains, {trucks_arrived} trucks -> "
                    f"Placed: {trains_placed} trains, {trucks_placed} trucks")
            
            # Debug: Verify mappings are correct
            total_train_pickups = sum(
                len(wagon.pickup_container_ids)
                for train in self.active_trains.values()
                for wagon in train.wagons
            )
            total_truck_pickups = sum(
                len(getattr(truck, 'pickup_container_ids', set()))
                for truck in self.logistics.active_trucks.values()
            )

            if self.validate:
                print(f"  Assigned pickups: {total_train_pickups} to trains, {total_truck_pickups} to trucks")
                print(f"  Mapping sizes: {len(self.logistics.pickup_to_train)} train mappings, "
                    f"{len(self.logistics.pickup_to_truck)} truck mappings")
        
    def analyze_available_moves(self) -> Dict[str, int]:
        """Analyze and categorize available moves."""
        moves = self.logistics.find_moves_optimized()
        move_categories = defaultdict(int)
        
        for move_id, move_data in moves.items():
            move_type = f"{move_data['source_type']}_to_{move_data['dest_type']}"
            move_categories[move_type] += 1
        
        return dict(move_categories)

    def debug_vehicle_state(self):
        """Debug current vehicle state."""
        print(f"\n=== Vehicle State Debug ===")
        print(f"Active trains: {len(self.active_trains)}")
        for train_id, train in self.active_trains.items():
            containers = sum(len(w.containers) for w in train.wagons)
            pickups = sum(len(w.pickup_container_ids) for w in train.wagons)
            print(f"  {train_id}: {containers} containers, {pickups} pickups")
            # Show wagon details
            for i, wagon in enumerate(train.wagons):
                if wagon.containers or wagon.pickup_container_ids:
                    print(f"    Wagon {i}: {len(wagon.containers)} containers, {len(wagon.pickup_container_ids)} pickups")
        
        # Check for buried containers
        if self.active_trains:
            buried_count = 0
            for train in self.active_trains.values():
                for wagon in train.wagons:
                    for container_id in wagon.pickup_container_ids:
                        # Check if container is buried
                        if container_id in self.logistics.yard_container_index:
                            pos = self.logistics.yard_container_index[container_id]
                            row, bay, tier, split = pos
                            if tier > 0:  # Not on ground level
                                # Check if accessible
                                above_empty = True
                                for check_tier in range(tier + 1, self.yard.n_tiers):
                                    if self.yard.get_container_at(row, bay, check_tier, split):
                                        above_empty = False
                                        break
                                if not above_empty:
                                    buried_count += 1
            
            if buried_count > 0:
                print(f"  WARNING: {buried_count} requested containers are buried in yard!")

        print(f"Active trucks: {len(self.logistics.active_trucks)}")
        for pos, truck in self.logistics.active_trucks.items():
            print(f"  {truck.truck_id} at {pos}: {len(truck.containers)} containers, "
                f"{len(getattr(truck, 'pickup_container_ids', set()))} pickups")
        
        print(f"Scheduled arrivals: {len(self.scheduled_trains)} trains, {len(self.scheduled_trucks)} trucks")
        if self.scheduled_trains:
            next_train_time = self.scheduled_trains[0][0]
            current_dt = self.current_datetime + timedelta(seconds=self.current_time)
            print(f"  Next train in: {(next_train_time - current_dt).total_seconds()/3600:.1f} hours")
        
        # Check pickup mappings vs actual
        print(f"Pickup mappings: {len(self.logistics.pickup_to_train)} to trains, "
            f"{len(self.logistics.pickup_to_truck)} to trucks")
        
        # Verify consistency
        actual_train_pickups = sum(
            len(wagon.pickup_container_ids) 
            for train in self.active_trains.values() 
            for wagon in train.wagons
        )
        actual_truck_pickups = sum(
            len(getattr(truck, 'pickup_container_ids', set()))
            for truck in self.logistics.active_trucks.values()
        )
        print(f"Actual pickups: {actual_train_pickups} on trains, {actual_truck_pickups} on trucks")

    def step(self, action: int):
        """Execute action and advance simulation."""
        # Get available moves
        moves = self.logistics.find_moves_optimized()
        move_list = list(moves.items())
        
        # Validate action
        if action < 0 or action >= len(move_list):
            # Invalid action - wait
            reward = -0.1
            terminated = False
            truncated = False
            info = {'error': 'Invalid action', 'move_count': len(move_list)}
        else:
            # Execute move
            move_id, move_data = move_list[action]
            
            # Create RMGC controller if needed (lazy loading)
            if not hasattr(self, 'rmgc'):
                from simulation.terminal_components.RMGC import RMGC_Controller
                self.rmgc = RMGC_Controller(self.yard, self.logistics, heads=2)
            
            # Debug print for failed moves
            if hasattr(self, '_debug') and self._debug:
                print(f"\nAttempting move: {move_id}")
                print(f"  Type: {move_data['move_type']}")
                print(f"  Container: {move_data.get('container_id', 'Unknown')}")
                print(f"  Route: {move_data['source_type']} -> {move_data['dest_type']}")
            
            # Execute move
            distance, time_taken = self.rmgc.execute_move(move_data)
            
            if time_taken > 0:
                # Successful move
                current_dt = self.current_datetime + timedelta(seconds=self.current_time)
                
                # Calculate reward
                reward = self.reward_calculator.calculate_reward(
                    move_data,
                    distance,
                    time_taken,
                    current_dt,
                    self.active_trains,
                    self.logistics.active_trucks,
                    self.container_pickup_schedules,
                    self.logistics
                )

                # Advance time
                self._advance_time(time_taken)
                
                # Log move
                self._log_move(move_data, distance, time_taken, reward)
                
                # Update metrics
                self.episode_metrics['total_moves'] += 1
                self.episode_metrics['total_distance'] += distance
                self.episode_metrics['total_time'] += time_taken
                self.daily_metrics['distances'].append(distance)
                self.daily_metrics['rewards'].append(reward)
                
                # Sync yard index after successful move
                self.logistics.sync_yard_index()
            else:
                # Failed move
                reward = -1.0
                self._advance_time(60)  # Penalty wait
                
                if hasattr(self, '_debug') and self._debug:
                    print(f"  FAILED! Distance: {distance}, Time: {time_taken}")
            
            terminated = False
            truncated = self.current_day >= self.max_days
        
        # Get next observation
        obs = self._get_observation()
        info = self._get_info()
        info['move_list'] = moves
        info['day'] = self.current_day
        info['time_of_day'] = self.current_time / self.seconds_per_day
        
        if not(action < 0 or action >= len(move_list)) and time_taken > 0:
            info['last_move_time'] = time_taken
        # Add move breakdown
        info['move_breakdown'] = self.analyze_available_moves()
        
        return obs, reward, terminated, truncated, info
    
    # In ContainerTerminal._advance_time:
    def _advance_time(self, seconds: float):
        """Advance simulation time and handle day transitions."""
        self.current_time += seconds
        
        # Update simulation datetime
        current_dt = self.current_datetime + timedelta(seconds=self.current_time)
        self.logistics.current_datetime = current_dt  # Pass to logistics
        
        # Check for day transition
        if self.current_time >= self.seconds_per_day:
            self._end_of_day()
            self.current_day += 1
            self.current_time -= self.seconds_per_day
            self.current_datetime = self.current_datetime + timedelta(days=1)
            self._generate_daily_schedule()
        
        # Process arrivals BEFORE reorganization
        self._process_arrivals()
        self._update_container_priorities()
        
        # CRITICAL: Ensure mappings are synced before reorganization
        self.logistics.sync_pickup_mappings()
        
        # Reorganize logistics (this will also sync mappings internally)
        departures, penalty = self.logistics.reorganize_logistics()
        
        # Track metrics
        if penalty != 0:
            self.episode_metrics['total_penalties'] += abs(penalty)
            self.episode_metrics['forced_departures'] += 1
    
    def _end_of_day(self):
        """Process end of day tasks."""
        # Update daily metrics
        self.daily_metrics['day_end_containers'] = len(self.logistics.yard_container_index)
        self.daily_metrics['trains_processed'] = len([t for t in self.active_trains.values() if t.status == 'DEPARTED'])
        self.daily_metrics['trucks_processed'] = sum(1 for _ in self.logistics.truck_positions)
        
        # Log daily summary
        logger.info(f"Day {self.current_day} complete: "
                   f"{self.episode_metrics['total_moves']} moves, "
                   f"{len(self.logistics.yard_container_index)} containers in yard")
    
    def _update_container_priorities(self):
        """Update container priorities based on wait time."""
        current_dt = self.current_datetime + timedelta(seconds=self.current_time)
        
        for container_id in self.logistics.yard_container_index:
            if container_id in self.container_arrival_times:
                # Update priority in container object
                pos = self.logistics.yard_container_index[container_id]
                row, bay, tier, split = pos
                
                try:
                    container = self.yard.get_container_at(row, bay, tier, split)
                    if container:
                        container.update_priority()
                except:
                    continue
    
    def _get_observation(self) -> np.ndarray:
        """Get current state observation."""
        obs_parts = []
        
        # 1. Yard state (flattened)
        yard_state = self.yard.get_full_state_tensor(flatten=True)
        obs_parts.append(yard_state.cpu().numpy())
        
        # 2. Rail and parking states
        rail_state = self.logistics.get_rail_state_tensor(as_tensor=False).flatten()
        parking_state = self.logistics.get_parking_state_tensor(as_tensor=False).flatten()
        obs_parts.append(np.concatenate([rail_state, parking_state]))
        
        # 3. Vehicle properties
        train_props = self.logistics.get_train_properties_tensor(as_tensor=False).flatten()
        truck_props = self.logistics.get_truck_properties_tensor(as_tensor=False).flatten()
        obs_parts.append(np.concatenate([train_props, truck_props]))
        
        # 4. Queue states
        queue_state = self.logistics.get_queue_state_tensor(as_tensor=False)
        obs_parts.append(queue_state)
        
        # 5. Temporal features
        current_dt = self.current_datetime + timedelta(seconds=self.current_time)
        temporal_features = self.temporal_encoder.encode_daily_schedule(
            self.current_time,
            self.current_day,
            self.active_trains,
            self.logistics.active_trucks,
            self.logistics.trains,
            self.logistics.trucks,
            self.container_pickup_schedules,
            self.container_arrival_times
        )
        obs_parts.append(temporal_features)
        
        # Concatenate all parts
        observation = np.concatenate(obs_parts).astype(np.float32)
        
        return observation
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information."""
        current_dt = self.current_datetime + timedelta(seconds=self.current_time)
        
        # Get moves and rank them by urgency
        moves = self.logistics.find_moves_optimized()
        ranked_moves = []
        
        for move_id, move_data in moves.items():
            urgency, value = self.move_evaluator.evaluate_move_urgency(
                move_data,
                current_dt,
                self.active_trains,
                self.logistics.active_trucks,
                self.container_pickup_schedules,
                self.logistics
            )
            ranked_moves.append((urgency * value, move_id))
        
        ranked_moves.sort(reverse=True)
        ranked_move_list = [move_id for _, move_id in ranked_moves]
        
        info = {
            'move_list': moves,
            'ranked_move_list': ranked_move_list,
            'containers_in_yard': len(self.logistics.yard_container_index),
            'trains_in_terminal': len(self.active_trains),
            'trucks_in_terminal': len(self.logistics.active_trucks),
            'train_queue_size': self.logistics.trains.size(),
            'truck_queue_size': self.logistics.trucks.size(),
            'time_of_day': self.current_time / self.seconds_per_day,
            'day': self.current_day,
            'metrics': dict(self.episode_metrics)
        }
        
        return info
    
    def _log_move(self, move_data: Dict, distance: float, time: float, reward: float):
        """Log move to history."""
        move_record = {
            'day': self.current_day,
            'time': self.current_time,
            'container_id': move_data.get('container_id'),
            'move_type': move_data.get('move_type'),
            'source_type': move_data.get('source_type'),
            'dest_type': move_data.get('dest_type'),
            'distance': distance,
            'time': time,
            'reward': reward
        }
        
        self.move_history.append(move_record)
        
        # Write to log file if specified
        if self.move_log_file:
            import csv
            with open(self.move_log_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=move_record.keys())
                if f.tell() == 0:  # Write header if file is empty
                    writer.writeheader()
                writer.writerow(move_record)
    
    def render(self, mode='human'):
        """Render the environment."""
        if mode == 'human':
            print(f"\n=== Day {self.current_day}, Time: {self.current_time/3600:.1f}h ===")
            print(f"Containers in yard: {len(self.logistics.yard_container_index)}")
            print(f"Active trains: {len(self.active_trains)}")
            print(f"Active trucks: {len(self.logistics.active_trucks)}")
            print(f"Available moves: {len(self.logistics.available_moves_cache)}")
            print(f"Total moves today: {self.episode_metrics['total_moves']}")
            print(f"Total penalties: {self.episode_metrics['total_penalties']}")
            
        return None
    
    def close(self):
        """Clean up resources."""
        pass


# Update the test section:
if __name__ == '__main__':
    print("Testing ContainerTerminal Environment")
    print("=" * 60)
    
    # Create environment with debug enabled
    env = ContainerTerminal(
        n_rows=5,
        n_bays=15,
        n_tiers=4,
        n_railtracks=4,
        split_factor=4,
        max_days=3,
        seed=42
    )
    env._debug = True  # Enable debug printing
    
    print(f"Environment created")
    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.n}")
    
    # Reset environment
    obs, info = env.reset()
    print(f"\nInitial state:")
    print(f"Observation shape: {obs.shape}")
    print(f"Containers in yard: {info['containers_in_yard']}")
    print(f"Available moves: {len(info['move_list'])}")
    
    # Analyze initial moves
    print("\nInitial move breakdown:")
    move_breakdown = env.analyze_available_moves()
    for move_type, count in sorted(move_breakdown.items()):
        print(f"  {move_type}: {count}")
    
    # Check for vehicle moves specifically
    vehicle_moves = 0
    moves = info['move_list']
    for move_id, move_data in moves.items():
        if move_data['source_type'] in ['train', 'truck'] or move_data['dest_type'] in ['train', 'truck']:
            vehicle_moves += 1
            if vehicle_moves <= 5:  # Show first 5 vehicle moves
                print(f"\nVehicle move example {vehicle_moves}:")
                print(f"  ID: {move_id}")
                print(f"  Container: {move_data.get('container_id', 'Unknown')}")
                print(f"  Type: {move_data['move_type']}")
                print(f"  Route: {move_data['source_type']} -> {move_data['dest_type']}")
    
    print(f"\nTotal vehicle-related moves: {vehicle_moves}/{len(moves)}")
    
    # Run a few steps with better action selection
    print("\nRunning 10 steps with prioritized actions...")
    total_reward = 0
    
    for step in range(10):

        # Debug every 3 steps
        if step % 3 == 0:
            env.debug_vehicle_state()

        # Try to select vehicle-related moves first
        moves = info.get('move_list', {})
        move_list = list(moves.items())
        
        # Find a vehicle move
        vehicle_action = None
        for i, (move_id, move_data) in enumerate(move_list):
            if move_data['source_type'] in ['train', 'truck'] or move_data['dest_type'] in ['train', 'truck']:
                vehicle_action = i
                break
        
        # Use vehicle move if found, otherwise random
        if vehicle_action is not None and step < 5:
            action = vehicle_action
        else:
            action = random.randint(0, max(0, len(move_list) - 1)) if move_list else 0
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"\nStep {step + 1}: Action={action}, Reward={reward:.2f}, Time={env.current_time:.2f} "
              f"Moves available={len(info.get('move_list', {}))}")
        
        # Show move breakdown change
        if 'move_breakdown' in info:
            print("  Move types:", ", ".join(f"{k}:{v}" for k, v in info['move_breakdown'].items()))
        
        if terminated or truncated:
            break
    
    print(f"\nTotal reward: {total_reward:.2f}")
    print(f"Final metrics: {info['metrics']}")