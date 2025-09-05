import heapq
import json
import numpy as np
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Set, Tuple, Any, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict

# Import required components
from simulation.terminal_components.vehicles.Train import Train
from simulation.terminal_components.storage_units.Container import Container
from simulation.terminal_components.systems.ContainerFactory import ContainerFactory

# ==================== SCHEDULER CONSTANTS ====================

# Train Configuration
TRAIN_NUM_WAGONS = 29  # Fixed number of wagons per train
TRAIN_WAGON_LENGTH_FT = 80  # Fixed wagon length in feet
TRAIN_WAGON_LENGTH_M = 24.384  # Fixed wagon length in meters
TRAIN_MIN_WAGON_UTILIZATION = 0.9  # Minimum 80% wagon utilization

# Time Configuration
WEEK_MINUTES = 7 * 24 * 60  # Total minutes in a week
DAY_MINUTES = 24 * 60  # Total minutes in a day
HOUR_MINUTES = 60  # Minutes in an hour
CONTAINER_PREP_TIME_MINUTES = 30  # Minutes before arrival to prepare containers

# Weekday Mapping
WEEKDAY_NAMES = {
    'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
    'Friday': 4, 'Saturday': 5, 'Sunday': 6
}
WEEKDAY_SHORT_NAMES = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
WEEKDAY_FULL_NAMES = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# Visualization Constants
GANTT_FIGURE_WIDTH = 16
GANTT_FIGURE_HEIGHT = 8
GANTT_RECT_HEIGHT = 0.8
GANTT_LABEL_MIN_DURATION_HOURS = 2  # Minimum duration to show label
GANTT_GRID_ALPHA = 0.3
GANTT_DAYLINE_ALPHA = 0.5
GANTT_COLORS_COUNT = 12

# Rail Management
DEFAULT_NUM_RAILS = 10
RAIL_PREFIX = "Rail_"


class EventType(Enum):
    """Types of train events."""
    ARRIVAL_PREP = "arrival_prep"  # Prepare containers before arrival
    ARRIVAL = "arrival"
    ARRIVAL_COMPLETE = "arrival_complete"
    DEPARTURE = "departure"
    DEPARTURE_COMPLETE = "departure_complete"


class WeekdayTime(NamedTuple):
    """Represents a time on a specific weekday."""
    weekday: int  # 0=Monday, 6=Sunday
    hour: int
    minute: int
    
    @classmethod
    def from_string(cls, weekday_str: str, time_str: str) -> 'WeekdayTime':
        """Create from weekday name and time string."""
        # Clean weekday string and handle common typos
        weekday_str = weekday_str.replace('Tueday', 'Tuesday').strip()
        weekday = WEEKDAY_NAMES.get(weekday_str, 0)
        
        # Parse time string
        hour, minute = cls._parse_time_string(time_str)
        
        # Handle hour overflow (e.g., 25:00 becomes 01:00 next day)
        if hour >= 24:
            hour = hour % 24
            weekday = (weekday + 1) % 7
        
        return cls(weekday, hour, minute)
    
    @staticmethod
    def _parse_time_string(time_str: str) -> Tuple[int, int]:
        """Parse time string to hour and minute."""
        time_str = time_str.strip()
        
        if ':' in time_str:
            parts = time_str.split(':')
            hour = int(parts[0])
            minute = int(parts[1][:2]) if len(parts[1]) >= 2 else 0
        else:
            # Handle formats like "0800" or "800"
            if len(time_str) == 3:
                time_str = '0' + time_str
            if len(time_str) == 4:
                hour = int(time_str[:2])
                minute = int(time_str[2:])
            else:
                hour, minute = 0, 0
        
        return hour, minute
    
    def to_datetime(self, reference_date: datetime) -> datetime:
        """Convert to concrete datetime given a reference date."""
        days_ahead = self.weekday - reference_date.weekday()
        if days_ahead < 0:
            days_ahead += 7
        
        target_date = reference_date.date() + timedelta(days=days_ahead)
        return datetime.combine(target_date, time(self.hour, self.minute))
    
    def minutes_in_week(self) -> int:
        """Get total minutes from start of week (Monday 00:00)."""
        return self.weekday * DAY_MINUTES + self.hour * HOUR_MINUTES + self.minute
    
    def __str__(self) -> str:
        return f"{WEEKDAY_SHORT_NAMES[self.weekday]} {self.hour:02d}:{self.minute:02d}"


@dataclass
class AbstractTrainSchedule:
    """Abstract train schedule that repeats weekly."""
    train_id: str
    operator: str
    destination: str
    arrival: WeekdayTime
    arrival_complete: WeekdayTime
    departure: WeekdayTime
    departure_complete: WeekdayTime
    rail: Optional[int] = None
    
    def get_occupancy_minutes(self) -> Tuple[int, int]:
        """Get occupancy period in minutes from week start."""
        return (self.arrival.minutes_in_week(), 
                self.departure_complete.minutes_in_week())
    
    def get_duration_minutes(self) -> int:
        """Get total duration in minutes, handling week wrap."""
        start = self.arrival.minutes_in_week()
        end = self.departure_complete.minutes_in_week()
        
        if end < start:  # Wraps around week
            return (WEEK_MINUTES - start) + end
        return end - start


@dataclass
class ConcreteTrainEvent:
    """Concrete event for processing during simulation."""
    timestamp: datetime
    event_type: EventType
    schedule: AbstractTrainSchedule
    pickup_assignments: Dict[str, int] = field(default_factory=dict)  # container_id -> wagon_index
    
    def __lt__(self, other):
        return self.timestamp < other.timestamp


@dataclass
class ContainerAssignment:
    """Assignment of a container to a specific wagon."""
    container_id: str
    wagon_index: int
    position_in_wagon: Optional[int] = None  # Position within the wagon


class TrainScheduler:
    """
    Train scheduler with proper container management.
    Trains arrive fully loaded and pick up specific containers by ID.
    """
    
    def __init__(self, 
                 driving_plan_path: str,
                 num_rails: int = DEFAULT_NUM_RAILS,
                 container_factory: Optional[ContainerFactory] = None):
        """
        Initialize the train scheduler.
        
        Args:
            driving_plan_path: Path to the driving plan JSON
            num_rails: Number of available rails
            container_factory: Factory for container generation
        """
        self.num_rails = num_rails
        self.prep_time = timedelta(minutes=CONTAINER_PREP_TIME_MINUTES)
        
        # Load driving plan
        with open(driving_plan_path, 'r') as f:
            self.driving_plan = json.load(f)['driving_plan']
        
        # Initialize container factory
        self.container_factory = container_factory or ContainerFactory()
        
        # Schedule storage
        self.abstract_schedules: List[AbstractTrainSchedule] = []
        self.rail_assignments: Dict[str, int] = {}
        
        # Train management
        self.train_pool: Dict[str, Train] = {}
        self.active_trains: Dict[str, Train] = {}
        
        # Container pickup management
        # Maps container IDs to their assigned wagon on next available train
        self.pickup_queue: Dict[str, ContainerAssignment] = {}
        
        # Event queue
        self.event_queue: List[ConcreteTrainEvent] = []
        
        # Metrics
        self.metrics = self._initialize_metrics()
        
        # Build and optimize schedule
        self._build_abstract_schedule()
        self._assign_rails()
    
    def _initialize_metrics(self) -> Dict[str, int]:
        """Initialize performance metrics."""
        return {
            'total_arrivals': 0,
            'total_departures': 0,
            'containers_on_arrival': 0,
            'containers_picked_up': 0,
            'containers_delivered': 0,
            'wagons_utilized': 0,
            'wagons_underutilized': 0,
            'trains_processed': 0
        }
    
    def _build_abstract_schedule(self):
        """Build abstract weekly schedule from driving plan."""
        for train_code, train_info in self.driving_plan['trains'].items():
            operator = train_info.get('operator', 'Unknown')
            destination = train_info.get('destination', 'Unknown')
            
            # Pre-create train object for reuse
            self._get_or_create_train(train_code, operator, destination)
            
            # Process each schedule entry
            for plan_key, plan_entry in train_info['plan'].items():
                self._process_plan_entry(
                    train_code, operator, destination, 
                    plan_key, plan_entry
                )
    
    def _get_or_create_train(self, train_code: str, operator: str, destination: str) -> Train:
        """Get existing train or create new one."""
        if train_code not in self.train_pool:
            train = Train(
                train_id=train_code,
                num_wagons=TRAIN_NUM_WAGONS,
                wagon_length=TRAIN_WAGON_LENGTH_M
            )
            # Add metadata
            train.operator = operator
            train.destination = destination
            self.train_pool[train_code] = train
        return self.train_pool[train_code]
    
    def _process_plan_entry(self, train_code: str, operator: str, 
                           destination: str, plan_key: str, plan_entry: Dict):
        """Process a single plan entry and create schedules."""
        arrival = plan_entry['arrival']
        departure = plan_entry['departure']
        mirrored_days = plan_entry.get('mirrored_on', [])
        
        # Parse times
        arrival_wt = WeekdayTime.from_string(arrival[0], arrival[1])
        arrival_complete_wt = WeekdayTime.from_string(arrival[2], arrival[3])
        departure_wt = WeekdayTime.from_string(departure[0], departure[1])
        departure_complete_wt = WeekdayTime.from_string(departure[2], departure[3])
        
        # Create main schedule
        main_schedule = AbstractTrainSchedule(
            train_id=f"{train_code}_{plan_key}",
            operator=operator,
            destination=destination,
            arrival=arrival_wt,
            arrival_complete=arrival_complete_wt,
            departure=departure_wt,
            departure_complete=departure_complete_wt
        )
        self.abstract_schedules.append(main_schedule)
        
        # Create mirrored schedules
        for mirror_day in mirrored_days:
            self._create_mirrored_schedule(
                train_code, operator, destination, plan_key,
                mirror_day, arrival_wt, arrival_complete_wt,
                departure_wt, departure_complete_wt
            )
    
    def _create_mirrored_schedule(self, train_code: str, operator: str,
                                 destination: str, plan_key: str,
                                 mirror_day: str, 
                                 original_arrival: WeekdayTime,
                                 original_arrival_complete: WeekdayTime,
                                 original_departure: WeekdayTime,
                                 original_departure_complete: WeekdayTime):
        """Create a mirrored schedule for a specific day."""
        # Parse mirror day
        mirror_weekday = WEEKDAY_NAMES.get(mirror_day, 0)
        day_diff = mirror_weekday - original_arrival.weekday
        
        # Adjust all times by the day difference
        mirror_schedule = AbstractTrainSchedule(
            train_id=f"{train_code}_{plan_key}_mirror_{mirror_day}",
            operator=operator,
            destination=destination,
            arrival=WeekdayTime(
                mirror_weekday,
                original_arrival.hour,
                original_arrival.minute
            ),
            arrival_complete=WeekdayTime(
                (original_arrival_complete.weekday + day_diff) % 7,
                original_arrival_complete.hour,
                original_arrival_complete.minute
            ),
            departure=WeekdayTime(
                (original_departure.weekday + day_diff) % 7,
                original_departure.hour,
                original_departure.minute
            ),
            departure_complete=WeekdayTime(
                (original_departure_complete.weekday + day_diff) % 7,
                original_departure_complete.hour,
                original_departure_complete.minute
            )
        )
        self.abstract_schedules.append(mirror_schedule)
    
    def _assign_rails(self):
        """Assign rails to schedules using greedy algorithm."""
        print(f"\nAssigning {len(self.abstract_schedules)} schedules to {self.num_rails} rails...")
        
        # Sort by arrival time for greedy assignment
        sorted_schedules = sorted(
            self.abstract_schedules,
            key=lambda s: s.arrival.minutes_in_week()
        )
        
        rail_schedules = {i: [] for i in range(self.num_rails)}
        assigned_count = 0
        unassigned = []
        
        for schedule in sorted_schedules:
            rail = self._find_available_rail(schedule, rail_schedules)
            
            if rail is not None:
                schedule.rail = rail
                rail_schedules[rail].append(schedule)
                self.rail_assignments[schedule.train_id] = rail
                assigned_count += 1
            else:
                unassigned.append(schedule.train_id)
        
        print(f"Assigned {assigned_count}/{len(self.abstract_schedules)} schedules")
        if unassigned:
            print(f"Warning: {len(unassigned)} schedules unassigned (need more rails)")
    
    def _find_available_rail(self, schedule: AbstractTrainSchedule, 
                            rail_schedules: Dict[int, List]) -> Optional[int]:
        """Find first available rail for a schedule."""
        for rail in range(self.num_rails):
            if all(not self._schedules_overlap(schedule, existing) 
                   for existing in rail_schedules[rail]):
                return rail
        return None
    
    def _schedules_overlap(self, s1: AbstractTrainSchedule, 
                          s2: AbstractTrainSchedule) -> bool:
        """Check if two schedules overlap, handling week wrap."""
        s1_start = s1.arrival.minutes_in_week()
        s1_end = s1.departure_complete.minutes_in_week()
        s2_start = s2.arrival.minutes_in_week()
        s2_end = s2.departure_complete.minutes_in_week()
        
        # Handle week wrap-around cases
        if s1_end < s1_start:  # s1 wraps
            if s2_end < s2_start:  # Both wrap
                return True
            # Only s1 wraps
            return (s2_start < s1_end or s2_end > s1_start or
                   (s2_start >= s1_start and s2_end <= WEEK_MINUTES))
        
        if s2_end < s2_start:  # Only s2 wraps
            return (s1_start < s2_end or s1_end > s2_start or
                   (s1_start >= s2_start and s1_end <= WEEK_MINUTES))
        
        # Neither wraps - simple overlap check
        return not (s1_end <= s2_start or s2_end <= s1_start)
    
    def assign_container_pickup(self, container_id: str, wagon_index: Optional[int] = None):
        """
        Assign a container ID to be picked up by the next available train.
        
        Args:
            container_id: ID of container to be picked up
            wagon_index: Optional specific wagon index (0-28), auto-assigns if None
        """
        if wagon_index is not None and (wagon_index < 0 or wagon_index >= TRAIN_NUM_WAGONS):
            raise ValueError(f"Wagon index must be between 0 and {TRAIN_NUM_WAGONS-1}")
        
        # If no wagon specified, distribute evenly across wagons
        if wagon_index is None:
            # Simple round-robin assignment
            existing_assignments = len(self.pickup_queue)
            wagon_index = existing_assignments % TRAIN_NUM_WAGONS
        
        self.pickup_queue[container_id] = ContainerAssignment(
            container_id=container_id,
            wagon_index=wagon_index
        )
    
    def generate_arriving_containers(self, operator: str, current_time: datetime) -> List[Container]:
        """
        Generate containers for an arriving train.
        The number and type depend on the operator's container distribution.
        Trains arrive fully loaded to capacity.
        """
        containers = []
        
        # Generate containers to fill the train
        # The factory will determine the mix based on operator
        for wagon_idx in range(TRAIN_NUM_WAGONS):
            wagon_containers = self._fill_wagon(operator, current_time)
            containers.extend(wagon_containers)
        
        return containers
    
    def _fill_wagon(self, operator: str, arrival_time: datetime) -> List[Container]:
        """
        Fill a wagon with containers based on operator's typical distribution.
        Ensures wagon meets minimum utilization requirement.
        """
        wagon_containers = []
        used_length = 0.0
        
        # Keep adding containers until wagon is sufficiently full
        while used_length / TRAIN_WAGON_LENGTH_M < TRAIN_MIN_WAGON_UTILIZATION:
            # Generate a single container based on operator distribution
            container = self.container_factory.create_containers(
                operator="BOX",  # Using default mapping
                direction="Import",
                n_containers=1,
                base_arrival_date=arrival_time,
                current_date=arrival_time
            )[0]
            
            # Check if it fits
            if used_length + container.length_m <= TRAIN_WAGON_LENGTH_M:
                wagon_containers.append(container)
                used_length += container.length_m
            else:
                # Wagon is full enough or can't fit more
                break
        
        # Verify minimum utilization
        utilization = used_length / TRAIN_WAGON_LENGTH_M
        if utilization < TRAIN_MIN_WAGON_UTILIZATION and wagon_containers:
            # Try to swap or adjust to meet requirement
            # For now, accept if we have at least one container
            pass
        
        return wagon_containers
    
    def initialize_for_period(self, start_date: datetime, end_date: datetime):
        """Initialize concrete events for simulation period."""
        self.event_queue.clear()
        events = []
        
        # Generate events for each week in period
        current_week_start = start_date - timedelta(days=start_date.weekday())
        
        while current_week_start < end_date:
            for schedule in self.abstract_schedules:
                if schedule.rail is None:
                    continue
                
                # Convert to concrete times
                arrival_dt = schedule.arrival.to_datetime(current_week_start)
                
                # Only include if within period
                if arrival_dt >= start_date and arrival_dt <= end_date:
                    # Create all events for this train
                    self._create_train_events(schedule, arrival_dt, events)
            
            current_week_start += timedelta(weeks=1)
        
        # Build heap
        heapq.heapify(events)
        self.event_queue = events
    
    def _create_train_events(self, schedule: AbstractTrainSchedule, 
                           arrival_dt: datetime, events: List):
        """Create all events for a train schedule."""
        arrival_complete_dt = arrival_dt + timedelta(
            minutes=(schedule.arrival_complete.minutes_in_week() - 
                    schedule.arrival.minutes_in_week())
        )
        departure_dt = arrival_dt + timedelta(
            minutes=(schedule.departure.minutes_in_week() - 
                    schedule.arrival.minutes_in_week())
        )
        departure_complete_dt = arrival_dt + timedelta(
            minutes=(schedule.departure_complete.minutes_in_week() - 
                    schedule.arrival.minutes_in_week())
        )
        
        # Prep event will handle container generation and pickup assignments
        prep_event = ConcreteTrainEvent(
            arrival_dt - self.prep_time,
            EventType.ARRIVAL_PREP,
            schedule
        )
        
        # Assign pickups to this train from the queue
        if self.pickup_queue:
            # Take assignments for this train
            assignments_to_take = min(len(self.pickup_queue), TRAIN_NUM_WAGONS * 3)  # Reasonable limit
            for container_id in list(self.pickup_queue.keys())[:assignments_to_take]:
                assignment = self.pickup_queue[container_id]
                prep_event.pickup_assignments[container_id] = assignment.wagon_index
        
        events.extend([
            prep_event,
            ConcreteTrainEvent(arrival_dt, EventType.ARRIVAL, schedule),
            ConcreteTrainEvent(arrival_complete_dt, EventType.ARRIVAL_COMPLETE, schedule),
            ConcreteTrainEvent(departure_dt, EventType.DEPARTURE, schedule),
            ConcreteTrainEvent(departure_complete_dt, EventType.DEPARTURE_COMPLETE, schedule)
        ])
    
    def process_event(self, event: ConcreteTrainEvent, 
                     current_time: datetime) -> Dict[str, Any]:
        """Process a train event."""
        base_train_id = event.schedule.train_id.split('_')[0]
        train = self._get_active_train(base_train_id, event.schedule)
        
        result = {
            'event_type': event.event_type,
            'train_id': base_train_id,
            'rail': event.schedule.rail,
            'timestamp': event.timestamp,
            'action': None
        }
        
        handler = {
            EventType.ARRIVAL_PREP: self._handle_arrival_prep,
            EventType.ARRIVAL: self._handle_arrival,
            EventType.ARRIVAL_COMPLETE: self._handle_arrival_complete,
            EventType.DEPARTURE: self._handle_departure,
            EventType.DEPARTURE_COMPLETE: self._handle_departure_complete
        }.get(event.event_type)
        
        if handler:
            handler(train, event, current_time, result)
        
        return result
    
    def _get_active_train(self, base_train_id: str, 
                         schedule: AbstractTrainSchedule) -> Train:
        """Get or activate a train."""
        if base_train_id not in self.active_trains:
            train = self.train_pool[base_train_id]
            self.active_trains[base_train_id] = train
        return self.active_trains[base_train_id]
    
    def _handle_arrival_prep(self, train: Train, event: ConcreteTrainEvent,
                           current_time: datetime, result: Dict):
        """Handle arrival preparation - generate full train of containers and setup pickups."""
        # Generate containers to fill the train
        arriving_containers = self.generate_arriving_containers(
            event.schedule.operator, 
            current_time
        )
        
        # Load containers onto train
        for container in arriving_containers:
            train.add_container(container)
        
        # Setup pickup assignments
        for container_id, wagon_idx in event.pickup_assignments.items():
            train.add_pickup_container(container_id, wagon_index=wagon_idx)
            # Remove from global queue
            if container_id in self.pickup_queue:
                del self.pickup_queue[container_id]
        
        result['action'] = 'containers_prepared'
        result['containers_on_arrival'] = len(arriving_containers)
        result['pickups_assigned'] = len(event.pickup_assignments)
        result['total_containers'] = train.get_container_count()
        
        self.metrics['containers_on_arrival'] += len(arriving_containers)
    
    def _handle_arrival(self, train: Train, event: ConcreteTrainEvent,
                       current_time: datetime, result: Dict):
        """Handle train arrival."""
        train.arrival_time = current_time
        train.rail_track = f"{RAIL_PREFIX}{event.schedule.rail}"
        train.status = "waiting"
        result['action'] = 'train_arrived'
        self.metrics['total_arrivals'] += 1
    
    def _handle_arrival_complete(self, train: Train, event: ConcreteTrainEvent,
                                current_time: datetime, result: Dict):
        """Handle arrival completion - begin loading/unloading."""
        train.status = "loading"
        result['action'] = 'loading_started'
        
        # Count containers to be delivered (unloaded)
        delivered = train.get_container_count()
        self.metrics['containers_delivered'] += delivered
        
        # Count pickups
        pickups = len(train.get_all_pickup_container_ids())
        self.metrics['containers_picked_up'] += pickups
    
    def _handle_departure(self, train: Train, event: ConcreteTrainEvent,
                        current_time: datetime, result: Dict):
        """Handle train departure."""
        train.status = "departing"
        result['action'] = 'train_departing'
        
        # Calculate wagon utilization
        utilized, underutilized = self._calculate_wagon_utilization(train)
        self.metrics['wagons_utilized'] += utilized
        self.metrics['wagons_underutilized'] += underutilized
    
    def _handle_departure_complete(self, train: Train, event: ConcreteTrainEvent,
                                  current_time: datetime, result: Dict):
        """Handle departure completion."""
        # Count what's leaving
        containers_departing = train.get_container_count()
        pickups_completed = len(train.get_all_pickup_container_ids())
        
        # Reset train for reuse
        self._reset_train(train)
        del self.active_trains[train.train_id]
        
        result['action'] = 'train_departed'
        result['containers_departing'] = containers_departing
        result['pickups_completed'] = pickups_completed
        
        self.metrics['total_departures'] += 1
        self.metrics['trains_processed'] += 1
    
    def _calculate_wagon_utilization(self, train: Train) -> Tuple[int, int]:
        """Calculate wagon utilization statistics."""
        utilized = 0
        underutilized = 0
        
        for wagon in train.wagons:
            if wagon._used_length > 0:
                utilization = wagon._used_length / wagon.length
                if utilization >= TRAIN_MIN_WAGON_UTILIZATION:
                    utilized += 1
                else:
                    underutilized += 1
        
        return utilized, underutilized
    
    def _reset_train(self, train: Train):
        """Reset train for reuse."""
        # Clear all wagon data
        for wagon in train.wagons:
            wagon.containers.clear()
            wagon.pickup_container_ids.clear()
            wagon._used_length = 0.0
        
        # Clear train indexes
        train.container_locations.clear()
        train.wagons_with_space = set(range(len(train.wagons)))
        train.empty_wagons = set(range(len(train.wagons)))
        train._total_containers = 0
        train._total_pickup_ids = 0
        
        # Reset timing
        train.arrival_time = None
        train.departure_time = None
        train.loading_start_time = None
        train.loading_complete_time = None
        train.status = "arriving"
    
    def get_weekly_gantt(self) -> plt.Figure:
        """Generate weekly Gantt chart of train schedule."""
        fig, ax = plt.subplots(figsize=(GANTT_FIGURE_WIDTH, GANTT_FIGURE_HEIGHT))
        
        colors = plt.cm.Set3(np.linspace(0, 1, GANTT_COLORS_COUNT))
        
        # Group by rail
        rail_schedules = defaultdict(list)
        for schedule in self.abstract_schedules:
            if schedule.rail is not None:
                rail_schedules[schedule.rail].append(schedule)
        
        # Plot schedules
        for rail, schedules in rail_schedules.items():
            for schedule in schedules:
                self._plot_schedule(ax, schedule, rail, colors)
        
        # Format chart
        self._format_gantt_chart(ax)
        
        plt.tight_layout()
        return fig
    
    def _plot_schedule(self, ax, schedule: AbstractTrainSchedule, 
                       rail: int, colors):
        """Plot a single schedule on the Gantt chart."""
        start_min = schedule.arrival.minutes_in_week()
        end_min = schedule.departure_complete.minutes_in_week()
        color = colors[hash(schedule.train_id) % len(colors)]
        
        if end_min < start_min:  # Week wrap
            # Draw two rectangles
            self._draw_schedule_rect(ax, start_min / HOUR_MINUTES, 
                                    (WEEK_MINUTES - start_min) / HOUR_MINUTES,
                                    rail, color)
            self._draw_schedule_rect(ax, 0, end_min / HOUR_MINUTES, 
                                    rail, color)
        else:
            # Single rectangle
            duration_hours = (end_min - start_min) / HOUR_MINUTES
            self._draw_schedule_rect(ax, start_min / HOUR_MINUTES, 
                                    duration_hours, rail, color)
            
            # Add label if space permits
            if duration_hours > GANTT_LABEL_MIN_DURATION_HOURS:
                label_pos = start_min / HOUR_MINUTES + duration_hours / 2
                ax.text(label_pos, rail + GANTT_RECT_HEIGHT / 2,
                       schedule.train_id.split('_')[0],
                       ha='center', va='center', fontsize=7)
    
    def _draw_schedule_rect(self, ax, x: float, width: float, 
                           rail: int, color):
        """Draw a rectangle for a schedule."""
        rect = patches.Rectangle(
            (x, rail),
            width,
            GANTT_RECT_HEIGHT,
            linewidth=1,
            edgecolor='black',
            facecolor=color,
            alpha=0.7
        )
        ax.add_patch(rect)
    
    def _format_gantt_chart(self, ax):
        """Format the Gantt chart axes and labels."""
        ax.set_ylim(-0.5, self.num_rails - 0.5)
        ax.set_xlim(0, 7 * 24)
        ax.set_xlabel('Time (hours from Monday 00:00)')
        ax.set_ylabel('Rail')
        ax.set_title('Weekly Train Schedule (Repeats Every Week)')
        
        # Rail labels
        ax.set_yticks(range(self.num_rails))
        ax.set_yticklabels([f'{RAIL_PREFIX}{i}' for i in range(self.num_rails)])
        
        # Day markers
        for day in range(8):
            ax.axvline(x=day * 24, color='gray', 
                      linestyle='--', alpha=GANTT_DAYLINE_ALPHA)
            if day < 7:
                ax.text((day + 0.5) * 24, -1, WEEKDAY_FULL_NAMES[day],
                       ha='center', va='top', fontsize=9)
        
        ax.grid(True, alpha=GANTT_GRID_ALPHA, axis='x')
    
    def get_next_train_arrival(self, current_time: datetime) -> Optional[Tuple[datetime, AbstractTrainSchedule]]:
        """Get next train arrival after current time."""
        current_week_minutes = self._get_week_minutes(current_time)
        
        next_arrival = None
        next_schedule = None
        min_wait = float('inf')
        
        for schedule in self.abstract_schedules:
            if schedule.rail is None:
                continue
            
            arrival_minutes = schedule.arrival.minutes_in_week()
            wait = self._calculate_wait_time(current_week_minutes, arrival_minutes)
            
            if wait < min_wait:
                min_wait = wait
                next_schedule = schedule
                next_arrival = current_time + timedelta(minutes=wait)
        
        return (next_arrival, next_schedule) if next_schedule else None
    
    def _get_week_minutes(self, dt: datetime) -> int:
        """Convert datetime to minutes from week start."""
        weekday = dt.weekday()
        return weekday * DAY_MINUTES + dt.hour * HOUR_MINUTES + dt.minute
    
    def _calculate_wait_time(self, current_minutes: int, target_minutes: int) -> int:
        """Calculate wait time to target, handling week wrap."""
        if target_minutes > current_minutes:
            return target_minutes - current_minutes
        else:
            return WEEK_MINUTES - current_minutes + target_minutes
    
    def get_trains_on_day(self, weekday: int) -> List[AbstractTrainSchedule]:
        """Get all trains scheduled for a specific weekday."""
        day_start = weekday * DAY_MINUTES
        day_end = (weekday + 1) * DAY_MINUTES
        
        trains = []
        for schedule in self.abstract_schedules:
            if self._train_on_day(schedule, day_start, day_end):
                trains.append(schedule)
        
        return sorted(trains, key=lambda s: s.arrival.minutes_in_week())
    
    def _train_on_day(self, schedule: AbstractTrainSchedule, 
                     day_start: int, day_end: int) -> bool:
        """Check if train is present on a specific day."""
        arrival_min = schedule.arrival.minutes_in_week()
        departure_min = schedule.departure_complete.minutes_in_week()
        
        if departure_min < arrival_min:  # Wraps around week
            return arrival_min < day_end or departure_min >= day_start
        else:
            return arrival_min < day_end and departure_min >= day_start
    
    def get_metrics(self) -> Dict[str, int]:
        """Return performance metrics."""
        return self.metrics.copy()
    
    def get_schedule_summary(self) -> Dict[str, Any]:
        """Get comprehensive schedule summary."""
        summary = {
            'total_weekly_trains': len(self.abstract_schedules),
            'trains_by_operator': defaultdict(int),
            'trains_by_weekday': defaultdict(int),
            'rail_utilization': {},
            'assigned_trains': 0,
            'unassigned_trains': []
        }
        
        # Analyze schedules
        for schedule in self.abstract_schedules:
            summary['trains_by_operator'][schedule.operator] += 1
            summary['trains_by_weekday'][schedule.arrival.weekday] += 1
            
            if schedule.rail is None:
                summary['unassigned_trains'].append(schedule.train_id)
            else:
                summary['assigned_trains'] += 1
        
        # Calculate rail utilization
        for rail in range(self.num_rails):
            summary['rail_utilization'][f'{RAIL_PREFIX}{rail}'] = \
                self._calculate_rail_utilization(rail)
        
        return summary
    
    def _calculate_rail_utilization(self, rail: int) -> Dict[str, float]:
        """Calculate utilization for a specific rail."""
        total_minutes = 0
        
        for schedule in self.abstract_schedules:
            if schedule.rail == rail:
                total_minutes += schedule.get_duration_minutes()
        
        return {
            'minutes_per_week': total_minutes,
            'hours_per_week': total_minutes / HOUR_MINUTES,
            'utilization_percent': (total_minutes / WEEK_MINUTES) * 100
        }
    
    def analyze_train_capacity(self, operator: str) -> Dict[str, Any]:
        """
        Analyze the capacity and container distribution for a specific operator.
        Shows how many containers of each type can fit on their trains.
        """
        analysis = {
            'operator': operator,
            'wagons_per_train': TRAIN_NUM_WAGONS,
            'wagon_length_ft': TRAIN_WAGON_LENGTH_FT,
            'min_utilization': TRAIN_MIN_WAGON_UTILIZATION * 100,
            'sample_configurations': []
        }
        
        # Generate sample configurations
        test_configs = [
            {'name': 'All 20ft', 'sizes': [20]},
            {'name': 'All 40ft', 'sizes': [40]},
            {'name': 'Mixed 20/40ft', 'sizes': [20, 40]},
            {'name': 'Mixed 30/45ft', 'sizes': [30, 45]},
        ]
        
        for config in test_configs:
            total_containers = 0
            wagon_configs = []
            
            for wagon_num in range(TRAIN_NUM_WAGONS):
                used_length = 0
                containers_in_wagon = []
                
                for size in config['sizes'] * 10:  # Repeat pattern
                    if used_length + size <= TRAIN_WAGON_LENGTH_FT:
                        if used_length + size >= TRAIN_WAGON_LENGTH_FT * TRAIN_MIN_WAGON_UTILIZATION:
                            # Wagon would be sufficiently utilized
                            containers_in_wagon.append(size)
                            used_length += size
                            break
                        else:
                            containers_in_wagon.append(size)
                            used_length += size
                
                if used_length >= TRAIN_WAGON_LENGTH_FT * TRAIN_MIN_WAGON_UTILIZATION:
                    total_containers += len(containers_in_wagon)
                    wagon_configs.append({
                        'containers': containers_in_wagon,
                        'utilization': (used_length / TRAIN_WAGON_LENGTH_FT) * 100
                    })
            
            analysis['sample_configurations'].append({
                'name': config['name'],
                'total_containers': total_containers,
                'wagons_used': len(wagon_configs),
                'avg_utilization': sum(w['utilization'] for w in wagon_configs) / len(wagon_configs) if wagon_configs else 0
            })
        
        return analysis
    
    def get_pickup_queue_status(self) -> Dict[str, Any]:
        """Get status of the container pickup queue."""
        wagon_distribution = defaultdict(int)
        for assignment in self.pickup_queue.values():
            wagon_distribution[assignment.wagon_index] += 1
        
        return {
            'total_pending_pickups': len(self.pickup_queue),
            'distribution_by_wagon': dict(wagon_distribution),
            'container_ids': list(self.pickup_queue.keys())[:10],  # First 10 for preview
            'next_train_can_handle': min(len(self.pickup_queue), TRAIN_NUM_WAGONS * 3)
        }
    
    def simulate_train_loading(self, operator: str) -> Dict[str, Any]:
        """
        Simulate loading a train for a specific operator.
        Shows the actual container distribution based on the factory.
        """
        current_time = datetime.now()
        containers = self.generate_arriving_containers(operator, current_time)
        
        # Analyze the generated containers
        wagon_analysis = []
        current_wagon = []
        current_length = 0
        wagon_idx = 0
        
        for container in containers:
            if current_length + container.length_m <= TRAIN_WAGON_LENGTH_M:
                current_wagon.append(container)
                current_length += container.length_m
            else:
                # Save wagon analysis
                if current_wagon:
                    wagon_analysis.append({
                        'wagon_index': wagon_idx,
                        'containers': len(current_wagon),
                        'types': [c.container_type for c in current_wagon],
                        'utilization': (current_length / TRAIN_WAGON_LENGTH_M) * 100
                    })
                    wagon_idx += 1
                
                # Start new wagon
                current_wagon = [container]
                current_length = container.length_m
        
        # Don't forget last wagon
        if current_wagon:
            wagon_analysis.append({
                'wagon_index': wagon_idx,
                'containers': len(current_wagon),
                'types': [c.container_type for c in current_wagon],
                'utilization': (current_length / TRAIN_WAGON_LENGTH_M) * 100
            })
        
        # Container type distribution
        type_distribution = defaultdict(int)
        for container in containers:
            type_distribution[container.container_type] += 1
        
        return {
            'operator': operator,
            'total_containers': len(containers),
            'wagons_loaded': len(wagon_analysis),
            'container_type_distribution': dict(type_distribution),
            'average_wagon_utilization': sum(w['utilization'] for w in wagon_analysis) / len(wagon_analysis) if wagon_analysis else 0,
            'sample_wagons': wagon_analysis[:5]  # First 5 wagons as sample
        }