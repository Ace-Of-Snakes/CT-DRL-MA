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


class EventType(Enum):
    """Types of train events."""
    ARRIVAL_PREP = "arrival_prep"  # 30 minutes before arrival - generate containers
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
        weekdays = {
            'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
            'Friday': 4, 'Saturday': 5, 'Sunday': 6
        }
        
        # Handle typos
        weekday_str = weekday_str.replace('Tueday', 'Tuesday').strip()
        weekday = weekdays.get(weekday_str, 0)
        
        # Parse time
        time_str = time_str.strip()
        if ':' in time_str:
            parts = time_str.split(':')
            hour = int(parts[0])
            minute = int(parts[1][:2]) if len(parts[1]) >= 2 else 0
        else:
            if len(time_str) == 3:
                time_str = '0' + time_str
            elif len(time_str) != 4:
                hour, minute = 0, 0
            else:
                hour = int(time_str[:2])
                minute = int(time_str[2:])
        
        # Handle hour overflow
        if hour >= 24:
            hour = hour % 24
            weekday = (weekday + 1) % 7
        
        return cls(weekday, hour, minute)
    
    def to_datetime(self, reference_date: datetime) -> datetime:
        """Convert to concrete datetime given a reference date."""
        # Find the next occurrence of this weekday
        days_ahead = self.weekday - reference_date.weekday()
        if days_ahead < 0:
            days_ahead += 7
        
        target_date = reference_date.date() + timedelta(days=days_ahead)
        return datetime.combine(target_date, time(self.hour, self.minute))
    
    def minutes_in_week(self) -> int:
        """Get total minutes from start of week (Monday 00:00)."""
        return self.weekday * 24 * 60 + self.hour * 60 + self.minute
    
    def __str__(self) -> str:
        weekdays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        return f"{weekdays[self.weekday]} {self.hour:02d}:{self.minute:02d}"


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
    rail: Optional[int] = None  # Assigned during rail allocation
    
    def get_occupancy_minutes(self) -> Tuple[int, int]:
        """Get occupancy period in minutes from week start."""
        return (self.arrival.minutes_in_week(), 
                self.departure_complete.minutes_in_week())
    
    def overlaps_with(self, other: 'AbstractTrainSchedule') -> bool:
        """Check if this schedule overlaps with another."""
        s1_start = self.arrival.minutes_in_week()
        s1_end = self.departure_complete.minutes_in_week() 
        s2_start = other.arrival.minutes_in_week()
        s2_end = other.departure_complete.minutes_in_week()
        
        week_minutes = 7 * 24 * 60
        
        # Check both normal and wrapped cases
        def ranges_overlap(a_start, a_end, b_start, b_end):
            return not (a_end <= b_start or b_end <= a_start)
        
        # Case 1: Neither wraps
        if s1_end >= s1_start and s2_end >= s2_start:
            return ranges_overlap(s1_start, s1_end, s2_start, s2_end)
        
        # Case 2: s1 wraps around week
        if s1_end < s1_start:
            if s2_end < s2_start:
                # Both wrap - they definitely overlap
                return True
            else:
                # Only s1 wraps
                return (ranges_overlap(s1_start, week_minutes, s2_start, s2_end) or
                        ranges_overlap(0, s1_end, s2_start, s2_end))
        
        # Case 3: s2 wraps around week  
        if s2_end < s2_start:
            # Only s2 wraps
            return (ranges_overlap(s1_start, s1_end, s2_start, week_minutes) or
                    ranges_overlap(s1_start, s1_end, 0, s2_end))
        
        return False


@dataclass
class ConcreteTrainEvent:
    """Concrete event for processing during simulation."""
    timestamp: datetime
    event_type: EventType
    schedule: AbstractTrainSchedule
    
    def __lt__(self, other):
        return self.timestamp < other.timestamp


class TrainScheduler:
    """
    Train scheduler with abstract weekly schedule.
    Manages train arrivals, departures, and container generation.
    """
    
    def __init__(self, 
                 driving_plan_path: str,
                 num_rails: int = 10,  # Increased default from 4 to 10
                 container_factory: Optional[ContainerFactory] = None,
                 prep_time_minutes: int = 30):
        """
        Initialize the train scheduler with abstract schedule.
        
        Args:
            driving_plan_path: Path to the driving plan JSON
            num_rails: Number of available rails
            container_factory: Factory for container generation
            prep_time_minutes: Minutes before arrival to generate containers
        """
        self.num_rails = num_rails
        self.prep_time = timedelta(minutes=prep_time_minutes)
        
        # Load driving plan
        with open(driving_plan_path, 'r') as f:
            self.driving_plan = json.load(f)['driving_plan']
        
        # Initialize container factory
        self.container_factory = container_factory or ContainerFactory()
        
        # Abstract weekly schedule
        self.abstract_schedules: List[AbstractTrainSchedule] = []
        
        # Rail assignments (schedule -> rail)
        self.rail_assignments: Dict[str, int] = {}
        
        # Train pool - reusable Train objects
        self.train_pool: Dict[str, Train] = {}
        
        # Active trains tracking
        self.active_trains: Dict[str, Train] = {}
        
        # Concrete event queue for simulation
        self.event_queue: List[ConcreteTrainEvent] = []
        
        # Performance metrics
        self.metrics = {
            'total_arrivals': 0,
            'total_departures': 0,
            'containers_generated': 0,
            'containers_left_on_train': 0,
            'pickup_containers_distributed': 0
        }
        
        # Build abstract schedule
        self._build_abstract_schedule()
        
        # Assign rails to schedules
        self._assign_rails()
    
    def _build_abstract_schedule(self):
        """Build abstract weekly schedule from driving plan."""
        for train_code, train_info in self.driving_plan['trains'].items():
            operator = train_info.get('operator', 'Unknown')
            destination = train_info.get('destination', 'Unknown')
            
            # Pre-allocate Train object
            if train_code not in self.train_pool:
                train = Train(
                    train_id=train_code,
                    num_wagons=10,
                    wagon_length=24.384
                )
                # Add custom attributes
                train.operator = operator
                train.destination = destination
                self.train_pool[train_code] = train
            
            # Parse each plan entry
            for plan_key, plan_entry in train_info['plan'].items():
                arrival = plan_entry['arrival']
                departure = plan_entry['departure']
                mirrored_days = plan_entry.get('mirrored_on', [])
                
                # Parse arrival times
                arr_day, arr_time, arr_complete_day, arr_complete_time = arrival
                arrival_wt = WeekdayTime.from_string(arr_day, arr_time)
                arrival_complete_wt = WeekdayTime.from_string(arr_complete_day, arr_complete_time)
                
                # Parse departure times
                dep_day, dep_time, dep_complete_day, dep_complete_time = departure
                departure_wt = WeekdayTime.from_string(dep_day, dep_time)
                departure_complete_wt = WeekdayTime.from_string(dep_complete_day, dep_complete_time)
                
                # Create schedule for main day
                schedule = AbstractTrainSchedule(
                    train_id=f"{train_code}_{plan_key}",
                    operator=operator,
                    destination=destination,
                    arrival=arrival_wt,
                    arrival_complete=arrival_complete_wt,
                    departure=departure_wt,
                    departure_complete=departure_complete_wt
                )
                self.abstract_schedules.append(schedule)
                
                # Create schedules for mirrored days
                for mirror_day in mirrored_days:
                    # Calculate day offset
                    mirror_arrival_wt = WeekdayTime.from_string(mirror_day, arr_time)
                    
                    # Adjust other times relative to new arrival day
                    day_diff = mirror_arrival_wt.weekday - arrival_wt.weekday
                    
                    mirror_schedule = AbstractTrainSchedule(
                        train_id=f"{train_code}_{plan_key}_mirror_{mirror_day}",
                        operator=operator,
                        destination=destination,
                        arrival=mirror_arrival_wt,
                        arrival_complete=WeekdayTime(
                            (arrival_complete_wt.weekday + day_diff) % 7,
                            arrival_complete_wt.hour,
                            arrival_complete_wt.minute
                        ),
                        departure=WeekdayTime(
                            (departure_wt.weekday + day_diff) % 7,
                            departure_wt.hour,
                            departure_wt.minute
                        ),
                        departure_complete=WeekdayTime(
                            (departure_complete_wt.weekday + day_diff) % 7,
                            departure_complete_wt.hour,
                            departure_complete_wt.minute
                        )
                    )
                    self.abstract_schedules.append(mirror_schedule)
    
    def _assign_rails(self):
        """Assign rails to abstract schedules using improved greedy algorithm."""
        print(f"\nAssigning {len(self.abstract_schedules)} schedules to {self.num_rails} rails...")
        
        # Sort schedules by arrival time in week
        sorted_schedules = sorted(
            self.abstract_schedules, 
            key=lambda s: s.arrival.minutes_in_week()
        )
        
        # Track rail occupancies
        rail_schedules = {i: [] for i in range(self.num_rails)}
        assigned_count = 0
        unassigned_list = []
        
        # Process each schedule
        for schedule in sorted_schedules:
            assigned = False
            
            # Try each rail
            for rail in range(self.num_rails):
                can_fit = True
                
                # Check against all existing schedules on this rail
                for existing in rail_schedules[rail]:
                    if self._schedules_overlap(schedule, existing):
                        can_fit = False
                        break
                
                if can_fit:
                    # Assign to this rail
                    schedule.rail = rail
                    rail_schedules[rail].append(schedule)
                    self.rail_assignments[schedule.train_id] = rail
                    assigned = True
                    assigned_count += 1
                    break
            
            if not assigned:
                unassigned_list.append(schedule.train_id)
        
        print(f"Successfully assigned {assigned_count} out of {len(self.abstract_schedules)} schedules")
        if unassigned_list:
            print(f"Could not assign {len(unassigned_list)} schedules - may need more rails")
            print(f"Unassigned: {unassigned_list[:5]}..." if len(unassigned_list) > 5 else f"Unassigned: {unassigned_list}")
    
    def _schedules_overlap(self, s1: AbstractTrainSchedule, s2: AbstractTrainSchedule) -> bool:
        """Check if two schedules overlap, handling week wrap-around."""
        # Get time ranges in minutes from week start
        s1_start = s1.arrival.minutes_in_week()
        s1_end = s1.departure_complete.minutes_in_week()
        s2_start = s2.arrival.minutes_in_week()
        s2_end = s2.departure_complete.minutes_in_week()
        
        # Normalize for week wrap-around
        week_minutes = 7 * 24 * 60
        
        # Check both normal and wrapped cases
        def ranges_overlap(a_start, a_end, b_start, b_end):
            return not (a_end <= b_start or b_end <= a_start)
        
        # Case 1: Neither wraps
        if s1_end >= s1_start and s2_end >= s2_start:
            return ranges_overlap(s1_start, s1_end, s2_start, s2_end)
        
        # Case 2: s1 wraps around week
        if s1_end < s1_start:
            # Split s1 into two parts: [start, week_end] and [0, end]
            # Check if s2 overlaps with either part
            if s2_end < s2_start:
                # Both wrap - they definitely overlap
                return True
            else:
                # Only s1 wraps
                return (ranges_overlap(s1_start, week_minutes, s2_start, s2_end) or
                        ranges_overlap(0, s1_end, s2_start, s2_end))
        
        # Case 3: s2 wraps around week
        if s2_end < s2_start:
            # Only s2 wraps
            return (ranges_overlap(s1_start, s1_end, s2_start, week_minutes) or
                    ranges_overlap(s1_start, s1_end, 0, s2_end))
        
        return False
    
    def initialize_for_period(self, start_date: datetime, end_date: datetime):
        """Initialize concrete events for a specific time period."""
        self.event_queue.clear()
        events = []
        
        # Generate events for each week in the period
        current_week_start = start_date - timedelta(days=start_date.weekday())
        
        while current_week_start < end_date:
            for schedule in self.abstract_schedules:
                if schedule.rail is None:
                    continue
                
                # Convert abstract times to concrete for this week
                arrival_dt = schedule.arrival.to_datetime(current_week_start)
                arrival_complete_dt = schedule.arrival_complete.to_datetime(current_week_start)
                departure_dt = schedule.departure.to_datetime(current_week_start)
                departure_complete_dt = schedule.departure_complete.to_datetime(current_week_start)
                
                # Only include events within our period
                if departure_complete_dt >= start_date and arrival_dt <= end_date:
                    events.extend([
                        ConcreteTrainEvent(
                            arrival_dt - self.prep_time, 
                            EventType.ARRIVAL_PREP, 
                            schedule
                        ),
                        ConcreteTrainEvent(arrival_dt, EventType.ARRIVAL, schedule),
                        ConcreteTrainEvent(arrival_complete_dt, EventType.ARRIVAL_COMPLETE, schedule),
                        ConcreteTrainEvent(departure_dt, EventType.DEPARTURE, schedule),
                        ConcreteTrainEvent(departure_complete_dt, EventType.DEPARTURE_COMPLETE, schedule)
                    ])
            
            current_week_start += timedelta(weeks=1)
        
        # Build heap
        for event in events:
            heapq.heappush(self.event_queue, event)
    
    def get_next_train_arrival(self, current_time: datetime) -> Optional[Tuple[datetime, AbstractTrainSchedule]]:
        """Get the next train arrival after current time."""
        current_weekday = current_time.weekday()
        current_minutes = current_time.hour * 60 + current_time.minute
        current_week_minutes = current_weekday * 24 * 60 + current_minutes
        
        next_arrival = None
        next_schedule = None
        min_wait = float('inf')
        
        for schedule in self.abstract_schedules:
            if schedule.rail is None:
                continue
            
            arrival_minutes = schedule.arrival.minutes_in_week()
            
            # Calculate wait time
            if arrival_minutes > current_week_minutes:
                # Same week
                wait = arrival_minutes - current_week_minutes
            else:
                # Next week
                wait = (7 * 24 * 60) - current_week_minutes + arrival_minutes
            
            if wait < min_wait:
                min_wait = wait
                next_schedule = schedule
                
                # Calculate actual datetime
                days_ahead = wait // (24 * 60)
                remaining_minutes = wait % (24 * 60)
                next_arrival = current_time + timedelta(days=days_ahead, minutes=remaining_minutes)
        
        return (next_arrival, next_schedule) if next_schedule else None
    
    def get_trains_on_day(self, weekday: int) -> List[AbstractTrainSchedule]:
        """Get all trains scheduled for a specific weekday."""
        trains = []
        day_start = weekday * 24 * 60
        day_end = (weekday + 1) * 24 * 60
        
        for schedule in self.abstract_schedules:
            # Check if train is present on this day
            arrival_min = schedule.arrival.minutes_in_week()
            departure_min = schedule.departure_complete.minutes_in_week()
            
            # Handle week wrap-around
            if departure_min < arrival_min:
                # Train wraps around week
                if arrival_min < day_end or departure_min >= day_start:
                    trains.append(schedule)
            else:
                # Normal case
                if arrival_min < day_end and departure_min >= day_start:
                    trains.append(schedule)
        
        return sorted(trains, key=lambda s: s.arrival.minutes_in_week())
    
    def process_event(self, event: ConcreteTrainEvent, current_time: datetime) -> Dict[str, Any]:
        """Process a concrete train event."""
        # Get or create train instance
        base_train_id = event.schedule.train_id.split('_')[0]
        
        if base_train_id not in self.active_trains:
            # Get from pool or create new
            if base_train_id in self.train_pool:
                train = self.train_pool[base_train_id]
            else:
                train = Train(
                    train_id=base_train_id,
                    num_wagons=10,
                    wagon_length=24.384
                )
                train.operator = event.schedule.operator
                train.destination = event.schedule.destination
                self.train_pool[base_train_id] = train
            
            self.active_trains[base_train_id] = train
        else:
            train = self.active_trains[base_train_id]
        
        result = {
            'event_type': event.event_type,
            'train_id': base_train_id,
            'rail': event.schedule.rail,
            'timestamp': event.timestamp,
            'action': None
        }
        
        if event.event_type == EventType.ARRIVAL_PREP:
            containers = self._generate_containers_for_train(
                train, event.schedule.operator, current_time
            )
            result['action'] = 'containers_generated'
            result['containers'] = containers
            result['count'] = len(containers)
            self.metrics['containers_generated'] += len(containers)
            
        elif event.event_type == EventType.ARRIVAL:
            train.arrival_time = current_time
            train.rail_track = f"Rail_{event.schedule.rail}"
            train.status = "waiting"
            result['action'] = 'train_arrived'
            self.metrics['total_arrivals'] += 1
            
        elif event.event_type == EventType.ARRIVAL_COMPLETE:
            train.status = "loading"
            result['action'] = 'arrival_complete'
            
        elif event.event_type == EventType.DEPARTURE:
            train.status = "departing"
            result['action'] = 'train_departing'
            
        elif event.event_type == EventType.DEPARTURE_COMPLETE:
            left_containers = train.get_container_count()
            self.metrics['containers_left_on_train'] += left_containers
            
            # Reset and remove from active
            self._reset_train(train)
            del self.active_trains[base_train_id]
            
            result['action'] = 'train_departed'
            result['containers_left'] = left_containers
            self.metrics['total_departures'] += 1
        
        return result
    
    def _generate_containers_for_train(self, train: Train, operator: str, 
                                      current_time: datetime) -> List[Container]:
        """Generate containers for an arriving train."""
        container_counts = {
            'boxXpress': 30,
            'Kombiverkehr': 25,
            'IGS': 20,
            'Hutchinson Ports intermodal': 35,
            'Hellmann': 25,
            'Transfracht/ DB Cargo': 30,
            'DHL/ DB Cargo': 40,
            'Metrans': 35
        }
        
        count = container_counts.get(operator, 25)
        import_count = int(count * 0.7)
        export_count = count - import_count
        
        containers = []
        
        if import_count > 0:
            import_containers = self.container_factory.create_containers(
                operator=self._map_operator_name(operator),
                direction="Import",
                n_containers=import_count,
                base_arrival_date=current_time,
                current_date=current_time
            )
            containers.extend(import_containers)
        
        if export_count > 0:
            export_containers = self.container_factory.create_containers(
                operator=self._map_operator_name(operator),
                direction="Export",
                n_containers=export_count,
                base_arrival_date=current_time,
                current_date=current_time
            )
            containers.extend(export_containers)
        
        for container in containers:
            train.add_container(container)
        
        return containers
    
    def _map_operator_name(self, operator: str) -> str:
        """Map driving plan operator names to container factory names."""
        return "BOX"  # Default mapping
    
    def _reset_train(self, train: Train):
        """Reset train for reuse."""
        for wagon in train.wagons:
            wagon.containers.clear()
            wagon.pickup_container_ids.clear()
            wagon._used_length = 0.0
        
        train.container_locations.clear()
        train.wagons_with_space = set(range(len(train.wagons)))
        train.empty_wagons = set(range(len(train.wagons)))
        train._total_containers = 0
        train._total_pickup_ids = 0
        
        train.arrival_time = None
        train.departure_time = None
        train.loading_start_time = None
        train.loading_complete_time = None
        train.status = "arriving"
    
    def get_weekly_gantt(self) -> plt.Figure:
        """Generate a Gantt chart of the abstract weekly schedule."""
        fig, ax = plt.subplots(figsize=(16, 8))
        
        colors = plt.cm.Set3(np.linspace(0, 1, 12))
        
        # Group schedules by rail
        rail_schedules = defaultdict(list)
        for schedule in self.abstract_schedules:
            if schedule.rail is not None:
                rail_schedules[schedule.rail].append(schedule)
        
        # Plot each schedule
        for rail, schedules in rail_schedules.items():
            for schedule in schedules:
                start_min = schedule.arrival.minutes_in_week()
                end_min = schedule.departure_complete.minutes_in_week()
                
                # Handle week wrap-around
                if end_min < start_min:
                    # Draw two rectangles
                    # First part: from arrival to end of week
                    duration1 = (7 * 24 * 60) - start_min
                    rect1 = patches.Rectangle(
                        (start_min / 60, rail),
                        duration1 / 60,
                        0.8,
                        linewidth=1,
                        edgecolor='black',
                        facecolor=colors[hash(schedule.train_id) % len(colors)],
                        alpha=0.7
                    )
                    ax.add_patch(rect1)
                    
                    # Second part: from start of week to departure
                    rect2 = patches.Rectangle(
                        (0, rail),
                        end_min / 60,
                        0.8,
                        linewidth=1,
                        edgecolor='black',
                        facecolor=colors[hash(schedule.train_id) % len(colors)],
                        alpha=0.7
                    )
                    ax.add_patch(rect2)
                else:
                    # Normal case
                    duration = end_min - start_min
                    rect = patches.Rectangle(
                        (start_min / 60, rail),
                        duration / 60,
                        0.8,
                        linewidth=1,
                        edgecolor='black',
                        facecolor=colors[hash(schedule.train_id) % len(colors)],
                        alpha=0.7
                    )
                    ax.add_patch(rect)
                
                # Add label if space permits
                duration_hours = (end_min - start_min) / 60
                if duration_hours > 2:
                    label_pos = start_min / 60 + duration_hours / 2
                    ax.text(label_pos, rail + 0.4, 
                           schedule.train_id.split('_')[0],
                           ha='center', va='center', fontsize=7)
        
        # Format axes
        ax.set_ylim(-0.5, self.num_rails - 0.5)
        ax.set_xlim(0, 7 * 24)
        ax.set_xlabel('Time (hours from Monday 00:00)')
        ax.set_ylabel('Rail')
        ax.set_title('Weekly Train Schedule (Abstract - Repeats Every Week)')
        
        # Add rail labels
        ax.set_yticks(range(self.num_rails))
        ax.set_yticklabels([f'Rail {i}' for i in range(self.num_rails)])
        
        # Add day markers and labels
        weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        for day in range(8):
            ax.axvline(x=day * 24, color='gray', linestyle='--', alpha=0.5)
            if day < 7:
                ax.text((day + 0.5) * 24, -1, weekdays[day], 
                       ha='center', va='top', fontsize=9)
        
        # Add grid
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        return fig
    
    def get_metrics(self) -> Dict[str, int]:
        """Return performance metrics."""
        return self.metrics.copy()
    
    def analyze_schedule_conflicts(self) -> Dict[str, Any]:
        """Analyze scheduling conflicts to understand rail assignment issues."""
        analysis = {
            'total_schedules': len(self.abstract_schedules),
            'assigned_schedules': sum(1 for s in self.abstract_schedules if s.rail is not None),
            'unassigned_schedules': sum(1 for s in self.abstract_schedules if s.rail is None),
            'rails_used': len(set(s.rail for s in self.abstract_schedules if s.rail is not None)),
            'longest_occupancies': [],
            'overlapping_pairs': []
        }
        
        # Find longest occupancies
        for schedule in self.abstract_schedules:
            duration_min = schedule.departure_complete.minutes_in_week() - schedule.arrival.minutes_in_week()
            if duration_min < 0:  # Wraps around week
                duration_min += 7 * 24 * 60
            
            analysis['longest_occupancies'].append({
                'train_id': schedule.train_id,
                'duration_hours': duration_min / 60,
                'arrival': str(schedule.arrival),
                'departure': str(schedule.departure_complete),
                'assigned_rail': schedule.rail
            })
        
        # Sort by duration
        analysis['longest_occupancies'].sort(key=lambda x: x['duration_hours'], reverse=True)
        analysis['longest_occupancies'] = analysis['longest_occupancies'][:10]  # Top 10
        
        # Find overlapping unassigned pairs
        unassigned = [s for s in self.abstract_schedules if s.rail is None]
        for i, s1 in enumerate(unassigned[:5]):  # Check first 5 unassigned
            for s2 in self.abstract_schedules:
                if s2 != s1 and self._schedules_overlap(s1, s2):
                    analysis['overlapping_pairs'].append({
                        'train1': s1.train_id,
                        'train2': s2.train_id,
                        'train2_rail': s2.rail
                    })
        
        return analysis
    
    def get_schedule_summary(self) -> Dict[str, Any]:
        """Get summary of the abstract schedule."""
        summary = {
            'total_weekly_trains': len(self.abstract_schedules),
            'trains_by_operator': defaultdict(int),
            'trains_by_weekday': defaultdict(int),
            'rail_utilization': {},
            'unassigned_trains': []
        }
        
        for schedule in self.abstract_schedules:
            summary['trains_by_operator'][schedule.operator] += 1
            summary['trains_by_weekday'][schedule.arrival.weekday] += 1
            
            if schedule.rail is None:
                summary['unassigned_trains'].append(schedule.train_id)
        
        # Calculate rail utilization
        for rail in range(self.num_rails):
            rail_minutes = 0
            for schedule in self.abstract_schedules:
                if schedule.rail == rail:
                    start = schedule.arrival.minutes_in_week()
                    end = schedule.departure_complete.minutes_in_week()
                    if end < start:
                        end += 7 * 24 * 60
                    rail_minutes += (end - start)
            
            summary['rail_utilization'][f'Rail_{rail}'] = {
                'minutes_per_week': rail_minutes,
                'hours_per_week': rail_minutes / 60,
                'utilization_percent': (rail_minutes / (7 * 24 * 60)) * 100
            }
        
        return summary
    
"""
Test script for the refactored TrainScheduler with abstract weekly schedule.
"""

from datetime import datetime, timedelta
import json

# Assuming the refactored TrainScheduler is imported
# from simulation.terminal_components.systems.TrainScheduler import TrainScheduler

def test_abstract_scheduler():
    """Test the abstract scheduler functionality."""
    
    print("=" * 80)
    print("TESTING REFACTORED TRAIN SCHEDULER WITH ABSTRACT WEEKLY SCHEDULE")
    print("=" * 80)
    
    # Initialize scheduler
    scheduler = TrainScheduler(
        driving_plan_path="simulation/data/driving_plan.json",
        num_rails=6,  # Increased from 6 to 10 for better capacity
        prep_time_minutes=30
    )
    
    # 1. Show schedule summary
    print("\n1. ABSTRACT SCHEDULE SUMMARY")
    print("-" * 40)
    summary = scheduler.get_schedule_summary()
    print(f"Total weekly trains: {summary['total_weekly_trains']}")
    print(f"Unassigned trains: {len(summary['unassigned_trains'])}")
    
    print("\nTrains by operator:")
    for operator, count in summary['trains_by_operator'].items():
        print(f"  {operator}: {count}")
    
    print("\nTrains by weekday:")
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    for day_idx, count in summary['trains_by_weekday'].items():
        print(f"  {weekdays[day_idx]}: {count}")
    
    print("\nRail utilization:")
    for rail, util in summary['rail_utilization'].items():
        print(f"  {rail}: {util['utilization_percent']:.1f}% ({util['hours_per_week']:.1f} hours/week)")
    
    # Add conflict analysis
    print("\n1b. CONFLICT ANALYSIS")
    print("-" * 40)
    conflicts = scheduler.analyze_schedule_conflicts()
    print(f"Assigned: {conflicts['assigned_schedules']}/{conflicts['total_schedules']}")
    print(f"Rails used: {conflicts['rails_used']}/{scheduler.num_rails}")
    print("\nLongest occupancies:")
    for occ in conflicts['longest_occupancies'][:5]:
        rail_str = f"Rail {occ['assigned_rail']}" if occ['assigned_rail'] is not None else "UNASSIGNED"
        print(f"  {occ['train_id']}: {occ['duration_hours']:.1f} hours ({occ['arrival']} to {occ['departure']}) - {rail_str}")
    
    # 2. Test weekday queries
    print("\n2. WEEKDAY QUERIES")
    print("-" * 40)
    
    # Get trains for Monday
    monday_trains = scheduler.get_trains_on_day(0)  # 0 = Monday
    print(f"Trains on Monday: {len(monday_trains)}")
    if monday_trains:
        print("First 3 Monday trains:")
        for train in monday_trains[:3]:
            print(f"  {train.train_id}: arrives {train.arrival}, departs {train.departure_complete}")
    
    # Get trains for Friday
    friday_trains = scheduler.get_trains_on_day(4)  # 4 = Friday
    print(f"\nTrains on Friday: {len(friday_trains)}")
    
    # 3. Test next train lookup
    print("\n3. NEXT TRAIN LOOKUP")
    print("-" * 40)
    
    # Test for different times
    test_times = [
        datetime(2024, 1, 1, 8, 0),   # Monday 8:00
        datetime(2024, 1, 3, 14, 30), # Wednesday 14:30
        datetime(2024, 1, 5, 20, 0),  # Friday 20:00
        datetime(2024, 1, 7, 23, 45), # Sunday 23:45
    ]
    
    for test_time in test_times:
        result = scheduler.get_next_train_arrival(test_time)
        if result:
            next_arrival, schedule = result
            wait_time = (next_arrival - test_time).total_seconds() / 3600
            print(f"From {test_time.strftime('%A %H:%M')}:")
            print(f"  Next train: {schedule.train_id.split('_')[0]}")
            print(f"  Arrives: {next_arrival.strftime('%A %H:%M')} (in {wait_time:.1f} hours)")
            print(f"  On rail: {schedule.rail}")
    
    # 4. Test concrete event generation
    print("\n4. CONCRETE EVENT GENERATION")
    print("-" * 40)
    
    # Initialize for a specific week
    sim_start = datetime(2024, 1, 8)  # Monday, Jan 8, 2024
    sim_end = sim_start + timedelta(days=7)
    
    scheduler.initialize_for_period(sim_start, sim_end)
    print(f"Generated {len(scheduler.event_queue)} events for the week")
    
    # Show first 10 events
    print("\nFirst 10 events:")
    temp_queue = list(scheduler.event_queue)[:10]
    for event in sorted(temp_queue):
        print(f"  {event.timestamp.strftime('%a %H:%M')} - {event.event_type.value} - {event.schedule.train_id.split('_')[0]}")
    
    # 5. Simulate one day
    print("\n5. SIMULATING ONE DAY")
    print("-" * 40)
    
    sim_current = sim_start
    sim_day_end = sim_start + timedelta(days=1)
    events_processed = 0
    
    print(f"Simulating Monday, {sim_start.strftime('%Y-%m-%d')}...")
    
    while scheduler.event_queue and scheduler.event_queue[0].timestamp < sim_day_end:
        event = scheduler.event_queue[0]
        if event.timestamp <= sim_current:
            event = scheduler.event_queue.pop(0)
            result = scheduler.process_event(event, sim_current)
            
            if result['action'] == 'containers_generated':
                print(f"  {event.timestamp.strftime('%H:%M')} - Generated {result['count']} containers for {event.schedule.train_id.split('_')[0]}")
            elif result['action'] == 'train_arrived':
                print(f"  {event.timestamp.strftime('%H:%M')} - Train {event.schedule.train_id.split('_')[0]} arrived on Rail {event.schedule.rail}")
            elif result['action'] == 'train_departed':
                print(f"  {event.timestamp.strftime('%H:%M')} - Train {event.schedule.train_id.split('_')[0]} departed ({result['containers_left']} containers left)")
            
            events_processed += 1
        
        # Advance time to next event
        if scheduler.event_queue:
            sim_current = min(scheduler.event_queue[0].timestamp, sim_day_end)
        else:
            break
    
    print(f"\nProcessed {events_processed} events")
    print(f"Metrics: {scheduler.get_metrics()}")
    
    # 6. Generate Gantt chart
    print("\n6. GENERATING WEEKLY GANTT CHART")
    print("-" * 40)
    
    fig = scheduler.get_weekly_gantt()
    fig.savefig("weekly_train_schedule.png", dpi=150, bbox_inches='tight')
    print("Gantt chart saved as 'weekly_train_schedule.png'")
    print("This chart represents the abstract weekly schedule that repeats every week.")
    
    # 7. Test abstract schedule properties
    print("\n7. ABSTRACT SCHEDULE PROPERTIES")
    print("-" * 40)
    
    # Show some example schedules
    print("Example abstract schedules:")
    for schedule in scheduler.abstract_schedules[:3]:
        print(f"\nTrain: {schedule.train_id}")
        print(f"  Operator: {schedule.operator}")
        print(f"  Arrival: {schedule.arrival} (minute {schedule.arrival.minutes_in_week()} of week)")
        print(f"  Departure: {schedule.departure_complete} (minute {schedule.departure_complete.minutes_in_week()} of week)")
        print(f"  Rail: {schedule.rail}")
        print(f"  Occupancy: {(schedule.departure_complete.minutes_in_week() - schedule.arrival.minutes_in_week()) / 60:.1f} hours")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    # Create a minimal driving plan for testing if the file doesn't exist
    test_driving_plan = {
        "driving_plan": {
            "trains": {
                "TRN001": {
                    "operator": "boxXpress",
                    "destination": "Hamburg",
                    "plan": {
                        "1": {
                            "arrival": ["Monday", "08:00", "Monday", "08:30"],
                            "departure": ["Monday", "14:00", "Monday", "14:30"],
                            "mirrored_on": ["Wednesday", "Friday"]
                        }
                    }
                },
                "TRN002": {
                    "operator": "Kombiverkehr",
                    "destination": "Munich",
                    "plan": {
                        "1": {
                            "arrival": ["Tuesday", "10:00", "Tuesday", "10:30"],
                            "departure": ["Tuesday", "16:00", "Tuesday", "16:30"],
                            "mirrored_on": ["Thursday"]
                        }
                    }
                },
                "TRN003": {
                    "operator": "Metrans",
                    "destination": "Prague",
                    "plan": {
                        "1": {
                            "arrival": ["Monday", "15:00", "Monday", "15:30"],
                            "departure": ["Tuesday", "09:00", "Tuesday", "09:30"],
                            "mirrored_on": []
                        }
                    }
                }
            }
        }
    }
    
    # Save test driving plan if needed
    import os
    if not os.path.exists("simulation/data/driving_plan.json"):
        import json
        os.makedirs("simulation/data", exist_ok=True)
        with open("simulation/data/driving_plan.json", "w") as f:
            json.dump(test_driving_plan, f, indent=2)
        print("Created test driving_plan.json")
    
    # Run tests
    test_abstract_scheduler()