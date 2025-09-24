import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from simulation.terminal_components.vehicles.Train import Train
from simulation.terminal_components.systems.train_tools.TimeEncoder import WeeklyTimeEncoder
from simulation.terminal_components.systems.train_tools.DPParser import DrivingPlanParser

# Constants
WEEK_SECONDS = 604800
HOUR_SECONDS = 3600
BUFFER_TIME_HOURS = 4  # Buffer between trains on same track

@dataclass
class ScheduledTrain:
    """Data class for a scheduled train."""
    train: Train
    track_id: int
    arrival_seconds: float  # Seconds from week start
    departure_seconds: float  # Seconds from week start
    arrival_angle: float  # Radians for visualization
    departure_angle: float  # Radians for visualization
    operator: str
    stay_hours: float
    
    @property
    def duration_seconds(self) -> float:
        """Get duration handling weekly wraparound."""
        if self.departure_seconds < self.arrival_seconds:
            return (WEEK_SECONDS - self.arrival_seconds) + self.departure_seconds
        return self.departure_seconds - self.arrival_seconds

@dataclass
class TrainSchedule:
    """Container for the complete schedule."""
    num_tracks: int
    scheduled_trains: List[ScheduledTrain] = field(default_factory=list)
    unscheduled_trains: List[Train] = field(default_factory=list)
    track_occupancy: Dict[int, List[ScheduledTrain]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize track occupancy."""
        for i in range(self.num_tracks):
            if i not in self.track_occupancy:
                self.track_occupancy[i] = []
    
    def add_train(self, scheduled_train: ScheduledTrain):
        """Add a scheduled train to the schedule."""
        self.scheduled_trains.append(scheduled_train)
        track_id = scheduled_train.track_id
        if track_id not in self.track_occupancy:
            self.track_occupancy[track_id] = []
        self.track_occupancy[track_id].append(scheduled_train)
    
    def get_utilization(self) -> float:
        """Calculate overall track utilization."""
        total_capacity = self.num_tracks * WEEK_SECONDS
        total_used = sum(st.duration_seconds for st in self.scheduled_trains)
        return (total_used / total_capacity) * 100 if total_capacity > 0 else 0
    
    def get_track_utilization(self, track_id: int) -> float:
        """Calculate utilization for specific track."""
        if track_id not in self.track_occupancy:
            return 0
        used = sum(st.duration_seconds for st in self.track_occupancy[track_id])
        return (used / WEEK_SECONDS) * 100


class TrainScheduler:
    """Scheduler for optimizing train placement on tracks."""
    
    def __init__(self, num_tracks: int = 4):
        """Initialize scheduler with number of available tracks."""
        self.num_tracks = num_tracks
        self.encoder = WeeklyTimeEncoder()
        self.buffer_seconds = BUFFER_TIME_HOURS * HOUR_SECONDS
    
    def schedule_trains(self, trains: List[Train]) -> TrainSchedule:
        """
        Schedule trains using a best-fit algorithm with conflict resolution.
        
        Args:
            trains: List of Train objects with schedule_encoded metadata
            
        Returns:
            TrainSchedule object with scheduled and unscheduled trains
        """
        schedule = TrainSchedule(num_tracks=self.num_tracks)
        
        # Sort trains by stay duration (longer stays first for better packing)
        sorted_trains = sorted(
            trains,
            key=lambda t: t.schedule_encoded['stay_duration']['hours'],
            reverse=True
        )
        
        for train in sorted_trains:
            best_track = self._find_best_track(train, schedule)
            
            if best_track is not None:
                scheduled = self._create_scheduled_train(train, best_track)
                schedule.add_train(scheduled)
            else:
                schedule.unscheduled_trains.append(train)
        
        return schedule
    
    def _find_best_track(self, train: Train, schedule: TrainSchedule) -> Optional[int]:
        """
        Find the best track for a train using least-waste strategy.
        
        Returns:
            Track ID or None if no track available
        """
        train_arrival = train.schedule_encoded['arrival']['seconds']
        train_departure = train.schedule_encoded['departure']['seconds']
        
        best_track = None
        min_waste = float('inf')
        
        for track_id in range(self.num_tracks):
            if self._can_fit_on_track(train_arrival, train_departure, 
                                     schedule.track_occupancy[track_id]):
                # Calculate wasted space if placed on this track
                waste = self._calculate_waste(train_arrival, train_departure,
                                             schedule.track_occupancy[track_id])
                if waste < min_waste:
                    min_waste = waste
                    best_track = track_id
        
        return best_track
    
    def _can_fit_on_track(self, arrival: float, departure: float, 
                         track_trains: List[ScheduledTrain]) -> bool:
        """Check if train can fit on track without conflicts."""
        for scheduled in track_trains:
            if self._has_conflict(arrival, departure, 
                                 scheduled.arrival_seconds, 
                                 scheduled.departure_seconds):
                return False
        return True
    
    def _has_conflict(self, arr1: float, dep1: float, 
                    arr2: float, dep2: float) -> bool:
        """
        Check if two trains have time conflict with buffer.
        Handles weekly wraparound.
        """
        # Add buffer
        arr1_buffered = (arr1 - self.buffer_seconds) % WEEK_SECONDS
        dep1_buffered = (dep1 + self.buffer_seconds) % WEEK_SECONDS
        
        # Helper function to check if a point is within an interval (handling wraparound)
        def point_in_interval(point: float, start: float, end: float) -> bool:
            if start <= end:
                # Normal interval
                return start <= point <= end
            else:
                # Wraparound interval
                return point >= start or point <= end
        
        # Check if any part of train1's buffered interval overlaps with train2's interval
        # or any part of train2's interval overlaps with train1's buffered interval
        
        # Check if train1's arrival or departure falls within train2's interval
        if point_in_interval(arr1_buffered, arr2, dep2):
            return True
        if point_in_interval(dep1_buffered, arr2, dep2):
            return True
        
        # Check if train2's arrival or departure falls within train1's buffered interval
        if point_in_interval(arr2, arr1_buffered, dep1_buffered):
            return True
        if point_in_interval(dep2, arr1_buffered, dep1_buffered):
            return True
        
        # Special case: one interval completely contains the other
        # This handles cases where both trains wrap or one completely encompasses the other
        
        # Check if train1 completely contains train2
        if arr1_buffered <= dep1_buffered:  # Train1 doesn't wrap
            if arr2 <= dep2:  # Train2 doesn't wrap
                # Already handled above
                pass
            else:  # Train2 wraps
                # If train1 doesn't wrap and train2 does, they conflict if train1 
                # spans midnight
                if arr1_buffered == 0 or dep1_buffered >= WEEK_SECONDS - 1:
                    return True
        else:  # Train1 wraps
            if arr2 > dep2:  # Train2 also wraps
                # Both wrap - they definitely overlap
                return True
            else:  # Train2 doesn't wrap
                # Train1 wraps, train2 doesn't
                # They DON'T conflict only if train2 is entirely in the gap
                gap_start = dep1_buffered
                gap_end = arr1_buffered
                if arr2 >= gap_start and dep2 <= gap_end:
                    return False
                else:
                    return True
        
        return False
    
    def _calculate_waste(self, arrival: float, departure: float,
                        track_trains: List[ScheduledTrain]) -> float:
        """Calculate wasted time if train is placed on track."""
        if not track_trains:
            return 0
        
        # Find gaps before and after
        gaps = []
        for scheduled in track_trains:
            gap_before = (arrival - scheduled.departure_seconds) % WEEK_SECONDS
            gap_after = (scheduled.arrival_seconds - departure) % WEEK_SECONDS
            gaps.extend([gap_before, gap_after])
        
        return min(gaps) if gaps else 0
    
    def _create_scheduled_train(self, train: Train, track_id: int) -> ScheduledTrain:
        """Create ScheduledTrain object from Train and track assignment."""
        sched = train.schedule_encoded
        
        return ScheduledTrain(
            train=train,
            track_id=track_id,
            arrival_seconds=sched['arrival']['seconds'],
            departure_seconds=sched['departure']['seconds'],
            arrival_angle=sched['arrival']['angle'],
            departure_angle=sched['departure']['angle'],
            operator=sched['operator'],
            stay_hours=sched['stay_duration']['hours']
        )
    
    def visualize_schedule(self, schedule: TrainSchedule, figsize: Tuple[int, int] = (14, 10)):
        """
        Create circular Gantt chart visualization.
        
        Args:
            schedule: TrainSchedule object to visualize
            figsize: Figure size tuple
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize, subplot_kw=dict(projection='polar'))
        fig.suptitle('Train Schedule - Circular Weekly View', fontsize=16, fontweight='bold')
        
        # Flatten axes for easier iteration
        axes = axes.flatten()
        
        # Colors for different operators
        operators = list(set(st.operator for st in schedule.scheduled_trains))
        colors = plt.cm.Set3(np.linspace(0, 1, len(operators)))
        color_map = dict(zip(operators, colors))
        
        # Plot each track
        for track_id in range(min(self.num_tracks, 4)):  # Max 4 subplots
            ax = axes[track_id]
            self._plot_track(ax, track_id, schedule, color_map)
        
        # Hide unused subplots
        for i in range(self.num_tracks, 4):
            axes[i].set_visible(False)
        
        # Add legend
        if schedule.scheduled_trains:
            handles = [plt.Line2D([0], [0], color=color, lw=4, label=op) 
                      for op, color in color_map.items()]
            fig.legend(handles, color_map.keys(), loc='center', 
                      bbox_to_anchor=(0.5, -0.05), ncol=min(len(operators), 4))
        
        # Add statistics
        stats_text = (f"Total Trains Scheduled: {len(schedule.scheduled_trains)}\n"
                     f"Trains Unscheduled: {len(schedule.unscheduled_trains)}\n"
                     f"Overall Utilization: {schedule.get_utilization():.1f}%")
        fig.text(0.02, 0.02, stats_text, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
        
        plt.tight_layout()
        return fig
    
    def _plot_track(self, ax, track_id: int, schedule: TrainSchedule, color_map: Dict):
        """Plot single track as circular chart."""
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        
        # Set up the week labels
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        day_angles = np.linspace(0, 2*np.pi, 8)
        ax.set_xticks(day_angles[:-1])
        ax.set_xticklabels(days)
        
        # Plot trains on this track
        track_trains = schedule.track_occupancy.get(track_id, [])
        
        for st in track_trains:
            # Handle wraparound
            if st.departure_angle < st.arrival_angle:
                # Train wraps around Sunday/Monday
                angles1 = np.linspace(st.arrival_angle, 2*np.pi, 50)
                angles2 = np.linspace(0, st.departure_angle, 50)
                
                ax.fill_between(angles1, 0.3, 0.9, color=color_map[st.operator], alpha=0.7)
                ax.fill_between(angles2, 0.3, 0.9, color=color_map[st.operator], alpha=0.7)
                
                # Add train ID at midpoint
                mid_angle = st.arrival_angle + (2*np.pi - st.arrival_angle) / 2
                ax.text(mid_angle, 0.6, st.train.train_id.split('_')[0], 
                       rotation=np.degrees(mid_angle)-90 if mid_angle > np.pi else np.degrees(mid_angle)+90,
                       fontsize=8, ha='center', va='center')
            else:
                # Normal case
                angles = np.linspace(st.arrival_angle, st.departure_angle, 100)
                ax.fill_between(angles, 0.3, 0.9, color=color_map[st.operator], alpha=0.7)
                
                # Add train ID
                mid_angle = (st.arrival_angle + st.departure_angle) / 2
                ax.text(mid_angle, 0.6, st.train.train_id.split('_')[0],
                       rotation=np.degrees(mid_angle)-90 if mid_angle > np.pi else np.degrees(mid_angle)+90,
                       fontsize=8, ha='center', va='center')
        
        # Format
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Track {track_id + 1}\nUtilization: {schedule.get_track_utilization(track_id):.1f}%',
                    pad=20)


# Example usage
if __name__ == "__main__":
    # Parse trains
    parser = DrivingPlanParser()
    trains = parser.create_trains()
    
    # Schedule trains
    scheduler = TrainScheduler(num_tracks=13)
    schedule = scheduler.schedule_trains(trains)
    
    # Print results
    print(f"Scheduled {len(schedule.scheduled_trains)} out of {len(trains)} trains")
    print(f"Overall utilization: {schedule.get_utilization():.1f}%")
    
    for track_id in range(scheduler.num_tracks):
        track_trains = schedule.track_occupancy[track_id]
        print(f"\nTrack {track_id + 1}: {len(track_trains)} trains, "
              f"{schedule.get_track_utilization(track_id):.1f}% utilization")
    
    # Visualize
    fig = scheduler.visualize_schedule(schedule)
    plt.show()