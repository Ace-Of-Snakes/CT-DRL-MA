# simulation/planning/train_scheduler.py
"""Best-fit train-to-track scheduler with optional visualization."""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

from simulation.core.vehicles.train import Train
from simulation.planning.time_encoder import WeeklyTimeEncoder
from simulation.core.constants import SECONDS_PER_WEEK, SECONDS_PER_HOUR


DEFAULT_BUFFER_TIME_HOURS: int = 4


@dataclass
class ScheduledTrain:
    """A train assigned to a track with timing metadata."""
    train: Train
    track_id: int
    arrival_seconds: float
    departure_seconds: float
    arrival_angle: float
    departure_angle: float
    operator: str
    stay_hours: float

    @property
    def duration_seconds(self) -> float:
        """Duration handling weekly wraparound."""
        if self.departure_seconds < self.arrival_seconds:
            return (SECONDS_PER_WEEK - self.arrival_seconds) + self.departure_seconds
        return self.departure_seconds - self.arrival_seconds


@dataclass
class TrainSchedule:
    """Complete schedule for all tracks."""
    num_tracks: int
    scheduled_trains: List[ScheduledTrain] = field(default_factory=list)
    unscheduled_trains: List[Train] = field(default_factory=list)
    track_occupancy: Dict[int, List[ScheduledTrain]] = field(default_factory=dict)

    def __post_init__(self):
        for i in range(self.num_tracks):
            self.track_occupancy.setdefault(i, [])

    def add_train(self, st: ScheduledTrain) -> None:
        """Add a scheduled train, keeping per-track list sorted."""
        self.scheduled_trains.append(st)
        track = self.track_occupancy.setdefault(st.track_id, [])
        track.append(st)
        track.sort(key=lambda s: s.arrival_seconds)

    def get_utilization(self) -> float:
        """Overall track utilization (%)."""
        capacity = self.num_tracks * SECONDS_PER_WEEK
        used = sum(st.duration_seconds for st in self.scheduled_trains)
        return (used / capacity) * 100 if capacity > 0 else 0.0

    def get_track_utilization(self, track_id: int) -> float:
        """Utilization for a single track (%)."""
        trains = self.track_occupancy.get(track_id, [])
        used = sum(st.duration_seconds for st in trains)
        return (used / SECONDS_PER_WEEK) * 100


class TrainScheduler:
    """Best-fit scheduler minimizing wasted track capacity."""

    def __init__(
        self,
        num_tracks: int = 4,
        buffer_hours: int = DEFAULT_BUFFER_TIME_HOURS,
    ):
        self.num_tracks = num_tracks
        self.encoder = WeeklyTimeEncoder()
        self.buffer_seconds = buffer_hours * SECONDS_PER_HOUR

    # ================================================================
    # Scheduling
    # ================================================================

    def schedule_trains(self, trains: List[Train]) -> TrainSchedule:
        """Schedule trains using best-fit (longest-first)."""
        schedule = TrainSchedule(num_tracks=self.num_tracks)

        sorted_trains = sorted(
            trains,
            key=lambda t: t.schedule_encoded['stay_duration']['hours'],
            reverse=True,
        )

        for train in sorted_trains:
            best_track = self._find_best_track(train, schedule)
            if best_track is not None:
                schedule.add_train(self._create_scheduled_train(train, best_track))
            else:
                schedule.unscheduled_trains.append(train)

        return schedule

    def _find_best_track(self, train: Train, schedule: TrainSchedule) -> Optional[int]:
        """Find track with least waste for this train."""
        arr = train.schedule_encoded['arrival']['seconds']
        dep = train.schedule_encoded['departure']['seconds']

        best_track = None
        min_waste = float('inf')

        for track_id in range(self.num_tracks):
            occupants = schedule.track_occupancy[track_id]
            if self._can_fit(arr, dep, occupants):
                waste = self._calculate_waste(arr, dep, occupants)
                if waste < min_waste:
                    min_waste = waste
                    best_track = track_id

        return best_track

    def _can_fit(
        self, arrival: float, departure: float,
        track_trains: List[ScheduledTrain],
    ) -> bool:
        """Check if train fits on track without conflicts."""
        for st in track_trains:
            if self._has_conflict(arrival, departure, st.arrival_seconds, st.departure_seconds):
                return False
        return True

    def _has_conflict(
        self,
        arr1: float, dep1: float,
        arr2: float, dep2: float,
    ) -> bool:
        """Check time conflict with buffer, handling weekly wraparound."""
        arr1_buf = (arr1 - self.buffer_seconds) % SECONDS_PER_WEEK
        dep1_buf = (dep1 + self.buffer_seconds) % SECONDS_PER_WEEK

        def _in_interval(point: float, start: float, end: float) -> bool:
            if start <= end:
                return start <= point <= end
            return point >= start or point <= end

        if _in_interval(arr1_buf, arr2, dep2):
            return True
        if _in_interval(dep1_buf, arr2, dep2):
            return True
        if _in_interval(arr2, arr1_buf, dep1_buf):
            return True
        if _in_interval(dep2, arr1_buf, dep1_buf):
            return True

        # Both wrap → guaranteed overlap
        if arr1_buf > dep1_buf:
            if arr2 > dep2:
                return True
            gap_start, gap_end = dep1_buf, arr1_buf
            return not (arr2 >= gap_start and dep2 <= gap_end)

        return False

    def _calculate_waste(
        self, arrival: float, departure: float,
        track_trains: List[ScheduledTrain],
    ) -> float:
        """Minimum gap to any existing train (lower = tighter fit)."""
        if not track_trains:
            return 0.0
        gaps = []
        for st in track_trains:
            gaps.append((arrival - st.departure_seconds) % SECONDS_PER_WEEK)
            gaps.append((st.arrival_seconds - departure) % SECONDS_PER_WEEK)
        return min(gaps)

    def _create_scheduled_train(self, train: Train, track_id: int) -> ScheduledTrain:
        """Build ScheduledTrain from Train metadata."""
        s = train.schedule_encoded
        return ScheduledTrain(
            train=train,
            track_id=track_id,
            arrival_seconds=s['arrival']['seconds'],
            departure_seconds=s['departure']['seconds'],
            arrival_angle=s['arrival']['angle'],
            departure_angle=s['departure']['angle'],
            operator=s['operator'],
            stay_hours=s['stay_duration']['hours'],
        )

    # ================================================================
    # Visualization (lazy-loaded matplotlib)
    # ================================================================

    def visualize_schedule(
        self,
        schedule: TrainSchedule,
        figsize: Tuple[int, int] = (14, 10),
    ):
        """Create circular Gantt chart visualization. Returns matplotlib Figure."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(
            2, 2, figsize=figsize,
            subplot_kw=dict(projection='polar'),
        )
        fig.suptitle('Train Schedule - Circular Weekly View', fontsize=16, fontweight='bold')
        axes = axes.flatten()

        operators = list({st.operator for st in schedule.scheduled_trains})
        colors = plt.cm.Set3(np.linspace(0, 1, max(1, len(operators))))
        color_map = dict(zip(operators, colors))

        for track_id in range(min(self.num_tracks, 4)):
            self._plot_track(axes[track_id], track_id, schedule, color_map)

        for i in range(self.num_tracks, 4):
            axes[i].set_visible(False)

        if schedule.scheduled_trains:
            handles = [
                plt.Line2D([0], [0], color=c, lw=4, label=op)
                for op, c in color_map.items()
            ]
            fig.legend(
                handles, color_map.keys(),
                loc='center', bbox_to_anchor=(0.5, -0.05),
                ncol=min(len(operators), 4),
            )

        stats_text = (
            f"Scheduled: {len(schedule.scheduled_trains)}\n"
            f"Unscheduled: {len(schedule.unscheduled_trains)}\n"
            f"Utilization: {schedule.get_utilization():.1f}%"
        )
        fig.text(
            0.02, 0.02, stats_text, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5),
        )

        plt.tight_layout()
        return fig

    @staticmethod
    def _plot_track(ax, track_id: int, schedule: TrainSchedule, color_map: Dict) -> None:
        """Plot single track as circular chart."""
        import matplotlib.pyplot as plt

        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)

        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        day_angles = np.linspace(0, 2 * np.pi, 8)
        ax.set_xticks(day_angles[:-1])
        ax.set_xticklabels(days)

        for st in schedule.track_occupancy.get(track_id, []):
            color = color_map.get(st.operator, 'gray')
            label = st.train.train_id.split('_')[0]

            if st.departure_angle < st.arrival_angle:
                a1 = np.linspace(st.arrival_angle, 2 * np.pi, 50)
                a2 = np.linspace(0, st.departure_angle, 50)
                ax.fill_between(a1, 0.3, 0.9, color=color, alpha=0.7)
                ax.fill_between(a2, 0.3, 0.9, color=color, alpha=0.7)
                mid = st.arrival_angle + (2 * np.pi - st.arrival_angle) / 2
            else:
                angles = np.linspace(st.arrival_angle, st.departure_angle, 100)
                ax.fill_between(angles, 0.3, 0.9, color=color, alpha=0.7)
                mid = (st.arrival_angle + st.departure_angle) / 2

            rot = np.degrees(mid) - 90 if mid > np.pi else np.degrees(mid) + 90
            ax.text(mid, 0.6, label, rotation=rot, fontsize=8, ha='center', va='center')

        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.grid(True, alpha=0.3)
        ax.set_title(
            f'Track {track_id + 1}\n{schedule.get_track_utilization(track_id):.1f}%',
            pad=20,
        )