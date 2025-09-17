# tests/test_train_scheduler_refactored.py
import os
import json
import unittest
from datetime import datetime
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

WEEK_SECONDS = 7 * 24 * 3600
BUFFER_TIME_HOURS = 0.2
BUFFER_SECONDS = BUFFER_TIME_HOURS * 3600


# Robust imports for your tree layout
from simulation.terminal_components.systems.train_tools.DPParser import DrivingPlanParser
from simulation.terminal_components.systems.train_tools.TrainScheduler import TrainScheduler, TrainSchedule, ScheduledTrain


class TestTrainSchedulerRefactored(unittest.TestCase):
    def setUp(self):
        # create a minimal driving plan JSON compatible with DPParser (expects operator_short)
        self.plan_path = "simulation/data/test_driving_plan_dp.json"
        os.makedirs(os.path.dirname(self.plan_path), exist_ok=True)

        plan = {
            "driving_plan": {
                "trains": {
                    "TRN001": {
                        "operator_short": "boxXpress",
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
                        "operator_short": "Metrans",
                        "destination": "Prague",
                        "plan": {
                            "1": {
                                "arrival": ["Monday", "12:00", "Monday", "12:30"],
                                "departure": ["Tuesday", "09:00", "Tuesday", "09:30"],
                                "mirrored_on": ["Thursday"]
                            }
                        }
                    },
                    "TRN003": {
                        "operator_short": "Kombiverkehr",
                        "destination": "Munich",
                        "plan": {
                            "1": {
                                "arrival": ["Tuesday", "14:00", "Tuesday", "14:30"],
                                "departure": ["Wednesday", "10:00", "Wednesday", "10:30"],
                                "mirrored_on": ["Thursday", "Saturday"]
                            }
                        }
                    }
                }
            }
        }
        with open(self.plan_path, "w") as f:
            json.dump(plan, f, indent=2)

        self.parser = DrivingPlanParser(json_path=self.plan_path)
        self.trains = self.parser.create_trains()
        self.scheduler = TrainScheduler(num_tracks=6)

    def _interval_conflict(self, arr1: float, dep1: float, arr2: float, dep2: float) -> bool:
        """
        Prüft Überlappung zwischen [arr1-buffer, dep1+buffer] und [arr2, dep2] (mit Wochen-wrap).
        Spiegelung der Logik aus dem Scheduler (vereinfacht).
        """
        def in_interval(x: float, start: float, end: float) -> bool:
            if start <= end:
                return start <= x <= end
            # wrap
            return x >= start or x <= end

        a1 = (arr1 - BUFFER_SECONDS) % WEEK_SECONDS
        d1 = (dep1 + BUFFER_SECONDS) % WEEK_SECONDS
        a2 = arr2 % WEEK_SECONDS
        d2 = dep2 % WEEK_SECONDS

        # any endpoint in the other interval (both directions)
        if in_interval(a1, a2, d2) or in_interval(d1, a2, d2):
            return True
        if in_interval(a2, a1, d1) or in_interval(d2, a1, d1):
            return True

        # Spezialfall: beide wrapen → als Konflikt werten
        if a1 > d1 and a2 > d2:
            return True
        return False

    def _assert_no_track_conflicts(self, schedule: TrainSchedule):
        for track_id, trains in schedule.track_occupancy.items():
            n = len(trains)
            for i in range(n):
                for j in range(i + 1, n):
                    st1 = trains[i]
                    st2 = trains[j]
                    has_conflict = self._interval_conflict(
                        st1.arrival_seconds, st1.departure_seconds,
                        st2.arrival_seconds, st2.departure_seconds
                    )
                    self.assertFalse(
                        has_conflict,
                        msg=f"Conflict on track {track_id}: {st1.train.train_id} vs {st2.train.train_id}"
                    )

    def test_schedule_trains_no_conflicts_and_utilization(self):
        schedule = self.scheduler.schedule_trains(self.trains)

        # basic sanity
        self.assertGreater(len(self.trains), 0, "no trains parsed from DP")
        self.assertGreater(len(schedule.scheduled_trains), 0, "no trains scheduled")
        # check conflicts
        self._assert_no_track_conflicts(schedule)

        # print a few summary lines (not assertions, just informative)
        total_scheduled = len(schedule.scheduled_trains)
        total_unscheduled = len(schedule.unscheduled_trains)
        util = schedule.get_utilization()
        print(f"Scheduled: {total_scheduled}, Unscheduled: {total_unscheduled}, Overall Utilization: {util:.1f}%")

    def test_visualize_and_save(self):
        schedule = self.scheduler.schedule_trains(self.trains)
        fig = self.scheduler.visualize_schedule(schedule, figsize=(10, 8))
        out = "test_weekly_schedule_refactored.png"
        fig.savefig(out, dpi=120, bbox_inches='tight')
        plt.close(fig)
        self.assertTrue(os.path.exists(out), "Gantt chart image was not saved")

    def test_per_track_utilization_nonnegative(self):
        schedule = self.scheduler.schedule_trains(self.trains)
        for track_id in range(self.scheduler.num_tracks):
            util = schedule.get_track_utilization(track_id)
            self.assertGreaterEqual(util, 0.0, f"Negative utilization on track {track_id}")


if __name__ == "__main__":
    unittest.main()