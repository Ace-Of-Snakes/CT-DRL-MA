# test_container_estimator.py

import unittest
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import statistics

# Import the classes (adjust path as needed)
from simulation.terminal_components.storage_units.Container import (
    Container, CONTAINER_LENGTHS, CONTAINER_HEIGHT_STANDARD, 
    CONTAINER_HEIGHT_HIGH_CUBE, CONTAINER_WIDTH_STANDARD, CONTAINER_WIDTH_SPECIAL
)
from simulation.estimators.EstimatorDeparture import (
    StandardDepartureEstimator
)
from simulation.estimators.ManageEstimators import (
    DepartureEstimatorManager
)

class TestContainer(unittest.TestCase):
    """Test suite for the Container dataclass."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.base_arrival = datetime(2024, 1, 1)
        self.base_departure = datetime(2024, 1, 15)
    
    def test_container_creation(self):
        """Test basic container creation."""
        container = Container(
            container_id="TEST001",
            direction="Import",
            container_type="FEU",
            arrival_date=self.base_arrival,
            departure_date=self.base_departure,
            goods_type="Regular"
        )
        
        self.assertEqual(container.container_id, "TEST001")
        self.assertEqual(container.direction, "Import")
        self.assertEqual(container.container_type, "FEU")
        self.assertEqual(container.goods_type, "Regular")
        self.assertEqual(container.arrival_date, self.base_arrival)
        self.assertEqual(container.departure_date, self.base_departure)
    
    def test_default_dimensions(self):
        """Test that default dimensions are set correctly."""
        # Standard container
        container = Container(
            container_id="TEST002",
            direction="Export",
            container_type="FEU",
            arrival_date=self.base_arrival,
            departure_date=self.base_departure,
            is_high_cube=False
        )
        
        self.assertEqual(container.height, CONTAINER_HEIGHT_STANDARD)
        self.assertEqual(container.length, CONTAINER_LENGTHS["FEU"])
        self.assertEqual(container.width, CONTAINER_WIDTH_STANDARD)
        
        # High cube container
        hc_container = Container(
            container_id="TEST003",
            direction="Export",
            container_type="FEU",
            arrival_date=self.base_arrival,
            departure_date=self.base_departure,
            is_high_cube=True
        )
        
        self.assertEqual(hc_container.height, CONTAINER_HEIGHT_HIGH_CUBE)
    
    def test_special_container_types(self):
        """Test special container types (Trailer, Swap Body)."""
        trailer = Container(
            container_id="TRL001",
            direction="Import",
            container_type="Trailer",
            arrival_date=self.base_arrival,
            departure_date=self.base_departure
        )
        
        self.assertTrue(trailer.is_special_type)
        self.assertEqual(trailer.width, CONTAINER_WIDTH_SPECIAL)
        
        swap_body = Container(
            container_id="SWP001",
            direction="Export",
            container_type="Swap Body",
            arrival_date=self.base_arrival,
            departure_date=self.base_departure
        )
        
        self.assertTrue(swap_body.is_special_type)
        self.assertEqual(swap_body.width, CONTAINER_WIDTH_SPECIAL)
    
    def test_days_calculations(self):
        """Test days_in_terminal and days_until_departure calculations."""
        container = Container(
            container_id="TEST004",
            direction="Import",
            container_type="FEU",
            arrival_date=self.base_arrival,
            departure_date=self.base_departure
        )
        
        # Test days in terminal
        current_date = self.base_arrival + timedelta(days=5)
        self.assertEqual(container.days_in_terminal(current_date), 5)
        
        # Test with arrival date
        self.assertEqual(container.days_in_terminal(self.base_arrival), 0)
        
        # Test days until departure (after estimation is set)
        container.estimated_departure = self.base_departure
        days_until = container.days_until_departure(current_date)
        self.assertEqual(days_until, 9)  # 15 - 6 = 9 days


class TestStandardDepartureEstimator(unittest.TestCase):
    """Test suite for the StandardDepartureEstimator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.estimator = StandardDepartureEstimator()
        self.manager = DepartureEstimatorManager(self.estimator)
        self.base_arrival = datetime(2024, 1, 1)
    
    def create_container(self, stay_days: int) -> Container:
        """Helper to create a container with specified stay duration."""
        return Container(
            container_id=f"TEST_{stay_days}",
            direction="Import",
            container_type="FEU",
            arrival_date=self.base_arrival,
            departure_date=self.base_arrival + timedelta(days=stay_days),
            goods_type="Regular"
        )
    
    def test_perfect_accuracy_short_stay(self):
        """Test that very short stays have perfect accuracy."""
        container = self.create_container(2)  # MIN_ACCURACY_DAYS
        
        # Run multiple estimations
        errors = []
        for _ in range(100):
            estimated = self.estimator.estimate_departure(container, self.base_arrival)
            error = (estimated - container.departure_date).days
            errors.append(error)
        
        # All estimates should be perfect (0 error)
        self.assertTrue(all(e == 0 for e in errors), 
                       f"Short stay should have perfect accuracy, got errors: {set(errors)}")
    
    def test_accuracy_improves_near_departure(self):
        """Test that accuracy improves as departure approaches."""
        container = self.create_container(30)  # 30-day stay
        
        # Test at different points in time
        errors_early = []
        errors_late = []
        
        # Early in stay (day 5)
        current_early = self.base_arrival + timedelta(days=5)
        for _ in range(100):
            estimated = self.estimator.estimate_departure(container, current_early)
            error = abs((estimated - container.departure_date).days)
            errors_early.append(error)
        
        # Late in stay (3 days before departure)
        current_late = container.departure_date - timedelta(days=3)
        for _ in range(100):
            estimated = self.estimator.estimate_departure(container, current_late)
            error = abs((estimated - container.departure_date).days)
            errors_late.append(error)
        
        # Average error should be lower late in stay
        avg_error_early = statistics.mean(errors_early)
        avg_error_late = statistics.mean(errors_late)
        
        self.assertLess(avg_error_late, avg_error_early,
                       f"Late error ({avg_error_late:.2f}) should be less than early ({avg_error_early:.2f})")
    
    def test_estimation_not_in_past(self):
        """Test that estimations don't go into the past for containers still in terminal."""
        container = self.create_container(10)
        
        # Simulate being 8 days into the stay
        current_date = self.base_arrival + timedelta(days=8)
        
        # Run multiple estimations
        for _ in range(50):
            estimated = self.estimator.estimate_departure(container, current_date)
            self.assertGreaterEqual(estimated, current_date,
                                  "Estimation should not be in the past")
    
    def test_accuracy_calculation(self):
        """Test the accuracy calculation method directly."""
        # Test perfect accuracy cases
        self.assertEqual(self.estimator._calculate_accuracy(2, 10), 1.0)  # Short stay
        self.assertEqual(self.estimator._calculate_accuracy(30, 2), 1.0)  # Imminent departure
        
        # Test boost period
        accuracy_boost = self.estimator._calculate_accuracy(30, 5)
        self.assertGreater(accuracy_boost, 0.8)
        
        # Test peak uncertainty
        accuracy_peak = self.estimator._calculate_accuracy(120, 60)
        self.assertAlmostEqual(accuracy_peak, 0.3, delta=0.05)
        
        # Test very long stays
        accuracy_long = self.estimator._calculate_accuracy(160, 100)
        self.assertAlmostEqual(accuracy_long, 0.45, delta=0.05)
    
    def test_error_bounds(self):
        """Test that errors stay within reasonable bounds."""
        container = self.create_container(60)  # 60-day stay
        
        errors = []
        for _ in range(100):
            estimated = self.estimator.estimate_departure(container, self.base_arrival)
            error = (estimated - container.departure_date).days
            errors.append(error)
        
        max_expected_error = max(7, 60 * 0.3)  # As defined in the estimator
        
        self.assertTrue(all(abs(e) <= max_expected_error * 1.1 for e in errors),
                       f"Errors should be bounded, got max: {max(abs(e) for e in errors)}")


class TestEstimatorVisualization(unittest.TestCase):
    """Test suite for visualizing the estimator behavior."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.estimator = StandardDepartureEstimator()
        self.base_arrival = datetime(2024, 1, 1)
    
    def test_plot_accuracy_curve(self):
        """Plot the accuracy curve for different stay durations."""
        stay_durations = list(range(1, 180))
        accuracies_at_arrival = []
        accuracies_midway = []
        accuracies_near_departure = []
        
        for days in stay_durations:
            # At arrival
            acc_arrival = self.estimator._calculate_accuracy(days, days)
            accuracies_at_arrival.append(acc_arrival)
            
            # Midway through stay
            remaining_midway = days // 2
            acc_midway = self.estimator._calculate_accuracy(days, remaining_midway)
            accuracies_midway.append(acc_midway)
            
            # Near departure (5 days before)
            remaining_near = min(5, days - 1)
            acc_near = self.estimator._calculate_accuracy(days, remaining_near)
            accuracies_near_departure.append(acc_near)
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Accuracy curves
        ax1.plot(stay_durations, accuracies_at_arrival, 'b-', label='At Arrival', linewidth=2)
        ax1.plot(stay_durations, accuracies_midway, 'g--', label='Midway Through Stay', linewidth=2)
        ax1.plot(stay_durations, accuracies_near_departure, 'r-.', label='Near Departure (5 days)', linewidth=2)
        
        ax1.set_xlabel('Total Stay Duration (days)')
        ax1.set_ylabel('Prediction Accuracy')
        ax1.set_title('Departure Estimation Accuracy vs Container Stay Duration')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_ylim([0, 1.05])
        
        # Add reference lines
        ax1.axhline(y=0.3, color='gray', linestyle=':', alpha=0.5, label='Min Accuracy')
        ax1.axhline(y=0.45, color='gray', linestyle=':', alpha=0.5, label='Late Accuracy')
        ax1.axhline(y=0.85, color='gray', linestyle=':', alpha=0.5, label='Boost Accuracy')
        ax1.axvline(x=120, color='orange', linestyle=':', alpha=0.5, label='Peak Uncertainty Day')
        
        # Plot 2: Error distribution visualization
        test_durations = [7, 30, 60, 120, 160]
        errors_by_duration = []
        
        for duration in test_durations:
            container = Container(
                container_id=f"TEST_{duration}",
                direction="Import",
                container_type="FEU",
                arrival_date=self.base_arrival,
                departure_date=self.base_arrival + timedelta(days=duration),
                goods_type="Regular"
            )
            
            errors = []
            for _ in range(500):
                estimated = self.estimator.estimate_departure(container, self.base_arrival)
                error = (estimated - container.departure_date).days
                errors.append(error)
            errors_by_duration.append(errors)
        
        # Create box plots
        positions = range(1, len(test_durations) + 1)
        bp = ax2.boxplot(errors_by_duration, positions=positions, widths=0.6)
        ax2.set_xticklabels([f'{d} days' for d in test_durations])
        ax2.set_xlabel('Container Stay Duration')
        ax2.set_ylabel('Estimation Error (days)')
        ax2.set_title('Distribution of Estimation Errors by Stay Duration')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='red', linestyle='-', alpha=0.3)
        
        # Add mean markers
        for i, errors in enumerate(errors_by_duration):
            mean_error = statistics.mean(errors)
            ax2.plot(i + 1, mean_error, 'ro', markersize=8)
        
        plt.tight_layout()
        plt.savefig('estimator_accuracy_analysis.png', dpi=150)
        plt.show()
        
        print("\n=== Accuracy Statistics ===")
        for duration in test_durations:
            idx = stay_durations.index(duration) if duration in stay_durations else -1
            if idx >= 0:
                print(f"\n{duration}-day stay:")
                print(f"  Accuracy at arrival: {accuracies_at_arrival[idx]:.2%}")
                print(f"  Accuracy midway: {accuracies_midway[idx]:.2%}")
                print(f"  Accuracy near departure: {accuracies_near_departure[idx]:.2%}")
    
    def test_estimation_convergence(self):
        """Test how estimations converge as time progresses."""
        container = Container(
            container_id="CONV_TEST",
            direction="Import", 
            container_type="FEU",
            arrival_date=self.base_arrival,
            departure_date=self.base_arrival + timedelta(days=60),
            goods_type="Regular"
        )
        
        # Track estimation errors over time
        days_elapsed = []
        mean_errors = []
        std_errors = []
        
        for day in range(0, 61, 2):
            current_date = self.base_arrival + timedelta(days=day)
            errors = []
            
            for _ in range(100):
                estimated = self.estimator.estimate_departure(container, current_date)
                error = abs((estimated - container.departure_date).days)
                errors.append(error)
            
            days_elapsed.append(day)
            mean_errors.append(statistics.mean(errors))
            std_errors.append(statistics.stdev(errors) if len(errors) > 1 else 0)
        
        # Plot convergence
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.errorbar(days_elapsed, mean_errors, yerr=std_errors, 
                   fmt='o-', capsize=5, capthick=2, label='Mean ± Std Dev')
        ax.fill_between(days_elapsed, 
                        [m - s for m, s in zip(mean_errors, std_errors)],
                        [m + s for m, s in zip(mean_errors, std_errors)],
                        alpha=0.3)
        
        ax.set_xlabel('Days Since Arrival')
        ax.set_ylabel('Absolute Estimation Error (days)')
        ax.set_title('Estimation Error Convergence Over Time (60-day stay)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('estimation_convergence.png', dpi=150)
        plt.show()


def run_all_tests():
    """Run all test suites and generate visualizations."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestContainer))
    suite.addTests(loader.loadTestsFromTestCase(TestStandardDepartureEstimator))
    suite.addTests(loader.loadTestsFromTestCase(TestEstimatorVisualization))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_all_tests()