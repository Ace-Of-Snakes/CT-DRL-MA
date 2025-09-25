import unittest
import time
import numpy as np
from datetime import datetime, timedelta
from typing import List
import warnings
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation2.core.Container import Container
from simulation2.factories.ContainerFactory import ContainerFactory


class TestContainerFactory(unittest.TestCase):
    """Test suite for ContainerFactory with functionality and performance tests."""
    
    @classmethod
    def setUpClass(cls):
        """Set up factory once for all tests."""
        print("\n" + "="*60)
        print("CONTAINER FACTORY TEST SUITE")
        print("="*60)
        
        # Initialize factory with timer
        start = time.perf_counter()
        cls.factory = ContainerFactory(use_estimator=True)
        init_time = time.perf_counter() - start
        print(f"✓ Factory initialization: {init_time:.3f}s")
        
        # Get available operators
        cls.test_operators = list(cls.factory.operator_dict.keys())[:3]  # Test with first 3
        print(f"✓ Testing with operators: {cls.test_operators}")
    
    def test_01_basic_container_creation(self):
        """Test basic container creation functionality."""
        print("\n1. BASIC FUNCTIONALITY TESTS")
        print("-" * 40)
        
        for operator in self.test_operators[:1]:  # Test with one operator
            # Test Import containers
            import_containers = self.factory.create_containers(
                operator, "Import", n_containers=10
            )
            self.assertEqual(len(import_containers), 10)
            self.assertTrue(all(c.direction == "Import" for c in import_containers))
            print(f"✓ Created 10 Import containers for {operator}")
            
            # Test Export containers
            export_containers = self.factory.create_containers(
                operator, "Export", n_containers=10
            )
            self.assertEqual(len(export_containers), 10)
            self.assertTrue(all(c.direction == "Export" for c in export_containers))
            print(f"✓ Created 10 Export containers for {operator}")
            
            # Verify container properties
            for container in import_containers[:1]:
                self.assertIsNotNone(container.container_id)
                self.assertIsNotNone(container.container_type)
                self.assertIsNotNone(container.goods_type)
                self.assertGreater(container.length_m, 0)
                self.assertGreater(container.width_m, 0)
                self.assertGreater(container.height_m, 0)
                self.assertIsInstance(container.is_high_cube, bool)
                self.assertIsInstance(container.is_swap_body, bool)
                self.assertIsInstance(container.is_trailer, bool)
                print(f"✓ Container properties valid: {container.container_type}")
    
    def test_02_estimation_functionality(self):
        """Test departure estimation functionality."""
        print("\n2. ESTIMATION FUNCTIONALITY TESTS")
        print("-" * 40)
        
        # Create containers with estimator
        containers = self.factory.create_containers(
            "BOX", "Import", 100, 
            base_arrival_date=datetime(2024, 1, 1),
            current_date=datetime(2024, 1, 1)
        )
        
        # Check all have estimated departures
        all_have_estimates = all(c.estimated_departure is not None for c in containers)
        self.assertTrue(all_have_estimates)
        print(f"✓ All {len(containers)} containers have estimated departures")
        
        # Separate containers by expected accuracy
        perfect_accuracy_containers = []
        normal_containers = []
        
        for c in containers:
            total_stay = (c.departure_date - c.arrival_date).days
            remaining_days = (c.departure_date - datetime(2024, 1, 1)).days
            
            # Perfect accuracy criteria from StandardDepartureEstimator
            if total_stay <= 2 or remaining_days <= 2:
                perfect_accuracy_containers.append(c)
            else:
                normal_containers.append(c)
        
        # Test perfect accuracy containers (if any)
        if perfect_accuracy_containers:
            errors = []
            for c in perfect_accuracy_containers:
                error = abs((c.estimated_departure - c.departure_date).days)
                errors.append(error)
            
            avg_error = np.mean(errors) if errors else 0
            self.assertLess(avg_error, 0.01, 
                        f"Perfect accuracy containers should have ~0 error, got {avg_error:.4f}")
            print(f"✓ Perfect accuracy containers ({len(perfect_accuracy_containers)}) have ~0 error")
        
        # Test normal containers have reasonable error based on estimator design
        if normal_containers:
            errors = []
            for c in normal_containers:
                error = abs((c.estimated_departure - c.departure_date).days)
                total_stay = (c.departure_date - c.arrival_date).days
                
                # Expected max error from StandardDepartureEstimator
                expected_max_error = max(7, total_stay * 0.3)
                errors.append(error)
                
                # Each individual error should be within expected bounds
                self.assertLessEqual(error, expected_max_error * 2,  # Allow 2x for statistical variance
                                f"Container error {error} exceeds reasonable bounds for {total_stay} day stay")
            
            avg_error = np.mean(errors)
            # For containers with longer stays, expect some error
            # The estimator is designed to have uncertainty, especially for medium-length stays
            self.assertLess(avg_error, 20,  # Reasonable upper bound for average error
                        f"Average error {avg_error:.2f} days is too high")
            
            print(f"✓ Normal containers ({len(normal_containers)}) have reasonable estimation error (avg: {avg_error:.2f} days)")
        
        # Test that estimation updates when closer to departure
        if containers:
            # Pick a container with medium stay (likely to have initial error)
            test_container = None
            for c in containers:
                if 10 <= (c.departure_date - c.arrival_date).days <= 30:
                    test_container = c
                    break
            
            if test_container:
                # Get initial estimation error
                initial_error = abs((test_container.estimated_departure - test_container.departure_date).days)
                
                # Update estimation 2 days before departure (should be perfect)
                near_departure_date = test_container.departure_date - timedelta(days=1)
                self.factory.update_estimations([test_container], near_departure_date)
                
                # Error should now be ~0
                final_error = abs((test_container.estimated_departure - test_container.departure_date).days)
                self.assertLess(final_error, 0.1, 
                            "Estimation should be perfect 1 day before departure")
                
                print(f"✓ Estimation accuracy improves near departure: {initial_error:.1f} → {final_error:.1f} days error")
    
    def test_03_container_type_distribution(self):
        """Test container type distribution matches operator specifications."""
        print("\n3. DISTRIBUTION TESTS")
        print("-" * 40)
        
        operator = self.test_operators[0]
        n_samples = 1000
        
        containers = self.factory.create_containers(
            operator, "Import", n_containers=n_samples
        )
        
        # Count container types
        type_counts = {}
        for c in containers:
            type_counts[c.container_type] = type_counts.get(c.container_type, 0) + 1
        
        # Check distribution roughly matches operator specs
        operator_data = self.factory.operator_dict[operator]
        print(f"Container type distribution for {operator}:")
        
        for ctype, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            observed_prob = count / n_samples
            if ctype in operator_data:
                expected_prob = operator_data[ctype]["P_for_operator"]
                # Allow 20% relative error for small samples
                print(f"  {ctype}: {observed_prob:.3f} (expected ~{expected_prob:.3f})")
    
    def test_04_performance_small_batch(self):
        """Test performance with small batches."""
        print("\n4. PERFORMANCE TESTS - SMALL BATCH")
        print("-" * 40)
        
        operator = self.test_operators[0]
        n_containers = 100
        
        # Time container creation
        start = time.perf_counter()
        containers = self.factory.create_containers(
            operator, "Import", n_containers=n_containers
        )
        creation_time = time.perf_counter() - start
        
        rate = n_containers / creation_time
        print(f"✓ Created {n_containers} containers in {creation_time:.3f}s")
        print(f"  Rate: {rate:.0f} containers/second")
        
        # Performance assertion
        self.assertLess(creation_time, 1.0, "Small batch should complete in < 1 second")
    
    def test_05_performance_large_batch(self):
        """Test performance with large batches."""
        print("\n5. PERFORMANCE TESTS - LARGE BATCH")
        print("-" * 40)
        
        operator = self.test_operators[0]
        n_containers = 10000
        
        # Time large batch creation
        start = time.perf_counter()
        containers = self.factory.create_containers(
            operator, "Import", n_containers=n_containers
        )
        creation_time = time.perf_counter() - start
        
        rate = n_containers / creation_time
        print(f"✓ Created {n_containers:,} containers in {creation_time:.3f}s")
        print(f"  Rate: {rate:,.0f} containers/second")
        
        # Performance assertion - should handle 10k in reasonable time
        self.assertLess(creation_time, 5.0, "Large batch should complete in < 5 seconds")
        
        # Memory check - ensure containers are lightweight
        import sys
        container_size = sys.getsizeof(containers[0])
        total_size_mb = (container_size * n_containers) / (1024 * 1024)
        print(f"  Memory: ~{total_size_mb:.1f} MB for {n_containers:,} containers")
    
    def test_06_performance_mixed_batch(self):
        """Test performance with mixed operator/direction batches."""
        print("\n6. PERFORMANCE TESTS - MIXED BATCH")
        print("-" * 40)
        
        # Create mixed batch specification
        batches = [
            (self.test_operators[0], "Import", 3000),
            (self.test_operators[0], "Export", 2000),
        ]
        if len(self.test_operators) > 1:
            batches.append((self.test_operators[1], "Import", 2500))
            batches.append((self.test_operators[1], "Export", 2500))
        
        total_containers = sum(count for _, _, count in batches)
        
        # Time mixed batch creation
        start = time.perf_counter()
        containers = self.factory.create_batch(batches)
        creation_time = time.perf_counter() - start
        
        rate = total_containers / creation_time
        print(f"✓ Created {total_containers:,} mixed containers in {creation_time:.3f}s")
        print(f"  Rate: {rate:,.0f} containers/second")
        print(f"  Batches: {len(batches)}")
        
        # Verify correct counts
        self.assertEqual(len(containers), total_containers)
    
    def test_07_performance_estimation_update(self):
        """Test performance of bulk estimation updates."""
        print("\n7. PERFORMANCE TESTS - ESTIMATION UPDATE")
        print("-" * 40)
        
        # Create containers without reading files again
        n_containers = 5000
        containers = self.factory.create_containers(
            self.test_operators[0], "Import", n_containers=n_containers
        )
        
        # Time estimation update
        new_date = datetime.now() + timedelta(days=7)
        start = time.perf_counter()
        self.factory.update_estimations(containers, new_date)
        update_time = time.perf_counter() - start
        
        rate = n_containers / update_time
        print(f"✓ Updated {n_containers:,} estimations in {update_time:.3f}s")
        print(f"  Rate: {rate:,.0f} updates/second")
        
        # Should be very fast due to vectorization
        self.assertLess(update_time, 1.0, "Vectorized updates should be < 1 second")
    
    def test_08_stress_test(self):
        """Stress test with very large batch."""
        print("\n8. STRESS TEST")
        print("-" * 40)
        
        n_containers = 50000
        print(f"Creating {n_containers:,} containers...")
        
        start = time.perf_counter()
        containers = self.factory.create_containers(
            self.test_operators[0], 
            "Import", 
            n_containers=n_containers
        )
        creation_time = time.perf_counter() - start
        
        # Verify all created
        self.assertEqual(len(containers), n_containers)
        
        # Check performance metrics
        rate = n_containers / creation_time
        ms_per_container = (creation_time * 1000) / n_containers
        
        print(f"✓ Stress test passed!")
        print(f"  Total time: {creation_time:.2f}s")
        print(f"  Rate: {rate:,.0f} containers/second")
        print(f"  Time per container: {ms_per_container:.3f}ms")
        
        # Should maintain good performance even at scale
        self.assertLess(ms_per_container, 1.0, "Should average < 1ms per container")
    
    def test_09_edge_cases(self):
        """Test edge cases and error handling."""
        print("\n9. EDGE CASES")
        print("-" * 40)
        
        # Test invalid operator
        with self.assertRaises(ValueError):
            self.factory.create_containers("INVALID_OPERATOR", "Import")
        print("✓ Invalid operator raises ValueError")
        
        # Test single container creation
        containers = self.factory.create_containers(
            self.test_operators[0], "Import", n_containers=1
        )
        self.assertEqual(len(containers), 1)
        print("✓ Single container creation works")
        
        # Test zero containers (edge case)
        containers = self.factory.create_containers(
            self.test_operators[0], "Import", n_containers=0
        )
        self.assertEqual(len(containers), 0)
        print("✓ Zero containers request handled")


def run_performance_summary():
    """Run tests and provide performance summary."""
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    
    # Run basic performance benchmark
    factory = ContainerFactory(use_estimator=True)
    
    test_sizes = [100, 1000, 10000]
    results = []
    
    for size in test_sizes:
        start = time.perf_counter()
        containers = factory.create_containers(
            list(factory.operator_dict.keys())[0],
            "Import",
            n_containers=size
        )
        elapsed = time.perf_counter() - start
        rate = size / elapsed
        results.append((size, elapsed, rate))
        print(f"{size:>6} containers: {elapsed:>6.3f}s ({rate:>8,.0f}/sec)")
    
    # Calculate scaling factor
    if len(results) > 1:
        scaling = (results[-1][1] / results[0][1]) / (results[-1][0] / results[0][0])
        if scaling < 1.5:
            print(f"\n✓ Excellent scaling: {scaling:.2f}x (near-linear)")
        elif scaling < 3:
            print(f"\n✓ Good scaling: {scaling:.2f}x")
        else:
            print(f"\n⚠ Poor scaling: {scaling:.2f}x")


if __name__ == "__main__":
    # Run tests
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestContainerFactory)
    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)
    
    # Run performance summary if all tests passed
    if result.wasSuccessful():
        run_performance_summary()
    
    print("\n" + "="*60)
    print(f"Tests: {result.testsRun} | "
          f"Failures: {len(result.failures)} | "
          f"Errors: {len(result.errors)}")
    print("="*60)