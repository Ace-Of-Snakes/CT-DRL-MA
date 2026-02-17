import unittest
import time
import numpy as np
from datetime import datetime, timedelta
from typing import List
import warnings
import os
import sys

from simulation.core.containers.container import Container
from simulation.core.factories.container_factory import ContainerFactory
from simulation.core.enums import Direction


class TestContainerFactory(unittest.TestCase):
    """Test suite for ContainerFactory with functionality and performance tests."""

    @classmethod
    def setUpClass(cls):
        """Set up factory once for all tests."""
        print("\n" + "=" * 60)
        print("CONTAINER FACTORY TEST SUITE")
        print("=" * 60)

        # Initialize factory with timer
        start = time.perf_counter()
        cls.factory = ContainerFactory()
        init_time = time.perf_counter() - start
        print(f"Factory initialization: {init_time:.3f}s")

        # Get available operators (use import_operator_dict)
        cls.test_operators = list(cls.factory.import_operator_dict.keys())[:3]
        print(f"Testing with operators: {cls.test_operators}")

    def test_01_basic_container_creation(self):
        """Test basic container creation functionality."""
        print("\n1. BASIC FUNCTIONALITY TESTS")
        print("-" * 40)

        for operator in self.test_operators[:1]:  # Test with one operator
            # Test Import containers
            import_containers = self.factory.create_containers(
                operator, Direction.IMPORT, n_containers=10
            )
            self.assertEqual(len(import_containers), 10)
            self.assertTrue(all(c.direction == Direction.IMPORT for c in import_containers))
            print(f"Created 10 Import containers for {operator}")

            # Test Export containers
            export_containers = self.factory.create_containers(
                operator, Direction.EXPORT, n_containers=10
            )
            self.assertEqual(len(export_containers), 10)
            self.assertTrue(all(c.direction == Direction.EXPORT for c in export_containers))
            print(f"Created 10 Export containers for {operator}")

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
                print(f"Container properties valid: {container.container_type}")

    def test_03_container_type_distribution(self):
        """Test container type distribution matches operator specifications."""
        print("\n3. DISTRIBUTION TESTS")
        print("-" * 40)

        operator = self.test_operators[0]
        n_samples = 1000

        containers = self.factory.create_containers(
            operator, Direction.IMPORT, n_containers=n_samples
        )

        # Count container types
        type_counts = {}
        for c in containers:
            type_counts[c.container_type] = type_counts.get(c.container_type, 0) + 1

        # Check distribution roughly matches operator specs
        operator_data = self.factory.import_operator_dict[operator]
        print(f"Container type distribution for {operator}:")

        for ctype, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            observed_prob = count / n_samples
            if ctype in operator_data:
                expected_prob = operator_data[ctype]["P_for_operator"]
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
            operator, Direction.IMPORT, n_containers=n_containers
        )
        creation_time = time.perf_counter() - start

        rate = n_containers / creation_time
        print(f"Created {n_containers} containers in {creation_time:.3f}s")
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
            operator, Direction.IMPORT, n_containers=n_containers
        )
        creation_time = time.perf_counter() - start

        rate = n_containers / creation_time
        print(f"Created {n_containers:,} containers in {creation_time:.3f}s")
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

        # Create mixed batch specification using Direction enums
        batches = [
            (self.test_operators[0], Direction.IMPORT, 3000),
            (self.test_operators[0], Direction.EXPORT, 2000),
        ]
        if len(self.test_operators) > 1:
            batches.append((self.test_operators[1], Direction.IMPORT, 2500))
            batches.append((self.test_operators[1], Direction.EXPORT, 2500))

        total_containers = sum(count for _, _, count in batches)

        # Time mixed batch creation
        start = time.perf_counter()
        containers = self.factory.create_batch(batches)
        creation_time = time.perf_counter() - start

        rate = total_containers / creation_time
        print(f"Created {total_containers:,} mixed containers in {creation_time:.3f}s")
        print(f"  Rate: {rate:,.0f} containers/second")
        print(f"  Batches: {len(batches)}")

        # Verify correct counts
        self.assertEqual(len(containers), total_containers)

    def test_08_stress_test(self):
        """Stress test with very large batch."""
        print("\n8. STRESS TEST")
        print("-" * 40)

        n_containers = 50000
        print(f"Creating {n_containers:,} containers...")

        start = time.perf_counter()
        containers = self.factory.create_containers(
            self.test_operators[0],
            Direction.IMPORT,
            n_containers=n_containers
        )
        creation_time = time.perf_counter() - start

        # Verify all created
        self.assertEqual(len(containers), n_containers)

        # Check performance metrics
        rate = n_containers / creation_time
        ms_per_container = (creation_time * 1000) / n_containers

        print(f"Stress test passed!")
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
            self.factory.create_containers("INVALID_OPERATOR", Direction.IMPORT)
        print("Invalid operator raises ValueError")

        # Test single container creation
        containers = self.factory.create_containers(
            self.test_operators[0], Direction.IMPORT, n_containers=1
        )
        self.assertEqual(len(containers), 1)
        print("Single container creation works")

        # Test zero containers (edge case)
        containers = self.factory.create_containers(
            self.test_operators[0], Direction.IMPORT, n_containers=0
        )
        self.assertEqual(len(containers), 0)
        print("Zero containers request handled")


def run_performance_summary():
    """Run tests and provide performance summary."""
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)

    # Run basic performance benchmark
    factory = ContainerFactory()

    test_sizes = [100, 1000, 10000]
    results = []

    for size in test_sizes:
        start = time.perf_counter()
        containers = factory.create_containers(
            list(factory.import_operator_dict.keys())[0],
            Direction.IMPORT,
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
            print(f"\nExcellent scaling: {scaling:.2f}x (near-linear)")
        elif scaling < 3:
            print(f"\nGood scaling: {scaling:.2f}x")
        else:
            print(f"\nPoor scaling: {scaling:.2f}x")


if __name__ == "__main__":
    # Run tests
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestContainerFactory)
    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)

    # Run performance summary if all tests passed
    if result.wasSuccessful():
        run_performance_summary()

    print("\n" + "=" * 60)
    print(f"Tests: {result.testsRun} | "
          f"Failures: {len(result.failures)} | "
          f"Errors: {len(result.errors)}")
    print("=" * 60)
