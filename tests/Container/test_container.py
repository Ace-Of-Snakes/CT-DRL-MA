# test_container.py

import unittest
from datetime import datetime, timedelta

from simulation.core.containers.container import Container
from simulation.core.enums import Direction, GoodsType


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
            direction=Direction.IMPORT,
            container_type="FEU",
            arrival_date=self.base_arrival,
            departure_date=self.base_departure,
            goods_type=GoodsType.REGULAR
        )

        self.assertEqual(container.container_id, "TEST001")
        self.assertEqual(container.direction, Direction.IMPORT)
        self.assertEqual(container.container_type, "FEU")
        self.assertEqual(container.goods_type, GoodsType.REGULAR)
        self.assertEqual(container.arrival_date, self.base_arrival)
        self.assertEqual(container.departure_date, self.base_departure)

    def test_default_dimensions(self):
        """Test that default dimensions are set correctly."""
        # Standard container (FEU defaults: length_m=12.2, height_m=2.44, width_m=2.44)
        container = Container(
            container_id="TEST002",
            direction=Direction.EXPORT,
            container_type="FEU",
            arrival_date=self.base_arrival,
            departure_date=self.base_departure,
            is_high_cube=False
        )

        self.assertEqual(container.height_m, 2.44)
        self.assertEqual(container.length_m, 12.2)
        self.assertEqual(container.width_m, 2.44)

        # Compatibility properties
        self.assertEqual(container.height, 2.44)
        self.assertEqual(container.length, 12.2)
        self.assertEqual(container.width, 2.44)

        # High cube container
        hc_container = Container(
            container_id="TEST003",
            direction=Direction.EXPORT,
            container_type="FEU",
            arrival_date=self.base_arrival,
            departure_date=self.base_departure,
            is_high_cube=True,
            height_m=2.90
        )

        self.assertEqual(hc_container.height_m, 2.90)

    def test_special_container_types(self):
        """Test special container types (Trailer, Swap Body)."""
        trailer = Container(
            container_id="TRL001",
            direction=Direction.IMPORT,
            container_type="Trailer",
            arrival_date=self.base_arrival,
            departure_date=self.base_departure,
            is_trailer=True,
            width_m=2.55
        )

        self.assertTrue(trailer.is_special_type)
        self.assertEqual(trailer.width_m, 2.55)

        swap_body = Container(
            container_id="SWP001",
            direction=Direction.EXPORT,
            container_type="Swap Body",
            arrival_date=self.base_arrival,
            departure_date=self.base_departure,
            is_swap_body=True,
            width_m=2.55
        )

        self.assertTrue(swap_body.is_special_type)
        self.assertEqual(swap_body.width_m, 2.55)

    def test_days_calculations(self):
        """Test days_in_terminal and days_until_departure calculations."""
        container = Container(
            container_id="TEST004",
            direction=Direction.IMPORT,
            container_type="FEU",
            arrival_date=self.base_arrival,
            departure_date=self.base_departure
        )

        # Test days in terminal
        current_date = self.base_arrival + timedelta(days=5)
        self.assertEqual(container.days_in_terminal(current_date), 5)

        # Test with arrival date
        self.assertEqual(container.days_in_terminal(self.base_arrival), 0)

        # Test days until departure (uses departure_date directly, no estimator)
        days_until = container.days_until_departure(current_date)
        self.assertEqual(days_until, 9)  # 15 - 6 = 9 days (Jan 6 to Jan 15)

    def test_is_special_type_property(self):
        """Test the is_special_type property for non-special containers."""
        regular = Container(
            container_id="REG001",
            direction=Direction.IMPORT,
            container_type="FEU",
            arrival_date=self.base_arrival,
            departure_date=self.base_departure,
            is_swap_body=False,
            is_trailer=False
        )
        self.assertFalse(regular.is_special_type)

    def test_goods_type_enum_values(self):
        """Test that goods type enum values work correctly."""
        for goods_type in [GoodsType.REGULAR, GoodsType.REEFER, GoodsType.DANGEROUS_GOODS]:
            container = Container(
                container_id=f"GT_{goods_type.value}",
                direction=Direction.IMPORT,
                container_type="FEU",
                arrival_date=self.base_arrival,
                departure_date=self.base_departure,
                goods_type=goods_type
            )
            self.assertEqual(container.goods_type, goods_type)

    def test_direction_enum_values(self):
        """Test that direction enum values work correctly."""
        import_container = Container(
            container_id="DIR_IMP",
            direction=Direction.IMPORT,
            container_type="FEU",
            arrival_date=self.base_arrival,
            departure_date=self.base_departure
        )
        self.assertEqual(import_container.direction, Direction.IMPORT)
        self.assertEqual(import_container.direction.value, "Import")

        export_container = Container(
            container_id="DIR_EXP",
            direction=Direction.EXPORT,
            container_type="FEU",
            arrival_date=self.base_arrival,
            departure_date=self.base_departure
        )
        self.assertEqual(export_container.direction, Direction.EXPORT)
        self.assertEqual(export_container.direction.value, "Export")


def run_all_tests():
    """Run all test suites."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestContainer))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_all_tests()
