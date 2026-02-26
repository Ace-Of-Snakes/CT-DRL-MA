# tests/test_train_loader_rearrange.py
import unittest
from datetime import datetime, timedelta

from simulation.core.vehicles.train import Train
from simulation.core.containers.container import Container
from simulation.core.facilities.yard import StorageYard
from simulation.core.enums import Direction, GoodsType
from simulation.planning.train_loader import TrainLoader


class TestTrainLoaderRearrange(unittest.TestCase):
    def _make_container(self, cid: str, goods_type: GoodsType = GoodsType.REGULAR) -> Container:
        now = datetime.now()
        return Container(
            container_id=cid,
            direction=Direction.IMPORT,
            container_type="40FT",
            arrival_date=now,
            departure_date=now + timedelta(days=2),
            goods_type=goods_type,
            length_ft=40,
            length_m=12.2,
            width_m=2.44,
            height_m=2.59,
            is_high_cube=False,
            is_swap_body=False,
            is_trailer=False
        )

    def setUp(self):
        # Minimal yard; TrainLoader only needs yard if it must classify by pickup IDs
        self.yard = StorageYard(
            n_rows=2,
            n_bays=10,
            n_tiers=2,
            coordinates=[],
            validate=False
        )

    def test_rearrange_reefers_to_ends_and_dg_to_center(self):
        # Create train with 6 wagons
        train = Train(train_id="TRN_TEST", num_wagons=6)

        # Old indices: 0..5
        # Classify via actual containers:
        #  - wag 1 (index 1) -> Reefer
        #  - wag 2 (index 2) -> DG
        #  - wag 4 (index 4) -> Reefer
        #  - wag 0, 5 -> Regular
        #  - wag 3 -> stays empty (Regular by absence)
        c_reg0 = self._make_container("REG0", goods_type=GoodsType.REGULAR)
        c_reef1 = self._make_container("REEF1", goods_type=GoodsType.REEFER)
        c_dg2 = self._make_container("DG2", goods_type=GoodsType.DANGEROUS_GOODS)
        c_reef4 = self._make_container("REEF4", goods_type=GoodsType.REEFER)
        c_reg5 = self._make_container("REG5", goods_type=GoodsType.REGULAR)

        self.assertTrue(train.add_container(c_reg0, wagon_index=0))
        self.assertTrue(train.add_container(c_reef1, wagon_index=1))
        self.assertTrue(train.add_container(c_dg2, wagon_index=2))
        # wagon 3 intentionally empty
        self.assertTrue(train.add_container(c_reef4, wagon_index=4))
        self.assertTrue(train.add_container(c_reg5, wagon_index=5))

        # Keep original wagon objects for expected order computation
        orig_wagons = train.wagons[:]

        # Expected new order (indices before rearrange):
        # reefers at ends, DG centered, regulars fill around:
        # reefer_idxs = [1, 4], dg_idxs = [2], reg_idxs = [0, 3, 5]
        # half_reef = 1 -> front_reefers=[1], back_reefers=[4]
        # half_reg = 1 -> left_regular=[0], right_regular=[3,5]
        # new_order_old_indices = [1] + [0] + [2] + [3,5] + [4] = [1,0,2,3,5,4]
        expected_old_indices = [1, 0, 2, 3, 5, 4]
        expected_wagon_ids = [orig_wagons[i].wagon_id for i in expected_old_indices]

        # Rearrange
        old_to_new = TrainLoader.rearrange_wagons_for_goods(train, self.yard)

        # Validate wagon order by IDs
        actual_wagon_ids = [w.wagon_id for w in train.wagons]
        self.assertEqual(
            expected_wagon_ids, actual_wagon_ids,
            "Wagon order after rearrange does not match expected goods-aware layout"
        )

        # Validate mapping old_index -> new_index returned by TrainLoader
        # Build inverse: new_index -> old_index
        inv = {new: old for old, new in enumerate(old_to_new)}
        reconstructed_old_indices = [inv[i] for i in range(len(train.wagons))]
        self.assertEqual(
            expected_old_indices, reconstructed_old_indices,
            "Returned index mapping (old->new) is inconsistent with the final order"
        )

        # Check container location indices updated
        # c_reef1 was in old wagon 1 -> new index 0
        wagon, pos = train.find_container("REEF1")
        self.assertIsNotNone(wagon, "Reefer container not found after rearrange")
        self.assertEqual(train.wagons[0].wagon_id, wagon.wagon_id, "REEF1 not moved to expected new wagon index 0")

        # c_dg2 was in old wagon 2 -> new index 2
        wagon, pos = train.find_container("DG2")
        self.assertIsNotNone(wagon, "DG container not found after rearrange")
        self.assertEqual(train.wagons[2].wagon_id, wagon.wagon_id, "DG2 not positioned near center as expected")

        # c_reef4 was in old wagon 4 -> new index 5 (end)
        wagon, pos = train.find_container("REEF4")
        self.assertIsNotNone(wagon, "Reefer container 2 not found after rearrange")
        self.assertEqual(train.wagons[5].wagon_id, wagon.wagon_id, "REEF4 not moved to last wagon as expected")

        # Empty wagon was old index 3 -> new index 3; should be tracked as empty
        self.assertIn(3, train.empty_wagons, "Empty wagon should remain tracked as empty at its new index")


if __name__ == "__main__":
    unittest.main()