# simulation/planning/train_loader.py
"""High-performance train loader using first-fit-decreasing bin packing."""
import numpy as np
from collections import Counter
from typing import List, Optional, Dict
from dataclasses import dataclass
from datetime import datetime

from simulation.core.containers.container import Container
from simulation.core.vehicles.train import Train
from simulation.core.facilities.yard import OptimizedStorageYard
from simulation.core.enums import Direction
from simulation.config.train_config import TrainLoaderConfig

LENGTH_TOLERANCE_M: float = 0.001


@dataclass(slots=True)
class LoadingStats:
    """Statistics for a train loading operation."""
    total_containers_generated: int
    containers_loaded: int
    wagons_used: int
    avg_utilization: float
    time_elapsed_ms: float


class TrainLoader:
    """Train loader using first-fit-decreasing for wagon packing."""

    def __init__(
        self,
        container_factory,
        overgeneration_factor: Optional[float] = None,
    ):
        if not container_factory:
            raise ValueError("Container factory is required")
        self.factory = container_factory
        self.overgeneration_factor = (
            overgeneration_factor if overgeneration_factor is not None
            else TrainLoaderConfig.OVERGENERATION_FACTOR
        )

    # ================================================================
    # Loading
    # ================================================================

    def load_train(
        self,
        train: Train,
        operator: str,
        current_date: Optional[datetime] = None,
    ) -> Train:
        """Load a single train with containers (first-fit-decreasing)."""
        if not operator:
            raise ValueError("Operator must be specified for train loading")
        if not train or not train.wagons:
            return train

        num_wagons = len(train.wagons)
        containers = self.factory.create_containers(
            operator=operator,
            direction=Direction.IMPORT,
            n_containers=int(num_wagons * self.overgeneration_factor),
            base_arrival_date=current_date,
        )
        if containers:
            self._first_fit_decreasing(train, containers)
        return train

    def load_multiple_trains(
        self,
        trains: List[Train],
        operator: str,
        current_date: Optional[datetime] = None,
    ) -> List[Train]:
        """Load multiple trains with shared batch container generation."""
        if not operator:
            raise ValueError("Operator must be specified for train loading")
        if not trains:
            return trains

        total_wagons = sum(len(t.wagons) for t in trains)
        all_containers = self.factory.create_containers(
            operator=operator,
            direction=Direction.IMPORT,
            n_containers=int(total_wagons * self.overgeneration_factor),
            base_arrival_date=current_date,
        )
        if not all_containers:
            return trains

        idx = 0
        for train in trains:
            n = int(len(train.wagons) * self.overgeneration_factor)
            batch = all_containers[idx:idx + n]
            if batch:
                self._first_fit_decreasing(train, batch)
            idx += n
            if idx >= len(all_containers):
                break

        return trains

    # ================================================================
    # Core packing algorithm
    # ================================================================

    def _first_fit_decreasing(
        self,
        train: Train,
        containers: List[Container],
    ) -> None:
        """
        Single-pass first-fit-decreasing bin packing.

        Sort containers longest-first, then assign each to the first wagon
        with enough remaining capacity. O(n·log·n + n·m) where n=containers, m=wagons.
        """
        num_wagons = len(train.wagons)
        if num_wagons == 0:
            return

        # Sort longest-first for tighter packing
        sorted_containers = sorted(containers, key=lambda c: c.length_m, reverse=True)

        # Track remaining capacity per wagon
        remaining = np.array(
            [train.wagons[i].get_available_length() for i in range(num_wagons)],
            dtype=np.float64,
        )

        for container in sorted_containers:
            length = container.length_m
            # Find first wagon with enough space
            for w_idx in range(num_wagons):
                if remaining[w_idx] >= length - LENGTH_TOLERANCE_M:
                    if train.add_container(container, wagon_index=w_idx):
                        remaining[w_idx] = train.wagons[w_idx].get_available_length()
                        break

    # ================================================================
    # Wagon rearrangement
    # ================================================================

    @staticmethod
    def rearrange_wagons_for_goods(train: Train, yard: OptimizedStorageYard) -> List[int]:
        """
        Reorder wagons in-place: reefer at ends, DG in center, regular between.
        Returns old→new index mapping.
        """
        wagons = train.wagons
        n = len(wagons)
        if n <= 1:
            return list(range(n))

        reefer_idxs, dg_idxs, reg_idxs = [], [], []

        for i, wagon in enumerate(wagons):
            has_reefer, has_dg = False, False

            for container in wagon.get_container_list():
                if container.goods_type == "Reefer":
                    has_reefer = True
                elif container.goods_type == "DangerousGoods":
                    has_dg = True
                if has_reefer and has_dg:
                    break

            if not (has_reefer or has_dg) and wagon.pickup_container_ids:
                for cid in wagon.pickup_container_ids:
                    container = yard.get_container(cid)
                    if not container:
                        continue
                    if container.goods_type == "Reefer":
                        has_reefer = True
                    elif container.goods_type == "DangerousGoods":
                        has_dg = True
                    if has_reefer and has_dg:
                        break

            if has_dg:
                dg_idxs.append(i)
            elif has_reefer:
                reefer_idxs.append(i)
            else:
                reg_idxs.append(i)

        half_r = (len(reefer_idxs) + 1) // 2
        half_reg = len(reg_idxs) // 2

        new_order = (
            reefer_idxs[:half_r]
            + reg_idxs[:half_reg]
            + dg_idxs
            + reg_idxs[half_reg:]
            + reefer_idxs[half_r:]
        )

        if len(new_order) != n:
            new_order = list(range(n))

        train.wagons = [wagons[i] for i in new_order]

        old_to_new = {old: new for new, old in enumerate(new_order)}
        for loc in train.container_locations.values():
            loc.wagon_index = old_to_new.get(loc.wagon_index, loc.wagon_index)

        train.wagons_with_space = {i for i, w in enumerate(train.wagons) if not w.is_full()}
        train.empty_wagons = {i for i, w in enumerate(train.wagons) if w.is_empty()}

        return [old_to_new.get(i, i) for i in range(n)]

    # ================================================================
    # Stats / analysis
    # ================================================================

    def get_loading_stats(self, train: Train) -> LoadingStats:
        """Calculate loading statistics for a train."""
        stats = train.get_stats()
        return LoadingStats(
            total_containers_generated=0,
            containers_loaded=stats['total_containers'],
            wagons_used=sum(1 for w in train.wagons if not w.is_empty()),
            avg_utilization=stats['utilization_rate'],
            time_elapsed_ms=0,
        )

    @staticmethod
    def analyze_container_distribution(containers: List[Container]) -> Dict:
        """Analyze container type and length distribution."""
        if not containers:
            return {}

        types = Counter(c.container_type for c in containers)
        lengths_m = [c.length_m for c in containers]
        length_keys = Counter(f"{l:.3f}m" for l in lengths_m)

        return {
            'total_containers': len(containers),
            'type_distribution': dict(types),
            'length_distribution': dict(length_keys),
            'min_length': min(lengths_m),
            'max_length': max(lengths_m),
            'avg_length': sum(lengths_m) / len(lengths_m),
            'unique_types': len(types),
            'unique_lengths': len(length_keys),
        }

    @staticmethod
    def debug_wagon_state(train: Train) -> Dict:
        """Debug wagon loading state."""
        total_capacity = 0.0
        total_used = 0.0
        wagon_stats = []

        for i, wagon in enumerate(train.wagons):
            capacity = wagon.length
            used = wagon._used_length
            container_list = wagon.get_container_list()
            total_capacity += capacity
            total_used += used

            wagon_stats.append({
                'wagon_id': i,
                'capacity': capacity,
                'used': used,
                'available': wagon.get_available_length(),
                'utilization': (used / capacity * 100) if capacity > 0 else 0,
                'num_containers': len(container_list),
                'container_lengths': [c.length_m for c in container_list],
                'container_types': [c.container_type for c in container_list],
            })

        return {
            'wagon_details': wagon_stats,
            'total_capacity': total_capacity,
            'total_used': total_used,
            'overall_utilization': (total_used / total_capacity * 100) if total_capacity > 0 else 0,
            'wagons_empty': sum(1 for w in train.wagons if w.is_empty()),
            'wagons_full': sum(1 for w in train.wagons if w.is_full()),
        }