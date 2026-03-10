"""Diagnostic: verify dest masks for TRAIN_TO_YARD with yard fill.

Checks whether import containers on the train have valid yard
destinations when the yard is partially filled with distractor containers.
"""
import random
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation.training.curriculum_trainer import create_env_factory, build_agent_config
from simulation.training.eval import ScalableEvalScenario, EvalParams
from simulation.rl.multihead_dqn.config import Dims


ROWS, BAYS, TIERS = 5, 58, 5
TRACKS, SPLIT_FACTOR = 7, 20

env_factory = create_env_factory(
    rows=ROWS, bays=BAYS, tiers=TIERS,
    tracks=TRACKS, split_factor=SPLIT_FACTOR,
    export_ratio=1.0, max_retries=2,
)

cfg = build_agent_config(
    rows=ROWS, bays=BAYS, tiers=TIERS,
    split_factor=SPLIT_FACTOR, tracks=TRACKS,
)
dims: Dims = cfg.unified


# ── Helpers ────────────────────────────────────────────────────

def _setup_scenario(n_imports, n_exports, yard_fill_pct, seed=42):
    """Create env and set up an eval scenario, return (env, scenario)."""
    from simulation.training.scenarios._base import TUTORIAL_TIME
    env = env_factory()
    env.current_time = TUTORIAL_TIME  # encoder needs this for urgency
    params = EvalParams(
        n_imports=n_imports,
        n_exports=n_exports,
        yard_fill_pct=yard_fill_pct,
        seed=seed,
    )
    sc = ScalableEvalScenario(params)
    sc.setup(env)
    return env, sc


def _get_dest_mask_info(env):
    """Get state tensor + dest_mask_fn, return info about RAIL source positions."""
    state = env._encode_state()
    dest_mask_fn = env._make_dest_mask_fn(state)

    # Find all occupied RAIL positions (import containers on trains)
    rail_start = dims.rail_row_start
    rail_end = dims.rail_row_end
    n_splits = dims.n_splits
    n_tiers = dims.n_tiers

    from simulation.env.state_encoder import CH

    results = []
    for r in range(rail_start, rail_end):
        for s in range(n_splits):
            for t in range(n_tiers):
                if state[CH.CONTAINER_START, r, s, t] > 0.5:
                    mask, n_spl = dest_mask_fn(r, s, t)
                    # Count valid yard destinations
                    yard_mask = mask[dims.yard_row_start:dims.yard_row_end]
                    n_valid = int(yard_mask.sum())
                    results.append({
                        'rail_row': r,
                        'split': s,
                        'tier': t,
                        'n_splits_needed': n_spl,
                        'n_valid_yard_dests': n_valid,
                        'total_mask_sum': int(mask.sum()),
                    })
    return results


def _count_yard_containers(env):
    """Count containers by size in the yard."""
    size_counts = {}
    for rec in env.yard.iter_records():
        ft = rec.container.length_ft
        size_counts[ft] = size_counts.get(ft, 0) + 1
    return size_counts


def _yard_occupancy_stats(env):
    """Report yard occupancy per tier."""
    occ = env.yard.occupancy_mask  # shape (n_tiers, n_rows, n_splits)
    stats = {}
    for t in range(env.yard.n_tiers):
        total = occ[t].size
        occupied = int(occ[t].sum())
        stats[t] = {'occupied': occupied, 'total': total, 'pct': occupied / total}
    return stats


# ── Tests ──────────────────────────────────────────────────────

class TestDestMaskWithFill:
    """Core tests: do TRAIN_TO_YARD dest masks have valid positions?"""

    @pytest.mark.parametrize("fill_pct", [0.0, 0.25, 0.50])
    def test_imports_have_valid_destinations(self, fill_pct):
        """Every import on the train should have at least 1 valid yard dest."""
        env, sc = _setup_scenario(
            n_imports=10, n_exports=10, yard_fill_pct=fill_pct, seed=42,
        )
        results = _get_dest_mask_info(env)
        assert len(results) > 0, f"No imports found on rail at fill={fill_pct}"

        failed = [r for r in results if r['n_valid_yard_dests'] == 0]
        print(f"\n--- fill={fill_pct:.0%}: {len(results)} imports on rail ---")
        for r in results:
            tag = "BLOCKED" if r['n_valid_yard_dests'] == 0 else "ok"
            print(
                f"  [{tag}] rail_row={r['rail_row']} s={r['split']} t={r['tier']} "
                f"n_splits={r['n_splits_needed']} → {r['n_valid_yard_dests']} valid dests"
            )

        if failed:
            print(f"\n  Yard container sizes: {_count_yard_containers(env)}")
            print(f"  Yard occupancy per tier:")
            for t, s in _yard_occupancy_stats(env).items():
                print(f"    tier {t}: {s['occupied']}/{s['total']} ({s['pct']:.1%})")

        assert len(failed) == 0, (
            f"fill={fill_pct}: {len(failed)}/{len(results)} imports have "
            f"ZERO valid yard destinations!"
        )

    @pytest.mark.parametrize("fill_pct", [0.0, 0.25, 0.50])
    def test_dest_mask_count_by_size(self, fill_pct):
        """Report how many valid destinations each container size gets."""
        env, sc = _setup_scenario(
            n_imports=20, n_exports=0, yard_fill_pct=fill_pct, seed=123,
        )
        results = _get_dest_mask_info(env)

        # Group by n_splits_needed
        by_size = {}
        for r in results:
            ns = r['n_splits_needed']
            by_size.setdefault(ns, []).append(r['n_valid_yard_dests'])

        print(f"\n--- fill={fill_pct:.0%}: dest counts by container size ---")
        for ns in sorted(by_size):
            vals = by_size[ns]
            print(
                f"  n_splits={ns:2d}: "
                f"min={min(vals):3d} max={max(vals):3d} "
                f"avg={np.mean(vals):.1f} "
                f"blocked={sum(1 for v in vals if v == 0)}/{len(vals)}"
            )

        # At least report — no hard assertion here
        total_blocked = sum(1 for r in results if r['n_valid_yard_dests'] == 0)
        print(f"  Total blocked: {total_blocked}/{len(results)}")


class TestSizeMismatchStacking:
    """Specifically test whether mixed sizes cause stacking problems."""

    def test_small_on_large_not_stackable_in_mask(self):
        """A 20ft container should NOT need to stack on a 40ft container.

        The dest mask should offer tier-0 positions in empty bays,
        not just stacking positions on occupied bays.
        """
        env, sc = _setup_scenario(
            n_imports=5, n_exports=0, yard_fill_pct=0.50, seed=99,
        )
        results = _get_dest_mask_info(env)

        # Check that tier-0 positions are available even at 50% fill
        # 50% fill means ~145 distractors in 5*58=290 tier-0 slots
        # So ~145 tier-0 slots are free — should have destinations
        occ_stats = _yard_occupancy_stats(env)
        print(f"\n--- Yard occupancy at 50% fill ---")
        for t, s in occ_stats.items():
            print(f"  tier {t}: {s['occupied']}/{s['total']} ({s['pct']:.1%})")

        yard_sizes = _count_yard_containers(env)
        print(f"  Yard container sizes: {yard_sizes}")

        for r in results:
            print(
                f"  Import: n_splits={r['n_splits_needed']} "
                f"→ {r['n_valid_yard_dests']} valid dests"
            )
            assert r['n_valid_yard_dests'] > 0, (
                f"Import with n_splits={r['n_splits_needed']} has NO valid "
                f"yard destinations at 50% fill!"
            )

    def test_homogeneous_40ft_fill_with_20ft_imports(self):
        """Force 40ft fill, 20ft imports — verify 20ft can still find space."""
        from simulation.training.scenarios._base import TUTORIAL_TIME
        env = env_factory()
        env.current_time = TUTORIAL_TIME

        # Manually fill 50% of bays with 40ft containers at tier 0
        from simulation.core.containers.container import Container
        from simulation.core.enums import Direction, GoodsType
        from simulation.core.facilities.yard import PlacementResult
        from simulation.training.scenarios._base import TUTORIAL_TIME, DEFAULT_DEPARTURE

        rng = random.Random(42)
        fill_bays = list(range(0, 58, 2))  # every other bay = 29 bays
        for bay in fill_bays:
            for row in range(ROWS):
                c = Container(
                    container_id=f"FILL_{bay}_{row}",
                    direction=Direction.IMPORT,
                    container_type="40DC",
                    arrival_date=TUTORIAL_TIME,
                    departure_date=DEFAULT_DEPARTURE,
                    goods_type=GoodsType.REGULAR,
                    length_ft=40, length_m=12.2, width_m=2.44,
                )
                env.yard.add_container(
                    c, PlacementResult(row=row, bay=bay, tier=0, start_split=0),
                )

        # Now try to get dest mask for a 20ft import
        from simulation.core.vehicles.train import Train
        from simulation.training.scenarios._base import _make_train, _slot_train

        imp_c = Container(
            container_id="IMP_20ft",
            direction=Direction.IMPORT,
            container_type="20DC",
            arrival_date=TUTORIAL_TIME,
            departure_date=DEFAULT_DEPARTURE,
            goods_type=GoodsType.REGULAR,
            length_ft=20, length_m=6.1, width_m=2.44,
        )
        train = _make_train("TR0", containers=[imp_c], num_wagons=5)
        _slot_train(env, train, track_id=0, anchor_bay=5)

        state = env._encode_state()
        dest_mask_fn = env._make_dest_mask_fn(state)

        # Find the import on rail
        from simulation.env.state_encoder import CH
        rail_start = dims.rail_row_start
        rail_end = dims.rail_row_end

        found = False
        for r in range(rail_start, rail_end):
            for s in range(dims.n_splits):
                for t in range(dims.n_tiers):
                    if state[CH.CONTAINER_START, r, s, t] > 0.5:
                        mask, n_spl = dest_mask_fn(r, s, t)
                        yard_mask = mask[dims.yard_row_start:dims.yard_row_end]
                        n_valid = int(yard_mask.sum())
                        print(
                            f"\n  20ft import: n_splits={n_spl}, "
                            f"valid_yard_dests={n_valid}"
                        )
                        # The odd bays (1,3,5,...) are completely free
                        # so there should be PLENTY of tier-0 space
                        assert n_valid > 0, (
                            "20ft import has NO valid yard destinations "
                            "despite 29 fully free odd-numbered bays!"
                        )
                        found = True
        assert found, "Could not find the 20ft import on the rail!"

    def test_homogeneous_40ft_fill_with_40ft_imports(self):
        """Force 40ft fill, 40ft imports — verify stacking works at tier 1."""
        from simulation.training.scenarios._base import TUTORIAL_TIME
        env = env_factory()
        env.current_time = TUTORIAL_TIME

        from simulation.core.containers.container import Container
        from simulation.core.enums import Direction, GoodsType
        from simulation.core.facilities.yard import PlacementResult
        from simulation.training.scenarios._base import TUTORIAL_TIME, DEFAULT_DEPARTURE

        # Fill ALL tier-0 of rows 0-2 with 40ft
        for bay in range(58):
            for row in range(3):
                c = Container(
                    container_id=f"FILL_{bay}_{row}",
                    direction=Direction.IMPORT,
                    container_type="40DC",
                    arrival_date=TUTORIAL_TIME,
                    departure_date=DEFAULT_DEPARTURE,
                    goods_type=GoodsType.REGULAR,
                    length_ft=40, length_m=12.2, width_m=2.44,
                )
                env.yard.add_container(
                    c, PlacementResult(row=row, bay=bay, tier=0, start_split=0),
                )

        # 40ft import on train
        from simulation.training.scenarios._base import _make_train, _slot_train

        imp_c = Container(
            container_id="IMP_40ft",
            direction=Direction.IMPORT,
            container_type="40DC",
            arrival_date=TUTORIAL_TIME,
            departure_date=DEFAULT_DEPARTURE,
            goods_type=GoodsType.REGULAR,
            length_ft=40, length_m=12.2, width_m=2.44,
        )
        train = _make_train("TR0", containers=[imp_c], num_wagons=5)
        _slot_train(env, train, track_id=0, anchor_bay=5)

        state = env._encode_state()
        dest_mask_fn = env._make_dest_mask_fn(state)

        from simulation.env.state_encoder import CH
        rail_start = dims.rail_row_start
        rail_end = dims.rail_row_end

        found = False
        for r in range(rail_start, rail_end):
            for s in range(dims.n_splits):
                for t in range(dims.n_tiers):
                    if state[CH.CONTAINER_START, r, s, t] > 0.5:
                        mask, n_spl = dest_mask_fn(r, s, t)
                        yard_mask = mask[dims.yard_row_start:dims.yard_row_end]
                        n_valid = int(yard_mask.sum())
                        print(
                            f"\n  40ft import: n_splits={n_spl}, "
                            f"valid_yard_dests={n_valid}"
                        )
                        # Rows 0-2 are full at tier 0 → tier 1 should be available
                        # Rows 3-4 are fully empty at tier 0 → available too
                        assert n_valid > 0, (
                            "40ft import has NO valid yard destinations "
                            "despite rows 3-4 being completely empty + tier 1 on rows 0-2!"
                        )
                        found = True
        assert found, "Could not find the 40ft import on the rail!"


class TestFillDistribution:
    """Verify distractor placement works as expected."""

    @pytest.mark.parametrize("fill_pct", [0.25, 0.50])
    def test_distractor_count(self, fill_pct):
        """Check that the right number of distractors were placed."""
        env, sc = _setup_scenario(
            n_imports=10, n_exports=10, yard_fill_pct=fill_pct, seed=42,
        )
        expected_total = int(BAYS * ROWS * fill_pct)
        # Count non-export, non-import containers
        n_distractor = 0
        n_export = 0
        n_task = 0
        for rec in env.yard.iter_records():
            cid = rec.container.container_id
            if "_D" in cid:
                n_distractor += 1
            elif "_EXP" in cid:
                n_export += 1
            else:
                n_task += 1

        print(
            f"\n--- fill={fill_pct:.0%}: "
            f"expected={expected_total} distractors, "
            f"actual={n_distractor}, exports={n_export}, other={n_task}"
        )
        # Some distractors may fail placement (30 attempts), allow 10% slack
        assert n_distractor >= expected_total * 0.8, (
            f"Only {n_distractor}/{expected_total} distractors placed"
        )

    def test_free_tier0_slots_at_50pct(self):
        """At 50% fill, verify there are still free tier-0 positions."""
        env, sc = _setup_scenario(
            n_imports=10, n_exports=10, yard_fill_pct=0.50, seed=42,
        )
        occ = env.yard.occupancy_mask  # (tiers, rows, splits)
        # Count free bays at tier 0 (a bay = split_factor consecutive splits)
        n_free_bays_total = 0
        for row in range(ROWS):
            for bay in range(BAYS):
                s_start = bay * SPLIT_FACTOR
                s_end = s_start + SPLIT_FACTOR
                if not occ[0, row, s_start:s_end].any():
                    n_free_bays_total += 1

        total_bay_slots = ROWS * BAYS
        print(
            f"\n--- 50% fill: {n_free_bays_total}/{total_bay_slots} "
            f"fully-free bay slots at tier 0 "
            f"({n_free_bays_total/total_bay_slots:.1%})"
        )
        # Must have substantial free space
        assert n_free_bays_total > 50, (
            f"Only {n_free_bays_total} free bay slots at tier 0 — "
            "not enough space for imports!"
        )
