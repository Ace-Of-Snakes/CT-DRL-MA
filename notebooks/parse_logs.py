"""
Parse CNN×DQN comparison logs back into the `results` dict
expected by cnn_comparison.ipynb.

Usage:
    python parse_logs.py <logfile.txt> [--output cnn_dqn_results.json]

The script writes a JSON file that can be loaded in the notebook:

    import json
    with open("cnn_dqn_results.json") as f:
        results = json.load(f)
"""

import re
import json
import argparse
from pathlib import Path

# ── Scenario metadata (matches unified_tutorial_runner.py) ───────────

SCENARIO_NAMES = {
    1: "park_truck",
    2: "train_import",
    3: "yard_to_truck",
    4: "yard_to_train",
    5: "import_and_export",
    6: "park_then_import",
    7: "full_chain_truck",
    8: "delivery_to_train",
    9: "unbury_for_truck",
    10: "unbury_for_train",
    11: "random_import_chain",
    12: "random_export_chain",
    13: "multi_truck_serving",
    14: "deep_unbury",
    15: "multi_delivery_park",
    16: "delivery_to_train_pipeline",
    17: "bidirectional_train_simple",
    18: "concurrent_import_export",
    19: "mixed_fleet_dispatch",
    20: "crowded_yard_pickup",
    21: "full_mini_day",
}

# Reverse lookup: name -> scenario id
NAME_TO_ID = {v: k for k, v in SCENARIO_NAMES.items()}

N_TIERS = 10


def parse_log(text: str) -> dict:
    """Parse the full log text and return a results dict keyed by 'cnn+dqn'."""

    results = {}

    # Split into per-run blocks using the header pattern
    # [1/15] CNN: BASELINE  ×  DQN: BASELINE
    header_re = re.compile(
        r"\[(\d+)/(\d+)\]\s+CNN:\s+(\w+)\s+[×x]\s+DQN:\s+(\w+)"
    )

    # Find all header positions
    headers = list(header_re.finditer(text))
    if not headers:
        raise ValueError("No run headers found in log. Expected pattern like "
                         "'[1/15] CNN: BASELINE × DQN: BASELINE'")

    for idx, match in enumerate(headers):
        run_num = int(match.group(1))
        total_runs = int(match.group(2))
        cnn_name = match.group(3).lower()
        dqn_name = match.group(4).lower()
        run_key = f"{cnn_name}+{dqn_name}"

        # Slice this run's text block
        start = match.start()
        end = headers[idx + 1].start() if idx + 1 < len(headers) else len(text)
        block = text[start:end]

        result = parse_run_block(block, cnn_name, dqn_name)
        if result is not None:
            results[run_key] = result
            print(f"  Parsed [{run_num}/{total_runs}] {cnn_name}+{dqn_name}: "
                  f"{'MASTERED' if result['mastered'] else 'NOT MASTERED'} "
                  f"in {result['epochs']} epochs")

    return results


def parse_run_block(block: str, cnn_name: str, dqn_name: str) -> dict | None:
    """Parse a single run block and return a result dict."""

    # ── Parameters ───────────────────────────────────────────────────
    m = re.search(r"Total parameters:\s+([\d,]+)", block)
    n_params = int(m.group(1).replace(",", "")) if m else 0

    m = re.search(r"Backbone parameters:\s+([\d,]+)", block)
    backbone_params = int(m.group(1).replace(",", "")) if m else 0

    # ── Result line ──────────────────────────────────────────────────
    # Result: MASTERED in 336 epochs (3511s wall time)
    # Result: NOT MASTERED after 500 epochs (5200s wall time)
    result_re = re.compile(
        r"Result:\s+(MASTERED|NOT MASTERED)\s+(?:in|after)\s+(\d+)\s+epochs\s+"
        r"\((\d+(?:\.\d+)?)s\s+wall\s+time\)"
    )
    m = result_re.search(block)
    if m:
        mastered = m.group(1) == "MASTERED"
        epochs = int(m.group(2))
        wall_time = float(m.group(3))
    else:
        # Try to infer from "ALL X TIERS MASTERED at epoch N"
        m_all = re.search(r"ALL\s+\d+\s+TIERS\s+MASTERED\s+at\s+epoch\s+(\d+)", block)
        if m_all:
            mastered = True
            epochs = int(m_all.group(1))
        else:
            # Find the last epoch number mentioned
            epoch_matches = re.findall(r"Epoch\s+(\d+)/(\d+)", block)
            if epoch_matches:
                epochs = int(epoch_matches[-1][0])
                mastered = False
            else:
                print(f"  WARNING: Could not parse result for {cnn_name}+{dqn_name}, skipping")
                return None
        wall_time = 0.0

    # ── Tier reached ─────────────────────────────────────────────────
    # "Tier reached: 10/10"
    m = re.search(r"Tier reached:\s+(\d+)/(\d+)", block)
    if m:
        current_tier = int(m.group(1))
        n_tiers = int(m.group(2))
    else:
        # Count mastered tiers from the log
        tier_mastered = re.findall(r"TIER\s+\d+\s+MASTERED", block)
        current_tier = len(tier_mastered)
        n_tiers = N_TIERS

    # ── Final pass rates (last epoch block before Result) ────────────
    pass_rates = extract_final_pass_rates(block)

    return {
        "cnn": cnn_name,
        "dqn": dqn_name,
        "epochs": epochs,
        "mastered": mastered,
        "current_tier": current_tier,
        "n_tiers": n_tiers,
        "pass_rates": pass_rates,
        "scenario_names": {str(k): v for k, v in SCENARIO_NAMES.items()},
        "wall_time_s": wall_time,
        "n_params": n_params,
        "backbone_params": backbone_params,
    }


def extract_final_pass_rates(block: str) -> dict:
    """Extract per-scenario pass rates from the last epoch progress block."""

    # Find all epoch blocks: "Epoch NNN/500 — Tier ..."
    epoch_starts = [m.start() for m in re.finditer(r"Epoch\s+\d+/\d+\s+—", block)]

    # Also check for a final "Result:" line after the last epoch
    result_pos = block.rfind("Result:")
    if result_pos == -1:
        result_pos = len(block)

    # Use the last epoch block (up to the Result line or end of block)
    if epoch_starts:
        last_epoch_start = epoch_starts[-1]
        last_block = block[last_epoch_start:result_pos]
    else:
        last_block = block[:result_pos]

    # Parse scenario lines like:
    #   [====================] 100.0% OK S1: park_truck
    #   [=================---] 85.0%    S21: full_mini_day
    scenario_re = re.compile(
        r"\[=*-*\]\s+([\d.]+)%\s+(?:OK\s+)?S(\d+):\s+(\w+)"
    )

    pass_rates = {}
    for m in scenario_re.finditer(last_block):
        rate = float(m.group(1)) / 100.0
        scenario_id = int(m.group(2))
        pass_rates[str(scenario_id)] = rate

    # Fill in any missing scenarios with 0.0
    # (locked tiers won't have progress bars in the log)
    for sid in SCENARIO_NAMES:
        if str(sid) not in pass_rates:
            pass_rates[str(sid)] = 0.0

    return pass_rates


def main():
    parser = argparse.ArgumentParser(
        description="Parse CNN×DQN comparison logs into JSON results"
    )
    parser.add_argument("logfile", type=Path, help="Path to the log file")
    parser.add_argument(
        "--output", "-o", type=Path, default=Path("cnn_dqn_results.json"),
        help="Output JSON file (default: cnn_dqn_results.json)"
    )
    args = parser.parse_args()

    if not args.logfile.exists():
        raise FileNotFoundError(f"Log file not found: {args.logfile}")

    print(f"Parsing: {args.logfile}")
    text = args.logfile.read_text(encoding="utf-8", errors="replace")

    results = parse_log(text)

    print(f"\nParsed {len(results)} run(s)")

    # Write JSON
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {args.output}")

    # Quick summary
    print(f"\n{'CNN':<15} {'DQN':<15} {'Epochs':>7} {'Mastered':>9} {'Tier':>6} {'Wall(s)':>8}")
    print("-" * 65)
    for key, r in sorted(results.items()):
        status = "Yes" if r["mastered"] else "No"
        tier = f"{r['current_tier']}/{r['n_tiers']}"
        print(f"{r['cnn']:<15} {r['dqn']:<15} {r['epochs']:>7} {status:>9} {tier:>6} {r['wall_time_s']:>8.0f}")


if __name__ == "__main__":
    main()
