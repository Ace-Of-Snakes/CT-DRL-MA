# Bridging Scenarios Plan (Phase 2 Pre-Training)

> **Status**: ON HOLD — waiting for Phase 2 Run 2 results before deciding.
> Created 2026-03-05 based on Run 1 diagnostics:
> Level 0 (50 containers/day): reward=-129.3, moves=379, truck_wait=170.9min, reshuffle=55.4%

## Problem Diagnosis

The tutorials teach isolated skills in near-empty yards.
Phase 2 drops the agent into continuous 7-day operations with 50+ containers/day.
The gap manifests as:
- **55% reshuffle rate** — agent does YARD_TO_YARD compulsively without purpose
- **170 min truck wait** — agent doesn't prioritize serving waiting vehicles
- **Negative total reward** — penalties (idle, waiting, reshuffles) overwhelm productive moves
- **Low throughput** — only ~54 moves/day when ~100+ needed

## Proposed Scenarios

### S32: Yard Triage (Priority Under Pressure)

**Goal**: Teach the agent to serve waiting vehicles BEFORE reshuffling.

**Setup**:
- Yard pre-filled: 25-30 containers across rows 0-3, mixed Import/Export
  - 5 IMPORT containers with matching pickup trucks (already parked)
  - 5 EXPORT containers with a train wanting them
  - 15-20 distractors (no one is waiting for these)
- 1 train docked, departure in ~30 steps (urgency)
- 5 trucks parked, waiting

**Success criteria**:
- All 5 pickup trucks served (containers delivered, trucks departed)
- All 5 exports loaded on train
- `max_steps = 80`
- `expected_moves = 15` (optional soft budget: 5 pickups + 5 loads + ~5 restacks max)

**What it teaches**:
- Prioritize exports/pickups over aimless reshuffling
- Work from a pre-populated yard (not near-empty)
- Serve multiple vehicles under time pressure
- YARD_TO_YARD should only happen to unbury a target, not randomly

**Placement strategy**:
- Bury 2-3 of the targets under 1 blocker each (force *some* Y2Y but not 55%)
- Spread targets across bays 5-50 (not clustered)

```python
class YardTriage(TutorialScenario):
    id = 32
    name = "yard_triage"
    description = "Serve 5 trucks + load 5 exports from pre-filled yard"
    max_steps = 80
    expected_moves = 15  # soft budget
    repeatable = True
```

---

### S33: Sustained Flow (Mini Continuous Operations)

**Goal**: Teach the agent to handle arrivals over time (not just pre-placed entities).

**Setup** (multi-phase, simulating a condensed day):
- Start with 10 containers in yard (5 Import waiting for pickup, 5 Export)
- 3 trucks arrive at setup (park from queue)
- 1 train docked with 5 imports + wanting 5 exports
- After parking the 3 trucks, more work appears:
  - The 5 train imports need storing
  - The 5 exports need loading
  - The 3 trucks need containers delivered

**Success criteria**:
- All 5 exports loaded on train
- All 3 pickup trucks served
- All 5 train imports stored in yard
- `max_steps = 120`

**What it teaches**:
- Interleave parking, importing, exporting, delivering
- Don't get stuck in one mode (e.g., only doing imports)
- Pipeline thinking: park → import → store → locate → deliver

```python
class SustainedFlow(TutorialScenario):
    id = 33
    name = "sustained_flow"
    description = "Continuous operations: interleave park, import, export, deliver"
    max_steps = 120
    repeatable = True
```

---

### S34: Reshuffle Discipline (Anti-Pattern Training)

**Goal**: Specifically penalize excessive reshuffling. Teach "do Y2Y only when it enables a delivery."

**Setup**:
- Yard: 20 containers, 3 targets buried at tier 0 under 1-2 blockers each
- 3 pickup trucks parked, one per target
- NO distractors that need moving (everything else is irrelevant)

**Success criteria**:
- All 3 trucks served
- Move budget: `expected_moves = 9` (3 targets × (1-2 unbury + 1 deliver))
- `max_steps = 40`

**What it teaches**:
- Y2Y is a MEANS to an end, not the end itself
- Tight move budget forces efficiency
- Similar to S14 (deep unbury) but with 3 simultaneous targets

```python
class ReshuffleDiscipline(TutorialScenario):
    id = 34
    name = "reshuffle_discipline"
    description = "Serve 3 buried targets with minimal reshuffles"
    max_steps = 40
    expected_moves = 9
    repeatable = True
```

---

## Tier Placement

These would form a new **Tier 13: Operational Readiness**, added after the current capstone:

```python
TIER_13 = ("Operational readiness", [S32, S33, S34])
```

Rationale: the agent should already master all individual skills (Tiers 0-12)
before learning to compose them under operational pressure.

## Integration into Phase 2

Instead of re-running the full Phase 1 tutorial from scratch:

1. Load the existing tutorial checkpoint (already saved)
2. Run 50-100 additional epochs on Tiers 0-13 (all scenarios including new ones)
3. The old tiers act as retention training; Tier 13 teaches the missing skills
4. Then proceed to the level system

This avoids recomputing the 300-epoch Phase 1 from scratch.

## File Structure

```
simulation/training/scenarios/
  10_operational/          # NEW directory
    __init__.py
    s32_yard_triage.py
    s33_sustained_flow.py
    s34_reshuffle_discipline.py
  _registry.py             # Add TIER_13 import + registration
```

## Decision Point

**Wait for Run 2 results.** If the agent shows improvement from online learning
(reward trending positive, reshuffle rate dropping), these scenarios may not be
needed. If Run 2 looks similar to Run 1, implement these and do a short
pre-training pass before re-running Phase 2.
