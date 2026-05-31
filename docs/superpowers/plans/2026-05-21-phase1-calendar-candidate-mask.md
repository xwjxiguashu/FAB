# Phase 1 Calendar Candidate Mask Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first-stage local code from `报告.md` section 10: a stable resource-calendar dispatch environment with fixed-length candidate pools, padding, action masks, dry-run feasibility checks, and commit output for Lot-level and Wafer-level schedules.

**Architecture:** Add a standalone Phase 1 environment layer in `rl_environment.py` that wraps the existing `TwoPopulationScheduler`/calendar utilities without changing NSGA-II or PPO code. The environment exposes per-machine candidate pools of `(Lot, Machine, PPID)` actions, uses dry-run scheduling against copied Machine and Chamber/Side calendars, and commits only valid masked actions into a persistent `ScheduleState`.

**Tech Stack:** Python 3, NumPy, dataclasses, existing `FABenv` modules (`core.py`, `resource_calendar.py`, `state.py`), pytest for local verification.

---

## Spec Slice From `报告.md`

阶段 1 requires:

- Keep and improve two-level resource calendars: Machine and Chamber/Side.
- For each candidate `(Lot, Machine, PPID)`, run a dry-run insertability check.
- Commit valid actions into consistent Lot-level and Wafer-level schedules.
- Build a fixed-length candidate pool for a machine decision.
- Candidate pool comes from arrived and unfinished Lots.
- Use Top-K, padding, and action mask to produce `A_fixed^m`.
- Mask invalid actions caused by Lot not arrived, unsupported Machine/Recipe, missing PPID stages, or calendar insert failure.
- Do not introduce RL/PPO training in this phase.

## File Structure

- Create: `rl_environment.py`
  - Owns Phase 1 environment objects.
  - Defines `DispatchAction`, `CandidatePool`, `DispatchCommitResult`.
  - Defines `ResourceCalendarEnv` with `build_candidate_pool()`, `dry_run_action()`, `commit_action_index()`, and `advance_time()`.
  - Reuses existing calendar functions from the wrapped scheduler instance.

- Modify: `__init__.py`
  - Export `ResourceCalendarEnv`, `DispatchAction`, `CandidatePool`, `DispatchCommitResult`.

- Create: `tests/test_rl_environment.py`
  - Builds a small deterministic scheduler.
  - Verifies fixed-length candidate pool and padding mask.
  - Verifies not-arrived and machine-incompatible lots are filtered.
  - Verifies dry-run does not mutate calendars.
  - Verifies commit writes conflict-free Lot/Wafer schedules and updates remaining lots.
  - Verifies selecting a masked index raises `ValueError`.

- Create: `run_phase1_environment_demo.py`
  - Local smoke demo using the existing small instance builder.
  - Builds candidate pools, commits valid actions, prints schedule arrays and objective summary.

---

### Task 1: Candidate Pool Tests

**Files:**
- Create: `tests/test_rl_environment.py`
- Read: `run_small_instance_gantt.py`
- Read: `resource_calendar.py`
- Read: `state.py`

- [ ] **Step 1: Create the failing test file**

Create `tests/test_rl_environment.py` with this content:

```python
import sys
from pathlib import Path

import numpy as np
import pytest

PACKAGE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from FABenv import ResourceCalendarEnv
from FABenv.run_small_instance_gantt import build_small_encoder


def build_validated_encoder():
    encoder = build_small_encoder()
    encoder.validate_problem_definition()
    return encoder


def test_candidate_pool_has_fixed_length_padding_and_mask():
    encoder = build_validated_encoder()
    env = ResourceCalendarEnv(encoder, current_time=0.0, top_k=4)

    pool = env.build_candidate_pool(machine=1)

    assert pool.machine == 1
    assert len(pool.actions) == 4
    assert pool.action_mask.shape == (4,)
    assert pool.features.shape == (4, len(env.feature_names))
    assert pool.action_mask.tolist() == [True, True, False, False]
    assert [(a.lot, a.machine, a.ppid) for a in pool.valid_actions()] == [
        (1, 1, 101),
        (1, 1, 0),
    ]


def test_candidate_pool_filters_not_arrived_and_incompatible_machine():
    encoder = build_validated_encoder()
    env = ResourceCalendarEnv(encoder, current_time=1.6, top_k=6)

    machine_2_pool = env.build_candidate_pool(machine=2)
    valid_actions = machine_2_pool.valid_actions()

    assert all(action.machine == 2 for action in valid_actions)
    assert {action.lot for action in valid_actions} == {1}
    assert all(action.lot != 2 for action in valid_actions)
    assert all(action.ppid in {201, 0} for action in valid_actions)
```

The expected valid action `(1, 1, 0)` is the explicit wait/no-op action for Machine 1. It is valid when at least one real candidate exists, so later policy code can choose to wait without confusing padding with no-op.

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```powershell
python -m pytest tests/test_rl_environment.py::test_candidate_pool_has_fixed_length_padding_and_mask tests/test_rl_environment.py::test_candidate_pool_filters_not_arrived_and_incompatible_machine -v
```

Expected: FAIL with an import error similar to:

```text
ImportError: cannot import name 'ResourceCalendarEnv' from 'FABenv'
```

- [ ] **Step 3: Commit the failing tests**

```powershell
git add tests/test_rl_environment.py
git commit -m "test: specify phase1 candidate pool behavior"
```

---

### Task 2: Environment Dataclasses And Candidate Pool

**Files:**
- Create: `rl_environment.py`
- Modify: `__init__.py`
- Test: `tests/test_rl_environment.py`

- [ ] **Step 1: Add the Phase 1 environment module**

Create `rl_environment.py` with this implementation:

```python
from dataclasses import dataclass, field

import numpy as np

from .state import ScheduleState


CANDIDATE_FEATURE_NAMES = (
    "waiting_time",
    "due_urgency",
    "priority",
    "estimated_duration",
    "machine_load",
    "qtime_risk",
    "is_wait",
)


@dataclass(frozen=True)
class DispatchAction:
    lot: int = 0
    machine: int = 0
    ppid: int = 0
    estimated_start: float = 0.0
    estimated_end: float = 0.0
    score: float = 0.0
    is_wait: bool = False
    is_padding: bool = True


@dataclass
class CandidatePool:
    machine: int
    current_time: float
    actions: list
    action_mask: np.ndarray
    features: np.ndarray
    invalid_reasons: list = field(default_factory=list)

    def valid_indices(self):
        return [idx for idx, is_valid in enumerate(self.action_mask) if bool(is_valid)]

    def valid_actions(self):
        return [self.actions[idx] for idx in self.valid_indices()]


@dataclass
class DispatchCommitResult:
    action: DispatchAction
    lot_schedule: np.ndarray
    wafer_schedule: np.ndarray
    state: ScheduleState


class ResourceCalendarEnv:
    feature_names = CANDIDATE_FEATURE_NAMES

    def __init__(self, encoder, current_time=None, top_k=8):
        self.encoder = encoder
        self.top_k = int(top_k)
        self.encoder.validate_problem_definition()

        if current_time is None:
            current_time = min(float(v) for v in self.encoder.arrival_times.values())

        self.current_time = float(current_time)
        self.state = ScheduleState()
        self.remaining_lots = set(range(1, self.encoder.num_lots + 1))
        self.lot_schedule = np.empty((0, 5), dtype=float)
        self.wafer_schedule = np.empty((0, 9), dtype=float)

    def reset(self, current_time=None, state=None, remaining_lots=None):
        if current_time is None:
            current_time = min(float(v) for v in self.encoder.arrival_times.values())

        self.current_time = float(current_time)
        self.state = ScheduleState() if state is None else state
        self.remaining_lots = (
            set(range(1, self.encoder.num_lots + 1))
            if remaining_lots is None
            else set(int(lot) for lot in remaining_lots)
        )
        self.lot_schedule = np.empty((0, 5), dtype=float)
        self.wafer_schedule = np.empty((0, 9), dtype=float)
        return self

    def advance_time(self, next_time):
        next_time = float(next_time)
        if next_time < self.current_time:
            raise ValueError("next_time must be greater than or equal to current_time")
        self.current_time = next_time

    def _copy_state(self, state):
        return ScheduleState(
            machine_available_time=dict(state.machine_available_time),
            chamber_available_time=dict(state.chamber_available_time),
            machine_calendar=self.encoder.copy_calendar(state.machine_calendar),
            chamber_calendar=self.encoder.copy_calendar(state.chamber_calendar),
        )

    def _padding_action(self):
        return DispatchAction(is_padding=True)

    def _wait_action(self, machine):
        return DispatchAction(
            lot=0,
            machine=int(machine),
            ppid=0,
            estimated_start=self.current_time,
            estimated_end=self.current_time,
            score=-1.0,
            is_wait=True,
            is_padding=False,
        )

    def _arrived_unfinished_lots(self):
        return [
            lot for lot in sorted(self.remaining_lots)
            if float(self.encoder.arrival_times[lot]) <= self.current_time
        ]

    def _iter_structural_actions(self, machine):
        machine = int(machine)

        for lot in self._arrived_unfinished_lots():
            machine_list = set(int(m) for m in self.encoder.get_machine_list(lot))
            if machine not in machine_list:
                continue

            for ppid in self.encoder.get_ppid_list(lot, machine):
                ppid = int(ppid)
                if (lot, machine, ppid) not in self.encoder.ppid_steps:
                    continue
                yield lot, machine, ppid

    def _select_stage_resource(self, machine, candidate_resources, wafer_current_time, chamber_calendar):
        candidate_resources = np.asarray(candidate_resources, dtype=float)
        best = None

        for row in candidate_resources:
            chamber = int(row[0])
            side = int(row[1])
            process_time = float(row[2])
            resource_key = (int(machine), chamber, side)
            start_time = self.encoder.find_earliest_slot(
                chamber_calendar.get(resource_key, []),
                wafer_current_time,
                process_time,
            )
            end_time = start_time + process_time
            candidate = (end_time, start_time, chamber, side, process_time, resource_key)

            if best is None or candidate < best:
                best = candidate

        if best is None:
            raise ValueError("candidate_resources is empty")

        end_time, start_time, chamber, side, process_time, resource_key = best
        return chamber, side, process_time, start_time, end_time, resource_key

    def _schedule_lot_on_state(self, lot, machine, ppid, state):
        lot = int(lot)
        machine = int(machine)
        ppid = int(ppid)
        steps = self.encoder.get_process_steps(lot, machine, ppid)
        wafer_count = int(self.encoder.wafer_counts[lot])
        earliest_release_time = max(
            self.current_time,
            float(self.encoder.arrival_times[lot]),
        )
        machine_busy_intervals = state.machine_calendar.get(machine, [])
        lot_release_time = self.encoder.find_earliest_slot(
            machine_busy_intervals,
            earliest_release_time,
            0.0,
        )

        for _ in range(20):
            trial_state = self._copy_state(state)
            wafer_rows = []
            lot_start_time = np.inf
            lot_end_time = -np.inf

            for wafer_id in range(1, wafer_count + 1):
                wafer_current_time = lot_release_time

                for stage_index, stage in enumerate(steps, start=1):
                    chamber, side, process_time, start_time, end_time, resource_key = (
                        self._select_stage_resource(
                            machine,
                            stage,
                            wafer_current_time,
                            trial_state.chamber_calendar,
                        )
                    )
                    self.encoder.add_calendar_interval(
                        trial_state.chamber_calendar,
                        resource_key,
                        start_time,
                        end_time,
                    )
                    trial_state.chamber_available_time[resource_key] = max(
                        trial_state.chamber_available_time.get(resource_key, self.current_time),
                        end_time,
                    )
                    wafer_current_time = end_time
                    lot_start_time = min(lot_start_time, start_time)
                    lot_end_time = max(lot_end_time, end_time)
                    wafer_rows.append([
                        lot,
                        wafer_id,
                        machine,
                        ppid,
                        stage_index,
                        chamber,
                        side,
                        start_time,
                        end_time,
                    ])

            lot_duration = max(0.0, lot_end_time - lot_release_time)
            machine_slot_start = self.encoder.find_earliest_slot(
                machine_busy_intervals,
                earliest_release_time,
                lot_duration,
            )

            if abs(machine_slot_start - lot_release_time) <= 1e-9:
                self.encoder.add_calendar_interval(
                    trial_state.machine_calendar,
                    machine,
                    lot_release_time,
                    lot_end_time,
                )
                trial_state.machine_available_time[machine] = max(
                    trial_state.machine_available_time.get(machine, self.current_time),
                    lot_end_time,
                )
                lot_row = np.array(
                    [[lot, machine, ppid, lot_start_time, lot_end_time]],
                    dtype=float,
                )
                wafer_array = np.array(wafer_rows, dtype=float)
                return DispatchCommitResult(
                    action=DispatchAction(
                        lot=lot,
                        machine=machine,
                        ppid=ppid,
                        estimated_start=lot_start_time,
                        estimated_end=lot_end_time,
                        score=0.0,
                        is_wait=False,
                        is_padding=False,
                    ),
                    lot_schedule=lot_row,
                    wafer_schedule=wafer_array,
                    state=trial_state,
                )

            lot_release_time = machine_slot_start

        raise RuntimeError(f"Could not find a stable slot for Lot {lot} on Machine {machine}")

    def dry_run_action(self, action):
        if action.is_padding:
            raise ValueError("Cannot dry-run a padding action")
        if action.is_wait:
            return DispatchCommitResult(
                action=action,
                lot_schedule=np.empty((0, 5), dtype=float),
                wafer_schedule=np.empty((0, 9), dtype=float),
                state=self._copy_state(self.state),
            )
        return self._schedule_lot_on_state(
            action.lot,
            action.machine,
            action.ppid,
            self._copy_state(self.state),
        )

    def _candidate_score_and_features(self, lot, machine, ppid, dry_run):
        due_date = float(self.encoder.due_dates.get(lot, np.inf))
        waiting_time = max(0.0, self.current_time - float(self.encoder.arrival_times[lot]))
        due_urgency = 0.0 if not np.isfinite(due_date) else -(due_date - self.current_time)
        priority = float(self.encoder.priorities.get(lot, 0.0))
        estimated_duration = float(dry_run.lot_schedule[0, 4] - dry_run.lot_schedule[0, 3])
        machine_load = self.encoder.calendar_busy_time(
            self.state.machine_calendar,
            int(machine),
            self.current_time,
        )
        steps = self.encoder.get_process_steps(lot, machine, ppid)
        qtime_risk = self.encoder.estimate_qtime_risk(lot, machine, ppid, steps)
        features = np.array([
            waiting_time,
            due_urgency,
            priority,
            estimated_duration,
            machine_load,
            qtime_risk,
            0.0,
        ], dtype=float)
        score = (
            2.0 * priority
            + 1.0 * waiting_time
            + 0.2 * due_urgency
            - 0.1 * estimated_duration
            - 0.1 * machine_load
            - 0.5 * qtime_risk
        )
        return float(score), features

    def build_candidate_pool(self, machine, top_k=None):
        machine = int(machine)
        top_k = self.top_k if top_k is None else int(top_k)
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        valid_rows = []
        invalid_reasons = []

        for lot, machine, ppid in self._iter_structural_actions(machine):
            seed_action = DispatchAction(
                lot=lot,
                machine=machine,
                ppid=ppid,
                is_padding=False,
            )

            try:
                dry_run = self.dry_run_action(seed_action)
            except Exception as exc:
                invalid_reasons.append({
                    "lot": lot,
                    "machine": machine,
                    "ppid": ppid,
                    "reason": str(exc),
                })
                continue

            score, features = self._candidate_score_and_features(
                lot,
                machine,
                ppid,
                dry_run,
            )
            action = DispatchAction(
                lot=lot,
                machine=machine,
                ppid=ppid,
                estimated_start=float(dry_run.lot_schedule[0, 3]),
                estimated_end=float(dry_run.lot_schedule[0, 4]),
                score=score,
                is_wait=False,
                is_padding=False,
            )
            valid_rows.append((action, features))

        valid_rows.sort(key=lambda item: (-item[0].score, item[0].lot, item[0].ppid))

        actions = [row[0] for row in valid_rows[: max(0, top_k - 1)]]
        feature_rows = [row[1] for row in valid_rows[: max(0, top_k - 1)]]

        if len(actions) > 0:
            wait_action = self._wait_action(machine)
            actions.append(wait_action)
            feature_rows.append(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=float))

        action_mask = [True] * len(actions)

        while len(actions) < top_k:
            actions.append(self._padding_action())
            feature_rows.append(np.zeros(len(self.feature_names), dtype=float))
            action_mask.append(False)

        return CandidatePool(
            machine=machine,
            current_time=self.current_time,
            actions=actions,
            action_mask=np.array(action_mask, dtype=bool),
            features=np.vstack(feature_rows).astype(float),
            invalid_reasons=invalid_reasons,
        )
```

- [ ] **Step 2: Export the new API**

Modify `__init__.py` to this exact content:

```python
from .core import TwoPopulationScheduler
from .rl_environment import (
    CandidatePool,
    DispatchAction,
    DispatchCommitResult,
    ResourceCalendarEnv,
)
from .state import ScheduleState

__all__ = [
    "TwoPopulationScheduler",
    "ScheduleState",
    "ResourceCalendarEnv",
    "DispatchAction",
    "CandidatePool",
    "DispatchCommitResult",
]
```

- [ ] **Step 3: Run candidate pool tests**

Run:

```powershell
python -m pytest tests/test_rl_environment.py::test_candidate_pool_has_fixed_length_padding_and_mask tests/test_rl_environment.py::test_candidate_pool_filters_not_arrived_and_incompatible_machine -v
```

Expected: PASS.

- [ ] **Step 4: Commit**

```powershell
git add rl_environment.py __init__.py tests/test_rl_environment.py
git commit -m "feat: add phase1 candidate pool environment"
```

---

### Task 3: Dry-Run And Commit Semantics

**Files:**
- Modify: `tests/test_rl_environment.py`
- Modify: `rl_environment.py`

- [ ] **Step 1: Add failing dry-run and commit tests**

Append these tests to `tests/test_rl_environment.py`:

```python
def test_dry_run_does_not_mutate_environment_calendars():
    encoder = build_validated_encoder()
    env = ResourceCalendarEnv(encoder, current_time=0.0, top_k=4)
    pool = env.build_candidate_pool(machine=1)
    action = pool.actions[0]

    before_machine_calendar = dict(env.state.machine_calendar)
    before_chamber_calendar = dict(env.state.chamber_calendar)

    dry_run = env.dry_run_action(action)

    assert dry_run.lot_schedule.shape == (1, 5)
    assert dry_run.wafer_schedule.shape[1] == 9
    assert env.state.machine_calendar == before_machine_calendar
    assert env.state.chamber_calendar == before_chamber_calendar


def test_commit_valid_action_updates_state_and_outputs_consistent_schedules():
    encoder = build_validated_encoder()
    env = ResourceCalendarEnv(encoder, current_time=0.0, top_k=4)
    pool = env.build_candidate_pool(machine=1)

    result = env.commit_action_index(machine=1, action_index=0, pool=pool)

    assert result.action.lot == 1
    assert result.action.machine == 1
    assert result.action.ppid == 101
    assert 1 not in env.remaining_lots
    assert env.lot_schedule.shape == (1, 5)
    assert env.wafer_schedule.shape == (6, 9)
    assert env.state.machine_calendar[1] == [
        (float(env.lot_schedule[0, 3]), float(env.lot_schedule[0, 4]))
    ]

    encoder.validate_no_interval_overlap(env.state.machine_calendar, "machine_calendar")
    encoder.validate_no_interval_overlap(env.state.chamber_calendar, "chamber_calendar")


def test_commit_masked_padding_index_raises_value_error():
    encoder = build_validated_encoder()
    env = ResourceCalendarEnv(encoder, current_time=0.0, top_k=4)
    pool = env.build_candidate_pool(machine=1)

    with pytest.raises(ValueError, match="masked action"):
        env.commit_action_index(machine=1, action_index=3, pool=pool)
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```powershell
python -m pytest tests/test_rl_environment.py::test_dry_run_does_not_mutate_environment_calendars tests/test_rl_environment.py::test_commit_valid_action_updates_state_and_outputs_consistent_schedules tests/test_rl_environment.py::test_commit_masked_padding_index_raises_value_error -v
```

Expected: FAIL because `ResourceCalendarEnv.commit_action_index` is not implemented.

- [ ] **Step 3: Implement commit support**

Append these methods inside `ResourceCalendarEnv` in `rl_environment.py`, after `build_candidate_pool()`:

```python
    def _append_schedule_rows(self, lot_schedule, wafer_schedule):
        if lot_schedule.size > 0:
            self.lot_schedule = (
                lot_schedule
                if self.lot_schedule.size == 0
                else np.vstack((self.lot_schedule, lot_schedule))
            )

        if wafer_schedule.size > 0:
            self.wafer_schedule = (
                wafer_schedule
                if self.wafer_schedule.size == 0
                else np.vstack((self.wafer_schedule, wafer_schedule))
            )

    def commit_action_index(self, machine, action_index, pool=None):
        machine = int(machine)
        action_index = int(action_index)
        pool = self.build_candidate_pool(machine) if pool is None else pool

        if pool.machine != machine:
            raise ValueError("pool.machine does not match the requested machine")

        if action_index < 0 or action_index >= len(pool.actions):
            raise ValueError("action_index is outside the candidate pool")

        if not bool(pool.action_mask[action_index]):
            raise ValueError("Cannot commit a masked action")

        action = pool.actions[action_index]
        result = self.dry_run_action(action)
        self.state = result.state

        if not action.is_wait:
            self.remaining_lots.discard(int(action.lot))
            self._append_schedule_rows(result.lot_schedule, result.wafer_schedule)

        return result
```

- [ ] **Step 4: Run all environment tests**

Run:

```powershell
python -m pytest tests/test_rl_environment.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add rl_environment.py tests/test_rl_environment.py
git commit -m "feat: commit masked phase1 dispatch actions"
```

---

### Task 4: Local Demo Script

**Files:**
- Create: `run_phase1_environment_demo.py`

- [ ] **Step 1: Add a runnable local demo**

Create `run_phase1_environment_demo.py`:

```python
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from FABenv import ResourceCalendarEnv
from FABenv.run_helpers import format_objectives
from FABenv.run_small_instance_gantt import build_small_encoder


def choose_first_real_action(pool):
    for index, action in enumerate(pool.actions):
        if pool.action_mask[index] and not action.is_wait:
            return index
    for index, action in enumerate(pool.actions):
        if pool.action_mask[index]:
            return index
    return None


def main():
    encoder = build_small_encoder()
    encoder.validate_problem_definition()
    env = ResourceCalendarEnv(encoder, current_time=0.0, top_k=4)

    while env.remaining_lots:
        committed = False

        for machine in range(1, encoder.num_machines + 1):
            pool = env.build_candidate_pool(machine)
            action_index = choose_first_real_action(pool)

            if action_index is None:
                continue

            action = pool.actions[action_index]
            if action.is_wait:
                continue

            result = env.commit_action_index(machine, action_index, pool=pool)
            print(
                "commit "
                f"lot={result.action.lot} "
                f"machine={result.action.machine} "
                f"ppid={result.action.ppid} "
                f"start={result.lot_schedule[0, 3]:.3f} "
                f"end={result.lot_schedule[0, 4]:.3f}"
            )
            committed = True
            break

        if committed:
            continue

        future_arrivals = [
            float(encoder.arrival_times[lot])
            for lot in env.remaining_lots
            if float(encoder.arrival_times[lot]) > env.current_time
        ]
        if not future_arrivals:
            raise RuntimeError("No valid candidates and no future arrivals remain")
        env.advance_time(min(future_arrivals))

    env.lot_schedule = env.lot_schedule[np.argsort(env.lot_schedule[:, 3])]
    env.wafer_schedule = env.wafer_schedule[np.argsort(env.wafer_schedule[:, 7])]
    encoder.validate_final_schedule_completeness(env.lot_schedule, env.wafer_schedule)
    objectives = encoder.evaluate_objectives(env.lot_schedule, env.wafer_schedule)

    print("lot_schedule=")
    print(env.lot_schedule)
    print("wafer_schedule_rows=", len(env.wafer_schedule))
    print("objectives=", format_objectives(objectives))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the demo**

Run:

```powershell
python run_phase1_environment_demo.py
```

Expected output includes four `commit` lines, `wafer_schedule_rows= 20`, and an `objectives=` line.

- [ ] **Step 3: Commit**

```powershell
git add run_phase1_environment_demo.py
git commit -m "chore: add phase1 environment demo"
```

---

### Task 5: Regression And Existing Script Smoke Check

**Files:**
- Read: `run_small_instance_gantt.py`
- Read: `run_medium_uncertainty_gantt.py`
- Read: `run_large_instance_gantt.py`

- [ ] **Step 1: Run the new tests**

Run:

```powershell
python -m pytest tests/test_rl_environment.py -v
```

Expected: PASS.

- [ ] **Step 2: Run the existing small instance script**

Run:

```powershell
python run_small_instance_gantt.py
```

Expected: command completes and prints:

```text
Small FAB instance completed
```

- [ ] **Step 3: Run the Phase 1 demo**

Run:

```powershell
python run_phase1_environment_demo.py
```

Expected: command completes, validates final schedule completeness, and prints:

```text
wafer_schedule_rows= 20
```

- [ ] **Step 4: Final commit**

If Task 5 required any fixes, commit them:

```powershell
git add rl_environment.py __init__.py tests/test_rl_environment.py run_phase1_environment_demo.py
git commit -m "fix: stabilize phase1 environment checks"
```

If no fixes were needed, do not create an empty commit.

---

## Self-Review

**Spec coverage:** This plan covers the full Stage 1 scope from `报告.md`: two-level calendars, dry-run feasibility, commit consistency, fixed Top-K candidate pools, padding, action mask, filtering by arrival/unfinished status, Machine/PPID feasibility, stage existence, and calendar insertability. PPO, attention, DDT, and reward design are intentionally excluded because they start in later stages.

**Placeholder scan:** The plan contains no deferred implementation markers. Every code-changing task includes concrete code and every verification step includes exact commands and expected results.

**Type consistency:** `DispatchAction`, `CandidatePool`, `DispatchCommitResult`, `ResourceCalendarEnv`, and method names are consistent across tests, implementation, exports, and demo script.

**Risk notes:** The Phase 1 environment uses a deterministic earliest-finish Chamber/Side selector. That is enough for Stage 1 feasibility and mask validation, while learned stage-resource scoring remains outside this phase.
