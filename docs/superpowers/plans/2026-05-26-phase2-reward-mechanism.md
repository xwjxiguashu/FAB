# Phase2 Reward Mechanism Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the documented three-stage Phase 2 SAS reward mechanism with complete reward component logging.

**Architecture:** Keep reward calculation inside `FAB_RL/FABenv/rl_environment.py` so `sas_step(...)` remains the single owner of resource-calendar execution, state transition, and reward assembly. Add a component-level helper used by `compute_sas_reward(...)`, mutate `info` with the full reward decomposition, and keep default behavior at R0 by leaving shaping and terminal reward disabled unless explicitly configured.

**Tech Stack:** Python dataclasses, NumPy, pytest, existing `ResourceCalendarEnv.sas_step(...)`, existing `RewardConfig`, existing `compute_sas_reward(...)`.

---

## Scope Check

This plan implements only the reward mechanism described in `项目方案.md` Section 8. It does not implement the full Phase 2 PPO driver, observation encoder, policy, rollout buffer, or inference fallback. The output is independently testable through reward unit tests and existing Phase 1 regression tests.

## File Structure

- Modify: `FAB_RL/FABenv/rl_environment.py`
  - Extend `RewardConfig` with R1 and R2 weights.
  - Add `compute_sas_reward_components(info, config=None)`.
  - Update `compute_sas_reward(info, config=None)` to write component fields into `info` and return `reward_total`.
  - Update `sas_step(...)` info dictionaries to initialize the full reward decomposition and stop overwriting `reward_execute` / `reward_wait` after reward computation.
  - Change priority penalty assembly to use current candidate-pool relative priority, not completed-lot priority.
- Create: `FAB_RL/FABenv/tests/test_phase2_reward.py`
  - Test R0 execution outcomes.
  - Test R1 shaping decomposition.
  - Test R2 terminal reward decomposition.
  - Test `sas_step(...)` exposes all reward fields.
- Verify: `python -m pytest FAB_RL/FABenv/tests/test_phase2_reward.py -v`
- Verify: `python -m pytest FAB_RL/FABenv/tests -v`

---

### Task 1: Add R0 Reward Component Tests

**Files:**
- Create: `FAB_RL/FABenv/tests/test_phase2_reward.py`
- Modify: none

- [ ] **Step 1: Write the failing tests**

Create `FAB_RL/FABenv/tests/test_phase2_reward.py` with:

```python
import sys
from pathlib import Path


PHASE1_DIR = Path(__file__).resolve().parents[1]
if str(PHASE1_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE1_DIR))


from rl_environment import RewardConfig, compute_sas_reward


def test_r0_success_reward_records_execute_and_total():
    info = {
        "insertion_success": True,
        "insertion_failed": False,
        "mask_invalid": False,
        "wait_or_noop": False,
        "selected_lot_start": 10.0,
        "selected_lot_end": 20.0,
        "selected_lot_process_time": 10.0,
        "current_time": 10.0,
        "due_date": 100.0,
        "new_qtime_violation": 0.0,
        "priority_rank_penalty": 0.0,
    }

    reward = compute_sas_reward(info, RewardConfig())

    assert reward == 0.20
    assert info["reward_execute"] == 0.20
    assert info["reward_wait"] == 0.0
    assert info["reward_tardy"] == 0.0
    assert info["reward_qtime"] == 0.0
    assert info["reward_priority"] == 0.0
    assert info["reward_progress"] == 0.0
    assert info["reward_shape"] == 0.0
    assert info["reward_terminal"] == 0.0
    assert info["reward_total"] == 0.20


def test_r0_wait_reward_records_wait_only():
    info = {
        "insertion_success": False,
        "insertion_failed": False,
        "mask_invalid": False,
        "wait_or_noop": True,
    }

    reward = compute_sas_reward(info, RewardConfig())

    assert reward == -0.02
    assert info["reward_execute"] == 0.0
    assert info["reward_wait"] == -0.02
    assert info["reward_shape"] == 0.0
    assert info["reward_terminal"] == 0.0
    assert info["reward_total"] == -0.02


def test_r0_mask_invalid_reward_records_execute_penalty():
    info = {
        "insertion_success": False,
        "insertion_failed": False,
        "mask_invalid": True,
        "wait_or_noop": False,
    }

    reward = compute_sas_reward(info, RewardConfig())

    assert reward == -0.50
    assert info["reward_execute"] == -0.50
    assert info["reward_wait"] == 0.0
    assert info["reward_total"] == -0.50
```

- [ ] **Step 2: Run the tests to verify RED**

Run:

```bash
python -m pytest FAB_RL/FABenv/tests/test_phase2_reward.py -v
```

Expected: FAIL because `compute_sas_reward(...)` currently returns totals but does not populate `reward_tardy`, `reward_qtime`, `reward_priority`, `reward_progress`, or `reward_total`.

- [ ] **Step 3: Implement minimal reward component writing**

In `FAB_RL/FABenv/rl_environment.py`, replace `RewardConfig` and `compute_sas_reward(...)` with:

```python
@dataclass
class RewardConfig:
    insert_success_reward: float = 0.20
    insert_fail_penalty: float = -0.40
    mask_invalid_penalty: float = -0.50
    wait_penalty: float = -0.02
    tardy_weight: float = -0.05
    qtime_weight: float = -0.08
    priority_weight: float = -0.03
    progress_weight: float = 0.01
    terminal_tardy_lot_weight: float = -0.20
    terminal_total_tardiness_weight: float = -0.10
    terminal_qtime_weight: float = -0.15
    terminal_utilization_weight: float = 0.05
    terminal_priority_weight: float = -0.05
    reward_clip_min: float = -1.0
    reward_clip_max: float = 1.0
    use_light_shaping: bool = False
    use_terminal_reward: bool = False


def _empty_reward_components():
    return {
        "reward_execute": 0.0,
        "reward_wait": 0.0,
        "reward_tardy": 0.0,
        "reward_qtime": 0.0,
        "reward_priority": 0.0,
        "reward_progress": 0.0,
        "reward_shape": 0.0,
        "reward_terminal": 0.0,
        "reward_total": 0.0,
    }


def compute_sas_reward_components(info, config=None):
    if config is None:
        config = RewardConfig()

    components = _empty_reward_components()

    if info.get("mask_invalid"):
        components["reward_execute"] = config.mask_invalid_penalty
    elif info.get("wait_or_noop"):
        components["reward_wait"] = config.wait_penalty
    elif info.get("insertion_failed"):
        components["reward_execute"] = config.insert_fail_penalty
    elif info.get("insertion_success"):
        components["reward_execute"] = config.insert_success_reward

    components["reward_shape"] = (
        components["reward_tardy"]
        + components["reward_qtime"]
        + components["reward_priority"]
        + components["reward_progress"]
    )
    raw_total = (
        components["reward_execute"]
        + components["reward_wait"]
        + components["reward_shape"]
        + components["reward_terminal"]
    )
    components["reward_total"] = float(
        np.clip(raw_total, config.reward_clip_min, config.reward_clip_max)
    )
    return components


def compute_sas_reward(info, config=None):
    components = compute_sas_reward_components(info, config)
    info.update(components)
    return components["reward_total"]
```

- [ ] **Step 4: Run the tests to verify GREEN**

Run:

```bash
python -m pytest FAB_RL/FABenv/tests/test_phase2_reward.py -v
```

Expected: PASS for the three R0 tests.

- [ ] **Step 5: Commit**

Run:

```bash
git add FAB_RL/FABenv/rl_environment.py FAB_RL/FABenv/tests/test_phase2_reward.py
git commit -m "test: cover phase2 r0 reward components"
```

---

### Task 2: Add R1 Light Shaping Tests

**Files:**
- Modify: `FAB_RL/FABenv/tests/test_phase2_reward.py`
- Modify: `FAB_RL/FABenv/rl_environment.py`

- [ ] **Step 1: Write the failing shaping test**

Append this test to `FAB_RL/FABenv/tests/test_phase2_reward.py`:

```python
def test_r1_light_shaping_records_each_component():
    info = {
        "insertion_success": True,
        "insertion_failed": False,
        "mask_invalid": False,
        "wait_or_noop": False,
        "selected_lot_start": 10.0,
        "selected_lot_end": 30.0,
        "selected_lot_process_time": 20.0,
        "current_time": 10.0,
        "due_date": 20.0,
        "new_qtime_violation": 5.0,
        "priority_rank_penalty": 2.0,
    }
    config = RewardConfig(use_light_shaping=True)

    reward = compute_sas_reward(info, config)

    assert info["reward_execute"] == 0.20
    assert info["reward_tardy"] == -0.025
    assert info["reward_qtime"] == -0.02
    assert info["reward_priority"] == -0.06
    assert info["reward_progress"] == 0.01
    assert info["reward_shape"] == -0.095
    assert reward == 0.105
    assert info["reward_total"] == 0.105
```

- [ ] **Step 2: Run the test to verify RED**

Run:

```bash
python -m pytest FAB_RL/FABenv/tests/test_phase2_reward.py::test_r1_light_shaping_records_each_component -v
```

Expected: FAIL because `compute_sas_reward_components(...)` does not yet apply light shaping.

- [ ] **Step 3: Implement light shaping**

Inside `compute_sas_reward_components(...)`, replace the `elif info.get("insertion_success"):` branch with:

```python
    elif info.get("insertion_success"):
        components["reward_execute"] = config.insert_success_reward

        if config.use_light_shaping:
            horizon = max(
                info.get("selected_lot_end", 0.0) - info.get("current_time", 0.0),
                1e-9,
            )
            due_date = info.get("due_date", np.inf)
            tardy_norm = float(
                np.clip(
                    max(0.0, info.get("selected_lot_end", 0.0) - due_date) / horizon,
                    0.0,
                    1.0,
                )
            )
            process_time = max(info.get("selected_lot_process_time", 1.0), 1e-9)
            qtime_norm = float(
                np.clip(
                    info.get("new_qtime_violation", 0.0) / process_time,
                    0.0,
                    1.0,
                )
            )
            progress_norm = float(
                np.clip(
                    (
                        info.get("selected_lot_end", 0.0)
                        - info.get("selected_lot_start", 0.0)
                    )
                    / horizon,
                    0.0,
                    1.0,
                )
            )
            components["reward_tardy"] = config.tardy_weight * tardy_norm
            components["reward_qtime"] = config.qtime_weight * qtime_norm
            components["reward_priority"] = (
                config.priority_weight * info.get("priority_rank_penalty", 0.0)
            )
            components["reward_progress"] = config.progress_weight * progress_norm
```

- [ ] **Step 4: Run all reward tests**

Run:

```bash
python -m pytest FAB_RL/FABenv/tests/test_phase2_reward.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add FAB_RL/FABenv/rl_environment.py FAB_RL/FABenv/tests/test_phase2_reward.py
git commit -m "feat: add phase2 light reward shaping"
```

---

### Task 3: Add R2 Terminal Reward Tests

**Files:**
- Modify: `FAB_RL/FABenv/tests/test_phase2_reward.py`
- Modify: `FAB_RL/FABenv/rl_environment.py`

- [ ] **Step 1: Write the failing terminal reward test**

Append this test to `FAB_RL/FABenv/tests/test_phase2_reward.py`:

```python
def test_r2_terminal_reward_only_applies_when_episode_done():
    info = {
        "insertion_success": False,
        "insertion_failed": False,
        "mask_invalid": False,
        "wait_or_noop": False,
        "episode_done": True,
        "tardy_lot_count_norm": 0.5,
        "total_tardiness_norm": 0.25,
        "qtime_violation_count_norm": 0.2,
        "machine_utilization_norm": 0.8,
        "priority_violation_norm": 0.4,
    }
    config = RewardConfig(use_terminal_reward=True)

    reward = compute_sas_reward(info, config)

    assert info["reward_terminal"] == -0.135
    assert reward == -0.135
    assert info["reward_total"] == -0.135


def test_r2_terminal_reward_is_zero_before_episode_done():
    info = {
        "insertion_success": False,
        "insertion_failed": False,
        "mask_invalid": False,
        "wait_or_noop": False,
        "episode_done": False,
        "tardy_lot_count_norm": 1.0,
        "total_tardiness_norm": 1.0,
        "qtime_violation_count_norm": 1.0,
        "machine_utilization_norm": 1.0,
        "priority_violation_norm": 1.0,
    }
    config = RewardConfig(use_terminal_reward=True)

    reward = compute_sas_reward(info, config)

    assert info["reward_terminal"] == 0.0
    assert reward == 0.0
    assert info["reward_total"] == 0.0
```

- [ ] **Step 2: Run the tests to verify RED**

Run:

```bash
python -m pytest FAB_RL/FABenv/tests/test_phase2_reward.py::test_r2_terminal_reward_only_applies_when_episode_done FAB_RL/FABenv/tests/test_phase2_reward.py::test_r2_terminal_reward_is_zero_before_episode_done -v
```

Expected: FAIL because terminal reward is not implemented.

- [ ] **Step 3: Implement terminal reward**

After the execution-result branch in `compute_sas_reward_components(...)`, before computing `reward_shape`, insert:

```python
    if config.use_terminal_reward and info.get("episode_done"):
        components["reward_terminal"] = (
            config.terminal_tardy_lot_weight
            * info.get("tardy_lot_count_norm", 0.0)
            + config.terminal_total_tardiness_weight
            * info.get("total_tardiness_norm", 0.0)
            + config.terminal_qtime_weight
            * info.get("qtime_violation_count_norm", 0.0)
            + config.terminal_utilization_weight
            * info.get("machine_utilization_norm", 0.0)
            + config.terminal_priority_weight
            * info.get("priority_violation_norm", 0.0)
        )
```

- [ ] **Step 4: Run all reward tests**

Run:

```bash
python -m pytest FAB_RL/FABenv/tests/test_phase2_reward.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add FAB_RL/FABenv/rl_environment.py FAB_RL/FABenv/tests/test_phase2_reward.py
git commit -m "feat: add phase2 terminal reward"
```

---

### Task 4: Update `sas_step(...)` Reward Info Assembly

**Files:**
- Modify: `FAB_RL/FABenv/tests/test_phase2_reward.py`
- Modify: `FAB_RL/FABenv/rl_environment.py`

- [ ] **Step 1: Write the failing `sas_step(...)` info test**

Append this test to `FAB_RL/FABenv/tests/test_phase2_reward.py`:

```python
from problem_instances import build_small_demo_encoder
from rl_environment import ResourceCalendarEnv


def test_sas_step_info_contains_complete_reward_decomposition():
    encoder = build_small_demo_encoder()
    env = ResourceCalendarEnv(encoder)
    machine = 1
    pool = env.build_candidate_pool(machine)
    valid_indices = [
        index
        for index, is_valid in enumerate(pool.action_mask)
        if bool(is_valid) and not pool.actions[index].is_wait
    ]
    assert valid_indices

    result = env.sas_step(machine, valid_indices[0], pool=pool, reward_config=RewardConfig())

    assert result.committed is True
    for key in (
        "reward_execute",
        "reward_wait",
        "reward_tardy",
        "reward_qtime",
        "reward_priority",
        "reward_progress",
        "reward_shape",
        "reward_terminal",
        "reward_total",
    ):
        assert key in result.info
    assert result.reward == result.info["reward_total"]
```

- [ ] **Step 2: Run the test to verify RED**

Run:

```bash
python -m pytest FAB_RL/FABenv/tests/test_phase2_reward.py::test_sas_step_info_contains_complete_reward_decomposition -v
```

Expected: FAIL if any `sas_step(...)` branch lacks the new reward fields or if branch-specific post-processing overwrites only partial fields.

- [ ] **Step 3: Add a reusable blank reward dict helper for `sas_step(...)`**

In `FAB_RL/FABenv/rl_environment.py`, keep `_empty_reward_components()` from Task 1 and use it when building each `info` dictionary. For every `info = { ... }` inside `sas_step(...)`, remove these keys:

```python
"reward_execute": 0.0,
"reward_wait": 0.0,
"reward_shape": 0.0,
"reward_terminal": 0.0,
```

Then immediately after each info dictionary literal, add:

```python
            info.update(_empty_reward_components())
```

Use the same indentation as the branch.

- [ ] **Step 4: Stop overwriting component fields after reward computation**

In `sas_step(...)`, replace every pattern like this:

```python
            reward = compute_sas_reward(info, reward_config)
            info["reward_execute"] = reward
```

or:

```python
            reward = compute_sas_reward(info, reward_config)
            info["reward_wait"] = reward
```

with:

```python
            reward = compute_sas_reward(info, reward_config)
```

because `compute_sas_reward(...)` now writes all component fields.

- [ ] **Step 5: Run the `sas_step(...)` test**

Run:

```bash
python -m pytest FAB_RL/FABenv/tests/test_phase2_reward.py::test_sas_step_info_contains_complete_reward_decomposition -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
git add FAB_RL/FABenv/rl_environment.py FAB_RL/FABenv/tests/test_phase2_reward.py
git commit -m "feat: expose phase2 reward components from sas step"
```

---

### Task 5: Use Candidate-Pool Relative Priority Penalty

**Files:**
- Modify: `FAB_RL/FABenv/tests/test_phase2_reward.py`
- Modify: `FAB_RL/FABenv/rl_environment.py`

- [ ] **Step 1: Write the failing priority penalty test**

Append this test to `FAB_RL/FABenv/tests/test_phase2_reward.py`:

```python
def test_priority_penalty_uses_current_candidate_pool_not_completed_lots():
    encoder = build_small_demo_encoder()
    env = ResourceCalendarEnv(encoder)
    machine = 1
    pool = env.build_candidate_pool(machine)
    real_indices = [
        index
        for index, action in enumerate(pool.actions)
        if bool(pool.action_mask[index]) and not action.is_wait and not action.is_padding
    ]
    assert len(real_indices) >= 2

    selected_index = min(
        real_indices,
        key=lambda index: encoder.priorities.get(int(pool.actions[index].lot), 0.0),
    )
    selected_priority = encoder.priorities.get(int(pool.actions[selected_index].lot), 0.0)
    expected_penalty = sum(
        max(0.0, encoder.priorities.get(int(pool.actions[index].lot), 0.0) - selected_priority)
        for index in real_indices
        if index != selected_index
    )

    result = env.sas_step(
        machine,
        selected_index,
        pool=pool,
        reward_config=RewardConfig(use_light_shaping=True),
    )

    assert result.info["priority_rank_penalty"] == expected_penalty
    assert result.info["reward_priority"] == RewardConfig().priority_weight * expected_penalty
```

- [ ] **Step 2: Run the test to verify RED**

Run:

```bash
python -m pytest FAB_RL/FABenv/tests/test_phase2_reward.py::test_priority_penalty_uses_current_candidate_pool_not_completed_lots -v
```

Expected: FAIL because existing priority penalty uses completed lots instead of current candidate-pool alternatives.

- [ ] **Step 3: Implement candidate-pool relative priority**

In `sas_step(...)`, replace the current block:

```python
        priority_rank_penalty = 0.0
        if hasattr(self.encoder, "priorities"):
            lot_priority = float(self.encoder.priorities.get(int(action.lot), 0.0))
            for completed in self.completed_lots - {int(action.lot)}:
                other_priority = float(self.encoder.priorities.get(completed, 0.0))
                priority_rank_penalty += max(0.0, other_priority - lot_priority)
```

with:

```python
        priority_rank_penalty = 0.0
        if hasattr(self.encoder, "priorities"):
            lot_priority = float(self.encoder.priorities.get(int(action.lot), 0.0))
            for other_action, is_valid in zip(pool.actions, pool.action_mask):
                other_action = self._coerce_action(other_action)
                if (
                    not bool(is_valid)
                    or other_action.is_padding
                    or other_action.is_wait
                    or int(other_action.lot) == int(action.lot)
                ):
                    continue
                other_priority = float(
                    self.encoder.priorities.get(int(other_action.lot), 0.0)
                )
                priority_rank_penalty += max(0.0, other_priority - lot_priority)
```

- [ ] **Step 4: Run the priority test**

Run:

```bash
python -m pytest FAB_RL/FABenv/tests/test_phase2_reward.py::test_priority_penalty_uses_current_candidate_pool_not_completed_lots -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add FAB_RL/FABenv/rl_environment.py FAB_RL/FABenv/tests/test_phase2_reward.py
git commit -m "fix: compute priority reward from current candidate pool"
```

---

### Task 6: Run Regression Tests and Verify Documentation Alignment

**Files:**
- Verify: `项目方案.md`
- Verify: `FAB_RL/FABenv/rl_environment.py`
- Verify: `FAB_RL/FABenv/tests/test_phase2_reward.py`

- [ ] **Step 1: Run reward tests**

Run:

```bash
python -m pytest FAB_RL/FABenv/tests/test_phase2_reward.py -v
```

Expected: all tests PASS.

- [ ] **Step 2: Run existing FABenv tests**

Run:

```bash
python -m pytest FAB_RL/FABenv/tests -v
```

Expected: all tests PASS.

- [ ] **Step 3: Verify documentation and code mention the same reward fields**

Run:

```bash
python -c "from pathlib import Path; doc=Path('项目方案.md').read_text(encoding='utf-8'); code=Path('FAB_RL/FABenv/rl_environment.py').read_text(encoding='utf-8'); fields=['reward_execute','reward_wait','reward_tardy','reward_qtime','reward_priority','reward_progress','reward_shape','reward_terminal','reward_total']; missing=[f for f in fields if f not in doc or f not in code]; assert not missing, missing; print('reward field alignment passed')"
```

Expected output:

```text
reward field alignment passed
```

- [ ] **Step 4: Commit final verification updates if any files changed**

Run:

```bash
git status --short
```

If only expected files are modified, run:

```bash
git add FAB_RL/FABenv/rl_environment.py FAB_RL/FABenv/tests/test_phase2_reward.py 项目方案.md docs/superpowers/plans/2026-05-26-phase2-reward-mechanism.md
git commit -m "docs: align phase2 reward mechanism plan"
```

Expected: commit succeeds, or git reports nothing to commit if every task was already committed.

---

## Self-Review

Spec coverage:
- R0 legal-action reward is covered by Task 1.
- R1 light shaping and per-component reward logging are covered by Task 2.
- R2 terminal reward is covered by Task 3.
- `sas_step(...)` info decomposition is covered by Task 4.
- Candidate-pool relative priority penalty is covered by Task 5.
- Regression and documentation alignment checks are covered by Task 6.

Placeholder scan:
- No task contains placeholders such as TBD, TODO, or "implement later".
- Every code-changing step includes concrete code.
- Every verification step includes an exact command and expected result.

Type consistency:
- `RewardConfig`, `compute_sas_reward(...)`, `compute_sas_reward_components(...)`, `ResourceCalendarEnv`, and `build_small_demo_encoder()` names match existing or explicitly introduced symbols.
- Reward field names match `项目方案.md` Section 8.5.
