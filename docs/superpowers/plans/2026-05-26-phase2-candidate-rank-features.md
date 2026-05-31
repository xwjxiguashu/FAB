# Phase2 Candidate Rank Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add four lightweight candidate-pool relative features to Phase 2 SAS observations: `priority_rank_norm`, `due_slack_rank_norm`, `is_best_priority`, and `is_most_urgent_due`.

**Architecture:** Keep the feature enhancement inside `ResourceCalendarEnv.build_candidate_pool(...)`, because that is where the final top-k candidates, wait action, padding rows, mask, and `CandidatePool.features` shape are assembled. Existing candidate scoring, action masks, resource calendar execution, reward logic, PPO logic, and `sas_step(...)` behavior remain unchanged.

**Tech Stack:** Python, NumPy, pytest, existing `FAB_RL/FABenv/rl_environment.py`, existing `problem_instances.build_small_demo_encoder()`.

---

## Scope Check

This plan implements only the four rank/best state-input features documented in `项目方案.md` Section 6.1.1. It does not implement reward changes, observation encoder files, PPO trainer files, policy network changes, or inference fallback. The change is independently testable through `CandidatePool.features` and `ResourceCalendarEnv.feature_names`.

## File Structure

- Modify: `FAB_RL/FABenv/rl_environment.py`
  - Append four names to `ResourceCalendarEnv.feature_names`.
  - Add a small helper, `_apply_candidate_rank_features(actions, feature_rows, mask)`, called at the end of `build_candidate_pool(...)` before constructing the `CandidatePool`.
  - Keep wait rows and padding rows at zero for the four new columns.
- Create: `FAB_RL/FABenv/tests/test_phase2_candidate_rank_features.py`
  - Test the feature names exist in order.
  - Test real valid candidates receive normalized priority and due-slack ranks.
  - Test best-priority and most-urgent-due flags.
  - Test wait and padding rows keep rank/best features at zero.
- Verify: `python -m pytest FAB_RL/FABenv/tests/test_phase2_candidate_rank_features.py -v`
- Verify: `python -m pytest FAB_RL/FABenv/tests -v`

---

### Task 1: Add Feature Name Contract Tests

**Files:**
- Create: `FAB_RL/FABenv/tests/test_phase2_candidate_rank_features.py`
- Modify: none

- [ ] **Step 1: Write the failing test**

Create `FAB_RL/FABenv/tests/test_phase2_candidate_rank_features.py` with:

```python
import sys
from pathlib import Path


PHASE1_DIR = Path(__file__).resolve().parents[1]
if str(PHASE1_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE1_DIR))


from rl_environment import ResourceCalendarEnv


def _feature_index(name):
    return ResourceCalendarEnv.feature_names.index(name)


def test_candidate_rank_feature_names_are_appended_to_existing_features():
    assert ResourceCalendarEnv.feature_names[-4:] == (
        "priority_rank_norm",
        "due_slack_rank_norm",
        "is_best_priority",
        "is_most_urgent_due",
    )
```

- [ ] **Step 2: Run the test to verify RED**

Run:

```bash
python -m pytest FAB_RL/FABenv/tests/test_phase2_candidate_rank_features.py::test_candidate_rank_feature_names_are_appended_to_existing_features -v
```

Expected: FAIL because the four feature names do not exist yet.

- [ ] **Step 3: Append feature names**

In `FAB_RL/FABenv/rl_environment.py`, update `ResourceCalendarEnv.feature_names` from:

```python
        "priority",
        "due_slack",
    )
```

to:

```python
        "priority",
        "due_slack",
        "priority_rank_norm",
        "due_slack_rank_norm",
        "is_best_priority",
        "is_most_urgent_due",
    )
```

- [ ] **Step 4: Run the test to verify GREEN**

Run:

```bash
python -m pytest FAB_RL/FABenv/tests/test_phase2_candidate_rank_features.py::test_candidate_rank_feature_names_are_appended_to_existing_features -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add FAB_RL/FABenv/rl_environment.py FAB_RL/FABenv/tests/test_phase2_candidate_rank_features.py
git commit -m "test: define phase2 candidate rank feature contract"
```

---

### Task 2: Populate Rank Features for Real Candidates

**Files:**
- Modify: `FAB_RL/FABenv/tests/test_phase2_candidate_rank_features.py`
- Modify: `FAB_RL/FABenv/rl_environment.py`

- [ ] **Step 1: Write the failing rank-value test**

Append this test to `FAB_RL/FABenv/tests/test_phase2_candidate_rank_features.py`:

```python
from problem_instances import build_small_demo_encoder


def test_real_candidates_receive_priority_and_due_slack_ranks():
    encoder = build_small_demo_encoder()
    env = ResourceCalendarEnv(encoder, top_k=8)
    pool = env.build_candidate_pool(machine=1)

    priority_idx = _feature_index("priority")
    due_slack_idx = _feature_index("due_slack")
    priority_rank_idx = _feature_index("priority_rank_norm")
    due_slack_rank_idx = _feature_index("due_slack_rank_norm")

    real_indices = [
        index
        for index, action in enumerate(pool.actions)
        if bool(pool.action_mask[index]) and not action.is_wait and not action.is_padding
    ]
    assert len(real_indices) >= 2

    priorities = {index: pool.features[index, priority_idx] for index in real_indices}
    due_slacks = {index: pool.features[index, due_slack_idx] for index in real_indices}
    expected_priority_order = sorted(real_indices, key=lambda index: (-priorities[index], index))
    expected_due_order = sorted(real_indices, key=lambda index: (due_slacks[index], index))
    n = len(real_indices)

    for rank, index in enumerate(expected_priority_order, start=1):
        expected = (n - rank + 1) / n
        assert pool.features[index, priority_rank_idx] == expected

    for rank, index in enumerate(expected_due_order, start=1):
        expected = (n - rank + 1) / n
        assert pool.features[index, due_slack_rank_idx] == expected
```

- [ ] **Step 2: Run the test to verify RED**

Run:

```bash
python -m pytest FAB_RL/FABenv/tests/test_phase2_candidate_rank_features.py::test_real_candidates_receive_priority_and_due_slack_ranks -v
```

Expected: FAIL because the new feature columns remain zero.

- [ ] **Step 3: Implement rank population helper**

In `FAB_RL/FABenv/rl_environment.py`, add this method inside `ResourceCalendarEnv` after `build_candidate_pool(...)` and before the `# --- Public interfaces` comment:

```python
    def _apply_candidate_rank_features(self, actions, feature_rows, mask):
        if not feature_rows:
            return feature_rows

        priority_idx = self.feature_names.index("priority")
        due_slack_idx = self.feature_names.index("due_slack")
        priority_rank_idx = self.feature_names.index("priority_rank_norm")
        due_slack_rank_idx = self.feature_names.index("due_slack_rank_norm")
        best_priority_idx = self.feature_names.index("is_best_priority")
        urgent_due_idx = self.feature_names.index("is_most_urgent_due")

        real_indices = [
            index
            for index, action in enumerate(actions)
            if bool(mask[index])
            and not self._coerce_action(action).is_wait
            and not self._coerce_action(action).is_padding
        ]
        if not real_indices:
            return feature_rows

        n_real = len(real_indices)
        priority_order = sorted(
            real_indices,
            key=lambda index: (-feature_rows[index][priority_idx], index),
        )
        due_order = sorted(
            real_indices,
            key=lambda index: (feature_rows[index][due_slack_idx], index),
        )

        for rank, index in enumerate(priority_order, start=1):
            feature_rows[index][priority_rank_idx] = (n_real - rank + 1) / n_real
        for rank, index in enumerate(due_order, start=1):
            feature_rows[index][due_slack_rank_idx] = (n_real - rank + 1) / n_real

        max_priority = max(feature_rows[index][priority_idx] for index in real_indices)
        min_due_slack = min(feature_rows[index][due_slack_idx] for index in real_indices)
        for index in real_indices:
            feature_rows[index][best_priority_idx] = float(
                feature_rows[index][priority_idx] == max_priority
            )
            feature_rows[index][urgent_due_idx] = float(
                feature_rows[index][due_slack_idx] == min_due_slack
            )

        return feature_rows
```

- [ ] **Step 4: Call the helper before returning `CandidatePool`**

In `build_candidate_pool(...)`, insert this line immediately before `return CandidatePool(`:

```python
        feature_rows = self._apply_candidate_rank_features(actions, feature_rows, mask)
```

- [ ] **Step 5: Run the rank-value test**

Run:

```bash
python -m pytest FAB_RL/FABenv/tests/test_phase2_candidate_rank_features.py::test_real_candidates_receive_priority_and_due_slack_ranks -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
git add FAB_RL/FABenv/rl_environment.py FAB_RL/FABenv/tests/test_phase2_candidate_rank_features.py
git commit -m "feat: add candidate priority and due rank features"
```

---

### Task 3: Mark Best Priority and Most Urgent Due Candidates

**Files:**
- Modify: `FAB_RL/FABenv/tests/test_phase2_candidate_rank_features.py`
- Modify: `FAB_RL/FABenv/rl_environment.py`

- [ ] **Step 1: Write the best-marker test**

Append this test to `FAB_RL/FABenv/tests/test_phase2_candidate_rank_features.py`:

```python
def test_best_priority_and_most_urgent_due_flags_match_pool_extremes():
    encoder = build_small_demo_encoder()
    env = ResourceCalendarEnv(encoder, top_k=8)
    pool = env.build_candidate_pool(machine=1)

    priority_idx = _feature_index("priority")
    due_slack_idx = _feature_index("due_slack")
    best_priority_idx = _feature_index("is_best_priority")
    urgent_due_idx = _feature_index("is_most_urgent_due")

    real_indices = [
        index
        for index, action in enumerate(pool.actions)
        if bool(pool.action_mask[index]) and not action.is_wait and not action.is_padding
    ]
    assert real_indices

    max_priority = max(pool.features[index, priority_idx] for index in real_indices)
    min_due_slack = min(pool.features[index, due_slack_idx] for index in real_indices)

    for index in real_indices:
        assert pool.features[index, best_priority_idx] == float(
            pool.features[index, priority_idx] == max_priority
        )
        assert pool.features[index, urgent_due_idx] == float(
            pool.features[index, due_slack_idx] == min_due_slack
        )
```

- [ ] **Step 2: Run the marker test**

Run:

```bash
python -m pytest FAB_RL/FABenv/tests/test_phase2_candidate_rank_features.py::test_best_priority_and_most_urgent_due_flags_match_pool_extremes -v
```

Expected: PASS if Task 2 helper already implemented the marker fields. If it fails, compare the helper code in Task 2 Step 3 and fix the marker assignment exactly as shown there.

- [ ] **Step 3: Run all candidate rank tests**

Run:

```bash
python -m pytest FAB_RL/FABenv/tests/test_phase2_candidate_rank_features.py -v
```

Expected: PASS.

- [ ] **Step 4: Commit**

Run:

```bash
git add FAB_RL/FABenv/rl_environment.py FAB_RL/FABenv/tests/test_phase2_candidate_rank_features.py
git commit -m "test: cover candidate best marker features"
```

---

### Task 4: Keep Wait and Padding Rank Features at Zero

**Files:**
- Modify: `FAB_RL/FABenv/tests/test_phase2_candidate_rank_features.py`
- Modify: `FAB_RL/FABenv/rl_environment.py`

- [ ] **Step 1: Write the wait and padding zero test**

Append this test to `FAB_RL/FABenv/tests/test_phase2_candidate_rank_features.py`:

```python
def test_wait_and_padding_rows_keep_rank_features_zero():
    encoder = build_small_demo_encoder()
    env = ResourceCalendarEnv(encoder, top_k=8)
    pool = env.build_candidate_pool(machine=1)

    rank_indices = [
        _feature_index("priority_rank_norm"),
        _feature_index("due_slack_rank_norm"),
        _feature_index("is_best_priority"),
        _feature_index("is_most_urgent_due"),
    ]

    non_real_indices = [
        index
        for index, action in enumerate(pool.actions)
        if action.is_wait or action.is_padding or not bool(pool.action_mask[index])
    ]
    assert non_real_indices

    for index in non_real_indices:
        assert pool.features[index, rank_indices].tolist() == [0.0, 0.0, 0.0, 0.0]
```

- [ ] **Step 2: Run the wait and padding test**

Run:

```bash
python -m pytest FAB_RL/FABenv/tests/test_phase2_candidate_rank_features.py::test_wait_and_padding_rows_keep_rank_features_zero -v
```

Expected: PASS if Task 2 helper filters out wait and padding rows. If it fails, update `_apply_candidate_rank_features(...)` so `real_indices` excludes `action.is_wait`, `action.is_padding`, and masked rows exactly as shown in Task 2 Step 3.

- [ ] **Step 3: Run all candidate rank tests**

Run:

```bash
python -m pytest FAB_RL/FABenv/tests/test_phase2_candidate_rank_features.py -v
```

Expected: PASS.

- [ ] **Step 4: Commit**

Run:

```bash
git add FAB_RL/FABenv/rl_environment.py FAB_RL/FABenv/tests/test_phase2_candidate_rank_features.py
git commit -m "test: keep non-real candidate rank features zero"
```

---

### Task 5: Run Regression and Documentation Alignment Checks

**Files:**
- Verify: `FAB_RL/FABenv/rl_environment.py`
- Verify: `FAB_RL/FABenv/tests/test_phase2_candidate_rank_features.py`
- Verify: `项目方案.md`

- [ ] **Step 1: Run candidate rank feature tests**

Run:

```bash
python -m pytest FAB_RL/FABenv/tests/test_phase2_candidate_rank_features.py -v
```

Expected: all tests PASS.

- [ ] **Step 2: Run existing FABenv tests**

Run:

```bash
python -m pytest FAB_RL/FABenv/tests -v
```

Expected: all tests PASS.

- [ ] **Step 3: Verify code and documentation agree on the four feature names**

Run:

```bash
python -c "from pathlib import Path; doc=Path('项目方案.md').read_text(encoding='utf-8'); code=Path('FAB_RL/FABenv/rl_environment.py').read_text(encoding='utf-8'); fields=['priority_rank_norm','due_slack_rank_norm','is_best_priority','is_most_urgent_due']; missing=[f for f in fields if f not in doc or f not in code]; assert not missing, missing; print('candidate rank feature alignment passed')"
```

Expected output:

```text
candidate rank feature alignment passed
```

- [ ] **Step 4: Commit final plan and documentation updates**

Run:

```bash
git status --short
```

If only expected files are modified, run:

```bash
git add FAB_RL/FABenv/rl_environment.py FAB_RL/FABenv/tests/test_phase2_candidate_rank_features.py 项目方案.md docs/superpowers/plans/2026-05-26-phase2-candidate-rank-features.md
git commit -m "docs: plan phase2 candidate rank features"
```

Expected: commit succeeds, or git reports nothing to commit if earlier tasks already committed every change.

---

## Self-Review

Spec coverage:
- `priority_rank_norm` is covered by Task 1 and Task 2.
- `due_slack_rank_norm` is covered by Task 1 and Task 2.
- `is_best_priority` is covered by Task 1 and Task 3.
- `is_most_urgent_due` is covered by Task 1 and Task 3.
- Wait and padding rows staying zero are covered by Task 4.
- Documentation-code alignment and regression checks are covered by Task 5.

Placeholder scan:
- No task contains unresolved placeholder text.
- Every code-changing step contains exact code or exact replacement instructions.
- Every test step contains an exact command and expected result.

Type consistency:
- `ResourceCalendarEnv.feature_names`, `build_candidate_pool(...)`, `CandidatePool.features`, `pool.action_mask`, and `build_small_demo_encoder()` match existing project symbols.
- The four new feature names match `项目方案.md` Section 6.1.1 exactly.
