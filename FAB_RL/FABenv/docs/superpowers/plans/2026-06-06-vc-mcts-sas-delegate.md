# VC-MCTS SAS Delegate Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect VC-MCTS to SAS through a dispatch delegate so VC-MCTS decides reservation/no-op timing while rule or SAS policy code chooses the concrete dispatch action.

**Architecture:** Add a small `dispatch_delegate.py` boundary with `RuleDispatchDelegate` and `SASPolicyDispatchDelegate`. VC-MCTS root actions will use `delegate_dispatch` instead of enumerating TopK dispatch candidates when a delegate is enabled, while reservation rollouts and episode execution call the same delegate interface. The default rule delegate preserves current FIFO behavior before a checkpoint-backed SAS delegate is enabled.

**Tech Stack:** Python, pytest, PyTorch policy checkpoints via `model_checkpoint.load_policy_checkpoint`, existing `Phase2EpisodeDriver`, `Phase2ObservationEncoder`, `ResourceCalendarEnv`, VC-MCTS reservation modules.

---

## File Structure

- Create `FAB_RL/FABenv/dispatch_delegate.py`
  - Owns the dispatch delegate interface.
  - Provides `RuleDispatchDelegate`, `SASPolicyDispatchDelegate`, and `load_sas_policy_delegate`.
  - Returns action indices only; it does not mutate the driver or ledger.
- Modify `FAB_RL/FABenv/reservation_simulator.py`
  - Add optional `dispatch_delegate` parameter to reservation-aware rollout.
  - Keep existing `strategy` behavior as the default path.
- Modify `FAB_RL/FABenv/vc_mcts_planner.py`
  - Add `use_delegate_dispatch` config flag.
  - Add `dispatch_delegate` to planner.
  - Build one `delegate_dispatch` root edge when enabled.
  - Apply delegated dispatch during branch evaluation and real episode execution.
- Modify `FAB_RL/FABenv/vc_mcts_probe.py`
  - Add CLI args `--dispatch-delegate {topk,rule,sas}`, `--sas-checkpoint`, and `--sas-stochastic`.
  - Construct the correct delegate once per seed and pass it into planner/episode.
- Create `FAB_RL/FABenv/tests/test_dispatch_delegate.py`
  - Unit coverage for rule delegate, policy fallback, and checkpoint loading.
- Modify `FAB_RL/FABenv/tests/test_vc_mcts_planner.py`
  - Add integration tests proving rule delegate preserves small-instance completion and uses a single `delegate_dispatch` edge.
- Modify `FAB_RL/FABenv/tests/test_vc_mcts_planner.py` probe coverage
  - Assert `vc_mcts_probe.run_seed(..., dispatch_delegate="rule")` completes and returns the selected delegate mode.

---

### Task 1: Add Dispatch Delegate Boundary

**Files:**
- Create: `FAB_RL/FABenv/dispatch_delegate.py`
- Test: `FAB_RL/FABenv/tests/test_dispatch_delegate.py`

- [ ] **Step 1: Write the failing tests**

Add these tests to `FAB_RL/FABenv/tests/test_dispatch_delegate.py`:

```python
import torch

from dispatch_delegate import (
    RuleDispatchDelegate,
    SASPolicyDispatchDelegate,
    load_sas_policy_delegate,
)
from model_checkpoint import save_policy_checkpoint
from phase2_sas_policy import Phase2SASActorCritic
from phase2_sas_driver import Phase2EpisodeDriver
from phase2_sas_observation import Phase2ObservationEncoder
from rl_environment import ResourceCalendarEnv, RewardConfig


def _driver(env, max_steps=200):
    return Phase2EpisodeDriver(
        env,
        Phase2ObservationEncoder(),
        RewardConfig(),
        max_steps=max_steps,
    )


class WaitFirstPolicy(torch.nn.Module):
    def parameters(self):
        return iter(())

    def greedy_action(self, candidate_features, candidate_mask, global_features):
        wait_index = int(candidate_mask.shape[1] - 1)
        return {
            "action": torch.tensor([wait_index]),
            "log_prob": torch.tensor([0.0]),
            "value": torch.tensor([0.0]),
        }

    def sample_action(self, candidate_features, candidate_mask, global_features):
        return self.greedy_action(candidate_features, candidate_mask, global_features)


def test_rule_dispatch_delegate_matches_driver_rule(small_encoder):
    env = ResourceCalendarEnv(small_encoder, top_k=8, w_lookahead=4.0)
    driver = _driver(env)
    driver.reset_episode()
    machine = driver.select_next_machine(driver.get_dispatchable_machines())
    pool = driver.env.build_candidate_pool(machine)

    delegate = RuleDispatchDelegate(strategy="FIFO")
    selected = delegate.select_action_index(driver, machine, pool=pool)

    assert selected == driver._rule_action_index(pool, "FIFO")


def test_policy_dispatch_delegate_falls_back_when_policy_selects_wait(small_encoder):
    env = ResourceCalendarEnv(small_encoder, top_k=8, w_lookahead=4.0)
    driver = _driver(env)
    driver.reset_episode()
    machine = driver.select_next_machine(driver.get_dispatchable_machines())
    pool = driver.env.build_candidate_pool(machine)

    delegate = SASPolicyDispatchDelegate(
        WaitFirstPolicy(),
        fallback_delegate=RuleDispatchDelegate(strategy="FIFO"),
    )
    selected = delegate.select_action_index(driver, machine, pool=pool)

    assert selected == driver._rule_action_index(pool, "FIFO")


def test_load_sas_policy_delegate_loads_checkpoint(tmp_path, small_encoder):
    env = ResourceCalendarEnv(small_encoder, top_k=8, w_lookahead=4.0)
    driver = _driver(env)
    driver.reset_episode()
    machine = driver.select_next_machine(driver.get_dispatchable_machines())
    pool = driver.env.build_candidate_pool(machine)
    observation = driver.observation_encoder.encode(machine, pool, driver.env)
    policy = Phase2SASActorCritic(
        candidate_dim=observation.candidate_features.shape[1],
        global_dim=observation.global_features.shape[0],
        hidden_dim=16,
    )
    path = tmp_path / "sas.pt"
    save_policy_checkpoint(
        policy,
        path,
        candidate_dim=observation.candidate_features.shape[1],
        global_dim=observation.global_features.shape[0],
        hidden_dim=16,
        policy_type="single",
    )

    delegate = load_sas_policy_delegate(
        str(path),
        fallback_delegate=RuleDispatchDelegate(strategy="FIFO"),
    )
    selected = delegate.select_action_index(driver, machine, pool=pool)

    assert selected is None or isinstance(selected, int)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
cd E:\code\FAB
python -m pytest FAB_RL/FABenv/tests/test_dispatch_delegate.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'dispatch_delegate'`.

- [ ] **Step 3: Implement `dispatch_delegate.py`**

Create `FAB_RL/FABenv/dispatch_delegate.py`:

```python
"""Dispatch delegates used by VC-MCTS.

VC-MCTS owns reservation timing. A dispatch delegate owns the concrete
candidate index when the selected branch is "dispatch now".
"""

from dataclasses import dataclass

import torch


def _is_real_dispatch_action(pool, action_index):
    if action_index is None:
        return False
    index = int(action_index)
    if index < 0 or index >= len(pool.actions):
        return False
    if not bool(pool.action_mask[index]):
        return False
    action = pool.actions[index]
    if getattr(action, "is_padding", False) or getattr(action, "is_wait", False):
        return False
    return int(getattr(action, "ppid", 0)) != 0


@dataclass
class RuleDispatchDelegate:
    strategy: str = "FIFO"

    def select_action_index(self, driver, machine, pool=None):
        pool = driver.env.build_candidate_pool(machine) if pool is None else pool
        return driver._rule_action_index(pool, self.strategy)


class SASPolicyDispatchDelegate:
    def __init__(self, policy, *, stochastic=False, fallback_delegate=None):
        self.policy = policy
        self.stochastic = bool(stochastic)
        self.fallback_delegate = fallback_delegate

    def _policy_device(self):
        try:
            return next(self.policy.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _policy_output(self, observation):
        device = self._policy_device()
        candidate_features = torch.as_tensor(
            observation.candidate_features, dtype=torch.float32, device=device
        ).unsqueeze(0)
        candidate_mask = torch.as_tensor(
            observation.candidate_mask, dtype=torch.bool, device=device
        ).unsqueeze(0)
        global_features = torch.as_tensor(
            observation.global_features, dtype=torch.float32, device=device
        ).unsqueeze(0)
        with torch.no_grad():
            if self.stochastic:
                return self.policy.sample_action(
                    candidate_features, candidate_mask, global_features
                )
            return self.policy.greedy_action(
                candidate_features, candidate_mask, global_features
            )

    def select_action_index(self, driver, machine, pool=None):
        pool = driver.env.build_candidate_pool(machine) if pool is None else pool
        observation = driver.observation_encoder.encode(machine, pool, driver.env)
        output = self._policy_output(observation)
        action_index = int(output["action"].detach().cpu().reshape(-1)[0])
        if _is_real_dispatch_action(pool, action_index):
            return action_index
        if self.fallback_delegate is None:
            return None
        return self.fallback_delegate.select_action_index(driver, machine, pool=pool)


def load_sas_policy_delegate(
    checkpoint_path,
    *,
    stochastic=False,
    fallback_delegate=None,
    map_location="cpu",
):
    from model_checkpoint import load_policy_checkpoint

    policy, _checkpoint = load_policy_checkpoint(
        checkpoint_path, map_location=map_location
    )
    return SASPolicyDispatchDelegate(
        policy,
        stochastic=stochastic,
        fallback_delegate=fallback_delegate,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```powershell
cd E:\code\FAB
python -m pytest FAB_RL/FABenv/tests/test_dispatch_delegate.py -q
```

Expected: PASS.

---

### Task 2: Use Delegates in Reservation Rollouts

**Files:**
- Modify: `FAB_RL/FABenv/reservation_simulator.py`
- Test: `FAB_RL/FABenv/tests/test_reservation_simulator.py`

- [ ] **Step 1: Write the failing test**

Append this test to `FAB_RL/FABenv/tests/test_reservation_simulator.py`:

```python
from dispatch_delegate import RuleDispatchDelegate
from reservation_simulator import run_rule_episode_with_reservations


def test_reservation_rollout_accepts_dispatch_delegate(small_encoder):
    env = ResourceCalendarEnv(small_encoder, top_k=8, w_lookahead=4.0)
    driver = _driver(env, max_steps=200)
    driver.reset_episode()

    summary = run_rule_episode_with_reservations(
        driver,
        strategy="SPT",
        dispatch_delegate=RuleDispatchDelegate(strategy="FIFO"),
        max_steps=200,
    )

    assert summary["completed_lots"] == 4
    assert summary["termination_reason"] == "all_lots_completed"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
cd E:\code\FAB
python -m pytest FAB_RL/FABenv/tests/test_reservation_simulator.py::test_reservation_rollout_accepts_dispatch_delegate -q
```

Expected: FAIL with `TypeError: run_rule_episode_with_reservations() got an unexpected keyword argument 'dispatch_delegate'`.

- [ ] **Step 3: Implement delegate use in rollout**

In `FAB_RL/FABenv/reservation_simulator.py`, change:

```python
def run_rule_episode_with_reservations(
    driver,
    ledger=None,
    strategy="FIFO",
    max_steps=None,
):
```

to:

```python
def run_rule_episode_with_reservations(
    driver,
    ledger=None,
    strategy="FIFO",
    max_steps=None,
    dispatch_delegate=None,
):
```

Then replace:

```python
action_index = driver._rule_action_index(decision.pool, strategy)
```

with:

```python
if dispatch_delegate is None:
    action_index = driver._rule_action_index(decision.pool, strategy)
else:
    action_index = dispatch_delegate.select_action_index(
        driver, machine, pool=decision.pool
    )
```

- [ ] **Step 4: Run tests**

Run:

```powershell
cd E:\code\FAB
python -m pytest FAB_RL/FABenv/tests/test_reservation_simulator.py -q
```

Expected: PASS.

---

### Task 3: Integrate Delegated Dispatch into VC-MCTS

**Files:**
- Modify: `FAB_RL/FABenv/vc_mcts_planner.py`
- Modify: `FAB_RL/FABenv/tests/test_vc_mcts_planner.py`

- [ ] **Step 1: Write failing VC-MCTS tests**

Append these tests to `FAB_RL/FABenv/tests/test_vc_mcts_planner.py`:

```python
from dispatch_delegate import RuleDispatchDelegate


def test_planner_builds_single_delegate_dispatch_action(small_encoder):
    env = ResourceCalendarEnv(small_encoder, top_k=8, w_lookahead=4.0)
    driver = _driver(env)
    driver.reset_episode()
    ledger = ReservationLedger()
    machine = driver.select_next_machine(driver.get_dispatchable_machines())
    planner = VCMCTSPlanner(
        VCMCTSConfig(n_iter=1, top_k_dispatch=3, top_b_reserve=0, use_delegate_dispatch=True),
        dispatch_delegate=RuleDispatchDelegate(strategy="FIFO"),
    )

    actions = planner.build_root_actions(driver, ledger, machine)

    assert [action.kind for action in actions].count("delegate_dispatch") == 1
    assert [action.kind for action in actions].count("dispatch") == 0


def test_vc_mcts_episode_completes_with_rule_dispatch_delegate(small_encoder):
    env = ResourceCalendarEnv(small_encoder, top_k=8, w_lookahead=4.0)
    driver = _driver(env, max_steps=200)
    driver.reset_episode()
    planner = VCMCTSPlanner(
        VCMCTSConfig(n_iter=4, top_k_dispatch=2, top_b_reserve=1, use_delegate_dispatch=True),
        dispatch_delegate=RuleDispatchDelegate(strategy="FIFO"),
    )

    summary = run_vc_mcts_reservation_episode(
        driver,
        planner=planner,
        max_steps=200,
        dispatch_delegate=RuleDispatchDelegate(strategy="FIFO"),
    )

    assert summary["completed_lots"] == 4
    assert summary["dispatch_delegate"] == "rule:FIFO"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
cd E:\code\FAB
python -m pytest FAB_RL/FABenv/tests/test_vc_mcts_planner.py::test_planner_builds_single_delegate_dispatch_action FAB_RL/FABenv/tests/test_vc_mcts_planner.py::test_vc_mcts_episode_completes_with_rule_dispatch_delegate -q
```

Expected: FAIL because `VCMCTSConfig` does not accept `use_delegate_dispatch` and `run_vc_mcts_reservation_episode` does not accept `dispatch_delegate`.

- [ ] **Step 3: Implement VC-MCTS delegate support**

In `FAB_RL/FABenv/vc_mcts_planner.py`:

1. Add to `VCMCTSConfig`:

```python
use_delegate_dispatch: bool = False
```

2. Change planner constructor:

```python
def __init__(self, config=None, rollout_evaluator=None, dispatch_delegate=None):
    self.config = config if config is not None else VCMCTSConfig()
    self.rollout_evaluator = rollout_evaluator
    self.dispatch_delegate = dispatch_delegate
```

3. In `build_root_actions`, before enumerating TopK dispatch actions, add:

```python
if self.config.use_delegate_dispatch:
    pool = driver.env.build_candidate_pool(machine)
    delegate = self.dispatch_delegate
    action_index = None if delegate is None else delegate.select_action_index(
        driver, machine, pool=pool
    )
    if action_index is not None:
        action = pool.actions[int(action_index)]
        actions.append(
            VCMCTSAction(
                kind="delegate_dispatch",
                machine=machine,
                action_index=int(action_index),
                lot=int(action.lot),
                ppid=int(action.ppid),
                prior=max(1e-6, float(getattr(action, "score", 0.0)) + 1.0),
            )
        )
else:
    # keep existing TopK dispatch enumeration here
```

4. In `evaluate_action`, pass the delegate to rollout:

```python
run_rule_episode_with_reservations(
    branch_driver,
    ledger=branch_ledger,
    strategy=self.config.rollout_strategy,
    max_steps=self.config.rollout_max_steps or branch_driver.max_steps,
    dispatch_delegate=self.dispatch_delegate,
)
```

5. In `_apply_action`, treat `delegate_dispatch` like dispatch but ask the delegate again:

```python
if action.kind in ("dispatch", "delegate_dispatch"):
    pool = driver.env.build_candidate_pool(action.machine)
    action_index = action.action_index
    if action.kind == "delegate_dispatch" and self.dispatch_delegate is not None:
        action_index = self.dispatch_delegate.select_action_index(
            driver, action.machine, pool=pool
        )
    if action_index is None:
        advance_to_next_event_with_ledger(driver, ledger)
        return
    driver.step_with_action(action.machine, action_index, pool=pool)
    return
```

6. Add `dispatch_delegate=None` to `run_vc_mcts_reservation_episode`. Use it for real execution:

```python
if action.kind in ("dispatch", "delegate_dispatch"):
    pool = driver.env.build_candidate_pool(action.machine)
    action_index = action.action_index
    delegate = dispatch_delegate or getattr(planner, "dispatch_delegate", None)
    if action.kind == "delegate_dispatch" and delegate is not None:
        action_index = delegate.select_action_index(
            driver, action.machine, pool=pool
        )
    if action_index is None:
        next_time = advance_to_next_event_with_ledger(driver, ledger)
        if next_time is None:
            if not driver.termination_reason:
                driver.termination_reason = "no_future_event"
            break
        steps += 1
        continue
    result = driver.step_with_action(action.machine, action_index, pool=pool)
    episode_reward += float(result.reward)
    steps += 1
    continue
```

7. Add summary label:

```python
delegate = dispatch_delegate or getattr(planner, "dispatch_delegate", None)
if delegate is not None:
    summary["dispatch_delegate"] = getattr(delegate, "label", delegate.__class__.__name__)
```

- [ ] **Step 4: Run VC-MCTS tests**

Run:

```powershell
cd E:\code\FAB
python -m pytest FAB_RL/FABenv/tests/test_vc_mcts_planner.py -q
```

Expected: PASS.

---

### Task 4: Add Probe CLI Support

**Files:**
- Modify: `FAB_RL/FABenv/vc_mcts_probe.py`
- Modify: `FAB_RL/FABenv/tests/test_vc_mcts_planner.py`

- [ ] **Step 1: Write failing probe test**

Append this test to `FAB_RL/FABenv/tests/test_vc_mcts_planner.py`:

```python
def test_vc_mcts_probe_supports_rule_dispatch_delegate():
    row = run_vc_mcts_seed(
        instance="small",
        seed=0,
        strategy="FIFO",
        w_lookahead=4.0,
        top_b=1,
        top_k_dispatch=2,
        n_iter=2,
        max_steps=200,
        skip_oracle=True,
        rollout_max_steps=20,
        dispatch_delegate="rule",
    )

    assert row["oracle"] is None
    assert row["vc_mcts"]["completed_lots"] == 4.0
    assert row["vc_mcts"]["dispatch_delegate"] == "rule:FIFO"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
cd E:\code\FAB
python -m pytest FAB_RL/FABenv/tests/test_vc_mcts_planner.py::test_vc_mcts_probe_supports_rule_dispatch_delegate -q
```

Expected: FAIL with `TypeError: run_seed() got an unexpected keyword argument 'dispatch_delegate'`.

- [ ] **Step 3: Implement probe args and factory**

In `FAB_RL/FABenv/vc_mcts_probe.py`:

1. Import delegates:

```python
from dispatch_delegate import RuleDispatchDelegate, load_sas_policy_delegate
```

2. Add helper:

```python
def _make_dispatch_delegate(mode, strategy, sas_checkpoint=None, sas_stochastic=False):
    if mode in (None, "topk"):
        return None, False
    fallback = RuleDispatchDelegate(strategy=strategy)
    if mode == "rule":
        return fallback, True
    if mode == "sas":
        if not sas_checkpoint:
            raise ValueError("--sas-checkpoint is required when --dispatch-delegate=sas")
        return load_sas_policy_delegate(
            sas_checkpoint,
            stochastic=sas_stochastic,
            fallback_delegate=fallback,
        ), True
    raise ValueError(f"unknown dispatch delegate mode: {mode!r}")
```

3. Extend `run_seed`, `_run_seed_job`, and `main` signatures with:

```python
dispatch_delegate="topk",
sas_checkpoint=None,
sas_stochastic=False,
```

4. Construct planner:

```python
delegate, use_delegate_dispatch = _make_dispatch_delegate(
    dispatch_delegate,
    strategy,
    sas_checkpoint=sas_checkpoint,
    sas_stochastic=sas_stochastic,
)
planner = VCMCTSPlanner(
    VCMCTSConfig(
        n_iter=n_iter,
        top_k_dispatch=top_k_dispatch,
        top_b_reserve=top_b,
        rollout_strategy=strategy,
        rollout_max_steps=rollout_max_steps or max_steps,
        use_delegate_dispatch=use_delegate_dispatch,
    ),
    dispatch_delegate=delegate,
)
```

5. Pass `dispatch_delegate=delegate` to `run_vc_mcts_reservation_episode`.

6. Add CLI args:

```python
parser.add_argument("--dispatch-delegate", choices=["topk", "rule", "sas"], default="topk")
parser.add_argument("--sas-checkpoint", default=None)
parser.add_argument("--sas-stochastic", action="store_true")
```

- [ ] **Step 4: Run probe tests**

Run:

```powershell
cd E:\code\FAB
python -m pytest FAB_RL/FABenv/tests/test_vc_mcts_planner.py::test_vc_mcts_probe_supports_rule_dispatch_delegate -q
```

Expected: PASS.

---

### Task 5: Full Verification

**Files:**
- No new files.

- [ ] **Step 1: Run focused test suite**

Run:

```powershell
cd E:\code\FAB
python -m pytest FAB_RL/FABenv/tests/test_dispatch_delegate.py FAB_RL/FABenv/tests/test_reservation_simulator.py FAB_RL/FABenv/tests/test_vc_mcts_planner.py -q
```

Expected: PASS.

- [ ] **Step 2: Run a small delegated probe**

Run:

```powershell
cd E:\code\FAB\FAB_RL\FABenv
python vc_mcts_probe.py `
  --instance small `
  --seeds 1 `
  --strategy FIFO `
  --skip-oracle `
  --dispatch-delegate rule `
  --top-b 1 `
  --top-k-dispatch 2 `
  --n-iter 2 `
  --max-steps 200 `
  --rollout-max-steps 20 `
  --max-decisions 20
```

Expected: JSON output has `vc_mcts.completed_lots = 4.0` and `vc_mcts.dispatch_delegate = "rule:FIFO"`.

- [ ] **Step 3: Run optional SAS checkpoint smoke**

Run only when a checkpoint path is available:

```powershell
cd E:\code\FAB\FAB_RL\FABenv
python vc_mcts_probe.py `
  --instance small `
  --seeds 1 `
  --strategy FIFO `
  --skip-oracle `
  --dispatch-delegate sas `
  --sas-checkpoint pressure_mh_hard.pt `
  --top-b 1 `
  --top-k-dispatch 2 `
  --n-iter 2 `
  --max-steps 200 `
  --rollout-max-steps 20 `
  --max-decisions 20
```

Expected: command runs without invalid-action loops. If checkpoint dimensions do not match the small instance observation dimensions, use a checkpoint trained with matching observation dimensions or run `--dispatch-delegate rule` as the control.

- [ ] **Step 4: Commit**

Run:

```powershell
cd E:\code\FAB
git add FAB_RL/FABenv/dispatch_delegate.py FAB_RL/FABenv/reservation_simulator.py FAB_RL/FABenv/vc_mcts_planner.py FAB_RL/FABenv/vc_mcts_probe.py FAB_RL/FABenv/tests/test_dispatch_delegate.py FAB_RL/FABenv/tests/test_reservation_simulator.py FAB_RL/FABenv/tests/test_vc_mcts_planner.py FAB_RL/FABenv/docs/superpowers/plans/2026-06-06-vc-mcts-sas-delegate.md
git commit -m "feat: add sas dispatch delegate for vc mcts"
```

Expected: commit succeeds after tests pass.

---

## Self-Review

- Spec coverage: The plan separates VC reservation decisions from dispatch selection, preserves the current rule path, adds SAS checkpoint loading, and exposes probe CLI controls.
- Placeholder scan: No `TBD`, `TODO`, or undefined implementation hooks remain.
- Type consistency: The delegate API is consistently `select_action_index(driver, machine, pool=None)`; planner and simulator call the same method.
