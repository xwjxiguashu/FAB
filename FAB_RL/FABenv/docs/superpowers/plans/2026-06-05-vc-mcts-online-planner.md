# VC-MCTS Online Planner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first online VC-MCTS reservation planner slice on top of the existing ROP, reservation ledger, and oracle rollout simulator.

**Architecture:** Keep SAS/rule dispatch and `ResourceCalendarEnv` unchanged. Add a narrow `vc_mcts_planner.py` module that builds root actions (`dispatch`, `reserve`, `no_op`), evaluates each branch with the same reservation-aware rollout path, and returns a trace explaining root visits and objective estimates.

**Tech Stack:** Python, pytest, existing `FAB_RL/FABenv` bare-import modules.

---

## File Structure

- Create: `FAB_RL/FABenv/vc_mcts_planner.py`
  - Owns VC-MCTS config, action dataclasses, root-edge statistics, objective comparison, branch rollout evaluation, and `VCMCTSPlanner.plan()`.
- Create: `FAB_RL/FABenv/vc_mcts_probe.py`
  - CLI probe comparing `baseline`, `oracle`, and `vc_mcts` rows on `small`, `pressure`, and `late_hi`.
- Create: `FAB_RL/FABenv/tests/test_vc_mcts_planner.py`
  - Unit tests for root selection and integration tests for one complete reservation-aware episode.
- Modify: `FAB_RL/FABenv/reservation_simulator.py`
  - Export public helpers for cloned rollouts and ledger-aware event advance so the planner does not depend on private names.
- Modify: `FAB_RL/FABenv/oracle_reservation_probe.py`
  - Keep current oracle behavior, but optionally import the new probe only by CLI; do not change oracle semantics.

The first implementation is a root-level MCTS/rollout planner. It does not yet add multi-level subtree reuse, SAS policy priors, or multi-head critic leaf evaluation. Those are deliberately deferred until this slice has stable metrics.

---

### Task 1: Public Rollout Helpers

**Files:**
- Modify: `FAB_RL/FABenv/reservation_simulator.py`
- Test: `FAB_RL/FABenv/tests/test_vc_mcts_planner.py`

- [ ] **Step 1: Write failing tests for public clone and advance helpers**

Create `FAB_RL/FABenv/tests/test_vc_mcts_planner.py` with this initial content:

```python
from phase2_sas_driver import Phase2EpisodeDriver
from phase2_sas_observation import Phase2ObservationEncoder
from reservation_ledger import ReservationLedger
from reservation_simulator import (
    advance_to_next_event_with_ledger,
    clone_driver_for_rollout,
    clone_ledger_for_rollout,
)
from rl_environment import ResourceCalendarEnv, RewardConfig


def _driver(env, max_steps=200):
    return Phase2EpisodeDriver(
        env,
        Phase2ObservationEncoder(),
        RewardConfig(),
        max_steps=max_steps,
    )


def test_clone_helpers_copy_driver_and_ledger_without_mutating_original(small_encoder):
    env = ResourceCalendarEnv(small_encoder, top_k=8, w_lookahead=4.0)
    driver = _driver(env)
    driver.reset_episode()
    ledger = ReservationLedger()
    ledger.reserve(machine=1, future_lot=2, eta=1.5, created_at=0.0, expires_at=3.0)

    cloned_driver = clone_driver_for_rollout(driver)
    cloned_ledger = clone_ledger_for_rollout(ledger)
    cloned_ledger.release(1)
    cloned_driver.env.advance_time(1.0)

    assert ledger.is_reserved(1)
    assert driver.env.current_time == 0.0
    assert cloned_driver.env.current_time == 1.0


def test_advance_to_next_event_with_ledger_sees_reserved_lot_eta(small_encoder):
    env = ResourceCalendarEnv(small_encoder, top_k=8, w_lookahead=4.0)
    driver = _driver(env)
    driver.reset_episode()
    ledger = ReservationLedger()
    ledger.reserve(machine=1, future_lot=2, eta=1.5, created_at=0.0, expires_at=3.0)

    advanced_to = advance_to_next_event_with_ledger(driver, ledger)

    assert advanced_to == 1.5
    assert driver.env.current_time == 1.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run from `FAB_RL/FABenv`:

```powershell
python -m pytest tests/test_vc_mcts_planner.py -q
```

Expected: FAIL with `ImportError` because `advance_to_next_event_with_ledger`, `clone_driver_for_rollout`, and `clone_ledger_for_rollout` are not exported.

- [ ] **Step 3: Export public wrappers in `reservation_simulator.py`**

Add these functions below `_advance_to_next_event_with_ledger` and `_clone_driver`:

```python
def advance_to_next_event_with_ledger(driver, ledger):
    """Public wrapper used by VC-MCTS no-op branches."""
    return _advance_to_next_event_with_ledger(driver, ledger)


def clone_ledger_for_rollout(ledger):
    """Public wrapper for non-destructive branch rollouts."""
    return _clone_ledger(ledger)


def clone_driver_for_rollout(driver):
    """Public wrapper for non-destructive branch rollouts."""
    return _clone_driver(driver)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```powershell
python -m pytest tests/test_vc_mcts_planner.py -q
```

Expected: `2 passed`.

- [ ] **Step 5: Commit**

```powershell
git add reservation_simulator.py tests/test_vc_mcts_planner.py
git commit -m "test: expose reservation rollout helpers"
```

---

### Task 2: Root Action Model and Objective Comparison

**Files:**
- Create: `FAB_RL/FABenv/vc_mcts_planner.py`
- Modify: `FAB_RL/FABenv/tests/test_vc_mcts_planner.py`

- [ ] **Step 1: Add failing tests for objective ordering and root action stats**

Append to `tests/test_vc_mcts_planner.py`:

```python
from vc_mcts_planner import (
    VCMCTSAction,
    VCMCTSEdgeStats,
    VCMCTSObjective,
    compare_objectives,
)


def test_objective_comparison_is_qtime_then_o2_then_utilization():
    baseline = VCMCTSObjective(qtime_violation_count=0.0, priority_weighted_wait=10.0, avg_utilization=0.5)
    worse_qtime = VCMCTSObjective(qtime_violation_count=1.0, priority_weighted_wait=0.0, avg_utilization=1.0)
    better_o2 = VCMCTSObjective(qtime_violation_count=0.0, priority_weighted_wait=8.0, avg_utilization=0.1)
    better_util = VCMCTSObjective(qtime_violation_count=0.0, priority_weighted_wait=10.0, avg_utilization=0.6)

    assert compare_objectives(better_o2, baseline) < 0
    assert compare_objectives(worse_qtime, baseline) > 0
    assert compare_objectives(better_util, baseline) < 0


def test_edge_stats_tracks_visits_and_mean_objective():
    action = VCMCTSAction(kind="reserve", machine=1, future_lot=2, eta=1.5, prior=0.7)
    stats = VCMCTSEdgeStats(action=action)

    stats.record(VCMCTSObjective(qtime_violation_count=0.0, priority_weighted_wait=10.0, avg_utilization=0.5))
    stats.record(VCMCTSObjective(qtime_violation_count=0.0, priority_weighted_wait=6.0, avg_utilization=0.7))

    assert stats.visits == 2
    assert stats.mean_objective == VCMCTSObjective(
        qtime_violation_count=0.0,
        priority_weighted_wait=8.0,
        avg_utilization=0.6,
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
python -m pytest tests/test_vc_mcts_planner.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'vc_mcts_planner'`.

- [ ] **Step 3: Create `vc_mcts_planner.py` with dataclasses and comparison**

Add:

```python
"""Online VC-MCTS reservation planner.

This first slice is a root-level MCTS planner: build root actions, evaluate
branches with reservation-aware rollouts, and choose by visit count with a
lexicographic objective tie-break.
"""
from dataclasses import dataclass, field
import math


@dataclass(frozen=True)
class VCMCTSConfig:
    n_iter: int = 24
    top_k_dispatch: int = 3
    top_b_reserve: int = 2
    exploration_c: float = 1.5
    qtime_penalty: float = 10000.0
    util_weight: float = 1.0
    min_priority_gap: float = 0.0
    reservation_ttl: float = 1.0
    rollout_strategy: str = "FIFO"
    rollout_max_steps: int | None = None


@dataclass(frozen=True)
class VCMCTSObjective:
    qtime_violation_count: float
    priority_weighted_wait: float
    avg_utilization: float


@dataclass(frozen=True)
class VCMCTSAction:
    kind: str
    machine: int | None = None
    action_index: int | None = None
    lot: int | None = None
    ppid: int | None = None
    future_lot: int | None = None
    eta: float | None = None
    prior: float = 1.0


@dataclass
class VCMCTSEdgeStats:
    action: VCMCTSAction
    visits: int = 0
    total_qtime: float = 0.0
    total_o2: float = 0.0
    total_util: float = 0.0

    def record(self, objective):
        self.visits += 1
        self.total_qtime += float(objective.qtime_violation_count)
        self.total_o2 += float(objective.priority_weighted_wait)
        self.total_util += float(objective.avg_utilization)

    @property
    def mean_objective(self):
        if self.visits <= 0:
            return None
        return VCMCTSObjective(
            qtime_violation_count=self.total_qtime / self.visits,
            priority_weighted_wait=self.total_o2 / self.visits,
            avg_utilization=self.total_util / self.visits,
        )


@dataclass
class VCMCTSDecisionTrace:
    selected_action: VCMCTSAction
    edge_stats: list[VCMCTSEdgeStats] = field(default_factory=list)


def compare_objectives(left, right):
    left_key = (
        float(left.qtime_violation_count),
        float(left.priority_weighted_wait),
        -float(left.avg_utilization),
    )
    right_key = (
        float(right.qtime_violation_count),
        float(right.priority_weighted_wait),
        -float(right.avg_utilization),
    )
    return (left_key > right_key) - (left_key < right_key)


def objective_to_score(objective, config):
    return (
        -float(config.qtime_penalty) * float(objective.qtime_violation_count)
        -float(objective.priority_weighted_wait)
        +float(config.util_weight) * float(objective.avg_utilization)
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```powershell
python -m pytest tests/test_vc_mcts_planner.py -q
```

Expected: `4 passed`.

- [ ] **Step 5: Commit**

```powershell
git add vc_mcts_planner.py tests/test_vc_mcts_planner.py
git commit -m "feat: add vc mcts root action model"
```

---

### Task 3: Root Action Builder

**Files:**
- Modify: `FAB_RL/FABenv/vc_mcts_planner.py`
- Modify: `FAB_RL/FABenv/tests/test_vc_mcts_planner.py`

- [ ] **Step 1: Add failing tests for dispatch and reserve root actions**

Append:

```python
from vc_mcts_planner import VCMCTSConfig, VCMCTSPlanner


def test_planner_builds_dispatch_and_noop_actions(small_encoder):
    env = ResourceCalendarEnv(small_encoder, top_k=8, w_lookahead=4.0)
    driver = _driver(env)
    driver.reset_episode()
    ledger = ReservationLedger()
    machine = driver.select_next_machine(driver.get_dispatchable_machines())
    planner = VCMCTSPlanner(VCMCTSConfig(n_iter=1, top_k_dispatch=2, top_b_reserve=0))

    actions = planner.build_root_actions(driver, ledger, machine)

    assert actions[0].kind == "no_op"
    assert [a.kind for a in actions].count("dispatch") == 2
    assert all(a.machine == machine for a in actions if a.kind == "dispatch")


def test_planner_builds_reserve_actions_from_rop(small_encoder):
    env = ResourceCalendarEnv(small_encoder, top_k=8, w_lookahead=4.0)
    driver = _driver(env)
    driver.reset_episode()
    ledger = ReservationLedger()
    planner = VCMCTSPlanner(VCMCTSConfig(n_iter=1, top_k_dispatch=1, top_b_reserve=2))

    actions = planner.build_root_actions(driver, ledger, machine=1)

    assert any(a.kind == "reserve" and a.future_lot == 2 for a in actions)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
python -m pytest tests/test_vc_mcts_planner.py -q
```

Expected: FAIL with `ImportError` or `AttributeError` because `VCMCTSPlanner` is not implemented.

- [ ] **Step 3: Implement root action builder**

Add imports near the top of `vc_mcts_planner.py`:

```python
from reservation_rop import detect_reservation_opportunities
```

Add this class:

```python
class VCMCTSPlanner:
    def __init__(self, config=None, rollout_evaluator=None):
        self.config = config if config is not None else VCMCTSConfig()
        self.rollout_evaluator = rollout_evaluator

    def build_root_actions(self, driver, ledger, machine):
        machine = int(machine)
        actions = [VCMCTSAction(kind="no_op", machine=machine, prior=0.05)]

        pool = driver.env.build_candidate_pool(machine)
        dispatch = []
        for index, action in enumerate(pool.actions):
            if not bool(pool.action_mask[index]):
                continue
            if getattr(action, "is_padding", False) or getattr(action, "is_wait", False):
                continue
            if int(action.ppid) == 0:
                continue
            dispatch.append((float(getattr(action, "score", 0.0)), index, action))
        dispatch.sort(key=lambda row: (-row[0], row[1]))
        for score, index, action in dispatch[: self.config.top_k_dispatch]:
            actions.append(
                VCMCTSAction(
                    kind="dispatch",
                    machine=machine,
                    action_index=int(index),
                    lot=int(action.lot),
                    ppid=int(action.ppid),
                    prior=max(1e-6, float(score) + 1.0),
                )
            )

        opportunities = detect_reservation_opportunities(
            driver.env,
            machines=[machine],
            ledger=ledger,
            top_b=self.config.top_b_reserve,
            min_priority_gap=self.config.min_priority_gap,
        )
        for opportunity in opportunities:
            actions.append(
                VCMCTSAction(
                    kind="reserve",
                    machine=int(opportunity.machine),
                    future_lot=int(opportunity.future_lot),
                    eta=float(opportunity.eta),
                    prior=max(1e-6, float(opportunity.score)),
                )
            )
        return actions
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```powershell
python -m pytest tests/test_vc_mcts_planner.py -q
```

Expected: all tests in `test_vc_mcts_planner.py` pass.

- [ ] **Step 5: Commit**

```powershell
git add vc_mcts_planner.py tests/test_vc_mcts_planner.py
git commit -m "feat: build vc mcts root actions"
```

---

### Task 4: Branch Rollout Evaluation

**Files:**
- Modify: `FAB_RL/FABenv/vc_mcts_planner.py`
- Modify: `FAB_RL/FABenv/tests/test_vc_mcts_planner.py`

- [ ] **Step 1: Add failing tests for non-destructive branch evaluation**

Append:

```python
def test_evaluate_action_is_non_destructive(small_encoder):
    env = ResourceCalendarEnv(small_encoder, top_k=8, w_lookahead=4.0)
    driver = _driver(env)
    driver.reset_episode()
    ledger = ReservationLedger()
    machine = driver.select_next_machine(driver.get_dispatchable_machines())
    planner = VCMCTSPlanner(VCMCTSConfig(n_iter=1, top_k_dispatch=1, top_b_reserve=1))
    action = next(a for a in planner.build_root_actions(driver, ledger, machine) if a.kind == "dispatch")

    objective = planner.evaluate_action(driver, ledger, action)

    assert objective.priority_weighted_wait >= 0.0
    assert len(driver.env.completed_lots) == 0
    assert driver.env.lot_schedule.shape[0] == 0
    assert ledger.reserved_machines() == set()


def test_reserve_branch_records_reservation_only_in_rollout(small_encoder):
    env = ResourceCalendarEnv(small_encoder, top_k=8, w_lookahead=4.0)
    driver = _driver(env)
    driver.reset_episode()
    ledger = ReservationLedger()
    planner = VCMCTSPlanner(VCMCTSConfig(n_iter=1, top_k_dispatch=1, top_b_reserve=2))
    reserve = next(a for a in planner.build_root_actions(driver, ledger, machine=1) if a.kind == "reserve")

    objective = planner.evaluate_action(driver, ledger, reserve)

    assert objective.priority_weighted_wait >= 0.0
    assert not ledger.is_reserved(1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
python -m pytest tests/test_vc_mcts_planner.py -q
```

Expected: FAIL with `AttributeError: 'VCMCTSPlanner' object has no attribute 'evaluate_action'`.

- [ ] **Step 3: Implement branch evaluator**

Add imports:

```python
from reservation_simulator import (
    advance_to_next_event_with_ledger,
    clone_driver_for_rollout,
    clone_ledger_for_rollout,
    run_rule_episode_with_reservations,
    schedule_metrics_with_priority_wait,
)
```

Add methods to `VCMCTSPlanner`:

```python
    def evaluate_action(self, driver, ledger, action):
        if self.rollout_evaluator is not None:
            return self.rollout_evaluator(driver, ledger, action, self.config)

        branch_driver = clone_driver_for_rollout(driver)
        branch_ledger = clone_ledger_for_rollout(ledger)
        self._apply_action(branch_driver, branch_ledger, action)
        run_rule_episode_with_reservations(
            branch_driver,
            ledger=branch_ledger,
            strategy=self.config.rollout_strategy,
            max_steps=self.config.rollout_max_steps or branch_driver.max_steps,
        )
        metrics = schedule_metrics_with_priority_wait(branch_driver.env.encoder, branch_driver.env)
        return VCMCTSObjective(
            qtime_violation_count=float(metrics["qtime_violation_count"]),
            priority_weighted_wait=float(metrics["priority_weighted_wait"]),
            avg_utilization=float(metrics["avg_utilization"]),
        )

    def _apply_action(self, driver, ledger, action):
        if action.kind == "no_op":
            advance_to_next_event_with_ledger(driver, ledger)
            return
        if action.kind == "reserve":
            ledger.reserve(
                action.machine,
                action.future_lot,
                eta=action.eta,
                created_at=driver.env.current_time,
                expires_at=float(action.eta) + float(self.config.reservation_ttl),
                reason="vc_mcts",
            )
            return
        if action.kind == "dispatch":
            pool = driver.env.build_candidate_pool(action.machine)
            driver.step_with_action(action.machine, action.action_index, pool=pool)
            return
        raise ValueError(f"unknown VC-MCTS action kind: {action.kind!r}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```powershell
python -m pytest tests/test_vc_mcts_planner.py -q
```

Expected: all tests in `test_vc_mcts_planner.py` pass.

- [ ] **Step 5: Commit**

```powershell
git add vc_mcts_planner.py tests/test_vc_mcts_planner.py
git commit -m "feat: evaluate vc mcts rollout branches"
```

---

### Task 5: Root-Level MCTS Selection

**Files:**
- Modify: `FAB_RL/FABenv/vc_mcts_planner.py`
- Modify: `FAB_RL/FABenv/tests/test_vc_mcts_planner.py`

- [ ] **Step 1: Add failing tests for visit-count selection**

Append:

```python
def test_planner_selects_best_action_by_visits_with_injected_evaluator(small_encoder):
    env = ResourceCalendarEnv(small_encoder, top_k=8, w_lookahead=4.0)
    driver = _driver(env)
    driver.reset_episode()
    ledger = ReservationLedger()

    def evaluator(_driver, _ledger, action, _config):
        if action.kind == "reserve":
            return VCMCTSObjective(qtime_violation_count=0.0, priority_weighted_wait=1.0, avg_utilization=0.5)
        return VCMCTSObjective(qtime_violation_count=0.0, priority_weighted_wait=10.0, avg_utilization=0.5)

    planner = VCMCTSPlanner(
        VCMCTSConfig(n_iter=8, top_k_dispatch=1, top_b_reserve=2),
        rollout_evaluator=evaluator,
    )
    trace = planner.plan(driver, ledger, machine=1)

    assert trace.selected_action.kind == "reserve"
    assert sum(edge.visits for edge in trace.edge_stats) == 8
    assert any(edge.action.kind == "reserve" and edge.visits > 0 for edge in trace.edge_stats)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
python -m pytest tests/test_vc_mcts_planner.py -q
```

Expected: FAIL with `AttributeError: 'VCMCTSPlanner' object has no attribute 'plan'`.

- [ ] **Step 3: Implement UCT selection and final action choice**

Add methods to `VCMCTSPlanner`:

```python
    def plan(self, driver, ledger, machine):
        actions = self.build_root_actions(driver, ledger, machine)
        edges = [VCMCTSEdgeStats(action=action) for action in actions]
        for _ in range(max(1, int(self.config.n_iter))):
            edge = self._select_edge(edges)
            objective = self.evaluate_action(driver, ledger, edge.action)
            edge.record(objective)
        selected = self._choose_final_action(edges)
        return VCMCTSDecisionTrace(selected_action=selected.action, edge_stats=edges)

    def _select_edge(self, edges):
        for edge in edges:
            if edge.visits == 0:
                return edge
        total_visits = sum(edge.visits for edge in edges)
        log_total = math.log(max(total_visits, 1))

        def uct(edge):
            mean = edge.mean_objective
            exploitation = objective_to_score(mean, self.config)
            exploration = (
                float(self.config.exploration_c)
                * float(edge.action.prior)
                * math.sqrt(log_total / max(edge.visits, 1))
            )
            return exploitation + exploration

        return max(edges, key=uct)

    def _choose_final_action(self, edges):
        def key(edge):
            objective = edge.mean_objective
            if objective is None:
                return (-1, 0.0, 0.0, 0.0)
            return (
                edge.visits,
                -float(objective.qtime_violation_count),
                -float(objective.priority_weighted_wait),
                float(objective.avg_utilization),
            )
        return max(edges, key=key)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```powershell
python -m pytest tests/test_vc_mcts_planner.py -q
```

Expected: all tests in `test_vc_mcts_planner.py` pass.

- [ ] **Step 5: Commit**

```powershell
git add vc_mcts_planner.py tests/test_vc_mcts_planner.py
git commit -m "feat: select vc mcts root action"
```

---

### Task 6: Online VC-MCTS Episode Runner

**Files:**
- Modify: `FAB_RL/FABenv/vc_mcts_planner.py`
- Modify: `FAB_RL/FABenv/tests/test_vc_mcts_planner.py`

- [ ] **Step 1: Add failing integration test**

Append:

```python
from vc_mcts_planner import run_vc_mcts_reservation_episode


def test_vc_mcts_episode_completes_small_instance(small_encoder):
    env = ResourceCalendarEnv(small_encoder, top_k=8, w_lookahead=4.0)
    driver = _driver(env, max_steps=200)
    driver.reset_episode()
    planner = VCMCTSPlanner(VCMCTSConfig(n_iter=4, top_k_dispatch=2, top_b_reserve=1))

    summary = run_vc_mcts_reservation_episode(driver, planner=planner, max_steps=200)

    assert summary["completed_lots"] == 4
    assert summary["vc_mcts_decisions"] > 0
    assert summary["active_reservations"] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
python -m pytest tests/test_vc_mcts_planner.py -q
```

Expected: FAIL with `ImportError` because `run_vc_mcts_reservation_episode` does not exist.

- [ ] **Step 3: Implement episode runner**

Add imports:

```python
from reservation_ledger import ReservationLedger
from reservation_simulator import finalize_reservation_ledger
```

Add this function to `vc_mcts_planner.py`:

```python
def run_vc_mcts_reservation_episode(driver, planner=None, ledger=None, max_steps=None):
    planner = planner if planner is not None else VCMCTSPlanner()
    ledger = ledger if ledger is not None else ReservationLedger()
    steps = 0
    decisions = 0
    reservations_made = 0
    episode_reward = 0.0
    limit = int(driver.max_steps if max_steps is None else max_steps)

    while steps < limit:
        done, reason = driver.is_episode_done()
        if done:
            driver.termination_reason = reason
            break

        ledger.release_expired(driver.env.current_time)
        machines = [m for m in driver.get_dispatchable_machines() if m not in ledger.reserved_machines()]
        if not machines:
            next_time = advance_to_next_event_with_ledger(driver, ledger)
            if next_time is None:
                if not driver.termination_reason:
                    driver.termination_reason = "no_future_event"
                break
            steps += 1
            continue

        machine = driver.select_next_machine(machines)
        trace = planner.plan(driver, ledger, machine)
        decisions += 1
        action = trace.selected_action
        if action.kind == "reserve":
            ledger.reserve(
                action.machine,
                action.future_lot,
                eta=action.eta,
                created_at=driver.env.current_time,
                expires_at=float(action.eta) + float(planner.config.reservation_ttl),
                reason="vc_mcts",
            )
            reservations_made += 1
            steps += 1
            continue
        if action.kind == "no_op":
            advance_to_next_event_with_ledger(driver, ledger)
            steps += 1
            continue

        pool = driver.env.build_candidate_pool(action.machine)
        result = driver.step_with_action(action.machine, action.action_index, pool=pool)
        episode_reward += float(result.reward)
        steps += 1

    finalize_reservation_ledger(ledger, driver.env)
    summary = driver._summary(steps, episode_reward)
    summary["vc_mcts_decisions"] = int(decisions)
    summary["reservations_made"] = int(reservations_made)
    summary["active_reservations"] = len(ledger.reserved_machines())
    return summary
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```powershell
python -m pytest tests/test_vc_mcts_planner.py -q
```

Expected: all tests in `test_vc_mcts_planner.py` pass.

- [ ] **Step 5: Run reservation regression tests**

Run:

```powershell
python -m pytest tests/test_reservation_ledger.py tests/test_reservation_rop.py tests/test_reservation_simulator.py tests/test_vc_mcts_planner.py -q
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit**

```powershell
git add vc_mcts_planner.py tests/test_vc_mcts_planner.py
git commit -m "feat: run online vc mcts reservation episode"
```

---

### Task 7: VC-MCTS Probe CLI

**Files:**
- Create: `FAB_RL/FABenv/vc_mcts_probe.py`
- Test: `FAB_RL/FABenv/tests/test_vc_mcts_planner.py`

- [ ] **Step 1: Add failing smoke test for CLI main**

Append:

```python
from vc_mcts_probe import run_seed as run_vc_mcts_seed


def test_vc_mcts_probe_run_seed_returns_baseline_oracle_and_mcts():
    row = run_vc_mcts_seed(
        instance="small",
        seed=0,
        strategy="FIFO",
        w_lookahead=4.0,
        top_b=1,
        top_k_dispatch=2,
        n_iter=2,
        max_steps=200,
    )

    assert set(row) >= {"seed", "baseline", "oracle", "vc_mcts", "delta"}
    assert row["baseline"]["completed_lots"] == 4.0
    assert row["vc_mcts"]["completed_lots"] == 4.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
python -m pytest tests/test_vc_mcts_planner.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'vc_mcts_probe'`.

- [ ] **Step 3: Create `vc_mcts_probe.py`**

Add:

```python
"""Compare rule baseline, oracle reservation, and online VC-MCTS."""
import argparse
import json

from oracle_reservation_probe import _encoder_factory, _driver, _full_horizon_lookahead
from reservation_simulator import run_oracle_reservation_episode, schedule_metrics_with_priority_wait
from rl_environment import ResourceCalendarEnv
from vc_mcts_planner import VCMCTSConfig, VCMCTSPlanner, run_vc_mcts_reservation_episode


def run_seed(
    instance,
    seed,
    strategy,
    w_lookahead,
    top_b,
    top_k_dispatch,
    n_iter,
    max_steps,
    process_noise=False,
):
    factory = _encoder_factory(instance)

    baseline_encoder = factory()
    baseline_env = ResourceCalendarEnv(
        baseline_encoder,
        top_k=8,
        w_lookahead=w_lookahead,
        process_noise_enabled=process_noise,
        noise_seed=seed,
    )
    baseline_driver = _driver(baseline_env, max_steps)
    baseline_driver.reset_episode()
    baseline_summary = baseline_driver.run_rule_episode(strategy=strategy)
    baseline_metrics = schedule_metrics_with_priority_wait(baseline_encoder, baseline_env)

    oracle_encoder = factory()
    oracle_env = ResourceCalendarEnv(
        oracle_encoder,
        top_k=8,
        w_lookahead=_full_horizon_lookahead(oracle_encoder),
        process_noise_enabled=process_noise,
        noise_seed=seed,
    )
    oracle_driver = _driver(oracle_env, max_steps)
    oracle_driver.reset_episode()
    oracle_summary = run_oracle_reservation_episode(
        oracle_driver,
        strategy=strategy,
        top_b=top_b,
        max_steps=max_steps,
    )
    oracle_metrics = schedule_metrics_with_priority_wait(oracle_encoder, oracle_env)

    mcts_encoder = factory()
    mcts_env = ResourceCalendarEnv(
        mcts_encoder,
        top_k=8,
        w_lookahead=w_lookahead,
        process_noise_enabled=process_noise,
        noise_seed=seed,
    )
    mcts_driver = _driver(mcts_env, max_steps)
    mcts_driver.reset_episode()
    planner = VCMCTSPlanner(
        VCMCTSConfig(
            n_iter=n_iter,
            top_k_dispatch=top_k_dispatch,
            top_b_reserve=top_b,
            rollout_strategy=strategy,
            rollout_max_steps=max_steps,
        )
    )
    mcts_summary = run_vc_mcts_reservation_episode(
        mcts_driver,
        planner=planner,
        max_steps=max_steps,
    )
    mcts_metrics = schedule_metrics_with_priority_wait(mcts_encoder, mcts_env)

    return {
        "seed": int(seed),
        "baseline": {**baseline_summary, **baseline_metrics},
        "oracle": {**oracle_summary, **oracle_metrics},
        "vc_mcts": {**mcts_summary, **mcts_metrics},
        "delta": {
            "oracle_o2": oracle_metrics["priority_weighted_wait"] - baseline_metrics["priority_weighted_wait"],
            "vc_mcts_o2": mcts_metrics["priority_weighted_wait"] - baseline_metrics["priority_weighted_wait"],
            "oracle_qtime": oracle_metrics["qtime_violation_count"] - baseline_metrics["qtime_violation_count"],
            "vc_mcts_qtime": mcts_metrics["qtime_violation_count"] - baseline_metrics["qtime_violation_count"],
        },
    }


def main(
    instance="small",
    seeds=1,
    strategy="FIFO",
    w_lookahead=4.0,
    top_b=2,
    top_k_dispatch=3,
    n_iter=24,
    max_steps=500,
    process_noise=False,
):
    rows = [
        run_seed(
            instance=instance,
            seed=seed,
            strategy=strategy,
            w_lookahead=w_lookahead,
            top_b=top_b,
            top_k_dispatch=top_k_dispatch,
            n_iter=n_iter,
            max_steps=max_steps,
            process_noise=process_noise,
        )
        for seed in range(int(seeds))
    ]
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return rows


def _cli():
    parser = argparse.ArgumentParser(description="VC-MCTS online reservation probe")
    parser.add_argument("--instance", choices=["small", "pressure", "late_hi"], default="small")
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--strategy", choices=["first_valid", "FIFO", "SPT", "EDD", "CR", "ATC"], default="FIFO")
    parser.add_argument("--w-lookahead", type=float, default=4.0)
    parser.add_argument("--top-b", type=int, default=2)
    parser.add_argument("--top-k-dispatch", type=int, default=3)
    parser.add_argument("--n-iter", type=int, default=24)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--noise", action="store_true")
    args = parser.parse_args()
    main(
        instance=args.instance,
        seeds=args.seeds,
        strategy=args.strategy,
        w_lookahead=args.w_lookahead,
        top_b=args.top_b,
        top_k_dispatch=args.top_k_dispatch,
        n_iter=args.n_iter,
        max_steps=args.max_steps,
        process_noise=args.noise,
    )


if __name__ == "__main__":
    _cli()
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```powershell
python -m pytest tests/test_vc_mcts_planner.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Smoke the CLI**

Run:

```powershell
python vc_mcts_probe.py --instance small --seeds 1 --strategy FIFO --top-b 1 --top-k-dispatch 2 --n-iter 2 --max-steps 200
```

Expected: prints one JSON row with `baseline`, `oracle`, and `vc_mcts`; all complete 4 lots.

- [ ] **Step 6: Commit**

```powershell
git add vc_mcts_probe.py tests/test_vc_mcts_planner.py
git commit -m "feat: add vc mcts probe cli"
```

---

### Task 8: Regression and Go/No-Go Runs

**Files:**
- Existing tests under `FAB_RL/FABenv/tests/`
- Optional output: `FAB_RL/FABenv/results/vc_mcts_late_hi_fifo.jsonl`

- [ ] **Step 1: Run full test suite**

Run from `FAB_RL/FABenv`:

```powershell
python -m pytest tests/ -q
```

Expected: all tests pass.

- [ ] **Step 2: Run existing oracle gate before judging VC-MCTS**

Run:

```powershell
python oracle_reservation_probe.py --instance late_hi --seeds 2 --strategy FIFO --top-b 3 --max-steps 600 --out results/oracle_reservation_late_hi_fifo.jsonl
```

Expected: JSON rows are written. If oracle does not improve `priority_weighted_wait` without increasing `qtime_violation_count`, stop and adjust the `late_hi` instance generator before tuning VC-MCTS.

- [ ] **Step 3: Run online VC-MCTS smoke on the same instance**

Run:

```powershell
python vc_mcts_probe.py --instance late_hi --seeds 2 --strategy FIFO --top-b 3 --top-k-dispatch 3 --n-iter 24 --max-steps 600
```

Expected: JSON prints `vc_mcts` rows with `completed_lots` equal to the instance lot count, no active reservations, and `vc_mcts` O2 no worse than baseline on at least one seed before increasing budget.

- [ ] **Step 4: Commit if regression passes**

```powershell
git add results/oracle_reservation_late_hi_fifo.jsonl
git commit -m "test: record vc mcts gate smoke"
```

---

## Self-Review

**Spec coverage:** This plan implements report §5.2 root actions, §5.7 same-source rollout, §5.8 ROP/TopB gating, §5.10 traceable root stats, and §6.2.3 stage 1-2 online planner plumbing. It intentionally does not yet implement CRN noise bundles, priority-capability robustness `rho_pc`, subtree reuse, or multi-head critic leaf evaluation.

**Placeholder scan:** No task uses placeholder language. Deferred items are explicitly out of this slice.

**Type consistency:** `VCMCTSAction`, `VCMCTSObjective`, `VCMCTSConfig`, `VCMCTSEdgeStats`, `VCMCTSDecisionTrace`, `VCMCTSPlanner`, and `run_vc_mcts_reservation_episode` are introduced before later tasks reference them.
