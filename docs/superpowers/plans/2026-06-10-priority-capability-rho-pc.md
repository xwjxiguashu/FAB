# Priority-Capability Rho PC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将报告 8 的机制二落地：把现有 `rho_pc` 简化加权产能代理升级为二部匹配对冲水位，并接入 VC-MCTS 的 UCT、trace 与消融实验。

**Architecture:** 新增独立 `priority_capability_matching.py` 负责构造窗内高优先级 future lots 与可兑现机台的二部图，并计算 `rho_pc_before/rho_pc_after/delta_rho_pc`。`vc_mcts_planner.py` 只消费该模块给出的边级估值，不把匹配逻辑塞进搜索器；ROP 仍是宽触发闸门，机制二只做树内估值。

**Tech Stack:** Python, numpy, scipy `linear_sum_assignment`, pytest, existing `FAB_RL/FABenv` bare-import modules.

---

## File Structure

- Create: `FAB_RL/FABenv/priority_capability_matching.py`
  - Owns high-priority future lot filtering, compatibility graph construction, maximum weighted bipartite matching, and action-level waterline deltas.
- Create: `FAB_RL/FABenv/tests/test_priority_capability_matching.py`
  - Focused tests for matching correctness, monotonicity, degeneration, and action deltas.
- Modify: `FAB_RL/FABenv/vc_mcts_planner.py`
  - Replace the current `reserved_compatible_capacity()`/`edge_rho_pc()` scalar proxy with matching-backed edge fields while keeping `use_rho_pc=False` behavior unchanged.
- Modify: `FAB_RL/FABenv/tests/test_vc_mcts_mechanisms.py`
  - Update existing mechanism-two tests from "reserve gets positive scalar, dispatch gets zero" to "edge records before/after/delta waterline".
- Modify: `FAB_RL/FABenv/scripts/probes/vc_mcts_probe.py`
  - Add CLI flags for `rho_pc_alpha` and high-priority threshold.
- Modify: `FAB_RL/FABenv/scripts/probes/vc_mcts_trace_summary.py`
  - Add aggregate diagnostics for `delta_rho_pc` buckets and reserve hit/waste correlation.

Do not change SAS policy, lower-layer scheduling, ROP admission rules, or reservation ledger semantics in this plan.

---

### Task 1: Add Matching Module Tests

**Files:**
- Create: `FAB_RL/FABenv/tests/test_priority_capability_matching.py`

- [x] **Step 1: Write failing tests for pure weighted matching**

Create `FAB_RL/FABenv/tests/test_priority_capability_matching.py` with this content:

```python
import pytest

from priority_capability_matching import (
    CapabilityGraph,
    CapabilityMatchEdge,
    solve_weighted_capability_matching,
)


def test_weighted_matching_prefers_larger_total_priority():
    graph = CapabilityGraph(
        machines=(1, 2),
        future_lots=(10, 11),
        lot_weights={10: 10.0, 11: 4.0},
        edges=(
            CapabilityMatchEdge(machine=1, future_lot=10),
            CapabilityMatchEdge(machine=1, future_lot=11),
            CapabilityMatchEdge(machine=2, future_lot=11),
        ),
    )

    result = solve_weighted_capability_matching(graph)

    assert result.total_weight == pytest.approx(14.0)
    assert result.normalized_waterline == pytest.approx(1.0)
    assert result.pairs == frozenset({(1, 10), (2, 11)})


def test_weighted_matching_reports_uncovered_priority_mass():
    graph = CapabilityGraph(
        machines=(1,),
        future_lots=(10, 11),
        lot_weights={10: 10.0, 11: 5.0},
        edges=(
            CapabilityMatchEdge(machine=1, future_lot=11),
        ),
    )

    result = solve_weighted_capability_matching(graph)

    assert result.total_weight == pytest.approx(5.0)
    assert result.normalized_waterline == pytest.approx(5.0 / 15.0)
    assert result.uncovered_lots == frozenset({10})


def test_empty_future_lots_has_zero_waterline():
    graph = CapabilityGraph(
        machines=(1, 2),
        future_lots=(),
        lot_weights={},
        edges=(),
    )

    result = solve_weighted_capability_matching(graph)

    assert result.total_weight == 0.0
    assert result.normalized_waterline == 0.0
    assert result.pairs == frozenset()
```

- [x] **Step 2: Run the focused test and verify it fails**

Run from `FAB_RL/FABenv`:

```powershell
python -m pytest tests/test_priority_capability_matching.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'priority_capability_matching'`.

- [x] **Step 3: Commit the failing tests**

```powershell
git add FAB_RL/FABenv/tests/test_priority_capability_matching.py
git commit -m "test: define priority capability matching behavior"
```

---

### Task 2: Implement Pure Matching Core

**Files:**
- Create: `FAB_RL/FABenv/priority_capability_matching.py`
- Test: `FAB_RL/FABenv/tests/test_priority_capability_matching.py`

- [x] **Step 1: Add matching dataclasses and solver**

Create `FAB_RL/FABenv/priority_capability_matching.py`:

```python
"""Priority-capability matching waterline for VC-MCTS mechanism 2."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class CapabilityMatchEdge:
    machine: int
    future_lot: int


@dataclass(frozen=True)
class CapabilityGraph:
    machines: tuple[int, ...]
    future_lots: tuple[int, ...]
    lot_weights: dict[int, float]
    edges: tuple[CapabilityMatchEdge, ...]


@dataclass(frozen=True)
class CapabilityMatchResult:
    total_weight: float
    normalized_waterline: float
    pairs: frozenset[tuple[int, int]]
    uncovered_lots: frozenset[int]


def solve_weighted_capability_matching(graph: CapabilityGraph) -> CapabilityMatchResult:
    machines = tuple(int(m) for m in graph.machines)
    future_lots = tuple(int(h) for h in graph.future_lots)
    total_possible = float(sum(float(graph.lot_weights.get(h, 0.0)) for h in future_lots))
    if not machines or not future_lots or total_possible <= 0.0:
        return CapabilityMatchResult(
            total_weight=0.0,
            normalized_waterline=0.0,
            pairs=frozenset(),
            uncovered_lots=frozenset(future_lots),
        )

    edge_set = {(int(edge.machine), int(edge.future_lot)) for edge in graph.edges}
    n = max(len(machines), len(future_lots))
    large_cost = 1e9
    cost = np.full((n, n), large_cost, dtype=float)

    for i, machine in enumerate(machines):
        for j, lot in enumerate(future_lots):
            if (machine, lot) in edge_set:
                cost[i, j] = -float(graph.lot_weights.get(lot, 0.0))

    rows, cols = linear_sum_assignment(cost)
    pairs = []
    total = 0.0
    for row, col in zip(rows, cols):
        if row >= len(machines) or col >= len(future_lots):
            continue
        if cost[row, col] >= large_cost:
            continue
        machine = machines[row]
        lot = future_lots[col]
        pairs.append((machine, lot))
        total += float(graph.lot_weights.get(lot, 0.0))

    covered = {lot for _machine, lot in pairs}
    uncovered = frozenset(lot for lot in future_lots if lot not in covered)
    return CapabilityMatchResult(
        total_weight=float(total),
        normalized_waterline=float(total / total_possible),
        pairs=frozenset(pairs),
        uncovered_lots=uncovered,
    )
```

- [x] **Step 2: Run the matching tests**

```powershell
python -m pytest tests/test_priority_capability_matching.py -q
```

Expected: PASS, `3 passed`.

- [x] **Step 3: Commit the solver**

```powershell
git add FAB_RL/FABenv/priority_capability_matching.py FAB_RL/FABenv/tests/test_priority_capability_matching.py
git commit -m "feat: add priority capability matching core"
```

---

### Task 3: Build Capability Graph From Environment State

**Files:**
- Modify: `FAB_RL/FABenv/priority_capability_matching.py`
- Modify: `FAB_RL/FABenv/tests/test_priority_capability_matching.py`

- [x] **Step 1: Add environment-level tests**

Append to `FAB_RL/FABenv/tests/test_priority_capability_matching.py`:

```python
from reservation_ledger import ReservationLedger
from rl_environment import ResourceCalendarEnv
from priority_capability_matching import (
    build_priority_capability_graph,
    rho_pc_state,
)


def test_build_graph_uses_only_visible_future_lots(small_encoder):
    env = ResourceCalendarEnv(small_encoder, top_k=8, w_lookahead=4.0)
    env.reset()
    graph = build_priority_capability_graph(env, ReservationLedger())

    upcoming = set(int(lot) for lot in env.upcoming_lots())

    assert set(graph.future_lots).issubset(upcoming)
    assert set(graph.machines).issubset(set(range(1, env.encoder.num_machines + 1)))


def test_w_zero_degenerates_to_zero_waterline(small_encoder):
    env = ResourceCalendarEnv(small_encoder, top_k=8, w_lookahead=0.0)
    env.reset()

    result = rho_pc_state(env, ReservationLedger())

    assert result.normalized_waterline == 0.0
    assert result.pairs == frozenset()


def test_reserved_machine_is_removed_from_free_capacity(small_encoder):
    env = ResourceCalendarEnv(small_encoder, top_k=8, w_lookahead=4.0)
    env.reset()
    upcoming = [int(lot) for lot in env.upcoming_lots()]
    assert upcoming
    machine = int(env.encoder.feasible_machines[upcoming[0]][0])

    ledger = ReservationLedger()
    ledger.reserve(
        machine=machine,
        future_lot=upcoming[0],
        eta=float(env.encoder.arrival_times[upcoming[0]]),
        created_at=float(env.current_time),
        expires_at=float(env.current_time) + 4.0,
    )

    graph = build_priority_capability_graph(env, ledger)

    assert machine not in graph.machines
```

- [x] **Step 2: Run tests to verify new functions are missing**

```powershell
python -m pytest tests/test_priority_capability_matching.py -q
```

Expected: FAIL with import errors for `build_priority_capability_graph` and `rho_pc_state`.

- [x] **Step 3: Implement graph construction using the existing env as source of truth**

Append to `FAB_RL/FABenv/priority_capability_matching.py`:

```python
def _compatible_ppids(encoder, lot: int, machine: int) -> tuple[int, ...]:
    ppids = []
    for ppid in encoder.feasible_ppids.get((int(lot), int(machine)), []):
        try:
            steps = encoder.get_process_steps(int(lot), int(machine), int(ppid))
        except (KeyError, ValueError):
            continue
        if steps:
            ppids.append(int(ppid))
    return tuple(ppids)


def _priority_threshold(env, priority_threshold):
    if priority_threshold is not None:
        return float(priority_threshold)
    priorities = getattr(env.encoder, "priorities", {})
    visible = [int(lot) for lot in env.upcoming_lots()]
    if not visible:
        return float("inf")
    values = sorted(float(priorities.get(lot, 0.0)) for lot in visible)
    return values[len(values) // 2]


def _edge_is_time_redeemable(env, lot: int, machine: int) -> bool:
    for ppid in _compatible_ppids(env.encoder, lot, machine):
        dry_run, _reason = env._dry_run_candidate(int(lot), int(machine), int(ppid))
        if dry_run is None:
            continue
        if float(dry_run.get("qtime_risk", 0.0)) <= 0.0:
            return True
    return False


def build_priority_capability_graph(
    env,
    ledger,
    priority_threshold=None,
    blocked_machines=(),
) -> CapabilityGraph:
    reserved_machines = set(int(m) for m in ledger.reserved_machines())
    blocked = set(int(m) for m in blocked_machines)
    machines = tuple(
        int(machine)
        for machine in range(1, int(env.encoder.num_machines) + 1)
        if int(machine) not in reserved_machines and int(machine) not in blocked
    )
    threshold = _priority_threshold(env, priority_threshold)
    priorities = getattr(env.encoder, "priorities", {})
    future_lots = tuple(
        int(lot)
        for lot in env.upcoming_lots()
        if float(priorities.get(int(lot), 0.0)) >= threshold
    )
    weights = {lot: float(priorities.get(lot, 0.0)) for lot in future_lots}

    edges = []
    for machine in machines:
        for lot in future_lots:
            if int(machine) not in {int(m) for m in env.encoder.feasible_machines.get(lot, [])}:
                continue
            if _edge_is_time_redeemable(env, lot, machine):
                edges.append(CapabilityMatchEdge(machine=machine, future_lot=lot))

    return CapabilityGraph(
        machines=machines,
        future_lots=future_lots,
        lot_weights=weights,
        edges=tuple(edges),
    )


def rho_pc_state(env, ledger, priority_threshold=None, blocked_machines=()):
    graph = build_priority_capability_graph(
        env,
        ledger,
        priority_threshold=priority_threshold,
        blocked_machines=blocked_machines,
    )
    return solve_weighted_capability_matching(graph)
```

- [x] **Step 4: Run the tests**

```powershell
python -m pytest tests/test_priority_capability_matching.py -q
```

Expected: PASS.

- [x] **Step 5: Commit graph construction**

```powershell
git add FAB_RL/FABenv/priority_capability_matching.py FAB_RL/FABenv/tests/test_priority_capability_matching.py
git commit -m "feat: build rho pc graph from environment state"
```

---

### Task 4: Compute Action-Level Waterline Deltas

**Files:**
- Modify: `FAB_RL/FABenv/priority_capability_matching.py`
- Modify: `FAB_RL/FABenv/tests/test_priority_capability_matching.py`

- [x] **Step 1: Add action-delta tests**

Append:

```python
from vc_mcts_planner import VCMCTSAction
from priority_capability_matching import rho_pc_for_action


def test_dispatch_does_not_increase_rho_pc(small_encoder):
    env = ResourceCalendarEnv(small_encoder, top_k=8, w_lookahead=4.0)
    env.reset()
    ledger = ReservationLedger()
    base = rho_pc_state(env, ledger)
    action = VCMCTSAction(kind="dispatch", machine=1, action_index=0, lot=1, ppid=1)

    result = rho_pc_for_action(env, ledger, action)

    assert result.before == base.normalized_waterline
    assert result.after <= result.before
    assert result.delta <= 0.0


def test_no_op_preserves_rho_pc(small_encoder):
    env = ResourceCalendarEnv(small_encoder, top_k=8, w_lookahead=4.0)
    env.reset()
    ledger = ReservationLedger()

    result = rho_pc_for_action(env, ledger, VCMCTSAction(kind="no_op", machine=1))

    assert result.after == result.before
    assert result.delta == 0.0


def test_reserve_records_forced_pair_when_feasible(small_encoder):
    env = ResourceCalendarEnv(small_encoder, top_k=8, w_lookahead=4.0)
    env.reset()
    ledger = ReservationLedger()
    upcoming = [int(lot) for lot in env.upcoming_lots()]
    assert upcoming
    lot = upcoming[0]
    machine = int(env.encoder.feasible_machines[lot][0])

    action = VCMCTSAction(
        kind="reserve",
        machine=machine,
        future_lot=lot,
        eta=float(env.encoder.arrival_times[lot]),
    )
    result = rho_pc_for_action(env, ledger, action)

    assert result.after >= result.before
    assert (machine, lot) in result.forced_pairs
```

- [x] **Step 2: Run tests to verify `rho_pc_for_action` is missing**

```powershell
python -m pytest tests/test_priority_capability_matching.py -q
```

Expected: FAIL with import error for `rho_pc_for_action`.

- [x] **Step 3: Add action-result dataclass and action evaluator**

Append to `priority_capability_matching.py`:

```python
@dataclass(frozen=True)
class RhoPcActionResult:
    before: float
    after: float
    delta: float
    before_pairs: frozenset[tuple[int, int]]
    after_pairs: frozenset[tuple[int, int]]
    forced_pairs: frozenset[tuple[int, int]]


def _solve_with_forced_pair(env, ledger, action, priority_threshold):
    if action.kind != "reserve" or action.machine is None or action.future_lot is None:
        return rho_pc_state(env, ledger, priority_threshold=priority_threshold), frozenset()

    forced = (int(action.machine), int(action.future_lot))
    graph = build_priority_capability_graph(env, ledger, priority_threshold=priority_threshold)
    edge_set = {(edge.machine, edge.future_lot) for edge in graph.edges}
    if forced not in edge_set:
        return solve_weighted_capability_matching(graph), frozenset()

    remaining_machines = tuple(m for m in graph.machines if m != forced[0])
    remaining_lots = tuple(h for h in graph.future_lots if h != forced[1])
    remaining_edges = tuple(
        edge
        for edge in graph.edges
        if edge.machine != forced[0] and edge.future_lot != forced[1]
    )
    remaining_graph = CapabilityGraph(
        machines=remaining_machines,
        future_lots=remaining_lots,
        lot_weights=graph.lot_weights,
        edges=remaining_edges,
    )
    rest = solve_weighted_capability_matching(remaining_graph)
    forced_weight = float(graph.lot_weights.get(forced[1], 0.0))
    total_possible = float(sum(graph.lot_weights.get(h, 0.0) for h in graph.future_lots))
    total = forced_weight + rest.total_weight
    normalized = 0.0 if total_possible <= 0.0 else total / total_possible
    pairs = frozenset(set(rest.pairs) | {forced})
    covered = {lot for _machine, lot in pairs}
    return CapabilityMatchResult(
        total_weight=float(total),
        normalized_waterline=float(normalized),
        pairs=pairs,
        uncovered_lots=frozenset(h for h in graph.future_lots if h not in covered),
    ), frozenset({forced})


def rho_pc_for_action(env, ledger, action, priority_threshold=None) -> RhoPcActionResult:
    before = rho_pc_state(env, ledger, priority_threshold=priority_threshold)
    if action.kind in ("dispatch", "delegate_dispatch") and action.machine is not None:
        after = rho_pc_state(
            env,
            ledger,
            priority_threshold=priority_threshold,
            blocked_machines=(int(action.machine),),
        )
        forced = frozenset()
    elif action.kind == "reserve":
        after, forced = _solve_with_forced_pair(env, ledger, action, priority_threshold)
    else:
        after = before
        forced = frozenset()

    return RhoPcActionResult(
        before=float(before.normalized_waterline),
        after=float(after.normalized_waterline),
        delta=float(after.normalized_waterline - before.normalized_waterline),
        before_pairs=before.pairs,
        after_pairs=after.pairs,
        forced_pairs=forced,
    )
```

- [x] **Step 4: Run tests**

```powershell
python -m pytest tests/test_priority_capability_matching.py -q
```

Expected: PASS.

- [x] **Step 5: Commit action deltas**

```powershell
git add FAB_RL/FABenv/priority_capability_matching.py FAB_RL/FABenv/tests/test_priority_capability_matching.py
git commit -m "feat: compute action rho pc waterline deltas"
```

---

### Task 5: Integrate Matching-Backed Rho PC Into VC-MCTS Trace

**Files:**
- Modify: `FAB_RL/FABenv/vc_mcts_planner.py`
- Modify: `FAB_RL/FABenv/tests/test_vc_mcts_mechanisms.py`

- [x] **Step 1: Update mechanism tests for trace fields**

In `FAB_RL/FABenv/tests/test_vc_mcts_mechanisms.py`, replace the old mechanism-two assertions that require dispatch to have `0.0` rho with:

```python
def test_planner_populates_matching_rho_pc_fields_when_enabled(small_encoder):
    env = ResourceCalendarEnv(small_encoder, top_k=8, w_lookahead=4.0)
    driver = _driver(env)
    driver.reset_episode()
    ledger = ReservationLedger()

    planner = VCMCTSPlanner(
        VCMCTSConfig(n_iter=1, top_k_dispatch=1, top_b_reserve=1, use_rho_pc=True)
    )
    trace = planner.plan(driver, ledger, machine=1)

    edge_dicts = [edge.to_dict() for edge in trace.edge_stats]
    assert all("rho_pc_before" in item for item in edge_dicts)
    assert all("rho_pc_after" in item for item in edge_dicts)
    assert all("delta_rho_pc" in item for item in edge_dicts)
    assert any(item["delta_rho_pc"] >= 0.0 for item in edge_dicts if item["kind"] == "reserve")
```

- [x] **Step 2: Run tests to capture the missing trace fields**

```powershell
python -m pytest tests/test_vc_mcts_mechanisms.py::test_planner_populates_matching_rho_pc_fields_when_enabled -q
```

Expected: FAIL because `rho_pc_before`, `rho_pc_after`, and `delta_rho_pc` are not emitted.

- [x] **Step 3: Modify `VCMCTSConfig` and `VCMCTSEdgeStats`**

In `FAB_RL/FABenv/vc_mcts_planner.py`, add config fields:

```python
    rho_pc_alpha: float = 1.0
    rho_pc_priority_threshold: float | None = None
```

Replace `rho_pc: float = 0.0` in `VCMCTSEdgeStats` with:

```python
    rho_pc_before: float = 0.0
    rho_pc_after: float = 0.0
    delta_rho_pc: float = 0.0
```

Update `to_dict()` to include:

```python
            "rho_pc": float(self.rho_pc_after),
            "rho_pc_before": float(self.rho_pc_before),
            "rho_pc_after": float(self.rho_pc_after),
            "delta_rho_pc": float(self.delta_rho_pc),
```

- [x] **Step 4: Populate edge fields from `rho_pc_for_action`**

At the top of `vc_mcts_planner.py`, import:

```python
from priority_capability_matching import rho_pc_for_action
```

In `VCMCTSPlanner.plan()`, replace the existing `edge.rho_pc = edge_rho_pc(...)` loop with:

```python
        if self.config.use_rho_pc:
            for edge in edges:
                rho = rho_pc_for_action(
                    driver.env,
                    ledger,
                    edge.action,
                    priority_threshold=self.config.rho_pc_priority_threshold,
                )
                edge.rho_pc_before = rho.before
                edge.rho_pc_after = rho.after
                edge.delta_rho_pc = rho.delta
```

- [x] **Step 5: Run mechanism trace test**

```powershell
python -m pytest tests/test_vc_mcts_mechanisms.py::test_planner_populates_matching_rho_pc_fields_when_enabled -q
```

Expected: PASS.

- [x] **Step 6: Commit trace integration**

```powershell
git add FAB_RL/FABenv/vc_mcts_planner.py FAB_RL/FABenv/tests/test_vc_mcts_mechanisms.py
git commit -m "feat: expose matching rho pc fields in vc mcts traces"
```

---

### Task 6: Use Alpha Interpolation In UCT Selection

**Files:**
- Modify: `FAB_RL/FABenv/vc_mcts_planner.py`
- Modify: `FAB_RL/FABenv/tests/test_vc_mcts_mechanisms.py`

- [x] **Step 1: Add a UCT bias test using injected evaluator**

Replace the existing rho bias test with:

```python
def test_rho_pc_alpha_biases_visits_toward_positive_delta_edges(small_encoder):
    env = ResourceCalendarEnv(small_encoder, top_k=8, w_lookahead=4.0)
    driver = _driver(env)
    driver.reset_episode()
    ledger = ReservationLedger()

    def evaluator(_driver, _ledger, _action, _config):
        return VCMCTSObjective(
            qtime_violation_count=0.0,
            qtime_violation_total=0.0,
            priority_weighted_wait=5.0,
            avg_utilization=0.5,
        )

    planner = VCMCTSPlanner(
        VCMCTSConfig(
            n_iter=16,
            top_k_dispatch=1,
            top_b_reserve=2,
            use_rho_pc=True,
            rho_pc_alpha=0.0,
        ),
        rollout_evaluator=evaluator,
    )
    trace = planner.plan(driver, ledger, machine=1)

    positive_delta_edges = [edge for edge in trace.edge_stats if edge.delta_rho_pc > 0.0]
    non_positive_edges = [edge for edge in trace.edge_stats if edge.delta_rho_pc <= 0.0]
    assert positive_delta_edges
    assert max(edge.visits for edge in positive_delta_edges) >= max(edge.visits for edge in non_positive_edges)
```

- [x] **Step 2: Run the test**

```powershell
python -m pytest tests/test_vc_mcts_mechanisms.py::test_rho_pc_alpha_biases_visits_toward_positive_delta_edges -q
```

Expected: FAIL until `_select_edge()` uses `rho_pc_alpha`.

- [x] **Step 3: Add normalized exploitation helper**

In `VCMCTSPlanner._select_edge()`, before `def uct(edge):`, compute normalized q scores:

```python
        raw_scores = {
            id(edge): objective_to_score(edge.mean_objective, self.config)
            for edge in edges
            if edge.mean_objective is not None
        }
        if raw_scores:
            min_score = min(raw_scores.values())
            max_score = max(raw_scores.values())
            score_span = max(max_score - min_score, 1e-9)
        else:
            min_score = 0.0
            score_span = 1.0
```

Replace the exploitation block with:

```python
            raw_score = objective_to_score(mean, self.config)
            if self.config.use_rho_pc:
                q_hat = (raw_score - min_score) / score_span
                alpha = min(1.0, max(0.0, float(self.config.rho_pc_alpha)))
                exploitation = (
                    alpha * float(q_hat)
                    + (1.0 - alpha) * float(edge.rho_pc_after)
                )
                exploitation += float(self.config.rho_pc_weight) * float(edge.delta_rho_pc)
            else:
                exploitation = raw_score
```

This preserves old behavior when `use_rho_pc=False`. `rho_pc_weight` remains as an additive compatibility knob for old probes; new experiments should sweep `rho_pc_alpha`.

- [x] **Step 4: Run the focused UCT test**

```powershell
python -m pytest tests/test_vc_mcts_mechanisms.py::test_rho_pc_alpha_biases_visits_toward_positive_delta_edges -q
```

Expected: PASS.

- [x] **Step 5: Run all mechanism tests**

```powershell
python -m pytest tests/test_vc_mcts_mechanisms.py -q
```

Expected: PASS.

- [x] **Step 6: Commit UCT interpolation**

```powershell
git add FAB_RL/FABenv/vc_mcts_planner.py FAB_RL/FABenv/tests/test_vc_mcts_mechanisms.py
git commit -m "feat: guide vc mcts with alpha-interpolated rho pc"
```

---

### Task 7: Wire Probe CLI Flags

**Files:**
- Modify: `FAB_RL/FABenv/scripts/probes/vc_mcts_probe.py`

- [x] **Step 1: Add CLI arguments**

In `parse_args()` add:

```python
    parser.add_argument("--rho-pc-alpha", type=float, default=1.0)
    parser.add_argument("--rho-pc-priority-threshold", type=float, default=None)
```

- [x] **Step 2: Thread values through `run_seed()`**

Add parameters to `run_seed()`:

```python
    rho_pc_alpha=1.0,
    rho_pc_priority_threshold=None,
```

Pass them into `VCMCTSConfig`:

```python
            rho_pc_alpha=rho_pc_alpha,
            rho_pc_priority_threshold=rho_pc_priority_threshold,
```

When building worker argument dictionaries, pass:

```python
        rho_pc_alpha=args.rho_pc_alpha,
        rho_pc_priority_threshold=args.rho_pc_priority_threshold,
```

- [x] **Step 3: Smoke the CLI help**

```powershell
python scripts/probes/vc_mcts_probe.py --help
```

Expected: output contains `--rho-pc-alpha` and `--rho-pc-priority-threshold`.

- [x] **Step 4: Commit CLI wiring**

```powershell
git add FAB_RL/FABenv/scripts/probes/vc_mcts_probe.py
git commit -m "feat: expose rho pc alpha in vc mcts probe"
```

---

### Task 8: Extend Trace Summary With Delta Rho Buckets

**Files:**
- Modify: `FAB_RL/FABenv/scripts/probes/vc_mcts_trace_summary.py`
- Test: `FAB_RL/FABenv/tests/test_vc_mcts_trace_summary.py`

- [x] **Step 1: Add trace summary test**

Append to `FAB_RL/FABenv/tests/test_vc_mcts_trace_summary.py`:

```python
import json

from vc_mcts_trace_summary import summarize_trace_file


def test_trace_summary_reports_delta_rho_pc_buckets(tmp_path):
    path = tmp_path / "trace.jsonl"
    rows = [
        {
            "selected_action": {"kind": "reserve", "future_lot": 3},
            "edges": [
                {"kind": "reserve", "delta_rho_pc": 0.50, "mean_o2": 8.0},
                {"kind": "delegate_dispatch", "delta_rho_pc": -0.25, "mean_o2": 10.0},
            ],
            "diagnostics": {"reserve_was_available": True, "reserve_was_selected": True},
        },
        {
            "selected_action": {"kind": "delegate_dispatch"},
            "edges": [
                {"kind": "reserve", "delta_rho_pc": 0.10, "mean_o2": 11.0},
                {"kind": "delegate_dispatch", "delta_rho_pc": 0.0, "mean_o2": 9.0},
            ],
            "diagnostics": {"reserve_was_available": True, "reserve_was_selected": False},
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    summary = summarize_trace_file(path)

    assert summary["rho_pc_edge_count"] == 4
    assert summary["rho_pc_positive_delta_edges"] == 2
    assert summary["rho_pc_selected_reserve_delta_avg"] == 0.5
```

- [x] **Step 2: Run summary test**

```powershell
python -m pytest tests/test_vc_mcts_trace_summary.py::test_trace_summary_reports_delta_rho_pc_buckets -q
```

Expected: FAIL because summary fields are missing.

- [x] **Step 3: Add summary aggregation**

In `summarize_trace_file()`, collect `delta_rho_pc` from every edge:

```python
    rho_deltas = []
    positive_delta_edges = 0
    selected_reserve_deltas = []
```

Inside the per-row loop:

```python
        selected_kind = row.get("selected_action", {}).get("kind")
        for edge in row.get("edges", []):
            if "delta_rho_pc" not in edge:
                continue
            delta = float(edge["delta_rho_pc"])
            rho_deltas.append(delta)
            if delta > 0.0:
                positive_delta_edges += 1
            if selected_kind == "reserve" and edge.get("kind") == "reserve":
                selected_reserve_deltas.append(delta)
```

Before returning summary:

```python
    summary["rho_pc_edge_count"] = int(len(rho_deltas))
    summary["rho_pc_positive_delta_edges"] = int(positive_delta_edges)
    summary["rho_pc_delta_avg"] = None if not rho_deltas else float(sum(rho_deltas) / len(rho_deltas))
    summary["rho_pc_selected_reserve_delta_avg"] = (
        None
        if not selected_reserve_deltas
        else float(sum(selected_reserve_deltas) / len(selected_reserve_deltas))
    )
```

- [x] **Step 4: Run trace summary tests**

```powershell
python -m pytest tests/test_vc_mcts_trace_summary.py -q
```

Expected: PASS.

- [x] **Step 5: Commit trace summary**

```powershell
git add FAB_RL/FABenv/scripts/probes/vc_mcts_trace_summary.py FAB_RL/FABenv/tests/test_vc_mcts_trace_summary.py
git commit -m "feat: summarize rho pc trace diagnostics"
```

---

### Task 9: End-to-End Verification

**Files:**
- No new files.

- [x] **Step 1: Run focused tests**

```powershell
python -m pytest tests/test_priority_capability_matching.py tests/test_vc_mcts_mechanisms.py tests/test_vc_mcts_trace_summary.py -q
```

Expected: PASS.

- [x] **Step 2: Run VC-MCTS planner stack**

```powershell
python -m pytest tests/test_reservation_ledger.py tests/test_reservation_rop.py tests/test_reservation_simulator.py tests/test_vc_mcts_planner.py tests/test_dispatch_delegate.py -q
```

Expected: PASS.

- [x] **Step 3: Run a small probe with mechanism two on**

```powershell
python scripts/probes/vc_mcts_probe.py --instance late_hi --seeds 1 --strategy FIFO --skip-oracle --n-iter 4 --top-b 1 --dispatch-delegate rule --use-rho-pc --rho-pc-alpha 0.6 --trace-out artifacts/results/rho_pc_smoke_trace.jsonl --trace-summary-out artifacts/results/rho_pc_smoke_summary.json
```

Expected: process exits 0, summary JSON contains `rho_pc_edge_count > 0`.

- [x] **Step 4: Run root structure test from repo root**

Run from `E:\code\FAB`:

```powershell
python -m pytest tests/ -q
```

Expected: PASS. If the structure test fails because a new script path must be whitelisted, update `tests/test_project_structure.py` with the exact new script path and rerun.

- [x] **Step 5: Commit verification-only structure changes if needed**

```powershell
git add tests/test_project_structure.py
git commit -m "test: allow rho pc probe paths"
```

Only run this commit if Step 4 required a structure-test update.

---

### Task 10: Alpha Scan And Delta-Rho Ablation

**Files:**
- Create: `FAB_RL/FABenv/scripts/probes/rho_pc_ablation.py`
- Create: `FAB_RL/FABenv/artifacts/results/rho_pc_ablation_README.md`

- [x] **Step 1: Add a narrow ablation script**

Create `FAB_RL/FABenv/scripts/probes/rho_pc_ablation.py`:

```python
"""Run mechanism-2 alpha scan for VC-MCTS rho_pc."""

from pathlib import Path
import json
import subprocess
import sys


FABENV_ROOT = Path(__file__).resolve().parents[2]
PROBE = FABENV_ROOT / "scripts" / "probes" / "vc_mcts_probe.py"
OUT_DIR = FABENV_ROOT / "artifacts" / "results" / "rho_pc_ablation"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for alpha in (1.0, 0.8, 0.6, 0.4):
        summary_path = OUT_DIR / f"late_hi_alpha_{alpha:.1f}_summary.json"
        trace_path = OUT_DIR / f"late_hi_alpha_{alpha:.1f}_trace.jsonl"
        cmd = [
            sys.executable,
            str(PROBE),
            "--instance",
            "late_hi",
            "--seeds",
            "3",
            "--strategy",
            "FIFO",
            "--skip-oracle",
            "--n-iter",
            "8",
            "--top-b",
            "2",
            "--dispatch-delegate",
            "rule",
            "--use-rho-pc",
            "--rho-pc-alpha",
            str(alpha),
            "--trace-out",
            str(trace_path),
            "--trace-summary-out",
            str(summary_path),
        ]
        subprocess.run(cmd, cwd=str(FABENV_ROOT), check=True)
        rows.append({"alpha": alpha, "summary": str(summary_path), "trace": str(trace_path)})
    manifest = OUT_DIR / "manifest.json"
    manifest.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(str(manifest))


if __name__ == "__main__":
    main()
```

- [x] **Step 2: Add result README**

Create `FAB_RL/FABenv/artifacts/results/rho_pc_ablation_README.md`:

```markdown
# Rho PC Ablation

This folder stores mechanism-two alpha-scan smoke outputs.

Primary knobs:
- `rho_pc_alpha=1.0`: current qhat-only behavior.
- `rho_pc_alpha<1.0`: interpolates UCT exploitation toward matching-backed `rho_pc_after`.

Primary diagnostics:
- O2 and Q-time from `vc_mcts_probe.py`.
- `rho_pc_positive_delta_edges` and `rho_pc_selected_reserve_delta_avg` from trace summaries.
- Reserve hit/waste diagnostics from existing trace summary fields.
```

- [x] **Step 3: Run the ablation smoke**

```powershell
python scripts/probes/rho_pc_ablation.py
```

Expected: exits 0 and prints `artifacts\results\rho_pc_ablation\manifest.json`.

- [x] **Step 4: Commit ablation tooling**

```powershell
git add FAB_RL/FABenv/scripts/probes/rho_pc_ablation.py FAB_RL/FABenv/artifacts/results/rho_pc_ablation_README.md
git commit -m "chore: add rho pc alpha ablation probe"
```

---

## Self-Review

Spec coverage:
- Report 8 says mechanism two should move from DyRo-style scalar/free-capacity proxy to bipartite matching waterline. Tasks 1-4 implement the matching waterline.
- Report 8 says UCT should use `E(s,a)=feasible·[alpha·qhat+(1-alpha)·rho_pc]-lambda*c_qt`. Task 6 implements the alpha interpolation while leaving online lambda as future mechanism-one consistency work.
- Report 8 says trace should expose mechanism-two fields and support alpha scan plus delta-rho bucket ablation. Tasks 5, 8, and 10 cover those deliverables.
- Report 8 says ROP remains a gate and should not be replaced. This plan leaves `reservation_rop.py` unchanged.

Placeholder scan:
- The plan has no unresolved placeholder markers and every verification step names an exact command.

Type consistency:
- `RhoPcActionResult.before/after/delta` maps to `VCMCTSEdgeStats.rho_pc_before/rho_pc_after/delta_rho_pc`.
- `rho_pc_alpha` is threaded from probe CLI to `VCMCTSConfig` to `_select_edge()`.
- `rho_pc_priority_threshold` is threaded to `rho_pc_for_action()` and then graph construction.

---

Plan complete and saved to `docs/superpowers/plans/2026-06-10-priority-capability-rho-pc.md`. Two execution options:

**1. Subagent-Driven (recommended)** - dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** - execute tasks in this session using executing-plans, batch execution with checkpoints.
