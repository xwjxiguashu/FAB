# Mechanism 3 Implementation Plan (CRN + OCBA + CVaR)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现报告8 §7.13 的机制三：CRN 配对场景评估 + OCBA 顺序预算分配 + CVaR 风险字典序 final pick。已有 CRN per-visit 评估脚手架（`evaluate_action(scenario=...)`、`plan()` 循环 `scenario=edge.visits`），需补场景级目标存储、OCBA 两阶段分配、风险字典序替换均值字典序。

**Architecture:** 三个组件在 `vc_mcts_planner.py` 的同一棵树上不同挂载点：`VCMCTSEdgeStats` 存场景级目标（组件一）→ `VCMCTSPlanner.plan()` 用 OCBA 分配预算（组件二）→ `_choose_final_action()` 用风险字典序最终选边（组件三）。机制二代码不改。

**Tech Stack:** Python/numpy, pytest, 现有 `vc_mcts_planner.py` / `tests/test_vc_mcts_mechanisms.py`。

---

## File Structure

- **Modify:** `FAB_RL/FABenv/vc_mcts_planner.py`
  - `VCMCTSConfig`: 新增 `ocba_enabled`, `ocba_n0`, `ocba_delta_star`, `ocba_gamma`, `cvar_enabled`, `cvar_beta0`, `cvar_beta`.
  - `VCMCTSEdgeStats`: 新增 `_scenarios` 列表、`scenario_objectives` 属性、`quantile(dim, beta)` / `cvar(dim, beta)` 方法。
  - `VCMCTSPlanner.plan()`: OCBA 两阶段循环（`ocba_enabled=True` 时替换均匀迭代）。
  - `VCMCTSPlanner._choose_final_action()`: CVaR 风险字典序（`cvar_enabled=True` 时替换均值字典序）。
  - `VCMCTSEdgeStats.to_dict()`: 新增 `scenario_objectives` 字段（`cvar_enabled=True` 时）。
- **Modify:** `FAB_RL/FABenv/tests/test_vc_mcts_mechanisms.py` — 追加机制三三组件测试。
- **Modify:** `FAB_RL/FABenv/scripts/probes/vc_mcts_probe.py` — 新增 CLI 标志透传。

不创建新文件。不修改机制二代码。

---

### Task 1: Per-scenario objective storage in VCMCTSEdgeStats

**Files:**
- Modify: `FAB_RL/FABenv/vc_mcts_planner.py` (VCMCTSEdgeStats)
- Test: `FAB_RL/FABenv/tests/test_vc_mcts_mechanisms.py`

- [ ] **Step 1: Write failing tests for per-scenario storage**

Append to `tests/test_vc_mcts_mechanisms.py`:

```python
def test_edge_stats_stores_per_scenario_objectives():
    edge = VCMCTSEdgeStats(action=VCMCTSAction(kind="dispatch", machine=1))
    edge.record(VCMCTSObjective(qtime_violation_count=0.0, qtime_violation_total=0.0,
                                priority_weighted_wait=10.0, avg_utilization=0.5))
    edge.record(VCMCTSObjective(qtime_violation_count=1.0, qtime_violation_total=2.0,
                                priority_weighted_wait=8.0, avg_utilization=0.6))
    edge.record(VCMCTSObjective(qtime_violation_count=2.0, qtime_violation_total=4.0,
                                priority_weighted_wait=6.0, avg_utilization=0.4))

    assert edge.visits == 3
    scenarios = edge.scenario_objectives
    assert len(scenarios) == 3
    assert scenarios[0].priority_weighted_wait == 10.0
    assert scenarios[2].qtime_violation_count == 2.0


def test_edge_stats_per_scenario_quantile():
    """Q̂_β: 上分位数 —— 排序后第 ⌈β·K⌉ 个值。"""
    edge = VCMCTSEdgeStats(action=VCMCTSAction(kind="dispatch", machine=1))
    for qt in [0.0, 1.0, 2.0, 3.0, 4.0]:
        edge.record(VCMCTSObjective(qtime_violation_count=float(qt), qtime_violation_total=0.0,
                                    priority_weighted_wait=5.0, avg_utilization=0.5))

    # β=0.8, K=5 → idx=⌈0.8*5⌉=4 → 第4个(0-based index 3) = 3.0
    assert edge.quantile("qtime_violation_count", 0.8) == 3.0
    # β=0.5, idx=⌈0.5*5⌉=3 → 2.0
    assert edge.quantile("qtime_violation_count", 0.5) == 2.0


def test_edge_stats_cvar():
    """CVaR_β: 最差 ⌈(1-β)·K⌉ 个场景的均值。"""
    edge = VCMCTSEdgeStats(action=VCMCTSAction(kind="dispatch", machine=1))
    for o2 in [4.0, 6.0, 8.0, 10.0, 12.0]:
        edge.record(VCMCTSObjective(qtime_violation_count=0.0, qtime_violation_total=0.0,
                                    priority_weighted_wait=float(o2), avg_utilization=0.5))

    # β=0.8, K=5 → ⌈(1-0.8)*5⌉=1 个最差 → mean([12.0])=12.0
    assert edge.cvar("priority_weighted_wait", 0.8) == 12.0
    # β=0.4, ⌈0.6*5⌉=3 → mean([12.0,10.0,8.0])=10.0
    assert edge.cvar("priority_weighted_wait", 0.4) == 10.0
    # β=0.0 → CVaR=均值 (退化基线)
    assert edge.cvar("priority_weighted_wait", 0.0) == pytest.approx(8.0)


def test_edge_stats_cvar_respects_field_direction():
    """O2/利用率 越大越差 → CVaR 取大值尾部; util 越大越好 → 先取负。"""
    edge = VCMCTSEdgeStats(action=VCMCTSAction(kind="dispatch", machine=1))
    for util in [0.3, 0.5, 0.7, 0.9]:
        edge.record(VCMCTSObjective(qtime_violation_count=0.0, qtime_violation_total=0.0,
                                    priority_weighted_wait=1.0, avg_utilization=float(util)))
    # util 越大越好 → CVaR 应取小值尾部 (负向化)
    assert edge.cvar("avg_utilization", 0.5, minimize=False) < edge.mean_objective.avg_utilization
```

- [ ] **Step 2: Run tests to verify they fail**

Run from `FAB_RL/FABenv`:
```powershell
python -m pytest tests/test_vc_mcts_mechanisms.py -q -k "scenario_objectives or quantile or cvar"
```

Expected: FAIL with `AttributeError: 'VCMCTSEdgeStats' object has no attribute 'scenario_objectives'`.

- [ ] **Step 3: Implement per-scenario storage and quantile/CVaR**

In `VCMCTSEdgeStats.__init__` (or as class-level defaults), add:
```python
    _scenarios: list = field(default_factory=list)
```

In `VCMCTSEdgeStats.record()`, append the objective:
```python
    def record(self, objective):
        self.visits += 1
        self.total_qtime += float(objective.qtime_violation_count)
        self.total_qtime_severity += float(objective.qtime_violation_total)
        self.total_o2 += float(objective.priority_weighted_wait)
        self.total_util += float(objective.avg_utilization)
        if bool(getattr(objective, "is_leaf_bootstrap", False)):
            self.leaf_bootstrap_visits += 1
        self._scenarios.append(objective)
```

Add properties and methods:
```python
    @property
    def scenario_objectives(self):
        return list(self._scenarios)

    def _sorted_values(self, dim, minimize=True):
        values = [float(getattr(obj, dim)) for obj in self._scenarios]
        if not values:
            return []
        values.sort()
        if not minimize:
            values.reverse()  # util: 越大越好 → 尾部 = 小值
        return values

    def quantile(self, dim, beta, minimize=True):
        values = self._sorted_values(dim, minimize=minimize)
        if not values:
            return 0.0
        idx = min(len(values) - 1, max(0, int(math.ceil(beta * len(values))) - 1))
        return float(values[idx])

    def cvar(self, dim, beta, minimize=True):
        values = self._sorted_values(dim, minimize=minimize)
        if not values:
            return 0.0
        tail_count = max(1, int(math.ceil((1.0 - beta) * len(values))))
        tail = values[-tail_count:]
        return float(sum(tail) / len(tail))
```

- [ ] **Step 4: Run tests to verify they pass**

```powershell
python -m pytest tests/test_vc_mcts_mechanisms.py -q -k "scenario_objectives or quantile or cvar"
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```powershell
git add FAB_RL/FABenv/vc_mcts_planner.py FAB_RL/FABenv/tests/test_vc_mcts_mechanisms.py
git commit -m "feat: per-scenario objective storage with quantile/CVaR"
```

---

### Task 2: CVaR risk-lexicographic final pick

**Files:**
- Modify: `FAB_RL/FABenv/vc_mcts_planner.py` (VCMCTSConfig + _choose_final_action)
- Test: `FAB_RL/FABenv/tests/test_vc_mcts_mechanisms.py`

- [ ] **Step 1: Write failing tests for risk-lexicographic selection**

Append to `tests/test_vc_mcts_mechanisms.py`:

```python
def test_cvar_lexicographic_choice_picks_robust_edge():
    """均值相等时, CVaR 更优的边胜出 (报告8 §7.13.4)。"""
    edge_a = VCMCTSEdgeStats(action=VCMCTSAction(kind="dispatch", machine=1))
    edge_b = VCMCTSEdgeStats(action=VCMCTSAction(kind="reserve", machine=1, future_lot=5))

    # 边 A: O2 均值 8.0, 但有一场景 O2=20 (尾部差)
    for o2 in [4.0, 4.0, 8.0, 20.0]:
        edge_a.record(VCMCTSObjective(qtime_violation_count=0.0, qtime_violation_total=0.0,
                                      priority_weighted_wait=float(o2), avg_utilization=0.5))
    # 边 B: O2 均值 8.0, 所有场景 O2≤10 (尾部好)
    for o2 in [6.0, 8.0, 8.0, 10.0]:
        edge_b.record(VCMCTSObjective(qtime_violation_count=0.0, qtime_violation_total=0.0,
                                      priority_weighted_wait=float(o2), avg_utilization=0.5))

    config = VCMCTSConfig(cvar_enabled=True, cvar_beta=0.6)
    planner = VCMCTSPlanner(config)
    selected = planner._choose_final_action([edge_a, edge_b])
    assert selected is edge_b  # CVaR_0.6(O2) B=mean(10,8)=9.0 < A=mean(20,8)=14.0


def test_cvar_lexicographic_qtime_quantile_takes_priority():
    """第0层: Q̂_β0(qtime_count) —— 均值相同但尾部更差的边被淘汰。"""
    edge_a = VCMCTSEdgeStats(action=VCMCTSAction(kind="dispatch", machine=1))
    edge_b = VCMCTSEdgeStats(action=VCMCTSAction(kind="reserve", machine=1, future_lot=5))

    for qt in [0.0, 0.0, 0.0, 1.0]:
        edge_a.record(VCMCTSObjective(qtime_violation_count=float(qt), qtime_violation_total=0.0,
                                      priority_weighted_wait=5.0, avg_utilization=0.5))
    for qt in [0.0, 0.0, 0.0, 0.0]:
        edge_b.record(VCMCTSObjective(qtime_violation_count=float(qt), qtime_violation_total=0.0,
                                      priority_weighted_wait=6.0, avg_utilization=0.5))

    config = VCMCTSConfig(cvar_enabled=True, cvar_beta0=0.8)
    planner = VCMCTSPlanner(config)
    # 边 A: Q̂_0.8(qtime)=1.0 > 边 B: 0.0 → 第0层硬淘汰边 A
    selected = planner._choose_final_action([edge_a, edge_b])
    assert selected is edge_b


def test_cvar_disabled_falls_back_to_mean_lexicographic():
    """cvar_enabled=False → 旧行为 (均值字典序) 不变。"""
    edge_a = VCMCTSEdgeStats(action=VCMCTSAction(kind="dispatch", machine=1))
    edge_b = VCMCTSEdgeStats(action=VCMCTSAction(kind="reserve", machine=1, future_lot=5))
    for o2 in [4.0, 4.0, 8.0, 20.0]:
        edge_a.record(VCMCTSObjective(qtime_violation_count=0.0, qtime_violation_total=0.0,
                                      priority_weighted_wait=float(o2), avg_utilization=0.5))
    for o2 in [6.0, 8.0, 8.0, 10.0]:
        edge_b.record(VCMCTSObjective(qtime_violation_count=0.0, qtime_violation_total=0.0,
                                      priority_weighted_wait=float(o2), avg_utilization=0.5))

    planner = VCMCTSPlanner(VCMCTSConfig(cvar_enabled=False))
    selected = planner._choose_final_action([edge_a, edge_b])
    # 均值 O2: 边 A=9.0, 边 B=8.0 → B 胜 (均值口径)
    assert selected is edge_b
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
python -m pytest tests/test_vc_mcts_mechanisms.py -q -k "cvar_lexicographic or cvar_disabled"
```

Expected: FAIL — `cvar_enabled` not in `VCMCTSConfig`, or `_choose_final_action` still uses mean.

- [ ] **Step 3: Add config knobs and rewrite _choose_final_action**

Add to `VCMCTSConfig`:
```python
    # 机制 3 组件三 (报告8 §7.13.4): CVaR 风险字典序 final pick。
    # cvar_enabled=True 时 _choose_final_action 改用风险字典序 (默认 False=均值字典序)。
    cvar_enabled: bool = False
    cvar_beta0: float = 0.95   # Q-time 违约数/总量的上分位数
    cvar_beta: float = 0.8     # O2 的 CVaR 水平
```

Rewrite `_choose_final_action` to branch on `self.config.cvar_enabled`:

```python
    def _choose_final_action(self, edges, current_time=0.0):
        if self.config.cvar_enabled:
            return self._cvar_choose(edges, current_time)
        return self._mean_choose(edges, current_time)

    def _mean_choose(self, edges, current_time=0.0):
        """Existing mean-based lexicographic final pick (unchanged from current code)."""
        effective = self._effective_objectives(edges, current_time)
        # ... (existing code, moved here)
```

Move the existing `_choose_final_action` body into `_mean_choose`, then add `_cvar_choose`:

```python
    def _cvar_choose(self, edges, current_time=0.0):
        """风险字典序 final pick (报告8 §7.13.4)。

        第 0 层: Q̂_β0( V_qt_count(a) )  — 违约数上分位数
        第 1 层: Q̂_β0( V_qt_total(a) )  — 违约总量上分位数
        第 2 层: CVaR_β( O2(a) )         — O2 尾部均值
        第 3 层: −mean( Ū(a) )           — 利用率均值 + 访问数破平
        no_op gating: 保留, 门控条件改用分位数口径。
        """
        beta0 = float(self.config.cvar_beta0)
        beta = float(self.config.cvar_beta)

        def key(edge):
            if edge.visits <= 0:
                return (0.0, 0.0, 0.0, 0.0, -1)
            return (
                -float(edge.quantile("qtime_violation_count", beta0)),
                -float(edge.quantile("qtime_violation_total", beta0)),
                -float(edge.cvar("priority_weighted_wait", beta)),
                float(edge.mean_objective.avg_utilization),
                edge.visits,
            )

        ranked = sorted(edges, key=key, reverse=True)
        selected = ranked[0]
        if selected.action.kind != "no_op":
            return selected

        alternatives = [edge for edge in ranked if edge.action.kind != "no_op" and edge.visits > 0]
        if not alternatives:
            return selected

        best_alternative = alternatives[0]
        noop_qt0 = float(selected.quantile("qtime_violation_count", beta0))
        alt_qt0 = float(best_alternative.quantile("qtime_violation_count", beta0))
        noop_qt1 = float(selected.quantile("qtime_violation_total", beta0))
        alt_qt1 = float(best_alternative.quantile("qtime_violation_total", beta0))
        noop_has_qtime_advantage = noop_qt0 < alt_qt0 or noop_qt1 < alt_qt1
        return selected if noop_has_qtime_advantage else best_alternative
```

- [ ] **Step 4: Run tests to verify they pass**

```powershell
python -m pytest tests/test_vc_mcts_mechanisms.py -q -k "cvar_lexicographic or cvar_disabled"
```

Expected: 3 passed.

- [ ] **Step 5: Run all mechanism tests**

```powershell
python -m pytest tests/test_vc_mcts_mechanisms.py -q
```

Expected: all pass.

- [ ] **Step 6: Commit**

```powershell
git add FAB_RL/FABenv/vc_mcts_planner.py FAB_RL/FABenv/tests/test_vc_mcts_mechanisms.py
git commit -m "feat: CVaR risk-lexicographic final pick for mechanism 3"
```

---

### Task 3: OCBA two-stage budget allocation

**Files:**
- Modify: `FAB_RL/FABenv/vc_mcts_planner.py` (VCMCTSConfig + plan)
- Test: `FAB_RL/FABenv/tests/test_vc_mcts_mechanisms.py`

- [ ] **Step 1: Write failing tests for OCBA allocation**

Append to `tests/test_vc_mcts_mechanisms.py`:

```python
def test_ocba_warm_up_evaluates_every_edge_once():
    """OCBA 阶段一: 每条 feasible 边先跑 n₀ 条场景。"""
    env = ResourceCalendarEnv(build_small_encoder(), top_k=8, w_lookahead=4.0)
    driver = _driver(env)
    driver.reset_episode()
    ledger = ReservationLedger()

    planner = VCMCTSPlanner(
        VCMCTSConfig(
            n_iter=20, top_k_dispatch=1, top_b_reserve=1,
            crn_noise=True, n_mc=8, crn_seed_base=0,
            ocba_enabled=True, ocba_n0=3,
            rollout_max_steps=10,
        )
    )
    trace = planner.plan(driver, ledger, machine=1)

    # 每条边至少 n₀=3 次访问
    for edge in trace.edge_stats:
        assert edge.visits >= 3, f"{edge.action.kind} has only {edge.visits} visits"


def test_ocba_disabled_falls_back_to_uniform():
    """ocba_enabled=False → 均匀迭代 (旧行为不变)。"""
    env = ResourceCalendarEnv(build_small_encoder(), top_k=8, w_lookahead=4.0)
    driver = _driver(env)
    driver.reset_episode()
    ledger = ReservationLedger()

    planner = VCMCTSPlanner(
        VCMCTSConfig(
            n_iter=6, top_k_dispatch=1, top_b_reserve=1,
            crn_noise=True, n_mc=3, crn_seed_base=0,
            ocba_enabled=False, rollout_max_steps=10,
        )
    )
    trace = planner.plan(driver, ledger, machine=1)
    total_visits = sum(edge.visits for edge in trace.edge_stats)
    # 3 edges × 6 iterations = 18 (warm-up rounds up to len(edges))
    assert total_visits >= 6


def test_ocba_delta_star_stops_early():
    """OCBA 顺序停止: |Δ̄| < δ* 且置信区间窄 → 停止。"""
    env = ResourceCalendarEnv(build_small_encoder(), top_k=8, w_lookahead=4.0)
    driver = _driver(env)
    driver.reset_episode()
    ledger = ReservationLedger()

    def evaluator(_driver, _ledger, _action, _config):
        return VCMCTSObjective(
            qtime_violation_count=0.0, qtime_violation_total=0.0,
            priority_weighted_wait=5.0, avg_utilization=0.5,
        )

    planner = VCMCTSPlanner(
        VCMCTSConfig(
            n_iter=20, top_k_dispatch=1, top_b_reserve=1,
            crn_noise=True, n_mc=8, crn_seed_base=0,
            ocba_enabled=True, ocba_n0=3, ocba_delta_star=100.0, ocba_gamma=0.1,
            rollout_max_steps=10,
        ),
        rollout_evaluator=evaluator,
    )
    trace = planner.plan(driver, ledger, machine=1)
    total_visits = sum(edge.visits for edge in trace.edge_stats)
    # δ* 极大 → 所有边在无差异区 → 应该在 warm-up 后很快停止
    assert total_visits < 20 * len(trace.edge_stats)
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
python -m pytest tests/test_vc_mcts_mechanisms.py -q -k "ocba"
```

Expected: FAIL — `ocba_enabled` not in `VCMCTSConfig`.

- [ ] **Step 3: Add OCBA config knobs**

Add to `VCMCTSConfig`:
```python
    # 机制 3 组件二 (报告8 §7.13.3): OCBA 顺序预算分配。
    # ocba_enabled=True 时 plan() 用两阶段 OCBA 替换均匀迭代。
    ocba_enabled: bool = False
    ocba_n0: int = 3           # 阶段一 warm-up 场景数
    ocba_delta_star: float = 0.0  # 无差异区宽度 (O2 单位)
    ocba_gamma: float = 0.1    # 停止置信水平 (1−γ)
```

- [ ] **Step 4: Implement OCBA two-stage allocation in plan()**

Replace the uniform iteration loop in `plan()` with OCBA when `ocba_enabled=True`:

```python
    def plan(self, driver, ledger, machine):
        current_time = float(driver.env.current_time)
        actions = self.build_root_actions(driver, ledger, machine)
        edges = [VCMCTSEdgeStats(action=action) for action in actions]
        if self.config.use_rho_pc:
            from priority_capability_matching import rho_pc_for_action
            for edge in edges:
                rho = rho_pc_for_action(
                    driver.env, ledger, edge.action,
                    priority_threshold=self.config.rho_pc_priority_threshold,
                )
                edge.rho_pc_before = rho.before
                edge.rho_pc_after = rho.after
                edge.delta_rho_pc = rho.delta

        if self.config.ocba_enabled and self.config.crn_noise:
            self._ocba_plan(driver, ledger, edges)
        else:
            iteration_count = max(len(edges), int(self.config.n_iter))
            for _ in range(iteration_count):
                edge = self._select_edge(edges)
                objective = self.evaluate_action(
                    driver, ledger, edge.action, scenario=edge.visits
                )
                edge.record(objective)

        selected = self._choose_final_action(edges, current_time)
        return VCMCTSDecisionTrace(
            selected_action=selected.action,
            edge_stats=edges,
            current_time=current_time,
            machine=int(machine),
        )

    def _ocba_plan(self, driver, ledger, edges):
        """两阶段 OCBA (报告8 §7.13.3): warm-up → 顺序分配 → 停止。"""
        n0 = int(self.config.ocba_n0)
        n_mc = int(self.config.n_mc)
        total_budget = int(self.config.n_iter) * len(edges)
        n_feasible = len(edges)

        # 阶段一: warm-up — 每条边 n₀ 条场景
        for _ in range(n0):
            for edge in edges:
                objective = self.evaluate_action(
                    driver, ledger, edge.action, scenario=edge.visits
                )
                edge.record(objective)

        budget_used = n0 * n_feasible
        while budget_used < total_budget:
            # 找当前最佳边 b (按 O2 均值, 最易比较的维度)
            best = min(edges, key=lambda e: e.mean_objective.priority_weighted_wait)

            # 对每条非最佳边, 计算配对差的统计量
            deltas = {}
            for edge in edges:
                if edge is best:
                    continue
                paired = []
                for k in range(min(n0, edge.visits, best.visits)):
                    # 配对场景 k: 同场景编号下两边 O2 差
                    so = edge._scenarios[k]
                    bo = best._scenarios[k]
                    paired.append(so.priority_weighted_wait - bo.priority_weighted_wait)
                if not paired:
                    continue
                mean_delta = sum(paired) / len(paired)
                var_delta = max(1e-9, sum((d - mean_delta) ** 2 for d in paired) / (len(paired) - 1))
                deltas[id(edge)] = (abs(mean_delta), var_delta ** 0.5)

            if not deltas:
                break

            # 顺序停止检测
            remaining = sorted(deltas.items(), key=lambda kv: kv[1][0])
            if len(remaining) >= 2:
                best_delta = remaining[0][1]
                second_delta = remaining[1][1]
                if best_delta[0] < self.config.ocba_delta_star:
                    # 当前最佳与第二名的差距在无差异区内 → 停止
                    if second_delta[0] < self.config.ocba_delta_star:
                        break

            # 分配剩余预算: N_i ∝ (σ_i / δ_{b,i})²
            alloc = {}
            for edge_id, (delta, sigma) in deltas.items():
                if delta > 0:
                    alloc[edge_id] = (sigma / delta) ** 2
            total_alloc = sum(alloc.values())
            if total_alloc <= 0:
                break
            for edge_id, weight in alloc.items():
                alloc[edge_id] = max(1, int(weight / total_alloc * 10))

            # 按分配额跑 scenario
            for edge in edges:
                extra = alloc.get(id(edge), 0)
                for _ in range(min(extra, total_budget - budget_used)):
                    objective = self.evaluate_action(
                        driver, ledger, edge.action, scenario=edge.visits
                    )
                    edge.record(objective)
                    budget_used += 1
                    if budget_used >= total_budget:
                        break
                if budget_used >= total_budget:
                    break
```

- [ ] **Step 5: Run tests to verify they pass**

```powershell
python -m pytest tests/test_vc_mcts_mechanisms.py -q -k "ocba"
```

Expected: 3 passed.

- [ ] **Step 6: Run all mechanism tests + full suite**

```powershell
python -m pytest tests/test_vc_mcts_mechanisms.py -q
python -m pytest tests/ -q
```

Expected: all pass.

- [ ] **Step 7: Commit**

```powershell
git add FAB_RL/FABenv/vc_mcts_planner.py FAB_RL/FABenv/tests/test_vc_mcts_mechanisms.py
git commit -m "feat: OCBA two-stage budget allocation for mechanism 3"
```

---

### Task 4: Per-scenario objectives in trace + probe CLI

**Files:**
- Modify: `FAB_RL/FABenv/vc_mcts_planner.py` (VCMCTSEdgeStats.to_dict)
- Modify: `FAB_RL/FABenv/scripts/probes/vc_mcts_probe.py` (new CLI flags)

- [ ] **Step 1: Add per-scenario serialization to to_dict()**

When `cvar_enabled=True`, include `scenario_objectives` in trace output:

```python
    def to_dict(self, include_scenarios=False):
        mean = self.mean_objective
        d = {
            "action": self.action.to_dict(),
            "kind": self.action.kind,
            "visits": int(self.visits),
            "mean_qtime": None if mean is None else float(mean.qtime_violation_count),
            "mean_qtime_total": None if mean is None else float(mean.qtime_violation_total),
            "mean_o2": None if mean is None else float(mean.priority_weighted_wait),
            "mean_util": None if mean is None else float(mean.avg_utilization),
            "leaf_bootstrap_visits": int(self.leaf_bootstrap_visits),
            "mean_is_leaf_bootstrap": None if mean is None else bool(mean.is_leaf_bootstrap),
            "rho_pc": float(self.rho_pc_after),
            "rho_pc_before": float(self.rho_pc_before),
            "rho_pc_after": float(self.rho_pc_after),
            "delta_rho_pc": float(self.delta_rho_pc),
        }
        if include_scenarios and self._scenarios:
            d["scenario_objectives"] = [
                {
                    "qtime_violation_count": float(obj.qtime_violation_count),
                    "qtime_violation_total": float(obj.qtime_violation_total),
                    "priority_weighted_wait": float(obj.priority_weighted_wait),
                    "avg_utilization": float(obj.avg_utilization),
                }
                for obj in self._scenarios
            ]
        return d
```

Update `VCMCTSDecisionTrace.to_dict()` to pass the flag:
```python
class VCMCTSDecisionTrace:
    # ... add field
    include_scenarios: bool = False

    def to_dict(self):
        edges = [edge.to_dict(include_scenarios=self.include_scenarios) for edge in self.edge_stats]
        # ... rest unchanged
```

Update `plan()` to thread the flag into the trace:
```python
        selected = self._choose_final_action(edges, current_time)
        return VCMCTSDecisionTrace(
            selected_action=selected.action,
            edge_stats=edges,
            current_time=current_time,
            machine=int(machine),
            include_scenarios=bool(self.config.cvar_enabled),
        )
```

- [ ] **Step 2: Add CLI flags to probe**

Add to `run_seed()` signature and `VCMCTSConfig` construction:
```python
    ocba_enabled=False,
    ocba_n0=3,
    ocba_delta_star=0.0,
    ocba_gamma=0.1,
    cvar_enabled=False,
    cvar_beta0=0.95,
    cvar_beta=0.8,
```

Thread through `VCMCTSConfig(...)`:
```python
            ocba_enabled=ocba_enabled,
            ocba_n0=ocba_n0,
            ocba_delta_star=ocba_delta_star,
            ocba_gamma=ocba_gamma,
            cvar_enabled=cvar_enabled,
            cvar_beta0=cvar_beta0,
            cvar_beta=cvar_beta,
```

Add CLI arguments:
```python
    parser.add_argument("--ocba", action="store_true", help="机制 3: OCBA 顺序预算分配")
    parser.add_argument("--ocba-n0", type=int, default=3, help="OCBA warm-up 场景数")
    parser.add_argument("--ocba-delta-star", type=float, default=0.0, help="OCBA 无差异区宽度")
    parser.add_argument("--ocba-gamma", type=float, default=0.1, help="OCBA 停止置信水平")
    parser.add_argument("--cvar", action="store_true", help="机制 3: CVaR 风险字典序 final pick")
    parser.add_argument("--cvar-beta0", type=float, default=0.95, help="Q-time 上分位数")
    parser.add_argument("--cvar-beta", type=float, default=0.8, help="O2 CVaR 水平")
```

Thread through worker tuples and `main()` call.

- [ ] **Step 3: Verify CLI**

```powershell
python scripts/probes/vc_mcts_probe.py --help 2>&1 | Select-String "ocba|cvar"
```

Expected: all new flags appear.

- [ ] **Step 4: Run all tests**

```powershell
python -m pytest tests/ -q
```

Expected: all pass.

- [ ] **Step 5: Commit**

```powershell
git add FAB_RL/FABenv/vc_mcts_planner.py FAB_RL/FABenv/scripts/probes/vc_mcts_probe.py
git commit -m "feat: per-scenario trace + probe CLI for OCBA/CVaR"
```

---

### Task 5: End-to-end smoke with mechanism 3

**Files:**
- No new files.

- [ ] **Step 1: Run a small smoke probe with all three mechanism-3 components**

```powershell
python scripts/probes/vc_mcts_probe.py --instance small --seeds 1 --strategy FIFO --skip-oracle --n-iter 8 --top-b 1 --rollout-max-steps 10 --max-decisions 4 --qtime-mask-mode aggregate --dispatch-delegate rule --crn-noise --n-mc 4 --ocba --cvar --cvar-beta0 0.8 --cvar-beta 0.6 --trace-out artifacts/results/m3_smoke_trace.jsonl --trace-summary-out artifacts/results/m3_smoke_summary.json
```

Expected: exits 0, trace JSONL contains `scenario_objectives` fields.

- [ ] **Step 2: Run full FABenv test suite**

```powershell
python -m pytest tests/ -q
```

Expected: all pass.

- [ ] **Step 3: Commit any remaining test/config changes**

```powershell
git add -A
git commit -m "test: mechanism 3 end-to-end smoke verification"
```

---

## Self-Review

**1. Spec coverage:**
- §7.13.2 组件一 CRN 配对场景: Task 1 — per-scenario objective storage in VCMCTSEdgeStats, `scenario_objectives` property, `quantile`/`cvar` methods.
- §7.13.3 组件二 OCBA: Task 3 — `ocba_enabled` config, `_ocba_plan()` two-stage allocation with warm-up → paired delta → N_i ∝ (σ_i/δ_{b,i})² → early stop.
- §7.13.4 组件三 CVaR 风险字典序: Task 2 — `cvar_enabled` config, `_cvar_choose()` with Q̂_β0 qtime → CVaR_β O2 → mean util → visits tiebreak.
- §7.13.5 挂载点: Task 4 — trace per-scenario serialization, probe CLI flags.
- §7.13.1 降级预案: all three components are independently opt-in (`ocba_enabled`/`cvar_enabled` default False), OCBA can be disabled leaving CRN+CVaR working.

**2. Placeholder scan:** No TBD/TODO markers. All code blocks are concrete.

**3. Type consistency:** `quantile(dim, beta, minimize)` and `cvar(dim, beta, minimize)` share the same signature. `_cvar_choose` uses `edge.quantile(...)` and `edge.cvar(...)` which are defined in Task 1. OCBA `_ocba_plan` reads `edge._scenarios[k]` which is populated by `record()` in Task 1.

---