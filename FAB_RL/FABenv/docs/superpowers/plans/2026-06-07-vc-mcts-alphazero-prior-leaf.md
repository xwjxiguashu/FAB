# VC-MCTS AlphaZero 增强：SAS 先验 p(s,a) + 多头 Critic 叶子价值截断

日期：2026-06-07
状态：实现文档（可直接落地）

## 背景与现状（已核实）

报告"诚实边界 (3)"与"下一步工作 1"指出，VC-MCTS 当前只把 SAS 当作**派工 delegate**（挑哪个候选派），尚未接入两项 AlphaZero 式增强：

1. **SAS 先验 p(s,a) 未进 PUCT**：`vc_mcts_planner.py:_select_edge` 的探索项用 `edge.action.prior`，而 prior 来自启发式候选打分（dispatch=`score+1.0`、reserve=`opportunity.score`、no_op=`0.05`），不是策略 softmax 概率。
2. **多头 Critic 叶子截断未接入**：`evaluate_action` 是完整 rollout-to-terminal + 精确指标，没有用 Critic value 在叶子处 bootstrap。

本文档实现这两项，并保持**默认行为完全不变**（config 显式开启）。

## 已核实的关键事实

- 多头 Critic 现为 **3 通道** `(exec, qtime, util)`（`progress` 已删，CLAUDE.md 那段为 stale）。见 [rl_environment.py:342-406](../../../rl_environment.py)。
- 通道→Objective 映射只有两维干净可用：
  - `qtime` 通道**逐步 telescoping**：`r_qtime = -new_qtime_violation/num_lots`，`Σ_t = 终局总违反`。故 `V_qtime(leaf)` 估计**剩余**新违反数 → `remaining_count ≈ max(0, -V_qtime·num_lots)`。
  - `util` 通道**终局**：`r_util = avg_machine_utilization`（仅终局非零）→ `V_util(leaf)` 直接估计终局利用率。
- **O2（priority_weighted_wait）与 qtime_violation_total 没有对应通道**——SAS 从未在 O2 上训练。这是叶子截断绕不开的缺口。

## 三项设计决策（已确认）

| 决策 | 选择 |
|---|---|
| 代码布局 | 新建独立文件 `vc_mcts_alphazero.py`，planner 仅做最小钩子改动 |
| O2/qtime_total 缺口 | **混合**：`qtime_count`/`util` 用 Critic bootstrap；`O2`/`qtime_total` 用部分 rollout 到截断深度后的**实排程实测**（单调下界，明确标注为 partial-horizon 估计） |
| 默认/兼容 | 默认全关：`prior_source='heuristic'`、`use_leaf_value=False`；新增 `prior_source='policy'`、`use_leaf_value=True` + `leaf_rollout_depth` 显式启用 |

---

## 1. 新文件：`vc_mcts_alphazero.py`

```python
"""AlphaZero-style augmentations for VC-MCTS.

Two pieces, both opt-in (the planner keeps its heuristic priors and
full-rollout evaluation by default):

1. SASPolicyPriorProvider — turns the trained SAS policy's masked softmax
   over the candidate pool into a PUCT prior p(s,a). The policy never models
   reservations, so reserve edges get a fixed exploration prior injected by
   the planner; this module only exposes the per-candidate distribution.

2. MultiHeadCriticLeafValue — queries the multi-head critic at a leaf state
   and converts the (qtime, util) channel values into the two objective
   dimensions they cover:
     qtime channel telescopes  -> remaining_violation_count = max(0, -V_qtime * num_lots)
     util channel is terminal   -> avg_utilization          = clip(V_util, 0, 1)
   The other two objective dimensions (priority_weighted_wait / O2 and
   qtime_violation_total) have no critic channel — the planner fills those
   from the partial-rollout schedule (a partial-horizon actual, monotone
   lower bound). See the planner's `_leaf_value_objective`.
"""
import numpy as np
import torch


def _policy_device(policy):
    try:
        return next(policy.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _observation_tensors(observation, device):
    candidate_features = torch.as_tensor(
        observation.candidate_features, dtype=torch.float32, device=device
    ).unsqueeze(0)
    candidate_mask = torch.as_tensor(
        observation.candidate_mask, dtype=torch.bool, device=device
    ).unsqueeze(0)
    global_features = torch.as_tensor(
        observation.global_features, dtype=torch.float32, device=device
    ).unsqueeze(0)
    return candidate_features, candidate_mask, global_features


class SASPolicyPriorProvider:
    """Expose the SAS policy's masked softmax over a candidate pool.

    Works with both the single-head (`Phase2SASActorCritic`) and multi-head
    (`Phase2SASMultiHeadActorCritic`) policies — both return ``probs`` from
    ``greedy_action``. The returned vector has one entry per pool slot;
    padded / masked slots are exactly 0.0.
    """

    def __init__(self, policy):
        self.policy = policy

    @property
    def label(self):
        return "sas_prior"

    def candidate_probs(self, driver, machine, pool=None):
        pool = driver.env.build_candidate_pool(machine) if pool is None else pool
        observation = driver.observation_encoder.encode(machine, pool, driver.env)
        device = _policy_device(self.policy)
        cand, mask, glob = _observation_tensors(observation, device)
        with torch.no_grad():
            output = self.policy.greedy_action(cand, mask, glob)
        probs = output["probs"].detach().cpu().reshape(-1).numpy().astype(float)
        # Defensive: zero-out anything the mask marks invalid (greedy already
        # masked logits, but padding slots beyond the distribution stay 0).
        pool_mask = np.asarray(pool.action_mask, dtype=bool)
        if probs.shape[0] == pool_mask.shape[0]:
            probs = np.where(pool_mask, probs, 0.0)
        return probs


class MultiHeadCriticLeafValue:
    """Query the multi-head critic at a (machine, pool) leaf observation.

    Returns ``{"qtime": V_qtime, "util": V_util}``. Requires a policy that
    implements ``critic_values`` (the multi-head variant); a single-head
    policy has no per-channel critic and is rejected at construction.
    """

    def __init__(self, policy, qtime_channel="qtime", util_channel="util"):
        if not hasattr(policy, "critic_values"):
            raise TypeError(
                "MultiHeadCriticLeafValue requires a multi-head policy with "
                "critic_values(); got a single-head policy."
            )
        self.policy = policy
        self.qtime_channel = qtime_channel
        self.util_channel = util_channel

    @property
    def label(self):
        return "multihead_leaf_value"

    def estimate(self, driver, machine, pool=None):
        pool = driver.env.build_candidate_pool(machine) if pool is None else pool
        observation = driver.observation_encoder.encode(machine, pool, driver.env)
        device = _policy_device(self.policy)
        cand, mask, glob = _observation_tensors(observation, device)
        with torch.no_grad():
            values = self.policy.critic_values(cand, mask, glob)
        return {
            "qtime": float(values[self.qtime_channel].detach().cpu().reshape(-1)[0]),
            "util": float(values[self.util_channel].detach().cpu().reshape(-1)[0]),
        }


def critic_to_objective_dims(critic_values, partial_metrics, num_lots):
    """Map (qtime, util) critic values + partial metrics -> objective dims.

    Returns a dict with the four VCMCTSObjective fields:
      qtime_violation_count : partial committed count + critic-estimated
                              remaining (telescoping channel).
      qtime_violation_total : partial-horizon actual (no critic channel).
      priority_weighted_wait: partial-horizon actual (no critic channel).
      avg_utilization       : critic terminal estimate, clipped to [0, 1].
    """
    num_lots = max(float(num_lots), 1.0)
    remaining_count = max(0.0, -float(critic_values["qtime"]) * num_lots)
    partial_count = float(partial_metrics.get("qtime_violation_count", 0.0))
    util = min(1.0, max(0.0, float(critic_values["util"])))
    return {
        "qtime_violation_count": partial_count + remaining_count,
        "qtime_violation_total": float(partial_metrics.get("qtime_violation_total", 0.0)),
        "priority_weighted_wait": float(partial_metrics.get("priority_weighted_wait", 0.0)),
        "avg_utilization": util,
    }


def load_sas_alphazero(checkpoint_path, *, map_location="cpu", require_multihead=False):
    """Factory: load a checkpoint and build prior provider (+ leaf value).

    Returns ``(prior_provider, leaf_value_or_None, policy)``. ``leaf_value`` is
    None when the checkpoint is a single-head policy (no per-channel critic).
    """
    from model_checkpoint import load_policy_checkpoint

    policy, _checkpoint = load_policy_checkpoint(checkpoint_path, map_location=map_location)
    prior_provider = SASPolicyPriorProvider(policy)
    leaf_value = None
    if hasattr(policy, "critic_values"):
        leaf_value = MultiHeadCriticLeafValue(policy)
    elif require_multihead:
        raise TypeError("checkpoint is single-head but require_multihead=True")
    return prior_provider, leaf_value, policy
```

---

## 2. 补丁：`vc_mcts_planner.py`

### 2.1 `VCMCTSConfig` 新增字段

在 `@dataclass(frozen=True) class VCMCTSConfig` 末尾追加：

```python
    # --- AlphaZero 增强（默认全关，向后兼容）---
    prior_source: str = "heuristic"     # "heuristic" | "policy"
    policy_reserve_prior: float = 0.15  # policy 模式下每条 reserve 边的原始先验质量（renorm 前）
    use_leaf_value: bool = False        # True: 部分 rollout + 多头 Critic bootstrap
    leaf_rollout_depth: int = 8         # 截断深度（传给 run_rule_episode_with_reservations 的 max_steps）
```

### 2.2 文件顶部 import

```python
from dataclasses import dataclass, field, replace   # 增加 replace
from vc_mcts_alphazero import critic_to_objective_dims
```

### 2.3 `VCMCTSPlanner.__init__` 新增两个注入点

```python
    def __init__(self, config=None, rollout_evaluator=None, dispatch_delegate=None,
                 prior_provider=None, leaf_value=None):
        self.config = config if config is not None else VCMCTSConfig()
        self.rollout_evaluator = rollout_evaluator
        self.dispatch_delegate = dispatch_delegate
        self.prior_provider = prior_provider   # SASPolicyPriorProvider | None
        self.leaf_value = leaf_value           # MultiHeadCriticLeafValue | None
```

### 2.4 `build_root_actions` 末尾：策略先验改写

在 `return actions` 之前插入（保留原有启发式 prior 构造逻辑不动，只在 policy 模式下整体重写为概率分布）：

```python
        if (
            self.config.prior_source == "policy"
            and self.prior_provider is not None
        ):
            actions = self._assign_policy_priors(actions, driver, machine, pool)
        return actions
```

并新增方法：

```python
    def _assign_policy_priors(self, actions, driver, machine, pool):
        """Overwrite edge priors with the SAS policy's softmax p(s,a).

        dispatch/delegate_dispatch -> probs[action_index]
        no_op                      -> sum of probs over valid wait actions
        reserve                    -> fixed config.policy_reserve_prior (the
                                      policy does not model reservations)
        All raw priors are renormalized to sum to 1 across the built edges so
        PUCT sees a proper prior distribution.
        """
        probs = self.prior_provider.candidate_probs(driver, machine, pool=pool)

        wait_prob = 0.0
        for index, action in enumerate(pool.actions):
            if not bool(pool.action_mask[index]):
                continue
            if getattr(action, "is_wait", False) and index < probs.shape[0]:
                wait_prob += float(probs[index])

        raw = []
        for action in actions:
            if action.kind in ("dispatch", "delegate_dispatch"):
                idx = int(action.action_index)
                value = float(probs[idx]) if 0 <= idx < probs.shape[0] else 0.0
            elif action.kind == "no_op":
                value = float(wait_prob)
            elif action.kind == "reserve":
                value = float(self.config.policy_reserve_prior)
            else:
                value = 0.0
            raw.append(max(1e-6, value))

        total = float(sum(raw))
        return [
            replace(action, prior=float(value) / total)
            for action, value in zip(actions, raw)
        ]
```

### 2.5 `evaluate_action`：叶子价值截断分支

把现有方法改为：

```python
    def evaluate_action(self, driver, ledger, action):
        if self.rollout_evaluator is not None:
            return self.rollout_evaluator(driver, ledger, action, self.config)

        branch_driver = clone_driver_for_rollout(driver)
        branch_ledger = clone_ledger_for_rollout(ledger)
        self._apply_action(branch_driver, branch_ledger, action)

        if self.config.use_leaf_value and self.leaf_value is not None:
            objective = self._leaf_value_objective(branch_driver, branch_ledger)
            if objective is not None:
                return objective
            # objective is None -> branch already terminal: fall through to the
            # exact metrics on the finished schedule below.

        run_rule_episode_with_reservations(
            branch_driver,
            ledger=branch_ledger,
            strategy=self.config.rollout_strategy,
            max_steps=self.config.rollout_max_steps or branch_driver.max_steps,
            dispatch_delegate=self.dispatch_delegate,
        )
        metrics = schedule_metrics_with_priority_wait(branch_driver.env.encoder, branch_driver.env)
        return VCMCTSObjective(
            qtime_violation_count=float(metrics["qtime_violation_count"]),
            qtime_violation_total=float(metrics["qtime_violation_total"]),
            priority_weighted_wait=float(metrics["priority_weighted_wait"]),
            avg_utilization=float(metrics["avg_utilization"]),
        )
```

新增方法：

```python
    def _leaf_value_objective(self, branch_driver, branch_ledger):
        """Partial rollout to leaf_rollout_depth, then bootstrap with the critic.

        Returns a VCMCTSObjective, or None if the branch reached a terminal
        state during the partial rollout (caller then uses exact metrics — the
        partial schedule IS the full schedule, so no bootstrap is needed).
        """
        run_rule_episode_with_reservations(
            branch_driver,
            ledger=branch_ledger,
            strategy=self.config.rollout_strategy,
            max_steps=int(self.config.leaf_rollout_depth),
            dispatch_delegate=self.dispatch_delegate,
        )

        done, _reason = branch_driver.is_episode_done()
        if done:
            return None  # finished within the truncation budget -> use exact metrics

        # Pick a representative machine for the leaf state-value query.
        machine = self._leaf_machine(branch_driver, branch_ledger)
        if machine is None:
            return None  # no decision point available -> treat as terminal

        partial = schedule_metrics_with_priority_wait(
            branch_driver.env.encoder, branch_driver.env
        )
        critic_values = self.leaf_value.estimate(branch_driver, machine)
        dims = critic_to_objective_dims(
            critic_values,
            partial,
            num_lots=branch_driver.env.encoder.num_lots,
        )
        return VCMCTSObjective(
            qtime_violation_count=dims["qtime_violation_count"],
            qtime_violation_total=dims["qtime_violation_total"],
            priority_weighted_wait=dims["priority_weighted_wait"],
            avg_utilization=dims["avg_utilization"],
        )

    def _leaf_machine(self, driver, ledger):
        machines = [
            m for m in driver.get_dispatchable_machines()
            if m not in ledger.reserved_machines()
        ]
        if not machines:
            # advance one event on the throwaway clone to surface a machine
            if advance_to_next_event_with_ledger(driver, ledger) is None:
                return None
            machines = [
                m for m in driver.get_dispatchable_machines()
                if m not in ledger.reserved_machines()
            ]
        if not machines:
            return None
        return driver.select_next_machine(machines)
```

> 注：`_leaf_machine` 在抛弃用的 clone 上调用 `advance_to_next_event_with_ledger`，可安全 mutate。

---

## 3. 补丁：`vc_mcts_probe.py`（CLI 接线，可选）

### 3.1 import

```python
from vc_mcts_alphazero import load_sas_alphazero
```

### 3.2 `run_seed` 新增参数

在签名追加：`prior_source="heuristic", use_leaf_value=False, leaf_rollout_depth=8, alphazero_checkpoint=None,`。

构造 planner 处替换为：

```python
    prior_provider = None
    leaf_value = None
    if (prior_source == "policy" or use_leaf_value) and alphazero_checkpoint:
        prior_provider, leaf_value, _policy = load_sas_alphazero(
            alphazero_checkpoint,
            require_multihead=use_leaf_value,
        )

    planner = VCMCTSPlanner(
        VCMCTSConfig(
            n_iter=n_iter,
            top_k_dispatch=top_k_dispatch,
            top_b_reserve=top_b,
            rollout_strategy=strategy,
            rollout_max_steps=rollout_max_steps or max_steps,
            use_delegate_dispatch=use_delegate_dispatch,
            prior_source=prior_source,
            use_leaf_value=use_leaf_value,
            leaf_rollout_depth=leaf_rollout_depth,
        ),
        dispatch_delegate=delegate,
        prior_provider=prior_provider,
        leaf_value=leaf_value,
    )
```

### 3.3 `_cli` 新增 argparse

```python
    parser.add_argument("--prior-source", choices=["heuristic", "policy"], default="heuristic")
    parser.add_argument("--use-leaf-value", action="store_true")
    parser.add_argument("--leaf-rollout-depth", type=int, default=8)
    parser.add_argument("--alphazero-checkpoint", default=None)
```

并把这些参数串到 `main` → `run_seed`（以及 `_run_seed_job` 的 tuple 解包，与现有 `sas_checkpoint` 同样处理）。

用法：

```powershell
# 仅启用 SAS 先验 p(s,a)
python vc_mcts_probe.py --instance late_hi --seeds 5 --prior-source policy --alphazero-checkpoint model.pt
# 启用叶子价值截断（需多头 checkpoint）
python vc_mcts_probe.py --instance late_hi --seeds 5 --use-leaf-value --leaf-rollout-depth 8 --alphazero-checkpoint model.pt
# 两者同时 + SAS 派工 delegate
python vc_mcts_probe.py --instance late_hi --seeds 5 --prior-source policy --use-leaf-value `
    --dispatch-delegate sas --sas-checkpoint model.pt --alphazero-checkpoint model.pt
```

---

## 4. 测试：`tests/test_vc_mcts_alphazero.py`

```python
"""Tests for the VC-MCTS AlphaZero augmentations (prior + leaf value)."""
import numpy as np
import pytest

from dispatch_delegate import RuleDispatchDelegate
from phase2_sas_driver import Phase2EpisodeDriver
from phase2_sas_observation import Phase2ObservationEncoder
from phase2_sas_policy import (
    Phase2SASActorCritic,
    Phase2SASMultiHeadActorCritic,
)
from problem_instances import build_small_encoder
from rl_environment import ResourceCalendarEnv, RewardConfig
from vc_mcts_alphazero import (
    MultiHeadCriticLeafValue,
    SASPolicyPriorProvider,
    critic_to_objective_dims,
)
from vc_mcts_planner import VCMCTSConfig, VCMCTSPlanner


CANDIDATE_DIM = 18
GLOBAL_DIM = 9  # Phase2ObservationEncoder(lookahead=False)


def _driver():
    env = ResourceCalendarEnv(build_small_encoder(), top_k=8)
    env.reset()
    driver = Phase2EpisodeDriver(
        env, Phase2ObservationEncoder(), RewardConfig(), max_steps=200
    )
    driver.reset_episode()
    return driver


def _multihead_policy():
    torch_seed()
    return Phase2SASMultiHeadActorCritic(CANDIDATE_DIM, GLOBAL_DIM, hidden_dim=32)


def torch_seed():
    import torch
    torch.manual_seed(0)


# --- SASPolicyPriorProvider ------------------------------------------------

def test_prior_provider_is_valid_distribution():
    driver = _driver()
    machine = driver.select_next_machine(driver.get_dispatchable_machines())
    pool = driver.env.build_candidate_pool(machine)
    provider = SASPolicyPriorProvider(_multihead_policy())

    probs = provider.candidate_probs(driver, machine, pool=pool)

    assert probs.shape[0] == len(pool.actions)
    # padded / masked slots are exactly zero
    assert np.all(probs[~np.asarray(pool.action_mask, dtype=bool)] == 0.0)
    # valid mass sums to ~1
    assert probs.sum() == pytest.approx(1.0, abs=1e-5)


def test_prior_provider_works_with_single_head():
    driver = _driver()
    machine = driver.select_next_machine(driver.get_dispatchable_machines())
    pool = driver.env.build_candidate_pool(machine)
    provider = SASPolicyPriorProvider(
        Phase2SASActorCritic(CANDIDATE_DIM, GLOBAL_DIM, hidden_dim=32)
    )
    probs = provider.candidate_probs(driver, machine, pool=pool)
    assert probs.sum() == pytest.approx(1.0, abs=1e-5)


# --- planner policy-prior wiring ------------------------------------------

def test_policy_prior_renormalizes_edges():
    driver = _driver()
    machine = driver.select_next_machine(driver.get_dispatchable_machines())
    from reservation_ledger import ReservationLedger

    planner = VCMCTSPlanner(
        VCMCTSConfig(prior_source="policy", policy_reserve_prior=0.2),
        prior_provider=SASPolicyPriorProvider(_multihead_policy()),
    )
    actions = planner.build_root_actions(driver, ReservationLedger(), machine)

    priors = [a.prior for a in actions]
    assert all(p > 0.0 for p in priors)
    assert sum(priors) == pytest.approx(1.0, abs=1e-6)


def test_heuristic_prior_is_default_unchanged():
    driver = _driver()
    machine = driver.select_next_machine(driver.get_dispatchable_machines())
    from reservation_ledger import ReservationLedger

    planner = VCMCTSPlanner(VCMCTSConfig())  # defaults
    actions = planner.build_root_actions(driver, ReservationLedger(), machine)
    # heuristic no_op prior is the fixed 0.05; edges do NOT renormalize to 1
    noop = [a for a in actions if a.kind == "no_op"][0]
    assert noop.prior == pytest.approx(0.05)


# --- MultiHeadCriticLeafValue + mapping -----------------------------------

def test_leaf_value_estimate_returns_channels():
    driver = _driver()
    machine = driver.select_next_machine(driver.get_dispatchable_machines())
    leaf = MultiHeadCriticLeafValue(_multihead_policy())
    values = leaf.estimate(driver, machine)
    assert set(values) == {"qtime", "util"}
    assert np.isfinite(values["qtime"]) and np.isfinite(values["util"])


def test_leaf_value_rejects_single_head():
    with pytest.raises(TypeError):
        MultiHeadCriticLeafValue(Phase2SASActorCritic(CANDIDATE_DIM, GLOBAL_DIM))


def test_critic_to_objective_dims_mapping():
    partial = {
        "qtime_violation_count": 2.0,
        "qtime_violation_total": 5.0,
        "priority_weighted_wait": 30.0,
        "avg_utilization": 0.1,
    }
    # V_qtime = -0.1, num_lots=10 -> remaining = 1.0 ; total = 2 + 1 = 3
    dims = critic_to_objective_dims(
        {"qtime": -0.1, "util": 0.7}, partial, num_lots=10
    )
    assert dims["qtime_violation_count"] == pytest.approx(3.0)
    assert dims["qtime_violation_total"] == 5.0          # partial-horizon actual
    assert dims["priority_weighted_wait"] == 30.0        # partial-horizon actual
    assert dims["avg_utilization"] == pytest.approx(0.7)  # critic terminal estimate


def test_critic_util_clipped_and_qtime_floored():
    partial = {"qtime_violation_count": 0.0}
    dims = critic_to_objective_dims({"qtime": 0.5, "util": 1.5}, partial, num_lots=4)
    assert dims["qtime_violation_count"] == 0.0  # max(0, -0.5*4) floored at 0
    assert dims["avg_utilization"] == 1.0        # clipped to [0,1]


# --- end-to-end: leaf-value planner produces a finite objective ------------

def test_evaluate_action_leaf_value_path():
    driver = _driver()
    machine = driver.select_next_machine(driver.get_dispatchable_machines())
    from reservation_ledger import ReservationLedger

    ledger = ReservationLedger()
    planner = VCMCTSPlanner(
        VCMCTSConfig(use_leaf_value=True, leaf_rollout_depth=4),
        leaf_value=MultiHeadCriticLeafValue(_multihead_policy()),
    )
    actions = planner.build_root_actions(driver, ledger, machine)
    dispatch = [a for a in actions if a.kind in ("dispatch", "delegate_dispatch")][0]
    objective = planner.evaluate_action(driver, ledger, dispatch)

    assert np.isfinite(objective.qtime_violation_count)
    assert np.isfinite(objective.priority_weighted_wait)
    assert 0.0 <= objective.avg_utilization <= 1.0


def test_leaf_value_default_off_uses_full_rollout():
    """Default config must keep the exact full-rollout path (regression)."""
    driver = _driver()
    machine = driver.select_next_machine(driver.get_dispatchable_machines())
    from reservation_ledger import ReservationLedger

    ledger = ReservationLedger()
    planner = VCMCTSPlanner(VCMCTSConfig())  # use_leaf_value=False
    actions = planner.build_root_actions(driver, ledger, machine)
    dispatch = [a for a in actions if a.kind in ("dispatch", "delegate_dispatch")][0]
    objective = planner.evaluate_action(driver, ledger, dispatch)
    # full rollout on small instance completes -> 0 qtime violations expected
    assert objective.qtime_violation_count >= 0.0
```

> 测试用 `build_small_encoder`（4-lot，rollout 快）。叶子截断 `leaf_rollout_depth=4` 在 small 上可能在预算内跑完（走 `return None` → 精确指标分支），`test_evaluate_action_leaf_value_path` 仅断言对象有限/合法，不依赖具体走哪条分支；如需稳定命中 bootstrap 分支，把实例换成 `build_pressure_test_encoder` 并设较小 depth。

---

## 5. 落地顺序与验证

```powershell
# 从 FAB_RL/FABenv/ 运行
python -m pytest tests/test_vc_mcts_alphazero.py -q          # 新测试
python -m pytest tests/ -q                                   # 全量回归（默认路径不变）
# 行为对比（需一个多头 checkpoint，例如 train_phase2_sas_ppo.py --mode multihead 产出）
python vc_mcts_probe.py --instance late_hi --seeds 5 --skip-oracle            # 基线（heuristic + full rollout）
python vc_mcts_probe.py --instance late_hi --seeds 5 --skip-oracle `
    --prior-source policy --use-leaf-value --alphazero-checkpoint model.pt    # 增强
```

## 6. 诚实边界（实现后应同步进报告）

- **SAS 先验 p(s,a)**：已接入 PUCT（dispatch/no_op 用策略 softmax，reserve 用固定探索先验 `policy_reserve_prior` 后整体 renorm）。reserve 的先验**不是**策略产物——策略从不建模预留，这是注入的探索质量。
- **多头 Critic 叶子截断**：`qtime`（telescoping→剩余违反数）与 `util`（终局利用率）两维由 Critic bootstrap；**O2（priority_weighted_wait）与 qtime_violation_total 仍是部分 rollout 到 `leaf_rollout_depth` 后的实排程 partial-horizon 实测**，因为 SAS 多头 Critic 没有这两个通道。若要让这两维也由价值网络估计，需新增 `o2` 通道并重训 checkpoint（超出本次范围）。
```
