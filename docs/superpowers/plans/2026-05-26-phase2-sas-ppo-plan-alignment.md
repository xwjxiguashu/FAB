# Phase2 SAS-PPO Project Plan Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Update `项目方案.md` so its Stage 2 SAS-PPO implementation plan strictly aligns with `报告.md` Section 10 and the detailed SAS single-training requirements in Section 9.

**Architecture:** This is a documentation-only alignment pass. The update keeps the existing five-module Stage 2 architecture, but tightens the project plan around four missing contracts: deterministic constrained Machine selection, complete SAS transition fields, explicit episode termination rules, and verification criteria that prove the plan matches the report.

**Tech Stack:** Markdown documentation, Python standard library text checks, existing repository files `报告.md` and `项目方案.md`. No runtime code changes are part of this plan.

---

## Scope Check

This plan modifies only `项目方案.md`. It does not implement Phase 2 code, does not add tests, and does not change `FAB_RL/FABenv` source files. The purpose is to make the written project方案 executable and unambiguous before implementation begins.

## File Structure

- Modify: `项目方案.md`
  - Add a short alignment addendum under Section 1.1.
  - Replace Section 5.2 Machine selection rule.
  - Add Section 5.4 Episode termination conditions.
  - Tighten Section 9 training data flow so `obs_t`, `obs_{t+1}`, `machine_id`, and `info` are explicitly recorded.
  - Update Section 10.2 `Phase2EpisodeDriver` signature and responsibility text.
  - Replace Section 10.5 `RolloutStep` draft with `StepInfo` plus a report-aligned `RolloutStep`.
  - Expand Section 13.1 and 13.2 acceptance criteria.
- Verify: run Python text assertions against `项目方案.md`.
- Do not modify: `报告.md`, `FAB_RL/FABenv/*.py`, tests, or generated outputs.

---

### Task 1: Add an explicit Stage 2 alignment addendum

**Files:**
- Modify: `项目方案.md:29-43`
- Test: Python text assertion against `项目方案.md`

- [ ] **Step 1: Insert the alignment addendum after the table in Section 1.1**

In `项目方案.md`, find this exact block:

```markdown
| 拖期、Q-time、priority、进度只作为轻量 shaping，基础策略收敛后逐步加入 | 8.2 节 |

---
```

Replace it with this exact block:

```markdown
| 拖期、Q-time、priority、进度只作为轻量 shaping，基础策略收敛后逐步加入 | 8.2 节 |

补充对齐要求：

- `RolloutStep` 必须记录 `machine_id` 与 `info`，否则无法从训练轨迹追溯当前动作对应的 Machine、Lot、PPID、插入结果和 reward 分解。
- Machine 选择规则采用报告第9.1节推荐的确定性受限优先规则：`machine_t = argmin_m (next_available_time_m, num_feasible_candidates_m, machine_id_m)`。
- Episode driver 必须显式维护 `total_wait_steps_per_episode`、`consecutive_failed_actions` 和 `unrecoverable_error`，并在终止时给出明确 `termination_reason`。
- Buffer 内部可以存储拆分后的 `candidate_features`、`candidate_mask` 和 `global_features`，但文档上必须把这些字段视为 `obs_t` 的展开形式，并保留 `next_observation` 对齐报告中的 `obs_{t+1}`。

---
```

- [ ] **Step 2: Verify the addendum exists**

Run:

```bash
python - <<'PY'
from pathlib import Path
text = Path('项目方案.md').read_text(encoding='utf-8')
required = [
    '补充对齐要求：',
    'RolloutStep` 必须记录 `machine_id` 与 `info`',
    'machine_t = argmin_m (next_available_time_m, num_feasible_candidates_m, machine_id_m)',
    'termination_reason',
    'next_observation` 对齐报告中的 `obs_{t+1}`',
]
for item in required:
    assert item in text, item
print('Task 1 alignment addendum checks passed')
PY
```

Expected output:

```text
Task 1 alignment addendum checks passed
```

- [ ] **Step 3: Commit only if explicitly authorized**

If the user explicitly requested commits before execution, run:

```bash
git add 项目方案.md
git commit -m "docs: add phase2 alignment addendum"
```

Expected output includes a new commit hash and the message `docs: add phase2 alignment addendum`.

---

### Task 2: Replace the Machine selection rule with the report-aligned constrained rule

**Files:**
- Modify: `项目方案.md:275-282`
- Modify: `项目方案.md:503-550`
- Test: Python text assertion against `项目方案.md`

- [ ] **Step 1: Replace Section 5.2**

In `项目方案.md`, find this exact section:

```markdown
### 5.2 多台 Machine 同时可调度时的规则

为避免阶段2引入组合动作空间，建议使用固定规则逐台决策：

- 优先按 Machine 编号从小到大；或
- 优先按最早空闲时间排序。

推荐先采用**按 Machine 编号从小到大**，因为实现简单且可复现。
```

Replace it with this exact section:

```markdown
### 5.2 多台 Machine 同时可调度时的规则

为避免阶段2引入组合动作空间，阶段2仍然使用固定规则逐台决策，但该规则应与报告第9.1节保持一致：

```text
machine_t = argmin_m (next_available_time_m, num_feasible_candidates_m, machine_id_m)
```

含义如下：

1. `next_available_time_m` 优先保证最早空闲的 Machine 先处理；
2. `num_feasible_candidates_m` 在同一空闲时刻优先处理候选动作更少、更受限的 Machine；
3. `machine_id_m` 仅作为 tie-breaker，保证训练和推理可复现。

该规则比单纯按 Machine 编号排序更贴近报告中的 SAS 单训设定，因为它优先处理当前调度状态中更容易失去可行窗口的 Machine，同时仍然避免组合动作空间。
```

- [ ] **Step 2: Add implementation guidance under Section 10.2**

In `项目方案.md`, find this exact text near the end of Section 10.2:

```markdown
### 说明

`Phase2EpisodeDriver` 是阶段2的核心控制器，负责把环境变成一个真正可训练的 episode 流程。
```

Replace it with this exact text:

```markdown
### 说明

`Phase2EpisodeDriver` 是阶段2的核心控制器，负责把环境变成一个真正可训练的 episode 流程。

`select_next_machine(machines)` 必须实现 Section 5.2 的确定性规则：

```text
machine_t = argmin_m (next_available_time_m, num_feasible_candidates_m, machine_id_m)
```

实现时可以通过以下信息计算排序 key：

1. `next_available_time_m`：从 Machine-level calendar 或环境的 machine availability 摘要中读取；
2. `num_feasible_candidates_m`：对每台候选 Machine 调用 `build_candidate_pool(machine)` 后统计真实有效候选数量，不统计 padding；
3. `machine_id_m`：直接使用 Machine 编号作为最后一层稳定排序键。
```

- [ ] **Step 3: Verify Machine selection wording**

Run:

```bash
python - <<'PY'
from pathlib import Path
text = Path('项目方案.md').read_text(encoding='utf-8')
required = [
    '### 5.2 多台 Machine 同时可调度时的规则',
    'machine_t = argmin_m (next_available_time_m, num_feasible_candidates_m, machine_id_m)',
    '候选动作更少、更受限的 Machine',
    '`select_next_machine(machines)` 必须实现 Section 5.2 的确定性规则',
    '不统计 padding',
]
for item in required:
    assert item in text, item
assert '推荐先采用**按 Machine 编号从小到大**' not in text
print('Task 2 machine selection checks passed')
PY
```

Expected output:

```text
Task 2 machine selection checks passed
```

- [ ] **Step 4: Commit only if explicitly authorized**

If the user explicitly requested commits before execution, run:

```bash
git add 项目方案.md
git commit -m "docs: align phase2 machine selection rule"
```

Expected output includes a new commit hash and the message `docs: align phase2 machine selection rule`.

---

### Task 3: Make SAS transition storage match the report

**Files:**
- Modify: `项目方案.md:432-452`
- Modify: `项目方案.md:675-710`
- Test: Python text assertion against `项目方案.md`

- [ ] **Step 1: Replace the training data flow block in Section 9**

In `项目方案.md`, find this exact block:

```markdown
```text
1. 创建 encoder
2. 创建 ResourceCalendarEnv
3. 创建 Phase2EpisodeDriver
4. 构建 observation
5. policy 输出 masked action distribution
6. 训练时 sample action_index
7. 调用 env.sas_step(machine, action_index, pool, reward_config)
8. 记录 transition（插入失败只返回失败惩罚，不临时替换为另一个动作）
9. 若无调度动作则规则推进时间
10. episode 结束后计算 advantage / return
11. PPO update
12. 定期 greedy inference 验证
```
```

Replace it with this exact block:

```markdown
```text
1. 创建 encoder
2. 创建 ResourceCalendarEnv
3. 创建 Phase2EpisodeDriver
4. 构建 obs_t：candidate_features、candidate_mask、global_features、action_indices、valid_action_count
5. policy 输出 masked action distribution
6. 训练时 sample action_index，并记录 logπ_old 与 value
7. 调用 env.sas_step(machine, action_index, pool, reward_config)
8. 记录 transition：
   - machine_id / current_time
   - obs_t 展开字段：candidate_features、candidate_mask、global_features、action_indices、valid_action_count
   - action_index / logπ_old / value / reward / done
   - obs_{t+1}：next_observation
   - info：selected_lot、selected_ppid、insertion_success、insertion_failed、mask_invalid、wait_or_noop、selected_lot_start、selected_lot_end、selected_lot_process_time、new_qtime_violation、priority_rank_penalty、reward_execute、reward_wait、reward_shape、reward_terminal
9. 插入失败只返回失败惩罚，不临时替换为另一个动作
10. 若无调度动作则规则推进时间
11. episode 结束后计算 advantage / return
12. PPO update
13. 定期 greedy inference 验证
```
```

- [ ] **Step 2: Replace Section 10.5 `phase2_ppo_buffer.py` draft**

In `项目方案.md`, replace the entire current Section 10.5 code block with this exact section:

```markdown
### 10.5 `phase2_ppo_buffer.py`

```python
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StepInfo:
    selected_lot: int | None = None
    selected_ppid: int | None = None
    insertion_success: bool = False
    insertion_failed: bool = False
    mask_invalid: bool = False
    wait_or_noop: bool = False
    selected_lot_start: float | None = None
    selected_lot_end: float | None = None
    selected_lot_process_time: float | None = None
    new_qtime_violation: float = 0.0
    priority_rank_penalty: float = 0.0
    reward_execute: float = 0.0
    reward_wait: float = 0.0
    reward_shape: float = 0.0
    reward_terminal: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class RolloutStep:
    machine_id: int
    current_time: float
    candidate_features: object
    candidate_mask: object
    global_features: object
    action_indices: object
    valid_action_count: int
    action: int
    log_prob: float
    value: float
    reward: float
    done: bool
    next_observation: object | None
    info: StepInfo


class Phase2RolloutBuffer:
    def __init__(self, gamma: float = 0.99, gae_lambda: float = 0.95) -> None:
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.steps: list[RolloutStep] = []
        self.returns: list[float] = []
        self.advantages: list[float] = []

    def add(self, step: RolloutStep) -> None:
        self.steps.append(step)

    def finish_episode(self, last_value: float = 0.0) -> None:
        self.compute_returns_and_advantages(last_value=last_value)

    def compute_returns_and_advantages(self, last_value: float = 0.0) -> None:
        returns: list[float] = []
        advantages: list[float] = []
        gae = 0.0
        next_value = last_value
        for step in reversed(self.steps):
            non_terminal = 0.0 if step.done else 1.0
            delta = step.reward + self.gamma * next_value * non_terminal - step.value
            gae = delta + self.gamma * self.gae_lambda * non_terminal * gae
            advantages.append(gae)
            returns.append(gae + step.value)
            next_value = step.value
        self.advantages = list(reversed(advantages))
        self.returns = list(reversed(returns))

    def get_training_batches(self, batch_size: int):
        raise NotImplementedError('Batch collation is implemented in Task 6 of the Phase 2 code plan')

    def clear(self) -> None:
        self.steps.clear()
        self.returns.clear()
        self.advantages.clear()
```

说明：

- `candidate_features`、`candidate_mask`、`global_features`、`action_indices` 和 `valid_action_count` 是报告中 `obs_t` 的展开形式；
- `next_observation` 对应报告中的 `obs_{t+1}`，terminal step 可记录为 `None`；
- `machine_id` 对应报告中的 `machine_t`；
- `info` 保留 reward 分解与资源日历执行结果，用于日志、debug 和验收；
- PPO 更新仍主要使用 `candidate_features`、`candidate_mask`、`global_features`、`action`、`log_prob`、`value`、`reward` 和 `done`，其余字段用于对齐报告和诊断训练行为。
```

- [ ] **Step 3: Verify transition fields**

Run:

```bash
python - <<'PY'
from pathlib import Path
text = Path('项目方案.md').read_text(encoding='utf-8')
required = [
    'class StepInfo:',
    'selected_lot: int | None = None',
    'reward_terminal: float = 0.0',
    'class RolloutStep:',
    'machine_id: int',
    'current_time: float',
    'action_indices: object',
    'valid_action_count: int',
    'next_observation: object | None',
    'info: StepInfo',
    'obs_t 的展开形式',
    'obs_{t+1}',
]
for item in required:
    assert item in text, item
print('Task 3 transition storage checks passed')
PY
```

Expected output:

```text
Task 3 transition storage checks passed
```

- [ ] **Step 4: Commit only if explicitly authorized**

If the user explicitly requested commits before execution, run:

```bash
git add 项目方案.md
git commit -m "docs: align phase2 rollout transition fields"
```

Expected output includes a new commit hash and the message `docs: align phase2 rollout transition fields`.

---

### Task 4: Add explicit episode termination conditions and driver counters

**Files:**
- Modify: `项目方案.md:284-292`
- Modify: `项目方案.md:503-545`
- Test: Python text assertion against `项目方案.md`

- [ ] **Step 1: Insert Section 5.4 after Section 5.3**

In `项目方案.md`, find this exact block:

```markdown
因此，阶段2的时间推进不由学习控制，而由规则控制。

---
```

Replace it with this exact block:

```markdown
因此，阶段2的时间推进不由学习控制，而由规则控制。

### 5.4 Episode 终止条件

阶段2的 episode 必须显式返回 `done` 和 `termination_reason`。终止条件与报告第9.8节保持一致：

1. 所有 Lot 均已完成 Lot-level 与 wafer-level 排程；
2. `current_time` 超过当前 planning horizon，且没有可继续调度的已到达 Lot；
3. 候选池为空、未来无 Lot 到达、未来无 Machine 释放，环境无法继续推进；
4. `total_wait_steps_per_episode > max_total_wait_steps_per_episode`；
5. `consecutive_failed_actions > max_failed_actions`，其中 `max_failed_actions` 推荐默认取 `3 × K_action`；
6. 出现资源日历不可恢复错误，例如已提交区间冲突、Lot / wafer stage 缺失或状态回滚失败。

`Phase2EpisodeDriver` 应维护以下计数与状态：

```text
total_wait_steps_per_episode
consecutive_failed_actions
unrecoverable_error
termination_reason
```

每次成功插入动作后，`consecutive_failed_actions` 归零；每次 mask invalid、dry-run 失败或 commit rollback 后，`consecutive_failed_actions` 加一；每次 wait / no-op 后，`total_wait_steps_per_episode` 加一。

---
```

- [ ] **Step 2: Update the `Phase2EpisodeDriver.__init__` signature in Section 10.2**

In `项目方案.md`, find this exact signature block:

```python
    def __init__(
        self,
        env,
        observation_encoder,
        reward_config,
        max_steps: int = 10000,
    ) -> None:
        ...
```

Replace it with this exact signature block:

```python
    def __init__(
        self,
        env,
        observation_encoder,
        reward_config,
        planning_horizon: float | None = None,
        max_steps: int = 10000,
        max_total_wait_steps_per_episode: int = 1000,
        max_failed_actions: int | None = None,
    ) -> None:
        ...
```

- [ ] **Step 3: Add driver termination methods in Section 10.2**

In the same `Phase2EpisodeDriver` signature block, find this exact method list portion:

```python
    def advance_to_next_event(self) -> float | None:
        ...

    def run_policy_episode(
```

Replace it with this exact method list portion:

```python
    def advance_to_next_event(self) -> float | None:
        ...

    def record_step_result(self, step_result) -> None:
        ...

    def is_episode_done(self) -> tuple[bool, str]:
        ...

    def run_policy_episode(
```

- [ ] **Step 4: Add driver counter explanation after the Section 10.2 implementation guidance**

In `项目方案.md`, after the list that ends with this exact line:

```markdown
3. `machine_id_m`：直接使用 Machine 编号作为最后一层稳定排序键。
```

Insert this exact text:

```markdown

`record_step_result(step_result)` 负责更新 driver 计数器：

1. `step_result.committed = True` 时，`consecutive_failed_actions = 0`；
2. `step_result.mask_invalid = True`、`step_result.insertion_failed = True` 或 commit rollback 时，`consecutive_failed_actions += 1`；
3. `step_result.wait_or_noop = True` 时，`total_wait_steps_per_episode += 1`；
4. 出现资源日历不可恢复错误时，设置 `unrecoverable_error = True` 并写入 `termination_reason`。

`is_episode_done()` 返回 `(done, termination_reason)`，其中 `termination_reason` 必须来自 Section 5.4 的六类终止条件之一。
```

- [ ] **Step 5: Verify termination conditions and driver counters**

Run:

```bash
python - <<'PY'
from pathlib import Path
text = Path('项目方案.md').read_text(encoding='utf-8')
required = [
    '### 5.4 Episode 终止条件',
    'done` 和 `termination_reason`',
    'total_wait_steps_per_episode > max_total_wait_steps_per_episode',
    'consecutive_failed_actions > max_failed_actions',
    '3 × K_action',
    'unrecoverable_error',
    'planning_horizon: float | None = None',
    'max_total_wait_steps_per_episode: int = 1000',
    'max_failed_actions: int | None = None',
    'def record_step_result(self, step_result) -> None:',
    'def is_episode_done(self) -> tuple[bool, str]:',
]
for item in required:
    assert item in text, item
print('Task 4 episode termination checks passed')
PY
```

Expected output:

```text
Task 4 episode termination checks passed
```

- [ ] **Step 6: Commit only if explicitly authorized**

If the user explicitly requested commits before execution, run:

```bash
git add 项目方案.md
git commit -m "docs: specify phase2 episode termination rules"
```

Expected output includes a new commit hash and the message `docs: specify phase2 episode termination rules`.

---

### Task 5: Expand tests and acceptance criteria to cover the alignment gaps

**Files:**
- Modify: `项目方案.md:1007-1038`
- Test: Python text assertion against `项目方案.md`

- [ ] **Step 1: Replace Section 13.1 required tests**

In `项目方案.md`, replace the current Section 13.1 list with this exact section:

```markdown
### 13.1 必做测试

1. **候选池 + mask 测试**
   - padding 不会被提交；
   - masked 位置不会被 greedy 或 sample 选中；
   - 当所有候选都被 mask 时，不执行 SAS softmax，而是由 driver 规则推进时间。

2. **Machine 选择规则测试**
   - 多台 Machine 同时可调度时，选择 `(next_available_time_m, num_feasible_candidates_m, machine_id_m)` 字典序最小的 Machine；
   - 当两台 Machine 空闲时间相同，优先选择真实有效候选数更少的 Machine；
   - 当前两项都相同，使用 Machine 编号作为稳定 tie-breaker。

3. **规则触发测试**
   - 有可调度 Machine 时能触发 SAS；
   - 无可调度 Machine 时能推进到下一事件；
   - 时间推进后 `current_time` 必须严格增加，除非 episode 已经终止。

4. **Transition 字段测试**
   - `RolloutStep` 记录 `machine_id`、`current_time`、`action_indices` 和 `valid_action_count`；
   - `RolloutStep.info` 记录 selected Lot / PPID、插入成功失败、mask invalid、wait/no-op 和 reward 分解；
   - `next_observation` 与下一次 SAS 决策观测一致，terminal step 可为 `None`。

5. **Episode 终止条件测试**
   - 所有 Lot 完成时终止；
   - 超过 planning horizon 且没有可继续调度 Lot 时终止；
   - 候选池为空且未来无 arrival / Machine release 时终止；
   - 超过 `max_total_wait_steps_per_episode` 时终止；
   - 超过 `max_failed_actions` 时终止；
   - 资源日历不可恢复错误时终止，并返回明确 `termination_reason`。

6. **单 episode 闭环测试**
   - 可以从初始状态执行到完成；
   - 不出现死循环或状态损坏；
   - driver 返回 episode reward、完成 Lot 数、step 数、wait 数、失败动作数和 `termination_reason`。

7. **PPO 烟雾测试**
   - rollout 后 advantage 可正确计算；
   - 一次 update 能成功 backward；
   - PPO 更新只使用训练阶段真实采样的 action，不使用推理 fallback 替换动作。

8. **最终 schedule 校验**
   - `validate_schedule()` 返回通过；
   - 不产生 machine / chamber 时间冲突；
   - Lot / Wafer 完整性校验通过。
```

- [ ] **Step 2: Replace Section 13.2 success signs**

In `项目方案.md`, find this exact Section 13.2 block:

```markdown
### 13.2 成功标志

若满足以下条件，则可认为阶段2实现成功：

- 训练脚本可稳定运行多个 episode；
- 推理脚本能生成一个合法 schedule；
- reward 曲线不崩溃；
- 最终 schedule 能通过校验；
- 现有阶段1环境逻辑未被破坏。
```

Replace it with this exact block:

```markdown
### 13.2 成功标志

若满足以下条件，则可认为阶段2实现成功：

- 训练脚本可稳定运行多个 episode；
- 推理脚本能生成一个合法 schedule；
- reward 曲线不崩溃；
- 最终 schedule 能通过校验；
- 现有阶段1环境逻辑未被破坏；
- Machine 选择规则与报告第9.1节一致；
- rollout 中每个 SAS transition 都能追溯 `machine_id`、`obs_t`、`action_index`、`logπ_old`、`reward`、`obs_{t+1}` 和 `info`；
- episode 结束时总能给出明确 `termination_reason`；
- 训练阶段不使用推理 fallback 替换采样动作。
```

- [ ] **Step 3: Verify acceptance criteria**

Run:

```bash
python - <<'PY'
from pathlib import Path
text = Path('项目方案.md').read_text(encoding='utf-8')
required = [
    '**Machine 选择规则测试**',
    '**Transition 字段测试**',
    '**Episode 终止条件测试**',
    '字典序最小的 Machine',
    'RolloutStep.info',
    'terminal step 可为 `None`',
    '返回明确 `termination_reason`',
    '训练阶段真实采样的 action',
    'Machine 选择规则与报告第9.1节一致',
    '训练阶段不使用推理 fallback 替换采样动作',
]
for item in required:
    assert item in text, item
print('Task 5 acceptance criteria checks passed')
PY
```

Expected output:

```text
Task 5 acceptance criteria checks passed
```

- [ ] **Step 4: Commit only if explicitly authorized**

If the user explicitly requested commits before execution, run:

```bash
git add 项目方案.md
git commit -m "docs: expand phase2 alignment acceptance criteria"
```

Expected output includes a new commit hash and the message `docs: expand phase2 alignment acceptance criteria`.

---

### Task 6: Run the full document alignment check

**Files:**
- Verify: `项目方案.md`
- No source changes in this task

- [ ] **Step 1: Run the full text assertion check**

Run:

```bash
python - <<'PY'
from pathlib import Path
text = Path('项目方案.md').read_text(encoding='utf-8')
required = [
    '补充对齐要求：',
    'machine_t = argmin_m (next_available_time_m, num_feasible_candidates_m, machine_id_m)',
    '`select_next_machine(machines)` 必须实现 Section 5.2 的确定性规则',
    'class StepInfo:',
    'class RolloutStep:',
    'machine_id: int',
    'next_observation: object | None',
    'info: StepInfo',
    '### 5.4 Episode 终止条件',
    'total_wait_steps_per_episode',
    'consecutive_failed_actions',
    'unrecoverable_error',
    'termination_reason',
    '**Machine 选择规则测试**',
    '**Transition 字段测试**',
    '**Episode 终止条件测试**',
]
for item in required:
    assert item in text, item
for forbidden in [
    '推荐先采用**按 Machine 编号从小到大**',
    '记录 transition（插入失败只返回失败惩罚，不临时替换为另一个动作）',
]:
    assert forbidden not in text, forbidden
print('Full phase2 plan alignment checks passed')
PY
```

Expected output:

```text
Full phase2 plan alignment checks passed
```

- [ ] **Step 2: Review the changed Markdown sections**

Run:

```bash
git diff -- 项目方案.md
```

Expected result:

```text
The diff only changes 项目方案.md and contains updates to Sections 1.1, 5.2, 5.4, 9, 10.2, 10.5, 13.1, and 13.2.
```

- [ ] **Step 3: Commit only if explicitly authorized**

If the user explicitly requested commits before execution and earlier task commits were not created, run:

```bash
git add 项目方案.md
git commit -m "docs: align phase2 sas ppo project plan with report"
```

Expected output includes a new commit hash and the message `docs: align phase2 sas ppo project plan with report`.

---

## Self-Review

**Spec coverage:** This plan covers all previously identified gaps: `RolloutStep.machine_id`, `RolloutStep.info`, deterministic Machine selection, explicit episode termination conditions, and transition observation structure.

**Placeholder scan:** The plan contains concrete Markdown replacements, concrete Python assertion commands, and concrete verification output. It does not require unresolved design choices before editing `项目方案.md`.

**Type consistency:** `StepInfo`, `RolloutStep`, `Phase2EpisodeDriver.record_step_result`, and `Phase2EpisodeDriver.is_episode_done` are named consistently across tasks. `machine_id` is used for code-facing dataclass fields, while `machine_t` is used only in report-alignment formulas.
