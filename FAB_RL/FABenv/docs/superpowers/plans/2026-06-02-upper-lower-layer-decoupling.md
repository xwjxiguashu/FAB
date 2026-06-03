# 上下层解耦重构（报告 §1.5）：下层唯一真相 + 窄接口

> REQUIRED SUB-SKILL: superpowers:subagent-driven-development

**目标:** 按已批准的设计文档 `docs/superpowers/specs/2026-06-02-upper-lower-layer-decoupling-design.md`（方案 A + 五点），把当前在 `rl_environment.py` 里重写的一整套下层排程（组批 + 实例选择 + 时序排程 + 机台/腔体两级收敛 + 噪声）收敛为下层唯一真相。下层暴露两个共享同一 list-scheduling 核心的窄接口：状态无关的 `estimate()`（保留不动）与新增的状态相关 `schedule_on_calendar()`。上层 dry-run / commit 改为薄封装，只负责「校验 / 持久化 / 组装 numpy schema」。删除上层重复函数与死代码 `_prof.py`。

**已接受的代价（设计 §2.1、§10）:** 实例选择规则从上层「`find_earliest_slot` 字典序」统一到下层「`argmin max(ready, free)`」，lot/wafer 的 start/end 会偏移，利用率/违规数可能微变，已训练 checkpoint 的数值可比性下降可接受。**收益:** qtime mask 预判的 makespan 与 commit 实际 makespan 首次一致，根治 CLAUDE.md §1.5 记录的历史 bug。

**不在范围（设计 §12）:** DDT / 前瞻预留 / attention / HGNN / PPO-Lagrangian 调参。不改 RL 策略、奖励通道、候选池流水线语义（mask→filter→score 顺序不变）。

---

## 工作目录与运行约定

- 所有命令、所有 pytest 均从 `FAB_RL/FABenv/` 为工作目录运行（包用裸 import，工作目录须在 `sys.path` 上）。
- 测试用 `python -m pytest tests/<file>.py -q` 运行。
- 冒烟 demo：`python run_phase1_environment_demo.py`、`python run_phase2_sas_inference_demo.py`。
- 不要 git commit，除非用户明确要求。

---

## 接口契约（所有任务必须严格遵守）

### 共享核心（放 `lower_layer_estimator.py` 内，设计 §13）

现有 `_run_list_schedule(sub_batches, stage_times, n_stages, instance_counts)` 已是「实例 free_time 初值=0、`argmin(instance_free[s])` 选实例」的纯核心。本次新增一个**带绝对资源键、可指定 free 初值、返回占用区间**的核心，供 `estimate` step6 与 `schedule_on_calendar` 共用：

```python
def schedule_deterministic(
    sub_batches,            # list[int]，每个子批 wafer 数（仅用 len 与 wafer 展开）
    stage_times,            # (n_batches, n_stages) ndarray，各子批各阶段实际加工时长
    stage_resource_options, # list[list[(chamber, side, base_pt)]]，每阶段可用实例（顺序固定）
    machine,                # int，用于组装 resource_key=(machine, chamber, side)
    instance_free_init,     # dict[resource_key -> float]，各实例起始空闲时刻（缺失视为 0.0）
    lot_release_time=0.0,   # float，第一阶段最早就绪时刻（绝对）
):
    """确定性 list scheduling，返回 (lot_start, lot_end, batch_intervals)。

    规则（与 _run_list_schedule 一致，扩展为带绝对资源键 + 任意 free 初值）：
      for 每个子批 b（FIFO）:
        ready = lot_release_time
        for 每个阶段 s:
          对该阶段每个候选实例 i: cand_start = max(ready, free[resource_key_i])
          选 cand_start 最小者（并列取 options 顺序靠前者，与 argmin 一致）
          start = cand_start_best; end = start + stage_times[b, s]
          free[resource_key_best] = end; ready = end
    返回:
      lot_start = min over (b, s) start
      lot_end   = max over (b, s) end   （= 各子批末阶段 end 的最大值）
      batch_intervals: list[(b_index, stage_index_1based, resource_key, start, end)]
    """
```

要点：
- `instance_free_init` 用 `resource_key=(machine, chamber, side)` 作键；这就是 free 初值=0（估时）还是真实日历空闲（排程）的唯一区别。
- 选实例规则 = 「使 `start=max(ready, free)` 最小的实例」，并列时取 `stage_resource_options[s]` 中靠前者。这与现有 `_run_list_schedule` 的 `argmin(instance_free[s])` 在 free 初值=0 且 ready 单调时等价（见 Task 2 一致性断言）。
- 纯函数，**不读 encoder、不读 state、不碰日历对象**；只吃数组与 free 初值 dict。

### 新增下层接口 `schedule_on_calendar`（放新建 `lower_layer_scheduler.py`，设计 §5.2）

```python
@dataclass
class ScheduleResult:
    lot_start: float
    lot_end: float
    batch_intervals: list           # [(resource_key, start, end), ...]，resource_key=(machine,chamber,side)
    machine_interval: tuple         # (machine, lot_start, lot_end)
    subbatch_wafer_map: list        # [[wafer_id, ...], ...]，每个子批的 wafer-id 列表（1-based）
    infeasible_reason: str = ""     # 成功为 ""，失败为原因串

def schedule_on_calendar(lot, machine, ppid, encoder, calendar_state,
                         earliest_release, noise_rng=None) -> ScheduleResult:
    ...
```

语义（设计 §5.2、§6、§7）：
- 读 `calendar_state.machine_calendar` / `chamber_calendar` 得各实例当前空闲时刻作 free 初值。
- `noise_rng=None` → 用各阶段 μ（dry-run / 规划）；传入 rng → 按 stage σ 采样 `μ + N(0,σ)`（commit / 执行，报告 §2.4.6），逐 (子批, stage) 采一次，clamp ≥ 1e-6。
- 含机台/腔体两级槽位收敛（从上层 `_dry_run_candidate`/`_simulate_action` 下沉）：先按 free 初值排子批得 `lot_end`，再用 `find_earliest_slot` 求机台槽位起点；若与 `lot_release_time` 不一致则以新起点重排，最多 20 次；不收敛 → `infeasible_reason="calendar_no_stable_slot"`。
- **非破坏性**：为推进 free 可临时占用，返回前必须逐字节还原，**绝不改变传入的 `calendar_state`**。登记与否由上层决定。
- 失败（任一阶段无可用实例 / 不收敛）→ 返回 `ScheduleResult(infeasible_reason=...)`，其余字段置空/默认；不抛异常。

### 下沉的辅助（从 `rl_environment.py` 移入 `lower_layer_scheduler.py`，设计 §8）

- `_allowed_resources_for(machine)` 的逻辑 → 下层内部按 `(encoder, machine)` 读 `encoder.machine_resources`，返回允许的 `(chamber, side)` frozenset（None=不过滤）。可在下层自带一个 `{machine: frozenset}` 缓存。
- `_stage_process_sigma(lot, machine, ppid, stage_id)` 的逻辑 → 下层读 `encoder.process_time_sigma`。**噪声来源仍在上层**（环境持有 `_noise_rng` 与 `process_noise_enabled` 决定是否传 rng），噪声应用在下层。

### 上层保留不动

`estimate` / `_estimate_cache` / `qtime_safe_mask`（n_mc=20）/ `is_doomed`（n_mc=10）/ `build_candidate_pool` / `sas_step` / reward 全家 / `rollback_last_commit` / `validate_schedule` / `_candidate_features`（消费的 dry_run dict 字段 key 不变）。

---

## 文件结构

```
FAB_RL/FABenv/
  lower_layer_estimator.py     ← 改：新增 schedule_deterministic；estimate step6 复用它
  lower_layer_scheduler.py     ← 新建：ScheduleResult + schedule_on_calendar + 下沉的 allowed/sigma
  rl_environment.py            ← 改：_dry_run_candidate / _simulate_action 薄封装；删 4 个旧函数
  _prof.py                     ← 删
  tests/
    conftest.py                ← 新建：sys.path 注入 + 共享 fixtures（若 pytest 套件已空）
    test_decoupling_consistency.py  ← 新建：estimate.μ == schedule_on_calendar 空日历 lot_end
    test_decoupling_rollback.py     ← 新建：dry-run 后日历逐字节还原、非破坏性
```

> `tests/` 目录下的旧测试已删（git status 显示大量 `D tests/test_*.py`）。Task 0 重建 `conftest.py`，本次只加针对性测试，不恢复旧套件（设计 §13）。

---

## Task 0：重建 tests/ 基础设施

**做什么:** 确认/创建 `tests/conftest.py`，让 pytest 从 `FAB_RL/FABenv/` 跑时能裸 import 包模块，并提供构造 encoder/env 的共享 fixture。

**新建 `tests/conftest.py`:**

```python
"""pytest 配置：把 FABenv 目录注入 sys.path（包用裸 import），并提供共享 fixtures。"""
import os
import sys

import pytest

FABENV_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if FABENV_DIR not in sys.path:
    sys.path.insert(0, FABENV_DIR)


@pytest.fixture
def small_encoder():
    from problem_instances import build_small_encoder
    return build_small_encoder()


@pytest.fixture
def small_env(small_encoder):
    from rl_environment import ResourceCalendarEnv
    env = ResourceCalendarEnv(small_encoder, top_k=8)
    env.reset()
    return env
```

**验证:** `python -m pytest tests/ -q --collect-only` 不报 import 错误。已核实：小实例工厂为 `build_small_encoder()`（`problem_instances.py:245`，4×2 实例），压力实例为 `build_pressure_test_encoder(seed=2026)`（:387）。`ScheduleState()` 无参即空日历（`state.py:11`，所有字段 default_factory）。

---

## Task 1：下层新增 `schedule_deterministic`（TDD）

**做什么:** 在 `lower_layer_estimator.py` 新增上面契约里的 `schedule_deterministic(...)` 纯函数。先写测试再写实现。

**先写测试 `tests/test_decoupling_consistency.py`（本任务部分）:**

```python
import numpy as np
from lower_layer_estimator import schedule_deterministic


def test_schedule_deterministic_empty_free_single_batch():
    # 1 子批，2 阶段，每阶段 1 实例，free 初值空 → 串行 0..pt1..pt1+pt2
    sub_batches = [4]
    stage_times = np.array([[3.0, 5.0]])
    options = [[(1, 1, 3.0)], [(2, 1, 5.0)]]
    free = {}
    lot_start, lot_end, intervals = schedule_deterministic(
        sub_batches, stage_times, options, machine=1,
        instance_free_init=free, lot_release_time=0.0,
    )
    assert lot_start == 0.0
    assert lot_end == 8.0
    assert intervals[0] == (0, 1, (1, 1, 1), 0.0, 3.0)
    assert intervals[1] == (0, 2, (1, 2, 1), 3.0, 8.0)


def test_schedule_deterministic_respects_free_init():
    # 同上但 stage1 实例已忙到 t=10 → 第一阶段从 10 开始
    sub_batches = [4]
    stage_times = np.array([[3.0, 5.0]])
    options = [[(1, 1, 3.0)], [(2, 1, 5.0)]]
    free = {(1, 1, 1): 10.0}
    lot_start, lot_end, intervals = schedule_deterministic(
        sub_batches, stage_times, options, machine=1,
        instance_free_init=free, lot_release_time=0.0,
    )
    assert lot_start == 10.0
    assert lot_end == 18.0


def test_schedule_deterministic_picks_earliest_instance():
    # 2 实例，free 不同 → 选 start 最小者
    sub_batches = [4]
    stage_times = np.array([[3.0]])
    options = [[(1, 1, 3.0), (1, 2, 3.0)]]
    free = {(1, 1, 1): 5.0, (1, 1, 2): 0.0}
    _, lot_end, intervals = schedule_deterministic(
        sub_batches, stage_times, options, machine=1,
        instance_free_init=free, lot_release_time=0.0,
    )
    assert intervals[0][2] == (1, 1, 2)   # 选了 free=0 的实例
    assert lot_end == 3.0
```

**再实现:** 按契约写函数。实现要点：
- `n_batches = len(sub_batches)`，`n_stages = stage_times.shape[1]`。
- `free` 取传入 dict 的副本（`dict(instance_free_init)`，不改原 dict）。
- 双重循环 b × s；每个 stage 遍历 `stage_resource_options[s]` 的候选实例，算 `cand_start = max(ready, free.get((machine,ch,sd), 0.0))`，取最小（并列取靠前）；`end = cand_start + stage_times[b,s]`；更新 `free[key]=end`、`ready=end`。
- 收集 `(b, s_1based, key, start, end)`。`lot_start=min(all start)`，`lot_end=max(all end)`。

**验证:** `python -m pytest tests/test_decoupling_consistency.py -q`（三条全过）。

---

## Task 2：`estimate` step6 复用 `schedule_deterministic`（一致性断言）

**做什么:** 把 `estimate()` 的 step6（行 325–351，手写的 `instance_free_mu` + `argmin` 重建 `per_instance_occupancy`）改为调用 `schedule_deterministic`，free 初值=空 dict、`lot_release_time=0.0`、`stage_times=μ 平铺`。**保证 `mu_finish`、`per_instance_occupancy`、`bottleneck_stage` 数值不变。**

**先加断言测试（追加到 `tests/test_decoupling_consistency.py`）:**

```python
def test_estimate_step6_unchanged_after_refactor(small_encoder):
    # 对每个可行 (lot, machine, ppid)，estimate 的 per_instance_occupancy 区间数与
    # mu_finish 应与 schedule_deterministic 空日历重排吻合
    from lower_layer_estimator import estimate
    enc = small_encoder
    lot, machine = 1, int(enc.get_machine_list(1)[0])
    ppid = int(enc.get_ppid_list(lot, machine)[0])
    res = estimate(lot, machine, ppid, enc, state=None, n_mc=1)
    # n_mc=1 且 σ 走 μ 路径时，mu_finish 应等于 occupancy 中末阶段最大 end
    max_end = max(e for _, _s, e in res["per_instance_occupancy"])
    assert abs(res["mu_finish"] - max_end) < 1e-6
```

> 注意：`estimate` 现在 step5 蒙特卡洛与 step6 均值路径分开。`mu_finish` 来自 MC，可能不严格等于均值路径 makespan（MC 的 E[max] ≥ max(E)）。因此该断言用 `n_mc=1` 且确保走 μ 路径——核对 `monte_carlo_makespan`：当 `stage_sigma` 全 0 时用 `np.tile(stage_mu)`，makespan 即均值路径，与 step6 一致。**实现 Task 2 时**：若 σ 非 0 导致 MC≠step6，改断言为「step6 occupancy 的 makespan 等于 σ=0 时的均值路径 makespan」而非比 `mu_finish`。先 grep 确认 small 实例是否有 `process_time_sigma`。

**再改 `estimate` step6:** 用
```python
mu_stage_times = np.tile(np.asarray(stage_mu), (n_batches, 1))
_, _, occ_intervals = schedule_deterministic(
    sub_batches, mu_stage_times, stage_resource_options,
    machine, instance_free_init={}, lot_release_time=0.0,
)
per_instance_occupancy = [(key, s, e) for (_b, _st, key, s, e) in occ_intervals]
```
替换行 325–351 的手写循环。`bottleneck_stage`（step7）保持不变。

**验证:** 上面断言 + `python -m pytest tests/test_decoupling_consistency.py -q` 全过。再跑一次 `python run_phase1_environment_demo.py` 确认 estimate 路径无异常。

---

## Task 3：新建 `lower_layer_scheduler.py` + `schedule_on_calendar`（TDD）

**做什么:** 新建文件，实现 `ScheduleResult` + `schedule_on_calendar` + 下沉的 `_allowed_resources` / `_stage_sigma` helper。核心算法复用 Task 1 的 `schedule_deterministic`，外层套机台/腔体两级收敛（从上层 `_dry_run_candidate` 行 1706–1775 的迭代逻辑搬下来），并从 `calendar_state` 读 free 初值。

**关键实现骨架:**

```python
"""下层状态相关日历排程引擎（报告 §1.5、§2.4.6）。

与 lower_layer_estimator.estimate() 共享 schedule_deterministic 核心：
estimate 状态无关（free 初值=0、可缓存），本模块状态相关（free 初值=真实日历空闲）。
非破坏性：不改传入的 calendar_state。
"""
from dataclasses import dataclass, field

import numpy as np

from lower_layer_estimator import compute_sub_batches, schedule_deterministic


@dataclass
class ScheduleResult:
    lot_start: float = 0.0
    lot_end: float = 0.0
    batch_intervals: list = field(default_factory=list)
    machine_interval: tuple = None
    subbatch_wafer_map: list = field(default_factory=list)
    infeasible_reason: str = ""


def _allowed_resources(encoder, machine):
    declared = getattr(encoder, "machine_resources", {})
    if not declared:
        return None
    return frozenset(
        (int(ch), int(sd)) for ch, sd in declared.get(int(machine), [])
    )


def _stage_sigma(encoder, lot, machine, ppid, stage_id):
    sigmas = getattr(encoder, "process_time_sigma", {}).get(
        (int(lot), int(machine), int(ppid))
    )
    if not sigmas:
        return 0.0
    idx = int(stage_id) - 1
    if idx < 0 or idx >= len(sigmas):
        return 0.0
    return max(0.0, float(sigmas[idx]))


def _build_stage_options(encoder, machine, steps):
    """每阶段 → [(chamber, side, base_pt), ...]，过滤到 machine 允许的资源集。"""
    allowed = _allowed_resources(encoder, machine)
    options = []
    for stage in steps:
        arr = np.asarray(stage, dtype=float)
        rows = []
        for r in arr:
            ch, sd, pt = int(r[0]), int(r[1]), float(r[2])
            if allowed is not None and (ch, sd) not in allowed:
                continue
            rows.append((ch, sd, pt))
        if not rows:  # fallback：全资源
            rows = [(int(r[0]), int(r[1]), float(r[2])) for r in arr]
        options.append(rows)
    return options


def _free_init_from_calendar(calendar_state, machine, stage_options):
    """从真实日历读各实例当前空闲时刻（= 该资源最后一个区间的 end，无则 0）。"""
    cc = calendar_state.chamber_calendar
    free = {}
    for opts in stage_options:
        for ch, sd, _pt in opts:
            key = (int(machine), ch, sd)
            intervals = cc.get(key, [])
            free[key] = float(intervals[-1][1]) if intervals else 0.0
    return free


def schedule_on_calendar(lot, machine, ppid, encoder, calendar_state,
                         earliest_release, noise_rng=None):
    lot, machine, ppid = int(lot), int(machine), int(ppid)
    try:
        steps = encoder.get_process_steps(lot, machine, ppid)
    except (KeyError, ValueError):
        return ScheduleResult(infeasible_reason="ppid_stage_missing")
    if not steps:
        return ScheduleResult(infeasible_reason="ppid_stage_missing")

    wafer_count = int(encoder.wafer_counts[lot])
    side_capacity = getattr(encoder, "side_capacity", None)
    if side_capacity is None or int(side_capacity) <= 0:
        side_capacity = wafer_count
    sub_batches = compute_sub_batches(wafer_count, int(side_capacity))

    stage_options = _build_stage_options(encoder, machine, steps)
    if any(len(o) == 0 for o in stage_options):
        return ScheduleResult(infeasible_reason="chamber_side_unavailable")

    n_batches = len(sub_batches)
    n_stages = len(steps)

    # stage_times：μ 或 μ+噪声（逐 子批×stage 采样）
    base_mu = np.array([min(pt for _c, _s, pt in opts) for opts in stage_options])
    if noise_rng is None:
        stage_times = np.tile(base_mu, (n_batches, 1))
    else:
        stage_times = np.empty((n_batches, n_stages), dtype=float)
        for b in range(n_batches):
            for s in range(n_stages):
                sigma = _stage_sigma(encoder, lot, machine, ppid, s + 1)
                delta = float(noise_rng.normal(0.0, sigma)) if sigma > 0 else 0.0
                stage_times[b, s] = max(1e-6, base_mu[s] + delta)

    free_init = _free_init_from_calendar(calendar_state, machine, stage_options)
    mc = calendar_state.machine_calendar

    # 机台/腔体两级收敛（从上层下沉）
    lot_release_time = encoder.find_earliest_slot(
        mc.get(machine, []), float(earliest_release), 0.0,
    )
    for _ in range(20):
        lot_start, lot_end, intervals = schedule_deterministic(
            sub_batches, stage_times, stage_options, machine,
            instance_free_init=free_init, lot_release_time=lot_release_time,
        )
        lot_duration = max(0.0, lot_end - lot_release_time)
        slot = encoder.find_earliest_slot(
            mc.get(machine, []), float(earliest_release), lot_duration,
        )
        if abs(slot - lot_release_time) <= 1e-9:
            break
        lot_release_time = slot
    else:
        return ScheduleResult(infeasible_reason="calendar_no_stable_slot")

    batch_intervals = [(key, s, e) for (_b, _st, key, s, e) in intervals]
    # 子批 wafer 映射（1-based 连续切分）
    subbatch_wafer_map = []
    cursor = 0
    for bsz in sub_batches:
        subbatch_wafer_map.append(list(range(cursor + 1, cursor + bsz + 1)))
        cursor += bsz

    return ScheduleResult(
        lot_start=lot_start,
        lot_end=lot_end,
        batch_intervals=batch_intervals,
        machine_interval=(machine, lot_release_time, lot_end),
        subbatch_wafer_map=subbatch_wafer_map,
        infeasible_reason="",
    )
```

> **收敛逻辑注意:** 旧上层在每次迭代真往日历 add 区间再 rollback（因它要校验插入无冲突）。这里 `schedule_deterministic` 用 free 初值算，无需碰真实日历——free 初值来自日历各实例最后 end，已含已提交占用；机台层用 `find_earliest_slot` 求起点。`free_init` 在循环内不变（实例空闲只取决于已提交状态），变化的只是 `lot_release_time`，故每次重算即可。**非破坏性天然满足**（全程不 add 真实日历）。
>
> **腔体空档回填差异:** 旧上层每阶段用 `find_earliest_slot` 可回填腔体日历中间空档；新核心用 free 初值（=最后 end）顺排，不回填中间空档。这是设计 §10 明示接受的数值变化（统一到下层 `argmin max(ready,free)` 口径）。若 Task 8 回归显示 `pressure` 利用率明显恶化，再评估是否给 `schedule_deterministic` 加「按 busy 区间找空档」选项——但**默认不做**，保持与 estimate 口径一致。

**测试（追加到 `tests/test_decoupling_consistency.py`）—— 核心一致性断言（设计 §11.1）:**

```python
def test_schedule_on_calendar_matches_estimate_on_empty(small_encoder):
    from lower_layer_estimator import estimate
    from lower_layer_scheduler import schedule_on_calendar
    from state import ScheduleState
    enc = small_encoder
    empty = ScheduleState()   # 空日历；按真实构造器签名调整
    for lot in [1]:
        machine = int(enc.get_machine_list(lot)[0])
        ppid = int(enc.get_ppid_list(lot, machine)[0])
        est = estimate(lot, machine, ppid, enc, state=None, n_mc=1)
        res = schedule_on_calendar(
            lot, machine, ppid, enc, empty,
            earliest_release=float(enc.arrival_times[lot]), noise_rng=None,
        )
        assert res.infeasible_reason == ""
        # 空日历、无噪声、release=arrival：lot_end - arrival 应等于 μ 路径 makespan
        makespan = res.lot_end - res.machine_interval[1]
        assert abs(makespan - (est["mu_finish"])) < 1e-6  # 见下注
```

> `estimate.mu_finish` 是 MC 均值；当 small 实例 σ=0 时 MC==μ 路径==`schedule_on_calendar` 无噪声。若有 σ，改比 σ=0 路径（构造 σ=0 的 encoder 或直接比 step6 occupancy makespan）。**实现时按 small 实例实际 σ 情况二选一，并在测试注释写明依据。**

**验证:** `python -m pytest tests/test_decoupling_consistency.py -q` 全过。先 grep `state.py` 确认 `ScheduleState` 的真实构造方式（空日历如何建）。

---

## Task 4：上层 `_simulate_action` 改薄封装（commit 路径）

**做什么:** 把 `rl_environment.py` 的 `_simulate_action`（行 2161–2315）改为调用 `schedule_on_calendar`，删内部 20 次收敛循环与子批排程。噪声：`process_noise_enabled` 时传 `self._noise_rng`，否则 None。

**新实现:**

```python
def _simulate_action(self, action, state):
    """commit 路径：调下层 schedule_on_calendar 取区间 → 持久化 + 组装 schema。"""
    from lower_layer_scheduler import schedule_on_calendar
    lot, machine, ppid = int(action.lot), int(action.machine), int(action.ppid)
    earliest_release = max(self.current_time, float(self.encoder.arrival_times[lot]))
    rng = self._noise_rng if self.process_noise_enabled else None

    res = schedule_on_calendar(
        lot, machine, ppid, self.encoder, state,
        earliest_release=earliest_release, noise_rng=rng,
    )
    if res.infeasible_reason:
        raise RuntimeError(
            f"schedule_on_calendar failed for Lot {lot}: {res.infeasible_reason}"
        )

    # 持久化腔体区间 + 机台区间（沿用现有 add + 异常回滚语义）
    added = []
    try:
        for resource_key, start, end in res.batch_intervals:
            self.encoder.add_calendar_interval(
                state.chamber_calendar, resource_key, start, end,
            )
            added.append((resource_key, start, end))
        m_id, m_start, m_end = res.machine_interval
        self.encoder.add_calendar_interval(
            state.machine_calendar, m_id, m_start, m_end,
        )
    except Exception:
        self.encoder.rollback_calendar_intervals(state.chamber_calendar, added)
        raise

    # 更新可用时间
    state.machine_available_time[machine] = max(
        state.machine_available_time.get(machine, self.current_time),
        float(res.lot_end),
    )
    for resource_key, _s, end in res.batch_intervals:
        state.chamber_available_time[resource_key] = max(
            state.chamber_available_time.get(resource_key, self.current_time),
            float(end),
        )

    # 组装 (n,9) wafer_schedule：子批内 wafer 同进同出共享区间
    # batch_intervals 顺序 = (b, stage) 行优先；按子批分组展开
    trial_rows = []
    n_stages = len(self.encoder.get_process_steps(lot, machine, ppid))
    for b_idx, wafer_ids in enumerate(res.subbatch_wafer_map):
        stage_slice = res.batch_intervals[b_idx * n_stages:(b_idx + 1) * n_stages]
        for stage_id, (resource_key, start, end) in enumerate(stage_slice, start=1):
            _m, chamber, side = resource_key
            for wafer_id in wafer_ids:
                trial_rows.append([
                    lot, wafer_id, machine, ppid, stage_id,
                    chamber, side, start, end,
                ])

    lot_schedule = np.asarray(
        [[lot, machine, ppid, res.machine_interval[1], res.lot_end]], dtype=float,
    )
    wafer_schedule = np.asarray(trial_rows, dtype=float)
    return lot_schedule, wafer_schedule, state
```

> **批区间顺序约定:** `schedule_deterministic` 收集 `(b, stage)` 行优先，故 `batch_intervals[b*n_stages + (s-1)]` 对应子批 b 阶段 s。Task 1 实现必须保证这个顺序，Task 4 这里依赖它。在 Task 1 测试里加一条多子批顺序断言锁死。

**验证:** `python run_phase2_sas_inference_demo.py` 跑通；`python -m pytest tests/test_decoupling_rollback.py -q`（Task 6 后补全，先确保 demo 通）。

---

## Task 5：上层 `_dry_run_candidate` 改薄封装（候选特征路径）

**做什么:** 把 `_dry_run_candidate`（行 1665–1787）改为调 `schedule_on_calendar(noise_rng=None)`，组装现有 dry_run dict 字段。删内部 20 次收敛循环、删对 `_lot_sub_batches`/`_select_earliest_stage_resource` 的调用。

**新实现:**

```python
def _dry_run_candidate(self, lot, machine, ppid):
    """候选特征路径：调下层 schedule_on_calendar(μ) → 组装 dry_run dict。"""
    from lower_layer_scheduler import schedule_on_calendar
    lot, machine, ppid = int(lot), int(machine), int(ppid)
    try:
        steps = self.encoder.get_process_steps(lot, machine, ppid)
    except (KeyError, ValueError):
        return None, "ppid_stage_missing"
    if not steps:
        return None, "ppid_stage_missing"

    earliest_release = max(self.current_time, float(self.encoder.arrival_times[lot]))
    res = schedule_on_calendar(
        lot, machine, ppid, self.encoder, self.state,
        earliest_release=earliest_release, noise_rng=None,
    )
    if res.infeasible_reason:
        return None, res.infeasible_reason

    result = {
        "steps": steps,
        "lot_release_time": float(res.machine_interval[1]),
        "lot_start_time": float(res.lot_start),
        "lot_end_time": float(res.lot_end),
        "total_process_time": self.encoder.estimate_plan_total_process_time(
            steps, self.encoder.wafer_counts[lot],
        ),
        "qtime_risk": self.encoder.estimate_qtime_risk(lot, machine, ppid, steps),
    }
    return result, ""
```

> 字段 key 与旧版完全一致（`_candidate_features` 行 1920–1923 消费 `total_process_time`/`lot_end_time`/`steps`/`qtime_risk`，行 1901 注释列出全集），不破坏候选特征。
>
> **非破坏性确认:** 新 `_dry_run_candidate` 全程不 add 真实日历（`schedule_on_calendar` 非破坏），不再需要旧版的 `added_chamber`/`added_machine` 还原逻辑。这正是设计 §11.4 回滚不变量要测的点。

**验证:** `python run_phase1_environment_demo.py` + `python run_phase2_sas_inference_demo.py` 跑通；候选池构建无异常。

---

## Task 6：删旧函数 + 写回滚不变量测试

**做什么:**
1. 删 `rl_environment.py` 的 `_lot_sub_batches`（行 1789–1800）、`_select_earliest_stage_resource`（行 1820–1884）。`_allowed_resources_for`（行 1802–1818）、`_stage_process_sigma`（行 1886–1899）逻辑已下沉到 `lower_layer_scheduler.py`，删上层版本。
2. grep 确认无残留引用：`_lot_sub_batches`、`_select_earliest_stage_resource`、`_allowed_resources_for`、`_stage_process_sigma`、`_allowed_resources_cache` 在 `rl_environment.py` 及全包内无其他调用点（`__init__` 里的 `self._allowed_resources_cache = {}` 一并删）。
3. 写回滚/非破坏性测试。

**先 grep:**
```
grep -rn "_lot_sub_batches\|_select_earliest_stage_resource\|_allowed_resources_for\|_stage_process_sigma\|_allowed_resources_cache" .
```
确认仅剩定义处（待删）。若有其他引用，先消除。

**新建 `tests/test_decoupling_rollback.py`:**

```python
import copy

import numpy as np


def _calendar_snapshot(state):
    return (
        copy.deepcopy(state.machine_calendar),
        copy.deepcopy(state.chamber_calendar),
    )


def test_dry_run_is_non_destructive(small_env):
    """dry_run_action 后真实日历逐字节还原（设计 §11.4）。"""
    env = small_env
    before = _calendar_snapshot(env.state)
    machine = int(env.encoder.get_machine_list(1)[0])
    pool = env.build_candidate_pool(machine)
    # 取第一个有效真实动作
    idx = next(
        i for i, (a, m) in enumerate(zip(pool.actions, pool.action_mask))
        if bool(m) and not env._coerce_action(a).is_padding
        and not env._coerce_action(a).is_wait
    )
    dry = env.dry_run_action(env._coerce_action(pool.actions[idx]))
    after = _calendar_snapshot(env.state)
    assert dry.success
    assert before[0] == after[0]   # machine_calendar 不变
    assert before[1] == after[1]   # chamber_calendar 不变


def test_commit_then_rollback_restores(small_env):
    """commit 后 rollback_last_commit 还原（既有不变量，确认重构未破坏）。"""
    env = small_env
    before = _calendar_snapshot(env.state)
    machine = int(env.encoder.get_machine_list(1)[0])
    pool = env.build_candidate_pool(machine)
    idx = next(
        i for i, (a, m) in enumerate(zip(pool.actions, pool.action_mask))
        if bool(m) and not env._coerce_action(a).is_padding
        and not env._coerce_action(a).is_wait
    )
    env.commit_action_index(machine, idx, pool=pool)
    env.rollback_last_commit()
    after = _calendar_snapshot(env.state)
    assert before[0] == after[0]
    assert before[1] == after[1]
```

**验证:** `python -m pytest tests/ -q` 全过；`python run_phase1_environment_demo.py`、`python run_phase2_sas_inference_demo.py` 跑通。grep 确认旧函数名零引用。

> 注意 fixture/属性真实名：`get_machine_list`、`_coerce_action`、`rollback_last_commit`、`build_candidate_pool` 实现前先 grep 核对签名；测试里取第一个有效动作的 `next(...)` 若 small 实例首机台无真实动作，换 lot/machine 或遍历机台。

---

## Task 7：删死代码 `_prof.py`

**做什么:** 删 `FAB_RL/FABenv/_prof.py`（设计 §3.4：无 import 引用的临时 cProfile 脚本；它探测的 commit 热点解耦后已变）。

**先确认零引用:**
```
grep -rn "_prof\|import _prof\|from _prof" .
```
仅匹配文件自身则安全删除。

**删除:** `rm FAB_RL/FABenv/_prof.py`（PowerShell `Remove-Item`）。

**验证:** `python -m pytest tests/ -q` 仍全过；两个 demo 仍跑通。

---

## Task 8：回归对比（行为不崩溃 + 数值变化在预期方向）

**做什么:** 用 `evaluate_baselines.py` 对比解耦前后（设计 §11.3）。重构在分支上做，故对比 = 重构后跑一遍记录指标，确认：
1. 无可行性崩溃（无大量 `insertion_failed` / `infeasible`）。
2. Q-time / 利用率 / 违规数变化方向合理（接受偏移，但不应出现「全部 lot 无法调度」之类崩溃）。

**命令:**
```powershell
python evaluate_baselines.py --instance small --seeds 8
python evaluate_baselines.py --instance pressure --seeds 5
```

**判据:**
- `small`（4-lot）：仍 0 违规、所有规则可完成（CLAUDE.md 已知该实例不区分策略）。
- `pressure`（50-lot）：各规则均能完成全部 lot，无成片 `insertion_failed`；利用率与解耦前同量级（设计 §10 接受微变）。
- **核心成功标志（设计 §10 收益）**：抽查若干 commit 的实际 makespan 与 `qtime_safe_mask` 用 `estimate` 预判的 makespan 现已同口径（同一 `schedule_deterministic` 核心）。可在 `pressure` 跑一条 episode，打印某 lot 的 `dry_run.lot_end_time - lot_release_time` 与 `estimate(...).mu_finish`，确认接近（差异仅来自噪声/日历占用，不再是两套算法的系统性偏差）。

**验证:** 两条命令跑完无异常；记录指标到 PR/笔记。若 `pressure` 出现可行性崩溃，回到 Task 3 的「腔体空档回填差异」注释评估是否需补空档回填。

---

## 完成标准（总）

- [ ] `lower_layer_estimator.py` 有 `schedule_deterministic`，`estimate` step6 复用它，数值不变。
- [ ] `lower_layer_scheduler.py` 提供 `schedule_on_calendar` + `ScheduleResult`，与 `estimate` 共享核心。
- [ ] `rl_environment.py` 的 `_dry_run_candidate` / `_simulate_action` 为薄封装；`_lot_sub_batches` / `_select_earliest_stage_resource` / `_allowed_resources_for` / `_stage_process_sigma` / `_allowed_resources_cache` 已删，全包零残留引用。
- [ ] `_prof.py` 已删。
- [ ] `tests/` 下一致性断言（estimate.μ == schedule_on_calendar 空日历）+ 回滚不变量测试全过。
- [ ] 两个 demo 跑通；`evaluate_baselines.py` small/pressure 无可行性崩溃。
- [ ] qtime mask 预判 makespan 与 commit 实际 makespan 同口径（核心收益验证）。

## 实现者注意（贯穿全程）

- **每个 Task 前先 grep 核对真实符号名**（`build_small_test_encoder`、`ScheduleState` 构造、`get_machine_list`、`process_time_sigma` 是否存在等）。计划里的名字是依据当前阅读得出的，但实现前必须核对，不要照抄出错。
- **不改候选池流水线语义**（mask→filter→score 顺序、18 维特征 key、reward 通道）。
- **不 git commit**，除非用户要求。
- 遇到 small 实例 σ 情况影响一致性断言写法时，按 Task 2/3 注释里的「二选一」就近决定并写明依据。
