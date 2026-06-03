# 上下层解耦设计文档（报告 §1.5）

> 日期：2026-06-02
> 状态：已批准设计方向（方案 A + 五点），待落成实施计划
> 关联报告：`项目报告_完善版.md` §1.5（两层决策架构与下层启发式估时器）、§2.4.6（噪声）

## 1. 背景与目标

报告 §1.5 规定系统严格分两层、接口**单向且窄**：

- **上层（RL）**：决定「哪个 lot / 哪台 machine / 哪个 ppid / 什么时机」。只向下层传 `(lot, machine, ppid)` 与各阶段 `(μ_s, σ_s)`。
- **下层（固定规则估时/排程，非 RL）**：算工件内部「组批 + list scheduling + 蒙特卡洛」，向上层返回完成时间分布等少数输出。**下层不知道上层的 RL / 候选池 / 预留逻辑；上层不依赖下层的组批 / 流水 / 实例选择内部细节。**

当前代码违背了这个契约：下层逻辑被在上层**重写了一遍**，形成两套独立实现，口径不一致（正是 CLAUDE.md §1.5 记录的历史 bug 根源——qtime mask 看到的 makespan 与 commit 实际 makespan 对不上）。

**本次目标**：把下层逻辑收敛为唯一真相，上层只通过窄接口调用；删除上层重复实现；物理拆分模块。

## 2. 已批准的设计决策

1. **成功标准**：以下层为唯一真相，dry-run/commit 复用下层口径，**接受调度数值（lot/wafer 的 start/end、利用率、违规数）发生变化**；已训练 checkpoint 的数值可比性下降可接受。
2. **物理形态**：物理拆文件，窄接口分隔——下层排程/登记逻辑从 `rl_environment.py` 抽到独立模块。
3. **方案 A**：下层抽出单一 list-scheduling 核心，对外暴露两个薄接口（状态无关估时 + 状态相关日历排程），二者共享核心。否决方案 B（上层平移嵌入→解耦不彻底）与方案 C（estimate 读日历→破坏缓存、性能崩）。

## 3. 当前边界与问题

### 3.1 下层 `lower_layer_estimator.py`（已基本合规）

`estimate()` 已返回报告 §1.5 的完整交付物：`(μ_finish, σ_finish)` + `per_instance_occupancy` + `bottleneck_stage` + `stage_mu/stage_sigma/n_batches`。它**状态无关**（不读 `state`），因此可被 `_estimate_cache` 缓存（键 `(lot,machine,ppid,n_mc)`）。辅助核心：`compute_sub_batches`（组批）、`_run_list_schedule`（流水排程）、`monte_carlo_makespan`（蒙特卡洛）。

### 3.2 上层 `rl_environment.py`（边界泄漏）

dry-run / commit **完全没有调用 `estimate()`**，而是自己又实现了一套「组批 + 实例选择 + 时序排程 + 占用区间生成 + 机台/腔体两级槽位收敛」：

| 函数 | 行号 | 问题 |
|------|------|------|
| `_dry_run_candidate` | 1665 | 自带 20 次迭代收敛排程，不调用 `estimate()` |
| `_simulate_action` | 2161 | 与 `_dry_run_candidate` 平行，commit 时再排一遍（额外做噪声注入 + wafer 行生成） |
| `_lot_sub_batches` | 1789 | 重复下层 `compute_sub_batches` |
| `_select_earliest_stage_resource` | 1820 | 重复下层 list-scheduling 的实例选择 |
| `_allowed_resources_for` | 1802 | 仅服务于上面的重写排程（机台允许资源集合） |
| `_stage_process_sigma` | 1886 | 仅服务于 commit 噪声采样 |

### 3.3 根本张力

`estimate()` 为可缓存而**状态无关**（算「空机台 makespan 分布」，实例 free_time 初值=0）；dry-run/commit **状态相关**（必须在真实日历空档插入，考虑已提交占用，commit 还要注入噪声）。三者本质是同一件事——「给定子批 + 各阶段加工时间 + 各实例起始空闲时刻，跑贪心 list scheduling 得占用区间」——区别仅在 **free_time 初值**（0 vs 真实日历）、**加工时间**（MC 采样 vs μ vs μ+噪声）、**是否登记日历**。故应共享一个核心。

### 3.4 死代码核查结论

除解耦本身要删的重复函数外，仅 `_prof.py` 可删：

| 候选 | 引用情况 | 处置 |
|------|----------|------|
| `_prof.py` | 无任何 import 引用；临时 cProfile 脚本，探测的 commit 热点解耦后即改变 | **删除** |
| `problem.py`（`ProblemDefinitionMixin`） | 被 `problem_instances.py:11` 引用 | 保留 |
| `resource_calendar.py`（`CalendarDecoderMixin`） | 被 `problem_instances.py:12` 引用 | 保留 |
| `compute_sas_reward` / `compute_sas_reward_components` | 被 `sas_step` 单头标量路径 4 处调用 + `__init__.py` 导出 | 保留 |

## 4. 目标架构

```
上层  rl_environment.py (ResourceCalendarEnv)
        候选池 / qtime mask / is_doomed / sas_step / reward / 候选特征
        dry_run_action ─┐
        commit ─────────┤   仅通过窄接口调用下层
                        ↓
下层  lower_layer_estimator.py        lower_layer_scheduler.py（新建）
        estimate()  [状态无关·可缓存]   schedule_on_calendar()  [状态相关]
              └──────────┬───────────────────────┘
                  共享 list-scheduling 核心 _run_list_schedule
```

- **`lower_layer_estimator.py`**：保留状态无关估时（`estimate` / `monte_carlo_makespan` / `compute_sub_batches`），喂 `qtime_safe_mask`（n_mc=20）与 `is_doomed`（n_mc=10）。维持 `_estimate_cache` 行为不变。
- **`lower_layer_scheduler.py`（新建）**：状态相关的日历排程引擎，喂 dry-run/commit。
- **共享核心**：list-scheduling 纯函数（现 `_run_list_schedule` 的实例选择逻辑 `argmin max(ready,free)`）由两模块共用，保证估时与执行口径一致。落点（放 estimator 内导出，或独立 `_list_schedule.py`）由实施计划定。

## 5. 下层接口契约

### 5.1 保留：`estimate(...)`（状态无关）

签名与返回不变，继续可缓存。

### 5.2 新增：`schedule_on_calendar(...)`（状态相关）

```
schedule_on_calendar(lot, machine, ppid, encoder, calendar_state,
                     earliest_release, noise_rng=None) -> ScheduleResult
```

- 读 `calendar_state`（machine_calendar + chamber_calendar）各实例当前空闲时刻作为 free_time 初值。
- 跑共享 list-scheduling 核心；`noise_rng=None` 用 μ（dry-run / 规划），传入 rng 则按 stage σ 采样 μ+噪声（commit / 执行，报告 §2.4.6）。
- 含机台/腔体两级槽位收敛逻辑（从 `_dry_run_candidate`/`_simulate_action` 下沉）。
- **非破坏性**：内部为推进 free_time 可临时占用，返回前还原，**不改变传入的 `calendar_state`**——登记与否由上层决定。
- 返回 `ScheduleResult`（绝对时刻）：
  - `lot_start`, `lot_end`
  - `batch_intervals`: 每个 (子批, stage) 的 `(resource_key, start, end)`
  - `machine_interval`: `(machine, lot_start, lot_end)`
  - `subbatch_wafer_map`: 每个子批的 wafer-id 列表（供上层展开 wafer_schedule）
  - `infeasible_reason`: 失败时的原因串（如 `chamber_side_unavailable` / `calendar_no_stable_slot`），成功为空

具体字段类型与命名由实施计划细化；要点是**下层只产出区间与映射，不绑定上层的 numpy schema、不持久化日历**。

## 6. 上层改造

- `_dry_run_candidate` → 调 `schedule_on_calendar(noise_rng=None)` 取区间 → 校验所有区间可 add（无冲突）→ 不持久化 → 组装现有 `dry_run_info` 字段（`lot_release_time`/`lot_start_time`/`lot_end_time`/`total_process_time`/`qtime_risk`）。删内部排程循环。
- `_simulate_action` → 调 `schedule_on_calendar(noise_rng=self._noise_rng if process_noise_enabled else None)` 取区间 → add 持久化 + 更新 `machine_available_time`/`chamber_available_time` + 把 `subbatch_wafer_map` 展开成 (n,9) `wafer_schedule`、组装 (n,5) `lot_schedule`。删内部排程循环。
- 失败语义保持：dry-run 失败回 `(None, reason)`；commit 失败回滚已加区间并抛错（由 `sas_step` 现有逻辑处理为 `insertion_failed`）。

## 7. 噪声与 schema 归属

- **噪声**：来源在上层（环境持有 `_noise_rng` 与 `process_noise_enabled`，决定是否/用哪个种子），应用在下层（排程时按 stage σ 采样 delta）。`_stage_process_sigma` 的「从 encoder 读 σ」逻辑随之移入下层。符合 §2.4.6「环境每 step 采样实际加工时间」。
- **schema**：`lot_schedule (n,5)` / `wafer_schedule (n,9)` 的 numpy 组装**留在上层**（环境对外契约），下层只回结构化区间 + wafer 映射。

## 8. 函数处置清单

| 函数/文件 | 处置 | 去向 |
|-----------|------|------|
| `_dry_run_candidate` 排程循环 | 改写为薄封装 | 调下层 |
| `_simulate_action` 排程循环 | 改写为薄封装 | 调下层 |
| `_lot_sub_batches` | 删 | 用下层 `compute_sub_batches` |
| `_select_earliest_stage_resource` | 删 | 并入下层排程核心 |
| `_allowed_resources_for` | 移入下层 | `lower_layer_scheduler.py` |
| `_stage_process_sigma` | 移入下层 | `lower_layer_scheduler.py` |
| `_prof.py` | 删（死代码） | — |
| `estimate` / `_estimate_cache` / `qtime_safe_mask` / `is_doomed` / `build_candidate_pool` / `sas_step` / reward 全家 / `rollback_last_commit` / `validate_schedule` | 保留不动 | — |

## 9. 数据流对比

**解耦前**：`sas_step` → `dry_run_action`/`_dry_run_candidate`（上层自排程）；commit → `_simulate_action`（上层再排一遍）。`qtime_safe_mask` → `estimate`（另一套）。**两套口径**。

**解耦后**：`sas_step` → dry-run/commit → `schedule_on_calendar`；`qtime_safe_mask`/`is_doomed` → `estimate`。两接口**共享 list-scheduling 核心**，口径一致。

## 10. 风险与预期数值变化

- 实例选择规则从上层「`find_earliest_slot` 字典序」统一到下层「`argmin max(ready,free)`」，**lot/wafer 的 start/end 会偏移**，利用率/违规数可能微变（已接受）。
- 收益：qtime mask 预判的 makespan 与 commit 实际 makespan **首次一致**，根治 §1.5 历史 bug。
- 性能：dry-run/commit 仍是 CPU 瓶颈（CLAUDE.md），但下沉不引入新的全局复制；`schedule_on_calendar` 的非破坏性临时占用需注意 add/rollback 开销，不得退化为 O(全日历) 复制。
- `_estimate_cache` 行为不变（估时仍状态无关）。

## 11. 验证策略（无 pytest 套件）

测试套件已被移除（见 CLAUDE.md），本次用以下方式验证正确性：

1. **一致性断言（核心目标）**：对若干 `(lot,machine,ppid)`，比对 `estimate()` 的 `μ_finish` 与 `schedule_on_calendar(noise_rng=None)` 在空日历上的 `lot_end` 应吻合（同一核心，应一致）。
2. **冒烟**：`run_phase1_environment_demo.py`、`run_phase2_sas_inference_demo.py` 跑通无异常。
3. **行为对比**：`evaluate_baselines.py --instance pressure --seeds N` 解耦前后指标对比，确认变化在预期方向、无可行性崩溃（无大量 `insertion_failed`）。
4. **回滚不变量**：dry-run 后日历逐字节还原（`rollback_last_commit` / 非破坏性约束）。

> 实施计划可酌情为本次解耦新增针对性测试（不依赖已删的旧套件）。是否恢复 pytest 由用户在审阅时决定。

## 12. 不在本次范围

DDT / 前瞻预留 / set-/cross-attention（报告 §5、§6.2.2）、HGNN 状态表示、双注意力 SAS、PPO-Lagrangian 调参——均为后续 Phase，不在本次解耦内。本次只动「下层估时/排程的边界」，不改 RL 策略、奖励通道、候选池流水线语义。

## 13. 已定决策（审阅后固化）

- 共享 list-scheduling 核心放 `lower_layer_estimator.py` 内导出复用，**不**新建 `_list_schedule.py`。
- 为本次解耦逻辑**新增针对性 pytest**（至少覆盖 §11 第 1 点的下层估时/排程一致性断言、§11 第 4 点的回滚不变量），**不**恢复此前移除的旧测试套件。
