# 阶段1项目计划书：资源日历环境 + 候选池 + Action Mask

## 1. 项目背景与阶段定位

本计划书面向开发实施团队，依据《面向半导体 FAB 机台组调度问题的项目报告》第10章“分阶段实现路线”中的阶段1制定。阶段1的定位是后续 SAS-PPO、单注意力 SAS-PPO、双注意力 SAS-PPO 以及完整 DDA-PPO 的工程基础。

本阶段不引入强化学习训练、不实现 Actor-Critic、不设计 DDT 决策时间智能体，也不实现注意力网络。阶段1的核心任务是先把调度环境本身做稳定：在给定 Lot、Machine、PPID、Chamber / Side 等信息后，系统能够生成结构化候选动作，使用资源日历进行 dry-run 可插入性检查，在 commit 后生成一致的 Lot-level 和 Wafer-level 排程结果，并通过 action mask 屏蔽结构性不可行动作。

阶段1完成后，应形成一个可重复、可验证、无资源冲突的基础调度环境。该环境既可以独立用于规则调度与测试，也可以作为阶段2 SAS-PPO 的环境接口。

## 2. 阶段1建设目标

阶段1建设目标可以概括为：

> 建立一个以资源日历为核心、以固定长度候选池控制动作规模、以 action mask 过滤结构性不可行动作、以 dry-run / commit / rollback 保证排程一致性的基础调度环境。

具体目标包括：

1. 建立 Machine 层和 Chamber / Side 层两级资源日历。
2. 支持候选 `(Lot, Machine, PPID)` 动作的可插入性检查。
3. 支持 dry-run，保证试探性检查不污染正式资源日历。
4. 支持 commit，将成功动作写入 Machine 日历、Chamber / Side 日历、Lot-level schedule 和 Wafer-level schedule。
5. 支持 rollback，在插入失败或中途异常时恢复动作前状态。
6. 建立固定长度候选池机制，包括候选生成、Top-K、padding 和真实动作映射。
7. 建立 action mask，过滤未到达 Lot、不可加工 Machine、不可用 PPID、PPID stage 缺失、资源结构不满足和 padding 动作。
8. 建立排程结果一致性校验机制，确保 Lot-level 与 Wafer-level 结果完整、无冲突、可复现。
9. 为阶段2 SAS-PPO 预留候选池特征、mask、动作索引、真实动作映射和执行反馈信息。

## 3. 范围边界

### 3.1 纳入范围

阶段1纳入以下内容：

- 单个 Machine Group 内部的调度环境。
- Lot、Machine、PPID、Chamber / Side、wafer、stage 等对象的工程表示。
- Machine 层资源日历。
- Chamber / Side 层资源日历。
- 候选动作 `(Lot, Machine, PPID)` 的生成。
- Top-K 候选池截断。
- padding 到固定候选池长度。
- action mask 生成。
- dry-run 可插入性检查。
- commit 正式插入。
- rollback 状态恢复。
- Lot-level schedule 输出。
- Wafer-level schedule 输出。
- 排程完整性、资源不重叠和 mask 正确性验证。

### 3.2 不纳入范围

阶段1不纳入以下内容：

- PPO、SAS-PPO、DDA-PPO 或其他强化学习训练。
- Actor-Critic 网络。
- DDT 决策时间智能体。
- 单注意力或双注意力模型。
- 神经网络候选动作评分。
- 鲁棒调度、CVaR 优化或多场景不确定性建模。
- 跨 Machine Group 或全厂级调度。
- 设备维护、停机、故障、PM 和机况退化。
- setup time、recipe 切换时间和 chamber 清洗时间。
- operator / 人员约束。
- 批处理、合批、拆批和 lot merge / split。

### 3.3 Q-time、due date 与 priority 的处理边界

阶段1需要保留 Q-time、due date 和 priority 等字段，以便候选评分、日志分析和后续 reward 设计使用。但在阶段1中：

- Q-time 默认不作为统一 hard mask，除非输入规则明确指定某类 Q-time 违背必须禁止。
- due date 不作为 hard mask，只用于候选排序、日志分析和后续评价。
- priority 不作为 hard mask，只用于候选排序、日志分析和后续评价。

action mask 主要负责结构性不可行动作过滤，而不是把所有目标偏好都变成硬约束。

## 4. 总体技术架构

阶段1采用工程闭环型架构：

```text
输入实例
  ↓
初始化调度状态与资源日历
  ↓
选择当前决策 Machine
  ↓
生成候选动作集合
  ↓
Top-K / padding 得到固定长度候选池
  ↓
生成 action mask
  ↓
dry-run 检查资源日历可插入性
  ↓
commit 成功动作，或 rollback 失败动作
  ↓
输出 Lot-level / Wafer-level schedule
  ↓
验证资源冲突、排程完整性和 mask 正确性
```

该架构的核心原则是：资源日历负责可执行性，候选池负责控制动作规模，mask 负责结构性过滤，dry-run / commit / rollback 负责状态一致性，验证模块负责证明排程结果可信。

## 5. 模块划分与职责

### 5.1 输入数据定义模块

职责：定义阶段1所需的最小输入对象和字段。

核心输入包括：

- Lot 信息：`lot_id`、`arrival_time`、`due_date`、`priority`、`wafer_count`、`recipe`。
- Machine 信息：`machine_id`、所属 Machine Group、可加工 Recipe 集合、可用 PPID 集合。
- PPID 信息：`ppid_id`、适用 Machine、适用 Recipe、stage 列表。
- Stage 信息：`stage_id`、候选 Chamber / Side、process time、顺序约束。
- Chamber / Side 信息：`machine_id`、`chamber_id`、`side_id`、可用性。
- Q-time 信息：可选的 stage 间等待上限或风险配置。

输出：标准化后的 problem instance，供调度状态初始化和候选动作生成使用。

### 5.2 调度状态模块

职责：维护当前调度过程中的动态状态。

状态至少包括：

- 当前时刻 `t_now`。
- 未完成 Lot 集合。
- 已完成 Lot 集合。
- 各 Machine 的下一可用时间。
- Machine 资源日历。
- Chamber / Side 资源日历。
- 已提交的 Lot-level schedule。
- 已提交的 Wafer-level schedule。
- 当前 planning window 信息。

输出：可被候选生成、dry-run 和验证模块读取的统一状态对象。

### 5.3 资源日历模块

职责：维护两层资源占用信息，并提供插入检查能力。

两层日历包括：

1. Machine 日历：记录每台 Machine 的 Lot-level 占用区间。
2. Chamber / Side 日历：记录每个 `(machine, chamber, side)` 的 wafer stage 占用区间。

核心能力包括：

- 查询资源在时间区间内是否空闲。
- 查找某个动作可插入的最早开始时间。
- 检查 Machine 区间是否与既有 Lot 占用重叠。
- 检查 Chamber / Side 区间是否与既有 wafer stage 占用重叠。
- 计算候选动作对应的 Lot-level 开始/结束时间。
- 计算候选动作对应的 Wafer-level stage 开始/结束时间。

输出：插入计划或不可插入原因。

### 5.4 候选动作生成模块

职责：在当前决策 Machine 下生成局部候选动作。

候选动作定义为：

```text
a = (lot, machine, ppid)
```

生成规则：

1. Lot 已经到达，即 `arrival_time <= t_now`。
2. Lot 尚未完成。
3. Machine 支持该 Lot 的 Recipe。
4. `(lot, machine)` 存在可用 PPID。
5. PPID 的 stage 定义存在且完整。
6. PPID 所需 Chamber / Side 在当前 Machine 上存在。

输出：结构化候选动作列表，每个候选包含真实动作字段、基础特征和初始可行性原因。

### 5.5 Top-K 与 padding 模块

职责：把可变长度候选集合转换为固定长度候选池。

处理规则：

```text
如果候选数 > K_action：
    根据启发式分数保留 Top-K

如果候选数 < K_action：
    使用 padding action 补齐

如果候选数 = K_action：
    原样保留
```

阶段1的 Top-K 可使用启发式评分，不引入神经网络。启发式分数可以考虑：

- due date 紧迫度。
- Q-time 风险。
- priority。
- waiting time。
- estimated process time。
- resource conflict risk。

输出：固定长度候选池 `A_fixed^m`，长度为 `K_action`。

### 5.6 Action Mask 模块

职责：为固定长度候选池生成 mask。

mask 取值：

```text
mask_i = 1：候选动作结构上可供策略选择
mask_i = 0：候选动作不可选择
```

mask=0 的典型原因包括：

- padding action。
- Lot 未到达。
- Lot 已完成。
- Machine 不支持该 Lot 的 Recipe。
- `(lot, machine)` 无可用 PPID。
- PPID stage 缺失或不完整。
- 所需 Chamber / Side 在当前 Machine 上不存在。
- dry-run 已确认该动作无法插入资源日历。

输出：与候选池等长的 `candidate_mask`，以及每个 mask=0 动作的原因。

### 5.7 Dry-run 可插入性检查模块

职责：在不修改正式状态的情况下检查候选动作是否可以插入。

dry-run 应完成：

1. 复制或隔离当前资源日历状态。
2. 尝试计算候选动作的 Lot-level 区间。
3. 尝试计算所有 wafer stage 的 Chamber / Side 区间。
4. 检查 Machine 日历冲突。
5. 检查 Chamber / Side 日历冲突。
6. 返回可插入计划或失败原因。

关键要求：dry-run 不得污染正式资源日历、Lot-level schedule 或 Wafer-level schedule。

输出：`DryRunResult`，包含是否成功、拟插入区间、失败原因和后续 commit 所需信息。

### 5.8 Commit / Rollback 模块

职责：将成功动作正式写入状态，或在失败时恢复动作前状态。

commit 应完成：

- 写入 Machine 日历。
- 写入 Chamber / Side 日历。
- 追加 Lot-level schedule。
- 追加 Wafer-level schedule。
- 更新 Lot 完成状态。
- 更新 Machine 下一可用时间。
- 更新当前状态摘要。

rollback 应完成：

- 移除本次动作写入的 Machine 区间。
- 移除本次动作写入的 Chamber / Side 区间。
- 移除本次动作写入的 Lot-level schedule 行。
- 移除本次动作写入的 Wafer-level schedule 行。
- 恢复 Lot 状态、Machine 状态和当前状态摘要。

阶段1建议采用“插入日志”方式支持 rollback：每次 commit 记录本次新增的所有区间和 schedule 行。如果 commit 中途失败，可根据插入日志撤销，而不是依赖全局重建。

输出：`CommitResult` 或 `RollbackResult`。

### 5.9 输出与验证模块

职责：输出排程结果，并验证结果是否满足阶段1硬约束。

输出包括：

Lot-level schedule：

```text
[lot, machine, ppid, start_time, end_time]
```

Wafer-level schedule：

```text
[lot, wafer_id, machine, ppid, stage_id, chamber, side, start_time, end_time]
```

验证内容包括：

- 同一 Machine 上 Lot-level 区间不重叠。
- 同一 `(machine, chamber, side)` 上 wafer stage 区间不重叠。
- 每个已完成 Lot 都有一条 Lot-level schedule。
- 每个已完成 Lot 的所有 wafer 都完成 PPID 定义的全部 stage。
- Wafer-level stage 顺序符合 PPID stage 顺序。
- Lot-level 开始/结束时间覆盖对应 wafer-level 最早开始和最晚结束。
- action mask 与候选动作结构性可行性一致。
- dry-run 失败不会改变正式状态。
- rollback 后状态与动作前状态一致。

## 6. 关键接口草案

本节给出阶段1计划层面的接口约定，用于指导后续实现；具体函数名和类名可在编码阶段按项目规范调整。

### 6.1 构建候选池

```text
build_candidate_pool(state, machine_id, K_action) -> CandidatePool
```

输入：当前调度状态、当前决策 Machine、候选池长度。  
输出：固定长度候选池，包括候选动作、候选特征、mask 和 mask reason。

### 6.2 生成原始候选动作

```text
generate_raw_candidates(state, machine_id) -> list[DispatchAction]
```

输入：当前状态和当前 Machine。  
输出：未经过 Top-K 和 padding 的真实候选动作列表。

### 6.3 候选动作排序

```text
score_candidate(action, state) -> float
```

输入：候选动作和当前状态。  
输出：启发式候选分数，用于 Top-K。

### 6.4 生成 action mask

```text
build_action_mask(candidate_pool, state) -> MaskResult
```

输入：固定长度候选池和当前状态。  
输出：mask 数组和每个无效候选的原因。

### 6.5 Dry-run 检查

```text
dry_run(action, state) -> DryRunResult
```

输入：候选动作和当前状态。  
输出：可插入计划或失败原因。正式状态不得被修改。

### 6.6 Commit 动作

```text
commit(action, dry_run_result, state) -> CommitResult
```

输入：候选动作、dry-run 结果和当前状态。  
输出：更新后的状态、插入日志和执行信息。

### 6.7 Rollback 动作

```text
rollback(commit_log, state) -> RollbackResult
```

输入：本次 commit 的插入日志和当前状态。  
输出：恢复后的状态。

### 6.8 验证排程

```text
validate_schedule(state) -> ValidationReport
```

输入：当前调度状态。  
输出：验证是否通过、错误列表、冲突区间、缺失 wafer stage 和 mask 异常。

## 7. 数据流与状态一致性要求

### 7.1 正常成功路径

```text
1. 从 state 中读取当前 Machine 和当前时刻。
2. 生成 raw candidates。
3. 使用 Top-K / padding 得到固定长度候选池。
4. 生成 action mask。
5. 选择一个 mask=1 的候选动作。
6. 对该动作执行 dry-run。
7. dry-run 成功后执行 commit。
8. 更新资源日历、Lot schedule、Wafer schedule 和状态摘要。
9. 执行 validate_schedule。
```

### 7.2 Dry-run 失败路径

```text
1. 选择候选动作。
2. 执行 dry-run。
3. dry-run 返回失败原因。
4. 不修改正式资源日历。
5. 不追加 schedule 行。
6. 记录失败原因。
7. 重新生成 mask 或进入下一决策。
```

### 7.3 Commit 中途失败路径

```text
1. dry-run 成功。
2. commit 开始写入资源日历。
3. 中途发现不可恢复异常。
4. 根据 commit_log 执行 rollback。
5. 恢复动作前状态。
6. 返回 commit_failed 和 rollback_success 标记。
```

### 7.4 候选池为空路径

```text
1. 当前 Machine 无可用候选。
2. 生成全 padding 候选池。
3. mask 全部为 0。
4. 阶段1环境返回 no_action_available。
5. 上层调度逻辑决定推进时间或选择其他 Machine。
```

## 8. 交付物清单

阶段1至少交付以下内容：

1. 阶段1输入数据字段说明。
2. 调度状态对象设计说明。
3. Machine 与 Chamber / Side 资源日历实现。
4. 候选动作生成逻辑。
5. Top-K、padding 和 action mask 实现。
6. dry-run 可插入性检查实现。
7. commit / rollback 实现。
8. Lot-level schedule 输出。
9. Wafer-level schedule 输出。
10. 排程验证工具。
11. 最小可运行 demo。
12. 单元测试与端到端测试。
13. 阶段1验收报告或测试结果摘要。
14. 面向阶段2 SAS-PPO 的环境接口说明。

## 9. 里程碑计划

### M1：输入对象与状态结构定义

目标：完成阶段1最小数据模型和调度状态定义。

主要任务：

- 定义 Lot、Machine、PPID、Stage、Chamber / Side 的必要字段。
- 定义调度状态结构。
- 定义 Lot-level 和 Wafer-level schedule 输出格式。
- 定义基础校验规则，例如字段缺失、PPID stage 缺失、Machine 不支持 Recipe。

验收标准：

- 能加载一个小型 problem instance。
- 能初始化空资源日历。
- 能输出初始状态摘要。
- 能识别明显非法输入。

### M2：资源日历与插入检查

目标：完成 Machine 与 Chamber / Side 两层资源日历。

主要任务：

- 实现 Machine 区间插入与重叠检查。
- 实现 Chamber / Side 区间插入与重叠检查。
- 实现 earliest slot 搜索。
- 实现 dry-run 可插入性检查。
- 实现 commit 和 rollback。

验收标准：

- 同一 Machine 上不会产生 Lot-level 区间重叠。
- 同一 Chamber / Side 上不会产生 wafer stage 区间重叠。
- dry-run 不改变正式状态。
- commit 后日历和 schedule 一致。
- rollback 后状态恢复到动作前。

### M3：候选池、Top-K、padding 与 mask

目标：完成固定长度候选池机制。

主要任务：

- 实现 raw candidate 生成。
- 实现启发式 candidate scoring。
- 实现 Top-K 截断。
- 实现 padding。
- 实现 action mask 与 mask reason。
- 将 dry-run 结果纳入 mask 或候选可行性信息。

验收标准：

- 候选池长度恒等于 `K_action`。
- padding action 必须 mask=0。
- 未到达 Lot 必须 mask=0。
- 不可加工 Machine 必须 mask=0。
- 不可用 PPID 必须 mask=0。
- PPID stage 缺失必须 mask=0。
- 资源日历无法插入的候选可被识别并说明原因。

### M4：端到端 demo、测试与验收

目标：形成阶段1完整闭环。

主要任务：

- 构建 small problem instance。
- 从初始化状态开始运行完整调度流程。
- 输出 Lot-level schedule 和 Wafer-level schedule。
- 执行资源冲突验证。
- 执行候选池与 mask 验证。
- 输出阶段1测试结果摘要。

验收标准：

- demo 可重复运行。
- 所有已完成 Lot 的 Lot-level / Wafer-level schedule 完整。
- Machine 无重叠。
- Chamber / Side 无重叠。
- mask 与候选可行性一致。
- dry-run、commit、rollback 的状态一致性测试通过。
- 输出可供阶段2 SAS-PPO 使用的 candidate pool、mask、真实动作映射和执行信息。

## 10. 测试方案

### 10.1 单元测试

建议覆盖以下测试：

1. Machine 日历区间插入成功。
2. Machine 日历区间重叠被拒绝。
3. Chamber / Side 日历区间插入成功。
4. Chamber / Side 日历区间重叠被拒绝。
5. dry-run 成功不修改正式状态。
6. dry-run 失败不修改正式状态。
7. commit 成功后日历和 schedule 更新。
8. rollback 后日历和 schedule 恢复。
9. 候选池长度固定为 `K_action`。
10. padding action mask=0。
11. 未到达 Lot mask=0。
12. Machine 不支持 Recipe mask=0。
13. PPID 不可用 mask=0。
14. PPID stage 缺失 mask=0。
15. 候选池为空时返回全 padding 和全 0 mask。

### 10.2 集成测试

建议覆盖以下场景：

1. 单 Lot、单 Machine、单 PPID、单 Chamber / Side。
2. 多 Lot、单 Machine、单 PPID，验证 Lot-level 顺序不重叠。
3. 单 Lot、多 wafer、多 stage，验证 Wafer-level stage 顺序。
4. 多 Machine、多 PPID，验证候选动作生成。
5. 多 Chamber / Side，验证底层资源冲突处理。
6. 候选数超过 `K_action`，验证 Top-K。
7. 候选数少于 `K_action`，验证 padding。
8. dry-run 成功但 commit 中途失败，验证 rollback。

### 10.3 端到端验收测试

端到端测试应从 problem instance 开始，完整运行：

```text
初始化状态
  → 选择 Machine
  → 构建候选池
  → 生成 mask
  → dry-run
  → commit / rollback
  → 输出 schedule
  → validate_schedule
```

最终验收结果应报告：

- 总 Lot 数。
- 已完成 Lot 数。
- Lot-level schedule 行数。
- Wafer-level schedule 行数。
- Machine 冲突数。
- Chamber / Side 冲突数。
- mask 异常数。
- dry-run 状态污染数。
- rollback 失败数。

## 11. 阶段1验收标准

阶段1完成必须满足以下条件：

1. 能在至少一个 small problem instance 上完成端到端调度。
2. 输出 Lot-level schedule 和 Wafer-level schedule。
3. 所有 Machine 日历区间无重叠。
4. 所有 Chamber / Side 日历区间无重叠。
5. 每个完成 Lot 的 wafer stage 数量与 PPID 定义一致。
6. 每个 Lot-level 区间覆盖对应 Wafer-level 区间。
7. 候选池长度固定为 `K_action`。
8. padding action 全部 mask=0。
9. 结构性不可行动作全部 mask=0，并有明确原因。
10. dry-run 不污染正式状态。
11. commit 成功后状态一致。
12. rollback 成功后状态恢复。
13. 候选池、mask、真实动作映射和执行反馈可被阶段2 SAS-PPO 直接读取。

## 12. 风险与应对

### 风险1：dry-run 污染正式资源日历

影响：后续候选判断和排程输出不可信。  
应对：dry-run 必须使用状态副本、事务日志或纯计算结果返回；测试中比较 dry-run 前后状态哈希或快照。

### 风险2：commit 与 dry-run 结果不一致

影响：dry-run 认为可行，但 commit 后出现冲突。  
应对：commit 应尽量复用 dry-run 生成的插入计划，不重新计算另一套结果；commit 后立即运行局部验证。

### 风险3：rollback 无法完全恢复状态

影响：一次失败动作会污染整个 episode。  
应对：commit 期间记录插入日志，包含所有新增区间和 schedule 行；rollback 测试必须覆盖中途失败场景。

### 风险4：mask 过严导致候选池长期为空

影响：后续 SAS-PPO 没有可学习动作。  
应对：阶段1区分结构性 hard mask 与软目标风险；Q-time、due date、priority 默认不作为 hard mask。

### 风险5：mask 过松导致大量不可插入动作

影响：阶段2训练时失败惩罚过多，学习效率下降。  
应对：阶段1可将 dry-run 失败原因纳入 mask reason 或候选可行性信息；保留失败统计用于调参。

### 风险6：Top-K 过早丢弃潜在优质动作

影响：后续策略只能在受限候选中学习。  
应对：Top-K 评分保持简单透明；在测试中记录被截断候选数量和分数分布；保留调大 `K_action` 的配置。

### 风险7：Lot-level 与 Wafer-level schedule 不一致

影响：输出无法对应 PPT 中两层决策变量。  
应对：validate_schedule 必须检查 Lot-level 区间与 Wafer-level 最早开始、最晚结束之间的覆盖关系。

## 13. 对阶段2 SAS-PPO 的接口预留

阶段1完成后，应向阶段2提供以下信息：

```text
SAS_input = {
  current_time,
  current_machine,
  candidate_actions,
  candidate_features,
  candidate_mask,
  action_index_to_real_action,
  global_state_features,
  calendar_summary_features
}
```

执行动作后，应返回：

```text
SAS_step_result = {
  selected_action_index,
  selected_lot,
  selected_machine,
  selected_ppid,
  insertion_success,
  insertion_failed,
  mask_invalid,
  wait_or_noop,
  selected_lot_start,
  selected_lot_end,
  selected_lot_process_time,
  new_qtime_violation,
  priority_rank_penalty,
  failure_reason,
  updated_state
}
```

这些字段能够支持阶段2中的 SAS reward、transition 记录和 PPO 更新。阶段1不计算 PPO loss，但必须保证环境接口具备记录 `(obs_t, action_index, mask, reward_info, obs_{t+1}, done)` 的能力。

## 14. 最终交付结论

阶段1不是强化学习算法实现阶段，而是智能调度环境的地基。只有当资源日历、候选池、mask、dry-run、commit、rollback 和验证机制全部稳定后，后续 SAS-PPO 才能在可信环境中训练。

因此，阶段1的完成标志不是模型指标提升，而是：

> 任意候选动作都能被明确判定为可插入或不可插入；任意成功 commit 都能生成一致的 Lot-level 与 Wafer-level schedule；任意失败插入都不会污染正式状态；任意固定长度候选池都能给出正确 mask 和真实动作映射。

达到上述标准后，项目即可进入阶段2“规则触发的 SAS-PPO”。
