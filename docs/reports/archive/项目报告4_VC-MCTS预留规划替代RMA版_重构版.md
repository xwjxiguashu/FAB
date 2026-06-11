# 面向半导体 FAB 机台组调度的 SAS + VC-MCTS 在线预留项目报告（代码重构版）

> 本报告按 2026-06-08 本地代码状态重构，参考根目录旧版 `项目报告4_VC-MCTS预留规划替代RMA版(2).md`、`CLAUDE.md`、`FAB_RL/FABenv/docs/superpowers/plans/` 下的阶段计划以及当前 `FAB_RL/FABenv/` 代码。凡叙述与旧报告不一致处，以当前本地代码为准。

---

## 0. 当前结论

当前项目已经形成一条可运行的两层调度原型：

1. 下层排程器负责给定 `(lot, machine, ppid)` 后的批处理、阶段资源选择和资源日历插入，是所有完成时间估计、dry-run、commit 的单一事实来源。
2. 上层 SAS 负责在当前可派工候选池中选择 `(lot, ppid)`，已经具备单头 PPO 与多头 PPO 两条训练路径。
3. Q-time、priority、utilization 不再被简单混成一个奖励权重，而是在候选池、mask、向量奖励、多头 critic 与 VC-MCTS 目标中分层处理。
4. 旧方案中拟学习的 RMA 预留代理已被 VC-MCTS 在线搜索替代。VC-MCTS 将 `reserve(machine, future_lot)` 作为搜索边，使用 reservation ledger 与真实环境 clone 做反事实 rollout。
5. SAS policy 已通过 `DispatchDelegate` 接入 VC-MCTS，可作为 rollout 派工基策略，也可作为 PUCT 先验和叶子 critic 估值的可选 AlphaZero 风格增强。
6. `late_hi` 压力实例、baseline 多 seed 评估、oracle reservation probe、VC-MCTS trace summary 与若干消融脚本已经落地。

因此，本阶段的项目重点已经从“能否表达预留决策”推进到“如何在更大 seed、更强噪声、更多实例族中验证 VC-MCTS + SAS 的稳定优势”。

---

## 1. 问题定义与工程边界

### 1.1 调度对象

本项目研究的是半导体 FAB 机台组调度问题。输入包括：

- Lot：晶圆批次，具有到达时间、晶圆数量、priority、due date、可加工 PPID 等属性；
- Machine：机台组中的具体机台，不同机台可支持不同 recipe/PPID；
- PPID / process steps：工艺路线，由多个 stage 组成；
- Chamber / Side：每台 machine 内部的细粒度资源；
- Q-time limits：stage 间材料等待窗口，当前主要体现在 `(stage_i, stage_j)` 的链式约束中。

上层决策不是直接排每片 wafer 的完整路径，而是在事件推进过程中为当前 idle machine 选择一个可执行的 `(lot, ppid)`，并由下层排程器确定 lot 内部 sub-batch、stage、chamber/side 的具体时间区间。

### 1.2 当前代码边界

当前活动代码位于 `FAB_RL/FABenv/`。根目录 `core.py`、`__init__.py` 以及 `MAMHSA_for_fjsp-master/` 属于参考或历史代码，不是当前 RL 环境主线。

当前报告不再把以下旧设想写成已实现功能：

- 不再声称存在学习式 RMA 预留代理；
- 不再声称 DDT hold agent 已完成训练闭环；
- 不再声称已经使用异构图神经网络或双注意力 actor；
- 不再声称已对公开 SMT2020 等基准完成正式对标；
- 不再把 reward shaping 写成解决 Q-time 和 priority 的主要机制。

当前可据代码确认的主线是：`ResourceCalendarEnv` + lower-layer scheduler/estimator + Phase2 SAS PPO + reservation ledger + VC-MCTS planner。

---

## 2. 总体架构

当前系统采用上下两层架构：

```text
Phase1CalendarProblem
  -> ResourceCalendarEnv
      -> build_candidate_pool()
      -> dry_run_action()
      -> commit_action_index()
      -> sas_step()
  -> Phase2EpisodeDriver
      -> rule / PPO / multihead PPO episode
  -> reservation_simulator + ReservationLedger
      -> oracle reservation / reservation-aware rollout
  -> VCMCTSPlanner
      -> no_op / delegate_dispatch / reserve
```

### 2.1 下层：固定规则排程

下层不学习策略，只负责在给定派工动作后进行确定性或带噪声的工艺内部排程。核心代码为：

- `lower_layer_estimator.py`
  - `compute_sub_batches`
  - `schedule_deterministic`
  - `estimate`
  - `qtime_violation_probability`
- `lower_layer_scheduler.py`
  - `ScheduleResult`
  - `schedule_on_calendar`

`schedule_deterministic` 是共享核心。`estimate()` 在空 free-time 条件下调用它，得到可缓存的完成时间分布；`schedule_on_calendar()` 在当前 committed calendar 上调用它，得到真实 state-dependent 时间区间。

### 2.2 上层：SAS 派工与 PPO 训练

上层 SAS 的职责是选择当前机台候选池中的动作。核心代码为：

- `rl_environment.py`：候选池、mask、dry-run、commit、奖励；
- `phase2_sas_observation.py`：将候选池和环境状态编码为 policy 输入；
- `phase2_sas_policy.py`：单头 actor-critic 与多头 actor-critic；
- `phase2_ppo_buffer.py`：单头和多头 rollout buffer；
- `phase2_ppo_trainer.py`：PPO 与 multi-head PPO；
- `phase2_sas_driver.py`：完整 episode 驱动、规则 baseline、policy rollout 与 greedy inference。

### 2.3 在线预留：VC-MCTS

VC-MCTS 负责“是否把某台 idle machine 留给未来可见 lot”的在线搜索，不再训练一个单独的预留网络。核心代码为：

- `reservation_ledger.py`：预留记录、TTL、consume/release；
- `reservation_rop.py`：Reservation Opportunity Point 候选生成；
- `reservation_simulator.py`：ledger-aware episode、clone、oracle reservation；
- `dispatch_delegate.py`：rule/SAS 派工委托；
- `vc_mcts_planner.py`：root-level VC-MCTS；
- `vc_mcts_probe.py`：baseline/oracle/VC-MCTS 对比入口；
- `vc_mcts_trace_summary.py`：JSONL trace 汇总。

---

## 3. 下层排程器：单一事实来源

旧版环境中曾经存在多处重复排程逻辑，容易造成估计路径和提交路径不一致。当前代码已经将下层逻辑集中到共享核心。

### 3.1 子批处理

`compute_sub_batches(n_wafers, side_capacity)` 将一个 lot 的晶圆拆成若干 sub-batch。若设置了 chamber side capacity，则同一 sub-batch 内 wafer 共享同一个 stage 的 chamber/side 时间区间，符合批处理“同进同出”的建模方式。

代码仍保留两类输出 schema：

```text
lot_schedule:    [lot, machine, ppid, start_time, end_time]
wafer_schedule:  [lot, wafer_id, machine, ppid, stage_id, chamber, side, start_time, end_time]
```

因为 wafer rows 会展开到每片 wafer，重建 calendar 时必须对相同 `(resource, start, end)` 去重。当前 `ResourceCalendarEnv.validate_schedule` 和 encoder 的 final validation 已处理这一点。

### 3.2 估计与日历排程解耦

`estimate()` 与 `schedule_on_calendar()` 的区别不是算法不同，而是 free-time 来源不同：

- `estimate()` 使用空 free-time，得到 state-independent 的 base makespan distribution，适合 qtime mask 和候选池评分缓存；
- `schedule_on_calendar()` 读取当前 committed calendar，返回具体的 `ScheduleResult`，适合 dry-run 和 commit。

这种设计使 Q-time 预测、候选评分、dry-run、commit 都服从同一个 stage list-scheduling 规则，避免“mask 认为安全但 commit 路径实际使用另一套算法”的偏差。

### 3.3 噪声语义

加工时间噪声由 `lower_layer_scheduler._stage_times` 和环境 `_simulate_action` 管理。当前语义是：

- dry-run 默认走均值路径，不推进 commit noise RNG；
- commit 在 `process_noise_enabled=True` 时才使用共享噪声 RNG；
- chain-joint Q-time mask 可以使用显式 RNG 做多次 noisy dry-run；
- commit 不会复用带均值的 dry-run 结果覆盖真实噪声语义。

该边界由 `tests/test_qtime_chain_mask_rng.py` 覆盖，避免候选池构建次数改变后续真实执行噪声。

---

## 4. 环境、候选池与约束顺序

### 4.1 `ResourceCalendarEnv`

`ResourceCalendarEnv` 是当前调度环境核心。它维护：

- 当前时间；
- machine/chamber calendar；
- completed lots；
- commit log 与 rollback；
- lower-layer estimate cache；
- 候选池构建参数；
- Q-time mask 模式；
- process noise 配置。

关键接口包括：

- `build_candidate_pool(machine)`
- `dry_run_action(action_index)`
- `commit_action_index(action_index)`
- `sas_step(action_index)`
- `rollback_last_commit()`
- `validate_schedule()`

### 4.2 候选池流程

候选池生成遵循固定顺序：

```text
结构可行动作
  -> Q-time safe mask
  -> priority filter / ranking
  -> CandidateScore 排序
  -> TopK
  -> padding + mask
```

这一顺序体现项目当前的字典序建模：

1. Q-time 是硬约束，优先级最高；
2. priority 是强偏好，在 Q-time safe 集合内发挥作用；
3. utilization 是软目标，只在前两层之后参与权衡。

候选特征固定为 18 维，包含 real/wait 标记、score、arrival、waiting、machine slot、load、process time、predicted completion、qtime risk、wafer count、priority、due slack 及 rank 类特征。actor 通过 mask 将 padding 动作置为无效。

### 4.3 Q-time mask 模式

当前 `ResourceCalendarEnv.qtime_mask_mode` 支持三类模式：

- `aggregate`：使用聚合 deadline proxy 与完成时间均值/方差判断；
- `chain`：dry-run 后按真实 `q_time_limits` 链检查 stage 间等待；
- `chain_joint`：多次 noisy dry-run 估计任意链窗口违约概率，作为联合机会约束。

`chain` 与 `chain_joint` 是当前较符合代码逻辑的 Q-time 路径，因为 pressure / late_hi 实例已经显式定义 `(1,2)`、`(2,3)` 等 stage 间约束。

---

## 5. 奖励、向量目标与多头 PPO

### 5.1 标量奖励

`compute_sas_reward_components` 与 `compute_sas_reward` 保留了标量 PPO 路径。当前即时奖励主要服务于动作合法性和训练稳定性。SAS 不承担全局 wait 决策，因此在 SAS 设置中 `wait_penalty` 应保持为 0。

### 5.2 向量奖励

多头路径使用 `RewardVectorConfig` 和 `compute_sas_reward_vector`。当前 reward vector 通道固定为：

```text
exec
qtime
util
```

其中：

- `exec`：即时执行合法性和稠密反馈；
- `qtime`：终局 Q-time residual / violation cost；
- `util`：终局资源利用相关软目标。

旧版曾出现的 `progress` critic head 已不再作为多头 critic 的通道。`RewardConfig.progress_weight` 仍可参与标量 reward shaping，但不构成多头 critic 的独立目标。

### 5.3 多头 actor-critic

`Phase2SASMultiHeadActorCritic` 共享 candidate encoder 和 actor head，但 critic 返回 `{exec, qtime, util}` 三个 value。`MultiHeadRolloutBuffer` 存储向量 reward，`MultiHeadPPOTrainer` 对各通道独立计算 advantage，再由 `combine_channel_advantages` 按配置合成策略更新所需 advantage。

这种设计的意义在于：Q-time、utilization、执行合法性不再被过早混合到一个标量 critic，降低不同量纲目标互相污染的风险。

### 5.4 PPO-Lagrangian

`MultiHeadPPOTrainer` 支持 `use_qtime_lagrangian=True`。启用后，Q-time 通道权重由 Lagrange multiplier `lambda_qtime` 自适应调整，而不是固定 `w_qtime`。这将 Q-time residual 更接近 CMDP 约束处理方式。

当前实现边界是：Lagrangian 作用于训练目标层面，不能替代候选池中的 Q-time mask。换言之，mask 是决策时硬筛选，Lagrangian 是训练时对残余违约的软压力。

---

## 6. Episode 驱动与训练入口

### 6.1 `Phase2EpisodeDriver`

`Phase2EpisodeDriver` 将环境单步动作组织成完整 episode。它负责：

- 选择下一台 idle machine；
- 构建候选池与 observation；
- 调用 rule 或 policy 选择动作；
- 处理 wait、failed action、no future event 等终止条件；
- 汇总 completed lots、Q-time、utilization 等指标。

当前支持：

- `run_rule_episode(strategy=...)`
- `run_policy_episode(policy, buffer)`
- `run_multihead_policy_episode(policy, buffer)`
- `run_greedy_episode(policy)`

训练时若 policy 采样到失败动作，rollout buffer 保留原始 action 与 log_prob，不在训练轨迹中替换为 fallback 动作。这是 PPO ratio 正确性的关键不变量。

### 6.2 训练入口

`train_phase2_sas_ppo.py` 提供主要训练入口，支持 small、random、pressure、multihead 等模式，并已经支持 `late_hi` 相关路径。`parallel_rollout.py` 为 multihead 训练提供多进程 rollout collector，将多个 worker 的 episode 合并到一个 `MultiHeadRolloutBuffer` 中再做 PPO update。

`train_late_hi.py` 是专门面向 `late_hi` 的训练 launcher，用于避免 Windows spawn 多进程导入 `__main__` 时的问题。

---

## 7. 实例、baseline 与评估协议

### 7.1 实例族

`problem_instances.py` 当前包含三个关键实例构造：

- `build_small_encoder()`：小规模 sanity check；
- `build_pressure_test_encoder()`：50-lot 压力实例，含 staggered arrivals 与 stage Q-time limits；
- `build_late_hi_encoder()`：高 priority 与晚到达强相关的 reservation go/no-go 实例。

`late_hi` 的研究意义在于：只有当高优先级 lot 往往较晚到达，且当前派工可能占用其兼容机台时，预留动作才具备结构性价值。否则 reserve 长期输给 dispatch 并不一定是算法失败，也可能是实例中没有预留杠杆。

### 7.2 规则 baseline

`evaluate_baselines.py` 提供多 seed baseline 评估。规则策略由 `Phase2EpisodeDriver._rule_action_index` 统一实现，包括：

```text
first_valid
FIFO
SPT
EDD
CR
ATC
```

重要的是，规则 baseline 和 RL policy 使用相同的 qtime-safe candidate pool。因此对比时不是“规则没看到约束、RL 看到了约束”，而是在同一约束候选池内比较选择策略。

### 7.3 噪声与 seed

评估中的 seed 对应 process noise realization。`process_noise_enabled=True` 时，commit 路径注入加工时间噪声；dry-run 仍保持非破坏性和均值路径，除非特定 Q-time chance mask 显式使用 Monte Carlo RNG。

### 7.4 trace 与表格

当前已具备：

- `vc_mcts_probe.py`：baseline / oracle / VC-MCTS 对比；
- `oracle_reservation_probe.py`：oracle reservation 上界或 go/no-go 探针；
- `vc_mcts_trace_summary.py`：汇总 reserve availability、reserve selection rate、O2 gap、Q-time gap 等；
- `compile_comparison_table.py`：汇总 probe JSON 形成比较表；
- `exp_qtime_chain.py`：aggregate vs chain Q-time mask 消融；
- `exp_arrival_prob.py`：arrival probability weighting 消融。

这些脚本构成当前论文实验的最小闭环，但正式结论仍需要更大 seed 数、更系统的实例族和 checkpoint 对照。

---

## 8. Reservation 与 VC-MCTS

### 8.1 Reservation ledger

`ReservationLedger` 只负责记录和维护 reservation 状态：

- 预留哪台 machine；
- 面向哪个 future_lot；
- ETA 与 TTL；
- 到期释放；
- 目标 lot 到达并被 dispatch 后 consume；
- 拒绝同一 future_lot 被不同 machine 重复预留。

ledger 本身不判断“值不值得预留”。是否接受 reserve 由 oracle 或 VC-MCTS 通过 rollout 结果决定。

### 8.2 ROP：预留机会生成

`detect_reservation_opportunities` 根据当前 idle machine、lookahead window、future lots、兼容 PPID、已有 reservation 等信息生成候选 `ReservationOpportunity`。ROP 是计算门控，不是最终决策器。

换言之，ROP 只回答“哪些 reserve 候选值得拿去搜索”，VC-MCTS 回答“在当前状态下 reserve、dispatch、no_op 哪个更好”。

### 8.3 VC-MCTS action space

当前 `VCMCTSPlanner` 在 root 层构造三类动作：

```text
no_op
delegate_dispatch(machine)
reserve(machine, future_lot)
```

当 `use_delegate_dispatch=True` 时，具体 `(lot, ppid)` 选择交给 delegate，planner 只比较 reserve / no_op / delegate_dispatch 的高层分支。这降低了 root 分支数，也让 SAS policy 能够自然接入 MCTS rollout。

### 8.4 目标函数

`VCMCTSObjective` 使用字典序比较，核心维度包括：

1. Q-time violation count；
2. priority-weighted wait，即 O2 proxy；
3. utilization；
4. Q-time violation total。

当前代码还记录 `is_leaf_bootstrap` 等 trace 字段，用于区分完整 rollout 目标和叶子 critic bootstrap 目标。

### 8.5 no_op gating

`no_op` 不是普通等待动作。当前 planner 对 no_op 设置 gating：只有当 no_op 在 Q-time count 或 Q-time total 上严格优于非 no_op 边时，才允许它胜出。这个机制用于避免轻预算 MCTS 中 no_op 因搜索噪声或 visits 统计而过度胜出。

### 8.6 dispatch delegate

`dispatch_delegate.py` 建立了清晰接口：

- `RuleDispatchDelegate(strategy)`：用 FIFO/SPT/EDD/CR/ATC 等规则选当前派工；
- `SASPolicyDispatchDelegate(policy, fallback_delegate)`：用训练好的 SAS policy 选动作，失败时回退到规则；
- `load_sas_policy_delegate(checkpoint_path, ...)`：从 checkpoint 构造 SAS delegate。

这使 VC-MCTS 不必关心派工细节，只需要在 rollout 或 root 边中请求 delegate 给出当前可执行动作。

---

## 9. AlphaZero 风格可选增强

`vc_mcts_alphazero.py` 提供两类默认关闭的增强。

### 9.1 SAS policy prior

`SASPolicyPriorProvider` 将 SAS policy 对候选池的 masked softmax 概率映射为 MCTS root edge prior。由于 SAS policy 不建模 reserve/no_op，reserve 与 no_op 仍需要 planner 注入固定探索 prior。

对应配置包括：

- `prior_source="heuristic"`：默认路径；
- `prior_source="policy"`：启用 SAS policy prior，需要 checkpoint。

当前 `vc_mcts_probe.py` 已做防呆：请求 policy prior 但未提供 checkpoint 会直接报错，而不是静默退回 heuristic。

### 9.2 Multi-head critic leaf value

`MultiHeadCriticLeafValue` 使用 multihead critic 对叶子状态估值，以减少完整 rollout-to-terminal 成本。映射逻辑大致为：

- `qtime` critic 映射为剩余 violation 的近似；
- `util` critic 映射为 utilization 估计；
- 没有 critic 通道的 O2 与 qtime_total 由 partial-horizon rollout 补齐。

因此 leaf value 是近似路径，不等价于完整 rollout。trace 中的 `is_leaf_bootstrap` 和 edge stats 中的 leaf bootstrap 统计用于识别这种估值来源。

与 prior 相同，`use_leaf_value=True` 也需要 checkpoint；否则 probe 会报错。

### 9.3 arrival probability weighting

`arrival_prob_weighting=True` 会按 ETA 距离对 reserve payoff 做折扣，避免远期 future lot 的预留收益被过度乐观估计。对应参数包括 `arrival_prob_decay` 和 lookahead window。

---

## 10. 当前测试与质量状态

本地测试套件覆盖了当前主线的不变量。按前一次完整验证结果，`FAB_RL/FABenv/tests/` 为：

```text
82 passed
```

重要测试包括：

- `test_decoupling_consistency.py`：`estimate` 与 `schedule_on_calendar` 共享确定性核心；
- `test_decoupling_rollback.py`：dry-run 非破坏性、commit 后 rollback 可恢复；
- `test_hard_pressure_instance.py`：pressure 实例包含有效 Q-time limits 和 staggered arrivals；
- `test_late_hi_instance.py`：late_hi priority 与 late arrival 强相关，并接入 eval/probe factory；
- `test_qtime_chain_mask_rng.py`：chain mask dry-run 不推进 commit noise RNG；
- `test_reservation_ledger.py`：reservation consume/release、重复 future_lot 拒绝；
- `test_reservation_rop.py`：ROP 生成、TopB、已预留机台和 lot 过滤；
- `test_reservation_simulator.py`：reservation-aware rollout 与 priority wait metric；
- `test_dispatch_delegate.py`：rule/SAS delegate 与 fallback；
- `test_vc_mcts_planner.py`：root action、objective、no_op gating、trace、probe；
- `test_vc_mcts_trace_summary.py`：trace summary 与 reserve lost_to 统计；
- `test_vc_mcts_alphazero.py`：policy prior、leaf value、checkpoint requirement。

这些测试说明当前结构已经具备较可靠的工程闭环，但它们不等同于大规模实验显著性验证。

---

## 11. 已完成、可选、待验证清单

| 类别 | 状态 | 代码证据 |
|---|---|---|
| 下层共享排程核心 | 已完成 | `lower_layer_estimator.py`, `lower_layer_scheduler.py` |
| dry-run / commit 解耦 | 已完成 | `rl_environment.py`, `test_decoupling_rollback.py` |
| estimate cache | 已完成 | `ResourceCalendarEnv._estimate_cache`, `estimate(cache=...)` |
| Q-time aggregate / chain / chain_joint mask | 已完成 | `rl_environment.py`, `exp_qtime_chain.py` |
| 单头 PPO | 已完成 | `Phase2SASActorCritic`, `Phase2PPOTrainer` |
| 多头 PPO | 已完成 | `Phase2SASMultiHeadActorCritic`, `MultiHeadPPOTrainer` |
| PPO-Lagrangian | 已完成 | `MultiHeadPPOConfig.use_qtime_lagrangian` |
| pressure 实例 | 已完成 | `build_pressure_test_encoder` |
| late_hi 实例 | 已完成 | `build_late_hi_encoder` |
| baseline 多 seed harness | 已完成 | `evaluate_baselines.py` |
| reservation ledger / ROP | 已完成 | `reservation_ledger.py`, `reservation_rop.py` |
| oracle reservation probe | 已完成 | `oracle_reservation_probe.py`, `reservation_simulator.py` |
| VC-MCTS planner | 已完成 | `vc_mcts_planner.py` |
| SAS delegate 接入 VC-MCTS | 已完成 | `dispatch_delegate.py` |
| policy prior / leaf bootstrap | 可选增强，已落地 | `vc_mcts_alphazero.py` |
| 大 seed 噪声实验 | 待系统验证 | `vc_mcts_probe.py`, `compile_comparison_table.py` |
| 公开 benchmark 对标 | 待完成 | 当前无正式实现 |
| 学习式 RMA | 已替代，不作为当前路线 | VC-MCTS 替代 |

---

## 12. 与旧报告的主要差异

旧报告中很多内容属于设计稿或前一阶段研究叙事。当前重构后应采用以下表述：

1. “RMA 预留代理”改为“VC-MCTS 在线预留规划器”。
2. “奖励学习是否 hold”改为“reservation-aware rollout 比较 reserve / dispatch / no_op”。
3. “下层估时器与环境排程各自实现”改为“共享 `schedule_deterministic` 核心”。
4. “Q-time 作为 shaping”改为“候选池 mask + 终局 residual / Lagrangian”。
5. “priority shaping”改为“Q-time safe 集合内的 priority filter / ranking + O2 objective”。
6. “多目标加权奖励”改为“向量 reward + 多头 critic + advantage 合成”。
7. “SAS 与 MCTS 混杂”改为“VC-MCTS 决定 reserve/no_op/delegate_dispatch，delegate 决定具体派工”。
8. “AlphaZero 化是主路径”改为“policy prior 与 leaf value 是默认关闭的可选增强”。

---

## 13. 下一阶段建议

### 13.1 实验验证

下一阶段应优先完成实验矩阵，而不是继续扩展结构：

- deterministic 与 process-noise 两套设置；
- pressure 与 late_hi 两类实例；
- rule delegate 与 SAS delegate 对比；
- heuristic prior 与 policy prior 对比；
- full rollout 与 leaf bootstrap 对比；
- aggregate / chain / chain_joint Q-time mask 消融；
- arrival probability weighting 消融；
- 多 seed 均值、标准差和 per-seed trace 诊断。

### 13.2 训练与 checkpoint

当前 SAS checkpoint 在不同噪声强度、不同 arrival-priority 相关结构下可能出现 seed-level 交叉。建议单独训练：

- pressure deterministic checkpoint；
- pressure noise checkpoint；
- late_hi deterministic checkpoint；
- late_hi noise checkpoint。

并在 `compile_comparison_table.py` 输出中区分 checkpoint 来源，避免把“策略泛化能力不足”误判为“VC-MCTS 预留无效”。

### 13.3 论文表达边界

论文中应避免过度声称“全局最优调度”。当前方法本质是：

```text
固定机台事件推进顺序
  + SAS 当前派工选择
  + VC-MCTS 局部在线预留搜索
  + 下层确定性/带噪声日历仿真
```

其贡献可以稳健表述为：在具备有限可见未来到达、Q-time 硬约束和高优先级晚到达结构的 FAB 机台组场景中，用搜索式反事实预留替代学习式 hold/RMA，降低 reward 设计风险，并复用 SAS policy 作为派工 delegate、搜索先验和叶子估值来源。

---

## 14. 总结

本项目当前已经从早期“环境 + PPO 派工”的原型，推进到“下层排程单一事实来源 + 多头 SAS + 在线 VC-MCTS 预留”的闭环系统。下层代码保证估计、dry-run、commit 的一致性；SAS 负责当前候选池内派工；VC-MCTS 负责在有限前瞻窗口内搜索是否为未来 lot 预留 idle machine；AlphaZero 风格 policy prior 和 critic leaf value 作为可选加速或增强路径存在。

最重要的架构判断是：预留不再通过一个额外 reward 通道学习，而是通过真实 ledger-aware rollout 直接比较反事实后果。这一点解决了旧 RMA 方案中类别不平衡、奖励收缩、greedy under-hold 和重复计奖的主要风险，也使当前代码更容易验证、调试和写成可信的项目报告。

后续工作的核心不是再堆叠新模块，而是扩大实验规模，明确 SAS delegate、policy prior、leaf bootstrap、arrival probability weighting 在不同实例与噪声条件下的贡献边界。
