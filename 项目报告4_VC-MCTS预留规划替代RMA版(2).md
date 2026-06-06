# 面向半导体 FAB 机台组调度问题的项目报告（完善版）

> 本版在初版《报告方案》基础上做了系统性完善，核心变化集中在七处：
> （1）补齐**到达模型与有限前瞻机制**，把「黑盒随机到达 + 可见即将到达」这一真实设定写进 MDP；
> （2）将 Q-time、priority 从**奖励信号**升级为**机制约束**（机会约束 mask + priority 候选池过滤），确立「Q-time > priority > 利用率」的字典序结构（Q-time 为阶段间队列时间硬约束；拖期为独立的交付期指标，见 §4.10）；
> （3）显式处理**加工时间不确定性**（纯工艺噪声），将 Q-time 改写为**机会约束**、safety margin 改为 z_ε·σ 自适应；并将 Q-time 硬约束建模为 **CMDP、用 PPO-Lagrangian（Safe RL）** 处理残差违规（3.3）；
> （4）明确**两层架构**：下层固定规则估算工件内部批处理的完成时间分布，上层 RL 做派工/时机决策（1.5）；
> （5）将奖励由**加权标量**改为**向量**，Critic 由**单头**改为**多头**，GAE **逐目标独立计算**；
> （6）将原 DDT 的无目标 `hold`（及上一版的学习式 RMA）改为 **VC-MCTS 预留规划器（Visibility-aware Constrained Monte-Carlo Tree Search，可见性感知的约束式搜索）**：采用宽触发式 ROP（Reservation Opportunity Point）作为算力闸门构造潜在预留候选，把 `reserve(machine, future_lot)` 作为**搜索树的边**，用已有资源日历当忠实仿真器对"现在派 vs 留给未来"两条分支 rollout 比较，从而在结构上消除学习式预留的奖励坍缩与 greedy 阈值顽疾；SAS 仍走 RL 训练并被复用为搜索先验/基策略/叶子估值（§5）；
> （7）新增「**与主流方法对比及可借鉴点**」一章，并给出基准与验证建议。

---

## 0. 修订说明：相对初版的关键变化

| 维度 | 初版做法 | 完善版做法 | 解决的问题 |
|------|----------|-----------|-----------|
| Q-time（阶段间队列时间）| shaping 软惩罚 + 特征 | **机会约束 mask**（`deadline−μ<z_ε·σ`，违规概率≤ε）+ 残差通道（固定权重或自适应 λ） | 硬约束不能靠"劝"；与交付期拖期是不同的量（见 §4.10）|
| priority | shaping 软惩罚（-0.02） | **候选池过滤**（qtime-safe 集合内）+ **VC-MCTS 目标绑定式资源预留（搜索，非学习）** | 弱信号撑不起"高优先级优先"；无目标 hold 与学习式预留都易坍缩 |
| 即时奖励 | 执行 + 质量混在一起 | **只剩执行合法性** | 消除成功奖励与质量信号抵消 |
| 奖励形态 | 加权标量 | **向量**，各通道独立 | 分量不丢、权重可解释 |
| Critic | 单头 V(s) | **多头** V_k(s) | 消除目标间梯度干扰 |
| 优势估计 | 单标量 GAE | **逐目标 GAE + 归一化加权** | 权重作用在无量纲量上，稳定 |
| 到达模型 | 未显式建模 | **黑盒到达 + 有限前瞻窗 + 预留机会点 ROP** | 让"为未来高优先级 Lot 预留哪台机器"在窗内可仿真、可搜索、可验证 |
| 加工时间 | 视为确定值 | **随机噪声（μ,σ 入状态）+ Q-time 机会约束** | 真实工艺有波动，硬约束须改写为概率约束 |
| 架构 | 未区分层次 | **两层：下层固定规则估时（1.5）+ 上层 RL 派工/时机** | 内部批处理时序需算准，但无取舍，不必用 RL |
| 评价/对标 | 自定义多目标指标 | 增加**公开基准与启发式对标**（分层评估，不强主张帕累托） | 可信度、可复现 |

---

## 1. 算法升级总体框架

> **以资源日历为环境，以候选池 + 约束 mask 控制动作空间，以多头 PPO 训练可学习的 SAS 派工策略，并在其上叠加 VC-MCTS 预留规划器（搜索式、非学习）处理"为未来高优先级 Lot 预留资源"的时机决策。**

### 1.1 总体流程

```text
事件触发或当前时刻推进
      ↓
读取资源日历状态 s_t（含前瞻窗内即将到达 Lot）
      ↓
检测是否存在预留机会点 ROP（Reservation Opportunity Point）
      ↓
若无 ROP：跳过预留搜索，按"最受约束优先"规则逐台调用 SAS 派工
若有 ROP：构造候选预留对 P_t={(空闲 machine m, 未来可保护 lot h)}，并按软机会分数截断为 Top-B，建搜索树
      ↓
VC-MCTS 预留规划器：在 [no_op] + dispatch 边 + reserve(machine m, future_lot h) 边中
  以约束式 UCT 选择、reservation-aware rollout 评估，并按
  qtime_violation_count → qtime_violation_total → O2(priority_weighted_wait) → utilization
  的目标优先序定动作；no_op 只有在 Q-time count/total 严格优于非 no_op 边时才允许胜出（搜索，非学习；详见 §5）
      ↓
对被 reserve 的机器登记 reservation ledger（目标 lot、机器、ETA、超时、兑现/释放）
未被 reserve 的空闲机器继续进入 SAS 派工流程
      ↓
按"最受约束优先"规则选出当前决策的空闲 Machine m（多台空闲时逐台，见 1.3）
      ↓
生成 Machine m 的结构可行候选动作池 A_t^m
      ↓
① qtime-safe mask → ② priority 过滤 → ③ CandidateScore 排序 → ④ TopK+padding+mask（详见 3.1）
      ↓
SAS Actor 在固定长度候选池中选择 action_index
      ↓
环境映射为真实动作 (lot, machine, ppid)，资源日历 dry-run / commit
      ↓
计算向量 reward 和下一状态
      ↓
PPO 用多头 Critic + 逐目标 GAE 联合更新
```

> 上图把原先 DDT 的"是否 hold"、以及上一版学习式 RMA 的"预留概率"，改成 **VC-MCTS 的搜索式目标绑定预留**：规划器不在所有空闲点都决策，而是在宽触发式 ROP 发现"可能有预留价值"的场景后建树，由对 reserve 与 dispatch/no_op 分支的 rollout 模拟比较决定是否预留、留给谁。**当前代码第一阶段尚未把训练好的 SAS policy 接入下层派工器**：`vc_mcts_planner.py` 直接从 `ResourceCalendarEnv.build_candidate_pool(machine)` 取 Top-K dispatch 边，并用规则 rollout（如 FIFO）评估；SAS policy 接入将作为后续 `delegate_dispatch` 接口完成。这样把上一版 RMA 想用奖励学的反事实收益，先改为由仿真直接、精确算出（§5.5/§5.7）。

各模块职责：

```text
下层估时器：算 (μ_finish,σ_finish)+占用区间，供 mask/打分/日历登记用（1.5，固定规则+蒙特卡洛，非RL）
前瞻窗：让预留规划器能看见即将到达的 Lot，从而把窗内未来 lot 作确定性实体放进搜索转移模型
ROP 检测器：采用"宽触发 + 软评分"发现潜在预留场景，作为搜索的算力闸门，只在可能有价值时建树，且不替规划器做"值不值得预留"的决策
VC-MCTS 预留规划器：在 [no_op]/dispatch/reserve(machine,future_lot) 边上搜索，负责目标绑定式资源预留（搜索，非学习，§5）
候选池：控制 SAS 当前派工动作规模
qtime-safe mask：硬约束，屏蔽会使未注定违规的可见 Lot 踩穿 qtime 的动作（详见 3.2）
priority 过滤：强偏好，在 qtime-safe 集合内优先高优先级 Lot
Top-K 打分：软目标启发式，仅用于排序与补足
SAS Actor：目标架构中学习当前机器的候选动作选择概率，并可复用为搜索先验/基策略/叶子估值；当前已落地 VC-MCTS slice 暂用候选池 Top-K dispatch + 规则 rollout 代替训练好的 SAS policy
多头 Critic：分目标估计状态价值（兼作搜索的叶子截断估值）
资源日历：执行可行性校验、预留 ledger、状态转移，并作为搜索的忠实仿真器（dry-run/commit）
向量 Reward：执行合法性 + 软目标，分通道提供 SAS 的训练信号（预留不再用奖励）
PPO：逐目标 GAE 更新 SAS 策略与多个价值头
```

### 1.2 MDP 形式化定义

环境状态 `s_t` 在初版基础上**补充两类信息**（加粗为新增）：

```text
s_t = {
  当前时刻 t_now,
  未完成 Lot 集合：arrival / due_date / priority / wafer_count,
  **每个已可见 Lot 的 qtime 窗口截止时刻 qtime_deadline(l)**,
  **前瞻窗 [t_now, t_now + W_lookahead] 内即将到达的 Lot 及其 priority / qtime / ETA**,
  空闲 Machine 集合 M_idle,
  **每台 Machine 的能力：可加工 recipe 集合 / 可用 ppid（异型柔性下为关键信息，见 1.3）**,
  每台 Machine 的 Lot-level calendar,
  每个 (machine, chamber, side) 的 wafer-level calendar,
  已提交的 Lot / Wafer 操作,
  当前窗口内已完成 / 未完成 Lot 状态
}
```

网络输入仍为特征化摘要 `z_t = featurize(s_t)`，新增特征：

```text
z_t 新增:
  每个候选 Lot 的剩余 qtime（qtime_deadline - 预计开始时刻）
  前瞻窗内即将到达的高优先级 Lot 数量、最紧 qtime、最早 ETA
  "若现在排满，前瞻窗内可能无法服务的高优先级 Lot 估计数"
  预留候选对特征：future_lot h 与 idle_machine m 的兼容性、ETA 冲突、priority gap、可替代机器数
  机器能力相关：候选动作的 (lot,machine) 匹配度、机器可服务的窗内高优先级 lot 数（异型柔性）
```

完整日历保留在环境中，网络只用可计算摘要。

### 1.3 局部动作空间

**机器设定（本报告默认）**：机台组内机器为**异型 + 柔性**——不同机器可加工的 recipe/工件集合不同（异型），且一个工件通常可被多台机器中的若干台加工（柔性）。机器能力（可加工 recipe 集合、可用 ppid）作为机器特征进入状态。同型为同一组机器能力相同的退化特例，本框架自动覆盖（候选池与可行性过滤逻辑不变）。

单台 Machine `m` 空闲时，动作为 `a_t^m = (lot, ppid)`，`machine = m` 由状态给定。多台同时空闲时**逐台决策**（避免组合动作空间），每台决策后立即更新日历与 mask 再处理下一台。**处理顺序采用"最受约束优先"**（异型柔性下顺序影响结果，须讲究）：

```text
排序键: 主键 = 该机器候选池里最紧的 qtime 余量（越紧越先决策）
        次键 = 该机器 qtime-safe 候选数（越少越先，即能力最受限者优先）
理由: 异型下机器能力不同，先处理"手上有快违规工件"或"选择余地小"的机器，
      避免其候选被余地大的机器抢走而陷入空闲/违规。处理顺序由固定规则给定，
      不作为预留规划器决策（否则徒增分支因子）。同型退化为"最早空闲优先"即可。
```

> **决策粒度（需在实现时定死）**：预留规划器的"是否预留以及预留给谁"与上述"逐台决策"如何嵌套有两种取法，本报告默认取后者：(i) 全局取法——规划器全局决一次预留，随后 SAS 对所有空闲机台逐台派工；(ii) 逐台取法（默认）——每选出一台空闲机台 m 作为一个决策点，先由 VC-MCTS 对候选预留对搜索决策；未被预留的机器再由 SAS 对 m 选动作。逐台取法语义更一致（预留本就是针对"某台机台是否留给未来"的决策），但决策点更多、建树更频；因此用宽触发式 ROP 当算力闸门，只在存在**潜在**预留机会时才建树，且把是否值得预留交给搜索的模拟比较；实现时二选一并在状态中标明"当前决策针对哪台机台"。

> **设计边界（须诚实声明）**：本框架中**机器分配本身不由策略学习**——VC-MCTS 规划器决定"是否以及为谁预留机器"、SAS 学"派哪个 (lot, ppid)"，但"用哪台机器"由固定规则按顺序给定。机器分配是经由"逐台顺序决策"**隐式涌现**的：每台机器的候选池只含它能加工的 lot，SAS 为每台依次选择，最终分配由这一序列决定。其代价是分配质量**对固定机器排序敏感**，且非全局最优。这是相对"同时学作业+机器分配"的主流 FJSP 工作（如多指针网络、HGNN 的作业-机器双输出）的一个范围限制。若后续要消除该限制，可将机器选择也纳入动作（扩为 `(machine, lot, ppid)` 候选），但会扩大动作/分支空间、增加 SAS 训练与搜索难度——建议作为后续工作，而非基础版目标。

### 1.4 三层字典序约束结构（新增）

本项目的目标不是"几个目标加权"，而是**严格分先后**的三层结构。**关键前提（以代码实现为准）：阶段间 Q-time（材料队列时间，`q_time_limits`）是硬约束，工件衔接超窗即报废，目标严格为 0，归入第 0 层；交付期拖期（due date）是独立的量，仅作评估指标、不进奖励（当前压力实例下较宽松）**：

```text
第 0 层（硬）：阶段间 Q-time 不可违（材料队列时间）—— 机会约束 mask（窗内）+ cost/λ 兜底（窗外）
第 1 层（强）：高优先级 Lot 优先                  —— 候选池过滤 + VC-MCTS 预留，但不得踩穿第 0 层
第 2 层（软）：机台利用率（进度为辅助代理）       —— 奖励通道，唯一可让步的目标
```

执行顺序「先 mask → 再过滤 → 再打分」天然实现字典序：会害 Q-time 的高优先级动作在第 0 层被屏蔽，第 1 层无机会强推，**Q-time 永远优先于 priority**。

> 由此，真正"可互相权衡的软目标"只有利用率一项（拖期当前不进奖励，仅作评估指标）。这意味着本项目本质是「**单软目标 + 两层约束（Q-time 硬、优先级强）**」，而非多目标权衡问题——下文奖励与 Critic 的三通道设计据此调整（4.5、4.7），不作"多目标帕累托"包装。

### 1.5 两层决策架构与下层启发式估时器（新增）

本项目的决策天然分两层，**只有上层用 RL，下层用固定规则（非 RL、非随机搜索）**。明确区分二者是整个方案能成立的前提。**两层接口（须单向、窄）**：上层只向下层传 `(工件 O, machine m, ppid p)` 与各阶段加工时间的 `(μ_s, σ_s)`；下层只向上层返回完成时间分布等少数输出（见本节末"下层交付物"），**上层不依赖下层的组批/流水/实例选择等内部细节，下层也不知道上层的 RL/候选池/预留逻辑**。其中 `estimated_earliest_start`（候选在 m 上的最早开始时刻）由**上层从资源日历推得**（不是下层产出），下层在此基础上算 `finish_time` 分布——二者分工要分清。

> 术语澄清：下层的**排程逻辑**是固定规则（确定性的组批 + list scheduling），但它作用在**随机的加工时间**上（2.4），故其**输出是完成时间的分布** `(μ_finish, σ_finish)`，而非单一值。"固定规则"指算法不学习、不随机搜索；"分布输出"来自输入（各阶段加工时间）的随机性。二者不矛盾。

```text
上层（SAS 用 RL 派工 + VC-MCTS 搜索预留）：决定 哪个工件 / 哪台 machine / 哪个 ppid / 什么时机
  —— 存在真实取舍（利用率 vs 优先级 vs 不违 Q-time），故用 RL 学
下层（启发式估时器）：一个工件被派到 (machine, ppid) 后，其一批 wafer 在 step 内部
  如何组批、每批进哪个 chamber 实例、各批时序
  —— 无取舍（wafer 对称、step 内 chamber 间无 Q-time、目标单一=尽早完成），故用规则算
```

**下层问题结构（来自问题定义）。** 单个工件的一个 step：ppid 锁定 chamber 类型的经过顺序（A→B→C，工艺路径不可变）；每个 chamber 类型有多个物理实例可选（柔性）；每个 chamber 内多个 side 同进同出（批处理机，一次成批、整批同时开始结束）；工件不可拆批（**指作为一个派工单元整体派出、其 wafer 不与其他工件混入同一炉**，而非"机器被单块占用"），wafer 间无顺序约束。当 wafer 数 > side 容量时，工件内部分成多个**子批**，子批属同一工件、彼此可在阶段间流水重叠（故机器各阶段在重叠时段被不同子批占用，并非单块占用）。这是一个**带批处理机的柔性流水车间（hybrid flow shop with batching）**。

**下层启发式算法。**

```text
1. 组批: 满批优先, N 片 wafer → ⌈N/side容量⌉ 个子批
   前提(a): 批处理时间与装载片数无关——批处理炉通常如此；若相关需另算。
   注意: N 非容量整数倍时，最后一个子批不满；在前提(a)下，非满批子批仍占
         整批的批处理时间（不按片数缩放），makespan 估计须据此计。
2. 流水排程 (list scheduling):
   初始化: 每个阶段每个实例的 free_time[inst] = 该实例当前空闲时刻
   for 每个子批 b（固定发批顺序，如 FIFO）:
     for 每个阶段 s（按 ppid 顺序）:
       ready = b 在上一阶段的完成时间（首阶段为 b 可开工时刻）  # 子批自身前序约束
       # 选"能让 b 最早开工"的实例，而非单纯"最早空闲"的实例
       inst* = argmin over 阶段s的实例 inst of  max(ready, free_time[inst])
       start[b,s] = max(ready, free_time[inst*])
       end[b,s]   = start[b,s] + 批处理时间(s)
       free_time[inst*] = end[b,s]            # 关键: 占用后立即更新该实例空闲时刻
   makespan = max_b end[b, 末阶段]
3. (可选) 用 NEH 优化发批顺序，子批少时收益有限
```
> 两个实现要点：(1) 实例选择用 `argmin max(ready, free)`（最早能开工）而非 `argmin free`（最早空闲）——多实例忙闲不均时二者结果不同，后者会把子批挤到同一实例；(2) 分配后**必须更新** `free_time[inst*]`，否则所有子批会被算到同一最早空闲实例上（批处理机一次只能一炉）。两者都是贪心，仍不保证全局最优，但描述需精确。
> makespan 主要由**瓶颈阶段**（有效吞吐最低 = 实例数×(1/批时间) 最小的阶段）决定，组批与排程服务于"喂饱瓶颈"。问题规模小、无取舍，贪心已接近最优，**不用 RL、不用重型优化（MILP/GA）**。

**下层交付物（上层的地基）。** 在加工时间不确定性（2.4）下，下层输出的不是单值而是分布刻画。**计算方式**：对固定的组批+排程规则，用各阶段加工时间的 (μ_s, σ_s) 做**蒙特卡洛采样**——多次采样各阶段实际时间、每次跑一遍 list scheduling、对得到的 makespan 取均值与标准差，即得 `(μ_finish, σ_finish)`（这天然解决下面易错点(1)的 E[max] 偏置）。规模小时蒙特卡洛开销可接受；若仍嫌慢，可用均值路径算 μ 再加偏置修正、用 `sqrt(Σσ_s²)` 近似 σ。

> **开销警示（异型柔性下尤其要注意）**：下层被上层每次候选评估调用一次，而每次调用要跑 K 次蒙特卡洛、每次蒙特卡洛跑一遍 list scheduling，构成 `候选数 × K × list scheduling` 的三重开销；异型柔性下候选池大，可能成为训练瓶颈。缓解：(1) 对同一 `(O, m, ppid)` 在候选池稳定期间**缓存**估时结果（输入不变则结果不变，因下层是固定规则）；(2) K 取较小值（如 20–50）即可获得稳定的 (μ,σ)；(3) 真急可退化为"均值路径 + 偏置修正"的解析近似，省掉蒙特卡洛。

```text
estimate(O, machine m, ppid p) →
  (μ_finish, σ_finish)     # 完成时间均值与标准差；喂上层 estimated_finish 与机会约束(2.4.3)
  per_instance_occupancy   # 各 chamber 实例占用区间（按 μ 登记）；喂资源日历做冲突检测
  bottleneck_stage         # 瓶颈阶段；可作上层候选池打分特征
```
> **两个下层接口（解耦后，共享同一 list-scheduling 核心 `schedule_deterministic`）**：**状态无关的 `estimate`**（实例 free 初值=0、结果可缓存，供 qtime mask / 候选打分用）与**状态相关的 `schedule_on_calendar`**（从真实资源日历读各实例当前空闲时刻作 free 初值、算绝对占用区间，是上层 dry-run / commit 的**薄封装**，非破坏性）。前者估时、后者落地，因共享核心而**口径一致**——这保证 qtime mask 预判的 makespan 与 commit 实际 makespan 同口径（修复了二者此前用两套算法导致的系统性偏差）。
> 资源日历的不确定性处理：登记占用时用**期望区间（按 μ）**做冲突检测与后续 lot 的估时；待该工件实际加工完成、真实时间揭晓后，再用**实际值回填**日历。即"规划用 μ、执行后用实际值校正"。
> 两个易错点（2.4.2 已述，此处落到下层）：(1) makespan 常为"多并行子批取最晚完成"，`E[max] > max(E[·])`，**仅用均值路径会系统性低估** μ_finish，需上偏修正或（推荐）蒙特卡洛；(2) 沿关键路径累加独立噪声，`σ_finish ≈ sqrt(Σσ_s²)`。
> 衡量下层做得对的标准是**"估得准"而非"makespan 全局最优"**——一个估时偏 10% 的最优调度器，对上层 Q-time 判断的伤害大于一个估时精确的贪心调度器。

### 1.6 下层估时器的误差感知学习式加速（可选增强）

> 本节是 §1.5 下层估时器的**优化变体**，不是独立研究内容，也不改变上层 RL 架构。核心主张不是"用神经网络把估时算得更快"（代理模型替代昂贵仿真是成熟套路，非本项目原创），而是：**在 Q-time 报废这类零容忍硬约束下，直接用代理模型估时是危险的——本节提出一套误差感知的混合估时：远离约束阈值用快速代理、贴近阈值自动回退精确蒙特卡洛，并以"决策翻转率"而非回归误差验证代理质量。** 加速是副产品，约束安全是主张。

**1.6.1 动机：先测量，再决定（不优化未实测的瓶颈）。** §1.5 的"开销警示"已指出，下层估时器被上层每次候选评估调用一次，构成 `候选数 × K × list-scheduling` 的三重开销，异型柔性下候选池大、可能成为训练瓶颈。但这是一个**预防性警示，而非实测瓶颈**。因此本节的第一步不是实现代理模型，而是 **profile**：给现有蒙特卡洛估时器加计时，跑一轮训练，测两个数——(a) 下层估时占单步训练耗时的比例；(b) §1.5 缓存（同一 `(O,m,ppid)` 输入不变即缓存）的命中率。仅当估时占比显著、且 §1.5 缓存收益已用尽时，才引入下文的学习式代理；否则将本节列为未来工作。（"占比显著"的阈值无通用定值，应据 profile 实测结果与代理的实现/维护成本权衡确定，而非预设一个固定百分比。）这一"先 profile 再优化"的姿态本身写入实验记录，作为工程严谨性的体现。

**1.6.2 为何可学：被代理的是一个确定性、解耦的函数。** §1.5 已把下层拆成**状态无关的 `estimate`**（实例 free 初值=0、结果可缓存）与状态相关的 `schedule_on_calendar` 两个口子。代理模型只替换**前者**——它学的是"工件在 (machine, ppid) 下其内部批处理本身需要多久"，这是个**不依赖当前日历、不依赖上层决策**的量，可缓存、可离线预计算。后者（从真实日历读各实例空闲时刻、算绝对占用区间，是 dry-run/commit 的薄封装）**仍走原精确算法，不被代理**——这条分工线必须画清，否则 commit 的实际占用会与代理预测漂移。

```text
被代理（estimate，状态无关）:  (O, m, ppid, 各阶段 μ_s/σ_s) → (μ_finish, σ_finish)
                                可缓存、可离线造样本、可批量推理
不被代理（schedule_on_calendar）: 从资源日历读 free 初值 → 绝对占用区间（甘特图）
                                上层 dry-run/commit 仍用精确 list scheduling
```

> **为什么两路必须分开（防止误读为"set-encoder 出甘特图再回传"）**：下层共享同一 list-scheduling 核心，但对外是两个口子，调用频率与所需量根本不同：
> - **估时口**（喂 mask/打分）：候选池里**每个** `(O,m,ppid)` 都要估 → **极高频**，是开销大头；要的是**状态无关**的完成时间分布 `(μ,σ)`（两个标量，**无甘特图**），可缓存/可学 → **由 set-encoder 一次前向替代**。
> - **落地口**（喂日历 commit）：仅对**被选中的那一个**动作执行 → **每步一次、低频**；要的是**状态相关**的绝对占用区间（即**甘特图** `per_instance_occupancy`，依赖当前日历各实例空闲时刻），**无法缓存、必须精确** → **永远走精确 list scheduling，不被替换**。
>
> 故 **set-encoder 只替估时口；甘特图永远出自落地口**。二者不能合并：合成全精确则开销大头未省（引入代理无意义），合成全用代理则拿近似占真实日历、后续估时与冲突检测误差滚雪球。共享同一核心保证两口口径一致（估时预判 makespan 与 commit 实际 makespan 同源不漂移）。

> 因下层排程逻辑是**固定规则**（§1.5），对同一输入永远给同一分布输出——本质是确定函数，仅"用蒙特卡洛求值"较慢。这带来两个好处：(1) 代理拟合的目标稳定、好学；(2) 训练数据**完全免费、自动标注**（见 1.6.3）。这比代理一个带噪仿真器的常见情形更干净。

**1.6.3 代理模型定义与免费监督数据。**

```text
输入（按结构组织，不拍平）:
  子批集合 {b_i}（变长、无序）: 每个 b_i = [片数/是否满批, 各阶段(μ_s,σ_s), 各阶段实例数/side容量]
  全局特征: bottleneck_stage（§1.5 下层交付物已产出）, ppid 阶段数, 总 wafer 数
输出:    完成时间分布参数 (μ̂_finish, σ̂_finish)
网络（set-encoder，置换不变）:
  共享 φ 编码每个子批 → e_i；对称聚合 ⊕(sum/mean) → g；ρ(g, 全局特征) → (μ̂, σ̂)
  理由: 子批"变长 + 无序(wafer 对称, §1.5)"，set-encoder 用结构焊死置换不变与变长支持，
        不必让 MLP 拿样本去硬学这个对称性；不用 GAT/TFT（无时序、下层无工件间图结构）。
  先 Deep Sets（共享 φ + sum）跑通；若子批争抢同一实例的耦合抓不准，再升级 set-attention。
输出端用 NLL（核心，对齐 mask）:
  不把 σ 当第二个回归目标，而用高斯负对数似然让网络"诚实报告把握"——
  σ̂ 直接进机会约束 z_ε·σ̂，其质量与 μ̂ 同等要命（σ 估小→漏放报废动作）。
（可选稳健增强）解析骨架 + 残差:
  μ̂ = μ_0 + Δμ, σ̂ = softplus(σ_0 + Δσ)，其中 μ_0=瓶颈阶段吞吐估计、σ_0=sqrt(Σσ_s²)；
  网络只学修正量，小实例下更省样本、外推更稳。
```

```text
监督数据生成（离线，不碰 RL）:
  for 每个结构合法的 (O, m, ppid) × 各阶段 (μ_s,σ_s) 采样点:
      多次跑现有蒙特卡洛估时器，每次记录其实际完成时间 t   # 既有组件即 ground-truth
      存 (输入 → 一组观测 t)；用高斯 NLL 训练（无需预先标 σ，σ 由观测离散度学出）
  规模小可近似覆盖到穷举；训练稳定且快，保留蒙特卡洛作标注与边界兜底。
```

**1.6.4 误差感知的混合估时（核心，本节真正的贡献点）。** 代理误差只有在**翻转机会约束判定**时才真正伤害决策。须对齐 §3.2 的 mask 口径：mask 不是只判被派 lot 自身，而是对某候选做 dry-run 后，**逐个可见 lot**（已排除 `is_doomed` 者）按 `deadline(l) − μ_finish(l) < z_ε·σ_finish(l)`（§2.4.3）判定。因此"何时回退"也必须**逐可见-lot**施加——某可见 lot 只要落在其阈值带内，就对该 lot 回退精确估时：

```text
对每个候选 (O, m, ppid):
  C_sim = dry_run_commit(...)
  for l in visible_lots:               # 对齐 §3.2
      if is_doomed(l): continue        # §3.2 既有：注定违规者不作屏蔽依据
      (μ̂, σ̂) = 代理(l, C_sim)
      slack = (deadline(l) − μ̂) − z_ε·σ̂   # 带符号裕量；>0 安全、<0 违规
      if |slack| > τ_safe:             # 远离阈值（两侧皆然）：代理不会翻转判定
          用代理判定（快）；slack<0 则该 l 触发 mask
      else:                            # 贴近阈值：代理不可信
          对该 l 回退精确蒙特卡洛重算后再判（慢，仅少数边界 lot）
```

> **关键修正**：判据用**带符号裕量** `slack` 的绝对值 `|slack| > τ_safe`，而非 `|deadline − μ̂|`。后者会在"μ̂ 已远超 deadline（注定违规一侧）"时取绝对值后变大、被误判为"远离阈值可信代理"——恰恰放过了最该回退的危险区。`slack` 的**符号**决定是否 mask，`|slack|` 决定是否回退，二者分开。

> 远离阈值的多数候选用代理秒算，真正决定 mask 的少数边界候选才花精确算力——既拿到加速，又不在硬约束判定上冒险。`τ_safe` 由 1.6.5 的决策翻转率校准。

**1.6.5 验证口径：决策翻转率，而非回归误差。** §1.5 立下的标准是"估得准优先于 makespan 最优"。据此，代理的验收**不看 RMSE**，看它对 mask 判定的影响：

```text
决策翻转率 = #{候选: 代理与蒙特卡洛给出不同 mask 判定} / #{全部候选}
报告: 留出测试集上的 μ̂/σ̂ 误差 + 决策翻转率（含混合策略前后对比）
判据: 翻转率需 ≤ 预设上限（上限为示意，据可接受的报废风险定）；超限则调大 τ_safe（更多 lot 回退蒙特卡洛）
```

> **诚实边界（与基金场景的关键差异）**：代理模型替代昂贵仿真在调度/优化领域是成熟方法，本节非方法学原创。基金将其用于离线优化的非瓶颈指标评价，代理误差仅影响"挑哪个解"——是软后果；本项目代理输出**直接喂 Q-time 机会约束 mask**，误差可致报废动作漏网——是硬后果。故本项目不能沿用"算得快就行"的态度，必须额外做 1.6.4 的阈值回退与 1.6.5 的翻转率验证。这是本项目比基金场景更严苛、因而独有的安全机制。
> **加速效果的内在张力（须实测）**：若实例中大量候选贴着 Q-time 阈值（§7.4 压力实例正是故意把约束调到"会咬"），回退比例会偏高，加速收益相应打折。回退占比与净加速比是必须实测的数，不可假设。
> **与既有内容的区分**：(1) 与 §7.3 借鉴点 2（Tassel 自监督预训练）正交——那是**加速策略学习**（作用于 agent，缓解稀疏奖励冷启动），本节是**加速环境估时**（作用于下层），可并存；(2) 与上层未来工件概率嵌入（如引入）正交——那在上层 attention 输入做文章，本节在下层估时实现做文章，分属不同层。

**1.6.6 分阶段开关。** 与 §3.3 对 PPO-Lagrangian "先固定权重、后自适应"的口吻一致：

```text
基础版: 全程用蒙特卡洛估时器跑通主线，不引入代理。
增强版（仅当 1.6.1 profile 确认估时为瓶颈时）:
  ① 离线造监督数据、训练代理；② 上线纯代理 + 阈值回退混合估时；
  ③ 用决策翻转率验收，校准 τ_safe；保留蒙特卡洛作标注与边界兜底。
```


---

## 2. 到达模型与前瞻机制（新增）

这是本项目区别于通用 FJSP 的关键设定，也是 VC-MCTS 预留规划器存在的根本理由。

### 2.1 黑盒到达 + 有限前瞻窗

工件**随机到达，到达率未知**（不能假设泊松或任何分布），但系统具备**有限前瞻能力**：一个 Lot 在真正进入待派工池之前，会在前瞻窗 `W_lookahead` 内变得**可见**（例如已从上游工序 release、在途、ETA 已知），可见信息包括它的 **优先级** 与 **qtime 窗口**。前瞻窗之外仍为完全黑盒。

> 注（基准实例 vs 模型假设）：用于测试的实例**可以用具体到达过程（如 Poisson）生成**其到达序列（见 §7.4 `build_pressure_test_encoder` 的错峰到达），但这只是数据生成方式；**agent 的状态/策略不得依赖到达率或分布形式**，前瞻窗之外一律按完全黑盒处理。二者不矛盾。

### 2.2 违规来源的两分

qtime 违规据此分为两类，可预判性完全不同：

```text
(a) 已可见 Lot 之间的相互挤占 —— 可预判
    已到达 + 前瞻窗内即将到达的 Lot，其 qtime_deadline 已知；
    某动作是否害它们踩穿窗口，当前时刻可按完成时间分布估算违规概率 → 机会约束 mask 屏蔽。
    对被调度 Lot 自身可保证违规概率≤ε；对其他可见 Lot 为显著降低（其真实
    开始时刻仍依赖后续决策）；已注定违规的 Lot 不作屏蔽依据（见 3.2）。

(b) 前瞻窗之外到达的 Lot 抢占资源 —— 不可预判
    现在把机器排满，窗外突然到来紧急 Lot 无机器可用 → 违规。
    信息上无法在当前时刻完全避免 → 用残差通道（固定权重或自适应 λ）做统计性兜底，
    并由 VC-MCTS 在可见窗内搜索选择性预留、由残差通道在窗外做统计性兜底。
```

随机到达只削弱 (b) 的可预判性，**(a) 照常可 mask**，且通常是违规的大头。

### 2.3 高优先级资源预留（VC-MCTS 预留规划器的核心职责）

因为前瞻窗内能看见即将到达的高优先级 Lot，"为它预留资源"从**不可做**变为**可仿真、可预判**。但这里不能把预留写成无目标的 `hold`，否则只知道"等不等"，不知道"为谁等、留哪台机器、等到了算不算成功"；上一版用学习式 RMA 给预留绑定目标，但学习范式仍会因 greedy 阈值与类别不平衡而坍缩（§5.1）。因此本报告把预留职责改由搜索承担：

```text
VC-MCTS 预留规划器（搜索式，非学习）
核心问题: 在有限前瞻窗内，是否需要把某台空闲机器 m 绑定给某个未来高优先级 Lot h？
决策方式: 把 reserve(machine m, future_lot h) 与 dispatch/no_op 作为搜索树的边，
          用资源日历仿真器 rollout 两条分支、按字典序目标比较（详见 §5）
```

**预留机会点 ROP（Reservation Opportunity Point）。** ROP 不是预留规则本身，而是**触发搜索的机会检测器与算力闸门**。本文采用"宽触发 + 软评分 + 搜索决策"的三层机制：硬触发层只判断"这个预留候选对是否有基本意义且不违法"，priority gap、ETA 距离、机器稀缺性、当前派工机会成本等因素不作为硬门槛，而作为搜索的 pair 特征与候选排序依据。换言之，ROP 只回答"是否值得建树搜一搜"，真正的"要不要预留"由 VC-MCTS 对 reserve 与 dispatch 分支的模拟比较决定。

**第一层：硬可行过滤（只保留必要条件）。** 一个候选预留对 `(m,h)` 进入搜索（成为 reserve 边）前，只需满足：

```text
1. 当前存在空闲机器 m，且 m 未被其他 reservation 冻结；
2. h 位于前瞻窗 [t_now, t_now+W_lookahead] 内，尚未到达但已可见；
3. m 结构上能够加工 h，即 compatible(m,h)=1；
4. h 不是 is_doomed(h)，即不是无论如何都无法挽救的 Q-time 注定违规 Lot；
5. 预留 m 不会直接导致已可见 Lot 的 Q-time 机会约束失效。
```

**第二层：软机会评分（只排序，不替搜索决策）。** 对通过硬过滤的 `(m,h)` 计算启发式预留机会分数 `S_res(m,h)`，用于排序与截断 Top-B 候选池（即 reserve 边集合），而不是直接决定预留：

```text
S_res(m,h)
  = priority_gain(h)
  + qtime_urgency(h)
  + machine_scarcity(h)
  + eta_overlap_if_dispatch_now(m,h)
  - current_lot_delay_cost(m)
  - idle_cost_until_eta(h)
```

其中 priority gap、ETA 远近、机器稀缺性、当前派工会不会覆盖 h 的 ETA、预留造成的 idle cost 等，都进入 `pair_feature(m,h)` 与 `S_res(m,h)`，但不作为硬规则删除候选。

**候选预留对。** ROP 上构造的是“可能值得预留”的候选集合，而不是“规则已经判定必须预留”的集合：

```text
P_t = TopB_by_S_res({(m,h) | m ∈ M_idle, h ∈ F_lookahead,
                              compatible(m,h)=1,
                              not is_doomed(h),
                              qtime_safe_after_reserve(m,h)=1})
```

搜索根节点的预留边集合为：

```text
reserve 边 ∈ { reserve(m,h) | (m,h) ∈ P_t }，与 dispatch 边、no_op 边一同进入搜索（§5.2）
```

若搜索选中 `reserve(m,h)`，环境在 reservation ledger 中登记 `(machine=m, target_lot=h, reserve_start=t_now, target_eta=ETA_h, expire_time, status=pending)`；机器 m 暂不进入 SAS 当前派工池，等 h 到达后优先尝试在 m 上开工。若 h 到达后成功使用该预留，记为 `hit`；若 h 未按时到达、被其他机器服务、预留超时或因 Q-time/结构可行性无法兑现，记为 `miss/waste` 并释放机器。

> 同型 vs 异型的预留差异：同型机器可互换，预留可弱化为"保留 k 台空闲额度"（数量预留）；**异型柔性下必须保留"能加工 h 的特定合适机器 m"（身份预留）**。因此搜索必须显式建模 `future_lot × machine` 的匹配关系（reserve 边即 (m,h) 对），而不能只看前瞻窗聚合标量。注意预留是**按机器粒度**的：只冻结被选中的机器，不冻结整个机台组。

这使预留从模糊的 `hold` 变成有对象、有责任、有验收标准的决策：**为哪个未来高优先级 Lot，留哪台机器，是否命中，是否浪费**。预留决策落在前瞻窗内时可预判；前瞻窗外到达仍不可预判，继续由残差通道与统计性安全余量处理。

### 2.4 加工时间不确定性与 Q-time 机会约束（新增）

> 本节确立一个贯穿全方案的设定：加工时间是**随机的**（来源：纯工艺噪声），估计时只知道分布、实际值加工完才揭晓。这把 Q-time 从"确定性硬约束"改写为**机会约束**，并让 safety margin 获得明确的概率含义。

**2.4.1 噪声模型。** 每个 (batch, stage) 的实际加工时间为标称均值加独立无偏噪声：

```text
p_actual(b,s) = μ(b,s) + ε,   ε ~ (mean 0, var σ²),  各 (b,s) 独立
设定: 噪声为"来源一"——纯工艺随机抖动，无偏、独立、估计时不可知、做完才揭晓。
      μ 与 σ 均已知（作为工艺参数），二者都进入 agent 状态特征。
```
> 此设定排除了"系统性差异（隐藏变量）"与"长尾故障"——前者应补状态特征而非当噪声，后者需单独的故障建模；本方案在纯噪声假设下成立，若实测数据呈多峰/长尾需另行处理。

**2.4.2 对完成时间的传导。** 下层估时器（见 1.5）输出的不再是单一 `finish_time`，而是其**分布刻画** `(μ_finish, σ_finish)`：

```text
makespan 沿关键路径累加各阶段噪声 → 方差按路径长度增长（σ_finish ≈ sqrt(Σ σ_s²)）
但独立噪声部分抵消 → 变异系数(σ/μ)小于单段
注意 max 操作的偏置: makespan 常为"多并行批次取最晚完成"，E[max] > max(E[·])
  ⇒ 仅用各阶段均值算 makespan 会系统性低估完成时间；
     需对 μ_finish 做上偏修正，或对噪声做蒙特卡洛采样估期望。
```

**2.4.3 Q-time 改写为机会约束。** 因噪声无偏独立，`finish_time` 近似正态（CLT），故"违规概率"有闭式：

```text
P(违规) = P(finish_time > deadline) = Φ( (μ_finish − deadline) / σ_finish )
机会约束: 屏蔽 P(违规) > ε 的动作
  ⇔ 屏蔽满足  deadline − μ_finish < z_ε · σ_finish  的动作
其中 z_ε 为容忍违规概率 ε 对应的分位数（如 ε=2% → z_ε≈2.05）
```

**2.4.4 safety margin 获得概率含义（顺带成为一个理论贡献）。** 由 2.4.3，原先拍脑袋的固定 `qtime_safety_margin` 被替换为：

```text
qtime_safety_margin(l) = z_ε · σ_finish(l)     # 不再是常数
```
即 margin = "容忍违规概率"×"该工件完成时间波动"。波动大的工件自动留厚垫、波动小的留薄垫——这正是**自适应安全裕度**，且有明确的概率解释。`z_ε`（或等价的 ε）成为唯一需要设定的、领域可解释的旋钮。

**2.4.5 必须诚实的结论：随机下无法绝对归零。** 正态噪声尾部无限长，无论 margin 多大，实际加工时间暴长导致违规的概率恒 > 0。因此全方案中关于 mask 的保证一律表述为**"违规概率 ≤ ε"**，而非"违规归零"。可控的是 ε（通过 z_ε / margin），不是绝对的 0。

**2.4.6 对训练与评价的要求。**

```text
训练: 环境每 step 采样实际加工时间 p_actual；agent 状态用 (μ, σ)，
      Q-time 判定与 reward 用采样出的实际值 —— 否则 agent 没见过波动，学不会留余量。
评价: Q-time 违规率在多次随机 rollout 上统计（报告期望 + 分布/分位数），而非单次确定性回放。
```

---

## 3. 约束与候选池：qtime 硬约束 + priority 强偏好

### 3.1 候选池四步流水线（顺序不可调换）

```text
第0步 结构可行集: A_t^m = {(l,p) | l 已到达且未完成, m 可加工 l, p 是 (l,m) 可用 PPID,
                              PPID 步骤完整, 所需 Chamber/Side 存在}   # 结构可行性，最先做
      ↓ ① qtime-safe mask     —— 硬约束：从结构可行集中屏蔽会断送可见 lot 的动作
      ↓ ② priority filter      —— 强偏好（在 ① 存活的 qtime-safe 集合内）
      ↓ ③ CandidateScore 排序   —— 软目标启发式打分排序
      ↓ ④ TopK + padding + 有效mask —— 排序后取前 K；不足 K 则 padding 补齐到 K，
                                       并生成 candidate_mask 标记哪些是真实/哪些是 padding
                                       → 对齐到定长 A_fixed^m
```

各步职责须分清，**顺序不可调换**：

- **第 0 步（结构可行性）必须最先**：先确定"机器 m 结构上能做哪些 (l,p)"，后续 ①②③ 才在这个集合上运算。若把结构过滤后置，前面几步会在"机器根本不能加工的候选"上做无意义计算——逻辑顺序错误。
- **① qtime mask**：在结构可行集上施加机会约束（3.2），屏蔽会断送可见 lot 的动作。
- **② priority filter**：在 ① 存活的集合内做优先级过滤（3.4）。
- **③ CandidateScore**：仅对存活候选**打分排序**，不删除动作（删除已由 ①② 完成）。
- **④ TopK + padding**：排序后取前 K 个；若存活候选 < K，padding 补齐到定长 K，并用 `candidate_mask` 区分真实候选与 padding（供 masked softmax 用）。

> `candidate_mask` 在此只标记 **padding 位 + （冗余保险用的）结构不可行位**；qtime 与 priority 已在 ①② 通过删除候选实现，不靠 mask。即喂给 SAS 的候选池**已是 qtime-safe 且 priority 过滤后**的（见 6.2.1 状态说明）。

### 3.2 qtime-safe mask（窗内可预判部分）

```text
function qtime_safe_mask(candidate_actions, visible_lots, C_t, config):
    mask = ones(len(candidate_actions))
    for i, (lot, ppid) in enumerate(candidate_actions):
        C_sim = dry_run_commit(C_t, lot, machine_m, ppid)
        for l in visible_lots:                 # 已到达 + 前瞻窗内即将到达
            if is_doomed(l, C_t, config):       # 已注定违规(连等待也救不回)的 lot
                continue                        # 不作屏蔽依据，否则会"全屏蔽→死锁"
            # 机会约束（见 2.4.3）：用完成时间分布判断违规概率
            mu, sigma = estimated_next_finish(l, C_sim)   # 见下方"等待 lot 估时口径"
            margin = config.z_eps * sigma                  # 自适应 safety margin
            if qtime_deadline(l) - mu < margin:            # ⇔ P(违规) > ε
                mask[i] = 0
                break
    return mask          # 若全部被屏蔽 → 交回上层时机层推进时间（不罚，见 4.5）
```

> **等待 lot 的估时口径（须明确，否则 `estimated_next_finish` 无定义）**：等待 lot `l` 尚未派工，其将来用哪台机器/ppid、何时开始都未定。本 mask 取**乐观估计**——在 `C_sim` 下，假设 `l` 能用它**最早可用的可行 (machine, ppid)** 立即开工，由下层估时器（1.5）算其 `(μ,σ)`。
> **该口径的正确语义（勿误读为"保守"）**：乐观估计回答的是"提交候选 i 之后，`l` 是否**还有救**"。仅当 i 使 `l` 连最乐观情况都注定违规（即 i 断了 `l` 的最后生路）时才屏蔽 i；若 `l` 仍可救，则不屏蔽，信任后续决策步骤及时派 `l`（同一 mask 在后续步骤会再次判定）。因此本 mask 屏蔽的是"会**当场断送**某等待 lot 的动作"，而非"任何可能让其变紧的动作"——这也是为何对其他 lot 只能"显著降低"违规、而非绝对保证（真实开始时刻依赖后续决策）。
> 若希望更强的保护，可改用**悲观估计**（按 `l` 较晚的可能开工时刻判定），mask 会更激进地屏蔽，但需配合 `is_doomed` 与"全屏蔽→交回时机层推进时间"防止过度限制导致空池。乐观/悲观口径二选一并在实现中固定。

两点必须说清，否则逻辑不成立：

1. **必须排除"已注定违规"的 lot。** 若某 lot 即便本步等待也无法在窗内被服务（`qtime_deadline(l) - 最早可能开始(均值) < 0`），它会让**每个**候选动作都触发屏蔽，导致候选池全空→强制等待→既排不了任何工件、那个 lot 也照样违规（死锁）。因此 `is_doomed` 的 lot 不计入屏蔽依据；它的违规已不可避免，应直接计入指标、并让调度继续推进，而不是拖垮整池。
2. **保证强度要诚实（随机版）。** 加工时间随机下不存在"违规归零"（2.4.5）。本 mask 保证的是：对**被调度 lot 自身**，违规概率 ≤ ε；对**其他可见 lot**，由于其真实开始时刻仍取决于后续决策，仅是**显著降低**可预判的违规概率。`z_ε` 越大越保守（违规概率越低、利用率越低）。

> 控制流补充：本步候选池经 mask 后为空时，视为"本次派工不可行"，控制权交回上层时机层（VC-MCTS 或规则推进时间），不计执行失败惩罚。

### 3.3 Q-time 约束的处理：PPO-Lagrangian（约束式 RL）

mask 已结构性压低违规（窗内、机会约束），但仍有残差违规：**窗外到达的工件、机会约束的尾部、以及 doomed lot**。如何在训练目标层面压住这部分残差，是本节要解决的问题。

**3.3.1 为何不用固定惩罚权重。** 最朴素的做法是把违规作为惩罚塞进奖励：`reward = 利用率 − w·违规`。但固定权重 `w` 有根本困难：`w` 太小则违规压不到目标、太大则过度保守牺牲利用率；二者量纲不同、`w` 无明确物理含义、且最优 `w` 随训练进程变化（早期策略差需大 `w`、后期需小 `w`），固定常数两头不讨好。这正是初版"#5 手调权重难收敛"的根源——把"违规率 ≤ ε"这个**有明确含义的约束**，被迫翻译成"惩罚权重 = w"这个**无明确含义的常数**去瞎调。

**3.3.2 问题的正确形式：约束优化 / CMDP。** 本项目应表述为**约束优化**问题：

```text
最大化   E[利用率]
约束于   E[Q-time 违规] ≤ ε     （ε = 可容忍的期望违规率/报废率，领域可解释）
```

这在 RL 中对应 **CMDP（Constrained Markov Decision Process，约束马尔可夫决策过程）**：一个 reward（利用率）待最大化，一个 cost（违规）待约束在阈值内——与本问题字面对应。

**3.3.3 PPO-Lagrangian 解法（让权重自己学）。** 引入**可学习的拉格朗日乘子** λ≥0，将约束并入目标：

```text
max_π min_{λ≥0}  E[利用率] − λ·( E[违规] − ε )

实现（与 4.3/4.5 的 V_qtime 残差通道一致；三通道，无 progress）：
  策略优势:  A = w_exec·norm(Â_exec) + w_util·norm(Â_util) + λ·norm(Â_qtime)
                # r_qtime 本身为负(cost)，故统一加法；λ 即 qtime 通道的自适应权重
  对偶上升:  λ ← max(0, λ + η_λ·( Ê[违规] − ε ))
```

λ 不再手设，而由**对偶上升**自动调节：违规率 > ε 时 λ 自动调大（加重惩罚、逼策略保守），违规率 < ε 时 λ 自动调小（放松、争取利用率），最终自动停在"违规率 ≈ ε"的约束边界。你只需设含义明确的 ε，不需猜无含义的 w。

**3.3.4 与机会约束 mask 的分工（两道防线）。** 二者互补，对应违规来源两分（2.2）：

```text
机会约束 mask（3.2）：决策时结构性挡掉"窗内可预判"的违规（硬挡）
PPO-Lagrangian（本节）：训练目标层面压住 mask 挡不住的残差——窗外到达、随机尾部（软压）
```

**3.3.5 理论依据。** 该方法有完整的数学与文献根基，非工程临时技巧：

```text
优化根基: 拉格朗日对偶与 KKT 条件——约束优化转鞍点求解，对偶上升为标准数值方法
          （见凸优化经典教材，如 Boyd & Vandenberghe）
RL 形式化: CMDP 框架与拉格朗日求解的合理性、表格情形收敛性 —— Altman (1999)
深度实现: CPO (Achiam et al. 2017)；PPO-Lagrangian / TRPO-Lagrangian 作为
          Safe RL 基线 (Ray, Achiam, Amodei 2019, OpenAI Safety Gym)；
          λ 更新稳定性改进 —— PID-Lagrangian (Stooke et al. 2020)
所属子领域: Safe RL（安全强化学习），活跃方向，有成熟开源库与 benchmark
```

> **诚实的边界（答辩须拿捏）**：(1) 严格收敛保证主要在凸/表格情形成立，深度神经网络 + 非凸 + 函数近似下保证打折，实践中 λ 通常收敛到约束边界附近但非带证明的全局最优（深度 RL 通病，非本方法独有）；(2) 它保证的是**期望约束** `E[违规] ≤ ε`，而非"每次都不违规"——这与 2.4.5"随机下无法绝对归零、只能违规概率 ≤ ε"的结论**完全自洽**；(3) λ 可能震荡，`η_λ` 须远小于策略学习率，必要时用 PID-Lagrangian 稳定。
> **实施建议**：基础版可先用固定 `w_qtime`（大值）跑通，验证流程后再切换为 PPO-Lagrangian 的自适应 λ（`use_qtime_lagrangian`）；二者作用于同一 `V_qtime` 残差通道，区别仅在权重是常数还是自适应，**不并用**。

### 3.4 priority 候选池过滤（强偏好）

```text
function priority_filter(safe_actions, config):
    if len(safe_actions) == 0: return safe_actions
    max_pri = max(priority(a.lot) for a in safe_actions)
    if config.priority_filter_mode == "strict":
        return [a for a in safe_actions
                if priority(a.lot) >= max_pri - config.priority_min_gap]
    else:  # "soft"：不删动作，仅在 CandidateScore 中大幅加权，保留探索
        return safe_actions
```

建议先 `soft`（保留探索，避免策略坍缩），稳定后视需要切 `strict`。

---

## 4. 基础方案：候选池资源日历 SAS-PPO

SAS-PPO 不一步到位。基础版只学"当前 Machine 选哪个 (Lot, PPID)"，由规则触发；"是否等待/预留"由上层 VC-MCTS 搜索处理，不在 SAS 学习范围内。

### 4.1 候选动作池生成

见 3.1。`CandidateScore` 中**移除 priority 项**（已上移到候选池过滤）：

```text
CandidateScore(l,m,p) = due_date_urgency(l) + qtime_slack(l) + waiting_time(l)
                        - estimated_process_time_mean(l,m,p) - resource_conflict_risk(l,m,p)
```

`K_action` 训练初期取 30–50，中等 50–100，大规模 100–200。**异型柔性下注意**：一个工件可有多个 (machine, ppid) 组合，候选池规模比同型时更大，K_action 需相应上调或加强 Top-K 打分筛选；同时下层估时器对每个 (工件,机器,ppid) 组合都要调用一次，调用频次上升——这进一步要求下层保持快速（规则/贪心，不可用群优化等重型求解，否则候选池一大即拖垮训练）。

### 4.2 可变动作维度处理（保留）

不构造固定全局动作空间，采用**候选池索引动作空间**：Actor 输出 `action_index ∈ {0,...,K_action-1}`，环境映射 `candidate_actions[action_index] = (lot, ppid)`。不同时刻候选数经 Top-K / padding / mask 对齐到 `K_action`，Actor 输出固定维 logits，用 masked softmax；候选池为空或全被 mask 时不执行 softmax，改由上层时机层推进时间，避免分母为 0。

### 4.3 SAS Actor-Critic（多头 Critic）

Actor 先用 MLP，再升级为机台条件化注意力（候选动作编码 `x_i = [lot_feat, ppid_feat, machine_feat, calendar_feat]`，机台为 query 做多头注意力，masked softmax 输出）。

**Critic 改为多头**，共享候选 embedding 与注意力表示：

```text
g_t = Pooling(h_m, e_1,...,e_K, global_features)
V_exec(z_t)     = MLP_exec(g_t)        # 即时执行（密集）：合法性 + packing 利用率质量
V_qtime(z_t)    = MLP_qtime(g_t)       # 阶段间 Q-time 违规残差（密集，逐步）
V_util(z_t)     = MLP_util(g_t)        # 唯一软目标：利用率（终局）
```
> 通道说明（三通道，与代码一致 `MULTIHEAD_CHANNELS=(exec,qtime,util)`）：**已删除 progress 通道**——它度量 `completed/num_lots`，在能完成全部 lot 的实例上恒为 1.0、是无梯度的死重。也**没有独立的 V_tardy 头**——拖期不进奖励（仅作评估指标，见 §4.10）。多头在此并非为"多目标权衡"，而是为分离尺度与时间尺度迥异的信号：执行（密集即时，含 packing）、Q-time 残差（密集逐步，报废相关）、利用率（终局软目标）。

### 4.4 资源日历执行与失败处理（保留）

`TryInsert(lot, machine, ppid, C_t)` 校验 Machine 可插入性、PPID 步骤完整性、Chamber/Side 可用时间、插入后时间表冲突。

**训练阶段轨迹一致性（关键）**：若插入失败，**不得**把执行动作替换为候选池中下一个动作——PPO 需要"采样动作与其 logπ"一致，替换会使概率比与梯度估计偏离真实策略。处理方式：保留原采样动作、返回失败惩罚、不提交任何日历修改、把环境推进到明确的下一状态。

推理阶段无梯度更新，可工程 fallback：按概率从高到低尝试，提交第一个可插入动作；全失败则 wait/no-op。

### 4.5 奖励通道设计

即时 reward **只回答**"这一步动作合不合法、可不可执行"。质量/约束信号下沉：

**通道 EXEC（即时、密集，含 packing 利用率质量）**
```text
r_exec = +0.20 + w_pack·packing   if insertion_success   # packing = total_work / (lot_end-lot_start)
         -0.40                      if insertion_failed
         -0.50                      if mask_invalid        # 防御性分支，正常不应触发，见下
          0.00                      otherwise
```
> **packing（利用率向的逐步边际质量）**：`total_work`=该 lot 所有(子批×阶段)加工时间之和，`lot_end−lot_start`=它占用机台的跨度。比值越高=资源排得越紧凑、内部空闲越少（撞腔体争用会拉长跨度→packing 变小）。这把原先**恒定 +0.20** 的执行奖励变成**逐步可区分好坏**的信号——agent 每步就能感知"这个候选选得紧不紧凑"，而不必等终局。`w_pack` 默认 0.10。

> 关于 mask_invalid：在 masked softmax 正确实现下（mask=0 的位置概率恒为 0），训练时**采样不到** padding / 无效动作，此分支理论上不可达。因此 -0.50 仅作为**防御性断言**（捕捉实现 bug、数值下溢导致的异常采样），不应被当作"让 Actor 学会避开无效动作"的训练信号——避开无效动作由 mask 本身保证，不靠惩罚。

**wait 的处理（SAS 阶段）**
```text
SAS 是规则触发（池非空才激活），且在完整系统中"是否等待/预留"由 VC-MCTS 上层时机层决策，
故 SAS 自身从不主动选择 wait。SAS 唯一会遇到的 wait 是：
  候选池为空 / 经 qtime mask 后全被屏蔽 → 由环境/上层时机层接管推进时间，r_exec = 0.0（不罚）
"主动等待/预留"的时间成本属于 VC-MCTS 预留规划器的记账（ledger idle cost），不在 SAS 奖励内。
```
> 修正：原先在 SAS 列出"主动 wait -0.02"与职责划分矛盾——SAS 不拥有 wait 动作。主动等待成本统一归 VC-MCTS 预留规划器的 ledger idle 项。

**约束残差（密集逐步）+ 软目标（终局）**
```text
r_qtime = -(new_qtime_violation_t / num_lots)   每步       # 阶段间 Q-time 违规残差，逐步密集
r_util  = +normalize(avg_machine_utilization)   仅终局      # 唯一软目标
```
> **qtime 通道改为逐步密集（关键）**：`new_qtime_violation_t`=本步这次派工**新造成**的 Q-time 违反增量（`q_after−q_before`）。因 Σ_t new_qtime_violation = 终局总违反，逐步与终局**总惩罚等价（telescoping）、不双重计数**，但 credit 大幅改善——agent 直接知道**是哪一次派工**害了 Q-time，而非把信号挤在终局靠 50 步 bootstrap 回传。
> 奖励侧**没有独立的 r_tardy 通道**——拖期不进奖励，仅作评估指标（见 §4.10）；`r_qtime` 度量**阶段间 Q-time**（`q_time_limits`）的违规数。
> **已删除 r_progress 通道**：原 `completed/num_lots` 在能完成全部 lot 的实例上恒为 1.0，是无梯度死重。
> 当前三通道里 **exec 与 qtime 是逐步密集、util 是终局**——只剩 util 一个终局通道，缓解了"信号全挤终局"的稀疏问题。
> 若终局过稀疏需补 shaping，**必须用 potential-based 形式** `F = γΦ(s') − Φ(s)`，保证不改变最优策略。

### 4.6 RewardConfig 与 compute_sas_reward（向量）

```text
function compute_sas_reward_vector(info, config) -> vector:
    r = zeros([EXEC, QTIME, UTIL])    # 三通道；无 PROGRESS（死重）、无独立 TARDY（拖期不进奖励）
    # --- EXEC（密集，含 packing 利用率质量）---
    if info.mask_invalid:    r[EXEC] = config.mask_invalid_penalty;  return r   # 防御性，正常不可达
    if info.wait_or_noop:    r[EXEC] = 0.0;  return r   # SAS 的 wait 只来自空池/全屏蔽，不罚
    if info.insertion_failed: r[EXEC] = config.insert_fail_penalty;  return r
    if info.insertion_success:
        packing = info.total_work / max(info.lot_span, eps)
        r[EXEC] = config.insert_success_reward + config.w_pack * packing
    # --- QTIME（密集逐步）：每步罚本次派工新造成的违反，非仅终局 ---
    r[QTIME] = -(info.new_qtime_violation / info.num_lots)
    # --- UTIL（终局）---
    if info.is_terminal:
        r[UTIL] = +normalize(info.avg_machine_utilization)               # 唯一软目标
    return r        # 向量；不跨通道求和、不跨通道 clip

RewardVectorConfig:
  insert_success_reward=+0.20  insert_fail_penalty=-0.40
  mask_invalid_penalty=-0.50   # 仅防御性断言，正常不触发
  w_pack=0.10                  # exec 通道 packing(利用率向)信号强度
  w_exec=1.0  w_qtime=3.0（大，硬约束残差）  w_util=0.5   # 作用在归一化 advantage 上；无 w_progress
  qtime_mask_enabled=True  z_eps=...(违规概率ε对应分位数, 如 ε=2%→2.05)  W_lookahead=...
  # safety margin 不再是常数，由 z_eps·σ_finish 自适应得到（见 2.4.4）
  use_qtime_lagrangian=False  qtime_lambda_init=0.0  qtime_cost_budget=0.0
  priority_filter_mode="soft"  priority_min_gap=...
  # 无 w_tardy（拖期不进奖励，仅评估指标）；SAS 无 wait_penalty（主动等待成本归 VC-MCTS ledger）
```
> 约束残差的两种权重方式（二选一，不并用，避免重复计数）：(i) **固定权重** `w_qtime`（大值，体现硬约束优先）；(ii) **自适应拉格朗日** `λ_qtime`（对偶上升，把"期望违规 ≤ budget"作为约束，见 3.3）。后者更原则化，对应 `use_qtime_lagrangian=True`，此时 `w_qtime` 由 `λ_qtime` 取代。

### 4.7 多头 Critic + 逐目标 GAE + PPO loss

```text
for k in {exec, qtime, util}:        # 三通道逐通道独立 GAE（无 progress、无独立 tardy 头）
    delta_t^k = r_t^k + γ·V_k(z_{t+1})·(1-done) - V_k(z_t)
    Â_t^k     = delta_t^k + γ·λ_gae·(1-done)·Â_{t+1}^k
    R_t^k     = Â_t^k + V_k(z_t)

A_t = w_exec·norm(Â_exec) + w_util·norm(Â_util)
    + w_qtime·norm(Â_qtime)        # r_qtime 本身为负(cost)，故统一加法；
                                    # 若 use_qtime_lagrangian，则 w_qtime → 自适应 λ_qtime

L_actor = - E_t[min(ρ_t·A_t, clip(ρ_t,1-ε,1+ε)·A_t)]
L_value = Σ_k c_k·MSE(V_k(z_t), R_t^k)
L_total = L_actor + L_value - c_e·H(π)
```

> 适用范围：本节公式以 **SAS 的通道集合 {exec, qtime, util}** 为例（第 4 章为基础 SAS-PPO）。SAS 只有 {exec, qtime, util} 三个通道；VC-MCTS 预留规划器不走 PPO，无 GAE。

> 两点务必注意，否则多头会"看着干净、实则学不动"：
> 1. **终局通道信号稀疏（现已大幅缓解）。** 改造后 **exec 与 qtime 均为逐步密集**，只剩 **util 一个终局通道**只在最后一步非零、靠 bootstrap 回传。util 头可用较大 `c_k` 或更长训练；必要时加 **potential-based shaping**（`F=γΦ(s')−Φ(s)`，不改变最优策略）。这正是把 qtime 从终局改为逐步（`new_qtime_violation`）的动机——让头号约束信号变密集、credit 可定位。
> 2. **归一化在 batch 层面做。** `norm(Â_k)` 用整个 batch 的 mean/std，而非单步；exec/qtime 每步非零、util 多为 bootstrap 传播值，尺度不同，**逐通道**归一化后再加权才能让权重表达真实重要度。

### 4.8 SAS transition 字段

```text
SAS_transition = (obs_t, machine_t, candidate_features_t, candidate_mask_t,
                  action_index_t, logπ_old_t, reward_vector_t, obs_{t+1}, done_t, info_t)

info_t 至少含: selected_lot/ppid, insertion_success/failed, mask_invalid,
  wait_or_noop, selected_lot_start/end/process_time,
  visible_lots_remaining_qtime, is_terminal, 各通道 reward 分量
```

### 4.9 终止条件与失败处理（保留 + 补充）

终止：所有 Lot 完成；`t_now` 超 horizon 且无可调度 Lot；候选池空且未来无 Lot 到达 / 无 Machine 释放；`total_wait_steps > max`；连续插入失败 > `max_failed_actions`；日历不可恢复错误。

失败处理：采样到 padding/mask=0 → 正常不可达（masked softmax 保证），若发生则按实现 bug 处理，r_exec=-0.50 仅作防御；候选池空/全屏蔽 → 交回上层时机层推进时间，r_exec=0.0（不罚）；dry-run 失败 → 保留 (s,a,logπ)，r_exec=-0.40，不提交；commit 中途失败 → rollback，r_exec=-0.40；连续失败过多 → 终止，标 infeasible。

### 4.10 最终评价指标

```text
F(S) = [ q_time_violation_count, total_q_time_violation,  # [0,1] 阶段间 Q-time（硬约束，趋零、违规概率≤ε）
         tardy_count, total_tardiness,                    # [2,3] 交付期拖期（独立的量）
         priority_violation,                              # [4] 旧逆序量（对在线到达不公平，仅作参考，见 §7.5）
         -average_machine_utilization,                    # [5] 唯一软目标
         priority_weighted_wait,                          # [6] O2：优先级加权等待（priority 维度的优化/奖励主目标）
         avoidable_priority_violation ]                   # [7] O1：可避免优先级逆序（公平展示指标）
```
> 说明（与代码一致）：**Q-time（阶段间队列时间）与拖期（交付期）是独立的两组量**，`evaluate_objectives` 返回 **8 维向量**分别给出，不再合并（索引 [0..5] 与历史一致、下游按位读取；[6][7] 为 §7.5 阶段 5a 追加）。Q-time 是硬约束（趋零、违规概率≤ε），由 `q_time_limits` 度量；拖期由 due date 决定（当前压力实例下较宽松，常为 0），仅作评估指标、不进奖励。利用率取负以统一为最小化；展示时以正向报告。
> **priority 维度（§7.5 修正）：** 优化/奖励主目标取 **O2 `priority_weighted_wait` = Σ_l priority_l·max(0, start_l−arrival_l)**（到达后越晚开工、优先级越高罚越大；在线到达下良定义、天然惩罚空转）；**O1 `avoidable_priority_violation`** 作公平展示（逆序仅在"高优先级 Lot 在低优先级开工时已到达"时计）；旧 `priority_violation`（全体按开工排序的逆序总量）因对"晚到必晚开"不公平（实测 88% 不可避免），**降级为参考**。
> 评估口径：阶段间 **Q-time 违规数**（`q_time_violation_count`，由 `q_time_limits` 度量，与交付期拖期 `tardy` 是两组独立的量）作为硬约束门槛，主要看其是否被压到接近 0（及违规概率是否 ≤ ε）；`utilization` 是唯一真正可权衡的软目标。因可权衡软目标仅一项，**不做多目标帕累托/超体积的强主张**（见 7.4 的相应调整）。训练用小尺度向量 reward，评估用上述指标在多次随机 rollout 上的统计。

### 4.11 基础 SAS-PPO 算法流程

```text
Algorithm: Candidate-pool Calendar SAS-PPO (multi-head)
1. Observe s_t, extract z_t (含前瞻窗信息)
2. Select idle machine m by fixed rule
3. Build structurally feasible pool A_t^m （第0步：结构可行性，最先）
4. ① qtime_safe_mask → ② priority_filter → ③ CandidateScore 排序 → ④ TopK+pad+mask = A_fixed^m
5. If pool empty or all masked: execute wait (forced), store transition, advance time
6. SAS computes π(a|z_t, A_fixed^m); sample action_index (train) / rank (infer)
7. If padding/mask=0: r_exec=-0.50, store, continue
8. TryInsert(lot,m,ppid)
9. If success: commit, compute reward_vector
10. If fail (train): keep sampled action, r_exec=-0.40, no commit, advance
11. If fail (infer): try next candidate / wait
12. Store transition (with reward_vector). Repeat until window done
13. Per-objective GAE; update Actor + multi-head Critic with clipped PPO
```

---

## 5. 创新方案：可见性分区的约束式 MCTS 预留规划（VC-MCTS，替代学习式 RMA）

> **VC-MCTS：Visibility-aware Constrained Monte-Carlo Tree Search for Reservation Planning（可见性感知、Q-time 约束、以显式 `reserve(machine,future_lot)` 为搜索动作的在线预留规划器）。** 它取代上一版的学习式 RMA：预留决策不再靠 PPO 智能体从稀疏/事件级奖励里学，而是在每个预留机会点用一棵搜索树、以已有的资源日历 dry-run/commit 当忠实仿真器，把"现在派 vs 留给未来"两条分支**模拟比一比**。SAS 仍负责当前机器的 `(lot,ppid)` 派工，且被复用为搜索的先验与基策略。

### 5.1 为什么用搜索取代学习式 RMA

上一版把预留做成可学的 RMA，目的是修掉旧 DDT 的三处坍缩（无目标 hold、奖励重复计数、greedy 临门一脚顶不过阈值）。RMA 的目标绑定动作与事件级记账确实修掉了前两处，但**第三处是学习范式的结构性顽疾**：只要贪心推理在"高优先级到达前那一下" `P(reserve)` 顶不过 0.5，再丰富的 set-encoder 也无济于事，且 ROP 宽触发会制造大量 `no_reservation` 正确样本，类别不平衡持续把策略推向"从不预留"的角落。

VC-MCTS 换一个范式来根除这一点：**"要不要为未来高优先级 Lot 留这台机器"不再去学一个概率，而是把 `reserve(m,h)` 和 `dispatch(lot,ppid)` 都作为搜索树的边，各自用基策略 rollout 到底、按字典序目标比较，谁好选谁。** 没有梯度、没有 value-head 阈值、没有 advantage 归一化，于是 RMA 的三个失败模式（奖励坍缩、greedy under-hold、类别不平衡）在结构上都不存在。其代价是把训练期成本搬到了推理期（每个预留机会点要算若干次仿真，见 5.8 算力闸门与 5.11 时延约束）。

> 关键定位（写进 related work 用）：本方案是近期 **DyRo-MCTS（Chen et al., 2025）** 的**原理性推广**。DyRo-MCTS 因为完全看不见未来到达，只能"不采样未来 + 用一个钝的鲁棒性代理 ρ(s,a)（机器空闲分布）补偿"，并据此得出"避免早期空闲"的结论。本项目恰恰相反：在**有限前瞻窗内未来到达可见**（ETA/priority/qtime 已知），于是"定向空闲（预留）"从浪费变为最优——窗内用**显式仿真 + reserve 边**取代钝代理，窗外才退回 DyRo 式代理。**当前瞻窗宽 W=0、且无 Q-time 约束、加工时间确定时，本方案退化为 DyRo-MCTS**。这给了一个干净的"特例—推广"叙事，而非另起炉灶。

### 5.2 一棵搜索树，三个挂载点

三个机制创新不是平级模块，而是同一棵树上三个不同环节的挂载点，执行有严格先后，正好复刻 §1.4 的字典序：

```text
节点 = 状态 s（资源日历 + reservation ledger + 窗内可见未来 lot 作"确定性实体"）
边   = 动作 a ∈ { dispatch(lot,ppid) } ∪ { reserve(m,h) } ∪ { no_op }

挂载点 A —— 准入层（机制 1：Q-time 约束）       对应字典序第 0 层（硬）
   扩展时决定哪些边合法：窗内可预判违规硬剪枝 + 窗外残差用 λ 软成本（5.4）
挂载点 B —— 选择/估值层（机制 2：优先级-能力鲁棒性）对应字典序第 1 层（强）
   改写 DyRo-UCT 的利用项：价值 × 可行性 × 能力感知鲁棒性 + 显式 reserve 边（5.5）
挂载点 C —— 评估层（机制 3：双重不确定性）       贯穿全树的可信度保证
   每条 rollout 在加工噪声 + 窗内确定到达下跑，CRN 压方差，多路求统计量（5.6）
```

字典序由执行顺序天然实现：先 A 把会踩穿 Q-time 的边掐掉，再 B 在合法边里按优先级鲁棒性+价值选，利用率作为最后可让步项进入价值估计；C 保证前两者用到的每个估计量在双重不确定性下算得准。

### 5.3 统一选择公式（三机制焊在一处的约束式 DyRo-UCT）

DyRo-MCTS 的核心是把 PUCT 的利用项 q 换成 `α·q+(1−α)·ρ`。本方案目标架构在此基础上再嵌入约束准入与优先级鲁棒性，并让机制 3 提供其中每个估计量的可信均值：

```text
a* = argmax_a [ E(s,a) + c · p(s,a) · √n(s) / (1 + n(s,a)) ]

E(s,a) = feasible(s,a) · [ α·q̂(s,a) + (1−α)·ρ̂_pc(s,a) ] − λ_qtime · ĉ_qtime(s,a)

其中:
  feasible(s,a)  ∈{0,1}  机制1硬侧：窗内可预判 Q-time 违规的边置 0，连选择都进不来
  λ_qtime·ĉ      机制1软侧：窗外/残差期望违规预算的软压；λ_qtime 与 §3.3 PPO-Lagrangian 同一乘子
  q̂(s,a)         价值（O2 加权等待 + 利用率，负向归一），由 C 多路 rollout 估
  ρ̂_pc(s,a)      机制2：优先级-能力感知鲁棒性（5.5），由 C 估
  p(s,a)         先验概率，目标架构中来自训练好的 SAS（5.8）；当前代码第一阶段用 action.prior 近似
  c,α            探索常数 / 价值-鲁棒性插值（DyRo 默认 α≈0.5–0.6）
```

三机制的融合点就在这一个 `E` 上：**机制 1 决定 E 里哪些边存在、哪些项加进来（`feasible` 与 `λ·ĉ`）；机制 2 决定 E 的鲁棒性成分 `ρ̂_pc` 长什么样；机制 3 决定 E 里每个带帽估计量算得准不准。** 当前代码实现没有完整落地 `ρ̂_pc` 与 SAS 先验，而是采用 root-level UCT：`objective_to_score = -qtime_count_penalty - qtime_total_penalty - O2 + util`。最终动作也已从早期“按访问次数”改为 **objective-first 字典序选择**，并加入 no_op gating（见 §6.2.2 与 §7.6）。

### 5.4 机制 1：Q-time 约束准入（硬剪枝 + λ 软成本）

DyRo-MCTS 是纯软目标（加权拖期），**没有任何硬约束**——这正是本项目相对它的第一处实质增量。把 Q-time 机会约束嵌进树：

```text
硬侧（窗内可预判，对应字典序第 0 层）：
  扩展某节点的候选边时，对每个候选做 dry-run，逐个可见 lot 按 §2.4.3
    deadline(l) − μ_finish(l) < z_ε·σ_finish(l)   （违规概率 > ε）
  判定；任一可见 lot（已排除 is_doomed）落入风险带 → 该边 feasible=0，不进树。
  这与 §3.2 SAS 的 qtime-safe mask 同口径、同一下层估时器（§1.5），不另立一套。

软侧（窗外/残差，对应 cost-MCTS）：
  每条 rollout 末端统计期望 Q-time 违规 ĉ_qtime(s,a)（窗外到达造成、无法在当前点完全预判的部分），
  以 λ_qtime·ĉ_qtime 进入选择公式做软压。
  λ_qtime = §3.3 PPO-Lagrangian 训练 SAS 时学到的同一个对偶乘子。
```

> 统一性卖点：**同一个拉格朗日乘子既出现在离线训练 SAS 的目标里、又出现在在线 MCTS 的选择公式里**，使"硬挡（mask/剪枝）+ 软压（λ）"两道防线在训练与推理两侧口径一致，而非两套孤立工程。安全/约束式 MCTS 在 Safe RL 里有零星工作，但用在半导体 Q-time 报废约束 + 加工噪声上是空白。

### 5.5 机制 2：优先级-能力感知鲁棒性 + 显式 reserve 边

DyRo 的 ρ(s,a) 是**对所有机器一视同仁**的聚合空闲标量（避免早期空闲）。异型柔性下这太钝：真正该度量的不是"总空闲少"，而是"**保留了多少与窗内高优先级到达兼容的产能**"。

```text
ρ_pc(s,a) = Σ_class  w_priority(class) · reserved_compatible_capacity(class, [t_now, t_now+W])
  reserved_compatible_capacity(class,·): 在前瞻窗内、对某优先级类别仍可用的、
    能加工该类别工件的产能积分（按机器能力分桶，而非全机器聚合）。
  w_priority(class): 高优先级类别权重更大。
```

两种可见性区间用不同手段，这是本方案的"双区制"实质：

```text
窗内（可见，主战场）：未来 lot h 的 ETA 已知 → 作为确定性实体进入转移模型，
  reserve(m,h) 作为显式树边。要不要预留、留给谁，由 MCTS 模拟 reserve 分支与
  dispatch 分支的 O2 之差直接得出 —— 不需要任何手工鲁棒性代理，也不需要学。
  （这恰是 DyRo 因零可见性做不到、本项目能做的事；reserve 分支的 O2 增益就是
    上一版 RMA 想用 r_cf 近似的反事实收益，现在被 rollout 精确算出。）
窗外（不可见，尾部兜底）：真正不可预测的到达 → 保留 ρ_pc 这一鲁棒性项，
  鼓励"为高优先级类别保留兼容产能"，但不绑定具体目标。
```

reserve 边与 ρ_pc 语义自洽：预留一台兼容高优先级的机器会主动拉高 ρ_pc，于是搜索会"发现"预留是提升优先级鲁棒性的手段，而无需手写预留规则。

### 5.6 机制 3：双重不确定性下的可信估值

DyRo 只有**到达**一种不确定性，加工时间确定。本项目多一层**加工噪声 (μ,σ)**（§2.4），意味着 rollout 本身随机、q̂/ρ̂/ĉ 都是随机量：

```text
每条 rollout：
  下层 schedule_on_calendar 传 noise_rng，按 μ+N(0,σ) 采样各 (sub_batch,stage)（§2.4.6）；
  窗内可见 lot 按真实 ETA 进仿真，窗外到达一律不采样（守住 §2.1 黑盒假设）。
CRN（公共随机数，方差缩减，成败关键）：
  同一节点下所有候选边复用同一组 N_mc 个噪声种子，公共随机性在比较时相减抵消，
  你比的是动作本身差异而非噪声 → N_mc 取小（3–8）即可给出稳定排序。
估计量：
  q̂/ρ̂_pc/ĉ_qtime 均为 N_mc 条 rollout 的均值；用决策翻转率（§1.6.5 同口径）
  而非回归误差验收 N_mc 是否够。
```

> 诚实边界：窗内"只模拟可见 lot"不是退而求其次——预留价值恰恰来自窗内那个可见的高优先级 lot，所以它已完整捕捉预留收益；窗外是谁也算不准的部分，本就该交给 ρ_pc 与残差兜底。这条故事干净、可写进报告。

### 5.7 rollout 内部与树策略同源（命门）

最易踩、却决定估计是否失真的一点：rollout（叶子往下用基策略续跑到底）**必须和树策略用同一套三机制规则**，否则你在用一个"无视 Q-time、无视 ledger 的钝策略"模拟未来，q̂/ρ̂ 系统性失真。

```text
rollout 内部基策略（当前用 FIFO 等复合规则；目标增强可换成训练好的 SAS）必须：
  (1) 也走 qtime-safe 候选池（机制1硬挡在仿真里照样生效，否则 q̂ 乐观偏高）；
  (2) 尊重 reservation ledger：已 reserve 的机器不被基策略抢去派别的 lot，
      等目标 h 到达优先上 —— 否则 reserve 分支与 dispatch 分支的 q̂ 无差异，
      预留收益被仿真抹平（这正是要算的反事实）；
  (3) 注入加工噪声且复用 CRN 种子（机制3）。
```

### 5.8 ROP 作为算力闸门；SAS 作先验与基策略（目标增强；当前第一阶段用规则替代）

三机制叠加会推高单次决策成本，用现成的**宽触发 ROP + TopB**（§5.1.1 仍沿用）当统一闸门，让昂贵搜索只在可能有预留杠杆处发生：

```text
ROP 未命中的决策点 → 目标架构中直接用 SAS 先验贪心派工；当前代码由规则/候选池派工逻辑接管。
ROP 命中           → 才建树，reserve 边只取 S_res 排序 TopB 对，dispatch 边只取候选池 TopK。
```

DyRo-MCTS 不是抛弃学习——它用离线策略当 PUCT 先验 p(s,a) 并引导 rollout，且实证"先验质量越高、搜索结果越好"。据此给上一版做不稳的 RL 工作一个体面去处：

```text
SAS（待接入）    → 复用为：MCTS 的先验 p(s,a) + rollout 基策略 + 叶子估值（多头 Critic V_k，§4.3/4.7，
                  替代"rollout 跑到底"，AlphaZero 式截断，省时延）。
当前第一阶段     → 使用 `ResourceCalendarEnv.build_candidate_pool(machine)` 的 TopK dispatch 边与 FIFO 等规则 rollout，
                  已先验证 reservation ledger / ROP / search / trace / no_op gating / workers 流程。
RMA（学不稳）    → 降级为"预留候选提议器"：不自己拍板，只向树提供值得搜的 reserve(m,h) 分支
                  （即 5.1.1 的 ROP+S_res TopB），由 MCTS 验证值不值。
（可选）专家迭代  → 拿 MCTS 改进后的决策当标签回去重训先验 p(s,a)（方法非原创，不作卖点）。
```

### 5.9 主循环伪代码（VC-MCTS）

```text
def vc_mcts_decide(env, decision_point, sas_prior, N_iter, N_mc):
    rop_pairs = env.detect_rop(decision_point)            # §5.1.1 宽触发 + S_res
    if not rop_pairs:
        return sas_prior.greedy_action(env, decision_point)  # 无杠杆，免费走先验
    root = Node(env.snapshot())                           # 深拷贝 state+ledger（非破坏，扩 §dry_run）
    for _ in range(N_iter):
        node, path = select(root)                         # 用 5.3 的约束式 DyRo-UCT 下行
        child = expand(node)                              # 仅 feasible=1 的边（机制1硬侧）
        q, rho, c_qt = evaluate(child, sas_prior, N_mc)   # 机制3：CRN 多路噪声 rollout，rollout 内同源(5.7)
        backprop(path, q, rho, c_qt)                      # 更新 n,q̂,ρ̂,ĉ
    return objective_first_choice(root.edges)              # 当前代码：qtime_count→qtime_total→O2→util→visits，并带 no_op gating
# 选中 reserve(m,h) → 登记 ledger；dispatch → 交 SAS commit；no_op → 机器留到下一事件
```

### 5.10 可解释性输出（强于学习式 RMA）

搜索天然可解释：对被选动作，可导出根节点各边的访问次数与 q̂/ρ̂/ĉ 分解（"为何 reserve(m,h) 胜过现在派 X：模拟 O2 低多少、Q-time 风险多少、占用了哪台兼容机器的产能"）、reserve 分支与 dispatch 分支的反事实 O2 差、预留命中/超时/浪费的事件记账（ledger 同 5.1.1）。这比学习式 RMA 的 logits 解释更直接，因为每个数都是模拟出来的、可复算。

### 5.11 失败/退化与 oracle 上界（go/no-go，一步不能省）

```text
ROP 常空            → 实例缺预留杠杆，先做下面的 oracle 验证，别堆搜索深度；
单次决策时延过高     → ① 用多头 Critic 当叶子估值替代跑到底；② 跨决策点复用子树（DyRo 已用）；
                      ③ 收紧 ROP/TopB；④ 调小 N_iter/N_mc（CRN 下小 N_mc 已够）；
估计方差大、翻转多   → 加大 N_mc 或加 CRN，按决策翻转率（§1.6.5）校准；
reserve 分支总不胜出 → 检查 oracle 是否也不胜出；若 oracle 无收益则是实例无杠杆，非搜索问题。
```

> **实现优先级（与上一版同一纪律，但 oracle 在 MCTS 框架下更自然）**：先证明预留有上界收益，再投入搜索调参。**oracle reservation 上界 = 信息完整（含窗外真实到达）+ 大 N_iter 的 VC-MCTS**——也就是把同一套搜索器放开信息与预算跑离线，得到 O2 下界。在 `late_hi`（高优先级晚到、corr≈0.97）上确认该下界**既显著低于 SAS-only、又低于规则预留地板（§7.5 的 O2≈862）**，是开工 go/no-go 的唯一闸门；红灯就回去改实例生成器（§9.8），而不是加搜索深度或调权重。这样"同一套仿真器既在线决策、又离线自证杠杆"形成闭环。

---

## 6. 实现路线与配置规格

### 6.1 分阶段实现路线

```text
阶段 1：资源日历环境 + 下层启发式估时器（1.5）+ 候选池 + 结构 mask（不训练，先把环境与估时做稳）
阶段 2：规则触发 SAS-PPO，仅即时执行通道（标量、单头）—— 验证闭环
阶段 3：加 机会约束 qtime mask + priority filter(soft) —— 验证 qtime 违规率 ≤ ε、总违规大幅下降（机制，无 reward 改动）
阶段 4：加终局向量 + 多头 Critic + 逐目标 GAE —— 验证软目标可学、权重可调；若残差违规明显，将 qtime 残差通道权重改为自适应 λ（3.3）
阶段 5：单注意力 → 双注意力 SAS（消融对比）—— 至此 SAS 作为成熟可用的派工策略冻结
阶段 6（go/no-go）：oracle 预留上界验证 —— 信息完整 + 大预算的 VC-MCTS 离线跑 late_hi，确认 O2 下界
  显著低于 SAS-only 且低于规则地板（§7.5 的 862）；红灯则改实例生成器（§9.8），不进阶段 7
阶段 7：VC-MCTS 在线预留规划（§5）—— SAS 作先验/基策略/叶子估值；先只开机制 1（Q-time 约束准入）跑通，
  再依次加机制 3（噪声 rollout + CRN）、机制 2（优先级-能力鲁棒性 + reserve 边），每步单独消融
阶段 8：small/medium/large 实例 + 公开基准评估（VC-MCTS vs SAS-only vs 规则 vs DyRo 式无约束 MCTS）
```

每阶段先稳定再进下一阶段；振荡时回退上一阶段，而非堆搜索深度或 reward 分量。**SAS 仍走 RL 训练（阶段 2–5）；预留不再训练，改为阶段 6–7 的搜索（§5.1 已述理由）。**

### 6.2 配置规格（SAS 策略 与 VC-MCTS 预留规划器）

> 本节是可直接对照实现的配置规格。贯穿原则：**SAS（学习式）回答「当前机器派哪个」（以当前机台 m + 已到达候选池为中心）；VC-MCTS 预留规划器（搜索式，非学习）回答「是否为未来高优先级 Lot 预留哪台机器」（以前瞻窗 + 空闲机器集合 + 预留候选对为中心，§5）**。两者职责不混，配置才干净。预留规划器只冻结被 reserve 的机器；其余机器仍由 SAS 正常派工。SAS 同时被规划器复用为先验、基策略与叶子估值（§5.8）。

#### 6.2.1 SAS 配置

**状态（三层）**

```text
候选动作级 x_i（每个候选 (lot,ppid) 一行，共 K_action 行）:
  Lot:   due_date_urgency=(due-t_now)/H, remaining_qtime=qtime_deadline-预计开始,
         priority, waiting_time=(t_now-arrival)/H, wafer_count
  PPID:  step_count, process_time_mean(μ)/std(σ), required_chamber_side_count
  匹配:  estimated_earliest_start, finish_time_mu(μ_finish), finish_time_sigma(σ_finish),
         qtime_violation_prob=Φ((μ_finish−deadline)/σ_finish), insertion_slack,
         chamber_conflict_risk, resource_conflict_risk
机器级 machine_m_features（单行，作注意力 query）:
  next_available_time, short_window_util, scheduled_lot_count
全局级 global_features（单行，喂 Critic）:
  t_now归一化, 未完成lot占比, M_idle数, 平均利用率,
  最大qtime风险, 最高due紧迫度, 有效候选数/K_action

张量形状:
  candidate_features:[K_action, d_cand]  candidate_mask:[K_action]
  machine_features:[d_machine]           global_features:[d_global]
```
> `remaining_qtime` 与加工时间的 (μ,σ) 必须进特征（σ 让 Actor 学会对高波动工序更保守，见 2.4）。喂进 SAS 的候选池**已是 qtime-safe**（候选池生成阶段已按机会约束 mask），故 `candidate_mask` 只管结构不可行 + padding，不重复做 qtime。`qtime_violation_prob` 特征**与 mask 不冗余**：mask 只做二元裁剪（≤ε 留、>ε 删），而该特征给存活候选传递"在合法边界内有多紧张"（如 0.5% vs 1.9% 都合法但风险不同），让 SAS 在 qtime-safe 候选中也能偏好更安全的。归一化方式与各维度数需按真实数据定。
> **当前实现与上述目标规格的差异（以代码为准）**：上面是 SAS 状态的**目标设计**（含若干尚未落地的特征）；**已实现的 SAS** 观测见 `rl_environment.py` 与 `phase2_sas_observation.py`，为：候选级 **18 维** `feature_names`（`is_real, is_wait, score, arrival_time, waiting_time, machine_slot_start, machine_load, total_process_time, predicted_completion, stage_count, qtime_risk, wafer_count, priority, due_slack, priority_rank_norm, due_slack_rank_norm, is_best_priority, is_most_urgent_due`——以 `qtime_risk` 作为 Q-time 风险代理，尚未单列 `finish_time_mu/σ` 与 `qtime_violation_prob`）；全局级 **9 维**（`lookahead=True` 时 13 维，含前瞻摘要）；机器级特征作注意力 query。后续补全 (μ_finish,σ_finish)/`qtime_violation_prob` 等特征时，向本目标规格对齐。

**动作**

```text
action_index ∈ {0,...,K_action-1}  → candidate_actions[idx]=(lot,ppid) → execute(lot,m,ppid)
logits∈R^{K_action} → masked softmax（仅 candidate_mask=1 上归一化）
训练采样 / 推理 argmax 或按概率 fallback；多机同时空闲逐台决策；池空或全 mask 不跑 softmax，交回上层时机层
```

**奖励（向量：EXEC + QTIME + UTIL 三通道；无 progress、无独立 tardy）**

```text
EXEC（即时密集）: +0.20 + w_pack·packing 成功 / -0.40插入失败 / -0.50 mask_invalid(防御性)
  packing = total_work/(lot_end-lot_start)：利用率向的逐步边际质量(撞争用→跨度长→packing 小)
  wait: SAS 不主动 wait；空池/全屏蔽时 r_exec=0.0（主动等待成本归 VC-MCTS ledger）
QTIME（即时密集）: r_qtime=-(new_qtime_violation_t/num_lots) 每步  [Σ_t=终局总违反, telescoping]
UTIL（终局）: r_util=+norm(avg_util)[唯一软目标]
收口: 各通道独立不求和不clip；util 终局 normalize 用 running mean/std；
  已删 progress 通道(恒为1.0死重)；无独立 tardy 通道（拖期不进奖励，仅评估指标）
Critic: 多头 V_exec/V_qtime/V_util，逐目标 GAE，
  A=w_exec·norm(Â_exec)+w_util·norm(Â_util)+w_qtime·norm(Â_qtime)  (r_qtime 为负 cost,统一加法)
  (w_qtime 可由自适应 λ_qtime 取代，见 3.3)
```

#### 6.2.2 VC-MCTS 预留规划器配置（搜索式，非学习）

**触发器：宽触发式 ROP 检测（沿用，作为搜索算力闸门）**

```text
输入: 当前资源日历 C_t、空闲机器集合 M_idle、前瞻窗未来 lot 集合 F_lookahead、当前 SAS 候选池摘要
输出: 是否建树（触发搜索）；候选预留对 P_t；软机会分数 S_res（仅排序，不决策）

硬触发条件（必须满足，保证 reserve 边有意义且不违法）:
  1. 存在空闲机器 m，且 m 未被其他 reservation 冻结；
  2. 存在前瞻窗内尚未到达但已可见的 future lot h；
  3. compatible(m,h)=1；
  4. not is_doomed(h)；
  5. reserve(m,h) 不会使其他已可见 Lot 直接违反 Q-time 机会约束。

软评分因素（不作硬删除，只进入 pair 特征与 Top-B 排序）:
  priority_gap(h), ETA(h)-t_now, qtime_slack_after_ETA(h),
  compatible_machine_count(h), machine_scarcity(h),
  overlap(end_current_candidate_on_m, ETA(h)),
  saved_priority_wait_estimate(h), lost_priority_wait_estimate(current_lots), idle_cost_until_eta(h).

触发逻辑:
  P_raw = 所有满足硬触发条件的 (m,h)
  若 P_raw 为空: 不建树，直接让 SAS 先验贪心派工（绝大多数决策点走这里，近乎免费）
  若 P_raw 非空: P_t = TopB_by_S_res(P_raw)，建树搜索
```

**搜索树定义（节点/边/转移）**

```text
节点 state: 资源日历 C + reservation ledger + 窗内可见 future lot（作确定性实体，ETA 已知）
根候选边:   [no_op] ∪ { dispatch(lot,ppid) | TopK 先验 } ∪ { reserve(m,h) | (m,h)∈P_t }
转移（可见性分区，§5.5/§5.6）:
  窗内未来 lot：按真实 ETA 进入仿真（确定性）；
  加工时间：noise_rng 采样 μ+N(0,σ)（§2.4.6），CRN 同节点共种子；
  窗外到达：一律不采样（守 §2.1 黑盒），其影响由机制 2 的 ρ_pc 兜底。
```

**选择 / 估值（当前代码实现：root-level UCT + objective-first final choice）**

```text
root action:
  no_op
  TopK dispatch(lot,ppid)       # 来自 ResourceCalendarEnv.build_candidate_pool(machine)
  TopB reserve(machine,future_lot)

selection during search:
  UCT(edge) = objective_to_score(mean_objective, config)
              + exploration_c * action.prior * sqrt(log(total_visits) / visits)
  objective_to_score =
      -qtime_penalty       * qtime_violation_count
      -qtime_total_penalty * qtime_violation_total
      -priority_weighted_wait
      +util_weight * avg_utilization

warm-up:
  iteration_count = max(len(root_edges), n_iter)
  # 即使 n_iter 很小，也保证每条 root edge 至少被 rollout 一次，避免 reserve 边未访问。

final choice:
  按以下字典序选择 edge：
  1. qtime_violation_count 更少
  2. qtime_violation_total 更少
  3. priority_weighted_wait 更少
  4. avg_utilization 更高
  5. visits 更多

no_op gating:
  若最终候选为 no_op，且存在 dispatch/reserve 替代边，则 no_op 必须在
  qtime_violation_count 或 qtime_violation_total 上严格优于最佳非 no_op 边；
  否则 no_op 降级为最佳 dispatch/reserve 边。
```

> 说明：上面是当前 `vc_mcts_planner.py` 的实现口径，已经从早期草案的“最终按 visits 定动作”修正为 **objective-first**。这一修正是必要的：在 `late_hi` 完整轻预算验证中，未加 no_op gating 时 120 次决策里 no_op 达 89 次，episode 只完成 31/50；加入 no_op gating 后 no_op 降为 1 次，episode 完整完成 50/50。

**评估（机制 3：rollout，须与树策略同源，§5.7）**

```text
叶子估值二选一:
  (a) 当前实现: 用规则基策略（FIFO/SPT/EDD/CR/ATC 等）续跑至 max_steps 或 rollout_max_steps，
      回报 = schedule_metrics_with_priority_wait 给出的 qtime_count/qtime_total/O2/util。
  (b) 目标增强: 后续可用训练好的 SAS policy 作为 rollout policy，并用多头 Critic V_k
      做截断叶子估值（省时延，AlphaZero 式）。
rollout 内部必须（命门）:
  ① 走 qtime-safe 候选池（机制1硬挡在仿真里照样生效）；
  ② 尊重 ledger：已 reserve 的机器不被基策略抢派，等 h 到优先上（否则反事实被抹平）；
  ③ 注入噪声 + 复用 CRN 种子。
```

**预算与超参**

```text
n_iter             : 每决策点搜索迭代数；当前轻预算验证用 2，诊断时可取 1/4
rollout_max_steps  : 单条 rollout 截断步数；当前 late_hi 轻预算用 60
exploration_c      : UCT 探索常数；当前默认 1.5
qtime_penalty      : qtime_violation_count 的分数惩罚；当前默认 10000
qtime_total_penalty: qtime_violation_total 的分数惩罚；当前默认 1000
TopB/TopK          : reserve 候选 / dispatch 边的截断宽度；当前 late_hi 轻预算用 TopB=3, TopK=2
max_decisions      : 诊断/轻预算上限；完整轻预算用 120
workers            : vc_mcts_probe.py 支持按 seed 并行；多 worker 时每个 seed 独立写 trace/summary
```

**预留登记与记账（ledger，沿用，供解释与评估）**

```text
ledger 字段:
  reservation_id, machine_id, target_lot_id, start_time, target_eta,
  expire_time=min(ETA(h)+Δ, qtime_deadline(h)-safety_margin),
  status∈{pending,hit,miss,waste,released}, idle_time, saved_wait_est, actual_saved_wait, blocked_lots
说明: 这里的 hit/miss/waste 不再用于回填奖励（无训练），而仅用于在线执行的预留兑现/释放逻辑与 §5.10 解释/评估。
```

> 规划器核心评价同上一版：`reservation_hit_rate`、`reservation_waste_time`、`saved_priority_wait`、`O2 priority-weighted wait` 是否优于 SAS-only。若只降低旧 priority_violation 却显著拉高 O2 或 idle，视为失败。差别在于：现在这些都是搜索可解释、可复算的量（§5.10），而非靠梯度学出。

#### 6.2.3 分阶段开关（配置层面）

```text
阶段 0（go/no-go）：Oracle 预留上界验证
  oracle = 信息完整（含窗外真实到达）+ 大 N_iter 的 VC-MCTS，离线在 late_hi 上跑出 O2 下界。
  确认该下界显著优于 SAS-only 且低于规则地板（§7.5 的 862）；若 oracle≈SAS-only，
  说明实例没有预留杠杆，先改实例生成器（§9.8），不进后续阶段。

阶段 1：ROP + ledger + 在线仿真器跑通（无搜索）
  实现 ROP 检测、TopB pair、reservation ledger、可从中途状态续跑的 simulate_to_end；
  策略先用基规则，验证 dry-run/commit 快照非破坏、ledger 兑现/释放正确。

阶段 2：机制 1（约束准入）单开
  把 qtime-safe 硬剪枝 + λ_qtime 软成本接入选择公式；验证搜索不产出违规边、q̂ 不乐观偏高。

阶段 3：机制 3（噪声 rollout + CRN）
  rollout 注入噪声、同源化（5.7）；用决策翻转率（§1.6.5）校准 N_mc。

阶段 4：机制 2（优先级-能力鲁棒性 + reserve 边）
  ρ 由 DyRo 钝代理升级为 ρ_pc；启用 reserve 边。验证 O2 / 预留命中率是否真的优于 SAS-only。

阶段 5（可选）：专家迭代
  拿 VC-MCTS 改进后的决策当标签回训先验 p(s,a)（方法非原创，不作卖点，仅提速在线搜索）。
```

> 实现优先级：**先证明预留有上界收益（阶段 0），再投入搜索调参**。没有 oracle 上界收益时，搜索压不过 SAS-only 不是超参问题，而是实例本身不给预留发挥空间——此时改实例生成器，不是加搜索深度。


---

---

## 7. 与主流方法的对比及可借鉴点（新增）

> 本节基于近两年（2023–2026）公开文献，对照本项目定位，指出可借鉴之处。文献以作者/年份/出处标注，仅作技术参考。

### 7.1 主流技术地图

近年 RL 调度（含 FJSP / FAB）的主流范式可归纳为如下几条线：

1. **异构图神经网络（HGNN）状态表示**：把调度状态建成析取图（operation/machine 为节点，工序先后边与机器共享边为两类异构边），用 HGNN 提取特征。代表：Song et al.（2022/2023，析取图 + 异构 GNN 学习派工规则）、Tang & Dong（HGNNR，Machines 2024，关系子图分解 + 多头注意力融合）、HGT-Scheduler（异构图 Transformer，2026）。其核心论点：把"工序流依赖"与"资源竞争"两类边**分开建模**，比同质图更有表达力。

2. **注意力 / 双注意力 actor**：Wang et al.（2023）用自注意力分别抽取 machine 与 operation 特征，提出 dual-attention 网络，泛化性好。**本项目的双注意力 SAS 正属于这一线，与主流对齐。**

3. **多智能体 / 协同 RL**：把"选作业"与"选机器"或"多区域"拆给多个智能体协同（如 Nature Sci. Rep. 2025 的多目标协同 MARL、IJPR 2025 晶圆厂多区域协同 MARL）。本项目上一版的 RMA+SAS 双智能体属于此线；**本版已把预留从"学习式智能体"改为"搜索式规划器"，故不再主张此线为强项。**

4. **面向真实 FAB 的大规模 / 自监督 + 进化策略训练**：Tassel et al.（WSC 2023）用自监督预训练 + 进化策略训练全局派工网络，在 SMT2020 测试床上优于传统层级派工规则，尤其对占多数的常规 Lot；Stöckermann et al.（2025）在真实工业数据上对比开源模型的可扩展性。该线还普遍指出**学术 RL 缺少在真实 FAB 的验证与部署**（tandfonline 文献综述 2025）。

5. **约束式 / 安全强化学习（Safe RL）**：把"硬约束"作为一等公民、而非奖励惩罚来处理（CMDP + 拉格朗日对偶，深度实现以 PPO-Lagrangian 为代表，文献见 3.3.5）。**本项目把 Q-time 硬约束建模为 CMDP、用 PPO-Lagrangian 处理（3.3），正属于这一线**——这给本项目的约束处理提供了明确的学术坐标，而非孤立工程方案。

6. **在线前瞻规划 / 蒙特卡洛树搜索（MCTS / rollout）**：在决策时对局面做前瞻仿真而非纯反应式派工。经典 rollout（Bertsekas et al. 1997；随机调度 rollout）保证不劣于其基启发式；近两年活跃于动态柔性车间：He（Expert Systems 2025）带运输约束的 MCTS、Saqlain（2023）FJSP-MCTS、Wang（2020）PPO 引导的并行机 MCTS。**与本项目最相关的是 DyRo-MCTS（Chen et al., 2025）**：它把"动作鲁棒性估计"融入 MCTS，引导局面走向对未来到达更易适应的状态，但因完全不可见未来到达，只能用一个钝的机器空闲分布代理（结论是"避免早期空闲"）。**本项目的 VC-MCTS（§5）正属于这一线，且是 DyRo-MCTS 的原理性推广**——在有限前瞻窗内未来到达可见，故用显式 reserve 边 + 仿真比较取代钝代理（窗内"定向空闲/预留"反而最优），窗外才退回 DyRo 式代理；并叠加 Q-time 机会约束准入（DyRo 无任何硬约束）。当窗宽 W=0、无约束、加工确定时退化为 DyRo-MCTS。

### 7.2 本项目相对主流的定位

| 维度 | 主流常见做法 | 本项目 | 评价 |
|------|-------------|--------|------|
| 状态表示 | 析取图 + HGNN | 候选池手工特征 + 注意力（异型柔性下建议后期补 HGNN） | 主流更强表达力；本项目更轻、与资源日历耦合更紧 |
| Actor | 双注意力 / 指针网络 | 双注意力 | 对齐 |
| 多目标 | 多为加权标量或单一 makespan | 向量通道 + 多头 Critic（分离尺度，非多目标权衡） | 工程更稳；软目标仅利用率，不主张多目标先进性 |
| 硬约束 | 多用惩罚，少数用 mask | 机会约束 mask + PPO-Lagrangian（CMDP/Safe RL，λ 自适应） | **本项目更严谨，有理论出处** |
| 动态触发 | 多为事件触发即派工；少数用 MCTS/rollout（DyRo-MCTS 2025） | VC-MCTS 搜索"何时派 + 为谁预留"，窗内可见到达 + Q-time 约束准入 | **相对 DyRo-MCTS 有差异点（见下）** |
| 可行性 | 部分靠后修复 | 资源日历 dry-run/commit | **本项目工程更扎实** |
| 验证 | 缺真实/公开基准 | 待补（见 7.4） | **需补强** |

总体：本项目在**硬约束严谨性、动态决策时机、可执行性**上领先常见做法；在**状态表示的表达力**和**基准验证**上落后主流，是主要可改进项。（软目标仅利用率一项，故不把"多目标"列为强项。）

> 严谨界定（避免夸大创新）：表中"更先进/更严谨"是**相对调度领域常见做法**而言，不等于方法学上的原创。其中向量奖励 + 多头 Critic 是**从多目标 RL 借鉴的成熟做法**；双注意力 SAS 与主流（Wang 2023）**对齐而非领先**；PPO-Lagrangian / CMDP 同样是 **Safe RL 的成熟方法（非本项目首创）**，本项目的贡献在于**将其与半导体 Q-time 报废约束对接、并与机会约束 mask 组成"硬挡+软压"两道防线**这一应用与组合，而非方法本身。本项目真正的差异点集中在 **可见性分区的约束式前瞻搜索（VC-MCTS，§5）**——即在 DyRo-MCTS 的鲁棒 MCTS 之上，(i) 用窗内可见到达 + 显式 `reserve(m,h)` 边取代钝鲁棒性代理（身份绑定预留 vs DyRo 的灵活性/鲁棒性预留），(ii) 把 Q-time 报废硬约束作为机会约束准入嵌进树（DyRo 纯软目标、无硬约束），(iii) 在加工噪声下用 CRN 多路 rollout 求可信估值（DyRo 仅到达一种不确定性）。**须诚实：rollout/MCTS 本身是成熟方法（非原创），创新在上述组合与半导体 Q-time 约束的对接**；正式发表前应再做一轮文献核实，并显式引 Bertsekas(1997)、DyRo-MCTS(2025) 划清边界。

### 7.3 可直接借鉴的点

1. **用异构图增强状态表示（异型柔性下推荐，作为后期增强）**：异型柔性下，"哪个工件能上哪些机器"是复杂的多对多匹配、"多个工件抢同几台机器"是资源竞争结构，候选池手工特征难以充分表达。可把 Machine / Lot（/ Chamber-Side）建成**异构图**节点，"工件-机器可加工"（柔性匹配边）、"机器-机器资源竞争"、"工序先后"建成多类边，用一层 **HGNN** 产生节点 embedding。**注意：HGNN 与双注意力是互补而非替代**——HGNN 在"状态表示层"把局面（含高阶匹配/竞争关系）编码成更好的 embedding，双注意力在"决策层"拿这些 embedding 做候选选择。即 HGNN 给双注意力喂更强的输入特征，而非取代它。这是用 Song/Tang 思路的**增量增强**，保留资源日历与候选池索引动作空间不变。**建议作为后期增强**（基础版先用手工特征+双注意力跑通，再上 HGNN 做消融对比），避免一上来架构过重、训不动。

2. **自监督 / 启发式预训练做 warm start**：借鉴 Tassel 的自监督预训练 + ES，或更简单地用现有派工规则（`run_rule_episode` 的 FIFO/SPT/EDD/CR/ATC）生成的高质量轨迹做行为克隆预训练，再 PPO 微调，缓解稀疏终局奖励下的冷启动与振荡（用于训练 SAS；预留侧改用 §5 搜索后，warm start 仅作用于 SAS 与可选的搜索先验回训 §6.2.3 阶段 5）。（注：项目当前未实现 NSGA-II 等元启发式，故 warm start 的轨迹源以现有派工规则为准。）

3. **对标公开测试床 SMT2020**：主流 FAB 调度普遍以 SMT2020 + 传统层级派工规则为基线。即使本项目最终用自有实例，也应在 SMT2020（或其子集）上与启发式派工规则对比 on-time 率与 cycle-time，否则结果难以横向比较、可信度受限。

4. **评估口径**：因可权衡软目标仅利用率一项，**不做多目标帕累托/超体积的强主张**。评估应分层报告：硬约束（阶段间 Q-time 违规数与违规概率是否 ≤ ε）作为首要门槛，利用率作为在满足约束前提下的优化指标，优先级违反度作为偏好满足度，拖期（交付期）另列。若未来问题扩展出第二个真正可权衡的软目标，再引入帕累托/超体积分析。

5. **关注验证/部署 gap**：文献综述（2025）反复指出学术 RL 调度少有真实部署。本项目的资源日历 dry-run/commit 恰是面向可执行性的工程优势，应在报告中作为差异化卖点强调，并补一节"如何从仿真过渡到真实派工"。

### 7.4 建议的基准与验证

```text
基线对比：FIFO / SPT / EDD / CR / ATC 等派工规则（evaluate_baselines.py 已实现，与 RL 共用同一 qtime-safe 候选池）+ SAS-PPO greedy；元启发式（如 NSGA-II）当前项目未实现，如需横向对比须另行补充
公开测试床：SMT2020（HV/LM、LV/HM 场景），报告各优先级 Lot 的 on-time% 与 cycle-time
消融实验：单注意力 vs 双注意力；有/无 qtime mask；有/无 VC-MCTS 预留（VC-MCTS vs SAS-only vs DyRo-MCTS）；机制 1 vs +机制 3 vs +机制 2（逐层加）；标量 vs 向量奖励 + 多头 Critic
分层评估：硬约束门槛（阶段间 Q-time 违规数、违规概率≤ε）+ 利用率 + 优先级（主判据 O2 加权等待 priority_weighted_wait、公平展示 O1 可避免逆序 avoidable_priority_violation，旧 priority_violation 仅参考，见 §7.5）+ 拖期（另列）（软目标仅利用率，不做帕累托/超体积强主张）
鲁棒性：不同到达强度、不同前瞻窗 W_lookahead、不同 qtime 紧度下的违规率与拖期
```

**自建可区分压力基准（`build_pressure_test_encoder`，50 Lot × 10 机台）。** 要让上述对标真正区分策略，实例须同时满足「约束会咬、软目标有余量」，否则贪心规则已近最优、RL 无取胜空间（实测：早期版本所有约束都不咬时，FIFO/SPT/EDD/CR/ATC 与 SAS-PPO 在违规数/利用率上几乎不可区分）。本实例据此设定两个旋钮：

```text
阶段间 Q-time：对工序 (1→2)、(2→3) 设上限 qtime_limit（默认 3.0）。
  Q-time 是 lot 材料的队列时间约束（与最终选哪台机器/哪个 ppid 无关），
  故对所有可被调度的 (lot, machine, ppid) 登记；腔体争用使阶段衔接被拖时即违反，
  让"派哪个 lot / 何时派"直接影响下游能否及时衔接（单一贪心规则做不好的地方）。
错峰到达：50 lot 按 Poisson 到达（指数间隔，均值 arrival_mean_gap，默认 0.6；lot1 在 t=0），
  到达率略超系统吞吐 → 机台始终有活、util 由派工质量而非到达饥饿决定，并让
  "现在派 vs 等下一个更优 lot"成为有意义的决策（正是搜索要导航的对比场景）。
  （经扫描定稿：gap=0.6 时好规则 util ~0.83、CR 顶 0.888，且 Q-time 违反在规则间从 6 到 36 拉开，
   形成最清晰的 Q-time/利用率权衡，给 RL 留出"双赢点"空间。）
```

> 实测该实例已构成 **Q-time / 利用率 的多目标权衡**：CR 利用率最高但 Q-time 违反最多，FIFO/EDD 保守、违反少但利用率低——正是 RL 要导航的对象。两个旋钮可调强度（越小越紧）。
> **重要修复（与代码对齐）：** 此实例此前从未设置 `q_time_limits`，导致 `compute_q_time_violation` 恒为 0 → Q-time 指标、`r_qtime` 奖励通道、§3.3 的 PPO-Lagrangian（`mean_violation` 恒 0 故 λ 永不启动）**全部静默失效**。补齐阶段间 Q-time 后三者同时复活。

### 7.5 阶段 5a 实证：规则预留机制与 priority 目标修正（VC-MCTS 搜索式预留的动机）

在上 RL / 搜索之前，先以**规则版预留探针**（非学习，`Phase2EpisodeDriver.ddt_reserve`，自带前瞻窗）验证 §2.3"为即将到达的高优先级 Lot 预留资源"这条线是否可行、是否值得做成 VC-MCTS 搜索：前瞻窗 `[t_now, t_now+win]` 内若有优先级比当前候选池最优高 ≥`gap` 的即将到达 Lot，就让当前空闲机台暂不派工（预留），等其到达再派。脚本 `ddt_reserve_probe.py`（结果 `results/ddt_reserve_probe.txt`）。

**指标病灶（为何旧 `priority_violation` 不可信）。** 旧 `priority_violation`（全体 Lot 按开工时间排序的逆序总量 `Σ max(0, p_j−p_i)`）对**在线到达病态**：一个**晚到**的高优先级 Lot 必然晚开工，于是所有早到的低优先级 Lot 都被计入它的逆序——而 Lot **不可能在到达前开工**，这部分逆序物理不可避免。实测压力实例 strict 规则基线：可避免部分仅 220.8（**11.6%**），不可避免部分 1674.7（**88.4%**）。靠预留把该指标砸到 219（≈可避免地板）是用**大面积空转**买下不可避免部分 = gaming，并非真实改善。故 §4.10 据此把 priority 维度改为 O2（主）/ O1（展示），旧量降级为参考。

**规则版预留结果（pressure，FIFO，3 seed；除 util 外越小越好）：**

| 配置 | O2 加权等待 ↓ | O1 可避免逆序 ↓ | util ↑ | 阶段间 Q-time 违规 ↓ | 旧 priority_violation（参考）|
|---|---|---|---|---|---|
| soft，无预留 | **862.5** | 220.3 | 0.844 | 10.7 | 1895.5 |
| soft，预留 w4 | 1888.8 (+119%) | 664.8 | 0.759 | 20.7 | 1846 |
| strict，无预留 | 874.6 | 220.3 | 0.830 | 8.7 | 1895.5 |
| strict，预留 w2 | 880.1 (+0.6%) | 243.1 | 0.827 | 19.3 | 1536 |
| strict，预留 w4 | 975.3 (+11.5%) | **11.7 (−95%)** | 0.748 | 24.0 | 219 |

> **结论。**（1）在真正要优化的 **O2** 上，钝的规则预留**处处是负**（strict_w2 打平、strict_w4 +11.5% 更差、soft +119% 灾难）——它靠空转（1200–1750 机台·步）实现重排，把所有 Lot 的等待都拉长，得不偿失；这恰说明 **O2 正确区分了"有用重排"与"浪费空转"**，旧指标的 −88% 是 gaming。（2）但 **O1** 显示 strict_w4 把可避免逆序 220→12（**−95%**），**优先级杠杆真实存在、可抓**，只是规则版抓得太糙（全有或全无地按阈值预留），且必须与优先级感知的池内选择（strict 过滤）耦合，否则（soft）预留只是空转、连 O1 都更差。（3）故规则版**证伪了"钝预留改善正确目标"**，同时为 **§5 的 VC-MCTS 搜索式预留** 立下明确动机：需**外科手术式选择性预留**（仅当 高优先级少等×高权重 > 他人多等×低权重 才留），规则穿不过这针眼，VC-MCTS 通过对 `reserve(machine,future_lot)` 边与 dispatch 边的 rollout 模拟比较（直接、精确地算出两者的 O2 之差）来做这类选择，目标用 **O2** 且与 Q-time 约束联动（预留会推高 Q-time，由机制 1 准入层挡住）。开放问题 = VC-MCTS 能否靠选择性预留把 O2 压到规则地板（soft 无预留的 862）以下；在投入搜索前必须先用 oracle（信息完整 + 大预算的同一套搜索）验证上界。

---

### 7.6 当前代码进度与 VC-MCTS 在线预留实证（与仓库对齐）

截至当前代码版本，VC-MCTS 已不再只是方案设计，已经完成第一阶段在线预留闭环。相关实现文件如下：

```text
reservation_ledger.py       # reservation ledger，按 machine 记录预留，并防止同一 future_lot 被重复预留
reservation_rop.py          # ROP 候选生成，跳过已预留机器与已预留 future_lot
reservation_simulator.py    # reservation-aware rollout/episode helper，提供 clone 与 ledger-aware 推进
vc_mcts_planner.py          # root-level VC-MCTS planner: no_op / dispatch / reserve 三类 root edge
vc_mcts_probe.py            # baseline / oracle / VC-MCTS probe，支持 --workers 并行、trace/summary 输出
vc_mcts_trace_summary.py    # JSONL trace 汇总，统计 reserve 可用率、选择率、gap、重复预留等诊断字段
```

**当前实现边界。** 代码已经跑通“VC-MCTS 管预留 + 规则 rollout/TopK dispatch 管派工”的第一阶段；训练好的 SAS policy 尚未作为下层派工器接入。因此，报告中凡涉及“SAS 作为搜索先验/基策略/叶子估值”的内容应理解为目标架构或下一阶段集成方向，而非当前已完成项。当前实际策略为：

```text
root action = no_op
            + TopK dispatch(lot,ppid) from ResourceCalendarEnv.build_candidate_pool(machine)
            + TopB reserve(machine,future_lot) from ROP

rollout policy = rule strategy（当前验证主要用 FIFO）
final choice   = qtime_count → qtime_total → O2 → utilization → visits
no_op gating   = no_op 必须在 qtime_count 或 qtime_total 上严格优于非 no_op 边，否则降级
```

**关键工程修复。**

1. **trace / summary 诊断闭环。** 每次 VC-MCTS 决策可写 JSONL，包含 selected_action、edges、mean_qtime、mean_qtime_total、mean_o2、mean_util 与 diagnostics；summary 自动统计 `reserve_available_decisions`、`reserve_selected_decisions`、`reserve_selection_rate_when_available`、`reserve_o2_gap_vs_best_non_reserve_avg`、`reserve_qtime_total_gap_vs_best_non_reserve_avg` 等字段。
2. **qtime-first objective。** `VCMCTSObjective` 已加入 `qtime_violation_total`；最终选择不再以 visits 为第一优先级，而是先看 Q-time 违规数量与严重度，再看 O2 与利用率。
3. **no_op gating。** 早期完整轻预算中 no_op 过多（120 次决策里 no_op=89，episode 只完成 31/50）。加入 gating 后，no_op 降到 1 次，完整 episode 能完成 50/50。
4. **重复预留去重。** `ReservationLedger` 增加 `reserved_lots()` / `is_lot_reserved()`，并拒绝同一 future_lot 被不同机器重复预留；ROP 生成时跳过已预留 lot。trace summary 新增 `duplicate_selected_reserve_lots`。
5. **多进程 seed 并行。** `vc_mcts_probe.py` 新增 `--workers`；多 worker 时每个 seed 自动写独立 trace/summary，避免多个进程同时 append 同一 JSONL。

**late_hi 完整轻预算验证。** 运行命令：

```powershell
python vc_mcts_probe.py `
  --instance late_hi `
  --seeds 2 `
  --workers 2 `
  --strategy FIFO `
  --skip-oracle `
  --top-b 3 `
  --top-k-dispatch 2 `
  --n-iter 2 `
  --max-steps 600 `
  --rollout-max-steps 60 `
  --max-decisions 120 `
  --trace-out results\vc_mcts_late_hi_complete_light_gated_s2_trace.jsonl `
  --trace-summary-out results\vc_mcts_late_hi_complete_light_gated_s2_summary.json `
  --progress-every 10
```

输出文件按 seed 拆分：

```text
results/vc_mcts_late_hi_complete_light_gated_s2_seed0_trace.jsonl
results/vc_mcts_late_hi_complete_light_gated_s2_seed0_summary.json
results/vc_mcts_late_hi_complete_light_gated_s2_seed1_trace.jsonl
results/vc_mcts_late_hi_complete_light_gated_s2_seed1_summary.json
```

结果（`--noise` 未开启，因此该轮主要验证确定性重复与并行流程；两个 seed 仍可能因并行进程/内部排序细节产生轻微差异）：

| seed | baseline qtime | VC qtime | baseline O2 | VC O2 | O2 改善 | completed | termination | reservations | reserve 选择率 | no_op | duplicate reserve |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|
| 0 | 20 | 0 | 1664.32 | 1137.27 | -527.05 | 50/50 | all_lots_completed | 19 | 42.2% | 1 | [] |
| 1 | 20 | 0 | 1664.32 | 1156.86 | -507.46 | 50/50 | all_lots_completed | 20 | 45.5% | 1 | [] |

阶段性结论：

```text
1. VC-MCTS 已经具备在线预留能力：会生成 reserve、评估 reserve、选择 reserve、兑现/释放 ledger。
2. 在 late_hi 完整轻预算下，两个 seed 都完成 50/50 lot，且 Q-time violation 从 baseline 的 20 降到 0。
3. O2(priority_weighted_wait) 稳定改善约 500+，说明不是单纯牺牲 O2 换 Q-time。
4. no_op gating 解决了早期“等待过多导致 episode 跑不完”的问题。
5. duplicate_selected_reserve_lots 为空，说明同一 future_lot 重复预留 bug 已修复。
6. 当前结果仍是轻预算 + rule rollout 版本；下一阶段应验证 noise 鲁棒性，并把 dispatch 从 TopK/rule delegate 逐步替换为训练好的 SAS policy。
```

因此，当前项目状态可表述为：**VC-MCTS 预留模块第一阶段已经闭环并在 late_hi 上通过完整轻预算验证；尚未完成与训练型 SAS policy 的最终 delegate 集成，也尚未进行 noise/multi-instance 鲁棒性验证。**

---

## 8. 最终结论

本项目本质上是面向 FAB 机台组的特殊并行机调度问题：机台组内机器为**异型 + 柔性**（机器能力不同、一个工件可被多台机器加工），需为待排产 Lot 决定加工 Machine、PPID、上/下机时刻，以及 wafer 在 Chamber/Side 上的顺序与时间，并在**工件随机到达（有限前瞻可见）与加工时间随机噪声的双重不确定性**下，满足 Q-time 硬约束（阶段间队列时间，随机下表述为机会约束 P(违规)≤ε）、优先级强偏好，并尽量提高机台利用率（唯一可权衡的软目标；拖期为独立交付期指标）。方案采用**两层架构**：下层用固定规则估算工件内部批处理的完成时间分布，上层用强化学习做派工与时机决策。

完善后的方案可概括为：

1. 显式建模**黑盒到达 + 有限前瞻**，把"为未来高优先级 Lot 预留资源"变为窗内可仿真、可搜索、可预判 + 窗外统计兜底；并用 **VC-MCTS（可见性感知的约束式 MCTS）以 `reserve(machine,future_lot)` 为搜索边**驱动选择性预留（§5），作为相对主流（含 DyRo-MCTS）的差异点；
2. 确立 **Q-time > priority > 利用率** 的字典序结构（Q-time 为阶段间队列时间硬约束；拖期为独立的交付期指标，不进奖励）：Q-time 用**机会约束 mask**（随机加工时间下 `deadline−μ<z_ε·σ`，违规概率≤ε）在决策时硬挡，并以 **PPO-Lagrangian（CMDP/Safe RL 框架，λ 自适应对偶上升）** 在训练目标层面压住残差违规；priority 用候选池过滤（+VC-MCTS 目标绑定式搜索预留）、利用率为唯一软目标；
3. 显式处理**加工时间不确定性**（纯工艺噪声，μ/σ 入状态）：完成时间为分布，Q-time 改写为机会约束，safety margin = z_ε·σ 自适应、有概率含义，训练时注入噪声、评价在多次随机 rollout 上统计；
4. 即时奖励**只管动作合法性**，质量信号下沉到**终局向量**，由**多头 Critic + 逐目标 GAE**分离学习，权重作用在归一化 advantage 上；
5. **两层架构**：下层用固定规则（满批组批 + list scheduling + 蒙特卡洛）估算工件内部批处理的完成时间分布 (μ,σ)，作为上层 Q-time 判断与候选池打分的地基；上层用 RL 做有取舍的派工/时机决策（1.5）；
6. Actor 用机台条件化 / 双注意力，资源日历 dry-run/commit 保证可执行性，训练保持 PPO 轨迹一致性；
7. 分阶段推进（环境+估时器 → 单头执行闭环 → mask/过滤 → 多头向量 → 双注意力 SAS → oracle 上界 go/no-go → VC-MCTS 在线预留 → 公开基准评估）；
8. 借鉴主流的**异构图状态表示、自监督/启发式 warm start、SMT2020 基准、分层评估（硬约束门槛 + 利用率，不做帕累托/超体积强主张）**，补强状态表达力与验证可信度。

> **在不改变 FAB 机台组调度问题定义的前提下，构建一个面向"工件随机到达 + 加工时间随机"双重不确定性的两层调度框架：下层以固定规则估算工件内部批处理的完成时间分布，上层以有限前瞻、字典序约束（Q-time 机会约束 mask + priority 候选池）、混合决策（SAS 用 PPO 学当前派工 + VC-MCTS 搜索目标绑定式资源预留）、双注意力动作选择与 PPO 训练为核心，并对标公开基准（含 DyRo-MCTS）的智能调度原型。**

---

## 9. VC-MCTS 预留规划器自查与一致性 review

**9.1 本次修订解决的原问题。** 上一版把预留做成可学的 RMA，修掉了旧 DDT 的无目标 hold 与奖励重复计数，但未能根除学习范式的结构性顽疾——greedy 临门一脚顶不过阈值、ROP 宽触发造成 no_reservation 类别不平衡，使策略持续坍缩到"从不预留"。本版用 VC-MCTS 把"要不要为某未来高优先级 Lot 留这台机器"从"学一个概率"改为"模拟 reserve 与 dispatch 两条分支、按字典序比一比"，从而在结构上消除奖励坍缩、greedy under-hold 与类别不平衡三个失败模式（§5.1）。

**9.2 与三层字典序结构是否冲突。** 不冲突，且实现得更直接。Q-time 仍是第 0 层硬约束——搜索的准入层（机制 1）对违反机会约束的边硬剪枝、对窗外残差用 λ_qtime 软压（§5.4），与 SAS 同口径、同下层估时器；priority 是第 1 层强偏好——由优先级-能力鲁棒性 ρ_pc 与 reserve 边承载（机制 2，§5.5）；utilization 仍是唯一可让步软目标，进入价值估计 q̂。执行顺序"先剪枝→再按 ρ_pc+价值选"天然实现字典序。

**9.3 与 SAS 职责是否混淆。** 目标架构中不混淆：VC-MCTS 负责“是否预留/为谁预留/是否等待”，SAS 负责“当前机器派哪个 `(lot,ppid)`”。但**当前代码第一阶段尚未接入训练好的 SAS policy**，因此 `vc_mcts_planner.py` 仍把 Top-K dispatch 作为 root edge，与 no_op/reserve 一起由 VC-MCTS 选择；rollout 也使用 FIFO 等规则策略。这是为了先把 reservation ledger、ROP、rollout、trace、去重与 no_op gating 跑通。下一阶段应把 dispatch edge 抽象为 `delegate_dispatch`，先接 rule delegate 验证等价，再替换为 SAS policy。

**9.4 ROP 是否会过于苛刻。** 仍为宽触发式，且角色更纯：现在它只是搜索的算力闸门——硬规则保留结构可行、前瞻窗可见、非注定违规、Q-time 安全等必要条件，S_res 只排序截断 TopB；是否真的预留交给树搜索的模拟比较，而非规则或概率（§5.8、§6.2.2）。

**9.5 动作（分支）空间是否会过大。** 有风险，但当前代码已通过“宽触发 ROP + TopB reserve + TopK dispatch + no_op”控制分支因子，并强制 `iteration_count=max(len(root_edges), n_iter)`，保证小预算下每条 root edge 至少 rollout 一次。后续若接入 SAS delegate，可进一步把多个 dispatch edge 收缩为一个 `delegate_dispatch` 边，降低分支因子。

**9.6 是否仍有奖励重复计数风险。** 不再有——本版预留不训练、无奖励通道，reserve 分支的反事实收益由 rollout 直接、精确地模拟得出（reserve 分支尊重 ledger、dispatch 分支不尊重，两条 O2 之差即收益，§5.5/§5.7），取代了上一版担心会与终局 O2 重复计数的近似 `r_cf`。须守住的命门改为"rollout 内部与树策略同源"（§5.7），否则 q̂ 失真。

**9.7 是否可落地。** 可落地，但成本从训练期搬到推理期：每个 VC-MCTS 决策点要 clone driver/ledger 并跑若干条 reservation-aware rollout。当前已通过 `rollout_max_steps`、`max_decisions`、trace summary、`--workers` 并行和轻预算参数把 late_hi 完整验证控制在可运行范围内。后续缓解方向仍是 §5.11 所述的 Critic 叶子估值、子树复用、收紧 ROP/TopB、以及用 SAS delegate 减少 dispatch 分支。

**9.8 当前仍需实验验证的边界。** 预留收益依赖实例中是否存在"晚到高优先级 Lot + 兼容机器稀缺 + 当前派工会挤占未来"的结构。当前 `late_hi` 已证明存在预留杠杆：轻预算 VC-MCTS 可把 baseline 的 Q-time violation count 从 20 降到 0，并使 O2 改善约 500+。但这仍是 `--noise` 未开启、rule rollout、单实例族的结果；下一步需要做三类验证：（1）开启 `--noise` 的鲁棒性；（2）不同到达强度/不同 qtime 紧度的实例族；（3）接入 SAS delegate 后与 rule delegate 的消融对比。若某实例中 reserve 边长期输给 dispatch，不应盲目堆搜索深度，而应先检查该实例是否真的具有预留杠杆。
