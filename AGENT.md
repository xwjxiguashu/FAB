# AGENT.md

## 1. 本次会话的总体总结

本次会话主要完成了以下几类工作：

1. **分析当前项目的调度框架与下一步算法方向**；
2. **根据代码现状重写项目建模说明**；
3. **围绕半导体车间调度方向筛选近三年高质量论文**；
4. **逐篇阅读 `lunwen/15/` 目录中的前 7 篇论文并形成中文总结**；
5. **按用户要求把项目判断、文档修改和论文总结持续整理到 `AGENT.md` 文件中。**

当前项目目录：`C:\Users\HP\Desktop\code\FABenv`

本轮会话中，用户曾在 IDE 中打开但尚未进一步处理的文件包括：

- `run_large_instance_gantt.py`
- `rolling.py`
- `local_search.py`

这些文件目前**没有被修改**。

---

## 2. 当前项目的核心判断

### 2.1 项目本质

根据对现有代码的阅读，本项目不是普通静态 FJSP，而是一个：

- 面向 **FAB 机台组** 的调度原型；
- 使用 **动态滚动调度**；
- 显式考虑 **Lot 动态到达**；
- 同时进行 **Machine / PPID / Chamber-Side** 多层决策；
- 输出 **Lot 级 + Wafer 级** 排程；
- 通过 **Machine 日历** 与 **Chamber/Side 日历** 保证可行性；
- 用 **NSGA-II** 搜索策略权重；
- 用 **Q-time、拖期、优先级违背、机台利用率** 做多目标评价。

### 2.2 当前已经具备的实现能力

- 滚动窗口调度；
- freeze window / 固定近期操作；
- `Lot → Machine → PPID → Stage → Chamber/Side` 层次式决策；
- 资源日历解码器；
- NSGA-II 多目标搜索；
- 甘特图绘制与 CSV 导出；
- Q-time 违背计算；
- 并行个体评价；
- 最终排程完整性校验。

### 2.3 当前尚未真正实现的内容

尽管旧文档中曾经描述过一些更强能力，但代码层面目前**尚未真正完成**：

- 多场景加工时间扰动评价；
- `mean + CVaR` 鲁棒目标聚合；
- 神经网络策略；
- 强化学习训练；
- policy-based local search / VNS；
- Q-time 硬约束修复；
- Chamber/Side 利用率目标。

### 2.4 后续方向判断

根据当前代码结构，本项目最自然的下一步创新方向是：

> **学习增强的半导体 FAB 动态滚动调度**

更具体地说：

- 保留当前的资源日历解码器和可行性保证机制；
- 用 **神经网络 / Attention / RL** 替代当前线性打分策略；
- 优先从 **Lot 选择层** 开始替换；
- 后续逐步扩展到 **Machine/PPID** 和 **Chamber/Side**；
- 最终形成 “**学习策略 + 可行性解码器 + 多目标评价**” 的框架。

### 2.5 已修改的项目文档

已根据代码现状重写并更新：

- `项目建模说明.md`

更新后的文档明确：

- 当前项目是“**确定性动态滚动调度 + NSGA-II 策略权重进化优化**”；
- 明确区分“**当前已实现**”与“**未来扩展方向**”；
- 不再把尚未实现的 CVaR / RL / 神经策略写成已完成内容。

---

## 3. 论文检索与阅读进度

### 3.1 总体论文方向

在对当前项目定位之后，会话中围绕近三年（约 2023–2026）的半导体车间调度论文做了定向筛选，认为最值得持续跟踪的方向包括：

- 半导体前道 fab 动态派工；
- DRL / Evolution Strategies / Self-Supervised Learning；
- Cluster tool / Wet station / Wet clean 调度；
- Q-time / zero-wait / robot transfer；
- 真实工业级仿真 / benchmark 泛化；
- 可解释调度策略；
- 细粒度设备内部调度。

### 3.2 已完成详细总结的论文

目前已对 `lunwen/15/` 中前 **7 篇论文** 做了系统阅读与中文总结：

1. `1.Scalability of Reinforcement Learning Methods .pdf`
2. `2.Explainable AI for reinforcement learning based dynamic scheduling .pdf`
3. `3.Dispatching_in_Real_Frontend_Fabs_With_Industrial_Grade_Discrete-Event_Simulations_by_Deep_Reinforcement_Learning_with_Evolution_Strategies.pdf`
4. `4.Semiconductor Fab Scheduling with Self-Supervised and Reinforcement.pdf`
5. `5.Deep reinforcement learning for.pdf`
6. `6.Scheduling of Automated Wet-Etch Stations with One Robot in.pdf`
7. `7.Machine Learning-based Dispatching for a Wet Clean.pdf`

---

## 4. 论文 1–7 的详细总结

---

## 4.1 第 1 篇论文

**文件**：`lunwen/15/1.Scalability of Reinforcement Learning Methods .pdf`

**题目**：
**Scalability of Reinforcement Learning Methods for Dispatching in Semiconductor Frontend Fabs: A Comparison of Open-Source Models with Real Industry Datasets**

### 研究问题

这篇论文不是简单证明 RL 能不能用于半导体 fab dispatching，而是进一步问：

> **当问题规模、约束复杂度和仿真复杂度提高时，RL 方法是否还能有效扩展到真实工业级前道 FAB？**

这意味着论文重点不在“一个算法比另一个算法好一点”，而在：

- 小型 benchmark 和真实 fab 差异有多大；
- public benchmark 上有效的方法是否能迁移到工业级场景；
- PPO 和 ES/CMA-ES 在复杂调度环境中的 scalability 差异是什么；
- 并行训练、多随机种子、多 scenario 是否能提高泛化能力。

### 研究背景与动机

作者指出现有半导体 RL 调度研究有几个共同问题：

- 研究使用的数据集不同，难以直接比较；
- 同一模型在不同 simulator 上表现可能不同；
- stochastic simulator 会受 random seeds 影响；
- MiniFab、SMT2020 等 benchmark 不足以完全反映真实前道 fab 复杂性；
- 真实工业数据通常不可公开，导致研究和工业之间有鸿沟。

因此，这篇论文的真正价值是：

> **把“调度算法能不能跑”上升为“调度算法在不同复杂度半导体系统中的扩展性比较”问题。**

### 方法与实验框架

论文比较了 3 类 testbed：

1. **MiniFab**：小规模公开半导体调度模型；
2. **SMT2020**：更复杂、更接近真实 fab 的 benchmark；
3. **Real industry dataset**：真实工业级前道 fab 数据。

在算法上比较：

- **PPO**：经典 policy-gradient RL；
- **Evolution Strategies / CMA-ES**：黑盒、可并行、episode-level 优化方法。

他们还比较：

- 控制单个工具组 vs 多个工具组；
- 单 scenario 训练 vs 多 scenario 训练；
- 多 random seed 训练；
- 不同 CPU 并行数下的训练速度和收益。

### 核心发现

#### 1. PPO 的有效性依赖问题规模

- 在 MiniFab 上，PPO 可以取得明显改进；
- 在更复杂的 SMT2020 和工业级 fab 上，PPO 很难稳定收敛；
- 原因是 delayed reward、sample correlation、长 horizon 和 value estimation error。

#### 2. ES / CMA-ES 更适合复杂半导体调度

- ES 在大规模复杂环境中更稳；
- 因为它不依赖 value function；
- 可以直接根据整轮仿真 KPI 优化策略；
- 更适合长周期 delayed effect 强、reward 稀疏的调度环境。

#### 3. 多工具控制的改进空间更大

- 只控制某个瓶颈工具时，改进有限；
- 控制更多 bottleneck / critical tools 时，潜在收益更高；
- 说明局部优化工具组合的选择非常重要。

#### 4. 多场景训练提高泛化

- 多 loading scenarios 和多 random seeds 训练能减少过拟合；
- 多样化训练集有助于让策略跨环境、跨负载泛化；
- 这对工业部署比单场景最优更重要。

#### 5. 计算成本是必须报告的结果

- 论文专门分析 CPU 并行扩展性；
- 说明工业级调度论文必须讨论算力消耗，而不仅是 KPI 提升。

### 对当前项目的启发

这篇论文与当前项目的关系非常直接：

- 你当前已经在用 NSGA-II 搜索策略权重，本质上比 PPO 更接近 ES 路线；
- 如果未来引入神经策略，最自然的做法是：
  - 保留当前解码器；
  - 把线性打分器替换成神经 scorer；
  - 用 ES / CMA-ES / NSGA-II 优化，而不是先做 PPO。

此外，它还启发了后续实验设计：

- 多 seed；
- 多规模实例；
- 多 loading scenario；
- 多 bottleneck / 多决策层控制对比；
- 训练成本与并行效率分析。

---

## 4.2 第 2 篇论文

**文件**：`lunwen/15/2.Explainable AI for reinforcement learning based dynamic scheduling .pdf`

**题目**：
**Explainable AI for reinforcement learning based dynamic scheduling solutions in semiconductor manufacturing**

### 研究问题

这篇论文聚焦于：

> **半导体 DRL 调度策略虽然有效，但如果生产专家不能理解它为何这样调度，就难以落地。**

因此，它研究的不是新的调度算法，而是：

- 如何解释 RL agent 的 dispatching logic；
- 如何证明 agent 学到的是“制造逻辑合理”的策略；
- 如何提高黑盒调度系统的 trustworthiness；
- 如何帮助数据科学家改进策略网络本身。

### 研究动机

作者指出在半导体制造场景中，调度系统要被接受，需要同时面向：

- 工艺专家；
- 生产管理者；
- 调度工程师；
- 数据科学家。

而神经网络通常被看作黑盒，尤其 RL 更难解释。因此如果不引入解释框架，就会出现：

- 现场不信任；
- 无法判断是否 reward hacking；
- 无法发现模型是不是依赖了不合理特征；
- 无法知道模型和传统启发式的差异到底在哪里。

### 提出的解释框架

论文最核心的贡献是提出一个 **holistic XRL framework**，从不同层次解释 agent：

1. **Queue-wise SHAP**
   - 分析某一决策时刻当前 queue 中，哪些特征对每个 lot 的 score 影响最大；
   - 适配可变长度队列。

2. **Counterfactual sensitivity analysis**
   - 逐步修改被选中 lot 的某个特征，看看何时会触发 agent 改选别的 lot；
   - 用于分析特征敏感性和决策稳健性。

3. **Attention network analysis**
   - 分析 lot-on-lot attention matrix；
   - 解释 queue 中 lot 之间如何相互影响。

4. **Decision tree surrogate**
   - 用可解释的树模型近似 agent 的决策逻辑；
   - 从中归纳接近 dispatching rule 的决策路径。

5. **Heuristic overlap / Venn analysis**
   - 分析 RL agent 的选择与 FIFO、EDD、SRPT、setup rule 等规则的重合度；
   - 看它更像哪种 heuristic，或者是否学出了混合策略。

### 主要发现

- 多种解释方法虽然角度不同，但能得出一致结论；
- 重要特征包括 step due date、setup、batching-related features、remaining cycle time 等；
- agent 的行为并不是单纯复制 FIFO 或某个规则，而是在规则基础上学习了更细粒度的策略；
- 解释结果可用于发现模型是否有异常行为；
- 可解释性越强，越容易促进工业落地。

### 对当前项目的启发

当前项目虽然还没有神经网络策略，但它已经有：

- 线性策略权重；
- 明确的特征向量；
- 决策轨迹；
- Q-time、due date、priority、machine load 等可解释特征。

因此可以先做两层工作：

1. **先解释当前线性策略权重**；
2. **未来若上 Attention/RL，再引入类似 SHAP / counterfactual / heuristic overlap 的模块。**

换句话说，这篇论文说明：

> 你的未来工作不应只停留在“性能更好”，还应包括“为什么这样调度”的解释层。

---

## 4.3 第 3 篇论文

**文件**：`lunwen/15/3.Dispatching_in_Real_Frontend_Fabs_With_Industrial_Grade_Discrete-Event_Simulations_by_Deep_Reinforcement_Learning_with_Evolution_Strategies.pdf`

**题目**：
**Dispatching in Real Frontend Fabs With Industrial Grade Discrete-Event Simulations by Deep Reinforcement Learning with Evolution Strategies**

### 研究问题

这篇论文研究：

> **在真实工业级半导体前道 FAB 数字孪生中，能否通过 DRL + ES 直接改善 bottleneck tools 的 dispatching 质量？**

它关注的不是 benchmark，而是更接近真实 fab 的数字孪生环境。

### 场景与环境

- 使用工业级离散事件仿真器；
- 仿真模型参数来自真实 fab 历史数据；
- 包括 machine breakdown、process variability、loading scenario 等；
- 规模达到 1000+ pieces of equipment；
- 重点控制 **lithography clusters**，因为它们是 bottleneck。

### 方法框架

整体流程是：

- 仿真器在每次 bottleneck 设备可用时触发 callback；
- 当前 queue 中的 legal lots 作为 observation；
- attention-based neural policy 为每个 lot 计算 score；
- 选 score 最高者 dispatch；
- 完整 episode 跑完后，计算 FF / throughput / tardiness；
- 用 ES / CMA-ES 更新策略参数。

### 关键设计

1. **只控制 bottleneck tools**
   - 没有试图让 RL 控制全厂所有设备；
   - 只控制最关键的 lithography clusters；
   - 其他设备继续使用默认 rule-based dispatching。

2. **Self-attention policy**
   - 用于处理可变长度 lot queue；
   - 保留相对关系信息；
   - 输出每个 lot 的优先级分数。

3. **Evolution Strategies 训练**
   - 避免设计 step-level dense reward；
   - 直接基于 episode-level KPI 优化。

### 实验结论

- 学习策略在若干 loading scenario 上优于当前规则；
- 多场景训练比单场景更稳；
- FF 目标需要结合 tardiness penalty，否则可能牺牲交付表现；
- 对工业级场景而言，即使只提升 1–3%，也具有明显价值。

### 对当前项目的启发

这篇论文和你当前项目的相似点非常强：

- 都是 dispatching-oriented；
- 都更适合用 ES/进化搜索，而不是 PPO；
- 都适合先控制关键层，而不是端到端替换全部逻辑；
- 都适合保留底层可行性机制，把学习集中在评分或排序层。

对你最直接的建议是：

> 未来先把 **Lot 选择层** 替换成学习策略，保留 Machine / PPID / Chamber 解码器。

---

## 4.4 第 4 篇论文

**文件**：`lunwen/15/4.Semiconductor Fab Scheduling with Self-Supervised and Reinforcement.pdf`

**题目**：
**Semiconductor Fab Scheduling with Self-Supervised and Reinforcement Learning**

### 研究问题

研究如何在 SMT2020 这种：

- 连续运行；
- stochastic；
- dynamic；
- large-scale；
- 具有 Hot Lots、CQT、setup、batching、tool family 等复杂约束

的半导体环境中训练一个全局 lot dispatching agent。

### 主要贡献

#### 1. 13 个 Lot 状态特征

论文为每个 lot 构造了 13 个特征，例如：

- Critical Ratio；
- 距离 due date 的剩余时间；
- release 以来总等待时间；
- 上一工序后的等待时间；
- lot-to-lens dedication 数量；
- lot priority；
- 剩余工序平均加工时间；
- setup time；
- 当前工序平均处理时间；
- compatible machines 数量；
- minimum / maximum batch size；
- tool family。

#### 2. Self-attention policy network

使用 self-attention 对 legal lots 进行全局建模，输出每个 lot 的 priority score。

#### 3. Tool family embedding 的自监督预训练

这是本文的重要创新之一：

- 先通过自监督任务学习 tool family embedding；
- 再把这个 embedding 固定，用于调度策略网络；
- 这样有助于编码不同设备族之间的差异。

#### 4. WIP-aware objective

目标函数不仅评价已完成 lot，还评价当前 WIP lot 的未来风险，以减少短视行为。

#### 5. 用 Natural Evolution Strategies 训练策略

避免直接对不平滑、离散的调度目标反向传播。

### 主要结果

- 在 SMT2020 的 HV/LM 和 LV/HM 场景中，都优于层级启发式；
- 在 Regular Lots 上改善尤其明显；
- 自监督 embedding 带来了有效的设备族信息表达；
- WIP-aware 目标使策略更适合连续 fab 环境。

### 对当前项目的启发

- 你未来可以为 Machine / PPID / Chamber 设计 embedding，而不是直接用 ID；
- 未来可考虑引入“自监督特征学习 + 进化式策略优化”；
- 你的 Q-time、priority、arrival、machine load 特征可进一步丰富成 attention 输入。

---

## 4.5 第 5 篇论文

**文件**：`lunwen/15/5.Deep reinforcement learning for.pdf`

**题目**：
**Deep reinforcement learning for scheduling semiconductor cluster tools in varying configurations**

### 研究问题

关注的是：

> **单台或单组 cluster tool 内部，VTM / ATM robot、process modules、load locks 等资源之间的微观调度。**

这与前面几篇 fab-level dispatching 不同，它更偏：

- wafer transfer scheduling；
- robot action scheduling；
- cluster tool 内部微观资源协调。

### 方法框架

#### Part 1：VTM 环境

- 1 个 dual-arm VTM robot；
- 6 个 PM；
- 2 个 LL；
- 所有模块在 vacuum 环境；
- 状态包括模块是否持 wafer、剩余时间、waiting time 等；
- 动作空间为 29 个 VTM 动作；
- 使用 DQN；
- 奖励包括 processed wafer reward、valid action reward、No-action penalty。

#### Part 2：VTM + ATM 环境

- 增加 ATM robot、load port、aligner；
- 显式考虑 vacuum / atmospheric state 切换；
- 引入 robot movement time；
- 使用 MADQN；
- 两个 robot 用不同 DQN policy。

### 关键方法点

1. **Action masking**
   - 所有非法动作 Q 值置为负无穷；
   - 保证 agent 只在合法动作中选。

2. **Generalization tests**
   - 测试 process time、cleaning、设备配置变化；
   - 测试单环境训练与跨环境测试。

3. **真实 cluster tool 结构建模**
   - 不再只模拟 VTM 和 PM；
   - 更接近实际 cluster equipment。

### 结果

- 单智能体 DQN 在 Part 1 中优于 rule-based scheduler；
- 多智能体 MADQN 在 Part 2 中表现稳健；
- 长训练在复杂环境中更有利于形成抽象策略；
- 最高 productivity 改进可达 **8.9%**。

### 对当前项目的启发

这篇对你的最大启发是：

- **action masking 非常重要**；
- 你的 Machine / PPID / Chamber 选择如果以后用 RL，必须显式设计合法动作掩码；
- Chamber/Side 层资源调度非常像 cluster tool 内部调度，可借鉴它的局部状态建模方式。

---

## 4.6 第 6 篇论文

**文件**：`lunwen/15/6.Scheduling of Automated Wet-Etch Stations with One Robot in.pdf`

**题目**：
**Scheduling of Automated Wet-Etch Stations with One Robot in Semiconductor Manufacturing via Constraint Answer Set Programming**

### 研究问题

研究带单机器人搬运的 AWS（Automated Wet-Etch Station）调度问题，关键约束包括：

- alternating chemical / water baths；
- robot transfer；
- zero-wait；
- local storage；
- bath capacity；
- makespan 最小化。

### 核心思路

- 用 **CASP / clingcon** 建模；
- 通过显式约束描述：
  - bath 上加工区间；
  - robot transfer 区间；
  - zero-wait/no intermediate storage；
  - overlap 检测；
  - 工艺顺序；
  - deadline 相关条件。

### 结果

- 普通 ASP 因 grounding bottleneck 无法处理较大问题；
- CASP 比 ASP 更适合大域调度约束；
- 与 MILP、CP+GVDR 相比，CASP 在 **best solution 的求解时间** 上表现更优；
- 在 50% case 中，CASP 拿到了更优或相同 makespan；
- 时间上通常快 1–2 个数量级。

### 对当前项目的启发

- 你的 Q-time 可以借鉴其 zero-wait 硬约束建模；
- 若未来加入 transfer/robot 资源，可直接参考其 bath+robot 双资源约束建模；
- 适合作为“显式约束层”文献，而当前项目则更偏“学习策略 + 显式解码”。

---

## 4.7 第 7 篇论文

**文件**：`lunwen/15/7.Machine Learning-based Dispatching for a Wet Clean.pdf`

**题目**：
**Machine Learning-based Dispatching for a Wet Clean Station in Semiconductor Manufacturing**

### 研究问题

在湿法清洗站中，调度器无法获得设备内部详细过程，只能看到：

- 每个 lot 的 track-in time；
- 每个 lot 的 track-out time；
- 当前 recipe 与已有 recipe 的组合关系。

论文研究的是：

> **在这种“内部过程不可见”的场景下，能否直接从历史日志中学习 dispatching surrogate，用来选择下一个最值得投放的 recipe。**

### 关键建模思想

#### 1. 引入 OPi

- `APi`：某个 recipe 序列集合的实际处理时间；
- `OPi`：该集合中 lot 同时处理所带来的 overlap processing time；
- `OPi` 越大，组合越优；
- 因此 dispatching 目标变成预测各候选 recipe 组合的 `OPi`。

#### 2. One-hot 序列输入

- 将长度为 `NS` 的 recipe 序列编码为 `NS × K` one-hot 矩阵；
- 再展平为长度 `NS × K` 的向量作为输入。

### 模型比较

论文比较：

- MLR；
- DNN；
- CNN1d；
- CNN3d。

### 主要结果

#### 1. MLR 不足

- 散点图表明 MLR 对 OPi 的拟合能力较差；
- 说明线性模型无法充分表达这种组合型 dispatching 问题。

#### 2. DNN / CNN 更适合

- DNN、CNN1d、CNN3d 预测更稳定；
- 当 bath 使用更复杂时，CNN 类模型更有优势；
- 当多数 recipe 都包含某个关键 bath 时，模型预测也更稳定。

#### 3. 单 lot 调度

- 与 same-recipe 和 random 两种 fab 中常见简单规则比较；
- DNN 和 CNN1d 在 `sc1` 中比 random 低约 35% 以上 makespan；
- sc3 中各方法接近，说明所有 recipe 都很像时调度差异不大；
- sc4/sc5 中高维模型更稳。

#### 4. 参数量差异

- DNN 参数很多；
- CNN1d 和 CNN3d 的参数量仅约为 DNN 的 10% 左右；
- 对于算力有限场景更有实际意义。

#### 5. Multiple-lot dispatching

- 一次选择多个后续 lot 可以减少排序时间；
- Pick 2 和 Pick 3 对 10000 lots 的排序时间仍然在可接受范围；
- 在真实 fab 中具有应用潜力。

### 对当前项目的启发

这是当前最值得你借鉴的“日志驱动 surrogate dispatching”论文：

- 如果某些设备内部过程不可见，只能拿到进出时间记录，就可以学习 surrogate；
- 你未来若遇到黑盒机台，也可以：
  - 用输入序列建模；
  - 预测组合效果；
  - 反过来做 dispatching；
- 这和你当前显式资源日历模型形成很好的互补：
  - 显式可见资源用日历解码；
  - 黑盒内部过程用 surrogate 近似。

---

## 5. 当前会话中的文档与文件变化

### 已更新文件

- `项目建模说明.md`
- `AGENT.md`

### 尚未修改但已在 IDE 中打开

- `rolling.py`
- `local_search.py`

### 当前未完成但可继续的方向

1. 继续整理 `lunwen/15/` 中后续论文；
2. 把论文 1–7 的方法与当前项目代码结构做逐一映射；
3. 针对 `rolling.py` 分析滚动调度策略与论文中的 dispatching 逻辑差异；
4. 针对 `local_search.py` 评估是否适合作为局部搜索增强模块；
5. 若开始实现下一步创新，建议优先：
   - 用学习模型替换 Lot scorer；
   - 保留当前解码器与约束校验；
   - 增强 Q-time 为硬约束或半硬约束；
   - 增加多场景/鲁棒性实验；
   - 后续补可解释性分析。

---

## 6. 对后续助手的建议

如果后续继续本项目，建议优先做以下工作之一：

### 方向 A：代码级分析

重点阅读：

- `rolling.py`
- `resource_calendar.py`
- `objectives.py`
- `problem.py`

目标：

- 明确当前滚动调度决策流；
- 明确当前策略权重如何作用于解码器；
- 明确 Q-time 与 frozen operations 在代码中的真实作用。

### 方向 B：论文方法映射

把论文 1–7 的关键思想映射到当前项目：

- RL / ES / CMA-ES；
- Attention lot scorer；
- Self-supervised embedding；
- XAI；
- action masking；
- CASP-like explicit constraints；
- log-driven surrogate dispatching。

### 方向 C：实现下一步创新

建议顺序：

1. 先做 **Lot 选择层的学习替换**；
2. 保持 Machine / PPID / Chamber 解码器不变；
3. 再扩展到更细粒度策略；
4. 最后补多场景鲁棒性、可解释性和 transfer / robot 约束。

---

## 7. 当前 AGENT.md 的用途

本文件用于：

- 保存当前对话中的主要结论；
- 记录项目现状与文档改动；
- 保存论文阅读进度；
- 为后续继续分析代码、写文献综述或实现算法提供上下文。