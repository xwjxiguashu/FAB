# 派工规则基线 + 多 seed 评测 (报告 §7.4 / §4.10 / §2.4.6)

> 论文实验体系的第一步：建立对照基线与统计严谨的评测协议，回答"当前 SAS-PPO
> 相对启发式到底赢没赢"。在投入建 DDT 之前先用数据确认项目有没有戏。

**状态: 已完成 (2026-05-31)。** 8 个基线测试 + 29 项相关回归全绿。

## 交付物
- `Phase2EpisodeDriver.run_rule_episode(strategy)`: 支持 first_valid/FIFO/SPT/EDD/CR/ATC。
  - `_rule_action_index()`: 在与 RL 相同的 qtime-safe 候选池上按规则排序 (公平对比)，
    proc 用下层估时器相对 makespan μ (走缓存)。
- `evaluate_baselines.py`: 多策略 × 多 seed 评测 + 聚合 (mean/std) + 对比表 + CLI。
  - seed = 一次加工噪声实现 (process_noise_enabled + noise_seed)。
  - 指标: Q-time/拖期违规、利用率、优先级违反 (来自 encoder.evaluate_objectives)。
  - 可选 `--checkpoint` 纳入 RL 贪心对比。
- `tests/test_baselines.py` (8 个): 各规则跑完且排程完整、未知策略报错、指标提取、多 seed 聚合。

## 首个发现 (决策相关)
small(4 lots) 实例**无法区分策略**：FIFO/SPT/EDD/CR/ATC 全部 0 违规、利用率 0.794、
全完成。→ 论文数据必须用更难/更大的实例 (pressure 50 lots 或调参实例)，
而 pressure 单 episode 仍慢 (commit/wafer 仿真 ~2.8s/步) → 与"性能优化"一步耦合。

## 后续
- 在 pressure 或新建的"有区分度"实例上跑出差异化对比表。
- 训练出 SAS-PPO 检查点后用 `--checkpoint` 纳入对比，得到第一张 RL vs 启发式表。
- 接 SMT2020 (报告 §7.4) 做可横向比较的基线。
