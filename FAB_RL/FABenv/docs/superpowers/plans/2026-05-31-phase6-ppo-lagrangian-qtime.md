# Phase 6 (部分): PPO-Lagrangian 处理 Q-time 残差约束

> 实现项目报告_完善版.md §3.3：把 Q-time 硬约束建模为 CMDP，用 PPO-Lagrangian
> 自适应 λ 对偶上升处理 mask 挡不住的残差违规（窗外到达 + 机会约束尾部 + doomed lot）。
> 采用"新增并存"策略，默认关闭（`use_qtime_lagrangian=False`），不改变现有固定 w_qtime 行为。

**状态: 已完成 (2026-05-31)。** 全部测试通过，端到端 `--mode multihead --qtime-lagrangian` 可跑通。

## 接口契约

- 违规率 (cost) = qtime 通道终局奖励的相反数：`cost = -Σ_t reward_vector[qtime]`
  （`r_qtime = -violation_count/num_lots`，仅末步非零）。
- 对偶上升：`λ ← clip(max(0, λ + η_λ·(violation − ε)), 0, λ_max)`。
- qtime 通道权重：启用时用 `self.lambda_qtime`，否则用固定 `w_qtime`。二者不并用。

## 完成的改动

### MultiHeadPPOConfig (phase2_ppo_trainer.py)
- [x] 新增字段 `use_qtime_lagrangian=False, qtime_lambda_init=0.0, qtime_cost_budget=0.0,
      qtime_lambda_lr=0.05, qtime_lambda_max=1e3`

### MultiHeadPPOTrainer (phase2_ppo_trainer.py)
- [x] `__init__` 初始化 `self.lambda_qtime = config.qtime_lambda_init`
- [x] `qtime_weight()` — 启用返回 λ，否则返回固定 w_qtime
- [x] `update_lambda(mean_violation)` — 对偶上升，禁用时 no-op
- [x] `episode_qtime_cost(buffer)` — 从 buffer 提取违规率
- [x] `_combine_advantages` — qtime 通道改用 `qtime_weight()`
- [x] `train()` — 每 episode 更新前算 cost、更新后对偶上升；history 增 `qtime_cost`/`lambda_qtime`

### train_phase2_sas_ppo.py
- [x] `build_multihead_training_components` 透传 Lagrangian 参数
- [x] `main` + CLI: `--qtime-lagrangian / --qtime-budget / --qtime-lambda-lr`

### 测试 tests/test_phase2_qtime_lagrangian.py (10 个，全绿)
- [x] config 默认关闭
- [x] qtime_weight 在启用/禁用下分别返回 λ / w_qtime
- [x] 对偶上升：超预算增大、低于预算减小并 clamp 0、clamp 上限、禁用时 no-op
- [x] episode_qtime_cost 提取与无违规归零
- [x] torch smoke：train 路径跑通、违规超预算后 λ 增大

## 诚实边界 (报告 §3.3.5)
- 保证的是期望约束 `E[违规] ≤ ε`，非"每次不违规"（与 §2.4.5 自洽）。
- 深度非凸下无严格收敛保证；λ 可能震荡，`η_λ` 须 ≪ 策略 lr，必要时上 PID-Lagrangian。

## 后续 (未做)
- 多 rollout 的 `Ê[violation]` 平滑 (当前单 episode/更新)。
- 与 train_with_driver_factory (random 课程) 的集成。
- PID-Lagrangian 稳定化 (Stooke et al. 2020)。
