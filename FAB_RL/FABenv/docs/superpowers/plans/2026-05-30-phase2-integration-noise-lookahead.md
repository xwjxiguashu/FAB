# Phase 2 集成: 多头接入主循环 + 噪声注入 + 前瞻窗

> REQUIRED SUB-SKILL: superpowers:subagent-driven-development

**目标:** 三件事，全部"新增并存"，不破坏现有单头标量管线（保持现有测试通过）。
1. **终局字段**: sas_step 的 info 填充 is_terminal / num_lots / avg_machine_utilization / qtime_violation_count / completed_lots（向量奖励终局通道依赖这些）
2. **噪声注入（报告 §2.4.6）**: 环境每步可选采样实际加工时间 p_actual = μ + ε；agent 状态仍用 (μ,σ)，Q-time 判定与 reward 用采样值
3. **前瞻窗（报告 §2.1）**: 环境维护 W_lookahead，可见"即将到达 Lot"；全局特征加前瞻摘要
4. **多头 driver 接入**: driver 新增 run_multihead_policy_episode，喂 MultiHeadRolloutBuffer
5. **端到端训练**: train_phase2_sas_ppo.py 新增 multihead 训练路径

## 关键环境约束
- 真实解释器: `D:\Anaconda\python.exe`（本机 python 是 stub）
- 运行测试设 `$env:KMP_DUPLICATE_LIB_OK="TRUE"`，torch 测试与 scipy/matplotlib 测试分组跑，勿一次性 `pytest tests/`

## 接口契约（全程一致）

### 终局 info 字段（Task 1）
sas_step 成功提交那一步，info 增补：
```python
info["is_terminal"] = (len(self.remaining_lots) == 0)   # episode 是否结束
info["num_lots"] = int(self.encoder.num_lots)
info["completed_lots"] = len(self.completed_lots)
info["qtime_violation_count"] = <当前累计 q_time 违规数>   # encoder.compute_q_time_violation(wafer_schedule)[0]
info["avg_machine_utilization"] = <平均机台利用率>          # 复用 evaluate_objectives 的利用率口径或简化
```
非终局步这些字段也填（is_terminal=False，其余照常算），向量奖励函数只在 is_terminal=True 时用 soft 通道。

### 噪声注入（Task 2）— ResourceCalendarEnv 新增可选开关
```python
ResourceCalendarEnv(..., process_noise_enabled=False, noise_seed=None)
```
- 默认 False → 完全等同现有确定性行为（现有测试不受影响）
- True 时：commit 阶段对每个 (wafer, stage) 的 process_time 采样 p_actual = μ + N(0,σ)，σ 来自 encoder.process_time_sigma；clip 到 ≥1e-6
- 候选特征 / dry-run 仍用 μ（agent 看 μ,σ；执行用采样值）—— 报告"规划用 μ、执行后用实际值校正"
- 用独立 np.random.Generator(noise_seed) 保证可复现

### 前瞻窗（Task 3）— ResourceCalendarEnv 新增
```python
ResourceCalendarEnv(..., w_lookahead=0.0)   # 默认 0 → 无前瞻，等同现有
def visible_lots(self):  # 已到达 + [t_now, t_now+w_lookahead] 内即将到达
def lookahead_summary(self):  # dict: upcoming_count, max_priority, min_remaining_qtime, earliest_eta
```
- 全局特征从 9 维 → 13 维（新增 4: upcoming_count_norm, lookahead_max_priority, lookahead_min_qtime, lookahead_earliest_eta）
- 用新方法 `build_global_features_v2`，保留旧 9 维方法不动；ObservationEncoder 增 `lookahead=False` 开关，默认走旧 9 维

### 多头 driver（Task 4）
```python
driver.run_multihead_policy_episode(policy, buffer, stochastic=True, reward_vector_config=None)
```
- 用 compute_sas_reward_vector 算向量奖励
- buffer 是 MultiHeadRolloutBuffer
- policy 是 Phase2SASMultiHeadActorCritic
- step 存 reward_vector + values(dict)
- 新增 _add_multihead_rollout_step 辅助
- 不动现有 run_policy_episode

### 端到端训练（Task 5）
train_phase2_sas_ppo.py 新增 `build_multihead_training_components()` + mode="multihead" 路径，用 MultiHeadPPOTrainer。

---

### Task 1: sas_step 终局字段填充
**文件:** rl_environment.py（改 sas_step 成功分支的 info）+ tests/test_phase2_terminal_info.py
- 测试: 成功提交步 info 含 is_terminal/num_lots/completed_lots/qtime_violation_count/avg_machine_utilization
- 最后一个 lot 完成时 is_terminal=True
- 不破坏现有 sas_step 测试

### Task 2: 噪声注入
**文件:** rl_environment.py（__init__ 加参数 + _simulate_action 采样）+ tests/test_phase2_noise_injection.py
- 测试: process_noise_enabled=False 时结果确定（同 seed 两次一致 + 等于现有行为）
- True 时两个不同 noise_seed 产生不同 wafer_schedule 时间
- True 时同 noise_seed 可复现
- 候选特征仍用 μ（不受噪声影响）

### Task 3: 前瞻窗
**文件:** rl_environment.py（visible_lots/lookahead_summary）+ phase2_sas_observation.py（build_global_features_v2 + lookahead 开关）+ tests/test_phase2_lookahead.py
- 测试: w_lookahead=0 时 visible_lots 只含已到达
- w_lookahead>0 时含窗内即将到达
- lookahead_summary 字段正确
- ObservationEncoder(lookahead=True) 产 13 维全局特征；默认仍 9 维

### Task 4: 多头 driver 接入
**文件:** phase2_sas_driver.py（run_multihead_policy_episode + _add_multihead_rollout_step）+ tests/test_phase2_multihead_driver.py
- 测试: 跑一个 multihead episode，buffer 收到 MultiHeadRolloutStep，reward_vector shape (4,)，values 含 4 通道
- 不破坏现有 driver 测试

### Task 5: 端到端 multihead 训练
**文件:** train_phase2_sas_ppo.py（build_multihead_training_components + mode 路径）+ tests/test_phase2_multihead_train_smoke.py
- 测试: build_multihead_training_components 返回多头 policy + MultiHeadPPOTrainer；跑 1-2 episode 不报错，loss 字段齐全
