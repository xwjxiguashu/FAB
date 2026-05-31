# Phase 2 R1: 向量奖励 + 多头 Critic + 逐目标 GAE

> REQUIRED SUB-SKILL: superpowers:subagent-driven-development

**目标:** 实现项目报告第4章基础方案 R1 的核心机制：奖励向量化、Critic 多头、逐目标 GAE。采用"新增并存"策略，不破坏现有单头标量管线（保持 51 个测试通过）。

## 接口契约（所有任务必须严格遵守）

### 通道定义（报告 §4.5 R1，四通道）

```python
REWARD_CHANNELS = ("exec", "qtime", "util", "progress")  # 顺序固定，全程一致
```

- `exec`: 即时密集，每步非零。+0.20 成功 / -0.40 插入失败 / -0.50 mask_invalid / 0.0 wait
- `qtime`: 终局，仅 episode 末步非零。= -norm(qtime_violation_count / num_lots)。硬约束残差(=拖期)
- `util`: 终局，仅末步非零。= +norm(avg_machine_utilization)。唯一软目标
- `progress`: 终局，仅末步非零。= +norm(completed_lots / num_lots)。利用率代理（计数制）

向量 = `np.array([r_exec, r_qtime, r_util, r_progress], dtype=float)`，shape (4,)。
各通道独立，不跨通道求和、不跨通道 clip。

### 优势加权（报告 §4.7）

```
A_t = w_exec·norm(Â_exec) + w_util·norm(Â_util) + w_progress·norm(Â_progress) - w_qtime·norm(Â_qtime)
```
注意 qtime 是 cost，做减项（其奖励本身为负，norm 后用减号是因为权重表达"硬约束优先级"）。
实际上 r_qtime 已是负值，统一处理：`A = Σ_k sign_k · w_k · norm(Â_k)`，其中 qtime 的 sign 由报告减项语义决定。
简化实现：advantage 加权时 `A = w_exec·n(Â_exec) + w_util·n(Â_util) + w_progress·n(Â_progress) + w_qtime·n(Â_qtime)`，
因 r_qtime 已含负号，w_qtime 取正值即可（避免双重取负）。**实现时统一为加法，靠 reward 符号体现 cost。**

---

### Task A: rl_environment.py — 向量奖励

**新增（不改现有 compute_sas_reward / RewardConfig）:**

1. `RewardVectorConfig` dataclass:
```python
@dataclass
class RewardVectorConfig:
    insert_success_reward: float = 0.20
    insert_fail_penalty: float = -0.40
    mask_invalid_penalty: float = -0.50
    # 终局通道权重（作用在归一化 advantage 上，由 trainer 使用）
    w_exec: float = 1.0
    w_qtime: float = 3.0    # 大值，硬约束优先
    w_util: float = 0.5
    w_progress: float = 0.3
    channels: tuple = ("exec", "qtime", "util", "progress")
```

2. `compute_sas_reward_vector(info, config=None) -> dict`:
   - 返回 `{"reward_vector": np.array([4]), "r_exec":..., "r_qtime":..., "r_util":..., "r_progress":...}`
   - exec 通道：mask_invalid→penalty；wait_or_noop→0.0；insertion_failed→fail_penalty；insertion_success→success_reward
   - 终局通道（仅当 info["is_terminal"]=True 时非零）：
     - r_qtime = -(info["qtime_violation_count"] / max(info["num_lots"],1))
     - r_util  = +info["avg_machine_utilization"]
     - r_progress = +(info["completed_lots"] / max(info["num_lots"],1))
   - 非终局步：qtime/util/progress 三通道为 0.0

**测试 `tests/test_phase2_reward_vector.py`:**
- 成功步：reward_vector[0]==0.20，其余==0
- 失败步：reward_vector[0]==-0.40
- wait 步：reward_vector[0]==0.0
- 终局步（含 qtime_violation_count/num_lots/util/completed）：通道 1,2,3 非零且符号正确
- reward_vector shape == (4,)

### Task B: phase2_sas_policy.py — 多头 Critic

**新增 `Phase2SASMultiHeadActorCritic`（不改现有 Phase2SASActorCritic）:**
- 共享 candidate_encoder + actor_head（与单头相同）
- critic 改为 `nn.ModuleDict`，4 个独立 value 头（exec/qtime/util/progress），各自 `Linear(hidden+global → hidden → 1)`
- `critic_values(...)` 返回 dict{channel: value tensor}
- `evaluate_actions(...)` 返回 `{"log_prob", "entropy", "values": dict, "probs"}`
- `sample_action`/`greedy_action` 的 "value" 改为 "values" dict

**测试 `tests/test_phase2_multihead_policy.py`（torch，环境若无 torch 跳过）:**
- 4 个 value 头输出 shape (batch,)
- evaluate_actions 返回 values dict 含 4 通道

### Task C: phase2_ppo_buffer.py — 逐通道 GAE

**新增 `MultiHeadRolloutStep` + `MultiHeadRolloutBuffer`（不改现有）:**
- step 存 `reward_vector`(4,), `values`(dict 4通道)
- `compute_returns_and_advantages(last_values: dict)`：对每个通道独立跑 GAE
- 产出 `advantages: dict{channel: list}`, `returns: dict{channel: list}`

**测试 `tests/test_phase2_multihead_buffer.py`（纯 numpy，可测）:**
- 单通道退化时与现有 GAE 数值一致
- 4 通道各自独立计算

### Task D: phase2_ppo_trainer.py — 多头 PPO

**新增 `MultiHeadPPOTrainer`（不改现有）:**
- 逐通道归一化 advantage 后加权求和得 A_t（按契约公式）
- value loss = Σ_k c_k·MSE(V_k, R_k)
- actor loss 用加权 A_t

**测试（torch，环境若无 torch 跳过）:** smoke test 能跑一步 update。
