"""Phase 2 PPO Rollout Buffer — 存储 episode 轨迹并计算 GAE 回报与优势。

RolloutStep: 单步轨迹记录 (观察→动作→奖励→下一观察)
StepInfo:   步骤的详细奖励/惩罚分解
Phase2RolloutBuffer: 存储完整 episode 轨迹，提供 GAE 计算和批次化训练数据生成
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

import numpy as np


@dataclass
class StepInfo:
    """单步的详细分解信息 — 用于调试和奖励塑形分析。

    包含: 选中的 Lot/PPID, 插入状态 (成功/失败/掩码/等待),
          时间信息 (开始/结束/加工时间), Q-time 新增违反,
          优先级排名惩罚, 以及完整的奖励分量分解。
    """
    selected_lot: Optional[int] = None
    selected_ppid: Optional[int] = None
    insertion_success: bool = False
    insertion_failed: bool = False
    mask_invalid: bool = False
    wait_or_noop: bool = False
    selected_lot_start: Optional[float] = None
    selected_lot_end: Optional[float] = None
    selected_lot_process_time: Optional[float] = None
    new_qtime_violation: float = 0.0
    priority_rank_penalty: float = 0.0
    # 奖励分量
    reward_execute: float = 0.0
    reward_wait: float = 0.0
    reward_tardy: float = 0.0
    reward_qtime: float = 0.0
    reward_priority: float = 0.0
    reward_progress: float = 0.0
    reward_shape: float = 0.0
    reward_terminal: float = 0.0
    reward_total: float = 0.0
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RolloutStep:
    """单步 rollout 记录 — PPO 训练的最小数据单元。

    Attributes:
        machine_id: 决策机台。
        current_time: 决策时刻。
        candidate_features: (pool_size, 18) 候选特征。
        candidate_mask: (pool_size,) bool 掩码。
        global_features: (9,) 全局特征。
        action_indices: (pool_size,) 动作索引。
        valid_action_count: 有效动作数。
        action: 实际选择的动作索引。
        log_prob: 策略在该动作上的对数概率。
        value: Critic 的状态价值估计。
        reward: 环境返回的即时奖励。
        done: 该步后 episode 是否结束。
        next_observation: Phase2Observation 或 None (终态)。
        info: StepInfo 详细分解。
    """
    machine_id: int
    current_time: float
    candidate_features: object
    candidate_mask: object
    global_features: object
    action_indices: object
    valid_action_count: int
    action: int
    log_prob: float
    value: float
    reward: float
    done: bool
    next_observation: Optional[object]
    info: StepInfo


class Phase2RolloutBuffer:
    """Phase 2 Rollout Buffer — 存储轨迹并计算 GAE。

    GAE (Generalized Advantage Estimation):
      delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
      A_t = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}
      R_t = A_t + V(s_t)

    支持:
      - add(): 添加一步
      - finish_episode(): 计算 GAE 回报和优势
      - get_training_batches(): 批次化训练数据生成
      - clear(): 清空 buffer
    """

    def __init__(self, gamma=0.99, gae_lambda=0.95):
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.steps = []
        self.returns = []
        self.advantages = []

    def add(self, step):
        """添加一步 rollout 记录。"""
        self.steps.append(step)

    def finish_episode(self, last_value=0.0):
        """计算 GAE 回报和优势函数。

        Args:
            last_value: 终态的 Value 估计 (默认 0.0)。
        """
        self.compute_returns_and_advantages(last_value=last_value)

    def compute_returns_and_advantages(self, last_value=0.0):
        """从后向前递推计算 GAE。

        GAE 递推公式:
          delta = r + gamma * V_next * non_terminal - V
          gae   = delta + gamma * lambda * non_terminal * gae_prev
          A     = gae
          R     = gae + V
        """
        returns = []
        advantages = []
        gae = 0.0
        next_value = float(last_value)
        for step in reversed(self.steps):
            non_terminal = 0.0 if step.done else 1.0
            delta = step.reward + self.gamma * next_value * non_terminal - step.value
            gae = delta + self.gamma * self.gae_lambda * non_terminal * gae
            advantages.append(gae)
            returns.append(gae + step.value)
            next_value = step.value
        self.advantages = list(reversed(advantages))
        self.returns = list(reversed(returns))

    def get_training_batches(self, batch_size):
        """生成批次化的训练数据。

        将存储的 steps 按 batch_size 切分为多个批次，
        每批次包含 numpy 数组形式的 features, masks, actions, rewards 等。

        Yields:
            dict: 每批次训练数据的 numpy 字典。
        """
        if not self.steps:
            return
        if not self.returns or not self.advantages:
            self.finish_episode(last_value=0.0)
        batch_size = max(1, int(batch_size))
        indices = list(range(len(self.steps)))
        for start in range(0, len(indices), batch_size):
            selected = indices[start:start + batch_size]
            yield {
                "machine_id": np.asarray(
                    [self.steps[index].machine_id for index in selected],
                    dtype=np.int64,
                ),
                "current_time": np.asarray(
                    [self.steps[index].current_time for index in selected],
                    dtype=np.float32,
                ),
                "candidate_features": np.stack(
                    [self.steps[index].candidate_features for index in selected]
                ).astype(np.float32),
                "candidate_mask": np.stack(
                    [self.steps[index].candidate_mask for index in selected]
                ).astype(bool),
                "global_features": np.stack(
                    [self.steps[index].global_features for index in selected]
                ).astype(np.float32),
                "action_indices": np.stack(
                    [self.steps[index].action_indices for index in selected]
                ).astype(np.int64),
                "valid_action_count": np.asarray(
                    [self.steps[index].valid_action_count for index in selected],
                    dtype=np.int64,
                ),
                "actions": np.asarray(
                    [self.steps[index].action for index in selected],
                    dtype=np.int64,
                ),
                "old_log_probs": np.asarray(
                    [self.steps[index].log_prob for index in selected],
                    dtype=np.float32,
                ),
                "values": np.asarray(
                    [self.steps[index].value for index in selected],
                    dtype=np.float32,
                ),
                "rewards": np.asarray(
                    [self.steps[index].reward for index in selected],
                    dtype=np.float32,
                ),
                "dones": np.asarray(
                    [self.steps[index].done for index in selected],
                    dtype=bool,
                ),
                "returns": np.asarray(
                    [self.returns[index] for index in selected],
                    dtype=np.float32,
                ),
                "advantages": np.asarray(
                    [self.advantages[index] for index in selected],
                    dtype=np.float32,
                ),
            }

    def clear(self):
        """清空所有存储的步骤、回报和优势。"""
        self.steps.clear()
        self.returns.clear()
        self.advantages.clear()


# 固定四通道顺序（报告 §4.7）
MULTIHEAD_CHANNELS = ("exec", "qtime", "util", "progress")


@dataclass
class MultiHeadRolloutStep:
    """单步 rollout 数据（多头逐通道版本）。

    字段与 RolloutStep 相同，但使用：
      - values: Dict[str, float]   （替代 value: float）每个通道独立 critic 价值。
      - reward_vector: object      （替代 reward: float）每个通道独立奖励向量。
    """
    machine_id: int
    current_time: float
    candidate_features: object
    candidate_mask: object
    global_features: object
    action_indices: object
    valid_action_count: int
    action: int
    log_prob: float
    values: Dict[str, float]
    reward_vector: object
    done: bool
    next_observation: Optional[object] = None
    info: Optional[StepInfo] = None


class MultiHeadRolloutBuffer:
    """多头逐通道 GAE Rollout Buffer。

    对每个通道 k 独立计算 GAE（报告 §4.7）:
      delta_t^k = r_t^k + gamma * V_k(s_{t+1}) * (1 - done) - V_k(s_t)
      A_t^k     = delta_t^k + gamma * lambda * (1 - done) * A_{t+1}^k
      R_t^k     = A_t^k + V_k(s_t)
    """

    def __init__(self, gamma=0.99, gae_lambda=0.95, channels=MULTIHEAD_CHANNELS):
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.channels = tuple(channels)
        self.steps: List[MultiHeadRolloutStep] = []
        self.advantages: Dict[str, List[float]] = {c: [] for c in self.channels}
        self.returns: Dict[str, List[float]] = {c: [] for c in self.channels}

    def add(self, step):
        """添加一步 rollout 记录。"""
        self.steps.append(step)

    def finish_episode(self, last_values: Optional[Dict[str, float]] = None):
        """计算每通道 GAE 优势与回报。

        Args:
            last_values: 每个通道终态的 Value 估计 (默认全 0)。
        """
        self.compute_returns_and_advantages(last_values)

    def compute_returns_and_advantages(self, last_values: Optional[Dict[str, float]] = None):
        """对每个通道独立从后向前递推计算 GAE。"""
        if last_values is None:
            last_values = {c: 0.0 for c in self.channels}

        n = len(self.steps)
        self.advantages = {c: [0.0] * n for c in self.channels}
        self.returns = {c: [0.0] * n for c in self.channels}
        if n == 0:
            return

        for c in self.channels:
            k = self.channels.index(c)
            gae = 0.0
            next_value = float(last_values.get(c, 0.0))
            for t in reversed(range(n)):
                step = self.steps[t]
                reward = float(np.asarray(step.reward_vector)[k])
                value = float(step.values[c])
                non_terminal = 0.0 if step.done else 1.0
                delta = reward + self.gamma * next_value * non_terminal - value
                gae = delta + self.gamma * self.gae_lambda * non_terminal * gae
                self.advantages[c][t] = gae
                self.returns[c][t] = gae + value
                next_value = value

    def get_training_batches(self, batch_size) -> Iterator[Dict[str, Any]]:
        """生成批次化的训练数据（多头版本）。

        Yields:
            dict: 含与标量版相同的字段
                (machine_id/current_time/candidate_features/candidate_mask/
                 global_features/action_indices/valid_action_count/actions/
                 old_log_probs/dones)，外加每通道
                 values_<c>/advantages_<c>/returns_<c>。
        """
        if not self.steps:
            return
        n = len(self.steps)
        if not any(len(v) == n for v in self.advantages.values()):
            self.finish_episode()
        batch_size = max(1, int(batch_size))
        indices = list(range(n))
        for start in range(0, n, batch_size):
            selected = indices[start:start + batch_size]
            batch = {
                "machine_id": np.asarray(
                    [self.steps[i].machine_id for i in selected], dtype=np.int64,
                ),
                "current_time": np.asarray(
                    [self.steps[i].current_time for i in selected], dtype=np.float32,
                ),
                "candidate_features": np.stack(
                    [self.steps[i].candidate_features for i in selected]
                ).astype(np.float32),
                "candidate_mask": np.stack(
                    [self.steps[i].candidate_mask for i in selected]
                ).astype(bool),
                "global_features": np.stack(
                    [self.steps[i].global_features for i in selected]
                ).astype(np.float32),
                "action_indices": np.stack(
                    [self.steps[i].action_indices for i in selected]
                ).astype(np.int64),
                "valid_action_count": np.asarray(
                    [self.steps[i].valid_action_count for i in selected], dtype=np.int64,
                ),
                "actions": np.asarray(
                    [self.steps[i].action for i in selected], dtype=np.int64,
                ),
                "old_log_probs": np.asarray(
                    [self.steps[i].log_prob for i in selected], dtype=np.float32,
                ),
                "dones": np.asarray(
                    [self.steps[i].done for i in selected], dtype=bool,
                ),
            }
            for c in self.channels:
                batch[f"values_{c}"] = np.asarray(
                    [float(self.steps[i].values[c]) for i in selected], dtype=np.float32,
                )
                batch[f"advantages_{c}"] = np.asarray(
                    [self.advantages[c][i] for i in selected], dtype=np.float32,
                )
                batch[f"returns_{c}"] = np.asarray(
                    [self.returns[c][i] for i in selected], dtype=np.float32,
                )
            yield batch

    def clear(self):
        """清空所有存储的步骤、每通道回报与优势。"""
        self.steps = []
        self.advantages = {c: [] for c in self.channels}
        self.returns = {c: [] for c in self.channels}

    def __len__(self):
        return len(self.steps)