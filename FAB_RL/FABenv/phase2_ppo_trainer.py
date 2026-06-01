"""Phase 2 PPO Trainer — 完整的 PPO 训练循环。

PPOConfig: PPO 超参数配置
Phase2PPOTrainer: 实现 PPO 更新逻辑
  - 收集 episode (collect_episode)
  - 策略损失: clipped PPO 目标 L = -min(ratio*A, clip(ratio, 1-eps, 1+eps)*A)
  - 价值损失: MSE(V, returns)
  - 熵奖励: 鼓励探索
  - 梯度裁剪 + Adam 优化
  - 支持 minibatch 迭代和多 epoch 更新
"""

from dataclasses import dataclass

import numpy as np

try:  # torch 可能在某些环境下损坏 — 纯 numpy 部分 (如 combine_channel_advantages) 仍可使用
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover - 仅在 torch 损坏环境触发
    torch = None
    F = None

from phase2_ppo_buffer import MultiHeadRolloutBuffer, Phase2RolloutBuffer


@dataclass
class PPOConfig:
    """PPO 训练超参数。

    Attributes:
        gamma: 折扣因子 (默认 0.99)。
        gae_lambda: GAE lambda 参数 (默认 0.95)。
        clip_ratio: PPO 裁剪范围 epsilon (默认 0.2)。
        value_coef: 价值损失系数 (默认 0.5)。
        entropy_coef: 熵奖励系数 (默认 0.01)。
        learning_rate: Adam 学习率 (默认 3e-4)。
        train_epochs: 每轮收集后的更新 epoch 数 (默认 4)。
        minibatch_size: 小批次大小 (默认 32)。
        max_grad_norm: 梯度裁剪范数上限 (默认 0.5)。
    """
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    learning_rate: float = 3e-4
    train_epochs: int = 4
    minibatch_size: int = 32
    max_grad_norm: float = 0.5


class Phase2PPOTrainer:
    """Phase 2 PPO 训练器。

    训练流程 (每 episode):
      1. driver.run_policy_episode() → 收集完整 episode 轨迹到 buffer
      2. buffer.finish_episode() → 计算 GAE 回报和优势
      3. update_policy() → 多 epoch minibatch PPO 更新
         - 优势归一化 (batch size > 1 时)
         - clipped PPO 策略损失
         - MSE 价值损失
         - 熵奖励
         - 梯度裁剪 + Adam step
    """

    def __init__(self, policy, optimizer, config):
        self.policy = policy
        self.optimizer = optimizer
        self.config = config

    def _device(self):
        """获取策略网络所在的 torch device。"""
        try:
            return next(self.policy.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _tensor_batch(self, batch):
        """将 numpy 批次数据转换为 torch tensor (自动移至策略设备)。"""
        device = self._device()
        return {
            "candidate_features": torch.as_tensor(
                batch["candidate_features"],
                dtype=torch.float32,
                device=device,
            ),
            "candidate_mask": torch.as_tensor(
                batch["candidate_mask"],
                dtype=torch.bool,
                device=device,
            ),
            "global_features": torch.as_tensor(
                batch["global_features"],
                dtype=torch.float32,
                device=device,
            ),
            "actions": torch.as_tensor(
                batch["actions"],
                dtype=torch.long,
                device=device,
            ),
            "old_log_probs": torch.as_tensor(
                batch["old_log_probs"],
                dtype=torch.float32,
                device=device,
            ),
            "returns": torch.as_tensor(
                batch["returns"],
                dtype=torch.float32,
                device=device,
            ),
            "advantages": torch.as_tensor(
                batch["advantages"],
                dtype=torch.float32,
                device=device,
            ),
        }

    def _collate(self, buffer):
        """将整个 buffer 的所有 step 整理为单个批次。"""
        return self._tensor_batch({
            "candidate_features": np.stack([step.candidate_features for step in buffer.steps]),
            "candidate_mask": np.stack([step.candidate_mask for step in buffer.steps]),
            "global_features": np.stack([step.global_features for step in buffer.steps]),
            "actions": np.asarray([step.action for step in buffer.steps], dtype=np.int64),
            "old_log_probs": np.asarray([step.log_prob for step in buffer.steps], dtype=np.float32),
            "returns": np.asarray(buffer.returns, dtype=np.float32),
            "advantages": np.asarray(buffer.advantages, dtype=np.float32),
        })

    def collect_episode(self, driver, buffer, stochastic=True):
        """收集一个完整的 episode 轨迹。"""
        return driver.run_policy_episode(self.policy, buffer=buffer, stochastic=stochastic)

    def _evaluate_batch(self, batch):
        """对批次数据运行策略评估 (用于计算损失)。"""
        return self.policy.evaluate_actions(
            batch["candidate_features"],
            batch["candidate_mask"],
            batch["global_features"],
            batch["actions"],
        )

    def compute_policy_loss(self, batch):
        """计算 clipped PPO 策略损失。

        L_CLIP = -mean(min(ratio * A, clip(ratio, 1-eps, 1+eps) * A))
        其中 ratio = exp(new_log_prob - old_log_prob)
        """
        output = self._evaluate_batch(batch)
        advantages = batch["advantages"]
        # 优势归一化 (batch size > 1 时)
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        ratio = torch.exp(output["log_prob"] - batch["old_log_probs"])
        unclipped = ratio * advantages
        clipped = torch.clamp(
            ratio,
            1.0 - self.config.clip_ratio,
            1.0 + self.config.clip_ratio,
        ) * advantages
        return -torch.min(unclipped, clipped).mean()

    def compute_value_loss(self, batch):
        """计算价值函数 MSE 损失: MSE(V(s), returns)。"""
        output = self._evaluate_batch(batch)
        return F.mse_loss(output["value"], batch["returns"])

    def compute_entropy_bonus(self, batch):
        """计算策略熵 (用于鼓励探索)。"""
        output = self._evaluate_batch(batch)
        return output["entropy"].mean()

    def _minibatches(self, buffer):
        """将 buffer 切分为 minibatch 生成器。

        优先使用 buffer.get_training_batches() (GPU memory efficient)，
        否则回退到单批次 (collate)。
        """
        if hasattr(buffer, "get_training_batches"):
            yield from buffer.get_training_batches(self.config.minibatch_size)
            return
        # 回退: 整个 buffer 作为一个批次
        yield {
            "candidate_features": np.stack([step.candidate_features for step in buffer.steps]),
            "candidate_mask": np.stack([step.candidate_mask for step in buffer.steps]),
            "global_features": np.stack([step.global_features for step in buffer.steps]),
            "actions": np.asarray([step.action for step in buffer.steps], dtype=np.int64),
            "old_log_probs": np.asarray([step.log_prob for step in buffer.steps], dtype=np.float32),
            "returns": np.asarray(buffer.returns, dtype=np.float32),
            "advantages": np.asarray(buffer.advantages, dtype=np.float32),
        }

    def update_policy(self, buffer):
        """执行 PPO 策略更新。

        流程:
          1. 若 buffer 为空，返回零统计
          2. 若未计算 GAE，先计算
          3. 对 train_epochs 轮，遍历 minibatches:
             - 计算策略损失 + 价值损失 + 熵奖励
             - 总损失 = policy_loss + value_coef * value_loss - entropy_coef * entropy
             - 反向传播 + 梯度裁剪 + optimizer.step()
          4. 返回最后一轮的 loss 统计

        Returns:
            {"policy_loss", "value_loss", "entropy"}
        """
        if not buffer.steps:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        if not buffer.returns or not buffer.advantages:
            buffer.finish_episode(last_value=0.0)

        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        for _ in range(self.config.train_epochs):
            for raw_batch in self._minibatches(buffer):
                batch = self._tensor_batch(raw_batch)
                output = self._evaluate_batch(batch)

                # 优势归一化
                advantages = batch["advantages"]
                if advantages.numel() > 1:
                    advantages = (
                        (advantages - advantages.mean())
                        / (advantages.std(unbiased=False) + 1e-8)
                    )

                # PPO clipped 损失
                ratio = torch.exp(output["log_prob"] - batch["old_log_probs"])
                unclipped = ratio * advantages
                clipped = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_ratio,
                    1.0 + self.config.clip_ratio,
                ) * advantages
                policy_loss = -torch.min(unclipped, clipped).mean()

                # 价值损失 + 熵
                value_loss = F.mse_loss(output["value"], batch["returns"])
                entropy = output["entropy"].mean()

                # 总损失 = 策略损失 + 价值损失 - 熵奖励
                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy
                )

                # 优化步骤
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    self.config.max_grad_norm,
                )
                self.optimizer.step()

                stats = {
                    "policy_loss": float(policy_loss.detach().cpu()),
                    "value_loss": float(value_loss.detach().cpu()),
                    "entropy": float(entropy.detach().cpu()),
                }
        return stats

    def train(self, driver, num_episodes, episode_logger=None):
        """完整的 PPO 训练循环。

        对每 episode:
          1. 重置 driver → 收集 episode → 计算 GAE → 更新策略
          2. 记录历史 (episode 摘要 + loss 统计)

        Returns:
            history: list[dict] 每 episode 的摘要和 loss。
        """
        history = []
        for episode in range(int(num_episodes)):
            driver.reset_episode()
            buffer = Phase2RolloutBuffer(
                gamma=self.config.gamma,
                gae_lambda=self.config.gae_lambda,
            )
            summary = self.collect_episode(driver, buffer, stochastic=True)
            if buffer.steps:
                buffer.finish_episode(last_value=0.0)
                stats = self.update_policy(buffer)
            else:
                stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
            row = {"episode": episode, **summary, **stats}
            history.append(row)
            if episode_logger is not None:
                episode_logger.log(row)
        return history

    def train_with_driver_factory(self, driver_factory, num_episodes, episode_logger=None):
        """PPO 训练循环，每个 episode 由 factory 构建新的 driver。"""
        history = []
        for episode in range(int(num_episodes)):
            factory_result = driver_factory(episode)
            metadata = {}
            if isinstance(factory_result, tuple):
                driver, metadata = factory_result
            else:
                driver = factory_result
            driver.reset_episode()
            buffer = Phase2RolloutBuffer(
                gamma=self.config.gamma,
                gae_lambda=self.config.gae_lambda,
            )
            summary = self.collect_episode(driver, buffer, stochastic=True)
            if buffer.steps:
                buffer.finish_episode(last_value=0.0)
                stats = self.update_policy(buffer)
            else:
                stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
            row = {"episode": episode, **dict(metadata), **summary, **stats}
            history.append(row)
            if episode_logger is not None:
                episode_logger.log(row)
        return history


# ---------------------------------------------------------------------------
# 多头 PPO 训练器（报告 §4.7）— 不修改上方现有 PPOConfig / Phase2PPOTrainer
# ---------------------------------------------------------------------------

@dataclass
class MultiHeadPPOConfig:
    """多头 PPO 训练超参数。

    在标准 PPO 超参数基础上，额外增加四通道优势加权权重。
    逐通道归一化优势后按权重求和得到统一优势 A_t（报告 §4.7）。

    Attributes:
        gamma: 折扣因子。
        gae_lambda: GAE lambda 参数。
        clip_ratio: PPO 裁剪范围 epsilon。
        value_coef: 价值损失系数 (各通道共用)。
        entropy_coef: 熵奖励系数。
        learning_rate: Adam 学习率。
        train_epochs: 每轮更新 epoch 数。
        minibatch_size: 小批次大小。
        max_grad_norm: 梯度裁剪范数上限。
        w_exec/w_qtime/w_util/w_progress: 各通道优势权重 (均取正，
            r_qtime 奖励本身为负值，靠 reward 符号体现 cost，故统一加法)。
        channels: 通道顺序 (固定 exec/qtime/util/progress)。
        use_qtime_lagrangian: 是否把 Q-time 残差约束改为 PPO-Lagrangian 自适应 λ
            (报告 §3.3)。True 时 qtime 通道权重由可学习的 λ 取代固定 w_qtime；
            False 时沿用固定 w_qtime (默认，基础版先跑通)。二者作用于同一 qtime
            残差通道，不并用。
        qtime_lambda_init: λ 初值 (≥0)。
        qtime_cost_budget: ε —— 可容忍的期望违规率 (报告 CMDP 的约束阈值)，
            领域可解释；对偶上升把违规率逼近此值。
        qtime_lambda_lr: η_λ —— 对偶上升步长，须远小于策略学习率以保证稳定。
        qtime_lambda_max: λ 上限，防止违规长期超预算时 λ 发散。
    """
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    learning_rate: float = 3e-4
    train_epochs: int = 4
    minibatch_size: int = 32
    max_grad_norm: float = 0.5
    w_exec: float = 1.0
    w_qtime: float = 3.0
    w_util: float = 0.5
    w_progress: float = 0.3
    channels: tuple = ("exec", "qtime", "util", "progress")
    # PPO-Lagrangian (报告 §3.3) — 默认关闭，沿用固定 w_qtime
    use_qtime_lagrangian: bool = False
    qtime_lambda_init: float = 0.0
    qtime_cost_budget: float = 0.0
    qtime_lambda_lr: float = 0.05
    qtime_lambda_max: float = 1e3

    def channel_weight(self, channel):
        """获取指定通道的权重 (w_<channel>)。"""
        return float(getattr(self, f"w_{channel}"))


def combine_channel_advantages(adv_dict, config, normalize=True):
    """逐通道归一化优势后加权求和 (报告 §4.7)。

    A_t = w_exec·norm(Â_exec) + w_qtime·norm(Â_qtime)
          + w_util·norm(Â_util) + w_progress·norm(Â_progress)

    Args:
        adv_dict: {channel: np.ndarray(N,)} 各通道优势。
        config: MultiHeadPPOConfig，提供通道权重与通道顺序。
        normalize: 是否对每通道优势做 (a - mean)/(std + 1e-8) 归一化。

    Returns:
        np.ndarray(N,) 合并后的统一优势。
    """
    combined = None
    for channel in config.channels:
        adv = np.asarray(adv_dict[channel], dtype=np.float64)
        if normalize:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        weighted = config.channel_weight(channel) * adv
        combined = weighted if combined is None else combined + weighted
    return combined


class MultiHeadPPOTrainer:
    """多头逐通道 Critic 的 PPO 训练器 (报告 §4.7)。

    与 Phase2PPOTrainer 的区别:
      - 策略网络是 Phase2SASMultiHeadActorCritic，evaluate_actions 返回
        values 为 {channel: tensor} 字典。
      - 优势: 逐通道归一化后按 w_<channel> 加权求和 → 统一优势 A_t
        (在 torch 中复刻 combine_channel_advantages 的逻辑以支持反向传播)。
      - 价值损失: Σ_c value_coef·MSE(V_c, R_c) (各通道独立 returns)。
      - 策略损失: clipped PPO 目标，使用加权后的 A_t。
    """

    def __init__(self, policy, optimizer, config):
        self.policy = policy
        self.optimizer = optimizer
        self.config = config
        self.channels = tuple(config.channels)
        # PPO-Lagrangian 的可学习乘子 (报告 §3.3)。仅 use_qtime_lagrangian 时生效。
        self.lambda_qtime = float(getattr(config, "qtime_lambda_init", 0.0))

    def qtime_weight(self):
        """qtime 残差通道当前使用的权重。

        - use_qtime_lagrangian=True: 用自适应乘子 λ (对偶上升学习, 报告 §3.3)。
        - 否则: 用固定 w_qtime (基础版)。
        二者作用于同一 qtime 残差通道, 不并用。
        """
        if getattr(self.config, "use_qtime_lagrangian", False):
            return float(self.lambda_qtime)
        return self.config.channel_weight("qtime")

    def update_lambda(self, mean_violation):
        """拉格朗日乘子对偶上升 (报告 §3.3.3):

            λ ← clip( max(0, λ + η_λ·(Ê[violation] − ε)), 0, λ_max )

        违规率 > ε → λ 增大 (加重惩罚, 逼策略保守);
        违规率 < ε → λ 减小 (放松, 争取利用率);
        最终停在违规率 ≈ ε 的约束边界。仅 use_qtime_lagrangian 时更新。

        Args:
            mean_violation: 本轮 (或多轮均值) 的期望 Q-time 违规率 (≥0)。

        Returns:
            更新后的 λ (禁用时恒为初值)。
        """
        if not getattr(self.config, "use_qtime_lagrangian", False):
            return float(self.lambda_qtime)
        lr = float(self.config.qtime_lambda_lr)
        budget = float(self.config.qtime_cost_budget)
        lam_max = float(self.config.qtime_lambda_max)
        updated = self.lambda_qtime + lr * (float(mean_violation) - budget)
        self.lambda_qtime = float(min(max(0.0, updated), lam_max))
        return self.lambda_qtime

    def episode_qtime_cost(self, buffer):
        """从 buffer 提取本 episode 的期望 Q-time 违规率 (cost)。

        qtime 通道终局奖励 r_qtime = -(violation_count / num_lots) ≤ 0
        (仅末步非零)，故违规率 = -Σ_t reward_vector[qtime通道]。

        Returns:
            违规率 (≥0)。buffer 为空时返回 0.0。
        """
        if "qtime" not in self.channels:
            return 0.0
        k = self.channels.index("qtime")
        total = 0.0
        for step in buffer.steps:
            total += float(np.asarray(step.reward_vector)[k])
        return -total + 0.0  # +0.0 规整 -0.0 → 0.0，避免日志出现 -0.0

    def _device(self):
        """获取策略网络所在的 torch device。"""
        try:
            return next(self.policy.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _tensor_batch(self, batch):
        """将 numpy 批次数据转换为 torch tensor (含每通道 advantages/returns)。"""
        device = self._device()
        tensors = {
            "candidate_features": torch.as_tensor(
                batch["candidate_features"], dtype=torch.float32, device=device,
            ),
            "candidate_mask": torch.as_tensor(
                batch["candidate_mask"], dtype=torch.bool, device=device,
            ),
            "global_features": torch.as_tensor(
                batch["global_features"], dtype=torch.float32, device=device,
            ),
            "actions": torch.as_tensor(
                batch["actions"], dtype=torch.long, device=device,
            ),
            "old_log_probs": torch.as_tensor(
                batch["old_log_probs"], dtype=torch.float32, device=device,
            ),
        }
        for channel in self.channels:
            tensors[f"advantages_{channel}"] = torch.as_tensor(
                batch[f"advantages_{channel}"], dtype=torch.float32, device=device,
            )
            tensors[f"returns_{channel}"] = torch.as_tensor(
                batch[f"returns_{channel}"], dtype=torch.float32, device=device,
            )
        return tensors

    def _combine_advantages(self, batch):
        """在 torch 中逐通道归一化并加权求和 (复刻 combine_channel_advantages)。

        qtime 通道的权重经 qtime_weight() 取得: 启用 PPO-Lagrangian 时为自适应 λ,
        否则为固定 w_qtime (报告 §3.3.3 / §4.7)。
        """
        combined = None
        for channel in self.channels:
            adv = batch[f"advantages_{channel}"]
            if adv.numel() > 1:
                adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
            weight = self.qtime_weight() if channel == "qtime" else self.config.channel_weight(channel)
            weighted = weight * adv
            combined = weighted if combined is None else combined + weighted
        return combined

    def _evaluate_batch(self, batch):
        """对批次数据运行策略评估。"""
        return self.policy.evaluate_actions(
            batch["candidate_features"],
            batch["candidate_mask"],
            batch["global_features"],
            batch["actions"],
        )

    def update_policy(self, buffer):
        """执行多头 PPO 策略更新。

        流程:
          1. 若 buffer 为空，返回零统计。
          2. 若未计算 GAE，先计算各通道 GAE。
          3. 对 train_epochs 轮，遍历 minibatches:
             - 逐通道归一化优势并加权求和 → A_t
             - clipped PPO 策略损失 (用 A_t)
             - value_loss = Σ_c value_coef·MSE(V_c, R_c)
             - total = policy_loss + value_loss - entropy_coef·entropy
             - 反向传播 + 梯度裁剪 + optimizer.step()
          4. 返回最后一轮的 loss 统计。

        Returns:
            {"policy_loss", "value_loss", "entropy"}
        """
        if not buffer.steps:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        n = len(buffer.steps)
        if not any(len(v) == n for v in buffer.advantages.values()):
            buffer.finish_episode()

        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        for _ in range(self.config.train_epochs):
            for raw_batch in buffer.get_training_batches(self.config.minibatch_size):
                batch = self._tensor_batch(raw_batch)
                output = self._evaluate_batch(batch)

                # 逐通道归一化 + 加权求和 → 统一优势 A_t
                advantages = self._combine_advantages(batch)

                # PPO clipped 策略损失
                ratio = torch.exp(output["log_prob"] - batch["old_log_probs"])
                unclipped = ratio * advantages
                clipped = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_ratio,
                    1.0 + self.config.clip_ratio,
                ) * advantages
                policy_loss = -torch.min(unclipped, clipped).mean()

                # 多通道价值损失: Σ_c value_coef·MSE(V_c, R_c)
                value_loss = None
                for channel in self.channels:
                    channel_vl = F.mse_loss(
                        output["values"][channel],
                        batch[f"returns_{channel}"],
                    )
                    term = self.config.value_coef * channel_vl
                    value_loss = term if value_loss is None else value_loss + term

                entropy = output["entropy"].mean()

                # 总损失 = 策略损失 + 价值损失 - 熵奖励
                loss = policy_loss + value_loss - self.config.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    self.config.max_grad_norm,
                )
                self.optimizer.step()

                stats = {
                    "policy_loss": float(policy_loss.detach().cpu()),
                    "value_loss": float(value_loss.detach().cpu()),
                    "entropy": float(entropy.detach().cpu()),
                }
        return stats

    def train(self, driver, num_episodes, episode_logger=None, reward_vector_config=None,
              on_episode=None):
        """多头端到端训练循环。每 episode: reset → 收集 → GAE → update。

        on_episode(episode_idx, row): 可选回调，每个 episode 结束后调用 —— 用于
        进度打印与周期性保存检查点 (长训练被中断时仍留有最新模型)。
        """
        history = []
        for episode in range(int(num_episodes)):
            driver.reset_episode()
            buffer = MultiHeadRolloutBuffer(
                gamma=self.config.gamma, gae_lambda=self.config.gae_lambda,
                channels=self.config.channels,
            )
            summary = driver.run_multihead_policy_episode(
                self.policy, buffer=buffer, stochastic=True,
                reward_vector_config=reward_vector_config,
            )
            if buffer.steps:
                buffer.finish_episode(last_values={c: 0.0 for c in self.config.channels})
                # 用更新前的 λ 算优势 (与本轮采样一致)，再对偶上升供下一轮使用。
                qtime_cost = self.episode_qtime_cost(buffer)
                stats = self.update_policy(buffer)
                self.update_lambda(qtime_cost)
            else:
                qtime_cost = 0.0
                stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
            row = {
                "episode": episode, **summary, **stats,
                "qtime_cost": float(qtime_cost),
                "lambda_qtime": float(self.lambda_qtime),
            }
            history.append(row)
            if episode_logger is not None:
                episode_logger.log(row)
            if on_episode is not None:
                on_episode(episode, row)
        return history
