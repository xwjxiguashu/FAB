"""Phase 2 SAS 策略网络 — Actor-Critic 架构 + 掩码 Categorical 策略。

Phase2SASActorCritic:
  - 候选编码器: 2 层 MLP (Linear→Tanh→Linear→Tanh), 将 18 维候选特征编码为 hidden_dim 维
  - Actor 头: Linear(hidden_dim→1) → 每个候选一个 logit
  - Critic 头: 掩码池化候选编码 + 全局特征 → MLP → 标量 Value
  - 掩码池化: 仅对有效候选 (mask=True) 的编码求均值

MaskedCategoricalPolicy:
  - 将无效候选的 logit 设为 -inf → Categorical 分布
  - 支持采样 (sample)、贪心 (greedy)、评估 (evaluate_actions)

Phase2SASMultiHeadActorCritic:
  - 与单头共享 candidate_encoder 和 actor_head
  - 独立 value 头 (exec/qtime/util), 各自 Linear(hidden+global→hidden→1)
  - critic_values 返回 {channel: value} 字典
"""

import torch
import torch.nn as nn


class MaskedCategoricalPolicy(nn.Module):
    """掩码 Categorical 策略 — 仅从有效动作中采样。

    对无效动作的 logit 填充 -inf，使 Categorical 分布概率为 0。
    要求每行至少有一个有效动作。
    """

    def forward(self, logits, mask):
        """构建掩码后的 Categorical 分布。"""
        mask = mask.bool()
        if not torch.any(mask, dim=-1).all():
            raise ValueError("masked categorical requires at least one valid action per row")
        masked_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        return torch.distributions.Categorical(logits=masked_logits)

    def sample(self, logits, mask):
        """从掩码分布中采样一个动作。

        Returns:
            {"action", "log_prob", "entropy", "probs"}
        """
        distribution = self.forward(logits, mask)
        action = distribution.sample()
        return {
            "action": action,
            "log_prob": distribution.log_prob(action),
            "entropy": distribution.entropy(),
            "probs": distribution.probs,
        }

    def greedy(self, logits, mask):
        """贪心选择 — 取概率最大的有效动作。

        Returns:
            {"action", "log_prob", "entropy", "probs"}
        """
        distribution = self.forward(logits, mask)
        action = torch.argmax(distribution.probs, dim=-1)
        return {
            "action": action,
            "log_prob": distribution.log_prob(action),
            "entropy": distribution.entropy(),
            "probs": distribution.probs,
        }


class Phase2SASActorCritic(nn.Module):
    """Phase 2 SAS Actor-Critic 策略网络。

    架构:
      candidate_encoder: 18 → hidden_dim → hidden_dim (2 层 MLP + Tanh)
      actor_head:        hidden_dim → 1 (每候选一个 logit)
      critic:           hidden_dim + global_dim → hidden_dim → 1 (标量 Value)

    前向计算:
      1. encode_candidates(features) → 编码每候选
      2. actor_logits → 每候选得分
      3. critic_value → 掩码池化+全局特征 → Value 估计
    """

    def __init__(self, candidate_dim, global_dim, hidden_dim=128):
        super().__init__()
        # 候选编码器: 2 层 MLP
        self.candidate_encoder = nn.Sequential(
            nn.Linear(candidate_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        # Actor: 每候选 → 标量 logit
        self.actor_head = nn.Linear(hidden_dim, 1)
        # Critic: 池化候选编码 + 全局特征 → Value
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim + global_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.masked_policy = MaskedCategoricalPolicy()

    def encode_candidates(self, candidate_features):
        """将 (batch, pool_size, candidate_dim) 编码为 (batch, pool_size, hidden_dim)。"""
        return self.candidate_encoder(candidate_features)

    def actor_logits(self, candidate_features):
        """计算每候选的 logit — squeeze 最后一维。"""
        encoded = self.encode_candidates(candidate_features)
        return self.actor_head(encoded).squeeze(-1)

    def critic_value(self, candidate_features, candidate_mask, global_features):
        """计算状态 Value — 掩码池化 + 全局特征。

        池化: sum(encoded * mask) / sum(mask)，仅有效候选参与平均。
        """
        encoded = self.encode_candidates(candidate_features)
        mask = candidate_mask.bool().unsqueeze(-1)
        masked_encoded = encoded.masked_fill(~mask, 0.0)
        denom = mask.sum(dim=1).clamp(min=1).to(encoded.dtype)
        pooled = masked_encoded.sum(dim=1) / denom
        state_repr = torch.cat([pooled, global_features], dim=-1)
        return self.critic(state_repr).squeeze(-1)

    def forward(self, candidate_features, candidate_mask, global_features):
        """前向传播: 返回 (logits, value)。"""
        logits = self.actor_logits(candidate_features)
        value = self.critic_value(candidate_features, candidate_mask, global_features)
        return logits, value

    def sample_action(self, candidate_features, candidate_mask, global_features):
        """随机采样动作 (训练时使用)。

        Returns:
            {"action", "log_prob", "entropy", "probs", "value"}
        """
        logits, value = self.forward(candidate_features, candidate_mask, global_features)
        output = self.masked_policy.sample(logits, candidate_mask)
        output["value"] = value
        return output

    def greedy_action(self, candidate_features, candidate_mask, global_features):
        """贪心动作 (推理/评估时使用)。

        Returns:
            {"action", "log_prob", "entropy", "probs", "value"}
        """
        logits, value = self.forward(candidate_features, candidate_mask, global_features)
        output = self.masked_policy.greedy(logits, candidate_mask)
        output["value"] = value
        return output

    def evaluate_actions(self, candidate_features, candidate_mask, global_features, actions):
        """评估指定动作的 log_prob 和 entropy (PPO 更新时使用)。

        Returns:
            {"log_prob", "entropy", "value", "probs"}
        """
        logits, value = self.forward(candidate_features, candidate_mask, global_features)
        distribution = self.masked_policy(logits, candidate_mask)
        return {
            "log_prob": distribution.log_prob(actions),
            "entropy": distribution.entropy(),
            "value": value,
            "probs": distribution.probs,
        }


class Phase2SASMultiHeadActorCritic(nn.Module):
    """Phase 2 SAS 多头 Critic Actor-Critic 策略网络。

    与 Phase2SASActorCritic 的区别:
      - candidate_encoder 与 actor_head 结构相同 (但为独立参数实例)
      - 用 nn.ModuleDict 建 4 个独立 value 头, 每通道一个:
        Linear(hidden_dim + global_dim → hidden_dim) → Tanh → Linear(hidden_dim → 1)
      - critic_values 返回 {channel: value} 字典 (每个 value 形状 (batch,))
      - sample_action / greedy_action / evaluate_actions 输出键为 "values" (dict)

    通道顺序固定: ("exec", "qtime", "util")
    """

    def __init__(self, candidate_dim, global_dim, hidden_dim=128,
                 channels=("exec", "qtime", "util")):
        super().__init__()
        self.channels = tuple(channels)
        # 候选编码器: 2 层 MLP (与单头结构相同)
        self.candidate_encoder = nn.Sequential(
            nn.Linear(candidate_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        # Actor: 每候选 → 标量 logit (与单头结构相同)
        self.actor_head = nn.Linear(hidden_dim, 1)
        # 独立 Critic 头, 每通道一个
        self.critics = nn.ModuleDict({
            c: nn.Sequential(
                nn.Linear(hidden_dim + global_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            ) for c in self.channels
        })
        self.masked_policy = MaskedCategoricalPolicy()

    def encode_candidates(self, candidate_features):
        """将 (batch, pool_size, candidate_dim) 编码为 (batch, pool_size, hidden_dim)。"""
        return self.candidate_encoder(candidate_features)

    def actor_logits(self, candidate_features):
        """计算每候选的 logit — squeeze 最后一维。"""
        encoded = self.encode_candidates(candidate_features)
        return self.actor_head(encoded).squeeze(-1)

    def critic_values(self, candidate_features, candidate_mask, global_features):
        """计算每通道状态 Value — 掩码池化 + 全局特征 → 4 个头。

        池化: sum(encoded * mask) / sum(mask)，仅有效候选参与平均 (与单头一致)。

        Returns:
            {channel: value}，每个 value 形状 (batch,)
        """
        encoded = self.encode_candidates(candidate_features)
        mask = candidate_mask.bool().unsqueeze(-1)
        masked_encoded = encoded.masked_fill(~mask, 0.0)
        denom = mask.sum(dim=1).clamp(min=1).to(encoded.dtype)
        pooled = masked_encoded.sum(dim=1) / denom
        state_repr = torch.cat([pooled, global_features], dim=-1)
        return {c: self.critics[c](state_repr).squeeze(-1) for c in self.channels}

    def forward(self, candidate_features, candidate_mask, global_features):
        """前向传播: 返回 (logits, values_dict)。"""
        logits = self.actor_logits(candidate_features)
        values = self.critic_values(candidate_features, candidate_mask, global_features)
        return logits, values

    def sample_action(self, candidate_features, candidate_mask, global_features):
        """随机采样动作 (训练时使用)。

        Returns:
            {"action", "log_prob", "entropy", "probs", "values"}
        """
        logits, values = self.forward(candidate_features, candidate_mask, global_features)
        output = self.masked_policy.sample(logits, candidate_mask)
        output["values"] = values
        return output

    def greedy_action(self, candidate_features, candidate_mask, global_features):
        """贪心动作 (推理/评估时使用)。

        Returns:
            {"action", "log_prob", "entropy", "probs", "values"}
        """
        logits, values = self.forward(candidate_features, candidate_mask, global_features)
        output = self.masked_policy.greedy(logits, candidate_mask)
        output["values"] = values
        return output

    def evaluate_actions(self, candidate_features, candidate_mask, global_features, actions):
        """评估指定动作的 log_prob 和 entropy (PPO 更新时使用)。

        Returns:
            {"log_prob", "entropy", "values", "probs"}
        """
        logits, values = self.forward(candidate_features, candidate_mask, global_features)
        distribution = self.masked_policy(logits, candidate_mask)
        return {
            "log_prob": distribution.log_prob(actions),
            "entropy": distribution.entropy(),
            "values": values,
            "probs": distribution.probs,
        }
