"""PPO-Lagrangian (报告 §3.3) — Q-time 残差约束的自适应 λ 对偶上升。

把"期望违规 ≤ ε"作为约束 (CMDP)，用拉格朗日对偶上升让 qtime 通道权重自学习：
    λ ← max(0, λ + η_λ·(Ê[violation] − ε))
违规率 = qtime 通道终局奖励的相反数 (r_qtime = -violation_count/num_lots)。

纯逻辑测试 (λ 更新 / 权重选择 / cost 提取) 不依赖 torch；
端到端 update_policy 的 smoke 测试需要 torch (无则跳过)。
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from phase2_ppo_buffer import MultiHeadRolloutBuffer, MultiHeadRolloutStep
from phase2_ppo_trainer import MultiHeadPPOConfig, MultiHeadPPOTrainer

CHANNELS = ("exec", "qtime", "util", "progress")


def _make_buffer(qtime_terminal_reward):
    """构造一个 5 步 buffer，仅末步 qtime 通道非零 (= 终局违规残差)。"""
    buf = MultiHeadRolloutBuffer(channels=CHANNELS)
    for i in range(5):
        terminal = i == 4
        rv = np.array(
            [0.2, qtime_terminal_reward if terminal else 0.0, 0.0, 0.0],
            dtype=float,
        )
        buf.add(MultiHeadRolloutStep(
            machine_id=1, current_time=float(i),
            candidate_features=np.zeros((3, 18), dtype=np.float32),
            candidate_mask=np.ones(3, dtype=bool),
            global_features=np.zeros(9, dtype=np.float32),
            action_indices=np.arange(3, dtype=np.int64),
            valid_action_count=3, action=0, log_prob=-1.0,
            values={c: 0.0 for c in CHANNELS},
            reward_vector=rv, done=terminal, next_observation=None, info=None,
        ))
    return buf


class TestLagrangianConfig:
    def test_defaults_off(self):
        """默认关闭 Lagrangian，沿用固定 w_qtime。"""
        cfg = MultiHeadPPOConfig()
        assert cfg.use_qtime_lagrangian is False
        assert cfg.qtime_lambda_init == 0.0
        assert cfg.qtime_cost_budget == 0.0
        assert hasattr(cfg, "qtime_lambda_lr")
        assert hasattr(cfg, "qtime_lambda_max")


class TestQtimeWeightSelection:
    def test_fixed_weight_when_disabled(self):
        cfg = MultiHeadPPOConfig(w_qtime=3.0, use_qtime_lagrangian=False)
        trainer = MultiHeadPPOTrainer(policy=None, optimizer=None, config=cfg)
        assert trainer.qtime_weight() == pytest.approx(3.0)

    def test_lambda_weight_when_enabled(self):
        cfg = MultiHeadPPOConfig(
            w_qtime=3.0, use_qtime_lagrangian=True, qtime_lambda_init=0.7,
        )
        trainer = MultiHeadPPOTrainer(policy=None, optimizer=None, config=cfg)
        # 启用时忽略固定 w_qtime，改用当前 λ
        assert trainer.lambda_qtime == pytest.approx(0.7)
        assert trainer.qtime_weight() == pytest.approx(0.7)


class TestDualAscent:
    def test_lambda_increases_when_violation_exceeds_budget(self):
        cfg = MultiHeadPPOConfig(
            use_qtime_lagrangian=True, qtime_lambda_init=0.0,
            qtime_cost_budget=0.02, qtime_lambda_lr=0.5,
        )
        trainer = MultiHeadPPOTrainer(policy=None, optimizer=None, config=cfg)
        new_lambda = trainer.update_lambda(mean_violation=0.10)
        # λ ← max(0, 0 + 0.5*(0.10 - 0.02)) = 0.04
        assert new_lambda == pytest.approx(0.04)
        assert trainer.lambda_qtime == pytest.approx(0.04)

    def test_lambda_decreases_and_clamps_at_zero(self):
        cfg = MultiHeadPPOConfig(
            use_qtime_lagrangian=True, qtime_lambda_init=0.01,
            qtime_cost_budget=0.05, qtime_lambda_lr=0.5,
        )
        trainer = MultiHeadPPOTrainer(policy=None, optimizer=None, config=cfg)
        # 0.01 + 0.5*(0.0 - 0.05) = -0.014 → clamp 0
        new_lambda = trainer.update_lambda(mean_violation=0.0)
        assert new_lambda == pytest.approx(0.0)

    def test_lambda_clamped_at_max(self):
        cfg = MultiHeadPPOConfig(
            use_qtime_lagrangian=True, qtime_lambda_init=9.0,
            qtime_cost_budget=0.0, qtime_lambda_lr=10.0, qtime_lambda_max=10.0,
        )
        trainer = MultiHeadPPOTrainer(policy=None, optimizer=None, config=cfg)
        new_lambda = trainer.update_lambda(mean_violation=1.0)
        assert new_lambda == pytest.approx(10.0)

    def test_update_noop_when_disabled(self):
        cfg = MultiHeadPPOConfig(use_qtime_lagrangian=False, qtime_lambda_init=0.0)
        trainer = MultiHeadPPOTrainer(policy=None, optimizer=None, config=cfg)
        new_lambda = trainer.update_lambda(mean_violation=0.5)
        assert new_lambda == pytest.approx(0.0)


class TestEpisodeQtimeCost:
    def test_cost_is_negated_qtime_channel_sum(self):
        cfg = MultiHeadPPOConfig(use_qtime_lagrangian=True)
        trainer = MultiHeadPPOTrainer(policy=None, optimizer=None, config=cfg)
        buf = _make_buffer(qtime_terminal_reward=-0.08)  # violation rate = 0.08
        cost = trainer.episode_qtime_cost(buf)
        assert cost == pytest.approx(0.08)

    def test_cost_zero_when_no_violation(self):
        cfg = MultiHeadPPOConfig(use_qtime_lagrangian=True)
        trainer = MultiHeadPPOTrainer(policy=None, optimizer=None, config=cfg)
        buf = _make_buffer(qtime_terminal_reward=0.0)
        assert trainer.episode_qtime_cost(buf) == pytest.approx(0.0)


class TestLagrangianTrainSmoke:
    def test_train_updates_lambda_and_reports(self):
        torch = pytest.importorskip("torch")
        from phase2_sas_policy import Phase2SASMultiHeadActorCritic
        from phase2_ppo_trainer import MultiHeadPPOTrainer as Trainer

        net = Phase2SASMultiHeadActorCritic(
            candidate_dim=18, global_dim=9, hidden_dim=16, channels=CHANNELS,
        )
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        cfg = MultiHeadPPOConfig(
            train_epochs=1, minibatch_size=4,
            use_qtime_lagrangian=True, qtime_lambda_init=0.1,
            qtime_cost_budget=0.0, qtime_lambda_lr=0.5,
        )
        trainer = Trainer(net, opt, cfg)
        buf = _make_buffer(qtime_terminal_reward=-0.20)  # violation 0.20 > budget 0
        buf.finish_episode(last_values={c: 0.0 for c in CHANNELS})
        stats = trainer.update_policy(buf)
        # update_policy 仍返回原有字段
        assert "policy_loss" in stats and "value_loss" in stats
        # λ 对偶上升后应反映在 trainer.lambda_qtime / stats 上
        trainer.update_lambda(trainer.episode_qtime_cost(buf))
        assert trainer.lambda_qtime > 0.1  # 违规超预算 → λ 增大
