import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pytest

# 纯逻辑测试：advantage 加权（不依赖 torch）
from phase2_ppo_trainer import MultiHeadPPOConfig, combine_channel_advantages

CHANNELS = ("exec", "qtime", "util", "progress")


class TestAdvantageCombination:
    def test_combine_weights(self):
        cfg = MultiHeadPPOConfig(w_exec=1.0, w_qtime=2.0, w_util=0.5, w_progress=0.3)
        adv = {
            "exec": np.array([1.0, -1.0]),
            "qtime": np.array([2.0, -2.0]),
            "util": np.array([0.0, 0.0]),
            "progress": np.array([1.0, 1.0]),
        }
        combined = combine_channel_advantages(adv, cfg, normalize=False)
        # exec*1 + qtime*2 + util*0.5 + progress*0.3
        assert combined[0] == pytest.approx(1.0 * 1.0 + 2.0 * 2.0 + 0.0 * 0.5 + 1.0 * 0.3)
        assert combined[1] == pytest.approx(-1.0 * 1.0 + -2.0 * 2.0 + 0.0 * 0.5 + 1.0 * 0.3)

    def test_combine_with_normalization(self):
        cfg = MultiHeadPPOConfig()
        adv = {c: np.array([1.0, 2.0, 3.0]) for c in CHANNELS}
        combined = combine_channel_advantages(adv, cfg, normalize=True)
        assert combined.shape == (3,)


class TestMultiHeadTrainerTorch:
    def test_update_smoke(self):
        torch = pytest.importorskip("torch")
        from phase2_sas_policy import Phase2SASMultiHeadActorCritic
        from phase2_ppo_buffer import MultiHeadRolloutBuffer, MultiHeadRolloutStep
        from phase2_ppo_trainer import MultiHeadPPOTrainer, MultiHeadPPOConfig

        net = Phase2SASMultiHeadActorCritic(candidate_dim=18, global_dim=9, hidden_dim=16, channels=CHANNELS)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        cfg = MultiHeadPPOConfig(train_epochs=1, minibatch_size=4)
        trainer = MultiHeadPPOTrainer(net, opt, cfg)

        buf = MultiHeadRolloutBuffer(channels=CHANNELS)
        for i in range(5):
            buf.add(MultiHeadRolloutStep(
                machine_id=1, current_time=0.0,
                candidate_features=np.random.randn(5, 18).astype(np.float32),
                candidate_mask=np.ones(5, dtype=bool),
                global_features=np.random.randn(9).astype(np.float32),
                action_indices=np.arange(5, dtype=np.int64),
                valid_action_count=5, action=0, log_prob=-1.0,
                values={c: 0.0 for c in CHANNELS},
                reward_vector=np.array([0.2, 0.0, 0.0, 0.0]),
                done=(i == 4), next_observation=None, info=None,
            ))
        buf.finish_episode(last_values={c: 0.0 for c in CHANNELS})
        stats = trainer.update_policy(buf)
        assert "policy_loss" in stats and "value_loss" in stats
