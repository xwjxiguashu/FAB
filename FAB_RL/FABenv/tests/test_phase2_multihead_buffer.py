import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pytest
from phase2_ppo_buffer import MultiHeadRolloutStep, MultiHeadRolloutBuffer

CHANNELS = ("exec", "qtime", "util", "progress")

def _make_step(reward_vector, values, done):
    return MultiHeadRolloutStep(
        machine_id=1, current_time=0.0,
        candidate_features=np.zeros((4, 18), dtype=np.float32),
        candidate_mask=np.ones(4, dtype=bool),
        global_features=np.zeros(9, dtype=np.float32),
        action_indices=np.arange(4, dtype=np.int64),
        valid_action_count=4, action=0,
        log_prob=0.0,
        values={c: float(v) for c, v in zip(CHANNELS, values)},
        reward_vector=np.asarray(reward_vector, dtype=float),
        done=done, next_observation=None, info=None,
    )

class TestMultiHeadBuffer:
    def test_single_channel_matches_scalar_gae(self):
        # 用单通道(exec)对比标准 GAE 数值
        from phase2_ppo_buffer import Phase2RolloutBuffer, RolloutStep, StepInfo
        gamma, lam = 0.99, 0.95
        rewards = [1.0, 0.5, -0.2]
        values = [0.4, 0.3, 0.1]
        dones = [False, False, True]

        # 多头 buffer（只看 exec 通道）
        mh = MultiHeadRolloutBuffer(gamma=gamma, gae_lambda=lam, channels=CHANNELS)
        for r, v, d in zip(rewards, values, dones):
            rv = [r, 0.0, 0.0, 0.0]
            vv = [v, 0.0, 0.0, 0.0]
            mh.add(_make_step(rv, vv, d))
        mh.finish_episode(last_values={c: 0.0 for c in CHANNELS})

        # 标量 buffer 对照
        sc = Phase2RolloutBuffer(gamma=gamma, gae_lambda=lam)
        for r, v, d in zip(rewards, values, dones):
            sc.add(RolloutStep(
                machine_id=1, current_time=0.0,
                candidate_features=np.zeros((4,18)), candidate_mask=np.ones(4,dtype=bool),
                global_features=np.zeros(9), action_indices=np.arange(4),
                valid_action_count=4, action=0, log_prob=0.0, value=v,
                reward=r, done=d, next_observation=None, info=StepInfo()))
        sc.finish_episode(last_value=0.0)

        for i in range(len(rewards)):
            assert mh.advantages["exec"][i] == pytest.approx(sc.advantages[i], rel=1e-6)
            assert mh.returns["exec"][i] == pytest.approx(sc.returns[i], rel=1e-6)

    def test_channels_independent(self):
        mh = MultiHeadRolloutBuffer(gamma=0.99, gae_lambda=0.95, channels=CHANNELS)
        mh.add(_make_step([1.0, -0.5, 0.3, 0.2], [0.1, 0.2, 0.3, 0.4], False))
        mh.add(_make_step([0.5, -0.2, 0.1, 0.1], [0.0, 0.0, 0.0, 0.0], True))
        mh.finish_episode(last_values={c: 0.0 for c in CHANNELS})
        for c in CHANNELS:
            assert len(mh.advantages[c]) == 2
            assert len(mh.returns[c]) == 2

    def test_get_training_batches_has_vector_fields(self):
        mh = MultiHeadRolloutBuffer(gamma=0.99, gae_lambda=0.95, channels=CHANNELS)
        for _ in range(3):
            mh.add(_make_step([1.0,0.0,0.0,0.0],[0.1,0.1,0.1,0.1], False))
        mh.steps[-1].done = True
        mh.finish_episode(last_values={c: 0.0 for c in CHANNELS})
        batches = list(mh.get_training_batches(batch_size=2))
        assert len(batches) >= 1
        b = batches[0]
        # 每通道的 advantages/returns 必须存在
        for c in CHANNELS:
            assert f"advantages_{c}" in b
            assert f"returns_{c}" in b
        assert "candidate_features" in b and "actions" in b and "old_log_probs" in b
