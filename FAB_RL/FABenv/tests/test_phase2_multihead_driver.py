import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pytest
torch = pytest.importorskip("torch")
from problem_instances import build_small_encoder
from rl_environment import ResourceCalendarEnv, RewardVectorConfig
from phase2_sas_observation import Phase2ObservationEncoder
from phase2_sas_driver import Phase2EpisodeDriver
from phase2_ppo_buffer import MultiHeadRolloutBuffer, MultiHeadRolloutStep, MULTIHEAD_CHANNELS
from phase2_sas_policy import Phase2SASMultiHeadActorCritic

def _build():
    env = ResourceCalendarEnv(build_small_encoder(), top_k=8); env.reset()
    enc = Phase2ObservationEncoder()
    driver = Phase2EpisodeDriver(env, enc, RewardVectorConfig())
    m = env.get_candidate_machines()[0]
    obs = enc.encode(m, env.build_candidate_pool(m), env)
    policy = Phase2SASMultiHeadActorCritic(
        candidate_dim=obs.candidate_features.shape[1],
        global_dim=obs.global_features.shape[0],
        hidden_dim=16, channels=MULTIHEAD_CHANNELS)
    return env, driver, policy

class TestMultiHeadDriver:
    def test_episode_fills_buffer(self):
        env, driver, policy = _build()
        driver.reset_episode()
        buf = MultiHeadRolloutBuffer(channels=MULTIHEAD_CHANNELS)
        summary = driver.run_multihead_policy_episode(policy, buffer=buf, stochastic=True)
        assert len(buf.steps) > 0
        step = buf.steps[0]
        assert isinstance(step, MultiHeadRolloutStep)
        assert step.reward_vector.shape == (4,)
        assert set(step.values.keys()) == set(MULTIHEAD_CHANNELS)
        assert "steps" in summary and "episode_reward" in summary

    def test_buffer_gae_runs(self):
        env, driver, policy = _build()
        driver.reset_episode()
        buf = MultiHeadRolloutBuffer(channels=MULTIHEAD_CHANNELS)
        driver.run_multihead_policy_episode(policy, buffer=buf, stochastic=True)
        buf.finish_episode(last_values={c: 0.0 for c in MULTIHEAD_CHANNELS})
        for c in MULTIHEAD_CHANNELS:
            assert len(buf.advantages[c]) == len(buf.steps)
