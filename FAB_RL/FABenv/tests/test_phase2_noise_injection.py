import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pytest
from problem_instances import build_small_encoder
from rl_environment import ResourceCalendarEnv

def _commit_first_lot(env):
    machine = env.get_candidate_machines()[0]
    pool = env.build_candidate_pool(machine)
    idx = next(i for i, (a, m) in enumerate(zip(pool.actions, pool.action_mask))
               if m and not a.is_wait and not a.is_padding)
    env.commit_action_index(machine, idx, pool=pool)
    return env.wafer_schedule.copy()

class TestNoiseInjection:
    def test_disabled_is_deterministic_and_matches_baseline(self):
        # 默认（无噪声）两次结果一致
        env1 = ResourceCalendarEnv(build_small_encoder(), top_k=8); env1.reset()
        env2 = ResourceCalendarEnv(build_small_encoder(), top_k=8); env2.reset()
        w1 = _commit_first_lot(env1)
        w2 = _commit_first_lot(env2)
        assert np.allclose(w1, w2)

    def test_enabled_reproducible_same_seed(self):
        env1 = ResourceCalendarEnv(build_small_encoder(), top_k=8, process_noise_enabled=True, noise_seed=123); env1.reset()
        env2 = ResourceCalendarEnv(build_small_encoder(), top_k=8, process_noise_enabled=True, noise_seed=123); env2.reset()
        w1 = _commit_first_lot(env1)
        w2 = _commit_first_lot(env2)
        assert np.allclose(w1, w2)

    def test_enabled_differs_from_deterministic(self):
        env_det = ResourceCalendarEnv(build_small_encoder(), top_k=8); env_det.reset()
        env_noise = ResourceCalendarEnv(build_small_encoder(), top_k=8, process_noise_enabled=True, noise_seed=7); env_noise.reset()
        w_det = _commit_first_lot(env_det)
        w_noise = _commit_first_lot(env_noise)
        # 时间列（7,8 = start,end）应有差异
        assert not np.allclose(w_det[:, 7:9], w_noise[:, 7:9])

    def test_candidate_features_use_mu_not_noise(self):
        # 候选池特征不受 noise 影响（用 μ）
        env_det = ResourceCalendarEnv(build_small_encoder(), top_k=8); env_det.reset()
        env_noise = ResourceCalendarEnv(build_small_encoder(), top_k=8, process_noise_enabled=True, noise_seed=7); env_noise.reset()
        m = env_det.get_candidate_machines()[0]
        f_det = env_det.build_candidate_pool(m).features
        f_noise = env_noise.build_candidate_pool(m).features
        assert np.allclose(f_det, f_noise)
