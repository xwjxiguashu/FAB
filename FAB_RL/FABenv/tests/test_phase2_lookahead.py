import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pytest
from problem_instances import build_small_encoder
from rl_environment import ResourceCalendarEnv
from phase2_sas_observation import Phase2ObservationEncoder

class TestLookahead:
    def test_w0_only_arrived(self):
        # small encoder arrival: lot1=0, lot2=1.5, lot3=3, lot4=4
        env = ResourceCalendarEnv(build_small_encoder(), top_k=8, w_lookahead=0.0); env.reset()
        vis = set(env.visible_lots())
        assert vis == {1}  # t=0 只有 lot1 到达
        assert env.upcoming_lots() == []

    def test_lookahead_window_includes_upcoming(self):
        env = ResourceCalendarEnv(build_small_encoder(), top_k=8, w_lookahead=2.0); env.reset()
        # t=0, 窗 [0,2] 内即将到达: lot2(1.5)
        assert 2 in env.upcoming_lots()
        assert 1 in env.visible_lots()  # 已到达
        assert 2 in env.visible_lots()  # 窗内即将到达

    def test_lookahead_summary_fields(self):
        env = ResourceCalendarEnv(build_small_encoder(), top_k=8, w_lookahead=2.0); env.reset()
        s = env.lookahead_summary()
        assert "upcoming_count" in s and "max_priority" in s
        assert "min_remaining_qtime" in s and "earliest_eta" in s
        assert s["upcoming_count"] == 1  # 只有 lot2
        assert s["earliest_eta"] == pytest.approx(1.5)
        assert s["max_priority"] == pytest.approx(4.0)  # lot2 priority=4

    def test_summary_empty_window(self):
        env = ResourceCalendarEnv(build_small_encoder(), top_k=8, w_lookahead=0.0); env.reset()
        s = env.lookahead_summary()
        assert s["upcoming_count"] == 0
        assert s["max_priority"] == 0.0

    def test_observation_default_9dim(self):
        env = ResourceCalendarEnv(build_small_encoder(), top_k=8); env.reset()
        enc = Phase2ObservationEncoder()  # 默认 lookahead=False
        m = env.get_candidate_machines()[0]
        obs = enc.encode(m, env.build_candidate_pool(m), env)
        assert obs.global_features.shape[0] == 9

    def test_observation_lookahead_13dim(self):
        env = ResourceCalendarEnv(build_small_encoder(), top_k=8, w_lookahead=2.0); env.reset()
        enc = Phase2ObservationEncoder(lookahead=True)
        m = env.get_candidate_machines()[0]
        obs = enc.encode(m, env.build_candidate_pool(m), env)
        assert obs.global_features.shape[0] == 13
        # 前 9 维与无前瞻一致
        enc9 = Phase2ObservationEncoder(lookahead=False)
        obs9 = enc9.encode(m, env.build_candidate_pool(m), env)
        assert np.allclose(obs.global_features[:9], obs9.global_features)
