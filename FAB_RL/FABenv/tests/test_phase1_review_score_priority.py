import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pytest
from problem_instances import build_small_encoder
from rl_environment import ResourceCalendarEnv


class TestScoreNotExploding:
    def test_score_bounded_when_slack_small(self):
        # 推进时间让某些 lot 的 qtime_slack 很小，score 不应爆炸
        env = ResourceCalendarEnv(build_small_encoder(), top_k=8)
        env.reset()
        env.advance_time(15.0)  # 接近多数 lot 的 qtime_deadline
        for m in env.get_candidate_machines():
            pool = env.build_candidate_pool(m)
            real = pool.features[pool.action_mask]
            if real.size:
                score_col = real[:, env.feature_names.index("score")]
                # 不再出现 1/slack 的千量级爆炸
                assert np.all(np.abs(score_col) < 100.0)


class TestPriorityFilter:
    def test_soft_mode_keeps_all_priorities(self):
        # 默认 soft：候选池含多个优先级的 lot
        env = ResourceCalendarEnv(build_small_encoder(), top_k=16)
        env.reset()
        env.advance_time(5.0)  # 让 lot1-4 都到达
        # machine 1 可加工 lot1,2,4
        pool = env.build_candidate_pool(1)
        lots = {a.lot for a, ok in zip(pool.actions, pool.action_mask)
                if ok and not a.is_wait and not a.is_padding}
        assert len(lots) >= 2  # soft 模式保留多个 lot

    def test_strict_mode_keeps_only_highest_priority(self):
        # strict：machine 1 上只留最高优先级的 lot
        env = ResourceCalendarEnv(build_small_encoder(), top_k=16,
                                  priority_filter_mode="strict")
        env.reset()
        env.advance_time(5.0)
        pool = env.build_candidate_pool(1)
        lots = {a.lot for a, ok in zip(pool.actions, pool.action_mask)
                if ok and not a.is_wait and not a.is_padding}
        # machine1 可加工的已到达 lot 中, lot2 priority=4.0 最高
        # strict 应只留 priority 最高的 lot(s)
        if lots:
            pris = {env.encoder.priorities[l] for l in lots}
            assert len(pris) == 1  # 只剩一个优先级层级
            assert max(pris) == max(env.encoder.priorities[l]
                                    for l in [1, 2, 4])  # machine1 可加工的

    def test_default_mode_is_soft(self):
        env = ResourceCalendarEnv(build_small_encoder(), top_k=8)
        assert env.priority_filter_mode == "soft"
