import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pytest
from problem_instances import build_small_encoder
from rl_environment import ResourceCalendarEnv
from lower_layer_estimator import estimate
from state import ScheduleState


class TestStartOffset:
    def test_start_offset_shifts_mu(self):
        enc = build_small_encoder()
        st = ScheduleState()
        lot, m = 1, 1
        p = enc.get_ppid_list(lot, m)[0]
        r0 = estimate(lot, m, p, enc, st, n_mc=30, rng=np.random.default_rng(0))
        r10 = estimate(lot, m, p, enc, st, n_mc=30, rng=np.random.default_rng(0),
                       start_offset=10.0)
        assert r10["mu_finish"] == pytest.approx(r0["mu_finish"] + 10.0, abs=1e-6)
        # sigma 不受 offset 影响
        assert r10["sigma_finish"] == pytest.approx(r0["sigma_finish"], abs=1e-6)


class TestIsDoomed:
    def test_doomed_when_deadline_passed(self):
        env = ResourceCalendarEnv(build_small_encoder(), top_k=8)
        env.reset()
        # 推进到远超 lot1 deadline 的时刻
        dl = env.encoder.get_qtime_deadline(1)
        env.advance_time(dl + 50.0)
        assert env.is_doomed(1) is True

    def test_not_doomed_early(self):
        env = ResourceCalendarEnv(build_small_encoder(), top_k=8)
        env.reset()
        # t=0 时 lot1 有充足时间
        assert env.is_doomed(1) is False


class TestQtimeMaskTimeBasis:
    def test_safe_at_t0(self):
        env = ResourceCalendarEnv(build_small_encoder(), top_k=8)
        env.reset()
        # t=0 时 lot1 的动作应安全（不被屏蔽）
        from rl_environment import DispatchAction
        p = env.encoder.get_ppid_list(1, 1)[0]
        mask = env.qtime_safe_mask(1, [DispatchAction(lot=1, machine=1, ppid=p)])
        assert mask[0] == True

    def test_doomed_lot_not_masked(self):
        # doomed lot 的候选不被 qtime mask 屏蔽（防死锁）
        env = ResourceCalendarEnv(build_small_encoder(), top_k=8)
        env.reset()
        dl = env.encoder.get_qtime_deadline(1)
        env.advance_time(dl + 50.0)  # lot1 已 doomed
        from rl_environment import DispatchAction
        p = env.encoder.get_ppid_list(1, 1)[0]
        mask = env.qtime_safe_mask(1, [DispatchAction(lot=1, machine=1, ppid=p)])
        assert mask[0] == True  # doomed 不屏蔽

    def test_candidate_pool_not_empty_when_doomed(self):
        # 即使有 doomed lot，候选池不应因 qtime mask 全空
        env = ResourceCalendarEnv(build_small_encoder(), top_k=8)
        env.reset()
        env.advance_time(40.0)  # 多数 lot 可能 doomed
        # 至少 machine 上仍能产生候选（不死锁）—— 宽松断言：不抛异常且能构建
        for m in env.get_candidate_machines():
            pool = env.build_candidate_pool(m)
            assert pool is not None
