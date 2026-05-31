"""加工噪声下的腔体区间重叠 bug 回归 (pressure 实例)。

bug: _select_earliest_stage_resource 按规划 μ 预留槽位，commit 时噪声把区间拉长到
     start+p_actual (p_actual>μ)，插入比预留更长的区间 → 与后续已提交区间重叠，
     add_calendar_interval 抛 ValueError。small 实例太简单不触发，pressure 才暴露。
修复: 噪声须在"找槽位"阶段就计入，使预留的空档足够容纳实际区间。
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from problem_instances import build_pressure_test_encoder, build_small_encoder
from rl_environment import ResourceCalendarEnv, RewardConfig
from phase2_sas_observation import Phase2ObservationEncoder
from phase2_sas_driver import Phase2EpisodeDriver


class TestSlotDurationWithDelta:
    """单元测试: _select_earliest_stage_resource 支持把噪声 delta 计入预留时长。"""

    def test_positive_delta_reserves_longer_slot(self):
        enc = build_small_encoder()
        env = ResourceCalendarEnv(enc)
        env.reset()
        lot = 1
        machine = enc.get_machine_list(lot)[0]
        ppid = enc.get_ppid_list(lot, int(machine))[0]
        steps = enc.get_process_steps(lot, int(machine), ppid)
        stage = steps[0]
        chamber_calendar = {}
        base = env._select_earliest_stage_resource(int(machine), stage, 0.0, chamber_calendar)
        shifted = env._select_earliest_stage_resource(
            int(machine), stage, 0.0, chamber_calendar, process_time_delta=2.0,
        )
        rk_b, s_b, e_b = base
        rk_s, s_s, e_s = shifted
        # delta 把实际时长 (end-start) 增加 2.0
        assert (e_s - s_s) == pytest.approx((e_b - s_b) + 2.0)

    def test_negative_delta_clamped_positive(self):
        enc = build_small_encoder()
        env = ResourceCalendarEnv(enc)
        env.reset()
        lot = 1
        machine = enc.get_machine_list(lot)[0]
        ppid = enc.get_ppid_list(lot, int(machine))[0]
        stage = enc.get_process_steps(lot, int(machine), ppid)[0]
        rk, s, e = env._select_earliest_stage_resource(
            int(machine), stage, 0.0, {}, process_time_delta=-1e9,
        )
        assert e > s  # 时长被 clamp 到 > 0


class TestPressureNoiseCommitNoOverlap:
    """集成回归: pressure + noise 下连续提交不应抛区间重叠 (seed=0 原先必崩)。"""

    def test_commit_sequence_does_not_overlap(self):
        enc = build_pressure_test_encoder()
        env = ResourceCalendarEnv(enc, process_noise_enabled=True, noise_seed=0)
        env.reset()
        driver = Phase2EpisodeDriver(env, Phase2ObservationEncoder(), RewardConfig())
        driver.reset_episode()
        committed = 0
        for _ in range(40):
            if committed >= 12:
                break
            machines = driver.get_dispatchable_machines()
            if not machines:
                if driver.advance_to_next_event() is None:
                    break
                continue
            m = driver.select_next_machine(machines)
            decision = driver.build_decision(m)
            idx = driver._rule_action_index(decision.pool, "EDD")
            if idx is None:
                if driver.advance_to_next_event() is None:
                    break
                continue
            # 修复前这里会抛 ValueError(区间重叠)
            result = driver.step_with_action(m, idx, pool=decision.pool)
            if result.info.get("insertion_success"):
                committed += 1
        assert committed > 0
