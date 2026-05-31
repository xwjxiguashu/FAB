"""_dry_run_candidate 性能重构回归: 用 add+rollback 取代 copy_calendar。

正确性要求: dry-run 不得改变 state (跑完后日历必须逐字节还原)，且结果与原行为一致
(由现有候选池/驱动测试间接保证)。本文件聚焦 state-pristine 不变量。
"""

import copy
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from problem_instances import build_small_encoder, build_pressure_test_encoder
from rl_environment import ResourceCalendarEnv


def _snapshot(env):
    return (
        copy.deepcopy(env.state.machine_calendar),
        copy.deepcopy(env.state.chamber_calendar),
    )


class TestDryRunPristine:
    def test_state_unchanged_after_dry_run_empty(self):
        enc = build_small_encoder()
        env = ResourceCalendarEnv(enc)
        env.reset()
        before = _snapshot(env)
        lot = 1
        m = enc.get_machine_list(lot)[0]
        p = enc.get_ppid_list(lot, int(m))[0]
        dr, reason = env._dry_run_candidate(lot, int(m), int(p))
        assert dr is not None
        after = _snapshot(env)
        assert before == after  # state 必须原样还原

    def test_state_unchanged_after_dry_run_dense(self):
        """提交若干 lot 后日历变密，dry-run 仍须还原 state。"""
        enc = build_pressure_test_encoder()
        env = ResourceCalendarEnv(enc)
        env.reset()
        # 提交几个 lot 让日历变密
        committed = 0
        for lot in range(1, enc.num_lots + 1):
            if committed >= 6:
                break
            machines = enc.get_machine_list(lot)
            if len(machines) == 0:
                continue
            m = int(machines[0])
            ppids = enc.get_ppid_list(lot, m)
            if len(ppids) == 0:
                continue
            pool = env.build_candidate_pool(m)
            # 找到该机台一个有效真实动作并提交
            for idx, (act, ok) in enumerate(zip(pool.actions, pool.action_mask)):
                if ok and not act.is_padding and not act.is_wait and int(act.ppid) != 0:
                    env.commit_action_index(m, idx, pool=pool)
                    committed += 1
                    break
        before = _snapshot(env)
        # 对一个未完成 lot 做 dry-run
        for lot in range(1, enc.num_lots + 1):
            if lot in env.completed_lots:
                continue
            machines = enc.get_machine_list(lot)
            if len(machines) == 0:
                continue
            m = int(machines[0])
            ppids = enc.get_ppid_list(lot, m)
            if len(ppids) == 0:
                continue
            env._dry_run_candidate(lot, m, int(ppids[0]))
            break
        after = _snapshot(env)
        assert before == after

    def test_dry_run_result_fields(self):
        enc = build_small_encoder()
        env = ResourceCalendarEnv(enc)
        env.reset()
        lot = 1
        m = int(enc.get_machine_list(lot)[0])
        p = int(enc.get_ppid_list(lot, m)[0])
        dr, _ = env._dry_run_candidate(lot, m, p)
        for k in ("steps", "lot_release_time", "lot_start_time", "lot_end_time",
                  "total_process_time", "qtime_risk"):
            assert k in dr
        assert dr["lot_end_time"] >= dr["lot_start_time"]
