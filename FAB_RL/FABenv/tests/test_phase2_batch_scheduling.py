"""杠杆 B: dry-run/commit 按子批排程 (报告 §1.5 批处理建模)。

要求: 工件按 ⌈N/side_capacity⌉ 个子批排，同一子批的 wafer 在每个 stage 共享同一
(chamber, side, start, end) 区间 (批处理机"同进同出")，而非逐片串行。
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lower_layer_estimator import compute_sub_batches
from problem_instances import build_pressure_test_encoder
from rl_environment import ResourceCalendarEnv


def _commit_first(env, enc):
    """在某个可调度机台上提交第一个有效真实动作，返回 committed lot id 或 None。"""
    for m in env.get_candidate_machines():
        pool = env.build_candidate_pool(int(m))
        for idx, (a, ok) in enumerate(zip(pool.actions, pool.action_mask)):
            if ok and not a.is_padding and not a.is_wait and int(a.ppid) != 0:
                env.commit_action_index(int(m), idx, pool=pool)
                return int(a.lot)
    return None


class TestBatchScheduling:
    def test_same_subbatch_wafers_share_interval(self):
        enc = build_pressure_test_encoder()
        env = ResourceCalendarEnv(enc)
        env.reset()
        lot = _commit_first(env, enc)
        assert lot is not None
        wc = int(enc.wafer_counts[lot])
        cap = int(getattr(enc, "side_capacity", wc))
        batches = compute_sub_batches(wc, cap)  # e.g. [4,4,2]

        ws = np.asarray(env.wafer_schedule, dtype=float)
        ws = ws[ws[:, 0] == lot]
        n_stages = len({int(r[4]) for r in ws})
        # 每个 wafer × stage 一行
        assert ws.shape[0] == wc * n_stages

        # 分配 wafer 到子批 (满批优先, 顺序): wafers 1..4 | 5..8 | 9..10
        wafer_to_batch = {}
        cursor = 0
        for b_idx, b_size in enumerate(batches):
            for w in range(cursor + 1, cursor + b_size + 1):
                wafer_to_batch[w] = b_idx
            cursor += b_size

        # 同一 (batch, stage) 的 wafer 必须共享 (chamber, side, start, end)
        groups = {}
        for r in ws:
            w = int(r[1]); stage = int(r[4]); b = wafer_to_batch[w]
            key = (b, stage)
            cell = (int(r[5]), int(r[6]), round(float(r[7]), 6), round(float(r[8]), 6))
            groups.setdefault(key, set()).add(cell)
        for key, cells in groups.items():
            assert len(cells) == 1, f"(batch,stage)={key} wafers not sharing interval: {cells}"

    def test_distinct_intervals_per_stage_match_batch_count(self):
        enc = build_pressure_test_encoder()
        env = ResourceCalendarEnv(enc)
        env.reset()
        lot = _commit_first(env, enc)
        assert lot is not None
        wc = int(enc.wafer_counts[lot])
        cap = int(getattr(enc, "side_capacity", wc))
        n_batches = len(compute_sub_batches(wc, cap))

        ws = np.asarray(env.wafer_schedule, dtype=float)
        ws = ws[ws[:, 0] == lot]
        # 每个 stage 的不同 (chamber,side,start,end) 区间数应 == 子批数 (按批不按片)
        per_stage = {}
        for r in ws:
            stage = int(r[4])
            cell = (int(r[5]), int(r[6]), round(float(r[7]), 6), round(float(r[8]), 6))
            per_stage.setdefault(stage, set()).add(cell)
        for stage, cells in per_stage.items():
            assert len(cells) == n_batches, f"stage {stage}: {len(cells)} intervals != {n_batches} batches"

    def test_schedule_validates(self):
        enc = build_pressure_test_encoder()
        env = ResourceCalendarEnv(enc)
        env.reset()
        for _ in range(5):
            if _commit_first(env, enc) is None:
                break
        # 部分排程的腔体日历应无重叠 (批区间不被当成重复冲突)
        env.encoder.validate_no_interval_overlap(env.state.chamber_calendar, "chamber_calendar")
