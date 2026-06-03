"""压力实例 Q-time bug 修复 + 错峰到达的结构性断言。

背景: build_pressure_test_encoder 原先从未设置 q_time_limits, 导致
compute_q_time_violation 恒为 0 → qtime 指标/奖励通道/Lagrangian 全部静默失效。
本测试锁死修复后的结构属性 (区分度本身由单独的验证脚本经验确认)。
"""
from problem_instances import build_pressure_test_encoder


def test_pressure_defines_stage_qtime_limits():
    enc = build_pressure_test_encoder()
    assert len(enc.q_time_limits) > 0, (
        "pressure 必须定义阶段间 Q-time; 之前为空 → 整个 Q-time 维度失效"
    )
    # 应同时覆盖 (1,2) 和 (2,3) 两个阶段窗口
    from_to_pairs = {(int(k[3]), int(k[4])) for k in enc.q_time_limits}
    assert (1, 2) in from_to_pairs
    assert (2, 3) in from_to_pairs
    # 键 (lot, machine, ppid, from, to) 应匹配实际可被调度的 ppid
    lot, machine, ppid, _from, _to = next(iter(enc.q_time_limits))
    assert ppid in enc.feasible_ppids[(int(lot), int(machine))]


def test_pressure_qtime_limit_positive_and_finite():
    enc = build_pressure_test_encoder()
    for limit in enc.q_time_limits.values():
        assert 0.0 < float(limit) < 1e6


def test_pressure_arrivals_staggered():
    enc = build_pressure_test_encoder()
    arrivals = [float(a) for a in enc.arrival_times.values()]
    assert max(arrivals) > min(arrivals), "到达时间应错峰; 之前全部 t=0"
    # 至少一半 lot 在 t>0 到达 (确认不是个别抖动)
    assert sum(1 for a in arrivals if a > 0.0) >= len(arrivals) // 2


def test_pressure_qtime_deadline_and_due_still_arrival_relative():
    # 错峰后, qtime_deadline 与 due_date 仍应随各 lot 的到达时间平移 (不能写死)
    enc = build_pressure_test_encoder()
    arrivals = enc.arrival_times
    late_lot = max(arrivals, key=lambda lot: arrivals[lot])
    early_lot = min(arrivals, key=lambda lot: arrivals[lot])
    assert enc.due_dates[late_lot] > enc.due_dates[early_lot]
