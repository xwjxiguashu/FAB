"""estimate_qtime_risk 倒排索引优化的正确性锁定 (profile §1.6.1 瓶颈修复)。

原实现每次线性扫描整张 q_time_limits (压力实例下数千条) → 实测 ≈70% 训练墙钟。
改为按 (lot, machine, ppid) 建索引后, 结果必须与暴力线性扫描逐字节一致。
"""
import numpy as np

from problem_instances import build_pressure_test_encoder, build_small_encoder


def _brute_force_qtime_risk(enc, lot, machine, ppid, steps):
    """原始线性扫描实现的独立重写, 作为参考真值。"""
    risk = 0.0
    for key, q_time_limit in enc.q_time_limits.items():
        key_lot, key_machine, key_ppid, from_stage, to_stage = map(int, key)
        if (key_lot, key_machine, key_ppid) != (int(lot), int(machine), int(ppid)):
            continue
        if from_stage < 1 or to_stage > len(steps):
            continue
        intermediate_time = 0.0
        for stage_id in range(from_stage, to_stage - 1):
            resources = np.asarray(steps[stage_id], dtype=float)
            intermediate_time += float(np.min(resources[:, 2]))
        risk += max(0.0, intermediate_time - float(q_time_limit))
    return float(risk)


def _all_processes(enc):
    """枚举有 Q-time 约束的 (lot, machine, ppid)。"""
    seen = set()
    for key in enc.q_time_limits:
        seen.add((int(key[0]), int(key[1]), int(key[2])))
    return sorted(seen)


def test_indexed_risk_matches_bruteforce_pressure():
    enc = build_pressure_test_encoder()
    processes = _all_processes(enc)
    assert processes, "压力实例应有 Q-time 约束"
    for lot, machine, ppid in processes:
        steps = enc.get_process_steps(lot, machine, ppid)
        expected = _brute_force_qtime_risk(enc, lot, machine, ppid, steps)
        got = enc.estimate_qtime_risk(lot, machine, ppid, steps)
        assert got == expected, f"({lot},{machine},{ppid}): {got} != {expected}"


def test_indexed_risk_zero_for_unconstrained_process():
    # 无 Q-time 约束的进程 risk 应为 0 (索引 .get 回退空元组)
    enc = build_small_encoder()
    enc.q_time_limits = {}  # 触发索引按新源重建
    # 任取一个可行进程
    lot, machine = next(iter(enc.feasible_ppids))
    ppid = next(iter(enc.feasible_ppids[(lot, machine)]))
    steps = enc.get_process_steps(lot, machine, ppid)
    assert enc.estimate_qtime_risk(lot, machine, ppid, steps) == 0.0


def test_index_rebuilds_when_source_dict_replaced():
    # builder 会在构造后整体重赋 q_time_limits; 索引须随源对象身份变化而重建
    enc = build_pressure_test_encoder()
    lot, machine, ppid = _all_processes(enc)[0]
    steps = enc.get_process_steps(lot, machine, ppid)
    enc.estimate_qtime_risk(lot, machine, ppid, steps)  # 建立首个索引
    first_index = enc._qtime_limits_index

    # 整体替换为新 dict (模拟 builder 行为), 加一条新约束
    new_limits = dict(enc.q_time_limits)
    enc.q_time_limits = new_limits
    enc.estimate_qtime_risk(lot, machine, ppid, steps)
    assert enc._qtime_limits_index is not first_index, "源替换后索引应重建"
    assert enc._qtime_limits_index_src is new_limits
