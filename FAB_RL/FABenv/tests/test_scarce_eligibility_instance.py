"""能力稀缺实例 (报告8 §12.2 能力稀缺度旋钮): 部分工件只能在部分机台加工。"""
import numpy as np

from problem_instances import (
    build_late_hi_encoder,
    build_late_hi_scarce_encoder,
    build_pressure_test_encoder,
)


def test_default_density_keeps_full_flexibility():
    enc = build_pressure_test_encoder(seed=7)
    for lot in range(1, enc.num_lots + 1):
        assert enc.feasible_machines[lot] == list(range(1, enc.num_machines + 1))


def test_scarce_density_limits_eligible_machines():
    enc = build_pressure_test_encoder(seed=7, eligibility_density=0.3)
    k = 3  # round(0.3 * 10)
    for lot in range(1, enc.num_lots + 1):
        machines = enc.feasible_machines[lot]
        assert len(machines) == k
        assert machines == sorted(set(machines))
        assert all(1 <= m <= enc.num_machines for m in machines)

    # ppid / q_time_limits 只对可加工 (lot, machine) 登记
    for (lot, machine) in enc.feasible_ppids:
        assert machine in enc.feasible_machines[lot]
    for (lot, machine, _ppid, _f, _t) in enc.q_time_limits:
        assert machine in enc.feasible_machines[lot]

    # 不是所有 lot 共享同一子集 (确实存在异型差异)
    distinct = {tuple(enc.feasible_machines[lot]) for lot in range(1, enc.num_lots + 1)}
    assert len(distinct) > 1


def test_scarce_instance_is_deterministic_and_runnable():
    a = build_late_hi_scarce_encoder(seed=11)
    b = build_late_hi_scarce_encoder(seed=11)
    assert a.feasible_machines == b.feasible_machines

    from rl_environment import ResourceCalendarEnv

    env = ResourceCalendarEnv(a, top_k=8, w_lookahead=4.0)
    env.reset()
    pool = env.build_candidate_pool(1)
    assert pool is not None


def test_late_hi_scarce_keeps_priority_arrival_correlation():
    enc = build_late_hi_scarce_encoder(seed=11)
    lots = sorted(enc.arrival_times)
    arrivals = np.array([enc.arrival_times[l] for l in lots])
    priorities = np.array([enc.priorities[l] for l in lots])
    corr = float(np.corrcoef(arrivals, priorities)[0, 1])
    assert corr > 0.9


def test_scarce_flag_does_not_change_full_flex_instance():
    """density=1.0 与不传参数的实例逐位一致 (不破坏既有结果可比性)。"""
    base = build_late_hi_encoder(seed=5)
    flagged = build_pressure_test_encoder(
        seed=5, priority_mode="late_hi", eligibility_density=1.0
    )
    assert base.feasible_machines == flagged.feasible_machines
    assert base.priorities == flagged.priorities
    assert base.arrival_times == flagged.arrival_times
    key = next(iter(base.ppid_steps))
    assert all(
        np.array_equal(x, y)
        for x, y in zip(base.ppid_steps[key], flagged.ppid_steps[key])
    )
