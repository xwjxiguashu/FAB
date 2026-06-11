"""estimate() 的 MC 采样必须可复现 (可复现性 bug 修复, 2026-06-11)。

历史 bug: estimate()/monte_carlo_makespan 在 rng=None 时用无种子的
np.random.default_rng() (OS 熵) → (μ_fin, σ_fin) 逐进程随机 → aggregate qtime
mask 在边界候选上翻转 → 同一确定性实例的 FIFO 调度跨进程漂移 (late_hi_scarce
上 O2 出现 3306.32 / 3244.29 两种结果, 7:3 分布)。estimate 的结果缓存键是
(lot, machine, ppid, n_mc), 故 rng 必须由同一键确定性派生。
"""
import numpy as np

from lower_layer_estimator import estimate, monte_carlo_makespan


def _first_lmp(encoder):
    (lot, machine), ppids = next(iter(encoder.feasible_ppids.items()))
    return lot, machine, ppids[0]


def test_estimate_is_deterministic_across_calls(small_encoder, small_env):
    lot, machine, ppid = _first_lmp(small_encoder)

    first = estimate(lot, machine, ppid, small_encoder, small_env.state, n_mc=20)
    second = estimate(lot, machine, ppid, small_encoder, small_env.state, n_mc=20)

    assert first["mu_finish"] == second["mu_finish"]
    assert first["sigma_finish"] == second["sigma_finish"]


def test_estimate_streams_differ_by_key(small_encoder, small_env):
    """不同 (lot, machine, ppid) 的噪声流相互独立 (非同一条复用流)。"""
    keys = list(small_encoder.feasible_ppids.items())
    (lot_a, machine_a), ppids_a = keys[0]
    (lot_b, machine_b), ppids_b = keys[1]

    a = estimate(lot_a, machine_a, ppids_a[0], small_encoder, small_env.state, n_mc=20)
    b = estimate(lot_b, machine_b, ppids_b[0], small_encoder, small_env.state, n_mc=20)

    # 不同 lot/ppid 的 makespan 分布几乎必然不同 (相等只可能是巧合复用)
    assert (a["mu_finish"], a["sigma_finish"]) != (b["mu_finish"], b["sigma_finish"])


def test_monte_carlo_makespan_default_rng_is_deterministic():
    sub_batches = [4, 4, 2]
    stage_mu = [2.0, 3.0]
    stage_sigma = [0.2, 0.3]
    instance_counts = [2, 2]

    r1 = monte_carlo_makespan(sub_batches, stage_mu, stage_sigma, instance_counts, n_mc=30)
    r2 = monte_carlo_makespan(sub_batches, stage_mu, stage_sigma, instance_counts, n_mc=30)

    assert r1 == r2
