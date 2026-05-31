"""下层估时器结果缓存 (报告 §1.5 开销警示)。

estimate() 的完成时间分布只取决于 (lot, machine, ppid, n_mc) 等静态输入；
start_offset 仅在返回时加到 mu_finish 上。故可对 base 结果 (offset=0) 缓存，
对同一 (lot, machine, ppid, n_mc) 复用，避免每个候选每步重跑蒙特卡洛。
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lower_layer_estimator import estimate
from problem_instances import build_small_encoder
from rl_environment import ResourceCalendarEnv


def _first_action():
    enc = build_small_encoder()
    lot = 1
    machine = enc.get_machine_list(lot)[0]
    ppid = enc.get_ppid_list(lot, int(machine))[0]
    return enc, int(lot), int(machine), int(ppid)


class TestEstimateCache:
    def test_cache_populated_on_first_call(self):
        enc, lot, machine, ppid = _first_action()
        env = ResourceCalendarEnv(enc)
        env.reset()
        cache = {}
        estimate(lot, machine, ppid, enc, env.state, n_mc=20, cache=cache)
        assert (lot, machine, ppid, 20) in cache
        assert len(cache) == 1

    def test_second_call_is_identical_hit(self):
        """缓存命中 → 结果逐字节一致 (无缓存时 MC 随机会不同)。"""
        enc, lot, machine, ppid = _first_action()
        env = ResourceCalendarEnv(enc)
        env.reset()
        cache = {}
        r1 = estimate(lot, machine, ppid, enc, env.state, n_mc=20, cache=cache)
        r2 = estimate(lot, machine, ppid, enc, env.state, n_mc=20, cache=cache)
        assert r1["mu_finish"] == r2["mu_finish"]
        assert r1["sigma_finish"] == r2["sigma_finish"]
        assert len(cache) == 1  # 没有新增 → 命中

    def test_start_offset_applied_on_hit(self):
        """命中时 start_offset 仍逐次重新施加，不污染缓存。"""
        enc, lot, machine, ppid = _first_action()
        env = ResourceCalendarEnv(enc)
        env.reset()
        cache = {}
        base = estimate(lot, machine, ppid, enc, env.state, n_mc=20, start_offset=0.0, cache=cache)
        shifted = estimate(lot, machine, ppid, enc, env.state, n_mc=20, start_offset=100.0, cache=cache)
        assert shifted["mu_finish"] == pytest.approx(base["mu_finish"] + 100.0)
        assert shifted["sigma_finish"] == pytest.approx(base["sigma_finish"])
        assert len(cache) == 1  # 仍是命中，未重算

    def test_cache_value_is_actually_used(self):
        """投毒缓存：注入伪 base，estimate 应返回伪值 + offset，证明走的是缓存。"""
        enc, lot, machine, ppid = _first_action()
        env = ResourceCalendarEnv(enc)
        env.reset()
        cache = {
            (lot, machine, ppid, 20): {
                "mu_finish": 12345.0, "sigma_finish": 7.0, "bottleneck_stage": 1,
                "per_instance_occupancy": [], "stage_mu": [1.0], "stage_sigma": [0.1],
                "n_batches": 1,
            }
        }
        r = estimate(lot, machine, ppid, enc, env.state, n_mc=20, start_offset=5.0, cache=cache)
        assert r["mu_finish"] == pytest.approx(12350.0)
        assert r["sigma_finish"] == pytest.approx(7.0)

    def test_no_cache_path_unchanged(self):
        """不传 cache → 行为不变，仍返回合理分布。"""
        enc, lot, machine, ppid = _first_action()
        env = ResourceCalendarEnv(enc)
        env.reset()
        r = estimate(lot, machine, ppid, enc, env.state, n_mc=20)
        assert r["mu_finish"] > 0
        assert r["sigma_finish"] >= 0


class TestEnvEstimateCache:
    def test_env_owns_and_clears_cache(self):
        enc = build_small_encoder()
        env = ResourceCalendarEnv(enc)
        env.reset()
        assert env._estimate_cache == {}
        # 构建候选池会触发 qtime mask / is_doomed → 填充缓存
        machine = env.get_candidate_machines()[0]
        env.build_candidate_pool(machine)
        assert len(env._estimate_cache) > 0
        # reset 清空
        env.reset()
        assert env._estimate_cache == {}
