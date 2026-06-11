"""优化③: 候选池两段式 qtime 预筛 (aggregate 粗筛 → 仅 TopK+裕量 跑 chain)。"""
import numpy as np


def _env_all_arrived(small_encoder, top_k, prescreen=False, margin=None):
    from rl_environment import ResourceCalendarEnv

    env = ResourceCalendarEnv(small_encoder, top_k=top_k)
    env.reset()
    env.advance_time(4.0)  # small 实例全部 4 个 lot 到达 (且均未 doomed)
    assert env.qtime_mask_mode == "chain_joint"
    if prescreen:
        env.qtime_mask_prescreen = True
    if margin is not None:
        env.qtime_prescreen_margin = int(margin)
    return env


def _chain_call_count(env, machine):
    calls = {"n": 0}
    original = env._chain_mask_wafer_schedule

    def counting(action, noise_rng=None):
        calls["n"] += 1
        return original(action, noise_rng=noise_rng)

    env._chain_mask_wafer_schedule = counting
    try:
        env.build_candidate_pool(machine)
    finally:
        del env.__dict__["_chain_mask_wafer_schedule"]
    return calls["n"]


def _pool_signature(pool):
    return [
        (int(a.lot), int(a.ppid), bool(a.is_wait), bool(a.is_padding), bool(m))
        for a, m in zip(pool.actions, pool.action_mask)
    ]


def test_prescreen_defaults_off(small_encoder):
    env = _env_all_arrived(small_encoder, top_k=2)
    assert getattr(env, "qtime_mask_prescreen", False) is False


def test_prescreen_reduces_chain_mask_calls(small_encoder):
    """margin=0 时只有 TopK 个候选接受 chain 检查 (全量是全部结构候选)。"""
    full_env = _env_all_arrived(small_encoder, top_k=2)
    pre_env = _env_all_arrived(small_encoder, top_k=2, prescreen=True, margin=0)

    full_calls = _chain_call_count(full_env, machine=1)
    pre_calls = _chain_call_count(pre_env, machine=1)

    assert full_calls > 0
    assert pre_calls < full_calls


def test_prescreen_with_large_margin_equals_full_pool(small_encoder):
    """裕量覆盖全部候选时, 两段式池与全量 chain mask 池完全一致。"""
    full_env = _env_all_arrived(small_encoder, top_k=2)
    pre_env = _env_all_arrived(small_encoder, top_k=2, prescreen=True, margin=10**6)

    full_pool = full_env.build_candidate_pool(1)
    pre_pool = pre_env.build_candidate_pool(1)

    assert _pool_signature(full_pool) == _pool_signature(pre_pool)
    assert np.array_equal(full_pool.features, pre_pool.features)
