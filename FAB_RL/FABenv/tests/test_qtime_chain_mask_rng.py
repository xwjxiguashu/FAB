"""Q-time chain_joint mask must not consume execution-noise RNG."""

import numpy as np

from problem_instances import build_late_hi_encoder
from rl_environment import ResourceCalendarEnv


def _first_commit_after_extra_pool_builds(extra_builds):
    env = ResourceCalendarEnv(
        build_late_hi_encoder(),
        top_k=8,
        process_noise_enabled=True,
        noise_seed=7,
    )
    env.reset()
    assert env.qtime_mask_mode == "chain_joint"
    machine = 1

    for _ in range(extra_builds):
        env.build_candidate_pool(machine)

    pool = env.build_candidate_pool(machine)
    action_index = next(
        index
        for index, (action, is_valid) in enumerate(zip(pool.actions, pool.action_mask))
        if bool(is_valid)
        and not action.is_wait
        and not action.is_padding
        and int(action.ppid) != 0
    )
    result = env.commit_action_index(machine, action_index, pool=pool)
    return result.lot_schedule.copy(), result.wafer_schedule.copy()


def test_chain_joint_mask_dry_run_does_not_advance_commit_noise_rng():
    lot_once, wafer_once = _first_commit_after_extra_pool_builds(extra_builds=0)
    lot_twice, wafer_twice = _first_commit_after_extra_pool_builds(extra_builds=1)

    assert np.array_equal(lot_once, lot_twice)
    assert np.array_equal(wafer_once, wafer_twice)


# ---------------------------------------------------------------------------
# 优化④: chain mask 免拷贝轻量 dry-run
# ---------------------------------------------------------------------------

def _real_pool_actions(env, machine):
    pool = env.build_candidate_pool(machine)
    return [
        action
        for action, is_valid in zip(pool.actions, pool.action_mask)
        if bool(is_valid)
        and not action.is_wait
        and not action.is_padding
        and int(action.ppid) != 0
    ]


def test_lightweight_chain_dryrun_matches_full_dry_run(small_encoder):
    """同种子 rng 下轻量路径的 wafer_schedule 必须与 dry_run_action 完全一致。"""
    env = ResourceCalendarEnv(small_encoder, top_k=8)
    env.reset()
    actions = _real_pool_actions(env, machine=1)
    assert actions

    for action in actions:
        rng_full = np.random.default_rng(123)
        rng_light = np.random.default_rng(123)
        full = env.dry_run_action(action, noise_rng=rng_full)
        light = env._chain_mask_wafer_schedule(action, noise_rng=rng_light)
        if not full.success:
            assert light is None
            continue
        assert light is not None
        assert np.array_equal(np.asarray(full.wafer_schedule), light)

    # 均值路径 (noise_rng=None) 同样一致
    action = actions[0]
    full = env.dry_run_action(action)
    light = env._chain_mask_wafer_schedule(action)
    assert np.array_equal(np.asarray(full.wafer_schedule), light)


def test_chain_joint_mask_does_not_mutate_real_state(small_encoder):
    import copy

    env = ResourceCalendarEnv(small_encoder, top_k=8)
    env.reset()
    assert env.qtime_mask_mode == "chain_joint"
    actions = _real_pool_actions(env, machine=1)
    assert actions

    before_machine = copy.deepcopy(env.state.machine_calendar)
    before_chamber = copy.deepcopy(env.state.chamber_calendar)

    env._qtime_chain_joint_mask(1, actions)
    env._qtime_chain_mask(1, actions)

    assert env.state.machine_calendar == before_machine
    assert env.state.chamber_calendar == before_chamber
