"""Q-time chain mask must not consume execution-noise RNG."""

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
    env.qtime_mask_mode = "chain"
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


def test_chain_mask_dry_run_does_not_advance_commit_noise_rng():
    lot_once, wafer_once = _first_commit_after_extra_pool_builds(extra_builds=0)
    lot_twice, wafer_twice = _first_commit_after_extra_pool_builds(extra_builds=1)

    assert np.array_equal(lot_once, lot_twice)
    assert np.array_equal(wafer_once, wafer_twice)
