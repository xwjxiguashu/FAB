import sys
from pathlib import Path


PHASE1_DIR = Path(__file__).resolve().parents[1]
if str(PHASE1_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE1_DIR))


from problem_instances import build_small_encoder
from rl_environment import RewardConfig, ResourceCalendarEnv, compute_sas_reward


def test_r0_success_reward_records_execute_and_total():
    info = {
        "insertion_success": True,
        "insertion_failed": False,
        "mask_invalid": False,
        "wait_or_noop": False,
        "selected_lot_start": 10.0,
        "selected_lot_end": 20.0,
        "selected_lot_process_time": 10.0,
        "current_time": 10.0,
        "due_date": 100.0,
        "new_qtime_violation": 0.0,
        "priority_rank_penalty": 0.0,
    }

    reward = compute_sas_reward(info, RewardConfig())

    assert reward == 0.20
    assert info["reward_execute"] == 0.20
    assert info["reward_wait"] == 0.0
    assert info["reward_tardy"] == 0.0
    assert info["reward_qtime"] == 0.0
    assert info["reward_priority"] == 0.0
    assert info["reward_progress"] == 0.0
    assert info["reward_shape"] == 0.0
    assert info["reward_terminal"] == 0.0
    assert info["reward_total"] == 0.20


def test_r1_light_shaping_records_each_component():
    info = {
        "insertion_success": True,
        "insertion_failed": False,
        "mask_invalid": False,
        "wait_or_noop": False,
        "selected_lot_start": 10.0,
        "selected_lot_end": 30.0,
        "selected_lot_process_time": 20.0,
        "current_time": 10.0,
        "due_date": 20.0,
        "new_qtime_violation": 5.0,
        "priority_rank_penalty": 2.0,
    }
    config = RewardConfig(use_light_shaping=True)

    reward = compute_sas_reward(info, config)

    assert info["reward_execute"] == 0.20
    assert info["reward_tardy"] == -0.025
    assert info["reward_qtime"] == -0.02
    assert info["reward_priority"] == -0.06
    assert info["reward_progress"] == 0.01
    assert info["reward_shape"] == -0.095
    assert reward == 0.105
    assert info["reward_total"] == 0.105


def test_r2_terminal_reward_only_applies_when_episode_done():
    info = {
        "insertion_success": False,
        "insertion_failed": False,
        "mask_invalid": False,
        "wait_or_noop": False,
        "episode_done": True,
        "tardy_lot_count_norm": 0.5,
        "total_tardiness_norm": 0.25,
        "qtime_violation_count_norm": 0.2,
        "machine_utilization_norm": 0.8,
        "priority_violation_norm": 0.4,
    }
    config = RewardConfig(use_terminal_reward=True)

    reward = compute_sas_reward(info, config)

    assert info["reward_terminal"] == -0.135
    assert reward == -0.135
    assert info["reward_total"] == -0.135


def test_sas_step_info_contains_complete_reward_decomposition():
    encoder = build_small_encoder()
    env = ResourceCalendarEnv(encoder)
    pool = env.build_candidate_pool(1)
    valid_index = next(
        index
        for index, is_valid in enumerate(pool.action_mask)
        if bool(is_valid) and not pool.actions[index].is_wait
    )

    result = env.sas_step(1, valid_index, pool=pool, reward_config=RewardConfig())

    assert result.committed is True
    for key in (
        "reward_execute",
        "reward_wait",
        "reward_tardy",
        "reward_qtime",
        "reward_priority",
        "reward_progress",
        "reward_shape",
        "reward_terminal",
        "reward_total",
    ):
        assert key in result.info
    assert result.reward == result.info["reward_total"]
