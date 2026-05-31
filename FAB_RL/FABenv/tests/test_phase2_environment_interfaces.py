import sys
from pathlib import Path


PHASE1_DIR = Path(__file__).resolve().parents[1]
if str(PHASE1_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE1_DIR))


from problem_instances import build_small_encoder
from rl_environment import ResourceCalendarEnv, SASObservation


def test_reset_restores_multi_episode_initial_state():
    encoder = build_small_encoder()
    env = ResourceCalendarEnv(encoder)
    pool = env.build_candidate_pool(1)
    valid_index = next(
        index
        for index, is_valid in enumerate(pool.action_mask)
        if bool(is_valid) and not pool.actions[index].is_wait
    )
    env.sas_step(1, valid_index, pool=pool)
    assert env.completed_lots

    summary = env.reset(current_time=3.0)

    assert summary["current_time"] == 3.0
    assert summary["completed_lots"] == set()
    assert summary["remaining_lots"] == set(range(1, encoder.num_lots + 1))
    assert env.current_time == 3.0
    assert env.completed_lots == set()
    assert env.lot_schedule.shape == (0, 5)
    assert env.wafer_schedule.shape == (0, 9)


def test_get_candidate_machines_returns_machines_with_valid_real_candidates():
    encoder = build_small_encoder()
    env = ResourceCalendarEnv(encoder)

    machines = env.get_candidate_machines()

    assert machines
    for machine in machines:
        pool = env.build_candidate_pool(machine)
        assert any(
            bool(is_valid) and not action.is_wait and not action.is_padding
            for action, is_valid in zip(pool.actions, pool.action_mask)
        )


def test_next_event_time_returns_future_arrival_or_release():
    encoder = build_small_encoder()
    env = ResourceCalendarEnv(encoder, current_time=0.5)

    assert env.next_event_time() == 1.5


def test_build_sas_observation_wraps_candidate_pool_and_feature_names():
    encoder = build_small_encoder()
    env = ResourceCalendarEnv(encoder)

    observation = env.build_sas_observation(1)

    assert isinstance(observation, SASObservation)
    assert observation.machine == 1
    assert observation.current_time == env.current_time
    assert observation.candidate_features.shape[1] == len(env.feature_names)
    assert observation.feature_names == env.feature_names
    assert observation.action_index_to_real_action
