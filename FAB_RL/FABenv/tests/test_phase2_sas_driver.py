import sys
from pathlib import Path

import torch


PHASE1_DIR = Path(__file__).resolve().parents[1]
if str(PHASE1_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE1_DIR))


from phase2_ppo_buffer import Phase2RolloutBuffer
from phase2_sas_driver import Phase2EpisodeDriver
from phase2_sas_observation import Phase2ObservationEncoder
from phase2_sas_policy import Phase2SASActorCritic
from problem_instances import build_small_encoder
from rl_environment import ResourceCalendarEnv, RewardConfig


class PassThroughObservationEncoder:
    def encode(self, machine, pool, env):
        return env.build_sas_observation(machine)


def test_select_next_machine_uses_constrained_lexicographic_rule():
    encoder = build_small_encoder()
    env = ResourceCalendarEnv(encoder)
    driver = Phase2EpisodeDriver(env, PassThroughObservationEncoder(), RewardConfig())

    machines = driver.get_dispatchable_machines()
    selected = driver.select_next_machine(machines)

    def expected_key(machine):
        pool = env.build_candidate_pool(machine)
        real_count = sum(
            bool(is_valid) and not action.is_wait and not action.is_padding
            for action, is_valid in zip(pool.actions, pool.action_mask)
        )
        return (
            env.state.machine_available_time.get(machine, env.current_time),
            real_count,
            machine,
        )

    assert selected == min(machines, key=expected_key)


def test_run_rule_episode_with_first_valid_action_stops_cleanly():
    encoder = build_small_encoder()
    env = ResourceCalendarEnv(encoder)
    driver = Phase2EpisodeDriver(
        env,
        PassThroughObservationEncoder(),
        RewardConfig(),
        max_steps=200,
    )

    summary = driver.run_rule_episode(strategy="first_valid")

    assert summary["steps"] > 0
    assert summary["termination_reason"] in {
        "all_lots_completed",
        "no_future_event",
        "max_steps_exceeded",
        "planning_horizon_exceeded",
        "max_total_wait_steps_exceeded",
        "max_failed_actions_exceeded",
        "unrecoverable_error",
    }
    assert "episode_reward" in summary


def test_advance_to_next_event_moves_time_forward_and_counts_wait():
    encoder = build_small_encoder()
    env = ResourceCalendarEnv(encoder, current_time=0.5)
    driver = Phase2EpisodeDriver(env, PassThroughObservationEncoder(), RewardConfig())

    next_time = driver.advance_to_next_event()

    assert next_time == 1.5
    assert env.current_time == 1.5
    assert driver.total_wait_steps_per_episode == 1


def test_run_policy_episode_records_rollout_with_next_observation_and_info():
    torch.manual_seed(0)
    encoder = build_small_encoder()
    env = ResourceCalendarEnv(encoder, top_k=8)
    observation_encoder = Phase2ObservationEncoder()
    driver = Phase2EpisodeDriver(env, observation_encoder, RewardConfig(), max_steps=20)
    sample_pool = env.build_candidate_pool(1)
    sample_observation = observation_encoder.encode(1, sample_pool, env)
    policy = Phase2SASActorCritic(
        candidate_dim=sample_observation.candidate_features.shape[1],
        global_dim=sample_observation.global_features.shape[0],
        hidden_dim=32,
    )
    buffer = Phase2RolloutBuffer()

    summary = driver.run_policy_episode(policy, buffer=buffer, stochastic=False)

    assert summary["steps"] > 0
    assert buffer.steps
    step = buffer.steps[0]
    assert step.machine_id > 0
    assert step.action in step.action_indices.tolist()
    assert isinstance(step.info.raw, dict)
    assert "reward_total" in step.info.raw
    if not step.done:
        assert step.next_observation is None or hasattr(step.next_observation, "candidate_features")


def test_run_greedy_episode_uses_policy_probabilities_and_validates_schedule():
    torch.manual_seed(0)
    encoder = build_small_encoder()
    env = ResourceCalendarEnv(encoder, top_k=8)
    observation_encoder = Phase2ObservationEncoder()
    driver = Phase2EpisodeDriver(env, observation_encoder, RewardConfig(), max_steps=200)
    sample_pool = env.build_candidate_pool(1)
    sample_observation = observation_encoder.encode(1, sample_pool, env)
    policy = Phase2SASActorCritic(
        candidate_dim=sample_observation.candidate_features.shape[1],
        global_dim=sample_observation.global_features.shape[0],
        hidden_dim=32,
    )

    summary = driver.run_greedy_episode(policy)
    validation = env.validate_schedule(partial=True)

    assert summary["steps"] > 0
    assert summary["termination_reason"]
    assert validation.passed
