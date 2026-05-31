import sys
from pathlib import Path

import numpy as np


PHASE1_DIR = Path(__file__).resolve().parents[1]
if str(PHASE1_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE1_DIR))


from phase2_sas_observation import Phase2ObservationEncoder
from problem_generator import (
    build_easy_config,
    build_hard_config,
    build_random_encoder,
    sample_random_problem_config,
)
from rl_environment import ResourceCalendarEnv


def _encoder_signature(encoder):
    process_signature = []
    for key in sorted(encoder.ppid_steps):
        stage_signature = tuple(
            tuple(tuple(float(value) for value in row) for row in np.asarray(stage))
            for stage in encoder.ppid_steps[key]
        )
        process_signature.append((key, stage_signature))
    return {
        "num_lots": encoder.num_lots,
        "num_machines": encoder.num_machines,
        "arrival_times": encoder.arrival_times,
        "due_dates": encoder.due_dates,
        "priorities": encoder.priorities,
        "feasible_machines": encoder.feasible_machines,
        "feasible_ppids": encoder.feasible_ppids,
        "ppid_steps": tuple(process_signature),
        "q_time_limits": encoder.q_time_limits,
    }


def test_random_encoder_is_reproducible_for_the_same_seed():
    config = build_easy_config(seed=123)

    first = build_random_encoder(config)
    second = build_random_encoder(config)

    assert _encoder_signature(first) == _encoder_signature(second)


def test_random_encoder_changes_when_seed_changes():
    first = build_random_encoder(build_easy_config(seed=123))
    second = build_random_encoder(build_easy_config(seed=124))

    assert _encoder_signature(first) != _encoder_signature(second)


def test_random_encoder_builds_valid_environment_and_observation_shapes():
    encoder = build_random_encoder(build_easy_config(seed=7))
    env = ResourceCalendarEnv(encoder, top_k=8)
    observation_encoder = Phase2ObservationEncoder()

    machines = env.get_candidate_machines()
    assert machines

    machine = machines[0]
    pool = env.build_candidate_pool(machine)
    observation = observation_encoder.encode(machine, pool, env)

    assert observation.candidate_features.shape == (8, len(env.feature_names))
    assert observation.candidate_features.shape[1] == 18
    assert observation.global_features.shape == (9,)
    assert observation.valid_action_count > 0


def test_curriculum_config_uses_split_seed_ranges_and_difficulty_metadata():
    train_config = sample_random_problem_config(episode=0, split="train")
    validation_config = sample_random_problem_config(episode=0, split="validation")
    test_config = sample_random_problem_config(episode=0, split="test")
    later_config = sample_random_problem_config(episode=200, split="train")

    assert train_config.seed == 0
    assert validation_config.seed == 10000
    assert test_config.seed == 20000
    assert train_config.difficulty == "easy"
    assert later_config.difficulty in {"medium", "hard"}
    assert later_config.num_lots >= train_config.num_lots
    assert later_config.num_machines >= train_config.num_machines


def test_hard_config_is_tighter_and_sparser_than_easy_config():
    easy = build_easy_config(seed=1)
    hard = build_hard_config(seed=1)

    assert hard.num_lots > easy.num_lots
    assert hard.num_machines > easy.num_machines
    assert hard.due_tightness < easy.due_tightness
    assert hard.qtime_probability > easy.qtime_probability
    assert hard.machine_eligibility_ratio < easy.machine_eligibility_ratio
