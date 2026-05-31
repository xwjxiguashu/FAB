import sys
from pathlib import Path

import numpy as np


PHASE1_DIR = Path(__file__).resolve().parents[1]
if str(PHASE1_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE1_DIR))


from phase2_sas_observation import Phase2ObservationEncoder
from problem_instances import build_small_encoder
from rl_environment import ResourceCalendarEnv


def test_observation_encoder_outputs_fixed_shapes_and_rank_features():
    encoder = build_small_encoder()
    env = ResourceCalendarEnv(encoder, top_k=8)
    pool = env.build_candidate_pool(1)
    obs_encoder = Phase2ObservationEncoder()

    observation = obs_encoder.encode(1, pool, env)

    assert observation.machine_id == 1
    assert observation.candidate_features.shape == (8, len(env.feature_names))
    assert observation.candidate_mask.shape == (8,)
    assert observation.global_features.shape == (9,)
    assert observation.action_indices.tolist() == list(range(8))
    assert observation.valid_action_count == int(np.sum(pool.action_mask))


def test_batch_observations_stacks_numpy_arrays():
    encoder = build_small_encoder()
    env = ResourceCalendarEnv(encoder, top_k=8)
    obs_encoder = Phase2ObservationEncoder()
    obs1 = obs_encoder.encode(1, env.build_candidate_pool(1), env)
    obs2 = obs_encoder.encode(2, env.build_candidate_pool(2), env)

    batch = obs_encoder.batch_observations([obs1, obs2])

    assert batch["candidate_features"].shape == (2, 8, len(env.feature_names))
    assert batch["candidate_mask"].shape == (2, 8)
    assert batch["global_features"].shape == (2, 9)
