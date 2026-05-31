import sys
from pathlib import Path


PHASE1_DIR = Path(__file__).resolve().parents[1]
if str(PHASE1_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE1_DIR))


from phase2_ppo_buffer import Phase2RolloutBuffer, RolloutStep, StepInfo

import numpy as np


def _step(reward, value, done=False):
    return RolloutStep(
        machine_id=1,
        current_time=0.0,
        candidate_features=None,
        candidate_mask=None,
        global_features=None,
        action_indices=None,
        valid_action_count=1,
        action=0,
        log_prob=0.0,
        value=value,
        reward=reward,
        done=done,
        next_observation=None,
        info=StepInfo(),
    )


def test_buffer_computes_returns_and_advantages():
    buffer = Phase2RolloutBuffer(gamma=1.0, gae_lambda=1.0)
    buffer.add(_step(1.0, 0.5))
    buffer.add(_step(2.0, 0.25, done=True))

    buffer.finish_episode(last_value=0.0)

    assert buffer.returns == [3.0, 2.0]
    assert buffer.advantages == [2.5, 1.75]


def test_buffer_yields_training_batches_with_rollout_metadata():
    buffer = Phase2RolloutBuffer(gamma=1.0, gae_lambda=1.0)
    buffer.add(
        RolloutStep(
            machine_id=1,
            current_time=0.0,
            candidate_features=np.zeros((4, 18), dtype=np.float32),
            candidate_mask=np.asarray([True, False, False, False]),
            global_features=np.zeros(9, dtype=np.float32),
            action_indices=np.arange(4),
            valid_action_count=1,
            action=0,
            log_prob=-0.1,
            value=0.5,
            reward=1.0,
            done=True,
            next_observation=None,
            info=StepInfo(selected_lot=1, reward_total=1.0),
        )
    )
    buffer.finish_episode(last_value=0.0)

    batch = next(buffer.get_training_batches(batch_size=1))

    assert batch["machine_id"].tolist() == [1]
    assert batch["current_time"].tolist() == [0.0]
    assert batch["candidate_features"].shape == (1, 4, 18)
    assert batch["candidate_mask"].shape == (1, 4)
    assert batch["global_features"].shape == (1, 9)
    assert batch["actions"].tolist() == [0]
    assert np.allclose(batch["old_log_probs"], [-0.1])
    assert np.allclose(batch["returns"], [1.0])
    assert np.allclose(batch["advantages"], [0.5])
