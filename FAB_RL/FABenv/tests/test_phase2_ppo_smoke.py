import sys
from pathlib import Path

import numpy as np
import torch


PHASE1_DIR = Path(__file__).resolve().parents[1]
if str(PHASE1_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE1_DIR))


from phase2_ppo_buffer import Phase2RolloutBuffer, RolloutStep, StepInfo
from phase2_ppo_trainer import PPOConfig, Phase2PPOTrainer
from phase2_sas_policy import Phase2SASActorCritic
import train_phase2_sas_ppo


def test_ppo_update_runs_one_backward_step():
    policy = Phase2SASActorCritic(candidate_dim=18, global_dim=9, hidden_dim=32)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    trainer = Phase2PPOTrainer(policy, optimizer, PPOConfig(train_epochs=1, minibatch_size=2))
    buffer = Phase2RolloutBuffer()

    for action in [0, 1]:
        buffer.add(
            RolloutStep(
                machine_id=1,
                current_time=0.0,
                candidate_features=np.random.randn(4, 18).astype("float32"),
                candidate_mask=np.asarray([True, True, False, False]),
                global_features=np.random.randn(9).astype("float32"),
                action_indices=np.arange(4),
                valid_action_count=2,
                action=action,
                log_prob=-0.69,
                value=0.0,
                reward=1.0,
                done=False,
                next_observation=None,
                info=StepInfo(),
            )
        )
    buffer.finish_episode(last_value=0.0)

    stats = trainer.update_policy(buffer)

    assert "policy_loss" in stats
    assert "value_loss" in stats
    assert "entropy" in stats


class FakeEpisodeLogger:
    def __init__(self):
        self.rows = []

    def log(self, row):
        self.rows.append(dict(row))


def test_ppo_trainer_collects_episode_and_train_returns_history():
    torch.manual_seed(0)
    components = train_phase2_sas_ppo.build_training_components()
    buffer = Phase2RolloutBuffer()

    summary = components["trainer"].collect_episode(
        components["driver"],
        buffer,
        stochastic=False,
    )
    buffer.finish_episode(last_value=0.0)
    stats = components["trainer"].update_policy(buffer)

    assert summary["steps"] > 0
    assert buffer.steps
    assert {"policy_loss", "value_loss", "entropy"} <= set(stats)

    components = train_phase2_sas_ppo.build_training_components()
    history = components["trainer"].train(components["driver"], num_episodes=1)

    assert len(history) == 1
    assert history[0]["steps"] > 0
    assert "termination_reason" in history[0]


def test_ppo_trainer_logs_episode_rows():
    torch.manual_seed(0)
    components = train_phase2_sas_ppo.build_training_components()
    logger = FakeEpisodeLogger()

    history = components["trainer"].train(
        components["driver"],
        num_episodes=1,
        episode_logger=logger,
    )

    assert len(logger.rows) == 1
    assert logger.rows == history


def test_ppo_trainer_trains_with_generated_driver_factory():
    torch.manual_seed(0)
    components = train_phase2_sas_ppo.build_training_components(hidden_dim=32)
    driver_factory = train_phase2_sas_ppo.build_curriculum_driver_factory(
        top_k=8,
        observation_encoder=components["observation_encoder"],
        reward_config=components["reward_config"],
        split="train",
    )

    logger = FakeEpisodeLogger()
    history = components["trainer"].train_with_driver_factory(
        driver_factory,
        num_episodes=2,
        episode_logger=logger,
    )

    assert len(history) == 2
    assert logger.rows == history
    assert history[0]["seed"] == 0
    assert history[1]["seed"] == 1
    assert history[0]["difficulty"] == "easy"
    assert history[0]["steps"] > 0
    assert history[1]["steps"] > 0
    assert history[0]["num_lots"] > 0
    assert history[0]["num_machines"] > 0
