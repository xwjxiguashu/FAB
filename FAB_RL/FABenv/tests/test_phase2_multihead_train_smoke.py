import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
torch = pytest.importorskip("torch")
from train_phase2_sas_ppo import build_multihead_training_components
from phase2_ppo_trainer import MultiHeadPPOTrainer
from phase2_sas_policy import Phase2SASMultiHeadActorCritic


class TestMultiHeadTrainSmoke:
    def test_components_built(self):
        comp = build_multihead_training_components(hidden_dim=16)
        assert isinstance(comp["trainer"], MultiHeadPPOTrainer)
        assert isinstance(comp["policy"], Phase2SASMultiHeadActorCritic)

    def test_train_two_episodes(self):
        comp = build_multihead_training_components(hidden_dim=16)
        history = comp["trainer"].train(
            comp["driver"], num_episodes=2,
            reward_vector_config=comp["reward_vector_config"])
        assert len(history) == 2
        for row in history:
            assert "policy_loss" in row and "value_loss" in row and "entropy" in row
            assert "episode" in row

    def test_train_with_lookahead_and_noise(self):
        comp = build_multihead_training_components(
            hidden_dim=16, lookahead=True, w_lookahead=2.0,
            process_noise_enabled=True, noise_seed=42)
        history = comp["trainer"].train(
            comp["driver"], num_episodes=1,
            reward_vector_config=comp["reward_vector_config"])
        assert len(history) == 1
