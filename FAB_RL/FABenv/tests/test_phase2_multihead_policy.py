import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
torch = pytest.importorskip("torch")
from phase2_sas_policy import Phase2SASMultiHeadActorCritic

CHANNELS = ("exec", "qtime", "util", "progress")

class TestMultiHeadPolicy:
    def _net(self):
        return Phase2SASMultiHeadActorCritic(candidate_dim=18, global_dim=9, hidden_dim=32, channels=CHANNELS)

    def _inputs(self, batch=2, pool=5):
        feats = torch.randn(batch, pool, 18)
        mask = torch.ones(batch, pool, dtype=torch.bool)
        glob = torch.randn(batch, 9)
        return feats, mask, glob

    def test_critic_values_four_heads(self):
        net = self._net()
        feats, mask, glob = self._inputs()
        values = net.critic_values(feats, mask, glob)
        assert set(values.keys()) == set(CHANNELS)
        for c in CHANNELS:
            assert values[c].shape == (2,)

    def test_sample_action_returns_values_dict(self):
        net = self._net()
        feats, mask, glob = self._inputs()
        out = net.sample_action(feats, mask, glob)
        assert "values" in out and set(out["values"].keys()) == set(CHANNELS)
        assert "action" in out and "log_prob" in out

    def test_evaluate_actions_returns_values_dict(self):
        net = self._net()
        feats, mask, glob = self._inputs()
        actions = torch.zeros(2, dtype=torch.long)
        out = net.evaluate_actions(feats, mask, glob, actions)
        assert set(out["values"].keys()) == set(CHANNELS)
        assert "log_prob" in out and "entropy" in out

    def test_greedy_action(self):
        net = self._net()
        feats, mask, glob = self._inputs()
        out = net.greedy_action(feats, mask, glob)
        assert "values" in out
