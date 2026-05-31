import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import tempfile
import numpy as np
import pytest
torch = pytest.importorskip("torch")
from phase2_sas_policy import Phase2SASActorCritic, Phase2SASMultiHeadActorCritic
from phase2_ppo_buffer import MULTIHEAD_CHANNELS
from model_checkpoint import save_policy_checkpoint, load_policy_checkpoint

class TestCheckpoint:
    def test_single_head_roundtrip(self):
        net = Phase2SASActorCritic(candidate_dim=18, global_dim=9, hidden_dim=16)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.pt")
            save_policy_checkpoint(net, path, candidate_dim=18, global_dim=9,
                                   hidden_dim=16, policy_type="single")
            loaded, ckpt = load_policy_checkpoint(path)
            assert isinstance(loaded, Phase2SASActorCritic)
            assert ckpt["policy_type"] == "single"
            # 权重一致
            feats = torch.randn(2, 5, 18); mask = torch.ones(2,5,dtype=torch.bool); glob = torch.randn(2,9)
            with torch.no_grad():
                l1, v1 = net(feats, mask, glob)
                l2, v2 = loaded(feats, mask, glob)
            assert torch.allclose(l1, l2, atol=1e-6)

    def test_multihead_roundtrip(self):
        net = Phase2SASMultiHeadActorCritic(candidate_dim=18, global_dim=9, hidden_dim=16, channels=MULTIHEAD_CHANNELS)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "mh.pt")
            save_policy_checkpoint(net, path, candidate_dim=18, global_dim=9,
                                   hidden_dim=16, policy_type="multihead", channels=MULTIHEAD_CHANNELS)
            loaded, ckpt = load_policy_checkpoint(path)
            assert isinstance(loaded, Phase2SASMultiHeadActorCritic)
            assert ckpt["policy_type"] == "multihead"
            feats = torch.randn(2, 5, 18); mask = torch.ones(2,5,dtype=torch.bool); glob = torch.randn(2,9)
            with torch.no_grad():
                logits1, values1 = net(feats, mask, glob)
                logits2, values2 = loaded(feats, mask, glob)
            assert torch.allclose(logits1, logits2, atol=1e-6)
            for c in MULTIHEAD_CHANNELS:
                assert torch.allclose(values1[c], values2[c], atol=1e-6)
