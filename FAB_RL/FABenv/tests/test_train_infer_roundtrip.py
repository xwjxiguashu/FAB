import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import tempfile
import pytest
torch = pytest.importorskip("torch")
from train_phase2_sas_ppo import main as train_main
from run_phase2_sas_inference_demo import run_demo_episode

class TestRoundtrip:
    def test_single_train_save_load_infer(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "single.pt")
            train_main(num_episodes=1, mode="small", save_path=path)
            assert os.path.exists(path)
            summary = run_demo_episode(max_steps=300, checkpoint_path=path)
            assert "validation_passed" in summary

    def test_multihead_train_save_load_infer(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "mh.pt")
            train_main(num_episodes=1, mode="multihead", save_path=path)
            assert os.path.exists(path)
            summary = run_demo_episode(max_steps=300, checkpoint_path=path)
            assert "validation_passed" in summary
