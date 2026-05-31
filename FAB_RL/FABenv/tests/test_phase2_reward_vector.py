import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pytest
from rl_environment import RewardVectorConfig, compute_sas_reward_vector


class TestRewardVector:
    def test_success_step_exec_only(self):
        info = {"insertion_success": True, "mask_invalid": False,
                "insertion_failed": False, "wait_or_noop": False, "is_terminal": False}
        out = compute_sas_reward_vector(info)
        v = out["reward_vector"]
        assert v.shape == (4,)
        assert v[0] == pytest.approx(0.20)
        assert v[1] == 0.0 and v[2] == 0.0 and v[3] == 0.0

    def test_failed_step(self):
        info = {"insertion_failed": True, "is_terminal": False}
        out = compute_sas_reward_vector(info)
        assert out["reward_vector"][0] == pytest.approx(-0.40)

    def test_mask_invalid_step(self):
        info = {"mask_invalid": True, "is_terminal": False}
        out = compute_sas_reward_vector(info)
        assert out["reward_vector"][0] == pytest.approx(-0.50)

    def test_wait_step_zero(self):
        info = {"wait_or_noop": True, "is_terminal": False}
        out = compute_sas_reward_vector(info)
        assert out["reward_vector"][0] == 0.0

    def test_terminal_channels(self):
        info = {"insertion_success": True, "is_terminal": True,
                "qtime_violation_count": 2.0, "num_lots": 10,
                "avg_machine_utilization": 0.7, "completed_lots": 8}
        out = compute_sas_reward_vector(info)
        v = out["reward_vector"]
        assert v[0] == pytest.approx(0.20)          # exec
        assert v[1] == pytest.approx(-0.2)          # qtime = -(2/10)
        assert v[2] == pytest.approx(0.7)           # util
        assert v[3] == pytest.approx(0.8)           # progress = 8/10

    def test_non_terminal_soft_channels_zero(self):
        info = {"insertion_success": True, "is_terminal": False,
                "qtime_violation_count": 5.0, "num_lots": 10}
        out = compute_sas_reward_vector(info)
        v = out["reward_vector"]
        assert v[1] == 0.0 and v[2] == 0.0 and v[3] == 0.0

    def test_config_custom_weights(self):
        cfg = RewardVectorConfig(w_qtime=5.0)
        assert cfg.w_qtime == 5.0
        assert cfg.channels == ("exec", "qtime", "util", "progress")
