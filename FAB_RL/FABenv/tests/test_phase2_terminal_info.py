import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pytest
from problem_instances import build_small_encoder
from rl_environment import ResourceCalendarEnv, RewardConfig

class TestTerminalInfo:
    def _env(self):
        env = ResourceCalendarEnv(build_small_encoder(), top_k=8)
        env.reset()
        return env

    def test_success_step_has_terminal_fields(self):
        env = self._env()
        machine = env.get_candidate_machines()[0]
        pool = env.build_candidate_pool(machine)
        # 找第一个真实有效动作
        idx = next(i for i, (a, m) in enumerate(zip(pool.actions, pool.action_mask))
                   if m and not a.is_wait and not a.is_padding)
        result = env.sas_step(machine, idx, pool=pool, reward_config=RewardConfig())
        info = result.info
        assert "is_terminal" in info
        assert "num_lots" in info and info["num_lots"] == env.encoder.num_lots
        assert "completed_lots" in info
        assert "qtime_violation_count" in info
        assert "avg_machine_utilization" in info

    def test_is_terminal_false_when_lots_remain(self):
        env = self._env()
        machine = env.get_candidate_machines()[0]
        pool = env.build_candidate_pool(machine)
        idx = next(i for i, (a, m) in enumerate(zip(pool.actions, pool.action_mask))
                   if m and not a.is_wait and not a.is_padding)
        result = env.sas_step(machine, idx, pool=pool)
        # small encoder 有 4 个 lot，派 1 个后仍有剩余
        assert result.info["is_terminal"] is False
        assert result.info["completed_lots"] == 1

    def test_avg_utilization_nonnegative(self):
        env = self._env()
        machine = env.get_candidate_machines()[0]
        pool = env.build_candidate_pool(machine)
        idx = next(i for i, (a, m) in enumerate(zip(pool.actions, pool.action_mask))
                   if m and not a.is_wait and not a.is_padding)
        result = env.sas_step(machine, idx, pool=pool)
        assert result.info["avg_machine_utilization"] >= 0.0
        assert result.info["qtime_violation_count"] >= 0.0
