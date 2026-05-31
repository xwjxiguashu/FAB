"""派工规则基线 + 多 seed 评测 (报告 §7.4 基线对比 / §4.10 指标 / §2.4.6 多 rollout)。

基线: FIFO / SPT / EDD / CR / ATC，在与 RL 相同的 qtime-safe 候选池上排序选动作。
评测: 多 seed (加工噪声实现) × 多策略 → 聚合均值/标准差。
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from problem_instances import build_small_encoder
from rl_environment import ResourceCalendarEnv, RewardConfig
from phase2_sas_observation import Phase2ObservationEncoder
from phase2_sas_driver import Phase2EpisodeDriver
import evaluate_baselines as eb


RULES = ("FIFO", "SPT", "EDD", "CR", "ATC")


def _driver(noise_seed=None):
    enc = build_small_encoder()
    env = ResourceCalendarEnv(
        enc, process_noise_enabled=noise_seed is not None, noise_seed=noise_seed,
    )
    env.reset()
    return enc, env, Phase2EpisodeDriver(env, Phase2ObservationEncoder(), RewardConfig())


class TestRuleSelection:
    @pytest.mark.parametrize("rule", RULES)
    def test_rule_episode_completes_all_lots(self, rule):
        enc, env, driver = _driver()
        driver.reset_episode()
        summary = driver.run_rule_episode(strategy=rule)
        assert summary["completed_lots"] == enc.num_lots
        assert summary["termination_reason"] == "all_lots_completed"
        # 排程完整性校验通过
        assert enc.validate_final_schedule_completeness(env.lot_schedule, env.wafer_schedule)

    def test_unknown_strategy_raises(self):
        enc, env, driver = _driver()
        driver.reset_episode()
        with pytest.raises(ValueError):
            driver.run_rule_episode(strategy="NOPE")


class TestMetrics:
    def test_objective_metrics_extracted(self):
        enc, env, driver = _driver()
        driver.reset_episode()
        driver.run_rule_episode(strategy="EDD")
        m = eb.schedule_metrics(enc, env)
        for key in ("qtime_violation_count", "total_tardiness", "avg_utilization",
                    "priority_violation", "completed_lots"):
            assert key in m
        assert 0.0 <= m["avg_utilization"] <= 1.0 + 1e-9
        assert m["completed_lots"] == enc.num_lots


class TestHarness:
    def test_evaluate_baselines_aggregates_over_seeds(self):
        results = eb.evaluate(
            strategies=("FIFO", "EDD", "ATC"),
            seeds=(0, 1, 2),
            encoder_factory=build_small_encoder,
        )
        # 每个策略一行，含均值/标准差聚合
        for strat in ("FIFO", "EDD", "ATC"):
            assert strat in results
            row = results[strat]
            assert "avg_utilization_mean" in row
            assert "qtime_violation_count_mean" in row
            assert "n_seeds" in row and row["n_seeds"] == 3
            assert "avg_utilization_std" in row
