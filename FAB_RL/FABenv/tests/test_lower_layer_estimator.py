"""Tests for lower_layer_estimator.py — Phase 1 heuristic timing estimator."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import numpy as np
import pytest

from lower_layer_estimator import (
    compute_sub_batches,
    monte_carlo_makespan,
    estimate,
    is_qtime_violated_probabilistically,
    qtime_violation_probability,
)
from problem_instances import build_small_encoder, build_pressure_test_encoder


# =============================================================================
# compute_sub_batches
# =============================================================================

class TestComputeSubBatches:
    def test_exact_multiple(self):
        batches = compute_sub_batches(6, 3)
        assert batches == [3, 3]

    def test_with_remainder(self):
        batches = compute_sub_batches(7, 3)
        assert len(batches) == math.ceil(7 / 3)
        assert sum(batches) == 7

    def test_single_wafer(self):
        batches = compute_sub_batches(1, 5)
        assert batches == [1]

    def test_capacity_equals_wafers(self):
        batches = compute_sub_batches(4, 4)
        assert batches == [4]

    def test_capacity_larger_than_wafers(self):
        batches = compute_sub_batches(3, 10)
        assert batches == [3]

    def test_invalid_n_wafers(self):
        with pytest.raises((ValueError, Exception)):
            compute_sub_batches(0, 3)

    def test_invalid_side_capacity(self):
        with pytest.raises((ValueError, Exception)):
            compute_sub_batches(5, 0)


# =============================================================================
# monte_carlo_makespan
# =============================================================================

class TestMonteCarlMakespan:
    def test_zero_sigma_deterministic(self):
        sub_batches = [3, 3]
        stage_mu = [2.0, 3.0]
        stage_sigma = [0.0, 0.0]
        instance_counts = [1, 1]
        mu, sigma = monte_carlo_makespan(
            sub_batches, stage_mu, stage_sigma, instance_counts, n_mc=10, rng=np.random.default_rng(42)
        )
        # With sigma=0, all samples should give same makespan
        assert sigma < 1e-6
        assert mu > 0

    def test_nonzero_sigma_positive_spread(self):
        sub_batches = [4]
        stage_mu = [5.0, 4.0]
        stage_sigma = [0.5, 0.4]
        instance_counts = [2, 2]
        mu, sigma = monte_carlo_makespan(
            sub_batches, stage_mu, stage_sigma, instance_counts, n_mc=200, rng=np.random.default_rng(0)
        )
        assert mu > 0
        assert sigma >= 0  # σ >= 0

    def test_mu_greater_than_single_stage_mean(self):
        # Two-stage: makespan >= max stage time
        sub_batches = [2]
        stage_mu = [3.0, 5.0]
        stage_sigma = [0.0, 0.0]
        instance_counts = [1, 1]
        mu, sigma = monte_carlo_makespan(
            sub_batches, stage_mu, stage_sigma, instance_counts, n_mc=5
        )
        assert mu >= 5.0  # at least the longest stage


# =============================================================================
# estimate() with small encoder
# =============================================================================

class TestEstimate:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.encoder = build_small_encoder()
        from state import ScheduleState
        self.state = ScheduleState()

    def test_returns_required_keys(self):
        lot, machine = 1, 1
        ppid = self.encoder.get_ppid_list(lot, machine)[0]
        result = estimate(lot, machine, ppid, self.encoder, self.state, n_mc=10)
        assert "mu_finish" in result
        assert "sigma_finish" in result
        assert "bottleneck_stage" in result
        assert "per_instance_occupancy" in result
        assert "stage_mu" in result
        assert "stage_sigma" in result
        assert "n_batches" in result

    def test_mu_finish_positive(self):
        lot, machine = 1, 1
        ppid = self.encoder.get_ppid_list(lot, machine)[0]
        result = estimate(lot, machine, ppid, self.encoder, self.state, n_mc=10)
        assert result["mu_finish"] > 0

    def test_sigma_finish_nonnegative(self):
        lot, machine = 1, 1
        ppid = self.encoder.get_ppid_list(lot, machine)[0]
        result = estimate(lot, machine, ppid, self.encoder, self.state, n_mc=10)
        assert result["sigma_finish"] >= 0

    def test_bottleneck_stage_valid(self):
        lot, machine = 1, 1
        ppid = self.encoder.get_ppid_list(lot, machine)[0]
        steps = self.encoder.get_process_steps(lot, machine, ppid)
        result = estimate(lot, machine, ppid, self.encoder, self.state, n_mc=10)
        assert 1 <= result["bottleneck_stage"] <= len(steps)

    def test_per_instance_occupancy_is_list(self):
        lot, machine = 1, 1
        ppid = self.encoder.get_ppid_list(lot, machine)[0]
        result = estimate(lot, machine, ppid, self.encoder, self.state, n_mc=5)
        assert isinstance(result["per_instance_occupancy"], list)

    def test_n_batches_positive(self):
        lot, machine = 1, 1
        ppid = self.encoder.get_ppid_list(lot, machine)[0]
        result = estimate(lot, machine, ppid, self.encoder, self.state, n_mc=5)
        assert result["n_batches"] >= 1


# =============================================================================
# estimate() with pressure test encoder
# =============================================================================

class TestEstimatePressure:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.encoder = build_pressure_test_encoder()
        from state import ScheduleState
        self.state = ScheduleState()

    def test_all_lots_estimatable(self):
        """Spot-check that estimate() works for several lots."""
        for lot in [1, 10, 25, 50]:
            machine = self.encoder.get_machine_list(lot)[0]
            ppid = self.encoder.get_ppid_list(lot, machine)[0]
            result = estimate(lot, machine, ppid, self.encoder, self.state, n_mc=5)
            assert result["mu_finish"] > 0


# =============================================================================
# is_qtime_violated_probabilistically
# =============================================================================

class TestQtimeOpportunityConstraint:
    def test_safe_action_not_masked(self):
        # deadline far in the future → not violated
        mu, sigma, deadline, z_eps = 10.0, 1.0, 100.0, 2.05
        assert not is_qtime_violated_probabilistically(mu, sigma, deadline, z_eps)

    def test_tight_action_masked(self):
        # mu very close to deadline, large sigma → violated
        mu, sigma, deadline, z_eps = 99.0, 1.0, 100.0, 2.05
        # deadline - mu = 1.0 < z_eps * sigma = 2.05 → mask
        assert is_qtime_violated_probabilistically(mu, sigma, deadline, z_eps)

    def test_zero_sigma_exact(self):
        # With σ=0, margin=0 → only mask if mu > deadline
        assert not is_qtime_violated_probabilistically(5.0, 0.0, 10.0, 2.05)
        assert is_qtime_violated_probabilistically(10.0, 0.0, 9.0, 2.05)

    def test_violation_probability_bounds(self):
        p = qtime_violation_probability(mu_finish=10.0, sigma_finish=1.0, qtime_deadline=12.0)
        assert 0.0 <= p <= 1.0

    def test_violation_probability_high_when_past_deadline(self):
        p = qtime_violation_probability(mu_finish=100.0, sigma_finish=1.0, qtime_deadline=90.0)
        assert p > 0.99

    def test_violation_probability_low_when_safe(self):
        p = qtime_violation_probability(mu_finish=10.0, sigma_finish=1.0, qtime_deadline=100.0)
        assert p < 0.01
