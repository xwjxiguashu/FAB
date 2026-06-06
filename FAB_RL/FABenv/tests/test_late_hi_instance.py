"""Tests for the late_hi discriminating instance (报告4 §9.8 go/no-go gate).

The go/no-go oracle验证 must run on an instance where high-priority lots
arrive late (corr≈0.97), because only that structure gives reservation any
leverage. The default pressure instance uses uncorrelated U(0,10) priorities,
which cannot discriminate "reservation useless" from "instance has no lever".
"""
import numpy as np

from problem_instances import build_late_hi_encoder, build_pressure_test_encoder


def _priority_arrival_corr(encoder):
    lots = sorted(encoder.arrival_times)
    arrivals = np.array([encoder.arrival_times[l] for l in lots], dtype=float)
    priorities = np.array([encoder.priorities[l] for l in lots], dtype=float)
    return float(np.corrcoef(arrivals, priorities)[0, 1])


def test_late_hi_priority_correlates_with_late_arrival():
    encoder = build_late_hi_encoder(seed=2026)
    corr = _priority_arrival_corr(encoder)
    assert corr > 0.9, f"late_hi must couple high priority with late arrival, corr={corr:.3f}"


def test_late_hi_priorities_stay_in_range():
    encoder = build_late_hi_encoder(seed=2026)
    values = list(encoder.priorities.values())
    assert all(0.0 <= p <= 10.0 for p in values)


def test_late_hi_keeps_pressure_structure():
    """late_hi reuses the pressure scaffold: 50 lots, qtime limits live."""
    encoder = build_late_hi_encoder(seed=2026)
    assert encoder.num_lots == 50
    assert encoder.num_machines == 10
    assert len(encoder.q_time_limits) > 0


def test_default_pressure_priorities_remain_uncorrelated():
    """Adding late_hi must not change the default random-priority behavior."""
    encoder = build_pressure_test_encoder(seed=2026)
    corr = _priority_arrival_corr(encoder)
    assert abs(corr) < 0.6, f"default pressure should stay uncorrelated, corr={corr:.3f}"


def test_probe_workers_match_serial():
    """--workers parallelizes across seeds and must match the serial result."""
    from oracle_reservation_probe import main

    serial = main(instance="small", seeds=2, max_steps=200, out=None, workers=1)
    parallel = main(instance="small", seeds=2, max_steps=200, out=None, workers=2)

    def by_seed(rows):
        return {
            r["seed"]: round(r["delta"]["priority_weighted_wait"], 6)
            for r in rows
        }

    assert len(parallel) == 2
    assert by_seed(parallel) == by_seed(serial)


def test_oracle_full_horizon_lookahead_covers_all_arrivals():
    """The oracle must be information-complete: its lookahead spans every arrival."""
    from oracle_reservation_probe import _full_horizon_lookahead

    encoder = build_late_hi_encoder(seed=2026)
    lookahead = _full_horizon_lookahead(encoder)
    assert lookahead > max(encoder.arrival_times.values())


def test_late_hi_wired_into_eval_and_probe_factories():
    """Both evaluation harnesses must resolve the late_hi instance."""
    from evaluate_baselines import ENCODER_FACTORIES
    from oracle_reservation_probe import _encoder_factory

    eval_encoder = ENCODER_FACTORIES["late_hi"]()
    probe_encoder = _encoder_factory("late_hi")()

    for encoder in (eval_encoder, probe_encoder):
        assert encoder.num_lots == 50
        assert _priority_arrival_corr(encoder) > 0.9
