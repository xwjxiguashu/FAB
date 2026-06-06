from phase2_sas_driver import Phase2EpisodeDriver
from phase2_sas_observation import Phase2ObservationEncoder
from reservation_ledger import ReservationLedger
from reservation_simulator import (
    finalize_reservation_ledger,
    is_reservation_rollout_better,
    run_rule_episode_with_reservations,
    schedule_metrics_with_priority_wait,
)
from rl_environment import RewardConfig, ResourceCalendarEnv


def _driver(env):
    return Phase2EpisodeDriver(env, Phase2ObservationEncoder(), RewardConfig())


def test_schedule_metrics_include_priority_weighted_wait(small_encoder):
    env = ResourceCalendarEnv(small_encoder, top_k=8)
    env.reset()
    driver = _driver(env)
    driver.reset_episode()
    driver.run_rule_episode(strategy="FIFO")

    metrics = schedule_metrics_with_priority_wait(small_encoder, env)

    assert metrics["completed_lots"] == 4.0
    assert metrics["priority_weighted_wait"] >= 0.0
    assert "avg_utilization" in metrics


def test_forced_reservation_consumes_target_lot_when_it_arrives(small_encoder):
    env = ResourceCalendarEnv(small_encoder, top_k=8, w_lookahead=4.0)
    env.reset()
    ledger = ReservationLedger()
    ledger.reserve(machine=1, future_lot=2, eta=1.5, created_at=0.0, expires_at=4.0)

    summary = run_rule_episode_with_reservations(
        _driver(env),
        ledger=ledger,
        strategy="FIFO",
        max_steps=200,
    )

    lot2_rows = env.lot_schedule[env.lot_schedule[:, 0] == 2]
    assert summary["completed_lots"] == 4
    assert lot2_rows.shape[0] == 1
    assert int(lot2_rows[0, 1]) == 1
    assert not ledger.is_reserved(1)


def test_rollout_comparison_rejects_qtime_gain_when_priority_wait_regresses():
    baseline = {
        "qtime_violation_count": 20.0,
        "priority_weighted_wait": 1664.0,
        "avg_utilization": 0.86,
    }
    reserve = {
        "qtime_violation_count": 6.0,
        "priority_weighted_wait": 2321.0,
        "avg_utilization": 0.71,
    }

    assert not is_reservation_rollout_better(baseline, reserve)


def test_rollout_comparison_rejects_worse_qtime_even_with_priority_gain():
    baseline = {
        "qtime_violation_count": 0.0,
        "priority_weighted_wait": 10.0,
        "avg_utilization": 0.6,
    }
    reserve = {
        "qtime_violation_count": 1.0,
        "priority_weighted_wait": 1.0,
        "avg_utilization": 0.9,
    }

    assert not is_reservation_rollout_better(baseline, reserve)


def test_rollout_comparison_prefers_lower_priority_wait_when_qtime_ties():
    baseline = {
        "qtime_violation_count": 0.0,
        "priority_weighted_wait": 10.0,
        "avg_utilization": 0.6,
    }
    reserve = {
        "qtime_violation_count": 0.0,
        "priority_weighted_wait": 5.0,
        "avg_utilization": 0.4,
    }

    assert is_reservation_rollout_better(baseline, reserve)


def test_finalize_reservation_ledger_clears_when_all_lots_completed(small_env):
    ledger = ReservationLedger()
    ledger.reserve(machine=1, future_lot=99, eta=99.0, created_at=0.0, expires_at=100.0)
    small_env.remaining_lots = set()

    released = finalize_reservation_ledger(ledger, small_env)

    assert [r.machine for r in released] == [1]
    assert ledger.reserved_machines() == set()
