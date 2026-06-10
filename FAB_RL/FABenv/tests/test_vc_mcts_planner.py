import json
from pathlib import Path

import pytest

from dispatch_delegate import RuleDispatchDelegate
from phase2_sas_driver import Phase2EpisodeDriver
from phase2_sas_observation import Phase2ObservationEncoder
from reservation_ledger import ReservationLedger
from reservation_simulator import (
    advance_to_next_event_with_ledger,
    clone_driver_for_rollout,
    clone_ledger_for_rollout,
)
from rl_environment import ResourceCalendarEnv, RewardConfig


def _driver(env, max_steps=200):
    return Phase2EpisodeDriver(
        env,
        Phase2ObservationEncoder(),
        RewardConfig(),
        max_steps=max_steps,
    )


def test_clone_helpers_copy_driver_and_ledger_without_mutating_original(small_encoder):
    env = ResourceCalendarEnv(small_encoder, top_k=8, w_lookahead=4.0)
    driver = _driver(env)
    driver.reset_episode()
    ledger = ReservationLedger()
    ledger.reserve(machine=1, future_lot=2, eta=1.5, created_at=0.0, expires_at=3.0)

    cloned_driver = clone_driver_for_rollout(driver)
    cloned_ledger = clone_ledger_for_rollout(ledger)
    cloned_ledger.release(1)
    cloned_driver.env.advance_time(1.0)

    assert ledger.is_reserved(1)
    assert driver.env.current_time == 0.0
    assert cloned_driver.env.current_time == 1.0


def test_advance_to_next_event_with_ledger_sees_reserved_lot_eta(small_encoder):
    env = ResourceCalendarEnv(small_encoder, top_k=8, w_lookahead=4.0)
    driver = _driver(env)
    driver.reset_episode()
    ledger = ReservationLedger()
    ledger.reserve(machine=1, future_lot=2, eta=1.5, created_at=0.0, expires_at=3.0)

    advanced_to = advance_to_next_event_with_ledger(driver, ledger)

    assert advanced_to == 1.5
    assert driver.env.current_time == 1.5


from vc_mcts_planner import (
    VCMCTSAction,
    VCMCTSConfig,
    VCMCTSEdgeStats,
    VCMCTSObjective,
    VCMCTSPlanner,
    compare_objectives,
    run_vc_mcts_reservation_episode,
)


def test_objective_comparison_is_qtime_then_o2_then_utilization():
    baseline = VCMCTSObjective(qtime_violation_count=0.0, priority_weighted_wait=10.0, avg_utilization=0.5)
    worse_qtime = VCMCTSObjective(qtime_violation_count=1.0, priority_weighted_wait=0.0, avg_utilization=1.0)
    worse_qtime_total = VCMCTSObjective(
        qtime_violation_count=0.0,
        qtime_violation_total=1.0,
        priority_weighted_wait=0.0,
        avg_utilization=1.0,
    )
    better_o2 = VCMCTSObjective(qtime_violation_count=0.0, priority_weighted_wait=8.0, avg_utilization=0.1)
    better_util = VCMCTSObjective(qtime_violation_count=0.0, priority_weighted_wait=10.0, avg_utilization=0.6)

    assert compare_objectives(better_o2, baseline) < 0
    assert compare_objectives(worse_qtime, baseline) > 0
    assert compare_objectives(worse_qtime_total, baseline) > 0
    assert compare_objectives(better_util, baseline) < 0


def test_edge_stats_tracks_visits_and_mean_objective():
    action = VCMCTSAction(kind="reserve", machine=1, future_lot=2, eta=1.5, prior=0.7)
    stats = VCMCTSEdgeStats(action=action)

    stats.record(VCMCTSObjective(
        qtime_violation_count=0.0,
        qtime_violation_total=2.0,
        priority_weighted_wait=10.0,
        avg_utilization=0.5,
    ))
    stats.record(VCMCTSObjective(
        qtime_violation_count=0.0,
        qtime_violation_total=0.0,
        priority_weighted_wait=6.0,
        avg_utilization=0.7,
    ))

    assert stats.visits == 2
    assert stats.mean_objective == VCMCTSObjective(
        qtime_violation_count=0.0,
        qtime_violation_total=1.0,
        priority_weighted_wait=8.0,
        avg_utilization=0.6,
    )
    assert stats.to_dict()["mean_qtime_total"] == 1.0


def test_planner_final_choice_prioritizes_objective_before_visits():
    planner = VCMCTSPlanner()
    reserve = VCMCTSEdgeStats(action=VCMCTSAction(kind="reserve"))
    dispatch = VCMCTSEdgeStats(action=VCMCTSAction(kind="dispatch"))
    reserve.record(VCMCTSObjective(
        qtime_violation_count=0.0,
        qtime_violation_total=0.0,
        priority_weighted_wait=100.0,
        avg_utilization=0.1,
    ))
    for _ in range(10):
        dispatch.record(VCMCTSObjective(
            qtime_violation_count=0.0,
            qtime_violation_total=1.0,
            priority_weighted_wait=1.0,
            avg_utilization=1.0,
        ))

    selected = planner._choose_final_action([dispatch, reserve])

    assert selected.action.kind == "reserve"


def test_planner_final_choice_demotes_noop_without_qtime_advantage():
    planner = VCMCTSPlanner()
    noop = VCMCTSEdgeStats(action=VCMCTSAction(kind="no_op"))
    dispatch = VCMCTSEdgeStats(action=VCMCTSAction(kind="dispatch"))
    noop.record(VCMCTSObjective(
        qtime_violation_count=0.0,
        qtime_violation_total=0.0,
        priority_weighted_wait=1.0,
        avg_utilization=0.1,
    ))
    dispatch.record(VCMCTSObjective(
        qtime_violation_count=0.0,
        qtime_violation_total=0.0,
        priority_weighted_wait=10.0,
        avg_utilization=0.5,
    ))

    selected = planner._choose_final_action([noop, dispatch])

    assert selected.action.kind == "dispatch"


def test_planner_final_choice_allows_noop_with_qtime_advantage():
    planner = VCMCTSPlanner()
    noop = VCMCTSEdgeStats(action=VCMCTSAction(kind="no_op"))
    dispatch = VCMCTSEdgeStats(action=VCMCTSAction(kind="dispatch"))
    noop.record(VCMCTSObjective(
        qtime_violation_count=0.0,
        qtime_violation_total=0.0,
        priority_weighted_wait=100.0,
        avg_utilization=0.1,
    ))
    dispatch.record(VCMCTSObjective(
        qtime_violation_count=0.0,
        qtime_violation_total=1.0,
        priority_weighted_wait=1.0,
        avg_utilization=0.5,
    ))

    selected = planner._choose_final_action([noop, dispatch])

    assert selected.action.kind == "no_op"


def test_planner_builds_dispatch_and_noop_actions(small_encoder):
    env = ResourceCalendarEnv(small_encoder, top_k=8, w_lookahead=4.0)
    driver = _driver(env)
    driver.reset_episode()
    ledger = ReservationLedger()
    machine = driver.select_next_machine(driver.get_dispatchable_machines())
    planner = VCMCTSPlanner(VCMCTSConfig(n_iter=1, top_k_dispatch=2, top_b_reserve=0))
    pool = driver.env.build_candidate_pool(machine)
    expected_dispatch_count = min(
        2,
        sum(
            1
            for action, is_valid in zip(pool.actions, pool.action_mask)
            if bool(is_valid)
            and not getattr(action, "is_padding", False)
            and not getattr(action, "is_wait", False)
            and int(action.ppid) != 0
        ),
    )

    actions = planner.build_root_actions(driver, ledger, machine)

    assert actions[0].kind == "no_op"
    assert [a.kind for a in actions].count("dispatch") == expected_dispatch_count
    assert all(a.machine == machine for a in actions if a.kind == "dispatch")


def test_planner_builds_single_delegate_dispatch_action(small_encoder):
    env = ResourceCalendarEnv(small_encoder, top_k=8, w_lookahead=4.0)
    driver = _driver(env)
    driver.reset_episode()
    ledger = ReservationLedger()
    machine = driver.select_next_machine(driver.get_dispatchable_machines())
    planner = VCMCTSPlanner(
        VCMCTSConfig(
            n_iter=1,
            top_k_dispatch=3,
            top_b_reserve=0,
            use_delegate_dispatch=True,
        ),
        dispatch_delegate=RuleDispatchDelegate(strategy="FIFO"),
    )

    actions = planner.build_root_actions(driver, ledger, machine)

    assert [action.kind for action in actions].count("delegate_dispatch") == 1
    assert [action.kind for action in actions].count("dispatch") == 0


def test_planner_builds_reserve_actions_from_rop(small_encoder):
    env = ResourceCalendarEnv(small_encoder, top_k=8, w_lookahead=4.0)
    driver = _driver(env)
    driver.reset_episode()
    ledger = ReservationLedger()
    planner = VCMCTSPlanner(VCMCTSConfig(n_iter=1, top_k_dispatch=1, top_b_reserve=2))

    actions = planner.build_root_actions(driver, ledger, machine=1)

    assert any(a.kind == "reserve" and a.future_lot == 2 for a in actions)


def test_evaluate_action_is_non_destructive(small_encoder):
    env = ResourceCalendarEnv(small_encoder, top_k=8, w_lookahead=4.0)
    driver = _driver(env)
    driver.reset_episode()
    ledger = ReservationLedger()
    machine = driver.select_next_machine(driver.get_dispatchable_machines())
    planner = VCMCTSPlanner(VCMCTSConfig(n_iter=1, top_k_dispatch=1, top_b_reserve=1))
    action = next(a for a in planner.build_root_actions(driver, ledger, machine) if a.kind == "dispatch")

    objective = planner.evaluate_action(driver, ledger, action)

    assert objective.priority_weighted_wait >= 0.0
    assert len(driver.env.completed_lots) == 0
    assert driver.env.lot_schedule.shape[0] == 0
    assert ledger.reserved_machines() == set()


def test_reserve_branch_records_reservation_only_in_rollout(small_encoder):
    env = ResourceCalendarEnv(small_encoder, top_k=8, w_lookahead=4.0)
    driver = _driver(env)
    driver.reset_episode()
    ledger = ReservationLedger()
    planner = VCMCTSPlanner(VCMCTSConfig(n_iter=1, top_k_dispatch=1, top_b_reserve=2))
    reserve = next(a for a in planner.build_root_actions(driver, ledger, machine=1) if a.kind == "reserve")

    objective = planner.evaluate_action(driver, ledger, reserve)

    assert objective.priority_weighted_wait >= 0.0
    assert not ledger.is_reserved(1)


def test_planner_selects_best_action_by_visits_with_injected_evaluator(small_encoder):
    env = ResourceCalendarEnv(small_encoder, top_k=8, w_lookahead=4.0)
    driver = _driver(env)
    driver.reset_episode()
    ledger = ReservationLedger()

    def evaluator(_driver, _ledger, action, _config):
        if action.kind == "reserve":
            return VCMCTSObjective(qtime_violation_count=0.0, priority_weighted_wait=1.0, avg_utilization=0.5)
        return VCMCTSObjective(qtime_violation_count=0.0, priority_weighted_wait=10.0, avg_utilization=0.5)

    planner = VCMCTSPlanner(
        VCMCTSConfig(n_iter=8, top_k_dispatch=1, top_b_reserve=2),
        rollout_evaluator=evaluator,
    )
    trace = planner.plan(driver, ledger, machine=1)

    assert trace.selected_action.kind == "reserve"
    assert sum(edge.visits for edge in trace.edge_stats) == 8
    assert any(edge.action.kind == "reserve" and edge.visits > 0 for edge in trace.edge_stats)


def test_planner_warms_up_every_root_edge_even_with_tiny_budget(small_encoder):
    env = ResourceCalendarEnv(small_encoder, top_k=8, w_lookahead=4.0)
    driver = _driver(env)
    driver.reset_episode()
    ledger = ReservationLedger()
    planner = VCMCTSPlanner(VCMCTSConfig(n_iter=1, top_k_dispatch=1, top_b_reserve=1))

    trace = planner.plan(driver, ledger, machine=1)

    assert {edge.action.kind for edge in trace.edge_stats} >= {
        "no_op",
        "dispatch",
        "reserve",
    }
    assert all(edge.visits >= 1 for edge in trace.edge_stats)


def test_vc_mcts_episode_completes_small_instance(small_encoder):
    env = ResourceCalendarEnv(small_encoder, top_k=8, w_lookahead=4.0)
    driver = _driver(env, max_steps=200)
    driver.reset_episode()
    planner = VCMCTSPlanner(VCMCTSConfig(n_iter=4, top_k_dispatch=2, top_b_reserve=1))

    summary = run_vc_mcts_reservation_episode(driver, planner=planner, max_steps=200)

    assert summary["completed_lots"] == 4
    assert summary["vc_mcts_decisions"] > 0
    assert summary["active_reservations"] == 0


def test_vc_mcts_episode_completes_with_rule_dispatch_delegate(small_encoder):
    env = ResourceCalendarEnv(small_encoder, top_k=8, w_lookahead=4.0)
    driver = _driver(env, max_steps=200)
    driver.reset_episode()
    delegate = RuleDispatchDelegate(strategy="FIFO")
    planner = VCMCTSPlanner(
        VCMCTSConfig(
            n_iter=4,
            top_k_dispatch=2,
            top_b_reserve=1,
            use_delegate_dispatch=True,
        ),
        dispatch_delegate=delegate,
    )

    summary = run_vc_mcts_reservation_episode(
        driver,
        planner=planner,
        max_steps=200,
        dispatch_delegate=delegate,
    )

    assert summary["completed_lots"] == 4
    assert summary["dispatch_delegate"] == "rule:FIFO"


def test_vc_mcts_episode_dispatches_ready_reserved_target(small_encoder):
    env = ResourceCalendarEnv(small_encoder, top_k=8, w_lookahead=4.0)
    driver = _driver(env, max_steps=200)
    driver.reset_episode()
    ledger = ReservationLedger()
    ledger.reserve(machine=1, future_lot=2, eta=1.5, created_at=0.0, expires_at=4.0)

    def evaluator(_driver, _ledger, action, _config):
        if action.kind == "no_op":
            return VCMCTSObjective(qtime_violation_count=0.0, priority_weighted_wait=0.0, avg_utilization=0.0)
        return VCMCTSObjective(qtime_violation_count=0.0, priority_weighted_wait=10.0, avg_utilization=0.0)

    planner = VCMCTSPlanner(
        VCMCTSConfig(n_iter=2, top_k_dispatch=1, top_b_reserve=0),
        rollout_evaluator=evaluator,
    )

    summary = run_vc_mcts_reservation_episode(
        driver,
        planner=planner,
        ledger=ledger,
        max_steps=200,
    )

    lot2_rows = env.lot_schedule[env.lot_schedule[:, 0] == 2]
    assert lot2_rows.shape[0] == 1
    assert int(lot2_rows[0, 1]) == 1
    assert float(lot2_rows[0, 3]) == 1.5
    assert not ledger.is_reserved(1)


from vc_mcts_probe import (
    _seed_output_path,
    main as run_vc_mcts_probe,
    run_seed as run_vc_mcts_seed,
)


def test_vc_mcts_probe_run_seed_uses_sas_delegate_by_default():
    row = run_vc_mcts_seed(
        instance="small",
        seed=0,
        strategy="FIFO",
        w_lookahead=4.0,
        top_b=1,
        top_k_dispatch=2,
        n_iter=2,
        max_steps=200,
        skip_oracle=True,
    )

    assert set(row) >= {"seed", "baseline", "oracle", "vc_mcts", "delta"}
    assert row["oracle"] is None
    assert row["baseline"]["completed_lots"] == 4.0
    assert row["vc_mcts"]["completed_lots"] == 4.0
    assert row["vc_mcts"]["dispatch_delegate"].startswith("sas:")


def test_vc_mcts_episode_writes_trace_and_stops_at_max_decisions(small_encoder):
    env = ResourceCalendarEnv(small_encoder, top_k=8, w_lookahead=4.0)
    driver = _driver(env, max_steps=200)
    driver.reset_episode()
    traces = []
    planner = VCMCTSPlanner(VCMCTSConfig(n_iter=2, top_k_dispatch=1, top_b_reserve=1))

    summary = run_vc_mcts_reservation_episode(
        driver,
        planner=planner,
        max_steps=200,
        max_decisions=2,
        trace_writer=traces.append,
    )

    assert summary["termination_reason"] == "max_decisions_exceeded"
    assert summary["vc_mcts_decisions"] == 2
    assert len(traces) == 2
    assert set(traces[0]) >= {
        "time",
        "machine",
        "selected_action",
        "edges",
        "diagnostics",
    }
    assert traces[0]["diagnostics"]["edge_count"] >= 1


def test_vc_mcts_probe_can_skip_oracle_and_write_trace(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    summary_path = tmp_path / "trace_summary.json"

    row = run_vc_mcts_seed(
        instance="small",
        seed=0,
        strategy="FIFO",
        w_lookahead=4.0,
        top_b=1,
        top_k_dispatch=1,
        n_iter=1,
        max_steps=200,
        skip_oracle=True,
        rollout_max_steps=20,
        max_decisions=2,
        trace_out=str(trace_path),
        trace_summary_out=str(summary_path),
        stop_after_reserve_available=1,
    )

    assert row["oracle"] is None
    assert row["delta"]["oracle_o2"] is None
    assert row["vc_mcts"]["termination_reason"] == "reserve_available_limit_exceeded"
    lines = trace_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["decisions"] == 1
    assert summary["reserve_available_decisions"] == 1


def test_vc_mcts_probe_rejects_unknown_delegate_modes():
    # "sas" 与 "rule" (机制 2 消融用, 无需 checkpoint) 合法; 其他一律拒绝
    with pytest.raises(ValueError):
        run_vc_mcts_seed(
            instance="small",
            seed=0,
            strategy="FIFO",
            w_lookahead=4.0,
            top_b=1,
            top_k_dispatch=2,
            n_iter=2,
            max_steps=200,
            skip_oracle=True,
            rollout_max_steps=20,
            dispatch_delegate="topk",
        )


def test_vc_mcts_probe_seed_output_path_inserts_seed_before_suffix():
    assert Path(_seed_output_path("results/run_trace.jsonl", seed=3)).as_posix() == (
        "results/run_seed3_trace.jsonl"
    )
    assert Path(_seed_output_path("results/run_summary.json", seed=3)).as_posix() == (
        "results/run_seed3_summary.json"
    )


def test_vc_mcts_probe_workers_write_per_seed_trace_and_summary(tmp_path):
    trace_path = tmp_path / "worker_trace.jsonl"
    summary_path = tmp_path / "worker_summary.json"

    rows = run_vc_mcts_probe(
        instance="small",
        seeds=2,
        workers=2,
        strategy="FIFO",
        w_lookahead=4.0,
        top_b=1,
        top_k_dispatch=1,
        n_iter=1,
        max_steps=200,
        skip_oracle=True,
        rollout_max_steps=20,
        max_decisions=2,
        stop_after_reserve_available=1,
        trace_out=str(trace_path),
        trace_summary_out=str(summary_path),
    )

    assert [row["seed"] for row in rows] == [0, 1]
    for seed in (0, 1):
        seed_trace = tmp_path / f"worker_seed{seed}_trace.jsonl"
        seed_summary = tmp_path / f"worker_seed{seed}_summary.json"
        assert seed_trace.exists()
        assert seed_summary.exists()
        assert json.loads(seed_summary.read_text(encoding="utf-8"))["decisions"] == 1
