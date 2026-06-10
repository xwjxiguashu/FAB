"""Tests for VC-MCTS 机制 2 (ρ_pc priority-capability robustness) and
机制 3 (CRN noisy multi-rollout)."""
import numpy as np
import pytest

from phase2_sas_driver import Phase2EpisodeDriver
from phase2_sas_observation import Phase2ObservationEncoder
from reservation_ledger import ReservationLedger
from rl_environment import ResourceCalendarEnv, RewardConfig
from vc_mcts_planner import (
    VCMCTSConfig,
    VCMCTSObjective,
    VCMCTSPlanner,
    mean_objective,
    run_vc_mcts_reservation_episode,
)


def _driver(env, max_steps=200):
    return Phase2EpisodeDriver(
        env,
        Phase2ObservationEncoder(),
        RewardConfig(),
        max_steps=max_steps,
    )


# ---------------------------------------------------------------------------
# 机制 3: CRN keyed noise + multi-rollout averaging
# ---------------------------------------------------------------------------

def test_crn_keyed_rng_is_order_independent_and_seed_sensitive(small_encoder):
    """CRN 命门: 噪声按 (crn_seed, lot, ppid) 键控, 与 commit 顺序无关。"""
    env = ResourceCalendarEnv(small_encoder, top_k=8)
    env.enable_process_noise(crn_seed=7)

    # 同 (seed, lot, ppid) → 完全相同的噪声序列 (顺序无关性的根因)
    draw_a = env._crn_noise_rng(lot=2, ppid=1).normal(0.0, 1.0, size=5)
    draw_b = env._crn_noise_rng(lot=2, ppid=1).normal(0.0, 1.0, size=5)
    assert np.allclose(draw_a, draw_b)

    # 不同 lot / 不同 seed → 不同噪声
    draw_other_lot = env._crn_noise_rng(lot=3, ppid=1).normal(0.0, 1.0, size=5)
    assert not np.allclose(draw_a, draw_other_lot)

    env.enable_process_noise(crn_seed=8)
    draw_other_seed = env._crn_noise_rng(lot=2, ppid=1).normal(0.0, 1.0, size=5)
    assert not np.allclose(draw_a, draw_other_seed)


def test_enable_process_noise_turns_on_noise_and_crn_mode(small_encoder):
    env = ResourceCalendarEnv(small_encoder, top_k=8)
    assert env.process_noise_enabled is False
    assert env._crn_seed is None

    env.enable_process_noise(crn_seed=3)

    assert env.process_noise_enabled is True
    assert env._crn_seed == 3


def test_crn_commit_same_lot_same_noise_across_clones(small_encoder):
    """同一 lot 在两个独立 clone 上用同一 crn_seed commit → 完成时间一致 (CRN)。"""
    from reservation_simulator import clone_driver_for_rollout

    env = ResourceCalendarEnv(small_encoder, top_k=8, w_lookahead=4.0)
    driver = _driver(env)
    driver.reset_episode()
    machine = driver.select_next_machine(driver.get_dispatchable_machines())
    pool = env.build_candidate_pool(machine)
    action_index = next(
        i
        for i, (a, ok) in enumerate(zip(pool.actions, pool.action_mask))
        if bool(ok) and not getattr(a, "is_wait", False)
        and not getattr(a, "is_padding", False) and int(a.ppid) != 0
    )

    ends = []
    for _ in range(2):
        clone = clone_driver_for_rollout(driver)
        clone.env.enable_process_noise(crn_seed=11)
        result = clone.step_with_action(machine, action_index, pool=clone.env.build_candidate_pool(machine))
        assert result.committed
        ends.append(float(clone.env.lot_schedule[-1, 4]))

    assert ends[0] == ends[1]


def test_mean_objective_averages_each_field():
    objs = [
        VCMCTSObjective(
            qtime_violation_count=2.0,
            qtime_violation_total=4.0,
            priority_weighted_wait=10.0,
            avg_utilization=0.4,
        ),
        VCMCTSObjective(
            qtime_violation_count=0.0,
            qtime_violation_total=0.0,
            priority_weighted_wait=6.0,
            avg_utilization=0.8,
        ),
    ]
    mean = mean_objective(objs)
    assert mean.qtime_violation_count == 1.0
    assert mean.qtime_violation_total == 2.0
    assert mean.priority_weighted_wait == 8.0
    assert mean.avg_utilization == pytest.approx(0.6)


def test_mean_objective_handles_empty_and_none():
    assert mean_objective([]) is None
    assert mean_objective([None, None]) is None


def test_crn_multi_rollout_equals_mean_of_single_draws(small_encoder):
    """evaluate_action(crn) 应等于对 n_mc 个种子各跑一次的均值 (机制 3 定义)。"""
    env = ResourceCalendarEnv(small_encoder, top_k=8, w_lookahead=4.0)
    driver = _driver(env)
    driver.reset_episode()
    ledger = ReservationLedger()
    machine = driver.select_next_machine(driver.get_dispatchable_machines())

    config = VCMCTSConfig(
        n_iter=1,
        top_k_dispatch=1,
        top_b_reserve=0,
        crn_noise=True,
        n_mc=4,
        crn_seed_base=100,
        rollout_max_steps=40,
    )
    planner = VCMCTSPlanner(config)
    action = next(
        a for a in planner.build_root_actions(driver, ledger, machine)
        if a.kind == "dispatch"
    )

    combined = planner.evaluate_action(driver, ledger, action)
    singles = [
        planner._evaluate_action_once(driver, ledger, action, noise_seed=100 + k)
        for k in range(4)
    ]
    expected = mean_objective(singles)

    assert combined.priority_weighted_wait == expected.priority_weighted_wait
    assert combined.qtime_violation_count == expected.qtime_violation_count
    assert combined.avg_utilization == expected.avg_utilization


def test_crn_multi_rollout_is_deterministic_across_calls(small_encoder):
    """同一 planner 重复评估同一动作 → 完全一致 (可复算性)。"""
    env = ResourceCalendarEnv(small_encoder, top_k=8, w_lookahead=4.0)
    driver = _driver(env)
    driver.reset_episode()
    ledger = ReservationLedger()
    machine = driver.select_next_machine(driver.get_dispatchable_machines())
    planner = VCMCTSPlanner(
        VCMCTSConfig(
            n_iter=1, top_k_dispatch=1, top_b_reserve=0,
            crn_noise=True, n_mc=3, crn_seed_base=7, rollout_max_steps=40,
        )
    )
    action = next(
        a for a in planner.build_root_actions(driver, ledger, machine)
        if a.kind == "dispatch"
    )

    first = planner.evaluate_action(driver, ledger, action)
    second = planner.evaluate_action(driver, ledger, action)

    assert first == second


# ---------------------------------------------------------------------------
# 机制 2: ρ_pc priority-capability robustness
# ---------------------------------------------------------------------------

def test_planner_populates_matching_rho_pc_fields_when_enabled(small_encoder):
    """开启机制 2 时每条边记录二部匹配水位 before/after/delta (报告8 §7.12.4)。"""
    env = ResourceCalendarEnv(small_encoder, top_k=8, w_lookahead=4.0)
    driver = _driver(env)
    driver.reset_episode()
    ledger = ReservationLedger()

    off = VCMCTSPlanner(VCMCTSConfig(n_iter=1, top_k_dispatch=1, top_b_reserve=1))
    trace_off = off.plan(driver, ledger, machine=1)
    assert all(
        edge.rho_pc_before == 0.0
        and edge.rho_pc_after == 0.0
        and edge.delta_rho_pc == 0.0
        for edge in trace_off.edge_stats
    )

    planner = VCMCTSPlanner(
        VCMCTSConfig(n_iter=1, top_k_dispatch=1, top_b_reserve=1, use_rho_pc=True)
    )
    trace = planner.plan(driver, ledger, machine=1)

    edge_dicts = [edge.to_dict() for edge in trace.edge_stats]
    assert all("rho_pc_before" in item for item in edge_dicts)
    assert all("rho_pc_after" in item for item in edge_dicts)
    assert all("delta_rho_pc" in item for item in edge_dicts)
    assert any(item["delta_rho_pc"] >= 0.0 for item in edge_dicts if item["kind"] == "reserve")


def test_rho_pc_biases_selection_toward_reserve_with_injected_evaluator(small_encoder):
    """ρ_pc 偏置应能在 reserve 与 dispatch 目标接近时把搜索引向 reserve。"""
    env = ResourceCalendarEnv(small_encoder, top_k=8, w_lookahead=4.0)
    driver = _driver(env)
    driver.reset_episode()
    ledger = ReservationLedger()

    # reserve 与 dispatch 的 rollout 目标完全相同 → 不开 ρ_pc 时无偏好
    def evaluator(_driver, _ledger, _action, _config):
        return VCMCTSObjective(
            qtime_violation_count=0.0,
            qtime_violation_total=0.0,
            priority_weighted_wait=5.0,
            avg_utilization=0.5,
        )

    planner = VCMCTSPlanner(
        VCMCTSConfig(
            n_iter=12, top_k_dispatch=1, top_b_reserve=2,
            use_rho_pc=True, rho_pc_weight=1000.0,
        ),
        rollout_evaluator=evaluator,
    )
    trace = planner.plan(driver, ledger, machine=1)

    reserve_edges = [e for e in trace.edge_stats if e.action.kind == "reserve"]
    dispatch_edges = [e for e in trace.edge_stats if e.action.kind == "dispatch"]
    assert reserve_edges and dispatch_edges
    # 大 ρ_pc 权重 → reserve 边获得显著更多 UCT 访问
    assert max(e.visits for e in reserve_edges) > max(e.visits for e in dispatch_edges)


# ---------------------------------------------------------------------------
# 端到端: 两机制同时开启仍能跑完
# ---------------------------------------------------------------------------

def test_episode_completes_with_both_mechanisms_enabled(small_encoder):
    env = ResourceCalendarEnv(
        small_encoder, top_k=8, w_lookahead=4.0,
    )
    driver = _driver(env, max_steps=200)
    driver.reset_episode()
    planner = VCMCTSPlanner(
        VCMCTSConfig(
            n_iter=4,
            top_k_dispatch=2,
            top_b_reserve=1,
            crn_noise=True,
            n_mc=3,
            crn_seed_base=42,
            use_rho_pc=True,
            rho_pc_weight=2.0,
            rollout_max_steps=80,
        )
    )

    summary = run_vc_mcts_reservation_episode(driver, planner=planner, max_steps=200)

    assert summary["completed_lots"] == 4
    assert summary["vc_mcts_decisions"] > 0
    assert summary["active_reservations"] == 0
