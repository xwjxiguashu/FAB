"""Tests for 机制 2 (报告8 §7.12): priority-capability bipartite matching waterline."""
import pytest

from reservation_ledger import ReservationLedger
from rl_environment import ResourceCalendarEnv
from vc_mcts_planner import VCMCTSAction
from priority_capability_matching import (
    CapabilityGraph,
    CapabilityMatchEdge,
    build_priority_capability_graph,
    rho_pc_for_action,
    rho_pc_state,
    solve_weighted_capability_matching,
)


def test_weighted_matching_prefers_larger_total_priority():
    graph = CapabilityGraph(
        machines=(1, 2),
        future_lots=(10, 11),
        lot_weights={10: 10.0, 11: 4.0},
        edges=(
            CapabilityMatchEdge(machine=1, future_lot=10),
            CapabilityMatchEdge(machine=1, future_lot=11),
            CapabilityMatchEdge(machine=2, future_lot=11),
        ),
    )

    result = solve_weighted_capability_matching(graph)

    assert result.total_weight == pytest.approx(14.0)
    assert result.normalized_waterline == pytest.approx(1.0)
    assert result.pairs == frozenset({(1, 10), (2, 11)})


def test_weighted_matching_reports_uncovered_priority_mass():
    graph = CapabilityGraph(
        machines=(1,),
        future_lots=(10, 11),
        lot_weights={10: 10.0, 11: 5.0},
        edges=(
            CapabilityMatchEdge(machine=1, future_lot=11),
        ),
    )

    result = solve_weighted_capability_matching(graph)

    assert result.total_weight == pytest.approx(5.0)
    assert result.normalized_waterline == pytest.approx(5.0 / 15.0)
    assert result.uncovered_lots == frozenset({10})


def test_empty_future_lots_has_zero_waterline():
    graph = CapabilityGraph(
        machines=(1, 2),
        future_lots=(),
        lot_weights={},
        edges=(),
    )

    result = solve_weighted_capability_matching(graph)

    assert result.total_weight == 0.0
    assert result.normalized_waterline == 0.0
    assert result.pairs == frozenset()


# ---------------------------------------------------------------------------
# Graph construction from environment state
# ---------------------------------------------------------------------------

def test_build_graph_uses_only_visible_future_lots(small_encoder):
    env = ResourceCalendarEnv(small_encoder, top_k=8, w_lookahead=4.0)
    env.reset()
    graph = build_priority_capability_graph(env, ReservationLedger())

    upcoming = set(int(lot) for lot in env.upcoming_lots())

    assert set(graph.future_lots).issubset(upcoming)
    assert set(graph.machines).issubset(set(range(1, env.encoder.num_machines + 1)))


def test_w_zero_degenerates_to_zero_waterline(small_encoder):
    env = ResourceCalendarEnv(small_encoder, top_k=8, w_lookahead=0.0)
    env.reset()

    result = rho_pc_state(env, ReservationLedger())

    assert result.normalized_waterline == 0.0
    assert result.pairs == frozenset()


def test_reserved_machine_is_removed_from_free_capacity(small_encoder):
    env = ResourceCalendarEnv(small_encoder, top_k=8, w_lookahead=4.0)
    env.reset()
    upcoming = [int(lot) for lot in env.upcoming_lots()]
    assert upcoming
    machine = int(env.encoder.feasible_machines[upcoming[0]][0])

    ledger = ReservationLedger()
    ledger.reserve(
        machine=machine,
        future_lot=upcoming[0],
        eta=float(env.encoder.arrival_times[upcoming[0]]),
        created_at=float(env.current_time),
        expires_at=float(env.current_time) + 4.0,
    )

    graph = build_priority_capability_graph(env, ledger)

    assert machine not in graph.machines


# ---------------------------------------------------------------------------
# Action-level waterline deltas (Δρ_pc, 报告8 §7.12.2 性质 1)
# ---------------------------------------------------------------------------

def test_dispatch_does_not_increase_rho_pc(small_encoder):
    env = ResourceCalendarEnv(small_encoder, top_k=8, w_lookahead=4.0)
    env.reset()
    ledger = ReservationLedger()
    base = rho_pc_state(env, ledger)
    action = VCMCTSAction(kind="dispatch", machine=1, action_index=0, lot=1, ppid=1)

    result = rho_pc_for_action(env, ledger, action)

    assert result.before == base.normalized_waterline
    assert result.after <= result.before
    assert result.delta <= 0.0


def test_no_op_preserves_rho_pc(small_encoder):
    env = ResourceCalendarEnv(small_encoder, top_k=8, w_lookahead=4.0)
    env.reset()
    ledger = ReservationLedger()

    result = rho_pc_for_action(env, ledger, VCMCTSAction(kind="no_op", machine=1))

    assert result.after == result.before
    assert result.delta == 0.0


def test_reserve_records_forced_pair_when_feasible(small_encoder):
    env = ResourceCalendarEnv(small_encoder, top_k=8, w_lookahead=4.0)
    env.reset()
    ledger = ReservationLedger()
    upcoming = [int(lot) for lot in env.upcoming_lots()]
    assert upcoming
    lot = upcoming[0]
    machine = int(env.encoder.feasible_machines[lot][0])

    action = VCMCTSAction(
        kind="reserve",
        machine=machine,
        future_lot=lot,
        eta=float(env.encoder.arrival_times[lot]),
    )
    result = rho_pc_for_action(env, ledger, action)

    assert result.after >= result.before
    assert (machine, lot) in result.forced_pairs
