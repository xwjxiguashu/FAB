"""Tests for 机制 2 (报告8 §7.12): priority-capability bipartite matching waterline."""
import pytest

from priority_capability_matching import (
    CapabilityGraph,
    CapabilityMatchEdge,
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
