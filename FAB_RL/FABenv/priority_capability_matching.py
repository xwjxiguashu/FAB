"""Priority-capability matching waterline for VC-MCTS mechanism 2 (报告8 §7.12).

ρ_pc(s) = max weighted bipartite matching between idle/unblocked machines and
window-visible high-priority future lots, normalized by the total priority
mass — the "可兑现对冲水位". 1.0 means every visible high-priority lot has a
capability-compatible, Q-time-redeemable machine held for it.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class CapabilityMatchEdge:
    machine: int
    future_lot: int


@dataclass(frozen=True)
class CapabilityGraph:
    machines: tuple[int, ...]
    future_lots: tuple[int, ...]
    lot_weights: dict[int, float]
    edges: tuple[CapabilityMatchEdge, ...]


@dataclass(frozen=True)
class CapabilityMatchResult:
    total_weight: float
    normalized_waterline: float
    pairs: frozenset[tuple[int, int]]
    uncovered_lots: frozenset[int]


def solve_weighted_capability_matching(graph: CapabilityGraph) -> CapabilityMatchResult:
    machines = tuple(int(m) for m in graph.machines)
    future_lots = tuple(int(h) for h in graph.future_lots)
    total_possible = float(sum(float(graph.lot_weights.get(h, 0.0)) for h in future_lots))
    if not machines or not future_lots or total_possible <= 0.0:
        return CapabilityMatchResult(
            total_weight=0.0,
            normalized_waterline=0.0,
            pairs=frozenset(),
            uncovered_lots=frozenset(future_lots),
        )

    edge_set = {(int(edge.machine), int(edge.future_lot)) for edge in graph.edges}
    # 方阵填充 + 大成本占位让 linear_sum_assignment 可以"不匹配"某行/列
    n = max(len(machines), len(future_lots))
    large_cost = 1e9
    cost = np.full((n, n), large_cost, dtype=float)

    for i, machine in enumerate(machines):
        for j, lot in enumerate(future_lots):
            if (machine, lot) in edge_set:
                cost[i, j] = -float(graph.lot_weights.get(lot, 0.0))

    rows, cols = linear_sum_assignment(cost)
    pairs = []
    total = 0.0
    for row, col in zip(rows, cols):
        if row >= len(machines) or col >= len(future_lots):
            continue
        if cost[row, col] >= large_cost:
            continue
        machine = machines[row]
        lot = future_lots[col]
        pairs.append((machine, lot))
        total += float(graph.lot_weights.get(lot, 0.0))

    covered = {lot for _machine, lot in pairs}
    uncovered = frozenset(lot for lot in future_lots if lot not in covered)
    return CapabilityMatchResult(
        total_weight=float(total),
        normalized_waterline=float(total / total_possible),
        pairs=frozenset(pairs),
        uncovered_lots=uncovered,
    )
