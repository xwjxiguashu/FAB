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


def _compatible_ppids(encoder, lot: int, machine: int) -> tuple[int, ...]:
    ppids = []
    for ppid in encoder.feasible_ppids.get((int(lot), int(machine)), []):
        try:
            steps = encoder.get_process_steps(int(lot), int(machine), int(ppid))
        except (KeyError, ValueError):
            continue
        if steps:
            ppids.append(int(ppid))
    return tuple(ppids)


def _priority_threshold(env, priority_threshold):
    """p_hi: 高优先级类阈值; 未显式给出时取窗内可见 lot 优先级的中位数。"""
    if priority_threshold is not None:
        return float(priority_threshold)
    priorities = getattr(env.encoder, "priorities", {})
    visible = [int(lot) for lot in env.upcoming_lots()]
    if not visible:
        return float("inf")
    values = sorted(float(priorities.get(lot, 0.0)) for lot in visible)
    return values[len(values) // 2]


def _edge_is_time_redeemable(env, lot: int, machine: int) -> bool:
    """(m,h) 时间可兑现 ⇔ 某 ppid 的 dry-run 可行且结构 Q-time 风险为 0 (§7.12.2)。"""
    for ppid in _compatible_ppids(env.encoder, lot, machine):
        dry_run, _reason = env._dry_run_candidate(int(lot), int(machine), int(ppid))
        if dry_run is None:
            continue
        if float(dry_run.get("qtime_risk", 0.0)) <= 0.0:
            return True
    return False


def build_priority_capability_graph(
    env,
    ledger,
    priority_threshold=None,
    blocked_machines=(),
) -> CapabilityGraph:
    reserved_machines = set(int(m) for m in ledger.reserved_machines())
    blocked = set(int(m) for m in blocked_machines)
    machines = tuple(
        int(machine)
        for machine in range(1, int(env.encoder.num_machines) + 1)
        if int(machine) not in reserved_machines and int(machine) not in blocked
    )
    threshold = _priority_threshold(env, priority_threshold)
    priorities = getattr(env.encoder, "priorities", {})
    future_lots = tuple(
        int(lot)
        for lot in env.upcoming_lots()
        if float(priorities.get(int(lot), 0.0)) >= threshold
    )
    weights = {lot: float(priorities.get(lot, 0.0)) for lot in future_lots}

    edges = []
    for machine in machines:
        for lot in future_lots:
            if int(machine) not in {int(m) for m in env.encoder.feasible_machines.get(lot, [])}:
                continue
            if _edge_is_time_redeemable(env, lot, machine):
                edges.append(CapabilityMatchEdge(machine=machine, future_lot=lot))

    return CapabilityGraph(
        machines=machines,
        future_lots=future_lots,
        lot_weights=weights,
        edges=tuple(edges),
    )


def rho_pc_state(env, ledger, priority_threshold=None, blocked_machines=()):
    graph = build_priority_capability_graph(
        env,
        ledger,
        priority_threshold=priority_threshold,
        blocked_machines=blocked_machines,
    )
    return solve_weighted_capability_matching(graph)


@dataclass(frozen=True)
class RhoPcActionResult:
    before: float
    after: float
    delta: float
    before_pairs: frozenset[tuple[int, int]]
    after_pairs: frozenset[tuple[int, int]]
    forced_pairs: frozenset[tuple[int, int]]


def _solve_with_forced_pair(env, ledger, action, priority_threshold):
    """reserve(m,h): 匹配被钉死包含 (m,h)，其余节点上再求最大匹配 (s ⊕ a 语义)。"""
    if action.kind != "reserve" or action.machine is None or action.future_lot is None:
        return rho_pc_state(env, ledger, priority_threshold=priority_threshold), frozenset()

    forced = (int(action.machine), int(action.future_lot))
    graph = build_priority_capability_graph(env, ledger, priority_threshold=priority_threshold)
    edge_set = {(edge.machine, edge.future_lot) for edge in graph.edges}
    if forced not in edge_set:
        return solve_weighted_capability_matching(graph), frozenset()

    remaining_machines = tuple(m for m in graph.machines if m != forced[0])
    remaining_lots = tuple(h for h in graph.future_lots if h != forced[1])
    remaining_edges = tuple(
        edge
        for edge in graph.edges
        if edge.machine != forced[0] and edge.future_lot != forced[1]
    )
    remaining_graph = CapabilityGraph(
        machines=remaining_machines,
        future_lots=remaining_lots,
        lot_weights=graph.lot_weights,
        edges=remaining_edges,
    )
    rest = solve_weighted_capability_matching(remaining_graph)
    forced_weight = float(graph.lot_weights.get(forced[1], 0.0))
    total_possible = float(sum(graph.lot_weights.get(h, 0.0) for h in graph.future_lots))
    total = forced_weight + rest.total_weight
    normalized = 0.0 if total_possible <= 0.0 else total / total_possible
    pairs = frozenset(set(rest.pairs) | {forced})
    covered = {lot for _machine, lot in pairs}
    return CapabilityMatchResult(
        total_weight=float(total),
        normalized_waterline=float(normalized),
        pairs=pairs,
        uncovered_lots=frozenset(h for h in graph.future_lots if h not in covered),
    ), frozenset({forced})


def rho_pc_for_action(env, ledger, action, priority_threshold=None) -> RhoPcActionResult:
    """边级对冲水位: ρ̂_pc(s,a) = ρ̃_pc(s ⊕ a), Δρ_pc = ρ̃_pc(s ⊕ a) − ρ̃_pc(s)。

    dispatch(m): m 在占用区间内移出 M_t (blocked) → Δ ≤ 0;
    reserve(m,h): 匹配钉死含 (m,h) → Δ ≥ 0; no_op: 不变。
    """
    before = rho_pc_state(env, ledger, priority_threshold=priority_threshold)
    if action.kind in ("dispatch", "delegate_dispatch") and action.machine is not None:
        after = rho_pc_state(
            env,
            ledger,
            priority_threshold=priority_threshold,
            blocked_machines=(int(action.machine),),
        )
        forced = frozenset()
    elif action.kind == "reserve":
        after, forced = _solve_with_forced_pair(env, ledger, action, priority_threshold)
    else:
        after = before
        forced = frozenset()

    return RhoPcActionResult(
        before=float(before.normalized_waterline),
        after=float(after.normalized_waterline),
        delta=float(after.normalized_waterline - before.normalized_waterline),
        before_pairs=before.pairs,
        after_pairs=after.pairs,
        forced_pairs=forced,
    )
