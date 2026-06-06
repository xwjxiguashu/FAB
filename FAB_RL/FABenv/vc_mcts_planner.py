"""Online VC-MCTS reservation planner.

This first slice is a root-level MCTS planner: build root actions, evaluate
branches with reservation-aware rollouts, and choose by visit count with a
lexicographic objective tie-break.
"""
from dataclasses import dataclass, field
import json
import math
import sys

from reservation_ledger import ReservationLedger
from reservation_rop import detect_reservation_opportunities
from reservation_simulator import (
    advance_to_next_event_with_ledger,
    clone_driver_for_rollout,
    clone_ledger_for_rollout,
    finalize_reservation_ledger,
    run_rule_episode_with_reservations,
    schedule_metrics_with_priority_wait,
)


@dataclass(frozen=True)
class VCMCTSConfig:
    n_iter: int = 24
    top_k_dispatch: int = 3
    top_b_reserve: int = 2
    exploration_c: float = 1.5
    qtime_penalty: float = 10000.0
    qtime_total_penalty: float = 1000.0
    util_weight: float = 1.0
    min_priority_gap: float = 0.0
    reservation_ttl: float = 1.0
    rollout_strategy: str = "FIFO"
    rollout_max_steps: int | None = None


@dataclass(frozen=True)
class VCMCTSObjective:
    qtime_violation_count: float
    priority_weighted_wait: float
    avg_utilization: float
    qtime_violation_total: float = 0.0


@dataclass(frozen=True)
class VCMCTSAction:
    kind: str
    machine: int | None = None
    action_index: int | None = None
    lot: int | None = None
    ppid: int | None = None
    future_lot: int | None = None
    eta: float | None = None
    prior: float = 1.0

    def to_dict(self):
        return {
            "kind": self.kind,
            "machine": self.machine,
            "action_index": self.action_index,
            "lot": self.lot,
            "ppid": self.ppid,
            "future_lot": self.future_lot,
            "eta": self.eta,
            "prior": float(self.prior),
        }


@dataclass
class VCMCTSEdgeStats:
    action: VCMCTSAction
    visits: int = 0
    total_qtime: float = 0.0
    total_qtime_severity: float = 0.0
    total_o2: float = 0.0
    total_util: float = 0.0

    def record(self, objective):
        self.visits += 1
        self.total_qtime += float(objective.qtime_violation_count)
        self.total_qtime_severity += float(objective.qtime_violation_total)
        self.total_o2 += float(objective.priority_weighted_wait)
        self.total_util += float(objective.avg_utilization)

    @property
    def mean_objective(self):
        if self.visits <= 0:
            return None
        return VCMCTSObjective(
            qtime_violation_count=self.total_qtime / self.visits,
            qtime_violation_total=self.total_qtime_severity / self.visits,
            priority_weighted_wait=self.total_o2 / self.visits,
            avg_utilization=self.total_util / self.visits,
        )

    def to_dict(self):
        mean = self.mean_objective
        return {
            "action": self.action.to_dict(),
            "kind": self.action.kind,
            "visits": int(self.visits),
            "mean_qtime": None if mean is None else float(mean.qtime_violation_count),
            "mean_qtime_total": None if mean is None else float(mean.qtime_violation_total),
            "mean_o2": None if mean is None else float(mean.priority_weighted_wait),
            "mean_util": None if mean is None else float(mean.avg_utilization),
        }


@dataclass
class VCMCTSDecisionTrace:
    selected_action: VCMCTSAction
    edge_stats: list[VCMCTSEdgeStats] = field(default_factory=list)
    current_time: float = 0.0
    machine: int | None = None

    def to_dict(self):
        edges = [edge.to_dict() for edge in self.edge_stats]
        reserve_edges = [edge for edge in edges if edge["kind"] == "reserve"]
        dispatch_edges = [edge for edge in edges if edge["kind"] == "dispatch"]
        noop_edges = [edge for edge in edges if edge["kind"] == "no_op"]

        def best_metric(items, metric):
            values = [item[metric] for item in items if item[metric] is not None]
            return None if not values else min(values)

        diagnostics = {
            "edge_count": len(edges),
            "reserve_edge_count": len(reserve_edges),
            "reserve_total_visits": int(sum(item["visits"] for item in reserve_edges)),
            "selected_kind": self.selected_action.kind,
            "reserve_was_available": bool(reserve_edges),
            "reserve_was_selected": self.selected_action.kind == "reserve",
            "best_reserve_o2": best_metric(reserve_edges, "mean_o2"),
            "best_dispatch_o2": best_metric(dispatch_edges, "mean_o2"),
            "best_noop_o2": best_metric(noop_edges, "mean_o2"),
            "best_reserve_qtime_total": best_metric(reserve_edges, "mean_qtime_total"),
            "best_dispatch_qtime_total": best_metric(dispatch_edges, "mean_qtime_total"),
            "best_noop_qtime_total": best_metric(noop_edges, "mean_qtime_total"),
        }
        return {
            "time": float(self.current_time),
            "machine": self.machine,
            "selected_action": self.selected_action.to_dict(),
            "edges": edges,
            "diagnostics": diagnostics,
        }


def compare_objectives(left, right):
    left_key = (
        float(left.qtime_violation_count),
        float(left.qtime_violation_total),
        float(left.priority_weighted_wait),
        -float(left.avg_utilization),
    )
    right_key = (
        float(right.qtime_violation_count),
        float(right.qtime_violation_total),
        float(right.priority_weighted_wait),
        -float(right.avg_utilization),
    )
    return (left_key > right_key) - (left_key < right_key)


def objective_to_score(objective, config):
    return (
        -float(config.qtime_penalty) * float(objective.qtime_violation_count)
        -float(config.qtime_total_penalty) * float(objective.qtime_violation_total)
        -float(objective.priority_weighted_wait)
        + float(config.util_weight) * float(objective.avg_utilization)
    )


class VCMCTSPlanner:
    def __init__(self, config=None, rollout_evaluator=None):
        self.config = config if config is not None else VCMCTSConfig()
        self.rollout_evaluator = rollout_evaluator

    def plan(self, driver, ledger, machine):
        current_time = float(driver.env.current_time)
        actions = self.build_root_actions(driver, ledger, machine)
        edges = [VCMCTSEdgeStats(action=action) for action in actions]
        iteration_count = max(len(edges), int(self.config.n_iter))
        for _ in range(iteration_count):
            edge = self._select_edge(edges)
            objective = self.evaluate_action(driver, ledger, edge.action)
            edge.record(objective)
        selected = self._choose_final_action(edges)
        return VCMCTSDecisionTrace(
            selected_action=selected.action,
            edge_stats=edges,
            current_time=current_time,
            machine=int(machine),
        )

    def _select_edge(self, edges):
        for edge in edges:
            if edge.visits == 0:
                return edge
        total_visits = sum(edge.visits for edge in edges)
        log_total = math.log(max(total_visits, 1))

        def uct(edge):
            mean = edge.mean_objective
            exploitation = objective_to_score(mean, self.config)
            exploration = (
                float(self.config.exploration_c)
                * float(edge.action.prior)
                * math.sqrt(log_total / max(edge.visits, 1))
            )
            return exploitation + exploration

        return max(edges, key=uct)

    def _choose_final_action(self, edges):
        def key(edge):
            objective = edge.mean_objective
            if objective is None:
                return (0.0, 0.0, 0.0, 0.0, -1)
            return (
                -float(objective.qtime_violation_count),
                -float(objective.qtime_violation_total),
                -float(objective.priority_weighted_wait),
                float(objective.avg_utilization),
                edge.visits,
            )

        ranked = sorted(edges, key=key, reverse=True)
        selected = ranked[0]
        if selected.action.kind != "no_op":
            return selected

        alternatives = [
            edge
            for edge in ranked
            if edge.action.kind != "no_op" and edge.mean_objective is not None
        ]
        noop_objective = selected.mean_objective
        if not alternatives or noop_objective is None:
            return selected

        best_alternative = alternatives[0]
        alternative_objective = best_alternative.mean_objective
        noop_has_qtime_advantage = (
            float(noop_objective.qtime_violation_count)
            < float(alternative_objective.qtime_violation_count)
            or float(noop_objective.qtime_violation_total)
            < float(alternative_objective.qtime_violation_total)
        )
        return selected if noop_has_qtime_advantage else best_alternative

    def build_root_actions(self, driver, ledger, machine):
        machine = int(machine)
        actions = [VCMCTSAction(kind="no_op", machine=machine, prior=0.05)]

        pool = driver.env.build_candidate_pool(machine)
        dispatch = []
        for index, action in enumerate(pool.actions):
            if not bool(pool.action_mask[index]):
                continue
            if getattr(action, "is_padding", False) or getattr(action, "is_wait", False):
                continue
            if int(action.ppid) == 0:
                continue
            dispatch.append((float(getattr(action, "score", 0.0)), index, action))
        dispatch.sort(key=lambda row: (-row[0], row[1]))
        for score, index, action in dispatch[: self.config.top_k_dispatch]:
            actions.append(
                VCMCTSAction(
                    kind="dispatch",
                    machine=machine,
                    action_index=int(index),
                    lot=int(action.lot),
                    ppid=int(action.ppid),
                    prior=max(1e-6, float(score) + 1.0),
                )
            )

        opportunities = detect_reservation_opportunities(
            driver.env,
            machines=[machine],
            ledger=ledger,
            top_b=self.config.top_b_reserve,
            min_priority_gap=self.config.min_priority_gap,
        )
        for opportunity in opportunities:
            actions.append(
                VCMCTSAction(
                    kind="reserve",
                    machine=int(opportunity.machine),
                    future_lot=int(opportunity.future_lot),
                    eta=float(opportunity.eta),
                    prior=max(1e-6, float(opportunity.score)),
                )
            )
        return actions

    def evaluate_action(self, driver, ledger, action):
        if self.rollout_evaluator is not None:
            return self.rollout_evaluator(driver, ledger, action, self.config)

        branch_driver = clone_driver_for_rollout(driver)
        branch_ledger = clone_ledger_for_rollout(ledger)
        self._apply_action(branch_driver, branch_ledger, action)
        run_rule_episode_with_reservations(
            branch_driver,
            ledger=branch_ledger,
            strategy=self.config.rollout_strategy,
            max_steps=self.config.rollout_max_steps or branch_driver.max_steps,
        )
        metrics = schedule_metrics_with_priority_wait(branch_driver.env.encoder, branch_driver.env)
        return VCMCTSObjective(
            qtime_violation_count=float(metrics["qtime_violation_count"]),
            qtime_violation_total=float(metrics["qtime_violation_total"]),
            priority_weighted_wait=float(metrics["priority_weighted_wait"]),
            avg_utilization=float(metrics["avg_utilization"]),
        )

    def _apply_action(self, driver, ledger, action):
        if action.kind == "no_op":
            advance_to_next_event_with_ledger(driver, ledger)
            return
        if action.kind == "reserve":
            ledger.reserve(
                action.machine,
                action.future_lot,
                eta=action.eta,
                created_at=driver.env.current_time,
                expires_at=float(action.eta) + float(self.config.reservation_ttl),
                reason="vc_mcts",
            )
            return
        if action.kind == "dispatch":
            pool = driver.env.build_candidate_pool(action.machine)
            driver.step_with_action(action.machine, action.action_index, pool=pool)
            return
        raise ValueError(f"unknown VC-MCTS action kind: {action.kind!r}")


def _valid_real_action_index_for_lot(pool, lot):
    lot = int(lot)
    for index, (action, is_valid) in enumerate(zip(pool.actions, pool.action_mask)):
        if not bool(is_valid):
            continue
        if getattr(action, "is_padding", False) or getattr(action, "is_wait", False):
            continue
        if int(action.lot) == lot and int(action.ppid) != 0:
            return int(index)
    return None


def _dispatch_reserved_target_if_ready(driver, ledger):
    env = driver.env
    for machine in sorted(ledger.reserved_machines()):
        record = ledger.get(machine)
        if record is None or int(record.future_lot) in env.completed_lots:
            ledger.release(machine)
            continue
        if float(env.current_time) < float(record.eta):
            continue
        pool = env.build_candidate_pool(machine)
        action_index = _valid_real_action_index_for_lot(pool, record.future_lot)
        if action_index is None:
            if float(env.current_time) >= float(record.expires_at):
                ledger.release(machine)
            continue
        result = driver.step_with_action(machine, action_index, pool=pool)
        if result.committed:
            ledger.consume_for_lot(machine, record.future_lot)
            return result
    return None


def _write_trace(trace_writer, trace):
    if trace_writer is None:
        return
    record = trace.to_dict()
    if callable(trace_writer):
        trace_writer(record)
        return
    trace_writer.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_vc_mcts_reservation_episode(
    driver,
    planner=None,
    ledger=None,
    max_steps=None,
    max_decisions=None,
    stop_after_reserve_available=None,
    stop_after_reserve_selected=None,
    trace_writer=None,
    progress_every=0,
):
    """Run an online VC-MCTS episode while honoring existing reservations."""
    planner = planner if planner is not None else VCMCTSPlanner()
    ledger = ledger if ledger is not None else ReservationLedger()
    steps = 0
    decisions = 0
    reserve_available_seen = 0
    reserve_selected_seen = 0
    reservations_made = 0
    episode_reward = 0.0
    limit = int(driver.max_steps if max_steps is None else max_steps)

    while steps < limit:
        done, reason = driver.is_episode_done()
        if done:
            driver.termination_reason = reason
            break

        ledger.release_expired(driver.env.current_time)
        reserved_result = _dispatch_reserved_target_if_ready(driver, ledger)
        if reserved_result is not None:
            episode_reward += float(reserved_result.reward)
            steps += 1
            continue

        machines = [
            m for m in driver.get_dispatchable_machines()
            if m not in ledger.reserved_machines()
        ]
        if not machines:
            next_time = advance_to_next_event_with_ledger(driver, ledger)
            if next_time is None:
                if not driver.termination_reason:
                    driver.termination_reason = "no_future_event"
                break
            steps += 1
            continue

        machine = driver.select_next_machine(machines)
        trace = planner.plan(driver, ledger, machine)
        decisions += 1
        _write_trace(trace_writer, trace)
        diagnostics = trace.to_dict()["diagnostics"]
        if diagnostics["reserve_was_available"]:
            reserve_available_seen += 1
        if diagnostics["reserve_was_selected"]:
            reserve_selected_seen += 1
        if progress_every and decisions % int(progress_every) == 0:
            print(
                "[vc_mcts] decision "
                f"{decisions} t={driver.env.current_time:.3f} "
                f"machine={machine} selected={trace.selected_action.kind}",
                file=sys.stderr,
                flush=True,
            )
        if (
            stop_after_reserve_available is not None
            and reserve_available_seen >= int(stop_after_reserve_available)
        ):
            driver.termination_reason = "reserve_available_limit_exceeded"
            break
        if (
            stop_after_reserve_selected is not None
            and reserve_selected_seen >= int(stop_after_reserve_selected)
        ):
            driver.termination_reason = "reserve_selected_limit_exceeded"
            break
        if max_decisions is not None and decisions >= int(max_decisions):
            driver.termination_reason = "max_decisions_exceeded"
            break
        action = trace.selected_action
        if action.kind == "reserve":
            ledger.reserve(
                action.machine,
                action.future_lot,
                eta=action.eta,
                created_at=driver.env.current_time,
                expires_at=float(action.eta) + float(planner.config.reservation_ttl),
                reason="vc_mcts",
            )
            reservations_made += 1
            steps += 1
            continue
        if action.kind == "no_op":
            next_time = advance_to_next_event_with_ledger(driver, ledger)
            if next_time is None:
                if not driver.termination_reason:
                    driver.termination_reason = "no_future_event"
                break
            steps += 1
            continue
        if action.kind == "dispatch":
            pool = driver.env.build_candidate_pool(action.machine)
            result = driver.step_with_action(action.machine, action.action_index, pool=pool)
            episode_reward += float(result.reward)
            steps += 1
            continue
        raise ValueError(f"unknown VC-MCTS action kind: {action.kind!r}")

    finalize_reservation_ledger(ledger, driver.env)
    summary = driver._summary(steps, episode_reward)
    summary["vc_mcts_decisions"] = int(decisions)
    summary["reservations_made"] = int(reservations_made)
    summary["reserve_available_decisions"] = int(reserve_available_seen)
    summary["reserve_selected_decisions"] = int(reserve_selected_seen)
    summary["active_reservations"] = len(ledger.reserved_machines())
    return summary
