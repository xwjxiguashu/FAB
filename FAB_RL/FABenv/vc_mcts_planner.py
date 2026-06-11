"""Online VC-MCTS reservation planner.

This first slice is a root-level MCTS planner: build root actions, evaluate
branches with reservation-aware rollouts, and choose with a lexicographic
objective tie-break.
"""
from dataclasses import dataclass, field, replace
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
    use_delegate_dispatch: bool = False
    prior_source: str = "heuristic"
    policy_reserve_prior: float = 0.15
    use_leaf_value: bool = False
    leaf_rollout_depth: int = 8
    arrival_prob_weighting: bool = False
    arrival_prob_decay: float = 1.0
    lookahead_window: float = 4.0
    # 机制 3 (报告 §7.9): CRN 多路噪声 rollout。crn_noise=True 时每次 evaluate_action
    # 跑 n_mc 条带噪 rollout 并取均值 Ê[obj]，各候选边复用同一组 crn_seed_base+k 种子
    # (公共随机数 → 比较时共同噪声相减抵消)。crn_noise=False (默认) → 行为不变。
    crn_noise: bool = False
    n_mc: int = 1
    crn_seed_base: int = 0
    # 优化① (2026-06-11): rollout clone 上的 qtime mask 口径覆盖。None (默认) =
    # 与真实 env 相同 (行为不变)。设为 "aggregate" 时只降级搜索估值用的 rollout
    # (~8-9x 提速)，真实决策池与 commit 准入仍走 env 自身口径 (chain_joint)，
    # 硬约束防线不受影响；所有边共享同一降级偏差，字典序比较中近似抵消。
    rollout_qtime_mask_mode: str | None = None
    # 机制 2 (报告8 §7.12): 优先级-能力对冲水位 ρ_pc (二部匹配裕量)。use_rho_pc=True
    # 时每条 root 边算 ρ̃_pc(s⊕a) 的 before/after/delta，并把 UCT exploitation 换成
    # α·q̂ + (1−α)·ρ̂_pc 插值 (rho_pc_alpha=1.0 即纯 q̂，等价旧行为作消融基线)；
    # rho_pc_weight 保留为 Δρ_pc 的加性兼容旋钮。只影响搜索引导，最终
    # objective-first 字典序选择不变 (硬约束 Q-time→O2→util 保证不受影响)。
    use_rho_pc: bool = False
    rho_pc_weight: float = 0.0
    rho_pc_alpha: float = 1.0
    rho_pc_priority_threshold: float | None = None


@dataclass(frozen=True)
class VCMCTSObjective:
    qtime_violation_count: float
    priority_weighted_wait: float
    avg_utilization: float
    qtime_violation_total: float = 0.0
    is_leaf_bootstrap: bool = False


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
    leaf_bootstrap_visits: int = 0
    # 机制 2: 二部匹配对冲水位 (状态导出, plan() 时算一次, 不随 rollout 变)
    rho_pc_before: float = 0.0
    rho_pc_after: float = 0.0
    delta_rho_pc: float = 0.0

    def record(self, objective):
        self.visits += 1
        self.total_qtime += float(objective.qtime_violation_count)
        self.total_qtime_severity += float(objective.qtime_violation_total)
        self.total_o2 += float(objective.priority_weighted_wait)
        self.total_util += float(objective.avg_utilization)
        if bool(getattr(objective, "is_leaf_bootstrap", False)):
            self.leaf_bootstrap_visits += 1

    @property
    def mean_objective(self):
        if self.visits <= 0:
            return None
        return VCMCTSObjective(
            qtime_violation_count=self.total_qtime / self.visits,
            qtime_violation_total=self.total_qtime_severity / self.visits,
            priority_weighted_wait=self.total_o2 / self.visits,
            avg_utilization=self.total_util / self.visits,
            is_leaf_bootstrap=bool(self.leaf_bootstrap_visits > 0),
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
            "leaf_bootstrap_visits": int(self.leaf_bootstrap_visits),
            "mean_is_leaf_bootstrap": None if mean is None else bool(mean.is_leaf_bootstrap),
            "rho_pc": float(self.rho_pc_after),
            "rho_pc_before": float(self.rho_pc_before),
            "rho_pc_after": float(self.rho_pc_after),
            "delta_rho_pc": float(self.delta_rho_pc),
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
        dispatch_edges = [
            edge for edge in edges
            if edge["kind"] in ("dispatch", "delegate_dispatch")
        ]
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


def blend_objectives(p, arrive_obj, miss_obj):
    """Probability-weighted blend of two objectives (方向2 最小验证).

    ``p`` is the arrival probability of the reserved future lot. With prob ``p``
    the lot arrives and the reserve branch (``arrive_obj``) is realized; with
    prob ``1-p`` it does not, and the no-reserve alternative (``miss_obj``,
    typically the best dispatch/no_op branch) is what actually happens — so the
    reserve's *expected* value discounts toward the miss outcome as ``p`` drops.
    """
    q = 1.0 - float(p)
    return VCMCTSObjective(
        qtime_violation_count=p * arrive_obj.qtime_violation_count
        + q * miss_obj.qtime_violation_count,
        qtime_violation_total=p * arrive_obj.qtime_violation_total
        + q * miss_obj.qtime_violation_total,
        priority_weighted_wait=p * arrive_obj.priority_weighted_wait
        + q * miss_obj.priority_weighted_wait,
        avg_utilization=p * arrive_obj.avg_utilization
        + q * miss_obj.avg_utilization,
        is_leaf_bootstrap=bool(
            arrive_obj.is_leaf_bootstrap or miss_obj.is_leaf_bootstrap
        ),
    )


def mean_objective(objectives):
    """Arithmetic mean of a list of objectives (机制 3: Ê[obj] over N_mc rollouts)."""
    objectives = [obj for obj in objectives if obj is not None]
    if not objectives:
        return None
    n = float(len(objectives))
    return VCMCTSObjective(
        qtime_violation_count=sum(o.qtime_violation_count for o in objectives) / n,
        qtime_violation_total=sum(o.qtime_violation_total for o in objectives) / n,
        priority_weighted_wait=sum(o.priority_weighted_wait for o in objectives) / n,
        avg_utilization=sum(o.avg_utilization for o in objectives) / n,
        is_leaf_bootstrap=any(o.is_leaf_bootstrap for o in objectives),
    )


class VCMCTSPlanner:
    def __init__(
        self,
        config=None,
        rollout_evaluator=None,
        dispatch_delegate=None,
        prior_provider=None,
        leaf_value=None,
    ):
        self.config = config if config is not None else VCMCTSConfig()
        self.rollout_evaluator = rollout_evaluator
        self.dispatch_delegate = dispatch_delegate
        self.prior_provider = prior_provider
        self.leaf_value = leaf_value

    def plan(self, driver, ledger, machine):
        current_time = float(driver.env.current_time)
        actions = self.build_root_actions(driver, ledger, machine)
        edges = [VCMCTSEdgeStats(action=action) for action in actions]
        if self.config.use_rho_pc:
            from priority_capability_matching import rho_pc_for_action

            for edge in edges:
                rho = rho_pc_for_action(
                    driver.env,
                    ledger,
                    edge.action,
                    priority_threshold=self.config.rho_pc_priority_threshold,
                )
                edge.rho_pc_before = rho.before
                edge.rho_pc_after = rho.after
                edge.delta_rho_pc = rho.delta
        iteration_count = max(len(edges), int(self.config.n_iter))
        for _ in range(iteration_count):
            edge = self._select_edge(edges)
            objective = self.evaluate_action(driver, ledger, edge.action)
            edge.record(objective)
        selected = self._choose_final_action(edges, current_time)
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

        # 机制 2 (报告8 §7.12.3): q̂ 在边集内 min-max 归一化到 [0,1]，使其与
        # ρ̂_pc (天然 ∈ [0,1]) 可比，才能做 α 插值。
        raw_scores = {
            id(edge): objective_to_score(edge.mean_objective, self.config)
            for edge in edges
            if edge.mean_objective is not None
        }
        if raw_scores:
            min_score = min(raw_scores.values())
            max_score = max(raw_scores.values())
            score_span = max(max_score - min_score, 1e-9)
        else:
            min_score = 0.0
            score_span = 1.0

        def uct(edge):
            mean = edge.mean_objective
            raw_score = objective_to_score(mean, self.config)
            if self.config.use_rho_pc:
                # E(s,a) = α·q̂ + (1−α)·ρ̂_pc (只引导搜索, 不改最终字典序选择);
                # rho_pc_weight·Δρ_pc 保留为旧探针的加性兼容旋钮。
                q_hat = (raw_score - min_score) / score_span
                alpha = min(1.0, max(0.0, float(self.config.rho_pc_alpha)))
                exploitation = (
                    alpha * float(q_hat)
                    + (1.0 - alpha) * float(edge.rho_pc_after)
                )
                exploitation += float(self.config.rho_pc_weight) * float(edge.delta_rho_pc)
            else:
                exploitation = raw_score
            exploration = (
                float(self.config.exploration_c)
                * float(edge.action.prior)
                * math.sqrt(log_total / max(edge.visits, 1))
            )
            return exploitation + exploration

        return max(edges, key=uct)

    def _arrival_prob(self, eta, now):
        """Arrival probability of a future lot, decaying with ETA distance."""
        if eta is None or not self.config.arrival_prob_weighting:
            return 1.0
        dist = max(0.0, float(eta) - float(now))
        window = max(1e-9, float(self.config.lookahead_window))
        return float(math.exp(-float(self.config.arrival_prob_decay) * dist / window))

    def _effective_objectives(self, edges, current_time):
        """Per-edge objective used for the final pick.

        Identical to ``edge.mean_objective`` unless arrival-prob weighting is on,
        in which case each reserve edge's objective is blended toward the best
        non-reserve branch by its future lot's arrival probability (方向2).
        """
        effective = {id(edge): edge.mean_objective for edge in edges}
        if not self.config.arrival_prob_weighting:
            return effective
        non_reserve = [
            edge
            for edge in edges
            if edge.action.kind != "reserve" and edge.mean_objective is not None
        ]
        if not non_reserve:
            return effective
        miss_obj = max(
            non_reserve,
            key=lambda edge: objective_to_score(edge.mean_objective, self.config),
        ).mean_objective
        for edge in edges:
            if edge.action.kind == "reserve" and edge.mean_objective is not None:
                p = self._arrival_prob(edge.action.eta, current_time)
                effective[id(edge)] = blend_objectives(p, edge.mean_objective, miss_obj)
        return effective

    def _choose_final_action(self, edges, current_time=0.0):
        effective = self._effective_objectives(edges, current_time)

        def key(edge):
            objective = effective[id(edge)]
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
            if edge.action.kind != "no_op" and effective[id(edge)] is not None
        ]
        noop_objective = effective[id(selected)]
        if not alternatives or noop_objective is None:
            return selected

        best_alternative = alternatives[0]
        alternative_objective = effective[id(best_alternative)]
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
        if self.config.use_delegate_dispatch:
            action_index = None
            if self.dispatch_delegate is not None:
                action_index = self.dispatch_delegate.select_action_index(
                    driver,
                    machine,
                    pool=pool,
                )
            if action_index is not None:
                action = pool.actions[int(action_index)]
                actions.append(
                    VCMCTSAction(
                        kind="delegate_dispatch",
                        machine=machine,
                        action_index=int(action_index),
                        lot=int(action.lot),
                        ppid=int(action.ppid),
                        prior=max(
                            1e-6,
                            float(getattr(action, "score", 0.0)) + 1.0,
                        ),
                    )
                )
        else:
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
        if self.config.prior_source == "policy" and self.prior_provider is not None:
            actions = self._assign_policy_priors(actions, driver, machine, pool)
        return actions

    def _assign_policy_priors(self, actions, driver, machine, pool):
        """Overwrite root priors with SAS policy probabilities plus reserve mass."""
        probs = self.prior_provider.candidate_probs(driver, machine, pool=pool)
        prob_count = len(probs)

        wait_prob = 0.0
        for index, action in enumerate(pool.actions):
            if not bool(pool.action_mask[index]):
                continue
            if getattr(action, "is_wait", False) and index < prob_count:
                wait_prob += float(probs[index])

        raw_priors = []
        for action in actions:
            if action.kind in ("dispatch", "delegate_dispatch"):
                index = int(action.action_index)
                prior = float(probs[index]) if 0 <= index < prob_count else 0.0
            elif action.kind == "no_op":
                prior = wait_prob
            elif action.kind == "reserve":
                prior = float(self.config.policy_reserve_prior)
            else:
                prior = 0.0
            raw_priors.append(max(1e-6, prior))

        total = float(sum(raw_priors))
        return [
            replace(action, prior=float(prior) / total)
            for action, prior in zip(actions, raw_priors)
        ]

    def evaluate_action(self, driver, ledger, action):
        if self.rollout_evaluator is not None:
            return self.rollout_evaluator(driver, ledger, action, self.config)

        if not self.config.crn_noise or int(self.config.n_mc) <= 1:
            return self._evaluate_action_once(driver, ledger, action, noise_seed=None)

        # 机制 3 (报告 §7.9): N_mc 条带噪 rollout 取均值 Ê[obj]。同一节点下所有候选边
        # 复用同一组种子 crn_seed_base + k，公共随机数在比较时相减抵消 → 比的是动作差异
        # 而非噪声。小 n_mc (3–8) 即可给出稳定排序。
        samples = []
        for k in range(int(self.config.n_mc)):
            seed = int(self.config.crn_seed_base) + k
            samples.append(
                self._evaluate_action_once(driver, ledger, action, noise_seed=seed)
            )
        return mean_objective(samples)

    def _evaluate_action_once(self, driver, ledger, action, noise_seed=None):
        branch_driver = clone_driver_for_rollout(driver)
        branch_ledger = clone_ledger_for_rollout(ledger)
        if self.config.rollout_qtime_mask_mode:
            branch_driver.env.qtime_mask_mode = str(self.config.rollout_qtime_mask_mode)
        if noise_seed is not None:
            branch_driver.env.enable_process_noise(noise_seed)
        self._apply_action(branch_driver, branch_ledger, action)
        if self.config.use_leaf_value and self.leaf_value is not None:
            objective = self._leaf_value_objective(branch_driver, branch_ledger)
            if objective is not None:
                return objective

        run_rule_episode_with_reservations(
            branch_driver,
            ledger=branch_ledger,
            strategy=self.config.rollout_strategy,
            max_steps=self.config.rollout_max_steps or branch_driver.max_steps,
            dispatch_delegate=self.dispatch_delegate,
        )
        metrics = schedule_metrics_with_priority_wait(branch_driver.env.encoder, branch_driver.env)
        return VCMCTSObjective(
            qtime_violation_count=float(metrics["qtime_violation_count"]),
            qtime_violation_total=float(metrics["qtime_violation_total"]),
            priority_weighted_wait=float(metrics["priority_weighted_wait"]),
            avg_utilization=float(metrics["avg_utilization"]),
        )

    def _leaf_value_objective(self, branch_driver, branch_ledger):
        """Partial rollout to a leaf, then bootstrap covered objective dimensions."""
        run_rule_episode_with_reservations(
            branch_driver,
            ledger=branch_ledger,
            strategy=self.config.rollout_strategy,
            max_steps=int(self.config.leaf_rollout_depth),
            dispatch_delegate=self.dispatch_delegate,
        )

        done, _reason = branch_driver.is_episode_done()
        if done:
            return None

        machine = self._leaf_machine(branch_driver, branch_ledger)
        if machine is None:
            return None

        partial_metrics = schedule_metrics_with_priority_wait(
            branch_driver.env.encoder,
            branch_driver.env,
        )
        critic_values = self.leaf_value.estimate(branch_driver, machine)
        from vc_mcts_alphazero import critic_to_objective_dims

        dims = critic_to_objective_dims(
            critic_values,
            partial_metrics,
            num_lots=branch_driver.env.encoder.num_lots,
        )
        return VCMCTSObjective(
            qtime_violation_count=float(dims["qtime_violation_count"]),
            qtime_violation_total=float(dims["qtime_violation_total"]),
            priority_weighted_wait=float(dims["priority_weighted_wait"]),
            avg_utilization=float(dims["avg_utilization"]),
            is_leaf_bootstrap=True,
        )

    def _leaf_machine(self, driver, ledger):
        machines = [
            machine
            for machine in driver.get_dispatchable_machines()
            if machine not in ledger.reserved_machines()
        ]
        if not machines:
            if advance_to_next_event_with_ledger(driver, ledger) is None:
                return None
            machines = [
                machine
                for machine in driver.get_dispatchable_machines()
                if machine not in ledger.reserved_machines()
            ]
        if not machines:
            return None
        return driver.select_next_machine(machines)

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
        if action.kind in ("dispatch", "delegate_dispatch"):
            pool = driver.env.build_candidate_pool(action.machine)
            action_index = action.action_index
            if action.kind == "delegate_dispatch" and self.dispatch_delegate is not None:
                action_index = self.dispatch_delegate.select_action_index(
                    driver,
                    action.machine,
                    pool=pool,
                )
            if action_index is None:
                advance_to_next_event_with_ledger(driver, ledger)
                return
            driver.step_with_action(action.machine, action_index, pool=pool)
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
    dispatch_delegate=None,
):
    """Run an online VC-MCTS episode while honoring existing reservations."""
    planner = planner if planner is not None else VCMCTSPlanner()
    ledger = ledger if ledger is not None else ReservationLedger()
    dispatch_delegate = dispatch_delegate or getattr(planner, "dispatch_delegate", None)
    steps = 0
    decisions = 0
    reserve_available_seen = 0
    reserve_selected_seen = 0
    reservations_made = 0
    reservations_consumed = 0
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
            reservations_consumed += 1
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
        if action.kind in ("dispatch", "delegate_dispatch"):
            pool = driver.env.build_candidate_pool(action.machine)
            action_index = action.action_index
            if action.kind == "delegate_dispatch" and dispatch_delegate is not None:
                action_index = dispatch_delegate.select_action_index(
                    driver,
                    action.machine,
                    pool=pool,
                )
            if action_index is None:
                next_time = advance_to_next_event_with_ledger(driver, ledger)
                if next_time is None:
                    if not driver.termination_reason:
                        driver.termination_reason = "no_future_event"
                    break
                steps += 1
                continue
            result = driver.step_with_action(action.machine, action_index, pool=pool)
            episode_reward += float(result.reward)
            steps += 1
            continue
        raise ValueError(f"unknown VC-MCTS action kind: {action.kind!r}")

    finalize_reservation_ledger(ledger, driver.env)
    summary = driver._summary(steps, episode_reward)
    summary["vc_mcts_decisions"] = int(decisions)
    summary["reservations_made"] = int(reservations_made)
    summary["reservations_consumed"] = int(reservations_consumed)
    summary["reserve_available_decisions"] = int(reserve_available_seen)
    summary["reserve_selected_decisions"] = int(reserve_selected_seen)
    summary["active_reservations"] = len(ledger.reserved_machines())
    if dispatch_delegate is not None:
        summary["dispatch_delegate"] = getattr(
            dispatch_delegate,
            "label",
            dispatch_delegate.__class__.__name__,
        )
    return summary
