"""Rollout helpers for reservation go/no-go experiments.

This module is the Scheme C bridge: it keeps the existing SAS/rule dispatch
path intact, wraps it with a reservation ledger, and exposes metrics suitable
for oracle comparisons before implementing online VC-MCTS.
"""
import copy

import numpy as np

from phase2_sas_driver import Phase2EpisodeDriver
from reservation_ledger import ReservationLedger
from reservation_rop import detect_reservation_opportunities


def schedule_metrics_with_priority_wait(encoder, env):
    """Return evaluation metrics plus an O2-style weighted waiting proxy."""
    obj = encoder.evaluate_objectives(
        env.lot_schedule,
        env.wafer_schedule,
        current_time=0.0,
    )
    metrics = {
        "qtime_violation_count": float(obj[0]),
        "qtime_violation_total": float(obj[1]),
        "tardy_count": float(obj[2]),
        "total_tardiness": float(obj[3]),
        "priority_violation": float(obj[4]),
        "avg_utilization": float(-obj[5]),
        "completed_lots": float(len(env.completed_lots)),
    }
    priority_wait = 0.0
    lot_schedule = np.asarray(env.lot_schedule, dtype=float).reshape((-1, 5))
    for row in lot_schedule:
        lot = int(row[0])
        start = float(row[3])
        arrival = float(encoder.arrival_times.get(lot, 0.0))
        priority = float(getattr(encoder, "priorities", {}).get(lot, 1.0))
        priority_wait += priority * max(0.0, start - arrival)
    metrics["priority_weighted_wait"] = float(priority_wait)
    return metrics


def is_reservation_rollout_better(baseline_metrics, reserve_metrics):
    """Return True only when reserve passes Q-time and improves O2.

    Q-time is a gate here, not the optimization objective. This prevents the
    oracle from accepting a reservation that reduces Q-time violations by
    buying a large priority-weighted waiting regression.
    """
    return _reservation_rollout_better(
        baseline_metrics,
        reserve_metrics,
        qtime_tolerance=0.0,
        min_o2_gain=1e-9,
        min_util_gain=1e-9,
    )


def _reservation_rollout_better(
    baseline_metrics,
    reserve_metrics,
    qtime_tolerance=0.0,
    min_o2_gain=1e-9,
    min_util_gain=1e-9,
):
    baseline_q = float(baseline_metrics.get("qtime_violation_count", 0.0))
    reserve_q = float(reserve_metrics.get("qtime_violation_count", 0.0))
    if reserve_q > baseline_q + float(qtime_tolerance):
        return False

    baseline_o2 = float(baseline_metrics.get("priority_weighted_wait", 0.0))
    reserve_o2 = float(reserve_metrics.get("priority_weighted_wait", 0.0))
    if reserve_o2 < baseline_o2 - float(min_o2_gain):
        return True
    if abs(reserve_o2 - baseline_o2) <= float(min_o2_gain):
        baseline_util = float(baseline_metrics.get("avg_utilization", 0.0))
        reserve_util = float(reserve_metrics.get("avg_utilization", 0.0))
        return reserve_util > baseline_util + float(min_util_gain)
    return False


def finalize_reservation_ledger(ledger, env):
    """Release completed, expired, and terminal leftover reservations."""
    released = list(ledger.release_expired(env.current_time))
    completed = set(getattr(env, "completed_lots", set()))
    all_done = len(getattr(env, "remaining_lots", set())) == 0
    for machine in sorted(ledger.reserved_machines()):
        record = ledger.get(machine)
        if record is None:
            continue
        if all_done or int(record.future_lot) in completed:
            released_record = ledger.release(machine)
            if released_record is not None:
                released.append(released_record)
    released.sort(key=lambda r: (r.machine, r.future_lot))
    return released


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


def _next_event_time_with_ledger(env, ledger):
    times = []
    base_time = env.next_event_time()
    if base_time is not None:
        times.append(float(base_time))
    for machine in ledger.reserved_machines():
        record = ledger.get(machine)
        if record is None:
            continue
        if record.eta > env.current_time:
            times.append(float(record.eta))
        if record.expires_at > env.current_time:
            times.append(float(record.expires_at))
    if not times:
        return None
    return min(times)


def _advance_to_next_event_with_ledger(driver, ledger):
    next_time = _next_event_time_with_ledger(driver.env, ledger)
    if next_time is None:
        return None
    if float(next_time) <= float(driver.env.current_time):
        driver.unrecoverable_error = True
        driver.termination_reason = "unrecoverable_error"
        return None
    driver.env.advance_time(next_time)
    driver.total_wait_steps_per_episode += 1
    return float(next_time)


def _dispatch_reserved_target_if_ready(driver, ledger):
    env = driver.env
    for machine in sorted(ledger.reserved_machines()):
        record = ledger.get(machine)
        if record is None or record.future_lot in env.completed_lots:
            ledger.release(machine)
            continue
        if float(env.current_time) < record.eta:
            continue
        pool = env.build_candidate_pool(machine)
        action_index = _valid_real_action_index_for_lot(pool, record.future_lot)
        if action_index is None:
            if float(env.current_time) >= record.expires_at:
                ledger.release(machine)
            continue
        result = driver.step_with_action(machine, action_index, pool=pool)
        if result.committed:
            ledger.consume_for_lot(machine, record.future_lot)
            return result
    return None


def run_rule_episode_with_reservations(
    driver,
    ledger=None,
    strategy="FIFO",
    max_steps=None,
    dispatch_delegate=None,
):
    """Run a rule episode while honoring a reservation ledger."""
    if ledger is None:
        ledger = ReservationLedger()
    if strategy not in driver.RULE_STRATEGIES:
        raise ValueError(f"unknown strategy: {strategy}")

    steps = 0
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
            next_time = _advance_to_next_event_with_ledger(driver, ledger)
            if next_time is None:
                if not driver.termination_reason:
                    driver.termination_reason = "no_future_event"
                break
            steps += 1
            continue

        machine = driver.select_next_machine(machines)
        decision = driver.build_decision(machine)
        if dispatch_delegate is None:
            action_index = driver._rule_action_index(decision.pool, strategy)
        else:
            action_index = dispatch_delegate.select_action_index(
                driver,
                machine,
                pool=decision.pool,
            )
        if action_index is None:
            driver.consecutive_failed_actions += 1
            driver.failed_actions_per_episode += 1
            steps += 1
            continue

        result = driver.step_with_action(machine, action_index, pool=decision.pool)
        episode_reward += float(result.reward)
        steps += 1

    finalize_reservation_ledger(ledger, driver.env)
    summary = driver._summary(steps, episode_reward)
    summary["active_reservations"] = len(ledger.reserved_machines())
    return summary


def _clone_ledger(ledger):
    clone = ReservationLedger()
    for machine in sorted(ledger.reserved_machines()):
        record = ledger.get(machine)
        if record is None:
            continue
        clone.reserve(
            record.machine,
            record.future_lot,
            eta=record.eta,
            created_at=record.created_at,
            expires_at=record.expires_at,
            reason=record.reason,
        )
    return clone


def _clone_driver(driver):
    env = copy.deepcopy(driver.env)
    observation_encoder = copy.deepcopy(driver.observation_encoder)
    reward_config = copy.deepcopy(driver.reward_config)
    return Phase2EpisodeDriver(
        env,
        observation_encoder,
        reward_config,
        planning_horizon=driver.planning_horizon,
        max_steps=driver.max_steps,
        max_total_wait_steps_per_episode=driver.max_total_wait_steps_per_episode,
        max_failed_actions=driver.max_failed_actions,
    )


def advance_to_next_event_with_ledger(driver, ledger):
    """Public wrapper used by VC-MCTS no-op branches."""
    return _advance_to_next_event_with_ledger(driver, ledger)


def clone_ledger_for_rollout(ledger):
    """Public wrapper for non-destructive branch rollouts."""
    return _clone_ledger(ledger)


def clone_driver_for_rollout(driver):
    """Public wrapper for non-destructive branch rollouts."""
    return _clone_driver(driver)


def _rollout_metrics(driver, ledger, strategy, max_steps):
    branch_driver = _clone_driver(driver)
    branch_ledger = _clone_ledger(ledger)
    run_rule_episode_with_reservations(
        branch_driver,
        ledger=branch_ledger,
        strategy=strategy,
        max_steps=max_steps,
    )
    return schedule_metrics_with_priority_wait(branch_driver.env.encoder, branch_driver.env)


def choose_oracle_reservation(
    driver,
    ledger,
    opportunities,
    strategy="FIFO",
    max_steps=None,
):
    """Return the opportunity whose forced-reserve rollout beats baseline."""
    if not opportunities:
        return None, None, None
    limit = int(driver.max_steps if max_steps is None else max_steps)
    baseline_metrics = _rollout_metrics(driver, ledger, strategy, limit)
    best = None
    best_metrics = None
    for opportunity in opportunities:
        trial_ledger = _clone_ledger(ledger)
        trial_ledger.reserve(
            opportunity.machine,
            opportunity.future_lot,
            eta=opportunity.eta,
            created_at=driver.env.current_time,
            expires_at=opportunity.eta + 1.0,
            reason="oracle_probe",
        )
        reserve_metrics = _rollout_metrics(driver, trial_ledger, strategy, limit)
        if is_reservation_rollout_better(baseline_metrics, reserve_metrics):
            if best_metrics is None or is_reservation_rollout_better(best_metrics, reserve_metrics):
                best = opportunity
                best_metrics = reserve_metrics
    return best, baseline_metrics, best_metrics


def run_oracle_reservation_episode(
    driver,
    strategy="FIFO",
    top_b=4,
    max_steps=None,
    min_priority_gap=0.0,
):
    """Run an online oracle probe episode.

    At each ROP hit the oracle evaluates forced-reserve rollouts against the
    no-new-reservation branch and commits a reservation only if it wins.
    """
    ledger = ReservationLedger()
    steps = 0
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
        opportunities = detect_reservation_opportunities(
            driver.env,
            machines=machines,
            ledger=ledger,
            top_b=top_b,
            min_priority_gap=min_priority_gap,
        )
        opportunity, _baseline, _reserve = choose_oracle_reservation(
            driver,
            ledger,
            opportunities,
            strategy=strategy,
            max_steps=limit,
        )
        if opportunity is not None:
            ledger.reserve(
                opportunity.machine,
                opportunity.future_lot,
                eta=opportunity.eta,
                created_at=driver.env.current_time,
                expires_at=opportunity.eta + 1.0,
                reason="oracle_probe",
            )
            reservations_made += 1
            continue

        if not machines:
            next_time = _advance_to_next_event_with_ledger(driver, ledger)
            if next_time is None:
                if not driver.termination_reason:
                    driver.termination_reason = "no_future_event"
                break
            steps += 1
            continue

        machine = driver.select_next_machine(machines)
        decision = driver.build_decision(machine)
        action_index = driver._rule_action_index(decision.pool, strategy)
        if action_index is None:
            driver.consecutive_failed_actions += 1
            driver.failed_actions_per_episode += 1
            steps += 1
            continue

        result = driver.step_with_action(machine, action_index, pool=decision.pool)
        episode_reward += float(result.reward)
        steps += 1

    finalize_reservation_ledger(ledger, driver.env)
    summary = driver._summary(steps, episode_reward)
    summary["reservations_made"] = int(reservations_made)
    summary["active_reservations"] = len(ledger.reserved_machines())
    return summary
