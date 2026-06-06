"""ROP detection for reservation planning.

ROP detection is deliberately a wide trigger: it proposes candidate
``(machine, future_lot)`` pairs that might be worth searching, but it does not
decide that reservation is good. Oracle/MCTS owns that comparison.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ReservationOpportunity:
    machine: int
    future_lot: int
    eta: float
    score: float
    priority_gap: float
    compatible_ppids: tuple


def _valid_real_actions(pool):
    out = []
    for action, is_valid in zip(pool.actions, pool.action_mask):
        if not bool(is_valid):
            continue
        if getattr(action, "is_wait", False) or getattr(action, "is_padding", False):
            continue
        if int(action.ppid) == 0:
            continue
        out.append(action)
    return out


def _compatible_ppids(encoder, lot, machine):
    ppids = []
    for ppid in encoder.feasible_ppids.get((int(lot), int(machine)), []):
        try:
            steps = encoder.get_process_steps(int(lot), int(machine), int(ppid))
        except (KeyError, ValueError):
            continue
        if steps:
            ppids.append(int(ppid))
    return tuple(ppids)


def detect_reservation_opportunities(
    env,
    machines=None,
    ledger=None,
    top_b=4,
    min_priority_gap=0.0,
):
    """Return ranked reservation opportunities for visible future lots.

    Args:
        env: ``ResourceCalendarEnv`` with lookahead enabled.
        machines: optional iterable of idle machine ids to inspect.
        ledger: optional ``ReservationLedger``; reserved machines are skipped.
        top_b: maximum number of opportunities to return.
        min_priority_gap: require future priority to beat the best current pool
            priority by at least this amount. ``0`` keeps the detector wide.
    """
    top_b = int(top_b)
    if top_b <= 0:
        return []

    if machines is None:
        machines = env.get_candidate_machines()
    reserved = ledger.reserved_machines() if ledger is not None else set()
    reserved_lots = ledger.reserved_lots() if ledger is not None else set()
    upcoming = list(env.upcoming_lots())
    if not upcoming:
        return []

    now = float(env.current_time)
    priorities = getattr(env.encoder, "priorities", {})
    arrivals = getattr(env.encoder, "arrival_times", {})
    opportunities = []

    for machine in machines:
        machine = int(machine)
        if machine in reserved:
            continue

        pool = env.build_candidate_pool(machine)
        current_actions = _valid_real_actions(pool)
        current_best_priority = max(
            (
                float(priorities.get(int(action.lot), 0.0))
                for action in current_actions
            ),
            default=0.0,
        )

        for lot in upcoming:
            lot = int(lot)
            if lot in reserved_lots:
                continue
            ppids = _compatible_ppids(env.encoder, lot, machine)
            if not ppids:
                continue
            future_priority = float(priorities.get(lot, 0.0))
            priority_gap = future_priority - current_best_priority
            if priority_gap < float(min_priority_gap):
                continue
            eta = float(arrivals.get(lot, now))
            eta_distance = max(eta - now, 1e-9)
            scarcity = 1.0 / max(1, len(env.encoder.feasible_machines.get(lot, [])))
            score = priority_gap + scarcity + 1.0 / (1.0 + eta_distance)
            opportunities.append(
                ReservationOpportunity(
                    machine=machine,
                    future_lot=lot,
                    eta=eta,
                    score=float(score),
                    priority_gap=float(priority_gap),
                    compatible_ppids=ppids,
                )
            )

    opportunities.sort(
        key=lambda item: (
            -item.score,
            item.eta,
            item.machine,
            item.future_lot,
        )
    )
    return opportunities[:top_b]
