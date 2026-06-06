from reservation_ledger import ReservationLedger
from reservation_rop import detect_reservation_opportunities
from rl_environment import ResourceCalendarEnv


def test_detects_future_high_priority_compatible_lots(small_encoder):
    env = ResourceCalendarEnv(small_encoder, top_k=8, w_lookahead=4.0)
    env.reset()

    opportunities = detect_reservation_opportunities(env, machines=[1, 2], top_b=10)

    assert opportunities
    assert all(o.machine in {1, 2} for o in opportunities)
    assert all(o.future_lot in small_encoder.feasible_machines for o in opportunities)
    assert all(o.eta > env.current_time for o in opportunities)


def test_detect_reservation_opportunities_respects_top_b_and_ranking(small_encoder):
    env = ResourceCalendarEnv(small_encoder, top_k=8, w_lookahead=4.0)
    env.reset()

    opportunities = detect_reservation_opportunities(env, machines=[1, 2], top_b=2)

    assert len(opportunities) == 2
    assert opportunities[0].score >= opportunities[1].score


def test_detect_reservation_opportunities_ignores_reserved_machines(small_encoder):
    env = ResourceCalendarEnv(small_encoder, top_k=8, w_lookahead=4.0)
    env.reset()
    ledger = ReservationLedger()
    ledger.reserve(machine=1, future_lot=2, eta=1.5, created_at=0.0, expires_at=2.5)

    opportunities = detect_reservation_opportunities(
        env,
        machines=[1, 2],
        ledger=ledger,
        top_b=10,
    )

    assert opportunities
    assert all(o.machine != 1 for o in opportunities)


def test_detect_reservation_opportunities_ignores_reserved_future_lots(small_encoder):
    env = ResourceCalendarEnv(small_encoder, top_k=8, w_lookahead=4.0)
    env.reset()
    unfiltered = detect_reservation_opportunities(env, machines=[1, 2], top_b=10)
    target = unfiltered[0]
    ledger = ReservationLedger()
    other_machine = 1 if target.machine != 1 else 2
    ledger.reserve(
        machine=other_machine,
        future_lot=target.future_lot,
        eta=target.eta,
        created_at=0.0,
        expires_at=target.eta + 1.0,
    )

    opportunities = detect_reservation_opportunities(
        env,
        machines=[target.machine],
        ledger=ledger,
        top_b=10,
    )

    assert all(o.future_lot != target.future_lot for o in opportunities)


def test_detect_reservation_opportunities_returns_empty_without_lookahead(small_encoder):
    env = ResourceCalendarEnv(small_encoder, top_k=8, w_lookahead=0.0)
    env.reset()

    opportunities = detect_reservation_opportunities(env, machines=[1, 2], top_b=10)

    assert opportunities == []
