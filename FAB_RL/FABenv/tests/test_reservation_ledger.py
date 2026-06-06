from reservation_ledger import ReservationLedger


def test_reserve_marks_machine_unavailable_until_target_arrives():
    ledger = ReservationLedger()

    record = ledger.reserve(
        machine=2,
        future_lot=7,
        eta=5.0,
        created_at=1.0,
        expires_at=6.0,
        reason="unit-test",
    )

    assert record.machine == 2
    assert ledger.is_reserved(2)
    assert ledger.reserved_machines() == {2}
    assert ledger.reserved_lots() == {7}
    assert ledger.is_lot_reserved(7)
    assert ledger.get(2).future_lot == 7


def test_release_expired_drops_only_stale_reservations():
    ledger = ReservationLedger()
    ledger.reserve(1, 10, eta=3.0, created_at=0.0, expires_at=4.0)
    ledger.reserve(2, 11, eta=8.0, created_at=0.0, expires_at=9.0)

    released = ledger.release_expired(now=5.0)

    assert [r.machine for r in released] == [1]
    assert not ledger.is_reserved(1)
    assert ledger.is_reserved(2)


def test_consume_for_lot_releases_matching_reservation():
    ledger = ReservationLedger()
    ledger.reserve(3, 12, eta=4.0, created_at=0.0, expires_at=8.0)
    ledger.reserve(4, 13, eta=5.0, created_at=0.0, expires_at=8.0)

    consumed = ledger.consume_for_lot(machine=3, lot=12)

    assert consumed is not None
    assert consumed.machine == 3
    assert consumed.future_lot == 12
    assert not ledger.is_reserved(3)
    assert ledger.is_reserved(4)


def test_consume_for_lot_keeps_non_matching_reservation():
    ledger = ReservationLedger()
    ledger.reserve(3, 12, eta=4.0, created_at=0.0, expires_at=8.0)

    consumed = ledger.consume_for_lot(machine=3, lot=99)

    assert consumed is None
    assert ledger.is_reserved(3)


def test_reserving_same_future_lot_on_another_machine_is_rejected():
    ledger = ReservationLedger()
    ledger.reserve(1, 12, eta=4.0, created_at=0.0, expires_at=8.0)

    try:
        ledger.reserve(2, 12, eta=4.5, created_at=0.0, expires_at=8.5)
    except ValueError as exc:
        assert "future lot 12" in str(exc)
    else:
        raise AssertionError("duplicate future lot reservation should be rejected")

    assert ledger.reserved_machines() == {1}
    assert ledger.reserved_lots() == {12}
