"""Reservation ledger for future-lot machine holds.

The ledger is intentionally small and policy-agnostic: it records which
machine is frozen for which future lot, and leaves the decision to reserve to
ROP/oracle/MCTS layers.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ReservationRecord:
    machine: int
    future_lot: int
    eta: float
    created_at: float
    expires_at: float
    reason: str = ""


class ReservationLedger:
    """In-memory reservation ledger keyed by machine id."""

    def __init__(self):
        self._records = {}

    def reserve(
        self,
        machine,
        future_lot,
        eta,
        created_at,
        expires_at,
        reason="",
    ):
        machine = int(machine)
        future_lot = int(future_lot)
        for record in self._records.values():
            if record.future_lot == future_lot and record.machine != machine:
                raise ValueError(
                    f"future lot {future_lot} is already reserved "
                    f"on machine {record.machine}"
                )
        record = ReservationRecord(
            machine=machine,
            future_lot=future_lot,
            eta=float(eta),
            created_at=float(created_at),
            expires_at=float(expires_at),
            reason=str(reason),
        )
        self._records[record.machine] = record
        return record

    def is_reserved(self, machine):
        return int(machine) in self._records

    def reserved_machines(self):
        return set(self._records)

    def reserved_lots(self):
        return {record.future_lot for record in self._records.values()}

    def is_lot_reserved(self, lot):
        return int(lot) in self.reserved_lots()

    def get(self, machine):
        return self._records.get(int(machine))

    def release(self, machine):
        return self._records.pop(int(machine), None)

    def release_expired(self, now):
        now = float(now)
        expired = [
            record
            for record in self._records.values()
            if record.expires_at <= now
        ]
        for record in expired:
            self._records.pop(record.machine, None)
        expired.sort(key=lambda r: (r.machine, r.future_lot))
        return expired

    def consume_for_lot(self, machine, lot):
        machine = int(machine)
        lot = int(lot)
        record = self._records.get(machine)
        if record is None or record.future_lot != lot:
            return None
        return self._records.pop(machine)
