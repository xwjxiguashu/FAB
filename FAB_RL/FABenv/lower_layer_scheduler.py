
"""State-aware lower-layer calendar scheduler.

This module shares the deterministic list-scheduling core with
``lower_layer_estimator.estimate``. The estimator stays state independent
(empty free times); this module reads committed calendar state to produce
absolute, non-destructive schedule intervals.
"""
from dataclasses import dataclass, field

import numpy as np

from lower_layer_estimator import compute_sub_batches, schedule_deterministic


@dataclass
class ScheduleResult:
    lot_start: float = 0.0
    lot_end: float = 0.0
    batch_intervals: list = field(default_factory=list)
    machine_interval: tuple = None
    subbatch_wafer_map: list = field(default_factory=list)
    infeasible_reason: str = ""


def _allowed_resources(encoder, machine):
    declared = getattr(encoder, "machine_resources", {})
    if not declared:
        return None
    return frozenset(
        (int(chamber), int(side))
        for chamber, side in declared.get(int(machine), [])
    )


def _stage_sigma(encoder, lot, machine, ppid, stage_id):
    sigmas = getattr(encoder, "process_time_sigma", {}).get(
        (int(lot), int(machine), int(ppid))
    )
    if not sigmas:
        return 0.0
    idx = int(stage_id) - 1
    if idx < 0 or idx >= len(sigmas):
        return 0.0
    return max(0.0, float(sigmas[idx]))


def _build_stage_options(encoder, machine, steps):
    allowed = _allowed_resources(encoder, machine)
    stage_options = []
    for stage in steps:
        arr = np.asarray(stage, dtype=float)
        rows = []
        for row in arr:
            chamber, side, process_time = int(row[0]), int(row[1]), float(row[2])
            if allowed is not None and (chamber, side) not in allowed:
                continue
            rows.append((chamber, side, process_time))
        if not rows:
            rows = [
                (int(row[0]), int(row[1]), float(row[2]))
                for row in arr
            ]
        stage_options.append(rows)
    return stage_options


def _free_init_from_calendar(calendar_state, machine, stage_options):
    free = {}
    chamber_calendar = calendar_state.chamber_calendar
    for options in stage_options:
        for chamber, side, _process_time in options:
            key = (int(machine), int(chamber), int(side))
            intervals = chamber_calendar.get(key, [])
            free[key] = float(intervals[-1][1]) if intervals else 0.0
    return free


def _subbatch_wafer_map(sub_batches):
    mapping = []
    cursor = 0
    for batch_size in sub_batches:
        mapping.append(list(range(cursor + 1, cursor + int(batch_size) + 1)))
        cursor += int(batch_size)
    return mapping


def _stage_times(stage_options, n_batches, lot, machine, ppid, encoder, noise_rng):
    n_stages = len(stage_options)
    base_mu = np.array(
        [min(process_time for _chamber, _side, process_time in options)
         for options in stage_options],
        dtype=float,
    )
    if noise_rng is None:
        return np.tile(base_mu, (n_batches, 1))

    times = np.empty((n_batches, n_stages), dtype=float)
    for b in range(n_batches):
        for s in range(n_stages):
            sigma = _stage_sigma(encoder, lot, machine, ppid, s + 1)
            delta = float(noise_rng.normal(0.0, sigma)) if sigma > 0.0 else 0.0
            times[b, s] = max(1e-6, float(base_mu[s]) + delta)
    return times


def schedule_on_calendar(
    lot,
    machine,
    ppid,
    encoder,
    calendar_state,
    earliest_release,
    noise_rng=None,
):
    """Schedule a lot against committed calendar state without mutating it."""
    lot, machine, ppid = int(lot), int(machine), int(ppid)

    try:
        steps = encoder.get_process_steps(lot, machine, ppid)
    except (KeyError, ValueError):
        return ScheduleResult(infeasible_reason="ppid_stage_missing")
    if not steps:
        return ScheduleResult(infeasible_reason="ppid_stage_missing")

    stage_options = _build_stage_options(encoder, machine, steps)
    if any(len(options) == 0 for options in stage_options):
        return ScheduleResult(infeasible_reason="chamber_side_unavailable")

    wafer_count = int(encoder.wafer_counts[lot])
    side_capacity = getattr(encoder, "side_capacity", None)
    if side_capacity is None or int(side_capacity) <= 0:
        side_capacity = wafer_count
    sub_batches = compute_sub_batches(wafer_count, int(side_capacity))
    n_batches = len(sub_batches)
    times = _stage_times(
        stage_options, n_batches, lot, machine, ppid, encoder, noise_rng
    )
    free_init = _free_init_from_calendar(calendar_state, machine, stage_options)

    machine_calendar = calendar_state.machine_calendar
    lot_release_time = encoder.find_earliest_slot(
        machine_calendar.get(machine, []),
        float(earliest_release),
        0.0,
    )

    for _ in range(20):
        lot_start, lot_end, intervals = schedule_deterministic(
            sub_batches,
            times,
            stage_options,
            machine,
            instance_free_init=free_init,
            lot_release_time=lot_release_time,
        )
        lot_duration = max(0.0, float(lot_end) - float(lot_release_time))
        machine_slot_start = encoder.find_earliest_slot(
            machine_calendar.get(machine, []),
            float(earliest_release),
            lot_duration,
        )
        if abs(machine_slot_start - lot_release_time) <= 1e-9:
            batch_intervals = [
                (resource_key, start, end)
                for _b, _stage, resource_key, start, end in intervals
            ]
            return ScheduleResult(
                lot_start=float(lot_start),
                lot_end=float(lot_end),
                batch_intervals=batch_intervals,
                machine_interval=(machine, float(lot_release_time), float(lot_end)),
                subbatch_wafer_map=_subbatch_wafer_map(sub_batches),
                infeasible_reason="",
            )
        lot_release_time = machine_slot_start

    return ScheduleResult(infeasible_reason="calendar_no_stable_slot")
