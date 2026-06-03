import numpy as np

from lower_layer_estimator import estimate, schedule_deterministic
from state import ScheduleState


def test_schedule_deterministic_empty_free_single_batch():
    sub_batches = [4]
    stage_times = np.array([[3.0, 5.0]])
    options = [[(1, 1, 3.0)], [(2, 1, 5.0)]]

    lot_start, lot_end, intervals = schedule_deterministic(
        sub_batches,
        stage_times,
        options,
        machine=1,
        instance_free_init={},
        lot_release_time=0.0,
    )

    assert lot_start == 0.0
    assert lot_end == 8.0
    assert intervals == [
        (0, 1, (1, 1, 1), 0.0, 3.0),
        (0, 2, (1, 2, 1), 3.0, 8.0),
    ]


def test_schedule_deterministic_respects_free_init_without_mutating_input():
    sub_batches = [4]
    stage_times = np.array([[3.0, 5.0]])
    options = [[(1, 1, 3.0)], [(2, 1, 5.0)]]
    free = {(1, 1, 1): 10.0}

    lot_start, lot_end, _intervals = schedule_deterministic(
        sub_batches,
        stage_times,
        options,
        machine=1,
        instance_free_init=free,
        lot_release_time=0.0,
    )

    assert lot_start == 10.0
    assert lot_end == 18.0
    assert free == {(1, 1, 1): 10.0}


def test_schedule_deterministic_picks_earliest_instance_and_ties_by_order():
    sub_batches = [4]
    stage_times = np.array([[3.0]])
    options = [[(1, 1, 3.0), (1, 2, 3.0), (2, 1, 3.0)]]
    free = {(1, 1, 1): 5.0, (1, 1, 2): 0.0, (1, 2, 1): 0.0}

    _lot_start, lot_end, intervals = schedule_deterministic(
        sub_batches,
        stage_times,
        options,
        machine=1,
        instance_free_init=free,
        lot_release_time=0.0,
    )

    assert intervals[0][2] == (1, 1, 2)
    assert lot_end == 3.0


def test_schedule_deterministic_keeps_batch_stage_order():
    sub_batches = [2, 2]
    stage_times = np.array([[1.0, 2.0], [3.0, 4.0]])
    options = [[(1, 0, 1.0)], [(2, 0, 2.0)]]

    _lot_start, _lot_end, intervals = schedule_deterministic(
        sub_batches,
        stage_times,
        options,
        machine=7,
        instance_free_init={},
        lot_release_time=0.0,
    )

    assert [(b, stage) for b, stage, _key, _start, _end in intervals] == [
        (0, 1),
        (0, 2),
        (1, 1),
        (1, 2),
    ]


def test_estimate_step6_uses_same_deterministic_core(small_encoder):
    enc = small_encoder
    lot = 1
    machine = int(enc.get_machine_list(lot)[0])
    ppid = int(enc.get_ppid_list(lot, machine)[0])

    res = estimate(lot, machine, ppid, enc, state=None, n_mc=1)

    # The sampled mu_finish may include process sigma; step6 occupancy is the
    # deterministic mean path and should be internally self-consistent.
    max_end = max(end for _key, _start, end in res["per_instance_occupancy"])
    assert max_end > 0.0
    assert len(res["per_instance_occupancy"]) == (
        int(res["n_batches"]) * len(res["stage_mu"])
    )


def test_schedule_on_calendar_matches_estimate_mean_path_on_empty(small_encoder):
    from lower_layer_scheduler import schedule_on_calendar

    enc = small_encoder
    empty = ScheduleState()
    lot = 1
    machine = int(enc.get_machine_list(lot)[0])
    ppid = int(enc.get_ppid_list(lot, machine)[0])

    est = estimate(lot, machine, ppid, enc, state=None, n_mc=1)
    res = schedule_on_calendar(
        lot,
        machine,
        ppid,
        enc,
        empty,
        earliest_release=float(enc.arrival_times[lot]),
        noise_rng=None,
    )

    assert res.infeasible_reason == ""
    makespan = res.lot_end - res.machine_interval[1]
    occupancy_makespan = max(
        end for _key, _start, end in est["per_instance_occupancy"]
    )
    assert abs(makespan - occupancy_makespan) < 1e-6
