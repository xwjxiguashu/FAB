import sys
from pathlib import Path


PHASE1_DIR = Path(__file__).resolve().parents[1]
if str(PHASE1_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE1_DIR))


from problem_instances import build_pressure_test_encoder
import run_phase1_environment_demo


def test_pressure_test_encoder_has_requested_scale():
    encoder = build_pressure_test_encoder()

    assert encoder.num_lots == 50
    assert encoder.num_machines == 10
    assert set(encoder.wafer_counts.values()) == {10}

    for lot in range(1, 51):
        assert list(encoder.feasible_machines[lot]) == list(range(1, 11))
        assert encoder.arrival_times[lot] >= 0.0
        assert encoder.due_dates[lot] > encoder.arrival_times[lot]
        assert lot in encoder.priorities

        for machine in range(1, 11):
            ppids = encoder.feasible_ppids[(lot, machine)]
            assert len(ppids) == 5
            assert len(set(ppids)) == 5

            for ppid in ppids:
                steps = encoder.ppid_steps[(lot, machine, ppid)]
                assert len(steps) == 3
                for stage in steps:
                    assert stage.shape[1] == 3
                    assert stage.shape[0] >= 2
                    assert (stage[:, 2] > 0.0).all()

    assert encoder.validate_problem_definition() is True


def test_phase1_demo_uses_pressure_test_encoder():
    encoder = run_phase1_environment_demo.build_demo_encoder()

    assert encoder.num_lots == 50
    assert encoder.num_machines == 10
    assert set(encoder.wafer_counts.values()) == {10}
    assert len(encoder.feasible_ppids[(1, 1)]) == 5


def test_phase1_demo_pressure_run_exercises_all_machines():
    encoder = run_phase1_environment_demo.build_demo_encoder()
    lot_schedule, wafer_schedule, _objectives = (
        run_phase1_environment_demo.run_demo_schedule(encoder=encoder, verbose=False)
    )

    assert lot_schedule.shape == (50, 5)
    assert wafer_schedule.shape == (1500, 9)
    assert set(lot_schedule[:, 1].astype(int)) == set(range(1, 11))
    assert encoder.validate_final_schedule_completeness(lot_schedule, wafer_schedule) is True


def test_phase1_demo_writes_pressure_output_files():
    encoder = run_phase1_environment_demo.build_demo_encoder()
    lot_schedule, wafer_schedule, objectives = (
        run_phase1_environment_demo.run_demo_schedule(encoder=encoder, verbose=False)
    )

    output_paths = run_phase1_environment_demo.export_pressure_outputs(
        encoder,
        lot_schedule,
        wafer_schedule,
        objectives,
    )

    assert output_paths["output_dir"].name == "pressure_outputs"
    assert output_paths["output_dir"].parent == PHASE1_DIR
    assert output_paths["lot_schedule_csv"].is_file()
    assert output_paths["wafer_schedule_csv"].is_file()
    assert output_paths["summary_txt"].is_file()
    assert output_paths["lot_gantt_png"].is_file()
    assert output_paths["wafer_gantt_png"].is_file()
    assert output_paths["lot_gantt_png"].read_bytes().startswith(b"\x89PNG")
    assert output_paths["wafer_gantt_png"].read_bytes().startswith(b"\x89PNG")
    assert "lots=50" in output_paths["summary_txt"].read_text(encoding="utf-8")
    assert "wafer_rows=1500" in output_paths["summary_txt"].read_text(encoding="utf-8")
