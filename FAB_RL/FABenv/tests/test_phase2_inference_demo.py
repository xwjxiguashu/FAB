import sys
from pathlib import Path


PHASE1_DIR = Path(__file__).resolve().parents[1]
if str(PHASE1_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE1_DIR))


import run_phase2_sas_inference_demo
import train_phase2_sas_ppo


def test_training_components_can_be_built():
    components = train_phase2_sas_ppo.build_training_components()

    assert "env" in components
    assert "policy" in components
    assert "trainer" in components
    assert "buffer" in components


def test_training_main_runs_configured_number_of_episodes():
    history = train_phase2_sas_ppo.main(num_episodes=1)

    assert isinstance(history, list)
    assert len(history) == 1
    assert history[0]["steps"] > 0
    assert "termination_reason" in history[0]


def test_training_main_runs_pressure_mode():
    history = train_phase2_sas_ppo.main(num_episodes=1, mode="pressure")

    assert isinstance(history, list)
    assert len(history) == 1
    assert history[0]["steps"] > 0
    assert history[0]["completed_lots"] > 0


def test_inference_demo_runs_and_validates_schedule():
    summary = run_phase2_sas_inference_demo.run_demo_episode(max_steps=200)

    assert "termination_reason" in summary
    assert "validation_passed" in summary
    assert "machine_conflicts" in summary
    assert "chamber_conflicts" in summary
    assert "completed_lots" in summary
    assert summary["steps"] <= 200
    assert summary["completed_lots"] == 4
    assert summary["termination_reason"] == "all_lots_completed"
