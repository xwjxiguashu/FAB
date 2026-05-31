import sys
from pathlib import Path


PHASE1_DIR = Path(__file__).resolve().parents[1]
if str(PHASE1_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE1_DIR))


import __init__ as fabenv


def test_phase2_public_exports_exist():
    for name in (
        "Phase2EpisodeDriver",
        "Phase2ObservationEncoder",
        "Phase2SASActorCritic",
        "Phase2RolloutBuffer",
        "Phase2PPOTrainer",
        "RandomProblemConfig",
        "build_random_encoder",
        "sample_random_problem_config",
    ):
        assert hasattr(fabenv, name)
