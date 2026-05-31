import sys
from pathlib import Path


PHASE1_DIR = Path(__file__).resolve().parents[1]
if str(PHASE1_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE1_DIR))


from training_logger import TensorBoardTrainingLogger


class FakeWriter:
    def __init__(self):
        self.scalars = []
        self.flush_count = 0
        self.close_count = 0

    def add_scalar(self, tag, scalar_value, global_step):
        self.scalars.append((tag, scalar_value, global_step))

    def flush(self):
        self.flush_count += 1

    def close(self):
        self.close_count += 1


def test_tensorboard_logger_writes_training_and_loss_scalars():
    writer = FakeWriter()
    logger = TensorBoardTrainingLogger("unused", writer=writer)

    logger.log(
        {
            "episode": 3,
            "episode_reward": 12.5,
            "completed_lots": 4,
            "steps": 9,
            "wait_steps": 1,
            "failed_actions": 2,
            "consecutive_failed_actions": 0,
            "termination_reason": "all_lots_completed",
            "policy_loss": -0.12,
            "value_loss": 1.5,
            "entropy": 0.7,
        }
    )

    assert ("train/episode_reward", 12.5, 3) in writer.scalars
    assert ("train/completed_lots", 4.0, 3) in writer.scalars
    assert ("train/steps", 9.0, 3) in writer.scalars
    assert ("loss/policy_loss", -0.12, 3) in writer.scalars
    assert ("loss/value_loss", 1.5, 3) in writer.scalars
    assert ("loss/entropy", 0.7, 3) in writer.scalars
    assert not any(tag == "train/termination_reason" for tag, _, _ in writer.scalars)
    assert writer.flush_count == 1


def test_tensorboard_logger_writes_random_problem_metadata():
    writer = FakeWriter()
    logger = TensorBoardTrainingLogger("unused", writer=writer)

    logger.log(
        {
            "episode": 1,
            "seed": 7,
            "difficulty": "easy",
            "num_lots": 12,
            "num_machines": 3,
        }
    )

    assert ("problem/seed", 7.0, 1) in writer.scalars
    assert ("problem/num_lots", 12.0, 1) in writer.scalars
    assert ("problem/num_machines", 3.0, 1) in writer.scalars
    assert not any(tag == "problem/difficulty" for tag, _, _ in writer.scalars)
    assert writer.flush_count == 1


def test_tensorboard_logger_close_is_idempotent():
    writer = FakeWriter()
    logger = TensorBoardTrainingLogger("unused", writer=writer)

    logger.close()
    logger.close()

    assert writer.close_count == 1
