"""Training metric loggers for Phase 2 PPO."""


class TensorBoardTrainingLogger:
    """Write per-episode PPO metrics to TensorBoard."""

    SCALAR_TAGS = {
        "episode_reward": "train/episode_reward",
        "completed_lots": "train/completed_lots",
        "steps": "train/steps",
        "wait_steps": "train/wait_steps",
        "failed_actions": "train/failed_actions",
        "consecutive_failed_actions": "train/consecutive_failed_actions",
        "policy_loss": "loss/policy_loss",
        "value_loss": "loss/value_loss",
        "entropy": "loss/entropy",
        "seed": "problem/seed",
        "num_lots": "problem/num_lots",
        "num_machines": "problem/num_machines",
    }

    def __init__(self, log_dir, writer=None):
        self.log_dir = log_dir
        self._closed = False
        if writer is None:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError as exc:
                raise RuntimeError(
                    "TensorBoard logging requires tensorboard. "
                    "Install it or run without --tensorboard-logdir."
                ) from exc
            writer = SummaryWriter(log_dir=log_dir)
        self.writer = writer

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    def log(self, row):
        episode = int(row["episode"])
        for key, tag in self.SCALAR_TAGS.items():
            if key not in row:
                continue
            value = row[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            self.writer.add_scalar(tag, float(value), episode)
        self.writer.flush()

    def close(self):
        if self._closed:
            return
        self.writer.close()
        self._closed = True
