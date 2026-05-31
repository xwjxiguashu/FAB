# FABenv — 半导体晶圆厂(FAB)机台组调度强化学习环境
#
# 本包提供两层代码:
#   Phase 1: 基于日历的资源调度环境和启发式派工规则
#   Phase 2: SAS(单智能体调度) PPO 强化学习策略网络与训练管线

import sys
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent
if str(_PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_DIR))

# Phase 1 — 问题定义与 RL 环境
from problem_generator import (
    RandomProblemConfig,
    build_easy_config,
    build_hard_config,
    build_medium_config,
    build_random_encoder,
    sample_random_problem_config,
)
from problem_instances import (
    Phase1CalendarProblem,
    build_pressure_test_encoder,
    build_small_encoder,
)
from rl_environment import (
    CandidatePool,
    DispatchAction,
    DispatchCommitResult,
    DryRunResult,
    MaskResult,
    RewardConfig,
    ResourceCalendarEnv,
    RollbackResult,
    SASStepResult,
    ValidationReport,
    compute_sas_reward,
)
from state import ScheduleState

# Phase 2 — SAS PPO 强化学习管线
from phase2_ppo_buffer import Phase2RolloutBuffer, RolloutStep, StepInfo
from phase2_ppo_trainer import PPOConfig, Phase2PPOTrainer
from phase2_sas_driver import Phase2DispatchDecision, Phase2EpisodeDriver
from phase2_sas_observation import Phase2Observation, Phase2ObservationEncoder
from phase2_sas_policy import MaskedCategoricalPolicy, Phase2SASActorCritic

__all__ = [
    # Phase 1
    "Phase1CalendarProblem",
    "build_pressure_test_encoder",
    "build_small_encoder",
    "ScheduleState",
    "ResourceCalendarEnv",
    "DispatchAction",
    "CandidatePool",
    "DispatchCommitResult",
    "DryRunResult",
    "MaskResult",
    "RollbackResult",
    "ValidationReport",
    "RewardConfig",
    "SASStepResult",
    "compute_sas_reward",
    # Phase 1 random problem generation
    "RandomProblemConfig",
    "build_easy_config",
    "build_medium_config",
    "build_hard_config",
    "build_random_encoder",
    "sample_random_problem_config",
    # Phase 2
    "Phase2RolloutBuffer",
    "RolloutStep",
    "StepInfo",
    "PPOConfig",
    "Phase2PPOTrainer",
    "Phase2DispatchDecision",
    "Phase2EpisodeDriver",
    "Phase2Observation",
    "Phase2ObservationEncoder",
    "MaskedCategoricalPolicy",
    "Phase2SASActorCritic",
]