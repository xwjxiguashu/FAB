"""Phase 2 SAS 推理演示 — 用训练好的策略网络运行贪心 episode 并验证调度。

run_demo_episode():
  1. 复用 build_training_components() 构建策略网络 + 环境
  2. 运行贪心 episode (greedy policy + 失败回退)
  3. 验证部分调度 (partial=True，允许未完成 Lot)
  4. 返回摘要 (含验证结果和冲突数)
"""

from pathlib import Path
import sys

FABENV_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = FABENV_ROOT / "scripts"
for path in (
    FABENV_ROOT,
    SCRIPT_ROOT / "run",
    SCRIPT_ROOT / "evaluation",
    SCRIPT_ROOT / "experiments",
    SCRIPT_ROOT / "probes",
):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from train_phase2_sas_ppo import (
    build_multihead_training_components,
    build_training_components,
)


def run_demo_episode(max_steps=1000, checkpoint_path=None):
    """运行贪心策略推理并验证调度。

    Args:
        max_steps: 最大步数限制。
        checkpoint_path: 可选 .pt checkpoint 路径。
            - None: 用新建 (未训练) 策略 (保持原行为)。
            - 非空: 加载训练好的策略；env/driver 仍由 build_*_components 构建，
              但 policy 替换为加载的。根据 checkpoint 的 policy_type 选择
              单头 (build_training_components) 或多头 (build_multihead_training_components)
              的环境，以保证观察维度匹配。

    Returns:
        summary: 包含 episode 统计 + 验证结果。
    """
    if checkpoint_path:
        from model_checkpoint import load_policy_checkpoint

        loaded_policy, ckpt = load_policy_checkpoint(checkpoint_path)
        if ckpt.get("policy_type") == "multihead":
            components = build_multihead_training_components(lookahead=False)
            # 多头 driver 持有 RewardVectorConfig，而 run_greedy_episode 经
            # env.sas_step 使用标量奖励路径；推理时改用标量 RewardConfig，
            # 仅用于汇总 episode_reward (不参与训练)。
            from rl_environment import RewardConfig

            components["driver"].reward_config = RewardConfig()
        else:
            components = build_training_components()
        policy = loaded_policy
    else:
        components = build_training_components()
        policy = components["policy"]

    driver = components["driver"]
    driver.max_steps = int(max_steps)

    # 贪心 episode
    summary = driver.run_greedy_episode(policy)

    # 部分调度验证 (允许未完成 Lot)
    validation = components["env"].validate_schedule(partial=True)
    summary["validation_passed"] = bool(validation.passed)
    summary["machine_conflicts"] = validation.machine_conflicts
    summary["chamber_conflicts"] = validation.chamber_conflicts
    summary["lot_schedule_rows"] = validation.lot_schedule_rows
    summary["wafer_schedule_rows"] = validation.wafer_schedule_rows
    summary["validation_errors"] = validation.errors
    return summary


def main():
    """运行推理演示并输出摘要。"""
    summary = run_demo_episode()
    print(summary)
    return summary


if __name__ == "__main__":
    main()