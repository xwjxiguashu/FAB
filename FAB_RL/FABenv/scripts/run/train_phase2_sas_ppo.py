"""Phase 2 SAS PPO training entry point."""

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

import argparse

import torch

from phase2_ppo_buffer import MULTIHEAD_CHANNELS, Phase2RolloutBuffer
from phase2_ppo_trainer import (
    MultiHeadPPOConfig,
    MultiHeadPPOTrainer,
    PPOConfig,
    Phase2PPOTrainer,
)
from phase2_sas_driver import Phase2EpisodeDriver
from phase2_sas_observation import Phase2ObservationEncoder
from phase2_sas_policy import Phase2SASActorCritic, Phase2SASMultiHeadActorCritic
from problem_generator import build_random_encoder, sample_random_problem_config
from problem_instances import (
    build_late_hi_encoder,
    build_pressure_test_encoder,
    build_small_encoder,
)
from rl_environment import ResourceCalendarEnv, RewardConfig, RewardVectorConfig
from training_logger import TensorBoardTrainingLogger


def resolve_device(device=None):
    """解析训练设备。

    - None / "auto": 有 CUDA 显卡则用 "cuda"，否则回退 "cpu"。
    - "cuda": 强制用 GPU；若不可用则报错提示 (避免静默退化)。
    - "cpu": 强制用 CPU。

    注意 (诚实告知)：本项目训练瓶颈在环境仿真 (候选池 dry-run / commit，纯 CPU)，
    神经网络很小，GPU 对端到端训练速度提升有限，主要价值在网络较大或批量较大时。
    """
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "请求 --device cuda 但 torch.cuda.is_available()=False；"
            "当前 torch 可能是 CPU 版 (torch.__version__ 含 +cpu)。"
            "请安装 CUDA 版 torch，或用 --device auto/cpu。"
        )
    return torch.device(device)


def build_driver_for_encoder(
    encoder,
    top_k=8,
    observation_encoder=None,
    reward_config=None,
    planning_horizon=None,
    max_steps=10000,
):
    env = ResourceCalendarEnv(encoder, top_k=top_k)
    env.reset()
    observation_encoder = observation_encoder or Phase2ObservationEncoder()
    reward_config = reward_config or RewardConfig()
    driver = Phase2EpisodeDriver(
        env,
        observation_encoder,
        reward_config,
        planning_horizon=planning_horizon,
        max_steps=max_steps,
    )
    return env, driver


def build_curriculum_driver_factory(
    top_k=8,
    observation_encoder=None,
    reward_config=None,
    split="train",
    max_steps=10000,
):
    observation_encoder = observation_encoder or Phase2ObservationEncoder()
    reward_config = reward_config or RewardConfig()

    def driver_factory(episode):
        config = sample_random_problem_config(episode, split=split)
        encoder = build_random_encoder(config)
        _, driver = build_driver_for_encoder(
            encoder,
            top_k=top_k,
            observation_encoder=observation_encoder,
            reward_config=reward_config,
            max_steps=max_steps,
        )
        metadata = {
            "seed": int(config.seed),
            "difficulty": config.difficulty,
            "num_lots": int(config.num_lots),
            "num_machines": int(config.num_machines),
        }
        return driver, metadata

    return driver_factory

def build_training_components(
    top_k=8,
    hidden_dim=128,
    learning_rate=3e-4,
    encoder_factory=None,
    device=None,
):
    device = resolve_device(device)
    encoder = build_small_encoder() if encoder_factory is None else encoder_factory()
    env = ResourceCalendarEnv(encoder, top_k=top_k)
    env.reset()

    observation_encoder = Phase2ObservationEncoder()
    reward_config = RewardConfig()

    sample_machine = env.get_candidate_machines()[0]
    sample_pool = env.build_candidate_pool(sample_machine)
    sample_observation = observation_encoder.encode(sample_machine, sample_pool, env)

    policy = Phase2SASActorCritic(
        candidate_dim=sample_observation.candidate_features.shape[1],
        global_dim=sample_observation.global_features.shape[0],
        hidden_dim=hidden_dim,
    ).to(device)

    config = PPOConfig(learning_rate=learning_rate)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    trainer = Phase2PPOTrainer(policy, optimizer, config)

    driver = Phase2EpisodeDriver(env, observation_encoder, reward_config)
    buffer = Phase2RolloutBuffer(gamma=config.gamma, gae_lambda=config.gae_lambda)

    return {
        "encoder": encoder,
        "env": env,
        "observation_encoder": observation_encoder,
        "reward_config": reward_config,
        "policy": policy,
        "optimizer": optimizer,
        "trainer": trainer,
        "driver": driver,
        "buffer": buffer,
        "candidate_dim": int(sample_observation.candidate_features.shape[1]),
        "global_dim": int(sample_observation.global_features.shape[0]),
        "hidden_dim": int(hidden_dim),
        "policy_type": "single",
        "channels": None,
        "device": str(device),
    }


def build_multihead_training_components(top_k=8, hidden_dim=128, learning_rate=3e-4,
                                        encoder_factory=None, lookahead=False, w_lookahead=0.0,
                                        process_noise_enabled=False, noise_seed=None,
                                        use_qtime_lagrangian=False, qtime_cost_budget=0.0,
                                        qtime_lambda_lr=0.05, qtime_lambda_init=0.0,
                                        priority_filter_mode="soft", priority_min_gap=0.0,
                                        device=None):
    device = resolve_device(device)
    encoder = build_small_encoder() if encoder_factory is None else encoder_factory()
    env = ResourceCalendarEnv(encoder, top_k=top_k, w_lookahead=w_lookahead,
                              process_noise_enabled=process_noise_enabled, noise_seed=noise_seed,
                              priority_filter_mode=priority_filter_mode,
                              priority_min_gap=priority_min_gap)
    env.reset()
    observation_encoder = Phase2ObservationEncoder(lookahead=lookahead)
    reward_vector_config = RewardVectorConfig()
    sample_machine = env.get_candidate_machines()[0]
    sample_pool = env.build_candidate_pool(sample_machine)
    sample_obs = observation_encoder.encode(sample_machine, sample_pool, env)
    policy = Phase2SASMultiHeadActorCritic(
        candidate_dim=sample_obs.candidate_features.shape[1],
        global_dim=sample_obs.global_features.shape[0],
        hidden_dim=hidden_dim, channels=MULTIHEAD_CHANNELS).to(device)
    config = MultiHeadPPOConfig(
        learning_rate=learning_rate,
        use_qtime_lagrangian=use_qtime_lagrangian,
        qtime_cost_budget=qtime_cost_budget,
        qtime_lambda_lr=qtime_lambda_lr,
        qtime_lambda_init=qtime_lambda_init,
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    trainer = MultiHeadPPOTrainer(policy, optimizer, config)
    driver = Phase2EpisodeDriver(env, observation_encoder, reward_vector_config)
    return {
        "encoder": encoder, "env": env, "observation_encoder": observation_encoder,
        "reward_vector_config": reward_vector_config, "policy": policy,
        "optimizer": optimizer, "trainer": trainer, "driver": driver,
        "candidate_dim": int(sample_obs.candidate_features.shape[1]),
        "global_dim": int(sample_obs.global_features.shape[0]),
        "hidden_dim": int(hidden_dim),
        "policy_type": "multihead",
        "channels": MULTIHEAD_CHANNELS,
        "device": str(device),
    }


def _run_training(components, num_episodes, mode, episode_logger=None, on_episode=None):
    if mode == "small" or mode == "pressure":
        return components["trainer"].train(
            components["driver"],
            num_episodes=num_episodes,
            episode_logger=episode_logger,
        )
    if mode == "multihead":
        return components["trainer"].train(
            components["driver"],
            num_episodes=num_episodes,
            episode_logger=episode_logger,
            reward_vector_config=components["reward_vector_config"],
            on_episode=on_episode,
        )
    if mode == "random":
        driver_factory = build_curriculum_driver_factory(
            top_k=int(components["env"].top_k),
            observation_encoder=components["observation_encoder"],
            reward_config=components["reward_config"],
            split="train",
        )
        return components["trainer"].train_with_driver_factory(
            driver_factory,
            num_episodes=num_episodes,
            episode_logger=episode_logger,
        )
    raise ValueError("mode must be 'small', 'random', 'pressure', or 'multihead'")


def _run_multihead_parallel(components, num_episodes, n_workers, encoder_kind,
                            episode_logger=None, on_episode=None,
                            process_noise_enabled=False, noise_seed_base=0):
    """多进程并行 multihead 训练 (报告 §7.1 多 worker 扩展)。

    每个 iter: 广播权重 → N 个 worker 各跑 1 个 episode → steps 顺序拼进一个 buffer
    (GAE 在 done 处自动归零，跨 episode 不串味) → 一次 PPO 更新 → λ 对偶上升用
    **N 个 episode 的平均违规率** (Ê[violation] 估计更稳，正是报告 §3.3 想要的)。
    """
    import math

    from phase2_ppo_buffer import MultiHeadRolloutBuffer
    from parallel_rollout import ParallelRolloutCollector, make_spec

    trainer = components["trainer"]
    policy = components["policy"]
    channels = tuple(components["channels"])
    spec = make_spec(
        encoder_kind=encoder_kind,
        candidate_dim=components["candidate_dim"],
        global_dim=components["global_dim"],
        hidden_dim=components["hidden_dim"],
        channels=channels,
        top_k=int(getattr(components["env"], "top_k", 8)),
        process_noise_enabled=process_noise_enabled,
        noise_seed_base=noise_seed_base,
        priority_filter_mode=getattr(components["env"], "priority_filter_mode", "soft"),
        priority_min_gap=getattr(components["env"], "priority_min_gap", 0.0),
    )
    import time as _time

    n_iters = max(1, math.ceil(int(num_episodes) / int(n_workers)))
    history = []
    episode_idx = 0
    _t_pool = _time.time()
    with ParallelRolloutCollector(n_workers, spec) as collector:
        print(f"[parallel] pool ready in {_time.time()-_t_pool:.1f}s", flush=True)
        for it in range(n_iters):
            _t_it = _time.time()
            results = collector.collect(policy.state_dict())
            _collect_dt = _time.time() - _t_it
            buffer = MultiHeadRolloutBuffer(
                gamma=trainer.config.gamma, gae_lambda=trainer.config.gae_lambda,
                channels=channels,
            )
            summaries = []
            for steps, summary in results:
                for s in steps:
                    buffer.add(s)
                summaries.append(summary)
            if not buffer.steps:
                continue
            buffer.finish_episode(last_values={c: 0.0 for c in channels})
            total_cost = trainer.episode_qtime_cost(buffer)
            stats = trainer.update_policy(buffer)
            trainer.update_lambda(total_cost / max(1, len(results)))  # 平均违规率

            mean_reward = sum(float(s["episode_reward"]) for s in summaries) / len(summaries)
            mean_completed = sum(float(s["completed_lots"]) for s in summaries) / len(summaries)
            # 原始(未归一化)指标 —— 真实学习曲线看这两个，而非被常数通道淹没的 mean_reward
            mean_util = sum(float(s.get("avg_utilization", 0.0)) for s in summaries) / len(summaries)
            mean_qtime = sum(float(s.get("qtime_violation_count", 0.0)) for s in summaries) / len(summaries)
            for summary in summaries:
                row = {
                    "episode": episode_idx, "iter": it,
                    "episode_reward": float(summary["episode_reward"]),
                    "completed_lots": int(summary["completed_lots"]),
                    "avg_utilization": float(summary.get("avg_utilization", 0.0)),
                    "qtime_violation_count": float(summary.get("qtime_violation_count", 0.0)),
                    "termination_reason": summary["termination_reason"],
                    **stats,
                    "qtime_cost": float(total_cost / max(1, len(results))),
                    "lambda_qtime": float(trainer.lambda_qtime),
                }
                history.append(row)
                if episode_logger is not None:
                    episode_logger.log(row)
                episode_idx += 1
            print(f"[parallel iter {it+1}/{n_iters}] {len(results)} eps  "
                  f"collect={_collect_dt:.1f}s ({_collect_dt/max(1,len(results)):.1f}s/ep) "
                  f"mean_reward={mean_reward:.3f} util={mean_util:.3f} qtime={mean_qtime:.1f} "
                  f"completed={mean_completed:.1f} "
                  f"loss={stats['policy_loss']:.4f} lambda={trainer.lambda_qtime:.4f}",
                  flush=True)
            if on_episode is not None:
                on_episode(episode_idx - 1, history[-1])
    return history


def main(num_episodes=3, mode="small", tensorboard_logdir=None, save_path=None,
         use_qtime_lagrangian=False, qtime_cost_budget=0.0, qtime_lambda_lr=0.05,
         device=None, instance="small", save_every=0, parallel=0,
         priority_filter_mode="soft", priority_min_gap=0.0):
    if mode == "multihead":
        mh_encoder_factory = {
            "pressure": build_pressure_test_encoder,
            "late_hi": build_late_hi_encoder,
        }.get(instance)
        components = build_multihead_training_components(
            encoder_factory=mh_encoder_factory,
            use_qtime_lagrangian=use_qtime_lagrangian,
            qtime_cost_budget=qtime_cost_budget,
            qtime_lambda_lr=qtime_lambda_lr,
            priority_filter_mode=priority_filter_mode,
            priority_min_gap=priority_min_gap,
            device=device,
        )
    elif mode == "pressure":
        components = build_training_components(
            encoder_factory=build_pressure_test_encoder, device=device,
        )
    else:
        components = build_training_components(device=device)
    print(f"[train] mode={mode} instance={instance} device={components.get('device')} "
          f"(cuda_available={torch.cuda.is_available()})")

    def _save_checkpoint():
        from model_checkpoint import save_policy_checkpoint
        policy_type = "multihead" if mode == "multihead" else "single"
        save_policy_checkpoint(
            components["policy"], save_path,
            candidate_dim=components["candidate_dim"],
            global_dim=components["global_dim"],
            hidden_dim=components["hidden_dim"],
            policy_type=policy_type, channels=components.get("channels"),
            metadata={"mode": mode, "instance": instance,
                      "num_episodes": int(num_episodes),
                      "top_k": int(getattr(components["env"], "top_k", 8))},
        )

    # 周期性保存：长训练被中断 (如 10 分钟工具上限) 仍留有最新模型。
    on_episode = None
    if save_path and save_every and int(save_every) > 0:
        def on_episode(ep, row):
            if (ep + 1) % int(save_every) == 0:
                _save_checkpoint()
                print(f"[ckpt] saved at episode {ep + 1}: util/q via history; "
                      f"util_row={row.get('episode_reward'):.3f}", flush=True)

    use_parallel = mode == "multihead" and int(parallel) > 1

    def _run(episode_logger=None):
        if use_parallel:
            print(f"[parallel] {parallel} worker 进程 (CPU env) 并行采集", flush=True)
            return _run_multihead_parallel(
                components, num_episodes, int(parallel), instance,
                episode_logger=episode_logger, on_episode=on_episode,
            )
        return _run_training(components, num_episodes, mode,
                             episode_logger=episode_logger, on_episode=on_episode)

    if tensorboard_logdir:
        with TensorBoardTrainingLogger(tensorboard_logdir) as episode_logger:
            history = _run(episode_logger=episode_logger)
    else:
        history = _run()
    print(history[-3:] if len(history) > 3 else history)
    if save_path:
        _save_checkpoint()
        print(f"Saved policy checkpoint to: {save_path}")
    return history


def _parse_args():
    parser = argparse.ArgumentParser(description="Train Phase 2 SAS PPO")
    parser.add_argument(
        "--mode",
        choices=["small", "random", "pressure", "multihead"],
        default="small",
    )
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument(
        "--tensorboard-logdir",
        default=None,
        help="Optional TensorBoard log directory for live training metrics",
    )
    parser.add_argument(
        "--save-path",
        default=None,
        help="Optional path (.pt) to save the trained policy checkpoint",
    )
    parser.add_argument(
        "--qtime-lagrangian",
        action="store_true",
        help="(multihead only) 用 PPO-Lagrangian 自适应 λ 处理 Q-time 残差约束 (报告 §3.3)，"
             "取代固定 w_qtime",
    )
    parser.add_argument(
        "--qtime-budget", type=float, default=0.0,
        help="ε —— 可容忍的期望 Q-time 违规率 (PPO-Lagrangian 的约束阈值)",
    )
    parser.add_argument(
        "--qtime-lambda-lr", type=float, default=0.05,
        help="η_λ —— 拉格朗日乘子对偶上升步长 (须远小于策略学习率)",
    )
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto",
        help="训练设备：auto=有显卡用 CUDA 否则 CPU；cuda=强制 GPU；cpu=强制 CPU",
    )
    parser.add_argument(
        "--instance", choices=["small", "pressure", "late_hi"], default="small",
        help="(multihead only) 训练实例：small(4 lots)、pressure(50 lots 随机优先级) "
             "或 late_hi(50 lots, 高优先级晚到, 与到达高度正相关)",
    )
    parser.add_argument(
        "--save-every", type=int, default=0,
        help="每 N 个 episode 保存一次检查点 (需配合 --save-path)；长训练被中断时留有最新模型",
    )
    parser.add_argument(
        "--parallel", type=int, default=0,
        help="(multihead only) 多进程并行环境的 worker 数 (>1 启用)；用多核 CPU 加速采集",
    )
    parser.add_argument(
        "--priority-mode", choices=["soft", "strict"], default="soft",
        help="候选池优先级过滤模式 (报告 §3.4)：soft=不删候选(默认); "
             "strict=只保留最高优先级候选，RL 物理上无法选低优先级",
    )
    parser.add_argument(
        "--priority-min-gap", type=float, default=0.0,
        help="strict 模式下的优先级容差 (priority >= max_pri - gap 的候选保留)",
    )
    return parser.parse_args()


def _run_cli():
    args = _parse_args()
    main(
        num_episodes=args.episodes,
        mode=args.mode,
        tensorboard_logdir=args.tensorboard_logdir,
        save_path=args.save_path,
        use_qtime_lagrangian=args.qtime_lagrangian,
        qtime_cost_budget=args.qtime_budget,
        qtime_lambda_lr=args.qtime_lambda_lr,
        device=args.device,
        instance=args.instance,
        save_every=args.save_every,
        parallel=args.parallel,
        priority_filter_mode=args.priority_mode,
        priority_min_gap=args.priority_min_gap,
    )


def _run_default():
    """无命令行参数时的默认运行 (双击 / 直接 python train_phase2_sas_ppo.py)。

    改这里的参数即可调整无参运行行为。带参数运行时走 _run_cli() (见下方 __main__)。
    save_path 锚定到本文件目录：VSCode 的 cwd 常是工作区根，相对路径会存错地方。
    """
    import os as _os
    _artifact_dir = Path(__file__).resolve().parents[2] / "artifacts" / "checkpoints"
    _artifact_dir.mkdir(parents=True, exist_ok=True)

    # 拉满核心：worker = 逻辑核 - 2（留给主进程/系统）。
    # 关键：PPO 更新次数 = episodes / parallel；为保持训练质量，episodes 随 worker 数放大，
    # 使更新次数固定为 _ITERS（≈50）。wall time 几乎不变，但每次更新用的数据更多。
    _WORKERS = max(1, (_os.cpu_count() or 4) - 2)
    _ITERS = 50
    main(
        mode="multihead",
        instance="pressure",
        num_episodes=_WORKERS * _ITERS,   # 例: 14 worker → 700 episode → 仍 ~50 次更新
        parallel=_WORKERS,                # 启动时各 worker import torch 会静默几十秒，勿中断
        save_every=_WORKERS * 5,          # 约每 5 次迭代存一次 checkpoint
        save_path=_os.path.join(str(_artifact_dir), "pressure_mh_hard.pt"),
        device="auto",                    # 瓶颈在 CPU 仿真，GPU 无加速
    )


if __name__ == "__main__":
    # CLI 优先：有命令行参数 → 走 _parse_args()/_run_cli() (CLAUDE.md 文档的用法)；
    # 无参数 → 回退到 _run_default() 的硬编码默认 (双击直跑 pressure 训练的便利)。
    # (历史教训：此前 __main__ 无条件硬编码 pressure 且注释掉 _run_cli()，导致
    #  `--instance late_hi` 等 CLI 参数被静默忽略、误覆盖 pressure_mh_hard.pt。)
    import sys as _sys
    if len(_sys.argv) > 1:
        _run_cli()
    else:
        _run_default()
