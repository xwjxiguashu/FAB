"""Phase 2 SAS PPO training entry point."""

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
from problem_instances import build_pressure_test_encoder, build_small_encoder
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
                                        device=None):
    device = resolve_device(device)
    encoder = build_small_encoder() if encoder_factory is None else encoder_factory()
    env = ResourceCalendarEnv(encoder, top_k=top_k, w_lookahead=w_lookahead,
                              process_noise_enabled=process_noise_enabled, noise_seed=noise_seed)
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


def _run_training(components, num_episodes, mode, episode_logger=None):
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


def main(num_episodes=3, mode="small", tensorboard_logdir=None, save_path=None,
         use_qtime_lagrangian=False, qtime_cost_budget=0.0, qtime_lambda_lr=0.05,
         device=None):
    if mode == "multihead":
        components = build_multihead_training_components(
            use_qtime_lagrangian=use_qtime_lagrangian,
            qtime_cost_budget=qtime_cost_budget,
            qtime_lambda_lr=qtime_lambda_lr,
            device=device,
        )
    elif mode == "pressure":
        components = build_training_components(
            encoder_factory=build_pressure_test_encoder, device=device,
        )
    else:
        components = build_training_components(device=device)
    print(f"[train] mode={mode} device={components.get('device')} "
          f"(cuda_available={torch.cuda.is_available()})")
    if tensorboard_logdir:
        with TensorBoardTrainingLogger(tensorboard_logdir) as episode_logger:
            history = _run_training(
                components,
                num_episodes,
                mode,
                episode_logger=episode_logger,
            )
    else:
        history = _run_training(components, num_episodes, mode)
    print(history)
    if save_path:
        from model_checkpoint import save_policy_checkpoint

        policy_type = "multihead" if mode == "multihead" else "single"
        save_policy_checkpoint(
            components["policy"],
            save_path,
            candidate_dim=components["candidate_dim"],
            global_dim=components["global_dim"],
            hidden_dim=components["hidden_dim"],
            policy_type=policy_type,
            channels=components.get("channels"),
            metadata={
                "mode": mode,
                "num_episodes": int(num_episodes),
                "top_k": int(getattr(components["env"], "top_k", 8)),
            },
        )
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
    return parser.parse_args()


if __name__ == "__main__":
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
    )
