"""多进程并行环境 rollout 采集 (PPO 训练提速)。

瓶颈是 CPU 环境仿真 (候选池 dry-run / commit)、网络极小 → GPU 无益、多核才有用。
本模块用 N 个常驻 worker 进程各跑一个 episode，主进程聚合后做一次 PPO 更新。

正确性关键: PPO 在 episode 末步 done=True，GAE 递推 gae=δ+γλ(1-done)·gae 在 done 处
自动归零、不跨 episode 串味。故把 N 个 episode 的 steps **顺序拼进同一个 buffer** 即可，
无需改 update 逻辑 (见 MultiHeadRolloutBuffer.compute_returns_and_advantages)。

worker 一律用 CPU (env-bound；避免 CUDA + spawn 在子进程的复杂度)。
"""

import io
import multiprocessing as mp
import os

# 每个 worker 进程限制为单线程 BLAS/OpenMP —— 否则 N 个进程各自起满核线程会
# 严重超额订阅 (oversubscription) 导致比串行还慢。须在 import torch/numpy 之前设。
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import torch

from phase2_ppo_buffer import MULTIHEAD_CHANNELS, MultiHeadRolloutStep
from phase2_sas_driver import Phase2EpisodeDriver
from phase2_sas_observation import Phase2ObservationEncoder
from phase2_sas_policy import Phase2SASMultiHeadActorCritic
from problem_instances import build_pressure_test_encoder, build_small_encoder
from rl_environment import ResourceCalendarEnv, RewardConfig, RewardVectorConfig


ENCODER_FACTORIES = {
    "small": build_small_encoder,
    "pressure": build_pressure_test_encoder,
}

# 每个 worker 进程的常驻对象 (spawn 后由 _init_worker 填充)。
_WORKER = {}


def _state_dict_to_bytes(state_dict):
    """把策略 state_dict 序列化为 bytes (CPU)，便于跨进程传输。"""
    buf = io.BytesIO()
    cpu_sd = {k: v.detach().cpu() for k, v in state_dict.items()}
    torch.save(cpu_sd, buf)
    return buf.getvalue()


def _init_worker(spec):
    """worker 进程初始化: 构建 env / driver / policy (CPU) 并常驻。"""
    torch.set_num_threads(1)  # 单线程，避免多 worker 线程超额订阅 (见模块顶部说明)
    encoder = ENCODER_FACTORIES[spec["encoder_kind"]]()
    # 每个 worker 用不同噪声 seed → 多场景采样 (报告 §2.4.6)
    worker_id = 0
    ident = getattr(mp.current_process(), "_identity", None)
    if ident:
        worker_id = int(ident[0])
    noise_seed = None
    if spec["process_noise_enabled"]:
        noise_seed = int(spec["noise_seed_base"]) + worker_id

    env = ResourceCalendarEnv(
        encoder,
        top_k=spec["top_k"],
        w_lookahead=spec["w_lookahead"],
        process_noise_enabled=spec["process_noise_enabled"],
        noise_seed=noise_seed,
    )
    env.reset()
    obs_enc = Phase2ObservationEncoder(lookahead=spec["lookahead"])
    policy = Phase2SASMultiHeadActorCritic(
        candidate_dim=spec["candidate_dim"],
        global_dim=spec["global_dim"],
        hidden_dim=spec["hidden_dim"],
        channels=tuple(spec["channels"]),
    )
    policy.eval()
    _WORKER.update({
        "env": env,
        "driver": Phase2EpisodeDriver(env, obs_enc, RewardConfig()),
        "policy": policy,
        "rvc": RewardVectorConfig(),
        "channels": tuple(spec["channels"]),
    })


def _lean_steps(steps):
    """剥掉 next_observation / info (更新不需要)，减小回传 payload。"""
    for s in steps:
        s.next_observation = None
        s.info = None
    return steps


def _run_episode(state_bytes):
    """worker: 载入最新权重，跑一个 multihead episode，回传 (steps, summary)。"""
    from phase2_ppo_buffer import MultiHeadRolloutBuffer

    policy = _WORKER["policy"]
    policy.load_state_dict(torch.load(io.BytesIO(state_bytes), map_location="cpu"))
    driver = _WORKER["driver"]
    buf = MultiHeadRolloutBuffer(channels=_WORKER["channels"])
    driver.reset_episode()
    with torch.no_grad():
        summary = driver.run_multihead_policy_episode(
            policy, buffer=buf, stochastic=True,
            reward_vector_config=_WORKER["rvc"],
        )
    return _lean_steps(buf.steps), summary


class ParallelRolloutCollector:
    """常驻 N 个 worker 进程，每次 collect 并行采集 N 个 episode。

    用法:
        spec = make_spec(...)
        collector = ParallelRolloutCollector(n_workers, spec)
        steps_and_summaries = collector.collect(policy.state_dict())  # 长度 = n_workers
        collector.close()
    """

    def __init__(self, n_workers, spec):
        self.n_workers = int(n_workers)
        ctx = mp.get_context("spawn")  # Windows 默认；显式指定保证一致
        self.pool = ctx.Pool(
            processes=self.n_workers,
            initializer=_init_worker,
            initargs=(spec,),
        )

    def collect(self, state_dict):
        """广播权重并并行采集 n_workers 个 episode。

        Returns:
            list[(steps, summary)]，长度 = n_workers。
        """
        state_bytes = _state_dict_to_bytes(state_dict)
        return self.pool.map(_run_episode, [state_bytes] * self.n_workers)

    def close(self):
        self.pool.close()
        self.pool.join()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def make_spec(encoder_kind, candidate_dim, global_dim, hidden_dim=128,
              channels=MULTIHEAD_CHANNELS, top_k=8, lookahead=False, w_lookahead=0.0,
              process_noise_enabled=False, noise_seed_base=0):
    """构造 worker 初始化 spec (须全部可 pickle)。"""
    return {
        "encoder_kind": encoder_kind,
        "candidate_dim": int(candidate_dim),
        "global_dim": int(global_dim),
        "hidden_dim": int(hidden_dim),
        "channels": tuple(channels),
        "top_k": int(top_k),
        "lookahead": bool(lookahead),
        "w_lookahead": float(w_lookahead),
        "process_noise_enabled": bool(process_noise_enabled),
        "noise_seed_base": int(noise_seed_base),
    }
