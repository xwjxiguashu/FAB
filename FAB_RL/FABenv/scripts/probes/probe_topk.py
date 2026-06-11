"""便宜探针: 用现有 pressure_mh.pt (K=8 训练) 在不同 top_k 下做 greedy 推理。

问题: soft 模式下 TopK=8 截断了 ~92% 候选, 把决策权交给了启发式 score。
若放大 K, RL 的可选集打开, 行为/util 会不会变? 网络是 per-candidate 的, 能吃任意 K。
注意: 训练时 K=8, 这里 K>8 是训练/推理分布不一致的探针, 只看趋势, 不当最终结论。
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
import numpy as np

from problem_instances import build_pressure_test_encoder
from rl_environment import ResourceCalendarEnv, RewardConfig
from phase2_sas_observation import Phase2ObservationEncoder
from phase2_sas_driver import Phase2EpisodeDriver
from evaluate_baselines import schedule_metrics, _load_policy

SEEDS = [0, 1]
TOP_KS = [8, 16, 32, 64]
OUT = str(FABENV_ROOT / "artifacts" / "results" / "probe_topk.txt")


def run_one(policy, top_k, seed):
    enc = build_pressure_test_encoder()
    env = ResourceCalendarEnv(
        enc, top_k=top_k, process_noise_enabled=True, noise_seed=seed,
        priority_filter_mode="soft",
    )
    env.reset()
    driver = Phase2EpisodeDriver(env, Phase2ObservationEncoder(), RewardConfig())
    driver.reset_episode()
    summary = driver.run_greedy_episode(policy)
    m = schedule_metrics(enc, env)
    m["termination_reason"] = summary["termination_reason"]
    return m


def main():
    policy = _load_policy(str(FABENV_ROOT / "artifacts" / "checkpoints" / "pressure_mh.pt"))
    lines = ["=== TopK 探针 (pressure, soft, greedy, pressure_mh.pt) ==="]
    for top_k in TOP_KS:
        utils, pris, comps, qv = [], [], [], []
        for seed in SEEDS:
            m = run_one(policy, top_k, seed)
            utils.append(m["avg_utilization"])
            pris.append(m["priority_violation"])
            comps.append(m["completed_lots"])
            qv.append(m["qtime_violation_count"])
        line = (
            f"top_k={top_k:<3} util={np.mean(utils):.3f}±{np.std(utils):.3f}  "
            f"priority_viol={np.mean(pris):8.1f}±{np.std(pris):.1f}  "
            f"qtime_viol={np.mean(qv):.1f}  completed={np.mean(comps):.0f}"
        )
        print(line, flush=True)
        lines.append(line)
    import os
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[written] {OUT}")


if __name__ == "__main__":
    main()
