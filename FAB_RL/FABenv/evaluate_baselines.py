"""派工规则基线 + 多 seed 评测 (报告 §7.4 基线对比, §4.10 指标, §2.4.6 多 rollout 统计)。

回答一个生死问题: 当前 SAS-PPO 相对启发式派工规则 (FIFO/SPT/EDD/CR/ATC) 到底赢没赢。

评测协议:
  - 多 seed: 每个 seed 是一次加工噪声实现 (process_noise_enabled + noise_seed=seed)，
    对应报告 §2.4.6 "在多次随机 rollout 上统计"。
  - 指标 (报告 §4.10): Q-time/拖期违规 (硬约束门槛, 越小越好) + 机台利用率 (唯一软目标) +
    优先级违反度。分层报告，不做帕累托强主张。
  - 公平性: 所有规则与 RL 用相同 build_candidate_pool (qtime-safe + priority 过滤)，
    仅"在候选池里挑哪个"不同。

用法:
    python evaluate_baselines.py                         # small 实例, 默认 seed/策略
    python evaluate_baselines.py --instance pressure --seeds 5
    python evaluate_baselines.py --checkpoint model.pt   # 额外纳入 RL greedy 对比
"""

import argparse
import json
import os

import numpy as np

from problem_instances import build_small_encoder, build_pressure_test_encoder
from rl_environment import ResourceCalendarEnv, RewardConfig
from phase2_sas_observation import Phase2ObservationEncoder
from phase2_sas_driver import Phase2EpisodeDriver


DEFAULT_RULES = ("FIFO", "SPT", "EDD", "CR", "ATC")

#: 指标键 → (在 evaluate_objectives 向量中的索引, 是否取负)。
#: evaluate_objectives 返回 [q_count, q_total, tardy_count, total_tardiness, priority_violation, -avg_util]。
_METRIC_SPEC = {
    "qtime_violation_count": (0, False),
    "qtime_violation_total": (1, False),
    "tardy_count": (2, False),
    "total_tardiness": (3, False),
    "priority_violation": (4, False),
    "avg_utilization": (5, True),   # 存的是 -util，取负还原
}

#: 报告/展示时的指标顺序与方向 (↓ 越小越好 / ↑ 越大越好)。
REPORT_METRICS = (
    ("qtime_violation_count", "↓"),
    ("total_tardiness", "↓"),
    ("priority_violation", "↓"),
    ("avg_utilization", "↑"),
    ("completed_lots", "↑"),
)


def schedule_metrics(encoder, env):
    """从 env 当前排程提取评测指标 (报告 §4.10)。"""
    obj = encoder.evaluate_objectives(env.lot_schedule, env.wafer_schedule, current_time=0.0)
    metrics = {}
    for key, (idx, negate) in _METRIC_SPEC.items():
        val = float(obj[idx])
        metrics[key] = -val if negate else val
    metrics["completed_lots"] = float(len(env.completed_lots))
    return metrics


def _build_env(encoder, seed, noise, lookahead, w_lookahead):
    return ResourceCalendarEnv(
        encoder,
        process_noise_enabled=(noise and seed is not None),
        noise_seed=seed,
        w_lookahead=w_lookahead,
    )


def run_rule_seed(encoder_factory, strategy, seed, noise=True, lookahead=False, w_lookahead=0.0):
    """单 (策略, seed) 跑一个规则 episode，返回指标 dict (含 termination_reason)。"""
    encoder = encoder_factory()
    env = _build_env(encoder, seed, noise, lookahead, w_lookahead)
    env.reset()
    driver = Phase2EpisodeDriver(env, Phase2ObservationEncoder(lookahead=lookahead), RewardConfig())
    driver.reset_episode()
    summary = driver.run_rule_episode(strategy=strategy)
    m = schedule_metrics(encoder, env)
    m["termination_reason"] = summary["termination_reason"]
    m["steps"] = float(summary["steps"])
    return m


def run_policy_seed(encoder_factory, policy, seed, lookahead=False, w_lookahead=0.0, noise=True):
    """单 seed 跑 RL 贪心 episode (推理)，返回指标 dict。"""
    encoder = encoder_factory()
    env = _build_env(encoder, seed, noise, lookahead, w_lookahead)
    env.reset()
    driver = Phase2EpisodeDriver(env, Phase2ObservationEncoder(lookahead=lookahead), RewardConfig())
    driver.reset_episode()
    summary = driver.run_greedy_episode(policy)
    m = schedule_metrics(encoder, env)
    m["termination_reason"] = summary["termination_reason"]
    m["steps"] = float(summary["steps"])
    return m


def _aggregate(per_seed_metrics):
    """对一个策略的多 seed 指标列表聚合 mean/std。"""
    row = {"n_seeds": len(per_seed_metrics)}
    numeric_keys = [k for k in per_seed_metrics[0] if k != "termination_reason"]
    for key in numeric_keys:
        vals = np.array([m[key] for m in per_seed_metrics], dtype=float)
        row[f"{key}_mean"] = float(vals.mean())
        row[f"{key}_std"] = float(vals.std())
    # 完成率: 多少 seed 跑到 all_lots_completed
    completed = sum(1 for m in per_seed_metrics if m.get("termination_reason") == "all_lots_completed")
    row["all_completed_rate"] = completed / len(per_seed_metrics)
    return row


def evaluate(strategies=DEFAULT_RULES, seeds=(0, 1, 2, 3, 4), encoder_factory=build_small_encoder,
             policies=None, noise=True, lookahead=False, w_lookahead=0.0):
    """多策略 × 多 seed 评测，返回 {name: 聚合行}。

    Args:
        strategies: 规则名元组。
        seeds: seed 元组 (每个是一次噪声实现)。
        encoder_factory: 实例工厂。
        policies: 可选 {name: policy}，纳入 RL 贪心对比。
        noise: 是否注入加工噪声 (False 则各 seed 相同, std=0)。
    """
    results = {}
    for strategy in strategies:
        per_seed = [
            run_rule_seed(encoder_factory, strategy, seed, noise=noise,
                          lookahead=lookahead, w_lookahead=w_lookahead)
            for seed in seeds
        ]
        results[strategy] = _aggregate(per_seed)
    for name, policy in (policies or {}).items():
        per_seed = [
            run_policy_seed(encoder_factory, policy, seed, lookahead=lookahead,
                            w_lookahead=w_lookahead, noise=noise)
            for seed in seeds
        ]
        results[name] = _aggregate(per_seed)
    return results


def _load_done(out_path):
    """读取 JSONL 已完成的 (name, seed) → 结果，支持断点续跑。"""
    done = {}
    if out_path and os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                done[(rec["name"], rec["seed"])] = rec["metrics"]
    return done


def evaluate_to_file(out_path, strategies=DEFAULT_RULES, seeds=(0, 1, 2),
                     encoder_factory=build_small_encoder, policies=None,
                     noise=True, lookahead=False, w_lookahead=0.0, verbose=True):
    """逐 (name, seed) 评测并增量写入 JSONL (可断点续跑)。

    每条记录: {"name", "seed", "metrics": {...}}。已存在的 (name, seed) 跳过，
    故被超时杀掉后重跑会自动继续。返回聚合后的 {name: 行}。
    """
    done = _load_done(out_path)
    jobs = [(s, seed, "rule") for s in strategies for seed in seeds]
    jobs += [(name, seed, "policy") for name in (policies or {}) for seed in seeds]

    for name, seed, kind in jobs:
        if (name, seed) in done:
            continue
        if kind == "rule":
            m = run_rule_seed(encoder_factory, name, seed, noise=noise,
                              lookahead=lookahead, w_lookahead=w_lookahead)
        else:
            m = run_policy_seed(encoder_factory, policies[name], seed,
                                lookahead=lookahead, w_lookahead=w_lookahead, noise=noise)
        done[(name, seed)] = m
        if out_path:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"name": name, "seed": seed, "metrics": m}) + "\n")
        if verbose:
            print(f"[done] {name} seed={seed}: util={m['avg_utilization']:.3f} "
                  f"tardy={m['total_tardiness']:.1f} qv={m['qtime_violation_count']:.0f} "
                  f"reason={m['termination_reason']}", flush=True)

    # 聚合
    names = list(strategies) + list(policies or {})
    results = {}
    for name in names:
        per_seed = [done[(name, seed)] for seed in seeds if (name, seed) in done]
        if per_seed:
            results[name] = _aggregate(per_seed)
    return results


def format_table(results):
    """把 evaluate 的结果格式化为可读对比表 (mean±std)。"""
    names = list(results.keys())
    header = f"{'strategy':<12}" + "".join(f"{m+d:>22}" for m, d in REPORT_METRICS)
    lines = [header, "-" * len(header)]
    for name in names:
        row = results[name]
        cells = [f"{name:<12}"]
        for metric, _dir in REPORT_METRICS:
            mean = row.get(f"{metric}_mean", float("nan"))
            std = row.get(f"{metric}_std", 0.0)
            cells.append(f"{mean:>10.3f}±{std:<7.3f}".rjust(22))
        lines.append("".join(cells))
    lines.append(f"\n(n_seeds={results[names[0]]['n_seeds']}, "
                 f"all_completed_rate per策略: "
                 + ", ".join(f"{n}={results[n]['all_completed_rate']:.0%}" for n in names) + ")")
    return "\n".join(lines)


def _load_policy(checkpoint_path):
    from model_checkpoint import load_policy_checkpoint
    bundle = load_policy_checkpoint(checkpoint_path)
    # load_policy_checkpoint 返回 policy 或 (policy, meta)
    return bundle[0] if isinstance(bundle, tuple) else bundle


def main():
    parser = argparse.ArgumentParser(description="派工规则基线 + 多 seed 评测")
    parser.add_argument("--instance", choices=["small", "pressure"], default="small")
    parser.add_argument("--seeds", type=int, default=5, help="seed 数量 (0..N-1)")
    parser.add_argument("--strategies", default=",".join(DEFAULT_RULES),
                        help="逗号分隔的规则名")
    parser.add_argument("--no-noise", action="store_true", help="关闭加工噪声 (各 seed 确定相同)")
    parser.add_argument("--checkpoint", default=None, help="可选 RL 策略检查点 (.pt)，纳入贪心对比")
    parser.add_argument("--out", default=None,
                        help="增量写入 JSONL 路径 (断点续跑)；适合 pressure 等慢实例")
    args = parser.parse_args()

    encoder_factory = build_pressure_test_encoder if args.instance == "pressure" else build_small_encoder
    strategies = tuple(s.strip() for s in args.strategies.split(",") if s.strip())
    seeds = tuple(range(args.seeds))
    policies = None
    if args.checkpoint:
        policies = {"SAS-PPO": _load_policy(args.checkpoint)}

    if args.out:
        results = evaluate_to_file(
            args.out, strategies=strategies, seeds=seeds,
            encoder_factory=encoder_factory, policies=policies, noise=not args.no_noise,
        )
    else:
        results = evaluate(
            strategies=strategies, seeds=seeds, encoder_factory=encoder_factory,
            policies=policies, noise=not args.no_noise,
        )
    print(f"\n=== 基线评测: instance={args.instance}, seeds={len(seeds)}, "
          f"noise={'off' if args.no_noise else 'on'} ===")
    print(format_table(results))


if __name__ == "__main__":
    main()
