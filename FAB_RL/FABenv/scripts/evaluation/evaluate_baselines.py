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
import json
import multiprocessing as mp
import os

# worker 进程限制单线程 BLAS/OpenMP —— N 个进程各起满核线程会严重超额订阅 (oversubscription)，
# 比串行还慢。须在 import torch/numpy 之前设 (与 parallel_rollout.py 同策略)。
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

from problem_instances import (
    build_small_encoder,
    build_pressure_test_encoder,
    build_late_hi_encoder,
    build_late_hi_scarce_encoder,
)
from rl_environment import ResourceCalendarEnv, RewardConfig
from phase2_sas_observation import Phase2ObservationEncoder
from phase2_sas_driver import Phase2EpisodeDriver


DEFAULT_RULES = ("FIFO", "SPT", "EDD", "CR", "ATC")

#: encoder 工厂注册表 (worker 按 kind 字符串解析，避免跨进程 pickle 函数对象)。
ENCODER_FACTORIES = {
    "small": build_small_encoder,
    "pressure": build_pressure_test_encoder,
    "late_hi": build_late_hi_encoder,
    "late_hi_scarce": build_late_hi_scarce_encoder,
}

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


def _build_env(encoder, seed, noise, lookahead, w_lookahead,
               priority_mode="soft", priority_min_gap=0.0):
    return ResourceCalendarEnv(
        encoder,
        process_noise_enabled=(noise and seed is not None),
        noise_seed=seed,
        w_lookahead=w_lookahead,
        priority_filter_mode=priority_mode,
        priority_min_gap=priority_min_gap,
    )


def run_rule_seed(encoder_factory, strategy, seed, noise=True, lookahead=False, w_lookahead=0.0,
                  priority_mode="soft", priority_min_gap=0.0):
    """单 (策略, seed) 跑一个规则 episode，返回指标 dict (含 termination_reason)。"""
    encoder = encoder_factory()
    env = _build_env(encoder, seed, noise, lookahead, w_lookahead, priority_mode, priority_min_gap)
    env.reset()
    driver = Phase2EpisodeDriver(env, Phase2ObservationEncoder(lookahead=lookahead), RewardConfig())
    driver.reset_episode()
    summary = driver.run_rule_episode(strategy=strategy)
    m = schedule_metrics(encoder, env)
    m["termination_reason"] = summary["termination_reason"]
    m["steps"] = float(summary["steps"])
    return m


def run_policy_seed(encoder_factory, policy, seed, lookahead=False, w_lookahead=0.0, noise=True,
                    priority_mode="soft", priority_min_gap=0.0):
    """单 seed 跑 RL 贪心 episode (推理)，返回指标 dict。"""
    encoder = encoder_factory()
    env = _build_env(encoder, seed, noise, lookahead, w_lookahead, priority_mode, priority_min_gap)
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


# ---------------------------------------------------------------------------
# 多进程并行评测 (CPU 多核)。每个 (策略|策略名, seed) 是独立 episode，互不共享状态，
# 故天然可并行。worker 常驻进程池，按 kind 字符串解析 encoder 工厂、在进程内加载一次
# checkpoint，避免跨进程 pickle 函数对象 / 重复 IO。spawn 上下文 (Windows 默认)。
# ---------------------------------------------------------------------------

#: worker 进程常驻对象 (spawn 后由 _init_eval_worker 填充)。
_EVAL_WORKER = {}


def _init_eval_worker(spec):
    """worker 初始化: 解析 encoder 工厂、加载 checkpoint 策略 (各一次) 并常驻。"""
    import torch
    torch.set_num_threads(1)  # 单线程，避免多 worker 超额订阅 (见模块顶部)
    _EVAL_WORKER["encoder_factory"] = ENCODER_FACTORIES[spec["encoder_kind"]]
    _EVAL_WORKER["noise"] = spec["noise"]
    _EVAL_WORKER["lookahead"] = spec["lookahead"]
    _EVAL_WORKER["w_lookahead"] = spec["w_lookahead"]
    _EVAL_WORKER["priority_mode"] = spec.get("priority_mode", "soft")
    _EVAL_WORKER["priority_min_gap"] = spec.get("priority_min_gap", 0.0)
    _EVAL_WORKER["policies"] = {
        name: _load_policy(path) for name, path in (spec.get("policy_checkpoints") or {}).items()
    }


def _run_eval_job(job):
    """worker: 跑单个 (name, seed, kind) job，回传 (name, seed, metrics)。"""
    name, seed, kind = job
    ef = _EVAL_WORKER["encoder_factory"]
    kw = dict(noise=_EVAL_WORKER["noise"], lookahead=_EVAL_WORKER["lookahead"],
              w_lookahead=_EVAL_WORKER["w_lookahead"],
              priority_mode=_EVAL_WORKER["priority_mode"],
              priority_min_gap=_EVAL_WORKER["priority_min_gap"])
    if kind == "rule":
        m = run_rule_seed(ef, name, seed, **kw)
    else:
        m = run_policy_seed(ef, _EVAL_WORKER["policies"][name], seed, **kw)
    return name, seed, m


def evaluate_to_file_parallel(out_path, workers, strategies=DEFAULT_RULES, seeds=(0, 1, 2),
                              encoder_kind="small", policy_checkpoints=None,
                              noise=True, lookahead=False, w_lookahead=0.0,
                              priority_mode="soft", priority_min_gap=0.0, verbose=True):
    """多进程版 evaluate_to_file: N 个 worker 并行跑各 (name, seed) job。

    与串行版语义一致 (相同 JSONL 断点续跑、相同聚合)，仅把 job 分发到进程池。
    policy_checkpoints: {name: checkpoint_path}，worker 内加载 (而非传 policy 对象)。
    结果按完成顺序 (imap_unordered) 增量落盘，聚合与到达顺序无关。
    """
    done = _load_done(out_path)
    jobs = [(s, seed, "rule") for s in strategies for seed in seeds]
    jobs += [(name, seed, "policy") for name in (policy_checkpoints or {}) for seed in seeds]
    pending = [(name, seed, kind) for (name, seed, kind) in jobs if (name, seed) not in done]

    if pending:
        spec = {
            "encoder_kind": encoder_kind,
            "noise": noise,
            "lookahead": lookahead,
            "w_lookahead": w_lookahead,
            "priority_mode": priority_mode,
            "priority_min_gap": priority_min_gap,
            "policy_checkpoints": policy_checkpoints or {},
        }
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers, initializer=_init_eval_worker, initargs=(spec,)) as pool:
            for name, seed, m in pool.imap_unordered(_run_eval_job, pending):
                done[(name, seed)] = m
                if out_path:
                    with open(out_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"name": name, "seed": seed, "metrics": m}) + "\n")
                if verbose:
                    print(f"[done] {name} seed={seed}: util={m['avg_utilization']:.3f} "
                          f"tardy={m['total_tardiness']:.1f} qv={m['qtime_violation_count']:.0f} "
                          f"reason={m['termination_reason']}", flush=True)

    names = list(strategies) + list(policy_checkpoints or {})
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


def _parse_checkpoints(checkpoint, checkpoints):
    """合并 --checkpoint (单个, 名 SAS-PPO) 与 --checkpoints (多个 name=path) → {name: path}。

    用于学习曲线: 一次评测多个里程碑 checkpoint。配合相同 --out JSONL，已评过的
    (name, seed) 自动跳过，故只跑新增的里程碑。
    """
    out = {}
    if checkpoint:
        out["SAS-PPO"] = checkpoint
    for pair in (checkpoints or "").split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise ValueError(f"--checkpoints 项须为 name=path，得到 {pair!r}")
        name, path = pair.split("=", 1)
        out[name.strip()] = path.strip()
    return out


def format_markdown(results, instance, n_seeds, noise):
    """把结果渲染成 Markdown 表格 (论文/笔记可直接粘贴)。"""
    head = "| strategy | " + " | ".join(f"{m} {d}" for m, d in REPORT_METRICS) + " | completed% |"
    sep = "|" + "---|" * (len(REPORT_METRICS) + 2)
    lines = [
        f"### 基线对比 (instance={instance}, n_seeds={n_seeds}, "
        f"noise={'on' if noise else 'off'})",
        "",
        head,
        sep,
    ]
    for name, row in results.items():
        cells = [name]
        for metric, _dir in REPORT_METRICS:
            mean = row.get(f"{metric}_mean", float("nan"))
            std = row.get(f"{metric}_std", 0.0)
            cells.append(f"{mean:.2f}±{std:.2f}")
        cells.append(f"{row.get('all_completed_rate', 0.0):.0%}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(instance="small", seeds=5, strategies=DEFAULT_RULES, no_noise=False,
         checkpoint=None, checkpoints=None, out=None, workers=1,
         priority_mode="soft", priority_min_gap=0.0, markdown_out=None):
    """多 seed 基线 + RL 评测。可直接以关键字参数调用 (VSCode 点 Run)，CLI 见 _cli()。"""
    encoder_factory = ENCODER_FACTORIES.get(instance, build_small_encoder)
    if isinstance(strategies, str):
        strategies = tuple(s.strip() for s in strategies.split(",") if s.strip())
    else:
        strategies = tuple(strategies)
    seed_tuple = tuple(range(seeds))
    policy_checkpoints = _parse_checkpoints(checkpoint, checkpoints) or None

    if workers <= 0:
        workers = max(1, (os.cpu_count() or 1) - 1)

    if out and os.path.dirname(out):
        os.makedirs(os.path.dirname(out), exist_ok=True)

    if workers > 1:
        # 并行路径: 在 worker 进程内加载 checkpoint，故传 path 而非 policy 对象。
        out_path = out or f"baselines_{instance}.jsonl"  # 并行须落盘聚合
        results = evaluate_to_file_parallel(
            out_path, workers, strategies=strategies, seeds=seed_tuple,
            encoder_kind=instance, policy_checkpoints=policy_checkpoints,
            noise=not no_noise,
            priority_mode=priority_mode, priority_min_gap=priority_min_gap,
        )
    else:
        policies = None
        if policy_checkpoints:
            policies = {name: _load_policy(path) for name, path in policy_checkpoints.items()}
        if out:
            results = evaluate_to_file(
                out, strategies=strategies, seeds=seed_tuple,
                encoder_factory=encoder_factory, policies=policies, noise=not no_noise,
            )
        else:
            results = evaluate(
                strategies=strategies, seeds=seed_tuple, encoder_factory=encoder_factory,
                policies=policies, noise=not no_noise,
            )

    print(f"\n=== 基线评测: instance={instance}, seeds={len(seed_tuple)}, "
          f"noise={'off' if no_noise else 'on'}, workers={workers}, "
          f"priority={priority_mode} ===")
    print(format_table(results))

    if markdown_out:
        os.makedirs(os.path.dirname(markdown_out) or ".", exist_ok=True)
        with open(markdown_out, "w", encoding="utf-8") as f:
            f.write(format_markdown(results, instance, len(seed_tuple), not no_noise) + "\n")
        print(f"\n[markdown] 结果表已写入 {markdown_out}")
    return results


def _cli():
    parser = argparse.ArgumentParser(description="派工规则基线 + 多 seed 评测")
    parser.add_argument("--instance", choices=["small", "pressure", "late_hi", "late_hi_scarce"], default="small")
    parser.add_argument("--seeds", type=int, default=5, help="seed 数量 (0..N-1)")
    parser.add_argument("--strategies", default=",".join(DEFAULT_RULES),
                        help="逗号分隔的规则名")
    parser.add_argument("--no-noise", action="store_true", help="关闭加工噪声 (各 seed 确定相同)")
    parser.add_argument("--checkpoint", default=None, help="可选 RL 策略检查点 (.pt)，纳入贪心对比 (名 SAS-PPO)")
    parser.add_argument("--checkpoints", default=None,
                        help="逗号分隔 name=path，多个里程碑 checkpoint 一起评测 (学习曲线)；"
                             "可与 --checkpoint 合并")
    parser.add_argument("--out", default=None,
                        help="增量写入 JSONL 路径 (断点续跑)；适合 pressure 等慢实例")
    parser.add_argument("--markdown-out", default=None,
                        help="可选: 把结果表写成 Markdown 文件 (论文可直接粘贴)")
    parser.add_argument("--workers", type=int, default=1,
                        help="并行 worker 进程数 (>1 用多核; 各 (策略,seed) 独立可并行)。"
                             "<=0 取 CPU 核数-1")
    parser.add_argument("--priority-mode", choices=["soft", "strict"], default="soft",
                        help="候选池优先级过滤模式 (报告 §3.4)：soft=不删候选仅重排(默认); "
                             "strict=只保留最高优先级候选,让所有策略/RL 都物理上无法选低优先级")
    parser.add_argument("--priority-min-gap", type=float, default=0.0,
                        help="strict 模式下的优先级容差 (priority >= max_pri - gap 的候选保留)")
    args = parser.parse_args()
    main(instance=args.instance, seeds=args.seeds, strategies=args.strategies,
         no_noise=args.no_noise, checkpoint=args.checkpoint, checkpoints=args.checkpoints,
         out=args.out, workers=args.workers, priority_mode=args.priority_mode,
         priority_min_gap=args.priority_min_gap, markdown_out=args.markdown_out)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        _cli()
        raise SystemExit

    # —— 直接在 VSCode 点 Run 就出 pressure 主结果表 (无需命令行) ——
    # 改下面的参数即可；要用命令行就把这段 main(...) 注释掉、解开末尾的 _cli()。
    # 路径锚定到本文件所在目录: VSCode 的 cwd 常是工作区根, 相对路径会找不到 checkpoint。
    FABENV_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    CHECKPOINTS = os.path.join(FABENV_ROOT, "artifacts", "checkpoints")
    RESULTS = os.path.join(FABENV_ROOT, "artifacts", "results")
    # 困难实例 (gap=0.6 + 阶段间 Q-time) 主表: 重训的 pressure_mh_hard.pt vs 启发式规则。
    # 注意: out/markdown 用 *hard* 新文件名 —— 旧 pressure_main.jsonl 缓存的是旧易实例结果,
    #       复用会读到过期数字 (JSONL 断点续跑按 (name,seed) 跳过)。
    main(
        instance="pressure",
        seeds=5,
        checkpoint=os.path.join(CHECKPOINTS, "pressure_mh_hard.pt"),   # 新 3 通道重训模型 → SAS-PPO 贪心对比
        workers=10,                             # 30 个 job, 10 worker 够用 (启动各 worker import torch 会静默几十秒)
        # _v2 全新路径: 旧 pressure_hard_main.jsonl 缓存的是 4 通道模型结果, 复用会显示旧数字。
        out=os.path.join(RESULTS, "pressure_hard_v2.jsonl"),
        markdown_out=os.path.join(RESULTS, "pressure_hard_v2_table.md"),
    )
    # 旧 strict 对照已退役: pressure_mh_strict.pt 是在旧易实例上训的, 与新困难实例不可比。
    # _cli()  # 命令行用法
