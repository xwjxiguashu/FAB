"""Oracle reservation go/no-go probe.

This script compares normal rule dispatch against the Scheme C oracle
reservation wrapper. It is intentionally a probe, not the online VC-MCTS
implementation.
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
import os
import sys
import time

# 多进程跨 seed 并行时，每个 worker 钉死 BLAS/OpenMP 单线程，避免 N×worker 超额订阅
# (与 parallel_rollout.py 同纪律)。必须在 import numpy (经 reservation_simulator) 之前设置。
for _thread_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_thread_var, "1")

import multiprocessing as mp

from phase2_sas_driver import Phase2EpisodeDriver
from phase2_sas_observation import Phase2ObservationEncoder
from problem_instances import (
    build_late_hi_encoder,
    build_late_hi_scarce_encoder,
    build_pressure_test_encoder,
    build_small_encoder,
)
from reservation_simulator import (
    run_oracle_reservation_episode,
    schedule_metrics_with_priority_wait,
)
from rl_environment import ResourceCalendarEnv, RewardConfig


def _encoder_factory(instance):
    if instance == "small":
        return build_small_encoder
    if instance == "pressure":
        return build_pressure_test_encoder
    if instance == "late_hi":
        return build_late_hi_encoder
    if instance == "late_hi_scarce":
        return build_late_hi_scarce_encoder
    raise ValueError(f"unknown instance: {instance}")


def _driver(env, max_steps):
    return Phase2EpisodeDriver(
        env,
        Phase2ObservationEncoder(),
        RewardConfig(),
        max_steps=max_steps,
    )


def _full_horizon_lookahead(encoder):
    """Lookahead window that spans every arrival (information-complete oracle).

    The go/no-go oracle (报告4 §6.2.3 阶段 0) must see *all* future arrivals,
    not just a fixed window — otherwise it is not an upper bound. From any
    t_now >= 0, ``t_now + W`` still covers ``max(arrival)`` when W = max+1.
    """
    arrivals = list(encoder.arrival_times.values())
    if not arrivals:
        return 1.0e6
    return float(max(arrivals)) + 1.0


def run_seed(instance, seed, strategy, w_lookahead, top_b, max_steps,
             oracle_full_horizon=True, process_noise=False):
    factory = _encoder_factory(instance)

    baseline_encoder = factory()
    baseline_env = ResourceCalendarEnv(
        baseline_encoder,
        top_k=8,
        w_lookahead=w_lookahead,
        process_noise_enabled=process_noise,
        noise_seed=seed,
    )
    baseline_driver = _driver(baseline_env, max_steps)
    baseline_driver.reset_episode()
    baseline_summary = baseline_driver.run_rule_episode(strategy=strategy)
    baseline_metrics = schedule_metrics_with_priority_wait(baseline_encoder, baseline_env)

    oracle_encoder = factory()
    oracle_lookahead = (
        _full_horizon_lookahead(oracle_encoder)
        if oracle_full_horizon
        else w_lookahead
    )
    oracle_env = ResourceCalendarEnv(
        oracle_encoder,
        top_k=8,
        w_lookahead=oracle_lookahead,
        process_noise_enabled=process_noise,
        noise_seed=seed,
    )
    oracle_driver = _driver(oracle_env, max_steps)
    oracle_driver.reset_episode()
    oracle_summary = run_oracle_reservation_episode(
        oracle_driver,
        strategy=strategy,
        top_b=top_b,
        max_steps=max_steps,
    )
    oracle_metrics = schedule_metrics_with_priority_wait(oracle_encoder, oracle_env)

    return {
        "seed": int(seed),
        "baseline": {**baseline_summary, **baseline_metrics},
        "oracle": {**oracle_summary, **oracle_metrics},
        "delta": {
            "qtime_violation_count": (
                oracle_metrics["qtime_violation_count"]
                - baseline_metrics["qtime_violation_count"]
            ),
            "priority_weighted_wait": (
                oracle_metrics["priority_weighted_wait"]
                - baseline_metrics["priority_weighted_wait"]
            ),
            "avg_utilization": (
                oracle_metrics["avg_utilization"]
                - baseline_metrics["avg_utilization"]
            ),
        },
    }


def _run_seed_job(args):
    """Picklable worker for multiprocessing: one seed → one result row."""
    (instance, seed, strategy, w_lookahead, top_b, max_steps,
     oracle_full_horizon, process_noise) = args
    t0 = time.time()
    row = run_seed(
        instance, seed, strategy, w_lookahead, top_b, max_steps,
        oracle_full_horizon=oracle_full_horizon, process_noise=process_noise,
    )
    row["_elapsed_s"] = time.time() - t0
    return row


def _print_seed_done(row):
    d = row["delta"]
    print(
        f"[probe] seed {row['seed']} done in {row.get('_elapsed_s', 0.0):.0f}s: "
        f"O2_delta={d['priority_weighted_wait']:+.1f} "
        f"qtime_delta={d['qtime_violation_count']:+.0f} "
        f"reservations={row['oracle'].get('reservations_made', 0)}",
        file=sys.stderr,
        flush=True,
    )


def main(
    instance="small",
    seeds=1,
    strategy="FIFO",
    w_lookahead=4.0,
    top_b=2,
    max_steps=500,
    out=None,
    oracle_full_horizon=True,
    workers=1,
    process_noise=False,
):
    n = int(seeds)
    workers = max(1, int(workers))
    rows_by_seed = {}
    out_file = None
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        out_file = open(out, "w", encoding="utf-8")

    def _record(row):
        rows_by_seed[int(row["seed"])] = row
        if out_file is not None:
            # 增量写入 + flush: 每个 seed 一完成即落盘, 可被 tail/监视器看到进度, 且崩溃不丢已算结果。
            out_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_file.flush()
        _print_seed_done(row)

    jobs = [
        (instance, seed, strategy, w_lookahead, top_b, max_steps,
         oracle_full_horizon, process_noise)
        for seed in range(n)
    ]
    try:
        if workers > 1 and n > 1:
            # seed 之间独立 → 跨 seed 并行 (spawn, 与 evaluate_baselines/parallel_rollout 同模式)。
            print(f"[probe] running {n} seeds on {workers} workers (spawn)...",
                  file=sys.stderr, flush=True)
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=min(workers, n)) as pool:
                for row in pool.imap_unordered(_run_seed_job, jobs):
                    _record(row)
        else:
            for job in jobs:
                seed = job[1]
                print(f"[probe] seed {seed}/{n - 1} running...", file=sys.stderr, flush=True)
                _record(_run_seed_job(job))
    finally:
        if out_file is not None:
            out_file.close()

    rows = [rows_by_seed[s] for s in sorted(rows_by_seed)]
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return rows


def _cli():
    parser = argparse.ArgumentParser(description="Oracle reservation go/no-go probe")
    parser.add_argument("--instance", choices=["small", "pressure", "late_hi"], default="small")
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--strategy", choices=Phase2EpisodeDriver.RULE_STRATEGIES, default="FIFO")
    parser.add_argument("--w-lookahead", type=float, default=4.0)
    parser.add_argument("--top-b", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--oracle-window",
        action="store_true",
        help="restrict oracle to --w-lookahead instead of the full horizon "
             "(default oracle is information-complete: sees all arrivals)",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="parallelize across seeds with N spawn processes (seeds are independent)",
    )
    parser.add_argument(
        "--noise", action="store_true",
        help="enable per-seed process-noise realizations (§2.4.6); without it every "
             "seed is the identical deterministic run, so multi-seed adds no info",
    )
    args = parser.parse_args()
    main(
        instance=args.instance,
        seeds=args.seeds,
        strategy=args.strategy,
        w_lookahead=args.w_lookahead,
        top_b=args.top_b,
        max_steps=args.max_steps,
        out=args.out,
        oracle_full_horizon=not args.oracle_window,
        workers=args.workers,
        process_noise=args.noise,
    )


if __name__ == "__main__":
    _cli()
