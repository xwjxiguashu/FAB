"""Compare rule baseline, oracle reservation, and online VC-MCTS."""
import argparse
import json
import os
import sys
import time

for _thread_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_var, "1")

import multiprocessing as mp

from oracle_reservation_probe import (
    _driver,
    _encoder_factory,
    _full_horizon_lookahead,
)
from reservation_simulator import (
    run_oracle_reservation_episode,
    schedule_metrics_with_priority_wait,
)
from rl_environment import ResourceCalendarEnv
from vc_mcts_planner import (
    VCMCTSConfig,
    VCMCTSPlanner,
    run_vc_mcts_reservation_episode,
)
from vc_mcts_trace_summary import summarize_trace_file


def _seed_output_path(path, seed):
    if not path:
        return None
    directory, name = os.path.split(path)
    stem, ext = os.path.splitext(name)
    for suffix in ("_trace", "_summary"):
        if stem.endswith(suffix):
            stem = f"{stem[:-len(suffix)]}_seed{int(seed)}{suffix}"
            break
    else:
        stem = f"{stem}_seed{int(seed)}"
    return os.path.join(directory, stem + ext)


def run_seed(
    instance,
    seed,
    strategy,
    w_lookahead,
    top_b,
    top_k_dispatch,
    n_iter,
    max_steps,
    process_noise=False,
    skip_oracle=False,
    rollout_max_steps=None,
    max_decisions=None,
    stop_after_reserve_available=None,
    stop_after_reserve_selected=None,
    trace_out=None,
    trace_summary_out=None,
    progress_every=0,
):
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

    oracle_row = None
    oracle_metrics = None
    if not skip_oracle:
        oracle_encoder = factory()
        oracle_env = ResourceCalendarEnv(
            oracle_encoder,
            top_k=8,
            w_lookahead=_full_horizon_lookahead(oracle_encoder),
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
        oracle_row = {**oracle_summary, **oracle_metrics}

    mcts_encoder = factory()
    mcts_env = ResourceCalendarEnv(
        mcts_encoder,
        top_k=8,
        w_lookahead=w_lookahead,
        process_noise_enabled=process_noise,
        noise_seed=seed,
    )
    mcts_driver = _driver(mcts_env, max_steps)
    mcts_driver.reset_episode()
    planner = VCMCTSPlanner(
        VCMCTSConfig(
            n_iter=n_iter,
            top_k_dispatch=top_k_dispatch,
            top_b_reserve=top_b,
            rollout_strategy=strategy,
            rollout_max_steps=rollout_max_steps or max_steps,
        )
    )
    trace_file = None
    try:
        if trace_out:
            os.makedirs(os.path.dirname(trace_out) or ".", exist_ok=True)
            trace_file = open(trace_out, "a", encoding="utf-8")
        mcts_summary = run_vc_mcts_reservation_episode(
            mcts_driver,
            planner=planner,
            max_steps=max_steps,
            max_decisions=max_decisions,
            stop_after_reserve_available=stop_after_reserve_available,
            stop_after_reserve_selected=stop_after_reserve_selected,
            trace_writer=trace_file,
            progress_every=progress_every,
        )
    finally:
        if trace_file is not None:
            trace_file.close()
    mcts_metrics = schedule_metrics_with_priority_wait(mcts_encoder, mcts_env)
    trace_summary = None
    if trace_summary_out:
        if not trace_out:
            raise ValueError("trace_summary_out requires trace_out")
        trace_summary = summarize_trace_file(trace_out)
        os.makedirs(os.path.dirname(trace_summary_out) or ".", exist_ok=True)
        with open(trace_summary_out, "w", encoding="utf-8") as handle:
            json.dump(trace_summary, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

    return {
        "seed": int(seed),
        "baseline": {**baseline_summary, **baseline_metrics},
        "oracle": oracle_row,
        "vc_mcts": {**mcts_summary, **mcts_metrics},
        "trace_summary": trace_summary,
        "delta": {
            "oracle_o2": None
            if oracle_metrics is None
            else (
                oracle_metrics["priority_weighted_wait"]
                - baseline_metrics["priority_weighted_wait"]
            ),
            "vc_mcts_o2": (
                mcts_metrics["priority_weighted_wait"]
                - baseline_metrics["priority_weighted_wait"]
            ),
            "oracle_qtime": None
            if oracle_metrics is None
            else (
                oracle_metrics["qtime_violation_count"]
                - baseline_metrics["qtime_violation_count"]
            ),
            "vc_mcts_qtime": (
                mcts_metrics["qtime_violation_count"]
                - baseline_metrics["qtime_violation_count"]
            ),
        },
    }


def _run_seed_job(args):
    (
        instance,
        seed,
        strategy,
        w_lookahead,
        top_b,
        top_k_dispatch,
        n_iter,
        max_steps,
        process_noise,
        skip_oracle,
        rollout_max_steps,
        max_decisions,
        stop_after_reserve_available,
        stop_after_reserve_selected,
        trace_out,
        trace_summary_out,
        progress_every,
    ) = args
    t0 = time.time()
    row = run_seed(
        instance=instance,
        seed=seed,
        strategy=strategy,
        w_lookahead=w_lookahead,
        top_b=top_b,
        top_k_dispatch=top_k_dispatch,
        n_iter=n_iter,
        max_steps=max_steps,
        process_noise=process_noise,
        skip_oracle=skip_oracle,
        rollout_max_steps=rollout_max_steps,
        max_decisions=max_decisions,
        stop_after_reserve_available=stop_after_reserve_available,
        stop_after_reserve_selected=stop_after_reserve_selected,
        trace_out=trace_out,
        trace_summary_out=trace_summary_out,
        progress_every=progress_every,
    )
    row["_elapsed_s"] = time.time() - t0
    return row


def _print_seed_done(row):
    delta = row["delta"]
    vc = row["vc_mcts"]
    print(
        f"[vc_mcts_probe] seed {row['seed']} done in "
        f"{row.get('_elapsed_s', 0.0):.0f}s: "
        f"O2_delta={delta['vc_mcts_o2']:+.1f} "
        f"qtime_delta={delta['vc_mcts_qtime']:+.0f} "
        f"reservations={vc.get('reservations_made', 0)} "
        f"completed={vc.get('completed_lots', 0)}",
        file=sys.stderr,
        flush=True,
    )


def main(
    instance="small",
    seeds=1,
    strategy="FIFO",
    w_lookahead=4.0,
    top_b=2,
    top_k_dispatch=3,
    n_iter=24,
    max_steps=500,
    process_noise=False,
    skip_oracle=False,
    rollout_max_steps=None,
    max_decisions=None,
    stop_after_reserve_available=None,
    stop_after_reserve_selected=None,
    trace_out=None,
    trace_summary_out=None,
    progress_every=0,
    workers=1,
):
    n = int(seeds)
    workers = max(1, int(workers))
    use_per_seed_outputs = workers > 1 and n > 1
    jobs = []
    for seed in range(n):
        seed_trace_out = (
            _seed_output_path(trace_out, seed)
            if use_per_seed_outputs
            else trace_out
        )
        seed_summary_out = (
            _seed_output_path(trace_summary_out, seed)
            if use_per_seed_outputs
            else trace_summary_out
        )
        jobs.append(
            (
                instance,
                seed,
                strategy,
                w_lookahead,
                top_b,
                top_k_dispatch,
                n_iter,
                max_steps,
                process_noise,
                skip_oracle,
                rollout_max_steps,
                max_decisions,
                stop_after_reserve_available,
                stop_after_reserve_selected,
                seed_trace_out,
                seed_summary_out,
                progress_every,
            )
        )

    rows_by_seed = {}
    if workers > 1 and n > 1:
        print(
            f"[vc_mcts_probe] running {n} seeds on {workers} workers (spawn)...",
            file=sys.stderr,
            flush=True,
        )
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=min(workers, n)) as pool:
            for row in pool.imap_unordered(_run_seed_job, jobs):
                rows_by_seed[int(row["seed"])] = row
                _print_seed_done(row)
    else:
        for job in jobs:
            seed = job[1]
            print(
                f"[vc_mcts_probe] seed {seed}/{n - 1} running...",
                file=sys.stderr,
                flush=True,
            )
            row = _run_seed_job(job)
            rows_by_seed[int(row["seed"])] = row
            _print_seed_done(row)

    rows = [rows_by_seed[seed] for seed in sorted(rows_by_seed)]
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return rows


def _cli():
    parser = argparse.ArgumentParser(description="VC-MCTS online reservation probe")
    parser.add_argument("--instance", choices=["small", "pressure", "late_hi"], default="small")
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument(
        "--strategy",
        choices=["first_valid", "FIFO", "SPT", "EDD", "CR", "ATC"],
        default="FIFO",
    )
    parser.add_argument("--w-lookahead", type=float, default=4.0)
    parser.add_argument("--top-b", type=int, default=2)
    parser.add_argument("--top-k-dispatch", type=int, default=3)
    parser.add_argument("--n-iter", type=int, default=24)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--skip-oracle", action="store_true")
    parser.add_argument("--rollout-max-steps", type=int, default=None)
    parser.add_argument("--max-decisions", type=int, default=None)
    parser.add_argument("--stop-after-reserve-available", type=int, default=None)
    parser.add_argument("--stop-after-reserve-selected", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--trace-out", default=None)
    parser.add_argument("--trace-summary-out", default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--noise", action="store_true")
    args = parser.parse_args()
    main(
        instance=args.instance,
        seeds=args.seeds,
        strategy=args.strategy,
        w_lookahead=args.w_lookahead,
        top_b=args.top_b,
        top_k_dispatch=args.top_k_dispatch,
        n_iter=args.n_iter,
        max_steps=args.max_steps,
        process_noise=args.noise,
        skip_oracle=args.skip_oracle,
        rollout_max_steps=args.rollout_max_steps,
        max_decisions=args.max_decisions,
        stop_after_reserve_available=args.stop_after_reserve_available,
        stop_after_reserve_selected=args.stop_after_reserve_selected,
        trace_out=args.trace_out,
        trace_summary_out=args.trace_summary_out,
        progress_every=args.progress_every,
        workers=args.workers,
    )


if __name__ == "__main__":
    _cli()
