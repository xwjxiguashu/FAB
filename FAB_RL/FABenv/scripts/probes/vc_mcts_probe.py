"""Compare rule baseline, oracle reservation, and online VC-MCTS."""

from pathlib import Path
import sys

FABENV_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = FABENV_ROOT / "scripts"
DEFAULT_SAS_CHECKPOINT = FABENV_ROOT / "artifacts" / "checkpoints" / "pressure_mh.pt"
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

for _thread_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_var, "1")

import multiprocessing as mp

from dispatch_delegate import RuleDispatchDelegate, load_sas_policy_delegate
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
from vc_mcts_alphazero import load_sas_alphazero
from vc_mcts_trace_summary import summarize_trace_file


def _make_dispatch_delegate(
    mode,
    strategy,
    sas_checkpoint=None,
    sas_stochastic=False,
):
    fallback = RuleDispatchDelegate(strategy=strategy)
    if mode == "rule":
        return fallback, True
    if mode in (None, "sas"):
        checkpoint = Path(sas_checkpoint) if sas_checkpoint else DEFAULT_SAS_CHECKPOINT
        return load_sas_policy_delegate(
            str(checkpoint),
            stochastic=sas_stochastic,
            fallback_delegate=fallback,
        ), True
    raise ValueError(f"unknown dispatch delegate mode: {mode!r}")


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
    dispatch_delegate="sas",
    sas_checkpoint=None,
    sas_stochastic=False,
    prior_source="heuristic",
    use_leaf_value=False,
    leaf_rollout_depth=8,
    alphazero_checkpoint=None,
    crn_noise=False,
    n_mc=1,
    use_rho_pc=False,
    rho_pc_weight=0.0,
    rho_pc_alpha=1.0,
    rho_pc_priority_threshold=None,
    qtime_mask_mode=None,
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
    if qtime_mask_mode:
        baseline_env.qtime_mask_mode = str(qtime_mask_mode)
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
        if qtime_mask_mode:
            oracle_env.qtime_mask_mode = str(qtime_mask_mode)
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
    if qtime_mask_mode:
        mcts_env.qtime_mask_mode = str(qtime_mask_mode)
    mcts_driver = _driver(mcts_env, max_steps)
    mcts_driver.reset_episode()
    delegate, use_delegate_dispatch = _make_dispatch_delegate(
        dispatch_delegate,
        strategy,
        sas_checkpoint=sas_checkpoint,
        sas_stochastic=sas_stochastic,
    )
    prior_provider = None
    leaf_value = None
    needs_alphazero_checkpoint = prior_source == "policy" or use_leaf_value
    if needs_alphazero_checkpoint and not alphazero_checkpoint:
        raise ValueError(
            "alphazero checkpoint is required when --prior-source policy "
            "or --use-leaf-value is enabled"
        )
    if needs_alphazero_checkpoint:
        prior_provider, leaf_value, _policy = load_sas_alphazero(
            alphazero_checkpoint,
            require_multihead=use_leaf_value,
        )

    planner = VCMCTSPlanner(
        VCMCTSConfig(
            n_iter=n_iter,
            top_k_dispatch=top_k_dispatch,
            top_b_reserve=top_b,
            rollout_strategy=strategy,
            rollout_max_steps=rollout_max_steps or max_steps,
            use_delegate_dispatch=use_delegate_dispatch,
            prior_source=prior_source,
            use_leaf_value=use_leaf_value,
            leaf_rollout_depth=leaf_rollout_depth,
            crn_noise=crn_noise,
            n_mc=n_mc,
            crn_seed_base=1000 * (int(seed) + 1),
            use_rho_pc=use_rho_pc,
            rho_pc_weight=rho_pc_weight,
            rho_pc_alpha=rho_pc_alpha,
            rho_pc_priority_threshold=rho_pc_priority_threshold,
        ),
        dispatch_delegate=delegate,
        prior_provider=prior_provider,
        leaf_value=leaf_value,
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
            dispatch_delegate=delegate,
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
        dispatch_delegate,
        sas_checkpoint,
        sas_stochastic,
        prior_source,
        use_leaf_value,
        leaf_rollout_depth,
        alphazero_checkpoint,
        crn_noise,
        n_mc,
        use_rho_pc,
        rho_pc_weight,
        rho_pc_alpha,
        rho_pc_priority_threshold,
        qtime_mask_mode,
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
        dispatch_delegate=dispatch_delegate,
        sas_checkpoint=sas_checkpoint,
        sas_stochastic=sas_stochastic,
        prior_source=prior_source,
        use_leaf_value=use_leaf_value,
        leaf_rollout_depth=leaf_rollout_depth,
        alphazero_checkpoint=alphazero_checkpoint,
        crn_noise=crn_noise,
        n_mc=n_mc,
        use_rho_pc=use_rho_pc,
        rho_pc_weight=rho_pc_weight,
        rho_pc_alpha=rho_pc_alpha,
        rho_pc_priority_threshold=rho_pc_priority_threshold,
        qtime_mask_mode=qtime_mask_mode,
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
    dispatch_delegate="sas",
    sas_checkpoint=None,
    sas_stochastic=False,
    prior_source="heuristic",
    use_leaf_value=False,
    leaf_rollout_depth=8,
    alphazero_checkpoint=None,
    crn_noise=False,
    n_mc=1,
    use_rho_pc=False,
    rho_pc_weight=0.0,
    rho_pc_alpha=1.0,
    rho_pc_priority_threshold=None,
    qtime_mask_mode=None,
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
                dispatch_delegate,
                sas_checkpoint,
                sas_stochastic,
                prior_source,
                use_leaf_value,
                leaf_rollout_depth,
                alphazero_checkpoint,
                crn_noise,
                n_mc,
                use_rho_pc,
                rho_pc_weight,
                rho_pc_alpha,
                rho_pc_priority_threshold,
                qtime_mask_mode,
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
    parser.add_argument(
        "--dispatch-delegate",
        choices=["sas", "rule"],
        default="sas",
    )
    parser.add_argument("--sas-checkpoint", default=None)
    parser.add_argument("--sas-stochastic", action="store_true")
    parser.add_argument(
        "--prior-source",
        choices=["heuristic", "policy"],
        default="heuristic",
    )
    parser.add_argument("--use-leaf-value", action="store_true")
    parser.add_argument("--leaf-rollout-depth", type=int, default=8)
    parser.add_argument("--alphazero-checkpoint", default=None)
    parser.add_argument("--noise", action="store_true")
    parser.add_argument(
        "--crn-noise",
        action="store_true",
        help="机制 3: 搜索 rollout 内注入 CRN 键控加工噪声并按 n_mc 取均值",
    )
    parser.add_argument("--n-mc", type=int, default=1, help="机制 3: CRN 多路 rollout 数")
    parser.add_argument(
        "--use-rho-pc",
        action="store_true",
        help="机制 2: UCT 引导叠加优先级-能力鲁棒性 ρ_pc",
    )
    parser.add_argument("--rho-pc-weight", type=float, default=0.0, help="机制 2: Δρ_pc 加性兼容权重")
    parser.add_argument(
        "--rho-pc-alpha",
        type=float,
        default=1.0,
        help="机制 2: E(s,a)=alpha*q_hat+(1-alpha)*rho_pc 插值 (1.0=纯 q_hat 旧行为)",
    )
    parser.add_argument(
        "--rho-pc-priority-threshold",
        type=float,
        default=None,
        help="机制 2: 高优先级类阈值 p_hi (缺省取窗内可见优先级中位数)",
    )
    parser.add_argument(
        "--qtime-mask-mode",
        choices=["aggregate", "chain", "chain_joint"],
        default=None,
        help="覆盖环境默认 qtime mask 口径 (chain_joint 较精确但慢 ~8x; 缺省不覆盖)",
    )
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
        dispatch_delegate=args.dispatch_delegate,
        sas_checkpoint=args.sas_checkpoint,
        sas_stochastic=args.sas_stochastic,
        prior_source=args.prior_source,
        use_leaf_value=args.use_leaf_value,
        leaf_rollout_depth=args.leaf_rollout_depth,
        alphazero_checkpoint=args.alphazero_checkpoint,
        crn_noise=args.crn_noise,
        n_mc=args.n_mc,
        use_rho_pc=args.use_rho_pc,
        rho_pc_weight=args.rho_pc_weight,
        rho_pc_alpha=args.rho_pc_alpha,
        rho_pc_priority_threshold=args.rho_pc_priority_threshold,
        qtime_mask_mode=args.qtime_mask_mode,
    )


if __name__ == "__main__":
    _cli()
