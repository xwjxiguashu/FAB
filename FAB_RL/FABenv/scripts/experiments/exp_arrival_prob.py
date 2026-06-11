# -*- coding: utf-8 -*-
"""方向2 最小验证：概率加权预留 vs 确定性预留 (late_hi, noise).

A = 现状(确定性预留, arrival_prob_weighting=False)
B = 概率加权预留(arrival_prob_weighting=True): reserve 收益按"窗内工件按 ETA 距离
    衰减的到达概率"加权，远 ETA 的预留被折扣，避免押注不一定来的工件。
对比 Q-time(均值/尾部=最坏种子) 与 预留浪费率(made-consumed)/util/O2。
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
import os
import statistics as st

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from oracle_reservation_probe import _driver, _encoder_factory
from rl_environment import ResourceCalendarEnv
from reservation_simulator import schedule_metrics_with_priority_wait
from vc_mcts_planner import VCMCTSConfig, VCMCTSPlanner, run_vc_mcts_reservation_episode

SEEDS = [0, 1]
MAX_STEPS = 500
W_LOOKAHEAD = 4.0
DECAY = 2.0          # 到达概率衰减率: p = exp(-DECAY * (eta-now)/window)
LIGHT = dict(n_iter=5, top_b_reserve=2, top_k_dispatch=2, rollout_max_steps=40)


def run_one(weighting, seed):
    build = _encoder_factory("late_hi")
    encoder = build()  # 结构固定(seed=2026), 仅噪声随 seed 变
    env = ResourceCalendarEnv(
        encoder, top_k=8, w_lookahead=W_LOOKAHEAD,
        process_noise_enabled=True, noise_seed=seed,
    )
    driver = _driver(env, MAX_STEPS)
    driver.reset_episode()
    cfg = VCMCTSConfig(
        rollout_strategy="FIFO",
        arrival_prob_weighting=weighting,
        arrival_prob_decay=DECAY,
        lookahead_window=W_LOOKAHEAD,
        **LIGHT,
    )
    planner = VCMCTSPlanner(cfg)
    summary = run_vc_mcts_reservation_episode(driver, planner=planner, max_steps=MAX_STEPS)
    metrics = schedule_metrics_with_priority_wait(encoder, env)
    made = int(summary.get("reservations_made", 0))
    consumed = int(summary.get("reservations_consumed", 0))
    return dict(
        seed=seed,
        qtime=float(metrics["qtime_violation_count"]),
        o2=float(metrics["priority_weighted_wait"]),
        util=float(metrics["avg_utilization"]),
        made=made,
        consumed=consumed,
        waste=made - consumed,
        waste_rate=(made - consumed) / made if made else 0.0,
        completed=int(metrics["completed_lots"]),
    )


def agg(rows):
    q = [r["qtime"] for r in rows]
    return dict(
        qtime_mean=st.mean(q),
        qtime_max=max(q),              # 尾部 = 最坏种子
        o2_mean=st.mean(r["o2"] for r in rows),
        util_mean=st.mean(r["util"] for r in rows),
        made_mean=st.mean(r["made"] for r in rows),
        consumed_mean=st.mean(r["consumed"] for r in rows),
        waste_mean=st.mean(r["waste"] for r in rows),
        waste_rate_mean=st.mean(r["waste_rate"] for r in rows),
    )


def main():
    print(f"# 方向2 最小验证  seeds={SEEDS}  decay={DECAY}  window={W_LOOKAHEAD}  config={LIGHT}")
    results = {}
    for label, weighting in [("A 确定性预留(现状)", False), ("B 概率加权预留", True)]:
        rows = []
        for seed in SEEDS:
            r = run_one(weighting, seed)
            rows.append(r)
            print(f"[{label}] seed {seed}: qtime={r['qtime']:.0f} o2={r['o2']:.0f} "
                  f"util={r['util']:.3f} made={r['made']} consumed={r['consumed']} "
                  f"waste={r['waste']}({r['waste_rate']*100:.0f}%) done={r['completed']}/50",
                  flush=True)
        results[label] = (rows, agg(rows))

    print("\n========== 汇总 (越小越好: qtime/o2/waste; 越大越好: util) ==========")
    hdr = f"{'策略':<20}{'qtime均值':>10}{'qtime尾部':>10}{'O2均值':>10}{'util均值':>10}{'预留均值':>10}{'兑现均值':>10}{'浪费率':>9}"
    print(hdr)
    for label, (_rows, a) in results.items():
        print(f"{label:<20}{a['qtime_mean']:>10.1f}{a['qtime_max']:>10.0f}{a['o2_mean']:>10.0f}"
              f"{a['util_mean']:>10.3f}{a['made_mean']:>10.1f}{a['consumed_mean']:>10.1f}"
              f"{a['waste_rate_mean']*100:>8.0f}%")


if __name__ == "__main__":
    main()
