# -*- coding: utf-8 -*-
"""Q-time 链 最小验证：聚合代理 mask vs 链感知 mask (late_hi, noise, FIFO).

隔离 mask 效果：用最简单的 FIFO 派工(无 VC 搜索),只切换 env.qtime_mask_mode。
A = aggregate(现状): mask 比"单一聚合 deadline 代理 vs 总完成 μ"
B = chain: mask 对每个候选 dry-run, 用真实 compute_q_time_violation 评估阶段链 (1,2)/(2,3)
看 B 能不能把 Q-time 违规(均值 + 尾部=最坏种子)进一步压向 0。
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

import sys

SEEDS = [0, 1, 2, 3, 4]
MAX_STEPS = 500
INSTANCE = sys.argv[1] if len(sys.argv) > 1 else "late_hi"


def run_one(mode, seed):
    encoder = _encoder_factory(INSTANCE)()
    env = ResourceCalendarEnv(
        encoder, top_k=8, w_lookahead=4.0,
        process_noise_enabled=True, noise_seed=seed,
    )
    env.qtime_mask_mode = "chain_joint"
    driver = _driver(env, MAX_STEPS)
    driver.reset_episode()
    driver.run_rule_episode(strategy="FIFO")
    m = schedule_metrics_with_priority_wait(encoder, env)
    return dict(
        seed=seed,
        qtime=float(m["qtime_violation_count"]),
        qtime_total=float(m["qtime_violation_total"]),
        o2=float(m["priority_weighted_wait"]),
        util=float(m["avg_utilization"]),
        completed=int(m["completed_lots"]),
    )


def main():
    print(f"# Q-time 链 最小验证 (FIFO, {INSTANCE}, noise)  seeds={SEEDS}")
    results = {}
    for label, mode in [("A 聚合代理(现状)", "aggregate"), ("B 链感知 mask", "chain")]:
        rows = [run_one(mode, s) for s in SEEDS]
        for r in rows:
            print(f"[{label}] seed {r['seed']}: qtime={r['qtime']:.0f} "
                  f"qtime_total={r['qtime_total']:.1f} o2={r['o2']:.0f} "
                  f"util={r['util']:.3f} done={r['completed']}/50", flush=True)
        results[label] = rows

    print("\n========== 汇总 (越小越好: qtime/o2; 越大越好: util) ==========")
    print(f"{'策略':<18}{'qtime均值':>10}{'qtime尾部':>10}{'qtime_total均值':>16}{'O2均值':>10}{'util均值':>10}")
    for label, rows in results.items():
        q = [r["qtime"] for r in rows]
        print(f"{label:<18}{st.mean(q):>10.2f}{max(q):>10.0f}"
              f"{st.mean(r['qtime_total'] for r in rows):>16.2f}"
              f"{st.mean(r['o2'] for r in rows):>10.0f}"
              f"{st.mean(r['util'] for r in rows):>10.3f}")


if __name__ == "__main__":
    main()
