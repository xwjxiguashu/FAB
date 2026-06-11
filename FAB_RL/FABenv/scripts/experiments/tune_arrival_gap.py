"""扫描 arrival_mean_gap，给困难压力实例定稿到达强度。

目标：找到一个 gap，使 util 由派工质量（而非到达饥饿）决定——好规则利用率有余量
（~0.75–0.88，不饱和不饿死），且 Q-time 仍在规则间有区分度。
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
import functools
import multiprocessing as mp
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from problem_instances import build_pressure_test_encoder
from evaluate_baselines import run_rule_seed

GAPS = [0.4, 0.6, 0.8]
QTIME = 3.0
RULES = ["FIFO", "SPT", "EDD", "CR", "ATC"]
SEEDS = [0]


def run_job(job):
    gap, rule, seed = job
    factory = functools.partial(
        build_pressure_test_encoder, qtime_limit=QTIME, arrival_mean_gap=gap,
    )
    m = run_rule_seed(factory, rule, seed, noise=True)
    return gap, rule, seed, m


def main():
    jobs = [(g, r, s) for g in GAPS for r in RULES for s in SEEDS]
    results = {}
    ctx = mp.get_context("spawn")
    with ctx.Pool(6) as pool:
        for gap, rule, seed, m in pool.imap_unordered(run_job, jobs):
            results.setdefault(gap, {})[rule] = m
            print(f"[done] gap={gap} {rule} seed={seed}: "
                  f"util={m['avg_utilization']:.3f} qv={m['qtime_violation_count']:.0f}",
                  flush=True)

    lines = [f"\n=== arrival_mean_gap 扫描 (qtime_limit={QTIME}, {len(SEEDS)} seed) ==="]
    for g in GAPS:
        lines.append(f"-- gap={g} --")
        for r in RULES:
            m = results[g][r]
            lines.append(
                f"  {r:5} util={m['avg_utilization']:.3f}  "
                f"qtime_viol={m['qtime_violation_count']:.0f}  "
                f"tardy={m['total_tardiness']:.0f}  "
                f"priority={m['priority_violation']:.0f}"
            )
    out = "\n".join(lines)
    print(out)
    results_dir = Path(__file__).resolve().parents[2] / "artifacts" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "tune_arrival_gap.txt", "w", encoding="utf-8") as f:
        f.write(out + "\n")


if __name__ == "__main__":
    main()
