"""Aggregate mechanism A/B/C probe JSON into a comparison table."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

RUNS = [
    ("A: deterministic rollout", "A_baseline_noise.json"),
    ("B: + 机制3 CRN (n_mc=4)", "B_crn.json"),
    ("C: + 机制2 ρ_pc + 机制3 CRN", "C_crn_rhopc.json"),
]


def load(path):
    p = HERE / path
    if not p.exists() or p.stat().st_size == 0:
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def summarize(rows):
    n = len(rows)
    base_o2 = [r["baseline"]["priority_weighted_wait"] for r in rows]
    base_qt = [r["baseline"]["qtime_violation_count"] for r in rows]
    vc_o2 = [r["vc_mcts"]["priority_weighted_wait"] for r in rows]
    vc_qt = [r["vc_mcts"]["qtime_violation_count"] for r in rows]
    completed = [r["vc_mcts"]["completed_lots"] for r in rows]
    reservations = [r["vc_mcts"].get("reservations_made", 0) for r in rows]
    util = [r["vc_mcts"]["avg_utilization"] for r in rows]
    elapsed = [r.get("_elapsed_s", 0.0) for r in rows]
    mean = lambda xs: sum(xs) / max(len(xs), 1)
    return {
        "n": n,
        "baseline_o2": mean(base_o2),
        "baseline_qt": mean(base_qt),
        "vc_o2": mean(vc_o2),
        "vc_qt": mean(vc_qt),
        "o2_improve_pct": 100.0 * (mean(vc_o2) - mean(base_o2)) / max(mean(base_o2), 1e-9),
        "completed": mean(completed),
        "reservations": mean(reservations),
        "util": mean(util),
        "elapsed_s": mean(elapsed),
        "per_seed_vc_o2": vc_o2,
    }


def main():
    print(f"{'run':<32} {'baseQt':>7} {'vcQt':>5} {'baseO2':>9} {'vcO2':>9} "
          f"{'O2%':>7} {'compl':>6} {'resv':>5} {'util':>6} {'sec':>6}")
    print("-" * 105)
    for label, path in RUNS:
        rows = load(path)
        if rows is None:
            print(f"{label:<32} (pending)")
            continue
        s = summarize(rows)
        print(f"{label:<32} {s['baseline_qt']:>7.1f} {s['vc_qt']:>5.1f} "
              f"{s['baseline_o2']:>9.1f} {s['vc_o2']:>9.1f} {s['o2_improve_pct']:>+7.1f} "
              f"{s['completed']:>6.1f} {s['reservations']:>5.1f} {s['util']:>6.3f} "
              f"{s['elapsed_s']:>6.0f}")
        print(f"{'  per-seed vc O2:':<32} "
              f"{', '.join(f'{x:.1f}' for x in s['per_seed_vc_o2'])}")


if __name__ == "__main__":
    main()
