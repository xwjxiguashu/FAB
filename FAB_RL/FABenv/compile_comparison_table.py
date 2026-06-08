"""Compile final comparison table from vc_mcts_probe JSON outputs + existing baseline data."""
import json
import numpy as np

# ---------------------------------------------------------------------------
# Existing baseline data (from phase5b_dda_table_late_hi_wq0.md, 5 seeds noise on)
# ---------------------------------------------------------------------------
BASELINES_5SEED = {
    "FIFO":     dict(qtime=13.60, qtime_std=6.86, o2=1612.53, o2_std=83.92,  util=0.82, util_std=0.02, o1=169.39, o1_std=7.43),
    "EDD":      dict(qtime=13.60, qtime_std=6.86, o2=1612.53, o2_std=83.92,  util=0.82, util_std=0.02, o1=169.39, o1_std=7.43),
    "CR":       dict(qtime=34.00, qtime_std=7.69, o2=4326.11, o2_std=58.29,  util=0.87, util_std=0.02, o1=756.67, o1_std=19.60),
    "ATC":      dict(qtime=7.20,  qtime_std=3.25, o2=1864.26, o2_std=47.02,  util=0.87, util_std=0.03, o1=245.68, o1_std=6.02),
    "SAS-only": dict(qtime=5.60,  qtime_std=2.33, o2=1685.28, o2_std=23.65,  util=0.86, util_std=0.01, o1=231.12, o1_std=7.60),
}

# Oracle: from oracle_reservation_late_hi_verdict.md (3 seeds, noise on)
ORACLE = dict(qtime=2.0, o2=1424.5, util=0.784, o2_pct=-12.7)

def load_probe_results(json_path):
    """Load vc_mcts_probe stdout JSON → per-seed rows."""
    with open(json_path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    return rows


def aggregate_probe_rows(rows, key="vc_mcts"):
    """Aggregate per-seed metrics for a given key (baseline or vc_mcts)."""
    qtime_vals = [r[key]["qtime_violation_count"] for r in rows]
    o2_vals    = [r[key]["priority_weighted_wait"] for r in rows]
    util_vals  = [r[key]["avg_utilization"] for r in rows]
    completed  = [r[key]["completed_lots"] for r in rows]
    return dict(
        qtime=np.mean(qtime_vals), qtime_std=np.std(qtime_vals),
        o2=np.mean(o2_vals),       o2_std=np.std(o2_vals),
        util=np.mean(util_vals),   util_std=np.std(util_vals),
        completed=np.mean(completed),
        n=len(rows),
    )


def fmt_mean_std(mean, std, fmt=".1f"):
    return f"{mean:{fmt}}±{std:{fmt}}"


def print_table(results):
    """Print markdown comparison table."""
    headers = ["策略", "Q-time违规↓", "O2加权等待↓", "O2改善vs FIFO", "avg利用率↑", "完成率"]
    col_w = [22, 16, 18, 16, 14, 8]

    # header
    h = "|" + "|".join(f" {headers[i]:<{col_w[i]}} " for i in range(len(headers))) + "|"
    sep = "|" + "|".join("-" * (col_w[i] + 2) for i in range(len(headers))) + "|"
    print(h)
    print(sep)

    fifo_o2 = results["FIFO"]["o2"]

    for name, r in results.items():
        delta_pct = (r["o2"] - fifo_o2) / fifo_o2 * 100
        sign = "+" if delta_pct > 0 else ""
        cells = [
            name,
            fmt_mean_std(r["qtime"], r.get("qtime_std", 0), ".1f"),
            fmt_mean_std(r["o2"], r.get("o2_std", 0), ".1f"),
            f"{sign}{delta_pct:.1f}%",
            fmt_mean_std(r["util"], r.get("util_std", 0), ".3f"),
            f"{r.get('completed', 50):.0f}/50",
        ]
        row = "|" + "|".join(f" {cells[i]:<{col_w[i]}} " for i in range(len(cells))) + "|"
        print(row)


def main():
    import sys

    rule_path = "results/vc_mcts_late_hi_rule_5seed.json"
    sas_path  = "results/vc_mcts_late_hi_sas_5seed.json"

    results = dict(BASELINES_5SEED)

    # Load VC-MCTS results if available
    try:
        rule_rows = load_probe_results(rule_path)
        results["VC-MCTS+rule"] = aggregate_probe_rows(rule_rows, "vc_mcts")
        # Also re-compute FIFO from probe baseline to cross-check
        fifo_from_probe = aggregate_probe_rows(rule_rows, "baseline")
        print(f"[cross-check] FIFO from probe: O2={fifo_from_probe['o2']:.1f} qtime={fifo_from_probe['qtime']:.1f}")
        print()
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"[warn] rule probe results not ready: {e}", file=sys.stderr)

    try:
        sas_rows = load_probe_results(sas_path)
        results["VC-MCTS+SAS"] = aggregate_probe_rows(sas_rows, "vc_mcts")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"[warn] SAS probe results not ready: {e}", file=sys.stderr)

    # Add Oracle (3-seed)
    results["Oracle(3seed)"] = dict(
        qtime=ORACLE["qtime"], qtime_std=0,
        o2=ORACLE["o2"], o2_std=0,
        util=ORACLE["util"], util_std=0,
        completed=50,
    )

    print("### 基线对比表 (instance=late_hi, noise=on)")
    print("#### 规则基线来自 5 seeds；VC-MCTS 来自新 5-seed 运行；Oracle 来自 3-seed windowed oracle")
    print()
    print_table(results)

    # Also print raw seed-level VC-MCTS data if available
    try:
        print()
        print("#### VC-MCTS + rule delegate per-seed")
        print("| seed | baseline O2 | VC O2 | Δ O2 | baseline qtime | VC qtime | reservations | reserve率 |")
        print("|---|---|---|---|---|---|---|---|")
        for r in rule_rows:
            d = r["delta"]
            print(f"| {r['seed']} | {r['baseline']['priority_weighted_wait']:.1f} | "
                  f"{r['vc_mcts']['priority_weighted_wait']:.1f} | {d['vc_mcts_o2']:+.1f} | "
                  f"{r['baseline']['qtime_violation_count']:.0f} | {r['vc_mcts']['qtime_violation_count']:.0f} | "
                  f"{r['vc_mcts'].get('reservations_made', '?')} | "
                  f"{r['vc_mcts'].get('reserve_selection_rate', '?')} |")
    except Exception:
        pass

    try:
        print()
        print("#### VC-MCTS + SAS delegate per-seed")
        print("| seed | baseline O2 | VC O2 | Δ O2 | baseline qtime | VC qtime | reservations | reserve率 |")
        print("|---|---|---|---|---|---|---|---|")
        for r in sas_rows:
            d = r["delta"]
            print(f"| {r['seed']} | {r['baseline']['priority_weighted_wait']:.1f} | "
                  f"{r['vc_mcts']['priority_weighted_wait']:.1f} | {d['vc_mcts_o2']:+.1f} | "
                  f"{r['baseline']['qtime_violation_count']:.0f} | {r['vc_mcts']['qtime_violation_count']:.0f} | "
                  f"{r['vc_mcts'].get('reservations_made', '?')} | "
                  f"{r['vc_mcts'].get('reserve_selection_rate', '?')} |")
    except Exception:
        pass


if __name__ == "__main__":
    main()
