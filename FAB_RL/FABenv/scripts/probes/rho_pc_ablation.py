"""Run mechanism-2 alpha scan for VC-MCTS rho_pc (报告8 §7.12.5).

Configs: OFF (use_rho_pc=False 旧行为) + alpha ∈ {1.0, 0.8, 0.6, 0.4}.
Each config runs the vc_mcts_probe on late_hi with the rule dispatch delegate
(no checkpoint dependency) and writes per-config metrics JSON + trace +
trace summary under artifacts/results/rho_pc_ablation/.
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
import json
import subprocess

PROBE = FABENV_ROOT / "scripts" / "probes" / "vc_mcts_probe.py"
OUT_DIR = FABENV_ROOT / "artifacts" / "results" / "rho_pc_ablation"

CONFIGS = [
    ("off", None),
    ("alpha_1.0", 1.0),
    ("alpha_0.6", 0.6),
    ("alpha_0.4", 0.4),
]


def main(seeds=3, n_iter=4, top_b=2, rollout_max_steps=60, workers=3):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, alpha in CONFIGS:
        summary_path = OUT_DIR / f"late_hi_{label}_summary.json"
        trace_path = OUT_DIR / f"late_hi_{label}_trace.jsonl"
        result_path = OUT_DIR / f"late_hi_{label}_rows.json"
        if trace_path.exists():
            trace_path.unlink()
        cmd = [
            sys.executable,
            str(PROBE),
            "--instance", "late_hi",
            "--seeds", str(seeds),
            "--strategy", "FIFO",
            "--skip-oracle",
            "--n-iter", str(n_iter),
            "--top-b", str(top_b),
            "--rollout-max-steps", str(rollout_max_steps),
            "--max-decisions", "150",
            "--qtime-mask-mode", "aggregate",
            "--workers", str(workers),
            "--dispatch-delegate", "rule",
            "--trace-out", str(trace_path),
            "--trace-summary-out", str(summary_path),
        ]
        if alpha is not None:
            cmd += ["--use-rho-pc", "--rho-pc-alpha", str(alpha)]
        print(f"[rho_pc_ablation] running {label} ...", file=sys.stderr, flush=True)
        proc = subprocess.run(
            cmd, cwd=str(FABENV_ROOT), check=True, capture_output=True, text=True
        )
        result_path.write_text(proc.stdout, encoding="utf-8")
        rows.append(
            {
                "label": label,
                "alpha": alpha,
                "rows": str(result_path),
                "summary": str(summary_path),
                "trace": str(trace_path),
            }
        )
    manifest = OUT_DIR / "manifest.json"
    manifest.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(str(manifest))


def report():
    """Aggregate per-config rows/summaries into one comparison table."""
    mean = lambda xs: sum(xs) / max(len(xs), 1)
    header = (
        f"{'config':<10} {'baseO2':>9} {'vcO2':>9} {'O2%':>7} {'baseQt':>7} {'vcQt':>5} "
        f"{'util':>6} {'resv':>5} {'resvRate':>8} {'dRhoAvg':>8} {'dRhoSel':>8}"
    )
    print(header)
    print("-" * len(header))
    for label, _alpha in CONFIGS:
        rows_path = OUT_DIR / f"late_hi_{label}_rows.json"
        if not rows_path.exists():
            print(f"{label:<10} (pending)")
            continue
        rows = json.loads(rows_path.read_text(encoding="utf-8"))
        base_o2 = mean([r["baseline"]["priority_weighted_wait"] for r in rows])
        vc_o2 = mean([r["vc_mcts"]["priority_weighted_wait"] for r in rows])
        base_qt = mean([r["baseline"]["qtime_violation_count"] for r in rows])
        vc_qt = mean([r["vc_mcts"]["qtime_violation_count"] for r in rows])
        util = mean([r["vc_mcts"]["avg_utilization"] for r in rows])
        resv = mean([r["vc_mcts"].get("reservations_made", 0) for r in rows])
        o2_pct = 100.0 * (vc_o2 - base_o2) / max(base_o2, 1e-9)

        # 多 seed 且 workers>1 时 summary 按 seed 切分 (…_seed<k>_summary.json)
        summary_paths = sorted(OUT_DIR.glob(f"late_hi_{label}_seed*_summary.json"))
        single = OUT_DIR / f"late_hi_{label}_summary.json"
        if not summary_paths and single.exists():
            summary_paths = [single]
        rates, d_avgs, d_sels = [], [], []
        for path in summary_paths:
            summary = json.loads(path.read_text(encoding="utf-8"))
            if summary.get("reserve_selection_rate_when_available") is not None:
                rates.append(summary["reserve_selection_rate_when_available"])
            if summary.get("rho_pc_delta_avg") is not None:
                d_avgs.append(summary["rho_pc_delta_avg"])
            if summary.get("rho_pc_selected_reserve_delta_avg") is not None:
                d_sels.append(summary["rho_pc_selected_reserve_delta_avg"])
        fmt = lambda xs: "-" if not xs else f"{mean(xs):.3f}"
        print(
            f"{label:<10} {base_o2:>9.1f} {vc_o2:>9.1f} {o2_pct:>+7.1f} {base_qt:>7.1f} "
            f"{vc_qt:>5.1f} {util:>6.3f} {resv:>5.1f} {fmt(rates):>8} {fmt(d_avgs):>8} "
            f"{fmt(d_sels):>8}"
        )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        report()
    else:
        main()
