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
    ("alpha_0.8", 0.8),
    ("alpha_0.6", 0.6),
    ("alpha_0.4", 0.4),
]


def main(seeds=3, n_iter=8, top_b=2, rollout_max_steps=60, workers=3):
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


if __name__ == "__main__":
    main()
