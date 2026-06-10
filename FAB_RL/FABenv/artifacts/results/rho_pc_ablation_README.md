# Rho PC Ablation (机制 2, 报告8 §7.12.5)

This folder stores mechanism-two alpha-scan outputs produced by
`scripts/probes/rho_pc_ablation.py` (late_hi, FIFO rule delegate, 3 seeds).

Configs:
- `off`: `use_rho_pc=False` — 旧行为基线 (raw-score UCT exploitation).
- `alpha_1.0`: mechanism on, exploitation = normalized q_hat only (天然消融基线).
- `alpha_0.8 / 0.6 / 0.4`: interpolation toward the matching waterline `rho_pc_after`.

Primary diagnostics:
- O2 / Q-time deltas vs FIFO baseline from `late_hi_<label>_rows.json`.
- `rho_pc_edge_count`, `rho_pc_positive_delta_edges`, `rho_pc_delta_avg`,
  `rho_pc_selected_reserve_delta_avg` from `late_hi_<label>_summary.json`.
- Reserve selection-rate / O2-gap fields from the existing trace summary.

Interpretation note (architecture constraint): the current planner is
root-level with deterministic per-edge rollouts (cloned RNG), so edge means are
fixed after the warm-up visit and UCT guidance can only change the final pick
through the visit tie-break. The decision-level claim to check is therefore
**harmlessness** (alpha<1 not worse than alpha=1/off, 报告8 §7.12.5 退化检验);
the mechanism's positive evidence is the waterline diagnostics (Δρ_pc buckets
vs reserve selection) which the trace summaries expose.
