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

## Results (2026-06-10, late_hi, FIFO rule delegate, aggregate mask, 3 seeds, n_iter=4)

| config | vc O2 (mean) | per-seed O2 | O2% vs FIFO | Q-time | util | resv rate |
|---|---:|---|---:|---:|---:|---:|
| off | 1078.4 | 1024.3 / **1186.6** / 1024.3 | −35.2% | 0 | 0.798 | 0.357 |
| alpha_1.0 | 1006.0 | 1024.3 / 969.6 / 1024.3 | −39.6% | 0 | 0.800 | 0.355 |
| alpha_0.6 | 1024.3 | 1024.3 / 1024.3 / 1024.3 | −38.5% | 0 | 0.803 | 0.370 |
| alpha_0.4 | 1024.3 | 1024.3 / 1024.3 / 1024.3 | −38.5% | 0 | 0.803 | 0.370 |

Findings:
1. **Harmlessness holds, with a small gain**: every mechanism-on config beats
   OFF on mean O2 (1078.4 → 1006.0–1024.3), keeps Q-time at 0, completes 50/50,
   and slightly raises utilization. 退化检验通过。
2. **alpha<1 removes seed-level wobble**: OFF/alpha_1.0 vary across seeds
   (1186.6 worst-case at OFF); alpha_0.6/0.4 pin all three seeds to the same
   schedule (variance → 0). ρ̂_pc acts as a deterministic structural
   tie-breaker in visit allocation.
3. **Degenerate-chain prediction confirmed (报告8 §7.12.2 性质 3)**: late_hi is
   capability-homogeneous (every lot compatible with all 10 machines), so
   Δρ_pc ≈ 0 on every edge (float dust ~1e-18) — the matching correctly reports
   "no capability-scarcity hedging leverage here". Contrast run on the
   heterogeneous `small` instance (`rho_pc_small_*`): dispatch edges show real
   waterline drops (delta = −0.571 when dispatching the only machine compatible
   with an upcoming high-priority lot), reserve edges hold the line at 0.
   Demonstrating the full decision-level power of mechanism 2 needs a
   capability-scarce late_hi variant (报告8 §12.2 相变扫描 structural knob).

## Results (2026-06-11, late_hi_scarce, eligibility_density=0.3, PYTHONHASHSEED=0, aggregate mask, rule delegate, 2 seeds)

Same deterministic FIFO baseline for all rows: O2 = 3306.3, Q-time = 18, util = 0.829.

| config | VC O2 | O2% | VC Q-time | util | resv |
|---|---:|---:|---:|---:|---:|
| VC, mechanism 2 OFF | 2910.7 / 2874.3 (per seed) | -12.0% / -13.1% | **28 / 18** | 0.745 / 0.817 | 13 |
| VC, mechanism 2 ON (alpha=0.6) | 2939.2 (both seeds) | -11.1% | **12** | 0.766 | 12 |

Findings (the leverage-instance test 报告8 §7.12.2 性质 2 was waiting for):
1. **OFF harms the hard constraint under capability scarcity**: blind reserve/dispatch
   guidance pushes Q-time 18 → 28 on seed 0. ON pushes it 18 → **12** (-33% vs
   baseline) on both seeds — lexicographically (Q-time first) ON strictly
   dominates both OFF and FIFO: (12, 2939) < (18, 2874) < (28, 2911).
2. The O2 cost of that hard-constraint gain is ~1.5pp of improvement
   (-11.1% vs -12/-13%) — exactly the intended dictionary-order trade.
3. Δρ_pc is now genuinely nonzero (delta_avg = -0.013): dispatch displacement
   is visible on the scarce instance, unlike capability-homogeneous late_hi.
4. ON also pins both seeds to one schedule (the deterministic ρ̂_pc tie-break),
   removing the cross-process wobble that turned out to be the PYTHONHASHSEED
   reproducibility bug (str-hash-order-dependent iteration somewhere in the
   scheduling path — separate debugging task; fix the env var for all
   multi-worker comparisons until located).

## CORRECTION (2026-06-11, after the estimate-rng reproducibility fix)

The late_hi_scarce table above is **retracted**. Those numbers were produced
before the estimate() unseeded-rng bug was fixed: the OFF/ON "difference"
(Q-time 28/18 vs 12) was per-process sampling drift of the estimator stream,
not an effect of mechanism 2. Post-fix deterministic rerun (baseline now
O2 3244.3 / Q-time 14, bit-identical across seeds and processes):

| config | VC O2 | VC Q-time | util | resv |
|---|---:|---:|---:|---:|
| VC, mechanism 2 OFF | 2910.7 (-10.3%) | 28 | 0.745 | 13 |
| VC, mechanism 2 ON (alpha=0.6) | 2910.7 (-10.3%) | 28 | 0.745 | 13 |

ON and OFF now produce **identical schedules**. With the root-level planner and
deterministic per-edge evaluation, edge means are fixed after warm-up, so the
alpha-interpolated UCT guidance can only act through the final-pick visit
tie-break — and no tie was flipped in 150 decisions on this instance. The
honest status of mechanism 2 as currently implemented:
- it is **harmless** (never degrades the lexicographic objective), and
- its delta_rho_pc trace fields are a real, replayable leverage diagnostic
  (nonzero only under capability scarcity, per §7.12.2 property 2/3),
- but it has **no decision channel** under deterministic evaluation. To give it
  one: (a) enable mechanism-3 noisy CRN evaluation so visit allocation affects
  estimate quality, or (b) add a controlled rho_pc tie-break layer to the
  lexicographic final pick (design change), or (c) keep it as diagnostics +
  groundwork for a deeper tree. Decision pending.
