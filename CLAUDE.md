# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

A semiconductor FAB (wafer fab) machine-group scheduling research prototype. The core problem: given a set of Lots (wafer batches), Machines, PPIDs (process recipes), and Chamber/Side resources, decide which Lot to dispatch to which Machine with which PPID, and when — while satisfying Q-time hard constraints and maximizing utilization.

The active development lives entirely in `FAB_RL/FABenv/`. The root-level `core.py`, `__init__.py`, and `MAMHSA_for_fjsp-master/` are reference/legacy code — do not modify them when working on the RL environment.

## Commands

All commands must be run from `FAB_RL/FABenv/` as the working directory. The package uses bare imports (no package prefix), so the working directory must be on `sys.path`.

```powershell
# Run all tests
cd FAB_RL\FABenv; python -m pytest tests/ -v

# Run a single test file
python -m pytest tests/test_phase2_reward.py -v

# Run a single test by name
python -m pytest tests/test_phase2_sas_policy.py -v -k "test_sample_returns_valid_action"

# Run the Phase 1 environment demo
python run_phase1_environment_demo.py

# Run the Phase 2 inference demo
python run_phase2_sas_inference_demo.py

# Launch PPO training (the flag is --mode, NOT --instance; default mode=small, episodes=3)
python train_phase2_sas_ppo.py --mode small --episodes 200
# Modes: small | random | pressure | multihead
#   small/random/pressure → single-head Phase2PPOTrainer (scalar reward)
#   multihead             → MultiHeadPPOTrainer (per-channel vector reward)
# Optional: --tensorboard-logdir <dir>  --save-path <model.pt>
# Device: --device auto|cpu|cuda (auto = CUDA if available else CPU; resolve_device())
```

GPU note: training auto-selects device via `resolve_device()` (policy `.to(device)`; the trainers/driver already follow the policy's device, so no other change is needed). The installed torch must be a CUDA build for `--device cuda`/`auto` to use the GPU — a plain `2.x.y+cpu` wheel reports `cuda.is_available()=False`. **GPU does not speed up these runs**: the bottleneck is the CPU environment simulation (candidate-pool dry-run / commit), and the networks are tiny — GPU helps only the (negligible) net forward/backward.

Parallel rollout (`parallel_rollout.py`, multihead only): `--parallel N` spawns N persistent CPU worker processes (`ParallelRolloutCollector`, `multiprocessing` spawn), each running one episode per iteration with a broadcast copy of the policy; the main process pools all N episodes' steps into **one** `MultiHeadRolloutBuffer` (GAE resets at each episode's `done=True`, so concatenation is correct) and does one PPO update. This is the real training speedup since the env sim is CPU-bound and embarrassingly parallel across episodes — ~3.5× on 4 workers (one serial pressure RL episode ≈ 106s; 4 in parallel ≈ 109s). Workers are pinned to 1 BLAS/OpenMP thread (env vars set at `parallel_rollout` import + `torch.set_num_threads(1)`) to avoid oversubscription. The Lagrangian `mean_violation` uses the per-iteration mean over N episodes (a better `Ê[violation]` estimate). `--save-every N` checkpoints periodically so a long run survives interruption (training does not resume-from-checkpoint — a relaunch starts fresh).

Dependencies: `numpy`, `torch`, `scipy`, `pytest`. No `requirements.txt` exists; install manually.

## Architecture

### Two-Layer Design (from 项目报告.md)

The system is architecturally split into two non-overlapping layers:

**Lower layer (fixed rules, non-RL):** Given a committed dispatch `(lot, machine, ppid)`, the lower layer computes how the wafers flow through internal stages — batching, list scheduling, and (new) Monte Carlo timing estimation. Lives in `lower_layer_estimator.py` and the calendar simulation inside `rl_environment.py`. This layer never learns; it only estimates.

**Upper layer (RL):** Decides *which* `(lot, ppid)` to dispatch to the current idle machine and *when* (DDT agent, Phase 5+). SAS (dispatch selection) is the trained agent. Lives in the `phase2_*` modules. Two policy/trainer variants coexist: a **single-head** path (scalar reward, `Phase2SASActorCritic` + `Phase2PPOTrainer`) and a **multi-head** path (vector reward, `Phase2SASMultiHeadActorCritic` + `MultiHeadPPOTrainer`) selected by `--mode multihead`.

### Core Data Flow

```
encoder (Phase1CalendarProblem)
    └── ResourceCalendarEnv          ← RL environment
          ├── build_candidate_pool() ← generates CandidatePool (K_action fixed-size)
          ├── dry_run_action()       ← non-destructive feasibility check (state copy)
          ├── commit_action_index()  ← writes to real state
          └── sas_step()            ← one RL step: dry-run → commit → reward
                    ↓
          Phase2EpisodeDriver        ← orchestrates machine selection + event advance
                    ↓
          Phase2RolloutBuffer        ← stores (obs, action, log_prob, reward, done)
                    ↓
          Phase2PPOTrainer           ← GAE + clipped PPO update
```

### Key Classes

**`Phase1CalendarProblem`** (`problem_instances.py`) — the problem encoder. Multiple-inherits `ProblemDefinitionMixin` (data: lots, machines, PPIDs, stages, q-time limits) and `CalendarDecoderMixin` (calendar interval operations: insert, find-slot, conflict-check). This is the object passed everywhere as `encoder`.

**`ResourceCalendarEnv`** (`rl_environment.py`) — the RL gym-like environment. Holds the live `ScheduleState` (calendars, completed lots). Candidate pool pipeline: generate structurally feasible actions → (new) qtime-safe mask → priority filter → TopK score → pad to fixed size `K_action`.

**`ScheduleState`** (`state.py`) — mutable state: `machine_calendar`, `chamber_calendar`, `machine_available_time`, `chamber_available_time`, `commit_log`. Shared between Phase 1 and legacy root code.

**`Phase2SASActorCritic`** (`phase2_sas_policy.py`) — MLP candidate encoder (18-dim → hidden) + actor head (per-candidate logit) + critic head (masked pooling + global features → scalar value).

**`Phase2SASMultiHeadActorCritic`** (`phase2_sas_policy.py`) — multi-head critic variant. Shares the encoder/actor structure but `critic_values()` returns a `{channel: value}` dict over four reward channels: `exec` (immediate dense), `qtime` (terminal, hard-constraint residual = tardiness), `util` (terminal, the only soft objective), `progress` (terminal, utilization proxy). Trained per-channel by `MultiHeadPPOTrainer` with channel advantages combined via `combine_channel_advantages()` (`MultiHeadPPOConfig` weights — e.g. `w_qtime` large to encode hard-constraint priority). The reward vector itself is produced by the env / `RewardConfig` plumbing in `rl_environment.py`.

**PPO-Lagrangian for the Q-time residual (报告 §3.3)** — `MultiHeadPPOTrainer` can treat the Q-time constraint as a CMDP instead of a fixed-weight penalty. When `MultiHeadPPOConfig.use_qtime_lagrangian=True`, the `qtime` channel's advantage weight becomes an adaptive Lagrange multiplier `self.lambda_qtime` (via `qtime_weight()`) instead of the fixed `w_qtime`. Dual ascent runs once per episode in `train()`: `update_lambda(mean_violation)` does `λ ← clip(max(0, λ + η_λ·(violation − ε)), 0, λ_max)`, where the per-episode violation rate is recovered from the buffer by `episode_qtime_cost()` (`= -Σ_t reward_vector[qtime]`, since `r_qtime = -violation_count/num_lots` is nonzero only at the terminal step). Config knobs: `qtime_lambda_init`, `qtime_cost_budget` (ε), `qtime_lambda_lr` (η_λ, keep ≪ policy lr), `qtime_lambda_max`. CLI: `python train_phase2_sas_ppo.py --mode multihead --qtime-lagrangian --qtime-budget 0.02 --qtime-lambda-lr 0.1`. The fixed-weight and Lagrangian paths act on the **same** `qtime` channel and are never combined. History rows gain `qtime_cost` and `lambda_qtime`.

### Schedule Data Schemas

- `lot_schedule`: numpy `(n, 5)` — columns: `[lot, machine, ppid, start_time, end_time]`
- `wafer_schedule`: numpy `(n, 9)` — columns: `[lot, wafer_id, machine, ppid, stage_id, chamber, side, start_time, end_time]`

### Candidate Pool (18 features, index 0–17)

Features per candidate action: `is_real`, `is_wait`, `score`, `arrival_time`, `waiting_time`, `machine_slot_start`, `machine_load`, `total_process_time`, `predicted_completion`, `stage_count`, `qtime_risk`, `wafer_count`, `priority`, `due_slack`, `priority_rank_norm`, `due_slack_rank_norm`, `is_best_priority`, `is_most_urgent_due`.

Mask convention: `True` = valid action (real or wait); `False` = padding. Padding positions get `masked_fill(-inf)` in the actor's softmax.

### Three-Tier Constraint Structure (报告 Section 1.4)

The candidate pool pipeline enforces a strict dictionary order:
1. **Q-time/tardiness (hard):** qtime-safe mask screens actions that would make `deadline − μ_finish < z_ε·σ_finish`
2. **Priority (strong):** priority filter keeps only highest-priority lots in the safe set (soft mode = reorder only, strict mode = remove lower-priority)
3. **Utilization (soft):** CandidateScore is the only place where utilization/urgency trade-offs are made

This order is enforced structurally: `mask → filter → score`. Never permute these steps.

### Critical Invariants

**PPO trajectory consistency:** During training, if `sas_step()` returns `insertion_failed`, the sampled action is kept in the rollout buffer as-is with `r_exec = -0.40`. Do NOT replace it with the next feasible action. This would corrupt the `log_prob` used in the PPO ratio. The substitution fallback is inference-only (`run_greedy_episode`).

**Dry-run is non-destructive:** `dry_run_action()` deep-copies the state and works on the copy. `commit_action_index()` modifies the real state and appends to `commit_log` (enabling `rollback_last_commit()`).

**Wait semantics (Phase 2+):** SAS never owns the wait decision. When the candidate pool is empty or all-masked, `r_exec = 0.0` (not penalized). The wait cost belongs to the DDT agent (Phase 6). `RewardConfig.wait_penalty` should be `0.0` for SAS.

**Calendar intervals are sorted:** `add_calendar_interval()` uses `bisect_right` to maintain `(start, end)` sorted order. Inserting an overlapping interval raises `ValueError`.

### Lower-Layer Estimator (`lower_layer_estimator.py`)

Reports the completion time distribution `(μ_finish, σ_finish)` for a given `(lot, machine, ppid)` via:
1. Batch sizing: `compute_sub_batches(n_wafers, side_capacity)` → ⌈N/C⌉ sub-batches
2. List scheduling: assigns sub-batches to stage instances (FIFO, earliest-free)
3. Monte Carlo: samples process time noise per `(sub_batch, stage)` and runs list scheduling `n_mc` times

The key formula (Section 2.4.3 of the report): action is masked when `deadline − μ_finish < z_ε · σ_finish`, where `z_ε ≈ 2.05` for ε = 2% violation probability.

**Wafer batch scheduling in dry-run/commit (报告 §1.5)** — `_dry_run_candidate` and `_simulate_action` schedule **sub-batches**, not individual wafers: a lot's N wafers become `compute_sub_batches(N, side_capacity)` = ⌈N/side_capacity⌉ batches (via `_lot_sub_batches`; `side_capacity` unset → N, i.e. one batch — matching the estimator's default). Each sub-batch occupies **one** `(chamber, side)` interval per stage; the batch's wafers all share that `(chamber, side, start, end)` ("同进同出" batch processing). `wafer_schedule` still has N×stages rows (schema unchanged) but batched wafers have identical times, and the calendar gets one interval per (batch, stage) — so anything that **rebuilds a calendar from wafer rows must dedup** identical `(resource, start, end)` (done in both `ResourceCalendarEnv.validate_schedule` and `encoder.validate_final_schedule_completeness`). This made dry-run/commit consistent with the lower-layer estimator (previously commit serialized wafers while the estimator batched — the qtime mask saw makespans that didn't match reality) and cut the 50-lot pressure episode ~3× (132.9s→43.1s).

**Estimate result cache (报告 §1.5 开销警示)** — `estimate()` takes an optional `cache` dict. The makespan distribution depends only on `(lot, machine, ppid, n_mc)` (static encoder data — `state` is *not* read in the computation), and `start_offset` is merely added to `mu_finish` on return. So the **base** result (offset 0) is cached keyed by `(lot, machine, ppid, n_mc)`, and `start_offset` is re-applied fresh on every call via `_with_start_offset()` (never mutate the cached base). `ResourceCalendarEnv` owns `self._estimate_cache`, passes it into both `qtime_safe_mask` (n_mc=20) and `is_doomed` (n_mc=10), and clears it only on `reset()` (the base is time/state-independent, so it is valid for the whole episode — unlike `_doomed_cache`, which clears on `advance_time`). This cut the 50-lot pressure-instance candidate-pool build from ~0.49s to ~0.086s. Correctness rests on `estimate` being state-independent; if a future change makes it read `state`, the cache invalidation must move to `advance_time`/`commit` too.

## Development Plan (superpowers convention)

Implementation plans live in `FAB_RL/FABenv/docs/superpowers/plans/` as Markdown files with TDD checkbox steps:
1. Write failing test → 2. Run to confirm failure → 3. Implement → 4. Run to confirm pass

Recent plans (newest reflect current work): `2026-05-30-phase1-phase2-report-remodel.md`, `2026-05-30-phase1-review-fixes.md`, `2026-05-30-phase2-vector-reward-multihead.md` (the multi-head subsystem above), `2026-05-30-phase2-integration-noise-lookahead.md`.

> Note: the root-level `AGENT.md` is a stale session log describing an earlier NSGA-II / rolling-schedule codebase (`rolling.py`, `local_search.py`, `objectives.py`) that does not exist in `FAB_RL/FABenv/`. Treat it as historical notes, not a description of the current code.

## Baseline Evaluation (报告 §7.4)

`evaluate_baselines.py` is the multi-seed comparison harness for paper-style evaluation. Dispatching-rule baselines live in `Phase2EpisodeDriver.run_rule_episode(strategy=...)` (`RULE_STRATEGIES = first_valid | FIFO | SPT | EDD | CR | ATC`); `_rule_action_index()` ranks the **same** qtime-safe candidate pool the RL sees (so constraint handling is identical — only the in-pool choice differs), using the cached lower-layer estimate for processing time. Each "seed" is one processing-noise realization (`process_noise_enabled` + `noise_seed=seed`, per §2.4.6). Metrics come from `encoder.evaluate_objectives()` → `schedule_metrics()` (Q-time/tardiness violations, utilization, priority violation). `evaluate(strategies, seeds, encoder_factory, policies=...)` aggregates mean/std; pass `policies={"SAS-PPO": policy}` (or `--checkpoint`) to fold in an RL greedy run.

```powershell
python evaluate_baselines.py --instance small --seeds 8
python evaluate_baselines.py --instance pressure --seeds 5 --checkpoint model.pt
```

Known finding: the 4-lot `small` instance does **not** discriminate strategies (all rules hit 0 violations / identical utilization) — meaningful comparison needs the `pressure` (50-lot) instance, which is currently slow per-step in `commit`/wafer simulation.

## Reference Code

`MAMHSA_for_fjsp-master/` contains a reference multi-agent attention PPO implementation for generic FJSP (flexible job-shop scheduling). It uses a disjunctive graph + heterogeneous GNN state representation. Consult it for attention architecture patterns; do not import from it in the FABenv package.
