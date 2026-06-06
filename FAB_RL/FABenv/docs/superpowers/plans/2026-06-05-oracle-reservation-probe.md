# Oracle Reservation Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Scheme C go/no-go layer for reservation planning: ROP detection, reservation ledger, rollout simulation, and an oracle probe comparing SAS/rule-only dispatch against selective reservation.

**Architecture:** Keep SAS dispatch unchanged and add a narrow reservation layer around the existing `ResourceCalendarEnv`/`Phase2EpisodeDriver`. The reservation layer can freeze a machine for a visible future lot, then the normal candidate-pool and calendar commit machinery schedules the rest.

**Tech Stack:** Python, pytest, existing `FAB_RL/FABenv` modules with bare imports.

---

### Task 1: Reservation Ledger

**Files:**
- Create: `FAB_RL/FABenv/reservation_ledger.py`
- Test: `FAB_RL/FABenv/tests/test_reservation_ledger.py`

- [x] **Step 1: Write failing tests**

Create tests for adding a reservation, checking machine availability, expiring a stale reservation, and consuming a reservation when the target lot arrives.

- [x] **Step 2: Run failing tests**

Run: `python -m pytest tests/test_reservation_ledger.py -q`
Expected: import failure because `reservation_ledger` does not exist.

- [x] **Step 3: Implement the ledger**

Add `ReservationRecord` and `ReservationLedger` with `reserve`, `reserved_machines`, `get`, `release_expired`, and `consume_for_lot`.

- [x] **Step 4: Run tests**

Run: `python -m pytest tests/test_reservation_ledger.py -q`
Expected: all tests pass.

### Task 2: ROP Detection

**Files:**
- Create: `FAB_RL/FABenv/reservation_rop.py`
- Test: `FAB_RL/FABenv/tests/test_reservation_rop.py`

- [x] **Step 1: Write failing tests**

Create tests that verify ROP finds compatible future high-priority lots, ranks candidates by score, respects `top_b`, and ignores already reserved machines.

- [x] **Step 2: Run failing tests**

Run: `python -m pytest tests/test_reservation_rop.py -q`
Expected: import failure because `reservation_rop` does not exist.

- [x] **Step 3: Implement ROP detection**

Add `ReservationOpportunity`, `detect_reservation_opportunities`, and compatibility helpers that use `env.upcoming_lots()`, `env.build_candidate_pool(machine)`, and `encoder.get_process_steps`.

- [x] **Step 4: Run tests**

Run: `python -m pytest tests/test_reservation_rop.py -q`
Expected: all tests pass.

### Task 3: Reservation Simulator and Oracle Probe

**Files:**
- Create: `FAB_RL/FABenv/reservation_simulator.py`
- Create: `FAB_RL/FABenv/oracle_reservation_probe.py`
- Test: `FAB_RL/FABenv/tests/test_reservation_simulator.py`

- [x] **Step 1: Write failing tests**

Create tests that run a small episode with no reservation, a forced reservation that eventually dispatches its target lot, and an oracle selector that returns `reserve` only when the reserve rollout objective is better.

- [x] **Step 2: Run failing tests**

Run: `python -m pytest tests/test_reservation_simulator.py -q`
Expected: import failure because `reservation_simulator` does not exist.

- [x] **Step 3: Implement simulator**

Add episode runners that wrap the existing rule-based dispatch loop, skip reserved machines until their target arrives, force the target lot when feasible, and report the same metrics as `evaluate_baselines.schedule_metrics`.

- [x] **Step 4: Implement probe CLI**

Add `oracle_reservation_probe.py` with arguments for instance, seeds, strategy, window, top-b, and output path. It prints/optionally writes JSONL rows for `baseline`, `forced_oracle`, and `delta`.

- [x] **Step 5: Run tests**

Run: `python -m pytest tests/test_reservation_simulator.py -q`
Expected: all tests pass.

### Task 4: Regression

**Files:**
- Existing tests under `FAB_RL/FABenv/tests/`

- [x] **Step 1: Run full suite**

Run from `FAB_RL/FABenv`: `python -m pytest tests/ -q`
Expected: all tests pass.

- [x] **Step 2: Smoke the probe**

Run from `FAB_RL/FABenv`: `python oracle_reservation_probe.py --instance small --seeds 1 --max-steps 200`
Expected: prints JSON summary and exits successfully.

---

### Task 5: late_hi go/no-go gate (报告4 §6.2.3 阶段 0)

The Task 1–3 machinery only *builds* the probe. The report's actual gate
(§5.11 / §6.2.3 阶段 0 / §9.8) requires running it on a **discriminating
instance** where high-priority lots arrive late (corr≈0.97) and an
**information-complete** oracle (sees all future arrivals, not a fixed window).

**Files:**
- Edit: `problem_instances.py` (add `priority_mode="late_hi"` + `build_late_hi_encoder`)
- Edit: `oracle_reservation_probe.py` (info-complete oracle via `_full_horizon_lookahead`; `late_hi` factory)
- Edit: `evaluate_baselines.py` (`late_hi` in `ENCODER_FACTORIES` + `--instance`)
- Test: `tests/test_late_hi_instance.py`

- [x] **Step 1: late_hi instance generator (corr≈0.97), TDD** — `build_late_hi_encoder`, measured corr=0.979.
- [x] **Step 2: wire late_hi into probe + evaluate_baselines factories**, TDD.
- [x] **Step 3: make the oracle information-complete** (`_full_horizon_lookahead`; default on, `--oracle-window` to restrict), TDD.
- [ ] **Step 4: run the gate and record the verdict.**
  Run: `python oracle_reservation_probe.py --instance late_hi --seeds 2 --strategy FIFO --top-b 3 --max-steps 600 --out results/oracle_reservation_late_hi_fifo.jsonl`
  Verdict (报告 §7.5 floor was soft-FIFO O2≈862 on *pressure*; on late_hi compare oracle O2 vs the no-reservation rule floor):
  - 🟢 oracle O2 ≪ rule floor → proceed to 阶段 7 VC-MCTS (also then train SAS to check the harder `oracle < SAS-only` bound).
  - 🔴 oracle ≈ rule floor → fix the instance generator (§9.8), do NOT add search depth.
- [ ] **Step 5 (deferred): SAS-only baseline** — needs a trained `model.pt`; fold into the gate once a checkpoint exists (`--checkpoint`).
