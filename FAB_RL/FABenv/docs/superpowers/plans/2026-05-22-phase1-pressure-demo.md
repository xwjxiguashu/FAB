# Phase 1 Pressure Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the Phase 1 environment demo into a pressure test with 50 lots, 10 wafers per lot, 10 machines, and 5 PPIDs per lot-machine pair.

**Architecture:** Add a deterministic pressure-test problem factory beside the existing small factory, then point the runnable demo at it. Keep the small factory intact so existing examples remain available.

**Tech Stack:** Python, NumPy, pytest, existing `Phase1CalendarProblem` and `ResourceCalendarEnv`.

---

### Task 1: Pressure Instance Factory

**Files:**
- Create: `FAB_RL/FABenv/tests/test_phase1_pressure_demo.py`
- Modify: `FAB_RL/FABenv/problem_instances.py`
- Modify: `FAB_RL/FABenv/__init__.py`

- [ ] **Step 1: Write the failing test**

Create `FAB_RL/FABenv/tests/test_phase1_pressure_demo.py` with a test that imports `build_pressure_test_encoder`, checks `num_lots == 50`, every lot has 10 wafers, all 10 machines are feasible, and every `(lot, machine)` has 5 PPIDs with process steps.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest FAB_RL/FABenv/tests/test_phase1_pressure_demo.py -v`
Expected: FAIL because `build_pressure_test_encoder` does not exist.

- [ ] **Step 3: Write minimal implementation**

Add `build_pressure_test_encoder(seed=2026)` to `problem_instances.py`. Use deterministic NumPy generation, 3 stages per PPID, 5 chambers, 2 sides, and all machines feasible for all lots.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest FAB_RL/FABenv/tests/test_phase1_pressure_demo.py -v`
Expected: PASS.

### Task 2: Demo Uses Pressure Instance

**Files:**
- Modify: `FAB_RL/FABenv/run_phase1_environment_demo.py`

- [ ] **Step 1: Write the failing test**

Extend `FAB_RL/FABenv/tests/test_phase1_pressure_demo.py` to assert that `run_phase1_environment_demo.build_demo_encoder()` returns the pressure-test encoder.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest FAB_RL/FABenv/tests/test_phase1_pressure_demo.py -v`
Expected: FAIL because `build_demo_encoder` does not exist.

- [ ] **Step 3: Write minimal implementation**

Add `build_demo_encoder()` to the demo and call it from `main()`. It should return `build_pressure_test_encoder()`. Increase `top_k` to at least 16 so each machine candidate pool can expose several real PPID options.

- [ ] **Step 4: Run tests and pressure demo**

Run: `python -m pytest FAB_RL/FABenv/tests/test_phase1_pressure_demo.py -v`
Expected: PASS.

Run: `python FAB_RL/FABenv/run_phase1_environment_demo.py`
Expected: completes without exceptions, schedules 50 lot rows, and validates all wafer rows.
