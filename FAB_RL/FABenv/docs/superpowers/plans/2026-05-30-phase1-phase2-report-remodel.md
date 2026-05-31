# Phase 1 & 2 Remodel per 项目报告.md (完善版)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor Phase 1 (environment + lower-layer estimator) and Phase 2 (rule-triggered SAS-PPO) to match the revised 项目报告.md. Key changes per the report:

1. **Lower-layer heuristic estimator** (Section 1.5): batching + list scheduling + Monte Carlo → (μ_finish, σ_finish)
2. **Stochastic processing times** (Section 2.4): each stage has (μ, σ); Q-time becomes an opportunity constraint
3. **Q-time opportunity constraint mask** (Section 3.2): `deadline − μ_finish < z_ε·σ_finish` → mask out
4. **Priority filter** (Section 3.4): soft mode — keep actions, reorder in CandidateScore
5. **Dictionary-order constraint structure** (Section 1.4): mask → filter → score
6. **Phase 2 reward semantics** (Section 4.5): SAS wait = 0.0 (not −0.02); SAS does not own wait

**Architecture invariant:** Lower-layer estimator is a pure heuristic (no RL). It outputs (μ_finish, σ_finish) via Monte Carlo over list-scheduling. Upper-layer RL (SAS-PPO) uses these estimates for candidate scoring and qtime mask.

---

### Task 1: Create lower_layer_estimator.py

**Files:**
- Create: `FAB_RL/FABenv/lower_layer_estimator.py`

- [ ] **Step 1: Write the failing test**

Create `FAB_RL/FABenv/tests/test_lower_layer_estimator.py` with tests that:
- `estimate()` returns dict with keys `mu_finish`, `sigma_finish`, `bottleneck_stage`
- `mu_finish > 0` and `sigma_finish >= 0`
- With σ=0 noise, `sigma_finish ≈ 0` and `mu_finish` matches deterministic makespan
- `compute_sub_batches(n_wafers, side_capacity)` returns ⌈n/c⌉ sub-batches

- [ ] **Step 2: Run test to verify it fails**

`python -m pytest FAB_RL/FABenv/tests/test_lower_layer_estimator.py -v`

- [ ] **Step 3: Implement lower_layer_estimator.py**

Module with:
- `compute_sub_batches(n_wafers, side_capacity)` → list of sub-batch sizes
- `list_schedule_makespan(sub_batches, stages_mu, stages_sigma, n_samples)` → (μ, σ) via Monte Carlo
- `estimate(lot, machine, ppid, encoder, state, n_mc=50)` → dict with mu_finish, sigma_finish, bottleneck_stage, per_instance_occupancy

- [ ] **Step 4: Run test to verify it passes**

`python -m pytest FAB_RL/FABenv/tests/test_lower_layer_estimator.py -v`

---

### Task 2: Update problem.py — stochastic process times + qtime_deadline

**Files:**
- Modify: `FAB_RL/FABenv/problem.py`

- [ ] **Step 1: Write the failing test**

Add to a new test file `FAB_RL/FABenv/tests/test_problem_v2.py`:
- `ProblemDefinitionMixin` accepts `process_time_sigma` dict: `{(lot,machine,ppid): [σ per stage]}`
- `ProblemDefinitionMixin` accepts `qtime_deadline` dict: `{lot: float}`
- `get_process_time_sigma(lot, machine, ppid)` returns σ values for each stage

- [ ] **Step 2: Run test to verify it fails**

`python -m pytest FAB_RL/FABenv/tests/test_problem_v2.py -v`

- [ ] **Step 3: Implement changes in problem.py**

Add to `ProblemDefinitionMixin.__init__`:
```python
self.process_time_sigma = process_time_sigma or {}   # {(lot,machine,ppid): [σ per stage, ...]]}
self.qtime_deadline = qtime_deadline or {}           # {lot: float} absolute deadline
```
Add `get_process_time_sigma(lot, machine, ppid)` method.

- [ ] **Step 4: Run test**

`python -m pytest FAB_RL/FABenv/tests/test_problem_v2.py -v`

---

### Task 3: Update problem_instances.py — generate σ and qtime_deadline

**Files:**
- Modify: `FAB_RL/FABenv/problem_instances.py`

- [ ] **Step 1: Write the failing test**

Test that `build_small_encoder()` and `build_pressure_test_encoder()` both produce instances with:
- `encoder.process_time_sigma` is populated
- `encoder.qtime_deadline` is populated for all lots
- `encoder.z_eps` exists (float, default 2.05)

- [ ] **Step 2: Run test to verify it fails**

`python -m pytest FAB_RL/FABenv/tests/test_problem_instances_v2.py -v`

- [ ] **Step 3: Implement changes**

Add `process_time_sigma` generation (σ = 0.05×μ as default).
Add `qtime_deadline` (arrival + qtime_window, default qtime_window = 2×sum_of_process_times).
Add `z_eps=2.05` as attribute.

- [ ] **Step 4: Run test**

`python -m pytest FAB_RL/FABenv/tests/test_problem_instances_v2.py -v`

---

### Task 4: Update rl_environment.py — qtime mask + priority filter + updated score/features

**Files:**
- Modify: `FAB_RL/FABenv/rl_environment.py`

- [ ] **Step 1: Write the failing test**

Test that:
- `env.qtime_safe_mask(machine, candidate_actions)` returns bool array
- Actions that would cause `deadline - mu_finish < z_eps * sigma_finish` are masked out
- `env.priority_filter(safe_actions, mode="soft")` returns same list (no removal)
- `env.priority_filter(safe_actions, mode="strict")` removes lower-priority lots
- CandidateScore does NOT include priority term (verify score formula)
- New features include: `remaining_qtime`, `mu_finish`, `sigma_finish`, `qtime_violation_prob`

- [ ] **Step 2: Run test to verify it fails**

`python -m pytest FAB_RL/FABenv/tests/test_phase1_v2_env.py -v`

- [ ] **Step 3: Implement changes in rl_environment.py**

Changes:
1. Add `qtime_safe_mask(machine, candidate_actions, z_eps)` method using lower_layer_estimator
2. Add `priority_filter(safe_actions, mode, priority_min_gap)` method
3. Update `_candidate_features()` to 22 features (add remaining_qtime, mu_finish, sigma_finish, qtime_violation_prob)
4. Update `CandidateScore`: remove priority, use `due_date_urgency + qtime_slack + waiting_time - proc_time_mean - resource_conflict_risk`
5. Update `build_candidate_pool()` to apply mask → filter → TopK → pad pipeline

- [ ] **Step 4: Run test**

`python -m pytest FAB_RL/FABenv/tests/test_phase1_v2_env.py -v`

---

### Task 5: Update Phase 2 reward semantics — SAS does not own wait

**Files:**
- Modify: `FAB_RL/FABenv/rl_environment.py`

- [ ] **Step 1: Write the failing test**

Test that when `wait_or_noop=True`, `compute_sas_reward` returns 0.0 (not -0.02).
Test that `RewardConfig.wait_penalty` default is 0.0.

- [ ] **Step 2: Run test to verify it fails**

`python -m pytest FAB_RL/FABenv/tests/test_phase2_reward_v2.py -v`

- [ ] **Step 3: Implement changes**

Change `RewardConfig.wait_penalty` default from `-0.02` to `0.0`.
Update docstring to explain: SAS does not own wait; wait cost belongs to DDT.

- [ ] **Step 4: Run test**

`python -m pytest FAB_RL/FABenv/tests/test_phase2_reward_v2.py -v`

---

### Task 6: Update phase2_sas_observation.py — new features

**Files:**
- Modify: `FAB_RL/FABenv/phase2_sas_observation.py`

- [ ] **Step 1: Write the failing test**

Test that `Phase2ObservationEncoder.encode()` produces observations with:
- `global_features` has dimension 9 (unchanged)
- `candidate_features` has dimension 22 (old 18 + remaining_qtime + mu_finish + sigma_finish + qtime_violation_prob)

- [ ] **Step 2: Run test to verify it fails**

`python -m pytest FAB_RL/FABenv/tests/test_phase2_obs_v2.py -v`

- [ ] **Step 3: Implement changes**

Update `CANDIDATE_DIM = 22` and update `encode()` to pass new features through.

- [ ] **Step 4: Run test**

`python -m pytest FAB_RL/FABenv/tests/test_phase2_obs_v2.py -v`

---

### Task 7: Run full test suite to verify no regressions

- [ ] **Step 1: Run all existing tests**

`python -m pytest FAB_RL/FABenv/tests/ -v`

- [ ] **Step 2: Fix any regressions**

Address failures while keeping new tests green.
