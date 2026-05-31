# Phase 2 SAS-PPO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the Phase 2 rule-triggered SAS-PPO prototype described in `项目方案.md` and make all Phase 2 tests pass.

**Architecture:** Preserve the existing Phase 1 resource-calendar environment and fill the missing Phase 2 interfaces around it. The driver owns episode control and deterministic machine selection, the observation encoder owns fixed tensor inputs, the actor-critic owns masked action/value evaluation, the buffer owns rollout storage/GAE, and the trainer owns PPO updates.

**Tech Stack:** Python, NumPy, PyTorch, pytest, existing `FAB_RL/FABenv` modules.

---

## File Structure

- Modify `FAB_RL/FABenv/rl_environment.py`: verify and complete reset, candidate machine discovery, next-event time, SAS observation, rank features, reward decomposition.
- Modify `FAB_RL/FABenv/phase2_sas_driver.py`: add full episode loop, next-event advancement, terminal reasons, rollout recording, greedy inference fallback.
- Modify `FAB_RL/FABenv/phase2_sas_observation.py`: ensure stable observation dataclass, global features, batch conversion.
- Modify `FAB_RL/FABenv/phase2_sas_policy.py`: harden masked categorical behavior, sampling, greedy, evaluation.
- Modify `FAB_RL/FABenv/phase2_ppo_buffer.py`: add `get_training_batches`, robust GAE, StepInfo conversion.
- Modify `FAB_RL/FABenv/phase2_ppo_trainer.py`: add collect/train methods, loss helpers, minibatch updates.
- Modify `FAB_RL/FABenv/train_phase2_sas_ppo.py`: build components, run multiple episodes, save optional checkpoint.
- Modify `FAB_RL/FABenv/run_phase2_sas_inference_demo.py`: run greedy/fallback inference and validate schedule.
- Modify/add tests in `FAB_RL/FABenv/tests/test_phase2_*.py`: assert all acceptance points from `项目方案.md`.

---

### Task 1: Baseline and Existing Test Inventory

**Files:**
- Read/run: `FAB_RL/FABenv/tests/test_phase2_*.py`

- [ ] **Step 1: Run all existing Phase 2 tests to establish RED baseline**

Run:
```bash
python -m pytest FAB_RL/FABenv/tests/test_phase2_*.py -q
```
Expected: Some failures showing the missing Phase 2 behavior.

- [ ] **Step 2: Inspect failed tests and map each failure to files**

Run:
```bash
python -m pytest FAB_RL/FABenv/tests/test_phase2_*.py -q -vv
```
Expected: Failure names identify missing driver/trainer/buffer/inference behavior.

---

### Task 2: Environment Interface Acceptance Tests

**Files:**
- Test: `FAB_RL/FABenv/tests/test_phase2_environment_interfaces.py`
- Modify: `FAB_RL/FABenv/rl_environment.py`

- [ ] **Step 1: Ensure tests cover reset, candidate machines, next event, and SAS observation**

Use or add tests equivalent to:
```python
def test_environment_exposes_phase2_training_interfaces():
    encoder = build_small_encoder()
    env = ResourceCalendarEnv(encoder)

    summary = env.reset()

    assert summary["current_time"] == 0.0
    assert summary["remaining_lots"] == encoder.num_lots
    machines = env.get_candidate_machines()
    assert machines
    observation = env.build_sas_observation(machines[0])
    assert observation.machine == machines[0]
    assert observation.pool is not None
    assert observation.mask is not None
```

- [ ] **Step 2: Run interface test and verify failure if behavior is missing**

Run:
```bash
python -m pytest FAB_RL/FABenv/tests/test_phase2_environment_interfaces.py -q
```
Expected: FAIL only for unimplemented or incorrect environment helper behavior.

- [ ] **Step 3: Implement minimal environment interface fixes**

In `rl_environment.py`, ensure these methods exist and preserve Phase 1 behavior:
```python
def reset(self, current_time=0.0, initial_state=None, completed_lots=None):
    # restore initial schedules, remaining lots, calendars, current time
    # return dict with current_time, completed_lots, remaining_lots


def get_candidate_machines(self):
    machines = []
    for machine in self.encoder.MACHINE_LIST:
        pool = self.build_candidate_pool(machine)
        if any(mask and not action.is_padding for action, mask in zip(pool.actions, pool.action_mask)):
            machines.append(int(machine))
    return machines


def next_event_time(self):
    # return min future arrival or resource-release time > current_time, else None


def build_sas_observation(self, machine):
    pool = self.build_candidate_pool(machine)
    mask = self.build_action_mask(machine, pool.actions)
    return SASObservation(machine=int(machine), current_time=float(self.current_time), pool=pool, mask=mask)
```

- [ ] **Step 4: Run interface test and verify GREEN**

Run:
```bash
python -m pytest FAB_RL/FABenv/tests/test_phase2_environment_interfaces.py -q
```
Expected: PASS.

---

### Task 3: Driver Rule-Triggered Episode Behavior

**Files:**
- Test: `FAB_RL/FABenv/tests/test_phase2_sas_driver.py`
- Modify: `FAB_RL/FABenv/phase2_sas_driver.py`

- [ ] **Step 1: Ensure tests cover deterministic machine selection**

Use or add tests equivalent to:
```python
def test_select_next_machine_uses_availability_candidate_count_and_id():
    encoder = build_small_encoder()
    env = ResourceCalendarEnv(encoder)
    observation_encoder = Phase2ObservationEncoder()
    driver = Phase2EpisodeDriver(env, observation_encoder, RewardConfig())

    machines = env.get_candidate_machines()
    selected = driver.select_next_machine(machines)

    expected = min(
        machines,
        key=lambda m: (
            env.state.machine_available_time.get(m, env.current_time),
            sum(
                bool(valid) and not action.is_wait and not action.is_padding
                for action, valid in zip(env.build_candidate_pool(m).actions, env.build_candidate_pool(m).action_mask)
            ),
            int(m),
        ),
    )
    assert selected == expected
```

- [ ] **Step 2: Ensure tests cover terminal summaries and no-dead-loop rule episode**

Use or add tests equivalent to:
```python
def test_rule_episode_returns_required_summary_fields():
    env = ResourceCalendarEnv(build_small_encoder())
    driver = Phase2EpisodeDriver(env, Phase2ObservationEncoder(), RewardConfig(), max_steps=200)

    summary = driver.run_rule_episode()

    assert {"steps", "episode_reward", "completed_lots", "wait_steps", "failed_actions", "termination_reason"} <= set(summary)
    assert summary["steps"] <= 200
    assert summary["termination_reason"]
```

- [ ] **Step 3: Run driver tests to verify RED**

Run:
```bash
python -m pytest FAB_RL/FABenv/tests/test_phase2_sas_driver.py -q
```
Expected: FAIL for any missing driver methods such as `advance_to_next_event`, `run_policy_episode`, or `run_greedy_episode`.

- [ ] **Step 4: Implement minimal driver fixes**

In `phase2_sas_driver.py`, implement:
```python
def advance_to_next_event(self):
    next_time = self.env.next_event_time()
    if next_time is None:
        return None
    if next_time <= self.env.current_time:
        self.unrecoverable_error = True
        self.termination_reason = "unrecoverable_error"
        return None
    self.env.advance_time(next_time)
    self.total_wait_steps_per_episode += 1
    return float(next_time)
```

Also implement `run_policy_episode` to repeatedly build decisions, call `policy.sample_action` or `policy.greedy_action`, execute `env.sas_step`, compute next observation, add `RolloutStep` when a buffer is provided, and return the required summary.

Implement `run_greedy_episode` with inference fallback by sorting policy probabilities descending and trying candidate indices until one commits.

- [ ] **Step 5: Run driver tests and verify GREEN**

Run:
```bash
python -m pytest FAB_RL/FABenv/tests/test_phase2_sas_driver.py -q
```
Expected: PASS.

---

### Task 4: Observation Encoder Behavior

**Files:**
- Test: `FAB_RL/FABenv/tests/test_phase2_sas_observation.py`
- Modify: `FAB_RL/FABenv/phase2_sas_observation.py`

- [ ] **Step 1: Ensure tests cover fixed shapes and global features**

Use or add tests equivalent to:
```python
def test_observation_encoder_outputs_fixed_phase2_fields():
    env = ResourceCalendarEnv(build_small_encoder())
    pool = env.build_candidate_pool(env.get_candidate_machines()[0])
    obs = Phase2ObservationEncoder().encode(env.get_candidate_machines()[0], pool, env)

    assert obs.candidate_features.shape[0] == env.top_k
    assert obs.candidate_mask.shape == (env.top_k,)
    assert obs.global_features.shape == (9,)
    assert obs.action_indices.tolist() == list(range(env.top_k))
    assert obs.valid_action_count == int(obs.candidate_mask.sum())
```

- [ ] **Step 2: Run observation tests to verify RED/GREEN**

Run:
```bash
python -m pytest FAB_RL/FABenv/tests/test_phase2_sas_observation.py -q
```
Expected: FAIL if shapes or fields are wrong; PASS after minimal fixes.

- [ ] **Step 3: Implement minimal encoder fixes**

Ensure `build_global_features` returns exactly:
```python
[
    current_time_norm,
    completed_ratio,
    remaining_ratio,
    machine_id_norm,
    machine_busy_time,
    valid_action_count_norm,
    candidate_score_mean,
    candidate_waiting_time_max,
    candidate_due_slack_min,
]
```
If normalization is enabled, scale current time and machine busy time by a stable positive horizon/due-date denominator.

- [ ] **Step 4: Run observation tests and verify GREEN**

Run:
```bash
python -m pytest FAB_RL/FABenv/tests/test_phase2_sas_observation.py -q
```
Expected: PASS.

---

### Task 5: Masked Policy Behavior

**Files:**
- Test: `FAB_RL/FABenv/tests/test_phase2_sas_policy.py`
- Modify: `FAB_RL/FABenv/phase2_sas_policy.py`

- [ ] **Step 1: Ensure tests cover sample/greedy never selecting masked actions**

Use or add tests equivalent to:
```python
def test_masked_policy_never_selects_invalid_greedy_action():
    policy = Phase2SASActorCritic(candidate_dim=18, global_dim=9, hidden_dim=16)
    candidate_features = torch.randn(1, 4, 18)
    candidate_mask = torch.tensor([[False, True, False, True]])
    global_features = torch.randn(1, 9)

    result = policy.greedy_action(candidate_features, candidate_mask, global_features)

    assert int(result["action"].item()) in {1, 3}
    assert torch.all(result["probs"][~candidate_mask] == 0)
```

- [ ] **Step 2: Run policy tests to verify RED/GREEN**

Run:
```bash
python -m pytest FAB_RL/FABenv/tests/test_phase2_sas_policy.py -q
```
Expected: FAIL if masks are mishandled; PASS after minimal fixes.

- [ ] **Step 3: Implement minimal policy hardening**

In `MaskedCategoricalPolicy.forward`, reject all-false masks with a clear `ValueError` and otherwise use finite masked logits:
```python
if not torch.any(mask.bool(), dim=-1).all():
    raise ValueError("masked categorical requires at least one valid action per row")
masked_logits = logits.masked_fill(~mask.bool(), torch.finfo(logits.dtype).min)
return torch.distributions.Categorical(logits=masked_logits)
```

- [ ] **Step 4: Run policy tests and verify GREEN**

Run:
```bash
python -m pytest FAB_RL/FABenv/tests/test_phase2_sas_policy.py -q
```
Expected: PASS.

---

### Task 6: Rollout Buffer and GAE

**Files:**
- Test: `FAB_RL/FABenv/tests/test_phase2_ppo_buffer.py`
- Modify: `FAB_RL/FABenv/phase2_ppo_buffer.py`

- [ ] **Step 1: Ensure tests cover RolloutStep required fields and GAE**

Use or add tests equivalent to:
```python
def test_rollout_buffer_records_phase2_transition_and_batches():
    buffer = Phase2RolloutBuffer(gamma=1.0, gae_lambda=1.0)
    step = RolloutStep(
        machine_id=1,
        current_time=0.0,
        candidate_features=np.zeros((4, 18), dtype=np.float32),
        candidate_mask=np.array([True, False, False, False]),
        global_features=np.zeros(9, dtype=np.float32),
        action_indices=np.arange(4),
        valid_action_count=1,
        action=0,
        log_prob=-0.1,
        value=0.5,
        reward=1.0,
        done=True,
        next_observation=None,
        info=StepInfo(selected_lot=1, reward_total=1.0),
    )
    buffer.add(step)
    buffer.finish_episode()

    assert buffer.returns == [1.0]
    assert buffer.advantages == [0.5]
    batch = next(buffer.get_training_batches(batch_size=1))
    assert batch["machine_id"].tolist() == [1]
```

- [ ] **Step 2: Run buffer tests to verify RED**

Run:
```bash
python -m pytest FAB_RL/FABenv/tests/test_phase2_ppo_buffer.py -q
```
Expected: FAIL if `get_training_batches` is missing.

- [ ] **Step 3: Implement minimal batch collation**

Add `get_training_batches` yielding dictionaries with NumPy arrays:
```python
def get_training_batches(self, batch_size):
    indices = list(range(len(self.steps)))
    for start in range(0, len(indices), int(batch_size)):
        selected = indices[start:start + int(batch_size)]
        yield {
            "machine_id": np.asarray([self.steps[i].machine_id for i in selected], dtype=np.int64),
            "candidate_features": np.stack([self.steps[i].candidate_features for i in selected]),
            "candidate_mask": np.stack([self.steps[i].candidate_mask for i in selected]),
            "global_features": np.stack([self.steps[i].global_features for i in selected]),
            "actions": np.asarray([self.steps[i].action for i in selected], dtype=np.int64),
            "old_log_probs": np.asarray([self.steps[i].log_prob for i in selected], dtype=np.float32),
            "returns": np.asarray([self.returns[i] for i in selected], dtype=np.float32),
            "advantages": np.asarray([self.advantages[i] for i in selected], dtype=np.float32),
        }
```

- [ ] **Step 4: Run buffer tests and verify GREEN**

Run:
```bash
python -m pytest FAB_RL/FABenv/tests/test_phase2_ppo_buffer.py -q
```
Expected: PASS.

---

### Task 7: PPO Trainer Smoke Behavior

**Files:**
- Test: `FAB_RL/FABenv/tests/test_phase2_ppo_smoke.py`
- Modify: `FAB_RL/FABenv/phase2_ppo_trainer.py`
- Modify: `FAB_RL/FABenv/phase2_sas_driver.py`

- [ ] **Step 1: Ensure tests cover collect episode and one PPO update**

Use or add tests equivalent to:
```python
def test_ppo_trainer_collects_and_updates_one_episode():
    components = build_training_components()
    buffer = Phase2RolloutBuffer()

    summary = components["trainer"].collect_episode(components["driver"], buffer, stochastic=True)
    buffer.finish_episode()
    stats = components["trainer"].update_policy(buffer)

    assert summary["steps"] > 0
    assert buffer.steps
    assert {"policy_loss", "value_loss", "entropy"} <= set(stats)
```

- [ ] **Step 2: Run PPO smoke test to verify RED**

Run:
```bash
python -m pytest FAB_RL/FABenv/tests/test_phase2_ppo_smoke.py -q
```
Expected: FAIL if collect/train helpers are missing.

- [ ] **Step 3: Implement minimal trainer helpers**

Add:
```python
def collect_episode(self, driver, buffer, stochastic=True):
    return driver.run_policy_episode(self.policy, buffer=buffer, stochastic=stochastic)


def train(self, driver, num_episodes):
    history = []
    for _ in range(int(num_episodes)):
        buffer = Phase2RolloutBuffer(self.config.gamma, self.config.gae_lambda)
        summary = self.collect_episode(driver, buffer, stochastic=True)
        buffer.finish_episode(last_value=0.0)
        stats = self.update_policy(buffer) if buffer.steps else {}
        history.append({**summary, **stats})
        driver.reset_episode()
    return history
```

Also expose `compute_policy_loss`, `compute_value_loss`, and `compute_entropy_bonus` as wrappers around the existing PPO update math if tests require those names.

- [ ] **Step 4: Run PPO smoke test and verify GREEN**

Run:
```bash
python -m pytest FAB_RL/FABenv/tests/test_phase2_ppo_smoke.py -q
```
Expected: PASS.

---

### Task 8: Training and Inference Entry Points

**Files:**
- Test: `FAB_RL/FABenv/tests/test_phase2_inference_demo.py`
- Modify: `FAB_RL/FABenv/train_phase2_sas_ppo.py`
- Modify: `FAB_RL/FABenv/run_phase2_sas_inference_demo.py`

- [ ] **Step 1: Ensure tests cover inference summary and validation**

Use or add tests equivalent to:
```python
def test_phase2_inference_demo_returns_validation_summary():
    summary = run_demo_episode(max_steps=200)

    assert "validation_passed" in summary
    assert "machine_conflicts" in summary
    assert "chamber_conflicts" in summary
    assert summary["steps"] <= 200
    assert summary["termination_reason"]
```

- [ ] **Step 2: Run inference tests to verify RED/GREEN**

Run:
```bash
python -m pytest FAB_RL/FABenv/tests/test_phase2_inference_demo.py -q
```
Expected: FAIL if demo does not use Phase 2 greedy/fallback logic; PASS after fixes.

- [ ] **Step 3: Implement minimal training entry point**

Make `build_training_components()` reset the environment before sampling shapes, build `Phase2RolloutBuffer`, and use `trainer.train(driver, num_episodes)` in `main()`.

- [ ] **Step 4: Implement minimal inference entry point**

Make `run_demo_episode()` call `driver.run_greedy_episode(policy)`, then `env.validate_schedule(partial=True)`, and return validation fields plus reward/step fields.

- [ ] **Step 5: Run inference tests and verify GREEN**

Run:
```bash
python -m pytest FAB_RL/FABenv/tests/test_phase2_inference_demo.py -q
```
Expected: PASS.

---

### Task 9: Full Phase 2 Regression

**Files:**
- All Phase 2 source and tests.

- [ ] **Step 1: Run all Phase 2 tests**

Run:
```bash
python -m pytest FAB_RL/FABenv/tests/test_phase2_*.py -q
```
Expected: PASS.

- [ ] **Step 2: Run Phase 1 pressure test to ensure no regression**

Run:
```bash
python -m pytest FAB_RL/FABenv/tests/test_phase1_pressure_demo.py -q
```
Expected: PASS.

- [ ] **Step 3: Run training smoke command**

Run:
```bash
python FAB_RL/FABenv/train_phase2_sas_ppo.py
```
Expected: Script exits successfully and prints episode/training summary.

- [ ] **Step 4: Run inference smoke command**

Run:
```bash
python FAB_RL/FABenv/run_phase2_sas_inference_demo.py
```
Expected: Script exits successfully and prints validation summary.

---

## Self-Review

- Spec coverage: plan covers environment interfaces, driver, observation, policy, buffer, trainer, train/inference entry points, tests, reward/mask/terminal fields.
- Placeholder scan: no `TBD`, no unbounded future work, no undefined tasks.
- Type consistency: method names match `项目方案.md` and existing Phase 2 modules.
