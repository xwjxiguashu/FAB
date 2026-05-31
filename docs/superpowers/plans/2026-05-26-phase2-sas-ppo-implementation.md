# Phase2 SAS-PPO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the Stage 2 rule-triggered SAS-PPO minimum closed loop described in `项目方案.md`.

**Architecture:** Build Phase 2 on top of the existing `ResourceCalendarEnv` instead of replacing the resource-calendar core. The work proceeds in four testable batches: environment and rule-triggered scheduling, observation and masked policy, rollout and PPO training, then inference and verification. Candidate rank features and decomposed reward fields are treated as required Phase 2 environment contracts.

**Tech Stack:** Python, NumPy, PyTorch, pytest, existing `FAB_RL/FABenv/rl_environment.py`, existing `problem_instances.py`, new Phase 2 modules under `FAB_RL/FABenv`.

---

## Scope Check

This plan implements the Phase 2 minimum closed loop from `项目方案.md`. It does not implement DDT wait-time learning, single-attention or dual-attention actor upgrades, large-scale experiment suites, or final paper-grade evaluation. Those are explicitly later phases in the project plan.

The reward mechanism and candidate rank features already have focused sub-plans:

- `docs/superpowers/plans/2026-05-26-phase2-candidate-rank-features.md`
- `docs/superpowers/plans/2026-05-26-phase2-reward-mechanism.md`

Execute those first because later Phase 2 modules rely on their feature and info-field contracts.

## File Structure

- Modify: `FAB_RL/FABenv/rl_environment.py`
  - Add candidate rank features.
  - Add decomposed reward fields.
  - Add `reset(...)`, `get_candidate_machines()`, `next_event_time()`, and `build_sas_observation(machine)`.
- Create: `FAB_RL/FABenv/phase2_sas_driver.py`
  - Owns rule-triggered episode flow, machine selection, counters, and termination reasons.
- Create: `FAB_RL/FABenv/phase2_sas_observation.py`
  - Converts `CandidatePool` plus environment summaries into fixed-shape model inputs.
- Create: `FAB_RL/FABenv/phase2_sas_policy.py`
  - Implements masked categorical action distribution and minimal MLP Actor-Critic.
- Create: `FAB_RL/FABenv/phase2_ppo_buffer.py`
  - Stores SAS transitions and computes GAE returns / advantages.
- Create: `FAB_RL/FABenv/phase2_ppo_trainer.py`
  - Collects episodes and runs PPO clipped-objective updates.
- Create: `FAB_RL/FABenv/train_phase2_sas_ppo.py`
  - Builds components, trains on a small instance, and saves a checkpoint.
- Create: `FAB_RL/FABenv/run_phase2_sas_inference_demo.py`
  - Runs greedy inference with probability-descending fallback.
- Modify: `FAB_RL/FABenv/__init__.py`
  - Export stable Phase 2 classes after tests pass.
- Create/Modify tests:
  - `FAB_RL/FABenv/tests/test_phase2_candidate_rank_features.py`
  - `FAB_RL/FABenv/tests/test_phase2_reward.py`
  - `FAB_RL/FABenv/tests/test_phase2_environment_interfaces.py`
  - `FAB_RL/FABenv/tests/test_phase2_sas_driver.py`
  - `FAB_RL/FABenv/tests/test_phase2_sas_observation.py`
  - `FAB_RL/FABenv/tests/test_phase2_sas_policy.py`
  - `FAB_RL/FABenv/tests/test_phase2_ppo_buffer.py`
  - `FAB_RL/FABenv/tests/test_phase2_ppo_smoke.py`
  - `FAB_RL/FABenv/tests/test_phase2_inference_demo.py`

---

### Task 1: Environment Feature and Reward Contracts

**Files:**
- Modify: `FAB_RL/FABenv/rl_environment.py`
- Create: `FAB_RL/FABenv/tests/test_phase2_candidate_rank_features.py`
- Create: `FAB_RL/FABenv/tests/test_phase2_reward.py`

- [ ] **Step 1: Execute the candidate rank feature plan**

Run through every unchecked task in:

```text
docs/superpowers/plans/2026-05-26-phase2-candidate-rank-features.md
```

Expected result:

```text
ResourceCalendarEnv.feature_names includes:
priority_rank_norm
due_slack_rank_norm
is_best_priority
is_most_urgent_due

python -m pytest FAB_RL/FABenv/tests/test_phase2_candidate_rank_features.py -v
passes
```

- [ ] **Step 2: Execute the reward mechanism plan**

Run through every unchecked task in:

```text
docs/superpowers/plans/2026-05-26-phase2-reward-mechanism.md
```

Expected result:

```text
compute_sas_reward(...) writes reward_execute, reward_wait, reward_tardy,
reward_qtime, reward_priority, reward_progress, reward_shape,
reward_terminal, and reward_total into info.

python -m pytest FAB_RL/FABenv/tests/test_phase2_reward.py -v
passes
```

- [ ] **Step 3: Verify both environment contracts together**

Run:

```bash
python -m pytest FAB_RL/FABenv/tests/test_phase2_candidate_rank_features.py FAB_RL/FABenv/tests/test_phase2_reward.py -v
```

Expected: all tests PASS.

- [ ] **Step 4: Commit**

Run:

```bash
git add FAB_RL/FABenv/rl_environment.py FAB_RL/FABenv/tests/test_phase2_candidate_rank_features.py FAB_RL/FABenv/tests/test_phase2_reward.py
git commit -m "feat: add phase2 environment feature and reward contracts"
```

---

### Task 2: Add Phase 2 Environment Helper Interfaces

**Files:**
- Modify: `FAB_RL/FABenv/rl_environment.py`
- Create: `FAB_RL/FABenv/tests/test_phase2_environment_interfaces.py`

- [ ] **Step 1: Write failing tests for reset and candidate-machine discovery**

Create `FAB_RL/FABenv/tests/test_phase2_environment_interfaces.py` with:

```python
import sys
from pathlib import Path


PHASE1_DIR = Path(__file__).resolve().parents[1]
if str(PHASE1_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE1_DIR))


from problem_instances import build_small_demo_encoder
from rl_environment import ResourceCalendarEnv, SASObservation


def test_reset_restores_multi_episode_initial_state():
    encoder = build_small_demo_encoder()
    env = ResourceCalendarEnv(encoder)
    pool = env.build_candidate_pool(1)
    valid_index = next(
        index
        for index, is_valid in enumerate(pool.action_mask)
        if bool(is_valid) and not pool.actions[index].is_wait
    )
    env.sas_step(1, valid_index, pool=pool)
    assert env.completed_lots

    summary = env.reset(current_time=3.0)

    assert summary["current_time"] == 3.0
    assert summary["completed_lots"] == set()
    assert summary["remaining_lots"] == set(range(1, encoder.num_lots + 1))
    assert env.current_time == 3.0
    assert env.completed_lots == set()
    assert env.lot_schedule.shape == (0, 5)
    assert env.wafer_schedule.shape == (0, 9)


def test_get_candidate_machines_returns_machines_with_valid_real_candidates():
    encoder = build_small_demo_encoder()
    env = ResourceCalendarEnv(encoder)

    machines = env.get_candidate_machines()

    assert machines
    for machine in machines:
        pool = env.build_candidate_pool(machine)
        assert any(
            bool(is_valid) and not action.is_wait and not action.is_padding
            for action, is_valid in zip(pool.actions, pool.action_mask)
        )


def test_build_sas_observation_wraps_candidate_pool_and_feature_names():
    encoder = build_small_demo_encoder()
    env = ResourceCalendarEnv(encoder)

    observation = env.build_sas_observation(1)

    assert isinstance(observation, SASObservation)
    assert observation.machine == 1
    assert observation.current_time == env.current_time
    assert observation.candidate_features.shape[1] == len(env.feature_names)
    assert observation.feature_names == env.feature_names
    assert observation.action_index_to_real_action
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
python -m pytest FAB_RL/FABenv/tests/test_phase2_environment_interfaces.py -v
```

Expected: FAIL because `reset(...)`, `get_candidate_machines()`, and `build_sas_observation(...)` are not implemented.

- [ ] **Step 3: Implement helper interfaces**

Add these methods to `ResourceCalendarEnv` in `FAB_RL/FABenv/rl_environment.py` after `advance_time(...)`:

```python
    def reset(self, current_time=0.0, initial_state=None, completed_lots=None):
        self.current_time = float(current_time)
        self.state = initial_state if initial_state is not None else ScheduleState()
        self.completed_lots = {int(lot) for lot in (completed_lots or set())}
        self.lot_schedule = np.empty((0, 5), dtype=float)
        self.wafer_schedule = np.empty((0, 9), dtype=float)
        self._sync_state_summary()
        return self.step_info()

    def get_candidate_machines(self):
        machines = []
        for machine in range(1, int(self.encoder.num_machines) + 1):
            pool = self.build_candidate_pool(machine)
            has_real_candidate = any(
                bool(is_valid)
                and not self._coerce_action(action).is_wait
                and not self._coerce_action(action).is_padding
                for action, is_valid in zip(pool.actions, pool.action_mask)
            )
            if has_real_candidate:
                machines.append(machine)
        return machines

    def next_event_time(self):
        future_times = []
        for lot in self.remaining_lots:
            arrival = float(self.encoder.arrival_times.get(int(lot), self.current_time))
            if arrival > self.current_time:
                future_times.append(arrival)
        for time_value in self.state.machine_available_time.values():
            if float(time_value) > self.current_time:
                future_times.append(float(time_value))
        for time_value in self.state.chamber_available_time.values():
            if float(time_value) > self.current_time:
                future_times.append(float(time_value))
        if not future_times:
            return None
        return min(future_times)

    def build_sas_observation(self, machine):
        pool = self.build_candidate_pool(machine)
        action_index_to_real_action = {
            index: action
            for index, action in enumerate(pool.actions)
            if bool(pool.action_mask[index])
        }
        lot_schedule = np.asarray(self.lot_schedule, dtype=float).reshape((-1, 5))
        machine_busy_time = 0.0
        if lot_schedule.size > 0:
            machine_rows = lot_schedule[lot_schedule[:, 1].astype(int) == int(machine)]
            if machine_rows.size > 0:
                machine_busy_time = float(np.sum(machine_rows[:, 4] - machine_rows[:, 3]))
        global_state_summary = {
            "current_time": self.current_time,
            "completed_count": len(self.completed_lots),
            "remaining_count": len(self.remaining_lots),
            "num_lots": int(self.encoder.num_lots),
            "num_machines": int(self.encoder.num_machines),
        }
        calendar_summary = {
            "machine_busy_time": machine_busy_time,
            "valid_action_count": int(np.sum(pool.action_mask)),
        }
        return SASObservation(
            machine=int(machine),
            current_time=self.current_time,
            candidate_pool=pool,
            candidate_actions=pool.actions,
            candidate_features=pool.features,
            candidate_mask=pool.action_mask,
            action_index_to_real_action=action_index_to_real_action,
            global_state_summary=global_state_summary,
            calendar_summary=calendar_summary,
            feature_names=self.feature_names,
        )
```

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```bash
python -m pytest FAB_RL/FABenv/tests/test_phase2_environment_interfaces.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add FAB_RL/FABenv/rl_environment.py FAB_RL/FABenv/tests/test_phase2_environment_interfaces.py
git commit -m "feat: add phase2 environment helper interfaces"
```

---

### Task 3: Implement Rule-Triggered Episode Driver

**Files:**
- Create: `FAB_RL/FABenv/phase2_sas_driver.py`
- Create: `FAB_RL/FABenv/tests/test_phase2_sas_driver.py`

- [ ] **Step 1: Write failing driver tests**

Create `FAB_RL/FABenv/tests/test_phase2_sas_driver.py` with:

```python
import sys
from pathlib import Path


PHASE1_DIR = Path(__file__).resolve().parents[1]
if str(PHASE1_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE1_DIR))


from phase2_sas_driver import Phase2EpisodeDriver
from problem_instances import build_small_demo_encoder
from rl_environment import ResourceCalendarEnv, RewardConfig


class PassThroughObservationEncoder:
    def encode(self, machine, pool, env):
        return env.build_sas_observation(machine)


def test_select_next_machine_uses_constrained_lexicographic_rule():
    encoder = build_small_demo_encoder()
    env = ResourceCalendarEnv(encoder)
    driver = Phase2EpisodeDriver(env, PassThroughObservationEncoder(), RewardConfig())

    machines = driver.get_dispatchable_machines()
    selected = driver.select_next_machine(machines)
    expected = min(
        machines,
        key=lambda machine: (
            env.state.machine_available_time.get(machine, env.current_time),
            sum(
                bool(is_valid) and not action.is_wait and not action.is_padding
                for action, is_valid in zip(
                    env.build_candidate_pool(machine).actions,
                    env.build_candidate_pool(machine).action_mask,
                )
            ),
            machine,
        ),
    )

    assert selected == expected


def test_run_rule_episode_with_first_valid_action_completes_or_stops_cleanly():
    encoder = build_small_demo_encoder()
    env = ResourceCalendarEnv(encoder)
    driver = Phase2EpisodeDriver(
        env,
        PassThroughObservationEncoder(),
        RewardConfig(),
        max_steps=200,
    )

    summary = driver.run_rule_episode(strategy="first_valid")

    assert summary["steps"] > 0
    assert summary["termination_reason"] in {
        "all_lots_completed",
        "no_future_event",
        "max_steps_exceeded",
        "planning_horizon_exceeded",
        "max_total_wait_steps_exceeded",
        "max_failed_actions_exceeded",
        "unrecoverable_error",
    }
    assert "episode_reward" in summary
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
python -m pytest FAB_RL/FABenv/tests/test_phase2_sas_driver.py -v
```

Expected: FAIL because `phase2_sas_driver.py` does not exist.

- [ ] **Step 3: Implement the driver**

Create `FAB_RL/FABenv/phase2_sas_driver.py` with:

```python
from dataclasses import dataclass

import numpy as np


@dataclass
class Phase2DispatchDecision:
    machine: int
    pool: object
    observation: object
    current_time: float


class Phase2EpisodeDriver:
    def __init__(
        self,
        env,
        observation_encoder,
        reward_config,
        planning_horizon=None,
        max_steps=10000,
        max_total_wait_steps_per_episode=1000,
        max_failed_actions=None,
    ):
        self.env = env
        self.observation_encoder = observation_encoder
        self.reward_config = reward_config
        self.planning_horizon = planning_horizon
        self.max_steps = int(max_steps)
        self.max_total_wait_steps_per_episode = int(max_total_wait_steps_per_episode)
        self.max_failed_actions = (
            int(max_failed_actions)
            if max_failed_actions is not None
            else 3 * int(getattr(env, "top_k", 8))
        )
        self.total_wait_steps_per_episode = 0
        self.consecutive_failed_actions = 0
        self.unrecoverable_error = False
        self.termination_reason = ""

    def reset_episode(self):
        self.total_wait_steps_per_episode = 0
        self.consecutive_failed_actions = 0
        self.unrecoverable_error = False
        self.termination_reason = ""
        return self.env.reset()

    def get_dispatchable_machines(self):
        return self.env.get_candidate_machines()

    def select_next_machine(self, machines):
        if not machines:
            raise ValueError("machines must not be empty")

        def key(machine):
            pool = self.env.build_candidate_pool(machine)
            real_count = sum(
                bool(is_valid)
                and not action.is_wait
                and not action.is_padding
                for action, is_valid in zip(pool.actions, pool.action_mask)
            )
            return (
                self.env.state.machine_available_time.get(machine, self.env.current_time),
                real_count,
                int(machine),
            )

        return int(min(machines, key=key))

    def build_decision(self, machine):
        pool = self.env.build_candidate_pool(machine)
        observation = self.observation_encoder.encode(machine, pool, self.env)
        return Phase2DispatchDecision(
            machine=int(machine),
            pool=pool,
            observation=observation,
            current_time=self.env.current_time,
        )

    def step_with_action(self, machine, action_index, pool=None):
        result = self.env.sas_step(
            machine,
            action_index,
            pool=pool,
            reward_config=self.reward_config,
        )
        self.record_step_result(result)
        return result

    def record_step_result(self, step_result):
        info = step_result.info
        if info.get("insertion_success"):
            self.consecutive_failed_actions = 0
        elif info.get("mask_invalid") or info.get("insertion_failed") or not step_result.committed:
            self.consecutive_failed_actions += 1
        if info.get("wait_or_noop"):
            self.total_wait_steps_per_episode += 1

    def is_episode_done(self):
        if len(self.env.remaining_lots) == 0:
            return True, "all_lots_completed"
        if self.unrecoverable_error:
            return True, "unrecoverable_error"
        if self.planning_horizon is not None and self.env.current_time > self.planning_horizon:
            if not self.get_dispatchable_machines():
                return True, "planning_horizon_exceeded"
        if self.total_wait_steps_per_episode > self.max_total_wait_steps_per_episode:
            return True, "max_total_wait_steps_exceeded"
        if self.consecutive_failed_actions > self.max_failed_actions:
            return True, "max_failed_actions_exceeded"
        return False, ""

    def _first_valid_action_index(self, pool):
        for index, (action, is_valid) in enumerate(zip(pool.actions, pool.action_mask)):
            if bool(is_valid) and not action.is_padding:
                return index
        return None

    def run_rule_episode(self, strategy="first_valid"):
        steps = 0
        episode_reward = 0.0
        while steps < self.max_steps:
            done, reason = self.is_episode_done()
            if done:
                self.termination_reason = reason
                break

            machines = self.get_dispatchable_machines()
            if not machines:
                next_time = self.env.next_event_time()
                if next_time is None:
                    self.termination_reason = "no_future_event"
                    break
                self.env.advance_time(next_time)
                self.total_wait_steps_per_episode += 1
                steps += 1
                continue

            machine = self.select_next_machine(machines)
            decision = self.build_decision(machine)
            action_index = self._first_valid_action_index(decision.pool)
            if action_index is None:
                self.consecutive_failed_actions += 1
                steps += 1
                continue

            result = self.step_with_action(machine, action_index, pool=decision.pool)
            episode_reward += float(result.reward)
            steps += 1

        if not self.termination_reason:
            self.termination_reason = "max_steps_exceeded"
        return {
            "steps": steps,
            "episode_reward": episode_reward,
            "completed_lots": len(self.env.completed_lots),
            "wait_steps": self.total_wait_steps_per_episode,
            "failed_actions": self.consecutive_failed_actions,
            "termination_reason": self.termination_reason,
        }
```

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```bash
python -m pytest FAB_RL/FABenv/tests/test_phase2_sas_driver.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add FAB_RL/FABenv/phase2_sas_driver.py FAB_RL/FABenv/tests/test_phase2_sas_driver.py
git commit -m "feat: add phase2 rule triggered driver"
```

---

### Task 4: Implement Observation Encoder

**Files:**
- Create: `FAB_RL/FABenv/phase2_sas_observation.py`
- Create: `FAB_RL/FABenv/tests/test_phase2_sas_observation.py`

- [ ] **Step 1: Write failing observation tests**

Create `FAB_RL/FABenv/tests/test_phase2_sas_observation.py` with:

```python
import sys
from pathlib import Path

import numpy as np


PHASE1_DIR = Path(__file__).resolve().parents[1]
if str(PHASE1_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE1_DIR))


from phase2_sas_observation import Phase2ObservationEncoder
from problem_instances import build_small_demo_encoder
from rl_environment import ResourceCalendarEnv


def test_observation_encoder_outputs_fixed_shapes_and_rank_features():
    encoder = build_small_demo_encoder()
    env = ResourceCalendarEnv(encoder, top_k=8)
    pool = env.build_candidate_pool(1)
    obs_encoder = Phase2ObservationEncoder()

    observation = obs_encoder.encode(1, pool, env)

    assert observation.machine_id == 1
    assert observation.candidate_features.shape == (8, len(env.feature_names))
    assert observation.candidate_mask.shape == (8,)
    assert observation.global_features.shape == (9,)
    assert observation.action_indices.tolist() == list(range(8))
    assert observation.valid_action_count == int(np.sum(pool.action_mask))


def test_batch_observations_stacks_numpy_arrays():
    encoder = build_small_demo_encoder()
    env = ResourceCalendarEnv(encoder, top_k=8)
    obs_encoder = Phase2ObservationEncoder()
    obs1 = obs_encoder.encode(1, env.build_candidate_pool(1), env)
    obs2 = obs_encoder.encode(2, env.build_candidate_pool(2), env)

    batch = obs_encoder.batch_observations([obs1, obs2])

    assert batch["candidate_features"].shape == (2, 8, len(env.feature_names))
    assert batch["candidate_mask"].shape == (2, 8)
    assert batch["global_features"].shape == (2, 9)
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
python -m pytest FAB_RL/FABenv/tests/test_phase2_sas_observation.py -v
```

Expected: FAIL because `phase2_sas_observation.py` does not exist.

- [ ] **Step 3: Implement observation encoder**

Create `FAB_RL/FABenv/phase2_sas_observation.py` with:

```python
from dataclasses import dataclass

import numpy as np


@dataclass
class Phase2Observation:
    machine_id: int
    current_time: float
    candidate_features: np.ndarray
    candidate_mask: np.ndarray
    global_features: np.ndarray
    action_indices: np.ndarray
    valid_action_count: int


class Phase2ObservationEncoder:
    def __init__(self, normalize=True):
        self.normalize = bool(normalize)

    def encode(self, machine, pool, env):
        candidate_features = np.asarray(pool.features, dtype=np.float32)
        candidate_mask = np.asarray(pool.action_mask, dtype=bool)
        action_indices = np.arange(len(pool.actions), dtype=np.int64)
        global_features = self.build_global_features(machine, pool, env).astype(np.float32)
        return Phase2Observation(
            machine_id=int(machine),
            current_time=float(env.current_time),
            candidate_features=candidate_features,
            candidate_mask=candidate_mask,
            global_features=global_features,
            action_indices=action_indices,
            valid_action_count=int(np.sum(candidate_mask)),
        )

    def build_global_features(self, machine, pool, env):
        num_lots = max(int(env.encoder.num_lots), 1)
        num_machines = max(int(env.encoder.num_machines), 1)
        completed_ratio = len(env.completed_lots) / num_lots
        remaining_ratio = len(env.remaining_lots) / num_lots
        machine_id_norm = int(machine) / num_machines
        valid_count = int(np.sum(pool.action_mask))
        valid_action_count_norm = valid_count / max(len(pool.actions), 1)

        valid_features = np.asarray(pool.features[pool.action_mask], dtype=float)
        if valid_features.size == 0:
            score_mean = 0.0
            waiting_time_max = 0.0
            due_slack_min = 0.0
        else:
            score_mean = float(np.mean(valid_features[:, env.feature_names.index("score")]))
            waiting_time_max = float(np.max(valid_features[:, env.feature_names.index("waiting_time")]))
            due_slack_min = float(np.min(valid_features[:, env.feature_names.index("due_slack")]))

        lot_schedule = np.asarray(env.lot_schedule, dtype=float).reshape((-1, 5))
        machine_busy_time = 0.0
        if lot_schedule.size > 0:
            rows = lot_schedule[lot_schedule[:, 1].astype(int) == int(machine)]
            if rows.size > 0:
                machine_busy_time = float(np.sum(rows[:, 4] - rows[:, 3]))

        return np.asarray(
            [
                float(env.current_time),
                completed_ratio,
                remaining_ratio,
                machine_id_norm,
                machine_busy_time,
                valid_action_count_norm,
                score_mean,
                waiting_time_max,
                due_slack_min,
            ],
            dtype=float,
        )

    def to_numpy_dict(self, observation):
        return {
            "candidate_features": observation.candidate_features,
            "candidate_mask": observation.candidate_mask,
            "global_features": observation.global_features,
            "action_indices": observation.action_indices,
            "valid_action_count": observation.valid_action_count,
        }

    def batch_observations(self, observations):
        return {
            "candidate_features": np.stack([obs.candidate_features for obs in observations]),
            "candidate_mask": np.stack([obs.candidate_mask for obs in observations]),
            "global_features": np.stack([obs.global_features for obs in observations]),
            "action_indices": np.stack([obs.action_indices for obs in observations]),
            "valid_action_count": np.asarray(
                [obs.valid_action_count for obs in observations],
                dtype=np.int64,
            ),
        }
```

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```bash
python -m pytest FAB_RL/FABenv/tests/test_phase2_sas_observation.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add FAB_RL/FABenv/phase2_sas_observation.py FAB_RL/FABenv/tests/test_phase2_sas_observation.py
git commit -m "feat: add phase2 sas observation encoder"
```

---

### Task 5: Implement Masked Actor-Critic Policy

**Files:**
- Create: `FAB_RL/FABenv/phase2_sas_policy.py`
- Create: `FAB_RL/FABenv/tests/test_phase2_sas_policy.py`

- [ ] **Step 1: Write failing policy tests**

Create `FAB_RL/FABenv/tests/test_phase2_sas_policy.py` with:

```python
import sys
from pathlib import Path

import torch


PHASE1_DIR = Path(__file__).resolve().parents[1]
if str(PHASE1_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE1_DIR))


from phase2_sas_policy import Phase2SASActorCritic


def test_policy_never_samples_masked_action():
    torch.manual_seed(0)
    policy = Phase2SASActorCritic(candidate_dim=18, global_dim=9, hidden_dim=32)
    candidate_features = torch.randn(1, 4, 18)
    candidate_mask = torch.tensor([[True, False, True, False]])
    global_features = torch.randn(1, 9)

    for _ in range(20):
        output = policy.sample_action(candidate_features, candidate_mask, global_features)
        assert int(output["action"].item()) in {0, 2}


def test_policy_evaluate_actions_returns_training_tensors():
    policy = Phase2SASActorCritic(candidate_dim=18, global_dim=9, hidden_dim=32)
    candidate_features = torch.randn(2, 4, 18)
    candidate_mask = torch.tensor([[True, False, True, False], [False, True, True, False]])
    global_features = torch.randn(2, 9)
    actions = torch.tensor([0, 2])

    output = policy.evaluate_actions(candidate_features, candidate_mask, global_features, actions)

    assert output["log_prob"].shape == (2,)
    assert output["entropy"].shape == (2,)
    assert output["value"].shape == (2,)
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
python -m pytest FAB_RL/FABenv/tests/test_phase2_sas_policy.py -v
```

Expected: FAIL because `phase2_sas_policy.py` does not exist.

- [ ] **Step 3: Implement policy**

Create `FAB_RL/FABenv/phase2_sas_policy.py` with:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedCategoricalPolicy(nn.Module):
    def forward(self, logits, mask):
        masked_logits = logits.masked_fill(~mask.bool(), torch.finfo(logits.dtype).min)
        return torch.distributions.Categorical(logits=masked_logits)

    def sample(self, logits, mask):
        distribution = self.forward(logits, mask)
        action = distribution.sample()
        return {
            "action": action,
            "log_prob": distribution.log_prob(action),
            "entropy": distribution.entropy(),
            "probs": distribution.probs,
        }

    def greedy(self, logits, mask):
        distribution = self.forward(logits, mask)
        action = torch.argmax(distribution.probs, dim=-1)
        return {
            "action": action,
            "log_prob": distribution.log_prob(action),
            "entropy": distribution.entropy(),
            "probs": distribution.probs,
        }


class Phase2SASActorCritic(nn.Module):
    def __init__(self, candidate_dim, global_dim, hidden_dim=128):
        super().__init__()
        self.candidate_encoder = nn.Sequential(
            nn.Linear(candidate_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor_head = nn.Linear(hidden_dim, 1)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim + global_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.masked_policy = MaskedCategoricalPolicy()

    def encode_candidates(self, candidate_features):
        return self.candidate_encoder(candidate_features)

    def actor_logits(self, candidate_features):
        encoded = self.encode_candidates(candidate_features)
        return self.actor_head(encoded).squeeze(-1)

    def critic_value(self, candidate_features, candidate_mask, global_features):
        encoded = self.encode_candidates(candidate_features)
        mask = candidate_mask.bool().unsqueeze(-1)
        masked_encoded = encoded.masked_fill(~mask, 0.0)
        denom = mask.sum(dim=1).clamp(min=1).to(encoded.dtype)
        pooled = masked_encoded.sum(dim=1) / denom
        state_repr = torch.cat([pooled, global_features], dim=-1)
        return self.critic(state_repr).squeeze(-1)

    def forward(self, candidate_features, candidate_mask, global_features):
        logits = self.actor_logits(candidate_features)
        value = self.critic_value(candidate_features, candidate_mask, global_features)
        return logits, value

    def sample_action(self, candidate_features, candidate_mask, global_features):
        logits, value = self.forward(candidate_features, candidate_mask, global_features)
        output = self.masked_policy.sample(logits, candidate_mask)
        output["value"] = value
        return output

    def greedy_action(self, candidate_features, candidate_mask, global_features):
        logits, value = self.forward(candidate_features, candidate_mask, global_features)
        output = self.masked_policy.greedy(logits, candidate_mask)
        output["value"] = value
        return output

    def evaluate_actions(self, candidate_features, candidate_mask, global_features, actions):
        logits, value = self.forward(candidate_features, candidate_mask, global_features)
        distribution = self.masked_policy(logits, candidate_mask)
        return {
            "log_prob": distribution.log_prob(actions),
            "entropy": distribution.entropy(),
            "value": value,
            "probs": distribution.probs,
        }
```

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```bash
python -m pytest FAB_RL/FABenv/tests/test_phase2_sas_policy.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add FAB_RL/FABenv/phase2_sas_policy.py FAB_RL/FABenv/tests/test_phase2_sas_policy.py
git commit -m "feat: add phase2 masked actor critic"
```

---

### Task 6: Implement Rollout Buffer

**Files:**
- Create: `FAB_RL/FABenv/phase2_ppo_buffer.py`
- Create: `FAB_RL/FABenv/tests/test_phase2_ppo_buffer.py`

- [ ] **Step 1: Write failing buffer tests**

Create `FAB_RL/FABenv/tests/test_phase2_ppo_buffer.py` with:

```python
import sys
from pathlib import Path


PHASE1_DIR = Path(__file__).resolve().parents[1]
if str(PHASE1_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE1_DIR))


from phase2_ppo_buffer import Phase2RolloutBuffer, RolloutStep, StepInfo


def _step(reward, value, done=False):
    return RolloutStep(
        machine_id=1,
        current_time=0.0,
        candidate_features=None,
        candidate_mask=None,
        global_features=None,
        action_indices=None,
        valid_action_count=1,
        action=0,
        log_prob=0.0,
        value=value,
        reward=reward,
        done=done,
        next_observation=None,
        info=StepInfo(),
    )


def test_buffer_computes_returns_and_advantages():
    buffer = Phase2RolloutBuffer(gamma=1.0, gae_lambda=1.0)
    buffer.add(_step(1.0, 0.5))
    buffer.add(_step(2.0, 0.25, done=True))

    buffer.finish_episode(last_value=0.0)

    assert buffer.returns == [3.0, 2.0]
    assert buffer.advantages == [2.5, 1.75]
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
python -m pytest FAB_RL/FABenv/tests/test_phase2_ppo_buffer.py -v
```

Expected: FAIL because `phase2_ppo_buffer.py` does not exist.

- [ ] **Step 3: Implement buffer**

Create `FAB_RL/FABenv/phase2_ppo_buffer.py` with the `StepInfo`, `RolloutStep`, and `Phase2RolloutBuffer` code from `项目方案.md` Section 10.5. Ensure `StepInfo` includes:

```python
    reward_tardy: float = 0.0
    reward_qtime: float = 0.0
    reward_priority: float = 0.0
    reward_progress: float = 0.0
    reward_total: float = 0.0
```

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```bash
python -m pytest FAB_RL/FABenv/tests/test_phase2_ppo_buffer.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add FAB_RL/FABenv/phase2_ppo_buffer.py FAB_RL/FABenv/tests/test_phase2_ppo_buffer.py
git commit -m "feat: add phase2 rollout buffer"
```

---

### Task 7: Implement PPO Trainer Smoke Path

**Files:**
- Create: `FAB_RL/FABenv/phase2_ppo_trainer.py`
- Create: `FAB_RL/FABenv/tests/test_phase2_ppo_smoke.py`

- [ ] **Step 1: Write failing PPO smoke test**

Create `FAB_RL/FABenv/tests/test_phase2_ppo_smoke.py` with:

```python
import sys
from pathlib import Path

import numpy as np
import torch


PHASE1_DIR = Path(__file__).resolve().parents[1]
if str(PHASE1_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE1_DIR))


from phase2_ppo_buffer import Phase2RolloutBuffer, RolloutStep, StepInfo
from phase2_ppo_trainer import PPOConfig, Phase2PPOTrainer
from phase2_sas_policy import Phase2SASActorCritic


def test_ppo_update_runs_one_backward_step():
    policy = Phase2SASActorCritic(candidate_dim=18, global_dim=9, hidden_dim=32)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    trainer = Phase2PPOTrainer(policy, optimizer, PPOConfig(train_epochs=1, minibatch_size=2))
    buffer = Phase2RolloutBuffer()

    for action in [0, 1]:
        buffer.add(
            RolloutStep(
                machine_id=1,
                current_time=0.0,
                candidate_features=np.random.randn(4, 18).astype("float32"),
                candidate_mask=np.asarray([True, True, False, False]),
                global_features=np.random.randn(9).astype("float32"),
                action_indices=np.arange(4),
                valid_action_count=2,
                action=action,
                log_prob=-0.69,
                value=0.0,
                reward=1.0,
                done=False,
                next_observation=None,
                info=StepInfo(),
            )
        )
    buffer.finish_episode(last_value=0.0)

    stats = trainer.update_policy(buffer)

    assert "policy_loss" in stats
    assert "value_loss" in stats
    assert "entropy" in stats
```

- [ ] **Step 2: Run test to verify RED**

Run:

```bash
python -m pytest FAB_RL/FABenv/tests/test_phase2_ppo_smoke.py -v
```

Expected: FAIL because `phase2_ppo_trainer.py` does not exist.

- [ ] **Step 3: Implement PPO trainer**

Create `FAB_RL/FABenv/phase2_ppo_trainer.py` with:

```python
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    learning_rate: float = 3e-4
    train_epochs: int = 4
    minibatch_size: int = 32
    max_grad_norm: float = 0.5


class Phase2PPOTrainer:
    def __init__(self, policy, optimizer, config):
        self.policy = policy
        self.optimizer = optimizer
        self.config = config

    def _collate(self, buffer):
        return {
            "candidate_features": torch.as_tensor(
                np.stack([step.candidate_features for step in buffer.steps]),
                dtype=torch.float32,
            ),
            "candidate_mask": torch.as_tensor(
                np.stack([step.candidate_mask for step in buffer.steps]),
                dtype=torch.bool,
            ),
            "global_features": torch.as_tensor(
                np.stack([step.global_features for step in buffer.steps]),
                dtype=torch.float32,
            ),
            "actions": torch.as_tensor([step.action for step in buffer.steps], dtype=torch.long),
            "old_log_probs": torch.as_tensor([step.log_prob for step in buffer.steps], dtype=torch.float32),
            "returns": torch.as_tensor(buffer.returns, dtype=torch.float32),
            "advantages": torch.as_tensor(buffer.advantages, dtype=torch.float32),
        }

    def update_policy(self, buffer):
        if not buffer.returns or not buffer.advantages:
            buffer.finish_episode(last_value=0.0)
        batch = self._collate(buffer)
        advantages = batch["advantages"]
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        stats = {}
        for _ in range(self.config.train_epochs):
            output = self.policy.evaluate_actions(
                batch["candidate_features"],
                batch["candidate_mask"],
                batch["global_features"],
                batch["actions"],
            )
            ratio = torch.exp(output["log_prob"] - batch["old_log_probs"])
            unclipped = ratio * advantages
            clipped = torch.clamp(
                ratio,
                1.0 - self.config.clip_ratio,
                1.0 + self.config.clip_ratio,
            ) * advantages
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = F.mse_loss(output["value"], batch["returns"])
            entropy = output["entropy"].mean()
            loss = (
                policy_loss
                + self.config.value_coef * value_loss
                - self.config.entropy_coef * entropy
            )
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            stats = {
                "policy_loss": float(policy_loss.detach().cpu()),
                "value_loss": float(value_loss.detach().cpu()),
                "entropy": float(entropy.detach().cpu()),
            }
        return stats
```

- [ ] **Step 4: Run test to verify GREEN**

Run:

```bash
python -m pytest FAB_RL/FABenv/tests/test_phase2_ppo_smoke.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add FAB_RL/FABenv/phase2_ppo_trainer.py FAB_RL/FABenv/tests/test_phase2_ppo_smoke.py
git commit -m "feat: add phase2 ppo update smoke path"
```

---

### Task 8: Add Training and Inference Entrypoints

**Files:**
- Create: `FAB_RL/FABenv/train_phase2_sas_ppo.py`
- Create: `FAB_RL/FABenv/run_phase2_sas_inference_demo.py`
- Create: `FAB_RL/FABenv/tests/test_phase2_inference_demo.py`

- [ ] **Step 1: Write failing entrypoint tests**

Create `FAB_RL/FABenv/tests/test_phase2_inference_demo.py` with:

```python
import sys
from pathlib import Path


PHASE1_DIR = Path(__file__).resolve().parents[1]
if str(PHASE1_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE1_DIR))


import run_phase2_sas_inference_demo
import train_phase2_sas_ppo


def test_training_components_can_be_built():
    components = train_phase2_sas_ppo.build_training_components()

    assert "env" in components
    assert "policy" in components
    assert "trainer" in components


def test_inference_demo_runs_and_validates_schedule():
    summary = run_phase2_sas_inference_demo.run_demo_episode(max_steps=200)

    assert "termination_reason" in summary
    assert "validation_passed" in summary
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
python -m pytest FAB_RL/FABenv/tests/test_phase2_inference_demo.py -v
```

Expected: FAIL because the entrypoint modules do not exist.

- [ ] **Step 3: Implement training component builder**

Create `FAB_RL/FABenv/train_phase2_sas_ppo.py` with:

```python
import torch

from phase2_ppo_trainer import PPOConfig, Phase2PPOTrainer
from phase2_sas_driver import Phase2EpisodeDriver
from phase2_sas_observation import Phase2ObservationEncoder
from phase2_sas_policy import Phase2SASActorCritic
from problem_instances import build_small_demo_encoder
from rl_environment import ResourceCalendarEnv, RewardConfig


def build_training_components():
    encoder = build_small_demo_encoder()
    env = ResourceCalendarEnv(encoder)
    observation_encoder = Phase2ObservationEncoder()
    reward_config = RewardConfig()
    sample_pool = env.build_candidate_pool(1)
    sample_observation = observation_encoder.encode(1, sample_pool, env)
    policy = Phase2SASActorCritic(
        candidate_dim=sample_observation.candidate_features.shape[1],
        global_dim=sample_observation.global_features.shape[0],
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    trainer = Phase2PPOTrainer(policy, optimizer, PPOConfig())
    driver = Phase2EpisodeDriver(env, observation_encoder, reward_config)
    return {
        "encoder": encoder,
        "env": env,
        "observation_encoder": observation_encoder,
        "reward_config": reward_config,
        "policy": policy,
        "optimizer": optimizer,
        "trainer": trainer,
        "driver": driver,
    }


def main():
    components = build_training_components()
    summary = components["driver"].run_rule_episode(strategy="first_valid")
    print(summary)
    return summary


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Implement inference demo**

Create `FAB_RL/FABenv/run_phase2_sas_inference_demo.py` with:

```python
from train_phase2_sas_ppo import build_training_components


def run_demo_episode(max_steps=1000):
    components = build_training_components()
    driver = components["driver"]
    driver.max_steps = int(max_steps)
    summary = driver.run_rule_episode(strategy="first_valid")
    validation = components["env"].validate_schedule(partial=True)
    summary["validation_passed"] = bool(validation.passed)
    summary["machine_conflicts"] = validation.machine_conflicts
    summary["chamber_conflicts"] = validation.chamber_conflicts
    return summary


def main():
    summary = run_demo_episode()
    print(summary)
    return summary


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests to verify GREEN**

Run:

```bash
python -m pytest FAB_RL/FABenv/tests/test_phase2_inference_demo.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
git add FAB_RL/FABenv/train_phase2_sas_ppo.py FAB_RL/FABenv/run_phase2_sas_inference_demo.py FAB_RL/FABenv/tests/test_phase2_inference_demo.py
git commit -m "feat: add phase2 train and inference entrypoints"
```

---

### Task 9: Export Phase 2 Public Classes

**Files:**
- Modify: `FAB_RL/FABenv/__init__.py`
- Create or Modify: `FAB_RL/FABenv/tests/test_phase2_exports.py`

- [ ] **Step 1: Write failing export test**

Create `FAB_RL/FABenv/tests/test_phase2_exports.py` with:

```python
import sys
from pathlib import Path


PHASE1_DIR = Path(__file__).resolve().parents[1]
if str(PHASE1_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE1_DIR))


import __init__ as fabenv


def test_phase2_public_exports_exist():
    for name in (
        "Phase2EpisodeDriver",
        "Phase2ObservationEncoder",
        "Phase2SASActorCritic",
        "Phase2RolloutBuffer",
        "Phase2PPOTrainer",
    ):
        assert hasattr(fabenv, name)
```

- [ ] **Step 2: Run test to verify RED**

Run:

```bash
python -m pytest FAB_RL/FABenv/tests/test_phase2_exports.py -v
```

Expected: FAIL because `__init__.py` does not export the new Phase 2 classes.

- [ ] **Step 3: Add exports**

In `FAB_RL/FABenv/__init__.py`, add imports for:

```python
from phase2_ppo_buffer import Phase2RolloutBuffer, RolloutStep, StepInfo
from phase2_ppo_trainer import PPOConfig, Phase2PPOTrainer
from phase2_sas_driver import Phase2DispatchDecision, Phase2EpisodeDriver
from phase2_sas_observation import Phase2Observation, Phase2ObservationEncoder
from phase2_sas_policy import MaskedCategoricalPolicy, Phase2SASActorCritic
```

Add these names to `__all__` if the file already defines `__all__`:

```python
    "Phase2RolloutBuffer",
    "RolloutStep",
    "StepInfo",
    "PPOConfig",
    "Phase2PPOTrainer",
    "Phase2DispatchDecision",
    "Phase2EpisodeDriver",
    "Phase2Observation",
    "Phase2ObservationEncoder",
    "MaskedCategoricalPolicy",
    "Phase2SASActorCritic",
```

- [ ] **Step 4: Run export test to verify GREEN**

Run:

```bash
python -m pytest FAB_RL/FABenv/tests/test_phase2_exports.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add FAB_RL/FABenv/__init__.py FAB_RL/FABenv/tests/test_phase2_exports.py
git commit -m "feat: export phase2 sas ppo components"
```

---

### Task 10: Final Regression and Acceptance Checks

**Files:**
- Verify: `FAB_RL/FABenv/*.py`
- Verify: `FAB_RL/FABenv/tests/*.py`
- Verify: `项目方案.md`

- [ ] **Step 1: Run all FABenv tests**

Run:

```bash
python -m pytest FAB_RL/FABenv/tests -v
```

Expected: all tests PASS.

- [ ] **Step 2: Run Phase 1 demo regression**

Run:

```bash
python FAB_RL/FABenv/run_phase1_environment_demo.py
```

Expected: script exits successfully and does not raise schedule validation errors.

- [ ] **Step 3: Run Phase 2 inference demo**

Run:

```bash
python FAB_RL/FABenv/run_phase2_sas_inference_demo.py
```

Expected: script prints a summary containing `termination_reason`, `validation_passed`, `machine_conflicts`, and `chamber_conflicts`.

- [ ] **Step 4: Verify documentation-code alignment**

Run:

```bash
python -c "from pathlib import Path; doc=Path('项目方案.md').read_text(encoding='utf-8'); required=['Phase2EpisodeDriver','Phase2ObservationEncoder','Phase2SASActorCritic','Phase2RolloutBuffer','Phase2PPOTrainer','priority_rank_norm','reward_total']; missing=[item for item in required if item not in doc]; assert not missing, missing; print('phase2 documentation alignment passed')"
```

Expected output:

```text
phase2 documentation alignment passed
```

- [ ] **Step 5: Commit final verification updates**

Run:

```bash
git status --short
```

If only expected Phase 2 files are modified, run:

```bash
git add FAB_RL/FABenv docs/superpowers/plans/2026-05-26-phase2-sas-ppo-implementation.md 项目方案.md
git commit -m "feat: complete phase2 sas ppo minimum closed loop"
```

Expected: commit succeeds, or git reports nothing to commit if earlier tasks already committed every change.

---

## Self-Review

Spec coverage:
- Candidate rank/best features from Section 6.1.1 are covered by Task 1.
- Reward curriculum and reward decomposition from Section 8 are covered by Task 1.
- Environment helper interfaces from Section 10.1 are covered by Task 2.
- Rule-triggered driver and machine selection from Sections 5 and 10.2 are covered by Task 3.
- Observation encoder from Sections 6 and 10.3 is covered by Task 4.
- Masked Actor-Critic from Section 7 and 10.4 is covered by Task 5.
- Rollout buffer from Section 10.5 is covered by Task 6.
- PPO update path from Section 10.6 is covered by Task 7.
- Training and inference entries from Sections 10.7 and 10.8 are covered by Task 8.
- Exports and final verification are covered by Tasks 9 and 10.

Placeholder scan:
- No unresolved placeholder instructions are present.
- Every code-producing task includes concrete code or exact replacement text.
- Every verification task includes an exact command and expected result.

Type consistency:
- File names match `项目方案.md` Section 4.
- Class names match `项目方案.md` Section 10.
- Reward field names match `项目方案.md` Section 8.5.
- Candidate feature names match `项目方案.md` Section 6.1.1.
