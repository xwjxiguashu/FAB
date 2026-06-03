
"""Phase 2 SAS Episode Driver — 编排完整的调度 episode。

Phase2EpisodeDriver 管理从初始状态到终止条件的完整 episode 流程:
  - 机台选择: 最早可用机台优先 (打破平局用候选数)
  - 事件推进: 跳到下一个到达/释放时间
  - 三种 episode 模式:
    1. run_rule_episode:     启发式 "first-valid" 规则
    2. run_policy_episode:   随机 PPO 策略 + rollout buffer
    3. run_greedy_episode:   贪心策略 + 失败回退到 wait

终止条件:
  - all_lots_completed:      所有 Lot 已完成
  - max_steps_exceeded:      超过最大步数
  - max_total_wait_steps:    全局等待步数超限
  - max_failed_actions:      连续失败动作超限
  - planning_horizon:        时间超窗口
  - no_future_event:         无机器可派且无未来事件
"""

from dataclasses import dataclass

import numpy as np
import torch

from phase2_ppo_buffer import (
    MULTIHEAD_CHANNELS,
    MultiHeadRolloutStep,
    RolloutStep,
    StepInfo,
)
from rl_environment import RewardVectorConfig, compute_sas_reward_vector


@dataclass
class Phase2DispatchDecision:
    """单次派工决策的上下文快照。

    Attributes:
        machine: 选中的机台。
        pool: CandidatePool 实例。
        observation: Phase2Observation 实例。
        current_time: 决策时刻。
    """
    machine: int
    pool: object
    observation: object
    current_time: float


class Phase2EpisodeDriver:
    """Phase 2 SAS Episode 驱动器。

    负责:
      - 机台选择 (select_next_machine): 最早可用 → 最多候选 → 最小 ID
      - 决策构建 (build_decision): 候选池 + 观察编码
      - 步骤执行 (step_with_action): 调用 env.sas_step() 并记录结果
      - 事件推进 (advance_to_next_event): 跳转到下一个到达/释放时间
      - 终止判断 (is_episode_done): 所有终止条件检查
    """

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
        # 连续失败阈值: 默认 3 * top_k
        self.max_failed_actions = (
            int(max_failed_actions)
            if max_failed_actions is not None
            else 3 * int(getattr(env, "top_k", 8))
        )
        # Episode 级计数器
        self.total_wait_steps_per_episode = 0
        self.consecutive_failed_actions = 0
        self.failed_actions_per_episode = 0
        self.unrecoverable_error = False
        self.termination_reason = ""

    # ---- Episode 生命周期 ----

    def reset_episode(self):
        """重置 episode 状态，返回 env.reset() 的结果。"""
        self.total_wait_steps_per_episode = 0
        self.consecutive_failed_actions = 0
        self.failed_actions_per_episode = 0
        self.unrecoverable_error = False
        self.termination_reason = ""
        return self.env.reset()

    # ---- 机台选择与决策 ----

    def get_dispatchable_machines(self):
        """返回当前可派工的机台列表 (至少有一个真实候选)。"""
        return self.env.get_candidate_machines()

    def select_next_machine(self, machines):
        """选择下一个派工机台。

        优先级:
          1. 最早可用时间 (machine_available_time)
          2. 真实候选数 (越多越好)
          3. 机台 ID (越小越好)
        """
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
        """为指定机台构建派工决策 (候选池 + 观察编码)。"""
        pool = self.env.build_candidate_pool(machine)
        observation = self.observation_encoder.encode(machine, pool, self.env)
        return Phase2DispatchDecision(
            machine=int(machine),
            pool=pool,
            observation=observation,
            current_time=self.env.current_time,
        )

    def step_with_action(self, machine, action_index, pool=None):
        """执行一步 (env.sas_step) 并记录结果。"""
        result = self.env.sas_step(
            machine,
            action_index,
            pool=pool,
            reward_config=self.reward_config,
        )
        self.record_step_result(result)
        return result

    def advance_to_next_event(self):
        """推进时间到下一个事件 (到达或资源释放)。

        Returns:
            下一个事件时间，或 None (无未来事件)。
        """
        next_time = self.env.next_event_time()
        if next_time is None:
            return None
        if float(next_time) <= float(self.env.current_time):
            self.unrecoverable_error = True
            self.termination_reason = "unrecoverable_error"
            return None
        self.env.advance_time(next_time)
        self.total_wait_steps_per_episode += 1
        return float(next_time)

    # ---- 步骤结果记录 ----

    def record_step_result(self, step_result):
        """更新 episode 级计数器 (失败/等待/连续失败)。"""
        info = step_result.info
        if info.get("insertion_success"):
            self.consecutive_failed_actions = 0
            return
        if info.get("mask_invalid") or info.get("insertion_failed"):
            self.consecutive_failed_actions += 1
            self.failed_actions_per_episode += 1
        if info.get("wait_or_noop"):
            self.total_wait_steps_per_episode += 1

    def is_episode_done(self):
        """检查 episode 是否满足终止条件。

        Returns:
            (is_done: bool, reason: str)
        """
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
        if not self.get_dispatchable_machines() and self.env.next_event_time() is None:
            return True, "no_future_event"
        return False, ""

    # ---- 内部辅助 ----

    def _first_valid_action_index(self, pool):
        """返回候选池中第一个有效真实动作的索引 (用于 rule-based)。"""
        for index, (action, is_valid) in enumerate(zip(pool.actions, pool.action_mask)):
            if bool(is_valid) and not action.is_padding:
                return index
        return None

    def _valid_real_candidates(self, pool):
        """返回 [(index, action)]，仅含有效、非 padding、非 wait、ppid≠0 的真实候选。"""
        out = []
        for index, (action, is_valid) in enumerate(zip(pool.actions, pool.action_mask)):
            if not bool(is_valid) or action.is_padding or getattr(action, "is_wait", False):
                continue
            if int(action.ppid) == 0:
                continue
            out.append((index, action))
        return out

    def _rule_action_index(self, pool, strategy):
        """按派工规则在有效真实候选中选动作索引 (报告 §7.4 基线)。

        规则键 (lot 编号作为确定性 tie-break):
          first_valid: 池序第一个
          FIFO: 最早到达 (arrival 升序)
          SPT:  最短加工时间 (proc 升序)
          EDD:  最早交期 (due_date 升序)
          CR:   临界比 (due-now)/proc 升序 (越小越紧急)
          ATC:  表观拖期成本 score 降序, score = (priority/proc)·exp(-max(0,due-proc-now)/(κ·p̄))
        proc 用下层估时器的相对 makespan μ (走缓存，便宜)。
        """
        candidates = self._valid_real_candidates(pool)
        if not candidates:
            return None
        if strategy == "first_valid":
            return candidates[0][0]

        from lower_layer_estimator import estimate

        enc = self.env.encoder
        now = float(self.env.current_time)
        cache = getattr(self.env, "_estimate_cache", None)

        rows = []  # (index, lot, due, arrival, proc, priority)
        for index, action in candidates:
            lot = int(action.lot)
            due = float(enc.due_dates.get(lot, np.inf))
            arrival = float(enc.arrival_times.get(lot, now))
            try:
                proc = float(estimate(
                    lot, int(action.machine), int(action.ppid),
                    enc, self.env.state, n_mc=10, cache=cache,
                )["mu_finish"])
            except Exception:
                proc = 1.0
            proc = max(proc, 1e-9)
            priority = float(enc.priorities.get(lot, 1.0))
            rows.append((index, lot, due, arrival, proc, priority))

        if strategy == "FIFO":
            best = min(rows, key=lambda r: (r[3], r[1]))
        elif strategy == "SPT":
            best = min(rows, key=lambda r: (r[4], r[1]))
        elif strategy == "EDD":
            best = min(rows, key=lambda r: (r[2], r[1]))
        elif strategy == "CR":
            best = min(rows, key=lambda r: ((r[2] - now) / r[4], r[1]))
        elif strategy == "ATC":
            p_bar = sum(r[4] for r in rows) / len(rows)
            kappa = 3.0

            def atc_score(r):
                slack = max(0.0, r[2] - r[4] - now)
                return (r[5] / r[4]) * float(np.exp(-slack / (kappa * p_bar)))

            # 最大 ATC score；tie-break 用较小 lot (取负转为 min)
            best = max(rows, key=lambda r: (atc_score(r), -r[1]))
        else:  # pragma: no cover - run_rule_episode 已校验
            raise ValueError(f"unknown strategy: {strategy}")
        return best[0]

    def _policy_device(self, policy):
        """获取策略网络所在的 torch device。"""
        try:
            return next(policy.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _policy_tensors(self, policy, observation):
        """将 Phase2Observation 转换为策略网络的输入张量 (添加 batch 维度)。"""
        device = self._policy_device(policy)
        return (
            torch.as_tensor(observation.candidate_features, dtype=torch.float32).unsqueeze(0).to(device),
            torch.as_tensor(observation.candidate_mask, dtype=torch.bool).unsqueeze(0).to(device),
            torch.as_tensor(observation.global_features, dtype=torch.float32).unsqueeze(0).to(device),
        )

    def _policy_output(self, policy, observation, stochastic):
        """运行策略网络前向传播 (no_grad)，返回采样/贪心输出。"""
        candidate_features, candidate_mask, global_features = self._policy_tensors(policy, observation)
        with torch.no_grad():
            if stochastic:
                return policy.sample_action(candidate_features, candidate_mask, global_features)
            return policy.greedy_action(candidate_features, candidate_mask, global_features)

    def _next_observation(self):
        """获取下一状态的观察 (若 episode 已结束或无可派工机台则返回 None)。"""
        done = self.is_episode_done()[0]
        if done:
            return None
        machines = self.get_dispatchable_machines()
        if not machines:
            return None
        machine = self.select_next_machine(machines)
        return self.build_decision(machine).observation

    def _step_info(self, info):
        """从 sas_step info 字典构建 StepInfo 数据类。"""
        return StepInfo(
            selected_lot=info.get("selected_lot"),
            selected_ppid=info.get("selected_ppid"),
            insertion_success=bool(info.get("insertion_success", False)),
            insertion_failed=bool(info.get("insertion_failed", False)),
            mask_invalid=bool(info.get("mask_invalid", False)),
            wait_or_noop=bool(info.get("wait_or_noop", False)),
            selected_lot_start=info.get("selected_lot_start"),
            selected_lot_end=info.get("selected_lot_end"),
            selected_lot_process_time=info.get("selected_lot_process_time"),
            new_qtime_violation=float(info.get("new_qtime_violation", 0.0)),
            priority_rank_penalty=float(info.get("priority_rank_penalty", 0.0)),
            reward_execute=float(info.get("reward_execute", 0.0)),
            reward_wait=float(info.get("reward_wait", 0.0)),
            reward_tardy=float(info.get("reward_tardy", 0.0)),
            reward_qtime=float(info.get("reward_qtime", 0.0)),
            reward_priority=float(info.get("reward_priority", 0.0)),
            reward_progress=float(info.get("reward_progress", 0.0)),
            reward_shape=float(info.get("reward_shape", 0.0)),
            reward_terminal=float(info.get("reward_terminal", 0.0)),
            reward_total=float(info.get("reward_total", 0.0)),
            raw=dict(info),
        )

    def _add_rollout_step(self, buffer, decision, action_index, policy_output, result, done):
        """将一步记录写入 RolloutBuffer (用于 PPO 训练)。"""
        if buffer is None:
            return
        next_observation = None if done else self._next_observation()
        buffer.add(
            RolloutStep(
                machine_id=int(decision.machine),
                current_time=float(decision.current_time),
                candidate_features=decision.observation.candidate_features.copy(),
                candidate_mask=decision.observation.candidate_mask.copy(),
                global_features=decision.observation.global_features.copy(),
                action_indices=decision.observation.action_indices.copy(),
                valid_action_count=int(decision.observation.valid_action_count),
                action=int(action_index),
                log_prob=float(policy_output["log_prob"].detach().cpu().reshape(-1)[0]),
                value=float(policy_output["value"].detach().cpu().reshape(-1)[0]),
                reward=float(result.reward),
                done=bool(done),
                next_observation=next_observation,
                info=self._step_info(result.info),
            )
        )

    def _add_multihead_rollout_step(
        self, buffer, decision, action_index, policy_output, result, done, reward_vector_config
    ):
        """将一步记录写入 MultiHeadRolloutBuffer (多头逐通道 PPO 训练)。"""
        if buffer is None:
            return
        next_observation = None if done else self._next_observation()
        rv = compute_sas_reward_vector(result.info, reward_vector_config)
        values = {
            c: float(policy_output["values"][c].detach().cpu().reshape(-1)[0])
            for c in MULTIHEAD_CHANNELS
        }
        buffer.add(
            MultiHeadRolloutStep(
                machine_id=int(decision.machine),
                current_time=float(decision.current_time),
                candidate_features=decision.observation.candidate_features.copy(),
                candidate_mask=decision.observation.candidate_mask.copy(),
                global_features=decision.observation.global_features.copy(),
                action_indices=decision.observation.action_indices.copy(),
                valid_action_count=int(decision.observation.valid_action_count),
                action=int(action_index),
                log_prob=float(policy_output["log_prob"].detach().cpu().reshape(-1)[0]),
                values=values,
                reward_vector=rv["reward_vector"],
                done=bool(done),
                next_observation=next_observation,
                info=self._step_info(result.info),
            )
        )

    def _summary(self, steps, episode_reward):
        """构建 episode 摘要字典。"""
        if not self.termination_reason:
            self.termination_reason = "max_steps_exceeded"
        # 原始(未归一化)指标，用于看真实学习曲线 (mean_reward 被常数通道淹没)。
        avg_utilization = 0.0
        qtime_violation_count = 0.0
        try:
            objs = self.env.encoder.evaluate_objectives(
                self.env.lot_schedule, self.env.wafer_schedule, current_time=0.0,
            )
            qtime_violation_count = float(objs[0])
            avg_utilization = float(-objs[5])
        except Exception:
            pass
        return {
            "steps": int(steps),
            "episode_reward": float(episode_reward),
            "completed_lots": len(self.env.completed_lots),
            "avg_utilization": avg_utilization,
            "qtime_violation_count": qtime_violation_count,
            "wait_steps": self.total_wait_steps_per_episode,
            "failed_actions": self.failed_actions_per_episode,
            "consecutive_failed_actions": self.consecutive_failed_actions,
            "termination_reason": self.termination_reason,
        }

    # ==========================================================================
    # 三种 Episode 模式
    # ==========================================================================

    #: 支持的派工规则基线 (报告 §7.4)。first_valid 为最朴素基线 (向后兼容)。
    RULE_STRATEGIES = ("first_valid", "FIFO", "SPT", "EDD", "CR", "ATC")

    def run_rule_episode(self, strategy="first_valid"):
        """运行启发式规则 episode (基准性能评估，不涉及 RL)。

        每个决策步:
          1. 选机台 → 构建候选池 (qtime-safe + priority 过滤后)
          2. 按 strategy 在有效真实候选中排序选动作
          3. 若无候选: 推进时间到下一个事件
          4. 执行动作 + 累积奖励

        Args:
            strategy: 派工规则之一 (RULE_STRATEGIES):
              - first_valid: 池中第一个有效动作 (朴素基线)
              - FIFO: 最早到达优先
              - SPT:  最短加工时间优先
              - EDD:  最早交期优先
              - CR:   临界比 (due-now)/proc 最小优先
              - ATC:  表观拖期成本 (优先级加权) 最大优先
            所有规则在与 RL 相同的 qtime-safe 候选池上排序，保证约束处理一致 (公平对比)。

        Raises:
            ValueError: strategy 不在 RULE_STRATEGIES 中。
        """
        if strategy not in self.RULE_STRATEGIES:
            raise ValueError(
                f"unknown strategy {strategy!r}; expected one of {self.RULE_STRATEGIES}"
            )
        steps = 0
        episode_reward = 0.0
        while steps < self.max_steps:
            done, reason = self.is_episode_done()
            if done:
                self.termination_reason = reason
                break

            machines = self.get_dispatchable_machines()
            if not machines:
                next_time = self.advance_to_next_event()
                if next_time is None:
                    if not self.termination_reason:
                        self.termination_reason = "no_future_event"
                    break
                steps += 1
                continue

            machine = self.select_next_machine(machines)
            decision = self.build_decision(machine)
            action_index = self._rule_action_index(decision.pool, strategy)
            if action_index is None:
                self.consecutive_failed_actions += 1
                self.failed_actions_per_episode += 1
                steps += 1
                continue

            result = self.step_with_action(machine, action_index, pool=decision.pool)
            episode_reward += float(result.reward)
            steps += 1

        return self._summary(steps, episode_reward)

    def run_policy_episode(self, policy, buffer=None, stochastic=True):
        """运行 PPO 策略 episode (训练用)。

        每个决策步:
          1. 选机台 → 策略网络前向 → 采样动作
          2. 执行动作 → 记录到 buffer (含 next_observation)
          3. 累积奖励

        Args:
            policy: Phase2SASActorCritic 策略网络。
            buffer: Phase2RolloutBuffer (None 时仅运行不记录)。
            stochastic: True=采样, False=贪心。

        Returns:
            episode 摘要字典。
        """
        steps = 0
        episode_reward = 0.0
        while steps < self.max_steps:
            done, reason = self.is_episode_done()
            if done:
                self.termination_reason = reason
                break

            machines = self.get_dispatchable_machines()
            if not machines:
                next_time = self.advance_to_next_event()
                if next_time is None:
                    if not self.termination_reason:
                        self.termination_reason = "no_future_event"
                    break
                steps += 1
                continue

            machine = self.select_next_machine(machines)
            decision = self.build_decision(machine)
            policy_output = self._policy_output(policy, decision.observation, stochastic)
            action_index = int(policy_output["action"].detach().cpu().reshape(-1)[0])
            result = self.step_with_action(machine, action_index, pool=decision.pool)
            episode_reward += float(result.reward)
            steps += 1

            # 在 step 之后判断 done 并记录 (含 next_observation)
            done, reason = self.is_episode_done()
            self._add_rollout_step(buffer, decision, action_index, policy_output, result, done)
            if done:
                self.termination_reason = reason
                break

        return self._summary(steps, episode_reward)

    def run_multihead_policy_episode(
        self, policy, buffer=None, stochastic=True, reward_vector_config=None
    ):
        """运行多头逐通道 PPO 策略 episode (训练用)。

        与 run_policy_episode 的区别:
          1. policy 为 Phase2SASMultiHeadActorCritic, sample/greedy 输出含 "values" dict。
          2. 每步成功后用 compute_sas_reward_vector 计算 4 通道向量奖励。
          3. 通过 _add_multihead_rollout_step 写入 MultiHeadRolloutBuffer。

        Args:
            policy: Phase2SASMultiHeadActorCritic 策略网络。
            buffer: MultiHeadRolloutBuffer (None 时仅运行不记录)。
            stochastic: True=采样, False=贪心。
            reward_vector_config: RewardVectorConfig (None 时使用默认)。

        Returns:
            episode 摘要字典 (episode_reward 为各步向量奖励之和)。
        """
        if reward_vector_config is None:
            reward_vector_config = RewardVectorConfig()

        steps = 0
        episode_reward = 0.0
        while steps < self.max_steps:
            done, reason = self.is_episode_done()
            if done:
                self.termination_reason = reason
                break

            machines = self.get_dispatchable_machines()
            if not machines:
                next_time = self.advance_to_next_event()
                if next_time is None:
                    if not self.termination_reason:
                        self.termination_reason = "no_future_event"
                    break
                steps += 1
                continue

            machine = self.select_next_machine(machines)
            decision = self.build_decision(machine)
            policy_output = self._policy_output(policy, decision.observation, stochastic)
            action_index = int(policy_output["action"].detach().cpu().reshape(-1)[0])
            # 多头路径用向量奖励，sas_step 的标量奖励不使用，故传 reward_config=None
            # 以使用默认标量 RewardConfig（避免向量配置缺失 wait_penalty 等字段）。
            result = self.env.sas_step(machine, action_index, pool=decision.pool, reward_config=None)
            self.record_step_result(result)
            rv = compute_sas_reward_vector(result.info, reward_vector_config)
            episode_reward += float(rv["reward_vector"].sum())
            steps += 1

            # 在 step 之后判断 done 并记录 (含 next_observation)
            done, reason = self.is_episode_done()
            self._add_multihead_rollout_step(
                buffer, decision, action_index, policy_output, result, done, reward_vector_config
            )
            if done:
                self.termination_reason = reason
                break

        return self._summary(steps, episode_reward)

    def run_greedy_episode(self, policy):
        """运行贪心策略 episode (推理/评估用)。

        与 run_policy_episode 的区别:
          1. 使用贪心动作 (argmax probs)
          2. 贪心失败时回退到下一个有效动作或 wait
          3. 不记录到 buffer
        """
        steps = 0
        episode_reward = 0.0
        while steps < self.max_steps:
            done, reason = self.is_episode_done()
            if done:
                self.termination_reason = reason
                break

            machines = self.get_dispatchable_machines()
            if not machines:
                next_time = self.advance_to_next_event()
                if next_time is None:
                    if not self.termination_reason:
                        self.termination_reason = "no_future_event"
                    break
                steps += 1
                continue

            machine = self.select_next_machine(machines)
            decision = self.build_decision(machine)
            policy_output = self._policy_output(policy, decision.observation, stochastic=False)

            # 按概率降序排列有效动作 (真实动作优先, wait 备选)
            probs = policy_output["probs"].detach().cpu().reshape(-1)
            ordered_indices = torch.argsort(probs, descending=True).tolist()
            real_indices = []
            wait_indices = []
            for action_index in ordered_indices:
                action = decision.pool.actions[int(action_index)]
                if not bool(decision.pool.action_mask[int(action_index)]) or action.is_padding:
                    continue
                if action.is_wait:
                    wait_indices.append(int(action_index))
                else:
                    real_indices.append(int(action_index))

            # 依次尝试真实动作 → wait，直到成功
            adopted_result = None
            for action_index in real_indices + wait_indices:
                result = self.step_with_action(machine, int(action_index), pool=decision.pool)
                adopted_result = result
                if result.committed or result.info.get("wait_or_noop"):
                    break

            if adopted_result is not None:
                episode_reward += float(adopted_result.reward)
            else:
                self.consecutive_failed_actions += 1
                self.failed_actions_per_episode += 1

            steps += 1
            done, reason = self.is_episode_done()
            if done:
                self.termination_reason = reason
                break

        return self._summary(steps, episode_reward)