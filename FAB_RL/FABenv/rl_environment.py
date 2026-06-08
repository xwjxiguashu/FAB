"""FAB 资源日历 RL 环境 — 单步派工/提交/回滚/验证的完整调度交互循环。

本模块是 Phase 1 的核心，提供:
  1. 数据类: DispatchAction, CandidatePool, RewardConfig, SASStepResult 等
  2. ResourceCalendarEnv: 完整 RL 环境，实现候选生成、打分、掩码、提交、回滚
  3. 奖励函数: compute_sas_reward / compute_sas_reward_components

调度流程:
  build_candidate_pool(machine) → 按启发式打分生成 Top-K 候选 + wait/padding
  sas_step(machine, action_index)  → 执行动作 + 计算奖励 + 状态转移
  validate_schedule()              → 校验最终调度的完整性与无冲突性
"""

from dataclasses import dataclass, field

import numpy as np

from state import ScheduleState


# =============================================================================
# 动作与结果数据类
# =============================================================================


@dataclass(frozen=True)
class DispatchAction:
    """一次派工动作。

    Attributes:
        lot: Lot 编号 (1-indexed)。wait 动作用首个候选的 lot；padding 用 0。
        machine: 目标机台编号 (1-indexed)。
        ppid: 工艺配方编号。wait 和 padding 时为 0。
        is_wait: True 表示等待动作 (不派工，让时间推进)。
        is_padding: True 表示填充动作 (占位，始终被掩码遮蔽)。
    """
    lot: int
    machine: int
    ppid: int
    is_wait: bool = False
    is_padding: bool = False


@dataclass
class DryRunResult:
    """dry-run (试算) 结果 — 在不修改环境状态的前提下模拟一次派工。

    用于候选打分和掩码验证，不产生副作用。
    """
    action: DispatchAction
    success: bool
    lot_schedule: np.ndarray     # (1, 5) 或空
    wafer_schedule: np.ndarray   # (n_wafers*n_stages, 9) 或空
    state: ScheduleState         # dry-run 后的状态副本
    machine_intervals: list = field(default_factory=list)
    chamber_intervals: list = field(default_factory=list)
    failure_reason: str = ""


@dataclass
class RollbackResult:
    """回滚结果 — 撤销最近一次 commit 后的状态。"""
    action: DispatchAction
    rolled_back: bool
    state: ScheduleState
    step_info: dict = field(default_factory=dict)
    failure_reason: str = ""


@dataclass
class DispatchCommitResult:
    """派工提交结果 — 确认将动作写入环境状态。"""
    action: DispatchAction
    lot_schedule: np.ndarray
    wafer_schedule: np.ndarray
    state: ScheduleState
    committed: bool = False
    commit_log: dict = field(default_factory=dict)
    step_info: dict = field(default_factory=dict)
    dry_run_result: object = None
    failure_reason: str = ""


@dataclass
class CandidatePool:
    """候选动作池 — 单机台上的 Top-K 候选 + 掩码 + 特征矩阵。

    Attributes:
        machine: 机台编号。
        actions: 候选动作列表 (real + wait + padding)。
        action_mask: bool 数组，True=有效动作。
        features: (pool_size, 18) 候选特征矩阵。
        invalid_reasons: 被排除候选的原因列表。
        no_action_available: True 表示无真实候选。
    """
    machine: int
    actions: list
    action_mask: np.ndarray
    features: np.ndarray
    invalid_reasons: list = field(default_factory=list)
    no_action_available: bool = False

    def valid_actions(self):
        """返回所有有效动作 (mask=True 的 actions)。"""
        return [
            action
            for action, is_valid in zip(self.actions, self.action_mask)
            if bool(is_valid)
        ]


@dataclass
class MaskResult:
    """动作掩码结果。"""
    mask: np.ndarray   # bool 数组
    reasons: list      # 每个被遮蔽动作的原因


@dataclass
class SASObservation:
    """SAS (Single-Agent Scheduling) 观察 — 单步调度的完整状态快照。

    包含候选池、全局状态摘要和日历摘要，用于 RL 策略网络的输入。
    """
    machine: int
    current_time: float
    candidate_pool: CandidatePool
    candidate_actions: list
    candidate_features: np.ndarray       # (pool_size, 18)
    candidate_mask: np.ndarray           # (pool_size,) bool
    action_index_to_real_action: dict    # 有效动作的索引映射
    global_state_summary: dict
    calendar_summary: dict
    feature_names: tuple = field(default_factory=tuple)


@dataclass
class _Candidate:
    """内部候选 — 将动作、特征和打分捆绑在一起。"""
    action: DispatchAction
    features: np.ndarray
    score: float


# =============================================================================
# 奖励配置与计算
# =============================================================================


@dataclass
class RewardConfig:
    """SAS 奖励函数配置。

    奖励 = 执行奖励 + 等待惩罚 + 塑形奖励 + 终态奖励，最后 clip 到 [reward_clip_min, reward_clip_max]。

    执行奖励:
      - insert_success_reward (0.20): 成功插入的奖励
      - insert_fail_penalty (-0.40): 插入失败 (dry-run 不通过) 的惩罚
      - mask_invalid_penalty (-0.50): 选择被遮蔽/填充动作的惩罚
      - wait_penalty (0.0): SAS 的 wait 惩罚（默认 0，SAS 不拥有 wait 决策；主动等待成本归 DDT）

    塑形奖励 (use_light_shaping=True 时生效):
      - tardy_weight * 归一化拖期
      - qtime_weight * 归一化 Q-time 违反
      - priority_weight * 优先级违反
      - progress_weight * 进度 (加工时间/时间窗)

    终态奖励 (use_terminal_reward=True 时生效):
      - 拖期 Lot 数、总拖期、Q-time 违反、利用率、优先级违反的加权和
    """
    # 执行奖励权重
    insert_success_reward: float = 0.20
    insert_fail_penalty: float = -0.40
    mask_invalid_penalty: float = -0.50
    wait_penalty: float = 0.0
    # 塑形奖励权重
    tardy_weight: float = -0.05
    qtime_weight: float = -0.08
    priority_weight: float = -0.03
    progress_weight: float = 0.01
    # 终态奖励权重
    terminal_tardy_lot_weight: float = -0.20
    terminal_total_tardiness_weight: float = -0.10
    terminal_qtime_weight: float = -0.15
    terminal_utilization_weight: float = 0.05
    terminal_priority_weight: float = -0.05
    # 裁剪范围
    reward_clip_min: float = -1.0
    reward_clip_max: float = 1.0
    # 开关
    use_light_shaping: bool = False
    use_terminal_reward: bool = False


@dataclass
class SASStepResult:
    """SAS 单步结果 — sas_step() 的返回值。"""
    action: DispatchAction
    reward: float
    info: dict
    committed: bool       # True=动作已生效 (状态已转移)
    done: bool            # True=episode 结束
    step_info: dict = field(default_factory=dict)
    failure_reason: str = ""


def _empty_reward_components():
    """创建零初始化的奖励分量字典。"""
    return {
        "reward_execute": 0.0,
        "reward_wait": 0.0,
        "reward_tardy": 0.0,
        "reward_qtime": 0.0,
        "reward_priority": 0.0,
        "reward_progress": 0.0,
        "reward_shape": 0.0,
        "reward_terminal": 0.0,
        "reward_total": 0.0,
    }


def compute_sas_reward_components(info, config=None):
    """计算 SAS 奖励的各个分量。

    奖励结构 (五部分):
      1. reward_execute: 插入成功/失败 或 掩码无效 的即时奖励
      2. reward_wait: 等待动作的惩罚
      3. reward_shape: 塑形奖励 (tardy + qtime + priority + progress)
      4. reward_terminal: 终态奖励 (仅 episode 结束时)
      5. reward_total: clip(sum(以上)) 并四舍五入到 12 位小数

    Args:
        info: 步骤信息字典，由 sas_step() 填充。
        config: RewardConfig 实例。

    Returns:
        奖励分量字典。
    """
    if config is None:
        config = RewardConfig()
    components = _empty_reward_components()

    # ---- 第一部分: 执行奖励 ----
    if info.get("mask_invalid"):
        components["reward_execute"] = config.mask_invalid_penalty
    elif info.get("wait_or_noop"):
        # SAS 的 wait 只来自"空池/全屏蔽"，不罚（主动等待成本归 DDT）
        components["reward_wait"] = config.wait_penalty  # 默认 0.0
    elif info.get("insertion_failed"):
        components["reward_execute"] = config.insert_fail_penalty
    elif info.get("insertion_success"):
        components["reward_execute"] = config.insert_success_reward

        # ---- 第二部分: 轻量塑形奖励 (可选) ----
        if config.use_light_shaping:
            # 归一化因子: 加工完成时刻与当前时刻的跨度
            horizon = max(
                info.get("selected_lot_end", 0.0) - info.get("current_time", 0.0),
                1e-9,
            )
            due_date = info.get("due_date", np.inf)

            # 拖期塑形: clip( max(0, end - due) / horizon, 0, 1 )
            tardy_norm = float(np.clip(
                max(0.0, info.get("selected_lot_end", 0.0) - due_date) / horizon,
                0.0, 1.0,
            ))

            # Q-time 塑形: clip( new_violation / process_time, 0, 1 )
            process_time = max(info.get("selected_lot_process_time", 1.0), 1e-9)
            qtime_norm = float(np.clip(
                info.get("new_qtime_violation", 0.0) / process_time,
                0.0, 1.0,
            ))

            # 进度塑形: clip( process_time / horizon, 0, 1 )
            progress_norm = float(np.clip(
                (info.get("selected_lot_end", 0.0) - info.get("selected_lot_start", 0.0)) / horizon,
                0.0, 1.0,
            ))

            components["reward_tardy"] = config.tardy_weight * tardy_norm
            components["reward_qtime"] = config.qtime_weight * qtime_norm
            components["reward_priority"] = (
                config.priority_weight * info.get("priority_rank_penalty", 0.0)
            )
            components["reward_progress"] = config.progress_weight * progress_norm

    # ---- 第三部分: 终态奖励 (可选) ----
    if config.use_terminal_reward and info.get("episode_done"):
        components["reward_terminal"] = (
            config.terminal_tardy_lot_weight
            * info.get("tardy_lot_count_norm", 0.0)
            + config.terminal_total_tardiness_weight
            * info.get("total_tardiness_norm", 0.0)
            + config.terminal_qtime_weight
            * info.get("qtime_violation_count_norm", 0.0)
            + config.terminal_utilization_weight
            * info.get("machine_utilization_norm", 0.0)
            + config.terminal_priority_weight
            * info.get("priority_violation_norm", 0.0)
        )

    # ---- 汇总与裁剪 ----
    components["reward_shape"] = (
        components["reward_tardy"]
        + components["reward_qtime"]
        + components["reward_priority"]
        + components["reward_progress"]
    )
    raw_total = (
        components["reward_execute"]
        + components["reward_wait"]
        + components["reward_shape"]
        + components["reward_terminal"]
    )
    components["reward_total"] = float(round(
        float(np.clip(raw_total, config.reward_clip_min, config.reward_clip_max)),
        12,
    ))
    for key, value in list(components.items()):
        components[key] = float(round(float(value), 12))
    return components


def compute_sas_reward(info, config=None):
    """计算 SAS 总奖励并回写到 info 字典。

    Returns:
        reward_total — 裁剪后的总奖励。
    """
    components = compute_sas_reward_components(info, config)
    info.update(components)
    return components["reward_total"]


# =============================================================================
# 向量奖励 (Phase 2, 报告 §4.5 R1) — 四通道, 不修改 compute_sas_reward / RewardConfig
# =============================================================================


@dataclass
class RewardVectorConfig:
    """向量奖励配置（报告 §4.5，三通道）。

    三通道:
      - exec  (即时密集): 成功插入基础奖励 + w_pack·packing_efficiency
                          (packing = total_work/span，即"这次派工把机台时间用得多紧"，
                          撞腔体争用→跨度变长→packing 变小，提供逐步可区分的利用率向信号)
      - qtime (即时密集): -new_qtime_violation/num_lots，每步对"这次派工新造成的 Q-time 违反"
                          直接惩罚（Σ_t = 终局总违反，telescoping，无双重计数，credit 更好）
      - util  (终局):     avg_machine_utilization，唯一软目标
    progress 通道已删除（恒为 1.0 的死重）。终局权重 w_* 作用在归一化 advantage 上（由 trainer 使用）。
    """
    insert_success_reward: float = 0.20
    insert_fail_penalty: float = -0.40
    mask_invalid_penalty: float = -0.50
    w_pack: float = 0.10     # exec 通道的 packing(利用率向)信号强度
    w_exec: float = 1.0
    w_qtime: float = 3.0     # 大值，体现硬约束优先
    w_util: float = 0.5
    channels: tuple = ("exec", "qtime", "util")


def compute_sas_reward_vector(info, config=None):
    """计算 SAS 向量奖励（报告 §4.5 R1）。

    Returns:
        dict: {"reward_vector": np.array([3]), "r_exec", "r_qtime", "r_util"}
        通道顺序: (exec, qtime, util)
    """
    if config is None:
        config = RewardVectorConfig()

    # --- exec 通道（即时密集）：成功插入基础 + packing(利用率向边际质量) ---
    r_exec = 0.0
    if info.get("mask_invalid"):
        r_exec = config.mask_invalid_penalty
    elif info.get("wait_or_noop"):
        r_exec = 0.0  # SAS 不拥有 wait，不罚
    elif info.get("insertion_failed"):
        r_exec = config.insert_fail_penalty
    elif info.get("insertion_success"):
        span = float(info.get("selected_lot_process_time", 0.0))
        total_work = float(info.get("selected_lot_total_work", 0.0))
        # packing = total_work/span ∈ 大致 1~N：撞腔体争用→span 拉长→packing 变小。
        packing = (total_work / span) if span > 1e-9 else 0.0
        r_exec = config.insert_success_reward + config.w_pack * packing

    # --- qtime 通道（即时密集）：逐步惩罚本次派工新造成的 Q-time 违反 ---
    # new_qtime_violation 仅成功提交时非零；Σ_t = 终局总违反（telescoping，无双重计数）。
    num_lots = max(float(info.get("num_lots", 1)), 1.0)
    r_qtime = -float(info.get("new_qtime_violation", 0.0)) / num_lots

    # --- util 通道（终局）：唯一软目标 ---
    r_util = 0.0
    if info.get("is_terminal"):
        r_util = float(info.get("avg_machine_utilization", 0.0))

    reward_vector = np.asarray([r_exec, r_qtime, r_util], dtype=float)
    return {
        "reward_vector": reward_vector,
        "r_exec": float(r_exec),
        "r_qtime": float(r_qtime),
        "r_util": float(r_util),
    }


# =============================================================================
# 调度验证报告
# =============================================================================


@dataclass(frozen=True)
class ValidationReport:
    """调度完整性校验报告。

    Attributes:
        passed: 校验是否通过。
        completed_lots: 已完成 Lot 数量。
        lot_schedule_rows: lot_schedule 行数。
        wafer_schedule_rows: wafer_schedule 行数。
        machine_conflicts: 机台日历冲突数。
        chamber_conflicts: 腔体日历冲突数。
        missing_lots: 缺失的 Lot (partial=False 时检查)。
        errors: 错误描述元组。
        partial: 是否为部分校验模式。
        validated_lots: 已验证的 Lot 集合。
    """
    passed: bool
    completed_lots: int
    lot_schedule_rows: int
    wafer_schedule_rows: int
    machine_conflicts: int
    chamber_conflicts: int
    missing_lots: tuple = ()
    errors: tuple = ()
    partial: bool = False
    validated_lots: tuple = ()


# =============================================================================
# ResourceCalendarEnv — 核心 RL 环境
# =============================================================================


class ResourceCalendarEnv:
    """基于资源日历的 FAB 调度 RL 环境。

    核心概念:
      - Lot (批次): 一个待加工的晶圆批次，在某一机台上用某一 PPID 一次加工完成。
      - Machine (机台): 可调度 Lot 的加工资源，拥有多个 Chamber 及 Side。
      - PPID (工艺配方): 定义了该 Lot 在该机台上的工艺步骤序列。
      - Stage (工艺步骤): 每步在某个 (chamber, side) 资源上执行一段处理时间。
      - Calendar (日历): 每个机台/腔体的已占用时间区间列表。

    18 个候选特征 (feature_names):
      0  is_real              是否真实候选 (1.0)
      1  is_wait              是否等待动作 (0/1)
      2  score                启发式打分
      3  arrival_time         Lot 到达时间
      4  waiting_time         已等待时间 (current_time - arrival)
      5  machine_slot_start   机台槽位开始时间
      6  machine_load         机台当前负载 (busy_time)
      7  total_process_time   总加工时间估算
      8  predicted_completion 预测完成时间
      9  stage_count          工艺步骤数
      10 qtime_risk           Q-time 风险评估值
      11 wafer_count          晶圆数量
      12 priority             Lot 优先级
      13 due_slack            交货松弛 (due_date - predicted_completion)
      14 priority_rank_norm   优先级排名归一化 (1.0=最高)
      15 due_slack_rank_norm  交货松弛排名归一化 (1.0=最紧急)
      16 is_best_priority     是否优先级最高 (0/1)
      17 is_most_urgent_due   是否交期最紧急 (0/1)
    """

    feature_names = (
        "is_real",
        "is_wait",
        "score",
        "arrival_time",
        "waiting_time",
        "machine_slot_start",
        "machine_load",
        "total_process_time",
        "predicted_completion",
        "stage_count",
        "qtime_risk",
        "wafer_count",
        "priority",
        "due_slack",
        "priority_rank_norm",
        "due_slack_rank_norm",
        "is_best_priority",
        "is_most_urgent_due",
    )

    def __init__(
        self,
        encoder,
        current_time=0.0,
        top_k=8,
        initial_state=None,
        completed_lots=None,
        _skip_validate=False,
        process_noise_enabled=False,
        noise_seed=None,
        w_lookahead=0.0,
        priority_filter_mode="soft",
        priority_min_gap=0.0,
    ):
        """初始化环境。

        Args:
            encoder: Phase1CalendarProblem 实例 (含问题数据和日历操作)。
            current_time: 初始仿真时钟。
            top_k: 候选池大小 (每个机台的候选动作数)。
            initial_state: 初始 ScheduleState (None 则用默认空状态)。
            completed_lots: 初始已完成 Lot 集合。
            _skip_validate: 跳过问题定义校验 (仅测试用)。
            process_noise_enabled: 是否在 commit 时对加工时长注入噪声 (报告 §2.4.6)。
                默认 False → 行为与确定性版本完全一致。规划 (dry-run/候选特征) 始终用 μ，
                仅 commit 路径 (_simulate_action) 用 p_actual = μ + N(0, σ)。
            noise_seed: 噪声 RNG 种子，保证可复现。
            w_lookahead: 有限前瞻窗宽度 (报告 §2.1)。默认 0.0 → 无前瞻，
                行为与现有版本完全一致。>0 时 visible_lots/lookahead_summary
                会纳入 (t_now, t_now+w_lookahead] 内即将到达的 Lot。
        """
        if not getattr(encoder, "_problem_validated", False) and not _skip_validate:
            encoder.validate_problem_definition()

        self.encoder = encoder
        self.current_time = float(current_time)
        self.top_k = int(top_k)
        self.state = initial_state if initial_state is not None else ScheduleState()
        self.completed_lots = {int(lot) for lot in (completed_lots or set())}
        self.lot_schedule = np.empty((0, 5), dtype=float)
        self.wafer_schedule = np.empty((0, 9), dtype=float)

        if self.top_k <= 0:
            raise ValueError("top_k must be positive")

        self.process_noise_enabled = bool(process_noise_enabled)
        self.w_lookahead = float(w_lookahead)
        self.noise_seed = noise_seed
        self._noise_rng = np.random.default_rng(noise_seed)
        self.priority_filter_mode = str(priority_filter_mode)
        self.priority_min_gap = float(priority_min_gap)
        # Q-time mask 口径: "aggregate" (现状: 单一聚合 deadline 代理)、
        # "chain" (按实际 q_time_limits 阶段链单次 dry-run 自检)、或
        # "chain_joint" (K 次独立带噪 dry-run 估 P(链上任一窗违规), 联合机会约束)。
        self.qtime_mask_mode = "aggregate"
        self.qtime_chain_mc = 8          # chain_joint 的蒙特卡洛采样数
        self.qtime_chain_threshold = 0.0  # mask 条件: 违规样本占比 > 此阈值 (0=任一即屏蔽)

        # is_doomed 缓存：键 lot → bool，时间推进/重置时失效（避免 mask 内 O(n²) 重算）
        self._doomed_cache = {}

        # 下层估时器结果缓存（报告 §1.5 开销警示）：键 (lot,machine,ppid,n_mc) → base 结果。
        # base 是 start_offset=0 的完成时间分布，只取决于静态输入（estimate 不读 state），
        # start_offset 在 estimate 内逐次重施，故跨步/跨机台可安全复用，仅 reset 时清空。
        self._estimate_cache = {}

        self._sync_state_summary()

    # ---- 属性 ----

    @property
    def remaining_lots(self):
        """返回尚未完成的 Lot 集合。"""
        return self._all_lots() - set(self.completed_lots)

    @remaining_lots.setter
    def remaining_lots(self, lots):
        remaining = {int(lot) for lot in lots}
        self.completed_lots = self._all_lots() - remaining

    # ---- 有限前瞻窗 (报告 §2.1) ----

    def visible_lots(self):
        """返回当前可见的未完成 Lot 列表。

        包含: (a) 已到达 (arrival <= t_now) 且未完成的 Lot;
              (b) 前瞻窗内即将到达 (t_now < arrival <= t_now + w_lookahead)
                  且未完成的 Lot。
        当 w_lookahead = 0 时仅含已到达未完成的 Lot。
        """
        t_now = self.current_time
        horizon = t_now + self.w_lookahead
        visible = []
        for lot in sorted(self.remaining_lots):
            arrival = float(self.encoder.arrival_times.get(int(lot), t_now))
            if arrival <= t_now or arrival <= horizon:
                visible.append(int(lot))
        return visible

    def upcoming_lots(self):
        """返回前瞻窗内即将到达但尚未到达的未完成 Lot 列表。

        条件: t_now < arrival <= t_now + w_lookahead 且未完成。
        """
        t_now = self.current_time
        horizon = t_now + self.w_lookahead
        upcoming = []
        for lot in sorted(self.remaining_lots):
            arrival = float(self.encoder.arrival_times.get(int(lot), t_now))
            if t_now < arrival <= horizon:
                upcoming.append(int(lot))
        return upcoming

    def lookahead_summary(self):
        """返回前瞻窗内即将到达 Lot 的摘要字典 (报告 §2.1)。

        字段:
          - upcoming_count: 窗内即将到达 Lot 数量。
          - max_priority: 这些 Lot 的最高 priority (无则 0.0)。
          - min_remaining_qtime: 最小剩余 qtime = min(qtime_deadline[l] - t_now)
            (无可用 deadline 或无 upcoming 则 0.0)。
          - earliest_eta: 最早 arrival (无则 0.0)。
        """
        t_now = self.current_time
        upcoming = self.upcoming_lots()
        if not upcoming:
            return {
                "upcoming_count": 0,
                "max_priority": 0.0,
                "min_remaining_qtime": 0.0,
                "earliest_eta": 0.0,
            }

        priorities = getattr(self.encoder, "priorities", {})
        qtime_deadline = getattr(self.encoder, "qtime_deadline", {})
        arrival_times = self.encoder.arrival_times

        max_priority = max(
            float(priorities.get(int(lot), 0.0)) for lot in upcoming
        )
        earliest_eta = min(
            float(arrival_times.get(int(lot), t_now)) for lot in upcoming
        )
        remaining_qtimes = []
        for lot in upcoming:
            deadline = float(qtime_deadline.get(int(lot), float("inf")))
            if deadline != float("inf"):
                remaining_qtimes.append(deadline - t_now)
        min_remaining_qtime = min(remaining_qtimes) if remaining_qtimes else 0.0

        return {
            "upcoming_count": len(upcoming),
            "max_priority": float(max_priority),
            "min_remaining_qtime": float(min_remaining_qtime),
            "earliest_eta": float(earliest_eta),
        }

    # ---- 时间与重置 ----

    def advance_time(self, next_time):
        """推进仿真时钟到指定时间。只能向前推进。"""
        next_time = float(next_time)
        if next_time < self.current_time:
            raise ValueError("next_time must be >= current_time")
        self.current_time = next_time
        self._doomed_cache = {}  # 时间改变 → doom 状态失效
        self._sync_state_summary()

    def reset(self, current_time=0.0, initial_state=None, completed_lots=None):
        """重置环境到初始状态。返回 step_info 字典。"""
        self.current_time = float(current_time)
        self.state = initial_state if initial_state is not None else ScheduleState()
        self.completed_lots = {int(lot) for lot in (completed_lots or set())}
        self.lot_schedule = np.empty((0, 5), dtype=float)
        self.wafer_schedule = np.empty((0, 9), dtype=float)
        self._doomed_cache = {}
        self._estimate_cache = {}
        self._sync_state_summary()
        return self.step_info()

    # ---- 机台选择与事件时间 ----

    def get_candidate_machines(self):
        """返回当前有可调度候选的机台列表。

        仅包含至少有一个真实 (非 wait/padding) 有效候选的机台。
        """
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
        """返回下一个需要推进到的事件时间 (Lot 到达或资源释放)。

        返回 None 表示没有未来事件 (episode 终止条件之一)。
        """
        future_times = []
        # 尚未到达的 Lot 的到达时间
        for lot in self.remaining_lots:
            arrival = float(self.encoder.arrival_times.get(int(lot), self.current_time))
            if arrival > self.current_time:
                future_times.append(arrival)
        # 机台和腔体的释放时间
        for time_value in self.state.machine_available_time.values():
            if float(time_value) > self.current_time:
                future_times.append(float(time_value))
        for time_value in self.state.chamber_available_time.values():
            if float(time_value) > self.current_time:
                future_times.append(float(time_value))
        if not future_times:
            return None
        return min(future_times)

    # ---- SAS 观察构建 ----

    def build_sas_observation(self, machine):
        """构建单步 SAS 观察 (SASObservation)。

        包含候选池、全局状态摘要 (完成数/剩余数) 和日历摘要 (机台忙时/有效动作数)。
        """
        machine = int(machine)
        pool = self.build_candidate_pool(machine)

        # 仅保留有效动作的索引映射
        action_index_to_real_action = {
            index: action
            for index, action in enumerate(pool.actions)
            if bool(pool.action_mask[index])
        }

        # 计算机台忙时
        lot_schedule = np.asarray(self.lot_schedule, dtype=float).reshape((-1, 5))
        machine_busy_time = 0.0
        if lot_schedule.size > 0:
            machine_rows = lot_schedule[lot_schedule[:, 1].astype(int) == machine]
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
            machine=machine,
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

    # ---- 候选池构建 ----

    def build_candidate_pool(self, machine, top_k=None):
        """为指定机台构建候选动作池。

        流程:
          1. _real_candidates(machine) → 所有结构可行的 (lot, ppid) 组合
          2. 按 score 降序排列，取前 top_k 个
          3. 若还有空位，追加一个 wait 动作 (使用首个候选的 lot)
          4. 不足 top_k 的部分用 padding 填充 (mask=False)
          5. 计算排名特征 (priority_rank_norm, due_slack_rank_norm 等)

        Returns:
            CandidatePool — 包含动作列表、特征矩阵 (pool_size, 18) 和掩码。
        """
        machine = int(machine)
        pool_size = self.top_k if top_k is None else int(top_k)
        if pool_size <= 0:
            raise ValueError("top_k must be positive")

        real_candidates, excluded_reasons = self._real_candidates(machine)

        # ① qtime-safe mask（报告 Section 3.2，仅过滤真实候选）
        qtime_mask_enabled = getattr(self.encoder, "z_eps", None) is not None
        if qtime_mask_enabled and real_candidates:
            candidate_actions_for_mask = [c.action for c in real_candidates]
            qtime_masks = self.qtime_safe_mask(machine, candidate_actions_for_mask)
            real_candidates = [
                c for c, ok in zip(real_candidates, qtime_masks) if ok
            ]

        # ② priority filter（报告 §3.1/§3.4，仅 strict 模式实际删减）
        if self.priority_filter_mode == "strict" and real_candidates:
            actions_only = [c.action for c in real_candidates]
            kept = self.priority_filter(
                actions_only, mode="strict", priority_min_gap=self.priority_min_gap,
            )
            kept_ids = set(id(a) for a in kept)
            real_candidates = [c for c in real_candidates if id(c.action) in kept_ids]

        # 按 score 降序 → lot → machine → ppid 排序
        real_candidates.sort(
            key=lambda candidate: (
                -candidate.score,
                candidate.action.lot,
                candidate.action.machine,
                candidate.action.ppid,
            )
        )

        actions = []
        feature_rows = []
        mask = []
        invalid_reasons = list(excluded_reasons)

        # Top-K 真实候选
        for candidate in real_candidates[:pool_size]:
            actions.append(candidate.action)
            feature_rows.append(candidate.features)
            mask.append(True)

        # 若有真实候选且池未满，追加 wait 动作
        if len(real_candidates) > 0 and len(actions) < pool_size:
            wait_lot = real_candidates[0].action.lot
            actions.append(
                DispatchAction(lot=wait_lot, machine=machine, ppid=0, is_wait=True)
            )
            feature_rows.append(self._wait_features())
            mask.append(True)

        # 不足 pool_size 的部分用 padding 填充
        while len(actions) < pool_size:
            padding_index = len(actions)
            actions.append(
                DispatchAction(lot=0, machine=machine, ppid=0, is_padding=True)
            )
            feature_rows.append(np.zeros(len(self.feature_names), dtype=float))
            mask.append(False)
            invalid_reasons.append({
                "index": padding_index,
                "action": actions[-1],
                "reason": "padding",
            })

        # 计算排名特征
        feature_rows = self._apply_candidate_rank_features(actions, feature_rows, mask)

        return CandidatePool(
            machine=machine,
            actions=actions,
            action_mask=np.asarray(mask, dtype=bool),
            features=np.asarray(feature_rows, dtype=float),
            invalid_reasons=invalid_reasons,
            no_action_available=(len(real_candidates) == 0),
        )

    def _apply_candidate_rank_features(self, actions, feature_rows, mask):
        """为候选特征矩阵填充排名相关特征。

        仅对真实候选 (非 wait/padding 且 mask=True) 计算:
          - priority_rank_norm:  优先级排名归一化 (最高=1.0)
          - due_slack_rank_norm: 交货松弛排名归一化 (最紧急=1.0)
          - is_best_priority:    是否优先级最高
          - is_most_urgent_due:  是否交期最紧急
        """
        if not feature_rows:
            return feature_rows

        priority_idx = self.feature_names.index("priority")
        due_slack_idx = self.feature_names.index("due_slack")
        priority_rank_idx = self.feature_names.index("priority_rank_norm")
        due_slack_rank_idx = self.feature_names.index("due_slack_rank_norm")
        best_priority_idx = self.feature_names.index("is_best_priority")
        urgent_due_idx = self.feature_names.index("is_most_urgent_due")

        # 筛选真实候选的索引
        real_indices = []
        for index, action in enumerate(actions):
            action = self._coerce_action(action)
            if (
                bool(mask[index])
                and not action.is_wait
                and not action.is_padding
            ):
                real_indices.append(index)
        if not real_indices:
            return feature_rows

        n_real = len(real_indices)

        # 按优先级降序排名 (高优先级=大 rank_norm)
        priority_order = sorted(
            real_indices,
            key=lambda index: (-feature_rows[index][priority_idx], index),
        )
        # 按松弛升序排名 (小松弛=紧急=大 rank_norm)
        due_order = sorted(
            real_indices,
            key=lambda index: (feature_rows[index][due_slack_idx], index),
        )

        for rank, index in enumerate(priority_order, start=1):
            feature_rows[index][priority_rank_idx] = (n_real - rank + 1) / n_real
        for rank, index in enumerate(due_order, start=1):
            feature_rows[index][due_slack_rank_idx] = (n_real - rank + 1) / n_real

        # 标记最优候选
        max_priority = max(feature_rows[index][priority_idx] for index in real_indices)
        min_due_slack = min(feature_rows[index][due_slack_idx] for index in real_indices)
        for index in real_indices:
            feature_rows[index][best_priority_idx] = float(
                feature_rows[index][priority_idx] == max_priority
            )
            feature_rows[index][urgent_due_idx] = float(
                feature_rows[index][due_slack_idx] == min_due_slack
            )

        return feature_rows

    # ==========================================================================
    # Public interfaces (Plan Section 6) — 公共 API
    # ==========================================================================

    def generate_raw_candidates(self, machine):
        """返回结构可行的候选 (不含 Top-K 截断和 padding)。

        Plan Section 6.2: generate_raw_candidates(state, machine_id) -> list[DispatchAction]
        """
        real_candidates, _ = self._real_candidates(machine)
        return [candidate.action for candidate in real_candidates]

    def score_candidate(self, action):
        """计算单个候选的启发式打分。

        Plan Section 6.3: score_candidate(action, state) -> float

        得分 = priority + waiting_time - 0.01*release_time - 0.001*proc_time - 0.001*qtime_risk

        padding/wait/ppid=0 动作返回 0.0；dry-run 不可行返回 -inf。
        """
        action = self._coerce_action(action)
        if action.is_padding or action.is_wait or int(action.ppid) == 0:
            return 0.0
        dry_run, _reason = self._dry_run_candidate(
            int(action.lot), int(action.machine), int(action.ppid),
        )
        if dry_run is None:
            return -float("inf")
        _features, score = self._candidate_features(
            int(action.lot), int(action.machine), int(action.ppid), dry_run,
        )
        return float(score)

    def build_action_mask(self, machine, actions):
        """为一组动作构建掩码和原因列表。

        Plan Section 6.4: build_action_mask(candidate_pool, state) -> MaskResult

        检查顺序: padding → wait/ppid=0 → machine_mismatch → lot_completed →
                    lot_not_arrived → machine_incompatible → recipe_incompatible →
                    dry_run_infeasible → True (有效)
        """
        machine = int(machine)
        mask_values = []
        reasons = []
        for idx, action in enumerate(actions):
            action = self._coerce_action(action)
            # padding 始终无效
            if action.is_padding:
                mask_values.append(False)
                reasons.append({
                    "index": idx, "action": action, "reason": "padding",
                })
                continue
            # wait / ppid=0 始终有效
            if action.is_wait or int(action.ppid) == 0:
                mask_values.append(True)
                continue
            # 机台不匹配
            if int(action.machine) != machine:
                mask_values.append(False)
                reasons.append({
                    "index": idx, "action": action,
                    "reason": "machine_mismatch",
                })
                continue
            lot = int(action.lot)
            # Lot 已完成
            if lot in self.completed_lots:
                mask_values.append(False)
                reasons.append({
                    "index": idx, "action": action,
                    "reason": "lot_completed",
                })
                continue
            # Lot 尚未到达
            if float(self.encoder.arrival_times[lot]) > self.current_time:
                mask_values.append(False)
                reasons.append({
                    "index": idx, "action": action,
                    "reason": "lot_not_arrived",
                })
                continue
            # 机台不兼容 (不在 feasible_machines 中)
            if not self._lot_can_run_on_machine(lot, machine):
                mask_values.append(False)
                reasons.append({
                    "index": idx, "action": action,
                    "reason": "machine_incompatible",
                })
                continue
            # 配方不兼容
            if not self._lot_recipe_matches_machine(lot, machine):
                mask_values.append(False)
                reasons.append({
                    "index": idx, "action": action,
                    "reason": "recipe_incompatible",
                })
                continue
            # dry-run 可行性检查
            dry_run, reason = self._dry_run_candidate(
                lot, int(action.machine), int(action.ppid),
            )
            if dry_run is None:
                mask_values.append(False)
                reasons.append({
                    "index": idx, "action": action,
                    "reason": reason or "dry_run_infeasible",
                })
                continue
            mask_values.append(True)
        return MaskResult(
            mask=np.asarray(mask_values, dtype=bool),
            reasons=reasons,
        )

    def is_doomed(self, lot, z_eps=None):
        """判断 lot 是否已注定 qtime 违规（即便最乐观立即开工也来不及）。

        报告 §3.2：doomed lot 不作为 qtime mask 屏蔽依据，否则会"全屏蔽→死锁"。
        用均值口径（不加 z·σ）：deadline - (earliest_start + mu_finish) < 0 → doomed。
        """
        from lower_layer_estimator import estimate
        lot = int(lot)
        cache = getattr(self, "_doomed_cache", None)
        if cache is None:
            cache = self._doomed_cache = {}
        if lot in cache:
            return cache[lot]
        qtime_dl = float(self.encoder.get_qtime_deadline(lot))
        if not np.isfinite(qtime_dl):
            cache[lot] = False
            return False
        # 该 lot 最乐观开工时刻
        arrival = float(self.encoder.arrival_times.get(lot, self.current_time))
        earliest_start = max(self.current_time, arrival)
        # 用该 lot 第一个可行 (machine, ppid) 最乐观估时
        machines = self.encoder.get_machine_list(lot)
        if len(machines) == 0:
            cache[lot] = False
            return False
        best_mu = None
        for m in machines:
            ppids = self.encoder.get_ppid_list(lot, int(m))
            for p in ppids:
                try:
                    res = estimate(lot, int(m), int(p), self.encoder, self.state,
                                   n_mc=10, start_offset=earliest_start,
                                   cache=self._estimate_cache)
                except Exception:
                    continue
                mu = float(res["mu_finish"])
                if best_mu is None or mu < best_mu:
                    best_mu = mu
        if best_mu is None:
            cache[lot] = False
            return False
        # 均值口径：连最乐观完成时刻都超 deadline → doomed
        doomed = bool(qtime_dl - best_mu < 0.0)
        cache[lot] = doomed
        return doomed

    def _qtime_chain_mask(self, machine, candidate_actions):
        """Chain-aware Q-time mask（方向: Q-time 链）。

        现状 mask 只比"单一聚合 deadline 代理 vs 总完成 μ"，从不看实际的阶段间
        q_time_limits 链 (1,2)/(2,3)。本版改为对每个候选做非破坏式 dry-run，用
        真正的 compute_q_time_violation 评估其阶段链窗口——即"屏蔽所筛 = 实际所罚"。
        doomed lot 仍不作屏蔽依据（防死锁，报告 §3.2）。
        """
        mask = np.ones(len(candidate_actions), dtype=bool)
        if len(getattr(self.encoder, "q_time_limits", {})) == 0:
            return mask
        for i, action in enumerate(candidate_actions):
            action = self._coerce_action(action)
            if action.is_padding or action.is_wait or int(action.ppid) == 0:
                continue
            if self.is_doomed(int(action.lot)):
                continue
            try:
                res = self.dry_run_action(action)
            except Exception:
                continue
            if not res.success or np.asarray(res.wafer_schedule).size == 0:
                continue
            count, _total = self.encoder.compute_q_time_violation(res.wafer_schedule)
            if float(count) > 0.0:
                mask[i] = False
        return mask

    def _qtime_chain_joint_mask(self, machine, candidate_actions):
        """Chain-aware 联合机会约束 mask（方向: Q-time 链 chance-constraint 版）。

        对每个候选做 K 次【独立带噪】非破坏式 dry-run，用真实 compute_q_time_violation
        逐次判断"阶段链 (1,2)/(2,3) 上是否任一窗违规"（联合 = 任一窗），估计违规概率
        p̂ = 违规样本数 / 有效样本数；p̂ > threshold 则屏蔽。即把 chain-μ 的单次确定判
        升级为"违规概率 ≤ ε"的联合机会约束，攻 chain-μ 漏掉的噪声尾部。
        doomed lot 仍不作屏蔽依据（防死锁）。
        """
        mask = np.ones(len(candidate_actions), dtype=bool)
        if len(getattr(self.encoder, "q_time_limits", {})) == 0:
            return mask
        k_mc = int(getattr(self, "qtime_chain_mc", 8))
        threshold = float(getattr(self, "qtime_chain_threshold", 0.0))
        for i, action in enumerate(candidate_actions):
            action = self._coerce_action(action)
            if action.is_padding or action.is_wait or int(action.ppid) == 0:
                continue
            if self.is_doomed(int(action.lot)):
                continue
            rng = np.random.default_rng((int(action.lot), int(action.ppid), int(self.noise_seed or 0)))
            violations = 0
            valid = 0
            for _ in range(k_mc):
                try:
                    res = self.dry_run_action(action, noise_rng=rng)
                except Exception:
                    continue
                if not res.success or np.asarray(res.wafer_schedule).size == 0:
                    continue
                valid += 1
                count, _total = self.encoder.compute_q_time_violation(res.wafer_schedule)
                if float(count) > 0.0:
                    violations += 1
            if valid > 0 and (violations / valid) > threshold:
                mask[i] = False
        return mask

    def qtime_safe_mask(self, machine, candidate_actions, z_eps=None):
        """Q-time 机会约束 mask（报告 §3.2）。

        本版语义（self 严格 + doomed 排除）：
          对每个候选 (lot, ppid)，用下层估时器在【绝对时间基准】下估完成分布(μ,σ)，
          若被调度 lot 自身违规概率 > ε（deadline - μ_finish < z_ε·σ_finish）则屏蔽。
          但若该 lot 已 is_doomed（连最乐观都来不及），则不屏蔽——否则 doomed lot
          会让其候选全被屏蔽导致死锁（报告 §3.2 要点）。
        时间基准：estimate 用 start_offset=max(current_time, arrival)，使 μ_finish 为绝对完成时刻，
          与绝对 qtime_deadline 同基准比较（修复 #2）。

        范围说明：#3 的"提交候选 i 后对其它 visible lot 的挤占"完整模拟开销大，本版
          聚焦"自身严格 + 时间基准正确 + doomed 排除框架"。is_doomed 已遍历该 lot 的
          可行 (machine, ppid) 组合，visible_lots 检查框架已就位，留作后续完整版扩展点。
        """
        from lower_layer_estimator import estimate, is_qtime_violated_probabilistically

        mask_mode = getattr(self, "qtime_mask_mode", "aggregate")
        if mask_mode == "chain":
            return self._qtime_chain_mask(machine, candidate_actions)
        if mask_mode == "chain_joint":
            return self._qtime_chain_joint_mask(machine, candidate_actions)

        if z_eps is None:
            z_eps = float(getattr(self.encoder, "z_eps", 2.05))

        mask = np.ones(len(candidate_actions), dtype=bool)

        for i, action in enumerate(candidate_actions):
            action = self._coerce_action(action)
            if action.is_padding or action.is_wait or int(action.ppid) == 0:
                continue

            lot = int(action.lot)
            ppid = int(action.ppid)

            qtime_dl = float(self.encoder.get_qtime_deadline(lot))
            if not np.isfinite(qtime_dl):
                continue  # 无截止约束，不屏蔽

            # doomed lot 不作屏蔽依据（#4，防死锁）
            if self.is_doomed(lot, z_eps=z_eps):
                continue

            arrival = float(self.encoder.arrival_times.get(lot, self.current_time))
            earliest_start = max(self.current_time, arrival)

            try:
                result = estimate(
                    lot, int(machine), ppid,
                    self.encoder, self.state, n_mc=20,
                    start_offset=earliest_start,
                    cache=self._estimate_cache,
                )
            except Exception:
                continue  # 估时失败不屏蔽（保守处理）

            mu_finish = float(result["mu_finish"])
            sigma_finish = float(result["sigma_finish"])

            # 绝对基准比较（#2）：deadline - μ < z_eps * σ → 屏蔽
            if is_qtime_violated_probabilistically(mu_finish, sigma_finish, qtime_dl, z_eps):
                mask[i] = False

        return mask

    def priority_filter(self, actions, mode="soft", priority_min_gap=0.0):
        """优先级过滤（报告 Section 3.4）。

        soft 模式：保留所有动作，不删除（探索友好）。
        strict 模式：只保留优先级 >= max_priority - priority_min_gap 的动作。
        """
        if not actions or mode == "soft":
            return actions  # soft: 不删动作，评分中已体现偏好

        real_actions = [
            a for a in actions
            if not self._coerce_action(a).is_padding
            and not self._coerce_action(a).is_wait
        ]
        if not real_actions:
            return actions

        max_pri = max(
            float(self.encoder.priorities.get(int(self._coerce_action(a).lot), 0.0))
            for a in real_actions
        )
        threshold = max_pri - float(priority_min_gap)

        filtered = []
        for a in actions:
            coerced = self._coerce_action(a)
            if coerced.is_padding or coerced.is_wait:
                filtered.append(a)
                continue
            lot_pri = float(self.encoder.priorities.get(int(coerced.lot), 0.0))
            if lot_pri >= threshold:
                filtered.append(a)
        return filtered if filtered else actions  # 全被过滤则回退原列表

    # ---- Action 试算与提交 ----

    def dry_run_action(self, action, noise_rng=None):
        """在状态副本上模拟执行一个动作 (不修改真实环境)。

        返回 DryRunResult，包含试算的 lot_schedule, wafer_schedule 和状态副本。
        padding 动作 success=False；wait 动作 success=True 但 schedule 为空。
        noise_rng: None=均值路径，不采样噪声；传 Generator 则独立采样。
        """
        action = self._coerce_action(action)
        dry_state = self._copy_state(self.state)

        if action.is_padding:
            return DryRunResult(
                action=action,
                success=False,
                lot_schedule=np.empty((0, 5), dtype=float),
                wafer_schedule=np.empty((0, 9), dtype=float),
                state=dry_state,
                failure_reason="padding",
            )

        if action.is_wait or int(action.ppid) == 0:
            return DryRunResult(
                action=action,
                success=True,
                lot_schedule=np.empty((0, 5), dtype=float),
                wafer_schedule=np.empty((0, 9), dtype=float),
                state=dry_state,
                failure_reason="",
            )

        try:
            lot_schedule, wafer_schedule, state = self._simulate_action(
                action,
                dry_state,
                noise_rng=noise_rng,
            )
            machine_intervals = [
                (int(row[1]), float(row[3]), float(row[4]))
                for row in lot_schedule.reshape((-1, 5))
            ]
            chamber_intervals = [
                ((int(row[2]), int(row[5]), int(row[6])), float(row[7]), float(row[8]))
                for row in wafer_schedule.reshape((-1, 9))
            ]
            return DryRunResult(
                action=action,
                success=True,
                lot_schedule=lot_schedule,
                wafer_schedule=wafer_schedule,
                state=state,
                machine_intervals=machine_intervals,
                chamber_intervals=chamber_intervals,
                failure_reason="",
            )
        except Exception as exc:
            return DryRunResult(
                action=action,
                success=False,
                lot_schedule=np.empty((0, 5), dtype=float),
                wafer_schedule=np.empty((0, 9), dtype=float),
                state=dry_state,
                failure_reason=str(exc),
            )

    def commit_action_index(self, machine, action_index, pool=None,
                           dry_run_result=None):
        """将候选池中指定索引的动作提交到环境 (修改真实状态)。

        Args:
            machine: 机台编号。
            action_index: 候选池中的动作索引。
            pool: 候选池 (None 则重新构建)。
            dry_run_result: 可复用的 dry-run 结果 (避免重复计算)。

        Returns:
            DispatchCommitResult。

        对于 wait 动作: committed=False，不修改状态但返回 step_info。
        对于真实动作: 将 lot/wafer 区间写入日历，标记 Lot 为已完成。
        """
        machine = int(machine)
        pool = self.build_candidate_pool(machine) if pool is None else pool

        if int(pool.machine) != machine:
            raise ValueError(
                f"candidate pool machine {pool.machine} does not match {machine}"
            )

        action_index = int(action_index)
        if action_index < 0 or action_index >= len(pool.actions):
            raise IndexError(f"action_index {action_index} is out of bounds")

        if not bool(pool.action_mask[action_index]):
            raise ValueError(f"cannot commit masked action at index {action_index}")

        action = self._coerce_action(pool.actions[action_index])
        if int(action.machine) != machine:
            raise ValueError(
                f"action machine {action.machine} does not match requested {machine}"
            )
        if action.is_padding:
            raise ValueError(f"cannot commit masked action at index {action_index}")

        # wait 动作: 不修改状态，直接返回
        if action.is_wait or int(action.ppid) == 0:
            info = self.step_info()
            return DispatchCommitResult(
                action=action,
                lot_schedule=np.empty((0, 5), dtype=float),
                wafer_schedule=np.empty((0, 9), dtype=float),
                state=self.state,
                committed=False,
                step_info=info,
            )

        reuse_dry_run = (
            dry_run_result is not None
            and dry_run_result.success
            and dry_run_result.action == action
            and not self.process_noise_enabled
        )

        # 记录提交前状态用于回滚
        before_counts = {
            "lot_schedule_rows": int(
                np.asarray(self.lot_schedule, dtype=float).reshape((-1, 5)).shape[0]
            ),
            "wafer_schedule_rows": int(
                np.asarray(self.wafer_schedule, dtype=float).reshape((-1, 9)).shape[0]
            ),
            "machine_available_time": dict(self.state.machine_available_time),
            "chamber_available_time": dict(self.state.chamber_available_time),
        }
        snapshot = self._snapshot_environment()
        try:
            if reuse_dry_run:
                # 复用 dry_run 结果: 直接应用日历区间
                for resource_key, start_time, end_time in \
                        dry_run_result.chamber_intervals:
                    self.encoder.add_calendar_interval(
                        self.state.chamber_calendar,
                        resource_key,
                        start_time,
                        end_time,
                    )
                for m_id, start_time, end_time in \
                        dry_run_result.machine_intervals:
                    self.encoder.add_calendar_interval(
                        self.state.machine_calendar,
                        m_id,
                        start_time,
                        end_time,
                    )
                self.state.machine_available_time[machine] = max(
                    self.state.machine_available_time.get(
                        machine, self.current_time,
                    ),
                    float(dry_run_result.lot_schedule.reshape((-1, 5))[0, 4]),
                )
                for resource_key, _start_time, end_time in \
                        dry_run_result.chamber_intervals:
                    self.state.chamber_available_time[resource_key] = max(
                        self.state.chamber_available_time.get(
                            resource_key, self.current_time,
                        ),
                        float(end_time),
                    )
                lot_schedule = dry_run_result.lot_schedule
                wafer_schedule = dry_run_result.wafer_schedule
            else:
                # 重新执行完整仿真
                lot_schedule, wafer_schedule, state = self._simulate_action(
                    action,
                    self.state,
                )
                self.state = state

            # 追加到全局调度表
            self.lot_schedule = self._append_schedule_rows(
                self.lot_schedule,
                lot_schedule,
                5,
            )
            self.wafer_schedule = self._append_schedule_rows(
                self.wafer_schedule,
                wafer_schedule,
                9,
            )
            self.completed_lots.add(int(action.lot))
        except Exception:
            # 发生异常时回滚到快照状态
            self._restore_environment(snapshot)
            raise

        commit_log = self._build_commit_log(
            action,
            lot_schedule,
            wafer_schedule,
            before_counts,
        )
        self.state.commit_log.append(commit_log)
        self._sync_state_summary()
        info = self.step_info()

        return DispatchCommitResult(
            action=action,
            lot_schedule=lot_schedule,
            wafer_schedule=wafer_schedule,
            state=self.state,
            committed=True,
            commit_log=commit_log,
            step_info=info,
        )

    # ---- 调度验证 ----

    def validate_schedule(self, partial=False):
        """验证当前调度的完整性和无冲突性。

        检查内容:
          1. 机台/腔体日历区间交叠 (冲突计数)
          2. 所有 Lot 是否均被调度 (partial=False)
          3. 晶圆级覆盖: 每 Lot 的所有 wafer_id 和 stage_id 均存在
          4. Stage 顺序: 晶圆级 stage 按 PPID 定义顺序排列
          5. Lot 级覆盖: lot_start ≤ min(wafer_start), lot_end ≥ max(wafer_end)

        Args:
            partial: True 时不检查"所有 Lot 全覆盖"。

        Returns:
            ValidationReport。
        """
        errors = []
        machine_calendar = {}
        chamber_calendar = {}

        lot_schedule = np.asarray(self.lot_schedule, dtype=float).reshape((-1, 5))
        wafer_schedule = np.asarray(self.wafer_schedule, dtype=float).reshape((-1, 9))

        # 重建日历用于冲突检测
        for row in lot_schedule:
            machine_calendar.setdefault(int(row[1]), []).append(
                (float(row[3]), float(row[4]))
            )
        # 批处理 (报告 §1.5): 同一子批的多片 wafer 共享一个 (chamber,side) 区间，
        # 重建日历时按 (resource, start, end) 去重，避免把"同批共享"误计为冲突。
        chamber_seen = {}
        for row in wafer_schedule:
            resource_key = (int(row[2]), int(row[5]), int(row[6]))
            interval = (float(row[7]), float(row[8]))
            seen = chamber_seen.setdefault(resource_key, set())
            if interval in seen:
                continue
            seen.add(interval)
            chamber_calendar.setdefault(resource_key, []).append(interval)

        machine_conflicts = self._count_calendar_conflicts(machine_calendar)
        chamber_conflicts = self._count_calendar_conflicts(chamber_calendar)

        # Lot 全覆盖检查
        expected_lots = self._all_lots()
        scheduled_lots = {int(row[0]) for row in lot_schedule}
        missing_lots = tuple(sorted(expected_lots - scheduled_lots))

        if machine_conflicts:
            errors.append(f"machine calendar has {machine_conflicts} conflict(s)")
        if chamber_conflicts:
            errors.append(f"chamber calendar has {chamber_conflicts} conflict(s)")

        if not partial:
            if missing_lots:
                errors.append(f"missing lots: {missing_lots}")
            try:
                self.encoder.validate_final_schedule_completeness(
                    lot_schedule,
                    wafer_schedule,
                )
            except Exception as exc:
                errors.append(str(exc))
        else:
            # partial 模式下的 Wafer/Stage 覆盖检查
            wafer_rows_by_lot = {}
            for row in wafer_schedule:
                lot = int(row[0])
                wafer_rows_by_lot.setdefault(lot, []).append(row)
            for row in lot_schedule:
                lot = int(row[0])
                machine = int(row[1])
                ppid = int(row[2])
                wafer_rows = wafer_rows_by_lot.get(lot, [])
                expected_wafer_ids = set(range(1, int(self.encoder.wafer_counts[lot]) + 1))
                actual_wafer_ids = {int(wafer_row[1]) for wafer_row in wafer_rows}
                expected_stages = set(range(1, len(self.encoder.ppid_steps[(lot, machine, ppid)]) + 1))
                if actual_wafer_ids != expected_wafer_ids:
                    errors.append(f"lot {lot} wafer ids are incomplete")
                for wafer_id in expected_wafer_ids:
                    actual_stages = {
                        int(wafer_row[4])
                        for wafer_row in wafer_rows
                        if int(wafer_row[1]) == wafer_id
                    }
                    if actual_stages != expected_stages:
                        errors.append(f"lot {lot} wafer {wafer_id} stages are incomplete")
                        break

        # ---- Stage 顺序和 Lot 级覆盖检查 ----
        for row in lot_schedule:
            lot = int(row[0])
            machine = int(row[1])
            ppid = int(row[2])
            lot_start = float(row[3])
            lot_end = float(row[4])

            wafer_rows = [
                wr for wr in wafer_schedule if int(wr[0]) == lot
            ]
            if not wafer_rows:
                continue

            # 晶圆级 stage 顺序必须与 PPID 定义一致
            wafer_stage_map = {}
            for wr in wafer_rows:
                wafer_id = int(wr[1])
                wafer_stage_map.setdefault(wafer_id, []).append(wr)

            steps = self.encoder.get_process_steps(lot, machine, ppid)
            expected_stage_ids = list(range(1, len(steps) + 1))
            for wafer_id, stage_rows in wafer_stage_map.items():
                stage_rows_sorted = sorted(stage_rows, key=lambda r: int(r[4]))
                for i in range(len(stage_rows_sorted) - 1):
                    curr_end = float(stage_rows_sorted[i][8])
                    next_start = float(stage_rows_sorted[i + 1][7])
                    if curr_end > next_start + 1e-9:
                        errors.append(
                            f"lot {lot} wafer {wafer_id} stage "
                            f"{int(stage_rows_sorted[i][4])}→"
                            f"{int(stage_rows_sorted[i + 1][4])} "
                            f"out of order "
                            f"(end={curr_end:.3f} > start={next_start:.3f})"
                        )
                actual_stage_ids = [int(r[4]) for r in stage_rows_sorted]
                if actual_stage_ids != expected_stage_ids:
                    errors.append(
                        f"lot {lot} wafer {wafer_id} stage sequence "
                        f"{actual_stage_ids} != expected {expected_stage_ids}"
                    )

            # Lot 级时间区间必须覆盖晶圆级最小/最大时间
            wafer_start = min(float(wr[7]) for wr in wafer_rows)
            wafer_end = max(float(wr[8]) for wr in wafer_rows)
            if lot_start > wafer_start + 1e-9:
                errors.append(
                    f"lot {lot} lot_start={lot_start:.3f} > "
                    f"min wafer start={wafer_start:.3f}"
                )
            if lot_end < wafer_end - 1e-9:
                errors.append(
                    f"lot {lot} lot_end={lot_end:.3f} < "
                    f"max wafer end={wafer_end:.3f}"
                )

        return ValidationReport(
            passed=len(errors) == 0,
            completed_lots=len(scheduled_lots),
            lot_schedule_rows=int(lot_schedule.shape[0]),
            wafer_schedule_rows=int(wafer_schedule.shape[0]),
            machine_conflicts=machine_conflicts,
            chamber_conflicts=chamber_conflicts,
            missing_lots=() if partial else missing_lots,
            errors=tuple(errors),
            partial=partial,
            validated_lots=tuple(sorted(scheduled_lots)),
        )

    # ==========================================================================
    # 内部方法 — 候选生成、打分、仿真
    # ==========================================================================

    def _real_candidates(self, machine):
        """为指定机台生成所有结构可行的候选。

        对每个未完成的 Lot:
          1. 检查基本可行性 (lot_completed, lot_not_arrived, machine/recipe_compatible)
          2. 遍历所有 PPID, 执行 dry_run
          3. 通过 dry_run 的候选计算特征和打分

        Returns:
            (candidates: list[_Candidate], invalid_reasons: list[dict])
        """
        candidates = []
        invalid_reasons = []

        for lot in range(1, int(self.encoder.num_lots) + 1):
            if lot in self.completed_lots:
                invalid_reasons.append({
                    "index": None,
                    "action": DispatchAction(
                        lot=lot, machine=machine, ppid=0, is_padding=True
                    ),
                    "reason": "lot_completed",
                })
                continue
            if float(self.encoder.arrival_times[lot]) > self.current_time:
                invalid_reasons.append({
                    "index": None,
                    "action": DispatchAction(
                        lot=lot, machine=machine, ppid=0, is_padding=True
                    ),
                    "reason": "lot_not_arrived",
                })
                continue
            if not self._lot_can_run_on_machine(lot, machine):
                invalid_reasons.append({
                    "index": None,
                    "action": DispatchAction(
                        lot=lot, machine=machine, ppid=0, is_padding=True
                    ),
                    "reason": "machine_incompatible",
                })
                continue
            if not self._lot_recipe_matches_machine(lot, machine):
                invalid_reasons.append({
                    "index": None,
                    "action": DispatchAction(
                        lot=lot, machine=machine, ppid=0, is_padding=True
                    ),
                    "reason": "recipe_incompatible",
                })
                continue

            ppid_list = self.encoder.get_ppid_list(lot, machine)
            if len(ppid_list) == 0:
                invalid_reasons.append({
                    "index": None,
                    "action": DispatchAction(
                        lot=lot, machine=machine, ppid=0, is_padding=True
                    ),
                    "reason": "ppid_unavailable",
                })
                continue

            for ppid in ppid_list:
                ppid = int(ppid)
                dry_run, dry_run_reason = self._dry_run_candidate(
                    lot, machine, ppid,
                )
                if dry_run is None:
                    invalid_reasons.append({
                        "index": None,
                        "action": DispatchAction(
                            lot=lot, machine=machine, ppid=ppid
                        ),
                        "reason": dry_run_reason or "dry_run_infeasible",
                    })
                    continue

                features, score = self._candidate_features(
                    lot,
                    machine,
                    ppid,
                    dry_run,
                )
                candidates.append(
                    _Candidate(
                        action=DispatchAction(lot=lot, machine=machine, ppid=ppid),
                        features=features,
                        score=score,
                    )
                )

        return candidates, invalid_reasons

    def _lot_can_run_on_machine(self, lot, machine):
        """检查 Lot 是否可在指定机台上加工 (基于 feasible_machines)。"""
        return int(machine) in {
            int(candidate_machine)
            for candidate_machine in self.encoder.get_machine_list(lot)
        }

    def _lot_recipe_matches_machine(self, lot, machine):
        """检查 Lot 的配方是否与机台兼容 (基于 machine_recipes)。

        如 machine_recipes 为空则默认兼容。
        """
        machine_recipes = getattr(self.encoder, "machine_recipes", {})
        if not machine_recipes:
            return True
        allowed = machine_recipes.get(int(machine))
        if allowed is None:
            return True
        lot_recipe = self.encoder.recipe.get(int(lot))
        return lot_recipe in allowed

    def _dry_run_candidate(self, lot, machine, ppid):
        """Build candidate features through the shared lower-layer scheduler."""
        from lower_layer_scheduler import schedule_on_calendar

        lot = int(lot)
        machine = int(machine)
        ppid = int(ppid)
        try:
            steps = self.encoder.get_process_steps(lot, machine, ppid)
        except (KeyError, ValueError):
            return None, "ppid_stage_missing"
        if not steps or len(steps) == 0:
            return None, "ppid_stage_missing"

        earliest_release = max(
            self.current_time,
            float(self.encoder.arrival_times[lot]),
        )
        res = schedule_on_calendar(
            lot,
            machine,
            ppid,
            self.encoder,
            self.state,
            earliest_release=earliest_release,
            noise_rng=None,
        )
        if res.infeasible_reason:
            return None, res.infeasible_reason

        result = {
            "steps": steps,
            "lot_release_time": float(res.machine_interval[1]),
            "lot_start_time": float(res.lot_start),
            "lot_end_time": float(res.lot_end),
            "total_process_time": self.encoder.estimate_plan_total_process_time(
                steps,
                self.encoder.wafer_counts[lot],
            ),
            "qtime_risk": self.encoder.estimate_qtime_risk(
                lot,
                machine,
                ppid,
                steps,
            ),
        }
        return result, ""

    def _candidate_features(self, lot, machine, ppid, dry_run):
        """从 dry_run 结果计算 18 维候选特征向量。

        特征取值:
          [0]=1.0 (is_real), [1]=0.0 (is_wait), [2]=score, [3]=arrival_time,
          [4]=waiting_time, [5]=lot_release_time, [6]=machine_load,
          [7]=total_process_time, [8]=predicted_completion, [9]=stage_count,
          [10]=qtime_risk, [11]=wafer_count, [12]=priority, [13]=due_slack,
          [14-17]=0.0 (排名特征由 _apply_candidate_rank_features 填充)

        启发式打分: priority + waiting_time - 0.01*release_time - 0.001*proc_time - 0.001*qtime_risk
        """
        arrival_time = float(self.encoder.arrival_times[int(lot)])
        waiting_time = max(0.0, self.current_time - arrival_time)
        machine_load = self.encoder.calendar_busy_time(
            self.state.machine_calendar,
            int(machine),
            self.current_time,
        )
        total_process_time = float(dry_run["total_process_time"])
        predicted_completion = float(dry_run["lot_end_time"])
        stage_count = float(len(dry_run["steps"]))
        qtime_risk = float(dry_run["qtime_risk"])
        wafer_count = float(self.encoder.wafer_counts[int(lot)])
        priority = float(self.encoder.priorities.get(int(lot), 0.0))
        due_date = float(self.encoder.due_dates.get(int(lot), np.inf))
        due_slack = 0.0
        if np.isfinite(due_date):
            due_slack = due_date - predicted_completion

        # 报告 Section 4.1: priority 已上移到 priority_filter，不再进入评分
        due_urgency = 0.0
        if np.isfinite(due_date):
            due_urgency = max(0.0, 1.0 - (due_date - predicted_completion) / max(due_date - arrival_time, 1e-9))
        # qtime_deadline 可能为 inf（无 Q-time 约束的 lot，如随机生成实例）。
        # 此时该 lot 无 qtime 余量概念，slack 项置 0，避免 inf 传入特征导致 NaN logits。
        qtime_deadline = float(self.encoder.get_qtime_deadline(int(lot)))
        qtime_slack = (
            max(0.0, qtime_deadline - predicted_completion)
            if np.isfinite(qtime_deadline)
            else 0.0
        )
        score = (
            due_urgency
            + waiting_time
            + 0.1 * qtime_slack          # 报告 §4.1: qtime_slack 线性正项
            - 0.001 * total_process_time
            - 0.001 * qtime_risk
        )
        features = np.asarray(
            [
                1.0,       # is_real
                0.0,       # is_wait
                score,
                arrival_time,
                waiting_time,
                float(dry_run["lot_release_time"]),
                machine_load,
                total_process_time,
                predicted_completion,
                stage_count,
                qtime_risk,
                wafer_count,
                priority,
                due_slack,
                0.0,       # priority_rank_norm (后填充)
                0.0,       # due_slack_rank_norm (后填充)
                0.0,       # is_best_priority (后填充)
                0.0,       # is_most_urgent_due (后填充)
            ],
            dtype=float,
        )
        return features, float(score)

    def _wait_features(self):
        """生成 wait 动作的特征向量 (仅 is_wait=1.0, 其余为 0)。"""
        features = np.zeros(len(self.feature_names), dtype=float)
        features[1] = 1.0
        return features

    # ---- 辅助方法 ----

    def _all_lots(self):
        """返回所有 Lot ID 的集合 {1, 2, ..., num_lots}。"""
        return set(range(1, int(self.encoder.num_lots) + 1))

    def _coerce_action(self, action):
        """将各种形式的 action 统一转换为 DispatchAction。

        支持: DispatchAction, namedtuple, 或带 .lot/.machine/.ppid 属性的任意对象。
        """
        if isinstance(action, DispatchAction):
            return action

        return DispatchAction(
            lot=int(action.lot),
            machine=int(action.machine),
            ppid=int(action.ppid),
            is_wait=bool(getattr(action, "is_wait", False)),
            is_padding=bool(getattr(action, "is_padding", False)),
        )

    def _copy_state(self, state):
        """深拷贝 ScheduleState (包括日历的列表拷贝)。"""
        return ScheduleState(
            machine_available_time=dict(getattr(state, "machine_available_time", {})),
            chamber_available_time=dict(getattr(state, "chamber_available_time", {})),
            machine_calendar=self.encoder.copy_calendar(
                getattr(state, "machine_calendar", {})
            ),
            chamber_calendar=self.encoder.copy_calendar(
                getattr(state, "chamber_calendar", {})
            ),
            current_time=float(getattr(state, "current_time", 0.0)),
            completed_lots=set(getattr(state, "completed_lots", set())),
            commit_log=list(getattr(state, "commit_log", [])),
            planning_window=getattr(state, "planning_window", None),
            schedules=dict(getattr(state, "schedules", {})),
        )

    def _snapshot_environment(self):
        """创建环境的完整快照 (用于异常时的原子回滚)。"""
        return {
            "current_time": self.current_time,
            "state": self._copy_state(self.state),
            "completed_lots": set(self.completed_lots),
            "lot_schedule": self.lot_schedule.copy(),
            "wafer_schedule": self.wafer_schedule.copy(),
        }

    def _restore_environment(self, snapshot):
        """从快照恢复环境状态。"""
        self.current_time = float(snapshot["current_time"])
        self.state = self._copy_state(snapshot["state"])
        self.completed_lots = set(snapshot["completed_lots"])
        self.lot_schedule = snapshot["lot_schedule"].copy()
        self.wafer_schedule = snapshot["wafer_schedule"].copy()
        self._sync_state_summary()

    def _sync_state_summary(self):
        """将环境的当前状态同步到 self.state 的摘要字段。"""
        self.state.current_time = self.current_time
        self.state.completed_lots = set(self.completed_lots)
        self.state.schedules["lot_schedule"] = self.lot_schedule
        self.state.schedules["wafer_schedule"] = self.wafer_schedule

    def step_info(self):
        """返回当前步骤的摘要信息字典。"""
        lot_schedule = np.asarray(self.lot_schedule, dtype=float).reshape((-1, 5))
        wafer_schedule = np.asarray(self.wafer_schedule, dtype=float).reshape((-1, 9))
        return {
            "current_time": self.current_time,
            "completed_lots": set(self.completed_lots),
            "remaining_lots": self.remaining_lots,
            "lot_schedule_rows": int(lot_schedule.shape[0]),
            "wafer_schedule_rows": int(wafer_schedule.shape[0]),
            "commit_count": len(self.state.commit_log),
            "done": len(self.remaining_lots) == 0,
        }

    def _build_commit_log(self, action, lot_schedule, wafer_schedule, before_counts):
        """构建单次提交的日志条目 (用于回滚和审计)。"""
        lot_rows = np.asarray(lot_schedule, dtype=float).reshape((-1, 5))
        wafer_rows = np.asarray(wafer_schedule, dtype=float).reshape((-1, 9))
        machine_intervals = [
            (int(row[1]), float(row[3]), float(row[4]))
            for row in lot_rows
        ]
        chamber_intervals = [
            ((int(row[2]), int(row[5]), int(row[6])), float(row[7]), float(row[8]))
            for row in wafer_rows
        ]
        return {
            "action": action,
            "machine_intervals": machine_intervals,
            "chamber_intervals": chamber_intervals,
            "lot_schedule_rows": int(lot_rows.shape[0]),
            "wafer_schedule_rows": int(wafer_rows.shape[0]),
            "before_counts": before_counts,
            "completed_lot": int(action.lot),
        }

    # ---- 回滚 ----

    def rollback_last_commit(self):
        """撤销最近一次提交，恢复环境到提交前状态。

        回滚步骤:
          1. 从 commit_log 取出最后一条记录
          2. 从 chamber/machine calendar 中精确移除对应区间
          3. 截断 lot_schedule / wafer_schedule 数组
          4. 恢复 available_time 快照
          5. 从 completed_lots 中移除该 Lot
        """
        if not self.state.commit_log:
            return RollbackResult(
                action=DispatchAction(lot=0, machine=0, ppid=0),
                rolled_back=False,
                state=self.state,
                failure_reason="no_commit_to_rollback",
            )

        last = self.state.commit_log[-1]
        action = last["action"]
        machine_intervals = last["machine_intervals"]
        chamber_intervals = last["chamber_intervals"]
        before_counts = last["before_counts"]

        # 移除腔体区间
        for resource_key, start_time, end_time in chamber_intervals:
            intervals = self.state.chamber_calendar.get(resource_key, [])
            for i in range(len(intervals) - 1, -1, -1):
                if (abs(intervals[i][0] - start_time) < 1e-9
                        and abs(intervals[i][1] - end_time) < 1e-9):
                    intervals.pop(i)
                    break
            if not intervals:
                self.state.chamber_calendar.pop(resource_key, None)

        # 移除机台区间
        for machine, start_time, end_time in machine_intervals:
            intervals = self.state.machine_calendar.get(machine, [])
            for i in range(len(intervals) - 1, -1, -1):
                if (abs(intervals[i][0] - start_time) < 1e-9
                        and abs(intervals[i][1] - end_time) < 1e-9):
                    intervals.pop(i)
                    break
            if not intervals:
                self.state.machine_calendar.pop(machine, None)

        # 截断调度数组并恢复状态快照
        self.lot_schedule = self.lot_schedule[:before_counts["lot_schedule_rows"]]
        self.wafer_schedule = self.wafer_schedule[:before_counts["wafer_schedule_rows"]]
        self.state.machine_available_time = before_counts["machine_available_time"]
        self.state.chamber_available_time = before_counts["chamber_available_time"]
        self.completed_lots.discard(int(action.lot))
        self.state.commit_log.pop()
        self._sync_state_summary()
        info = self.step_info()

        return RollbackResult(
            action=action,
            rolled_back=True,
            state=self.state,
            step_info=info,
            failure_reason="",
        )

    def _count_calendar_conflicts(self, calendar):
        """统计日历中的区间交叠冲突数。"""
        conflict_count = 0

        for intervals in calendar.values():
            sorted_intervals = sorted(intervals)
            for left, right in zip(sorted_intervals, sorted_intervals[1:]):
                if float(left[1]) > float(right[0]):
                    conflict_count += 1

        return conflict_count

    def _simulate_action(self, action, state, noise_rng=False):
        """Commit-path schedule wrapper around lower-layer calendar scheduling.

        ``noise_rng=False`` (默认) 用环境共享的 _noise_rng（受 process_noise_enabled
        门控）——commit 行为。``noise_rng=None`` 走均值路径，不采样噪声，供 dry-run
        和候选池构建使用。传入一个 Generator 则改用它（且不触动共享 rng），供
        chance-constraint mask 做独立多采样用。
        """
        from lower_layer_scheduler import schedule_on_calendar

        lot = int(action.lot)
        machine = int(action.machine)
        ppid = int(action.ppid)
        earliest_release = max(
            self.current_time,
            float(self.encoder.arrival_times[lot]),
        )
        if noise_rng is False:
            rng = self._noise_rng if self.process_noise_enabled else None
        elif noise_rng is None:
            rng = None
        else:
            rng = noise_rng
        res = schedule_on_calendar(
            lot,
            machine,
            ppid,
            self.encoder,
            state,
            earliest_release=earliest_release,
            noise_rng=rng,
        )
        if res.infeasible_reason:
            raise RuntimeError(
                f"schedule_on_calendar failed for Lot {lot}: {res.infeasible_reason}"
            )

        added_intervals = []
        try:
            for resource_key, start_time, end_time in res.batch_intervals:
                self.encoder.add_calendar_interval(
                    state.chamber_calendar,
                    resource_key,
                    start_time,
                    end_time,
                )
                added_intervals.append((resource_key, start_time, end_time))
            m_id, m_start, m_end = res.machine_interval
            self.encoder.add_calendar_interval(
                state.machine_calendar,
                m_id,
                m_start,
                m_end,
            )
        except Exception:
            self.encoder.rollback_calendar_intervals(
                state.chamber_calendar,
                added_intervals,
            )
            raise

        state.machine_available_time[machine] = max(
            state.machine_available_time.get(machine, self.current_time),
            float(res.lot_end),
        )
        for resource_key, _start_time, end_time in res.batch_intervals:
            state.chamber_available_time[resource_key] = max(
                state.chamber_available_time.get(resource_key, self.current_time),
                float(end_time),
            )

        steps = self.encoder.get_process_steps(lot, machine, ppid)
        n_stages = len(steps)
        trial_rows = []
        for b_idx, wafer_ids in enumerate(res.subbatch_wafer_map):
            stage_slice = res.batch_intervals[
                b_idx * n_stages:(b_idx + 1) * n_stages
            ]
            for stage_id, (resource_key, start_time, end_time) in enumerate(
                stage_slice,
                start=1,
            ):
                _machine, chamber, side = resource_key
                for wafer_id in wafer_ids:
                    trial_rows.append([
                        lot,
                        wafer_id,
                        machine,
                        ppid,
                        stage_id,
                        chamber,
                        side,
                        start_time,
                        end_time,
                    ])

        lot_schedule = np.asarray(
            [[lot, machine, ppid, res.machine_interval[1], res.lot_end]],
            dtype=float,
        )
        wafer_schedule = np.asarray(trial_rows, dtype=float)
        return lot_schedule, wafer_schedule, state

    # ---- SAS 步进 (RL 交互接口) ----

    def sas_step(self, machine, action_index, pool=None, reward_config=None):
        """执行一个 SAS 步骤: 选择动作 → 计算奖励 → 状态转移。

        RL 训练的主交互接口，包含四种情况:
          1. 掩码无效 / padding: reward = mask_invalid_penalty, committed=False
          2. wait / ppid=0:      reward = wait_penalty, committed=False
          3. dry_run 失败:       reward = insert_fail_penalty, committed=False
          4. 插入成功:           reward = insert_success + shaping, committed=True
                              + 终态奖励 (若 episode 结束且 use_terminal_reward)

        Args:
            machine: 机台编号。
            action_index: 候选池中的动作索引。
            pool: 候选池 (None 则重新构建)。
            reward_config: RewardConfig 实例。

        Returns:
            SASStepResult — 包含动作、奖励、info、是否提交、是否结束。
        """
        machine = int(machine)
        pool = self.build_candidate_pool(machine) if pool is None else pool
        if reward_config is None:
            reward_config = RewardConfig()

        action_index = int(action_index)
        if action_index < 0 or action_index >= len(pool.actions):
            raise IndexError(f"action_index {action_index} is out of bounds")

        action = self._coerce_action(pool.actions[action_index])

        # 情况 1: 掩码无效 / padding
        if not bool(pool.action_mask[action_index]) or action.is_padding:
            info = {
                "mask_invalid": True,
                "wait_or_noop": False,
                "insertion_success": False,
                "insertion_failed": False,
                "selected_lot": int(action.lot),
                "selected_machine": int(action.machine),
                "selected_ppid": int(action.ppid),
                "selected_lot_start": 0.0,
                "selected_lot_end": 0.0,
                "selected_lot_process_time": 0.0,
                "new_qtime_violation": 0.0,
                "priority_rank_penalty": 0.0,
                "current_time": self.current_time,
                "due_date": float(self.encoder.due_dates.get(int(action.lot), np.inf)),
                "reward_execute": 0.0,
                "reward_wait": 0.0,
                "reward_shape": 0.0,
                "reward_terminal": 0.0,
            }
            reward = compute_sas_reward(info, reward_config)
            return SASStepResult(
                action=action,
                reward=reward,
                info=info,
                committed=False,
                done=len(self.remaining_lots) == 0,
                step_info=self.step_info(),
                failure_reason="mask_invalid",
            )

        # 情况 2: wait 动作
        if action.is_wait or int(action.ppid) == 0:
            info = {
                "mask_invalid": False,
                "wait_or_noop": True,
                "insertion_success": False,
                "insertion_failed": False,
                "selected_lot": int(action.lot),
                "selected_machine": int(action.machine),
                "selected_ppid": int(action.ppid),
                "selected_lot_start": 0.0,
                "selected_lot_end": 0.0,
                "selected_lot_process_time": 0.0,
                "new_qtime_violation": 0.0,
                "priority_rank_penalty": 0.0,
                "current_time": self.current_time,
                "due_date": float(self.encoder.due_dates.get(int(action.lot), np.inf)),
                "reward_execute": 0.0,
                "reward_wait": 0.0,
                "reward_shape": 0.0,
                "reward_terminal": 0.0,
            }
            reward = compute_sas_reward(info, reward_config)
            return SASStepResult(
                action=action,
                reward=reward,
                info=info,
                committed=False,
                done=len(self.remaining_lots) == 0,
                step_info=self.step_info(),
            )

        # 情况 3: dry_run 失败
        dry = self.dry_run_action(action)
        if not dry.success:
            info = {
                "mask_invalid": False,
                "wait_or_noop": False,
                "insertion_success": False,
                "insertion_failed": True,
                "selected_lot": int(action.lot),
                "selected_machine": int(action.machine),
                "selected_ppid": int(action.ppid),
                "selected_lot_start": 0.0,
                "selected_lot_end": 0.0,
                "selected_lot_process_time": 0.0,
                "new_qtime_violation": 0.0,
                "priority_rank_penalty": 0.0,
                "current_time": self.current_time,
                "due_date": float(self.encoder.due_dates.get(int(action.lot), np.inf)),
                "reward_execute": 0.0,
                "reward_wait": 0.0,
                "reward_shape": 0.0,
                "reward_terminal": 0.0,
            }
            reward = compute_sas_reward(info, reward_config)
            return SASStepResult(
                action=action,
                reward=reward,
                info=info,
                committed=False,
                done=len(self.remaining_lots) == 0,
                step_info=self.step_info(),
                failure_reason=dry.failure_reason,
            )

        # 情况 4: 插入成功
        lot_start = float(dry.lot_schedule[0, 3]) if dry.lot_schedule.shape[0] > 0 else 0.0
        lot_end = float(dry.lot_schedule[0, 4]) if dry.lot_schedule.shape[0] > 0 else 0.0
        process_time = lot_end - lot_start
        due_date = float(self.encoder.due_dates.get(int(action.lot), np.inf))
        # 该 lot 的总加工工作量 (供 exec 通道 packing = total_work/span 用)
        try:
            work_steps = self.encoder.get_process_steps(int(action.lot), machine, int(action.ppid))
            total_work = float(self.encoder.estimate_plan_total_process_time(
                work_steps, self.encoder.wafer_counts[int(action.lot)],
            ))
        except (KeyError, ValueError):
            total_work = 0.0

        # 计算新增的 Q-time 违反
        q_before, _ = self.encoder.compute_q_time_violation(self.wafer_schedule)
        result = self.commit_action_index(machine, action_index, pool=pool)
        q_after, _ = self.encoder.compute_q_time_violation(self.wafer_schedule)
        new_qtime_violation = max(0.0, float(q_after) - float(q_before))

        # 优先级排名惩罚: 选择了低优先级 Lot 而跳过高优先级
        priority_rank_penalty = 0.0
        if hasattr(self.encoder, "priorities"):
            lot_priority = float(self.encoder.priorities.get(int(action.lot), 0.0))
            for other_action, is_valid in zip(pool.actions, pool.action_mask):
                other_action = self._coerce_action(other_action)
                if (
                    not bool(is_valid)
                    or other_action.is_padding
                    or other_action.is_wait
                    or int(other_action.lot) == int(action.lot)
                ):
                    continue
                other_priority = float(
                    self.encoder.priorities.get(int(other_action.lot), 0.0)
                )
                priority_rank_penalty += max(0.0, other_priority - lot_priority)

        # 终局通道字段 (供向量奖励的终局奖励使用)
        if self.lot_schedule.size > 0:
            objs = self.encoder.evaluate_objectives(
                self.lot_schedule, self.wafer_schedule, self.current_time
            )
            avg_machine_utilization = float(-objs[5])
        else:
            avg_machine_utilization = 0.0

        info = {
            "mask_invalid": False,
            "wait_or_noop": False,
            "insertion_success": True,
            "insertion_failed": False,
            "selected_lot": int(action.lot),
            "selected_machine": int(action.machine),
            "selected_ppid": int(action.ppid),
            "selected_lot_start": lot_start,
            "selected_lot_end": lot_end,
            "selected_lot_process_time": process_time,
            "selected_lot_total_work": total_work,
            "new_qtime_violation": new_qtime_violation,
            "priority_rank_penalty": priority_rank_penalty,
            "current_time": self.current_time,
            "due_date": due_date,
            "reward_execute": 0.0,
            "reward_wait": 0.0,
            "reward_shape": 0.0,
            "reward_terminal": 0.0,
            "is_terminal": len(self.remaining_lots) == 0,
            "num_lots": int(self.encoder.num_lots),
            "completed_lots": len(self.completed_lots),
            "qtime_violation_count": float(q_after),
            "avg_machine_utilization": avg_machine_utilization,
        }
        reward = compute_sas_reward(info, reward_config)

        return SASStepResult(
            action=action,
            reward=reward,
            info=info,
            committed=True,
            done=len(self.remaining_lots) == 0,
            step_info=result.step_info,
        )

    def _append_schedule_rows(self, schedule, rows, expected_columns):
        """向调度数组追加行 (自动 reshape 和 vstack)。"""
        rows = np.asarray(rows, dtype=float)
        if rows.size == 0:
            return schedule
        rows = rows.reshape((-1, expected_columns))
        if schedule.size == 0:
            return rows.copy()
        return np.vstack((schedule, rows))
