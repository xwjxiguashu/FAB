"""Phase 1 日历问题实例 — 组合问题定义与日历操作，并评估调度目标。

本模块定义:
  - Phase1CalendarProblem: 完整的问题对象 (多重继承 ProblemDefinitionMixin + CalendarDecoderMixin)
  - 多目标评估: Q-time 违反, 拖期, 优先级违反, 设备利用率
  - 工厂函数: build_small_encoder (4×2 小实例), build_pressure_test_encoder (50×10 压力测试)
"""

import numpy as np

from problem import ProblemDefinitionMixin
from resource_calendar import CalendarDecoderMixin


class Phase1CalendarProblem(ProblemDefinitionMixin, CalendarDecoderMixin):
    """Phase 1 日历环境的最小问题对象。

    组合了问题定义 (ProblemDefinitionMixin) 和日历操作 (CalendarDecoderMixin)，
    并提供目标函数评估和调度完整性校验。"""

    # ---- Q-time 约束评估 ----

    def compute_q_time_violation(self, wafer_schedule):
        """计算晶圆级 Q-time 违反情况。

        Q-time 约束: 对于指定的 (lot, machine, ppid, from_stage, to_stage)，
        to_stage 的开始时间不得晚于 from_stage 的结束时间 + q_time_limit。

        Returns:
            (violation_count, total_violation) — 违反次数和累计超时量。
        """
        wafer_schedule = np.array(wafer_schedule, dtype=float)

        if wafer_schedule.size == 0 or len(self.q_time_limits) == 0:
            return 0.0, 0.0

        # 建立 (lot, wafer, machine, ppid, stage) → row 查找表
        operation_rows = {}
        for row in wafer_schedule:
            lot = int(row[0])
            wafer_id = int(row[1])
            machine = int(row[2])
            ppid = int(row[3])
            stage_id = int(row[4])
            operation_rows[(lot, wafer_id, machine, ppid, stage_id)] = row

        violation_count = 0.0
        total_violation = 0.0

        for key, q_time_limit in self.q_time_limits.items():
            lot, machine, ppid, from_stage, to_stage = map(int, key)
            q_time_limit = float(q_time_limit)

            for wafer_id in range(1, int(self.wafer_counts[lot]) + 1):
                from_key = (lot, wafer_id, machine, ppid, from_stage)
                to_key = (lot, wafer_id, machine, ppid, to_stage)

                if from_key not in operation_rows or to_key not in operation_rows:
                    continue

                # Q-time 截止 = from_stage 结束时间 + 限制
                q_due_date = float(operation_rows[from_key][8]) + q_time_limit
                violation = max(0.0, float(operation_rows[to_key][7]) - q_due_date)
                if violation > 0.0:
                    violation_count += 1.0
                    total_violation += violation

        return violation_count, total_violation

    # ---- 优先级违反评估 ----

    def compute_priority_violation(self, ordered_rows):
        """计算调度的优先级违反程度。

        对按开始时间排序的 lot_schedule 行，累计所有"低优先级 Lot 排在
        高优先级 Lot 之前"的情况: sum(max(0, later_priority - earlier_priority))。

        该度量反映了调度序列与优先级顺序的偏离程度。
        """
        priority_violation = 0.0
        rows = list(ordered_rows)

        for left_index, left_row in enumerate(rows):
            left_priority = float(self.priorities.get(int(left_row[0]), 0.0))
            for right_row in rows[left_index + 1:]:
                right_priority = float(self.priorities.get(int(right_row[0]), 0.0))
                priority_violation += max(0.0, right_priority - left_priority)

        return float(priority_violation)

    # ---- 多目标评估 ----

    def evaluate_objectives(self, lot_schedule, wafer_schedule, current_time=0.0):
        """评估调度的 6 个目标函数。

        目标向量 (最小化方向):
          [0] q_time_count     — Q-time 违反次数
          [1] q_time_total     — Q-time 累计超时
          [2] tardy_count      — 拖期 Lot 数
          [3] total_tardiness  — 总拖期量
          [4] priority_violation — 优先级违反 (排序偏离)
          [5] -avg_utilization — 负平均设备利用率 (最小化 -利用率 = 最大化利用率)

        Returns:
            numpy 数组, shape=(6,), dtype=float。
        """
        lot_schedule = np.array(lot_schedule, dtype=float)
        wafer_schedule = np.array(wafer_schedule, dtype=float)

        if lot_schedule.size == 0:
            return np.zeros(6, dtype=float)

        # 拖期统计
        tardy_count = 0.0
        total_tardiness = 0.0

        for row in lot_schedule:
            lot = int(row[0])
            lot_end_time = float(row[4])
            due_date = float(self.due_dates.get(lot, np.inf))
            tardiness = max(0.0, lot_end_time - due_date)
            if tardiness > 0.0:
                tardy_count += 1.0
            total_tardiness += tardiness

        # 优先级违反: 按开始时间排序后评估序列质量
        ordered_rows = sorted(lot_schedule, key=lambda row: float(row[3]))
        priority_violation = self.compute_priority_violation(ordered_rows)

        # 平均利用率: busy_time / (machines * horizon)
        active_lot_rows = lot_schedule[lot_schedule[:, 4] > current_time]

        if active_lot_rows.size > 0:
            clipped_starts = np.maximum(active_lot_rows[:, 3], current_time)
            busy_time = float(np.sum(active_lot_rows[:, 4] - clipped_starts))
            horizon_end = float(np.max(active_lot_rows[:, 4]))
            horizon = max(horizon_end - float(current_time), 1e-9)
            avg_utilization = busy_time / (float(self.num_machines) * horizon)
        else:
            avg_utilization = 0.0

        q_count, q_total = self.compute_q_time_violation(wafer_schedule)
        return np.array([
            q_count,
            q_total,
            tardy_count,
            total_tardiness,
            priority_violation,
            -avg_utilization,  # 负号: 最小化方向
        ], dtype=float)

    # ---- 调度完整性校验 ----

    def validate_final_schedule_completeness(self, lot_schedule, wafer_schedule):
        """校验最终调度的完整性。

        检查清单:
          1. lot_schedule (n,5) 和 wafer_schedule (m,9) 形状正确且非空
          2. 每个 Lot 恰好出现一次 (无重复、无缺失)
          3. 每个 Lot 的所有 Wafer 和 Stage 均被覆盖
          4. 机台和腔体日历无区间交叠

        Returns:
            True 表示校验通过。
        """
        lot_schedule = np.array(lot_schedule, dtype=float)
        wafer_schedule = np.array(wafer_schedule, dtype=float)
        expected_lots = set(range(1, self.num_lots + 1))

        if lot_schedule.size == 0:
            raise ValueError("lot_schedule is empty")
        if wafer_schedule.size == 0:
            raise ValueError("wafer_schedule is empty")
        if lot_schedule.ndim != 2 or lot_schedule.shape[1] != 5:
            raise ValueError("lot_schedule must have 5 columns")
        if wafer_schedule.ndim != 2 or wafer_schedule.shape[1] != 9:
            raise ValueError("wafer_schedule must have 9 columns")

        # 每个 Lot 恰好一行
        scheduled_lots = [int(row[0]) for row in lot_schedule]
        if set(scheduled_lots) != expected_lots:
            raise ValueError("lot_schedule does not cover all lots exactly once")
        if len(scheduled_lots) != len(set(scheduled_lots)):
            raise ValueError("lot_schedule contains duplicate lots")

        # 每个 Lot 的 Wafer 和 Stage 全覆盖检查
        wafer_rows_by_lot = {}
        for row in wafer_schedule:
            lot = int(row[0])
            wafer_rows_by_lot.setdefault(lot, []).append(row)

        for row in lot_schedule:
            lot = int(row[0])
            machine = int(row[1])
            ppid = int(row[2])
            rows = wafer_rows_by_lot.get(lot, [])
            expected_wafer_ids = set(range(1, int(self.wafer_counts[lot]) + 1))
            actual_wafer_ids = {int(wafer_row[1]) for wafer_row in rows}
            expected_stages = set(range(1, len(self.ppid_steps[(lot, machine, ppid)]) + 1))

            if actual_wafer_ids != expected_wafer_ids:
                raise ValueError(f"lot {lot} wafer ids are incomplete")

            for wafer_id in expected_wafer_ids:
                actual_stages = {
                    int(wafer_row[4])
                    for wafer_row in rows
                    if int(wafer_row[1]) == wafer_id
                }
                if actual_stages != expected_stages:
                    raise ValueError(f"lot {lot} wafer {wafer_id} stages are incomplete")

        # 构建并验证日历无交叠
        machine_calendar = {}
        chamber_calendar = {}
        for row in lot_schedule:
            machine_calendar.setdefault(int(row[1]), []).append((float(row[3]), float(row[4])))
        # 批处理 (报告 §1.5): 同一子批的多片 wafer 共享一个 (chamber,side) 区间，
        # 故按 (resource, start, end) 去重，避免把"同批共享"误判为区间重叠。
        chamber_seen = {}
        for row in wafer_schedule:
            resource_key = (int(row[2]), int(row[5]), int(row[6]))
            interval = (float(row[7]), float(row[8]))
            seen = chamber_seen.setdefault(resource_key, set())
            if interval in seen:
                continue
            seen.add(interval)
            chamber_calendar.setdefault(resource_key, []).append(interval)

        for intervals in machine_calendar.values():
            intervals.sort()
        for intervals in chamber_calendar.values():
            intervals.sort()

        self.validate_no_interval_overlap(machine_calendar, "machine_calendar")
        self.validate_no_interval_overlap(chamber_calendar, "chamber_calendar")
        return True


# =============================================================================
# 问题实例工厂函数
# =============================================================================


def build_small_encoder():
    """构建 4 Lot × 2 机台的小规模测试实例。

    包含:
      - 4 个 Lot, 2 个机台 (Machine 1, 2)
      - 每 Lot 2-3 片晶圆
      - 每 (Lot, Machine) 有 1 个 PPID, 2 个工艺步骤
      - 每步有 2 个备选腔体资源 (chamber, side)
      - Q-time 约束: 所有 stage 1→2 间隔不超过 4-5 时间单位
      - 到达时间: t=0.0, 1.5, 3.0, 4.0
      - 优先级: Lot 2 (4.0) > Lot 3 (3.0) > Lot 1 (2.0) > Lot 4 (1.0)
    """
    num_lots = 4
    feasible_machines = {
        1: [1, 2],
        2: [1],
        3: [2],
        4: [1, 2],
    }
    feasible_ppids = {
        (1, 1): [101],
        (1, 2): [201],
        (2, 1): [102],
        (3, 2): [202],
        (4, 1): [103],
        (4, 2): [203],
    }
    ppid_steps = {
        # 每步 (chamber, side, processing_time) 数组
        (1, 1, 101): [
            np.array([[1, 0, 3.0], [2, 0, 2.5]]),
            np.array([[1, 1, 2.0], [2, 1, 2.8]]),
        ],
        (1, 2, 201): [
            np.array([[1, 0, 2.4], [2, 1, 3.0]]),
            np.array([[1, 1, 2.2], [2, 0, 2.6]]),
        ],
        (2, 1, 102): [
            np.array([[1, 0, 3.5], [2, 0, 4.0]]),
            np.array([[1, 1, 1.8], [2, 1, 2.2]]),
        ],
        (3, 2, 202): [
            np.array([[1, 1, 4.0], [2, 0, 3.6]]),
            np.array([[1, 0, 2.5], [2, 1, 2.0]]),
        ],
        (4, 1, 103): [
            np.array([[1, 0, 2.8], [2, 1, 3.2]]),
            np.array([[1, 1, 2.6], [2, 0, 2.1]]),
        ],
        (4, 2, 203): [
            np.array([[1, 0, 3.0], [2, 0, 2.7]]),
            np.array([[1, 1, 2.4], [2, 1, 2.9]]),
        ],
    }
    arrival_times = {
        1: 0.0,
        2: 1.5,
        3: 3.0,
        4: 4.0,
    }

    # 构造 process_time_sigma: σ = 5% × μ（每阶段最小加工时间）
    process_time_sigma = {}
    for lot in range(1, num_lots + 1):
        for machine in feasible_machines[lot]:
            for ppid in feasible_ppids[(lot, machine)]:
                steps = ppid_steps[(lot, machine, ppid)]
                sigmas = []
                for stage in steps:
                    stage_arr = np.asarray(stage, dtype=float)
                    mu_min = float(np.min(stage_arr[:, 2]))
                    sigmas.append(0.05 * mu_min)
                process_time_sigma[(lot, machine, ppid)] = sigmas

    # 构造 qtime_deadline: arrival + 2 × 所有阶段均值加工时间之和
    qtime_deadline = {}
    for lot in range(1, num_lots + 1):
        arrival = arrival_times[lot]
        machine = feasible_machines[lot][0]
        ppid = feasible_ppids[(lot, machine)][0]
        steps = ppid_steps[(lot, machine, ppid)]
        total_pt = sum(
            float(np.min(np.asarray(stage, dtype=float)[:, 2]))
            for stage in steps
        )
        qtime_deadline[lot] = arrival + 2.0 * total_pt

    encoder = Phase1CalendarProblem(
        num_lots=num_lots,
        num_machines=2,
        feasible_machines=feasible_machines,
        feasible_ppids=feasible_ppids,
        ppid_steps=ppid_steps,
        wafer_counts={
            1: 3,
            2: 2,
            3: 2,
            4: 3,
        },
        process_time_sigma=process_time_sigma,
        qtime_deadline=qtime_deadline,
        z_eps=2.05,
        side_capacity=4,
    )
    encoder.arrival_times = arrival_times
    encoder.due_dates = {
        1: 22.0,
        2: 18.0,
        3: 24.0,
        4: 28.0,
    }
    encoder.q_time_limits = {
        (1, 1, 101, 1, 2): 5.0,
        (1, 2, 201, 1, 2): 5.0,
        (2, 1, 102, 1, 2): 4.0,
        (3, 2, 202, 1, 2): 4.0,
        (4, 1, 103, 1, 2): 5.0,
        (4, 2, 203, 1, 2): 5.0,
    }
    encoder.priorities = {
        1: 2.0,
        2: 4.0,
        3: 3.0,
        4: 1.0,
    }
    encoder.recipe = {
        1: "R1",
        2: "R2",
        3: "R3",
        4: "R4",
    }
    encoder.machine_group = {
        1: "G1",
        2: "G1",
    }
    encoder.machine_resources = {
        1: [(1, 0), (1, 1), (2, 0), (2, 1)],
        2: [(1, 0), (1, 1), (2, 0), (2, 1)],
    }
    return encoder


def _late_hi_priorities(arrival_times, rng, target_corr=0.97):
    """高优先级晚到 (late_hi) 的 priority 构造 (报告4 §9.8)。

    priority = scale(arrival) + 噪声; 噪声幅度按目标 Pearson 相关系数设定:
    corr = 1/sqrt(1+k^2) ⇒ k = sqrt(1/corr^2 - 1)。仿射缩放到 [0,10] 不改变相关性。
    只有"高优先级 Lot 晚到"这种结构才给预留留有杠杆 (否则 oracle 无从区分
    "预留没用"与"实例没给预留发挥空间")。
    """
    lots = sorted(arrival_times)
    signal = np.array([arrival_times[l] for l in lots], dtype=float)
    signal_std = float(np.std(signal))
    target_corr = float(min(max(target_corr, 1e-3), 0.999))
    k = float(np.sqrt(1.0 / (target_corr ** 2) - 1.0))
    if signal_std > 0.0:
        noise = rng.normal(0.0, k * signal_std, size=len(signal))
    else:
        noise = np.zeros(len(signal))
    raw = signal + noise
    lo = float(np.min(raw))
    hi = float(np.max(raw))
    span = hi - lo if hi > lo else 1.0
    scaled = (raw - lo) / span * 10.0
    return {lot: float(scaled[i]) for i, lot in enumerate(lots)}


def build_pressure_test_encoder(
    seed=2026,
    qtime_limit=3.0,
    arrival_mean_gap=0.6,
    priority_mode="random",
    priority_arrival_corr=0.97,
):
    """构建 50 Lot × 10 机台的压力测试实例。

    参数:
      - 50 Lot, 10 机台, 每机台 5 PPID, 每 Lot 10 晶圆
      - 5 腔体, 2 面 (A/B), 3 个工艺步骤
      - 处理时间: 1.5 + 0.2*stage + U(0, 2.5), 每步 2-4 个备选资源
      - 到达时间: Poisson 到达 (指数间隔, 均值 arrival_mean_gap), lot1 在 t=0 (错峰)
      - 交货期: arrival + 180 + 0.5*lot_id
      - 优先级: priority_mode="random" → U(0, 10) (与到达无关, 默认);
                priority_mode="late_hi" → 与到达高度正相关 (高优先级晚到, 报告4 §9.8)
      - 配方: 5 种配方循环分配
      - 机台组: 每 5 台一组
      - 阶段间 Q-time: 对 (1,2) 与 (2,3) 设上限 qtime_limit (材料队列时间约束)

    可调旋钮 (制造区分度, 见 docs 上下层解耦后续):
      qtime_limit:      阶段间 Q-time 上限 (越小越易因腔体争用而违反 → 派工序列更关键)
      arrival_mean_gap: 错峰到达的平均间隔 (越大越分散 → util 余量越多, "等 vs 派"更有意义)

    历史 bug 修复: 此实例此前从不设置 q_time_limits, 导致 compute_q_time_violation
    恒为 0 → qtime 指标/奖励通道 r_qtime/§3.3 Lagrangian 全部静默失效。

    用途: Phase 1 环境演示的压力测试、Phase 2 RL 训练的大规模基准。
    """
    num_lots = 50
    num_machines = 10
    ppids_per_machine = 5
    wafers_per_lot = 10
    num_chambers = 5
    num_sides = 2
    num_steps = 3

    rng = np.random.default_rng(seed)

    # 所有 Lot 有相同晶圆数
    wafer_counts = {
        lot: wafers_per_lot
        for lot in range(1, num_lots + 1)
    }

    # 所有 Lot 在所有机台上均可加工 (完全柔性)
    feasible_machines = {
        lot: list(range(1, num_machines + 1))
        for lot in range(1, num_lots + 1)
    }

    feasible_ppids = {}
    ppid_steps = {}

    for lot in range(1, num_lots + 1):
        for machine in range(1, num_machines + 1):
            # PPID 编号: lot*10000 + machine*100 + ppid_index
            ppids = [
                lot * 10000 + machine * 100 + ppid_index
                for ppid_index in range(1, ppids_per_machine + 1)
            ]
            feasible_ppids[(lot, machine)] = ppids

            for ppid in ppids:
                steps = []
                for stage_id in range(1, num_steps + 1):
                    # 每步随机 2-4 个备选 (chamber, side) 资源
                    candidate_count = int(rng.integers(2, 5))
                    candidates = []
                    for _ in range(candidate_count):
                        chamber = int(rng.integers(1, num_chambers + 1))
                        side = int(rng.integers(0, num_sides))
                        base_time = 1.5 + 0.2 * stage_id
                        process_time = float(base_time + rng.uniform(0.0, 2.5))
                        candidates.append([chamber, side, process_time])
                    steps.append(np.array(candidates, dtype=float))
                ppid_steps[(lot, machine, ppid)] = steps

    # 错峰到达 (Poisson 到达过程): 指数间隔累加, lot1 在 t=0。
    # 制造 util 余量, 并让"现在派 vs 等下一个更优 lot"成为有意义的决策 (喂将来的 DDT)。
    inter_arrival_gaps = rng.exponential(arrival_mean_gap, size=num_lots)
    inter_arrival_gaps[0] = 0.0
    arrival_cumsum = np.cumsum(inter_arrival_gaps)
    arrival_times = {
        lot: float(arrival_cumsum[lot - 1])
        for lot in range(1, num_lots + 1)
    }

    # 构造 process_time_sigma: σ = 5% × μ（每阶段最小加工时间）
    process_time_sigma = {}
    for lot in range(1, num_lots + 1):
        for machine in feasible_machines[lot]:
            for ppid in feasible_ppids[(lot, machine)]:
                steps = ppid_steps[(lot, machine, ppid)]
                sigmas = []
                for stage in steps:
                    stage_arr = np.asarray(stage, dtype=float)
                    mu_min = float(np.min(stage_arr[:, 2]))
                    sigmas.append(0.05 * mu_min)
                process_time_sigma[(lot, machine, ppid)] = sigmas

    # 构造 qtime_deadline: arrival + 2 × 所有阶段均值加工时间之和
    qtime_deadline = {}
    for lot in range(1, num_lots + 1):
        arrival = arrival_times[lot]
        machine = feasible_machines[lot][0]
        ppid = feasible_ppids[(lot, machine)][0]
        steps = ppid_steps[(lot, machine, ppid)]
        total_pt = sum(
            float(np.min(np.asarray(stage, dtype=float)[:, 2]))
            for stage in steps
        )
        qtime_deadline[lot] = arrival + 2.0 * total_pt

    encoder = Phase1CalendarProblem(
        num_lots=num_lots,
        num_machines=num_machines,
        feasible_machines=feasible_machines,
        feasible_ppids=feasible_ppids,
        ppid_steps=ppid_steps,
        wafer_counts=wafer_counts,
        process_time_sigma=process_time_sigma,
        qtime_deadline=qtime_deadline,
        z_eps=2.05,
        side_capacity=4,
    )

    # 所有 Lot 在 t=0 到达
    encoder.arrival_times = arrival_times
    # 交货期 = 到达时间 + 180 + 0.5*lot_id (宽松截止期)
    encoder.due_dates = {
        lot: float(encoder.arrival_times[lot] + 180.0 + 0.5 * lot)
        for lot in range(1, num_lots + 1)
    }
    # 阶段间 Q-time 上限 (材料队列时间约束): (1,2) 与 (2,3) 两个窗口。
    # 键 (lot, machine, ppid, from_stage, to_stage) → limit; 对所有可被调度的
    # (machine, ppid) 都登记 (约束是 lot 材料属性, 与最终选哪台/哪个配方无关)。
    # mask 仍只 enforce lot 完工期 (qtime_deadline); 阶段间违反由指标/奖励/Lagrangian 处理。
    encoder.q_time_limits = {
        (lot, machine, ppid, from_stage, to_stage): float(qtime_limit)
        for lot in range(1, num_lots + 1)
        for machine in range(1, num_machines + 1)
        for ppid in feasible_ppids[(lot, machine)]
        for (from_stage, to_stage) in ((1, 2), (2, 3))
    }
    # 优先级: "random" (默认, U(0,10), 与到达无关) 或
    #          "late_hi" (与到达高度正相关, 高优先级晚到, 报告4 §9.8 go/no-go 用)
    if priority_mode == "late_hi":
        encoder.priorities = _late_hi_priorities(
            arrival_times, rng, target_corr=priority_arrival_corr
        )
    elif priority_mode == "random":
        encoder.priorities = {
            lot: float(rng.uniform(0.0, 10.0))
            for lot in range(1, num_lots + 1)
        }
    else:
        raise ValueError(f"unknown priority_mode: {priority_mode!r}")
    # 5 种配方循环
    encoder.recipe = {
        lot: f"R{1 + ((lot - 1) % 5)}"
        for lot in range(1, num_lots + 1)
    }
    # 每 5 台机台一组
    encoder.machine_group = {
        machine: f"G{1 + ((machine - 1) // 5)}"
        for machine in range(1, num_machines + 1)
    }
    # 所有腔体资源对所有机台可用
    encoder.machine_resources = {
        machine: [
            (chamber, side)
            for chamber in range(1, num_chambers + 1)
            for side in range(num_sides)
        ]
        for machine in range(1, num_machines + 1)
    }
    return encoder


def build_late_hi_encoder(
    seed=2026,
    qtime_limit=3.0,
    arrival_mean_gap=0.6,
    priority_arrival_corr=0.97,
):
    """高优先级晚到的压力实例 (报告4 §9.8 的 late_hi)。

    与 build_pressure_test_encoder 同骨架, 但 priority 与到达时间高度正相关
    (corr≈priority_arrival_corr): 高优先级 Lot 倾向晚到。这是 oracle 预留上界
    go/no-go 验证 (§6.2.3 阶段 0) 必须使用的区分实例 —— 只有这种"晚到高优先级 +
    当前派工会挤占未来"的结构才给选择性预留留出杠杆。
    """
    return build_pressure_test_encoder(
        seed=seed,
        qtime_limit=qtime_limit,
        arrival_mean_gap=arrival_mean_gap,
        priority_mode="late_hi",
        priority_arrival_corr=priority_arrival_corr,
    )


def format_objectives(objectives):
    """将目标向量格式化为可读字符串。

    Args:
        objectives: shape=(6,) numpy 数组 (来自 evaluate_objectives)。

    Returns:
        逗号分隔的 "key:value" 字符串。
    """
    objectives = np.asarray(objectives, dtype=float)
    return (
        f"q_time_count:{objectives[0]:.0f},"
        f"q_time_total:{objectives[1]:.3f},"
        f"tardy_count:{objectives[2]:.0f},"
        f"total_tardiness:{objectives[3]:.3f},"
        f"priority_violation:{objectives[4]:.3f},"
        f"avg_utilization:{-objectives[5]:.6f}"
    )