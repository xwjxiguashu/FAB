"""问题定义 Mixin — 半导体 FAB 调度问题的数据结构与合法性校验。

定义了 Lot(批次)、Machine(机台)、PPID(工艺配方)、工艺步骤(Stage)、
Q-time 约束等核心概念的数据结构，并提供了参数校验方法。
"""

import numpy as np


class ProblemDefinitionMixin:
    """FAB 调度问题定义的 Mixin 基类。

    存储并校验所有问题实例数据:
      - feasible_machines[lot] → [machine, ...]  每个 Lot 可选机台列表
      - feasible_ppids[(lot, machine)] → [ppid, ...]  每个 (Lot, 机台) 组合的可用配方
      - ppid_steps[(lot, machine, ppid)] → [stage_array, ...]  每个配方的工艺步骤
        每步为 (n,3) 数组，列依次为 [chamber, side, processing_time]
      - wafer_counts[lot] → int  每个 Lot 的晶圆数量
      - due_dates[lot] → float  交货期
      - priorities[lot] → float  优先级 (越大越紧急)
      - q_time_limits[(lot, machine, ppid, from_stage, to_stage)] → float  Q-time 约束
      - recipe[lot] → str  配方标识 (用于机台-配方兼容性检查)
      - machine_group[machine] → str  机台组标识
      - machine_resources[machine] → [(chamber, side), ...]  机台拥有的腔体资源
      - machine_recipes[machine] → [recipe, ...]  机台可加工的配方列表
    """

    def __init__(
        self,
        num_lots,
        num_machines,
        feasible_machines,
        feasible_ppids,
        ppid_steps,
        wafer_counts,
        due_dates=None,
        priorities=None,
        q_time_limits=None,
        recipe=None,
        machine_group=None,
        machine_resources=None,
        machine_recipes=None,
        process_time_sigma=None,   # {(lot, machine, ppid): [σ_stage1, σ_stage2, ...]}
        qtime_deadline=None,       # {lot: float} 绝对截止时刻（过期即报废）
        z_eps=2.05,                # 违规概率 ε 对应分位数 (ε≈2% → z_eps=2.05)
        side_capacity=None,        # 每个 side 的 wafer 容量（用于子批计算）
    ):
        self.num_lots = int(num_lots)
        self.num_machines = int(num_machines)
        self.feasible_machines = feasible_machines
        self.feasible_ppids = feasible_ppids
        self.ppid_steps = ppid_steps
        self.wafer_counts = wafer_counts
        self.due_dates = due_dates or {
            lot: np.inf for lot in range(1, self.num_lots + 1)
        }
        self.priorities = priorities or {
            lot: 0.0 for lot in range(1, self.num_lots + 1)
        }
        self.q_time_limits = q_time_limits or {}
        self.arrival_times = {}
        self._machine_list_cache = None
        self._ppid_list_cache = None
        self.recipe = recipe or {}
        self.machine_group = machine_group or {}
        self.machine_resources = machine_resources or {}
        self.machine_recipes = machine_recipes or {}
        self.process_time_sigma = process_time_sigma or {}
        self.qtime_deadline = qtime_deadline or {}
        self.z_eps = float(z_eps)
        self.side_capacity = int(side_capacity) if side_capacity is not None else None
        self._problem_validated = False

    def validate_problem_definition(self):
        """校验问题定义完整性。

        检查内容:
          1. 每个 Lot 的到达时间、晶圆数、可行机台、交货期、优先级非空且合法
          2. 每个 (Lot, Machine) 有至少一个 PPID
          3. 每个 (Lot, Machine, PPID) 的工艺步骤非空，每步为 (n,3) 形状
          4. Q-time 约束的键格式与 stage 索引合法
          5. 处理时间为正数

        校验通过后缓存机台和 PPID 查找表以加速后续查询。
        """
        # 逐 Lot 检查基本字段完整性
        for lot in range(1, self.num_lots + 1):
            if lot not in self.arrival_times:
                raise ValueError(f"arrival_times is missing Lot {lot}")
            if lot not in self.wafer_counts:
                raise ValueError(f"wafer_counts is missing Lot {lot}")
            if lot not in self.feasible_machines:
                raise ValueError(f"feasible_machines is missing Lot {lot}")
            if lot not in self.due_dates:
                raise ValueError(f"due_dates is missing Lot {lot}")
            if lot not in self.priorities:
                raise ValueError(f"priorities is missing Lot {lot}")
            if int(self.wafer_counts[lot]) <= 0:
                raise ValueError(f"wafer_counts[{lot}] must be positive")

            machine_list = np.asarray(self.feasible_machines[lot], dtype=int)
            if machine_list.size == 0:
                raise ValueError(f"Lot {lot} has no feasible machines")

            # 逐机台检查 PPID 和工艺步骤
            for machine in machine_list:
                machine = int(machine)
                if (lot, machine) not in self.feasible_ppids:
                    raise ValueError(f"feasible_ppids is missing key ({lot}, {machine})")

                ppid_list = np.asarray(self.feasible_ppids[(lot, machine)], dtype=int)
                if ppid_list.size == 0:
                    raise ValueError(f"Lot {lot} on machine {machine} has no PPIDs")

                for ppid in ppid_list:
                    key = (lot, machine, int(ppid))
                    if key not in self.ppid_steps:
                        raise ValueError(f"ppid_steps is missing key {key}")

                    steps = self.ppid_steps[key]
                    if len(steps) == 0:
                        raise ValueError(f"{key} has no process stages")

                    # 每步必须是 (n, 3) 形状: [chamber, side, time]
                    for stage_id, stage in enumerate(steps, start=1):
                        resources = np.asarray(stage, dtype=float)
                        if resources.ndim != 2 or resources.shape[1] != 3:
                            raise ValueError(f"{key} stage {stage_id} must have shape (n, 3)")
                        if resources.shape[0] == 0:
                            raise ValueError(f"{key} stage {stage_id} has no resources")
                        if np.any(resources[:, 2] <= 0.0):
                            raise ValueError(f"{key} stage {stage_id} has nonpositive time")

        # 校验 Q-time 约束
        for key, q_time_limit in self.q_time_limits.items():
            if len(key) != 5:
                raise ValueError(
                    "q_time_limits keys must be "
                    "(lot, machine, ppid, from_stage, to_stage)"
                )
            lot, machine, ppid, from_stage, to_stage = map(int, key)
            process_key = (lot, machine, ppid)
            if process_key not in self.ppid_steps:
                raise ValueError(f"q_time_limits references unknown process {process_key}")
            stage_count = len(self.ppid_steps[process_key])
            if from_stage < 1 or from_stage > stage_count:
                raise ValueError(f"q_time_limits has invalid from_stage in {key}")
            if to_stage < 1 or to_stage > stage_count:
                raise ValueError(f"q_time_limits has invalid to_stage in {key}")
            if to_stage <= from_stage:
                raise ValueError(f"q_time_limits requires to_stage > from_stage in {key}")
            if float(q_time_limit) < 0.0:
                raise ValueError(f"q_time_limit must be non-negative in {key}")

        # 缓存机台和 PPID 列表的 numpy 数组以加速后续查询
        self._machine_list_cache = {
            lot: np.asarray(self.feasible_machines[lot], dtype=int)
            for lot in range(1, self.num_lots + 1)
        }
        self._ppid_list_cache = {
            (lot, int(machine)): np.asarray(self.feasible_ppids[(lot, int(machine))], dtype=int)
            for lot in range(1, self.num_lots + 1)
            for machine in self._machine_list_cache[lot]
        }
        self._problem_validated = True
        return True

    def get_machine_list(self, lot):
        """返回 Lot 的可行机台列表 (numpy 数组)。"""
        if self._machine_list_cache is None:
            return np.asarray(self.feasible_machines[int(lot)], dtype=int)
        return self._machine_list_cache[int(lot)]

    def get_ppid_list(self, lot, machine):
        """返回 (Lot, Machine) 的可行 PPID 列表 (numpy 数组)。"""
        key = (int(lot), int(machine))
        if self._ppid_list_cache is None:
            return np.asarray(self.feasible_ppids[key], dtype=int)
        return self._ppid_list_cache[key]

    def get_process_steps(self, lot, machine, ppid):
        """返回 (Lot, Machine, PPID) 的工艺步骤列表。"""
        return self.ppid_steps[(int(lot), int(machine), int(ppid))]

    def get_process_time_sigma(self, lot, machine, ppid):
        """返回 (lot, machine, ppid) 各阶段的加工时间标准差列表。

        若未定义，返回各阶段均值的 5% 作为默认噪声。
        """
        key = (int(lot), int(machine), int(ppid))
        if key in self.process_time_sigma:
            return list(self.process_time_sigma[key])
        # 默认: σ = 5% × μ（每阶段取第一个资源的加工时间）
        steps = self.get_process_steps(lot, machine, ppid)
        sigma_list = []
        for stage in steps:
            stage_arr = np.asarray(stage, dtype=float)
            mu_min = float(np.min(stage_arr[:, 2])) if stage_arr.shape[0] > 0 else 1.0
            sigma_list.append(0.05 * mu_min)
        return sigma_list

    def get_qtime_deadline(self, lot):
        """返回 Lot 的绝对 Q-time 截止时刻。

        若未定义，返回 np.inf（无截止约束）。
        """
        return float(self.qtime_deadline.get(int(lot), np.inf))