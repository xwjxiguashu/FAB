"""日历解码器 Mixin — 基于区间的资源日历操作。

提供机台和腔体资源的时间区间管理:
  - 区间插入/删除 (维护有序列表)
  - 最早可用槽位查找 (贪心扫描)
  - 交叠冲突检测
  - 忙时统计
  - Q-time 风险评估
"""

from bisect import bisect_left, bisect_right

import numpy as np


class CalendarDecoderMixin:
    """资源日历区间操作的 Mixin 基类。

    所有日历均表示为 {resource_key: [(start, end), ...]}，
    其中区间列表按 start 升序维护 (通过 bisect 保证插入有序)。
    """

    # ---- 日历复制与基本操作 ----

    def copy_calendar(self, calendar):
        """深拷贝日历 (每个 resource_key 的区间列表重新创建)。"""
        return {
            key: list(intervals)
            for key, intervals in calendar.items()
        }

    def add_calendar_interval(self, calendar, resource_key, start_time, end_time):
        """向日历中插入一个时间区间。

        使用二分查找定位插入位置，并检查与前后区间的交叠。
        时间区间必须不重叠 (非重叠约束)。
        """
        start_time = float(start_time)
        end_time = float(end_time)

        if end_time <= start_time:
            return

        intervals = calendar.setdefault(resource_key, [])
        insert_key = (start_time, end_time)
        insert_pos = bisect_right(intervals, insert_key)

        # 检查与前一个区间是否交叠
        if insert_pos > 0:
            prev_start, prev_end = intervals[insert_pos - 1]
            if start_time < prev_end:
                raise ValueError(
                    f"Resource {resource_key} interval "
                    f"({start_time}, {end_time}) overlaps ({prev_start}, {prev_end})"
                )

        # 检查与后一个区间是否交叠
        if insert_pos < len(intervals):
            next_start, next_end = intervals[insert_pos]
            if end_time > next_start:
                raise ValueError(
                    f"Resource {resource_key} interval "
                    f"({start_time}, {end_time}) overlaps ({next_start}, {next_end})"
                )

        intervals.insert(insert_pos, insert_key)

    def remove_calendar_interval(self, calendar, resource_key, start_time, end_time):
        """从日历中精确移除一个时间区间。

        使用 bisect_left 定位，要求区间精确匹配，否则抛出异常。
        """
        intervals = calendar.get(resource_key)
        if not intervals:
            raise ValueError(f"Resource {resource_key} interval does not exist")

        target = (float(start_time), float(end_time))
        remove_pos = bisect_left(intervals, target)

        if remove_pos >= len(intervals) or intervals[remove_pos] != target:
            raise ValueError(
                f"Resource {resource_key} interval ({start_time}, {end_time}) does not exist"
            )

        intervals.pop(remove_pos)
        if not intervals:
            calendar.pop(resource_key, None)

    def rollback_calendar_intervals(self, calendar, added_intervals):
        """回滚多个已添加的区间 (按添加顺序逆序移除)。

        用于仿真过程中撤销临时插入的 chamber calendar 区间。
        """
        for resource_key, start_time, end_time in reversed(added_intervals):
            self.remove_calendar_interval(calendar, resource_key, start_time, end_time)

    # ---- 槽位查找 ----

    def find_earliest_slot(self, busy_intervals, earliest_start, duration):
        """在忙区间列表中查找最早可用槽位。

        算法: 从 earliest_start 开始，遍历忙区间，跳过被占用的时间段，
        直到找到一个长度为 duration 的空闲段。

        Args:
            busy_intervals: [(start, end), ...] 已占用的区间列表 (需有序)。
            earliest_start: 最早允许的开始时间。
            duration: 所需时间长度。

        Returns:
            最早可行的开始时间。
        """
        start = float(earliest_start)
        duration = float(duration)

        if duration < 0.0:
            raise ValueError("duration must be non-negative")

        # 二分定位到 earliest_start 附近的区间
        start_index = 0
        if busy_intervals:
            start_index = max(
                0,
                bisect_right(busy_intervals, (start, float("inf"))) - 1,
            )

        # 线性扫描: 跳过与当前 start 交叠的区间
        for busy_start, busy_end in busy_intervals[start_index:]:
            if busy_end <= start:
                continue
            if start + duration <= busy_start:
                return start
            start = max(start, busy_end)

        return start

    # ---- 验证与统计 ----

    def validate_no_interval_overlap(self, calendar, calendar_name):
        """验证日历中所有资源均无区间交叠。

        Args:
            calendar: 待验证的日历。
            calendar_name: 日历名称 (用于错误消息)。
        """
        for resource_key, intervals in calendar.items():
            intervals = sorted(intervals)
            for index in range(len(intervals) - 1):
                current_start, current_end = intervals[index]
                next_start, next_end = intervals[index + 1]
                if current_end > next_start:
                    raise ValueError(
                        f"{calendar_name} {resource_key} overlaps: "
                        f"({current_start}, {current_end}) and ({next_start}, {next_end})"
                    )

    def calendar_busy_time(self, calendar, resource_key, current_time=0.0):
        """计算某资源从 current_time 之后的总忙时。

        Args:
            calendar: 资源日历。
            resource_key: 资源标识。
            current_time: 只统计该时间之后的区间部分。

        Returns:
            总忙时 (浮点数)。
        """
        busy_time = 0.0

        for start_time, end_time in calendar.get(resource_key, []):
            if end_time <= current_time:
                continue
            busy_time += max(0.0, float(end_time) - max(float(start_time), current_time))

        return float(busy_time)

    # ---- 工艺时间估算 ----

    def estimate_plan_total_process_time(self, steps, wafer_count):
        """估算晶圆批次在给定工艺步骤集上的总加工时间。

        取每步中最短处理时间 (最快资源)，加总后乘以晶圆数。
        这是一个理想化下界估计，用于启发式打分。
        """
        total = 0.0

        for stage in steps:
            resources = np.asarray(stage, dtype=float)
            total += float(np.min(resources[:, 2]))

        return total * int(wafer_count)

    def _qtime_limits_for(self, lot, machine, ppid):
        """返回某 (lot, machine, ppid) 的 Q-time 约束 [(from_stage, to_stage, limit), ...]。

        性能 (profile §1.6.1)：`estimate_qtime_risk` 被候选池里每个候选每步调用一次，
        而 `q_time_limits` 在压力实例下有数千条 (≈ lots×machines×ppids×stage对)，每个
        (lot,machine,ppid) 只匹配其中极少数。原实现每次线性扫描整张表 → O(总约束数)，
        是实测训练瓶颈 (≈70% 墙钟)。此处按 (lot,machine,ppid) 建一次倒排索引，使每次
        查询只触及自己的约束。

        索引惰性构建并以源 dict 的身份做失效判定：实例构造后部分 builder 会整体重新
        赋值 `encoder.q_time_limits = {...}` (见 problem_instances.py)，故用 `is` 比对源
        对象，源被替换时自动重建。约束在 episode 内不就地修改，索引无需更细的失效。
        """
        src = self.q_time_limits
        index = getattr(self, "_qtime_limits_index", None)
        if index is None or getattr(self, "_qtime_limits_index_src", None) is not src:
            index = {}
            for key, limit in src.items():
                k_lot, k_machine, k_ppid, from_stage, to_stage = map(int, key)
                index.setdefault((k_lot, k_machine, k_ppid), []).append(
                    (from_stage, to_stage, float(limit))
                )
            self._qtime_limits_index = index
            self._qtime_limits_index_src = src
        return index.get((int(lot), int(machine), int(ppid)), ())

    def estimate_qtime_risk(self, lot, machine, ppid, steps):
        """估算 Q-time 风险 — 中间阶段累计时间超出 Q-time 限制的总量。

        对每对 (from_stage, to_stage) 有 Q-time 约束的 stage 对，计算
        中间阶段的最短处理时间之和，与 Q-time 限制作差，累加超出部分。

        Returns:
            Q-time 风险值 (非负浮点数)。
        """
        risk = 0.0
        n_steps = len(steps)

        for from_stage, to_stage, q_time_limit in self._qtime_limits_for(lot, machine, ppid):
            if from_stage < 1 or to_stage > n_steps:
                continue

            intermediate_time = 0.0
            for stage_id in range(from_stage, to_stage - 1):
                resources = np.asarray(steps[stage_id], dtype=float)
                intermediate_time += float(np.min(resources[:, 2]))

            risk += max(0.0, intermediate_time - q_time_limit)

        return float(risk)