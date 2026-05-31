"""调度状态数据类 — 维护滚动调度过程中的资源日历和可用时间。

该模块被 FAB_RL/FABenv (RL 版本) 和根目录遗留代码共用，
二者各自持有独立的副本。
"""

from dataclasses import dataclass, field


@dataclass
class ScheduleState:
    """滚动调度中的资源状态。

    Attributes:
        machine_available_time: {machine_id: 最早可用时间} 映射。
        chamber_available_time: {(machine, chamber, side): 最早可用时间} 映射。
        machine_calendar: {machine_id: [(start, end), ...]} 机台占用区间列表。
        chamber_calendar: {(machine, chamber, side): [(start, end), ...]} 腔体占用区间。
        current_time: 当前仿真时钟 (仅 RL 版本使用)。
        completed_lots: 已完成加工的 Lot 集合 (仅 RL 版本使用)。
        commit_log: 提交历史日志列表 (仅 RL 版本使用)。
        planning_window: 滚动窗口引用 (仅遗留代码使用)。
        schedules: {"lot_schedule": ..., "wafer_schedule": ...} 调度结果快照 (仅 RL 版本使用)。
    """
    machine_available_time: dict = field(default_factory=dict)
    chamber_available_time: dict = field(default_factory=dict)
    machine_calendar: dict = field(default_factory=dict)
    chamber_calendar: dict = field(default_factory=dict)
    current_time: float = 0.0
    completed_lots: set = field(default_factory=set)
    commit_log: list = field(default_factory=list)
    planning_window: object = None
    schedules: dict = field(default_factory=dict)