"""Phase 2 SAS 观察编码器 — 将候选池和全局状态编码为结构化观察张量。

Phase2Observation: 标准化的观察数据类
Phase2ObservationEncoder: 负责提取全局特征和候选特征，支持 numpy 字典导出和批次化

全局特征 (9 维):
  0. current_time           当前仿真时间
  1. completed_ratio        已完成 Lot 占比
  2. remaining_ratio        剩余 Lot 占比
  3. machine_id_norm        归一化机台编号
  4. machine_busy_time      该机台累计忙时
  5. valid_action_count_norm 归一化有效动作数
  6. score_mean             有效候选的 score 均值
  7. waiting_time_max       有效候选的最大等待时间
  8. due_slack_min          有效候选的最小交货松弛

可选前瞻扩展 (lookahead=True 时为 13 维，报告 §2.1):
  9.  upcoming_count_norm    前瞻窗内即将到达 Lot 数 / num_lots
  10. lookahead_max_priority 窗内即将到达 Lot 最高 priority
  11. lookahead_min_qtime    窗内即将到达 Lot 最小剩余 qtime
  12. lookahead_earliest_eta 窗内即将到达 Lot 最早 arrival

候选特征 (18 维，由 ResourceCalendarEnv._candidate_features 提供):
  特征列表与 ResourceCalendarEnv.feature_names 保持一致，共 18 个字段。

注意（Phase 1 完善版）：
  特征 [2] score 的计算公式已更新为报告 Section 4.1 的评分：
    due_urgency + 1/qtime_slack + waiting_time - 0.001*proc_time - 0.001*qtime_risk
  priority 已移出评分，改由 priority_filter 在候选池生成阶段处理。
  特征维度仍为 18，与 ResourceCalendarEnv.feature_names 保持一致。
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Phase2Observation:
    """Phase 2 标准化观察 — 策略网络的输入结构。

    Attributes:
        machine_id: 当前决策机台编号。
        current_time: 仿真时钟。
        candidate_features: (pool_size, 18) 候选特征矩阵。
        candidate_mask: (pool_size,) bool 有效动作掩码。
        global_features: (9,) 全局上下文特征。
        action_indices: (pool_size,) 动作索引数组 (0..pool_size-1)。
        valid_action_count: 有效动作数量。
    """
    machine_id: int
    current_time: float
    candidate_features: np.ndarray
    candidate_mask: np.ndarray
    global_features: np.ndarray
    action_indices: np.ndarray
    valid_action_count: int


class Phase2ObservationEncoder:
    """将候选池编码为 Phase2Observation。

    支持:
      - encode(): 从 (machine, pool, env) 构建完整观察
      - build_global_features(): 提取 9 维全局特征
      - to_numpy_dict(): 导出为 numpy 字典格式
      - batch_observations(): 将多个观察批次化为训练数据
    """

    def __init__(self, normalize=True, lookahead=False):
        self.normalize = bool(normalize)
        self.lookahead = bool(lookahead)

    def encode(self, machine, pool, env):
        """从候选池编码为 Phase2Observation。

        candidate_features 为 (pool_size, 18) 矩阵，列顺序由
        ResourceCalendarEnv.feature_names 定义。其中：
          - 列 [2] score：对应报告 Section 4.1 的新评分公式
              due_urgency + 1/qtime_slack + waiting_time
              - 0.001*proc_time - 0.001*qtime_risk
            priority 已移出评分，由候选池生成阶段的 priority_filter 处理。
        """
        candidate_features = np.asarray(pool.features, dtype=np.float32)
        candidate_mask = np.asarray(pool.action_mask, dtype=bool)
        action_indices = np.arange(len(pool.actions), dtype=np.int64)
        if self.lookahead:
            global_features = self.build_global_features_v2(machine, pool, env).astype(np.float32)
        else:
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
        """构建 9 维全局上下文特征向量。

        特征计算:
          0. current_time: 原始仿真时间
          1. completed_ratio: completed / total_lots
          2. remaining_ratio:  remaining / total_lots
          3. machine_id_norm:  machine_id / total_machines
          4. machine_busy_time: 该机台 lot_schedule 的累计 (end-start)
          5. valid_action_count_norm: valid_count / pool_size
          6. score_mean: 有效候选 score 均值
          7. waiting_time_max: 有效候选最大等待时间
          8. due_slack_min: 有效候选最小松弛 (最紧急)
        """
        num_lots = max(int(env.encoder.num_lots), 1)
        num_machines = max(int(env.encoder.num_machines), 1)
        completed_ratio = len(env.completed_lots) / num_lots
        remaining_ratio = len(env.remaining_lots) / num_lots
        machine_id_norm = int(machine) / num_machines
        valid_count = int(np.sum(pool.action_mask))
        valid_action_count_norm = valid_count / max(len(pool.actions), 1)

        # 有效候选的统计特征
        valid_features = np.asarray(pool.features[pool.action_mask], dtype=float)
        if valid_features.size == 0:
            score_mean = 0.0
            waiting_time_max = 0.0
            due_slack_min = 0.0
        else:
            score_mean = float(np.mean(valid_features[:, env.feature_names.index("score")]))
            waiting_time_max = float(np.max(valid_features[:, env.feature_names.index("waiting_time")]))
            due_slack_min = float(np.min(valid_features[:, env.feature_names.index("due_slack")]))

        # 机台忙时
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

    def build_global_features_v2(self, machine, pool, env):
        """构建 13 维全局特征 (9 维基础 + 4 维前瞻摘要，报告 §2.1)。

        前 9 维与 build_global_features 完全一致；后 4 维取自
        env.lookahead_summary():
          9.  upcoming_count_norm:   upcoming_count / num_lots
          10. lookahead_max_priority: 窗内即将到达 Lot 最高 priority (无则 0)
          11. lookahead_min_qtime:    最小剩余 qtime (无则 0)
          12. lookahead_earliest_eta: 最早 arrival (无则 0)
        """
        base = self.build_global_features(machine, pool, env)
        summary = env.lookahead_summary()
        num_lots = max(int(env.encoder.num_lots), 1)
        upcoming_count_norm = float(summary["upcoming_count"]) / num_lots
        lookahead = np.asarray(
            [
                upcoming_count_norm,
                float(summary["max_priority"]),
                float(summary["min_remaining_qtime"]),
                float(summary["earliest_eta"]),
            ],
            dtype=float,
        )
        return np.concatenate([base, lookahead])

    def to_numpy_dict(self, observation):
        """将 Phase2Observation 导出为 numpy 字典。"""
        return {
            "candidate_features": observation.candidate_features,
            "candidate_mask": observation.candidate_mask,
            "global_features": observation.global_features,
            "action_indices": observation.action_indices,
            "valid_action_count": observation.valid_action_count,
        }

    def batch_observations(self, observations):
        """将多个 Phase2Observation 堆叠为批次张量字典。"""
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