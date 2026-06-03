"""下层启发式估时器 — 工件内部批处理的完成时间分布估算。

本模块实现报告 Section 1.5 描述的下层固定规则估时器:
  - 组批: 满批优先，N片 wafer → ⌈N/side容量⌉ 个子批
  - 流水排程 (list scheduling): 逐子批 × 逐阶段，选最早可用实例
  - 蒙特卡洛: 对各阶段加工时间采样，得到 (μ_finish, σ_finish)

核心输出:
  estimate(lot, machine, ppid, encoder, state, n_mc) →
    {mu_finish, sigma_finish, bottleneck_stage, per_instance_occupancy}

设计原则 (来自报告):
  - 下层是"固定规则"——算法不学习、不随机搜索
  - "分布输出"来自输入加工时间的随机性 (纯工艺噪声，无偏独立)
  - 衡量标准: "估得准"而非"全局最优 makespan"
  - E[max] > max(E[·])，用蒙特卡洛规避均值路径的系统性低估
"""

import math

import numpy as np


# =============================================================================
# 子批计算
# =============================================================================


def compute_sub_batches(n_wafers, side_capacity):
    """将 N 片 wafer 分成 ⌈N/side_capacity⌉ 个子批。

    满批优先: 前面的子批尽量装满，最后一批装余下的。

    Args:
        n_wafers:      总 wafer 数 (正整数)。
        side_capacity: 每个 side 的最大容量 (正整数)。

    Returns:
        list[int] — 每个子批的 wafer 数量。
    """
    n_wafers = int(n_wafers)
    side_capacity = int(side_capacity)
    if n_wafers <= 0:
        raise ValueError("n_wafers must be positive")
    if side_capacity <= 0:
        raise ValueError("side_capacity must be positive")

    n_batches = math.ceil(n_wafers / side_capacity)
    batches = []
    remaining = n_wafers
    for _ in range(n_batches):
        b = min(remaining, side_capacity)
        batches.append(b)
        remaining -= b
    return batches


# =============================================================================
# 单次流水排程 (list scheduling with batching)
# =============================================================================


def _run_list_schedule(sub_batches, stage_times, n_stages, instance_counts):
    """执行一次确定性流水排程，返回 makespan 和各阶段完成时间。

    调度规则 (报告 Section 1.5):
      for 每个子批 b (FIFO 顺序):
        for 每个阶段 s (按 ppid 顺序):
          ready   = b 在上一阶段的完成时间
          free    = min over 阶段s的实例(实例空闲时间)
          start   = max(ready, free)
          end     = start + 批处理时间(s)
      makespan = max_b end[b, 末阶段]

    Args:
        sub_batches:    子批列表 (本函数不使用数量，仅计数)。
        stage_times:    (n_batches, n_stages) 数组，各子批在各阶段的实际加工时间。
        n_stages:       阶段数。
        instance_counts: list[int] — 每个阶段的实例 (chamber/side) 数量。

    Returns:
        (makespan, stage_end_times_per_batch) — makespan 为所有子批末阶段的最晚结束时间。
    """
    n_batches = len(sub_batches)
    # 各阶段每个实例的空闲时间 (初始为 0)
    instance_free = [
        [0.0] * instance_counts[s]
        for s in range(n_stages)
    ]
    # 每个子批在当前阶段的就绪时间
    batch_ready = [0.0] * n_batches
    stage_end = np.zeros((n_batches, n_stages), dtype=float)

    for b in range(n_batches):
        ready_time = batch_ready[b]
        for s in range(n_stages):
            # 找最早空闲的实例
            best_instance = int(np.argmin(instance_free[s]))
            earliest_free = instance_free[s][best_instance]
            start = max(ready_time, earliest_free)
            pt = float(stage_times[b, s])
            end = start + pt
            instance_free[s][best_instance] = end
            stage_end[b, s] = end
            ready_time = end  # 子批流水: 下阶段 ready = 本阶段结束

    makespan = float(np.max(stage_end[:, -1]))
    return makespan, stage_end


def schedule_deterministic(
    sub_batches,
    stage_times,
    stage_resource_options,
    machine,
    instance_free_init,
    lot_release_time=0.0,
):
    """Run deterministic list scheduling on absolute resource keys.

    This is the shared lower-layer scheduling core. ``estimate`` calls it with
    empty free times and ``schedule_on_calendar`` calls it with free times read
    from the committed calendar.

    Returns:
        (lot_start, lot_end, batch_intervals), where each interval is
        (batch_index, stage_index_1based, resource_key, start, end).
    """
    n_batches = len(sub_batches)
    stage_times = np.asarray(stage_times, dtype=float)
    if n_batches == 0:
        return float(lot_release_time), float(lot_release_time), []
    if stage_times.ndim != 2 or stage_times.shape[0] != n_batches:
        raise ValueError("stage_times must have shape (n_batches, n_stages)")

    n_stages = stage_times.shape[1]
    if len(stage_resource_options) != n_stages:
        raise ValueError("stage_resource_options length must equal n_stages")

    machine = int(machine)
    free = dict(instance_free_init or {})
    intervals = []

    for b in range(n_batches):
        ready = float(lot_release_time)
        for s in range(n_stages):
            options = stage_resource_options[s]
            if not options:
                raise ValueError(f"stage {s + 1} has no resource options")

            best_key = None
            best_start = None
            for chamber, side, _base_pt in options:
                key = (machine, int(chamber), int(side))
                cand_start = max(ready, float(free.get(key, 0.0)))
                if best_start is None or cand_start < best_start:
                    best_start = cand_start
                    best_key = key

            start = float(best_start)
            end = start + float(stage_times[b, s])
            free[best_key] = end
            ready = end
            intervals.append((b, s + 1, best_key, start, end))

    lot_start = min(start for _b, _s, _key, start, _end in intervals)
    lot_end = max(end for _b, _s, _key, _start, end in intervals)
    return float(lot_start), float(lot_end), intervals


# =============================================================================
# 蒙特卡洛采样 → (μ_finish, σ_finish)
# =============================================================================


def monte_carlo_makespan(
    sub_batches,
    stage_mu,
    stage_sigma,
    instance_counts,
    n_mc=50,
    rng=None,
):
    """通过蒙特卡洛采样估算 makespan 的均值和标准差。

    每次采样: 对各 (sub_batch, stage) 独立采样实际加工时间，
    运行一次 list_schedule，记录 makespan。
    n_mc 次后取 mean/std。

    噪声模型 (报告 2.4.1):
      p_actual(b,s) = μ(b,s) + ε, ε ~ N(0, σ²), 各 (b,s) 独立

    Args:
        sub_batches:     子批列表 (len = n_batches)。
        stage_mu:        list[float] — 各阶段均值加工时间 (n_stages,)。
        stage_sigma:     list[float] — 各阶段加工时间标准差 (n_stages,)。
        instance_counts: list[int] — 各阶段实例数 (n_stages,)。
        n_mc:            蒙特卡洛采样次数 (默认 50)。
        rng:             numpy Generator (None 则用 default_rng)。

    Returns:
        (mu_finish, sigma_finish) — makespan 的期望和标准差。
    """
    if rng is None:
        rng = np.random.default_rng()

    n_batches = len(sub_batches)
    n_stages = len(stage_mu)
    makespans = np.empty(n_mc, dtype=float)

    stage_mu_arr = np.asarray(stage_mu, dtype=float)
    stage_sigma_arr = np.asarray(stage_sigma, dtype=float)

    for sample_idx in range(n_mc):
        # shape (n_batches, n_stages): 各子批各阶段独立采样
        if np.all(stage_sigma_arr == 0.0):
            stage_times = np.tile(stage_mu_arr, (n_batches, 1))
        else:
            noise = rng.standard_normal((n_batches, n_stages))
            stage_times = stage_mu_arr[None, :] + noise * stage_sigma_arr[None, :]
            # 加工时间不能为负 (clip 到最小值 1e-6)
            stage_times = np.maximum(stage_times, 1e-6)

        makespan, _ = _run_list_schedule(sub_batches, stage_times, n_stages, instance_counts)
        makespans[sample_idx] = makespan

    mu_finish = float(np.mean(makespans))
    sigma_finish = float(np.std(makespans, ddof=0))
    return mu_finish, sigma_finish


# =============================================================================
# 主接口: estimate()
# =============================================================================


def _with_start_offset(base, start_offset):
    """返回 base 结果的浅拷贝，仅把 start_offset 加到 mu_finish 上 (报告 §1.5)。

    base 是 start_offset=0 的结果；start_offset 是确定性偏移，只平移 mu_finish，
    σ 与其余字段不变。返回新 dict，绝不修改 base (避免命中时重复累加偏移污染缓存)。
    """
    result = dict(base)
    result["mu_finish"] = float(base["mu_finish"]) + float(start_offset)
    return result


def estimate(lot, machine, ppid, encoder, state, n_mc=50, rng=None, start_offset=0.0, cache=None):
    """估算工件 (lot, machine, ppid) 的完成时间分布。

    报告 Section 1.5 的接口:
      estimate(O, machine m, ppid p) →
        (μ_finish, σ_finish)         完成时间均值与标准差
        per_instance_occupancy       各 chamber 实例占用区间 (按 μ 登记)
        bottleneck_stage             瓶颈阶段 (有效吞吐最低的阶段)

    算法:
      1. 获取 (lot, machine, ppid) 的工艺步骤
      2. 从每阶段的资源列表中提取均值加工时间 (stage_mu) 和标准差 (stage_sigma)
      3. 计算子批 (wafer 数 / side 容量)
      4. 各阶段实例数 = 当前 machine 上该阶段可用 chamber/side 数
      5. 蒙特卡洛 → (μ_finish, σ_finish)
      6. 用均值路径算 per_instance_occupancy (冲突检测用)
      7. 识别瓶颈阶段

    Args:
        lot:     Lot 编号 (1-indexed)。
        machine: 机台编号 (1-indexed)。
        ppid:    PPID 编号。
        encoder: Phase1CalendarProblem 实例 (含 problem + calendar 操作)。
        state:   当前 ScheduleState (用于读取日历空闲时间)。
        n_mc:    蒙特卡洛采样次数 (默认 50)。
        rng:     numpy Generator (None 则用 default_rng)。
        start_offset: 确定性开工时刻偏移 (默认 0.0)。mu_finish 整体加上该值，
                      σ 不变 (offset 是确定偏移)。start_offset=0 时行为与旧版完全一致。
        cache:   可选 dict (报告 §1.5 开销警示)。下层是固定规则、makespan 分布只取决于
                 (lot, machine, ppid, n_mc) 等静态输入 (本函数不读 state)，start_offset 仅在
                 返回时施加，故可缓存 base(offset=0) 结果并对同一键复用，避免每个候选每步
                 重跑蒙特卡洛。键 = (lot, machine, ppid, n_mc)。命中时不重算、只重施 offset。

    Returns:
        dict with keys:
          mu_finish (float):         完成时刻均值 (相对于 start_offset；
                                     start_offset=0 时即相对 t=0)
          sigma_finish (float):      完成时间标准差
          bottleneck_stage (int):    瓶颈阶段编号 (1-indexed)
          per_instance_occupancy (list): [(resource_key, mu_start, mu_end), ...]
          stage_mu (list[float]):    各阶段均值加工时间
          stage_sigma (list[float]): 各阶段标准差
          n_batches (int):           子批数量
    """
    lot = int(lot)
    machine = int(machine)
    ppid = int(ppid)

    # ---- 缓存命中 (报告 §1.5)：直接复用 base 结果，仅重施 start_offset ----
    cache_key = (lot, machine, ppid, int(n_mc))
    if cache is not None and cache_key in cache:
        return _with_start_offset(cache[cache_key], start_offset)

    # ---- Step 1: 获取工艺步骤 ----
    steps = encoder.get_process_steps(lot, machine, ppid)
    n_stages = len(steps)
    if n_stages == 0:
        raise ValueError(f"No process steps for (lot={lot}, machine={machine}, ppid={ppid})")

    # ---- Step 2: 提取各阶段的加工时间均值和标准差 ----
    # 每阶段 steps[s] 是 (n_resources, 3) 数组，列为 [chamber, side, process_time]
    # 取每阶段所有资源中的最小加工时间均值 (即最优选择下的均值)
    stage_mu = []
    stage_sigma_list = []
    stage_resource_options = []  # 每阶段可用资源的列表

    # 获取进程时间标准差字典 (可能不存在)
    proc_sigma_dict = getattr(encoder, "process_time_sigma", {})

    # 机台允许的资源集合
    declared_resources = getattr(encoder, "machine_resources", {})
    allowed_resources = None
    if declared_resources:
        allowed_resources = {
            (int(chamber), int(side))
            for chamber, side in declared_resources.get(machine, [])
        }

    for s_idx, stage in enumerate(steps):
        stage_arr = np.asarray(stage, dtype=float)  # (n_res, 3): [chamber, side, pt]
        valid_rows = []
        for row in stage_arr:
            ch, sd, pt = int(row[0]), int(row[1]), float(row[2])
            if allowed_resources is not None and (ch, sd) not in allowed_resources:
                continue
            valid_rows.append((ch, sd, pt))

        if not valid_rows:
            # fallback: 使用全部资源
            valid_rows = [(int(r[0]), int(r[1]), float(r[2])) for r in stage_arr]

        # 取最短加工时间作为代表值 (选最快实例的均值)
        min_pt = min(pt for _, _, pt in valid_rows)

        # σ: 从 encoder.process_time_sigma 取；若无则默认 5% 的 μ
        sigma_key = (lot, machine, ppid)
        if sigma_key in proc_sigma_dict:
            sigma_vals = proc_sigma_dict[sigma_key]
            s_sigma = float(sigma_vals[s_idx]) if s_idx < len(sigma_vals) else 0.05 * min_pt
        else:
            s_sigma = 0.05 * min_pt  # 默认 5% 噪声

        stage_mu.append(min_pt)
        stage_sigma_list.append(s_sigma)
        stage_resource_options.append(valid_rows)

    # ---- Step 3: 子批计算 ----
    wafer_count = int(encoder.wafer_counts[lot])
    # side 容量: 取第一阶段的最大 side 数量作为代理 (批处理炉通常相同)
    # 若未定义，默认每批 = 全部 wafer (无需分批)
    side_capacity = getattr(encoder, "side_capacity", None)
    if side_capacity is None:
        # 从阶段资源推断: 同一 (machine, chamber, side) 分组的资源
        # 简单起见，默认 side_capacity = wafer_count (不分批)
        side_capacity = wafer_count
    side_capacity = max(1, int(side_capacity))
    sub_batches = compute_sub_batches(wafer_count, side_capacity)
    n_batches = len(sub_batches)

    # ---- Step 4: 各阶段实例数 ----
    # 实例数 = 可用 (chamber, side) 对的数量
    instance_counts = [len(opts) for opts in stage_resource_options]
    instance_counts = [max(1, c) for c in instance_counts]  # 至少 1

    # ---- Step 5: 蒙特卡洛 → (μ_finish, σ_finish) ----
    if rng is None:
        rng = np.random.default_rng()

    mu_finish, sigma_finish = monte_carlo_makespan(
        sub_batches,
        stage_mu,
        stage_sigma_list,
        instance_counts,
        n_mc=n_mc,
        rng=rng,
    )

    # ---- Step 6: 均值路径的 per_instance_occupancy (用于日历登记) ----
    # 用均值加工时间跑一次确定性排程，记录各资源区间
    mu_stage_times = np.tile(np.asarray(stage_mu), (n_batches, 1))
    _lot_start, _lot_end, occ_intervals = schedule_deterministic(
        sub_batches,
        mu_stage_times,
        stage_resource_options,
        machine,
        instance_free_init={},
        lot_release_time=0.0,
    )
    per_instance_occupancy = [
        (resource_key, start, end)
        for _b, _stage, resource_key, start, end in occ_intervals
    ]

    # ---- Step 7: 瓶颈阶段识别 ----
    # 瓶颈 = 有效吞吐最低的阶段: 实例数 × (1/μ_stage_time) 最小
    throughputs = [
        instance_counts[s] / max(stage_mu[s], 1e-9)
        for s in range(n_stages)
    ]
    bottleneck_stage = int(np.argmin(throughputs)) + 1  # 1-indexed

    # ---- base 结果 (start_offset=0)：缓存此结果，再按需平移 ----
    base = {
        "mu_finish": mu_finish,   # 相对开工时刻 (offset=0)
        "sigma_finish": sigma_finish,
        "bottleneck_stage": bottleneck_stage,
        "per_instance_occupancy": per_instance_occupancy,
        "stage_mu": stage_mu,
        "stage_sigma": stage_sigma_list,
        "n_batches": n_batches,
    }
    if cache is not None:
        cache[cache_key] = base
    # 确定性开工偏移: 将 makespan 平移为绝对完成时刻
    return _with_start_offset(base, start_offset)


# =============================================================================
# 快捷工具: 机会约束判断
# =============================================================================


def is_qtime_violated_probabilistically(
    mu_finish,
    sigma_finish,
    qtime_deadline,
    z_eps=2.05,
):
    """检查完成时间分布是否违反 Q-time 机会约束。

    报告 Section 2.4.3:
      机会约束: 屏蔽 deadline − μ_finish < z_ε · σ_finish 的动作
      等价于:   P(finish_time > deadline) > ε

    Args:
        mu_finish:      完成时间均值。
        sigma_finish:   完成时间标准差。
        qtime_deadline: Q-time 截止时刻 (绝对时间)。
        z_eps:          分位数 (如 ε=2% → z_ε≈2.05)。

    Returns:
        True  表示"该动作应被屏蔽" (违反机会约束)。
        False 表示"该动作安全" (违规概率 ≤ ε)。
    """
    margin = z_eps * float(sigma_finish)
    return float(qtime_deadline) - float(mu_finish) < margin


def qtime_violation_probability(mu_finish, sigma_finish, qtime_deadline):
    """计算实际违规概率 P(finish_time > deadline)。

    假设 finish_time ~ N(μ, σ²) (CLT 近似)。

    Returns:
        float — 违规概率 (0.0 到 1.0)。
    """
    from scipy.stats import norm
    if float(sigma_finish) < 1e-12:
        return 0.0 if float(mu_finish) <= float(qtime_deadline) else 1.0
    z = (float(qtime_deadline) - float(mu_finish)) / float(sigma_finish)
    return float(1.0 - norm.cdf(z))
