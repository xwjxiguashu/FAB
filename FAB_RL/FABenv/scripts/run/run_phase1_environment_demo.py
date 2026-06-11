"""Phase 1 环境演示脚本 — 在 50 Lot × 10 机台压力测试实例上运行启发式调度。

核心逻辑:
  1. build_pressure_test_encoder() → 构建大规模随机问题实例
  2. ResourceCalendarEnv → 创建 RL 环境
  3. run_demo_schedule() → 逐机台扫描，取每个候选池的 first-valid 动作
  4. 若无真实候选 → 推进时间到下一个 Lot 到达事件
  5. 导出 CSV (lot/wafer schedule) + PNG (Gantt 图) 到 pressure_outputs/

输出文件:
  - pressure_lot_schedule.csv:    (n_lots, 5) 批次级调度
  - pressure_wafer_schedule.csv:  (n_wafers*n_stages, 9) 晶圆级调度
  - pressure_summary.txt:         调度摘要 (目标值、输出路径)
  - pressure_lot_gantt.png:       批次甘特图
  - pressure_wafer_gantt.png:     晶圆级甘特图
"""

from pathlib import Path
import sys

FABENV_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = FABENV_ROOT / "scripts"
for path in (
    FABENV_ROOT,
    SCRIPT_ROOT / "run",
    SCRIPT_ROOT / "evaluation",
    SCRIPT_ROOT / "experiments",
    SCRIPT_ROOT / "probes",
):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无头渲染 (无需 GUI)

import numpy as np

from problem_instances import build_pressure_test_encoder, format_objectives
from rl_environment import ResourceCalendarEnv

# CSV 表头
LOT_SCHEDULE_HEADER = "lot,machine,ppid,start_time,end_time"
WAFER_SCHEDULE_HEADER = (
    "lot,wafer_id,machine,ppid,stage_id,chamber,side,start_time,end_time"
)


def first_real_action_index(pool):
    """返回候选池中第一个真实 (非 wait/padding) 有效动作的索引。"""
    for action_index, action in enumerate(pool.actions):
        if not bool(pool.action_mask[action_index]):
            continue
        if action.is_wait or action.is_padding or int(action.ppid) == 0:
            continue
        return action_index
    return None


def sort_lot_schedule(lot_schedule):
    """按 (start_time, lot) 排序 lot_schedule。"""
    if lot_schedule.size == 0:
        return lot_schedule.reshape((0, 5))
    order = np.lexsort((lot_schedule[:, 0], lot_schedule[:, 3]))
    return lot_schedule[order]


def sort_wafer_schedule(wafer_schedule):
    """按 (start_time, lot, wafer_id, stage_id) 排序 wafer_schedule。"""
    if wafer_schedule.size == 0:
        return wafer_schedule.reshape((0, 9))
    order = np.lexsort(
        (
            wafer_schedule[:, 4],  # stage_id
            wafer_schedule[:, 1],  # wafer_id
            wafer_schedule[:, 0],  # lot
            wafer_schedule[:, 7],  # start_time
        )
    )
    return wafer_schedule[order]


def build_demo_encoder():
    """构建压力测试问题实例。"""
    return build_pressure_test_encoder()


def run_demo_schedule(encoder=None, top_k=16, verbose=True):
    """在指定问题实例上运行启发式调度。

    算法: 贪心 First-Valid 规则
      1. 遍历所有机台，构建候选池
      2. 对每个有真实候选的机台，取第一个有效动作并提交
      3. 若本轮无提交 → 推进时间到下一个 Lot 到达事件
      4. 直至所有 Lot 完成

    Returns:
        (lot_schedule, wafer_schedule, objectives)
    """
    encoder = build_demo_encoder() if encoder is None else encoder
    env = ResourceCalendarEnv(encoder, current_time=0.0, top_k=top_k)

    while env.remaining_lots:
        committed = False

        # 逐机台扫描: 每个机台尝试派一个 Lot
        for machine in range(1, int(encoder.num_machines) + 1):
            pool = env.build_candidate_pool(machine)
            action_index = first_real_action_index(pool)
            if action_index is None:
                continue

            result = env.commit_action_index(machine, action_index, pool=pool)
            action = result.action
            lot_row = result.lot_schedule[0]
            if verbose:
                print(
                    "commit "
                    f"lot={action.lot} machine={action.machine} ppid={action.ppid} "
                    f"start={lot_row[3]:.3f} end={lot_row[4]:.3f}"
                )
            committed = True

        if committed:
            continue

        # 无真实候选 → 推进时间到下一个到达事件
        future_arrivals = [
            float(encoder.arrival_times[lot])
            for lot in env.remaining_lots
            if float(encoder.arrival_times[lot]) > env.current_time
        ]
        if not future_arrivals:
            raise RuntimeError("No real candidate actions and no future arrivals remain")
        env.advance_time(min(future_arrivals))

    lot_schedule = sort_lot_schedule(env.lot_schedule)
    wafer_schedule = sort_wafer_schedule(env.wafer_schedule)
    encoder.validate_final_schedule_completeness(lot_schedule, wafer_schedule)
    objectives = encoder.evaluate_objectives(lot_schedule, wafer_schedule)
    return lot_schedule, wafer_schedule, objectives


# =============================================================================
# 输出导出
# =============================================================================


def default_pressure_output_dir():
    """返回默认输出目录 (FABenv/artifacts/pressure_outputs/)。"""
    return Path(__file__).resolve().parents[2] / "artifacts" / "pressure_outputs"


def _lot_color_map(lots, cmap_name="tab20"):
    """为每个 Lot 分配颜色 (用于甘特图)。"""
    import matplotlib.pyplot as plt

    lots = [int(lot) for lot in lots]
    cmap = plt.get_cmap(cmap_name, max(len(lots), 1))
    return {
        lot: cmap(index)
        for index, lot in enumerate(lots)
    }


def _save_lot_gantt(lot_schedule, output_path):
    """绘制并保存批次级甘特图 (每个 Lot 在机台上的时间条)。"""
    import matplotlib.pyplot as plt

    lot_schedule = np.asarray(lot_schedule, dtype=float)
    lots = sorted(set(lot_schedule[:, 0].astype(int)))
    machines = sorted(set(lot_schedule[:, 1].astype(int)))
    y_map = {
        machine: index
        for index, machine in enumerate(machines)
    }
    colors = _lot_color_map(lots)

    fig, ax = plt.subplots(figsize=(16, 7))
    for row in lot_schedule:
        lot = int(row[0])
        machine = int(row[1])
        ppid = int(row[2])
        start_time = float(row[3])
        end_time = float(row[4])
        duration = end_time - start_time
        ax.barh(
            y_map[machine],
            duration,
            left=start_time,
            height=0.62,
            color=colors[lot],
            edgecolor="black",
            linewidth=0.45,
        )
        ax.text(
            start_time + duration / 2,
            y_map[machine],
            f"L{lot}\nP{ppid}",
            ha="center",
            va="center",
            fontsize=5.5,
        )

    ax.set_yticks(list(y_map.values()))
    ax.set_yticklabels([f"Machine {machine}" for machine in machines])
    ax.set_xlabel("Time")
    ax.set_ylabel("Machine")
    ax.set_title("Pressure Test Lot Gantt")
    ax.grid(True, axis="x", linestyle="--", alpha=0.45)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_wafer_gantt(wafer_schedule, output_path):
    """绘制并保存晶圆级甘特图 (每 Wafer 在腔体资源上的时间条)。"""
    import matplotlib.pyplot as plt

    wafer_schedule = np.asarray(wafer_schedule, dtype=float)
    lots = sorted(set(wafer_schedule[:, 0].astype(int)))
    resources = sorted({
        (int(row[2]), int(row[5]), int(row[6]))
        for row in wafer_schedule
    })
    y_map = {
        resource: index
        for index, resource in enumerate(resources)
    }
    colors = _lot_color_map(lots)

    fig_height = max(10, len(resources) * 0.22)
    fig, ax = plt.subplots(figsize=(22, fig_height))
    for row in wafer_schedule:
        lot = int(row[0])
        resource = (int(row[2]), int(row[5]), int(row[6]))
        start_time = float(row[7])
        end_time = float(row[8])
        ax.barh(
            y_map[resource],
            end_time - start_time,
            left=start_time,
            height=0.64,
            color=colors[lot],
            edgecolor="black",
            linewidth=0.25,
        )

    ax.set_yticks(list(y_map.values()))
    ax.set_yticklabels([
        f"M{machine}-C{chamber}-S{side}"
        for machine, chamber, side in resources
    ], fontsize=6)
    ax.set_xlabel("Time")
    ax.set_ylabel("Machine-Chamber-Side")
    ax.set_title("Pressure Test Wafer Gantt")
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def export_pressure_outputs(
    encoder,
    lot_schedule,
    wafer_schedule,
    objectives,
    output_dir=None,
):
    """导出调度结果: CSV + 摘要 + 甘特图。

    Returns:
        输出文件路径字典。
    """
    output_dir = default_pressure_output_dir() if output_dir is None else Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lot_schedule_csv = output_dir / "pressure_lot_schedule.csv"
    wafer_schedule_csv = output_dir / "pressure_wafer_schedule.csv"
    summary_txt = output_dir / "pressure_summary.txt"
    lot_gantt_png = output_dir / "pressure_lot_gantt.png"
    wafer_gantt_png = output_dir / "pressure_wafer_gantt.png"

    # CSV 导出
    np.savetxt(
        lot_schedule_csv,
        lot_schedule,
        delimiter=",",
        header=LOT_SCHEDULE_HEADER,
        comments="",
        fmt="%.6f",
    )
    np.savetxt(
        wafer_schedule_csv,
        wafer_schedule,
        delimiter=",",
        header=WAFER_SCHEDULE_HEADER,
        comments="",
        fmt="%.6f",
    )

    # 甘特图
    _save_lot_gantt(lot_schedule, lot_gantt_png)
    _save_wafer_gantt(wafer_schedule, wafer_gantt_png)

    # 摘要
    used_machines = sorted(set(lot_schedule[:, 1].astype(int).tolist()))
    summary = "\n".join([
        "Phase 1 pressure demo completed",
        f"lots={int(encoder.num_lots)}",
        f"wafers_per_lot={int(encoder.wafer_counts[1])}",
        f"total_wafers={sum(int(v) for v in encoder.wafer_counts.values())}",
        f"machines={int(encoder.num_machines)}",
        "ppids_per_lot_machine=5",
        f"lot_rows={len(lot_schedule)}",
        f"wafer_rows={len(wafer_schedule)}",
        f"used_machines={used_machines}",
        f"objectives={format_objectives(objectives)}",
        f"lot_gantt_png={lot_gantt_png}",
        f"wafer_gantt_png={wafer_gantt_png}",
        "",
    ])
    summary_txt.write_text(summary, encoding="utf-8")

    return {
        "output_dir": output_dir,
        "lot_schedule_csv": lot_schedule_csv,
        "wafer_schedule_csv": wafer_schedule_csv,
        "summary_txt": summary_txt,
        "lot_gantt_png": lot_gantt_png,
        "wafer_gantt_png": wafer_gantt_png,
    }


def main():
    """运行 Phase 1 压力测试演示并导出结果。"""
    encoder = build_demo_encoder()
    lot_schedule, wafer_schedule, objectives = run_demo_schedule(
        encoder=encoder,
        verbose=False,
    )
    output_paths = export_pressure_outputs(
        encoder,
        lot_schedule,
        wafer_schedule,
        objectives,
    )

    print("Phase 1 pressure demo completed", flush=True)
    print(f"lots={encoder.num_lots}", flush=True)
    print(f"wafers_per_lot={encoder.wafer_counts[1]}", flush=True)
    print(f"machines={encoder.num_machines}", flush=True)
    print("ppids_per_lot_machine=5", flush=True)
    print(f"lot_rows={len(lot_schedule)}", flush=True)
    print(f"wafer_rows={len(wafer_schedule)}", flush=True)
    print(f"objectives={format_objectives(objectives)}", flush=True)
    print(f"output_dir={output_paths['output_dir']}", flush=True)
    print(f"lot_schedule_csv={output_paths['lot_schedule_csv']}", flush=True)
    print(f"wafer_schedule_csv={output_paths['wafer_schedule_csv']}", flush=True)
    print(f"summary_txt={output_paths['summary_txt']}", flush=True)
    print(f"lot_gantt_png={output_paths['lot_gantt_png']}", flush=True)
    print(f"wafer_gantt_png={output_paths['wafer_gantt_png']}", flush=True)


if __name__ == "__main__":
    main()
