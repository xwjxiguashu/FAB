# 杠杆 B: dry-run/commit 按子批排程 (报告 §1.5 批处理建模 + 提速)

> 修一个与报告不符、且污染 Q-time 判断的建模 bug：下层估时器 (喂 qtime mask) 用
> ⌈N/side_capacity⌉ 个子批，但 dry-run/commit 却把 N 片 wafer **逐片串行**排。
> 二者不一致 → mask 拿"算错"的完成时间判硬约束。B 让 dry-run/commit 与估时器、
> 与报告"批处理机一次成批、同进同出"一致：按子批排，子批内 wafer 共享区间。

**状态: 已完成 (2026-05-31)。** 全套测试绿 (149 passed)；pressure 单 episode 132.9s → **43.1s (~3.1×)**。
结果按预期变化 (batching → makespan 变短)：pressure EDD tardy 53.6→0.0、util 0.946→0.867。
仅 1 个测试需修 (validate_schedule 的冲突重建去重)，其余断言结构性、不受影响。

## 设计
- 调度单元 = 子批 `compute_sub_batches(wafer_count, side_capacity)` (如 10/4 → [4,4,2])。
  side_capacity 未设时默认 = wafer_count (1 批)，与估时器一致。
- 每个子批在每个 stage 选一个 (chamber,side) 实例，加 **1 个区间** 到 chamber_calendar；
  该子批的所有 wafer **共享** 该区间 (相同 chamber/side/start/end)。
- wafer_schedule 仍是 wafer_count × n_stages 行 (schema 不变)，只是同批 wafer 时刻相同。
- 噪声 (commit): 每 (子批, stage) 采样一次 (整批一个实际时间)。

## 受影响处
- `_simulate_action` (commit): 逐 wafer → 逐子批。
- `_dry_run_candidate`: 逐 wafer → 逐子批 (无噪声)。
- `validate_final_schedule_completeness`: 从 wafer 行重建 chamber_calendar 时，
  同批 wafer 产生重复区间 → 需 **去重** 再做 no-overlap 校验。
- `compute_q_time_violation`: 读 per-wafer start/end，同批共享 → 无需改。

## 预期效果
- 速度: 调度单元 10→3，dry-run/commit 各 ~3×。
- 结果: makespan 普遍变短 (3 批 vs 10 串) → 利用率↑、拖期/qtime↓。所有现有数字变化。
- 正确性: dry-run/commit 与估时器自洽，qtime mask 拿到的完成时间与实际一致。

## 测试 (TDD)
- [ ] 同子批 wafer 在每个 stage 共享 (chamber,side,start,end)。
- [ ] wafer_schedule 行数 = wafer_count × n_stages (不变)，全 wafer 覆盖。
- [ ] 每 stage 的 distinct chamber 区间数 ≤ n_sub_batches (按批不按片)。
- [ ] validate_final_schedule_completeness 通过 (去重后)。
- [ ] 全 small / pressure episode 完成且校验通过。
- [ ] 现有断言具体时刻/利用率的测试 → 更新为新的正确值 (逐一核对，不橡皮图章)。
