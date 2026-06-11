# Chain-Joint Mask Cost Optimizations (order ④→①→③)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** 把 chain_joint 默认口径下 VC-MCTS 探针 3h+/seed 的成本拉回可实验量级，按后果隔离度从低风险到高风险落地三个优化（②"1+K 重定时"明确不做——它动机制 1 的数学身份）。

**Architecture:** ④ chain mask 走免拷贝轻量 dry-run（语义零变化）；① VC-MCTS rollout clone 上可选降级 mask 口径（`VCMCTSConfig.rollout_qtime_mask_mode`，默认 None=不变）；③ 候选池两段式预筛（aggregate 粗筛 + 仅对 TopK+裕量 跑 chain，env 级开关默认关）。

**Tech Stack:** Python/numpy, pytest, 现有 `rl_environment.py` / `vc_mcts_planner.py` / `scripts/probes/vc_mcts_probe.py`。

---

### Task ④: chain mask 免拷贝 dry-run

- Files: `rl_environment.py`（新 `_chain_mask_wafer_schedule()`，两个 chain mask 改用之；`_simulate_action` 的 wafer 行组装抽成共享 helper）；Test: `tests/test_qtime_chain_mask_rng.py` 追加。
- [x] RED: 等价性测试（同种子 rng 下轻量路径 wafer_schedule == `dry_run_action().wafer_schedule`）+ 非破坏性测试（mask 调用前后真实日历不变）。
- [x] GREEN: 实现 helper，切换两个 chain mask；既有 rng/mask 行为测试守护。
- [x] 全套测试 + commit。

### Task ①: rollout 内降级 mask 口径

- Files: `vc_mcts_planner.py`（config 字段 + `_evaluate_action_once` clone 后设置）；`scripts/probes/vc_mcts_probe.py`（`--rollout-qtime-mask-mode` 透传）；Test: `tests/test_vc_mcts_mechanisms.py` 追加。
- [x] RED: 测试 clone env 拿到降级口径、真实 env 口径不变；默认 None 行为不变。
- [x] GREEN: 实现 + CLI 透传 + commit。

### Task ③: 两段式预筛（aggregate 粗筛 → 仅前 K+M 跑 chain）

- Files: `rl_environment.py`（env 开关 `qtime_mask_prescreen` 默认 False + `qtime_prescreen_margin`）；Test: 新增等价性测试（M 足够大时两段式池 == 全量池）+ doomed 防死锁保持。
- [x] RED→GREEN→commit，依据 build_candidate_pool 实际结构实现。

### 验收

- [x] FABenv 全套 + 根结构测试通过。
- [x] chain_joint 口径下轻预算探针实测提速并记录数字。
