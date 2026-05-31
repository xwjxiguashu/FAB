# 下层估时器结果缓存 (报告 §1.5 开销警示)

> 解决报告 §1.5 点名的训练瓶颈：`qtime_safe_mask` 与 `is_doomed` 每个候选每步都调
> `estimate(n_mc)` 跑蒙特卡洛，无缓存 → pressure(50 lots) 候选池构建 ~0.49s/次，
> 单 episode 数分钟，训练不实用。

**状态: 已完成 (2026-05-31)。** 6 个缓存测试全绿，105 项广义回归无破坏。

## 正确性依据
`estimate()` 的完成时间分布只取决于 `(lot, machine, ppid, n_mc)` 等静态 encoder 数据
（**计算中不读 state**），`start_offset` 仅在返回时加到 `mu_finish`。故：
- 缓存 base 结果 (offset=0)，键 `(lot, machine, ppid, n_mc)`。
- 每次调用用 `_with_start_offset()` 重施 offset，返回新 dict，**绝不修改缓存的 base**。
- base 与时间/状态无关 → 整 episode 有效，仅 `reset()` 清空（区别于 `_doomed_cache`
  在 `advance_time` 清空）。

## 改动
- [x] `lower_layer_estimator.py`: `estimate(..., cache=None)` + 顶部命中早返回 + 末尾缓存 base；
      新增 `_with_start_offset()` 辅助。
- [x] `rl_environment.py`: `__init__`/`reset` 维护 `self._estimate_cache`；
      `qtime_safe_mask`(n_mc=20) 与 `is_doomed`(n_mc=10) 传入 cache。
- [x] 测试 `tests/test_phase2_estimate_cache.py` (6 个)：命中一致、offset 重施不污染缓存、
      投毒缓存验证确实走缓存、无 cache 路径不变、env 拥有并在 reset 清空。

## 实测效果
- pressure(50) 候选池构建：**0.49s → 0.086s** (~5.7×)。
- 完整规则 episode：50 步全部完成，约 138s（此前难以完成）。

## 已知残留 (本次不处理)
- 单 episode 仍约 2.8s/步，瓶颈已转移到 `commit`/wafer 级仿真（随排程规模增长），
  与 MC 估时无关，属另一项优化。
- 若 `estimate` 将来改为读取 `state`，缓存失效点须同步移到 `advance_time`/`commit`。
