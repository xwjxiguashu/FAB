# 第一阶段 Review 严重问题修复

> REQUIRED SUB-SKILL: superpowers:subagent-driven-development

修复 code review 发现的 4 个严重问题。全部 TDD，不破坏现有测试。

## 环境
- 真实解释器: `D:\Anaconda\python.exe`（本机 python 是 stub）
- 命令开头 `Set-Location "d:\HuaweiMoveData\Users\XWJ\Desktop\FAB-main\FAB_RL\FABenv"`
- 测试前设 `$env:KMP_DUPLICATE_LIB_OK="TRUE"`

---

## Task R1: 接入 priority_filter (#1) + 修正 score 符号 (#9)

### #1 priority_filter 接入候选池流水线
报告 §3.1: `① qtime mask → ② priority filter → ③ TopK → ④ pad`。
当前 build_candidate_pool 只调了 ① qtime mask，漏了 ②。

在 rl_environment.py build_candidate_pool 中，qtime mask 之后、排序之前，插入 priority filter：
- env 增加可配置 `priority_filter_mode`（默认 "soft"）和 `priority_min_gap`（默认 0.0），作为 __init__ 参数
- soft 模式：priority_filter 返回原列表（行为不变）
- strict 模式：过滤掉低优先级 candidate
- priority_filter 当前签名吃 actions 列表，需要适配成能过滤 _Candidate 列表（或在调用处用 action 映射回 candidate）

实现建议：在 build_candidate_pool 里，qtime mask 后得到 real_candidates（_Candidate 列表）。调用：
```python
if self.priority_filter_mode == "strict" and real_candidates:
    actions_only = [c.action for c in real_candidates]
    kept = self.priority_filter(actions_only, mode="strict", priority_min_gap=self.priority_min_gap)
    kept_set = set(id(a) for a in kept)
    real_candidates = [c for c in real_candidates if id(c.action) in kept_set]
```
（用 id 匹配避免 DispatchAction 相等性问题；或更稳妥地按 lot+ppid 匹配）

### #9 score 符号修正
当前 _candidate_features 的 score:
```python
qtime_slack = max(0.0, get_qtime_deadline - predicted_completion)
score = due_urgency + waiting_time + (1.0/max(qtime_slack,1e-3)) - 0.001*proc - 0.001*qrisk
```
报告 §4.1: `CandidateScore = due_date_urgency + qtime_slack + waiting_time - proc_time_mean - resource_conflict_risk`
报告里 qtime_slack 是**正贡献的线性项**（slack 越大余量越足）。但当前代码用 `1/qtime_slack`（slack 越小 score 越高），方向相反且会爆炸到 1000 压倒其它项。

注意：报告本意"slack 大 = 余量足"应是正贡献，但从调度直觉"越紧急越该先派"角度，紧急(slack 小)反而应优先。这里**遵循报告 §4.1 原文**：用线性正项 qtime_slack。但要归一化避免量纲问题。

改为（遵循报告，线性，紧急度用 due_urgency 已体现）：
```python
score = due_urgency + waiting_time + qtime_slack_norm - 0.001*total_process_time - 0.001*qtime_risk
```
其中 qtime_slack_norm 用一个有界形式，例如不直接用原始 slack（量纲大），而用 `due_urgency` 已涵盖紧迫度。最简单且符合报告：直接用线性 `qtime_slack`（带一个小系数避免压倒），如 `0.1*qtime_slack`。

**实际采用**：score = due_urgency + waiting_time + 0.1*qtime_slack - 0.001*total_process_time - 0.001*qtime_risk
（去掉倒数项，改回报告的线性正贡献；系数 0.1 防止量纲压倒。）

测试 tests/test_phase1_review_score_priority.py:
- score 不再出现 1/slack 爆炸（构造 slack 极小的候选，score 有界，不超过某合理上限如 100）
- priority_filter_mode="strict" 时，候选池只含最高优先级 lot 的动作（用 small encoder，构造多个已到达 lot，strict 模式应只留 priority 最高的）
- priority_filter_mode="soft"（默认）时，候选池行为与之前一致（含多个优先级的 lot）

---

## Task R2: qtime mask 时间基准 + visible_lots + is_doomed (#2 #3 #4)

这三个关联，一起做。

### #2 时间基准对齐
estimate() 的 mu_finish 是相对 t=0 的 makespan。mask 比较 `deadline - mu_finish` 把绝对时刻减相对时长，基准不一致。

修复：estimate() 增加 `start_offset=0.0` 参数，makespan 各阶段起点加上 start_offset（即 _run_list_schedule 的 instance_free / batch_ready 初值变为 start_offset，或更简单：mu_finish += start_offset 在最后加）。
- 最简单正确做法：estimate 末尾 `mu_finish += start_offset`（σ 不变，offset 是确定值）
- mask 调用时传入 lot 的预计开始时刻 start_offset = max(current_time, arrival_time)（乐观估计：lot 能立即开工）

### #3 检查所有 visible_lots（不只被调度 lot 自身）
报告 §3.2: 提交候选 i 后，遍历所有 visible_lots，看 i 是否害某等待 lot 踩穿 qtime。
当前只检查候选自己的 lot。

完整实现较重（需 dry-run commit 后对每个 visible lot 估时）。本次采用**可行的中间版本**：
- 对候选动作 i 的 lot 自身：用 start_offset = max(current_time, arrival) 估时判断（被调度 lot，保证违规概率≤ε）
- 对其它 visible_lots（已到达+前瞻窗内即将到达，排除已完成和 i 自己的 lot）：用乐观估时（start_offset = max(current_time, 其 arrival)）判断它们是否"已经注定违规"——若某 visible lot 在**不提交 i**时就已 doomed，则它不作为屏蔽依据(#4)。
- 注：完整的"提交 i 后挤占其它 lot"需模拟 i 占用资源后再估其它 lot，开销大。本版先实现：i 自身机会约束 + visible lots 的 doomed 排除框架，为后续完整版铺路。在 docstring 注明这是"自身严格 + 其它 lot doomed 排除"的版本。

### #4 is_doomed 排除
新增 env 方法 `is_doomed(lot, start_offset_estimate)`：
- 若该 lot 即便立即开工（最乐观）也注定违规：deadline - (earliest_start + mu_finish) < 0（均值口径，不加 z·σ），则 doomed
- doomed lot 不作为屏蔽依据
- 实现：用 estimate 算该 lot 最乐观 mu_finish，earliest_start = max(current_time, arrival)，若 deadline - earliest_start - mu_finish < 0 → doomed

在 qtime_safe_mask 中：对候选 i，若 i 的 lot 自身 is_doomed，则不因它屏蔽（它已经注定，屏蔽无意义，且会导致死锁）。即 doomed lot 的候选动作**不被 qtime mask 屏蔽**（让它能被派出，违规计入指标）。

测试 tests/test_phase1_review_qtime_mask.py:
- estimate(start_offset=10) 的 mu_finish 比 start_offset=0 大 10（确定性偏移）
- 时间基准：current_time 推进后，mask 判断用绝对时刻（构造一个 lot，t=0 时安全，advance_time 到接近 deadline 后同动作被屏蔽）
- is_doomed: 构造一个 deadline 已过的 lot（deadline < current_time + 最乐观完成），is_doomed 返回 True
- doomed lot 的候选不被屏蔽（候选池不会因 doomed lot 全空）
- 现有 qtime mask 测试不回归

---

## 回归验证（两个 task 都做完后）
```powershell
Set-Location "d:\HuaweiMoveData\Users\XWJ\Desktop\FAB-main\FAB_RL\FABenv"
$env:KMP_DUPLICATE_LIB_OK="TRUE"
& "D:\Anaconda\python.exe" -m pytest tests/test_lower_layer_estimator.py tests/test_phase2_environment_interfaces.py tests/test_phase1_pressure_demo.py tests/test_phase2_candidate_rank_features.py tests/test_phase1_review_score_priority.py tests/test_phase1_review_qtime_mask.py -p no:cacheprovider -q
```
