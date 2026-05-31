"""cProfile 定位 pressure commit 热点。"""
import cProfile, pstats, io, time
from problem_instances import build_pressure_test_encoder
from rl_environment import ResourceCalendarEnv, RewardConfig
from phase2_sas_observation import Phase2ObservationEncoder
from phase2_sas_driver import Phase2EpisodeDriver

enc = build_pressure_test_encoder()
env = ResourceCalendarEnv(enc, top_k=8)
env.reset()
driver = Phase2EpisodeDriver(env, Phase2ObservationEncoder(), RewardConfig())
driver.reset_episode()

# 计数收敛循环迭代次数：monkeypatch _simulate_action 周围不易，改为统计 find_earliest_slot 调用
orig_find = enc.find_earliest_slot
counters = {"find_earliest_slot": 0, "add_interval": 0}
def counting_find(busy, earliest, duration):
    counters["find_earliest_slot"] += 1
    return orig_find(busy, earliest, duration)
enc.find_earliest_slot = counting_find
orig_add = enc.add_calendar_interval
def counting_add(cal, key, s, e):
    counters["add_interval"] += 1
    return orig_add(cal, key, s, e)
enc.add_calendar_interval = counting_add

def run_n_commits(n):
    committed = 0
    it = 0
    while committed < n and it < 400:
        it += 1
        machines = driver.get_dispatchable_machines()
        if not machines:
            if driver.advance_to_next_event() is None:
                break
            continue
        m = driver.select_next_machine(machines)
        decision = driver.build_decision(m)
        idx = driver._rule_action_index(decision.pool, "EDD")
        if idx is None:
            if driver.advance_to_next_event() is None:
                break
            continue
        r = driver.step_with_action(m, idx, pool=decision.pool)
        if r.info.get("insertion_success"):
            committed += 1
    return committed

t0 = time.time()
pr = cProfile.Profile()
pr.enable()
n = run_n_commits(12)
pr.disable()
elapsed = time.time() - t0
print(f"committed={n}, elapsed={elapsed:.1f}s, {elapsed/max(n,1):.2f}s/commit")
print(f"find_earliest_slot calls={counters['find_earliest_slot']}, add_interval calls={counters['add_interval']}")
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats("tottime")
ps.print_stats(18)
print(s.getvalue())
