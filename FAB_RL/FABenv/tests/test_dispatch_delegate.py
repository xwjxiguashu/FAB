import torch

from dispatch_delegate import (
    RuleDispatchDelegate,
    SASPolicyDispatchDelegate,
    load_sas_policy_delegate,
)
from model_checkpoint import save_policy_checkpoint
from phase2_sas_driver import Phase2EpisodeDriver
from phase2_sas_observation import Phase2ObservationEncoder
from phase2_sas_policy import Phase2SASActorCritic
from rl_environment import ResourceCalendarEnv, RewardConfig


def _driver(env, max_steps=200):
    return Phase2EpisodeDriver(
        env,
        Phase2ObservationEncoder(),
        RewardConfig(),
        max_steps=max_steps,
    )


class WaitFirstPolicy(torch.nn.Module):
    def parameters(self):
        return iter(())

    def greedy_action(self, candidate_features, candidate_mask, global_features):
        wait_index = int(candidate_mask.shape[1] - 1)
        return {
            "action": torch.tensor([wait_index]),
            "log_prob": torch.tensor([0.0]),
            "value": torch.tensor([0.0]),
        }

    def sample_action(self, candidate_features, candidate_mask, global_features):
        return self.greedy_action(candidate_features, candidate_mask, global_features)


def test_rule_dispatch_delegate_matches_driver_rule(small_encoder):
    env = ResourceCalendarEnv(small_encoder, top_k=8, w_lookahead=4.0)
    driver = _driver(env)
    driver.reset_episode()
    machine = driver.select_next_machine(driver.get_dispatchable_machines())
    pool = driver.env.build_candidate_pool(machine)

    delegate = RuleDispatchDelegate(strategy="FIFO")
    selected = delegate.select_action_index(driver, machine, pool=pool)

    assert selected == driver._rule_action_index(pool, "FIFO")


def test_policy_dispatch_delegate_falls_back_when_policy_selects_wait(small_encoder):
    env = ResourceCalendarEnv(small_encoder, top_k=8, w_lookahead=4.0)
    driver = _driver(env)
    driver.reset_episode()
    machine = driver.select_next_machine(driver.get_dispatchable_machines())
    pool = driver.env.build_candidate_pool(machine)

    delegate = SASPolicyDispatchDelegate(
        WaitFirstPolicy(),
        fallback_delegate=RuleDispatchDelegate(strategy="FIFO"),
    )
    selected = delegate.select_action_index(driver, machine, pool=pool)

    assert selected == driver._rule_action_index(pool, "FIFO")


def test_load_sas_policy_delegate_loads_checkpoint(tmp_path, small_encoder):
    env = ResourceCalendarEnv(small_encoder, top_k=8, w_lookahead=4.0)
    driver = _driver(env)
    driver.reset_episode()
    machine = driver.select_next_machine(driver.get_dispatchable_machines())
    pool = driver.env.build_candidate_pool(machine)
    observation = driver.observation_encoder.encode(machine, pool, driver.env)
    policy = Phase2SASActorCritic(
        candidate_dim=observation.candidate_features.shape[1],
        global_dim=observation.global_features.shape[0],
        hidden_dim=16,
    )
    path = tmp_path / "sas.pt"
    save_policy_checkpoint(
        policy,
        path,
        candidate_dim=observation.candidate_features.shape[1],
        global_dim=observation.global_features.shape[0],
        hidden_dim=16,
        policy_type="single",
    )

    delegate = load_sas_policy_delegate(
        str(path),
        fallback_delegate=RuleDispatchDelegate(strategy="FIFO"),
    )
    selected = delegate.select_action_index(driver, machine, pool=pool)

    assert selected is None or isinstance(selected, int)
