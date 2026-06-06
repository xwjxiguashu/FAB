"""Dispatch delegates used by VC-MCTS.

VC-MCTS owns reservation timing. A dispatch delegate owns the concrete
candidate index when the selected branch is "dispatch now".
"""

from dataclasses import dataclass

import torch


def _is_real_dispatch_action(pool, action_index):
    if action_index is None:
        return False
    index = int(action_index)
    if index < 0 or index >= len(pool.actions):
        return False
    if not bool(pool.action_mask[index]):
        return False
    action = pool.actions[index]
    if getattr(action, "is_padding", False) or getattr(action, "is_wait", False):
        return False
    return int(getattr(action, "ppid", 0)) != 0


@dataclass
class RuleDispatchDelegate:
    strategy: str = "FIFO"

    @property
    def label(self):
        return f"rule:{self.strategy}"

    def select_action_index(self, driver, machine, pool=None):
        pool = driver.env.build_candidate_pool(machine) if pool is None else pool
        return driver._rule_action_index(pool, self.strategy)


class SASPolicyDispatchDelegate:
    def __init__(self, policy, *, stochastic=False, fallback_delegate=None):
        self.policy = policy
        self.stochastic = bool(stochastic)
        self.fallback_delegate = fallback_delegate

    @property
    def label(self):
        mode = "stochastic" if self.stochastic else "greedy"
        return f"sas:{mode}"

    def _policy_device(self):
        try:
            return next(self.policy.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _policy_output(self, observation):
        device = self._policy_device()
        candidate_features = torch.as_tensor(
            observation.candidate_features,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        candidate_mask = torch.as_tensor(
            observation.candidate_mask,
            dtype=torch.bool,
            device=device,
        ).unsqueeze(0)
        global_features = torch.as_tensor(
            observation.global_features,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        with torch.no_grad():
            if self.stochastic:
                return self.policy.sample_action(
                    candidate_features,
                    candidate_mask,
                    global_features,
                )
            return self.policy.greedy_action(
                candidate_features,
                candidate_mask,
                global_features,
            )

    def select_action_index(self, driver, machine, pool=None):
        pool = driver.env.build_candidate_pool(machine) if pool is None else pool
        observation = driver.observation_encoder.encode(machine, pool, driver.env)
        output = self._policy_output(observation)
        action_index = int(output["action"].detach().cpu().reshape(-1)[0])
        if _is_real_dispatch_action(pool, action_index):
            return action_index
        if self.fallback_delegate is None:
            return None
        return self.fallback_delegate.select_action_index(driver, machine, pool=pool)


def load_sas_policy_delegate(
    checkpoint_path,
    *,
    stochastic=False,
    fallback_delegate=None,
    map_location="cpu",
):
    from model_checkpoint import load_policy_checkpoint

    policy, _checkpoint = load_policy_checkpoint(
        checkpoint_path,
        map_location=map_location,
    )
    return SASPolicyDispatchDelegate(
        policy,
        stochastic=stochastic,
        fallback_delegate=fallback_delegate,
    )
