"""AlphaZero-style helpers for VC-MCTS.

This module is intentionally planner-adjacent: it exposes SAS policy priors
and multi-head critic leaf values without changing VC-MCTS defaults.
"""

import numpy as np
import torch


def _policy_device(policy):
    try:
        return next(policy.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _observation_tensors(observation, device):
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
    return candidate_features, candidate_mask, global_features


class SASPolicyPriorProvider:
    """Expose a SAS policy's masked softmax over the candidate pool."""

    def __init__(self, policy):
        self.policy = policy

    @property
    def label(self):
        return "sas_prior"

    def candidate_probs(self, driver, machine, pool=None):
        pool = driver.env.build_candidate_pool(machine) if pool is None else pool
        observation = driver.observation_encoder.encode(machine, pool, driver.env)
        device = _policy_device(self.policy)
        candidate_features, candidate_mask, global_features = _observation_tensors(
            observation,
            device,
        )

        with torch.no_grad():
            output = self.policy.greedy_action(
                candidate_features,
                candidate_mask,
                global_features,
            )

        probs = output["probs"].detach().cpu().reshape(-1).numpy().astype(float)
        action_mask = np.asarray(pool.action_mask, dtype=bool)
        if probs.shape[0] == action_mask.shape[0]:
            probs = np.where(action_mask, probs, 0.0)
        return probs


class MultiHeadCriticLeafValue:
    """Query qtime/util critic channels from a multi-head SAS policy."""

    def __init__(self, policy, qtime_channel="qtime", util_channel="util"):
        if not hasattr(policy, "critic_values"):
            raise TypeError(
                "MultiHeadCriticLeafValue requires a multi-head policy with "
                "critic_values(); got a single-head policy."
            )
        self.policy = policy
        self.qtime_channel = qtime_channel
        self.util_channel = util_channel

    @property
    def label(self):
        return "multihead_leaf_value"

    def estimate(self, driver, machine, pool=None):
        pool = driver.env.build_candidate_pool(machine) if pool is None else pool
        observation = driver.observation_encoder.encode(machine, pool, driver.env)
        device = _policy_device(self.policy)
        candidate_features, candidate_mask, global_features = _observation_tensors(
            observation,
            device,
        )

        with torch.no_grad():
            values = self.policy.critic_values(
                candidate_features,
                candidate_mask,
                global_features,
            )

        return {
            "qtime": float(
                values[self.qtime_channel].detach().cpu().reshape(-1)[0]
            ),
            "util": float(values[self.util_channel].detach().cpu().reshape(-1)[0]),
        }


def critic_to_objective_dims(critic_values, partial_metrics, num_lots):
    """Map critic channels plus partial metrics to VC-MCTS objective fields."""
    lot_count = max(float(num_lots), 1.0)
    remaining_qtime_count = max(0.0, -float(critic_values["qtime"]) * lot_count)
    partial_qtime_count = float(partial_metrics.get("qtime_violation_count", 0.0))
    avg_utilization = min(1.0, max(0.0, float(critic_values["util"])))

    return {
        "qtime_violation_count": partial_qtime_count + remaining_qtime_count,
        "qtime_violation_total": float(
            partial_metrics.get("qtime_violation_total", 0.0)
        ),
        "priority_weighted_wait": float(
            partial_metrics.get("priority_weighted_wait", 0.0)
        ),
        "avg_utilization": avg_utilization,
    }


def load_sas_alphazero(checkpoint_path, *, map_location="cpu", require_multihead=False):
    """Load a SAS checkpoint and return prior/leaf-value helper objects.

    Returns:
        (prior_provider, leaf_value_or_none, policy)
    """
    from model_checkpoint import load_policy_checkpoint

    policy, _checkpoint = load_policy_checkpoint(
        checkpoint_path,
        map_location=map_location,
    )
    prior_provider = SASPolicyPriorProvider(policy)

    if hasattr(policy, "critic_values"):
        return prior_provider, MultiHeadCriticLeafValue(policy), policy
    if require_multihead:
        raise TypeError("checkpoint is single-head but require_multihead=True")
    return prior_provider, None, policy
