"""Tests for VC-MCTS AlphaZero augmentations: SAS priors and leaf values."""
import numpy as np
import pytest

from phase2_sas_driver import Phase2EpisodeDriver
from phase2_sas_observation import Phase2ObservationEncoder
from phase2_sas_policy import (
    Phase2SASActorCritic,
    Phase2SASMultiHeadActorCritic,
)
from problem_instances import build_small_encoder
from reservation_ledger import ReservationLedger
from rl_environment import ResourceCalendarEnv, RewardConfig
from vc_mcts_alphazero import (
    MultiHeadCriticLeafValue,
    SASPolicyPriorProvider,
    critic_to_objective_dims,
)
from vc_mcts_planner import VCMCTSConfig, VCMCTSPlanner


CANDIDATE_DIM = 18
GLOBAL_DIM = 9


def _driver():
    env = ResourceCalendarEnv(build_small_encoder(), top_k=8)
    env.reset()
    driver = Phase2EpisodeDriver(
        env,
        Phase2ObservationEncoder(),
        RewardConfig(),
        max_steps=200,
    )
    driver.reset_episode()
    return driver


def _multihead_policy():
    import torch

    torch.manual_seed(0)
    return Phase2SASMultiHeadActorCritic(CANDIDATE_DIM, GLOBAL_DIM, hidden_dim=32)


def test_prior_provider_is_valid_distribution():
    driver = _driver()
    machine = driver.select_next_machine(driver.get_dispatchable_machines())
    pool = driver.env.build_candidate_pool(machine)
    provider = SASPolicyPriorProvider(_multihead_policy())

    probs = provider.candidate_probs(driver, machine, pool=pool)

    assert probs.shape[0] == len(pool.actions)
    assert np.all(probs[~np.asarray(pool.action_mask, dtype=bool)] == 0.0)
    assert probs.sum() == pytest.approx(1.0, abs=1e-5)


def test_prior_provider_works_with_single_head_policy():
    driver = _driver()
    machine = driver.select_next_machine(driver.get_dispatchable_machines())
    pool = driver.env.build_candidate_pool(machine)
    provider = SASPolicyPriorProvider(
        Phase2SASActorCritic(CANDIDATE_DIM, GLOBAL_DIM, hidden_dim=32)
    )

    probs = provider.candidate_probs(driver, machine, pool=pool)

    assert probs.shape[0] == len(pool.actions)
    assert probs.sum() == pytest.approx(1.0, abs=1e-5)


def test_planner_policy_prior_renormalizes_root_edges():
    driver = _driver()
    machine = driver.select_next_machine(driver.get_dispatchable_machines())
    planner = VCMCTSPlanner(
        VCMCTSConfig(prior_source="policy", policy_reserve_prior=0.2),
        prior_provider=SASPolicyPriorProvider(_multihead_policy()),
    )

    actions = planner.build_root_actions(driver, ReservationLedger(), machine)
    priors = [action.prior for action in actions]

    assert all(prior > 0.0 for prior in priors)
    assert sum(priors) == pytest.approx(1.0, abs=1e-6)


def test_default_heuristic_prior_is_unchanged():
    driver = _driver()
    machine = driver.select_next_machine(driver.get_dispatchable_machines())
    planner = VCMCTSPlanner(VCMCTSConfig())

    actions = planner.build_root_actions(driver, ReservationLedger(), machine)
    no_op = next(action for action in actions if action.kind == "no_op")

    assert no_op.prior == pytest.approx(0.05)
    assert sum(action.prior for action in actions) != pytest.approx(1.0, abs=1e-6)


def test_leaf_value_estimate_returns_multihead_channels():
    driver = _driver()
    machine = driver.select_next_machine(driver.get_dispatchable_machines())
    leaf = MultiHeadCriticLeafValue(_multihead_policy())

    values = leaf.estimate(driver, machine)

    assert set(values) == {"qtime", "util"}
    assert np.isfinite(values["qtime"])
    assert np.isfinite(values["util"])


def test_leaf_value_rejects_single_head_policy():
    with pytest.raises(TypeError):
        MultiHeadCriticLeafValue(
            Phase2SASActorCritic(CANDIDATE_DIM, GLOBAL_DIM, hidden_dim=32)
        )


def test_critic_to_objective_dims_maps_available_channels():
    partial = {
        "qtime_violation_count": 2.0,
        "qtime_violation_total": 5.0,
        "priority_weighted_wait": 30.0,
        "avg_utilization": 0.1,
    }

    dims = critic_to_objective_dims(
        {"qtime": -0.1, "util": 0.7},
        partial,
        num_lots=10,
    )

    assert dims["qtime_violation_count"] == pytest.approx(3.0)
    assert dims["qtime_violation_total"] == 5.0
    assert dims["priority_weighted_wait"] == 30.0
    assert dims["avg_utilization"] == pytest.approx(0.7)


def test_critic_mapping_clips_util_and_floors_remaining_qtime():
    dims = critic_to_objective_dims(
        {"qtime": 0.5, "util": 1.5},
        {"qtime_violation_count": 0.0},
        num_lots=4,
    )

    assert dims["qtime_violation_count"] == 0.0
    assert dims["avg_utilization"] == 1.0


def test_evaluate_action_leaf_value_path_returns_finite_objective():
    driver = _driver()
    machine = driver.select_next_machine(driver.get_dispatchable_machines())
    ledger = ReservationLedger()
    planner = VCMCTSPlanner(
        VCMCTSConfig(use_leaf_value=True, leaf_rollout_depth=4),
        leaf_value=MultiHeadCriticLeafValue(_multihead_policy()),
    )
    actions = planner.build_root_actions(driver, ledger, machine)
    dispatch = next(
        action
        for action in actions
        if action.kind in {"dispatch", "delegate_dispatch"}
    )

    objective = planner.evaluate_action(driver, ledger, dispatch)

    assert np.isfinite(objective.qtime_violation_count)
    assert np.isfinite(objective.priority_weighted_wait)
    assert 0.0 <= objective.avg_utilization <= 1.0
    assert objective.is_leaf_bootstrap is True


def test_leaf_value_default_off_uses_full_rollout():
    driver = _driver()
    machine = driver.select_next_machine(driver.get_dispatchable_machines())
    ledger = ReservationLedger()
    planner = VCMCTSPlanner(VCMCTSConfig())
    actions = planner.build_root_actions(driver, ledger, machine)
    dispatch = next(
        action
        for action in actions
        if action.kind in {"dispatch", "delegate_dispatch"}
    )

    objective = planner.evaluate_action(driver, ledger, dispatch)

    assert objective.qtime_violation_count >= 0.0
    assert objective.priority_weighted_wait >= 0.0
    assert objective.is_leaf_bootstrap is False


def test_policy_prior_requires_checkpoint_when_enabled():
    from vc_mcts_probe import run_seed

    with pytest.raises(ValueError, match="alphazero checkpoint"):
        run_seed(
            instance="small",
            seed=0,
            strategy="FIFO",
            w_lookahead=4.0,
            top_b=1,
            top_k_dispatch=1,
            prior_source="policy",
            alphazero_checkpoint=None,
            n_iter=1,
            max_steps=20,
        )


def test_leaf_value_requires_checkpoint_when_enabled():
    from vc_mcts_probe import run_seed

    with pytest.raises(ValueError, match="alphazero checkpoint"):
        run_seed(
            instance="small",
            seed=0,
            strategy="FIFO",
            w_lookahead=4.0,
            top_b=1,
            top_k_dispatch=1,
            use_leaf_value=True,
            alphazero_checkpoint=None,
            n_iter=1,
            max_steps=20,
        )
