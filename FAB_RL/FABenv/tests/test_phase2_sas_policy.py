import sys
from pathlib import Path

import torch


PHASE1_DIR = Path(__file__).resolve().parents[1]
if str(PHASE1_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE1_DIR))


from phase2_sas_policy import Phase2SASActorCritic


def test_policy_never_samples_masked_action():
    torch.manual_seed(0)
    policy = Phase2SASActorCritic(candidate_dim=18, global_dim=9, hidden_dim=32)
    candidate_features = torch.randn(1, 4, 18)
    candidate_mask = torch.tensor([[True, False, True, False]])
    global_features = torch.randn(1, 9)

    for _ in range(20):
        output = policy.sample_action(candidate_features, candidate_mask, global_features)
        assert int(output["action"].item()) in {0, 2}


def test_policy_evaluate_actions_returns_training_tensors():
    policy = Phase2SASActorCritic(candidate_dim=18, global_dim=9, hidden_dim=32)
    candidate_features = torch.randn(2, 4, 18)
    candidate_mask = torch.tensor([[True, False, True, False], [False, True, True, False]])
    global_features = torch.randn(2, 9)
    actions = torch.tensor([0, 2])

    output = policy.evaluate_actions(
        candidate_features,
        candidate_mask,
        global_features,
        actions,
    )

    assert output["log_prob"].shape == (2,)
    assert output["entropy"].shape == (2,)
    assert output["value"].shape == (2,)


def test_policy_rejects_rows_without_valid_actions():
    policy = Phase2SASActorCritic(candidate_dim=18, global_dim=9, hidden_dim=32)
    candidate_features = torch.randn(1, 4, 18)
    candidate_mask = torch.tensor([[False, False, False, False]])
    global_features = torch.randn(1, 9)

    try:
        policy.greedy_action(candidate_features, candidate_mask, global_features)
    except ValueError as exc:
        assert "at least one valid action" in str(exc)
    else:
        raise AssertionError("Expected ValueError for all-masked policy row")
