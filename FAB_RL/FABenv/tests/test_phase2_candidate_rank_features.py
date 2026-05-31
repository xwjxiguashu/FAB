import sys
from pathlib import Path


PHASE1_DIR = Path(__file__).resolve().parents[1]
if str(PHASE1_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE1_DIR))


from problem_instances import build_pressure_test_encoder, build_small_encoder
from rl_environment import ResourceCalendarEnv


def _feature_index(name):
    return ResourceCalendarEnv.feature_names.index(name)


def _real_candidate_indices(pool):
    return [
        index
        for index, action in enumerate(pool.actions)
        if bool(pool.action_mask[index]) and not action.is_wait and not action.is_padding
    ]


def _pool_with_at_least_two_real_candidates(env):
    for machine in range(1, int(env.encoder.num_machines) + 1):
        pool = env.build_candidate_pool(machine=machine)
        if len(_real_candidate_indices(pool)) >= 2:
            return pool
    raise AssertionError("small encoder should expose at least two real candidates")


def test_candidate_rank_feature_names_are_appended_to_existing_features():
    assert ResourceCalendarEnv.feature_names[-4:] == (
        "priority_rank_norm",
        "due_slack_rank_norm",
        "is_best_priority",
        "is_most_urgent_due",
    )


def test_real_candidates_receive_priority_and_due_slack_ranks():
    encoder = build_pressure_test_encoder()
    env = ResourceCalendarEnv(encoder, top_k=8)
    pool = _pool_with_at_least_two_real_candidates(env)

    priority_idx = _feature_index("priority")
    due_slack_idx = _feature_index("due_slack")
    priority_rank_idx = _feature_index("priority_rank_norm")
    due_slack_rank_idx = _feature_index("due_slack_rank_norm")

    real_indices = _real_candidate_indices(pool)
    assert len(real_indices) >= 2

    priorities = {index: pool.features[index, priority_idx] for index in real_indices}
    due_slacks = {index: pool.features[index, due_slack_idx] for index in real_indices}
    expected_priority_order = sorted(real_indices, key=lambda index: (-priorities[index], index))
    expected_due_order = sorted(real_indices, key=lambda index: (due_slacks[index], index))
    n_real = len(real_indices)

    for rank, index in enumerate(expected_priority_order, start=1):
        expected = (n_real - rank + 1) / n_real
        assert pool.features[index, priority_rank_idx] == expected

    for rank, index in enumerate(expected_due_order, start=1):
        expected = (n_real - rank + 1) / n_real
        assert pool.features[index, due_slack_rank_idx] == expected


def test_best_priority_and_most_urgent_due_flags_match_pool_extremes():
    encoder = build_small_encoder()
    env = ResourceCalendarEnv(encoder, top_k=8)
    pool = env.build_candidate_pool(machine=1)

    priority_idx = _feature_index("priority")
    due_slack_idx = _feature_index("due_slack")
    best_priority_idx = _feature_index("is_best_priority")
    urgent_due_idx = _feature_index("is_most_urgent_due")

    real_indices = _real_candidate_indices(pool)
    assert real_indices

    max_priority = max(pool.features[index, priority_idx] for index in real_indices)
    min_due_slack = min(pool.features[index, due_slack_idx] for index in real_indices)

    for index in real_indices:
        assert pool.features[index, best_priority_idx] == float(
            pool.features[index, priority_idx] == max_priority
        )
        assert pool.features[index, urgent_due_idx] == float(
            pool.features[index, due_slack_idx] == min_due_slack
        )


def test_wait_and_padding_rows_keep_rank_features_zero():
    encoder = build_small_encoder()
    env = ResourceCalendarEnv(encoder, top_k=8)
    pool = env.build_candidate_pool(machine=1)

    rank_indices = [
        _feature_index("priority_rank_norm"),
        _feature_index("due_slack_rank_norm"),
        _feature_index("is_best_priority"),
        _feature_index("is_most_urgent_due"),
    ]

    non_real_indices = [
        index
        for index, action in enumerate(pool.actions)
        if action.is_wait or action.is_padding or not bool(pool.action_mask[index])
    ]
    assert non_real_indices

    for index in non_real_indices:
        assert pool.features[index, rank_indices].tolist() == [0.0, 0.0, 0.0, 0.0]
