import copy


def _calendar_snapshot(state):
    return (
        copy.deepcopy(state.machine_calendar),
        copy.deepcopy(state.chamber_calendar),
    )


def _first_real_action(env):
    for machine in range(1, int(env.encoder.num_machines) + 1):
        pool = env.build_candidate_pool(machine)
        for idx, (action, is_valid) in enumerate(zip(pool.actions, pool.action_mask)):
            action = env._coerce_action(action)
            if bool(is_valid) and not action.is_padding and not action.is_wait:
                return machine, pool, idx, action
    raise AssertionError("no valid real action found")


def test_dry_run_is_non_destructive(small_env):
    env = small_env
    before = _calendar_snapshot(env.state)
    _machine, _pool, _idx, action = _first_real_action(env)

    dry = env.dry_run_action(action)

    assert dry.success
    assert before == _calendar_snapshot(env.state)


def test_commit_then_rollback_restores_calendars(small_env):
    env = small_env
    before = _calendar_snapshot(env.state)
    machine, pool, idx, _action = _first_real_action(env)

    result = env.commit_action_index(machine, idx, pool=pool)
    rollback = env.rollback_last_commit()

    assert result.committed
    assert rollback.rolled_back
    assert before == _calendar_snapshot(env.state)
