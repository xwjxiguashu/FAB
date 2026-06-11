"""Pytest configuration for FABenv-local imports and shared fixtures."""
import os
import sys

import pytest


FABENV_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_DIR = os.path.join(FABENV_DIR, "scripts")
IMPORT_DIRS = (
    FABENV_DIR,
    os.path.join(SCRIPT_DIR, "run"),
    os.path.join(SCRIPT_DIR, "evaluation"),
    os.path.join(SCRIPT_DIR, "experiments"),
    os.path.join(SCRIPT_DIR, "probes"),
)
for import_dir in IMPORT_DIRS:
    if import_dir not in sys.path:
        sys.path.insert(0, import_dir)


@pytest.fixture
def small_encoder():
    from problem_instances import build_small_encoder

    return build_small_encoder()


@pytest.fixture
def small_env(small_encoder):
    from rl_environment import ResourceCalendarEnv

    env = ResourceCalendarEnv(small_encoder, top_k=8)
    env.reset()
    return env
