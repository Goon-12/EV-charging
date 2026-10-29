"""
Shared pytest configuration.

``src/`` is not a package (the scripts are run directly), so the tests put it on
``sys.path`` the same way the scripts rely on their own directory being importable.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_DIR / "src"

for _path in (str(SRC_DIR), str(PROJECT_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import config as cfg  # noqa: E402
from multiobjective_core import load_inputs  # noqa: E402


@pytest.fixture(scope="session")
def project_dir():
    """Absolute path of the project root."""
    return PROJECT_DIR


@pytest.fixture(scope="session")
def scenario():
    """
    The frozen 100-EV scenario plus its solar profile.

    Loaded once per session: loading and validating is not free and these are
    read-only inputs.
    """
    return load_inputs()


@pytest.fixture(scope="session")
def ev(scenario):
    return scenario[0]


@pytest.fixture(scope="session")
def solar(scenario):
    return scenario[1]


@pytest.fixture(scope="session")
def cfg_module():
    return cfg
