from __future__ import annotations

import pytest

from virtualhuman_agents.demo import sentinel_goal
from virtualhuman_agents.store import SQLiteStore


@pytest.fixture
def store(tmp_path):
    return SQLiteStore(tmp_path / "test.db")


@pytest.fixture
def goal():
    return sentinel_goal(locked_validation=True)
