from __future__ import annotations

from fastapi.testclient import TestClient
import pytest

from virtualhuman_agents.api import create_app
from virtualhuman_agents import cli as legacy_cli


def test_api_creates_and_starts_cro_project(tmp_path, goal):
    api_key = "synthetic-legacy-api-key-123"
    app = create_app(
        tmp_path / "api.db", tmp_path / "packages", api_key=api_key
    )
    client = TestClient(app)
    headers = {"X-API-Key": api_key}
    assert client.get("/health").status_code == 401
    assert client.get("/health", headers=headers).json()["status"] == "ok"
    response = client.post(
        "/projects", headers=headers, json=goal.model_dump(mode="json")
    )
    assert response.status_code == 201
    assert response.json()["state"] == "ready"
    started = client.post(f"/projects/{goal.id}/start", headers=headers)
    assert started.status_code == 200
    assert started.json()["project"]["state"] == "awaiting_results"
    audit = client.get(f"/projects/{goal.id}/audit", headers=headers)
    assert audit.status_code == 200
    assert audit.json()["chain_valid"] is True


def test_legacy_api_refuses_weak_or_placeholder_keys(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="at least 16"):
        create_app(tmp_path / "short.db", api_key="short")
    with pytest.raises(RuntimeError, match="placeholder"):
        create_app(
            tmp_path / "placeholder.db",
            api_key="provision-with-your-secret-manager",
        )


def test_legacy_cli_refuses_non_loopback_binding(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv", ["virtualhuman-agent", "serve", "--host", "0.0.0.0"]
    )
    with pytest.raises(ValueError, match="loopback"):
        legacy_cli.main()
