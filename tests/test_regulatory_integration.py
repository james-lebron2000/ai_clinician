from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from virtualhuman_agents.demo import run_demo
from virtualhuman_agents.models import ApprovalRole, ElectronicApproval
from virtualhuman_agents.orchestrator import PIOrchestrator
from virtualhuman_agents.regulatory import RegulatoryService
from virtualhuman_agents.store import ConflictError


def test_full_cro_fda_readiness_demo(tmp_path):
    summary = run_demo(tmp_path / "demo.db", tmp_path / "packages")
    assert summary["state"] == "completed"
    assert summary["rounds"] == 2
    assert len(summary["result_sites"]) == 2
    assert summary["audit_chain_valid"] is True
    assert summary["regulatory_ready"] is True
    assert summary["package_status"] == "qa_released"
    package_path = Path(summary["package_path"])
    assert package_path.exists()
    with zipfile.ZipFile(package_path) as archive:
        names = set(archive.namelist())
        assert "manifest.json" in names
        assert "PACKAGE-NOTICE.json" in names
        notice = json.loads(archive.read("PACKAGE-NOTICE.json"))
        assert notice["status"] == "staging_content_only"
        assert any(name.endswith("context-of-use.json") for name in names)
        assert any(name.endswith("data/observations.json") for name in names)


def test_approval_rejects_wrong_object_hash(store, goal, tmp_path):
    project = PIOrchestrator(store).create_project(goal)
    service = RegulatoryService(store, tmp_path / "packages")
    approval = ElectronicApproval(
        project_id=project.project_id,
        object_type="goal",
        object_id=goal.id,
        object_sha256="0" * 64,
        signer_user_id=goal.cro.sponsor_representative_user_id,
        signer_role=ApprovalRole.SPONSOR,
        meaning="approved",
        authentication_event_id="idp-event-12345",
    )
    with pytest.raises(ConflictError, match="hash does not match"):
        service.approve(approval)
