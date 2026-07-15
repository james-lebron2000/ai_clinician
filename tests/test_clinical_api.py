from __future__ import annotations

import sqlite3
from pathlib import Path

from fastapi.testclient import TestClient
import pytest
from pydantic import ValidationError

from ai_clinician.api import create_app
from ai_clinician.guidelines import release_review_hash
from ai_clinician.identity import GovernanceDirectory
from ai_clinician.models import (
    EvidenceCertainty,
    EvidenceToDecision,
    GuidelineChannel,
    GuidelineRecommendation,
    GuidelineRelease,
    GuidelineSource,
    Jurisdiction,
    PICO,
    RecommendationStrength,
    ResearchStudy,
)
from ai_clinician.privacy import PrivacyBoundaryError, PrivacyConfig
from ai_clinician.store import ResearchStore


DATA_STEWARD_TOKEN = "synthetic-data-steward-token-123"
GOVERNANCE_TOKEN = "synthetic-research-governance-token-123"
CLINICAL_AUTHOR_TOKEN = "synthetic-clinical-author-token-123"
REVIEWER_TOKEN = "independent-reviewer-token"


def _directory() -> GovernanceDirectory:
    return GovernanceDirectory(
        {
            DATA_STEWARD_TOKEN: {
                "subject": "data_steward_1",
                "roles": ["data_steward"],
            },
            GOVERNANCE_TOKEN: {
                "subject": "research_governance_1",
                "roles": ["research_governance"],
            },
            CLINICAL_AUTHOR_TOKEN: {
                "subject": "clinical_author_1",
                "roles": ["crc_clinical_expert"],
            },
            REVIEWER_TOKEN: {
                "subject": "reviewer_1",
                "roles": ["crc_clinical_expert"],
            },
        }
    )


def _recommendation(identifier: str = "rec_1") -> GuidelineRecommendation:
    return GuidelineRecommendation(
        id=identifier,
        title="Synthetic research recommendation",
        channel=GuidelineChannel.AI_CANDIDATE,
        jurisdiction=Jurisdiction.CN,
        guideline_version="0.1",
        pico=PICO(
            population="Synthetic mCRC cohort",
            intervention="Synthetic option A",
            comparator="Synthetic option B",
            outcomes=["OS"],
        ),
        eligibility=["Synthetic structured eligibility"],
        evidence_certainty=EvidenceCertainty.VERY_LOW,
        recommendation_strength=RecommendationStrength.RESEARCH_ONLY,
        rationale="Synthetic research rationale",
        safety_constraints=["No clinical use"],
        hard_safety_criteria=["No automatic orders"],
        sources=[
            GuidelineSource(
                organization="SYN",
                title="Synthetic licensed source",
                version="0.1",
                url="https://example.invalid/synthetic",
                source_locator="synthetic-section",
                license_status="synthetic-permitted",
            )
        ],
        evidence_to_decision=EvidenceToDecision(
            problem_priority="Synthetic judgment",
            desirable_effects="Synthetic judgment",
            undesirable_effects="Synthetic judgment",
            values_and_preferences="Synthetic judgment",
            balance_of_effects="Synthetic judgment",
            resource_use="Synthetic judgment",
            equity="Synthetic judgment",
            acceptability="Synthetic judgment",
            feasibility="Synthetic judgment",
            panel_conclusion="Synthetic research only",
        ),
        uncertainty="Synthetic uncertainty",
        abstention_conditions=["Any missing critical field"],
    )


def _client(tmp_path: Path) -> TestClient:
    raw = tmp_path / "raw"
    git = tmp_path / "repo"
    raw.mkdir()
    git.mkdir()
    privacy = PrivacyConfig(raw_root=raw, derived_root=tmp_path / "derived", git_root=git)
    store = ResearchStore(
        "api.db",
        privacy=privacy,
        audit_signing_key="synthetic-api-audit-signing-key-32-bytes",
        governance_directory=_directory(),
    )
    study = ResearchStudy(
        id="study_1",
        name="Synthetic mCRC study",
        context_of_use="Synthetic retrospective research only",
    )
    store.put(
        "study", study.id, study, governance_token=DATA_STEWARD_TOKEN
    )
    return TestClient(create_app(store, api_key="test-only-api-key-123"))


def test_api_is_authenticated_and_aggregate_only(tmp_path: Path) -> None:
    client = _client(tmp_path)
    assert client.get("/health").json()["automatic_orders"] is False
    assert client.get("/v1/studies").status_code == 401
    response = client.get(
        "/v1/studies", headers={"X-API-Key": "test-only-api-key-123"}
    )
    assert response.status_code == 200
    body = response.json()[0]
    assert body["object_id"] == "study_1"
    assert body["state"] == "draft"
    assert "payload" not in body
    assert "name" not in body

    paths = client.get("/openapi.json").json()["paths"]
    serialized = " ".join(paths).casefold()
    assert "patient" not in serialized
    assert "timeline" not in serialized
    assert "order" not in serialized
    assert "download" not in serialized


def test_audit_verification_endpoint(tmp_path: Path) -> None:
    client = _client(tmp_path)
    headers = {"X-API-Key": "test-only-api-key-123"}
    assert client.get("/v1/audit/verify", headers=headers).json() == {"valid": True}


def test_api_refuses_governed_reads_when_audit_signature_is_stripped(
    tmp_path: Path,
) -> None:
    client = _client(tmp_path)
    headers = {"X-API-Key": "test-only-api-key-123"}
    database = tmp_path / "derived" / "api.db"
    with sqlite3.connect(database) as connection:
        connection.execute("DROP TRIGGER audit_chain_no_update")
        connection.execute("UPDATE audit_chain SET entry_signature = ''")

    assert client.get("/v1/audit/verify", headers=headers).json() == {"valid": False}
    assert client.get("/v1/studies", headers=headers).status_code == 503


def test_review_must_bind_current_artifact_hash(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    git = tmp_path / "repo"
    raw.mkdir()
    git.mkdir()
    privacy = PrivacyConfig(raw_root=raw, derived_root=tmp_path / "derived", git_root=git)
    directory = _directory()
    store = ResearchStore(
        "api.db",
        privacy=privacy,
        audit_signing_key="synthetic-api-audit-signing-key-32-bytes",
        governance_directory=directory,
    )
    release = GuidelineRelease(
        name="Synthetic candidate",
        version="0.1",
        channel="ai_candidate",
        status="in_review",
        recommendation_ids=["rec_1"],
        recommendation_hashes={"rec_1": _recommendation().content_hash()},
        context_of_use="Synthetic research review only",
    )
    recommendation = _recommendation()
    store.put(
        "guideline_recommendation",
        recommendation.id,
        recommendation,
        governance_token=CLINICAL_AUTHOR_TOKEN,
    )
    store.put(
        "guideline_release",
        release.id,
        release,
        governance_token=GOVERNANCE_TOKEN,
    )
    client = TestClient(
        create_app(
            store,
            api_key="test-only-api-key-123",
            governance_directory=directory,
        )
    )
    safe_response = client.get(
        f"/v1/recommendations/{recommendation.id}",
        headers={"X-API-Key": "test-only-api-key-123"},
    )
    assert safe_response.status_code == 200
    serialized_summary = safe_response.text
    assert recommendation.title not in serialized_summary
    assert recommendation.rationale not in serialized_summary
    assert "recommendation" not in safe_response.json()

    release_response = client.get(
        f"/v1/guideline-releases/{release.id}",
        headers={"X-API-Key": "test-only-api-key-123"},
    )
    assert release_response.status_code == 200
    release_summary = release_response.json()
    assert release_summary["release_id"] == release.id
    for forbidden in (
        "payload",
        "name",
        "context_of_use",
        "recommendation_ids",
        "recommendation_hashes",
    ):
        assert forbidden not in release_summary

    leaky = GuidelineRecommendation.model_validate(
        _recommendation("rec_leak").model_copy(
            update={"title": "患者" + "王小明" + "于2024年接受治疗后进展"}
        ).model_dump()
    )
    import pytest

    with pytest.raises(PrivacyBoundaryError, match="patient identifier"):
        store.put(
            "guideline_recommendation",
            leaky.id,
            leaky,
            governance_token=CLINICAL_AUTHOR_TOKEN,
        )
    headers = {
        "X-API-Key": "test-only-api-key-123",
        "X-Governance-Token": REVIEWER_TOKEN,
    }
    payload = {
        "object_id": release.id,
        "object_hash": "b" * 64,
        "role": "crc_clinical_expert",
        "decision": "approve",
        "rationale": "Synthetic independent expert assessment.",
    }
    assert client.post("/v1/expert-reviews", headers=headers, json=payload).status_code == 409
    payload["object_hash"] = release_review_hash(release)
    assert client.post("/v1/expert-reviews", headers=headers, json=payload).status_code == 201


def test_api_refuses_to_start_without_a_strong_key(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    git = tmp_path / "repo"
    raw.mkdir()
    git.mkdir()
    privacy = PrivacyConfig(raw_root=raw, derived_root=tmp_path / "derived", git_root=git)
    store = ResearchStore("api.db", privacy=privacy)
    import pytest

    with pytest.raises(RuntimeError, match="at least 16"):
        create_app(store, api_key="short")
    with pytest.raises(RuntimeError, match="placeholder"):
        create_app(store, api_key="replace-with-an-approved-local-api-key")
    with pytest.raises(RuntimeError, match="placeholder"):
        create_app(store, api_key="provision-with-your-secret-manager")


def test_reviewer_can_review_a_new_artifact_version(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    git = tmp_path / "repo"
    raw.mkdir()
    git.mkdir()
    directory = _directory()
    store = ResearchStore(
        "review-versions.db",
        privacy=PrivacyConfig(
            raw_root=raw,
            derived_root=tmp_path / "derived",
            git_root=git,
        ),
        audit_signing_key="synthetic-api-audit-signing-key-32-bytes",
        governance_directory=directory,
    )
    first = _recommendation("rec_versioned")
    store.put(
        "guideline_recommendation",
        first.id,
        first,
        governance_token=CLINICAL_AUTHOR_TOKEN,
    )
    client = TestClient(
        create_app(
            store,
            api_key="test-only-api-key-123",
            governance_directory=directory,
        )
    )
    headers = {
        "X-API-Key": "test-only-api-key-123",
        "X-Governance-Token": REVIEWER_TOKEN,
    }
    first_review = {
        "object_id": first.id,
        "object_hash": first.content_hash(),
        "role": "crc_clinical_expert",
        "decision": "request_changes",
        "rationale": "Synthetic revision requested for the first version.",
    }
    assert client.post("/v1/expert-reviews", headers=headers, json=first_review).status_code == 201

    second = GuidelineRecommendation.model_validate(
        first.model_copy(
            update={"rationale": "Synthetic revised research rationale"}
        ).model_dump()
    )
    store.put(
        "guideline_recommendation",
        second.id,
        second,
        governance_token=CLINICAL_AUTHOR_TOKEN,
    )
    second_review = {
        **first_review,
        "object_hash": second.content_hash(),
        "decision": "approve",
        "rationale": "Synthetic revised version is acceptable for research.",
    }
    assert client.post("/v1/expert-reviews", headers=headers, json=second_review).status_code == 201
    assert client.post("/v1/expert-reviews", headers=headers, json=second_review).status_code == 409


def test_baseline_recommendation_cannot_self_publish(tmp_path: Path) -> None:
    baseline_payload = _recommendation("baseline_rec").model_dump(mode="python")
    baseline_payload.update(
        {
            "channel": GuidelineChannel.BASELINE_APPROVED,
            "recommendation_strength": RecommendationStrength.CONDITIONAL,
        }
    )
    baseline = GuidelineRecommendation.model_validate(baseline_payload)
    with pytest.raises(ValidationError):
        GuidelineRecommendation.model_validate(
            {
                **baseline.model_dump(mode="python"),
                "status": "released",
                "research_only": False,
            }
        )
    with pytest.raises(ValidationError):
        GuidelineRelease.model_validate(
            {
                "name": "Synthetic unsafe release",
                "version": "0.1",
                "channel": "baseline_approved",
                "status": "draft",
                "recommendation_ids": [baseline.id],
                "recommendation_hashes": {baseline.id: baseline.content_hash()},
                "context_of_use": "Synthetic retrospective research only",
                "research_only": False,
            }
        )

    raw = tmp_path / "raw"
    git = tmp_path / "repo"
    raw.mkdir()
    git.mkdir()
    store = ResearchStore(
        "self-publish.db",
        privacy=PrivacyConfig(
            raw_root=raw,
            derived_root=tmp_path / "derived",
            git_root=git,
        ),
        audit_signing_key="synthetic-api-audit-signing-key-32-bytes",
        governance_directory=_directory(),
    )
    bypass = GuidelineRecommendation.model_construct(
        **{
            **baseline.__dict__,
            "status": "released",
            "research_only": False,
        }
    )
    with pytest.raises(ValidationError):
        store.put(
            "guideline_recommendation",
            bypass.id,
            bypass,
            governance_token=CLINICAL_AUTHOR_TOKEN,
        )
