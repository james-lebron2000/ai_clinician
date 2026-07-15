from __future__ import annotations

import sqlite3
import subprocess
from pathlib import Path

import pytest

from ai_clinician.identity import GovernanceAuthenticationError, GovernanceDirectory
from ai_clinician.models import (
    DecisionPointType,
    ResearchStudy,
    TargetTrialSpec,
    TimelineValidationManifestRegistration,
)
from ai_clinician.privacy import (
    PrivacyBoundaryError,
    PrivacyConfig,
    assert_aggregate_safe_payload,
    pseudonymize_identifier,
)
from ai_clinician.store import ResearchStore


DATA_STEWARD_TOKEN = "synthetic-store-data-steward-token-123"
METHODOLOGIST_TOKEN = "synthetic-store-methodologist-token-123"


def _directory() -> GovernanceDirectory:
    return GovernanceDirectory(
        {
            DATA_STEWARD_TOKEN: {
                "subject": "store_data_steward",
                "roles": ["data_steward"],
            },
            METHODOLOGIST_TOKEN: {
                "subject": "store_methodologist",
                "roles": ["methodologist"],
            },
        }
    )


def _privacy(tmp_path: Path) -> PrivacyConfig:
    raw = tmp_path / "raw"
    derived = tmp_path / "derived"
    git = tmp_path / "repo"
    raw.mkdir(parents=True)
    git.mkdir(parents=True)
    return PrivacyConfig(raw_root=raw, derived_root=derived, git_root=git)


def test_privacy_roots_reject_git_and_traversal(tmp_path: Path) -> None:
    git = tmp_path / "repo"
    raw = git / "raw"
    git.mkdir()
    raw.mkdir()
    with pytest.raises(PrivacyBoundaryError, match="outside the Git"):
        PrivacyConfig(raw_root=raw, derived_root=tmp_path / "derived", git_root=git)

    privacy = _privacy(tmp_path / "safe")
    with pytest.raises(PrivacyBoundaryError, match="escapes"):
        privacy.resolve_derived("../leak.db")


def test_environment_cannot_spoof_the_real_git_worktree(
    tmp_path: Path, monkeypatch
) -> None:
    raw = tmp_path / "raw"
    fake_git = tmp_path / "fake-git"
    raw.mkdir()
    fake_git.mkdir()
    actual_repository = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("AI_CLINICIAN_RAW_ROOT", str(raw))
    monkeypatch.setenv("AI_CLINICIAN_DERIVED_ROOT", str(actual_repository))
    monkeypatch.setenv("AI_CLINICIAN_GIT_ROOT", str(fake_git))

    with pytest.raises(PrivacyBoundaryError, match="outside.*Git work tree"):
        PrivacyConfig.from_env()


def test_roots_are_rejected_inside_any_git_worktree(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    derived_repository = tmp_path / "derived-repository"
    configured_git = tmp_path / "configured-git"
    raw.mkdir()
    derived_repository.mkdir()
    configured_git.mkdir()
    subprocess.run(
        ["git", "init"], cwd=derived_repository, check=True, capture_output=True
    )

    with pytest.raises(PrivacyBoundaryError, match="every Git work tree"):
        PrivacyConfig(
            raw_root=raw,
            derived_root=derived_repository,
            git_root=configured_git,
        )


def test_pseudonymization_is_stable_and_payload_guard_rejects_patient_fields() -> None:
    first = pseudonymize_identifier("MRN-123", secret="a sufficiently long secret")
    second = pseudonymize_identifier("MRN-123", secret="a sufficiently long secret")
    assert first == second
    assert "MRN-123" not in first
    with pytest.raises(PrivacyBoundaryError, match="patient-level"):
        assert_aggregate_safe_payload({"result": {"patient_key": first}})
    with pytest.raises(PrivacyBoundaryError, match="placeholder"):
        pseudonymize_identifier(
            "MRN-123", secret="replace-with-an-approved-random-secret"
        )
    with pytest.raises(PrivacyBoundaryError, match="placeholder"):
        pseudonymize_identifier(
            "MRN-123", secret="provision-with-your-secret-manager"
        )


def test_store_is_versioned_append_only_and_auditable(tmp_path: Path) -> None:
    store = ResearchStore(
        "research/store.db",
        privacy=_privacy(tmp_path),
        audit_signing_key="synthetic-store-audit-signing-key-32-bytes",
        governance_directory=_directory(),
    )
    first_study = ResearchStudy(
        id="study_1",
        name="Synthetic mCRC study v1",
        context_of_use="Synthetic retrospective research only",
    )
    second_study = ResearchStudy.model_validate(
        first_study.model_copy(update={"name": "Synthetic mCRC study v2"}).model_dump()
    )
    first = store.put(
        "study", first_study.id, first_study, governance_token=DATA_STEWARD_TOKEN
    )
    second = store.put(
        "study", second_study.id, second_study, governance_token=DATA_STEWARD_TOKEN
    )
    assert first["version"] == 1
    assert second["version"] == 2
    assert store.get("study", first_study.id)["payload"]["name"].endswith("v2")
    assert store.verify_audit_chain()

    with sqlite3.connect(store.path) as connection, pytest.raises(
        sqlite3.IntegrityError, match="append-only"
    ):
        connection.execute(
            "UPDATE research_objects SET payload_json = '{}' WHERE object_id = 'study_1'"
        )


def test_store_refuses_patient_level_payload(tmp_path: Path) -> None:
    store = ResearchStore("store.db", privacy=_privacy(tmp_path))
    with pytest.raises(PrivacyBoundaryError, match="patient-level"):
        store.put("timeline", "unsafe", {"patient_id": "123", "events": []})
    with pytest.raises(PrivacyBoundaryError, match="pseudonyms"):
        store.put(
            "study",
            "pt_" + "a" * 64,
            ResearchStudy(
                name="Synthetic study",
                context_of_use="Synthetic retrospective research only",
            ),
        )
    with pytest.raises(TypeError, match="generic mappings are forbidden"):
        store.put(
            "study",
            "study_leak",
            {"summary": "患者" + "王小明" + "于2024年接受治疗后进展"},
        )
    with pytest.raises(ValueError, match="unsupported aggregate"):
        store.put("arbitrary_export", "export_1", {"summary": "synthetic"})


def test_governed_writes_require_authenticated_roles_and_real_trial_roots(
    tmp_path: Path,
) -> None:
    store = ResearchStore(
        "governed.db",
        privacy=_privacy(tmp_path),
        audit_signing_key="synthetic-store-audit-signing-key-32-bytes",
        governance_directory=_directory(),
    )
    study = ResearchStudy(
        id="study_governed",
        name="Synthetic governed study",
        context_of_use="Synthetic retrospective research only",
    )
    with pytest.raises(GovernanceAuthenticationError, match="credential"):
        store.put("study", study.id, study)

    spec = TargetTrialSpec(
        id="trial_missing_roots",
        name="Synthetic target trial",
        study_id="study_missing",
        cohort_snapshot_id="cohort_missing",
        cohort_snapshot_hash="1" * 64,
        data_snapshot_id="cohort_missing",
        analysis_code_sha256="2" * 64,
        analysis_config_sha256="3" * 64,
        baseline_release_id="baseline_missing",
        baseline_release_version=1,
        baseline_release_hash="4" * 64,
        baseline_recommendation_ids=["recommendation_missing"],
        baseline_recommendation_hashes={"recommendation_missing": "5" * 64},
        decision_point=DecisionPointType.FIRST_LINE_START,
        eligibility_criteria=["Synthetic eligibility"],
        strategies=["option-a", "option-b"],
        confounders=["ecog"],
        time_zero="First eligible decision",
        action_assigned_at_column="assigned_at",
        assignment_procedure="Pre-specified assignment emulation",
        follow_up="One year",
        follow_up_days=365,
        outcomes=["os_12m"],
        outcome_observed_at_columns={"os_12m": "os_observed_at"},
        causal_contrast="Intention-to-treat analogue",
        estimand="Risk difference",
        censoring_strategy="Inverse probability weighting",
        analysis_plan="IPTW marginal and doubly robust AIPW",
        ood_definition="Locked support detector",
        negative_control_outcome="Pre-treatment control",
        sensitivity_analyses=["Quantitative bias analysis"],
        preregistered=True,
    )
    with pytest.raises(ValueError, match="signed study and cohort"):
        store.put(
            "target_trial_spec",
            spec.id,
            spec,
            governance_token=METHODOLOGIST_TOKEN,
        )


def test_audit_verification_detects_payload_tampering(tmp_path: Path) -> None:
    store = ResearchStore(
        "store.db",
        privacy=_privacy(tmp_path),
        audit_signing_key="synthetic-store-audit-signing-key-32-bytes",
        governance_directory=_directory(),
    )
    study = ResearchStudy(
        id="study_1",
        name="Synthetic study",
        context_of_use="Synthetic retrospective research only",
    )
    store.put(
        "study", study.id, study, governance_token=DATA_STEWARD_TOKEN
    )
    with sqlite3.connect(store.path) as connection:
        connection.execute("DROP TRIGGER research_objects_no_update")
        connection.execute(
            "UPDATE research_objects SET payload_json = '{\"version\":\"tampered\"}'"
        )
    assert store.verify_audit_chain() is False


def test_audit_verification_rejects_signature_stripping_and_blocks_writes(
    tmp_path: Path,
) -> None:
    store = ResearchStore(
        "signed-chain.db",
        privacy=_privacy(tmp_path),
        audit_signing_key="synthetic-store-audit-signing-key-32-bytes",
        governance_directory=_directory(),
    )
    study = ResearchStudy(
        id="study_signature",
        name="Synthetic signature study",
        context_of_use="Synthetic retrospective research only",
    )
    store.put(
        "study", study.id, study, governance_token=DATA_STEWARD_TOKEN
    )
    with sqlite3.connect(store.path) as connection:
        connection.execute("DROP TRIGGER audit_chain_no_update")
        connection.execute("UPDATE audit_chain SET entry_signature = ''")

    assert store.verify_audit_chain() is False
    revised = ResearchStudy.model_validate(
        study.model_copy(update={"name": "Synthetic revised study"}).model_dump()
    )
    with pytest.raises(ValueError, match="verification failed before write"):
        store.put(
            "study",
            revised.id,
            revised,
            governance_token=DATA_STEWARD_TOKEN,
        )


def test_locked_artifacts_require_signed_audit_entries(tmp_path: Path) -> None:
    privacy = _privacy(tmp_path)
    manifest = TimelineValidationManifestRegistration(
        id="timeline_manifest_1",
        gold_snapshot_id="gold_snapshot_1",
        gold_snapshot_sha256="1" * 64,
        decision_view_manifest_sha256="2" * 64,
        locked_split_sha256="3" * 64,
        gold_total_case_count=500,
        locked_case_count=100,
        annotation_pair_count=500,
        annotation_kappa=0.85,
        extractor_release_id="extractor_release_1",
        extractor_sha256="4" * 64,
        evaluation_code_sha256="5" * 64,
    )
    unsigned = ResearchStore(
        "unsigned.db", privacy=privacy, governance_directory=_directory()
    )
    with pytest.raises(ValueError, match="signed audit"):
        unsigned.put(
            "timeline_validation_manifest",
            manifest.id,
            manifest,
            governance_token=DATA_STEWARD_TOKEN,
        )

    signed = ResearchStore(
        "signed.db",
        privacy=privacy,
        audit_signing_key="synthetic-audit-signing-key-32-bytes-long",
        governance_directory=_directory(),
    )
    signed.put(
        "timeline_validation_manifest",
        manifest.id,
        manifest,
        governance_token=DATA_STEWARD_TOKEN,
    )
    assert signed.verify_audit_chain() is True
    assert signed.audit_entries()[0]["entry_signature"]


def test_store_rejects_placeholder_audit_key(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="placeholder"):
        ResearchStore(
            "placeholder.db",
            privacy=_privacy(tmp_path),
            audit_signing_key="replace-with-at-least-32-random-secret-bytes",
        )
    with pytest.raises(ValueError, match="placeholder"):
        ResearchStore(
            "provision-placeholder.db",
            privacy=_privacy(tmp_path / "provision"),
            audit_signing_key="provision-at-least-32-random-bytes",
        )
