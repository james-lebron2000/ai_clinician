from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from ai_clinician.models import (
    CandidateDiseaseState,
    CausalEffectEstimate,
    CohortSnapshot,
    DataProfileRegistration,
    DecisionPoint,
    GuidelineChannel,
    LifecycleStatus,
    ModelRelease,
    RecommendationStrength,
    ResearchStudy,
    StateValidationReport,
    TimelineQualityReport,
    transition_study,
)
from ai_clinician.identity import GovernanceDirectory
from ai_clinician.orchestrator import ResearchWorkflow, WorkflowGateError
from ai_clinician.privacy import PrivacyConfig
from ai_clinician.store import ResearchStore


def _workflow(tmp_path: Path) -> ResearchWorkflow:
    raw = tmp_path / "raw"
    git = tmp_path / "repo"
    raw.mkdir()
    git.mkdir()
    privacy = PrivacyConfig(raw_root=raw, derived_root=tmp_path / "derived", git_root=git)
    token = "synthetic-workflow-data-steward-token-123"
    directory = GovernanceDirectory(
        {
            token: {
                "subject": "workflow_data_steward",
                "roles": ["data_steward"],
            }
        }
    )
    return ResearchWorkflow(
        ResearchStore(
            "metadata.db",
            privacy=privacy,
            audit_signing_key="synthetic-workflow-audit-signing-key-32-bytes",
            governance_directory=directory,
        ),
        governance_token=token,
    )


def test_required_refusal_contracts_are_fail_closed() -> None:
    with pytest.raises(ValidationError, match="require abstention"):
        DecisionPoint(
            patient_key="pt_abc",
            decision_type="first_line_start",
            decision_time=datetime.now(timezone.utc),
            missing_critical_features=["ECOG"],
        )


def test_cohort_snapshot_requires_fixed_patient_level_temporal_split() -> None:
    common = {
        "id": "cohort_synthetic",
        "study_id": "study_synthetic",
        "data_profile_id": "profile_synthetic",
        "source_data_sha256": "1" * 64,
        "patient_membership_sha256": "2" * 64,
        "timeline_snapshot_sha256": "3" * 64,
        "split_manifest_sha256": "4" * 64,
        "development_membership_sha256": "5" * 64,
        "tuning_membership_sha256": "6" * 64,
        "locked_test_membership_sha256": "7" * 64,
        "development_patient_count": 14,
        "tuning_patient_count": 3,
        "locked_test_patient_count": 3,
        "eligible_patient_count": 20,
        "locked_at": datetime.now(timezone.utc),
    }
    assert CohortSnapshot(**common).development_patient_count == 14
    with pytest.raises(ValidationError, match="70/15/15"):
        CohortSnapshot(
            **{
                **common,
                "development_patient_count": 13,
                "tuning_patient_count": 4,
            }
        )
    with pytest.raises(ValidationError, match="promotion gate"):
        CandidateDiseaseState(label="unstable", eligible_for_guideline=True)
    with pytest.raises(ValidationError, match="cannot report an effect"):
        CausalEffectEstimate(
            target_trial_id="trial_1",
            strategy="A",
            comparator="B",
            outcome="OS",
            estimate=1.0,
            ci_lower=0.5,
            ci_upper=1.5,
            method="AIPW",
            eligible_patients=10,
            strategy_patients=5,
            comparator_patients=5,
            common_support_fraction=0.2,
            strategy_effective_sample_size=3,
            comparator_effective_sample_size=3,
            maximum_absolute_smd=0.2,
            abstained=True,
            abstention_reasons=["LOW_OVERLAP"],
        )


def test_lifecycle_cannot_skip_or_pass_failed_timeline_gate(tmp_path: Path) -> None:
    workflow = _workflow(tmp_path)
    study = workflow.create_study("mCRC MVP", "Locked retrospective research only")
    with pytest.raises(ValueError, match="invalid research lifecycle"):
        transition_study(study, LifecycleStatus.COHORT_LOCKED)

    workflow.save_artifact(
        "data_profile",
        "profile_1",
        DataProfileRegistration(
            id="profile_1",
            source_snapshot_sha256="1" * 64,
            workbook_count=7,
            sheet_count=7,
            aggregate_patient_count=500,
            quarantined_row_count=0,
        ),
    )
    study = workflow.transition(
        study.id, LifecycleStatus.DATA_REGISTERED, data_profile_id="profile_1"
    )
    failed = TimelineQualityReport(
        id="quality_1",
        gold_total_case_count=500,
        reference_case_count=100,
        critical_event_counts={
            "treatment_line": 100,
            "progression": 100,
            "death": 100,
        },
        decision_view_patient_count=100,
        decision_view_count=100,
        decision_view_failure_count=0,
        critical_event_precision=0.89,
        critical_event_recall=0.95,
        critical_event_f1=0.91,
        treatment_line_macro_f1=0.92,
        date_within_14_days=0.93,
        cohen_kappa=0.81,
        model_gold_kappa=0.80,
        future_leakage_count=0,
        passed=False,
        failures=["precision_below_threshold:treatment_line"],
    )
    workflow.save_artifact("timeline_quality_report", "quality_1", failed)
    with pytest.raises(WorkflowGateError, match="downgrade"):
        workflow.transition(
            study.id,
            LifecycleStatus.TIMELINE_VALIDATED,
            timeline_quality_report_id="quality_1",
        )


def test_ai_channel_cannot_be_promoted_to_clinical_strength() -> None:
    from ai_clinician.models import (
        EvidenceCertainty,
        EvidenceToDecision,
        GuidelineRecommendation,
        GuidelineSource,
        Jurisdiction,
        PICO,
    )

    with pytest.raises(ValidationError, match="research_only recommendation strength"):
        GuidelineRecommendation(
            title="candidate strategy",
            channel=GuidelineChannel.AI_CANDIDATE,
            jurisdiction=Jurisdiction.CN,
            guideline_version="0.1",
            pico=PICO(population="mCRC", intervention="A", comparator="B", outcomes=["OS"]),
            eligibility=["research cohort"],
            evidence_certainty=EvidenceCertainty.VERY_LOW,
            recommendation_strength=RecommendationStrength.CONDITIONAL,
            rationale="observational candidate",
            safety_constraints=["expert review"],
            hard_safety_criteria=["ood == false"],
            sources=[
                GuidelineSource(
                    organization="Study",
                    title="Locked protocol",
                    version="1",
                    url="https://example.invalid/protocol",
                    source_locator="section 1",
                    license_status="metadata_only",
                )
            ],
            evidence_to_decision=EvidenceToDecision(
                problem_priority="important",
                desirable_effects="potential benefit",
                undesirable_effects="potential harm",
                values_and_preferences="not yet measured",
                balance_of_effects="uncertain",
                resource_use="not assessed",
                equity="not assessed",
                acceptability="research only",
                feasibility="research only",
                panel_conclusion="not approved",
            ),
            uncertainty="unmeasured confounding",
            abstention_conditions=["OOD"],
        )


def test_locked_model_release_requires_validation_and_independent_reviews() -> None:
    with pytest.raises(ValidationError, match="validation evidence"):
        ModelRelease(
            name="state model",
            version="0.1",
            context_of_use="Synthetic research state discovery",
            status="locked",
            model_sha256="a" * 64,
            code_sha256="b" * 64,
            data_snapshot_ids=["snapshot_1"],
            preprocessing_version="1",
            forbidden_uses=["clinical care"],
            expert_review_ids=["review_1", "review_2"],
            locked_at=datetime.now(timezone.utc),
        )


def test_state_validation_recomputes_relative_brier_improvement() -> None:
    with pytest.raises(ValidationError, match="derived from the locked scores"):
        StateValidationReport(
            model_release_id="model_synthetic",
            model_release_hash="1" * 64,
            candidate_state_ids=["state_synthetic"],
            data_snapshot_id="cohort_synthetic",
            data_snapshot_hash="2" * 64,
            partition_snapshot_id="partition_synthetic",
            partition_snapshot_hash="3" * 64,
            locked_test_patient_count=100,
            metrics_artifact_sha256="4" * 64,
            evaluation_code_sha256="5" * 64,
            expert_review_ids=["review_1", "review_2"],
            bootstrap_ari=0.80,
            calibration_slope=1.0,
            model_brier=0.30,
            baseline_brier=0.20,
            relative_brier_improvement=0.10,
            expert_confirmations=2,
            passed=True,
            locked_at=datetime.now(timezone.utc),
        )


@pytest.mark.parametrize(
    "target",
    [
        LifecycleStatus.TARGET_TRIALS_COMPLETED,
        LifecycleStatus.CANDIDATE_GUIDELINE,
        LifecycleStatus.RESEARCH_RELEASE,
    ],
)
def test_mvp_lifecycle_closures_apply_before_artifact_lookup(
    tmp_path: Path, target: LifecycleStatus
) -> None:
    workflow = _workflow(tmp_path)
    study = workflow.create_study(
        "Synthetic closed lifecycle", "Synthetic retrospective research only"
    )
    with pytest.raises(WorkflowGateError, match="disabled"):
        workflow.transition(study.id, target)


@pytest.mark.parametrize(
    "state",
    [
        LifecycleStatus.COHORT_LOCKED,
        LifecycleStatus.TARGET_TRIALS_COMPLETED,
        LifecycleStatus.CANDIDATE_GUIDELINE,
        LifecycleStatus.RESEARCH_RELEASE,
    ],
)
def test_store_refuses_direct_advanced_study_state(
    tmp_path: Path, state: LifecycleStatus
) -> None:
    workflow = _workflow(tmp_path)
    study = ResearchStudy(
        id=f"study_direct_{state.value}",
        name="Synthetic direct write",
        context_of_use="Synthetic retrospective research only",
        state=state,
    )
    with pytest.raises(ValueError, match="disabled|start in draft"):
        workflow.store.put(
            "study",
            study.id,
            study,
            governance_token=workflow.governance_token,
        )


def test_store_repeats_workflow_evidence_gate_for_direct_transition(
    tmp_path: Path,
) -> None:
    workflow = _workflow(tmp_path)
    study = workflow.create_study(
        "Synthetic persistence gate", "Synthetic retrospective research only"
    )
    bypass = study.model_copy(
        update={
            "state": LifecycleStatus.DATA_REGISTERED,
            "data_profile_id": "missing_profile",
            "updated_at": datetime.now(timezone.utc),
        }
    )
    with pytest.raises(ValueError, match="lifecycle gate failed"):
        workflow.store.put(
            "study",
            bypass.id,
            bypass,
            governance_token=workflow.governance_token,
        )
