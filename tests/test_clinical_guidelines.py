from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from ai_clinician.guidelines import (
    GuidelineGovernanceError,
    GuidelineRegistry,
    evaluate_recommendation_applicability,
    export_cpg_on_fhir,
)
from ai_clinician.models import (
    CandidateDiseaseState,
    CausalAuditArtifact,
    CausalEffectEstimate,
    DecisionPointType,
    EvidenceCertainty,
    EvidenceToDecision,
    ExpertReview,
    GuidelineChannel,
    GuidelineRecommendation,
    GuidelineSource,
    Jurisdiction,
    ModelRelease,
    PICO,
    RecommendationStrength,
    ReviewDecision,
    ReviewRole,
    StateValidationReport,
    TargetTrialResult,
    TargetTrialInputSnapshot,
    TargetTrialSpec,
    TreatmentOption,
)


def _causal_audit(
    spec: TargetTrialSpec,
    audit_type: str,
    identifier: str,
    input_snapshot: TargetTrialInputSnapshot,
) -> CausalAuditArtifact:
    return CausalAuditArtifact(
        id=identifier,
        target_trial_spec_id=spec.id,
        target_trial_spec_hash=spec.content_hash(),
        data_snapshot_id=spec.data_snapshot_id,
        input_snapshot_id=input_snapshot.id,
        input_snapshot_hash=input_snapshot.content_hash(),
        audit_type=audit_type,
        method="synthetic deterministic audit",
        result_sha256="e" * 64,
        code_sha256=spec.analysis_code_sha256,
        config_sha256=spec.analysis_config_sha256,
        passed=True,
    )


def _trial_input(spec: TargetTrialSpec, identifier: str) -> TargetTrialInputSnapshot:
    return TargetTrialInputSnapshot(
        id=identifier,
        target_trial_spec_id=spec.id,
        target_trial_spec_hash=spec.content_hash(),
        cohort_snapshot_id=spec.cohort_snapshot_id,
        cohort_snapshot_hash=spec.cohort_snapshot_hash,
        analytic_rows_sha256="1" * 64,
        patient_membership_sha256="2" * 64,
        eligible_patient_count=1200,
    )


def _option(
    identifier: str = "doublet",
    *,
    effects: list[CausalEffectEstimate] | None = None,
    result: TargetTrialResult | None = None,
) -> TreatmentOption:
    effect_items = effects or []
    return TreatmentOption(
        id=identifier,
        regimen_class=identifier,
        decision_point=DecisionPointType.FIRST_LINE_START,
        effect_estimate_ids=[effect.id for effect in effect_items],
        effect_estimate_hashes={
            effect.id: effect.content_hash() for effect in effect_items
        },
        target_trial_result_ids=[result.id] if result else [],
        target_trial_result_hashes=(
            {result.id: result.content_hash()} if result else {}
        ),
        benefits={"response": 0.6},
        harms={"grade_3_toxicity": 0.2},
        pareto_optimal=True,
        guideline_backed=True,
        automatic_order_allowed=False,
    )


def _recommendation(
    *,
    identifier: str = "rec-baseline",
    channel: GuidelineChannel = GuidelineChannel.BASELINE_APPROVED,
    state: CandidateDiseaseState | None = None,
    result: TargetTrialResult | None = None,
) -> GuidelineRecommendation:
    is_candidate = channel is GuidelineChannel.AI_CANDIDATE
    if is_candidate and result is None:
        # Structurally valid but intentionally unregistered evidence for the
        # negative governance test below.
        spec = TargetTrialSpec(
            id="unregistered-trial",
            name="unregistered synthetic trial",
            study_id="study-synthetic-mcrc",
            cohort_snapshot_id="snapshot-synthetic-locked",
            cohort_snapshot_hash="f" * 64,
            data_snapshot_id="snapshot-synthetic-locked",
            analysis_code_sha256="a" * 64,
            analysis_config_sha256="b" * 64,
            baseline_release_id="baseline-release-synthetic",
            baseline_release_version=1,
            baseline_release_hash="3" * 64,
            baseline_recommendation_ids=["baseline-rec-synthetic"],
            baseline_recommendation_hashes={"baseline-rec-synthetic": "4" * 64},
            decision_point=DecisionPointType.FIRST_LINE_START,
            eligibility_criteria=["synthetic"],
            strategies=["doublet", "alternative"],
            confounders=["ecog"],
            time_zero="decision",
            action_assigned_at_column="treatment_assigned_at",
            assignment_procedure="conditional",
            follow_up="12 months",
            follow_up_days=365,
            outcomes=["OS"],
            outcome_observed_at_columns={"OS": "os_observed_at"},
            causal_contrast="risk difference",
            estimand="risk difference",
            censoring_strategy="weighted",
            analysis_plan="IPTW marginal and AIPW",
            ood_definition="support",
            negative_control_outcome="control",
            sensitivity_analyses=["E-value"],
            preregistered=True,
        )
        fake_result_id = "unregistered-result"
        fake_input = _trial_input(spec, "unregistered-input")
        fake_effects = [
            CausalEffectEstimate(
                id=f"unregistered-effect-{index}",
                target_trial_id=spec.id,
                target_trial_result_id=fake_result_id,
                target_trial_spec_hash=spec.content_hash(),
                strategy="doublet",
                comparator="alternative",
                outcome="OS",
                estimate=0.05,
                ci_lower=0.01,
                ci_upper=0.09,
                method=method,
                corroborating_methods=[other],
                eligible_patients=1200,
                strategy_patients=600,
                comparator_patients=600,
                common_support_fraction=0.9,
                strategy_effective_sample_size=500,
                comparator_effective_sample_size=500,
                maximum_absolute_smd=0.05,
                sensitivity_analysis={"passed": "true"},
                negative_controls_passed=True,
                directionally_consistent=True,
            )
            for index, (method, other) in enumerate(
                [("AIPW", "IPTW"), ("IPTW", "AIPW")], start=1
            )
        ]
        result = TargetTrialResult(
            id=fake_result_id,
            spec=spec,
            effects=fake_effects,
            eligible_patient_count=1200,
            common_support_fraction=0.9,
            maximum_absolute_smd=0.05,
            input_snapshot=fake_input,
            negative_control_audit=_causal_audit(
                spec,
                "negative_control",
                "unregistered-negative-control",
                fake_input,
            ),
            sensitivity_audits=[
                _causal_audit(
                    spec,
                    "sensitivity_analysis",
                    "unregistered-sensitivity",
                    fake_input,
                )
            ],
            locked_at=datetime.now(timezone.utc),
        )
    candidate_state_id = state.id if state else "synthetic-state-1"
    candidate_state_hash = state.content_hash() if state else "d" * 64
    options = (
        [
            _option("doublet", effects=result.effects, result=result),
            _option("alternative", effects=result.effects, result=result),
        ]
        if is_candidate and result is not None
        else [_option()]
    )
    return GuidelineRecommendation(
        id=identifier,
        title="Synthetic first-line option",
        channel=channel,
        jurisdiction=Jurisdiction.CN,
        guideline_version="synthetic-1",
        pico=PICO(
            population="Synthetic adults with mCRC",
            intervention="doublet",
            comparator="alternative doublet",
            outcomes=["overall survival", "grade 3 toxicity"],
        ),
        eligibility=["ecog <= 2", "rass == 'wild_type'"],
        exclusions=["organ_failure == true"],
        evidence_certainty=(
            EvidenceCertainty.LOW
            if channel is GuidelineChannel.AI_CANDIDATE
            else EvidenceCertainty.MODERATE
        ),
        recommendation_strength=(
            RecommendationStrength.RESEARCH_ONLY
            if channel is GuidelineChannel.AI_CANDIDATE
            else RecommendationStrength.CONDITIONAL
        ),
        rationale="Synthetic rationale for software verification only.",
        safety_constraints=["Independent contraindication review is required."],
        hard_safety_criteria=["organ_failure == false"],
        sources=[
            GuidelineSource(
                organization="SYN",
                title="Synthetic guideline fixture",
                version="1",
                url="https://example.invalid/synthetic",
                source_locator="section 1",
                license_status="synthetic-test-content",
                content_sha256="a" * 64,
            )
        ],
        evidence_to_decision=EvidenceToDecision(
            problem_priority="important",
            desirable_effects="moderate expected benefit",
            undesirable_effects="clinically relevant toxicity",
            values_and_preferences="synthetic panel judgment",
            balance_of_effects="conditional balance",
            resource_use="resource implications reviewed",
            equity="equity impact uncertain",
            acceptability="acceptable for research review",
            feasibility="feasible in synthetic setting",
            panel_conclusion="conditional synthetic option",
        ),
        treatment_options=options,
        candidate_state_ids=(
            [candidate_state_id]
            if is_candidate
            else []
        ),
        candidate_state_hashes=(
            {candidate_state_id: candidate_state_hash} if is_candidate else {}
        ),
        uncertainty="Observational uncertainty remains.",
        abstention_conditions=["timeline_conflict == true"],
        research_only=True,
        automatic_order_allowed=False,
    )


def _approve(
    registry: GuidelineRegistry,
    release_id: str,
    reviewer: str,
    role: ReviewRole,
) -> ExpertReview:
    return registry.submit_review(
        ExpertReview(
            object_id=release_id,
            object_hash=registry.expected_review_hash(release_id),
            reviewer_key=reviewer,
            role=role,
            decision=ReviewDecision.APPROVE,
            rationale="Independent synthetic approval for test verification.",
        )
    )


def _register_candidate_evidence(
    registry: GuidelineRegistry,
) -> tuple[CandidateDiseaseState, TargetTrialResult]:
    model_review_ids = ["model-review-clinical", "model-review-method"]
    state_review_ids = ["state-review-1", "state-review-2"]
    model = ModelRelease(
        id="synthetic-state-model",
        name="synthetic state model",
        version="1",
        context_of_use="Synthetic research state discovery only",
        status="locked",
        model_sha256="b" * 64,
        code_sha256="c" * 64,
        data_snapshot_ids=["snapshot-synthetic-locked"],
        preprocessing_version="1",
        forbidden_uses=["clinical care"],
        validation_report_ids=["synthetic-state-validation"],
        expert_review_ids=model_review_ids,
        locked_at=datetime.now(timezone.utc),
    )
    validation = StateValidationReport(
        id="synthetic-state-validation",
        model_release_id=model.id,
        model_release_hash=model.content_hash(),
        candidate_state_ids=["synthetic-state-1"],
        data_snapshot_id="snapshot-synthetic-locked",
        data_snapshot_hash="f" * 64,
        partition_snapshot_id="synthetic-state-partition",
        partition_snapshot_hash="9" * 64,
        locked_test_patient_count=200,
        metrics_artifact_sha256="8" * 64,
        evaluation_code_sha256="7" * 64,
        expert_review_ids=state_review_ids,
        bootstrap_ari=0.80,
        calibration_slope=1.0,
        model_brier=0.18,
        baseline_brier=0.20,
        relative_brier_improvement=0.10,
        expert_confirmations=2,
        passed=True,
        locked_at=datetime.now(timezone.utc),
    )
    state = CandidateDiseaseState(
        id="synthetic-state-1",
        label="synthetic stable state",
        stability_ari=0.80,
        calibration_slope=1.0,
        relative_brier_improvement=0.10,
        expert_confirmations=2,
        model_release_id=model.id,
        model_release_hash=model.content_hash(),
        validation_report_id=validation.id,
        validation_report_hash=validation.content_hash(),
        expert_review_ids=state_review_ids,
        eligible_for_guideline=True,
    )
    model_reviews = [
        ExpertReview(
            id=model_review_ids[0],
            object_id=model.id,
            object_hash=model.content_hash(),
            reviewer_key="model-clinical-reviewer",
            role=ReviewRole.CRC_CLINICAL_EXPERT,
            decision=ReviewDecision.APPROVE,
            rationale="Independent synthetic clinical model review.",
        ),
        ExpertReview(
            id=model_review_ids[1],
            object_id=model.id,
            object_hash=model.content_hash(),
            reviewer_key="model-method-reviewer",
            role=ReviewRole.METHODOLOGIST,
            decision=ReviewDecision.APPROVE,
            rationale="Independent synthetic methodological model review.",
        ),
    ]
    state_reviews = [
        ExpertReview(
            id=review_id,
            object_id=state.id,
            object_hash=state.content_hash(),
            reviewer_key=f"state-clinical-reviewer-{index}",
            role=ReviewRole.CRC_CLINICAL_EXPERT,
            decision=ReviewDecision.APPROVE,
            rationale="Independent synthetic state interpretation review.",
        )
        for index, review_id in enumerate(state_review_ids, start=1)
    ]
    registry.register_candidate_state(
        state,
        model_release=model,
        validation_report=validation,
        state_reviews=state_reviews,
        model_reviews=model_reviews,
    )

    spec = TargetTrialSpec(
        id="synthetic-trial-1",
        name="synthetic locked target trial",
        study_id="study-synthetic-mcrc",
        cohort_snapshot_id="snapshot-synthetic-locked",
        cohort_snapshot_hash="f" * 64,
        data_snapshot_id="snapshot-synthetic-locked",
        analysis_code_sha256="a" * 64,
        analysis_config_sha256="b" * 64,
        baseline_release_id="baseline-release-synthetic",
        baseline_release_version=1,
        baseline_release_hash="3" * 64,
        baseline_recommendation_ids=["baseline-rec-synthetic"],
        baseline_recommendation_hashes={"baseline-rec-synthetic": "4" * 64},
        decision_point=DecisionPointType.FIRST_LINE_START,
        eligibility_criteria=["synthetic cohort"],
        strategies=["doublet", "alternative"],
        confounders=["ecog"],
        time_zero="first decision",
        action_assigned_at_column="treatment_assigned_at",
        assignment_procedure="conditional exchangeability",
        follow_up="12 months",
        follow_up_days=365,
        outcomes=["OS"],
        outcome_observed_at_columns={"OS": "os_observed_at"},
        causal_contrast="intention-to-treat analogue",
        estimand="risk difference",
        censoring_strategy="inverse weighting",
        analysis_plan="IPTW marginal and cross-fitted AIPW",
        ood_definition="locked feature support",
        negative_control_outcome="pre-treatment control",
        sensitivity_analyses=["E-value"],
        preregistered=True,
    )
    result_id = "synthetic-trial-result"
    input_snapshot = _trial_input(spec, "synthetic-input")
    effects = [
        CausalEffectEstimate(
            id=f"synthetic-effect-{index}",
            target_trial_id=spec.id,
            target_trial_result_id=result_id,
            target_trial_spec_hash=spec.content_hash(),
            strategy="doublet",
            comparator="alternative",
            outcome="OS",
            estimate=0.05,
            ci_lower=0.01,
            ci_upper=0.09,
            method=method,
            corroborating_methods=[other],
            eligible_patients=1200,
            strategy_patients=600,
            comparator_patients=600,
            common_support_fraction=0.90,
            strategy_effective_sample_size=500,
            comparator_effective_sample_size=500,
            maximum_absolute_smd=0.05,
            sensitivity_analysis={"passed": "true"},
            negative_controls_passed=True,
            directionally_consistent=True,
        )
        for index, (method, other) in enumerate(
            [
                ("cross_fitted_aipw", "iptw_marginal_structural_model"),
                ("iptw_marginal_structural_model", "cross_fitted_aipw"),
            ],
            start=1,
        )
    ]
    result = TargetTrialResult(
            id=result_id,
            spec=spec,
            effects=effects,
            eligible_patient_count=1200,
            common_support_fraction=0.90,
            maximum_absolute_smd=0.05,
            input_snapshot=input_snapshot,
            negative_control_audit=_causal_audit(
                spec,
                "negative_control",
                "synthetic-negative-control",
                input_snapshot,
            ),
            sensitivity_audits=[
                _causal_audit(
                    spec,
                    "sensitivity_analysis",
                    "synthetic-sensitivity",
                    input_snapshot,
                )
            ],
            locked_at=datetime.now(timezone.utc),
    )
    registry.register_causal_result(result)
    return state, result


def test_baseline_release_requires_committee_and_hash_bound_content():
    registry = GuidelineRegistry()
    recommendation = registry.register_recommendation(_recommendation())
    release = registry.create_release(
        name="Synthetic approved baseline",
        version="1",
        channel=GuidelineChannel.BASELINE_APPROVED,
        recommendation_ids=[recommendation.id],
        context_of_use="Research-only verification of computable guideline governance.",
    )
    _approve(registry, release.id, "expert-1", ReviewRole.CRC_CLINICAL_EXPERT)
    _approve(registry, release.id, "expert-2", ReviewRole.CRC_CLINICAL_EXPERT)
    with pytest.raises(GuidelineGovernanceError, match="expert committee"):
        registry.lock_release(release.id)
    _approve(registry, release.id, "method-1", ReviewRole.METHODOLOGIST)
    locked = registry.lock_release(release.id)
    assert locked.status == "locked"
    assert len(locked.expert_review_ids) == 3
    with pytest.raises(GuidelineGovernanceError, match="expert committee"):
        registry.publish_release(release.id, publisher_role="model_service")
    published = registry.publish_release(release.id, publisher_role="expert_committee")
    assert published.status == "released"
    assert published.automatic_order_allowed is False


def test_ai_candidate_never_promotes_to_baseline_and_needs_two_disciplines():
    registry = GuidelineRegistry()
    state, result = _register_candidate_evidence(registry)
    recommendation = registry.register_recommendation(
        _recommendation(
            identifier="rec-ai",
            channel=GuidelineChannel.AI_CANDIDATE,
            state=state,
            result=result,
        )
    )
    release = registry.create_release(
        name="Synthetic AI candidate",
        version="1",
        channel=GuidelineChannel.AI_CANDIDATE,
        recommendation_ids=[recommendation.id],
        context_of_use="Research hypothesis generation; never direct clinical care.",
    )
    _approve(registry, release.id, "expert-1", ReviewRole.CRC_CLINICAL_EXPERT)
    with pytest.raises(GuidelineGovernanceError, match="clinical and methodological"):
        registry.lock_release(release.id)
    _approve(registry, release.id, "method-1", ReviewRole.METHODOLOGIST)
    locked = registry.lock_release(release.id)
    assert locked.channel == "ai_candidate"
    assert locked.research_only is True
    with pytest.raises(GuidelineGovernanceError, match="automatic AI-candidate promotion"):
        registry.promote_ai_candidate_to_baseline(release.id)


def test_applicability_fails_closed_on_missing_ood_and_conflict():
    recommendation = _recommendation()
    applicable = evaluate_recommendation_applicability(
        recommendation,
        {
            "ecog": 1,
            "rass": "wild_type",
            "organ_failure": False,
            "timeline_conflict": False,
            "ood": False,
        },
    )
    assert applicable.applicable is True
    assert applicable.automatic_order_allowed is False

    missing = evaluate_recommendation_applicability(
        recommendation,
        {
            "ecog": 1,
            "organ_failure": False,
            "timeline_conflict": False,
        },
    )
    assert missing.abstained is True
    assert "MISSING_CRITICAL_FEATURE" in missing.abstention_reasons
    assert "OOD_STATUS_MISSING" in missing.abstention_reasons

    ood = evaluate_recommendation_applicability(
        recommendation,
        {
            "ecog": 1,
            "rass": "wild_type",
            "organ_failure": False,
            "timeline_conflict": False,
            "ood": True,
        },
    )
    assert ood.abstained is True
    assert "OUT_OF_DISTRIBUTION" in ood.abstention_reasons

    unsafe = evaluate_recommendation_applicability(
        recommendation,
        {
            "ecog": 1,
            "rass": "wild_type",
            "organ_failure": True,
            "timeline_conflict": False,
            "ood": False,
        },
    )
    assert unsafe.abstained is True
    assert "HARD_SAFETY_CRITERION_NOT_MET" in unsafe.abstention_reasons


def test_ai_candidate_rejects_unvalidated_state_and_effect_bindings():
    registry = GuidelineRegistry()
    with pytest.raises(Exception, match="promotion gate"):
        CandidateDiseaseState(
            id="unstable", label="unstable", eligible_for_guideline=True
        )
    with pytest.raises(GuidelineGovernanceError, match="unvalidated disease state"):
        registry.register_recommendation(
            _recommendation(identifier="rec-unbound", channel=GuidelineChannel.AI_CANDIDATE)
        )


def test_cpg_on_fhir_has_three_r4_resource_types_and_no_order_resource():
    artifacts = export_cpg_on_fhir(_recommendation())
    bundle = artifacts.as_bundle()
    resource_types = {
        entry["resource"]["resourceType"] for entry in bundle["entry"]
    }
    assert resource_types == {"PlanDefinition", "ActivityDefinition", "Library"}
    payload = json.dumps(bundle)
    assert "MedicationRequest" not in payload
    assert "ServiceRequest" not in payload
    assert '"valueBoolean": false' in payload
    assert artifacts.plan_definition["experimental"] is True
    assert all(
        action["condition"][0]["expression"]["expression"]
        == "RecommendationApplicable"
        for action in artifacts.plan_definition["action"]
    )
    cql_data = artifacts.library["content"][0]["data"]
    import base64

    assert 'define "RecommendationApplicable": false' in base64.b64decode(
        cql_data
    ).decode("utf-8")


def test_explicitly_conflicting_applicable_recommendations_force_abstention():
    registry = GuidelineRegistry()
    left = registry.register_recommendation(_recommendation(identifier="rec-left"))
    right = registry.register_recommendation(_recommendation(identifier="rec-right"))
    release = registry.create_release(
        name="Synthetic conflict fixture",
        version="1",
        channel=GuidelineChannel.BASELINE_APPROVED,
        recommendation_ids=[left.id, right.id],
        context_of_use="Research-only verification of conflict abstention behavior.",
    )
    _approve(registry, release.id, "expert-1", ReviewRole.CRC_CLINICAL_EXPERT)
    _approve(registry, release.id, "expert-2", ReviewRole.CRC_CLINICAL_EXPERT)
    _approve(registry, release.id, "safety-1", ReviewRole.SAFETY_REVIEWER)
    registry.lock_release(release.id)
    result = registry.evaluate(
        release.id,
        {
            "ecog": 1,
            "rass": "wild_type",
            "organ_failure": False,
            "timeline_conflict": False,
        },
        explicit_conflicts=[(left.id, right.id)],
    )
    assert result.abstained is True
    assert result.applicable_recommendation_ids == ()
    assert all(report.abstained for report in result.reports)
