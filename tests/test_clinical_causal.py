from __future__ import annotations

import numpy as np
import pytest

from ai_clinician.causal import (
    CausalAdmissionThresholds,
    CausalStrategyEngine,
    RESEARCH_ONLY_DISCLAIMER,
    validate_target_trial_spec,
)
from ai_clinician.models import DecisionPointType, TargetTrialSpec


def _spec() -> TargetTrialSpec:
    return TargetTrialSpec(
        id="trial-synthetic-first-line",
        name="Synthetic first-line target trial",
        study_id="study-synthetic-mcrc",
        cohort_snapshot_id="cohort-synthetic-locked",
        cohort_snapshot_hash="f" * 64,
        data_snapshot_id="cohort-synthetic-locked",
        analysis_code_sha256="a" * 64,
        analysis_config_sha256="b" * 64,
        baseline_release_id="baseline-release-synthetic",
        baseline_release_version=1,
        baseline_release_hash="3" * 64,
        baseline_recommendation_ids=["baseline-rec-synthetic"],
        baseline_recommendation_hashes={"baseline-rec-synthetic": "4" * 64},
        decision_point=DecisionPointType.FIRST_LINE_START,
        eligibility_criteria=["synthetic eligible cohort"],
        strategies=["strategy-A", "strategy-B"],
        confounders=["age_z", "ecog"],
        time_zero="first eligible mCRC treatment decision",
        action_assigned_at_column="treatment_assigned_at",
        assignment_grace_period_days=0,
        assignment_procedure="emulate random assignment conditional on confounders",
        follow_up="12 months from time zero",
        follow_up_days=365,
        outcomes=["os_12m", "toxicity"],
        outcome_observed_at_columns={
            "os_12m": "os_12m_observed_at",
            "toxicity": "toxicity_observed_at",
        },
        causal_contrast="intention-to-treat analogue",
        estimand="risk difference at 12 months",
        censoring_strategy="inverse probability of censoring weighting",
        analysis_plan="IPTW marginal structural model and cross-fitted AIPW",
        ood_definition="pre-specified feature support detector",
        negative_control_outcome="synthetic pre-treatment outcome",
        sensitivity_analyses=["E-value", "propensity truncation"],
        preregistered=True,
    )


def _overlapping_rows(n: int = 1_600) -> list[dict[str, object]]:
    rng = np.random.default_rng(2026)
    age_z = rng.normal(size=n)
    ecog = rng.binomial(2, 0.25, size=n).astype(float)
    propensity = 1 / (1 + np.exp(-(-0.15 + 0.25 * age_z - 0.15 * ecog)))
    treated = rng.binomial(1, propensity)
    response_probability = 1 / (
        1 + np.exp(-(-0.7 + 0.55 * treated - 0.2 * age_z - 0.25 * ecog))
    )
    os_12m = rng.binomial(1, response_probability)
    toxicity = rng.binomial(
        1, 1 / (1 + np.exp(-(-1.8 + 0.4 * treated + 0.25 * ecog)))
    )
    return [
        {
            "patient_id": f"synthetic-{index:05d}",
            "treatment_id": "strategy-A" if treated[index] else "strategy-B",
            "age_z": float(age_z[index]),
            "ecog": float(ecog[index]),
            "os_12m": float(os_12m[index]),
            "toxicity": float(toxicity[index]),
            "is_ood": False,
            "decision_time": "2025-01-15T00:00:00Z",
            "age_z_available_at": "2025-01-01T00:00:00Z",
            "ecog_available_at": "2025-01-14T00:00:00Z",
            "treatment_assigned_at": "2025-01-15T00:00:00Z",
            "os_12m_observed_at": "2025-12-31T00:00:00Z",
            "toxicity_observed_at": "2025-02-15T00:00:00Z",
        }
        for index in range(n)
    ]


def test_target_trial_runs_iptw_and_cross_fitted_dr_without_efficacy_claim():
    spec = _spec()
    validate_target_trial_spec(spec)
    result = CausalStrategyEngine(cross_fit_folds=3, random_seed=12).run(
        spec,
        _overlapping_rows(),
        outcome_column="os_12m",
        metric_directions={"os_12m": "maximize", "toxicity": "minimize"},
        negative_control_passed=True,
        sensitivity_analysis_passed=True,
    )
    assert result.abstained is False
    assert result.admission.admitted is True
    assert result.admission.common_support_coverage >= 0.8
    assert result.admission.maximum_abs_weighted_smd < 0.10
    assert {effect.method for effect in result.effects} == {
        "cross_fitted_aipw",
        "iptw_marginal_structural_model",
    }
    assert all(effect.ci_lower < effect.ci_upper for effect in result.effects)
    assert all(effect.core_model is not None for effect in result.effects)
    assert all(effect.core_model.clinical_efficacy_claim is False for effect in result.effects)
    assert all(effect.causal_claim_permitted is False for effect in result.effects)
    assert result.disclaimer == RESEARCH_ONLY_DISCLAIMER
    assert result.pareto_options
    assert all(option.is_order is False for option in result.pareto_options)
    assert any(option.pareto_optimal for option in result.pareto_options)


def test_minimum_action_count_gate_abstains_before_estimation():
    rows = _overlapping_rows(50)
    result = CausalStrategyEngine().run(
        _spec(),
        rows,
        outcome_column="os_12m",
        negative_control_passed=True,
        sensitivity_analysis_passed=True,
    )
    assert result.abstained is True
    assert "INSUFFICIENT_RAW_ACTION_COUNT" in result.abstention_reasons
    assert result.effects == ()
    assert result.pareto_options == ()


def test_positivity_failure_forces_abstention_and_no_treatment_options():
    rows: list[dict[str, object]] = []
    for index in range(1_200):
        treated = index >= 600
        rows.append(
            {
                "patient_id": f"positivity-{index:04d}",
                "treatment_id": "strategy-A" if treated else "strategy-B",
                "age_z": 10.0 if treated else -10.0,
                "ecog": 2.0 if treated else 0.0,
                "os_12m": float(index % 3 != 0),
                "toxicity": float(index % 7 == 0),
                "is_ood": False,
                "decision_time": "2025-01-15T00:00:00Z",
                "age_z_available_at": "2025-01-01T00:00:00Z",
                "ecog_available_at": "2025-01-14T00:00:00Z",
                "treatment_assigned_at": "2025-01-15T00:00:00Z",
                "os_12m_observed_at": "2025-12-31T00:00:00Z",
                "toxicity_observed_at": "2025-02-15T00:00:00Z",
            }
        )
    result = CausalStrategyEngine(
        cross_fit_folds=3, random_seed=3
    ).run(
        _spec(),
        rows,
        outcome_column="os_12m",
        negative_control_passed=True,
        sensitivity_analysis_passed=True,
    )
    assert result.abstained is True
    assert "POSITIVITY_OR_COMMON_SUPPORT_FAILURE" in result.abstention_reasons
    assert result.effects == ()
    assert result.pareto_options == ()


def test_wide_confidence_interval_and_ood_each_force_abstention():
    rows = _overlapping_rows(1_600)
    permissive_except_width = CausalAdmissionThresholds(
        maximum_ci_width=0.001,
    )
    wide = CausalStrategyEngine(
        thresholds=permissive_except_width, cross_fit_folds=3
    ).run(
        _spec(),
        rows,
        outcome_column="os_12m",
        negative_control_passed=True,
        sensitivity_analysis_passed=True,
    )
    assert wide.abstained is True
    assert "CONFIDENCE_INTERVAL_TOO_WIDE" in wide.abstention_reasons
    assert wide.effects == ()
    assert wide.pareto_options == ()

    for row in rows:
        row["is_ood"] = True
    ood = CausalStrategyEngine(
        thresholds=permissive_except_width, cross_fit_folds=3
    ).run(
        _spec(),
        rows,
        outcome_column="os_12m",
        ood_column="is_ood",
        negative_control_passed=True,
        sensitivity_analysis_passed=True,
    )
    assert ood.abstained is True
    assert "OUT_OF_DISTRIBUTION_INPUT" in ood.abstention_reasons


def test_missing_ood_assessment_forces_abstention() -> None:
    rows = _overlapping_rows()
    for row in rows:
        row.pop("is_ood")
    result = CausalStrategyEngine().run(
        _spec(),
        rows,
        outcome_column="os_12m",
        negative_control_passed=True,
        sensitivity_analysis_passed=True,
    )
    assert result.abstained is True
    assert result.effects == ()
    assert "OOD_STATUS_MISSING" in result.abstention_reasons


@pytest.mark.parametrize(
    ("mutation", "expected_reason"),
    [
        (lambda row: row.pop("ecog_available_at"), "CONFOUNDER_AVAILABILITY_MISSING_OR_INVALID"),
        (
            lambda row: row.__setitem__(
                "age_z_available_at", "2025-01-16T00:00:00Z"
            ),
            "POST_TIME_ZERO_CONFOUNDER",
        ),
        (lambda row: row.pop("decision_time"), "TIME_ZERO_PROVENANCE_MISSING_OR_INVALID"),
    ],
)
def test_time_zero_provenance_failures_force_abstention(mutation, expected_reason):
    rows = _overlapping_rows()
    mutation(rows[0])
    result = CausalStrategyEngine().run(
        _spec(),
        rows,
        outcome_column="os_12m",
        negative_control_passed=True,
        sensitivity_analysis_passed=True,
    )
    assert result.abstained is True
    assert result.effects == ()
    assert expected_reason in result.abstention_reasons


@pytest.mark.parametrize(
    ("column", "value", "expected_reason"),
    [
        (
            "treatment_assigned_at",
            "2025-02-01T00:00:00Z",
            "ACTION_ASSIGNMENT_OUTSIDE_TIME_ZERO_WINDOW",
        ),
        (
            "os_12m_observed_at",
            "2024-12-01T00:00:00Z",
            "OUTCOME_OBSERVATION_OUTSIDE_FOLLOW_UP_WINDOW",
        ),
        (
            "os_12m_observed_at",
            "2026-02-01T00:00:00Z",
            "OUTCOME_OBSERVATION_OUTSIDE_FOLLOW_UP_WINDOW",
        ),
    ],
)
def test_assignment_and_outcome_times_are_bound_to_time_zero(
    column, value, expected_reason
):
    rows = _overlapping_rows()
    rows[0][column] = value
    result = CausalStrategyEngine().run(
        _spec(),
        rows,
        outcome_column="os_12m",
        negative_control_passed=True,
        sensitivity_analysis_passed=True,
    )
    assert result.abstained is True
    assert result.effects == ()
    assert expected_reason in result.abstention_reasons


def test_non_preregistered_trial_is_rejected():
    unlocked = _spec().model_copy(update={"preregistered": False})
    with pytest.raises(ValueError, match="preregistered"):
        validate_target_trial_spec(unlocked)


def test_causal_admission_thresholds_cannot_be_weakened():
    with pytest.raises(ValueError, match="cannot be below 1000"):
        CausalAdmissionThresholds(minimum_eligible_patients=999)
    with pytest.raises(ValueError, match="OOD tolerance"):
        CausalAdmissionThresholds(maximum_ood_fraction=0.01)
