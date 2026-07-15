"""Strict public contracts for the research-only clinical guideline system."""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from datetime import datetime, timezone
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, str_strip_whitespace=True)


class GuidelineChannel(StrEnum):
    BASELINE_APPROVED = "baseline_approved"
    AI_CANDIDATE = "ai_candidate"


class LifecycleStatus(StrEnum):
    DRAFT = "draft"
    DATA_REGISTERED = "data_registered"
    TIMELINE_VALIDATED = "timeline_validated"
    COHORT_LOCKED = "cohort_locked"
    STATES_DISCOVERED = "states_discovered"
    TARGET_TRIALS_COMPLETED = "target_trials_completed"
    EXPERT_REVIEW = "expert_review"
    CANDIDATE_GUIDELINE = "candidate_guideline"
    RESEARCH_RELEASE = "research_release"
    RETIRED = "retired"


MVP_DISABLED_LIFECYCLE_STATES: frozenset[LifecycleStatus] = frozenset(
    {
        LifecycleStatus.TARGET_TRIALS_COMPLETED,
        LifecycleStatus.CANDIDATE_GUIDELINE,
        LifecycleStatus.RESEARCH_RELEASE,
    }
)


class ReleaseStatus(StrEnum):
    DRAFT = "draft"
    IN_REVIEW = "in_review"
    LOCKED = "locked"
    RELEASED = "released"
    RETIRED = "retired"


class Jurisdiction(StrEnum):
    CN = "CN"
    US = "US"
    INTERNATIONAL = "INTERNATIONAL"


class EvidenceCertainty(StrEnum):
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    VERY_LOW = "very_low"
    NOT_ASSESSED = "not_assessed"


class RecommendationStrength(StrEnum):
    STRONG = "strong"
    CONDITIONAL = "conditional"
    RESEARCH_ONLY = "research_only"


class ReviewDecision(StrEnum):
    APPROVE = "approve"
    REJECT = "reject"
    REQUEST_CHANGES = "request_changes"


class ReviewRole(StrEnum):
    CRC_CLINICAL_EXPERT = "crc_clinical_expert"
    METHODOLOGIST = "methodologist"
    DATA_STEWARD = "data_steward"
    SAFETY_REVIEWER = "safety_reviewer"
    PATIENT_REPRESENTATIVE = "patient_representative"


class EventType(StrEnum):
    DIAGNOSIS = "diagnosis"
    METASTASIS = "metastasis"
    ECOG = "ecog"
    PATHOLOGY_STAGE = "pathology_stage"
    MOLECULAR_TEST = "molecular_test"
    LABORATORY = "laboratory"
    SURGERY = "surgery"
    TREATMENT_START = "treatment_start"
    TREATMENT_END = "treatment_end"
    RESPONSE = "response"
    PROGRESSION = "progression"
    TOXICITY = "toxicity"
    FOLLOW_UP = "follow_up"
    DEATH = "death"


class TimePrecision(StrEnum):
    UNKNOWN = "unknown"
    YEAR = "year"
    MONTH = "month"
    DAY = "day"
    DATETIME = "datetime"


class DecisionPointType(StrEnum):
    FIRST_LINE_START = "first_line_start"
    RESPONSE_OR_PROGRESSION_REASSESSMENT = "response_or_progression_reassessment"
    SECOND_LINE_START = "second_line_start"
    THIRD_LINE_START = "third_line_start"


class ModelStatus(StrEnum):
    DEVELOPMENT = "development"
    LOCKED = "locked"
    RELEASED = "released"
    RETIRED = "retired"


Scalar = str | int | float | bool


class EvidenceSpan(StrictModel):
    source_document_id: str = Field(min_length=3)
    sheet_name: str | None = None
    cell_ref: str | None = None
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    text_sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    excerpt: str | None = Field(default=None, exclude=True, repr=False)

    @model_validator(mode="after")
    def valid_offsets(self) -> "EvidenceSpan":
        if self.end <= self.start:
            raise ValueError("evidence span end must follow start")
        return self


class ClinicalEvent(StrictModel):
    id: str = Field(default_factory=lambda: new_id("event"))
    patient_key: str = Field(min_length=3)
    event_type: EventType
    event_time: datetime | None = None
    time_precision: TimePrecision = TimePrecision.UNKNOWN
    code: str | None = None
    value: Scalar | None = None
    unit: str | None = None
    confidence: float = Field(ge=0, le=1)
    evidence_spans: list[EvidenceSpan] = Field(min_length=1)
    source_available_at: datetime | None = None
    conflicts: list[str] = Field(default_factory=list)
    missing_flags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def time_contract(self) -> "ClinicalEvent":
        if self.event_time is None and self.time_precision is not TimePrecision.UNKNOWN:
            raise ValueError("time precision must be unknown when event_time is missing")
        if self.event_time is not None and self.event_time.tzinfo is None:
            self.event_time = self.event_time.replace(tzinfo=timezone.utc)
        if self.source_available_at is not None and self.source_available_at.tzinfo is None:
            self.source_available_at = self.source_available_at.replace(tzinfo=timezone.utc)
        return self


class PatientTimeline(StrictModel):
    patient_key: str = Field(min_length=3)
    events: list[ClinicalEvent] = Field(default_factory=list)
    completeness: float = Field(ge=0, le=1)
    conflicts: list[str] = Field(default_factory=list)
    leakage_flags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def one_patient_only(self) -> "PatientTimeline":
        if any(event.patient_key != self.patient_key for event in self.events):
            raise ValueError("a timeline cannot contain another patient's event")
        return self


class DataProfileRegistration(StrictModel):
    id: str = Field(default_factory=lambda: new_id("data_profile"))
    source_snapshot_sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    workbook_count: int = Field(ge=1)
    sheet_count: int = Field(ge=1)
    aggregate_patient_count: int = Field(ge=1)
    quarantined_row_count: int = Field(ge=0)
    registered_at: datetime = Field(default_factory=utcnow)
    aggregate_only: Literal[True] = True


class CohortSnapshot(StrictModel):
    id: str = Field(default_factory=lambda: new_id("cohort"))
    study_id: str = Field(min_length=3)
    data_profile_id: str = Field(min_length=3)
    source_data_sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    patient_membership_sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    timeline_snapshot_sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    split_manifest_sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    development_membership_sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    tuning_membership_sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    locked_test_membership_sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    development_patient_count: int = Field(ge=1)
    tuning_patient_count: int = Field(ge=1)
    locked_test_patient_count: int = Field(ge=1)
    eligible_patient_count: int = Field(ge=3)
    locked_at: datetime
    research_only: Literal[True] = True

    @model_validator(mode="after")
    def exact_temporal_partition(self) -> "CohortSnapshot":
        total = self.eligible_patient_count
        development = (total * 70) // 100
        tuning = (total * 15) // 100
        if total >= 3:
            development = max(1, min(development, total - 2))
            tuning = max(1, min(tuning, total - development - 1))
        expected = (development, tuning, total - development - tuning)
        observed = (
            self.development_patient_count,
            self.tuning_patient_count,
            self.locked_test_patient_count,
        )
        if observed != expected:
            raise ValueError("cohort snapshot must use the fixed patient-level 70/15/15 split")
        if len(
            {
                self.development_membership_sha256,
                self.tuning_membership_sha256,
                self.locked_test_membership_sha256,
            }
        ) != 3:
            raise ValueError("cohort partition membership hashes must be distinct")
        return self

    def content_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude={"locked_at"})
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class TimelineGateThresholds(StrictModel):
    minimum_gold_total_cases: int = Field(default=500, ge=500)
    minimum_reference_cases: int = Field(default=100, ge=100)
    minimum_event_precision: float = Field(default=0.90, ge=0.90, le=1)
    minimum_event_recall: float = Field(default=0.90, ge=0.90, le=1)
    minimum_critical_event_f1: float = Field(default=0.90, ge=0.90, le=1)
    minimum_treatment_line_macro_f1: float = Field(default=0.90, ge=0.90, le=1)
    minimum_date_within_14_days: float = Field(default=0.90, ge=0.90, le=1)
    minimum_cohen_kappa: float = Field(default=0.80, ge=0.80, le=1)
    maximum_future_leakage_count: Literal[0] = 0


class TimelineValidationManifestRegistration(StrictModel):
    id: str = Field(default_factory=lambda: new_id("timeline_manifest"))
    gold_snapshot_id: str = Field(min_length=3)
    gold_snapshot_sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    decision_view_manifest_sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    locked_split_sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    gold_total_case_count: int = Field(ge=500)
    locked_case_count: int = Field(ge=100)
    annotation_pair_count: int = Field(ge=500)
    annotation_kappa: float = Field(ge=-1, le=1)
    extractor_release_id: str = Field(min_length=3)
    extractor_sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    evaluation_code_sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    locked: Literal[True] = True
    registered_at: datetime = Field(default_factory=utcnow)
    research_only: Literal[True] = True

    @model_validator(mode="after")
    def exact_locked_fraction(self) -> "TimelineValidationManifestRegistration":
        if self.locked_case_count * 5 != self.gold_total_case_count:
            raise ValueError("timeline validation manifest must lock exactly 20%")
        return self

    def content_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude={"registered_at"})
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class TimelineQualityReport(StrictModel):
    id: str = Field(default_factory=lambda: new_id("timeline_quality"))
    gold_total_case_count: int = Field(ge=0)
    reference_case_count: int = Field(ge=0)
    critical_event_counts: dict[str, int]
    decision_view_patient_count: int = Field(ge=0)
    decision_view_count: int = Field(ge=0)
    decision_view_failure_count: int = Field(ge=0)
    critical_event_precision: float = Field(ge=0, le=1)
    critical_event_recall: float = Field(ge=0, le=1)
    critical_event_f1: float = Field(ge=0, le=1)
    treatment_line_macro_f1: float = Field(ge=0, le=1)
    date_within_14_days: float = Field(ge=0, le=1)
    cohen_kappa: float = Field(ge=-1, le=1)
    model_gold_kappa: float = Field(ge=-1, le=1)
    future_leakage_count: int = Field(ge=0)
    thresholds: TimelineGateThresholds = Field(default_factory=TimelineGateThresholds)
    gold_snapshot_id: str | None = None
    gold_snapshot_sha256: str | None = Field(
        default=None, pattern=r"^[a-fA-F0-9]{64}$"
    )
    locked_split_sha256: str | None = Field(
        default=None, pattern=r"^[a-fA-F0-9]{64}$"
    )
    locked_fraction: float | None = Field(default=None, ge=0, le=1)
    decision_view_manifest_sha256: str | None = Field(
        default=None, pattern=r"^[a-fA-F0-9]{64}$"
    )
    extractor_release_id: str | None = None
    extractor_sha256: str | None = Field(
        default=None, pattern=r"^[a-fA-F0-9]{64}$"
    )
    evaluation_code_sha256: str | None = Field(
        default=None, pattern=r"^[a-fA-F0-9]{64}$"
    )
    manifest_registration_id: str | None = None
    manifest_registration_hash: str | None = Field(
        default=None, pattern=r"^[a-fA-F0-9]{64}$"
    )
    annotation_review_ids: list[str] = Field(default_factory=list)
    evaluated_at: datetime = Field(default_factory=utcnow)
    passed: bool
    failures: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def pass_matches_failures(self) -> "TimelineQualityReport":
        if set(self.critical_event_counts) != {"treatment_line", "progression", "death"}:
            raise ValueError("critical event counts must cover every hard-gated category")
        deterministic_pass = all(
            (
                self.reference_case_count >= self.thresholds.minimum_reference_cases,
                self.gold_total_case_count >= self.thresholds.minimum_gold_total_cases,
                self.reference_case_count * 5 == self.gold_total_case_count,
                all(value > 0 for value in self.critical_event_counts.values()),
                self.decision_view_patient_count >= self.reference_case_count,
                self.decision_view_count >= self.decision_view_patient_count,
                self.decision_view_failure_count == 0,
                self.critical_event_precision >= self.thresholds.minimum_event_precision,
                self.critical_event_recall >= self.thresholds.minimum_event_recall,
                self.critical_event_f1
                >= self.thresholds.minimum_critical_event_f1,
                self.treatment_line_macro_f1
                >= self.thresholds.minimum_treatment_line_macro_f1,
                self.date_within_14_days
                >= self.thresholds.minimum_date_within_14_days,
                self.cohen_kappa >= self.thresholds.minimum_cohen_kappa,
                self.future_leakage_count
                <= self.thresholds.maximum_future_leakage_count,
                bool(self.gold_snapshot_id),
                bool(self.gold_snapshot_sha256),
                bool(self.locked_split_sha256),
                self.locked_fraction == 0.20,
                bool(self.decision_view_manifest_sha256),
                bool(self.extractor_release_id),
                bool(self.extractor_sha256),
                bool(self.evaluation_code_sha256),
                bool(self.manifest_registration_id),
                bool(self.manifest_registration_hash),
                len(set(self.annotation_review_ids)) >= 3,
            )
        )
        if self.passed is not deterministic_pass:
            raise ValueError("timeline pass flag does not match fixed deterministic gates")
        if self.passed == bool(self.failures):
            raise ValueError("passed must be true exactly when failures is empty")
        return self

    def provenance_complete(self) -> bool:
        return all(
            (
                self.gold_snapshot_id,
                self.gold_snapshot_sha256,
                self.locked_split_sha256,
                self.locked_fraction == 0.20,
                self.decision_view_manifest_sha256,
                self.extractor_release_id,
                self.extractor_sha256,
                self.evaluation_code_sha256,
                self.manifest_registration_id,
                self.manifest_registration_hash,
                len(set(self.annotation_review_ids)) >= 3,
            )
        )

    def content_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude={"evaluated_at"})
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class DecisionPoint(StrictModel):
    id: str = Field(default_factory=lambda: new_id("decision"))
    patient_key: str = Field(min_length=3)
    decision_type: DecisionPointType
    decision_time: datetime
    observed_features: dict[str, Scalar | None] = Field(default_factory=dict)
    eligible_actions: list[str] = Field(default_factory=list)
    missing_critical_features: list[str] = Field(default_factory=list)
    ood: bool = False
    abstained: bool = False
    abstention_reasons: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def enforce_abstention(self) -> "DecisionPoint":
        required = self.ood or bool(self.missing_critical_features)
        if required and not self.abstained:
            raise ValueError("OOD or missing critical features require abstention")
        if self.abstained and not self.abstention_reasons:
            raise ValueError("abstention requires reasons")
        return self


class CandidateDiseaseState(StrictModel):
    id: str = Field(default_factory=lambda: new_id("state"))
    label: str = Field(min_length=1)
    description: str = ""
    feature_centroid: dict[str, float] = Field(default_factory=dict)
    feature_names: list[str] = Field(default_factory=list)
    patient_count: int = Field(default=0, ge=0)
    observation_count: int = Field(default=0, ge=0)
    source: str = "research_state_discovery"
    continuous_score: float | None = None
    stability_ari: float | None = Field(default=None, ge=-1, le=1)
    calibration_slope: float | None = None
    relative_brier_improvement: float | None = None
    expert_confirmations: int = Field(default=0, ge=0)
    model_release_id: str | None = None
    model_release_hash: str | None = Field(
        default=None, pattern=r"^[a-fA-F0-9]{64}$"
    )
    validation_report_id: str | None = None
    validation_report_hash: str | None = Field(
        default=None, pattern=r"^[a-fA-F0-9]{64}$"
    )
    expert_review_ids: list[str] = Field(default_factory=list)
    research_only: Literal[True] = True
    eligible_for_guideline: bool = False

    @model_validator(mode="after")
    def promotion_gate(self) -> "CandidateDiseaseState":
        if self.eligible_for_guideline:
            checks = [
                self.stability_ari is not None and self.stability_ari >= 0.75,
                self.calibration_slope is not None and 0.8 <= self.calibration_slope <= 1.2,
                self.relative_brier_improvement is not None
                and self.relative_brier_improvement >= 0.05,
                self.expert_confirmations >= 2,
                bool(self.model_release_id),
                bool(self.model_release_hash),
                bool(self.validation_report_id),
                bool(self.validation_report_hash),
                len(set(self.expert_review_ids)) >= 2,
            ]
            if not all(checks):
                raise ValueError("candidate state has not passed every promotion gate")
        return self

    def content_hash(self) -> str:
        payload = self.model_dump(
            mode="json",
            exclude={"expert_confirmations", "expert_review_ids", "eligible_for_guideline"},
        )
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class StateValidationReport(StrictModel):
    id: str = Field(default_factory=lambda: new_id("state_validation"))
    model_release_id: str = Field(min_length=3)
    model_release_hash: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    candidate_state_ids: list[str] = Field(min_length=1)
    data_snapshot_id: str = Field(min_length=3)
    data_snapshot_hash: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    partition_snapshot_id: str = Field(min_length=3)
    partition_snapshot_hash: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    locked_test_patient_count: int = Field(ge=1)
    metrics_artifact_sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    evaluation_code_sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    expert_review_ids: list[str] = Field(min_length=2)
    bootstrap_ari: float
    calibration_slope: float
    model_brier: float = Field(ge=0)
    baseline_brier: float = Field(gt=0)
    relative_brier_improvement: float
    expert_confirmations: int = Field(ge=0)
    passed: bool
    locked_at: datetime

    @model_validator(mode="after")
    def deterministic_gate(self) -> "StateValidationReport":
        measured_improvement = (
            self.baseline_brier - self.model_brier
        ) / self.baseline_brier
        if not math.isclose(
            self.relative_brier_improvement,
            measured_improvement,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "relative Brier improvement must be derived from the locked scores"
            )
        expected = all(
            (
                self.bootstrap_ari >= 0.75,
                0.8 <= self.calibration_slope <= 1.2,
                measured_improvement >= 0.05,
                self.expert_confirmations >= 2,
            )
        )
        if self.passed is not expected:
            raise ValueError("state validation pass flag does not match fixed thresholds")
        if len(set(self.candidate_state_ids)) != len(self.candidate_state_ids):
            raise ValueError("candidate state ids must be unique")
        if len(set(self.expert_review_ids)) != len(self.expert_review_ids):
            raise ValueError("state validation expert review ids must be unique")
        if self.expert_confirmations != len(set(self.expert_review_ids)):
            raise ValueError("expert confirmations must be review-bound")
        return self

    def content_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude={"locked_at"})
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class StatePartitionSnapshot(StrictModel):
    id: str = Field(default_factory=lambda: new_id("state_partition"))
    cohort_snapshot_id: str = Field(min_length=3)
    cohort_snapshot_hash: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    analysis_manifest_registration_id: str = Field(min_length=3)
    analysis_manifest_registration_hash: str = Field(
        pattern=r"^[a-fA-F0-9]{64}$"
    )
    analytic_rows_sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    development_membership_sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    tuning_membership_sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    locked_test_membership_sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    development_patient_count: int = Field(ge=1)
    tuning_patient_count: int = Field(ge=1)
    locked_test_patient_count: int = Field(ge=1)
    code_sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    locked_at: datetime
    research_only: Literal[True] = True

    def content_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude={"locked_at"})
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class StateTransition(StrictModel):
    source_state_id: str
    target_state_id: str
    transition_count: int = Field(ge=0)
    person_time_days: float = Field(ge=0)
    rate_per_day: float = Field(ge=0)
    treatment_conditioned: Literal[False] = False


class TargetTrialSpec(StrictModel):
    id: str = Field(default_factory=lambda: new_id("trial"))
    name: str = Field(min_length=3)
    study_id: str = Field(min_length=3)
    cohort_snapshot_id: str = Field(min_length=3)
    cohort_snapshot_hash: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    data_snapshot_id: str = Field(min_length=3)
    analysis_code_sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    analysis_config_sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    baseline_release_id: str = Field(min_length=3)
    baseline_release_version: int = Field(ge=1)
    baseline_release_hash: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    baseline_recommendation_ids: list[str] = Field(min_length=1)
    baseline_recommendation_hashes: dict[str, str] = Field(min_length=1)
    decision_point: DecisionPointType
    eligibility_criteria: list[str] = Field(min_length=1)
    strategies: list[str] = Field(min_length=2)
    confounders: list[str] = Field(min_length=1)
    time_zero: str = Field(min_length=3)
    action_assigned_at_column: str = Field(min_length=1)
    assignment_grace_period_days: int = Field(default=0, ge=0, le=14)
    assignment_procedure: str = Field(min_length=3)
    follow_up: str = Field(min_length=3)
    follow_up_days: int = Field(gt=0)
    outcomes: list[str] = Field(min_length=1)
    outcome_observed_at_columns: dict[str, str] = Field(min_length=1)
    causal_contrast: str = Field(min_length=3)
    estimand: str = Field(min_length=3)
    censoring_strategy: str = Field(min_length=3)
    analysis_plan: str = Field(min_length=3)
    ood_definition: str = Field(min_length=3)
    negative_control_outcome: str = Field(min_length=3)
    sensitivity_analyses: list[str] = Field(min_length=1)
    preregistered: bool = False

    @model_validator(mode="after")
    def binary_locked_contrast(self) -> "TargetTrialSpec":
        if len(self.strategies) != 2 or len(set(self.strategies)) != 2:
            raise ValueError("MVP target trials require exactly two distinct strategies")
        if len(set(self.confounders)) != len(self.confounders):
            raise ValueError("pre-specified confounders must be unique")
        if set(self.baseline_recommendation_ids) != set(
            self.baseline_recommendation_hashes
        ):
            raise ValueError("baseline recommendation ids require content hashes")
        if set(self.outcomes) != set(self.outcome_observed_at_columns):
            raise ValueError("every outcome requires a pre-specified observation-time column")
        if len(set(self.outcome_observed_at_columns.values())) != len(
            self.outcome_observed_at_columns
        ):
            raise ValueError("outcome observation-time columns must be unique")
        return self

    def content_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(self.model_dump(mode="json"), sort_keys=True).encode()
        ).hexdigest()


class CausalEffectEstimate(StrictModel):
    id: str = Field(default_factory=lambda: new_id("effect"))
    target_trial_id: str
    target_trial_result_id: str | None = None
    target_trial_spec_hash: str | None = Field(
        default=None, pattern=r"^[a-fA-F0-9]{64}$"
    )
    strategy: str
    comparator: str
    outcome: str
    estimate: float | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None
    method: str
    corroborating_methods: list[str] = Field(default_factory=list)
    eligible_patients: int = Field(ge=0)
    strategy_patients: int = Field(ge=0)
    comparator_patients: int = Field(ge=0)
    common_support_fraction: float = Field(ge=0, le=1)
    strategy_effective_sample_size: float = Field(ge=0)
    comparator_effective_sample_size: float = Field(ge=0)
    maximum_absolute_smd: float = Field(ge=0)
    sensitivity_analysis: dict[str, float | str] = Field(default_factory=dict)
    negative_controls_passed: bool = False
    directionally_consistent: bool = False
    abstained: bool = False
    abstention_reasons: list[str] = Field(default_factory=list)
    research_only: Literal[True] = True
    clinical_efficacy_claim: Literal[False] = False

    @model_validator(mode="after")
    def estimate_contract(self) -> "CausalEffectEstimate":
        if self.abstained:
            if not self.abstention_reasons:
                raise ValueError("abstained estimates require reasons")
            if any(value is not None for value in (self.estimate, self.ci_lower, self.ci_upper)):
                raise ValueError("abstained estimates cannot report an effect")
        else:
            if None in (self.estimate, self.ci_lower, self.ci_upper):
                raise ValueError("non-abstained estimates require an effect and 95% CI")
            assert self.ci_lower is not None and self.ci_upper is not None
            if self.ci_lower > self.ci_upper:
                raise ValueError("confidence interval is reversed")
        return self

    def content_hash(self) -> str:
        payload = self.model_dump(mode="json")
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class CausalAuditArtifact(StrictModel):
    id: str = Field(default_factory=lambda: new_id("causal_audit"))
    target_trial_spec_id: str = Field(min_length=3)
    target_trial_spec_hash: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    data_snapshot_id: str = Field(min_length=3)
    input_snapshot_id: str = Field(min_length=3)
    input_snapshot_hash: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    audit_type: Literal["negative_control", "sensitivity_analysis"]
    method: str = Field(min_length=3)
    result_sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    code_sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    config_sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    passed: bool
    created_at: datetime = Field(default_factory=utcnow)
    research_only: Literal[True] = True

    def content_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude={"created_at"})
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class NegativeControlAuditResult(StrictModel):
    method: str = Field(min_length=3)
    estimate: float
    ci_lower: float
    ci_upper: float
    null_value: float = 0.0
    equivalence_margin: float = Field(gt=0)
    passed: bool

    @model_validator(mode="after")
    def deterministic_pass(self) -> "NegativeControlAuditResult":
        if self.ci_lower > self.ci_upper:
            raise ValueError("negative-control confidence interval is reversed")
        expected = (
            self.ci_lower <= self.null_value <= self.ci_upper
            and abs(self.estimate - self.null_value) <= self.equivalence_margin
        )
        if self.passed is not expected:
            raise ValueError("negative-control pass flag does not match fixed rule")
        return self


class SensitivityAuditResult(StrictModel):
    method: str = Field(min_length=3)
    primary_effect_direction: Literal[-1, 1]
    sensitivity_effect_direction: Literal[-1, 1]
    robustness_value: float = Field(ge=0)
    minimum_robustness_value: float = Field(gt=0)
    passed: bool

    @model_validator(mode="after")
    def deterministic_pass(self) -> "SensitivityAuditResult":
        expected = (
            self.primary_effect_direction == self.sensitivity_effect_direction
            and self.robustness_value >= self.minimum_robustness_value
        )
        if self.passed is not expected:
            raise ValueError("sensitivity pass flag does not match fixed rule")
        return self


class TargetTrialInputSnapshot(StrictModel):
    id: str = Field(default_factory=lambda: new_id("trial_input"))
    target_trial_spec_id: str = Field(min_length=3)
    target_trial_spec_hash: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    cohort_snapshot_id: str = Field(min_length=3)
    cohort_snapshot_hash: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    analytic_rows_sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    patient_membership_sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    eligible_patient_count: int = Field(ge=1_000)
    created_at: datetime = Field(default_factory=utcnow)
    research_only: Literal[True] = True

    def content_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude={"created_at"})
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class TargetTrialResult(StrictModel):
    id: str = Field(default_factory=lambda: new_id("trial_result"))
    spec: TargetTrialSpec
    effects: list[CausalEffectEstimate] = Field(min_length=2)
    eligible_patient_count: int = Field(ge=1_000)
    common_support_fraction: float = Field(ge=0.80, le=1)
    maximum_absolute_smd: float = Field(ge=0, lt=0.10)
    ood_fraction: Literal[0.0] = 0.0
    input_snapshot: TargetTrialInputSnapshot
    negative_control_audit: CausalAuditArtifact
    sensitivity_audits: list[CausalAuditArtifact] = Field(min_length=1)
    negative_control_passed: Literal[True] = True
    sensitivity_analysis_passed: Literal[True] = True
    directionally_consistent: Literal[True] = True
    abstained: Literal[False] = False
    locked_at: datetime
    research_only: Literal[True] = True

    @model_validator(mode="after")
    def locked_result_contract(self) -> "TargetTrialResult":
        spec_hash = self.spec.content_hash()
        if not self.spec.preregistered:
            raise ValueError("target trial result requires a preregistered specification")
        if len({effect.id for effect in self.effects}) != len(self.effects):
            raise ValueError("target trial effect ids must be unique")
        if len({effect.method for effect in self.effects}) < 2:
            raise ValueError("target trial result requires two estimation methods")
        audits = [self.negative_control_audit, *self.sensitivity_audits]
        if (
            self.input_snapshot.target_trial_spec_id != self.spec.id
            or self.input_snapshot.target_trial_spec_hash != spec_hash
            or self.input_snapshot.cohort_snapshot_id != self.spec.cohort_snapshot_id
            or self.input_snapshot.cohort_snapshot_hash != self.spec.cohort_snapshot_hash
            or self.input_snapshot.eligible_patient_count != self.eligible_patient_count
        ):
            raise ValueError("target-trial input snapshot does not bind the result")
        if self.negative_control_audit.audit_type != "negative_control":
            raise ValueError("target trial requires one negative-control audit")
        if any(audit.audit_type != "sensitivity_analysis" for audit in self.sensitivity_audits):
            raise ValueError("target trial sensitivity audits have the wrong type")
        if len({audit.id for audit in audits}) != len(audits):
            raise ValueError("causal audit artifact ids must be unique")
        for audit in audits:
            if (
                not audit.passed
                or audit.target_trial_spec_id != self.spec.id
                or audit.target_trial_spec_hash != spec_hash
                or audit.data_snapshot_id != self.spec.data_snapshot_id
                or audit.input_snapshot_id != self.input_snapshot.id
                or audit.input_snapshot_hash != self.input_snapshot.content_hash()
                or audit.code_sha256 != self.spec.analysis_code_sha256
                or audit.config_sha256 != self.spec.analysis_config_sha256
            ):
                raise ValueError("causal audit artifact does not bind the locked protocol")
        for effect in self.effects:
            if effect.abstained:
                raise ValueError("locked target trial cannot contain abstained effects")
            if effect.target_trial_id != self.spec.id:
                raise ValueError("effect does not bind the target trial specification")
            if effect.target_trial_result_id != self.id:
                raise ValueError("effect does not bind this target trial result")
            if effect.target_trial_spec_hash != spec_hash:
                raise ValueError("effect target trial specification hash mismatch")
            if effect.outcome not in self.spec.outcomes:
                raise ValueError("effect outcome is not pre-specified")
            if {effect.strategy, effect.comparator} != set(self.spec.strategies):
                raise ValueError("effect contrast does not match pre-specified strategies")
        return self

    def content_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude={"locked_at"})
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class TreatmentOption(StrictModel):
    id: str = Field(default_factory=lambda: new_id("option"))
    regimen_class: str = Field(min_length=1)
    decision_point: DecisionPointType
    effect_estimate_ids: list[str] = Field(default_factory=list)
    effect_estimate_hashes: dict[str, str] = Field(default_factory=dict)
    target_trial_result_ids: list[str] = Field(default_factory=list)
    target_trial_result_hashes: dict[str, str] = Field(default_factory=dict)
    benefits: dict[str, float | str] = Field(default_factory=dict)
    harms: dict[str, float | str] = Field(default_factory=dict)
    pareto_optimal: bool = False
    rank: int | None = Field(default=None, ge=1)
    guideline_backed: bool = True
    abstained: bool = False
    abstention_reasons: list[str] = Field(default_factory=list)
    automatic_order_allowed: Literal[False] = False

    @model_validator(mode="after")
    def evidence_hash_contract(self) -> "TreatmentOption":
        if set(self.effect_estimate_ids) != set(self.effect_estimate_hashes):
            raise ValueError("every effect estimate id requires a content hash")
        if set(self.target_trial_result_ids) != set(self.target_trial_result_hashes):
            raise ValueError("every target-trial result id requires a content hash")
        if self.rank is not None:
            raise ValueError("research treatment options cannot carry a unique rank")
        return self


class PICO(StrictModel):
    population: str = Field(min_length=3)
    intervention: str = Field(min_length=1)
    comparator: str = Field(min_length=1)
    outcomes: list[str] = Field(min_length=1)


class GuidelineSource(StrictModel):
    organization: str = Field(min_length=2)
    title: str = Field(min_length=3)
    version: str = Field(min_length=1)
    url: str = Field(min_length=3)
    source_locator: str = Field(min_length=1)
    effective_from: datetime | None = None
    effective_until: datetime | None = None
    license_status: str = Field(min_length=2)
    content_sha256: str | None = Field(default=None, pattern=r"^[a-fA-F0-9]{64}$")


class EvidenceToDecision(StrictModel):
    """Human-authored GRADE Evidence-to-Decision judgments."""

    problem_priority: str = Field(min_length=3)
    desirable_effects: str = Field(min_length=3)
    undesirable_effects: str = Field(min_length=3)
    values_and_preferences: str = Field(min_length=3)
    balance_of_effects: str = Field(min_length=3)
    resource_use: str = Field(min_length=3)
    equity: str = Field(min_length=3)
    acceptability: str = Field(min_length=3)
    feasibility: str = Field(min_length=3)
    panel_conclusion: str = Field(min_length=3)
    human_judgment: Literal[True] = True


class GuidelineRecommendation(StrictModel):
    id: str = Field(default_factory=lambda: new_id("recommendation"))
    title: str = Field(min_length=3)
    channel: GuidelineChannel
    jurisdiction: Jurisdiction
    guideline_version: str = Field(min_length=1)
    pico: PICO
    eligibility: list[str] = Field(min_length=1)
    exclusions: list[str] = Field(default_factory=list)
    evidence_certainty: EvidenceCertainty
    recommendation_strength: RecommendationStrength
    rationale: str = Field(min_length=3)
    safety_constraints: list[str] = Field(min_length=1)
    hard_safety_criteria: list[str] = Field(min_length=1)
    sources: list[GuidelineSource] = Field(min_length=1)
    evidence_to_decision: EvidenceToDecision
    treatment_options: list[TreatmentOption] = Field(default_factory=list)
    candidate_state_ids: list[str] = Field(default_factory=list)
    candidate_state_hashes: dict[str, str] = Field(default_factory=dict)
    uncertainty: str = Field(min_length=1)
    abstention_conditions: list[str] = Field(min_length=1)
    status: Literal[ReleaseStatus.DRAFT, ReleaseStatus.IN_REVIEW] = (
        ReleaseStatus.DRAFT
    )
    research_only: Literal[True] = True
    automatic_order_allowed: Literal[False] = False
    created_at: datetime = Field(default_factory=utcnow)

    @model_validator(mode="after")
    def channel_boundary(self) -> "GuidelineRecommendation":
        if self.status not in {ReleaseStatus.DRAFT, ReleaseStatus.IN_REVIEW}:
            raise ValueError(
                "recommendations cannot publish themselves; use a reviewed release"
            )
        if self.channel is GuidelineChannel.AI_CANDIDATE:
            if self.recommendation_strength is not RecommendationStrength.RESEARCH_ONLY:
                raise ValueError("AI candidates must use research_only recommendation strength")
            if not self.research_only:
                raise ValueError("AI candidates can never leave research-only mode")
            if set(self.candidate_state_ids) != set(self.candidate_state_hashes):
                raise ValueError("every candidate state id requires a content hash")
        return self

    def content_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude={"created_at"})
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class ExpertReview(StrictModel):
    id: str = Field(default_factory=lambda: new_id("review"))
    object_id: str = Field(min_length=3)
    object_hash: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    reviewer_key: str = Field(min_length=3)
    role: ReviewRole
    decision: ReviewDecision
    rationale: str = Field(min_length=10)
    independent: bool = True
    reviewed_at: datetime = Field(default_factory=utcnow)


class GuidelineRelease(StrictModel):
    id: str = Field(default_factory=lambda: new_id("guideline"))
    name: str = Field(min_length=3)
    version: str = Field(min_length=1)
    channel: GuidelineChannel
    status: ReleaseStatus = ReleaseStatus.DRAFT
    recommendation_ids: list[str] = Field(min_length=1)
    recommendation_hashes: dict[str, str] = Field(min_length=1)
    expert_review_ids: list[str] = Field(default_factory=list)
    context_of_use: str = Field(min_length=10)
    research_only: Literal[True] = True
    automatic_order_allowed: Literal[False] = False
    locked_at: datetime | None = None
    created_at: datetime = Field(default_factory=utcnow)

    @model_validator(mode="after")
    def locked_release_contract(self) -> "GuidelineRelease":
        if set(self.recommendation_ids) != set(self.recommendation_hashes):
            raise ValueError("every recommendation requires a bound content hash")
        if self.status in {ReleaseStatus.LOCKED, ReleaseStatus.RELEASED}:
            if self.locked_at is None:
                raise ValueError("locked/released guidelines require locked_at")
            minimum_reviews = (
                3 if self.channel is GuidelineChannel.BASELINE_APPROVED else 2
            )
            if len(set(self.expert_review_ids)) < minimum_reviews:
                raise ValueError(
                    "locked/released guideline has insufficient independent reviews"
                )
        return self


class ApplicabilityReport(StrictModel):
    recommendation_id: str
    applicable: bool
    matched_criteria: list[str] = Field(default_factory=list)
    failed_criteria: list[str] = Field(default_factory=list)
    missing_critical_features: list[str] = Field(default_factory=list)
    ood: bool = False
    abstained: bool = False
    abstention_reasons: list[str] = Field(default_factory=list)
    independent_review_basis: list[str] = Field(default_factory=list)
    automatic_order_allowed: Literal[False] = False

    @model_validator(mode="after")
    def safety_contract(self) -> "ApplicabilityReport":
        if self.ood or self.missing_critical_features or self.failed_criteria:
            if not self.abstained:
                raise ValueError("failed applicability checks require abstention")
        if self.abstained and not self.abstention_reasons:
            raise ValueError("abstention requires reasons")
        return self


class ModelRelease(StrictModel):
    id: str = Field(default_factory=lambda: new_id("model"))
    name: str = Field(min_length=3)
    version: str = Field(min_length=1)
    context_of_use: str = Field(min_length=10)
    status: ModelStatus = ModelStatus.DEVELOPMENT
    model_sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    code_sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    data_snapshot_ids: list[str] = Field(min_length=1)
    preprocessing_version: str = Field(min_length=1)
    thresholds: dict[str, float] = Field(default_factory=dict)
    forbidden_uses: list[str] = Field(min_length=1)
    validation_report_ids: list[str] = Field(default_factory=list)
    expert_review_ids: list[str] = Field(default_factory=list)
    locked_at: datetime | None = None
    research_only: Literal[True] = True
    created_at: datetime = Field(default_factory=utcnow)

    @model_validator(mode="after")
    def locked_model_contract(self) -> "ModelRelease":
        if self.status in {ModelStatus.LOCKED, ModelStatus.RELEASED}:
            if self.locked_at is None:
                raise ValueError("locked/released model requires locked_at")
            if not self.validation_report_ids:
                raise ValueError("locked/released model requires validation evidence")
            if len(set(self.expert_review_ids)) < 2:
                raise ValueError("locked/released model requires two independent reviews")
        return self

    def content_hash(self) -> str:
        payload = self.model_dump(
            mode="json",
            exclude={"status", "expert_review_ids", "locked_at", "created_at"},
        )
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class InferenceTrace(StrictModel):
    id: str = Field(default_factory=lambda: new_id("trace"))
    model_release_id: str
    context_of_use: str = Field(min_length=10)
    input_sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    output_sha256: str | None = Field(default=None, pattern=r"^[a-fA-F0-9]{64}$")
    ood: bool = False
    uncertainty: float = Field(ge=0)
    abstained: bool = False
    abstention_reasons: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)

    @model_validator(mode="after")
    def refuse_ood(self) -> "InferenceTrace":
        if self.ood and not self.abstained:
            raise ValueError("OOD inference must abstain")
        if self.abstained:
            if not self.abstention_reasons:
                raise ValueError("abstention requires reasons")
            if self.output_sha256 is not None:
                raise ValueError("abstained inference cannot record a scientific output")
        return self


class ResearchStudy(StrictModel):
    id: str = Field(default_factory=lambda: new_id("study"))
    name: str = Field(min_length=3)
    context_of_use: str = Field(min_length=10)
    state: LifecycleStatus = LifecycleStatus.DRAFT
    data_profile_id: str | None = None
    timeline_quality_report_id: str | None = None
    cohort_snapshot_id: str | None = None
    state_model_release_id: str | None = None
    target_trial_ids: list[str] = Field(default_factory=list)
    guideline_release_id: str | None = None
    guideline_release_version: int | None = Field(default=None, ge=1)
    guideline_release_hash: str | None = Field(
        default=None, pattern=r"^[a-fA-F0-9]{64}$"
    )
    failures: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


ALLOWED_LIFECYCLE_TRANSITIONS: dict[LifecycleStatus, set[LifecycleStatus]] = {
    LifecycleStatus.DRAFT: {LifecycleStatus.DATA_REGISTERED},
    LifecycleStatus.DATA_REGISTERED: {LifecycleStatus.TIMELINE_VALIDATED},
    LifecycleStatus.TIMELINE_VALIDATED: {LifecycleStatus.COHORT_LOCKED},
    LifecycleStatus.COHORT_LOCKED: {LifecycleStatus.STATES_DISCOVERED},
    LifecycleStatus.STATES_DISCOVERED: {LifecycleStatus.TARGET_TRIALS_COMPLETED},
    LifecycleStatus.TARGET_TRIALS_COMPLETED: {LifecycleStatus.EXPERT_REVIEW},
    LifecycleStatus.EXPERT_REVIEW: {LifecycleStatus.CANDIDATE_GUIDELINE},
    LifecycleStatus.CANDIDATE_GUIDELINE: {LifecycleStatus.RESEARCH_RELEASE},
    LifecycleStatus.RESEARCH_RELEASE: {LifecycleStatus.RETIRED},
    LifecycleStatus.RETIRED: set(),
}


def transition_study(study: ResearchStudy, target: LifecycleStatus) -> ResearchStudy:
    if target not in ALLOWED_LIFECYCLE_TRANSITIONS[study.state]:
        raise ValueError(f"invalid research lifecycle transition: {study.state} -> {target}")
    if target is LifecycleStatus.DATA_REGISTERED and not study.data_profile_id:
        raise ValueError("data registration requires a fixed aggregate profile")
    if target is LifecycleStatus.TIMELINE_VALIDATED and not study.timeline_quality_report_id:
        raise ValueError("timeline validation requires a passing quality report")
    if target is LifecycleStatus.COHORT_LOCKED and not study.cohort_snapshot_id:
        raise ValueError("cohort lock requires a snapshot")
    if target is LifecycleStatus.STATES_DISCOVERED and not study.state_model_release_id:
        raise ValueError("state discovery requires a locked model release")
    if target is LifecycleStatus.TARGET_TRIALS_COMPLETED and not study.target_trial_ids:
        raise ValueError("target-trial completion requires at least one trial")
    if target in {LifecycleStatus.CANDIDATE_GUIDELINE, LifecycleStatus.RESEARCH_RELEASE}:
        if not all(
            (
                study.guideline_release_id,
                study.guideline_release_version,
                study.guideline_release_hash,
            )
        ):
            raise ValueError("guideline lifecycle requires a fixed release version and hash")
    return study.model_copy(update={"state": target, "updated_at": utcnow()})


__all__ = [
    "ApplicabilityReport",
    "CandidateDiseaseState",
    "CausalEffectEstimate",
    "CausalAuditArtifact",
    "NegativeControlAuditResult",
    "SensitivityAuditResult",
    "ClinicalEvent",
    "CohortSnapshot",
    "DataProfileRegistration",
    "DecisionPoint",
    "DecisionPointType",
    "EvidenceCertainty",
    "EvidenceToDecision",
    "EvidenceSpan",
    "EventType",
    "ExpertReview",
    "GuidelineChannel",
    "GuidelineRecommendation",
    "GuidelineRelease",
    "GuidelineSource",
    "InferenceTrace",
    "Jurisdiction",
    "LifecycleStatus",
    "MVP_DISABLED_LIFECYCLE_STATES",
    "ModelRelease",
    "ModelStatus",
    "PICO",
    "PatientTimeline",
    "RecommendationStrength",
    "ReleaseStatus",
    "ResearchStudy",
    "ReviewDecision",
    "ReviewRole",
    "StateTransition",
    "StatePartitionSnapshot",
    "StateValidationReport",
    "StrictModel",
    "TargetTrialSpec",
    "TargetTrialResult",
    "TargetTrialInputSnapshot",
    "TimePrecision",
    "TimelineQualityReport",
    "TimelineGateThresholds",
    "TimelineValidationManifestRegistration",
    "TreatmentOption",
    "transition_study",
]
