from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ProjectState(StrEnum):
    DRAFT = "draft"
    READY = "ready"
    EVIDENCE_REVIEW = "evidence_review"
    DESIGNING = "designing"
    WORK_ORDER_ISSUED = "work_order_issued"
    AWAITING_RESULTS = "awaiting_results"
    ANALYZING = "analyzing"
    DECIDING = "deciding"
    PAUSED = "paused"
    COMPLETED = "completed"
    REJECTED = "rejected"


class DecisionAction(StrEnum):
    ISSUE_WORK_ORDER = "issue_work_order"
    NEXT_ROUND = "next_round"
    COMPLETE = "complete"
    PAUSE = "pause"
    REJECT = "reject"


class FactorKind(StrEnum):
    CONTINUOUS = "continuous"
    INTEGER = "integer"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"


class StudyPhase(StrEnum):
    FEASIBILITY = "feasibility"
    DEVELOPMENT = "development"
    VALIDATION = "validation"
    LOCKED_VALIDATION = "locked_validation"


class RegulatoryPathway(StrEnum):
    RESEARCH_ONLY = "research_only"
    PRODUCT_SPECIFIC_SUBMISSION = "product_specific_submission"
    DDT_QUALIFICATION = "ddt_qualification"


class GLPApplicability(StrEnum):
    NON_GLP_EXPLORATORY = "non_glp_exploratory"
    GLP = "glp"
    HYBRID = "hybrid"


class AIRegulatoryImpact(StrEnum):
    OPERATIONAL_ONLY = "operational_only"
    SUPPORTS_INTERPRETATION = "supports_interpretation"
    SUPPORTS_REGULATORY_DECISION = "supports_regulatory_decision"


class ApprovalRole(StrEnum):
    SPONSOR = "sponsor"
    STUDY_DIRECTOR = "study_director"
    QUALITY_ASSURANCE = "quality_assurance"
    REGULATORY = "regulatory"
    RESEARCH_LEAD = "research_lead"


class FactorSpec(StrictModel):
    kind: FactorKind
    unit: str | None = None
    minimum: float | None = None
    maximum: float | None = None
    choices: list[str | bool] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_domain(self) -> "FactorSpec":
        if self.kind in {FactorKind.CONTINUOUS, FactorKind.INTEGER}:
            if self.minimum is None or self.maximum is None:
                raise ValueError("numeric factors require minimum and maximum")
            if self.minimum >= self.maximum:
                raise ValueError("factor minimum must be lower than maximum")
            if self.choices:
                raise ValueError("numeric factors cannot define choices")
        elif self.kind is FactorKind.CATEGORICAL:
            if len(self.choices) < 2:
                raise ValueError("categorical factors require at least two choices")
        elif self.kind is FactorKind.BOOLEAN:
            object.__setattr__(self, "choices", [False, True])
        return self


class Budget(StrictModel):
    currency: str = "CNY"
    maximum_cost: Annotated[float, Field(gt=0)]
    maximum_rounds: Annotated[int, Field(ge=1, le=4)] = 4
    initial_condition_limit: Annotated[int, Field(ge=4, le=24)] = 24
    subsequent_condition_limit: Annotated[int, Field(ge=4, le=16)] = 16


class Governance(StrictModel):
    ethics_approval_id: str = Field(min_length=3)
    consent_scope: list[str] = Field(min_length=1)
    sample_reuse_allowed: bool
    cross_project_training_allowed: bool
    data_residency: str = "CN"
    cross_border_transfer_allowed: bool = False
    approved_sites: list[str] = Field(min_length=1)


class CROStudyGovernance(StrictModel):
    sponsor_id: str = Field(min_length=2)
    cro_id: str = Field(min_length=2)
    study_id: str = Field(min_length=3)
    protocol_id: str = Field(min_length=3)
    protocol_version: str = Field(min_length=1)
    sap_id: str = Field(min_length=3, description="Pre-specified statistical analysis plan")
    study_phase: StudyPhase = StudyPhase.DEVELOPMENT
    study_director_user_id: str = Field(min_length=2)
    qau_user_id: str = Field(min_length=2)
    sponsor_representative_user_id: str = Field(min_length=2)
    blinded: bool = True
    randomization_required: bool = True
    record_retention_years: int = Field(default=10, ge=2, le=30)
    vendor_qualification_ids: list[str] = Field(min_length=1)


class RegulatoryProfile(StrictModel):
    pathway: RegulatoryPathway
    glp_applicability: GLPApplicability
    part11_required: bool = True
    fda_application_type: Literal["IND", "NDA", "BLA", "DDT", "NONE"] = "NONE"
    fda_application_number: str | None = None
    ectd_version: Literal["3.2.2", "4.0"] = "4.0"
    study_data_standard: Literal["SEND", "CUSTOM_WITH_REVIEWERS_GUIDE", "NOT_APPLICABLE"] = (
        "CUSTOM_WITH_REVIEWERS_GUIDE"
    )
    ai_regulatory_impact: AIRegulatoryImpact = AIRegulatoryImpact.SUPPORTS_INTERPRETATION
    question_of_interest: str = Field(min_length=5)
    model_role: str = Field(min_length=5)
    consequence_of_error: str = Field(min_length=5)
    early_fda_engagement_planned: bool = True
    regulatory_claims: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_pathway(self) -> "RegulatoryProfile":
        if self.pathway is RegulatoryPathway.RESEARCH_ONLY:
            if self.fda_application_type != "NONE":
                raise ValueError("research-only projects cannot declare an FDA application type")
        elif self.pathway is RegulatoryPathway.DDT_QUALIFICATION:
            if self.fda_application_type != "DDT":
                raise ValueError("DDT qualification projects must use application type DDT")
        elif self.fda_application_type not in {"IND", "NDA", "BLA"}:
            raise ValueError("product-specific submissions require IND, NDA, or BLA")
        return self


class QualityThresholds(StrictModel):
    maximum_within_plate_cv: float = Field(default=0.15, gt=0, lt=1)
    maximum_between_batch_cv: float = Field(default=0.20, gt=0, lt=1)
    minimum_z_prime: float = Field(default=0.50, ge=-1, le=1)
    minimum_interlab_icc: float = Field(default=0.75, ge=0, le=1)
    minimum_fidelity_score: float = Field(default=0.80, ge=0, le=1)
    minimum_driver_concordance: float = Field(default=0.90, ge=0, le=1)
    minimum_ai_dice: float = Field(default=0.90, ge=0, le=1)
    minimum_site_ai_dice: float = Field(default=0.85, ge=0, le=1)
    information_gain_stop_threshold: float = Field(default=0.05, gt=0, lt=1)
    required_consecutive_passes: Annotated[int, Field(ge=1, le=3)] = 2


class ResearchGoal(StrictModel):
    id: str = Field(default_factory=lambda: new_id("goal"))
    title: str = Field(min_length=3)
    disease: str = Field(min_length=2)
    therapeutic_context: str
    decision_context: str = Field(min_length=5)
    context_of_use: str = Field(min_length=5)
    target_phenotypes: list[str] = Field(min_length=1)
    primary_readout: str = Field(min_length=2)
    factor_space: dict[str, FactorSpec] = Field(min_length=1)
    budget: Budget
    governance: Governance
    cro: CROStudyGovernance
    regulatory: RegulatoryProfile
    quality: QualityThresholds = Field(default_factory=QualityThresholds)
    prohibited_actions: list[str] = Field(
        default_factory=lambda: [
            "gene_editing",
            "pathogen_work",
            "unapproved_human_sample",
            "cross_border_transfer",
            "out_of_consent_use",
        ]
    )
    created_at: datetime = Field(default_factory=utcnow)


Scalar = str | float | int | bool


class ExperimentalCondition(StrictModel):
    id: str = Field(default_factory=lambda: new_id("condition"))
    factors: dict[str, Scalar]
    estimated_cost: Annotated[float, Field(ge=0)] = 0.0
    rationale: str = ""

    def fingerprint(self) -> str:
        payload = json.dumps(self.factors, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode()).hexdigest()


class ControlSpec(StrictModel):
    control_type: Literal["negative", "positive", "mechanism"]
    name: str = Field(min_length=1)
    approved_material_id: str = Field(min_length=1)


class StopRules(StrictModel):
    maximum_rounds: Annotated[int, Field(ge=1, le=4)] = 4
    information_gain_below: float = Field(default=0.05, gt=0, lt=1)
    consecutive_quality_passes: Annotated[int, Field(ge=1, le=3)] = 2
    budget_exhaustion: bool = True
    pause_on_qc_failure: bool = True


class ExperimentPlan(StrictModel):
    id: str = Field(default_factory=lambda: new_id("plan"))
    project_id: str
    round_index: Annotated[int, Field(ge=1, le=4)]
    hypothesis_ids: list[str] = Field(min_length=1)
    conditions: list[ExperimentalCondition] = Field(min_length=1)
    controls: list[ControlSpec] = Field(min_length=3)
    biological_replicates: Annotated[int, Field(ge=3)] = 3
    technical_replicates: Annotated[int, Field(ge=2)] = 2
    sop_version: str = Field(min_length=1)
    protocol_id: str = Field(min_length=3)
    protocol_version: str = Field(min_length=1)
    sap_id: str = Field(min_length=3)
    randomization_seed: int = Field(ge=0)
    blinded: bool = True
    design_locked: bool = False
    acceptance_criteria: dict[str, float] = Field(min_length=1)
    stop_rules: StopRules = Field(default_factory=StopRules)
    estimated_cost: Annotated[float, Field(ge=0)]
    created_by: str = "experiment_design_agent"
    created_at: datetime = Field(default_factory=utcnow)

    @model_validator(mode="after")
    def check_unique_conditions(self) -> "ExperimentPlan":
        fingerprints = [condition.fingerprint() for condition in self.conditions]
        if len(fingerprints) != len(set(fingerprints)):
            raise ValueError("experiment conditions must be unique")
        return self


class ArtifactRef(StrictModel):
    uri: str
    sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    media_type: str = "application/octet-stream"


class Observation(StrictModel):
    condition_id: str
    replicate_id: str
    biological_replicate: str
    technical_replicate: int = Field(ge=1)
    value: float
    metrics: dict[str, float] = Field(default_factory=dict)
    control_type: Literal["negative", "positive", "mechanism"] | None = None
    valid: bool = True
    invalid_reason: str | None = None


class SubmittedQC(StrictModel):
    mycoplasma_negative: bool
    sample_identity_match: bool
    driver_concordance: float = Field(ge=0, le=1)
    fidelity_score: float = Field(ge=0, le=1)
    between_batch_cv: float | None = Field(default=None, ge=0)
    interlab_icc: float | None = Field(default=None, ge=0, le=1)
    ai_segmentation_dice: float | None = Field(default=None, ge=0, le=1)
    out_of_distribution_input_rejected: bool = False
    notes: list[str] = Field(default_factory=list)


class ProtocolDeviation(StrictModel):
    id: str = Field(default_factory=lambda: new_id("deviation"))
    category: Literal["minor", "major", "critical"]
    description: str = Field(min_length=5)
    impact_assessment: str = Field(min_length=5)
    disposition: Literal["open", "accepted", "corrected", "invalidates_result"] = "open"
    detected_by_user_id: str = Field(min_length=2)


class ResultBundle(StrictModel):
    id: str = Field(default_factory=lambda: new_id("result"))
    project_id: str
    plan_id: str
    site_id: str
    batch_id: str
    protocol_id: str = Field(min_length=3)
    protocol_version: str = Field(min_length=1)
    analyst_user_id: str = Field(min_length=2)
    instrument_ids: list[str] = Field(min_length=1)
    reagent_lot_ids: list[str] = Field(min_length=1)
    observations: list[Observation] = Field(min_length=1)
    submitted_qc: SubmittedQC
    raw_artifacts: list[ArtifactRef] = Field(default_factory=list)
    deviations: list[ProtocolDeviation] = Field(default_factory=list)
    failed_reason: str | None = None
    received_at: datetime = Field(default_factory=utcnow)


class AnalysisReport(StrictModel):
    id: str = Field(default_factory=lambda: new_id("analysis"))
    project_id: str
    plan_id: str
    result_id: str
    within_plate_cv: float | None
    z_prime: float | None
    between_batch_cv: float | None
    interlab_icc: float | None
    ai_segmentation_dice: float | None
    driver_concordance: float
    fidelity_score: float
    quality_passed: bool
    condition_scores: dict[str, float]
    invalid_observation_rate: float = Field(ge=0, le=1)
    anomalies: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)


class EvidenceRecord(StrictModel):
    id: str = Field(default_factory=lambda: new_id("evidence"))
    project_id: str
    claim: str = Field(min_length=5)
    title: str = Field(min_length=3)
    source_url: HttpUrl
    source_type: Literal["peer_reviewed", "regulatory", "patent", "internal", "protocol"]
    supports: list[str] = Field(default_factory=list)
    content_hash: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    retrieved_at: datetime = Field(default_factory=utcnow)


class Hypothesis(StrictModel):
    id: str = Field(default_factory=lambda: new_id("hypothesis"))
    project_id: str
    statement: str = Field(min_length=5)
    mechanism: str = Field(min_length=3)
    evidence_ids: list[str]
    testability_score: float = Field(ge=0, le=1)
    novelty_score: float = Field(ge=0, le=1)
    risk_score: float = Field(ge=0, le=1)
    proposed_factors: list[str]


class PolicyFinding(StrictModel):
    code: str
    severity: Literal["warning", "critical"]
    message: str


class PolicyReport(StrictModel):
    passed: bool
    findings: list[PolicyFinding] = Field(default_factory=list)


class AgentDecision(StrictModel):
    id: str = Field(default_factory=lambda: new_id("decision"))
    project_id: str
    action: DecisionAction
    rationale: str
    evidence_ids: list[str] = Field(default_factory=list)
    expected_information_gain: float | None = Field(default=None, ge=0, le=1)
    estimated_cost: float = Field(default=0, ge=0)
    uncertainty: float = Field(default=0, ge=0, le=1)
    requires_human_approval: bool = False
    approval_reasons: list[str] = Field(default_factory=list)
    next_plan_id: str | None = None
    created_at: datetime = Field(default_factory=utcnow)


class ElectronicApproval(StrictModel):
    id: str = Field(default_factory=lambda: new_id("approval"))
    project_id: str
    object_type: Literal[
        "goal", "plan", "result", "analysis", "controlled_artifact", "regulatory_package"
    ]
    object_id: str
    object_sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    signer_user_id: str = Field(min_length=2)
    signer_role: ApprovalRole
    meaning: Literal["authored", "reviewed", "approved", "qa_released"]
    authentication_event_id: str = Field(
        min_length=8,
        description="Reference to identity-provider re-authentication; never a raw credential",
    )
    signed_at: datetime = Field(default_factory=utcnow)


class PlateAssignment(StrictModel):
    well: str
    item_type: Literal["condition", "control"]
    item_id: str
    biological_replicate: int
    technical_replicate: int


class WorkOrder(StrictModel):
    id: str = Field(default_factory=lambda: new_id("workorder"))
    project_id: str
    plan_id: str
    site_id: str
    sop_version: str
    protocol_id: str
    protocol_version: str
    design_locked: bool
    assignments: list[PlateAssignment]
    instructions: list[str]
    status: Literal["issued", "acknowledged", "completed", "cancelled"] = "issued"
    issued_at: datetime = Field(default_factory=utcnow)


class ControlledArtifact(StrictModel):
    id: str = Field(default_factory=lambda: new_id("artifact"))
    project_id: str
    artifact_type: Literal[
        "protocol",
        "sop",
        "sap",
        "data_management_plan",
        "assay_validation",
        "model_card",
        "training_data_manifest",
        "software_validation",
        "qa_statement",
        "external_replication_report",
    ]
    title: str = Field(min_length=3)
    version: str = Field(min_length=1)
    uri: str
    sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    status: Literal["draft", "effective", "superseded"] = "effective"
    owner_user_id: str = Field(min_length=2)
    created_at: datetime = Field(default_factory=utcnow)


class ProjectSnapshot(StrictModel):
    project_id: str
    goal: ResearchGoal
    state: ProjectState
    spent_cost: float = Field(ge=0)
    consecutive_quality_passes: int = Field(ge=0)
    active_plan_id: str | None = None
    active_work_order_id: str | None = None
    last_decision: AgentDecision | None = None
    created_at: datetime
    updated_at: datetime


class AuditEvent(StrictModel):
    id: str = Field(default_factory=lambda: new_id("audit"))
    project_id: str
    actor: str
    action: str
    payload: dict[str, Any] = Field(default_factory=dict)
    previous_hash: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    event_hash: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    created_at: datetime = Field(default_factory=utcnow)


class ReadinessCheck(StrictModel):
    code: str
    passed: bool
    critical: bool = True
    evidence: str


class RegulatoryReadinessReport(StrictModel):
    id: str = Field(default_factory=lambda: new_id("readiness"))
    project_id: str
    pathway: RegulatoryPathway
    checks: list[ReadinessCheck]
    ready_for_qa_release: bool
    limitations: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=utcnow)


class RegulatoryPackage(StrictModel):
    id: str = Field(default_factory=lambda: new_id("regpkg"))
    project_id: str
    readiness_report_id: str
    package_format: Literal["ECTD_MODULE4_STAGING", "DDT_QUALIFICATION_STAGING"]
    artifact_uri: str
    sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    status: Literal["draft", "ready_for_qa", "qa_released"] = "draft"
    generated_at: datetime = Field(default_factory=utcnow)


class CycleOutcome(StrictModel):
    project: ProjectSnapshot
    decision: AgentDecision
    work_order: WorkOrder | None = None
    analysis: AnalysisReport | None = None
