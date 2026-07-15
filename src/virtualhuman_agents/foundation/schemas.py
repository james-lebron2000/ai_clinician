from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from ..models import StrictModel, new_id, utcnow


class BiologicalScale(StrEnum):
    MOLECULE = "molecule"
    CELL = "cell"
    TISSUE = "tissue"
    ORGAN = "organ"
    PERSON = "person"
    POPULATION = "population"


class Modality(StrEnum):
    GENOME = "genome"
    EPIGENOME = "epigenome"
    TRANSCRIPTOME = "transcriptome"
    PROTEOME = "proteome"
    METABOLOME = "metabolome"
    SINGLE_CELL = "single_cell"
    SPATIAL_OMICS = "spatial_omics"
    MICROSCOPY = "microscopy"
    HISTOPATHOLOGY = "histopathology"
    RADIOLOGY = "radiology"
    PHYSIOLOGICAL_WAVEFORM = "physiological_waveform"
    WEARABLE = "wearable"
    LABORATORY = "laboratory"
    MEDICATION = "medication"
    DIAGNOSIS = "diagnosis"
    PROCEDURE = "procedure"
    CLINICAL_TEXT = "clinical_text"
    ENVIRONMENT = "environment"
    PERTURBATION = "perturbation"
    OUTCOME = "outcome"


class AccessTier(StrEnum):
    OPEN = "open"
    REGISTERED = "registered"
    CONTROLLED = "controlled"
    LOCAL_ONLY = "local_only"


class PairingLevel(StrEnum):
    SAME_MEASUREMENT = "same_measurement"
    SAME_SPECIMEN = "same_specimen"
    SAME_SUBJECT_TIME_WINDOW = "same_subject_time_window"
    SAME_SUBJECT = "same_subject"
    COHORT_UNPAIRED = "cohort_unpaired"


class AllowedUse(StrEnum):
    AI_TRAINING = "ai_training"
    AI_VALIDATION = "ai_validation"
    COMMERCIAL_RND = "commercial_rnd"
    REGULATORY_SUBMISSION = "regulatory_submission"
    CROSS_PROJECT_REUSE = "cross_project_reuse"


class ConsentMode(StrEnum):
    DATASET_GOVERNED = "dataset_governed"
    INDIVIDUAL_DIRECTIVE = "individual_directive"


class WithdrawalAction(StrEnum):
    BLOCK_NEW_USE = "block_new_use"
    ASSESS_RETRAINING = "assess_retraining"
    DELETE_WHERE_PERMITTED = "delete_where_permitted"


class DerivedArtifactPolicy(StrEnum):
    EXPORT_ALLOWED = "export_allowed"
    APPROVED_ENVIRONMENT_ONLY = "approved_environment_only"
    CASE_BY_CASE_APPROVAL = "case_by_case_approval"


class SplitName(StrEnum):
    PRETRAIN = "pretrain"
    TRAIN = "train"
    TUNING = "tuning"
    INTERNAL_TEST = "internal_test"
    PROSPECTIVE_TEST = "prospective_test"
    EXTERNAL_SITE_TEST = "external_site_test"


DEVELOPMENT_SPLITS = {SplitName.PRETRAIN, SplitName.TRAIN, SplitName.TUNING}
LOCKED_SPLITS = {
    SplitName.INTERNAL_TEST,
    SplitName.PROSPECTIVE_TEST,
    SplitName.EXTERNAL_SITE_TEST,
}


class DatasetManifest(StrictModel):
    id: str = Field(default_factory=lambda: new_id("dataset"))
    version: str = Field(min_length=1)
    title: str = Field(min_length=3)
    source_name: str = Field(min_length=2)
    source_uri: str = Field(min_length=3)
    data_controller: str = Field(min_length=2)
    license_or_dua: str = Field(min_length=2)
    access_tier: AccessTier
    modalities: set[Modality] = Field(min_length=1)
    biological_scales: set[BiologicalScale] = Field(min_length=1)
    modality_scales: dict[Modality, BiologicalScale] = Field(min_length=1)
    pairing_level: PairingLevel = PairingLevel.COHORT_UNPAIRED
    subject_key_namespace: str = Field(
        min_length=2,
        description="Namespace within which pseudonymous subject keys are comparable",
    )
    subject_count: int = Field(ge=0)
    specimen_count: int = Field(ge=0)
    observation_count: int = Field(gt=0)
    site_ids: list[str] = Field(min_length=1)
    geography: list[str] = Field(min_length=1)
    allowed_uses: set[AllowedUse] = Field(min_length=1)
    consent_mode: ConsentMode = ConsentMode.DATASET_GOVERNED
    permitted_compute_regions: list[str] = Field(min_length=1)
    derived_model_policy: DerivedArtifactPolicy = DerivedArtifactPolicy.CASE_BY_CASE_APPROVAL
    contains_direct_identifiers: bool = False
    deidentification_method: str = Field(min_length=3)
    file_formats: list[str] = Field(min_length=1)
    checksum_manifest_uri: str = Field(min_length=3)
    ontology_versions: dict[str, str] = Field(default_factory=dict)
    acquisition_start: datetime | None = None
    acquisition_end: datetime | None = None
    created_at: datetime = Field(default_factory=utcnow)

    @model_validator(mode="after")
    def validate_manifest(self) -> "DatasetManifest":
        if set(self.modality_scales) != self.modalities:
            raise ValueError("modality_scales must map every and only declared modality")
        if not set(self.modality_scales.values()) <= self.biological_scales:
            raise ValueError("modality_scales references an undeclared biological scale")
        if self.acquisition_start and self.acquisition_end:
            if self.acquisition_start > self.acquisition_end:
                raise ValueError("acquisition_start must not follow acquisition_end")
        if self.pairing_level is not PairingLevel.COHORT_UNPAIRED and len(self.modalities) < 2:
            raise ValueError("paired datasets must declare at least two modalities")
        return self


class ConsentDirective(StrictModel):
    id: str = Field(default_factory=lambda: new_id("consent"))
    subject_key_namespace: str = Field(min_length=2)
    subject_key: str = Field(min_length=2)
    version: str = Field(min_length=1)
    ethics_approval_id: str = Field(min_length=3)
    allowed_uses: set[AllowedUse] = Field(min_length=1)
    allowed_modalities: set[Modality] = Field(min_length=1)
    permitted_compute_regions: list[str] = Field(min_length=1)
    effective_at: datetime
    expires_at: datetime | None = None
    withdrawn_at: datetime | None = None
    withdrawal_action: WithdrawalAction = WithdrawalAction.BLOCK_NEW_USE
    source_sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")

    @model_validator(mode="after")
    def validate_dates(self) -> "ConsentDirective":
        if self.expires_at and self.expires_at < self.effective_at:
            raise ValueError("consent expiry cannot precede its effective time")
        if self.withdrawn_at and self.withdrawn_at < self.effective_at:
            raise ValueError("consent withdrawal cannot precede its effective time")
        return self

    def permits(
        self,
        use: AllowedUse,
        modality: Modality,
        compute_region: str,
        at: datetime | None = None,
    ) -> bool:
        moment = at or utcnow()
        return all(
            [
                moment >= self.effective_at,
                self.expires_at is None or moment <= self.expires_at,
                self.withdrawn_at is None or moment < self.withdrawn_at,
                use in self.allowed_uses,
                modality in self.allowed_modalities,
                compute_region in self.permitted_compute_regions,
            ]
        )


class LineageEntityType(StrEnum):
    PERSON = "person"
    ENCOUNTER = "encounter"
    SPECIMEN = "specimen"
    ALIQUOT = "aliquot"
    DERIVATIVE = "derivative"
    ORGANOID = "organoid"
    REGION = "region"
    CELL = "cell"
    MOLECULE = "molecule"


class LineageRelation(StrEnum):
    COLLECTED_FROM = "collected_from"
    ALIQUOT_OF = "aliquot_of"
    DERIVED_FROM = "derived_from"
    CULTURED_FROM = "cultured_from"
    REGION_OF = "region_of"
    CELL_OF = "cell_of"
    MEASURED_FROM = "measured_from"


class SpecimenLineageEdge(StrictModel):
    id: str = Field(default_factory=lambda: new_id("lineage"))
    subject_key_namespace: str = Field(min_length=2)
    subject_key: str = Field(min_length=2)
    parent_id: str = Field(min_length=2)
    parent_type: LineageEntityType
    child_id: str = Field(min_length=2)
    child_type: LineageEntityType
    relation: LineageRelation
    evidence_sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")


class AlignmentEvidence(StrEnum):
    SOURCE_IDENTIFIER = "source_identifier"
    EXACT_COMEASUREMENT = "exact_comeasurement"
    TEMPORAL_DERIVATION = "temporal_derivation"
    INFERRED = "inferred"


class AlignmentEdge(StrictModel):
    id: str = Field(default_factory=lambda: new_id("alignment"))
    dataset_id: str
    left_asset_id: str
    right_asset_id: str
    pairing_level: PairingLevel
    evidence: AlignmentEvidence
    confidence: float = Field(ge=0, le=1)
    time_delta_seconds: float | None = Field(default=None, ge=0)
    evidence_sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")

    @model_validator(mode="after")
    def validate_alignment(self) -> "AlignmentEdge":
        if self.left_asset_id == self.right_asset_id:
            raise ValueError("alignment requires two distinct assets")
        if self.pairing_level is PairingLevel.COHORT_UNPAIRED:
            raise ValueError("unpaired cohorts cannot create an individual alignment edge")
        if (
            self.evidence is AlignmentEvidence.INFERRED
            and self.pairing_level
            in {PairingLevel.SAME_MEASUREMENT, PairingLevel.SAME_SPECIMEN}
        ):
            raise ValueError("inference cannot upgrade assets to exact measurement/specimen pairs")
        if (
            self.pairing_level is PairingLevel.SAME_SUBJECT_TIME_WINDOW
            and self.time_delta_seconds is None
        ):
            raise ValueError("time-window alignment requires an explicit time delta")
        return self

    def eligible_for_strong_contrastive(self) -> bool:
        return all(
            [
                self.pairing_level
                in {PairingLevel.SAME_MEASUREMENT, PairingLevel.SAME_SPECIMEN},
                self.evidence
                in {
                    AlignmentEvidence.SOURCE_IDENTIFIER,
                    AlignmentEvidence.EXACT_COMEASUREMENT,
                },
                self.confidence >= 0.95,
            ]
        )


class SampleIndexRecord(StrictModel):
    id: str = Field(default_factory=lambda: new_id("asset"))
    dataset_id: str
    subject_key: str | None = None
    specimen_key: str | None = None
    pairing_key: str | None = None
    lineage_node_id: str | None = None
    group_key: str = Field(
        min_length=2,
        description="Smallest independent unit that must remain in one split",
    )
    modality: Modality
    biological_scale: BiologicalScale
    site_id: str
    acquisition_time: datetime | None = None
    split: SplitName
    uri: str = Field(min_length=3)
    sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    direct_identifier_present: bool = False
    quality_status: Literal["usable", "quarantined", "rejected"] = "usable"


class RegistryFinding(StrictModel):
    code: str
    severity: Literal["warning", "critical"]
    message: str
    asset_ids: list[str] = Field(default_factory=list)


class RegistryAuditReport(StrictModel):
    passed: bool
    findings: list[RegistryFinding] = Field(default_factory=list)
    dataset_count: int = Field(ge=0)
    asset_count: int = Field(ge=0)
    subject_count: int = Field(ge=0)
    modality_counts: dict[str, int] = Field(default_factory=dict)
    split_counts: dict[str, int] = Field(default_factory=dict)


class HumanFMConfig(StrictModel):
    name: str = "HumanFM-v0"
    version: str = "0.1.0"
    hidden_dim: int = Field(default=256, ge=32, le=4096)
    num_attention_heads: int = Field(default=8, ge=1, le=64)
    modality_encoder_layers: int = Field(default=2, ge=1, le=48)
    cross_scale_layers: int = Field(default=2, ge=1, le=48)
    scale_latent_tokens: int = Field(default=4, ge=1, le=64)
    maximum_tokens_per_modality: int = Field(default=512, ge=1, le=65536)
    dropout: float = Field(default=0.1, ge=0, lt=1)
    modality_input_dims: dict[Modality, int] = Field(min_length=1)
    modality_scales: dict[Modality, BiologicalScale] = Field(min_length=1)
    active_scales: list[BiologicalScale] = Field(
        default_factory=lambda: list(BiologicalScale)
    )
    intervention_dim: int = Field(default=64, ge=1)
    state_dim: int = Field(default=256, ge=16)
    context_of_use: str = Field(min_length=10)

    @model_validator(mode="after")
    def validate_config(self) -> "HumanFMConfig":
        if self.hidden_dim % self.num_attention_heads:
            raise ValueError("hidden_dim must be divisible by num_attention_heads")
        if set(self.modality_input_dims) != set(self.modality_scales):
            raise ValueError("input dimensions and scale routes must cover identical modalities")
        if any(value <= 0 for value in self.modality_input_dims.values()):
            raise ValueError("all modality input dimensions must be positive")
        if not set(self.modality_scales.values()) <= set(self.active_scales):
            raise ValueError("a modality is routed to an inactive scale")
        if len(self.active_scales) != len(set(self.active_scales)):
            raise ValueError("active_scales must be unique")
        return self


class TrainingObjective(StrEnum):
    MASKED_RECONSTRUCTION = "masked_reconstruction"
    SELF_DISTILLATION = "self_distillation"
    CONTRASTIVE_ALIGNMENT = "contrastive_alignment"
    TEMPORAL_PREDICTION = "temporal_prediction"
    PERTURBATION_RESPONSE = "perturbation_response"
    CROSS_SCALE_CONSISTENCY = "cross_scale_consistency"
    TASK_SUPERVISION = "task_supervision"


class TrainingStageSpec(StrictModel):
    index: int = Field(ge=1)
    name: str = Field(min_length=3)
    objectives: list[TrainingObjective] = Field(min_length=1)
    required_modalities: set[Modality] = Field(default_factory=set)
    permitted_pairing_levels: set[PairingLevel] = Field(default_factory=set)
    trainable_components: list[str] = Field(min_length=1)
    frozen_components: list[str] = Field(default_factory=list)
    locked: bool = False
    exit_criteria: dict[str, float] = Field(min_length=1)


class MetricDirection(StrEnum):
    GREATER_OR_EQUAL = "greater_or_equal"
    LESS_OR_EQUAL = "less_or_equal"


class EvaluationGate(StrictModel):
    metric: str
    threshold: float
    direction: MetricDirection
    critical: bool = True
    subgroup: str = "overall"


class BenchmarkResult(StrictModel):
    metric: str
    value: float
    subgroup: str = "overall"
    dataset_id: str
    model_version: str
    baseline_value: float | None = None


class FoundationReadinessReport(StrictModel):
    model_name: str
    model_version: str
    context_of_use: str
    registry_audit_passed: bool
    passed: bool
    failed_gates: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=utcnow)


class DatasetSnapshot(StrictModel):
    id: str = Field(default_factory=lambda: new_id("snapshot"))
    manifest_ids: list[str] = Field(min_length=1)
    asset_ids: list[str] = Field(min_length=1)
    split_manifest_sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    consent_decision_sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    preprocessing_sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    ontology_versions: dict[str, str] = Field(min_length=1)
    created_by_user_id: str = Field(min_length=2)
    snapshot_sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    derived_model_policy_decision_sha256: str = Field(
        default="0" * 64, pattern=r"^[a-fA-F0-9]{64}$"
    )
    created_at: datetime = Field(default_factory=utcnow)


class ModelRelease(StrictModel):
    id: str = Field(default_factory=lambda: new_id("modelrelease"))
    model_name: str = Field(min_length=2)
    model_version: str = Field(min_length=1)
    dataset_snapshot_ids: list[str] = Field(min_length=1)
    config_sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    code_sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    container_digest: str = Field(pattern=r"^sha256:[a-fA-F0-9]{64}$")
    weights_sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    preprocessing_sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    context_of_use: str = Field(min_length=10)
    forbidden_uses: list[str] = Field(min_length=1)
    ood_detector_release_id: str = Field(min_length=3)
    status: Literal["development", "locked", "released", "retired"] = "development"
    created_at: datetime = Field(default_factory=utcnow)


class ApplicabilityStatus(StrEnum):
    IN_DOMAIN = "in_domain"
    OOD_REJECTED = "ood_rejected"
    UNDETERMINED = "undetermined"


class InferenceTrace(StrictModel):
    id: str = Field(default_factory=lambda: new_id("inference"))
    model_release_id: str
    input_sha256: list[str] = Field(min_length=1)
    applicability_status: ApplicabilityStatus
    ood_score: float
    ood_threshold: float
    uncertainty: float = Field(ge=0)
    output_uri: str | None = None
    output_sha256: str | None = Field(default=None, pattern=r"^[a-fA-F0-9]{64}$")
    rejected_reason: str | None = None
    created_at: datetime = Field(default_factory=utcnow)

    @model_validator(mode="after")
    def validate_abstention(self) -> "InferenceTrace":
        if self.applicability_status is ApplicabilityStatus.OOD_REJECTED:
            if not self.rejected_reason:
                raise ValueError("OOD rejection requires a reason")
            if self.output_uri or self.output_sha256:
                raise ValueError("rejected inputs cannot carry a scientific model output")
        elif self.applicability_status is ApplicabilityStatus.IN_DOMAIN:
            if not self.output_uri or not self.output_sha256:
                raise ValueError("in-domain inference requires a checksum-addressed output")
        return self
